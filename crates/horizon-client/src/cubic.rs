//! A cubic level's terrain and saves (docs/cubic-chunks.md, "v1
//! architecture"). The level's chunks hold its window of cubes
//! (`LevelShape::cubic_window`); this module makes those columns from cubes
//! and saves them back as cubes.
//!
//! - [`CubicLevel`]: the level's cube store (`region3d/`, with writes the
//!   loads see before they reach the disk), its generator for the cubes
//!   outside vanilla's range, and the bookkeeping that keeps unchanged cubes
//!   from being written again.
//! - [`CubicTerrain`]: a `ChunkGenerator` around the vanilla one. The
//!   vanilla generator makes each column's -64..320 once, whole (surface,
//!   carvers, decoration, structures), which is saved as cubes; the window's
//!   columns are then put together from cubes.
//! - `SaveDir` sends a cubic level's columns here (`SaveDir::cubic`), and
//!   its entities too.
//!
//! What a cube record holds (a vanilla chunk record of one section):
//! blocks and biomes, block entities, the scheduled block and fluid ticks
//! at positions in the cube (vanilla's `block_ticks`/`fluid_ticks`, with
//! absolute positions), and under `horizon:entities` the entities whose
//! `Pos` is in the cube (vanilla entity NBT). The cube is the unit of
//! entities, so a column's entities outside the window wait in their cubes'
//! records and nothing keeps them in memory. Entity saves and block saves
//! reach the same record, so each is a read-modify-write under one lock
//! (`CubicLevel::edit`): a block save keeps the record's entities, an entity
//! save keeps its blocks.
//!
//! Points of interest are not saved for a cubic level: Horizon keeps no POI
//! state besides what its blocks give (`PoiSections` is rebuilt from a
//! section's blocks, and the `poi/` files are written for vanilla's sake but
//! never read), so a loaded cube has its POIs again.

use crate::biome::Biomes;
use crate::chunk::{Chunk, ColumnLayout, LevelShape, Section, block_strategy};
use crate::cubic_gen::{
    CubicGeneration, CubicKind, Generation, PARTIAL, POPULATED, PopRegion, Population,
    needs_population, root_flags, set_root_flags,
};
use crate::save::{ChunkExtras, Persistent, SaveTables, decode_with, encode_with};
use crate::streaming::{ChunkGenerator, Proto};
use horizon_core::paletted::PalettedContainer;
use horizon_core::position::SectionPos;
use horizon_core::ticks::SavedTick;
use horizon_world::cubic::store::CubeStore;
use horizon_world::cubic::{MAX_CUBE_Y, MIN_CUBE_Y};
use horizon_world::nbt::{NamedTag, Tag};
use horizon_world::registry::BlockRegistry;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Condvar, Mutex};

/// The cube Ys vanilla's Overworld generator makes (-64..320), in a Hybrid
/// world.
pub use crate::cubic_gen::VANILLA_CUBES;
/// The cube whose record says the column's vanilla range was generated and
/// holds the column's own data (structure starts and references, the
/// surface heights).
pub const MARKER_CUBE: i32 = -4;
/// A shard's window, shared with the shard's terrain, whose worker threads
/// load columns into it (`CubicTerrain::window`). The game thread moves it
/// with the shard's world.
pub type ShardWindow = Arc<Mutex<ColumnLayout>>;

/// A new shard's window cell, holding `shape`'s column layout.
pub fn shard_window(shape: LevelShape) -> ShardWindow {
    Arc::new(Mutex::new(shape.column()))
}

/// What makes a shard's streaming: the level's terrain around the given
/// window cell (`CubicLevel::new_shard_streaming`).
pub type ShardMaker = Box<dyn Fn(ShardWindow) -> crate::streaming::Streaming + Send + Sync>;

/// The window a new cubic world opens with: vanilla's surface in its
/// middle (the default vertical distance's height; the first player
/// resizes and recentres it, `follow_cubic_window`).
pub const FIRST_WINDOW_MIN_Y: i32 = -80;
/// The window is `2V + 3` cubes tall around the player's cube (V is the
/// vertical distance in cubes, `Graphics.cubic_vertical`): V cubes each way
/// and one of hysteresis.
pub const fn window_cubes(vertical: i32) -> i32 {
    2 * vertical + 3
}
/// The most cubes a window may hold, whatever the players' ranges add up to.
pub const MAX_WINDOW_CUBES: i32 = 99;
/// A window is centred again when a player's cube is this many cubes (or
/// more) from its centre cube.
pub const RECENTRE_CUBES: i32 = 2;
/// The layers above and below the window the prefetch ring keeps ready in
/// memory (`cubic_prefetch`); a window moves at most this far at a time.
pub const RING: i32 = 2;
/// The vertical distance a guest has until its client says otherwise.
pub const DEFAULT_VERTICAL: u8 = crate::graphics::CUBIC_VERTICAL_DEFAULT as u8;
/// A guest's vertical distance, kept to what the setting allows.
pub fn clamp_vertical(vertical: u8) -> u8 {
    vertical.clamp(
        crate::graphics::CUBIC_VERTICAL_MIN as u8,
        crate::graphics::CUBIC_VERTICAL_MAX as u8,
    )
}
/// The key in a column's marker cube with its vanilla surface heights.
const SURFACE_KEY: &str = "horizon:cubic_surface";
/// The key in a cube record with the entities whose position is in the cube.
const ENTITIES_KEY: &str = "horizon:entities";

/// A cube as saved, for putting into a column that is loaded (a window
/// moving in place) or for building one.
#[derive(Clone)]
pub struct LoadedCube {
    pub pos: SectionPos,
    /// Blocks and biomes.
    pub section: Arc<Section>,
    /// Block entities at absolute positions in the cube.
    pub block_entities: Vec<((i32, i32, i32), crate::containers::BlockEntity)>,
    pub block_names: Vec<((i32, i32, i32), String)>,
    /// The entities saved in the cube (vanilla entity NBT, absolute `Pos`).
    pub entities: Vec<Tag>,
    /// Scheduled ticks at positions in the cube, as saved (`delay` relative
    /// to the save time).
    pub block_ticks: Vec<SavedTick>,
    pub fluid_ticks: Vec<SavedTick>,
    /// The marker cube's column data (empty elsewhere).
    pub structure_refs: Vec<(Arc<str>, (i32, i32))>,
    pub structure_starts: Vec<Arc<horizon_world::worldgen::structure::start::StructureStart>>,
    /// The marker cube's vanilla surface heights (`z << 4 | x`).
    pub surface: Option<Vec<i32>>,
    /// The cube had no record: the generator made it.
    pub generated: bool,
}

/// The cubes (first, last) a player in cube `cube` with vertical distance
/// `vertical` wants loaded: `[cube - V - 1, cube + V + 1]`.
pub fn wanted_cubes(cube: i32, vertical: u8) -> (i32, i32) {
    let v = i32::from(clamp_vertical(vertical));
    (cube - v - 1, cube + v + 1)
}

/// The union of players' cube ranges, capped at `MAX_WINDOW_CUBES` (around
/// the middle of the union when it is taller) and kept in the cubic range;
/// `None` for no players.
pub fn union_cubes(ranges: impl IntoIterator<Item = (i32, i32)>) -> Option<(i32, i32)> {
    // Players whose ranges do not meet are in shards of their own
    // (`cubic_shards`), so a union is of ranges that overlap; one taller than
    // the cap is a chain of them, and the outermost players stay on its edge.
    let (lo, hi) = ranges
        .into_iter()
        .reduce(|(a, b), (c, d)| (a.min(c), b.max(d)))?;
    let (mut lo, mut hi) = (lo, hi);
    if hi - lo + 1 > MAX_WINDOW_CUBES {
        let middle = lo + (hi - lo) / 2;
        lo = middle - MAX_WINDOW_CUBES / 2;
        hi = lo + MAX_WINDOW_CUBES - 1;
    }
    let (min, max) = (
        horizon_world::cubic::MIN_CUBE_Y,
        horizon_world::cubic::MAX_CUBE_Y,
    );
    let shift = (min - lo).max(0) - (hi - max).max(0);
    Some(((lo + shift).max(min), (hi + shift).min(max)))
}

/// The cubes (first, last) a window holds.
pub fn window_cube_range(window: LevelShape) -> (i32, i32) {
    (window.min_y >> 4, (window.max_y() >> 4) - 1)
}

/// The window of cubes `first..=last`.
pub fn window_of_cubes(level: LevelShape, (first, last): (i32, i32)) -> LevelShape {
    LevelShape::cubic_window(level, first * 16, (last - first + 1) * 16)
}

/// Whether the level's `window` should move to the cubes `wanted` (the
/// union of the players' ranges): its height changed (a vertical distance
/// did), or its centre cube is `RECENTRE_CUBES` or more from the wanted
/// one's. For one player the wanted centre is the player's cube.
pub fn needs_new_window(window: LevelShape, wanted: (i32, i32)) -> bool {
    if !window.cubic {
        return false;
    }
    let have = window_cube_range(window);
    let centre = |(lo, hi): (i32, i32)| (lo + hi).div_euclid(2);
    have.1 - have.0 != wanted.1 - wanted.0
        || (centre(have) - centre(wanted)).abs() >= RECENTRE_CUBES
}

/// The window a slide moves to on the way to `wanted`: each end moves at
/// most `RING` layers (what the ring has ready), unless the two do not
/// overlap, which is a jump to `wanted` whole.
pub fn next_window(have: (i32, i32), wanted: (i32, i32)) -> (i32, i32) {
    if wanted.1 < have.0 || wanted.0 > have.1 {
        return wanted;
    }
    let step = |from: i32, to: i32| from + (to - from).clamp(-RING, RING);
    let (lo, hi) = (step(have.0, wanted.0), step(have.1, wanted.1));
    (lo.min(hi), hi.max(lo))
}

impl LoadedCube {
    /// A cube of `section` with nothing else in it.
    pub fn empty(pos: SectionPos, section: Arc<Section>) -> Self {
        Self {
            pos,
            section,
            block_entities: Vec::new(),
            block_names: Vec::new(),
            entities: Vec::new(),
            block_ticks: Vec::new(),
            fluid_ticks: Vec::new(),
            structure_refs: Vec::new(),
            structure_starts: Vec::new(),
            surface: None,
            generated: false,
        }
    }
}

/// The cubes `next` holds that `have` does not, nearest the window's middle
/// last (they are layers on its two sides).
pub fn entering_cubes(have: (i32, i32), next: (i32, i32)) -> Vec<i32> {
    (next.0..=next.1)
        .filter(|cy| !(have.0..=have.1).contains(cy))
        .collect()
}

/// The runs of cubes `have` holds that `next` does not: below and above.
pub fn leaving_ranges(have: (i32, i32), next: (i32, i32)) -> Vec<std::ops::RangeInclusive<i32>> {
    let mut ranges = Vec::new();
    if next.0 > have.0 {
        ranges.push(have.0..=(next.0 - 1).min(have.1));
    }
    if next.1 < have.1 {
        ranges.push((next.1 + 1).max(have.0)..=have.1);
    }
    ranges
}

/// A cubic level's cube store and the writes queued for it, shared with its
/// writer thread, which drains the queue before it ends.
struct CubeIo {
    store: CubeStore,
    /// Cube records queued for the writer and not yet on disk: loads read
    /// these first.
    pending: Mutex<HashMap<SectionPos, Arc<NamedTag>>>,
    /// Writes in flight, and a signal for `flush`.
    queued: (Mutex<usize>, Condvar),
}

impl CubeIo {
    fn read(&self, pos: SectionPos) -> Option<NamedTag> {
        // Only the `Arc` is cloned under the lock; the record is copied
        // outside it, so loads never wait on a copy.
        let queued = self.pending.lock().unwrap().get(&pos).cloned();
        if let Some(tag) = queued {
            return Some((*tag).clone());
        }
        match self.store.read(pos) {
            Ok(tag) => tag,
            Err(e) => {
                log::warn!("Horizon: cube {pos:?} unreadable: {e}");
                None
            }
        }
    }

    fn write_now(&self, pos: SectionPos, tag: &Arc<NamedTag>) {
        if let Err(e) = store_record(&self.store, pos, tag) {
            log::warn!("Horizon: cube {pos:?} not saved: {e}");
        }
        let mut pending = self.pending.lock().unwrap();
        if pending.get(&pos).is_some_and(|t| Arc::ptr_eq(t, tag)) {
            pending.remove(&pos);
        }
        drop(pending);
        let (count, done) = &self.queued;
        let mut count = count.lock().unwrap();
        *count = count.saturating_sub(1);
        if *count == 0 {
            done.notify_all();
        }
    }

    fn flush(&self) {
        let (count, done) = &self.queued;
        let mut count = count.lock().unwrap();
        while *count > 0 {
            count = done.wait(count).unwrap();
        }
        drop(count);
        if let Err(e) = self.store.sync() {
            log::warn!("Horizon: cubes not synced: {e}");
        }
    }
}

/// Writes cube record `tag` at `pos`: in the store's uniform sidecar when it
/// is one block state and nothing else, else in a region file.
pub(crate) fn store_record(
    store: &CubeStore,
    pos: SectionPos,
    tag: &NamedTag,
) -> std::io::Result<()> {
    if is_uniform_record(tag) {
        store
            .write_uniform(pos, tag)
            .or_else(|_| store.write(pos, tag))
    } else {
        store.write(pos, tag)
    }
}

/// One cube of a vanilla column, as `finish_vanilla` saves it.
pub(crate) struct ColumnCube {
    pub cy: i32,
    /// `None` for a cube that needs no record (air with nothing in it; a
    /// missing vanilla cube reads as air).
    pub root: Option<NamedTag>,
    pub block_ticks: Vec<SavedTick>,
    pub fluid_ticks: Vec<SavedTick>,
    pub has_entities: bool,
}

/// A finished vanilla column's cubes as records (`vanilla_column_records`).
pub(crate) struct ColumnRecords {
    pub cubes: Vec<ColumnCube>,
    /// The column's surface heights, kept in its marker cube.
    pub heights: Vec<i32>,
}

/// The records of a finished vanilla column (`-64..320`) in a Hybrid world:
/// one per vanilla cube, `entities` (by cube, `split_by_cube`) in theirs, the
/// marker cube with the column's structures and surface heights. The one
/// definition of the record format: `finish_vanilla` and the world converter
/// both save through it.
pub(crate) fn vanilla_column_records(
    registry: &BlockRegistry,
    biomes: &Biomes,
    tables: Option<&SaveTables>,
    vanilla: &Chunk,
    entities: &BTreeMap<i32, Vec<Tag>>,
) -> ColumnRecords {
    let ctx = Codec {
        registry,
        biomes,
        tables,
    };
    let air = |v: u32| v == 0;
    let heights: Vec<i32> = (0..256)
        .map(|i| vanilla.world_surface(i & 15, i >> 4, air))
        .collect();
    let vanilla_base = vanilla.min_y() >> 4;
    let mut cubes = Vec::new();
    for (s, section) in vanilla.sections.iter().enumerate() {
        let cy = vanilla_base + s as i32;
        if !VANILLA_CUBES.contains(&cy) {
            continue;
        }
        let (block_ticks, fluid_ticks) = cube_ticks(vanilla, cy);
        let marker = cy == MARKER_CUBE;
        let low = cy * 16;
        // A missing vanilla cube reads as air: the air above the ground
        // needs no record (the marker always has one).
        let bare = !marker
            && section.blocks.uniform() == Some(0)
            && block_ticks.is_empty()
            && fluid_ticks.is_empty()
            && !entities.contains_key(&cy)
            && !vanilla
                .block_entities
                .iter()
                .any(|(p, _)| p.1 >= low && p.1 < low + 16);
        let here = entities.get(&cy).map_or(&[][..], |e| &e[..]);
        let root = (!bare).then(|| {
            let mut root = ctx.encode(vanilla, s, marker, here);
            if marker && let Tag::Compound(map) = &mut root.tag {
                map.insert(SURFACE_KEY.into(), Tag::IntArray(heights.clone()));
            }
            root
        });
        cubes.push(ColumnCube {
            cy,
            root,
            block_ticks,
            fluid_ticks,
            has_entities: !here.is_empty(),
        });
    }
    ColumnRecords { cubes, heights }
}

/// Vanilla's bedrock floor (y -64..-60) as `deepslate`, so the world can go
/// deeper.
pub(crate) fn replace_bedrock(chunk: &mut Chunk, bedrock: u32, deepslate: u32) {
    for y in chunk.min_y()..chunk.min_y() + 5 {
        for z in 0..16 {
            for x in 0..16 {
                if chunk.get(x, y, z) == bedrock {
                    chunk.set(x, y, z, deepslate);
                }
            }
        }
    }
}

/// One cubic level's cubes on disk and in generation.
pub struct CubicLevel {
    io: Arc<CubeIo>,
    send: Mutex<Option<Sender<(SectionPos, Arc<NamedTag>)>>>,
    /// Columns whose vanilla range is known to be generated (`has_vanilla`).
    vanilla_done: Mutex<std::collections::HashSet<(i32, i32)>>,
    wg: Arc<Generation>,
    /// Population state: flags, cubes in flight, the striped locks.
    pop: Population,
    /// The window the level's chunks hold now.
    window: Mutex<ColumnLayout>,
    /// A hash of each cube's blocks as last loaded, generated or saved: a
    /// save skips cubes whose blocks are the same and that hold no block
    /// entity.
    clean: Mutex<HashMap<SectionPos, u64>>,
    /// Cubes whose record holds entities, as far as this level has loaded or
    /// saved them: an entity save looks only at these and at the cubes it
    /// brings entities for.
    with_entities: Mutex<HashSet<SectionPos>>,
    /// Held while a cube record is read, changed and queued, so a block save
    /// and an entity save of one cube do not lose each other's data.
    edit: Mutex<()>,
    /// The save tables cube records are coded with (set by the terrain).
    tables: std::sync::OnceLock<Arc<SaveTables>>,
    /// Uniform cubes made by the generator, shared by every column.
    uniform: Mutex<HashMap<(u32, u16), Arc<Section>>>,
    /// The air section of a generated vanilla range's missing cubes.
    air: std::sync::OnceLock<Arc<Section>>,
    bedrock: Option<u32>,
    deepslate: Option<u32>,
    /// What cube records are decoded with for the far terrain.
    registry: &'static BlockRegistry,
    biomes: Arc<Biomes>,
    /// The far terrain the level remembers (`horizon_cube_lod/`): every cube
    /// saved erases the nodes over it.
    lod_cache: Option<Arc<crate::lod::cube_cache::CubeLodCache>>,
    /// The cubes just outside the window, ready in memory
    /// (`cubic_prefetch`).
    ring: crate::cubic_prefetch::Ring,
    /// Each column's vanilla surface heights, for the sky a window's top
    /// leaves open (`sky_cover`); `None` for a column without them.
    surfaces: Mutex<HashMap<(i32, i32), Option<Arc<Vec<i32>>>>>,
    /// Held while a column's vanilla range is written (`finish_vanilla`): two
    /// shards that both need a column generated write it once, and the second
    /// reads what the first wrote instead of overwriting edits made since.
    vanilla_write: Mutex<()>,
    /// Makes the streaming of another shard (set where the level's terrain is
    /// built).
    shard_maker: std::sync::OnceLock<ShardMaker>,
    /// The window cell of the primary shard's terrain.
    primary_window: std::sync::OnceLock<ShardWindow>,
}

impl CubicLevel {
    /// The cubic level saved in `level_dir` (its `region3d/`), generating
    /// with `seed` as `generation` says, its chunks holding `window`. `data`
    /// is the generator data of 26.3 (its biomes and features); without it
    /// the level has plains everywhere and no vegetation.
    pub fn new(
        level_dir: &Path,
        seed: i64,
        registry: &'static BlockRegistry,
        biomes: &Arc<Biomes>,
        window: LevelShape,
        generation: &CubicGeneration,
        data: Option<&Arc<horizon_world::worldgen::generator::GeneratorData>>,
    ) -> Result<Arc<Self>, String> {
        let wg = Arc::new(Generation::new(generation, seed, registry, data)?);
        let io = Arc::new(CubeIo {
            store: CubeStore::new(level_dir),
            pending: Mutex::default(),
            queued: (Mutex::new(0usize), Condvar::new()),
        });
        let (send, receive) = channel::<(SectionPos, Arc<NamedTag>)>();
        let writer = io.clone();
        std::thread::Builder::new()
            .name("Cube writer".into())
            .spawn(move || write_cubes(&writer, receive))
            .map_err(|e| e.to_string())?;
        let level = Arc::new(Self {
            io,
            send: Mutex::new(Some(send)),
            vanilla_done: Mutex::default(),
            wg,
            pop: Population::new(),
            window: Mutex::new(window.column()),
            clean: Mutex::default(),
            with_entities: Mutex::default(),
            edit: Mutex::default(),
            tables: std::sync::OnceLock::new(),
            uniform: Mutex::default(),
            air: std::sync::OnceLock::new(),
            bedrock: registry.default_state("minecraft:bedrock").map(|s| s.0),
            deepslate: registry.default_state("minecraft:deepslate").map(|s| s.0),
            registry,
            biomes: biomes.clone(),
            lod_cache: crate::lod::cube_cache::CubeLodCache::open(
                &level_dir.join("horizon_cube_lod"),
                &format!(
                    "{}generator {} {:016x}
",
                    crate::lod::cube_cache::fingerprint(seed, registry.len()),
                    generation.kind.id(),
                    settings_hash(&generation.settings_json)
                ),
            ),
            ring: crate::cubic_prefetch::Ring::new(),
            surfaces: Mutex::default(),
            vanilla_write: Mutex::new(()),
            shard_maker: std::sync::OnceLock::new(),
            primary_window: std::sync::OnceLock::new(),
        });
        level.ring.start(Arc::downgrade(&level));
        Ok(level)
    }

    /// A saved cube's section (blocks and biomes), for the far terrain:
    /// `None` when the cube was never saved (the generator's as it is).
    pub fn saved_section(&self, pos: SectionPos) -> Option<Arc<Section>> {
        let root = self.read(pos)?;
        let ctx = Codec {
            registry: self.registry,
            biomes: &self.biomes,
            tables: None,
        };
        let cube = ctx.decode(&root, pos.y)?;
        cube.sections.into_iter().next()
    }

    /// A far-terrain sampler of the level (`cubic_lod`), reading a Hybrid
    /// world's vanilla band from `vanilla`, the vanilla generator's.
    pub fn lod_sampler(
        self: &Arc<Self>,
        vanilla: Box<dyn crate::streaming::LodSampler>,
        registry: &BlockRegistry,
    ) -> crate::cubic_lod::CubicLodSampler {
        crate::cubic_lod::CubicLodSampler::new(
            self.clone(),
            vanilla,
            registry,
            self.lod_cache.clone(),
        )
    }

    /// Where a new world's spawn search starts (`game/initial_spawn.rs`): the
    /// chunk and the height to stand at, for a world that is not Hybrid. A
    /// Flat world's is the origin; a Custom world's, the first chunk of a
    /// spiral round it whose surface is above the sea (land).
    pub fn spawn_origin(&self) -> Option<((i32, i32), i32)> {
        let generator = &self.wg.generator;
        match self.wg.kind {
            CubicKind::Hybrid => None,
            CubicKind::Flat => {
                let y = generator.surface_y(8, 8, 320).unwrap_or(0);
                Some(((0, 0), y + 1))
            }
            CubicKind::Custom => {
                let sea = generator.sea_level();
                let mut best: Option<((i32, i32), i32)> = None;
                for radius in 0..24i32 {
                    for dz in -radius..=radius {
                        for dx in -radius..=radius {
                            if dx.abs().max(dz.abs()) != radius {
                                continue;
                            }
                            let (cx, cz) = (dx * 2, dz * 2);
                            let Some(y) = generator.surface_y(cx * 16 + 8, cz * 16 + 8, 320) else {
                                continue;
                            };
                            if y > sea {
                                return Some(((cx, cz), y + 1));
                            }
                            best.get_or_insert(((cx, cz), y + 1));
                        }
                    }
                }
                best.or(Some(((0, 0), sea + 1)))
            }
        }
    }

    /// The client's biome table (the ids sections and far nodes carry).
    pub fn client_biomes(&self) -> &Arc<Biomes> {
        &self.biomes
    }

    /// The client biome id of the block column (`x`, `z`) in the biome
    /// source.
    pub fn biome_id_at(&self, x: i32, z: i32) -> u16 {
        let id = self.wg.columns.biome(x >> 2, z >> 2);
        self.biomes.id(self.wg.columns.biome_name(id))
    }

    /// The window of the shard in place (`Game::swap_level` and a slide set
    /// it): what an entity save stands for (`save_entities`). A shard's
    /// terrain loads into the window of its own (`CubicTerrain::window`).
    pub fn window(&self) -> ColumnLayout {
        *self.window.lock().unwrap()
    }

    /// Moves the window of the shard in place: the columns it loads from now
    /// on hold `window`. The caller saved the leaving cubes and their
    /// entities (`save_leaving_entities`) first. What is known of the cubes
    /// that stay (`clean`) stays.
    pub fn set_window(&self, window: LevelShape) {
        *self.window.lock().unwrap() = window.column();
    }

    /// The window cell of the primary shard's terrain, which the game moves
    /// with the primary's world.
    pub fn primary_window(&self) -> Option<ShardWindow> {
        self.primary_window.get().cloned()
    }

    /// Records the primary shard's window cell (where its terrain is built).
    pub fn set_primary_window(&self, window: ShardWindow) {
        let _ = self.primary_window.set(window);
    }

    /// Lets the level make the streaming of other shards (`new_shard_streaming`).
    pub fn set_shard_maker(&self, maker: ShardMaker) {
        let _ = self.shard_maker.set(maker);
    }

    /// The streaming of a new shard around `window`: the same terrain, saves,
    /// generator and population as the level's others, loading into
    /// its own window. `None` when the level cannot make one.
    pub fn new_shard_streaming(&self, window: ShardWindow) -> Option<crate::streaming::Streaming> {
        self.shard_maker.get().map(|make| make(window))
    }

    /// Forgets that cube `pos` was loaded as it is (it left the window and
    /// was saved): memory for a flight of a million cubes would otherwise
    /// grow with it.
    pub fn release_cube(&self, pos: SectionPos) {
        self.clean.lock().unwrap().remove(&pos);
    }

    /// Aims the prefetch ring at the layers around `window` of `columns`
    /// (the loaded columns of shard `shard`), nearest to `centres` first
    /// (`cubic_prefetch`). Each shard has its own target; the ring holds all.
    pub fn retarget_shard_ring(
        &self,
        shard: u16,
        window: ColumnLayout,
        centres: &[(i32, i32)],
        columns: impl IntoIterator<Item = (i32, i32)>,
    ) {
        self.ring.retarget_shard(shard, window, centres, columns);
    }

    /// `retarget_shard_ring` for the primary shard and the window in place.
    pub fn retarget_ring(
        &self,
        centres: &[(i32, i32)],
        columns: impl IntoIterator<Item = (i32, i32)>,
    ) {
        self.ring.retarget(self.window(), centres, columns);
    }

    /// Drops shard `shard`'s ring target (the shard closed).
    pub fn drop_ring_target(&self, shard: u16) {
        self.ring.forget(shard);
    }

    /// Whether cube `pos` waits in the ring.
    pub fn is_prefetched(&self, pos: SectionPos) -> bool {
        self.ring.has(pos)
    }

    /// Cube `pos`'s record: a queued write's, else the disk's.
    pub fn read(&self, pos: SectionPos) -> Option<NamedTag> {
        self.io.read(pos)
    }

    fn write_later(&self, pos: SectionPos, mut tag: NamedTag) {
        // The record keeps its population flags whoever writes it.
        let flags = match self.pop.flags(pos) {
            Some(flags) => flags,
            None => {
                let flags = self.read(pos).map_or(0, |old| root_flags(&old));
                self.pop.learn_flags(pos, flags);
                flags
            }
        };
        set_root_flags(&mut tag, flags);
        self.pop.forget(pos);
        if let Some(cache) = &self.lod_cache {
            cache.invalidate(pos);
        }
        if self.is_marker(pos.y) {
            self.surfaces.lock().unwrap().remove(&(pos.x, pos.z));
        }
        let tag = Arc::new(tag);
        self.io.pending.lock().unwrap().insert(pos, tag.clone());
        // A cube read before this write is older than its record. Dropped
        // once the record can be read, so the ring (one for every shard) does
        // not read the old one again in between.
        self.ring.invalidate(pos);
        *self.io.queued.0.lock().unwrap() += 1;
        let sent = self
            .send
            .lock()
            .unwrap()
            .as_ref()
            .is_some_and(|send| send.send((pos, tag.clone())).is_ok());
        if !sent {
            // No writer: write here.
            self.io.write_now(pos, &tag);
        }
    }

    /// Waits until every queued cube is on disk.
    pub fn flush(&self) {
        self.io.flush();
    }

    /// Whether column (x, z)'s vanilla range was generated (its marker cube
    /// is saved).
    pub fn has_vanilla(&self, x: i32, z: i32) -> bool {
        if self.wg.kind != CubicKind::Hybrid {
            return true;
        }
        if self.vanilla_done.lock().unwrap().contains(&(x, z)) {
            return true;
        }
        let done = self.read(SectionPos::new(x, MARKER_CUBE, z)).is_some();
        if done {
            self.vanilla_done.lock().unwrap().insert((x, z));
        }
        done
    }

    /// The level's world type.
    pub fn kind(&self) -> CubicKind {
        self.wg.kind
    }

    /// The generator and what hangs off it (far terrain, spawn).
    pub fn generation(&self) -> &Arc<Generation> {
        &self.wg
    }

    /// Whether cube Y `cy` is vanilla's (a Hybrid world's band).
    pub fn is_vanilla(&self, cy: i32) -> bool {
        crate::cubic_gen::is_vanilla_cube(self.wg.kind, cy)
    }

    /// Whether `cy` is the cube that holds a Hybrid column's own data.
    fn is_marker(&self, cy: i32) -> bool {
        self.wg.kind == CubicKind::Hybrid && cy == MARKER_CUBE
    }

    /// Population counters (cubes generated and populated, mean times).
    pub fn population_stats(&self) -> crate::cubic_gen::PopulationStats {
        self.pop.stats()
    }

    /// The generator's terrain for cube `pos` (before population), uniform
    /// cubes shared, with the biomes of the biome source.
    fn generate(&self, pos: SectionPos) -> Arc<Section> {
        if !horizon_world::cubic::has_cube(pos) {
            return self.air.get_or_init(Default::default).clone();
        }
        let start = std::time::Instant::now();
        let mut values = Box::new([0u32; 4096]);
        let uniform = self.wg.generator.generate(pos, &mut values);
        let biomes = self.wg.biome_container(&self.biomes, pos.x, pos.z);
        let section = match (uniform, biomes.uniform()) {
            (Some(state), Some(biome)) => {
                let mut shared = self.uniform.lock().unwrap();
                shared
                    .entry((state, biome as u16))
                    .or_insert_with(|| {
                        Arc::new(Section {
                            blocks: PalettedContainer::new(block_strategy(), state),
                            biomes: biomes.clone(),
                            ..Section::default()
                        })
                    })
                    .clone()
            }
            (Some(state), None) => Arc::new(Section {
                blocks: PalettedContainer::new(block_strategy(), state),
                biomes,
                ..Section::default()
            }),
            (None, _) => {
                let mut section = Section {
                    blocks: Section::from_blocks(&values),
                    biomes,
                    ..Section::default()
                };
                section.compact();
                Arc::new(section)
            }
        };
        self.pop
            .generated
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.pop.generate_nanos.fetch_add(
            start.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        section
    }

    /// Cube `pos` as terrain, with whatever population wrote into it so far:
    /// a cube in flight, else its record, else the generator's. Whether it
    /// came from a record or a cube in flight is the second answer.
    fn terrain_section(&self, pos: SectionPos) -> (Arc<Section>, bool) {
        if let Some(section) = self.pop.cached(pos) {
            return (section, true);
        }
        if let Some(root) = self.read(pos) {
            self.pop.learn_flags(pos, root_flags(&root));
            if let Some(cube) = self.codec().decode(&root, pos.y)
                && let Some(section) = cube.sections.into_iter().next()
            {
                self.pop.keep(pos, section.clone());
                return (section, true);
            }
        }
        let section = self.generate(pos);
        self.pop.keep(pos, section.clone());
        (section, false)
    }

    /// Terrain of cube `pos` for a light view (no population).
    pub fn terrain(&self, pos: SectionPos) -> Arc<Section> {
        self.terrain_section(pos).0
    }

    /// Whether the population of `p` ran (its record's flag).
    fn is_populated(&self, p: SectionPos) -> bool {
        let flags = match self.pop.flags(p) {
            Some(flags) => flags,
            None => {
                let flags = self.read(p).map_or(0, |root| root_flags(&root));
                self.pop.learn_flags(p, flags);
                flags
            }
        };
        flags & POPULATED != 0
    }

    /// Runs the eight populations that touch `pos` that have not run
    /// (`c - d` for `d` in 0 or 1 on each axis), so the cube is final.
    fn ensure_populated_around(&self, pos: SectionPos) {
        for d in 0..8 {
            let p = SectionPos::new(pos.x - (d & 1), pos.y - (d >> 1 & 1), pos.z - (d >> 2 & 1));
            if needs_population(self.wg.kind, p)
                && horizon_world::cubic::has_cube_y(p.y)
                && !self.is_populated(p)
            {
                self.populate(p);
            }
        }
    }

    /// CubicWorldGen's population of cube `p`: its 2 x 2 x 2 region (terrain
    /// from the records that exist, else the generator) gets ores, lakes,
    /// vegetation and springs, and the cubes it changed are written, `p`'s
    /// record flagged populated. Regions that overlap are serialised by the
    /// striped locks, which every population takes in the same order.
    fn populate(&self, p: SectionPos) {
        let _locks = self.pop.lock_region(p);
        if self.is_populated(p) {
            return;
        }
        let start = std::time::Instant::now();
        let at = |d: usize| {
            SectionPos::new(
                p.x + (d & 1) as i32,
                p.y + (d >> 1 & 1) as i32,
                p.z + (d >> 2 & 1) as i32,
            )
        };
        let loaded: [(Arc<Section>, bool); 8] =
            std::array::from_fn(|d| self.terrain_section(at(d)));
        let sections: [Arc<Section>; 8] = std::array::from_fn(|d| loaded[d].0.clone());
        let mut region = PopRegion::new(
            p,
            &sections,
            self.wg.columns.clone(),
            self.wg.vegetation.clone(),
        );
        self.wg.generator.populate(p, &mut region);
        let dirty = region.dirty();
        let ctx = self.codec();
        for d in 0..8 {
            let q = at(d);
            if !horizon_world::cubic::has_cube(q) {
                continue;
            }
            let own = d == 0;
            if !own && !dirty[d] {
                continue;
            }
            let section = if dirty[d] {
                let mut section = Section {
                    blocks: Section::from_blocks(region.blocks(d)),
                    biomes: sections[d].biomes.clone(),
                    ..Section::default()
                };
                section.compact();
                Arc::new(section)
            } else {
                sections[d].clone()
            };
            let known = self.pop.flags(q).unwrap_or(0);
            let flags = if own {
                (known | POPULATED) & !PARTIAL
            } else if known & (POPULATED | PARTIAL) != 0 || (loaded[d].1 && self.read(q).is_some())
            {
                known
            } else {
                known | PARTIAL
            };
            self.pop.set_flags(q, flags);
            let mut one = Chunk::with_layout(q.x, q.z, ColumnLayout::window(q.y * 16, 16));
            one.sections[0] = section.clone();
            let entities = self
                .read(q)
                .map(|old| record_entities(&old))
                .unwrap_or_default();
            let root = ctx.encode(&one, 0, false, &entities);
            self.write_later(q, root);
            self.pop.keep(q, section);
        }
        self.pop
            .populations
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.pop.populate_nanos.fetch_add(
            start.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
    }

    fn note_clean(
        &self,
        pos: SectionPos,
        section: &Section,
        block_ticks: &[SavedTick],
        fluid_ticks: &[SavedTick],
    ) {
        let hash = cube_hash(section, block_ticks, fluid_ticks);
        self.clean.lock().unwrap().insert(pos, hash);
    }

    /// The save tables cube records are coded with.
    fn codec(&self) -> Codec<'_> {
        Codec {
            registry: self.registry,
            biomes: &self.biomes,
            tables: self.tables.get().map(|t| &**t),
        }
    }

    /// Sets the save tables `load_cube`/`save_cube` code records with (the
    /// terrain does it before its first load).
    pub fn set_tables(&self, tables: Arc<SaveTables>) {
        let _ = self.tables.set(tables);
    }

    /// Cube `pos` as saved: its record's blocks, block entities, entities and
    /// ticks. With no record, a cube outside vanilla's range is the
    /// generator's (`generated`, no entities), and one inside it is `None`:
    /// air in a column whose vanilla range was generated (`has_vanilla`).
    /// The cube is noted as unchanged (`save_cube` skips it until it
    /// changes) and its entities as saved there. For a window moving in
    /// place: the cubes entering a loaded column come from here.
    pub fn load_cube(&self, pos: SectionPos) -> Option<LoadedCube> {
        self.load_cube_with(pos, &self.codec())
    }

    fn load_cube_with(&self, pos: SectionPos, ctx: &Codec) -> Option<LoadedCube> {
        let cube = self.read_cube(pos, ctx)?;
        self.adopt_cube(&cube);
        Some(cube)
    }

    /// Notes cube `cube` as loaded: unchanged since, and holding entities or
    /// not, as far as saves are concerned.
    fn adopt_cube(&self, cube: &LoadedCube) {
        self.note_clean(
            cube.pos,
            &cube.section,
            &cube.block_ticks,
            &cube.fluid_ticks,
        );
        if !cube.generated {
            self.note_entities(cube.pos, !cube.entities.is_empty());
        }
    }

    /// Cube `pos` as it is now, changing nothing in the level's books (so
    /// the ring's thread may call it): `load_cube`'s answer.
    fn read_cube(&self, pos: SectionPos, ctx: &Codec) -> Option<LoadedCube> {
        // A cube outside vanilla's band is final once the eight populations
        // that touch it have run.
        if !self.is_vanilla(pos.y) {
            self.ensure_populated_around(pos);
        }
        if let Some(root) = self.read(pos)
            && let Some(mut cube) = ctx.decode(&root, pos.y)
        {
            self.pop.learn_flags(pos, root_flags(&root));
            let entities = record_entities(&root);
            let saved = cube.saved.take();
            let (block_ticks, fluid_ticks) = saved
                .map(|e| (e.block_ticks.clone(), e.fluid_ticks.clone()))
                .unwrap_or_default();
            let marker = self.is_marker(pos.y);
            let section = cube.sections[0].clone();
            return Some(LoadedCube {
                pos,
                section,
                block_entities: cube.block_entities,
                block_names: cube.block_names,
                entities,
                block_ticks,
                fluid_ticks,
                structure_refs: if marker {
                    cube.structure_refs
                } else {
                    Vec::new()
                },
                structure_starts: if marker {
                    cube.structure_starts
                } else {
                    Vec::new()
                },
                surface: if marker { surface_heights(&root) } else { None },
                generated: false,
            });
        }
        if self.is_vanilla(pos.y) {
            return None;
        }
        let mut cube = LoadedCube::empty(pos, self.terrain_section(pos).0);
        cube.generated = true;
        Some(cube)
    }

    /// Whether `make_cube` can answer for `pos` now: not for a cube in
    /// vanilla's range of a column whose vanilla generation has not run
    /// (that column goes the normal way), nor before the save tables are set.
    pub fn can_make_cube(&self, pos: SectionPos) -> bool {
        self.tables.get().is_some() && (!self.is_vanilla(pos.y) || self.has_vanilla(pos.x, pos.z))
    }

    /// Cube `pos` from its record, else the generator's, else air (a cube of
    /// a generated vanilla range with no record). The one place cubes are
    /// made; a generator needing neighbours extends it here. Changes nothing
    /// in the level's books: [`Self::ready_cube`] does that for the cube it
    /// hands out.
    pub fn make_cube(&self, pos: SectionPos) -> LoadedCube {
        self.read_cube(pos, &self.codec()).unwrap_or_else(|| {
            LoadedCube::empty(pos, self.air.get_or_init(Default::default).clone())
        })
    }

    /// Cube `pos` for a window sliding over it: the ring's when it is ready,
    /// else made now. The level notes it as loaded (unchanged, its entities
    /// as saved). For a cube in vanilla's range the column must have its
    /// vanilla generation (`can_make_cube`).
    pub fn ready_cube(&self, pos: SectionPos) -> LoadedCube {
        let cube = self.ring.take(pos).unwrap_or_else(|| self.make_cube(pos));
        self.adopt_cube(&cube);
        cube
    }

    fn note_entities(&self, pos: SectionPos, has: bool) {
        let mut set = self.with_entities.lock().unwrap();
        if has {
            set.insert(pos);
        } else {
            set.remove(&pos);
        }
    }

    /// Column (x, z) of the window, from saved cubes and the generator; its
    /// vanilla range must have been generated (`has_vanilla`). The column
    /// holds the entities of its cubes (`entities`) and their ticks.
    fn assemble(&self, x: i32, z: i32, window: ColumnLayout, ctx: &Codec) -> Chunk {
        let mut chunk = Chunk::with_layout(x, z, window);
        let base = window.min_y() >> 4;
        let mut surface = None;
        let (mut block_ticks, mut fluid_ticks) = (Vec::new(), Vec::new());
        for s in 0..window.sections() {
            let cy = base + s as i32;
            let pos = SectionPos::new(x, cy, z);
            let Some(cube) = self.load_cube_with(pos, ctx) else {
                // Missing from a generated column: air, as vanilla's empty
                // sections (and as saved).
                self.note_clean(pos, &chunk.sections[s], &[], &[]);
                continue;
            };
            chunk.sections[s] = cube.section;
            chunk.block_entities.extend(cube.block_entities);
            chunk.block_names.extend(cube.block_names);
            chunk.entities.extend(cube.entities);
            block_ticks.extend(cube.block_ticks);
            fluid_ticks.extend(cube.fluid_ticks);
            if self.is_marker(cy) {
                chunk.structure_refs = cube.structure_refs;
                chunk.structure_starts = cube.structure_starts;
                surface = cube.surface;
            }
        }
        if !block_ticks.is_empty() || !fluid_ticks.is_empty() {
            chunk.saved = Some(Arc::new(ChunkExtras {
                block_ticks,
                fluid_ticks,
                ..ChunkExtras::default()
            }));
        }
        if self.wg.kind != CubicKind::Hybrid {
            chunk.sky_covered = self.sky_cover(x, z, window);
            return chunk;
        }
        // The column's own data lives in the marker cube. A window that does
        // not hold it (another shard may) reads it: the same surface and
        // structure data whichever shard loads the column.
        if surface.is_none()
            && let Some(root) = self.read(SectionPos::new(x, MARKER_CUBE, z))
        {
            surface = surface_heights(&root);
            if let Some(cube) = ctx.decode(&root, MARKER_CUBE) {
                chunk.structure_refs = cube.structure_refs;
                chunk.structure_starts = cube.structure_starts;
            }
        }
        chunk.sky_covered = sky_covered(surface.as_deref(), window);
        chunk
    }

    /// A finished vanilla column (`-64..320`): its bedrock floor replaced by
    /// deepslate, its 24 cubes saved, and the window's column made of it and
    /// the generator's cubes.
    fn finish_vanilla(&self, mut vanilla: Chunk, window: ColumnLayout, ctx: &Codec) -> Chunk {
        let (x, z) = (vanilla.x, vanilla.z);
        // Two shards can both need this column generated. The first writes
        // its cubes; the second takes them from the store, so it does not write
        // the generated cubes over edits made since.
        let _write = self.vanilla_write.lock().unwrap();
        if self.vanilla_done.lock().unwrap().contains(&(x, z)) {
            return self.assemble(x, z, window, ctx);
        }
        self.strip_bedrock(&mut vanilla);
        // The column's entities by cube: each is saved in its cube's record.
        let entities = split_by_cube(std::mem::take(&mut vanilla.entities));
        let records =
            vanilla_column_records(self.registry, &self.biomes, ctx.tables, &vanilla, &entities);
        let heights = records.heights;
        let vanilla_base = vanilla.min_y() >> 4;
        for cube in records.cubes {
            let pos = SectionPos::new(x, cube.cy, z);
            let section = &vanilla.sections[(cube.cy - vanilla_base) as usize];
            self.note_clean(pos, section, &cube.block_ticks, &cube.fluid_ticks);
            let Some(root) = cube.root else {
                continue;
            };
            self.note_entities(pos, cube.has_entities);
            self.write_later(pos, root);
        }
        self.vanilla_done.lock().unwrap().insert((x, z));
        let mut chunk = Chunk::with_layout(x, z, window);
        let base = window.min_y() >> 4;
        for s in 0..window.sections() {
            let cy = base + s as i32;
            let pos = SectionPos::new(x, cy, z);
            let y = cy * 16;
            if let Some(i) = vanilla.section_index(y)
                && self.is_vanilla(cy)
            {
                chunk.sections[s] = vanilla.sections[i].clone();
            } else if !self.is_vanilla(cy) {
                // Outside the band: terrain with the populations around it run.
                if let Some(cube) = self.load_cube_with(pos, ctx) {
                    chunk.sections[s] = cube.section;
                    chunk.block_entities.extend(cube.block_entities);
                    chunk.block_names.extend(cube.block_names);
                    chunk.entities.extend(cube.entities);
                }
            }
        }
        let inside = |y: i32| y >= window.min_y() && y < window.max_y();
        chunk.block_entities = std::mem::take(&mut vanilla.block_entities)
            .into_iter()
            .filter(|(p, _)| inside(p.1))
            .collect();
        chunk.block_names = std::mem::take(&mut vanilla.block_names)
            .into_iter()
            .filter(|(p, _)| inside(p.1))
            .collect();
        chunk.postprocess = std::mem::take(&mut vanilla.postprocess)
            .into_iter()
            .filter(|p| inside(p[1]))
            .collect();
        chunk.structure_refs = std::mem::take(&mut vanilla.structure_refs);
        chunk.structure_starts = std::mem::take(&mut vanilla.structure_starts);
        // The window's ticks; the others are in their cubes' records.
        chunk.saved = vanilla.saved.take().map(|saved| {
            let mut saved = (*saved).clone();
            let keep = |t: &SavedTick| inside(t.pos.y);
            saved.block_ticks.retain(keep);
            saved.fluid_ticks.retain(keep);
            Arc::new(saved)
        });
        chunk.fresh = vanilla.fresh;
        chunk.generated = vanilla.generated;
        // Those in the window spawn now; all are in their records, and the
        // column's entity save replaces the window's.
        chunk.entities = entities
            .into_iter()
            .filter(|(cy, _)| inside(cy * 16))
            .flat_map(|(_, e)| e)
            .collect();
        chunk.sky_covered = sky_covered(Some(&heights), window);
        chunk
    }

    fn strip_bedrock(&self, chunk: &mut Chunk) {
        if let (Some(bedrock), Some(deepslate)) = (self.bedrock, self.deepslate) {
            replace_bedrock(chunk, bedrock, deepslate);
        }
    }

    /// The columns whose sky is cut off above `window` at column (`x`, `z`)
    /// (`Chunk::sky_covered`): from a Hybrid column's vanilla surface (its
    /// marker cube's), from the generator's surface elsewhere (cached).
    pub fn sky_cover(&self, x: i32, z: i32, window: ColumnLayout) -> Option<Arc<[u64; 4]>> {
        let cached = self.surfaces.lock().unwrap().get(&(x, z)).cloned();
        let surface = match cached {
            Some(surface) => surface,
            None => {
                let surface = if self.wg.kind == CubicKind::Hybrid {
                    self.read(SectionPos::new(x, MARKER_CUBE, z))
                        .and_then(|root| surface_heights(&root))
                        .map(Arc::new)
                } else {
                    Some(Arc::new(self.generated_surface(x, z)))
                };
                let mut surfaces = self.surfaces.lock().unwrap();
                if surfaces.len() >= 1 << 16 {
                    surfaces.clear();
                }
                surfaces.insert((x, z), surface.clone());
                surface
            }
        };
        sky_covered(surface.as_deref().map(Vec::as_slice), window)
    }

    /// One above the generator's surface of each block column of cube column
    /// (`x`, `z`) (`z << 4 | x`), looked at once per 8 x 8 blocks from y 320
    /// (columns with no surface read as far below).
    fn generated_surface(&self, x: i32, z: i32) -> Vec<i32> {
        let mut heights = vec![i32::MIN / 2; 256];
        let mut sample = [[0i32; 2]; 2];
        for (qz, row) in sample.iter_mut().enumerate() {
            for (qx, cell) in row.iter_mut().enumerate() {
                let (bx, bz) = (x * 16 + qx as i32 * 8 + 4, z * 16 + qz as i32 * 8 + 4);
                *cell = self
                    .wg
                    .generator
                    .surface_y(bx, bz, 320)
                    .map_or(i32::MIN / 2, |y| y + 1);
            }
        }
        for (i, h) in heights.iter_mut().enumerate() {
            let (bx, bz) = (i & 15, i >> 4);
            *h = sample[bz / 8][bx / 8];
        }
        heights
    }

    /// Saves the `entities` of column (`x`, `z`) in cubes `cubes` (those
    /// leaving a window that slides): each of those cubes' records takes the
    /// entities of the list that are in it, and loses the ones it had. Other
    /// cubes are untouched, unlike [`Self::save_entities`], whose list stands
    /// for the whole window.
    pub fn save_leaving_entities(
        &self,
        x: i32,
        z: i32,
        cubes: std::ops::RangeInclusive<i32>,
        entities: Vec<Tag>,
    ) {
        let mut groups = split_by_cube(entities);
        for cy in cubes {
            let here = groups.remove(&cy).unwrap_or_default();
            let pos = SectionPos::new(x, cy, z);
            let known = self.with_entities.lock().unwrap().contains(&pos);
            if here.is_empty() && !known {
                continue;
            }
            self.save_cube_entities(pos, here);
        }
        // An entity that strayed outside the cubes it was listed for keeps
        // its own cube's record.
        for (cy, here) in groups {
            self.add_cube_entities(SectionPos::new(x, cy, z), here);
        }
    }

    /// Saves column (x, z)'s `entities` (vanilla entity NBT) into the records
    /// of their cubes, by `Pos[1]`. The list stands for the column's cubes in
    /// the window: those cubes lose entities not in it, and cubes outside the
    /// window (that the list reaches only with an entity that strayed there)
    /// keep what they had. Call it before the window moves.
    pub fn save_entities(&self, x: i32, z: i32, entities: Vec<Tag>) {
        let window = self.window();
        let mut groups = split_by_cube(entities);
        let base = window.min_y() >> 4;
        let inside = base..base + window.sections() as i32;
        let cubes: Vec<i32> = inside
            .clone()
            .chain(groups.keys().copied().filter(|cy| !inside.contains(cy)))
            .collect();
        for cy in cubes {
            let here = groups.remove(&cy).unwrap_or_default();
            let pos = SectionPos::new(x, cy, z);
            let known = self.with_entities.lock().unwrap().contains(&pos);
            if here.is_empty() && !known {
                continue;
            }
            self.save_cube_entities(pos, here);
        }
    }

    /// Replaces the entities saved in cube `pos` with `entities` (those
    /// whose position is in it): the record's blocks stay. A cube with no
    /// record gets one of its generated blocks (air inside vanilla's range).
    pub fn save_cube_entities(&self, pos: SectionPos, entities: Vec<Tag>) {
        let ctx = self.codec();
        let _edit = self.edit.lock().unwrap();
        let mut root = match self.read(pos) {
            Some(root) => root,
            None if entities.is_empty() => {
                self.note_entities(pos, false);
                return;
            }
            None => self.blank_record(pos, &ctx),
        };
        self.note_entities(pos, !entities.is_empty());
        if record_entities(&root) == entities {
            return;
        }
        set_entities(&mut root, entities);
        self.write_later(pos, root);
    }

    /// Adds `entities` to the ones saved in cube `pos` (entities a column of
    /// an older save kept in its entity file).
    fn add_cube_entities(&self, pos: SectionPos, entities: Vec<Tag>) {
        let _edit = self.edit.lock().unwrap();
        let ctx = self.codec();
        let mut root = self
            .read(pos)
            .unwrap_or_else(|| self.blank_record(pos, &ctx));
        let mut all = record_entities(&root);
        all.extend(entities);
        self.note_entities(pos, !all.is_empty());
        set_entities(&mut root, all);
        self.write_later(pos, root);
    }

    /// A record of cube `pos` as the generator makes it.
    fn blank_record(&self, pos: SectionPos, ctx: &Codec) -> NamedTag {
        let section = if self.is_vanilla(pos.y) {
            Arc::new(Section::default())
        } else {
            self.terrain_section(pos).0
        };
        let mut one = Chunk::with_layout(pos.x, pos.z, ColumnLayout::window(pos.y * 16, 16));
        one.sections[0] = section;
        ctx.encode(&one, 0, false, &[])
    }

    /// Moves the entities a column of an older save kept in its entity file
    /// into its cubes (the caller empties the file).
    pub fn adopt_legacy_entities(&self, x: i32, z: i32, entities: Vec<Tag>) {
        for (cy, here) in split_by_cube(entities) {
            self.add_cube_entities(SectionPos::new(x, cy, z), here);
        }
    }

    /// Saves the cubes of `chunks` (window columns) that changed since they
    /// were loaded, generated or last saved, or that hold block entities.
    /// Entities are saved apart (`save_entities`).
    pub fn save_columns(
        &self,
        chunks: &[Arc<Chunk>],
        registry: &'static BlockRegistry,
        biomes: &Biomes,
        tables: Option<&SaveTables>,
        wait: bool,
    ) -> usize {
        let ctx = Codec {
            registry,
            biomes,
            tables,
        };
        let mut written = 0;
        for chunk in chunks {
            let base = chunk.min_y() >> 4;
            for s in 0..chunk.sections.len() {
                if self.save_cube_in(chunk, base + s as i32, &ctx) {
                    written += 1;
                }
            }
        }
        if wait {
            self.flush();
        }
        written
    }

    /// Saves cube `cy` of `chunk` (a column holding it) when its blocks or
    /// ticks changed since it was loaded, generated or last saved, or when
    /// it holds block entities; whether it did. The record's entities stay.
    /// For a window moving in place: the cubes leaving a loaded column go
    /// through here (and `save_cube_entities` with their entities).
    pub fn save_cube(&self, chunk: &Chunk, cy: i32) -> bool {
        self.save_cube_in(chunk, cy, &self.codec())
    }

    fn save_cube_in(&self, chunk: &Chunk, cy: i32, ctx: &Codec) -> bool {
        let s = (cy - (chunk.min_y() >> 4)) as usize;
        let Some(section) = chunk.sections.get(s) else {
            return false;
        };
        let pos = SectionPos::new(chunk.x, cy, chunk.z);
        let low = cy * 16;
        let has_block_entities = chunk
            .block_entities
            .iter()
            .any(|(p, _)| p.1 >= low && p.1 < low + 16);
        let (block_ticks, fluid_ticks) = cube_ticks(chunk, cy);
        let hash = cube_hash(section, &block_ticks, &fluid_ticks);
        let changed = self.clean.lock().unwrap().get(&pos) != Some(&hash);
        if !changed && !has_block_entities {
            return false;
        }
        let marker = self.is_marker(cy);
        let _edit = self.edit.lock().unwrap();
        let known = self.with_entities.lock().unwrap().contains(&pos);
        let old = (marker || known).then(|| self.read(pos)).flatten();
        let entities = old.as_ref().map(record_entities).unwrap_or_default();
        let mut root = ctx.encode(chunk, s, marker, &entities);
        if marker
            && let Some(old) = &old
            && let Some(surface) = old.tag.get(SURFACE_KEY).cloned()
            && let Tag::Compound(map) = &mut root.tag
        {
            map.insert(SURFACE_KEY.into(), surface);
        }
        self.write_later(pos, root);
        self.clean.lock().unwrap().insert(pos, hash);
        true
    }
}

impl Drop for CubicLevel {
    fn drop(&mut self) {
        self.send.lock().unwrap().take();
    }
}

/// The writer thread: cubes to disk in the order they were queued, until
/// the level is gone and its queue empty.
fn write_cubes(io: &CubeIo, receive: Receiver<(SectionPos, Arc<NamedTag>)>) {
    // While idle, the store's uniform cubes (kept in memory and written in
    // batches) are written too, so a crash loses at most a second of them.
    loop {
        match receive.recv_timeout(std::time::Duration::from_secs(1)) {
            Ok((pos, tag)) => io.write_now(pos, &tag),
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                if let Err(e) = io.store.flush_uniform() {
                    log::warn!("Horizon: uniform cubes not saved: {e}");
                }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
    if let Err(e) = io.store.sync() {
        log::warn!("Horizon: cubes not synced: {e}");
    }
}

/// What a cube record is encoded and decoded with.
struct Codec<'a> {
    registry: &'a BlockRegistry,
    biomes: &'a Biomes,
    tables: Option<&'a SaveTables>,
}

impl Codec<'_> {
    /// Section `s` of `chunk` as a cube record: a vanilla chunk record with
    /// that one section (`Y` an int past a byte), its block entities, and
    /// with `column` the column's structure starts and references.
    fn encode(&self, chunk: &Chunk, s: usize, column: bool, entities: &[Tag]) -> NamedTag {
        let cy = (chunk.min_y() >> 4) + s as i32;
        let low = cy * 16;
        let mut one = Chunk::with_layout(chunk.x, chunk.z, ColumnLayout::window(low, 16));
        one.sections[0] = chunk.sections[s].clone();
        let inside = |y: i32| y >= low && y < low + 16;
        one.block_entities = chunk
            .block_entities
            .iter()
            .filter(|(p, _)| inside(p.1))
            .cloned()
            .collect();
        one.block_names = chunk
            .block_names
            .iter()
            .filter(|(p, _)| inside(p.1))
            .cloned()
            .collect();
        if column {
            one.structure_refs = chunk.structure_refs.clone();
            one.structure_starts = chunk.structure_starts.clone();
        }
        let (block_ticks, fluid_ticks) = cube_ticks(chunk, cy);
        if !block_ticks.is_empty() || !fluid_ticks.is_empty() {
            one.saved = Some(Arc::new(ChunkExtras {
                block_ticks,
                fluid_ticks,
                ..ChunkExtras::default()
            }));
        }
        // Light is made again when a cube loads.
        one.lit = false;
        let shape = LevelShape::cubic_window(LevelShape::OVERWORLD, low, 16);
        let mut root = encode_with(&one, self.registry, self.biomes, self.tables, shape);
        set_entities(&mut root, entities.to_vec());
        root
    }

    /// A cube record of cube Y `cy` as a one-section column.
    fn decode(&self, root: &NamedTag, cy: i32) -> Option<Chunk> {
        decode_with(
            root,
            self.registry,
            self.biomes,
            self.tables,
            ColumnLayout::window(cy * 16, 16),
        )
    }
}

/// A hash of the settings text a level generates with (the far terrain cache
/// is keyed by it).
fn settings_hash(text: &str) -> u64 {
    text.bytes().fold(0xCBF2_9CE4_8422_2325u64, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01B3)
    })
}

/// The vanilla surface heights a marker cube keeps (256, `z << 4 | x`).
fn surface_heights(root: &NamedTag) -> Option<Vec<i32>> {
    match root.tag.get(SURFACE_KEY)? {
        Tag::IntArray(v) if v.len() == 256 => Some(v.clone()),
        _ => None,
    }
}

/// The columns whose sky is cut off above `window`: a block of the
/// generated surface (`surface`, one above the highest block) is above the
/// window's top. Below vanilla's range every column is covered.
fn sky_covered(surface: Option<&[i32]>, window: ColumnLayout) -> Option<Arc<[u64; 4]>> {
    let top = window.max_y();
    let mut bits = [0u64; 4];
    let mut any = false;
    for i in 0..256 {
        let covered = match surface {
            Some(heights) => heights[i] > top,
            None => top <= VANILLA_CUBES.start() * 16,
        };
        if covered {
            bits[i / 64] |= 1 << (i % 64);
            any = true;
        }
    }
    any.then(|| Arc::new(bits))
}

/// The entities a cube record holds.
fn record_entities(root: &NamedTag) -> Vec<Tag> {
    match root.tag.get(ENTITIES_KEY) {
        Some(Tag::List(list)) => list.clone(),
        _ => Vec::new(),
    }
}

/// Puts `entities` in a cube record, or takes the key out for none.
fn set_entities(root: &mut NamedTag, entities: Vec<Tag>) {
    if let Tag::Compound(map) = &mut root.tag {
        if entities.is_empty() {
            map.remove(&ENTITIES_KEY.into());
        } else {
            map.insert(ENTITIES_KEY.into(), Tag::List(entities));
        }
    }
}

/// The cube Y of the cube an entity is in; one without a position is
/// dropped (it could not be placed).
fn entity_cube(entity: &Tag) -> Option<i32> {
    let y = entity_y(entity)?;
    if !y.is_finite() {
        return None;
    }
    let cube = (y.floor() as i64).div_euclid(16);
    Some(cube.clamp(i64::from(MIN_CUBE_Y), i64::from(MAX_CUBE_Y)) as i32)
}

/// A column's entities by the cube Y they are in.
pub(crate) fn split_by_cube(entities: Vec<Tag>) -> BTreeMap<i32, Vec<Tag>> {
    let mut cubes: BTreeMap<i32, Vec<Tag>> = BTreeMap::new();
    for entity in entities {
        match entity_cube(&entity) {
            Some(cy) => cubes.entry(cy).or_default().push(entity),
            None => log::warn!("Horizon: an entity without a position was not saved"),
        }
    }
    cubes
}

/// The ticks of `chunk` at positions in cube `cy`: block, fluid.
fn cube_ticks(chunk: &Chunk, cy: i32) -> (Vec<SavedTick>, Vec<SavedTick>) {
    let Some(saved) = chunk.saved.as_deref() else {
        return (Vec::new(), Vec::new());
    };
    let here = |ticks: &[SavedTick]| {
        ticks
            .iter()
            .filter(|t| t.pos.y >> 4 == cy)
            .copied()
            .collect::<Vec<_>>()
    };
    (here(&saved.block_ticks), here(&saved.fluid_ticks))
}

/// Whether a cube record is one block state and one biome throughout with
/// nothing else in it: what the store keeps in its uniform sidecar.
pub(crate) fn is_uniform_record(root: &NamedTag) -> bool {
    let empty = |key: &str| match root.tag.get(key) {
        None => true,
        Some(Tag::List(list)) => list.is_empty(),
        _ => false,
    };
    if !empty("block_entities")
        || !empty("block_ticks")
        || !empty("fluid_ticks")
        || root.tag.get(ENTITIES_KEY).is_some()
        || root.tag.get(SURFACE_KEY).is_some()
    {
        return false;
    }
    let structures_empty = match root.tag.get("structures") {
        None => true,
        Some(s) => ["References", "starts"].iter().all(|k| match s.get(k) {
            None => true,
            Some(Tag::Compound(c)) => c.is_empty(),
            _ => false,
        }),
    };
    let one_state = |container: Option<&Tag>| match container {
        Some(c) => {
            c.get("data").is_none()
                && matches!(c.get("palette"), Some(Tag::List(p)) if p.len() == 1)
        }
        None => false,
    };
    let Some(Tag::List(sections)) = root.tag.get("sections") else {
        return false;
    };
    let [section] = &sections[..] else {
        return false;
    };
    structures_empty
        && matches!(section, Tag::Compound(c) if c.keys().all(|k| {
            matches!(k.to_string_lossy().as_str(), "Y" | "block_states" | "biomes")
        }))
        && one_state(section.get("block_states"))
        && one_state(section.get("biomes"))
}

/// An entity tag's Y (`Pos[1]`).
fn entity_y(entity: &Tag) -> Option<f64> {
    match entity.get("Pos")? {
        Tag::List(pos) => match pos.get(1)? {
            Tag::Double(bits) => Some(f64::from_bits(*bits)),
            Tag::Float(bits) => Some(f64::from(f32::from_bits(*bits))),
            _ => None,
        },
        _ => None,
    }
}

/// A hash of a cube's block states and the ticks saved in it.
fn cube_hash(section: &Section, block_ticks: &[SavedTick], fluid_ticks: &[SavedTick]) -> u64 {
    let mut hash = blocks_hash(section);
    let mut mix = |v: u64| hash = (hash ^ v).wrapping_mul(0x0100_0000_01B3);
    for (n, ticks) in [block_ticks, fluid_ticks].into_iter().enumerate() {
        mix(0x51 + n as u64);
        for t in ticks {
            mix(u64::from(t.kind.0));
            mix(t.pos.x as u32 as u64);
            mix(t.pos.y as u32 as u64);
            mix(t.pos.z as u32 as u64);
            mix(t.delay as u32 as u64);
            mix(t.priority as i32 as u32 as u64);
        }
    }
    hash
}

/// A hash of a section's block states.
fn blocks_hash(section: &Section) -> u64 {
    if let Some(v) = section.blocks.uniform() {
        return 0x9E37_79B9_7F4A_7C15 ^ u64::from(v);
    }
    let mut values = [0u32; 4096];
    section.unpack(&mut values);
    let mut hash = 0xCBF2_9CE4_8422_2325u64;
    for v in values {
        hash = (hash ^ u64::from(v)).wrapping_mul(0x0100_0000_01B3);
    }
    hash
}

/// The vanilla generator (with its saves) inside a cubic level's window.
pub struct CubicTerrain<G> {
    pub inner: Arc<Persistent<G>>,
    pub level: Arc<CubicLevel>,
    /// The window of the shard this terrain loads columns into.
    pub window: ShardWindow,
}

impl<G: ChunkGenerator> CubicTerrain<G> {
    fn window(&self) -> ColumnLayout {
        *self.window.lock().unwrap()
    }

    fn codec(&self) -> Codec<'_> {
        let tables = self
            .inner
            .save
            .tables
            .get_or_init(|| Arc::new(SaveTables::new(&self.inner.registry)));
        self.level.set_tables(tables.clone());
        Codec {
            registry: self.inner.registry.registry,
            biomes: &self.inner.biomes,
            tables: Some(tables.as_ref()),
        }
    }

    /// A vanilla-range column (the proto's, or its blocks) as the window's.
    fn window_of(&self, vanilla: &Chunk) -> Chunk {
        let window = self.window();
        let mut chunk = Chunk::with_layout(vanilla.x, vanilla.z, window);
        let base = window.min_y() >> 4;
        for s in 0..window.sections() {
            let cy = base + s as i32;
            if self.level.is_vanilla(cy) {
                if let Some(i) = vanilla.section_index(cy * 16) {
                    chunk.sections[s] = vanilla.sections[i].clone();
                }
            } else {
                chunk.sections[s] = self
                    .level
                    .terrain(SectionPos::new(vanilla.x, cy, vanilla.z));
            }
        }
        chunk
    }
}

impl<G: ChunkGenerator> ChunkGenerator for CubicTerrain<G> {
    type Worker = G::Worker;
    fn worker(&self) -> G::Worker {
        self.inner.worker()
    }
    /// The vanilla generator's far terrain (a Hybrid world's band of the cubic
    /// far terrain, `cubic_lod`).
    fn lod_sampler(self: Arc<Self>) -> Box<dyn crate::streaming::LodSampler> {
        ChunkGenerator::lod_sampler(self.inner.clone())
    }
    fn lod_surface(&self, worker: &mut G::Worker, x: i32, z: i32) -> Option<crate::lod::Surface> {
        self.inner.lod_surface(worker, x, z)
    }
    fn load(&self, x: i32, z: i32) -> Option<Chunk> {
        let window = self.window();
        // A window that doesn't reach vanilla's range needs no vanilla
        // column: its cubes are saved ones and the generator's.
        let base = window.min_y() >> 4;
        let reaches_vanilla = VANILLA_CUBES.clone().any(|cy| {
            self.level.is_vanilla(cy) && (base..base + window.sections() as i32).contains(&cy)
        });
        if reaches_vanilla && !self.level.has_vanilla(x, z) {
            return None;
        }
        // Entities a column of an older save kept in its entity file move
        // into their cubes first.
        let legacy = self.inner.save.load_entities(x, z);
        if !legacy.is_empty() {
            self.level.adopt_legacy_entities(x, z, legacy);
            self.inner.save.clear_entity_file(x, z);
        }
        Some(self.level.assemble(x, z, window, &self.codec()))
    }
    fn prepare_loaded(&self, worker: &mut G::Worker, chunk: &Chunk) {
        self.inner.prepare_loaded(worker, chunk);
    }
    fn idle(&self, worker: &mut G::Worker) {
        self.inner.idle(worker);
    }
    fn all_idle(&self) {
        self.inner.all_idle();
    }
    fn proto(&self, worker: &mut G::Worker, x: i32, z: i32) -> Proto {
        self.inner.proto(worker, x, z)
    }
    fn decorates(&self) -> bool {
        self.inner.decorates()
    }
    fn decorate(&self, worker: &mut G::Worker, region: &mut [Proto; 9]) {
        self.inner.decorate(worker, region);
    }
    fn finish(&self, proto: Proto) -> Chunk {
        let vanilla = self.inner.finish(proto);
        self.level
            .finish_vanilla(vanilla, self.window(), &self.codec())
    }
    fn is_decorated(&self, proto: &Proto) -> bool {
        self.inner.is_decorated(proto)
    }
    fn light_view(&self, proto: &Proto) -> Option<Chunk> {
        let vanilla = self.inner.light_view(proto)?;
        Some(self.window_of(&vanilla))
    }
    fn non_air_sections(&self, proto: &Proto) -> Option<crate::section_mask::SectionMask> {
        let view = self.light_view(proto)?;
        let mut mask = crate::section_mask::SectionMask::default();
        for (s, section) in view.sections.iter().enumerate() {
            if !section.is_uniform(0) {
                mask.insert(s);
            }
        }
        Some(mask)
    }
    fn cache_census(&self) -> Vec<(&'static str, usize, Option<usize>)> {
        self.inner.cache_census()
    }
    fn proto_heap_bytes(&self, proto: &Proto) -> ([usize; 6], [usize; 16]) {
        self.inner.proto_heap_bytes(proto)
    }
    fn save_protos(&self, protos: &[&Proto]) {
        self.inner.save_protos(protos);
    }
    fn save_finished(&self, chunk: Chunk) {
        self.inner.save_finished(chunk);
    }
}

/// The file in a world's folder that makes its Overworld cubic, with the
/// window the Overworld's chunks held when it was last saved.
const WORLD_FILE: &str = "horizon_cubic.json";

/// Whether the world in `world_root` is cubic (its Overworld).
pub fn is_cubic_world(world_root: &Path) -> bool {
    world_root.join(WORLD_FILE).is_file()
}

/// Makes the world in `world_root` cubic, its Overworld opening on its
/// first window. Done once, when the world is created.
pub fn mark_cubic_world(world_root: &Path) -> std::io::Result<()> {
    save_window(
        world_root,
        FIRST_WINDOW_MIN_Y,
        window_cubes(i32::from(DEFAULT_VERTICAL)) * 16,
    )
}

/// The bottom and height (blocks) of the window a cubic world's Overworld
/// last held. A save from before windows had a height of their own has 512.
pub fn saved_window(world_root: &Path) -> Option<(i32, i32)> {
    let text = std::fs::read_to_string(world_root.join(WORLD_FILE)).ok()?;
    let json: serde_json::Value = serde_json::from_str(&text).ok()?;
    let min_y = json["window_min_y"].as_i64()? as i32;
    let height = json["window_height"]
        .as_i64()
        .map_or(512, |h| h as i32)
        .max(16);
    Some((min_y, height))
}

/// Records the bottom and height (blocks) of the window a cubic world's
/// Overworld holds. The world type Create World chose stays (`cubic_settings`).
pub fn save_window(world_root: &Path, min_y: i32, height: i32) -> std::io::Result<()> {
    let mut json = serde_json::json!({
        "cubic": true,
        "window_min_y": min_y,
        "window_height": height,
    });
    if let Some(kind) = crate::cubic_settings::stored_type(world_root) {
        json["type"] = kind.into();
    }
    let temporary = world_root.join(format!("{WORLD_FILE}.tmp"));
    std::fs::write(&temporary, json.to_string())?;
    std::fs::rename(temporary, world_root.join(WORLD_FILE))
}

/// `shapes` with a cubic world's Overworld as its saved window (a flat or
/// debug world's never is).
pub fn cubic_shapes(
    world_root: &Path,
    mut shapes: crate::chunk::LevelShapes,
) -> crate::chunk::LevelShapes {
    use crate::save::{WorldGen, WorldPreset};
    if !is_cubic_world(world_root)
        || WorldGen::read(world_root)
            .is_some_and(|w| matches!(w.preset, WorldPreset::Flat | WorldPreset::Debug))
    {
        return shapes;
    }
    let (min_y, height) = saved_window(world_root).unwrap_or((
        FIRST_WINDOW_MIN_Y,
        window_cubes(i32::from(DEFAULT_VERTICAL)) * 16,
    ));
    shapes.overworld = LevelShape::cubic_window(shapes.overworld, min_y, height);
    shapes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_window_is_two_v_plus_three_cubes_around_the_player() {
        assert_eq!(window_cubes(8), 19);
        assert_eq!(wanted_cubes(4, 8), (-5, 13));
        let level = LevelShape::OVERWORLD;
        let window = window_of_cubes(level, wanted_cubes(4, 8));
        assert!(window.cubic);
        assert_eq!((window.min_y, window.height), (-80, 304));
        assert_eq!(window_cube_range(window), (-5, 13));
        // A far window keeps its exact cubes.
        let far = window_of_cubes(level, wanted_cubes(62_000, 32));
        assert_eq!(window_cube_range(far), (61_967, 62_033));
        // The vertical distance is kept to what the setting allows.
        assert_eq!(wanted_cubes(0, 0), wanted_cubes(0, 2));
        assert_eq!(wanted_cubes(0, 200), wanted_cubes(0, 32));
    }

    #[test]
    fn a_window_recentres_two_cubes_from_the_middle_or_when_v_changes() {
        let window = window_of_cubes(LevelShape::OVERWORLD, wanted_cubes(10, 8));
        assert!(!needs_new_window(window, wanted_cubes(10, 8)));
        assert!(!needs_new_window(window, wanted_cubes(11, 8)));
        assert!(!needs_new_window(window, wanted_cubes(9, 8)));
        assert!(needs_new_window(window, wanted_cubes(12, 8)));
        assert!(needs_new_window(window, wanted_cubes(8, 8)));
        assert!(needs_new_window(window, wanted_cubes(10, 9)), "V changed");
        assert!(!needs_new_window(
            LevelShape::OVERWORLD,
            wanted_cubes(10, 8)
        ));
    }

    #[test]
    fn several_players_share_the_union_of_their_ranges_up_to_99_cubes() {
        assert_eq!(union_cubes([]), None);
        assert_eq!(
            union_cubes([wanted_cubes(0, 8), wanted_cubes(5, 8)]),
            Some((-9, 14))
        );
        // Far apart: capped at 99 cubes around the middle of the union.
        let (lo, hi) = union_cubes([wanted_cubes(0, 8), wanted_cubes(500, 8)]).unwrap();
        assert_eq!(hi - lo + 1, MAX_WINDOW_CUBES);
        assert!(lo < 250 && hi > 250);
        // Kept in the cubic range.
        let (lo, hi) = union_cubes([wanted_cubes(horizon_world::cubic::MAX_CUBE_Y, 8)]).unwrap();
        assert_eq!(hi, horizon_world::cubic::MAX_CUBE_Y);
        assert_eq!(lo, hi - 18);
    }

    #[test]
    fn a_slide_moves_each_end_by_the_ring_at_most() {
        assert_eq!(next_window((0, 18), (1, 19)), (1, 19));
        assert_eq!(next_window((0, 18), (2, 20)), (2, 20));
        assert_eq!(next_window((0, 18), (5, 23)), (2, 20), "two layers a step");
        assert_eq!(next_window((0, 18), (-5, 13)), (-2, 16));
        // The window growing: its ends move apart by the ring.
        assert_eq!(next_window((0, 18), (-10, 28)), (-2, 20));
        // No overlap: a jump.
        assert_eq!(next_window((0, 18), (100, 118)), (100, 118));
    }

    #[test]
    fn a_slide_enters_and_leaves_whole_layers() {
        let runs = |ranges: Vec<std::ops::RangeInclusive<i32>>| -> Vec<(i32, i32)> {
            ranges.iter().map(|r| (*r.start(), *r.end())).collect()
        };
        assert_eq!(entering_cubes((0, 18), (2, 20)), [19, 20]);
        assert_eq!(runs(leaving_ranges((0, 18), (2, 20))), [(0, 1)]);
        assert_eq!(entering_cubes((0, 18), (-2, 16)), [-2, -1]);
        assert_eq!(runs(leaving_ranges((0, 18), (-2, 16))), [(17, 18)]);
        // Growing keeps everything; shrinking leaves both ends.
        assert_eq!(entering_cubes((0, 18), (-2, 20)), [-2, -1, 19, 20]);
        assert!(leaving_ranges((0, 18), (-2, 20)).is_empty());
        assert_eq!(runs(leaving_ranges((0, 18), (3, 15))), [(0, 2), (16, 18)]);
        // A jump: the whole old window leaves once, the whole new one enters.
        assert_eq!(runs(leaving_ranges((0, 18), (100, 118))), [(0, 18)]);
        assert_eq!(runs(leaving_ranges((0, 18), (-100, -82))), [(0, 18)]);
        assert_eq!(entering_cubes((0, 18), (100, 118)).len(), 19);
    }

    #[test]
    fn sky_is_covered_under_the_surface_only() {
        let window = ColumnLayout::window(-192, 512);
        let mut heights = vec![64; 256];
        heights[5] = 400;
        let covered = sky_covered(Some(&heights), window).unwrap();
        assert_eq!(covered[0], 1 << 5);
        assert!(sky_covered(Some(&heights), ColumnLayout::window(512, 512)).is_none());
        // Deep below vanilla's range every column is covered, known or not.
        let deep = sky_covered(None, ColumnLayout::window(-10_000, 512)).unwrap();
        assert_eq!(*deep, [u64::MAX; 4]);
        assert!(sky_covered(None, ColumnLayout::window(10_000, 512)).is_none());
    }

    #[test]
    fn entity_heights_read_from_pos() {
        let entity = Tag::compound([(
            "Pos",
            Tag::List(vec![
                Tag::Double(1.5f64.to_bits()),
                Tag::Double(70.25f64.to_bits()),
                Tag::Double(0.0f64.to_bits()),
            ]),
        )]);
        assert_eq!(entity_y(&entity), Some(70.25));
        assert_eq!(entity_y(&Tag::compound([("id", Tag::Int(1))])), None);
    }

    #[test]
    fn a_world_is_cubic_by_its_file() {
        let dir = std::env::temp_dir().join(format!("horizon-cubic-world-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        assert!(!is_cubic_world(&dir));
        mark_cubic_world(&dir).unwrap();
        assert!(is_cubic_world(&dir));
        assert_eq!(saved_window(&dir), Some((FIRST_WINDOW_MIN_Y, 304)));
        save_window(&dir, 4096, 160).unwrap();
        assert_eq!(saved_window(&dir), Some((4096, 160)));
        let shapes = cubic_shapes(&dir, crate::chunk::LevelShapes::VANILLA);
        assert!(shapes.overworld.cubic && !shapes.nether.cubic);
        assert_eq!(
            (shapes.overworld.min_y, shapes.overworld.height),
            (4096, 160)
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The game's assets, tables and a scratch level folder, when the test
    /// resources are there (`HORIZON_TEST_RESOURCES`).
    struct Fixture {
        assets: crate::assets::Assets,
        biomes: Arc<Biomes>,
        tables: Arc<SaveTables>,
        dir: std::path::PathBuf,
    }
    impl Fixture {
        fn new(name: &str) -> Option<Self> {
            let root = std::path::PathBuf::from(std::env::var_os("HORIZON_TEST_RESOURCES")?);
            let assets = crate::assets::Assets::load(&root).expect("assets");
            let biomes = Arc::new(Biomes::load(&root).expect("biomes"));
            let tables = Arc::new(SaveTables::new(&assets));
            let dir = std::env::temp_dir().join(format!(
                "horizon-cubic-{name}-{}-{:?}",
                std::process::id(),
                std::thread::current().id()
            ));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            Some(Self {
                assets,
                biomes,
                tables,
                dir,
            })
        }
        fn level(&self) -> Arc<CubicLevel> {
            self.level_of(CubicKind::Hybrid)
        }
        fn level_of(&self, kind: CubicKind) -> Arc<CubicLevel> {
            let window = LevelShape::cubic_window(LevelShape::OVERWORLD, -192, 512);
            let generation = CubicGeneration {
                kind,
                settings_json: String::new(),
            };
            let level = CubicLevel::new(
                &self.dir,
                1,
                self.assets.registry,
                &self.biomes,
                window,
                &generation,
                None,
            )
            .expect("level");
            level.set_tables(self.tables.clone());
            level
        }
        fn state(&self, name: &str) -> u32 {
            self.assets.registry.default_state(name).unwrap().0
        }
        fn codec(&self) -> Codec<'_> {
            Codec {
                registry: self.assets.registry,
                biomes: &self.biomes,
                tables: Some(&self.tables),
            }
        }
        fn uniform(&self, name: &str) -> Arc<Section> {
            Arc::new(Section {
                blocks: PalettedContainer::new(block_strategy(), self.state(name)),
                biomes: PalettedContainer::new(
                    crate::chunk::BIOME_STRATEGY,
                    u32::from(self.biomes.id("minecraft:plains")),
                ),
                ..Section::default()
            })
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    fn mob(id: i32, y: f64) -> Tag {
        Tag::compound([
            ("id", Tag::String("minecraft:pig".into())),
            ("Tag", Tag::Int(id)),
            (
                "Pos",
                Tag::List(vec![
                    Tag::Double(0.5f64.to_bits()),
                    Tag::Double(y.to_bits()),
                    Tag::Double(0.5f64.to_bits()),
                ]),
            ),
        ])
    }

    fn tick(kind: u32, x: i32, y: i32, z: i32, delay: i32) -> SavedTick {
        SavedTick {
            kind: horizon_core::ticks::TickTypeId(kind),
            pos: horizon_core::position::BlockPos::new(x, y, z),
            delay,
            priority: horizon_core::ticks::TickPriority::Normal,
        }
    }

    #[test]
    fn entities_and_ticks_round_trip_through_a_cube_record() {
        let Some(f) = Fixture::new("record") else {
            return;
        };
        let ctx = f.codec();
        let window = ColumnLayout::window(-192, 512);
        let mut chunk = Chunk::with_layout(3, -2, window);
        chunk.set(3, 70, 4, f.state("minecraft:stone"));
        chunk.saved = Some(Arc::new(ChunkExtras {
            block_ticks: vec![
                tick(0, 3 * 16 + 3, 70, -2 * 16 + 4, 7),
                tick(0, 48, 200, -32, 1),
            ],
            fluid_ticks: vec![tick(2, 50, 71, -30, 3)],
            ..ChunkExtras::default()
        }));
        let entities = vec![mob(1, 70.5), mob(2, 79.9)];
        let s = (4 - (window.min_y() >> 4)) as usize;
        let root = ctx.encode(&chunk, s, false, &entities);
        assert!(!is_uniform_record(&root));
        assert_eq!(record_entities(&root), entities);
        let cube = ctx.decode(&root, 4).expect("decodes");
        let saved = cube.saved.as_deref().expect("extras");
        // Only the ticks in this cube are in its record.
        assert_eq!(saved.block_ticks, [tick(0, 51, 70, -28, 7)]);
        assert_eq!(saved.fluid_ticks, [tick(2, 50, 71, -30, 3)]);
        // Without entities the key is gone.
        let bare = ctx.encode(&chunk, s, false, &[]);
        assert!(bare.tag.get(ENTITIES_KEY).is_none());
    }

    #[test]
    fn only_bare_uniform_cubes_are_uniform_records() {
        let Some(f) = Fixture::new("uniform") else {
            return;
        };
        let ctx = f.codec();
        let window = ColumnLayout::window(-192, 512);
        let mut chunk = Chunk::with_layout(0, 0, window);
        chunk.sections[3] = f.uniform("minecraft:deepslate");
        let root = ctx.encode(&chunk, 3, false, &[]);
        assert!(is_uniform_record(&root));
        // The store takes it and gives the same record back.
        let store = CubeStore::new(&f.dir);
        let pos = SectionPos::new(0, -9, 0);
        let root = {
            let mut one = Chunk::with_layout(0, 0, ColumnLayout::window(-9 * 16, 16));
            one.sections[0] = chunk.sections[3].clone();
            ctx.encode(&one, 0, false, &[])
        };
        store.write_uniform(pos, &root).expect("fits the sidecar");
        assert_eq!(store.read(pos).unwrap(), Some(root.clone()));
        // Entities, ticks or a block entity make it a full record.
        assert!(!is_uniform_record(&ctx.encode(
            &chunk,
            3,
            false,
            &[mob(1, -140.0)]
        )));
        let mut ticking = chunk.clone();
        ticking.saved = Some(Arc::new(ChunkExtras {
            fluid_ticks: vec![tick(2, 1, -140, 1, 3)],
            ..ChunkExtras::default()
        }));
        assert!(!is_uniform_record(&ctx.encode(&ticking, 3, false, &[])));
        // A cube of mixed blocks is not uniform.
        chunk.sections[3] = Arc::new(Section::default());
        chunk.set(1, -140, 1, f.state("minecraft:stone"));
        assert!(!is_uniform_record(&ctx.encode(&chunk, 3, false, &[])));
    }

    #[test]
    fn uniform_cubes_take_no_region_sector() {
        let Some(f) = Fixture::new("sectors") else {
            return;
        };
        let level = f.level();
        let window = level.window();
        let mut chunk = Chunk::with_layout(4, 5, window);
        for s in 0..8 {
            chunk.sections[s] = f.uniform("minecraft:deepslate");
        }
        chunk.set(1, -100, 1, f.state("minecraft:stone"));
        let written = level.save_columns(
            &[Arc::new(chunk)],
            f.assets.registry,
            &f.biomes,
            Some(&f.tables),
            true,
        );
        assert!(written >= 8);
        let base = window.min_y() >> 4;
        let store = &level.io.store;
        let uniform = SectionPos::new(4, base, 5);
        assert!(store.read(uniform).unwrap().is_some());
        assert!(store.uniform_path(uniform).exists());
        let loaded = level.load_cube(uniform).expect("saved");
        assert_eq!(
            loaded.section.blocks.uniform(),
            Some(f.state("minecraft:deepslate"))
        );
        assert!(!loaded.generated);
        // The one cube with a different block took a sector, the rest did not.
        let mixed = SectionPos::new(4, -7, 5);
        assert!(level.load_cube(mixed).is_some());
        assert!(store.region_path(mixed).exists());
        let sidecar = std::fs::metadata(store.uniform_path(uniform))
            .unwrap()
            .len();
        assert!(sidecar < 2048, "{sidecar} bytes");
    }

    #[test]
    fn a_columns_entities_split_into_cubes_and_come_back() {
        let Some(f) = Fixture::new("entities") else {
            return;
        };
        let level = f.level();
        let window = level.window();
        let (x, z) = (1, 2);
        let entities = vec![mob(1, 70.5), mob(2, 75.0), mob(3, -100.0), mob(4, 1000.0)];
        level.save_entities(x, z, entities.clone());
        level.flush();
        let cube = |cy: i32| level.load_cube(SectionPos::new(x, cy, z)).unwrap();
        assert_eq!(cube(4).entities, [mob(1, 70.5), mob(2, 75.0)]);
        assert_eq!(cube(-7).entities, [mob(3, -100.0)]);
        // Outside the window (cube 62): in its record too.
        assert_eq!(cube(62).entities, [mob(4, 1000.0)]);
        // The column of the window has those inside it.
        let column = level.assemble(x, z, window, &f.codec());
        let mut inside = column.entities.clone();
        inside.sort_by_key(|e| entity_y(e).unwrap() as i64);
        assert_eq!(inside, [mob(3, -100.0), mob(1, 70.5), mob(2, 75.0)]);
        // A later save of the column (one entity gone, one moved down a
        // cube) replaces the window's cubes' entities and leaves cube 62.
        level.save_entities(x, z, vec![mob(1, 70.5), mob(2, 60.0)]);
        level.flush();
        assert_eq!(cube(4).entities, [mob(1, 70.5)]);
        assert_eq!(cube(3).entities, [mob(2, 60.0)]);
        assert!(cube(-7).entities.is_empty());
        assert_eq!(cube(62).entities, [mob(4, 1000.0)]);
        // A new level over the same folder reads them.
        drop(level);
        let again = f.level();
        assert_eq!(
            again.load_cube(SectionPos::new(x, 62, z)).unwrap().entities,
            [mob(4, 1000.0)]
        );
        assert_eq!(
            again
                .assemble(x, z, again.window(), &f.codec())
                .entities
                .len(),
            2
        );
    }

    #[test]
    fn block_saves_and_entity_saves_keep_each_others_data() {
        let Some(f) = Fixture::new("merge") else {
            return;
        };
        let level = f.level();
        let (x, z) = (0, 0);
        level.save_entities(x, z, vec![mob(1, 70.5)]);
        // Load the column, change a block and a tick in the entities' cube,
        // and save the column.
        let mut column = level.assemble(x, z, level.window(), &f.codec());
        assert_eq!(column.entities, [mob(1, 70.5)]);
        column.set(2, 70, 2, f.state("minecraft:stone"));
        column.saved = Some(Arc::new(ChunkExtras {
            block_ticks: vec![tick(0, 2, 70, 2, 5)],
            ..ChunkExtras::default()
        }));
        let written = level.save_columns(
            &[Arc::new(column)],
            f.assets.registry,
            &f.biomes,
            Some(&f.tables),
            true,
        );
        assert!(written >= 1);
        let cube = level.load_cube(SectionPos::new(x, 4, z)).unwrap();
        assert_eq!(cube.entities, [mob(1, 70.5)]);
        assert_eq!(cube.block_ticks, [tick(0, 2, 70, 2, 5)]);
        assert_eq!(
            cube.section.get(((70 & 15) << 8) | (2 << 4) | 2),
            f.state("minecraft:stone")
        );
        // An entity save keeps the blocks and ticks.
        level.save_entities(x, z, vec![mob(1, 71.0), mob(2, 72.0)]);
        level.flush();
        let cube = level.load_cube(SectionPos::new(x, 4, z)).unwrap();
        assert_eq!(cube.entities.len(), 2);
        assert_eq!(cube.block_ticks.len(), 1);
        assert_eq!(
            cube.section.get(((70 & 15) << 8) | (2 << 4) | 2),
            f.state("minecraft:stone")
        );
        // The ticks come back with the column too.
        let column = level.assemble(x, z, level.window(), &f.codec());
        let saved = column.saved.as_deref().expect("ticks");
        assert_eq!(saved.block_ticks, [tick(0, 2, 70, 2, 5)]);
    }

    /// Waits (a few seconds at most) until `ready` holds.
    fn wait_until(ready: impl Fn() -> bool) -> bool {
        for _ in 0..500 {
            if ready() {
                return true;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        ready()
    }

    #[test]
    fn the_ring_has_the_layers_beyond_the_window_ready_and_drops_what_the_store_writes() {
        let Some(f) = Fixture::new("ring") else {
            return;
        };
        let level = f.level();
        // Cubes -12..=19 (the fixture's window): the ring is -14, -13, 20 and 21.
        level.retarget_ring(&[(0, 0)], [(0, 0), (1, 0)]);
        let ring: Vec<SectionPos> = [(0, 0), (1, 0)]
            .into_iter()
            .flat_map(|(x, z)| [-13, -14, 20, 21].map(|cy| SectionPos::new(x, cy, z)))
            .collect();
        assert!(wait_until(|| ring.iter().all(|&p| level.is_prefetched(p))));
        // A ready cube is the one `make_cube` makes. (Population saves the
        // cubes it writes, so it comes back as a record, not `generated`; and
        // the ring may take it up again at once, as its window still wants it.)
        let pos = SectionPos::new(0, 20, 0);
        let cube = level.ready_cube(pos);
        assert!(cube.section.same_as(&level.make_cube(pos).section));
        // The store writing a cube drops it; the ring loads it again, entities and all.
        let above = SectionPos::new(0, 21, 0);
        assert!(level.is_prefetched(above));
        level.save_cube_entities(above, vec![mob(1, 21.0 * 16.0 + 3.0)]);
        assert!(!level.is_prefetched(above) || level.ready_cube(above).entities.len() == 1);
        assert!(wait_until(|| level.is_prefetched(above)));
        assert_eq!(
            level.ready_cube(above).entities,
            [mob(1, 21.0 * 16.0 + 3.0)]
        );
    }

    #[test]
    fn the_ring_leaves_vanilla_cubes_of_columns_without_vanilla_generation_to_the_streaming() {
        let Some(f) = Fixture::new("ring-vanilla") else {
            return;
        };
        let level = f.level();
        // Cubes 0..=18: the ring below, -2 and -1, is vanilla's range; above it 19 is, 20 is not.
        level.set_window(LevelShape::cubic_window(LevelShape::OVERWORLD, 0, 304));
        level.retarget_ring(&[(0, 0)], [(0, 0)]);
        assert!(wait_until(|| level.is_prefetched(SectionPos::new(0, 20, 0))));
        std::thread::sleep(std::time::Duration::from_millis(200));
        for cy in [-1, -2, 19] {
            assert!(
                !level.is_prefetched(SectionPos::new(0, cy, 0)),
                "cube {cy} waits for its column's vanilla generation"
            );
            assert!(!level.can_make_cube(SectionPos::new(0, cy, 0)));
        }
        // Once the column has it, the next aim loads them (air where nothing was saved).
        level.vanilla_done.lock().unwrap().insert((0, 0));
        level.retarget_ring(&[(0, 0)], [(0, 0), (1, 0)]);
        assert!(wait_until(|| {
            [-1, -2, 19]
                .iter()
                .all(|&cy| level.is_prefetched(SectionPos::new(0, cy, 0)))
        }));
        let air = level.ready_cube(SectionPos::new(0, -1, 0));
        assert!(!air.generated && air.section.blocks.uniform() == Some(0));
    }

    /// A window sliding over a column: the sections both windows hold stay (the same
    /// `Arc`s), and the ones that enter, from the ring, are those the column assembled for
    /// the new window has.
    #[test]
    fn a_slide_keeps_the_arcs_of_unchanged_sections_and_enters_what_a_reload_would_hold() {
        let Some(f) = Fixture::new("slide") else {
            return;
        };
        let level = f.level();
        let a = LevelShape::cubic_window(LevelShape::OVERWORLD, 100 * 16, 19 * 16);
        let b = LevelShape::cubic_window(LevelShape::OVERWORLD, 102 * 16, 19 * 16);
        level.set_window(a);
        let ctx = f.codec();
        let before = level.assemble(0, 0, a.column(), &ctx);
        let mut world = crate::world::World::new(horizon_world::registry::BlockStateId(0), 0);
        world.shape = a;
        world.chunks.insert((0, 0), Arc::new(before.clone()));
        level.retarget_ring(&[(0, 0)], [(0, 0)]);
        assert!(wait_until(
            || (119..=120).all(|cy| level.is_prefetched(SectionPos::new(0, cy, 0)))
        ));
        level.set_window(b);
        world.relayout(b, |x, z, cy| {
            level.ready_cube(SectionPos::new(x, cy, z)).section
        });
        let after = &world.chunks[&(0, 0)];
        let reloaded = level.assemble(0, 0, b.column(), &ctx);
        assert_eq!(after.layout(), b.column());
        for (s, section) in after.sections.iter().enumerate() {
            assert!(section.same_as(&reloaded.sections[s]), "section {s}");
            // Cube 102 + s was in window A at index s + 2, up to its top.
            if s + 2 < before.sections.len() {
                assert!(
                    Arc::ptr_eq(section, &before.sections[s + 2]),
                    "section {s} moved"
                );
            }
        }
    }

    /// Two players at y 0 and y 20,000 are two shards over one level: each
    /// column is made from the cubes of its own window, from the same store.
    #[test]
    fn two_windows_over_one_level_each_make_their_own_columns_from_the_same_cubes() {
        let Some(f) = Fixture::new("shards") else {
            return;
        };
        let level = f.level_of(CubicKind::Custom);
        let near = LevelShape::cubic_window(LevelShape::OVERWORLD, -80, 19 * 16);
        let far = LevelShape::cubic_window(LevelShape::OVERWORLD, 20_000 - 144, 19 * 16);
        let (x, z) = (2, 5);
        let ctx = f.codec();
        // The upper shard builds a platform in cube 1250, and the store has it.
        let platform = f.uniform("minecraft:stone");
        let mut cube = Chunk::with_layout(x, z, ColumnLayout::window(1250 * 16, 16));
        cube.sections[0] = platform;
        assert!(level.save_cube(&cube, 1250));
        level.flush();
        let low = level.assemble(x, z, near.column(), &ctx);
        let high = level.assemble(x, z, far.column(), &ctx);
        assert_eq!((low.min_y(), high.min_y()), (-80, 20_000 - 144));
        assert_eq!(low.sections.len(), 19);
        assert_eq!(high.sections.len(), 19);
        // Each holds the cubes of its window and no other: the platform is the
        // upper column's alone, and the generator's terrain the lower's.
        let stone = f.state("minecraft:stone");
        let index = (1250 - (high.min_y() >> 4)) as usize;
        assert_eq!(high.sections[index].blocks.uniform(), Some(stone));
        assert!(
            low.sections.iter().any(|s| !s.is_uniform(0)),
            "terrain near the surface"
        );
        // The same cube, loaded by both layouts that hold it, is the same section.
        let overlap = LevelShape::cubic_window(LevelShape::OVERWORLD, 1250 * 16, 3 * 16);
        let again = level.assemble(x, z, overlap.column(), &ctx);
        assert!(again.sections[0].same_as(&high.sections[index]));
    }

    /// The column's sky data is the level's: whichever window loads the column
    /// is told the same covering, from the same surface.
    #[test]
    fn the_sky_over_a_column_is_one_answer_for_every_window() {
        let Some(f) = Fixture::new("sky") else {
            return;
        };
        let level = f.level_of(CubicKind::Custom);
        let (x, z) = (-3, 8);
        let under = LevelShape::cubic_window(LevelShape::OVERWORLD, -3000, 19 * 16);
        let above = LevelShape::cubic_window(LevelShape::OVERWORLD, 20_000, 19 * 16);
        let a = level.sky_cover(x, z, under.column());
        let b = level.sky_cover(x, z, above.column());
        assert!(a.is_some(), "a window deep under the surface is covered");
        assert!(b.is_none(), "a window above it sees the sky");
        // Asked again, in the other order and from a second view, the same.
        assert_eq!(level.sky_cover(x, z, above.column()), b);
        assert_eq!(level.sky_cover(x, z, under.column()), a);
    }

    /// An entity that leaves one shard's window is saved with its cube, and the
    /// shard whose window holds that cube loads it.
    #[test]
    fn an_entity_leaving_one_shard_is_loaded_by_the_shard_that_holds_its_cube() {
        let Some(f) = Fixture::new("carried") else {
            return;
        };
        let level = f.level_of(CubicKind::Custom);
        let (x, z) = (1, 1);
        let lower = LevelShape::cubic_window(LevelShape::OVERWORLD, 0, 19 * 16);
        let upper = LevelShape::cubic_window(LevelShape::OVERWORLD, 19 * 16, 19 * 16);
        level.set_window(lower);
        // The lower shard saves its entities, one in its top cube (18), and
        // that cube leaves it as the shard closes.
        let carried = mob(7, 18.0 * 16.0 + 4.5);
        level.save_entities(x, z, vec![carried.clone(), mob(8, 20.5)]);
        level.save_leaving_entities(x, z, 0..=18, vec![carried.clone(), mob(8, 20.5)]);
        level.flush();
        // The upper shard's window holds cube 19 and up: it does not have it...
        let theirs = level.assemble(x, z, upper.column(), &f.codec());
        assert!(theirs.entities.is_empty());
        // ...until the entity walks (is saved) into cube 19, which the upper
        // shard's column loads with it.
        let over = mob(7, 19.0 * 16.0 + 0.5);
        level.set_window(upper);
        level.save_entities(x, z, vec![over.clone()]);
        level.flush();
        let theirs = level.assemble(x, z, upper.column(), &f.codec());
        assert_eq!(theirs.entities, [over]);
        // The lower window's column still holds what its own cubes saved.
        let ours = level.assemble(x, z, lower.column(), &f.codec());
        assert_eq!(ours.entities.len(), 2, "{:?}", ours.entities);
    }

    #[test]
    fn entities_of_leaving_cubes_are_saved_there_and_come_back_with_the_window() {
        let Some(f) = Fixture::new("leaving") else {
            return;
        };
        let level = f.level();
        let a = LevelShape::cubic_window(LevelShape::OVERWORLD, 100 * 16, 19 * 16);
        level.set_window(a);
        let (x, z) = (3, -4);
        let (low, high) = (mob(1, 1610.5), mob(2, 1900.0));
        // The window's entities are saved whole; then cubes 100..=101 leave with theirs.
        level.save_entities(x, z, vec![low.clone(), high.clone()]);
        let gone = (100..=101).collect::<Vec<_>>();
        assert_eq!(gone.len(), 2);
        level.save_leaving_entities(x, z, 100..=101, vec![low.clone()]);
        level.flush();
        let cube = |cy: i32| level.load_cube(SectionPos::new(x, cy, z)).unwrap();
        assert_eq!(cube(100).entities, std::slice::from_ref(&low));
        assert_eq!(
            cube(118).entities,
            std::slice::from_ref(&high),
            "the other cubes are untouched"
        );
        // The window slides up: only the leaving cubes' lists were replaced.
        level.set_window(LevelShape::cubic_window(
            LevelShape::OVERWORLD,
            102 * 16,
            19 * 16,
        ));
        level.save_leaving_entities(x, z, 100..=101, Vec::new());
        level.flush();
        assert!(
            cube(100).entities.is_empty(),
            "an empty list empties the cube"
        );
        assert_eq!(cube(118).entities, std::slice::from_ref(&high));
        // Back down: the column holds what the cubes saved.
        level.save_leaving_entities(x, z, 100..=101, vec![low.clone()]);
        level.flush();
        level.set_window(a);
        let column = level.assemble(x, z, a.column(), &f.codec());
        let mut entities = column.entities.clone();
        entities.sort_by_key(|e| entity_y(e).unwrap() as i64);
        assert_eq!(entities, [low, high]);
    }

    #[test]
    fn points_of_interest_come_back_with_the_blocks() {
        let Some(f) = Fixture::new("poi") else {
            return;
        };
        // A cubic level saves no POI: a cube's POIs are those of its blocks.
        let ctx = f.codec();
        let mut one = Chunk::with_layout(0, 0, ColumnLayout::window(64, 16));
        one.set(5, 71, 5, f.state("minecraft:bell"));
        let root = ctx.encode(&one, 0, false, &[]);
        let loaded = ctx.decode(&root, 4).expect("decodes");
        let poi = crate::save::encode_poi(&loaded, &f.tables).expect("the bell is a POI");
        let again = crate::save::encode_poi(&one, &f.tables).unwrap();
        assert_eq!(poi, again);
    }

    /// Every block of a cube, by position in the cube.
    fn blocks_of(cube: &LoadedCube) -> Vec<u32> {
        let mut values = [0u32; 4096];
        cube.section.unpack(&mut values);
        values.to_vec()
    }

    #[test]
    fn custom_columns_load_without_vanilla() {
        let Some(f) = Fixture::new("custom-columns") else {
            return;
        };
        let level = f.level_of(CubicKind::Custom);
        assert!(level.has_vanilla(5, 5), "nothing to wait for");
        assert!(level.can_make_cube(SectionPos::new(5, 4, 5)));
        let column = level.assemble(5, 5, level.window(), &f.codec());
        // The surface (y 64-ish) is in the window: some cube has stone.
        assert!(column.sections.iter().any(|s| !s.is_uniform(0)));
        // Every cube a column holds has its populations behind it.
        for cy in -12..20 {
            for d in 0..8 {
                let p = SectionPos::new(5 - (d & 1), cy - (d >> 1 & 1), 5 - (d >> 2 & 1));
                assert!(level.is_populated(p), "{p:?}");
            }
        }
    }

    #[test]
    fn each_population_runs_once_whatever_the_order() {
        let (Some(a), Some(b)) = (Fixture::new("order-a"), Fixture::new("order-b")) else {
            return;
        };
        let (one, two) = (a.level_of(CubicKind::Custom), b.level_of(CubicKind::Custom));
        let cubes: Vec<SectionPos> = (0..3)
            .flat_map(|y| {
                (0..3).flat_map(move |z| (0..3).map(move |x| SectionPos::new(x, y + 2, z)))
            })
            .collect();
        let forward: Vec<LoadedCube> = cubes.iter().map(|&p| one.make_cube(p)).collect();
        let backward: Vec<LoadedCube> = cubes.iter().rev().map(|&p| two.make_cube(p)).collect();
        // The populations that touch the 3 x 3 x 3 block: 4 x 4 x 4 of them, each once.
        assert_eq!(one.population_stats().populations, 64);
        assert_eq!(two.population_stats().populations, 64);
        // Overlapping veins can differ by who went first; nearly every block agrees.
        let (mut same, mut total) = (0usize, 0usize);
        for (x, y) in forward.iter().zip(backward.iter().rev()) {
            let (bx, by) = (blocks_of(x), blocks_of(y));
            same += bx.iter().zip(&by).filter(|(p, q)| p == q).count();
            total += bx.len();
        }
        assert!(same * 1000 >= total * 995, "{same}/{total}");
    }

    #[test]
    fn populated_flags_survive_a_reopen() {
        let Some(f) = Fixture::new("flags") else {
            return;
        };
        let pos = SectionPos::new(1, 3, 1);
        let level = f.level_of(CubicKind::Custom);
        let before = blocks_of(&level.make_cube(pos));
        let runs = level.population_stats().populations;
        assert_eq!(runs, 8);
        level.flush();
        drop(level);
        let again = f.level_of(CubicKind::Custom);
        for d in 0..8 {
            let p = SectionPos::new(1 - (d & 1), 3 - (d >> 1 & 1), 1 - (d >> 2 & 1));
            assert!(again.is_populated(p), "{p:?} was populated");
        }
        let after = blocks_of(&again.make_cube(pos));
        assert_eq!(again.population_stats().populations, 0, "none run again");
        assert_eq!(before, after);
        // A cube only a neighbour's population wrote into is partial.
        let partial = again
            .read(SectionPos::new(2, 4, 2))
            .map_or(0, |r| root_flags(&r));
        assert_eq!(partial & POPULATED, 0);
    }

    #[test]
    fn the_far_generators_parameters_come_from_the_generator() {
        let Some(f) = Fixture::new("far") else {
            return;
        };
        let level = f.level_of(CubicKind::Custom);
        let setup = crate::cubic_lod::far_setup(&level, f.assets.registry).expect("terrain world");
        let tables = level.generation().generator.noise_tables().unwrap();
        for (desc, table) in setup.params.noises.iter().zip(&tables) {
            assert_eq!(desc.octaves.len(), table.octaves.len());
            assert_eq!(desc.factor, table.factor);
            assert_eq!(desc.frequency, table.frequency);
        }
        let shape = level.generation().generator.density_shape().unwrap();
        assert_eq!(setup.params.height_factor, shape[0]);
        assert_eq!(setup.params.height_variation_offset, shape[3]);
        assert_eq!(
            setup.params.sea_level,
            level.generation().generator.sea_level()
        );
        assert_eq!(setup.params.stone, f.state("minecraft:stone"));
        assert_eq!(setup.params.deepslate, f.state("minecraft:deepslate"));
        // The grass biome tops are grass.
        let plains = usize::from(f.biomes.id("minecraft:plains"));
        assert_eq!(
            setup.params.biomes[plains].top,
            f.state("minecraft:grass_block")
        );
        // A Flat world has nothing to evaluate.
        assert!(
            crate::cubic_lod::far_setup(&f.level_of(CubicKind::Flat), f.assets.registry).is_none()
        );
    }
}
