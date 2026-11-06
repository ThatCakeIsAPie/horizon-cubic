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
use crate::chunk::{BIOME_STRATEGY, Chunk, ColumnLayout, LevelShape, Section, block_strategy};
use crate::save::{ChunkExtras, Persistent, SaveTables, decode_with, encode_with};
use crate::streaming::{ChunkGenerator, Proto};
use horizon_core::paletted::PalettedContainer;
use horizon_core::position::SectionPos;
use horizon_core::ticks::SavedTick;
use horizon_world::cubic::generator::CubeGenerator;
use horizon_world::cubic::store::CubeStore;
use horizon_world::cubic::{MAX_CUBE_Y, MIN_CUBE_Y};
use horizon_world::nbt::{NamedTag, Tag};
use horizon_world::registry::BlockRegistry;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Condvar, Mutex};

/// The cube Ys vanilla's Overworld generator makes (-64..320).
pub const VANILLA_CUBES: std::ops::RangeInclusive<i32> = -4..=19;
/// The cube whose record says the column's vanilla range was generated and
/// holds the column's own data (structure starts and references, the
/// surface heights).
pub const MARKER_CUBE: i32 = -4;
/// A cubic window's height in blocks: 32 cubes.
pub const WINDOW_HEIGHT: i32 = 512;
/// How near a window's top or bottom a player may come before the window
/// moves (blocks).
pub const WINDOW_MARGIN: i32 = 96;
/// The window a new cubic world opens with: vanilla's surface in its middle.
pub const FIRST_WINDOW_MIN_Y: i32 = -192;
/// The key in a column's marker cube with its vanilla surface heights.
const SURFACE_KEY: &str = "horizon:cubic_surface";
/// The key in a cube record with the entities whose position is in the cube.
const ENTITIES_KEY: &str = "horizon:entities";

/// A cube as saved, for putting into a column that is loaded (a window
/// moving in place) or for building one.
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

/// The window around block height `y`: `WINDOW_HEIGHT` high, its bottom on
/// a multiple of 64 blocks, with `y` near its middle.
pub fn window_around(level: LevelShape, y: i32) -> LevelShape {
    let min_y = (y - WINDOW_HEIGHT / 2).div_euclid(64) * 64;
    LevelShape::cubic_window(level, min_y, WINDOW_HEIGHT)
}

/// Whether a player at height `y` in `window` should have a new window.
pub fn needs_new_window(window: LevelShape, y: f64) -> bool {
    window.cubic
        && (y < f64::from(window.min_y + WINDOW_MARGIN)
            || y >= f64::from(window.max_y() - WINDOW_MARGIN))
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
        if let Some(tag) = self.pending.lock().unwrap().get(&pos) {
            return Some((**tag).clone());
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
        let written = if is_uniform_record(tag) {
            self.store
                .write_uniform(pos, tag)
                .or_else(|_| self.store.write(pos, tag))
        } else {
            self.store.write(pos, tag)
        };
        if let Err(e) = written {
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

/// Makes the level's streaming again over the same generator (a moved
/// window keeps the generator, its saves and this level).
pub type StreamingFactory = Box<dyn Fn() -> Option<crate::streaming::Streaming> + Send + Sync>;

/// One cubic level's cubes on disk and in generation.
pub struct CubicLevel {
    io: Arc<CubeIo>,
    streaming: std::sync::OnceLock<StreamingFactory>,
    send: Mutex<Option<Sender<(SectionPos, Arc<NamedTag>)>>>,
    /// Columns whose vanilla range is known to be generated (`has_vanilla`).
    vanilla_done: Mutex<std::collections::HashSet<(i32, i32)>>,
    generator: Arc<CubeGenerator>,
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
    /// The biomes of the generator's cubes below and above vanilla's range.
    deep_biome: u16,
    sky_biome: u16,
    bedrock: Option<u32>,
    deepslate: Option<u32>,
    /// What cube records are decoded with for the far terrain.
    registry: &'static BlockRegistry,
    biomes: Arc<Biomes>,
}

impl CubicLevel {
    /// The cubic level saved in `level_dir` (its `region3d/`), generating
    /// with `seed`, its chunks holding `window`.
    pub fn new(
        level_dir: &Path,
        seed: i64,
        registry: &'static BlockRegistry,
        biomes: &Arc<Biomes>,
        window: LevelShape,
    ) -> Result<Arc<Self>, String> {
        let palette = horizon_world::cubic::generator::CubePalette::from_registry(registry)?;
        let generator = Arc::new(CubeGenerator::new(seed, palette));
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
            streaming: std::sync::OnceLock::new(),
            send: Mutex::new(Some(send)),
            vanilla_done: Mutex::default(),
            generator,
            window: Mutex::new(window.column()),
            clean: Mutex::default(),
            with_entities: Mutex::default(),
            edit: Mutex::default(),
            tables: std::sync::OnceLock::new(),
            uniform: Mutex::default(),
            deep_biome: biomes.id("minecraft:deep_dark"),
            sky_biome: biomes.id("minecraft:plains"),
            bedrock: registry.default_state("minecraft:bedrock").map(|s| s.0),
            deepslate: registry.default_state("minecraft:deepslate").map(|s| s.0),
            registry,
            biomes: biomes.clone(),
        });
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

    /// A far-terrain sampler of the level (`cubic_lod`), reading inside
    /// vanilla's range from `vanilla`, the vanilla generator's.
    pub fn lod_sampler(
        self: &Arc<Self>,
        vanilla: Box<dyn crate::streaming::LodSampler>,
        registry: &BlockRegistry,
    ) -> crate::cubic_lod::CubicLodSampler {
        let id = |name: &str| registry.default_state(name).map_or(0, |s| s.0);
        crate::cubic_lod::CubicLodSampler::new(
            self.clone(),
            self.generator.clone(),
            vanilla,
            id("minecraft:air"),
            id("minecraft:stone"),
            id("minecraft:deepslate"),
            vec![id("minecraft:water"), id("minecraft:lava")],
            self.deep_biome,
            self.sky_biome,
        )
    }

    /// The window the level's chunks hold.
    pub fn window(&self) -> ColumnLayout {
        *self.window.lock().unwrap()
    }

    /// Moves the level's window: the columns loaded from now on hold
    /// `window`. The caller saved the old window's columns and their
    /// entities (`save_entities` reads the window to know which cubes a
    /// column's entity list stands for) and drops them.
    pub fn set_window(&self, window: LevelShape) {
        *self.window.lock().unwrap() = window.column();
        self.clean.lock().unwrap().clear();
    }

    /// Sets how the level's streaming is made again (once, when it opens).
    pub fn set_streaming_factory(&self, factory: StreamingFactory) {
        let _ = self.streaming.set(factory);
    }

    /// A new streaming over the level's generator, for a moved window.
    pub fn new_streaming(&self) -> Option<crate::streaming::Streaming> {
        self.streaming.get()?()
    }

    /// Cube `pos`'s record: a queued write's, else the disk's.
    pub fn read(&self, pos: SectionPos) -> Option<NamedTag> {
        self.io.read(pos)
    }

    fn write_later(&self, pos: SectionPos, tag: NamedTag) {
        let tag = Arc::new(tag);
        self.io.pending.lock().unwrap().insert(pos, tag.clone());
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
        if self.vanilla_done.lock().unwrap().contains(&(x, z)) {
            return true;
        }
        let done = self.read(SectionPos::new(x, MARKER_CUBE, z)).is_some();
        if done {
            self.vanilla_done.lock().unwrap().insert((x, z));
        }
        done
    }

    /// The generator's cube `pos`, uniform cubes shared.
    pub fn generate(&self, pos: SectionPos) -> Arc<Section> {
        let mut values = Box::new([0u32; 4096]);
        let biome = if pos.y < *VANILLA_CUBES.start() {
            self.deep_biome
        } else {
            self.sky_biome
        };
        let section = match self.generator.generate(pos, &mut values) {
            Some(state) => {
                let mut uniform = self.uniform.lock().unwrap();
                return uniform
                    .entry((state, biome))
                    .or_insert_with(|| {
                        Arc::new(Section {
                            blocks: PalettedContainer::new(block_strategy(), state),
                            biomes: PalettedContainer::new(BIOME_STRATEGY, u32::from(biome)),
                            ..Section::default()
                        })
                    })
                    .clone();
            }
            None => {
                let mut section = Section {
                    blocks: Section::from_blocks(&values),
                    biomes: PalettedContainer::new(BIOME_STRATEGY, u32::from(biome)),
                    ..Section::default()
                };
                section.compact();
                section
            }
        };
        Arc::new(section)
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
        if let Some(root) = self.read(pos)
            && let Some(mut cube) = ctx.decode(&root, pos.y)
        {
            let entities = record_entities(&root);
            let saved = cube.saved.take();
            let (block_ticks, fluid_ticks) = saved
                .map(|e| (e.block_ticks.clone(), e.fluid_ticks.clone()))
                .unwrap_or_default();
            let marker = pos.y == MARKER_CUBE;
            let section = cube.sections[0].clone();
            self.note_clean(pos, &section, &block_ticks, &fluid_ticks);
            self.note_entities(pos, !entities.is_empty());
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
        if VANILLA_CUBES.contains(&pos.y) {
            return None;
        }
        let section = self.generate(pos);
        self.note_clean(pos, &section, &[], &[]);
        Some(LoadedCube {
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
            generated: true,
        })
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
            if cy == MARKER_CUBE {
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
        if surface.is_none()
            && !VANILLA_CUBES.contains(&base)
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
        self.strip_bedrock(&mut vanilla);
        // The column's surface, for sky light in windows below it.
        let air = |v: u32| v == 0;
        let mut heights = Vec::with_capacity(256);
        for i in 0..256 {
            heights.push(vanilla.world_surface(i & 15, i >> 4, air));
        }
        let vanilla_base = vanilla.min_y() >> 4;
        // The column's entities by cube: each is saved in its cube's record.
        let entities = split_by_cube(std::mem::take(&mut vanilla.entities));
        for (s, section) in vanilla.sections.iter().enumerate() {
            let cy = vanilla_base + s as i32;
            if !VANILLA_CUBES.contains(&cy) {
                continue;
            }
            let pos = SectionPos::new(x, cy, z);
            let (block_ticks, fluid_ticks) = cube_ticks(&vanilla, cy);
            self.note_clean(pos, section, &block_ticks, &fluid_ticks);
            // A missing vanilla cube reads as air: the air above the ground
            // needs no record (the marker always has one).
            let low = cy * 16;
            if cy != MARKER_CUBE
                && section.blocks.uniform() == Some(0)
                && block_ticks.is_empty()
                && fluid_ticks.is_empty()
                && !entities.contains_key(&cy)
                && !vanilla
                    .block_entities
                    .iter()
                    .any(|(p, _)| p.1 >= low && p.1 < low + 16)
            {
                continue;
            }
            let here = entities.get(&cy).map_or(&[][..], |e| &e[..]);
            self.note_entities(pos, !here.is_empty());
            let mut root = ctx.encode(&vanilla, s, cy == MARKER_CUBE, here);
            if cy == MARKER_CUBE
                && let Tag::Compound(map) = &mut root.tag
            {
                map.insert(SURFACE_KEY.into(), Tag::IntArray(heights.clone()));
            }
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
                && VANILLA_CUBES.contains(&cy)
            {
                chunk.sections[s] = vanilla.sections[i].clone();
            } else if !VANILLA_CUBES.contains(&cy) {
                chunk.sections[s] = self.generate(pos);
                self.note_clean(pos, &chunk.sections[s], &[], &[]);
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
        let (Some(bedrock), Some(deepslate)) = (self.bedrock, self.deepslate) else {
            return;
        };
        // Vanilla's floor is y -64..-60.
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
        let section = if VANILLA_CUBES.contains(&pos.y) {
            Arc::new(Section::default())
        } else {
            self.generate(pos)
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
        let marker = cy == MARKER_CUBE;
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
fn split_by_cube(entities: Vec<Tag>) -> BTreeMap<i32, Vec<Tag>> {
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
fn is_uniform_record(root: &NamedTag) -> bool {
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
}

impl<G: ChunkGenerator> CubicTerrain<G> {
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
        let window = self.level.window();
        let mut chunk = Chunk::with_layout(vanilla.x, vanilla.z, window);
        let base = window.min_y() >> 4;
        for s in 0..window.sections() {
            let cy = base + s as i32;
            if VANILLA_CUBES.contains(&cy) {
                if let Some(i) = vanilla.section_index(cy * 16) {
                    chunk.sections[s] = vanilla.sections[i].clone();
                }
            } else {
                chunk.sections[s] = self
                    .level
                    .generate(SectionPos::new(vanilla.x, cy, vanilla.z));
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
    /// The vanilla generator's far terrain (vanilla's range of the cubic
    /// far terrain, `cubic_lod`).
    fn lod_sampler(self: Arc<Self>) -> Box<dyn crate::streaming::LodSampler> {
        ChunkGenerator::lod_sampler(self.inner.clone())
    }
    fn lod_surface(&self, worker: &mut G::Worker, x: i32, z: i32) -> Option<crate::lod::Surface> {
        self.inner.lod_surface(worker, x, z)
    }
    fn load(&self, x: i32, z: i32) -> Option<Chunk> {
        let window = self.level.window();
        // A window that doesn't reach vanilla's range needs no vanilla
        // column: its cubes are saved ones and the generator's.
        let base = window.min_y() >> 4;
        let reaches_vanilla = VANILLA_CUBES
            .clone()
            .any(|cy| (base..base + window.sections() as i32).contains(&cy));
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
            .finish_vanilla(vanilla, self.level.window(), &self.codec())
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
    save_window(world_root, FIRST_WINDOW_MIN_Y)
}

/// The bottom of the window a cubic world's Overworld last held.
pub fn saved_window(world_root: &Path) -> Option<i32> {
    let text = std::fs::read_to_string(world_root.join(WORLD_FILE)).ok()?;
    let json: serde_json::Value = serde_json::from_str(&text).ok()?;
    json["window_min_y"].as_i64().map(|y| y as i32)
}

/// Records the bottom of the window a cubic world's Overworld holds.
pub fn save_window(world_root: &Path, min_y: i32) -> std::io::Result<()> {
    let json = serde_json::json!({ "cubic": true, "window_min_y": min_y });
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
    let min_y = saved_window(world_root).unwrap_or(FIRST_WINDOW_MIN_Y);
    shapes.overworld = LevelShape::cubic_window(shapes.overworld, min_y, WINDOW_HEIGHT);
    shapes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_follow_the_player_in_steps_of_64() {
        let w = window_around(LevelShape::OVERWORLD, 70);
        assert!(w.cubic);
        assert_eq!((w.min_y, w.max_y()), (-192, 320));
        assert!(!needs_new_window(w, 70.0));
        assert!(needs_new_window(w, f64::from(w.min_y + WINDOW_MARGIN - 1)));
        assert!(needs_new_window(w, f64::from(w.max_y() - WINDOW_MARGIN)));
        let far = window_around(LevelShape::OVERWORLD, 1_000_123);
        assert_eq!(far.min_y % 64, 0);
        assert!(f64::from(far.min_y + WINDOW_MARGIN) <= 1_000_123.0);
        assert!(f64::from(far.max_y() - WINDOW_MARGIN) > 1_000_123.0);
        assert!(!needs_new_window(far, 1_000_123.0));
        assert!(!needs_new_window(LevelShape::OVERWORLD, -10_000.0));
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
        assert_eq!(saved_window(&dir), Some(FIRST_WINDOW_MIN_Y));
        save_window(&dir, 4096).unwrap();
        assert_eq!(saved_window(&dir), Some(4096));
        let shapes = cubic_shapes(&dir, crate::chunk::LevelShapes::VANILLA);
        assert!(shapes.overworld.cubic && !shapes.nether.cubic);
        assert_eq!(shapes.overworld.min_y, 4096);
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
            let window = LevelShape::cubic_window(LevelShape::OVERWORLD, -192, 512);
            let level = CubicLevel::new(&self.dir, 1, self.assets.registry, &self.biomes, window)
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
                    BIOME_STRATEGY,
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
}
