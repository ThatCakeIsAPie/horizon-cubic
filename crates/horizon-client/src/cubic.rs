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
//!   keeps the entities outside the window that a column's file holds.

use crate::biome::Biomes;
use crate::chunk::{BIOME_STRATEGY, Chunk, ColumnLayout, LevelShape, Section, block_strategy};
use crate::save::{Persistent, SaveTables, decode_with, encode_with};
use crate::streaming::{ChunkGenerator, Proto};
use horizon_core::paletted::PalettedContainer;
use horizon_core::position::SectionPos;
use horizon_world::cubic::generator::CubeGenerator;
use horizon_world::cubic::store::CubeStore;
use horizon_world::nbt::{NamedTag, Tag};
use horizon_world::registry::BlockRegistry;
use std::collections::HashMap;
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
        if let Err(e) = self.store.write(pos, tag) {
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
    /// Entities of each column outside the window: kept in the column's
    /// entity file when the column's entities are saved.
    parked: Mutex<HashMap<(i32, i32), Vec<Tag>>>,
    /// Uniform cubes made by the generator, shared by every column.
    uniform: Mutex<HashMap<(u32, u16), Arc<Section>>>,
    /// The biomes of the generator's cubes below and above vanilla's range.
    deep_biome: u16,
    sky_biome: u16,
    bedrock: Option<u32>,
    deepslate: Option<u32>,
}

impl CubicLevel {
    /// The cubic level saved in `level_dir` (its `region3d/`), generating
    /// with `seed`, its chunks holding `window`.
    pub fn new(
        level_dir: &Path,
        seed: i64,
        registry: &BlockRegistry,
        biomes: &Biomes,
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
            parked: Mutex::default(),
            uniform: Mutex::default(),
            deep_biome: biomes.id("minecraft:deep_dark"),
            sky_biome: biomes.id("minecraft:plains"),
            bedrock: registry.default_state("minecraft:bedrock").map(|s| s.0),
            deepslate: registry.default_state("minecraft:deepslate").map(|s| s.0),
        });
        Ok(level)
    }

    /// A far-terrain sampler of the level (`cubic_lod`), reading inside
    /// vanilla's range from `vanilla`, the vanilla generator's.
    pub fn lod_sampler(
        &self,
        vanilla: Box<dyn crate::streaming::LodSampler>,
        registry: &BlockRegistry,
    ) -> crate::cubic_lod::CubicLodSampler {
        let id = |name: &str| registry.default_state(name).map_or(0, |s| s.0);
        crate::cubic_lod::CubicLodSampler::new(
            self.generator.clone(),
            vanilla,
            id("minecraft:air"),
            id("minecraft:stone"),
            id("minecraft:deepslate"),
            self.deep_biome,
            self.sky_biome,
        )
    }

    /// The window the level's chunks hold.
    pub fn window(&self) -> ColumnLayout {
        *self.window.lock().unwrap()
    }

    /// Moves the level's window: the columns loaded from now on hold
    /// `window`. The caller saved the old window's columns and drops them.
    pub fn set_window(&self, window: LevelShape) {
        *self.window.lock().unwrap() = window.column();
        self.clean.lock().unwrap().clear();
        self.parked.lock().unwrap().clear();
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

    fn note_clean(&self, pos: SectionPos, section: &Section) {
        self.clean.lock().unwrap().insert(pos, blocks_hash(section));
    }

    /// Column (x, z) of the window, from saved cubes and the generator; its
    /// vanilla range must have been generated (`has_vanilla`). Entities
    /// outside the window stay parked.
    fn assemble(&self, x: i32, z: i32, window: ColumnLayout, ctx: &Codec) -> Chunk {
        let mut chunk = Chunk::with_layout(x, z, window);
        let base = window.min_y() >> 4;
        let mut surface = None;
        for s in 0..window.sections() {
            let cy = base + s as i32;
            let pos = SectionPos::new(x, cy, z);
            if let Some(root) = self.read(pos)
                && let Some(cube) = ctx.decode(&root, cy)
            {
                chunk.sections[s] = cube.sections[0].clone();
                chunk.block_entities.extend(cube.block_entities);
                chunk.block_names.extend(cube.block_names);
                if cy == MARKER_CUBE {
                    chunk.structure_refs = cube.structure_refs;
                    chunk.structure_starts = cube.structure_starts;
                    surface = surface_heights(&root);
                }
                self.note_clean(pos, &chunk.sections[s]);
                continue;
            }
            if VANILLA_CUBES.contains(&cy) {
                // Missing from a generated column: air, as vanilla's empty sections.
                continue;
            }
            chunk.sections[s] = self.generate(pos);
            self.note_clean(pos, &chunk.sections[s]);
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
        for (s, section) in vanilla.sections.iter().enumerate() {
            let cy = vanilla_base + s as i32;
            if !VANILLA_CUBES.contains(&cy) {
                continue;
            }
            let pos = SectionPos::new(x, cy, z);
            let mut root = ctx.encode(&vanilla, s, cy == MARKER_CUBE);
            if cy == MARKER_CUBE
                && let Tag::Compound(map) = &mut root.tag
            {
                map.insert(SURFACE_KEY.into(), Tag::IntArray(heights.clone()));
            }
            self.write_later(pos, root);
            self.note_clean(pos, section);
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
                self.note_clean(pos, &chunk.sections[s]);
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
        chunk.saved = vanilla.saved.take();
        chunk.fresh = vanilla.fresh;
        chunk.generated = vanilla.generated;
        let entities = std::mem::take(&mut vanilla.entities);
        chunk.entities = self.park_outside(x, z, entities, window);
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

    /// `entities` (a column's, vanilla entity NBT) inside `window`; the rest
    /// replace the column's parked ones.
    fn park_outside(&self, x: i32, z: i32, entities: Vec<Tag>, window: ColumnLayout) -> Vec<Tag> {
        let (inside, outside): (Vec<Tag>, Vec<Tag>) = entities.into_iter().partition(|e| {
            entity_y(e)
                .is_none_or(|y| y >= f64::from(window.min_y()) && y < f64::from(window.max_y()))
        });
        let mut parked = self.parked.lock().unwrap();
        if outside.is_empty() {
            parked.remove(&(x, z));
        } else {
            parked.insert((x, z), outside);
        }
        inside
    }

    /// A column's entities to save: those in the world (`entities`) and the
    /// parked ones outside the window.
    pub fn with_parked(&self, x: i32, z: i32, mut entities: Vec<Tag>) -> Vec<Tag> {
        if let Some(parked) = self.parked.lock().unwrap().get(&(x, z)) {
            entities.extend(parked.iter().cloned());
        }
        entities
    }

    /// Saves the cubes of `chunks` (window columns) that changed since they
    /// were loaded, generated or last saved, or that hold block entities.
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
            for (s, section) in chunk.sections.iter().enumerate() {
                let cy = base + s as i32;
                let pos = SectionPos::new(chunk.x, cy, chunk.z);
                let low = cy * 16;
                let has_entities = chunk
                    .block_entities
                    .iter()
                    .any(|(p, _)| p.1 >= low && p.1 < low + 16);
                let hash = blocks_hash(section);
                let changed = self.clean.lock().unwrap().get(&pos) != Some(&hash);
                if !changed && !has_entities {
                    continue;
                }
                let marker = cy == MARKER_CUBE;
                let mut root = ctx.encode(chunk, s, marker);
                if marker
                    && let Some(old) = self.read(pos)
                    && let Some(surface) = old.tag.get(SURFACE_KEY).cloned()
                    && let Tag::Compound(map) = &mut root.tag
                {
                    map.insert(SURFACE_KEY.into(), surface);
                }
                self.write_later(pos, root);
                self.clean.lock().unwrap().insert(pos, hash);
                written += 1;
            }
        }
        if wait {
            self.flush();
        }
        written
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
    while let Ok((pos, tag)) = receive.recv() {
        io.write_now(pos, &tag);
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
    fn encode(&self, chunk: &Chunk, s: usize, column: bool) -> NamedTag {
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
        // Light is made again when a cube loads.
        one.lit = false;
        let shape = LevelShape::cubic_window(LevelShape::OVERWORLD, low, 16);
        encode_with(&one, self.registry, self.biomes, self.tables, shape)
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
        Codec {
            registry: self.inner.registry.registry,
            biomes: &self.inner.biomes,
            tables: Some(
                self.inner
                    .save
                    .tables
                    .get_or_init(|| Arc::new(SaveTables::new(&self.inner.registry)))
                    .as_ref(),
            ),
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
        let mut chunk = self.level.assemble(x, z, window, &self.codec());
        let entities = self.inner.save.load_entities(x, z);
        chunk.entities = self.level.park_outside(x, z, entities, window);
        Some(chunk)
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
}
