//! Resident client chunk storage. Sections are reference-counted so worker
//! threads can mesh and light immutable snapshots while the game thread edits
//! (copy-on-write through `Arc::make_mut`).
use horizon_core::paletted::{PalettedContainer, Strategy};
use horizon_world::registry::BlockStateId;
use std::sync::{Arc, OnceLock};

/// The vanilla Overworld's column: the default `ColumnLayout`, until every
/// use reads the layout instead (docs/world-height.md).
pub const MIN_Y: i32 = -64;
pub const HEIGHT: i32 = 384;
pub const SECTIONS: usize = (HEIGHT / 16) as usize;

/// The span of block Y every resident chunk holds: from `min_y`, `sections`
/// sections up. It covers each dimension of the open world (`LevelHeightAccessor`
/// of each level), so the Nether's `0..256` sits inside the Overworld's
/// `-64..320` column.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColumnLayout {
    min_y: i32,
    sections: usize,
}
impl ColumnLayout {
    pub const VANILLA: Self = Self {
        min_y: MIN_Y,
        sections: SECTIONS,
    };
    /// `DimensionType.MIN_Y`, `MAX_Y`: 12 bits of `BlockPos` Y, less 32.
    pub const LOWEST_Y: i32 = -2032;
    pub const HIGHEST_Y: i32 = 2031;
    /// A dimension type's `min_y` and `height`, as its codec accepts them:
    /// multiples of 16, at least 16 high, within `LOWEST_Y..=HIGHEST_Y`.
    pub fn new(min_y: i32, height: i32) -> Result<Self, String> {
        if min_y % 16 != 0 || height % 16 != 0 {
            return Err(format!(
                "min_y {min_y} and height {height} must be multiples of 16"
            ));
        }
        if height < 16 {
            return Err(format!("height {height} is below 16"));
        }
        if min_y < Self::LOWEST_Y
            || i64::from(min_y) + i64::from(height) > i64::from(Self::HIGHEST_Y) + 1
        {
            return Err(format!(
                "min_y {min_y} with height {height} leaves {}..={}",
                Self::LOWEST_Y,
                Self::HIGHEST_Y
            ));
        }
        Ok(Self {
            min_y,
            sections: (height / 16) as usize,
        })
    }
    /// The smallest layout holding every `(min_y, height)` of `levels`.
    pub fn covering(levels: impl IntoIterator<Item = (i32, i32)>) -> Result<Self, String> {
        let (mut low, mut high) = (i32::MAX, i32::MIN);
        for (min_y, height) in levels {
            let level = Self::new(min_y, height)?;
            low = low.min(level.min_y);
            high = high.max(level.max_y());
        }
        if low > high {
            return Err("no levels".to_owned());
        }
        Self::new(low, high - low)
    }
    /// The layout covering every dimension type a server synchronized.
    pub fn of_registries(registries: &horizon_protocol::registry::Dynamic) -> Result<Self, String> {
        let types = registries.len("minecraft:dimension_type");
        Self::covering((0..types).map(|i| {
            let d = registries.dimension(i as i32);
            (d.min_y, d.height)
        }))
    }
    pub fn min_y(self) -> i32 {
        self.min_y
    }
    /// One above the highest block.
    pub fn max_y(self) -> i32 {
        self.min_y + self.height()
    }
    pub fn height(self) -> i32 {
        self.sections as i32 * 16
    }
    pub fn sections(self) -> usize {
        self.sections
    }
    /// The index of the section holding block `y`, if the column holds it.
    pub fn section_index(self, y: i32) -> Option<usize> {
        usize::try_from((y - self.min_y) >> 4)
            .ok()
            .filter(|&s| s < self.sections)
    }
    /// The minimum block Y of section `index`.
    pub fn section_y(self, index: usize) -> i32 {
        self.min_y + index as i32 * 16
    }
    const fn pack(self) -> u64 {
        (self.min_y as u32 as u64) << 32 | self.sections as u64
    }
    fn unpack(bits: u64) -> Self {
        Self {
            min_y: (bits >> 32) as u32 as i32,
            sections: (bits & u64::from(u32::MAX)) as usize,
        }
    }
}
impl Default for ColumnLayout {
    fn default() -> Self {
        Self::VANILLA
    }
}

static LAYOUT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(ColumnLayout::VANILLA.pack());

/// The column new chunks get.
pub fn layout() -> ColumnLayout {
    ColumnLayout::unpack(LAYOUT.load(std::sync::atomic::Ordering::Relaxed))
}

/// Sets the column for the world being opened, before its first chunk (as
/// `set_block_states` sizes the palette): chunks of another layout must not
/// stay resident.
pub fn set_layout(layout: ColumnLayout) {
    LAYOUT.store(layout.pack(), std::sync::atomic::Ordering::Relaxed);
}

/// `true` for each section of the column: what callers that keep no
/// per-section record (of fluids, say) pass as "every section may have it".
pub fn every_section() -> &'static [bool] {
    // The tallest column vanilla allows: 4064 blocks.
    static ALL: [bool; 254] = [true; 254];
    &ALL[..layout().sections()]
}

pub use horizon_world::worldgen::overrides::LevelHeight;

/// The levels whose build range `level_height` keeps: the three a world has.
pub const LEVELS: [&str; 3] = [
    "minecraft:overworld",
    "minecraft:the_nether",
    "minecraft:the_end",
];
const fn pack_level(h: LevelHeight) -> u64 {
    (h.min_y as u32 as u64) << 32
        | (h.height as u64 & 0xffff) << 16
        | h.logical_height as u64 & 0xffff
}
static LEVEL_HEIGHTS: [std::sync::atomic::AtomicU64; 3] = [
    std::sync::atomic::AtomicU64::new(pack_level(LevelHeight {
        min_y: -64,
        height: 384,
        logical_height: 384,
    })),
    std::sync::atomic::AtomicU64::new(pack_level(LevelHeight {
        min_y: 0,
        height: 256,
        logical_height: 128,
    })),
    std::sync::atomic::AtomicU64::new(pack_level(LevelHeight {
        min_y: 0,
        height: 256,
        logical_height: 256,
    })),
];

/// The open world's build range of `level` (`LevelHeightAccessor`, from its
/// dimension type): vanilla's until `set_level_heights`. Other names get
/// the Overworld's.
pub fn level_height(level: &str) -> LevelHeight {
    let i = LEVELS.iter().position(|l| *l == level).unwrap_or(0);
    let bits = LEVEL_HEIGHTS[i].load(std::sync::atomic::Ordering::Relaxed);
    LevelHeight {
        min_y: (bits >> 32) as u32 as i32,
        height: (bits >> 16 & 0xffff) as i32,
        logical_height: (bits & 0xffff) as i32,
    }
}

/// Sets the open world's levels' build ranges (in `LEVELS` order) and the
/// column covering them, before the world's first chunk.
pub fn set_level_heights(heights: [LevelHeight; 3]) -> Result<ColumnLayout, String> {
    let layout = ColumnLayout::covering(heights.iter().map(|h| (h.min_y, h.height)))?;
    for (slot, h) in LEVEL_HEIGHTS.iter().zip(heights) {
        slot.store(pack_level(h), std::sync::atomic::Ordering::Relaxed);
    }
    set_layout(layout);
    Ok(layout)
}

/// The vanilla levels' build ranges.
pub fn vanilla_level_heights() -> [LevelHeight; 3] {
    LEVELS.map(|l| LevelHeight::vanilla(l).expect("a vanilla level"))
}

/// Biomes are 4x4x4 containers over the biome table (67 entries: a 7-bit
/// global palette).
pub const BIOME_STRATEGY: Strategy = Strategy::biomes(7);

/// Bits of the block states' global palette: 16 for vanilla's 35,723 states
/// (what vanilla servers and clients send), more when content mods add states
/// past 65,536 (docs/modpacks.md). Set once, before the first chunk.
static BLOCK_BITS: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(16);

/// Block states are `PalettedContainer`s over the registry: vanilla's 16-bit
/// global palette, wider with enough content mods.
pub fn block_strategy() -> Strategy {
    Strategy::blocks(BLOCK_BITS.load(std::sync::atomic::Ordering::Relaxed))
}

/// Sizes the block states' global palette for a registry of `states`.
pub fn set_block_states(states: usize) {
    let bits = horizon_core::paletted::ceil_log2(states).max(16);
    BLOCK_BITS.store(bits, std::sync::atomic::Ordering::Relaxed);
}
/// Packed light per block: sky light in the high nibble, block light in the low one.
#[derive(Clone)]
pub enum Light {
    Uniform(u8),
    Full(Box<[u8; 4096]>),
}
impl Light {
    #[inline]
    pub fn get(&self, i: usize) -> u8 {
        match self {
            Self::Uniform(v) => *v,
            Self::Full(a) => a[i],
        }
    }
}
#[inline]
fn column(x: i32, z: i32) -> usize {
    ((z & 15) << 4 | (x & 15)) as usize
}
#[inline]
pub fn index(x: i32, y: i32, z: i32) -> usize {
    ((y & 15) << 8 | (z & 15) << 4 | (x & 15)) as usize
}
#[derive(Clone)]
pub struct Section {
    pub blocks: PalettedContainer,
    /// Biome table index per 4x4x4 cell, `(y << 2 | z) << 2 | x`.
    pub biomes: PalettedContainer,
    pub light: Light,
}
impl Default for Section {
    fn default() -> Self {
        Self {
            blocks: PalettedContainer::new(block_strategy(), 0),
            biomes: PalettedContainer::new(BIOME_STRATEGY, 0),
            light: Light::Uniform(0xF0),
        }
    }
}
impl Section {
    #[inline]
    pub fn get(&self, i: usize) -> u32 {
        self.blocks.get(i)
    }
    pub fn set(&mut self, i: usize, v: u32) -> u32 {
        self.blocks.set(i, v)
    }
    /// All 4,096 block states in index order.
    pub fn unpack(&self, out: &mut [u32; 4096]) {
        self.blocks.unpack(out);
    }
    /// The block states of a whole section at once.
    pub fn from_blocks(values: &[u32; 4096]) -> PalettedContainer {
        PalettedContainer::from_values(block_strategy(), values)
    }
    #[inline]
    pub fn light(&self, i: usize) -> u8 {
        self.light.get(i)
    }
    pub fn is_uniform(&self, v: u32) -> bool {
        self.blocks.is_uniform(v)
    }
    /// Heap bytes held by the block, biome and light storage (benchmark accounting).
    pub fn heap_bytes(&self) -> usize {
        self.blocks.heap_bytes()
            + self.biomes.heap_bytes()
            + match &self.light {
                Light::Uniform(_) => 0,
                Light::Full(_) => 4096,
            }
    }
    /// Collapse storage that became uniform, and drop palette entries no
    /// block uses any more.
    pub fn compact(&mut self) {
        for container in [&mut self.blocks, &mut self.biomes] {
            if container.bits() != 0
                && let Some(v) = container.uniform()
            {
                *container = PalettedContainer::new(container.strategy(), v);
            }
        }
        if let Light::Full(a) = &self.light
            && a.iter().all(|&b| b == a[0])
        {
            self.light = Light::Uniform(a[0]);
        }
    }
}
#[derive(Clone)]
pub struct Chunk {
    pub x: i32,
    pub z: i32,
    /// The minimum block Y of `sections[0]` (the layout's when it was made).
    min_y: i32,
    pub sections: Vec<Arc<Section>>,
    /// `Heightmap.Types.WORLD_SURFACE` per column (`z << 4 | x`): one above
    /// the highest non-air block, or the column's minimum Y. Built on the first read; a
    /// `set` that can move a column's surface drops it.
    surface: OnceLock<Box<[i16; 256]>>,
    /// `Heightmap.Types.MOTION_BLOCKING` per column, kept like `surface`.
    motion_blocking: OnceLock<Box<[i16; 256]>>,
    pub lit: bool,
    /// Light read from a save with `isLightOn`: no light job is needed.
    pub light_loaded: bool,
    /// Newly generated (not loaded from a save): eligible for generation-time spawns.
    pub fresh: bool,
    /// Block entities read from a save, moved into the world on arrival.
    pub block_entities: Vec<((i32, i32, i32), crate::containers::BlockEntity)>,
    /// `CustomName` of the container block entities (shown as the screen title).
    pub block_names: Vec<((i32, i32, i32), String)>,
    /// Generation postprocessing marks (ProtoChunk.postProcessing) as world
    /// positions, applied once by LevelChunk.postProcessGeneration.
    pub postprocess: Vec<[i32; 3]>,
    /// Saved data Horizon keeps but does not simulate (save.rs).
    pub saved: Option<Arc<crate::save::ChunkExtras>>,
    /// Entities read from a save (vanilla entity NBT), spawned on arrival.
    pub entities: Vec<horizon_world::nbt::Tag>,
    /// Structure starts reaching into this chunk (`structures.References`):
    /// structure id and the start's chunk.
    pub structure_refs: Vec<(Arc<str>, (i32, i32))>,
}
impl Chunk {
    /// An all-air chunk in the current `layout`.
    pub fn empty(x: i32, z: i32) -> Self {
        Self::with_layout(x, z, layout())
    }
    pub fn with_layout(x: i32, z: i32, layout: ColumnLayout) -> Self {
        Self {
            x,
            z,
            min_y: layout.min_y(),
            sections: (0..layout.sections())
                .map(|_| Arc::new(Section::default()))
                .collect(),
            surface: OnceLock::new(),
            motion_blocking: OnceLock::new(),
            lit: false,
            light_loaded: false,
            fresh: false,
            block_entities: Vec::new(),
            block_names: Vec::new(),
            structure_refs: Vec::new(),
            postprocess: Vec::new(),
            saved: None,
            entities: Vec::new(),
        }
    }
    /// The minimum block Y of the column.
    #[inline]
    pub fn min_y(&self) -> i32 {
        self.min_y
    }
    /// One above the column's highest block.
    #[inline]
    pub fn max_y(&self) -> i32 {
        self.min_y + self.sections.len() as i32 * 16
    }
    /// The index of the section holding block `y`, if the column holds it.
    #[inline]
    pub fn section_index(&self, y: i32) -> Option<usize> {
        usize::try_from((y - self.min_y) >> 4)
            .ok()
            .filter(|&s| s < self.sections.len())
    }
    #[inline]
    pub fn get(&self, x: i32, y: i32, z: i32) -> u32 {
        match self.section_index(y) {
            Some(s) => self.sections[s].get(index(x, y, z)),
            None => 0,
        }
    }
    #[inline]
    pub fn block(&self, x: i32, y: i32, z: i32) -> BlockStateId {
        BlockStateId(self.get(x, y, z))
    }
    /// The highest y where `matches`, scanning down. Sections holding only
    /// `skip` are passed over whole: `matches` must be false for it.
    pub fn highest(&self, skip: u32, mut matches: impl FnMut(i32) -> bool) -> Option<i32> {
        self.sections
            .iter()
            .enumerate()
            .rev()
            .filter(|(_, section)| !section.is_uniform(skip))
            .find_map(|(s, _)| {
                let base = self.min_y + s as i32 * 16;
                (base..base + 16).rev().find(|&y| matches(y))
            })
    }
    pub fn set(&mut self, x: i32, y: i32, z: i32, v: u32) -> u32 {
        let Some(s) = self.section_index(y) else {
            return v;
        };
        let previous = Arc::make_mut(&mut self.sections[s]).set(index(x, y, z), v);
        // `Heightmap.update`: a write two or more below a map's height can't
        // move it. Any other change rebuilds that map on the next read.
        if previous != v {
            for map in [&mut self.surface, &mut self.motion_blocking] {
                if map
                    .get()
                    .is_some_and(|heights| y >= i32::from(heights[column(x, z)]) - 1)
                {
                    *map = OnceLock::new();
                }
            }
        }
        previous
    }
    /// `getHeight(Heightmap.Types.WORLD_SURFACE, x, z)` (`x`, `z` within the
    /// chunk): one above the highest block that isn't `air`, or the column's
    /// minimum Y.
    /// Sections must change only through `set` once this has been read.
    pub fn world_surface(&self, x: i32, z: i32, air: impl Fn(u32) -> bool) -> i32 {
        let surface = self.surface.get_or_init(|| self.prime(|v| !air(v)));
        i32::from(surface[column(x, z)])
    }
    /// `getHeight(Heightmap.Types.MOTION_BLOCKING, x, z)` (`x`, `z` within
    /// the chunk): one above the highest state where `blocks` (a motion
    /// blocking block or a fluid), or the column's minimum Y. Kept as
    /// `world_surface` is.
    pub fn motion_blocking(&self, x: i32, z: i32, blocks: impl Fn(u32) -> bool) -> i32 {
        let heights = self.motion_blocking.get_or_init(|| self.prime(blocks));
        i32::from(heights[column(x, z)])
    }
    /// `Heightmap.primeHeightmaps`: each section from the top, skipping
    /// those of one state that doesn't count, until every column is found.
    fn prime(&self, counts: impl Fn(u32) -> bool) -> Box<[i16; 256]> {
        let mut surface = Box::new([self.min_y as i16; 256]);
        let mut found = [false; 256];
        let mut left = 256;
        let mut values = [0u32; 4096];
        for (s, section) in self.sections.iter().enumerate().rev() {
            if left == 0 {
                break;
            }
            if section.blocks.uniform().is_some_and(|v| !counts(v)) {
                continue;
            }
            section.unpack(&mut values);
            let base = self.min_y + s as i32 * 16;
            for i in 0..256 {
                if found[i] {
                    continue;
                }
                if let Some(dy) = (0..16).rev().find(|&dy| counts(values[dy << 8 | i])) {
                    surface[i] = (base + dy as i32 + 1) as i16;
                    found[i] = true;
                    left -= 1;
                }
            }
        }
        surface
    }
    #[inline]
    pub fn light(&self, x: i32, y: i32, z: i32) -> u8 {
        let s = (y - self.min_y) >> 4;
        if s < 0 {
            return 0;
        }
        if s >= self.sections.len() as i32 {
            return 0xF0;
        }
        self.sections[s as usize].light(index(x, y, z))
    }
    /// Heap bytes held by the sections (benchmark accounting).
    pub fn heap_bytes(&self) -> usize {
        self.sections.iter().map(|s| s.heap_bytes()).sum::<usize>()
    }
    /// Biome table index at a block position (`ChunkAccess.getNoiseBiome` with
    /// the quart y clamped to the column).
    #[inline]
    pub fn biome(&self, x: i32, y: i32, z: i32) -> u16 {
        let qy = ((y - self.min_y) >> 2).clamp(0, self.sections.len() as i32 * 4 - 1);
        self.sections[(qy >> 2) as usize]
            .biomes
            .get(((qy & 3) << 4 | ((z & 15) >> 2) << 2 | (x & 15) >> 2) as usize) as u16
    }
    /// Sets the biome of one quart cell (`qy` counts from the column's bottom).
    pub fn set_biome_cell(&mut self, qx: i32, qy: i32, qz: i32, biome: u16) {
        let section = Arc::make_mut(&mut self.sections[(qy >> 2) as usize]);
        section.biomes.set(
            ((qy & 3) << 4 | (qz & 3) << 2 | (qx & 3)) as usize,
            u32::from(biome),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn highest_skips_only_sections_that_cannot_match() {
        let (air, cave_air, stone) = (0, 7, 1);
        let mut chunk = Chunk::empty(0, 0);
        // Stone low down, cave air (uniform in its section) above it, and a
        // section of air with one stone block higher up.
        for y in MIN_Y..MIN_Y + 40 {
            chunk.set(3, y, 4, stone);
        }
        for i in 0..4096 {
            Arc::make_mut(&mut chunk.sections[5]).set(i, cave_air);
        }
        chunk.set(3, 150, 4, stone);
        let full = |c: &Chunk, matches: &dyn Fn(u32) -> bool| {
            (MIN_Y..MIN_Y + HEIGHT)
                .rev()
                .find(|&y| matches(c.get(3, y, 4)))
        };
        for matches in [
            &(|v: u32| v != air) as &dyn Fn(u32) -> bool,
            &|v: u32| v == stone,
            &|v: u32| v == cave_air,
        ] {
            let fast = chunk.highest(air, |y| matches(chunk.get(3, y, 4)));
            assert_eq!(fast, full(&chunk, matches));
        }
        assert_eq!(
            chunk.highest(air, |y| chunk.get(3, y, 4) == stone),
            Some(150)
        );
        assert_eq!(Chunk::empty(0, 0).highest(air, |_| unreachable!()), None);
    }
    /// The kept WORLD_SURFACE heights match a scan of every column after
    /// each of a few thousand writes (air, cave air, stone, whole sections).
    #[test]
    fn world_surface_follows_writes() {
        let (air, cave_air, stone) = (0, 7, 1);
        let is_air = |v: u32| v == air || v == cave_air;
        let scan = |c: &Chunk, x: i32, z: i32| {
            (MIN_Y..MIN_Y + HEIGHT)
                .rev()
                .find(|&y| !is_air(c.get(x, y, z)))
                .map_or(MIN_Y, |y| y + 1)
        };
        let mut chunk = Chunk::empty(0, 0);
        assert_eq!(chunk.world_surface(5, 5, is_air), MIN_Y);
        for y in MIN_Y..MIN_Y + 70 {
            for i in 0..256 {
                chunk.set(i & 15, y, i >> 4, stone);
            }
        }
        for i in 0..4096 {
            Arc::make_mut(&mut chunk.sections[8]).set(i, cave_air);
        }
        let mut random = horizon_core::random::LegacyRandom::new(9);
        use horizon_core::random::RandomSource;
        for step in 0..4000 {
            let x = random.next_int_bound(16).unwrap();
            let z = random.next_int_bound(16).unwrap();
            let y = MIN_Y + 40 + random.next_int_bound(120).unwrap();
            let v = [air, cave_air, stone][random.next_int_bound(3).unwrap() as usize];
            chunk.set(x, y, z, v);
            if step % 7 == 0 {
                for i in 0..256 {
                    assert_eq!(
                        chunk.world_surface(i & 15, i >> 4, is_air),
                        scan(&chunk, i & 15, i >> 4)
                    );
                }
            }
        }
        // Out-of-range writes and unchanged states keep the map.
        let before = chunk.world_surface(3, 3, is_air);
        chunk.set(3, MIN_Y + HEIGHT, 3, stone);
        chunk.set(3, before - 1, 3, chunk.get(3, before - 1, 3));
        assert!(chunk.surface.get().is_some());
        assert_eq!(chunk.world_surface(3, 3, |_| unreachable!()), before);
    }
    /// MOTION_BLOCKING ignores states that don't count (air, grass) and
    /// follows writes as the surface map does, both maps kept at once.
    #[test]
    fn motion_blocking_follows_writes() {
        let (air, grass, water, stone) = (0, 3, 5, 1);
        let blocks = |v: u32| v == water || v == stone;
        let scan = |c: &Chunk, x: i32, z: i32| {
            (MIN_Y..MIN_Y + HEIGHT)
                .rev()
                .find(|&y| blocks(c.get(x, y, z)))
                .map_or(MIN_Y, |y| y + 1)
        };
        let mut chunk = Chunk::empty(0, 0);
        assert_eq!(chunk.motion_blocking(5, 5, blocks), MIN_Y);
        for y in MIN_Y..MIN_Y + 70 {
            for i in 0..256 {
                chunk.set(i & 15, y, i >> 4, stone);
            }
        }
        for i in 0..4096 {
            Arc::make_mut(&mut chunk.sections[8]).set(i, grass);
        }
        let mut random = horizon_core::random::LegacyRandom::new(11);
        use horizon_core::random::RandomSource;
        for step in 0..4000 {
            let x = random.next_int_bound(16).unwrap();
            let z = random.next_int_bound(16).unwrap();
            let y = MIN_Y + 40 + random.next_int_bound(120).unwrap();
            let v = [air, grass, water, stone][random.next_int_bound(4).unwrap() as usize];
            chunk.set(x, y, z, v);
            if step % 7 == 0 {
                for i in 0..256 {
                    assert_eq!(
                        chunk.motion_blocking(i & 15, i >> 4, blocks),
                        scan(&chunk, i & 15, i >> 4)
                    );
                    chunk.world_surface(i & 15, i >> 4, |v| v == air);
                }
            }
        }
    }

    #[test]
    fn layouts_follow_the_dimension_type_codec() {
        assert_eq!(ColumnLayout::new(-64, 384), Ok(ColumnLayout::VANILLA));
        assert_eq!(ColumnLayout::default(), ColumnLayout::VANILLA);
        // To The Max's Overworld: the highest top vanilla allows.
        let tall = ColumnLayout::new(-64, 2096).unwrap();
        assert_eq!((tall.sections(), tall.max_y()), (131, 2032));
        assert!(ColumnLayout::new(-2032, 4064).is_ok());
        for (min_y, height) in [(-64, 2112), (-2048, 64), (-60, 384), (0, 250), (0, 0)] {
            assert!(
                ColumnLayout::new(min_y, height).is_err(),
                "{min_y} {height}"
            );
        }
        assert_eq!(
            ColumnLayout::covering([(-64, 384), (0, 256), (0, 256)]),
            Ok(ColumnLayout::VANILLA)
        );
        assert_eq!(
            ColumnLayout::covering([(0, 256), (-64, 2096)]).map(ColumnLayout::sections),
            Ok(131)
        );
        assert!(ColumnLayout::covering([]).is_err());
        assert_eq!(tall.section_index(-65), None);
        assert_eq!(tall.section_index(-64), Some(0));
        assert_eq!(tall.section_index(2031), Some(130));
        assert_eq!(tall.section_index(2032), None);
        assert_eq!(tall.section_y(130), 2016);
        assert_eq!(ColumnLayout::unpack(tall.pack()), tall);
        let low = ColumnLayout::new(-2032, 16).unwrap();
        assert_eq!(ColumnLayout::unpack(low.pack()), low);
    }

    #[test]
    fn a_server_s_dimension_types_set_the_layout() {
        use horizon_protocol::packets::configuration::{RegistryData, RegistryEntry};
        use horizon_world::nbt::Tag;
        let mut registries = horizon_protocol::registry::Dynamic::default();
        let entry = |id: &str, data| RegistryEntry {
            id: id.into(),
            data,
        };
        registries.add(RegistryData {
            registry: "minecraft:dimension_type".into(),
            entries: vec![
                entry("minecraft:overworld", None),
                entry("minecraft:the_nether", None),
            ],
        });
        assert_eq!(
            ColumnLayout::of_registries(&registries),
            Ok(ColumnLayout::VANILLA)
        );
        registries.add(RegistryData {
            registry: "minecraft:dimension_type".into(),
            entries: vec![
                entry(
                    "minecraft:overworld",
                    Some(Tag::compound([
                        ("min_y", Tag::Int(-64)),
                        ("height", Tag::Int(2096)),
                    ])),
                ),
                entry("minecraft:the_nether", None),
            ],
        });
        let tall = ColumnLayout::of_registries(&registries).unwrap();
        assert_eq!((tall.min_y(), tall.sections()), (-64, 131));
    }

    #[test]
    fn a_tall_column_stores_its_whole_height() {
        let layout = ColumnLayout::new(-64, 2096).unwrap();
        let mut chunk = Chunk::with_layout(0, 0, layout);
        assert_eq!(
            (chunk.min_y(), chunk.max_y(), chunk.sections.len()),
            (-64, 2032, 131)
        );
        let stone = 1;
        assert_eq!(chunk.set(5, 2031, 6, stone), 0);
        assert_eq!(chunk.set(5, 2032, 6, stone), stone, "above the column");
        assert_eq!(chunk.get(5, 2031, 6), stone);
        assert_eq!(chunk.get(5, 2032, 6), 0);
        assert_eq!(chunk.world_surface(5, 6, |v| v == 0), 2032);
        assert_eq!(chunk.world_surface(0, 0, |v| v == 0), -64);
        assert_eq!(chunk.highest(0, |y| y < 1000), None);
        chunk.set(5, 999, 6, stone);
        assert_eq!(chunk.highest(0, |y| y < 1000), Some(999));
        assert_eq!(chunk.light(0, 2100, 0), 0xF0);
        assert_eq!(chunk.light(0, -65, 0), 0);
        chunk.set_biome_cell(0, 130 * 4 + 3, 0, 9);
        assert_eq!(chunk.biome(0, 5000, 0), 9);
    }
}
