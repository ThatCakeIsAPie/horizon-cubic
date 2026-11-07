//! Biomes for the custom generator: the input trait the integrator
//! implements, CWG/1.12 `(baseHeight, heightVariation)` and surface blocks
//! for every 26.3 overworld biome, and CWG's 5 x 5 blend (`BiomeSource`).
//!
//! # The trait
//!
//! [`BiomeColumns`] answers "which biome is at this 4 x 4 block column?" at
//! sea level (a quart-resolution lookup, like CWG's `getBiomesForGeneration`
//! sections). The integrator implements it over 26.3's multi-noise biome
//! source; [`FixedBiome`] is the single-biome implementation.
//!
//! # The 1.12 analogues
//!
//! CWG reads `Biome.getBaseHeight()` and `getHeightVariation()` of the 1.12
//! biome. 26.3 has biomes that did not exist; [`biome_info`] maps each to the
//! nearest 1.12 one (the comment on each row says which):
//!
//! * meadow, cherry grove, grove, snowy slopes: `ice_mountains` /
//!   `forest_hills` (0.45 / 0.3).
//! * jagged, frozen and stony peaks, windswept hills / forest / gravelly
//!   hills: `extreme_hills` (1.0 / 0.5).
//! * mangrove swamp: `swampland`. Pale garden: `roofed_forest`. Lush caves:
//!   `jungle`; dripstone caves and deep dark: `plains` (they are underground
//!   biomes; a sea-level lookup returns them rarely).
//! * badlands: `mesa` (0.1 / 0.2), wooded badlands: `mesa_rock` (1.5 / 0.025),
//!   eroded badlands: `mutated_mesa` (0.1 / 0.2).
//! * cold / frozen / lukewarm / warm oceans and their deep versions:
//!   `ocean` (-1.0 / 0.1) and `deep_ocean` (-1.8 / 0.1).
//!
//! Unknown biome names (modded) behave like plains.
//!
//! # The blend
//!
//! For each 4 x 4 section the heights and variations of the 5 x 5 sections
//! around it are averaged with weight `10 / sqrt(dx^2 + dz^2 + 0.2) /
//! (neighbourHeight + 2)` (absolute value, halved when the neighbour is taller
//! than the centre biome), then converted: `height = h * 17 / 64 - 1 / 256`,
//! `volatility = 2.4 * v + 4 / 15`. Per-block values interpolate bilinearly
//! between section corners.

use horizon_core::position::SectionPos;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

use super::settings::BiomeFilter;
use crate::worldgen::biome::BiomeId;

/// The biome lookup the generator needs. Both methods must be pure functions
/// of their arguments (same answer on every call and thread).
pub trait BiomeColumns: Send + Sync {
    /// The biome of the 4 x 4 column `quart_x`, `quart_z` (block `x >> 2`).
    fn biome(&self, quart_x: i32, quart_z: i32) -> BiomeId;
    /// The namespaced name (`minecraft:plains`) of a biome this source can
    /// return.
    fn biome_name(&self, id: BiomeId) -> &str;
    /// Every biome this source can return, when it knows them (empty when it
    /// does not): what decides which structures a level can have.
    fn possible_biomes(&self) -> Vec<BiomeId> {
        Vec::new()
    }
}

/// One biome everywhere (single-biome mode, tests).
#[derive(Clone, Debug)]
pub struct FixedBiome {
    name: String,
}

impl FixedBiome {
    pub fn new(name: &str) -> Self {
        Self {
            name: super::settings::normalize_biome(name),
        }
    }
}

impl BiomeColumns for FixedBiome {
    fn biome(&self, _: i32, _: i32) -> BiomeId {
        BiomeId(0)
    }
    fn possible_biomes(&self) -> Vec<BiomeId> {
        vec![BiomeId(0)]
    }
    fn biome_name(&self, _: BiomeId) -> &str {
        &self.name
    }
}

/// What the generator knows about a biome name (no block ids yet).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BiomeInfo {
    /// 1.12 `getBaseHeight()`.
    pub base_height: f32,
    /// 1.12 `getHeightVariation()`.
    pub height_variation: f32,
    /// Surface block (without the namespace).
    pub top: &'static str,
    /// Block under the surface.
    pub filler: &'static str,
}

const fn info(h: f32, v: f32, top: &'static str, filler: &'static str) -> BiomeInfo {
    BiomeInfo {
        base_height: h,
        height_variation: v,
        top,
        filler,
    }
}

const GRASS: &str = "grass_block";
const DIRT: &str = "dirt";

/// Every block name [`biome_info`] can return, for resolving up front.
pub const SURFACE_BLOCKS: &[&str] = &[
    "grass_block",
    "dirt",
    "sand",
    "gravel",
    "mycelium",
    "snow_block",
    "mud",
    "stone",
    "red_sand",
    "terracotta",
    "sandstone",
    "red_sandstone",
];

/// Height, variation and surface blocks of a biome (`name` with or without
/// `minecraft:`).
pub fn biome_info(name: &str) -> BiomeInfo {
    let n = name.strip_prefix("minecraft:").unwrap_or(name);
    match n {
        // ocean (-1.0 / 0.1), deep ocean (-1.8 / 0.1)
        "ocean" | "warm_ocean" | "lukewarm_ocean" => info(-1.0, 0.1, "sand", "sand"),
        "cold_ocean" | "frozen_ocean" => info(-1.0, 0.1, "gravel", "gravel"),
        "deep_ocean" | "deep_lukewarm_ocean" => info(-1.8, 0.1, "sand", "sand"),
        "deep_cold_ocean" | "deep_frozen_ocean" => info(-1.8, 0.1, "gravel", "gravel"),
        "mushroom_fields" => info(0.2, 0.3, "mycelium", DIRT),
        "plains" | "sunflower_plains" | "snowy_plains" | "dripstone_caves" | "deep_dark" => {
            info(0.125, 0.05, GRASS, DIRT)
        }
        "ice_spikes" => info(0.425, 0.45, "snow_block", DIRT),
        "desert" => info(0.125, 0.05, "sand", "sand"),
        "swamp" => info(-0.2, 0.1, GRASS, DIRT),
        "mangrove_swamp" => info(-0.2, 0.1, "mud", "mud"),
        "forest" | "birch_forest" | "dark_forest" | "pale_garden" => info(0.1, 0.2, GRASS, DIRT),
        "flower_forest" => info(0.1, 0.4, GRASS, DIRT),
        "old_growth_birch_forest" => info(0.2, 0.4, GRASS, DIRT),
        "jungle" | "sparse_jungle" | "bamboo_jungle" | "lush_caves" => info(0.1, 0.2, GRASS, DIRT),
        "taiga" | "snowy_taiga" | "old_growth_pine_taiga" | "old_growth_spruce_taiga" => {
            info(0.2, 0.2, GRASS, DIRT)
        }
        "savanna" => info(0.125, 0.05, GRASS, DIRT),
        "savanna_plateau" => info(1.5, 0.025, GRASS, DIRT),
        "windswept_savanna" => info(0.3625, 1.225, GRASS, DIRT),
        "windswept_hills" | "windswept_forest" | "windswept_gravelly_hills" => {
            info(1.0, 0.5, GRASS, DIRT)
        }
        "meadow" | "cherry_grove" | "grove" => info(0.45, 0.3, GRASS, DIRT),
        "snowy_slopes" => info(0.45, 0.3, "snow_block", DIRT),
        "frozen_peaks" | "jagged_peaks" => info(1.0, 0.5, "snow_block", "stone"),
        "stony_peaks" => info(1.0, 0.5, "stone", "stone"),
        "badlands" => info(0.1, 0.2, "red_sand", "terracotta"),
        "eroded_badlands" => info(0.1, 0.2, "red_sand", "terracotta"),
        "wooded_badlands" => info(1.5, 0.025, "red_sand", "terracotta"),
        "river" | "frozen_river" => info(-0.5, 0.0, GRASS, DIRT),
        "beach" | "snowy_beach" => info(0.0, 0.025, "sand", "sand"),
        "stony_shore" => info(0.1, 0.8, "stone", "stone"),
        _ => info(0.125, 0.05, GRASS, DIRT),
    }
}

/// Whether a biome is a Bryce-pillar mesa (eroded badlands).
pub fn is_bryce(name: &str) -> bool {
    name == "minecraft:eroded_badlands"
}
/// Whether a mesa has forest on its plateau (wooded badlands).
pub fn is_wooded_mesa(name: &str) -> bool {
    name == "minecraft:wooded_badlands"
}

/// `ConversionUtils.biomeHeightVanilla`.
pub fn convert_height(h: f64) -> f64 {
    h * 17.0 / 64.0 - 1.0 / 256.0
}
/// `ConversionUtils.biomeHeightVariationVanilla`.
pub fn convert_variation(v: f64) -> f64 {
    2.4 * v + 4.0 / 15.0
}

/// A biome with everything the generator needs, resolved once per biome id.
#[derive(Debug)]
pub struct ResolvedBiome {
    pub name: String,
    pub info: BiomeInfo,
    pub top: u32,
    pub filler: u32,
    pub filler_is_sand: bool,
    pub filler_is_red_sand: bool,
    pub bryce: bool,
    pub wooded_mesa: bool,
    /// Bit `i` set: replacer `i` applies in this biome.
    pub replacer_mask: Vec<u64>,
}

impl ResolvedBiome {
    pub fn replacer_applies(&self, i: usize) -> bool {
        self.replacer_mask[i >> 6] >> (i & 63) & 1 != 0
    }
}

/// Block ids of the surface table, plus the per-replacer biome filters, used
/// to resolve biomes.
pub struct BiomeResolver {
    blocks: HashMap<&'static str, u32>,
    filters: Vec<Option<BiomeFilter>>,
}

impl BiomeResolver {
    pub fn new(blocks: HashMap<&'static str, u32>, filters: Vec<Option<BiomeFilter>>) -> Self {
        Self { blocks, filters }
    }

    pub fn resolve(&self, name: &str) -> ResolvedBiome {
        let info = biome_info(name);
        let top = self.blocks[info.top];
        let filler = self.blocks[info.filler];
        let mut mask = vec![0_u64; self.filters.len().div_ceil(64).max(1)];
        for (i, f) in self.filters.iter().enumerate() {
            if f.as_ref().is_none_or(|f| f.test(name)) {
                mask[i >> 6] |= 1 << (i & 63);
            }
        }
        ResolvedBiome {
            name: name.to_owned(),
            info,
            top,
            filler,
            filler_is_sand: info.filler == "sand",
            filler_is_red_sand: info.filler == "red_sand",
            bryce: is_bryce(name),
            wooded_mesa: is_wooded_mesa(name),
            replacer_mask: mask,
        }
    }
}

// ---------------------------------------------------------------------------
// The blend, with per-thread caches
// ---------------------------------------------------------------------------

/// A multiplicative hash for the caches' integer keys (they are positions
/// and ids; SipHash's resistance to chosen keys buys nothing here).
#[derive(Default, Clone, Copy)]
struct IntHasher(u64);

impl IntHasher {
    #[inline]
    fn add(&mut self, v: u64) {
        self.0 = (self.0.rotate_left(5) ^ v).wrapping_mul(0xf135_7aea_2e62_a9c5);
    }
}

impl std::hash::Hasher for IntHasher {
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.add(u64::from(b));
        }
    }
    fn write_u32(&mut self, v: u32) {
        self.add(u64::from(v));
    }
    fn write_i32(&mut self, v: i32) {
        self.add(u64::from(v as u32));
    }
    fn write_u64(&mut self, v: u64) {
        self.add(v);
    }
    fn finish(&self) -> u64 {
        // The product's high bits are its best mixed; the table indexes by
        // the low ones.
        self.0.rotate_left(26)
    }
}

type IntMap<K, V> = HashMap<K, V, std::hash::BuildHasherDefault<IntHasher>>;

#[derive(Default)]
struct Caches {
    biomes: IntMap<(i32, i32), BiomeId>,
    terrain: IntMap<(i32, i32), (f64, f64)>,
    resolved: IntMap<u32, Rc<ResolvedBiome>>,
    /// Density lattice corners (`Lattice::for_cube`), which neighbouring
    /// cubes share.
    corners: IntMap<(i32, i32, i32), f64>,
}

thread_local! {
    static CACHES: RefCell<IntMap<u64, Caches>> = RefCell::new(IntMap::default());
}

const CACHE_LIMIT: usize = 1 << 16;

static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// The biome side of a generator: the lookup, the resolver and the blend.
pub struct BiomeBlend {
    id: u64,
    columns: Arc<dyn BiomeColumns>,
    resolver: BiomeResolver,
}

/// Height and volatility of a column, already in CWG's units (before
/// `heightFactor` and `heightVariationFactor`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColumnTerrain {
    pub height: f64,
    pub volatility: f64,
}

impl BiomeBlend {
    pub fn new(columns: Arc<dyn BiomeColumns>, resolver: BiomeResolver) -> Self {
        Self {
            id: NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            columns,
            resolver,
        }
    }

    pub fn columns(&self) -> &Arc<dyn BiomeColumns> {
        &self.columns
    }

    /// The density lattice corner at a block position, as this thread last
    /// computed it for this blend's terrain (`Lattice::for_cube`).
    pub fn cached_corner(&self, at: (i32, i32, i32)) -> Option<f64> {
        self.with(|c| c.corners.get(&at).copied())
    }

    /// Remembers a lattice corner (`cached_corner`).
    pub fn keep_corner(&self, at: (i32, i32, i32), density: f64) {
        self.with(|c| {
            c.corners.insert(at, density);
        });
    }

    fn with<R>(&self, f: impl FnOnce(&mut Caches) -> R) -> R {
        CACHES.with(|c| {
            let mut all = c.borrow_mut();
            if all.len() > 8 && !all.contains_key(&self.id) {
                all.clear();
            }
            let caches = all.entry(self.id).or_default();
            if caches.biomes.len() > CACHE_LIMIT {
                caches.biomes.clear();
            }
            if caches.terrain.len() > CACHE_LIMIT {
                caches.terrain.clear();
            }
            if caches.corners.len() > CACHE_LIMIT {
                caches.corners.clear();
            }
            f(caches)
        })
    }

    fn biome_id(&self, caches: &mut Caches, qx: i32, qz: i32) -> BiomeId {
        *caches
            .biomes
            .entry((qx, qz))
            .or_insert_with(|| self.columns.biome(qx, qz))
    }

    fn resolved(&self, caches: &mut Caches, id: BiomeId) -> Rc<ResolvedBiome> {
        if let Some(r) = caches.resolved.get(&id.0) {
            return r.clone();
        }
        let r = Rc::new(self.resolver.resolve(self.columns.biome_name(id)));
        caches.resolved.insert(id.0, r.clone());
        r
    }

    /// The biome of the block column `x`, `z`, resolved.
    pub fn biome_at(&self, x: i32, z: i32) -> Rc<ResolvedBiome> {
        self.with(|c| {
            let id = self.biome_id(c, x >> 2, z >> 2);
            self.resolved(c, id)
        })
    }

    /// The biome name at a block column.
    pub fn biome_name_at(&self, x: i32, z: i32) -> String {
        self.biome_at(x, z).name.clone()
    }

    /// The 16 x 16 biomes of a cube's columns, index `z << 4 | x`.
    pub fn cube_biomes(&self, cube: SectionPos) -> Vec<Rc<ResolvedBiome>> {
        self.with(|c| {
            let mut quarts: [Option<Rc<ResolvedBiome>>; 16] = Default::default();
            for qz in 0..4 {
                for qx in 0..4 {
                    let id = self.biome_id(c, cube.x * 4 + qx, cube.z * 4 + qz);
                    quarts[(qz * 4 + qx) as usize] = Some(self.resolved(c, id));
                }
            }
            (0..256)
                .map(|i| {
                    let (x, z) = (i & 15, i >> 4);
                    quarts[((z >> 2) * 4 + (x >> 2)) as usize].clone().unwrap()
                })
                .collect()
        })
    }

    /// The blended terrain at a section corner (`sx`, `sz` in 4-block units).
    fn corner(&self, caches: &mut Caches, sx: i32, sz: i32) -> (f64, f64) {
        if let Some(&v) = caches.terrain.get(&(sx, sz)) {
            return v;
        }
        let centre = biome_info(self.columns_name(caches, sx, sz).as_str()).base_height;
        let mut sum_v = 0.0;
        let mut sum_h = 0.0;
        let mut sum_w = 0.0;
        for nz in -2..=2_i32 {
            for nx in -2..=2_i32 {
                let name = self.columns_name(caches, sx + nx, sz + nz);
                let b = biome_info(&name);
                let h = f64::from(b.base_height);
                let v = f64::from(b.height_variation);
                let near = 10.0 / f64::from(nx * nx + nz * nz).mul_add(1.0, 0.2).sqrt();
                let denom = h + 2.0;
                let mut w = if denom.abs() < 1e-6 {
                    near * 1e6
                } else {
                    near / denom
                }
                .abs();
                if h > f64::from(centre) {
                    w /= 2.0;
                }
                sum_v += v * w;
                sum_h += h * w;
                sum_w += w;
            }
        }
        let out = (
            convert_height(sum_h / sum_w),
            convert_variation(sum_v / sum_w),
        );
        caches.terrain.insert((sx, sz), out);
        out
    }

    fn columns_name(&self, caches: &mut Caches, qx: i32, qz: i32) -> String {
        let id = self.biome_id(caches, qx, qz);
        self.columns.biome_name(id).to_owned()
    }

    /// Blended height and volatility at a block column: bilinear between the
    /// four section corners around it (`BiomeSource.getHeight` /
    /// `getVolatility`).
    pub fn terrain_at(&self, x: i32, z: i32) -> ColumnTerrain {
        self.with(|c| {
            let (x0, z0) = (x >> 2, z >> 2);
            if x & 3 == 0 && z & 3 == 0 {
                let (height, volatility) = self.corner(c, x0, z0);
                return ColumnTerrain { height, volatility };
            }
            let (xf, zf) = (f64::from(x & 3) * 0.25, f64::from(z & 3) * 0.25);
            let v00 = self.corner(c, x0, z0);
            let v01 = self.corner(c, x0, z0 + 1);
            let v10 = self.corner(c, x0 + 1, z0);
            let v11 = self.corner(c, x0 + 1, z0 + 1);
            let lerp = |t: f64, a: f64, b: f64| a + t * (b - a);
            ColumnTerrain {
                height: lerp(zf, lerp(xf, v00.0, v10.0), lerp(xf, v01.0, v11.0)),
                volatility: lerp(zf, lerp(xf, v00.1, v10.1), lerp(xf, v01.1, v11.1)),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Split;
    impl BiomeColumns for Split {
        fn biome(&self, qx: i32, _: i32) -> BiomeId {
            BiomeId(u32::from(qx >= 0))
        }
        fn biome_name(&self, id: BiomeId) -> &str {
            if id.0 == 0 {
                "minecraft:deep_ocean"
            } else {
                "minecraft:windswept_hills"
            }
        }
    }

    fn blend(columns: Arc<dyn BiomeColumns>) -> BiomeBlend {
        let blocks = SURFACE_BLOCKS
            .iter()
            .enumerate()
            .map(|(i, n)| (*n, i as u32))
            .collect();
        BiomeBlend::new(columns, BiomeResolver::new(blocks, Vec::new()))
    }

    #[test]
    fn a_fixed_biome_gives_its_own_height() {
        let b = blend(Arc::new(FixedBiome::new("plains")));
        let t = b.terrain_at(1000, -1000);
        assert!((t.height - convert_height(0.125)).abs() < 1e-9);
        // The table holds f32s: 0.05 is 0.05 to f32 precision.
        assert!((t.volatility - convert_variation(f64::from(0.05_f32))).abs() < 1e-9);
        assert_eq!(b.biome_at(5, 5).name, "minecraft:plains");
    }

    #[test]
    fn heights_blend_smoothly_across_a_biome_border() {
        let b = blend(Arc::new(Split));
        let mut last = b.terrain_at(-40, 0).height;
        let mut max_step: f64 = 0.0;
        for x in -39..40 {
            let h = b.terrain_at(x, 0).height;
            max_step = max_step.max((h - last).abs());
            last = h;
        }
        let deep = b.terrain_at(-40, 0).height;
        let high = b.terrain_at(40, 0).height;
        assert!(high > deep + 0.1);
        // CubicWorldGen's (and 1.12's) weights favour the lower biome
        // (`1 / (base + 2)`, taller neighbours halved), so between deep ocean
        // and hills most of the change comes as the last deep-ocean section
        // leaves the 5 x 5 window: a coast's slope, not a seam. The bilinear
        // corners keep it continuous: no step beyond a quarter of the span.
        assert!(max_step < (high - deep) / 4.0, "{max_step}");
    }

    #[test]
    fn every_surface_block_of_the_table_is_listed() {
        for n in [
            "ocean",
            "deep_cold_ocean",
            "mushroom_fields",
            "plains",
            "ice_spikes",
            "desert",
            "mangrove_swamp",
            "jungle",
            "windswept_savanna",
            "jagged_peaks",
            "stony_peaks",
            "badlands",
            "river",
            "beach",
            "stony_shore",
            "cherry_grove",
            "nonexistent",
        ] {
            let i = biome_info(n);
            assert!(SURFACE_BLOCKS.contains(&i.top), "{n}");
            assert!(SURFACE_BLOCKS.contains(&i.filler), "{n}");
        }
    }
}
