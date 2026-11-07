//! CubicWorldGen's "Custom Cubic" generator as a library: mostly
//! vanilla-looking terrain by default, fully customisable through a JSON
//! [`CustomGeneratorSettings`] with CWG's fields. It replaces v1's own
//! deep / sky generator (`cubic/generator.rs`).
//!
//! # API
//!
//! * [`CustomCubicGenerator::new`] builds a generator from a seed, settings, the
//!   block registry and a [`BiomeColumns`] (the integrator implements it over
//!   26.3's multi-noise biome source, queried at sea level for every 4 x 4
//!   column; apply [`CustomGeneratorSettings::biome_scale`] to the quart
//!   coordinates first). `settings.biome` non-empty selects a fixed biome
//!   instead.
//! * [`CustomCubicGenerator::generate`] fills a cube: density terrain,
//!   replacers, then caves and ravines. A cube's blocks are a pure function
//!   of (seed, settings, cube), so cubes can be generated in any order on
//!   any thread (`Send + Sync`; caches are per thread or sharded).
//! * [`CustomCubicGenerator::populate`] adds ores, lakes and springs, and
//!   calls [`CubeWorld::decorate_surface`] for the integrator's vegetation.
//!   It writes into the 2 x 2 x 2 cubes from the populated cube up (CWG's
//!   +8 offset), see [`populate`].
//! * [`CustomCubicGenerator::block_at`] is density plus replacers without
//!   carvers or population, for the far terrain; `resolution_blocks > 1`
//!   skips noise octaves finer than that. [`CustomCubicGenerator::surface_y`]
//!   finds the terrain surface for spawn searches.
//! * [`bounds`]: conservative 3D bounds on the density.
//!   [`CustomCubicGenerator::generate`] returns a provably uniform cube
//!   without sampling the lattice; [`CustomCubicGenerator::classify_box`],
//!   [`CustomCubicGenerator::uniform_block`] and
//!   [`CustomCubicGenerator::classify_node`] expose the test to the far
//!   terrain, and [`CustomCubicGenerator::gpu_terms`] the per-octave terms.
//! * [`CustomCubicGenerator::noise_tables`] exposes the octave tables for a
//!   GPU twin (see [`density`]).
//! * `cubeAreas` delegate whole cubes (generation and population) to nested
//!   generators; the first area in list order that contains the cube wins.
//! * Flat Cubic: [`CustomCubicGenerator::flat`].
//!
//! The cube index layout is `y << 8 | z << 4 | x`, as in `cubic/generator.rs`.
//!
//! # Adaptations of CWG to 26.3
//!
//! See [`settings`] (defaults, biome names, deepslate), [`biomes`] (heights of
//! 26.3's biomes), [`replacers`], [`carvers`] and [`populate`]. In short:
//! Horizon's vanilla-style Perlin replaces libnoise, the default ores are 26.3's,
//! the default replacers include the deepslate gradient, dungeons are not
//! generated, `riverSize` is unused. Structures (26.3's overworld sets) are
//! the integrator's, through [`structures`] and the
//! [`CubeWorld::place_structures`] hook.

use std::sync::Arc;

use horizon_core::position::SectionPos;

use crate::registry::BlockRegistry;

pub mod biomes;
pub mod bounds;
pub mod carvers;
pub mod climate_biomes;
pub mod density;
pub mod flat;
pub mod populate;
pub mod replacers;
pub mod rng;
pub mod settings;
pub mod structures;
#[doc(hidden)]
pub mod testing;
pub mod vegetation;

pub use biomes::{BiomeColumns, FixedBiome};
pub use bounds::{BoxClass, DensityBound, GpuTerms, NodeClass};
pub use climate_biomes::ClimateBiomes;
pub use density::{NoiseTable, OctaveTable};
pub use populate::CubeWorld;
pub use settings::{CustomGeneratorSettings, FlatGeneratorSettings};

use biomes::{BiomeBlend, BiomeResolver, SURFACE_BLOCKS};
use bounds::{Analyzer, Query};
use density::{Lattice, TerrainNoise, cell_at};
use replacers::{BlockCtx, Replacers, resolve_block};
use settings::{BlockStateDesc, IntAabb, ReplacerConfig};

struct Terrain {
    sizes: [i32; 3],
    noise: TerrainNoise,
    bound: DensityBound,
    blend: BiomeBlend,
    replacers: Replacers,
    carvers: carvers::Carvers,
    populator: populate::Populator,
    sea_level: i32,
    biome_scale: f64,
}

enum Kind {
    Terrain(Box<Terrain>),
    Flat(flat::FlatGenerator),
}

/// The custom cubic generator (also Flat Cubic).
pub struct CustomCubicGenerator {
    kind: Kind,
    areas: Vec<(IntAabb, CustomCubicGenerator)>,
}

impl CustomCubicGenerator {
    /// Builds the generator. Fails with a message naming the offending
    /// setting or block (`unknown block mod:thing`).
    pub fn new(
        seed: i64,
        settings: &CustomGeneratorSettings,
        registry: &'static BlockRegistry,
        biomes: Arc<dyn BiomeColumns>,
    ) -> Result<Self, String> {
        settings.validate()?;
        let columns: Arc<dyn BiomeColumns> = if settings.biome.is_empty() {
            biomes.clone()
        } else {
            Arc::new(FixedBiome::new(&settings.biome))
        };
        let mut blocks = std::collections::HashMap::new();
        for name in SURFACE_BLOCKS {
            blocks.insert(*name, resolve_block(registry, &BlockStateDesc::new(name))?);
        }
        let filters = settings
            .replacers
            .iter()
            .map(|r| r.biome_filter().cloned())
            .collect();
        let blend = BiomeBlend::new(columns, BiomeResolver::new(blocks, filters));
        let sea_level = settings
            .replacers
            .iter()
            .find_map(|r| match r {
                ReplacerConfig::MainSurface(c) => Some(c.ocean_level as i32),
                _ => None,
            })
            .unwrap_or(63);
        let mut areas = Vec::new();
        for (i, a) in settings.cube_areas.iter().enumerate() {
            let g = Self::new(seed, &a.settings, registry, biomes.clone())
                .map_err(|e| format!("cubeAreas[{i}]: {e}"))?;
            areas.push((a.area, g));
        }
        let noise = TerrainNoise::new(seed, settings);
        Ok(Self {
            kind: Kind::Terrain(Box::new(Terrain {
                sizes: [
                    settings.noise_sample_size_x,
                    settings.noise_sample_size_y,
                    settings.noise_sample_size_z,
                ],
                bound: DensityBound::new(&noise),
                noise,
                blend,
                replacers: Replacers::new(seed, &settings.replacers, registry)?,
                carvers: carvers::Carvers::new(seed, settings, registry)?,
                populator: populate::Populator::new(seed, settings, registry)?,
                sea_level,
                biome_scale: settings.biome_scale(),
            })),
            areas,
        })
    }

    /// A Flat Cubic generator.
    pub fn flat(
        settings: &FlatGeneratorSettings,
        registry: &BlockRegistry,
    ) -> Result<Self, String> {
        Ok(Self {
            kind: Kind::Flat(flat::FlatGenerator::new(settings, registry)?),
            areas: Vec::new(),
        })
    }

    /// Whether some `cubeAreas` entry hands cubes to other settings (what
    /// reproduces only the main terrain, as the far view's surface tiles,
    /// does not apply).
    pub fn has_areas(&self) -> bool {
        !self.areas.is_empty()
    }

    fn area(&self, cube: SectionPos) -> Option<&CustomCubicGenerator> {
        self.areas
            .iter()
            .find(|(a, _)| a.contains(cube.x, cube.y, cube.z))
            .map(|(_, g)| g)
    }

    /// Fills `out` (`y << 8 | z << 4 | x`) with the cube's blocks: terrain,
    /// replacers, caves and ravines. Returns `Some(id)` when the whole cube is
    /// one block (`out` is filled either way).
    pub fn generate(&self, cube: SectionPos, out: &mut [u32; 4096]) -> Option<u32> {
        if let Some(g) = self.area(cube) {
            return g.generate(cube, out);
        }
        match &self.kind {
            Kind::Flat(f) => f.generate(cube, out),
            Kind::Terrain(t) => {
                t.generate(cube, out);
                let first = out[0];
                out.iter().all(|&b| b == first).then_some(first)
            }
        }
    }

    /// The cube as the density lattice and the replacers make it, before the
    /// carvers and without the uniform early-out: what [`Self::uniform_block`]
    /// must agree with. For tests and probes.
    #[doc(hidden)]
    pub fn base_cube(&self, cube: SectionPos, out: &mut [u32; 4096]) {
        if let Some(g) = self.area(cube) {
            return g.base_cube(cube, out);
        }
        match &self.kind {
            Kind::Flat(f) => {
                f.generate(cube, out);
            }
            Kind::Terrain(t) => t.base(cube, out),
        }
    }

    /// [`Self::generate`] without the uniform early-out (the lattice, then
    /// the carvers), for timing and comparison.
    #[doc(hidden)]
    pub fn generate_lattice(&self, cube: SectionPos, out: &mut [u32; 4096]) {
        if let Some(g) = self.area(cube) {
            return g.generate_lattice(cube, out);
        }
        match &self.kind {
            Kind::Flat(f) => {
                f.generate(cube, out);
            }
            Kind::Terrain(t) => {
                t.base(cube, out);
                t.carvers.carve(cube, out);
            }
        }
    }

    /// Adds ores, lakes and springs for `cube` (and calls the vegetation
    /// hook). The 2 x 2 x 2 cubes from `cube` up must be readable and
    /// writable in `world`.
    pub fn populate(&self, cube: SectionPos, world: &mut dyn CubeWorld) {
        if let Some(g) = self.area(cube) {
            return g.populate(cube, world);
        }
        if let Kind::Terrain(t) = &self.kind {
            t.populator.populate(cube, world, &t.blend);
        }
    }

    /// The block at a world position from density and replacers only (no
    /// caves, ravines or population). `resolution_blocks <= 1` gives exactly
    /// the blocks `generate` starts from; larger values skip noise octaves
    /// finer than that many blocks and take gradients by finite differences.
    pub fn block_at(&self, x: i32, y: i32, z: i32, resolution_blocks: i32) -> u32 {
        let cube = SectionPos::new(x >> 4, y >> 4, z >> 4);
        if let Some(g) = self.area(cube) {
            return g.block_at(x, y, z, resolution_blocks);
        }
        match &self.kind {
            Kind::Flat(f) => f.block_at_y(y),
            Kind::Terrain(t) => t.block_at(x, y, z, resolution_blocks),
        }
    }

    /// [`Self::block_at`] at the heights `ys` of the column (`x`, `z`), in
    /// order, until `stop` holds for a block: the index of that height and its
    /// block, or `None`. The same answers as `block_at` for each height, with
    /// the column's noise columns and biome computed once instead of per
    /// height (a far view's sky-cover probes ask a few heights per column).
    pub fn first_at_column(
        &self,
        x: i32,
        z: i32,
        ys: &[i32],
        resolution_blocks: i32,
        mut stop: impl FnMut(u32) -> bool,
    ) -> Option<(usize, u32)> {
        let mut probe = None;
        for (i, &y) in ys.iter().enumerate() {
            let cube = SectionPos::new(x >> 4, y >> 4, z >> 4);
            let block = match (&self.kind, self.area(cube)) {
                (Kind::Terrain(t), None) => probe
                    .get_or_insert_with(|| ColumnProbe::new(t, x, z, resolution_blocks))
                    .block_at(y),
                _ => self.block_at(x, y, z, resolution_blocks),
            };
            if stop(block) {
                return Some((i, block));
            }
        }
        None
    }

    /// [`Self::block_at`] at each of the heights `ys` of the column (`x`,
    /// `z`), as [`Self::first_at_column`] computes them.
    pub fn blocks_at_column(&self, x: i32, z: i32, ys: &[i32], resolution_blocks: i32) -> Vec<u32> {
        let mut out = Vec::with_capacity(ys.len());
        self.first_at_column(x, z, ys, resolution_blocks, |b| {
            out.push(b);
            false
        });
        out
    }

    /// The terrain surface of a column: the topmost solid block's Y at or
    /// below `from_y` (searching down at most 4096 blocks), or, when `from_y`
    /// is inside terrain, the top of that solid run (searching up at most 512
    /// blocks). Water counts as air; carvers are ignored.
    pub fn surface_y(&self, x: i32, z: i32, from_y: i32) -> Option<i32> {
        let cube = SectionPos::new(x >> 4, from_y >> 4, z >> 4);
        if let Some(g) = self.area(cube) {
            return g.surface_y(x, z, from_y);
        }
        match &self.kind {
            Kind::Flat(f) => f.surface_y().filter(|&s| s <= from_y),
            Kind::Terrain(t) => t.surface_y(x, z, from_y),
        }
    }

    /// The topmost solid block of each column of chunk (`cx`, `cz`)
    /// (`z << 4 | x`), from density and replacers (carvers and population
    /// ignored): what generation-time heightmaps see, the blocks `generate`
    /// starts from. [`NO_TOP`] for a column with no solid block within
    /// 4,096 blocks under the lowest height the density bound proves has
    /// only air above it (searched up from sea level).
    pub fn chunk_tops(&self, cx: i32, cz: i32) -> Box<[i32; 256]> {
        match &self.kind {
            Kind::Flat(f) => Box::new([f.surface_y().unwrap_or(NO_TOP); 256]),
            Kind::Terrain(t) => t.chunk_tops(cx, cz),
        }
    }

    /// Whether a box of blocks touches a `cubeAreas` entry (their blocks come
    /// from other settings, so the bounds do not apply).
    fn touches_area(&self, min: [i32; 3], max: [i32; 3]) -> bool {
        self.areas.iter().any(|(a, _)| {
            a.min_x <= max[0] >> 4
                && a.max_x >= min[0] >> 4
                && a.min_y <= max[1] >> 4
                && a.max_y >= min[1] >> 4
                && a.min_z <= max[2] >> 4
                && a.max_z >= min[2] >> 4
        })
    }

    /// The terrain, when the bounds apply to the box of blocks `min..=max`
    /// (a terrain generator whose `cubeAreas` do not touch it).
    fn bounded(&self, min: [i32; 3], max: [i32; 3]) -> Option<&Terrain> {
        match &self.kind {
            Kind::Terrain(t) if !self.touches_area(min, max) => Some(t),
            _ => None,
        }
    }

    /// Classifies the blocks `min..=max` (inclusive) by density alone, as the
    /// lattice generator would make them: [`BoxClass::Air`] when no block's
    /// density is above 0, [`BoxClass::Solid`] when all are. The answer is
    /// conservative: [`BoxClass::Mixed`] means "not proven". Air at or below
    /// the sea level is water, so use [`Self::uniform_block`] to know the
    /// block. Flat Cubic and boxes touching a `cubeAreas` entry are `Mixed`.
    pub fn classify_box(&self, min: [i32; 3], max: [i32; 3]) -> BoxClass {
        self.bounded(min, max)
            .map_or(BoxClass::Mixed, |t| t.classify_box(min, max))
    }

    /// The one block that all of `min..=max` is after density and replacers
    /// (no carvers, no population), when the bounds prove it; `None` if not.
    pub fn uniform_block(&self, min: [i32; 3], max: [i32; 3]) -> Option<u32> {
        self.bounded(min, max)?.uniform_block(min, max)
    }

    /// Classifies a node of the far-terrain octree: the cube-aligned box of
    /// side `16 << level` blocks whose minimum corner is `(x, y, z) * side`
    /// (and its far faces, where voxel corners are sampled). Noise octaves
    /// with a wavelength below `min_wavelength` blocks are skipped as in
    /// [`Self::block_at`]; they are bounded by amplitude, so the answer holds
    /// for the truncated density and for the full one.
    pub fn classify_node(
        &self,
        level: u32,
        x: i32,
        y: i32,
        z: i32,
        min_wavelength: f64,
    ) -> NodeClass {
        let side = 16_i64 << level.min(24);
        let at = |v: i32| -> i32 { (i64::from(v) * side).clamp(-(1 << 30), 1 << 30) as i32 };
        let min = [at(x), at(y), at(z)];
        let max = min.map(|v| (i64::from(v) + side).min(i64::from(i32::MAX)) as i32);
        let Some(t) = self.bounded(min, max) else {
            return NodeClass {
                class: BoxClass::Mixed,
                block: None,
            };
        };
        t.analyzer().node(&Query::exact(min, max, min_wavelength))
    }

    /// The bound's per-octave terms and shape, as plain data for a GPU twin
    /// of the descent (see [`bounds`]); `None` for Flat Cubic.
    pub fn gpu_terms(&self) -> Option<GpuTerms> {
        match &self.kind {
            Kind::Terrain(t) => Some(t.bound.gpu_terms()),
            Kind::Flat(_) => None,
        }
    }

    /// The density's change bound `L(r)` for one column (see
    /// [`DensityBound::l`]); `None` for Flat Cubic.
    pub fn density_bound(&self) -> Option<&DensityBound> {
        match &self.kind {
            Kind::Terrain(t) => Some(&t.bound),
            Kind::Flat(_) => None,
        }
    }

    /// The sea level (the first `mainSurface` replacer's `oceanLevel`, else 63).
    pub fn sea_level(&self) -> i32 {
        match &self.kind {
            Kind::Terrain(t) => t.sea_level,
            Kind::Flat(_) => 0,
        }
    }

    /// The octave tables of the selector, low, high and depth noises (the
    /// order they were seeded in), for a GPU twin; `None` for Flat Cubic.
    pub fn noise_tables(&self) -> Option<[NoiseTable; 4]> {
        match &self.kind {
            Kind::Terrain(t) => Some(t.noise.tables()),
            Kind::Flat(_) => None,
        }
    }

    /// `[heightFactor, heightOffset, heightVariationFactor,
    /// heightVariationOffset, specialHeightVariationFactorBelowAverageY]`.
    pub fn density_shape(&self) -> Option<[f64; 5]> {
        match &self.kind {
            Kind::Terrain(t) => Some(t.noise.shape()),
            Kind::Flat(_) => None,
        }
    }

    /// Blended biome height and volatility of a block column, in the units of
    /// [`biomes`] (before `heightFactor` / `heightVariationFactor`), for a
    /// twin that does the blend itself.
    pub fn column_terrain(&self, x: i32, z: i32) -> Option<(f64, f64)> {
        match &self.kind {
            Kind::Terrain(t) => {
                let c = t.blend.terrain_at(x, z);
                Some((c.height, c.volatility))
            }
            Kind::Flat(_) => None,
        }
    }

    /// The factor by which the integrator should divide quart coordinates
    /// before asking its biome source (`biomeSize / 4`).
    pub fn biome_scale(&self) -> f64 {
        match &self.kind {
            Kind::Terrain(t) => t.biome_scale,
            Kind::Flat(_) => 1.0,
        }
    }
}

/// `Terrain::block_at` for the blocks of one column: the noise columns its
/// finite differences take (at `x`, `x + h` and `z + h`) and the column's
/// biome are computed once, when a height first needs them.
struct ColumnProbe<'a> {
    t: &'a Terrain,
    x: i32,
    z: i32,
    resolution: i32,
    /// The columns at (`x`, `z`), (`x + h`, `z`) and (`x`, `z + h`).
    columns: Option<[density::Column; 3]>,
    biome: Option<std::rc::Rc<biomes::ResolvedBiome>>,
}

impl<'a> ColumnProbe<'a> {
    fn new(t: &'a Terrain, x: i32, z: i32, resolution: i32) -> Self {
        Self {
            t,
            x,
            z,
            resolution,
            columns: None,
            biome: None,
        }
    }

    fn block_at(&mut self, y: i32) -> u32 {
        let (t, x, z, resolution) = (self.t, self.x, self.z, self.resolution);
        if let Some(block) = t
            .analyzer()
            .uniform_block_coarse_first(&t.block_query(x, y, z, resolution), COARSE_PROOF)
        {
            return block;
        }
        let (density, dx, dy, dz) = if resolution <= 1 {
            cell_at(&t.noise, &t.blend, t.sizes, x, y, z)
        } else {
            let mw = f64::from(resolution);
            let h = resolution.min(8);
            let [c0, cx, cz] = self.columns.get_or_insert_with(|| {
                let col = |x: i32, z: i32| t.noise.column(x, z, t.blend.terrain_at(x, z), mw);
                [col(x, z), col(x + h, z), col(x, z + h)]
            });
            let v = t.noise.density(x, y, z, c0, mw);
            let inv = 1.0 / f64::from(h);
            (
                v,
                (t.noise.density(x + h, y, z, cx, mw) - v) * inv,
                (t.noise.density(x, y + h, z, c0, mw) - v) * inv,
                (t.noise.density(x, y, z + h, cz, mw) - v) * inv,
            )
        };
        let biome = self.biome.get_or_insert_with(|| t.blend.biome_at(x, z));
        t.replacers.block(&BlockCtx {
            x,
            y,
            z,
            dx,
            dy,
            dz,
            density,
            biome,
        })
    }
}

/// A column without a solid block ([`CustomCubicGenerator::chunk_tops`]).
pub const NO_TOP: i32 = i32::MIN / 2;

/// The shortest noise wavelength, in blocks, a cube's first uniformity proof
/// evaluates (finer octaves are bounded by their amplitude).
const COARSE_PROOF: f64 = 16.0;

impl Terrain {
    fn analyzer(&self) -> Analyzer<'_> {
        Analyzer {
            noise: &self.noise,
            blend: &self.blend,
            replacers: &self.replacers,
            bound: &self.bound,
            sizes: self.sizes,
        }
    }

    fn classify_box(&self, min: [i32; 3], max: [i32; 3]) -> BoxClass {
        self.analyzer()
            .classify(&Query::lattice(self.sizes, min, max))
    }

    fn uniform_block(&self, min: [i32; 3], max: [i32; 3]) -> Option<u32> {
        // A sound proof answers the cube's true block whatever the octaves
        // it bounds exactly: try the cheap one first.
        self.analyzer()
            .uniform_block_coarse_first(&Query::lattice(self.sizes, min, max), COARSE_PROOF)
    }

    fn generate(&self, cube: SectionPos, out: &mut [u32; 4096]) {
        let min = [cube.x * 16, cube.y * 16, cube.z * 16];
        if let Some(block) = self.uniform_block(min, min.map(|v| v + 15)) {
            // Proven uniform before the carvers: no lattice. The carvers
            // still run over it (their walks may reach the cube).
            out.fill(block);
        } else {
            self.base(cube, out);
        }
        self.carvers.carve(cube, out);
    }

    /// The cube's blocks from the density lattice and the replacers, before
    /// the carvers.
    fn base(&self, cube: SectionPos, out: &mut [u32; 4096]) {
        self.base_with(cube, out, true);
    }

    /// `base`; with `prove_cells`, a lattice cell whose corners and gradient
    /// let the replacer analysis prove one block is filled with it instead of
    /// running the chain for each of its blocks (the same blocks).
    fn base_with(&self, cube: SectionPos, out: &mut [u32; 4096], prove_cells: bool) {
        let lattice = Lattice::for_cube(
            &self.noise,
            &self.blend,
            self.sizes,
            [cube.x, cube.y, cube.z],
        );
        let biomes = self.blend.cube_biomes(cube);
        let (bx, by, bz) = (cube.x * 16, cube.y * 16, cube.z * 16);
        // The replacers that can act in this cube, per biome of its columns.
        let mut plans: Vec<(*const biomes::ResolvedBiome, Vec<u16>)> = Vec::new();
        let mut column_plan = [0u8; 256];
        for (i, biome) in biomes.iter().enumerate() {
            let key = std::rc::Rc::as_ptr(biome);
            column_plan[i] = match plans.iter().position(|(k, _)| *k == key) {
                Some(p) => p as u8,
                None => {
                    plans.push((key, self.replacers.plan(by, by + 15, biome)));
                    (plans.len() - 1) as u8
                }
            };
        }
        let [cx, cy, cz] = lattice.cells();
        let mut proven = [false; 64];
        // A cell must lie in one 4 x 4 biome column (and the mask fit).
        let one_quart = 4 % self.sizes[0] == 0 && 4 % self.sizes[2] == 0 && cx * cy * cz <= 64;
        if prove_cells && one_quart {
            let [sx, sy, sz] = self.sizes.map(|v| v as usize);
            for cj in 0..cy {
                for ck in 0..cz {
                    for ci in 0..cx {
                        let v = lattice.cell_corners(ci, cj, ck);
                        if v.iter().any(|d| !d.is_finite()) {
                            continue;
                        }
                        let lo = v.iter().copied().fold(f64::INFINITY, f64::min);
                        let hi = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                        // Only cells on one side of the surface can be one block.
                        if lo <= 0.0 && hi > 0.0 {
                            continue;
                        }
                        // The blocks interpolate the corners (and their
                        // vertical differences): within their range but for
                        // a few ulps of rounding, which the pads cover.
                        let pad = 1e-9 * (1.0 + lo.abs().max(hi.abs()));
                        let dy = lattice.cell_dy(ci, cj, ck);
                        let y0 = by + (cj * sy) as i32;
                        let biome = &biomes[(ck * sz) << 4 | ci * sx];
                        let facts = replacers::BoxFacts {
                            dmin: lo - pad,
                            dmax: hi + pad,
                            dy: dy * (1.0 + 1e-9) + 1e-12,
                            y0,
                            y1: y0 + sy as i32 - 1,
                            biomes: Some(std::slice::from_ref(biome)),
                        };
                        if let Some(&[block]) = self.replacers.possible_blocks(&facts).as_deref() {
                            for ly in cj * sy..(cj + 1) * sy {
                                for lz in ck * sz..(ck + 1) * sz {
                                    let row = ly << 8 | lz << 4;
                                    out[row + ci * sx..row + (ci + 1) * sx].fill(block);
                                }
                            }
                            proven[(cj * cz + ck) * cx + ci] = true;
                        }
                    }
                }
            }
        }
        let skip = |ci: usize, cj: usize, ck: usize| proven[(cj * cz + ck) * cx + ci];
        let mut memo = self.replacers.memo();
        lattice.for_each_except(skip, |lx, ly, lz, dx, dy, dz, density| {
            let ctx = BlockCtx {
                x: bx + lx as i32,
                y: by + ly as i32,
                z: bz + lz as i32,
                dx,
                dy,
                dz,
                density,
                biome: &biomes[lz << 4 | lx],
            };
            let plan = &plans[column_plan[lz << 4 | lx] as usize].1;
            out[ly << 8 | lz << 4 | lx] = self.replacers.block_planned(plan, &ctx, &mut memo);
        });
    }

    fn block_at(&self, x: i32, y: i32, z: i32, resolution: i32) -> u32 {
        ColumnProbe::new(self, x, z, resolution).block_at(y)
    }

    /// `block_at`'s proof query for the block (`x`, `y`, `z`).
    fn block_query(&self, x: i32, y: i32, z: i32, resolution: i32) -> Query {
        // One bound over the points the answer is computed from, instead of
        // eight (lattice cell) or four (finite differences) densities.
        if resolution <= 1 {
            Query::lattice(self.sizes, [x, y, z], [x, y, z])
        } else {
            let h = resolution.min(8);
            Query {
                blocks: ([x, y, z], [x, y, z]),
                samples: ([x, y, z], [x + h, y + h, z + h]),
                dy_edge: f64::from(h),
                min_wavelength: f64::from(resolution),
                column: None,
            }
        }
    }

    /// `CustomCubicGenerator::chunk_tops`: the ceiling is the first height
    /// from sea level up, in steps of 256, under 4,096 blocks the bound
    /// proves air; the cubes under it are then read top down (a cube the
    /// bound proves air skipped, one it proves solid ending every column
    /// left, others from their density lattice) until every column has met a
    /// solid block.
    fn chunk_tops(&self, cx: i32, cz: i32) -> Box<[i32; 256]> {
        let mut tops = Box::new([NO_TOP; 256]);
        let (x0, z0) = (cx * 16, cz * 16);
        let class = |y0: i32, y1: i32| self.classify_box([x0, y0, z0], [x0 + 15, y1, z0 + 15]);
        let limit = crate::cubic::MAX_Y - 4096;
        let mut ceiling = self.sea_level.clamp(crate::cubic::MIN_Y, limit) & !15;
        while class(ceiling, ceiling + 4095) != BoxClass::Air {
            ceiling += 256;
            if ceiling > limit {
                return tops;
            }
        }
        let mut left = 256;
        let top_cube = ceiling >> 4;
        for cy in (top_cube - 256..top_cube).rev() {
            let y0 = cy * 16;
            if y0 < crate::cubic::MIN_Y {
                break;
            }
            match class(y0, y0 + 15) {
                BoxClass::Air => continue,
                BoxClass::Solid => {
                    for t in tops.iter_mut().filter(|t| **t == NO_TOP) {
                        *t = y0 + 15;
                    }
                    break;
                }
                BoxClass::Mixed => {}
            }
            let lattice = Lattice::for_cube(&self.noise, &self.blend, self.sizes, [cx, cy, cz]);
            let mut found = [NO_TOP; 256];
            lattice.for_each(|lx, ly, lz, _, _, _, density| {
                let i = lz << 4 | lx;
                if density > 0.0 && tops[i] == NO_TOP {
                    found[i] = found[i].max(y0 + ly as i32);
                }
            });
            for (t, f) in tops.iter_mut().zip(found) {
                if *t == NO_TOP && f != NO_TOP {
                    *t = f;
                    left -= 1;
                }
            }
            if left == 0 {
                break;
            }
        }
        tops
    }

    fn surface_y(&self, x: i32, z: i32, from_y: i32) -> Option<i32> {
        let col = self.noise.column(x, z, self.blend.terrain_at(x, z), 0.0);
        let solid = |y: i32| self.noise.density(x, y, z, &col, 0.0) > 0.0;
        let exact = |y: i32| cell_at(&self.noise, &self.blend, self.sizes, x, y, z).0 > 0.0;
        let mut y = from_y;
        if solid(y) {
            // Inside terrain: walk up to its top.
            let limit = from_y + 512;
            while y < limit && solid(y + 1) {
                y += 1;
            }
            if y >= limit {
                return None;
            }
        } else {
            let limit = from_y - 4096;
            let an = self.analyzer();
            // Samples at `y`, `y - 8`, ...; the bound skips runs of them that
            // cannot hold a solid block (the same blocks, fewer samples).
            let empty = |hi: i32, n: i32| {
                an.density_range(&Query::column(x, z, hi - 8 * (n - 1), hi, col))
                    .is_some_and(|r| r.max <= 0.0)
            };
            while y > limit && !solid(y) {
                let mut hop = 1;
                while hop < 512 && empty(y, hop * 2) {
                    hop *= 2;
                }
                y -= 8 * hop;
            }
            if y <= limit {
                return None;
            }
            while solid(y + 1) {
                y += 1;
            }
        }
        // Refine with the interpolated field the blocks come from.
        let mut top = y;
        for t in (y - 8)..=(y + 8) {
            if exact(t) {
                top = t;
            }
        }
        Some(top)
    }
}

#[cfg(test)]
mod bounds_tests;
#[cfg(test)]
mod tests;
