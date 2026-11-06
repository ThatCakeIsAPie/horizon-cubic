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
//! the default replacers include the deepslate gradient, structures
//! (strongholds, villages, ...) and dungeons are not generated, `riverSize`
//! is unused.

use std::sync::Arc;

use horizon_core::position::SectionPos;

use crate::registry::BlockRegistry;

pub mod biomes;
pub mod carvers;
pub mod climate_biomes;
pub mod density;
pub mod flat;
pub mod populate;
pub mod replacers;
pub mod rng;
pub mod settings;
pub mod vegetation;
#[doc(hidden)]
pub mod testing;

pub use biomes::{BiomeColumns, FixedBiome};
pub use climate_biomes::ClimateBiomes;
pub use density::{NoiseTable, OctaveTable};
pub use populate::CubeWorld;
pub use settings::{CustomGeneratorSettings, FlatGeneratorSettings};

use biomes::{BiomeBlend, BiomeResolver, SURFACE_BLOCKS};
use density::{Lattice, TerrainNoise, cell_at};
use replacers::{BlockCtx, Replacers, resolve_block};
use settings::{BlockStateDesc, IntAabb, ReplacerConfig};

struct Terrain {
    sizes: [i32; 3],
    noise: TerrainNoise,
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
        Ok(Self {
            kind: Kind::Terrain(Box::new(Terrain {
                sizes: [
                    settings.noise_sample_size_x,
                    settings.noise_sample_size_y,
                    settings.noise_sample_size_z,
                ],
                noise: TerrainNoise::new(seed, settings),
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

impl Terrain {
    fn generate(&self, cube: SectionPos, out: &mut [u32; 4096]) {
        let lattice = Lattice::for_cube(
            &self.noise,
            &self.blend,
            self.sizes,
            [cube.x, cube.y, cube.z],
        );
        let biomes = self.blend.cube_biomes(cube);
        let (bx, by, bz) = (cube.x * 16, cube.y * 16, cube.z * 16);
        lattice.for_each(|lx, ly, lz, dx, dy, dz, density| {
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
            out[ly << 8 | lz << 4 | lx] = self.replacers.block(&ctx);
        });
        self.carvers.carve(cube, out);
    }

    fn block_at(&self, x: i32, y: i32, z: i32, resolution: i32) -> u32 {
        let (density, dx, dy, dz) = if resolution <= 1 {
            cell_at(&self.noise, &self.blend, self.sizes, x, y, z)
        } else {
            let mw = f64::from(resolution);
            let h = resolution.min(8);
            let d = |x: i32, y: i32, z: i32| {
                let col = self.noise.column(x, z, self.blend.terrain_at(x, z), mw);
                self.noise.density(x, y, z, &col, mw)
            };
            let v = d(x, y, z);
            let inv = 1.0 / f64::from(h);
            (
                v,
                (d(x + h, y, z) - v) * inv,
                (d(x, y + h, z) - v) * inv,
                (d(x, y, z + h) - v) * inv,
            )
        };
        let biome = self.blend.biome_at(x, z);
        self.replacers.block(&BlockCtx {
            x,
            y,
            z,
            dx,
            dy,
            dz,
            density,
            biome: &biome,
        })
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
            while y > limit && !solid(y) {
                y -= 8;
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
mod tests;
