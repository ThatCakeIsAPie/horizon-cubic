//! Numeric material conditions and their column/block evaluation caches.
//!
//! This worker owns its condition caches so they cannot be reused accidentally
//! against a different column context.
mod codec;
mod density;
mod library;
mod noise;
mod rule;
use super::{
    DecodeError,
    anchor::{VerticalAnchor, WorldGenerationContext},
    material_surface::MaterialSurface,
};
use crate::registry::BlockStateId;
pub use codec::{MaterialBiomeTag, MaterialConditionReference, MaterialConditionRegistry};
pub use density::MaterialDensitySources;
use horizon_core::noise::Noise;
use horizon_core::{
    position::BlockPos,
    random::{PositionalRandom, RandomSource},
};
pub use library::MaterialRuleLibrary;
pub use noise::MaterialNoiseSources;
use rule::CompiledRule;
pub use rule::{DensityGate, MaterialRule, MaterialRuleError, RunMemo};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

/// Surface inputs come from the material system. Implementations should keep
/// world configuration fixed for the lifetime of a condition worker.
pub trait MaterialSurfaceQueries {
    /// Bind a compiled density root, optionally sampling the expected chunk now.
    fn bind_density(
        &mut self,
        _sampler: Arc<dyn horizon_core::density::DensitySampler>,
        _eager: bool,
    ) -> Result<usize, DecodeError> {
        Err(DecodeError::Field("missing material density provider"))
    }
    /// `bind_density(sampler, true)` for a density read only where `gate`'s
    /// input is in range (or, with `gated`, that `gate` describes): the eager
    /// sample may stop above the highest such height and, given a `lattice`
    /// period of the sampler's volume samples, start at the last multiple
    /// of it at or below the lowest. With `gated`, `density_heights` also
    /// covers the heights outside it.
    fn bind_gated_density(
        &mut self,
        sampler: Arc<dyn horizon_core::density::DensitySampler>,
        _gate: &DensityGate,
        _gated: bool,
        _lattice: Option<i32>,
    ) -> Result<usize, DecodeError> {
        self.bind_density(sampler, true)
    }
    fn density(&mut self, _index: usize, _position: BlockPos) -> Result<f32, MaterialRuleError> {
        Err(MaterialRuleError::MissingDensityProvider)
    }
    /// What an eager sample already tells about `density(index, _)`: where it
    /// can be above zero (or NaN). `None` when nothing is known.
    fn density_heights(&self, _index: usize) -> Option<DensityHeights> {
        None
    }
    fn preliminary_surface(&mut self, _position: BlockPos) -> Result<f32, MaterialRuleError> {
        Err(MaterialRuleError::MissingDensityProvider)
    }
    fn possible_biomes(&self) -> Option<BTreeSet<String>> {
        None
    }
    fn biome(&mut self, _position: BlockPos) -> Result<Option<Arc<str>>, MaterialRuleError> {
        Err(MaterialRuleError::MissingBiomeProvider)
    }
    /// `biome` as the index `biome_index` gives its name, for queries that
    /// number their biomes: biome conditions then compare numbers.
    fn biome_id(&mut self, _position: BlockPos) -> Result<Option<u32>, MaterialRuleError> {
        Err(MaterialRuleError::MissingBiomeProvider)
    }
    /// `biome_id` at `position`, with the lowest height down to which the
    /// column has that biome at every block (`position.y` when not known;
    /// it need not look below `floor`).
    fn biome_id_down(
        &mut self,
        position: BlockPos,
        _floor: i32,
    ) -> Result<Option<(u32, i32)>, MaterialRuleError> {
        Ok(self.biome_id(position)?.map(|id| (id, position.y)))
    }
    /// `Some` when `biome_id` answers: the index of the biome `name` (`None`
    /// inside when no biome has that name).
    fn biome_index(&self, _name: &str) -> Option<Option<u32>> {
        None
    }
    fn melts_frozen_ocean(
        &mut self,
        _biome: &str,
        _position: BlockPos,
        _sea_level: i32,
    ) -> Result<bool, MaterialRuleError> {
        Err(MaterialRuleError::MissingBiomeProvider)
    }
    fn cold_enough_to_snow(
        &mut self,
        _biome: &str,
        _position: BlockPos,
        _sea_level: i32,
    ) -> Result<bool, MaterialRuleError> {
        Err(MaterialRuleError::MissingBiomeProvider)
    }
    fn bind_noise(&mut self, _name: &str) -> Result<Arc<dyn Noise>, DecodeError> {
        Err(DecodeError::Field("missing material noise provider"))
    }
    fn surface_depth(&mut self, x: i32, z: i32) -> i32;
    fn surface_secondary(&mut self, x: i32, z: i32) -> f64;
    fn band(&mut self, _position: BlockPos) -> Result<BlockStateId, MaterialRuleError> {
        Err(MaterialRuleError::MissingBandProvider)
    }
}
impl<T: MaterialSurfaceQueries + ?Sized> MaterialSurfaceQueries for &mut T {
    fn bind_density(
        &mut self,
        sampler: Arc<dyn horizon_core::density::DensitySampler>,
        eager: bool,
    ) -> Result<usize, DecodeError> {
        (**self).bind_density(sampler, eager)
    }
    fn bind_gated_density(
        &mut self,
        sampler: Arc<dyn horizon_core::density::DensitySampler>,
        gate: &DensityGate,
        gated: bool,
        lattice: Option<i32>,
    ) -> Result<usize, DecodeError> {
        (**self).bind_gated_density(sampler, gate, gated, lattice)
    }
    fn density(&mut self, index: usize, position: BlockPos) -> Result<f32, MaterialRuleError> {
        (**self).density(index, position)
    }
    fn density_heights(&self, index: usize) -> Option<DensityHeights> {
        (**self).density_heights(index)
    }
    fn preliminary_surface(&mut self, position: BlockPos) -> Result<f32, MaterialRuleError> {
        (**self).preliminary_surface(position)
    }
    fn possible_biomes(&self) -> Option<BTreeSet<String>> {
        (**self).possible_biomes()
    }
    fn biome(&mut self, position: BlockPos) -> Result<Option<Arc<str>>, MaterialRuleError> {
        (**self).biome(position)
    }
    fn biome_id(&mut self, position: BlockPos) -> Result<Option<u32>, MaterialRuleError> {
        (**self).biome_id(position)
    }
    fn biome_id_down(
        &mut self,
        position: BlockPos,
        floor: i32,
    ) -> Result<Option<(u32, i32)>, MaterialRuleError> {
        (**self).biome_id_down(position, floor)
    }
    fn biome_index(&self, name: &str) -> Option<Option<u32>> {
        (**self).biome_index(name)
    }
    fn cold_enough_to_snow(
        &mut self,
        biome: &str,
        position: BlockPos,
        sea: i32,
    ) -> Result<bool, MaterialRuleError> {
        (**self).cold_enough_to_snow(biome, position, sea)
    }
    fn melts_frozen_ocean(
        &mut self,
        biome: &str,
        position: BlockPos,
        sea: i32,
    ) -> Result<bool, MaterialRuleError> {
        (**self).melts_frozen_ocean(biome, position, sea)
    }
    fn bind_noise(&mut self, name: &str) -> Result<Arc<dyn Noise>, DecodeError> {
        (**self).bind_noise(name)
    }
    fn surface_depth(&mut self, x: i32, z: i32) -> i32 {
        (**self).surface_depth(x, z)
    }
    fn surface_secondary(&mut self, x: i32, z: i32) -> f64 {
        (**self).surface_secondary(x, z)
    }
    fn band(&mut self, position: BlockPos) -> Result<BlockStateId, MaterialRuleError> {
        (**self).band(position)
    }
}
/// Inside the inclusive block box `min..=max`, a density is `<= 0` at every
/// height outside `y[0]..=y[1]` (an empty range when `y[0] > y[1]`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DensityHeights {
    pub min: [i32; 3],
    pub max: [i32; 3],
    pub y: [i32; 2],
}
impl DensityHeights {
    /// Whether the density at `p` is known to be `<= 0`.
    #[inline]
    pub fn never_positive(&self, p: BlockPos) -> bool {
        (p.y < self.y[0] || p.y > self.y[1])
            && (self.min[0]..=self.max[0]).contains(&p.x)
            && (self.min[1]..=self.max[1]).contains(&p.y)
            && (self.min[2]..=self.max[2]).contains(&p.z)
    }
}
impl<P: PositionalRandom> MaterialSurfaceQueries for MaterialSurface<P> {
    fn band(&mut self, position: BlockPos) -> Result<BlockStateId, MaterialRuleError> {
        MaterialSurface::band(self, [position.x, position.y, position.z])
            .ok_or(MaterialRuleError::InvalidBandIndex)
    }
    fn surface_depth(&mut self, x: i32, z: i32) -> i32 {
        MaterialSurface::surface_depth(self, x, z)
    }
    fn surface_secondary(&mut self, x: i32, z: i32) -> f64 {
        MaterialSurface::surface_secondary(self, x, z)
    }
}

#[derive(Clone, Debug)]
pub enum MaterialCondition {
    Hole,
    Steep,
    Biome(Vec<String>),
    BiomeTag(MaterialBiomeTag),
    Temperature,
    AbovePreliminarySurface,
    NoiseThreshold {
        noise: String,
        min: f64,
        max: f64,
        is_3d: bool,
    },
    YAbove {
        anchor: VerticalAnchor,
        surface_depth_multiplier: i32,
        add_stone_depth: bool,
    },
    Water {
        offset: i32,
        surface_depth_multiplier: i32,
        add_stone_depth: bool,
    },
    StoneDepth {
        offset: i32,
        add_surface_depth: bool,
        secondary_depth_range: i32,
        ceiling: bool,
    },
    VerticalGradient {
        random_name: String,
        true_at_and_below: VerticalAnchor,
        false_at_and_above: VerticalAnchor,
    },
    Not(Box<MaterialCondition>),
    Reference(MaterialConditionReference),
}

struct Context<S> {
    surface: S,
    generation: WorldGenerationContext,
    column_revision: i64,
    block_revision: i64,
    position: BlockPos,
    gradient: [i32; 2],
    depth: i32,
    water_height: i32,
    stone_above: i32,
    stone_below: i32,
    secondary_revision: i64,
    secondary: f64,
    minimum_revision: i64,
    minimum: i32,
    noises: Vec<CachedNoise>,
    noise_indices: BTreeMap<(String, bool), usize>,
    possible_biomes: Option<BTreeSet<String>>,
    biome: Option<Arc<str>>,
    biome_id: Option<u32>,
}
struct CachedNoise {
    sampler: Arc<dyn Noise>,
    is_3d: bool,
    revision: i64,
    value: f64,
    /// 3D noises: values down the current column from `column_top`, sampled
    /// together (the surface walks each column down in y).
    column: [f32; NOISE_COLUMN],
    column_top: i32,
    column_revision: i64,
}
const NOISE_COLUMN: usize = 16;
impl<S: MaterialSurfaceQueries> Context<S> {
    fn minimum_surface_level(&mut self) -> Result<i32, MaterialRuleError> {
        if self.minimum_revision != self.column_revision {
            // Vanilla stamps before sampling, including when sampling fails.
            self.minimum_revision = self.column_revision;
            let value = self.surface.preliminary_surface(BlockPos::new(
                self.position.x,
                0,
                self.position.z,
            ))?;
            // 26.3 floors before the saturating Java numeric conversion.
            let floor = value.floor() as i32;
            self.minimum = floor.wrapping_add(self.depth).wrapping_sub(8);
        }
        Ok(self.minimum)
    }
    fn biome(&mut self) -> Result<Option<Arc<str>>, MaterialRuleError> {
        if self.biome.is_none() {
            self.biome = self.surface.biome(self.position)?;
        }
        Ok(self.biome.clone())
    }
    fn biome_id(&mut self) -> Result<Option<u32>, MaterialRuleError> {
        if self.biome_id.is_none() {
            self.biome_id = self.surface.biome_id(self.position)?;
        }
        Ok(self.biome_id)
    }
    fn bind_noise(&mut self, name: &str, is_3d: bool) -> Result<usize, DecodeError> {
        let key = (name.to_owned(), is_3d);
        if let Some(index) = self.noise_indices.get(&key) {
            return Ok(*index);
        }
        let sampler = self.surface.bind_noise(name)?;
        let index = self.noises.len();
        self.noises.push(CachedNoise {
            sampler,
            is_3d,
            revision: if is_3d {
                self.block_revision
            } else {
                self.column_revision
            }
            .wrapping_sub(1),
            value: 0.0,
            column: [0.0; NOISE_COLUMN],
            column_top: 0,
            column_revision: self.column_revision.wrapping_sub(1),
        });
        self.noise_indices.insert(key, index);
        Ok(index)
    }
    fn noise(&mut self, index: usize) -> f64 {
        let noise = &mut self.noises[index];
        if noise.is_3d {
            // A pure function of the position: values sampled ahead for the
            // blocks below are the ones `get3` would give there.
            let y = self.position.y;
            let below = i64::from(noise.column_top) - i64::from(y);
            if noise.column_revision != self.column_revision
                || !(0..NOISE_COLUMN as i64).contains(&below)
            {
                let ys: [f64; NOISE_COLUMN] =
                    std::array::from_fn(|i| f64::from(y.wrapping_sub(i as i32)));
                noise.sampler.get3_column(
                    f64::from(self.position.x),
                    f64::from(self.position.z),
                    &ys,
                    1.0,
                    &mut noise.column,
                );
                noise.column_top = y;
                noise.column_revision = self.column_revision;
                return f64::from(noise.column[0]);
            }
            return f64::from(noise.column[below as usize]);
        }
        let revision = if noise.is_3d {
            self.block_revision
        } else {
            self.column_revision
        };
        if noise.revision != revision {
            noise.value = f64::from(noise.sampler.get3(
                f64::from(self.position.x),
                if noise.is_3d {
                    f64::from(self.position.y)
                } else {
                    0.0
                },
                f64::from(self.position.z),
            ));
            noise.revision = revision;
        }
        noise.value
    }
    fn secondary(&mut self) -> f64 {
        if self.secondary_revision != self.column_revision {
            self.secondary_revision = self.column_revision;
            self.secondary = self
                .surface
                .surface_secondary(self.position.x, self.position.z);
        }
        self.secondary
    }
}
/// What a rule program's run at one block depended on (`CompiledRule::apply_traced`).
struct RunTrace {
    /// The result holds down to this height of the run...
    low: i64,
    /// ...when the block's biome is the same (if `biome`)...
    biome: bool,
    /// ...and these tests (condition index, value), which vary block by
    /// block as pure functions of the position, give the same values...
    guards: [(u32, bool); RUN_GUARDS],
    guard_count: usize,
    /// ...unless something else tested varies block by block.
    varies: bool,
}
const RUN_GUARDS: usize = 4;
struct Compiled<R: RandomSource> {
    kind: Kind<R>,
    revision: i64,
    result: Option<bool>,
}
enum Kind<R: RandomSource> {
    Simple(MaterialCondition),
    Biome(BTreeSet<String>),
    /// `Biome` for queries that number their biomes: a bit per index.
    BiomeIds(Vec<u64>),
    Constant(bool),
    Noise {
        sampler: usize,
        min: f64,
        max: f64,
    },
    Gradient {
        low: i32,
        high: i32,
        random: R::Positional,
    },
    Not(Box<Compiled<R>>),
}
impl<R: RandomSource> Compiled<R> {
    fn constant_result(&self) -> Option<bool> {
        match &self.kind {
            Kind::Constant(value) => Some(*value),
            Kind::Not(child) => child.constant_result().map(|value| !value),
            _ => None,
        }
    }

    fn compile<S: MaterialSurfaceQueries>(
        condition: MaterialCondition,
        root: &R::Positional,
        context: &mut Context<S>,
        depth: usize,
    ) -> Result<Self, DecodeError> {
        if depth >= 256 {
            return Err(DecodeError::Field("material condition depth"));
        }
        let column = matches!(
            condition,
            MaterialCondition::Hole | MaterialCondition::Steep
        );
        let kind = match condition {
            MaterialCondition::BiomeTag(tag) => {
                return Self::compile(
                    MaterialCondition::Biome(tag.members().to_vec()),
                    root,
                    context,
                    depth + 1,
                );
            }
            MaterialCondition::Biome(names) => {
                if names.len() > 8192 {
                    return Err(DecodeError::Field("biome condition size"));
                }
                let names = names
                    .iter()
                    .map(|name| codec::identifier(name))
                    .collect::<Result<BTreeSet<_>, _>>()?;
                match &context.possible_biomes {
                    Some(possible) if names.is_disjoint(possible) => Kind::Constant(false),
                    Some(possible) if possible.is_subset(&names) => Kind::Constant(true),
                    _ => match names
                        .iter()
                        .map(|name| context.surface.biome_index(name))
                        .collect::<Option<Vec<_>>>()
                    {
                        // Names no biome has never match the block's biome.
                        Some(ids) => {
                            let mut bits = Vec::new();
                            for id in ids.into_iter().flatten() {
                                let word = id as usize / 64;
                                if bits.len() <= word {
                                    bits.resize(word + 1, 0);
                                }
                                bits[word] |= 1 << (id % 64);
                            }
                            Kind::BiomeIds(bits)
                        }
                        None => Kind::Biome(names),
                    },
                }
            }
            MaterialCondition::NoiseThreshold {
                noise,
                min,
                max,
                is_3d,
            } => Kind::Noise {
                sampler: context.bind_noise(&codec::identifier(&noise)?, is_3d)?,
                min,
                max,
            },
            MaterialCondition::Reference(reference) => {
                return Self::compile(reference.value().clone(), root, context, depth + 1);
            }
            MaterialCondition::Not(child) => {
                Kind::Not(Box::new(Self::compile(*child, root, context, depth + 1)?))
            }
            MaterialCondition::VerticalGradient {
                random_name,
                true_at_and_below,
                false_at_and_above,
            } => {
                let random_name = codec::identifier(&random_name)?;
                Kind::Gradient {
                    low: true_at_and_below.resolve(context.generation),
                    high: false_at_and_above.resolve(context.generation),
                    random: root.from_hash_of(&random_name).fork_positional(),
                }
            }
            other => Kind::Simple(other),
        };
        Ok(Self {
            kind,
            revision: if column {
                context.column_revision
            } else {
                context.block_revision
            }
            .wrapping_sub(1),
            result: None,
        })
    }
    /// Test with vanilla's per-revision result cache (standalone conditions).
    fn test<S: MaterialSurfaceQueries>(
        &mut self,
        context: &mut Context<S>,
    ) -> Result<bool, MaterialRuleError> {
        self.eval::<true, S>(context)
    }
    /// Test inside a compiled rule program, which reaches each node at most
    /// once per block: a block-level result is never asked for twice, and
    /// column-level ones (hole, steep) are pure in the column's state, so the
    /// revision cache would change nothing.
    ///
    /// Inlined into the rule program's loop: `eval`'s steps without its
    /// recursion (an inversion only flips the result).
    #[inline(always)]
    fn test_once<S: MaterialSurfaceQueries>(
        &self,
        context: &mut Context<S>,
    ) -> Result<bool, MaterialRuleError> {
        let mut node = self;
        let mut invert = false;
        while let Kind::Not(child) = &node.kind {
            invert = !invert;
            node = child;
        }
        let value = match &node.kind {
            Kind::Constant(value) => *value,
            Kind::Noise { sampler, min, max } => {
                let value = context.noise(*sampler);
                value >= *min && value <= *max
            }
            Kind::Simple(MaterialCondition::AbovePreliminarySurface) => {
                context.position.y >= context.minimum_surface_level()?
            }
            Kind::Simple(MaterialCondition::Temperature) => {
                let biome = context.biome()?.ok_or(MaterialRuleError::MissingBiome)?;
                context.surface.cold_enough_to_snow(
                    &biome,
                    context.position,
                    context.generation.sea_level,
                )?
            }
            _ => node.compute(context)?,
        };
        Ok(value != invert)
    }
    /// After `test_once` gave `value` at the context's block: raise `run.low`
    /// to the lowest height down the current run of solid blocks
    /// (y falls by one and the stone above grows by one per block; water,
    /// the bottom of the run and the column stay) where the test still gives
    /// `value`. Marks `run` as varying when that isn't a plain height bound.
    ///
    /// Returns true, leaving `run` as it is, for a test that varies block by
    /// block as a pure function of the position (3D noise, a gradient inside
    /// its band): the caller can test it again at another block.
    fn stable_below<S: MaterialSurfaceQueries>(
        &self,
        value: bool,
        context: &mut Context<S>,
        run: &mut RunTrace,
    ) -> bool {
        let mut node = self;
        let mut value = value;
        while let Kind::Not(child) = &node.kind {
            value = !value;
            node = child;
        }
        let y = i64::from(context.position.y);
        // Lowest height of the run giving `value`, for a test `y >= bound`.
        let at_least = |bound: i32| if value { i64::from(bound) } else { i64::MIN };
        let low = match &node.kind {
            Kind::Constant(_) => i64::MIN,
            Kind::BiomeIds(_) => {
                run.biome = true;
                i64::MIN
            }
            Kind::Noise { sampler, .. } => {
                if context.noises[*sampler].is_3d {
                    return true;
                }
                i64::MIN
            }
            Kind::Gradient { low, high, .. } => {
                if context.position.y <= *low {
                    i64::MIN
                } else if context.position.y >= *high {
                    i64::from(*high).max(i64::from(*low) + 1)
                } else {
                    return true;
                }
            }
            Kind::Simple(MaterialCondition::Hole | MaterialCondition::Steep) => i64::MIN,
            Kind::Simple(MaterialCondition::AbovePreliminarySurface) => {
                // Column-cached: computed by the test that gave `value`.
                match context.minimum_surface_level() {
                    Ok(minimum) => at_least(minimum),
                    Err(_) => {
                        run.varies = true;
                        i64::MIN
                    }
                }
            }
            Kind::Simple(MaterialCondition::YAbove {
                anchor,
                surface_depth_multiplier,
                add_stone_depth,
            }) => {
                if *add_stone_depth {
                    // y + stone above is the same down the run.
                    i64::MIN
                } else {
                    at_least(
                        anchor
                            .resolve(context.generation)
                            .wrapping_add(context.depth.wrapping_mul(*surface_depth_multiplier)),
                    )
                }
            }
            Kind::Simple(MaterialCondition::Water {
                offset,
                surface_depth_multiplier,
                add_stone_depth,
            }) => {
                if context.water_height == i32::MIN || *add_stone_depth {
                    i64::MIN
                } else {
                    at_least(
                        context
                            .water_height
                            .wrapping_add(*offset)
                            .wrapping_add(context.depth.wrapping_mul(*surface_depth_multiplier)),
                    )
                }
            }
            Kind::Simple(MaterialCondition::StoneDepth {
                offset,
                add_surface_depth,
                secondary_depth_range,
                ceiling,
            }) => {
                let surface = if *add_surface_depth { context.depth } else { 0 };
                let secondary = if *secondary_depth_range == 0 {
                    0
                } else {
                    (0.0 + ((context.secondary() - -1.0) / (1.0 - -1.0))
                        * (f64::from(*secondary_depth_range) - 0.0)) as i32
                };
                let limit = i64::from(
                    1i32.wrapping_add(*offset)
                        .wrapping_add(surface)
                        .wrapping_add(secondary),
                );
                if *ceiling {
                    // The stone below shrinks down the run: once within the
                    // limit, it stays.
                    if value {
                        i64::MIN
                    } else {
                        y - (i64::from(context.stone_below) - limit) + 1
                    }
                } else if value {
                    // The stone above grows down the run.
                    y - (limit - i64::from(context.stone_above))
                } else {
                    i64::MIN
                }
            }
            _ => {
                run.varies = true;
                i64::MIN
            }
        };
        run.low = run.low.max(low);
        false
    }
    #[inline]
    fn eval<const CACHE: bool, S: MaterialSurfaceQueries>(
        &mut self,
        context: &mut Context<S>,
    ) -> Result<bool, MaterialRuleError> {
        let column = match &mut self.kind {
            Kind::Constant(value) => return Ok(*value),
            Kind::Noise { sampler, min, max } => {
                let value = context.noise(*sampler);
                return Ok(value >= *min && value <= *max);
            }
            Kind::Not(child) => return child.eval::<CACHE, S>(context).map(|value| !value),
            Kind::Simple(MaterialCondition::AbovePreliminarySurface) => {
                return Ok(context.position.y >= context.minimum_surface_level()?);
            }
            Kind::Simple(MaterialCondition::Temperature) => {
                let biome = context.biome()?.ok_or(MaterialRuleError::MissingBiome)?;
                return context.surface.cold_enough_to_snow(
                    &biome,
                    context.position,
                    context.generation.sea_level,
                );
            }
            Kind::Simple(MaterialCondition::Hole | MaterialCondition::Steep) => true,
            _ => false,
        };
        if !CACHE {
            return self.compute(context);
        }
        let revision = if column {
            context.column_revision
        } else {
            context.block_revision
        };
        if self.revision == revision {
            return self
                .result
                .ok_or(MaterialRuleError::CachedConditionUnavailable);
        }
        self.revision = revision;
        let result = self.compute(context)?;
        self.result = Some(result);
        Ok(result)
    }
    #[inline]
    fn compute<S: MaterialSurfaceQueries>(
        &self,
        context: &mut Context<S>,
    ) -> Result<bool, MaterialRuleError> {
        let result = match &self.kind {
            Kind::Biome(names) => names.contains(
                context
                    .biome()?
                    .ok_or(MaterialRuleError::MissingBiome)?
                    .as_ref(),
            ),
            Kind::BiomeIds(bits) => {
                let id = context.biome_id()?.ok_or(MaterialRuleError::MissingBiome)?;
                bits.get(id as usize / 64)
                    .is_some_and(|word| word >> (id % 64) & 1 == 1)
            }
            Kind::Simple(MaterialCondition::Hole) => context.depth <= 0,
            Kind::Simple(MaterialCondition::Steep) => {
                context.gradient[0] <= -4 || context.gradient[1] >= 4
            }
            Kind::Simple(MaterialCondition::YAbove {
                anchor,
                surface_depth_multiplier,
                add_stone_depth,
            }) => {
                let y = context.position.y.wrapping_add(if *add_stone_depth {
                    context.stone_above
                } else {
                    0
                });
                y >= anchor
                    .resolve(context.generation)
                    .wrapping_add(context.depth.wrapping_mul(*surface_depth_multiplier))
            }
            Kind::Simple(MaterialCondition::Water {
                offset,
                surface_depth_multiplier,
                add_stone_depth,
            }) => {
                context.water_height == i32::MIN
                    || context.position.y.wrapping_add(if *add_stone_depth {
                        context.stone_above
                    } else {
                        0
                    }) >= context
                        .water_height
                        .wrapping_add(*offset)
                        .wrapping_add(context.depth.wrapping_mul(*surface_depth_multiplier))
            }
            Kind::Simple(MaterialCondition::StoneDepth {
                offset,
                add_surface_depth,
                secondary_depth_range,
                ceiling,
            }) => {
                let stone = if *ceiling {
                    context.stone_below
                } else {
                    context.stone_above
                };
                let surface = if *add_surface_depth { context.depth } else { 0 };
                let secondary = if *secondary_depth_range == 0 {
                    0
                } else {
                    (0.0 + ((context.secondary() - -1.0) / (1.0 - -1.0))
                        * (f64::from(*secondary_depth_range) - 0.0)) as i32
                };
                stone
                    <= 1i32
                        .wrapping_add(*offset)
                        .wrapping_add(surface)
                        .wrapping_add(secondary)
            }
            Kind::Gradient { low, high, random } => {
                let y = context.position.y;
                if y <= *low {
                    true
                } else if y >= *high {
                    false
                } else {
                    let probability = 1.0
                        + ((f64::from(y) - f64::from(*low)) / (f64::from(*high) - f64::from(*low)))
                            * (0.0 - 1.0);
                    f64::from(random.at(context.position).next_float()) < probability
                }
            }
            _ => unreachable!("all recursive and gradient definitions are compiled"),
        };
        Ok(result)
    }
}

/// A thread-local material-condition worker. Surface state and evaluations are
/// owned together; updates invalidate exactly the corresponding vanilla caches.
pub struct MaterialConditions<R: RandomSource, S: MaterialSurfaceQueries> {
    root: R::Positional,
    context: Context<S>,
    conditions: Vec<Compiled<R>>,
    rules: Vec<CompiledRule<R>>,
}
impl<R: RandomSource, S: MaterialSurfaceQueries> MaterialConditions<R, S> {
    pub fn new(generation: WorldGenerationContext, root: R::Positional, surface: S) -> Self {
        let possible_biomes = surface.possible_biomes();
        Self::with_possible_biomes(generation, root, surface, possible_biomes)
    }
    pub(crate) fn with_possible_biomes(
        generation: WorldGenerationContext,
        root: R::Positional,
        surface: S,
        possible_biomes: Option<BTreeSet<String>>,
    ) -> Self {
        Self {
            root,
            context: Context {
                surface,
                generation,
                column_revision: i64::MIN + 1,
                block_revision: i64::MIN + 1,
                position: BlockPos::new(0, 0, 0),
                gradient: [0, 0],
                depth: 0,
                water_height: 0,
                stone_above: 0,
                stone_below: 0,
                secondary_revision: i64::MIN,
                secondary: 0.0,
                minimum_revision: i64::MIN,
                minimum: 0,
                noises: Vec::new(),
                noise_indices: BTreeMap::new(),
                possible_biomes,
                biome: None,
                biome_id: None,
            },
            conditions: Vec::new(),
            rules: Vec::new(),
        }
    }
    /// Compile a condition at the current context revision. Gradient bounds and
    /// named random factories are bound now, before the first evaluation.
    pub fn add(&mut self, condition: MaterialCondition) -> Result<usize, DecodeError> {
        if self.conditions.len() >= 8192 {
            return Err(DecodeError::Field("material condition count"));
        }
        let compiled = Compiled::compile(condition, &self.root, &mut self.context, 0)?;
        let index = self.conditions.len();
        self.conditions.push(compiled);
        Ok(index)
    }
    pub fn update_xz(&mut self, x: i32, z: i32, gradient_x: i32, gradient_z: i32) {
        self.context.column_revision = self.context.column_revision.wrapping_add(1);
        self.context.block_revision = self.context.block_revision.wrapping_add(1);
        self.context.position.x = x;
        self.context.position.z = z;
        self.context.gradient = [gradient_x, gradient_z];
        self.context.depth = self.context.surface.surface_depth(x, z);
    }
    pub fn update_y(&mut self, stone_above: i32, stone_below: i32, water_height: i32, y: i32) {
        self.context.block_revision = self.context.block_revision.wrapping_add(1);
        self.context.biome = None;
        self.context.biome_id = None;
        self.context.position.y = y;
        self.context.stone_above = stone_above;
        self.context.stone_below = stone_below;
        self.context.water_height = water_height;
    }
    pub fn test(&mut self, index: usize) -> Result<bool, MaterialRuleError> {
        self.conditions
            .get_mut(index)
            .ok_or(MaterialRuleError::UnknownCondition)?
            .test(&mut self.context)
    }
    /// Direct biome lookup for the outer column scan, independent of condition caching.
    pub fn biome_at(&mut self, position: BlockPos) -> Result<Option<Arc<str>>, MaterialRuleError> {
        self.context.surface.biome(position)
    }
    pub fn melts_frozen_ocean(
        &mut self,
        biome: &str,
        position: BlockPos,
        sea_level: i32,
    ) -> Result<bool, MaterialRuleError> {
        self.context
            .surface
            .melts_frozen_ocean(biome, position, sea_level)
    }
    pub fn minimum_surface_level(&mut self) -> Result<i32, MaterialRuleError> {
        self.context.minimum_surface_level()
    }
    pub fn into_surface(self) -> S {
        self.context.surface
    }
    pub fn surface_depth(&self) -> i32 {
        self.context.depth
    }
    pub fn surface_secondary(&mut self) -> f64 {
        self.context.secondary()
    }
}
