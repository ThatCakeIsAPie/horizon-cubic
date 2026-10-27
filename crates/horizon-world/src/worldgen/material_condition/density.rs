//! Preliminary surface density sampling scoped to one material context.
use super::{DensityGate, DensityHeights, MaterialRuleError, MaterialSurfaceQueries};
use crate::{registry::BlockStateId, worldgen::DecodeError};
use horizon_core::{
    density::{DensitySampler, SamplerContext},
    noise::{DensityVolume, Noise},
    position::BlockPos,
};
use std::{collections::BTreeSet, sync::Arc};

/// A lazy unit-spaced XZ plane and scalar fallback for preliminary heights.
/// Construct a fresh provider for each material context, with that world's
/// compiled preliminary-surface density root and bound sampler context.
pub struct MaterialDensitySources<S: MaterialSurfaceQueries> {
    surface: S,
    expected: DensityVolume,
    densities: Vec<BoundDensity>,
    /// Eagerly sampled volumes, shared by bindings of the same density root.
    buffers: Vec<Eager>,
    retained_samples: usize,
    plane: DensityVolume,
    sampler: Arc<dyn DensitySampler>,
    context: SamplerContext,
    buffer: Option<Vec<f32>>,
}
struct BoundDensity {
    sampler: Arc<dyn DensitySampler>,
    buffer: Option<usize>,
    /// Never above zero in `expected` above its buffer's volume.
    gated: bool,
}
/// An eager sample over `volume`: `expected`, or its lower part (same
/// minimum, so interpolated densities give the same values), with the
/// heights where it is above zero or NaN.
struct Eager {
    samples: Vec<f32>,
    volume: DensityVolume,
    heights: [i32; 2],
    /// Not sampled until first read (`samples` empty).
    pending: bool,
}
impl<S: MaterialSurfaceQueries> MaterialDensitySources<S> {
    pub fn sampler_context_mut(&mut self) -> &mut SamplerContext {
        &mut self.context
    }
    pub fn into_parts(self) -> (S, SamplerContext) {
        (self.surface, self.context)
    }
    pub fn new(
        surface: S,
        expected: DensityVolume,
        sampler: Arc<dyn DensitySampler>,
        context: SamplerContext,
    ) -> Result<Self, DecodeError> {
        let [sx, _, sz] = expected.sizes();
        // Bound retained allocation. Normal chunk planes contain 256 samples.
        if sx * sz > 1 << 20 {
            return Err(DecodeError::Field("material preliminary plane size"));
        }
        let [x, _, z] = expected.minimum();
        let plane = DensityVolume::new([sx as i32, 1, sz as i32], [x, 0, z], [1; 3])
            .map_err(|_| DecodeError::Field("material preliminary plane"))?;
        Ok(Self {
            surface,
            expected,
            densities: Vec::new(),
            buffers: Vec::new(),
            retained_samples: 0,
            plane,
            sampler,
            context,
            buffer: None,
        })
    }
}
impl<S: MaterialSurfaceQueries> MaterialDensitySources<S> {
    /// Bind `sampler`, sampled now over `volume` when given (a part of
    /// `expected` with its minimum).
    fn bind(
        &mut self,
        sampler: Arc<dyn DensitySampler>,
        volume: Option<DensityVolume>,
        gated: bool,
        defer: bool,
    ) -> Result<usize, DecodeError> {
        if self.densities.len() >= 32768
            || volume.is_some_and(|volume| {
                volume.size() > 1 << 20 || self.retained_samples + volume.size() > 1 << 24
            })
        {
            return Err(DecodeError::Field("material density storage bound"));
        }
        let buffer = match volume {
            None => None,
            Some(volume) => {
                // Sampling a root over the same lattice and context is
                // deterministic, so another binding (iron and copper veins'
                // richness) reuses a sample covering its volume.
                let shared = self
                    .densities
                    .iter()
                    .filter(|bound| Arc::ptr_eq(&bound.sampler, &sampler))
                    .filter_map(|bound| bound.buffer)
                    .find(|&buffer| {
                        let known = self.buffers[buffer].volume;
                        known.minimum() == volume.minimum()
                            && known.steps() == volume.steps()
                            && (0..3).all(|i| known.sizes()[i] >= volume.sizes()[i])
                    });
                match shared {
                    Some(buffer) => Some(buffer),
                    None if defer => {
                        self.retained_samples += volume.size();
                        self.buffers.push(Eager {
                            samples: Vec::new(),
                            volume,
                            heights: [i32::MAX, i32::MIN],
                            pending: true,
                        });
                        Some(self.buffers.len() - 1)
                    }
                    None => {
                        let mut samples = vec![0.0; volume.size()];
                        sampler.sample_volume(&mut self.context, &mut samples, volume)?;
                        self.retained_samples += samples.len();
                        let heights = positive_heights(volume, &samples);
                        self.buffers.push(Eager {
                            samples,
                            volume,
                            heights,
                            pending: false,
                        });
                        Some(self.buffers.len() - 1)
                    }
                }
            }
        };
        let index = self.densities.len();
        self.densities.push(BoundDensity {
            sampler,
            buffer,
            gated,
        });
        Ok(index)
    }
}
impl<S: MaterialSurfaceQueries> MaterialSurfaceQueries for MaterialDensitySources<S> {
    fn bind_density(
        &mut self,
        sampler: Arc<dyn DensitySampler>,
        eager: bool,
    ) -> Result<usize, DecodeError> {
        let volume = eager.then_some(self.expected);
        self.bind(sampler, volume, false, false)
    }
    fn bind_gated_density(
        &mut self,
        sampler: Arc<dyn DensitySampler>,
        gate: &DensityGate,
        gated: bool,
        lattice: Option<i32>,
    ) -> Result<usize, DecodeError> {
        if self.expected.steps() != [1; 3] {
            return self.bind_density(sampler, true);
        }
        let [sx, sy, sz] = self.expected.sizes();
        let minimum = self.expected.minimum();
        // The gate's input varies with Y alone: one column gives every height.
        let column = DensityVolume::new([1, sy as i32, 1], minimum, [1; 3])
            .map_err(|_| DecodeError::Field("material density gate"))?;
        let mut inputs = vec![0.0; column.size()];
        gate.input
            .sample_volume(&mut self.context, &mut inputs, column)?;
        let in_range = |v: &f32| *v >= gate.min_inclusive && *v < gate.max_exclusive;
        let top = inputs.iter().rposition(in_range).unwrap_or(0);
        // Rows below the lowest in-range height can go too, by whole periods.
        let bottom = match lattice {
            Some(period) if period > 0 => {
                let low = inputs.iter().position(in_range).unwrap_or(0);
                low - low % period as usize
            }
            _ => 0,
        };
        let volume = DensityVolume::new(
            [sx as i32, (top - bottom) as i32 + 1, sz as i32],
            [minimum[0], minimum[1] + bottom as i32, minimum[2]],
            [1; 3],
        )
        .map_err(|_| DecodeError::Field("material density gate"))?;
        // A richness (not `gated`) is only read where its vein's density is
        // above zero: many chunks never read it.
        self.bind(sampler, Some(volume), gated, !gated)
    }
    fn density(&mut self, index: usize, position: BlockPos) -> Result<f32, MaterialRuleError> {
        let source = self
            .densities
            .get(index)
            .ok_or(MaterialRuleError::MissingDensityProvider)?;
        let position = [position.x, position.y, position.z];
        if let Some(buffer) = source.buffer {
            if self.buffers[buffer].pending {
                // Sampling is deterministic: the values eager sampling gives.
                let volume = self.buffers[buffer].volume;
                let mut samples = vec![0.0; volume.size()];
                source
                    .sampler
                    .sample_volume(&mut self.context, &mut samples, volume)?;
                let eager = &mut self.buffers[buffer];
                eager.heights = positive_heights(volume, &samples);
                eager.samples = samples;
                eager.pending = false;
            }
            let eager = &self.buffers[buffer];
            if let Some(index) = block_index(eager.volume, position) {
                return Ok(eager.samples[index]);
            }
        }
        Ok(source.sampler.sample_value(&mut self.context, position)?)
    }
    fn density_heights(&self, index: usize) -> Option<DensityHeights> {
        let bound = self.densities.get(index)?;
        let eager = &self.buffers[bound.buffer?];
        if self.expected.steps() != [1; 3] {
            return None;
        }
        let box_volume = if bound.gated {
            self.expected
        } else {
            eager.volume
        };
        let min = box_volume.minimum();
        let sizes = box_volume.sizes();
        Some(DensityHeights {
            min,
            max: std::array::from_fn(|i| min[i].wrapping_add(sizes[i] as i32 - 1)),
            y: eager.heights,
        })
    }
    fn preliminary_surface(&mut self, position: BlockPos) -> Result<f32, MaterialRuleError> {
        let position = [position.x, 0, position.z];
        if let Some(index) = self.plane.index_of_block(position) {
            if self.buffer.is_none() {
                // Retain the allocation and partial writes if sampling fails.
                let buffer = self.buffer.insert(vec![0.0; self.plane.size()]);
                self.sampler
                    .sample_volume(&mut self.context, buffer, self.plane)?;
            }
            Ok(self.buffer.as_ref().unwrap()[index])
        } else {
            Ok(self.sampler.sample_value(&mut self.context, position)?)
        }
    }
    fn possible_biomes(&self) -> Option<BTreeSet<String>> {
        self.surface.possible_biomes()
    }
    fn biome(&mut self, position: BlockPos) -> Result<Option<Arc<str>>, MaterialRuleError> {
        self.surface.biome(position)
    }
    fn biome_id(&mut self, position: BlockPos) -> Result<Option<u32>, MaterialRuleError> {
        self.surface.biome_id(position)
    }
    fn biome_id_down(
        &mut self,
        position: BlockPos,
        floor: i32,
    ) -> Result<Option<(u32, i32)>, MaterialRuleError> {
        self.surface.biome_id_down(position, floor)
    }
    fn biome_index(&self, name: &str) -> Option<Option<u32>> {
        self.surface.biome_index(name)
    }
    fn melts_frozen_ocean(
        &mut self,
        biome: &str,
        position: BlockPos,
        sea_level: i32,
    ) -> Result<bool, MaterialRuleError> {
        self.surface.melts_frozen_ocean(biome, position, sea_level)
    }
    fn cold_enough_to_snow(
        &mut self,
        biome: &str,
        position: BlockPos,
        sea_level: i32,
    ) -> Result<bool, MaterialRuleError> {
        self.surface.cold_enough_to_snow(biome, position, sea_level)
    }
    fn bind_noise(&mut self, name: &str) -> Result<Arc<dyn Noise>, DecodeError> {
        self.surface.bind_noise(name)
    }
    fn surface_depth(&mut self, x: i32, z: i32) -> i32 {
        self.surface.surface_depth(x, z)
    }
    fn surface_secondary(&mut self, x: i32, z: i32) -> f64 {
        self.surface.surface_secondary(x, z)
    }
    fn band(&mut self, position: BlockPos) -> Result<BlockStateId, MaterialRuleError> {
        self.surface.band(position)
    }
}

/// The lowest and highest block heights of a unit-step volume holding a
/// sample above zero or NaN (`[i32::MAX, i32::MIN]` when none does).
fn positive_heights(volume: DensityVolume, samples: &[f32]) -> [i32; 2] {
    let sy = volume.sizes()[1];
    let mut range = [usize::MAX, 0];
    for column in samples.chunks_exact(sy) {
        if let Some(low) = column.iter().position(|v| *v > 0.0 || v.is_nan()) {
            let high = column.iter().rposition(|v| *v > 0.0 || v.is_nan()).unwrap();
            range = [range[0].min(low), range[1].max(high)];
        }
    }
    if range[0] == usize::MAX {
        return [i32::MAX, i32::MIN];
    }
    let y = volume.minimum()[1];
    [
        y.wrapping_add(range[0] as i32),
        y.wrapping_add(range[1] as i32),
    ]
}

/// `DensityVolume::index_of_block` without its per-axis divisions for the
/// unit-step volumes the material stage samples (one or two lookups per block).
#[inline]
fn block_index(volume: DensityVolume, position: [i32; 3]) -> Option<usize> {
    if volume.steps() != [1; 3] {
        return volume.index_of_block(position);
    }
    let [mx, my, mz] = volume.minimum();
    let [sx, sy, sz] = volume.sizes();
    let x = position[0].wrapping_sub(mx) as u32 as usize;
    let y = position[1].wrapping_sub(my) as u32 as usize;
    let z = position[2].wrapping_sub(mz) as u32 as usize;
    (x < sx && y < sy && z < sz).then(|| y + (x + z * sx) * sy)
}

#[cfg(test)]
mod tests {
    use super::*;
    use horizon_core::density::SampleError;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Surface;
    impl MaterialSurfaceQueries for Surface {
        fn surface_depth(&mut self, _: i32, _: i32) -> i32 {
            0
        }
        fn surface_secondary(&mut self, _: i32, _: i32) -> f64 {
            0.0
        }
    }
    /// Position-dependent values; counts bulk samples.
    #[derive(Debug, Default)]
    struct Counted(AtomicUsize);
    impl DensitySampler for Counted {
        fn sample_value(&self, _: &mut SamplerContext, p: [i32; 3]) -> Result<f32, SampleError> {
            Ok((p[0] * 10_000 + p[1] * 100 + p[2]) as f32)
        }
        fn sample_volume(
            &self,
            context: &mut SamplerContext,
            buffer: &mut [f32],
            volume: DensityVolume,
        ) -> Result<(), SampleError> {
            self.0.fetch_add(1, Ordering::Relaxed);
            for z in 0..volume.sizes()[2] {
                for x in 0..volume.sizes()[0] {
                    for y in 0..volume.sizes()[1] {
                        let [mx, my, mz] = volume.minimum();
                        let p = [mx + x as i32, my + y as i32, mz + z as i32];
                        buffer[volume.index([x, y, z]).unwrap()] = self.sample_value(context, p)?;
                    }
                }
            }
            Ok(())
        }
    }
    fn sources() -> MaterialDensitySources<Surface> {
        let volume = DensityVolume::new([16, 24, 16], [32, -64, -16], [1; 3]).unwrap();
        MaterialDensitySources::new(
            Surface,
            volume,
            Arc::new(Counted::default()),
            SamplerContext::default(),
        )
        .unwrap()
    }

    #[test]
    fn eager_bindings_of_one_root_share_a_single_sample() {
        let mut sources = sources();
        let shared = Arc::new(Counted::default());
        let other = Arc::new(Counted::default());
        let a = sources.bind_density(shared.clone(), true).unwrap();
        let lazy = sources.bind_density(shared.clone(), false).unwrap();
        let b = sources.bind_density(shared.clone(), true).unwrap();
        let c = sources.bind_density(other.clone(), true).unwrap();
        assert_eq!(shared.0.load(Ordering::Relaxed), 1);
        assert_eq!(other.0.load(Ordering::Relaxed), 1);
        assert_eq!([a, lazy, b, c], [0, 1, 2, 3]);
        for p in [[32, -64, -16], [47, -41, -1], [40, -50, -9], [48, 0, 0]] {
            let position = BlockPos::new(p[0], p[1], p[2]);
            let expected = (p[0] * 10_000 + p[1] * 100 + p[2]) as f32;
            for index in [a, lazy, b, c] {
                assert_eq!(sources.density(index, position).unwrap(), expected);
            }
        }
        assert_eq!(shared.0.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn positive_heights_cover_every_sample_above_zero_or_nan() {
        let volume = DensityVolume::new([2, 4, 1], [0, -64, 0], [1; 3]).unwrap();
        // Y-innermost: column x=0 then x=1, from y -64 to -61.
        let heights = |samples: [f32; 8]| positive_heights(volume, &samples);
        assert_eq!(
            heights([0.0, -1.0, 0.0, -0.5, -1.0, 0.0, -0.0, -2.0]),
            [i32::MAX, i32::MIN]
        );
        assert_eq!(
            heights([0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0]),
            [-63, -62]
        );
        assert_eq!(
            heights([f32::NAN, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            [-64, -61]
        );
        let mut sources = sources();
        let density = sources
            .bind_density(Arc::new(Counted::default()), true)
            .unwrap();
        let lazy = sources
            .bind_density(Arc::new(Counted::default()), false)
            .unwrap();
        // Every sample of `Counted` over this volume is above zero.
        let known = sources.density_heights(density).unwrap();
        assert_eq!(
            (known.min, known.max, known.y),
            ([32, -64, -16], [47, -41, -1], [-64, -41])
        );
        assert_eq!(sources.density_heights(lazy), None);
    }

    /// The block's Y as its value.
    #[derive(Debug)]
    struct Height;
    impl DensitySampler for Height {
        fn sample_value(&self, _: &mut SamplerContext, p: [i32; 3]) -> Result<f32, SampleError> {
            Ok(p[1] as f32)
        }
    }

    #[test]
    fn gated_bindings_sample_below_the_gate_and_cover_the_heights_above() {
        let mut sources = sources();
        let gate = DensityGate {
            input: Arc::new(Height),
            min_inclusive: -60.0,
            max_exclusive: -50.0,
            lattices: [None; 2],
        };
        let shared = Arc::new(Counted::default());
        let density = sources
            .bind_gated_density(Arc::new(Counted::default()), &gate, true, None)
            .unwrap();
        let richness = sources
            .bind_gated_density(shared.clone(), &gate, false, None)
            .unwrap();
        // A full-volume binding of the same root needs its own sample.
        let full = sources.bind_density(shared.clone(), true).unwrap();
        // The richness binding samples on its first read.
        assert_eq!(shared.0.load(Ordering::Relaxed), 1);
        // Heights -64..=-51 were sampled; every `Counted` value there is > 0.
        let known = sources.density_heights(density).unwrap();
        assert_eq!(
            (known.min, known.max, known.y),
            ([32, -64, -16], [47, -41, -1], [-64, -51])
        );
        assert!(known.never_positive(BlockPos::new(40, -45, -9)));
        assert!(!known.never_positive(BlockPos::new(40, -51, -9)));
        for p in [[32, -64, -16], [47, -51, -1], [40, -45, -9]] {
            let position = BlockPos::new(p[0], p[1], p[2]);
            let expected = (p[0] * 10_000 + p[1] * 100 + p[2]) as f32;
            for index in [density, richness, full] {
                assert_eq!(sources.density(index, position).unwrap(), expected);
            }
        }
        assert_eq!(shared.0.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn gated_bindings_with_a_lattice_also_drop_whole_periods_below() {
        let mut sources = sources();
        let gate = DensityGate {
            input: Arc::new(Height),
            min_inclusive: -57.0,
            max_exclusive: -50.0,
            lattices: [Some(4); 2],
        };
        let density = sources
            .bind_gated_density(Arc::new(Counted::default()), &gate, true, Some(4))
            .unwrap();
        // In range from -57: the sample starts at -60 (-64 plus one period).
        assert_eq!(sources.buffers[0].volume.minimum(), [32, -60, -16]);
        assert_eq!(sources.buffers[0].volume.sizes(), [16, 10, 16]);
        let known = sources.density_heights(density).unwrap();
        assert_eq!(
            (known.min, known.max, known.y),
            ([32, -64, -16], [47, -41, -1], [-60, -51])
        );
        assert!(known.never_positive(BlockPos::new(40, -62, -9)));
        for p in [[32, -60, -16], [47, -51, -1], [40, -57, -9]] {
            let position = BlockPos::new(p[0], p[1], p[2]);
            let expected = (p[0] * 10_000 + p[1] * 100 + p[2]) as f32;
            assert_eq!(sources.density(density, position).unwrap(), expected);
        }
    }

    #[test]
    fn block_index_matches_index_of_block() {
        let unit = DensityVolume::new([16, 24, 16], [i32::MAX - 7, -64, -16], [1; 3]).unwrap();
        let strided = DensityVolume::new([5, 3, 5], [-8, -64, 0], [4, 8, 4]).unwrap();
        for volume in [unit, strided] {
            let [mx, my, mz] = volume.minimum();
            for dx in [-1, 0, 1, 4, 7, 8, 15, 16, 17, 20] {
                for dy in [-9, -1, 0, 8, 16, 23, 24] {
                    for dz in [-1, 0, 3, 15, 16, 20] {
                        let p = [
                            mx.wrapping_add(dx),
                            my.wrapping_add(dy),
                            mz.wrapping_add(dz),
                        ];
                        assert_eq!(block_index(volume, p), volume.index_of_block(p), "{p:?}");
                    }
                }
            }
            for p in [
                [i32::MIN, 0, 0],
                [i32::MAX, i32::MAX, i32::MAX],
                [0, i32::MIN, 0],
            ] {
                assert_eq!(block_index(volume, p), volume.index_of_block(p), "{p:?}");
            }
        }
    }
}
