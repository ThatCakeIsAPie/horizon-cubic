//! Climate coordinates derived from the six terrain-router climate fields.
use horizon_core::density::{DensitySampler, SampleError, SamplerContext};
use horizon_core::noise::{DensityError, DensityVolume};
use std::sync::Arc;
mod codec;
mod index;
pub use codec::{decode_parameter, decode_parameter_point};
pub use index::{
    ClimateIndex, ClimateSearch, IndexError, Nearest, Parameter, ParameterPoint, SearchHint,
};

/// Vanilla multiplies in float precision before the saturating Java long cast.
pub fn quantize(value: f32) -> i64 {
    (value * 10_000.0) as i64
}
pub fn unquantize(value: i64) -> f32 {
    value as f32 / 10_000.0
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TargetPoint {
    pub temperature: i64,
    pub humidity: i64,
    pub continentalness: i64,
    pub erosion: i64,
    pub depth: i64,
    pub weirdness: i64,
}
impl TargetPoint {
    pub fn from_values(values: [f32; 6]) -> Self {
        let [
            temperature,
            humidity,
            continentalness,
            erosion,
            depth,
            weirdness,
        ] = values.map(quantize);
        Self {
            temperature,
            humidity,
            continentalness,
            erosion,
            depth,
            weirdness,
        }
    }
    /// The seventh parameter is the zero offset used by biome nearest searches.
    pub const fn parameters(self) -> [i64; 7] {
        [
            self.temperature,
            self.humidity,
            self.continentalness,
            self.erosion,
            self.depth,
            self.weirdness,
            0,
        ]
    }
}
#[derive(Clone, Debug)]
pub struct ClimateSampler {
    samplers: [Arc<dyn DensitySampler>; 6],
    /// Per field, for [`Self::sample_lattice`]: exact in bulk, and its scalar
    /// value a function of position. Found once.
    lattice_kinds: Arc<std::sync::OnceLock<[(bool, bool); 6]>>,
}
impl ClimateSampler {
    /// Temperature, humidity, continentalness, erosion, depth, weirdness.
    pub fn new(samplers: [Arc<dyn DensitySampler>; 6]) -> Self {
        Self {
            samplers,
            lattice_kinds: Default::default(),
        }
    }
    /// Coordinates are biome quart positions, converted with wrapped << 2.
    /// Every child uses the same context and runs in field declaration order.
    pub fn sample(
        &self,
        context: &mut SamplerContext,
        quart: [i32; 3],
    ) -> Result<TargetPoint, SampleError> {
        let position = quart.map(|coordinate| coordinate.wrapping_shl(2));
        let mut values = [0.; 6];
        for (sampler, value) in self.samplers.iter().zip(&mut values) {
            *value = sampler.sample_value(context, position)?;
        }
        Ok(TargetPoint::from_values(values))
    }
    /// Prepare all six field buffers in declaration order on the quart lattice.
    /// Lookup quantizes the retained bulk values rather than resampling scalars.
    pub fn sample_volume(
        &self,
        context: &mut SamplerContext,
        quart_minimum: [i32; 3],
        extents: [i32; 3],
    ) -> Result<ClimateVolume, SampleError> {
        let volume = DensityVolume::new(extents, quart_minimum.map(|v| v.wrapping_shl(2)), [4; 3])?;
        let mut fields = std::array::from_fn(|_| vec![0.; volume.size()]);
        for (sampler, field) in self.samplers.iter().zip(&mut fields) {
            sampler.sample_volume(context, field, volume)?;
        }
        Ok(ClimateVolume {
            quart_minimum,
            volume,
            fields,
        })
    }
    /// The fields whose value depends on its position only, on a lattice of
    /// every `quart_steps` quarts: in bulk where that gives the same values
    /// ([`horizon_core::density::is_pointwise_scalar`]), else one by one
    /// ([`horizon_core::density::is_scalar_pure`]); the others are left to
    /// [`Self::sample_with_lattice`]. With all six and an `index`, also each
    /// sample's fresh answer there ([`ClimateIndex::nearest`]). `None` when
    /// no field qualifies: then a lattice saves nothing.
    pub fn sample_lattice<T>(
        &self,
        context: &mut SamplerContext,
        quart_minimum: [i32; 3],
        extents: [i32; 3],
        quart_steps: [i32; 3],
        index: Option<&ClimateIndex<T>>,
    ) -> Result<Option<ClimateLattice>, SampleError> {
        use horizon_core::density::{is_pointwise_scalar, is_scalar_pure};
        let kinds = *self.lattice_kinds.get_or_init(|| {
            self.samplers.each_ref().map(|s| {
                let bulk = is_pointwise_scalar(s);
                (bulk, bulk || is_scalar_pure(s))
            })
        });
        if !kinds.iter().any(|(_, pure)| *pure) {
            return Ok(None);
        }
        let volume = DensityVolume::new(
            extents,
            quart_minimum.map(|v| v.wrapping_shl(2)),
            quart_steps.map(|v| v.wrapping_shl(2)),
        )?;
        let mut fields: [Option<Vec<f32>>; 6] = Default::default();
        for ((sampler, field), (bulk, pure)) in self.samplers.iter().zip(&mut fields).zip(kinds) {
            let mut values = vec![0.; volume.size()];
            if bulk {
                sampler.sample_volume(context, &mut values, volume)?;
            } else if pure {
                let [sx, sy, sz] = volume.sizes();
                for (x, z) in (0..sx).flat_map(|x| (0..sz).map(move |z| (x, z))) {
                    for y in 0..sy {
                        let position = std::array::from_fn(|i| {
                            let at = [x, y, z][i] as i32;
                            volume.minimum()[i].wrapping_add(at.wrapping_mul(volume.steps()[i]))
                        });
                        let index = volume
                            .index([x, y, z])
                            .ok_or(DensityError::VolumeTooLarge)?;
                        values[index] = sampler.sample_value(context, position)?;
                    }
                }
            } else {
                continue;
            }
            *field = Some(values);
        }
        let nearest = match (index, fields.iter().all(Option::is_some)) {
            (Some(index), true) => Some(
                (0..volume.size())
                    .map(|i| {
                        let target = TargetPoint::from_values(std::array::from_fn(|f| {
                            fields[f].as_ref().unwrap()[i]
                        }));
                        index.nearest(target).ok()
                    })
                    .collect(),
            ),
            _ => None,
        };
        Ok(Some(ClimateLattice {
            quart_minimum,
            quart_steps,
            volume,
            fields,
            nearest,
        }))
    }
    /// [`Self::sample`], with the fields `lattice` holds read from it.
    pub fn sample_with_lattice(
        &self,
        context: &mut SamplerContext,
        lattice: &ClimateLattice,
        quart: [i32; 3],
    ) -> Result<Option<TargetPoint>, SampleError> {
        let Some(index) = lattice.index(quart) else {
            return Ok(None);
        };
        let position = quart.map(|coordinate| coordinate.wrapping_shl(2));
        let mut values = [0.; 6];
        for ((sampler, field), value) in self.samplers.iter().zip(&lattice.fields).zip(&mut values)
        {
            *value = match field {
                Some(field) => field[index],
                None => sampler.sample_value(context, position)?,
            };
        }
        Ok(Some(TargetPoint::from_values(values)))
    }
}

/// Climate fields sampled ahead of time on a coarse quart lattice
/// (`/locate biome`'s search grid), possibly with each sample's fresh answer.
#[derive(Debug)]
pub struct ClimateLattice {
    quart_minimum: [i32; 3],
    quart_steps: [i32; 3],
    volume: DensityVolume,
    fields: [Option<Vec<f32>>; 6],
    nearest: Option<Vec<Option<Nearest>>>,
}
impl ClimateLattice {
    fn index(&self, quart: [i32; 3]) -> Option<usize> {
        let mut relative = [0; 3];
        for i in 0..3 {
            let offset = quart[i].wrapping_sub(self.quart_minimum[i]);
            if offset < 0 || offset % self.quart_steps[i] != 0 {
                return None;
            }
            relative[i] = (offset / self.quart_steps[i]) as usize;
        }
        self.volume.index(relative)
    }
    /// The full target at `quart` and its fresh answer, when the lattice
    /// holds them (`Some(None)`: that search failed).
    pub fn nearest(&self, quart: [i32; 3]) -> Option<(TargetPoint, Option<Nearest>)> {
        let nearest = self.nearest.as_ref()?;
        let index = self.index(quart)?;
        let target = TargetPoint::from_values(std::array::from_fn(|f| {
            self.fields[f].as_ref().map_or(0., |field| field[index])
        }));
        Some((target, nearest[index]))
    }
}

#[derive(Debug)]
pub struct ClimateVolume {
    quart_minimum: [i32; 3],
    volume: DensityVolume,
    fields: [Vec<f32>; 6],
}
impl ClimateVolume {
    /// Coordinates must be inside the prepared quart volume. The bounded Rust
    /// API rejects invalid coordinates instead of Java's unchecked buffer alias.
    pub fn target(&self, quart: [i32; 3]) -> Option<TargetPoint> {
        let relative =
            std::array::from_fn(|i| quart[i].wrapping_sub(self.quart_minimum[i]) as usize);
        let index = self.volume.index(relative)?;
        Some(TargetPoint::from_values(std::array::from_fn(|i| {
            self.fields[i][index]
        })))
    }
}
