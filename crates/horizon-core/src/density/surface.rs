use super::*;
use std::{collections::HashMap, sync::Mutex};

/// The unsliced search primitive. Compiled terrain graphs wrap this in a Y=0
/// slice through `compile_surface_search`, so their upper bound is horizontal.
#[derive(Clone, Debug)]
pub struct FindTopSurfaceSampler {
    density: Arc<dyn DensitySampler>,
    upper_bound: Arc<dyn DensitySampler>,
    lower_bound: i32,
    cell_height: i32,
    /// Searches already made, shared by every context and thread, when each
    /// density value depends on its position alone ([`is_pointwise_scalar`]):
    /// a search is then a function of its column and upper bound. Neighbouring
    /// chunks' aquifers, surfaces and structures search the same columns.
    searches: Option<Arc<SearchMemo>>,
}
#[derive(Debug, Default)]
struct SearchMemo {
    shards: [Mutex<HashMap<(i32, i32, u32), i32>>; 16],
}
impl SearchMemo {
    const SHARD_LIMIT: usize = 1 << 16;
    fn shard(&self, key: (i32, i32, u32)) -> &Mutex<HashMap<(i32, i32, u32), i32>> {
        let mix =
            (key.0 as u32).wrapping_mul(0x9E37_79B9) ^ (key.1 as u32).wrapping_mul(0x85EB_CA6B);
        &self.shards[(mix >> 28) as usize]
    }
}
impl FindTopSurfaceSampler {
    pub fn new(
        density: Arc<dyn DensitySampler>,
        upper_bound: Arc<dyn DensitySampler>,
        lower_bound: i32,
        cell_height: i32,
    ) -> Result<Self, SampleError> {
        if cell_height <= 0 {
            return Err(DensityError::NonpositiveSteps.into());
        }
        Ok(Self {
            searches: is_pointwise_scalar(&density).then(Default::default),
            density,
            upper_bound,
            lower_bound,
            cell_height,
        })
    }
    fn find_surface_from(
        &self,
        context: &mut SamplerContext,
        x: i32,
        z: i32,
        upper: f32,
    ) -> Result<i32, SampleError> {
        let Some(searches) = &self.searches else {
            return self.search(context, x, z, upper);
        };
        let key = (x, z, upper.to_bits());
        let shard = searches.shard(key);
        if let Some(&surface) = shard.lock().unwrap().get(&key) {
            return Ok(surface);
        }
        let surface = self.search(context, x, z, upper)?;
        let mut shard = shard.lock().unwrap();
        if shard.len() >= SearchMemo::SHARD_LIMIT {
            shard.clear();
        }
        shard.insert(key, surface);
        Ok(surface)
    }
    fn search(
        &self,
        context: &mut SamplerContext,
        x: i32,
        z: i32,
        upper: f32,
    ) -> Result<i32, SampleError> {
        let quotient = upper / self.cell_height as f32;
        // Java 26.3 Mth.floor(float) widens to double, calls Math.floor,
        // then uses a saturating double-to-int conversion (NaN becomes zero).
        let floor = (quotient as f64).floor() as i32;
        let mut y = floor.wrapping_mul(self.cell_height);
        if y <= self.lower_bound {
            return Ok(self.lower_bound);
        }
        while y >= self.lower_bound {
            if self.density.sample_value(context, [x, y, z])? > 0.0 {
                return Ok(y);
            }
            // Malformed manually constructed bounds can otherwise wrap and
            // loop forever; serialized vanilla terrain bounds do not reach this.
            y = y
                .checked_sub(self.cell_height)
                .ok_or(SampleError::SurfaceSearchOverflow)?;
        }
        Ok(self.lower_bound)
    }
}
impl DensitySampler for FindTopSurfaceSampler {
    fn sample_value(
        &self,
        context: &mut SamplerContext,
        [x, y, z]: [i32; 3],
    ) -> Result<f32, SampleError> {
        let upper = self.upper_bound.sample_value(context, [x, y, z])?;
        Ok(self.find_surface_from(context, x, z, upper)? as f32)
    }
    fn sample_volume(
        &self,
        context: &mut SamplerContext,
        output: &mut [f32],
        volume: DensityVolume,
    ) -> Result<(), SampleError> {
        check_buffer(output, volume)?;
        if volume.sizes()[1] != 1 {
            return Err(SampleError::SurfaceRequiresSingleLayer);
        }
        self.upper_bound.sample_volume(context, output, volume)?;
        visit(volume, |index, [x, _, z]| {
            output[index] = self.find_surface_from(context, x, z, output[index])? as f32;
            Ok(())
        })
    }
}

pub fn compile_surface_search(
    density: Arc<dyn DensitySampler>,
    upper_bound: Arc<dyn DensitySampler>,
    lower_bound: i32,
    cell_height: i32,
) -> Result<SliceSampler, SampleError> {
    Ok(SliceSampler {
        input: Arc::new(FindTopSurfaceSampler::new(
            density,
            upper_bound,
            lower_bound,
            cell_height,
        )?),
        axes: SliceAxes::Y(0),
    })
}
