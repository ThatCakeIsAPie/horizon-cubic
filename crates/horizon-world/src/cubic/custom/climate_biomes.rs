//! [`BiomeColumns`] over 26.3's multi-noise biome source: the overworld's
//! climate (`GeneratorData::overworld_climate`) sampled at sea level, depth 0,
//! at the quart `(qx / biomeScale, qz / biomeScale)`. The same lookup the GPU
//! biome pass makes (`gfx/biome_gen`), so near and far terrain agree.
//!
//! Answers are pure functions of the arguments. The sampler context is per
//! thread (and per adapter); `BiomeBlend` caches the answers per thread, so
//! the search is not repeated for a quart a thread has seen.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use horizon_core::density::SamplerContext;

use super::biomes::{BiomeColumns, biome_info};
use crate::worldgen::biome::BiomeId;
use crate::worldgen::climate::{ClimateIndex, ClimateSampler};
use crate::worldgen::generator::{GeneratorData, GeneratorError};

/// The quart row sea level is in (`63 >> 2`).
pub const SEA_LEVEL_QUART_Y: i32 = 63 >> 2;

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

thread_local! {
    static CONTEXTS: RefCell<HashMap<u64, SamplerContext>> = RefCell::new(HashMap::new());
}

/// The overworld's biomes for a seed, as [`BiomeColumns`].
pub struct ClimateBiomes {
    id: u64,
    data: Arc<GeneratorData>,
    climate: ClimateSampler,
    index: Arc<ClimateIndex<u32>>,
    scale: f64,
}

impl ClimateBiomes {
    /// `scale` is [`super::CustomGeneratorSettings::biome_scale`].
    pub fn new(data: &Arc<GeneratorData>, seed: i64, scale: f64) -> Result<Self, GeneratorError> {
        let climate = data.overworld_climate(seed)?;
        Ok(Self::from_parts(
            data,
            climate.climate,
            climate.index,
            scale,
        ))
    }

    pub fn from_parts(
        data: &Arc<GeneratorData>,
        climate: ClimateSampler,
        index: Arc<ClimateIndex<u32>>,
        scale: f64,
    ) -> Self {
        Self {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            data: data.clone(),
            climate,
            index,
            scale: if scale > 0.0 { scale } else { 1.0 },
        }
    }

    /// The quart the climate is read at for the generator's quart.
    fn scaled(&self, q: i32) -> i32 {
        if self.scale == 1.0 {
            q
        } else {
            (f64::from(q) / self.scale).floor() as i32
        }
    }

    /// The biome id at an (already scaled) quart.
    pub fn id_at_quart(&self, qx: i32, qz: i32) -> u32 {
        CONTEXTS.with(|c| {
            let mut all = c.borrow_mut();
            if all.len() > 8 && !all.contains_key(&self.id) {
                all.clear();
            }
            let ctx = all.entry(self.id).or_insert_with(SamplerContext::cached);
            let Ok(mut target) = self.climate.sample(ctx, [qx, SEA_LEVEL_QUART_Y, qz]) else {
                return 0;
            };
            target.depth = 0;
            self.index
                .searcher()
                .find(target)
                .ok()
                .copied()
                .unwrap_or(0)
        })
    }

    pub fn data(&self) -> &Arc<GeneratorData> {
        &self.data
    }
    pub fn scale(&self) -> f64 {
        self.scale
    }
    pub fn climate(&self) -> &ClimateSampler {
        &self.climate
    }
    pub fn index(&self) -> &Arc<ClimateIndex<u32>> {
        &self.index
    }

    /// Per biome id of the data: CWG's height, variation, top and filler
    /// block names (`biome_info`), for the far terrain's tables.
    pub fn biome_table(&self) -> Vec<super::biomes::BiomeInfo> {
        self.data
            .biome_names()
            .iter()
            .map(|name| biome_info(name))
            .collect()
    }
}

impl BiomeColumns for ClimateBiomes {
    fn biome(&self, quart_x: i32, quart_z: i32) -> BiomeId {
        BiomeId(self.id_at_quart(self.scaled(quart_x), self.scaled(quart_z)))
    }
    fn biome_name(&self, id: BiomeId) -> &str {
        self.data.biome_name(id)
    }
    fn possible_biomes(&self) -> Vec<BiomeId> {
        let ids: std::collections::BTreeSet<u32> = self.index.values().copied().collect();
        ids.into_iter().map(BiomeId).collect()
    }
}
