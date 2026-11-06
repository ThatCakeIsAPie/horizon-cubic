//! The overworld's climate as a standalone sampler for cubic worlds
//! (`cubic::custom::climate_biomes`): the six climate roots of 26.3's
//! `minecraft:overworld` noise router, compiled for one seed, and the
//! multi-noise parameter list with biome ids as values. It is the same
//! climate the vanilla generator reads, so a Custom Cubic world lays its
//! biomes out as the seed's vanilla world does.
use super::*;
use horizon_core::density::graph::DensityCompiler;

/// What the cubic biome adapter and the GPU biome pass are built from.
pub struct OverworldClimate {
    pub climate: ClimateSampler,
    /// Parameter list; each entry's value is its biome's id in the data.
    pub index: Arc<super::super::climate::ClimateIndex<u32>>,
    /// The router and compiler, for `ClimateNoises::from_router`.
    pub router: NoiseRouter,
    pub compiler: DensityCompiler,
}

impl GeneratorData {
    /// The overworld's climate for `seed`, from the `minecraft:normal`
    /// preset's overworld stem.
    pub fn overworld_climate(&self, seed: i64) -> Result<OverworldClimate, GeneratorError> {
        let preset: Value = serde_json::from_str(
            self.world_presets
                .get("minecraft:normal")
                .ok_or_else(|| resource("missing world preset minecraft:normal"))?,
        )?;
        let generator = &preset["dimensions"]["minecraft:overworld"]["generator"];
        let settings_json: Value = match &generator["settings"] {
            Value::String(name) => serde_json::from_str(
                self.settings
                    .get(name)
                    .ok_or_else(|| resource(format!("missing noise settings {name}")))?,
            )?,
            inline => inline.clone(),
        };
        let legacy = settings_json["legacy_random_source"]
            .as_bool()
            .unwrap_or(false);
        let mut densities = DensityRegistry::from_json(&self.noises, &self.densities)?;
        let router =
            NoiseRouter::from_json(&settings_json["noise_router"].to_string(), &mut densities)?;
        let mut compiler = DensityCompiler::new_seeded(densities.into_graph(), seed, legacy);
        let climate = router
            .compile_climate(&mut compiler)
            .map_err(|e| resource(format!("climate: {e:?}")))?;
        let source = self
            .source_catalog
            .decode(&generator["biome_source"].to_string())?
            .resolve(&self.presets)
            .map_err(|e| resource(format!("biome source: {e}")))?;
        let ResolvedBiomeSource::MultiNoise(source) = source else {
            return Err(resource("the overworld's biome source is not multi-noise"));
        };
        let entries = source
            .parameters()
            .entries()
            .iter()
            .map(|(point, name)| (*point, self.biome_id(name).map_or(0, |id| id.0)))
            .collect();
        let index = super::super::climate::ClimateIndex::new(entries)
            .map_err(|e| resource(format!("climate index: {e:?}")))?;
        Ok(OverworldClimate {
            climate,
            index: Arc::new(index),
            router,
            compiler,
        })
    }
}
