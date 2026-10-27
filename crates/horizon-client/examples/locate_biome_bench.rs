//! `/locate biome` cost (`GeneratorWorker::find_closest_biome`) for a world,
//! with its data packs' world generation: seconds per search from one origin,
//! as `LocateCommand.locateBiome` runs it.
//!
//! locate_biome_bench <client resource dir> <world folder> [--min-y -64] [--max-y 2031]
//!     [--origin x,y,z]... biome...
use horizon_client::{datapack, save::Level};
use horizon_world::worldgen::generator::{BiomeSearch, GeneratorData, OverworldGenerator};
use std::{path::PathBuf, time::Instant};

type Error = Box<dyn std::error::Error + Send + Sync>;

fn main() -> Result<(), Error> {
    let mut args = std::env::args().skip(1);
    let usage = "usage: locate_biome_bench <resources> <world> biome...";
    let resources = PathBuf::from(args.next().ok_or(usage)?);
    let world = PathBuf::from(args.next().ok_or(usage)?);
    let (mut min_y, mut max_y) = (-64, 2031);
    let mut biomes = Vec::new();
    let mut origins = Vec::new();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--min-y" => min_y = args.next().ok_or("--min-y N")?.parse()?,
            "--max-y" => max_y = args.next().ok_or("--max-y N")?.parse()?,
            "--origin" => {
                let v: Vec<i32> = args
                    .next()
                    .ok_or("--origin x,y,z")?
                    .split(',')
                    .map(str::parse)
                    .collect::<Result<_, _>>()?;
                origins.push(<[i32; 3]>::try_from(v).map_err(|_| "--origin x,y,z")?);
            }
            _ => biomes.push(arg),
        }
    }
    let seed = Level::read(&world).ok_or("no level.dat")?.seed;
    let packs = datapack::discover(&world);
    let enabled = datapack::enabled_ids(&world, &packs);
    let enabled: Vec<_> = enabled
        .iter()
        .filter_map(|id| packs.iter().find(|p| &p.id == id))
        .collect();
    let (overrides, _) = datapack::worldgen_overrides(&enabled, |_| true);
    let registry = Box::leak(Box::new(horizon_world::level::data::load_registry(
        &resources,
    )?));
    let data = GeneratorData::load_with(&resources.join("worldgen"), registry, &overrides)?;
    let generator = OverworldGenerator::for_world(
        data,
        seed,
        "minecraft:normal",
        "minecraft:overworld",
        false,
    )?;
    let mut worker = generator.worker();
    println!(
        "seed {seed}, {} enabled packs, y {min_y}..{max_y}",
        enabled.len()
    );
    if origins.is_empty() {
        origins.push([0, 64, 0]);
    }
    for (origin, biome) in origins
        .iter()
        .flat_map(|o| biomes.iter().map(move |b| (*o, b)))
    {
        let allowed = |name: &str| name == biome.as_str();
        let search = BiomeSearch {
            allowed: &allowed,
            origin,
            radius: 6400,
            horizontal: 32,
            vertical: 64,
            min_y,
            max_y,
        };
        let start = Instant::now();
        let found = worker.find_closest_biome(&search);
        println!(
            "{origin:?} {biome}: {:?} in {:.2}s",
            found.map(|(_, p)| p),
            start.elapsed().as_secs_f64()
        );
    }
    Ok(())
}
