//! Far terrain sampling cost (`GeneratorWorker::lod_grid`) for a world,
//! with its data packs' world generation: milliseconds per far tile at each
//! tile level, as `lod::Lod` requests them (16 x 16 columns).
//!
//! lod_grid_bench <client resource dir> <world folder> [--tiles N] [--levels 0,2,4,6]
use horizon_client::{datapack, save::Level};
use horizon_world::worldgen::generator::{GeneratorData, OverworldGenerator};
use std::{path::PathBuf, time::Instant};

type Error = Box<dyn std::error::Error + Send + Sync>;

fn main() -> Result<(), Error> {
    let mut args = std::env::args().skip(1);
    let resources = PathBuf::from(
        args.next()
            .ok_or("usage: lod_grid_bench <resources> <world>")?,
    );
    let world = PathBuf::from(
        args.next()
            .ok_or("usage: lod_grid_bench <resources> <world>")?,
    );
    let mut tiles = 6;
    let mut levels = vec![0u8, 2, 4, 6];
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--tiles" => tiles = args.next().ok_or("--tiles N")?.parse()?,
            "--levels" => {
                levels = args
                    .next()
                    .ok_or("--levels 0,2,4")?
                    .split(',')
                    .map(str::parse)
                    .collect::<Result<_, _>>()?
            }
            _ => return Err(format!("unknown argument {arg}").into()),
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
    let start = Instant::now();
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
        "seed {seed}, {} enabled packs, generator ready in {:.1}s",
        enabled.len(),
        start.elapsed().as_secs_f64()
    );
    // Warm the worker's caches once.
    worker.lod_grid(0, 0, 16, 16)?;
    for level in levels {
        let size = 16i32 << level;
        let width = size / 16;
        let start = Instant::now();
        let mut spans = 0;
        // Compares outputs across builds: an exact speedup keeps it.
        let mut hash = std::hash::DefaultHasher::new();
        for i in 0..tiles {
            // Spread tiles around the origin, as a view would request them.
            let (tx, tz) = (i % 3 - 1, i / 3 - 1);
            let grid = worker.lod_grid(tx * size + width / 2, tz * size + width / 2, width, 16)?;
            spans += grid.iter().map(|(_, s)| s.len()).sum::<usize>();
            std::hash::Hash::hash(&format!("{grid:?}"), &mut hash);
        }
        let ms = start.elapsed().as_secs_f64() * 1000. / tiles as f64;
        println!(
            "level {level} ({size} blocks, stride {width}): {ms:.1} ms/tile, {:.1} spans/column, output {:016x}",
            spans as f64 / (tiles * 256) as f64,
            std::hash::Hasher::finish(&hash)
        );
    }
    Ok(())
}
