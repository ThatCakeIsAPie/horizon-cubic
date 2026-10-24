//! Chunk generation throughput for a world, with its data packs' world
//! generation (chunkgen_bench generates vanilla's): proto chunks per second
//! and their stages, then decoration of the inner 3x3 regions.
//!
//! world_chunkgen_bench <client resource dir> <world folder> [--size N] [--threads T] [--center CX CZ] [--no-ring-cache]
//!
//! The stronghold positions are kept in the world's overworld folder, as the
//! game does (`ring_cache`): the second run reads them. `--no-ring-cache`
//! searches for them without reading or writing that file.
use horizon_client::{datapack, ring_cache, save::Level};
use horizon_core::position::ChunkPos;
use horizon_world::worldgen::generator::{
    GeneratorData, GeneratorWorker, OverworldGenerator, ProtoChunk, StageTimings,
};
use std::{
    path::PathBuf,
    sync::{
        Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

type Error = Box<dyn std::error::Error + Send + Sync>;

fn main() -> Result<(), Error> {
    let usage = "usage: world_chunkgen_bench <resources> <world>";
    let mut args = std::env::args().skip(1);
    let resources = PathBuf::from(args.next().ok_or(usage)?);
    let world = PathBuf::from(args.next().ok_or(usage)?);
    let (mut size, mut threads, mut center) = (12i32, 6usize, (0, 0));
    let mut ring_cache = true;
    while let Some(arg) = args.next() {
        let mut value = || args.next().ok_or(format!("{arg} needs a value"));
        match arg.as_str() {
            "--no-ring-cache" => ring_cache = false,
            "--size" => size = value()?.parse()?,
            "--threads" => threads = value()?.parse()?,
            "--center" => center = (value()?.parse()?, value()?.parse()?),
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
    let generator =
        OverworldGenerator::for_world(data, seed, "minecraft:normal", "minecraft:overworld", true)?;
    if ring_cache {
        ring_cache::keep(
            &generator,
            &horizon_client::save::dimension_dir(&world, "minecraft:overworld"),
            &resources.join("worldgen"),
            seed,
            "minecraft:overworld",
        );
    }
    println!(
        "seed {seed}, {} enabled packs, generator ready in {:.1}s; {size}x{size} chunks around {center:?}, {threads} threads",
        enabled.len(),
        start.elapsed().as_secs_f64()
    );

    // A world's first structure lookup places the strongholds (concentric
    // rings with biome searches), once per session unless kept from the last
    // one: time it on its own.
    let kept = generator.rings().is_some();
    let start = Instant::now();
    generator
        .worker()
        .generate_proto(ChunkPos::new(center.0 + 1000, center.1 + 1000))?;
    println!(
        "first chunk, {} (once per world load): {:.1}s",
        if kept {
            "with the kept stronghold positions"
        } else {
            "with the stronghold ring search"
        },
        start.elapsed().as_secs_f64()
    );
    let rings = generator.rings().unwrap_or_default();
    let mut hash = std::hash::DefaultHasher::new();
    std::hash::Hash::hash(&format!("{rings:?}"), &mut hash);
    println!(
        "ring positions: {}, checksum {:016x}{}",
        rings.values().map(Vec::len).sum::<usize>(),
        std::hash::Hasher::finish(&hash),
        rings
            .values()
            .flatten()
            .next()
            .map(|c| format!(", first at chunk {} {}", c.x, c.z))
            .unwrap_or_default()
    );
    let (x0, z0) = (center.0 - size / 2, center.1 - size / 2);
    let count = (size * size) as usize;
    let next = AtomicUsize::new(0);
    let slots: Vec<Mutex<Option<ProtoChunk>>> = (0..count).map(|_| Mutex::new(None)).collect();
    let timings = Mutex::new(Vec::new());
    let start = Instant::now();
    std::thread::scope(|scope| -> Result<(), Error> {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                scope.spawn(|| -> Result<(), Error> {
                    let mut worker = generator.worker();
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= count {
                            timings.lock().unwrap().push(worker.stage_timings());
                            return Ok(());
                        }
                        let pos = ChunkPos::new(x0 + (i as i32 % size), z0 + (i as i32 / size));
                        *slots[i].lock().unwrap() = Some(worker.generate_proto(pos)?);
                    }
                })
            })
            .collect();
        handles.into_iter().try_for_each(|h| h.join().unwrap())
    })?;
    let proto_wall = start.elapsed();
    let mut hash = std::hash::DefaultHasher::new();
    for slot in &slots {
        chunk_hash(slot.lock().unwrap().as_ref().ok_or("chunk")?, &mut hash);
    }
    let terrain_hash = std::hash::Hasher::finish(&hash);
    let timings = timings.into_inner().unwrap();
    let per_chunk = |f: fn(&StageTimings) -> Duration| {
        timings.iter().map(f).sum::<Duration>().as_secs_f64() * 1000. / count as f64
    };
    println!(
        "terrain: {count} chunks in {:.1}s = {:.1} chunks/s, output {terrain_hash:016x}",
        proto_wall.as_secs_f64(),
        count as f64 / proto_wall.as_secs_f64()
    );
    println!(
        "  per chunk (one thread): biomes {:.1} ms, noise {:.1} ms, surface {:.1} ms, structures {:.1} ms, carvers {:.1} ms",
        per_chunk(|t| t.biomes),
        per_chunk(|t| t.noise),
        per_chunk(|t| t.surface),
        per_chunk(|t| t.structures),
        per_chunk(|t| t.carvers),
    );

    // The structure starts the references reached (cached by now): an exact
    // speedup keeps this checksum.
    let mut worker = generator.worker();
    let mut hash = std::hash::DefaultHasher::new();
    let mut starts = 0;
    for z in z0 - 8..z0 + size + 8 {
        for x in x0 - 8..x0 + size + 8 {
            let found = worker.structure_starts(ChunkPos::new(x, z));
            starts += found.len();
            std::hash::Hash::hash(&format!("{found:?}"), &mut hash);
        }
    }
    println!(
        "structure starts: {starts}, checksum {:016x}",
        std::hash::Hasher::finish(&hash)
    );
    // Decoration: disjoint 3x3 regions per phase run concurrently, as in the game.
    let mut chunks: Vec<Option<ProtoChunk>> =
        slots.into_iter().map(|s| s.into_inner().unwrap()).collect();
    let start = Instant::now();
    let mut decorated = 0;
    let busy = Mutex::new(Duration::ZERO);
    for phase in 0..9 {
        let mut regions = Vec::new();
        for z in 1..size - 1 {
            for x in 1..size - 1 {
                if (x % 3) * 3 + z % 3 != phase {
                    continue;
                }
                let mut region = Vec::with_capacity(9);
                for dz in -1..=1 {
                    for dx in -1..=1 {
                        region.push(chunks[((z + dz) * size + x + dx) as usize].take().unwrap());
                    }
                }
                regions.push(((x, z), region));
            }
        }
        decorated += regions.len();
        let queue = Mutex::new(regions);
        let done = Mutex::new(Vec::new());
        std::thread::scope(|scope| -> Result<(), Error> {
            let handles: Vec<_> = (0..threads)
                .map(|_| {
                    scope.spawn(|| -> Result<(), Error> {
                        let mut worker: GeneratorWorker = generator.worker();
                        loop {
                            let Some((at, region)) = queue.lock().unwrap().pop() else {
                                return Ok(());
                            };
                            let mut region: [ProtoChunk; 9] =
                                region.try_into().map_err(|_| "region size")?;
                            let t = Instant::now();
                            worker.decorate(&mut region)?;
                            *busy.lock().unwrap() += t.elapsed();
                            done.lock().unwrap().push((at, region));
                        }
                    })
                })
                .collect();
            handles.into_iter().try_for_each(|h| h.join().unwrap())
        })?;
        for ((x, z), region) in done.into_inner().unwrap() {
            for (i, chunk) in region.into_iter().enumerate() {
                let (dx, dz) = (i as i32 % 3 - 1, i as i32 / 3 - 1);
                chunks[((z + dz) * size + x + dx) as usize] = Some(chunk);
            }
        }
    }
    let decorate_wall = start.elapsed();
    let mut hash = std::hash::DefaultHasher::new();
    for chunk in &chunks {
        chunk_hash(chunk.as_ref().ok_or("chunk")?, &mut hash);
    }
    println!(
        "decoration: {decorated} chunks in {:.1}s = {:.1} chunks/s, {:.1} ms per chunk (one thread), output {:016x}",
        decorate_wall.as_secs_f64(),
        decorated as f64 / decorate_wall.as_secs_f64(),
        busy.into_inner().unwrap().as_secs_f64() * 1000. / decorated as f64,
        std::hash::Hasher::finish(&hash)
    );
    let total =
        proto_wall.as_secs_f64() * decorated as f64 / count as f64 + decorate_wall.as_secs_f64();
    println!(
        "overall: about {:.1} finished chunks/s on {threads} threads",
        decorated as f64 / total
    );
    Ok(())
}

/// Every block, quart biome and surface height of `chunk`, with its block
/// entities, entities and structure references: an exact speedup keeps it.
fn chunk_hash(chunk: &ProtoChunk, hash: &mut impl std::hash::Hasher) {
    let pos = chunk.position();
    hash.write_i32(pos.x);
    hash.write_i32(pos.z);
    let (min_y, height) = (chunk.terrain.min_y(), chunk.terrain.height());
    for z in 0..16 {
        for x in 0..16 {
            hash.write_i32(chunk.surface_height(x, z));
            for y in min_y..min_y + height {
                hash.write_u32(chunk.block(x, y, z).0);
            }
        }
    }
    for z in 0..4 {
        for x in 0..4 {
            for y in (min_y..min_y + height).step_by(4) {
                hash.write_u32(chunk.biome_id_at(x * 4, y, z * 4).0);
            }
        }
    }
    for kind in horizon_world::heightmap::HeightmapType::ALL {
        if let Some(map) = chunk.terrain.heightmap(kind) {
            for z in 0..16 {
                for x in 0..16 {
                    hash.write_i32(map.first_available(x, z).unwrap_or(i32::MIN));
                }
            }
        }
    }
    hash.write(format!("{:?}", chunk.terrain.postprocessing()).as_bytes());
    hash.write(format!("{:?}", chunk.terrain.block_ticks().pack(0)).as_bytes());
    hash.write(format!("{:?}", chunk.terrain.fluid_ticks().pack(0)).as_bytes());
    hash.write(format!("{:?}", chunk.block_entities).as_bytes());
    hash.write(format!("{:?}", chunk.entities).as_bytes());
    hash.write(format!("{:?}", chunk.structures()).as_bytes());
}
