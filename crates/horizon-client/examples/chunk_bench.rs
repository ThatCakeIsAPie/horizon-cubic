//! Resident chunk storage benchmark: generates a block of vanilla terrain, then
//! reports the resident block/biome memory, the cost of lookups, meshing and
//! lighting on it.
//!
//! chunk_bench <client resource dir> [--seed S] [--size N] [--world DIR]
//!
//! `--world` generates with that world folder's enabled data packs, at its
//! dimension types' height.
#![forbid(unsafe_code)]
use horizon_client::{
    assets::Assets,
    biome::Biomes,
    blocks::BlockFacts,
    chunk::{Chunk, layout},
    light::{LightTables, apply_lighting, light_chunk},
    mesher::{MeshTables, Mesher},
    models,
    streaming::{ChunkGenerator, Proto},
    vanillagen::VanillaTerrain,
};
use horizon_core::biome::BiomeZoom;
use std::{path::PathBuf, sync::Arc, time::Instant};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let root = PathBuf::from(args.first().ok_or("usage: chunk_bench RESOURCES")?);
    let flag = |name: &str, default: i64| -> i64 {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    };
    let (seed, size) = (flag("--seed", 12345), flag("--size", 8) as i32);
    let assets = Assets::load(&root)?;
    let render = assets.reload(&root)?;
    let facts = BlockFacts::load(&root, assets.registry)?;
    let biomes = Biomes::load(&root)?;
    let tables = MeshTables::new(
        &assets,
        &render,
        &facts,
        models::load_visibility(&root, assets.registry)?,
    )?;
    let light = LightTables::load(&root, assets.registry, &facts)?;
    let world = args
        .iter()
        .position(|a| a == "--world")
        .and_then(|i| args.get(i + 1))
        .map(PathBuf::from);
    let overrides = match &world {
        Some(world) => {
            let packs = horizon_client::datapack::discover(world);
            let enabled = horizon_client::datapack::enabled_ids(world, &packs);
            let enabled: Vec<_> = enabled
                .iter()
                .filter_map(|id| packs.iter().find(|p| &p.id == id))
                .collect();
            let (o, warnings) =
                horizon_client::datapack::worldgen_overrides(&enabled, |b| biomes.contains(b));
            for w in warnings {
                eprintln!("data pack: {w}");
            }
            let heights = horizon_client::chunk::LEVELS.map(|l| {
                o.level_height(l)
                    .unwrap_or_else(|| horizon_client::chunk::LevelHeight::vanilla(l).unwrap())
            });
            horizon_client::chunk::set_level_heights(heights)?;
            o
        }
        None => Default::default(),
    };
    let column = layout();
    println!(
        "column: y {}..{} ({} sections)",
        column.min_y(),
        column.max_y(),
        column.sections()
    );
    let terrain = VanillaTerrain::for_world_with(
        &root.join("worldgen"),
        &overrides,
        assets.registry,
        &biomes,
        seed,
        "minecraft:overworld",
        true,
    )?;
    let mut worker = terrain.worker();

    // Protos for size+4 columns, decorated for the inner size+2, finished as size+2.
    let started = Instant::now();
    let outer = size + 4;
    let mut protos: Vec<Option<Proto>> = (0..outer * outer)
        .map(|i| Some(terrain.proto(&mut worker, i % outer, i / outer)))
        .collect();
    eprintln!("proto: {:.2}s", started.elapsed().as_secs_f64());
    let started = Instant::now();
    let inner = size + 2;
    for z in 1..=inner {
        for x in 1..=inner {
            let mut region: [Proto; 9] = std::array::from_fn(|k| {
                let (dx, dz) = (k as i32 % 3 - 1, k as i32 / 3 - 1);
                protos[((z + dz) * outer + x + dx) as usize]
                    .take()
                    .expect("proto")
            });
            terrain.decorate(&mut worker, &mut region);
            for (k, p) in region.into_iter().enumerate() {
                let (dx, dz) = (k as i32 % 3 - 1, k as i32 / 3 - 1);
                protos[((z + dz) * outer + x + dx) as usize] = Some(p);
            }
        }
    }
    eprintln!("decorate: {:.2}s", started.elapsed().as_secs_f64());
    let started = Instant::now();
    let mut chunks: Vec<Chunk> = Vec::new();
    for z in 1..=inner {
        for x in 1..=inner {
            let proto = protos[(z * outer + x) as usize].take().expect("proto");
            let mut chunk = terrain.finish(proto);
            chunk.x = x;
            chunk.z = z;
            chunks.push(chunk);
        }
    }
    let finish = started.elapsed().as_secs_f64();
    println!(
        "finish (proto -> resident chunk): {:.1} ms per chunk",
        finish * 1e3 / chunks.len() as f64
    );

    let chunks: Vec<Arc<Chunk>> = chunks.into_iter().map(Arc::new).collect();
    let at = |x: i32, z: i32| chunks[((z - 1) * inner + x - 1) as usize].clone();

    // Light all inner-inner columns.
    let started = Instant::now();
    let mut lit = Vec::new();
    for z in 2..inner {
        for x in 2..inner {
            let region: [[Option<Arc<Chunk>>; 3]; 3] = std::array::from_fn(|dz| {
                std::array::from_fn(|dx| Some(at(x + dx as i32 - 1, z + dz as i32 - 1)))
            });
            let borrowed = horizon_client::mesher::borrow(&region);
            lit.push(((x, z), light_chunk(&borrowed, &light, true)));
        }
    }
    println!(
        "light: {:.2} ms per column",
        started.elapsed().as_secs_f64() * 1e3 / lit.len() as f64
    );
    let mut chunks: Vec<Chunk> = chunks.iter().map(|c| (**c).clone()).collect();
    for ((x, z), sections) in lit {
        apply_lighting(&mut chunks[((z - 1) * inner + x - 1) as usize], sections);
    }
    let chunks: Vec<Arc<Chunk>> = chunks.into_iter().map(Arc::new).collect();
    let at = |x: i32, z: i32| chunks[((z - 1) * inner + x - 1) as usize].clone();

    // Memory.
    let (mut bytes, mut sections, mut uniform) = (0usize, 0usize, 0usize);
    let top = chunks
        .iter()
        .flat_map(|c| (0..256).filter_map(move |i| c.highest(0, |y| c.get(i & 15, y, i >> 4) != 0)))
        .max()
        .unwrap_or(column.min_y());
    println!("highest block: y {top}");
    // Each distinct section once (shared ones are one allocation), with its
    // Arc header and struct, and an 8-byte slot per section of every column.
    let mut seen = std::collections::HashSet::new();
    for chunk in &chunks {
        for s in &chunk.sections {
            sections += 1;
            uniform += usize::from(s.is_uniform(s.get(0)));
            bytes += std::mem::size_of::<usize>();
            if seen.insert(Arc::as_ptr(s)) {
                bytes += s.heap_bytes()
                    + std::mem::size_of::<horizon_client::chunk::Section>()
                    + 2 * std::mem::size_of::<usize>();
            }
        }
    }
    println!(
        "resident: {} chunks, {} sections ({} uniform, {} distinct), {:.1} KiB per chunk ({:.2} MiB total)",
        chunks.len(),
        sections,
        uniform,
        seen.len(),
        bytes as f64 / 1024.0 / chunks.len() as f64,
        bytes as f64 / 1048576.0
    );

    // Meshing.
    let mut mesher = Mesher {
        nether_shading: false,
        assets: &assets,
        render: &render,
        tables: &tables,
        biomes: &biomes,
        visibility: tables.visibility.clone(),
        zoom: BiomeZoom::new(7),
        blend_radius: 2,
        smooth_lighting: true,
        cutout_leaves: true,
        cull_leaves: false,
    };
    let mut best = f64::MAX;
    let mut quads = 0usize;
    for _ in 0..3 {
        let started = Instant::now();
        let mut jobs = 0;
        quads = 0;
        for z in 2..inner {
            for x in 2..inner {
                let region: [[Option<Arc<Chunk>>; 3]; 3] = std::array::from_fn(|dz| {
                    std::array::from_fn(|dx| Some(at(x + dx as i32 - 1, z + dz as i32 - 1)))
                });
                let borrowed = horizon_client::mesher::borrow(&region);
                for sy in 0..column.sections() as i32 {
                    let mesh = mesher.mesh(&borrowed, sy);
                    quads += (mesh.solid.len() + mesh.translucent.len()) / 4;
                    jobs += 1;
                }
            }
        }
        best = best.min(started.elapsed().as_secs_f64() / f64::from(jobs));
    }
    println!(
        "mesh: {:.1} us per section ({} quads total)",
        best * 1e6,
        quads
    );

    // Random lookups, as collision and entity queries do.
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let started = Instant::now();
    let mut sum = 0u64;
    let lookups = 20_000_000u64;
    for _ in 0..lookups {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let cx = 2 + (state % u64::from((inner - 2) as u32)) as i32;
        let cz = 2 + ((state >> 8) % u64::from((inner - 2) as u32)) as i32;
        let chunk = &chunks[((cz - 1) * inner + cx - 1) as usize];
        let (x, y, z) = (
            (state >> 16) as i32 & 15,
            column.min_y() + ((state >> 24) % column.height() as u64) as i32,
            (state >> 40) as i32 & 15,
        );
        sum += u64::from(chunk.get(x, y, z));
    }
    println!(
        "random get: {:.2} ns per lookup (checksum {sum})",
        started.elapsed().as_secs_f64() * 1e9 / lookups as f64
    );
    // Coherent scan, as a column or area query does.
    let started = Instant::now();
    let mut sum = 0u64;
    for _ in 0..4 {
        for chunk in &chunks {
            for y in column.min_y()..column.max_y() {
                for z in 0..16 {
                    for x in 0..16 {
                        sum += u64::from(chunk.get(x, y, z));
                    }
                }
            }
        }
    }
    println!(
        "scan get: {:.2} ns per lookup (checksum {sum})",
        started.elapsed().as_secs_f64() * 1e9
            / (4.0 * chunks.len() as f64 * 256.0 * f64::from(column.height()))
    );
    // The process's peak resident memory, where the system reports it.
    if let Ok(status) = std::fs::read_to_string("/proc/self/status")
        && let Some(peak) = status.lines().find(|l| l.starts_with("VmHWM:"))
    {
        println!("peak resident: {}", peak["VmHWM:".len()..].trim());
    }
    Ok(())
}
