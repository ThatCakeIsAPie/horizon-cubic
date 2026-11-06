//! Tests of `cubic_convert`: a small vanilla world is built on disk, converted
//! and opened as a cubic level. They need `HORIZON_TEST_RESOURCES`.
use super::*;
use crate::chunk::{Chunk, index};
use crate::cubic::{CubicLevel, MARKER_CUBE};
use crate::cubic_gen::CubicGeneration;
use crate::save::{ChunkExtras, encode_with};
use horizon_core::ticks::{SavedTick, TickPriority, TickTypeId};
use horizon_world::compression::Compression;
use horizon_world::nbt::{Limits, NamedTag};
use horizon_world::region::{ChunkRecord, RegionFile};
use std::collections::BTreeMap;

fn resources() -> Option<PathBuf> {
    std::env::var_os("HORIZON_TEST_RESOURCES").map(PathBuf::from)
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("horizon-convert-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Every file under `dir` by relative path, with its bytes.
fn snapshot(dir: &Path) -> BTreeMap<String, Vec<u8>> {
    fn walk(root: &Path, dir: &Path, out: &mut BTreeMap<String, Vec<u8>>) {
        for entry in std::fs::read_dir(dir).unwrap().flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk(root, &path, out);
            } else {
                let name = path
                    .strip_prefix(root)
                    .unwrap()
                    .to_string_lossy()
                    .replace('\\', "/");
                out.insert(name, std::fs::read(&path).unwrap());
            }
        }
    }
    let mut out = BTreeMap::new();
    walk(dir, dir, &mut out);
    out
}

fn put(path: &Path, bytes: &[u8]) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, bytes).unwrap();
}

fn write_records(folder: &Path, records: Vec<((i32, i32), NamedTag)>) {
    let mut files: BTreeMap<(i32, i32), RegionFile> = BTreeMap::new();
    std::fs::create_dir_all(folder).unwrap();
    for ((x, z), root) in records {
        let key = (x >> 5, z >> 5);
        let file = files.entry(key).or_insert_with(|| {
            RegionFile::open_writable(folder.join(format!("r.{}.{}.mca", key.0, key.1)), true)
                .unwrap()
        });
        let record = ChunkRecord::from_nbt(&root, Compression::Zlib, 1, Limits::saved()).unwrap();
        file.write(horizon_core::position::ChunkPos::new(x, z), &record)
            .unwrap();
    }
}

fn pig(id: i32, x: f64, y: f64, z: f64) -> Tag {
    Tag::compound([
        ("id", Tag::String("minecraft:pig".into())),
        ("Tag", Tag::Int(id)),
        (
            "Pos",
            Tag::List(vec![
                Tag::Double(x.to_bits()),
                Tag::Double(y.to_bits()),
                Tag::Double(z.to_bits()),
            ]),
        ),
    ])
}

/// A vanilla world of three columns (one at negative coordinates, in
/// another region), a column the source never finished and one that does not
/// read.
struct Source {
    world: PathBuf,
    chunks: Vec<Chunk>,
    entities: usize,
    bedrock: u64,
}

fn make_source(root: &Path, assets: &crate::assets::Assets, biomes: &Biomes) -> Source {
    let world = root.join("World");
    let dim = dimension_dir(&world, OVERWORLD);
    let tables = SaveTables::new(assets);
    let state = |name: &str| assets.registry.default_state(name).unwrap().0;
    let (bedrock, stone, grass, deepslate) = (
        state("minecraft:bedrock"),
        state("minecraft:stone"),
        state("minecraft:grass_block"),
        state("minecraft:deepslate"),
    );
    let mut chunks = Vec::new();
    let mut bedrock_blocks = 0;
    for (n, &(cx, cz)) in [(0, 0), (1, 0), (-3, -40)].iter().enumerate() {
        let mut chunk = Chunk::empty(cx, cz);
        for z in 0..16 {
            for x in 0..16 {
                // Vanilla's floor: bedrock at -64, a ragged layer to -60.
                chunk.set(x, -64, z, bedrock);
                bedrock_blocks += 1;
                if (x + z) % 2 == 0 {
                    chunk.set(x, -62, z, bedrock);
                    bedrock_blocks += 1;
                }
                if x == 3 && z == 3 {
                    chunk.set(x, -60, z, bedrock);
                    bedrock_blocks += 1;
                }
                for y in -59..60 {
                    chunk.set(x, y, z, if y < 0 { deepslate } else { stone });
                }
                chunk.set(x, 60 + n as i32, z, grass);
                // The top of the world, a cube of its own.
                chunk.set(x, 319, z, stone);
            }
        }
        chunks.push(chunk);
    }
    // A block entity and a scheduled tick in the first column, a block
    // entity in a cube of its own in the third.
    let chest = state("minecraft:chest");
    chunks[0].set(5, 70, 5, chest);
    chunks[0].block_entities.push((
        (5, 70, 5),
        crate::containers::BlockEntity::Chest(vec![None; 27]),
    ));
    chunks[2].set(6, 200, 6, chest);
    chunks[2].block_entities.push((
        (-3 * 16 + 6, 200, -40 * 16 + 6),
        crate::containers::BlockEntity::Chest(vec![None; 27]),
    ));
    chunks[0].saved = Some(Arc::new(ChunkExtras {
        block_ticks: vec![SavedTick {
            kind: TickTypeId(0),
            pos: horizon_core::position::BlockPos::new(3, 70, 3),
            delay: 7,
            priority: TickPriority::Normal,
        }],
        ..ChunkExtras::default()
    }));
    let encode = |chunk: &Chunk| {
        encode_with(
            chunk,
            assets.registry,
            biomes,
            Some(&tables),
            LevelShape::OVERWORLD,
        )
    };
    let mut records: Vec<((i32, i32), NamedTag)> =
        chunks.iter().map(|c| ((c.x, c.z), encode(c))).collect();
    // Never finished: Status other than full. Unreadable: no position.
    let mut unfinished = encode(&Chunk::empty(2, 0));
    if let Tag::Compound(map) = &mut unfinished.tag {
        map.insert("Status".into(), Tag::String("minecraft:noise".into()));
    }
    records.push(((2, 0), unfinished));
    let mut broken = encode(&Chunk::empty(4, 0));
    if let Tag::Compound(map) = &mut broken.tag {
        map.remove(&"xPos".into());
    }
    records.push(((4, 0), broken));
    write_records(&dim.join("region"), records);
    // Entities live in their own region files: two in the first column
    // (one high), one in the third.
    let entity_record = |x: i32, z: i32, list: Vec<Tag>| {
        (
            (x, z),
            NamedTag {
                name: String::new().into(),
                tag: Tag::compound([
                    ("DataVersion", Tag::Int(horizon_world::DATA_VERSION)),
                    ("Position", Tag::IntArray(vec![x, z])),
                    ("Entities", Tag::List(list)),
                ]),
            },
        )
    };
    write_records(
        &dim.join("entities"),
        vec![
            entity_record(0, 0, vec![pig(1, 4.5, 70.0, 4.5), pig(2, 6.5, 250.0, 6.5)]),
            entity_record(-3, -40, vec![pig(3, -42.5, -50.0, -634.5)]),
        ],
    );
    // What is copied as it is.
    put(&world.join("players/data/a.dat"), b"player");
    put(&world.join("advancements/a.json"), b"{}");
    put(&world.join("stats/a.json"), b"{}");
    put(&world.join("data/scoreboard.dat"), b"board");
    put(&world.join("icon.png"), b"icon");
    put(&world.join("session.lock"), b"lock");
    put(
        &dimension_dir(&world, "minecraft:the_nether").join("region/r.0.0.mca"),
        b"nether",
    );
    put(
        &dimension_dir(&world, "minecraft:the_end").join("DIM1.txt"),
        b"end",
    );
    put(&dim.join("data/map_1.dat"), b"map");
    // The poi folder is not carried.
    put(&dim.join("poi/r.0.0.mca"), b"poi");
    crate::save::write_dat(
        &world.join("level.dat"),
        Tag::compound([(
            "Data",
            Tag::compound([
                ("DataVersion", Tag::Int(horizon_world::DATA_VERSION)),
                ("LevelName", Tag::String("World".into())),
                ("SpawnX", Tag::Int(0)),
                ("SpawnY", Tag::Int(70)),
                ("SpawnZ", Tag::Int(0)),
            ]),
        )]),
    )
    .unwrap();
    Source {
        world,
        chunks,
        entities: 3,
        bedrock: bedrock_blocks,
    }
}

fn open(dst: &Path, assets: &crate::assets::Assets, biomes: &Arc<Biomes>) -> Arc<CubicLevel> {
    let level = CubicLevel::new(
        &dimension_dir(dst, OVERWORLD),
        1,
        assets.registry,
        biomes,
        LevelShape::cubic_window(LevelShape::OVERWORLD, -80, 304),
        &CubicGeneration::of_world(dst),
        None,
    )
    .expect("opens");
    level.set_tables(Arc::new(SaveTables::new(assets)));
    level
}

#[test]
fn a_vanilla_world_becomes_a_hybrid_cubic_world() {
    let Some(resources) = resources() else {
        return;
    };
    let root = scratch("hybrid");
    let assets = crate::assets::Assets::load(&resources).expect("assets");
    let biomes = Arc::new(Biomes::load(&resources).expect("biomes"));
    let source = make_source(&root, &assets, &biomes);
    let before = snapshot(&source.world);
    let dst = root.join("World (Cubic)");
    let mut options = ConvertOptions::new(&resources);
    options.threads = 2;
    let mut seen = Vec::new();
    let report = convert_with(&source.world, &dst, &options, |p| seen.push(p)).expect("converts");

    // The source did not change by a byte, and nothing is left behind.
    assert_eq!(snapshot(&source.world), before);
    assert!(!root.join("World (Cubic).converting").exists());
    assert!(seen.iter().any(|p| matches!(p, Progress::Copying { .. })));

    // The counts.
    assert_eq!(report.columns_converted, 3);
    assert_eq!(report.columns_unfinished, 1);
    assert_eq!(report.columns_unreadable, 1);
    assert_eq!(report.unreadable, [(4, 0)]);
    assert_eq!(report.regions, 2);
    assert_eq!(report.entities, source.entities as u64);
    assert_eq!(report.block_entities, 2);
    assert_eq!(report.bedrock_replaced, source.bedrock);
    assert!(report.cubes_uniform > 0 && report.cubes_uniform < report.cubes_written);
    assert!(report.columns_per_second > 0.0);

    // The copy is a Hybrid cubic world, named for it.
    assert!(crate::cubic::is_cubic_world(&dst));
    let (kind, settings) = crate::cubic_settings::load_world_generation(&dst).unwrap();
    assert_eq!(kind, crate::cubic_settings::CubicKind::Hybrid);
    assert!(dst.join(crate::cubic_settings::SETTINGS_FILE).is_file());
    assert!(!settings.trim().is_empty());
    let (window_min, window_height) = crate::cubic::saved_window(&dst).unwrap();
    assert_eq!(
        window_height,
        window_cubes(i32::from(DEFAULT_VERTICAL)) * 16
    );
    assert!(window_min < 70 && window_min + window_height > 70);
    let level = crate::save::Level::read_summary(&dst).unwrap();
    assert_eq!(level.name, "World (Cubic)");

    // Everything else is the source's, byte for byte; the chunk folders are not.
    for (name, bytes) in &before {
        if name == "level.dat" || name == "session.lock" {
            continue;
        }
        let copied = dst.join(name);
        if is_overworld_chunk_data(Path::new(name)) {
            assert!(!copied.exists(), "{name} was carried over");
        } else {
            assert_eq!(&std::fs::read(&copied).unwrap(), bytes, "{name}");
        }
    }
    assert!(!dst.join("session.lock").exists());

    // Open it as a cubic level and compare the blocks.
    let opened = open(&dst, &assets, &biomes);
    let (bedrock, deepslate) = (
        assets.registry.default_state("minecraft:bedrock").unwrap().0,
        assets
            .registry
            .default_state("minecraft:deepslate")
            .unwrap()
            .0,
    );
    let (mut block_entities, mut entities, mut ticks) = (0, 0, 0);
    for chunk in &source.chunks {
        assert!(opened.has_vanilla(chunk.x, chunk.z));
        for cy in -4..=19 {
            let pos = SectionPos::new(chunk.x, cy, chunk.z);
            let cube = opened.load_cube(pos);
            if let Some(cube) = &cube {
                block_entities += cube.block_entities.len();
                entities += cube.entities.len();
                ticks += cube.block_ticks.len();
            }
            for y in cy * 16..cy * 16 + 16 {
                for z in 0..16 {
                    for x in 0..16 {
                        let was = chunk.get(x, y, z);
                        let now = cube
                            .as_ref()
                            .map_or(0, |c| c.section.get(index(x, y & 15, z)));
                        let expected = if was == bedrock { deepslate } else { was };
                        assert_eq!(
                            now, expected,
                            "block {x} {y} {z} of column {} {}",
                            chunk.x, chunk.z
                        );
                    }
                }
            }
        }
        // The marker cube carries the surface heights.
        let marker = opened
            .load_cube(SectionPos::new(chunk.x, MARKER_CUBE, chunk.z))
            .unwrap();
        assert_eq!(marker.surface.as_ref().map(Vec::len), Some(256));
    }
    assert_eq!(block_entities, 2);
    assert_eq!(entities, source.entities);
    assert_eq!(ticks, 1);
    // The high pig is in its own cube, the chest at 200 in cube 12.
    let high = opened.load_cube(SectionPos::new(0, 15, 0)).unwrap();
    assert_eq!(high.entities.len(), 1);
    let chest = opened.load_cube(SectionPos::new(-3, 12, -40)).unwrap();
    assert_eq!(chest.block_entities.len(), 1);
    // The columns the source never finished stay to be generated.
    assert!(!opened.has_vanilla(2, 0));
    assert!(!opened.has_vanilla(4, 0));
    drop(opened);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn bedrock_can_be_kept_and_the_wrong_worlds_are_refused() {
    let Some(resources) = resources() else {
        return;
    };
    let root = scratch("options");
    let assets = crate::assets::Assets::load(&resources).expect("assets");
    let biomes = Arc::new(Biomes::load(&resources).expect("biomes"));
    let source = make_source(&root, &assets, &biomes);
    let mut options = ConvertOptions::new(&resources);
    options.keep_bedrock = true;
    options.threads = 1;

    // An existing destination is not touched.
    let dst = root.join("Kept");
    std::fs::create_dir_all(&dst).unwrap();
    let err = convert_with(&source.world, &dst, &options, |_| {}).unwrap_err();
    assert!(err.contains("already exists"), "{err}");
    std::fs::remove_dir_all(&dst).unwrap();

    let report = convert_with(&source.world, &dst, &options, |_| {}).expect("converts");
    assert_eq!(report.bedrock_replaced, 0);
    let opened = open(&dst, &assets, &biomes);
    let bedrock = assets.registry.default_state("minecraft:bedrock").unwrap().0;
    let floor = opened.load_cube(SectionPos::new(0, -4, 0)).unwrap();
    assert_eq!(floor.section.get(index(0, 0, 0)), bedrock);
    drop(opened);

    // A converted world is not converted again.
    let again = convert_with(&dst, &root.join("Twice"), &options, |_| {}).unwrap_err();
    assert!(again.contains("already"), "{again}");

    // A taller Overworld (the To The Max data pack) is refused, and nothing is left.
    let pack = source
        .world
        .join("datapacks/tall/data/minecraft/dimension_type");
    std::fs::create_dir_all(&pack).unwrap();
    std::fs::write(
        pack.join("overworld.json"),
        r#"{"min_y": -2032, "height": 4064}"#,
    )
    .unwrap();
    let tall = root.join("Tall");
    let err = convert_with(&source.world, &tall, &options, |_| {}).unwrap_err();
    assert!(err.contains("not vanilla's height"), "{err}");
    assert!(!tall.exists() && !root.join("Tall.converting").exists());
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn a_name_is_free_for_the_copy_and_chunk_data_is_told_apart() {
    let root = scratch("names");
    assert_eq!(destination_for(&root, "A"), root.join("A (Cubic)"));
    std::fs::create_dir_all(root.join("A (Cubic)")).unwrap();
    assert_eq!(destination_for(&root, "A"), root.join("A (Cubic 2)"));
    assert!(is_overworld_chunk_data(Path::new(
        "dimensions/minecraft/overworld/region"
    )));
    assert!(is_overworld_chunk_data(Path::new("region")));
    assert!(!is_overworld_chunk_data(Path::new(
        "dimensions/minecraft/the_nether/region"
    )));
    assert!(!is_overworld_chunk_data(Path::new(
        "dimensions/minecraft/overworld/data"
    )));
    let _ = std::fs::remove_dir_all(&root);
}
