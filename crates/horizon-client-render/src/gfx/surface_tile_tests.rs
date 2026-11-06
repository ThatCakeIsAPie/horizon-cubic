use super::*;
use crate::gfx::cube_gen::{BiomeBlocks, NoiseDesc, OctaveDesc};
use horizon_core::{noise::PerlinNoise, random::LegacyRandom};
use horizon_world::cubic::custom::testing::{bound_presets, default_registry};
use horizon_world::cubic::custom::{CustomCubicGenerator, FixedBiome};

const AIR: u32 = 0;
const STONE: u32 = 1;
const DEEPSLATE: u32 = 5;
const DIRT: u32 = 7;
const GRASS: u32 = 9;
const WATER: u32 = 26;

fn validate(source: &str) {
    let module = naga::front::wgsl::parse_str(source).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
}

#[test]
fn shaders_validate() {
    validate(GEN_SHADER);
    validate(CULL_SHADER);
    validate(&draw_shader_source());
}

#[test]
fn gpu_layouts_match_the_shaders() {
    // Four vec4, two per-level tables of 32 vec4, five tables of 64 vec4.
    assert_eq!(
        std::mem::size_of::<GpuParams>(),
        4 * 16 + 2 * 32 * 16 + 5 * 64 * 16
    );
    // Six planes, then five vec4.
    assert_eq!(std::mem::size_of::<FrameUniform>(), 96 + 5 * 16);
    assert_eq!(TILE_WORDS, 3344);
    assert_eq!(COLUMN_WORDS, 13);
}

// ------------------------------------------------------------- the format

fn span(top: i32, bottom: i32, block: u32) -> Span {
    Span {
        top,
        bottom,
        block,
        deep_bottom: false,
    }
}

#[test]
fn the_tile_format_round_trips() {
    let key = TileKey::new(14, -3, 2);
    let mut columns = vec![Column::default(); COLUMN_COUNT];
    // A 9,000-block peak, the cubic range's ends, an island over ground, a
    // solid-to-the-sky column, an empty one under water.
    columns[0].spans = vec![span(9_000, OPEN_BOTTOM, GRASS)];
    columns[1].spans = vec![span(MAX_Y, MIN_Y, STONE)];
    columns[2].spans = vec![
        Span {
            top: 5_100,
            bottom: 5_000,
            block: GRASS,
            deep_bottom: false,
        },
        Span {
            top: 70,
            bottom: OPEN_BOTTOM,
            block: GRASS,
            deep_bottom: false,
        },
    ];
    columns[3].spans = vec![span(OPEN_TOP, -40, STONE)];
    columns[3].spans[0].deep_bottom = true;
    columns[4].water = true;
    columns[5] = Column {
        spans: vec![
            span(300, 290, 1),
            span(200, 190, 2),
            span(100, 90, 3),
            span(10, OPEN_BOTTOM, 4),
        ],
        water: false,
        overflow: true,
        sample: 3,
        biome: 300,
    };
    columns[6].spans = vec![span(-8_388_608, OPEN_BOTTOM, 0x1ffff)];
    let tile = TileData {
        key,
        info: TileInfo {
            max_spans: 4,
            highest: MAX_Y,
            skirt: -1_234_567,
            overflow_columns: 1,
            water_columns: 1,
            spans: 11,
        },
        columns,
    };
    let words = tile.to_words();
    assert_eq!(words.len(), TILE_WORDS);
    // The header the cull pass reads.
    assert_eq!(words[0] as i32, key.origin()[0]);
    assert_eq!(words[1] as i32, key.origin()[1]);
    assert_eq!(words[2], 14);
    assert_eq!(words[3], 4);
    assert_eq!(words[5] as i32, -1_234_567);
    // Column 0: one span, then its top in blocks.
    let c0 = HEADER_WORDS;
    assert_eq!(words[c0] & 7, 1);
    assert_eq!(words[c0 + 1] as i32, 9_000);
    assert_eq!(words[c0 + 2] as i32, OPEN_BOTTOM);
    // Column 5's head word.
    let c5 = HEADER_WORDS + 5 * COLUMN_WORDS;
    assert_eq!(words[c5], 4 | 1 << 4 | 3 << 5 | 300 << 8);
    assert_eq!(TileData::from_words(&words), Some(tile));
    assert!(TileData::from_words(&words[..TILE_WORDS - 1]).is_none());
}

#[test]
fn tile_keys_match_cube_keys() {
    let cube = CubeKey::new(15, -2, 7, 3);
    let tile = TileKey::of(cube);
    assert_eq!(tile.origin(), [cube.origin()[0], cube.origin()[2]]);
    assert_eq!(tile.size(), cube.size());
    assert_eq!(tile.column_width() * COLUMNS as i32, tile.size());
    assert_eq!(
        TileKey::containing(15, cube.origin()[0] + 5, cube.origin()[2]),
        tile
    );
    for s in 0..SAMPLES {
        for (cx, cz) in [(0, 0), (15, 15), (3, 9)] {
            let [x, z] = tile.sample_point(cx, cz, s);
            let w = tile.column_width();
            let [ox, oz] = tile.origin();
            assert!((ox + cx as i32 * w..ox + (cx as i32 + 1) * w).contains(&x));
            assert!((oz + cz as i32 * w..oz + (cz as i32 + 1) * w).contains(&z));
        }
    }
}

// --------------------------------------------------------------- the march

fn range(lo: i32, hi: i32, step: i32, solid: &dyn Fn(i32) -> bool) -> Range {
    Range {
        lo,
        hi,
        step,
        above: solid(hi + 1),
        below: solid(lo - 1),
    }
}

#[test]
fn a_floating_island_over_ground_is_two_spans() {
    let solid = |y: i32| y < 70 || (5_000..=5_100).contains(&y);
    let r = range(0, 6_000, 64, &solid);
    assert_eq!(find_top(&r, solid), 5_100);
    let (spans, overflow) = march(&r, solid);
    assert_eq!(spans, vec![(5_100, 5_000), (69, OPEN_BOTTOM)]);
    assert!(!overflow);
}

#[test]
fn an_overhang_leaves_a_gap_under_it() {
    // An arch: rock at 80..=100 over the ground below 60.
    let solid = |y: i32| y < 60 || (80..=100).contains(&y);
    let r = range(0, 200, 8, &solid);
    let (spans, _) = march(&r, solid);
    assert_eq!(spans, vec![(100, 80), (59, OPEN_BOTTOM)]);
}

#[test]
fn too_many_spans_merge_into_the_last() {
    let bands = [500, 400, 300, 200, 100];
    let solid = |y: i32| y < 0 || bands.iter().any(|b| (*b..=b + 10).contains(&y));
    let r = range(-50, 600, 4, &solid);
    let (spans, overflow) = march(&r, solid);
    assert!(overflow);
    assert_eq!(
        spans,
        vec![(510, 500), (410, 400), (310, 300), (210, OPEN_BOTTOM)]
    );
}

#[test]
fn open_ends_and_empty_lines() {
    let sky = |_: i32| true;
    let r = range(0, 100, 16, &sky);
    assert_eq!(find_top(&r, sky), OPEN_TOP);
    assert_eq!(march(&r, sky).0, vec![(OPEN_TOP, OPEN_BOTTOM)]);
    let void = |_: i32| false;
    let r = range(0, 100, 16, &void);
    assert_eq!(find_top(&r, void), OPEN_BOTTOM);
    assert!(march(&r, void).0.is_empty());
    // A floating slab: bounded both ways.
    let slab = |y: i32| (40..=55).contains(&y);
    let r = range(0, 100, 4, &slab);
    assert_eq!(march(&r, slab).0, vec![(55, 40)]);
}

#[test]
fn tall_ranges_take_longer_steps() {
    assert_eq!(Range::step_for(1 << 14, 0, 1000), STEP_CAP);
    assert_eq!(Range::step_for(4, 0, 1000), 4);
    let step = Range::step_for(1 << 20, -8_000_000, 8_000_000);
    assert!(16_000_000 / step <= MAX_STEPS);
}

// ------------------------------------------------------- the CPU reference

fn noise(seed: i64, octaves: usize, factor: f64, offset: f64, f: [f64; 3]) -> NoiseDesc {
    let mut random = LegacyRandom::new(seed);
    NoiseDesc {
        factor,
        offset,
        frequency: f,
        octaves: (0..octaves)
            .map(|_| {
                let p = PerlinNoise::new(&mut random);
                OctaveDesc {
                    permutations: *p.table().permutations(),
                    offsets: *p.table().offsets(),
                }
            })
            .collect(),
    }
}

/// A density with no noise: `d * vol + height - y`, `d` the depth transform
/// of 0.
fn flat_params() -> CubeGenParams {
    CubeGenParams {
        noises: [
            noise(1, 2, 0.0, 0.0, [0.01, 0.01, 0.01]),
            noise(2, 2, 0.0, 0.0, [0.01, 0.01, 0.01]),
            noise(3, 2, 0.0, 0.0, [0.01, 0.01, 0.01]),
            noise(4, 2, 0.0, 0.0, [0.01, 0.0, 0.01]),
        ],
        height_factor: 1.0,
        height_offset: 0.0,
        height_variation_factor: 1.0,
        height_variation_offset: 0.0,
        below_average_factor: 1.0,
        sea_level: 63,
        air: AIR,
        stone: STONE,
        deepslate: DEEPSLATE,
        water: WATER,
        biomes: vec![BiomeBlocks {
            top: GRASS,
            filler: DIRT,
        }],
        filler_depth: 4.0,
        deepslate_from: 0.0,
        deepslate_to: 8.0,
    }
}

#[test]
fn exact_heights_have_no_fixed_range() {
    let p = flat_params();
    // d = depth_transform(0) = -2 / 5.6 * 0.2 * 17 / 64 = -0.01897...; with
    // vol 1 the last solid block is the one below height - 0.019.
    for (height, top) in [(9_000.5f32, 9_000), (5_000_000.5, 5_000_000), (-40.5, -41)] {
        for level in [4u8, 14, 22] {
            let key = TileKey::new(level, -1, 0);
            let tile = cpu_tile(&p, key, &|points| {
                points
                    .iter()
                    .map(|_| ColumnSample {
                        height,
                        volatility: 1.0,
                        biome: 0,
                    })
                    .collect()
            });
            for c in &tile.columns {
                assert_eq!(c.spans.len(), 1, "{height} at level {level}");
                assert_eq!(c.spans[0].top, top, "{height} at level {level}");
                assert_eq!(c.spans[0].bottom, OPEN_BOTTOM);
                assert_eq!(c.spans[0].block, GRASS);
                assert_eq!(c.water, top < 63);
            }
            assert_eq!(tile.info.highest, top.max(63));
            assert!(tile.info.skirt <= top - SKIRT_MIN);
            assert_eq!(TileData::from_words(&tile.to_words()), Some(tile));
        }
    }
}

/// The generator's parameters in the GPU generator's form.
fn params_of(g: &CustomCubicGenerator) -> CubeGenParams {
    let tables = g.noise_tables().expect("a terrain generator");
    let shape = g.density_shape().unwrap();
    let noise = |t: &horizon_world::cubic::custom::NoiseTable| NoiseDesc {
        factor: t.factor,
        offset: t.offset,
        frequency: t.frequency,
        octaves: t
            .octaves
            .iter()
            .map(|o| OctaveDesc {
                permutations: o.permutations,
                offsets: o.offsets,
            })
            .collect(),
    };
    CubeGenParams {
        noises: std::array::from_fn(|i| noise(&tables[i])),
        height_factor: shape[0],
        height_offset: shape[1],
        height_variation_factor: shape[2],
        height_variation_offset: shape[3],
        below_average_factor: shape[4],
        sea_level: g.sea_level(),
        air: AIR,
        stone: STONE,
        deepslate: DEEPSLATE,
        water: WATER,
        biomes: vec![BiomeBlocks {
            top: GRASS,
            filler: DIRT,
        }],
        filler_depth: 4.0,
        deepslate_from: 0.0,
        deepslate_to: 8.0,
    }
}

fn generator(preset: &str) -> Arc<CustomCubicGenerator> {
    let settings = bound_presets()
        .into_iter()
        .find(|(name, _)| *name == preset)
        .expect("a preset")
        .1;
    Arc::new(
        CustomCubicGenerator::new(
            11,
            &settings,
            default_registry(),
            Arc::new(FixedBiome::new("plains")),
        )
        .unwrap(),
    )
}

/// The generator's own blended column terrain at the points.
fn generator_columns(g: Arc<CustomCubicGenerator>) -> PointColumns {
    Arc::new(move |points: &[[i32; 2]]| {
        points
            .iter()
            .map(|&[x, z]| {
                let (h, v) = g.column_terrain(x, z).unwrap();
                ColumnSample {
                    height: h as f32,
                    volatility: v as f32,
                    biome: 0,
                }
            })
            .collect()
    })
}

#[test]
fn heights_match_the_generators_surface() {
    let g = generator("defaults");
    let p = params_of(&g);
    let columns = generator_columns(g.clone());
    let (mut close, mut near, mut total, mut worst) = (0, 0, 0, 0);
    for key in [
        TileKey::new(0, 0, 0),
        TileKey::new(0, -7, 12),
        TileKey::new(2, 3, -1),
        TileKey::new(2, 20_000, -5),
    ] {
        let tile = cpu_tile(&p, key, &*columns);
        for (c, column) in tile.columns.iter().enumerate() {
            let [x, z] = key.sample_point(c % COLUMNS, c / COLUMNS, usize::from(column.sample));
            let top = column.spans.first().expect("terrain everywhere").top;
            let Some(surface) = g.surface_y(x, z, top + 200) else {
                continue;
            };
            // The continuous density against the 4 x 8 x 4 lattice's.
            let diff = (top - surface).abs();
            worst = worst.max(diff);
            close += usize::from(diff <= 3);
            near += usize::from(diff <= 16);
            total += 1;
        }
    }
    assert!(total >= 900, "{total}");
    assert!(
        close * 100 >= total * 90,
        "{close}/{total} within 3, worst {worst}"
    );
    assert!(
        near * 100 >= total * 99,
        "{near}/{total} within 16, worst {worst}"
    );
}

#[test]
fn floating_islands_are_layers_not_pillars() {
    let g = generator("floating islands");
    let p = params_of(&g);
    let columns = generator_columns(g);
    let key = TileKey::new(4, 1, -2);
    let tile = cpu_tile(&p, key, &*columns);
    let density = CpuDensity::new(&p, key.level);
    let mut layered = 0;
    for (c, column) in tile.columns.iter().enumerate() {
        let point = key.sample_point(c % COLUMNS, c / COLUMNS, usize::from(column.sample));
        let sample = columns(&[point])[0];
        let probe = density.probe(point, sample);
        // Top-down, apart, each bounded by a change of sign.
        for w in column.spans.windows(2) {
            assert!(w[0].bottom > w[1].top + 1, "{c}: {:?}", column.spans);
            // A gap: air between them, no pillar.
            assert!(!probe.solid(w[0].bottom - 1), "{c}: {:?}", column.spans);
        }
        for s in &column.spans {
            if s.top != OPEN_TOP {
                assert!(probe.solid(s.top) && !probe.solid(s.top + 1), "{c}: {s:?}");
            }
            if s.bottom != OPEN_BOTTOM && !column.overflow {
                assert!(
                    probe.solid(s.bottom) && !probe.solid(s.bottom - 1),
                    "{c}: {s:?}"
                );
            }
        }
        layered += usize::from(column.spans.len() >= 2);
    }
    assert!(layered > 0, "no column with an island over another span");
    assert_eq!(
        tile.info.max_spans as usize,
        tile.columns.iter().map(|c| c.spans.len()).max().unwrap()
    );
    assert_eq!(
        tile.info.overflow_columns as usize,
        tile.columns.iter().filter(|c| c.overflow).count()
    );
}

// ------------------------------------------------------------------- GPU

fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = crate::gfx::test_gpu::instance();
    let adapter = pollster::block_on(instance.request_adapter(&Default::default())).ok()?;
    pollster::block_on(adapter.request_device(&Default::default())).ok()
}

#[test]
fn gpu_matches_the_cpu_reference() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter: skipped");
        return;
    };
    for preset in ["defaults", "floating islands", "amplified"] {
        let g = generator(preset);
        let p = Arc::new(params_of(&g));
        let columns = generator_columns(g);
        let Some(mut tiles) = SurfaceTiles::new(
            &device,
            &queue,
            p.clone(),
            ColumnSource::Cpu(columns.clone()),
            64,
        ) else {
            eprintln!("no compute shaders: skipped");
            return;
        };
        let keys = [
            TileKey::new(4, 0, 0),
            TileKey::new(4, -3, 5),
            TileKey::new(8, 70, -2),
            TileKey::new(14, 1, 0),
            TileKey::new(18, -1, 0),
            // x = 5,000,000.
            TileKey::new(6, 4_882, 3),
        ];
        tiles.request(&keys);
        tiles.update();
        tiles.finish();
        let (mut same_count, mut close, mut spans, mut columns_seen) = (0, 0, 0, 0);
        for key in keys {
            assert!(tiles.ready(key));
            let gpu = tiles.read_tile(key).expect("a tile");
            let cpu = cpu_tile(&p, key, &*columns);
            assert_eq!(gpu.key, key);
            assert_eq!(tiles.info(key), Some(gpu.info), "{preset} {key:?}");
            for (g, c) in gpu.columns.iter().zip(&cpu.columns) {
                columns_seen += 1;
                if g.spans.len() == c.spans.len() && g.sample == c.sample {
                    same_count += 1;
                    for (a, b) in g.spans.iter().zip(&c.spans) {
                        spans += 1;
                        close += usize::from(
                            (i64::from(a.top) - i64::from(b.top)).abs() <= 1
                                && (i64::from(a.bottom) - i64::from(b.bottom)).abs() <= 1,
                        );
                        assert_eq!(a.block, b.block);
                    }
                }
                assert_eq!(g.biome, c.biome);
            }
        }
        eprintln!(
            "{preset}: {same_count}/{columns_seen} columns alike, {close}/{spans} spans within a block"
        );
        assert!(
            same_count * 100 >= columns_seen * 95,
            "{preset}: {same_count}/{columns_seen}"
        );
        assert!(close * 100 >= spans * 98, "{preset}: {close}/{spans}");
    }
}

#[test]
fn freed_slots_are_reused_and_counted() {
    let Some((device, queue)) = device() else {
        return;
    };
    let p = Arc::new(flat_params());
    let columns: PointColumns = Arc::new(|points: &[[i32; 2]]| {
        points
            .iter()
            .map(|_| ColumnSample {
                height: 100.5,
                volatility: 1.0,
                biome: 0,
            })
            .collect()
    });
    let Some(mut tiles) = SurfaceTiles::new(&device, &queue, p, ColumnSource::Cpu(columns), 4)
    else {
        return;
    };
    let keys: Vec<TileKey> = (0..6).map(|i| TileKey::new(14, i, 0)).collect();
    tiles.request(&keys);
    tiles.update();
    tiles.finish();
    // Four slots: two wait.
    assert_eq!(tiles.stats().resident, 4);
    assert_eq!(tiles.stats().pending, 2);
    assert_eq!(tiles.stats().bytes, 4 * TILE_BYTES);
    tiles.free(keys[0]);
    tiles.free(keys[1]);
    tiles.update();
    tiles.finish();
    assert!(keys[4..].iter().all(|&k| tiles.ready(k)));
    let tile = tiles.read_tile(keys[5]).unwrap();
    assert!(
        tile.columns
            .iter()
            .all(|c| c.spans == vec![span(100, OPEN_BOTTOM, GRASS)])
    );
    assert_eq!(tiles.complex(keys[5]), Some(0));
}

/// 256 tiles of the defaults at level 16 (run with `--ignored --nocapture`).
#[test]
#[ignore = "timing"]
fn a_batch_of_256_tiles() {
    let Some((device, queue)) = device() else {
        return;
    };
    let g = generator("defaults");
    let p = Arc::new(params_of(&g));
    let flat: PointColumns = Arc::new(|points: &[[i32; 2]]| {
        points
            .iter()
            .map(|&[x, z]| ColumnSample {
                height: 0.3 * ((x as f32 * 1e-4).sin() + (z as f32 * 1.3e-4).cos()),
                volatility: 0.8,
                biome: 0,
            })
            .collect()
    });
    let Some(mut tiles) = SurfaceTiles::new(&device, &queue, p, ColumnSource::Cpu(flat), 512)
    else {
        return;
    };
    let keys: Vec<TileKey> = (0..256)
        .map(|i| TileKey::new(16, i % 16 - 8, i / 16 - 8))
        .collect();
    tiles.request(&keys);
    tiles.update();
    tiles.finish();
    eprintln!("256 level-16 tiles: {:.2} ms", tiles.stats().last_batch_ms);
}
