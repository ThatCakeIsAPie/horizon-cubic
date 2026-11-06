use super::testing::*;
use super::*;

fn generator(seed: i64, biome: &str) -> (CustomCubicGenerator, &'static BlockRegistry) {
    let reg = default_registry();
    let g = CustomCubicGenerator::new(
        seed,
        &CustomGeneratorSettings::defaults(),
        reg,
        Arc::new(FixedBiome::new(biome)),
    )
    .unwrap();
    (g, reg)
}

fn cube(g: &CustomCubicGenerator, x: i32, y: i32, z: i32) -> Box<[u32; 4096]> {
    let mut out = Box::new([0_u32; 4096]);
    g.generate(SectionPos::new(x, y, z), &mut out);
    out
}

fn id(reg: &BlockRegistry, n: &str) -> u32 {
    reg.default_state(&format!("minecraft:{n}")).unwrap().0
}

#[test]
fn a_cube_is_a_pure_function_of_seed_and_position() {
    let (a, _) = generator(7, "plains");
    let (b, _) = generator(7, "plains");
    let (c, _) = generator(8, "plains");
    assert_eq!(cube(&a, 3, 4, -2)[..], cube(&b, 3, 4, -2)[..]);
    assert_ne!(cube(&a, 3, 3, -2)[..], cube(&c, 3, 3, -2)[..]);
}

#[test]
fn deep_cubes_are_stone_or_deepslate_and_high_cubes_are_air() {
    let (g, reg) = generator(1, "plains");
    let deep = cube(&g, 0, -40, 0);
    let (stone, deepslate, air) = (id(reg, "stone"), id(reg, "deepslate"), id(reg, "air"));
    let solid = deep
        .iter()
        .filter(|&&b| b == stone || b == deepslate)
        .count();
    assert!(solid > 3800, "{solid}");
    assert!(deep.iter().filter(|&&b| b == deepslate).count() > 3000);
    let sky = cube(&g, 0, 60, 0);
    assert!(sky.iter().all(|&b| b == air));
    assert_eq!(
        g.generate(SectionPos::new(0, 60, 0), &mut [0; 4096]),
        Some(air)
    );
}

#[test]
fn the_surface_has_stone_below_and_water_or_air_above() {
    let (g, reg) = generator(11, "plains");
    let (air, water) = (id(reg, "air"), id(reg, "water"));
    let y = g.surface_y(10, 10, 300).expect("a surface");
    assert!((20..160).contains(&y), "{y}");
    let below = g.block_at(10, y - 6, 10, 1);
    assert!(
        [id(reg, "stone"), id(reg, "deepslate"), id(reg, "dirt")].contains(&below),
        "{below}"
    );
    let above = g.block_at(10, y + 20, 10, 1);
    assert!(above == air || above == water);
    // The generated cube agrees with the point query away from caves.
    let c = cube(&g, 0, 4, 0);
    let mut same = 0;
    for (i, &b) in c.iter().enumerate() {
        let (x, z, ly) = ((i & 15) as i32, ((i >> 4) & 15) as i32, (i >> 8) as i32);
        if g.block_at(x, 64 + ly, z, 1) == b {
            same += 1;
        }
    }
    assert!(same > 3900, "{same}");
}

#[test]
fn neighbouring_cubes_agree_across_a_shared_face() {
    let (g, _) = generator(5, "plains");
    let a = cube(&g, 0, 3, 0);
    let b = cube(&g, 1, 3, 0);
    let mut same = 0;
    for y in 0..16 {
        for z in 0..16 {
            if (a[y << 8 | z << 4 | 15] == 0) == (b[y << 8 | z << 4] == 0) {
                same += 1;
            }
        }
    }
    assert!(same >= 230, "{same}");
}

#[test]
fn population_places_a_plausible_number_of_ores() {
    let (g, reg) = generator(3, "plains");
    let stone = id(reg, "stone");
    let mut world = MapWorld::new(stone);
    for dy in 0..2 {
        for dz in 0..2 {
            for dx in 0..2 {
                world.insert([dx, -4 + dy, dz], [stone; 4096]);
            }
        }
    }
    g.populate(SectionPos::new(0, -4, 0), &mut world);
    let ores: Vec<u32> = default_block_names()
        .iter()
        .filter(|n| n.ends_with("_ore"))
        .map(|n| reg.default_state(n).unwrap().0)
        .collect();
    let total: usize = world
        .cubes
        .values()
        .map(|c| c.iter().filter(|b| ores.contains(b)).count())
        .sum();
    assert!((1..2500).contains(&total), "{total}");
}

#[test]
fn json_settings_build_a_generator_and_unknown_blocks_are_named() {
    let reg = default_registry();
    let mut s = CustomGeneratorSettings::defaults();
    s.standard_ores[0].blockstate = BlockStateDesc::new("mod:mystery_ore");
    let e = CustomGeneratorSettings::from_json(&s.to_json_compact()).unwrap();
    let err = CustomCubicGenerator::new(1, &e, reg, Arc::new(FixedBiome::new("plains")))
        .err()
        .unwrap();
    assert!(err.contains("mod:mystery_ore"), "{err}");
}

#[test]
fn flat_cubic_fills_layers() {
    let reg = default_registry();
    let g = CustomCubicGenerator::flat(&FlatGeneratorSettings::default(), reg).unwrap();
    assert_eq!(g.block_at(0, 0, 0, 1), id(reg, "grass_block"));
    assert_eq!(g.block_at(0, -1, 0, 1), id(reg, "dirt"));
    assert_eq!(g.block_at(0, 5, 0, 1), id(reg, "air"));
    assert_eq!(g.surface_y(0, 0, 100), Some(0));
}

#[test]
fn cube_areas_delegate_by_box() {
    let reg = default_registry();
    let inner = CustomGeneratorSettings {
        replacers: vec![ReplacerConfig::DensityRange(
            settings::DensityRangeReplacerConfig {
                block_in_range: Some(BlockStateDesc::new("sand")),
                ..Default::default()
            },
        )],
        ..CustomGeneratorSettings::default()
    };
    let mut s = CustomGeneratorSettings::defaults();
    s.cube_areas.push(settings::CubeArea::from((
        IntAabb {
            min_x: 0,
            min_y: 0,
            min_z: 0,
            max_x: 0,
            max_y: 0,
            max_z: 0,
        },
        inner,
    )));
    let g = CustomCubicGenerator::new(1, &s, reg, Arc::new(FixedBiome::new("plains"))).unwrap();
    let sand = id(reg, "sand");
    assert!(cube(&g, 0, 0, 0).iter().all(|&b| b == sand));
    assert!(cube(&g, 1, 0, 0).iter().any(|&b| b != sand));
}

#[test]
fn noise_tables_follow_the_settings() {
    let (g, _) = generator(1, "plains");
    let t = g.noise_tables().unwrap();
    assert_eq!(t[0].octaves.len(), 8);
    assert_eq!(t[0].factor, 12.75);
    assert_eq!(t[3].frequency[1], 0.0);
}
