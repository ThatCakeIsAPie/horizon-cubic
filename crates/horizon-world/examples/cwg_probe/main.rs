//! A quick look at the custom cubic generator: a cross-section and timings.
//! `cargo run --release -p horizon-world --example cwg_probe`

use horizon_core::position::SectionPos;
use horizon_world::cubic::custom::testing::default_registry;
use horizon_world::cubic::custom::{CustomCubicGenerator, CustomGeneratorSettings, FixedBiome};
use std::sync::Arc;
use std::time::Instant;

fn main() {
    let reg = default_registry();
    let name = |n: &str| reg.default_state(&format!("minecraft:{n}")).unwrap().0;
    let g = CustomCubicGenerator::new(
        42,
        &CustomGeneratorSettings::defaults(),
        reg,
        Arc::new(FixedBiome::new("plains")),
    )
    .unwrap();
    let glyphs = [
        (name("air"), ' '),
        (name("stone"), '#'),
        (name("deepslate"), 'D'),
        (name("water"), '~'),
        (name("grass_block"), 'g'),
        (name("dirt"), 'd'),
        (name("sand"), 's'),
        (name("gravel"), 'v'),
    ];
    // Cross-section along x at z = 0, cubes -4..10 in y.
    let mut rows: Vec<String> = Vec::new();
    for cy in (-4..10).rev() {
        let mut lines = vec![String::new(); 16];
        for cx in 0..8 {
            let mut out = [0_u32; 4096];
            g.generate(SectionPos::new(cx, cy, 0), &mut out);
            for ly in 0..16 {
                for lx in 0..16 {
                    let b = out[ly << 8 | lx];
                    let c = glyphs.iter().find(|g| g.0 == b).map_or('?', |g| g.1);
                    lines[15 - ly].push(c);
                }
            }
        }
        rows.extend(lines);
    }
    for r in rows.iter().step_by(2) {
        println!("{r}");
    }
    println!(
        "surface at x=0..64: {:?}",
        (0..64)
            .step_by(16)
            .map(|x| g.surface_y(x, 0, 400))
            .collect::<Vec<_>>()
    );

    let t = Instant::now();
    let mut n = 0;
    for cz in 0..8 {
        for cx in 0..8 {
            for cy in -6..8 {
                let mut out = [0_u32; 4096];
                g.generate(SectionPos::new(100 + cx, cy, cz), &mut out);
                n += 1;
            }
        }
    }
    let e = t.elapsed();
    println!("{n} cubes in {e:?}: {:?} per cube", e / n);
    let t = Instant::now();
    let mut n = 0;
    for cx in 0..16 {
        for cy in -6..8 {
            let mut out = [0_u32; 4096];
            g.generate(SectionPos::new(100 + cx, cy, 20), &mut out);
            n += 1;
        }
    }
    let e = t.elapsed();
    println!("warm-ish row of {n} cubes: {:?} per cube", e / n);
}
