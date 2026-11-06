//! Checks the density bounds against the lattice generator and times the
//! early-out: `cargo run --profile fast -p horizon-world --example bounds_probe`.

use horizon_core::position::SectionPos;
use horizon_world::cubic::custom::testing::{bound_presets, default_registry};
use horizon_world::cubic::custom::{CustomCubicGenerator, FixedBiome};
use std::sync::Arc;
use std::time::Instant;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }
    fn range(&mut self, lo: i32, hi: i32) -> i32 {
        lo + (self.next() % (hi - lo + 1) as u64) as i32
    }
}

fn main() {
    let reg = default_registry();
    let mut rng = Rng(12345);
    // Nodes: a proven node is one block wherever `block_at` samples it.
    for (name, settings) in bound_presets() {
        let g = CustomCubicGenerator::new(2, &settings, reg, Arc::new(FixedBiome::new("plains")))
            .unwrap();
        let (mut proven, mut wrong, mut total) = (0, 0, 0);
        for _ in 0..600 {
            let level = rng.range(0, 3) as u32;
            let (x, z) = (rng.range(-8, 8), rng.range(-8, 8));
            let y = rng.range(-6, 24) >> level;
            let wl = f64::from(1 << rng.range(0, 3));
            total += 1;
            let Some(b) = g.classify_node(level, x, y, z, wl).block else {
                continue;
            };
            proven += 1;
            let side = 16 << level;
            for _ in 0..200 {
                let p = [
                    x * side + rng.range(0, side - 1),
                    y * side + rng.range(0, side - 1),
                    z * side + rng.range(0, side - 1),
                ];
                if g.block_at(p[0], p[1], p[2], wl as i32) != b {
                    wrong += 1;
                    println!("NODE MISMATCH {name} {level} {p:?}");
                    break;
                }
            }
        }
        println!("{name:17} nodes {total} proven {proven} wrong {wrong}");
    }
    for (name, settings) in bound_presets() {
        for biome in ["plains", "windswept_hills", "deep_ocean"] {
            let g = CustomCubicGenerator::new(7, &settings, reg, Arc::new(FixedBiome::new(biome)))
                .unwrap();
            let (mut boxes, mut uniform, mut bad) = (0, 0, 0);
            let mut out = [0_u32; 4096];
            for _ in 0..400 {
                let size = [1, 4, 16, 16, 32, 64][rng.range(0, 5) as usize];
                let min = [
                    rng.range(-300, 300),
                    rng.range(-300, 400),
                    rng.range(-300, 300),
                ];
                let max = min.map(|v| v + size - 1);
                boxes += 1;
                let Some(b) = g.uniform_block(min, max) else {
                    continue;
                };
                uniform += 1;
                'cubes: for cz in min[2] >> 4..=max[2] >> 4 {
                    for cy in min[1] >> 4..=max[1] >> 4 {
                        for cx in min[0] >> 4..=max[0] >> 4 {
                            g.base_cube(SectionPos::new(cx, cy, cz), &mut out);
                            for (i, &v) in out.iter().enumerate() {
                                let p = [
                                    cx * 16 + (i & 15) as i32,
                                    cy * 16 + (i >> 8) as i32,
                                    cz * 16 + ((i >> 4) & 15) as i32,
                                ];
                                if (0..3).all(|a| p[a] >= min[a] && p[a] <= max[a]) && v != b {
                                    bad += 1;
                                    println!("MISMATCH {name} {biome} {min:?}..{max:?}");
                                    break 'cubes;
                                }
                            }
                        }
                    }
                }
            }
            println!("{name:17} {biome:16} boxes {boxes} uniform {uniform} wrong {bad}");

            // Fractions of cubes the bounds prove uniform (before carvers),
            // and warm timings of the early-out against the lattice.
            let bands = [
                ("sky (y >= 320)", 20..40),
                ("near surface", -4..20),
                ("deep (y < -64)", -60..-4),
            ];
            for (label, ys) in bands {
                let mut cubes = Vec::new();
                for cz in 0..6 {
                    for cx in 0..6 {
                        for cy in ys.clone().step_by(2) {
                            cubes.push(SectionPos::new(200 + cx, cy, cz));
                        }
                    }
                }
                let proven = cubes
                    .iter()
                    .filter(|c| {
                        let min = [c.x * 16, c.y * 16, c.z * 16];
                        g.uniform_block(min, min.map(|v| v + 15)).is_some()
                    })
                    .count();
                for c in &cubes {
                    g.generate(*c, &mut out);
                }
                let t = Instant::now();
                for c in &cubes {
                    g.generate_lattice(*c, &mut out);
                }
                let t_old = t.elapsed().as_secs_f64();
                let t = Instant::now();
                for c in &cubes {
                    g.generate(*c, &mut out);
                }
                let t_new = t.elapsed().as_secs_f64();
                let n = cubes.len();
                println!(
                    "    {label:16} proven uniform {proven}/{n}; per cube: lattice {:.0} us, now {:.0} us",
                    t_old / n as f64 * 1e6,
                    t_new / n as f64 * 1e6
                );
            }
        }
    }
}
