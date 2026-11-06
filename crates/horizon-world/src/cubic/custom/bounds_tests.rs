//! Tests of the density bounds against the lattice generator.

use super::bounds::{GRADIENT_BOUND, VALUE_BOUND};
use super::testing::{bound_presets, default_registry};
use super::*;
use horizon_core::noise::PerlinNoise;
use horizon_core::random::XoroshiroRandom;

struct Lcg(u64);

impl Lcg {
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
    fn unit(&mut self) -> f64 {
        (self.next() % 1_000_000) as f64 / 1_000_000.0
    }
}

fn generators(seed: i64, biome: &str) -> Vec<(&'static str, CustomCubicGenerator)> {
    let reg = default_registry();
    bound_presets()
        .into_iter()
        .map(|(name, s)| {
            let g =
                CustomCubicGenerator::new(seed, &s, reg, Arc::new(FixedBiome::new(biome))).unwrap();
            (name, g)
        })
        .collect()
}

/// A box the bounds call uniform is that one block in the lattice
/// generator's cubes: three presets, three biomes, many boxes.
#[test]
fn a_uniform_box_is_uniform_in_the_lattice_generator() {
    let mut rng = Lcg(99);
    let mut checked = 0;
    for biome in ["plains", "windswept_hills", "deep_ocean"] {
        for (name, g) in generators(7, biome) {
            let mut out = Box::new([0_u32; 4096]);
            for _ in 0..120 {
                let size = [1, 4, 16, 16, 32, 64][rng.range(0, 5) as usize];
                let min = [
                    rng.range(-300, 300),
                    rng.range(-300, 400),
                    rng.range(-300, 300),
                ];
                let max = min.map(|v| v + size - 1);
                let Some(b) = g.uniform_block(min, max) else {
                    continue;
                };
                checked += 1;
                for cz in min[2] >> 4..=max[2] >> 4 {
                    for cy in min[1] >> 4..=max[1] >> 4 {
                        for cx in min[0] >> 4..=max[0] >> 4 {
                            g.base_cube(SectionPos::new(cx, cy, cz), &mut out);
                            for (i, &v) in out.iter().enumerate() {
                                let p = [
                                    cx * 16 + (i & 15) as i32,
                                    cy * 16 + (i >> 8) as i32,
                                    cz * 16 + ((i >> 4) & 15) as i32,
                                ];
                                if (0..3).all(|a| p[a] >= min[a] && p[a] <= max[a]) {
                                    assert_eq!(v, b, "{name} {biome} {min:?}..{max:?} at {p:?}");
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(checked > 100, "{checked}");
}

/// `generate` with the early-out is the lattice generation, caves included.
#[test]
fn the_early_out_changes_no_block() {
    for (name, g) in generators(3, "plains") {
        let (mut a, mut b) = (Box::new([0_u32; 4096]), Box::new([0_u32; 4096]));
        for cy in (-24..40).step_by(3) {
            for (cx, cz) in [(0, 0), (5, -3), (-9, 12)] {
                let c = SectionPos::new(cx, cy, cz);
                let r = g.generate(c, &mut a);
                g.generate_lattice(c, &mut b);
                assert_eq!(a[..], b[..], "{name} {c:?}");
                assert_eq!(r.is_some(), a.iter().all(|&v| v == a[0]));
            }
        }
    }
}

/// `block_at` with the bounds is the lattice's block.
#[test]
fn point_queries_agree_with_the_lattice() {
    let (name, g) = generators(5, "plains").remove(0);
    let mut out = Box::new([0_u32; 4096]);
    let mut rng = Lcg(5);
    for _ in 0..40 {
        let c = SectionPos::new(rng.range(-6, 6), rng.range(-6, 12), rng.range(-6, 6));
        g.base_cube(c, &mut out);
        for _ in 0..40 {
            let (lx, ly, lz) = (rng.range(0, 15), rng.range(0, 15), rng.range(0, 15));
            let want = out[(ly << 8 | lz << 4 | lx) as usize];
            let got = g.block_at(c.x * 16 + lx, c.y * 16 + ly, c.z * 16 + lz, 1);
            assert_eq!(got, want, "{name} {c:?} {lx} {ly} {lz}");
        }
    }
}

/// The defaults prove nearly every sky and deep cube uniform, and not the
/// band around the surface.
#[test]
fn defaults_prove_sky_and_deep_cubes_uniform() {
    let (_, g) = generators(1, "plains").remove(0);
    let frac = |ys: std::ops::Range<i32>| {
        let (mut n, mut u) = (0, 0);
        for cz in 0..6 {
            for cx in 0..6 {
                for cy in ys.clone() {
                    let min = [(30 + cx) * 16, cy * 16, cz * 16];
                    n += 1;
                    u += i32::from(g.uniform_block(min, min.map(|v| v + 15)).is_some());
                }
            }
        }
        f64::from(u) / f64::from(n)
    };
    let (sky, deep) = (frac(16..40), frac(-60..-12));
    assert!(sky > 0.97, "sky {sky}");
    assert!(deep > 0.9, "deep {deep}");
    assert!(frac(-2..10) < 0.9, "the surface band is not uniform");
}

/// The octave value and gradient bounds hold on a fine scan.
#[test]
fn the_octave_bounds_hold() {
    let mut worst_grad: f64 = 0.0;
    let mut worst_val: f64 = 0.0;
    for seed in 0..4 {
        let p = PerlinNoise::new(&mut XoroshiroRandom::new(seed));
        let mut rng = Lcg(seed as u64 + 1);
        for _ in 0..4000 {
            let (x, y, z) = (
                rng.unit() * 40.0 - 20.0,
                rng.unit() * 40.0 - 20.0,
                rng.unit() * 40.0 - 20.0,
            );
            let v = f64::from(p.get3(x, y, z));
            worst_val = worst_val.max(v.abs());
            let e = 1e-2;
            for (dx, dy, dz) in [(e, 0.0, 0.0), (0.0, e, 0.0), (0.0, 0.0, e)] {
                let w = f64::from(p.get3(x + dx, y + dy, z + dz));
                worst_grad = worst_grad.max((w - v).abs() / e);
            }
        }
    }
    assert!(worst_val <= VALUE_BOUND, "{worst_val}");
    assert!(worst_grad <= GRADIENT_BOUND, "{worst_grad}");
}

/// `VALUE_BOUND` and `GRADIENT_BOUND` are suprema of the best case over every
/// gradient assignment, certified on a grid plus the Lipschitz allowance of
/// the module docs of `bounds.rs`.
#[test]
fn the_octave_constants_are_certified() {
    fn fade(t: f64) -> f64 {
        t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
    }
    fn dfade(t: f64) -> f64 {
        30.0 * t * t * (1.0 - t) * (1.0 - t)
    }
    // The sum of the two largest |d_i|: the best of the +-d_a +- d_b.
    fn best(d: [f64; 3]) -> f64 {
        let a = d.map(f64::abs);
        a[0] + a[1] + a[2] - a[0].min(a[1]).min(a[2])
    }
    let n = 256;
    let h = 1.0 / f64::from(n);
    let (mut vmax, mut gmax): (f64, f64) = (0.0, 0.0);
    // The half cube: both functions are symmetric under u_a -> 1 - u_a.
    for iz in 0..=n / 2 {
        for iy in 0..=n / 2 {
            for ix in 0..=n / 2 {
                let u = [ix, iy, iz].map(|i| f64::from(i) * h);
                let f = u.map(fade);
                let df = u.map(dfade);
                let mut value = 0.0;
                let mut grad = [0.0; 3];
                for c in 0..8_usize {
                    let corner: [bool; 3] = std::array::from_fn(|a| (c >> a) & 1 == 1);
                    let om = |a: usize| if corner[a] { f[a] } else { 1.0 - f[a] };
                    let dom = |a: usize| if corner[a] { df[a] } else { -df[a] };
                    let w = om(0) * om(1) * om(2);
                    let d: [f64; 3] =
                        std::array::from_fn(|a| u[a] - if corner[a] { 1.0 } else { 0.0 });
                    value += w * best(d);
                    for (axis, g) in grad.iter_mut().enumerate() {
                        let mut dw = dom(axis);
                        for other in (0..3).filter(|&o| o != axis) {
                            dw *= om(other);
                        }
                        let mut v = d.map(|x| dw * x);
                        v[axis] += w;
                        *g += best(v);
                    }
                }
                vmax = vmax.max(value);
                gmax = grad.iter().fold(gmax, |m, &g| m.max(g));
            }
        }
    }
    assert!(vmax + 12.75 * h <= VALUE_BOUND, "{vmax}");
    assert!(gmax + 54.0 * h <= GRADIENT_BOUND, "{gmax}");
    // And the grid is not far from the constants: they are not slack by
    // accident (the supremum of V is 1.036, of F_x 3.75).
    assert!(vmax > 1.03 && gmax > 3.7, "{vmax} {gmax}");
}

/// Every biome of the table is inside the constants the large-box bound
/// uses.
#[test]
fn the_biome_table_is_inside_the_global_range() {
    use super::biomes::biome_info;
    for n in [
        "ocean",
        "deep_ocean",
        "mushroom_fields",
        "plains",
        "ice_spikes",
        "desert",
        "swamp",
        "forest",
        "flower_forest",
        "old_growth_birch_forest",
        "taiga",
        "savanna",
        "savanna_plateau",
        "windswept_savanna",
        "windswept_hills",
        "meadow",
        "snowy_slopes",
        "jagged_peaks",
        "badlands",
        "wooded_badlands",
        "river",
        "beach",
        "stony_shore",
        "unknown",
    ] {
        let i = biome_info(n);
        assert!((-1.8..=1.5).contains(&i.base_height), "{n}");
        assert!((0.0..=1.225).contains(&i.height_variation), "{n}");
    }
}

/// A far node proven uniform is that block wherever the far terrain samples
/// it, at the resolution it was classified for.
#[test]
fn nodes_are_uniform_at_every_resolution_they_claim() {
    let (_, g) = generators(2, "plains").remove(0);
    let mut rng = Lcg(2);
    let mut proven = 0;
    for _ in 0..200 {
        let level = rng.range(0, 3) as u32;
        let (x, z) = (rng.range(-8, 8), rng.range(-8, 8));
        let y = rng.range(-6, 24) >> level;
        let wl = f64::from(1 << rng.range(0, 3));
        let n = g.classify_node(level, x, y, z, wl);
        let Some(b) = n.block else { continue };
        proven += 1;
        let side = 16 << level;
        for _ in 0..60 {
            let p = [
                x * side + rng.range(0, side - 1),
                y * side + rng.range(0, side - 1),
                z * side + rng.range(0, side - 1),
            ];
            assert_eq!(g.block_at(p[0], p[1], p[2], wl as i32), b, "{level} {p:?}");
        }
    }
    assert!(proven > 20, "{proven}");
}
