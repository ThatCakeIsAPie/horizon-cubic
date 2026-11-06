//! Conservative 3D bounds on the density: "is this box uniform?" from one
//! centre sample, plus the per-octave terms a GPU descent can upload.
//!
//! A wrong "uniform" is a visible hole or wall, so every step rounds the safe
//! way and the whole is checked against the lattice generator by a property
//! test (`tests.rs`).
//!
//! # The octave bound
//!
//! One octave is vanilla's improved noise `P` of the point scaled by
//! `f_a * 2^i` per axis, weighted by `A_i = |factor| * 0.5^i / norm`.
//!
//! **The two constants are suprema, certified on a grid.** Inside a lattice
//! cell with local coordinates `u` in `[0, 1]^3`, `P = sum_c w_c(u) t_c(u)`
//! over the 8 corners, with `w_c` the product of the per-axis fades
//! (`f(u) = 6u^5 - 15u^4 + 10u^3` or `1 - f(u)`; `w_c >= 0`, `sum w_c = 1`)
//! and `t_c = g_c . (u - c)`. Every gradient `g_c` in the 16-entry table is
//! one of the 12 vectors with exactly two nonzero components, each `+-1`, and
//! the corners' gradients can be any of them (a permutation table makes them
//! independent for all this bound knows). So for a point `u`:
//!
//! * `max_g |P| = V(u) = sum_c w_c m(u - c)`, where `m(d)` is the sum of the
//!   two largest `|d_i|` (the best of the `+-d_a +- d_b`);
//! * `dP/du_x = sum_c g_c . v_c` with `v_c = (d w_c / d u_x)(u - c) + w_c e_x`,
//!   so `max_g |dP/du_x| = F_x(u) = sum_c m(v_c(u))`. The axes are alike.
//!
//! [`VALUE_BOUND`] bounds `sup V` and [`GRADIENT_BOUND`] bounds `sup F_x`
//! (the largest change of `P` per unit of phase along one axis; a change of
//! phase along several axes adds, as the path may go axis by axis). Both are
//! certified by `bounds_tests::the_octave_constants_are_certified`, which
//! evaluates `V` and `F_x` on a grid of spacing `h = 1/256` over `[0, 1/2]^3`
//! (the functions are symmetric under `u_a -> 1 - u_a`: mirroring an axis
//! permutes the corners and flips signs inside `m`, which ignores signs) and
//! adds the change a function can make within `h/2` per axis. That change is
//! bounded with the derivatives of the weights, `|f'| <= 15/8` and
//! `|f''| <= 10/sqrt 3 = 5.774` (`f'' = 60 u (1 - u) (2u - 1)` peaks at
//! `u = 1/2 +- 1/(2 sqrt 3)`, where `u (1 - u) = 1/6` and `|2u - 1| = 1/sqrt 3`),
//! `sum_c |d w_c / d u_a| = 2 f'_a`, `sum_c |d^2 w_c / d u_a d u_b| <= 4 f'_a f'_b`
//! (`a != b`) and `<= 2 |f''_a|` (`a = b`), and `|u - c| <= 1` per component:
//!
//! * `V` changes by at most `8.5` per unit of one coordinate
//!   (`sum_c |d w_c| m + sum_c w_c |d m| <= 2 * 15/8 * 2 + 1`), so by
//!   `8.5 * 3 * h/2 = 12.75 h` within `h/2` per axis;
//! * `F_x` changes at most `2 sum_c |d^2 w_c/du_k du_x| + sum_c |dw_c/du_x| + sum_c |dw_c/du_k|`
//!   per axis `k`, which is at most `2 * 14.0625 + 3.75 + 3.75 = 35.63`, so by
//!   `36 * 3 * h/2 = 54 h`.
//!
//! The grid gives `sup V = 1.0364` (reached near the cell centre, where all
//! eight corners weigh in) and `sup F_x = 3.75`; with the allowances,
//! `V <= 1.0862` and `F_x <= 3.9609`, so the constants are `1.1` and `4.0`
//! (a finer grid, `h = 1/512`, certifies `1.0613` and `3.8555`). This
//! replaces the first derivation here, which used the cruder
//! `|t_c| <= |dx| + |dy| + |dz|` and gave `1.5` and `6.7`; the bound on the
//! octaves is now about 1.7 times as tight, which proves far more nodes
//! uniform (every `lin` term and every amplitude-bounded octave shrinks).
//!
//! So between two points `p, q`, octave `i` changes by at most
//! `min(2 * V * A_i, A_i * G * 2^i * sum_a |f_a| |p_a - q_a|)`. For a
//! Euclidean distance `r` the sum is at most `|f| r` (Cauchy-Schwarz), which is
//! [`DensityBound::l`]'s form: `A_i * G * |(fX, fY, fZ)| * 2^i * r`.
//! When the octave value `p_i` at the box centre is known, the box range of
//! the octave is `[max(p_i - lin, -V), min(p_i + lin, V)]` (`V` is
//! [`VALUE_BOUND`]), which is what
//! the classifier uses (it is never wider than the cap form).
//!
//! # Through the density formula
//!
//! ```text
//! sel  = clamp(S, 0, 1)          range of S from its octaves; clamp keeps ranges
//! mix  = low + sel * (high - low) = (1 - sel) low + sel high
//!                                 in the hull of the ranges of low and high
//! a    = mix + d                  d = depth transform of the 2D depth noise;
//!                                 the transform is piecewise linear, so its
//!                                 range is taken at the range's ends and kinks
//! D    = a * vol + height - sign(vol) * y
//! ```
//!
//! `height` and `vol` are the blended biome terms of the section corners that
//! cover the box (the per-block values interpolate between them, so the
//! extremes are at the corners; boxes of more than 33 x 33 corners use the
//! global range of the biome table instead). For every corner, every
//! possible branch of `vol *= special` (it applies where `height > y`, so a
//! branch is possible when `height > y_min` / `height <= y_max`), and every
//! possible sign of `vol` (more than the corners' signs only when they
//! differ), `D` is linear in `a` and in `y`, so its range is the interval
//! product exactly. The result is the max and min over all of them: a
//! superset of the density at every point of the box.
//!
//! The generator samples the density on the world-aligned `noiseSampleSize`
//! lattice and interpolates trilinearly, so a block's density is a convex
//! combination of lattice corners. A cube's box is therefore expanded to the
//! lattice points that cover it, and the bound holds for the interpolated
//! blocks too.
//!
//! # The gradient the replacers see
//!
//! `dy` of a block is `(D(y + sy) - D(y)) / sy` interpolated, so
//! `|dy| <= 1 + (|vol|max * dMix + jump) / sy`, where `dMix` bounds the change
//! of `mix` over one lattice edge and `jump` the change of `vol` across the
//! `special` switch. It is infinite if the sign of `vol` can change.
//!
//! # Replacers
//!
//! See [`Replacers::possible_blocks`]: it follows every branch the density
//! range, the gradient bound, the Y range and the biome filters allow and
//! returns the set of blocks that can result. One block means uniform.
//!
//! # Octaves a node skips
//!
//! `min_wavelength` skips octaves like `OctaveNoise::get3` does. The skipped
//! ones are bounded by amplitude alone (`A_i * V` each side) so the answer
//! is valid for the truncated density and for the full one.

use std::rc::Rc;

use super::biomes::{BiomeBlend, ResolvedBiome, convert_height, convert_variation};
use super::density::{Column, OctaveNoise, TerrainNoise};
use super::replacers::{BoxFacts, Replacers};

/// Bound of `|dP/da|` for one axis of one octave (see the module docs; the
/// grid certifies 3.9609).
pub const GRADIENT_BOUND: f64 = 4.0;
/// Bound of `|P|` of one octave (the grid certifies 1.0862).
pub const VALUE_BOUND: f64 = 1.1;
/// Allowance for `f32` rounding inside `PerlinNoise`, per octave value.
pub const F32_SLACK: f64 = 4.0e-6;
/// Added to both ends of every density range.
pub const MARGIN: f64 = 1.0e-3;

/// 1.12's lowest and highest biome heights / variations in the table of
/// [`super::biomes::biome_info`] (a blend is a positive-weighted mean).
const TABLE_HEIGHT: [f64; 2] = [-1.8, 1.5];
const TABLE_VARIATION: [f64; 2] = [0.0, 1.225];
/// Boxes with more section corners than this use the table's global range.
const MAX_CORNERS: i64 = 33 * 33;
/// Boxes with more quart columns than this do not enumerate biomes.
const MAX_QUARTS: i64 = 4096;

/// One octave of a noise as plain data (the GPU twin uploads these).
#[derive(Clone, Debug, PartialEq)]
pub struct OctaveTerm {
    /// `|factor| * 0.5^i / norm`.
    pub amp: f64,
    /// `|f_a| * 2^i` per axis (`fY` is 0 for the planar depth noise).
    pub freq: [f64; 3],
}

impl OctaveTerm {
    /// Largest change of this octave within distance `r`.
    pub fn change(&self, r: f64) -> f64 {
        let len = self.freq.iter().map(|f| f * f).sum::<f64>().sqrt();
        (2.0 * VALUE_BOUND * self.amp).min(self.amp * GRADIENT_BOUND * len * r)
    }

    /// Largest change of this octave between points that differ by at most
    /// `delta` per axis.
    pub fn change_axes(&self, delta: [f64; 3]) -> f64 {
        let lin: f64 = (0..3).map(|a| self.freq[a] * delta[a]).sum();
        (2.0 * VALUE_BOUND * self.amp).min(self.amp * GRADIENT_BOUND * lin)
    }
}

/// One noise's bound terms.
#[derive(Clone, Debug, PartialEq)]
pub struct NoiseTerms {
    pub offset: f64,
    /// The 2D depth noise.
    pub planar: bool,
    pub octaves: Vec<OctaveTerm>,
}

impl NoiseTerms {
    fn new(n: &OctaveNoise, planar: bool) -> Self {
        let f = n.freq();
        Self {
            offset: n.offset(),
            planar,
            octaves: n
                .weights()
                .iter()
                .enumerate()
                .map(|(i, w)| {
                    let s = 2.0_f64.powi(i as i32);
                    OctaveTerm {
                        amp: n.factor().abs() * w,
                        freq: [
                            f[0].abs() * s,
                            if planar { 0.0 } else { f[1].abs() * s },
                            f[2].abs() * s,
                        ],
                    }
                })
                .collect(),
        }
    }

    /// `L(r)` of this noise: how much its value changes within distance `r`.
    pub fn change(&self, r: f64) -> f64 {
        self.octaves.iter().map(|o| o.change(r)).sum()
    }

    fn change_axes(&self, delta: [f64; 3], active: usize) -> f64 {
        self.octaves
            .iter()
            .take(active)
            .map(|o| o.change_axes(delta))
            .sum()
    }

    /// The largest `|value - offset|`.
    fn max_abs(&self) -> f64 {
        VALUE_BOUND * self.octaves.iter().map(|o| o.amp).sum::<f64>()
    }
}

/// Everything a GPU descent needs from the bound, as plain data. The noises
/// are in upload order: selector, low, high, depth.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuTerms {
    pub gradient_bound: f64,
    pub value_bound: f64,
    pub f32_slack: f64,
    pub margin: f64,
    pub noises: [NoiseTerms; 4],
    /// `[heightFactor, heightOffset, heightVariationFactor,
    /// heightVariationOffset, special]`, as [`TerrainNoise::shape`].
    pub shape: [f64; 5],
    /// Range of the blended biome height (before `heightFactor`).
    pub terrain_height: [f64; 2],
    /// Range of the blended biome volatility (before the variation factor).
    pub terrain_volatility: [f64; 2],
    /// Raw depth noise values where the depth transform changes slope.
    pub depth_kinks: [f64; 5],
}

/// The density's change bound, built from the settings' noise tables.
#[derive(Clone, Debug)]
pub struct DensityBound {
    sel: NoiseTerms,
    low: NoiseTerms,
    high: NoiseTerms,
    depth: NoiseTerms,
    shape: [f64; 5],
    vol_abs_max: f64,
}

/// Raw depth values where the transform has a kink.
const DEPTH_KINKS: [f64; 5] = [-10.0 / 3.0, -20.0 / 9.0, 0.0, 2.0 / 3.0, 1.0];

impl DensityBound {
    pub fn new(noise: &TerrainNoise) -> Self {
        let shape = noise.shape();
        let (vmin, vmax) = (
            convert_variation(TABLE_VARIATION[0]),
            convert_variation(TABLE_VARIATION[1]),
        );
        let mut vol_abs_max: f64 = 0.0;
        for raw in [vmin, vmax] {
            for k in [1.0, shape[4]] {
                vol_abs_max = vol_abs_max.max((raw * k * shape[2] + shape[3]).abs());
            }
        }
        Self {
            sel: NoiseTerms::new(noise.selector(), false),
            low: NoiseTerms::new(noise.low(), false),
            high: NoiseTerms::new(noise.high(), false),
            depth: NoiseTerms::new(noise.depth(), true),
            shape,
            vol_abs_max,
        }
    }

    /// An upper bound on how much `mix` (the blended low / high noise) can
    /// change within distance `r`: `max(dLow, dHigh) + dSelector * E`.
    pub fn l_mix(&self, r: f64) -> f64 {
        let e = self.low.max_abs()
            + self.high.max_abs()
            + self.low.offset.abs()
            + self.high.offset.abs();
        self.low.change(r).max(self.high.change(r)) + self.sel.change(r).min(1.0) * e
    }

    /// `L(r)`: an upper bound on how much the density can change within
    /// distance `r` of a point **of one column** (its height, volatility and
    /// depth term fixed): `|vol|max * (dMix + dDepth) + r`. The 2D terms'
    /// change over a region is bounded separately, from the section corners.
    pub fn l(&self, r: f64) -> f64 {
        self.vol_abs_max * (self.l_mix(r) + self.depth_slope() * self.depth.change(r)) + r
    }

    /// The steepest slope of the depth transform: `3 / 5.6` of `0.2 * 17 / 64`
    /// (`d > 0` before the `3d - 2` clamp; `d < 0` is flatter, by `0.9`).
    fn depth_slope(&self) -> f64 {
        3.0 / 5.6 * (0.2 * 17.0 / 64.0)
    }

    pub fn gpu_terms(&self) -> GpuTerms {
        GpuTerms {
            gradient_bound: GRADIENT_BOUND,
            value_bound: VALUE_BOUND,
            f32_slack: F32_SLACK,
            margin: MARGIN,
            noises: [
                self.sel.clone(),
                self.low.clone(),
                self.high.clone(),
                self.depth.clone(),
            ],
            shape: self.shape,
            terrain_height: [
                convert_height(TABLE_HEIGHT[0]),
                convert_height(TABLE_HEIGHT[1]),
            ],
            terrain_volatility: [
                convert_variation(TABLE_VARIATION[0]),
                convert_variation(TABLE_VARIATION[1]),
            ],
            depth_kinks: DEPTH_KINKS,
        }
    }
}

/// What a box is, by density alone (the caller applies sea level: air at or
/// below it is water; [`Analyzer::uniform_block`] does so through the
/// replacers).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoxClass {
    /// Density is `<= 0` everywhere in the box.
    Air,
    /// Density is `> 0` everywhere in the box.
    Solid,
    Mixed,
}

/// A box's class by density, and the one block it is made of after the
/// replacers, when that can be proven.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NodeClass {
    pub class: BoxClass,
    pub block: Option<u32>,
}

/// Bounds of the density over a box, and of its vertical gradient.
#[derive(Clone, Copy, Debug)]
pub struct DensityRange {
    pub min: f64,
    pub max: f64,
    /// An upper bound of `|dy|` (infinite when unknown).
    pub dy: f64,
}

/// What to bound: a box of blocks, the box of sample points their density
/// comes from, and how.
#[derive(Clone, Copy, Debug)]
pub struct Query {
    /// Inclusive block box (for the replacers: Y band, biomes).
    pub blocks: ([i32; 3], [i32; 3]),
    /// Inclusive box of the points whose density the blocks interpolate.
    pub samples: ([i32; 3], [i32; 3]),
    /// The vertical distance of the two samples a gradient is taken from.
    pub dy_edge: f64,
    pub min_wavelength: f64,
    /// A single exact column (its terrain and depth are known).
    pub column: Option<Column>,
}

impl Query {
    /// The blocks `min..=max` as the lattice generator makes them: from the
    /// densities at the lattice points that cover them.
    pub fn lattice(sizes: [i32; 3], min: [i32; 3], max: [i32; 3]) -> Self {
        let lo = std::array::from_fn(|a| min[a].div_euclid(sizes[a]) * sizes[a]);
        let hi = std::array::from_fn(|a| (max[a].div_euclid(sizes[a]) + 1) * sizes[a]);
        Self {
            blocks: (min, max),
            samples: (lo, hi),
            dy_edge: f64::from(sizes[1]),
            min_wavelength: 0.0,
            column: None,
        }
    }

    /// Densities evaluated at points of the box itself (far terrain). The
    /// replacers' gradient is taken over `min(min_wavelength, 8)` blocks, as
    /// `block_at` does (1 for the full density).
    pub fn exact(min: [i32; 3], max: [i32; 3], min_wavelength: f64) -> Self {
        Self {
            blocks: (min, max),
            samples: (min, max),
            dy_edge: min_wavelength.clamp(1.0, 8.0),
            min_wavelength,
            column: None,
        }
    }

    /// One column's points `y0..=y1` with its terrain known.
    pub fn column(x: i32, z: i32, y0: i32, y1: i32, column: Column) -> Self {
        Self {
            blocks: ([x, y0, z], [x, y1, z]),
            samples: ([x, y0, z], [x, y1, z]),
            dy_edge: 1.0,
            min_wavelength: 0.0,
            column: Some(column),
        }
    }
}

/// The range of a noise over a box given by its centre and half extents.
fn noise_range(n: &OctaveNoise, planar: bool, c: [f64; 3], h: [f64; 3], mwl: f64) -> (f64, f64) {
    let active = n.active_octaves(mwl, planar);
    let f = n.freq();
    let mut lo = 0.0;
    let mut hi = 0.0;
    for (i, w) in n.weights().iter().enumerate() {
        if i >= active {
            lo -= w * VALUE_BOUND;
            hi += w * VALUE_BOUND;
            continue;
        }
        let s = 2.0_f64.powi(i as i32);
        let p = n.octave_value(i, c[0], c[1], c[2], planar);
        let fy = if planar { 0.0 } else { f[1].abs() };
        let lin =
            GRADIENT_BOUND * s * (f[0].abs() * h[0] + fy * h[1] + f[2].abs() * h[2]) + F32_SLACK;
        lo += w * (p - lin).max(-VALUE_BOUND - F32_SLACK);
        hi += w * (p + lin).min(VALUE_BOUND + F32_SLACK);
    }
    let (a, b) = (lo * n.factor(), hi * n.factor());
    let (a, b) = if n.factor() < 0.0 { (b, a) } else { (a, b) };
    (a + n.offset(), b + n.offset())
}

/// The range of the depth transform of raw values in `[a, b]`.
fn depth_range(a: f64, b: f64) -> (f64, f64) {
    let mut lo = TerrainNoise::depth_transform(a).min(TerrainNoise::depth_transform(b));
    let mut hi = TerrainNoise::depth_transform(a).max(TerrainNoise::depth_transform(b));
    for k in DEPTH_KINKS {
        if k > a && k < b {
            let v = TerrainNoise::depth_transform(k);
            lo = lo.min(v);
            hi = hi.max(v);
        }
    }
    (lo, hi)
}

/// Runs the bounds against one generator's parts.
pub struct Analyzer<'a> {
    pub noise: &'a TerrainNoise,
    pub blend: &'a BiomeBlend,
    pub replacers: &'a Replacers,
    pub bound: &'a DensityBound,
    pub sizes: [i32; 3],
}

impl Analyzer<'_> {
    /// The `(height, volatility)` corner values (before the height factor)
    /// covering the columns of a sample box.
    fn corners(&self, lo: [i32; 3], hi: [i32; 3]) -> Vec<(f64, f64)> {
        let (qx0, qx1) = (lo[0] >> 2, (hi[0] + 3) >> 2);
        let (qz0, qz1) = (lo[2] >> 2, (hi[2] + 3) >> 2);
        let count = i64::from(qx1 - qx0 + 1) * i64::from(qz1 - qz0 + 1);
        if count > MAX_CORNERS {
            let (h, v) = (TABLE_HEIGHT, TABLE_VARIATION);
            return [h[0], h[1]]
                .into_iter()
                .flat_map(|h| {
                    [v[0], v[1]]
                        .into_iter()
                        .map(move |v| (convert_height(h), convert_variation(v)))
                })
                .collect();
        }
        let mut out = Vec::with_capacity(count as usize);
        for qz in qz0..=qz1 {
            for qx in qx0..=qx1 {
                let t = self.blend.terrain_at(qx * 4, qz * 4);
                out.push((t.height, t.volatility));
            }
        }
        out
    }

    /// Bounds of the density over the query's sample box.
    pub fn density_range(&self, q: &Query) -> Option<DensityRange> {
        let (lo, hi) = q.samples;
        let c: [f64; 3] = std::array::from_fn(|a| f64::from(lo[a] + hi[a]) * 0.5);
        let h: [f64; 3] = std::array::from_fn(|a| f64::from(hi[a] - lo[a]) * 0.5);
        let mwl = q.min_wavelength;
        let n = self.noise;
        let (s0, s1) = noise_range(n.selector(), false, c, h, mwl);
        let (l0, l1) = noise_range(n.low(), false, c, h, mwl);
        let (h0, h1) = noise_range(n.high(), false, c, h, mwl);
        let (c0, c1) = (s0.clamp(0.0, 1.0), s1.clamp(0.0, 1.0));
        let (m0, m1) = if c1 <= 0.0 {
            (l0, l1)
        } else if c0 >= 1.0 {
            (h0, h1)
        } else {
            (l0.min(h0), l1.max(h1))
        };
        let (d0, d1) = match q.column {
            Some(col) => (col.depth, col.depth),
            None => {
                let (e0, e1) = noise_range(n.depth(), true, c, [h[0], 0.0, h[2]], mwl);
                depth_range(e0, e1)
            }
        };
        let (a0, a1) = (m0 + d0, m1 + d1);
        if [a0, a1].iter().any(|v| v.is_nan()) {
            return None;
        }
        let cols = match q.column {
            Some(col) => vec![(col.height, col.volatility)],
            None => self.corners(lo, hi),
        };
        let [hf, ho, vf, vo, special] = n.shape();
        let (ylo, yhi) = (f64::from(lo[1]), f64::from(hi[1]));
        let heights: Vec<f64> = cols.iter().map(|c| c.0 * hf + ho).collect();
        let (hmin, hmax) = heights
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| {
                (a.min(v), b.max(v))
            });
        let special_possible = hmax > ylo;
        let normal_possible = hmin <= yhi;
        let mut branches = Vec::with_capacity(2);
        if special_possible {
            branches.push(special);
        }
        if normal_possible || !special_possible {
            branches.push(1.0);
        }
        // Signs of the volatility over the corners and branches.
        let mut signs = [false; 3];
        for c in &cols {
            for &k in &branches {
                let v = c.1 * k * vf + vo;
                signs[if v > 0.0 {
                    2
                } else if v < 0.0 {
                    0
                } else {
                    1
                }] = true;
            }
        }
        let distinct = signs.iter().filter(|s| **s).count();
        if distinct > 1 {
            signs = [true; 3];
        }
        let mut dmin = f64::INFINITY;
        let mut dmax = f64::NEG_INFINITY;
        for (c, &height) in cols.iter().zip(&heights) {
            for &k in &branches {
                let vol = c.1 * k * vf + vo;
                let (p0, p1) = (a0 * vol, a1 * vol);
                let (pmin, pmax) = (p0.min(p1), p0.max(p1));
                for (idx, on) in signs.iter().enumerate() {
                    if !on {
                        continue;
                    }
                    // -sign * y over [ylo, yhi]
                    let (t0, t1) = match idx {
                        2 => (-yhi, -ylo),
                        0 => (ylo, yhi),
                        _ => (0.0, 0.0),
                    };
                    dmin = dmin.min(pmin + height + t0);
                    dmax = dmax.max(pmax + height + t1);
                }
            }
        }
        if !dmin.is_finite() || !dmax.is_finite() {
            return None;
        }
        // The vertical gradient bound.
        let dy = if distinct > 1 {
            f64::INFINITY
        } else {
            let edge = q.dy_edge.max(1e-9);
            let b = self.bound;
            let ax = [0.0, edge, 0.0];
            let act = |nz: &OctaveNoise| nz.active_octaves(mwl, false);
            let dl = b.low.change_axes(ax, act(n.low()));
            let dh = b.high.change_axes(ax, act(n.high()));
            let ds = b.sel.change_axes(ax, act(n.selector())).min(1.0);
            let e = (h1 - l0).abs().max((h0 - l1).abs());
            let d_mix = dl.max(dh) + ds * e;
            let vol_abs = cols
                .iter()
                .flat_map(|c| branches.iter().map(move |&k| (c.1 * k * vf + vo).abs()))
                .fold(0.0_f64, f64::max);
            let jump = if branches.len() > 1 {
                let raw = cols.iter().map(|c| c.1.abs()).fold(0.0_f64, f64::max);
                (special - 1.0).abs() * raw * vf.abs() * a0.abs().max(a1.abs())
            } else {
                0.0
            };
            1.0 + (vol_abs * d_mix + jump) / edge
        };
        let pad = MARGIN + 1e-9 * dmin.abs().max(dmax.abs());
        Some(DensityRange {
            min: dmin - pad,
            max: dmax + pad,
            dy,
        })
    }

    /// The biomes of a block box's columns, `None` for very wide boxes.
    fn biomes(&self, min: [i32; 3], max: [i32; 3]) -> Option<Vec<Rc<ResolvedBiome>>> {
        let (qx0, qx1) = (min[0] >> 2, max[0] >> 2);
        let (qz0, qz1) = (min[2] >> 2, max[2] >> 2);
        if i64::from(qx1 - qx0 + 1) * i64::from(qz1 - qz0 + 1) > MAX_QUARTS {
            return None;
        }
        let mut out: Vec<Rc<ResolvedBiome>> = Vec::new();
        for qz in qz0..=qz1 {
            for qx in qx0..=qx1 {
                let b = self.blend.biome_at(qx * 4, qz * 4);
                if !out.iter().any(|o| Rc::ptr_eq(o, &b)) {
                    out.push(b);
                }
            }
        }
        Some(out)
    }

    /// Air / solid / mixed by density alone.
    pub fn classify(&self, q: &Query) -> BoxClass {
        match self.density_range(q) {
            Some(r) if r.max <= 0.0 => BoxClass::Air,
            Some(r) if r.min > 0.0 => BoxClass::Solid,
            _ => BoxClass::Mixed,
        }
    }

    /// The one block every block of the query's box ends up as after
    /// density and replacers (no carvers), when that can be proven.
    pub fn uniform_block(&self, q: &Query) -> Option<u32> {
        let r = self.density_range(q)?;
        let biomes = self.biomes(q.blocks.0, q.blocks.1);
        let facts = BoxFacts {
            dmin: r.min,
            dmax: r.max,
            dy: r.dy,
            y0: q.blocks.0[1],
            y1: q.blocks.1[1],
            biomes: biomes.as_deref(),
        };
        match self.replacers.possible_blocks(&facts)?.as_slice() {
            [b] => Some(*b),
            _ => None,
        }
    }

    /// Both of the above, for a far-terrain node.
    pub fn node(&self, q: &Query) -> NodeClass {
        NodeClass {
            class: self.classify(q),
            block: self.uniform_block(q),
        }
    }
}
