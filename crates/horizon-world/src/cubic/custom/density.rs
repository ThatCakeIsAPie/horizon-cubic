//! The terrain density of the custom generator (CWG stages 2 to 5).
//!
//! # Octave formula (shared with the GPU twin; do not change)
//!
//! A noise has `factor`, `offset`, frequencies `fX, fY, fZ` and `octaves`.
//! With `P_i` the `PerlinNoise::get3` of octave `i` (vanilla's improved
//! noise):
//!
//! ```text
//! norm  = sum_{i<octaves} 0.5^i
//! value = factor * sum_{i<octaves} (0.5^i / norm)
//!                 * P_i(wrap(x*fX*2^i), wrap(y*fY*2^i), wrap(z*fZ*2^i))
//!         + offset
//! ```
//!
//! The 2D depth noise uses `P_i.get3(wrap(x*fX*2^i), 0.0, wrap(z*fZ*2^i))`.
//! `wrap` is `horizon_core::noise::wrap`.
//!
//! # Seeding
//!
//! One `XoroshiroRandom::new(world_seed ^ NOISE_SEED_SALT)` builds the
//! octaves in this order: selector (octave 0 first ... last), low, high,
//! depth; each octave is `PerlinNoise::new(&mut random)`.
//! [`TerrainNoise::tables`] exposes them as [`NoiseTable`]s in the same
//! order, which map onto the GPU twin's `NoiseDesc` / `OctaveDesc`.
//!
//! # Density
//!
//! ```text
//! selector = clamp(sel * ..., 0, 1); mix = low + selector * (high - low)
//! d = depth(x, z); if d < 0 { d *= -0.3 }; d = clamp(d * 3 - 2, -2, 1)
//! d /= (d < 0 ? 5.6 : d > 0 ? 8 : 1); d *= 0.2 * 17 / 64
//! height = biomeHeight * heightFactor + heightOffset
//! vol    = biomeVolatility * (height > y ? special : 1)
//!          * heightVariationFactor + heightVariationOffset
//! density = (mix + d) * vol + height - signum(vol) * y
//! ```
//!
//! It is sampled on the world-aligned `noiseSampleSize` lattice and
//! interpolated trilinearly ([`Lattice`]), with the per-block gradients of
//! `IBuilder.forEachScaled` (the exact partial derivatives of the trilinear
//! function), so cubes agree on shared faces.

use horizon_core::noise::{PerlinNoise, wrap};
use horizon_core::random::XoroshiroRandom;

use super::biomes::{BiomeBlend, ColumnTerrain};
use super::settings::CustomGeneratorSettings;

/// Mixed into the world seed before the noises are built.
pub const NOISE_SEED_SALT: i64 = 0x4357_4743_5542_4943;

/// One octave's tables, in the GPU twin's shape.
#[derive(Clone, Debug, PartialEq)]
pub struct OctaveTable {
    pub permutations: [u8; 256],
    pub offsets: [f64; 3],
}

/// One noise's description for upload.
#[derive(Clone, Debug, PartialEq)]
pub struct NoiseTable {
    pub factor: f64,
    pub offset: f64,
    pub frequency: [f64; 3],
    pub octaves: Vec<OctaveTable>,
}

/// An octave sum with the formula in the module docs.
#[derive(Clone, Debug)]
pub struct OctaveNoise {
    factor: f64,
    offset: f64,
    freq: [f64; 3],
    octaves: Vec<PerlinNoise>,
    weights: Vec<f64>,
}

impl OctaveNoise {
    pub fn new(
        factor: f64,
        offset: f64,
        freq: [f64; 3],
        octaves: i32,
        random: &mut XoroshiroRandom,
    ) -> Self {
        let n = octaves.max(1) as usize;
        let norm: f64 = (0..n).map(|i| 0.5_f64.powi(i as i32)).sum();
        Self {
            factor,
            offset,
            freq,
            octaves: (0..n).map(|_| PerlinNoise::new(random)).collect(),
            weights: (0..n).map(|i| 0.5_f64.powi(i as i32) / norm).collect(),
        }
    }

    /// The 3D value; octaves whose wavelength (`1 / (max f * 2^i)` blocks) is
    /// below `min_wavelength` are skipped (octave 0 never is).
    pub fn get3(&self, x: f64, y: f64, z: f64, min_wavelength: f64) -> f64 {
        let fmax = self.freq.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
        let mut sum = 0.0;
        let mut scale = 1.0;
        for (i, p) in self.octaves.iter().enumerate() {
            if i > 0 && min_wavelength > 0.0 && fmax > 0.0 && 1.0 / (fmax * scale) < min_wavelength
            {
                break;
            }
            let v = p.get3(
                wrap(x * self.freq[0] * scale),
                wrap(y * self.freq[1] * scale),
                wrap(z * self.freq[2] * scale),
            );
            sum += self.weights[i] * f64::from(v);
            scale *= 2.0;
        }
        self.factor * sum + self.offset
    }

    /// The 2D value used by the depth noise (`y` is 0 in every octave).
    pub fn get2(&self, x: f64, z: f64, min_wavelength: f64) -> f64 {
        let fmax = self.freq[0].abs().max(self.freq[2].abs());
        let mut sum = 0.0;
        let mut scale = 1.0;
        for (i, p) in self.octaves.iter().enumerate() {
            if i > 0 && min_wavelength > 0.0 && fmax > 0.0 && 1.0 / (fmax * scale) < min_wavelength
            {
                break;
            }
            let v = p.get3(
                wrap(x * self.freq[0] * scale),
                0.0,
                wrap(z * self.freq[2] * scale),
            );
            sum += self.weights[i] * f64::from(v);
            scale *= 2.0;
        }
        self.factor * sum + self.offset
    }

    pub fn table(&self) -> NoiseTable {
        NoiseTable {
            factor: self.factor,
            offset: self.offset,
            frequency: self.freq,
            octaves: self
                .octaves
                .iter()
                .map(|p| OctaveTable {
                    permutations: *p.table().permutations(),
                    offsets: *p.table().offsets(),
                })
                .collect(),
        }
    }
}

/// The four noises and the shape parameters of the density.
#[derive(Clone, Debug)]
pub struct TerrainNoise {
    selector: OctaveNoise,
    low: OctaveNoise,
    high: OctaveNoise,
    depth: OctaveNoise,
    height_factor: f64,
    height_offset: f64,
    variation_factor: f64,
    variation_offset: f64,
    special: f64,
}

/// Per-column inputs of the density: blended biome terrain and the depth
/// transform.
#[derive(Clone, Copy, Debug)]
pub struct Column {
    pub height: f64,
    pub volatility: f64,
    pub depth: f64,
}

impl TerrainNoise {
    pub fn new(seed: i64, s: &CustomGeneratorSettings) -> Self {
        let mut r = XoroshiroRandom::new(seed ^ NOISE_SEED_SALT);
        let selector = OctaveNoise::new(
            s.selector_noise_factor,
            s.selector_noise_offset,
            [
                s.selector_noise_frequency_x,
                s.selector_noise_frequency_y,
                s.selector_noise_frequency_z,
            ],
            s.selector_noise_octaves,
            &mut r,
        );
        let low = OctaveNoise::new(
            s.low_noise_factor,
            s.low_noise_offset,
            [
                s.low_noise_frequency_x,
                s.low_noise_frequency_y,
                s.low_noise_frequency_z,
            ],
            s.low_noise_octaves,
            &mut r,
        );
        let high = OctaveNoise::new(
            s.high_noise_factor,
            s.high_noise_offset,
            [
                s.high_noise_frequency_x,
                s.high_noise_frequency_y,
                s.high_noise_frequency_z,
            ],
            s.high_noise_octaves,
            &mut r,
        );
        let depth = OctaveNoise::new(
            s.depth_noise_factor,
            s.depth_noise_offset,
            [s.depth_noise_frequency_x, 0.0, s.depth_noise_frequency_z],
            s.depth_noise_octaves,
            &mut r,
        );
        Self {
            selector,
            low,
            high,
            depth,
            height_factor: s.height_factor,
            height_offset: s.height_offset,
            variation_factor: s.height_variation_factor,
            variation_offset: s.height_variation_offset,
            special: s.special_height_variation_factor_below_average_y,
        }
    }

    /// Selector, low, high and depth tables (the upload order).
    pub fn tables(&self) -> [NoiseTable; 4] {
        [
            self.selector.table(),
            self.low.table(),
            self.high.table(),
            self.depth.table(),
        ]
    }

    /// `(height factor, height offset, variation factor, variation offset,
    /// special factor)` for the GPU twin.
    pub fn shape(&self) -> [f64; 5] {
        [
            self.height_factor,
            self.height_offset,
            self.variation_factor,
            self.variation_offset,
            self.special,
        ]
    }

    /// The depth transform of a column.
    pub fn depth_term(&self, x: i32, z: i32, min_wavelength: f64) -> f64 {
        let mut d = self.depth.get2(f64::from(x), f64::from(z), min_wavelength);
        if d < 0.0 {
            d *= -0.3;
        }
        d = (d * 3.0 - 2.0).clamp(-2.0, 1.0);
        if d < 0.0 {
            d /= 2.0 * 2.0 * 1.4;
        } else if d > 0.0 {
            d /= 8.0;
        }
        d * (0.2 * 17.0 / 64.0)
    }

    pub fn column(&self, x: i32, z: i32, t: ColumnTerrain, min_wavelength: f64) -> Column {
        Column {
            height: t.height,
            volatility: t.volatility,
            depth: self.depth_term(x, z, min_wavelength),
        }
    }

    /// The density at a block corner given its column.
    pub fn density(&self, x: i32, y: i32, z: i32, c: &Column, min_wavelength: f64) -> f64 {
        let (fx, fy, fz) = (f64::from(x), f64::from(y), f64::from(z));
        let sel = self
            .selector
            .get3(fx, fy, fz, min_wavelength)
            .clamp(0.0, 1.0);
        let mix = if sel <= 0.0 {
            self.low.get3(fx, fy, fz, min_wavelength)
        } else if sel >= 1.0 {
            self.high.get3(fx, fy, fz, min_wavelength)
        } else {
            let l = self.low.get3(fx, fy, fz, min_wavelength);
            let h = self.high.get3(fx, fy, fz, min_wavelength);
            l + sel * (h - l)
        };
        let height = c.height * self.height_factor + self.height_offset;
        let mut vol = c.volatility;
        if height > fy {
            vol *= self.special;
        }
        vol = vol * self.variation_factor + self.variation_offset;
        let sign = if vol > 0.0 {
            1.0
        } else if vol < 0.0 {
            -1.0
        } else {
            0.0
        };
        (mix + c.depth) * vol + height - sign * fy
    }
}

/// The sample lattice of one cube: densities at the corners of the
/// `noiseSampleSize` cells that cover it.
pub struct Lattice {
    pub sx: i32,
    pub sy: i32,
    pub sz: i32,
    /// Corners per axis.
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    values: Vec<f64>,
}

impl Lattice {
    /// Samples the corners of the cube `(cx, cy, cz)`.
    pub fn for_cube(
        noise: &TerrainNoise,
        blend: &BiomeBlend,
        sizes: [i32; 3],
        cube: [i32; 3],
    ) -> Self {
        let [sx, sy, sz] = sizes;
        let (nx, ny, nz) = (
            (16 / sx + 1) as usize,
            (16 / sy + 1) as usize,
            (16 / sz + 1) as usize,
        );
        let mut values = vec![0.0; nx * ny * nz];
        for k in 0..nz {
            let z = cube[2] * 16 + k as i32 * sz;
            for i in 0..nx {
                let x = cube[0] * 16 + i as i32 * sx;
                let col = noise.column(x, z, blend.terrain_at(x, z), 0.0);
                for j in 0..ny {
                    let y = cube[1] * 16 + j as i32 * sy;
                    values[(j * nz + k) * nx + i] = noise.density(x, y, z, &col, 0.0);
                }
            }
        }
        Self {
            sx,
            sy,
            sz,
            nx,
            ny,
            nz,
            values,
        }
    }

    fn at(&self, i: usize, j: usize, k: usize) -> f64 {
        self.values[(j * self.nz + k) * self.nx + i]
    }

    /// Calls `f(local x, local y, local z, dx, dy, dz, density)` for all 4096
    /// blocks, in `y, z, x` order, like `forEachScaled`.
    pub fn for_each(&self, mut f: impl FnMut(usize, usize, usize, f64, f64, f64, f64)) {
        let (sx, sy, sz) = (self.sx as usize, self.sy as usize, self.sz as usize);
        let (stx, sty, stz) = (1.0 / sx as f64, 1.0 / sy as f64, 1.0 / sz as f64);
        for cj in 0..(self.ny - 1) {
            for ck in 0..(self.nz - 1) {
                for ci in 0..(self.nx - 1) {
                    let c = |a: usize, b: usize, d: usize| self.at(ci + a, cj + b, ck + d);
                    // v[x][y][z]
                    let v = [
                        [[c(0, 0, 0), c(0, 0, 1)], [c(0, 1, 0), c(0, 1, 1)]],
                        [[c(1, 0, 0), c(1, 0, 1)], [c(1, 1, 0), c(1, 1, 1)]],
                    ];
                    // d/dy per (x edge, z edge)
                    let dy_corner = |a: usize, d: usize| (v[a][1][d] - v[a][0][d]) * sty;
                    for yr in 0..sy {
                        let fy = yr as f64 * sty;
                        let a = |xi: usize, zi: usize| {
                            v[xi][0][zi] + fy * (v[xi][1][zi] - v[xi][0][zi])
                        };
                        let (a00, a01, a10, a11) = (a(0, 0), a(0, 1), a(1, 0), a(1, 1));
                        for zr in 0..sz {
                            let fz = zr as f64 * stz;
                            let b0 = a00 + fz * (a01 - a00);
                            let b1 = a10 + fz * (a11 - a10);
                            let dx = (b1 - b0) * stx;
                            for xr in 0..sx {
                                let fx = xr as f64 * stx;
                                let val = b0 + fx * (b1 - b0);
                                let dz =
                                    ((a01 + fx * (a11 - a01)) - (a00 + fx * (a10 - a00))) * stz;
                                let dy0 =
                                    dy_corner(0, 0) + fz * (dy_corner(0, 1) - dy_corner(0, 0));
                                let dy1 =
                                    dy_corner(1, 0) + fz * (dy_corner(1, 1) - dy_corner(1, 0));
                                let dy = dy0 + fx * (dy1 - dy0);
                                f(ci * sx + xr, cj * sy + yr, ck * sz + zr, dx, dy, dz, val);
                            }
                        }
                    }
                }
            }
        }
    }
}

/// The trilinear value and gradient of the lattice cell containing a block,
/// for point queries (same numbers as [`Lattice::for_each`]).
pub fn cell_at(
    noise: &TerrainNoise,
    blend: &BiomeBlend,
    sizes: [i32; 3],
    x: i32,
    y: i32,
    z: i32,
) -> (f64, f64, f64, f64) {
    let [sx, sy, sz] = sizes;
    let (x0, y0, z0) = (
        x.div_euclid(sx) * sx,
        y.div_euclid(sy) * sy,
        z.div_euclid(sz) * sz,
    );
    let mut v = [[[0.0; 2]; 2]; 2];
    for (a, row) in v.iter_mut().enumerate() {
        let xx = x0 + a as i32 * sx;
        for d in 0..2 {
            let zz = z0 + d as i32 * sz;
            let col = noise.column(xx, zz, blend.terrain_at(xx, zz), 0.0);
            for (b, plane) in row.iter_mut().enumerate() {
                plane[d] = noise.density(xx, y0 + b as i32 * sy, zz, &col, 0.0);
            }
        }
    }
    let (fx, fy, fz) = (
        f64::from(x - x0) / f64::from(sx),
        f64::from(y - y0) / f64::from(sy),
        f64::from(z - z0) / f64::from(sz),
    );
    let a = |xi: usize, zi: usize| v[xi][0][zi] + fy * (v[xi][1][zi] - v[xi][0][zi]);
    let (a00, a01, a10, a11) = (a(0, 0), a(0, 1), a(1, 0), a(1, 1));
    let b0 = a00 + fz * (a01 - a00);
    let b1 = a10 + fz * (a11 - a10);
    let val = b0 + fx * (b1 - b0);
    let dx = (b1 - b0) / f64::from(sx);
    let dz = ((a01 + fx * (a11 - a01)) - (a00 + fx * (a10 - a00))) / f64::from(sz);
    let dyc = |xi: usize, zi: usize| (v[xi][1][zi] - v[xi][0][zi]) / f64::from(sy);
    let dy0 = dyc(0, 0) + fz * (dyc(0, 1) - dyc(0, 0));
    let dy1 = dyc(1, 0) + fz * (dyc(1, 1) - dyc(1, 0));
    let dy = dy0 + fx * (dy1 - dy0);
    (val, dx, dy, dz)
}

#[cfg(test)]
mod tests {
    use super::super::biomes::{BiomeResolver, FixedBiome, SURFACE_BLOCKS};
    use super::*;
    use std::sync::Arc;

    fn setup() -> (TerrainNoise, BiomeBlend, [i32; 3]) {
        let s = CustomGeneratorSettings::defaults();
        let blocks = SURFACE_BLOCKS
            .iter()
            .enumerate()
            .map(|(i, n)| (*n, i as u32))
            .collect();
        let blend = BiomeBlend::new(
            Arc::new(FixedBiome::new("plains")),
            BiomeResolver::new(blocks, Vec::new()),
        );
        (
            TerrainNoise::new(1234, &s),
            blend,
            [
                s.noise_sample_size_x,
                s.noise_sample_size_y,
                s.noise_sample_size_z,
            ],
        )
    }

    #[test]
    fn neighbouring_cubes_agree_on_shared_faces() {
        let (n, b, sizes) = setup();
        let a = Lattice::for_cube(&n, &b, sizes, [3, 4, -2]);
        let c = Lattice::for_cube(&n, &b, sizes, [4, 4, -2]);
        let mut left = std::collections::HashMap::new();
        a.for_each(|x, y, z, _, _, _, v| {
            if x == 15 {
                left.insert((y, z), v);
            }
        });
        // The first column of the next cube continues the same function:
        // compare its value to the point query of the neighbouring block.
        let mut checked = 0;
        c.for_each(|x, y, z, _, _, _, v| {
            if x == 0 && y % 5 == 0 {
                let (pv, ..) = cell_at(&n, &b, sizes, 64, 64 + y as i32, -32 + z as i32);
                assert!((pv - v).abs() < 1e-9);
                checked += 1;
            }
        });
        assert!(checked > 0 && !left.is_empty());
        // And a point query on the last column of the first cube is the same.
        let (pv, ..) = cell_at(&n, &b, sizes, 63, 64 + 3, -32 + 7);
        assert!((pv - left[&(3, 7)]).abs() < 1e-9);
    }

    #[test]
    fn terrain_is_solid_far_below_and_air_far_above() {
        let (n, b, _) = setup();
        let col = n.column(0, 0, b.terrain_at(0, 0), 0.0);
        assert!(n.density(0, -5000, 0, &col, 0.0) > 0.0);
        assert!(n.density(0, 5000, 0, &col, 0.0) < 0.0);
    }

    #[test]
    fn tables_have_one_entry_per_octave() {
        let (n, _, _) = setup();
        let t = n.tables();
        assert_eq!(t[0].octaves.len(), 8);
        assert_eq!(t[1].octaves.len(), 16);
        assert_eq!(t[3].octaves.len(), 16);
    }
}
