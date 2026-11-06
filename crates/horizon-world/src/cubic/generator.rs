//! Terrain for the cubes of a cubic Overworld that vanilla's generator does not
//! cover: the deep (cube Y < -4, blocks below -64) and the sky (cube Y >= 20,
//! blocks from 320 up).
//!
//! Everything is a function of world coordinates and the seed, so a cube and
//! its neighbours agree on every shared face and no cube needs another.
//!
//! * The deep is deepslate with tuff blobs, "cheese" caverns and "spaghetti"
//!   tunnels, lava in cave air under a per-region level, ores, lava/obsidian
//!   pockets and geodes. The cave fields are sampled on a world-aligned lattice
//!   of 4 blocks and interpolated trilinearly, in [`CubeGenerator::generate`]
//!   and [`CubeGenerator::block_at`] alike, so both give the same blocks.
//! * The sky is air with floating islands from y 600, in bands every 1000
//!   blocks. Each cell of 160 x 160 blocks holds at most one island per band,
//!   wholly inside the cell.
//!
//! Features (ores, pockets, geodes) are blobs placed per world-aligned cell by
//! hashing the cell, so they are the same from either side of a cube border.
//! [`CubeGenerator::block_at`] skips them (the far-terrain LOD does not need
//! them); [`CubeGenerator::block_at_with_features`] does not.

use horizon_core::noise::PerlinNoise;
use horizon_core::position::SectionPos;
use horizon_core::random::{PositionalRandom, RandomSource, XoroshiroRandom, mix_stafford13};

use crate::registry::BlockRegistry;

/// The lowest cube Y the sky generator owns.
const FIRST_SKY_CUBE: i32 = 20;
/// The highest cube Y the deep generator owns.
const LAST_DEEP_CUBE: i32 = -5;

/// Caves fade in between these block Ys, so the deep joins vanilla's bottom.
const CAVE_RAMP_TOP: i32 = -68;
const CAVE_RAMP_BOTTOM: i32 = -80;
const CHEESE_THRESHOLD: f32 = 0.5;
const SPAGHETTI_THRESHOLD: f32 = 0.085;
const TUFF_THRESHOLD: f32 = 0.32;
/// Slack when deciding that a lattice cell cannot hold a cave or tuff.
const EPS: f32 = 1e-3;
/// Returned by the deep block function for a cave (air or lava).
const CAVE: u32 = u32::MAX;
const UNSET: i32 = i32::MAX;

const ISLAND_BASE_Y: i32 = 600;
const ISLAND_BAND: i32 = 1000;
const ISLAND_SPREAD: i32 = 360;
const ISLAND_CELL: i32 = 160;
const ISLAND_CHANCE: f64 = 0.4;

const ORE_CELL: i32 = 16;
const ORE_REACH: i32 = 4;
const POCKET_REACH: i32 = 5;
const GEODE_CELL: i32 = 32;
const GEODE_REACH: i32 = 10;
/// Ores are placed only in cells centred below this block Y.
const FEATURE_TOP: i32 = -72;

/// One ore and the variant it has in deepslate.
#[derive(Clone, Copy, Debug)]
pub struct OreKind {
    pub ore: u32,
    pub deepslate_ore: u32,
}

/// Block state ids the generator writes, by role.
#[derive(Clone, Debug)]
pub struct CubePalette {
    pub air: u32,
    pub stone: u32,
    pub deepslate: u32,
    pub tuff: u32,
    pub lava: u32,
    /// Reserved: nothing writes water yet.
    pub water: u32,
    pub grass_block: u32,
    pub dirt: u32,
    pub gravel: u32,
    pub calcite: u32,
    pub amethyst: u32,
    pub obsidian: u32,
    /// Coal, copper, iron, gold, redstone, lapis, diamond, emerald, in that
    /// order. A shorter list just leaves the later ores out.
    pub ores: Vec<OreKind>,
}

impl CubePalette {
    /// The default state of each block, by vanilla name.
    pub fn from_registry(registry: &BlockRegistry) -> Result<Self, String> {
        let id = |name: &str| -> Result<u32, String> {
            registry
                .default_state(&format!("minecraft:{name}"))
                .map(|s| s.0)
                .ok_or_else(|| format!("the block registry has no minecraft:{name}"))
        };
        let mut ores = Vec::new();
        for name in ORE_NAMES {
            ores.push(OreKind {
                ore: id(&format!("{name}_ore"))?,
                deepslate_ore: id(&format!("deepslate_{name}_ore"))?,
            });
        }
        Ok(Self {
            air: id("air")?,
            stone: id("stone")?,
            deepslate: id("deepslate")?,
            tuff: id("tuff")?,
            lava: id("lava")?,
            water: id("water")?,
            grass_block: id("grass_block")?,
            dirt: id("dirt")?,
            gravel: id("gravel")?,
            calcite: id("calcite")?,
            amethyst: id("amethyst_block")?,
            obsidian: id("obsidian")?,
            ores,
        })
    }
}

const ORE_NAMES: [&str; 8] = [
    "coal", "copper", "iron", "gold", "redstone", "lapis", "diamond", "emerald",
];

/// Expected veins per 16 x 16 x 16 cell of ore `kind` at `depth` blocks below -64.
fn ore_rate(kind: usize, depth: f64) -> f64 {
    let fade = |from: f64, to: f64| from + (to - from) * (depth / 1200.0).min(1.0);
    match kind {
        0 => fade(0.14, 0.02),
        1 => fade(0.12, 0.02),
        2 => 0.3,
        3 => 0.18,
        4 => 0.28,
        5 => 0.2,
        6 => fade(0.02, 0.1),
        _ => 0.012,
    }
}

fn hash(seed: u64, salt: u64, x: i32, y: i32, z: i32) -> u64 {
    let mut h = mix_stafford13(seed ^ salt.wrapping_mul(0x9e37_79b9_7f4a_7c15));
    h = mix_stafford13(h ^ x as u32 as u64);
    h = mix_stafford13(h ^ (y as u32 as u64).rotate_left(21));
    mix_stafford13(h ^ (z as u32 as u64).rotate_left(42))
}

/// A small stream of rolls for one feature cell.
struct Rolls(u64);

impl Rolls {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        mix_stafford13(self.0)
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }
    fn int(&mut self, lo: i32, hi: i32) -> i32 {
        lo + (self.unit() * f64::from(hi - lo + 1)) as i32
    }
}

fn cell_range(lo: i32, hi: i32, reach: i32, size: i32) -> (i32, i32) {
    ((lo - reach).div_euclid(size), (hi + reach).div_euclid(size))
}

fn lerp3(c: &[f32; 8], fx: f32, fy: f32, fz: f32) -> f32 {
    let x00 = c[0] + (c[1] - c[0]) * fx;
    let x10 = c[2] + (c[3] - c[2]) * fx;
    let x01 = c[4] + (c[5] - c[4]) * fx;
    let x11 = c[6] + (c[7] - c[6]) * fx;
    let y0 = x00 + (x10 - x00) * fy;
    let y1 = x01 + (x11 - x01) * fy;
    y0 + (y1 - y0) * fz
}

/// 0 above the ramp, 1 below it.
fn cave_ramp(y: i32) -> f32 {
    if y >= CAVE_RAMP_TOP {
        0.0
    } else if y <= CAVE_RAMP_BOTTOM {
        1.0
    } else {
        (CAVE_RAMP_TOP - y) as f32 / (CAVE_RAMP_TOP - CAVE_RAMP_BOTTOM) as f32
    }
}

/// The four lattice fields at the 8 corners of one 4-block cell
/// (corner `i` is offset `i & 1`, `i >> 1 & 1`, `i >> 2` in X, Y, Z).
struct Corners {
    cheese: [f32; 8],
    spag_a: [f32; 8],
    spag_b: [f32; 8],
    tuff: [f32; 8],
}

impl Corners {
    fn from_fn(mut at: impl FnMut(usize) -> [f32; 4]) -> Self {
        let mut c = Self {
            cheese: [0.0; 8],
            spag_a: [0.0; 8],
            spag_b: [0.0; 8],
            tuff: [0.0; 8],
        };
        for i in 0..8 {
            let [a, b, d, e] = at(i);
            c.cheese[i] = a;
            c.spag_a[i] = b;
            c.spag_b[i] = d;
            c.tuff[i] = e;
        }
        c
    }
}

fn max_of(v: &[f32; 8]) -> f32 {
    v.iter().copied().fold(f32::MIN, f32::max)
}
fn min_of(v: &[f32; 8]) -> f32 {
    v.iter().copied().fold(f32::MAX, f32::min)
}
/// Whether `|v| < t` can hold anywhere between these corners.
fn near_zero_possible(v: &[f32; 8], t: f32) -> bool {
    !(min_of(v) >= t + EPS || max_of(v) <= -t - EPS)
}

#[derive(Clone, Copy)]
enum FeatureKind {
    Ore(usize),
    Pocket,
    Geode,
}

/// A blob placed by a feature cell.
struct Feature {
    kind: FeatureKind,
    center: [f64; 3],
    radius: [f64; 3],
    min: [i32; 3],
    max: [i32; 3],
}

impl Feature {
    fn new(kind: FeatureKind, center: [f64; 3], radius: [f64; 3]) -> Self {
        let mut min = [0; 3];
        let mut max = [0; 3];
        for i in 0..3 {
            min[i] = (center[i] - radius[i]).floor() as i32 - 1;
            max[i] = (center[i] + radius[i]).ceil() as i32;
        }
        Self {
            kind,
            center,
            radius,
            min,
            max,
        }
    }

    /// The block at `(x, y, z)` once this feature has been placed over `cur`.
    fn over(&self, pal: &CubePalette, x: i32, y: i32, z: i32, cur: u32) -> u32 {
        let p = [x, y, z];
        let mut d2 = 0.0;
        for i in 0..3 {
            let d = (f64::from(p[i]) + 0.5 - self.center[i]) / self.radius[i];
            d2 += d * d;
        }
        if d2 > 1.0 {
            return cur;
        }
        let base = cur == pal.deepslate || cur == pal.tuff;
        match self.kind {
            FeatureKind::Ore(i) if base => pal.ores[i].deepslate_ore,
            FeatureKind::Pocket if base => {
                let r = self.radius[0];
                if d2.sqrt() * r < r - 1.3 {
                    pal.lava
                } else {
                    pal.obsidian
                }
            }
            FeatureKind::Geode => {
                let r = self.radius[0];
                let d = d2.sqrt() * r;
                if d < r - 3.0 {
                    pal.air
                } else if d < r - 2.0 {
                    pal.amethyst
                } else {
                    pal.calcite
                }
            }
            _ => cur,
        }
    }
}

/// A floating island: a disc with a flat, slightly domed top and a cone below.
#[derive(Clone, Copy)]
struct Island {
    x: i32,
    y: i32,
    z: i32,
    radius: f64,
}

impl Island {
    fn up(&self) -> f64 {
        2.0 + self.radius * 0.08
    }
    fn down(&self) -> f64 {
        self.radius * 0.6
    }
    /// Block Y range the island can hold, inclusive.
    fn y_range(&self) -> (i32, i32) {
        (
            self.y - self.down() as i32 - 3,
            self.y + self.up() as i32 + 3,
        )
    }
    fn reach(&self) -> i32 {
        (self.radius * 1.2) as i32 + 1
    }
}

/// Generates the deep and the sky of a cubic Overworld.
pub struct CubeGenerator {
    seed: u64,
    palette: CubePalette,
    cheese_a: PerlinNoise,
    cheese_b: PerlinNoise,
    spag_a: PerlinNoise,
    spag_b: PerlinNoise,
    tuff: PerlinNoise,
    lava_region: PerlinNoise,
    lava_height: PerlinNoise,
    island_edge: PerlinNoise,
    gravel: PerlinNoise,
    features: bool,
}

impl CubeGenerator {
    pub fn new(seed: i64, palette: CubePalette) -> Self {
        let positional = XoroshiroRandom::new(seed).fork_positional();
        let noise = |name: &str| PerlinNoise::new(&mut positional.from_hash_of(name));
        Self {
            seed: seed as u64,
            palette,
            cheese_a: noise("horizon:cubic/cheese_a"),
            cheese_b: noise("horizon:cubic/cheese_b"),
            spag_a: noise("horizon:cubic/spaghetti_a"),
            spag_b: noise("horizon:cubic/spaghetti_b"),
            tuff: noise("horizon:cubic/tuff"),
            lava_region: noise("horizon:cubic/lava_region"),
            lava_height: noise("horizon:cubic/lava_height"),
            island_edge: noise("horizon:cubic/island_edge"),
            gravel: noise("horizon:cubic/island_gravel"),
            features: true,
        }
    }

    /// The same generator without ores, pockets and geodes (what `block_at` gives).
    pub fn without_features(mut self) -> Self {
        self.features = false;
        self
    }

    /// Whether this generator makes cube Y `y` (y < -4 or y >= 20).
    pub fn owns(cube_y: i32) -> bool {
        cube_y <= LAST_DEEP_CUBE || cube_y >= FIRST_SKY_CUBE
    }

    /// The lava level of the region around column `(x, z)`: caves below it
    /// hold lava, or `None` where the region is dry.
    pub fn lava_level(&self, x: i32, z: i32) -> Option<i32> {
        let (fx, fz) = (f64::from(x) / 320.0, f64::from(z) / 320.0);
        if self.lava_region.get2(fx, fz) < -0.04 {
            return None;
        }
        let n = f64::from(self.lava_height.get2(fx * 1.7 + 31.0, fz * 1.7 - 17.0)).clamp(-1.0, 1.0);
        Some((-208.0 + 96.0 * n).floor() as i32)
    }

    /// The cube's 4096 block state ids in section index order (`y << 8 | z << 4 | x`).
    /// Returns `Some(id)` when the whole cube is one state; `out` is then
    /// unspecified. A cube this generator does not own comes back as air.
    pub fn generate(&self, cube: SectionPos, out: &mut [u32; 4096]) -> Option<u32> {
        if cube.y <= LAST_DEEP_CUBE {
            self.generate_deep(cube, out)
        } else if cube.y >= FIRST_SKY_CUBE {
            self.generate_sky(cube, out)
        } else {
            Some(self.palette.air)
        }
    }

    /// The block at a world position, the same as `generate` without the
    /// features: ores, lava/obsidian pockets and geodes are skipped here (see
    /// [`Self::block_at_with_features`]). Costs about 40 noise samples.
    pub fn block_at(&self, x: i32, y: i32, z: i32) -> u32 {
        if ((LAST_DEEP_CUBE + 1) * 16..FIRST_SKY_CUBE * 16).contains(&y) {
            return self.palette.air;
        }
        if y < 0 {
            let (ox, oy, oz) = (x & !3, y & !3, z & !3);
            let corners = Corners::from_fn(|i| {
                self.lattice(
                    ox + 4 * (i & 1) as i32,
                    oy + 4 * (i >> 1 & 1) as i32,
                    oz + 4 * (i >> 2) as i32,
                )
            });
            let block = self.deep_block(&corners, [x - ox, y - oy, z - oz], y, true, true);
            if block == CAVE {
                self.cave_fluid(x, y, z)
            } else {
                block
            }
        } else {
            self.sky_block(x, y, z)
        }
    }

    /// Like [`Self::block_at`] but with the features, so it equals what
    /// `generate` gives (slow: scans the feature cells around the block).
    pub fn block_at_with_features(&self, x: i32, y: i32, z: i32) -> u32 {
        let mut block = self.block_at(x, y, z);
        if self.features && y < 0 && Self::owns(y.div_euclid(16)) {
            let mut list = Vec::new();
            self.collect_features([x, y, z], [x, y, z], &mut list);
            for f in &list {
                block = f.over(&self.palette, x, y, z, block);
            }
        }
        block
    }

    fn cave_fluid(&self, x: i32, y: i32, z: i32) -> u32 {
        match self.lava_level(x, z) {
            Some(level) if y < level => self.palette.lava,
            _ => self.palette.air,
        }
    }

    /// `[cheese, spaghetti a, spaghetti b, tuff]` at a lattice point.
    fn lattice(&self, x: i32, y: i32, z: i32) -> [f32; 4] {
        let (x, y, z) = (f64::from(x), f64::from(y), f64::from(z));
        let cheese = self.cheese_a.get3(x / 44.0, y / 32.0, z / 44.0)
            + 0.5 * self.cheese_b.get3(x / 22.0, y / 16.0, z / 22.0);
        [
            cheese,
            self.spag_a.get3(x / 28.0, y / 20.0, z / 28.0),
            self.spag_b.get3(x / 28.0 + 97.0, y / 20.0, z / 28.0 - 53.0),
            self.tuff.get3(x / 18.0, y / 14.0, z / 18.0),
        ]
    }

    /// The deep block at offset `l` (0..4 each) in a cell, or [`CAVE`].
    /// `maybe_cave` / `maybe_tuff` may be false only when the corners prove
    /// there is none.
    fn deep_block(
        &self,
        c: &Corners,
        l: [i32; 3],
        y: i32,
        maybe_cave: bool,
        maybe_tuff: bool,
    ) -> u32 {
        let (fx, fy, fz) = (l[0] as f32 * 0.25, l[1] as f32 * 0.25, l[2] as f32 * 0.25);
        let ramp = cave_ramp(y);
        if maybe_cave && ramp > 0.0 {
            let cave = lerp3(&c.cheese, fx, fy, fz) * ramp > CHEESE_THRESHOLD || {
                let a = lerp3(&c.spag_a, fx, fy, fz).abs();
                let b = lerp3(&c.spag_b, fx, fy, fz).abs();
                a.max(b) < SPAGHETTI_THRESHOLD * ramp
            };
            if cave {
                return CAVE;
            }
        }
        if maybe_tuff && lerp3(&c.tuff, fx, fy, fz) > TUFF_THRESHOLD {
            self.palette.tuff
        } else {
            self.palette.deepslate
        }
    }

    fn generate_deep(&self, cube: SectionPos, out: &mut [u32; 4096]) -> Option<u32> {
        let (ox, oy, oz) = (cube.x * 16, cube.y * 16, cube.z * 16);
        let mut lattice = [[0.0f32; 4]; 125];
        for j in 0..5 {
            for k in 0..5 {
                for i in 0..5 {
                    lattice[j * 25 + k * 5 + i] =
                        self.lattice(ox + 4 * i as i32, oy + 4 * j as i32, oz + 4 * k as i32);
                }
            }
        }
        let mut lava = [UNSET; 256];
        for cj in 0..4 {
            for ck in 0..4 {
                for ci in 0..4 {
                    let corners = Corners::from_fn(|n| {
                        lattice[(cj + (n >> 1 & 1)) * 25 + (ck + (n >> 2)) * 5 + ci + (n & 1)]
                    });
                    let cell_y = oy + 4 * cj as i32;
                    let ramp = cave_ramp(cell_y);
                    let maybe_cave = ramp > 0.0
                        && (max_of(&corners.cheese) * ramp > CHEESE_THRESHOLD - EPS
                            || (near_zero_possible(&corners.spag_a, SPAGHETTI_THRESHOLD)
                                && near_zero_possible(&corners.spag_b, SPAGHETTI_THRESHOLD)));
                    let maybe_tuff = max_of(&corners.tuff) > TUFF_THRESHOLD - EPS;
                    for ly in 0..4 {
                        let y = cell_y + ly;
                        for lz in 0..4 {
                            for lx in 0..4 {
                                let mut block = self.deep_block(
                                    &corners,
                                    [lx, ly, lz],
                                    y,
                                    maybe_cave,
                                    maybe_tuff,
                                );
                                let (x, z) = (4 * ci as i32 + lx, 4 * ck as i32 + lz);
                                if block == CAVE {
                                    let slot = &mut lava[(z * 16 + x) as usize];
                                    if *slot == UNSET {
                                        *slot = self.lava_level(ox + x, oz + z).unwrap_or(i32::MIN);
                                    }
                                    block = if y < *slot {
                                        self.palette.lava
                                    } else {
                                        self.palette.air
                                    };
                                }
                                let yy = 4 * cj as i32 + ly;
                                out[(yy << 8 | z << 4 | x) as usize] = block;
                            }
                        }
                    }
                }
            }
        }
        if self.features {
            self.place_features(cube, out);
        }
        let first = out[0];
        out.iter().all(|&b| b == first).then_some(first)
    }

    fn place_features(&self, cube: SectionPos, out: &mut [u32; 4096]) {
        let lo = [cube.x * 16, cube.y * 16, cube.z * 16];
        let hi = [lo[0] + 15, lo[1] + 15, lo[2] + 15];
        let mut list = Vec::new();
        self.collect_features(lo, hi, &mut list);
        for f in &list {
            let from = |i: usize| f.min[i].max(lo[i]);
            let to = |i: usize| f.max[i].min(hi[i]);
            for y in from(1)..=to(1) {
                for z in from(2)..=to(2) {
                    for x in from(0)..=to(0) {
                        let i = ((y - lo[1]) << 8 | (z - lo[2]) << 4 | (x - lo[0])) as usize;
                        out[i] = f.over(&self.palette, x, y, z, out[i]);
                    }
                }
            }
        }
    }

    /// Every feature whose blob can touch the block box `lo..=hi`, in a fixed
    /// order (kind, then cell by Y, Z, X), whatever the box.
    fn collect_features(&self, lo: [i32; 3], hi: [i32; 3], list: &mut Vec<Feature>) {
        let pal = &self.palette;
        let (x0, x1) = cell_range(lo[0], hi[0], ORE_REACH, ORE_CELL);
        let (y0, y1) = cell_range(lo[1], hi[1], ORE_REACH, ORE_CELL);
        let (z0, z1) = cell_range(lo[2], hi[2], ORE_REACH, ORE_CELL);
        for kind in 0..ORE_NAMES.len().min(pal.ores.len()) {
            for cy in y0..=y1 {
                let mid = cy * ORE_CELL + ORE_CELL / 2;
                if mid > FEATURE_TOP {
                    continue;
                }
                let depth = f64::from(-64 - mid).max(0.0);
                for cz in z0..=z1 {
                    for cx in x0..=x1 {
                        let mut rolls = Rolls(hash(self.seed, 0x0e + kind as u64, cx, cy, cz));
                        let expected = ore_rate(kind, depth);
                        let count =
                            expected.floor() as i32 + i32::from(rolls.unit() < expected.fract());
                        for _ in 0..count {
                            let center = [
                                f64::from(cx * ORE_CELL) + rolls.range(0.0, 16.0),
                                f64::from(cy * ORE_CELL) + rolls.range(0.0, 16.0),
                                f64::from(cz * ORE_CELL) + rolls.range(0.0, 16.0),
                            ];
                            let size = rolls.int(3, 9);
                            let r = 0.5 + 0.5 * f64::from(size).cbrt();
                            let radius = [
                                r * rolls.range(0.8, 1.3),
                                r * rolls.range(0.8, 1.3),
                                r * rolls.range(0.8, 1.3),
                            ];
                            list.push(Feature::new(FeatureKind::Ore(kind), center, radius));
                        }
                    }
                }
            }
        }
        // Lava pockets in an obsidian shell.
        let (x0, x1) = cell_range(lo[0], hi[0], POCKET_REACH, ORE_CELL);
        let (y0, y1) = cell_range(lo[1], hi[1], POCKET_REACH, ORE_CELL);
        let (z0, z1) = cell_range(lo[2], hi[2], POCKET_REACH, ORE_CELL);
        for cy in y0..=y1 {
            if cy * ORE_CELL + ORE_CELL / 2 > -160 {
                continue;
            }
            for cz in z0..=z1 {
                for cx in x0..=x1 {
                    let mut rolls = Rolls(hash(self.seed, 0x40, cx, cy, cz));
                    if rolls.unit() < 0.025 {
                        let center = [
                            f64::from(cx * ORE_CELL) + rolls.range(0.0, 16.0),
                            f64::from(cy * ORE_CELL) + rolls.range(0.0, 16.0),
                            f64::from(cz * ORE_CELL) + rolls.range(0.0, 16.0),
                        ];
                        let r = rolls.range(2.5, 4.0);
                        list.push(Feature::new(FeatureKind::Pocket, center, [r; 3]));
                    }
                }
            }
        }
        // Geodes: calcite shell, amethyst lining, hollow.
        let (x0, x1) = cell_range(lo[0], hi[0], GEODE_REACH, GEODE_CELL);
        let (y0, y1) = cell_range(lo[1], hi[1], GEODE_REACH, GEODE_CELL);
        let (z0, z1) = cell_range(lo[2], hi[2], GEODE_REACH, GEODE_CELL);
        for cy in y0..=y1 {
            if cy * GEODE_CELL + GEODE_CELL / 2 > -224 {
                continue;
            }
            for cz in z0..=z1 {
                for cx in x0..=x1 {
                    let mut rolls = Rolls(hash(self.seed, 0x50, cx, cy, cz));
                    if rolls.unit() < 0.1 {
                        let r = rolls.range(6.0, 8.5);
                        // Keep the whole geode inside its cell.
                        let cell = f64::from(GEODE_CELL);
                        let center = [
                            f64::from(cx * GEODE_CELL) + rolls.range(r, cell - r),
                            f64::from(cy * GEODE_CELL) + rolls.range(r, cell - r),
                            f64::from(cz * GEODE_CELL) + rolls.range(r, cell - r),
                        ];
                        list.push(Feature::new(FeatureKind::Geode, center, [r; 3]));
                    }
                }
            }
        }
    }

    // ---- sky ----

    /// The island of band `band` in island cell `(cx, cz)`, if there is one.
    fn island(&self, cx: i32, cz: i32, band: i32) -> Option<Island> {
        let mut rolls = Rolls(hash(self.seed, 0x60, cx, band, cz));
        if rolls.unit() >= ISLAND_CHANCE {
            return None;
        }
        let radius = rolls.range(10.0, 40.0);
        let margin = radius * 1.2 + 2.0;
        let span = f64::from(ISLAND_CELL) - 2.0 * margin;
        Some(Island {
            x: cx * ISLAND_CELL + (margin + rolls.unit() * span) as i32,
            y: ISLAND_BASE_Y + band * ISLAND_BAND + 20 + rolls.int(0, ISLAND_SPREAD),
            z: cz * ISLAND_CELL + (margin + rolls.unit() * span) as i32,
            radius,
        })
    }

    /// The bands whose islands can reach block Ys `lo..=hi`.
    fn bands(lo: i32, hi: i32) -> std::ops::RangeInclusive<i32> {
        let reach_up = ISLAND_SPREAD + 20 + 30;
        let first = (lo - ISLAND_BASE_Y - reach_up).div_euclid(ISLAND_BAND) + 1;
        let last = (hi - ISLAND_BASE_Y + 30).div_euclid(ISLAND_BAND);
        first.max(0)..=last
    }

    fn island_solid(&self, is: &Island, x: i32, y: i32, z: i32) -> bool {
        let dx = f64::from(x - is.x);
        let dz = f64::from(z - is.z);
        let h = f64::from(y - is.y);
        let r = (dx * dx + dz * dz).sqrt() / is.radius;
        if r >= 1.2 {
            return false;
        }
        let edge = self.island_edge.get3(
            f64::from(x) / 14.0,
            f64::from(y) / 10.0,
            f64::from(z) / 14.0,
        );
        let rn = r + f64::from(edge) * 0.18;
        if rn >= 1.0 {
            return false;
        }
        if h >= 0.0 {
            h < is.up() * (1.0 - rn * rn) + 1.0
        } else {
            -h <= is.down() * (1.0 - rn)
        }
    }

    /// The island block given whether it and the 4 blocks above it are solid.
    fn island_layer(&self, x: i32, y: i32, z: i32, solid: bool, above: [bool; 4]) -> u32 {
        let pal = &self.palette;
        if !solid {
            return pal.air;
        }
        match above.iter().position(|&s| !s) {
            Some(0) => pal.grass_block,
            Some(_) => {
                let patch =
                    self.gravel
                        .get3(f64::from(x) / 5.0, f64::from(y) / 5.0, f64::from(z) / 5.0);
                if patch > 0.35 { pal.gravel } else { pal.dirt }
            }
            None => pal.stone,
        }
    }

    fn sky_block(&self, x: i32, y: i32, z: i32) -> u32 {
        let (cx, cz) = (x.div_euclid(ISLAND_CELL), z.div_euclid(ISLAND_CELL));
        for band in Self::bands(y, y) {
            let Some(is) = self.island(cx, cz, band) else {
                continue;
            };
            let (lo, hi) = is.y_range();
            if y < lo || y > hi {
                continue;
            }
            let solid = |dy: i32| self.island_solid(&is, x, y + dy, z);
            if !solid(0) {
                continue;
            }
            return self.island_layer(x, y, z, true, [solid(1), solid(2), solid(3), solid(4)]);
        }
        self.palette.air
    }

    fn generate_sky(&self, cube: SectionPos, out: &mut [u32; 4096]) -> Option<u32> {
        let (ox, oy, oz) = (cube.x * 16, cube.y * 16, cube.z * 16);
        let (cx, cz) = (ox.div_euclid(ISLAND_CELL), oz.div_euclid(ISLAND_CELL));
        let mut islands = Vec::new();
        for band in Self::bands(oy, oy + 15) {
            let Some(is) = self.island(cx, cz, band) else {
                continue;
            };
            let (lo, hi) = is.y_range();
            let reach = is.reach();
            if oy + 15 < lo
                || oy > hi
                || ox + 15 < is.x - reach
                || ox > is.x + reach
                || oz + 15 < is.z - reach
                || oz > is.z + reach
            {
                continue;
            }
            islands.push(is);
        }
        if islands.is_empty() {
            return Some(self.palette.air);
        }
        out.fill(self.palette.air);
        let mut any = false;
        for is in &islands {
            let mut column = [false; 20];
            for z in 0..16 {
                for x in 0..16 {
                    let (wx, wz) = (ox + x, oz + z);
                    for (k, slot) in column.iter_mut().enumerate() {
                        *slot = self.island_solid(is, wx, oy + k as i32, wz);
                    }
                    for y in 0..16usize {
                        if !column[y] {
                            continue;
                        }
                        let above = [column[y + 1], column[y + 2], column[y + 3], column[y + 4]];
                        out[y << 8 | (z as usize) << 4 | x as usize] =
                            self.island_layer(wx, oy + y as i32, wz, true, above);
                        any = true;
                    }
                }
            }
        }
        if any { None } else { Some(self.palette.air) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const AIR: u32 = 0;
    const STONE: u32 = 1;
    const DEEPSLATE: u32 = 2;
    const TUFF: u32 = 3;
    const LAVA: u32 = 4;
    const GRASS: u32 = 6;
    const DIRT: u32 = 7;
    const GRAVEL: u32 = 8;
    const CALCITE: u32 = 9;
    const AMETHYST: u32 = 10;
    const OBSIDIAN: u32 = 11;
    const ORE_BASE: u32 = 20;

    fn palette() -> CubePalette {
        CubePalette {
            air: AIR,
            stone: STONE,
            deepslate: DEEPSLATE,
            tuff: TUFF,
            lava: LAVA,
            water: 5,
            grass_block: GRASS,
            dirt: DIRT,
            gravel: GRAVEL,
            calcite: CALCITE,
            amethyst: AMETHYST,
            obsidian: OBSIDIAN,
            ores: (0..8)
                .map(|i| OreKind {
                    ore: ORE_BASE + 2 * i,
                    deepslate_ore: ORE_BASE + 2 * i + 1,
                })
                .collect(),
        }
    }

    fn is_deepslate_ore(b: u32) -> bool {
        (ORE_BASE..ORE_BASE + 16).contains(&b) && b % 2 == 1
    }

    fn gen_cube(g: &CubeGenerator, x: i32, y: i32, z: i32) -> Box<[u32; 4096]> {
        let mut out = Box::new([0u32; 4096]);
        if let Some(id) = g.generate(SectionPos::new(x, y, z), &mut out) {
            out.fill(id);
        }
        out
    }

    fn index(x: i32, y: i32, z: i32) -> usize {
        ((y & 15) << 8 | (z & 15) << 4 | (x & 15)) as usize
    }

    #[test]
    fn the_same_seed_gives_the_same_cube_and_another_seed_a_different_one() {
        let a = CubeGenerator::new(42, palette());
        let b = CubeGenerator::new(42, palette());
        let c = CubeGenerator::new(43, palette());
        for (x, y, z) in [(3, -12, -7), (0, -40, 5), (-9, 40, 2), (11, 38, -4)] {
            assert_eq!(gen_cube(&a, x, y, z), gen_cube(&b, x, y, z));
        }
        let differs = (-8..8).any(|x| gen_cube(&a, x, -20, 0) != gen_cube(&c, x, -20, 0));
        assert!(differs);
    }

    #[test]
    fn it_owns_the_deep_and_the_sky_only() {
        assert!(CubeGenerator::owns(-5));
        assert!(!CubeGenerator::owns(-4));
        assert!(!CubeGenerator::owns(19));
        assert!(CubeGenerator::owns(20));
        assert!(CubeGenerator::owns(-100_000));
        assert!(CubeGenerator::owns(500_000));
    }

    #[test]
    fn generate_matches_block_at_without_features_across_cubes() {
        let g = CubeGenerator::new(7, palette()).without_features();
        for (cx, cy, cz) in [
            (0, -5, 0),
            (-3, -6, 4),
            (2, -30, -2),
            (1, -200, 1),
            (5, -2000, 5),
            (0, 37, 0),
            (4, 41, 3),
            (-2, 20, 1),
        ] {
            let cube = gen_cube(&g, cx, cy, cz);
            for y in 0..16 {
                for z in 0..16 {
                    for x in 0..16 {
                        let (wx, wy, wz) = (cx * 16 + x, cy * 16 + y, cz * 16 + z);
                        assert_eq!(
                            cube[index(x, y, z)],
                            g.block_at(wx, wy, wz),
                            "cube {cx} {cy} {cz} block {wx} {wy} {wz}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn island_cubes_match_block_at() {
        // Find cubes that really hold an island, then compare every block.
        let g = CubeGenerator::new(31337, palette());
        let mut out = Box::new([0u32; 4096]);
        let mut checked = 0;
        'search: for cy in 37..60 {
            for cx in -12..12 {
                for cz in -12..12 {
                    if g.generate(SectionPos::new(cx, cy, cz), &mut out).is_some() {
                        continue;
                    }
                    for y in 0..16 {
                        for z in 0..16 {
                            for x in 0..16 {
                                assert_eq!(
                                    out[index(x, y, z)],
                                    g.block_at(cx * 16 + x, cy * 16 + y, cz * 16 + z)
                                );
                            }
                        }
                    }
                    checked += 1;
                    if checked == 6 {
                        break 'search;
                    }
                }
            }
        }
        assert_eq!(checked, 6, "too few island cubes found");
    }

    #[test]
    fn generate_matches_the_per_block_view_with_features_including_across_borders() {
        let g = CubeGenerator::new(99, palette());
        let mut features_seen = 0;
        for cy in [-6, -20, -70] {
            for cx in 0..2 {
                for cz in 0..2 {
                    let cube = gen_cube(&g, cx, cy, cz);
                    for y in 0..16 {
                        for z in 0..16 {
                            for x in 0..16 {
                                let (wx, wy, wz) = (cx * 16 + x, cy * 16 + y, cz * 16 + z);
                                let b = cube[index(x, y, z)];
                                assert_eq!(b, g.block_at_with_features(wx, wy, wz));
                                features_seen += usize::from(b != g.block_at(wx, wy, wz));
                            }
                        }
                    }
                }
            }
        }
        assert!(features_seen > 0, "no feature fell in the sampled cubes");
    }

    #[test]
    fn neighbouring_cubes_agree_with_block_at_on_their_shared_face() {
        let g = CubeGenerator::new(5, palette()).without_features();
        for (a, b) in [
            (SectionPos::new(0, -9, 0), SectionPos::new(1, -9, 0)),
            (SectionPos::new(0, -9, 0), SectionPos::new(0, -9, 1)),
            (SectionPos::new(0, -9, 0), SectionPos::new(0, -8, 0)),
            (SectionPos::new(7, 38, 7), SectionPos::new(7, 39, 7)),
        ] {
            let (ca, cb) = (gen_cube(&g, a.x, a.y, a.z), gen_cube(&g, b.x, b.y, b.z));
            for i in 0..16 {
                for j in 0..16 {
                    // The last layer of `a` and the first of `b` along the axis that differs.
                    let (la, lb, wa, wb) = if a.x != b.x {
                        (
                            index(15, i, j),
                            index(0, i, j),
                            (a.x * 16 + 15, a.y * 16 + i, a.z * 16 + j),
                            (b.x * 16, b.y * 16 + i, b.z * 16 + j),
                        )
                    } else if a.z != b.z {
                        (
                            index(i, j, 15),
                            index(i, j, 0),
                            (a.x * 16 + i, a.y * 16 + j, a.z * 16 + 15),
                            (b.x * 16 + i, b.y * 16 + j, b.z * 16),
                        )
                    } else {
                        (
                            index(i, 15, j),
                            index(i, 0, j),
                            (a.x * 16 + i, a.y * 16 + 15, a.z * 16 + j),
                            (b.x * 16 + i, b.y * 16, b.z * 16 + j),
                        )
                    };
                    assert_eq!(ca[la], g.block_at(wa.0, wa.1, wa.2));
                    assert_eq!(cb[lb], g.block_at(wb.0, wb.1, wb.2));
                }
            }
        }
    }

    /// Counts of (air, lava, deepslate ore, solid, grass) over 64 cubes of one cube Y.
    fn census(g: &CubeGenerator, cy: i32) -> [usize; 5] {
        let mut n = [0; 5];
        for i in 0..64 {
            let cube = gen_cube(g, i % 8 - 4, cy, i / 8 - 4);
            for &b in cube.iter() {
                match b {
                    AIR => n[0] += 1,
                    LAVA => n[1] += 1,
                    GRASS => n[4] += 1,
                    DEEPSLATE | TUFF => n[3] += 1,
                    b if is_deepslate_ore(b) => n[2] += 1,
                    _ => {}
                }
            }
        }
        n
    }

    #[test]
    fn the_deep_has_caves_and_ores_at_every_depth_and_no_grass() {
        let g = CubeGenerator::new(1234, palette());
        let total = 64 * 4096;
        for cy in [-6, -7, -12, -40, -150, -1000, -20_000] {
            let [air, lava, ore, solid, grass] = census(&g, cy);
            let caves = (air + lava) as f64 / total as f64;
            assert!(
                (0.03..0.5).contains(&caves),
                "cube Y {cy}: {caves:.3} of the blocks are cave"
            );
            assert!(solid > total / 2, "cube Y {cy}: only {solid} solid blocks");
            assert_eq!(grass, 0, "cube Y {cy}: grass in the deep");
            if cy >= -7 {
                assert_eq!(lava, 0, "cube Y {cy}: lava above its level");
            }
            if cy <= -12 {
                assert!(ore > 0, "cube Y {cy}: no ore in 64 cubes");
            }
            assert!(
                ore < total / 20,
                "cube Y {cy}: {ore} ore blocks is too many"
            );
        }
    }

    #[test]
    fn lava_only_fills_caves_below_the_regions_level() {
        let g = CubeGenerator::new(77, palette());
        let (mut lava_blocks, mut dry_caves) = (0, 0);
        for cy in [-14, -20, -30] {
            for (cx, cz) in [(0, 0), (30, 4), (-25, 12), (6, -40), (50, 50)] {
                let cube = gen_cube(&g, cx, cy, cz);
                for y in 0..16 {
                    for z in 0..16 {
                        for x in 0..16 {
                            let (wx, wy, wz) = (cx * 16 + x, cy * 16 + y, cz * 16 + z);
                            let level = g.lava_level(wx, wz);
                            if let Some(l) = level {
                                assert!(l < -100, "lava level {l} reaches the upper deep");
                            }
                            match cube[index(x, y, z)] {
                                // Pockets sit in solid rock; cave lava is under the level.
                                LAVA if level.is_some_and(|l| wy < l) => lava_blocks += 1,
                                AIR if level.is_none() => dry_caves += 1,
                                _ => {}
                            }
                        }
                    }
                }
            }
        }
        assert!(lava_blocks > 0, "no lava below its level");
        assert!(dry_caves > 0, "every region is flooded");
    }

    #[test]
    fn caves_do_not_open_the_vanilla_bottom_layers() {
        let g = CubeGenerator::new(3, palette());
        for i in 0..32 {
            let cube = gen_cube(&g, i - 16, -5, i % 7);
            for y in 12..16 {
                for z in 0..16 {
                    for x in 0..16 {
                        let b = cube[index(x, y, z)];
                        assert!(
                            matches!(b, DEEPSLATE | TUFF) || is_deepslate_ore(b),
                            "block {b} at the vanilla bottom"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn geodes_and_lava_pockets_exist_deep_down_and_are_rare() {
        let g = CubeGenerator::new(2024, palette());
        let (mut calcite, mut amethyst, mut obsidian) = (0, 0, 0);
        for cy in [-14, -30, -60] {
            for cx in -6..6 {
                for cz in -6..6 {
                    for &b in gen_cube(&g, cx, cy, cz).iter() {
                        match b {
                            CALCITE => calcite += 1,
                            AMETHYST => amethyst += 1,
                            OBSIDIAN => obsidian += 1,
                            _ => {}
                        }
                    }
                }
            }
        }
        assert!(calcite > 0 && amethyst > 0, "no geode in 432 deep cubes");
        assert!(obsidian > 0, "no lava pocket in 432 deep cubes");
        assert!(calcite < 432 * 4096 / 50, "geodes are not rare");
    }

    #[test]
    fn most_sky_cubes_are_air_and_islands_have_grass_over_dirt() {
        let g = CubeGenerator::new(31337, palette());
        let (mut uniform, mut total) = (0, 0);
        let (mut islands, mut grass_over_dirt, mut stone, mut gravel) = (0, 0, 0, 0);
        let mut out = Box::new([0u32; 4096]);
        for cy in 20..80 {
            for cx in -10..10 {
                for cz in -10..10 {
                    total += 1;
                    match g.generate(SectionPos::new(cx, cy, cz), &mut out) {
                        Some(id) => {
                            assert_eq!(id, AIR);
                            uniform += 1;
                        }
                        None => {
                            islands += 1;
                            for at in 0..4096 {
                                match out[at] {
                                    GRASS if at >= 256 => {
                                        grass_over_dirt +=
                                            usize::from(matches!(out[at - 256], DIRT | GRAVEL));
                                    }
                                    STONE => stone += 1,
                                    GRAVEL => gravel += 1,
                                    _ => {}
                                }
                            }
                        }
                    }
                }
            }
        }
        assert!(uniform * 100 > total * 95, "{uniform} of {total} are air");
        assert!(islands > 0, "no island in the sampled area");
        assert!(grass_over_dirt > 0, "no grass over dirt");
        assert!(stone > 0, "islands have no stone core");
        assert!(gravel > 0, "no gravel");
        // Nothing between the vanilla top and the first band.
        for cy in 20..36 {
            assert_eq!(g.generate(SectionPos::new(0, cy, 0), &mut out), Some(AIR));
        }
    }

    #[test]
    fn islands_are_20_to_80_blocks_wide_and_stay_in_their_cells() {
        let g = CubeGenerator::new(8, palette());
        let (mut seen, mut widest) = (0, 0);
        for cx in -3..3 {
            for cz in -3..3 {
                for band in 0..3 {
                    let Some(is) = g.island(cx, cz, band) else {
                        continue;
                    };
                    seen += 1;
                    let (mut lo, mut hi) = (i32::MAX, i32::MIN);
                    for x in cx * 160 - 8..cx * 160 + 168 {
                        if (-3..=3).any(|dz| g.block_at(x, is.y, is.z + dz) != AIR) {
                            lo = lo.min(x);
                            hi = hi.max(x);
                        }
                    }
                    assert!(
                        lo >= cx * 160 && hi < cx * 160 + 160,
                        "island leaves its cell"
                    );
                    assert!(hi - lo < 90, "island {} wide", hi - lo + 1);
                    widest = widest.max(hi - lo + 1);
                }
            }
        }
        assert!(seen >= 5, "{seen} islands in 36 cells and 3 bands");
        assert!(widest >= 20);
    }

    #[test]
    #[ignore = "timing; run with --release -- --ignored --nocapture"]
    fn timing_ms_per_cube() {
        let g = CubeGenerator::new(1, palette());
        let mut out = Box::new([0u32; 4096]);
        for (name, cy) in [("deep -6", -6), ("deep -40", -40), ("deep -3000", -3000)] {
            let start = std::time::Instant::now();
            let n = 400;
            for i in 0..n {
                g.generate(SectionPos::new(i % 20, cy, i / 20), &mut out);
            }
            let ms = start.elapsed().as_secs_f64() * 1000.0 / f64::from(n);
            println!("{name}: {ms:.3} ms/cube");
        }
        let start = std::time::Instant::now();
        for i in 0..2000 {
            g.generate(SectionPos::new(i % 40, 30 + i / 400, i / 40 % 10), &mut out);
        }
        println!(
            "sky: {:.4} ms/cube",
            start.elapsed().as_secs_f64() * 1000.0 / 2000.0
        );
    }
}
