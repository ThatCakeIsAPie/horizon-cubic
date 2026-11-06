//! Population (CWG `CustomTerrainGenerator.populate`): lakes, ores, the
//! surface-decoration hook and liquid springs for one cube.
//!
//! # Window
//!
//! As in CubicChunks, populating the cube `(cx, cy, cz)` works at an offset of
//! 8 blocks: positions are `cube origin + 8 + nextInt(16)` on each axis and
//! the cube may write into the 2 x 2 x 2 cubes from `(cx, cy, cz)` up. Every
//! write of this module is clamped to that window (a vein that would leave it
//! is cut), so the integrator only has to make those eight cubes available.
//!
//! # Order
//!
//! `PrePopulator` (lakes; dungeons are not generated, see below), standard
//! ores, periodic Gaussian ores, [`CubeWorld::decorate_surface`], liquid
//! springs. The random is `coords_seed_random(world seed, cx, cy, cz)`.
//!
//! # Not generated
//!
//! * Dungeons: Horizon's `monster_room` feature needs a feature world with
//!   chests and spawners; `dungeons` / `dungeonCount` are read but unused.
//! * Vegetation, animals, snow: the integrator's `decorate_surface` hook.
//! * Lake freezing and light-based grass spreading (a lake edge turns dirt
//!   to grass when the block above is air).

use horizon_core::position::SectionPos;
use std::collections::HashSet;

use super::biomes::BiomeBlend;
use super::replacers::resolve_block;
use super::rng::{JRandom, coords_seed_random, cos_f32, hash_pos, sin_f32};
use super::settings::*;
use crate::registry::BlockRegistry;
use crate::worldgen::biome::BiomeId;

/// The blocks a cube's population reads and writes. Positions are world
/// block coordinates.
pub trait CubeWorld {
    fn block(&self, p: [i32; 3]) -> u32;
    fn set_block(&mut self, p: [i32; 3], state: u32);
    /// The biome of a block column. Informational: the generator takes the
    /// biomes from its own [`super::biomes::BiomeColumns`].
    fn biome(&self, _x: i32, _z: i32) -> BiomeId {
        BiomeId(0)
    }
    /// Called after the ores, where CWG places vegetation: 26.3's biome
    /// features (trees, grass, flowers, ...) belong here. `random` is the
    /// cube's population random, already advanced past lakes and ores.
    fn decorate_surface(&mut self, _cube: SectionPos, _random: &mut JRandom) {}
}

/// A compiled `GenerationCondition`.
#[derive(Debug)]
enum Cond {
    Any(Vec<Cond>),
    All(Vec<Cond>),
    NoneOf(Vec<Cond>),
    Rand(f64),
    PosRand(f64),
    PosRandSeed(f64, i64),
    Block { off: [i32; 3], ids: HashSet<u32> },
}

impl Cond {
    fn compile(c: &GenerationCondition, reg: &BlockRegistry) -> Result<Self, String> {
        let list = |v: &[GenerationCondition]| {
            v.iter()
                .map(|c| Self::compile(c, reg))
                .collect::<Result<Vec<_>, _>>()
        };
        Ok(match c {
            GenerationCondition::AnyOf(v) => Self::Any(list(v)?),
            GenerationCondition::AllOf(v) => Self::All(list(v)?),
            GenerationCondition::NoneOf(v) => Self::NoneOf(list(v)?),
            GenerationCondition::Rand(p) => Self::Rand(*p),
            GenerationCondition::PosRand(p) => Self::PosRand(*p),
            GenerationCondition::PosRandSeed { chance, seed } => Self::PosRandSeed(*chance, *seed),
            GenerationCondition::BlockMatch { x, y, z, blocks } => Self::Block {
                off: [*x, *y, *z],
                ids: blocks
                    .iter()
                    .map(|b| resolve_block(reg, b))
                    .collect::<Result<_, _>>()?,
            },
        })
    }

    fn eval(&self, rand: &mut JRandom, world: &dyn CubeWorld, seed: i64, p: [i32; 3]) -> bool {
        match self {
            Self::Any(v) => v.iter().any(|c| c.eval(rand, world, seed, p)),
            Self::All(v) => v.iter().all(|c| c.eval(rand, world, seed, p)),
            Self::NoneOf(v) => !v.iter().any(|c| c.eval(rand, world, seed, p)),
            Self::Rand(c) => f64::from(rand.float()) < *c,
            Self::PosRand(c) => pos_rand(seed, 0, p, *c),
            Self::PosRandSeed(c, s) => pos_rand(seed, *s as i32, p, *c),
            Self::Block { off, ids } => {
                ids.contains(&world.block([p[0] + off[0], p[1] + off[1], p[2] + off[2]]))
            }
        }
    }
}

fn pos_rand(seed: i64, salt: i32, p: [i32; 3], chance: f64) -> bool {
    let r = (hash_pos(seed, salt, p[0], p[1], p[2]) >> 24) & ((1 << 24) - 1);
    (r as f64) < chance * f64::from(1_u32 << 24)
}

struct Ore {
    block: u32,
    deep_block: Option<u32>,
    biomes: Option<HashSet<String>>,
    place: Option<Cond>,
    generate: Option<Cond>,
    size: i32,
    tries: i32,
    probability: f64,
    min_h: f64,
    max_h: f64,
    /// Periodic: mean, std dev, spacing (normalised).
    bell: Option<(f64, f64, f64)>,
}

struct Lake {
    block: u32,
    water: bool,
    biomes: HashSet<String>,
    select: FilterType,
    surface: UserFunction,
    main: UserFunction,
    generate: Option<Cond>,
}

/// The populator of one generator.
pub struct Populator {
    seed: i64,
    base: f64,
    variation: f64,
    ores: Vec<Ore>,
    lakes: Vec<Lake>,
    stone_set: HashSet<u32>,
    deep_set: HashSet<u32>,
    spring_stone: HashSet<u32>,
    air: u32,
    water: u32,
    lava: u32,
    stone: u32,
    dirt: u32,
    grass: u32,
    springs: bool,
}

fn round_i32(v: f64) -> i32 {
    (v + 0.5)
        .floor()
        .clamp(f64::from(i32::MIN), f64::from(i32::MAX)) as i32
}

impl Populator {
    pub fn new(
        seed: i64,
        s: &CustomGeneratorSettings,
        reg: &BlockRegistry,
    ) -> Result<Self, String> {
        let id = |n: &str| resolve_block(reg, &BlockStateDesc::new(n));
        let set = |names: &[&str]| -> Result<HashSet<u32>, String> {
            names.iter().map(|n| id(n)).collect()
        };
        let deep_variant = |d: &BlockStateDesc| -> Option<u32> {
            if !d.properties.is_empty() {
                return None;
            }
            let short = d.name.strip_prefix("minecraft:")?;
            if !short.ends_with("_ore") {
                return None;
            }
            reg.default_state(&format!("minecraft:deepslate_{short}"))
                .map(|i| i.0)
        };
        let biome_set = |b: &Option<Vec<BiomeDesc>>| {
            b.as_ref()
                .map(|v| v.iter().map(|b| b.0.clone()).collect::<HashSet<_>>())
        };
        let mut ores = Vec::new();
        for (i, o) in s.standard_ores.iter().enumerate() {
            let err = |e: String| format!("standardOres[{i}]: {e}");
            ores.push(Ore {
                block: resolve_block(reg, &o.blockstate).map_err(err)?,
                deep_block: deep_variant(&o.blockstate),
                biomes: biome_set(&o.biomes),
                place: o
                    .place_block_when
                    .as_ref()
                    .map(|c| Cond::compile(c, reg))
                    .transpose()
                    .map_err(err)?,
                generate: o
                    .generate_when
                    .as_ref()
                    .map(|c| Cond::compile(c, reg))
                    .transpose()
                    .map_err(err)?,
                size: o.spawn_size,
                tries: o.spawn_tries,
                probability: o.spawn_probability,
                min_h: o.min_height,
                max_h: o.max_height,
                bell: None,
            });
        }
        for (i, o) in s.periodic_gaussian_ores.iter().enumerate() {
            let err = |e: String| format!("periodicGaussianOres[{i}]: {e}");
            ores.push(Ore {
                block: resolve_block(reg, &o.blockstate).map_err(err)?,
                deep_block: deep_variant(&o.blockstate),
                biomes: biome_set(&o.biomes),
                place: o
                    .place_block_when
                    .as_ref()
                    .map(|c| Cond::compile(c, reg))
                    .transpose()
                    .map_err(err)?,
                generate: o
                    .generate_when
                    .as_ref()
                    .map(|c| Cond::compile(c, reg))
                    .transpose()
                    .map_err(err)?,
                size: o.spawn_size,
                tries: o.spawn_tries,
                probability: o.spawn_probability,
                min_h: o.min_height,
                max_h: o.max_height,
                bell: Some((o.height_mean, o.height_std_deviation, o.height_spacing)),
            });
        }
        let mut lakes = Vec::new();
        for (i, l) in s.lakes.iter().enumerate() {
            let block = resolve_block(reg, &BlockStateDesc::new(&l.block))
                .map_err(|e| format!("lakes[{i}]: {e}"))?;
            lakes.push(Lake {
                block,
                water: l.block.trim_start_matches("minecraft:") == "water",
                biomes: l.biomes.iter().map(|b| b.0.clone()).collect(),
                select: l.biome_select,
                surface: l.surface_probability.clone(),
                main: l.main_probability.clone(),
                generate: l
                    .generate_when
                    .as_ref()
                    .map(|c| Cond::compile(c, reg))
                    .transpose()
                    .map_err(|e| format!("lakes[{i}]: {e}"))?,
            });
        }
        Ok(Self {
            seed,
            base: s.expected_base_height,
            variation: s.expected_height_variation,
            ores,
            lakes,
            stone_set: set(&["stone", "granite", "diorite", "andesite"])?,
            deep_set: set(&["deepslate", "tuff"])?,
            spring_stone: set(&[
                "stone",
                "deepslate",
                "granite",
                "diorite",
                "andesite",
                "tuff",
            ])?,
            air: id("air")?,
            water: id("water")?,
            lava: id("lava")?,
            stone: id("stone")?,
            dirt: id("dirt")?,
            grass: id("grass_block")?,
            springs: !s.caves.is_empty() || !s.standard_ores.is_empty() || !s.lakes.is_empty(),
        })
    }

    /// Populates `cube`; `biome_at(x, z)` gives the namespaced biome name of a
    /// block column.
    pub fn populate(&self, cube: SectionPos, world: &mut dyn CubeWorld, blend: &BiomeBlend) {
        let mut rand = coords_seed_random(self.seed, cube.x, cube.y, cube.z);
        let min = [cube.x * 16, cube.y * 16, cube.z * 16];
        let biome = blend.biome_name_at(min[0] + 8, min[2] + 8);
        let win = Window { min };

        self.lakes(&mut rand, world, &win, &biome);
        for ore in &self.ores {
            if ore.biomes.as_ref().is_some_and(|b| !b.contains(&biome)) {
                continue;
            }
            self.ore(ore, &mut rand, world, &win);
        }
        world.decorate_surface(cube, &mut rand);
        if self.springs {
            self.spring(&mut rand, world, &win, true);
            self.spring(&mut rand, world, &win, false);
        }
    }

    // --- ores ------------------------------------------------------------

    fn ore(&self, o: &Ore, rand: &mut JRandom, world: &mut dyn CubeWorld, win: &Window) {
        let min_y = round_i32(o.min_h * self.variation + self.base);
        let max_y = round_i32(o.max_h * self.variation + self.base);
        match o.bell {
            None => {
                if win.min[1] + 8 > max_y || win.min[1] + 8 + 15 < min_y {
                    return;
                }
                for _ in 0..o.tries {
                    if rand.double() > o.probability {
                        continue;
                    }
                    let y = win.min[1] + rand.int(16) + 8;
                    if y > max_y || y < min_y {
                        continue;
                    }
                    let x = win.min[0] + rand.int(16) + 8;
                    let z = win.min[2] + rand.int(16) + 8;
                    self.try_vein(o, rand, world, win, [x, y, z]);
                }
            }
            Some((mean, sd, spacing)) => {
                let spacing = if spacing == 0.0 { 0.5 } else { spacing };
                let i_spacing = round_i32(spacing * self.variation).max(1);
                let i_mean = round_i32(mean * self.variation + self.base);
                let sd = sd * self.variation;
                for _ in 0..o.tries {
                    let y = win.min[1] + rand.int(16) + 8;
                    if y > max_y || y < min_y {
                        continue;
                    }
                    let m = bell_cyclic(f64::from(y), f64::from(i_mean), sd, f64::from(i_spacing));
                    if rand.double() > o.probability * m {
                        continue;
                    }
                    let x = win.min[0] + rand.int(16) + 8;
                    let z = win.min[2] + rand.int(16) + 8;
                    self.try_vein(o, rand, world, win, [x, y, z]);
                }
            }
        }
    }

    fn try_vein(
        &self,
        o: &Ore,
        rand: &mut JRandom,
        world: &mut dyn CubeWorld,
        win: &Window,
        p: [i32; 3],
    ) {
        if let Some(g) = &o.generate
            && !g.eval(rand, world, self.seed, p)
        {
            return;
        }
        self.vein(o, rand, world, win, p);
    }

    /// CWG's `CustomVeinGenerator` (a 1.12 `WorldGenMinable` clone).
    fn vein(
        &self,
        o: &Ore,
        rand: &mut JRandom,
        world: &mut dyn CubeWorld,
        win: &Window,
        p: [i32; 3],
    ) {
        let steps = o.size;
        let angle = rand.float() * std::f32::consts::PI;
        let reach = |t: f32| f64::from(t * steps as f32 / 8.0);
        let (x1, x2) = (
            f64::from(p[0]) + reach(sin_f32(angle)),
            f64::from(p[0]) - reach(sin_f32(angle)),
        );
        let (z1, z2) = (
            f64::from(p[2]) + reach(cos_f32(angle)),
            f64::from(p[2]) - reach(cos_f32(angle)),
        );
        let y1 = f64::from(p[1] + rand.int(3) - 2);
        let y2 = f64::from(p[1] + rand.int(3) - 2);
        for i in 0..steps {
            let progress = i as f32 / steps as f32;
            let t = f64::from(progress);
            let (cx, cy, cz) = (x1 + t * (x2 - x1), y1 + t * (y2 - y1), z1 + t * (z2 - z1));
            let size_factor = rand.double() * f64::from(steps) / 16.0;
            let diameter =
                f64::from(sin_f32(std::f32::consts::PI * progress) + 1.0) * size_factor + 1.0;
            self.ellipsoid(o, rand, world, win, [cx, cy, cz], diameter);
        }
    }

    fn ellipsoid(
        &self,
        o: &Ore,
        rand: &mut JRandom,
        world: &mut dyn CubeWorld,
        win: &Window,
        c: [f64; 3],
        d: f64,
    ) {
        let h = d / 2.0;
        let f = |v: f64| v.floor() as i32;
        for x in f(c[0] - h)..=f(c[0] + h) {
            let dx = (f64::from(x) + 0.5 - c[0]) / h;
            if dx * dx > 1.0 {
                continue;
            }
            for y in f(c[1] - h)..=f(c[1] + h) {
                let dy = (f64::from(y) + 0.5 - c[1]) / h;
                if dx * dx + dy * dy > 1.0 {
                    continue;
                }
                for z in f(c[2] - h)..=f(c[2] + h) {
                    let dz = (f64::from(z) + 0.5 - c[2]) / h;
                    if dx * dx + dy * dy + dz * dz > 1.0 {
                        continue;
                    }
                    let p = [x, y, z];
                    if !win.contains(p) {
                        continue;
                    }
                    let cur = world.block(p);
                    let allowed = match &o.place {
                        Some(cond) => cond.eval(rand, &*world, self.seed, p),
                        None => self.stone_set.contains(&cur) || self.deep_set.contains(&cur),
                    };
                    if allowed {
                        let block = if self.deep_set.contains(&cur) {
                            o.deep_block.unwrap_or(o.block)
                        } else {
                            o.block
                        };
                        world.set_block(p, block);
                    }
                }
            }
        }
    }

    // --- lakes -----------------------------------------------------------

    fn lakes(&self, rand: &mut JRandom, world: &mut dyn CubeWorld, win: &Window, biome: &str) {
        for lake in &self.lakes {
            let listed = lake.biomes.contains(biome);
            if (lake.select == FilterType::Include) != listed {
                continue;
            }
            let pop = [
                win.min[0] + rand.int(16) + 8,
                win.min[1] + rand.int(16) + 8,
                win.min[2] + rand.int(16) + 8,
            ];
            // The surface of the population cube at this column: the first
            // non-air block from the top, unless that is the top block.
            let top = win.min[1] + 8 + 15;
            let mut surface = None;
            for y in (win.min[1] + 8..=top).rev() {
                if world.block([pop[0], y, pop[2]]) != self.air {
                    if y != top {
                        surface = Some(y);
                    }
                    break;
                }
            }
            let (at, prob) = match surface {
                Some(y) => ([pop[0], y, pop[2]], lake.surface.get(f64::from(y))),
                None => (pop, lake.main.get(f64::from(pop[1]))),
            };
            if f64::from(rand.float()) < prob
                && lake
                    .generate
                    .as_ref()
                    .is_none_or(|g| g.eval(rand, &*world, self.seed, at))
            {
                self.lake(lake, rand, world, win, at);
            }
        }
    }

    fn is_liquid(&self, b: u32) -> bool {
        b == self.water || b == self.lava
    }
    fn is_solid(&self, b: u32) -> bool {
        b != self.air && !self.is_liquid(b)
    }

    /// Vanilla 1.12 `WorldGenLakes` at `position` (the surface block).
    fn lake(
        &self,
        lake: &Lake,
        rand: &mut JRandom,
        world: &mut dyn CubeWorld,
        win: &Window,
        position: [i32; 3],
    ) {
        let p = [position[0] - 8, position[1] - 4, position[2] - 8];
        let mut mask = [false; 2048];
        let n = rand.int(4) + 4;
        for _ in 0..n {
            let d0 = rand.double() * 6.0 + 3.0;
            let d1 = rand.double() * 4.0 + 2.0;
            let d2 = rand.double() * 6.0 + 3.0;
            let d3 = rand.double() * (16.0 - d0 - 2.0) + 1.0 + d0 / 2.0;
            let d4 = rand.double() * (8.0 - d1 - 4.0) + 2.0 + d1 / 2.0;
            let d5 = rand.double() * (16.0 - d2 - 2.0) + 1.0 + d2 / 2.0;
            for l in 1..15 {
                for i1 in 1..15 {
                    for j1 in 2..7 {
                        let e6 = (f64::from(l) - d3) / (d0 / 2.0);
                        let e7 = (f64::from(j1) - d4) / (d1 / 2.0);
                        let e8 = (f64::from(i1) - d5) / (d2 / 2.0);
                        if e6 * e6 + e7 * e7 + e8 * e8 < 1.0 {
                            mask[((l * 16 + i1) * 8 + j1) as usize] = true;
                        }
                    }
                }
            }
        }
        let at = |x: i32, z: i32, y: i32| mask[((x * 16 + z) * 8 + y) as usize];
        for x in 0..16 {
            for z in 0..16 {
                for y in 0..8 {
                    let edge = !at(x, z, y)
                        && ((x < 15 && at(x + 1, z, y))
                            || (x > 0 && at(x - 1, z, y))
                            || (z < 15 && at(x, z + 1, y))
                            || (z > 0 && at(x, z - 1, y))
                            || (y < 7 && at(x, z, y + 1))
                            || (y > 0 && at(x, z, y - 1)));
                    if edge {
                        let b = world.block([p[0] + x, p[1] + y, p[2] + z]);
                        if y >= 4 && self.is_liquid(b) {
                            return;
                        }
                        if y < 4 && !self.is_solid(b) && b != lake.block {
                            return;
                        }
                    }
                }
            }
        }
        for x in 0..16 {
            for z in 0..16 {
                for y in 0..8 {
                    if at(x, z, y) {
                        let q = [p[0] + x, p[1] + y, p[2] + z];
                        if win.contains(q) {
                            world.set_block(q, if y >= 4 { self.air } else { lake.block });
                        }
                    }
                }
            }
        }
        if lake.water {
            for x in 0..16 {
                for z in 0..16 {
                    for y in 4..8 {
                        if at(x, z, y) {
                            let below = [p[0] + x, p[1] + y - 1, p[2] + z];
                            let here = [p[0] + x, p[1] + y, p[2] + z];
                            if win.contains(below)
                                && world.block(below) == self.dirt
                                && world.block(here) == self.air
                            {
                                world.set_block(below, self.grass);
                            }
                        }
                    }
                }
            }
        }
        if lake.block == self.lava {
            for x in 0..16 {
                for z in 0..16 {
                    for y in 0..8 {
                        let edge = !at(x, z, y)
                            && ((x < 15 && at(x + 1, z, y))
                                || (x > 0 && at(x - 1, z, y))
                                || (z < 15 && at(x, z + 1, y))
                                || (z > 0 && at(x, z - 1, y))
                                || (y < 7 && at(x, z, y + 1))
                                || (y > 0 && at(x, z, y - 1)));
                        if edge && (y < 4 || rand.int(2) != 0) {
                            let q = [p[0] + x, p[1] + y, p[2] + z];
                            if win.contains(q) && self.is_solid(world.block(q)) {
                                world.set_block(q, self.stone);
                            }
                        }
                    }
                }
            }
        }
    }

    // --- springs -----------------------------------------------------------

    fn spring_probability(&self, y: i32, water: bool) -> f64 {
        let (y_scale, y_offset, value_scale) = if water {
            (-0.0242676003062542, 0.723583275161355, 0.00599930877922822)
        } else {
            (-0.0703727292987445, 1.01588640105311, 0.0127618337650875)
        };
        let vanilla_y = (f64::from(y) - self.base) / self.variation * 64.0 + 64.0;
        ((vanilla_y * y_scale + y_offset).atan() + std::f64::consts::FRAC_PI_2) * value_scale
    }

    fn spring(&self, rand: &mut JRandom, world: &mut dyn CubeWorld, win: &Window, water: bool) {
        for _ in 0..(if water { 50 } else { 20 }) {
            let y_off = rand.int(16) + 8;
            let prob = self.spring_probability(win.min[1] + y_off, water);
            if rand.double() > prob {
                continue;
            }
            let x = win.min[0] + rand.int(16) + 8;
            let z = win.min[2] + rand.int(16) + 8;
            let p = [x, win.min[1] + y_off, z];
            self.liquid_spring(world, p, if water { self.water } else { self.lava });
        }
    }

    /// 1.12 `WorldGenLiquids`: a source in a one-block gap of a stone wall.
    fn liquid_spring(&self, world: &mut dyn CubeWorld, p: [i32; 3], liquid: u32) {
        let stone = |w: &dyn CubeWorld, q: [i32; 3]| self.spring_stone.contains(&w.block(q));
        if !stone(world, [p[0], p[1] + 1, p[2]]) || !stone(world, [p[0], p[1] - 1, p[2]]) {
            return;
        }
        let here = world.block(p);
        if here != self.air && !self.spring_stone.contains(&here) {
            return;
        }
        let sides = [[-1, 0], [1, 0], [0, -1], [0, 1]];
        let (mut solid, mut open) = (0, 0);
        for s in sides {
            let q = [p[0] + s[0], p[1], p[2] + s[1]];
            if stone(world, q) {
                solid += 1;
            } else if world.block(q) == self.air {
                open += 1;
            }
        }
        if solid == 3 && open == 1 {
            world.set_block(p, liquid);
        }
    }
}

/// The 2 x 2 x 2 cubes a cube may write.
struct Window {
    min: [i32; 3],
}

impl Window {
    fn contains(&self, p: [i32; 3]) -> bool {
        (0..3).all(|i| p[i] >= self.min[i] && p[i] < self.min[i] + 32)
    }
}

/// `MathUtil.bellCurveProbabilityCyclic`: a Gaussian of Y that repeats every
/// `spacing` blocks.
pub fn bell_cyclic(y: f64, mean: f64, sd: f64, spacing: f64) -> f64 {
    let f = ((y - spacing / 2.0 - mean) % spacing).abs() - spacing / 2.0;
    (-(f * f) / (2.0 * sd * sd)).exp()
}
