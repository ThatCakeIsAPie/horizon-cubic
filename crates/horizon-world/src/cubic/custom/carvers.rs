//! CWG's cubic caves (`CubicCaveGenerator`) and ravines
//! (`CubicRavineGenerator`).
//!
//! A cube is carved by every walk that starts in an origin cube within
//! `RANGE = 8` cubes of it. Each origin cube's walks depend only on the
//! world seed, the carver and the origin, so the cube's blocks stay a pure
//! function of (seed, settings, cube).
//!
//! Speed: a walk is simulated once per origin cube and kept as the list of
//! spheres it carves (the walk itself does not depend on the cube being
//! generated; CWG's "this cube is unreachable, stop" early exit only skips
//! work). The lists live in a sharded, bounded cache; a cube then only tests
//! the spheres of the origins whose walk bounding box reaches it. Lava on the
//! walls of a sphere's box stops caves, water stops ravines, as in CWG.
//!
//! Differences from CWG: angles use `libm` so results match across
//! platforms; the default replaceable list is wider (see
//! [`CaveConfig::overworld`]); ravines use the replaceable list of the first
//! cave (stone, dirt and grass if there is none); the origin seeds follow
//! vanilla's `MapGenBase` (three longs from the world seed) because
//! CubicChunks' own helper was not read.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use horizon_core::position::SectionPos;

use super::replacers::resolve_block;
use super::rng::{JRandom, cos_f32, sin_f32};
use super::settings::{BlockStateDesc, CaveConfig, CustomGeneratorSettings};
use crate::registry::BlockRegistry;

/// Origin cubes within this many cubes of a cube carve it.
pub const RANGE: i32 = 8;

const SHARDS: usize = 16;
const SHARD_LIMIT: usize = 2048;

/// One carved ellipsoid.
#[derive(Clone, Copy, Debug)]
struct Sphere {
    x: f64,
    y: f64,
    z: f64,
    rh: f64,
    rv: f64,
}

/// Spheres per [`SphereRun`].
const RUN: usize = 16;

/// The bounds of a run of `RUN` consecutive spheres of a walk: a cube that
/// `local_box` would reject for each sphere is rejected for the run at once.
#[derive(Clone, Copy, Debug)]
struct SphereRun {
    min: [f64; 3],
    max: [f64; 3],
    rh: f64,
    rv: f64,
}

impl SphereRun {
    fn of(spheres: &[Sphere]) -> Self {
        let mut r = Self {
            min: [f64::MAX; 3],
            max: [f64::MIN; 3],
            rh: 0.0,
            rv: 0.0,
        };
        for s in spheres {
            for (i, v) in [s.x, s.y, s.z].into_iter().enumerate() {
                r.min[i] = r.min[i].min(v);
                r.max[i] = r.max[i].max(v);
            }
            r.rh = r.rh.max(s.rh);
            r.rv = r.rv.max(s.rv);
        }
        r
    }

    /// Whether `local_box` rejects every sphere of the run for the cube
    /// centred on `c`: the same tests, with the run's extremes (each bound
    /// only moves away from the cube as a sphere's radius shrinks).
    fn misses(&self, c: [f64; 3]) -> bool {
        self.max[0] < c[0] - 16.0 - self.rh * 2.0
            || self.max[1] < c[1] - 16.0 - self.rv * 2.0
            || self.max[2] < c[2] - 16.0 - self.rh * 2.0
            || self.min[0] > c[0] + 16.0 + self.rh * 2.0
            || self.min[1] > c[1] + 16.0 + self.rv * 2.0
            || self.min[2] > c[2] + 16.0 + self.rh * 2.0
    }
}

/// Everything one origin cube carves.
struct Walks {
    spheres: Vec<Sphere>,
    /// Bounds of `spheres` by runs of `RUN`.
    runs: Vec<SphereRun>,
    min: [f64; 3],
    max: [f64; 3],
    max_r: f64,
    /// Ravine only.
    widths: Option<Box<[f32; 256]>>,
    lava_height: i32,
}

impl Walks {
    fn new(spheres: Vec<Sphere>) -> Self {
        let mut min = [f64::MAX; 3];
        let mut max = [f64::MIN; 3];
        let mut max_r: f64 = 0.0;
        for s in &spheres {
            for (i, v) in [s.x, s.y, s.z].into_iter().enumerate() {
                min[i] = min[i].min(v);
                max[i] = max[i].max(v);
            }
            max_r = max_r.max(s.rh).max(s.rv);
        }
        let runs = spheres.chunks(RUN).map(SphereRun::of).collect();
        Self {
            spheres,
            runs,
            min,
            max,
            max_r,
            widths: None,
            lava_height: 0,
        }
    }

    fn may_reach(&self, centre: [f64; 3]) -> bool {
        let m = 16.0 + 2.0 * self.max_r;
        (0..3).all(|i| centre[i] >= self.min[i] - m && centre[i] <= self.max[i] + m)
    }

    /// The spheres that may touch the cube centred on `c`, in walk order:
    /// the runs whose bounds reach it.
    fn near(&self, c: [f64; 3]) -> impl Iterator<Item = &Sphere> {
        self.runs
            .iter()
            .zip(self.spheres.chunks(RUN))
            .filter(move |(run, _)| !run.misses(c))
            .flat_map(|(_, spheres)| spheres)
    }
}

#[derive(Default)]
struct Cache {
    shards: [Mutex<HashMap<[i32; 3], Option<Arc<Walks>>>>; SHARDS],
}

impl Cache {
    fn get(&self, key: [i32; 3], make: impl FnOnce() -> Option<Walks>) -> Option<Arc<Walks>> {
        let h = (key[0].wrapping_mul(73856093)
            ^ key[1].wrapping_mul(19349663)
            ^ key[2].wrapping_mul(83492791)) as u32 as usize;
        let shard = &self.shards[h % SHARDS];
        if let Some(v) = shard.lock().unwrap().get(&key) {
            return v.clone();
        }
        let v = make().map(Arc::new);
        let mut map = shard.lock().unwrap();
        if map.len() >= SHARD_LIMIT {
            map.clear();
        }
        map.insert(key, v.clone());
        v
    }
}

/// Which origin cubes of 8 x 8 x 8 have a walk (the activity draw only
/// depends on the origin), so that a cube does not draw for all of its
/// 17 x 17 x 17 origins every time.
#[derive(Default)]
struct ActiveOrigins {
    shards: [Mutex<HashMap<[i32; 3], Arc<[[i32; 3]]>>>; SHARDS],
}

const REGION: i32 = 8;

impl ActiveOrigins {
    fn region(
        &self,
        key: [i32; 3],
        seeds: &OriginSeeds,
        rarity: i32,
        y_ok: &impl Fn(i32) -> bool,
    ) -> Arc<[[i32; 3]]> {
        let h = (key[0].wrapping_mul(73856093)
            ^ key[1].wrapping_mul(19349663)
            ^ key[2].wrapping_mul(83492791)) as u32 as usize;
        let shard = &self.shards[h % SHARDS];
        if let Some(v) = shard.lock().unwrap().get(&key) {
            return v.clone();
        }
        let mut list = Vec::new();
        for z in 0..REGION {
            for y in 0..REGION {
                for x in 0..REGION {
                    let o = [
                        key[0] * REGION + x,
                        key[1] * REGION + y,
                        key[2] * REGION + z,
                    ];
                    if y_ok(o[1]) && seeds.random(o).int(rarity) == 0 {
                        list.push(o);
                    }
                }
            }
        }
        let v: Arc<[[i32; 3]]> = list.into();
        let mut map = shard.lock().unwrap();
        if map.len() >= SHARD_LIMIT {
            map.clear();
        }
        map.insert(key, v.clone());
        v
    }

    /// The origins within `RANGE` cubes of `cube` whose activity draw hits,
    /// in CWG's loop order (`z`, then `y`, then `x`).
    fn near(
        &self,
        cube: SectionPos,
        seeds: &OriginSeeds,
        rarity: i32,
        y_ok: impl Fn(i32) -> bool,
    ) -> Vec<[i32; 3]> {
        let lo = [cube.x - RANGE, cube.y - RANGE, cube.z - RANGE];
        let hi = [cube.x + RANGE, cube.y + RANGE, cube.z + RANGE];
        let mut out = Vec::new();
        for rz in lo[2].div_euclid(REGION)..=hi[2].div_euclid(REGION) {
            for ry in lo[1].div_euclid(REGION)..=hi[1].div_euclid(REGION) {
                for rx in lo[0].div_euclid(REGION)..=hi[0].div_euclid(REGION) {
                    let list = self.region([rx, ry, rz], seeds, rarity, &y_ok);
                    out.extend(
                        list.iter()
                            .copied()
                            .filter(|o| (0..3).all(|i| o[i] >= lo[i] && o[i] <= hi[i])),
                    );
                }
            }
        }
        out.sort_unstable_by_key(|o| (o[2], o[1], o[0]));
        out
    }
}

/// `MapGenBase`-style origin seeds.
#[derive(Clone, Copy)]
struct OriginSeeds {
    world: i64,
    k: i64,
    l: i64,
    m: i64,
}

impl OriginSeeds {
    fn new(world_seed: i64, salt: i64) -> Self {
        let mut r = JRandom::new(world_seed ^ salt);
        Self {
            world: world_seed ^ salt,
            k: r.long(),
            l: r.long(),
            m: r.long(),
        }
    }
    fn random(&self, o: [i32; 3]) -> JRandom {
        JRandom::new(
            i64::from(o[0]).wrapping_mul(self.k)
                ^ i64::from(o[1]).wrapping_mul(self.m)
                ^ i64::from(o[2]).wrapping_mul(self.l)
                ^ self.world,
        )
    }
}

struct CaveCarver {
    cfg: CaveConfig,
    cave_block: u32,
    allowed: Vec<u32>,
    dirt: u32,
    grass: u32,
    seeds: OriginSeeds,
    cache: Cache,
    active: ActiveOrigins,
}

struct RavineCarver {
    max_cube_y: i32,
    allowed: Vec<u32>,
    air: u32,
    lava: u32,
    water: u32,
    seeds: OriginSeeds,
    cache: Cache,
    active: ActiveOrigins,
}

/// All carvers of a generator.
pub struct Carvers {
    caves: Vec<CaveCarver>,
    ravine: Option<RavineCarver>,
    lava_ids: Vec<u32>,
}

impl Carvers {
    pub fn new(
        world_seed: i64,
        s: &CustomGeneratorSettings,
        registry: &BlockRegistry,
    ) -> Result<Self, String> {
        let id = |n: &str| resolve_block(registry, &BlockStateDesc::new(n));
        let (dirt, grass, lava, water, air) = (
            id("dirt")?,
            id("grass_block")?,
            id("lava")?,
            id("water")?,
            id("air")?,
        );
        let mut caves = Vec::new();
        for (i, cfg) in s.caves.iter().enumerate() {
            let allowed = cfg
                .is_block_replaceable
                .iter()
                .map(|b| resolve_block(registry, b))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| format!("caves[{i}].isBlockReplaceable: {e}"))?;
            caves.push(CaveCarver {
                cave_block: resolve_block(registry, &cfg.cave_block)
                    .map_err(|e| format!("caves[{i}].caveBlock: {e}"))?,
                allowed,
                dirt,
                grass,
                seeds: OriginSeeds::new(world_seed, 0x6361_7665 + i as i64),
                cache: Cache::default(),
                active: ActiveOrigins::default(),
                cfg: cfg.clone(),
            });
        }
        let ravine = if s.ravines {
            let allowed = match s.caves.first() {
                Some(c) if !c.is_block_replaceable.is_empty() => caves[0].allowed.clone(),
                _ => vec![id("stone")?, dirt, grass],
            };
            Some(RavineCarver {
                max_cube_y: (s.expected_base_height / 16.0).floor() as i32,
                allowed,
                air,
                lava,
                water,
                seeds: OriginSeeds::new(world_seed, 0x7261_7669),
                cache: Cache::default(),
                active: ActiveOrigins::default(),
            })
        } else {
            None
        };
        Ok(Self {
            caves,
            ravine,
            lava_ids: vec![lava],
        })
    }

    /// Whether some carver may replace `state`.
    fn replaces(&self, state: u32) -> bool {
        self.caves.iter().any(|c| c.allowed.contains(&state))
            || self.ravine.as_ref().is_some_and(|r| r.allowed.contains(&state))
    }

    pub fn is_empty(&self) -> bool {
        self.caves.is_empty() && self.ravine.is_none()
    }

    /// Carves `out`, the blocks of `cube` (index `y << 8 | z << 4 | x`).
    pub fn carve(&self, cube: SectionPos, out: &mut [u32; 4096]) {
        // A carver only changes blocks of its replaceable list: a cube of one
        // block that no list holds stays as it is (sky, water, bedrock...).
        let first = out[0];
        if !self.replaces(first) && out.iter().all(|&b| b == first) {
            return;
        }
        for c in &self.caves {
            c.carve(cube, out, &self.lava_ids);
        }
        if let Some(r) = &self.ravine {
            r.carve(cube, out);
        }
    }
}

fn centre(cube: SectionPos) -> [f64; 3] {
    [
        f64::from(cube.x * 16 + 8),
        f64::from(cube.y * 16 + 8),
        f64::from(cube.z * 16 + 8),
    ]
}

/// Local bounding box of a sphere in a cube, clamped; `None` if it misses.
fn local_box(cube: SectionPos, s: &Sphere) -> Option<[i32; 6]> {
    let c = centre(cube);
    if s.x < c[0] - 16.0 - s.rh * 2.0
        || s.y < c[1] - 16.0 - s.rv * 2.0
        || s.z < c[2] - 16.0 - s.rh * 2.0
        || s.x > c[0] + 16.0 + s.rh * 2.0
        || s.y > c[1] + 16.0 + s.rv * 2.0
        || s.z > c[2] + 16.0 + s.rh * 2.0
    {
        return None;
    }
    let f = |v: f64| v.floor() as i32;
    let min = [
        f(s.x - s.rh) - cube.x * 16 - 1,
        f(s.y - s.rv) - cube.y * 16 - 1,
        f(s.z - s.rh) - cube.z * 16 - 1,
    ];
    let max = [
        f(s.x + s.rh) - cube.x * 16 + 1,
        f(s.y + s.rv) - cube.y * 16 + 1,
        f(s.z + s.rh) - cube.z * 16 + 1,
    ];
    if (0..3).any(|i| max[i] <= 0 || min[i] >= 16) {
        return None;
    }
    Some([
        min[0].max(0),
        min[1].max(0),
        min[2].max(0),
        max[0].min(16),
        max[1].min(16),
        max[2].min(16),
    ])
}

/// Whether any block on the walls of the box is one of `ids`.
fn walls_have(out: &[u32; 4096], b: [i32; 6], ids: &[u32]) -> bool {
    let at = |x: i32, y: i32, z: i32| out[((y << 8) | (z << 4) | x) as usize];
    let hit = |v: u32| ids.contains(&v);
    for x in b[0]..b[3] {
        for y in b[1]..b[4] {
            if hit(at(x, y, b[2])) || hit(at(x, y, b[5] - 1)) {
                return true;
            }
        }
    }
    for x in b[0]..b[3] {
        for z in b[2]..b[5] {
            if hit(at(x, b[1], z)) || hit(at(x, b[4] - 1, z)) {
                return true;
            }
        }
    }
    for y in b[1]..b[4] {
        for z in b[2]..b[5] {
            if hit(at(b[0], y, z)) || hit(at(b[3] - 1, y, z)) {
                return true;
            }
        }
    }
    false
}

fn norm_dist(cube_min: i32, local: i32, centre: f64, scale: f64) -> f64 {
    (f64::from(cube_min + local) + 0.5 - centre) / scale
}

// --- caves ------------------------------------------------------------------

struct CaveWalker<'a> {
    cfg: &'a CaveConfig,
    spheres: Vec<Sphere>,
}

impl CaveWalker<'_> {
    #[allow(clippy::too_many_arguments)]
    fn node(
        &mut self,
        seed: i64,
        mut x: f64,
        mut y: f64,
        mut z: f64,
        base_size: f32,
        mut horiz_angle: f32,
        mut vert_angle: f32,
        start_walked: i32,
        mut max_walked: i32,
        vert_mod: f64,
    ) {
        let cfg = self.cfg;
        let mut rand = JRandom::new(seed);
        let mut horiz_change = 0.0_f32;
        let mut vert_change = 0.0_f32;
        if max_walked <= 0 {
            let radius = (RANGE - 1) * 16;
            max_walked = radius - rand.int(radius / 4);
        }
        let mut final_step = false;
        let mut walked;
        if start_walked == -1 {
            walked = max_walked / 2;
            final_step = true;
        } else {
            walked = start_walked;
        }
        let split_point = rand.int(max_walked / 2) + max_walked / 4;
        let (flatten, steeper) = (cfg.flatten_factor as f32, cfg.steeper_flatten_factor as f32);
        let (dir_change, prev_v, prev_h) = (
            cfg.direction_change_factor as f32,
            cfg.prev_vert_direction_change_weight as f32,
            cfg.prev_horiz_direction_change_weight as f32,
        );
        while walked < max_walked {
            let fraction = walked as f32 / max_walked as f32;
            let size_h =
                cfg.cave_size_add + f64::from(sin_f32(fraction * std::f32::consts::PI) * base_size);
            let size_v = size_h * vert_mod;
            let xz = cos_f32(vert_angle);
            let yy = sin_f32(vert_angle);
            x += f64::from(cos_f32(horiz_angle) * xz);
            y += f64::from(yy);
            z += f64::from(sin_f32(horiz_angle) * xz);
            if rand.int(cfg.steep_step_rarity) == 0 {
                vert_angle *= steeper;
            } else {
                vert_angle *= flatten;
            }
            vert_angle += vert_change * dir_change;
            horiz_angle += horiz_change * dir_change;
            vert_change *= prev_v;
            horiz_change *= prev_h;
            vert_change += (rand.float() - rand.float())
                * rand.float()
                * cfg.max_add_direction_change_vert as f32;
            horiz_change += (rand.float() - rand.float())
                * rand.float()
                * cfg.max_add_direction_change_horiz as f32;
            if !final_step && walked == split_point && base_size > 1.0 {
                let a = rand.long();
                let size_a = rand.float() * 0.5 + 0.5;
                self.node(
                    a,
                    x,
                    y,
                    z,
                    size_a,
                    horiz_angle - std::f32::consts::FRAC_PI_2,
                    vert_angle / 3.0,
                    walked,
                    max_walked,
                    1.0,
                );
                let b = rand.long();
                let size_b = rand.float() * 0.5 + 0.5;
                self.node(
                    b,
                    x,
                    y,
                    z,
                    size_b,
                    horiz_angle + std::f32::consts::FRAC_PI_2,
                    vert_angle / 3.0,
                    walked,
                    max_walked,
                    1.0,
                );
                return;
            }
            if rand.int(cfg.carve_step_rarity) == 0 && !final_step {
                walked += 1;
                continue;
            }
            self.spheres.push(Sphere {
                x,
                y,
                z,
                rh: size_h,
                rv: size_v,
            });
            if final_step {
                return;
            }
            walked += 1;
        }
    }
}

impl CaveCarver {
    fn walks(&self, o: [i32; 3]) -> Option<Walks> {
        let cfg = &self.cfg;
        let mut rand = self.seeds.random(o);
        if o[1] < cfg.cave_min_height
            || o[1] > cfg.cave_max_height
            || rand.int(cfg.cave_rarity) != 0
        {
            return None;
        }
        let nodes = {
            let a = rand.int(cfg.max_init_nodes + 1) + 1;
            let b = rand.int(a) + 1;
            rand.int(b)
        };
        let mut walker = CaveWalker {
            cfg,
            spheres: Vec::new(),
        };
        for _ in 0..nodes {
            let bx = f64::from(o[0] * 16 + rand.int(16));
            let by = f64::from(o[1] * 16 + rand.int(16));
            let bz = f64::from(o[2] * 16 + rand.int(16));
            let mut branches = 1;
            if rand.int(cfg.large_node_rarity) == 0 {
                let seed = rand.long();
                let size = 1.0 + rand.float() * 6.0;
                walker.node(seed, bx, by, bz, size, 0.0, 0.0, -1, -1, 0.5);
                branches += rand.int(cfg.large_node_max_branches);
            }
            for _ in 0..branches {
                let horiz = rand.float() * std::f32::consts::PI * 2.0;
                let vert = (rand.float() - 0.5) * 2.0 / 8.0;
                let mut size = rand.float() * 2.0 + rand.float();
                if rand.int(cfg.big_cave_rarity) == 0 {
                    size *= rand.float() * rand.float() * 3.0 + 1.0;
                }
                let seed = rand.long();
                walker.node(seed, bx, by, bz, size, horiz, vert, 0, 0, 1.0);
            }
        }
        if walker.spheres.is_empty() {
            return None;
        }
        Some(Walks::new(walker.spheres))
    }

    fn carve(&self, cube: SectionPos, out: &mut [u32; 4096], lava: &[u32]) {
        let c = centre(cube);
        let (min, max) = (self.cfg.cave_min_height, self.cfg.cave_max_height);
        let origins = self
            .active
            .near(cube, &self.seeds, self.cfg.cave_rarity, |y| {
                y >= min && y <= max
            });
        for o in origins {
            let Some(w) = self.cache.get(o, || self.walks(o)) else {
                continue;
            };
            if !w.may_reach(c) {
                continue;
            }
            for s in w.near(c) {
                self.carve_sphere(cube, out, s, lava);
            }
        }
    }

    fn carve_sphere(&self, cube: SectionPos, out: &mut [u32; 4096], s: &Sphere, lava: &[u32]) {
        let Some(b) = local_box(cube, s) else {
            return;
        };
        if walls_have(out, b, lava) {
            return;
        }
        let cfg = &self.cfg;
        let (cmx, cmy, cmz) = (cube.x * 16, cube.y * 16, cube.z * 16);
        for lx in b[0]..b[3] {
            let dx = norm_dist(cmx, lx, s.x, s.rh);
            for lz in b[2]..b[5] {
                let dz = norm_dist(cmz, lz, s.z, s.rh);
                if dx * dx + dz * dz >= 1.0 {
                    continue;
                }
                for ly in b[1]..b[4] {
                    let dy = norm_dist(cmy, ly, s.y, s.rv);
                    let i = ((ly << 8) | (lz << 4) | lx) as usize;
                    let state = out[i];
                    if !self.allowed.contains(&state) {
                        continue;
                    }
                    if dy > cfg.cave_floor_depth && dx * dx + dy * dy + dz * dz < 1.0 {
                        out[i] = self.cave_block;
                    } else if state == self.dirt {
                        let above = norm_dist(cmy, ly + 1, s.y, s.rv);
                        if above > cfg.cave_floor_depth && dx * dx + above * above + dz * dz < 1.0 {
                            out[i] = self.grass;
                        }
                    }
                }
            }
        }
    }
}

// --- ravines ------------------------------------------------------------------

const RAVINE_RARITY: i32 = 50 * 16 / 8;
const LAVA_HEIGHT_OFFSET: f64 = -10.0;
const LAVA_HEIGHT_Y_FACTOR: f64 = -0.1;
const VERT_SIZE_FACTOR: f64 = 3.0;
const RAVINE_SIZE_ADD: f64 = 1.5;
const MIN_RAND: f64 = 0.75;
const MAX_RAND: f64 = 1.0;
const FLATTEN: f32 = 0.7;
const DIR_CHANGE: f32 = 0.05;
const PREV_HORIZ: f32 = 0.5;
const PREV_VERT: f32 = 0.8;
const ADD_HORIZ: f32 = 4.0;
const ADD_VERT: f32 = 2.0;
const CARVE_RARITY: i32 = 4;
const STRETCH_Y: f64 = 6.0;

impl RavineCarver {
    fn walk(&self, o: [i32; 3]) -> Option<Walks> {
        let mut rand = self.seeds.random(o);
        if rand.int(RAVINE_RARITY) != 0 || o[1] > self.max_cube_y {
            return None;
        }
        let sx = f64::from(o[0] * 16 + rand.int(16));
        let sy = f64::from(o[1] * 16 + rand.int(16));
        let sz = f64::from(o[2] * 16 + rand.int(16));
        let mut vert_angle = rand.float() * std::f32::consts::PI * 2.0;
        let mut horiz_angle = (rand.float() - 0.5) * 2.0 / 8.0;
        let base = (rand.float() * 2.0 + rand.float()) * 2.0;
        let lava_height = (sy - (f64::from(base) + RAVINE_SIZE_ADD) * VERT_SIZE_FACTOR
            + LAVA_HEIGHT_OFFSET
            + sy * LAVA_HEIGHT_Y_FACTOR) as i32;
        let seed = rand.long();

        let mut rand = JRandom::new(seed);
        let (mut x, mut y, mut z) = (sx, sy, sz);
        let radius = (RANGE - 1) * 16;
        let max_walked = radius - rand.int(radius / 4);
        // Widths: the node's random generates them first.
        let mut widths = Box::new([0.0_f32; 256]);
        let mut value = 1.0_f32;
        for (i, w) in widths.iter_mut().enumerate() {
            if i == 0 || rand.int(3) == 0 {
                value = 1.0 + rand.float() * rand.float();
            }
            *w = value * value;
        }
        let mut horiz_change = 0.0_f32;
        let mut vert_change = 0.0_f32;
        let mut spheres = Vec::new();
        for walked in 0..max_walked {
            let fraction = walked as f32 / max_walked as f32;
            let mut size_h =
                RAVINE_SIZE_ADD + f64::from(sin_f32(fraction * std::f32::consts::PI) * base);
            let mut size_v = size_h * VERT_SIZE_FACTOR;
            size_h *= f64::from(rand.float()) * (MAX_RAND - MIN_RAND) + MIN_RAND;
            size_v *= f64::from(rand.float()) * (MAX_RAND - MIN_RAND) + MIN_RAND;
            let xz = cos_f32(vert_angle);
            let yy = sin_f32(vert_angle);
            x += f64::from(cos_f32(horiz_angle) * xz);
            y += f64::from(yy);
            z += f64::from(sin_f32(horiz_angle) * xz);
            vert_angle *= FLATTEN;
            vert_angle += vert_change * DIR_CHANGE;
            horiz_angle += horiz_change * DIR_CHANGE;
            vert_change *= PREV_VERT;
            horiz_change *= PREV_HORIZ;
            vert_change += (rand.float() - rand.float()) * rand.float() * ADD_VERT;
            horiz_change += (rand.float() - rand.float()) * rand.float() * ADD_HORIZ;
            if rand.int(CARVE_RARITY) == 0 {
                continue;
            }
            spheres.push(Sphere {
                x,
                y,
                z,
                rh: size_h,
                rv: size_v,
            });
        }
        let mut w = Walks::new(spheres);
        w.widths = Some(widths);
        w.lava_height = lava_height;
        Some(w)
    }

    fn carve(&self, cube: SectionPos, out: &mut [u32; 4096]) {
        let c = centre(cube);
        let max = self.max_cube_y;
        let origins = self
            .active
            .near(cube, &self.seeds, RAVINE_RARITY, |y| y <= max);
        for o in origins {
            let Some(w) = self.cache.get(o, || self.walk(o)) else {
                continue;
            };
            if !w.may_reach(c) {
                continue;
            }
            for s in w.near(c) {
                self.carve_sphere(cube, out, s, &w);
            }
        }
    }

    fn carve_sphere(&self, cube: SectionPos, out: &mut [u32; 4096], s: &Sphere, w: &Walks) {
        let Some(b) = local_box(cube, s) else {
            return;
        };
        if walls_have(out, b, &[self.water]) {
            return;
        }
        let Some(widths) = &w.widths else { return };
        let (cmx, cmy, cmz) = (cube.x * 16, cube.y * 16, cube.z * 16);
        for lx in b[0]..b[3] {
            let dx = norm_dist(cmx, lx, s.x, s.rh);
            for lz in b[2]..b[5] {
                let dz = norm_dist(cmz, lz, s.z, s.rh);
                if dx * dx + dz * dz >= 1.0 {
                    continue;
                }
                for ly in b[1]..b[4] {
                    let dy = norm_dist(cmy, ly, s.y, s.rv);
                    let wd = f64::from(widths[((ly + cmy) & 0xFF) as usize]);
                    if (dx * dx + dz * dz) * wd + dy * dy / STRETCH_Y >= 1.0 {
                        continue;
                    }
                    let i = ((ly << 8) | (lz << 4) | lx) as usize;
                    if !self.allowed.contains(&out[i]) {
                        continue;
                    }
                    out[i] = if cmy + ly < w.lava_height {
                        self.lava
                    } else {
                        self.air
                    };
                }
            }
        }
    }
}
