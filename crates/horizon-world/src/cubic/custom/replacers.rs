//! Block replacers (CWG stage 6): the chain that turns a density value, its
//! gradient and the biome into a block.
//!
//! Every replacer has `minY`/`maxY` (inclusive) and a biome filter and runs
//! in list order on the block the earlier ones chose (start: air).
//!
//! Differences from CWG:
//! * Noise sources are Horizon's: Perlin = `PerlinNoise` octaves, Simplex =
//!   `SimplexNoise` octaves, each octave at twice the frequency and half the
//!   amplitude, summed without normalisation (like libnoise's persistence
//!   0.5), seeded from the world seed and the replacer's `...Seed`.
//!   `GRASS_COLOR_NOISE` / `TEMPERATURE_NOISE` are single-octave Perlin noises
//!   with vanilla's fixed seeds 2345 / 1234.
//! * `mesaSurface` is simplified: vanilla's clay bands (generated from the
//!   world seed with vanilla's algorithm) and the top / filler / water-line
//!   rules, without Bryce pillars.
//! * `randomYGradient` hashes the position with Stafford mixing rather than
//!   Murmur; `filterBlocks` / `blockFilterType` are a Horizon extension.

use horizon_core::noise::{PerlinNoise, SimplexNoise, wrap};
use horizon_core::random::{XoroshiroRandom, mix_stafford13};

use super::biomes::ResolvedBiome;
use super::rng::{JRandom, hash_unit};
use super::settings::*;
use crate::chunk::BlockState;
use crate::registry::BlockRegistry;

/// Resolves a block state description against the registry.
pub fn resolve_block(registry: &BlockRegistry, desc: &BlockStateDesc) -> Result<u32, String> {
    let (name, props) = legacy_block(&desc.name, &desc.properties);
    let Some(default) = registry.default_state(&name) else {
        return Err(format!("unknown block {name} (from \"{}\")", desc.name));
    };
    if props.is_empty() {
        return Ok(default.0);
    }
    let mut state = BlockState {
        name: name.clone(),
        properties: Default::default(),
    };
    for (k, v) in &props {
        state.properties.insert(k.clone(), v.clone());
    }
    registry
        .resolve(&state)
        .map(|id| id.0)
        .map_err(|e| format!("block {name} with properties {props:?}: {e}"))
}

pub fn resolve_opt(
    registry: &BlockRegistry,
    desc: &Option<BlockStateDesc>,
) -> Result<Option<u32>, String> {
    desc.as_ref()
        .map(|d| resolve_block(registry, d))
        .transpose()
}

fn seed_for(world_seed: i64, noise_seed: i32) -> i64 {
    (mix_stafford13(world_seed as u64 ^ 0x5851_f42d_4c95_7f2d)
        ^ mix_stafford13(u64::from(noise_seed as u32).wrapping_add(0x1405_7b7e_f767_814f)))
        as i64
}

enum Octave {
    Perlin(PerlinNoise),
    Simplex(SimplexNoise),
}

/// Sum of octaves with persistence 0.5, times `factor` plus `offset`.
pub struct SurfaceNoise {
    octaves: Vec<Octave>,
    freq: [f64; 3],
    factor: f64,
    offset: f64,
}

impl SurfaceNoise {
    fn new(
        world_seed: i64,
        noise_seed: i32,
        simplex: bool,
        freq: [f64; 3],
        octaves: i32,
        factor: f64,
        offset: f64,
    ) -> Self {
        let mut r = XoroshiroRandom::new(seed_for(world_seed, noise_seed));
        Self {
            octaves: (0..octaves.max(1))
                .map(|_| {
                    if simplex {
                        Octave::Simplex(SimplexNoise::new(&mut r))
                    } else {
                        Octave::Perlin(PerlinNoise::new(&mut r))
                    }
                })
                .collect(),
            freq,
            factor,
            offset,
        }
    }

    pub fn get(&self, x: f64, y: f64, z: f64) -> f64 {
        let mut sum = 0.0;
        let mut scale = 1.0;
        let mut amp = 1.0;
        let flat = self.freq[1] == 0.0;
        for o in &self.octaves {
            let (px, py, pz) = (
                wrap(x * self.freq[0] * scale),
                wrap(y * self.freq[1] * scale),
                wrap(z * self.freq[2] * scale),
            );
            let v = match o {
                Octave::Perlin(p) => p.get3(px, py, pz),
                Octave::Simplex(s) => {
                    if flat {
                        s.get2(px, pz)
                    } else {
                        s.get3(px, py, pz)
                    }
                }
            };
            sum += amp * f64::from(v);
            scale *= 2.0;
            amp *= 0.5;
        }
        sum * self.factor + self.offset
    }
}

/// What a replacer sees for one block.
pub struct BlockCtx<'a> {
    pub x: i32,
    pub y: i32,
    pub z: i32,
    pub dx: f64,
    pub dy: f64,
    pub dz: f64,
    pub density: f64,
    pub biome: &'a ResolvedBiome,
}

struct Surface {
    noise: SurfaceNoise,
    max_depth: f64,
    weight: f64,
    ocean: f64,
    override_top: Option<u32>,
    override_filler: Option<u32>,
    top_thresholds: Vec<(f64, Option<u32>)>,
    filler_thresholds: Vec<(f64, Option<u32>)>,
    gravel: u32,
    sandstone: u32,
    red_sandstone: u32,
    air: u32,
}

impl Surface {
    fn top(&self, biome: &ResolvedBiome, depth: f64) -> u32 {
        for &(t, b) in self.top_thresholds.iter().rev() {
            if depth > t {
                return b.or(self.override_top).unwrap_or(biome.top);
            }
        }
        self.override_top.unwrap_or(biome.top)
    }
    fn filler(&self, biome: &ResolvedBiome, depth: f64) -> u32 {
        for &(t, b) in self.filler_thresholds.iter().rev() {
            if depth > t {
                return b.or(self.override_filler).unwrap_or(biome.filler);
            }
        }
        self.override_filler.unwrap_or(biome.filler)
    }

    fn apply(&self, prev: u32, c: &BlockCtx) -> u32 {
        self.apply_memo(prev, c, None)
    }

    fn apply_memo(&self, prev: u32, c: &BlockCtx, memo: Option<&mut [f64; 256]>) -> u32 {
        if prev == self.air {
            return prev;
        }
        if c.density > self.max_depth * c.dy.abs() || c.density < 0.0 {
            return prev;
        }
        let depth = column_noise(&self.noise, c, memo);
        let adjusted = c.density / c.dy.abs();
        if c.density + c.dy <= 0.0 {
            let y = f64::from(c.y);
            if y < self.ocean - 7.0 - depth {
                return self.gravel;
            }
            if y < self.ocean - 1.0 {
                return if depth > 0.0 {
                    self.filler(c.biome, depth)
                } else {
                    prev
                };
            }
            return if depth > 0.0 {
                self.top(c.biome, depth)
            } else {
                self.air
            };
        }
        let xz = (c.dx * c.dx + c.dz * c.dz).sqrt();
        if c.dy < 0.0 && adjusted < depth + 1.0 - self.weight * xz / c.dy {
            return c.biome.filler;
        }
        if c.biome.filler_is_sand && depth > 1.0 && f64::from(c.y) > self.ocean - depth {
            return if c.biome.filler_is_red_sand {
                self.red_sandstone
            } else {
                self.sandstone
            };
        }
        prev
    }
}

/// `noise` at the block's column (`y` 0), from `memo` (one cube's columns,
/// NaN until computed) when there is one: the value depends on the column
/// only, so a cube computes it once per column instead of once per block.
fn column_noise(noise: &SurfaceNoise, c: &BlockCtx, memo: Option<&mut [f64; 256]>) -> f64 {
    let get = || noise.get(f64::from(c.x), 0.0, f64::from(c.z));
    match memo {
        Some(memo) => {
            let v = &mut memo[((c.z & 15) << 4 | (c.x & 15)) as usize];
            if v.is_nan() {
                *v = get();
            }
            *v
        }
        None => get(),
    }
}

/// One cube's column values of the surface and mesa replacers, by replacer
/// (`Replacers::memo`, `Replacers::block_planned`).
pub struct ColumnMemo(Vec<Option<Box<[f64; 256]>>>);

impl ColumnMemo {
    fn slot(&mut self, i: u16) -> &mut [f64; 256] {
        self.0[i as usize].get_or_insert_with(|| Box::new([f64::NAN; 256]))
    }
}

struct DensityRange {
    in_range: Option<u32>,
    out_of_range: Option<u32>,
    /// A few block states: a slice scan beats hashing them.
    filter: Vec<u32>,
    include: bool,
    min: f64,
    max: f64,
}

struct Mesa {
    depth_noise: SurfaceNoise,
    bands: Vec<u32>,
    band_noise: PerlinNoise,
    band_freq: [f64; 2],
    band_factor: f64,
    band_offset: f64,
    mesa_depth: f64,
    height_offset: f64,
    height_scale: f64,
    water: f64,
    air: u32,
    stone: u32,
    coarse_dirt: u32,
    grass: u32,
    terracotta: u32,
    orange: u32,
}

impl Mesa {
    fn band(&self, x: i32, y: i32, z: i32) -> u32 {
        let n = self.band_noise.get3(
            f64::from(x) * self.band_freq[0],
            0.0,
            f64::from(z) * self.band_freq[1],
        );
        let offset = ((f64::from(n) * self.band_factor + self.band_offset) * 2.0).round() as i32;
        let n = self.bands.len() as i32;
        self.bands[(y + offset + 64).rem_euclid(n) as usize]
    }

    fn apply(&self, prev: u32, c: &BlockCtx) -> u32 {
        self.apply_memo(prev, c, None)
    }

    fn apply_memo(&self, prev: u32, c: &BlockCtx, memo: Option<&mut [f64; 256]>) -> u32 {
        if c.density < 0.0 {
            return prev;
        }
        let depth = column_noise(&self.depth_noise, c, memo);
        let orig = depth - 3.0;
        let coarse = (orig * std::f64::consts::PI).cos() > 0.0;
        let mut top = self.terracotta;
        let mut filler = c.biome.filler;
        if depth < 0.0 {
            top = self.air;
            filler = self.stone;
        }
        let y = f64::from(c.y);
        if y >= self.water - 1.0 {
            let forest_line = (86.0 - 64.0) / 64.0 * self.height_scale + self.height_offset;
            if c.biome.wooded_mesa && y >= forest_line + depth * 2.0 {
                top = if coarse { self.coarse_dirt } else { self.grass };
                filler = self.band(c.x, c.y, c.z);
            } else if y > self.water + 3.0 + depth {
                filler = self.band(c.x, c.y, c.z);
                top = if coarse { self.terracotta } else { filler };
            } else {
                top = self.orange;
                filler = self.orange;
            }
        }
        if c.density + c.dy <= 0.0 {
            return top;
        }
        if c.density / c.dy.abs() < self.mesa_depth {
            return filler;
        }
        prev
    }
}

struct Decoration {
    ground: Option<u32>,
    feature: Option<u32>,
    threshold: f64,
    noise: DecoNoise,
    ground_range: (f64, f64),
    feature_range: (f64, f64),
    min_y: i32,
    max_y: i32,
}

enum DecoNoise {
    Surface(SurfaceNoise),
    Vanilla {
        perlin: PerlinNoise,
        fx: f64,
        fz: f64,
        factor: f64,
        offset: f64,
    },
}

impl DecoNoise {
    fn get(&self, x: i32, y: i32, z: i32) -> f64 {
        match self {
            Self::Surface(s) => s.get(f64::from(x), f64::from(y), f64::from(z)),
            Self::Vanilla {
                perlin,
                fx,
                fz,
                factor,
                offset,
            } => {
                f64::from(perlin.get3(f64::from(x) * fx, 0.0, f64::from(z) * fz)) * factor + offset
            }
        }
    }
}

struct Gradient {
    block: u32,
    function: UserFunction,
    seed: i64,
    /// A few block states: a slice scan beats hashing them.
    filter: Vec<u32>,
    include: bool,
}

enum Kind {
    Noop,
    Density(DensityRange),
    Surface(Box<Surface>),
    Mesa(Box<Mesa>),
    Decoration(Box<Decoration>),
    Gradient(Gradient),
}

struct Compiled {
    min_y: i32,
    max_y: i32,
    kind: Kind,
}

/// The compiled replacer chain.
pub struct Replacers {
    list: Vec<Compiled>,
    air: u32,
}

/// Vanilla's mesa band colours.
fn mesa_bands(seed: i64, ids: &[u32; 7]) -> Vec<u32> {
    let [terracotta, orange, yellow, brown, red, white, light_gray] = *ids;
    let mut bands = vec![terracotta; 64];
    let mut r = JRandom::new(seed);
    let mut i = 0;
    while i < 64 {
        i += r.int(5) + 1;
        if i < 64 {
            bands[i as usize] = orange;
        }
        i += 1;
    }
    let stripe = |r: &mut JRandom, bands: &mut Vec<u32>, block: u32, extra: i32| {
        let n = r.int(4) + 2;
        for _ in 0..n {
            let k = r.int(3) + extra;
            let l = r.int(64);
            for m in 0..k {
                if l + m < 64 {
                    bands[(l + m) as usize] = block;
                }
            }
        }
    };
    stripe(&mut r, &mut bands, yellow, 1);
    stripe(&mut r, &mut bands, brown, 2);
    stripe(&mut r, &mut bands, red, 1);
    let j = r.int(3) + 3;
    let mut k = 0;
    for _ in 0..j {
        k += r.int(16) + 4;
        if k < 64 {
            bands[k as usize] = white;
            if k - 1 > 0 && r.boolean() {
                bands[(k - 1) as usize] = light_gray;
            }
            if k + 1 < 64 && r.boolean() {
                bands[(k + 1) as usize] = light_gray;
            }
        }
    }
    bands
}

impl Replacers {
    pub fn new(
        world_seed: i64,
        configs: &[ReplacerConfig],
        registry: &BlockRegistry,
    ) -> Result<Self, String> {
        let id = |n: &str| resolve_block(registry, &BlockStateDesc::new(n));
        let air = id("air")?;
        let mut list = Vec::new();
        for (index, cfg) in configs.iter().enumerate() {
            let err = |e: String| format!("replacers[{index}]: {e}");
            let kind = match cfg {
                ReplacerConfig::DensityRange(c) => {
                    let in_range = resolve_opt(registry, &c.block_in_range).map_err(err)?;
                    let out_of_range = resolve_opt(registry, &c.block_out_of_range).map_err(err)?;
                    let mut filter = Vec::new();
                    for b in &c.filter_blocks {
                        let id = resolve_block(registry, b).map_err(err)?;
                        if !filter.contains(&id) {
                            filter.push(id);
                        }
                    }
                    let include = c.block_filter_type == FilterType::Include;
                    let never_in = c.min_density.is_nan()
                        || c.max_density.is_nan()
                        || c.min_density >= c.max_density;
                    if (in_range.is_none() && out_of_range.is_none())
                        || (include && filter.is_empty())
                        || (out_of_range.is_none() && never_in)
                    {
                        Kind::Noop
                    } else {
                        Kind::Density(DensityRange {
                            in_range,
                            out_of_range,
                            filter,
                            include,
                            min: c.min_density,
                            max: c.max_density,
                        })
                    }
                }
                ReplacerConfig::MainSurface(c) => Kind::Surface(Box::new(
                    surface(
                        world_seed,
                        registry,
                        air,
                        c.surface_depth_noise_type,
                        c.surface_depth_noise_seed,
                        [
                            c.surface_depth_noise_frequency_x,
                            c.surface_depth_noise_frequency_y,
                            c.surface_depth_noise_frequency_z,
                        ],
                        c.surface_depth_noise_octaves,
                        c.surface_depth_noise_factor,
                        c.surface_depth_noise_offset,
                        c.max_surface_depth,
                        c.horizontal_gradient_depth_decrease_weight,
                        c.ocean_level,
                        &c.override_top,
                        &c.override_filler,
                        &[],
                        &[],
                    )
                    .map_err(err)?,
                )),
                ReplacerConfig::DepthBasedSurface(c) => Kind::Surface(Box::new(
                    surface(
                        world_seed,
                        registry,
                        air,
                        c.surface_depth_noise_type,
                        c.surface_depth_noise_seed,
                        [
                            c.surface_depth_noise_frequency_x,
                            c.surface_depth_noise_frequency_y,
                            c.surface_depth_noise_frequency_z,
                        ],
                        c.surface_depth_noise_octaves,
                        c.surface_depth_noise_factor,
                        c.surface_depth_noise_offset,
                        c.max_surface_depth,
                        c.horizontal_gradient_depth_decrease_weight,
                        c.ocean_level,
                        &c.override_top,
                        &c.override_filler,
                        &c.top_thresholds,
                        &c.filler_thresholds,
                    )
                    .map_err(err)?,
                )),
                ReplacerConfig::MesaSurface(c) => {
                    let ids = [
                        "terracotta",
                        "orange_terracotta",
                        "yellow_terracotta",
                        "brown_terracotta",
                        "red_terracotta",
                        "white_terracotta",
                        "light_gray_terracotta",
                    ];
                    let mut resolved = [0_u32; 7];
                    for (slot, n) in resolved.iter_mut().zip(ids) {
                        *slot = id(n)?;
                    }
                    let bands = match &c.clay_bands_override {
                        Some(list) if !list.is_empty() => list
                            .iter()
                            .map(|b| resolve_block(registry, b))
                            .collect::<Result<Vec<_>, _>>()
                            .map_err(err)?,
                        _ => mesa_bands(world_seed, &resolved),
                    };
                    let mut r = XoroshiroRandom::new(
                        seed_for(world_seed, c.custom_clay_bands_noise_seed) ^ 0x6d65_7361,
                    );
                    Kind::Mesa(Box::new(Mesa {
                        depth_noise: SurfaceNoise::new(
                            world_seed,
                            c.surface_depth_noise_seed,
                            c.surface_depth_noise_type == NoiseType::SimplexSpongeNoise,
                            [
                                c.surface_depth_noise_frequency_x,
                                c.surface_depth_noise_frequency_y,
                                c.surface_depth_noise_frequency_z,
                            ],
                            c.surface_depth_noise_octaves,
                            c.surface_depth_noise_factor,
                            c.surface_depth_noise_offset,
                        ),
                        bands,
                        band_noise: PerlinNoise::new(&mut r),
                        band_freq: [
                            c.clay_bands_noise_frequency_x,
                            c.clay_bands_noise_frequency_z,
                        ],
                        band_factor: c.clay_bands_noise_factor,
                        band_offset: c.clay_bands_noise_offset,
                        mesa_depth: c.mesa_depth,
                        height_offset: c.height_offset,
                        height_scale: c.height_scale,
                        water: c.water_height,
                        air,
                        stone: id("stone")?,
                        coarse_dirt: id("coarse_dirt")?,
                        grass: id("grass_block")?,
                        terracotta: resolved[0],
                        orange: resolved[1],
                    }))
                }
                ReplacerConfig::NoiseBasedSurfaceDecoration(c) => {
                    let noise = match c.noise_source {
                        DecorationNoiseSource::FlowNoisePerlin
                        | DecorationNoiseSource::SpongeNoiseSimplex => {
                            DecoNoise::Surface(SurfaceNoise::new(
                                world_seed,
                                c.custom_noise_seed,
                                c.noise_source == DecorationNoiseSource::SpongeNoiseSimplex,
                                [c.noise_freq_x, c.noise_freq_y, c.noise_freq_z],
                                c.custom_noise_octaves,
                                c.noise_factor,
                                c.noise_offset,
                            ))
                        }
                        source => {
                            let fixed = if source == DecorationNoiseSource::GrassColorNoise {
                                2345
                            } else {
                                1234
                            };
                            DecoNoise::Vanilla {
                                perlin: PerlinNoise::new(&mut XoroshiroRandom::new(fixed)),
                                fx: c.noise_freq_x,
                                fz: c.noise_freq_z,
                                factor: c.noise_factor,
                                offset: c.noise_offset,
                            }
                        }
                    };
                    Kind::Decoration(Box::new(Decoration {
                        ground: resolve_opt(registry, &c.ground_block).map_err(err)?,
                        feature: resolve_opt(registry, &c.feature_block).map_err(err)?,
                        threshold: c.surface_density_threshold,
                        noise,
                        ground_range: (c.ground_min_noise, c.ground_max_noise),
                        feature_range: (c.feature_min_noise, c.feature_max_noise),
                        min_y: c.min_y,
                        max_y: c.max_y,
                    }))
                }
                ReplacerConfig::RandomYGradient(c) => {
                    match resolve_opt(registry, &c.block_to_place).map_err(err)? {
                        None => Kind::Noop,
                        Some(block) => {
                            let mut filter = Vec::new();
                            for b in &c.filter_blocks {
                                let id = resolve_block(registry, b).map_err(err)?;
                                if !filter.contains(&id) {
                                    filter.push(id);
                                }
                            }
                            Kind::Gradient(Gradient {
                                block,
                                function: c.probability_function.clone(),
                                seed: seed_for(world_seed, c.seed),
                                filter,
                                include: c.block_filter_type == FilterType::Include,
                            })
                        }
                    }
                }
            };
            list.push(Compiled {
                min_y: cfg.min_y(),
                max_y: cfg.max_y(),
                kind,
            });
        }
        Ok(Self { list, air })
    }

    pub fn len(&self) -> usize {
        self.list.len()
    }

    pub fn is_empty(&self) -> bool {
        self.list.is_empty()
    }

    /// Runs the chain for one block.
    /// The replacers that can change a block of `biome` with Y in
    /// `y0..=y1`, in chain order: those the biome takes, minus no-ops, those
    /// whose Y range misses (a decoration's own), and gradients whose
    /// probability is at most 0 at every one of those Y (`apply` returns the
    /// block it got there).
    pub fn plan(&self, y0: i32, y1: i32, biome: &ResolvedBiome) -> Vec<u16> {
        let mut plan = Vec::new();
        for (i, r) in self.list.iter().enumerate() {
            if !biome.replacer_applies(i) {
                continue;
            }
            let live = match &r.kind {
                Kind::Noop => false,
                Kind::Decoration(d) => y1 >= d.min_y && y0 <= d.max_y,
                _ if y1 < r.min_y || y0 > r.max_y => false,
                Kind::Gradient(g) => (y0.max(r.min_y)..=y1.min(r.max_y))
                    .any(|y| !(g.function.get(f64::from(y)) <= 0.0)),
                _ => true,
            };
            if live {
                plan.push(i as u16);
            }
        }
        plan
    }

    /// `block` with the replacers of a `plan` made for the block's biome and
    /// a Y range holding it.
    /// An empty `ColumnMemo` for one cube's `block_planned` calls.
    pub fn memo(&self) -> ColumnMemo {
        ColumnMemo(self.list.iter().map(|_| None).collect())
    }

    pub fn block_planned(&self, plan: &[u16], c: &BlockCtx, memo: &mut ColumnMemo) -> u32 {
        let mut block = self.air;
        for &i in plan {
            let r = &self.list[i as usize];
            block = match &r.kind {
                Kind::Noop => continue,
                Kind::Decoration(d) => d.apply(block, c),
                _ if c.y < r.min_y || c.y > r.max_y => continue,
                Kind::Density(d) => d.apply(block, c.density),
                Kind::Surface(s) => s.apply_memo(block, c, Some(memo.slot(i))),
                Kind::Mesa(m) => m.apply_memo(block, c, Some(memo.slot(i))),
                Kind::Gradient(g) => g.apply(block, c),
            };
        }
        block
    }

    pub fn block(&self, c: &BlockCtx) -> u32 {
        let mut block = self.air;
        for (i, r) in self.list.iter().enumerate() {
            if !c.biome.replacer_applies(i) {
                continue;
            }
            block = match &r.kind {
                Kind::Noop => continue,
                Kind::Decoration(d) => d.apply(block, c),
                _ if c.y < r.min_y || c.y > r.max_y => continue,
                Kind::Density(d) => d.apply(block, c.density),
                Kind::Surface(s) => s.apply(block, c),
                Kind::Mesa(m) => m.apply(block, c),
                Kind::Gradient(g) => g.apply(block, c),
            };
        }
        block
    }
}

#[allow(clippy::too_many_arguments)]
fn surface(
    world_seed: i64,
    registry: &BlockRegistry,
    air: u32,
    noise_type: NoiseType,
    noise_seed: i32,
    freq: [f64; 3],
    octaves: i32,
    factor: f64,
    offset: f64,
    max_depth: f64,
    weight: f64,
    ocean: f64,
    top: &Option<BlockStateDesc>,
    filler: &Option<BlockStateDesc>,
    top_thresholds: &[DepthEntry],
    filler_thresholds: &[DepthEntry],
) -> Result<Surface, String> {
    let id = |n: &str| resolve_block(registry, &BlockStateDesc::new(n));
    let thresholds = |list: &[DepthEntry]| -> Result<Vec<(f64, Option<u32>)>, String> {
        let mut v = list
            .iter()
            .map(|e| Ok((e.y, Some(resolve_block(registry, &e.b)?))))
            .collect::<Result<Vec<_>, String>>()?;
        v.sort_by(|a, b| a.0.total_cmp(&b.0));
        Ok(v)
    };
    Ok(Surface {
        noise: SurfaceNoise::new(
            world_seed,
            noise_seed,
            noise_type == NoiseType::SimplexSpongeNoise,
            freq,
            octaves,
            factor,
            offset,
        ),
        max_depth,
        weight,
        ocean,
        override_top: resolve_opt(registry, top)?,
        override_filler: resolve_opt(registry, filler)?,
        top_thresholds: thresholds(top_thresholds)?,
        filler_thresholds: thresholds(filler_thresholds)?,
        gravel: id("gravel")?,
        sandstone: id("sandstone")?,
        red_sandstone: id("red_sandstone")?,
        air,
    })
}

impl DensityRange {
    fn apply(&self, prev: u32, density: f64) -> u32 {
        if self.filter.contains(&prev) != self.include {
            return prev;
        }
        if density > self.min && density < self.max {
            self.in_range.unwrap_or(prev)
        } else {
            self.out_of_range.unwrap_or(prev)
        }
    }
}

impl Decoration {
    fn apply(&self, prev: u32, c: &BlockCtx) -> u32 {
        if let Some(ground) = self.ground
            && c.y >= self.min_y
            && c.y < self.max_y
            && c.density > self.threshold
            && c.density + c.dy <= self.threshold
        {
            let v = self.noise.get(c.x, c.y, c.z);
            return if v >= self.ground_range.0 && v <= self.ground_range.1 {
                ground
            } else {
                prev
            };
        }
        if let Some(feature) = self.feature
            && c.y > self.min_y
            && c.y <= self.max_y
            && c.density <= self.threshold
            && c.density - c.dy > self.threshold
        {
            let v = self.noise.get(c.x, c.y, c.z);
            return if v >= self.feature_range.0 && v < self.feature_range.1 {
                feature
            } else {
                prev
            };
        }
        prev
    }
}

impl Gradient {
    fn apply(&self, prev: u32, c: &BlockCtx) -> u32 {
        if self.filter.contains(&prev) != self.include {
            return prev;
        }
        let p = self.function.get(f64::from(c.y));
        if p >= 1.0 {
            return self.block;
        }
        if p <= 0.0 {
            return prev;
        }
        if hash_unit(self.seed, 0, c.x, c.y, c.z) <= p {
            self.block
        } else {
            prev
        }
    }
}

// ---------------------------------------------------------------------------
// Conservative analysis for the bounds (`bounds.rs`)
// ---------------------------------------------------------------------------

/// What is known about a box of blocks, for [`Replacers::possible_blocks`].
pub struct BoxFacts<'a> {
    /// Bounds of the (interpolated) density of every block in the box.
    pub dmin: f64,
    pub dmax: f64,
    /// An upper bound of `|dy|`, the vertical gradient the replacers see.
    pub dy: f64,
    /// The block Y range of the box (inclusive).
    pub y0: i32,
    pub y1: i32,
    /// The biomes of the box's columns; `None` when unknown (any biome).
    pub biomes: Option<&'a [std::rc::Rc<ResolvedBiome>]>,
}

fn add(out: &mut Vec<u32>, b: u32) {
    if !out.contains(&b) {
        out.push(b);
    }
}

impl Surface {
    /// Every block `apply` can return that is not `prev`.
    fn outputs(&self, f: &BoxFacts, out: &mut Vec<u32>) -> bool {
        for b in [self.gravel, self.sandstone, self.red_sandstone, self.air] {
            add(out, b);
        }
        for b in [self.override_top, self.override_filler]
            .into_iter()
            .flatten()
        {
            add(out, b);
        }
        for (_, b) in self.top_thresholds.iter().chain(&self.filler_thresholds) {
            if let Some(b) = b {
                add(out, *b);
            }
        }
        let Some(biomes) = f.biomes else {
            return false;
        };
        for b in biomes {
            add(out, b.top);
            add(out, b.filler);
        }
        true
    }
}

impl Replacers {
    /// Every block this chain can produce for the blocks of a box, starting
    /// from air; `None` when it cannot tell (any block). A one-element result
    /// proves the box uniform. The analysis is a superset: it follows every
    /// branch the facts allow.
    ///
    /// Per kind (`prev` is each block that may reach the replacer):
    /// * `densityRange`: filtered-out blocks stay; the in-range block when the
    ///   whole density range is strictly inside `(min, max)`, the
    ///   out-of-range block when it is wholly outside, both otherwise.
    /// * `mainSurface`, `depthBasedSurface`: only act where
    ///   `0 <= density <= maxSurfaceDepth * |dy|` (with the gradient bound);
    ///   there every block they can place (and the biomes' top / filler) is
    ///   possible.
    /// * `mesaSurface`: unknown (any block) wherever it can act.
    /// * `noiseBasedSurfaceDecoration`: acts only on its Y band and within
    ///   `|dy|` of its threshold.
    /// * `randomYGradient`: the probability function is evaluated at the ends
    ///   of the box's Y range and at its breakpoints: all `>= 1` places the
    ///   block, all `<= 0` leaves it, anything between makes both possible.
    ///
    /// A replacer whose Y range only partly covers the box, or whose biome
    /// filter matches only some of the box's biomes, keeps the old blocks as
    /// well.
    pub fn possible_blocks(&self, f: &BoxFacts) -> Option<Vec<u32>> {
        let mut cur = vec![self.air];
        for (i, r) in self.list.iter().enumerate() {
            let (may_biome, all_biome) = match f.biomes {
                Some(b) => (
                    b.iter().any(|b| b.replacer_applies(i)),
                    b.iter().all(|b| b.replacer_applies(i)),
                ),
                None => (true, false),
            };
            if !may_biome || matches!(r.kind, Kind::Noop) {
                continue;
            }
            // Decorations check their own Y band.
            let (may_y, all_y) = if matches!(r.kind, Kind::Decoration(_)) {
                (true, true)
            } else {
                (
                    f.y1 >= r.min_y && f.y0 <= r.max_y,
                    f.y0 >= r.min_y && f.y1 <= r.max_y,
                )
            };
            if !may_y {
                continue;
            }
            let mut next = Vec::new();
            for &p in &cur {
                if !self.apply_set(
                    &r.kind,
                    p,
                    f,
                    (f.y0.max(r.min_y), f.y1.min(r.max_y)),
                    &mut next,
                ) {
                    return None;
                }
            }
            if !(all_biome && all_y) {
                for &p in &cur {
                    add(&mut next, p);
                }
            }
            if next.len() > 24 {
                return None;
            }
            cur = next;
        }
        Some(cur)
    }

    /// Adds every block `kind` can turn `prev` into to `out`; false when
    /// that is unknown.
    fn apply_set(
        &self,
        kind: &Kind,
        prev: u32,
        f: &BoxFacts,
        ey: (i32, i32),
        out: &mut Vec<u32>,
    ) -> bool {
        match kind {
            Kind::Noop => add(out, prev),
            Kind::Density(d) => {
                if d.filter.contains(&prev) != d.include {
                    add(out, prev);
                    return true;
                }
                let inside = f.dmin > d.min && f.dmax < d.max;
                let outside = f.dmax <= d.min || f.dmin >= d.max;
                if !outside {
                    add(out, d.in_range.unwrap_or(prev));
                }
                if !inside {
                    add(out, d.out_of_range.unwrap_or(prev));
                }
            }
            Kind::Surface(s) => {
                add(out, prev);
                // `density > max_depth * |dy| || density < 0` leaves `prev`.
                // (Written as `quiet` so that a NaN counts as "may act".)
                let quiet = f.dmax < 0.0 || f.dmin > s.max_depth.max(0.0) * f.dy;
                if prev != s.air && !quiet && !s.outputs(f, out) {
                    return false;
                }
            }
            Kind::Mesa(m) => {
                add(out, prev);
                let quiet = f.dmax < 0.0 || f.dmin > f.dy.max(m.mesa_depth.max(0.0) * f.dy);
                if !quiet {
                    return false;
                }
            }
            Kind::Decoration(d) => {
                add(out, prev);
                if let Some(ground) = d.ground
                    && f.y1 >= d.min_y
                    && f.y0 < d.max_y
                    && f.dmax > d.threshold
                    && f.dmin - f.dy <= d.threshold
                {
                    add(out, ground);
                }
                if let Some(feature) = d.feature
                    && f.y1 > d.min_y
                    && f.y0 <= d.max_y
                    && f.dmin <= d.threshold
                    && f.dmax + f.dy > d.threshold
                {
                    add(out, feature);
                }
            }
            Kind::Gradient(g) => {
                if g.filter.contains(&prev) != g.include {
                    add(out, prev);
                    return true;
                }
                // `ey` is the box's Y range intersected with the replacer's.
                let (pmin, pmax) = g.function.range(ey.0, ey.1);
                if pmin >= 1.0 {
                    add(out, g.block);
                } else if pmax <= 0.0 {
                    add(out, prev);
                } else {
                    add(out, prev);
                    add(out, g.block);
                }
            }
        }
        true
    }
}
