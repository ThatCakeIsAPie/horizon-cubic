//! Height-map tiles for the coarsest levels of a cubic world's far terrain.
//!
//! A far cube node (`lod::cube`, `cube_gen`) is a 16³ grid of voxels each
//! `1 << level` blocks wide. At level 15 a voxel is 32,768 blocks tall while the
//! terrain lies within a few hundred: the surface snaps to whole voxels, a
//! solid voxel's top can be the stone of deep below, and one noise sample per
//! huge voxel lines up with the noise's lattice into a regular tile pattern.
//! A [`TileKey`] covers the same X/Z footprint as a `CubeKey` of its level,
//! all Y, as a 16 × 16 grid of columns `1 << level` blocks wide, and holds per
//! column **exact block heights** (`i32`, no fixed range: a 9,000-block peak is
//! 9,000 blocks, and the cubic range of ±8.4 M is representable).
//!
//! # Columns are layered
//!
//! CubicWorldGen's density makes overhangs, arches and floating islands, and
//! presets put sky islands at many heights. So a column holds up to [`SPANS`]
//! solid spans, top-down, each with a top (the highest solid block), a bottom
//! (the lowest; [`OPEN_BOTTOM`] for the ground, which goes down without end in
//! a cubic world) and the top block. An island at y 5,000 over ground at y 70 is
//! two spans, not a 5,000-block pillar. A column with more spans keeps the top
//! `SPANS - 1` and merges the rest into the last one, and is counted in the
//! tile's `overflow_columns` ([`TileInfo::complex`]): the planner should refine
//! such tiles into 3D cube nodes sooner. Water: a column whose highest span lies
//! below the sea (or that has none) has water up to the sea's top plane. Water
//! between spans (under an island, in a sea cave) is not drawn.
//!
//! # Sampling: one jittered point per column
//!
//! Each column is one vertical line through a hashed quart inside it
//! (`biome_gen::column_jitter`): every span, the top block, the biome and the
//! water come from that line. Nothing is averaged or blended (the owner's call:
//! from very high up whole biomes side by side shimmer as they would, a true
//! picture). The jitter is what matters: the centres of power-of-two columns are
//! spaced by a power of two, which lines up with the noises' lattices (their
//! permutations repeat every 256 cells), so from y = 5,000,000 every coarse
//! column read the same value, a regular tile pattern.
//!
//! A point stands for itself, not for its column, so it keeps the detail a
//! point has: the climate keeps its layers down to 256 blocks
//! (`biome_gen::JITTER_DETAIL_LEVEL`) and the density its octaves down to 64
//! ([`POINT_DETAIL_LEVEL`]), at every level. With the column-wide cut of the
//! cube nodes, the climate of a coarse point is its first layer alone, whose
//! values crowd the middle of the parameter space: from y = 1,000,000 the
//! world was nearly all plains and forest.
//!
//! [`SAMPLES`] points per column with the highest top kept is supported (a
//! peak narrower than a column then shows when any point lands on it) but set
//! to one: four points made 289 level-15 tiles take 18 ms of GPU time instead
//! of 7, and at the coarse levels, where the points of a column are unrelated,
//! the highest of four is land almost wherever one point is, so the oceans all
//! but vanished from y = 1,000,000. With one point a mountain shows in
//! proportion to its area at the coarsest levels, and whole once the plan
//! refines to narrower columns.
//!
//! The biome, height and volatility at a point come from the GPU climate pass
//! (`biome_gen`, its unblended coarse columns, [`BiomeGenParams::jittered`]):
//! one node per tile and sample, the key's `y` being the sample index. Without
//! it ([`ColumnSource::Cpu`]) a CPU function of the points makes them.
//! Positions are whole blocks, read exactly at any distance with the
//! fixed-point phase of `cube_descent` (`cwg_noise.wgsl`).
//!
//! # Finding the spans, and their precision
//!
//! Per sample point the density is `(mix + d) * vol + height - sign(vol) * y`
//! with `mix` between the low and high noises (`cube_gen`'s bound of their
//! amplitudes for the octaves kept), the depth term `d` exact at the point, and `vol`
//! one value below the biome height and another above it. Each piece changes
//! sign only within `height + [mix_lo, mix_hi] * vol` (mirrored for a negative
//! volatility), so the march covers the hull of the two pieces and `height`,
//! two blocks wider, clamped to the cubic range; above and below it the sign is
//! known (one sample each). The march goes down from the top in steps of the
//! column width, at most [`STEP_CAP`] (64) blocks, and at most [`MAX_STEPS`]
//! steps (a taller range takes longer steps); every change of sign between two
//! steps is bisected to the block. So:
//!
//! * a top or bottom is exact to the block for the density evaluated (in `f32`
//!   on the GPU, so within a block where it is flat; `f32` holds every block
//!   coordinate of the cubic range);
//! * a solid run or a gap thinner than the step can be missed, which matches
//!   what a column of that width can show;
//! * the density is the continuous one, not CubicWorldGen's trilinear lattice
//!   (4 × 8 × 4 blocks), without octaves shorter than 64 blocks (or than the
//!   column, if narrower): next to the full generator's `surface_y` it differs
//!   by the lattice's rounding (a few blocks) plus what those octaves add (a
//!   few percent of the volatility).
//!
//! The march starts above anything the density can make solid, so the first
//! span is the topmost solid of the line: overhangs and floating parts above
//! the ground are the higher spans, never skipped.
//!
//! Blocks: a span's top is its biome's top block (`cube_gen`'s replacer rule:
//! the first solid under air or water); its bottom face is stone, or deepslate
//! below the middle of the deepslate gradient (a flag in the span).
//!
//! # Tile format
//!
//! [`TILE_WORDS`] `u32` per tile (13,376 bytes, [`TILE_BYTES`]): a header of
//! [`HEADER_WORDS`], then the 256 columns (`z * 16 + x`) of [`COLUMN_WORDS`]:
//!
//! ```text
//! header  0 origin x   1 origin z (blocks, i32)   2 level   3 most spans in a column
//!         4 highest y (tops and water)   5 skirt y (i32, the lowest any wall goes)
//!         6 columns that overflowed   7 columns with water   8 spans   9..16 zero
//! column  0 count (bits 0..2) | water << 3 | overflow << 4 | sample << 5 | biome << 8
//!         1 + 3i  span i: top (i32; OPEN_TOP when solid to the top of the world)
//!         2 + 3i          bottom (i32; OPEN_BOTTOM for the ground)
//!         3 + 3i          top block state (bits 0..16) | deepslate bottom << 31
//! ```
//!
//! Spans past the count hold `OPEN_BOTTOM, OPEN_BOTTOM, 0`. [`TileData`]
//! converts both ways.
//!
//! # Drawing
//!
//! GPU-only, by vertex pulling from the pool of tiles (`surface_draw.wgsl`,
//! appended to `terrain.wgsl` and `cube_far.wgsl`, so the fragment path, light,
//! fog and fade are the far cube nodes'). Blocky columns, as the cube far
//! terrain looks: every span is a box with a top face, a bottom face (not for
//! the ground) and four walls, coloured by its block's face from the far
//! material table (`cube_far`'s `cf_face`: the sprite, sampled at the level its
//! screen size gives, which is its smallest mip far away, tinted by biome).
//! Walls use the top block's side faces where the step between two columns is
//! no steeper than a slope of one (at these widths it stands for a slope
//! under the top block: plains stay green from the side), and stone (or
//! deepslate low down) where it is steeper: a cliff, as the far cube nodes'
//! rock below the surface. A span under another is lit as roofed over. Water is a flat quad at
//! the sea's top plane, in the translucent pass.
//!
//! * **No cracks.** The ground's walls go down to the neighbouring column's
//!   ground inside the tile and, at the tile's edge (or where the neighbour has
//!   no ground), down to the tile's skirt: the lowest the march could reach in
//!   the tile, minus the tile's relief (at least [`SKIRT_MIN`] blocks). A
//!   neighbouring tile of another level, or a cube node beside it (whose own
//!   borders are walls), is then closed off. Floating spans are closed boxes.
//! * **Culling.** A compute pass (`surface_cull.wgsl`) tests each tile's box
//!   against the frustum, writes its camera-relative origin and fade code, and
//!   compacts one indirect draw per tile and pass (`multi_draw_indirect_count`
//!   where the device has it, else a `multi_draw_indirect` over a cleared
//!   tail), each of `256 × most spans × 6 faces × 6` vertices; faces a column
//!   does not have collapse.
//! * **Large coordinates.** Tile origins are `i32` blocks minus the camera's
//!   integer part; heights are `i32` minus the camera's integer `y`; only then
//!   `f32`.
//!
//! # Cost (GTX 1070 Ti, Custom Cubic defaults)
//!
//! A batch of 256 level-14 tiles takes 4.7 ms of GPU time (3.2 ms the climate
//! pass, 1.5 ms the surface pass); 1,089 of them (a 33 × 33 square, 8.6 M
//! blocks across) are generated and their headers read back 31 ms after the
//! request. Taller terrain marches longer: with `heightFactor` 30,000 a batch
//! of level-9 tiles takes about 18 ms. A tile is 13,376 bytes in the pool,
//! plus 64 bytes of draw state.
//!
//! # API
//!
//! [`SurfaceTiles::new`] on a device with the generator's parameters (the same
//! [`CubeGenParams`] the far cube nodes use) and a [`ColumnSource`];
//! [`SurfaceTiles::enable_draw`] with the terrain's pipeline setup and the far
//! cube nodes' material and biome tables. Then [`SurfaceTiles::request`]
//! (batched, any number), [`SurfaceTiles::update`] once a frame (generates up to
//! [`BATCHES_PER_UPDATE`] batches of [`MAX_BATCH`] tiles, reads tile headers
//! back), [`SurfaceTiles::ready`], [`SurfaceTiles::info`] (with the complex
//! count), [`SurfaceTiles::retire`] (fade out, then free) or
//! [`SurfaceTiles::free`], [`SurfaceTiles::prepare`] before the world pass and
//! [`SurfaceTiles::draw`] in it, [`SurfaceTiles::stats`]. [`cpu_tile`] is the
//! reference the GPU is checked against, and the fallback without compute.
use super::{
    biome_gen::{BLEND_MAX_LEVEL, BiomeGen, BiomeGenParams, CpuBiomes, column_jitter},
    cube_descent::phase_constants,
    cube_far::DrawSetup,
    cube_gen::{
        ColumnTargets, CubeGenParams, DEPTH, HIGH, LOW, MAX_OCTAVES, SELECTOR, noise_bounds,
        octave_terms, perlin, used_octaves,
    },
};
use crate::{
    fast_hash::{FxHashMap, FxHashSet},
    lod::cube::CubeKey,
};
use glam::{DVec3, Vec4};
use horizon_core::{density::SamplerContext, noise::wrap};
use horizon_world::cubic::{MAX_Y, MIN_Y};
use std::{
    borrow::Cow,
    collections::VecDeque,
    sync::{Arc, Mutex, mpsc},
    time::{Duration, Instant},
};

/// Columns along a tile's side.
pub const COLUMNS: usize = 16;
/// Columns in a tile.
pub const COLUMN_COUNT: usize = COLUMNS * COLUMNS;
/// Solid spans a column holds.
pub const SPANS: usize = 4;
/// Sample points per column (the one with the highest top is kept). One: see
/// the module docs ("Sampling").
pub const SAMPLES: usize = 1;
/// The density at a sample point keeps the octaves a voxel of this level
/// (64 blocks) would: a point stands for itself, not for its column.
pub const POINT_DETAIL_LEVEL: u8 = 6;
/// Words of a tile's header.
pub const HEADER_WORDS: usize = 16;
/// Words of one column.
pub const COLUMN_WORDS: usize = 1 + 3 * SPANS;
/// Words of a tile.
pub const TILE_WORDS: usize = HEADER_WORDS + COLUMN_COUNT * COLUMN_WORDS;
/// Bytes of a tile.
pub const TILE_BYTES: u64 = TILE_WORDS as u64 * 4;
/// A span's bottom when it goes down without end (the ground), or "none".
pub const OPEN_BOTTOM: i32 = i32::MIN;
/// A span's top when it is solid up to the top of the world.
pub const OPEN_TOP: i32 = i32::MAX;
/// The march's largest step in blocks.
pub const STEP_CAP: i32 = 64;
/// The march's most steps (a taller range takes longer ones).
pub const MAX_STEPS: i32 = 1024;
/// The least a skirt reaches under a tile's lowest point.
pub const SKIRT_MIN: i32 = 16;
/// Tiles one dispatch generates (the climate pass takes [`SAMPLES`] nodes each).
pub const MAX_BATCH: usize = 256;
/// Batches [`SurfaceTiles::update`] starts at most.
pub const BATCHES_PER_UPDATE: usize = 8;
/// The lowest level the GPU climate pass serves unblended columns for.
pub const MIN_LEVEL: u8 = BLEND_MAX_LEVEL + 1;
/// The highest level (a tile 2^26 blocks wide; the world's octree root).
pub const MAX_LEVEL: u8 = crate::lod::cube::MAX_CUBE_LEVEL;
/// Pool slots by default (about 27 MB).
pub const DEFAULT_CAPACITY: u32 = 2048;
/// Levels in the parameter tables.
const LEVELS: usize = 32;
/// Words of a slot's entry in the draw table.
const TABLE_WORDS: usize = 4;
const NOT_SET: u32 = u32::MAX;
const FADE_MS: u32 = 600;

/// The generation shader: the shared noise, then `surface_gen.wgsl`.
pub(crate) const GEN_SHADER: &str = concat!(
    include_str!("cwg_noise.wgsl"),
    include_str!("surface_gen.wgsl")
);
/// The culling shader.
const CULL_SHADER: &str = include_str!("surface_cull.wgsl");

/// The drawing shader: `terrain.wgsl`, `cube_far.wgsl` (its material lookup),
/// then `surface_draw.wgsl`.
pub fn draw_shader_source() -> String {
    format!(
        "{}\n{}",
        super::cube_far::shader_source(),
        include_str!("surface_draw.wgsl")
    )
}

// ------------------------------------------------------------------- keys

/// A tile: the X/Z footprint of a `CubeKey` of the same level, all Y.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TileKey {
    pub level: u8,
    pub x: i32,
    pub z: i32,
}

impl TileKey {
    pub const fn new(level: u8, x: i32, z: i32) -> Self {
        Self { level, x, z }
    }
    /// The tile over a cube node.
    pub const fn of(key: CubeKey) -> Self {
        Self::new(key.level, key.x, key.z)
    }
    /// The tile of `level` holding block column `(x, z)`.
    pub const fn containing(level: u8, x: i32, z: i32) -> Self {
        Self::new(level, x >> (level + 4), z >> (level + 4))
    }
    /// Blocks along a side.
    pub const fn size(self) -> i32 {
        16 << self.level
    }
    /// Blocks along a column's side.
    pub const fn column_width(self) -> i32 {
        1 << self.level
    }
    /// The lowest corner, in blocks.
    pub const fn origin(self) -> [i32; 2] {
        [self.x * self.size(), self.z * self.size()]
    }
    /// The block column sample `sample` of column `(cx, cz)` is taken at.
    pub fn sample_point(self, cx: usize, cz: usize, sample: usize) -> [i32; 2] {
        let [ox, oz] = self.origin();
        let j = column_jitter(
            (ox >> self.level) + cx as i32,
            (oz >> self.level) + cz as i32,
            self.level,
            sample as u32,
        );
        [
            ox + ((cx as i32) << self.level) + j[0],
            oz + ((cz as i32) << self.level) + j[1],
        ]
    }
    /// The climate pass's node for sample `sample` (the key's `y` seeds the
    /// jitter).
    fn sample_node(self, sample: usize) -> CubeKey {
        CubeKey::new(self.level, self.x, sample as i32, self.z)
    }
}

// ----------------------------------------------------------------- format

/// One solid span of a column.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    /// The highest solid block, or [`OPEN_TOP`].
    pub top: i32,
    /// The lowest solid block, or [`OPEN_BOTTOM`] (the ground).
    pub bottom: i32,
    /// The top block's state id.
    pub block: u32,
    /// The bottom face is deepslate (else stone).
    pub deep_bottom: bool,
}

/// One column of a tile.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Column {
    /// Top-down, at most [`SPANS`].
    pub spans: Vec<Span>,
    /// Water up to the sea's top plane over the highest span.
    pub water: bool,
    /// More spans were found than kept: the last one holds the rest.
    pub overflow: bool,
    /// Which sample point the column took.
    pub sample: u8,
    pub biome: u16,
}

impl Column {
    /// The ground: the last span, when it is open below.
    pub fn ground(&self) -> Option<&Span> {
        self.spans.last().filter(|s| s.bottom == OPEN_BOTTOM)
    }
}

/// A tile's header.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TileInfo {
    /// The most spans any column holds (what the draw is sized by).
    pub max_spans: u32,
    /// The highest top or water plane's block.
    pub highest: i32,
    /// How low the skirts and ground walls at the tile's edge go.
    pub skirt: i32,
    /// Columns that had more than [`SPANS`] spans: the tile is complex.
    pub overflow_columns: u32,
    pub water_columns: u32,
    pub spans: u32,
}

impl TileInfo {
    /// Whether some column had more spans than a tile keeps: the planner
    /// should prefer 3D cube nodes here.
    pub fn complex(&self) -> bool {
        self.overflow_columns > 0
    }
}

/// A tile's contents on the CPU.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TileData {
    pub key: TileKey,
    pub info: TileInfo,
    /// `z * 16 + x`.
    pub columns: Vec<Column>,
}

impl TileData {
    pub fn column(&self, x: usize, z: usize) -> &Column {
        &self.columns[z * COLUMNS + x]
    }

    /// The tile's [`TILE_WORDS`] words (the module docs' layout).
    pub fn to_words(&self) -> Vec<u32> {
        let mut w = vec![0u32; TILE_WORDS];
        let [ox, oz] = self.key.origin();
        let i = &self.info;
        w[..9].copy_from_slice(&[
            ox as u32,
            oz as u32,
            u32::from(self.key.level),
            i.max_spans,
            i.highest as u32,
            i.skirt as u32,
            i.overflow_columns,
            i.water_columns,
            i.spans,
        ]);
        for (c, col) in self.columns.iter().enumerate() {
            let at = HEADER_WORDS + c * COLUMN_WORDS;
            w[at] = col.spans.len().min(SPANS) as u32
                | u32::from(col.water) << 3
                | u32::from(col.overflow) << 4
                | u32::from(col.sample & 3) << 5
                | u32::from(col.biome) << 8;
            for s in 0..SPANS {
                let span = col.spans.get(s);
                let at = at + 1 + s * 3;
                w[at] = span.map_or(OPEN_BOTTOM, |s| s.top) as u32;
                w[at + 1] = span.map_or(OPEN_BOTTOM, |s| s.bottom) as u32;
                w[at + 2] = span.map_or(0, |s| {
                    (s.block & 0x1ffff) | (u32::from(s.deep_bottom) << 31)
                });
            }
        }
        w
    }

    /// A tile from its words; `None` if they are not one (too short, a
    /// count past [`SPANS`]).
    pub fn from_words(words: &[u32]) -> Option<Self> {
        if words.len() < TILE_WORDS || words[2] > u32::from(MAX_LEVEL) {
            return None;
        }
        let level = words[2] as u8;
        let size = 16i32 << level;
        let key = TileKey::new(
            level,
            (words[0] as i32).div_euclid(size),
            (words[1] as i32).div_euclid(size),
        );
        let info = TileInfo {
            max_spans: words[3],
            highest: words[4] as i32,
            skirt: words[5] as i32,
            overflow_columns: words[6],
            water_columns: words[7],
            spans: words[8],
        };
        let mut columns = Vec::with_capacity(COLUMN_COUNT);
        for c in 0..COLUMN_COUNT {
            let at = HEADER_WORDS + c * COLUMN_WORDS;
            let head = words[at];
            let count = (head & 7) as usize;
            if count > SPANS {
                return None;
            }
            columns.push(Column {
                spans: (0..count)
                    .map(|s| {
                        let at = at + 1 + s * 3;
                        Span {
                            top: words[at] as i32,
                            bottom: words[at + 1] as i32,
                            block: words[at + 2] & 0x1ffff,
                            deep_bottom: words[at + 2] >> 31 == 1,
                        }
                    })
                    .collect(),
                water: head >> 3 & 1 == 1,
                overflow: head >> 4 & 1 == 1,
                sample: (head >> 5 & 3) as u8,
                biome: (head >> 8 & 0xffff) as u16,
            });
        }
        Some(Self { key, info, columns })
    }
}

// -------------------------------------------------------------- the march

/// The y range a sample point's march covers, its step, and the density's
/// sign above and below it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Range {
    pub lo: i32,
    pub hi: i32,
    pub step: i32,
    /// Solid just above `hi` (so up to the top of the world).
    pub above: bool,
    /// Solid just below `lo` (so down without end).
    pub below: bool,
}

impl Range {
    /// The step for a column `width` blocks wide over `lo..=hi`.
    pub(crate) fn step_for(width: i32, lo: i32, hi: i32) -> i32 {
        let step = width.clamp(1, STEP_CAP);
        let reach = hi - lo;
        if reach / step > MAX_STEPS {
            (reach + MAX_STEPS - 1) / MAX_STEPS
        } else {
            step
        }
    }
}

/// Bisects between `lo` (in state `lo_solid`) and `hi` above it (the other
/// state): the last y of `lo`'s state.
fn bisect(mut lo: i32, mut hi: i32, lo_solid: bool, solid: &mut impl FnMut(i32) -> bool) -> i32 {
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if solid(mid) == lo_solid {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

/// The highest solid block of a line: [`OPEN_TOP`] when it is solid above the
/// range, [`OPEN_BOTTOM`] when it has none.
pub(crate) fn find_top(r: &Range, mut solid: impl FnMut(i32) -> bool) -> i32 {
    if r.above {
        return OPEN_TOP;
    }
    let (mut prev, mut y) = (r.hi + 1, r.hi);
    loop {
        if solid(y) {
            return bisect(y, prev, true, &mut solid);
        }
        if y <= r.lo {
            break;
        }
        prev = y;
        y = (y - r.step).max(r.lo);
    }
    if r.below { r.lo - 1 } else { OPEN_BOTTOM }
}

/// The solid spans of a line, top-down: `(top, bottom)`, at most [`SPANS`]
/// (the rest merged into the last), and whether it overflowed.
pub(crate) fn march(r: &Range, mut solid: impl FnMut(i32) -> bool) -> (Vec<(i32, i32)>, bool) {
    let mut spans: Vec<(i32, i32)> = Vec::with_capacity(SPANS);
    let mut overflow = false;
    let mut inside = r.above;
    if inside {
        spans.push((OPEN_TOP, OPEN_BOTTOM));
    }
    let (mut prev, mut y) = (r.hi + 1, r.hi);
    loop {
        let here = solid(y);
        if here != inside {
            if here {
                let top = bisect(y, prev, true, &mut solid);
                if spans.len() < SPANS {
                    spans.push((top, OPEN_BOTTOM));
                } else {
                    overflow = true;
                }
            } else {
                let bottom = bisect(y, prev, false, &mut solid) + 1;
                spans.last_mut().expect("inside a span").1 = bottom;
            }
            inside = here;
        }
        if y <= r.lo {
            break;
        }
        prev = y;
        y = (y - r.step).max(r.lo);
    }
    if inside {
        spans.last_mut().expect("inside a span").1 = if r.below { OPEN_BOTTOM } else { r.lo };
    } else if r.below {
        if spans.len() < SPANS {
            spans.push((r.lo - 1, OPEN_BOTTOM));
        } else {
            overflow = true;
            spans.last_mut().expect("a full column").1 = OPEN_BOTTOM;
        }
    }
    (spans, overflow)
}

// ------------------------------------------------------------ the density

/// What a sample point holds before the density: CubicWorldGen's biome height
/// and volatility (the units of `cube_gen::NodeColumns`) and the biome id.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColumnSample {
    pub height: f32,
    pub volatility: f32,
    pub biome: u32,
}

/// Makes the samples at block columns `(x, z)`, in order.
pub type PointColumns = Arc<dyn Fn(&[[i32; 2]]) -> Vec<ColumnSample> + Send + Sync>;

/// Where the tiles' sample points get their biome, height and volatility.
#[derive(Clone)]
pub enum ColumnSource {
    /// The GPU climate pass (`biome_gen`), jittered, from these parameters.
    Gpu(Arc<BiomeGenParams>),
    /// A CPU function of the points: worlds the climate pass does not
    /// reproduce (another biome size, one biome), and tests.
    Cpu(PointColumns),
}

/// The climate pass's CPU reference at the points.
pub fn cpu_biome_points(source: Arc<CpuBiomes>) -> PointColumns {
    Arc::new(move |points: &[[i32; 2]]| {
        let mut context = SamplerContext::cached();
        points
            .iter()
            .map(|&[x, z]| {
                let (height, volatility, biome) = source.column_at(&mut context, x, z);
                ColumnSample {
                    height,
                    volatility,
                    biome,
                }
            })
            .collect()
    })
}

/// One octave on the CPU: scale per block, weight, table offsets and
/// permutation.
struct CpuOctave<'a> {
    scale: [f64; 3],
    weight: f32,
    offsets: [f64; 3],
    perm: &'a [u8; 256],
}

/// The density's noises for one level, the shader's arithmetic in `f32` after
/// an `f64` lattice split (as `cube_gen`'s reference).
pub(crate) struct CpuDensity<'a> {
    p: &'a CubeGenParams,
    octaves: [Vec<CpuOctave<'a>>; 4],
    offsets: [f32; 4],
    mix: [f32; 2],
    level: u8,
}

/// The lowest and highest value the low/high mix can take at `level`.
pub(crate) fn level_mix(p: &CubeGenParams, level: u8) -> [f32; 2] {
    let low = noise_bounds(p, LOW, used_octaves(p, LOW, level));
    let high = noise_bounds(p, HIGH, used_octaves(p, HIGH, level));
    [low.0.min(high.0) as f32, low.1.max(high.1) as f32]
}

impl<'a> CpuDensity<'a> {
    pub(crate) fn new(p: &'a CubeGenParams, level: u8) -> Self {
        let detail = level.min(POINT_DETAIL_LEVEL);
        let octaves = std::array::from_fn(|n| {
            let noise = &p.noises[n];
            octave_terms(noise, n == DEPTH)
                .into_iter()
                .zip(&noise.octaves)
                .take(used_octaves(p, n, detail))
                .map(|((scale, weight), o)| CpuOctave {
                    scale,
                    weight,
                    offsets: o.offsets,
                    perm: &o.permutations,
                })
                .collect()
        });
        Self {
            p,
            octaves,
            offsets: std::array::from_fn(|n| p.noises[n].offset as f32),
            mix: level_mix(p, detail),
            level,
        }
    }

    fn noise(&self, n: usize, pos: [f64; 3]) -> f32 {
        let mut sum = 0.0f32;
        for o in &self.octaves[n] {
            let mut cell = [0i32; 3];
            let mut frac = [0f32; 3];
            for a in 0..3 {
                let shifted = wrap(pos[a] * o.scale[a]) + o.offsets[a];
                let c = shifted.floor();
                cell[a] = (c as i64 & 255) as i32;
                frac[a] = (shifted - c) as f32;
            }
            sum += o.weight * perlin(o.perm, cell, frac);
        }
        sum + self.offsets[n]
    }

    /// The sample point's column terms.
    pub(crate) fn probe(&'a self, point: [i32; 2], sample: ColumnSample) -> Probe<'a> {
        let p = self.p;
        let raw = self.noise(DEPTH, [f64::from(point[0]), 0.0, f64::from(point[1])]);
        Probe {
            density: self,
            x: f64::from(point[0]),
            z: f64::from(point[1]),
            height: sample.height * p.height_factor as f32 + p.height_offset as f32,
            vol_below: sample.volatility
                * p.below_average_factor as f32
                * p.height_variation_factor as f32
                + p.height_variation_offset as f32,
            vol_above: sample.volatility * p.height_variation_factor as f32
                + p.height_variation_offset as f32,
            depth: depth_transform(raw),
        }
    }
}

/// `cwg_noise.wgsl`'s `depth_transform`, in `f32`.
fn depth_transform(raw: f32) -> f32 {
    let mut d = raw;
    if d < 0.0 {
        d *= -0.3;
    }
    d = (d * 3.0 - 2.0).clamp(-2.0, 1.0);
    if d < 0.0 {
        d /= 5.6;
    } else if d > 0.0 {
        d /= 8.0;
    }
    d * (0.2 * 17.0 / 64.0)
}

/// One sample point's vertical line.
pub(crate) struct Probe<'a> {
    density: &'a CpuDensity<'a>,
    x: f64,
    z: f64,
    height: f32,
    vol_below: f32,
    vol_above: f32,
    depth: f32,
}

/// Where one volatility's piece of the density can change sign.
fn piece(h: f32, vol: f32, mlo: f32, mhi: f32) -> (f32, f32) {
    if vol > 0.0 {
        (h + mlo * vol, h + mhi * vol)
    } else if vol < 0.0 {
        // density = M * vol + h + y: zero at y = -h - M * vol.
        (-h - mlo * vol, -h - mhi * vol)
    } else {
        (h, h)
    }
}

impl Probe<'_> {
    /// The density at block `y` (> 0 is solid).
    pub(crate) fn density(&self, y: i32) -> f32 {
        let d = self.density;
        let pos = [self.x, f64::from(y), self.z];
        let sel = d.noise(SELECTOR, pos).clamp(0.0, 1.0);
        let mixed = if sel <= 0.0 {
            d.noise(LOW, pos)
        } else if sel >= 1.0 {
            d.noise(HIGH, pos)
        } else {
            let low = d.noise(LOW, pos);
            let high = d.noise(HIGH, pos);
            low + sel * (high - low)
        };
        let yf = y as f32;
        let vol = if self.height > yf {
            self.vol_below
        } else {
            self.vol_above
        };
        let sign = if vol > 0.0 {
            1.0
        } else if vol < 0.0 {
            -1.0
        } else {
            0.0
        };
        (mixed + self.depth) * vol + self.height - sign * yf
    }

    pub(crate) fn solid(&self, y: i32) -> bool {
        self.density(y) > 0.0
    }

    /// The march's range (see the module docs).
    pub(crate) fn range(&self) -> Range {
        let [m0, m1] = self.density.mix;
        let (mlo, mhi) = (m0 + self.depth, m1 + self.depth);
        let a = piece(self.height, self.vol_below, mlo, mhi);
        let b = piece(self.height, self.vol_above, mlo, mhi);
        let lo_f = a.0.min(b.0).min(self.height);
        let hi_f = a.1.max(b.1).max(self.height);
        let (ymin, ymax) = (MIN_Y as f32, MAX_Y as f32);
        let lo = (lo_f - 2.0 - lo_f.abs() * 1.0e-6).floor().clamp(ymin, ymax) as i32;
        let hi = (hi_f + 2.0 + hi_f.abs() * 1.0e-6).ceil().clamp(ymin, ymax) as i32;
        Range {
            lo,
            hi,
            step: Range::step_for(1 << self.density.level, lo, hi),
            above: self.solid(hi + 1),
            below: self.solid(lo - 1),
        }
    }
}

/// The tile on the CPU: the GPU pass's algorithm with the sample points'
/// columns from `columns` (called once, with every point, `(tile column z *
/// 16 + x) * SAMPLES + sample` in order). The reference the GPU is checked
/// against, and the fallback without compute shaders.
pub fn cpu_tile(
    p: &CubeGenParams,
    key: TileKey,
    columns: &dyn Fn(&[[i32; 2]]) -> Vec<ColumnSample>,
) -> TileData {
    let density = CpuDensity::new(p, key.level);
    let mut points = Vec::with_capacity(COLUMN_COUNT * SAMPLES);
    for c in 0..COLUMN_COUNT {
        for s in 0..SAMPLES {
            points.push(key.sample_point(c % COLUMNS, c / COLUMNS, s));
        }
    }
    let samples = columns(&points);
    let biome_count = p.biomes.len().clamp(1, 256);
    let deep_below = ((p.deepslate_from + p.deepslate_to) / 2.0) as f32;
    let sea = p.sea_level;
    let mut out = Vec::with_capacity(COLUMN_COUNT);
    let mut info = TileInfo {
        highest: OPEN_BOTTOM,
        ..TileInfo::default()
    };
    let mut lowest = OPEN_TOP;
    for c in 0..COLUMN_COUNT {
        let probe = |s: usize| density.probe(points[c * SAMPLES + s], samples[c * SAMPLES + s]);
        let (mut best, mut best_top) = (0, OPEN_BOTTOM);
        for s in (0..SAMPLES).filter(|_| SAMPLES > 1) {
            let pr = probe(s);
            let top = find_top(&pr.range(), |y| pr.solid(y));
            if s == 0 || top > best_top {
                (best, best_top) = (s, top);
            }
        }
        let pr = probe(best);
        let range = pr.range();
        let (lines, overflow) = march(&range, |y| pr.solid(y));
        let sample = samples[c * SAMPLES + best];
        let biome = (sample.biome as usize).min(biome_count - 1);
        let top_block = p.biomes.get(biome).map_or(p.stone, |b| b.top) & 0x1ffff;
        let spans: Vec<Span> = lines
            .iter()
            .map(|&(top, bottom)| Span {
                top,
                bottom,
                block: top_block,
                deep_bottom: bottom != OPEN_BOTTOM && (bottom as f32) < deep_below,
            })
            .collect();
        let top0 = spans.first().map_or(OPEN_BOTTOM, |s| s.top);
        let water = top0 != OPEN_TOP && top0 < sea && (MIN_Y..=MAX_Y).contains(&sea);
        info.spans += spans.len() as u32;
        info.max_spans = info.max_spans.max(spans.len() as u32);
        if !spans.is_empty() {
            info.highest = info
                .highest
                .max(if top0 == OPEN_TOP { MAX_Y } else { top0 });
        }
        if water {
            info.highest = info.highest.max(sea);
            info.water_columns += 1;
        }
        info.overflow_columns += u32::from(overflow);
        lowest = lowest.min(range.lo - 1);
        out.push(Column {
            spans,
            water,
            overflow,
            sample: best as u8,
            biome: (sample.biome & 0xffff) as u16,
        });
    }
    let highest = info.highest.max(lowest);
    info.highest = highest;
    info.skirt = (lowest - SKIRT_MIN.max(highest - lowest)).max(MIN_Y);
    TileData {
        key,
        info,
        columns: out,
    }
}

// ------------------------------------------------------------ GPU layouts

/// `surface_gen.wgsl`'s `Params`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuParams {
    shape: [f32; 4],
    shape2: [f32; 4],
    world: [i32; 4],
    noffs: [f32; 4],
    used: [[u32; 4]; LEVELS],
    mix: [[f32; 4]; LEVELS],
    w: [[f32; 4]; 64],
    kint: [[u32; 4]; 64],
    kfrac: [[u32; 4]; 64],
    off: [[u32; 4]; 64],
    tops: [[u32; 4]; 64],
}

impl GpuParams {
    fn new(p: &CubeGenParams) -> Self {
        let mut out = <Self as bytemuck::Zeroable>::zeroed();
        out.shape = [
            p.height_factor as f32,
            p.height_offset as f32,
            p.height_variation_factor as f32,
            p.height_variation_offset as f32,
        ];
        out.shape2 = [
            p.below_average_factor as f32,
            ((p.deepslate_from + p.deepslate_to) / 2.0) as f32,
            0.0,
            0.0,
        ];
        out.world = [
            MIN_Y,
            MAX_Y,
            p.sea_level,
            p.biomes.len().clamp(1, 256) as i32,
        ];
        for (n, noise) in p.noises.iter().enumerate() {
            out.noffs[n] = noise.offset as f32;
            for (k, (scale, weight)) in octave_terms(noise, n == DEPTH).into_iter().enumerate() {
                let slot = n * MAX_OCTAVES + k;
                out.w[slot] = [scale[0] as f32, scale[1] as f32, scale[2] as f32, weight];
                let offsets = noise.octaves[k].offsets;
                for a in 0..3 {
                    let (ki, kf, ko) = phase_constants(scale[a], offsets[a]);
                    out.kint[slot][a] = ki;
                    out.kfrac[slot][a] = kf;
                    out.off[slot][a] = ko;
                }
            }
        }
        for level in 0..LEVELS {
            let l = level as u8;
            let detail = l.min(POINT_DETAIL_LEVEL);
            out.used[level] = std::array::from_fn(|n| used_octaves(p, n, detail) as u32);
            let [lo, hi] = level_mix(p, detail);
            out.mix[level] = [lo, hi, 0.0, 0.0];
        }
        for (i, b) in p.biomes.iter().take(256).enumerate() {
            out.tops[i / 4][i % 4] = b.top & 0x1ffff;
        }
        if p.biomes.is_empty() {
            out.tops[0][0] = p.stone;
        }
        out
    }
}

/// `surface_cull.wgsl`'s `Frame` (and `surface_draw.wgsl`'s `StFrame`).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FrameUniform {
    planes: [[f32; 4]; 6],
    cam_int: [i32; 4],
    cam_frac: [f32; 4],
    info: [u32; 4],
    world: [i32; 4],
    ids: [u32; 4],
}

// -------------------------------------------------------------- the tiles

/// Counters.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SurfaceStats {
    /// Tiles in the pool.
    pub resident: usize,
    /// Asked for, not generated yet.
    pub pending: usize,
    /// Tiles generated, ever.
    pub generated: u64,
    /// Dispatches, ever.
    pub batches: u64,
    /// Pool bytes in use.
    pub bytes: u64,
    /// Pool bytes in all.
    pub capacity_bytes: u64,
    /// Submit to finished (headers read back) of the last batch, and the most.
    pub last_batch_ms: f32,
    pub max_batch_ms: f32,
    /// GPU time of the last batch from timestamp queries (0 without them):
    /// the climate pass, the surface pass, and the slowest batch's total.
    pub last_biome_gpu_ms: f32,
    pub last_surface_gpu_ms: f32,
    pub max_gpu_ms: f32,
    /// The slowest batch's climate and surface passes.
    pub slowest_split_ms: [f32; 2],
    /// Every batch's GPU time, summed.
    pub gpu_ms: f64,
    /// Tiles of the last batch.
    pub last_batch_tiles: usize,
    /// Resident tiles with columns past [`SPANS`] spans.
    pub complex: usize,
}

struct Tile {
    slot: u32,
    info: Option<TileInfo>,
    dying: Option<u32>,
}

/// A batch on the GPU, its headers on the way back.
struct Inflight {
    tiles: Vec<(TileKey, u32)>,
    staging: wgpu::Buffer,
    submitted: Instant,
    done: Arc<Mutex<Option<(bool, Instant)>>>,
}

enum Columns {
    Gpu(BiomeGen),
    Cpu(PointColumns),
}

/// Timestamps around a batch's two passes, where the device has them.
struct Timer {
    set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    /// Nanoseconds a tick.
    period: f32,
}

/// Bytes of a batch's three timestamps.
const TIMER_BYTES: u64 = 24;

/// The draw pipelines and their buffers.
struct Drawer {
    table: wgpu::Buffer,
    solid_args: wgpu::Buffer,
    water_args: wgpu::Buffer,
    draw_counts: wgpu::Buffer,
    frame: wgpu::Buffer,
    cull: wgpu::ComputePipeline,
    cull_group: wgpu::BindGroup,
    draw_group: wgpu::BindGroup,
    solid: wgpu::RenderPipeline,
    water: wgpu::RenderPipeline,
    counts: bool,
}

/// The tiles of the coarse far terrain on one device: generation, the pool
/// and drawing (see the module docs).
pub struct SurfaceTiles {
    device: wgpu::Device,
    queue: wgpu::Queue,
    params: Arc<CubeGenParams>,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    jobs: wgpu::Buffer,
    column_buf: wgpu::Buffer,
    biome_buf: wgpu::Buffer,
    pool: wgpu::Buffer,
    capacity: u32,
    free: Vec<u32>,
    /// Slots below this have been used (the cull pass covers them).
    high_water: u32,
    columns: Columns,
    tiles: FxHashMap<TileKey, Tile>,
    pending: VecDeque<TileKey>,
    queued: FxHashSet<TileKey>,
    inflight: Vec<Inflight>,
    draw: Option<Drawer>,
    timer: Option<Timer>,
    epoch: Instant,
    stats: SurfaceStats,
}

fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

impl SurfaceTiles {
    /// The tiles on `device` with a pool of `capacity` tiles, or `None` when
    /// it cannot run the passes (no compute shaders, too few storage buffers,
    /// a pool past the storage binding size, a climate pass that does not
    /// build).
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        params: Arc<CubeGenParams>,
        source: ColumnSource,
        capacity: u32,
    ) -> Option<Self> {
        let limits = device.limits();
        if TILE_BYTES * u64::from(capacity) > limits.max_storage_buffer_binding_size
            || limits.max_storage_buffers_per_shader_stage < 6
            || capacity == 0
        {
            return None;
        }
        let columns = match source {
            ColumnSource::Gpu(biomes) => Columns::Gpu(BiomeGen::new(
                device.clone(),
                queue.clone(),
                Arc::new((*biomes).clone().jittered()),
            )?),
            ColumnSource::Cpu(f) => Columns::Cpu(f),
        };
        let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Surface tiles"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(GEN_SHADER)),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Surface tiles"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let buffer = |label, size: u64, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let gpu_params = GpuParams::new(&params);
        let (_, perms) = super::cube_gen::GpuParams::new(&params);
        let params_buf = buffer(
            "Surface params",
            std::mem::size_of::<GpuParams>() as u64,
            storage,
        );
        let perms_buf = buffer("Surface permutations", (perms.len() * 4) as u64, storage);
        let jobs = buffer("Surface jobs", (MAX_BATCH * 16) as u64, storage);
        let nodes = (MAX_BATCH * SAMPLES) as u64;
        let column_buf = buffer("Surface columns", nodes * 512 * 4, storage);
        let biome_buf = buffer("Surface column biomes", nodes * 256 * 4, storage);
        let pool = buffer(
            "Surface tile pool",
            TILE_BYTES * u64::from(capacity),
            storage | wgpu::BufferUsages::COPY_SRC,
        );
        queue.write_buffer(&params_buf, 0, bytemuck::bytes_of(&gpu_params));
        queue.write_buffer(&perms_buf, 0, bytemuck::cast_slice(&perms));
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Surface tiles"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                entry(0, &params_buf),
                entry(1, &perms_buf),
                entry(2, &jobs),
                entry(3, &column_buf),
                entry(4, &biome_buf),
                entry(5, &pool),
            ],
        });
        let timer = device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY)
            .then(|| Timer {
                set: device.create_query_set(&wgpu::QuerySetDescriptor {
                    label: Some("Surface tiles timing"),
                    ty: wgpu::QueryType::Timestamp,
                    count: 3,
                }),
                resolve: buffer(
                    "Surface tiles timing",
                    wgpu::QUERY_RESOLVE_BUFFER_ALIGNMENT,
                    wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                ),
                period: queue.get_timestamp_period(),
            });
        let internal = pollster::block_on(internal.pop());
        if let Some(e) = internal.or(pollster::block_on(validation.pop())) {
            log::warn!("Horizon renderer: no surface tiles ({e})");
            return None;
        }
        Some(Self {
            device: device.clone(),
            queue: queue.clone(),
            params,
            pipeline,
            bind_group,
            jobs,
            column_buf,
            biome_buf,
            pool,
            capacity,
            free: (0..capacity).rev().collect(),
            high_water: 0,
            columns,
            tiles: FxHashMap::default(),
            pending: VecDeque::new(),
            queued: FxHashSet::default(),
            inflight: Vec::new(),
            draw: None,
            timer,
            epoch: Instant::now(),
            stats: SurfaceStats {
                capacity_bytes: TILE_BYTES * u64::from(capacity),
                ..SurfaceStats::default()
            },
        })
    }

    /// Builds the draw: `setup` the terrain pipelines' (`cube_far::DrawSetup`),
    /// `materials` and `biomes` the far cube nodes' tables
    /// (`CubeFar::material_buffer`, `biome_buffer`), which must hold the
    /// generator's states. False when the pipelines do not build.
    pub fn enable_draw(
        &mut self,
        setup: &DrawSetup<'_>,
        materials: &wgpu::Buffer,
        biomes: &wgpu::Buffer,
    ) -> bool {
        let device = &self.device;
        let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let buffer = |label, size: u64, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let st = wgpu::BufferUsages::STORAGE;
        let dst = wgpu::BufferUsages::COPY_DST;
        let indirect = wgpu::BufferUsages::INDIRECT;
        let slots = u64::from(self.capacity);
        let table = buffer(
            "Surface draw table",
            slots * TABLE_WORDS as u64 * 4,
            st | dst,
        );
        let instances = buffer("Surface instances", slots * 16, st);
        let solid_args = buffer("Surface solid draws", slots * 16, st | indirect | dst);
        let water_args = buffer("Surface water draws", slots * 16, st | indirect | dst);
        let draw_counts = buffer("Surface draw counts", 8, st | indirect | dst);
        let frame = buffer(
            "Surface frame",
            std::mem::size_of::<FrameUniform>() as u64,
            wgpu::BufferUsages::UNIFORM | dst,
        );
        let compute = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty,
            count: None,
        };
        let storage = |read_only| wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let uniform = wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let cull_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Surface culling"),
            entries: &[
                compute(0, uniform),
                compute(1, storage(true)),
                compute(2, storage(true)),
                compute(3, storage(false)),
                compute(4, storage(false)),
                compute(5, storage(false)),
                compute(6, storage(false)),
            ],
        });
        let cull_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Surface culling"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(CULL_SHADER)),
        });
        let cull = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Surface culling"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Surface culling"),
                    bind_group_layouts: &[Some(&cull_layout)],
                    ..Default::default()
                }),
            ),
            module: &cull_module,
            entry_point: Some("cull_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let cull_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Surface culling"),
            layout: &cull_layout,
            entries: &[
                entry(0, &frame),
                entry(1, &table),
                entry(2, &self.pool),
                entry(3, &instances),
                entry(4, &solid_args),
                entry(5, &water_args),
                entry(6, &draw_counts),
            ],
        });
        let vertex = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX,
            ty,
            count: None,
        };
        let draw_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Surface draws"),
            entries: &[
                vertex(32, storage(true)),
                vertex(33, storage(true)),
                vertex(34, storage(true)),
                vertex(35, storage(true)),
                vertex(36, uniform),
            ],
        });
        let draw_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Surface draws"),
            layout: &draw_layout,
            entries: &[
                entry(32, materials),
                entry(33, biomes),
                entry(34, &self.pool),
                entry(35, &instances),
                entry(36, &frame),
            ],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Surface tiles terrain"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(draw_shader_source())),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Surface tiles terrain"),
            bind_group_layouts: &[Some(setup.bindings), Some(&draw_layout)],
            ..Default::default()
        });
        let constants = [
            ("SPRITE_ARRAYS", setup.sprite_constants[0]),
            ("SPRITE_SPLIT", setup.sprite_constants[1]),
        ];
        let pipeline = |label: &str, vs: &str, fs: &str, blend: Option<wgpu::BlendState>| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some(vs),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: Some(wgpu::Face::Back),
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::GreaterEqual),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(fs),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants: &constants,
                        ..Default::default()
                    },
                    targets: &[Some(wgpu::ColorTargetState {
                        format: setup.format,
                        blend,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let solid = pipeline("Surface tiles solid", "vs_surface", "fs_solid", None);
        let water = pipeline(
            "Surface tiles water",
            "vs_surface_water",
            "fs_translucent",
            Some(wgpu::BlendState::ALPHA_BLENDING),
        );
        if let Some(error) = pollster::block_on(validation.pop()) {
            log::warn!("Horizon renderer: surface tiles not drawn ({error})");
            return false;
        }
        // Tiles already resident are drawn from now.
        let now = self.now_ms();
        for tile in self.tiles.values() {
            let words = [
                1 | u32::from(tile.dying.is_some()) << 1,
                now,
                tile.dying.unwrap_or(NOT_SET),
                0,
            ];
            self.queue.write_buffer(
                &table,
                u64::from(tile.slot) * TABLE_WORDS as u64 * 4,
                bytemuck::cast_slice(&words),
            );
        }
        self.draw = Some(Drawer {
            table,
            solid_args,
            water_args,
            draw_counts,
            frame,
            cull,
            cull_group,
            draw_group,
            solid,
            water,
            counts: setup.counts,
        });
        true
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    fn now_ms(&self) -> u32 {
        (self.epoch.elapsed().as_millis() as u32).min(NOT_SET - 1)
    }

    /// Queues tiles to generate (any number; ones resident or queued are
    /// skipped). [`Self::update`] generates them in batches.
    pub fn request(&mut self, keys: &[TileKey]) {
        for &key in keys {
            if key.level <= MAX_LEVEL && !self.tiles.contains_key(&key) && self.queued.insert(key) {
                self.pending.push_back(key);
            }
        }
    }

    /// Whether `key` is generated (its batch is submitted: later GPU work,
    /// draws included, sees it).
    pub fn ready(&self, key: TileKey) -> bool {
        self.tiles.contains_key(&key)
    }

    /// The tile's header, once read back (a frame or two after `ready`).
    pub fn info(&self, key: TileKey) -> Option<TileInfo> {
        self.tiles.get(&key)?.info
    }

    /// Columns of the tile that had more spans than it keeps, once known: a
    /// tile with any is complex (refine it into cube nodes sooner).
    pub fn complex(&self, key: TileKey) -> Option<u32> {
        self.info(key).map(|i| i.overflow_columns)
    }

    /// Drops a tile at once (or its request).
    pub fn free(&mut self, key: TileKey) {
        if self.queued.remove(&key) {
            self.pending.retain(|k| *k != key);
        }
        if let Some(tile) = self.tiles.remove(&key) {
            self.write_table(tile.slot, [0, NOT_SET, NOT_SET, 0]);
            self.free.push(tile.slot);
        }
    }

    /// Fades a tile out, then frees it (a queued one is dropped at once).
    pub fn retire(&mut self, key: TileKey) {
        if self.queued.contains(&key) || self.draw.is_none() {
            return self.free(key);
        }
        let now = self.now_ms();
        let Some(tile) = self.tiles.get_mut(&key) else {
            return;
        };
        if tile.dying.is_none() {
            tile.dying = Some(now);
            let slot = tile.slot;
            self.write_word(slot, 0, 3);
            self.write_word(slot, 2, now);
        }
    }

    /// Every key resident or queued.
    pub fn keys(&self) -> impl Iterator<Item = TileKey> + '_ {
        self.tiles.keys().chain(self.pending.iter()).copied()
    }

    fn write_table(&self, slot: u32, words: [u32; TABLE_WORDS]) {
        if let Some(d) = &self.draw {
            self.queue.write_buffer(
                &d.table,
                u64::from(slot) * TABLE_WORDS as u64 * 4,
                bytemuck::cast_slice(&words),
            );
        }
    }

    fn write_word(&self, slot: u32, word: usize, value: u32) {
        if let Some(d) = &self.draw {
            self.queue.write_buffer(
                &d.table,
                (u64::from(slot) * TABLE_WORDS as u64 + word as u64) * 4,
                bytemuck::bytes_of(&value),
            );
        }
    }

    /// Reads back finished batches' headers, frees faded-out tiles and
    /// generates up to [`BATCHES_PER_UPDATE`] batches. Call once a frame.
    pub fn update(&mut self) {
        let _ = self.device.poll(wgpu::PollType::Poll);
        self.collect();
        let now = self.now_ms();
        let faded: Vec<TileKey> = self
            .tiles
            .iter()
            .filter(|(_, t)| t.dying.is_some_and(|d| now.saturating_sub(d) >= FADE_MS))
            .map(|(&k, _)| k)
            .collect();
        for key in faded {
            self.free(key);
        }
        for _ in 0..BATCHES_PER_UPDATE {
            if self.pending.is_empty() || self.free.is_empty() {
                break;
            }
            self.generate_batch();
        }
    }

    /// Takes in the headers of batches the GPU finished.
    fn collect(&mut self) {
        let mut i = 0;
        while i < self.inflight.len() {
            let done = *self.inflight[i].done.lock().unwrap();
            let Some((ok, at)) = done else {
                i += 1;
                continue;
            };
            let batch = self.inflight.swap_remove(i);
            let ms = at.duration_since(batch.submitted).as_secs_f32() * 1000.0;
            self.stats.last_batch_ms = ms;
            self.stats.max_batch_ms = self.stats.max_batch_ms.max(ms);
            self.stats.last_batch_tiles = batch.tiles.len();
            if !ok {
                continue;
            }
            let words: Vec<u32> = {
                let Ok(view) = batch.staging.slice(..).get_mapped_range() else {
                    continue;
                };
                bytemuck::cast_slice::<u8, u32>(&view).to_vec()
            };
            batch.staging.unmap();
            let stamps = batch.tiles.len() * HEADER_WORDS;
            if let (Some(t), Some(w)) = (&self.timer, words.get(stamps..stamps + 6)) {
                let at = |i: usize| u64::from(w[i * 2]) | (u64::from(w[i * 2 + 1]) << 32);
                let ms = |a: u64, b: u64| b.saturating_sub(a) as f32 * t.period / 1.0e6;
                self.stats.last_biome_gpu_ms = ms(at(0), at(1));
                self.stats.last_surface_gpu_ms = ms(at(1), at(2));
                if ms(at(0), at(2)) > self.stats.max_gpu_ms {
                    self.stats.max_gpu_ms = ms(at(0), at(2));
                    self.stats.slowest_split_ms = [ms(at(0), at(1)), ms(at(1), at(2))];
                }
                self.stats.gpu_ms += f64::from(ms(at(0), at(2)));
            }
            for (j, (key, slot)) in batch.tiles.iter().enumerate() {
                let h = &words[j * HEADER_WORDS..(j + 1) * HEADER_WORDS];
                if let Some(tile) = self.tiles.get_mut(key).filter(|t| t.slot == *slot) {
                    tile.info = Some(TileInfo {
                        max_spans: h[3],
                        highest: h[4] as i32,
                        skirt: h[5] as i32,
                        overflow_columns: h[6],
                        water_columns: h[7],
                        spans: h[8],
                    });
                }
            }
        }
    }

    /// Generates the next batch of pending tiles: one submission.
    fn generate_batch(&mut self) {
        let mut batch: Vec<(TileKey, u32)> = Vec::new();
        while batch.len() < MAX_BATCH {
            let Some(key) = self.pending.pop_front() else {
                break;
            };
            self.queued.remove(&key);
            let Some(slot) = self.free.pop() else {
                self.pending.push_front(key);
                self.queued.insert(key);
                break;
            };
            batch.push((key, slot));
        }
        if batch.is_empty() {
            return;
        }
        let n = batch.len();
        let jobs: Vec<[i32; 4]> = batch
            .iter()
            .map(|(key, slot)| {
                let [ox, oz] = key.origin();
                [ox, oz, i32::from(key.level), *slot as i32]
            })
            .collect();
        self.queue
            .write_buffer(&self.jobs, 0, bytemuck::cast_slice(&jobs));
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Surface tiles"),
            });
        if let Some(t) = &self.timer {
            // An empty pass that stamps the start.
            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Surface tiles start"),
                timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                    query_set: &t.set,
                    beginning_of_pass_write_index: Some(0),
                    end_of_pass_write_index: None,
                }),
            });
        }
        let nodes: Vec<CubeKey> = batch
            .iter()
            .flat_map(|(key, _)| (0..SAMPLES).map(|s| key.sample_node(s)))
            .collect();
        match &self.columns {
            Columns::Gpu(biomes) => biomes.encode(
                &mut encoder,
                &nodes,
                ColumnTargets {
                    columns: &self.column_buf,
                    biomes: &self.biome_buf,
                },
            ),
            Columns::Cpu(f) => {
                let (columns, biomes) = cpu_columns(f, &batch);
                self.queue
                    .write_buffer(&self.column_buf, 0, bytemuck::cast_slice(&columns));
                self.queue
                    .write_buffer(&self.biome_buf, 0, bytemuck::cast_slice(&biomes));
            }
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Surface tiles"),
                timestamp_writes: self
                    .timer
                    .as_ref()
                    .map(|t| wgpu::ComputePassTimestampWrites {
                        query_set: &t.set,
                        beginning_of_pass_write_index: Some(1),
                        end_of_pass_write_index: Some(2),
                    }),
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(n as u32, 1, 1);
        }
        let header_bytes = (HEADER_WORDS * 4) as u64;
        let timed = if self.timer.is_some() { TIMER_BYTES } else { 0 };
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Surface headers"),
            size: header_bytes * n as u64 + timed,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if let Some(t) = &self.timer {
            encoder.resolve_query_set(&t.set, 0..3, &t.resolve, 0);
            encoder.copy_buffer_to_buffer(
                &t.resolve,
                0,
                &staging,
                header_bytes * n as u64,
                TIMER_BYTES,
            );
        }
        for (i, (_, slot)) in batch.iter().enumerate() {
            encoder.copy_buffer_to_buffer(
                &self.pool,
                TILE_BYTES * u64::from(*slot),
                &staging,
                header_bytes * i as u64,
                header_bytes,
            );
        }
        let submitted = Instant::now();
        self.queue.submit([encoder.finish()]);
        let done = Arc::new(Mutex::new(None));
        let flag = done.clone();
        staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            *flag.lock().unwrap() = Some((r.is_ok(), Instant::now()));
        });
        let now = self.now_ms();
        for &(key, slot) in &batch {
            self.high_water = self.high_water.max(slot + 1);
            self.tiles.insert(
                key,
                Tile {
                    slot,
                    info: None,
                    dying: None,
                },
            );
            self.write_table(slot, [1, now, NOT_SET, 0]);
        }
        self.stats.generated += n as u64;
        self.stats.batches += 1;
        self.inflight.push(Inflight {
            tiles: batch,
            staging,
            submitted,
            done,
        });
    }

    /// Waits for every batch in flight and takes in their headers (tests,
    /// tools, measurements).
    pub fn finish(&mut self) {
        if self.inflight.is_empty() {
            return;
        }
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(30)),
        });
        self.collect();
    }

    /// A resident tile's contents, read back now (blocking): tests and tools.
    pub fn read_tile(&mut self, key: TileKey) -> Option<TileData> {
        let slot = self.tiles.get(&key)?.slot;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Surface tile readback"),
            size: TILE_BYTES,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Surface tile readback"),
            });
        encoder.copy_buffer_to_buffer(
            &self.pool,
            TILE_BYTES * u64::from(slot),
            &staging,
            0,
            TILE_BYTES,
        );
        let submission = self.queue.submit([encoder.finish()]);
        let (tx, rx) = mpsc::channel();
        staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let polled = self.device.poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: Some(Duration::from_secs(30)),
        });
        if polled.is_err() || !matches!(rx.recv_timeout(Duration::from_secs(30)), Ok(Ok(()))) {
            return None;
        }
        let words: Vec<u32> = {
            let view = staging.slice(..).get_mapped_range().ok()?;
            bytemuck::cast_slice::<u8, u32>(&view).to_vec()
        };
        staging.unmap();
        self.collect();
        TileData::from_words(&words)
    }

    /// The culling pass for this frame (before the world pass): `planes` the
    /// camera-relative frustum.
    pub fn prepare(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        camera: DVec3,
        planes: [Vec4; 6],
    ) {
        let Some(d) = &self.draw else {
            return;
        };
        if self.high_water == 0 {
            return;
        }
        let floor = camera.floor();
        let p = &self.params;
        let frame = FrameUniform {
            planes: planes.map(|p| p.to_array()),
            cam_int: [floor.x as i32, floor.y as i32, floor.z as i32, 0],
            cam_frac: [
                (camera.x - floor.x) as f32,
                (camera.y - floor.y) as f32,
                (camera.z - floor.z) as f32,
                0.0,
            ],
            info: [self.now_ms(), self.high_water, 0, 0],
            world: [
                p.sea_level,
                MIN_Y,
                MAX_Y,
                ((p.deepslate_from + p.deepslate_to) / 2.0).floor() as i32,
            ],
            ids: [p.water, p.stone, p.deepslate, 0],
        };
        self.queue
            .write_buffer(&d.frame, 0, bytemuck::bytes_of(&frame));
        encoder.clear_buffer(&d.draw_counts, 0, None);
        if !d.counts {
            // Draws past the count are empty ones.
            let bytes = u64::from(self.high_water) * 16;
            encoder.clear_buffer(&d.solid_args, 0, Some(bytes));
            encoder.clear_buffer(&d.water_args, 0, Some(bytes));
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Surface tiles culling"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&d.cull);
        pass.set_bind_group(0, &d.cull_group, &[]);
        pass.dispatch_workgroups(self.high_water.div_ceil(64), 1, 1);
    }

    /// Draws the tiles (`water` false) or their water, in the world pass with
    /// the terrain's group 0 bound.
    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>, water: bool) {
        let Some(d) = &self.draw else {
            return;
        };
        if self.high_water == 0 || self.tiles.is_empty() {
            return;
        }
        pass.set_pipeline(if water { &d.water } else { &d.solid });
        pass.set_bind_group(1, &d.draw_group, &[]);
        let args = if water { &d.water_args } else { &d.solid_args };
        if d.counts {
            pass.multi_draw_indirect_count(
                args,
                0,
                &d.draw_counts,
                u64::from(water) * 4,
                self.high_water,
            );
        } else {
            pass.multi_draw_indirect(args, 0, self.high_water);
        }
    }

    /// The counters.
    pub fn stats(&self) -> SurfaceStats {
        SurfaceStats {
            resident: self.tiles.len(),
            pending: self.pending.len(),
            bytes: self.tiles.len() as u64 * TILE_BYTES,
            complex: self
                .tiles
                .values()
                .filter(|t| t.info.is_some_and(|i| i.complex()))
                .count(),
            ..self.stats
        }
    }

    /// Whether every request is generated and its header read back.
    pub fn idle(&self) -> bool {
        self.pending.is_empty() && self.inflight.is_empty()
    }
}

// ------------------------------------------------------------ debug view

/// A temporary view for checking the tiles without the planner:
/// `HORIZON_SURFACE_TILES_DEBUG=<level>[,<radius>]` draws every tile of one
/// level in a square of `2 * radius + 1` tiles (default radius 8) around the
/// camera, in place of the far cube nodes, and logs the generation times.
/// `auto` for the level picks it by the camera's height above the sea:
/// `log2(height / 40) + 1`, about the level the plan uses straight below.
pub(crate) struct DebugTiles {
    pub tiles: SurfaceTiles,
    /// `None`: by the camera's height.
    fixed: Option<u8>,
    level: u8,
    radius: i32,
    centre: Option<TileKey>,
    asked: Option<(Instant, f64, u64)>,
    logged: bool,
}

impl DebugTiles {
    /// The view, when the variable asks for one and the tiles run on the far
    /// cube nodes' device.
    pub(crate) fn from_env(
        backend: &super::cube_gen::GpuBackend,
        far: &super::cube_far::CubeFar,
        setup: &DrawSetup<'_>,
    ) -> Option<Self> {
        let value = std::env::var("HORIZON_SURFACE_TILES_DEBUG").ok()?;
        let mut parts = value.split(',');
        let level = parts.next()?.trim();
        let fixed = match level {
            "auto" => None,
            l => Some(l.parse::<u8>().ok()?.clamp(MIN_LEVEL, MAX_LEVEL)),
        };
        let radius: i32 = parts
            .next()
            .and_then(|r| r.trim().parse().ok())
            .unwrap_or(8)
            .clamp(0, 64);
        let source = match &backend.biome_params {
            Some(biomes) => ColumnSource::Gpu(biomes.clone()),
            None => {
                log::warn!("surface tiles debug: no GPU climate pass in this world");
                return None;
            }
        };
        let side = (2 * radius + 1) as u32;
        let capacity = (side * side * 2).max(64);
        let mut tiles = SurfaceTiles::new(
            &backend.device,
            &backend.queue,
            backend.params.clone(),
            source,
            capacity,
        )?;
        if !tiles.enable_draw(setup, far.material_buffer(), far.biome_buffer()) {
            return None;
        }
        log::info!(
            "surface tiles debug: level {level}, radius {radius}, {capacity} slots of {TILE_BYTES} bytes"
        );
        Some(Self {
            tiles,
            fixed,
            level: fixed.unwrap_or(MIN_LEVEL),
            radius,
            centre: None,
            asked: None,
            logged: false,
        })
    }

    /// Asks for the square around the camera, generates, culls.
    pub(crate) fn frame(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        camera: DVec3,
        planes: [Vec4; 6],
    ) {
        let level = self.fixed.unwrap_or_else(|| {
            let height = (camera.y - 64.0).abs().max(80.0);
            ((height / 40.0).log2().floor() as i32 + 1)
                .clamp(i32::from(MIN_LEVEL), i32::from(MAX_LEVEL)) as u8
        });
        if level != self.level {
            self.level = level;
            self.centre = None;
        }
        let centre = TileKey::containing(level, camera.x.floor() as i32, camera.z.floor() as i32);
        if self.centre != Some(centre) {
            self.centre = Some(centre);
            let r = self.radius;
            let wanted: FxHashSet<TileKey> = (-r..=r)
                .flat_map(|dz| (-r..=r).map(move |dx| (dx, dz)))
                .map(|(dx, dz)| TileKey::new(level, centre.x + dx, centre.z + dz))
                .collect();
            let stale: Vec<TileKey> = self.tiles.keys().filter(|k| !wanted.contains(k)).collect();
            for key in stale {
                self.tiles.free(key);
            }
            let mut keys: Vec<TileKey> = wanted.into_iter().collect();
            // Nearest first.
            keys.sort_by_key(|k| (k.x - centre.x).abs().max((k.z - centre.z).abs()));
            self.tiles.request(&keys);
            let s = self.tiles.stats();
            self.asked = Some((Instant::now(), s.gpu_ms, s.batches));
            self.logged = false;
        }
        self.tiles.update();
        self.tiles.prepare(encoder, camera, planes);
        if let Some((asked, gpu_ms, batches)) = self.asked
            && self.tiles.idle()
            && !self.logged
        {
            let s = self.tiles.stats();
            log::info!(
                "surface tiles debug: {} tiles of level {} at y {:.0}: ready {:.1} ms after the \
                 request, {} batches, GPU {:.2} ms in all (last batch {} tiles: climate {:.2} ms + \
                 surface {:.2} ms; slowest batch {:.2} ms = {:.2} + {:.2}; submit to read back \
                 {:.2} ms, slowest {:.2} ms); {:.1} MB resident, {} complex",
                s.resident,
                level,
                camera.y,
                asked.elapsed().as_secs_f64() * 1000.0,
                s.batches - batches,
                s.gpu_ms - gpu_ms,
                s.last_batch_tiles,
                s.last_biome_gpu_ms,
                s.last_surface_gpu_ms,
                s.max_gpu_ms,
                s.slowest_split_ms[0],
                s.slowest_split_ms[1],
                s.last_batch_ms,
                s.max_batch_ms,
                s.bytes as f64 / 1_048_576.0,
                s.complex,
            );
            self.logged = true;
        }
    }
}

/// The CPU columns of a batch in the climate pass's layout: per node (tile *
/// SAMPLES + sample) 256 heights and 256 volatilities, and 256 biome ids.
fn cpu_columns(f: &PointColumns, batch: &[(TileKey, u32)]) -> (Vec<f32>, Vec<u32>) {
    let mut points = Vec::with_capacity(batch.len() * SAMPLES * COLUMN_COUNT);
    for (key, _) in batch {
        for s in 0..SAMPLES {
            for c in 0..COLUMN_COUNT {
                points.push(key.sample_point(c % COLUMNS, c / COLUMNS, s));
            }
        }
    }
    let samples = f(&points);
    let nodes = batch.len() * SAMPLES;
    let mut columns = vec![0f32; nodes * 512];
    let mut biomes = vec![0u32; nodes * 256];
    for (i, sample) in samples.iter().enumerate().take(nodes * COLUMN_COUNT) {
        let (node, c) = (i / COLUMN_COUNT, i % COLUMN_COUNT);
        columns[node * 512 + c] = sample.height;
        columns[node * 512 + 256 + c] = sample.volatility;
        biomes[node * 256 + c] = sample.biome;
    }
    (columns, biomes)
}

#[cfg(test)]
#[path = "surface_tile_tests.rs"]
mod tests;
