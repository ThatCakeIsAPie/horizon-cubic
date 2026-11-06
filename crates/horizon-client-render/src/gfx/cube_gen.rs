//! GPU generation of far terrain nodes for cubic worlds.
//!
//! A far node ([`CubeKey`], `lod::cube`) is a 16^3 grid of voxels, each
//! `1 << level` blocks wide. Far terrain needs no ores, caves, structures or
//! decoration, so it is generated here from the world generator's *density*
//! alone, on the GPU, at the voxel centres of a batch of nodes at once.
//!
//! # What runs on the GPU (`cube_gen.wgsl`)
//!
//! One workgroup per node, one thread per voxel column. The thread evaluates
//! the CubicWorldGen-style density
//!
//! ```text
//! selector = clamp(sel(x,y,z), 0, 1); mix = low + selector*(high-low)
//! d = depth(x,z); if d < 0 { d *= -0.3 }; d = clamp(d*3-2, -2, 1)
//! d = if d < 0 { d/5.6 } else { d/8 }; d *= 0.2*17/64
//! height = biomeHeight*heightFactor + heightOffset
//! vol = biomeVolatility * (if height > y { belowAverage } else { 1 })
//!       * heightVariationFactor + heightVariationOffset
//! density = (mix + d)*vol + height - signum(vol)*y        // > 0 is solid
//! ```
//!
//! down the column (from `1 + ceil(filler/size)` voxels above the node, so the
//! top of a node knows what lies over it), and applies simplified replacers:
//! the first solid voxel under air or water is the biome's top block; solid
//! voxels within `filler_depth` blocks under the surface are its filler;
//! deeper rock is stone, deepslate below `deepslate_from` and a hash gradient
//! up to `deepslate_to`; air at or below the sea level is water. Only block
//! state ids are written: biomes pass through from the input.
//!
//! Each noise is `factor * sum_i (0.5^i / norm) * P_i(wrap(x*fX*2^i), ..) +
//! offset`, `P_i` vanilla's `ImprovedNoise` (`horizon_core::noise::PerlinNoise`)
//! with the octave's own permutation table; the depth noise is 2D (`y = 0`).
//! Biome height and volatility per column come from the CPU ([`NodeColumns`]).
//!
//! # Precision at large coordinates
//!
//! GPUs have `f32`, worlds reach +-8 million blocks. For every node and
//! octave the CPU computes in `f64` the octave's input at the node origin,
//! `wrap(origin * f * 2^i) + table offset` ([`GpuHeader`]), and splits it
//! into an integer lattice cell (kept mod 256: the permutation index wraps
//! there, and `wrap` only ever subtracts multiples of 2^25) and an `f32`
//! fraction. The shader adds the voxel's offset from the node origin,
//! `local_blocks * f * 2^i`, in `f32` (small numbers), then re-splits the
//! sum into cell and fraction. Octaves whose wavelength is shorter than the
//! node's voxel size are skipped (their count is per node): they are noise at
//! a scale the voxel cannot show.
//!
//! # Batches and threads
//!
//! [`CubeGen`] owns the pipeline and a persistent [`NodePool`] (fixed-size
//! slots of u32 voxel states plus column biomes; the exact layout is documented
//! there) and generates up to [`MAX_BATCH`] nodes per dispatch into pool slots
//! (`generate_to_pool`: no readback, so a later GPU mesher can read the pool
//! buffer directly; `NodePool::free` releases a slot). Readback is a separate
//! copy from slots to a staging buffer (`read_back`), which `generate` and the
//! sampler use so CPU meshing keeps working. [`GpuCubeSampler`] implements `CubeSampler` on top: the
//! calling worker computes the biome columns (on its own thread, so the
//! complex biome source runs in parallel), sends `(key, columns)` to the one
//! GPU thread and waits for the node; the GPU thread takes whatever requests
//! are queued, up to [`BATCH_NODES`], per dispatch. Without compute shaders it
//! falls back to [`cpu_reference_node`] in the calling thread.
//!
//! # Bulk requests and the 2D pre-pass
//!
//! `GpuCubeSampler::request(&keys)` takes any number of keys. Per node the
//! calling thread makes the columns and runs [`classify`]: from the columns'
//! height and volatility and the noises' amplitude bounds it derives the y
//! range the surface can occupy; nodes wholly above it are filled as air or
//! water, nodes wholly below as rock, on the CPU with no 3D noise
//! ([`Class`], [`prepass_node`]). Only the rest reach the GPU thread, which
//! merges every queued message (up to [`BATCH_NODES`] nodes) and dispatches
//! chunks of up to [`MAX_BATCH`] nodes (bounded by the buffers and by free
//! pool slots). Cost follows the surface area, not the volume. The origin
//! split above is per node, so large batches far from the origin stay exact.
//! Pre-classified nodes use no pool slot (`CubeGen::generate_to_pool` only
//! places nodes the caller found [`Class::Mixed`]).
//!
//! # Columns from the GPU
//!
//! The CPU column closure is the last per-node CPU cost. `gfx/biome_gen` makes
//! the same [`NodeColumns`] with a compute pass; [`CubeGen::generate_to_pool_gpu`]
//! (and `generate_gpu_columns`, which reads the nodes back) takes a `fill`
//! callback that records that pass into the same command encoder, writing
//! straight into the buffers this shader reads ([`ColumnTargets`]). The
//! encoder runs it before the generation pass, so nothing is uploaded or read
//! back. The 2D pre-pass ([`classify`]) needs the columns on the CPU and so
//! does not run on this path: every node takes the 3D pass.
//!
//! # Wiring (for the integrator)
//!
//! 1. Fill a [`CubeGenParams`] from the CPU generator: the four noises (octave
//!    permutation tables and offsets come from each octave's `PerlinNoise`
//!    table: `table().permutations()`, `table().offsets()`), the height
//!    parameters, block state ids, and a (top, filler) pair per biome id.
//! 2. Build the column closure (`Fn(CubeKey) -> NodeColumns`): per voxel
//!    column `z * 16 + x` at the voxel centre, the biome's height, volatility
//!    and biome id.
//! 3. Make the sampler with `GpuCubeSampler::new(device, queue, params,
//!    columns)` (the renderer's device, or `with_own_device`) and hand it to
//!    the cube LOD producer; `fork()` gives each worker a handle.
use crate::gfx::biome_gen::{BiomeGen, BiomeGenParams};
use crate::gfx::cube_descent::GpuDescent;
use crate::lod::cube::{CubeKey, CubeNode, CubeSampler, NodeClass, Resident, SIDE, VOLUME, Voxel};
use bytemuck::Zeroable;
use horizon_core::noise::wrap;
use horizon_world::registry::BlockStateId;
use std::{
    borrow::Cow,
    sync::{Arc, Mutex, mpsc},
    time::Duration,
};

/// The generation shader: the shared noise (`cwg_noise.wgsl`), then
/// `cube_gen.wgsl`.
pub(crate) const SHADER: &str = concat!(
    include_str!("cwg_noise.wgsl"),
    include_str!("cube_gen.wgsl")
);

/// Nodes one [`CubeGen`] dispatch can hold.
pub const MAX_BATCH: usize = 512;
/// Nodes the [`GpuCubeSampler`]'s GPU thread takes per dispatch: small enough
/// not to hold the GPU for long.
pub const BATCH_NODES: usize = 4096;
/// Octaves kept per noise (further ones weigh `0.5^16` or less).
pub const MAX_OCTAVES: usize = 16;
/// Biome ids with their own (top, filler) pair; larger ids use the last.
pub const MAX_BIOMES: usize = 256;
/// The lowest level the sampler answers for.
pub const MIN_LEVEL: u8 = 2;

/// Noise indices in [`CubeGenParams::noises`].
pub const SELECTOR: usize = 0;
pub const LOW: usize = 1;
pub const HIGH: usize = 2;
/// The 2D depth noise.
pub const DEPTH: usize = 3;

/// One octave of a noise: one `ImprovedNoise`'s tables.
#[derive(Clone, Debug)]
pub struct OctaveDesc {
    pub permutations: [u8; 256],
    pub offsets: [f64; 3],
}

/// One noise, `factor * sum_i 0.5^i / norm * P_i(x*fX*2^i, ...) + offset`.
#[derive(Clone, Debug)]
pub struct NoiseDesc {
    pub factor: f64,
    pub offset: f64,
    /// Per block. The depth noise ignores the y frequency.
    pub frequency: [f64; 3],
    pub octaves: Vec<OctaveDesc>,
}

/// A biome's surface blocks, as block state ids.
#[derive(Clone, Copy, Debug)]
pub struct BiomeBlocks {
    pub top: u32,
    pub filler: u32,
}

/// Everything the generator needs, as plain data.
#[derive(Clone, Debug)]
pub struct CubeGenParams {
    /// Selector, low, high (3D), depth (2D): see [`SELECTOR`].
    pub noises: [NoiseDesc; 4],
    pub height_factor: f64,
    pub height_offset: f64,
    pub height_variation_factor: f64,
    pub height_variation_offset: f64,
    /// Volatility factor where the average height is above the point.
    pub below_average_factor: f64,
    pub sea_level: i32,
    pub air: u32,
    pub stone: u32,
    pub deepslate: u32,
    pub water: u32,
    /// Indexed by biome id.
    pub biomes: Vec<BiomeBlocks>,
    /// Blocks of filler under the top block (voxels of this size or more get
    /// none).
    pub filler_depth: f64,
    /// Below this y all rock is deepslate; up to `deepslate_to` it thins out.
    pub deepslate_from: f64,
    pub deepslate_to: f64,
}

/// The 2D inputs of one node: per voxel column `z * 16 + x`, at the column's
/// centre.
#[derive(Clone, Debug)]
pub struct NodeColumns {
    pub height: [f32; 256],
    pub volatility: [f32; 256],
    pub biome: [u16; 256],
}

impl NodeColumns {
    pub fn flat(height: f32, volatility: f32, biome: u16) -> Self {
        Self {
            height: [height; 256],
            volatility: [volatility; 256],
            biome: [biome; 256],
        }
    }
}

/// Makes a node's columns; called on the worker thread that asked.
pub type ColumnFn = Arc<dyn Fn(CubeKey) -> NodeColumns + Send + Sync>;

/// The two buffers the shader reads a batch's columns from, for a GPU pass to
/// fill (see [`CubeGen::generate_to_pool_gpu`]). Node `i` of the batch is the
/// `i`th key:
///
/// * `columns`: `f32`s, 512 per node: the 256 column heights (`z * 16 + x`),
///   then the 256 volatilities (`NodeColumns::height`, `volatility`);
/// * `biomes`: `u32`s, 256 per node: the column biome ids.
///
/// Both have storage usage and hold [`MAX_BATCH`] nodes.
#[derive(Clone, Copy)]
pub struct ColumnTargets<'a> {
    pub columns: &'a wgpu::Buffer,
    pub biomes: &'a wgpu::Buffer,
}

// ------------------------------------------------------------ GPU layouts

/// `cube_gen.wgsl`'s `Params`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuParams {
    height: [f32; 4],
    misc: [f32; 4],
    misc2: [f32; 4],
    ids: [u32; 4],
    biome_count: [u32; 4],
    noise_offsets: [f32; 4],
    octaves: [[f32; 4]; 64],
    biomes: [[u32; 4]; MAX_BIOMES],
}

/// `cube_gen.wgsl`'s `Header`: one node.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuHeader {
    info: [i32; 4],
    counts: [u32; 4],
    extra: [u32; 4],
    cell: [[i32; 4]; 64],
    frac: [[f32; 4]; 64],
}

/// The weights and scales of a noise's kept octaves.
pub(crate) fn octave_terms(noise: &NoiseDesc, depth: bool) -> Vec<([f64; 3], f32)> {
    let norm: f64 = (0..noise.octaves.len())
        .map(|i| 0.5f64.powi(i as i32))
        .sum();
    (0..noise.octaves.len().min(MAX_OCTAVES))
        .map(|i| {
            let k = 2f64.powi(i as i32);
            let f = noise.frequency;
            let scale = [f[0] * k, if depth { 0.0 } else { f[1] * k }, f[2] * k];
            (scale, (noise.factor * 0.5f64.powi(i as i32) / norm) as f32)
        })
        .collect()
}

impl GpuParams {
    /// The parameter block and the packed permutation tables (64 `u32` per
    /// octave slot, `noise * 16 + octave`, four table bytes per word).
    pub fn new(p: &CubeGenParams) -> (Self, Vec<u32>) {
        let mut out = Self::zeroed();
        out.height = [
            p.height_factor as f32,
            p.height_offset as f32,
            p.height_variation_factor as f32,
            p.height_variation_offset as f32,
        ];
        out.misc = [
            p.below_average_factor as f32,
            p.filler_depth as f32,
            (p.sea_level + 1) as f32,
            p.deepslate_from as f32,
        ];
        out.misc2 = [p.deepslate_to as f32, 0.0, 0.0, 0.0];
        out.ids = [p.air, p.stone, p.deepslate, p.water];
        let count = p.biomes.len().clamp(1, MAX_BIOMES);
        out.biome_count = [count as u32, 0, 0, 0];
        for (slot, b) in out.biomes.iter_mut().zip(&p.biomes).take(MAX_BIOMES) {
            *slot = [b.top, b.filler, 0, 0];
        }
        let mut perms = vec![0u32; 4 * MAX_OCTAVES * 64];
        for (n, noise) in p.noises.iter().enumerate() {
            out.noise_offsets[n] = noise.offset as f32;
            for (k, (scale, weight)) in octave_terms(noise, n == DEPTH).into_iter().enumerate() {
                let slot = n * MAX_OCTAVES + k;
                out.octaves[slot] = [scale[0] as f32, scale[1] as f32, scale[2] as f32, weight];
                let table = &noise.octaves[k].permutations;
                for (w, bytes) in table.as_chunks::<4>().0.iter().enumerate() {
                    perms[slot * 64 + w] = u32::from_le_bytes(*bytes);
                }
            }
        }
        (out, perms)
    }
}

/// How many octaves of noise `n` a node of `level` shows: those whose
/// wavelength is at least a voxel (the first always).
pub(crate) fn used_octaves(p: &CubeGenParams, n: usize, level: u8) -> usize {
    let noise = &p.noises[n];
    let kept = noise.octaves.len().min(MAX_OCTAVES);
    let f = noise.frequency;
    let top = if n == DEPTH {
        f[0].max(f[2])
    } else {
        f[0].max(f[1]).max(f[2])
    };
    let voxel = f64::from(1u32 << level);
    (0..kept)
        .filter(|&i| i == 0 || 1.0 / (top * 2f64.powi(i as i32)) >= voxel)
        .count()
}

/// Voxel layers computed above a node, so its top layer sees the surface.
pub(crate) fn extra_layers(p: &CubeGenParams, level: u8) -> u32 {
    ((p.filler_depth / f64::from(1u32 << level)).ceil() as u32).clamp(1, 4)
}

/// The octave input at the node origin: lattice cell mod 256 and fraction.
fn origin_split(origin: [i32; 3], scale: [f64; 3], offsets: [f64; 3]) -> ([i32; 3], [f32; 3]) {
    let mut cell = [0; 3];
    let mut frac = [0.0; 3];
    for a in 0..3 {
        let shifted = wrap(f64::from(origin[a]) * scale[a]) + offsets[a];
        let c = shifted.floor();
        cell[a] = (c as i64 & 255) as i32;
        frac[a] = (shifted - c) as f32;
    }
    (cell, frac)
}

impl GpuHeader {
    /// A node's header (the per-node, per-octave precision split).
    pub fn new(p: &CubeGenParams, key: CubeKey, slot: u32) -> Self {
        let mut h = Self::zeroed();
        h.extra[1] = slot;
        let origin = key.origin();
        h.info = [origin[0], origin[1], origin[2], i32::from(key.level)];
        h.extra[0] = extra_layers(p, key.level);
        for (n, noise) in p.noises.iter().enumerate() {
            h.counts[n] = used_octaves(p, n, key.level) as u32;
            for (k, (scale, _)) in octave_terms(noise, n == DEPTH).into_iter().enumerate() {
                let (cell, frac) = origin_split(origin, scale, noise.octaves[k].offsets);
                let slot = n * MAX_OCTAVES + k;
                h.cell[slot] = [cell[0], cell[1], cell[2], 0];
                h.frac[slot] = [frac[0], frac[1], frac[2], 0.0];
            }
        }
        h
    }
}

// ------------------------------------------------------------ 2D pre-pass

/// What the pre-pass knows of a node without the 3D noise.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    /// Every voxel is air or water: the surface lies below the node.
    Empty,
    /// Every voxel (and the layers the shader looks at above) is rock: the
    /// surface lies above the node.
    Solid,
    /// The surface may cross the node: the 3D pass decides.
    Mixed,
}

/// A bound on `|ImprovedNoise|`: the true maximum is a little under 1.
const PERLIN_BOUND: f64 = 1.05;
/// Blocks of slack on the surface range, for `f32` differences.
const SURFACE_SLACK: f64 = 1.0;

/// The range of a noise from its octave weights and offset.
pub(crate) fn noise_bounds(p: &CubeGenParams, n: usize, used: usize) -> (f64, f64) {
    let amplitude: f64 = octave_terms(&p.noises[n], n == DEPTH)
        .iter()
        .take(used)
        .map(|(_, w)| f64::from(*w).abs())
        .sum::<f64>()
        * PERLIN_BOUND;
    (
        p.noises[n].offset - amplitude,
        p.noises[n].offset + amplitude,
    )
}

/// The density's depth term from the depth noise value.
fn depth_term(v: f64) -> f64 {
    let mut d = if v < 0.0 { v * -0.3 } else { v };
    d = (d * 3.0 - 2.0).clamp(-2.0, 1.0);
    d = if d < 0.0 { d / 5.6 } else { d / 8.0 };
    d * (0.2 * 17.0 / 64.0)
}

/// Classifies a node from its columns and the noises' amplitude bounds alone.
///
/// Per column the density is `M * vol + height - y` (`vol > 0`) with `M`, the
/// noise mix plus depth term, bounded by the noise amplitudes; so below
/// `min(M * vol) + height` every voxel centre is solid and above
/// `max(M * vol) + height` every one is not (`vol` is taken at both its
/// below-average and its plain value). A column with `vol <= 0` is never
/// classified. The node is [`Class::Empty`] when its lowest centre is above
/// every column's ceiling, [`Class::Solid`] when the highest centre the shader
/// computes (including the layers above) is below every column's floor.
pub fn classify(p: &CubeGenParams, key: CubeKey, columns: &NodeColumns) -> Class {
    let level = key.level;
    let s = f64::from(1u32 << level);
    let used: [usize; 4] = std::array::from_fn(|n| used_octaves(p, n, level));
    let (low_lo, low_hi) = noise_bounds(p, LOW, used[LOW]);
    let (high_lo, high_hi) = noise_bounds(p, HIGH, used[HIGH]);
    let (depth_lo, depth_hi) = noise_bounds(p, DEPTH, used[DEPTH]);
    let mut depth_min = depth_term(depth_lo).min(depth_term(depth_hi));
    let mut depth_max = depth_term(depth_lo).max(depth_term(depth_hi));
    if depth_lo < 0.0 && depth_hi > 0.0 {
        depth_min = depth_min.min(depth_term(0.0));
        depth_max = depth_max.max(depth_term(0.0));
    }
    // The selector is clamped to 0..1: the mix is between low and high.
    let mix_lo = low_lo.min(high_lo) + depth_min;
    let mix_hi = low_hi.max(high_hi) + depth_max;
    let mut floor = f64::INFINITY;
    let mut ceiling = f64::NEG_INFINITY;
    for col in 0..256 {
        let height = f64::from(columns.height[col]) * p.height_factor + p.height_offset;
        let base = f64::from(columns.volatility[col]) * p.height_variation_factor;
        let vols = [
            base * p.below_average_factor + p.height_variation_offset,
            base + p.height_variation_offset,
        ];
        if vols.iter().any(|&v| v.is_nan() || v <= 0.0) {
            return Class::Mixed;
        }
        for vol in vols {
            floor = floor.min(mix_lo * vol + height);
            ceiling = ceiling.max(mix_hi * vol + height);
        }
    }
    let origin_y = f64::from(key.origin()[1]);
    let lowest = origin_y + 0.5 * s;
    let highest = origin_y + (15.5 + f64::from(extra_layers(p, level))) * s;
    if lowest > ceiling + SURFACE_SLACK {
        Class::Empty
    } else if highest < floor - SURFACE_SLACK {
        Class::Solid
    } else {
        Class::Mixed
    }
}

/// The block of a rock voxel (not top or filler) at layer `jj` of column
/// `(x, z)`: stone, or deepslate by the depth gradient.
fn rock_block(p: &CubeGenParams, key: CubeKey, x: usize, jj: i32, z: usize) -> u32 {
    let level = key.level;
    let origin = key.origin();
    let yc = origin[1] as f32 + ((f64::from(jj) + 0.5) * f64::from(1u32 << level)) as f32;
    if yc < p.deepslate_from as f32 {
        p.deepslate
    } else if yc < p.deepslate_to as f32 {
        let t = (p.deepslate_to as f32 - yc) / (p.deepslate_to as f32 - p.deepslate_from as f32);
        let voxel = [
            (origin[0] >> level) + x as i32,
            (origin[1] >> level) + jj,
            (origin[2] >> level) + z as i32,
        ];
        if hash_voxel(voxel) < t {
            p.deepslate
        } else {
            p.stone
        }
    } else {
        p.stone
    }
}

/// What the 2D pre-pass says of the whole node, for the planner
/// (`CubeSampler::classify`): uniform air or water draws nothing, uniform rock
/// is a closed shell, the rest needs sampling.
pub fn node_class(p: &CubeGenParams, key: CubeKey, columns: &NodeColumns) -> NodeClass {
    match classify(p, key, columns) {
        Class::Mixed => NodeClass::Mixed,
        Class::Solid => NodeClass::Rock,
        Class::Empty => {
            // Water below the sea, air above it: uniform unless the sea level
            // crosses the node.
            let s = f64::from(1u32 << key.level);
            let y = key.origin()[1] as f32;
            let sea = (p.sea_level + 1) as f32;
            let low = (y + (0.5 * s) as f32) < sea;
            let high = (y + (15.5 * s) as f32) < sea;
            if low == high {
                NodeClass::Air
            } else {
                NodeClass::Mixed
            }
        }
    }
}

/// The node of a classified key, filled on the CPU (`None` for
/// [`Class::Mixed`]).
pub fn fill_class(
    p: &CubeGenParams,
    key: CubeKey,
    columns: &NodeColumns,
    class: Class,
) -> Option<CubeNode> {
    let s = f64::from(1u32 << key.level);
    let origin_y = key.origin()[1] as f32;
    let state = |x: usize, y: usize, z: usize| match class {
        Class::Empty => {
            let yc = origin_y + ((y as f64 + 0.5) * s) as f32;
            if yc < (p.sea_level + 1) as f32 {
                p.water
            } else {
                p.air
            }
        }
        _ => rock_block(p, key, x, y as i32, z),
    };
    if class == Class::Mixed {
        return None;
    }
    Some(CubeNode::from_fn(key, |x, y, z| Voxel {
        state: BlockStateId(state(x, y, z)),
        biome: columns.biome[z * 16 + x],
    }))
}

/// The node when the 2D pre-pass settles it, else `None`: it needs the 3D pass.
pub fn prepass_node(p: &CubeGenParams, key: CubeKey, columns: &NodeColumns) -> Option<CubeNode> {
    fill_class(p, key, columns, classify(p, key, columns))
}

// ----------------------------------------------------------- CPU reference

fn fade(v: f32) -> f32 {
    v * v * v * (v * (v * 6.0 - 15.0) + 10.0)
}
fn lerp(d: f32, a: f32, b: f32) -> f32 {
    a + d * (b - a)
}

const GRADIENTS: [[f32; 3]; 16] = [
    [1.0, 1.0, 0.0],
    [-1.0, 1.0, 0.0],
    [1.0, -1.0, 0.0],
    [-1.0, -1.0, 0.0],
    [1.0, 0.0, 1.0],
    [-1.0, 0.0, 1.0],
    [1.0, 0.0, -1.0],
    [-1.0, 0.0, -1.0],
    [0.0, 1.0, 1.0],
    [0.0, -1.0, 1.0],
    [0.0, 1.0, -1.0],
    [0.0, -1.0, -1.0],
    [1.0, 1.0, 0.0],
    [0.0, -1.0, 1.0],
    [-1.0, 1.0, 0.0],
    [0.0, -1.0, -1.0],
];

/// Vanilla `ImprovedNoise` at a lattice cell and a fraction within it.
pub(crate) fn perlin(perm: &[u8; 256], cell: [i32; 3], f: [f32; 3]) -> f32 {
    let at = |i: i32| i32::from(perm[(i & 255) as usize]);
    let p = at(cell[0]);
    let q = at(cell[0].wrapping_add(1));
    let rows = [
        at(p.wrapping_add(cell[1])),
        at(q.wrapping_add(cell[1])),
        at(p.wrapping_add(cell[1]).wrapping_add(1)),
        at(q.wrapping_add(cell[1]).wrapping_add(1)),
    ];
    let v: [f32; 8] = std::array::from_fn(|i| {
        let g = GRADIENTS[(at(rows[i & 3]
            .wrapping_add(cell[2])
            .wrapping_add((i >> 2) as i32))
            & 15) as usize];
        g[0] * (f[0] - (i & 1) as f32)
            + g[1] * (f[1] - ((i >> 1) & 1) as f32)
            + g[2] * (f[2] - ((i >> 2) & 1) as f32)
    });
    let (a, b, c) = (fade(f[0]), fade(f[1]), fade(f[2]));
    let lerp2 = |v: &[f32]| lerp(b, lerp(a, v[0], v[1]), lerp(a, v[2], v[3]));
    lerp(c, lerp2(&v[..4]), lerp2(&v[4..]))
}

/// A noise at a world position, all in `f64` until the lattice fraction.
fn noise_reference(p: &CubeGenParams, n: usize, used: usize, pos: [f64; 3]) -> f32 {
    let noise = &p.noises[n];
    let mut sum = 0.0f32;
    for (k, (scale, weight)) in octave_terms(noise, n == DEPTH)
        .into_iter()
        .enumerate()
        .take(used)
    {
        let off = noise.octaves[k].offsets;
        let mut cell = [0i32; 3];
        let mut frac = [0f32; 3];
        for a in 0..3 {
            let shifted = wrap(pos[a] * scale[a]) + off[a];
            let c = shifted.floor();
            cell[a] = c as i32;
            frac[a] = (shifted - c) as f32;
        }
        sum += weight * perlin(&noise.octaves[k].permutations, cell, frac);
    }
    sum + noise.offset as f32
}

fn hash_voxel(v: [i32; 3]) -> f32 {
    let mut h = (v[0] as u32).wrapping_mul(0x9E37_79B1)
        ^ (v[1] as u32).wrapping_mul(0x85EB_CA6B)
        ^ (v[2] as u32).wrapping_mul(0xC2B2_AE35);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2C1B_3C6D);
    h ^= h >> 12;
    h = h.wrapping_mul(0x297A_2D39);
    h ^= h >> 15;
    (h >> 8) as f32 / 16_777_216.0
}

fn sign_of(v: f32) -> f32 {
    if v > 0.0 {
        1.0
    } else if v < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// The block state ids of a node, `y << 8 | z << 4 | x`, from the density at
/// the voxel centres in `f64` positions (the shader's replacers).
fn reference_states(p: &CubeGenParams, key: CubeKey, columns: &NodeColumns) -> Vec<u32> {
    let level = key.level;
    let s = f64::from(1u32 << level);
    let sf = s as f32;
    let origin = key.origin();
    let used: [usize; 4] = std::array::from_fn(|n| used_octaves(p, n, level));
    let extra = extra_layers(p, level) as i32;
    let mut out = vec![p.air; VOLUME];
    for z in 0..SIDE {
        for x in 0..SIDE {
            let col = z * 16 + x;
            let wx = f64::from(origin[0]) + (x as f64 + 0.5) * s;
            let wz = f64::from(origin[2]) + (z as f64 + 0.5) * s;
            let biome = (columns.biome[col] as usize).min(p.biomes.len().max(1) - 1);
            let blocks = p.biomes.get(biome).copied().unwrap_or(BiomeBlocks {
                top: p.stone,
                filler: p.stone,
            });
            let mut depth = noise_reference(p, DEPTH, used[DEPTH], [wx, 0.0, wz]);
            if depth < 0.0 {
                depth *= -0.3;
            }
            depth = (depth * 3.0 - 2.0).clamp(-2.0, 1.0);
            depth = if depth < 0.0 {
                depth / 5.6
            } else {
                depth / 8.0
            };
            depth *= 0.2 * 17.0 / 64.0;
            let height = columns.height[col] * p.height_factor as f32 + p.height_offset as f32;
            let mut run = 0u32;
            let mut seen_air = false;
            for jj in (0..16 + extra).rev() {
                let ly = (f64::from(jj) + 0.5) * s;
                let wy = f64::from(origin[1]) + ly;
                let yc = origin[1] as f32 + ly as f32;
                let pos = [wx, wy, wz];
                let selector = noise_reference(p, SELECTOR, used[SELECTOR], pos).clamp(0.0, 1.0);
                let low = noise_reference(p, LOW, used[LOW], pos);
                let high = noise_reference(p, HIGH, used[HIGH], pos);
                let mixed = low + selector * (high - low);
                let below = if height > yc {
                    p.below_average_factor as f32
                } else {
                    1.0
                };
                let vol = columns.volatility[col] * below * p.height_variation_factor as f32
                    + p.height_variation_offset as f32;
                let density = (mixed + depth) * vol + height - sign_of(vol) * yc;
                let block = if density <= 0.0 {
                    run = 0;
                    seen_air = true;
                    if yc < (p.sea_level + 1) as f32 {
                        p.water
                    } else {
                        p.air
                    }
                } else {
                    let surface = if seen_air { run } else { 1_000_000 };
                    run += 1;
                    if surface == 0 {
                        blocks.top
                    } else if (surface as f32) * sf < p.filler_depth as f32 {
                        blocks.filler
                    } else {
                        rock_block(p, key, x, jj, z)
                    }
                };
                if jj < 16 {
                    out[(jj as usize) << 8 | z << 4 | x] = block;
                }
            }
        }
    }
    out
}

fn build_node(key: CubeKey, states: &[u32], columns: &NodeColumns) -> CubeNode {
    CubeNode::from_fn(key, |x, y, z| Voxel {
        state: BlockStateId(states[CubeNode::index(x, y, z)]),
        biome: columns.biome[z * 16 + x],
    })
}

/// The node on the CPU, with every octave's input computed in `f64` at the
/// voxel centre (no origin split): the reference the GPU result is checked
/// against, and the fallback without compute shaders.
pub fn cpu_reference_node(p: &CubeGenParams, key: CubeKey, columns: &NodeColumns) -> CubeNode {
    build_node(key, &reference_states(p, key, columns), columns)
}

// -------------------------------------------------------------------- GPU

/// Words of one pool slot: [`SLOT_STATES`] voxel states, then 256 biomes.
pub const SLOT_WORDS: usize = VOLUME + 256;
/// Words of the voxel states at the start of a slot.
pub const SLOT_STATES: usize = VOLUME;
/// Bytes of one pool slot.
pub const SLOT_BYTES: u64 = (SLOT_WORDS * 4) as u64;

/// A persistent GPU storage buffer of fixed-size node slots, with a free list.
///
/// Slot `s` occupies words `s * SLOT_WORDS .. (s + 1) * SLOT_WORDS` of
/// [`NodePool::buffer`] (an `array<u32>` storage buffer, usage `STORAGE |
/// COPY_SRC`), laid out as:
///
/// * words `0..4096`: one `u32` block state id per voxel (palette-free: the
///   registry `BlockStateId.0`), index `y << 8 | z << 4 | x` with `x`, `y`,
///   `z` in `0..16` the voxel position in the node (the order of
///   `CubeNode::index`);
/// * words `4096..4352`: the biome id of each voxel column, index `z << 4 | x`
///   (the low 16 bits; the caller's `NodeColumns::biome`).
///
/// A slot is valid from the submission that generated it (queue order makes
/// later GPU passes see it) until [`NodePool::free`]; freeing makes the slot
/// reusable by a later `generate_to_pool`, so a GPU pass reading a slot must
/// be submitted before the slot is freed and regenerated.
pub struct NodePool {
    buffer: wgpu::Buffer,
    free: Arc<Mutex<Vec<u32>>>,
    capacity: u32,
}

/// A shared view of a [`NodePool`]: the buffer and the free list. The GPU
/// mesher (`gfx::cube_mesh`) and the renderer hold one to read slots, upload
/// CPU-sampled nodes into slots and free slots, while the generator thread
/// allocates from the same list.
#[derive(Clone)]
pub struct PoolHandle {
    buffer: wgpu::Buffer,
    free: Arc<Mutex<Vec<u32>>>,
    capacity: u32,
}

impl PoolHandle {
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
    pub fn capacity(&self) -> u32 {
        self.capacity
    }
    pub fn free_slots(&self) -> usize {
        self.free.lock().unwrap().len()
    }
    /// A free slot, or `None` when the pool is full.
    pub fn alloc(&self) -> Option<u32> {
        self.free.lock().unwrap().pop()
    }
    /// Returns `slot` to the pool.
    pub fn free(&self, slot: u32) {
        let mut free = self.free.lock().unwrap();
        debug_assert!(slot < self.capacity && !free.contains(&slot));
        free.push(slot);
    }
    /// Writes a node's [`SLOT_WORDS`] words into `slot` (queue order: later
    /// submissions see it).
    pub fn write(&self, queue: &wgpu::Queue, slot: u32, words: &[u32]) {
        debug_assert_eq!(words.len(), SLOT_WORDS);
        queue.write_buffer(
            &self.buffer,
            SLOT_BYTES * u64::from(slot),
            bytemuck::cast_slice(words),
        );
    }
}

/// A node's slot words (the layout above): states, then the biome of each
/// column (taken from the column's highest voxel that is not `air`, else its
/// lowest: a CPU node holds a biome per voxel, a slot one per column).
pub fn slot_words(node: &CubeNode, air: u32) -> Vec<u32> {
    let mut words = vec![0u32; SLOT_WORDS];
    for i in 0..VOLUME {
        words[i] = node.at(i).state.0;
    }
    for z in 0..SIDE {
        for x in 0..SIDE {
            let biome = (0..SIDE)
                .rev()
                .map(|y| node.get(x, y, z))
                .find(|v| v.state.0 != air)
                .unwrap_or_else(|| node.get(x, 0, z))
                .biome;
            words[SLOT_STATES + (z << 4 | x)] = u32::from(biome);
        }
    }
    words
}

impl NodePool {
    fn new(device: &wgpu::Device, capacity: u32) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cube node pool"),
            size: SLOT_BYTES * u64::from(capacity),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            buffer,
            free: Arc::new(Mutex::new((0..capacity).rev().collect())),
            capacity,
        }
    }
    /// The storage buffer holding every slot.
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
    /// A handle other threads and the mesher keep.
    pub fn handle(&self) -> PoolHandle {
        PoolHandle {
            buffer: self.buffer.clone(),
            free: self.free.clone(),
            capacity: self.capacity,
        }
    }
    pub fn capacity(&self) -> u32 {
        self.capacity
    }
    pub fn free_slots(&self) -> usize {
        self.free.lock().unwrap().len()
    }
    /// A free slot, or `None` when the pool is full.
    pub fn alloc(&self) -> Option<u32> {
        self.free.lock().unwrap().pop()
    }
    /// Returns `slot` to the pool.
    pub fn free(&self, slot: u32) {
        let mut free = self.free.lock().unwrap();
        debug_assert!(slot < self.capacity && !free.contains(&slot));
        free.push(slot);
    }
}

/// The compute pipeline, the node pool and the staging buffer.
pub struct CubeGen {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    headers: wgpu::Buffer,
    heights: wgpu::Buffer,
    biomes: wgpu::Buffer,
    pool: NodePool,
    staging: wgpu::Buffer,
    params: Arc<CubeGenParams>,
}

/// Slots of the pool the sampler makes (about 17 MB per 1000).
pub const DEFAULT_POOL_SLOTS: u32 = 2048;

impl CubeGen {
    /// The generator on this device with a pool of `slots` nodes, or `None` if
    /// it cannot run the shader (no compute shaders, too few storage buffers,
    /// or a pool beyond the device's storage binding size).
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        params: Arc<CubeGenParams>,
        slots: u32,
    ) -> Option<Self> {
        let limit = device.limits().max_storage_buffer_binding_size;
        if SLOT_BYTES * u64::from(slots) > limit {
            return None;
        }
        let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Cube generation"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SHADER)),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Cube generation"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let layout = pipeline.get_bind_group_layout(0);
        let buffer = |label, size: u64, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let (gpu_params, perms) = GpuParams::new(&params);
        let params_buf = buffer(
            "Cube params",
            std::mem::size_of::<GpuParams>() as u64,
            storage,
        );
        let perms_buf = buffer("Cube permutations", (perms.len() * 4) as u64, storage);
        let headers = buffer(
            "Cube headers",
            (std::mem::size_of::<GpuHeader>() * MAX_BATCH) as u64,
            storage,
        );
        let heights = buffer("Cube columns", (MAX_BATCH * 512 * 4) as u64, storage);
        let biomes = buffer("Cube biomes", (MAX_BATCH * 256 * 4) as u64, storage);
        let pool = NodePool::new(&device, slots);
        let staging = buffer(
            "Cube readback",
            SLOT_BYTES * MAX_BATCH as u64,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        queue.write_buffer(&params_buf, 0, bytemuck::bytes_of(&gpu_params));
        queue.write_buffer(&perms_buf, 0, bytemuck::cast_slice(&perms));
        fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
            wgpu::BindGroupEntry {
                binding,
                resource: buffer.as_entire_binding(),
            }
        }
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Cube generation"),
            layout: &layout,
            entries: &[
                entry(0, &params_buf),
                entry(1, &perms_buf),
                entry(2, &headers),
                entry(3, &heights),
                entry(4, &biomes),
                entry(5, pool.buffer()),
            ],
        });
        let internal = pollster::block_on(internal.pop());
        if let Some(e) = internal.or(pollster::block_on(validation.pop())) {
            log::warn!("Horizon renderer: far terrain generated on the CPU ({e})");
            return None;
        }
        Some(Self {
            device,
            queue,
            pipeline,
            bind_group,
            headers,
            heights,
            biomes,
            pool,
            staging,
            params,
        })
    }

    /// The node pool: the buffer to bind in a GPU mesher, and `free`.
    pub fn pool(&self) -> &NodePool {
        &self.pool
    }
    pub fn pool_mut(&mut self) -> &mut NodePool {
        &mut self.pool
    }
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Generates the nodes of `jobs` (at most [`MAX_BATCH`]) into newly
    /// allocated pool slots, one per job in order, with one submission and no
    /// readback. `None` (nothing allocated) when the pool lacks free slots.
    pub fn generate_to_pool(&mut self, jobs: &[(CubeKey, &NodeColumns)]) -> Option<Vec<u32>> {
        let n = jobs.len();
        assert!(n <= MAX_BATCH);
        if n == 0 {
            return Some(Vec::new());
        }
        if self.pool.free_slots() < n {
            return None;
        }
        let slots: Vec<u32> = (0..n).map(|_| self.pool.alloc().unwrap()).collect();
        let headers: Vec<GpuHeader> = jobs
            .iter()
            .zip(&slots)
            .map(|((key, _), &slot)| GpuHeader::new(&self.params, *key, slot))
            .collect();
        let mut hv = Vec::with_capacity(n * 512);
        let mut biomes = Vec::with_capacity(n * 256);
        for (_, c) in jobs {
            hv.extend_from_slice(&c.height);
            hv.extend_from_slice(&c.volatility);
            biomes.extend(c.biome.iter().map(|&b| u32::from(b)));
        }
        self.queue
            .write_buffer(&self.headers, 0, bytemuck::cast_slice(&headers));
        self.queue
            .write_buffer(&self.heights, 0, bytemuck::cast_slice(&hv));
        self.queue
            .write_buffer(&self.biomes, 0, bytemuck::cast_slice(&biomes));
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Cube generation"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Cube generation"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(n as u32, 1, 1);
        }
        self.queue.submit([encoder.finish()]);
        Some(slots)
    }

    /// [`Self::generate_to_pool`] with the columns made on the GPU: `fill`
    /// records the pass that writes them into the [`ColumnTargets`] (for the
    /// `keys`, in order) into the encoder, ahead of the generation pass. Nothing
    /// is uploaded. Same slots, same `None` when the pool is short. `fill`
    /// must not submit; one call per submission if it writes buffers with the
    /// queue.
    pub fn generate_to_pool_gpu(
        &mut self,
        keys: &[CubeKey],
        fill: impl FnOnce(&mut wgpu::CommandEncoder, ColumnTargets<'_>),
    ) -> Option<Vec<u32>> {
        let n = keys.len();
        assert!(n <= MAX_BATCH);
        if n == 0 {
            return Some(Vec::new());
        }
        if self.pool.free_slots() < n {
            return None;
        }
        let slots: Vec<u32> = (0..n).map(|_| self.pool.alloc().unwrap()).collect();
        let headers: Vec<GpuHeader> = keys
            .iter()
            .zip(&slots)
            .map(|(key, &slot)| GpuHeader::new(&self.params, *key, slot))
            .collect();
        self.queue
            .write_buffer(&self.headers, 0, bytemuck::cast_slice(&headers));
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Cube generation (GPU columns)"),
            });
        fill(
            &mut encoder,
            ColumnTargets {
                columns: &self.heights,
                biomes: &self.biomes,
            },
        );
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Cube generation"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(n as u32, 1, 1);
        }
        self.queue.submit([encoder.finish()]);
        Some(slots)
    }

    /// The nodes of `keys` (at most [`MAX_BATCH`]) with GPU-made columns,
    /// read back and freed again (biomes come from the slots); `None` on
    /// failure.
    pub fn generate_gpu_columns(
        &mut self,
        keys: &[CubeKey],
        fill: impl FnOnce(&mut wgpu::CommandEncoder, ColumnTargets<'_>),
    ) -> Option<Vec<CubeNode>> {
        let slots = self.generate_to_pool_gpu(keys, fill)?;
        let words = self.read_back(&slots);
        for &slot in &slots {
            self.pool.free(slot);
        }
        let words = words?;
        Some(
            keys.iter()
                .enumerate()
                .map(|(i, key)| {
                    let at = i * SLOT_WORDS;
                    let slot = &words[at..at + SLOT_WORDS];
                    CubeNode::from_fn(*key, |x, y, z| Voxel {
                        state: BlockStateId(slot[CubeNode::index(x, y, z)]),
                        biome: slot[SLOT_STATES + z * 16 + x] as u16,
                    })
                })
                .collect(),
        )
    }

    /// Copies `slots` (at most [`MAX_BATCH`]) to the staging buffer and reads
    /// them: [`SLOT_WORDS`] words each, in order. `None` if the readback failed.
    pub fn read_back(&mut self, slots: &[u32]) -> Option<Vec<u32>> {
        assert!(slots.len() <= MAX_BATCH);
        if slots.is_empty() {
            return Some(Vec::new());
        }
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Cube readback"),
            });
        for (i, &slot) in slots.iter().enumerate() {
            encoder.copy_buffer_to_buffer(
                self.pool.buffer(),
                SLOT_BYTES * u64::from(slot),
                &self.staging,
                SLOT_BYTES * i as u64,
                SLOT_BYTES,
            );
        }
        let bytes = SLOT_BYTES * slots.len() as u64;
        let submission = self.queue.submit([encoder.finish()]);
        let (tx, rx) = mpsc::channel();
        self.staging
            .slice(..bytes)
            .map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
        let polled = self.device.poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: Some(Duration::from_secs(30)),
        });
        if polled.is_err() || !matches!(rx.recv_timeout(Duration::from_secs(30)), Ok(Ok(()))) {
            return None;
        }
        let words = {
            let view = self.staging.slice(..bytes).get_mapped_range().ok()?;
            bytemuck::cast_slice::<u8, u32>(&view).to_vec()
        };
        self.staging.unmap();
        Some(words)
    }

    /// The nodes of `jobs` (at most [`MAX_BATCH`]), in order, generated into
    /// the pool, read back and freed again; `None` on failure.
    pub fn generate(&mut self, jobs: &[(CubeKey, &NodeColumns)]) -> Option<Vec<CubeNode>> {
        let slots = self.generate_to_pool(jobs)?;
        let words = self.read_back(&slots);
        for &slot in &slots {
            self.pool.free(slot);
        }
        let words = words?;
        Some(
            jobs.iter()
                .enumerate()
                .map(|(i, (key, columns))| {
                    let at = i * SLOT_WORDS;
                    build_node(*key, &words[at..at + SLOT_STATES], columns)
                })
                .collect(),
        )
    }

    /// The nodes of any number of jobs, in order, through the 3D pass only
    /// (chunks of at most [`MAX_BATCH`], fewer when the pool is short); `None`
    /// for a node whose chunk failed.
    pub fn generate_mixed(&mut self, jobs: &[(CubeKey, &NodeColumns)]) -> Vec<Option<CubeNode>> {
        let mut out = Vec::with_capacity(jobs.len());
        let mut at = 0;
        while at < jobs.len() {
            let size = (jobs.len() - at)
                .min(MAX_BATCH)
                .min(self.pool.free_slots().max(1));
            let chunk = &jobs[at..at + size];
            match self.generate(chunk) {
                Some(nodes) => out.extend(nodes.into_iter().map(Some)),
                None => out.extend(chunk.iter().map(|_| None)),
            }
            at += size;
        }
        out
    }

    /// The jobs through the 3D pass only (chunks of at most [`MAX_BATCH`],
    /// fewer when the pool is short), in order. A job with `slot[i]` set
    /// stays in the pool and answers [`Resident::Slot`] (its slot is the
    /// caller's to free); the others are read back and freed, as
    /// [`CubeGen::generate_mixed`] does. `None` for a chunk that failed.
    pub fn generate_resident(
        &mut self,
        jobs: &[(CubeKey, &NodeColumns)],
        slot: &[bool],
    ) -> Vec<Option<Resident>> {
        debug_assert_eq!(jobs.len(), slot.len());
        let mut out = Vec::with_capacity(jobs.len());
        let mut at = 0;
        while at < jobs.len() {
            let size = (jobs.len() - at)
                .min(MAX_BATCH)
                .min(self.pool.free_slots().max(1));
            let chunk = &jobs[at..at + size];
            let keep = &slot[at..at + size];
            match self.generate_to_pool(chunk) {
                None => out.extend(chunk.iter().map(|_| None)),
                Some(slots) => {
                    let wanted: Vec<usize> = (0..size).filter(|&i| !keep[i]).collect();
                    let read: Vec<u32> = wanted.iter().map(|&i| slots[i]).collect();
                    let words = self.read_back(&read);
                    for &s in &read {
                        self.pool.free(s);
                    }
                    for i in 0..size {
                        out.push(if keep[i] {
                            Some(Resident::Slot(slots[i]))
                        } else {
                            let w = wanted.iter().position(|&j| j == i).unwrap();
                            words.as_ref().map(|words| {
                                let base = w * SLOT_WORDS;
                                Resident::Node(build_node(
                                    chunk[i].0,
                                    &words[base..base + SLOT_STATES],
                                    chunk[i].1,
                                ))
                            })
                        });
                    }
                }
            }
            at += size;
        }
        out
    }

    /// [`CubeGen::generate_resident`] with the columns made on the GPU by
    /// `biomes` in the same submission as each chunk's generation: no CPU
    /// columns, no pre-pass. Nodes read back take their biomes from the slot.
    pub fn generate_resident_gpu(
        &mut self,
        keys: &[CubeKey],
        slot: &[bool],
        biomes: &BiomeGen,
    ) -> Vec<Option<Resident>> {
        debug_assert_eq!(keys.len(), slot.len());
        let mut out = Vec::with_capacity(keys.len());
        let mut at = 0;
        while at < keys.len() {
            let size = (keys.len() - at)
                .min(MAX_BATCH)
                .min(self.pool.free_slots().max(1));
            let chunk = &keys[at..at + size];
            let keep = &slot[at..at + size];
            match self.generate_to_pool_gpu(chunk, |encoder, targets| {
                biomes.encode(encoder, chunk, targets)
            }) {
                None => out.extend(chunk.iter().map(|_| None)),
                Some(slots) => {
                    let wanted: Vec<usize> = (0..size).filter(|&i| !keep[i]).collect();
                    let read: Vec<u32> = wanted.iter().map(|&i| slots[i]).collect();
                    let words = self.read_back(&read);
                    for &s in &read {
                        self.pool.free(s);
                    }
                    for i in 0..size {
                        out.push(if keep[i] {
                            Some(Resident::Slot(slots[i]))
                        } else {
                            let w = wanted.iter().position(|&j| j == i).unwrap();
                            words.as_ref().map(|words| {
                                let node = &words[w * SLOT_WORDS..(w + 1) * SLOT_WORDS];
                                Resident::Node(CubeNode::from_fn(chunk[i], |x, y, z| Voxel {
                                    state: BlockStateId(node[CubeNode::index(x, y, z)]),
                                    biome: node[SLOT_STATES + z * 16 + x] as u16,
                                }))
                            })
                        });
                    }
                }
            }
            at += size;
        }
        out
    }

    /// Any number of nodes: the 2D pre-pass settles what it can on the CPU
    /// (uniform air, water or rock, with no 3D noise run), and the rest go to
    /// the GPU in as few dispatches as the buffers allow. `Vec` in job order.
    pub fn generate_bulk(&mut self, jobs: &[(CubeKey, &NodeColumns)]) -> Vec<Option<CubeNode>> {
        let mut out: Vec<Option<CubeNode>> = Vec::with_capacity(jobs.len());
        let mut rest = Vec::new();
        for (i, (key, columns)) in jobs.iter().enumerate() {
            let node = prepass_node(&self.params, *key, columns);
            if node.is_none() {
                rest.push(i);
            }
            out.push(node);
        }
        let mixed: Vec<_> = rest.iter().map(|&i| jobs[i]).collect();
        for (i, node) in rest.into_iter().zip(self.generate_mixed(&mixed)) {
            out[i] = node;
        }
        out
    }
}

/// A device of its own for generation, if the machine has one that runs
/// compute shaders.
pub fn request_own_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    if cfg!(windows) {
        descriptor.backends = wgpu::Backends::PRIMARY;
    }
    let instance = wgpu::Instance::new(descriptor.with_env());
    let adapter = pollster::block_on(instance.request_adapter(&Default::default())).ok()?;
    pollster::block_on(adapter.request_device(&Default::default())).ok()
}

// ---------------------------------------------------------------- sampler

/// Jobs from one caller (the 3D pass only: the pre-pass ran on the caller).
struct Message {
    jobs: Vec<(CubeKey, NodeColumns)>,
    /// Whether the nodes stay in the pool ([`Resident::Slot`]) or are read back.
    slots: bool,
    reply: mpsc::Sender<Vec<Option<Resident>>>,
}

/// What the GPU thread is asked for.
enum Work {
    /// Nodes with their columns made by the caller.
    Jobs(Message),
    /// Nodes whose columns the GPU thread makes (`BiomeGen`).
    Keys(KeysMessage),
    /// The biome pass, once the sampler built it.
    Biomes(Box<BiomeGen>),
}

/// Keys from one caller; the GPU thread makes the columns.
struct KeysMessage {
    keys: Vec<CubeKey>,
    slots: bool,
    reply: mpsc::Sender<Vec<Option<Resident>>>,
}

/// What a sampler on the GPU gives the GPU mesher: the device it generates on,
/// the pool its nodes stay in, and the block states it can write.
#[derive(Clone)]
pub struct GpuBackend {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub pool: PoolHandle,
    /// Every block state id the generator writes (air, rock, water, and each
    /// biome's top and filler).
    pub states: Vec<u32>,
    /// What the generator makes the nodes from, for other far-terrain passes
    /// on the same device (`gfx::surface_tile`).
    pub params: Arc<CubeGenParams>,
    /// The GPU biome pass's inputs, once `GpuCubeSampler::with_biomes` set it.
    pub biome_params: Option<Arc<BiomeGenParams>>,
}

/// A [`CubeSampler`] generating levels [`MIN_LEVEL`] and up on the GPU. Every
/// handle (`fork`) shares one GPU thread that merges the queued requests.
/// The caller's thread makes the node's columns and runs the 2D pre-pass; only
/// nodes the surface may cross travel to the GPU thread.
pub struct GpuCubeSampler {
    params: Arc<CubeGenParams>,
    columns: ColumnFn,
    gpu: Option<mpsc::Sender<Work>>,
    backend: Option<GpuBackend>,
    /// The GPU descent (`gfx::cube_descent`), once `with_descent` made it.
    descent: Option<Arc<GpuDescent>>,
    /// Columns are made by the GPU thread (`with_biomes`): requests skip the
    /// CPU columns and the CPU pre-pass.
    gpu_columns: bool,
    reply: mpsc::Sender<Vec<Option<Resident>>>,
    replies: mpsc::Receiver<Vec<Option<Resident>>>,
}

impl GpuCubeSampler {
    /// A sampler on `device`/`queue`; without compute shaders there, it
    /// generates on the CPU ([`cpu_reference_node`]) in the calling threads.
    /// Give it the renderer's device for [`GpuCubeSampler::backend`] to be
    /// usable by the renderer's mesher (`gfx::cube_mesh`).
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        params: CubeGenParams,
        columns: ColumnFn,
    ) -> Self {
        let params = Arc::new(params);
        let mut backend = None;
        let gpu = CubeGen::new(
            device.clone(),
            queue.clone(),
            params.clone(),
            DEFAULT_POOL_SLOTS,
        )
        .map(|generator| {
            let mut states = vec![params.air, params.stone, params.deepslate, params.water];
            for b in &params.biomes {
                states.extend([b.top, b.filler]);
            }
            states.sort_unstable();
            states.dedup();
            backend = Some(GpuBackend {
                device,
                queue,
                pool: generator.pool.handle(),
                states,
                params: params.clone(),
                biome_params: None,
            });
            let (tx, rx) = mpsc::channel::<Work>();
            std::thread::Builder::new()
                .name("cube-gen-gpu".into())
                .spawn(move || gpu_thread(generator, rx))
                .expect("spawn the cube generation thread");
            tx
        });
        let mut sampler = Self::with(params, columns, gpu);
        sampler.backend = backend;
        sampler
    }

    /// A sampler on a device of its own, or on the CPU if there is none.
    pub fn with_own_device(params: CubeGenParams, columns: ColumnFn) -> Self {
        match request_own_device() {
            Some((device, queue)) => Self::new(device, queue, params, columns),
            None => Self::cpu(params, columns),
        }
    }

    /// A sampler that always generates on the CPU.
    pub fn cpu(params: CubeGenParams, columns: ColumnFn) -> Self {
        Self::with(Arc::new(params), columns, None)
    }

    fn with(
        params: Arc<CubeGenParams>,
        columns: ColumnFn,
        gpu: Option<mpsc::Sender<Work>>,
    ) -> Self {
        let (reply, replies) = mpsc::channel();
        Self {
            params,
            columns,
            gpu,
            backend: None,
            descent: None,
            gpu_columns: false,
            reply,
            replies,
        }
    }

    /// Adds the GPU descent (`gfx::cube_descent`) on the sampler's device,
    /// from the generator's density bound (`CustomCubicGenerator::gpu_terms`).
    /// Without a GPU generator, or when the terms do not match the
    /// parameters, the sampler is returned as it was. `CubeLod` uses the
    /// descent when asked (`CubeLod::set_gpu_descent`).
    pub fn with_descent(mut self, terms: &horizon_world::cubic::custom::GpuTerms) -> Self {
        if let Some(backend) = &self.backend {
            self.descent = GpuDescent::new(
                backend.device.clone(),
                backend.queue.clone(),
                &self.params,
                terms,
            )
            .map(Arc::new);
        }
        self
    }

    /// Makes the columns of the nodes `request_resident` takes on the GPU
    /// thread, in the same submission as their generation (`BiomeGen`), so a
    /// far node costs the calling worker no CPU work beyond bookkeeping. The
    /// 2D pre-pass is skipped with them (the GPU descent, `with_descent`, is
    /// what keeps uniform nodes from being requested). `classify` keeps using
    /// the CPU columns. Without a GPU generator, or when the device cannot
    /// run the biome pass, the sampler is returned as it was (CPU columns).
    pub fn with_biomes(mut self, params: BiomeGenParams) -> Self {
        let (Some(backend), Some(gpu)) = (&self.backend, &self.gpu) else {
            return self;
        };
        let params = Arc::new(params);
        let Some(biomes) = BiomeGen::new(
            backend.device.clone(),
            backend.queue.clone(),
            params.clone(),
        ) else {
            return self;
        };
        if gpu.send(Work::Biomes(Box::new(biomes))).is_ok() {
            self.gpu_columns = true;
            if let Some(backend) = &mut self.backend {
                backend.biome_params = Some(params);
            }
        }
        self
    }

    /// Whether the columns of requested nodes are made on the GPU.
    pub fn gpu_columns(&self) -> bool {
        self.gpu_columns
    }

    /// Whether requests go to the GPU.
    pub fn on_gpu(&self) -> bool {
        self.gpu.is_some()
    }

    /// The device, pool and block states of the GPU generator, if it runs.
    pub fn backend(&self) -> Option<&GpuBackend> {
        self.backend.as_ref()
    }

    /// The nodes at `keys` in one request: columns and the 2D pre-pass run
    /// here, in the caller, and every node left goes to the GPU thread in one
    /// message, which batches it with whatever else is queued into as few
    /// dispatches as the pool and buffers allow. Levels below [`MIN_LEVEL`]
    /// and failures are `None`.
    pub fn request(&mut self, keys: &[CubeKey]) -> Vec<Option<CubeNode>> {
        self.request_resident(keys, false)
            .into_iter()
            .map(|r| match r {
                Some(Resident::Node(node)) => Some(node),
                _ => None,
            })
            .collect()
    }

    /// As [`GpuCubeSampler::request`], but with `slots` the nodes the 3D pass
    /// made stay in the GPU pool and answer [`Resident::Slot`], never read
    /// back; the caller owns the slot until it frees it (`PoolHandle::free`).
    /// Nodes the pre-pass settled are [`Resident::Node`] (uniform, no slot).
    pub fn request_resident(&mut self, keys: &[CubeKey], slots: bool) -> Vec<Option<Resident>> {
        let mut out: Vec<Option<Resident>> = (0..keys.len()).map(|_| None).collect();
        if self.gpu_columns
            && let Some(gpu) = &self.gpu
        {
            let places: Vec<usize> = (0..keys.len())
                .filter(|&i| keys[i].level >= MIN_LEVEL)
                .collect();
            if places.is_empty() {
                return out;
            }
            let sent = gpu.send(Work::Keys(KeysMessage {
                keys: places.iter().map(|&i| keys[i]).collect(),
                slots,
                reply: self.reply.clone(),
            }));
            if sent.is_err() {
                return out;
            }
            for (i, node) in places
                .into_iter()
                .zip(self.replies.recv().unwrap_or_default())
            {
                out[i] = node;
            }
            return out;
        }
        let mut jobs = Vec::new();
        let mut places = Vec::new();
        for (i, &key) in keys.iter().enumerate() {
            if key.level < MIN_LEVEL {
                continue;
            }
            let columns = (self.columns)(key);
            match prepass_node(&self.params, key, &columns) {
                Some(node) => out[i] = Some(Resident::Node(node)),
                None => {
                    places.push(i);
                    jobs.push((key, columns));
                }
            }
        }
        if jobs.is_empty() {
            return out;
        }
        let nodes: Vec<Option<Resident>> = match &self.gpu {
            None => jobs
                .iter()
                .map(|(key, columns)| {
                    Some(Resident::Node(cpu_reference_node(
                        &self.params,
                        *key,
                        columns,
                    )))
                })
                .collect(),
            Some(gpu) => {
                let sent = gpu.send(Work::Jobs(Message {
                    jobs,
                    slots,
                    reply: self.reply.clone(),
                }));
                if sent.is_err() {
                    return out;
                }
                self.replies.recv().unwrap_or_default()
            }
        };
        for (i, node) in places.into_iter().zip(nodes) {
            out[i] = node;
        }
        out
    }
}

/// Merges every queued message (up to [`BATCH_NODES`] nodes) and runs them
/// through the generator; ends when every handle is gone.
fn gpu_thread(mut generator: CubeGen, messages: mpsc::Receiver<Work>) {
    let mut biomes: Option<BiomeGen> = None;
    while let Ok(first) = messages.recv() {
        let (mut jobs_in, mut keys_in) = (Vec::new(), Vec::new());
        let mut nodes = 0;
        let mut work = Some(first);
        while let Some(w) = work.take() {
            match w {
                Work::Biomes(b) => biomes = Some(*b),
                Work::Jobs(m) => {
                    nodes += m.jobs.len();
                    jobs_in.push(m);
                }
                Work::Keys(m) => {
                    nodes += m.keys.len();
                    keys_in.push(m);
                }
            }
            if nodes < BATCH_NODES {
                work = messages.try_recv().ok();
            }
        }
        if !jobs_in.is_empty() {
            let jobs: Vec<(CubeKey, &NodeColumns)> = jobs_in
                .iter()
                .flat_map(|m| m.jobs.iter().map(|(k, c)| (*k, c)))
                .collect();
            let slot: Vec<bool> = jobs_in
                .iter()
                .flat_map(|m| std::iter::repeat_n(m.slots, m.jobs.len()))
                .collect();
            let mut results = generator.generate_resident(&jobs, &slot).into_iter();
            drop(jobs);
            for message in jobs_in {
                let part: Vec<_> = results.by_ref().take(message.jobs.len()).collect();
                let _ = message.reply.send(part);
            }
        }
        if !keys_in.is_empty() {
            let keys: Vec<CubeKey> = keys_in
                .iter()
                .flat_map(|m| m.keys.iter().copied())
                .collect();
            let slot: Vec<bool> = keys_in
                .iter()
                .flat_map(|m| std::iter::repeat_n(m.slots, m.keys.len()))
                .collect();
            let mut results = match &biomes {
                Some(biomes) => generator.generate_resident_gpu(&keys, &slot, biomes),
                None => keys.iter().map(|_| None).collect(),
            }
            .into_iter();
            for message in keys_in {
                let part: Vec<_> = results.by_ref().take(message.keys.len()).collect();
                let _ = message.reply.send(part);
            }
        }
    }
}

impl CubeSampler for GpuCubeSampler {
    fn node(&mut self, key: CubeKey) -> Option<CubeNode> {
        self.request(&[key]).pop().flatten()
    }

    fn resident(&mut self, key: CubeKey) -> Option<Resident> {
        self.request_resident(&[key], true).pop().flatten()
    }

    fn resident_many(&mut self, keys: &[CubeKey]) -> Vec<Option<Resident>> {
        self.request_resident(keys, true)
    }

    fn gpu(&self) -> Option<GpuBackend> {
        self.backend.clone()
    }

    fn classify(&mut self, key: CubeKey) -> NodeClass {
        if key.level < MIN_LEVEL {
            return NodeClass::Mixed;
        }
        let columns = (self.columns)(key);
        node_class(&self.params, key, &columns)
    }

    fn descent(&self) -> Option<Arc<GpuDescent>> {
        self.descent.clone()
    }

    fn fork(&self) -> Option<Box<dyn CubeSampler>> {
        let mut fork = Self::with(self.params.clone(), self.columns.clone(), self.gpu.clone());
        fork.backend = self.backend.clone();
        fork.descent = self.descent.clone();
        fork.gpu_columns = self.gpu_columns;
        Some(Box::new(fork))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use horizon_core::{noise::PerlinNoise, random::LegacyRandom};

    const AIR: u32 = 0;
    const STONE: u32 = 1;
    const DEEPSLATE: u32 = 2;
    const WATER: u32 = 3;
    const GRASS: u32 = 4;
    const DIRT: u32 = 5;

    fn noise(seed: i64, octaves: usize, factor: f64, offset: f64, f: [f64; 3]) -> NoiseDesc {
        let mut random = LegacyRandom::new(seed);
        NoiseDesc {
            factor,
            offset,
            frequency: f,
            octaves: (0..octaves)
                .map(|_| {
                    let p = PerlinNoise::new(&mut random);
                    OctaveDesc {
                        permutations: *p.table().permutations(),
                        offsets: *p.table().offsets(),
                    }
                })
                .collect(),
        }
    }

    fn params() -> CubeGenParams {
        CubeGenParams {
            noises: [
                noise(1, 8, 1.0, 0.5, [0.004, 0.002, 0.004]),
                noise(2, 16, 1.0, 0.0, [0.002, 0.001, 0.002]),
                noise(3, 16, 1.0, 0.0, [0.002, 0.001, 0.002]),
                noise(4, 16, 1.0, 0.0, [0.003, 0.0, 0.003]),
            ],
            height_factor: 1.0,
            height_offset: 0.0,
            height_variation_factor: 1.0,
            height_variation_offset: 0.0,
            below_average_factor: 4.0,
            sea_level: 63,
            air: AIR,
            stone: STONE,
            deepslate: DEEPSLATE,
            water: WATER,
            biomes: vec![
                BiomeBlocks {
                    top: GRASS,
                    filler: DIRT,
                },
                BiomeBlocks {
                    top: DIRT,
                    filler: DIRT,
                },
            ],
            filler_depth: 4.0,
            deepslate_from: 0.0,
            deepslate_to: 8.0,
        }
    }

    fn columns(key: CubeKey) -> NodeColumns {
        let o = key.origin();
        let s = f64::from(1u32 << key.level);
        let mut c = NodeColumns::flat(64.0, 24.0, 0);
        for z in 0..16 {
            for x in 0..16 {
                let wx = f64::from(o[0]) + (x as f64 + 0.5) * s;
                let wz = f64::from(o[2]) + (z as f64 + 0.5) * s;
                c.height[z * 16 + x] = 64.0 + 20.0 * ((wx * 0.003).sin() as f32);
                c.volatility[z * 16 + x] = 20.0 + 4.0 * ((wz * 0.002).cos() as f32);
                c.biome[z * 16 + x] = ((wx + wz).abs() as u64 / 4096 % 2) as u16;
            }
        }
        c
    }

    fn states(n: &CubeNode) -> Vec<u32> {
        (0..VOLUME).map(|i| n.at(i).state.0).collect()
    }

    #[test]
    fn reference_perlin_is_vanillas() {
        let mut random = LegacyRandom::new(77);
        let vanilla = PerlinNoise::new(&mut random);
        let perm = *vanilla.table().permutations();
        let off = *vanilla.table().offsets();
        for &(x, y, z) in &[(0.3, 0.7, 0.1), (12.5, -3.25, 99.75), (-1234.5, 40.0, 7.77)] {
            let shifted: Vec<f64> = [x, y, z]
                .iter()
                .zip(off)
                .map(|(&v, o)| wrap(v) + o)
                .collect();
            let cell: [i32; 3] = std::array::from_fn(|i| shifted[i].floor() as i32);
            let frac: [f32; 3] = std::array::from_fn(|i| (shifted[i] - f64::from(cell[i])) as f32);
            assert_eq!(
                perlin(&perm, cell, frac).to_bits(),
                vanilla.get3(x, y, z).to_bits()
            );
        }
    }

    #[test]
    fn permutations_pack_four_bytes_a_word() {
        let p = params();
        let (gpu, perms) = GpuParams::new(&p);
        let table = &p.noises[LOW].octaves[3].permutations;
        let slot = LOW * MAX_OCTAVES + 3;
        for i in 0..256 {
            let word = perms[slot * 64 + i / 4];
            assert_eq!((word >> ((i % 4) * 8)) & 255, u32::from(table[i]));
        }
        // Weights of the selector's eight octaves sum to its factor.
        let sum: f32 = (0..8)
            .map(|k| gpu.octaves[SELECTOR * MAX_OCTAVES + k][3])
            .sum();
        assert!((sum - 1.0).abs() < 1e-5);
        // The depth noise has no y frequency.
        assert_eq!(gpu.octaves[DEPTH * MAX_OCTAVES][1], 0.0);
        assert_eq!(gpu.biome_count[0], 2);
    }

    #[test]
    fn header_splits_far_coordinates() {
        let p = params();
        let key = CubeKey::new(2, 78_125, 1, -40_000);
        let h = GpuHeader::new(&p, key, 0);
        assert_eq!(h.info[..3], key.origin());
        let o = &p.noises[LOW].octaves[0];
        let f = p.noises[LOW].frequency;
        let shifted = wrap(f64::from(key.origin()[0]) * f[0]) + o.offsets[0];
        assert_eq!(
            h.cell[LOW * MAX_OCTAVES][0],
            (shifted.floor() as i64 & 255) as i32
        );
        assert!((0.0..1.0).contains(&h.frac[LOW * MAX_OCTAVES][0]));
        // Level 2 voxels are 4 blocks: octaves shorter than that are dropped.
        assert!(h.counts[LOW] < 16 && h.counts[LOW] >= 1);
        assert!(used_octaves(&p, LOW, 3) <= used_octaves(&p, LOW, 2));
    }

    #[test]
    fn shader_validates() {
        let module = naga::front::wgsl::parse_str(SHADER).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
    }

    #[test]
    fn sampler_skips_fine_levels_and_air_above_terrain_is_uniform() {
        let p = params();
        let mut s = GpuCubeSampler::cpu(p, Arc::new(columns));
        assert!(s.node(CubeKey::new(1, 0, 0, 0)).is_none());
        // Far above the highest possible surface.
        let air = s.node(CubeKey::new(3, 0, 40, 0)).unwrap();
        assert_eq!(air.as_uniform().map(|v| v.state.0), Some(AIR));
        let ground = s.node(CubeKey::new(2, 0, 0, 0)).unwrap();
        assert!(ground.as_uniform().is_none());
    }

    #[test]
    fn prepass_agrees_with_the_reference() {
        let p = params();
        let mut seen = [0; 3];
        for level in [2u8, 3, 5] {
            for y in -6..10 {
                for (x, z) in [(0, 0), (3, -2), (78_125 >> (level - 2), 5)] {
                    let key = CubeKey::new(level, x, y, z);
                    let c = columns(key);
                    let class = classify(&p, key, &c);
                    seen[class as usize] += 1;
                    if let Some(node) = fill_class(&p, key, &c, class) {
                        let reference = cpu_reference_node(&p, key, &c);
                        assert_eq!(states(&node), states(&reference), "{key:?} {class:?}");
                    }
                }
            }
        }
        assert!(seen.iter().all(|&n| n > 0), "{seen:?}");
    }

    #[test]
    fn perlin_stays_within_the_bound() {
        let mut random = LegacyRandom::new(5);
        let n = PerlinNoise::new(&mut random);
        let mut max = 0.0f32;
        for i in 0..200_000 {
            let t = f64::from(i);
            max = max.max(n.get3(t * 0.0137, t * 0.0071, t * 0.0199).abs());
        }
        assert!(f64::from(max) <= PERLIN_BOUND, "{max}");
    }

    #[test]
    fn bulk_request_matches_single_nodes() {
        let mut s = GpuCubeSampler::cpu(params(), Arc::new(columns));
        let keys: Vec<_> = (0..6)
            .map(|i| CubeKey::new(2 + (i % 2) as u8, i, i % 3, 0))
            .collect();
        let bulk = s.request(&keys);
        for (key, node) in keys.iter().zip(&bulk) {
            let single = s.node(*key).unwrap();
            assert_eq!(states(node.as_ref().unwrap()), states(&single));
        }
    }

    fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = crate::gfx::test_gpu::instance();
        let adapter = pollster::block_on(instance.request_adapter(&Default::default())).ok()?;
        pollster::block_on(adapter.request_device(&Default::default())).ok()
    }

    #[test]
    fn gpu_matches_the_cpu_reference() {
        let Some((device, queue)) = device() else {
            eprintln!("no GPU adapter: skipped");
            return;
        };
        let params = Arc::new(params());
        let Some(mut generator) = CubeGen::new(device, queue, params.clone(), DEFAULT_POOL_SLOTS)
        else {
            eprintln!("no compute shaders: skipped");
            return;
        };
        let keys = [
            CubeKey::new(2, 0, 0, 0),
            CubeKey::new(2, -1, 0, 3),
            CubeKey::new(3, 1, 0, -2),
            CubeKey::new(2, 78_125, 0, 0),
            CubeKey::new(2, 78_125, 1, -77_000),
            CubeKey::new(4, -9_000, 0, 5_000),
            CubeKey::new(3, 0, 40, 0),
        ];
        let cols: Vec<NodeColumns> = keys.iter().map(|&k| columns(k)).collect();
        let jobs: Vec<_> = keys.iter().copied().zip(cols.iter()).collect();
        let nodes = generator.generate(&jobs).expect("generation");
        for ((key, c), node) in jobs.iter().zip(&nodes) {
            let reference = cpu_reference_node(&params, *key, c);
            let (a, b) = (states(node), states(&reference));
            let differing = a.iter().zip(&b).filter(|(a, b)| a != b).count();
            assert!(
                differing <= 8,
                "{key:?}: {differing} of {VOLUME} voxels differ"
            );
        }
        assert!(nodes[6].as_uniform().is_some());
    }

    #[test]
    fn sampler_on_gpu_answers_from_many_threads() {
        let Some((device, queue)) = device() else {
            return;
        };
        let sampler = GpuCubeSampler::new(device, queue, params(), Arc::new(columns));
        if !sampler.on_gpu() {
            return;
        }
        let threads: Vec<_> = (0..4)
            .map(|t| {
                let mut handle = sampler.fork().unwrap();
                std::thread::spawn(move || {
                    for i in 0..8 {
                        assert!(handle.node(CubeKey::new(2, t, 0, i)).is_some());
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
    }
}
