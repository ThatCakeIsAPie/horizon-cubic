//! Far-terrain columns for cubic worlds, made on the GPU.
//!
//! `cube_gen` generates a far node's voxels from the density; its 2D input,
//! [`NodeColumns`] (per voxel column a height, a volatility and a biome id),
//! used to come from a closure on the CPU, one node at a time. With thousands
//! of nodes in a batch that was the bottleneck. This module makes the columns
//! of a whole batch in one compute pass (`biome_gen.wgsl`), straight into the
//! buffers `cube_gen` reads (`CubeGen::generate_to_pool_gpu`).
//!
//! # What is computed
//!
//! For a voxel column centre (`origin + x * s + s / 2` blocks, `s` the voxel
//! size; x, z only):
//!
//! 1. **Climate** at sea level, as 26.3's noise router has it. A biome sample
//!    reads the climate at its quart's corner (`quart << 2`). Temperature,
//!    vegetation (humidity), continents, erosion and ridges (weirdness) are each
//!    a `NormalNoise` sampled at `(x * 0.25 + shift_x, 0, z * 0.25 + shift_z)`,
//!    where `shift_x` is the `offset` noise at `(x / 4, 0, z / 4)` times 4 and
//!    `shift_z` the same noise at `(z / 4, x / 4, 0)` times 4. A `NormalNoise`
//!    is a [`NoiseStack`](horizon_core::noise::NoiseStack): layers of a plain
//!    Perlin noise, each with a frequency and an amplitude (the normalisation
//!    factor and the value factor are folded into the amplitudes; both Perlin
//!    stacks are separate layers), summed in order. The shader sums the same
//!    layers. Depth is 0: the point is on the surface.
//!    The six values are quantized as `(v * 10000) as int`, like
//!    [`quantize`](horizon_world::worldgen::climate::quantize).
//! 2. **Biome.** The nearest point of the multi-noise parameter list, by
//!    vanilla's distance (per-parameter interval distance squared, summed, plus
//!    the squared offset). The shader runs `Climate.RTree.search` (without a
//!    previous result, so ties go to the first leaf in tree order, which is
//!    [`ClimateIndex::nearest`]) over the tree flattened by
//!    [`ClimateIndex::tree`]: depth first in child order with an explicit
//!    stack, a node entered while the best distance is strictly above its
//!    lower bound. Distances sum in `u32` with saturation (they are `i64`
//!    on the CPU; real targets never come near the limit).
//! 3. **Terrain** per biome: a table (biome id to base height and height
//!    variation, [`BiomeTerrain`]) supplied by the world generator. The
//!    per-biome blocks (top, filler) are `cube_gen`'s own table, by the same
//!    ids.
//!
//! # Blending and its cutover
//!
//! CubicWorldGen blends the table over a 5x5 window of 4-block sections:
//! weight `|10 / sqrt(i^2 + j^2 + 0.2) / (base + 2)|` per neighbour, halved
//! when the neighbour's base is above the centre biome's; the weighted
//! averages are converted with `height = h * 17/64 - 1/256` and
//! `volatility = 2.4 * v + 4/15`, computed at the 4-block grid corners and
//! interpolated bilinearly to the block. Here:
//!
//! * **Voxels of 1, 2, 4 or 8 blocks (levels 0 to [`BLEND_MAX_LEVEL`])** do
//!   that. A workgroup (one node) first fills a grid of the quart biomes it
//!   needs in shared memory (9 x 9 quarts at level 0 up to 36 x 36 at level
//!   3: one climate evaluation per quart, shared by the 25 windows it is in),
//!   then each column blends its two-by-two corners and interpolates at its
//!   centre, `x * s + s / 2` blocks from the origin, so a 1-block voxel is
//!   exactly CubicWorldGen's block.
//! * **Coarser voxels (level 4 and up)**: the blend radius (8 blocks) is
//!   below one voxel, so the column takes its own biome's terrain unblended
//!   (one climate evaluation per column). The reported biome is the column
//!   centre's.
//!
//! # Precision at large coordinates
//!
//! The same scheme as `cube_gen`. For each node and layer the CPU computes in
//! `f64` the layer's input at the node origin, `wrap(origin * xz_scale *
//! frequency) + table offset`, and splits it into the integer lattice cell (kept
//! mod 256, which is all the permutation sees) and an `f32` fraction
//! ([`HEADER_WORDS`] layout). The shader adds the column's offset from the
//! origin, which is small (at most a few hundred blocks, including the
//! shift's tens), in `f32`, and re-splits. The shift noise is evaluated the
//! same way; its value (a few blocks) is then added to the local position.
//! Positions at 5,000,000 blocks therefore keep the precision of positions at
//! 0.
//!
//! Layers whose wavelength (`1 / (xz_scale * frequency)` blocks) is shorter
//! than the voxel are skipped, except the first of each noise
//! ([`OCTAVE_CUT`]; the count is per node and noise): at that size they are
//! detail the voxel cannot show. For continents at level 6 this drops six of
//! the eighteen layers.
//!
//! # API
//!
//! * [`ClimateNoises::from_graph`] / [`from_router`](ClimateNoises::from_router)
//!   read the six climate noises (and the offset noise) out of the compiled
//!   density graph; they are `None` for a router of another shape, whose users
//!   keep the CPU path.
//! * [`BiomeGenParams::new`] takes those, the climate parameter list as a
//!   `ClimateIndex<u32>` (entry value = the biome id) and the terrain table.
//! * [`BiomeGen::new`] builds the pipeline; [`BiomeGen::encode`] records the
//!   pass for a batch (at most [`MAX_BATCH`] nodes) into an encoder, with the
//!   [`ColumnTargets`] of `CubeGen::generate_to_pool_gpu`;
//!   [`BiomeGen::compute_columns`] does it alone and reads the columns back
//!   (tests, tools).
//! * [`CpuBiomes`] / [`reference_columns`] make the same columns on the CPU
//!   with the real `ClimateSampler`: the fallback, and what the tests compare
//!   against (GPU and CPU differ by `f32` noise rounding, nothing else, but for
//!   the octave cut).
//!
//! ```ignore
//! let noises = ClimateNoises::from_router(router, &compiler).unwrap();
//! let params = Arc::new(BiomeGenParams::new(noises, &index, table).unwrap());
//! let biomes = BiomeGen::new(device.clone(), queue.clone(), params).unwrap();
//! let slots = cube_gen.generate_to_pool_gpu(&keys, |encoder, targets| {
//!     biomes.encode(encoder, &keys, targets);
//! });
//! ```
//!
//! # Cost
//!
//! A climate evaluation is about 40 to 60 Perlin layers (the shift noise
//! twice) and a tree descent. Level 6 is one evaluation per column, 256 per
//! node, with the octave cut about 40 layers: 1024 nodes are roughly 260 000
//! evaluations, 10 million Perlin evaluations, a few milliseconds on a
//! desktop GPU. Level 0 to 3 nodes share the quart grid (81 to 1296
//! evaluations per node) and add the 25-window blend per column corner. The
//! per-node header (about 5.4 KB, uploaded with the queue) and its CPU split
//! (about 170 layers x 3 axes) are the CPU cost of a node: microseconds.
use crate::gfx::cube_gen::{ColumnTargets, NodeColumns};
use crate::lod::cube::CubeKey;
use horizon_core::{
    density::{
        SamplerContext, ShiftNoiseKind,
        graph::{DensityCompiler, FunctionId, GraphNode, NoiseId},
    },
    noise::{Noise, wrap},
};
use horizon_world::worldgen::{
    climate::{ClimateIndex, ClimateSampler},
    router::{NoiseRouter, RouterField},
};
use std::{borrow::Cow, collections::HashMap, sync::Arc, sync::mpsc, time::Duration};

/// Nodes one dispatch can hold.
pub const MAX_BATCH: usize = 1024;
/// Layers kept per noise. Overworld continents need 18.
pub const MAX_LAYERS: usize = 24;
/// Noise groups: shift A, shift B, temperature, vegetation, continents,
/// erosion, ridges.
pub const GROUPS: usize = 7;
const SLOTS: usize = GROUPS * MAX_LAYERS;
/// The highest level whose voxels are blended (8 blocks); coarser ones take
/// their own biome.
pub const BLEND_MAX_LEVEL: u8 = 3;
/// A layer is skipped when its wavelength is below `voxel * OCTAVE_CUT`.
pub const OCTAVE_CUT: f64 = 1.0;
/// The quart y of sea level (block 63): the climate point's y. The fields are
/// 2D, so it only documents the sample.
pub const SEA_LEVEL_QUART_Y: i32 = 63 >> 2;

const SHIFT_A: usize = 0;
const SHIFT_B: usize = 1;

/// Words of a node header (`biome_gen.wgsl`'s `Header`): `info` (origin x, y,
/// z, level), 8 layer counts (one per group), then per slot `group *
/// MAX_LAYERS + layer` the lattice cell (xyz, padded to 4) and the fraction
/// (xyz, padded to 4).
pub const HEADER_WORDS: usize = 4 + 8 + SLOTS * 8;
const HEADER_CELL: usize = 12;
const HEADER_FRAC: usize = HEADER_CELL + SLOTS * 4;

/// 17/64 and 1/256 of CubicWorldGen's height conversion, 2.4 and 4/15 of the
/// volatility's.
const HEIGHT_SCALE: f32 = 17.0 / 64.0;
const HEIGHT_OFFSET: f32 = -1.0 / 256.0;
const VOLATILITY_SCALE: f32 = 2.4;
const VOLATILITY_OFFSET: f32 = 4.0 / 15.0;

// ------------------------------------------------------------ parameters

/// One layer of a noise stack: a plain Perlin noise at a frequency, weighted.
#[derive(Clone, Debug)]
pub struct ClimateLayer {
    pub frequency: f64,
    pub amplitude: f32,
    pub permutations: [u8; 256],
    pub offsets: [f64; 3],
}

/// A `NormalNoise` as its layers in sampling order.
#[derive(Clone, Debug, Default)]
pub struct ClimateNoise {
    pub layers: Vec<ClimateLayer>,
}

impl ClimateNoise {
    /// The layers of a seeded noise, or `None` if it is not a stack of plain
    /// Perlin noises. Zero-weight layers are dropped.
    pub fn from_noise(noise: &dyn Noise) -> Option<Self> {
        let stack = noise.as_stack()?;
        let mut layers = Vec::new();
        for layer in stack.layers() {
            let table = layer.noise.gradient_table()?;
            if layer.amplitude == 0.0 {
                continue;
            }
            layers.push(ClimateLayer {
                frequency: layer.frequency,
                amplitude: layer.amplitude,
                permutations: *table.permutations(),
                offsets: *table.offsets(),
            });
        }
        Some(Self { layers })
    }
}

/// The noises of the overworld's climate: the offset noise behind the shifts
/// and the five fields the biome source reads (depth is taken as 0).
#[derive(Clone, Debug)]
pub struct ClimateNoises {
    /// 0.25 in vanilla: the block-to-noise scale of every field.
    pub xz_scale: f64,
    pub offset: ClimateNoise,
    /// Temperature, vegetation, continents, erosion, ridges.
    pub fields: [ClimateNoise; 5],
}

/// The nodes under a climate root that are only wrappers.
fn unwrap_node(compiler: &DensityCompiler, mut id: FunctionId) -> Option<&GraphNode> {
    for _ in 0..32 {
        match compiler.graph().node(id)? {
            GraphNode::Reference { input, .. } | GraphNode::Cache(input) => id = *input,
            node => return Some(node),
        }
    }
    None
}

impl ClimateNoises {
    /// Reads the noises of the six climate roots (temperature, vegetation,
    /// continents, erosion, depth, ridges: the router's first six fields)
    /// from the compiled graph. Each of the five fields must be a shifted
    /// noise with `y_scale` 0, no y shift, the same `xz_scale`, `shift_x` =
    /// `shift_a(n)` and `shift_z` = `shift_b(n)` of one noise `n`, under any
    /// references and caches (vanilla's overworld). `None` otherwise.
    pub fn from_graph(compiler: &DensityCompiler, roots: [FunctionId; 6]) -> Option<Self> {
        let mut scale = None;
        let mut offset_noise: Option<NoiseId> = None;
        let mut fields = Vec::new();
        for (i, root) in roots.into_iter().enumerate() {
            if i == 4 {
                continue;
            }
            let GraphNode::Noise {
                noise,
                xz_scale,
                y_scale,
                shift_x,
                shift_y,
                shift_z,
            } = *unwrap_node(compiler, root)?
            else {
                return None;
            };
            if y_scale != 0.0 || *scale.get_or_insert(xz_scale) != xz_scale {
                return None;
            }
            let GraphNode::Constant(zero) = *unwrap_node(compiler, shift_y)? else {
                return None;
            };
            if zero != 0.0 {
                return None;
            }
            let (
                GraphNode::OffsetNoise {
                    noise: a,
                    kind: ShiftNoiseKind::ShiftA,
                },
                GraphNode::OffsetNoise {
                    noise: b,
                    kind: ShiftNoiseKind::ShiftB,
                },
            ) = (
                unwrap_node(compiler, shift_x)?,
                unwrap_node(compiler, shift_z)?,
            )
            else {
                return None;
            };
            if a != b || *offset_noise.get_or_insert(*a) != *a {
                return None;
            }
            fields.push(ClimateNoise::from_noise(
                &*compiler.get_noise_sampler(noise).ok()?,
            )?);
        }
        Some(Self {
            xz_scale: scale?,
            offset: ClimateNoise::from_noise(&*compiler.get_noise_sampler(offset_noise?).ok()?)?,
            fields: fields.try_into().ok()?,
        })
    }

    /// [`Self::from_graph`] for a decoded router.
    pub fn from_router(router: NoiseRouter, compiler: &DensityCompiler) -> Option<Self> {
        let roots = std::array::from_fn(|i| router.function(RouterField::ALL[i]));
        Self::from_graph(compiler, roots)
    }

    fn group(&self, g: usize) -> &ClimateNoise {
        match g {
            SHIFT_A | SHIFT_B => &self.offset,
            _ => &self.fields[g - 2],
        }
    }
}

/// What a biome contributes to the terrain: CubicWorldGen's 1.12-style base
/// height and height variation (the numbers fed to the blend, before the
/// conversion).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BiomeTerrain {
    pub base_height: f32,
    pub height_variation: f32,
}

/// Everything the pass needs, flattened: the noises, the search tree, the
/// entry-to-biome map and the terrain table.
#[derive(Clone, Debug)]
pub struct BiomeGenParams {
    noises: ClimateNoises,
    /// 16 words per tree node.
    tree: Vec<i32>,
    /// Entry biome ids, then (base, variation) f32 bits per biome id.
    data: Vec<u32>,
    entries: usize,
    table: usize,
}

impl BiomeGenParams {
    /// `index` holds the parameter list, each entry's value being its biome
    /// id; `table[id]` is the biome's terrain (ids past its end use the last).
    /// `None` if a noise has more than [`MAX_LAYERS`] layers, a parameter does
    /// not fit 32 bits or the table is empty.
    pub fn new(
        noises: ClimateNoises,
        index: &ClimateIndex<u32>,
        table: Vec<BiomeTerrain>,
    ) -> Option<Self> {
        if table.is_empty()
            || noises.offset.layers.len() > MAX_LAYERS
            || noises.fields.iter().any(|f| f.layers.len() > MAX_LAYERS)
        {
            return None;
        }
        let (spaces, links) = index.tree();
        let to_i32 = |v: i64| i32::try_from(v).ok();
        let mut tree = Vec::with_capacity(spaces.len() * 16);
        for (space, link) in spaces.iter().zip(links) {
            for p in space {
                tree.push(to_i32(p.min)?);
                tree.push(to_i32(p.max)?);
            }
            tree.push(link.0 as i32);
            tree.push(link.1 as i32);
        }
        let mut data: Vec<u32> = index.entries().iter().map(|(_, id)| *id).collect();
        let entries = data.len();
        for t in &table {
            data.push(t.base_height.to_bits());
            data.push(t.height_variation.to_bits());
        }
        Some(Self {
            noises,
            tree,
            data,
            entries,
            table: table.len(),
        })
    }

    /// The layers kept for group `g` at `level`.
    fn kept(&self, g: usize, level: u8) -> usize {
        let layers = &self.noises.group(g).layers;
        let voxel = f64::from(1u32 << level) * OCTAVE_CUT;
        layers
            .iter()
            .enumerate()
            .take(MAX_LAYERS)
            .take_while(|(k, l)| *k == 0 || 1.0 / (self.noises.xz_scale * l.frequency) >= voxel)
            .count()
    }

    /// `Params` (`biome_gen.wgsl`) as words.
    fn param_words(&self) -> Vec<u32> {
        let mut w = vec![0u32; 8 + SLOTS * 4];
        w[0] = self.table as u32;
        w[1] = self.entries as u32;
        w[4] = (self.noises.xz_scale as f32).to_bits();
        for g in 0..GROUPS {
            for (k, l) in self
                .noises
                .group(g)
                .layers
                .iter()
                .enumerate()
                .take(MAX_LAYERS)
            {
                let at = 8 + (g * MAX_LAYERS + k) * 4;
                w[at] = (l.frequency as f32).to_bits();
                w[at + 1] = l.amplitude.to_bits();
                w[at + 2] = ((self.noises.xz_scale * l.frequency) as f32).to_bits();
            }
        }
        w
    }

    /// Packed permutation tables, 64 words per slot.
    fn perm_words(&self) -> Vec<u32> {
        let mut w = vec![0u32; SLOTS * 64];
        for g in 0..GROUPS {
            for (k, l) in self
                .noises
                .group(g)
                .layers
                .iter()
                .enumerate()
                .take(MAX_LAYERS)
            {
                let slot = g * MAX_LAYERS + k;
                for (i, bytes) in l.permutations.as_chunks::<4>().0.iter().enumerate() {
                    w[slot * 64 + i] = u32::from_le_bytes(*bytes);
                }
            }
        }
        w
    }

    /// Appends a node's header (the per-node, per-layer precision split).
    pub fn push_header(&self, key: CubeKey, out: &mut Vec<u32>) {
        let origin = key.origin();
        let start = out.len();
        out.resize(start + HEADER_WORDS, 0);
        let h = &mut out[start..];
        h[..4].copy_from_slice(&[
            origin[0] as u32,
            origin[1] as u32,
            origin[2] as u32,
            u32::from(key.level),
        ]);
        let (ox, oz) = (f64::from(origin[0]), f64::from(origin[2]));
        for g in 0..GROUPS {
            h[4 + g] = self.kept(g, key.level) as u32;
            for (k, l) in self
                .noises
                .group(g)
                .layers
                .iter()
                .enumerate()
                .take(MAX_LAYERS)
            {
                let scale = self.noises.xz_scale * l.frequency;
                let o = l.offsets;
                // The origin's part of the layer's input, per lattice axis.
                let axes = if g == SHIFT_B {
                    [(oz * scale, o[0]), (ox * scale, o[1]), (0.0, o[2])]
                } else {
                    [(ox * scale, o[0]), (0.0, o[1]), (oz * scale, o[2])]
                };
                let slot = g * MAX_LAYERS + k;
                for (a, (v, off)) in axes.into_iter().enumerate() {
                    let (cell, frac) = split(v, off);
                    h[HEADER_CELL + slot * 4 + a] = cell as u32;
                    h[HEADER_FRAC + slot * 4 + a] = frac.to_bits();
                }
            }
        }
    }
}

fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

/// A lattice input split into its cell (mod 256) and an `f32` fraction.
fn split(value: f64, offset: f64) -> (i32, f32) {
    let shifted = wrap(value) + offset;
    let cell = shifted.floor();
    ((cell as i64 & 255) as i32, (shifted - cell) as f32)
}

// ------------------------------------------------------------------- GPU

/// The pipeline and the static buffers of the pass.
pub struct BiomeGen {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    params_buf: wgpu::Buffer,
    perms_buf: wgpu::Buffer,
    headers: wgpu::Buffer,
    tree_buf: wgpu::Buffer,
    data_buf: wgpu::Buffer,
    params: Arc<BiomeGenParams>,
}

impl BiomeGen {
    /// The pass on this device, or `None` if it cannot run the shader.
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        params: Arc<BiomeGenParams>,
    ) -> Option<Self> {
        let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Biome columns"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("biome_gen.wgsl"))),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Biome columns"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let buffer = |label, size: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let param_words = params.param_words();
        let perm_words = params.perm_words();
        let params_buf = buffer("Biome params", (param_words.len() * 4) as u64);
        let perms_buf = buffer("Biome permutations", (perm_words.len() * 4) as u64);
        let headers = buffer("Biome headers", (HEADER_WORDS * 4 * MAX_BATCH) as u64);
        let tree_buf = buffer("Biome tree", (params.tree.len() * 4) as u64);
        let data_buf = buffer("Biome data", (params.data.len() * 4) as u64);
        queue.write_buffer(&params_buf, 0, bytemuck::cast_slice(&param_words));
        queue.write_buffer(&perms_buf, 0, bytemuck::cast_slice(&perm_words));
        queue.write_buffer(&tree_buf, 0, bytemuck::cast_slice(&params.tree));
        queue.write_buffer(&data_buf, 0, bytemuck::cast_slice(&params.data));
        let internal = pollster::block_on(internal.pop());
        if let Some(e) = internal.or(pollster::block_on(validation.pop())) {
            log::warn!("Horizon renderer: far terrain biomes on the CPU ({e})");
            return None;
        }
        Some(Self {
            device,
            queue,
            pipeline,
            params_buf,
            perms_buf,
            headers,
            tree_buf,
            data_buf,
            params,
        })
    }

    /// Records the pass for `keys` (at most [`MAX_BATCH`], and what `out`
    /// holds) into `encoder`, writing `out` (layouts at [`ColumnTargets`]). The
    /// headers are written with the queue now: record at most one pass per
    /// submission.
    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        keys: &[CubeKey],
        out: ColumnTargets<'_>,
    ) {
        let n = keys.len();
        assert!(n <= MAX_BATCH);
        assert!(out.columns.size() >= (n * 512 * 4) as u64);
        assert!(out.biomes.size() >= (n * 256 * 4) as u64);
        if n == 0 {
            return;
        }
        let mut words = Vec::with_capacity(n * HEADER_WORDS);
        for key in keys {
            self.params.push_header(*key, &mut words);
        }
        self.queue
            .write_buffer(&self.headers, 0, bytemuck::cast_slice(&words));

        let layout = self.pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Biome columns"),
            layout: &layout,
            entries: &[
                entry(0, &self.params_buf),
                entry(1, &self.perms_buf),
                entry(2, &self.headers),
                entry(3, &self.tree_buf),
                entry(4, &self.data_buf),
                entry(5, out.columns),
                entry(6, out.biomes),
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Biome columns"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(n as u32, 1, 1);
    }

    /// The columns of `keys` (at most [`MAX_BATCH`]), computed and read back:
    /// for tests and tools. `None` on failure.
    pub fn compute_columns(&self, keys: &[CubeKey]) -> Option<Vec<NodeColumns>> {
        let n = keys.len();
        assert!(n <= MAX_BATCH);
        if n == 0 {
            return Some(Vec::new());
        }
        let make = |label, size: u64, usage| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
        let columns = make("Biome columns out", (n * 512 * 4) as u64, storage);
        let biomes = make("Biome ids out", (n * 256 * 4) as u64, storage);
        let bytes = (n * 768 * 4) as u64;
        let staging = make(
            "Biome readback",
            bytes,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Biome columns"),
            });
        self.encode(
            &mut encoder,
            keys,
            ColumnTargets {
                columns: &columns,
                biomes: &biomes,
            },
        );
        encoder.copy_buffer_to_buffer(&columns, 0, &staging, 0, (n * 512 * 4) as u64);
        encoder.copy_buffer_to_buffer(
            &biomes,
            0,
            &staging,
            (n * 512 * 4) as u64,
            (n * 256 * 4) as u64,
        );
        let submission = self.queue.submit([encoder.finish()]);
        let (tx, rx) = mpsc::channel();
        staging
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
        let words: Vec<u32> = {
            let view = staging.slice(..bytes).get_mapped_range().ok()?;
            bytemuck::cast_slice::<u8, u32>(&view).to_vec()
        };
        staging.unmap();
        Some(
            (0..n)
                .map(|i| {
                    let col = &words[i * 512..i * 512 + 512];
                    let ids = &words[n * 512 + i * 256..n * 512 + i * 256 + 256];
                    NodeColumns {
                        height: std::array::from_fn(|c| f32::from_bits(col[c])),
                        volatility: std::array::from_fn(|c| f32::from_bits(col[256 + c])),
                        biome: std::array::from_fn(|c| ids[c] as u16),
                    }
                })
                .collect(),
        )
    }
}

// ----------------------------------------------------------- CPU reference

/// The real climate sampler, the parameter list and the terrain table: what
/// the GPU pass reproduces.
#[derive(Clone, Debug)]
pub struct CpuBiomes {
    pub climate: ClimateSampler,
    pub index: Arc<ClimateIndex<u32>>,
    pub table: Arc<[BiomeTerrain]>,
}

impl CpuBiomes {
    pub fn new(
        climate: ClimateSampler,
        index: Arc<ClimateIndex<u32>>,
        table: Vec<BiomeTerrain>,
    ) -> Self {
        Self {
            climate,
            index,
            table: table.into(),
        }
    }

    /// The biome id at quart `(qx, qz)`: climate at sea level with depth 0,
    /// then a fresh nearest search.
    pub fn biome_at_quart(&self, context: &mut SamplerContext, qx: i32, qz: i32) -> u32 {
        let Ok(mut target) = self.climate.sample(context, [qx, SEA_LEVEL_QUART_Y, qz]) else {
            return 0;
        };
        target.depth = 0;
        self.index
            .searcher()
            .find(target)
            .ok()
            .copied()
            .unwrap_or(0)
    }

    /// The columns of a node (see the module docs), with every octave.
    pub fn columns(&self, key: CubeKey) -> NodeColumns {
        let origin = key.origin();
        let (oqx, oqz) = (origin[0] >> 2, origin[2] >> 2);
        let mut context = SamplerContext::cached();
        let mut cache: HashMap<(i32, i32), u32> = HashMap::new();
        assemble(key.level, &self.table, |qx, qz| {
            *cache
                .entry((qx, qz))
                .or_insert_with(|| self.biome_at_quart(&mut context, oqx + qx, oqz + qz))
        })
    }

    /// A column closure for `cube_gen`'s CPU path.
    pub fn column_fn(self: Arc<Self>) -> crate::gfx::cube_gen::ColumnFn {
        Arc::new(move |key| self.columns(key))
    }
}

/// The node's columns on the CPU with the real climate sampler.
pub fn reference_columns(source: &CpuBiomes, key: CubeKey) -> NodeColumns {
    source.columns(key)
}

fn terrain_of(table: &[BiomeTerrain], biome: u32) -> BiomeTerrain {
    table[(biome as usize).min(table.len() - 1)]
}

/// The quart grid of a blended level: the first quart relative to the node
/// origin and the width.
fn grid_layout(s: i32) -> (i32, i32) {
    let rel_min = ((s >> 1) >> 2) - 2;
    let q0_max = (15 * s + (s >> 1)) >> 2;
    (rel_min, q0_max + 3 - rel_min + 1)
}

/// The weighted terrain over the 5x5 quarts around grid cell `(gx, gz)`.
fn blend_corner(table: &[BiomeTerrain], grid: &[u32], width: i32, gx: i32, gz: i32) -> [f32; 2] {
    let at = |x: i32, z: i32| terrain_of(table, grid[(z * width + x) as usize]);
    let center = at(gx, gz).base_height;
    let (mut hs, mut vs, mut ws) = (0.0f32, 0.0f32, 0.0f32);
    for j in -2i32..=2 {
        for i in -2i32..=2 {
            let t = at(gx + i, gz + j);
            let mut w =
                (10.0 / ((i * i + j * j) as f32 + 0.2).sqrt() / (t.base_height + 2.0)).abs();
            if t.base_height > center {
                w *= 0.5;
            }
            hs += t.base_height * w;
            vs += t.height_variation * w;
            ws += w;
        }
    }
    [hs / ws, vs / ws]
}

/// Columns from a biome function over quarts relative to the node origin (the
/// shader's algorithm in `f32`).
fn assemble(
    level: u8,
    table: &[BiomeTerrain],
    mut biome_at: impl FnMut(i32, i32) -> u32,
) -> NodeColumns {
    let s = 1i32 << level;
    let mut out = NodeColumns::flat(0.0, 0.0, 0);
    let convert = |c: usize, out: &mut NodeColumns, h: f32, v: f32, biome: u32| {
        out.height[c] = h * HEIGHT_SCALE + HEIGHT_OFFSET;
        out.volatility[c] = v * VOLATILITY_SCALE + VOLATILITY_OFFSET;
        out.biome[c] = biome as u16;
    };
    if level > BLEND_MAX_LEVEL {
        for z in 0..16i32 {
            for x in 0..16i32 {
                let biome = biome_at((x * s + (s >> 1)) >> 2, (z * s + (s >> 1)) >> 2);
                let t = terrain_of(table, biome);
                convert(
                    (z * 16 + x) as usize,
                    &mut out,
                    t.base_height,
                    t.height_variation,
                    biome,
                );
            }
        }
        return out;
    }
    let (rel_min, width) = grid_layout(s);
    let grid: Vec<u32> = (0..width * width)
        .map(|i| biome_at(rel_min + i % width, rel_min + i / width))
        .collect();
    for z in 0..16i32 {
        for x in 0..16i32 {
            let (x0, z0) = (x * s + (s >> 1), z * s + (s >> 1));
            let (gx, gz) = ((x0 >> 2) - rel_min, (z0 >> 2) - rel_min);
            let c00 = blend_corner(table, &grid, width, gx, gz);
            let hv = if x0 & 3 == 0 && z0 & 3 == 0 {
                c00
            } else {
                let (tx, tz) = ((x0 & 3) as f32 * 0.25, (z0 & 3) as f32 * 0.25);
                let c10 = blend_corner(table, &grid, width, gx + 1, gz);
                let c01 = blend_corner(table, &grid, width, gx, gz + 1);
                let c11 = blend_corner(table, &grid, width, gx + 1, gz + 1);
                let mix = |a: f32, b: f32, t: f32| a + t * (b - a);
                std::array::from_fn(|k| mix(mix(c00[k], c10[k], tx), mix(c01[k], c11[k], tx), tz))
            };
            let biome = grid[(gz * width + gx) as usize];
            convert((z * 16 + x) as usize, &mut out, hv[0], hv[1], biome);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use horizon_core::{density::graph::DensityGraph, noise::NormalNoise};
    use horizon_world::worldgen::climate::{Parameter, ParameterPoint};

    /// First octave and amplitudes of the overworld climate noises.
    const NOISES: [(&str, i32, &[f64]); 6] = [
        ("offset", -3, &[1.0, 1.0, 1.0, 0.0]),
        ("temperature", -10, &[1.5, 0.0, 1.0, 0.0, 0.0, 0.0]),
        ("vegetation", -8, &[1.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
        (
            "continentalness",
            -9,
            &[1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0],
        ),
        ("erosion", -9, &[1.0, 1.0, 0.0, 1.0, 1.0]),
        ("ridge", -7, &[1.0, 2.0, 1.0, 0.0, 0.0, 0.0]),
    ];
    const BIOMES: usize = 48;

    struct Fixture {
        noises: ClimateNoises,
        cpu: CpuBiomes,
        params: Arc<BiomeGenParams>,
    }

    fn table() -> Vec<BiomeTerrain> {
        (0..BIOMES)
            .map(|i| BiomeTerrain {
                base_height: -1.8 + 3.0 * ((i * 7 % BIOMES) as f32 / BIOMES as f32),
                height_variation: 0.0 + 1.2 * ((i * 11 % BIOMES) as f32 / BIOMES as f32),
            })
            .collect()
    }

    /// A synthetic parameter list: random boxes over the climate cube.
    fn index() -> ClimateIndex<u32> {
        let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = move |n: i64| {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (seed >> 33) as i64 % n
        };
        let entries = (0..600)
            .map(|id| {
                let coordinates = std::array::from_fn(|_| {
                    let min = next(20_000) - 10_000;
                    Parameter::new(min, min + next(4_000))
                });
                let point = ParameterPoint {
                    coordinates,
                    offset: 0,
                };
                (point, (id % BIOMES) as u32)
            })
            .collect();
        ClimateIndex::new(entries).unwrap()
    }

    fn fixture(seed: i64) -> Fixture {
        let mut graph = DensityGraph::new();
        let ids: Vec<NoiseId> = NOISES
            .iter()
            .map(|(name, octave, amplitudes)| {
                graph
                    .register_noise(
                        format!("minecraft:{name}"),
                        NormalNoise::create_parity(*octave, amplitudes).unwrap(),
                    )
                    .unwrap()
            })
            .collect();
        let zero = graph.insert(GraphNode::Constant(0.0)).unwrap();
        let shift_x = graph
            .insert(GraphNode::OffsetNoise {
                noise: ids[0],
                kind: ShiftNoiseKind::ShiftA,
            })
            .unwrap();
        let shift_z = graph
            .insert(GraphNode::OffsetNoise {
                noise: ids[0],
                kind: ShiftNoiseKind::ShiftB,
            })
            .unwrap();
        let field = |graph: &mut DensityGraph, noise| {
            graph
                .insert(GraphNode::Noise {
                    noise,
                    xz_scale: 0.25,
                    y_scale: 0.0,
                    shift_x,
                    shift_y: zero,
                    shift_z,
                })
                .unwrap()
        };
        let roots: [FunctionId; 6] = [
            field(&mut graph, ids[1]),
            field(&mut graph, ids[2]),
            field(&mut graph, ids[3]),
            field(&mut graph, ids[4]),
            zero,
            field(&mut graph, ids[5]),
        ];
        let mut compiler = DensityCompiler::new_seeded(graph, seed, false);
        let noises = ClimateNoises::from_graph(&compiler, roots).expect("climate noises");
        let samplers = roots.map(|id| compiler.get_sampler(id).unwrap());
        let index = Arc::new(index());
        let cpu = CpuBiomes::new(ClimateSampler::new(samplers), index.clone(), table());
        let params = Arc::new(BiomeGenParams::new(noises.clone(), &index, table()).unwrap());
        Fixture {
            noises,
            cpu,
            params,
        }
    }

    #[test]
    fn shader_validates() {
        let module = naga::front::wgsl::parse_str(include_str!("biome_gen.wgsl")).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
    }

    #[test]
    fn noises_come_out_of_the_graph() {
        let f = fixture(7);
        // Zero-amplitude octaves are dropped; each octave is two layers.
        let counts: Vec<usize> = std::iter::once(&f.noises.offset)
            .chain(&f.noises.fields)
            .map(|n| n.layers.len())
            .collect();
        assert_eq!(counts, [6, 4, 4, 18, 8, 6]);
        assert_eq!(f.noises.xz_scale, 0.25);
    }

    #[test]
    fn header_splits_far_coordinates() {
        let f = fixture(7);
        let mut words = Vec::new();
        f.params
            .push_header(CubeKey::new(2, 78_125, 0, -77_000), &mut words);
        assert_eq!(words.len(), HEADER_WORDS);
        assert_eq!(words[0] as i32, 5_000_000);
        for slot in 0..SLOTS {
            for a in 0..3 {
                let cell = words[HEADER_CELL + slot * 4 + a] as i32;
                let frac = f32::from_bits(words[HEADER_FRAC + slot * 4 + a]);
                assert!((0..256).contains(&cell), "{cell}");
                assert!((0.0..=1.0).contains(&frac), "{frac}");
            }
        }
        // Coarser voxels drop short layers, never the first.
        let kept = |level| {
            let mut w = Vec::new();
            f.params.push_header(CubeKey::new(level, 0, 0, 0), &mut w);
            w[4..4 + GROUPS].to_vec()
        };
        let (fine, coarse) = (kept(0), kept(8));
        assert!(coarse.iter().all(|&c| c >= 1));
        assert!(coarse.iter().zip(&fine).all(|(c, f)| c <= f));
        assert!(coarse[4] < fine[4]);
    }

    #[test]
    fn flat_terrain_gives_flat_columns() {
        let mut f = fixture(3);
        f.cpu.table = vec![
            BiomeTerrain {
                base_height: 0.125,
                height_variation: 0.05,
            };
            BIOMES
        ]
        .into();
        for level in [0u8, 3, 5] {
            let c = reference_columns(&f.cpu, CubeKey::new(level, 3, 0, -2));
            let h = 0.125 * HEIGHT_SCALE + HEIGHT_OFFSET;
            let v = 0.05 * VOLATILITY_SCALE + VOLATILITY_OFFSET;
            assert!(
                c.height.iter().all(|x| (x - h).abs() < 1e-5),
                "level {level}"
            );
            assert!(c.volatility.iter().all(|x| (x - v).abs() < 1e-5));
        }
    }

    #[test]
    fn blended_columns_vary_smoothly_across_a_block() {
        let f = fixture(11);
        let c = reference_columns(&f.cpu, CubeKey::new(0, 5, 0, 5));
        // Bilinear between 4-block corners: within a row no step is larger
        // than the corner spread allows, and at least some columns differ.
        let steps = (0..16)
            .flat_map(|z| (1..16).map(move |x| (z, x)))
            .map(|(z, x)| (c.height[z * 16 + x] - c.height[z * 16 + x - 1]).abs())
            .fold(0.0f32, f32::max);
        assert!(steps < 0.5, "{steps}");
    }

    fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = crate::gfx::test_gpu::instance();
        let adapter = pollster::block_on(instance.request_adapter(&Default::default())).ok()?;
        pollster::block_on(adapter.request_device(&Default::default())).ok()
    }

    fn agreement(gpu: &NodeColumns, cpu: &NodeColumns) -> (usize, usize) {
        let biomes = (0..256).filter(|&i| gpu.biome[i] == cpu.biome[i]).count();
        let heights = (0..256)
            .filter(|&i| {
                (gpu.height[i] - cpu.height[i]).abs() < 2e-3
                    && (gpu.volatility[i] - cpu.volatility[i]).abs() < 2e-3
            })
            .count();
        (biomes, heights)
    }

    #[test]
    fn gpu_matches_the_cpu_reference() {
        let Some((device, queue)) = device() else {
            eprintln!("no GPU adapter: skipped");
            return;
        };
        let f = fixture(12345);
        let Some(gen_) = BiomeGen::new(device, queue, f.params.clone()) else {
            eprintln!("no compute shaders: skipped");
            return;
        };
        // Every level, near the origin, at negative coordinates and at 5 M.
        let keys = [
            CubeKey::new(0, 0, 0, 0),
            CubeKey::new(0, -3, 2, 7),
            CubeKey::new(1, 5, 0, -4),
            CubeKey::new(2, 0, 0, 0),
            CubeKey::new(2, 78_125, 0, 0),
            CubeKey::new(2, 78_125, 1, -77_000),
            CubeKey::new(3, -40, 0, 11),
            CubeKey::new(4, 1, 0, 2),
            CubeKey::new(5, -9_000, 0, 5_000),
            CubeKey::new(6, 4_882, 0, 4),
        ];
        let gpu = gen_.compute_columns(&keys).expect("columns");
        let (mut biomes, mut heights, mut total) = (0, 0, 0);
        for (key, g) in keys.iter().zip(&gpu) {
            let c = reference_columns(&f.cpu, *key);
            let (b, h) = agreement(g, &c);
            eprintln!("{key:?}: biomes {b}/256, terrain {h}/256");
            // The octave cut makes coarse levels differ a little more.
            let floor = if key.level >= 5 { 230 } else { 250 };
            assert!(b >= floor, "{key:?}: only {b} biomes agree");
            biomes += b;
            heights += h;
            total += 256;
        }
        assert!(biomes * 100 >= total * 99, "{biomes}/{total}");
        assert!(heights * 100 >= total * 95, "{heights}/{total}");
    }

    #[test]
    fn precision_holds_at_five_million_blocks() {
        let Some((device, queue)) = device() else {
            return;
        };
        let f = fixture(99);
        let Some(gen_) = BiomeGen::new(device, queue, f.params.clone()) else {
            return;
        };
        // Level 0 nodes at x = 5,000,000 and x = -5,000,000.
        let keys = [
            CubeKey::new(0, 312_500, 0, 100),
            CubeKey::new(0, -312_501, 0, -300_000),
        ];
        let gpu = gen_.compute_columns(&keys).expect("columns");
        for (key, g) in keys.iter().zip(&gpu) {
            let (b, h) = agreement(g, &reference_columns(&f.cpu, *key));
            assert!(b >= 253 && h >= 245, "{key:?}: biomes {b}, terrain {h}");
        }
    }

    /// 1024 level 6 nodes' columns (run with `--ignored --nocapture`).
    #[test]
    #[ignore = "timing"]
    fn thousand_level_six_nodes() {
        let Some((device, queue)) = device() else {
            return;
        };
        let f = fixture(5);
        let Some(gen_) = BiomeGen::new(device.clone(), queue, f.params.clone()) else {
            return;
        };
        let keys: Vec<_> = (0..1024)
            .map(|i| CubeKey::new(6, i % 32 - 16, 0, i / 32 - 16))
            .collect();
        let _ = gen_.compute_columns(&keys);
        let start = std::time::Instant::now();
        let columns = gen_.compute_columns(&keys).expect("columns");
        eprintln!("1024 level-6 nodes: {:?}", start.elapsed());
        assert_eq!(columns.len(), 1024);
    }
}
