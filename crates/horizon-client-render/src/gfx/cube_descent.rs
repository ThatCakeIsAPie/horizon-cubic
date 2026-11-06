//! The sparse descent of the far-terrain octree, on the GPU.
//!
//! The owner's rule, applied to every node from the root down (`lod::cube`'s
//! plan, but with the contents of the node in the question):
//!
//! 1. **Proved all air or all solid?** Stop. Nothing is drawn, except a solid
//!    node's closed shell.
//! 2. **The right size for its distance?** (the request's `detail` times
//!    its size: the plan's test for nodes out of view.) Stop and hand the node to the visual engine, which
//!    generates its 16^3 voxels (`cube_gen`), meshes and draws them.
//! 3. **Otherwise** subdivide into its 8 children.
//!
//! # The bound (`cube_descent.wgsl`)
//!
//! A node is decided by one centre sample of each of the four noises plus
//! the conservative bound of `horizon_world::cubic::custom::bounds`
//! (`Analyzer::density_range`, read its module docs for the derivation):
//!
//! * per octave, the value at the box centre, widened by
//!   `G * sum_a |f_a| 2^i h_a` and clipped to `+-1.5`; octaves a node's
//!   voxels cannot show (`cube_gen::used_octaves`, so the bound is for the
//!   density the generator draws) and those beyond the uploaded 16 are
//!   bounded by amplitude (`skip`);
//! * the selector's clamp, the hull of low and high, the depth transform at
//!   the ends of its range and its kinks;
//! * the biome terms from the **global range** of the biome table (every
//!   blend lies in it, so it is always safe). The CPU bound uses the blend's
//!   corners for boxes of up to 33 x 33 quarts, which is tighter at the
//!   finest levels only (level 2); the GPU columns of `biome_gen` could feed
//!   the same slot, and are not wired here;
//! * the density's range over the 4 x 2 x 3 combinations of height,
//!   volatility, the `special` branch and the volatility's sign.
//!
//! What the bound is *for* is the far generator's own rules (`cube_gen`'s
//! simplified replacers, not the CPU chain of `Replacers::possible_blocks`:
//! those are not what the far view draws). Density `<= 0` everywhere is air
//! above the sea and water below it, uniform unless the sea level crosses
//! the node (`cube_gen::node_class`'s test); density `> 0` everywhere, over
//! the node **and the voxel layers the generator reads above it**, is rock
//! (no top block or filler can show, stone or deepslate by depth), a closed
//! shell. Everything else answers `Mixed`, so a node the bound cannot settle
//! is subdivided, never wrongly dropped.
//!
//! # Precision at large coordinates
//!
//! `cube_gen` gives every node a header with the octave's phase at its
//! origin, computed in `f64` on the CPU. The descent makes the children on
//! the GPU, so there is no CPU turn to do that per node: the shader computes
//! the phase `c * scale + offset` (mod 256, the permutation period) in
//! **exact fixed point** instead, from the block coordinate `c` (`i32`) and
//! per-octave constants split at upload ([`phase_constants`]):
//! `scale * 2^24 = kint + kfrac / 2^16` and `c = ch * 8192 + cl`, so every
//! partial product fits 32 bits and the sum wraps modulo `2^32` = 256 cells.
//! The result is the cell (top 8 bits) and an exactly representable
//! fraction (low 24 bits). Its error is below `|c| * 2^-40 + 2^-22` cells
//! (`2.7e-5` at 30,000,000 blocks), which the shader adds to every octave's
//! reach, so the bound stays rigorous. [`phase`] is the Rust twin used by
//! the tests.
//!
//! # Passes
//!
//! The input is the plan's roots (`lod::cube::root_nodes`, at most 27 nodes
//! of the top level). Level by level, one compute dispatch per level, all
//! recorded in one command buffer with no CPU round trip between them:
//!
//! * pass `j` reads list `j & 1` (`count`, then the `vec4<i32>` nodes
//!   `x, y, z, level`) and appends to list `(j + 1) & 1` with an atomic
//!   counter (8 children at a time); the counter of the list about to be
//!   written is cleared by `clear_buffer` before the pass;
//! * every node lands in the output list as one of `Leaf` (right-sized),
//!   `Fine` (right-sized for a level below the GPU's: the CPU plan owns it),
//!   `Air`, `Rock` (proved uniform; stops here), or one of the `*Split`
//!   records (uniform, but around the loaded cubes, where the plan splits
//!   whatever the class, so the node goes on splitting);
//! * the dispatch is fixed (the list capacity), not indirect: threads past
//!   the list's count return at once, and 21 levels cost 21 x 2048 empty
//!   groups at worst.
//!
//! The distance and hysteresis rule is `plan_classified`'s: the camera is
//! passed as integer part and fraction (a block coordinate beyond 2^24 loses
//! no precision), the gap per axis is computed as `f32(lo - cam_i) - cam_f`,
//! and the node splits while the distance is below `detail * size` (the
//! plan's factor, times `MERGE` to cover a plan that has hysteresis). Nodes outside the world or farther than the maximum
//! distance are dropped, as in the plan.
//!
//! # Leaves and the generator
//!
//! The readback is the compact output list only (counts first, then exactly
//! that many nodes), so the CPU never walks the octree: [`DescentResult::leaves`]
//! are the keys a `GpuCubeSampler::request_resident` batch takes. Feeding
//! them to `CubeGen` without leaving the GPU would need `GpuHeader`s built
//! on the GPU (the same fixed-point phase) and the 2D columns from
//! `biome_gen`; the output list is laid out for that (one `vec4<i32>` per
//! node), but the hand-over is not wired.
//!
//! # Overflow
//!
//! A full list drops nodes and sets a flag ([`DescentResult::overflow`]).
//! For the planner's use (classes of uniform nodes) a dropped node is simply
//! unknown, which is `Mixed`: safe. A caller using the leaves must not
//! trust a result with the flag set.

use crate::gfx::cube_gen::{
    CubeGenParams, GpuParams, MAX_OCTAVES, extra_layers, octave_terms, used_octaves,
};
use crate::lod::cube::{CubeKey, LoadedBox, NodeClass, WORLD_BORDER, root_nodes};
use bytemuck::Zeroable;
use horizon_world::cubic::custom::GpuTerms;
use std::{
    borrow::Cow,
    sync::{Mutex, mpsc},
    time::{Duration, Instant},
};

/// The descent shader: the shared noise (`cwg_noise.wgsl`), then
/// `cube_descent.wgsl`.
pub(crate) const SHADER: &str = concat!(
    include_str!("cwg_noise.wgsl"),
    include_str!("cube_descent.wgsl")
);

/// Nodes one level's list holds.
pub const LIST_CAP: u32 = 1 << 17;
/// Entries of the output list.
pub const OUT_CAP: u32 = 1 << 18;
/// Levels with a table (node levels are at most 22).
const LEVELS: usize = 32;
const WORKGROUP: u32 = 64;

/// What the descent decided about a node (`cube_descent.wgsl`'s kinds).
pub const KIND_LEAF: u32 = 1;
pub const KIND_FINE: u32 = 2;
pub const KIND_AIR: u32 = 3;
pub const KIND_ROCK: u32 = 4;
pub const KIND_MIXED: u32 = 5;
pub const KIND_AIR_SPLIT: u32 = 6;
pub const KIND_ROCK_SPLIT: u32 = 7;

// ------------------------------------------------------------ GPU layouts

/// `cube_descent.wgsl`'s `Frame`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuFrame {
    cam_i: [i32; 4],
    cam_f: [f32; 4],
    rule: [f32; 4],
    world: [i32; 4],
    center: [i32; 4],
    radius: [i32; 4],
}

/// `cube_descent.wgsl`'s `Level`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuLevel {
    used: [u32; 4],
    skip: [f32; 4],
    extra: [u32; 4],
}

/// `cube_descent.wgsl`'s `Tables`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuTables {
    shape: [f32; 4],
    shape2: [f32; 4],
    terrain: [f32; 4],
    slack: [f32; 4],
    kinks0: [f32; 4],
    kinks1: [f32; 4],
    noffs: [f32; 4],
    levels: [GpuLevel; LEVELS],
    w: [[f32; 4]; 64],
    kint: [[u32; 4]; 64],
    kfrac: [[u32; 4]; 64],
    off: [[u32; 4]; 64],
}

/// Allowance for the GPU's `f32` evaluation of one octave, on top of the CPU
/// bound's own `F32_SLACK` (the sum goes to the shader as one value).
const GPU_OCTAVE_SLACK: f64 = 2.0e-5;
/// Relative padding of the density range for `f32` sums (a few ulps of the
/// largest magnitude involved, which includes `y`).
const RELATIVE_PAD: f64 = 4.0e-6;

/// The constants of one axis of one octave for the shader's fixed-point
/// phase: `(kint, kfrac, off)` (see the module docs).
pub fn phase_constants(scale: f64, offset: f64) -> (u32, u32, u32) {
    let k = (scale * 16_777_216.0).rem_euclid(4_294_967_296.0);
    let whole = k.floor();
    let frac = ((k - whole) * 65_536.0).floor().min(65_535.0);
    let off = (offset.rem_euclid(256.0) * 16_777_216.0).round() as u64 as u32;
    (whole as u64 as u32, frac as u32, off)
}

/// The shader's `phase`, in Rust: the octave input at block coordinate `c`
/// modulo 256 cells, in units of 2^-24 cell.
pub fn phase(c: i32, kint: u32, kfrac: u32, off: u32) -> u32 {
    let cl = (c & 8191) as u32;
    let ch = c >> 13;
    let t2 = cl.wrapping_mul(kfrac) >> 16;
    let t1 = ch.wrapping_mul(kfrac as i32) >> 3;
    (c as u32)
        .wrapping_mul(kint)
        .wrapping_add(t1 as u32)
        .wrapping_add(t2)
        .wrapping_add(off)
}

#[allow(clippy::needless_range_loop)]
fn build_tables(p: &CubeGenParams, t: &GpuTerms) -> Option<(Box<GpuTables>, Vec<u32>)> {
    if t.noises
        .iter()
        .zip(&p.noises)
        .any(|(a, b)| a.octaves.len() != b.octaves.len())
    {
        return None;
    }
    let (_, perms) = GpuParams::new(p);
    let mut out = Box::new(GpuTables::zeroed());
    let s = t.shape;
    out.shape = [s[0] as f32, s[1] as f32, s[2] as f32, s[3] as f32];
    out.shape2 = [
        s[4] as f32,
        (p.sea_level + 1) as f32,
        t.gradient_bound as f32,
        t.value_bound as f32,
    ];
    out.terrain = [
        t.terrain_height[0] as f32,
        t.terrain_height[1] as f32,
        t.terrain_volatility[0] as f32,
        t.terrain_volatility[1] as f32,
    ];
    out.slack = [
        (t.f32_slack + GPU_OCTAVE_SLACK) as f32,
        t.margin as f32,
        RELATIVE_PAD as f32,
        0.0,
    ];
    for (dst, src) in out.kinks0.iter_mut().zip(&t.depth_kinks) {
        *dst = *src as f32;
    }
    out.kinks1[0] = t.depth_kinks[4] as f32;
    for (n, noise) in p.noises.iter().enumerate() {
        out.noffs[n] = noise.offset as f32;
        for (k, (scale, weight)) in octave_terms(noise, n == crate::gfx::cube_gen::DEPTH)
            .into_iter()
            .enumerate()
        {
            let slot = n * MAX_OCTAVES + k;
            out.w[slot] = [
                scale[0].abs() as f32,
                scale[1].abs() as f32,
                scale[2].abs() as f32,
                weight,
            ];
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
        let lv = &mut out.levels[level];
        for n in 0..4 {
            let used = used_octaves(p, n, level as u8);
            lv.used[n] = used as u32;
            lv.skip[n] = (t.value_bound
                * t.noises[n]
                    .octaves
                    .iter()
                    .skip(used)
                    .map(|o| o.amp)
                    .sum::<f64>()) as f32;
        }
        lv.extra[0] = extra_layers(p, level as u8);
    }
    Some((out, perms))
}

// --------------------------------------------------------------- requests

/// One descent: from the camera out to `max_distance`.
#[derive(Clone, Copy, Debug)]
pub struct DescentRequest {
    pub camera: [f64; 3],
    pub max_distance: f64,
    /// A node splits while nearer than this many times its size: the plan's
    /// factor (`lod::cube::Rule`), times `MERGE` to cover a plan with
    /// hysteresis.
    pub detail: f64,
    /// The finest level the GPU path serves (`cube_gen::MIN_LEVEL`): a node
    /// of this level that should split is reported as [`KIND_FINE`].
    pub min_level: u8,
    /// The camera's cube and the loaded box, when the plan has one: nodes
    /// above the probe level meeting it split whatever their class.
    pub loaded: Option<([i32; 3], LoadedBox)>,
}

/// What a descent found, as node keys.
#[derive(Clone, Debug, Default)]
pub struct DescentResult {
    /// Right-sized nodes: the plan's leaves of the GPU's levels.
    pub leaves: Vec<CubeKey>,
    /// Nodes that should split below the GPU's levels (the CPU plan owns them).
    pub fine: Vec<CubeKey>,
    /// Proved uniform air or water, and stopped.
    pub air: Vec<CubeKey>,
    /// Proved uniform rock, and stopped (a closed shell).
    pub rock: Vec<CubeKey>,
    /// Proved uniform, but split on (around the loaded cubes).
    pub air_split: Vec<CubeKey>,
    pub rock_split: Vec<CubeKey>,
    /// A list was full: nodes may be missing.
    pub overflow: bool,
    /// Levels visited.
    pub passes: u32,
    /// Wall time of the descent including the readback.
    pub elapsed: Duration,
}

impl DescentResult {
    /// Every node proved uniform, with the planner's class for it.
    pub fn uniform(&self) -> impl Iterator<Item = (CubeKey, NodeClass)> + '_ {
        let air = self.air.iter().chain(&self.air_split);
        let rock = self.rock.iter().chain(&self.rock_split);
        air.map(|&k| (k, NodeClass::Air))
            .chain(rock.map(|&k| (k, NodeClass::Rock)))
    }

    /// How many nodes were proved uniform.
    pub fn uniform_count(&self) -> usize {
        self.air.len() + self.rock.len() + self.air_split.len() + self.rock_split.len()
    }
}

// ----------------------------------------------------------------- the GPU

struct Scratch {
    frame: wgpu::Buffer,
    lists: [wgpu::Buffer; 2],
    out: wgpu::Buffer,
    meta: wgpu::Buffer,
    items: wgpu::Buffer,
    groups: [wgpu::BindGroup; 2],
}

/// The descent's pipeline and buffers on one device.
pub struct GpuDescent {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    constants: wgpu::BindGroup,
    scratch: Mutex<Scratch>,
}

const LIST_HEADER: u64 = 16;

impl GpuDescent {
    /// The descent for the generator `params` describe, on this device, with
    /// the bound's terms; `None` when the terms do not match the params or
    /// the shader cannot be built here.
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        params: &CubeGenParams,
        terms: &GpuTerms,
    ) -> Option<Self> {
        let (tables, perms) = build_tables(params, terms)?;
        let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Cube descent"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SHADER)),
        });
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry {
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
        let constants_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Cube descent constants"),
            entries: &[
                entry(0, uniform),
                entry(1, storage(true)),
                entry(2, storage(true)),
                entry(3, storage(false)),
            ],
        });
        let lists_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Cube descent lists"),
            entries: &[entry(0, storage(true)), entry(1, storage(false))],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Cube descent"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Cube descent"),
                    bind_group_layouts: &[Some(&constants_layout), Some(&lists_layout)],
                    ..Default::default()
                }),
            ),
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
        let storage_usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let frame = buffer(
            "Cube descent frame",
            std::mem::size_of::<GpuFrame>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let tables_buffer = buffer(
            "Cube descent tables",
            std::mem::size_of::<GpuTables>() as u64,
            storage_usage,
        );
        let perms_buffer = buffer(
            "Cube descent permutations",
            (perms.len() * 4) as u64,
            storage_usage,
        );
        let list_bytes = LIST_HEADER + u64::from(LIST_CAP) * 16;
        let lists = [
            buffer("Cube descent list A", list_bytes, storage_usage),
            buffer("Cube descent list B", list_bytes, storage_usage),
        ];
        let out_bytes = LIST_HEADER + u64::from(OUT_CAP) * 16;
        let out = buffer(
            "Cube descent output",
            out_bytes,
            storage_usage | wgpu::BufferUsages::COPY_SRC,
        );
        let readback = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;
        let meta = buffer("Cube descent counts", LIST_HEADER, readback);
        let items = buffer("Cube descent readback", out_bytes, readback);
        queue.write_buffer(&tables_buffer, 0, bytemuck::bytes_of(&*tables));
        queue.write_buffer(&perms_buffer, 0, bytemuck::cast_slice(&perms));
        let bind = |label, layout, entries: &[wgpu::BindGroupEntry<'_>]| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout,
                entries,
            })
        };
        fn at(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
            wgpu::BindGroupEntry {
                binding,
                resource: buffer.as_entire_binding(),
            }
        }
        let constants = bind(
            "Cube descent constants",
            &constants_layout,
            &[
                at(0, &frame),
                at(1, &tables_buffer),
                at(2, &perms_buffer),
                at(3, &out),
            ],
        );
        let groups = [
            bind(
                "Cube descent A to B",
                &lists_layout,
                &[at(0, &lists[0]), at(1, &lists[1])],
            ),
            bind(
                "Cube descent B to A",
                &lists_layout,
                &[at(0, &lists[1]), at(1, &lists[0])],
            ),
        ];
        let internal = pollster::block_on(internal.pop());
        if let Some(e) = internal.or(pollster::block_on(validation.pop())) {
            log::warn!("Horizon renderer: far terrain descent unavailable ({e})");
            return None;
        }
        Some(Self {
            device,
            queue,
            pipeline,
            constants,
            scratch: Mutex::new(Scratch {
                frame,
                lists,
                out,
                meta,
                items,
                groups,
            }),
        })
    }

    fn frame(request: &DescentRequest, classify_only: bool) -> GpuFrame {
        let mut f = GpuFrame::zeroed();
        for a in 0..3 {
            let floor = request.camera[a].floor();
            f.cam_i[a] = floor as i32;
            f.cam_f[a] = (request.camera[a] - floor) as f32;
        }
        f.cam_f[3] = request.max_distance as f32;
        f.rule = [
            request.detail as f32,
            if classify_only { 1.0 } else { 0.0 },
            0.0,
            0.0,
        ];
        f.world = [
            WORLD_BORDER as i32,
            horizon_world::cubic::MIN_Y,
            horizon_world::cubic::MAX_Y,
            i32::from(request.min_level),
        ];
        if let Some((cube, loaded)) = request.loaded {
            f.center = [cube[0], cube[1], cube[2], 1];
            f.radius = [loaded.horizontal, loaded.vertical, 0, 0];
        }
        f
    }

    /// Runs the descent from the plan's roots. `None` if the GPU failed.
    pub fn descend(&self, request: &DescentRequest) -> Option<DescentResult> {
        let started = Instant::now();
        let roots = root_nodes(request.camera, request.max_distance);
        let mut result = DescentResult::default();
        let Some(top) = roots.first().map(|r| r.level) else {
            return Some(result);
        };
        let scratch = self.scratch.lock().ok()?;
        let min_level = request.min_level.min(top);
        let passes = u32::from(top - min_level) + 1;
        let frame = Self::frame(request, false);
        self.queue
            .write_buffer(&scratch.frame, 0, bytemuck::bytes_of(&frame));
        let mut list = vec![0_i32; 4 + roots.len() * 4];
        list[0] = roots.len() as i32;
        for (i, r) in roots.iter().enumerate() {
            list[4 + i * 4..8 + i * 4].copy_from_slice(&[r.x, r.y, r.z, i32::from(r.level)]);
        }
        self.queue
            .write_buffer(&scratch.lists[0], 0, bytemuck::cast_slice(&list));
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Cube descent"),
            });
        encoder.clear_buffer(&scratch.out, 0, Some(LIST_HEADER));
        encoder.clear_buffer(&scratch.lists[1], 0, Some(LIST_HEADER));
        for j in 0..passes {
            let parity = (j & 1) as usize;
            if j > 0 {
                encoder.clear_buffer(&scratch.lists[1 - parity], 0, Some(LIST_HEADER));
            }
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Cube descent level"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.constants, &[]);
            pass.set_bind_group(1, &scratch.groups[parity], &[]);
            pass.dispatch_workgroups(LIST_CAP / WORKGROUP, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&scratch.out, 0, &scratch.meta, 0, LIST_HEADER);
        let submission = self.queue.submit([encoder.finish()]);
        let meta = self.read(&scratch.meta, LIST_HEADER, Some(submission))?;
        let count = meta[0].min(OUT_CAP);
        result.overflow = meta[1] != 0 || meta[0] > OUT_CAP;
        result.passes = passes;
        if count > 0 {
            let bytes = u64::from(count) * 16;
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Cube descent readback"),
                });
            encoder.copy_buffer_to_buffer(&scratch.out, LIST_HEADER, &scratch.items, 0, bytes);
            let submission = self.queue.submit([encoder.finish()]);
            let words = self.read(&scratch.items, bytes, Some(submission))?;
            for node in words.as_chunks::<4>().0 {
                let kind = (node[3] >> 8) & 255;
                let key = CubeKey::new(
                    (node[3] & 255) as u8,
                    node[0] as i32,
                    node[1] as i32,
                    node[2] as i32,
                );
                match kind {
                    KIND_LEAF => result.leaves.push(key),
                    KIND_FINE => result.fine.push(key),
                    KIND_AIR => result.air.push(key),
                    KIND_ROCK => result.rock.push(key),
                    KIND_AIR_SPLIT => result.air_split.push(key),
                    KIND_ROCK_SPLIT => result.rock_split.push(key),
                    _ => {}
                }
            }
        }
        result.elapsed = started.elapsed();
        Some(result)
    }

    /// Classifies each of `keys` (at most [`LIST_CAP`]) by the bound alone, in
    /// order: [`KIND_AIR`], [`KIND_ROCK`] or [`KIND_MIXED`]. For tests and
    /// callers that want the bound without the descent.
    pub fn classify(&self, keys: &[CubeKey]) -> Option<Vec<u32>> {
        if keys.is_empty() {
            return Some(Vec::new());
        }
        if keys.len() > LIST_CAP as usize {
            return None;
        }
        let scratch = self.scratch.lock().ok()?;
        let request = DescentRequest {
            camera: [0.0; 3],
            max_distance: 0.0,
            // Classifying only: no node splits.
            detail: 1.0,
            min_level: 0,
            loaded: None,
        };
        let frame = Self::frame(&request, true);
        self.queue
            .write_buffer(&scratch.frame, 0, bytemuck::bytes_of(&frame));
        let mut list = vec![0_i32; 4 + keys.len() * 4];
        list[0] = keys.len() as i32;
        for (i, k) in keys.iter().enumerate() {
            list[4 + i * 4..8 + i * 4].copy_from_slice(&[k.x, k.y, k.z, i32::from(k.level)]);
        }
        self.queue
            .write_buffer(&scratch.lists[0], 0, bytemuck::cast_slice(&list));
        let bytes = keys.len() as u64 * 16;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Cube descent classify"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Cube descent classify"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.constants, &[]);
            pass.set_bind_group(1, &scratch.groups[0], &[]);
            pass.dispatch_workgroups(LIST_CAP / WORKGROUP, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&scratch.out, LIST_HEADER, &scratch.items, 0, bytes);
        let submission = self.queue.submit([encoder.finish()]);
        let words = self.read(&scratch.items, bytes, Some(submission))?;
        Some(
            words
                .as_chunks::<4>()
                .0
                .iter()
                .map(|node| (node[3] >> 8) & 255)
                .collect(),
        )
    }

    /// Maps the first `bytes` of a staging buffer and returns them as words.
    fn read(
        &self,
        staging: &wgpu::Buffer,
        bytes: u64,
        submission: Option<wgpu::SubmissionIndex>,
    ) -> Option<Vec<u32>> {
        let (tx, rx) = mpsc::channel();
        staging
            .slice(..bytes)
            .map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
        let polled = self.device.poll(wgpu::PollType::Wait {
            submission_index: submission,
            timeout: Some(Duration::from_secs(30)),
        });
        if polled.is_err() || !matches!(rx.recv_timeout(Duration::from_secs(30)), Ok(Ok(()))) {
            return None;
        }
        let words = {
            let view = staging.slice(..bytes).get_mapped_range().ok()?;
            bytemuck::cast_slice::<u8, u32>(&view).to_vec()
        };
        staging.unmap();
        Some(words)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fast_hash::{FxHashMap, FxHashSet};
    use crate::gfx::cube_gen::{BiomeBlocks, NoiseDesc, OctaveDesc};
    use crate::lod::cube::{DETAIL, LoadedBox, Rule, plan_classified};
    use horizon_core::noise::wrap;
    use horizon_world::cubic::custom::testing::{bound_presets, default_registry};
    use horizon_world::cubic::custom::{BoxClass, CustomCubicGenerator, FixedBiome};
    use std::sync::Arc;

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
    }

    fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = crate::gfx::test_gpu::instance();
        let adapter = pollster::block_on(instance.request_adapter(&Default::default())).ok()?;
        pollster::block_on(adapter.request_device(&Default::default())).ok()
    }

    /// The generator's parameters in the GPU generator's form.
    fn params_of(g: &CustomCubicGenerator) -> CubeGenParams {
        let tables = g.noise_tables().expect("a terrain generator");
        let shape = g.density_shape().unwrap();
        let noise = |t: &horizon_world::cubic::custom::NoiseTable| NoiseDesc {
            factor: t.factor,
            offset: t.offset,
            frequency: t.frequency,
            octaves: t
                .octaves
                .iter()
                .map(|o| OctaveDesc {
                    permutations: o.permutations,
                    offsets: o.offsets,
                })
                .collect(),
        };
        CubeGenParams {
            noises: std::array::from_fn(|i| noise(&tables[i])),
            height_factor: shape[0],
            height_offset: shape[1],
            height_variation_factor: shape[2],
            height_variation_offset: shape[3],
            below_average_factor: shape[4],
            sea_level: g.sea_level(),
            air: 0,
            stone: 1,
            deepslate: 5,
            water: 26,
            biomes: vec![BiomeBlocks { top: 9, filler: 7 }],
            filler_depth: 4.0,
            deepslate_from: 0.0,
            deepslate_to: 8.0,
        }
    }

    fn generators() -> Vec<(&'static str, CustomCubicGenerator)> {
        bound_presets()
            .into_iter()
            .map(|(name, s)| {
                let g = CustomCubicGenerator::new(
                    11,
                    &s,
                    default_registry(),
                    Arc::new(FixedBiome::new("plains")),
                )
                .unwrap();
                (name, g)
            })
            .collect()
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

    /// The fixed-point phase equals the f64 origin split of `cube_gen`'s
    /// headers to far better than the allowance the shader adds, up to the
    /// world border.
    #[test]
    fn the_fixed_point_phase_is_exact_at_the_world_border() {
        let mut rng = Lcg(5);
        let mut worst: f64 = 0.0;
        for &x in &[0, 1, -1, 30_000_000, -30_000_000, 29_999_999, 67_108_863] {
            for octave in 0..12 {
                let scale = 1.0 / 684.412 * f64::from(1 << octave);
                let offset = f64::from(rng.range(0, 25_599)) / 100.0 + 0.123_456_789;
                let (ki, kf, ko) = phase_constants(scale, offset);
                let got = f64::from(phase(x, ki, kf, ko)) / 16_777_216.0;
                let want = (wrap(f64::from(x) * scale) + offset).rem_euclid(256.0);
                let diff = (got - want).abs();
                let diff = diff.min(256.0 - diff);
                worst = worst.max(diff);
                // The shader's allowance for this coordinate.
                assert!(
                    diff <= 1.0e-12 * f64::from(x).abs() + 3.0e-7,
                    "x {x} octave {octave}: {diff}"
                );
            }
        }
        assert!(worst < 7.0e-5, "{worst}");
        // The planar depth noise: no y scale, only the offset.
        let (ki, kf, ko) = phase_constants(0.0, 77.77);
        assert_eq!(phase(123_456, ki, kf, ko), ko);
    }

    #[test]
    fn the_tables_follow_the_terms() {
        for (name, g) in generators() {
            let p = params_of(&g);
            let terms = g.gpu_terms().unwrap();
            let (tables, perms) = build_tables(&p, &terms).unwrap();
            assert_eq!(perms.len(), 4 * MAX_OCTAVES * 64, "{name}");
            for n in 0..4 {
                for level in [2_usize, 6, 22] {
                    let used = tables.levels[level].used[n] as usize;
                    assert!((1..=MAX_OCTAVES).contains(&used), "{name} {n} {level}");
                    // Every octave is either in use or in the skipped amplitude.
                    let all: f64 = terms.noises[n].octaves.iter().map(|o| o.amp).sum();
                    let kept: f64 = (0..used)
                        .map(|k| f64::from(tables.w[n * MAX_OCTAVES + k][3]).abs())
                        .sum();
                    let skipped = f64::from(tables.levels[level].skip[n]) / terms.value_bound;
                    assert!(
                        (kept + skipped - all).abs() < 1e-5 * all.max(1.0),
                        "{name} {n} {level}: {kept} + {skipped} vs {all}"
                    );
                }
            }
            // Coarser nodes use fewer octaves.
            assert!(
                tables.levels[8].used[1] <= tables.levels[2].used[1],
                "{name}"
            );
        }
    }

    /// The GPU bound is never more permissive than the CPU's `classify_node`:
    /// wherever the GPU proves air or rock, the CPU proves it too. Random
    /// nodes at several levels, three presets, coordinates out to the border.
    #[test]
    fn the_gpu_bound_is_never_more_permissive_than_the_cpu_bound() {
        let Some((device, queue)) = device() else {
            eprintln!("no GPU adapter: skipped");
            return;
        };
        let mut rng = Lcg(2024);
        let (mut proven, mut total) = (0, 0);
        for (name, g) in generators() {
            let p = params_of(&g);
            let terms = g.gpu_terms().unwrap();
            let descent = GpuDescent::new(device.clone(), queue.clone(), &p, &terms).unwrap();
            let mut keys = Vec::new();
            for _ in 0..1500 {
                let level = rng.range(2, 12) as u8;
                let span = (30_000_000 >> (4 + level)).max(1);
                let (x, z) = if rng.range(0, 3) == 0 {
                    (rng.range(-span, span), rng.range(-span, span))
                } else {
                    (rng.range(-8, 8), rng.range(-8, 8))
                };
                let ymax = (8_000_000 >> (4 + level)).clamp(1, 12);
                let y = rng.range(-ymax, ymax);
                keys.push(CubeKey::new(level, x, y, z));
            }
            let kinds = descent.classify(&keys).expect("the GPU answered");
            for (key, kind) in keys.iter().zip(kinds) {
                total += 1;
                // The CPU bound for the same truncated density.
                let cpu = g.classify_node(
                    u32::from(key.level),
                    key.x,
                    key.y,
                    key.z,
                    f64::from(1u32 << key.level),
                );
                match kind {
                    KIND_AIR => {
                        proven += 1;
                        assert_eq!(cpu.class, BoxClass::Air, "{name} {key:?}");
                    }
                    KIND_ROCK => {
                        proven += 1;
                        assert_eq!(cpu.class, BoxClass::Solid, "{name} {key:?}");
                    }
                    KIND_MIXED => {}
                    other => panic!("{name} {key:?}: kind {other}"),
                }
            }
        }
        assert!(proven > 200, "the bound proved {proven} of {total}");
    }

    /// The descent's leaves are the CPU plan's, minus the subtrees it proved
    /// uniform: the plan fed the descent's classes is the same set.
    #[test]
    fn the_descent_matches_the_plan_minus_uniform_subtrees() {
        let Some((device, queue)) = device() else {
            eprintln!("no GPU adapter: skipped");
            return;
        };
        let (_, g) = generators().remove(0);
        let p = params_of(&g);
        let terms = g.gpu_terms().unwrap();
        let descent = GpuDescent::new(device, queue, &p, &terms).unwrap();
        let none = LoadedBox {
            horizontal: 0,
            vertical: 0,
        };
        for (camera, max_distance) in [
            ([0.0, 100.0, 0.0], 8192.0),
            ([123_456.3, 70.0, -9_876.5], 100_000.0),
            ([29_000_000.5, 300.0, 1_000_000.25], 4_000_000.0),
            ([0.0, 100.0, 0.0], 67_108_864.0),
        ] {
            let request = DescentRequest {
                camera,
                max_distance,
                detail: DETAIL,
                min_level: 2,
                loaded: Some((camera.map(|v| (v / 16.0).floor() as i32), none)),
            };
            let found = descent.descend(&request).expect("the GPU answered");
            assert!(!found.overflow);
            let classes: FxHashMap<CubeKey, NodeClass> = found.uniform().collect();
            let class = |k: CubeKey| classes.get(&k).copied().unwrap_or(NodeClass::Mixed);
            let plan = plan_classified(
                camera,
                max_distance,
                &Rule::distance(DETAIL),
                none,
                &|_| false,
                &[],
                &class,
            )
            .leaves;
            // Plan leaves of the GPU's levels, and the coarser ones standing
            // for the plan's levels 0 and 1.
            let gpu: FxHashSet<CubeKey> = found.leaves.iter().chain(&found.rock).copied().collect();
            let fine: Vec<CubeKey> = found.fine.clone();
            let mut want = FxHashSet::default();
            for &leaf in &plan {
                if leaf.level >= 2 {
                    want.insert(leaf);
                } else {
                    let mut up = leaf;
                    while up.level < 2 {
                        up = up.parent();
                    }
                    assert!(fine.contains(&up), "{leaf:?} has no fine ancestor");
                }
            }
            assert_eq!(gpu.len(), want.len(), "{camera:?} {max_distance}");
            assert_eq!(gpu, want, "{camera:?} {max_distance}");
            // The descent actually pruned something.
            assert!(found.uniform_count() > 0, "{camera:?} {max_distance}");
        }
    }

    /// Nodes the GPU proves uniform really are, in the lattice generator,
    /// at sampled blocks (the CPU's own proof is the reference).
    #[test]
    fn proved_uniform_nodes_hold_at_sampled_blocks() {
        let Some((device, queue)) = device() else {
            eprintln!("no GPU adapter: skipped");
            return;
        };
        let mut rng = Lcg(9);
        for (name, g) in generators() {
            let p = params_of(&g);
            let terms = g.gpu_terms().unwrap();
            let descent = GpuDescent::new(device.clone(), queue.clone(), &p, &terms).unwrap();
            let found = descent
                .descend(&DescentRequest {
                    camera: [40.0, 80.0, -40.0],
                    max_distance: 16_384.0,
                    detail: DETAIL,
                    min_level: 2,
                    loaded: None,
                })
                .unwrap();
            for key in found.rock.iter().chain(&found.air).take(300) {
                let side = key.size();
                let o = key.origin();
                for _ in 0..40 {
                    let at = [
                        o[0] + rng.range(0, side - 1),
                        o[1] + rng.range(0, side - 1),
                        o[2] + rng.range(0, side - 1),
                    ];
                    let block = g.block_at(at[0], at[1], at[2], 1);
                    let rock = found.rock.contains(key);
                    let solid = block != p.air && block != p.water;
                    assert_eq!(solid, rock, "{name} {key:?} at {at:?}");
                }
            }
        }
    }
}
