//! Far cube terrain, generated, meshed and drawn on the GPU.
//!
//! The CPU only decides which nodes exist (`lod::cube::plan`, `CubeLod`). A node
//! arrives as a slot of the GPU node pool (generated there, or a CPU-sampled
//! node uploaded into one), the mesher (`cube_mesh`) turns it into quads in the
//! geometry arena, and every frame a compute pass culls the nodes and writes one
//! indirect draw per (node, direction, pass) that the vertex-pulling pipelines
//! of `cube_far.wgsl` draw, shaded by terrain.wgsl as the far tiles are.
//!
//! # Per frame
//!
//! 1. [`CubeFar::upload`] (the call that replaces `upload_cube_lods` for these
//!    nodes): polls the mesher, installs meshed nodes in the node table, hands
//!    the next batch to the mesher, and, once nothing is pending, fades the new
//!    nodes in and the replaced ones out, as `Renderer::upload_cube_lods` does.
//! 2. [`CubeFar::prepare`]: the culling pass (`cube_cull.wgsl`), before the
//!    world pass.
//! 3. [`CubeFar::draw`]: in the world pass, solid after the terrain, then the
//!    water in the translucent stage. One `multi_draw_indirect_count` per pass
//!    where the device draws counts the GPU wrote, else a `multi_draw_indirect`
//!    over the cleared (zero-sized) tail.
//!
//! # Node table
//!
//! [`NODE_WORDS`] words per node id: origin, level, fade times, flags, and the
//! twelve `(first quad, quads)` ranges. The ids are the draws' first instances.
use super::{
    cube_gen::{GpuBackend, PoolHandle, SLOT_BYTES},
    cube_mesh::{
        CubeMesher, MATERIAL_STRIDE, MAX_JOBS, MaterialTable, MeshJob, Meshed, NONE, cover_words,
        uniform_quads,
    },
};
use crate::{
    fast_hash::{FxHashMap, FxHashSet},
    lod::cube::{CubeKey, GpuContent, GpuNode},
};
use glam::{DVec3, Vec4};
use std::{borrow::Cow, collections::VecDeque, time::Instant};

/// Words per node in the node table.
pub const NODE_WORDS: usize = 32;
/// Nodes the table holds.
pub const MAX_NODES: u32 = 32_768;
/// Fade time in milliseconds (`lod::FADE`).
const FADE_MS: u32 = 600;
const NOT_SET: u32 = u32::MAX;
/// The arena is at most this many quads (128 MiB).
const MAX_ARENA_QUADS: u32 = 16 << 20;

/// Counters for `CubeLodStats` and the overlay.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FarStats {
    /// Nodes in the node table (including empty ones).
    pub nodes: usize,
    /// Waiting for the mesher.
    pub pending: usize,
    /// Nodes meshed on the GPU, ever.
    pub meshed: u64,
    /// Uniform nodes that needed no GPU work, ever.
    pub uniform: u64,
    /// Nodes the arena or table had no room for, ever.
    pub failed: u64,
    pub arena_bytes: u64,
    pub arena_capacity_bytes: u64,
    /// Pool bytes in use (slots not free).
    pub pool_bytes: u64,
    /// Whether a mesher batch is in flight.
    pub meshing: bool,
}

struct Held {
    id: Option<u32>,
    start: u32,
    total: u32,
    shown: Option<u32>,
    dying: Option<u32>,
}

/// `cube_cull.wgsl`'s `Frame`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FrameUniform {
    planes: [[f32; 4]; 6],
    cam_int: [i32; 4],
    cam_frac: [f32; 4],
    now: u32,
    nodes: u32,
    pad: [u32; 2],
}

/// The state a [`CubeFar`] needs from the renderer to build its pipelines.
pub struct DrawSetup<'a> {
    /// Group 0 of the terrain pipelines.
    pub bindings: &'a wgpu::BindGroupLayout,
    pub format: wgpu::TextureFormat,
    /// `SPRITE_ARRAYS` and `SPRITE_SPLIT`.
    pub sprite_constants: [f64; 2],
    /// `multi_draw_indirect_count` is available and safe on this backend.
    pub counts: bool,
}

pub struct CubeFar {
    device: wgpu::Device,
    queue: wgpu::Queue,
    mesher: CubeMesher,
    pool: PoolHandle,
    table: MaterialTable,
    air: u32,
    node_buffer: wgpu::Buffer,
    biomes: wgpu::Buffer,
    solid_args: wgpu::Buffer,
    translucent_args: wgpu::Buffer,
    draw_counts: wgpu::Buffer,
    frame_buffer: wgpu::Buffer,
    cull: wgpu::ComputePipeline,
    cull_group: wgpu::BindGroup,
    draw_group: wgpu::BindGroup,
    solid: wgpu::RenderPipeline,
    translucent: wgpu::RenderPipeline,
    counts: bool,
    nodes: FxHashMap<CubeKey, Held>,
    free_ids: Vec<u32>,
    next_id: u32,
    pending: FxHashMap<CubeKey, GpuNode>,
    order: VecDeque<CubeKey>,
    removed: FxHashSet<CubeKey>,
    /// The batch in the mesher belongs to a forgotten plan (`clear`).
    discard: bool,
    epoch: Instant,
    stats: FarStats,
}

/// The terrain shader with the vertex-pulling entry point appended.
pub fn shader_source() -> String {
    format!(
        "{}\n{}",
        include_str!("../terrain.wgsl"),
        include_str!("cube_far.wgsl")
    )
}

impl CubeFar {
    /// The GPU far terrain on the sampler's device, or `None` when it cannot
    /// run (too few storage buffers, shader errors, no room for the buffers).
    pub fn new(
        backend: &GpuBackend,
        states: usize,
        air: u32,
        setup: &DrawSetup<'_>,
    ) -> Option<Self> {
        let (device, queue) = (&backend.device, &backend.queue);
        let limits = device.limits();
        if limits.max_storage_buffers_per_shader_stage < 5 {
            return None;
        }
        let arena_quads =
            (limits.max_storage_buffer_binding_size / 8).min(u64::from(MAX_ARENA_QUADS)) as u32;
        let mesher = CubeMesher::new(
            device,
            queue,
            backend.pool.buffer(),
            arena_quads,
            states as u32,
            air,
        )?;
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
        let node_buffer = buffer(
            "Cube far nodes",
            u64::from(MAX_NODES) * (NODE_WORDS as u64) * 4,
            st | dst,
        );
        let instances = buffer("Cube far instances", u64::from(MAX_NODES) * 32, st);
        let biomes = buffer("Cube far biomes", 256 * 16, st | dst);
        let args_bytes = u64::from(MAX_NODES) * 6 * 16;
        let indirect = wgpu::BufferUsages::INDIRECT;
        let solid_args = buffer("Cube far solid draws", args_bytes, st | indirect | dst);
        let translucent_args = buffer("Cube far water draws", args_bytes, st | indirect | dst);
        let draw_counts = buffer("Cube far draw counts", 8, st | indirect | dst);
        let frame_buffer = buffer(
            "Cube far frame",
            std::mem::size_of::<FrameUniform>() as u64,
            wgpu::BufferUsages::UNIFORM | dst,
        );

        // Culling.
        let cull_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Cube far culling"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("cube_cull.wgsl"))),
        });
        let compute_entry = |binding, ty| wgpu::BindGroupLayoutEntry {
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
        let cull_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Cube far culling"),
            entries: &[
                compute_entry(
                    0,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                compute_entry(1, storage(true)),
                compute_entry(2, storage(false)),
                compute_entry(3, storage(false)),
                compute_entry(4, storage(false)),
                compute_entry(5, storage(false)),
            ],
        });
        let cull = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Cube far culling"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Cube far culling"),
                    bind_group_layouts: &[Some(&cull_layout)],
                    ..Default::default()
                }),
            ),
            module: &cull_module,
            entry_point: Some("cull_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
            wgpu::BindGroupEntry {
                binding,
                resource: buffer.as_entire_binding(),
            }
        }
        let cull_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Cube far culling"),
            layout: &cull_layout,
            entries: &[
                entry(0, &frame_buffer),
                entry(1, &node_buffer),
                entry(2, &instances),
                entry(3, &solid_args),
                entry(4, &translucent_args),
                entry(5, &draw_counts),
            ],
        });

        // Drawing: group 0 is the terrain's, group 1 the pulled buffers.
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Cube far terrain"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(shader_source())),
        });
        let vertex_storage = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: storage(true),
            count: None,
        };
        let draw_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Cube far draws"),
            entries: &[
                vertex_storage(30),
                vertex_storage(31),
                vertex_storage(32),
                vertex_storage(33),
            ],
        });
        let draw_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Cube far draws"),
            layout: &draw_layout,
            entries: &[
                entry(30, mesher.arena()),
                entry(31, &instances),
                entry(32, mesher.materials()),
                entry(33, &biomes),
            ],
        });
        let draw_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Cube far terrain"),
            bind_group_layouts: &[Some(setup.bindings), Some(&draw_layout)],
            ..Default::default()
        });
        let constants = [
            ("SPRITE_ARRAYS", setup.sprite_constants[0]),
            ("SPRITE_SPLIT", setup.sprite_constants[1]),
        ];
        let pipeline = |label: &str, fragment: &str, blend: Option<wgpu::BlendState>| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&draw_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_cube_quads"),
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
                    entry_point: Some(fragment),
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
        let solid = pipeline("Cube far solid", "fs_solid", None);
        let translucent = pipeline(
            "Cube far water",
            "fs_translucent",
            Some(wgpu::BlendState::ALPHA_BLENDING),
        );
        if let Some(error) = pollster::block_on(validation.pop()) {
            log::warn!("Horizon renderer: far terrain not drawn on the GPU ({error})");
            return None;
        }
        Some(Self {
            device: device.clone(),
            queue: queue.clone(),
            mesher,
            pool: backend.pool.clone(),
            table: MaterialTable::new(states),
            air,
            node_buffer,
            biomes,
            solid_args,
            translucent_args,
            draw_counts,
            frame_buffer,
            cull,
            cull_group,
            draw_group,
            solid,
            translucent,
            counts: setup.counts,
            nodes: FxHashMap::default(),
            free_ids: Vec::new(),
            next_id: 0,
            pending: FxHashMap::default(),
            order: VecDeque::new(),
            removed: FxHashSet::default(),
            discard: false,
            epoch: Instant::now(),
            stats: FarStats::default(),
        })
    }

    /// The device the nodes must be generated on.
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// The material table (`cube_far.wgsl`'s `cf_materials`), for other
    /// far-terrain draws that share it (`gfx::surface_tile`).
    pub fn material_buffer(&self) -> &wgpu::Buffer {
        self.mesher.materials()
    }

    /// The biome colour table (`cf_biomes`).
    pub fn biome_buffer(&self) -> &wgpu::Buffer {
        &self.biomes
    }

    fn now_ms(&self) -> u32 {
        (self.epoch.elapsed().as_millis() as u32).min(NOT_SET - 1)
    }

    /// Stores material records (and the biome colours, when they changed).
    pub fn set_materials(
        &mut self,
        records: &[(u32, [u32; MATERIAL_STRIDE])],
        biomes: Option<&[u32]>,
    ) {
        for (state, record) in records {
            self.table.set(*state, record);
        }
        self.mesher.write_materials(records);
        if let Some(biomes) = biomes {
            let n = biomes.len().min(256 * 4);
            self.queue
                .write_buffer(&self.biomes, 0, bytemuck::cast_slice(&biomes[..n]));
        }
    }

    /// Takes in nodes to mesh and the keys to retire once they are in.
    pub fn queue(&mut self, nodes: Vec<GpuNode>, removed: Vec<CubeKey>) {
        for node in nodes {
            self.removed.remove(&node.key);
            let key = node.key;
            if let Some(old) = self.pending.insert(key, node) {
                self.release(&old);
            } else {
                self.order.push_back(key);
            }
        }
        for key in removed {
            if let Some(old) = self.pending.remove(&key) {
                self.release(&old);
            }
            self.removed.insert(key);
        }
    }

    fn release(&self, node: &GpuNode) {
        if let GpuContent::Slot(slot) = node.content {
            self.pool.free(slot);
        }
    }

    fn write_node(&self, id: u32, words: &[u32; NODE_WORDS]) {
        self.queue.write_buffer(
            &self.node_buffer,
            u64::from(id) * (NODE_WORDS as u64) * 4,
            bytemuck::cast_slice(words),
        );
    }

    fn word(&self, id: u32, word: usize, value: u32) {
        self.queue.write_buffer(
            &self.node_buffer,
            (u64::from(id) * NODE_WORDS as u64 + word as u64) * 4,
            bytemuck::bytes_of(&value),
        );
    }

    fn take_id(&mut self) -> Option<u32> {
        self.free_ids.pop().or_else(|| {
            (self.next_id < MAX_NODES).then(|| {
                self.next_id += 1;
                self.next_id - 1
            })
        })
    }

    /// Puts a meshed (or CPU-made) node in the table, replacing the key's
    /// old one (which keeps its fade state).
    fn install(&mut self, key: CubeKey, start: u32, total: u32, ranges: [(u32, u32); 12]) {
        let shown = self.nodes.remove(&key).and_then(|old| {
            self.drop_held(&old);
            old.shown.filter(|_| old.dying.is_none())
        });
        let id = if total == 0 {
            None
        } else {
            match self.take_id() {
                Some(id) => Some(id),
                None => {
                    self.mesher.free(start, total);
                    self.stats.failed += 1;
                    return;
                }
            }
        };
        if let Some(id) = id {
            let mut words = [0u32; NODE_WORDS];
            let o = key.origin();
            words[0] = o[0] as u32;
            words[1] = o[1] as u32;
            words[2] = o[2] as u32;
            words[3] = u32::from(key.level);
            words[4] = shown.unwrap_or(NOT_SET);
            words[5] = NOT_SET;
            words[6] = 1;
            for (k, (first, count)) in ranges.iter().enumerate() {
                words[8 + 2 * k] = *first;
                words[9 + 2 * k] = *count;
            }
            self.write_node(id, &words);
        }
        self.nodes.insert(
            key,
            Held {
                id,
                start,
                total,
                shown,
                dying: None,
            },
        );
    }

    fn drop_held(&mut self, held: &Held) {
        if let Some(id) = held.id {
            self.word(id, 6, 0);
            self.free_ids.push(id);
        }
        self.mesher.free(held.start, held.total);
    }

    fn remove(&mut self, key: CubeKey) {
        if let Some(held) = self.nodes.remove(&key) {
            self.drop_held(&held);
        }
    }

    /// A uniform node needs no GPU work: air and fluid draw nothing, rock its
    /// six faces, written by the CPU.
    fn install_uniform(&mut self, key: CubeKey, state: u32, biome: u16) {
        self.stats.uniform += 1;
        let (solid, translucent) = if state == self.air {
            (Vec::new(), Vec::new())
        } else {
            uniform_quads(&self.table, state, biome.min(255) as u8)
        };
        // One range per (dir, pass), solid quads first in dir order.
        let mut all: Vec<[u32; 2]> = Vec::new();
        let mut ranges = [(0u32, 0u32); 12];
        let mut by_dir: [(Vec<[u32; 2]>, Vec<[u32; 2]>); 6] = Default::default();
        for quad in solid {
            by_dir[(quad[0] >> 20 & 7) as usize].0.push(quad);
        }
        for (dir, quad) in translucent {
            by_dir[dir as usize].1.push(quad);
        }
        let mut counts = [0u32; 12];
        for (dir, (s, t)) in by_dir.iter().enumerate() {
            counts[dir * 2] = s.len() as u32;
            counts[dir * 2 + 1] = t.len() as u32;
            all.extend(s);
            all.extend(t);
        }
        let total = all.len() as u32;
        let Some(first) = self.mesher.write_quads(&all) else {
            self.stats.failed += 1;
            return;
        };
        let mut at = first;
        for k in 0..12 {
            ranges[k] = (at, counts[k]);
            at += counts[k];
        }
        // The order written is dir 0 solid, dir 0 translucent, dir 1 ...: matches `k`.
        self.install(key, first, total, ranges);
    }

    /// Polls the mesher, starts the next batch, and fades the nodes in and
    /// out once nothing waits. Call every frame.
    pub fn upload(&mut self) {
        let now = self.now_ms();
        let faded: Vec<CubeKey> = self
            .nodes
            .iter()
            .filter(|(_, n)| n.dying.is_some_and(|d| now.saturating_sub(d) >= FADE_MS))
            .map(|(&k, _)| k)
            .collect();
        for key in faded {
            self.remove(key);
        }
        if let Some(meshed) = self.mesher.poll() {
            for Meshed {
                key,
                slot,
                start,
                ranges,
                failed,
                ..
            } in meshed
            {
                // The emit pass is submitted: the slot can be regenerated.
                self.pool.free(slot);
                if self.discard {
                    if !failed {
                        self.mesher.free(start, ranges.iter().map(|r| r.1).sum());
                    }
                    continue;
                }
                if failed {
                    self.stats.failed += 1;
                    continue;
                }
                self.stats.meshed += 1;
                let total = ranges.iter().map(|r| r.1).sum();
                self.install(key, start, total, ranges);
            }
        }
        if self.mesher.idle() {
            self.discard = false;
        }
        if !self.pending.is_empty() {
            let idle = self.mesher.idle();
            let mut jobs = Vec::new();
            let mut keep = VecDeque::new();
            while let Some(key) = self.order.pop_front() {
                let Some(node) = self.pending.remove(&key) else {
                    continue;
                };
                match node.content {
                    // Uniform nodes need no mesher: they go in at once.
                    GpuContent::Uniform(voxel) => {
                        self.install_uniform(key, voxel.state.0, voxel.biome)
                    }
                    GpuContent::Slot(slot) if idle && jobs.len() < MAX_JOBS => jobs.push(MeshJob {
                        key,
                        slot,
                        cover: node.cover.as_deref().map(cover_words),
                        neighbours: [NONE; 6],
                    }),
                    GpuContent::Slot(_) => {
                        self.pending.insert(key, node);
                        keep.push_back(key);
                    }
                }
            }
            self.order = keep;
            if !jobs.is_empty() {
                self.mesher.begin(jobs);
            }
        }
        if self.pending.is_empty() && self.mesher.idle() {
            let now = self.now_ms();
            let shown: Vec<(CubeKey, u32)> = self
                .nodes
                .iter_mut()
                .filter(|(_, n)| n.shown.is_none())
                .filter_map(|(&k, n)| {
                    n.shown = Some(now);
                    n.id.map(|id| (k, id))
                })
                .collect();
            for (_, id) in shown {
                self.word(id, 4, now);
            }
            for key in std::mem::take(&mut self.removed) {
                let Some(held) = self.nodes.get_mut(&key) else {
                    continue;
                };
                if held.dying.is_some() {
                    continue;
                }
                held.dying = Some(now);
                match held.id {
                    Some(id) => self.word(id, 5, now),
                    // Nothing drawn: nothing to fade.
                    None => self.remove(key),
                }
            }
        }
    }

    /// Whether `key` is meshed and in the table (drawn or known empty).
    pub fn has(&self, key: CubeKey) -> bool {
        self.nodes.contains_key(&key)
    }

    /// The counters.
    pub fn stats(&self) -> FarStats {
        let free = self.pool.free_slots() as u64;
        FarStats {
            nodes: self.nodes.len(),
            pending: self.pending.len(),
            arena_bytes: self.mesher.arena_used_bytes(),
            arena_capacity_bytes: u64::from(self.mesher.arena_capacity_quads()) * 8,
            pool_bytes: (u64::from(self.pool.capacity()) - free) * SLOT_BYTES,
            meshing: !self.mesher.idle(),
            ..self.stats
        }
    }

    /// Forgets every node and what waits.
    pub fn clear(&mut self) {
        for (_, node) in std::mem::take(&mut self.pending) {
            self.release(&node);
        }
        self.order.clear();
        self.removed.clear();
        self.discard = !self.mesher.idle();
        let keys: Vec<CubeKey> = self.nodes.keys().copied().collect();
        for key in keys {
            self.remove(key);
        }
    }

    /// The culling pass for this frame: `planes` the camera-relative frustum.
    pub fn prepare(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        camera: DVec3,
        planes: [Vec4; 6],
    ) {
        if self.next_id == 0 {
            return;
        }
        let floor = camera.floor();
        let frame = FrameUniform {
            planes: planes.map(|p| p.to_array()),
            cam_int: [floor.x as i32, floor.y as i32, floor.z as i32, 0],
            cam_frac: [
                (camera.x - floor.x) as f32,
                (camera.y - floor.y) as f32,
                (camera.z - floor.z) as f32,
                0.0,
            ],
            now: self.now_ms(),
            nodes: self.next_id,
            pad: [0; 2],
        };
        self.queue
            .write_buffer(&self.frame_buffer, 0, bytemuck::bytes_of(&frame));
        encoder.clear_buffer(&self.draw_counts, 0, None);
        if !self.counts {
            // Draws past the count are empty ones.
            let bytes = u64::from(self.next_id) * 6 * 16;
            encoder.clear_buffer(&self.solid_args, 0, Some(bytes));
            encoder.clear_buffer(&self.translucent_args, 0, Some(bytes));
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Cube far culling"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.cull);
        pass.set_bind_group(0, &self.cull_group, &[]);
        pass.dispatch_workgroups(self.next_id.div_ceil(64), 1, 1);
    }

    /// Draws the solid nodes (`water` false) or the water of the nodes, in the
    /// world pass with the terrain's group 0 bound.
    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>, water: bool) {
        if self.next_id == 0 || self.nodes.is_empty() {
            return;
        }
        pass.set_pipeline(if water {
            &self.translucent
        } else {
            &self.solid
        });
        pass.set_bind_group(1, &self.draw_group, &[]);
        let args = if water {
            &self.translucent_args
        } else {
            &self.solid_args
        };
        let max = self.next_id * 6;
        if self.counts {
            pass.multi_draw_indirect_count(args, 0, &self.draw_counts, u64::from(water) * 4, max);
        } else {
            pass.multi_draw_indirect(args, 0, max);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_drawing_shader_validates_with_the_terrain_shader() {
        let module = naga::front::wgsl::parse_str(&shader_source()).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
    }

    #[test]
    fn the_culling_shader_validates() {
        let module = naga::front::wgsl::parse_str(include_str!("cube_cull.wgsl")).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
    }

    #[test]
    fn the_frame_uniform_matches_the_shaders_layout() {
        // 6 planes, then vec4<i32> and vec4<f32> on 16-byte boundaries, then 4 words.
        assert_eq!(std::mem::size_of::<FrameUniform>(), 96 + 16 + 16 + 16);
        assert_eq!(NODE_WORDS, 8 + 12 * 2);
    }
}
