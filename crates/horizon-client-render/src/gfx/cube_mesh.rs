//! The GPU mesher of far cube nodes: pool slots in, compact quads in a GPU
//! geometry arena out. See `cube_far.rs` for the node table, culling and draws.
//!
//! # Quad format (8 bytes, two `u32`)
//!
//! ```text
//! word 0   bits 0..3   x     the quad's lowest voxel in the node (0..15)
//!          bits 4..7   y
//!          bits 8..11  z
//!          bits 12..15 w-1   extent in voxels along the face's first axis
//!          bits 16..19 h-1   extent along its second axis
//!          bits 20..22 dir   0 Down, 1 Up, 2 North, 3 South, 4 West, 5 East
//!                            (`horizon_assets::model::Direction` order)
//!          bit  23     shaded   sky light 4 (a roof over it), else 15
//!          bits 24     overlay  which of the block face's two textures
//!          bits 25..31 spare
//! word 1   bits 0..16  block state id
//!          bits 17..24 biome id (0 for faces with no biome tint)
//!          bits 25..31 spare
//! ```
//!
//! The first axis is x for Down/Up/North/South and z for West/East; the second
//! is z for Down/Up and y for the rest. The face lies on the voxel's own plane
//! pushed by one voxel for Up/South/East. Everything else (texture layer, tint,
//! translucency, emission) comes from the material table at draw time, so a
//! quad does not change when the table does. A merged quad tiles its texture
//! once per block (UVs in blocks from its corner, as `lod::mesh::emit_quad`).
//!
//! # Material table
//!
//! A storage buffer of [`MATERIAL_STRIDE`] words per block state id: word 0 the
//! header (bit 0 known, bit 1 opaque, bits 2..3 fluid, bits 4..11 HDR tier),
//! then per face slot `dir * 2 + overlay` two words: `layer | tint << 16 |
//! translucent << 19 | present << 20` and the constant tint colour. It is built
//! from `lod::mesh::far_style`, the lookup the CPU mesher uses.
//!
//! # Passes
//!
//! `count_main` counts the quads of every (job, direction) into
//! `counts[job * 12 + dir * 2 + translucent]`; the CPU reads the counts back
//! (48 bytes a node), takes one contiguous range per node from [`RangeAlloc`]
//! (the prefix sum over the node's twelve counts), and `emit_main` writes the
//! quads there. Voxels never leave the GPU.
use crate::{
    lod::{
        cube::CubeKey,
        mesh::{self, Styles, Tint},
    },
    mesher::OPAQUE_LAYER,
    streaming::WorkerContext,
};
use horizon_assets::model::Direction;
use horizon_world::registry::BlockStateId;
use std::{
    borrow::Cow,
    collections::BTreeMap,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

/// Words per block state in the material table.
pub const MATERIAL_STRIDE: usize = 32;
/// Nodes one batch meshes.
pub const MAX_JOBS: usize = 512;
/// Words per job in the jobs buffer.
pub const JOB_WORDS: usize = 24;
/// "No neighbour" in a job.
pub const NONE: u32 = u32::MAX;
/// The largest block state id a quad holds.
pub const MAX_STATE: u32 = (1 << 17) - 1;

// ------------------------------------------------------------------- quads

/// A quad, unpacked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Quad {
    pub x: u8,
    pub y: u8,
    pub z: u8,
    /// Extents in voxels, 1 to 16.
    pub w: u8,
    pub h: u8,
    pub dir: u8,
    pub shaded: bool,
    pub overlay: u8,
    pub state: u32,
    pub biome: u8,
}

impl Quad {
    pub fn pack(self) -> [u32; 2] {
        debug_assert!(self.x < 16 && self.y < 16 && self.z < 16);
        debug_assert!((1..=16).contains(&self.w) && (1..=16).contains(&self.h));
        debug_assert!(self.dir < 6 && self.overlay < 2 && self.state <= MAX_STATE);
        [
            u32::from(self.x)
                | u32::from(self.y) << 4
                | u32::from(self.z) << 8
                | u32::from(self.w - 1) << 12
                | u32::from(self.h - 1) << 16
                | u32::from(self.dir) << 20
                | u32::from(self.shaded) << 23
                | u32::from(self.overlay) << 24,
            self.state | u32::from(self.biome) << 17,
        ]
    }
    pub fn unpack(words: [u32; 2]) -> Self {
        let [a, b] = words;
        Self {
            x: (a & 15) as u8,
            y: (a >> 4 & 15) as u8,
            z: (a >> 8 & 15) as u8,
            w: (a >> 12 & 15) as u8 + 1,
            h: (a >> 16 & 15) as u8 + 1,
            dir: (a >> 20 & 7) as u8,
            shaded: a >> 23 & 1 == 1,
            overlay: (a >> 24 & 1) as u8,
            state: b & MAX_STATE,
            biome: (b >> 17 & 255) as u8,
        }
    }
}

// ---------------------------------------------------------- material table

/// `Tint` as the table holds it.
fn tint_code(tint: Tint) -> (u32, u32) {
    match tint {
        Tint::None => (0, 0),
        Tint::Grass => (1, 0),
        Tint::Foliage => (2, 0),
        Tint::Dry => (3, 0),
        Tint::Water => (4, 0),
        Tint::Constant(c) => (5, c),
    }
}

/// The record of a block state from its six directions' styles (in `dir`
/// order) and HDR tier.
pub(crate) fn encode_styles(styles: &[Styles; 6], tier: u8) -> [u32; MATERIAL_STRIDE] {
    let mut out = [0u32; MATERIAL_STRIDE];
    let first = styles[0];
    out[0] =
        1 | u32::from(first.opaque) << 1 | u32::from(first.fluid & 3) << 2 | u32::from(tier) << 4;
    for (dir, style) in styles.iter().enumerate() {
        for (overlay, face) in style.faces.iter().enumerate() {
            let Some(face) = face else { continue };
            let (tint, constant) = tint_code(face.tint);
            let slot = 1 + 2 * (dir * 2 + overlay);
            out[slot] =
                u32::from(face.layer) | tint << 16 | u32::from(face.translucent) << 19 | 1 << 20;
            out[slot + 1] = constant;
        }
    }
    out
}

/// A face slot of a record, as `lod::mesh::Style` (`None`: not present).
pub(crate) fn decode_face(
    record: &[u32; MATERIAL_STRIDE],
    dir: usize,
    overlay: usize,
) -> Option<mesh::Style> {
    let slot = 1 + 2 * (dir * 2 + overlay);
    let word = record[slot];
    if word >> 20 & 1 == 0 {
        return None;
    }
    Some(mesh::Style {
        layer: (word & 0xffff) as u16,
        tint: match word >> 16 & 7 {
            1 => Tint::Grass,
            2 => Tint::Foliage,
            3 => Tint::Dry,
            4 => Tint::Water,
            5 => Tint::Constant(record[slot + 1]),
            _ => Tint::None,
        },
        translucent: word >> 19 & 1 == 1,
    })
}

/// The record of `state`, from the shared far-style lookup.
pub fn material_record(context: &WorkerContext, state: u32) -> [u32; MATERIAL_STRIDE] {
    let id = BlockStateId(state);
    let styles: [Styles; 6] =
        std::array::from_fn(|d| mesh::far_style(context, id, Direction::ALL[d]));
    encode_styles(&styles, context.tables.hdr_tier(id))
}

/// Per biome id, four words: grass, foliage, dry foliage and water colours.
/// Grass takes the colour at the origin: the swamp's noise variation across a
/// biome is not drawn on the GPU.
pub fn biome_table(context: &WorkerContext) -> Vec<u32> {
    let count = context.biomes.list.len().min(256);
    let mut out = Vec::with_capacity(count * 4);
    for id in 0..count {
        let info = context.biomes.get(id as u16);
        out.extend([
            context.biomes.grass(id as u16, 0.0, 0.0),
            info.foliage,
            info.dry_foliage,
            info.water,
        ]);
    }
    out
}

/// The CPU copy of the material table, for the faces the CPU writes itself
/// (uniform nodes) and for the buffer's contents.
pub struct MaterialTable {
    words: Vec<u32>,
}

impl MaterialTable {
    pub fn new(states: usize) -> Self {
        Self {
            words: vec![0; states * MATERIAL_STRIDE],
        }
    }
    pub fn states(&self) -> usize {
        self.words.len() / MATERIAL_STRIDE
    }
    /// Stores a record; false when `state` is past the table.
    pub fn set(&mut self, state: u32, record: &[u32; MATERIAL_STRIDE]) -> bool {
        let at = state as usize * MATERIAL_STRIDE;
        match self.words.get_mut(at..at + MATERIAL_STRIDE) {
            Some(slot) => {
                slot.copy_from_slice(record);
                true
            }
            None => false,
        }
    }
    pub fn record(&self, state: u32) -> Option<[u32; MATERIAL_STRIDE]> {
        let at = state as usize * MATERIAL_STRIDE;
        self.words.get(at..at + MATERIAL_STRIDE)?.try_into().ok()
    }
    pub fn opaque(&self, state: u32) -> bool {
        self.record(state).is_some_and(|r| r[0] >> 1 & 1 == 1)
    }
    pub fn fluid(&self, state: u32) -> u32 {
        self.record(state).map_or(0, |r| r[0] >> 2 & 3)
    }
}

// ------------------------------------------------------------------- arena

/// First-fit allocator of quad ranges in the geometry arena, with coalescing.
#[derive(Debug)]
pub struct RangeAlloc {
    free: BTreeMap<u32, u32>,
    capacity: u32,
    used: u32,
}

impl RangeAlloc {
    pub fn new(capacity: u32) -> Self {
        Self {
            free: BTreeMap::from([(0, capacity)]),
            capacity,
            used: 0,
        }
    }
    pub fn capacity(&self) -> u32 {
        self.capacity
    }
    pub fn used(&self) -> u32 {
        self.used
    }
    /// A range of `len` quads (0: nothing, at 0).
    pub fn alloc(&mut self, len: u32) -> Option<u32> {
        if len == 0 {
            return Some(0);
        }
        let (&start, &size) = self.free.iter().find(|&(_, &size)| size >= len)?;
        self.free.remove(&start);
        if size > len {
            self.free.insert(start + len, size - len);
        }
        self.used += len;
        Some(start)
    }
    pub fn free(&mut self, start: u32, len: u32) {
        if len == 0 {
            return;
        }
        debug_assert!(start + len <= self.capacity && self.used >= len);
        self.used -= len;
        let mut start = start;
        let mut len = len;
        if let Some((&before, &size)) = self.free.range(..start).next_back()
            && before + size == start
        {
            self.free.remove(&before);
            start = before;
            len += size;
        }
        if let Some(&size) = self.free.get(&(start + len)) {
            self.free.remove(&(start + len));
            len += size;
        }
        self.free.insert(start, len);
    }
    /// The longest free range.
    pub fn largest_free(&self) -> u32 {
        self.free.values().copied().max().unwrap_or(0)
    }
}

// -------------------------------------------------------------------- jobs

/// A node to mesh.
#[derive(Clone, Debug)]
pub struct MeshJob {
    pub key: CubeKey,
    pub slot: u32,
    /// The columns the sky is blocked over: bit `z << 4 | x`.
    pub cover: Option<[u32; 8]>,
    /// Slots of the same-level neighbours in the pool, in `dir` order, whose
    /// voxels hide this node's border faces; [`NONE`] draws the border as a
    /// wall, as `lod::mesh::mesh_cube` does.
    pub neighbours: [u32; 6],
}

impl MeshJob {
    pub fn words(&self) -> [u32; JOB_WORDS] {
        let mut w = [0u32; JOB_WORDS];
        w[0] = self.slot;
        w[1] = u32::from(self.key.level);
        w[2] = u32::from(self.cover.is_some());
        w[4..10].copy_from_slice(&self.neighbours);
        if let Some(cover) = self.cover {
            w[12..20].copy_from_slice(&cover);
        }
        w
    }
}

/// The cover mask (`lod::cube::CoverMask`) as the shader's words.
pub fn cover_words(mask: &[u64; 4]) -> [u32; 8] {
    std::array::from_fn(|i| (mask[i / 2] >> (32 * (i % 2))) as u32)
}

/// A meshed node: per (dir, translucent) `(first quad, quads)`, index
/// `dir * 2 + translucent`. `failed`: the arena had no room.
#[derive(Clone, Debug)]
pub struct Meshed {
    pub key: CubeKey,
    pub slot: u32,
    /// The node's whole arena range starts here ( quads).
    pub start: u32,
    pub ranges: [(u32, u32); 12],
    pub failed: bool,
}

impl Meshed {
    pub fn total(&self) -> u32 {
        self.ranges.iter().map(|r| r.1).sum()
    }
    /// The node's whole range in the arena.
    pub fn span(&self) -> (u32, u32) {
        (self.start, self.total())
    }
}

// ----------------------------------------------------------------- mesher

/// `cube_mesh.wgsl`'s `Params`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    air: u32,
    state_count: u32,
    jobs: u32,
    pad: u32,
}

struct InFlight {
    jobs: Vec<MeshJob>,
    ready: Arc<AtomicBool>,
}

/// The compute pipelines, their buffers and the quad arena.
pub struct CubeMesher {
    device: wgpu::Device,
    queue: wgpu::Queue,
    count: wgpu::ComputePipeline,
    emit: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    params: wgpu::Buffer,
    materials: wgpu::Buffer,
    jobs: wgpu::Buffer,
    counts: wgpu::Buffer,
    staging: wgpu::Buffer,
    offsets: wgpu::Buffer,
    arena: wgpu::Buffer,
    ranges: RangeAlloc,
    state_count: u32,
    air: u32,
    inflight: Option<InFlight>,
}

impl CubeMesher {
    /// The mesher reading `pool` (the node pool's buffer), with room for
    /// `arena_quads` quads and a material table of `states` block states, or
    /// `None` when the device cannot hold them or run the shader.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pool: &wgpu::Buffer,
        arena_quads: u32,
        states: u32,
        air: u32,
    ) -> Option<Self> {
        let limit = device.limits().max_storage_buffer_binding_size;
        let materials_bytes = u64::from(states) * (MATERIAL_STRIDE as u64) * 4;
        let arena_bytes = u64::from(arena_quads) * 8;
        if materials_bytes > limit
            || arena_bytes > limit
            || pool.size() > limit
            || device.limits().max_storage_buffers_per_shader_stage < 6
            || states == 0
        {
            return None;
        }
        let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Cube mesher"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("cube_mesh.wgsl"))),
        });
        let storage = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Cube mesher"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage(1, true),
                storage(2, true),
                storage(3, true),
                storage(4, false),
                storage(5, true),
                storage(6, false),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Cube mesher"),
            bind_group_layouts: &[Some(&layout)],
            ..Default::default()
        });
        let pipeline = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let count = pipeline("count_main");
        let emit = pipeline("emit_main");
        let buffer = |label, size: u64, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let dst = wgpu::BufferUsages::COPY_DST;
        let st = wgpu::BufferUsages::STORAGE;
        let params = buffer(
            "Cube mesher params",
            std::mem::size_of::<Params>() as u64,
            wgpu::BufferUsages::UNIFORM | dst,
        );
        let materials = buffer("Cube materials", materials_bytes, st | dst);
        let jobs = buffer(
            "Cube mesh jobs",
            (MAX_JOBS * JOB_WORDS * 4) as u64,
            st | dst,
        );
        let counts = buffer(
            "Cube mesh counts",
            (MAX_JOBS * 12 * 4) as u64,
            st | wgpu::BufferUsages::COPY_SRC,
        );
        let staging = buffer(
            "Cube mesh counts readback",
            (MAX_JOBS * 12 * 4) as u64,
            wgpu::BufferUsages::MAP_READ | dst,
        );
        let offsets = buffer("Cube mesh offsets", (MAX_JOBS * 12 * 4) as u64, st | dst);
        let arena = buffer("Cube quad arena", arena_bytes, st | dst);
        fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
            wgpu::BindGroupEntry {
                binding,
                resource: buffer.as_entire_binding(),
            }
        }
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Cube mesher"),
            layout: &layout,
            entries: &[
                entry(0, &params),
                entry(1, pool),
                entry(2, &materials),
                entry(3, &jobs),
                entry(4, &counts),
                entry(5, &offsets),
                entry(6, &arena),
            ],
        });
        if let Some(error) = pollster::block_on(validation.pop()) {
            log::warn!("Horizon renderer: far terrain meshed on the CPU ({error})");
            return None;
        }
        Some(Self {
            device: device.clone(),
            queue: queue.clone(),
            count,
            emit,
            bind_group,
            params,
            materials,
            jobs,
            counts,
            staging,
            offsets,
            arena,
            ranges: RangeAlloc::new(arena_quads),
            state_count: states,
            air,
            inflight: None,
        })
    }

    /// The quad arena (storage; the vertex shader reads it).
    pub fn arena(&self) -> &wgpu::Buffer {
        &self.arena
    }
    /// The material table buffer.
    pub fn materials(&self) -> &wgpu::Buffer {
        &self.materials
    }
    pub fn arena_used_bytes(&self) -> u64 {
        u64::from(self.ranges.used()) * 8
    }
    pub fn arena_capacity_quads(&self) -> u32 {
        self.ranges.capacity()
    }
    pub fn state_count(&self) -> u32 {
        self.state_count
    }

    /// Writes the material records of `states` into the table.
    pub fn write_materials(&self, records: &[(u32, [u32; MATERIAL_STRIDE])]) {
        for (state, record) in records {
            if *state < self.state_count {
                self.queue.write_buffer(
                    &self.materials,
                    u64::from(*state) * (MATERIAL_STRIDE as u64) * 4,
                    bytemuck::cast_slice(record),
                );
            }
        }
    }

    /// Whether a batch can begin (one is in flight at a time).
    pub fn idle(&self) -> bool {
        self.inflight.is_none()
    }

    /// Counts the quads of `jobs` (at most [`MAX_JOBS`]): one submission. The
    /// result comes from [`CubeMesher::poll`].
    pub fn begin(&mut self, jobs: Vec<MeshJob>) {
        assert!(self.inflight.is_none() && jobs.len() <= MAX_JOBS);
        if jobs.is_empty() {
            return;
        }
        let words: Vec<u32> = jobs.iter().flat_map(MeshJob::words).collect();
        self.queue
            .write_buffer(&self.jobs, 0, bytemuck::cast_slice(&words));
        self.queue.write_buffer(
            &self.params,
            0,
            bytemuck::bytes_of(&Params {
                air: self.air,
                state_count: self.state_count,
                jobs: jobs.len() as u32,
                pad: 0,
            }),
        );
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Cube mesh count"),
            });
        self.dispatch(&mut encoder, &self.count, jobs.len());
        let bytes = (jobs.len() * 12 * 4) as u64;
        encoder.copy_buffer_to_buffer(&self.counts, 0, &self.staging, 0, bytes);
        self.queue.submit([encoder.finish()]);
        let ready = Arc::new(AtomicBool::new(false));
        let flag = ready.clone();
        self.staging
            .slice(..bytes)
            .map_async(wgpu::MapMode::Read, move |r| {
                flag.store(r.is_ok(), Ordering::Release);
            });
        self.inflight = Some(InFlight { jobs, ready });
    }

    fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        jobs: usize,
    ) {
        let groups = (jobs * 6) as u32;
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Cube mesher"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(groups.min(32768), groups.div_ceil(32768), 1);
    }

    /// If the counts of the batch in flight are back: takes its arena ranges,
    /// emits the quads (one submission) and returns the meshed nodes. The
    /// jobs' slots may be freed once this returns (the emit is submitted).
    pub fn poll(&mut self) -> Option<Vec<Meshed>> {
        self.inflight.as_ref()?;
        let _ = self.device.poll(wgpu::PollType::Poll);
        if !self.inflight.as_ref()?.ready.load(Ordering::Acquire) {
            return None;
        }
        let InFlight { jobs, .. } = self.inflight.take()?;
        let bytes = (jobs.len() * 12 * 4) as u64;
        let counts: Vec<u32> = {
            let view = self.staging.slice(..bytes).get_mapped_range().ok()?;
            bytemuck::cast_slice::<u8, u32>(&view).to_vec()
        };
        self.staging.unmap();
        let mut offsets = vec![NONE; jobs.len() * 12];
        let meshed: Vec<Meshed> = jobs
            .iter()
            .enumerate()
            .map(|(j, job)| {
                let c = &counts[j * 12..j * 12 + 12];
                let total: u32 = c.iter().sum();
                let mut ranges = [(0, 0); 12];
                let start = self.ranges.alloc(total);
                if let Some(start) = start {
                    let mut at = start;
                    for k in 0..12 {
                        ranges[k] = (at, c[k]);
                        offsets[j * 12 + k] = at;
                        at += c[k];
                    }
                }
                Meshed {
                    key: job.key,
                    slot: job.slot,
                    start: start.unwrap_or(0),
                    ranges,
                    failed: start.is_none(),
                }
            })
            .collect();
        self.queue
            .write_buffer(&self.offsets, 0, bytemuck::cast_slice(&offsets));
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Cube mesh emit"),
            });
        self.dispatch(&mut encoder, &self.emit, jobs.len());
        self.queue.submit([encoder.finish()]);
        Some(meshed)
    }

    /// Writes CPU-made quads into the arena (uniform nodes), returning their
    /// first index.
    pub fn write_quads(&mut self, quads: &[[u32; 2]]) -> Option<u32> {
        let start = self.ranges.alloc(quads.len() as u32)?;
        if !quads.is_empty() {
            self.queue.write_buffer(
                &self.arena,
                u64::from(start) * 8,
                bytemuck::cast_slice(quads),
            );
        }
        Some(start)
    }

    /// Returns `quads` quads at `start` to the arena.
    pub fn free(&mut self, start: u32, quads: u32) {
        self.ranges.free(start, quads);
    }
}

/// The quads of a uniform solid node with no cover: its six faces, whole.
/// `(solid, translucent)` lists by the table, as the shader would split them.
pub fn uniform_quads(
    table: &MaterialTable,
    state: u32,
    biome: u8,
) -> (Vec<[u32; 2]>, Vec<(u8, [u32; 2])>) {
    let mut solid = Vec::new();
    let mut translucent = Vec::new();
    let Some(record) = table.record(state) else {
        return (solid, translucent);
    };
    if table.fluid(state) != 0 {
        return (solid, translucent);
    }
    for dir in 0..6 {
        for overlay in 0..2 {
            let Some(face) = decode_face(&record, dir, overlay) else {
                continue;
            };
            let quad = Quad {
                x: 0,
                y: 0,
                z: 0,
                w: 16,
                h: 16,
                dir: dir as u8,
                shaded: false,
                overlay: overlay as u8,
                state,
                biome: if face.tint == Tint::None { 0 } else { biome },
            }
            .pack();
            if face.translucent {
                translucent.push((dir as u8, quad));
            } else {
                solid.push(quad);
            }
        }
    }
    (solid, translucent)
}

/// Opaque-layer flag helper for tests and the draw shader's expectations.
pub const fn is_opaque_layer(layer: u16) -> bool {
    layer & OPAQUE_LAYER != 0
}

/// The faces a node exposes, counted on the CPU with the shader's rules (the
/// reference the GPU count is checked against): voxels indexed `y << 8 | z << 4
/// | x`; `neighbours[dir]` the voxels of the same-level node on that side, if
/// known. Quads are counted unmerged (one per exposed face and overlay).
pub fn exposed_faces(
    states: &[u32],
    neighbours: [Option<&[u32]>; 6],
    table: &MaterialTable,
    air: u32,
) -> usize {
    let step: [[i32; 3]; 6] = [
        [0, -1, 0],
        [0, 1, 0],
        [0, 0, -1],
        [0, 0, 1],
        [-1, 0, 0],
        [1, 0, 0],
    ];
    let opaque = |s: u32| s != air && table.opaque(s);
    let mut n = 0;
    for y in 0..16i32 {
        for z in 0..16i32 {
            for x in 0..16i32 {
                let own = states[(y << 8 | z << 4 | x) as usize];
                if own == air {
                    continue;
                }
                let Some(record) = table.record(own) else {
                    continue;
                };
                for (dir, d) in step.iter().enumerate() {
                    let q = [x + d[0], y + d[1], z + d[2]];
                    let inside = q.iter().all(|c| (0..16).contains(c));
                    let next = if inside {
                        Some(states[(q[1] << 8 | q[2] << 4 | q[0]) as usize])
                    } else if table.fluid(own) != 0 {
                        continue;
                    } else {
                        neighbours[dir].map(|v| {
                            v[((q[1] & 15) << 8 | (q[2] & 15) << 4 | (q[0] & 15)) as usize]
                        })
                    };
                    if let Some(next) = next {
                        let f = table.fluid(next);
                        if next == own
                            || (next != air && (opaque(next) || (f != 0 && f == table.fluid(own))))
                        {
                            continue;
                        }
                    }
                    n += (0..2)
                        .filter(|&o| decode_face(&record, dir, o).is_some())
                        .count();
                }
            }
        }
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lod::mesh::{Style, Styles};

    #[test]
    fn quads_round_trip() {
        for dir in 0..6u8 {
            for (w, h) in [(1, 1), (16, 16), (3, 15), (16, 1)] {
                let q = Quad {
                    x: 15,
                    y: 0,
                    z: 7,
                    w,
                    h,
                    dir,
                    shaded: dir % 2 == 1,
                    overlay: dir % 2,
                    state: MAX_STATE - u32::from(dir),
                    biome: 255 - dir,
                };
                assert_eq!(Quad::unpack(q.pack()), q);
            }
        }
        // Fields do not overlap: setting one changes only its bits.
        let base = Quad {
            x: 0,
            y: 0,
            z: 0,
            w: 1,
            h: 1,
            dir: 0,
            shaded: false,
            overlay: 0,
            state: 0,
            biome: 0,
        };
        assert_eq!(base.pack(), [0, 0]);
        assert_eq!(Quad { dir: 5, ..base }.pack()[0], 5 << 20);
        assert_eq!(Quad { biome: 1, ..base }.pack()[1], 1 << 17);
    }

    fn styles() -> [Styles; 6] {
        let grass = |layer| Style {
            layer,
            tint: Tint::Grass,
            translucent: false,
        };
        std::array::from_fn(|d| Styles {
            faces: [
                Some(Style {
                    layer: (7 + d as u16) | OPAQUE_LAYER,
                    tint: Tint::None,
                    translucent: false,
                }),
                (d >= 2).then(|| grass(9)),
            ],
            fluid: 0,
            opaque: true,
        })
    }

    #[test]
    fn records_decode_to_the_styles_they_came_from() {
        let styles = styles();
        let record = encode_styles(&styles, 3);
        assert_eq!(record[0] & 1, 1);
        assert_eq!(record[0] >> 1 & 1, 1);
        assert_eq!(record[0] >> 4 & 255, 3);
        for d in 0..6 {
            for o in 0..2 {
                assert_eq!(decode_face(&record, d, o), styles[d].faces[o], "{d} {o}");
            }
        }
        let mut water = styles;
        water.fill(Styles {
            faces: [
                Some(Style {
                    layer: 4,
                    tint: Tint::Water,
                    translucent: true,
                }),
                None,
            ],
            fluid: 1,
            opaque: false,
        });
        let record = encode_styles(&water, 0);
        assert_eq!(record[0] >> 2 & 3, 1);
        assert_eq!(record[0] >> 1 & 1, 0);
        assert_eq!(decode_face(&record, 1, 0), water[1].faces[0]);
        let constant = Styles {
            faces: [
                Some(Style {
                    layer: 1,
                    tint: Tint::Constant(0x619961),
                    translucent: false,
                }),
                None,
            ],
            fluid: 0,
            opaque: true,
        };
        let record = encode_styles(&[constant; 6], 0);
        assert_eq!(decode_face(&record, 2, 0), constant.faces[0]);
    }

    #[test]
    fn the_table_matches_the_lookup_for_a_few_states() {
        let mut table = MaterialTable::new(8);
        let styles = styles();
        assert!(table.set(2, &encode_styles(&styles, 0)));
        assert!(!table.set(8, &encode_styles(&styles, 0)));
        assert!(table.opaque(2));
        assert!(!table.opaque(3), "an unknown state hides nothing");
        assert_eq!(table.fluid(2), 0);
        // A uniform solid node: the six faces, the grass overlay on its sides.
        let (solid, translucent) = uniform_quads(&table, 2, 5);
        assert_eq!(solid.len(), 6 + 4);
        assert!(translucent.is_empty());
        let overlay = Quad::unpack(solid[solid.len() - 1]);
        assert_eq!(
            (overlay.w, overlay.h, overlay.overlay, overlay.biome),
            (16, 16, 1, 5)
        );
        let base = Quad::unpack(solid[0]);
        assert_eq!(base.biome, 0, "an untinted face merges across biomes");
    }

    #[test]
    fn cover_words_follow_the_mask_bits() {
        let mut mask = [0u64; 4];
        for bit in [0usize, 31, 32, 63, 64, 255] {
            mask[bit / 64] |= 1 << (bit % 64);
        }
        let words = cover_words(&mask);
        for bit in 0..256 {
            let expect = mask[bit / 64] >> (bit % 64) & 1;
            assert_eq!(
                u64::from(words[bit / 32] >> (bit % 32) & 1),
                expect,
                "{bit}"
            );
        }
    }

    #[test]
    fn the_arena_gives_back_what_it_takes() {
        let mut arena = RangeAlloc::new(100);
        let a = arena.alloc(30).unwrap();
        let b = arena.alloc(30).unwrap();
        let c = arena.alloc(30).unwrap();
        assert_eq!((a, b, c), (0, 30, 60));
        assert_eq!(arena.alloc(11), None);
        assert_eq!(arena.used(), 90);
        arena.free(b, 30);
        assert_eq!(arena.alloc(31), None, "the hole is 30 wide");
        arena.free(a, 30);
        assert_eq!(arena.largest_free(), 60, "neighbouring holes coalesce");
        arena.free(c, 30);
        assert_eq!(arena.used(), 0);
        assert_eq!(arena.largest_free(), 100);
        assert_eq!(arena.alloc(100), Some(0));
        assert_eq!(arena.alloc(0), Some(0));
    }

    #[test]
    fn the_arena_survives_a_shuffled_workout() {
        let mut arena = RangeAlloc::new(4096);
        let mut live: Vec<(u32, u32)> = Vec::new();
        let mut seed = 12345u32;
        let mut next = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            seed >> 8
        };
        for _ in 0..2000 {
            if next() % 3 != 0 || live.is_empty() {
                let len = next() % 90 + 1;
                if let Some(start) = arena.alloc(len) {
                    for &(s, l) in &live {
                        assert!(start + len <= s || s + l <= start, "overlap");
                    }
                    live.push((start, len));
                }
            } else {
                let i = next() as usize % live.len();
                let (s, l) = live.swap_remove(i);
                arena.free(s, l);
            }
            assert_eq!(arena.used(), live.iter().map(|r| r.1).sum::<u32>());
        }
        for (s, l) in live {
            arena.free(s, l);
        }
        assert_eq!(arena.largest_free(), 4096);
    }

    #[test]
    fn exposed_faces_follow_the_meshers_rules() {
        let mut table = MaterialTable::new(8);
        table.set(1, &encode_styles(&styles(), 0));
        let mut water = styles();
        for s in &mut water {
            s.fluid = 1;
            s.opaque = false;
            s.faces = [
                Some(Style {
                    layer: 4,
                    tint: Tint::Water,
                    translucent: true,
                }),
                None,
            ];
        }
        table.set(2, &encode_styles(&water, 0));
        let air = 0;
        let mut full = vec![1u32; 4096];
        // A full node draws its six walls: 256 faces each, the overlay on four.
        assert_eq!(
            exposed_faces(&full, [None; 6], &table, air),
            6 * 256 + 4 * 256
        );
        // Same-level neighbours that hide its border leave only the inside.
        let solid = vec![1u32; 4096];
        let all = [Some(&solid[..]); 6];
        assert_eq!(exposed_faces(&full, all, &table, air), 0);
        // A hole in the middle shows its six inner faces to the neighbours around it.
        full[8 << 8 | 8 << 4 | 8] = air;
        assert_eq!(exposed_faces(&full, all, &table, air), 6 + 4);
        // Fluid edge faces are dropped; only its top meets air.
        let mut sea = vec![2u32; 4096];
        assert_eq!(exposed_faces(&sea, [None; 6], &table, air), 0);
        for z in 0..16 {
            for x in 0..16 {
                sea[15 << 8 | z << 4 | x] = air;
            }
        }
        assert_eq!(exposed_faces(&sea, [None; 6], &table, air), 256);
    }

    #[test]
    fn the_mesher_shader_validates() {
        let module = naga::front::wgsl::parse_str(include_str!("cube_mesh.wgsl")).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
    }

    fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = crate::gfx::test_gpu::instance();
        let adapter = pollster::block_on(instance.request_adapter(&Default::default())).ok()?;
        pollster::block_on(adapter.request_device(&Default::default())).ok()
    }

    /// The GPU's quad count for nodes against the CPU reference of exposed
    /// faces (merging only lowers it: the GPU count is at most the CPU's, and
    /// at least 1/256 of it, a whole 16 x 16 face as one quad).
    #[test]
    fn gpu_counts_agree_with_the_cpu_reference() {
        let Some((device, queue)) = device() else {
            eprintln!("no GPU adapter: skipped");
            return;
        };
        let pool = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test pool"),
            size: 4 * 4352 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let Some(mut mesher) = CubeMesher::new(&device, &queue, &pool, 1 << 16, 8, 0) else {
            eprintln!("no compute support: skipped");
            return;
        };
        let mut table = MaterialTable::new(8);
        let record = encode_styles(&styles(), 0);
        table.set(1, &record);
        mesher.write_materials(&[(1, record)]);
        // Node 0: solid ground under y = 6 with a pillar; node 1: a checker.
        let mut a = vec![0u32; 4352];
        let mut b = vec![0u32; 4352];
        for y in 0..16 {
            for z in 0..16 {
                for x in 0..16 {
                    let i = y << 8 | z << 4 | x;
                    a[i] = u32::from(y < 6 || (x == 4 && z == 9 && y < 12));
                    b[i] = u32::from((x + y + z) % 2 == 0);
                }
            }
        }
        for (slot, words) in [(0u64, &a), (1, &b)] {
            queue.write_buffer(&pool, slot * 4352 * 4, bytemuck::cast_slice(words));
        }
        let job = |slot| MeshJob {
            key: CubeKey::new(2, slot as i32, 0, 0),
            slot,
            cover: None,
            neighbours: [NONE; 6],
        };
        mesher.begin(vec![job(0), job(1)]);
        let mut meshed = None;
        for _ in 0..2000 {
            meshed = mesher.poll();
            if meshed.is_some() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let meshed = meshed.expect("the counts came back");
        for (m, voxels) in meshed.iter().zip([&a, &b]) {
            let reference = exposed_faces(&voxels[..4096], [None; 6], &table, 0);
            let total = m.total() as usize;
            assert!(!m.failed);
            assert!(
                total <= reference && total * 256 >= reference,
                "{total} vs {reference}"
            );
        }
        // The checker cannot merge at all: every exposed face is a quad.
        let reference = exposed_faces(&b[..4096], [None; 6], &table, 0);
        assert_eq!(meshed[1].total() as usize, reference);
    }
}
