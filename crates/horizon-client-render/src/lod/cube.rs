//! Far terrain of a cubic level: an octree of cube nodes, "chunks of chunks".
//!
//! A node [`CubeKey`] `{ level, x, y, z }` covers `(16 << level)³` blocks at
//! `[x, y, z] * (16 << level)` and holds a 16³ grid of voxels, each
//! `1 << level` blocks wide. Level 0 is one cube at full resolution. A parent
//! is built from its eight children ([`downsample`]) or sampled at its own
//! resolution by a [`CubeSampler`]. Nodes are meshed as closed shells
//! (`mesh::mesh_cube`), so no crack opens between neighbours of different
//! levels, and drawn through the far-tile pipeline.
//!
//! [`plan_classified`] chooses the nodes (pure, no threads): a node splits
//! while its voxels would span more than the Cubic Far Detail setting's pixels
//! on screen ([`FarView`], [`Rule`]), within a node budget. [`CubeLod`] plans
//! on a thread of its own, asks for a coarse coverage layer first after a big
//! change, queues the rest by direction (the view's first) and distance, and
//! samples and meshes the nodes in the background. [`node_kind`] is where far
//! nodes will become 2D height-map tiles.
use super::mesh::{self, Mesher};
use crate::{
    fast_hash::{FxHashMap, FxHashSet},
    gfx::{
        cube_descent::{DescentRequest, GpuDescent},
        cube_gen::{GpuBackend, MIN_LEVEL, slot_words},
        surface_tile::{TileInfo, TileKey},
    },
    mesher::SectionMesh,
    streaming::WorkerContext,
};
use horizon_core::position::SectionPos;
use horizon_world::registry::BlockStateId;
use std::{
    cell::RefCell,
    collections::VecDeque,
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};

/// The coarsest level: a node 67,108,864 blocks wide, wider than the world
/// (vanilla's border, +-30,000,000 blocks) and than the cube range vertically.
pub const MAX_CUBE_LEVEL: u8 = 22;
/// Voxels along one side of a node.
pub const SIDE: usize = 16;
/// Voxels in a node.
pub const VOLUME: usize = SIDE * SIDE * SIDE;

/// One node of the octree.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CubeKey {
    pub level: u8,
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl CubeKey {
    pub const fn new(level: u8, x: i32, y: i32, z: i32) -> Self {
        Self { level, x, y, z }
    }
    /// The node of `level` holding `cube`.
    pub const fn containing(level: u8, cube: SectionPos) -> Self {
        Self::new(level, cube.x >> level, cube.y >> level, cube.z >> level)
    }
    /// Blocks along one side.
    pub const fn size(self) -> i32 {
        16 << self.level
    }
    /// The node's lowest corner, in blocks.
    pub const fn origin(self) -> [i32; 3] {
        let size = self.size();
        [self.x * size, self.y * size, self.z * size]
    }
    /// The node one level up that holds this one (rounding toward negative
    /// infinity, so negative coordinates nest too).
    pub const fn parent(self) -> Self {
        Self::new(self.level + 1, self.x >> 1, self.y >> 1, self.z >> 1)
    }
    /// The eight nodes one level down. Index bit 0 is the X half, bit 1 the
    /// Z half, bit 2 the Y half.
    pub fn children(self) -> [Self; 8] {
        std::array::from_fn(|i| {
            Self::new(
                self.level - 1,
                self.x * 2 + (i & 1) as i32,
                self.y * 2 + (i >> 2) as i32,
                self.z * 2 + ((i >> 1) & 1) as i32,
            )
        })
    }
    /// Whether `cube` (a level-0 node) lies inside this node.
    pub const fn contains_cube(self, cube: SectionPos) -> bool {
        cube.x >> self.level == self.x
            && cube.y >> self.level == self.y
            && cube.z >> self.level == self.z
    }
    /// Whether `other` is this node or inside it.
    pub const fn contains(self, other: Self) -> bool {
        other.level <= self.level
            && other.x >> (self.level - other.level) == self.x
            && other.y >> (self.level - other.level) == self.y
            && other.z >> (self.level - other.level) == self.z
    }
    /// Distance in blocks from `point` to the nearest point of the node.
    pub fn distance(self, point: [f64; 3]) -> f64 {
        let size = f64::from(self.size());
        let mut sum = 0.0;
        for (c, o) in point.into_iter().zip(self.origin()) {
            let lo = f64::from(o);
            let gap = (lo - c).max(c - (lo + size)).max(0.0);
            sum += gap * gap;
        }
        sum.sqrt()
    }
}

/// One voxel: the block state standing in for `1 << level` blocks cubed, and
/// its biome (for tints).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Voxel {
    pub state: BlockStateId,
    pub biome: u16,
}

#[derive(Clone, Debug)]
enum Voxels {
    Uniform(Voxel),
    Dense(Box<[Voxel; VOLUME]>),
}

/// A node's voxels, indexed `y << 8 | z << 4 | x`. A node of one voxel
/// repeated (air, solid rock) stores just that voxel.
#[derive(Clone, Debug)]
pub struct CubeNode {
    pub key: CubeKey,
    voxels: Voxels,
    /// Columns (`z << 4 | x`) whose sky is blocked above the node: a set bit
    /// means something solid stands over the node there (an island, the
    /// vanilla terrain). `None`: all open. See [`CubeNode::column_covered`].
    covered: Option<Box<CoverMask>>,
}

/// One bit per column of a node, `z << 4 | x`.
pub type CoverMask = [u64; 4];

impl CubeNode {
    pub fn uniform(key: CubeKey, voxel: Voxel) -> Self {
        Self {
            key,
            voxels: Voxels::Uniform(voxel),
            covered: None,
        }
    }
    /// Voxels from `f(x, y, z)` (each 0..16); a node whose voxels all agree
    /// is stored as one.
    pub fn from_fn(key: CubeKey, mut f: impl FnMut(usize, usize, usize) -> Voxel) -> Self {
        let mut dense = Box::new([f(0, 0, 0); VOLUME]);
        for i in 1..VOLUME {
            dense[i] = f(i & 15, i >> 8, (i >> 4) & 15);
        }
        Self::dense(key, dense)
    }
    /// A node of exactly these voxels (`y << 8 | z << 4 | x`).
    pub fn from_dense(key: CubeKey, dense: Box<[Voxel; VOLUME]>) -> Self {
        Self::dense(key, dense)
    }
    fn dense(key: CubeKey, dense: Box<[Voxel; VOLUME]>) -> Self {
        let first = dense[0];
        if dense.iter().all(|&v| v == first) {
            Self::uniform(key, first)
        } else {
            Self {
                key,
                voxels: Voxels::Dense(dense),
                covered: None,
            }
        }
    }
    /// The node with its covered columns set (an empty mask is none).
    pub fn with_covered(mut self, covered: Option<Box<CoverMask>>) -> Self {
        self.covered = covered.filter(|mask| mask.iter().any(|&w| w != 0));
        self
    }
    pub fn covered_mask(&self) -> Option<&CoverMask> {
        self.covered.as_deref()
    }
    /// Whether the sky over the node is blocked at column (`x`, `z`).
    pub fn column_covered(&self, x: usize, z: usize) -> bool {
        let bit = z << 4 | x;
        self.covered
            .as_ref()
            .is_some_and(|mask| mask[bit >> 6] >> (bit & 63) & 1 == 1)
    }
    pub fn index(x: usize, y: usize, z: usize) -> usize {
        y << 8 | z << 4 | x
    }
    pub fn get(&self, x: usize, y: usize, z: usize) -> Voxel {
        self.at(Self::index(x, y, z))
    }
    pub fn at(&self, index: usize) -> Voxel {
        match &self.voxels {
            Voxels::Uniform(v) => *v,
            Voxels::Dense(d) => d[index],
        }
    }
    /// The only voxel of the node, if it has just one kind.
    pub fn as_uniform(&self) -> Option<Voxel> {
        match &self.voxels {
            Voxels::Uniform(v) => Some(*v),
            Voxels::Dense(_) => None,
        }
    }
}

/// How a voxel's block state counts when voxels are merged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Air,
    Fluid,
    Solid,
}

/// The node above eight children (in [`CubeKey::children`] order), each voxel
/// from its 2×2×2 child voxels: solid when at least 4 of 8 are solid (a tie
/// keeps solid), as the state seen most often among the solid ones, the
/// highest in Y winning a tie (grass over its dirt). Fluids do not count for
/// the majority; with no solid winner, at least 4 fluid voxels make a fluid
/// voxel. The rest is air.
pub fn downsample(
    children: [&CubeNode; 8],
    air: BlockStateId,
    kind: impl Fn(BlockStateId) -> Kind,
) -> CubeNode {
    let key = children[0].key.parent();
    // Eight equal uniform children: still uniform.
    if let Some(first) = children[0].as_uniform()
        && children.iter().all(|c| c.as_uniform() == Some(first))
    {
        return CubeNode::uniform(key, first).with_covered(merged_cover(&children));
    }
    let mut dense = Box::new(
        [Voxel {
            state: air,
            biome: 0,
        }; VOLUME],
    );
    for py in 0..SIDE {
        for pz in 0..SIDE {
            for px in 0..SIDE {
                let child = children[(px >> 3) | ((pz >> 3) << 1) | ((py >> 3) << 2)];
                let (cx, cy, cz) = ((px & 7) * 2, (py & 7) * 2, (pz & 7) * 2);
                let mut eight = [child.get(cx, cy, cz); 8];
                for (i, voxel) in eight.iter_mut().enumerate() {
                    *voxel = child.get(cx + (i & 1), cy + (i >> 2), cz + ((i >> 1) & 1));
                }
                dense[CubeNode::index(px, py, pz)] = merge_eight(&eight, air, &kind);
            }
        }
    }
    CubeNode::dense(key, dense).with_covered(merged_cover(&children))
}

/// The parent's covered columns: what covers the upper four children, a
/// parent column being covered when two of the four child columns under it
/// are.
fn merged_cover(children: &[&CubeNode; 8]) -> Option<Box<CoverMask>> {
    if children[4..].iter().all(|c| c.covered.is_none()) {
        return None;
    }
    let mut mask = Box::new([0u64; 4]);
    for pz in 0..SIDE {
        for px in 0..SIDE {
            let child = children[4 | (px >> 3) | ((pz >> 3) << 1)];
            let (cx, cz) = ((px & 7) * 2, (pz & 7) * 2);
            let covered = [(0, 0), (1, 0), (0, 1), (1, 1)]
                .into_iter()
                .filter(|&(dx, dz)| child.column_covered(cx + dx, cz + dz))
                .count();
            if covered >= 2 {
                let bit = pz << 4 | px;
                mask[bit >> 6] |= 1 << (bit & 63);
            }
        }
    }
    Some(mask)
}

/// `eight` is in `x | z << 1 | y << 2` order: the top four are the upper layer.
fn merge_eight(
    eight: &[Voxel; 8],
    air: BlockStateId,
    kind: &impl Fn(BlockStateId) -> Kind,
) -> Voxel {
    let kinds = eight.map(|v| kind(v.state));
    let count = |of: Kind| kinds.iter().filter(|&&k| k == of).count();
    let wanted = if count(Kind::Solid) >= 4 {
        Kind::Solid
    } else if count(Kind::Fluid) >= 4 {
        Kind::Fluid
    } else {
        return Voxel {
            state: air,
            biome: eight[0].biome,
        };
    };
    // (state, count, highest layer seen, the voxel there)
    let mut tally: smallvec::SmallVec<[(BlockStateId, u8, u8, Voxel); 8]> = Default::default();
    for (i, &voxel) in eight.iter().enumerate() {
        if kinds[i] != wanted {
            continue;
        }
        let layer = (i >> 2) as u8;
        match tally.iter_mut().find(|t| t.0 == voxel.state) {
            Some(t) => {
                t.1 += 1;
                if layer >= t.2 {
                    t.2 = layer;
                    t.3 = voxel;
                }
            }
            None => tally.push((voxel.state, 1, layer, voxel)),
        }
    }
    tally
        .into_iter()
        .max_by_key(|t| (t.1, t.2))
        .map(|t| t.3)
        .expect("a kind that was counted")
}

/// Where nodes come from. The game provides one backed by its generator.
pub trait CubeSampler: Send {
    /// The node at `key`'s own resolution (a voxel is `1 << level` blocks),
    /// or `None` when it cannot be had now. Coarse levels should be sampled
    /// directly: when a level above 3 answers `None`, the node is not built
    /// from its children.
    fn node(&mut self, key: CubeKey) -> Option<CubeNode>;
    /// As [`CubeSampler::node`] for the GPU path: a sampler that generates on
    /// the GPU answers [`Resident::Slot`] for a node it left in the GPU pool,
    /// never read back. Every other answers the node.
    fn resident(&mut self, key: CubeKey) -> Option<Resident> {
        self.node(key).map(Resident::Node)
    }
    /// [`CubeSampler::resident`] for many nodes at once, in order. A sampler
    /// whose generator batches (the GPU's) answers them in one request, so a
    /// worker keeps the GPU's batches full instead of waiting on one node.
    fn resident_many(&mut self, keys: &[CubeKey]) -> Vec<Option<Resident>> {
        keys.iter().map(|&key| self.resident(key)).collect()
    }
    /// What the sampler knows of the whole node without sampling it (the GPU
    /// generator's 2D pre-pass): lets the planner leave out subtrees of air
    /// and stop at rock. Called on a background thread; the default knows
    /// nothing.
    fn classify(&mut self, _key: CubeKey) -> NodeClass {
        NodeClass::Mixed
    }
    /// The GPU descent behind this sampler (`gfx::cube_descent`), if it has
    /// one: it proves whole subtrees of the octree uniform from the density
    /// bound, in place of [`CubeSampler::classify`] node by node
    /// ([`CubeLod::set_gpu_descent`]).
    fn descent(&self) -> Option<Arc<GpuDescent>> {
        None
    }
    /// The GPU generator behind this sampler (its device, node pool and block
    /// states), if any: what lets [`CubeLod`] mesh and draw on the GPU.
    fn gpu(&self) -> Option<GpuBackend> {
        None
    }
    /// An independent sampler for another thread, if this one can have one.
    /// Without it the producer runs a single worker.
    fn fork(&self) -> Option<Box<dyn CubeSampler>>;
    /// Whether the far terrain can be drawn as surface tiles at the coarse
    /// levels (`gfx::surface_tile`): a world whose terrain the tiles' generator
    /// reproduces (a Custom world without `cubeAreas`). False by default.
    fn surface(&self) -> bool {
        false
    }
    /// The sea's surface (the sea level + 1) when the sampler fills a voxel
    /// that holds it with water where the sea is open there: the CPU mesher
    /// draws such water up to the surface only (the GPU's reads the
    /// generator's own). `None` by default.
    fn sea_surface(&self) -> Option<i32> {
        None
    }
    /// Cubes that changed since this was last asked (the producer drops the
    /// nodes it kept in memory over them). Nothing by default.
    fn changes(&mut self) -> Changes {
        Changes::None
    }
}

/// A sampled node on the GPU path: its voxels, or a slot of the GPU node pool
/// (`gfx::cube_gen::NodePool`) they were generated into. The slot's owner
/// frees it.
#[derive(Debug)]
pub enum Resident {
    Node(CubeNode),
    Slot(u32),
}

/// What a [`CubeSampler`] reports as changed (see `CubeLodCache::invalidate`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Changes {
    None,
    /// These cubes were saved again.
    Cubes(Vec<SectionPos>),
    /// So many that the list was lost: forget every node.
    All,
}

// ---------------------------------------------------------------- the plan

/// The distance rule the plan used before detail followed the screen, kept
/// as the tests' reference ([`Rule::distance`]): a node splits while nearer
/// than this many times its size. A far voxel then spans about 1/40 of its
/// distance, 13 pixels on a 720-pixel, 70 degree view.
pub const DETAIL: f64 = 2.5;
/// A node the previous plan had split merges back only when its voxels are
/// this many times smaller on screen than the size that splits it (it is
/// this many times farther), so the camera moving a little back and forth
/// across a boundary does not flip nodes between levels.
pub const MERGE: f64 = 1.25;
/// Nodes this level or finer ask `is_loaded` about each cube they hold; coarser
/// ones that meet the loaded box simply subdivide.
const PROBE_LEVEL: u8 = 2;
/// A view whose detail in view is less than this many times the unzoomed
/// one's is planned as unzoomed: a bow's draw or a sprint's wider view does
/// not make the plan depend on where the camera looks.
const ZOOMED: f64 = 1.5;

/// What the player sees, for choosing detail by size on screen
/// ([`CubeLod::update`]).
///
/// A node splits while one of its voxels, `1 << level` blocks wide, would
/// span more than `target_px` pixels at the node's nearest point:
/// `voxel / distance * height_px / (2 * tan(fov_y / 2)) > target_px`. A
/// narrower field of view (the spyglass) makes every voxel bigger on screen,
/// so the nodes in view refine; nodes out of view keep the unzoomed rule.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FarView {
    /// The direction the camera looks (a unit vector).
    pub forward: [f64; 3],
    /// The vertical field of view drawn now, in radians: the spyglass, zoom
    /// and dynamic field of view included.
    pub fov_y: f64,
    /// The vertical field of view without any zoom (the FOV option), in
    /// radians: what nodes out of view are planned for.
    pub base_fov_y: f64,
    /// The viewport's width over its height.
    pub aspect: f64,
    /// The viewport's height in pixels.
    pub height_px: f64,
    /// The most pixels a voxel may span, at its node's nearest point, before
    /// the node splits (the Cubic Far Detail setting).
    pub target_px: f64,
}

impl FarView {
    /// A 70 degree, 16:9 view 720 pixels tall looking along +Z, aiming for
    /// `target_px` (the benchmark's window).
    pub fn standard(target_px: f64) -> Self {
        Self {
            forward: [0.0, 0.0, 1.0],
            fov_y: 70f64.to_radians(),
            base_fov_y: 70f64.to_radians(),
            aspect: 16.0 / 9.0,
            height_px: 720.0,
            target_px,
        }
    }
    /// The same view looking along `forward` with a field of view of
    /// `fov_y` radians (a zoom: the base stays).
    pub fn looking(self, forward: [f64; 3], fov_y: f64) -> Self {
        Self {
            forward,
            fov_y,
            ..self
        }
    }
    fn fov(fov: f64) -> f64 {
        if fov.is_finite() {
            fov.clamp(0.1f64.to_radians(), 179f64.to_radians())
        } else {
            70f64.to_radians()
        }
    }
    /// Pixels a block spans per block of distance with this vertical field of
    /// view: a voxel `w` blocks wide `d` blocks away spans `w / d * focal`.
    pub fn focal_px(&self, fov_y: f64) -> f64 {
        let height = if self.height_px.is_finite() {
            self.height_px.max(1.0)
        } else {
            720.0
        };
        height / (2.0 * (Self::fov(fov_y) / 2.0).tan())
    }
    /// Half the angle across the screen's diagonal.
    pub fn half_diagonal(&self) -> f64 {
        let aspect = if self.aspect.is_finite() && self.aspect > 0.0 {
            self.aspect
        } else {
            16.0 / 9.0
        };
        ((Self::fov(self.fov_y) / 2.0).tan() * (1.0 + aspect * aspect).sqrt()).atan()
    }
    /// How far a zoomed view may turn before the plan is redone: the plan's
    /// cone of finer nodes is this much wider than the screen.
    pub fn turn_margin(&self) -> f64 {
        (Self::fov(self.fov_y) * 0.25).max(0.5f64.to_radians())
    }
    /// The plan's rule for this view, with at most `budget` leaves.
    pub fn rule(&self, budget: usize) -> Rule {
        let target = if self.target_px.is_finite() {
            self.target_px.max(0.05)
        } else {
            8.0
        };
        // Out of view, the wider of the two fields of view: a sprint's or a
        // speed effect's wider view never asks for more than in view.
        let base = Self::fov(self.base_fov_y).max(Self::fov(self.fov_y));
        let out_of_view = self.focal_px(base) / (16.0 * target);
        let in_view = self.focal_px(self.fov_y) / (16.0 * target);
        if in_view < out_of_view * ZOOMED {
            return Rule {
                in_view: out_of_view,
                out_of_view,
                cone: None,
                budget,
            };
        }
        Rule {
            in_view,
            out_of_view,
            cone: Some(Cone::new(
                self.forward,
                self.half_diagonal() + self.turn_margin(),
            )),
            budget,
        }
    }
}

/// How finely a plan goes ([`plan_classified`]): a node splits while it is
/// nearer than `factor` times its size, `factor` being `in_view` for nodes
/// the cone meets and `out_of_view` for the rest (with [`FarView::rule`],
/// the voxel's size on screen against the target), and while the plan has
/// fewer than `budget` leaves.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rule {
    pub in_view: f64,
    pub out_of_view: f64,
    /// Where `in_view` applies; `None` when the two are the same.
    pub cone: Option<Cone>,
    /// The most leaves a plan has. Past it, the nodes whose voxels are
    /// smallest on screen (the farthest for their size) stay whole: the plan
    /// coarsens from the far end rather than failing. Splits around the
    /// loaded cubes are made whatever the budget.
    pub budget: usize,
}

impl Rule {
    /// The distance rule, the same everywhere and unbounded.
    pub const fn distance(detail: f64) -> Self {
        Self {
            in_view: detail,
            out_of_view: detail,
            cone: None,
            budget: usize::MAX,
        }
    }
    /// The rule with voxels `times` bigger on screen, and at most `budget`
    /// leaves (the coverage layer's).
    pub fn coarser(&self, times: f64, budget: usize) -> Self {
        Self {
            in_view: self.in_view / times,
            out_of_view: self.out_of_view / times,
            cone: self.cone,
            budget,
        }
    }
    /// The factor for `node` seen from `camera`.
    pub fn factor(&self, camera: [f64; 3], node: CubeKey) -> f64 {
        match &self.cone {
            Some(cone) if cone.meets(camera, node) => self.in_view,
            _ => self.out_of_view,
        }
    }
    /// The factor for a box of blocks `lo..hi` (a surface tile's terrain).
    pub fn factor_box(&self, camera: [f64; 3], lo: [f64; 3], hi: [f64; 3]) -> f64 {
        match &self.cone {
            Some(cone) if cone.meets_box(camera, lo, hi) => self.in_view,
            _ => self.out_of_view,
        }
    }
}

/// A cone of directions from the camera: the view, widened.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cone {
    axis: [f64; 3],
    half: f64,
}

impl Cone {
    /// Directions within `half` radians of `axis`.
    pub fn new(axis: [f64; 3], half: f64) -> Self {
        let len = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        let axis = if len > 1e-9 && len.is_finite() {
            axis.map(|v| v / len)
        } else {
            [0.0, 0.0, 1.0]
        };
        Self {
            axis,
            half: if half.is_finite() {
                half.clamp(0.0, std::f64::consts::PI)
            } else {
                std::f64::consts::PI
            },
        }
    }
    /// Whether any of `node` may lie inside the cone seen from `camera`:
    /// the node is taken as the sphere around it, whose own angular radius
    /// widens the cone, so a node straddling the edge counts in.
    pub fn meets(&self, camera: [f64; 3], node: CubeKey) -> bool {
        let size = f64::from(node.size());
        let lo = node.origin().map(f64::from);
        self.meets_box(camera, lo, lo.map(|v| v + size))
    }
    /// [`Cone::meets`] for a box of blocks `lo..hi`.
    pub fn meets_box(&self, camera: [f64; 3], lo: [f64; 3], hi: [f64; 3]) -> bool {
        let d = [0, 1, 2].map(|a| (lo[a] + hi[a]) / 2.0 - camera[a]);
        let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        let e = [0, 1, 2].map(|a| (hi[a] - lo[a]) / 2.0);
        let radius = (e[0] * e[0] + e[1] * e[1] + e[2] * e[2]).sqrt();
        if len <= radius {
            return true;
        }
        let reach = self.half + (radius / len).asin();
        if reach >= std::f64::consts::PI {
            return true;
        }
        (d[0] * self.axis[0] + d[1] * self.axis[1] + d[2] * self.axis[2]) / len >= reach.cos()
    }
}

/// The box of cubes around the camera's cube that the game may have loaded:
/// `horizontal` cubes each way in X and Z, `vertical` in Y. `is_loaded` is
/// only asked inside it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LoadedBox {
    pub horizontal: i32,
    pub vertical: i32,
}

fn cube_of(camera: [f64; 3]) -> SectionPos {
    let c = camera.map(|v| (v / 16.0).floor() as i32);
    SectionPos::new(c[0], c[1], c[2])
}

/// The nodes that draw the world out to `max_distance` blocks from `camera`
/// except the cubes `is_loaded` reports (which the near renderer draws):
/// non-overlapping, nearest first. Starting from nodes about as wide as the
/// range, a node splits while the camera is nearer than [`DETAIL`] times its
/// size, or while it holds loaded cubes (down to single cubes, which are
/// skipped), and a node farther than `max_distance` is left out.
pub fn plan(
    camera: [f64; 3],
    max_distance: f64,
    loaded: LoadedBox,
    is_loaded: &dyn Fn(SectionPos) -> bool,
) -> Vec<CubeKey> {
    plan_from(camera, max_distance, loaded, is_loaded, &[])
}

/// [`plan`] with hysteresis: a node that `previous` (the last plan) split
/// stays split until the camera is [`MERGE`] times farther than the distance
/// that splits it. The invariants of `plan` hold for any `previous`.
pub fn plan_from(
    camera: [f64; 3],
    max_distance: f64,
    loaded: LoadedBox,
    is_loaded: &dyn Fn(SectionPos) -> bool,
    previous: &[CubeKey],
) -> Vec<CubeKey> {
    plan_classified(
        camera,
        max_distance,
        &Rule::distance(DETAIL),
        loaded,
        is_loaded,
        previous,
        &|_| NodeClass::Mixed,
    )
    .leaves
}

/// Vanilla's world border, in blocks each way from the origin.
pub const WORLD_BORDER: i64 = 30_000_000;

/// Whether any of the node lies inside the world: within the border
/// horizontally and the cube range vertically. Overflow-safe at any level.
pub fn in_world(node: CubeKey) -> bool {
    let size = i64::from(node.size());
    // Whether the node's blocks `[at, at + size)` meet blocks `[lo, end)`.
    let within = |c: i32, lo: i64, end: i64| {
        let at = i64::from(c) * size;
        at < end && at + size > lo
    };
    // X and Z: vanilla's bounds, -30,000,000 inclusive to 30,000,000
    // exclusive; Y: the cube range, both ends inclusive.
    within(node.x, -WORLD_BORDER, WORLD_BORDER)
        && within(node.z, -WORLD_BORDER, WORLD_BORDER)
        && within(
            node.y,
            i64::from(horizon_world::cubic::MIN_Y),
            i64::from(horizon_world::cubic::MAX_Y) + 1,
        )
}

/// What the 2D pre-pass knows of a whole node (`CubeSampler::classify`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeClass {
    /// Unknown, or the surface may cross it.
    Mixed,
    /// Every voxel is air or every voxel is fluid: it draws nothing, and nor
    /// does anything inside it.
    Air,
    /// Every voxel is rock: it is a leaf however near, a closed shell.
    Rock,
}

/// The nodes the plan starts from: those of the first level at least as wide
/// as `max_distance` that the range around `camera` meets (at most 27).
pub fn root_nodes(camera: [f64; 3], max_distance: f64) -> Vec<CubeKey> {
    let top = (0..=MAX_CUBE_LEVEL)
        .find(|&l| f64::from(16 << l) >= max_distance)
        .unwrap_or(MAX_CUBE_LEVEL);
    let size = f64::from(16 << top);
    let span = |c: f64| {
        (
            ((c - max_distance) / size).floor() as i32,
            ((c + max_distance) / size).floor() as i32,
        )
    };
    let mut roots = Vec::new();
    let (x0, x1) = span(camera[0]);
    let (y0, y1) = span(camera[1]);
    let (z0, z1) = span(camera[2]);
    for x in x0..=x1 {
        for y in y0..=y1 {
            for z in z0..=z1 {
                roots.push(CubeKey::new(top, x, y, z));
            }
        }
    }
    roots
}

/// What [`plan_far`] chose.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Plan {
    /// The cube leaves: non-overlapping, nearest first.
    pub leaves: Vec<CubeKey>,
    /// The surface tiles (with [`PlanInput::surface`]), nearest first. No cube
    /// leaf lies in a tile's footprint, at any height.
    pub tiles: Vec<TileKey>,
    /// Nodes left out as uniform air or fluid, with their whole subtrees.
    pub pruned: usize,
    /// The budget kept some nodes whole that the rule would have split.
    pub limited: bool,
    /// How many times its target the first node the budget kept whole was
    /// (its voxels' size on screen over the target; 1 when the budget did
    /// not bind): the plan reaches about `target * coarsened`.
    pub coarsened: f64,
    /// The tile budget kept some tiles whole.
    pub tiles_limited: bool,
}

/// What the plan needs to draw the coarse far terrain as surface tiles
/// (`gfx::surface_tile`): from [`SURFACE_LEVEL`] up, a quadtree of tiles over
/// X and Z, each standing for the whole height of its footprint.
#[derive(Clone, Copy)]
pub struct Surface<'a> {
    /// The headers of the tiles generated so far: complex columns, heights.
    pub info: &'a dyn Fn(TileKey) -> Option<TileInfo>,
    /// Where a tile not generated yet is taken to have its terrain.
    pub sea_level: i32,
    /// The most tiles a plan holds.
    pub budget: usize,
    /// The last plan's tiles, for the hysteresis.
    pub previous: &'a [TileKey],
}

/// Everything [`plan_far`] plans from.
#[derive(Clone, Copy)]
pub struct PlanInput<'a> {
    pub camera: [f64; 3],
    pub max_distance: f64,
    pub rule: &'a Rule,
    pub loaded: LoadedBox,
    pub is_loaded: &'a dyn Fn(SectionPos) -> bool,
    /// The last plan's cube leaves, for the hysteresis.
    pub previous: &'a [CubeKey],
    pub class: &'a dyn Fn(CubeKey) -> NodeClass,
    /// Surface tiles at the coarse levels; `None`: cube nodes everywhere.
    pub surface: Option<Surface<'a>>,
}

/// Where a cube node's footprint lies with respect to the tiles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Region {
    /// All of it is planned as cube nodes (or no tile plan reaches it).
    Cubes,
    /// All of it is under one tile: the tile draws it.
    Tiles,
    /// Both: the node splits until each part is one or the other.
    Split,
}

/// A node the plan may split, by how far over the target its voxels are.
#[derive(Clone, Copy, Debug)]
struct Candidate {
    /// It meets the loaded cubes, or both tiles and cube nodes: it splits
    /// whatever the budget.
    forced: bool,
    ratio: f64,
    distance: f64,
    key: CubeKey,
    region: Region,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == std::cmp::Ordering::Equal
    }
}
impl Eq for Candidate {}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.forced
            .cmp(&other.forced)
            .then(self.ratio.total_cmp(&other.ratio))
            .then(other.key.cmp(&self.key))
    }
}

/// What becomes of a node the plan reaches.
enum Fate {
    /// Outside the world or the range, all loaded, or under a tile: nothing.
    Out,
    /// Uniform air or fluid: left out with its subtree.
    Pruned,
    Leaf(f64, CubeKey),
    Split(Candidate),
}

/// The tile quadtree a plan chose: its tiles, the footprints planned as cube
/// nodes instead, and the footprints split further.
#[derive(Default)]
struct Quadtree {
    tiles: FxHashSet<TileKey>,
    cubes: FxHashSet<TileKey>,
    split: FxHashSet<TileKey>,
}

impl Quadtree {
    /// The region of `node`'s footprint, from that footprint alone (its
    /// parent's was split, or it is a root).
    fn region(&self, node: CubeKey) -> Region {
        let t = TileKey::new(node.level, node.x, node.z);
        if self.tiles.contains(&t) {
            Region::Tiles
        } else if self.split.contains(&t) {
            Region::Split
        } else {
            // A cube footprint, or one no tile plan reaches (out of range).
            Region::Cubes
        }
    }
}

/// A tile the plan may split.
#[derive(Clone, Copy, Debug)]
struct TileCandidate {
    forced: bool,
    ratio: f64,
    distance: f64,
    key: TileKey,
}

impl PartialEq for TileCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == std::cmp::Ordering::Equal
    }
}
impl Eq for TileCandidate {}
impl PartialOrd for TileCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for TileCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.forced
            .cmp(&other.forced)
            .then(self.ratio.total_cmp(&other.ratio))
            .then(other.key.cmp(&self.key))
    }
}

enum TileFate {
    Out,
    /// Planned as cube nodes of its level.
    Cubes,
    Leaf(f64),
    Split(TileCandidate),
}

/// Whether any of a tile's footprint lies within the world border.
fn tile_in_world(t: TileKey) -> bool {
    let size = i64::from(t.size());
    let within = |c: i32| {
        let at = i64::from(c) * size;
        at < WORLD_BORDER && at + size > -WORLD_BORDER
    };
    within(t.x) && within(t.z)
}

/// The blocks a tile's terrain is taken to fill: its footprint, from the
/// lowest its ground can reach to its highest top (or water), from its header;
/// before it is generated, a band around the sea.
pub fn tile_box(t: TileKey, info: Option<TileInfo>, sea_level: i32) -> ([f64; 3], [f64; 3]) {
    let [ox, oz] = t.origin();
    let size = f64::from(t.size());
    let (lo, hi) = match info {
        Some(i) => (
            i.skirt.max(horizon_world::cubic::MIN_Y),
            i.highest.min(horizon_world::cubic::MAX_Y) + 1,
        ),
        None => (sea_level - UNKNOWN_BELOW, sea_level + UNKNOWN_ABOVE),
    };
    (
        [f64::from(ox), f64::from(lo), f64::from(oz)],
        [
            f64::from(ox) + size,
            f64::from(hi.max(lo)),
            f64::from(oz) + size,
        ],
    )
}

/// How far below and above the sea an unknown tile's terrain is taken to lie.
const UNKNOWN_BELOW: i32 = 64;
const UNKNOWN_ABOVE: i32 = 192;

/// Distance in blocks from `point` to the nearest point of the box.
fn box_distance(point: [f64; 3], lo: [f64; 3], hi: [f64; 3]) -> f64 {
    let mut sum = 0.0;
    for a in 0..3 {
        let gap = (lo[a] - point[a]).max(point[a] - hi[a]).max(0.0);
        sum += gap * gap;
    }
    sum.sqrt()
}

/// The tile quadtree: from the roots' columns down, a tile splits into its
/// four children by `rule` (on the box of its terrain, so a tall mountain
/// or a sky island refines as near as it reaches), the ones furthest over
/// their target first, up to `surface.budget` tiles. A footprint becomes cube
/// nodes where [`node_kind`] says so (below [`SURFACE_LEVEL`]; a complex tile
/// sooner) and where it meets the loaded cubes or holds the camera: the
/// camera's own footprint is never a tile.
fn plan_tiles(
    input: &PlanInput,
    surface: &Surface,
    roots: &[CubeKey],
) -> (Quadtree, Vec<TileKey>, bool) {
    let camera = input.camera;
    let center = cube_of(camera);
    let h = input.loaded.horizontal.max(0);
    // The loaded box's footprint, in blocks: it always holds the camera.
    let near = (
        [i64::from(center.x - h) * 16, i64::from(center.z - h) * 16],
        [
            i64::from(center.x + h + 1) * 16,
            i64::from(center.z + h + 1) * 16,
        ],
    );
    let meets_near = |t: TileKey| {
        let [ox, oz] = t.origin().map(i64::from);
        let size = i64::from(t.size());
        ox < near.1[0] && ox + size > near.0[0] && oz < near.1[1] && oz + size > near.0[1]
    };
    // The footprints the last plan split: those holding its tiles, or its
    // cube leaves at least one level finer.
    let mut was_split = FxHashSet::<TileKey>::default();
    for &t in surface.previous {
        let mut level = t.level;
        while level < MAX_CUBE_LEVEL {
            level += 1;
            let up = level - t.level;
            if !was_split.insert(TileKey::new(level, t.x >> up, t.z >> up)) {
                break;
            }
        }
    }
    for &c in input.previous {
        let mut level = c.level.max(SURFACE_LEVEL - 1);
        while level < MAX_CUBE_LEVEL {
            level += 1;
            let up = level - c.level;
            if !was_split.insert(TileKey::new(level, c.x >> up, c.z >> up)) {
                break;
            }
        }
    }
    let fate = |t: TileKey| -> TileFate {
        if !tile_in_world(t) {
            return TileFate::Out;
        }
        let info = (surface.info)(t);
        let (lo, hi) = tile_box(t, info, surface.sea_level);
        let distance = box_distance(camera, lo, hi);
        if distance > input.max_distance {
            return TileFate::Out;
        }
        let as_node = CubeKey::new(t.level, t.x, 0, t.z);
        if meets_near(t) {
            return if t.level <= SURFACE_LEVEL {
                TileFate::Cubes
            } else {
                TileFate::Split(TileCandidate {
                    forced: true,
                    ratio: f64::INFINITY,
                    distance,
                    key: t,
                })
            };
        }
        if node_kind(as_node, None, info) == NodeKind::Cube {
            return TileFate::Cubes;
        }
        let merge = if was_split.contains(&t) { MERGE } else { 1.0 };
        let ratio = input.rule.factor_box(camera, lo, hi) * f64::from(t.size()) * merge / distance;
        if ratio > 1.0 {
            TileFate::Split(TileCandidate {
                forced: false,
                ratio,
                distance,
                key: t,
            })
        } else {
            TileFate::Leaf(distance)
        }
    };
    let mut q = Quadtree::default();
    let mut leaves: Vec<(f64, TileKey)> = Vec::new();
    let mut heap = std::collections::BinaryHeap::new();
    let mut count = 0usize;
    let take = |t: TileKey,
                f: TileFate,
                q: &mut Quadtree,
                leaves: &mut Vec<(f64, TileKey)>,
                heap: &mut std::collections::BinaryHeap<TileCandidate>| match f {
        TileFate::Out => {}
        TileFate::Cubes => {
            q.cubes.insert(t);
        }
        TileFate::Leaf(d) => leaves.push((d, t)),
        TileFate::Split(c) => heap.push(c),
    };
    let mut columns: Vec<TileKey> = roots
        .iter()
        .map(|r| TileKey::new(r.level, r.x, r.z))
        .collect();
    columns.sort_unstable();
    columns.dedup();
    for t in columns {
        let f = fate(t);
        if matches!(f, TileFate::Leaf(_) | TileFate::Split(_)) {
            count += 1;
        }
        take(t, f, &mut q, &mut leaves, &mut heap);
    }
    let mut full = false;
    let mut limited = false;
    while let Some(c) = heap.pop() {
        if !c.forced && full {
            leaves.push((c.distance, c.key));
            continue;
        }
        let t = c.key;
        let children: [TileKey; 4] = std::array::from_fn(|i| {
            TileKey::new(
                t.level - 1,
                t.x * 2 + (i & 1) as i32,
                t.z * 2 + (i >> 1) as i32,
            )
        });
        let fates = children.map(&fate);
        let added = fates
            .iter()
            .filter(|f| matches!(f, TileFate::Leaf(_) | TileFate::Split(_)))
            .count();
        if !c.forced && count - 1 + added > surface.budget {
            full = true;
            limited = true;
            leaves.push((c.distance, t));
            continue;
        }
        count = count - 1 + added;
        q.split.insert(t);
        for (child, f) in children.into_iter().zip(fates) {
            take(child, f, &mut q, &mut leaves, &mut heap);
        }
    }
    leaves.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    q.tiles = leaves.iter().map(|&(_, t)| t).collect();
    (q, leaves.into_iter().map(|(_, t)| t).collect(), limited)
}

/// The plan with what is known of nodes' contents and how fine it goes, cube
/// nodes everywhere ([`plan_far`] without tiles).
pub fn plan_classified(
    camera: [f64; 3],
    max_distance: f64,
    rule: &Rule,
    loaded: LoadedBox,
    is_loaded: &dyn Fn(SectionPos) -> bool,
    previous: &[CubeKey],
    class: &dyn Fn(CubeKey) -> NodeClass,
) -> Plan {
    plan_far(&PlanInput {
        camera,
        max_distance,
        rule,
        loaded,
        is_loaded,
        previous,
        class,
        surface: None,
    })
}

/// The plan: a node `class` calls [`NodeClass::Air`] is left out with its
/// whole subtree, one it calls [`NodeClass::Rock`] stays a leaf (nothing
/// inside shows), unless it holds loaded cubes. A node splits by `rule` (with
/// hysteresis: one `previous` split splits while its ratio is above
/// `1 / MERGE`), the nodes furthest over their target first, until
/// `rule.budget` leaves. With `surface`, the tile quadtree is planned first
/// ([`plan_tiles`]) and the cube nodes fill the footprints it leaves them:
/// a node under a tile is left out, a node that is partly under one splits
/// (or, all rock, is left out: the tiles' skirts close the ground).
pub fn plan_far(input: &PlanInput) -> Plan {
    let PlanInput {
        camera,
        max_distance,
        rule,
        loaded,
        is_loaded,
        previous,
        class,
        surface,
    } = *input;
    // The nodes holding a leaf of the last plan: the ones it split.
    let mut was_split = FxHashSet::<CubeKey>::default();
    for &leaf in previous {
        let mut up = leaf;
        while up.level < MAX_CUBE_LEVEL {
            up = up.parent();
            if !was_split.insert(up) {
                break;
            }
        }
    }
    let roots = root_nodes(camera, max_distance);
    let mut out = Plan {
        coarsened: 1.0,
        ..Plan::default()
    };
    let quadtree = match surface {
        Some(s) if roots.first().is_some_and(|r| r.level >= SURFACE_LEVEL) => {
            let (q, tiles, limited) = plan_tiles(input, &s, &roots);
            out.tiles = tiles;
            out.tiles_limited = limited;
            Some(q)
        }
        _ => None,
    };
    let center = cube_of(camera);
    let fate = |node: CubeKey, parent: Option<Region>| -> Fate {
        if !in_world(node) {
            return Fate::Out;
        }
        let distance = node.distance(camera);
        if distance > max_distance {
            return Fate::Out;
        }
        let region = match (&quadtree, parent) {
            (None, _) => Region::Cubes,
            (Some(_), Some(Region::Cubes)) => Region::Cubes,
            (Some(q), _) => q.region(node),
        };
        if region == Region::Tiles {
            return Fate::Out;
        }
        let split = |forced| {
            Fate::Split(Candidate {
                forced,
                ratio: f64::INFINITY,
                distance,
                key: node,
                region,
            })
        };
        match loaded_state(node, center, loaded, is_loaded) {
            Loaded::All => Fate::Out,
            Loaded::Some => split(true),
            Loaded::None => match class(node) {
                NodeClass::Air => Fate::Pruned,
                // Rock reaching under a tile: the tile's skirts close it.
                NodeClass::Rock if region == Region::Split => Fate::Out,
                NodeClass::Rock => Fate::Leaf(distance, node),
                NodeClass::Mixed if region == Region::Split => split(true),
                NodeClass::Mixed if node.level == 0 => Fate::Leaf(distance, node),
                NodeClass::Mixed => {
                    let merge = if was_split.contains(&node) {
                        MERGE
                    } else {
                        1.0
                    };
                    let ratio =
                        rule.factor(camera, node) * f64::from(node.size()) * merge / distance;
                    if ratio > 1.0 {
                        Fate::Split(Candidate {
                            forced: false,
                            ratio,
                            distance,
                            key: node,
                            region,
                        })
                    } else {
                        Fate::Leaf(distance, node)
                    }
                }
            },
        }
    };
    let mut leaves: Vec<(f64, CubeKey)> = Vec::new();
    let mut heap = std::collections::BinaryHeap::new();
    // Leaves and candidates so far: what the plan holds if nothing more splits.
    let mut count = 0usize;
    let take = |f: Fate,
                leaves: &mut Vec<(f64, CubeKey)>,
                heap: &mut std::collections::BinaryHeap<Candidate>,
                pruned: &mut usize| match f {
        Fate::Out => {}
        Fate::Pruned => *pruned += 1,
        Fate::Leaf(distance, key) => leaves.push((distance, key)),
        Fate::Split(c) => heap.push(c),
    };
    for root in roots {
        let f = fate(root, None);
        if matches!(f, Fate::Leaf(..) | Fate::Split(_)) {
            count += 1;
        }
        take(f, &mut leaves, &mut heap, &mut out.pruned);
    }
    let mut full = false;
    while let Some(c) = heap.pop() {
        if !c.forced && full {
            leaves.push((c.distance, c.key));
            continue;
        }
        let fates = c.key.children().map(|child| fate(child, Some(c.region)));
        let added = fates
            .iter()
            .filter(|f| matches!(f, Fate::Leaf(..) | Fate::Split(_)))
            .count();
        if !c.forced && count - 1 + added > rule.budget {
            // Every node left is at most this far over its target.
            full = true;
            out.limited = true;
            out.coarsened = c.ratio;
            leaves.push((c.distance, c.key));
            continue;
        }
        count = count - 1 + added;
        for f in fates {
            take(f, &mut leaves, &mut heap, &mut out.pruned);
        }
    }
    leaves.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    out.leaves = leaves.into_iter().map(|(_, node)| node).collect();
    out
}

enum Loaded {
    None,
    Some,
    All,
}

fn loaded_state(
    node: CubeKey,
    center: SectionPos,
    loaded: LoadedBox,
    is_loaded: &dyn Fn(SectionPos) -> bool,
) -> Loaded {
    let side = 1_i64 << node.level;
    let meets = |c: i32, n: i32, radius: i32| {
        let lo = i64::from(n) * side;
        lo <= i64::from(c) + i64::from(radius) && lo + side > i64::from(c) - i64::from(radius)
    };
    if !(meets(center.x, node.x, loaded.horizontal)
        && meets(center.y, node.y, loaded.vertical)
        && meets(center.z, node.z, loaded.horizontal))
    {
        return Loaded::None;
    }
    if node.level > PROBE_LEVEL {
        return Loaded::Some;
    }
    let side = side as i32;
    let (mut yes, mut no) = (0, 0);
    for dy in 0..side {
        for dz in 0..side {
            for dx in 0..side {
                let cube = SectionPos::new(
                    (node.x << node.level) + dx,
                    (node.y << node.level) + dy,
                    (node.z << node.level) + dz,
                );
                if is_loaded(cube) {
                    yes += 1;
                } else {
                    no += 1;
                }
            }
        }
    }
    match (yes, no) {
        (0, _) => Loaded::None,
        (_, 0) => Loaded::All,
        _ => Loaded::Some,
    }
}

// ------------------------------------------------------------- node kinds

/// From this level up, the far terrain is drawn as 2D height-map tiles
/// (`gfx::surface_tile`) rather than 16^3 cube nodes, where tiles run (a
/// Custom world on the GPU): a quadtree over X and Z, each tile standing for
/// the whole height of its footprint, with per column a few solid spans at
/// exact heights. A level-12 cube node is 65,536 blocks wide with voxels 4,096
/// blocks tall: nearly all terrain lies inside one or two of them, so it snaps
/// the surface to whole voxels, where a tile keeps every block of height.
/// A guess to measure: with exact heights tiles may look right lower down.
pub const SURFACE_LEVEL: u8 = 12;
/// A tile flagged complex (some columns have more solid spans than a tile
/// holds: floating islands, overhangs, a cavern's ceiling) turns into cube
/// nodes this many levels sooner, coming from the coarse end.
pub const COMPLEX_LIFT: u8 = 3;

/// How a far node is drawn ([`node_kind`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeKind {
    /// A 2D height-map tile over the node's X and Z (all Y).
    Tile,
    /// A 16^3 cube node.
    Cube,
}

/// How the footprint of `key` at its level is drawn where tiles run: the one
/// place this is decided. A tile at [`SURFACE_LEVEL`] or up, cube nodes below;
/// a tile whose header says it is complex turns into cube nodes
/// [`COMPLEX_LIFT`] levels sooner. `tile` is that header, `None` while the
/// tile is not generated: the tile is drawn first (it is the fastest coverage)
/// and refined into cube nodes once known to be complex. `view` is for a
/// screen-size threshold later (a tile stays a tile while its height steps are
/// below a pixel).
pub fn node_kind(key: CubeKey, view: Option<&FarView>, tile: Option<TileInfo>) -> NodeKind {
    let _ = view;
    let from = match tile {
        Some(info) if info.complex() => SURFACE_LEVEL.saturating_add(COMPLEX_LIFT),
        _ => SURFACE_LEVEL,
    };
    if key.level >= from {
        NodeKind::Tile
    } else {
        NodeKind::Cube
    }
}

// ---------------------------------------------------------------- producer

/// Nodes kept after sampling, so a node meshed again (or the parent of
/// children already sampled) is not sampled twice.
const CACHE_NODES: usize = 1024;
/// How often the plan is recomputed when the camera stays in one cube, for
/// cubes that loaded or unloaded meanwhile.
const REPLAN: Duration = Duration::from_millis(250);
/// How long a node whose sampling failed waits before it is asked again.
const RETRY: Duration = Duration::from_secs(1);
/// The plan is redone when the view's detail changes by more than this ratio
/// either way: the spyglass in and out.
const REFOCUS: f64 = 1.25;
/// The coverage layer is the plan for voxels this many times bigger on
/// screen...
pub const COVER_SCALE: f64 = 8.0;
/// ...with at most this many nodes.
pub const COVER_NODES: usize = 768;
/// A plan with more than one in this many of its nodes undrawn (nothing
/// delivered over their place: a teleport, joining) asks for the coverage
/// layer first.
const COVER_HOLES: usize = 4;
/// The longest the coverage layer holds the rest of the plan back.
const COVER_WAIT: Duration = Duration::from_millis(1500);
/// The most nodes a plan holds unless [`CubeLod::set_budget`] says
/// otherwise; the GPU's node table and arena may lower it
/// ([`CubeLod::budget`]).
pub const NODE_BUDGET: usize = 120_000;
/// The environment variable that replaces [`NODE_BUDGET`] (a number of nodes),
/// for benchmarks (`tools/cubic_far_bench.py --budget`).
pub const BUDGET_VAR: &str = "HORIZON_FAR_NODES";
/// After a jump (a move farther than [`JUMP`]) the plan waits this long at
/// most for the GPU descent's classes of the new place: without them every
/// node there counts as mixed and the plan asks for the sky too.
const DESCENT_WAIT: Duration = Duration::from_millis(300);
const JUMP: f64 = 4096.0;
/// The queue takes its sectors in a new order when the view turned this far
/// (radians).
const TURN: f64 = 0.087;
/// Above this many nodes, the passes over the whole plan (releasing nodes,
/// looking for holes) run at most every [`SCAN_EVERY`].
const SLOW_SCAN: usize = 8192;
const SCAN_EVERY: Duration = Duration::from_millis(200);
/// The most nodes [`CubeLod::removed`] lets go at a time. The GPU renderer
/// fades each out and frees it with a buffer write of its own (about a
/// microsecond each), so a whole plan let go at once after a teleport cost
/// over 200 ms in one frame at 120,000 nodes.
pub const RELEASE_BATCH: usize = 4096;

/// Progress of the far terrain, for the overlay and benchmarks
/// (`CubeLod::stats`).
#[derive(Clone, Copy, Debug, Default)]
pub struct CubeLodStats {
    /// Nodes in the current plan.
    pub planned: usize,
    /// Planned nodes handed out and drawn (or, with nothing to draw, known
    /// empty: see `uniform_skipped`).
    pub ready: usize,
    /// Planned nodes still to sample, mesh or upload.
    pub pending: usize,
    /// Planned nodes that are uniform and draw nothing (air, fluid), and
    /// nodes the planner left out because the 2D pre-pass proved their whole
    /// subtree uniform air or fluid (never requested).
    pub uniform_skipped: usize,
    /// Nodes generated on the GPU, ever.
    pub gpu_generated: u64,
    /// Nodes meshed on the GPU, ever (the renderer's count).
    pub gpu_meshed: u64,
    /// Bytes of the quad arena and the node pool in use.
    pub arena_bytes: u64,
    pub pool_bytes: u64,
    /// Every planned node is drawn or known empty, and the GPU mesher has
    /// nothing waiting.
    pub complete: bool,
    /// GPU descents run (`CubeLod::set_gpu_descent`), ever.
    pub descent_runs: u64,
    /// Nodes the last descent proved uniform (air, fluid or rock).
    pub descent_uniform: usize,
    /// Right-sized nodes the last descent found (the leaves of the GPU's
    /// levels, before the loaded cubes and the classes are applied).
    pub descent_leaves: usize,
    /// Wall time of the last descent, with its readback, in milliseconds.
    pub descent_ms: f32,
    /// The last descent filled a list: its classes are partial (safe).
    pub descent_overflow: bool,
    /// When the plan last changed.
    pub plan_changed: Option<Instant>,
    /// When the current plan last became complete; none while it is not.
    pub plan_complete: Option<Instant>,
    /// When every part of the plan's footprint last had a delivered node
    /// drawn over it (no holes, the coverage layer counting); none while some
    /// part has nothing.
    pub plan_covered: Option<Instant>,
    /// Planned nodes with nothing drawn over their place yet.
    pub holes: usize,
    /// Coverage nodes the last big change asked for first.
    pub cover_nodes: usize,
    /// The coverage layer is holding the rest of the plan back.
    pub covering: bool,
    /// The most leaves the plan may have now.
    pub budget: usize,
    /// The budget kept the plan coarser than its target.
    pub limited: bool,
    /// The target in pixels per voxel the plan was made for, and the size it
    /// reaches (bigger when the budget binds).
    pub target_px: f32,
    pub voxel_px: f32,
    /// The vertical field of view of the last update, in degrees (the spyglass
    /// narrows it).
    pub fov_deg: f32,
    /// The planner's time for the last plan, in milliseconds.
    pub plan_ms: f32,
    /// Surface tiles in the plan (counted in `planned` too), and of them
    /// those drawn.
    pub tiles: usize,
    pub tiles_ready: usize,
    /// The renderer's tiles: in its pool, their bytes, and those complex.
    pub tiles_resident: usize,
    pub tile_bytes: u64,
    pub tiles_complex: usize,
    /// The tile budget kept some tiles whole.
    pub tiles_limited: bool,
}

/// What a worker made of a node: a CPU mesh, or (the GPU path) a node for the
/// renderer's GPU mesher.
enum Product {
    Mesh(SectionMesh),
    Gpu(GpuNode),
}

struct Done {
    key: CubeKey,
    product: Option<Product>,
}

/// What a node holds on the GPU path.
#[derive(Clone, Copy, Debug)]
pub enum GpuContent {
    /// Every voxel the same, with no cover: it needs no slot and is meshed
    /// without the GPU (air and fluid draw nothing, rock its six faces).
    Uniform(Voxel),
    /// A slot of the GPU node pool (`PoolHandle`) the renderer owns from here:
    /// it frees the slot once the node is meshed, or when the node goes.
    Slot(u32),
}

/// A node handed to the renderer's GPU mesher (`CubeLod::take_gpu_nodes`).
#[derive(Debug)]
pub struct GpuNode {
    pub key: CubeKey,
    pub content: GpuContent,
    /// The columns the sky is blocked over (`CubeNode::covered_mask`).
    pub cover: Option<Box<CoverMask>>,
    /// The distinct block states of a CPU node (empty for a generated one:
    /// the generator's states are known from the start).
    pub states: Vec<u32>,
}

/// What `CubeLod::take_gpu_nodes` hands the renderer.
#[derive(Debug, Default)]
pub struct GpuBatch {
    pub nodes: Vec<GpuNode>,
    /// Material records for block states the renderer has not seen
    /// (`gfx::cube_mesh::material_record`).
    pub materials: Vec<(u32, [u32; crate::gfx::cube_mesh::MATERIAL_STRIDE])>,
    /// The biome colour table, the first time.
    pub biomes: Option<Vec<u32>>,
}

// ---------------------------------------------------------------- sectors

/// The queue sorts the plan by direction from the camera: 16 azimuths in
/// each of three bands (down, level, up), plus sector 0 for the nodes around
/// the camera, which are in view whichever way it looks. Turning the camera
/// only reorders the sectors ([`sector_order`]): no node is sorted again.
const AZIMUTHS: usize = 16;
const SECTORS: usize = 1 + 3 * AZIMUTHS;

/// The sector of `node` seen from `camera`.
fn sector(camera: [f64; 3], node: CubeKey) -> u8 {
    let size = f64::from(node.size());
    let o = node.origin();
    let d = [0, 1, 2].map(|a| f64::from(o[a]) + size / 2.0 - camera[a]);
    let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    if len <= size {
        return 0;
    }
    let band = if d[1] < -0.5 * len {
        0
    } else if d[1] > 0.5 * len {
        2
    } else {
        1
    };
    let turn = (d[2].atan2(d[0]) + std::f64::consts::PI) / std::f64::consts::TAU;
    let azimuth = ((turn * AZIMUTHS as f64) as usize).min(AZIMUTHS - 1);
    (1 + band * AZIMUTHS + azimuth) as u8
}

/// The middle direction of sector `s` (not 0).
fn sector_axis(s: usize) -> [f64; 3] {
    let (band, azimuth) = ((s - 1) / AZIMUTHS, (s - 1) % AZIMUTHS);
    let angle =
        (azimuth as f64 + 0.5) / AZIMUTHS as f64 * std::f64::consts::TAU - std::f64::consts::PI;
    let y = [-0.8, 0.0, 0.8][band];
    let h = (1.0f64 - y * y).sqrt();
    [h * angle.cos(), y, h * angle.sin()]
}

/// The sectors in the order the queue takes them for a camera looking along
/// `forward`: around the camera first, then by angle from the view's centre.
fn sector_order(forward: [f64; 3]) -> Vec<u8> {
    let mut by_angle: Vec<(f64, u8)> = (1..SECTORS)
        .map(|s| {
            let a = sector_axis(s);
            (
                -(a[0] * forward[0] + a[1] * forward[1] + a[2] * forward[2]),
                s as u8,
            )
        })
        .collect();
    by_angle.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    std::iter::once(0)
        .chain(by_angle.into_iter().map(|(_, s)| s))
        .collect()
}

fn angle_between(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let la = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    let lb = (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]).sqrt();
    if la * lb <= 0.0 || !(la * lb).is_finite() {
        return 0.0;
    }
    (dot / (la * lb)).clamp(-1.0, 1.0).acos()
}

/// The jobs, in the order the workers take them.
struct Queue {
    /// Taken first, in order: the coverage layer, then holes, new nodes and
    /// retries.
    first: VecDeque<CubeKey>,
    /// The rest of the plan by sector, nearest first in each.
    sectors: Vec<VecDeque<CubeKey>>,
    /// The order the sectors are taken in: the ones the camera faces first.
    order: Vec<u8>,
    /// Queued keys the plan dropped since: skipped when taken.
    dropped: FxHashSet<CubeKey>,
    running: FxHashSet<CubeKey>,
}

impl Default for Queue {
    fn default() -> Self {
        Self {
            first: VecDeque::new(),
            sectors: (0..SECTORS).map(|_| VecDeque::new()).collect(),
            order: (0..SECTORS as u8).collect(),
            dropped: FxHashSet::default(),
            running: FxHashSet::default(),
        }
    }
}

impl Queue {
    fn clear(&mut self) {
        self.first.clear();
        for s in &mut self.sectors {
            s.clear();
        }
        self.dropped.clear();
    }
    /// The next job.
    fn pop(&mut self) -> Option<CubeKey> {
        loop {
            let mut next = self.first.pop_front();
            if next.is_none() {
                for &s in &self.order {
                    if let Some(key) = self.sectors[usize::from(s)].pop_front() {
                        next = Some(key);
                        break;
                    }
                }
            }
            let key = next?;
            if self.dropped.remove(&key) || self.running.contains(&key) {
                continue;
            }
            return Some(key);
        }
    }
    #[cfg(test)]
    fn drain(&mut self) -> Vec<CubeKey> {
        std::iter::from_fn(|| self.pop()).collect()
    }
}

#[derive(Default)]
struct Cache {
    nodes: FxHashMap<CubeKey, (Arc<CubeNode>, u64)>,
    tick: u64,
}

impl Cache {
    /// Drops the nodes holding any of `cubes`.
    fn forget(&mut self, cubes: &[SectionPos]) {
        self.nodes
            .retain(|key, _| !cubes.iter().any(|&c| key.contains_cube(c)));
    }
    fn get(&mut self, key: CubeKey) -> Option<Arc<CubeNode>> {
        self.tick += 1;
        let tick = self.tick;
        let entry = self.nodes.get_mut(&key)?;
        entry.1 = tick;
        Some(entry.0.clone())
    }
    fn insert(&mut self, node: Arc<CubeNode>) {
        self.tick += 1;
        if self.nodes.len() >= CACHE_NODES
            && let Some(oldest) = self
                .nodes
                .iter()
                .min_by_key(|(_, (_, tick))| *tick)
                .map(|(&key, _)| key)
        {
            self.nodes.remove(&oldest);
        }
        self.nodes.insert(node.key, (node, self.tick));
    }
}

/// What the classifier thread has found of nodes, and what it is to look at.
#[derive(Default)]
struct Classes {
    known: KnownClasses,
    queue: VecDeque<CubeKey>,
    queued: FxHashSet<CubeKey>,
}

/// Classes kept in each generation of [`KnownClasses`].
const CLASS_NODES: usize = 1 << 18;

/// The classes found so far, in two generations of at most [`CLASS_NODES`]:
/// a full one becomes the older and a new one starts, so the classes of the
/// last answers are always all there. Emptied whole when full, as it was, it
/// kept only the newest descent's at Ultra (one answer holds 150,000 or so),
/// and the plan swung between that one's detail and the union's: a still
/// camera's far view went back and forth between levels, the descent asked
/// again at each plan's detail, for good.
#[derive(Default)]
struct KnownClasses {
    newer: FxHashMap<CubeKey, NodeClass>,
    older: FxHashMap<CubeKey, NodeClass>,
}

impl KnownClasses {
    fn get(&self, key: &CubeKey) -> Option<NodeClass> {
        self.newer.get(key).or_else(|| self.older.get(key)).copied()
    }

    /// Files `found`, starting a new generation first when they would not
    /// fit in this one.
    fn extend(&mut self, found: impl ExactSizeIterator<Item = (CubeKey, NodeClass)>) {
        if self.newer.len() + found.len() > CLASS_NODES {
            self.older = std::mem::take(&mut self.newer);
        }
        self.newer.extend(found);
    }

    fn insert(&mut self, key: CubeKey, class: NodeClass) {
        if self.newer.len() >= CLASS_NODES {
            self.older = std::mem::take(&mut self.newer);
        }
        self.newer.insert(key, class);
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.newer.len() + self.older.len()
    }
}

impl Shared {
    fn new() -> Self {
        Self {
            classes: Mutex::default(),
            classes_wake: Condvar::new(),
            queue: Mutex::default(),
            wake: Condvar::new(),
            stop: AtomicBool::new(false),
            cache: Mutex::default(),
            gpu_path: AtomicBool::new(false),
            descent_on: AtomicBool::new(false),
            descent: Mutex::default(),
            descent_wake: Condvar::new(),
            classes_dirty: AtomicBool::new(false),
            planner: Mutex::default(),
            planner_wake: Condvar::new(),
        }
    }
}

/// The request the descent thread is to run next (the latest wins), and what
/// the last run found.
#[derive(Default)]
struct DescentSlot {
    request: Option<DescentRequest>,
    busy: bool,
    runs: u64,
    uniform: usize,
    leaves: usize,
    millis: f32,
    overflow: bool,
}

struct Shared {
    classes: Mutex<Classes>,
    classes_wake: Condvar,
    queue: Mutex<Queue>,
    wake: Condvar,
    stop: AtomicBool,
    cache: Mutex<Cache>,
    /// The workers make [`GpuNode`]s (`CubeLod::set_gpu_path`).
    gpu_path: AtomicBool,
    /// Classes come from the GPU descent (`CubeLod::set_gpu_descent`), not
    /// from the classifier thread.
    descent_on: AtomicBool,
    descent: Mutex<DescentSlot>,
    descent_wake: Condvar,
    /// The descent added classes the plan has not looked at yet.
    classes_dirty: AtomicBool,
    /// The planner thread's job and answer (one at a time).
    planner: Mutex<PlanSlot>,
    planner_wake: Condvar,
}

// ---------------------------------------------------------------- planner

/// Which cubes of the loaded box `is_loaded` reported, taken on the game
/// thread for the planner thread.
#[derive(Clone, Debug, Default)]
struct LoadedSnapshot {
    lo: [i32; 3],
    size: [i32; 3],
    bits: Vec<u64>,
}

impl LoadedSnapshot {
    fn take(center: SectionPos, loaded: LoadedBox, is_loaded: &dyn Fn(SectionPos) -> bool) -> Self {
        let (h, v) = (loaded.horizontal.max(0), loaded.vertical.max(0));
        let lo = [center.x - h, center.y - v, center.z - h];
        let size = [2 * h + 1, 2 * v + 1, 2 * h + 1];
        let n = (size[0] as usize) * (size[1] as usize) * (size[2] as usize);
        let mut bits = vec![0u64; n.div_ceil(64)];
        let mut i = 0usize;
        for y in 0..size[1] {
            for z in 0..size[2] {
                for x in 0..size[0] {
                    if is_loaded(SectionPos::new(lo[0] + x, lo[1] + y, lo[2] + z)) {
                        bits[i >> 6] |= 1 << (i & 63);
                    }
                    i += 1;
                }
            }
        }
        Self { lo, size, bits }
    }
    fn get(&self, cube: SectionPos) -> bool {
        let r = [
            i64::from(cube.x) - i64::from(self.lo[0]),
            i64::from(cube.y) - i64::from(self.lo[1]),
            i64::from(cube.z) - i64::from(self.lo[2]),
        ];
        if (0..3).any(|a| r[a] < 0 || r[a] >= i64::from(self.size[a])) {
            return false;
        }
        let i = ((r[1] * i64::from(self.size[2]) + r[2]) * i64::from(self.size[0]) + r[0]) as usize;
        self.bits[i >> 6] >> (i & 63) & 1 == 1
    }
}

/// What a plan job needs for surface tiles.
struct SurfaceJob {
    /// The tiles' headers known so far.
    info: Arc<FxHashMap<TileKey, TileInfo>>,
    sea_level: i32,
    budget: usize,
    /// The tiles in place, for the hysteresis.
    previous: Arc<Vec<TileKey>>,
}

/// A plan for the planner thread to make.
struct PlanJob {
    /// [`CubeLod::clear`] counts up: a job of an older count starts afresh.
    epoch: u64,
    camera: [f64; 3],
    max_distance: f64,
    rule: Rule,
    loaded: LoadedBox,
    snapshot: LoadedSnapshot,
    /// The plan in place, for the hysteresis.
    previous: Arc<Vec<CubeKey>>,
    /// Plan surface tiles at the coarse levels.
    surface: Option<SurfaceJob>,
}

/// A far item: a cube node or a surface tile.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum FarKey {
    Cube(CubeKey),
    Tile(TileKey),
}

/// A plan made, as the change from the one before.
struct PlanDone {
    epoch: u64,
    /// The plan, nearest first.
    leaves: Arc<Vec<CubeKey>>,
    changed: bool,
    added: Vec<CubeKey>,
    removed: Vec<CubeKey>,
    /// The plan's tiles, nearest first, and the change.
    tiles: Arc<Vec<TileKey>>,
    tiles_added: Vec<TileKey>,
    tiles_removed: Vec<TileKey>,
    /// The plan with each node's sector, nearest first (empty when unchanged).
    order: Vec<(u8, CubeKey)>,
    /// The coverage layer's leaves and tiles (empty when unchanged).
    cover: FxHashSet<CubeKey>,
    cover_tiles: FxHashSet<TileKey>,
    /// The plan in tree order ([`tree_order`], empty when unchanged): the
    /// nodes inside any node are a run of it.
    tree: Vec<(u128, CubeKey)>,
    /// Cube leaves and tiles by footprint in Z order ([`footprint_order`]):
    /// those inside a footprint are a run of it.
    footprints: Vec<(u64, u8, FarKey)>,
    pruned: usize,
    limited: bool,
    coarsened: f64,
    tiles_limited: bool,
    elapsed: Duration,
}

/// Where a node's corner falls on the Z-order curve of the world's cubes:
/// the nodes inside a node of level `l` are exactly those of a lower level
/// whose code lies in `[code, code + 8^l)`.
fn tree_order(key: CubeKey) -> u128 {
    // Cube coordinates offset to be non-negative: 24 bits each.
    let axis = |c: i32| ((i64::from(c) << key.level) + (1 << 23)) as u64;
    let (x, y, z) = (axis(key.x), axis(key.y), axis(key.z));
    let mut code = 0u128;
    for bit in 0..24 {
        code |= u128::from(x >> bit & 1) << (3 * bit)
            | u128::from(z >> bit & 1) << (3 * bit + 1)
            | u128::from(y >> bit & 1) << (3 * bit + 2);
    }
    code
}

/// [`tree_order`] for the footprint (X and Z) of a node or tile of `level` at
/// `(x, z)`: the footprints inside one of level `l` are exactly those of a
/// level up to `l` whose code lies in `[code, code + 4^l)`.
fn footprint_order(level: u8, x: i32, z: i32) -> u64 {
    let axis = |c: i32| ((i64::from(c) << level) + (1 << 23)) as u64;
    let (x, z) = (axis(x), axis(z));
    let mut code = 0u64;
    for bit in 0..24 {
        code |= (x >> bit & 1) << (2 * bit) | (z >> bit & 1) << (2 * bit + 1);
    }
    code
}

#[derive(Default)]
struct PlanSlot {
    job: Option<PlanJob>,
    busy: bool,
    done: Option<PlanDone>,
}

/// The plan a job asks for, with `rule` and no hysteresis when `fresh`.
fn plan_job(job: &PlanJob, rule: &Rule, fresh: bool, class: &dyn Fn(CubeKey) -> NodeClass) -> Plan {
    let is_loaded = |cube: SectionPos| job.snapshot.get(cube);
    let info = |key: TileKey| job.surface.as_ref().and_then(|s| s.info.get(&key).copied());
    plan_far(&PlanInput {
        camera: job.camera,
        max_distance: job.max_distance,
        rule,
        loaded: job.loaded,
        is_loaded: &is_loaded,
        previous: if fresh { &[] } else { job.previous.as_slice() },
        class,
        surface: job.surface.as_ref().map(|s| Surface {
            info: &info,
            sea_level: s.sea_level,
            budget: s.budget,
            previous: if fresh { &[] } else { s.previous.as_slice() },
        }),
    })
}

/// The change from `last` to `next`: what `next` adds, and what it drops.
fn diff<K: Copy + Eq + std::hash::Hash>(
    next: &[K],
    last: &FxHashSet<K>,
) -> (Vec<K>, Vec<K>, FxHashSet<K>) {
    let set: FxHashSet<K> = next.iter().copied().collect();
    let added = next.iter().copied().filter(|k| !last.contains(k)).collect();
    let removed = last.iter().copied().filter(|k| !set.contains(k)).collect();
    (added, removed, set)
}

/// Makes the plan of `job`, and its coverage layer, as the change from
/// `last` and `last_tiles` (the plan before, which becomes this one).
fn make_plan(
    job: &PlanJob,
    last: &mut FxHashSet<CubeKey>,
    last_tiles: &mut FxHashSet<TileKey>,
    class: &dyn Fn(CubeKey) -> NodeClass,
) -> PlanDone {
    let started = Instant::now();
    let plan = plan_job(job, &job.rule, false, class);
    let (added, removed, next) = diff(&plan.leaves, last);
    let (tiles_added, tiles_removed, next_tiles) = diff(&plan.tiles, last_tiles);
    let changed = !(added.is_empty()
        && removed.is_empty()
        && tiles_added.is_empty()
        && tiles_removed.is_empty());
    *last = next;
    *last_tiles = next_tiles;
    let mut done = PlanDone {
        epoch: job.epoch,
        leaves: job.previous.clone(),
        changed,
        added,
        removed,
        tiles: job
            .surface
            .as_ref()
            .map_or_else(Arc::default, |s| s.previous.clone()),
        tiles_added,
        tiles_removed,
        order: Vec::new(),
        cover: FxHashSet::default(),
        cover_tiles: FxHashSet::default(),
        tree: Vec::new(),
        footprints: Vec::new(),
        pruned: plan.pruned,
        limited: plan.limited,
        coarsened: plan.coarsened,
        tiles_limited: plan.tiles_limited,
        elapsed: Duration::ZERO,
    };
    if changed {
        done.order = plan
            .leaves
            .iter()
            .map(|&k| (sector(job.camera, k), k))
            .collect();
        done.tree = plan.leaves.iter().map(|&k| (tree_order(k), k)).collect();
        done.tree.sort_unstable();
        done.footprints = plan
            .leaves
            .iter()
            .map(|&k| (footprint_order(k.level, k.x, k.z), k.level, FarKey::Cube(k)))
            .chain(
                plan.tiles
                    .iter()
                    .map(|&t| (footprint_order(t.level, t.x, t.z), t.level, FarKey::Tile(t))),
            )
            .collect();
        done.footprints.sort_unstable();
        let cover = plan_job(
            job,
            &job.rule.coarser(COVER_SCALE, COVER_NODES),
            true,
            class,
        );
        done.cover = cover.leaves.into_iter().collect();
        done.cover_tiles = cover.tiles.into_iter().collect();
        done.leaves = Arc::new(plan.leaves);
        done.tiles = Arc::new(plan.tiles);
    }
    done.elapsed = started.elapsed();
    done
}

/// The planner thread: plans off the game thread, one job at a time.
fn plan_loop(shared: &Shared) {
    let mut last = FxHashSet::default();
    let mut last_tiles = FxHashSet::default();
    let mut epoch = None;
    loop {
        let job = {
            let mut slot = shared.planner.lock().unwrap();
            loop {
                if shared.stop.load(Ordering::Relaxed) {
                    return;
                }
                if let Some(job) = slot.job.take() {
                    slot.busy = true;
                    break job;
                }
                slot = shared.planner_wake.wait(slot).unwrap();
            }
        };
        if epoch != Some(job.epoch) {
            epoch = Some(job.epoch);
            last.clear();
            last_tiles.clear();
        }
        let class = |key: CubeKey| classify_cached(shared, key);
        let done = make_plan(&job, &mut last, &mut last_tiles, &class);
        let mut slot = shared.planner.lock().unwrap();
        slot.busy = false;
        slot.done = Some(done);
    }
}

/// Whether the plan waits for the coverage layer.
#[derive(Debug)]
enum Phase {
    Refine,
    /// The coverage layer (`wave`: coverage nodes and the coarsest planned
    /// nodes, one over each hole) is all that is queued, since `since`.
    Cover {
        since: Instant,
        wave: Vec<FarKey>,
    },
}

/// A planned node with nothing drawn over its place, and the coverage leaf
/// (a node or a tile) over it when the coverage layer has one.
type Hole = (CubeKey, Option<FarKey>);

/// What the last plan job was asked with.
#[derive(Clone, Copy, Debug)]
struct Asked {
    cube: SectionPos,
    camera: [f64; 3],
    max_distance: f64,
    loaded: LoadedBox,
    rule: Rule,
    at: Instant,
}

/// Whether the plan for `now` differs enough from the one for `before` to be
/// made again: the detail changed by [`REFOCUS`], the zoom came or went, a
/// zoomed view turned past its margin, or the budget moved a tenth.
fn refocused(before: &Rule, now: &Rule, view: &FarView) -> bool {
    let off = |a: f64, b: f64| a > b * REFOCUS || b > a * REFOCUS;
    if off(before.in_view, now.in_view) || off(before.out_of_view, now.out_of_view) {
        return true;
    }
    if before.budget.abs_diff(now.budget) * 10 > before.budget.max(1) {
        return true;
    }
    match (&before.cone, &now.cone) {
        (None, None) => false,
        (Some(a), Some(b)) => angle_between(a.axis, b.axis) > view.turn_margin(),
        _ => true,
    }
}

/// How far the arena's room for nodes may move before the budget follows it,
/// as (lower, upper) fractions of the budget held: the bytes a node takes move
/// with the plan (finer nodes, more empty ones), and a budget that followed
/// every move asked for a plan of another size each time (`refocused`), which
/// moved the bytes again: at Ultra, where the arena binds, the far terrain
/// went back and forth between levels. The lower bound keeps the arena's
/// room for the nodes a plan replaces mostly whole.
const ARENA_CAP_BAND: (f64, f64) = (0.9, 1.25);

/// Whether the arena's room for `fits` nodes leaves a held cap of `cap` as
/// it is.
fn arena_cap_holds(cap: usize, fits: usize) -> bool {
    let (lo, hi) = ARENA_CAP_BAND;
    (fits as f64) >= cap as f64 * lo && (fits as f64) <= cap as f64 * hi
}

/// The most surface tiles a plan holds.
pub const TILE_BUDGET: usize = 3584;
/// Surface tiles the renderer's pool holds (`Renderer::use_surface_tiles`):
/// a plan's, the ones it replaces and the coverage layer's (about 110 MB).
pub const TILE_CAPACITY: u32 = 8192;
/// Tile headers kept for the plan past which those of tiles neither planned
/// nor drawn are forgotten.
const TILE_INFO_KEPT: usize = 4 * TILE_BUDGET;
/// Levels a coverage leaf is coarser than the plan, about: `log2(COVER_SCALE)`.
const COVER_LEVELS: u8 = 3;

/// What the renderer's surface tiles are to do ([`CubeLod::take_tile_work`]).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TileWork {
    /// Tiles to generate (`SurfaceTiles::request`), coverage first, then
    /// nearest first.
    pub request: Vec<TileKey>,
    /// Tiles to fade out (`SurfaceTiles::retire`): what replaces them is in.
    pub retire: Vec<TileKey>,
    /// Requests to drop, never drawn (`SurfaceTiles::free`).
    pub free: Vec<TileKey>,
}

/// The plan's surface tiles on the game thread.
#[derive(Default)]
struct TileState {
    /// Tiles run ([`CubeLod::set_surface`]).
    on: bool,
    sea_level: i32,
    plan: Arc<Vec<TileKey>>,
    wanted: FxHashSet<TileKey>,
    /// Generated and drawn (their header read back), not retired.
    delivered: FxHashSet<TileKey>,
    /// Delivered and not planned: drawn until what replaces them is in.
    stale: FxHashSet<TileKey>,
    /// Requested and not delivered yet.
    asked: FxHashSet<TileKey>,
    /// Planned and not delivered.
    missing: usize,
    /// The headers known, for the planner.
    info: Arc<FxHashMap<TileKey, TileInfo>>,
    /// The coverage layer's tiles for the current plan.
    cover: FxHashSet<TileKey>,
    /// Planned tiles with nothing drawn over their footprint.
    holes: Vec<TileKey>,
    work: TileWork,
}

/// The far terrain of a cubic level, built in the background.
///
/// ```ignore
/// // Once, when the level loads:
/// let mut far = CubeLod::new(Box::new(generator_sampler), context.clone(), air);
/// // Every frame:
/// let meshes = far.update(camera, max_distance, &view, LoadedBox { horizontal: h, vertical: v },
///                         |cube| world.has_cube(cube));
/// renderer.queue_cube_lods(meshes, far.removed());
/// renderer.upload_cube_lods(budget);
/// // On leaving the level:
/// far.clear();
/// renderer.clear_cube_lods();
/// ```
///
/// Meshes are handed out once per node. A node leaves through [`removed`]
/// only when every node that now covers its place has been handed out (or
/// none is wanted: out of range), so the renderer never opens a hole.
///
/// The plan is made on a thread of its own ([`plan_far`] with the view's
/// [`Rule`]). After a big change (a teleport, joining: a quarter of the plan
/// has nothing drawn over it) only the coverage layer is queued: the plan for
/// voxels [`COVER_SCALE`] times coarser, a few hundred nodes, over the holes.
/// Once it is in and shown, the rest of the plan is queued, holes first, then
/// by direction (the view's first) and distance; the coverage nodes stay
/// drawn until the nodes under them are in.
///
/// With surface tiles on ([`CubeLod::set_surface`]), the coarse levels are
/// tiles the renderer generates itself: [`CubeLod::take_tile_work`] hands it
/// the requests, fades and drops, and [`CubeLod::surface_feedback`] notes the
/// tiles it generated (their headers: a complex tile is planned again as cube
/// nodes). A tile stays drawn until everything planned in its footprint is
/// in, and a node under a planned tile until the tile is.
///
/// [`removed`]: CubeLod::removed
pub struct CubeLod {
    shared: Arc<Shared>,
    results: mpsc::Receiver<Done>,
    threads: Vec<thread::JoinHandle<()>>,
    /// The current plan, nearest first.
    plan: Arc<Vec<CubeKey>>,
    wanted: FxHashSet<CubeKey>,
    /// The plan with each node's sector, nearest first: the queue's source.
    order: Vec<(u8, CubeKey)>,
    /// The plan in tree order (`tree_order`).
    tree: Vec<(u128, CubeKey)>,
    /// The plan's leaves and tiles by footprint (`footprint_order`).
    footprints: Vec<(u64, u8, FarKey)>,
    tiles: TileState,
    /// The sampler's world can have surface tiles (`CubeSampler::surface`).
    surface_allowed: bool,
    surface_stats: Option<crate::gfx::surface_tile::SurfaceStats>,
    /// The coverage layer's leaves for the current plan.
    cover_plan: FxHashSet<CubeKey>,
    /// Coverage nodes asked for (not in the plan): their results are kept.
    cover: FxHashSet<CubeKey>,
    /// Planned nodes with nothing drawn over their place, and the coverage
    /// leaf over each when the plan had one.
    holes: Vec<Hole>,
    /// Planned nodes holding an older plan's (`find_holes`): queued first.
    holding: Vec<CubeKey>,
    holes_dirty: bool,
    holes_at: Instant,
    phase: Phase,
    /// Handed out and not reported removed.
    delivered: FxHashSet<CubeKey>,
    /// Delivered nodes not in the plan: coverage nodes, and an older plan's
    /// nodes still covering a place.
    stale: FxHashSet<CubeKey>,
    /// Planned nodes not delivered.
    missing: usize,
    failed: FxHashMap<CubeKey, Instant>,
    retry_at: Instant,
    release_dirty: bool,
    released_at: Instant,
    /// Stale nodes found releasable, handed out [`RELEASE_BATCH`] at a time.
    releasing: Vec<CubeKey>,
    asked: Option<Asked>,
    /// A replan is owed (new classes came while the planner was busy).
    replan: bool,
    /// [`CubeLod::clear`]s so far.
    epoch: u64,
    camera: [f64; 3],
    /// The direction the queue's sector order was made for.
    facing: Option<[f64; 3]>,
    budget: usize,
    /// The sampler's GPU generator, when it has one.
    backend: Option<GpuBackend>,
    /// Nodes finished for the GPU mesher, not yet taken.
    gpu_out: Vec<GpuNode>,
    context: Option<Arc<WorkerContext>>,
    /// Block states the renderer has material records for.
    known_states: FxHashSet<u32>,
    biomes_sent: bool,
    air: BlockStateId,
    stats: CubeLodStats,
    /// Delivered nodes that draw nothing, and how many of them are planned.
    empty: FxHashSet<CubeKey>,
    empty_planned: usize,
    /// Nodes the planner left out as uniform, from the class cache.
    pruned: usize,
    coarsened: f64,
    render: Option<crate::gfx::cube_far::FarStats>,
    /// The most nodes the quad arena holds two plans of, held between
    /// [`ARENA_CAP_BAND`]'s bounds ([`CubeLod::set_render_stats`]).
    arena_cap: Option<usize>,
    /// The sampler has a GPU descent.
    has_descent: bool,
    /// What the descent was last asked for, and its run count then.
    descent_asked: Option<(SectionPos, f64, LoadedBox, u64)>,
    descent_runs_asked: u64,
    /// The plan holds for the descent's answer after a jump, since.
    jumped: Option<Instant>,
}

impl CubeLod {
    /// `air` is the registry's air state; voxels of any other state are drawn.
    pub fn new(
        sampler: Box<dyn CubeSampler>,
        context: Arc<WorkerContext>,
        air: BlockStateId,
    ) -> Self {
        let wanted_threads = (horizon_threads::worker_count() / 2).clamp(2, 4);
        let backend = sampler.gpu();
        let surface_allowed = sampler.surface();
        // One more fork classifies nodes for the planner, on a thread of its own.
        let classifier = backend.is_some().then(|| sampler.fork()).flatten();
        let descent = sampler.descent();
        let mut samplers = Vec::new();
        while samplers.len() + 1 < wanted_threads
            && let Some(fork) = sampler.fork()
        {
            samplers.push(fork);
        }
        samplers.push(sampler);
        let shared = Arc::new(Shared::new());
        let (tx, results) = mpsc::channel();
        let mut threads: Vec<thread::JoinHandle<()>> = classifier
            .map(|mut sampler| {
                let shared = shared.clone();
                thread::Builder::new()
                    .name("Cube far classes".into())
                    .spawn(move || classify_loop(&shared, sampler.as_mut()))
                    .expect("spawn the cube classifier thread")
            })
            .into_iter()
            .collect();
        let has_descent = descent.is_some();
        if let Some(descent) = descent {
            let shared = shared.clone();
            threads.push(
                thread::Builder::new()
                    .name("Cube far descent".into())
                    .spawn(move || descent_loop(&shared, &descent))
                    .expect("spawn the cube descent thread"),
            );
        }
        {
            let shared = shared.clone();
            threads.push(
                thread::Builder::new()
                    .name("Cube far planner".into())
                    .spawn(move || plan_loop(&shared))
                    .expect("spawn the cube planner thread"),
            );
        }
        threads.extend(samplers.into_iter().map(|sampler| {
            let (shared, tx, context) = (shared.clone(), tx.clone(), context.clone());
            let backend = backend.clone();
            thread::Builder::new()
                .name("Cube far terrain".into())
                .spawn(move || work(shared, tx, context, sampler, air, backend))
                .expect("spawn a cube far terrain thread")
        }));
        let mut lod = Self::assemble(shared, results, threads, backend, Some(context), air);
        lod.has_descent = has_descent;
        lod.surface_allowed = surface_allowed;
        // Benchmarks and autoresearch loops set the node budget from outside.
        if let Some(nodes) = std::env::var(BUDGET_VAR)
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
        {
            lod.set_budget(nodes);
        }
        lod
    }

    fn assemble(
        shared: Arc<Shared>,
        results: mpsc::Receiver<Done>,
        threads: Vec<thread::JoinHandle<()>>,
        backend: Option<GpuBackend>,
        context: Option<Arc<WorkerContext>>,
        air: BlockStateId,
    ) -> Self {
        let now = Instant::now();
        Self {
            shared,
            results,
            threads,
            plan: Arc::default(),
            wanted: FxHashSet::default(),
            order: Vec::new(),
            tree: Vec::new(),
            footprints: Vec::new(),
            tiles: TileState {
                sea_level: backend.as_ref().map_or(63, |b| b.params.sea_level),
                ..TileState::default()
            },
            surface_allowed: false,
            surface_stats: None,
            cover_plan: FxHashSet::default(),
            cover: FxHashSet::default(),
            holes: Vec::new(),
            holding: Vec::new(),
            holes_dirty: false,
            holes_at: now,
            phase: Phase::Refine,
            delivered: FxHashSet::default(),
            stale: FxHashSet::default(),
            missing: 0,
            failed: FxHashMap::default(),
            retry_at: now,
            release_dirty: false,
            released_at: now,
            releasing: Vec::new(),
            asked: None,
            replan: false,
            epoch: 0,
            camera: [0.0; 3],
            facing: None,
            budget: NODE_BUDGET,
            gpu_out: Vec::new(),
            context,
            known_states: FxHashSet::default(),
            biomes_sent: false,
            air,
            stats: CubeLodStats::default(),
            empty: FxHashSet::default(),
            empty_planned: 0,
            pruned: 0,
            coarsened: 1.0,
            render: None,
            arena_cap: None,
            has_descent: false,
            descent_asked: None,
            descent_runs_asked: 0,
            jumped: None,
            backend,
        }
    }

    /// The renderer's GPU far terrain counters (`Renderer::cube_far_stats`),
    /// for [`CubeLod::stats`].
    pub fn set_render_stats(&mut self, stats: Option<crate::gfx::cube_far::FarStats>) {
        self.render = stats;
        if let Some(r) = stats
            && r.nodes >= 1024
            && r.arena_bytes > 0
        {
            let per_node = (r.arena_bytes / r.nodes as u64).max(1);
            let fits = (r.arena_capacity_bytes / 2 / per_node) as usize;
            self.arena_cap = Some(match self.arena_cap {
                Some(cap) if arena_cap_holds(cap, fits) => cap,
                _ => fits,
            });
        }
    }

    /// The renderer's surface tile counters (`Renderer::surface_stats`).
    pub fn set_surface_stats(&mut self, stats: Option<crate::gfx::surface_tile::SurfaceStats>) {
        self.surface_stats = stats;
    }

    /// Whether this world's far terrain can have surface tiles: a Custom world
    /// without `cubeAreas` (`CubeSampler::surface`) whose generator runs on
    /// the GPU with its climate pass.
    pub fn surface_allowed(&self) -> bool {
        self.surface_allowed
            && self
                .backend
                .as_ref()
                .is_some_and(|b| b.biome_params.is_some())
    }

    /// Turns surface tiles on (the renderer runs them:
    /// `Renderer::use_surface_tiles`) or off. They need the GPU path and a
    /// world that allows them. A change forgets every node handed out, as
    /// [`CubeLod::clear`]: clear the renderer's too. Returns whether it changed.
    pub fn set_surface(&mut self, on: bool) -> bool {
        let on = on && self.surface_allowed() && self.gpu_path();
        if self.tiles.on == on {
            return false;
        }
        self.clear();
        self.tiles.on = on;
        true
    }

    /// Whether the plan has surface tiles.
    pub fn surface_on(&self) -> bool {
        self.tiles.on
    }

    /// Notes the tiles the renderer generated since the last call: `info`
    /// reads a tile's header once it is generated and read back
    /// (`Renderer::surface_tile_info`). Call each frame before `update`.
    pub fn surface_feedback(&mut self, info: impl Fn(TileKey) -> Option<TileInfo>) {
        if self.tiles.asked.is_empty() {
            return;
        }
        let arrived: Vec<(TileKey, TileInfo)> = self
            .tiles
            .asked
            .iter()
            .filter_map(|&k| info(k).map(|i| (k, i)))
            .collect();
        for (key, info) in arrived {
            self.deliver_tile(key, info);
        }
    }

    /// Notes a tile as drawn, and its header for the plan: a complex tile is
    /// planned again as cube nodes, a tall one refines as near as it reaches.
    fn deliver_tile(&mut self, key: TileKey, info: TileInfo) {
        self.tiles.asked.remove(&key);
        if !self.tiles.delivered.insert(key) {
            return;
        }
        self.release_dirty = true;
        self.holes_dirty = true;
        if self.tiles.wanted.contains(&key) {
            self.tiles.missing = self.tiles.missing.saturating_sub(1);
        } else {
            self.tiles.stale.insert(key);
        }
        if self.tiles.info.get(&key) == Some(&info) {
            return;
        }
        let (wanted, delivered) = (&self.tiles.wanted, &self.tiles.delivered);
        let map = Arc::make_mut(&mut self.tiles.info);
        if map.len() >= TILE_INFO_KEPT {
            map.retain(|k, _| wanted.contains(k) || delivered.contains(k));
        }
        map.insert(key, info);
        if (info.complex() && key.level < SURFACE_LEVEL.saturating_add(COMPLEX_LIFT))
            || info.highest > self.tiles.sea_level + UNKNOWN_ABOVE
        {
            self.replan = true;
        }
    }

    /// What the renderer's surface tiles are to do since the last call
    /// (`Renderer::queue_surface_tiles`). Call after [`CubeLod::removed`],
    /// which finds the tiles to fade out.
    pub fn take_tile_work(&mut self) -> TileWork {
        std::mem::take(&mut self.tiles.work)
    }

    /// The most nodes a plan may hold (a terrain memory target, in nodes).
    pub fn set_budget(&mut self, nodes: usize) {
        self.budget = nodes.max(COVER_NODES);
    }

    /// The budget for the next plan: [`CubeLod::set_budget`]'s, lowered on
    /// the GPU path so the plan and the nodes it replaces fit in the node
    /// table and the quad arena (each holds two plans' worth at the quads
    /// per node measured so far).
    pub fn budget(&self) -> usize {
        let mut budget = self.budget;
        if self.gpu_path() {
            budget = budget.min(crate::gfx::cube_far::MAX_NODES as usize / 2);
            if let Some(cap) = self.arena_cap {
                budget = budget.min(cap);
            }
        }
        budget.max(COVER_NODES)
    }

    /// Progress of the plan. `complete` is exact: every planned node was
    /// handed out (a mesh, or a GPU node) and, on the GPU path, the renderer
    /// holds no node still waiting for its mesh; nodes that draw nothing count
    /// as ready.
    pub fn stats(&self) -> CubeLodStats {
        let mut out = self.stats;
        out.planned = self.plan.len() + self.tiles.plan.len();
        out.pending = self.missing + self.tiles.missing;
        out.ready = out.planned - out.pending;
        out.tiles = self.tiles.plan.len();
        out.tiles_ready = self.tiles.plan.len() - self.tiles.missing;
        if let Some(s) = self.surface_stats {
            out.tiles_resident = s.resident;
            out.tile_bytes = s.bytes;
            out.tiles_complex = s.complex;
        }
        out.uniform_skipped = self.pruned + self.empty_planned;
        if let Some(render) = self.render {
            out.gpu_meshed = render.meshed + render.uniform;
            out.arena_bytes = render.arena_bytes;
            out.pool_bytes = render.pool_bytes;
        }
        out.complete = self.is_complete();
        out.holes = self.holes.len() + self.tiles.holes.len();
        out.covering = matches!(self.phase, Phase::Cover { .. });
        out.budget = self.asked.map_or(self.budget(), |a| a.rule.budget);
        if let Ok(slot) = self.shared.descent.lock() {
            out.descent_runs = slot.runs;
            out.descent_uniform = slot.uniform;
            out.descent_leaves = slot.leaves;
            out.descent_ms = slot.millis;
            out.descent_overflow = slot.overflow;
        }
        out
    }

    /// Everything handed out has been meshed and shown: the renderer holds
    /// nothing waiting (always, on the CPU path).
    fn shown(&self) -> bool {
        self.gpu_out.is_empty()
            && (!self.gpu_path() || self.render.is_some_and(|r| r.pending == 0 && !r.meshing))
    }

    fn is_complete(&self) -> bool {
        !(self.plan.is_empty() && self.tiles.plan.is_empty())
            && self.missing == 0
            && self.tiles.missing == 0
            && self.shown()
    }

    fn track_completion(&mut self, now: Instant) {
        match (self.is_complete(), self.stats.plan_complete) {
            (true, None) => self.stats.plan_complete = Some(now),
            (false, Some(_)) => self.stats.plan_complete = None,
            _ => {}
        }
        let holes = !(self.holes.is_empty() && self.tiles.holes.is_empty());
        let covered = !(self.plan.is_empty() && self.tiles.plan.is_empty())
            && !holes
            && matches!(self.phase, Phase::Refine)
            && self.shown();
        match (covered, self.stats.plan_covered) {
            (true, None) => self.stats.plan_covered = Some(now),
            (false, Some(_)) if holes => self.stats.plan_covered = None,
            _ => {}
        }
    }

    /// Whether nodes take the GPU path now (`set_gpu_path`).
    pub fn on_gpu_path(&self) -> bool {
        self.shared.gpu_path.load(Ordering::Relaxed)
    }

    /// The sampler's GPU generator, if it has one: the device the renderer's
    /// GPU mesher must run on (`gfx::cube_mesh`) to read its nodes.
    pub fn gpu_backend(&self) -> Option<&GpuBackend> {
        self.backend.as_ref()
    }

    /// Switches between the GPU path (`on`, and the sampler has a GPU
    /// generator: workers hand out [`GpuNode`]s, `take_gpu_nodes`) and the CPU
    /// path (meshes from `update`). A change forgets every node handed out, as
    /// [`CubeLod::clear`]: clear the renderer's too (`clear_cube_lods`).
    /// Returns whether it changed.
    pub fn set_gpu_path(&mut self, on: bool) -> bool {
        let on = on && self.backend.is_some();
        if self.shared.gpu_path.load(Ordering::Relaxed) == on {
            return false;
        }
        self.shared.gpu_path.store(on, Ordering::Relaxed);
        self.clear();
        true
    }

    /// Whether workers make [`GpuNode`]s.
    pub fn gpu_path(&self) -> bool {
        self.shared.gpu_path.load(Ordering::Relaxed)
    }

    /// Switches the planner's classes between the GPU descent (`on`, and the
    /// sampler has one: `GpuCubeSampler::with_descent`) and the classifier
    /// thread's 2D pre-pass. With the descent, each replan's camera cube sends
    /// one request to the GPU, which walks the octree level by level, proves
    /// whole subtrees uniform from the density bound and returns their keys;
    /// the plan, hysteresis and loaded cubes stay the CPU's, and nodes the
    /// descent has not classified yet count as mixed. The nodes a plan keeps
    /// are those of the CPU plan minus the proven-uniform subtrees. Returns
    /// whether it changed.
    pub fn set_gpu_descent(&mut self, on: bool) -> bool {
        let on = on && self.has_descent;
        if self.shared.descent_on.swap(on, Ordering::Relaxed) == on {
            return false;
        }
        self.asked = None;
        self.descent_asked = None;
        true
    }

    /// Whether the planner's classes come from the GPU descent.
    pub fn gpu_descent(&self) -> bool {
        self.shared.descent_on.load(Ordering::Relaxed)
    }

    /// Sends the descent its request for this camera (once per cube, range,
    /// box and detail), splitting as finely as the plan's out-of-view rule
    /// does (a zoom's finer nodes in view are not worth classifying the
    /// whole sphere for). Returns whether the plan should hold for its
    /// answer: the first plan always, and for [`DESCENT_WAIT`] after a jump.
    fn ask_descent(
        &mut self,
        camera: [f64; 3],
        max_distance: f64,
        loaded: LoadedBox,
        rule: &Rule,
        now: Instant,
    ) -> bool {
        let cube = cube_of(camera);
        // Asked again when it moves a fifth or more: not for every small
        // change of the budget's reach.
        let detail = rule.out_of_view / self.coarsened.max(1.0) * MERGE;
        let bits = (detail.max(1e-3).log2() * 4.0).round() as i64 as u64;
        let asked = (cube, max_distance, loaded, bits);
        if self.descent_asked != Some(asked) {
            self.descent_asked = Some(asked);
            let mut slot = self.shared.descent.lock().unwrap();
            self.descent_runs_asked = slot.runs;
            slot.request = Some(DescentRequest {
                camera,
                max_distance,
                // Up to `MERGE` times the plain rule: the plan's hysteresis.
                detail,
                min_level: MIN_LEVEL,
                loaded: Some(([cube.x, cube.y, cube.z], loaded)),
            });
            drop(slot);
            self.shared.descent_wake.notify_one();
        }
        let slot = self.shared.descent.lock().unwrap();
        let answered = slot.runs > self.descent_runs_asked && slot.request.is_none() && !slot.busy;
        if slot.runs == 0 {
            return slot.request.is_some() || slot.busy;
        }
        drop(slot);
        if answered {
            self.jumped = None;
        }
        self.jumped
            .is_some_and(|at| !answered && now.saturating_duration_since(at) < DESCENT_WAIT)
    }

    /// The nodes finished for the GPU mesher since the last call (their slots
    /// are the caller's from here), for the nodes in the plan.
    pub fn take_gpu_nodes(&mut self) -> GpuBatch {
        let nodes = std::mem::take(&mut self.gpu_out);
        let mut wanted: Vec<u32> = Vec::new();
        if let Some(backend) = &self.backend {
            wanted.extend(&backend.states);
        }
        for node in &nodes {
            if let GpuContent::Uniform(voxel) = node.content {
                wanted.push(voxel.state.0);
            }
            wanted.extend(&node.states);
        }
        let mut materials = Vec::new();
        for state in wanted.into_iter().filter(|_| self.context.is_some()) {
            let context = self.context.as_ref().expect("a context");
            if state != self.air.0
                && state <= crate::gfx::cube_mesh::MAX_STATE
                && self.known_states.insert(state)
            {
                materials.push((
                    state,
                    crate::gfx::cube_mesh::material_record(context, state),
                ));
            }
        }
        let biomes = self
            .context
            .as_ref()
            .filter(|_| !self.biomes_sent)
            .map(|context| {
                self.biomes_sent = true;
                crate::gfx::cube_mesh::biome_table(context)
            });
        GpuBatch {
            nodes,
            materials,
            biomes,
        }
    }

    /// Returns a node's slot to the pool.
    fn release(&self, node: &GpuNode) {
        if let (GpuContent::Slot(slot), Some(backend)) = (node.content, &self.backend) {
            backend.pool.free(slot);
        }
    }

    /// Plans (again, when the camera crossed into another cube, the range,
    /// the loaded box or the view's detail changed, or a moment passed),
    /// queues what is missing, and returns the meshes finished since: only
    /// for nodes in the plan and the coverage layer.
    pub fn update(
        &mut self,
        camera: [f64; 3],
        max_distance: f64,
        view: &FarView,
        loaded: LoadedBox,
        is_loaded: impl Fn(SectionPos) -> bool,
    ) -> Vec<(CubeKey, SectionMesh)> {
        self.camera = camera;
        let now = Instant::now();
        let cube = cube_of(camera);
        self.face(view.forward);
        let rule = view.rule(self.budget());
        self.stats.target_px = view.target_px as f32;
        self.stats.fov_deg = view.fov_y.to_degrees() as f32;
        // A plan the planner finished goes in.
        let done = self.shared.planner.lock().unwrap().done.take();
        if let Some(done) = done {
            self.install(done, now);
        }
        if self.jumped.is_none()
            && self
                .asked
                .is_some_and(|a| distance(a.camera, camera) > JUMP)
        {
            self.jumped = Some(now);
        }
        // With the descent on, the first plan waits for its first answer, and
        // a plan after a jump for a moment: a plan of nothing but unclassified
        // (mixed) nodes would queue the whole octree.
        let hold = self.shared.descent_on.load(Ordering::Relaxed)
            && self.ask_descent(camera, max_distance, loaded, &rule, now);
        if self.shared.classes_dirty.swap(false, Ordering::Relaxed) {
            self.replan = true;
        }
        let stale = self.replan
            || match &self.asked {
                Some(a) => {
                    a.cube != cube
                        || a.max_distance != max_distance
                        || a.loaded != loaded
                        || now.saturating_duration_since(a.at) >= self.replan_every()
                        || refocused(&a.rule, &rule, view)
                }
                None => true,
            };
        if stale && !hold {
            let mut slot = self.shared.planner.lock().unwrap();
            if slot.job.is_none() && !slot.busy && slot.done.is_none() {
                slot.job = Some(PlanJob {
                    epoch: self.epoch,
                    camera,
                    max_distance,
                    rule,
                    loaded,
                    snapshot: LoadedSnapshot::take(cube, loaded, &is_loaded),
                    previous: self.plan.clone(),
                    surface: self.tiles.on.then(|| SurfaceJob {
                        info: self.tiles.info.clone(),
                        sea_level: self.tiles.sea_level,
                        budget: TILE_BUDGET,
                        previous: self.tiles.plan.clone(),
                    }),
                });
                drop(slot);
                self.shared.planner_wake.notify_one();
                self.replan = false;
                self.jumped = None;
                self.asked = Some(Asked {
                    cube,
                    camera,
                    max_distance,
                    loaded,
                    rule,
                    at: now,
                });
            }
        }
        let mut out = Vec::new();
        while let Ok(done) = self.results.try_recv() {
            let key = done.key;
            if !(self.wanted.contains(&key) || self.cover.contains(&key))
                || self.delivered.contains(&key)
            {
                if let Some(Product::Gpu(node)) = &done.product {
                    self.release(node);
                }
                continue;
            }
            match done.product {
                Some(Product::Mesh(mesh)) => {
                    self.deliver(key, mesh.solid.is_empty() && mesh.translucent.is_empty());
                    out.push((key, mesh));
                }
                Some(Product::Gpu(node)) => {
                    let empty = matches!(node.content,
                        GpuContent::Uniform(voxel) if voxel.state == self.air);
                    if matches!(node.content, GpuContent::Slot(_)) && node.states.is_empty() {
                        self.stats.gpu_generated += 1;
                    }
                    self.deliver(key, empty);
                    self.gpu_out.push(node);
                }
                None => {
                    self.failed.insert(key, now);
                }
            }
        }
        self.retry(now);
        self.recheck_holes(now, false);
        self.advance_phase(now);
        self.track_completion(now);
        out
    }

    /// How often the plan is redone with nothing else asking: [`REPLAN`], or
    /// four times what the last plan took, so a big plan keeps the planner
    /// thread busy a quarter of the time at most.
    fn replan_every(&self) -> Duration {
        REPLAN.max(Duration::from_secs_f32(self.stats.plan_ms.max(0.0) * 0.004))
    }

    /// Notes `key` as handed out (`empty`: it draws nothing).
    fn deliver(&mut self, key: CubeKey, empty: bool) {
        if !self.delivered.insert(key) {
            return;
        }
        self.release_dirty = true;
        self.holes_dirty = true;
        if empty {
            self.empty.insert(key);
        }
        if self.wanted.contains(&key) {
            self.missing = self.missing.saturating_sub(1);
            if empty {
                self.empty_planned += 1;
            }
        } else {
            self.stale.insert(key);
        }
    }

    /// Puts in a plan the planner made: the plan, its holes, and the queue.
    fn install(&mut self, done: PlanDone, now: Instant) {
        if done.epoch != self.epoch {
            return;
        }
        self.pruned = done.pruned;
        self.coarsened = done.coarsened;
        self.stats.limited = done.limited;
        self.stats.tiles_limited = done.tiles_limited;
        self.stats.voxel_px = self.stats.target_px * done.coarsened.max(1.0) as f32;
        self.stats.plan_ms = done.elapsed.as_secs_f32() * 1000.0;
        if !done.changed {
            return;
        }
        self.stats.plan_changed = Some(now);
        self.release_dirty = true;
        self.releasing.clear();
        let mut dropped = Vec::new();
        for &key in &done.removed {
            self.wanted.remove(&key);
            if self.delivered.contains(&key) {
                self.stale.insert(key);
                if self.empty.contains(&key) {
                    self.empty_planned = self.empty_planned.saturating_sub(1);
                }
            } else {
                self.missing = self.missing.saturating_sub(1);
                dropped.push(key);
            }
        }
        for &key in &done.added {
            self.wanted.insert(key);
            if self.delivered.contains(&key) {
                self.stale.remove(&key);
                if self.empty.contains(&key) {
                    self.empty_planned += 1;
                }
            } else {
                self.missing += 1;
            }
        }
        // Tiles: the new ones are asked for at once (the renderer makes them
        // on its own), dropped ones never drawn are let go.
        self.tiles.cover = done.cover_tiles;
        for &key in &done.tiles_removed {
            self.tiles.wanted.remove(&key);
            if self.tiles.delivered.contains(&key) {
                self.tiles.stale.insert(key);
            } else {
                self.tiles.missing = self.tiles.missing.saturating_sub(1);
                if !self.tiles.cover.contains(&key) && self.tiles.asked.remove(&key) {
                    self.tiles.work.free.push(key);
                }
            }
        }
        for &key in &done.tiles_added {
            self.tiles.wanted.insert(key);
            if self.tiles.delivered.contains(&key) {
                self.tiles.stale.remove(&key);
            } else {
                self.tiles.missing += 1;
                if self.tiles.asked.insert(key) {
                    self.tiles.work.request.push(key);
                }
            }
        }
        let planned = done.leaves.len() + done.tiles.len();
        let big = (done.added.len() + done.tiles_added.len()) * COVER_HOLES > planned;
        self.plan = done.leaves;
        self.tiles.plan = done.tiles;
        self.order = done.order;
        self.tree = done.tree;
        self.footprints = done.footprints;
        self.cover_plan = done.cover;
        // The holes: all of them after a big change, else the old ones still
        // planned and the new nodes with nothing over them.
        let fresh: Vec<CubeKey> = if big {
            self.plan.to_vec()
        } else {
            done.added
                .iter()
                .copied()
                .filter(|k| !self.delivered.contains(k))
                .collect()
        };
        let covering = matches!(self.phase, Phase::Cover { .. });
        let (new_holes, holding) = self.find_holes(&fresh, big || covering);
        let fresh_tiles: Vec<TileKey> = if big {
            self.tiles.plan.to_vec()
        } else {
            done.tiles_added
                .iter()
                .copied()
                .filter(|k| !self.tiles.delivered.contains(k))
                .collect()
        };
        let new_tile_holes = self.find_tile_holes(&fresh_tiles, big || covering);
        if big {
            self.holes = new_holes;
            self.holding = holding;
            self.tiles.holes = new_tile_holes;
        } else {
            let wanted = &self.wanted;
            self.holes.retain(|(k, _)| wanted.contains(k));
            self.holes.extend(new_holes);
            self.holding.retain(|k| wanted.contains(k));
            self.holding.extend(holding);
            let wanted = &self.tiles.wanted;
            self.tiles.holes.retain(|k| wanted.contains(k));
            self.tiles.holes.extend(new_tile_holes);
        }
        self.holes_dirty = false;
        self.holes_at = now;
        let holes = self.holes.len() + self.tiles.holes.len();
        if holes > 0 {
            self.stats.plan_covered = None;
        }
        if (big && holes * COVER_HOLES > planned) || covering {
            self.begin_cover(now);
        } else if big {
            self.fill_queue();
        } else {
            // A small change: the new nodes go first, the dropped ones are
            // skipped if they are queued.
            let rank = self.rank();
            let mut first: Vec<CubeKey> = done
                .added
                .iter()
                .copied()
                .filter(|k| !self.delivered.contains(k))
                .collect();
            first.sort_by(|a, b| self.priority(&rank, a).total_cmp(&self.priority(&rank, b)));
            let mut queue = self.shared.queue.lock().unwrap();
            for key in &first {
                queue.dropped.remove(key);
            }
            queue.dropped.extend(dropped);
            queue.first.extend(first);
            drop(queue);
            self.shared.wake.notify_all();
        }
    }

    /// The rank of each sector in the queue's order now.
    fn rank(&self) -> [u8; SECTORS] {
        let order = sector_order(self.facing.unwrap_or([0.0, 0.0, 1.0]));
        let mut rank = [0u8; SECTORS];
        for (i, s) in order.into_iter().enumerate() {
            rank[usize::from(s)] = i as u8;
        }
        rank
    }

    /// The order a short list is queued in: by sector rank, then distance.
    fn priority(&self, rank: &[u8; SECTORS], key: &CubeKey) -> f64 {
        f64::from(rank[usize::from(sector(self.camera, *key))]) * 1e9
            + key.distance(self.camera).min(9e8)
    }

    /// The planned nodes among `keys` with nothing drawn over their place:
    /// not delivered, under no delivered node (a coverage node or an older
    /// plan's, kept by `removed`), and over none (an older plan's finer
    /// nodes, kept until this one is in). With `cover`, each comes with the
    /// coverage leaf over it, if any, and only a node as fine as that leaf or
    /// finer counts as drawn over it. Also returns the planned nodes holding
    /// an older plan's nodes: once in, those are let go (after a teleport,
    /// most of the last plan, far away under a few coarse nodes).
    fn find_holes(&self, keys: &[CubeKey], cover: bool) -> (Vec<Hole>, Vec<CubeKey>) {
        // Only nodes outside the plan can hold or be held by a planned node.
        let mut levels = 0u32;
        let mut inner = FxHashSet::default();
        for &s in &self.stale {
            levels |= 1 << s.level;
            let mut up = s;
            while up.level < MAX_CUBE_LEVEL && inner.insert(up.parent()) {
                up = up.parent();
            }
        }
        let mut holes = Vec::new();
        let mut holding = Vec::new();
        if !cover {
            for &key in keys {
                if self.delivered.contains(&key) {
                    continue;
                }
                if inner.contains(&key) {
                    holding.push(key);
                    continue;
                }
                let held = (key.level + 1..=MAX_CUBE_LEVEL)
                    .filter(|l| levels >> l & 1 == 1)
                    .any(|l| self.stale.contains(&ancestor(key, l)));
                if !held && !self.tile_drawn_over(key, MAX_CUBE_LEVEL) {
                    holes.push((key, None));
                }
            }
            return (holes, holding);
        }
        // Per node above a hole: whether it or a node over it is drawn, and
        // the coverage leaf at or over it. Siblings share the walk.
        let mut memo: FxHashMap<CubeKey, (bool, Option<CubeKey>)> = FxHashMap::default();
        let mut path = Vec::new();
        for &key in keys {
            if self.delivered.contains(&key) {
                continue;
            }
            if inner.contains(&key) {
                holding.push(key);
                continue;
            }
            path.clear();
            let mut found = (false, None);
            let mut up = key;
            while up.level < MAX_CUBE_LEVEL {
                up = up.parent();
                if let Some(&m) = memo.get(&up) {
                    found = m;
                    break;
                }
                path.push(up);
            }
            // A node drawn over the place counts only at the coverage leaf's
            // size or finer: after a teleport, the last plan's nodes millions
            // of blocks wide still stand over the new place, a voxel there
            // tens of thousands of blocks tall.
            for &n in path.iter().rev() {
                found = if self.cover_plan.contains(&n) {
                    (self.stale.contains(&n), Some(n))
                } else if found.1.is_some() {
                    (found.0 || self.stale.contains(&n), found.1)
                } else {
                    (false, None)
                };
                memo.insert(n, found);
            }
            if found.0 {
                continue;
            }
            // Where the coverage layer has a tile over the node, the tile.
            let over = match found.1 {
                Some(c) => Some(FarKey::Cube(c)),
                None => self.cover_tile_over(key).map(FarKey::Tile),
            };
            let limit = match over {
                Some(FarKey::Cube(c)) => c.level,
                Some(FarKey::Tile(t)) => t.level,
                None => key.level,
            };
            if !self.tile_drawn_over(key, limit) {
                holes.push((key, over));
            }
        }
        (holes, holding)
    }

    /// Whether a drawn tile of level `limit` or finer covers `key`'s footprint.
    fn tile_drawn_over(&self, key: CubeKey, limit: u8) -> bool {
        if self.tiles.delivered.is_empty() {
            return false;
        }
        (key.level.max(SURFACE_LEVEL)..=limit.min(MAX_CUBE_LEVEL)).any(|l| {
            let up = l - key.level;
            self.tiles
                .delivered
                .contains(&TileKey::new(l, key.x >> up, key.z >> up))
        })
    }

    /// The coverage layer's tile over `key`'s footprint, if it has one.
    fn cover_tile_over(&self, key: CubeKey) -> Option<TileKey> {
        if self.tiles.cover.is_empty() {
            return None;
        }
        (key.level.max(SURFACE_LEVEL)..=MAX_CUBE_LEVEL)
            .map(|l| {
                let up = l - key.level;
                TileKey::new(l, key.x >> up, key.z >> up)
            })
            .find(|t| self.tiles.cover.contains(t))
    }

    /// The planned tiles among `keys` with nothing drawn over their
    /// footprint: not drawn, under no drawn tile (with `cover`, none more than
    /// [`COVER_LEVELS`] coarser counts) and over no drawn node or tile of an
    /// older plan.
    fn find_tile_holes(&self, keys: &[TileKey], cover: bool) -> Vec<TileKey> {
        if keys.is_empty() {
            return Vec::new();
        }
        // The footprints holding an older plan's tiles or nodes.
        let mut inner = FxHashSet::<TileKey>::default();
        let mut add = |level: u8, x: i32, z: i32, own: bool| {
            if own {
                inner.insert(TileKey::new(level, x, z));
            }
            let mut l = level;
            while l < MAX_CUBE_LEVEL {
                l += 1;
                let up = l - level;
                if !inner.insert(TileKey::new(l, x >> up, z >> up)) {
                    break;
                }
            }
        };
        for &t in &self.tiles.stale {
            add(t.level, t.x, t.z, false);
        }
        for &c in &self.stale {
            add(c.level, c.x, c.z, true);
        }
        keys.iter()
            .copied()
            .filter(|t| {
                if self.tiles.delivered.contains(t) || inner.contains(t) {
                    return false;
                }
                let limit = if cover {
                    t.level.saturating_add(COVER_LEVELS).min(MAX_CUBE_LEVEL)
                } else {
                    MAX_CUBE_LEVEL
                };
                !(t.level + 1..=limit).any(|l| {
                    let up = l - t.level;
                    self.tiles
                        .delivered
                        .contains(&TileKey::new(l, t.x >> up, t.z >> up))
                })
            })
            .collect()
    }

    /// Drops the holes that something now covers: the node itself, or the
    /// coverage node over it. A pass over every hole runs at most every
    /// [`SCAN_EVERY`] while there are many.
    fn recheck_holes(&mut self, now: Instant, force: bool) {
        if !self.holes_dirty || (self.holes.is_empty() && self.tiles.holes.is_empty()) {
            return;
        }
        if !force
            && self.holes.len() > SLOW_SCAN
            && now.saturating_duration_since(self.holes_at) < SCAN_EVERY
        {
            return;
        }
        self.holes_dirty = false;
        self.holes_at = now;
        let (delivered, tiles) = (&self.delivered, &self.tiles.delivered);
        self.holes.retain(|(key, over)| {
            !(delivered.contains(key)
                || match over {
                    Some(FarKey::Cube(c)) => delivered.contains(c),
                    Some(FarKey::Tile(t)) => tiles.contains(t),
                    None => false,
                })
        });
        self.tiles.holes.retain(|t| !tiles.contains(t));
    }

    /// Queues only the coverage layer over the holes: a coverage node over a
    /// hole where the layer has one, else the hole itself (the plan is as
    /// coarse there).
    fn begin_cover(&mut self, now: Instant) {
        let since = match self.phase {
            Phase::Cover { since, .. } => since,
            Phase::Refine => now,
        };
        let mut seen = FxHashSet::default();
        let wave: Vec<FarKey> = self
            .holes
            .iter()
            .map(|&(key, over)| over.unwrap_or(FarKey::Cube(key)))
            .chain(self.holding.iter().map(|&k| FarKey::Cube(k)))
            .chain(self.tiles.holes.iter().map(|&t| FarKey::Tile(t)))
            .filter(|k| seen.insert(*k) && !self.far_delivered(*k))
            .collect();
        if wave.is_empty() {
            self.phase = Phase::Refine;
            self.fill_queue();
            return;
        }
        // The coverage tiles go to the renderer first: they are the fastest.
        let tiles: Vec<TileKey> = wave
            .iter()
            .filter_map(|k| match *k {
                FarKey::Tile(t) => Some(t),
                FarKey::Cube(_) => None,
            })
            .filter(|t| self.tiles.asked.insert(*t))
            .collect();
        self.tiles.work.request.splice(0..0, tiles);
        let rank = self.rank();
        let mut cubes: Vec<CubeKey> = wave
            .iter()
            .filter_map(|k| match *k {
                FarKey::Cube(c) => Some(c),
                FarKey::Tile(_) => None,
            })
            .collect();
        cubes.sort_by(|a, b| self.priority(&rank, a).total_cmp(&self.priority(&rank, b)));
        self.cover = cubes
            .iter()
            .copied()
            .filter(|k| !self.wanted.contains(k))
            .collect();
        self.stats.cover_nodes = wave.len();
        let mut queue = self.shared.queue.lock().unwrap();
        queue.clear();
        queue.first.extend(cubes);
        drop(queue);
        self.shared.wake.notify_all();
        self.phase = Phase::Cover { since, wave };
    }

    /// Whether a node or tile is drawn.
    fn far_delivered(&self, key: FarKey) -> bool {
        match key {
            FarKey::Cube(c) => self.delivered.contains(&c),
            FarKey::Tile(t) => self.tiles.delivered.contains(&t),
        }
    }

    /// Ends the coverage phase once its nodes are in and shown (or failed,
    /// or after [`COVER_WAIT`]), and queues the rest of the plan.
    fn advance_phase(&mut self, now: Instant) {
        let Phase::Cover { since, wave } = &self.phase else {
            return;
        };
        let landed = wave.iter().all(|k| match *k {
            FarKey::Cube(c) => {
                self.delivered.contains(&c)
                    || self.failed.contains_key(&c)
                    || !(self.wanted.contains(&c) || self.cover.contains(&c))
            }
            FarKey::Tile(t) => {
                self.tiles.delivered.contains(&t)
                    || !(self.tiles.asked.contains(&t) || self.tiles.wanted.contains(&t))
            }
        });
        if !(landed && self.shown()) && now.saturating_duration_since(*since) < COVER_WAIT {
            return;
        }
        self.phase = Phase::Refine;
        self.holes_dirty = true;
        self.recheck_holes(now, true);
        self.fill_queue();
    }

    /// Queues every planned node not handed out: holes first, then by sector
    /// (the view's first) and distance.
    fn fill_queue(&mut self) {
        let rank = self.rank();
        let now = Instant::now();
        let waiting = |k: &CubeKey| {
            !self.delivered.contains(k)
                && self
                    .failed
                    .get(k)
                    .is_none_or(|at| now.saturating_duration_since(*at) >= RETRY)
        };
        let mut firsts = FxHashSet::default();
        let mut first: Vec<CubeKey> = self
            .holes
            .iter()
            .map(|h| h.0)
            .chain(self.holding.iter().copied())
            .filter(|k| waiting(k) && firsts.insert(*k))
            .collect();
        first.sort_by(|a, b| self.priority(&rank, a).total_cmp(&self.priority(&rank, b)));
        let mut queue = self.shared.queue.lock().unwrap();
        queue.clear();
        queue.first.extend(first);
        for &(s, key) in &self.order {
            if waiting(&key) && !firsts.contains(&key) {
                queue.sectors[usize::from(s)].push_back(key);
            }
        }
        queue.order = sector_order(self.facing.unwrap_or([0.0, 0.0, 1.0]));
        drop(queue);
        self.shared.wake.notify_all();
    }

    /// Asks again for nodes whose sampling failed a while ago.
    fn retry(&mut self, now: Instant) {
        if now < self.retry_at || self.failed.is_empty() {
            return;
        }
        self.retry_at = now + RETRY / 2;
        let (wanted, cover) = (&self.wanted, &self.cover);
        self.failed
            .retain(|k, _| wanted.contains(k) || cover.contains(k));
        let due: Vec<CubeKey> = self
            .failed
            .iter()
            .filter(|(_, at)| now.saturating_duration_since(**at) >= RETRY)
            .map(|(&k, _)| k)
            .collect();
        if due.is_empty() {
            return;
        }
        for key in &due {
            self.failed.remove(key);
        }
        let mut queue = self.shared.queue.lock().unwrap();
        queue.first.extend(due);
        drop(queue);
        self.shared.wake.notify_all();
    }

    /// Reorders the queue's sectors when the view turned past [`TURN`].
    fn face(&mut self, forward: [f64; 3]) {
        if self
            .facing
            .is_some_and(|f| angle_between(f, forward) <= TURN)
        {
            return;
        }
        self.facing = Some(forward);
        let order = sector_order(forward);
        self.shared.queue.lock().unwrap().order = order;
    }

    /// The nodes the renderer can drop: handed out before, no longer in the
    /// plan, and with every node over their place handed out too.
    ///
    /// At most [`RELEASE_BATCH`] a call: the renderer retires each with
    /// writes of its own, and a teleport lets a whole plan go at once.
    pub fn removed(&mut self) -> Vec<CubeKey> {
        if self.releasing.is_empty() {
            self.releasing = self.releasable();
        }
        let take = self.releasing.len().min(RELEASE_BATCH);
        let mut released: Vec<CubeKey> = self.releasing.drain(..take).collect();
        // Planned again since it was found, or a plan put in since: kept.
        released.retain(|k| self.stale.contains(k) && !self.wanted.contains(k));
        for key in &released {
            self.delivered.remove(key);
            self.stale.remove(key);
            self.empty.remove(key);
        }
        let camera = self.camera;
        released.sort_by_cached_key(|k| (k.distance(camera) * 16.0) as u64);
        released
    }

    /// The stale nodes nothing needs any more: every node over their place
    /// is handed out, or none is wanted. A pass over the stale nodes, at most
    /// every [`SCAN_EVERY`] while many planned nodes are missing.
    fn releasable(&mut self) -> Vec<CubeKey> {
        let now = Instant::now();
        if (self.stale.is_empty() && self.tiles.stale.is_empty()) || !self.release_dirty {
            return Vec::new();
        }
        // A pass costs about the stale nodes and the missing ones.
        if self.missing + self.stale.len() > SLOW_SCAN
            && now.saturating_duration_since(self.released_at) < SCAN_EVERY
        {
            return Vec::new();
        }
        self.release_dirty = false;
        self.released_at = now;
        // A node holding a planned node still missing covers a place nothing
        // else draws yet: the planned nodes inside it are a run of the tree.
        let blocked = |key: CubeKey| {
            if self.missing == 0 {
                return false;
            }
            let lo = tree_order(key);
            let hi = lo + (1u128 << (3 * u32::from(key.level)));
            let start = self.tree.partition_point(|e| e.0 < lo);
            self.tree[start..]
                .iter()
                .take_while(|e| e.0 < hi)
                .any(|e| e.1.level < key.level && !self.delivered.contains(&e.1))
        };
        // Per node over a stale one: whether it or one over it is planned
        // and not out yet (it then covers the stale node's place).
        let mut memo: FxHashMap<CubeKey, bool> = FxHashMap::default();
        let mut path = Vec::new();
        let mut released = Vec::new();
        for &key in &self.stale {
            // Under a planned tile not in yet, or (a coarse node) over one.
            if blocked(key)
                || self.tile_pending_over(key)
                || (key.level > SURFACE_LEVEL
                    && self.pending_in(TileKey::new(key.level, key.x, key.z), true))
            {
                continue;
            }
            path.clear();
            let mut covered = false;
            let mut up = key;
            while up.level < MAX_CUBE_LEVEL {
                up = up.parent();
                if let Some(&m) = memo.get(&up) {
                    covered = m;
                    break;
                }
                path.push(up);
            }
            for &n in path.iter().rev() {
                covered = covered || (self.wanted.contains(&n) && !self.delivered.contains(&n));
                memo.insert(n, covered);
            }
            if !covered {
                released.push(key);
            }
        }
        // Tiles: faded out once nothing planned in or over their footprint
        // waits (few, and the renderer's fade is one write each).
        let retire: Vec<TileKey> = self
            .tiles
            .stale
            .iter()
            .copied()
            .filter(|&t| !(self.pending_in(t, false) || self.pending_over(t)))
            .collect();
        for t in &retire {
            self.tiles.delivered.remove(t);
            self.tiles.stale.remove(t);
        }
        self.tiles.work.retire.extend(retire);
        released
    }

    /// Whether a planned tile not drawn yet covers `key`'s footprint.
    fn tile_pending_over(&self, key: CubeKey) -> bool {
        self.tiles.missing > 0
            && (key.level.max(SURFACE_LEVEL)..=MAX_CUBE_LEVEL).any(|l| {
                let up = l - key.level;
                let t = TileKey::new(l, key.x >> up, key.z >> up);
                self.tiles.wanted.contains(&t) && !self.tiles.delivered.contains(&t)
            })
    }

    /// Whether something planned and not drawn yet lies in the footprint
    /// `fp` (a tile's, or a node's at its level): a tile, or (unless
    /// `tiles_only`) a node.
    fn pending_in(&self, fp: TileKey, tiles_only: bool) -> bool {
        if self.missing + self.tiles.missing == 0 {
            return false;
        }
        let lo = footprint_order(fp.level, fp.x, fp.z);
        let hi = lo + (1u64 << (2 * u32::from(fp.level)));
        let start = self.footprints.partition_point(|e| e.0 < lo);
        self.footprints[start..]
            .iter()
            .take_while(|e| e.0 < hi)
            .any(|e| {
                e.1 <= fp.level
                    && match e.2 {
                        FarKey::Cube(c) => !tiles_only && !self.delivered.contains(&c),
                        FarKey::Tile(t) => !self.tiles.delivered.contains(&t),
                    }
            })
    }

    /// Whether a planned tile or node not drawn yet has a footprint holding
    /// the footprint `fp` (a coarser one).
    fn pending_over(&self, fp: TileKey) -> bool {
        if self.missing + self.tiles.missing == 0 {
            return false;
        }
        (fp.level + 1..=MAX_CUBE_LEVEL).any(|l| {
            let up = l - fp.level;
            let (x, z) = (fp.x >> up, fp.z >> up);
            let t = TileKey::new(l, x, z);
            if self.tiles.wanted.contains(&t) && !self.tiles.delivered.contains(&t) {
                return true;
            }
            let code = footprint_order(l, x, z);
            let start = self.footprints.partition_point(|e| (e.0, e.1) < (code, l));
            self.footprints[start..]
                .iter()
                .take_while(|e| e.0 == code && e.1 == l)
                .any(|e| matches!(e.2, FarKey::Cube(c) if !self.delivered.contains(&c)))
        })
    }

    /// Forgets everything: nodes handed out, queued and cached.
    pub fn clear(&mut self) {
        self.shared.queue.lock().unwrap().clear();
        self.shared.cache.lock().unwrap().nodes.clear();
        {
            // A plan in the works belongs to the old count: dropped.
            let mut slot = self.shared.planner.lock().unwrap();
            slot.job = None;
            slot.done = None;
        }
        self.epoch += 1;
        // Results of jobs running now are dropped as unwanted.
        self.plan = Arc::default();
        self.wanted.clear();
        self.order.clear();
        self.tree.clear();
        self.footprints.clear();
        self.tiles = TileState {
            on: self.tiles.on,
            sea_level: self.tiles.sea_level,
            ..TileState::default()
        };
        self.cover_plan.clear();
        self.cover.clear();
        self.holes.clear();
        self.holding.clear();
        self.phase = Phase::Refine;
        self.delivered.clear();
        self.stale.clear();
        self.releasing.clear();
        self.missing = 0;
        self.failed.clear();
        self.empty.clear();
        self.empty_planned = 0;
        self.stats.plan_changed = None;
        self.stats.plan_complete = None;
        self.stats.plan_covered = None;
        self.asked = None;
        self.replan = false;
        self.jumped = None;
        while let Ok(done) = self.results.try_recv() {
            if let Some(Product::Gpu(node)) = &done.product {
                self.release(node);
            }
        }
        for node in std::mem::take(&mut self.gpu_out) {
            self.release(&node);
        }
    }
}

/// The node of `level` holding `key` (`level` at least `key.level`).
fn ancestor(key: CubeKey, level: u8) -> CubeKey {
    let up = level - key.level;
    CubeKey::new(level, key.x >> up, key.y >> up, key.z >> up)
}

fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

impl Drop for CubeLod {
    fn drop(&mut self) {
        {
            // Under the lock: a worker between its stop check and its wait
            // must not miss the wake.
            let _queue = self.shared.queue.lock();
            self.shared.stop.store(true, Ordering::Relaxed);
        }
        self.shared.wake.notify_all();
        {
            let _classes = self.shared.classes.lock();
            self.shared.classes_wake.notify_all();
        }
        {
            let _descent = self.shared.descent.lock();
            self.shared.descent_wake.notify_all();
        }
        {
            let _planner = self.shared.planner.lock();
            self.shared.planner_wake.notify_all();
        }
        for thread in self.threads.drain(..) {
            let _ = thread.join();
        }
        self.clear();
    }
}

/// What is known of `key`; unknown nodes are queued for the classifier thread
/// and count as mixed until it has looked.
fn classify_cached(shared: &Shared, key: CubeKey) -> NodeClass {
    if key.level < 2 {
        return NodeClass::Mixed;
    }
    let mut classes = shared.classes.lock().unwrap();
    if let Some(class) = classes.known.get(&key) {
        return class;
    }
    if shared.descent_on.load(Ordering::Relaxed) {
        // The descent fills the classes; nothing is asked of the classifier.
        return NodeClass::Mixed;
    }
    if classes.queued.insert(key) {
        classes.queue.push_back(key);
        drop(classes);
        shared.classes_wake.notify_one();
    }
    NodeClass::Mixed
}

/// Runs the GPU descent for each request (the latest wins) and files the
/// nodes it proved uniform as classes.
fn descent_loop(shared: &Shared, descent: &GpuDescent) {
    loop {
        let request = {
            let mut slot = shared.descent.lock().unwrap();
            loop {
                if shared.stop.load(Ordering::Relaxed) {
                    return;
                }
                if let Some(request) = slot.request.take() {
                    slot.busy = true;
                    break request;
                }
                slot = shared.descent_wake.wait(slot).unwrap();
            }
        };
        let found = descent.descend(&request);
        if let Some(found) = &found {
            let mut classes = shared.classes.lock().unwrap();
            let uniform: Vec<_> = found.uniform().collect();
            classes.known.extend(uniform.into_iter());
        }
        let mut slot = shared.descent.lock().unwrap();
        slot.busy = false;
        slot.runs += 1;
        if let Some(found) = found {
            slot.uniform = found.uniform_count();
            slot.leaves = found.leaves.len();
            slot.millis = found.elapsed.as_secs_f32() * 1000.0;
            slot.overflow = found.overflow;
        }
        drop(slot);
        shared.classes_dirty.store(true, Ordering::Relaxed);
    }
}

fn classify_loop(shared: &Shared, sampler: &mut dyn CubeSampler) {
    loop {
        let key = {
            let mut classes = shared.classes.lock().unwrap();
            loop {
                if shared.stop.load(Ordering::Relaxed) {
                    return;
                }
                if let Some(key) = classes.queue.pop_front() {
                    break key;
                }
                classes = shared.classes_wake.wait(classes).unwrap();
            }
        };
        let class = sampler.classify(key);
        let mut classes = shared.classes.lock().unwrap();
        classes.queued.remove(&key);
        classes.known.insert(key, class);
    }
}

/// The kind of a state, by name, cached per thread.
fn classifier(context: &WorkerContext, air: BlockStateId) -> impl Fn(BlockStateId) -> Kind + '_ {
    let cache = RefCell::new(FxHashMap::<u32, Kind>::default());
    move |state| {
        *cache.borrow_mut().entry(state.0).or_insert_with(|| {
            if state == air {
                return Kind::Air;
            }
            match context
                .assets
                .registry
                .state(state)
                .map_or("", |s| s.name.as_str())
            {
                "minecraft:air" | "minecraft:cave_air" | "minecraft:void_air" => Kind::Air,
                "minecraft:water" | "minecraft:lava" | "minecraft:bubble_column" => Kind::Fluid,
                _ => Kind::Solid,
            }
        })
    }
}

fn work(
    shared: Arc<Shared>,
    tx: mpsc::Sender<Done>,
    context: Arc<WorkerContext>,
    mut sampler: Box<dyn CubeSampler>,
    air: BlockStateId,
    backend: Option<GpuBackend>,
) {
    horizon_threads::behind_server();
    horizon_threads::far_terrain_current();
    let mut mesher = Mesher::default();
    let kind = classifier(&context, air);
    loop {
        let gpu = backend
            .as_ref()
            .filter(|_| shared.gpu_path.load(Ordering::Relaxed));
        // On the GPU path a worker takes a batch of the first jobs and asks
        // the sampler for them at once (`resident_many`); on the CPU path one.
        let take = if gpu.is_some() { GPU_BATCH } else { 1 };
        if let Some(backend) = gpu {
            // The pool is full while the renderer meshes what is in it: wait
            // for room rather than fail the batch (a failed node waits RETRY).
            let mut waited = 0;
            while backend.pool.free_slots() < GPU_BATCH
                && waited < POOL_WAITS
                && !shared.stop.load(Ordering::Relaxed)
            {
                thread::sleep(POOL_WAIT);
                waited += 1;
            }
        }
        let keys: Vec<CubeKey> = {
            let mut queue = shared.queue.lock().unwrap();
            loop {
                if shared.stop.load(Ordering::Relaxed) {
                    return;
                }
                let mut keys = Vec::new();
                // The finest levels are sampled on the CPU one by one, before
                // the batch's GPU request: a few per batch, so they never hold
                // the GPU's nodes back for long. The rest wait at the front,
                // and the GPU's nodes behind them come in meanwhile.
                let (mut cpu, mut later) = (0, Vec::new());
                while keys.len() < take
                    && later.len() < 8 * take
                    && let Some(key) = queue.pop()
                {
                    if gpu.is_some() && key.level < MIN_LEVEL {
                        if cpu == CPU_PER_BATCH {
                            later.push(key);
                            continue;
                        }
                        cpu += 1;
                    }
                    queue.running.insert(key);
                    keys.push(key);
                }
                for key in later.into_iter().rev() {
                    queue.first.push_front(key);
                }
                if !keys.is_empty() {
                    break keys;
                }
                queue = shared.wake.wait(queue).unwrap();
            }
        };
        match sampler.changes() {
            Changes::None => {}
            Changes::Cubes(cubes) => shared.cache.lock().unwrap().forget(&cubes),
            Changes::All => shared.cache.lock().unwrap().nodes.clear(),
        }
        let mut fetched: FxHashMap<CubeKey, Option<Resident>> = FxHashMap::default();
        if gpu.is_some() && !shared.stop.load(Ordering::Relaxed) {
            let uncached: Vec<CubeKey> = {
                let mut cache = shared.cache.lock().unwrap();
                keys.iter()
                    .copied()
                    .filter(|&k| cache.get(k).is_none())
                    .collect()
            };
            if !uncached.is_empty() {
                let answers = sampler.resident_many(&uncached);
                fetched.extend(uncached.into_iter().zip(answers));
            }
        }
        for key in keys {
            let product = match gpu {
                Some(backend) => gpu_node(
                    &shared,
                    sampler.as_mut(),
                    key,
                    air,
                    &kind,
                    backend,
                    fetched.remove(&key),
                )
                .map(Product::Gpu),
                None => node(&shared, sampler.as_mut(), key, air, &kind).map(|node| {
                    let mut mesh = SectionMesh::default();
                    let sea = sampler.sea_surface();
                    mesh::mesh_cube(&mut mesher, &node, air, sea, &context, &mut mesh);
                    mesh.sort_solid();
                    Product::Mesh(mesh)
                }),
            };
            shared.queue.lock().unwrap().running.remove(&key);
            if tx.send(Done { key, product }).is_err() {
                return;
            }
        }
    }
}

/// Jobs a worker on the GPU path takes and asks for at once.
const GPU_BATCH: usize = 64;
/// At most this many of them are of the levels sampled on the CPU.
const CPU_PER_BATCH: usize = 2;
/// A worker finding the GPU pool without room for a batch waits this long, at
/// most this many times, before asking anyway.
const POOL_WAIT: Duration = Duration::from_millis(2);
const POOL_WAITS: u32 = 250;

/// A CPU node as a [`GpuNode`]: uniform and uncovered ones need no slot, the
/// rest are uploaded into one. `None` when the pool is full.
fn gpu_from_node(node: &CubeNode, air: BlockStateId, backend: &GpuBackend) -> Option<GpuNode> {
    let cover = node.covered_mask().map(|mask| Box::new(*mask));
    let mut states = Vec::new();
    if node.as_uniform().is_none() {
        let mut seen = FxHashSet::<u32>::default();
        for i in 0..VOLUME {
            if seen.insert(node.at(i).state.0) {
                states.push(node.at(i).state.0);
            }
        }
    }
    let content = match node.as_uniform() {
        Some(voxel) if cover.is_none() => GpuContent::Uniform(voxel),
        _ => {
            let slot = backend.pool.alloc()?;
            backend
                .pool
                .write(&backend.queue, slot, &slot_words(node, air.0));
            GpuContent::Slot(slot)
        }
    };
    Some(GpuNode {
        key: node.key,
        content,
        cover,
        states,
    })
}

/// The node at `key` for the GPU path: a slot the GPU generated it into, or a
/// CPU node (saved cubes, downsampled fine levels, the pre-pass's uniform
/// ones) uploaded to one.
/// `fetched` is the sampler's answer when the worker already asked for it in
/// a batch (`Some`, which may be `None`); otherwise it is asked here.
fn gpu_node(
    shared: &Shared,
    sampler: &mut dyn CubeSampler,
    key: CubeKey,
    air: BlockStateId,
    kind: &impl Fn(BlockStateId) -> Kind,
    backend: &GpuBackend,
    fetched: Option<Option<Resident>>,
) -> Option<GpuNode> {
    let cached = shared.cache.lock().unwrap().get(key);
    if let Some(node) = cached {
        if let Some(Some(Resident::Slot(slot))) = fetched {
            backend.pool.free(slot);
        }
        return gpu_from_node(&node, air, backend);
    }
    if shared.stop.load(Ordering::Relaxed) {
        if let Some(Some(Resident::Slot(slot))) = fetched {
            backend.pool.free(slot);
        }
        return None;
    }
    let answer = match fetched {
        Some(answer) => answer,
        None => sampler.resident(key),
    };
    match answer {
        Some(Resident::Slot(slot)) => Some(GpuNode {
            key,
            content: GpuContent::Slot(slot),
            cover: None,
            states: Vec::new(),
        }),
        Some(Resident::Node(node)) => {
            debug_assert_eq!(node.key, key);
            let node = Arc::new(node);
            shared.cache.lock().unwrap().insert(node.clone());
            gpu_from_node(&node, air, backend)
        }
        None if (1..=3).contains(&key.level) => {
            let node = node(shared, sampler, key, air, kind)?;
            gpu_from_node(&node, air, backend)
        }
        None => None,
    }
}

/// The node at `key`: cached, sampled, or (fine levels the sampler cannot
/// answer) downsampled from its children.
fn node(
    shared: &Shared,
    sampler: &mut dyn CubeSampler,
    key: CubeKey,
    air: BlockStateId,
    kind: &impl Fn(BlockStateId) -> Kind,
) -> Option<Arc<CubeNode>> {
    if let Some(node) = shared.cache.lock().unwrap().get(key) {
        return Some(node);
    }
    if shared.stop.load(Ordering::Relaxed) {
        return None;
    }
    let node = match sampler.node(key) {
        Some(node) => node,
        None if (1..=3).contains(&key.level) => {
            let children = key.children();
            let mut nodes = Vec::with_capacity(8);
            for child in children {
                nodes.push(node(shared, sampler, child, air, kind)?);
            }
            let refs: [&CubeNode; 8] = std::array::from_fn(|i| &*nodes[i]);
            downsample(refs, air, kind)
        }
        None => return None,
    };
    debug_assert_eq!(node.key, key);
    let node = Arc::new(node);
    shared.cache.lock().unwrap().insert(node.clone());
    Some(node)
}

#[cfg(test)]
mod tests {
    #[test]
    fn known_classes_keep_the_last_answers_whole() {
        use super::{CubeKey, KnownClasses, NodeClass};
        let key = |i: usize| CubeKey::new(2, i as i32, 0, 0);
        let answer = |from: usize| (from..from + 150_000).map(move |i| (key(i), NodeClass::Air));
        let mut known = KnownClasses::default();
        // Two answers of 150,000 (an Ultra descent's): both kept.
        known.extend(answer(0));
        known.extend(answer(150_000));
        assert_eq!(known.len(), 300_000);
        assert!(known.get(&key(0)).is_some() && known.get(&key(299_999)).is_some());
        // A third: the first goes, the last two stay whole.
        known.extend(answer(300_000));
        assert!(known.get(&key(0)).is_none());
        assert!(known.get(&key(150_000)).is_some() && known.get(&key(449_999)).is_some());
    }

    #[test]
    fn the_arena_cap_holds_through_small_moves() {
        // Within the band the held cap stays; past it the cap follows.
        assert!(super::arena_cap_holds(100_000, 100_000));
        assert!(super::arena_cap_holds(100_000, 91_000));
        assert!(super::arena_cap_holds(100_000, 124_000));
        assert!(!super::arena_cap_holds(100_000, 89_000));
        assert!(!super::arena_cap_holds(100_000, 126_000));
    }

    use super::*;

    const AIR: BlockStateId = BlockStateId(0);
    const STONE: BlockStateId = BlockStateId(1);
    const GRASS: BlockStateId = BlockStateId(2);
    const WATER: BlockStateId = BlockStateId(3);
    const DIRT: BlockStateId = BlockStateId(4);

    fn kind(state: BlockStateId) -> Kind {
        match state {
            AIR => Kind::Air,
            WATER => Kind::Fluid,
            _ => Kind::Solid,
        }
    }
    fn voxel(state: BlockStateId) -> Voxel {
        Voxel { state, biome: 0 }
    }
    fn key(level: u8, x: i32, y: i32, z: i32) -> CubeKey {
        CubeKey::new(level, x, y, z)
    }

    #[test]
    fn keys_nest_at_negative_coordinates_and_the_world_edges() {
        for k in [
            key(0, -1, -1, -1),
            key(0, -17, 3, -2),
            key(3, -1, 0, 5),
            key(5, -3, -4, 2),
            key(0, 524_287, -524_288, -1_875_000),
        ] {
            let parent = k.parent();
            assert_eq!(parent.level, k.level + 1);
            assert!(parent.children().contains(&k), "{k:?}");
            assert!(parent.contains(k) && !k.contains(parent));
            for child in k.children_if_any() {
                assert_eq!(child.parent(), k);
            }
        }
        // Origins are exact multiples, floor-nested.
        assert_eq!(key(0, -1, 0, 0).origin(), [-16, 0, 0]);
        assert_eq!(key(2, -1, -1, 1).origin(), [-64, -64, 64]);
        assert_eq!(key(0, -1, 0, 0).parent().origin(), [-32, 0, 0]);
        // The extremes: y -8,388,608 and +8M blocks, x 30M, fit.
        let top = key(12, 0, 127, 0);
        assert_eq!(top.origin(), [0, 127 * 65_536, 0]);
        assert_eq!(key(0, 0, -(1 << 19), 0).origin()[1], -8_388_608);
        assert_eq!(
            key(0, 0, -(1 << 19), 0).containing_root(),
            key(MAX_CUBE_LEVEL, 0, (-(1i32 << 19)) >> MAX_CUBE_LEVEL, 0)
        );
        assert_eq!(key(0, 1_875_000, 0, 0).origin()[0], 30_000_000);
    }

    impl CubeKey {
        fn children_if_any(self) -> Vec<CubeKey> {
            if self.level == 0 {
                vec![]
            } else {
                self.children().to_vec()
            }
        }
        fn containing_root(self) -> CubeKey {
            let mut k = self;
            while k.level < MAX_CUBE_LEVEL {
                k = k.parent();
            }
            k
        }
    }

    #[test]
    fn a_node_holds_exactly_its_cubes() {
        let node = key(2, -1, 0, 1);
        assert!(node.contains_cube(SectionPos::new(-4, 0, 4)));
        assert!(node.contains_cube(SectionPos::new(-1, 3, 7)));
        assert!(!node.contains_cube(SectionPos::new(0, 0, 4)));
        assert!(!node.contains_cube(SectionPos::new(-5, 0, 4)));
        assert!(!node.contains_cube(SectionPos::new(-4, 4, 4)));
        assert_eq!(CubeKey::containing(2, SectionPos::new(-1, 3, 7)), node);
        assert_eq!(node.distance([-32.0, 32.0, 96.0]), 0.0);
        assert_eq!(node.distance([-100.0, 0.0, 64.0]), 36.0);
    }

    fn children_of(f: impl Fn(usize, usize, usize, usize) -> BlockStateId) -> Vec<CubeNode> {
        // f(child, x, y, z)
        let parent = key(1, 0, 0, 0);
        parent
            .children()
            .into_iter()
            .enumerate()
            .map(|(i, k)| CubeNode::from_fn(k, |x, y, z| voxel(f(i, x, y, z))))
            .collect()
    }
    fn down(children: &[CubeNode]) -> CubeNode {
        downsample(std::array::from_fn(|i| &children[i]), AIR, kind)
    }

    #[test]
    fn equal_uniform_children_stay_uniform() {
        let children = children_of(|_, _, _, _| STONE);
        let parent = down(&children);
        assert_eq!(parent.key, key(1, 0, 0, 0));
        assert_eq!(parent.as_uniform(), Some(voxel(STONE)));
    }

    #[test]
    fn a_voxel_is_solid_from_four_of_eight() {
        // In child 0, the voxel at (0,0,0) covers child voxels 0..2 each way.
        // `solid` of its eight cells are stone, the rest air.
        let with = |solid: usize| {
            let children = children_of(|c, x, y, z| {
                let cell = (x & 1) | ((z & 1) << 1) | ((y & 1) << 2);
                if c == 0 && x < 2 && y < 2 && z < 2 && cell < solid {
                    STONE
                } else {
                    AIR
                }
            });
            down(&children).get(0, 0, 0).state
        };
        assert_eq!(with(3), AIR);
        assert_eq!(with(4), STONE, "a tie keeps solid");
        assert_eq!(with(8), STONE);
        // The next voxel over saw only air.
        let children = children_of(|c, x, y, z| {
            if c == 0 && x == 0 && y == 0 && z == 0 {
                STONE
            } else {
                AIR
            }
        });
        assert_eq!(down(&children).get(1, 0, 0).state, AIR);
    }

    #[test]
    fn the_most_common_solid_wins_and_ties_go_to_the_higher_layer() {
        let at = |top: BlockStateId, bottom: BlockStateId| {
            let children = children_of(|c, x, y, z| {
                if c == 0 && x < 2 && y < 2 && z < 2 {
                    if y == 1 { top } else { bottom }
                } else {
                    AIR
                }
            });
            down(&children).get(0, 0, 0).state
        };
        assert_eq!(at(GRASS, DIRT), GRASS, "4 against 4: the top layer's");
        assert_eq!(at(DIRT, GRASS), DIRT);
        // Five stone cells against three grass: stone, wherever it is.
        let children = children_of(|c, x, y, z| {
            if c == 0 && x < 2 && y < 2 && z < 2 {
                if y == 1 && ((x == 0 && z == 0) || z == 1) {
                    GRASS
                } else {
                    STONE
                }
            } else {
                AIR
            }
        });
        // y=1 cells with (x=0,z=0) or z=1: three grass; five stone.
        assert_eq!(down(&children).get(0, 0, 0).state, STONE);
    }

    #[test]
    fn fluid_never_beats_a_solid_majority_and_needs_four_to_stand() {
        let mix = |solid: usize, fluid: usize| {
            let children = children_of(|c, x, y, z| {
                if c != 0 || x >= 2 || y >= 2 || z >= 2 {
                    return AIR;
                }
                let cell = (x & 1) | ((z & 1) << 1) | ((y & 1) << 2);
                if cell < solid {
                    STONE
                } else if cell < solid + fluid {
                    WATER
                } else {
                    AIR
                }
            });
            down(&children).get(0, 0, 0).state
        };
        assert_eq!(mix(4, 4), STONE);
        assert_eq!(mix(3, 5), WATER, "no solid majority, fluid has 5");
        assert_eq!(mix(0, 4), WATER);
        assert_eq!(mix(3, 3), AIR);
        assert_eq!(mix(2, 3), AIR);
    }

    #[test]
    fn each_parent_voxel_reads_its_own_child_and_corner() {
        // Stone only in child 5 (x half 1, z half 0, y half 1), its last
        // cell: parent voxel (8 + 7, 8 + 7, 7)... check positions map there.
        let children = children_of(|c, _, _, _| if c == 5 { STONE } else { AIR });
        let parent = down(&children);
        assert_eq!(parent.get(8, 8, 0).state, STONE);
        assert_eq!(parent.get(15, 15, 7).state, STONE);
        assert_eq!(parent.get(7, 8, 0).state, AIR);
        assert_eq!(parent.get(8, 7, 0).state, AIR);
        assert_eq!(parent.get(8, 8, 8).state, AIR);
    }

    // ---- plan

    const BOX: LoadedBox = LoadedBox {
        horizontal: 2,
        vertical: 1,
    };
    fn near(camera: [f64; 3]) -> impl Fn(SectionPos) -> bool {
        let c = cube_of(camera);
        move |p| {
            (p.x - c.x).abs() <= BOX.horizontal
                && (p.z - c.z).abs() <= BOX.horizontal
                && (p.y - c.y).abs() <= BOX.vertical
        }
    }

    /// How many leaves hold `cube`.
    fn holders(leaves: &FxHashSet<CubeKey>, cube: SectionPos) -> usize {
        (0..=MAX_CUBE_LEVEL)
            .filter(|&l| leaves.contains(&CubeKey::containing(l, cube)))
            .count()
    }

    fn check_plan(camera: [f64; 3], max: f64) -> Vec<CubeKey> {
        let loaded = near(camera);
        let leaves = plan(camera, max, BOX, &loaded);
        let set: FxHashSet<CubeKey> = leaves.iter().copied().collect();
        assert_eq!(set.len(), leaves.len(), "no node twice");
        // Non-overlapping: no leaf has a leaf ancestor.
        for &leaf in &leaves {
            let mut up = leaf;
            while up.level < MAX_CUBE_LEVEL {
                up = up.parent();
                assert!(!set.contains(&up), "{leaf:?} inside {up:?}");
            }
        }
        // Every unloaded cube in range is held once, loaded ones never.
        let c = cube_of(camera);
        let r = (max / 16.0).ceil() as i32 + 1;
        let mut covered = 0;
        for y in c.y - r..=c.y + r {
            for z in c.z - r..=c.z + r {
                for x in c.x - r..=c.x + r {
                    let cube = SectionPos::new(x, y, z);
                    let n = holders(&set, cube);
                    if loaded(cube) {
                        assert_eq!(n, 0, "loaded {cube:?} drawn");
                        continue;
                    }
                    let d = CubeKey::containing(0, cube).distance(camera);
                    if d <= max {
                        assert_eq!(n, 1, "{cube:?} at {d} held {n} times");
                        covered += 1;
                    }
                }
            }
        }
        assert!(covered > 1000);
        leaves
    }

    /// Non-overlap only, for ranges too big to walk cube by cube.
    fn check_disjoint(camera: [f64; 3], max: f64) -> Vec<CubeKey> {
        let leaves = plan(camera, max, BOX, &near(camera));
        let set: FxHashSet<CubeKey> = leaves.iter().copied().collect();
        assert_eq!(set.len(), leaves.len());
        for &leaf in &leaves {
            let mut up = leaf;
            while up.level < MAX_CUBE_LEVEL {
                up = up.parent();
                assert!(!set.contains(&up), "{leaf:?} inside {up:?}");
            }
        }
        leaves
    }

    #[test]
    fn the_plan_covers_the_range_once_minus_the_loaded_cubes() {
        check_plan([5.0, 70.0, 5.0], 400.0);
        check_plan([-1234.5, -3.0, 777.7], 300.0);
        check_plan([-15.9, -0.1, 0.1], 250.0);
        check_plan([29_999_000.0, 8_000_000.0, -29_999_000.0], 300.0);
        check_plan([100.0, -8_388_000.0, 100.0], 200.0);
    }

    #[test]
    fn the_level_grows_with_distance() {
        let camera = [3.0, 50.0, 3.0];
        let leaves = check_disjoint(camera, 2000.0);
        let mut by_level = [0; MAX_CUBE_LEVEL as usize + 1];
        for leaf in &leaves {
            by_level[leaf.level as usize] += 1;
            // Never coarser than the rule allows (the loaded box refines more).
            if leaf.level > 0 {
                let d = leaf.distance(camera);
                assert!(d >= DETAIL * f64::from(leaf.size()), "{leaf:?} at {d}");
            }
        }
        assert!(by_level[0] > 0 && by_level[3] > 0, "{by_level:?}");
        // Leaves are nearest first, and the nearest is a single cube.
        assert_eq!(leaves[0].level, 0);
        let near_level = |d: f64| {
            leaves
                .iter()
                .filter(|l| l.distance(camera) <= d)
                .map(|l| l.level)
                .max()
                .unwrap()
        };
        assert!(near_level(60.0) < near_level(600.0));
        assert!(near_level(600.0) < near_level(2000.0));
        // Detail is bounded: a few thousand nodes for a 2 km radius.
        assert!(leaves.len() < 20_000, "{}", leaves.len());
    }

    #[test]
    fn nothing_is_drawn_where_everything_is_loaded() {
        let camera = [0.5, 0.5, 0.5];
        // Everything within 3 cubes loaded: the small range is all near.
        let all = |_: SectionPos| true;
        let leaves = plan(
            camera,
            40.0,
            LoadedBox {
                horizontal: 100,
                vertical: 100,
            },
            &all,
        );
        assert!(leaves.is_empty());
    }

    #[test]
    fn a_small_camera_move_changes_few_nodes() {
        let a = [1000.0, 64.0, 1000.0];
        let b = [1003.0, 64.0, 1000.0];
        let pa: FxHashSet<_> = plan(a, 1500.0, BOX, &near(a)).into_iter().collect();
        let pb: FxHashSet<_> = plan(b, 1500.0, BOX, &near(b)).into_iter().collect();
        let changed = pa.symmetric_difference(&pb).count();
        assert!(changed * 4 < pa.len(), "{changed} of {}", pa.len());
    }

    fn assert_disjoint(leaves: &[CubeKey]) {
        let set: FxHashSet<CubeKey> = leaves.iter().copied().collect();
        assert_eq!(set.len(), leaves.len(), "no node twice");
        for &leaf in leaves {
            let mut up = leaf;
            while up.level < MAX_CUBE_LEVEL {
                up = up.parent();
                assert!(!set.contains(&up), "{leaf:?} inside {up:?}");
            }
        }
    }

    #[test]
    fn a_split_node_merges_back_only_well_past_the_distance_that_split_it() {
        let nowhere = |_: SectionPos| false;
        let node = key(3, 0, 0, 0);
        let children = node.children().to_vec();
        // 340 blocks away: past 2.5 x 128 = 320 (a fresh plan keeps the node
        // whole) and inside 1.25 times that (a plan that had it split keeps
        // it split).
        let camera = [-340.0, 64.0, 64.0];
        let fresh = plan_from(camera, 2000.0, BOX, &nowhere, &[]);
        assert!(fresh.contains(&node));
        let kept = plan_from(camera, 2000.0, BOX, &nowhere, &children);
        assert_disjoint(&kept);
        assert!(!kept.contains(&node));
        assert!(children.iter().all(|c| kept.contains(c)), "{kept:?}");
        // 500 blocks away it merges, and nothing of its children stays.
        let away = [-500.0, 64.0, 64.0];
        let merged = plan_from(away, 2000.0, BOX, &nowhere, &children);
        assert_disjoint(&merged);
        assert!(merged.contains(&node));
        assert!(!merged.iter().any(|l| node.contains(*l) && *l != node));
        // Closer than the split distance, a plan that never split it does.
        let close = [-300.0, 64.0, 64.0];
        let split = plan_from(close, 2000.0, BOX, &nowhere, &[node]);
        assert!(!split.contains(&node));
    }

    #[test]
    fn a_camera_moving_a_little_back_and_forth_changes_no_node() {
        // Both in the same cube, so the loaded box is the same.
        let a = [1000.5, 64.5, 1000.5];
        let b = [1003.5, 64.5, 1000.5];
        let max = 1500.0;
        // Leaves near enough to both cameras not to be cut by the range.
        let settled = |leaves: &[CubeKey]| -> FxHashSet<CubeKey> {
            leaves
                .iter()
                .copied()
                .filter(|l| l.distance(a) < max - 10.0 && l.distance(b) < max - 10.0)
                .collect()
        };
        let mut previous = plan_from(a, max, BOX, &near(a), &[]);
        let mut seen = Vec::new();
        for step in 0..6 {
            let camera = if step % 2 == 0 { b } else { a };
            previous = plan_from(camera, max, BOX, &near(camera), &previous);
            assert_disjoint(&previous);
            seen.push(settled(&previous));
        }
        assert!(!seen[0].is_empty());
        for later in &seen[1..] {
            assert_eq!(*later, seen[0], "the plan flipped on a 3 block move");
        }
    }

    #[test]
    fn a_parent_column_is_covered_when_two_of_the_four_columns_over_it_are() {
        let cover = |columns: &[(usize, usize)]| {
            let mut mask = Box::new([0u64; 4]);
            for &(x, z) in columns {
                let bit = z << 4 | x;
                mask[bit >> 6] |= 1 << (bit & 63);
            }
            Some(mask)
        };
        let mut children = children_of(|_, _, _, _| STONE);
        assert!(!down(&children).column_covered(0, 0));
        // Child 4 is the upper one at x = z = 0: parent column (0, 0) holds
        // its columns (0..2, 0..2), parent column (1, 0) the ones at x 2..4.
        children[4] = children[4]
            .clone()
            .with_covered(cover(&[(0, 0), (1, 0), (2, 0)]));
        // A lower child's cover is under the node: it counts for nothing.
        children[0] = children[0]
            .clone()
            .with_covered(cover(&[(4, 4), (5, 4), (4, 5), (5, 5)]));
        let parent = down(&children);
        assert!(parent.column_covered(0, 0), "two of four");
        assert!(!parent.column_covered(1, 0), "one of four");
        assert!(!parent.column_covered(2, 2), "the lower child's");
        assert_eq!(parent.as_uniform(), Some(voxel(STONE)));
    }

    #[test]
    fn forgetting_cubes_drops_the_kept_nodes_over_them() {
        let mut cache = Cache::default();
        let kept = [key(0, 0, 0, 0), key(1, 5, 5, 5), key(2, -1, 0, 0)];
        let dropped = [key(1, 0, 0, 0), key(3, 0, 0, 0), key(0, 1, 1, 1)];
        for k in kept.iter().chain(&dropped) {
            cache.insert(Arc::new(CubeNode::uniform(*k, voxel(STONE))));
        }
        cache.forget(&[SectionPos::new(1, 1, 1)]);
        assert!(kept.iter().all(|k| cache.nodes.contains_key(k)));
        assert!(dropped.iter().all(|k| !cache.nodes.contains_key(k)));
    }

    // ---- producer

    /// A world of stone below y = 20.
    struct Flat;
    impl CubeSampler for Flat {
        fn node(&mut self, key: CubeKey) -> Option<CubeNode> {
            let scale = 1 << key.level;
            let oy = key.origin()[1];
            Some(CubeNode::from_fn(key, |_, y, _| {
                let wy = oy + y as i32 * scale;
                voxel(if wy < 20 { STONE } else { AIR })
            }))
        }
        fn fork(&self) -> Option<Box<dyn CubeSampler>> {
            Some(Box::new(Flat))
        }
    }

    #[test]
    fn nodes_the_sampler_lacks_are_built_from_their_children() {
        struct OnlyCubes;
        impl CubeSampler for OnlyCubes {
            fn node(&mut self, key: CubeKey) -> Option<CubeNode> {
                (key.level == 0).then(|| Flat.node(key).unwrap())
            }
            fn fork(&self) -> Option<Box<dyn CubeSampler>> {
                None
            }
        }
        let shared = Shared::new();
        let k = key(2, 0, 0, 0);
        let built = node(&shared, &mut OnlyCubes, k, AIR, &kind).unwrap();
        let direct = Flat.node(k).unwrap();
        for i in 0..VOLUME {
            assert_eq!(built.at(i), direct.at(i), "voxel {i}");
        }
        // Too coarse to build from children.
        assert!(node(&shared, &mut OnlyCubes, key(6, 0, 0, 0), AIR, &kind).is_none());
        // The children are cached for the next parent.
        assert!(shared.cache.lock().unwrap().nodes.len() >= 8);
    }

    fn lod_for_test() -> CubeLod {
        let shared = Arc::new(Shared::new());
        let (_tx, results) = mpsc::channel();
        CubeLod::assemble(shared, results, Vec::new(), None, None, AIR)
    }

    /// Puts `keys` in as the plan, as the planner thread would hand it over,
    /// with `cover` as its coverage layer.
    fn put(lod: &mut CubeLod, keys: &[CubeKey], cover: &[CubeKey]) {
        put_far(lod, keys, &[], cover);
    }

    /// [`put`] with surface tiles.
    fn put_far(lod: &mut CubeLod, keys: &[CubeKey], tiles: &[TileKey], cover: &[CubeKey]) {
        let next_tiles: FxHashSet<TileKey> = tiles.iter().copied().collect();
        let tiles_added = tiles
            .iter()
            .copied()
            .filter(|k| !lod.tiles.wanted.contains(k))
            .collect();
        let tiles_removed = lod
            .tiles
            .wanted
            .iter()
            .copied()
            .filter(|k| !next_tiles.contains(k))
            .collect();
        let mut footprints: Vec<(u64, u8, FarKey)> = keys
            .iter()
            .map(|&k| (footprint_order(k.level, k.x, k.z), k.level, FarKey::Cube(k)))
            .chain(
                tiles
                    .iter()
                    .map(|&t| (footprint_order(t.level, t.x, t.z), t.level, FarKey::Tile(t))),
            )
            .collect();
        footprints.sort_unstable();
        let next: FxHashSet<CubeKey> = keys.iter().copied().collect();
        let added = keys
            .iter()
            .copied()
            .filter(|k| !lod.wanted.contains(k))
            .collect();
        let removed = lod
            .wanted
            .iter()
            .copied()
            .filter(|k| !next.contains(k))
            .collect();
        let camera = lod.camera;
        let mut leaves = keys.to_vec();
        leaves.sort_by(|a, b| a.distance(camera).total_cmp(&b.distance(camera)));
        let mut tree: Vec<(u128, CubeKey)> = keys.iter().map(|&k| (tree_order(k), k)).collect();
        tree.sort_unstable();
        let done = PlanDone {
            epoch: lod.epoch,
            tree,
            order: leaves.iter().map(|&k| (sector(camera, k), k)).collect(),
            leaves: Arc::new(leaves),
            changed: true,
            added,
            removed,
            cover: cover.iter().copied().collect(),
            tiles: Arc::new(tiles.to_vec()),
            tiles_added,
            tiles_removed,
            cover_tiles: FxHashSet::default(),
            footprints,
            pruned: 0,
            limited: false,
            coarsened: 1.0,
            tiles_limited: false,
            elapsed: Duration::ZERO,
        };
        lod.install(done, Instant::now());
    }

    #[test]
    fn a_node_is_removed_only_once_what_replaces_it_is_in() {
        let mut lod = lod_for_test();
        let coarse = key(1, 0, 0, 0);
        let kids = coarse.children();
        put(&mut lod, &[coarse], &[]);
        lod.deliver(coarse, false);
        // The plan now wants the eight children; none is in yet.
        put(&mut lod, &kids, &[]);
        assert!(lod.removed().is_empty(), "children missing: a hole");
        for &k in &kids[..7] {
            lod.deliver(k, false);
        }
        assert!(lod.removed().is_empty(), "one child missing");
        lod.deliver(kids[7], false);
        assert_eq!(lod.removed(), vec![coarse]);
        assert!(lod.removed().is_empty(), "reported once");
        // The other way: children out, parent wanted.
        put(&mut lod, &[coarse], &[]);
        assert!(lod.removed().is_empty(), "the parent is not in");
        lod.deliver(coarse, false);
        assert_eq!(lod.removed().len(), 8);
        // Out of range, nothing replaces it.
        put(&mut lod, &[], &[]);
        assert_eq!(lod.removed(), vec![coarse]);
    }

    #[test]
    fn the_plan_is_dropped_by_clear() {
        let mut lod = lod_for_test();
        put(&mut lod, &[key(0, 0, 0, 0)], &[]);
        lod.deliver(key(0, 0, 0, 0), false);
        lod.clear();
        assert!(lod.plan.is_empty() && lod.delivered.is_empty());
        assert!(lod.removed().is_empty());
        assert_eq!(lod.stats().pending, 0);
    }

    // ---- detail by size on screen

    const NONE: LoadedBox = LoadedBox {
        horizontal: 0,
        vertical: 0,
    };

    fn mixed(_: CubeKey) -> NodeClass {
        NodeClass::Mixed
    }

    /// The leaf of `plan` holding the block at `p`.
    fn leaf_at(plan: &[CubeKey], p: [f64; 3]) -> CubeKey {
        let cube = cube_of(p);
        *plan
            .iter()
            .find(|k| k.contains_cube(cube))
            .unwrap_or_else(|| panic!("nothing holds {p:?}"))
    }

    /// Pixels a voxel of `node` spans at its nearest point from `camera`,
    /// with a vertical field of view of `fov` degrees, 720 pixels tall.
    fn voxel_px(node: CubeKey, camera: [f64; 3], fov: f64) -> f64 {
        let focal = 720.0 / (2.0 * (fov.to_radians() / 2.0).tan());
        f64::from(1 << node.level) / node.distance(camera) * focal
    }

    /// The angle in degrees between `forward` and the direction to `node`'s
    /// centre.
    fn off_axis(node: CubeKey, camera: [f64; 3], forward: [f64; 3]) -> f64 {
        let half = f64::from(node.size()) / 2.0;
        let o = node.origin();
        let d = [0, 1, 2].map(|a| f64::from(o[a]) + half - camera[a]);
        angle_between(d, forward).to_degrees()
    }

    #[test]
    fn a_narrower_view_refines_the_nodes_in_view_only() {
        let camera = [0.5, 100.5, 0.5];
        let ahead = [0.0, 0.0, 1.0];
        let view = FarView::standard(8.0);
        // The spyglass: a tenth of the field of view.
        let zoom = view.looking(ahead, 7f64.to_radians());
        let range = 30_000.0;
        let wide = plan_classified(
            camera,
            range,
            &view.rule(usize::MAX),
            NONE,
            &|_| false,
            &[],
            &mixed,
        );
        let narrow = plan_classified(
            camera,
            range,
            &zoom.rule(usize::MAX),
            NONE,
            &|_| false,
            &[],
            &mixed,
        );
        let far_ahead = [0.5, 100.5, 8000.5];
        let far_behind = [0.5, 100.5, -8000.5];
        // Ten times the magnification is a bit over three levels finer.
        let (w, n) = (
            leaf_at(&wide.leaves, far_ahead).level,
            leaf_at(&narrow.leaves, far_ahead).level,
        );
        assert!(n + 3 <= w, "ahead: level {n} zoomed, {w} not");
        assert_eq!(
            leaf_at(&narrow.leaves, far_behind),
            leaf_at(&wide.leaves, far_behind),
            "behind the camera the zoom changes nothing"
        );
        // On screen, no voxel is over the target at its node's nearest point;
        // and each leaf's parent would have been.
        for (plan, fov, within) in [(&wide.leaves, 70.0, 20.0), (&narrow.leaves, 7.0, 2.0)] {
            let mut seen = 0;
            for &leaf in plan.iter().filter(|k| k.level > 0) {
                if off_axis(leaf, camera, ahead) > within {
                    continue;
                }
                seen += 1;
                let px = voxel_px(leaf, camera, fov);
                assert!(px <= 8.0 + 1e-9, "{leaf:?}: {px} px at {fov} degrees");
                let parent = voxel_px(leaf.parent(), camera, fov);
                assert!(parent > 8.0, "{leaf:?} split needlessly: {parent} px");
            }
            assert!(seen > 20, "{seen} leaves in view at {fov} degrees");
        }
        // Behind the zoomed camera, the unzoomed rule holds.
        for &leaf in narrow.leaves.iter().filter(|k| k.level > 0) {
            if off_axis(leaf, camera, ahead) > 120.0 {
                assert!(voxel_px(leaf, camera, 70.0) <= 8.0 + 1e-9, "{leaf:?}");
            }
        }
    }

    #[test]
    fn a_finer_target_asks_for_more_nodes_and_the_budget_coarsens_evenly() {
        let camera = [0.5, 100.5, 0.5];
        let range = 8000.0;
        let plan_for = |target: f64, budget: usize| {
            plan_classified(
                camera,
                range,
                &FarView::standard(target).rule(budget),
                NONE,
                &|_| false,
                &[],
                &mixed,
            )
        };
        let coarse = plan_for(16.0, usize::MAX);
        let fine = plan_for(4.0, usize::MAX);
        assert!(!coarse.limited && !fine.limited);
        assert!(fine.leaves.len() > coarse.leaves.len() * 4);
        let budget = fine.leaves.len() / 3;
        let capped = plan_for(4.0, budget);
        assert!(capped.limited && capped.coarsened > 1.0);
        assert!(capped.leaves.len() <= budget, "{}", capped.leaves.len());
        assert_disjoint(&capped.leaves);
        // What is smallest on screen stays whole first: no leaf's voxels are
        // over `coarsened` times the target, and no node was split under it.
        let over = |k: CubeKey| voxel_px(k, camera, 70.0) / 4.0;
        for &leaf in capped.leaves.iter().filter(|k| k.level > 0) {
            assert!(over(leaf) <= capped.coarsened + 1e-9, "{leaf:?}");
            assert!(over(leaf.parent()) >= capped.coarsened - 1e-9, "{leaf:?}");
        }
        // Far out it is coarser than the full plan, and every block in range
        // is still drawn once.
        let far = [0.5, 100.5, 7000.5];
        assert!(leaf_at(&capped.leaves, far).level >= leaf_at(&fine.leaves, far).level);
        for p in [
            far,
            [40.5, 100.5, 20.5],
            [-3000.5, -300.5, 4000.5],
            [5000.5, 1500.5, -100.5],
        ] {
            let cube = cube_of(p);
            let holders = capped
                .leaves
                .iter()
                .filter(|k| k.contains_cube(cube))
                .count();
            assert_eq!(holders, 1, "{p:?}");
        }
    }

    #[test]
    fn zooming_out_merges_only_past_the_hysteresis() {
        let camera = [0.5, 100.5, 0.5];
        let ahead = [0.0, 0.0, 1.0];
        let view = FarView::standard(8.0);
        let zoomed = view.looking(ahead, 7f64.to_radians());
        let range = 30_000.0;
        let plan_with = |v: &FarView, previous: &[CubeKey]| {
            plan_classified(
                camera,
                range,
                &v.rule(usize::MAX),
                NONE,
                &|_| false,
                previous,
                &mixed,
            )
            .leaves
        };
        let narrow = plan_with(&zoomed, &[]);
        let back = plan_with(&view, &narrow);
        assert_disjoint(&back);
        assert!(back.len() < narrow.len());
        // A node stays split after the zoom only while its voxels are more
        // than 1 / MERGE of the target on screen; a leaf is never over it.
        let focal = view.focal_px(view.fov_y);
        let ratio = |k: CubeKey| f64::from(1 << k.level) / k.distance(camera) * focal / 8.0;
        for &leaf in back.iter().filter(|k| k.level > 0) {
            assert!(ratio(leaf) <= 1.0 + 1e-9, "{leaf:?}");
            assert!(ratio(leaf.parent()) * MERGE > 1.0, "{leaf:?} kept split");
        }
        // A sprint's wider view and back again changes nothing.
        let sprint = view.looking(ahead, 77f64.to_radians());
        let settled = plan_with(&view, &plan_with(&view, &[]));
        let after = plan_with(&view, &plan_with(&sprint, &settled));
        let set = |v: &[CubeKey]| v.iter().copied().collect::<FxHashSet<_>>();
        assert_eq!(set(&after), set(&settled));
    }

    #[test]
    fn tiles_from_the_surface_level_complex_ones_sooner() {
        let high = key(SURFACE_LEVEL, 3, 0, -2);
        // Not generated yet: a tile first.
        assert_eq!(node_kind(high, None, None), NodeKind::Tile);
        let plain = Some(TileInfo::default());
        let complex = Some(TileInfo {
            overflow_columns: 3,
            ..TileInfo::default()
        });
        assert_eq!(node_kind(high, None, plain), NodeKind::Tile);
        assert_eq!(node_kind(high.children()[0], None, plain), NodeKind::Cube);
        // A complex tile turns into cube nodes sooner.
        assert_eq!(node_kind(high, None, complex), NodeKind::Cube);
        let higher = key(SURFACE_LEVEL + COMPLEX_LIFT, 0, 0, 0);
        assert_eq!(node_kind(higher, None, complex), NodeKind::Tile);
    }

    // ---- surface tiles in the plan

    const WHOLE: f64 = 67_108_864.0;

    /// Air above y 1,000, rock below y -1,000: the terrain lies between.
    fn banded(k: CubeKey) -> NodeClass {
        let [_, y, _] = k.origin();
        if i64::from(y) > 1000 {
            NodeClass::Air
        } else if i64::from(y) + i64::from(k.size()) < -1000 {
            NodeClass::Rock
        } else {
            NodeClass::Mixed
        }
    }

    /// A Whole World plan with tiles, at 16 px, with the headers `info` gives.
    fn tiled(camera: [f64; 3], info: &dyn Fn(TileKey) -> Option<TileInfo>) -> Plan {
        let loaded = near(camera);
        plan_far(&PlanInput {
            camera,
            max_distance: WHOLE,
            rule: &FarView::standard(16.0).rule(usize::MAX),
            loaded: BOX,
            is_loaded: &loaded,
            previous: &[],
            class: &banded,
            surface: Some(Surface {
                info,
                sea_level: 63,
                budget: usize::MAX,
                previous: &[],
            }),
        })
    }

    /// Whether the footprint of `t` holds the block column `(x, z)`.
    fn tile_holds(t: TileKey, x: i64, z: i64) -> bool {
        let [ox, oz] = t.origin().map(i64::from);
        let size = i64::from(t.size());
        (ox..ox + size).contains(&x) && (oz..oz + size).contains(&z)
    }

    #[test]
    fn no_cube_node_shares_a_footprint_with_a_tile() {
        let camera = [0.5, 100.5, 0.5];
        let plan = tiled(camera, &|_| None);
        assert!(plan.tiles.len() > 100, "{} tiles", plan.tiles.len());
        assert!(!plan.leaves.is_empty());
        let tiles: FxHashSet<TileKey> = plan.tiles.iter().copied().collect();
        assert_eq!(tiles.len(), plan.tiles.len());
        // Tiles overlap neither each other nor any node's footprint, at any height.
        let mut under = FxHashSet::default();
        for &t in &plan.tiles {
            for l in t.level + 1..=MAX_CUBE_LEVEL {
                let up = l - t.level;
                let a = TileKey::new(l, t.x >> up, t.z >> up);
                assert!(!tiles.contains(&a), "{t:?} inside {a:?}");
                under.insert(a);
            }
        }
        for &c in &plan.leaves {
            assert!(
                !under.contains(&TileKey::new(c.level, c.x, c.z)),
                "{c:?} over a tile"
            );
            for l in c.level..=MAX_CUBE_LEVEL {
                let up = l - c.level;
                let t = TileKey::new(l, c.x >> up, c.z >> up);
                assert!(!tiles.contains(&t), "{c:?} under {t:?}");
            }
        }
        // Far away the ground is a tile, and no tile is finer than the
        // surface level.
        assert!(
            plan.tiles
                .iter()
                .any(|&t| tile_holds(t, 1_000_000, -2_000_000))
        );
        assert!(plan.tiles.iter().all(|t| t.level >= SURFACE_LEVEL));
    }

    #[test]
    fn the_camera_footprint_is_never_a_tile() {
        for camera in [
            [0.5, 100.5, 0.5],
            // Deep underground, in a cavern: the surface far above stays
            // tiles elsewhere, the rock between hides it.
            [12_345.5, -2000.5, -6789.5],
            // On a corner of the surface level's tiles.
            [65_535.5, 80.5, 65_535.5],
        ] {
            let plan = tiled(camera, &|_| None);
            assert!(!plan.tiles.is_empty());
            let c = cube_of(camera);
            let h = i64::from(BOX.horizontal);
            for &t in &plan.tiles {
                for dz in -h..=h {
                    for dx in -h..=h {
                        let x = (i64::from(c.x) + dx) * 16;
                        let z = (i64::from(c.z) + dz) * 16;
                        assert!(
                            !tile_holds(t, x, z),
                            "{t:?} over the loaded cubes at {camera:?}"
                        );
                    }
                }
            }
            // Around the camera, the far terrain is cube nodes.
            assert!(
                plan.leaves.iter().any(|k| k.distance(camera) < 64.0),
                "{camera:?}"
            );
        }
    }

    #[test]
    fn a_complex_tile_turns_into_cube_nodes_sooner() {
        let camera = [0.5, 100.5, 0.5];
        let first = tiled(camera, &|_| None);
        let pick = |level: u8| {
            *first
                .tiles
                .iter()
                .find(|t| t.level == level)
                .unwrap_or_else(|| panic!("no tile of level {level}"))
        };
        let lifted = pick(SURFACE_LEVEL + 1);
        let kept = pick(SURFACE_LEVEL + COMPLEX_LIFT);
        // The same heights the plan assumes before a tile is generated.
        let complex = TileInfo {
            overflow_columns: 7,
            highest: 63 + UNKNOWN_ABOVE - 1,
            skirt: 63 - UNKNOWN_BELOW,
            ..TileInfo::default()
        };
        let info = move |t: TileKey| (t == lifted || t == kept).then_some(complex);
        let second = tiled(camera, &info);
        assert!(
            !second.tiles.contains(&lifted),
            "a complex tile below the lift stays"
        );
        let inside = |c: &CubeKey| {
            let [x, _, z] = c.origin().map(i64::from);
            tile_holds(lifted, x, z) && c.size() <= lifted.size()
        };
        assert!(second.leaves.iter().any(inside), "no node under {lifted:?}");
        // From the lift's level up, a complex tile is still a tile.
        assert!(second.tiles.contains(&kept));
    }

    #[test]
    fn a_tile_fades_out_only_once_the_nodes_under_it_are_in() {
        let mut lod = lod_for_test();
        lod.tiles.on = true;
        let tile = TileKey::new(SURFACE_LEVEL, 3, -2);
        put_far(&mut lod, &[], &[tile], &[]);
        assert_eq!(lod.take_tile_work().request, vec![tile]);
        lod.surface_feedback(|k| (k == tile).then_some(TileInfo::default()));
        assert!(lod.stats().complete);
        // The plan turns the tile into the cube nodes of its footprint.
        let node = CubeKey::new(SURFACE_LEVEL, tile.x, 0, tile.z);
        let kids = node.children();
        put_far(&mut lod, &kids, &[], &[]);
        assert!(lod.removed().is_empty());
        assert!(
            lod.take_tile_work().retire.is_empty(),
            "nothing under it is in"
        );
        for &k in &kids[..7] {
            lod.deliver(k, false);
        }
        assert!(lod.removed().is_empty());
        assert!(lod.take_tile_work().retire.is_empty(), "one node missing");
        lod.deliver(kids[7], false);
        assert!(lod.removed().is_empty());
        assert_eq!(lod.take_tile_work().retire, vec![tile]);
        // And back: the nodes stay until the tile is in again.
        put_far(&mut lod, &[], &[tile], &[]);
        assert_eq!(lod.take_tile_work().request, vec![tile]);
        assert!(lod.removed().is_empty(), "the tile is not in");
        lod.surface_feedback(|k| (k == tile).then_some(TileInfo::default()));
        assert_eq!(lod.removed().len(), 8);
    }

    // ---- coverage first, and the queue's order

    #[test]
    fn a_big_change_draws_the_coverage_layer_first() {
        let mut lod = lod_for_test();
        let camera = [5000.5, 120.5, -3000.5];
        lod.camera = camera;
        let job = PlanJob {
            epoch: lod.epoch,
            camera,
            max_distance: 50_000.0,
            rule: FarView::standard(8.0).rule(usize::MAX),
            loaded: NONE,
            snapshot: LoadedSnapshot::default(),
            previous: Arc::default(),
            surface: None,
        };
        let done = make_plan(
            &job,
            &mut FxHashSet::default(),
            &mut FxHashSet::default(),
            &mixed,
        );
        let plan: Vec<CubeKey> = done.leaves.to_vec();
        lod.install(done, Instant::now());
        assert!(matches!(lod.phase, Phase::Cover { .. }));
        assert!(lod.stats().plan_covered.is_none());
        // Only the layer is queued: few nodes, one over each planned node.
        let wave = lod.shared.queue.lock().unwrap().drain();
        assert!(wave.len() <= COVER_NODES && wave.len() * 10 < plan.len());
        for &leaf in &plan {
            let over = wave.iter().filter(|w| w.contains(leaf)).count();
            assert_eq!(over, 1, "{leaf:?}");
        }
        // Nearest first within the layer's direction around the camera.
        assert!(wave[0].distance(camera) < 1.0);
        // The layer comes in: the place is covered, and the rest is queued.
        for &w in &wave {
            lod.deliver(w, false);
        }
        let now = Instant::now();
        lod.recheck_holes(now, false);
        lod.advance_phase(now);
        lod.track_completion(now);
        assert!(matches!(lod.phase, Phase::Refine));
        assert!(lod.stats().plan_covered.is_some());
        let rest = lod.shared.queue.lock().unwrap().drain();
        let delivered_plan = plan.iter().filter(|k| wave.contains(k)).count();
        assert_eq!(rest.len(), plan.len() - delivered_plan);
        // The coverage nodes stay until the nodes under them are in.
        if let Some(earlier) = Instant::now().checked_sub(Duration::from_secs(1)) {
            lod.released_at = earlier;
        }
        assert!(lod.removed().is_empty());
        for &k in &rest {
            lod.deliver(k, false);
        }
        let gone: FxHashSet<CubeKey> = lod.removed().into_iter().collect();
        let layer: FxHashSet<CubeKey> =
            wave.iter().copied().filter(|k| !plan.contains(k)).collect();
        assert_eq!(gone, layer);
        assert!(lod.stats().complete || lod.gpu_path());
    }

    #[test]
    fn a_small_change_skips_the_coverage_layer() {
        let mut lod = lod_for_test();
        let coarse = key(6, 0, 0, 0);
        put(&mut lod, &[coarse], &[]);
        lod.deliver(coarse, false);
        lod.advance_phase(Instant::now());
        lod.shared.queue.lock().unwrap().clear();
        // Its children replace it: the parent still covers them.
        put(&mut lod, &coarse.children(), &[coarse]);
        assert!(matches!(lod.phase, Phase::Refine));
        assert!(lod.holes.is_empty());
        let queued = lod.shared.queue.lock().unwrap().drain();
        assert_eq!(queued.len(), 8);
    }

    #[test]
    fn the_queue_takes_what_the_camera_faces_first() {
        let mut lod = lod_for_test();
        let camera = [0.5, 70.5, 0.5];
        lod.camera = camera;
        // Level-4 nodes (256 blocks) in a ring 2,000 blocks out, every 30 degrees.
        let ring: Vec<CubeKey> = (0..12)
            .map(|i| {
                let a = f64::from(i) * 30f64.to_radians();
                CubeKey::containing(4, cube_of([2000.0 * a.sin(), 70.0, 2000.0 * a.cos()]))
            })
            .collect();
        let near = key(4, 0, 0, 0);
        let mut all = ring.clone();
        all.push(near);
        all.sort_by(|a, b| a.distance(camera).total_cmp(&b.distance(camera)));
        lod.order = all.iter().map(|&k| (sector(camera, k), k)).collect();
        let first_of = |lod: &mut CubeLod, facing: [f64; 3]| {
            lod.facing = None;
            lod.face(facing);
            lod.fill_queue();
            lod.shared.queue.lock().unwrap().drain()
        };
        // Looking along +Z: the node around the camera, then the one ahead,
        // the one behind last.
        let order = first_of(&mut lod, [0.0, 0.0, 1.0]);
        assert_eq!(order.len(), all.len());
        assert_eq!(order[0], near);
        assert_eq!(order[1], ring[0], "{order:?}");
        assert_eq!(*order.last().unwrap(), ring[6]);
        // Turned around: behind comes first. Nothing is sorted again, the
        // sectors are taken in another order.
        let turned = first_of(&mut lod, [0.0, 0.0, -1.0]);
        assert_eq!(turned[1], ring[6]);
        assert_eq!(*turned.last().unwrap(), ring[0]);
        // A hole comes before anything, whichever way the camera looks.
        lod.holes = vec![(ring[6], None)];
        let holed = first_of(&mut lod, [0.0, 0.0, 1.0]);
        assert_eq!(holed[0], ring[6]);
        assert_eq!(holed.iter().filter(|&&k| k == ring[6]).count(), 1);
    }

    #[test]
    fn a_turn_reorders_the_queue_taken_from() {
        let mut queue = Queue::default();
        let camera = [0.5, 0.5, 0.5];
        let east = key(3, 20, 0, 0);
        let west = key(3, -21, 0, 0);
        for k in [east, west] {
            queue.sectors[usize::from(sector(camera, k))].push_back(k);
        }
        queue.order = sector_order([1.0, 0.0, 0.0]);
        assert_eq!(queue.pop(), Some(east));
        queue.sectors[usize::from(sector(camera, east))].push_back(east);
        queue.order = sector_order([-1.0, 0.0, 0.0]);
        assert_eq!(queue.pop(), Some(west));
        // A key the plan dropped is skipped.
        queue.dropped.insert(east);
        assert_eq!(queue.pop(), None);
    }
}

#[cfg(test)]
mod whole_world_tests {
    use super::*;

    const NONE: LoadedBox = LoadedBox {
        horizontal: 0,
        vertical: 0,
    };
    const WORLD: f64 = 67_108_864.0;

    fn disjoint(plan: &[CubeKey]) {
        let set: FxHashSet<CubeKey> = plan.iter().copied().collect();
        for &leaf in plan {
            let mut up = leaf;
            while up.level < MAX_CUBE_LEVEL {
                up = up.parent();
                assert!(!set.contains(&up), "{leaf:?} lies inside {up:?}");
            }
        }
    }

    #[test]
    fn keys_stay_exact_at_the_top_level() {
        let top = CubeKey::new(MAX_CUBE_LEVEL, -1, -1, 0);
        assert_eq!(top.size(), 1 << 26);
        assert_eq!(top.origin(), [-(1 << 26), -(1 << 26), 0]);
        assert_eq!(top.children()[7].level, MAX_CUBE_LEVEL - 1);
        assert!(top.contains(top.children()[3]));
        assert!(in_world(CubeKey::new(MAX_CUBE_LEVEL, 0, 0, 0)));
        // The border: the last node inside it, and one wholly past it.
        let edge = 30_000_000 / 16;
        assert!(in_world(CubeKey::new(0, edge - 1, 0, 0)));
        assert!(!in_world(CubeKey::new(0, edge, 0, 0)));
        assert!(!in_world(CubeKey::new(0, 0, 1 << 19, 0)));
        assert!(in_world(CubeKey::new(0, 0, -(1 << 19), 0)));
    }

    #[test]
    fn the_whole_world_plan_is_bounded_disjoint_and_inside_the_world() {
        let Plan {
            leaves: plan,
            pruned,
            ..
        } = plan_classified(
            [0.0, 100.0, 0.0],
            WORLD,
            &Rule::distance(DETAIL),
            NONE,
            &|_| false,
            &[],
            &|_| NodeClass::Mixed,
        );
        assert_eq!(pruned, 0);
        assert!(!plan.is_empty() && plan.len() < 100_000, "{}", plan.len());
        assert!(plan.iter().all(|k| in_world(*k)));
        assert!(plan.iter().any(|k| k.level >= 18), "coarse nodes far out");
        assert!(plan.iter().any(|k| k.level == 0), "fine ones near");
        disjoint(&plan);
    }

    #[test]
    fn nodes_known_uniform_are_left_out_or_stay_whole() {
        let camera = [0.0, 100.0, 0.0];
        let all = |_: SectionPos| false;
        let rule = Rule::distance(DETAIL);
        let mixed =
            plan_classified(camera, WORLD, &rule, NONE, &all, &[], &|_| NodeClass::Mixed).leaves;
        // Everything above y = 1000 is air, everything below y = -1000 rock.
        let class = |k: CubeKey| {
            let [_, y, _] = k.origin();
            if i64::from(y) > 1000 {
                NodeClass::Air
            } else if i64::from(y) + i64::from(k.size()) < -1000 {
                NodeClass::Rock
            } else {
                NodeClass::Mixed
            }
        };
        let Plan {
            leaves: plan,
            pruned,
            ..
        } = plan_classified(camera, WORLD, &rule, NONE, &all, &[], &class);
        assert!(pruned > 0);
        assert!(
            plan.len() < mixed.len() / 2,
            "{} of {}",
            plan.len(),
            mixed.len()
        );
        assert!(plan.iter().all(|k| class(*k) != NodeClass::Air));
        // A rock node is a leaf however near the camera: none of its parts is planned.
        for k in plan.iter().filter(|k| class(**k) == NodeClass::Rock) {
            assert!(!plan.iter().any(|o| o != k && k.contains(*o)));
        }
        disjoint(&plan);
    }

    #[test]
    fn a_node_with_loaded_cubes_still_splits_around_them() {
        let loaded = LoadedBox {
            horizontal: 2,
            vertical: 2,
        };
        let plan = plan_classified(
            [8.0, 8.0, 8.0],
            4096.0,
            &Rule::distance(DETAIL),
            loaded,
            &|_| true,
            &[],
            &|_| NodeClass::Rock,
        )
        .leaves;
        // Every cube around the camera is drawn by the near renderer: none planned inside.
        for k in &plan {
            for dy in -2..=2 {
                for dz in -2..=2 {
                    for dx in -2..=2 {
                        assert!(!k.contains_cube(SectionPos::new(dx, dy, dz)), "{k:?}");
                    }
                }
            }
        }
    }
}
