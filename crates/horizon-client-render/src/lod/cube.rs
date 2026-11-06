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
//! [`plan`] chooses the nodes (pure, no threads); [`CubeLod`] samples and
//! meshes them in the background.
use super::mesh::{self, Mesher};
use crate::{
    fast_hash::{FxHashMap, FxHashSet},
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

/// The coarsest level: a node 65,536 blocks wide.
pub const MAX_CUBE_LEVEL: u8 = 12;
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
}

impl CubeNode {
    pub fn uniform(key: CubeKey, voxel: Voxel) -> Self {
        Self {
            key,
            voxels: Voxels::Uniform(voxel),
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
    fn dense(key: CubeKey, dense: Box<[Voxel; VOLUME]>) -> Self {
        let first = dense[0];
        if dense.iter().all(|&v| v == first) {
            Self::uniform(key, first)
        } else {
            Self {
                key,
                voxels: Voxels::Dense(dense),
            }
        }
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
        return CubeNode::uniform(key, first);
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
    CubeNode::dense(key, dense)
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
    /// An independent sampler for another thread, if this one can have one.
    /// Without it the producer runs a single worker.
    fn fork(&self) -> Option<Box<dyn CubeSampler>>;
}

// ---------------------------------------------------------------- the plan

/// The nodes subdivide while closer than this many times their size.
pub const DETAIL: f64 = 2.5;
/// Nodes this level or finer ask `is_loaded` about each cube they hold; coarser
/// ones that meet the loaded box simply subdivide.
const PROBE_LEVEL: u8 = 2;

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
    let center = cube_of(camera);
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
    // Whole nodes only: the cube range of a level is a power of two.
    let y_limit = 1_i32 << (19 - top);
    let mut stack = Vec::new();
    let (x0, x1) = span(camera[0]);
    let (y0, y1) = span(camera[1]);
    let (z0, z1) = span(camera[2]);
    for x in x0..=x1 {
        for y in y0.max(-y_limit)..=y1.min(y_limit - 1) {
            for z in z0..=z1 {
                stack.push(CubeKey::new(top, x, y, z));
            }
        }
    }
    let mut leaves = Vec::new();
    while let Some(node) = stack.pop() {
        let distance = node.distance(camera);
        if distance > max_distance {
            continue;
        }
        match loaded_state(node, center, loaded, is_loaded) {
            Loaded::All => continue,
            Loaded::Some => {}
            Loaded::None if node.level > 0 && distance < DETAIL * f64::from(node.size()) => {}
            Loaded::None => {
                leaves.push((distance, node));
                continue;
            }
        }
        stack.extend(node.children());
    }
    leaves.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    leaves.into_iter().map(|(_, node)| node).collect()
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

// ---------------------------------------------------------------- producer

/// Nodes kept after sampling, so a node meshed again (or the parent of
/// children already sampled) is not sampled twice.
const CACHE_NODES: usize = 1024;
/// How often the plan is recomputed when the camera stays in one cube, for
/// cubes that loaded or unloaded meanwhile.
const REPLAN: Duration = Duration::from_millis(250);
/// How long a node whose sampling failed waits before it is asked again.
const RETRY: Duration = Duration::from_secs(1);

struct Done {
    key: CubeKey,
    mesh: Option<SectionMesh>,
}

#[derive(Default)]
struct Queue {
    /// Waiting, nearest first.
    jobs: VecDeque<CubeKey>,
    running: FxHashSet<CubeKey>,
}

#[derive(Default)]
struct Cache {
    nodes: FxHashMap<CubeKey, (Arc<CubeNode>, u64)>,
    tick: u64,
}

impl Cache {
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

struct Shared {
    queue: Mutex<Queue>,
    wake: Condvar,
    stop: AtomicBool,
    cache: Mutex<Cache>,
}

/// The far terrain of a cubic level, built in the background.
///
/// ```ignore
/// // Once, when the level loads:
/// let mut far = CubeLod::new(Box::new(generator_sampler), context.clone(), air);
/// // Every frame:
/// let meshes = far.update(camera, max_distance, LoadedBox { horizontal: h, vertical: v },
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
/// [`removed`]: CubeLod::removed
pub struct CubeLod {
    shared: Arc<Shared>,
    results: mpsc::Receiver<Done>,
    threads: Vec<thread::JoinHandle<()>>,
    /// The current plan, nearest first.
    plan: Vec<CubeKey>,
    wanted: FxHashSet<CubeKey>,
    /// Handed out and not reported removed.
    delivered: FxHashSet<CubeKey>,
    failed: FxHashMap<CubeKey, Instant>,
    planned: Option<(SectionPos, f64, LoadedBox, Instant)>,
    camera: [f64; 3],
}

impl CubeLod {
    /// `air` is the registry's air state; voxels of any other state are drawn.
    pub fn new(
        sampler: Box<dyn CubeSampler>,
        context: Arc<WorkerContext>,
        air: BlockStateId,
    ) -> Self {
        let wanted_threads = (horizon_threads::worker_count() / 2).clamp(2, 4);
        let mut samplers = Vec::new();
        while samplers.len() + 1 < wanted_threads
            && let Some(fork) = sampler.fork()
        {
            samplers.push(fork);
        }
        samplers.push(sampler);
        let shared = Arc::new(Shared {
            queue: Mutex::default(),
            wake: Condvar::new(),
            stop: AtomicBool::new(false),
            cache: Mutex::default(),
        });
        let (tx, results) = mpsc::channel();
        let threads = samplers
            .into_iter()
            .map(|sampler| {
                let (shared, tx, context) = (shared.clone(), tx.clone(), context.clone());
                thread::Builder::new()
                    .name("Cube far terrain".into())
                    .spawn(move || work(shared, tx, context, sampler, air))
                    .expect("spawn a cube far terrain thread")
            })
            .collect();
        Self::assemble(shared, results, threads)
    }

    fn assemble(
        shared: Arc<Shared>,
        results: mpsc::Receiver<Done>,
        threads: Vec<thread::JoinHandle<()>>,
    ) -> Self {
        Self {
            shared,
            results,
            threads,
            plan: Vec::new(),
            wanted: FxHashSet::default(),
            delivered: FxHashSet::default(),
            failed: FxHashMap::default(),
            planned: None,
            camera: [0.0; 3],
        }
    }

    /// Plans (again, when the camera crossed into another cube, the range or
    /// the loaded box changed, or a moment passed), queues what is missing
    /// nearest first, and returns the meshes finished since: only for nodes
    /// in the plan.
    pub fn update(
        &mut self,
        camera: [f64; 3],
        max_distance: f64,
        loaded: LoadedBox,
        is_loaded: impl Fn(SectionPos) -> bool,
    ) -> Vec<(CubeKey, SectionMesh)> {
        self.camera = camera;
        let now = Instant::now();
        let cube = cube_of(camera);
        let stale = match self.planned {
            Some((c, d, l, at)) => {
                c != cube || d != max_distance || l != loaded || now - at >= REPLAN
            }
            None => true,
        };
        if stale {
            self.plan = plan(camera, max_distance, loaded, &is_loaded);
            self.wanted = self.plan.iter().copied().collect();
            self.planned = Some((cube, max_distance, loaded, now));
            self.failed
                .retain(|key, at| self.wanted.contains(key) && now - *at < RETRY);
            self.enqueue();
        }
        let mut out = Vec::new();
        while let Ok(done) = self.results.try_recv() {
            if !self.wanted.contains(&done.key) {
                continue;
            }
            match done.mesh {
                Some(mesh) => {
                    self.delivered.insert(done.key);
                    out.push((done.key, mesh));
                }
                None => {
                    self.failed.insert(done.key, now);
                }
            }
        }
        out
    }

    fn enqueue(&mut self) {
        let mut queue = self.shared.queue.lock().unwrap();
        queue.jobs.clear();
        let now = Instant::now();
        for &key in &self.plan {
            let retry = self.failed.get(&key).is_none_or(|at| now - *at >= RETRY);
            if retry && !self.delivered.contains(&key) && !queue.running.contains(&key) {
                queue.jobs.push_back(key);
            }
        }
        drop(queue);
        self.shared.wake.notify_all();
    }

    /// The nodes the renderer can drop: handed out before, no longer in the
    /// plan, and with every node over their place handed out too.
    pub fn removed(&mut self) -> Vec<CubeKey> {
        // Plan nodes still missing, and their ancestors: a node holding one
        // of them still covers a place nothing else draws yet.
        let mut blocked = FxHashSet::default();
        for &key in self.plan.iter().filter(|k| !self.delivered.contains(k)) {
            let mut up = key;
            while up.level < MAX_CUBE_LEVEL && blocked.insert(up.parent()) {
                up = up.parent();
            }
        }
        let wanted = &self.wanted;
        let delivered = &self.delivered;
        let released: Vec<CubeKey> = delivered
            .iter()
            .copied()
            .filter(|&key| {
                if wanted.contains(&key) || blocked.contains(&key) {
                    return false;
                }
                // An ancestor in the plan that is not out yet covers it.
                let mut up = key;
                while up.level < MAX_CUBE_LEVEL {
                    up = up.parent();
                    if wanted.contains(&up) && !delivered.contains(&up) {
                        return false;
                    }
                }
                true
            })
            .collect();
        for key in &released {
            self.delivered.remove(key);
        }
        let camera = self.camera;
        let mut released = released;
        released.sort_unstable_by(|a, b| a.distance(camera).total_cmp(&b.distance(camera)));
        released
    }

    /// Forgets everything: nodes handed out, queued and cached.
    pub fn clear(&mut self) {
        {
            let mut queue = self.shared.queue.lock().unwrap();
            queue.jobs.clear();
        }
        self.shared.cache.lock().unwrap().nodes.clear();
        // Results of jobs running now are dropped as unwanted.
        self.plan.clear();
        self.wanted.clear();
        self.delivered.clear();
        self.failed.clear();
        self.planned = None;
        while self.results.try_recv().is_ok() {}
    }
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
        for thread in self.threads.drain(..) {
            let _ = thread.join();
        }
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
) {
    horizon_threads::behind_server();
    horizon_threads::far_terrain_current();
    let mut mesher = Mesher::default();
    let kind = classifier(&context, air);
    loop {
        let key = {
            let mut queue = shared.queue.lock().unwrap();
            loop {
                if shared.stop.load(Ordering::Relaxed) {
                    return;
                }
                if let Some(key) = queue.jobs.pop_front() {
                    queue.running.insert(key);
                    break key;
                }
                queue = shared.wake.wait(queue).unwrap();
            }
        };
        let mesh = node(&shared, sampler.as_mut(), key, air, &kind).map(|node| {
            let mut mesh = SectionMesh::default();
            mesh::mesh_cube(&mut mesher, &node, air, &context, &mut mesh);
            mesh.sort_solid();
            mesh
        });
        shared.queue.lock().unwrap().running.remove(&key);
        if tx.send(Done { key, mesh }).is_err() {
            return;
        }
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
            key(12, 0, -128, 0)
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
        let shared = Shared {
            queue: Mutex::default(),
            wake: Condvar::new(),
            stop: AtomicBool::new(false),
            cache: Mutex::default(),
        };
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
        let shared = Arc::new(Shared {
            queue: Mutex::default(),
            wake: Condvar::new(),
            stop: AtomicBool::new(false),
            cache: Mutex::default(),
        });
        let (_tx, results) = mpsc::channel();
        CubeLod::assemble(shared, results, Vec::new())
    }

    #[test]
    fn a_node_is_removed_only_once_what_replaces_it_is_in() {
        let mut lod = lod_for_test();
        let coarse = key(1, 0, 0, 0);
        let kids = coarse.children();
        lod.delivered.insert(coarse);
        // The plan now wants the eight children; none is in yet.
        lod.plan = kids.to_vec();
        lod.wanted = kids.into_iter().collect();
        assert!(lod.removed().is_empty(), "children missing: a hole");
        for &k in &kids[..7] {
            lod.delivered.insert(k);
        }
        assert!(lod.removed().is_empty(), "one child missing");
        lod.delivered.insert(kids[7]);
        assert_eq!(lod.removed(), vec![coarse]);
        assert!(lod.removed().is_empty(), "reported once");
        // The other way: children out, parent wanted.
        lod.plan = vec![coarse];
        lod.wanted = [coarse].into_iter().collect();
        assert!(lod.removed().is_empty(), "the parent is not in");
        lod.delivered.insert(coarse);
        assert_eq!(lod.removed().len(), 8);
        // Out of range, nothing replaces it.
        lod.plan.clear();
        lod.wanted.clear();
        assert_eq!(lod.removed(), vec![coarse]);
    }

    #[test]
    fn the_plan_is_dropped_by_clear() {
        let mut lod = lod_for_test();
        lod.plan = vec![key(0, 0, 0, 0)];
        lod.wanted = lod.plan.iter().copied().collect();
        lod.delivered.insert(key(0, 0, 0, 0));
        lod.clear();
        assert!(lod.plan.is_empty() && lod.delivered.is_empty());
        assert!(lod.removed().is_empty());
    }
}
