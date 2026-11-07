//! Custom Cubic's structures in the far view (docs/cubic-v2-plan.md, "Far
//! view: the owner's decisions and later work", item 4).
//!
//! The far view first draws its terrain without structures, as before. Once
//! every planned node is drawn ([`LateContent::view`] says so), a background
//! thread at idle priority looks for structure starts around the camera,
//! nearest and in front first. It asks the planner only at the chunks where
//! a set can start one ([`CubicStructures::candidates`]; sets fewer than
//! [`MIN_SPACING`] chunks apart, mineshafts and buried treasure, stay
//! underground and are left out). No chunk is generated for it.
//!
//! A start found is *sketched*. Every population area its columns cross is
//! placed by the near terrain's own code ([`CubicStructures::place`]) into a
//! scratch copy of the generator's cubes. The blocks it changes, other than
//! to air, are the structure's voxels. The far nodes over the start's box
//! are then made again with those voxels ([`FarStructures::overlay`]):
//! * Level 0 takes them exactly.
//! * A coarser voxel shows the structure's most common block when the
//!   structure fills at least 1/64 of it, whatever the terrain there (built
//!   blocks win over natural ones, as item 5 asks).
//!
//! The nodes' kept copies on disk are erased under the structure, so they
//! are rebuilt with it. Villages, temples and monuments thus fill in over
//! time.
//!
//! Height-map tiles (levels 12 and up) are left as they are. A tile column
//! spans 4,096 blocks or more, many times the largest structure, so a
//! structure changes neither a tile's heights nor its materials.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};

use horizon_core::position::{ChunkPos, SectionPos};
use horizon_world::cubic::custom::CustomCubicGenerator;
use horizon_world::cubic::custom::populate::CubeWorld;
use horizon_world::cubic::custom::structures::CubicStructures;
use horizon_world::registry::BlockStateId;
use horizon_world::worldgen::structure::start::StructureStart;

use crate::lod::cube::{CubeKey, CubeNode, LateContent, VOLUME, Voxel};
use crate::lod::cube_cache::CubeLodCache;

/// Placement sectors this many chunks apart or more are searched (see the
/// module docs).
pub const MIN_SPACING: i32 = 8;
/// How far around the camera starts are looked for, in blocks, at most (the
/// far distance when it is less).
const RADIUS: f64 = 4096.0;
/// The search starts again around the camera when it moved this far.
const RESEARCH: f64 = 512.0;
/// Sketched blocks kept, at most (4 bytes each): the farthest sketches go
/// first.
const MAX_BLOCKS: usize = 8 << 20;
/// Side of the index's cells, in blocks.
const CELL: i32 = 256;
/// A start whose box spans more cubes than this is sketched without the
/// terrain's heights around it (its box's own).
const MAX_CUBES: usize = 16_384;

/// One structure's voxels: the blocks its placement changes (but to air), by
/// cube, each `index << 20 | state` with the cube's `y << 8 | z << 4 | x`
/// index.
pub struct Sketch {
    min: [i32; 3],
    max: [i32; 3],
    cubes: Vec<(SectionPos, Box<[u32]>)>,
}

impl Sketch {
    fn blocks(&self) -> usize {
        self.cubes.iter().map(|(_, b)| b.len()).sum()
    }

    fn touches(&self, min: [i32; 3], max: [i32; 3]) -> bool {
        (0..3).all(|i| self.min[i] <= max[i] && self.max[i] >= min[i])
    }
}

#[derive(Default)]
struct Index {
    sketches: HashMap<u64, Arc<Sketch>>,
    cells: HashMap<(i32, i32), Vec<u64>>,
    blocks: usize,
    next: u64,
}

impl Index {
    fn cells_of(min: [i32; 3], max: [i32; 3]) -> impl Iterator<Item = (i32, i32)> {
        let (x0, x1) = (min[0].div_euclid(CELL), max[0].div_euclid(CELL));
        let (z0, z1) = (min[2].div_euclid(CELL), max[2].div_euclid(CELL));
        (x0..=x1).flat_map(move |x| (z0..=z1).map(move |z| (x, z)))
    }

    fn insert(&mut self, sketch: Sketch) {
        let id = self.next;
        self.next += 1;
        for cell in Self::cells_of(sketch.min, sketch.max) {
            self.cells.entry(cell).or_default().push(id);
        }
        self.blocks += sketch.blocks();
        self.sketches.insert(id, Arc::new(sketch));
    }

    fn remove(&mut self, id: u64) {
        let Some(sketch) = self.sketches.remove(&id) else {
            return;
        };
        self.blocks -= sketch.blocks();
        for cell in Self::cells_of(sketch.min, sketch.max) {
            if let Some(ids) = self.cells.get_mut(&cell) {
                ids.retain(|&i| i != id);
                if ids.is_empty() {
                    self.cells.remove(&cell);
                }
            }
        }
    }

    fn touching(&self, min: [i32; 3], max: [i32; 3]) -> Vec<Arc<Sketch>> {
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        for cell in Self::cells_of(min, max) {
            for id in self.cells.get(&cell).into_iter().flatten() {
                if seen.insert(*id) {
                    let sketch = &self.sketches[id];
                    if sketch.touches(min, max) {
                        out.push(sketch.clone());
                    }
                }
            }
        }
        out
    }
}

#[derive(Clone, Copy, Default)]
struct View {
    camera: [f64; 3],
    forward: [f64; 3],
    radius: f64,
    complete: bool,
    known: bool,
}

struct Inner {
    structures: Arc<CubicStructures>,
    generator: Arc<CustomCubicGenerator>,
    cache: Option<Arc<CubeLodCache>>,
    air: u32,
    view: Mutex<View>,
    wake: Condvar,
    stop: AtomicBool,
    index: RwLock<Index>,
    /// Sketches in the index (read without the lock on every node).
    count: AtomicUsize,
    refreshes: Mutex<Vec<([i32; 3], [i32; 3])>>,
    started: AtomicBool,
}

/// A cubic level's structures for its far view: the background search, the
/// sketches it made, and their voxels for the far nodes. Shared by the far
/// samplers; the search stops when the last one goes.
pub struct FarStructures {
    inner: Arc<Inner>,
}

impl Drop for FarStructures {
    fn drop(&mut self) {
        self.inner.stop.store(true, Ordering::Relaxed);
        self.inner.wake.notify_all();
    }
}

impl FarStructures {
    pub fn new(
        structures: Arc<CubicStructures>,
        generator: Arc<CustomCubicGenerator>,
        cache: Option<Arc<CubeLodCache>>,
        air: u32,
    ) -> Arc<Self> {
        Arc::new(Self {
            inner: Arc::new(Inner {
                structures,
                generator,
                cache,
                air,
                view: Mutex::default(),
                wake: Condvar::new(),
                stop: AtomicBool::new(false),
                index: RwLock::default(),
                count: AtomicUsize::new(0),
                refreshes: Mutex::default(),
                started: AtomicBool::new(false),
            }),
        })
    }

    /// Whether a sketch has blocks in the node `key` (cheap when there is
    /// none).
    pub fn touches(&self, key: CubeKey) -> bool {
        if self.inner.count.load(Ordering::Relaxed) == 0 {
            return false;
        }
        let (min, max) = key_box(key);
        let sketches = self.inner.index.read().unwrap().touching(min, max);
        sketches.iter().any(|s| {
            s.cubes.iter().any(|(p, _)| {
                let o = [p.x * 16, p.y * 16, p.z * 16];
                (0..3).all(|i| o[i] <= max[i] && o[i] + 15 >= min[i])
            })
        })
    }

    /// `node` with the voxels of the sketches over it (the module docs'
    /// rules), and whether they changed it.
    pub fn overlay(&self, node: CubeNode) -> (CubeNode, bool) {
        if self.inner.count.load(Ordering::Relaxed) == 0 {
            return (node, false);
        }
        let key = node.key;
        let (min, max) = key_box(key);
        let sketches = self.inner.index.read().unwrap().touching(min, max);
        if sketches.is_empty() {
            return (node, false);
        }
        overlay(node, &sketches, self.inner.air)
    }

    /// Sketches `start` now (for tests and probes): its voxels go in the
    /// index as the search's would.
    pub fn sketch_now(&self, start: &StructureStart) -> bool {
        match sketch(&self.inner, start) {
            Some(s) => {
                self.inner.index.write().unwrap().insert(s);
                self.inner.count.fetch_add(1, Ordering::Relaxed);
                true
            }
            None => false,
        }
    }

    /// The sketched blocks in `min..=max` (inclusive), for tests and probes.
    pub fn sketched(&self, min: [i32; 3], max: [i32; 3]) -> Vec<([i32; 3], u32)> {
        let index = self.inner.index.read().unwrap();
        let mut out = Vec::new();
        for sketch in index.touching(min, max) {
            for (pos, blocks) in &sketch.cubes {
                for &b in blocks.iter() {
                    let i = (b >> 20) as i32;
                    let p = [
                        pos.x * 16 + (i & 15),
                        pos.y * 16 + (i >> 8),
                        pos.z * 16 + (i >> 4 & 15),
                    ];
                    if (0..3).all(|a| (min[a]..=max[a]).contains(&p[a])) {
                        out.push((p, b & 0xF_FFFF));
                    }
                }
            }
        }
        out
    }

    fn start_search(&self) {
        if self.inner.started.swap(true, Ordering::Relaxed) {
            return;
        }
        let inner = self.inner.clone();
        if let Err(e) = std::thread::Builder::new()
            .name("Cube far structures".into())
            .spawn(move || {
                horizon_threads::idle_current();
                search(&inner);
            })
        {
            log::warn!("Horizon: no far structures: {e}");
        }
    }
}

impl LateContent for FarStructures {
    fn view(&self, camera: [f64; 3], forward: [f64; 3], max_distance: f64, complete: bool) {
        self.start_search();
        let mut view = self.inner.view.lock().unwrap();
        let woke = complete && !view.complete;
        *view = View {
            camera,
            forward,
            radius: max_distance.min(RADIUS),
            complete,
            known: true,
        };
        drop(view);
        if woke {
            self.inner.wake.notify_all();
        }
    }

    fn take_refreshes(&self) -> Vec<([i32; 3], [i32; 3])> {
        std::mem::take(&mut *self.inner.refreshes.lock().unwrap())
    }
}

/// A node's box of blocks (inclusive).
fn key_box(key: CubeKey) -> ([i32; 3], [i32; 3]) {
    let o = key.origin();
    let size = key.size();
    (o, o.map(|v| v.saturating_add(size - 1)))
}

/// The search (see the module docs), until the samplers are gone.
fn search(inner: &Inner) {
    let mut visited: HashSet<(i32, i32)> = HashSet::new();
    let mut seen: HashSet<(usize, i32, i32)> = HashSet::new();
    let mut todo: Vec<ChunkPos> = Vec::new();
    let mut around: Option<[f64; 3]> = None;
    loop {
        // Wait for a complete far view.
        let view = {
            let mut view = inner.view.lock().unwrap();
            while !inner.stop.load(Ordering::Relaxed) && !(view.known && view.complete) {
                view = inner
                    .wake
                    .wait_timeout(view, std::time::Duration::from_millis(500))
                    .unwrap()
                    .0;
            }
            *view
        };
        if inner.stop.load(Ordering::Relaxed) {
            return;
        }
        let moved = around.is_none_or(|a| {
            let (dx, dz) = (a[0] - view.camera[0], a[2] - view.camera[2]);
            (dx * dx + dz * dz).sqrt() > RESEARCH
        });
        if moved {
            around = Some(view.camera);
            todo = candidates(inner, &view, &visited);
            forget_far(inner, &view, &mut visited, &mut seen);
        }
        let Some(chunk) = todo.pop() else {
            // All found here: wait for the camera to move on.
            let view = inner.view.lock().unwrap();
            let _ = inner
                .wake
                .wait_timeout(view, std::time::Duration::from_millis(1000));
            continue;
        };
        if !visited.insert((chunk.x, chunk.z)) {
            continue;
        }
        for start in inner.structures.starts_at(chunk).iter() {
            if !seen.insert((start.structure, start.chunk.x, start.chunk.z)) {
                continue;
            }
            let Some(sketch) = sketch(inner, start) else {
                continue;
            };
            // The kept nodes under it are made again, with it.
            if let Some(cache) = &inner.cache {
                for (pos, _) in &sketch.cubes {
                    cache.invalidate(*pos);
                }
            }
            let bounds = (sketch.min, sketch.max);
            inner.index.write().unwrap().insert(sketch);
            inner.count.fetch_add(1, Ordering::Relaxed);
            inner.refreshes.lock().unwrap().push(bounds);
        }
    }
}

/// The candidate chunks around the camera not visited yet, the next one
/// last: nearest first, those in front before those behind.
fn candidates(inner: &Inner, view: &View, visited: &HashSet<(i32, i32)>) -> Vec<ChunkPos> {
    let r = view.radius.max(0.0);
    let (cx, cz) = (view.camera[0], view.camera[2]);
    let chunk = |v: f64| (v.floor() as i64).clamp(-(1 << 30), 1 << 30) as i32 >> 4;
    let min = ChunkPos::new(chunk(cx - r), chunk(cz - r));
    let max = ChunkPos::new(chunk(cx + r), chunk(cz + r));
    let (fx, fz) = (view.forward[0], view.forward[2]);
    let flen = (fx * fx + fz * fz).sqrt();
    let mut found: Vec<(f64, ChunkPos)> = (inner.structures.candidates(min, max, MIN_SPACING))
        .into_iter()
        .filter(|c| !visited.contains(&(c.x, c.z)))
        .filter_map(|c| {
            let (dx, dz) = (f64::from(c.x * 16 + 8) - cx, f64::from(c.z * 16 + 8) - cz);
            let d = (dx * dx + dz * dz).sqrt();
            if d > r {
                return None;
            }
            // 1 in front, up to 3 behind.
            let behind = if flen > 0.0 && d > 0.0 {
                2.0 - (dx * fx + dz * fz) / (d * flen)
            } else {
                1.0
            };
            Some((d * behind, c))
        })
        .collect();
    found.sort_by(|a, b| b.0.total_cmp(&a.0));
    found.into_iter().map(|(_, c)| c).collect()
}

/// Drops the sketches past twice the radius when the index is over its
/// size, farthest first (their chunks can be found again).
fn forget_far(
    inner: &Inner,
    view: &View,
    visited: &mut HashSet<(i32, i32)>,
    seen: &mut HashSet<(usize, i32, i32)>,
) {
    let mut index = inner.index.write().unwrap();
    if index.blocks <= MAX_BLOCKS {
        return;
    }
    let distance = |s: &Sketch| {
        let (x, z) = (
            f64::from(s.min[0] / 2 + s.max[0] / 2) - view.camera[0],
            f64::from(s.min[2] / 2 + s.max[2] / 2) - view.camera[2],
        );
        (x * x + z * z).sqrt()
    };
    let mut far: Vec<(f64, u64)> = (index.sketches.iter())
        .map(|(&id, s)| (distance(s), id))
        .filter(|(d, _)| *d > 2.0 * view.radius)
        .collect();
    far.sort_by(|a, b| b.0.total_cmp(&a.0));
    for (_, id) in far {
        if index.blocks <= MAX_BLOCKS / 2 {
            break;
        }
        index.remove(id);
        inner.count.fetch_sub(1, Ordering::Relaxed);
    }
    // Let the search find what it dropped when the camera comes back.
    visited.clear();
    seen.clear();
}

/// The blocks a start's placement changes (see the module docs), or `None`
/// when it places none.
fn sketch(inner: &Inner, start: &StructureStart) -> Option<Sketch> {
    let b = start.bounding_box;
    // Pieces that settle on the ground leave their box's heights: the
    // terrain's around the box count too.
    let (mut y0, mut y1) = (b.min[1], b.max[1]);
    let span = |lo: i32, hi: i32| i64::from(((hi - 8) >> 4) - ((lo - 8) >> 4) + 1);
    let columns = span(b.min[0], b.max[0]) * span(b.min[2], b.max[2]);
    let mid = [(b.min[0] + b.max[0]) / 2, (b.min[2] + b.max[2]) / 2];
    for (x, z) in [
        (b.min[0], b.min[2]),
        (b.max[0], b.min[2]),
        (b.min[0], b.max[2]),
        (b.max[0], b.max[2]),
        (mid[0], mid[1]),
    ] {
        let top = inner.structures.column_top(x, z);
        if top > horizon_world::cubic::custom::NO_TOP
            && columns * span(y0.min(top - 16), y1.max(top + 32)) <= MAX_CUBES as i64
        {
            y0 = y0.min(top - 16);
            y1 = y1.max(top + 32);
        }
    }
    let mut terrain: HashMap<SectionPos, Arc<[u32; 4096]>> = HashMap::new();
    let mut changed: HashMap<SectionPos, Vec<u32>> = HashMap::new();
    for cy in (y0 - 8) >> 4..=(y1 - 8) >> 4 {
        for cz in (b.min[2] - 8) >> 4..=(b.max[2] - 8) >> 4 {
            for cx in (b.min[0] - 8) >> 4..=(b.max[0] - 8) >> 4 {
                let cube = SectionPos::new(cx, cy, cz);
                let mut world = Scratch::new(inner, cube, &mut terrain);
                inner.structures.place(cube, &mut world);
                for (p, state) in world.writes {
                    let inside = (b.min[0]..=b.max[0]).contains(&p[0])
                        && (b.min[2]..=b.max[2]).contains(&p[2]);
                    let pos = SectionPos::new(p[0] >> 4, p[1] >> 4, p[2] >> 4);
                    let i = ((p[1] & 15) << 8 | (p[2] & 15) << 4 | (p[0] & 15)) as u32;
                    if inside
                        && state != inner.air
                        && state != 0
                        && terrain[&pos][i as usize] != state
                    {
                        changed.entry(pos).or_default().push(i << 20 | state);
                    }
                }
            }
        }
        // Cubes below the ones still to place are done with.
        terrain.retain(|p, _| p.y >= cy);
    }
    if changed.is_empty() {
        return None;
    }
    let mut min = [i32::MAX; 3];
    let mut max = [i32::MIN; 3];
    let cubes = changed
        .into_iter()
        .map(|(pos, mut blocks)| {
            // The last write of a block wins.
            blocks.reverse();
            let mut once = HashSet::new();
            blocks.retain(|b| once.insert(b >> 20));
            for (i, v) in [pos.x, pos.y, pos.z].into_iter().enumerate() {
                min[i] = min[i].min(v * 16);
                max[i] = max[i].max(v * 16 + 15);
            }
            (pos, blocks.into_boxed_slice())
        })
        .collect();
    Some(Sketch { min, max, cubes })
}

/// A population's 2 x 2 x 2 cubes from the generator (terrain and carvers),
/// for a placement whose writes are kept.
struct Scratch {
    origin: [i32; 3],
    cubes: [Box<[u32; 4096]>; 8],
    writes: Vec<([i32; 3], u32)>,
}

impl Scratch {
    fn new(
        inner: &Inner,
        cube: SectionPos,
        terrain: &mut HashMap<SectionPos, Arc<[u32; 4096]>>,
    ) -> Self {
        let cubes = std::array::from_fn(|d| {
            let pos = SectionPos::new(
                cube.x + (d as i32 & 1),
                cube.y + (d as i32 >> 1 & 1),
                cube.z + (d as i32 >> 2 & 1),
            );
            let blocks = terrain.entry(pos).or_insert_with(|| {
                let mut out = Box::new([0u32; 4096]);
                inner.generator.generate(pos, &mut out);
                Arc::from(out)
            });
            Box::new(**blocks)
        });
        Self {
            origin: [cube.x * 16, cube.y * 16, cube.z * 16],
            cubes,
            writes: Vec::new(),
        }
    }

    fn slot(&self, p: [i32; 3]) -> Option<(usize, usize)> {
        let l = [0, 1, 2].map(|i| p[i].wrapping_sub(self.origin[i]));
        if l.iter().any(|v| !(0..32).contains(v)) {
            return None;
        }
        let d = (l[0] >> 4 | (l[1] >> 4) << 1 | (l[2] >> 4) << 2) as usize;
        Some((
            d,
            ((l[1] & 15) << 8 | (l[2] & 15) << 4 | (l[0] & 15)) as usize,
        ))
    }
}

impl CubeWorld for Scratch {
    fn block(&self, p: [i32; 3]) -> u32 {
        self.slot(p).map_or(0, |(d, i)| self.cubes[d][i])
    }
    fn set_block(&mut self, p: [i32; 3], state: u32) {
        if let Some((d, i)) = self.slot(p) {
            self.cubes[d][i] = state;
            self.writes.push((p, state));
        }
    }
}

/// `node` with `sketches`' voxels (the module docs' rules), and whether they
/// changed it.
fn overlay(node: CubeNode, sketches: &[Arc<Sketch>], air: u32) -> (CubeNode, bool) {
    let key = node.key;
    let (min, max) = key_box(key);
    let shift = u32::from(key.level);
    // Per voxel: (state, count) of the structure's blocks in it.
    let mut counts: HashMap<usize, Vec<(u32, u32)>> = HashMap::new();
    for sketch in sketches {
        for (pos, blocks) in &sketch.cubes {
            let o = [pos.x * 16, pos.y * 16, pos.z * 16];
            if (0..3).any(|i| o[i] > max[i] || o[i] + 15 < min[i]) {
                continue;
            }
            for &b in blocks.iter() {
                let i = (b >> 20) as i32;
                let p = [o[0] + (i & 15), o[1] + (i >> 8), o[2] + (i >> 4 & 15)];
                if (0..3).any(|a| p[a] < min[a] || p[a] > max[a]) {
                    continue;
                }
                let v = [0, 1, 2].map(|a| ((p[a] - min[a]) >> shift) as usize);
                let slot = counts.entry(CubeNode::index(v[0], v[1], v[2])).or_default();
                let state = b & 0xF_FFFF;
                match slot.iter_mut().find(|(s, _)| *s == state) {
                    Some(e) => e.1 += 1,
                    None => slot.push((state, 1)),
                }
            }
        }
    }
    if counts.is_empty() {
        return (node, false);
    }
    // A coarse voxel shows the structure from 1/64 of its volume.
    let volume = 1u64 << (3 * shift);
    let need = (volume / 64).max(1);
    let mut dense: Box<[Voxel; VOLUME]> = Box::new(std::array::from_fn(|index| node.at(index)));
    let mut changed = false;
    for (index, states) in counts {
        let total: u64 = states.iter().map(|s| u64::from(s.1)).sum();
        if total < need {
            continue;
        }
        let Some(&(state, _)) = states.iter().max_by_key(|s| s.1) else {
            continue;
        };
        if state != air {
            dense[index].state = BlockStateId(state);
            changed = true;
        }
    }
    if !changed {
        return (node, false);
    }
    let covered = node.covered_mask().cloned().map(Box::new);
    (CubeNode::from_dense(key, dense).with_covered(covered), true)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sketch_at(min: [i32; 3], blocks: &[([i32; 3], u32)]) -> Arc<Sketch> {
        let mut cubes: HashMap<SectionPos, Vec<u32>> = HashMap::new();
        for &(p, state) in blocks {
            let pos = SectionPos::new(p[0] >> 4, p[1] >> 4, p[2] >> 4);
            let i = ((p[1] & 15) << 8 | (p[2] & 15) << 4 | (p[0] & 15)) as u32;
            cubes.entry(pos).or_default().push(i << 20 | state);
        }
        Arc::new(Sketch {
            min,
            max: min.map(|v| v + 15),
            cubes: (cubes.into_iter())
                .map(|(p, b)| (p, b.into_boxed_slice()))
                .collect(),
        })
    }

    fn stone_node(key: CubeKey) -> CubeNode {
        CubeNode::uniform(
            key,
            Voxel {
                state: BlockStateId(1),
                biome: 0,
            },
        )
    }

    /// Level 0 takes a structure's blocks exactly; a coarse voxel shows the
    /// structure's most common block from 1/64 of its volume, and not below.
    #[test]
    fn structures_show_in_far_voxels_where_they_fill_enough_of_one() {
        let key = CubeKey {
            level: 0,
            x: 0,
            y: 0,
            z: 0,
        };
        let sketch = sketch_at([0; 3], &[([3, 4, 5], 7)]);
        let (node, _) = overlay(stone_node(key), &[sketch], 0);
        assert_eq!(node.get(3, 4, 5).state, BlockStateId(7));
        assert_eq!(node.get(3, 4, 6).state, BlockStateId(1));
        // Level 2: a voxel is 64 blocks, so one block fills enough of it;
        // level 4: 4,096, so 64 are needed.
        let coarse = CubeKey {
            level: 4,
            x: 0,
            y: 0,
            z: 0,
        };
        let few: Vec<([i32; 3], u32)> = (0..63).map(|i| ([i % 16, i / 16, 0], 9)).collect();
        let (node, changed) = overlay(stone_node(coarse), &[sketch_at([0; 3], &few)], 0);
        assert!(!changed);
        assert_eq!(
            node.get(0, 0, 0).state,
            BlockStateId(1),
            "63 blocks are too few"
        );
        let mut enough = few.clone();
        enough.push(([15, 3, 0], 9));
        enough.push(([1, 1, 1], 8));
        let (node, changed) = overlay(stone_node(coarse), &[sketch_at([0; 3], &enough)], 0);
        assert!(changed);
        assert_eq!(
            node.get(0, 0, 0).state,
            BlockStateId(9),
            "the most common block"
        );
        assert_eq!(node.get(1, 0, 0).state, BlockStateId(1));
    }

    #[test]
    fn the_index_finds_sketches_by_their_box() {
        let mut index = Index::default();
        index.insert(Sketch {
            min: [500, 60, -20],
            max: [520, 80, 10],
            cubes: vec![(
                SectionPos::new(31, 4, -1),
                vec![5 << 20 | 3].into_boxed_slice(),
            )],
        });
        assert_eq!(index.touching([510, 0, 0], [511, 100, 1]).len(), 1);
        assert!(index.touching([0, 0, 0], [255, 255, 255]).is_empty());
        assert!(index.touching([510, 90, 0], [511, 100, 1]).is_empty());
        index.remove(0);
        assert!(index.touching([510, 0, 0], [511, 100, 1]).is_empty());
        assert_eq!(index.blocks, 0);
    }
}
