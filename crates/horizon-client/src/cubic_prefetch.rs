//! The prefetch ring of a cubic level (docs/cubic-v2-plan.md, "Residency"):
//! for every loaded column, the cubes of the next [`RING`] layers above and
//! below the window, kept ready in memory so that a window sliding by a layer
//! never waits for the disk or the generator.
//!
//! One background thread loads them, nearest columns first, through
//! [`CubicLevel::make_cube`] (the level's single place that turns a cube
//! position into a cube: saved record, else the generator's, behind
//! [`CubicLevel::ready_cube`]). The game thread retargets the ring whenever
//! the window or the loaded columns change ([`Ring::retarget`]) and takes
//! cubes out when the window slides over them ([`Ring::take`]). The level
//! drops a cube from the ring when the store writes it
//! ([`Ring::invalidate`]), so a ready cube is never older than its record.
//!
//! What it holds is bounded: at most [`MAX_READY`] cubes (nearest columns
//! first), and the ring's target is the only thing it loads.
//!
//! A cube inside vanilla's range exists only once its column's vanilla
//! generation ran ([`CubicLevel::has_vanilla`]). The ring skips those of a
//! column without it: the column goes the normal way, through the streaming's
//! generator, when the window reaches the range.
use crate::chunk::ColumnLayout;
use crate::cubic::{CubicLevel, LoadedCube, RING};
use horizon_core::position::SectionPos;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, Weak};

/// The most cubes the ring holds ready.
pub const MAX_READY: usize = 12_288;

/// The cubes of the ring around `window` for column (`x`, `z`): the layers
/// just outside it, nearest first, below then above.
pub fn ring_cubes(window: ColumnLayout, x: i32, z: i32) -> impl Iterator<Item = SectionPos> {
    let low = window.min_y() >> 4;
    let high = low + window.sections() as i32;
    (1..=RING)
        .flat_map(move |k| [low - k, high - 1 + k])
        .map(move |cy| SectionPos::new(x, cy, z))
}

/// One shard's window and loaded columns (nearest the players first).
#[derive(PartialEq)]
struct Target {
    window: ColumnLayout,
    columns: Vec<(i32, i32)>,
}

#[derive(Default)]
struct State {
    /// What the ring is for now, in the order it is loaded.
    wanted: Vec<SectionPos>,
    /// The windows and columns `wanted` was made from, by shard.
    targets: BTreeMap<u16, Target>,
    /// Where in `wanted` the cubes before are ready or skipped.
    cursor: usize,
    ready: HashMap<SectionPos, LoadedCube>,
    /// Cubes of columns without vanilla generation, until the next target.
    skipped: HashSet<SectionPos>,
    /// The cube the thread is loading, and whether it was written meanwhile.
    in_flight: Option<SectionPos>,
    stale: bool,
}

struct Shared {
    state: Mutex<State>,
    wake: Condvar,
    stop: AtomicBool,
}

/// A level's prefetch ring.
pub struct Ring {
    shared: Arc<Shared>,
}

impl Default for Ring {
    fn default() -> Self {
        Self::new()
    }
}

impl Ring {
    /// A ring with no thread: nothing is loaded until [`Self::start`].
    pub fn new() -> Self {
        Self {
            shared: Arc::new(Shared {
                state: Mutex::default(),
                wake: Condvar::new(),
                stop: AtomicBool::new(false),
            }),
        }
    }

    /// Starts the thread that fills the ring from `level`. It ends with the
    /// ring, or when the level is gone.
    pub fn start(&self, level: Weak<CubicLevel>) {
        let shared = self.shared.clone();
        let spawned = std::thread::Builder::new()
            .name("Cube ring".into())
            .spawn(move || run(&shared, &level));
        if let Err(e) = spawned {
            log::warn!("Horizon: no cube ring thread ({e}); window moves load cubes themselves");
        }
    }

    /// Aims the ring at the layers around `window` of `columns` (the level's
    /// loaded columns), nearest to `centres` (the players' columns) first:
    /// the target of the primary shard.
    pub fn retarget(
        &self,
        window: ColumnLayout,
        centres: &[(i32, i32)],
        columns: impl IntoIterator<Item = (i32, i32)>,
    ) {
        self.retarget_shard(0, window, centres, columns);
    }

    /// `retarget` for shard `shard`: the ring serves every shard's window,
    /// each with its own columns. Nothing changes when the shard's window and
    /// columns are the same as the last time; cubes no longer wanted are
    /// dropped.
    pub fn retarget_shard(
        &self,
        shard: u16,
        window: ColumnLayout,
        centres: &[(i32, i32)],
        columns: impl IntoIterator<Item = (i32, i32)>,
    ) {
        let mut columns: Vec<(i32, i32)> = columns.into_iter().collect();
        let distance = |&(x, z): &(i32, i32)| -> i64 {
            centres
                .iter()
                .map(|&(cx, cz)| {
                    let (dx, dz) = (i64::from(x - cx), i64::from(z - cz));
                    dx * dx + dz * dz
                })
                .min()
                .unwrap_or(0)
        };
        columns.sort_unstable_by_key(|c| (distance(c), *c));
        let target = Target { window, columns };
        let mut st = self.shared.state.lock().unwrap();
        if st.targets.get(&shard) == Some(&target) {
            return;
        }
        st.targets.insert(shard, target);
        self.rebuild(&mut st);
        drop(st);
        self.shared.wake.notify_all();
    }

    /// Drops shard `shard`'s target (it closed): its cubes are no longer wanted.
    pub fn forget(&self, shard: u16) {
        let mut st = self.shared.state.lock().unwrap();
        if st.targets.remove(&shard).is_some() {
            self.rebuild(&mut st);
            drop(st);
            self.shared.wake.notify_all();
        }
    }

    /// What is wanted now, from every shard's target.
    fn rebuild(&self, st: &mut State) {
        let mut wanted = Vec::new();
        // Layer by layer outward, so the nearest layers of every column come
        // before the farther ones.
        for k in 0..RING * 2 {
            for target in st.targets.values() {
                for &(x, z) in &target.columns {
                    if let Some(pos) = ring_cubes(target.window, x, z).nth(k as usize) {
                        wanted.push(pos);
                    }
                }
            }
        }
        wanted.truncate(MAX_READY);
        let keep: HashSet<SectionPos> = wanted.iter().copied().collect();
        st.ready.retain(|pos, _| keep.contains(pos));
        st.skipped.clear();
        st.wanted = wanted;
        st.cursor = 0;
    }

    /// Whether cube `pos` is ready.
    pub fn has(&self, pos: SectionPos) -> bool {
        self.shared.state.lock().unwrap().ready.contains_key(&pos)
    }

    /// Takes cube `pos` out of the ring, when it is ready.
    pub fn take(&self, pos: SectionPos) -> Option<LoadedCube> {
        let mut st = self.shared.state.lock().unwrap();
        let cube = st.ready.remove(&pos);
        if cube.is_some() {
            st.cursor = 0;
        }
        cube
    }

    /// Drops cube `pos` (the store wrote it): the ring loads it again if it
    /// still wants it.
    pub fn invalidate(&self, pos: SectionPos) {
        let mut st = self.shared.state.lock().unwrap();
        let had = st.ready.remove(&pos).is_some();
        if st.in_flight == Some(pos) {
            st.stale = true;
        }
        if had {
            st.cursor = 0;
            drop(st);
            self.shared.wake.notify_all();
        }
    }

    /// The cubes ready now.
    pub fn len(&self) -> usize {
        self.shared.state.lock().unwrap().ready.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Drop for Ring {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::SeqCst);
        self.shared.wake.notify_all();
    }
}

/// The next cube to load: the first wanted one not ready or skipped.
fn next_missing(st: &mut State) -> Option<SectionPos> {
    while st.cursor < st.wanted.len() {
        let pos = st.wanted[st.cursor];
        if st.ready.contains_key(&pos) || st.skipped.contains(&pos) {
            st.cursor += 1;
        } else if st.ready.len() >= MAX_READY {
            return None;
        } else {
            return Some(pos);
        }
    }
    None
}

fn run(shared: &Shared, level: &Weak<CubicLevel>) {
    loop {
        let pos = {
            let mut st = shared.state.lock().unwrap();
            loop {
                if shared.stop.load(Ordering::SeqCst) {
                    return;
                }
                if let Some(pos) = next_missing(&mut st) {
                    st.in_flight = Some(pos);
                    st.stale = false;
                    break pos;
                }
                // A level that is gone ends the thread at the next wake-up.
                let (guard, _) = shared
                    .wake
                    .wait_timeout(st, std::time::Duration::from_secs(5))
                    .unwrap();
                st = guard;
                if level.strong_count() == 0 {
                    return;
                }
            }
        };
        let Some(level) = level.upgrade() else {
            return;
        };
        let cube = if level.can_make_cube(pos) {
            Some(level.make_cube(pos))
        } else {
            None
        };
        drop(level);
        let mut st = shared.state.lock().unwrap();
        st.in_flight = None;
        match cube {
            Some(cube) if !std::mem::take(&mut st.stale) => {
                st.ready.insert(pos, cube);
            }
            Some(_) => {}
            None => {
                st.skipped.insert(pos);
            }
        }
        st.stale = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn window(low: i32, cubes: usize) -> ColumnLayout {
        ColumnLayout::window(low * 16, cubes as i32 * 16)
    }

    #[test]
    fn the_ring_is_two_layers_each_side_nearest_first() {
        let cubes: Vec<i32> = ring_cubes(window(0, 19), 3, -4).map(|p| p.y).collect();
        assert_eq!(cubes, [-1, 19, -2, 20]);
        assert!(ring_cubes(window(0, 19), 3, -4).all(|p| (p.x, p.z) == (3, -4)));
    }

    #[test]
    fn a_target_wants_nearest_columns_first_and_drops_what_it_no_longer_wants() {
        let ring = Ring::new();
        ring.retarget(window(0, 19), &[(0, 0)], [(5, 5), (0, 0), (1, 0)]);
        {
            let st = ring.shared.state.lock().unwrap();
            assert_eq!(st.wanted.len(), 12);
            let firsts: Vec<(i32, i32)> = st.wanted[..3].iter().map(|p| (p.x, p.z)).collect();
            assert_eq!(firsts, [(0, 0), (1, 0), (5, 5)]);
            // The nearest layer of every column before the farther ones.
            assert!(st.wanted[..3].iter().all(|p| p.y == -1));
        }
        // A cube that was ready and is not wanted any more goes.
        let pos = SectionPos::new(0, -1, 0);
        ring.shared
            .state
            .lock()
            .unwrap()
            .ready
            .insert(pos, crate::cubic::LoadedCube::empty(pos, Arc::default()));
        assert!(ring.has(pos));
        ring.retarget(window(0, 19), &[(0, 0)], [(0, 0), (1, 0), (5, 5), (9, 9)]);
        assert!(ring.has(pos), "still wanted");
        ring.retarget(window(40, 19), &[(0, 0)], [(0, 0)]);
        assert!(!ring.has(pos));
        assert!(ring.is_empty());
    }

    #[test]
    fn every_shard_has_its_own_target_and_a_closed_shard_takes_its_cubes_with_it() {
        let ring = Ring::new();
        ring.retarget_shard(0, window(0, 19), &[(0, 0)], [(0, 0)]);
        ring.retarget_shard(1, window(1250, 19), &[(0, 0)], [(0, 0), (1, 0)]);
        let wanted = |ring: &Ring| ring.shared.state.lock().unwrap().wanted.clone();
        let all = wanted(&ring);
        // Four cubes for the lower shard's column, four for each of the upper's two.
        assert_eq!(all.len(), 4 + 8);
        assert!(all.contains(&SectionPos::new(0, -1, 0)));
        assert!(all.contains(&SectionPos::new(1, 1269, 0)));
        // Retargeting one shard leaves the other's cubes ready.
        let kept = SectionPos::new(0, -1, 0);
        ring.shared
            .state
            .lock()
            .unwrap()
            .ready
            .insert(kept, crate::cubic::LoadedCube::empty(kept, Arc::default()));
        ring.retarget_shard(1, window(1260, 19), &[(0, 0)], [(0, 0)]);
        assert!(ring.has(kept));
        assert_eq!(wanted(&ring).len(), 4 + 4);
        // A closed shard stops being wanted.
        ring.forget(1);
        assert_eq!(wanted(&ring).len(), 4);
        ring.forget(0);
        assert!(wanted(&ring).is_empty() && ring.is_empty());
    }

    #[test]
    fn a_written_cube_leaves_the_ring_and_a_taken_one_is_given_once() {
        let ring = Ring::new();
        ring.retarget(window(0, 19), &[(0, 0)], [(0, 0)]);
        let (a, b) = (SectionPos::new(0, -1, 0), SectionPos::new(0, 19, 0));
        {
            let mut st = ring.shared.state.lock().unwrap();
            for pos in [a, b] {
                st.ready
                    .insert(pos, crate::cubic::LoadedCube::empty(pos, Arc::default()));
            }
        }
        ring.invalidate(a);
        assert!(!ring.has(a) && ring.has(b));
        assert!(ring.take(b).is_some());
        assert!(ring.take(b).is_none());
    }

    #[test]
    fn a_cube_written_while_it_loads_is_not_kept() {
        let ring = Ring::new();
        let pos = SectionPos::new(0, -1, 0);
        {
            let mut st = ring.shared.state.lock().unwrap();
            st.in_flight = Some(pos);
        }
        ring.invalidate(pos);
        assert!(ring.shared.state.lock().unwrap().stale);
    }
}
