//! The cubes a viewer keeps in a cubic level: vanilla's horizontal view
//! (`ChunkTrackingView.isWithinDistance`, with the neighbours' buffer of two
//! columns) times a vertical range of cubes around the viewer's cube.

use super::{MAX_CUBE_Y, MIN_CUBE_Y};
use horizon_core::position::SectionPos;

/// The cubes around `center` a viewer keeps: columns within `horizontal`
/// (a view distance in chunks) and cube Ys within `vertical` of the centre's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CubeWindow {
    pub center: SectionPos,
    pub horizontal: i32,
    pub vertical: i32,
}

impl CubeWindow {
    pub const fn new(center: SectionPos, horizontal: i32, vertical: i32) -> Self {
        Self {
            center,
            horizontal,
            vertical,
        }
    }

    /// Whether column (x, z) is in view: `ChunkTrackingView.isWithinDistance`
    /// with `includeOuterChunksAdjacentToViewBorder`, as the server sends
    /// chunks to a player.
    pub fn has_column(&self, x: i32, z: i32) -> bool {
        let reach = |a: i32, b: i32| (i64::from(a) - i64::from(b)).abs().saturating_sub(2).max(0);
        let dx = reach(x, self.center.x);
        let dz = reach(z, self.center.z);
        let r = i64::from(self.horizontal.max(0));
        dx * dx + dz * dz < r * r
    }

    /// Whether the window keeps cube Y `y`.
    pub fn has_cube_y(&self, y: i32) -> bool {
        (i64::from(y) - i64::from(self.center.y)).abs() <= i64::from(self.vertical.max(0))
            && super::has_cube_y(y)
    }

    pub fn contains(&self, cube: SectionPos) -> bool {
        self.has_column(cube.x, cube.z) && self.has_cube_y(cube.y)
    }

    /// The cube Ys the window keeps, lowest first, clipped to the level's.
    fn ys(&self) -> std::ops::RangeInclusive<i32> {
        let v = self.vertical.max(0);
        self.center.y.saturating_sub(v).max(MIN_CUBE_Y)
            ..=self.center.y.saturating_add(v).min(MAX_CUBE_Y)
    }

    /// Every cube of the window, nearest the centre first (squared distance
    /// in cubes, the vertical axis counted as the horizontal ones); equally
    /// near cubes in y, then z, then x order.
    pub fn cubes(&self) -> Vec<SectionPos> {
        if self.horizontal <= 0 {
            return Vec::new();
        }
        // |dx| - 2 < r keeps columns up to r + 1 away.
        let reach = self.horizontal.saturating_add(1);
        let c = self.center;
        let mut out = Vec::new();
        for y in self.ys() {
            for z in c.z.saturating_sub(reach)..=c.z.saturating_add(reach) {
                for x in c.x.saturating_sub(reach)..=c.x.saturating_add(reach) {
                    if self.has_column(x, z) {
                        out.push(SectionPos::new(x, y, z));
                    }
                }
            }
        }
        out.sort_by_key(|p| distance_squared(*p, c));
        out
    }

    /// What moving from `old` to `self` changes: the cubes to load (in
    /// `self`, not in `old`, nearest first) and the cubes to drop (in `old`,
    /// not in `self`).
    pub fn changes(&self, old: &CubeWindow) -> (Vec<SectionPos>, Vec<SectionPos>) {
        let load = self
            .cubes()
            .into_iter()
            .filter(|p| !old.contains(*p))
            .collect();
        let drop = old
            .cubes()
            .into_iter()
            .filter(|p| !self.contains(*p))
            .collect();
        (load, drop)
    }
}

fn distance_squared(a: SectionPos, b: SectionPos) -> i64 {
    let d = |a: i32, b: i32| i64::from(a) - i64::from(b);
    let (x, y, z) = (d(a.x, b.x), d(a.y, b.y), d(a.z, b.z));
    x * x + y * y + z * z
}

#[cfg(test)]
mod tests {
    use super::super::has_cube;
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn columns_follow_the_servers_tracking_view() {
        // ChunkTrackingView at view distance 2: |d| - 2 per axis, squared sum < 4.
        let w = CubeWindow::new(SectionPos::new(0, 0, 0), 2, 0);
        assert!(w.has_column(3, 3));
        assert!(w.has_column(-3, 0) && w.has_column(0, 3));
        assert!(!w.has_column(4, 0) && !w.has_column(4, 3));
        assert!(!w.has_column(-4, -1));
        let columns = w.cubes().len();
        // A 7 x 7 square: every |d| <= 3, none at 4.
        assert_eq!(columns, 49);
    }

    #[test]
    fn cubes_are_the_window_nearest_first() {
        let w = CubeWindow::new(SectionPos::new(5, -3, -7), 3, 2);
        let cubes = w.cubes();
        assert_eq!(cubes[0], w.center);
        let set: HashSet<_> = cubes.iter().copied().collect();
        assert_eq!(set.len(), cubes.len());
        for y in -10..=4 {
            for z in -14..=0 {
                for x in -2..=12 {
                    let p = SectionPos::new(x, y, z);
                    assert_eq!(set.contains(&p), w.contains(p), "{p:?}");
                }
            }
        }
        let distances: Vec<_> = cubes
            .iter()
            .map(|p| distance_squared(*p, w.center))
            .collect();
        assert!(distances.windows(2).all(|d| d[0] <= d[1]));
        // Five layers of the same columns.
        assert_eq!(cubes.len() % 5, 0);
    }

    #[test]
    fn moving_up_one_cube_swaps_a_layer() {
        let old = CubeWindow::new(SectionPos::new(0, 10, 0), 4, 3);
        let new = CubeWindow::new(SectionPos::new(0, 11, 0), 4, 3);
        let (load, drop) = new.changes(&old);
        let layer = old.cubes().iter().filter(|p| p.y == 10).count();
        assert_eq!(load.len(), layer);
        assert_eq!(drop.len(), layer);
        assert!(load.iter().all(|p| p.y == 14));
        assert!(drop.iter().all(|p| p.y == 7));
        assert!(new.changes(&new).0.is_empty());
    }

    #[test]
    fn the_window_stops_at_the_levels_ends() {
        let top = CubeWindow::new(SectionPos::new(0, MAX_CUBE_Y, 0), 1, 4);
        assert!(top.cubes().iter().all(|p| has_cube(*p)));
        assert_eq!(top.cubes().iter().map(|p| p.y).max(), Some(MAX_CUBE_Y));
        assert_eq!(top.cubes().iter().map(|p| p.y).min(), Some(MAX_CUBE_Y - 4));
        let bottom = CubeWindow::new(SectionPos::new(0, MIN_CUBE_Y, 0), 1, 4);
        assert_eq!(bottom.cubes().iter().map(|p| p.y).min(), Some(MIN_CUBE_Y));
        assert!(!bottom.contains(SectionPos::new(0, MIN_CUBE_Y - 1, 0)));
        assert!(
            CubeWindow::new(SectionPos::new(0, 0, 0), 0, 4)
                .cubes()
                .is_empty()
        );
    }
}
