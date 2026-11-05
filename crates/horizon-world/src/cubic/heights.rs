//! The highest block of each column of a cubic level across its cubes
//! (Cubic Chunks' `SurfaceTracker`). Sky light, weather, spawning and the
//! far terrain ask for a column's top: a bounded level reads it from its
//! heightmap, a cubic one finds it among the cubes it knows.
//!
//! A cube contributes its [`CubeTops`]: for each of its 256 columns, the
//! local y of its highest block that counts (a heightmap type's predicate).
//! [`ColumnTops`] keeps those of one column's cubes by cube Y. Cubes that are
//! not known (not loaded, never generated) contribute nothing: for sky light
//! that is "open above", as in Cubic Chunks.
//!
//! The queries are exact at cube boundaries: a cube's tops say where its
//! highest counted block is, not what lies under it, so questions about a
//! height inside one cube read that cube's blocks.

use std::collections::BTreeMap;

/// For each column `(x, z)` of one cube, the local y (0..16) of its highest
/// block that counts, if any.
#[derive(Clone, PartialEq, Eq)]
pub struct CubeTops([i8; 256]);

impl std::fmt::Debug for CubeTops {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let columns = self.0.iter().filter(|&&t| t != Self::NONE).count();
        write!(f, "CubeTops({columns} columns)")
    }
}

impl CubeTops {
    const NONE: i8 = -1;
    /// A cube with no block that counts.
    pub const EMPTY: Self = Self([Self::NONE; 256]);

    /// The tops of a cube's 4096 blocks in section index order
    /// (`y << 8 | z << 4 | x`), counting the blocks `counts` accepts.
    pub fn from_blocks(blocks: &[u32; 4096], counts: impl Fn(u32) -> bool) -> Self {
        let mut tops = Self::EMPTY;
        for (column, top) in tops.0.iter_mut().enumerate() {
            if let Some(y) = (0..16usize)
                .rev()
                .find(|&y| counts(blocks[y << 8 | column]))
            {
                *top = y as i8;
            }
        }
        tops
    }

    /// The local y of column (x, z)'s top in this cube (x and z taken
    /// modulo 16).
    pub fn get(&self, x: i32, z: i32) -> Option<i32> {
        let top = self.0[column(x, z)];
        (top != Self::NONE).then_some(i32::from(top))
    }

    /// Sets column (x, z)'s top (`None`: no block counts; y taken modulo 16).
    pub fn set(&mut self, x: i32, z: i32, top: Option<i32>) {
        self.0[column(x, z)] = top.map_or(Self::NONE, |y| (y & 15) as i8);
    }

    pub fn is_empty(&self) -> bool {
        self.0.iter().all(|&t| t == Self::NONE)
    }
}

fn column(x: i32, z: i32) -> usize {
    ((z & 15) << 4 | (x & 15)) as usize
}

/// The tops of one column of cubes (16 x 16 block columns), by cube Y.
#[derive(Clone, Debug, Default)]
pub struct ColumnTops {
    cubes: BTreeMap<i32, CubeTops>,
}

impl ColumnTops {
    /// Records cube `cube_y`'s tops, replacing what it had.
    pub fn set(&mut self, cube_y: i32, tops: CubeTops) {
        if tops.is_empty() {
            self.cubes.remove(&cube_y);
        } else {
            self.cubes.insert(cube_y, tops);
        }
    }

    /// Forgets cube `cube_y` (unloaded, or its tops not known any more).
    pub fn remove(&mut self, cube_y: i32) {
        self.cubes.remove(&cube_y);
    }

    /// The highest known block Y of block column (x, z) (x and z taken
    /// modulo 16).
    pub fn top(&self, x: i32, z: i32) -> Option<i32> {
        Self::highest(self.cubes.iter().rev(), x, z)
    }

    /// The highest known block Y of block column (x, z) in the cubes below
    /// cube `cube_y`.
    pub fn top_below(&self, x: i32, z: i32, cube_y: i32) -> Option<i32> {
        Self::highest(self.cubes.range(..cube_y).rev(), x, z)
    }

    /// Whether a known cube above cube `cube_y` has a block that counts in
    /// block column (x, z). False means open sky over the cube, unknown
    /// cubes counted as open (Cubic Chunks' rule).
    pub fn covered_above(&self, x: i32, z: i32, cube_y: i32) -> bool {
        cube_y < i32::MAX
            && self
                .cubes
                .range(cube_y + 1..)
                .any(|(_, tops)| tops.get(x, z).is_some())
    }

    /// Whether no cube of this column has a block that counts.
    pub fn is_empty(&self) -> bool {
        self.cubes.is_empty()
    }

    fn highest<'a>(
        mut cubes: impl Iterator<Item = (&'a i32, &'a CubeTops)>,
        x: i32,
        z: i32,
    ) -> Option<i32> {
        cubes.find_map(|(&cy, tops)| tops.get(x, z).map(|local| cy.wrapping_shl(4) + local))
    }
}

#[cfg(test)]
mod tests {
    use super::super::{MAX_CUBE_Y, MAX_Y, MIN_CUBE_Y, MIN_Y};
    use super::*;

    fn blocks(solid: &[(i32, i32, i32)]) -> [u32; 4096] {
        let mut out = [0; 4096];
        for &(x, y, z) in solid {
            out[(y << 8 | z << 4 | x) as usize] = 1;
        }
        out
    }

    #[test]
    fn a_cubes_tops_are_each_columns_highest_counted_block() {
        let tops = CubeTops::from_blocks(
            &blocks(&[(0, 0, 0), (0, 9, 0), (15, 15, 15), (3, 4, 5)]),
            |b| b != 0,
        );
        assert_eq!(tops.get(0, 0), Some(9));
        assert_eq!(tops.get(15, 15), Some(15));
        assert_eq!(tops.get(3, 5), Some(4));
        assert_eq!(tops.get(5, 3), None);
        // Coordinates are taken within the cube.
        assert_eq!(tops.get(-16, 16), Some(9));
        assert!(CubeTops::from_blocks(&blocks(&[]), |b| b != 0).is_empty());
        assert!(!tops.is_empty());
    }

    #[test]
    fn a_columns_top_is_its_highest_known_cube() {
        let mut column = ColumnTops::default();
        let mut low = CubeTops::EMPTY;
        low.set(2, 3, Some(7));
        let mut high = CubeTops::EMPTY;
        high.set(2, 3, Some(1));
        column.set(-5, low);
        column.set(40_000, high);
        assert_eq!(column.top(2, 3), Some(40_000 * 16 + 1));
        assert_eq!(column.top(3, 2), None);
        // Below the high cube, the low one answers.
        assert_eq!(column.top_below(2, 3, 40_000), Some(-5 * 16 + 7));
        assert_eq!(column.top_below(2, 3, -5), None);
        assert!(column.covered_above(2, 3, -5));
        assert!(!column.covered_above(2, 3, 40_000));
        assert!(!column.covered_above(3, 2, -100));
        // An unloaded cube no longer counts; an empty one is not kept.
        column.remove(40_000);
        assert_eq!(column.top(2, 3), Some(-5 * 16 + 7));
        assert!(!column.covered_above(2, 3, -5));
        column.set(-5, CubeTops::EMPTY);
        assert!(column.is_empty());
    }

    #[test]
    fn tops_at_the_levels_ends_keep_their_sign() {
        let mut column = ColumnTops::default();
        let mut tops = CubeTops::EMPTY;
        tops.set(0, 0, Some(15));
        column.set(MIN_CUBE_Y, tops.clone());
        assert_eq!(column.top(0, 0), Some(MIN_Y + 15));
        column.set(MAX_CUBE_Y, tops);
        assert_eq!(column.top(0, 0), Some(MAX_Y));
        assert_eq!(column.top_below(0, 0, MAX_CUBE_Y), Some(MIN_Y + 15));
        assert!(!column.covered_above(0, 0, MAX_CUBE_Y));
    }
}
