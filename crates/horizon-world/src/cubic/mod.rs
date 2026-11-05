//! Cubic Chunks: levels whose height is not bounded by their dimension type
//! (docs/cubic-chunks.md). A cube is one 16 x 16 x 16 section, addressed by
//! its [`SectionPos`]; a column of a cubic level holds no blocks of its own,
//! only the cubes loaded around the players.
//!
//! This module is the engine-agnostic part: the range of cubes a level may
//! have, where cubes are saved ([`store`]), which cubes a viewer keeps
//! ([`window`]) and the highest known block of each column across its
//! cubes ([`heights`]).

use horizon_core::position::SectionPos;

pub mod heights;
pub mod store;
pub mod window;

/// The lowest cube Y of a cubic level. `SectionPos::pack` keeps 20 bits of
/// Y, so every cube from `MIN_CUBE_Y` to [`MAX_CUBE_Y`] round-trips through
/// a packed section position.
pub const MIN_CUBE_Y: i32 = -(1 << 19);
/// The highest cube Y of a cubic level.
pub const MAX_CUBE_Y: i32 = (1 << 19) - 1;
/// The lowest block Y those cubes hold (-8,388,608).
pub const MIN_Y: i32 = MIN_CUBE_Y * 16;
/// The highest block Y those cubes hold (8,388,607).
pub const MAX_Y: i32 = MAX_CUBE_Y * 16 + 15;

/// Whether a cubic level has cubes at cube Y `y`.
pub const fn has_cube_y(y: i32) -> bool {
    MIN_CUBE_Y <= y && y <= MAX_CUBE_Y
}

/// Whether a cubic level has the cube `cube` (X and Z are bounded by the
/// world border, as in a vanilla level, not here).
pub const fn has_cube(cube: SectionPos) -> bool {
    has_cube_y(cube.y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use horizon_core::position::BlockPos;

    #[test]
    fn the_cube_range_survives_section_packing() {
        for y in [
            MIN_CUBE_Y,
            MIN_CUBE_Y + 1,
            -1,
            0,
            1,
            MAX_CUBE_Y - 1,
            MAX_CUBE_Y,
        ] {
            let cube = SectionPos::new(-1_875_000, y, 1_874_999);
            assert_eq!(SectionPos::unpack(cube.pack()), cube, "{y}");
        }
        // One cube past either end wraps around.
        let past = SectionPos::new(0, MAX_CUBE_Y + 1, 0);
        assert_ne!(SectionPos::unpack(past.pack()), past);
    }

    #[test]
    fn the_block_range_is_the_cubes() {
        assert_eq!(BlockPos::new(0, MIN_Y, 0).section().y, MIN_CUBE_Y);
        assert_eq!(BlockPos::new(0, MAX_Y, 0).section().y, MAX_CUBE_Y);
        assert_eq!(BlockPos::new(0, MAX_Y + 1, 0).section().y, MAX_CUBE_Y + 1);
        assert!((MIN_Y..=MAX_Y).contains(&-8_388_608));
        assert!((MIN_Y..=MAX_Y).contains(&8_388_607));
        assert!(!has_cube_y(MAX_CUBE_Y + 1) && !has_cube_y(MIN_CUBE_Y - 1));
    }
}
