//! Flat Cubic: CWG's superflat without a height limit. Each layer holds the
//! blocks from `fromY` up to (excluding) `toY`; where layers overlap the one
//! with the highest `fromY` wins; everything else is air.

use horizon_core::position::SectionPos;

use super::replacers::resolve_block;
use super::settings::{BlockStateDesc, FlatGeneratorSettings};
use crate::registry::BlockRegistry;

pub struct FlatGenerator {
    /// `(from, to, block)` sorted by `from`.
    layers: Vec<(i32, i32, u32)>,
    air: u32,
}

impl FlatGenerator {
    pub fn new(s: &FlatGeneratorSettings, registry: &BlockRegistry) -> Result<Self, String> {
        let air = resolve_block(registry, &BlockStateDesc::new("air"))?;
        let mut layers = Vec::new();
        for (key, l) in &s.layers {
            let block = resolve_block(registry, &l.block_state)
                .map_err(|e| format!("flat layer {key}: {e}"))?;
            layers.push((l.from_y, l.to_y, block));
        }
        layers.sort_by_key(|l| l.0);
        Ok(Self { layers, air })
    }

    /// The block at height `y`.
    pub fn block_at_y(&self, y: i32) -> u32 {
        self.layers
            .iter()
            .rev()
            .find(|l| l.0 <= y && y < l.1)
            .map_or(self.air, |l| l.2)
    }

    /// The highest solid layer top (the surface), if any layer is not air.
    pub fn surface_y(&self) -> Option<i32> {
        self.layers
            .iter()
            .filter(|l| l.2 != self.air)
            .map(|l| l.1 - 1)
            .max()
    }

    /// Fills the cube (index `y << 8 | z << 4 | x`); `Some(id)` if uniform.
    pub fn generate(&self, cube: SectionPos, out: &mut [u32; 4096]) -> Option<u32> {
        let first = self.block_at_y(cube.y * 16);
        let mut uniform = true;
        for y in 0..16 {
            let b = self.block_at_y(cube.y * 16 + y);
            uniform &= b == first;
            out[(y as usize) << 8..((y as usize) + 1) << 8].fill(b);
        }
        uniform.then_some(first)
    }
}
