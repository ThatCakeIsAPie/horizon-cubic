//! What a cubic level's far terrain is sampled from (docs/cubic-chunks.md):
//! the cube octree (`lod::cube`) asks for a node of 16³ voxels, each
//! `1 << level` blocks wide. Inside vanilla's -64..320 a voxel is read from
//! the vanilla generator's far-terrain columns (one grid per node column, kept
//! for the nodes above and below it); outside, from the cube generator.

use crate::cubic::CubicLevel;
use crate::lod::Column;
use crate::lod::cube::{CubeKey, CubeNode, CubeSampler, Kind, Voxel, downsample};
use crate::streaming::LodSampler;
use horizon_core::position::SectionPos;
use horizon_world::cubic::generator::CubeGenerator;
use horizon_world::registry::BlockStateId;
use std::collections::VecDeque;
use std::sync::Arc;

/// Node columns of vanilla far terrain kept per sampler.
const KEPT_GRIDS: usize = 64;
/// Vanilla's Overworld range, which the vanilla generator's columns cover.
const VANILLA: std::ops::Range<i32> = -64..320;

/// A cubic level's far-terrain sampler (`CubeSampler`), one per far
/// terrain thread.
pub struct CubicLodSampler {
    /// The level's saved cubes: the finest nodes are made of them where
    /// they are (the player's edits, vanilla's decorated terrain).
    level: Arc<CubicLevel>,
    generator: Arc<CubeGenerator>,
    vanilla: Box<dyn LodSampler>,
    air: u32,
    stone: u32,
    deepslate: u32,
    /// The states downsampling counts as fluid (water and lava sources).
    fluids: Vec<u32>,
    deep_biome: u16,
    sky_biome: u16,
    grids: VecDeque<((u8, i32, i32), Arc<Vec<Column>>)>,
}

impl CubicLodSampler {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        level: Arc<CubicLevel>,
        generator: Arc<CubeGenerator>,
        vanilla: Box<dyn LodSampler>,
        air: u32,
        stone: u32,
        deepslate: u32,
        fluids: Vec<u32>,
        deep_biome: u16,
        sky_biome: u16,
    ) -> Self {
        Self {
            level,
            generator,
            vanilla,
            air,
            stone,
            deepslate,
            fluids,
            deep_biome,
            sky_biome,
            grids: VecDeque::new(),
        }
    }

    /// The vanilla far columns under node column (`level`, `x`, `z`): 16 x 16,
    /// one at each voxel's centre.
    fn grid(&mut self, key: CubeKey) -> Arc<Vec<Column>> {
        let id = (key.level, key.x, key.z);
        if let Some((_, grid)) = self.grids.iter().find(|(k, _)| *k == id) {
            return grid.clone();
        }
        let origin = key.origin();
        let step = 1i32 << key.level;
        let half = step / 2;
        let grid = Arc::new(
            self.vanilla
                .grid(origin[0] + half, origin[2] + half, step, 16),
        );
        if self.grids.len() >= KEPT_GRIDS {
            self.grids.pop_front();
        }
        self.grids.push_back((id, grid.clone()));
        grid
    }

    /// The vanilla column's block at height `y` (`step` blocks a voxel): its
    /// span's, stone below a span's top voxel (a far column's run is one
    /// material all the way down), air in a gap.
    fn vanilla_voxel(&self, column: &Column, y: i32, step: i32) -> Voxel {
        for span in column.spans.iter() {
            let (min, max) = (i32::from(span.min), i32::from(span.max));
            if (min..max).contains(&y) {
                let state = if y < max - step.max(4) && span.state.0 != self.air {
                    self.stone
                } else {
                    span.state.0
                };
                return Voxel {
                    state: BlockStateId(state),
                    biome: span.biome,
                };
            }
        }
        Voxel {
            state: BlockStateId(self.air),
            biome: column.surface.map_or(self.sky_biome, |s| s.biome),
        }
    }
}

impl CubicLodSampler {
    /// A level 0 node from its saved cube, if it was saved.
    fn saved_cube(&self, pos: SectionPos) -> Option<CubeNode> {
        let section = self.level.saved_section(pos)?;
        let key = CubeKey::new(0, pos.x, pos.y, pos.z);
        if let Some(state) = section.blocks.uniform() {
            let biome = section.biomes.get(0) as u16;
            return Some(CubeNode::uniform(
                key,
                Voxel {
                    state: BlockStateId(state),
                    biome,
                },
            ));
        }
        let mut blocks = [0u32; 4096];
        section.unpack(&mut blocks);
        Some(CubeNode::from_fn(key, |x, y, z| Voxel {
            state: BlockStateId(blocks[y << 8 | z << 4 | x]),
            biome: section.biomes.get((y >> 2) << 4 | (z >> 2) << 2 | (x >> 2)) as u16,
        }))
    }

    /// Air, fluid or solid, as the downsampling counts it.
    fn kind(&self, state: BlockStateId) -> Kind {
        if state.0 == self.air {
            Kind::Air
        } else if self.fluids.contains(&state.0) {
            Kind::Fluid
        } else {
            Kind::Solid
        }
    }

    /// A node of the finest levels made of the saved cubes it holds (the
    /// others sampled as usual), or `None` when none of them was saved.
    fn of_saved_cubes(&mut self, key: CubeKey) -> Option<CubeNode> {
        match key.level {
            0 => self.saved_cube(SectionPos::new(key.x, key.y, key.z)),
            1 => {
                let children = key.children();
                let saved: Vec<Option<CubeNode>> = children
                    .iter()
                    .map(|c| self.saved_cube(SectionPos::new(c.x, c.y, c.z)))
                    .collect();
                if saved.iter().all(Option::is_none) {
                    return None;
                }
                let mut nodes = Vec::with_capacity(8);
                for (child, saved) in children.iter().zip(saved) {
                    nodes.push(match saved {
                        Some(node) => node,
                        None => self.sampled(*child)?,
                    });
                }
                let air = BlockStateId(self.air);
                let refs: [&CubeNode; 8] = std::array::from_fn(|i| &nodes[i]);
                Some(downsample(refs, air, |s| self.kind(s)))
            }
            _ => None,
        }
    }
}

impl CubeSampler for CubicLodSampler {
    fn node(&mut self, key: CubeKey) -> Option<CubeNode> {
        if key.level <= 1
            && let Some(node) = self.of_saved_cubes(key)
        {
            return Some(node);
        }
        self.sampled(key)
    }

    fn fork(&self) -> Option<Box<dyn CubeSampler>> {
        Some(Box::new(Self {
            level: self.level.clone(),
            generator: self.generator.clone(),
            vanilla: self.vanilla.fork()?,
            air: self.air,
            stone: self.stone,
            deepslate: self.deepslate,
            fluids: self.fluids.clone(),
            deep_biome: self.deep_biome,
            sky_biome: self.sky_biome,
            grids: VecDeque::new(),
        }))
    }
}

impl CubicLodSampler {
    /// A node sampled from the generators at its own resolution.
    fn sampled(&mut self, key: CubeKey) -> Option<CubeNode> {
        let origin = key.origin();
        let size = key.size();
        let step = 1i32 << key.level;
        let half = step / 2;
        let low = origin[1];
        let high = origin[1].saturating_add(size);
        // A node wholly above the sky's first islands' band or in the deep
        // with nothing but deepslate around: one voxel.
        let grid = (high > VANILLA.start && low < VANILLA.end).then(|| self.grid(key));
        if let Some(grid) = &grid
            && grid.len() < 256
        {
            // The vanilla columns aren't ready: try again later.
            return None;
        }
        let generator = self.generator.clone();
        let (air, deepslate) = (self.air, self.deepslate);
        let (deep_biome, sky_biome) = (self.deep_biome, self.sky_biome);
        Some(CubeNode::from_fn(key, |x, y, z| {
            let wy = low + y as i32 * step + half;
            if let Some(grid) = &grid
                && VANILLA.contains(&wy)
            {
                return self.vanilla_voxel(&grid[z * 16 + x], wy, step);
            }
            let wx = origin[0] + x as i32 * step + half;
            let wz = origin[2] + z as i32 * step + half;
            if wy < VANILLA.start {
                // Coarse deep voxels are solid: caves there are never seen
                // from afar, and each would be faces to draw.
                let state = if step >= 4 {
                    deepslate
                } else {
                    generator.block_at(wx, wy, wz)
                };
                return Voxel {
                    state: BlockStateId(state),
                    biome: deep_biome,
                };
            }
            let state = generator.block_at(wx, wy, wz);
            Voxel {
                state: BlockStateId(if state == 0 { air } else { state }),
                biome: sky_biome,
            }
        }))
    }
}
