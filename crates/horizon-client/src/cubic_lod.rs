//! What a cubic level's far terrain is sampled from (docs/cubic-chunks.md):
//! the cube octree (`lod::cube`) asks for a node of 16³ voxels, each
//! `1 << level` blocks wide. Inside vanilla's -64..320 a voxel is read from
//! the vanilla generator's far-terrain columns (one grid per node column, kept
//! for the nodes above and below it); outside, from the cube generator.
//!
//! What was seen is kept (`lod::cube_cache`, `<dimension>/horizon_cube_lod/`):
//! a node of level 1 made of saved cubes (the player's edits, vanilla's
//! decorated terrain), or sampled outside vanilla's range where the generator
//! is exact, is stored; a coarser node is built from its eight stored
//! children and stored too, up to `MAX_CACHED_LEVEL`. A node is asked for from
//! the store first, then built from stored children or saved cubes, and only
//! then sampled. `CubeLodCache::invalidate` (called by the level when it saves
//! a cube) erases the stored nodes over a cube.
//!
//! Nodes also carry which columns have something solid over them
//! (`CubeNode::column_covered`): the level's terrain over a node below its
//! surface, the generator's islands over anything under them. The mesher
//! darkens those voxels.

use crate::cubic::CubicLevel;
use crate::lod::Column;
use crate::lod::cube::{
    Changes, CoverMask, CubeKey, CubeNode, CubeSampler, Kind, Voxel, downsample,
};
use crate::lod::cube_cache::{CubeLodCache, MAX_CACHED_LEVEL};
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
/// Nodes this level or finer get their sky cover worked out; coarser ones
/// are drawn lit.
const COVER_LEVEL: u8 = 8;
/// How far above a node the islands that shade it are looked for, and the
/// vertical spacing of the looks (an island is tens of blocks thick).
const COVER_REACH: i32 = 1200;
const COVER_STEP: i32 = 24;

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
    /// The level's kept nodes, shared by every sampler and the saves.
    cache: Option<Arc<CubeLodCache>>,
    /// How far this sampler has read the cache's changes.
    cursor: u64,
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
        cache: Option<Arc<CubeLodCache>>,
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
            cursor: cache.as_ref().map_or(0, |c| c.cursor()),
            cache,
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
        // The sampler takes each cell's corner and samples its centre.
        let grid = Arc::new(self.vanilla.grid(origin[0], origin[2], step, 16));
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
    /// others sampled as usual), or `None` when none of them was saved. The
    /// flag says whether it is worth keeping (see [`Self::keep`]).
    fn of_saved_cubes(&mut self, key: CubeKey) -> Option<(CubeNode, bool)> {
        match key.level {
            0 => {
                let node = self.saved_cube(SectionPos::new(key.x, key.y, key.z))?;
                Some((self.covered(node), false))
            }
            1 => {
                let children = key.children();
                let saved: Vec<Option<CubeNode>> = children
                    .iter()
                    .map(|c| self.saved_cube(SectionPos::new(c.x, c.y, c.z)))
                    .collect();
                if saved.iter().all(Option::is_none) {
                    return None;
                }
                let all_saved = saved.iter().all(Option::is_some);
                let mut nodes = Vec::with_capacity(8);
                for (child, saved) in children.iter().zip(saved) {
                    nodes.push(match saved {
                        Some(node) => node,
                        None => self.sampled(*child)?,
                    });
                }
                let air = BlockStateId(self.air);
                let refs: [&CubeNode; 8] = std::array::from_fn(|i| &nodes[i]);
                let node = downsample(refs, air, |s| self.kind(s));
                let (cover, exact) = self.cover_of(key);
                // Inside vanilla's range, cubes not saved yet are the far
                // columns' guess: the node waits until they are.
                let keep = exact && (all_saved || !in_vanilla(key));
                Some((node.with_covered(cover), keep))
            }
            _ => None,
        }
    }

    /// `node` with the sky cover of its place.
    fn covered(&mut self, node: CubeNode) -> CubeNode {
        let (cover, _) = self.cover_of(node.key);
        node.with_covered(cover)
    }

    /// Which columns of node `key` have something solid over the node, and
    /// whether that is certain (vanilla's far columns can be missing yet).
    ///
    /// Below vanilla's range everything is under its terrain. Inside it a
    /// column is covered when the vanilla far column rises above the node.
    /// From the node's top (or vanilla's top) upward the generator is asked
    /// for blocks a few dozen blocks apart for [`COVER_REACH`] blocks: its
    /// islands. Coarse nodes are not shaded.
    fn cover_of(&mut self, key: CubeKey) -> (Option<Box<CoverMask>>, bool) {
        let origin = key.origin();
        let top = origin[1].saturating_add(key.size());
        if top <= VANILLA.start {
            return (Some(Box::new([u64::MAX; 4])), true);
        }
        if key.level > COVER_LEVEL {
            return (None, true);
        }
        let mut mask = Box::new([0u64; 4]);
        let mut exact = true;
        if origin[1] < VANILLA.end {
            let grid = self.grid(key);
            if grid.len() == 256 {
                for (i, column) in grid.iter().enumerate() {
                    if column.spans.iter().any(|span| i32::from(span.max) > top) {
                        mask[i >> 6] |= 1 << (i & 63);
                    }
                }
            } else {
                exact = false;
            }
        }
        let generator = self.generator.clone();
        let air = self.air;
        cover_columns(
            origin,
            key.level,
            top.max(VANILLA.end),
            &mut mask,
            |x, y, z| {
                let state = generator.block_at(x, y, z);
                state != 0 && state != air
            },
        );
        (Some(mask), exact)
    }

    /// Stores `node` and whatever nodes it now completes (a parent whose
    /// eight children are stored), unless a cube under it was saved since
    /// `since` (the cache's cursor when the node's building started).
    fn keep(&mut self, node: &CubeNode, since: u64) {
        let Some(cache) = self.cache.clone() else {
            return;
        };
        if !unchanged(&cache, node.key, since) {
            return;
        }
        cache.put(node);
        let mut key = node.key;
        while key.level < MAX_CACHED_LEVEL {
            let parent = key.parent();
            if cache.contains(parent) || self.of_cached_children(parent, since).is_none() {
                break;
            }
            key = parent;
        }
    }

    /// Level `key` made of its eight stored children, stored itself.
    fn of_cached_children(&mut self, key: CubeKey, since: u64) -> Option<CubeNode> {
        let cache = self.cache.clone()?;
        if key.level < 2 || key.level > MAX_CACHED_LEVEL {
            return None;
        }
        let mut nodes = Vec::with_capacity(8);
        for child in key.children() {
            nodes.push(cache.get(child)?);
        }
        let air = BlockStateId(self.air);
        let refs: [&CubeNode; 8] = std::array::from_fn(|i| &nodes[i]);
        let node = downsample(refs, air, |s| self.kind(s));
        if unchanged(&cache, key, since) {
            cache.put(&node);
        }
        Some(node)
    }
}

/// Whether node `key` lies within vanilla's generated range.
fn in_vanilla(key: CubeKey) -> bool {
    let low = key.origin()[1];
    low.saturating_add(key.size()) > VANILLA.start && low < VANILLA.end
}

/// Whether no cube under `key` was saved since the cache's `since`.
fn unchanged(cache: &CubeLodCache, key: CubeKey, since: u64) -> bool {
    let mut cursor = since;
    match cache.changes_since(&mut cursor) {
        Changes::None => true,
        Changes::Cubes(cubes) => !cubes.iter().any(|&c| key.contains_cube(c)),
        Changes::All => false,
    }
}

/// Sets the bit of each column of the node at `origin` (`16 << level` blocks
/// wide) that has a solid block (`solid(x, y, z)`) somewhere over `start`,
/// looking every [`COVER_STEP`] blocks up to [`COVER_REACH`]. A voxel wider
/// than 32 blocks is looked at in its four quarters.
fn cover_columns(
    origin: [i32; 3],
    level: u8,
    start: i32,
    mask: &mut CoverMask,
    solid: impl Fn(i32, i32, i32) -> bool,
) {
    let step = 1i32 << level;
    let quarters = [step / 4, step * 3 / 4];
    let middle = [step / 2];
    let offsets: &[i32] = if step >= 32 { &quarters } else { &middle };
    for z in 0..16usize {
        for x in 0..16usize {
            let bit = z << 4 | x;
            if mask[bit >> 6] >> (bit & 63) & 1 == 1 {
                continue;
            }
            let covered = offsets.iter().any(|&dz| {
                offsets.iter().any(|&dx| {
                    let wx = origin[0] + x as i32 * step + dx;
                    let wz = origin[2] + z as i32 * step + dz;
                    (0..COVER_REACH / COVER_STEP)
                        .any(|k| solid(wx, start + COVER_STEP / 3 + k * COVER_STEP, wz))
                })
            });
            if covered {
                mask[bit >> 6] |= 1 << (bit & 63);
            }
        }
    }
}

impl CubeSampler for CubicLodSampler {
    fn node(&mut self, key: CubeKey) -> Option<CubeNode> {
        let since = self.cache.as_ref().map_or(0, |c| c.cursor());
        if let Some(node) = self.cache.as_ref().and_then(|c| c.get(key)) {
            return Some(node);
        }
        if let Some(node) = self.of_cached_children(key, since) {
            return Some(node);
        }
        if key.level <= 1
            && let Some((node, keep)) = self.of_saved_cubes(key)
        {
            if keep {
                self.keep(&node, since);
            }
            return Some(node);
        }
        let node = self.sampled(key)?;
        // The generator is exact outside vanilla's range.
        if key.level == 1 && !in_vanilla(key) {
            self.keep(&node, since);
        }
        Some(node)
    }

    fn changes(&mut self) -> Changes {
        match &self.cache {
            Some(cache) => cache.changes_since(&mut self.cursor),
            None => Changes::None,
        }
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
            cursor: self.cache.as_ref().map_or(0, |c| c.cursor()),
            cache: self.cache.clone(),
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
        let node = CubeNode::from_fn(key, |x, y, z| {
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
        });
        Some(self.covered(node))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn covered(mask: &CoverMask, x: usize, z: usize) -> bool {
        let bit = z << 4 | x;
        mask[bit >> 6] >> (bit & 63) & 1 == 1
    }

    #[test]
    fn an_island_shades_the_columns_under_it() {
        // A slab of rock over x, z 40..80, 500 blocks above the node.
        let slab = |x: i32, y: i32, z: i32| {
            (40..80).contains(&x) && (40..80).contains(&z) && (500..510).contains(&y)
        };
        // Level 2: voxels 4 blocks wide, the node 64 blocks, top at 64.
        let mut mask = [0u64; 4];
        cover_columns([0, 0, 0], 2, 64, &mut mask, slab);
        for z in 0..16 {
            for x in 0..16 {
                // A voxel's middle is at 4 * i + 2.
                let under = (40..80).contains(&(4 * x + 2)) && (40..80).contains(&(4 * z + 2));
                assert_eq!(covered(&mask, x as usize, z as usize), under, "{x} {z}");
            }
        }
        // Looking from above the slab finds nothing.
        let mut mask = [0u64; 4];
        cover_columns([0, 0, 0], 2, 600, &mut mask, slab);
        assert_eq!(mask, [0; 4]);
        // Beyond the reach nothing counts.
        let mut mask = [0u64; 4];
        let far = |_: i32, y: i32, _: i32| y > 64 + COVER_REACH + 100;
        cover_columns([0, 0, 0], 2, 64, &mut mask, far);
        assert_eq!(mask, [0; 4]);
    }

    #[test]
    fn a_wide_voxel_is_shaded_by_an_island_over_part_of_it() {
        // Level 6: voxels 64 blocks wide, looked at in quarters (16 and 48).
        let pillar =
            |x: i32, y: i32, z: i32| (10..20).contains(&x) && (10..20).contains(&z) && y > 1000;
        let mut mask = [0u64; 4];
        cover_columns([0, 0, 0], 6, 1024, &mut mask, pillar);
        assert!(covered(&mask, 0, 0));
        assert!(!covered(&mask, 1, 0));
    }
}
