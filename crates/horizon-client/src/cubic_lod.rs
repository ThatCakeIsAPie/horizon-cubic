//! What a cubic level's far terrain is sampled from (docs/cubic-chunks.md):
//! the cube octree (`lod::cube`) asks for a node of 16³ voxels, each
//! `1 << level` blocks wide.
//!
//! - Levels 0 and 1 come from saved cubes (what the player saw and built),
//!   else from CubicWorldGen's density and replacers (`block_at`, no caves
//!   or population).
//! - Levels 2 and up come from `GpuCubeSampler` (`far`, built from the
//!   generator's noise tables: [`far_setup`]), on the renderer's device when
//!   [`CubicLodSampler::with_device`] gave it one, else its CPU reference.
//!   Its 2D input, the node's columns, is made with the biome pass
//!   (`gfx/biome_gen`) when the biome size is the default, else from the
//!   generator's own blend (on the CPU: the fixed-biome and non-default
//!   `biomeSize` worlds keep that, and the CPU pre-pass with it).
//!   - On the GPU path of a Custom level (sparse descent on) the columns are
//!     made by the GPU thread in the same submission as the generation
//!     (`GpuCubeSampler::with_biomes`): the far workers do no per-node CPU
//!     work, and the descent keeps uniform nodes from being requested.
//!   - Otherwise (CPU reference, Hybrid) the columns are made on the calling
//!     worker with `CpuBiomes`, the biome pass's reference, which also feeds
//!     the 2D pre-pass (`classify`).
//! - A Hybrid world's vanilla band (-64..320) is read from the vanilla
//!   generator's far-terrain columns (one grid per node column, kept for the
//!   nodes above and below it); a node straddling the band's edge is sampled
//!   per voxel from both.
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
//! (`CubeNode::column_covered`): terrain, overhangs or floating land over a
//! node (probed with the generator's `block_at` at a few heights above it).
//! The mesher darkens those voxels. Nodes generated on the GPU into pool
//! slots carry none.

use crate::cubic::CubicLevel;
use crate::cubic_gen::{CubicKind, Generation};
use crate::gfx::biome_gen::{BiomeGenParams, BiomeTerrain, ClimateNoises, CpuBiomes};
use crate::gfx::cube_gen::{
    BiomeBlocks, ColumnFn, CubeGenParams, GpuBackend, GpuCubeSampler, NodeColumns, NoiseDesc,
    OctaveDesc,
};
use crate::lod::Column;
use crate::lod::cube::{
    Changes, CoverMask, CubeKey, CubeNode, CubeSampler, Kind, NodeClass, Resident, Voxel,
    downsample,
};
use crate::lod::cube_cache::{CubeLodCache, MAX_CACHED_LEVEL};
use crate::streaming::LodSampler;
use horizon_core::position::SectionPos;
use horizon_world::registry::{BlockRegistry, BlockStateId};
use std::collections::VecDeque;
use std::sync::Arc;

/// Node columns of vanilla far terrain kept per sampler.
const KEPT_GRIDS: usize = 64;
/// Vanilla's Overworld range, which the vanilla generator's columns cover.
const VANILLA: std::ops::Range<i32> = -64..320;
/// Nodes this level or finer get their sky cover worked out; coarser ones
/// are drawn lit.
const COVER_LEVEL: u8 = 8;
/// How far above a node's top the land that shades it is looked for: a few
/// looks, spread wider the higher they go (a mountain's slope, an overhang,
/// floating land).
const COVER_LOOKS: [i32; 6] = [2, 12, 36, 100, 260, 600];

/// A cubic level's far-terrain sampler (`CubeSampler`), one per far
/// terrain thread.
pub struct CubicLodSampler {
    /// The level's saved cubes: the finest nodes are made of them where
    /// they are (the player's edits, vanilla's decorated terrain).
    level: Arc<CubicLevel>,
    wg: Arc<Generation>,
    vanilla: Box<dyn LodSampler>,
    air: u32,
    stone: u32,
    deepslate: u32,
    /// The states downsampling counts as fluid (water and lava sources).
    fluids: Vec<u32>,
    grids: VecDeque<((u8, i32, i32), Arc<Vec<Column>>)>,
    /// The level's kept nodes, shared by every sampler and the saves.
    cache: Option<Arc<CubeLodCache>>,
    /// Writes kept nodes (and the parents they complete) on a thread of its
    /// own, shared by every fork: see [`Keeper`].
    keeper: Option<Arc<Keeper>>,
    /// How far this sampler has read the cache's changes.
    cursor: u64,
    /// Levels 2 and up outside a Hybrid band: the GPU sampler (or its CPU
    /// reference); `None` for a Flat world.
    far: Option<Box<dyn CubeSampler>>,
    /// What `far` is made of, to remake it on a device ([`Self::with_device`]).
    far_setup: Option<FarSetup>,
    /// The biome of air with no column to say (plains).
    plains: u16,
    /// Custom Cubic's structures, found in the background and drawn into the
    /// nodes they reach (`cubic_far_structures`); shared by every fork.
    structures: Option<Arc<crate::cubic_far_structures::FarStructures>>,
}

/// The inputs of the far generator: parameters and the node columns.
#[derive(Clone)]
pub struct FarSetup {
    pub params: CubeGenParams,
    pub columns: ColumnFn,
    /// The same climate, flattened for the GPU biome pass (`gfx/biome_gen`);
    /// `None` for worlds whose columns come from the generator's own blend
    /// (another biome size, a single biome) or whose noises do not fit.
    pub biomes: Option<BiomeGenParams>,
}

/// The block state a name gives, air when unknown.
fn state_of(registry: &BlockRegistry, name: &str) -> u32 {
    registry.default_state(name).map_or(0, |s| s.0)
}

/// `CubeGenParams` and the column function of the level's generator, for
/// `GpuCubeSampler`. `None` for a Flat world (nothing to evaluate). The
/// per-biome tables are indexed by the client's biome ids (the ids nodes
/// carry), mapped by name from the generator data's.
pub fn far_setup(level: &CubicLevel, registry: &BlockRegistry) -> Option<FarSetup> {
    let wg = level.generation();
    let tables = wg.generator.noise_tables()?;
    let shape = wg.generator.density_shape()?;
    let desc = |t: &horizon_world::cubic::custom::NoiseTable| NoiseDesc {
        factor: t.factor,
        offset: t.offset,
        frequency: t.frequency,
        octaves: t
            .octaves
            .iter()
            .map(|o| OctaveDesc {
                permutations: o.permutations,
                offsets: o.offsets,
            })
            .collect(),
    };
    let biomes = level.client_biomes();
    // The client's biome names by id.
    let mut names: Vec<&str> = vec!["minecraft:plains"; 256];
    for (name, &id) in &biomes.index {
        if let Some(slot) = names.get_mut(usize::from(id)) {
            *slot = name;
        }
    }
    let top = |name: &str| state_of(registry, &format!("minecraft:{name}"));
    let infos: Vec<horizon_world::cubic::custom::biomes::BiomeInfo> = names
        .iter()
        .map(|n| horizon_world::cubic::custom::biomes::biome_info(n))
        .collect();
    let params = CubeGenParams {
        noises: [
            desc(&tables[0]),
            desc(&tables[1]),
            desc(&tables[2]),
            desc(&tables[3]),
        ],
        height_factor: shape[0],
        height_offset: shape[1],
        height_variation_factor: shape[2],
        height_variation_offset: shape[3],
        below_average_factor: shape[4],
        sea_level: wg.generator.sea_level(),
        air: state_of(registry, "minecraft:air"),
        stone: state_of(registry, "minecraft:stone"),
        deepslate: state_of(registry, "minecraft:deepslate"),
        water: state_of(registry, "minecraft:water"),
        biomes: infos
            .iter()
            .map(|i| BiomeBlocks {
                top: top(i.top),
                filler: top(i.filler),
            })
            .collect(),
        filler_depth: 3.0,
        deepslate_from: 0.0,
        deepslate_to: 8.0,
    };
    let scale = wg.settings.as_ref().map_or(1.0, |s| s.biome_scale());
    let terrain: Vec<BiomeTerrain> = infos
        .iter()
        .map(|i| BiomeTerrain {
            base_height: i.base_height,
            height_variation: i.height_variation,
        })
        .collect();
    let mut gpu_biomes = None;
    let columns: ColumnFn = match (
        &wg.climate,
        scale == 1.0 && wg.settings.as_ref().is_some_and(|s| s.biome.is_empty()),
    ) {
        (Some(climate), true) => {
            // The climate's biome ids are the generator data's: the index
            // maps them to the client's by name.
            let data = climate.data();
            let entries = climate
                .index()
                .entries()
                .iter()
                .map(|(point, id)| {
                    (
                        *point,
                        u32::from(
                            biomes
                                .id(data.biome_name(horizon_world::worldgen::biome::BiomeId(*id))),
                        ),
                    )
                })
                .collect();
            let index =
                Arc::new(horizon_world::worldgen::climate::ClimateIndex::new(entries).ok()?);
            // The GPU pass reads the same climate, from a router compiled
            // for the seed again (the adapter keeps only the sampler).
            gpu_biomes = data
                .overworld_climate(wg.seed)
                .ok()
                .and_then(|c| ClimateNoises::from_router(c.router, &c.compiler))
                .and_then(|noises| BiomeGenParams::new(noises, &index, terrain.clone()));
            Arc::new(CpuBiomes::new(climate.climate().clone(), index, terrain)).column_fn()
        }
        _ => generator_columns(wg.clone(), biomes.clone()),
    };
    Some(FarSetup {
        params,
        columns,
        biomes: gpu_biomes,
    })
}

/// A node's columns from the generator's own blend (`column_terrain`) and
/// biome source, for worlds whose biomes the GPU pass does not reproduce
/// (another biome size, a single biome).
fn generator_columns(wg: Arc<Generation>, biomes: Arc<crate::biome::Biomes>) -> ColumnFn {
    Arc::new(move |key: CubeKey| {
        let origin = key.origin();
        let step = 1i32 << key.level;
        let mut columns = NodeColumns::flat(0.0, 0.0, 0);
        for z in 0..16usize {
            for x in 0..16usize {
                let wx = origin[0] + x as i32 * step + step / 2;
                let wz = origin[2] + z as i32 * step + step / 2;
                let (h, v) = wg.generator.column_terrain(wx, wz).unwrap_or((0.0, 0.0));
                columns.height[z * 16 + x] = h as f32;
                columns.volatility[z * 16 + x] = v as f32;
                let id = wg.columns.biome(wx >> 2, wz >> 2);
                columns.biome[z * 16 + x] = biomes.id(wg.columns.biome_name(id));
            }
        }
        columns
    })
}

impl CubicLodSampler {
    pub fn new(
        level: Arc<CubicLevel>,
        vanilla: Box<dyn LodSampler>,
        registry: &BlockRegistry,
        cache: Option<Arc<CubeLodCache>>,
    ) -> Self {
        let id = |name: &str| state_of(registry, name);
        let level_plains = level.client_biomes().id("minecraft:plains");
        let far_setup = far_setup(&level, registry);
        let far: Option<Box<dyn CubeSampler>> = far_setup.as_ref().map(|f| {
            Box::new(GpuCubeSampler::cpu(f.params.clone(), f.columns.clone()))
                as Box<dyn CubeSampler>
        });
        let structures = (level.generation().structures.clone())
            .filter(|_| level.generation().kind == CubicKind::Custom)
            .map(|s| {
                let trees = (level.generation().vegetation.clone()).map(|vegetation| {
                    crate::cubic_far_structures::FarTrees {
                        vegetation,
                        columns: level.generation().columns.clone(),
                        states: crate::cubic_far_structures::FarTrees::tree_states(registry),
                    }
                });
                crate::cubic_far_structures::FarStructures::new(
                    s,
                    trees,
                    level.generation().generator.clone(),
                    cache.clone(),
                    id("minecraft:air"),
                )
            });
        Self {
            wg: level.generation().clone(),
            level,
            vanilla,
            air: id("minecraft:air"),
            stone: id("minecraft:stone"),
            deepslate: id("minecraft:deepslate"),
            fluids: vec![id("minecraft:water"), id("minecraft:lava")],
            grids: VecDeque::new(),
            cursor: cache.as_ref().map_or(0, |c| c.cursor()),
            keeper: cache.as_ref().and_then(|c| {
                Keeper::start(
                    c.clone(),
                    id("minecraft:air"),
                    vec![id("minecraft:water"), id("minecraft:lava")],
                )
            }),
            cache,
            far,
            far_setup,
            plains: level_plains,
            structures,
        }
    }

    /// The far generator on the renderer's device and queue (levels 2 and up
    /// then generate there, and the renderer can mesh and draw the nodes
    /// without reading them back). Without compute shaders there it stays
    /// the CPU reference.
    pub fn with_device(mut self, device: wgpu::Device, queue: wgpu::Queue) -> Self {
        if let Some(f) = &self.far_setup {
            let sampler = GpuCubeSampler::new(device, queue, f.params.clone(), f.columns.clone());
            // The sparse descent proves subtrees uniform from CubicWorldGen's
            // density bound alone, so only where that density is the whole
            // world: Custom (Hybrid's vanilla band is another generator's).
            let sampler = match self.wg.generator.gpu_terms() {
                Some(terms) if self.wg.kind == CubicKind::Custom => sampler.with_descent(&terms),
                _ => sampler,
            };
            // Columns on the GPU only where the descent replaces the 2D
            // pre-pass (which needs CPU columns): otherwise every uniform
            // node would be generated.
            let sampler = match &f.biomes {
                Some(biomes) if sampler.descent().is_some() => sampler.with_biomes(biomes.clone()),
                _ => sampler,
            };
            self.far = Some(Box::new(sampler));
        }
        self
    }

    /// Whether node `key` overlaps a Hybrid world's vanilla band.
    fn in_band(&self, key: CubeKey) -> bool {
        self.wg.kind == CubicKind::Hybrid && in_vanilla(key)
    }

    /// Whether the far generator makes node `key`: level 2 or up, and none
    /// of it vanilla's.
    fn far_makes(&self, key: CubeKey) -> bool {
        self.far.is_some() && key.level >= 2 && !self.in_band(key)
    }
}

impl CubicLodSampler {
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
            biome: column.surface.map_or(self.plains, |s| s.biome),
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
                        None => self.sampled_plain(*child)?,
                    });
                }
                let air = BlockStateId(self.air);
                let refs: [&CubeNode; 8] = std::array::from_fn(|i| &nodes[i]);
                let node = downsample(refs, air, |s| self.kind(s));
                let (cover, exact) = self.cover_of(key);
                // Inside vanilla's range, cubes not saved yet are the far
                // columns' guess: the node waits until they are.
                let keep = exact && (all_saved || !self.in_band(key));
                Some((node.with_covered(cover), keep))
            }
            _ => None,
        }
    }

    /// `node` with the sky cover of its place.
    fn covered(&mut self, node: CubeNode) -> CubeNode {
        let started = std::time::Instant::now();
        let (cover, _) = self.cover_of(node.key);
        if crate::lod::cube::profile::on() {
            crate::lod::cube::profile::add(&crate::lod::cube::profile::S_COVER, started);
        }
        node.with_covered(cover)
    }

    /// Which columns of node `key` have something solid over the node, and
    /// whether that is certain (vanilla's far columns can be missing yet).
    ///
    /// In a Hybrid world, below the vanilla band everything is under its
    /// terrain, and inside it a column is covered when the vanilla far
    /// column rises above the node. From the node's top (or the band's top)
    /// upward the generator is asked for blocks at a few heights
    /// ([`COVER_LOOKS`]): terrain, overhangs, floating land. Coarse nodes
    /// are not shaded.
    fn cover_of(&mut self, key: CubeKey) -> (Option<Box<CoverMask>>, bool) {
        let hybrid = self.wg.kind == CubicKind::Hybrid;
        let origin = key.origin();
        let top = origin[1].saturating_add(key.size());
        if hybrid && top <= VANILLA.start {
            return (Some(Box::new([u64::MAX; 4])), true);
        }
        if key.level > COVER_LEVEL {
            return (None, true);
        }
        let mut mask = Box::new([0u64; 4]);
        let mut exact = true;
        if hybrid && origin[1] < VANILLA.end {
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
        let generator = self.wg.generator.clone();
        let (air, fluids) = (self.air, self.fluids.clone());
        let resolution = 1i32 << key.level;
        let start = if hybrid { top.max(VANILLA.end) } else { top };
        // The density bound proves most tiles of 4 x 4 columns free of land
        // at every look: no probe is made over those.
        let tile = 4 * resolution;
        let (low, high) = (
            start.saturating_add(COVER_LOOKS[0]),
            start.saturating_add(COVER_LOOKS[COVER_LOOKS.len() - 1]),
        );
        // Then each look's slab of a tile that is not: the slabs high over the
        // terrain are proved free too (most of the probes left were there, and
        // a probe is a whole `block_at`), the same bound over less of the box.
        // Which columns come out covered does not change.
        let air_box = |min: [i32; 3], max: [i32; 3]| {
            generator.classify_box(min, max) == horizon_world::cubic::custom::BoxClass::Air
        };
        let boxes = std::time::Instant::now();
        let mut open = [[false; COVER_LOOKS.len()]; 16];
        for (i, open) in open.iter_mut().enumerate() {
            let x = origin[0] + (i as i32 & 3) * tile;
            let z = origin[2] + (i as i32 >> 2) * tile;
            if air_box([x, low, z], [x + tile - 1, high, z + tile - 1]) {
                continue;
            }
            for (j, &up) in COVER_LOOKS.iter().enumerate() {
                let y = start.saturating_add(up);
                open[j] = !air_box([x, y, z], [x + tile - 1, y, z + tile - 1]);
            }
        }
        if crate::lod::cube::profile::on() {
            crate::lod::cube::profile::add(&crate::lod::cube::profile::S_COVER_BOX, boxes);
        }
        let solid = |state: u32| state != 0 && state != air && !fluids.contains(&state);
        let check = crate::lod::cube::far_knob("HORIZON_FAR_COVER_CHECK", 0) != 0;
        // Each column's looks that are left in one probe of the generator: the
        // column's noise and biome are worked out once for them all
        // (`first_at_column`, the same answer as `block_at` at each height).
        let step = resolution;
        let quarters = [step / 4, step * 3 / 4];
        let middle = [step / 2];
        let offsets: &[i32] = if step >= 32 { &quarters } else { &middle };
        let mut ys = Vec::with_capacity(COVER_LOOKS.len());
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
                        let tx = ((wx - origin[0]) / tile).clamp(0, 3);
                        let tz = ((wz - origin[2]) / tile).clamp(0, 3);
                        let open = &open[(tz * 4 + tx) as usize];
                        ys.clear();
                        ys.extend(
                            COVER_LOOKS
                                .iter()
                                .enumerate()
                                .filter(|&(j, _)| open[j])
                                .map(|(_, &up)| start.saturating_add(up)),
                        );
                        let found = !ys.is_empty()
                            && generator
                                .first_at_column(wx, wz, &ys, resolution, |b| solid(b))
                                .is_some();
                        if check {
                            let old = COVER_LOOKS.iter().any(|&up| {
                                solid(generator.block_at(wx, start.saturating_add(up), wz, resolution))
                            });
                            if old != found {
                                log::warn!(
                                    "Horizon far cover: column {wx} {wz} over {start} (level {}) is {found}, per-height probes say {old}",
                                    key.level
                                );
                            }
                        }
                        found
                    })
                });
                if covered {
                    mask[bit >> 6] |= 1 << (bit & 63);
                }
            }
        }
        (Some(mask), exact)
    }

    /// Stores `node` and whatever nodes it now completes (a parent whose
    /// eight children are stored), unless a cube under it was saved since
    /// `since` (the cache's cursor when the node's building started).
    fn keep(&mut self, node: &CubeNode, since: u64) {
        let started = std::time::Instant::now();
        // On the keeper's thread when it has room; here otherwise.
        let queued = self
            .keeper
            .as_ref()
            .is_some_and(|k| k.queue(node.clone(), since));
        if !queued {
            self.keep_timed(node, since);
        }
        if crate::lod::cube::profile::on() {
            crate::lod::cube::profile::add(&crate::lod::cube::profile::S_KEEP, started);
        }
    }

    fn keep_timed(&mut self, node: &CubeNode, since: u64) {
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

/// The cache writes of [`CubicLodSampler::keep`] on a thread of their own: a
/// node, then each parent it completes (read back from its eight children,
/// downsampled and stored). They took the far workers 2 to 13 s of every 40
/// in a fill after a teleport (research/far-fill); nothing they write is
/// needed for the nodes being made, only by later lookups. A full queue
/// (`QUEUE`) sends the write back to the worker, as it was.
pub struct Keeper {
    tx: std::sync::Mutex<std::sync::mpsc::SyncSender<(CubeNode, u64)>>,
}

impl Keeper {
    const QUEUE: usize = 1024;

    fn start(cache: Arc<CubeLodCache>, air: u32, fluids: Vec<u32>) -> Option<Arc<Self>> {
        let (tx, rx) = std::sync::mpsc::sync_channel::<(CubeNode, u64)>(Self::QUEUE);
        std::thread::Builder::new()
            .name("Cube far keeper".into())
            .spawn(move || {
                horizon_threads::behind_server();
                horizon_threads::far_terrain_current();
                let kind = |state: BlockStateId| {
                    if state.0 == air {
                        Kind::Air
                    } else if fluids.contains(&state.0) {
                        Kind::Fluid
                    } else {
                        Kind::Solid
                    }
                };
                while let Ok((node, since)) = rx.recv() {
                    keep_into(&cache, air, &kind, &node, since);
                }
            })
            .ok()?;
        Some(Arc::new(Self {
            tx: std::sync::Mutex::new(tx),
        }))
    }

    /// Hands `node` to the thread: false when its queue is full.
    fn queue(&self, node: CubeNode, since: u64) -> bool {
        self.tx
            .lock()
            .is_ok_and(|tx| tx.try_send((node, since)).is_ok())
    }
}

/// [`CubicLodSampler::keep`]'s work: stores `node` and each parent it now
/// completes, unless a cube under one was saved since `since`.
fn keep_into(
    cache: &CubeLodCache,
    air: u32,
    kind: &impl Fn(BlockStateId) -> Kind,
    node: &CubeNode,
    since: u64,
) {
    if !unchanged(cache, node.key, since) {
        return;
    }
    cache.put(node);
    let mut key = node.key;
    while key.level < MAX_CACHED_LEVEL {
        let parent = key.parent();
        if cache.contains(parent) || parent.level < 2 {
            break;
        }
        let mut nodes = Vec::with_capacity(8);
        for child in parent.children() {
            match cache.get(child) {
                Some(n) => nodes.push(n),
                None => return,
            }
        }
        let refs: [&CubeNode; 8] = std::array::from_fn(|i| &nodes[i]);
        let built = downsample(refs, BlockStateId(air), kind);
        if unchanged(cache, parent, since) {
            cache.put(&built);
        }
        key = parent;
    }
}

/// Whether node `key` overlaps vanilla's generated range (-64..320).
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
/// wide) that has a solid block (`solid(x, y, z)`) at one of the
/// [`COVER_LOOKS`] heights over `start`. A voxel wider than 32 blocks is
/// looked at in its four quarters.
/// The cover's rule with one probe per height (`cover_of` asks a column's
/// looks in one probe now): for the tests.
#[cfg(test)]
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
                    COVER_LOOKS
                        .iter()
                        .any(|&up| solid(wx, start.saturating_add(up), wz))
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
        let node = self.with_structures(node);
        // The generator is exact outside a Hybrid world's band (and
        // everywhere in the others).
        if key.level == 1 && !self.in_band(key) {
            self.keep(&node, since);
        }
        Some(node)
    }

    /// Levels 2 and up outside the band stay in the GPU pool when the far
    /// generator runs on the renderer's device; saved cubes and kept nodes
    /// still come first.
    fn resident(&mut self, key: CubeKey) -> Option<Resident> {
        if self.far_makes(key) {
            let since = self.cache.as_ref().map_or(0, |c| c.cursor());
            if let Some(node) = self.cache.as_ref().and_then(|c| c.get(key)) {
                return Some(Resident::Node(node));
            }
            if let Some(node) = self.of_cached_children(key, since) {
                return Some(Resident::Node(node));
            }
            if self.has_structures(key)
                && let Some(node) = self.structure_node(key)
            {
                return Some(Resident::Node(node));
            }
            return self.far.as_mut()?.resident(key);
        }
        self.node(key).map(Resident::Node)
    }

    /// The far generator's nodes in one request (it batches them on the
    /// GPU); kept nodes, saved cubes and the rest one by one, as `resident`.
    fn resident_many(&mut self, keys: &[CubeKey]) -> Vec<Option<Resident>> {
        let mut out: Vec<Option<Resident>> = (0..keys.len()).map(|_| None).collect();
        let mut far_keys = Vec::new();
        let mut far_places = Vec::new();
        for (i, &key) in keys.iter().enumerate() {
            if self.far_makes(key) {
                let started = std::time::Instant::now();
                let since = self.cache.as_ref().map_or(0, |c| c.cursor());
                if let Some(node) = self.cache.as_ref().and_then(|c| c.get(key)) {
                    out[i] = Some(Resident::Node(node));
                } else if let Some(node) = self.of_cached_children(key, since) {
                    out[i] = Some(Resident::Node(node));
                } else if self.has_structures(key)
                    && let Some(node) = self.structure_node(key)
                {
                    // Drawn with a structure's voxels: sampled here.
                    out[i] = Some(Resident::Node(node));
                } else {
                    far_keys.push(key);
                    far_places.push(i);
                }
                if crate::lod::cube::profile::on() {
                    crate::lod::cube::profile::add(&crate::lod::cube::profile::S_CACHE, started);
                }
            } else {
                out[i] = self.node(key).map(Resident::Node);
            }
        }
        if !far_keys.is_empty()
            && let Some(far) = self.far.as_mut()
        {
            let started = std::time::Instant::now();
            let answers = far.resident_many(&far_keys);
            if crate::lod::cube::profile::on() {
                crate::lod::cube::profile::add(&crate::lod::cube::profile::S_FAR, started);
            }
            for (i, answer) in far_places.into_iter().zip(answers) {
                out[i] = answer;
            }
        }
        out
    }

    fn classify(&mut self, key: CubeKey) -> NodeClass {
        if self.far_makes(key) {
            return self
                .far
                .as_mut()
                .map_or(NodeClass::Mixed, |far| far.classify(key));
        }
        NodeClass::Mixed
    }

    fn gpu(&self) -> Option<GpuBackend> {
        self.far.as_ref()?.gpu()
    }

    /// Surface tiles reproduce CubicWorldGen's main terrain from the GPU
    /// climate pass: a Custom world (not Hybrid's vanilla band) without
    /// `cubeAreas`, whose far generator runs on the GPU with its climate.
    fn surface(&self) -> bool {
        self.wg.kind == CubicKind::Custom
            && !self.wg.generator.has_areas()
            && self.gpu().is_some_and(|b| b.biome_params.is_some())
    }

    fn descent(&self) -> Option<std::sync::Arc<crate::gfx::cube_descent::GpuDescent>> {
        self.far.as_ref()?.descent()
    }

    fn sea_surface(&self) -> Option<i32> {
        self.far.as_ref()?.sea_surface()
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
            wg: self.wg.clone(),
            vanilla: self.vanilla.fork()?,
            air: self.air,
            stone: self.stone,
            deepslate: self.deepslate,
            fluids: self.fluids.clone(),
            grids: VecDeque::new(),
            cursor: self.cache.as_ref().map_or(0, |c| c.cursor()),
            cache: self.cache.clone(),
            keeper: self.keeper.clone(),
            far: match &self.far {
                Some(far) => Some(far.fork()?),
                None => None,
            },
            far_setup: self.far_setup.clone(),
            plains: self.plains,
            structures: self.structures.clone(),
        }))
    }

    fn late(&self) -> Option<Arc<dyn crate::lod::cube::LateContent>> {
        (self.structures.clone()).map(|s| s as Arc<dyn crate::lod::cube::LateContent>)
    }
}

impl CubicLodSampler {
    /// Whether a structure the far search found reaches node `key`.
    fn has_structures(&self, key: CubeKey) -> bool {
        self.structures.as_ref().is_some_and(|s| s.touches(key))
    }

    /// A node the far generator would leave in the GPU pool, sampled here
    /// to take a structure's voxels, or `None` when they change nothing (the
    /// GPU then makes it as before: a node the CPU made over the sea draws
    /// its water and floor a little unlike the GPU's own). Like the GPU's
    /// nodes, it carries no sky cover (they are drawn lit).
    fn structure_node(&mut self, key: CubeKey) -> Option<CubeNode> {
        let node = self.sampled_plain(key)?;
        let (node, changed) = self.structures.as_ref()?.overlay(node);
        changed.then_some(node)
    }

    /// A sampled node with the voxels of the structures over it.
    fn with_structures(&self, node: CubeNode) -> CubeNode {
        match &self.structures {
            Some(s) => s.overlay(node).0,
            None => node,
        }
    }

    /// A node sampled from the generators at its own resolution, with its
    /// sky cover.
    fn sampled(&mut self, key: CubeKey) -> Option<CubeNode> {
        let started = std::time::Instant::now();
        let node = self.sampled_plain(key);
        if crate::lod::cube::profile::on() {
            crate::lod::cube::profile::add(&crate::lod::cube::profile::S_SAMPLE, started);
        }
        Some(self.covered(node?))
    }

    /// Cube `key` (level 0) as the generator makes it whole: terrain and
    /// carvers (`generate`, on its lattice), not 4,096 point queries.
    fn generated_cube(&self, key: CubeKey) -> CubeNode {
        let mut blocks = Box::new([0u32; 4096]);
        let pos = SectionPos::new(key.x, key.y, key.z);
        self.wg.generator.generate(pos, &mut blocks);
        // Biomes vary by 4 x 4 columns.
        let mut biomes = [0u16; 16];
        for (i, biome) in biomes.iter_mut().enumerate() {
            let x = key.x * 16 + (i as i32 & 3) * 4;
            let z = key.z * 16 + (i as i32 >> 2) * 4;
            *biome = self.level.biome_id_at(x, z);
        }
        let air = self.air;
        CubeNode::from_fn(key, |x, y, z| {
            let state = blocks[y << 8 | z << 4 | x];
            Voxel {
                state: BlockStateId(if state == 0 { air } else { state }),
                biome: biomes[(z >> 2) << 2 | (x >> 2)],
            }
        })
    }

    /// [`Self::sampled`] without the sky cover.
    fn sampled_plain(&mut self, key: CubeKey) -> Option<CubeNode> {
        // Levels 2 and up outside the band: the far generator.
        if self.far_makes(key)
            && let Some(node) = self.far.as_mut().and_then(|far| far.node(key))
        {
            return Some(node);
        }
        let origin = key.origin();
        let size = key.size();
        let step = 1i32 << key.level;
        let half = step / 2;
        let low = origin[1];
        let high = origin[1].saturating_add(size);
        let band = self.wg.kind == CubicKind::Hybrid;
        let grid = (band && high > VANILLA.start && low < VANILLA.end).then(|| self.grid(key));
        if let Some(grid) = &grid
            && grid.len() < 256
        {
            // The vanilla columns aren't ready: try again later.
            return None;
        }
        // Outside a Hybrid band the finest nodes are whole cubes (level 1:
        // its eight, downsampled).
        if grid.is_none() && key.level == 0 {
            return Some(self.generated_cube(key));
        }
        if grid.is_none() && key.level == 1 {
            let nodes: Vec<CubeNode> = key
                .children()
                .iter()
                .map(|c| self.generated_cube(*c))
                .collect();
            let air = BlockStateId(self.air);
            let refs: [&CubeNode; 8] = std::array::from_fn(|i| &nodes[i]);
            return Some(downsample(refs, air, |s| self.kind(s)));
        }
        let generator = self.wg.generator.clone();
        let air = self.air;
        let level = self.level.clone();
        let node = CubeNode::from_fn(key, |x, y, z| {
            let wy = low + y as i32 * step + half;
            if let Some(grid) = &grid
                && VANILLA.contains(&wy)
            {
                return self.vanilla_voxel(&grid[z * 16 + x], wy, step);
            }
            let wx = origin[0] + x as i32 * step + half;
            let wz = origin[2] + z as i32 * step + half;
            let state = generator.block_at(wx, wy, wz, step);
            Voxel {
                state: BlockStateId(if state == 0 { air } else { state }),
                biome: level.biome_id_at(wx, wz),
            }
        });
        Some(node)
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
    fn land_over_a_node_shades_the_columns_under_it() {
        // A thick slab over x, z 40..80, 90 to 400 blocks over the node's top.
        let slab = |x: i32, y: i32, z: i32| {
            (40..80).contains(&x) && (40..80).contains(&z) && (154..464).contains(&y)
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
        // Beyond the looks' reach nothing counts.
        let mut mask = [0u64; 4];
        let far = |_: i32, y: i32, _: i32| y > 64 + COVER_LOOKS[5] + 100;
        cover_columns([0, 0, 0], 2, 64, &mut mask, far);
        assert_eq!(mask, [0; 4]);
    }

    #[test]
    fn a_wide_voxel_is_shaded_by_land_over_part_of_it() {
        // Level 6: voxels 64 blocks wide, looked at in quarters (16 and 48).
        let pillar =
            |x: i32, y: i32, z: i32| (10..20).contains(&x) && (10..20).contains(&z) && y > 1000;
        let mut mask = [0u64; 4];
        cover_columns([0, 0, 0], 6, 1024 - 3, &mut mask, pillar);
        assert!(covered(&mask, 0, 0));
        assert!(!covered(&mask, 1, 0));
    }
}
