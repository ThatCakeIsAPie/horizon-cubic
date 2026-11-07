//! Structures for a cube's population (the integrator's
//! [`CubeWorld::place_structures`] hook): 26.3's overworld structure sets over
//! Custom Cubic terrain, placed by vanilla's rules (docs/cubic-v2-plan.md,
//! "Structures").
//!
//! # Starts
//!
//! A chunk column's starts are vanilla's ([`StructurePlanner`]: the structure
//! sets' placements, as the settings tune them, and each structure's own
//! generation), asked through a [`StartContext`] over the terrain generator:
//!
//! * Heightmaps are the terrain's surface: the topmost solid block of the
//!   density and replacers ([`CustomCubicGenerator::chunk_tops`]; carvers and
//!   population ignored, as vanilla's generation heightmaps come from the
//!   noise), with the sea filling up to sea level. A column has one surface,
//!   however high: a village stands on a mountain top 9,000 blocks up as on a
//!   plain.
//! * The level's height for the structures is vanilla's bottom (-64), so
//!   mineshafts, strongholds, trial chambers and ancient cities lie at
//!   vanilla's depths, and no top: a start is never refused for being high.
//! * Biomes are the generator's own (one per 4 x 4 column, the same at every
//!   height).
//!
//! # Placement
//!
//! The population of cube `c` places, into its area (`c * 16 + 8 ..
//! c * 16 + 24` on each axis, the area its ores and vegetation take), every
//! piece of the starts whose columns cross the area, at the height the
//! heightmaps give. Reads see the population's 2 x 2 x 2 cubes and stone
//! beyond them; the pieces draw only into the area. Every block of a
//! structure belongs to one area, so each is placed once, whichever cube
//! populates first. A piece whose height its first placement picks (a
//! temple's ground, a shipwreck's) is settled once, the same way by every
//! population ([`crate::worldgen::structure::start::settle_start`]).
//!
//! Chests take their loot tables and spawners their mobs; block entities and
//! the entities the pieces carry (villagers, guardians, a witch) go to the
//! [`CubeWorld`].
//!
//! # Terrain adaptation
//!
//! Each structure's `terrain_adaptation` (beard_thin, beard_box, bury,
//! encapsulate) and its jigsaw junctions shape the terrain as vanilla's
//! beardifier does: [`CubicStructures::beard`] is a chunk's
//! (`Beardifier.forStructuresInChunk`), which generation adds to the density
//! ([`CustomCubicGenerator::generate_adapted`]). The starts are decided on the
//! unadapted terrain, as vanilla's are; the pieces are placed on the adapted
//! one ([`CubicStructures::placed_height`]).
//!
//! # Not done
//!
//! * The far view shows no structures (as it shows no trees).
//! * Starts and references are not on the level's chunks: `/locate`, maps
//!   and structure mob spawns (witch huts, monuments, outposts) don't know
//!   these structures.

use std::collections::{BTreeSet, HashMap};
use std::sync::{Arc, Mutex};

use horizon_core::density::Beardifier;
use horizon_core::position::{BlockPos, ChunkPos, SectionPos};
use horizon_core::random::{LegacyRandom, RandomSource};

use super::CustomCubicGenerator;
use super::biomes::BiomeColumns;
use super::populate::CubeWorld;
use super::settings::{CustomGeneratorSettings, VanillaPlacement};
use crate::heightmap::HeightmapType;
use crate::level::{Behaviours, LevelAccess, Ticks};
use crate::registry::BlockStateId;
use crate::worldgen::anchor::WorldGenerationContext;
use crate::worldgen::biome::BiomeId;
use crate::worldgen::biome_temperature::BiomeTemperature;
use crate::worldgen::generator::GeneratorData;
use crate::worldgen::live::LiveFeatures;
use crate::worldgen::structure::BoundingBox;
use crate::worldgen::structure::jigsaw::JigsawContext;
use crate::worldgen::structure::start::{NoiseBlock, NoiseColumn, StartContext, StructureStart};
use crate::worldgen::tree_blocks::TreeBlocks;
use crate::worldgen::{SetTuning, StructurePlanner};

/// The structures' bottom of the level: vanilla's.
const MIN_Y: i32 = -64;
/// The height a structure's height providers see (vanilla's 384).
const HEIGHT: i32 = 384;
/// The population's region: 2 cubes on each axis.
const REGION: i32 = 32;
/// A column without a surface.
const NO_SURFACE: i32 = super::NO_TOP;
/// Chunks' column tops kept per shard.
const TOPS_PER_SHARD: usize = 1 << 10;
/// Cube columns' start lists kept per shard.
const AREAS_PER_SHARD: usize = 1 << 10;
const SHARDS: usize = 32;

/// The starts over a cube column's population areas, with each one's
/// decoration step and index in the step.
type AreaStarts = Arc<Vec<(usize, usize, Arc<StructureStart>)>>;

/// The structures of one Custom Cubic generator.
pub struct CubicStructures {
    planner: StructurePlanner,
    behaviours: Arc<Behaviours>,
    features: Arc<LiveFeatures>,
    blocks: Arc<TreeBlocks>,
    data: Arc<GeneratorData>,
    generator: Arc<CustomCubicGenerator>,
    columns: Arc<dyn BiomeColumns>,
    seed: i64,
    sea_level: i32,
    air: u32,
    water: u32,
    stone: u32,
    /// The data's biome id of each source biome id the source names.
    biome_ids: HashMap<u32, BiomeId>,
    /// Topmost solid block of each column of a chunk, by shard.
    tops: Box<[Mutex<HashMap<(i32, i32), Arc<[i32; 256]>>>]>,
    /// The starts over each cube column's areas (every cube of a column
    /// populates the same columns), by shard.
    areas: Box<[Mutex<HashMap<(i32, i32), AreaStarts>>]>,
    /// Each chunk's terrain adaptation, by shard.
    beards: Box<[Mutex<HashMap<(i32, i32), Beard>>]>,
    /// Each chunk's column tops with the terrain adapted (chunks the beard
    /// reaches only), by shard.
    adapted: Box<[Mutex<HashMap<(i32, i32), Arc<[i32; 256]>>>]>,
}

/// A chunk's terrain adaptation (`None`: no structure adapts it).
type Beard = Option<Arc<Beardifier>>;

impl CubicStructures {
    /// The structures `settings` switch on, or `None` when the bundle has no
    /// structures or live features, or every set is off.
    pub fn new(
        data: &Arc<GeneratorData>,
        seed: i64,
        settings: &CustomGeneratorSettings,
        generator: Arc<CustomCubicGenerator>,
        columns: Arc<dyn BiomeColumns>,
    ) -> Option<Self> {
        let mut biome_ids = HashMap::new();
        for id in columns.possible_biomes() {
            if let Some(data_id) = data.biome_id(columns.biome_name(id)) {
                biome_ids.insert(id.0, data_id);
            }
        }
        let possible: BTreeSet<BiomeId> = if biome_ids.is_empty() {
            (0..data.biome_names().len() as u32).map(BiomeId).collect()
        } else {
            biome_ids.values().copied().collect()
        };
        let tune = |name: &str| -> Option<SetTuning> {
            if !settings.structure_enabled(name) {
                return None;
            }
            let entry = settings.structure_sets.get(name);
            Some(match settings.structure_placement(name) {
                Some(VanillaPlacement::Spread {
                    spacing,
                    separation,
                    frequency,
                }) => SetTuning {
                    spacing: Some(spacing),
                    separation: Some(separation),
                    frequency: Some(frequency),
                    ..SetTuning::default()
                },
                Some(VanillaPlacement::Rings {
                    distance,
                    spread,
                    count,
                }) => SetTuning {
                    distance: Some(distance),
                    spread: Some(spread),
                    count: Some(count),
                    ..SetTuning::default()
                },
                // A set this module does not know: the entry's own fields.
                None => SetTuning {
                    spacing: entry.and_then(|t| t.spacing),
                    separation: entry.and_then(|t| t.separation),
                    frequency: entry.and_then(|t| t.frequency),
                    distance: entry.and_then(|t| t.distance),
                    spread: entry.and_then(|t| t.spread),
                    count: entry.and_then(|t| t.count),
                },
            })
        };
        let planner = match StructurePlanner::new(data, &possible, tune) {
            Ok(Some(p)) => p,
            Ok(None) => return None,
            Err(e) => {
                log::warn!("Horizon: no structures for the cubic generator: {e}");
                return None;
            }
        };
        if planner.set_names().is_empty() {
            return None;
        }
        let missing = |what: &str| {
            log::warn!("Horizon: no structures for the cubic generator: no {what}");
        };
        let Some(features) = LiveFeatures::new(data).ok().flatten() else {
            missing("live features");
            return None;
        };
        let features = Arc::new(features);
        let Some(behaviours) = data.behaviours().cloned() else {
            missing("block behaviours");
            return None;
        };
        let Some(blocks) = features.tree_blocks() else {
            missing("block facts");
            return None;
        };
        let registry = data.registry();
        let state = |name: &str| registry.default_state(name).map(|s| s.0);
        Some(Self {
            planner,
            behaviours,
            features,
            blocks,
            data: data.clone(),
            sea_level: generator.sea_level(),
            generator,
            columns,
            seed,
            air: state("minecraft:air")?,
            water: state("minecraft:water")?,
            stone: state("minecraft:stone")?,
            biome_ids,
            tops: (0..SHARDS).map(|_| Mutex::default()).collect(),
            areas: (0..SHARDS).map(|_| Mutex::default()).collect(),
            beards: (0..SHARDS).map(|_| Mutex::default()).collect(),
            adapted: (0..SHARDS).map(|_| Mutex::default()).collect(),
        })
    }

    /// The names of the structure sets that generate.
    pub fn set_names(&self) -> Vec<String> {
        self.planner.set_names()
    }

    /// The starts chunk `chunk` owns.
    pub fn starts_at(&self, chunk: ChunkPos) -> Arc<Vec<Arc<StructureStart>>> {
        self.planner.starts_at(&mut self.context(), chunk)
    }

    /// The starts whose box crosses the block columns `min..=max` (X, Z).
    pub fn touching(&self, min: [i32; 2], max: [i32; 2]) -> Vec<Arc<StructureStart>> {
        (self.planner.touching(&mut self.context(), min, max).into_iter())
            .map(|(_, _, s)| s)
            .collect()
    }

    /// A structure's name (`minecraft:village_plains`).
    pub fn structure_name(&self, structure: usize) -> &str {
        self.planner.structure_name(structure)
    }

    fn context(&self) -> Context<'_> {
        Context {
            s: self,
            valid: BTreeSet::new(),
            temperatures: Vec::new(),
        }
    }

    /// The data's biome of the source's quart column.
    fn biome(&self, qx: i32, qz: i32) -> BiomeId {
        let id = self.columns.biome(qx, qz);
        match self.biome_ids.get(&id.0) {
            Some(&found) => found,
            None => self
                .data
                .biome_id(self.columns.biome_name(id))
                .unwrap_or(BiomeId(0)),
        }
    }

    /// The starts over the population areas of cube column (`cx`, `cz`).
    fn area_starts(&self, cx: i32, cz: i32) -> AreaStarts {
        let shard = &self.areas[shard_of(cx, cz)];
        if let Some(found) = shard.lock().unwrap().get(&(cx, cz)) {
            return found.clone();
        }
        let (x, z) = (cx * 16 + 8, cz * 16 + 8);
        let starts = Arc::new(
            self.planner
                .touching(&mut self.context(), [x, z], [x + 15, z + 15]),
        );
        let mut areas = shard.lock().unwrap();
        if areas.len() >= AREAS_PER_SHARD {
            areas.clear();
        }
        areas.entry((cx, cz)).or_insert(starts).clone()
    }

    /// The topmost solid block of column (`x`, `z`), or [`NO_SURFACE`]
    /// (found a chunk at a time: [`CustomCubicGenerator::chunk_tops`]).
    pub fn column_top(&self, x: i32, z: i32) -> i32 {
        self.chunk_tops((x >> 4, z >> 4))[((z & 15) << 4 | (x & 15)) as usize]
    }

    /// [`CustomCubicGenerator::chunk_tops`] of chunk `key`, kept.
    fn chunk_tops(&self, key: (i32, i32)) -> Arc<[i32; 256]> {
        let shard = &self.tops[shard_of(key.0, key.1)];
        if let Some(tops) = shard.lock().unwrap().get(&key) {
            return tops.clone();
        }
        let found: Arc<[i32; 256]> = Arc::from(self.generator.chunk_tops(key.0, key.1));
        let mut tops = shard.lock().unwrap();
        if tops.len() >= TOPS_PER_SHARD {
            tops.clear();
        }
        tops.insert(key, found.clone());
        found
    }

    /// Vanilla's terrain adaptation for chunk `chunk`
    /// (`Beardifier.forStructuresInChunk` over the starts whose box crosses
    /// it), the same for every cube of the chunk's column; `None` when no
    /// structure adapts the terrain there. The starts are decided on the
    /// unadapted terrain ([`Self::column_top`]), as vanilla's are.
    pub fn beard(&self, chunk: ChunkPos) -> Option<Arc<Beardifier>> {
        let key = (chunk.x, chunk.z);
        let shard = &self.beards[shard_of(key.0, key.1)];
        if let Some(found) = shard.lock().unwrap().get(&key) {
            return found.clone();
        }
        let (x, z) = (chunk.x * 16, chunk.z * 16);
        let starts: Vec<Arc<StructureStart>> = (self.planner)
            .touching(&mut self.context(), [x, z], [x + 15, z + 15])
            .into_iter()
            .map(|(_, _, s)| s)
            .collect();
        let beard = self.planner.beardifier(&starts, chunk).map(Arc::new);
        let mut beards = shard.lock().unwrap();
        if beards.len() >= AREAS_PER_SHARD {
            beards.clear();
        }
        beards.entry(key).or_insert(beard).clone()
    }

    /// The topmost solid block of column (`x`, `z`) once the structures
    /// have adapted the terrain: the ground their pieces are placed on.
    pub fn adapted_top(&self, x: i32, z: i32) -> i32 {
        let key = (x >> 4, z >> 4);
        let i = ((z & 15) << 4 | (x & 15)) as usize;
        let Some(beard) = self.beard(ChunkPos::new(key.0, key.1)) else {
            return self.column_top(x, z);
        };
        let shard = &self.adapted[shard_of(key.0, key.1)];
        if let Some(tops) = shard.lock().unwrap().get(&key) {
            return tops[i];
        }
        let mut tops = Box::new(*self.chunk_tops(key));
        self.generator.adapt_tops(key.0, key.1, &mut tops, &beard);
        let top = tops[i];
        let mut adapted = shard.lock().unwrap();
        if adapted.len() >= TOPS_PER_SHARD {
            adapted.clear();
        }
        adapted.insert(key, Arc::from(tops));
        top
    }

    /// [`Self::height`] on the adapted terrain ([`Self::adapted_top`]): what
    /// the pieces see when they are placed.
    pub fn placed_height(&self, kind: HeightmapType, x: i32, z: i32) -> i32 {
        self.height_of(kind, self.adapted_top(x, z))
    }

    /// Generation's heightmap `kind` at column (`x`, `z`): one above the
    /// highest block of its kind, the sea filling to sea level.
    pub fn height(&self, kind: HeightmapType, x: i32, z: i32) -> i32 {
        self.height_of(kind, self.column_top(x, z))
    }

    fn height_of(&self, kind: HeightmapType, top: i32) -> i32 {
        let solid = if top == NO_SURFACE { MIN_Y } else { top + 1 };
        match kind {
            HeightmapType::OceanFloor | HeightmapType::OceanFloorWg => solid,
            _ => solid.max(self.sea_level),
        }
    }

    /// The column's blocks from [`MIN_Y`] up to its surface, by what
    /// structures ask of them (air past the end).
    fn noise_blocks(&self, x: i32, z: i32) -> NoiseColumn {
        let top = self.column_top(x, z).max(self.sea_level - 1);
        if top < MIN_Y {
            return NoiseColumn {
                min: MIN_Y,
                blocks: Vec::new(),
            };
        }
        let ys: Vec<i32> = (MIN_Y..=top).collect();
        let blocks = self
            .generator
            .blocks_at_column(x, z, &ys, 1)
            .into_iter()
            .map(|b| {
                if b == self.air {
                    NoiseBlock::Air
                } else if b == self.water {
                    NoiseBlock::Water
                } else {
                    NoiseBlock::Solid
                }
            })
            .collect();
        NoiseColumn {
            min: MIN_Y,
            blocks,
        }
    }

    /// Places the parts of the structures in the area of cube `cube`'s
    /// population (see the module docs).
    pub fn place(&self, cube: SectionPos, world: &mut dyn CubeWorld) {
        let min = [cube.x * 16 + 8, cube.y * 16 + 8, cube.z * 16 + 8];
        let max = [min[0] + 15, min[1] + 15, min[2] + 15];
        let starts = self.area_starts(cube.x, cube.z);
        if starts.is_empty() {
            return;
        }
        let chunk_box = BoundingBox::new(min, max);
        let select = BoundingBox::new(
            [min[0], crate::cubic::MIN_Y, min[2]],
            [max[0], crate::cubic::MAX_Y, max[2]],
        );
        let decoration = decoration_seed(self.seed, min);
        let mut region = Region {
            world,
            min: [cube.x * 16, cube.y * 16, cube.z * 16],
            stone: self.stone,
            random: LegacyRandom::new(decoration),
            s: self,
        };
        let mut ticks = Ticks::new(0);
        // Vanilla places the pieces after the noise: on the adapted ground.
        let heights = |kind: HeightmapType, x: i32, z: i32| self.placed_height(kind, x, z);
        let resources = self.planner.resources().clone();
        for (step, index, start) in starts.iter() {
            let (step, index) = (*step, *index);
            let mut random =
                LegacyRandom::new(decoration.wrapping_add(index as i64 + 10_000 * step as i64));
            let mut settle_random = LegacyRandom::new(settle_seed(self.seed, start.chunk, start.structure));
            let settle_box = BoundingBox::new(
                [start.chunk.x * 16, crate::cubic::MIN_Y, start.chunk.z * 16],
                [start.chunk.x * 16 + 15, crate::cubic::MAX_Y, start.chunk.z * 16 + 15],
            );
            let (placed, entities) = self.behaviours.place_structure_part(
                &mut region,
                &mut ticks,
                &self.features,
                start,
                &resources,
                &*self.blocks,
                select,
                chunk_box,
                settle_box,
                &mut settle_random,
                &mut random,
                &heights,
            );
            if !placed {
                log::debug!(
                    "Horizon: a piece of {} at chunk {:?} failed to place in cube {:?}",
                    start.name,
                    start.chunk,
                    cube
                );
            }
            for entity in entities {
                region.world.add_entity(entity);
            }
        }
    }
}

/// The cache shard of a column.
fn shard_of(x: i32, z: i32) -> usize {
    (x.wrapping_mul(0x9E37) ^ z.wrapping_mul(0x85EB)) as u32 as usize % SHARDS
}

/// `WorldgenRandom.setDecorationSeed` of the area's corner, with its height
/// mixed in (each cube of a column draws its own).
fn decoration_seed(seed: i64, min: [i32; 3]) -> i64 {
    let mut random = LegacyRandom::new(seed);
    let a = random.next_long() | 1;
    let b = random.next_long() | 1;
    let c = random.next_long() | 1;
    (i64::from(min[0]).wrapping_mul(a))
        .wrapping_add(i64::from(min[2]).wrapping_mul(b))
        .wrapping_add(i64::from(min[1]).wrapping_mul(c))
        ^ seed
}

/// The random a start's pieces are settled with: the start's own.
fn settle_seed(seed: i64, chunk: ChunkPos, structure: usize) -> i64 {
    seed ^ i64::from(chunk.x).wrapping_mul(341_873_128_712)
        ^ i64::from(chunk.z).wrapping_mul(132_897_987_541)
        ^ (structure as i64).wrapping_mul(0x5DEE_CE66)
}

/// The start queries of one population or lookup.
struct Context<'a> {
    s: &'a CubicStructures,
    valid: BTreeSet<BiomeId>,
    temperatures: Vec<BiomeTemperature>,
}

impl JigsawContext for Context<'_> {
    fn min_y(&self) -> i32 {
        MIN_Y
    }
    fn max_y(&self) -> i32 {
        crate::cubic::MAX_Y
    }
    fn base_height(&mut self, x: i32, z: i32, kind: HeightmapType) -> i32 {
        self.s.height(kind, x, z)
    }
    /// One biome per column at every height.
    fn column_has_valid_biome(&mut self, x: i32, z: i32, _min_y: i32, _max_y: i32) -> bool {
        self.valid.contains(&self.s.biome(x >> 2, z >> 2))
    }
}

impl StartContext for Context<'_> {
    fn seed(&self) -> i64 {
        self.s.seed
    }
    fn sea_level(&self) -> i32 {
        self.s.sea_level
    }
    fn base_column(&mut self, x: i32, z: i32, kind: HeightmapType) -> Vec<bool> {
        let column = self.s.noise_blocks(x, z);
        column
            .blocks
            .iter()
            .map(|b| match kind {
                HeightmapType::OceanFloor | HeightmapType::OceanFloorWg => b.is_solid(),
                _ => *b != NoiseBlock::Air,
            })
            .collect()
    }
    fn base_columns(&mut self, x: i32, z: i32, kinds: &[HeightmapType]) -> Vec<Vec<bool>> {
        kinds.iter().map(|&k| self.base_column(x, z, k)).collect()
    }
    fn noise_column(&mut self, x: i32, z: i32) -> NoiseColumn {
        self.s.noise_blocks(x, z)
    }
    fn cold_enough_to_snow(&mut self, biome: BiomeId, pos: [i32; 3]) -> bool {
        if self.temperatures.is_empty() {
            self.temperatures = (self.s.data.temperature_settings().iter())
                .map(|t| BiomeTemperature::new(*t))
                .collect();
        }
        let sea = self.s.sea_level;
        match self.temperatures.get_mut(biome.0 as usize) {
            Some(t) => t.cold_enough_to_snow(BlockPos::new(pos[0], pos[1], pos[2]), sea),
            None => false,
        }
    }
    fn generation(&self) -> WorldGenerationContext {
        WorldGenerationContext::new(MIN_Y, HEIGHT, self.s.sea_level)
    }
    fn noise_biome(&mut self, quart: [i32; 3]) -> BiomeId {
        self.s.biome(quart[0], quart[2])
    }
    fn noise_biomes(&mut self, quarts: &[[i32; 3]]) -> Vec<BiomeId> {
        quarts.iter().map(|q| self.s.biome(q[0], q[2])).collect()
    }
    fn set_valid_biomes(&mut self, biomes: Option<&BTreeSet<BiomeId>>) {
        self.valid = match biomes {
            Some(biomes) => biomes.clone(),
            None => (0..self.s.data.biome_names().len() as u32)
                .map(BiomeId)
                .collect(),
        };
    }
}

/// The population's 32 blocks as a level for the pieces: stone beyond them.
struct Region<'a, 'w> {
    world: &'w mut dyn CubeWorld,
    min: [i32; 3],
    stone: u32,
    random: LegacyRandom,
    s: &'a CubicStructures,
}

impl Region<'_, '_> {
    fn inside(&self, p: BlockPos) -> bool {
        (0..REGION).contains(&(p.x.wrapping_sub(self.min[0])))
            && (0..REGION).contains(&(p.y.wrapping_sub(self.min[1])))
            && (0..REGION).contains(&(p.z.wrapping_sub(self.min[2])))
    }
}

impl LevelAccess for Region<'_, '_> {
    fn block_state(&self, pos: BlockPos) -> BlockStateId {
        if self.inside(pos) {
            BlockStateId(self.world.block([pos.x, pos.y, pos.z]))
        } else {
            BlockStateId(self.stone)
        }
    }
    fn set_raw(&mut self, pos: BlockPos, state: BlockStateId) -> Option<BlockStateId> {
        if !self.inside(pos) {
            return None;
        }
        let old = self.world.block([pos.x, pos.y, pos.z]);
        self.world.set_block([pos.x, pos.y, pos.z], state.0);
        Some(BlockStateId(old))
    }
    fn is_loaded(&self, pos: BlockPos) -> bool {
        self.inside(pos)
    }
    fn min_y(&self) -> i32 {
        self.min[1]
    }
    fn max_y(&self) -> i32 {
        self.min[1] + REGION
    }
    fn random(&mut self) -> &mut LegacyRandom {
        &mut self.random
    }
    fn game_time(&self) -> i64 {
        0
    }
    fn biome_name(&self, pos: BlockPos) -> Option<&str> {
        let id = self.s.biome(pos.x >> 2, pos.z >> 2);
        Some(self.s.data.biome_name(id))
    }
    fn world_seed(&self) -> i64 {
        self.s.seed
    }
    fn load_block_entity(&mut self, pos: BlockPos, nbt: crate::nbt::Compound) {
        if self.inside(pos) {
            self.world.set_block_entity([pos.x, pos.y, pos.z], nbt);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoration_seeds_differ_by_height() {
        let a = decoration_seed(42, [8, 8, 8]);
        let b = decoration_seed(42, [8, 24, 8]);
        assert_ne!(a, b);
        assert_eq!(a, decoration_seed(42, [8, 8, 8]));
    }
}
