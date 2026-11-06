//! A cubic level's window on the server (docs/cubic-chunks.md, "v1
//! architecture", and docs/cubic-v2-plan.md, "Residency"): the level's chunks
//! hold a window of cubes around its players, `2V + 3` cubes tall (V is the
//! vertical distance the players asked for, `horizon:cubic_view`), and the
//! window slides by whole cube layers, in place, when a player's cube is two
//! or more from its centre, or when V changes.
//!
//! A slide never reloads the level. The cubes that enter come from the
//! prefetch ring (`cubic_prefetch`: the next two layers beyond the window,
//! ready in memory for every loaded column), the ones that leave are saved
//! with their entities, block entities and ticks, and the sections both
//! windows hold stay as they are (`World::relayout`). The hub then tells the
//! players the window and sends them only the entering cubes
//! (`Hub::cubic_window_changed`).
use super::persist::EntityScope;
use super::{Dim, Game};
use horizon_client::cubic::{self, LoadedCube, VANILLA_CUBES};
use horizon_client::fast_hash::FxHashSet;
use horizon_core::position::{BlockPos, ChunkPos, SectionPos};
use horizon_core::ticks::TickArea;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Server ticks a slide waits for the ring to have its cubes before it makes
/// the missing ones itself.
const RING_WAIT: u32 = 20;
/// Ticks between aiming the ring again at the columns the level holds.
const RETARGET_EVERY: u32 = 20;

/// What the server remembers of the window between ticks.
#[derive(Default)]
pub(super) struct SlideState {
    /// Ticks a due slide has waited for the ring.
    waiting: u32,
    /// Ticks since the ring was aimed.
    since_retarget: u32,
}

/// A player in the cubic level: its cube, its vertical distance and column.
struct Viewer {
    cube: i32,
    vertical: u8,
    column: (i32, i32),
}

impl Game<'_> {
    /// The cubic level of the server's players: when the cubes they want
    /// loaded (the union of their ranges, `cubic::union_cubes`) call for
    /// another window, it slides there (`slide_cubic_window`); the ring is
    /// kept aimed at the layers beyond the window meanwhile.
    pub(super) fn follow_cubic_window(&mut self) -> horizon_client::Result<()> {
        if self.options.execution_role.is_client() || !self.level_shapes.overworld.cubic {
            return Ok(());
        }
        let viewers: Vec<Viewer> = self.lan.as_ref().map_or_else(Vec::new, |h| {
            h.hub
                .guests
                .iter()
                .filter(|g| g.placed && g.dimension == Dim::Overworld.id())
                .map(|g| Viewer {
                    cube: (g.position[1].floor() as i32) >> 4,
                    vertical: g.vertical,
                    column: g.chunk(),
                })
                .collect()
        });
        // TODO(shards): the union of every player's range is one window for the
        // level; vertical shards (docs/cubic-v2-plan.md) give players whose
        // ranges do not meet windows of their own.
        let Some(wanted) = cubic::union_cubes(
            viewers
                .iter()
                .map(|v| cubic::wanted_cubes(v.cube, v.vertical)),
        ) else {
            return Ok(());
        };
        let due = cubic::needs_new_window(self.level_shapes.overworld, wanted);
        self.cubic_slide.since_retarget += 1;
        if !due {
            self.cubic_slide.waiting = 0;
            if self.cubic_slide.since_retarget < RETARGET_EVERY {
                return Ok(());
            }
        }
        let back = self.dim;
        self.enter_level(None, Dim::Overworld)?;
        let result = if due {
            self.slide_cubic_window(wanted, &viewers)
        } else {
            self.aim_ring(&viewers);
            Ok(())
        };
        self.enter_level(None, back)?;
        result
    }

    /// Aims the level's prefetch ring at the layers beyond the window of the
    /// columns it holds, nearest the players first.
    fn aim_ring(&mut self, viewers: &[Viewer]) {
        self.cubic_slide.since_retarget = 0;
        let Some(level) = self.save.cubic.get() else {
            return;
        };
        let centres: Vec<(i32, i32)> = viewers.iter().map(|v| v.column).collect();
        level.retarget_ring(&centres, self.world.chunks.keys().copied());
    }

    /// Slides the level in place towards the cubes `wanted`: at most
    /// `cubic::RING` layers at each end per slide, unless a player is outside
    /// the window or the two do not overlap, which is a jump. It waits (up to
    /// `RING_WAIT` ticks) until the ring has the entering cubes of every
    /// loaded column. In order:
    ///
    /// 1. Columns that would enter vanilla's range without having been
    ///    generated there are taken out, to load again the normal way.
    /// 2. The leaving cubes are saved: blocks (when changed), block entities
    ///    and ticks, then their entities, which leave the world; block
    ///    entities and ticks leave the schedulers.
    /// 3. The level's window, the saved window and the shape move.
    /// 4. `World::relayout` puts the entering sections in, from the ring; their
    ///    entities, block entities and ticks join the world as a loaded
    ///    column's do.
    /// 5. The streaming shifts its per-section state.
    /// 6. Every column is lit again (the coarse tool: `dirty_light`).
    /// 7. The hub tells the players and queues the entering cubes for them.
    fn slide_cubic_window(
        &mut self,
        wanted: (i32, i32),
        viewers: &[Viewer],
    ) -> horizon_client::Result<()> {
        let Some(level) = self.save.cubic.get().cloned() else {
            return Ok(());
        };
        let old_shape = self.world.shape;
        let have = cubic::window_cube_range(old_shape);
        let inside = viewers.iter().all(|v| (have.0..=have.1).contains(&v.cube));
        let next = if inside {
            cubic::next_window(have, wanted)
        } else {
            wanted
        };
        if next == have {
            return Ok(());
        }
        let jump = next.1 < have.0 || next.0 > have.1;
        let entering = cubic::entering_cubes(have, next);
        let leaving = cubic::leaving_ranges(have, next);
        let mut columns: Vec<(i32, i32)> = self.world.chunks.keys().copied().collect();
        columns.sort_unstable();

        // 1. Columns without their vanilla range cannot get its cubes from the
        // ring or the generator: they go through the streaming again.
        let redo: HashSet<(i32, i32)> = if entering.iter().any(|cy| VANILLA_CUBES.contains(cy)) {
            columns
                .iter()
                .copied()
                .filter(|&(x, z)| !level.has_vanilla(x, z))
                .collect()
        } else {
            HashSet::new()
        };
        columns.retain(|c| !redo.contains(c));
        let ready = columns.iter().all(|&(x, z)| {
            entering
                .iter()
                .all(|&cy| level.is_prefetched(SectionPos::new(x, cy, z)))
        });
        if !ready && inside && !jump && self.cubic_slide.waiting < RING_WAIT {
            // The margin keeps the players in loaded cubes meanwhile.
            self.cubic_slide.waiting += 1;
            self.aim_ring(viewers);
            return Ok(());
        }
        self.cubic_slide.waiting = 0;
        log::info!(
            "Horizon: cubic window {}..={} -> {}..={} ({} entering, {} columns)",
            have.0,
            have.1,
            next.0,
            next.1,
            entering.len(),
            columns.len()
        );
        if !redo.is_empty() {
            self.unload_columns_now(&redo);
            if let Some(h) = self.lan.as_mut() {
                h.hub.reload_columns(self.dim.id(), &redo);
            }
        }

        // 2. Save what leaves.
        let in_scope: HashSet<(i32, i32)> = columns.iter().copied().collect();
        let time = self.world.game_time;
        if !leaving.is_empty() {
            self.save_leaving_cubes(&level, &columns, &leaving, &in_scope, time);
        }

        // 3. The window moves.
        let new_shape = cubic::window_of_cubes(old_shape, next);
        level.set_window(new_shape);
        *self.level_shapes.get_mut(self.dim.id()) = new_shape;
        if let Err(e) = cubic::save_window(&self.world_root, new_shape.min_y, new_shape.height) {
            log::warn!("Horizon: the cubic window was not recorded: {e}");
        }

        // 4. The entering sections, from the ring.
        let mut cubes: HashMap<(i32, i32, i32), LoadedCube> = HashMap::new();
        for &(x, z) in &columns {
            for &cy in &entering {
                cubes.insert((x, cy, z), level.ready_cube(SectionPos::new(x, cy, z)));
            }
        }
        let (old_layout, new_layout) = (old_shape.column(), new_shape.column());
        let air = {
            let mut air = [Arc::new(horizon_client::chunk::Section::default())];
            horizon_client::chunk::share_uniform(&mut air);
            let [air] = air;
            air
        };
        self.world.relayout(new_shape, |x, z, cy| {
            cubes
                .get(&(x, cy, z))
                .map_or_else(|| air.clone(), |cube| cube.section.clone())
        });
        for &(x, z) in &columns {
            if let Some(chunk) = self.world.chunks.get_mut(&(x, z)) {
                Arc::make_mut(chunk).sky_covered = level.sky_cover(x, z, new_layout);
            }
        }
        for ((x, _, z), cube) in cubes {
            self.join_entering_cube((x, z), cube, time);
        }

        // 5. What the streaming and the caches keep by section index.
        self.streaming.shift_layout(old_layout, new_layout);
        self.random_tick_sections.clear();
        self.portals.sections.clear();

        // 6. Light: the entering layers and the ones beside them, as a whole
        // column's light job does it.
        self.world.dirty_light.extend(columns.iter().copied());

        // 7. The players.
        if let Some(h) = self.lan.as_mut() {
            h.hub.cubic_window_changed(self.dim.id(), new_shape);
        }
        self.aim_ring(viewers);
        Ok(())
    }

    /// Saves the cubes of `leaving` (ranges of cube Ys) in `columns`, which
    /// are about to leave the window: their blocks and block entities and
    /// ticks (`CubicLevel::save_cube`), then their entities, which are taken
    /// out of the world, and last their block entities and scheduled ticks,
    /// which are taken out of the stores.
    fn save_leaving_cubes(
        &mut self,
        level: &Arc<horizon_client::cubic::CubicLevel>,
        columns: &[(i32, i32)],
        leaving: &[std::ops::RangeInclusive<i32>],
        in_scope: &HashSet<(i32, i32)>,
        time: i64,
    ) {
        // Blocks, with the block entities and ticks as saved copies carry them.
        for &(x, z) in columns {
            let Some(chunk) = self.world.chunks.get(&(x, z)) else {
                continue;
            };
            let copy = self.with_entities([chunk]);
            for range in leaving {
                for cy in range.clone() {
                    level.save_cube(&copy[0], cy);
                    level.release_cube(SectionPos::new(x, cy, z));
                }
            }
        }
        // Entities of the leaving cubes: saved with their cubes, removed from the world.
        self.spawn_loaded_entities();
        for range in leaving {
            let scope = EntityScope::cubes(in_scope, range.clone());
            for ((x, z), tags) in self.entity_chunks_in(&scope, true) {
                let count = tags.len();
                level.save_leaving_entities(x, z, range.clone(), tags);
                if count > 0 {
                    self.native_entities_unload(self.dim.id(), [x, z], count);
                }
            }
        }
        // Block entities and scheduled ticks.
        let lo_x = columns.iter().map(|c| c.0).min().unwrap_or(0);
        let hi_x = columns.iter().map(|c| c.0).max().unwrap_or(0);
        let lo_z = columns.iter().map(|c| c.1).min().unwrap_or(0);
        let hi_z = columns.iter().map(|c| c.1).max().unwrap_or(0);
        for range in leaving {
            let (low, high) = (*range.start() * 16, *range.end() * 16 + 15);
            let inside = |p: &(i32, i32, i32)| {
                (low..=high).contains(&p.1) && in_scope.contains(&(p.0 >> 4, p.2 >> 4))
            };
            self.world.block_entities.unload(|p| inside(p));
            self.world.block_names.retain(|p, _| !inside(p));
            let area = TickArea::new(
                BlockPos::new(lo_x * 16, low, lo_z * 16),
                BlockPos::new(hi_x * 16 + 15, high, hi_z * 16 + 15),
            );
            let _ = self.block_ticks.blocks.take_area(area, time);
            let _ = self.block_ticks.fluids.take_area(area, time);
        }
    }

    /// An entering cube's block entities, entities and ticks join the world
    /// as those of a column that loaded do.
    fn join_entering_cube(&mut self, (x, z): (i32, i32), cube: LoadedCube, time: i64) {
        if !cube.block_entities.is_empty() {
            self.world.block_entities.load_chunk(cube.block_entities);
        }
        self.world.block_names.extend(cube.block_names);
        if !cube.entities.is_empty() {
            self.world.loaded_entities.push(((x, z), cube.entities));
        }
        let pos = ChunkPos::new(x, z);
        if !cube.block_ticks.is_empty() {
            self.block_ticks
                .blocks
                .add_saved(pos, cube.block_ticks, time);
        }
        if !cube.fluid_ticks.is_empty() {
            self.block_ticks
                .fluids
                .add_saved(pos, cube.fluid_ticks, time);
        }
    }

    /// Takes `columns` out of the level in place as the streaming unloads a far
    /// one (saved, with their entities and block entities), so it loads them again.
    fn unload_columns_now(&mut self, columns: &HashSet<(i32, i32)>) {
        let keys: FxHashSet<(i32, i32)> = columns.iter().copied().collect();
        let unloaded = self.streaming.unload_columns(&mut self.world, &keys);
        self.native_chunks_changed(&unloaded);
        for chunk in &unloaded {
            self.world.modified.remove(&(chunk.x, chunk.z));
        }
        let saved = self.with_entities(&unloaded);
        self.unload_entities(&unloaded);
        self.world
            .block_entities
            .unload(|p| keys.contains(&(p.0 >> 4, p.2 >> 4)));
        self.world
            .block_names
            .retain(|p, _| !keys.contains(&(p.0 >> 4, p.2 >> 4)));
        if let Err(e) = self
            .save
            .save_later(&saved, self.pack.assets.registry, world_biomes!(self))
        {
            log::warn!("Horizon: failed to save columns taken out of the window: {e}");
        }
    }
}
