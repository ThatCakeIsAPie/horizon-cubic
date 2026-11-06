//! A cubic level's windows on the server (docs/cubic-chunks.md, "v1
//! architecture", and docs/cubic-v2-plan.md, "Residency"): the level's chunks
//! hold a window of cubes around its players, `2V + 3` cubes tall (V is the
//! vertical distance the players asked for, `horizon:cubic_view`), and the
//! window slides by whole cube layers, in place, when a player's cube is two
//! or more from its centre, or when V changes.
//!
//! **Shards.** Players whose wanted ranges overlap share a window; a player
//! far above or below the others has a shard of its own: another `World`
//! with its own window and streaming over the same `CubicLevel` (the cubes on
//! disk, the generator and the population are shared; the windows are
//! disjoint, so no cube is loaded or simulated twice). `cubic_shards::plan`
//! says who shares, `cubic_shards::settle` keeps that to the moves that can be
//! made now, and this module carries it out each server tick
//! (`follow_cubic_window`): shards that meet merge (the one that closes saves
//! all it holds, the other slides over it), a player that left the others'
//! range splits into a new shard (the old one shrinks first), and a player
//! that changes shard gets the level again like a dimension change (`Respawn`,
//! the new window, its columns and cubes). The saved window
//! (`horizon_cubic.json`) is the primary shard's, the one a world opens with
//! (shard 0, which also owns the level's forced chunks, attachments and
//! raids). The columns' sky tops and the marker cube's structure data are
//! the `CubicLevel`'s, read from the store by whichever shard loads the
//! column.
//!
//! A slide never reloads the level. The cubes that enter come from the
//! prefetch ring (`cubic_prefetch`: the next two layers beyond the window,
//! ready in memory for every loaded column), the ones that leave are saved
//! with their entities, block entities and ticks, and the sections both
//! windows hold stay as they are (`World::relayout`). The hub then tells the
//! players the window and sends them only the entering cubes
//! (`Hub::cubic_window_changed`).
use super::levels::LevelId;
use super::persist::EntityScope;
use super::{Dim, Game};
use horizon_client::cubic::{self, LoadedCube};
use horizon_client::cubic_shards::{self, Existing, Group, Member, Range};
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

/// What the server remembers of a shard's window between ticks.
#[derive(Default)]
pub(super) struct SlideState {
    /// Ticks a due slide has waited for the ring.
    waiting: u32,
    /// Ticks since the ring was aimed.
    since_retarget: u32,
}

/// A player in the cubic level: its cube, its vertical distance and column.
struct Viewer {
    conn: u32,
    cube: i32,
    vertical: u8,
    column: (i32, i32),
    shard: u16,
}

impl Game<'_> {
    /// The Overworld's shards and the windows (cubes) they hold now.
    fn overworld_shards(&self) -> Vec<Existing> {
        let mut shards: Vec<Existing> = self
            .levels
            .parked_shards(Dim::Overworld)
            .into_iter()
            .filter_map(|id| {
                let world = self.levels.parked_shard_world(Dim::Overworld, id)?;
                Some(Existing {
                    id,
                    window: cubic::window_cube_range(world.shape),
                })
            })
            .collect();
        if self.dim == Dim::Overworld {
            shards.push(Existing {
                id: self.shard,
                window: cubic::window_cube_range(self.world.shape),
            });
        }
        shards.sort_by_key(|s| s.id);
        shards
    }

    /// The cubic level's shards follow their players: each player is in the
    /// shard whose window covers it (`cubic_shards::plan`), shards merge and
    /// split, and each shard's window slides towards the cubes its players
    /// want loaded (`step_shard`); the ring is kept aimed at the layers beyond
    /// each window meanwhile.
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
                    conn: g.conn,
                    cube: (g.position[1].floor() as i32) >> 4,
                    vertical: g.vertical,
                    column: g.chunk(),
                    shard: g.shard,
                })
                .collect()
        });
        let existing = self.overworld_shards();
        let members: Vec<Member> = viewers
            .iter()
            .map(|v| Member {
                key: v.conn,
                cube: v.cube,
                range: cubic::wanted_cubes(v.cube, v.vertical),
                shard: v.shard,
            })
            .collect();
        let planned = cubic_shards::plan(&members, &existing);
        let groups = cubic_shards::settle(planned.groups, &members, &existing);
        let back = self.here();
        // A shard that cannot be carried out this tick is tried again the next
        // (the plan is made from where the players are): it never stops the server.
        if let Err(e) = self.carry_out_shards(&groups, &viewers, existing) {
            log::error!("Horizon: the cubic shards could not follow their players: {e}");
        }
        // The level that was in place may have closed (a merge).
        let back = match back.shard {
            Some(shard) if !self.shard_exists(back.dim, shard) => LevelId::shard(back.dim, 0),
            _ => back,
        };
        self.enter_level(None, back)
    }

    /// Carries out the groups: in order, the merges (the shard that closes is
    /// saved whole, its survivor slides over it), every other existing shard
    /// sliding towards its window, the new shards, the players moving to
    /// their shard. Shards left without players close after a few ticks
    /// (`close_idle_levels`).
    fn carry_out_shards(
        &mut self,
        groups: &[Group],
        viewers: &[Viewer],
        existing: Vec<Existing>,
    ) -> horizon_client::Result<()> {
        let mut windows: HashMap<u16, Range> = existing.iter().map(|e| (e.id, e.window)).collect();
        let viewer = |conn: u32| viewers.iter().find(|v| v.conn == conn);
        // The shard each player ends in.
        let mut target: HashMap<u32, u16> = viewers.iter().map(|v| (v.conn, v.shard)).collect();

        // 1. Merges: the shards that join another save everything they hold
        // and close; their players are moved below.
        for g in groups {
            for &gone in &g.absorbed {
                self.close_shard(gone)?;
                windows.remove(&gone);
            }
        }
        // 2. Every shard that stays slides towards its window.
        for g in groups.iter().filter(|g| g.shard.is_some()) {
            let shard = g.shard.unwrap_or_default();
            if !self.shard_exists(Dim::Overworld, shard) {
                continue;
            }
            let incoming = g
                .members
                .iter()
                .any(|k| viewer(*k).is_some_and(|v| v.shard != shard));
            let cubes: Vec<i32> = g
                .members
                .iter()
                .filter_map(|k| viewer(*k).map(|v| v.cube))
                .collect();
            let keep = (
                cubes.iter().copied().min().unwrap_or(g.window.0),
                cubes.iter().copied().max().unwrap_or(g.window.1),
            );
            let others: Vec<Range> = windows
                .iter()
                .filter(|(id, _)| **id != shard)
                .map(|(_, w)| *w)
                .collect();
            let here: Vec<&Viewer> = viewers.iter().filter(|v| v.shard == shard).collect();
            let now = self.step_shard(shard, g.window, keep, &here, &others, incoming)?;
            windows.insert(shard, now);
            for &k in &g.members {
                // Players move once the window holds their cube.
                let cube = viewer(k).map_or(keep.0, |v| v.cube);
                if (now.0..=now.1).contains(&cube) {
                    target.insert(k, shard);
                }
            }
        }
        // 3. New shards.
        for g in groups.iter().filter(|g| g.shard.is_none()) {
            let cubes: Vec<i32> = g
                .members
                .iter()
                .filter_map(|k| viewer(*k).map(|v| v.cube))
                .collect();
            let keep = (
                cubes.iter().copied().min().unwrap_or(g.window.0),
                cubes.iter().copied().max().unwrap_or(g.window.1),
            );
            let others: Vec<Range> = windows.values().copied().collect();
            let Some(window) = cubic_shards::clip_window(g.window, &others, keep) else {
                continue;
            };
            let id = (1..)
                .find(|id| !windows.contains_key(id) && !self.shard_exists(Dim::Overworld, *id))
                .unwrap_or(1);
            match self.create_shard(id, window) {
                Ok(()) => {
                    windows.insert(id, window);
                    for &k in &g.members {
                        target.insert(k, id);
                    }
                }
                Err(e) => {
                    log::error!("Horizon: no cubic shard for the players above or below: {e}")
                }
            }
        }
        // 4. Players change shard.
        for v in viewers {
            if let Some(&to) = target.get(&v.conn)
                && to != v.shard
            {
                self.move_guest_to_shard(v.conn, to);
            }
        }
        Ok(())
    }

    /// One shard's step towards `wanted` (the cubes its players want loaded):
    /// it slides when due, else the ring is kept aimed. `keep` is the cubes
    /// (first, last) its players are in and `others` the other shards' windows,
    /// which the new window must not meet. With `force` the window goes to
    /// `wanted` at once (players are moving in). Returns the window it holds
    /// after.
    fn step_shard(
        &mut self,
        shard: u16,
        wanted: Range,
        keep: Range,
        viewers: &[&Viewer],
        others: &[Range],
        force: bool,
    ) -> horizon_client::Result<Range> {
        self.enter_level(None, LevelId::shard(Dim::Overworld, shard))?;
        let due = force || cubic::needs_new_window(self.world.shape, wanted);
        self.cubic_slide.since_retarget += 1;
        if !due {
            self.cubic_slide.waiting = 0;
            if self.cubic_slide.since_retarget >= RETARGET_EVERY {
                self.aim_ring(viewers);
            }
            return Ok(cubic::window_cube_range(self.world.shape));
        }
        self.slide_cubic_window(wanted, keep, viewers, others, force)?;
        Ok(cubic::window_cube_range(self.world.shape))
    }

    /// Aims the level in place's prefetch ring at the layers beyond the window
    /// of the columns it holds, nearest the players first.
    fn aim_ring(&mut self, viewers: &[&Viewer]) {
        self.cubic_slide.since_retarget = 0;
        let Some(level) = self.save.cubic.get() else {
            return;
        };
        let centres: Vec<(i32, i32)> = viewers.iter().map(|v| v.column).collect();
        level.retarget_shard_ring(
            self.shard,
            self.world.shape.column(),
            &centres,
            self.world.chunks.keys().copied(),
        );
    }

    /// Slides the level in place towards the cubes `wanted`: at most
    /// `cubic::RING` layers at each end per slide, unless a player is outside
    /// the window or the two do not overlap, which is a jump (and `force`
    /// is one). The window shrinks at once, keeps off `others` and holds the
    /// cubes `keep`. It waits (up to `RING_WAIT` ticks) until the ring has
    /// the entering cubes of every loaded column. In order:
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
    /// 6. The entering cubes are lit, and the light that spreads out of them
    ///    into the kept cubes beside them (not every column whole).
    /// 7. The hub tells the players and queues the entering cubes for them.
    fn slide_cubic_window(
        &mut self,
        wanted: Range,
        keep: Range,
        viewers: &[&Viewer],
        others: &[Range],
        force: bool,
    ) -> horizon_client::Result<()> {
        let Some(level) = self.save.cubic.get().cloned() else {
            return Ok(());
        };
        let old_shape = self.world.shape;
        let have = cubic::window_cube_range(old_shape);
        let inside = viewers.iter().all(|v| (have.0..=have.1).contains(&v.cube));
        let next = if force || (wanted.0 >= have.0 && wanted.1 <= have.1) || !inside {
            wanted
        } else {
            cubic::next_window(have, wanted)
        };
        // Not into another shard's cubes, and never off the players'.
        let Some(next) = cubic_shards::clip_window(next, others, keep) else {
            return Ok(());
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
        let redo: HashSet<(i32, i32)> = if entering.iter().any(|&cy| level.is_vanilla(cy)) {
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
        if !ready && inside && !jump && !force && self.cubic_slide.waiting < RING_WAIT {
            // The margin keeps the players in loaded cubes meanwhile.
            self.cubic_slide.waiting += 1;
            self.aim_ring(viewers);
            return Ok(());
        }
        self.cubic_slide.waiting = 0;
        log::info!(
            "Horizon: cubic window (shard {}) {}..={} -> {}..={} ({} entering, {} columns)",
            self.shard,
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
        self.set_shard_window(&level, new_shape);
        if self.shard == cubic_shards::PRIMARY {
            *self.level_shapes.get_mut(self.dim.id()) = new_shape;
            if let Err(e) = cubic::save_window(&self.world_root, new_shape.min_y, new_shape.height)
            {
                log::warn!("Horizon: the cubic window was not recorded: {e}");
            }
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

        // 6. Light: the entering layers, and what spreads out of them into the
        // kept cubes beside them (`Streaming::relight_slide`), not every
        // column whole.
        let kept_light = self.streaming.relight_slide(&mut self.world, &entering);

        // 7. The players: the window, the kept sections whose light changed
        // (the entering cubes are lit now, and go out with their light).
        if let Some(h) = self.lan.as_mut() {
            h.hub.cubic_window_changed(self.dim.id(), new_shape);
            h.hub
                .cubic_light_changed(self.dim.id(), &self.world, &kept_light);
        }
        self.aim_ring(viewers);
        Ok(())
    }

    /// Moves the window of the shard in place (the world's shape, the cell its
    /// terrain loads into and the level's window for entity saves).
    fn set_shard_window(
        &mut self,
        level: &Arc<cubic::CubicLevel>,
        shape: horizon_client::chunk::LevelShape,
    ) {
        level.set_window(shape);
        let cell = self.shard_window.clone().or_else(|| level.primary_window());
        if let Some(cell) = cell {
            *cell.lock().unwrap() = shape.column();
        }
    }

    /// Opens shard `id` of the Overworld around `window` (cubes), with no
    /// players yet.
    fn create_shard(&mut self, id: u16, window: Range) -> horizon_client::Result<()> {
        let shape = cubic::window_of_cubes(self.level_shapes.overworld, window);
        let parked = self.open_shard(id, shape)?;
        self.levels.insert_shard(parked);
        log::info!(
            "Horizon: cubic shard {id} opened at cubes {}..={}",
            window.0,
            window.1
        );
        Ok(())
    }

    /// Saves everything shard `id` holds (blocks, entities, block entities and
    /// ticks go to the cubes' records) and closes it. The primary never closes.
    pub(super) fn close_shard(&mut self, id: u16) -> horizon_client::Result<()> {
        if id == cubic_shards::PRIMARY || !self.shard_exists(Dim::Overworld, id) {
            return Ok(());
        }
        let back = if self.dim == Dim::Overworld && self.shard == id {
            LevelId::shard(Dim::Overworld, cubic_shards::PRIMARY)
        } else {
            self.here()
        };
        self.enter_level(None, LevelId::shard(Dim::Overworld, id))?;
        let chunks: Vec<_> = self.world.chunks.values().cloned().collect();
        self.native_chunks_changed(&chunks);
        self.save_level(true)?;
        for uuid in self.persist.uuids.values() {
            self.entity_attachments.remove(uuid);
        }
        if let Some(level) = self.save.cubic.get().cloned() {
            let (first, last) = cubic::window_cube_range(self.world.shape);
            for &(x, z) in self.world.chunks.keys() {
                for cy in first..=last {
                    level.release_cube(SectionPos::new(x, cy, z));
                }
            }
            level.drop_ring_target(id);
        }
        self.enter_level(None, back)?;
        self.levels.remove_shard(Dim::Overworld, id);
        if let Some(h) = self.lan.as_mut() {
            h.hub.forget_level(Dim::Overworld.id(), id);
        }
        log::info!("Horizon: cubic shard {id} closed");
        Ok(())
    }

    /// Player `conn` goes to shard `to`: the level changes under it as for a
    /// dimension change (`Respawn` keeping its data, the new window, then its
    /// columns and cubes), at the same place. A vehicle it rides stays behind
    /// in its shard (saved with its cube when that leaves its window).
    fn move_guest_to_shard(&mut self, conn: u32, to: u16) {
        let Some(mut host) = self.lan.take() else {
            return;
        };
        let back = self.here();
        if let Some(i) = host.hub.guests.iter().position(|g| g.conn == conn)
            && self
                .enter_level(Some(&mut host), LevelId::shard(Dim::Overworld, to))
                .is_ok()
        {
            let g = &mut host.hub.guests[i];
            g.vehicle = None;
            let (at, yaw, pitch) = (g.position, g.yaw, g.pitch);
            let cooldown = host.play.get(&conn).map_or(0, |p| p.portal.cooldown);
            let ctx = self.server_ctx();
            host.hub
                .change_level(i, at, yaw, Some(pitch), false, cooldown, &ctx);
        }
        if let Err(e) = self.enter_level(Some(&mut host), back) {
            log::error!("could not reopen the level in place: {e}");
        }
        self.lan = Some(host);
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
