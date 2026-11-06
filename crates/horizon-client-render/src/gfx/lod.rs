//! Far meshes share the terrain arenas and passes, with their actual bounds
//! rather than the 16³ boxes used by the near-section residency walk.
use super::{Frustum, Renderer, SectionGpu, cube_far, cube_gen::GpuBackend};
use crate::{
    fast_hash::{FxHashMap, FxHashSet},
    lod::{
        MAX_LEVEL, TileKey,
        cube::{CubeKey, GpuBatch},
    },
    mesher::SectionMesh,
};
use glam::{DVec3, Vec3};
use std::{
    collections::VecDeque,
    time::{Duration, Instant},
};

/// Finished tile meshes not uploaded yet, the tiles they replace (removed
/// once every mesh is in) and the columns they leave out.
#[derive(Default)]
pub(super) struct Pending {
    meshes: FxHashMap<TileKey, SectionMesh>,
    /// The order they go up in: as handed over (`Lod::update`: finer tiles
    /// first, so no tile clipped around a finer one is drawn before it).
    order: VecDeque<TileKey>,
    removed: FxHashSet<TileKey>,
    cutout: Option<FxHashSet<(i32, i32)>>,
}

impl Pending {
    /// Takes in the next job, with the same result as applying both in
    /// turn: newer meshes replace waiting ones, tiles meshed again are not
    /// removed, and tiles removed now drop their waiting meshes.
    fn add(
        &mut self,
        meshes: Vec<(TileKey, SectionMesh)>,
        removed: Vec<TileKey>,
        cutout: Option<FxHashSet<(i32, i32)>>,
    ) {
        for (key, mesh) in meshes {
            self.removed.remove(&key);
            if self.meshes.insert(key, mesh).is_none() {
                self.order.push_back(key);
            }
        }
        for key in removed {
            self.meshes.remove(&key);
            self.removed.insert(key);
        }
        if cutout.is_some() {
            self.cutout = cutout;
        }
    }
}

/// A far tile of either kind: a column tile of the surface model, or a node of
/// a cubic level's octree (`lod::cube`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(super) enum FarKey {
    Tile(TileKey),
    Cube(CubeKey),
}

impl FarKey {
    pub fn level(self) -> u8 {
        match self {
            Self::Tile(key) => key.level,
            Self::Cube(key) => key.level,
        }
    }
    pub fn origin(self) -> [i32; 3] {
        match self {
            Self::Tile(key) => key.origin(),
            Self::Cube(key) => key.origin(),
        }
    }
}

pub(super) fn far_tile<'a>(
    tiles: &'a std::collections::HashMap<TileKey, Tile>,
    cubes: &'a std::collections::HashMap<CubeKey, Tile>,
    key: FarKey,
) -> Option<&'a Tile> {
    match key {
        FarKey::Tile(key) => tiles.get(&key),
        FarKey::Cube(key) => cubes.get(&key),
    }
}

pub(super) fn far_tile_mut<'a>(
    tiles: &'a mut std::collections::HashMap<TileKey, Tile>,
    cubes: &'a mut std::collections::HashMap<CubeKey, Tile>,
    key: FarKey,
) -> Option<&'a mut Tile> {
    match key {
        FarKey::Tile(key) => tiles.get_mut(&key),
        FarKey::Cube(key) => cubes.get_mut(&key),
    }
}

/// Finished node meshes not uploaded yet and the nodes to retire once they are in.
#[derive(Default)]
pub(super) struct CubePending {
    meshes: FxHashMap<CubeKey, SectionMesh>,
    order: VecDeque<CubeKey>,
    removed: FxHashSet<CubeKey>,
}

impl CubePending {
    /// As [`Pending::add`]: newer meshes replace waiting ones, nodes meshed
    /// again are not removed, nodes removed drop their waiting meshes.
    fn add(&mut self, meshes: Vec<(CubeKey, SectionMesh)>, removed: Vec<CubeKey>) {
        for (key, mesh) in meshes {
            self.removed.remove(&key);
            if self.meshes.insert(key, mesh).is_none() {
                self.order.push_back(key);
            }
        }
        for key in removed {
            self.meshes.remove(&key);
            self.removed.insert(key);
        }
    }
}

pub(super) struct Tile {
    pub mesh: SectionGpu,
    bounds: Bounds,
    /// When it began to fade in; none while the rest of its job uploads.
    shown: Option<Instant>,
    /// When its replacements began to fade in; it goes once they have.
    dying: Option<Instant>,
    /// Dying with nothing to replace it (out of range): it fades out.
    alone: bool,
    /// Its mesh before near chunks took some of its columns, drawn in its
    /// place until they have faded in (from the job's end, the instant), so
    /// no hole shows through them.
    previous: Option<(SectionGpu, Option<Instant>)>,
    /// Its slot in occlusion culling's results (`cull::TILE`), kept while
    /// meshes of the same tile replace one another.
    pub cull_slot: u32,
}

/// How long a tile takes to fade in over the ones it replaces.
pub(super) const FADE: Duration = Duration::from_millis(600);

/// A tile or section fading in: 0 drawn whole, else 1 + `level` * 129 +
/// `shown`, `shown` 1 to 63 sixty-fourths of `FADE` (`fade_out_code`: 65 to
/// 128, fading out). Chunk by chunk (columns of its own grid, wider for
/// coarse tiles: 2^`level` blocks), each starting in turn over the first half
/// and fading over the second, as a fine ordered pattern fixed to the terrain
/// (terrain.wgsl's `faded`). It is drawn a hair toward the camera, the newest
/// most, over the solid tiles it replaces, which stay until it is whole
/// (`upload_lods`): no pixel fought over, no hole. None: nothing to draw yet.
pub(super) fn fade_code(shown: Option<Instant>, level: u8, now: Instant) -> Option<u32> {
    let faded = now.saturating_duration_since(shown?).as_secs_f32() / FADE.as_secs_f32();
    let shown = (faded * 64.0) as u32 + 1;
    Some(if shown >= 64 {
        0
    } else {
        1 + u32::from(level) * 129 + shown
    })
}
/// A section past the render distance that far terrain is coming in under:
/// whole, a fading-out code at its start (terrain.wgsl's `fade_depth` draws
/// leaving sections in front of far terrain).
pub(super) const HELD: u32 = 1 + 64;
/// `fade_code` for one leaving since `since`, chunk by chunk as it came, over
/// what stays behind it; once `FADE` has passed it draws nothing.
pub(super) fn fade_out_code(since: Instant, level: u8, now: Instant) -> u32 {
    let faded = now.saturating_duration_since(since).as_secs_f32() / FADE.as_secs_f32();
    1 + u32::from(level) * 129 + 64 + ((faded * 64.0) as u32 + 1).min(64)
}

#[derive(Clone, Copy)]
struct Bounds {
    min: Vec3,
    max: Vec3,
}

impl Bounds {
    fn mesh(mesh: &SectionMesh) -> Option<Self> {
        let [min, max] = mesh.bounds?;
        Some(Self { min, max })
    }

    /// In view: the plan alone decides which tiles are in range (a circle),
    /// so a tile leaving it fades out (`upload_lods`). Cut here at the range,
    /// tiles popped out as the player moved, before the plan dropped them.
    fn visible(self, origin: DVec3, frustum: &Frustum) -> Option<u64> {
        let min = origin + self.min.as_dvec3();
        let max = origin + self.max.as_dvec3();
        if !frustum.intersects(min.as_vec3(), max.as_vec3()) {
            return None;
        }
        Some(((min + max) * 0.5).length_squared().to_bits())
    }
}

/// A tile's instance: like a section's, with the UV step of merged faces.
/// A fading tile's step is negated past its `fade_code`, which the shaders
/// read back as the whole part (they take the step as `fract(abs(w))`).
pub(super) fn instance(origin: [f32; 3], fade: u32) -> [f32; 4] {
    let step = 1.0 / crate::mesher::TILED_UV_STEPS;
    [
        origin[0],
        origin[1],
        origin[2],
        if fade == 0 {
            step
        } else {
            -(fade as f32 + step)
        },
    ]
}

pub(super) struct Visible {
    pub key: FarKey,
    pub distance: u64,
    pub origin: [f32; 3],
    pub fade: u32,
    /// Replaced, standing until its replacements are whole: its water goes
    /// after theirs (`Renderer::render`'s water tiers), so depth leaves it only
    /// where they have none yet.
    pub old: bool,
}

impl Tile {
    /// The mesh drawn this frame: the previous one while it stands in.
    pub fn drawn(&self) -> &SectionGpu {
        self.previous.as_ref().map_or(&self.mesh, |(mesh, _)| mesh)
    }
    pub fn drawn_mut(&mut self) -> &mut SectionGpu {
        match &mut self.previous {
            Some((mesh, _)) => mesh,
            None => &mut self.mesh,
        }
    }
    /// The box occlusion culling tests (`cull::Cull`), around its origin: its
    /// mesh's bounds grown by a block, while that mesh is the one drawn
    /// whole (no stand-in, no fade). A fading tile's dither moves its depth
    /// toward the camera (terrain.wgsl's `fade_depth`), out of the box.
    pub fn cull_bounds(&self, fade: u32) -> Option<[[f32; 3]; 2]> {
        (fade == 0 && self.previous.is_none()).then(|| {
            [
                (self.bounds.min - 1.0).to_array(),
                (self.bounds.max + 1.0).to_array(),
            ]
        })
    }
    /// Both meshes it holds in the arena: its own and the stand-in, if any.
    pub fn meshes_mut(&mut self) -> impl Iterator<Item = &mut SectionGpu> {
        std::iter::once(&mut self.mesh).chain(self.previous.as_mut().map(|(mesh, _)| mesh))
    }
    pub fn visible(
        &self,
        key: FarKey,
        camera: DVec3,
        frustum: &Frustum,
        now: Instant,
    ) -> Option<Visible> {
        let fade = match self.dying {
            Some(since) if self.alone => fade_out_code(since, key.level(), now),
            _ => fade_code(self.shown, key.level(), now)?,
        };
        let origin = DVec3::from(key.origin().map(f64::from)) - camera;
        Some(Visible {
            key,
            distance: self.bounds.visible(origin, frustum)?,
            origin: origin.as_vec3().to_array(),
            fade,
            old: self.dying.is_some() || self.previous.is_some(),
        })
    }
}

impl Renderer {
    /// A tile meshed again (finer detail) takes its place at once, but for
    /// one meshed around near chunks or finer tiles that are new (`keep`):
    /// its old mesh stands in while they fade in. A new one waits for the rest of its job
    /// (`upload_lods`), then fades in.
    pub fn upload_lod(&mut self, key: TileKey, mesh: &SectionMesh, keep: bool) {
        if let Some(copies) = &mut self.capture_copies {
            copies.tile(key, mesh);
        }
        let old_slot = self.lods.get(&key).map(|tile| tile.cull_slot);
        let (shown, previous) = match self.lods.remove(&key) {
            Some(old) if old.dying.is_none() => match old.previous {
                Some(previous) => {
                    self.release_mesh(old.mesh);
                    (old.shown, Some(previous))
                }
                None if keep => (old.shown, Some((old.mesh, None))),
                None => {
                    self.release_mesh(old.mesh);
                    (old.shown, None)
                }
            },
            Some(old) => {
                self.release_tile(old);
                (None, None)
            }
            None => (None, None),
        };
        match Bounds::mesh(mesh).zip(self.upload_mesh(mesh)) {
            Some((bounds, mesh)) => {
                let cull_slot = old_slot.unwrap_or_else(|| {
                    self.lod_cull_free.pop().unwrap_or_else(|| {
                        self.lod_cull_next += 1;
                        self.lod_cull_next - 1
                    })
                });
                self.lods.insert(
                    key,
                    Tile {
                        mesh,
                        bounds,
                        shown,
                        dying: None,
                        alone: false,
                        previous,
                        cull_slot,
                    },
                );
            }
            None => {
                if let Some((previous, _)) = previous {
                    self.release_mesh(previous);
                }
                self.lod_cull_free.extend(old_slot);
            }
        }
    }

    fn release_tile(&mut self, tile: Tile) {
        self.release_mesh(tile.mesh);
        if let Some((previous, _)) = tile.previous {
            self.release_mesh(previous);
        }
    }

    pub fn remove_lod(&mut self, key: TileKey) {
        if let Some(copies) = &mut self.capture_copies {
            copies.forget_tile(key);
        }
        if let Some(tile) = self.lods.remove(&key) {
            self.lod_cull_free.push(tile.cull_slot);
            self.release_tile(tile);
        }
    }

    pub fn clear_lods(&mut self) {
        if let Some(copies) = &mut self.capture_copies {
            copies.clear_tiles();
        }
        self.lod_pending = None;
        // No far terrain left to fade out over.
        for (key, since) in std::mem::take(&mut self.lod_leaving) {
            if self
                .sections
                .find(key)
                .is_some_and(|s| s.leaving == Some(since))
            {
                self.release_section(key);
            }
        }
        // Cube tiles share the culling slots: they keep theirs.
        let slots: Vec<u32> = self.lods.values().map(|tile| tile.cull_slot).collect();
        for (_, tile) in std::mem::take(&mut self.lods) {
            self.release_tile(tile);
        }
        if self.lod_cubes.is_empty() {
            self.lod_cull_free.clear();
            self.lod_cull_next = 0;
        } else {
            self.lod_cull_free.extend(slots);
        }
        self.set_lod_cutout(Default::default());
    }

    /// A node meshed again keeps its slot and fade; a new one waits for the
    /// rest of its job (`upload_cube_lods`) before it fades in.
    fn upload_cube_lod(&mut self, key: CubeKey, mesh: &SectionMesh) {
        let (old_slot, shown) = match self.lod_cubes.remove(&key) {
            Some(old) => {
                let kept = (old.cull_slot, old.shown.filter(|_| old.dying.is_none()));
                self.release_tile(old);
                (Some(kept.0), kept.1)
            }
            None => (None, None),
        };
        match Bounds::mesh(mesh).zip(self.upload_mesh(mesh)) {
            Some((bounds, mesh)) => {
                let cull_slot = old_slot.unwrap_or_else(|| {
                    self.lod_cull_free.pop().unwrap_or_else(|| {
                        self.lod_cull_next += 1;
                        self.lod_cull_next - 1
                    })
                });
                self.lod_cubes.insert(
                    key,
                    Tile {
                        mesh,
                        bounds,
                        shown,
                        dying: None,
                        alone: false,
                        previous: None,
                        cull_slot,
                    },
                );
            }
            None => self.lod_cull_free.extend(old_slot),
        }
    }

    /// Whether far cube terrain can be meshed and drawn on the GPU here: the
    /// device runs compute shaders and draws indirect with a first instance.
    pub fn gpu_far_available(&self) -> bool {
        self.multi_draw
            && self.adapter.get_downlevel_capabilities().flags.contains(
                wgpu::DownlevelFlags::COMPUTE_SHADERS | wgpu::DownlevelFlags::VERTEX_STORAGE,
            )
            && self.device.limits().max_storage_buffers_per_shader_stage >= 6
    }

    /// Makes the GPU far terrain ready for nodes of `backend`'s pool, which
    /// must be on this renderer's device. `states` the registry's block state
    /// count, `air` its air state. False if it cannot run (stay on the CPU).
    pub fn use_gpu_cube_lods(&mut self, backend: &GpuBackend, states: usize, air: u32) -> bool {
        if self.cube_far.is_some() {
            return self
                .cube_far
                .as_ref()
                .is_some_and(|far| *far.device() == backend.device);
        }
        if !self.gpu_far_available() || backend.device != self.device {
            return false;
        }
        let setup = cube_far::DrawSetup {
            bindings: &self.terrain_layout,
            format: self.frame_format,
            sprite_constants: self.terrain_sprites,
            counts: self
                .device
                .features()
                .contains(wgpu::Features::MULTI_DRAW_INDIRECT_COUNT)
                && self.info.backend != wgpu::Backend::Dx12,
        };
        self.cube_far = cube_far::CubeFar::new(backend, states, air, &setup);
        if let Some(far) = &self.cube_far {
            self.surface_debug = super::surface_tile::DebugTiles::from_env(backend, far, &setup);
        }
        self.cube_far.is_some()
    }

    /// Queues the nodes `CubeLod::take_gpu_nodes` returned for meshing.
    pub fn queue_gpu_cube_lods(&mut self, batch: GpuBatch) {
        if let Some(far) = &mut self.cube_far {
            far.set_materials(&batch.materials, batch.biomes.as_deref());
            far.queue(batch.nodes, Vec::new());
        }
    }

    /// The GPU far terrain's counters, if it runs.
    pub fn cube_far_stats(&self) -> Option<cube_far::FarStats> {
        self.cube_far.as_ref().map(|far| far.stats())
    }

    pub fn remove_cube_lod(&mut self, key: CubeKey) {
        if let Some(tile) = self.lod_cubes.remove(&key) {
            self.lod_cull_free.push(tile.cull_slot);
            self.release_tile(tile);
        }
    }

    /// Drops every cube tile and what waits to upload.
    pub fn clear_cube_lods(&mut self) {
        self.cube_pending = None;
        if let Some(far) = &mut self.cube_far {
            far.clear();
        }
        for (_, tile) in std::mem::take(&mut self.lod_cubes) {
            self.lod_cull_free.push(tile.cull_slot);
            self.release_tile(tile);
        }
        if self.lods.is_empty() {
            self.lod_cull_free.clear();
            self.lod_cull_next = 0;
        }
    }

    /// Queues what `CubeLod::update` and `CubeLod::removed` returned for
    /// `upload_cube_lods`, joining a job still uploading.
    pub fn queue_cube_lods(&mut self, meshes: Vec<(CubeKey, SectionMesh)>, removed: Vec<CubeKey>) {
        if meshes.is_empty() && removed.is_empty() {
            return;
        }
        if let Some(far) = &mut self.cube_far {
            far.queue(Vec::new(), removed.clone());
        }
        self.cube_pending
            .get_or_insert_default()
            .add(meshes, removed);
    }

    /// Uploads queued node meshes for about `budget` (at least one). Once all
    /// are in they fade in together, and the nodes they replace stand whole
    /// for `FADE` and go: nothing opens a hole while the new ones come in.
    pub fn upload_cube_lods(&mut self, budget: Duration) {
        let start = Instant::now();
        if let Some(far) = &mut self.cube_far {
            far.upload();
        }
        let faded: Vec<_> = self
            .lod_cubes
            .iter()
            .filter(|(_, tile)| tile.dying.is_some_and(|d| start - d >= FADE))
            .map(|(&key, _)| key)
            .collect();
        for key in faded {
            self.remove_cube_lod(key);
        }
        while let Some(pending) = &mut self.cube_pending {
            let next = std::iter::from_fn(|| pending.order.pop_front())
                .find(|key| pending.meshes.contains_key(key));
            let Some(key) = next else {
                let pending = self.cube_pending.take().expect("pending nodes");
                for tile in self.lod_cubes.values_mut() {
                    tile.shown.get_or_insert(start);
                }
                for key in pending.removed {
                    if let Some(tile) = self.lod_cubes.get_mut(&key)
                        && tile.dying.is_none()
                    {
                        tile.dying = Some(start);
                    }
                }
                return;
            };
            let mesh = pending.meshes.remove(&key).expect("pending node");
            self.upload_cube_lod(key, &mesh);
            if start.elapsed() >= budget {
                return;
            }
        }
    }

    /// Queues a finished mesh job (`Lod::update`, `Lod::removed`,
    /// `Lod::take_cutout`) for `upload_lods`, joining one still uploading
    /// (`Pending::add`).
    pub fn queue_lods(
        &mut self,
        meshes: Vec<(TileKey, SectionMesh)>,
        removed: Vec<TileKey>,
        cutout: Option<FxHashSet<(i32, i32)>>,
    ) {
        if meshes.is_empty() && removed.is_empty() && cutout.is_none() {
            return;
        }
        self.lod_pending
            .get_or_insert_default()
            .add(meshes, removed, cutout);
    }

    /// Uploads queued tile meshes for about `budget` (at least one), then,
    /// once all are in, fades them in over the tiles they replace.
    pub fn upload_lods(&mut self, budget: Duration) {
        let start = Instant::now();
        if self.lod_pending.is_some() {
            // Far tiles fill the arenas too: large translucent pages wait for them.
            self.shrink_gate.wait(start, super::SHRINK_SETTLE);
        }
        self.release_left(start);
        let faded: Vec<_> = self
            .lods
            .iter()
            .filter(|(_, tile)| tile.dying.is_some_and(|d| start - d >= FADE))
            .map(|(&key, _)| key)
            .collect();
        for key in faded {
            self.remove_lod(key);
        }
        let stood_in: Vec<_> = self
            .lods
            .values_mut()
            .filter_map(|tile| match tile.previous {
                Some((_, Some(since))) if start - since >= FADE => tile.previous.take(),
                _ => None,
            })
            .collect();
        for (previous, _) in stood_in {
            self.release_mesh(previous);
        }
        // The columns near chunks take in this job, and the tiles holding the
        // job's new finer ones: a tile meshed again around them keeps its
        // old mesh until they are in, so no hole shows through them.
        let mut around_new = FxHashSet::default();
        let new = self
            .lod_pending
            .iter()
            .flat_map(|pending| pending.meshes.keys())
            .filter(|key| !self.lods.contains_key(key))
            .chain(
                self.lods
                    .iter()
                    .filter(|(_, tile)| tile.shown.is_none())
                    .map(|(key, _)| key),
            );
        for &key in new {
            let mut key = key;
            while key.level < MAX_LEVEL && around_new.insert(key.parent()) {
                key = key.parent();
            }
        }
        let taken: Vec<_> = self
            .lod_pending
            .iter()
            .flat_map(|pending| pending.cutout.iter().flatten())
            .filter(|column| !self.lod_cutout.contains(column))
            .copied()
            .collect();
        while let Some(pending) = &mut self.lod_pending {
            // Keys meshed again keep their place; removed ones are skipped.
            let next = std::iter::from_fn(|| pending.order.pop_front())
                .find(|key| pending.meshes.contains_key(key));
            let Some(key) = next else {
                let pending = self.lod_pending.take().expect("pending tiles");
                // A tile fades in over the ones it replaces. One over nothing
                // shows whole: its hidden chunks would open holes to the sky
                // or the sea bed (dark patches, as far terrain first loads).
                let replaced: FxHashSet<TileKey> = pending
                    .removed
                    .iter()
                    .copied()
                    .filter(|key| self.lods.contains_key(key))
                    .collect();
                let mut above_replaced = FxHashSet::default();
                for &key in &replaced {
                    let mut key = key;
                    while key.level < MAX_LEVEL && above_replaced.insert(key.parent()) {
                        key = key.parent();
                    }
                }
                let over_replaced = |key: TileKey| {
                    let mut ancestor = key;
                    replaced.contains(&key)
                        || above_replaced.contains(&key)
                        || (key.level..MAX_LEVEL).any(|_| {
                            ancestor = ancestor.parent();
                            replaced.contains(&ancestor)
                        })
                };
                // A removed tile no new tile overlaps, and no near chunk takes
                // a column of, leaves the plan (out of range): it fades out.
                // Over near chunks fading in it stays whole: two patterns
                // fading at once left holes between them.
                // Two tiles overlap when one is the other or its ancestor:
                // `kept` holds the new tiles and the near chunks' columns
                // (level 0), `above` their ancestors.
                let mut kept: FxHashSet<TileKey> = self
                    .lods
                    .iter()
                    .filter(|(_, tile)| tile.shown.is_none())
                    .map(|(&key, _)| key)
                    .collect();
                let cut = pending.cutout.as_ref().unwrap_or(&self.lod_cutout);
                kept.extend(cut.iter().map(|&(x, z)| TileKey { x, z, level: 0 }));
                let mut above = FxHashSet::default();
                for &key in &kept {
                    let mut key = key;
                    while key.level < MAX_LEVEL && above.insert(key.parent()) {
                        key = key.parent();
                    }
                }
                let alone = |key: TileKey| {
                    let mut ancestor = key;
                    !kept.contains(&key)
                        && !above.contains(&key)
                        && !(key.level..MAX_LEVEL).any(|_| {
                            ancestor = ancestor.parent();
                            kept.contains(&ancestor)
                        })
                };
                let whole = start.checked_sub(FADE).unwrap_or(start);
                for (&key, tile) in &mut self.lods {
                    if tile.shown.is_none() {
                        tile.shown = Some(if over_replaced(key) { start } else { whole });
                    }
                    if let Some((_, since)) = &mut tile.previous {
                        since.get_or_insert(start);
                    }
                }
                for key in pending.removed {
                    let alone = alone(key);
                    if let Some(tile) = self.lods.get_mut(&key)
                        && tile.dying.is_none()
                    {
                        tile.dying = Some(start);
                        tile.alone = alone;
                    }
                }
                if let Some(cutout) = pending.cutout {
                    self.set_lod_cutout(cutout);
                }
                return;
            };
            let mesh = pending.meshes.remove(&key).expect("pending tile");
            let [x, _, z] = key.origin();
            let size = key.size();
            let keep = mesh.keep
                || around_new.contains(&key)
                || taken.iter().any(|&(cx, cz)| {
                    (x..x + size).contains(&(cx * 16)) && (z..z + size).contains(&(cz * 16))
                });
            self.upload_lod(key, &mesh, keep);
            if start.elapsed() >= budget {
                return;
            }
        }
    }

    /// The tiles drawn for good, not fading out (`render-capture`).
    pub(super) fn standing_lods(&self) -> FxHashSet<TileKey> {
        self.lods
            .iter()
            .filter(|(_, tile)| tile.dying.is_none())
            .map(|(&key, _)| key)
            .collect()
    }

    /// The chunk columns the resident tiles now leave out (`Lod::take_cutout`).
    /// Removed sections they no longer leave out are released.
    pub fn set_lod_cutout(&mut self, cutout: crate::fast_hash::FxHashSet<(i32, i32)>) {
        self.lod_cutout = cutout;
        let released: Vec<_> = self
            .lod_retiring
            .iter()
            .copied()
            .filter(|key| !self.lod_cutout.contains(&(key.0, key.2)))
            .collect();
        for key in released {
            self.lod_retiring.remove(&key);
            self.leave_section(key);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesher::TerrainVertex;

    #[test]
    fn tiles_fade_in_on_their_own_grid() {
        let start = Instant::now();
        let at = |ms| start + Duration::from_millis(ms);
        // terrain.wgsl's `faded` reads the level and sixty-fourths back.
        let decode = |code: u32| ((code - 1) / 129, (code - 1) % 129);
        assert_eq!(fade_code(None, 0, at(0)), None, "waits for its job");
        assert_eq!(fade_code(Some(start), 3, at(0)).map(decode), Some((3, 1)));
        let mut last = 0;
        for ms in [0, 100, 300, 500, 590] {
            let (level, shown) = decode(fade_code(Some(start), 10, at(ms)).unwrap());
            assert_eq!(level, 10);
            assert!(shown > last && shown < 64, "{shown} at {ms} ms");
            last = shown;
        }
        assert_eq!(fade_code(Some(start), 10, at(600)), Some(0), "whole");
        // Fading out: 65 to 128, then nothing drawn (128) until released.
        let mut last = 64;
        for ms in [0, 100, 300, 500, 590] {
            let (level, out) = decode(fade_out_code(start, 3, at(ms)));
            assert_eq!(level, 3);
            assert!(out > last && out < 128, "{out} at {ms} ms");
            last = out;
        }
        assert_eq!(decode(fade_out_code(start, 3, at(700))), (3, 128));
        // A section's largest code keeps its UV step exact (terrain_uv).
        let w = 129.0f32 + 1.0 / 32768.0;
        assert_eq!(w.fract(), 1.0 / 32768.0);
    }

    #[test]
    fn joined_jobs_apply_as_if_in_turn() {
        let key = |x| TileKey { x, z: 0, level: 0 };
        let mut pending = Pending::default();
        // The first job meshes 1 and 2 and retires 3 and 4.
        pending.add(
            vec![
                (key(1), SectionMesh::default()),
                (key(2), SectionMesh::default()),
            ],
            vec![key(3), key(4)],
            Some([(0, 0)].into_iter().collect()),
        );
        // The second meshes 3 again and retires 2.
        pending.add(vec![(key(3), SectionMesh::default())], vec![key(2)], None);
        let mut meshes: Vec<_> = pending.meshes.keys().map(|k| k.x).collect();
        meshes.sort_unstable();
        let mut removed: Vec<_> = pending.removed.iter().map(|k| k.x).collect();
        removed.sort_unstable();
        assert_eq!((meshes, removed), (vec![1, 3], vec![2, 4]));
        assert!(pending.cutout.as_ref().is_some_and(|c| c.contains(&(0, 0))));
        // They go up in the order handed over, each once.
        pending.add(vec![(key(1), SectionMesh::default())], vec![], None);
        let order: Vec<_> = pending
            .order
            .iter()
            .filter(|k| pending.meshes.contains_key(k))
            .map(|k| k.x)
            .collect();
        assert_eq!(order, [1, 3]);
    }

    #[test]
    fn large_tile_bounds_include_water_and_do_not_cull_by_origin() {
        let vertex = |pos| TerrainVertex::tiled(pos, [0.0; 2], 0, 0, 0);
        let mesh = SectionMesh {
            solid: vec![vertex([0.0, -64.0, 0.0]), vertex([1024.0, 100.0, 1024.0])],
            translucent: vec![vertex([512.0, 120.0, 512.0])],
            bounds: Some([Vec3::new(0.0, -64.0, 0.0), Vec3::new(1024.0, 120.0, 1024.0)]),
            ..Default::default()
        };
        let bounds = Bounds::mesh(&mesh).unwrap();
        assert_eq!(bounds.min, Vec3::new(0.0, -64.0, 0.0));
        assert_eq!(bounds.max, Vec3::new(1024.0, 120.0, 1024.0));
        let view = Frustum {
            planes: super::super::frustum(
                super::super::world_projection(70.0, 16.0 / 9.0, 8192.0)
                    * crate::camera::view_matrix(0.0, 0.0),
            ),
        };
        // The tile origin is far behind, but its far edge reaches the camera.
        // Negative world origins therefore remain visible.
        assert!(
            bounds
                .visible(DVec3::new(-1000.0, -80.0, -512.0), &view)
                .is_some()
        );
        // Wholly behind the camera: out of view.
        assert!(
            bounds
                .visible(DVec3::new(-512.0, -80.0, -2000.0), &view)
                .is_none()
        );
        assert!(Bounds::mesh(&SectionMesh::default()).is_none());
    }
}
