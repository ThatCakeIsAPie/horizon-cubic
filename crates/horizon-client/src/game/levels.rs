//! A server's levels, one per dimension (`MinecraftServer.levels`).
//!
//! The game's tick code reads one level: the one in place, held in `Game`'s
//! own fields (the world, its streaming and save, block ticks, entities).
//! A server (the dedicated server and single player's server thread) keeps
//! the other dimensions' levels here and ticks each one with players or forced
//! chunks by swapping it into place (`enter_level`), the way `as_guest` swaps a player
//! in, so the tick code runs unchanged, one level at a time.
//!
//! A cubic Overworld may be several levels (vertical shards, docs/cubic-v2-plan.md,
//! "Residency"): each is a `World`, `Streaming` and window of its own over the
//! same cubes (`CubicLevel`), and a level is named by its dimension and shard
//! (`LevelId`). The primary shard (0) is the one the dimension-keyed code
//! finds.
//!
//! Shared by every level, as vanilla's derived levels share the
//! Overworld's level data: the game time, the day time and the weather, the
//! game rules, the scoreboard, the maps and the entities waiting to arrive
//! through a portal.

use super::{
    Dim, ExperienceOrb, FallingBlock, Game, ItemEntity, Mob, boats, commands,
    creaking::CreakingState, dragon, fireworks, lan::LanHost, lightning, machines, minecarts,
    ominous_item_spawner, persist, raids::Raiding, sculk::SculkState, tnt, trident,
};
use horizon_client::{
    Result, block_entity_anim::BlockEntityAnims, random_tick_sections::RandomTickSections,
    save::SaveDir, streaming::Streaming, tick_chunk_membership::TickChunkMembership,
    ticket_levels::ChunkTickets, vanillagen::VanillaTerrain, world::World,
};
use horizon_world::level::{LevelEvent, Ticks};
use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};

/// Server ticks a level other than the Overworld stays loaded with no player
/// in it, before it is saved and closed (vanilla unloads a level's chunks
/// once no ticket holds them; the Overworld stays loaded).
const IDLE_TICKS: u32 = 300;

/// A level: a dimension and, in a cubic one, the vertical shard. Without a
/// shard it means the shard in place when the dimension is the one in place,
/// else the dimension's primary.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct LevelId {
    pub dim: Dim,
    pub shard: Option<u16>,
}

impl From<Dim> for LevelId {
    fn from(dim: Dim) -> Self {
        Self { dim, shard: None }
    }
}

impl LevelId {
    /// The dimension's id.
    pub(super) fn id(&self) -> &'static str {
        self.dim.id()
    }

    /// The level a guest is in.
    pub(super) fn of(g: &horizon_client::lan::guest::Guest) -> Self {
        Self::shard(Dim::from_id(&g.dimension), g.shard)
    }

    /// Shard `shard` of `dim`.
    pub(super) fn shard(dim: Dim, shard: u16) -> Self {
        Self {
            dim,
            shard: Some(shard),
        }
    }
}

/// A level that is not in place: every per-level field of `Game`.
pub(super) struct Parked {
    /// The vertical shard of its dimension this level is.
    shard: u16,
    /// The shard's window cell (`None` for the primary's, which the cubic
    /// level keeps).
    shard_window: Option<horizon_client::cubic::ShardWindow>,
    /// The shard's window sliding (`game/cubic.rs`).
    slide: super::cubic::SlideState,
    world: World,
    streaming: Streaming,
    locator: Option<Arc<VanillaTerrain>>,
    save: Arc<SaveDir>,
    block_ticks: Ticks,
    tick_chunks: TickChunkMembership,
    tickets: ChunkTickets,
    random_tick_sections: RandomTickSections,
    level_events: Vec<LevelEvent>,
    /// Entity portal processes and cooldowns belong to this level's entities.
    portal_states: HashMap<u32, horizon_client::nether_portal::PortalState>,
    falling: horizon_client::join_seq::JoinList<FallingBlock>,
    paintings: Vec<horizon_client::entities::Painting>,
    item_frames: Vec<horizon_client::entities::ItemFrame>,
    armor_stands: horizon_client::join_seq::JoinList<horizon_client::entities::ArmorStand>,
    cushions: Vec<horizon_client::entities::Cushion>,
    others: Vec<commands::OtherEntity>,
    fangs: super::fangs::FangState,
    pub(super) boats: horizon_client::join_seq::JoinList<boats::Boat>,
    pub(super) carts: horizon_client::join_seq::JoinList<minecarts::Minecart>,
    jukebox_sounds: HashMap<(i32, i32, i32), u64>,
    bolts: Vec<lightning::Bolt>,
    crystals: Vec<dragon::Crystal>,
    dragon: Option<dragon::Dragon>,
    dragon_fireballs: Vec<dragon::DragonFireball>,
    breath_clouds: Vec<dragon::Breath>,
    tridents: horizon_client::join_seq::JoinList<trident::ThrownTrident>,
    rockets: horizon_client::join_seq::JoinList<fireworks::Rocket>,
    ominous_spawners: Vec<ominous_item_spawner::OminousItemSpawner>,
    tnts: horizon_client::join_seq::JoinList<tnt::PrimedTnt>,
    hoppers: HashMap<(i32, i32, i32), machines::HopperState>,
    pub(super) mobs: horizon_client::join_seq::JoinList<Mob>,
    /// The entity ids of the level's fence knots (`knot_ids`).
    knot_ids: HashMap<[i32; 3], u32>,
    projectiles: horizon_client::join_seq::JoinList<horizon_client::projectiles::Projectile>,
    dropped: horizon_client::join_seq::JoinList<ItemEntity>,
    orbs: horizon_client::join_seq::JoinList<ExperienceOrb>,
    persist: persist::EntityMemory,
    lectern_pulses: Vec<((i32, i32, i32), i32)>,
    lid_openers: HashMap<(i32, i32, i32), i32>,
    lan_events: Vec<(i32, [i32; 3], i32, Option<u32>)>,
    lan_block_events: Vec<([i32; 3], i32, i32, u32)>,
    lan_entity_events: Vec<(i32, i8, [f64; 3])>,
    lan_critical_hits: Vec<(Option<u32>, i32, bool)>,
    /// The hub's tracked entities of this level (`LanHost::states`), as its last tick described
    /// them: what its players' moves are checked against before this tick describes them again.
    lan_states: horizon_client::lan::entity::States,
    /// Only the per-level parts of these are used (`swap_level`).
    anims: BlockEntityAnims,
    raiding: Raiding,
    sculk: SculkState,
    creaking: CreakingState,
    /// Natural spawning's candidate cache: each level keeps its own players' squares.
    spawn_scratch: super::natural_spawning::SpawnScratch,
    /// Server ticks since a player was last in the level.
    idle: u32,
}

/// Which passes of the server tick the level in place runs: the first
/// reads the network (`Hub::poll`), the last ends the tick for every player
/// (`Hub::end_tick`). A tick with one level is both. `running` is the
/// tick rate manager's answer for this server tick, asked once before the
/// levels (`MinecraftServer.tickServer`); `None` outside `tick_server`.
#[derive(Clone, Copy)]
pub(super) struct Pass {
    pub first: bool,
    pub last: bool,
    pub running: Option<bool>,
    /// Set once this level's player packets were handled before block ticks.
    pub packets_drained: bool,
}

impl Default for Pass {
    fn default() -> Self {
        Self {
            first: true,
            last: true,
            running: None,
            packets_drained: false,
        }
    }
}

/// The levels not in place, by dimension.
#[derive(Default)]
pub(super) struct Levels {
    parked: HashMap<(Dim, u16), Parked>,
    /// The levels whose open the mods heard (`native_level_load`), each to
    /// be told closed once.
    heard_open: Vec<Dim>,
}

impl Levels {
    /// Clear every parked level and dimensions recorded as heard open.
    pub(super) fn clear(&mut self) {
        self.parked.clear();
        self.heard_open.clear();
    }

    /// The key of `dim`'s primary level among the parked ones: its lowest shard.
    fn primary_key(&self, dim: Dim) -> Option<(Dim, u16)> {
        self.parked
            .keys()
            .filter(|k| k.0 == dim)
            .min_by_key(|k| k.1)
            .copied()
    }

    /// The shard `dim`'s level not in place is for a bare dimension: the primary's.
    /// The shard a bare `dim` means among the parked levels: its primary.
    pub(super) fn primary_shard_of(&self, dim: Dim) -> u16 {
        self.primary_shard(dim)
    }

    fn primary_shard(&self, dim: Dim) -> u16 {
        self.primary_key(dim).map_or(0, |k| k.1)
    }

    /// The primary shard of `dim`, counting the level in place (`here`).
    pub(super) fn primary_shard_or(&self, dim: Dim, here_dim: Dim, here_shard: u16) -> u16 {
        let parked = self.primary_key(dim).map(|k| k.1);
        match (parked, here_dim == dim) {
            (Some(p), true) => p.min(here_shard),
            (Some(p), false) => p,
            (None, true) => here_shard,
            (None, false) => 0,
        }
    }

    /// The world of shard `shard` of `dim`, when that level is parked.
    pub(super) fn parked_shard_world(&self, dim: Dim, shard: u16) -> Option<&World> {
        self.parked.get(&(dim, shard)).map(|level| &level.world)
    }

    /// Keeps a new shard's level with the others.
    pub(super) fn insert_shard(&mut self, parked: Parked) {
        self.parked.insert((Dim::Overworld, parked.shard), parked);
    }

    /// Drops the level of shard `shard` of `dim`.
    pub(super) fn remove_shard(&mut self, dim: Dim, shard: u16) {
        self.parked.remove(&(dim, shard));
    }

    /// The shards of `dim` that are parked, lowest first.
    pub(super) fn parked_shards(&self, dim: Dim) -> Vec<u16> {
        let mut shards: Vec<u16> = self
            .parked
            .keys()
            .filter(|k| k.0 == dim)
            .map(|k| k.1)
            .collect();
        shards.sort_unstable();
        shards
    }

    /// A parked level, without opening or loading it.
    pub(super) fn parked_world(&self, dim: Dim) -> Option<&World> {
        let key = self.primary_key(dim)?;
        self.parked.get(&key).map(|level| &level.world)
    }

    /// The worlds of the levels not in place.
    pub(super) fn parked_worlds(&self) -> impl Iterator<Item = &World> {
        self.parked.values().map(|level| &level.world)
    }

    /// The world of the level not in place, in `dim`, to change.
    pub(super) fn parked_world_mut(&mut self, dim: Dim) -> Option<&mut World> {
        let key = self.primary_key(dim)?;
        self.parked.get_mut(&key).map(|level| &mut level.world)
    }

    /// The level not in place, in `dim`.
    pub(super) fn parked_mut(&mut self, dim: Dim) -> Option<&mut Parked> {
        let key = self.primary_key(dim)?;
        self.parked.get_mut(&key)
    }

    /// The entities of a parked level, counted as `Game::entity_count` counts the one in place.
    pub(super) fn parked_entity_count(&self, dim: Dim) -> Option<usize> {
        let key = self.primary_key(dim)?;
        let l = self.parked.get(&key)?;
        Some(
            l.mobs.iter().filter(|m| !m.removed).count()
                + l.dropped.len()
                + l.orbs.len()
                + l.projectiles.len()
                + l.carts.len()
                + l.boats.len()
                + l.falling.len()
                + l.paintings.len()
                + l.item_frames.len()
                + l.armor_stands.len()
                + l.cushions.len()
                + l.tridents.len()
                + l.rockets.len()
                + l.tnts.len(),
        )
    }

    /// The mobs of a parked level.
    pub(super) fn parked_mobs(&self, dim: Dim) -> Option<&[Mob]> {
        let key = self.primary_key(dim)?;
        self.parked.get(&key).map(|level| level.mobs.as_slice())
    }

    /// A parked level's loose entities, for `horizon-report`: items, experience orbs,
    /// projectiles, minecarts and boats.
    pub(super) fn parked_entity_counts(&self, dim: Dim) -> Option<[usize; 5]> {
        let key = self.primary_key(dim)?;
        self.parked.get(&key).map(|l| {
            [
                l.dropped.as_slice().len(),
                l.orbs.as_slice().len(),
                l.projectiles.as_slice().len(),
                l.carts.as_slice().len(),
                l.boats.as_slice().len(),
            ]
        })
    }

    /// The levels that are not in place, by dimension.
    /// The forced chunks of every level not in place, released.
    pub(super) fn release_parked_forced_chunks(&mut self) {
        for level in self.parked.values_mut() {
            level.streaming.forced.clear();
        }
    }

    /// The dimensions with a level not in place, once each.
    pub(super) fn parked_dims(&self) -> Vec<Dim> {
        let mut dims: Vec<Dim> = Vec::new();
        for (dim, _) in self.parked.keys() {
            if !dims.contains(dim) {
                dims.push(*dim);
            }
        }
        dims
    }
}

impl Game<'_> {
    /// A new level for `dim`, opened from the save (`MinecraftServer.createLevels`).
    fn open_level(&self, dim: Dim) -> Result<Parked> {
        let (streaming, save, locator) = super::world_loading::make_streaming(
            &self.res,
            &self.pack,
            &self.options,
            &self.settings,
            &self.world_root,
            dim,
            self.level_shapes.get(dim.id()),
            self.level.seed,
            &self.level.enabled_features,
        )?;
        Ok(self.new_parked(dim, streaming, save, locator))
    }

    /// The Overworld's save and vanilla terrain, wherever its level is.
    fn overworld_parts(
        &self,
    ) -> Option<(
        Arc<horizon_client::save::SaveDir>,
        Option<Arc<horizon_client::vanillagen::VanillaTerrain>>,
    )> {
        if self.dim == Dim::Overworld {
            return Some((self.save.clone(), self.end.locator()));
        }
        let key = self.levels.primary_key(Dim::Overworld)?;
        let p = self.levels.parked.get(&key)?;
        Some((p.save.clone(), p.locator.clone()))
    }

    /// A new shard of the cubic Overworld around `shape`'s window, with its
    /// own world and streaming over the primary's level (`CubicLevel`): the
    /// same store, generator, population and saves. Empty until its players
    /// and columns arrive.
    pub(super) fn open_shard(
        &self,
        shard: u16,
        shape: horizon_client::chunk::LevelShape,
    ) -> Result<Parked> {
        let dim = Dim::Overworld;
        let (save, locator) = self.overworld_parts().ok_or("the Overworld is not open")?;
        let level = save.cubic.get().cloned().ok_or("the level is not cubic")?;
        let window = horizon_client::cubic::shard_window(shape);
        let mut streaming = level
            .new_shard_streaming(window.clone())
            .ok_or("the level cannot make a shard")?;
        super::world_loading::configure_streaming(
            &mut streaming,
            &save,
            &self.options,
            &self.settings,
            dim,
            self.level.seed,
        );
        // The level's forced chunks are the primary shard's.
        streaming.forced.clear();
        let mut parked = self.new_parked(dim, streaming, save, None);
        parked.shard = shard;
        parked.shard_window = Some(window);
        parked.world.shape = shape;
        parked.locator = locator;
        Ok(parked)
    }

    /// A level of `dim` over `streaming` and `save`, with nothing in it yet.
    fn new_parked(
        &self,
        dim: Dim,
        streaming: horizon_client::streaming::Streaming,
        save: Arc<horizon_client::save::SaveDir>,
        locator: Option<Arc<horizon_client::vanillagen::VanillaTerrain>>,
    ) -> Parked {
        let mut world = World::new(self.res.facts().air, self.level.seed);
        world.attachments = save.load_attachments();
        world.shape = self.level_shapes.get(dim.id());
        world.fast_lava = dim == Dim::Nether;
        Parked {
            shard: 0,
            shard_window: None,
            slide: Default::default(),
            world,
            streaming,
            locator,
            save,
            block_ticks: Ticks::new(self.level.seed as i32),
            tick_chunks: Default::default(),
            tickets: Default::default(),
            random_tick_sections: Default::default(),
            level_events: Vec::new(),
            portal_states: HashMap::new(),
            falling: Default::default(),
            paintings: Vec::new(),
            item_frames: Vec::new(),
            armor_stands: Default::default(),
            cushions: Vec::new(),
            others: Vec::new(),
            fangs: Default::default(),
            boats: Default::default(),
            carts: Default::default(),
            jukebox_sounds: HashMap::new(),
            bolts: Vec::new(),
            crystals: Vec::new(),
            dragon: None,
            dragon_fireballs: Vec::new(),
            breath_clouds: Vec::new(),
            tridents: Default::default(),
            rockets: Default::default(),
            ominous_spawners: Vec::new(),
            tnts: Default::default(),
            hoppers: HashMap::new(),
            mobs: Default::default(),
            knot_ids: HashMap::new(),
            projectiles: Default::default(),
            dropped: Default::default(),
            orbs: Default::default(),
            persist: Default::default(),
            lectern_pulses: Vec::new(),
            lid_openers: HashMap::new(),
            lan_events: Vec::new(),
            lan_block_events: Vec::new(),
            lan_entity_events: Vec::new(),
            lan_critical_hits: Vec::new(),
            lan_states: Default::default(),
            anims: Default::default(),
            raiding: Default::default(),
            sculk: Default::default(),
            creaking: Default::default(),
            spawn_scratch: Default::default(),
            idle: 0,
        }
    }

    /// Exchange the level in place with `p`.
    fn swap_level(&mut self, p: &mut Parked) {
        use std::mem::swap;
        swap(&mut self.world, &mut p.world);
        swap(&mut self.streaming, &mut p.streaming);
        swap(&mut self.shard, &mut p.shard);
        swap(&mut self.shard_window, &mut p.shard_window);
        swap(&mut self.cubic_slide, &mut p.slide);
        // `CubicLevel::save_entities` stands for the window of the level in place.
        if let Some(level) = self.save.cubic.get() {
            level.set_window(self.world.shape);
        }
        let mut locator = self.end.locator();
        swap(&mut locator, &mut p.locator);
        self.end.set_locator(locator);
        self.share_spawn_conditions();
        swap(&mut self.save, &mut p.save);
        swap(&mut self.block_ticks, &mut p.block_ticks);
        swap(&mut self.tick_chunks, &mut p.tick_chunks);
        swap(&mut self.tickets, &mut p.tickets);
        swap(&mut self.random_tick_sections, &mut p.random_tick_sections);
        swap(&mut self.level_events, &mut p.level_events);
        swap(&mut self.portals.entities, &mut p.portal_states);
        swap(&mut self.falling, &mut p.falling);
        swap(&mut self.paintings, &mut p.paintings);
        swap(&mut self.item_frames, &mut p.item_frames);
        swap(&mut self.armor_stands, &mut p.armor_stands);
        swap(&mut self.cushions, &mut p.cushions);
        swap(&mut self.others, &mut p.others);
        swap(&mut self.fangs, &mut p.fangs);
        swap(&mut self.boats, &mut p.boats);
        swap(&mut self.carts, &mut p.carts);
        swap(&mut self.jukebox_sounds, &mut p.jukebox_sounds);
        swap(&mut self.bolts, &mut p.bolts);
        swap(&mut self.crystals, &mut p.crystals);
        swap(&mut self.dragon, &mut p.dragon);
        swap(&mut self.dragon_fireballs, &mut p.dragon_fireballs);
        swap(&mut self.breath_clouds, &mut p.breath_clouds);
        swap(&mut self.tridents, &mut p.tridents);
        swap(&mut self.rockets, &mut p.rockets);
        swap(&mut self.ominous_spawners, &mut p.ominous_spawners);
        swap(&mut self.tnts, &mut p.tnts);
        swap(&mut self.hoppers, &mut p.hoppers);
        swap(&mut self.mobs, &mut p.mobs);
        swap(&mut self.knot_ids, &mut p.knot_ids);
        swap(&mut self.creatures.projectiles, &mut p.projectiles);
        swap(&mut self.dropped, &mut p.dropped);
        swap(&mut self.orbs, &mut p.orbs);
        swap(&mut self.persist, &mut p.persist);
        swap(&mut self.lectern_pulses, &mut p.lectern_pulses);
        swap(&mut self.lid_openers, &mut p.lid_openers);
        swap(&mut self.lan_events, &mut p.lan_events);
        swap(&mut self.lan_block_events, &mut p.lan_block_events);
        swap(&mut self.lan_entity_events, &mut p.lan_entity_events);
        swap(&mut self.lan_critical_hits, &mut p.lan_critical_hits);
        let (a, b) = (&mut self.be_anims, &mut p.anims);
        swap(&mut a.chests, &mut b.chests);
        swap(&mut a.books, &mut b.books);
        swap(&mut a.shulkers, &mut b.shulkers);
        swap(&mut a.bells, &mut b.bells);
        swap(&mut a.conduits, &mut b.conduits);
        swap(&mut a.skulls, &mut b.skulls);
        swap(&mut a.pots, &mut b.pots);
        swap(&mut a.gateways, &mut b.gateways);
        swap(&mut a.gateway_servers, &mut b.gateway_servers);
        swap(&mut a.conduit_servers, &mut b.conduit_servers);
        self.raiding.swap_level(&mut p.raiding);
        self.sculk.swap_level(&mut p.sculk);
        self.creaking.swap_level(&mut p.creaking);
        swap(&mut self.spawn_scratch, &mut p.spawn_scratch);
        // What only the client draws starts over.
        self.particles.clear();
        self.pickups.clear();
    }

    /// The level in place.
    pub(super) fn here(&self) -> LevelId {
        LevelId::shard(self.dim, self.shard)
    }

    /// Put `target`'s level in place, opening it first if it is not loaded.
    /// The level that was in place is kept with the others; the shared
    /// clock and weather go with the swap. `host` is the hub when the caller
    /// holds it (`self.lan` is taken during the network tick). A bare
    /// dimension is the shard in place when it is the dimension in place, else
    /// the dimension's primary.
    pub(super) fn enter_level(
        &mut self,
        mut host: Option<&mut LanHost>,
        target: impl Into<LevelId>,
    ) -> Result<()> {
        // A client holds no levels.
        if self.options.execution_role.is_client() {
            return Ok(());
        }
        let target: LevelId = target.into();
        let shard = target.shard.unwrap_or_else(|| {
            if target.dim == self.dim {
                self.shard
            } else {
                self.levels.primary_shard(target.dim)
            }
        });
        let target = target.dim;
        if (target, shard) != (self.dim, self.shard) {
            let (mut p, opened) = match self.levels.parked.remove(&(target, shard)) {
                Some(p) => (p, false),
                None if shard == 0 => (self.open_level(target)?, true),
                None => {
                    return Err(format!("{} has no shard {shard}", target.id()).into());
                }
            };
            p.world.game_time = self.world.game_time;
            p.world.day_time = self.world.day_time;
            p.world.weather = self.world.weather;
            // The game rules and the difficulty are the server's, not a
            // level's: natural regeneration holds in every dimension.
            p.world.health_rules = self.world.health_rules;
            // The sculk tables' watched blocks are built once per game.
            p.world.watched = std::mem::take(&mut self.world.watched);
            self.swap_level(&mut p);
            // The tracked entities go with their level: a level's moves, read before its tick
            // describes its entities again, would otherwise meet the last level's (the anticheat
            // missed the boat under a player while anyone was in another dimension).
            let states = match host.as_deref_mut() {
                Some(h) => Some(&mut h.states),
                None => self.lan.as_mut().map(|h| &mut h.states),
            };
            if let Some(states) = states {
                std::mem::swap(states, &mut p.lan_states);
            }
            p.idle = 0;
            // The swap put the level that was in place into `p`, with its shard.
            let left = (std::mem::replace(&mut self.dim, target), p.shard);
            self.levels.parked.insert(left, p);
            if opened {
                // The new streaming's radius and meshing options.
                self.apply_settings();
                self.native_level_opened(target);
            }
        }
        let shape = self.world.shape;
        let (dim, shard) = (self.dim, self.shard);
        let host = match host {
            Some(h) => Some(h),
            None => self.lan.as_mut(),
        };
        if let Some(h) = host {
            h.hub.set_level(dim.id(), shard, Some(shape));
        }
        Ok(())
    }

    /// The entity's memory (its UUID and the compound it was loaded with)
    /// moves with it from the level not in place to the level in place, so
    /// it is saved in the other level as it was.
    pub(super) fn move_entity_memory(&mut self, from: Dim, id: u32) {
        let Some(key) = self.levels.primary_key(from) else {
            return;
        };
        let Some(p) = self.levels.parked.get_mut(&key) else {
            return;
        };
        let uuid = p.persist.uuids.remove(&id);
        let raw = p.persist.raw.remove(&id);
        if let Some(u) = uuid {
            self.persist.uuids.insert(id, u);
        }
        if let Some(r) = raw {
            self.persist.raw.insert(id, r);
        }
    }

    /// Far terrain for a client in `dim`: the level in place may be another
    /// one, the server ticking its levels in turn.
    pub(super) fn lod_source_in(
        &self,
        dim: Dim,
    ) -> Option<Arc<dyn horizon_client::streaming::LodSource>> {
        if dim == self.dim {
            return Some(self.streaming.lod_source());
        }
        let parked = self.levels.parked.get(&self.levels.primary_key(dim)?)?;
        Some(parked.streaming.lod_source())
    }

    /// A far-terrain sampler of `dim`'s level when it is cubic (the cube
    /// octree's, `cubic_lod`): the cube generator past vanilla's range, the
    /// level's vanilla far terrain inside it.
    pub(super) fn cubic_sampler_in(
        &self,
        dim: Dim,
    ) -> Option<horizon_client::cubic_lod::CubicLodSampler> {
        let (streaming, save) = if dim == self.dim {
            (&self.streaming, &self.save)
        } else {
            let parked = self.levels.parked.get(&self.levels.primary_key(dim)?)?;
            (&parked.streaming, &parked.save)
        };
        let level = save.cubic.get()?;
        let vanilla = streaming.lod_source().sampler();
        Some(level.lod_sampler(vanilla, self.pack.assets.registry))
    }

    /// Put `dim`'s level in place for a command whose source is there
    /// (`execute in`, `execute at` a player in another level), opening it if
    /// nobody is in it; returns the level to put back afterwards. Only a
    /// server holds other levels.
    pub(super) fn enter_command_level(&mut self, dim: impl Into<LevelId>) -> Option<LevelId> {
        let dim: LevelId = dim.into();
        if self.options.execution_role.is_client()
            || (dim.dim == self.dim && dim.shard.is_none_or(|s| s == self.shard))
        {
            return None;
        }
        let back = self.here();
        match self.enter_level(None, dim) {
            Ok(()) => Some(back),
            Err(e) => {
                log::error!("could not open {}: {e}", dim.id());
                None
            }
        }
    }

    /// One server tick (`MinecraftServer.tickServer`): each level with
    /// players or forced chunks in turn (`tickChildren`), its chunks streamed
    /// and loaded and its world ticked. The first pass reads the network and runs the
    /// tick's shared parts (the clock, the weather, functions, the autosave);
    /// the next ones only their level. While `paused`, only the connections
    /// run. The native server tick hook brackets it (`ServerTickStart` before
    /// the first level, `ServerTickEnd` after the last).
    pub(super) fn tick_server(&mut self, paused: bool) -> Result<()> {
        // `tickRateManager.tick()` once per server tick, before any level
        // (each `ServerLevel.tick` only reads `runsNormally`), and the native
        // `ServerTickStart` before that level work, where NeoForge fires
        // `ServerTickEvent.Pre`. A paused pass and a held server tick nothing.
        let running = (!paused && !self.agent.hold_server).then(|| self.clock.world_runs());
        let server = self.serving.is_some() || self.options.dedicated.is_some();
        // A config edited while the server is paused (empty) still reloads.
        if server {
            self.native_configs_poll();
        }
        if let Some(running) = running
            && server
        {
            self.native_lifecycle_server_tick_start(running);
        }
        let observe_level =
            running.is_some() && (self.serving.is_some() || self.options.dedicated.is_some());
        let dims = self.levels_to_tick();
        let order = dims.clone();
        let n = dims.len();
        let mut result = Ok(());
        for (k, (dim, shard)) in dims.into_iter().enumerate() {
            // Each level's share of the tick (the first one also reads the network and runs the
            // tick's shared parts): `horizon-report`'s `level_*` spans.
            let _level_span = horizon_client::spans::guard(match dim {
                Dim::Overworld => "level_overworld",
                Dim::Nether => "level_the_nether",
                Dim::End => "level_the_end",
            });
            if let Err(e) = self.enter_level(None, LevelId::shard(dim, shard)) {
                result = Err(e);
                break;
            }
            self.level_pass = Pass {
                first: k == 0,
                last: k + 1 == n,
                running,
                packets_drained: false,
            };
            let observe = observe_level && shard == 0;
            if observe {
                self.native_level_tick_pre(dim.id());
            }
            self.update_dedicated_streaming();
            self.update_world();
            result = horizon_client::block_contact::in_level(
                (self.serving.is_some() || self.options.dedicated.is_some()).then_some(dim.id()),
                || {
                    if paused {
                        self.tick_lan(true);
                        Ok(())
                    } else if k == 0 {
                        self.tick()
                    } else {
                        let running = running.unwrap_or_else(|| self.clock.world_runs());
                        // ClientLevel.tick: the sky flash fades in a frozen level too.
                        self.tick_weather_effects();
                        self.tick_world(running)
                    }
                },
            );
            if result.is_err() {
                break;
            }
            if observe {
                self.native_level_tick_post(dim.id());
            }
            self.publish_server_debug(dim);
            // An entity that crossed into a level that already ticked this
            // server tick is in it when the tick ends (`addDuringTeleport`).
            if running == Some(true) {
                let mut moved = false;
                for &(earlier, earlier_shard) in &order[..k] {
                    if earlier_shard == 0
                        && !self.arrival_chunks(earlier).is_empty()
                        && self.enter_level(None, LevelId::shard(earlier, 0)).is_ok()
                    {
                        self.land_portal_arrivals();
                        moved = true;
                    }
                }
                if moved {
                    self.enter_level(None, LevelId::shard(dim, shard))?;
                }
            }
        }
        self.level_pass = Pass::default();
        if result.is_err() {
            self.native_lifecycle.cancel_server_tick();
        }
        result?;
        // A cubic level's window follows its players (game/cubic.rs).
        if let Err(error) = self.follow_cubic_window() {
            self.native_lifecycle.cancel_server_tick();
            return Err(error);
        }
        if let Err(error) = self.close_idle_levels() {
            self.native_lifecycle.cancel_server_tick();
            return Err(error);
        }
        // `ServerTickEnd`: an errored tick or an idle-close failure (at
        // the `?` above) never reaches this site.
        self.native_lifecycle_server_tick_end(paused);
        Ok(())
    }

    /// `tick_server` as a server thread runs it. With `HORIZON_SPIKES=MS`, a
    /// tick longer than MS is logged with its thread CPU time and spans.
    pub(super) fn tick_server_timed(&mut self, paused: bool) -> Result<()> {
        self.tick_server_spans(paused, |_, _, _, _| {})
    }

    /// `tick_server_timed`, handing the tick's wall time, thread CPU time
    /// and spans (milliseconds, summed by name) to `record` while spans are on, with how many
    /// spans the tick itself recorded: a dedicated server's performance report turns them on to
    /// record its ticks.
    pub(super) fn tick_server_spans(
        &mut self,
        paused: bool,
        record: impl FnOnce(f32, Option<f32>, Vec<horizon_client::spans::Span>, usize),
    ) -> Result<()> {
        if self.options.spikes.is_none() && !horizon_client::spans::enabled() {
            return self.tick_server(paused);
        }
        // Spans recorded since the last tick ran between ticks (the chunk pump of the wait): they
        // stay in the tick's list, but only the tick's own count as recorded in it.
        let between = horizon_client::spans::take_raw();
        let (started, cpu) = (Instant::now(), horizon_client::spans::thread_cpu());
        let result = self.tick_server(paused);
        let wall = started.elapsed().as_secs_f32() * 1000.0;
        let cpu = horizon_client::spans::thread_cpu()
            .zip(cpu)
            .map(|(end, start)| (end - start).as_secs_f32() * 1000.0);
        // Taken every tick: nothing else takes this thread's spans.
        let raw = horizon_client::spans::take_raw();
        let in_tick = raw.len();
        let spans = if between.is_empty() {
            horizon_client::spans::merge(&raw)
        } else {
            horizon_client::spans::merge(&[between, raw].concat())
        };
        let notes = horizon_client::spans::take_notes();
        if let Some(threshold) = self.options.spikes
            && wall > threshold
        {
            let cpu = cpu.map_or_else(String::new, |c| format!(", cpu {c:.1} ms"));
            log::info!(
                "Horizon: slow tick {wall:.1} ms{cpu} ({} mobs): {}{}",
                self.mobs.len(),
                horizon_client::spans::describe(&spans),
                slowest_steps(&notes)
            );
        }
        record(wall, cpu, spans, in_tick);
        result
    }

    /// The end of `MinecraftServer.runServer`'s loop for a server thread: a
    /// finished `/tick sprint` reports to the operators as the server, then
    /// the next tick starts one tick length after this one, or straight away
    /// while the sprint runs (`checkShouldSprintThisTick`). A server more
    /// than two seconds behind skips the ticks it missed: that returns how far
    /// behind it was, for the overload warning.
    pub(super) fn wait_for_next_server_tick(
        &mut self,
        next: &mut Instant,
        paused: bool,
    ) -> Option<Duration> {
        if let Some(report) = self.clock.take_sprint_report() {
            // ServerTickRateManager.finishTickSprint: the server's sendSuccess.
            self.console_broadcast = Some(true);
            self.sprint_report(&report);
            self.console_broadcast = None;
            if let Some(mut host) = self.lan.take() {
                self.send_admin_feedback(&mut host, None);
                self.lan = Some(host);
            }
            self.admin_feedback.clear();
        }
        let now = Instant::now();
        if !paused && self.clock.sprinting() {
            *next = now;
            return None;
        }
        *next += self.clock.tick_duration();
        if *next > now {
            // The chunks loading (the players', force-loaded ones, a
            // `/pregen` task) keep the workers fed until the next tick.
            self.pump_chunks(*next, paused);
            let now = Instant::now();
            if *next > now {
                std::thread::sleep(*next - now);
            }
        } else if now - *next > Duration::from_secs(2) {
            let behind = now - *next;
            *next = now;
            return Some(behind);
        }
        None
    }

    /// The levels this server tick runs, in `MinecraftServer.levels` order
    /// (the Overworld, the Nether, the End): every level with a player in
    /// it, forced chunks or a `/pregen` task, or the Overworld while idle. A
    /// cubic Overworld's shards each run, in order, when they have players;
    /// whatever else keeps the dimension active runs its primary shard.
    pub(super) fn levels_to_tick(&self) -> Vec<(Dim, u16)> {
        let guests = self
            .lan
            .as_ref()
            .map_or(&[][..], |h| h.hub.guests.as_slice());
        let mut dims: Vec<(Dim, u16)> = Vec::new();
        for d in [Dim::Overworld, Dim::Nether, Dim::End] {
            // `/forceload`ed chunks, or a native mod's tickets.
            let held = |s: &Streaming| !s.forced.is_empty() || !s.mod_forced.is_empty();
            let forced = if d == self.dim {
                held(&self.streaming)
            } else {
                self.levels
                    .primary_key(d)
                    .and_then(|k| self.levels.parked.get(&k))
                    .is_some_and(|p| held(&p.streaming))
            };
            let active = forced
                // A viewer count not yet told to 0 (`tick_mod_container_viewers`).
                || self.mod_container_viewers.keys().any(|(k, _)| *k == d.id())
                || self.player_bed_levels().any(|b| b == d)
                // A game test runs in its own level, players or not.
                || self.game_tests.has_tests_in(d)
                || self.pregen_level() == Some(d)
                // A pending portal arrival keeps its destination level
                // running so it can land and tick (the portal ticket).
                || !self.arrival_chunks(d).is_empty()
                // A portal ticket keeps its dimension active
                // (`TicketType.shouldKeepDimensionActive`).
                || !self.portal_ticket_chunks(d).is_empty();
            let mut shards: Vec<u16> = guests
                .iter()
                .filter(|g| g.dimension == d.id())
                .map(|g| g.shard)
                .filter(|s| self.shard_exists(d, *s))
                .collect();
            if active {
                shards.push(self.levels.primary_shard_or(d, self.dim, self.shard));
            }
            shards.sort_unstable();
            shards.dedup();
            dims.extend(shards.into_iter().map(|s| (d, s)));
        }
        if dims.is_empty() {
            dims.push((
                Dim::Overworld,
                self.levels
                    .primary_shard_or(Dim::Overworld, self.dim, self.shard),
            ));
        }
        dims
    }

    /// Whether the level of shard `shard` of `dim` is open (in place or parked).
    pub(super) fn shard_exists(&self, dim: Dim, shard: u16) -> bool {
        (self.dim == dim && self.shard == shard) || self.levels.parked.contains_key(&(dim, shard))
    }

    /// The chunks holding a portal ticket in `dim`'s level (`placePortalTicket`).
    pub(super) fn portal_ticket_chunks(&self, dim: Dim) -> Vec<(i32, i32)> {
        if dim == self.dim {
            self.tickets.portal_chunks().collect()
        } else {
            self.levels
                .primary_key(dim)
                .and_then(|k| self.levels.parked.get(&k))
                .map_or_else(Vec::new, |p| p.tickets.portal_chunks().collect())
        }
    }

    /// Count the parked levels' empty ticks, and save and close the ones
    /// (other than the Overworld) nobody has been in for a while.
    pub(super) fn close_idle_levels(&mut self) -> Result<()> {
        let occupied = self.levels_to_tick();
        // A vehicle waiting there for its rider's arrival in another level.
        let waiting: Vec<Dim> = self
            .levels
            .parked
            .keys()
            .map(|k| k.0)
            .filter(|d| self.vehicle_waits_in(*d))
            .collect();
        let mut closing = Vec::new();
        for (key, p) in &mut self.levels.parked {
            if occupied.contains(key) || waiting.contains(&key.0) {
                p.idle = 0;
                continue;
            }
            p.idle += 1;
            // The Overworld's shards close with `close_idle_shards`, when they
            // are empty.
            if key.0 != Dim::Overworld && p.idle >= IDLE_TICKS {
                closing.push(*key);
            }
        }
        if closing.is_empty() {
            return Ok(());
        }
        let back = self.here();
        for key in closing {
            self.enter_level(None, LevelId::shard(key.0, key.1))?;
            // Its chunks leave with it: the mods hear before its last save.
            let chunks: Vec<_> = self.world.chunks.values().cloned().collect();
            self.native_chunks_changed(&chunks);
            self.save_level(true)?;
            // Its entities' attachments are saved with them; they leave memory too.
            for uuid in self.persist.uuids.values() {
                self.entity_attachments.remove(uuid);
            }
            self.enter_level(None, back)?;
            self.levels.parked.remove(&key);
            self.native_level_closed(key.0);
        }
        Ok(())
    }

    /// The mods hear `dim`'s level open (`LevelEvent.Load`), on a server.
    pub(super) fn native_level_opened(&mut self, dim: Dim) {
        if self.remote.is_some()
            || self.options.execution_role.is_client()
            || self.levels.heard_open.contains(&dim)
        {
            return;
        }
        self.levels.heard_open.push(dim);
        self.native_level_load(dim.id());
    }

    /// The mods hear `dim`'s level close (`LevelEvent.Unload`), once for
    /// each open they heard.
    fn native_level_closed(&mut self, dim: Dim) {
        let Some(i) = self.levels.heard_open.iter().position(|d| *d == dim) else {
            return;
        };
        self.levels.heard_open.remove(i);
        self.native_level_unload(dim.id());
    }

    /// Every level still open closes with the server, however it ends
    /// (`MinecraftServer.stopServer`'s `level.close()` loop), in vanilla's
    /// order of its levels: the Overworld, the Nether, the End.
    pub(super) fn native_levels_closing(&mut self) {
        for dim in [Dim::Overworld, Dim::Nether, Dim::End] {
            self.native_level_closed(dim);
        }
    }

    /// Save every parked level (`MinecraftServer.saveAllChunks`); the level
    /// in place is the caller's to save.
    pub(super) fn save_parked_levels(&mut self, wait: bool) -> Result<()> {
        if self.levels.parked.is_empty() {
            return Ok(());
        }
        let back = self.here();
        let mut keys: Vec<(Dim, u16)> = self.levels.parked.keys().copied().collect();
        keys.sort_by_key(|k| (k.1, k.0 as u8));
        for (dim, shard) in keys {
            self.enter_level(None, LevelId::shard(dim, shard))?;
            self.save_level(wait)?;
        }
        self.enter_level(None, back)
    }
}

/// A slow tick's line's end: the steps it noted as its slowest (`spans::note`), whose packets
/// or which player's event, or nothing.
fn slowest_steps(notes: &[(f32, String)]) -> String {
    if notes.is_empty() {
        return String::new();
    }
    let list: Vec<String> = notes
        .iter()
        .map(|(ms, what)| format!("{what} {ms:.1}"))
        .collect();
    format!("; slowest: {}", list.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levels_clear_empties_parked_and_heard_open() {
        let mut levels = Levels::default();
        levels.heard_open.push(Dim::Nether);
        assert!(!levels.heard_open.is_empty());
        levels.clear();
        assert!(levels.heard_open.is_empty());
        assert!(levels.parked.is_empty());
    }
}
