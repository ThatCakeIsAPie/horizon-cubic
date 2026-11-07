# Cubic as a Horizon mod: plan

Status: research and design only (2026-11-07). Nothing has been posted to arismoko/horizon or upstream. Branch
`plan/cubic-mod`.

**Target: arismoko/horizon's mod API.** Lyle chose to build the mod on **arismoko/horizon**, the modding fork where
Horizon's mod API is made (`horizon-mod-api`/`mod_api`, a side-split `Host`, handles, Parts, attachments, slots, the
internals tier, wasm mods). Its work keeps merging upstream (upstream's `docs/design/mod-hotload.md`: "C and E are
arismoko's work, and we build on it"). The first version of this plan proposed native hook points directly to
upstream (`native-hooks.json`, horizon-mixins bundles). That version is kept at `eb0f38881` and superseded by this
one.

**Goal.** Players install Cubic Chunks as a native mod on that API. Cubic's own code lives in the mod crate. The
engine gives it the few general features the API still lacks, proposed as issues on arismoko/horizon, in its format.

**Recommendation in one paragraph.**
- Most of Cubic (45k of its 53k lines) is in its own files and moves into the mod crate as is.
- Of the dozen engine needs the inventory found, the fork's API already covers some fully: mod messages, F3 debug
  entries and config files.
- It covers others in part:
  - dimensions and chunk generators (#1742, but a column at a time, within −2032..2031);
  - the spawn position;
  - refusing a player at login;
  - render stages.
- Slots and the internals tier reach a few more: the world-created moment, the status report, command-line flags.
- What's left is **ten issue-sized asks** (section 4). Lifting the build-height limit for #1742's dimensions comes
  first.
- **The far view is the hardest.** Its GPU generator and mesher can't cross `mod_api` (no wgpu types do, #808), so it
  needs a far-terrain ask on the render roadmap, or stays on internals as a stopgap.

**Not in v1:**
- **Vertical shards.** LAN guests share one window again, as in v1.
- **The renderer's re-key on a window slide.** It waits for the "windowed levels" ask.

## 1. Where the fork stands

```
horizon-rust/horizon          upstream; main 35f3dad20 on 2026-11-07
  └─ ThatCakeIsAPie/horizon-ttm     To The Max (synced to upstream c4e4800d4 on 2026-11-05)
       └─ ThatCakeIsAPie/horizon-cubic   this repo; main 70922aea5
```

- **Merge-base with upstream:** `c4e4800d4` (2026-11-05). Upstream has moved **2,061 commits** since.
- **Since the merge-base the fork adds** 236 files, +58,187 −831: 101 new files (48k lines) and 135 edited upstream
  files (+10,101 −831). That diff mixes two projects:

| Part | Range | Size |
|---|---|---|
| To The Max (TTM) sync | `c4e4800d4..08b8ebd84` | 70 files, +5,518 −290 |
| **Cubic Chunks** | `08b8ebd84..main` (starts at `4deea333f`, "Cubic Chunks phase 1") | 182 files, +52,687 −559 |

- **Cubic alone:** 91 new files (45,189 lines) and 90 edited upstream files (+7,498 −559; HANDOFF.md is the 91st).
- **Upstream churn on what Cubic edits:** 54 of those 90 files have changed upstream since the merge-base. They hold
  5,291 of Cubic's added lines.
  - `game.rs` lost 6,893 lines upstream (split into modules).
  - `gfx.rs` gained 1,128, and `lan/hub.rs` 1,097.
  - A rebase would be heavy. A port onto hooks is the better way forward anyway.

### Not Cubic (stays out of the mod)

TTM's features are in the same fork but aren't Cubic. Lyle's decisions (2026-11-07):

| Feature | Decision |
|---|---|
| Ultimine | its own mod |
| Big Ore Deposits (`tools/ttm/big-ore-deposits`) | its own data pack |
| Data-pack biomes' ambient particles, music and ambient sounds (`a123999ca`) | offer upstream as a vanilla-parity PR (below) |
| The speed-ups (table below) | for each one upstream lacks, Lyle chooses: suggest it again, or leave it |
| Windows cursor fix; Create World's 26.2 pack upgrade | dropped (upstream already has the upgrade tool itself, `upgrade-datapack`, from Lyle's PRs #3369–#3374) |

**Data-pack biome effects upstream.** Upstream (35f3dad20) reads `minecraft:visual/ambient_particles` and the
music attributes only from vanilla's own `data/minecraft/worldgen/biome/<name>.json` (`animate_tick.rs:383`,
`music.rs:197`). So a data pack's biomes in other namespaces get no particles, music or ambient sounds. The fork reads
them from the world's biome table (`horizon-client-render/src/biome.rs`). That is vanilla behaviour upstream still
lacks: a parity PR.

**The speed-ups.** Lyle remembers most of them as merged upstream. Checked against upstream main 35f3dad20 by their
code (the fork's types and functions) and by behaviour, **none of them is upstream yet**. What was merged from
Lyle's account is related work: the level-shape series (#3480–#3490), data-pack biomes (#3491, #7393, #7528), the
data-pack upgrade (#3369–#3374), `perf/worldgen-surface-biomes` (#3366) and `perf/far-saved-chunks` (#3391).

| Speed-up | Fork commit | In the fork's main | Upstream (35f3dad20) |
|---|---|---|---|
| Stronghold ring cache (`ring_cache.rs`, `keep_rings`: positions kept with the world) | `7b7964755` | yes | no |
| `/locate biome` off the game thread, on all cores | `82d0f525a`, `fe8e063ce` | yes | no (still on the game thread) |
| Far terrain bake (`lod/bake.rs`) | `2c93f3585` | yes | no |
| Lattice reuse in the noise interpolator (`LatticeSamples`) | `bfec4acb2` | yes | no |
| Biome search hints (`SearchHint`) | `80535105c` | yes | no |
| Decoration feature bitsets (`set_feature_bit`) | merged in `142c137d1` | yes | no |
| Lazy jigsaw templates (`LazyTemplate`) | `242467244` | yes | no |

Each would be its own PR. Upstream has changed the same code since, so each one needs a rebase and a fresh
measurement before it's suggested.

General changes that sit inside the Cubic range but aren't Cubic (plain PRs of their own, to upstream or the mod API fork):
- `remote_chunks.rs`: `blocks_of` split out of the section decoder;
- `generator_decoration.rs`: `spawner_nbt` and `loot_table_nbt` extracted;
- `simple.rs`: the temple, shipwreck and swamp-hut height decision moved out of `post_process`
  (`LegacyPiece::settle`, `height_from_terrain`; behaviour-preserving);
- `world_loading.rs`: the `configure_streaming` split;
- the column-replace diff in `streaming.rs` (keep sections that are `same_as` the new ones, remesh only the rest);
- the dithered chunk fade-in (`terrain.wgsl`, `oit_world.wgsl`): a vanilla-look change, argued on its own or dropped.

## 2. Inventory

### 2.1 Cubic's new files: move into mod crates as they are

| Lines | Where | What |
|---|---|---|
| 11,237 | `horizon-client-render/src/gfx/` (17 files) | GPU far view: `cube_gen` (+ WGSL), `cube_descent`, `cube_mesh`, `cube_far`, `biome_gen`, `surface_tile`, culling and draw shaders, `cwg_noise.wgsl` |
| 10,005 | `horizon-world/src/cubic/custom/` (16) | the CubicWorldGen port: settings, density, bounds, biomes, replacers, carvers, populate, structures, vegetation |
| 6,222 | `horizon-client-render/src/lod/` | `cube.rs` (octree planner, workers, coverage, late content), `cube_cache.rs` (disk cache) |
| 4,027 | `horizon-client/src/title/` | `cubic_customize.rs`, `cubic_map.rs`, `cubic_preview.rs` |
| 3,321 | `horizon-client/src/cubic.rs` | `CubicLevel`: cubes, population, records, columns from cubes, the ring |
| 1,710 | `horizon-world/src/cubic/` | `store.rs` (3D region store), `window.rs`, `heights.rs` |
| 1,060 | `horizon-client/src/cubic_lod.rs` | far sampler over the level |
| 881 + 451 | `cubic_convert.rs` (+ tests) | vanilla world → Hybrid copy |
| 740 / 738 / 395 / 247 | `cubic_gen.rs`, `cubic_shards.rs`, `cubic_prefetch.rs`, `cubic_settings.rs` | generation glue, vertical shards, prefetch ring, settings files |
| 645 | `horizon-client/src/game/cubic.rs` | window following and sliding on the server |
| 279 | `horizon-protocol/src/cubic.rs` | `horizon:cubic_*` payloads |
| 71 | `worldgen/overworld_climate.rs` | climate adapter |
| ~2,900 | `tools/`, `docs/`, `research/`, examples | benches, checks, docs; tools move with the mod, research stays in the fork |

### 2.2 Cubic's edits to upstream files, by what they need from the engine

Inventoried hunk by hunk (two passes over `git diff 08b8ebd84 main`). Line counts are approximate.

| Need | Main files | ≈ lines | What the edits do | Disposition |
|---|---|---|---|---|
| **Height and shape** | `chunk.rs`, `world.rs`, `streaming.rs`, `gfx.rs`, `gfx/resident.rs`, `gpu_culling.rs`, `occlusion.rs`, `capture.rs`, `section_mask.rs`, `save.rs`, `lan/chunks.rs`, `remote_chunks.rs`, void checks in 4 files | 900 | `LevelShape.cubic` and `ColumnLayout::window`. Heightmaps as `u16` offsets from `min_y` (an `i16` can't hold y near 1,000,000). Section Y as an int in saves. `Chunk`/`World`/`Streaming`/`Renderer::relayout`, which shifts everything keyed by section index when the window slides. `World::void_y()` | Asks 1 and 5 |
| **Generator registration and world types** | `title/create_world.rs`, `title.rs`, `world_loading.rs`, `initial_spawn.rs`, `game.rs` (CLI), `buffet.rs`, `flat.rs` | 600 | The Cubic Chunks world-type button and Customize screen (`Mode::CreateCubic`). Writes `cubicgen.json` and `horizon_cubic.json` at creation. Edit World › Convert to Cubic Chunks… with a background job. `--cubic` and `--convert-cubic`. A generator-chosen spawn | Asks 2 and 7; slots for flags and world created |
| **Storage and persistence** | `save.rs`, `game/persist.rs`, `game/entity.rs`, `horizon-core/ticks`, `game.rs` | 330 | `SaveDir.cubic`, so `save_chunks`/`save_entities` go to cube records. `EntityScope` (columns plus a cube-Y range) replaces the chunk set in the entity saves. `ChunkTicks::take_if`/`add_saved` move ticks between cubes and columns | Ask 4 |
| **Level backend** | `world_loading.rs` (`make_streaming`), `levels.rs` | 250 | Builds `CubicLevel`/`CubicTerrain` instead of the vanilla column save for a cubic shape. Opens a shard over the primary level's store | Asks 3 and 4 |
| **Streaming and LAN protocol** | `lan/hub.rs`, `lan/guest.rs`, `lan/conn.rs`, `lan/mod.rs`, `lan/handoff.rs`, `lan_actions.rs`, `remote.rs`, `remote_chunks.rs`, `net.rs`, `protocol/buf.rs`, `item.rs`, `chunk.rs` | 1,500 | The configure handshake: refuse non-Horizon clients, switch to wide block positions. The hub's per-guest window, cube queue and paced `horizon:cubes` batches, plus window and light updates. The client's `horizon:cubic_window`/`cubes` handlers and in-place relayout. A process-wide `wide_positions` flag in the protocol | Mod messages; asks 1, 5, 8 and 9 |
| **Vertical shards** | `levels.rs`, `game.rs`, `hub.rs`, about 20 gameplay files (`beds`, `bossbars`, `raids`…) | 600 | `LevelId {dim, shard}` replaces `Dim` everywhere a level is chosen. Every "guest is here" test also checks the shard | Drop for v1 (section 5.3) |
| **Lighting** | `light.rs`, `chunk_map.rs`, `chunk.rs` | 760 | `relight_entering`: incremental sky and block light for entering layers (369 lines, self-contained). `sky_covered` on `LightWorld` and `Chunk`: no sky above a window whose column is covered higher up | `relight_entering` into the mod on internals; ask 5's window event |
| **Far terrain and LOD rendering** | `gfx.rs`, `gfx/lod.rs`, `lod/mesh.rs`, `lod.rs`, `game.rs`, `render_tick.rs`, `levels.rs`, `streaming.rs` | 1,350 | `FarKey {Tile, Cube}`. `cube_lod` replaces the column LOD in a cubic level. About 15 renderer touch points: fields, mesh walkers chained with `lod_cubes`, prepare, opaque and translucent draws. `Renderer::use_gpu_cube_lods` and friends. A public material lookup split out of `lod/mesh.rs` | Ask 6 |
| **Worldgen surface** | `generator.rs`, `generator_structures.rs`, `structure/start.rs`, `legacy.rs`, `simple.rs`, `temple.rs`, `feature_graph.rs`, `live.rs`, `feature_world.rs`, `growth.rs`, `climate/index.rs`, `noise/stack.rs` | 800 | Public accessors (`behaviours`, `temperature_settings`, climate `values`/`tree`, noise tables). `StructurePlanner`/`SetTuning`. `place_in_box`, `settle_start`. Biome-checked feature placement. Placement gates and height overrides on `LevelFeatureWorld` | Internals for now (each logged as a request); `StructurePlanner` moves into the mod |
| **UI and tooling** | `graphics.rs`, `graphics_ui.rs`, `debug_lines.rs`, `control.rs`, `agent.rs`, lang files | 650 | Vertical Distance, Far Terrain on GPU and Cubic Far Detail settings. F3 `Cubic:`/`Far:` lines. `status.cube_lod`. Agent modes for the Cubic screens | Existing `native_settings`/configs, hooks F1–F2 |

The cross-cutting problems, in order of cost:
1. **Section index, not section Y.** The renderer, streaming and world key sections by their index in the column.
   A sliding window therefore re-keys about ten structures (`Renderer::shift_layout`, `Resident::rekey`,
   `gpu_culling` `Records`/`Fades`, `Occlusion::relayout`, `Copies::rekey`, `SectionMask`, `DirtySections`,
   `Streaming::shift_layout`, `Chunk::relayout`).
2. **The level is a `Dim`.** Shards made it `(Dim, shard)` in about 20 files.
3. **The title screen is a closed `enum Mode`.** It is matched in about 15 places.
4. **The renderer has no pass registry.** Far terrain is wired into it at about 15 places.
5. **Block positions on the wire are switched by a process-global flag.**

## 3. What arismoko/horizon already has

Researched on `arismoko/main` (8ce9e9e679, 7,226 commits past upstream's main, 8 behind) and its book
(<https://arismoko.github.io/horizon/>, sources in `docs/modding/src`).

**How a mod reaches the game:**
- **Hooks**, from the catalog in `crates/horizon-mod-api/src/hooks/`. Each takes its NeoForge 26.3 name, side and
  contract where a counterpart exists. Registration goes through `Hooks::register` and the `Registrar`. Behaviour
  attaches through Parts and attachments.
- **Slots.** Every `Game` method (and every function of the game module) has a head and a return point compiled into
  the **prebuilt game**, filled at load. Lower crates get only the functions listed in
  `crates/horizon-mod-api-gen/slots/`, each measured first, because a slot on a hot function costs on every call.
  - API form: `head`/`return`, records of API-typed arguments, `cancel` and `result`.
  - Raw form (native only): the real receiver and arguments.
  - Refused for now: `replace`, `modify_arg` and locals.
- **The internals tier.** A mod declares the game internals it uses: native, hash-pinned, logged. Each one counts as
  a request for an API. Listed call sites add `before_call`, `after_call`, `modify_value` and `wrap_call` (native).
- **Helpers** splice source and rebuild the game. They are the last resort, native only, and always come with a hook
  request.
- **The direction:** "one clean break" (no shims; an old mod rebuilds), "API first", and #589's estimate that slots
  plus internals cover about 98% of today's raw mixins.

**The needs from section 2, mapped:**

| Need (first version's name) | Status on arismoko/horizon | What's there | What's missing |
|---|---|---|---|
| Level shapes (A1) | **partial** | `Registrar::register_dimension(name, DimensionSettings)` (#1742); `Host::level_dimension_shape(LevelRef)` (#1268); data-pack dimensions as real levels (`Dim::Extra`, `LevelShapes::extra`) | build height capped at what the codec takes (`min_y` from −2032, top at most 2031); one shape per dimension at startup, not per world; one biome |
| Level backend: generator (A2) | **partial** | `Registrar::register_chunk_generator` with `ChunkGenerator::build_terrain(&mut dyn ChunkAccess)` on worldgen workers | a column of fixed height, terrain only (no population or features); no cubes, no 2×2×2 population, no structure placement through it |
| Level backend: storage (A2) | **gap** | `level_save` is a notice only; #857 (typed chunk attachments on load and save) is open | a mod storing a level's chunks or cubes and their entities |
| Level backend: far source (A2) | **gap** | none | see far terrain |
| Spawn origin (A3) | **partial** | hook `create_spawn_position` (`LevelEvent.CreateSpawnPosition`) | whether it can set the position is unverified; else a head slot on the spawn search |
| World types on Create World (B1) | **gap** | #888 (open): `register_preset_editors`, a preset's editor screen | a mod's world type (with its own level shapes and generator) as a Create World choice |
| World created (B2) | **slots** | a return slot on the world-creation function in `world_loading.rs` | none |
| Screens over the title menu (B3) | **partial** | mod screens open only over the world (`native_mod_screen_open`) | opening over the title menu and Create World; #903 (a hook before a screen builds its widgets) is near |
| Edit World actions (B4) | **gap** | none | an action on Edit World with a background job (Convert to Cubic Chunks…) |
| Command-line flags (B5) | **slots** | a head slot on option parsing in `game.rs` | none, if its arguments are API-typed; else internals |
| Mod messages (C1) | **exists** | `Host::net_send_to_player`/`net_send_to_server`, `net_from_client`/`net_from_server` | caps of 30 KB and 32 messages a tick each way |
| Configuration handshake (C2) | **partial** | `player_login` refuses with a reason (after configuration); `player_client_brand`; #800 (open, mod dependencies before joining) | adding configuration-phase payloads, refusing during configuration |
| Wide block positions (C3) | **gap** | none | wire positions past the packed ±2048 Y, negotiated per connection |
| Per-guest streaming (C4) | **partial** | `chunk_watch`, attachments on players | a paced bulk lane per player (cube data is far above the message caps) |
| Windowed levels / relayout (D1) | **gap** | none; #1268 and #967 are reads | a level whose loaded vertical range moves with its players |
| Far terrain (E1) | **gap** | none | a mod's far-terrain source drawn by the engine |
| Render layers (E2) | **partial** | `render_level_stage`, `submit_custom_geometry` (unlit, before opaque), `world_draw_context`; open #864–#867, #884, roadmap #808 | lit, fogged geometry in the opaque and translucent passes with the world's depth; mod GPU work (#808 keeps wgpu out of `mod_api`) |
| Block materials (E3) | **gap** | none | the engine's block material lookup for a mod's own meshing (folds into far terrain) |
| F3 lines (F1) | **exists** | `register_debug_entries`, `debug_entry_lines` (`DebugEntry { id, group }`) | `DebugGroup` is a closed enum (Cubic's lines join an existing group) |
| `horizon-ctl status` (F2) | **slots** | a return slot on the status function in `control.rs` | none, if API-typed |
| Settings (F3) | **exists** (configs) | `register_configs`, the config-loaded hooks | rows on the Graphics page (Vertical Distance, Far Terrain on GPU, Cubic Far Detail); the configs screen may do |

## 4. The asks: issue-sized proposals for arismoko/horizon

Only the gaps, in the fork's format (Ask, Hook or handle, NeoForge 26.3 counterpart, Who uses it, Phase), one
concern each. The hook-request form in `injections.md` is used where the ask is one hook. Order is the order they
unblock Cubic.

| # | Ask | Hook or handle | NeoForge 26.3 counterpart | Who uses it | Phase |
|---|---|---|---|---|---|
| 1 ([#1903](https://github.com/arismoko/horizon/issues/1903)) | **Levels taller than the codec's build height.** Lift #1742's −2032..2031 cap for Horizon-only levels, and the data-pack path's. Section Y as an int in saves, heightmaps relative to `min_y`, `SectionMask` past 32 sections (done), and wide block positions on the wire for Horizon clients, negotiated per connection; vanilla clients refused with a message | `DimensionSettings` accepts larger `min_y`/`height`, with an opt-in (`horizon_only: true`) | none (vanilla's codec stops at 4064 blocks) | Cubic Chunks; tall-world packs; sky and void worlds | Hooks |
| 2 | **Per-world level shapes and generators: world types.** A mod's world type on Create World that chooses its levels' shapes and generators for that world, with its editor screen (#888's) | `Registrar::register_world_type(id, WorldType { label, levels, editor })` | the `WorldPreset` registry and `RegisterPresetEditorsEvent` | Cubic Chunks (Custom, Hybrid, Flat Cubic); amplified-like and custom-noise presets | Hooks |
| 3 | **Cube generators.** A generator that fills 16³ cubes on worldgen workers, any height, with population over a 2×2×2 region afterwards and the structure placement it calls | `register_cube_generator` with `CubeGenerator::{generate(&mut CubeAccess), populate(&mut RegionAccess)}` | none (Cubic Chunks' own `ICubeGenerator`) | Cubic Chunks; sky-island and deep-world generators | Hooks |
| 4 | **Level storage.** A mod stores a level's chunks or cubes, with their entities and ticks; the region files stay the default | `register_level_storage` with a `LevelStorage` trait (load and save of columns or cubes, entities by area) | none | Cubic Chunks (cube regions, uniform-cube sidecar); one-file region formats | Hooks |
| 5 | **Windowed levels.** A level whose loaded vertical range is a window that moves with its players. The engine owns the relayout of everything keyed by section index (world, streaming, renderer, light); a listen hook says what entered | `LevelRef::set_window(min_y, height)`, hook `level_window_changed` | none | Cubic Chunks; any level taller than what can be loaded at once | Hooks |
| 6 | **Far terrain.** A client far-terrain source (nodes of voxels or meshes by level) that the engine plans, meshes, caches and draws, lit and fogged like its own; the engine's block material lookup with it | `register_far_terrain` with `FarTerrainSource`; on the render roadmap #808 (with #864–#867) | none (Distant Horizons and Voxy are separate mods there) | Cubic Chunks' far view; Distant-Horizons-style mods | Rendering |
| 7 | **Screens over the title menu, and Edit World actions.** Mod screens on the title menu and Create World (#888's editors need it too), and an Edit World action that runs a background job with progress | mod screens' open target; `register_world_actions` | `ScreenEvent` (screens); none (Edit World) | Cubic Chunks (Customize, Convert to Cubic Chunks…); world optimizers and converters | Hooks |
| 8 | **Configuration-phase handshake.** Add payloads and refuse a client during configuration, with a message | hook `configuration_tasks` | `RegisterConfigurationTasksEvent`, `ICustomConfigurationTask` | Cubic Chunks (wide positions, Horizon-only levels); any server needing a mod on the client | Hooks |
| 9 | **A bulk lane for mod messages.** A per-player paced lane above the 30 KB / 32-a-tick caps, for a mod that declares it | `Host::net_send_bulk(player, channel, data)` | `PayloadRegistrar` with `PacketDistributor` (no such caps) | Cubic Chunks (cubes to LAN guests); maps and schematics | Hooks |
| 10 | **Debug groups.** A mod's own F3 group | `DebugGroup::Mod(id)` | none | Cubic Chunks (`Cubic:` and `Far:` lines); any mod with its own F3 lines | Hooks |

Covered without asks (slots or existing API): world created, command-line flags, `horizon-ctl status`, settings via
configs, the spawn position if `create_spawn_position` can set it (to verify; otherwise a slot).

**Not asks.** These are kept in the mod on internals until an ask covers them. Each internal is logged and counts as
a request:
- the window-slide relight (`relight_entering`) before ask 5;
- the GPU far view before ask 6.

Vertical shards stay out (section 5.3).

## 5. The mod

### 5.1 Format and layout

A native mod in the fork's format (`docs/modding/src/how-to/native-mods`, `reference/manifest.md`):
- a flat project with `horizon-mod.toml`, `Cargo.toml` and `src/`, implementing `Hooks`;
- it depends on `horizon-mod-api` (lib `mod_api`), never on the game, so `horizon-client` is refused with
  `E_GAME_DEPENDENCY`. The game's internals come through the generated `horizon-internals` facade, declared with
  `cargo horizon-mod internals sync`;
- dependencies only from registry crates already in Horizon's lock file, at matching versions; path and git
  dependencies are refused.

wasm is out: chunk generators are native only, and the far view is GPU work.

```
ThatCakeIsAPie/horizon-cubic           this repo (Lyle, 2026-11-07)
  mod/
    horizon-mod.toml                   [mod] id = "cubic"; [native] horizon = "<commit>" (the prebuilt game it targets)
    Cargo.toml                         horizon-mod-api, horizon-internals (generated), registry crates in Horizon's lock
    src/
      lib.rs                           Hooks: register (dimensions/world types, generator, storage, far terrain, configs)
      world/                           horizon-world/src/cubic/** (store, window, heights, the custom generator, structures)
      level/                           CubicLevel, generation glue, prefetch, convert, settings files, payloads
      far/                             cube octree planner, cache, the voxel mesher; the GPU generator on internals until ask 6
      ui/                              Customize, map and preview as mod screens (after ask 7)
    data/lang/{en_us,pt_br}.json, data/presets/*.json
  tools/                               the benches and checks, run against the mod build
```

### 5.2 Install path

What is documented, and what isn't:
- **On the fork:** the launcher builds a native mod against the **prebuilt game**: a `prebuilt-<commit>` prerelease of
  arismoko/horizon's main, published daily (seen: `prebuilt-fba095d919`, 2026-11-07). It builds in seconds when the
  mod needs only the API and internals. Helper injections force a full game build (1½ to 3½ minutes; the first build
  downloads Rust, 20 to 30 minutes). Mods install from the Mods page, from a GitHub release (`mods add-github`), or as
  `horizon-native-mods/<id>.hmod`. The registry lists native mods after a review.
- **The fork's code hardcodes it:** `crates/horizon-mod-package/src/index/native.rs` sets `PREBUILT_REPO =
  "arismoko/horizon"`, commented "Diverges from upstream: horizon-rust/horizon has none".
- **Upstream (horizon-rust/horizon) has no `horizon-mod-api` crate and no prebuilt route.** Its launcher (0.2.50)
  installs `.hmod` content and wasm mods and builds horizon-mixins bundles. So a mod on the fork's API plays only on
  the fork's game until the API reaches upstream.
- **Unknown, to ask arismoko/Aria:**
  - how a player gets a launcher that uses the fork's prebuilt games: a build of the fork's launcher, a setting or
    channel in the official one, or importing a fork build (`build_import.rs`, "a horizon-client built on your fork's
    Actions");
  - whether and when `mod_api` and the prebuilt route go upstream.

### 5.3 Vertical shards

Left out of v1 (Lyle). LAN guests share one window over the union of their ranges, capped at 99 cubes, as in v1. For
a later version, ask for several loaded copies of one level by key, which is also useful for per-player instances.
Then bring `cubic_shards.rs` back on it.

## 6. Roadmap

| Phase | What | Done when |
|---|---|---|
| 0 | Lyle approves the texts (section 8). We file the tracker issue and ask 1 on arismoko/horizon, then `/claim` ask 1 | both filed: #1902 and #1903 (2026-11-07); the claim waits for Lyle |
| 1 | Ask 1, then asks 3, 4 and 5 (generator, storage, windows), each as its own issue and PR, in the fork's PR template with `pr_ready.py` | merged on arismoko/horizon |
| 2 | Asks 2 and 7 (world types, screens), then 8 and 9 (network), then 6 (far terrain, with the render roadmap) and 10 | merged |
| 3 | Port Cubic into `mod/`: the crate on `mod_api`; internals for what an ask hasn't covered yet; slots for world created, flags and status. Shards dropped. **Hybrid tested first**: create, play across the vanilla band's edges, save, reopen, convert a vanilla world; it ships only if it works | the fork's tests pass in the mod, and `cube_load_bench` and `cubic_far_bench` are within noise of the fork |
| 4 | Build through the launcher on the prebuilt game, on a clean instance; existing Cubic worlds open | a Custom, a Hybrid and a converted world play and save |
| 5 | Release `cubic.hmod` on GitHub (`mods add-github`); registry listing after review | installs from GitHub |
| — | Until then, this fork's own builds stay the way to play Cubic (Lyle) | |

## 7. Risks

- **Acceptance and pace.**
  - The fork is maintained by an agent swarm (Dyna) with its own roadmap (#771) and board.
  - Asks 3 to 6 are big, and the maintainers may want another shape, or fold them into their tracks:
    - ask 6 into rendering #808;
    - ask 4 into chunk attachments #857;
    - ask 2 into preset editors #888.
- **One clean break.** The API has no shims: a changed hook breaks the mod until it rebuilds. The mod follows the
  fork's main.
- **Two games.** Until `mod_api` reaches upstream, the mod runs only on the fork's builds, and the player's way to get
  them is unknown (section 5.2).
- **GPU far view.** #808's rule (no wgpu across `mod_api`) means the GPU generator, descent and mesher run on
  internals until ask 6 lands. Ask 6 must let a source do its own GPU work, or the far view falls back to the CPU
  sampler, and the far fill gets slower.
- **Hot paths.** Slots cost on every call. Cube generation, far workers and light stay inside the mod's own objects,
  reached once through the generator, storage and far-terrain traits.
- **Upstream churn.** 2,061 upstream commits since the merge-base, and 54 of the 90 edited files changed upstream.
  The port rewrites the glue.
- **Save compatibility.** Existing Cubic worlds (`horizon_cubic.json`, `cubicgen.json`, cube records, the far cache)
  must open unchanged under the mod.

## 8. Issues on arismoko/horizon

Lyle approved both ("Post both"). They were posted from Lyle's account on 2026-11-07 with this text (hard line
breaks unwrapped):
- the tracker: [#1902](https://github.com/arismoko/horizon/issues/1902);
- ask 1, taller levels: [#1903](https://github.com/arismoko/horizon/issues/1903).

Not done yet, each waiting for Lyle's OK:
- linking #1903 from the tracker's list;
- `/claim` on #1903;
- asking Aria about the install path (section 5.2).

### 8.1 The tracker

> 🤖 Filed by ThatCakeIsAPie's AI agent (Claude Code, Opus 5.5), not by ThatCakeIsAPie personally.

**Cubic Chunks as a mod: what it needs from the mod API**

**Ask.** Cubic Chunks (vertically unbounded worlds in 16³ cubes, CubicWorldGen's generator, and a far view that
reaches the whole world) is playable today as a fork of Horizon (ThatCakeIsAPie/horizon-cubic). We'd like it to be
a native mod on `mod_api` instead. Most of it is its own code. This tracker lists what the API lacks, as one issue
each; we'll file and claim them one at a time, starting with the first.

**What's covered already.**
- mod messages (`net_send_to_player`);
- debug entries;
- configs;
- dimensions and chunk generators (#1742), within the codec's build height;
- the spawn position (`create_spawn_position`);
- slots for the world-created moment, command-line flags and the status report.

**The asks.**
1. Levels taller than the codec's build height (#1742's cap), with wide block positions for Horizon clients.
2. World types on Create World that choose their levels' shapes and generators per world (with #888's editors).
3. Cube generators: 16³ cubes on worldgen workers, with population over 2×2×2 regions.
4. Level storage: a mod stores a level's chunks or cubes and their entities (near #857).
5. Windowed levels: a level whose loaded vertical range moves with its players, with the relayout owned by the
   engine.
6. Far terrain: a client far-terrain source the engine plans, meshes and draws (with #808).
7. Mod screens over the title menu and Create World, and Edit World actions with a background job.
8. A configuration-phase handshake (`RegisterConfigurationTasksEvent`).
9. A bulk lane for mod messages, above the 30 KB / 32-a-tick caps.
10. A mod's own F3 debug group.

**Who uses it.** Cubic Chunks first, and each ask names its other users.

**Phase.** Hooks

### 8.2 Ask 1

> 🤖 Filed by ThatCakeIsAPie's AI agent (Claude Code, Opus 5.5), not by ThatCakeIsAPie personally.

**Levels taller than the codec's build height**

**Ask.** A native mod's dimension (and a data pack's) can be taller than vanilla's codec allows, for clients that
are Horizon. Today `DimensionSettings` refuses a `min_y` below −2032 or a top above 2031 (#1742), because vanilla's
codec and packed block positions stop there. Cubic Chunks needs levels millions of blocks tall. Its fork made these
work:
- section Y written as an int when it doesn't fit a byte;
- heightmaps stored relative to `min_y`;
- the 32-section column masks gone (done upstream: `SectionMask`);
- block positions on the wire as three ints, negotiated per connection, with a vanilla client refused with a message.

**Hook or handle.** `DimensionSettings` accepts any multiple-of-16 `min_y` and `height` within ±30,000,000 when the
mod sets `horizon_only: true`. The engine then:
- negotiates wide block positions with the client during configuration;
- refuses a vanilla client entering such a level, with a message;
- saves section Y and heightmaps in the wide forms.

**NeoForge 26.3 counterpart.** none (vanilla's `DimensionType` codec stops at a height of 4064).

**Who uses it.** Cubic Chunks (ThatCakeIsAPie/horizon-cubic); tall-world and deep-world dimensions; sky and void worlds.

**Phase.** Hooks

## 9. Decisions and open questions

Lyle's answers (2026-11-07):
- **Where to propose:** arismoko/horizon, the mod API fork (Lyle, "rework it there"). We file the issues there. Each
  issue or PR text is shown to Lyle before it's posted.
- **Filing:** we file the proposals ourselves.
- **Interim release:** none planned. The fork's GitHub builds are the stopgap, and the goal is the hooks. A
  pinned-Nightly mod release is a fallback, only if the hooks stall.
- **Dithered fade-in:** keep it. It is proposed upstream as its own feature, separate from the Cubic hooks.
- **Where the mod lives:** this repo, ThatCakeIsAPie/horizon-cubic. No new repo.
- **Hybrid:** kept if it works. It's tested before it goes into v1 (roadmap phase 3). Its vanilla band needs the
  vanilla generator next to the cube generator (ask 3 should allow a cube generator to wrap the vanilla one).

- **Vertical shards:** left out of v1, kept as a suggestion for a later version (section 5.3).
- **TTM's features:** Ultimine becomes its own mod and Big Ore Deposits its own data pack. Data-pack biome effects go
  upstream as a parity PR. Each speed-up upstream lacks (all seven, section 1) is Lyle's choice: suggest it again or
  leave it. The Windows cursor fix and the 26.2 pack upgrade on Create World are dropped.

Open, for arismoko/Aria rather than Lyle: how players get the fork's prebuilt-game route, and whether `mod_api` goes
upstream (section 5.2). Texts are still shown to Lyle before they're posted.
