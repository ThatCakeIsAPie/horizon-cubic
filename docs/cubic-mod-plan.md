# Cubic as a Horizon mod: plan

Status: research and design only (2026-11-07). Nothing here has been proposed to upstream yet; nothing is
pushed. Branch `plan/cubic-mod`.

**Goal.** Players install Cubic Chunks from the launcher as a native mod (a horizon-mixins bundle released as an
`.hnmod` and listed in the registry), on a current Horizon Nightly. Cubic's own code lives in the mod's crates. The
engine gives it a small set of new, general, versioned hook points, proposed upstream, instead of the ~7.5k lines of
edits the fork makes to upstream files today.

**Recommendation in one paragraph.** Most of Cubic (45k of its 53k lines) is already in its own files and moves into
mod crates as is. The rest reduces to about a dozen hook families plus a dozen small, plain upstream PRs. Five of
the hook families do almost all the work:
- level shapes,
- a level backend (generator, storage, far source),
- world types on the Create World screen,
- a networking handshake with per-guest streaming,
- pluggable far terrain with render phases.

Two things are not worth hooking for a first release:
- **Vertical shards**: drop them, and LAN guests share one window again, as in v1.
- **The renderer's re-key on a window slide**: a narrow "relayout" API upstream, or an OVERWRITE on a pinned Nightly
  until upstream takes it.

Start with an RFC issue upstream (draft at the end), then the plain PRs, which are easy to accept and unblock the rest.

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

TTM's features are in the same fork but are not Cubic. Each should become a plain upstream PR, a mod of its own, or
stay fork-only (open question 2):
- the stronghold ring cache; `/locate biome` off the game thread;
- the Windows cursor fix; the 26.2 data-pack upgrade on Create World;
- the far terrain bake;
- lattice reuse, biome search hints, decoration feature bitsets and lazy jigsaw templates (worldgen speed);
- pack biomes' ambient particles, music and sounds;
- Ultimine;
- Big Ore Deposits (`tools/ttm/`).

General changes that sit inside the Cubic range but aren't Cubic (plain upstream PRs, section 4):
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
| **Height and shape** | `chunk.rs`, `world.rs`, `streaming.rs`, `gfx.rs`, `gfx/resident.rs`, `gpu_culling.rs`, `occlusion.rs`, `capture.rs`, `section_mask.rs`, `save.rs`, `lan/chunks.rs`, `remote_chunks.rs`, void checks in 4 files | 900 | `LevelShape.cubic` and `ColumnLayout::window`. Heightmaps as `u16` offsets from `min_y` (an `i16` can't hold y near 1,000,000). Section Y as an int in saves. `Chunk`/`World`/`Streaming`/`Renderer::relayout`, which shifts everything keyed by section index when the window slides. `World::void_y()` | Hooks A1 and D1, plain PRs P1–P3 |
| **Generator registration and world types** | `title/create_world.rs`, `title.rs`, `world_loading.rs`, `initial_spawn.rs`, `game.rs` (CLI), `buffet.rs`, `flat.rs` | 600 | The Cubic Chunks world-type button and Customize screen (`Mode::CreateCubic`). Writes `cubicgen.json` and `horizon_cubic.json` at creation. Edit World › Convert to Cubic Chunks… with a background job. `--cubic` and `--convert-cubic`. A generator-chosen spawn | Hooks A4, B1–B4 |
| **Storage and persistence** | `save.rs`, `game/persist.rs`, `game/entity.rs`, `horizon-core/ticks`, `game.rs` | 330 | `SaveDir.cubic`, so `save_chunks`/`save_entities` go to cube records. `EntityScope` (columns plus a cube-Y range) replaces the chunk set in the entity saves. `ChunkTicks::take_if`/`add_saved` move ticks between cubes and columns | Hook A2 (`ColumnStore`), plain PRs P4–P5 |
| **Level backend** | `world_loading.rs` (`make_streaming`), `levels.rs` | 250 | Builds `CubicLevel`/`CubicTerrain` instead of the vanilla column save for a cubic shape. Opens a shard over the primary level's store | Hook A2 |
| **Streaming and LAN protocol** | `lan/hub.rs`, `lan/guest.rs`, `lan/conn.rs`, `lan/mod.rs`, `lan/handoff.rs`, `lan_actions.rs`, `remote.rs`, `remote_chunks.rs`, `net.rs`, `protocol/buf.rs`, `item.rs`, `chunk.rs` | 1,500 | The configure handshake: refuse non-Horizon clients, switch to wide block positions. The hub's per-guest window, cube queue and paced `horizon:cubes` batches, plus window and light updates. The client's `horizon:cubic_window`/`cubes` handlers and in-place relayout. A process-wide `wide_positions` flag in the protocol | Hooks C1–C4, D1 |
| **Vertical shards** | `levels.rs`, `game.rs`, `hub.rs`, about 20 gameplay files (`beds`, `bossbars`, `raids`…) | 600 | `LevelId {dim, shard}` replaces `Dim` everywhere a level is chosen. Every "guest is here" test also checks the shard | Drop for v1 (section 3.3) |
| **Lighting** | `light.rs`, `chunk_map.rs`, `chunk.rs` | 760 | `relight_entering`: incremental sky and block light for entering layers (369 lines, self-contained). `sky_covered` on `LightWorld` and `Chunk`: no sky above a window whose column is covered higher up | Plain PR P6, hook D1's slide event, `relight_entering` into the mod |
| **Far terrain and LOD rendering** | `gfx.rs`, `gfx/lod.rs`, `lod/mesh.rs`, `lod.rs`, `game.rs`, `render_tick.rs`, `levels.rs`, `streaming.rs` | 1,350 | `FarKey {Tile, Cube}`. `cube_lod` replaces the column LOD in a cubic level. About 15 renderer touch points: fields, mesh walkers chained with `lod_cubes`, prepare, opaque and translucent draws. `Renderer::use_gpu_cube_lods` and friends. A public material lookup split out of `lod/mesh.rs` | Hooks E1–E3 |
| **Worldgen surface** | `generator.rs`, `generator_structures.rs`, `structure/start.rs`, `legacy.rs`, `simple.rs`, `temple.rs`, `feature_graph.rs`, `live.rs`, `feature_world.rs`, `growth.rs`, `climate/index.rs`, `noise/stack.rs` | 800 | Public accessors (`behaviours`, `temperature_settings`, climate `values`/`tree`, noise tables). `StructurePlanner`/`SetTuning`. `place_in_box`, `settle_start`. Biome-checked feature placement. Placement gates and height overrides on `LevelFeatureWorld` | Plain PRs P7–P9. `StructurePlanner` moves into the mod once internals are public |
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

## 3. Hook design

### 3.1 Principles

- **Same style as today:** named, versioned entries in `native-hooks.json`, `native_*` functions with documented
  contracts, `NATIVE_HOOKS_API` raised per addition, never changed (a new version instead). Mixin bundles target the
  name, trait mods get a `Hooks` method.
- **Coarse, not hot.** Cubic's hot paths (cube generation, far workers, light) stay inside the mod's own objects. A
  hook hands the engine a *provider object* once (a generator, a store, a far-terrain source), not a call per block.
  Today's contracts are `listen`, `cancel`, `collect` and `replace`. A provider is `replace` on an `Option<Box<dyn …>>`
  out-parameter that starts `None`.
- **No crate cycles.** The mod's crates depend on `horizon-client-render` and `horizon-world`, and `horizon-client`
  depends on the mod. So a hook point in a lower crate can't call the mod directly: it reads a registry (a
  `OnceLock`'d list of trait objects) that a `horizon-client` helper fills at start, as `horizon_world::loot`'s hook
  already does through `native_load_hooks`.
- **General first.** Each hook is named for what any mod would use it for (level backends, world types, far terrain),
  not for Cubic.

### 3.2 Proposed hook points

Signatures are sketches in the existing style. `LevelOpen` and the like are small context structs.

**A. Levels: shape, backend, spawn (server)**

| Hook | Contract and signature | Call site (fork's main) | Replaces | Beyond Cubic |
|---|---|---|---|---|
| A1 `level_shapes` | collect: `native_level_shapes(&mut self, world_root: &Path, shapes: &mut LevelShapes)` | end of `world_loading::level_shapes` (`world_loading.rs:637`) | `cubic::cubic_shapes`, the `cubic` flag's source | worlds with their own heights per save, without a data pack; void and sky worlds |
| A2 `level_backend` | replace: `native_level_backend(&mut self, open: &LevelOpen, backend: &mut Option<LevelBackend>)`. `LevelBackend { generator: Arc<dyn DynChunkGenerator>, store: Option<Arc<dyn ColumnStore>>, far: Option<Arc<dyn FarSource>> }` | `make_streaming`, after the vanilla `VanillaTerrain` is built (it's passed in `open` so a backend can wrap it, as Hybrid does) | `world_loading.rs:679-960`'s cubic branch, `SaveDir.cubic`, the `save_chunks`/`save_entities` diversions (`save.rs:2934-3030`), `cubic_sampler_in` | generator mods (Terra-style, void and skyblock, custom noise), other save formats (one-file regions, a database), worlds served from elsewhere |
| A3 `spawn_origin` | replace: `native_spawn_origin(&mut self, dim: &str, origin: &mut Option<((i32, i32), i32)>)` | `initial_spawn.rs:106` | the cubic spawn search | any generator whose spawn rules differ |

A2 needs two engine generalizations in the same PR:
- **`DynChunkGenerator`:** `Streaming::new` is generic over a `ChunkGenerator` with an associated `Worker`, so a
  type-erased adapter is needed.
- **`ColumnStore`:** a trait with `save_columns`, `save_entities` and `load`. `SaveDir` holds an optional one, and the
  vanilla region save stays the default.

**B. World types and the title screen (client)**

| Hook | Contract and signature | Call site | Replaces | Beyond Cubic |
|---|---|---|---|---|
| B1 `world_types` | collect: `native_world_types(&mut self, types: &mut Vec<WorldTypeEntry>)`. An entry is `{ id, label_key, allowed(&WorldOptions), customize: Option<mod screen>, settings: Option<String> }` | Create World's world-type cycle (`create_world.rs`) | `WorldOptions.cubic`/`cubic_json`, the Cubic Chunks button and Customize button, `cubic_allowed` | any worldgen mod's presets, e.g. amplified-like presets, floating islands or Terra configs |
| B2 `world_created` | listen: `native_world_created(&mut self, root: &Path, world_type: &str, settings: &str)` | `world_loading.rs:221` | `mark_cubic_world`, `save_world_generation` | the same mods |
| B3 mod screens on the title screen | helper and hook extension of API 111 (`native_mod_screen_open`): also over the title menu and Create World | `title.rs` | `Mode::CreateCubic`, `ConvertConfirm`, `Converting` and their 15 match arms | config screens for any mod before a world exists |
| B4 `edit_world_actions` | collect: `native_edit_world_actions(&mut self, world: &WorldEntry, actions: &mut Vec<WorldAction>)`, plus a background-job progress helper | Edit World | "Convert to Cubic Chunks…", `start_convert`/`poll_convert` | world optimizers, pruners, converters, exporters |
| B5 `command_line` | collect: flags with help text, and a pre-start headless action (`&[String] → Option<exit code>`) | `game.rs` option parsing and start | `--cubic`, `--convert-cubic`, `--keep-bedrock` | headless tools in mods (exports, benches) |

**C. Networking**

| Hook | Contract and signature | Call site | Replaces | Beyond Cubic |
|---|---|---|---|---|
| C1 existing mod messages | API 40 (`native_net_send_to_player`, `native_net_from_client`, …) as is | none (already there) | `horizon:cubic_view` and `horizon:cubic_window` payloads | none (already general) |
| C2 `configure` | replace: `native_configure(&mut self, conn: &ConfigureContext, decision: &mut ConfigureDecision)`, where a decision is accept with extra payloads, or refuse with a reason | `lan/conn.rs:986` (`login_and_configure`) | the Horizon-only refusal and the `MODE_CHANNEL` payload | servers that require a mod, with a clean refusal message |
| C3 wide block positions (engine) | `horizon-protocol`: block positions read and written through a per-connection codec context, negotiated in configuration. A helper turns it on for a connection | `buf.rs` `Pos`, `item.rs`, `chunk.rs` block-entity Y | the process-wide `wide_positions` flag and `WideHold` | worlds past the packed position's ±2048 Y |
| C4 `hub_guest_tick` and `guest_level_changed` | listen: per guest each hub tick, with a bulk-send helper paced by the hub. Per-guest mod state through attachments | `lan/hub.rs` tick loop and `set_level` | `hub.rs:4385-4760`, `Guest.vertical`/`cube_queue` | anything that streams per-player data (maps, schematics, far views of other servers) |

Mod messages are capped at 30,000 bytes and 32 a tick each way. Cube streaming needs either a bulk lane (C4's
helper) or higher caps for a mod that asks for them.

**D. Levels whose vertical window moves (engine)**

| Item | Kind | Where | Replaces |
|---|---|---|---|
| D1 relayout API | engine. `World::relayout(shape, fill)`, `Streaming::shift_layout`, `Renderer::shift_layout`, owned by upstream, plus a helper `native_level_relayout(dim, min_y, height)` and a listen hook `level_relayout` (old and new layout, entering layers) | `world.rs`, `streaming.rs`, `gfx.rs` and the section-keyed stores | the whole re-key cluster (about 300 lines over 10 files), the slide's relight trigger (`streaming.rs:2346`) |

**Alternative to D1:** key sections by absolute section Y (an `i32`) everywhere. That removes the re-keying, but
it's a much bigger change for upstream. D1 keeps today's keys and gives upstream one owned function to maintain.
Beyond Cubic it serves any level whose loaded height changes at run time (tall data-pack worlds loaded around the
player).

**E. Far terrain (client render)**

| Hook | Contract and signature | Call site | Replaces | Beyond Cubic |
|---|---|---|---|---|
| E1 `far_terrain` | replace: `native_far_terrain(&mut self, level: &FarLevel, far: &mut Option<Box<dyn FarTerrain>>)`. `FarTerrain` has `update(camera, view, loaded) -> FarFrame`, `stats() -> Value` and `clear()` | `prepare_lod` (`game.rs:17473`), the per-frame update (`game.rs:17854-17943`), `render_tick.rs:125` | `cube_lod` beside `lod`, `CubeLod` wiring, the GPU path toggles | Distant-Horizons- and Voxy-style far views, other far sources |
| E2 render phases | a registry in `horizon-client-render`: `register_terrain_layer(Box<dyn TerrainLayer>)` with `prepare(encoder, camera, planes)`, `draw(phase: Opaque/Translucent, pass, frame)`, `meshes()` for memory stats. Reached from the client through a helper | `gfx.rs` at about 15 sites: fields, prepare, the two draws, mesh walkers | `cube_far`, `surface_tiles`, `lod_cubes` in `Renderer`, `gfx/lod.rs`'s cube methods, `FarKey` | any mod drawing terrain-like geometry with the world's depth and fog (holograms, voxel overlays) |
| E3 block materials (engine) | `lod::mesh::Material` lookup made public | `lod/mesh.rs` | the `Tint`/`Style` reshaping | any mod meshing blocks |

`native_world_drawn` already exists, but it runs after the world is finished. Far terrain has to draw inside the
opaque and translucent passes, with the world's depth.

**F. Small registries (client)**

| Hook | Contract | Call site | Replaces |
|---|---|---|---|
| F1 `debug_lines` | collect: lines for F3's left or right column | `debug_lines.rs:53, 270` | `debug_cubic_lines` |
| F2 `status` | collect: `&mut serde_json::Value` added to `horizon-ctl status` | `control.rs:1494` | `status.cube_lod` |
| F3 settings | existing `native_settings` / `native_register_configs` | none (already there) | the three Graphics settings in `graphics.rs`/`graphics_ui.rs` |

### 3.3 What isn't a hook, and how the mod handles it

| Edit | Why not a hook | Plan |
|---|---|---|
| Vertical shards (`LevelId {dim, shard}`, about 20 files) | It changes what a level *is*, in every system that picks one | **Drop for v1.** LAN guests share one window over the union of their ranges, capped at 99 cubes, as in v1 (`several_players_share_the_union_of_their_ranges_up_to_99_cubes`). Later, propose an opaque `LevelKey` upstream (useful for per-player instances and arenas) |
| The re-key cluster, if D1 is refused | Ten private structures | OVERWRITE injections on a pinned Nightly, or wait for D1 |
| Title `Mode` arms, if B3 is refused | A closed enum | OVERWRITE on a pinned Nightly. Or make the Customize screen a mod screen opened from the world type (B1) and convert from a CLI/command only |
| Dithered fade-in | Not Cubic | Kept. Proposed upstream as a feature of its own, separate from the Cubic hooks (Lyle, 2026-11-07) |
| `relight_entering` internals (`increase_pass`, `Region`, `Light`) | Private helpers | A `pub` exposure PR (P6), then the code lives in the mod |

## 4. Plain upstream PRs (no hooks, each under ~500 lines)

These are refactors or accessors that make sense without Cubic. They go first: easy to review, and they shrink what
the hooks must carry.

1. **P1** `World::void_y()`, used by the four void checks (`lan_actions.rs:4122`, `motion/environment.rs:127`,
   `mobs/ai.rs:1155`, `arrows.rs:843`).
2. **P2** Heightmaps stored relative to `min_y` (`u16`), instead of absolute `i16`.
3. **P3** Section Y written as an int when it doesn't fit a byte (`save.rs:1432`), and `SectionMask::shifted`.
4. **P4** `ChunkTicks::take_if` / `add_saved` (moving scheduled ticks between containers).
5. **P5** `EntityScope` (columns, optionally a Y range) for entity saves.
6. **P6** `sky_covered` on `LightWorld`/`Chunk`, and `pub(crate)` → `pub` for the light passes `relight_entering`
   needs.
7. **P7** Worldgen accessors: `GeneratorData::behaviours`, `temperature_settings`; climate index `values`/`tree`;
   `Noise::gradient_table`/`as_stack`; public `StructurePlanner` building blocks.
8. **P8** `LegacyPiece::settle`/`height_from_terrain` and `place_in_box` (the `simple.rs`/`temple.rs` refactor).
9. **P9** `place_with_biome_check` and `LevelFeatureWorld` placement gates.
10. **P10** `ChunkGenerator::spread_loads` (a defaulted capability) and the column-replace diff in streaming.
11. **P11** `remote_chunks::blocks_of`, `spawner_nbt`/`loot_table_nbt`, and the `configure_streaming` split.

## 5. The bundle

```
ThatCakeIsAPie/horizon-cubic        this repo (Lyle, 2026-11-07); the bundle in a folder, e.g. mod/
  horizon-mixin.toml                 [mod] id = "cubic"; [compatibility] horizon_revision only while pinned
  horizon-native.json                (in the .hnmod) "hooks": N, the NATIVE_HOOKS_API the hooks need
  helpers/                           thin adapters: each named hook → the crates
    levels.rs                        A1–A3: shapes, backend, spawn
    world_types.rs                   B1–B5
    net.rs                           C1–C4
    far.rs                           E1, E2 registration
    ui.rs                            F1–F3, settings
  crates/
    cubic-world/                     horizon-world/src/cubic/** (store, window, heights, custom generator, structures)
    cubic-level/                     CubicLevel, generation glue, prefetch, convert, settings files, protocol payloads
    cubic-far/                       lod/cube*, cubic_lod, gfx cube_*/biome_gen/surface_tile + WGSL, the voxel mesher
    cubic-ui/                        Customize, map and preview screens (as mod screens)
  data/
    lang/en_us.json, lang/pt_br.json
    presets/*.json                   CubicWorldGen presets
  tools/                             cube_load_bench.py, cubic_far_bench.py, cubic_far_check.py, cubic_structure_check.py
```

```toml
[mod]
id = "cubic"
version = "0.1.0"

[[injection]]
id = "level_backend"
hook = "level_backend"
hook_version = 1
kind = "HEAD"
helper = "helpers/levels.rs"

[[crate]]
id = "cubic-far"
path = "crates/cubic-far"      # depends on horizon-client-render, horizon-world, wgpu (the workspace's versions)
```

The crates depend on the workspace's own `horizon-world`, `horizon-client-render` and `wgpu`, so they always match
the build's versions. `horizon-client` depends on the crates (the default `dependent`).

## 6. Roadmap

| Phase | What | Done when |
|---|---|---|
| 0 | This plan agreed with Lyle. We file the RFC issue (section 8) upstream once Lyle has read its text (each issue and PR text is shown to Lyle before posting) | the maintainers' view on A2, D1 and E2 is known |
| 1 | Plain PRs P1–P11, one at a time (AGENTS.md: small PRs, claims, `pr_ready.py`) | merged upstream |
| 2 | Hook PRs, in the order they unlock:<br>1. A1 `level_shapes`<br>2. A2 `level_backend` (+ `DynChunkGenerator`, `ColumnStore`)<br>3. B1–B3 world types and title mod screens<br>4. C2/C3 handshake and wide positions<br>5. C4 hub streaming<br>6. D1 relayout<br>7. E1–E3 far terrain<br>8. F1–F2 | merged, `NATIVE_HOOKS_API` raised |
| 3 | Port Cubic onto a current Nightly:<br>- move the new files into the four crates;<br>- write the helpers;<br>- for any hook not merged yet, an OVERWRITE injection pinned to that Nightly (for development only: no pinned release unless the hooks stall);<br>- shards dropped | the fork's tests pass in the mod repo, and `cube_load_bench` and `cubic_far_bench` are within noise of the fork |
| 4 | The bundle builds through the launcher (`native-build`) on a clean instance; existing Cubic worlds open. **Hybrid is tested first** (Lyle hasn't played it): create, play across the vanilla band's edges, save, reopen, convert a vanilla world. It ships in v1 only if it works | a Custom world, a Hybrid world and a converted world play and save |
| 5 | `.hnmod` release on the hooks (`hooks: N`), GitHub release. Until then the fork's own builds stay the way to play Cubic. A pinned-Nightly mod release is a fallback, only if the hooks stall upstream | installs from GitHub in the launcher |
| 6 | Registry listing (needs a review) | listed |

## 7. Risks

- **Acceptance.** A2 (level backends), D1 (relayout) and E2 (render layers) are bigger than today's event hooks.
  Upstream may want another design, or none: the plan works on OVERWRITEs pinned to a Nightly meanwhile. The cost is
  a re-port for each Nightly the mod moves to.
- **Upstream churn.** 2,061 commits since the merge-base, and 54 of the 90 edited files changed upstream (`game.rs`
  split, `gfx.rs` and `hub.rs` +1.1k each). The port is effectively a rewrite of the glue, not a rebase.
- **Crate direction.** Hooks in `horizon-world` and `horizon-client-render` must use registries, or the mod crates
  form a cycle.
- **Hot paths.** Far workers, cube generation and light must not go through per-call hooks. Provider objects keep
  them inside the mod.
- **Build time on players' machines.** Native mods build on install. Cubic's crates are about 45k lines plus WGSL.
  That's minutes on a laptop, and memory-heavy (the fork's own builds starve a 7 GB machine).
- **Networking limits.** Mod messages are capped at 30 KB and 32 per tick. Cube streaming needs C4's bulk lane or
  higher caps.
- **Save compatibility.** Existing Cubic worlds (`horizon_cubic.json`, `cubicgen.json`, cube records, the far cache)
  must open unchanged. The record format moves into the mod as is.
- **Losing shards** changes LAN play for guests far apart vertically (back to one shared window).
- **Trust and review.** Native mods are unsandboxed. A registry listing needs a review of a large mod.

## 8. Draft: first upstream issue (for Lyle to review; not posted)

> Filed by ThatCakeIsAPie's AI agent (Claude Code, Opus 5.5), not by ThatCakeIsAPie personally.

**RFC: hook points for level backends (shapes, generators, storage, far terrain)**

Native mods can change almost everything a player does, but nothing about the world's shape, how its terrain is
generated or stored, or how it's drawn far away: `native-hooks.json` has no hook points for any of those today. I'd
like to propose a small, versioned set, and I'm ready to do the PRs.

What it would let a mod do:
- give a world type its own generator (a Terra-style generator, a void or skyblock world, a custom noise router) and
  offer it on Create World with its own settings screen;
- store a level in another format;
- draw far terrain its own way (Distant-Horizons- or Voxy-style), inside the world's passes, with its depth and fog;
- run levels whose loaded height follows the player.

The motivating case is Cubic Chunks: vertically unbounded worlds in 16³ cubes, playable today as a fork. With these
hooks it would become an ordinary native mod, with no fork left.

Proposed, in order (each its own PR, each a new hook version, nothing changed for existing mods):
1. `level_shapes` (collect): a mod adjusts a world's level shapes when it opens.
2. `level_backend` (replace): a mod supplies a level's generator, column store and far source when it opens, with
   the vanilla generator passed in to wrap. This needs a type-erased `ChunkGenerator` adapter and a `ColumnStore`
   trait, with the region save as the default.
3. `world_types` / `world_created`, and mod screens over the title screen: a mod's world type on Create World, with a
   Customize screen.
4. `configure` and a negotiated wide block-position codec: a server can require a mod and refuse clients without it.
5. `hub_guest_tick`: per-guest streaming of a mod's own data, paced by the hub.
6. A relayout API for levels whose vertical window moves.
7. `far_terrain` and a terrain-layer registry in the renderer (prepare, and opaque/translucent draws with the
   world's depth).
8. `debug_lines` and `status`.

Before any of these, a handful of small refactors that stand on their own: `World::void_y`, heightmaps relative to
`min_y`, int section Y in saves, `ChunkTicks` transfer, and worldgen accessors.

Questions for maintainers:
- Is the "provider object through a `replace` out-parameter" shape acceptable for 2 and 7? The idea is to keep hot
  paths out of the hook layer.
- Would you prefer 6 as an owned relayout function, or a move to absolute section Y keys?
- Any parts you'd rather not have as hooks at all?

🤖 Generated with [Claude Code](https://claude.com/claude-code)

## 9. Decisions and open questions

Lyle's answers (2026-11-07):
- **Filing upstream:** we file the proposals with Horizon. Each issue or PR text is shown to Lyle before it's posted.
- **Interim release:** none planned. The fork's GitHub builds are the stopgap, and the goal is the hooks. A
  pinned-Nightly mod release is a fallback, only if the hooks stall.
- **Dithered fade-in:** keep it. It is proposed upstream as its own feature, separate from the Cubic hooks.
- **Where the mod lives:** this repo, ThatCakeIsAPie/horizon-cubic. No new repo.
- **Hybrid:** kept if it works. It's tested before it goes into v1 (roadmap phase 4). It leans on the vanilla generator
  through A2's `open.vanilla`.

Still open:
1. **Shards.** OK to drop vertical shards for the first mod release (LAN guests share one window again)?
2. **TTM's features** (Ultimine, pack biome sounds, the stronghold cache, worldgen speedups…): upstream PRs, mods of
   their own, or fork-only?
