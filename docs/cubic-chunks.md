# Cubic Chunks

A cubic level has no height limit worth the name: blocks from y −8,388,608
to 8,388,607, loaded in 16 × 16 × 16 **cubes** around the players instead of
whole columns. This is a port of the idea of the
[Cubic Chunks](https://github.com/OpenCubicChunks/CubicChunks3) mod to Horizon. It is
not a port of its code: Horizon is Rust, and its world model differs from
NeoForge's.

Status: **v1, playable in single player** (see "v1 architecture" below). How to play:
- **Create World**, World tab: turn on **Cubic Chunks**. Or from the command line:
  `horizon-client --world NAME --cubic`.
- The Overworld is vanilla between y −64 and 320, with no bedrock floor.
- Custom Cubic worlds (the default) are CubicWorldGen's generator in every cube; Hybrid keeps vanilla's range and uses it outside; Flat is endless layers (`cubic::custom`, `cubic_gen.rs`).
- The far terrain is the cube octree.
- Vanilla levels and the Nether and End are unchanged.

Where the code is:
- `horizon-world/src/cubic/`: the cube range, store, window, tops and generator.
- `horizon-client/src/cubic.rs` and `cubic_lod.rs`: the level's cubes and far-terrain sampler.
- `horizon-client/src/game/cubic.rs`: the window that follows the players.
- `horizon-client-render/src/lod/cube.rs`: the octree far terrain.
- `horizon-protocol/src/cubic.rs`: the payloads.

No other Cubic Chunks port for Horizon exists. As of 2026-11-05, nothing came up in
horizon-rust/horizon's code, branches, issues and PRs, or in its forks'
branches. Java ports for 26.x do exist (Xyberg-001/CubicWorlds for Fabric 26.3,
DerkMorris/CubicChunks3-VibecodedEdition, 00jeser/CubicChunksAI for NeoForge
26.1.2, the owner's CubicChunks3UpdateTest for 26.1). They are mods for the
Java game, and Horizon cannot load them.

## Decisions

| | Choice | Why |
|---|---|---|
| Cube size | One section, 16³ (CC3 defaults to 32³) | `Section` is already the unit of storage, meshing, light and the network; a cube is a `SectionPos` |
| Height | Cubes −2¹⁹ ..= 2¹⁹−1 (blocks ±8.4 M) | `SectionPos::pack` keeps 20 bits of Y, so every packed position stays exact; well past CC3's "at least a million" |
| Opt-in | Per level, chosen at Create World; vanilla levels never change | Parity harnesses keep testing the vanilla game |
| Saves | `region3d/y<cy>/r.<x>.<z>.mca`, one vanilla region file per 32 × 32 cubes of one cube Y (CC's layout) | Reuses `RegionFile`'s tested sectors, oversize files and write order |
| Network | Horizon-only `horizon:cube*` payloads; vanilla clients cannot join a cubic level | `Pos` packs Y in 12 bits and the section count comes from the dimension type (see below) |
| Sky light | Unknown cubes above count as open sky (CC's rule) | Nothing can scan to the top of an unbounded column; not vanilla-exact, and `test_light_parity` cannot check it |
| World generation | Vanilla's noise router evaluated per cube; outside the router's range it keeps its own answer (solid below, air above for the Overworld) | The density functions are defined everywhere; the surface rules, carvers and decoration need neighbours (phase 8) |
| To The Max | Not a target: its 4096-high data pack is a bounded-level feature | The owner's call (2026-11-05) |

## Where the bounds live today

Notes from a survey of `main` on 2026-11-05. Root abbreviations: **R** is `crates/horizon-client-render/src`, **W** is
`crates/horizon-world/src`, **L** is `crates/horizon-client/src`, **P** is `crates/horizon-protocol/src`.

- **Storage.** `LevelShape` and `ColumnLayout` (R `chunk.rs`) cap a level at −2032..2031.
  `Chunk` is a dense `Vec<Arc<Section>>` from the column's `min_y`, and its `get` returns air
  outside the column, so "unloaded" and "air" look the same. `SectionMask` (R `section_mask.rs`) holds 256
  sections. `ChunkMap` is keyed by `(x, z)`.
- **Streaming.** All of `Streaming`'s state is per column: `ChunkGenerator::proto`, `decorate` on a 3 × 3 region,
  `finish`, `Light(x, z)`. Only meshing is per section. Tickets (L `ticket_levels.rs`) are 2D.
- **World generation.** Everything works on columns. One entry point fits: `fill_noise_inner`
  (W `worldgen/terrain.rs`) already takes a volume covering part of the chunk's Y. The blockers are:
  - the surface rules walk down from the heightmap and look below (W `worldgen/material_stage.rs`);
  - `CarvingMask` holds at most 4064 rows;
  - decoration's writable box spans the whole column;
  - anchors are counted from the bottom and the top.
- **Saves.** A vanilla column record has a byte `Y` per section and `yPos` (L `save.rs`).
- **Light.** `light_chunk` (R `light.rs`) floods a 3 × 3 region of whole columns, seeding the sky from the top
  layer. `lowest_sky_source` walks down from `layout.max_y()`.
- **Heightmaps.** Client heightmaps are `[i16; 256]` (R `chunk.rs`), which overflow past ±32,767.
  W `heightmap.rs` refuses anything taller than 4064.
- **Network.** `LevelChunkWithLight` is a column whose section count comes from the dimension type.
  `Pos` packs Y in 12 bits (P `buf.rs`), and 20 play packet fields use it. Single player goes through the same
  wire (`connection::Pipe`). `horizon:lod` (L `lod_net.rs`) is the precedent for a Horizon-only payload.
- **Rendering.** `SectionKey` is `(cx, section_index, cz)` (R `gfx.rs`). The occlusion search allocates a box
  `side² × height` (R `gfx/occlusion.rs`). The mesher's region is 3 × 3 columns.
- **Gameplay.** There are about 400 bound sites:
  - `World::bottom_y`/`top_y`: 108 calls in 44 files;
  - vanilla `MIN_Y`/`HEIGHT`/`SECTIONS`: 237 references;
  - `outside_build_height` in placement.
- **Far terrain.** Tiles of 16 × 16 surface columns with `i16` spans: a surface model.

## v1 architecture: a moving window of cubes

The first playable version keeps Horizon's column `Chunk` and its single layout per level. A cubic
level's `LevelShape` (`cubic: true`) is a **window**: `2V + 3` cubes (19 by default, 304 blocks) around the players
(`LevelShape::cubic_window`, `ColumnLayout::window`). That layout isn't held to a dimension type's
±2032.

- **Every index stays relative to one layout.** That covers light regions, `SectionKey`, `DirtySections`,
  `Streaming`'s keys and the mesher. Inside the window, the game is a level as high as the window, and code that reads
  `bottom_y`/`top_y` sees the window.
- **Cubes are the unit on disk and in generation.**
  - A column of the window is assembled from cubes: saved ones (`CubeStore`, `region3d/`), the vanilla
    generator's for a Hybrid world's −64..320, and `cubic::custom` (CubicWorldGen's generator, populated 2 x 2 x 2 cubes at a time) elsewhere.
  - The vanilla range of a column is generated once, whole, with vanilla's pipeline: surface, carvers,
    decoration, structures. Its bedrock floor is replaced by deepslate, and it is saved as cubes.
  - Modified cubes are saved. Unmodified generated cubes outside the vanilla range are not, because the
    generator makes them again.
- **Sliding.** The window is `2V + 3` cubes tall, centred on the player's cube (V is the Vertical
  Distance setting, sent as `horizon:cubic_view`; with several players, the union of their ranges, at
  most 99 cubes). It slides by whole layers, in place, when a player's cube is two or more from its
  centre cube or V changes (`game/cubic.rs`).
  - The cubes that enter come from the prefetch ring (`cubic_prefetch.rs`): the two layers beyond the
    window for every loaded column, ready in memory.
  - The leaving cubes are saved with their entities, block entities and ticks. The sections both
    windows hold keep their `Arc`s (`World::relayout`).
  - The hub sends `horizon:cubic_window`, then only the entering cubes (`horizon:cubes`), nearest
    column first, paced like chunk batches. The client lays out its level in place.
- **Sky light.** A column is open to the sky at the window's top only if no known cube above has a block
  (`ColumnTops::covered_above`, which also counts the vanilla surface). Otherwise its top is dark.
- **Network.** Single player runs on its server thread over the byte `Pipe`, so the wire matters there
  too.
  - A Horizon client joining a cubic level gets `horizon:cubic_window` before its chunks, and both ends
    switch block positions to the wide form (`Pos`'s 12 bits of Y can't hold ±8.4 M).
  - Vanilla clients are refused.
- **Entities, block entities and scheduled ticks** are saved per cube (a cube record holds the entities whose position is in it, `horizon:entities`), so those outside the window wait in their cubes. Uniform cubes (one block state, nothing else) live in a per-region sidecar `y<cy>/u.<x>.<z>.dat`, not in a region-file sector. POI is not saved: it is rebuilt from blocks.
  Void damage applies only below the cubic range.
- **Far terrain** is the cube octree LOD (`lod/cube.rs`): nodes of `(16 << level)³` blocks, 16³ voxels
  each, coarser further out. They are sampled from the same generators (and from saved cubes near the
  player), meshed with the far mesher, and drawn through the far-tile pipeline. The 2D far terrain is off
  in cubic levels.

## Phases

Each phase lands as small PRs (AGENTS.md: under ~500 lines), with tests. Vanilla levels stay byte-identical
throughout.

1. **Primitives** (done, `horizon-world/src/cubic/`):
   - the cube range (`MIN_CUBE_Y`..`MAX_CUBE_Y`);
   - `CubeStore`, the 3D region files;
   - `CubeWindow`, the cubes a viewer keeps (vanilla's `ChunkTrackingView` horizontally, times a vertical
     range, nearest first);
   - `ColumnTops`, each column's highest known block across cubes, CC's `SurfaceTracker`.
2. **Cube codec and generator.**
   - A cube's NBT: a vanilla section compound with `x`/`y`/`z`, its block entities and ticks.
   - `CubeGenerator::generate(SectionPos) -> Section`, the noise router over a `[16, 16, 16]` volume through
     `fill_noise_inner`.
   - Oracle test: the cubes stacked over −64..320 equal the column generator's noise stage.
3. **The level is cubic.**
   - `LevelShape` gets a cubic mode that skips the 2032 checks.
   - Create World's option, saved in `level.dat`'s Horizon data.
   - `Chunk` holds a sparse set of sections keyed by absolute section Y, with `get -> Option`, so an unloaded
     cube is not air.
4. **Streaming in 3D.**
   - `CubeWindow` drives generation, loading and saving in single player, without the network.
   - Meshing keys on absolute section Y.
   - Occlusion searches a Y window around the camera.
   - Test: start at y = 1,000,000, move 20 cubes up, and the cubes and meshes below are freed.
5. **Light.** Flood fill over cubes with CC's sky rule (`ColumnTops::covered_above`). Block light is unchanged.
6. **Network.**
   - `horizon:cube` payloads: a cube with light, cube block changes, forget, cache centre.
   - A Horizon host refuses vanilla clients in a cubic level.
   - LAN and the server thread.
7. **Gameplay bounds.**
   - `bottom_y`/`top_y` become questions about loaded cubes.
   - Spawning, weather and heightmaps go through `ColumnTops`.
   - Heights become `i32`.
8. **Generation past the noise stage.**
   - Surface rules per cube, given the cube above.
   - Carvers and decoration on 2 × 2 × 2 cube regions (CC's way around status deadlocks).
   - Structures at their vanilla heights.
9. **Far terrain.** Surface tiles from `ColumnTops`, then a 3D far terrain if the surface model is not enough.

## Risks

- **Sky light** has no vanilla oracle. CC3 ended up with a heuristic (unloaded means open, a 128-block search),
  and so will this port.
- **Generation that needs neighbours** (surface depth, decoration's 3 × 3, carvers) deadlocked CC's
  status graph. Generate cube regions in a fixed order, not per-cube statuses.
- **The ~400 column assumptions**, and code that reads unloaded as air: a cube that never loaded must not
  read as void to mobs, fluids or collisions.
- **Protocol.** Every cubic path, single player included, needs the custom payloads before it can be
  played.
- **Saves.** The `region3d` format is Horizon's own: vanilla cannot read a cubic level, and nothing can
  convert one back.
