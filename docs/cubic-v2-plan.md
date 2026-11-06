# Cubic Chunks v2: the plan

Written 2026-11-05, after v1 became playable. In the owner's words, "in this fork, nothing is sacred". Speed comes first, and anything in Horizon may change to get it.

v2 adds:
- CubicWorldGen-style terrain the player can customise;
- loading per cube instead of a moving window;
- far terrain generated on the GPU;
- compact storage;
- multiplayer.

Every feature below must earn its place against the budgets in "Budgets". The plan is laid out so the costly work happens once and is shared:
- The terrain shape is a single formula, which runs on the CPU for near cubes and on the GPU for far nodes.
- A uniform cube stays a single value end to end: in the generator, on disk, on the wire, in memory, in the mesher and in light.
- Moving vertically loads one new layer of cubes and drops another, the same way moving sideways loads columns.

## Principles

1. **Skip, then cache, then compute.**
   - Most of an unbounded world is uniform air or uniform stone, and the generator can prove it from the noise bounds (see "Density bounds").
   - Work on such a cube, and on everything downstream of it, should be skipped, not just made faster.
2. **Uniform is a value, not an array.**
   - `generate` returns `Option<u32>`, and storage keeps a sidecar of uniform cubes.
   - The wire sends one id for a uniform cube.
   - In memory it is a `Section` with a single-value palette and `Light::Uniform`.
   - The mesher skips it, or gives it its shell only.
3. **One formula, two backends.**
   - The CubicWorldGen density (`horizon-world/src/cubic/custom/density.rs`) is the reference.
   - `gfx/cube_gen.wgsl` evaluates the same formula for far nodes.
   - A test holds them together.
   - Octaves finer than a node's voxel are skipped on both backends.
4. **Distance decides fidelity.**
   - Cubes near the player get the full pipeline: terrain, carvers, population, vegetation and light.
   - Far-terrain levels 0–1 come from saved cubes, or from the CPU density without carvers.
   - Levels 2 and up come from the GPU, and show only the terrain shape and the surface.
5. **Simulate less than you show.**
   - Random ticks, fluids and block ticks run only within the simulation distance in 3D, not over whole columns.
6. **Never stall a frame or a tick.**
   - Generation, population, IO, meshing and GPU readback all run off the game threads.
   - Uploads per frame have a budget.
   - The server applies finished cubes in a time-boxed step each tick.
7. **Measure every claim.**
   - `horizon-ctl status` reports the counters listed under "Budgets".
   - The perf script in the repo checks them against the budgets.

## Architecture

### Residency: loaded cube ranges per column

v1 has a single 512-block window shared by every column. When the player nears its edge, everything near the player reloads. v2 removes the window.

- **Each column holds its own contiguous range of cubes.**
  - `Chunk` already has its own `min_y` and a dense `Vec<Arc<Section>>`. The range grows or shrinks one cube at a time at either end, through the relayout primitive from the in-place shift work.
  - `LevelShape` keeps only the absolute limits (±2¹⁹ cubes) and the flag that the level is cubic.
  - Code that indexes sections through the level's shape instead of the chunk's own layout must change. About 540 `.sections` uses in 72 files are to audit.
- **What a column holds** is the union, over players who see it horizontally, of `[player cube − V, player cube + V]`. V is the vertical view distance, a new setting. There is one cube of hysteresis, so a player at a cube boundary doesn't thrash.
  - Players far apart vertically in the same area are rare. The union is capped at 256 cubes per column. Beyond the cap, the player who came first keeps the range, and this is documented.
  - A gap inside the range costs one `Arc<Section>` of uniform air, about 100 bytes.
- **Moving vertically** adds a layer of cubes at one end of each column in view and drops one at the other. This happens once per 16 blocks of travel, so there is no reload and no blink.
- **The wire** carries cubes, not windows:
  - `horizon:cubes {x, z, cubes: [(cy, section | uniform id, light)]}`;
  - `horizon:cube_range {x, z, min_cy, max_cy}`, which drops cubes outside the range.
  - Vanilla's column packet stays in use for vanilla levels only.
- **Sky light** follows Cubic Chunks' rule:
  - Each column's top comes from `ColumnTops`, which is persisted per column and covers unloaded cubes.
  - A cube above the top is uniform sky light 15.
  - Light spreads by BFS only inside loaded cubes.
  - Adding a layer relights that layer and the layer next to it. Today a whole column is relit.
- **Simulation:**
  - Random ticks, block ticks and fluids run only in cubes within the simulation distance of a player, horizontally and vertically.
  - Entities are kept per cube (see storage) and unload with their cube.

### Generation: CubicWorldGen's Custom Cubic, adapted to 26.3

The full spec is the research report, summarised in the module docs of `horizon-world/src/cubic/custom/`.

- **World types**, chosen at Create World:
  - **Custom Cubic**, the default. CubicWorldGen's generator in every cube.
  - **Hybrid**. Vanilla 26.3 between −64 and 320, with villages, noise caves and the rest, and Custom Cubic outside that band. This is CubicWorldGen's Hybrid, and it is what v1 was, minus the floating islands.
  - **Flat Cubic**.
- **Settings** are CubicWorldGen's v9 JSON, with the same field names:
  - noise (selector, low, high, depth);
  - height and variation;
  - sample sizes;
  - standard and periodic-Gaussian ores;
  - caves, ravines and lakes;
  - replacers;
  - biome and biome size;
  - cube areas.
  - 26.3 changes the defaults: 26.3's ores with deepslate variants, a deepslate gradient at y 0–8, and 26.3's biomes mapped to 1.12 height and variation.
  - The JSON is saved in the world folder as `cubicgen.json`, next to `horizon_cubic.json`.
- **The per-cube pipeline:**
  1. Density bound early-out.
  2. Density on the noise-sample lattice, trilinear per block.
  3. Replacers.
  4. Carvers: CubicWorldGen's 3D caves and ravines, with origin-cube walks cached.
  - The terrain is a pure function of the seed, the settings and the cube.
- **Population** follows CubicWorldGen's +8 offset:
  - A cube is populated once, after the cubes at +x, +y and +z exist as terrain.
  - It writes into the 2 × 2 × 2 cubes starting at itself: ores, lakes, springs, and 26.3's own vegetation features for the biome at the cube's surface.
  - A cube is sent and simulated only once the eight populations touching it have run. This is Cubic Chunks' rule.
  - Populations run in parallel on a checkerboard of four phases by (x mod 2, z mod 2), so no two write the same cube at once.
- **Structures:** none in Custom Cubic, as in CubicWorldGen, which implemented only strongholds. Hybrid keeps vanilla's structures in its band.

### Density bounds

For a column the density is
`density = (mix + d)·vol + height − sign(vol)·y`, with `|mix + d| ≤ B` and `B` from the noise factors and offsets. So:
- **Uniform solid:** a cube whose whole Y range satisfies `y < height − B·|vol|` for every column of the cube is solid before carvers.
- **Uniform air or water:** a cube whose whole range satisfies `y > height + B·|vol|` is air above sea level, or water up to it.
- `height` and `vol` come from the biome blend, which is cheap and 2D. Their extremes per 16 × 16, and per far-terrain node, give bounds for a whole node.
- The bounds are used three times:
  1. The CPU generator returns a uniform cube without touching the noise. Carvers still run in solid cubes, but only when a cave walk's bounding box meets the cube.
  2. The far-terrain planner skips whole octree subtrees that are uniform air, or uniform solid and covered. This makes the far terrain O(surface) instead of O(volume).
  3. The GPU never sees a uniform node.

### Far terrain

- The octree, with voxels merged 2×2×2 per level, stays as built in v1.
- The far2 branch adds a disk cache of nodes keyed by seed, settings hash and node; plan hysteresis; and shadows from overhangs.
- **Sources by level:**
  - Levels 0–1: saved cubes when visited. Otherwise the CPU density and replacers (`block_at`, without carvers).
  - Levels 2 and up: the GPU (`gfx/cube_gen`), batched. The biome columns for a node come from the CPU, which is cheap and 2D. The CPU reference version is the fallback when the device has no compute.
- **Precision:** the CPU computes each octave's lattice origin in f64 per node, and the shader adds local offsets in f32. This holds at ±8 M blocks.
- Hybrid's vanilla band stays on the CPU far grid. Porting 26.3's density-function router to WGSL is deferred.

### Storage

The store2 branch adds:
- A sidecar of uniform cubes. A uniform cube costs 8 bytes or less on disk.
- Entities, block ticks and POI per cube, replacing v1's entities parked per column.
- `CubicLevel::load_cube` and `save_cube`.
- A writer thread that coalesces writes and drains on drop.
- Per-cube flags: `populated`, plus a "partial" flag for a cube that holds only writes from a neighbour's population.

### Client

- The renderer keys meshes by absolute section position. Adding or dropping a layer meshes only the new layer and its neighbours. This is the shift-client branch.
- The far-terrain gap at the edge of loaded cubes is filled by the octree's loaded-box test.
- Uploads per frame have a budget, as Horizon's uploads already do.

## Budgets

Measured on the owner's machine (8 cores, 24 GB), at render distance 16, vertical distance 8 and far terrain 128 chunks, with Custom Cubic defaults:

| What | Budget | Counter |
|---|---|---|
| Cube terrain, non-uniform | ≤ 0.3 ms average, ≤ 1 ms p95 | `gen_ms` |
| Cube terrain, uniform by bound | ≤ 5 µs | `gen_uniform` |
| Population, surface cube | ≤ 1 ms | `populate_ms` |
| Vertical layer change, server | ≤ 20 ms of work, spread over ticks; never a tick over 50 ms | `layer_ms`, tick time |
| Vertical layer change, client | Meshes only the new layer; no frame over 20 ms | frame time p99 |
| Far terrain on the GPU | ≥ 500 nodes/s | `lod_gpu_nodes` |
| Far terrain after a teleport | Filled to 128 chunks in ≤ 3 s | perf script |
| Frame rate | 120 fps sustained while flying | `fps` |
| Memory | ≤ 2.5 GB for the process; ≤ 128 B for a uniform cube in memory | `memory` |
| Disk | ≤ 8 B per uniform cube | perf script |
| Wire | ≤ 8 B per uniform cube | `net_bytes` |

## Order of work

1. **In flight now, as parallel branches:**
   - `cubic/shift-client`: the relayout primitive and meshing only what is new.
   - `cubic/store2`: storage.
   - `cubic/far2`: far terrain cache, hysteresis and shadows.
   - `cubic/cwg`: the generator core.
   - `cubic/gpu-lod`: the GPU far nodes.
2. **Integration** (the lead, in this checkout):
   1. Merge all five branches.
   2. Per-column cube ranges and the vertical view distance setting. Delete the window-move path.
   3. CubicWorldGen integration: the cube streamer (generate, populate on the checkerboard, attach); the biome source adapter; vegetation through 26.3's features; the world types; the far-terrain sampler on the CPU and GPU; density bounds in the generator and the planner.
3. **UI and tools:**
   - The Customize screen, with CubicWorldGen's five pages (Basic, Ores, Lakes, Caves, Advanced terrain) and Presets for JSON import and export.
   - A Create World type selector.
   - An F3 line: cube, column range, populated, far-terrain nodes and GPU queue.
   - The perf script in the repo.
   - Vanilla to cubic world conversion: vanilla columns are written as cubes and the world gets the marker.
4. **Multiplayer:**
   - Per-player ranges, as the union with the cap described above.
   - Tracking in the hub per player and per cube.
   - A trial on a dedicated server.
5. **Finish:**
   - One combined `cargo test`. Not before: the owner's rule.
   - The perf script against the budgets.
   - A playtest through `horizon-ctl`.
   - Install to `E:\Horizon Cubic`.
   - Update `HANDOFF.md` and these docs.
   - Push to horizon-cubic.

## Cut or deferred, on purpose

- CubicWorldGen's structure toggles: kept as fields, nothing generated, as in CubicWorldGen itself.
- `mesaSurface`: simplified clay bands.
- The replacer and cube-area editors in the Customize screen: CubicWorldGen never had them either. The JSON covers them.
- Hybrid's vanilla band on the GPU, which needs the 26.3 router in WGSL.
- Cubic to vanilla conversion. Only vanilla to cubic is planned.
- Floating islands and v1's deep generator: removed. Custom Cubic replaces them.
