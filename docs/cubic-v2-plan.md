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

### Residency: a window that slides one cube at a time, in vertical shards

v1 has a single 512-block window shared by every column. When the player nears its edge, everything near the player reloads.

v2 keeps a window per World, but makes it as small and as cheap to move as per-cube loading:
- **The window's size.** It is `2V + 1` cubes tall plus one cube of hysteresis, where V is the vertical view distance, a new setting.
- **Moving vertically.** The window shifts by one cube layer at a time, in place (`World::relayout`, from the shift-client branch). This happens once per 16 blocks of travel. Every column drops one layer and gains one. Sections that stay keep their `Arc`, their mesh, their GPU slot and their light.
- **Why a window.** A window per World costs the same as per-cube loading for the cubes the players need. It keeps every system's section indexing as it is. Ranges per column would mean auditing about 540 `.sections` uses in 72 files.
- **Multiplayer uses vertical shards.** Players whose windows overlap share one World, whose window is the union of theirs.
  - A player whose window stops overlapping the others splits off into a shard of its own: another World with its own window and Streaming, in the same dimension, backed by the same `CubicLevel`.
  - Shards merge when their windows meet.
  - Shards never hold the same cube, so nothing is simulated twice. Entities leaving a window are saved with their cube, as at a window's edge today.
  - The server's map of levels goes from one World per dimension to a list of shards per dimension. A player belongs to one shard.
- **The wire** carries cubes, not whole columns, once a window has moved:
  - `horizon:cubic_window {min_y, height}` relays out the client's window in place.
  - `horizon:cubes {x, z, cubes: [(cy, section | uniform id, light)]}` sends only the entering layer.
  - Vanilla's column packet carries the first full columns only.
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

## North star: a Voxy-class far view, entirely on the GPU

The owner's end goal is the look of Voxy on Java: full blocks out to the horizon, at distances of thousands of chunks. The Rust rebuild should beat it. The owner also pointed to the unreleased Meridian LOD mod, shown filling a whole world in seconds: far terrain has to appear almost at once, not trickle in.

v2 lays the ground for this. v3 finishes it with the far-terrain path running on the GPU end to end:
1. **Generate on the GPU** (`gfx/cube_gen`, v2). A node's 16³ voxels are written into a slot of a GPU-resident node pool. Reading them back to the CPU is optional, and only v2's CPU mesher uses it.
   - **In huge batches.** Far terrain is only noise, so a ring of the plan, possibly thousands of nodes and on the order of 1024 × 1024 chunks at the coarse levels, goes out in one dispatch rather than one node at a time.
   - **A 2D pre-pass bounds the surface.** Per voxel column it computes the depth noise and the biome height and volatility once. From the noise amplitude it derives the Y range where the surface can be. Nodes outside that range are uniform without any 3D noise, so the cost scales with surface area, not volume.
2. **Biomes on the GPU.**
   - Port 26.3's climate sampler (six normal noises) to WGSL.
   - Flatten the biome parameter tree into a buffer.
   - The CPU then gives far nodes nothing at all.
3. **Mesh on the GPU.**
   - A compute pass turns pool slots into compact quads of 8 bytes each (position in node, size, face, block texture id, light and tint) in a GPU geometry arena.
   - Faces are merged greedily per block texture, so the far view keeps block textures, tints and block-scale UVs, as `mesh_cube` does today.
4. **Select and draw on the GPU.**
   - A compute traversal of the octree picks nodes by screen-space error.
   - It culls them against the frustum and the previous frame's Hi-Z, reusing `gpu_occlusion.wgsl` from near-terrain GPU culling.
   - It writes the indirect calls for one `multi_draw_indexed_indirect_count`. The vertex shader pulls quads from storage, with no vertex buffers.
5. **What the player has seen** overrides generated terrain:
   - Saved cubes are downsampled (levels 0–1, v2) and uploaded into the same pool.
   - An edited cube re-uploads its nodes only.
6. **Light.**
   - Sky light comes from a per-column top kept on the GPU.
   - Block light comes from emitters baked per voxel when downsampled.
   - Near and far should not show a seam in brightness.

Fallback: devices without compute or `MULTI_DRAW_INDIRECT_COUNT` keep v2's CPU path. This is the same split `gfx/gpu_culling.rs` makes today.

### Whole World, then autoresearch

The target is the Meridian LOD mod's whole-world toggle. In a cubic level, LOD Distance's top step is **Whole World**:
- The octree's root covers vanilla's world border (±30 M blocks) and the whole cube range (±8.4 M blocks).
- The plan subdivides by distance as usual.
- Subtrees the 2D pre-pass proves uniform are never requested.

The owner's order is **speed first, then detail**:
1. **Now: the medium and large changes** that decide how fast distant terrain can load:
   - GPU generation in huge batches;
   - the 2D surface pre-pass;
   - GPU meshing and drawing;
   - GPU biomes;
   - Whole World.
2. **Then: a benchmark that gives one number:**
   - It launches a cubic world offscreen at Whole World, and teleports far away.
   - It times **until the plan is complete**, meaning every planned node is drawn or known empty, from `CubeLodStats`.
   - It also reports frame time p99 meanwhile.
3. **Then: autoresearch loops** (the `autoresearch` skill) over the small knobs, with that number as the metric:
   - batch sizes;
   - the octave cut-off;
   - `DETAIL` and the merge factor;
   - the pool and arena sizes;
   - workgroup sizes;
   - upload budgets;
   - the pre-pass bound;
   - CPU worker counts.
4. **Detail comes after speed:** decorated far terrain (trees, snow) and Hybrid's vanilla band on the GPU.

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
   2. The window slides one layer at a time, in place, on the server too, with `horizon:cubes` for the entering layer. Add the vertical view distance setting. Delete the reload path for window moves.
   3. CubicWorldGen integration: the cube streamer (generate, populate on the checkerboard, attach); the biome source adapter; vegetation through 26.3's features; the world types; the far-terrain sampler on the CPU and GPU; density bounds in the generator and the planner.
3. **UI and tools:**
   - The Customize screen, with CubicWorldGen's five pages (Basic, Ores, Lakes, Caves, Advanced terrain) and Presets for JSON import and export.
   - A Create World type selector.
   - An F3 line: cube, column range, populated, far-terrain nodes and GPU queue.
   - The perf script in the repo.
   - Vanilla to cubic world conversion: vanilla columns are written as cubes and the world gets the marker.
4. **Multiplayer:**
   - Vertical shards, as described under "Residency".
   - Tracking in the hub per player and per shard.
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
