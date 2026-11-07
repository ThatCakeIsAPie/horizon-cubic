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
  - **As built** (`cubic_shards.rs`, `game/cubic.rs`, `game/levels.rs`, `lan/hub.rs`):
    - A shard is a `World`, `Streaming`, window cell (`CubicTerrain::window`) and `SlideState` of its own, parked in `Levels.parked` under `(Dim, shard)`. The level in place is `(Game.dim, Game.shard)`; `enter_level` takes a `LevelId` and a bare `Dim` means the shard in place, else the dimension's primary. `Guest.shard` and `Hub.shard` name a player's and the hub's.
    - `cubic_shards::plan` groups players by their ranges: touching ranges (gap 0 or less) merge, a gap of two cubes or more splits, a gap of one keeps things as they are (the hysteresis). A chain taller than 99 cubes is cut between players. `settle` keeps only the moves that cannot put a cube in two shards.
    - A merge saves everything the closing shard holds, then the survivor slides over its cubes (a jump) and the players move. A split shrinks the old shard first (a shrink is at once), then opens the new shard. A player that changes shard is sent the level like a dimension change (`Respawn`, window, columns, cubes); a vehicle it rides stays in the old shard.
    - The `CubicLevel` is shared. Column data stays one answer: sky covering comes from its surface cache (`sky_cover`), the structure data and surface heights from the marker cube's record (a window without the marker reads it), generation of a column's vanilla range is written once (`vanilla_write`), and the prefetch ring has a target per shard. The saved window (`horizon_cubic.json`) is the primary shard's (shard 0, which also owns forced chunks, attachments and raids); shard 0 never closes.
- **The wire** carries cubes, not whole columns, once a window has moved:
  - `horizon:cubic_window {min_y, height}` relays out the client's window in place.
  - `horizon:cubes {x, z, cubes: [(cy, section | uniform id, light)]}` sends only the entering layer.
  - Vanilla's column packet carries the first full columns only.
- **Sky light** follows Cubic Chunks' rule:
  - Each column's top comes from `ColumnTops`, which is persisted per column and covers unloaded cubes.
  - A cube above the top is uniform sky light 15.
  - Light spreads by BFS only inside loaded cubes.
  - Adding a layer relights that layer and the layer next to it (`light::relight_entering`); no column is relit whole.
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
- **Structures:** Custom Cubic has 26.3's overworld structures (CubicWorldGen made only strongholds). Hybrid keeps vanilla's structures in its band and has none outside it. See "Structures in Custom Cubic".

### Structures in Custom Cubic (2026-11-07)

The owner asked for structures with sliders, like the terrain's, defaulting to as close to vanilla as possible, so that a custom world can switch some off or tune them.

- **What generates:** 26.3's 18 overworld structure sets: villages, pillager outposts, woodland mansions, desert pyramids, jungle temples, swamp huts, igloos, abandoned camps, ruined portals, trail ruins, ocean monuments, ocean ruins, shipwrecks, buried treasure, mineshafts, trial chambers, ancient cities and strongholds. A mod's sets generate too, untuned.
- **Settings:** `structureSets`, a Horizon extension to the CubicWorldGen JSON, maps a set's name to its switch and placement numbers: spacing, separation and chance for a random spread; distance, spread and count for strongholds' rings. A missing set or field means vanilla's numbers. CubicWorldGen's own toggles (`villages`, `temples`, ...) still switch their groups off. The Customize screen's Structures page edits these settings (Custom Cubic only).
- **Starts:** vanilla's placement and structure code (`StructurePlanner`, over the same `StructureState` the vanilla generator uses), asked through a start context over the Custom Cubic generator:
  - The heightmaps are the terrain's surface: the topmost solid block from density and replacers (`column_top`), ignoring carvers and population. The sea fills up to sea level.
  - A column has one surface, however high. A village stands on a peak 9,000 blocks up as it does on a plain.
  - The structures see vanilla's bottom of the level (-64) and no top. Mineshafts, strongholds, trial chambers and ancient cities sit at vanilla's depths, and no start is refused for being too high.
- **Placement** rides on population, between the lakes and the ores (vanilla's structure steps). The population of cube `c` places, into its area (`c * 16 + 8` to `+ 24` on each axis), every piece of the starts within 8 chunks whose columns cross the area. Each block of a structure belongs to exactly one area, so it is placed once, whichever cube populates first.
  - The pieces are chosen by their columns, not by their boxes, because a piece that picks its height when placed (a temple on the ground) only crosses the right cube once placed.
  - Such pieces are *settled* once, as a first placement in the start's own chunk with the start's own random, so every population and thread finds the same height (`LegacyPiece::settle`).
  - Reads see the population's 2 × 2 × 2 cubes and stone beyond them. Writes stay in those cubes.
  - Chests get their loot tables and spawners their mobs. Block entities and entities (villagers, guardians, the witch) go into the cube records.
  - A test makes a village's cubes in two opposite orders on two levels and checks that its building blocks and block entities are the same. The decoration around may differ by a few blocks, as it does without structures: a population's ores, trees and a village's snow piles reach into neighbouring areas, so the first of two neighbours decides those.
- **Cost:** the cube bench (`HORIZON_CUBE_BENCH_DATA=1`, 2,197 cubes around the origin, one thread) goes from 0.52 to 0.56 ms per cube, about 9% more, with `HORIZON_CUBE_BENCH_NO_STRUCTURES=1` as the comparison. At first a cube cost twice as much, and a village's cubes four times: a mineshaft corridor asks the heightmap of almost every block it places. The heightmaps are now found a chunk at a time (`chunk_tops`), reading the cubes from the bound's ceiling down by their density lattice. That costs 1.7 µs a column, against 260 µs for a single column's search.
- Structures write blocks the way generation does (`LevelFeatureWorld::raw`, vanilla's `WorldGenRegion.setBlock`), without the live level's neighbour shape updates; templates update their edges' shapes themselves.
- **Lookups** (`structures/locate`): `/locate structure`, explorer and treasure maps, and the structures a position is in (structure mob spawns: witch huts, monuments, outposts; advancements; cats) answer from the planner, not from data stored per chunk.
  - The level's `VanillaTerrain` (the locator every caller uses) hands its structure queries to the level's `CubicStructures` (`set_cubic_structures`): `starts_at`, `locate` and `locate_for_map`. Maps keep the starts they pointed at in the reference file, as superflat worlds do.
  - A Custom column's chunk gets its references when it is assembled (`CubicLevel::structure_refs`: vanilla's `createReferences`, the starts within 8 chunks whose box crosses it).
  - Terrain populated before the world had structures: a new record flag, `STRUCTURES` (bit 2 of `horizon:cubic_flags`), marks a cube whose population placed them. A start whose first piece's middle cube was populated without the flag was never built (an older world, or one made with structures off and switched on later), so `/locate` and maps pass over it and no chunk references it.
  - Hybrid is unchanged: its vanilla band keeps vanilla's starts, references and locator.
- **Not done yet:**
  - Terrain adaptation (vanilla's beardifier): the terrain is not flattened under villages and outposts, or dug out around buried structures.
  - Structures in the far view: "Far view: the owner's decisions and later work", item 4.
  - Existing Custom Cubic worlds get structures in the cubes they generate from now on, as a Minecraft update does: a structure that starts in new terrain is cut off where it reaches terrain made before.

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

### Sparse descent: the 3D test that replaces the 2D bound

The owner suggested this, after Houdini's sparse solving for smoke and fire:
- Divide the world into coarse cells.
- Test each cell for "anything in here". Stop where the answer is no, and subdivide where it is yes.
- Far away, stop at the LOD level that distance needs. Close by, keep subdividing down to blocks.

**Why the 2D bound is not enough.** The 2D pre-pass is conservative: it never marks a node wrongly. But it assumes one surface band per column. In a world tuned for 3D noise (spaghetti, floating islands, huge overhangs, or a small height factor with large noise factors), that band covers almost every height, so the pass skips almost nothing.

**The 3D test** uses a Lipschitz bound on the density. Each octave `i` changes by at most `min(2·amp_i, amp_i · freq_i · G · r)` within distance `r` of a point:
- `G` is the gradient bound of one ImprovedNoise octave (4.0, certified; see `bounds.rs`).
- Summed over the four noises, through the selector's clamp, the depth transform and the `height − y` term, this gives `L(r)`.
- One sample at a node's centre decides the node:
  - `|density(centre)| > L(half-diagonal)` means the node is uniform: all solid or all air, with water up to sea level.
  - Otherwise subdivide.
- Octaves the node's resolution skips are bounded by their amplitude alone.

**The descent rule** (the owner's two checks). For each node, from the root down:
1. **Uniform?** One sample plus `L`. If the node is proved all air or all solid, stop. Nothing is drawn, except a solid node's shell where it meets air.
2. **The right size for its distance?** This is the plan's `DETAIL` × size test. It needs no noise, so it is free. If it passes, stop and hand the node to the visual engine: generate its 16³ voxels, mesh them, draw them.
3. **Otherwise** subdivide into its 8 children and repeat.

Each node costs one density sample, about a hundred arithmetic operations, so a GPU decides millions of nodes per frame. Only the surviving leaves pay for full generation.

**Done** (`gfx/cube_descent.rs`, `cube_descent.wgsl`, `lod/cube.rs`):
- **The bound in WGSL.** A node is decided by one centre sample of each of the four noises plus `bounds.rs`'s box bound, ported term for term: per-octave value widened by `G · Σ|f_a|2^i h_a` and clipped to `±V`; octaves the node's voxels cannot show (and those beyond the 16 uploaded) bounded by amplitude; the selector clamp, the hull of low and high, the depth transform at the ends and kinks; the density range over the 4 × 2 × 3 combinations of height, volatility, the `special` branch and the volatility's sign.
  - The biome terms are the **global range** of the biome table, which is always safe. The CPU bound is tighter only for boxes of up to 33 × 33 quarts (level 2); feeding `biome_gen`'s columns into the same slot is not wired.
  - What the bound is *for* is the far generator's own rules (`cube_gen`'s simplified replacers), not the CPU replacer chain. Density `≤ 0` everywhere is air above the sea and water below it, uniform unless the sea level crosses the node. Density `> 0` over the node **and the voxel layers the generator reads above it** is rock: no top block or filler can show. Anything else is `Mixed`, so a node the bound cannot settle is subdivided, never wrongly dropped.
- **Precision.** The children are made on the GPU, so there is no CPU turn to split each node's octave origin in `f64`. The shader computes `c · scale + offset` modulo 256 in exact fixed point instead: per-octave constants `kint + kfrac/2^16 = scale · 2^24` and `c = ch · 8192 + cl` keep every partial product in 32 bits. The error is below `|c| · 2^-40 + 2^-22` cells (2.7 · 10⁻⁵ at 30,000,000 blocks), and the shader adds it to every octave's reach, so the bound stays rigorous.
- **Passes.** One compute dispatch per level, from the plan's roots (`root_nodes`, at most 27) down to level 2, all recorded in one command buffer with no CPU round trip. Pass `j` reads list `j & 1` and appends the 8 children of every node that must split to list `(j + 1) & 1` with an atomic counter. Every node also lands in one output list as a right-sized leaf, a node the CPU plan owns below level 2, proved-uniform air or rock, or a uniform node around the loaded cubes that is split on anyway. The distance rule is `plan_classified`'s: camera passed as integer part and fraction, `dist < DETAIL · scale · size`, nodes outside the world or past the maximum distance dropped. Readback is the compact output only (counts first, then exactly that many nodes).
- **In `CubeLod`.** `GpuCubeSampler::with_descent(terms)` builds the descent on the sampler's device; `CubeLod::set_gpu_descent(true)` makes the planner take its classes from it. The CPU plan, hysteresis and loaded cubes are unchanged. One request per camera cube goes to a descent thread, which files the uniform nodes in the same class cache the classifier thread used (which is left idle), then the plan runs again. The descent runs with the hysteresis scale (`MERGE`), so it covers every node the plan can visit. Nodes it has not classified count as `Mixed`, so nothing is lost while it is behind. The first plan waits for the first answer. `CubeLodStats` gains `descent_runs`, `descent_uniform`, `descent_leaves`, `descent_ms` and `descent_overflow`; `uniform_skipped` counts nodes the descent pruned. The leaf set is the CPU plan's minus proven-uniform subtrees, which a test holds.
- **`G` and `V` are tighter.** The module docs of `bounds.rs` now derive the two octave constants as suprema over every gradient assignment, `max |P| = Σ w_c m(u − c)` and `max |∂P/∂u_x| = Σ m(v_c)` (`m` = sum of the two largest components), certified on a grid with a Lipschitz allowance by `bounds_tests::the_octave_constants_are_certified`. `G` goes from 6.7 to 4.0 (the true supremum is 3.75) and `V` from 1.5 to 1.1 (true 1.036): `lin` shrinks by 40 %, and every amplitude-bounded octave by 27 %.
- **Not done.**
  - Handing the leaves to `CubeGen` without leaving the GPU. The descent returns the compact leaf list, the batch `request_resident` takes; the same fixed-point phase could build `GpuHeader`s on the GPU, and the columns would come from `biome_gen`.
  - Per-node biome corners from `biome_gen` in place of the global range.
  - Hybrid's vanilla band (the density there is not this formula).

**Uses:**
- **GPU far terrain.** The descent runs level by level as compute passes over a compacted list of undecided nodes. The descent stops at the level the distance plan wants. Work then scales with surface area in any world, islands and spaghetti included.
- **CPU cubes.** One density sample, plus the 2D biome terms, decides whether a cube is uniform before the 75-sample lattice runs. Carvers still run where a cave walk reaches.
- **The planner** never requests a subtree the descent proved uniform.

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
2. **Biomes on the GPU** (`gfx/biome_gen`, written; not yet wired into the far-terrain producer).
   - Done: 26.3's climate at sea level (the offset-noise shifts and the five shifted `NormalNoise` fields, as Perlin layers with per-node, per-layer `f64` origin splits) runs in WGSL. The parameter tree is flattened (`ClimateIndex::tree`) and searched on the GPU with `Climate.RTree`'s branch and bound.
   - Done: CubicWorldGen's 5x5 quart blend for voxels of 8 blocks or less, and the column's own biome for coarser ones. One pass makes a whole batch's `NodeColumns` straight into `cube_gen`'s input buffers (`CubeGen::generate_to_pool_gpu`). The CPU path (`CpuBiomes`) stays as the fallback and the test reference.
   - Not done: the per-biome table comes from the terrain generator (`BiomeTerrain`); the 2D pre-pass (`classify`) still needs CPU columns, so GPU-column nodes all take the 3D pass; wiring into `GpuCubeSampler`/the LOD producer.
   - The CPU then gives far nodes nothing at all, once wired.
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

The target is the Meridian LOD mod's whole-world toggle. From memory of the owner's video, the estimate was about 20 s to show an LOD of the whole world, on a top gaming CPU. Watched frame by frame, the video shows a coarse whole-world picture in 1.5 s, refined over tens of seconds afterwards (see "What the Meridian video shows" below). Our first benchmark on this 4-core/8-thread i7 was 25.8 s per hop. The aim was to beat that clearly: under 10 s.

**Where it stands (2026-11-06): 2.1 s per hop** (`tools/cubic_far_bench.py --hops 3`: 2.4, 2.6 and 1.4 s for about 7,000 planned nodes; 12.6 s at spawn including startup; 93 fps min, 117 fps mean). Two changes got there:
1. Reads of cubes never saved are answered from an in-memory index, with per-region locks. Before, every read queued behind the store's global locks while the near terrain wrote cubes, at 30–600 ms per read.
2. Unsaved fine nodes (levels 0–1) are built from whole generated cubes instead of 4,096 point queries.
The GPU columns and the batched worker requests showed no gain beyond the noise between runs (single hops of one build varied by about ±10 s then); the descent was already on in the first benchmark and has never been measured on its own. The next lever is the spawn fill. In a cubic level, LOD Distance's top step is **Whole World**:
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

### What the Meridian video shows (2026-11-06)

The owner downloaded the video and we went through it frame by frame. What it shows:
- **On joining and after each teleport,** chat prints "Preparing distant terrain data…" and then "Distant terrain ready (1.5 s)". After that the whole world is on screen. From y = 1,000,000 and 2,000,000 it is a flat plane out to the border's edge.
- **That first picture is coarse.** The whole-world option's tooltip says coarse terrain appears first and is refined progressively. After a teleport to the ground, its F3 shows about 670 builds waiting, done at about 15 builds/s. Full detail therefore arrives over tens of seconds, and trees appear on distant hills seconds after the land.
- **From high up, the ground is a point-sampled colour field.** It shimmers like static, and the End's islands make moiré rings. Colour is not averaged at the coarse levels.
- **Detail follows screen size.** At 2:43–2:50 he is in the End at y ≈ 2,000,000 and looks through the spyglass. The far islands first show as fine static. Within about a second, the bands under the spyglass sharpen into block shapes, one stretch at a time. Zooming in narrows the field of view, which makes each voxel bigger on screen, so the nodes in view get refined.
- **Its F3** shows:
  - a native Rust part, about 180 MB off-heap;
  - 235–430 MB of GPU terrain buffers;
  - "9 LOD / 2 capture workers";
  - a terrain memory target of about 2.6 GB;
  - a build rate limited by the CPU.
- **Its settings:**
  - Render the entire world, LOD distance and Detail (Normal/High/Extreme);
  - Trees, Grass and flowers;
  - Record seen terrain;
  - Extended clouds, LOD anti-aliasing and Supersampled LOD;
  - Horizon fade (start 55 %), Atmospheric haze and Remove vanilla terrain fog.

The two numbers measure different things. Meridian's 1.5 s is a coarse picture of the whole world; its full detail takes tens of seconds. Our 2.1 s is every planned node at its final detail. But nothing shows until the nearest nodes land, and nothing coarse covers the rest in the meantime.

What we take from it:
1. **Coarse first.** On a teleport, request a coarse whole-world layer first (a few hundred nodes at high levels), draw it, then refine nearest-first. `CubeLod::removed` already keeps a node drawn until everything over its place is delivered. What is missing is asking for the coarse ancestors before the fine leaves. The benchmark should then report two times: the first full-coverage picture, and complete.
2. **A height-map layer for the absolute coarsest levels** (branch `cubic/far-surface`). A 16³ node at level 15 has voxels 32,768 blocks tall, while all the terrain lies within a few hundred blocks. So the surface snaps to whole voxels, and the top face can be the stone below the surface.
   - The owner's order: height-map tiles at the absolute coarsest levels, refining into 3D cube nodes, then finer nodes, then full blocks.
   - **Exact heights, no fixed range.** The owner wants realistic mountains (a Minecraft Everest of about 9,000 blocks) visible from extremely far away. A tile column keeps its height in blocks.
   - **Layered columns, for players' edge cases.** CubicWorldGen presets can make sky islands at any height, overhangs and arches; later generators may make ceilings. Each column holds up to K solid spans (top, bottom, top block; K = 4 to start), drawn as boxes. One height per column would draw a sky island as a pillar from the ground.
   - **Complex places go 3D sooner.** A column with more spans than K merges its lowest ones and flags the tile as complex. The node-kind decision is per node (`node_kind(key, view, tile_info)`), so complex tiles become 3D cube nodes at a greater distance than simple ones.
3. **No colour blending (owner's decision).** Meridian's shimmer from very high up is whole biomes side by side (desert, ocean, plains), which is true to the world. No one plays at y 5,000,000. So each tile column takes one colour sample: its top block.
4. **Fog follows the far view.**
   - Fixed on 2026-11-06: cubic worlds never told the fog that far terrain was on (`FogInputs::distant_terrain` checked only the column LOD). Atmospheric fog therefore stopped at about 1,000 blocks, and from high up the ground faded into the sky.
   - Later: a horizon fade setting like Meridian's.
5. **Choose detail by size on screen, not by distance alone.**
   - The plan splits a node while it is nearer than `DETAIL` (2.5) times its size, whatever the field of view. So a far voxel spans about 1/40 of its distance, about 25 pixels on a 1280-pixel, 70° view.
   - That is why the far view looks blocky, and why "complete" does not mean full detail.
   - **The owner's ideal:** a block stays a block until it is smaller than a pixel on screen. Then it steps down to 2×2, then 4×4 when that is about a pixel, and so on, coarse first and refining up to that.
     - At 1080p and 70°, a block is one pixel at about 770 blocks away, and a 16-block chunk at about 12,000.
     - There is no point drawing single blocks so far away that whole chunks are a pixel.
     - The Far Detail setting's top step is that switch point (about 1 px per voxel), and the default is as close to it as the hardware allows. A rough estimate is about 150,000 far nodes for a whole-world view at 1 px, against about 7,000 at today's `DETAIL`.
   - To get there:
     - choose levels by projected voxel size, using the field of view and the resolution;
     - refine what is on screen first;
     - make the target a setting (like Meridian's Detail), and measure the fill time at each step.
6. **The coarse levels alias.** From y = 5,000,000 the ground is a regular tile pattern, which is wrong data rather than biomes. Each coarse voxel point-samples the noise at a spacing that lines up with the noise's period, so every node gets the same values. The height-map tiles sample each column at a hashed (jittered) offset inside the column, which breaks the alignment for free.

### Far view: the owner's decisions and later work (2026-11-06)

These came from the owner while reviewing the Meridian video. They are recorded here so they outlive any one session. Items marked *later* are not being built yet.

1. **Detail by size on screen** is the rule everywhere, underground too (see 5 above).
2. **Occlusion: generate behind, refine only when seen.** *Later.*
   - Terrain hidden behind other terrain (behind a mountain, or beyond the rock walls of a cave) still gets a low-detail version, so turning or moving never shows a hole.
   - It is not refined until the player can see it. Hi-Z or occlusion-query results from earlier frames feed the planner's priority: a hidden node keeps its coarse level, and a node that becomes visible goes to the front of the refine queue.
   - This matters most underground, where most far terrain is behind rock.
3. **Far views of the player's own builds, updated in the background.** *Later; the next job after the far-view rework.*
   - **What exists** (`cubic/far2`): a disk cache of far nodes, `lod::cube_cache` in `<dimension>/horizon_cube_lod/`.
     - It holds levels 1–8 (nodes up to 4,096 blocks wide).
     - A level-1 node made from saved cubes (the player's edits included) is kept. A coarser node is built from its eight kept children and kept too. Kept nodes come before the GPU generator.
     - Saving a cube erases the kept nodes over it (`CubeLodCache::invalidate`). They are rebuilt the next time the plan asks.
   - **Where it falls short:**
     - A parent is built from its children only when all eight are kept. Otherwise the generator makes it fresh and the build vanishes at that level. So a build shows in the far view only where the player has been close enough for the surrounding area to be kept.
     - Above level 8 nothing is kept, so builds never show beyond nodes 4,096 blocks wide.
     - Rebuilding is lazy, on the plan's request, not in the background as the player builds.
   - **The fix** (the owner's idea):
     - When a cube is edited, mark its far nodes dirty all the way up the levels.
     - Rebuild them in the background at low priority, coalescing many edits.
     - A parent with only some children kept takes the rest from the generator, CPU or GPU, and merges them, so it never drops the build.
     - Keep every level (or the edited columns' spans) on disk, so builds survive a reopen.
     - Upload only the changed nodes, so the player sees their build's LOD update while they build.
   - The height-map tiles take edits the same way: rebuild the affected columns' spans, and keep edited tiles in the disk cache.
4. **Structures fill in over time, in the background.** *Later; Custom Cubic's structures generate since 2026-11-07, near terrain only.*
   - Structureless terrain is generated first, as now.
   - A low-priority CPU job runs structure placement (structure starts and piece layout, not full chunk generation) over the regions the far view covers, nearest and on-screen first.
   - When it finds a structure, it regenerates the far nodes and tiles that the structure's bounding boxes touch, at their current levels, with the structure's blocks voxelized into them.
   - Villages, temples and strongholds then appear in the far view over time, Meridian-style coarse-then-refine but for content. Hybrid's vanilla structures can use the same path.
5. **The thin-feature flex: a 1-block-wide build visible from 50,000 blocks.** *Later, after 3; nice to have, not a priority.*
   - When downsampling, player-built blocks win over natural ones in a mixed voxel.
   - Features thinner than a pixel are drawn at a minimum width of one pixel, the way games draw distant power lines.
6. **Huge underground caverns** (Skyrim's Blackreach, scaled up) are something players will build or generate.
   - The screen-space rule and coarse-first must work with the camera deep in rock: the coverage layer is 3D nodes around the camera, not surface tiles.
   - Occlusion (2) and **far block light** (glowstone, lava and shroomlight lighting distant cavern walls) are what caverns need most. Today far terrain is lit by sky light only, so a lit cavern looks black past the near area.
7. **Other edge cases players will push:**
   - sky islands at any height (layered tile columns);
   - overhangs and arches;
   - ceilings and stacked layers;
   - extreme height settings: no height cap in tiles; the cube range is ±8.4 M blocks, the border ±30 M;
   - very tall thin towers (5 above).
8. **A realistic-scale terrain preset.** *Later; terrain-generator work, not far-view work.* The tall-mountain test (`heightFactor` 64 → 30,000, `heightVariationFactor` 64 → 1,000) only stretches the terrain upwards. Every hill becomes a needle thousands of blocks tall and a couple of hundred wide. A Minecraft Everest of about 9,000 blocks also needs the terrain stretched sideways: lower horizontal noise frequencies and a bigger biome scale, so mountains are miles wide as well as tall. Erosion-like shaping may follow. The far view already draws any height.

### Detail by size on screen, coverage first (2026-11-06, `cubic/far-detail`)

Points 1 and 5 above are built (`lod/cube.rs`):
- **The rule.** A node splits while one of its voxels would span more than `target_px` pixels at the node's nearest point: `voxel / distance * height_px / (2 tan(fov_y / 2)) > target_px` (`FarView`, `Rule`). The hysteresis is kept (`MERGE`). The game passes the field of view drawn now (spyglass and dynamic FOV included), the view direction and the viewport (`CubeLod::update`).
  - Nodes in view use the current field of view, nodes out of view the unzoomed one, so the spyglass refines only what it looks at. A zoom of less than 1.5 times (a bow, a sprint) changes nothing.
  - The plan is redone when the detail changes by a quarter (spyglass in and out), and when a zoomed view turns past a quarter of its field of view.
- **Cubic Far Detail** (Video Settings › Distance, key `cubicFarDetail`): Low 16 px, Normal 8, High 4, Ultra 2, Extreme 1. Extreme is the owner's switch point: a block stays a block until it is about a pixel.
- **A budget** (`NODE_BUDGET`, `HORIZON_FAR_NODES` for benchmarks): past it, the plan splits the nodes furthest over their target first and stops, so what is smallest on screen stays coarse; on the GPU path it is also capped so two plans fit the node table and the quad arena.
- **Coverage first.** When a quarter of a new plan has nothing drawn over it (a teleport, joining), only the coverage layer is queued: the plan for voxels 8 times bigger, at most 768 nodes, one node over each hole, and the planned nodes that hold the last plan's (so those can go). Once it is in and shown, the rest is queued; `removed` keeps the coverage nodes until the nodes under them are in. `status`'s `cube_lod.plan_covered_ago_s` stamps when the plan last had no hole.
- **On screen first.** The queue is holes first, then the plan by direction from the camera (48 sectors and the nodes around the camera), the sectors the camera faces first, nearest first in each. Turning only reorders the sectors.
- **The planner runs on its own thread**, and hands the game thread the change from the last plan: little while moving, the whole plan once after a teleport.
- **Surface tiles in the plan** (after merging `cubic/far-surface`; Custom worlds without `cubeAreas`, on the GPU path; Hybrid, Flat and the CPU path keep cube nodes everywhere):
  - From `SURFACE_LEVEL` (12) up, the plan is first a quadtree of tiles over X and Z (`plan_tiles`). A tile splits into its four children by the same screen-space rule, measured to the nearest point of the box its terrain fills (its header's skirt to its highest top; before it is generated, 64 below to 192 above the sea), so tall mountains and sky islands refine as near as they reach. Its own budget: `TILE_BUDGET` (3,584 tiles; the renderer's pool holds `TILE_CAPACITY`, 8,192, about 110 MB).
  - `node_kind(key, view, tile)` decides per footprint: a tile from `SURFACE_LEVEL` up, cube nodes below; a tile whose header counts complex columns turns into cube nodes `COMPLEX_LIFT` (3) levels sooner. A tile is generated first (it is the fastest coverage) and planned again as cube nodes once known to be complex.
  - The cube octree then fills the footprints the quadtree left to cube nodes: a node under a tile is left out, a node partly under one splits (rock partly under one is left out: the tiles' skirts close the ground). No cube node and tile share a footprint, at any height.
  - The footprints meeting the loaded cubes, the camera's own included, are always cube nodes (at the surface level's size): from a cavern deep down, the tiles elsewhere still draw the surface far above and the rock hides it.
  - The renderer makes the tiles itself (`Renderer::use_surface_tiles`, `queue_surface_tiles`): they are asked for as soon as they are planned, coverage tiles first, and count as delivered once their header is read back (`CubeLod::surface_feedback`). A tile is faded out (`SurfaceTiles::retire`) once everything planned in or over its footprint is in; a node under a planned tile stays until the tile is in. The coverage layer is tiles where it reaches them.
  - `status`'s `cube_lod` has `surface`, `tiles`, `tiles_ready`, `tiles_resident`, `tile_bytes`, `tiles_complex`; `planned`, `ready`, `cover_s` and `far_fill_s` count tiles as nodes.
  - `SURFACE_LEVEL` is a guess to measure: with exact heights, tiles may look right lower down.

**Measured** (`tools/cubic_far_bench.py --detail <step>`, Whole World, 1280 x 720 offscreen, 4-core i7 and GTX 1070 Ti; medians of 2 runs x 3 hops of 5,000,000 blocks; nodes and MB at the last hop):

| Step | Target px | Reached px | Nodes | Arena MB (peak) | `cover_s` | `far_fill_s` | Spawn cover / fill s | fps steady | fps min |
|---|---|---|---|---|---|---|---|---|---|
| main (`DETAIL` 2.5, ~13 px) | - | - | 7,444 | 11 | 2.15 (= fill) | 2.15 | 13.6 / 13.6 | 118 | 47 |
| Low | 16 | 16 | 4,116 | 9 | 0.08 | 0.60 | 2.0 / 9.3 | 118 | 95 |
| Normal | 8 | 8 | 13,507 | 15 | 0.08 | 2.81 | 0.9 / 11.2 | 120 | 97 |
| High | 4 | 4 | 44,842 | 47 | 0.14 | 16.7 | 1.4 / 30.8 | 120 | 97 |
| Ultra | 2 | 2.5 (budget) | 120,000 | 96 | 0.21 | 48.2 | 1.6 / 54.5 | 120 | 61 |
| Extreme | 1 | 2.5 (budget) | 120,000 | 95 | 0.22 | 48.4 | 6.6 / 54.1 | 118 | 69 |

- The coverage layer is in after 0.1–0.2 s at every step (main had no coverage: nothing showed until the nearest nodes landed). Normal is finer than main and fills about as fast: workers now wait for room in the GPU pool instead of failing a batch for a second, and take at most two CPU-sampled nodes per batch.
- The node budget (120,000) stops Ultra and Extreme at about 2.5 px at 720 pixels; it grows with the viewport's height squared, so at 1440 pixels High meets it too. True 1 px at 720 pixels needs about 700,000 nodes by the counts above (nodes grow with the square of the detail): about 700 MB of arena and a fill of four to five minutes at the measured ~2,600 nodes/s. Not measured yet: it needs a build with a bigger node table.
- Teleport hitch, measured before `RELEASE_BATCH`: the longest frame after a teleport was 17–81 ms on main, 117–131 ms at High and 235–281 ms at Ultra. The renderer retires each released node with buffer writes of its own (`CubeFar::upload`), and a teleport released the whole last plan in one frame; `removed` now hands out 4,096 at a time. To re-measure.
- Fill rate (~2,600 planned nodes/s, ~1,700 generated on the GPU) is the next lever, for the autoresearch loops: GPU batch sizes, the pool, the mesher's batch.
**With surface tiles** (the owner's quick set, `220f3ef986`, one run each, `D:\ClaudeData\far-detail-runs\quick`):

| Step | Nodes (tiles) | `cover_s` | `far_fill_s` | Spawn cover / fill s | fps steady / min | Longest frame after a teleport (p99) | Tile MB |
|---|---|---|---|---|---|---|---|
| **High (default)** | 35,794 (3,582) | 0.18 | 18.6 | 0.9 / 35.0 | 120 / 61 | 48 ms (35) | 75 |
| Ultra | 123,583 (3,584) | 0.31 | 72.2 | 1.1 / 76.7 | 103 / 60 | 116 ms (68) | 75 |

- **High is the default:** 4 times faster to complete than Ultra, at a full 120 fps, with a 48 ms longest frame after a teleport. It also stays under the node budget, which the spyglass needs to have room to refine.
- **`RELEASE_BATCH` worked:** the longest frame after a teleport fell from 235–281 ms to 116 ms at Ultra, and from 117–131 ms to 48 ms at High.
- **Ultra is slower with tiles** (72 s against 48 s without; 103 fps against 120). Not investigated yet. Candidates are the tile draw (each tile draws a fixed 256 × spans × 36 vertices, with no compaction) and the bigger plan. The tile budget (3,584) is full at every step (`tiles_limited`).
- **Coverage:** a teleport is covered at once (`cover_s` measured from the first plan; the coarse tiles already drawn cover the new place).
- **Views:** from y 300 to 5,000,000 there are no holes and no log errors. From 100,000 up, the ground is a mix of real biomes and oceans; the regular pattern is gone.
- **Tall world** (peaks of 8,000–12,000 blocks): complete in 1.7–5.3 s from y 60,000, with 16 complex tiles. From y 20,000 at 60° it took over 180 s (budget-limited).
- **Cavern:** at y -2,000, in rock and in a carved open space, it is complete in 5.7 s and 0.2 s, with no holes, no errors and no tile over the camera.
- **Spyglass, at High:** the field of view goes to 7°, and the plan refines what it looks at (35,881 → 123,580 nodes, the budget, in 25 s). Zoomed in, the budget binds at about 9 px per voxel. It goes back to 50,007 nodes 11 s after letting go. The first attempt, at Ultra, was invalid: the plan was already at the budget, and the check let go before its screenshot.
- **Dark walls fixed.** Both far meshers lit a face by whether its own voxel saw the sky, so only a column's top voxel was in the sun, and tall cliffs and pillars drew near-black. A face now takes the light of the cell in front of it, as in Minecraft. The tall world's pillars now draw with lit stone walls (`D:\ClaudeData\far-detail-runs\light\tall_cmp.png`).
- **Not run:** `SURFACE_LEVEL` 10/14, the per-step sweep with tiles at Low/Normal/Extreme, and true 1 px.

### The owner's first play test of the rework (2026-11-06, evening)

The owner's verdict: "Not perfect yet, but WOW, this is insanity!" These are the issues the owner reported, what caused each one, and the fix.

1. **Water vanished at closer and middle distances.** There were two causes, one at each end of the cube levels.
   - **Levels 0–2.** The sea's surface (y 64) falls on a node's top. The mesher leaves out a fluid's faces on a node's edge, which keeps far oceans from showing a grid of node outlines (Nov 5). That left out the sea's own surface too, and a node of nothing but sea under it was dropped as uniform. The sea showed its floor nearest the camera, at Ultra from 1,500 blocks up.
     - **Fix (`MeshJob::sea_top`):** a fluid's top on a node whose top is the surface is drawn, by both meshers. A node of all sea up to the surface counts as mixed, in the pre-pass and in the descent, and draws that top (`uniform_quads`).
   - **Levels 7 and up.** The generator made a voxel water only when its centre lay below the surface (sea level + 1). From 128-block voxels up, the centre of the voxel holding the surface lies above it, so that voxel was air.
     - **Fix (`cube_gen` `water_at_sea`):** an open voxel holding the surface is water when the column is open just under the surface (one more density sample, in that row only). The pre-pass and the GPU descent treat an all-open voxel as water when its bottom lies below the surface.
     - **Drawing:** both meshers draw water in such a voxel up to the surface only, never as a slab 2,048 blocks deep (`cube_far.wgsl` takes the surface from the cull pass; the CPU mesher from `CubeSampler::sea_surface`).
   - **Found on the way:** a uniform rock node drew its top, south and east faces one voxel from its bottom, north and west instead of on its far side (`uniform_quads`). Fixed.
2. **A gap followed the player between far and near terrain.** The near renderer draws a circle of columns (`streaming::in_view_distance`). The octree left out every lit cube in the square the game holds, render distance + 2. The ring in between was drawn by neither.
   - **Fix:** the octree leaves out only lit cubes within the circle.
3. **Dark stripes on far cliffs** (the face-light change had caused them). A wall on a node's side took its own column's sky light at the node's top. A cliff that rises past the node has rock over it there, so its walls went dark one node at a time.
   - **Fix:** a wall on a node's side looks into the neighbour node where it is known (open from that node's top down to the cell in front), and is lit where it is not.
4. **Z-fighting through the spyglass at y 10,000.** A coarse tile or node stays drawn until everything planned in its place is in, which can take tens of seconds when zoomed. Meanwhile the finer nodes that are already in are drawn in the same place: same depth push, water and flat ground in the very same planes.
   - **Fix (`cube_far.wgsl` `cf_by_level`):** coarser far terrain is pushed back by 2e-5 of its distance per level, so the finer node wins.
5. **Ultra went back and forth between levels.** Reproduced at 1080p from y 1,500: with the camera still, there were 23 plans in 30 s, alternating between about 2.4 and 4.3 px per voxel, and the plan never completed. The cause is the class cache.
   - **The cycle:**
     - At Ultra, one descent answer holds about 150,000 uniform nodes. The cache held at most 262,144 (`CLASS_NODES`) and was emptied whole when full, keeping only the newest answer.
     - With the union of a fine and a coarse answer, the plan reached 2.4 px. With one alone, it reached 4.3 px.
     - The descent is asked at the detail the last plan reached (`coarsened`), so every plan asked again, about 6 times a second, and the cycle never ended.
   - **Fix (`KnownClasses`):** two generations, a full one becoming the older, so the last answers are always all kept.
   - **Also:** the arena's cap on the budget now holds until the room moves outside 0.9–1.25 of it (`ARENA_CAP_BAND`). It was not this cycle's cause, but it could start the same kind of swing where the arena binds.
6. **The spyglass refines slowly.** It was mostly item 5's cycle. With the cycle gone, the zoomed view at y 10,000 completes in 3.8 s, where it had not completed after 120 s. A zoom that needs many more new nodes is still bound by fill rate (2,600–3,500 nodes a second). The fill-rate loops (Order of work 6.5) are the lever there.
7. **Chunks flashed in.** Vanilla's chunk fade-in mixes a new section toward the fog colour, so a flash of sky showed where a chunk came in.
   - **Change (the owner's request):** solid terrain dithers in, as far terrain fades. Translucent faces draw whole from the start, as far water already does.
   - Not changed: the Metal opaque pass, which cannot discard, still mixes toward the fog colour, and so does the opt-in visibility buffer.

Checked in the game (`tools/cubic_far_check.py`, `D:\ClaudeData\far-fixes-runs\r2`):
- **Ultra at 1920 × 1080, from y 1,500:** no new plan in 30 s with the camera still (23 before). The view completes in 14.4 s; before, it never completed.
- **Spyglass at y 10,000, looking down (High):** complete in 3.8 s (15,147 → 38,860 nodes). Before, it was not complete after 120 s. Most of the slow refinement the owner saw was the same class-cache cycle, not fill rate.
- **Water:** at Ultra from y 1,500, the sea nearest the camera showed sand and gravel; it now has its water (`far-fixes-runs\ultra_water_cmp.png`). Through the spyglass, rivers and seas show where there were dry riverbeds.
- **Cliffs (tall world):** no node-edge stripes. Walls under overhangs are still drawn at sky light 4, which is harsher than Minecraft, where light spreads sideways under an overhang. Waiting on the owner: is that what still looks off? Softening it is one constant (`SHADED` and the shaders' 64).
- No holes and no log errors in any run.
- Tests:
  - A full test pass ran before the class-cache, sea-top and uniform-face changes. Its one failure was a flaky LAN test that passes alone.
  - After those changes, the render crate's whole suite ran: 1,636 passed.

### Where the GPU far view stands

Done: generate, mesh and draw (stages 1, 3 and most of 4) run on the GPU for a cubic level's far nodes; the CPU only chooses which nodes exist.
- **Material table** (`gfx/cube_mesh.rs`): a storage buffer of 32 words per block state (layer, tint kind, translucency, constant tint, opaque, fluid, HDR tier for each direction and overlay). It is built from `lod::mesh::far_style`, the one lookup the CPU mesher uses too.
- **Compute mesher** (`cube_mesh.wgsl`): one workgroup per (node, direction) walks the 16 slices. Faces follow `mesh_cube`'s rules (a different fluid or a transparent neighbour exposes one; a node's edge is a wall, a fluid's edge is left out). Runs along one axis are merged, then identical runs of the rows after, so a merged quad tiles its texture once per block.
  - A count pass gives each node's quads per (direction, solid/water). The CPU reads those 48 bytes back, takes one range per node from the arena's first-fit allocator (`RangeAlloc`), and the emit pass writes the quads there. Voxels never leave the GPU.
  - Quads are 8 bytes: x, y, z, w-1, h-1, direction, shaded and overlay in word 0, state id and biome in word 1 (layout in the module docs). Texture, tint and emission are looked up at draw time.
  - Neighbour slots are part of a job (`MeshJob::neighbours`) and the shader hides border faces behind them, but `CubeFar` passes none yet: borders stay walls, which is what keeps levels crack-free until a node's neighbours are remeshed when they change.
  - Uniform nodes from the pre-pass take no slot: air draws nothing, a fluid only the sea's surface on its top (`MeshJob::sea_top`), and rock gets its six faces written by the CPU.
- **Draw** (`gfx/cube_far.rs`, `cube_far.wgsl`, `cube_cull.wgsl`): a vertex-pulling pipeline appended to `terrain.wgsl`, so the fragment stage, fog, light, fade dither and emission are the far tiles'. A culling compute pass tests each node against the frustum and the camera's side of each direction, writes the instance data and compacts one indirect draw per (node, direction, pass), drawn with `multi_draw_indirect_count` (or `multi_draw_indirect` over a cleared tail where counts are unavailable, DX12 included). Node origins are `i32` blocks minus the camera's integer part, so positions stay exact at 30,000,000 blocks. The Hi-Z test of `gpu_occlusion.wgsl` has a marked hook in `cube_cull.wgsl`.
- **`CubeLod` switches paths** with `set_gpu_path` (the sampler must have a GPU generator on the renderer's device and the "Far Terrain on GPU" setting must be Auto). On the GPU path workers hand out slots (`GpuNode`) instead of meshes; CPU-sampled nodes (saved cubes, downsampled fine levels) are written into pool slots with `write_buffer`. The renderer frees a slot once its emit pass is submitted, and a node's arena range and table entry when the plan drops it.
- **Whole World**: the octree root is level 22, the LOD Distance slider's top step reads "Whole World", and the planner leaves out nodes outside the world border and cube range. A classifier thread asks the sampler's 2D pre-pass (`node_class`) about nodes: uniform air or fluid subtrees are left out, uniform rock stays one closed leaf.
- **Progress**: `CubeLod::stats()` (`CubeLodStats`) counts planned, ready, pending and uniform-skipped nodes, GPU nodes generated and meshed, arena and pool bytes, and stamps when the plan changed and when it last became complete (every planned node drawn or known empty, nothing waiting in the mesher).

Not done: node selection on the GPU (traversal by screen-space error), Hi-Z occlusion of far nodes, wiring the GPU biome columns (`gfx/biome_gen.rs`, landed separately) into this path, remeshing a node when its neighbours change (so no border culling), GPU far water in OIT or shader-pack frames, the swamp's grass noise across a biome, and sorting far water.

**Surface tiles** (`gfx/surface_tile.rs`, branch `cubic/far-surface`; points 2, 3 and 6 of the Meridian list; not wired into the plan yet):
- A `TileKey { level, x, z }` covers a `CubeKey`'s footprint, all Y, as 16 × 16 columns. Each column holds up to 4 solid spans (top, bottom, top block; the ground's bottom is open), exact `i32` block heights with no fixed range, and water over the highest span when it lies below the sea. More spans: the rest merge into the last and the tile counts the column as complex (`TileInfo::complex`), for the planner to prefer cube nodes there.
- Generated on the GPU in batches of 256 (`surface_gen.wgsl`): the climate pass (`biome_gen`, jittered) gives each column one sample point at a hashed quart, with the climate's layers down to 256 blocks and the density's octaves down to 64 (a point, not a column average); that line is marched (column-wide steps, at most 64 blocks) and bisected to the block. No colour blending (the owner's call); the jitter is what removes the regular tile pattern from high up. Several points per column with the highest kept is a constant away (`SAMPLES`), but four took 2.5 times the GPU time and, at coarse levels, turned the oceans into land.
- Measured (GTX 1070 Ti, defaults): 256 level-14 tiles in 4.7 ms of GPU time (climate 3.2 + surface 1.5); 1,089 tiles generated and read back 31 ms after the request. 13,376 bytes a tile.
- Drawn by vertex pulling through `terrain.wgsl` and `cube_far.wgsl`'s material lookup (`cf_face`): blocky span boxes, skirts at tile edges down to the tile's lowest possible ground minus its relief, water at the sea's plane; culled per tile into indirect draws.
- The noise maths of the four far-terrain compute shaders now lives once, in `gfx/cwg_noise.wgsl`.
- Wired into the plan (see "Detail by size on screen, coverage first" above): `SurfaceTiles` is built beside `CubeFar` (`GpuBackend::params`, `biome_params`, `CubeFar::material_buffer`, `biome_buffer`), `update` and `prepare` run with `CubeFar::prepare`, `draw` next to `CubeFar::draw`. The temporary debug view is gone.


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
6. **Far-view rework (2026-11-06, merged).** Steps 1–5 above are done.
   1. `cubic/far-detail`:
      - detail by size on screen (FOV and spyglass aware);
      - a Far Detail setting (top step about 1 px per voxel);
      - a coarse-first coverage layer;
      - on-screen-first priority;
      - the benchmark's two numbers, `cover_s` and `far_fill_s`;
      - the per-node `node_kind` seam.
   2. `cubic/far-surface`: layered height-map tiles on the GPU for the coarsest levels, with exact heights and jittered samples.
   3. Lead: merge both, wire the tiles into the plan through `node_kind`, one test pass, benchmark at each Far Detail step, screenshots from y 300 to 5,000,000, install. Done, except benchmarking every step: only High and Ultra were run with tiles.
   4. Still to measure: `SURFACE_LEVEL` 10/14, Low/Normal/Extreme with tiles, true 1 px. Then find why Ultra is slower with tiles; the first suspect is tile draw compaction.
      The owner's first play test is fixed: mid-distance water, the gap at the render distance, dark cliff stripes, spyglass z-fighting, Ultra's swings, and the chunk fade-in dithered. See "The owner's first play test of the rework".
   5. Then, in this order:
      1. far views of the player's builds;
      2. occlusion-driven refinement;
      3. far block light;
      4. structures in the far view (Custom has structures now);
      5. the thin-feature flex.
      See "Far view: the owner's decisions and later work".

## Cut or deferred, on purpose

- CubicWorldGen's `alternateStrongholdsPositions`: kept as a field, unused.
- `mesaSurface`: simplified clay bands.
- The replacer and cube-area editors in the Customize screen: CubicWorldGen never had them either. The JSON covers them.
- Hybrid's vanilla band on the GPU, which needs the 26.3 router in WGSL.
- Cubic to vanilla conversion. Only vanilla to cubic is planned.
- Floating islands and v1's deep generator: removed. Custom Cubic replaces them.
