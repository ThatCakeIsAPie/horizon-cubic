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
2. **A surface layer for the coarsest levels.** A 16³ node at level 15 has voxels 32,768 blocks tall, while all the terrain lies within a few hundred blocks. So the surface snaps to whole voxels, and the top face can be the stone below the surface. Above some level, nodes should be 2.5D tiles holding, per column, a height, a top block and water, taken from the 2D pre-pass's columns. That is 16 times less work per node, with no vertical snapping.
3. **Filter colour at the coarse levels.** Average the column colours, as mipmaps do, so the far ground does not shimmer.
4. **Fog follows the far view.**
   - Fixed on 2026-11-06: cubic worlds never told the fog that far terrain was on (`FogInputs::distant_terrain` checked only the column LOD). Atmospheric fog therefore stopped at about 1,000 blocks, and from high up the ground faded into the sky.
   - Later: a horizon fade setting like Meridian's.
5. **Choose detail by size on screen, not by distance alone.**
   - The plan splits a node while it is nearer than `DETAIL` (2.5) times its size, whatever the field of view. So a far voxel spans about 1/40 of its distance, about 25 pixels on a 1280-pixel, 70° view.
   - That is why the far view looks blocky, and why "complete" does not mean full detail.
   - Voxy-class detail needs about 1–2 pixels per voxel. To get there:
     - choose levels by projected voxel size, using the field of view and the resolution;
     - refine what is on screen first;
     - make the target a setting (like Meridian's Detail), and measure the fill time at each step.
6. **The coarse levels alias.** From y = 5,000,000 the ground is a regular tile pattern. Each coarse voxel point-samples the noise at a spacing that lines up with the noise's period. The surface layer (2) with averaged colour (3) fixes this.

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
- **Seam for the 2D tiles:** `node_kind(key, view, tile)` and `SURFACE_LEVEL`. It answers `Cube` until the tile module gives tile information.

**Measured** (`tools/cubic_far_bench.py --detail <step>`, Whole World, 1280 x 720 offscreen, 4-core i7 and GTX 1070 Ti; medians of 2 runs x 3 hops of 5,000,000 blocks; nodes and MB at the last hop):

| Step | Target px | Reached px | Nodes | Arena MB (peak) | `cover_s` | `far_fill_s` | Spawn cover / fill s | fps steady | fps min |
|---|---|---|---|---|---|---|---|---|---|
| main (`DETAIL` 2.5, ~13 px) | - | - | 7,444 | 11 | 2.15 (= fill) | 2.15 | 13.6 / 13.6 | 118 | 47 |
| Low | 16 | 16 | 4,116 | 9 | 0.08 | 0.60 | 2.0 / 9.3 | 118 | 95 |
| Normal | 8 | 8 | 13,507 | 15 | 0.08 | 2.81 | 0.9 / 11.2 | 120 | 97 |
| High | 4 | 4 | 44,842 | 47 | 0.14 | 16.7 | 1.4 / 30.8 | 120 | 97 |
| **Ultra (default)** | 2 | 2.5 (budget) | 120,000 | 96 | 0.21 | 48.2 | 1.6 / 54.5 | 120 | 61 |
| Extreme | 1 | 2.5 (budget) | 120,000 | 95 | 0.22 | 48.4 | 6.6 / 54.1 | 118 | 69 |

- The coverage layer is in after 0.1–0.2 s at every step (main had no coverage: nothing showed until the nearest nodes landed). Normal is finer than main and fills about as fast: workers now wait for room in the GPU pool instead of failing a batch for a second, and take at most two CPU-sampled nodes per batch.
- The node budget (120,000) stops Ultra and Extreme at about 2.5 px at 720 pixels; it grows with the viewport's height squared, so at 1440 pixels High meets it too. True 1 px at 720 pixels needs about 700,000 nodes by the counts above (nodes grow with the square of the detail): about 700 MB of arena and a fill of four to five minutes at the measured ~2,600 nodes/s. Not measured yet: it needs a build with a bigger node table.
- Teleport hitch, measured before `RELEASE_BATCH`: the longest frame after a teleport was 17–81 ms on main, 117–131 ms at High and 235–281 ms at Ultra. The renderer retires each released node with buffer writes of its own (`CubeFar::upload`), and a teleport released the whole last plan in one frame; `removed` now hands out 4,096 at a time. To re-measure.
- Fill rate (~2,600 planned nodes/s, ~1,700 generated on the GPU) is the next lever, for the autoresearch loops: GPU batch sizes, the pool, the mesher's batch.
- Not run yet (`tools/cubic_far_check.py`): the spyglass (`--spyglass`), a cavern at y -2,000 (`--cavern`), the teleport hitch after `RELEASE_BATCH` (`--hitch`).

### Where the GPU far view stands

Done: generate, mesh and draw (stages 1, 3 and most of 4) run on the GPU for a cubic level's far nodes; the CPU only chooses which nodes exist.
- **Material table** (`gfx/cube_mesh.rs`): a storage buffer of 32 words per block state (layer, tint kind, translucency, constant tint, opaque, fluid, HDR tier for each direction and overlay). It is built from `lod::mesh::far_style`, the one lookup the CPU mesher uses too.
- **Compute mesher** (`cube_mesh.wgsl`): one workgroup per (node, direction) walks the 16 slices. Faces follow `mesh_cube`'s rules (a different fluid or a transparent neighbour exposes one; a node's edge is a wall, a fluid's edge is left out). Runs along one axis are merged, then identical runs of the rows after, so a merged quad tiles its texture once per block.
  - A count pass gives each node's quads per (direction, solid/water). The CPU reads those 48 bytes back, takes one range per node from the arena's first-fit allocator (`RangeAlloc`), and the emit pass writes the quads there. Voxels never leave the GPU.
  - Quads are 8 bytes: x, y, z, w-1, h-1, direction, shaded and overlay in word 0, state id and biome in word 1 (layout in the module docs). Texture, tint and emission are looked up at draw time.
  - Neighbour slots are part of a job (`MeshJob::neighbours`) and the shader hides border faces behind them, but `CubeFar` passes none yet: borders stay walls, which is what keeps levels crack-free until a node's neighbours are remeshed when they change.
  - Uniform nodes from the pre-pass take no slot: air and fluid draw nothing, rock gets its six faces written by the CPU.
- **Draw** (`gfx/cube_far.rs`, `cube_far.wgsl`, `cube_cull.wgsl`): a vertex-pulling pipeline appended to `terrain.wgsl`, so the fragment stage, fog, light, fade dither and emission are the far tiles'. A culling compute pass tests each node against the frustum and the camera's side of each direction, writes the instance data and compacts one indirect draw per (node, direction, pass), drawn with `multi_draw_indirect_count` (or `multi_draw_indirect` over a cleared tail where counts are unavailable, DX12 included). Node origins are `i32` blocks minus the camera's integer part, so positions stay exact at 30,000,000 blocks. The Hi-Z test of `gpu_occlusion.wgsl` has a marked hook in `cube_cull.wgsl`.
- **`CubeLod` switches paths** with `set_gpu_path` (the sampler must have a GPU generator on the renderer's device and the "Far Terrain on GPU" setting must be Auto). On the GPU path workers hand out slots (`GpuNode`) instead of meshes; CPU-sampled nodes (saved cubes, downsampled fine levels) are written into pool slots with `write_buffer`. The renderer frees a slot once its emit pass is submitted, and a node's arena range and table entry when the plan drops it.
- **Whole World**: the octree root is level 22, the LOD Distance slider's top step reads "Whole World", and the planner leaves out nodes outside the world border and cube range. A classifier thread asks the sampler's 2D pre-pass (`node_class`) about nodes: uniform air or fluid subtrees are left out, uniform rock stays one closed leaf.
- **Progress**: `CubeLod::stats()` (`CubeLodStats`) counts planned, ready, pending and uniform-skipped nodes, GPU nodes generated and meshed, arena and pool bytes, and stamps when the plan changed and when it last became complete (every planned node drawn or known empty, nothing waiting in the mesher).

Not done: node selection on the GPU (traversal by screen-space error), Hi-Z occlusion of far nodes, wiring the GPU biome columns (`gfx/biome_gen.rs`, landed separately) into this path, remeshing a node when its neighbours change (so no border culling), GPU far water in OIT or shader-pack frames, the swamp's grass noise across a biome, and sorting far water.


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
