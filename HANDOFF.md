# Handoff: `ttm/bake-lod` (2026-10-24)

Work done on the Windows PC to play the JJThunder To The Max world on Horizon, and
to make its far terrain (LODs) and chunk loading faster. Branch: `ttm/bake-lod`,
off `main` at `3051dda`.

## Prompt for the next agent

> Pick up the Horizon To The Max work on branch `ttm/bake-lod` of
> ThatCakeIsAPie/horizon-ttm. Read `HANDOFF.md` at the repo root first. Continue
> the autoresearch loop on chunk generation described under "Autoresearch loop":
> one change at a time, measured with `world_chunkgen_bench` interleaved against
> the previous build, kept only if finished chunks/s improves beyond noise with
> every output checksum unchanged. Stop when the loop has 10 kept commits in
> all (4 are done). Local commits only: don't push or merge to `main` without
> asking me. Follow AGENTS.md.

(Done on the Linux laptop, 2026-10-24: the loop has its 10 kept commits; see
"Autoresearch loop". Nothing pushed.)

## What's on this branch

1. **Baked LODs** (`crates/horizon-client/src/lod/bake.rs`, hooked into
   `SavedLodSampler` in `save.rs`). The generator's far samples (`grid`) and
   detail-chunk columns are saved per dimension in `<dimension>/horizon_lod/`
   (append-only zlib records in bucket files, crash-safe) and read back on the
   next launch. Saved chunks still override baked data, so edits show. A
   fingerprint (game exe size+mtime, data packs, block/biome registry sizes,
   column layout) guards the folder; a mismatch empties it. ~9 MB for a
   256-chunk LOD distance. Tests: `lod::bake::tests`.
2. **Faster far sampling** (`lod_grid` in `horizon-world/src/worldgen/generator.rs`):
   - Biome lookups memoized per quart within a tile (`Biomes::Memo`,
     `BiomeMemo`): the surface rules were re-resolving the biome (~50 µs each)
     for 8 depths of every column. Exact: identical output (checked by checksum).
   - Approximations for wide cells only: cells >= 4 blocks use one biome per
     column for painting; the vertical density lattice is 16 blocks for cells
     >= 32 wide and 32 for >= 128 (8 otherwise).
   - Far sampling threads: `(cores/2 - 1).clamp(1, 6)` (was `cores/4`, max 3),
     `SAMPLE_JOBS` 4 -> 6, threads call `horizon_threads::behind_server()`.
   - Result, ms per far tile on one thread: level 0 152 -> 13, level 4 159 -> 65,
     level 8 185 -> 58.
3. **Windows mouse fix** (`keep_captured_cursor` in `game.rs`): Windows only
   confines the hidden cursor, and the clip gets dropped (alt-tab, overlays), so
   turning walked the cursor out of the window and clicks landed outside. Now
   the captured cursor is re-centred and re-clipped on every move (GLFW does the
   same). Not yet confirmed in a long play session.
4. **Benchmarks** (`crates/horizon-client/examples/`):
   - `lod_grid_bench <resources> <world> [--tiles N] [--levels 0,2,4]`: far tile
     cost per level, with an output checksum.
   - `world_chunkgen_bench <resources> <world> [--size N] [--threads T] [--center CX CZ]`:
     full chunk generation with the world's data packs: the one-time stronghold
     search, per-stage times, a structure-start checksum, decoration.
     It keeps the stronghold positions in the world folder as the game does
     (see 5), so the second run reads them; `--no-ring-cache` searches. It
     also prints a checksum of all ring positions (the structure checksum
     only covers the area around `--center`).
5. **Stronghold positions kept with the world** (`crates/horizon-client/src/ring_cache.rs`,
   hooked in `world_loading::make_streaming` via `VanillaTerrain::keep_rings`).
   `<dimension>/horizon_rings.json` holds every concentric-ring set's chunks
   under a fingerprint (format, version, exe size+mtime, seed, dimension, a
   digest of the resources' `worldgen/` files and of the world's data packs);
   a mismatch searches again and rewrites the file (written whole, then
   renamed). horizon-world side: `OverworldGenerator::{rings, preset_rings,
   on_rings}`; `preset_rings` refuses positions that don't give each ring set
   its count, or that its biome search could not return
   (`RingSearch::may_find`: within 7 chunks of the ring chunk).
   Measured with `world_chunkgen_bench` (TTM, seed 0, 6 threads):
   - first chunk: 26.6 s at the branch's previous commit, 24.2 s with
     `--no-ring-cache`, 23.7 s on the first cached run (searches and writes),
     **0.6 s** on the second (reads);
   - ring positions identical searched and read (128, checksum
     `42c7fef9c3919780`); structure-start checksum identical to the previous
     commit at (0, 0) (`66c9d0cf7819f968`), at stronghold ring chunk -15 -103
     (`0daeb25dec9b3371`) and at ring chunk 223 -203, size 4 (3 starts,
     `67ae2d4769bb55e9`).
   - Note: TTM's `stronghold.json` only allows `minecraft:underground` and
     `minecraft:underground_deep`, so many ring chunks get no stronghold start
     (the same before and after; not checked against vanilla).

## Autoresearch loop (chunk generation)

Goal: more finished chunks/s in `world_chunkgen_bench`, generation output
bit-identical. **Done: 10 kept commits** (the two bench-checksum commits are
setup and don't count): 4 on the Windows PC, then 6 on the Linux laptop from
three agents working in parallel, one area each (branches `ttm/cg-surface`,
`ttm/cg-noise`, `ttm/cg-structures`), cherry-picked together here.

**Setup.** TTM world copy (never the real save: the bench writes
`horizon_rings.json` into it), seed 0, `--size 16 --threads 6`. The bench's
output checksums: terrain (every block, quart biome, heightmap, post-processing
mark, block and fluid tick, block entity, entity, structure reference), the
same after decoration, structure starts, and ring positions. All must match
the previous build's and be the same run to run.

**Measuring.** Background load swings the numbers by 20% or more (the first
baseline, 35.0, was taken under load; interleaved it is 43.9). So each
candidate runs interleaved with the previous build's binary, 3+ pairs, and is
kept when its median is >= 2% better with ranges not overlapping. Keep each
build's bench binary so it can be rerun.

| Commit | Change | Interleaved finished chunks/s |
|---|---|---|
| `fe8a9dcb` | Surface biome conditions compare biome ids (bitsets), no `Arc<str>` clone per block | 43.9 -> 50.4 (its message says 35.0 -> 48.9; that baseline was under load) |
| `c2b6961f` | `references`: build start chunks no other thread holds, then wait for the rest | 49.3 -> 50.8 |
| `593f6d5a` | Per-worker memo of structure checks' single noise-biome samples | 51.9 -> 57.0 |
| `a129fb20` | Surface: skip rewriting the block already there (stage heightmaps kept) | 57.0 -> 60.3 |
| discarded | Same "build free ones first" for neighbour biome chunks | 50.6 -> 50.8 |
| discarded | Ore vein / uncached conditions out of line (smaller `apply` frames) | 59.0 -> 59.5 |

Windows PC (i7 4c/8t): **43.9 -> ~60 finished chunks/s (+37%)**.

Linux laptop (Ryzen 5 7640U 6c/12t; numbers are higher here, so compare
within a row only; each row interleaved with its branch's previous build):

| Commit | Change | Interleaved finished chunks/s |
|---|---|---|
| `c32d434a` | Ore veins: skip heights where their eagerly sampled density can't be > 0 (TTM's rule starts with two veins) | 108.7 -> 115.7 |
| `fc5a81ea` | Surface: the compiled rule runs as a flat program (jumps past failed conditions), not a recursive tree | 124.3 -> 134.4 |
| `47715c83` | `DensityVolume` accessors `#[inline]` (the fast build has no LTO; `steps` was 3.6% of CPU) | 109.0 -> 114.3 |
| `5e0a3185` | Noise fill reuses the terrain-density cells the biome stage sampled (TTM's ridges are a spline of that density); 1024-chunk cache, only for pointwise samplers | 121.6 -> 129.3 |
| `3bf7bcf8` | Structure column checks sample the column's climate in one batched volume, as vanilla's `couldStructureExistInColumn` does (was ~525 single samples a column) | 116.4 -> 138.2 |
| `47bc505f` | Decoration setup: biome list by bitset, one tick tracker per region, one-section placeholder chunks | 136.6 -> 139.4 (6 pairs) |
| discarded | Perlin `add_volume`: reuse a Y sample identical to the previous one | 123.8 -> 124.3 |
| discarded | Splines as flat arrays with a last-interval hint (spline self 6.1% -> 5.1%) | 124.4 -> 125.1 |

**Combined (all 10), laptop, 5 interleaved rounds:** `9d03d424` 115.7
(115.0-118.5) -> **158.9 (158.5-162.6), +37%**; each branch alone: surface
133.5, noise 129.9, structures 138.5 (the gains overlap, so they don't add).
All four checksums identical in all 25 runs. Per chunk on one thread, surface
18.8 -> 13.9 ms. Laptop bench tooling (outside the repo): `~/Work/ttm-bench`
(`build.sh`, `ab.sh`, `profile.sh` + `tools/top.py` for samply, `BRIEF.md`,
`log-*.md` with every attempt).

**Profile at `a129fb20`** (a sampling profiler suspending the workers, Windows
without admin; on Linux use `perf`/`samply`): surface 41% of samples
(`CompiledRule::apply` 28.5% inclusive, mostly call overhead per node per solid
block: ~118k solid blocks a chunk since TTM reaches y 2032), noise fill 23%
(`CubicSpline::sample` 12% inclusive, half in climate `sample_volume` for the
biome stage, half in `fill_noise`; TTM's splines take y-dependent inputs, so
they run per cell corner), decoration 12%, structure starts 10% (half of it
waiting on another thread's build of the same start chunk, line 323 of
`generator_structures.rs`), biome-chunk waits 3-4%, Perlin `get3` 9% self (4%
from per-column 2D noise conditions, 2.4% climate samples for structures).

**Ideas not tried yet** (from the laptop agents' round-1 reports; several were
done in round 2, below):
- Surface: a per-column cache of the current quart cell in
  `SurfaceQueries::biome_id` (up to ~3.8% of CPU); sample the shared ore
  richness volume only up to the heights its gates allow (~1.4%); fold
  `Compiled::test` into one match without revision stamps for block-level
  nodes. The per-column outcome memo ranks below these (biome changes with y in
  mixed cells; gradients and the sulfur 3D noise would have to be keyed).
- Noise: single-point climate samples (structure and carver biome checks,
  ~10% of CPU) could read the same cached cells; the fill loop's per-block
  `dyn` aquifer call (`compute_substance` ~4%); biome-name lookups and the 9
  neighbour `BiomeChunk` clones are small (<= 1.5%). Perlin `get3` (11% self)
  is tight; only fewer calls help.
- Structures: what's left is three slow start chunks; the abandoned camp's
  58 ms is 52 ms of first-use template loading (Horizon loads a whole pool's
  templates at once, vanilla each on first try). "Help while waiting" gains
  little: other start chunks take ~0.01 ms.
- Decoration (thin now): dripstone clusters 21% of it (block reads through four
  layers), ore `any_height_at_least` ~4%, placement biome lookups ~6%; each is
  0.5-1% overall, so group them to clear the 2% bar.

**Pitfalls met:** `Debug` of `ProtoChunkTicks` includes a `HashSet` (random
order): hash `pack(0)` instead. In Rust 2024, `while let Some(x) =
m.lock().unwrap().pop()` holds the lock for the whole body.

**On another machine:** the bench needs Horizon's prepared resources
(`.local/horizon-client/26.3`, from the launcher or `horizon-prepare`) and a
copy of the To The Max world with its data pack; neither is in the repo.
Numbers are per machine: take a new baseline there before comparing.

### Round 2 (Linux laptop, 2026-10-24/25)

Same loop and areas, branches `ttm/cg2-{surface,noise,structures}`, stop at 5
kept or 15 discarded each; combined as `ttm/cg2-combined`.

**Measuring, tightened.** Bench builds use `codegen-units=1`, non-incremental:
with the fast profile's 16 units, unrelated edits moved inlining and swung
results ~6%. One unit still flipped the aquifer's fast path in or out of
`fill_noise` with the size of `Pipeline::generate` (noise 6.6 vs 8.3 ms), so the
round's base `43025047` pins `NoiseBasedAquifer::compute_substance` with
`#[inline(always)]`. Smaller flips remain (`PerlinNoise::sample`,
`BlockSection::get`, `BiomeZoom::quart_at_block`): stages can trade off from
codegen alone, so candidates are scored on overall chunks/s. A 6-pair run that
wins every pair, with a >= 2% median and the targeted stage better in every
pair, also keeps when drift makes the unpaired ranges overlap. Each worktree
builds in its own target dir: cargo fingerprints workspace crates by relative
path and mtime, so a shared dir can link another worktree's `horizon-world`.
The builds are byte-deterministic, so kept commits were checked by rebuilding
them and comparing bytes.

| Commit | Change | Interleaved finished chunks/s |
|---|---|---|
| `43025047` | Noise fill calls the aquifer through its concrete type; its fast path `#[inline(always)]` | ~3% (base of the round) |
| `2c35e9d3` | Column searches for the top solid block memoized across chunks and threads (only for position-only densities) | 173.3 -> 179.9 |
| `be717823` | Noise fill visits and writes one section at a time; all-default sections set at once | 176.15 -> 179.75 |
| `94a58f2f` | Biome stage builds the neighbour biome chunks no other thread holds, then waits for the rest (~0.5 s of lock waits in a 1.1 s terrain phase) | 181.75 -> 192.6 |
| `7b672e3d` | Climate volume gets one more top layer for the fill; biome-id bitset, biome name memo | 190.85 -> 194.8 |
| `1cdb36af` | Surface biome cache keeps each mixed quart cell's corner jitter and 8 biomes | 171.7 -> 176.55 |
| `0e4ad05e` | 3D noise conditions sampled 16 blocks down the column (`Noise::get3_column`); conditions in a compiled rule tested without the revision cache | 174.95 -> 186.4 |
| `9ffa1c00` | Ore vein volumes sampled only up to their gate's highest in-range height (115 rows, not ~2100) | 185.7 -> 194.55 |
| `e09fb9dd` | Surface walk: no-op heightmap updates skipped below the tops, stone-depth scan reads reused | 193.85 -> 201.55 |
| `aac037d2` | Column snapshot per column, vein volumes trimmed below their gate, lazy richness, Perlin 4 heights a step, inlined condition test | 201.1 -> 205.9 |
| `df81a9de` | Block reads inlined from region to palette, multiply instead of a division per read | 169.25 -> 174.55 |
| `b2c56c9c` | Decoration: column scans, per-chunk heightmap reads, no zoom in single-biome sections | 175.0 -> 180.75 |
| `a9a22ec0` | Jigsaw pool templates loaded on first use, as vanilla does (abandoned camp: 180 loads -> 18) | 180.75 -> 185.45 |

Discarded: 9 noise, 9 surface, 4 structures, with numbers in the laptop's
`~/Work/ttm-bench/log2-*.md`. Of note: `#[inline(always)]` on the block-read
path (-1.8%) and on Perlin's helpers (-0.4%); reusing the fill's 2 MB density
buffer per thread (surface +0.3 ms a chunk every run); jigsaw string sharing
(structures 0.8 -> 0.4 ms, but +0.3% overall).

**All builds, laptop, cgu1, 7-way interleaved, 5 rounds** (median, range):

| Build | Finished chunks/s |
|---|---|
| `9d03d424` (before round 1) | 124.2 (122.8-125.7) |
| `4c3a22fb` (round 1) | 169.0 (168.3-171.8), **+36%** |
| `43025047` (round-2 base) | 174.8 (173.4-178.3) |
| noise branch alone | 197.7 |
| surface branch alone | 207.8 |
| structures branch alone | 189.2 |
| **`a9a22ec0` (round 2 combined)** | **275.4 (272.0-279.4)**: +63% on round 1, **2.2x** the start |

All four checksums identical in all 35 runs. Per chunk on one of 6 threads,
start -> now: noise 12.0 -> 4.5 ms, surface 16.7 -> 6.4, structures 5.4 -> 1.1,
biomes 6.3 -> 4.1, decoration 6.3 -> 4.4.

**Ideas left after round 2:**
- Biomes: still 4.1 ms on 6 threads against ~3.7 ms on one; the rest of the
  biome-chunk waiting.
- Noise: spline volumes compute coordinates only where needed (<= ~2% of CPU);
  the aquifer's `compute_substance_below` neighbourhood search (~2%); carver
  biome checks from cached biome chunks (~1%, only if single-point and bulk
  climate agree, which the checksums would show).
- Structures: one trial_chambers start (82-89 ms, chunk 14 5) plus a thread
  waiting on it is most of the stage; template NBT parsing is a third of it.
- Decoration: no single item; ore loop hoists, a tag-lookup memo, a cheaper
  uniform-section test and palette skips together come to ~0.7% overall.

### Round 3 (Linux laptop, 2026-10-27)

Same loop and rules on `1278802c` (round 2 plus far terrain), branches
`ttm/cg3-{noise,surface,deco}`, combined as `ttm/cg3-combined`. The laptop ran
in power-saver throughout (one ~15:31-15:43 performance window; the A/B run in
it was redone), so absolute numbers are lower than round 2's table. All three
agents stopped early, each finding nothing left in its area likely to clear 2%:
4 kept / 13 discarded (noise), 3 / 7 (surface), 3 / 6 (decoration). Every kept
commit was rebuilt and is byte-identical to the binary its A/B measured.

| Commit | Change | Interleaved finished chunks/s |
|---|---|---|
| `2b83c652` | Noise fill by 16-block column runs: all-positive runs are the default block, all-non-positive runs above the aquifer's skip height and sea level are air, only mixed runs ask the aquifer; all-air sections not written | 224.15 -> 239.05 |
| `37d71e97` | Carving mask: an origin's biome is not sampled when no carver of any possible biome's carver lists passes its start test there (exact) | 239.6 -> 246.1 |
| `7ddd0446` | Per-quart-column climate search hint; above the air height the fill computes densities for their sign only (no 2 MB beardifier buffer, all-negative cells written as 0) | 242.85 -> 254.05 |
| `23c19a3e` | Spline volumes evaluated from flat arrays instead of per-node closures returning `Result` (same coordinate sampling order) | 253.35 -> 259.35 |
| `e8acd99c` | Surface rule results reused down runs of solid blocks, unchanged spans stepped over; biome cache takes scalar x, y, z (removes a store-forwarding stall) | 226.2 -> 231.85 |
| `7e151553` | Reused surface results guarded by position-only tests (the sulfur caves' 3D noise sent a quarter of blocks down the slow path) | 233.0 -> 239.85 |
| `c497bd19` | One-pass memo lookup, biome corner columns, 3D noise batches stop at the run's bottom | 240.95 -> 247.25 |
| `73907c4d` | Ore blobs by rows, lake spot terms per axis, tag and biome-list lookups as bitsets/memos | 224.6 -> 230.0 |
| `373fb71c` | Jigsaw names as `Arc<str>`, jigsaw orientation table; tree and biome-check upkeep (structures 1.4 -> 0.5 ms) | 228.95 -> 236.45 |
| `37cb991b` | Per-chunk OCEAN_FLOOR_WG bounds answer most ore attempts without column reads; pure ore rules memoized; fewer re-reads; Fx hasher for `ProtoChunkTicks` positions | 234.35 -> 241.45 |

**All builds, laptop power-saver, cgu1, 4-way interleaved, 5 rounds** (median, range):

| Build | Finished chunks/s |
|---|---|
| `9d03d424` (before round 1) | 101.5 (100.4-101.6) |
| `1278802c` (round 2 + far terrain) | 229.3 (227.7-230.4) |
| round 3 without `37cb991b` | 320.8 (315.5-324.8) |
| **round 3 combined** | **324.0 (321.0-326.1)**: **+41%** on round 2, **3.2x** the start |

All four checksums identical in all 20 runs. Per chunk on one of 6 threads,
round 2 -> round 3: noise 5.4 -> 3.3 ms, surface 7.6 -> 5.4, biomes 4.9 -> 4.0,
decoration 5.6 -> 4.6, structures 1.3 -> 0.5, carvers 0.6 -> 0.0. `37cb991b`'s
own evidence was mixed (a reversed run gave +1.2%); combined it is +1.0% with
decoration 5.2 -> 4.6 ms in every run, so it stays.

**Ideas left after round 3** (attempt logs in the laptop's `~/Work/ttm-bench/log3-*.md`):
- Biomes: Perlin `add_volume` (~40%) and the flat spline (~38%) are near their
  limit without a new design, e.g. lazy right-hand sides in bulk min/multiply.
- Noise: the aquifer's `compute_substance_below` is ~30% of the stage, mostly
  cave and underwater blocks where neighbour statuses differ.
- Surface: 3D noise is ~37% of the stage (real evaluations; trimming batches
  cost as much as it saved); block writes ~7.5% (`BlockSection::set` palette
  lookup per write, a bulk write would help); eager ore-vein volumes ~7%
  (~28k samples a chunk, ~500 read).
- Decoration: flat profile; the block-read path (`TerrainBlocks::block`,
  packed storage, `ChunkIndex::get`) is ~12% of the stage. Unmeasured small
  patches `deco-r3-scan-foliage.patch`, `deco-r3-scan-lake.patch` (~0.1 ms).
- Structures: ~80 ms of build CPU plus ~45 ms of waits left in the measured
  phase; preloading pool templates on a waiting thread is worth <= 1%.

## Far terrain speed (branch `ttm/lod-far`, on top of round 2)

Goal from the user: distant terrain appears far faster; it may be barebones
and approximate as long as its shape matches, structures can come later.

| Commit | Change |
|---|---|
| `f10ded8b` | `lod_grid`: one biome per 4x4 far columns (cells up to 64 blocks). Painting asked through the biome zoom, whose strays missed the per-quart memo: ~40% of a tile was climate samples |
| `7480abc9` | `lod_grid`: density only in the surface band found on a 64-block coarse lattice (16-22% of the layers); `SamplerContext::with_planes` keeps y-fixed slice planes (2D noise, 57% of a tile) so the two passes share them |
| `089dcb2e` | `SavedLodSampler::grid`: saved chunks only in tiles up to 4 blocks a column; region chunk lists kept 10 s; decoded saved chunks shared by the sampling threads (1024, 30 s) |
| `5566da17` | Up to hardware threads - 2 far-sampling threads once the near terrain has loaded (not measured on its own) |

- `lod_grid_bench` (TTM seed 0, 25 tiles a level), ms per tile, before ->
  after: level 2 47 -> 17-20, level 3 61 -> 16-18, level 4 62 -> 17-22,
  level 6 56 -> 34-37. Surface and water heights are identical (the bench's
  `heights` checksum); surface blocks identical at levels 2-4; biomes differ
  in 0.1% / 6% / 16% of level 2 / 3 / 4 columns (coarser biome cells).
- In game the limit was elsewhere: the saved-chunk overlay decoded up to 256
  saved chunks per far tile (11.7k saved chunks in the To The Max save). On a
  copy of that save (render 6, 6291 far tiles, nothing baked), far tiles at
  60 s: ~1450 before the branch, ~3400 after; the whole view in ~2 minutes
  instead of over 15. With the overlay off entirely it was 61 s: tiles of 4
  blocks a column still decode saved chunks (`SAVED_LOD_STRIDE` in save.rs).
- How to measure: `ar/fill.py`-style runs (fresh world copy without
  `horizon_lod/`, the player's `options.txt` and `horizon-graphics.txt`,
  `horizon-ctl launch --offscreen`, poll `status` -> `lod.sampled_target_tiles`).
  Launch with no output pipes: the game inherits horizon-ctl's and a capturing
  caller then waits for the game to exit.
- Next ideas: lower `SAVED_LOD_STRIDE` to 2 (saved terrain within ~512 blocks
  only), keep saved-chunk summaries in the bake keyed by their timestamp (so
  later sessions skip decoding), and the far view's frame rate drops to ~30
  with ~10M far quads resident.

## Profile of the To The Max world (seed 0, 6 threads, i7 4c/8t)

- One-time **stronghold ring search: ~25 CPU-seconds per world load**. Chunk
  generation waits on it (`StructureState::rings` in `generator_structures.rs`).
  Now only on a world's first load, or after the game, its resources or the
  data packs change (5 above).
- Steady state ~46 finished chunks/s. Per chunk on one thread: surface rules
  71 ms (~60%), noise 23 ms, decoration 13 ms, biomes 10 ms, structures 9 ms,
  carvers <1 ms.
- Climate biome samples cost ~80 µs each in this world (TTM's climate density
  functions); the stronghold search does roughly 300k of them.

## To-do

1. ~~Cache stronghold ring positions with the world~~: done (5 above). Not yet
   seen in the game itself: the log should say "stronghold positions kept in"
   on the first load and "read from" on the next.
2. **Chunk generation speed**: round 1 done, 10 kept commits, see "Autoresearch
   loop" (+37% on the Windows PC, then +37% more on the laptop); round 2 kept
   13 more, 2.2x the round-1 start on the laptop; round 3 kept 10 more, +41%,
   3.2x the start.
3. **Confirm the mouse fix** in real play on Windows (alt-tab, notifications).
4. **Clean timing of baked LODs**: launch the same world twice with nothing else
   running and log `lod.sampled_target_tiles` / `target_tiles` from
   `horizon-ctl status` over time. A first attempt showed the second launch at
   949/1849 tiles in under a minute vs ~95 s on the first.
5. Decide whether the wide-cell LOD approximations (item 2 above) look right in
   game, and whether to keep both benchmark examples.
6. Run `python3 tools/pr_ready.py` (harnesses need `.local/`) before merging to
   `main`; split into smaller commits/PRs if wanted (AGENTS.md asks for < ~500
   lines per PR).

## The Windows setup this was tested on

- Game: `E:\Horizon To The Max\` (moved from `%APPDATA%\horizon-ttm\`, where
  the Claude desktop app redirected AppData writes into its own package
  storage, then off the C: drive) (`play.bat`, `new-world.bat`, saves, options, `shaderpacks/`;
  `play.bat` opens the title screen;
  the previous client is kept as `horizon-client.previous.exe`). Desktop
  shortcut "Horizon To The Max" runs `play.bat`. `play.bat` uses the prepared
  resources of any installed Horizon launcher version
  (`%APPDATA%\horizon-launcher\versions\<v>\.local\horizon-client\26.3`).
- Build: `cargo +stable-x86_64-pc-windows-msvc build --profile fast -p horizon-client --bin horizon-client`
  (default toolchain there is 1.83, too old; the repo needs 1.85+). Set
  `CARGO_TARGET_DIR` outside the checkout (e.g. `%LOCALAPPDATA%\horizon-ttm-build`)
  if builds fail with "os error 32" (a folder watcher locking new files), and
  `git config core.longpaths true` before checking out on Windows.
- The dev build keeps `saves/`, `options.txt` and `horizon-graphics.txt` in its
  working directory: start it from its home folder (as `play.bat` does).
