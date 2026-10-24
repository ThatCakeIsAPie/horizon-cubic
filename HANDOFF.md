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

**Ideas not tried yet** (from the laptop agents' reports):
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
2. **Chunk generation speed**: loop done, 10 kept commits, see "Autoresearch
   loop" (+37% on the Windows PC, then +37% more on the laptop).
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

- Game: `%APPDATA%\horizon-ttm\` (`play.bat`, `new-world.bat`, saves, options;
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
