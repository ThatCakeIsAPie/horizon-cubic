# Handoff: `ttm/bake-lod` (2026-10-24)

Work done on the Windows PC to play the JJThunder To The Max world on Horizon, and
to make its far terrain (LODs) and chunk loading faster. Branch: `ttm/bake-lod`,
off `main` at `3051dda`.

## Prompt for the next agent

> Pick up the Horizon To The Max work on branch `ttm/bake-lod` of
> ThatCakeIsAPie/horizon-ttm. Read `HANDOFF.md` at the repo root first: it says
> what's done, how it was measured, and the to-do list. Start with to-do item 1
> (cache the stronghold ring positions with the world), measure it with
> `world_chunkgen_bench` before and after, and keep generation output identical
> (the bench prints a structure-start checksum). Follow AGENTS.md. Don't push or
> merge to `main` without asking me.

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
2. **Surface rules: 71 ms/chunk.** Profile `MaterialConditions::try_apply` on
   TTM's rule tree in full chunk generation and find what repeats (noise,
   biome, stone-depth queries). Keep output identical.
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
