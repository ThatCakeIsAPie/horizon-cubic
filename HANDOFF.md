# Handoff: `merge/upstream-2026-10-31` (2026-10-31)

Horizon To The Max (ThatCakeIsAPie/horizon-ttm) brought up to upstream Horizon
(ludovic111/horizon) `main` at `d6e8bd67a`, 7,201 commits past where the fork
branched (`b65e082fd`). The previous handoff (`ttm/bake-lod` and the chunk
generation rounds) is `git show 3ae5bf6c1:HANDOFF.md`.

## Prompt for the next agent

> Pick up branch `merge/upstream-2026-10-31` of ThatCakeIsAPie/horizon-ttm.
> Read `HANDOFF.md` first. Port the surface-stage commits listed under
> "Not ported" onto upstream's specialised rule tree (`SurfaceVariants` in
> `material_condition/rule.rs`) one at a time, measured with
> `world_chunkgen_bench` interleaved against the previous build, kept only
> when finished chunks/s improves with every output checksum unchanged.
> Don't push or merge to `main` without asking me. Follow AGENTS.md.

## How the branch was made

Not a plain merge: about two thirds of our 90 commits had gone upstream as PRs
(level shapes 1-11, data pack worldgen 1-4, upgrade-datapack 1-4, the Data
Packs screen, far tiles, several worldgen perf PRs) and upstream reworked them
since, so merging would have brought the old copies back. Instead the branch
starts at `upstream/main` and re-applies only the work upstream lacks
(`git cherry-pick -x`, so every commit names its original), then records
`origin/main` as merged (`-s ours`) so `main` fast-forwards to it.

### Re-applied (ported where upstream changed underneath)

- **Data-pack biomes**: PR #4682's branch (`fork/feat/datapack-biomes-local`),
  then the network registry, particles/music/ambience, commands and docs
  commits (#3493/#3494's content). The host's biome registry keeps upstream's
  `from_pack` entries and `changed_dimension_type`.
- **Create World upgrades a 26.2 pack on the way in** (`datapack::install`):
  only that part of `3ae5bf6c1`; the Data Packs screen is upstream's.
- **Baked far terrain** (`lod/bake.rs`): grid samples only now. Upstream's
  `lod::cache` keeps detail chunks, so the bake dropped its detail half
  (format 2: a format-1 folder is emptied). Fingerprint takes the level's
  `LevelShape` instead of the old global layout.
- **Far sampling threads**: half the threads less one below 12 threads
  (4c/8t: 3), upstream's count at 12+, more once the near terrain has loaded,
  `SAMPLE_JOBS` 10. The wide-cell approximations are upstream's and apply
  only with Reduced far detail (`farDetail`), not always as before.
- **Windows cursor fix** (`keep_captured_cursor`).
- **Stronghold ring cache** (`ring_cache.rs`): presets the rings before
  upstream's background `search_rings_behind` starts.
- **Chunk generation perf, exact**: noise-fill lattice reuse, ore-vein heights
  and gates (heights read from the buffer at apply time, since upstream's
  `retarget` refills it), lazy richness, 3D noise down the column, biome stage
  top layer, biome search hints, spline flat evaluator, carver origin skip,
  decoration ore rows/columns, jigsaw lazy templates and shared names,
  surface search memo, `#[inline]` packed storage.
- **/locate biome** on the search lattice, typed searches off the game thread
  (heights from `World::bottom_y`/`top_y`).
- Benchmarks: `world_chunkgen_bench`, `lod_grid_bench` (now `--approximate`).

### Upstream bugs fixed on this branch

- `horizon-threads` failed to build on Windows (boot screen #3214 lacked the
  `windows-sys` GDI/window features) and failed clippy (`&mut` font count).
- Noise settings could not read 26.3's `{id, properties}` block states that
  upstream's own `upgrade-datapack` writes: the To The Max world failed to
  load with `Resource("block state")`.
- LAN host had two `S::Attack` arms (anticheat, then #5123's queued-move
  stop); the second never ran. Merged into one.
- `native_hooks` test split its own source on LF; fails on a CRLF checkout.

### Not ported (upstream did the same job another way; re-measure first)

- Surface flat program `fc5a81eaa` and the run memos built on it
  (`e8acd99ce`, `7e1515538`, `c497bd194`), the column snapshot (part of
  `8207c39c4`) and the settled-height write shortcut (part of `e09fb9dda`):
  upstream specialises the rule per gradient band (`SurfaceVariants`) and
  caches each column's blocks.
- Noise fill: concrete aquifer and inline paths (`ca3dfeb89`, `43025047e`),
  per-section writes and whole column runs (`be717823f`, `2b83c6526`), signs
  above the open height (part of `7ddd0446f`, and its lint `2a1366d65`):
  upstream writes solid runs at once (`fill_solid_run`) and skips the aquifer
  for solid and open blocks. Measured: noise is now 7.5 ms a chunk, as on the
  old main, so these are not worth porting.
- Superseded by upstream: surface skip-rewrite `a129fb20b` (upstream's
  `Column::set_block` skips it), structure free-start builds `c2b6961fb`,
  all the level-shape commits (upstream's `LevelShape` model, no global).

## Measured (TTM seed 0, 16x16 chunks, 6 threads, i7 4c/8t, 3 interleaved rounds)

| build | finished chunks/s | surface ms/chunk | noise | structures |
| --- | --- | --- | --- | --- |
| old `main` (`3ae5bf6c1`) | 161-180 | 10.1-10.7 | 7.0-7.5 | 1.1-3.5 |
| upstream + our features, no perf (`ttm-sync/perf-base`) | 70-78 | 37-41 | 15-17 | 5.4-7.4 |
| this branch | 120-123 | 25.0-25.5 | 7.4-7.5 | 1.0-1.1 |

Terrain output `3ad45dd6b7ce4332` and decoration output `d4f45ed1658b457c` are
the same for all three: the branch generates To The Max as the old main did.
Structure starts `30e5f7fd98d30739` match between perf-base and this branch
(the old main's `4d8baae0...` differs through upstream's structure changes).
The remaining gap is the surface stage: the "Not ported" surface series.

Checks: clippy `-D warnings --all-targets` and `cargo test` (lib, bins,
tests) clean for horizon-core, -world, -client-render, -client, -port-pack,
-threads on Windows. Not run: `tools/pr_ready.py` harnesses (need `.local/`),
a real play session with this build.

## To-do

1. Port the surface series onto `SurfaceVariants` (the ~15 ms a chunk gap).
2. Play the To The Max world on this build (the bake's format change re-bakes
   far terrain once; the ring cache; the cursor fix; data-pack biomes).
3. Bring over `ttm/ultimine` and `ttm/big-ore-deposits` (laptop session,
   based on the old main `fe8e063c`): ultimine touches `game.rs`, `chat.rs`,
   `lan_actions.rs` and `remote.rs`, which upstream changed.
4. Offer the upstream fixes above as PRs to ludovic111/horizon.

## Windows setup

As before (`git show 3ae5bf6c1:HANDOFF.md`, "The Windows setup"): build with
`cargo +stable-x86_64-pc-windows-msvc build --profile fast -p horizon-client`
and a `CARGO_TARGET_DIR` outside the checkout. This branch was built in
`D:\ClaudeData\horizon-ttm-sync` with target `D:\ClaudeData\horizon-ttm-sync-target`.
