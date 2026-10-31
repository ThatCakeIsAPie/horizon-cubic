# Handoff: `merge/upstream-2026-10-31` (2026-10-31)

Horizon To The Max (ThatCakeIsAPie/horizon-ttm) brought up to upstream Horizon
(ludovic111/horizon) `main` at `d6e8bd67a`, 7,201 commits past where the fork
branched (`b65e082fd`). The previous handoff (`ttm/bake-lod` and the chunk
generation rounds) is `git show 3ae5bf6c1:HANDOFF.md`.

## Prompt for the next agent

> Pick up branch `merge/upstream-2026-10-31` of ThatCakeIsAPie/horizon-ttm.
> Read `HANDOFF.md` first. The surface series is ported (see "Surface
> stage, ported onto `SurfaceVariants`"); next is the To-do list: play the
> To The Max world on this build, then bring over `ttm/ultimine` and
> `ttm/big-ore-deposits`. Don't push or merge to `main` without asking me.
> Follow AGENTS.md.

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

### Surface stage, ported onto `SurfaceVariants`

One commit each, measured against the commit before it (numbers in each
message): the flat rule program (`fc5a81eaa`), which `SurfaceVariants`
now specialises op by op; the run memo with its step over unchanged spans
(`e8acd99ce`), each entry clamped to the heights its band variant serves
(`SurfaceVariants::rule_down`), so a hit is always for the variant that
made it; memo guards for 3D noise and in-band gradients (`7e1515538`); the
one-pass lookup, biome corner columns and noise batches per run
(`c497bd194`); upstream's column cache filled section by section (the
snapshot part of `8207c39c4`); and the settled-height write shortcut (part
of `e09fb9dda`). The skip over unchanged spans relies on upstream's
`Column::set_block`, which already skips writing a block that is there.

Their other parts were already on the branch (ore-vein volume cuts, the
four-lane Perlin column, the biome bitset). Not ported: `8207c39c4`'s condition test inlined without recursion
(the memo's `test_once` is upstream's `eval::<false>`); try it next if
the surface stage still matters.

### Not ported (upstream did the same job another way; re-measure first)

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
| this branch before the surface series (`d48790847`) | 120-123 | 25.0-25.5 | 7.4-7.5 | 1.0-1.1 |
| this branch (`f091f580a`) | 151.6-163.6 | 11.4-12.8 | 7.7-9.0 | 1.2-1.3 |

The last row was measured in a second session, interleaved with the old main
and `d48790847` (4 rounds, no other load): the old main 162.4-173.0 (surface
10.4-11.5), `d48790847` 107.6-119.8 (surface 25.4-29.4) there.

Terrain output `3ad45dd6b7ce4332` and decoration output `d4f45ed1658b457c` are
the same for every row: the branch generates To The Max as the old main did.
Structure starts `30e5f7fd98d30739` match between perf-base and this branch
(the old main's `4d8baae0...` differs through upstream's structure changes).
The surface stage is now about 1 ms a chunk behind the old main; finished
chunks/s about 5% behind.

Checks: clippy `-D warnings --all-targets` and `cargo test` (lib, bins,
tests) clean for horizon-core, -world, -client-render, -client, -port-pack,
-threads on Windows (after the surface series, re-run for horizon-core and
-world only: those are the only crates it touched). Not run: `tools/pr_ready.py` harnesses (need `.local/`),
a real play session with this build.

## To-do

1. Done: the surface series is ported (15 ms a chunk gained; about 1 ms
   left, see "Surface stage, ported onto `SurfaceVariants`").
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
