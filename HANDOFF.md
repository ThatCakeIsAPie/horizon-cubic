# Handoff: Horizon Cubic (2026-11-06)

**ThatCakeIsAPie/horizon-cubic** is the Cubic Chunks fork of the fork:

```
horizon-rust/horizon  (upstream Horizon)
  └─ ThatCakeIsAPie/horizon-ttm    (Horizon To The Max: ultimine, ring cache, far terrain bake, ...)
       └─ ThatCakeIsAPie/horizon-cubic   (this repo: vertically unbounded levels)
```

Both GitHub repos are private, and GitHub can't fork a repo into the account
that owns it, so this is a plain repository with `ttm` and `upstream` as
remotes (fetch only: their push URLs are disabled here).

To take in TTM's and upstream's changes:
- `git fetch ttm upstream`, then `git merge ttm/main`. When TTM is behind upstream, also merge
  `upstream/main`.
- Keep this file and `docs/cubic-chunks.md` as this repo has them.

The port's plan and progress are in `docs/cubic-v2-plan.md`; `docs/cubic-chunks.md` has the v1 design notes.

This history starts from horizon-ttm's `merge/upstream-2026-11-05`: TTM brought up to upstream
`c4e4800d4` on 2026-11-05. The rest of this file is that sync's handoff. The previous TTM handoff
is `git show 58fc487fa:HANDOFF.md`.

## Prompt for the next agent

> Pick up `main` of ThatCakeIsAPie/horizon-cubic. Read `HANDOFF.md`, then `docs/cubic-v2-plan.md`,
> which is the plan, the budgets and what is done. Cubic Chunks v2 is playable in single player.
> The owner's order is speed first, then detail. The far view was reworked on 2026-11-06 (see "Far view
> rework" below). The next steps are in `docs/cubic-v2-plan.md` › "Order of work" item 6:
> - the remaining measurements;
> - far views of the player's builds;
> - occlusion-driven refinement;
> - far block light;
> - structures filling in;
> - the thin-feature flex.
> Then run autoresearch on fill rate. Follow AGENTS.md. Run tests once after a batch of changes, not
> after each one.

## Cubic Chunks v2 (2026-11-06)

### How to play

- **Folder:** `E:\Horizon Cubic` holds `play.bat`, `README.txt` and this build's binaries
  (`--profile fast`).
- **Create a world:** Create New World › World tab › Cubic Chunks: Custom Cubic, Hybrid or
  Flat Cubic. Customize… opens CubicWorldGen's five pages and the Presets JSON box.
  `--world NAME --cubic` makes a Custom world.
- **Settings:** Video Settings › Distance has Vertical Distance (cubes), LOD Distance (top step
  Whole World) and Far Terrain on GPU.
- **F3:** shows `Cubic:` and `Far:` lines.

### What v2 is

The plan is `docs/cubic-v2-plan.md`. The parts, and the branches they came from (merged):

| Branch | What |
|---|---|
| `cubic/shift-client` | The client lays out a new window in place. Kept sections keep their meshes. |
| `cubic/slide` | The window slides by cube layers, from a prefetch ring. Clients get `horizon:cubes` for the entering layer only. `horizon:cubic_view` carries the vertical distance. |
| `cubic/light` | A slide lights only its entering layers, plus what they reach. |
| `cubic/store2` | Uniform cubes in a sidecar, about 6 bytes each. Entities and ticks are stored per cube. |
| `cubic/cwg` | The CubicWorldGen port (`horizon-world/src/cubic/custom/`): settings v9 JSON, density, biome blend, replacers, caves and ravines, ores, lakes, springs, flat. |
| `cubic/bounds` | A proven 3D density bound: uniform cubes skip the lattice. |
| `cubic/integrate` | The world types; CWG population (populated flags, striped locks); vegetation from 26.3's biome features; the climate biome adapter; the far sampler. The v1 deep/sky generator (floating islands) is deleted. |
| `cubic/customize` | The world-type cycle, the Customize screen and its preview, `cubicgen.json`. |
| `cubic/far2` | The far-terrain node cache, plan hysteresis and overhang shading. |
| `cubic/gpu-lod`, `gpu-biome`, `gpu-mesh`, `gpu-descent` | Far nodes generated, biome-sampled, meshed, culled and drawn on the GPU. Also Whole World (octree root level 22), `CubeLodStats`, and the sparse descent: uniform, or right-sized, or subdivide. |

Lead commits on main:
- the F3 lines;
- `horizon-ctl status` `cube_lod`;
- `tools/cubic_far_bench.py`;
- the descent wired for Custom;
- batched GPU requests from the far workers;
- the test-pass fixes.

### Converting a vanilla world (`cubic/convert`)

`crates/horizon-client/src/cubic_convert.rs` copies a vanilla world to a cubic Hybrid world and leaves
the original alone. Design, what is and is not carried over, and the options are in
`docs/cubic-chunks.md` ("Converting a world").
- CLI: `horizon-client --game-dir DIR --convert-cubic "<world>" [--keep-bedrock]` writes
  `saves/<world> (Cubic)`.
- UI: Edit World › Convert to Cubic Chunks… (confirm, background run with a progress line, the copy is
  selected in the list).
- Record format: `cubic::vanilla_column_records` is the one place a vanilla column becomes cube records;
  `finish_vanilla` and the converter both use it.
- Tests: `cargo test -p horizon-client-lib --lib cubic_convert` (needs `HORIZON_TEST_RESOURCES`).

### How it was tested

- **Test pass:** one full `cargo test` of horizon-core, -world, -protocol, -client-render,
  -client-lib and -client, with `HORIZON_TEST_RESOURCES` set to the launcher's 26.3 resources.
  - First run: 5,491 passed, 13 failed, 175 ignored.
  - Of the 13 failures, three were bugs, all fixed: `saved_ys` missed unflushed uniform cubes, `in_world`'s edge was inclusive, and the far mesher's sky test asked whether air is opaque. The other ten were stale tests.
  - After the fixes, the world, render and client-lib lib targets pass with 0 failures. The other targets passed in the first run.
- **Smoke run** (`D:\ClaudeData\cubic-play\smoke.sh`, offscreen, scratch home):
  - A Custom world spawned on land at y 75–81: savanna, rivers, acacia trees, grass and flowers.
  - Climbing from 80 to 600 and diving to −600 slid the window 2 layers at a time, with jumps for long moves.
  - 116–120 fps, no errors in the log.
- **Far-fill benchmark** (`tools/cubic_far_bench.py --hops 2`, Whole World):
  - First benchmark run, before the batching fix: spawn 30.1 s, hops 33.2 s and 18.4 s (**mean 25.8 s**), about 7,500 planned nodes, fps min 100 and mean 119.
  - After `cubic/store-reads` (an in-memory index of saved cubes, per-region locks, and fine nodes from whole generated cubes): **2.1 s mean per hop** (2.4 / 2.6 / 1.4 s) and 12.6 s at spawn, at 93–117 fps. The owner's reference, the Meridian LOD mod, took about 20 s on a top CPU.

### Far view rework (2026-11-06, `cubic/far-detail` + `cubic/far-surface`, merged)

The design, the measurements and the owner's decisions are in `docs/cubic-v2-plan.md`:
- "What the Meridian video shows";
- "Far view: the owner's decisions and later work";
- "Detail by size on screen, coverage first".

What was built:
- **Detail by size on screen.** The FOV and the spyglass are taken into account.
- **Cubic Far Detail** in Video Settings › Distance: Low, Normal, **High (default)**, Ultra, Extreme.
- **A node budget.** Past it, the plan coarsens what is smallest on screen first.
- **A coarse-first coverage layer,** and refinement of what is on screen first.
- **Layered height-map tiles** for the coarsest levels in Custom worlds (`gfx/surface_tile.rs`):
  - exact block heights, with up to 4 spans per column;
  - complex tiles go 3D sooner.
- **Far faces lit from the cell in front of them.** Tall walls used to draw near-black.
- **The fog follows the far view** in cubic worlds.

Tools:
- `tools/cubic_far_bench.py` reports `cover_s` and `far_fill_s`, with `--detail` and `--budget`.
- `tools/cubic_far_check.py` covers `--hitch`, `--views`, `--tall`, `--spyglass` and `--cavern`.

Tested:
- **One full test pass:** every target passed except one tile-agent test with a threshold too strict for coarse columns. It was fixed and rerun.
- **Measured at High with tiles:** a teleport is covered in 0.18 s and complete in 19 s, at 120 fps, with a 48 ms longest frame. The spyglass refines what it looks at. Caverns at y −2,000 show no holes.

The owner's first play test, fixed: the plan's "The owner's first play test of the rework" lists each issue with its cause. In short:
- sea water in coarse voxels;
- the gap at the render distance's edge;
- dark stripes on far cliffs;
- z-fighting where coarse terrain waits over finer;
- Ultra's plan swinging with the arena estimate;
- the chunk fade-in, now dithered.

`tools/cubic_far_check.py` gained `--spyglass-at Y:PITCH` and `--watch S` (plan swings while the camera is still).

### Known gaps

- **Far view, not measured yet:**
  - `SURFACE_LEVEL` 10/14;
  - Low/Normal/Extreme with tiles;
  - true 1 px.
- **Ultra is slower with tiles than without** (72 s against 48 s to complete, 103 fps against 120). The first suspect is tile draw compaction.
- **The spyglass refines at the fill rate** (2,600–3,500 nodes/s). At y 10,000 a zoom wants about 90,000 new nodes, so it takes tens of seconds. The fill-rate work is the lever.
- **Far terrain doesn't show the player's builds** beyond the disk cache's limits; see the plan.
- **Vertical shards (branch `cubic/shards`).** LAN guests are grouped into shards by their vertical ranges (`cubic_shards.rs`); see `docs/cubic-v2-plan.md`, "Residency", "As built". Not done: vehicles carried across shards, mod API calls (`native_*` by dimension name) reach the shard in place only, and no trial on a long-running dedicated server yet.
- **No structures in Custom Cubic,** as in CubicWorldGen. The structure toggles are marked `*`.
- **Not generated yet:** snow and ice layers, dungeons and Bryce pillars.
- **Seen from inside solid rock in spectator,** the far terrain's rock shells around the loaded box show as walls.
- **The slide's relight runs on the tick thread.** A cave-heavy slide may take more than a tick.
- **The Customize preview and layout** have not been looked at in the game.
- **Unmeasured:** cube generate and populate costs (`CubicLevel::population_stats()`).

## What the TTM sync brought

1. `142c137d1` **Merge of upstream main.** Where the fork and upstream did the same job, upstream's
   version was kept. The commit message lists each case:
   - data-pack biomes: upstream's one table, `world_biomes!`;
   - the surface rule engine and the ore feature: upstream's, so the fork's flat-program and run-memo
     surface series is dropped;
   - far terrain threads.

   Kept from the fork:
   - the stronghold ring cache;
   - `/locate biome` off the game thread;
   - the Windows cursor fix;
   - Create World's 26.2 pack upgrade;
   - the far terrain bake;
   - lattice reuse;
   - biome search hints;
   - the decoration feature bitsets;
   - lazy jigsaw templates.
2. `a123999ca` **Pack biomes' ambient particles, music and sounds**, re-ported onto upstream's biome
   table. Upstream had no equivalent.
3. `312d69a40` **Ultimine** (`ttm/ultimine-v2`), ported onto upstream's Extras page. It gets a lang-keyed
   header and caption (en_us, pt_br) and reads the Sneak mapping.
4. `033c5fdbc` **Big Ore Deposits** (`tools/ttm/big-ore-deposits/make_pack.py`).
5. **LAN host fix.** The merge kept the fork's old early `break` in the Attack arm beside upstream's
   `held` flag. It broke `an_attacks_swing_goes_with_it_past_the_move_behind`, and `hub.rs` is now
   upstream's.

Not brought over, as the owner asked (2026-11-05):
- `ttm/mekanism` (112 commits on the old base `fe8e063c`). It isn't finished.
  - Notes for whoever ports it: its hooks target `containers.rs`, `world.rs`, `hud.rs` and `items.rs` in
    `horizon-client`. Those now live in `horizon-client-render` and `horizon-items`.
  - So `Tile`, `LevelState` and the `data()` gate need to sit in `horizon-client-render`, and
    `impl Stack for ItemStack` in `horizon-items`.
- The To The Max chunk generation perf series beyond what merged cleanly. Upstream reworked the same code.

## Checked (Windows, i7 4c/8t, Rust 1.99 MSVC)

- `cargo clippy --workspace --all-targets -- -D warnings`: clean.
- `cargo fmt --all --check`: clean.
- `cargo test -p horizon-core -p horizon-world -p horizon-client-render -p horizon-threads
  -p horizon-client -p horizon-client-lib --lib --bins --tests`: all pass except
  `gfx::end_sky::tests::separate_draws_match_combined_celestial_pixels`.
  - That test draws with the real GPU, and its pixels differ by a few levels on this machine's NVIDIA
    driver.
  - Its code is upstream's, unchanged by this branch.

Not run:
- `tools/pr_ready.py` and the parity harnesses (they need `.local/` in the checkout);
- a play session;
- `world_chunkgen_bench` against the old main.

## Build on this PC

`cargo +stable-x86_64-pc-windows-msvc build --profile fast -p horizon-client`
with `CARGO_TARGET_DIR` outside the checkout.

| Repository | Checkout | Target directory |
|---|---|---|
| horizon-cubic (this) | `D:\ClaudeData\horizon-cubic` | `D:\ClaudeData\horizon-cubic-target` |
| horizon-ttm `merge/upstream-2026-11-05` | `D:\ClaudeData\horizon-ttm-combine` | `D:\ClaudeData\horizon-ttm-sync-target` |

`horizon-client-lib` needs Rust 1.90+. The default toolchain here (1.83 GNU) is too old.
