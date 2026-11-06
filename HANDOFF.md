# Handoff: Horizon Cubic (2026-11-05)

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

The port's plan and progress are in `docs/cubic-chunks.md`.

This history starts from horizon-ttm's `merge/upstream-2026-11-05`: TTM brought up to upstream
`c4e4800d4` on 2026-11-05. The rest of this file is that sync's handoff. The previous TTM handoff
is `git show 58fc487fa:HANDOFF.md`.

## Prompt for the next agent

> Pick up `main` of ThatCakeIsAPie/horizon-cubic. Read `HANDOFF.md`, then `docs/cubic-chunks.md`
> ("v1 architecture"). Cubic Chunks v1 is playable in single player. Next is the "Next" list
> below, starting with keeping loaded sections across a window move instead of reloading them.
> Test in the real game with `horizon-ctl` as described under "How it was tested". Ask me
> before pushing. Follow AGENTS.md.

## Cubic Chunks v1 (2026-11-05)

### How to play

- **Folder:** `E:\Horizon Cubic` holds `play.bat`, a README and this build's binaries, built
  with `--profile fast`. `play.bat` uses the Horizon launcher's prepared resources.
- **New world:** Create New World, then on the World tab turn **Cubic Chunks** ON. Or run
  `play.bat --world NAME --cubic [--creative] [--seed N]`.
- **What a cubic world is:**
  - Vanilla terrain between −64 and 320, with no bedrock floor.
  - Below that, deepslate caves with ores, lava, geodes and obsidian pockets.
  - Above, open sky with floating islands in bands every 1000 blocks from 600.
  - The range is ±8.4 M blocks.
  - The cube octree draws the far terrain.

### How it works

See `docs/cubic-chunks.md`, "v1 architecture". The code:

| File | What it holds |
|---|---|
| `horizon-world/src/cubic/` | The cube range, `CubeStore` (`region3d/y<cy>/r.x.z.mca`), `CubeWindow`, `ColumnTops`, `generator.rs` (deep and sky) |
| `horizon-client/src/cubic.rs` | `CubicLevel` (cube store with queued writes, clean hashes, parked entities, a streaming factory) and `CubicTerrain`, the `ChunkGenerator` around vanilla. A cube record is a vanilla chunk record holding one section |
| `horizon-client/src/game/cubic.rs` | The window follows the first player, saving without waiting and keeping the generator (`renew_level_in_place`) |
| `horizon-client/src/cubic_lod.rs` | The far terrain's sampler |
| `horizon-client-render/src/lod/cube.rs`, `lod/mesh.rs` (`mesh_cube`), `gfx/lod.rs` | The octree producer and renderer |
| `horizon-protocol` (`buf.rs`, `cubic.rs`), `lan/hub.rs`, `game/remote*.rs` | Wide positions and the `horizon:cubic` / `horizon:cubic_window` payloads |

How the far terrain's sampler builds nodes:
- Levels 0 and 1 come from saved cubes, downsampled.
- Other nodes use the vanilla far sampler for −64..320 and the cube generator outside it.

Engine changes outside the cubic modules:
- `LevelShape::cubic` and `ColumnLayout::window`.
- Chunk heightmaps stored as `u16` offsets from the chunk's `min_y`.
- `Chunk::sky_covered`, which is used in light seeding and in `lowest_sky_source`.
- Section `Y` saved as an int when it doesn't fit a byte.
- `World::void_y`.
- The Create World switch, id 36.

Built in parallel by subagents and merged:
- `cubic/cube-generator`, the generator;
- `cubic/lod-octree`, the octree far terrain;
- `cubic/wire`, the network side.

### How it was tested

The game ran offscreen with a control socket, through `horizon-ctl launch --game ... --world W
--cubic --offscreen --wait --resources <launcher's 26.3> --game-dir <scratch>`. The scripts are in
the session scratchpad. They are easy to redo: tp, `block X Y Z`, `status`, `screenshot`.

Checked:
- Spawn on the surface.
- tp to y 400, 3000, −2000 and 1,000,000: terrain, caves, lava, islands and far terrain all
  drew.
- Edits at y −2001 and 999,996 survived a quit and a reopen.
- A climb from 100 to 900 and down to −280 in steps of 40-60 blocks: 10 window moves, each
  reloading in about 2 s with no server stall. The game ran at 120 fps.
- No void damage in survival at −280.
- Create World with the switch on through the menus makes a cubic world.

Tests: `cargo test` for horizon-world, horizon-protocol, horizon-client-render, horizon-client-lib
and horizon-client (lib, tests). All pass except the GPU pixel test
`gfx::end_sky::tests::separate_draws_match_combined_celestial_pixels`, upstream's, which fails on
this machine's driver. clippy `-D warnings` and fmt are clean.

### Known gaps

- **A window move reloads the nearby terrain.** Columns are cleared and sent again; the far
  terrain stays drawn. It happens every ~130-250 blocks of vertical travel.
- **Single player only.**
  - The window follows the first player. LAN guests at other heights get that player's window.
  - A dedicated server has not been tried.
  - Vanilla clients are refused.
- **Terrain at a window edge.** Entities and block entities outside the window wait in the
  save. Mobs at the window edge may see "unloaded" as air for a tick.
- **Not saved in cubic levels:** POI (villager workstations, bells) and block ticks.
- **Far terrain:**
  - Coarse deep nodes are drawn solid.
  - Unvisited vanilla terrain at levels 2+ is undecorated (no trees until near).
  - Islands cast no shadow on what's below.
- Once in the window, the deep generator's lava and caves are not flowed or updated by vanilla
  rules until something changes there.

### Next

1. **Shift loaded sections in place on a window move.** Remap `SectionKey` and keep the overlap,
   on both the server and the client, so moving vertically doesn't reload.
2. **Per-player windows on a server.** This needs true cube residency (phases 3-4 of
   `docs/cubic-chunks.md`).
3. **POI and block ticks in cube records.**
4. **Far terrain:** decorate levels 2+ from saved cubes as they are visited, and add hysteresis
   to the plan.

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
