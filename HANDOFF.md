# Handoff: `merge/upstream-2026-11-05` (2026-11-05)

The fork (ThatCakeIsAPie/horizon-ttm) brought up to upstream Horizon
(horizon-rust/horizon, formerly ludovic111/horizon) `main` at `c4e4800d4`:
8,851 commits since the last sync (`d6e8bd67a`, 2026-10-31). The previous
handoff (the 2026-10-31 sync and the surface series) is
`git show 58fc487fa:HANDOFF.md`.

Cubic Chunks work continues on `feat/cubic-chunks`, branched from here:
read `docs/cubic-chunks.md` there.

## Prompt for the next agent

> Pick up branch `feat/cubic-chunks` of ThatCakeIsAPie/horizon-ttm (based on
> `merge/upstream-2026-11-05`). Read `HANDOFF.md`, then
> `docs/cubic-chunks.md`: phase 1 is done; start phase 2 (the cube codec,
> pulling the section encoder out of `save.rs`'s `encode_with` with vanilla
> output byte-identical, and a cube noise generator with the stacked-cubes
> oracle). Don't push or merge to `main` without asking me. Follow AGENTS.md.

## What is on this branch

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

| Branch | Worktree | Target directory |
|---|---|---|
| This branch | `D:\ClaudeData\horizon-ttm-combine` | `D:\ClaudeData\horizon-ttm-sync-target` |
| `feat/cubic-chunks` | `D:\ClaudeData\horizon-cubic` | `D:\ClaudeData\horizon-cubic-target` |

`horizon-client-lib` needs Rust 1.90+. The default toolchain here (1.83 GNU) is too old.
