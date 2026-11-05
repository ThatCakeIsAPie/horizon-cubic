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

> Pick up `main` of ThatCakeIsAPie/horizon-cubic. Read `HANDOFF.md`, then
> `docs/cubic-chunks.md`: phase 1 is done. Start phase 2:
> - the cube codec: pull the section encoder out of `save.rs`'s `encode_with`, keeping vanilla's
>   output byte-identical;
> - a cube noise generator, with the stacked-cubes oracle.
>
> Ask me before pushing. Follow AGENTS.md.

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
