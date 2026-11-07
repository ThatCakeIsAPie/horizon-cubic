# Autoresearch lane B: near cube loading (generation + population)

Overnight loop for the laptop (Framework, Ryzen 5 7640U, 7 GB RAM, Arch/Omarchy). The owner reads the dashboard in the morning. Lane A, the far view's fill on the desktop GPU, is separate: never touch its files or its branch.

## System context

Horizon Cubic is Minecraft 26.3 in Rust with Cubic Chunks. When a player moves or teleports, the server makes the cubes around them. Each cube is 16 × 16 × 16 blocks.
- **Terrain:** the CubicWorldGen "Custom Cubic" generator gives density noise, then surface replacers, then carvers.
- **Population:** CubicWorldGen populates each cube's 2 × 2 × 2 region with ores, lakes, vegetation and springs. Its eight neighbours' populations must have run before a cube is final.
- **Glue:** `CubicLevel::make_cube` (crates/horizon-client/src/cubic.rs) generates, populates and reads a cube back.

The benchmark, the ignored test `cubic::tests::cube_load_bench`, does exactly what a teleport does: it makes the 9 × 9 × 13 cubes around spawn in a fresh world, one thread, fixed seed. It prints the time per cube and an FNV hash of every block of every cube.

## The metric

```
python3 tools/cube_load_bench.py --resources "$RES" --target-dir "$TARGET" --jobs 4 --runs 5
```

- `SCORE:` is the **median ms per cube over 5 runs. Lower is better.**
- `GUARD:` is PASS only if every run's terrain hash equals the baseline's: the terrain must stay **bit-identical**. A faster generator that changes one block is a different generator, not an optimisation.
- `--tests` also runs the generator's unit tests: `cubic` in horizon-client-lib, and `cubic` in horizon-world.
- The tool builds first (`cargo test -p horizon-client-lib --lib --profile fast --no-run`), then runs the test binary 5 times. A build takes minutes; each run takes about 2 s here.

Set up the checkout first:
- Clone: `git clone https://github.com/ThatCakeIsAPie/horizon-cubic ~/Work/horizon-cubic`, then `git switch -c research/cube-load` from `main`.
- `RES=~/.local/share/horizon-launcher/versions/v0.2.27/.local/horizon-client/26.3`.
- `TARGET=~/Work/horizon-cubic-target`. Never put it under /tmp: on this machine /tmp is a 3.6 GB RAM disk.
- The tool keeps its scratch levels in `research/cube-load/tmp/`, which is gitignored.

**Baseline.** On the unchanged code, run the command with `--write-baseline` (this machine's hash; Linux's libm may differ from the desktop's). Then run it twice more without the flag. Record the three medians.
- **Min-delta:** the larger of 3% and 3 × (max − min of the three medians). Write it in the JSONL config line.

## What you may change

- `crates/horizon-world/src/cubic/custom/`: `density.rs`, `replacers.rs`, `carvers.rs`, `populate.rs`, `vegetation.rs`, `biomes.rs`, `climate_biomes.rs`, `rng.rs`, `bounds.rs`, and `mod.rs`'s generation and population paths.
- `crates/horizon-client/src/cubic_gen.rs`: the population scheduling, its locks, and what it reads and writes.
- `crates/horizon-client/src/cubic.rs`: `make_cube`, `read_cube`, `ensure_populated_around`, `populate`, `terrain`, `terrain_section`, and the record encode/decode on that path. Do not touch the tests module, and especially not `cube_load_bench`.
- New private helpers in those files.

## Frozen

- The benchmark test and `tools/cube_load_bench.py`. Also `--spec`: it stays at the default, one thread.
- `settings.rs`, the settings' defaults and their meaning (CubicWorldGen's JSON).
- The save format: what a record holds on disk.
- Everything outside the files above: the far view (`lod/`, `gfx/`), the renderer, networking, the client.
- `Cargo.toml` and its profiles, dependencies and features. No new crates.

## The loop

One idea per iteration, smallest change that tests it:
1. Read `research/cube-load/autoresearch.jsonl` and `autoresearch_dashboard.md` to know what has been tried.
2. Make the change. Before building, `pgrep horizon-client`. If the owner is playing, wait and check again in a few minutes. Never build while a game client runs.
3. Run the metric (5 runs).
4. **Crash, or GUARD: FAIL:** status `crash` or `guard_fail`. Revert with `git checkout -- <the files you changed>`.
5. **SCORE beats the best by at least min-delta:** run the metric again as a confirmation. If the confirmation also beats it, run it once more with `--tests`.
   - All pass: `git commit` with a message saying what changed and the scores, then `git push origin research/cube-load`. Status `keep`; the confirmation's median is the new best.
   - Otherwise: status `discard` (noise), and revert.
6. **Anything else:** status `discard`, and revert.
7. Append the result line to `autoresearch.jsonl`, regenerate `autoresearch_dashboard.md`, and commit both (with the keep, or on their own every few iterations), then push.

The JSONL config header (line 0) holds:
- `constrained_files`;
- the eval command;
- `"metric": "ms_per_cube (lower is better)"`;
- `"guard": "terrain hash equals baseline; unit tests"`;
- the baseline;
- the min-delta.

Each result line has: iteration, commit, score, delta, guard, status, description, timestamp.

## Strategy, in order

1. **Measure where the time goes.**
   - The bench prints the generate and populate split.
   - If `perf` is available (`perf record -g` on the test binary with `--exact cubic::tests::cube_load_bench --ignored`), use it. Otherwise time sections with temporary `Instant`s and remove them before keeping anything.
   - Note the top 5 hot spots in the dashboard.
2. **Quick wins:**
   - allocations in hot loops (Vec/HashMap per cube, per column, per block): reuse buffers;
   - repeated lookups: block state by name, biome per block, settings parsed per call;
   - `Section::set` per block where a whole cube could be filled from an array (`PalettedContainer` writes);
   - an encode-then-decode round trip `make_cube` may not need.
3. **Noise and density:**
   - octave loops, Perlin gradients, interpolation grid sizes. Where CubicWorldGen samples a coarse grid and interpolates, keep exactly the same arithmetic order: f32 and f64 results must stay bit-identical.
   - Hoist per-column work out of per-block loops.
4. **Population:**
   - ore and vein placement, lakes, vegetation;
   - avoid repeated neighbour-cube reads;
   - cache height or surface searches;
   - the striped locks and the bookkeeping per population.
5. **Storage on the path:** what `populate` writes and `read` reads back per cube. Batch it, or keep it in memory where the code allows without changing the saved format.

If 10 iterations in a row are discarded, move to the next item. After the list, go back to the biggest hot spot with a new angle. Keep going until the owner stops you in the morning. Stop early only if every idea you can find is spent; then write the final report.

## Do not

- Change the terrain, even slightly. No skipping populations, approximating noise, reordering float operations, or using fewer octaves.
- Change the bench, its spec, the tool, or the baseline file (except `--write-baseline` once at the start).
- Run two cargo builds at once, or any build while `horizon-client` runs. This machine has 7 GB; at about 90% of RAM + swap, systemd-oomd kills the terminal and every agent with it. `-j4`. Keep /tmp clean.
- Launch the game.
- Use sleep inhibitors or systemd-inhibit.
- Push to `main` or force-push. Only `research/cube-load`.
- Leave timing prints or debug code in a kept commit.

## Subagents

The owner wants as many working as the machine allows. Here that means:
- **One** builder/bencher: you.
- Up to **2 read-only research subagents** at a time. They study the hot code and propose the next ideas, ranked, with the exact change. They never build, run tests or edit files.

Give each subagent a narrow area (for example "density.rs's octave loop" or "populate.rs's ore placement"), and feed their best ideas into the loop.

## Morning report

When stopped or done, write `research/cube-load/REPORT.md`:
- the baseline and best ms per cube, with the percentage;
- the kept changes, each with its effect;
- what did not work, and why;
- where the time goes now;
- what to try next.

Then commit and push.
