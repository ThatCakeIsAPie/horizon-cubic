# Overnight 2026-11-06/07: chunk loading, three lanes

All on branch `research/far-fill`. That branch has lane B (`research/cube-load`) merged in up to its final commit, 34daa578e2. None of it is on `main` yet.

## Results

All numbers are from the desktop (i7, 4 cores / 8 threads, GTX 1070 Ti), unless noted. "Before" is 743db86f57, the code at the start of the night.

| What | Before | After | Change |
|---|---|---|---|
| **Far view complete after a teleport** (y 300, High; median of 12 hops) | 17.3 s | **about 8.3 s** | **−52%** |
| Far view complete at spawn | about 34 s | about 15 s | −56% |
| Far view in the owner's height-variation-600 world, at spawn | 90–95 s | 52 s | −45% |
| **Near terrain in after a teleport to the surface** (y 80: generated, lit, meshed) | about 7.7 s | **about 4.7 s** | **−39%** |
| Cube generation and population, 1 thread, desktop | 1.14 ms/cube | 0.29 ms/cube | −75% |
| Same, 8 threads (wall time per cube) | 0.38 ms | 0.075 ms (checkerboard order) | −80% |
| Laptop, surface / sky / deep (ms per cube) | 0.73 / 0.20 / 0.30 | 0.20 / 0.013 / 0.12 | −73% / −94% / −61% |

What every number holds to:
- The terrain is **bit-identical**: the hashes of every block are unchanged.
- The far view is unchanged: the same plan size and voxel size, and the sky-cover check found 0 differences.

## The lanes

**Lane A, the far view's fill** (desktop): `research/far-fill/REPORT.md`, `autoresearch_dashboard.md`.
- More far workers, and back to normal priority once the near terrain is in.
- The sky cover is proved per slab, and asks each column's heights in one probe.
- Cache writes moved to a keeper thread.
- Lane B's generator merged in: the far view's finest nodes call the same `generate()`.

**Lane B, cube generation and population** (laptop): `research/cube-load/REPORT.md`.
- 14+ kept changes, all with exact output, including:
  - records encoded once;
  - carvers skipping work that cannot reach the cube;
  - plain-cube and held-block caches;
  - coarse-first uniform proofs.

**Lane C, near terrain in the game** (desktop, after lane B finished): `research/near-fill/`.
- A trace showed the near loading pool half busy. The cube code barely scaled with threads: 8 threads were only about 2× one, because columns made side by side wait on each other's population locks.
- The streaming scheduler now starts a cubic level's columns away from those in flight (`spread_pick`).
- In the benchmark that gave +20% throughput at 8 threads; in the game, near-in went from 5.25 s to 4.7 s.

## Things to know before merging

- **Memory:**
  - Lane B's population cache keeps up to 6,144 in-flight cubes as 16 KB block arrays. That is up to about 100 MB in the worst case, against a few KB each as packed sections. Its report names the two commits to revisit if memory matters.
  - The far view's keeper thread queues at most 1,024 nodes, about 25 MB.
- **Tests:**
  - `horizon-world`'s `settings::tests::the_preset_survives_a_json_round_trip` fails in optimized (`fast`) builds, on unchanged main too, on both Windows and Linux. It is not from tonight.
  - `cubic::store::tests::reads_go_on_while_uniform_cubes_persist` failed once under load and passed 5 of 5 alone; it is a timing test.
  - The full test pass on this branch: see the end of this file.
- **Tools added:**
  - `tools/far_fill_eval.py`, `tools/cube_load_bench.py` (`--spec`, `--settings`, `--baseline`, `HORIZON_CUBE_BENCH_ORDER`);
  - `tools/cubic_far_bench.py` (`--y`, near times, `--trace-near`);
  - `HORIZON_FAR_*` and `HORIZON_NEAR_*` overrides, with defaults unchanged;
  - `HORIZON_FAR_PROFILE=1` (worker profile plus streaming job CPU, every 5 s).
- **The fill is now CPU-bound** with every thread busy. What remains is the minimum exact work, plus some lock contention left in the cube code, where 4 threads give 2.55× instead of 4×.

## Next

- **Contention left in the cube code:** the global population cache and flags mutexes, and the store's pending map. Sharding them is the obvious next step.
- **The far view's finest levels (0–1)** are still made on the CPU with the full generator. Making them on the GPU with the exact replacers would be the architectural step.
- **Far views of the player's builds,** occlusion-driven refinement, and the rest of the plan's later work (docs/cubic-v2-plan.md, Order of work 6.5).

## Full test pass on this branch (5c8b6ac934, debug, 6 crates)

- **5,566 passed.** The one failure was `gfx::end_sky::tests::separate_draws_match_combined_celestial_pixels`, a GPU pixel test of the End sky (code untouched tonight). It passed 3 of 3 on its own, so it is flaky alongside the other GPU tests.
- **The preset JSON round trip passes in this debug pass.** It fails only in the optimized `fast` profile, on unchanged main too.
