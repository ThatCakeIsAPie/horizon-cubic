# Autoresearch lane A: the far view's fill rate (GPU)

Overnight loop on the desktop: i7, 4 cores / 8 threads, 24 GB RAM, GTX 1070 Ti, Windows. Lane B (near cube generation) runs on the laptop on `research/cube-load`; never touch its files or branch.

## System context

The far view of a cubic world is a cube octree planned on the CPU (`lod/cube.rs`, `plan_far`, the planner thread). Its nodes are produced like this:
- **Generate:** generated on the GPU in batches (`gfx/cube_gen.rs` + `cube_gen.wgsl`, after a GPU climate pass, `gfx/biome_gen.rs`). They are classified by a GPU sparse descent (`gfx/cube_descent.rs`).
- **Mesh:** meshed on the GPU (`gfx/cube_mesh.rs` + `cube_mesh.wgsl`: a count pass, a readback of 48 bytes a node, arena ranges, an emit pass).
- **Draw:** installed and drawn (`gfx/cube_far.rs`).
- **Surface tiles:** the coarsest levels are height-map tiles (`gfx/surface_tile.rs` + `surface_gen.wgsl`).
- **Workers:** `lod/cube.rs`'s producer workers ask the sampler for nodes in batches (`resident_many`).

Measured fill rate: about 2,600 planned nodes/s, about 1,700 of them generated on the GPU. The plan doc (`docs/cubic-v2-plan.md`, "Detail by size on screen", "Where the GPU far view stands") has the history.

## The metric

```
python tools/far_fill_eval.py --resources C:/Users/Lyle/AppData/Roaming/horizon-launcher/versions/v0.2.46/.local/horizon-client/26.3
```

- It builds the fast profile into `D:/ClaudeData/horizon-cubic-target`, then launches the game offscreen twice (`--launches 2`). Each launch fills at spawn, then fills after 3 teleports 5,000,000 blocks apart, at Cubic Far Detail High.
- `SCORE:` is the **median seconds to complete a fill over the 6 teleports. Lower is better.**
- `GUARD:` keeps quality from being traded for speed. Against `research/far-fill/baseline.json`, each launch must:
  - complete every fill;
  - plan within 3% of the baseline's node count;
  - keep a voxel size no coarser than 2% over the baseline's;
  - keep steady fps within 10%;
  - keep cover time within 20%.
- `--no-build --bin-dir <dir>` benches a build made elsewhere (a builder's worktree).

**Baseline.** On unchanged `research/far-fill`, run with `--write-baseline`, then twice more. Min-delta is the larger of 5% and 3 × the spread of the three scores.

Before a keep, also run the far view's tests, which hold the GPU generator and mesher to their CPU references:

```
cargo +stable-x86_64-pc-windows-msvc test -p horizon-client-render --lib --profile fast --target-dir D:/ClaudeData/horizon-cubic-target -- cube surface lod:: biome
```

## What you may change

How fast the same far view arrives:
- `gfx/cube_gen.rs`: `MAX_BATCH`, `BATCH_NODES`, the pool's size and waits, the GPU thread that merges requests, column inputs, readback.
- `gfx/cube_mesh.rs`, `cube_mesh.wgsl`: `MAX_JOBS`, the count → readback → emit round trip, workgroup layout.
- `gfx/cube_far.rs`: `upload`. How many mesher batches are in flight, when the next starts, install costs.
- `gfx/cube_descent.rs`, `cube_descent.wgsl`: descent cost, output sizes.
- `gfx/biome_gen.rs`, `biome_gen.wgsl`, `cube_gen.wgsl`, `cwg_noise.wgsl`: kernel efficiency, with **identical output**.
- `gfx/surface_tile.rs`, `surface_gen.wgsl`: tile batch sizes and requests.
- `lod/cube.rs`: everything below the plan: the producer workers (count, batch sizes, waits), the queue's order within the same set of nodes, caches, `RELEASE_BATCH`, how and when results are installed.
- Adding `HORIZON_FAR_*` environment overrides for tunables (defaults unchanged) is encouraged: a parameter sweep then needs no rebuild.

## Frozen

- **What is planned:** the screen-space rule, `DETAIL`, the Far Detail steps, `NODE_BUDGET`, `MERGE`, `REFOCUS`, the coverage layer's sizes (`COVER_*`), `SURFACE_LEVEL`, `COMPLEX_LIFT`, `TILE_BUDGET`. The guards check the plan's size and voxel size.
- **What the far terrain looks like:** generation output, materials, lighting, water rules. The GPU-vs-CPU reference tests must pass.
- The tools (`tools/cubic_far_bench.py`, `tools/far_fill_eval.py`), the baseline file, and `Cargo.toml` (profiles, dependencies).
- Everything outside the far view: near terrain, worldgen, networking, gameplay. Lane B's files.

## The loop

1. Read `research/far-fill/autoresearch.jsonl` and `autoresearch_dashboard.md`.
2. Make one change, then run the metric.
3. **Crash, or GUARD: FAIL:** status `crash` or `guard_fail`. Revert with `git checkout -- <files>`.
4. **SCORE beats the best by at least min-delta:** run the metric again (`--no-build`). If that also beats it, run the far view's tests.
   - All pass: commit on `research/far-fill` and `git push origin research/far-fill`. Status `keep`; the new best is the confirmation's score.
   - Otherwise: status `discard`, and revert.
5. **Anything else:** `discard`, and revert.
6. Append to `autoresearch.jsonl` (config line 0: files, eval command, metric, guards, baseline, min-delta). Regenerate `autoresearch_dashboard.md`, and commit both regularly.

## Parallel workers on one GPU

The benchmark needs the GPU and the CPU to itself: a build running beside it skews the fill time.
- **The lead** (the "Cubic Chunks port" session) owns the benchmark, the JSONL and every keep/discard.
- **Up to 2 builder subagents**, each in its own git worktree (`D:/ClaudeData/far-wt-1`, `far-wt-2`) and target dir (`D:/ClaudeData/far-wt-N-target`), branched from the current best. Each:
  - takes an idea from the lead, implements it, and builds with `-j4`;
  - reports the commit and binary folder;
  - then waits for the next idea, rebasing on the best when told.
- **The bench lock:**
  - Before a benchmark, the lead creates `D:/ClaudeData/far-research/bench.lock` and waits until no builder reports a build running.
  - A builder never starts a build while the lock exists.
  - The lead deletes the lock after the benchmark.
- **Read-only research subagents** (up to 2) study hot paths and propose ranked, exact changes; they never build or edit.

## Strategy, in order

1. **Measure first:**
   - `cubic_far_bench.py --trace` prints the counters every second during a fill;
   - pending vs ready, GPU generated and meshed totals, `descent_ms`, `plan_ms`.
   - Find what the fill waits on: the GPU generator, the mesher's round trip, the workers, the planner, or the frame (uploads per frame).
2. **Batch and pipeline sizes:**
   - `MAX_BATCH`, `BATCH_NODES`, `MAX_JOBS`;
   - pool slots;
   - more than one mesher batch in flight;
   - the count readback's latency: map it a frame later instead of waiting.
3. **Producer side:**
   - workers' batch sizes and count;
   - pool waits (`POOL_WAIT`, `POOL_WAITS`);
   - the order in which a hop's nodes are asked for (same set, cheaper order).
4. **Kernels:**
   - `cube_gen.wgsl` per-voxel work: noise octaves per voxel column, the extra layers above;
   - the descent's cost after a jump;
   - the mesher's two passes.
5. **Frame-side limits:** uploads and installs per frame, `RELEASE_BATCH`, fade work.

If 10 iterations in a row are discarded, move to the next item. After the list, go back to the biggest wait with a new angle. Run until the owner checks in the morning.

## Do not

- Change what is planned or how it looks (see Frozen); the guards and tests catch some of it, so don't try.
- Run a build during a benchmark, or two benchmarks at once.
- Push to `main` or force-push; only `research/far-fill`.
- Leave timing prints in kept commits.

## Morning report

Write `research/far-fill/REPORT.md`:
- the baseline and best fill time, and the percentage;
- the kept changes with their effects;
- what did not work;
- where the fill waits now;
- the next ideas.

Then commit and push.
