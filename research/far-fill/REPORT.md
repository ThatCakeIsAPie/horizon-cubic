# Lane A report: the far view's fill (desktop GPU)

This is the overnight autoresearch loop on the desktop: i7 with 4 cores and 8 threads, 24 GB RAM, GTX 1070 Ti, Windows, `fast` profile.
- **Metric:** `tools/far_fill_eval.py`, the median seconds for the far view to complete after a teleport. One eval is 2 launches × 3 hops of 5,000,000 blocks each, at Cubic Far Detail High.
- **Guards** (all against the baseline):
  - no timeout;
  - the planned node count within 3%;
  - voxel size within 2%;
  - steady fps within 10%;
  - cover time within 20%.
- **Records:** the full log is in `autoresearch.jsonl` and `autoresearch_dashboard.md`; the branch is `research/far-fill`.
- **Draft:** this report is updated as the loop goes on.

## Result

| | Fill after a teleport | Fill at spawn |
|---|---|---|
| Baseline (743db86f57) | **17.27 s** (three evals: 17.80, 17.06, 17.27) | about 34 s |
| Now (80315a7ddf) | **about 8.3 s** (evals 8.25 and 8.85; median of the 12 hops pooled is about 8.45) | 15–18 s |

That is about **51% less time, so 2× faster**.

- The far view's look is unchanged.
- Every guard passed.
- At every keep, the plan was the same size (36,500 nodes) at the same voxel size (4 px).

## What was kept

Each item gives the fill before and after.

1. **More far workers** (iteration 3): 17.3 s → 14.8 s.
   - The fill was waiting on its producer workers. The GPU's node pool peaked at 14–34 MB of 68, and the GPU was half idle.
   - The old default was half the threads, at most 4. The new one is the threads minus 2, between 2 and 8.
2. **8 workers, back to normal priority once the near terrain is in** (iteration 6): 14.8 s → about 13.5 s. Spawn went from 29 s to 24 s.
   - The cube workers had lowered themselves to below-normal priority on Windows for good. They now step behind the near terrain only while it loads, through `CubeLod::set_near_loading`, as the column far terrain's threads already did.
   - Workers is now the thread count, between 2 and 8.
3. **Lane B's generator merged** (iterations 7 and 11): 13.4 s → 10.7 s.
   - The opt-in profile (`HORIZON_FAR_PROFILE=1`) showed the workers spending about 88% of their time in the sampler. About 40% of the keys are levels 0–1, made on the CPU by the real generator (`generate` plus a sky-cover probe).
   - Lane B's work on `generate` therefore paid twice.
   - The Windows terrain hash is unchanged.
4. **Sky cover: a slab per look** (iteration 9): 10.7 s → about 9.65 s pooled.
   - Of the CPU-node work, the cover probes were the biggest part: 13–33 s of every 40 worker-seconds.
   - A tile of 4 × 4 columns used to skip its probes only when the bound proved the whole 600-block box above it free. Each look's slab is now proved on its own.
   - `HORIZON_FAR_COVER_CHECK=1` re-probes every skipped point. It found 0 differences.
5. **Cache writes on a keeper thread** (iteration 10): pooled about 9.65 s → 9.05 s.
   - `keep` stored each CPU node, then built its completed parents by reading their eight children back from disk. That was 2–13 s of every 40 worker-seconds, and is now about 0.1 s.
   - The queue is bounded at 1,024 nodes. When it is full, the worker writes itself, as it did before.
6. **One column probe for a column's looks** (iteration 11): pooled about 9.05 s → 8.45 s.
   - Lane B wrote `first_at_column` for this. It works out the column's noise and biome once for all of its heights, each one still `block_at`'s answer; its own test covers about 15,000 points.
   - The cover check found 0 differences.
   - Cover went from 9–21 s to 7–17 s per 40 worker-seconds.

`HORIZON_FAR_*` overrides now exist for the far pipeline's tunables, with unchanged defaults. A value can be tried without a rebuild (`lod::cube::far_knob`).

## What did not work

- **The GPU side:** nothing there limited the fill. These made no gain beyond noise:
  - a GPU batch of 128;
  - a pool of 8,192 slots;
  - a pool reserve.

  The GPU thread is busy 5–9% of the time.
- **Retry after 250 ms instead of 1 s:** no gain.
- **More CPU nodes per batch:** no gain.
- **Tile batches per frame 4/16:** no gain. 2 looked slightly better but did not clear the noise.
- **Descent wait of 100 ms:** worse. **0 ms:** failed the cover guard; cover doubled to 0.43 s.
- **Skipping the cover for all-air nodes:** rejected without a run. A parent made of cached children merges their covers, so its shading would change.

## Where the fill waits now

Per 40 worker-seconds (8 workers, 5 s):
- `generate` for levels 0–1: 13–14 s;
- the sky cover: 7–17 s;
- GPU requests: about 1.5 s;
- queue waits at the start of a hop: 0.3–3.5 s.

The workers make 20,000–23,500 keys per 5 s, against 17,000 at iteration 6. The fill is CPU-bound with all 8 threads busy.

## Next

- **Lane B's generator** is the main lever. Both `generate` and `block_at`'s early-out are on this path.
- **The cover's slab proofs:** up to 112 `classify_box` calls per node. Measure them on their own; a cheaper or coarse-first box proof would help.
- **Replan cadence and tile batches:** being tested as these lines are written.
- **Further out:**
  - GPU generation for level 1, which would need the exact replacers on the GPU, an architectural change;
  - keeping levels 0–1 across hops.
