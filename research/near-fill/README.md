# Lane C: near terrain in the game after a teleport (desktop)

**The metric.** `tools/cubic_far_bench.py --y 80 --hops 3` teleports to the surface three times. Near-in time is measured from the teleport until `status`'s `terrain_loading` (`Streaming::terrain_loading`: cubes generating, lighting, or waiting for meshes) goes false and stays false for 2 s.

## Before and after

| | Near-in after a teleport (2 launches × 3 hops) | Far view complete after the same hops |
|---|---|---|
| Before tonight (743db86f57, plus the status field) | 7.5–7.9 s | 24–29 s |
| Lanes A and B (5b7a3e7368…3e7b30a316) | 5.1–5.8 s | 11–14 s |
| Plus the spread scheduler (5c8b6ac934) | about 4.7 s median (at spawn 2.8–2.9 s, against 3.3) | 11–14 s |

## What the trace showed

`cubic_far_bench --trace-near` and `HORIZON_FAR_PROFILE=1`:
- **Arrival:** columns arrived steadily at about 120/s for about 6 s, 640 columns in all. The client's meshing and lighting kept up.
- **Work per column:** each loads in about 16 ms of CPU, so 650–840 columns a hop take 10–14 s of CPU. Yet the streaming pool was only about 55% busy, counting CPU time.
- **Not the scheduler's caps:** raising the in-flight caps changed nothing. Those are `HORIZON_NEAR_GEN_PER_WORKER` from 3 to 8/16 and `HORIZON_NEAR_LIGHT_PER_WORKER` from 4 to 12 (`sweep1-scheduler-caps.txt`).
- **The cube benchmark across threads:**
  - Tonight's code: 1 thread 0.25 ms/cube, 2 threads 0.24, 4 threads 0.155, 8 threads 0.12–0.17.
  - Before tonight: 1.06, 0.56 and 0.38.
  - Columns made side by side wait on each other's population region locks.
- **Hand-out order, 8 threads (`HORIZON_CUBE_BENCH_ORDER`):** cubes one by one 0.12; whole columns 0.10, as the game does; columns in a checkerboard 0.080. With the checkerboard, 2/4/8 threads give 1.8/2.55/3.3× one.

## Kept

`ChunkGenerator::spread_loads`, which the cubic generator turns on. The streaming scheduler takes, from the nearest 4 × room columns, those none of whose neighbours is in flight or already picked. It falls back to the others only when no such column is left (`spread_pick`, with a test). `HORIZON_NEAR_SPREAD=0` turns it off. `sweep2-spread.txt` has the A/B runs.

## Files

- `near_compare.sh`: builds before and now, and runs both.
- `near_sweep.sh`: runs settings, one launch each.
