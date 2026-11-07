# Lane A: far view fill (desktop) - autoresearch dashboard

**Metric:** far_fill_s: median s to complete a fill over 6 teleports (2 launches x 3 hops), lower is better
**Baseline:** 17.267 s | **Best:** 8.994 s (-47.9%) | **Min-delta:** 1.5 s | **Iterations:** 9
**Guard:** no timeout; planned within 3%, voxel_px within 2%, fps_steady within 10%, cover_s within 20% of baseline; far view tests

**Kept:** 4 | **Discarded:** 0 | **Screened:** 27 | **Crashed:** 0 | **Guard failures:** 0

| # | Score (s) | Delta | Status | Commit | Description |
|---|---|---|---|---|---|
| 0 | 17.804 |  | baseline | 743db86f57 | baseline run 1 |
| 0 | 17.056 |  | baseline | 743db86f57 | baseline run 2 |
| 0 | 17.267 |  | baseline | 743db86f57 | baseline run 3 |
| 1 | 19.216 | +1.949 | screen | e74a9bbb1e | sweep 1 (1 launch, noisy): control (knob build, nothing set); peak pool 14-34 MB of 68: not pool-bound |
| 1 | 15.335 | -1.932 | screen | e74a9bbb1e | sweep 1 (1 launch, noisy): HORIZON_FAR_WORKERS=6; peak pool 14-34 MB of 68: not pool-bound |
| 1 | - |  | screen | e74a9bbb1e | sweep 1 (1 launch, noisy): HORIZON_FAR_WORKERS=8 (launch failed, flaky); peak pool 14-34 MB of 68: not pool-bound |
| 1 | 17.086 | -0.181 | screen | e74a9bbb1e | sweep 1 (1 launch, noisy): HORIZON_FAR_PRIORITY=2; peak pool 14-34 MB of 68: not pool-bound |
| 1 | - |  | screen | e74a9bbb1e | sweep 1 (1 launch, noisy): HORIZON_FAR_GPU_BATCH=128 (launch failed, flaky); peak pool 14-34 MB of 68: not pool-bound |
| 1 | 17.186 | -0.081 | screen | e74a9bbb1e | sweep 1 (1 launch, noisy): GPU_BATCH=128 POOL_NEED=4; peak pool 14-34 MB of 68: not pool-bound |
| 1 | 17.496 | +0.229 | screen | e74a9bbb1e | sweep 1 (1 launch, noisy): POOL_SLOTS=8192 (launch failed; 17.496 on rerun); peak pool 14-34 MB of 68: not pool-bound |
| 1 | 16.647 | -0.620 | screen | e74a9bbb1e | sweep 1 (1 launch, noisy): POOL_SLOTS=8192 GPU_BATCH=128 POOL_NEED=4; peak pool 14-34 MB of 68: not pool-bound |
| 1 | 17.201 | -0.066 | screen | e74a9bbb1e | sweep 1 (1 launch, noisy): RETRY_MS=250; peak pool 14-34 MB of 68: not pool-bound |
| 1 | 16.516 | -0.751 | screen | e74a9bbb1e | sweep 1 (1 launch, noisy): TILE_BATCHES=2; peak pool 14-34 MB of 68: not pool-bound |
| 1 | 17.145 | -0.122 | screen | e74a9bbb1e | sweep 1 (1 launch, noisy): TILE_BATCHES=16; peak pool 14-34 MB of 68: not pool-bound |
| 1 | - |  | screen | e74a9bbb1e | sweep 1 (1 launch, noisy): DESCENT_WAIT_MS=100 (launch failed, flaky); peak pool 14-34 MB of 68: not pool-bound |
| 1 | 16.342 | -0.925 | screen | e74a9bbb1e | sweep 1 (1 launch, noisy): CPU_PER_BATCH=8; peak pool 14-34 MB of 68: not pool-bound |
| 2 | 15.126 | -2.141 | screen | e74a9bbb1e | sweep 2 (2 launches): HORIZON_FAR_WORKERS=6 (full eval) |
| 2 | 14.662 | -2.605 | screen | e74a9bbb1e | sweep 2 (2 launches): HORIZON_FAR_WORKERS=8 |
| 2 | 15.012 | -2.255 | screen | e74a9bbb1e | sweep 2 (2 launches): WORKERS=6 PRIORITY=2 (spawn 24.4/25.3 s vs 28.8/28.1) |
| 2 | 15.130 | -2.137 | screen | e74a9bbb1e | sweep 2 (2 launches): WORKERS=6 CPU_PER_BATCH=8 |
| 2 | 14.883 | -2.384 | screen | e74a9bbb1e | sweep 2 (2 launches): WORKERS=6 TILE_BATCHES=2 |
| 3 | 14.849 | -2.418 | keep | 64dba72697 | far workers = threads - 2 (2..8): 6 on the desktop, was half (2..4) = 4; sweep 15.126, confirmation 14.849; tests 206 pass |
| 4 | 15.256 | +0.407 | screen | 64dba72697 | sweep 3 on the 6-worker default (2 launches): PRIORITY=2 (spawn 24.0/23.3) |
| 4 | 14.189 | -0.660 | screen | 64dba72697 | sweep 3 on the 6-worker default (2 launches): PRIORITY=2 TILE_BATCHES=2 |
| 4 | 13.824 | -1.025 | screen | 64dba72697 | sweep 3 on the 6-worker default (2 launches): WORKERS=8 |
| 4 | 12.768 | -2.081 | screen | 64dba72697 | sweep 3 on the 6-worker default (2 launches): WORKERS=8 PRIORITY=2 (spawn 21.3/22.6) |
| 4 | 15.316 | +0.467 | screen | 64dba72697 | sweep 3 on the 6-worker default (2 launches): TILE_BATCHES=4 |
| 5 | 14.324 | -0.525 | screen | 64dba72697 | sweep 4 (2 launches): WORKERS=8 PRIORITY=2 repeat (one hop 19.7; pooled 12 hops with sweep 3: median 13.15 vs 15.0 for the 6-worker default) |
| 5 | 14.079 | -0.770 | screen | 64dba72697 | sweep 4 (2 launches): WORKERS=8 PRIORITY=1 (= below normal on Windows, behind_server is a no-op there) |
| 5 | 12.945 | -1.904 | screen | 64dba72697 | sweep 4 (2 launches): WORKERS=10 PRIORITY=2 |
| 6 | 13.402 | -1.447 | keep | 24017ce228 | 8 workers (threads, 2..8) + normal priority once near terrain is in (below normal while it loads); evals 14.044 and 13.402, pooled 12-hop median 13.55 vs 15.0; spawn fill ~24 s vs ~29.5; profile: 88% of worker time in resident_many, GPU thread 5-9% busy, 40% of keys are CPU-sampled levels 0-1 (~14 ms each) |
| 7 | 10.654 | -2.748 | keep | 5b7a3e7368 | merged lane B (origin/research/cube-load 6debf91272): the far view's level 0-1 nodes use the same generate(); evals 10.741 and 10.654 (best was 13.402), spawn fill 17.7-20.4 s (was ~24). Windows terrain hash unchanged (4c048498d55916e2), cube bench surface 1.138 -> 0.456 ms. Tests: far view 206 pass; client cubic 84 pass; world cubic 52 pass + the preset JSON round trip, which fails on unchanged main in the fast profile, and a timing test that failed once under load and passed 5 of 5 after |
| 8 | 10.276 | -0.378 | screen | (merge 67d659e111 + sampler profile, uncommitted) | merged lane B 67d659e111 (generate speedups; Windows hash unchanged, cube bench 0.390 ms) + sampler counters: 10.276 and 10.498, no gain over 10.654 (within noise). Profile per 5 s: cover 13-33 s, sample 7-14 s, keep 1.5-9 s of ~40 worker-s: the sky cover probes are the biggest cost |
| 9 | 8.994 | -1.660 | keep | 64e40f1647 | sky cover: skip each look's slab the density bound proves free (was only the whole 600-block box); evals 8.994 / 9.822, pooled median ~9.65 vs 10.65; 0 cover mismatches with HORIZON_FAR_COVER_CHECK; min-delta now 8.7% of best |
