# Lane B autoresearch: cube loading (ms per cube, lower is better)

- Baseline: **0.7325 ms/cube**. Best: **0.6647 ms/cube** (+9.3% faster).
- Min-delta: 4.1%. Iterations: 3; kept 1.
- Updated 2026-11-06 20:31.

## Hot spots

1. terrain generation (terrain_section -> generate): ~341 ms of ~775 ms per run (44%)
2. record encode in populate (save.rs encode_with, frozen file): ~180 ms (23%)
3. ores (populate.rs ore/try_vein/ellipsoid): ~70 ms (9%)
4. reading back records for entities/flags (store read + decode): ~58 ms (7.5%), fixed in iteration 2
5. section rebuild after population (Section::from_blocks + compact) and PopRegion setup: ~90 ms (12%)

## Iterations

| # | status | score | delta | guard | commit | description |
|---|---|---|---|---|---|---|
| 0 | baseline | 0.7325 | +0.0% | PASS | 0bcb7db9e | unchanged code; medians 0.7423, 0.7323, 0.7325 |
| 1 | discard | 0.7485 | +2.2% | PASS | - | read records via Arc (no deep clone of queued records) for entities/flags/read_cube: no gain; records mostly come from disk |
| 2 | keep | 0.6647 | -9.3% | PASS | d88e519d4 | plain-cube cache: populate skips reading records it wrote with no entities; read_cube answers them from memory (first run 0.6765, confirm 0.6647). Known failing on main too: horizon-world settings::the_preset_survives_a_json_round_trip |
