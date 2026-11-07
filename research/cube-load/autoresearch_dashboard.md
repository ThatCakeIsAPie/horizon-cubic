# Lane B autoresearch: cube loading (ms per cube, lower is better)

| spec | cubes | baseline | best | change | min-delta |
|---|---|---|---|---|---|
| surface | `4,-4,8,1` | 0.7325 | **0.6231** | -14.9% | 4.1% |
| sky | `4,20,32,1` | 0.2018 | **0.0562** | -72.2% | 6.5% |
| deep | `4,-40,-28,1` | 0.2991 | **0.2578** | -13.8% | 3.0% |

Iterations: 7; kept 4. Known failing test (fails on main too): horizon-world cubic::custom::settings::tests::the_preset_survives_a_json_round_trip (fails on unchanged main on Linux).
Updated 2026-11-06 20:53.

## Hot spots

1. terrain generation (terrain_section -> generate): ~341 ms of ~775 ms per run (44%)
2. record encode in populate (save.rs encode_with, frozen file): ~180 ms (23%)
3. ores (populate.rs ore/try_vein/ellipsoid): ~70 ms (9%)
4. reading back records for entities/flags (store read + decode): ~58 ms (7.5%), fixed in iteration 2
5. section rebuild after population (Section::from_blocks + compact) and PopRegion setup: ~90 ms (12%)

## Iterations

| # | spec | status | score | delta | guard | commit | description |
|---|---|---|---|---|---|---|---|
| 0 | surface | baseline | 0.7325 | +0.0% | PASS | 0bcb7db9e | unchanged code; medians 0.7423, 0.7323, 0.7325 |
| 1 | surface | discard | 0.7485 | +2.2% | PASS | - | read records via Arc (no deep clone of queued records) for entities/flags/read_cube: no gain; records mostly come from disk |
| 2 | surface | keep | 0.6647 | -9.3% | PASS | d88e519d4 | plain-cube cache: populate skips reading records it wrote with no entities; read_cube answers them from memory (first run 0.6765, confirm 0.6647). Known failing on main too: horizon-world settings::the_preset_survives_a_json_round_trip |
| 3 | sky | keep | 0.0562 | -72.2% | PASS | 6db0c789b | carvers skip uniform cubes of a block no carver replaces: sky 0.2018 -> 0.0562 (-72%, first 0.0554); surface 0.6458, deep 0.2787 (unchanged) |
| 4 | deep | baseline | 0.2816 | -5.9% | PASS | d88e519d4 | deep spec re-measured after iteration 2 (plain cache): medians 0.2826, 0.2807; carvers are 176 ms of 295 ms here (stone is replaceable) |
| 5 | deep | keep | 0.2578 | -8.5% | PASS | be61d382b | carver sphere runs: bounds per 16 spheres skip runs local_box would reject (12.2k sphere tests/cube -> few hundred). first 0.2609, confirm 0.2578 |
| 5 | surface | keep | 0.6231 | -6.3% | PASS | be61d382b | same change (iteration 5) on surface: first 0.6253, confirm 0.6231 with --tests |
