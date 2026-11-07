# Lane B autoresearch: cube loading (ms per cube, lower is better)

| spec | cubes | baseline | best | change | min-delta |
|---|---|---|---|---|---|
| surface | `4,-4,8,1` | 0.7325 | **0.2517** | -65.6% | 4.1% |
| sky | `4,20,32,1` | 0.2018 | **0.0130** | -93.6% | 6.5% |
| deep | `4,-40,-28,1` | 0.2991 | **0.1262** | -57.8% | 3.0% |

Iterations: 34; kept 21. Known failing test (fails on main too): horizon-world cubic::custom::settings::tests::the_preset_survives_a_json_round_trip (fails on unchanged main on Linux).
Updated 2026-11-06 23:14.

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
| 6 | deep | discard | 0.2584 | +0.2% | PASS | - | carver walks resolved per 8^3 region with OnceLocks (no per-origin mutex/hash): deep 0.2584, surface 0.6257, sky 0.0550: no gain |
| 7 | surface | keep | 0.4502 | -27.7% | PASS | e52179485 | deferred record encoding: populate queues section+entities+flags, NBT made once on first need (writer or read); writer skips superseded records. First 0.4511, confirm 0.4502. Disk records verified identical (flush + hash of every record) on all specs; make+flush 1.37 s -> 0.57 s |
| 7 | sky | keep | 0.0231 | -58.9% | PASS | e52179485 | same change: first 0.0236, confirm 0.0231 (uniform air cubes were spending 24 us each encoding) |
| 7 | deep | keep | 0.2099 | -18.6% | PASS | e52179485 | same change: first 0.2119, confirm 0.2099 |
| 8 | deep | discard | 0.2078 | -1.0% | PASS | - | carver replaceable lists as a bitset instead of Vec::contains: deep -1%, surface 0.4470 (-0.7%): noise |
| 9 | surface | keep | 0.4032 | -10.4% | PASS | d162110b5 | replacer filters (densityRange, randomYGradient) as small Vec instead of HashSet: first 0.4136, confirm 0.4032. Deep 0.2139/0.2149 vs previous commit re-measured 0.2133 (drift), sky 0.0231 |
| 10 | surface | discard | 0.3975 | -1.4% | PASS | - | per-cube memo of surface/mesa column depth noise: -1.4%, below min-delta |
| 11 | surface | keep | 0.3786 | -6.1% | PASS | faac3ad72 | ore/spring block sets and blockMatch ids as small Vec instead of HashSet, deepslate test once: first 0.3862, confirm 0.3786; deep 0.2119 |
| 12 | surface | keep | 0.3217 | -15.0% | PASS | eb7f9e2a2 | population cache holds changed cubes as block arrays; section built once on demand (record encode / read_cube / terrain view): first 0.3201, confirm 0.3217. Disk records verified identical. Deep 0.2088, sky 0.0230 |
| 13 | deep | discard | 0.2164 | +3.1% | PASS | - | SphereRun of 8 spheres instead of 16: deep 0.2164, surface 0.3249: no gain |
| 14 | deep | keep | 0.1949 | -7.1% | PASS | 92b5d4f42 | sphere runs rejected with local_box's exact box test (extent c+-r vs c0, c0+17): first 0.1985, confirm 0.1949 |
| 14 | surface | keep | 0.3075 | -4.4% | PASS | 92b5d4f42 | same change: first 0.3077, confirm 0.3075 (--tests); sky 0.0226 |
| 15 | surface | discard | 0.2960 | -3.7% | PASS | - | per-cube replacer plans (active replacers per biome, dead gradients dropped): -3.7%, below 4.1% min-delta; patch kept for a combined replacer change |
| 16 | surface | discard | 0.2978 | -3.2% | PASS | - | lattice corner cache alone (per-thread, keyed by blend id): -3.2%, below min-delta |
| 17 | surface | keep | 0.2816 | -8.4% | PASS | 24a5eb340 | corner cache + per-cube replacer plans together (both exact, each ~3-4% alone): first 0.2880, confirm 0.2816; generate 135 -> 122 us/cube. Deep 0.1994 / sky 0.0240 vs HEAD re-measured 0.1982 / 0.0228 (noise) |
| 18 | deep | discard | 0.1924 | -1.3% | PASS | - | sphere distance tables (dz, dy per sphere) alone: deep -1.3% (vs re-measured HEAD -2.9%): below min-delta; folded into 19 |
| 19 | deep | keep | 0.1625 | -16.6% | PASS | 111967cb6 | carver origins walked row by row (no Vec, no sort), may_reach inside the cache lock, multiplicative hash for position maps, sphere distance tables: first 0.1678, confirm 0.1625; generate 92 -> 76 us |
| 19 | surface | keep | 0.2615 | -7.1% | PASS | 111967cb6 | same change: first 0.2645, confirm 0.2615 (--tests); sky 0.0229 |
| 20 | sky | keep | 0.0182 | -21.2% | PASS | 305283927 | copy-on-write population region cubes (uniform / shared cache array until written): first 0.0186, confirm 0.0182; surface 0.2631, deep 0.1635 (noise) |
| 21 | sky | keep | 0.0157 | -13.7% | PASS | 356f37d14 | biome blend thread-local caches use a multiplicative integer hasher instead of SipHash: first 0.0156, confirm 0.0157 (--tests); surface 0.2577, deep 0.1606 |
| 22 | deep | keep | 0.1395 | -14.2% | PASS | c861bf9c0 | region reads mixed sections in place via the palette, unpacks only on first write: first 0.1444, confirm 0.1395 (--tests); surface 0.2577, sky 0.0157 |
| 23 | deep | discard | 0.1356 | -2.8% | PASS | - | lava/water wall scans skipped when the cube cannot hold any: -2.8% alone (below min-delta); folded into 24 |
| 24 | deep | keep | 0.1262 | -9.5% | PASS | a6eb67668 | wall-scan skip + bitset replaceable lists in carve_sphere: first 0.1302, confirm 0.1262 (--tests); surface 0.2517, sky 0.0155 |
| 24 | surface | keep | 0.2517 | -3.7% | PASS | a6eb67668 | measurement on iteration 24 code (surface best tracking; cumulative small gains since 19) |
| 25 | surface | discard | 0.2453 | -2.5% | PASS | - | population functions generic over the world (PopRegion block/set_block inline instead of dyn calls): -2.5%, below min-delta; patch kept |
| 26 | deep | discard | 0.1251 | -0.9% | PASS | - | JRandom as an inlined copy of the Java LCG (no cross-crate calls): deep -0.9%, surface -0.9%, sky -2%: below min-delta (and mostly a non-LTO profile effect) |
| 27 | sky | keep | 0.0130 | -17.2% | PASS | d53a190fb | uniformity proven first from coarse octaves (wavelength >= 16; finer bounded by amplitude, gradient from all): first 0.0132, confirm 0.0130 (--tests); deep 0.1278, surface 0.2534 (noise) |
