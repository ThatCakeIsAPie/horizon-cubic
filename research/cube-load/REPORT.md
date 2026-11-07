# Lane B report: cube loading (generation and population)

The overnight autoresearch loop on the laptop: Framework 13, Ryzen 5 7640U (6 cores), 7 GB RAM, Arch/Omarchy, Rust 1.99.1, `fast` profile.

The metric is `tools/cube_load_bench.py`: the median ms per cube over 5 runs, single thread, in a fresh world.
- The guard is the terrain hash, which must be bit-identical.
- The full log is `autoresearch.jsonl` and `autoresearch_dashboard.md`. The branch is `research/cube-load`, which never touched main.

## Result

| spec | cubes | baseline | now | change |
|---|---|---|---|---|
| surface | `4,-4,8,1` (spawn) | 0.7325 ms | **0.194–0.200 ms** | **−73%** (3.7× faster) |
| sky | `4,20,32,1` (y 320–528, all air) | 0.2018 ms | **0.0128–0.0130 ms** | **−94%** (16×) |
| deep | `4,-40,-28,1` (y −640…−448, stone) | 0.2991 ms | **0.116 ms** | **−61%** (2.6×) |

The final measurement on the branch head (2d90ae9) was surface 0.1998 (runs 0.1952–0.2022), sky 0.0130 and deep 0.1158. The best confirmation medians in the log are lower: surface 0.1941, sky 0.0128. The laptop drifts by about 2% between runs.

- All three terrain hashes are unchanged.
- Each spec's baseline hash was written from the unchanged generator code (0bcb7db).
- The changes that touch what is saved (items 1, 4, 7 and 19) were also checked another way. A temporary test (not committed) made the cubes, flushed the writer, and hashed every record read back from disk (their NBT). It gave the original code's hash on all three specs each time.
- Making the surface spec's cubes and flushing every record to disk now takes 0.30 s, against 1.37 s for the original code.
- `generate` alone, which is what the far view's CPU nodes call, went from 190 to about 70 µs per cube at the surface, 117 to about 64 µs in deep rock, and 84 to 4 µs in the sky.
- A multi-threaded smoke test (specs `4,-4,8,4`, `4,-40,-28,6` and `6,-4,8,6`, three runs each) ran clean: no crash or deadlock. Deep with 6 threads gives the single-thread hash. The surface hashes vary between runs, as the bench documents for overlapping populations.

One test fails, and it fails on unchanged main too: `horizon-world` `cubic::custom::settings::tests::the_preset_survives_a_json_round_trip`. It is an f64 mismatch in the lake UserFunction round trip on Linux, in frozen `settings.rs`. Every other `cubic` test passes: 84 in horizon-client-lib and 55 in horizon-world, two of them new (below).

## What was kept

These are in commit order. The percentages are each change's confirmation median against the best before it.

1. **Plain-cube cache.** The population cache marks a cube "plain" when its latest record is the section a population wrote with no entities. `populate` then stops reading that record back, and `read_cube` returns the cached section instead of reading and decoding it. Surface −9%.
2. **Carvers skip uniform cubes they cannot change.** A carver only replaces blocks on its replaceable list, so a one-block cube of air, water or bedrock is left alone. Sky −72%.
3. **Sphere runs.** Walks keep bounds per 16 consecutive spheres. Deep −8.5%, surface −6%. This was later tightened by item 8.
4. **Records encoded once, when first needed.** `populate` queues the section, entities and flags. The NBT is made by the writer thread, or by a read if one comes first, and the writer skips any record a newer write replaced.
   - At the surface a population wrote about 4.4 records, each encoded on the game thread by frozen `save.rs::encode_with`.
   - Surface −28%, sky −59%, deep −19%.
   - This is a real reduction in work, not just a move to another thread: make plus flush went from 1.37 s to 0.57 s.
5. **Replacer block filters as small slices**, not `HashSet`s. Surface −10%.
6. **Ore and spring block sets as small slices.** Surface −6%.
7. **Cubes in flight keep a population's block array** (`Held::Blocks`). The packed section is built once, when something asks for one. Surface −15%.
8. **Sphere runs rejected by `local_box`'s exact box test.** The test is `c + r < c0` or `c − r ≥ c0 + 17`, from the run's extent. Before, about 743 spheres per cube passed to touch about 39. Deep −6.7%, surface −4.4%.
9. **Shared lattice corners plus per-cube replacer plans.**
   - Corner densities are cached per thread, keyed by the biome blend's unique id; one blend means one terrain and one noise.
   - Each biome gets the replacers that can act at the cube's Y. A gradient is dropped only if its probability is ≤ 0 at each Y; NaN stays live.
   - Surface −8.4%.
10. **Carver origins walked row by row** in CWG's z, y, x order (no `Vec`, no sort), with `may_reach` under the lock, a multiplicative hash, and sphere distance tables. Deep −16.6%, surface −7%.
11. **Copy-on-write population regions:** a region cube is the uniform state or the cache's array until first written. Sky −19%.
12. **Multiplicative hash for the biome blend's thread-local caches.** Sky −14%.
13. **Regions read mixed sections in place** through the palette, unpacking only on the first write. Deep −13%.
14. **`carve_sphere`:** the lava or water wall scans are skipped when the cube cannot hold any, and the replaceable lists are a bitset. Deep −9.5%.
15. **Coarse-first uniformity proof.** The noise is bounded from octaves of a wavelength of 16 blocks or more first, with finer ones bounded by amplitude and the gradient from all octaves; the full proof follows if needed. A sound proof gives the true block either way. Sky −16%. `block_at`'s early-out uses it too.
16. **Carver walks held per region of origins** (a `OnceLock` each). Deep −3.3%.
17. **Lattice cells proven by the replacer analysis.**
    - A 4×8×4 cell whose 8 corners are on one side of the surface gets `possible_blocks` from its exact corner range, its edge gradient, its Y and its single biome quart.
    - A proven cell is filled with that block and skips 128 replacer chains.
    - A new test (`proven_lattice_cells_match_the_chain`) compares it with the full chain over about 600 cubes.
    - Surface −10%.
18. **Surface and mesa depth noise memoised per column.** The profile showed `SimplexNoise::get2` at 6%. Surface −5.4%.
19. **Generated mixed cubes stay block arrays** until a section is needed. Border cubes, generated only for their neighbours' populations, are never packed on the game thread. Surface −7.9%. It had been −2.8% (discarded) before item 17.

For lane A (the far view): `CustomCubicGenerator::first_at_column` and `blocks_at_column`, column probes that reuse the noise columns and biome across heights. The test `column_probes_answer_as_block_at` compares them with `block_at` over about 15k points at resolutions 1, 2, 4 and 8.

**Memory trade-off (items 7 and 19).** A cube in flight that nobody has asked a section of is held as a 16 KB array instead of a compact section of a few KB. The population cache keeps at most 6,144 cubes, so the worst case is about 100 MB. `read_cube` swaps a returned cube back to its section.

## What did not work (discarded, the code reverted)

- **Records read through an `Arc` instead of cloned.** The cost was the decode, not the clone.
- **Sphere runs of 8.**
- **A 32-block coarse cutoff.**
- **Narrowing carve columns to the inside run.**
- **A reused generator output buffer.**
- **A pool of 16 KB arrays.** So the libc time is not allocator churn.
- **Batched lattice-corner lookups plus 16 quart biomes instead of 256 `Rc` clones.**
- **Ore cell access:** distance tables, a fused `replace_ore`, and population generic over `CubeWorld` (no dyn calls), each alone and together. At most −2.5%; the ore cost is the per-cell work itself.
- **Walk simulation:**
  - `libm::sincosf` for the sin/cos pairs. Bit-identical to `sinf`/`cosf` over all 2³² f32 inputs, but no gain.
  - a table of `sin(walked / max · π)`;
  - `JRandom` as an inlined Java LCG (−1%, mostly a non-LTO artifact).
- **Bounds lookups batched into one thread-local access:** sky −3.8%, below min-delta.
- **Kept only in combined form**, each below min-delta alone:
  - replacer plans and the corner cache (item 9);
  - sphere tables (item 10);
  - the wall-scan skip and bitset (item 14).

## Where the time goes now

`perf` is not installed, and samply needs `perf_event_paranoid` ≤ 1, which is 2 here. I used an `LD_PRELOAD` `SIGPROF` sampler outside the repo instead: instruction pointers mapped through `nm`, kept on the bench thread. On a larger surface run (13×13×13 cubes), self time is:
- ore veins (`try_vein` with the inlined ellipsoid, plus `PopRegion::block`): about 15%;
- Perlin noise (`get3`, `gradients_at`, frozen in horizon-core): about 11%;
- libc (memcpy, memset, malloc): about 12%;
- cave carving (`carve_sphere`): about 6%;
- origin enumeration (`near`): about 6%;
- section packing: about 3%;
- the bench's own hash (outside the timing): about 8%.

Deep rock is almost all generate: carving, origin enumeration, and about 23 ms per run of first-time cave walk simulation for about 1,560 walks.

## What to try next

Most of the last ten iterations were within noise. The remaining cost is frozen noise, or work that is already the minimum for bit-identical output: ore cells, carving and walk simulation. So the loop stopped here rather than spend more of the night on sub-noise changes. Ideas that might still pay:
- **Avoid the copy-on-write copies.** About 5,500 copies of 16 KB per surface run, roughly 88 MB of memcpy, maybe 2–4%. A population would have to take its cubes out of the cache while it holds their stripe locks. Done naively, a concurrent `terrain()` reader could re-insert a stale cube, so the cache would need to know which cubes are in a population.
- **Ores:** an ellipsoid-level fast path for a region cube that holds no replaceable state, such as all air: skip its cells without reading them. The `win.contains` test and the distances remain.
- **The replacer chain** (about 13 ns per block over about 2.8 active replacers): a per-replacer pass over a whole cube instead of a per-block chain, which might vectorise.
- **Release profile:** these numbers are for `fast` (no LTO). Some of the discarded micro-changes, such as inlining across crates, would matter even less with thin LTO, and the structural ones should hold.
