# Lane B report: cube loading (generation and population)

The overnight autoresearch loop on the laptop: Framework 13, Ryzen 5 7640U (6 cores), 7 GB RAM, Arch/Omarchy, Rust 1.99.1, `fast` profile.

The metric is `tools/cube_load_bench.py`: the median ms per cube over 5 runs, single thread, in a fresh world.
- The guard is the terrain hash, which must be bit-identical.
- The full log is `autoresearch.jsonl` and `autoresearch_dashboard.md`. The branch is `research/cube-load`.

## Result

| spec | cubes | baseline | best | change |
|---|---|---|---|---|
| surface | `4,-4,8,1` (spawn) | 0.7325 ms | **~0.25 ms** | **−66%** (2.9× faster) |
| sky | `4,20,32,1` (y 320–528, all air) | 0.2018 ms | **0.0155 ms** | **−92%** (13×) |
| deep | `4,-40,-28,1` (y −640…−448, stone) | 0.2991 ms | **0.1262 ms** | **−58%** (2.4×) |

- All three terrain hashes are unchanged.
- Each spec's baseline hash was written from the unchanged generator code (0bcb7db).
- The changes that touch what is saved were also checked another way. A temporary test (not committed) made the cubes, flushed the writer, and hashed every record read back from disk (the NBT Debug strings). It gave the same hash as the original code on all three specs.
- `generate` alone, which is what the far view's CPU nodes call, went from 190 to about 100 µs per cube at the surface, 117 to 67–74 µs in deep rock, and 84 to 5 µs in the sky.

One test fails, and it fails on unchanged main too: `horizon-world` `cubic::custom::settings::tests::the_preset_survives_a_json_round_trip`. It is an f64 mismatch in the lake UserFunction round trip on Linux, in frozen `settings.rs`. Every other `cubic` test passes: 84 in horizon-client-lib and 53 in horizon-world.

## What was kept

These are in commit order. The percentages are each change's confirmation median against the best before it.

1. **Plain-cube cache** (d88e519). The population cache marks a cube "plain" when its latest record is the section a population wrote with no entities. `populate` then stops reading that record back for flags and entities, and `read_cube` returns the cached section instead of reading and decoding the record. Surface −9%; deep about −6%.
2. **Carvers skip uniform cubes they cannot change** (sky). A carver only replaces blocks on its replaceable list, so a one-block cube of air, water or bedrock is left alone. Sky −72% (generate went from 84 to 7 µs).
3. **Sphere runs.** Walks keep bounds per 16 consecutive spheres, and runs that cannot reach the cube are skipped. Deep −8.5%, surface −6%. This was later tightened by item 8.
4. **Records encoded once, when first needed.** `populate` queues the section, entities and flags. The NBT is made by the writer thread, or by a read if one comes first, and the writer skips any record a newer write replaced.
   - At the surface a population wrote about 4.4 records, each encoded on the game thread by `save.rs::encode_with` (frozen).
   - Surface −28%, sky −59%, deep −19%.
   - Making the cubes plus flushing them all to disk went from 1.37 s to 0.57 s at the surface, so this is a real reduction in work, not just a move to another thread.
5. **Replacer filters as small slices.** densityRange and randomYGradient filters were `HashSet` lookups, tested on every block. Surface −10%.
6. **Ore and spring block sets as small slices**, with the deepslate test done once. Surface −6%.
7. **Cubes in flight keep a population's block array** (`Held::Blocks`). The packed section (`from_blocks` plus `compact`) is built once, on demand: by encoding, by `read_cube`, or by a terrain view. Before, every population that touched a cube re-packed it and the next one unpacked it again. Surface −15%.
   - Memory: a held array is 16 KB against a compact section's few KB, for cubes in flight that nothing has read yet. The cache holds at most 6,144 cubes. `read_cube` swaps a cube back to its section when it returns it.
8. **Sphere runs rejected by `local_box`'s exact box test.** The test is `c + r < c0` or `c − r ≥ c0 + 17` on any axis. The old test was the looser first one, which let about 743 spheres per cube through to touch about 39. Deep −6.7%, surface −4.4%.
9. **Shared lattice corners plus per-cube replacer plans** (24a5eb3).
   - Lattice corner densities are cached per thread, keyed by the biome blend's unique id; one blend means one terrain and one noise.
   - Each biome in a cube gets the ordered list of replacers that can act at that cube's Y. A gradient is dropped only if its probability is ≤ 0 at each of the cube's Y, computed exactly as `apply` computes it, and NaN stays live.
   - Surface −8.4%; generate went from 135 to 122 µs.
10. **Carver origins walked row by row** (no `Vec`, no sort, still in CWG's z, y, x order). `may_reach` runs under the cache lock, so only walks that reach the cube clone an `Arc`. Position maps use a multiplicative hash. Sphere distances are tabled per sphere. Deep −16.6%, surface −7%.
11. **Copy-on-write population regions.** A region cube is the uniform state or the cache's array until the population first writes to it. Sky −19%.
12. **The biome blend's thread-local caches hash with a multiply** instead of SipHash. Sky −14%; surface and deep about −1.5%.
13. **Regions read mixed sections in place** through the palette, unpacking only on the first write. Before, every carved cube was unpacked by all 8 populations. Deep −13%.
14. **Wall-scan skip plus bitset in `carve_sphere`.** The lava or water wall scans are skipped when the cube cannot hold any, and the replaceable lists are a bitset. Deep −9.5%.

## What did not work (discarded, the code reverted)

- **Reading queued records through an `Arc` instead of cloning them.** No gain: most records were already on disk, so the cost was the decode, not the clone.
- **Carver walks resolved per 8³ region with `OnceLock`s.** No gain on its own. The lookup cost turned out to be the hashing and `Arc` traffic, which item 10 removed.
- **Sphere runs of 8 instead of 16.** No gain.
- **Per-cube memo of the surface depth noise per column.** −1.4%: Surface's noise was not the cost.
- **Each of these alone was below min-delta, and they were kept only in a combined form:**
  - replacer plans (−3.7%);
  - the corner cache (−3.2%);
  - sphere distance tables (−1.3%);
  - the wall-scan skip (−2.8%).
- **Population functions generic over `CubeWorld`** (no dyn calls in the ore loop): −2.5%, below min-delta. The patch is kept in the session notes.
- **`JRandom` as an inlined copy of the Java LCG:** about −1%. That would mostly be an effect of the non-LTO `fast` profile anyway; release uses thin LTO.

## Where the time goes now

These were measured with temporary `Instant` timers, since `perf` is not installed.
- **Surface** (about 263 ms per run):
  - generate is about 185 ms: about 90 ms base (lattice about 33, replacer chain about 55), about 60 ms carvers, about 10 ms bounds;
  - ores about 55 ms: 1.49M ellipsoid cells per run, about 37 ns each;
  - the rest is bookkeeping.
- **Deep** (about 133 ms per run): almost all generate.
  - carvers: about 23 ms first-time cave walk simulation, about 25 ms carving the about 46 spheres per cube that hit, and the rest origin enumeration;
  - bounds about 9 ms.
- **Sky** (about 16 ms per run): the bounds (`uniform_block`, about 5 µs per cube: noise ranges about 2.3, 25 terrain corners, 16 biomes) and population bookkeeping.

## What to try next

- **Ores:** the 37 ns per cell is mostly the per-cell f64 division (which must stay), the palette read, and dyn calls. The generic-world patch (−2.5%) plus an ellipsoid-level fast path could add up.
- **Bounds:**
  - batch the 25 `terrain_at` and 16 `biome_at` calls into one thread-local access;
  - share the 5×5 corner grid between `uniform_block` and `Lattice::for_cube`.
- **Walk simulation (about 23 ms per run in deep and at the surface):** it is RNG- and `libm::sinf`-bound and exact, so the only option is making the draws cheaper. With LTO this may already be done.
- **The replacer chain:** about 13 ns per block across about 2.8 active replacers. A per-replacer pass over the whole cube, instead of the per-block chain, might vectorise.
- **Multi-thread loading** is outside this lane's spec (one thread). The plain cache, deferred records and held arrays are all lock-protected as before.
