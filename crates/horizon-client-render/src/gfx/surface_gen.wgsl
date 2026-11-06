// Height-map tiles of the coarsest far terrain (see surface_tile.rs for the
// scheme and the tile format). Appended to cwg_noise.wgsl. One workgroup per
// tile, one thread per column: each column takes SAMPLES jittered sample
// points (biome_gen's columns, already in `columns`/`col_biomes`), finds the
// highest solid block of each, keeps the sample with the highest one and
// marches that vertical line for its solid spans.

const MAX_OCTAVES: u32 = 16u;
const SPANS: u32 = 4u;
const SAMPLES: u32 = 1u;
const HEADER_WORDS: u32 = 16u;
const COLUMN_WORDS: u32 = 13u;
const TILE_WORDS: u32 = 3344u;
const OPEN_BOTTOM: i32 = -2147483647 - 1;
const OPEN_TOP: i32 = 2147483647;
const STEP_CAP: i32 = 64;
const MAX_STEPS: i32 = 1024;
const SKIRT_MIN: i32 = 16;

struct Params {
    // heightFactor, heightOffset, heightVariationFactor, heightVariationOffset
    shape: vec4<f32>,
    // special (below-average) volatility factor, deepslate below this y
    shape2: vec4<f32>,
    // lowest y, highest y, sea level, biome count
    world: vec4<i32>,
    // the four noises' offsets
    noffs: vec4<f32>,
    // per level: octaves in use for selector, low, high, depth
    used: array<vec4<u32>, 32>,
    // per level: lowest and highest value of the low/high mix
    mix: array<vec4<f32>, 32>,
    // per octave slot (noise * 16 + octave): w = signed weight
    w: array<vec4<f32>, 64>,
    // fixed-point phase constants per axis (cwg_noise.wgsl's `phase`)
    kint: array<vec4<u32>, 64>,
    kfrac: array<vec4<u32>, 64>,
    off: array<vec4<u32>, 64>,
    // each biome's top block, four to a vec4
    tops: array<vec4<u32>, 64>,
}

@group(0) @binding(0) var<storage, read> params: Params;
@group(0) @binding(1) var<storage, read> perms: array<u32>;
// Per tile: origin x, origin z (blocks), level, pool slot.
@group(0) @binding(2) var<storage, read> jobs: array<vec4<i32>>;
// Per sample node (tile * SAMPLES + sample), biome_gen's layout: 256 heights,
// then 256 volatilities; and 256 biome ids.
@group(0) @binding(3) var<storage, read> columns: array<f32>;
@group(0) @binding(4) var<storage, read> col_biomes: array<u32>;
@group(0) @binding(5) var<storage, read_write> tiles: array<u32>;

var<workgroup> wg_spans: atomic<u32>;
var<workgroup> wg_max_spans: atomic<u32>;
var<workgroup> wg_highest: atomic<i32>;
var<workgroup> wg_lowest: atomic<i32>;
var<workgroup> wg_overflow: atomic<u32>;
var<workgroup> wg_water: atomic<u32>;

// One sample point: block column, the density's 2D terms, the level's octaves.
struct Probe {
    x: i32,
    z: i32,
    lv: u32,
    height: f32,
    vol_below: f32,
    vol_above: f32,
    depth: f32,
}

fn octave_at(slot: u32, c: vec3<i32>) -> f32 {
    let ki = params.kint[slot];
    let kf = params.kfrac[slot];
    let ko = params.off[slot];
    let px = phase(c.x, ki.x, kf.x, ko.x);
    let py = phase(c.y, ki.y, kf.y, ko.y);
    let pz = phase(c.z, ki.z, kf.z, ko.z);
    let cell = vec3<i32>(i32(px >> 24u), i32(py >> 24u), i32(pz >> 24u));
    let f = vec3<f32>(f32(px & 0xFFFFFFu), f32(py & 0xFFFFFFu), f32(pz & 0xFFFFFFu))
        * (1.0 / 16777216.0);
    return perlin(slot, cell, f);
}

fn noise_at(n: u32, c: vec3<i32>, used: u32) -> f32 {
    var sum = 0.0;
    for (var k = 0u; k < used; k++) {
        let slot = n * MAX_OCTAVES + k;
        sum += params.w[slot].w * octave_at(slot, c);
    }
    return sum + params.noffs[n];
}

// CubicWorldGen's density at block (x, y, z) of the probe's column: > 0 is solid.
fn density(p: Probe, y: i32) -> f32 {
    let used = params.used[p.lv];
    let c = vec3<i32>(p.x, y, p.z);
    let sel = clamp(noise_at(0u, c, used.x), 0.0, 1.0);
    var mixed = 0.0;
    if sel <= 0.0 {
        mixed = noise_at(1u, c, used.y);
    } else if sel >= 1.0 {
        mixed = noise_at(2u, c, used.z);
    } else {
        let low = noise_at(1u, c, used.y);
        let high = noise_at(2u, c, used.z);
        mixed = low + sel * (high - low);
    }
    let yf = f32(y);
    let vol = select(p.vol_above, p.vol_below, p.height > yf);
    return (mixed + p.depth) * vol + p.height - sign(vol) * yf;
}

fn solid(p: Probe, y: i32) -> bool {
    return density(p, y) > 0.0;
}

// Where the density can change sign along one piece of the column (one
// volatility): above and below it the sign is fixed.
fn piece(h: f32, vol: f32, mlo: f32, mhi: f32) -> vec2<f32> {
    if vol > 0.0 {
        return vec2<f32>(h + mlo * vol, h + mhi * vol);
    }
    if vol < 0.0 {
        // density = M * vol + h + y: zero at y = -h - M * vol.
        return vec2<f32>(-h - mlo * vol, -h - mhi * vol);
    }
    return vec2<f32>(h, h);
}

struct Range {
    lo: i32,
    hi: i32,
    step: i32,
    above: bool,
    below: bool,
}

// The y range the march covers (surface_tile.rs's `Probe::range`).
fn range_of(p: Probe, level: u32) -> Range {
    let m = params.mix[p.lv];
    let mlo = m.x + p.depth;
    let mhi = m.y + p.depth;
    let a = piece(p.height, p.vol_below, mlo, mhi);
    let b = piece(p.height, p.vol_above, mlo, mhi);
    let lo_f = min(min(a.x, b.x), p.height);
    let hi_f = max(max(a.y, b.y), p.height);
    let ymin = f32(params.world.x);
    let ymax = f32(params.world.y);
    var r: Range;
    r.lo = i32(clamp(floor(lo_f - 2.0 - abs(lo_f) * 1.0e-6), ymin, ymax));
    r.hi = i32(clamp(ceil(hi_f + 2.0 + abs(hi_f) * 1.0e-6), ymin, ymax));
    let width = 1i << level;
    var step = clamp(width, 1, STEP_CAP);
    let reach = r.hi - r.lo;
    if reach / step > MAX_STEPS {
        step = (reach + MAX_STEPS - 1) / MAX_STEPS;
    }
    r.step = step;
    r.above = solid(p, r.hi + 1);
    r.below = solid(p, r.lo - 1);
    return r;
}

// Bisects between `lo` (state `lo_solid`) and `hi` above it (the other
// state): returns the last y of `lo`'s state, the boundary being between it
// and the next.
fn bisect(p: Probe, lo_in: i32, hi_in: i32, lo_solid: bool) -> i32 {
    var lo = lo_in;
    var hi = hi_in;
    loop {
        if hi - lo <= 1 {
            break;
        }
        let mid = lo + (hi - lo) / 2;
        if solid(p, mid) == lo_solid {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return lo;
}

// The highest solid block, OPEN_TOP when the column is solid up past the
// range, OPEN_BOTTOM when it has none.
fn find_top(p: Probe, r: Range) -> i32 {
    if r.above {
        return OPEN_TOP;
    }
    var prev = r.hi + 1;
    var y = r.hi;
    loop {
        if solid(p, y) {
            return bisect(p, y, prev, true);
        }
        if y <= r.lo {
            break;
        }
        prev = y;
        y = max(y - r.step, r.lo);
    }
    if r.below {
        return r.lo - 1;
    }
    return OPEN_BOTTOM;
}

fn probe_of(job: vec4<i32>, node: u32, col: u32, x: u32, z: u32, s: u32) -> Probe {
    let level = u32(job.z);
    let j = column_jitter((job.x >> level) + i32(x), (job.y >> level) + i32(z), level, s);
    var p: Probe;
    p.x = job.x + i32(x << level) + j.x;
    p.z = job.y + i32(z << level) + j.y;
    p.lv = level;
    let base = columns[node * 512u + col];
    let vol = columns[node * 512u + 256u + col];
    p.height = base * params.shape.x + params.shape.y;
    p.vol_below = vol * params.shape2.x * params.shape.z + params.shape.w;
    p.vol_above = vol * params.shape.z + params.shape.w;
    let used = params.used[level];
    p.depth = depth_transform(noise_at(3u, vec3<i32>(p.x, 0, p.z), used.w));
    return p;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let tile = wg.x;
    let job = jobs[tile];
    let level = u32(job.z);
    let slot = u32(job.w);
    let x = lid.x;
    let z = lid.y;
    let col = z * 16u + x;
    if col == 0u {
        atomicStore(&wg_spans, 0u);
        atomicStore(&wg_max_spans, 0u);
        atomicStore(&wg_highest, OPEN_BOTTOM);
        atomicStore(&wg_lowest, OPEN_TOP);
        atomicStore(&wg_overflow, 0u);
        atomicStore(&wg_water, 0u);
    }
    workgroupBarrier();

    // The sample with the highest top (the first on ties).
    var best = 0u;
    var best_top = OPEN_BOTTOM;
    for (var s = 0u; s < SAMPLES && SAMPLES > 1u; s++) {
        let p = probe_of(job, tile * SAMPLES + s, col, x, z, s);
        let top = find_top(p, range_of(p, level));
        if s == 0u || top > best_top {
            best = s;
            best_top = top;
        }
    }
    let node = tile * SAMPLES + best;
    let p = probe_of(job, node, col, x, z, best);
    let r = range_of(p, level);

    // March the chosen line: the solid spans from the top down.
    var tops = array<i32, 4>(OPEN_BOTTOM, OPEN_BOTTOM, OPEN_BOTTOM, OPEN_BOTTOM);
    var bottoms = array<i32, 4>(OPEN_BOTTOM, OPEN_BOTTOM, OPEN_BOTTOM, OPEN_BOTTOM);
    var count = 0u;
    var overflow = false;
    var inside = r.above;
    if inside {
        tops[0] = OPEN_TOP;
        count = 1u;
    }
    var prev = r.hi + 1;
    var y = r.hi;
    loop {
        let here = solid(p, y);
        if here != inside {
            if here {
                let t = bisect(p, y, prev, true);
                if count < SPANS {
                    tops[count] = t;
                    count++;
                } else {
                    overflow = true;
                }
            } else {
                bottoms[count - 1u] = bisect(p, y, prev, false) + 1;
            }
            inside = here;
        }
        if y <= r.lo {
            break;
        }
        prev = y;
        y = max(y - r.step, r.lo);
    }
    if inside {
        bottoms[count - 1u] = select(r.lo, OPEN_BOTTOM, r.below);
    } else if r.below {
        if count < SPANS {
            tops[count] = r.lo - 1;
            count++;
        } else {
            overflow = true;
        }
        bottoms[count - 1u] = OPEN_BOTTOM;
    }

    let biome = col_biomes[node * 256u + col];
    let b = min(biome, u32(max(params.world.w, 1)) - 1u);
    let top_block = params.tops[b >> 2u][b & 3u];
    let sea = params.world.z;
    var top0 = OPEN_BOTTOM;
    if count > 0u {
        top0 = tops[0];
    }
    let water = top0 != OPEN_TOP && top0 < sea && sea >= params.world.x && sea <= params.world.y;
    let out = slot * TILE_WORDS + HEADER_WORDS + col * COLUMN_WORDS;
    tiles[out] = count | (u32(water) << 3u) | (u32(overflow) << 4u) | (best << 5u)
        | ((biome & 0xffffu) << 8u);
    for (var i = 0u; i < SPANS; i++) {
        let at = out + 1u + i * 3u;
        if i < count {
            tiles[at] = bitcast<u32>(tops[i]);
            tiles[at + 1u] = bitcast<u32>(bottoms[i]);
            let deep = bottoms[i] != OPEN_BOTTOM && f32(bottoms[i]) < params.shape2.y;
            tiles[at + 2u] = (top_block & 0x1ffffu) | (u32(deep) << 31u);
        } else {
            tiles[at] = bitcast<u32>(OPEN_BOTTOM);
            tiles[at + 1u] = bitcast<u32>(OPEN_BOTTOM);
            tiles[at + 2u] = 0u;
        }
    }

    atomicAdd(&wg_spans, count);
    atomicMax(&wg_max_spans, count);
    if count > 0u {
        atomicMax(&wg_highest, select(top0, params.world.y, top0 == OPEN_TOP));
    }
    if water {
        atomicMax(&wg_highest, sea);
        atomicAdd(&wg_water, 1u);
    }
    if overflow {
        atomicAdd(&wg_overflow, 1u);
    }
    atomicMin(&wg_lowest, r.lo - 1);
    workgroupBarrier();

    if col == 0u {
        let h = slot * TILE_WORDS;
        let lowest = atomicLoad(&wg_lowest);
        var highest = atomicLoad(&wg_highest);
        if highest < lowest {
            highest = lowest;
        }
        let skirt = max(lowest - max(SKIRT_MIN, highest - lowest), params.world.x);
        tiles[h] = bitcast<u32>(job.x);
        tiles[h + 1u] = bitcast<u32>(job.y);
        tiles[h + 2u] = level;
        tiles[h + 3u] = atomicLoad(&wg_max_spans);
        tiles[h + 4u] = bitcast<u32>(highest);
        tiles[h + 5u] = bitcast<u32>(skirt);
        tiles[h + 6u] = atomicLoad(&wg_overflow);
        tiles[h + 7u] = atomicLoad(&wg_water);
        tiles[h + 8u] = atomicLoad(&wg_spans);
        for (var i = 9u; i < HEADER_WORDS; i++) {
            tiles[h + i] = 0u;
        }
    }
}
