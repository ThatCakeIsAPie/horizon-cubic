// Far-terrain columns (height, volatility, biome) on the GPU: see biome_gen.rs.
// One workgroup per node, one thread per voxel column. Levels up to
// BLEND_MAX_LEVEL first fill a workgroup grid of quart biomes (26.3's climate
// at sea level, nearest parameter point in the flattened R-tree), then blend
// the biomes' terrain over 5x5 quart windows at the grid corners and
// interpolate; coarser levels take the column's own biome.

const MAX_LAYERS: u32 = 24u;
const SLOTS: u32 = 168u; // 7 groups of MAX_LAYERS
const GRID_CELLS: u32 = 1296u; // 36 x 36
const BLEND_MAX_LEVEL: u32 = 3u;
const STACK: u32 = 192u;

struct Params {
    // x: biomes in the terrain table, y: word offset of that table in `data`,
    // z: 1 = coarse columns sampled at a jittered point (`column_jitter`)
    info: vec4<u32>,
    // x: the climate noises' xz scale (0.25)
    scalars: vec4<f32>,
    // per slot: x layer frequency, y amplitude, z xz scale * frequency
    layers: array<vec4<f32>, 168>,
}

struct Header {
    // origin x, y, z (blocks), level
    info: vec4<i32>,
    // layers in use per group (7 groups, 8 words)
    counts: array<vec4<u32>, 2>,
    // lattice cell (mod 256) of each layer at the node origin, per axis
    cell: array<vec4<i32>, 168>,
    // fraction within that cell
    frac: array<vec4<f32>, 168>,
}

@group(0) @binding(0) var<storage, read> params: Params;
@group(0) @binding(1) var<storage, read> perms: array<u32>;
@group(0) @binding(2) var<storage, read> headers: array<Header>;
// 4 vec4 per tree node: (min0 max0 min1 max1) (min2 max2 min3 max3)
// (min4 max4 min5 max5) (min6 max6 first-child-or-entry count)
@group(0) @binding(3) var<storage, read> tree: array<vec4<i32>>;
// entry -> biome id words, then the terrain table (base, variation) as f32 bits
@group(0) @binding(4) var<storage, read> data: array<u32>;
@group(0) @binding(5) var<storage, read_write> out_columns: array<f32>;
@group(0) @binding(6) var<storage, read_write> out_biomes: array<u32>;

var<workgroup> grid: array<u32, 1296>;

// GRAD, perm and perlin: cwg_noise.wgsl, which this file is appended to.

fn group_count(node: u32, g: u32) -> u32 {
    return headers[node].counts[g >> 2u][g & 3u];
}

// One layer at `local`, the layer's input minus the node origin's part.
fn layer_value(node: u32, slot: u32, local: vec3<f32>) -> f32 {
    let p = headers[node].frac[slot].xyz + local;
    let fl = floor(p);
    let cell = headers[node].cell[slot].xyz + vec3<i32>(fl);
    return perlin(slot, cell, p - fl);
}

// Vanilla shift_a: offset noise at (x/4, 0, z/4), times 4.
fn shift_a(node: u32, lx: f32, lz: f32) -> f32 {
    let count = group_count(node, 0u);
    var sum = 0.0;
    for (var k = 0u; k < count; k++) {
        let oc = params.layers[k];
        sum += oc.y * layer_value(node, k, vec3<f32>(lx * oc.z, 0.0, lz * oc.z));
    }
    return sum * 4.0;
}

// Vanilla shift_b: offset noise at (z/4, x/4, 0), times 4.
fn shift_b(node: u32, lx: f32, lz: f32) -> f32 {
    let count = group_count(node, 1u);
    var sum = 0.0;
    for (var k = 0u; k < count; k++) {
        let slot = MAX_LAYERS + k;
        let oc = params.layers[slot];
        sum += oc.y * layer_value(node, slot, vec3<f32>(lz * oc.z, lx * oc.z, 0.0));
    }
    return sum * 4.0;
}

// A shifted climate noise at the shifted xz (blocks, relative to the node
// origin): the noise samples (x*xz_scale + shift_x) * frequency, y = 0.
fn field(node: u32, g: u32, px: f32, pz: f32) -> f32 {
    let count = group_count(node, g);
    var sum = 0.0;
    for (var k = 0u; k < count; k++) {
        let slot = g * MAX_LAYERS + k;
        let oc = params.layers[slot];
        sum += oc.y * layer_value(node, slot, vec3<f32>(px * oc.x, 0.0, pz * oc.x));
    }
    return sum;
}

fn quantize(v: f32) -> i32 {
    return i32(v * 10000.0);
}

// Interval distance of vanilla's Parameter.
fn pd(lo: i32, hi: i32, v: i32) -> u32 {
    let above = v - hi;
    if above > 0 {
        return u32(above);
    }
    return u32(max(lo - v, 0));
}

fn sq(d: u32) -> u32 {
    let c = min(d, 65535u);
    return c * c;
}

fn adds(a: u32, b: u32) -> u32 {
    let s = a + b;
    return select(s, 0xFFFFFFFFu, s < a);
}

// Squared distance from tree node `id`'s box to the target point
// (temperature, humidity, continentalness, erosion | depth, weirdness, offset).
fn node_dist(id: u32, a: vec4<i32>, b: vec3<i32>) -> u32 {
    let n0 = tree[id * 4u];
    let n1 = tree[id * 4u + 1u];
    let n2 = tree[id * 4u + 2u];
    let n3 = tree[id * 4u + 3u];
    var d = sq(pd(n0.x, n0.y, a.x));
    d = adds(d, sq(pd(n0.z, n0.w, a.y)));
    d = adds(d, sq(pd(n1.x, n1.y, a.z)));
    d = adds(d, sq(pd(n1.z, n1.w, a.w)));
    d = adds(d, sq(pd(n2.x, n2.y, b.x)));
    d = adds(d, sq(pd(n2.z, n2.w, b.y)));
    d = adds(d, sq(pd(n3.x, n3.y, b.z)));
    return d;
}

// Climate.RTree.search without a previous result: depth first in child
// order, a node entered only while the best fitness is strictly above its
// lower bound, a leaf taking the lead when it is strictly closer. Returns the
// entry index.
fn search(a: vec4<i32>, b: vec3<i32>) -> u32 {
    var stack: array<u32, 192>;
    var sp = 1u;
    stack[0] = 0u;
    var best = 0u;
    var best_fit = 0xFFFFFFFFu;
    loop {
        if sp == 0u {
            break;
        }
        sp = sp - 1u;
        let id = stack[sp];
        let lower = node_dist(id, a, b);
        if id != 0u && !(best_fit > lower) {
            continue;
        }
        let link = tree[id * 4u + 3u];
        let count = u32(link.w);
        if count == 0u {
            best = u32(link.z);
            best_fit = lower;
            continue;
        }
        var c = count;
        loop {
            if c == 0u {
                break;
            }
            c = c - 1u;
            if sp < STACK {
                stack[sp] = u32(link.z) + c;
                sp = sp + 1u;
            }
        }
    }
    return best;
}

// The biome id at block offset (lx, lz) from the node origin (a quart corner):
// climate at sea level (depth 0), then the nearest parameter point.
fn biome_at(node: u32, lx: f32, lz: f32) -> u32 {
    let xz = params.scalars.x;
    let px = lx * xz + shift_a(node, lx, lz);
    let pz = lz * xz + shift_b(node, lx, lz);
    let temperature = quantize(field(node, 2u, px, pz));
    let humidity = quantize(field(node, 3u, px, pz));
    let continentalness = quantize(field(node, 4u, px, pz));
    let erosion = quantize(field(node, 5u, px, pz));
    let weirdness = quantize(field(node, 6u, px, pz));
    let entry = search(
        vec4<i32>(temperature, humidity, continentalness, erosion),
        vec3<i32>(0, weirdness, 0),
    );
    return data[entry];
}

// (base height, height variation) of a biome id.
fn terrain(biome: u32) -> vec2<f32> {
    let i = min(biome, params.info.x - 1u);
    let o = params.info.y + i * 2u;
    return vec2<f32>(bitcast<f32>(data[o]), bitcast<f32>(data[o + 1u]));
}

// CubicWorldGen's weighted average of the terrain over the 5x5 quarts around
// grid cell (gx, gz): (height, variation) before conversion.
fn blend_corner(gx: i32, gz: i32, width: i32) -> vec2<f32> {
    let center = terrain(grid[u32(gz * width + gx)]).x;
    var hs = 0.0;
    var vs = 0.0;
    var ws = 0.0;
    for (var j = -2; j <= 2; j++) {
        for (var i = -2; i <= 2; i++) {
            let t = terrain(grid[u32((gz + j) * width + gx + i)]);
            var w = abs(10.0 / sqrt(f32(i * i + j * j) + 0.2) / (t.x + 2.0));
            if t.x > center {
                w = w * 0.5;
            }
            hs += t.x * w;
            vs += t.y * w;
            ws += w;
        }
    }
    return vec2<f32>(hs / ws, vs / ws);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let node = wg.x;
    let tid = lid.y * 16u + lid.x;
    let level = u32(headers[node].info.w);
    let s = 1i << level;
    let blend = level <= BLEND_MAX_LEVEL;

    // Quart grid of biomes: cells rel_min .. rel_min + width - 1 (quarts from
    // the node origin), enough for every column's two corners and their
    // 5x5 windows.
    var rel_min = 0;
    var width = 1;
    if blend {
        rel_min = ((s >> 1u) >> 2u) - 2;
        let q0max = (15 * s + (s >> 1u)) >> 2u;
        width = q0max + 3 - rel_min + 1;
    }
    let total = select(0, width * width, blend);
    for (var i = i32(tid); i < total; i += 256) {
        let gi = i % width;
        let gj = i / width;
        grid[i] = biome_at(node, f32((rel_min + gi) * 4), f32((rel_min + gj) * 4));
    }
    workgroupBarrier();

    let x = i32(lid.x);
    let z = i32(lid.y);
    var biome = 0u;
    var hv = vec2<f32>(0.0, 0.0);
    if blend {
        let x0 = x * s + (s >> 1u);
        let z0 = z * s + (s >> 1u);
        let gx = (x0 >> 2u) - rel_min;
        let gz = (z0 >> 2u) - rel_min;
        biome = grid[u32(gz * width + gx)];
        let c00 = blend_corner(gx, gz, width);
        if (x0 & 3) == 0 && (z0 & 3) == 0 {
            hv = c00;
        } else {
            let tx = f32(x0 & 3) * 0.25;
            let tz = f32(z0 & 3) * 0.25;
            let c10 = blend_corner(gx + 1, gz, width);
            let c01 = blend_corner(gx, gz + 1, width);
            let c11 = blend_corner(gx + 1, gz + 1, width);
            hv = mix(mix(c00, c10, vec2<f32>(tx)), mix(c01, c11, vec2<f32>(tx)), vec2<f32>(tz));
        }
    } else {
        // The column's centre, or with `jitter` (params.info.z) a hashed quart
        // of it (cwg_noise.wgsl's `column_jitter`, the sample index from the
        // node's y).
        var at = vec2<i32>(s >> 1u, s >> 1u);
        if params.info.z != 0u {
            let o = headers[node].info;
            at = column_jitter((o.x >> level) + x, (o.z >> level) + z, level,
                u32(o.y >> (level + 4u)));
        }
        biome = biome_at(node, f32(x * s + at.x), f32(z * s + at.y));
        hv = terrain(biome);
    }
    let col = tid;
    out_columns[node * 512u + col] = hv.x * (17.0 / 64.0) - 1.0 / 256.0;
    out_columns[node * 512u + 256u + col] = hv.y * 2.4 + 4.0 / 15.0;
    out_biomes[node * 256u + col] = biome;
}
