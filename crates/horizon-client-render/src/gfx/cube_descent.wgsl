// Sparse descent of the far-terrain octree (see cube_descent.rs for the scheme).
//
// One thread per node of the current level's list. Each node is classified
// with one centre sample of every noise plus the conservative bound of
// `horizon_world::cubic::custom::bounds` (`Analyzer::density_range`), then
// goes to exactly one of: the output as uniform air / uniform rock, the output
// as a leaf, or (split) the next level's list as 8 children.

const MAX_OCTAVES: u32 = 16u;
const KIND_LEAF: u32 = 1u;
const KIND_FINE: u32 = 2u;
const KIND_AIR: u32 = 3u;
const KIND_ROCK: u32 = 4u;
const KIND_MIXED: u32 = 5u;
// Uniform, but around the loaded cubes, where the plan splits whatever the
// class: recorded so the planner finds the class of the children's parents.
const KIND_SPLIT: u32 = 3u;
const PROBE_LEVEL: i32 = 2;

struct Frame {
    // Camera: integer part of each coordinate.
    cam_i: vec4<i32>,
    // Camera: fraction of each coordinate; w is the maximum distance.
    cam_f: vec4<f32>,
    // x: the factor of a node's size the camera must be nearer than to split
    // (DETAIL times the hysteresis scale); y: 1 = classify the list only.
    rule: vec4<f32>,
    // World border, lowest y, highest y, the level that is not split further
    // (such a node is handed over as "fine").
    world: vec4<i32>,
    // The camera's cube; w = 1: nodes meeting the loaded box above the probe
    // level split whatever their class.
    center: vec4<i32>,
    // Loaded box: horizontal and vertical radius in cubes.
    radius: vec4<i32>,
}

struct Level {
    // Octaves in use for selector, low, high, depth at this level.
    used: vec4<u32>,
    // 1.5 times the amplitude of the octaves not in use (and not uploaded).
    skip: vec4<f32>,
    // x: voxel layers the generator computes above a node.
    extra: vec4<u32>,
}

struct Tables {
    // heightFactor, heightOffset, heightVariationFactor, heightVariationOffset
    shape: vec4<f32>,
    // special factor, sea level + 1, gradient bound, value bound
    shape2: vec4<f32>,
    // lowest and highest biome height, lowest and highest volatility
    terrain: vec4<f32>,
    // x: slack on an octave value, y: margin, z: relative padding, w: unused
    slack: vec4<f32>,
    kinks0: vec4<f32>,
    kinks1: vec4<f32>,
    // The four noises' offsets.
    noffs: vec4<f32>,
    levels: array<Level, 32>,
    // xyz: |frequency| of the octave per block; w: signed weight.
    w: array<vec4<f32>, 64>,
    // Fixed-point phase constants per axis (see `phase`).
    kint: array<vec4<u32>, 64>,
    kfrac: array<vec4<u32>, 64>,
    off: array<vec4<u32>, 64>,
}

struct Out {
    count: atomic<u32>,
    overflow: atomic<u32>,
    pad0: u32,
    pad1: u32,
    items: array<vec4<i32>>,
}

struct ListIn {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    items: array<vec4<i32>>,
}

struct ListOut {
    count: atomic<u32>,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    items: array<vec4<i32>>,
}

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> tables: Tables;
@group(0) @binding(2) var<storage, read> perms: array<u32>;
@group(0) @binding(3) var<storage, read_write> out: Out;
@group(1) @binding(0) var<storage, read> in_list: ListIn;
@group(1) @binding(1) var<storage, read_write> next_list: ListOut;

var<private> GRAD: array<vec3<f32>, 16> = array<vec3<f32>, 16>(
    vec3<f32>(1.0, 1.0, 0.0), vec3<f32>(-1.0, 1.0, 0.0),
    vec3<f32>(1.0, -1.0, 0.0), vec3<f32>(-1.0, -1.0, 0.0),
    vec3<f32>(1.0, 0.0, 1.0), vec3<f32>(-1.0, 0.0, 1.0),
    vec3<f32>(1.0, 0.0, -1.0), vec3<f32>(-1.0, 0.0, -1.0),
    vec3<f32>(0.0, 1.0, 1.0), vec3<f32>(0.0, -1.0, 1.0),
    vec3<f32>(0.0, 1.0, -1.0), vec3<f32>(0.0, -1.0, -1.0),
    vec3<f32>(1.0, 1.0, 0.0), vec3<f32>(0.0, -1.0, 1.0),
    vec3<f32>(-1.0, 1.0, 0.0), vec3<f32>(0.0, -1.0, -1.0),
);

// ------------------------------------------------------------- the noise

fn perm(table: u32, i: i32) -> i32 {
    let idx = u32(i) & 255u;
    let word = perms[table * 64u + (idx >> 2u)];
    return i32((word >> ((idx & 3u) * 8u)) & 255u);
}

fn lerp(d: f32, a: f32, b: f32) -> f32 {
    return a + d * (b - a);
}

fn fade(v: f32) -> f32 {
    return v * v * v * (v * (v * 6.0 - 15.0) + 10.0);
}

fn corner(table: u32, row: i32, cz: i32, d: vec3<f32>) -> f32 {
    let g = GRAD[u32(perm(table, row + cz)) & 15u];
    return dot(g, d);
}

// Vanilla ImprovedNoise at a lattice cell and a fraction within it (the same
// function as cube_gen.wgsl's).
fn perlin(table: u32, cell: vec3<i32>, f: vec3<f32>) -> f32 {
    let p = perm(table, cell.x);
    let q = perm(table, cell.x + 1);
    let lo = perm(table, p + cell.y);
    let up = perm(table, p + cell.y + 1);
    let lox = perm(table, q + cell.y);
    let upx = perm(table, q + cell.y + 1);
    let z0 = cell.z;
    let z1 = cell.z + 1;
    let v0 = corner(table, lo, z0, vec3<f32>(f.x, f.y, f.z));
    let v1 = corner(table, lox, z0, vec3<f32>(f.x - 1.0, f.y, f.z));
    let v2 = corner(table, up, z0, vec3<f32>(f.x, f.y - 1.0, f.z));
    let v3 = corner(table, upx, z0, vec3<f32>(f.x - 1.0, f.y - 1.0, f.z));
    let v4 = corner(table, lo, z1, vec3<f32>(f.x, f.y, f.z - 1.0));
    let v5 = corner(table, lox, z1, vec3<f32>(f.x - 1.0, f.y, f.z - 1.0));
    let v6 = corner(table, up, z1, vec3<f32>(f.x, f.y - 1.0, f.z - 1.0));
    let v7 = corner(table, upx, z1, vec3<f32>(f.x - 1.0, f.y - 1.0, f.z - 1.0));
    let a = fade(f.x);
    let b = fade(f.y);
    let c = fade(f.z);
    let l0 = lerp(b, lerp(a, v0, v1), lerp(a, v2, v3));
    let l1 = lerp(b, lerp(a, v4, v5), lerp(a, v6, v7));
    return lerp(c, l0, l1);
}

// The octave's input at block coordinate `c`, `c * scale + offset` modulo 256
// (the permutation table's period), in 2^-24 cells, exactly enough for any
// block coordinate: `scale * 2^24 = kint + kfrac / 2^16` (mod 2^32) and
// `c = ch * 8192 + cl`, so every product fits 32 bits and the sum wraps
// modulo 2^32 = 256 cells. This replaces the f64 origin split of cube_gen's
// headers (the children of a node are made on the GPU, with no CPU turn to
// split them). The error is below 2^-16 * 2^-24 cells per block of |c|.
fn phase(c: i32, kint: u32, kfrac: u32, off: u32) -> u32 {
    let cl = u32(c & 8191);
    let ch = c >> 13u;
    let t2 = (cl * kfrac) >> 16u;
    let t1 = (ch * i32(kfrac)) >> 3u;
    return bitcast<u32>(c) * kint + bitcast<u32>(t1) + t2 + off;
}

fn octave_value(slot: u32, c: vec3<i32>) -> f32 {
    let ki = tables.kint[slot];
    let kf = tables.kfrac[slot];
    let ko = tables.off[slot];
    let px = phase(c.x, ki.x, kf.x, ko.x);
    let py = phase(c.y, ki.y, kf.y, ko.y);
    let pz = phase(c.z, ki.z, kf.z, ko.z);
    let cell = vec3<i32>(
        i32((px >> 24u) & 255u),
        i32((py >> 24u) & 255u),
        i32((pz >> 24u) & 255u),
    );
    let f = vec3<f32>(
        f32(px & 0xFFFFFFu),
        f32(py & 0xFFFFFFu),
        f32(pz & 0xFFFFFFu),
    ) * (1.0 / 16777216.0);
    return perlin(slot, cell, f);
}

// ------------------------------------------------------------- the bound

// Range of noise `n` over the box of centre `c` and half extents `h` (blocks):
// `bounds.rs`'s `noise_range`, with the extra allowance for the phase error.
fn noise_range(n: u32, c: vec3<i32>, h: vec3<f32>, lv: u32) -> vec2<f32> {
    let used = tables.levels[lv].used[n];
    let grad = tables.shape2.z;
    let bound = tables.shape2.w + tables.slack.x;
    let ca = abs(vec3<f32>(c));
    let eps = ca * 1.0e-12 + vec3<f32>(3.0e-7);
    var lo = 0.0;
    var hi = 0.0;
    for (var k = 0u; k < used; k++) {
        let slot = n * MAX_OCTAVES + k;
        let ow = tables.w[slot];
        let p = octave_value(slot, c);
        var reach = ow.x * h.x + ow.z * h.z + eps.x + eps.z;
        if ow.y != 0.0 {
            reach += ow.y * h.y + eps.y;
        }
        let lin = grad * reach + tables.slack.x;
        let a = max(p - lin, -bound);
        let b = min(p + lin, bound);
        if ow.w >= 0.0 {
            lo += ow.w * a;
            hi += ow.w * b;
        } else {
            lo += ow.w * b;
            hi += ow.w * a;
        }
    }
    let skip = tables.levels[lv].skip[n];
    let off = tables.noffs[n];
    return vec2<f32>(lo - skip + off, hi + skip + off);
}

fn depth_transform(raw: f32) -> f32 {
    var d = raw;
    if d < 0.0 {
        d = d * -0.3;
    }
    d = clamp(d * 3.0 - 2.0, -2.0, 1.0);
    if d < 0.0 {
        d = d / 5.6;
    } else if d > 0.0 {
        d = d / 8.0;
    }
    return d * (0.2 * 17.0 / 64.0);
}

fn depth_range(a: f32, b: f32) -> vec2<f32> {
    let ta = depth_transform(a);
    let tb = depth_transform(b);
    var lo = min(ta, tb);
    var hi = max(ta, tb);
    for (var i = 0u; i < 5u; i++) {
        var k = tables.kinks1.x;
        if i < 4u {
            k = tables.kinks0[i];
        }
        if k > a && k < b {
            let v = depth_transform(k);
            lo = min(lo, v);
            hi = max(hi, v);
        }
    }
    return vec2<f32>(lo, hi);
}

// Bounds of the density over the box (min, max, ok): `bounds.rs`'s
// `density_range` for a box of far terrain, with the biome terms from the
// global range of the biome table (every column's height and volatility lies
// in it).
fn density_range(c: vec3<i32>, h: vec3<f32>, ylo: f32, yhi: f32, lv: u32) -> vec3<f32> {
    let sel = noise_range(0u, c, h, lv);
    let low = noise_range(1u, c, h, lv);
    let high = noise_range(2u, c, h, lv);
    let dep = noise_range(3u, c, vec3<f32>(h.x, 0.0, h.z), lv);
    let c0 = clamp(sel.x, 0.0, 1.0);
    let c1 = clamp(sel.y, 0.0, 1.0);
    var m = vec2<f32>(min(low.x, high.x), max(low.y, high.y));
    if c1 <= 0.0 {
        m = low;
    } else if c0 >= 1.0 {
        m = high;
    }
    let dr = depth_range(dep.x, dep.y);
    let a0 = m.x + dr.x;
    let a1 = m.y + dr.y;
    if a0 != a0 || a1 != a1 {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    let hf = tables.shape.x;
    let ho = tables.shape.y;
    let vf = tables.shape.z;
    let vo = tables.shape.w;
    let special = tables.shape2.x;
    let tr = tables.terrain;
    var heights = array<f32, 2>(tr.x * hf + ho, tr.y * hf + ho);
    let hmin = min(heights[0], heights[1]);
    let hmax = max(heights[0], heights[1]);
    let special_possible = hmax > ylo;
    let normal_possible = hmin <= yhi;
    var ks = array<f32, 2>(special, 1.0);
    var nb = 0u;
    if special_possible {
        ks[nb] = special;
        nb++;
    }
    if normal_possible || !special_possible {
        ks[nb] = 1.0;
        nb++;
    }
    var vs = array<f32, 2>(tr.z, tr.w);
    // The signs the volatility can have (bit 0: negative, 1: zero, 2: positive).
    var sign_bits = 0u;
    for (var vi = 0u; vi < 2u; vi++) {
        for (var bi = 0u; bi < nb; bi++) {
            let vol = vs[vi] * ks[bi] * vf + vo;
            if vol > 0.0 {
                sign_bits |= 4u;
            } else if vol < 0.0 {
                sign_bits |= 1u;
            } else {
                sign_bits |= 2u;
            }
        }
    }
    if countOneBits(sign_bits) > 1u {
        sign_bits = 7u;
    }
    var dmin = 3.0e38;
    var dmax = -3.0e38;
    var mag = 0.0;
    let ymag = max(abs(ylo), abs(yhi));
    for (var hi = 0u; hi < 2u; hi++) {
        let height = heights[hi];
        for (var vi = 0u; vi < 2u; vi++) {
            for (var bi = 0u; bi < nb; bi++) {
                let vol = vs[vi] * ks[bi] * vf + vo;
                let p0 = a0 * vol;
                let p1 = a1 * vol;
                let pmin = min(p0, p1);
                let pmax = max(p0, p1);
                for (var idx = 0u; idx < 3u; idx++) {
                    if ((sign_bits >> idx) & 1u) == 0u {
                        continue;
                    }
                    var t0 = 0.0;
                    var t1 = 0.0;
                    if idx == 2u {
                        t0 = -yhi;
                        t1 = -ylo;
                    } else if idx == 0u {
                        t0 = ylo;
                        t1 = yhi;
                    }
                    dmin = min(dmin, pmin + height + t0);
                    dmax = max(dmax, pmax + height + t1);
                    mag = max(mag, abs(pmin) + abs(pmax) + abs(height) + ymag);
                }
            }
        }
    }
    if dmin > dmax {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    let pad = tables.slack.y + tables.slack.z * mag;
    return vec3<f32>(dmin - pad, dmax + pad, 1.0);
}

// What the generator makes of the node, as far as the bound proves: uniform
// air / water, uniform rock, or neither. The box covers the node and the voxel
// layers the generator looks at above it (so the node's top voxels know they
// are not under air).
fn classify_node(key: vec3<i32>, level: u32) -> u32 {
    let side = i32(16) << level;
    let s = i32(1) << level;
    let extra = i32(tables.levels[level].extra.x);
    let tall = side + extra * s;
    let origin = key * side;
    let c = vec3<i32>(origin.x + side / 2, origin.y + tall / 2, origin.z + side / 2);
    let h = vec3<f32>(f32(side) * 0.5, f32(tall) * 0.5, f32(side) * 0.5);
    let ylo = f32(origin.y);
    let yhi = f32(origin.y + tall);
    let r = density_range(c, h, ylo, yhi, level);
    if r.z < 0.5 {
        return KIND_MIXED;
    }
    if r.y <= 0.0 {
        // Water below the sea, air above it: uniform unless the sea level
        // crosses the node (cube_gen's `node_class`).
        let sea = tables.shape2.y;
        let fs = f32(s);
        let below_low = (ylo + 0.5 * fs) < sea;
        let below_high = (ylo + 15.5 * fs) < sea;
        if below_low == below_high {
            return KIND_AIR;
        }
        return KIND_MIXED;
    }
    if r.x > 0.0 {
        return KIND_ROCK;
    }
    return KIND_MIXED;
}

// ------------------------------------------------------------- the descent

fn within(c: i32, size: i32, lo: i32, hi: i32) -> bool {
    let at = c * size;
    return at <= hi && at + size > lo;
}

fn meets(c: i32, n: i32, level: u32, radius: i32) -> bool {
    let side = i32(1) << level;
    let lo = n * side;
    return lo <= c + radius && lo + side > c - radius;
}

fn emit(node: vec4<i32>, kind: u32) {
    let idx = atomicAdd(&out.count, 1u);
    if idx >= arrayLength(&out.items) {
        atomicOr(&out.overflow, 2u);
        return;
    }
    out.items[idx] = vec4<i32>(node.xyz, i32(u32(node.w) | (kind << 8u)));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = min(in_list.count, arrayLength(&in_list.items));
    if i >= n {
        return;
    }
    let node = in_list.items[i];
    let level = u32(node.w);
    if frame.rule.y > 0.5 {
        let kind = classify_node(node.xyz, level);
        out.items[i] = vec4<i32>(node.xyz, i32(level | (kind << 8u)));
        return;
    }
    let size = i32(16) << level;
    let lo = node.xyz * size;
    let border = frame.world.x;
    if !(within(node.x, size, -border, border)
        && within(node.z, size, -border, border)
        && within(node.y, size, frame.world.y, frame.world.z)) {
        return;
    }
    // The distance from the camera to the nearest point of the node.
    let gap_lo = vec3<f32>(lo - frame.cam_i.xyz) - frame.cam_f.xyz;
    let gap_hi = vec3<f32>(frame.cam_i.xyz - (lo + size)) + frame.cam_f.xyz;
    let gap = max(max(gap_lo, gap_hi), vec3<f32>(0.0));
    let dist = length(gap);
    if dist > frame.cam_f.w {
        return;
    }
    let forced = frame.center.w != 0
        && i32(level) > PROBE_LEVEL
        && meets(frame.center.x, node.x, level, frame.radius.x)
        && meets(frame.center.y, node.y, level, frame.radius.y)
        && meets(frame.center.z, node.z, level, frame.radius.x);
    let kind = classify_node(node.xyz, level);
    if kind == KIND_AIR || kind == KIND_ROCK {
        if !forced {
            emit(node, kind);
            return;
        }
        emit(node, kind + KIND_SPLIT);
    }
    let split = forced || (level > 0u && dist < frame.rule.x * f32(size));
    if !split {
        emit(node, KIND_LEAF);
        return;
    }
    if i32(level) <= frame.world.w {
        emit(node, KIND_FINE);
        return;
    }
    let base = atomicAdd(&next_list.count, 8u);
    if base + 8u > arrayLength(&next_list.items) {
        atomicOr(&out.overflow, 1u);
        return;
    }
    for (var j = 0; j < 8; j++) {
        next_list.items[base + u32(j)] = vec4<i32>(
            node.x * 2 + (j & 1),
            node.y * 2 + (j >> 2),
            node.z * 2 + ((j >> 1) & 1),
            node.w - 1,
        );
    }
}
