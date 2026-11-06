// The noise maths the far-terrain compute shaders share (cube_gen.wgsl,
// cube_descent.wgsl, biome_gen.wgsl, surface_gen.wgsl): each is assembled as
// this file followed by its own, and declares the `perms` storage buffer
// (`array<u32>`, 64 words per permutation table, four table bytes a word)
// that `perm` reads.

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

// Vanilla ImprovedNoise at a lattice cell and a fraction within it.
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
// modulo 2^32 = 256 cells (cube_descent.rs's `phase_constants`). The error is
// below 2^-16 * 2^-24 cells per block of |c|.
fn phase(c: i32, kint: u32, kfrac: u32, off: u32) -> u32 {
    let cl = u32(c & 8191);
    let ch = c >> 13u;
    let t2 = (cl * kfrac) >> 16u;
    let t1 = (ch * i32(kfrac)) >> 3u;
    return bitcast<u32>(c) * kint + bitcast<u32>(t1) + t2 + off;
}

// CubicWorldGen's depth transform: the raw 2D depth noise to the term added to
// the noise mix.
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

fn jitter_mix(v: u32) -> u32 {
    var h = v;
    h = h ^ (h >> 16u);
    h = h * 0x7FEB352Du;
    h = h ^ (h >> 15u);
    h = h * 0x846CA68Bu;
    h = h ^ (h >> 16u);
    return h;
}

// Where a column of `1 << level` blocks is sampled (biome_gen.rs's
// `column_jitter`): a hashed offset from its lowest corner, in whole quarts
// (4 blocks), so sample points do not line up with the noises' lattices as
// the centres of power-of-two columns do. `cx`, `cz`: the column's index at
// `level` (block coordinate >> level); `seed`: which sample of the column.
fn column_jitter(cx: i32, cz: i32, level: u32, seed: u32) -> vec2<i32> {
    let quarts = max((1u << level) >> 2u, 1u);
    let h = jitter_mix((bitcast<u32>(cx) * 0x9E3779B1u)
        ^ jitter_mix(bitcast<u32>(cz) ^ (level << 24u) ^ (seed * 0x85EBCA77u)));
    let g = jitter_mix(h ^ 0x68E31DA4u);
    return vec2<i32>(i32(h & (quarts - 1u)) * 4, i32(g & (quarts - 1u)) * 4);
}
