// Far-terrain generation for cubic worlds (see cube_gen.rs for the scheme).
// One workgroup per node; one thread per voxel column, walking the column from
// the top down so the surface replacers know what lies above each voxel.

const MAX_OCTAVES: u32 = 16u;
const NODE_VOXELS: u32 = 4096u;
const SLOT_WORDS: u32 = 4352u; // 4096 states, then 256 column biomes

struct Params {
    // height factor, height offset, height variation factor, height variation offset
    height: vec4<f32>,
    // below-average-y factor, filler depth (blocks), water top (sea level + 1), deepslate from y
    misc: vec4<f32>,
    // deepslate to y
    misc2: vec4<f32>,
    // air, stone, deepslate, water
    ids: vec4<u32>,
    biome_count: vec4<u32>,
    noise_offsets: vec4<f32>,
    // xyz: frequency of the octave (per block), w: weight
    octaves: array<vec4<f32>, 64>,
    // top block, filler block
    biomes: array<vec4<u32>, 256>,
}

struct Header {
    // origin x, y, z (blocks), level
    info: vec4<i32>,
    // octaves in use for selector, low, high, depth
    counts: vec4<u32>,
    // voxel layers to compute above the node, pool slot of the node
    extra: vec4<u32>,
    // lattice cell (mod 256) of the octave at the node origin
    cell: array<vec4<i32>, 64>,
    // fraction within that cell
    frac: array<vec4<f32>, 64>,
}

@group(0) @binding(0) var<storage, read> params: Params;
@group(0) @binding(1) var<storage, read> perms: array<u32>;
@group(0) @binding(2) var<storage, read> headers: array<Header>;
@group(0) @binding(3) var<storage, read> columns: array<f32>;
@group(0) @binding(4) var<storage, read> biomes: array<u32>;
@group(0) @binding(5) var<storage, read_write> out: array<u32>;

// GRAD, perm and perlin: cwg_noise.wgsl, which this file is appended to.

// The weighted octave sum of noise `n` at `local` blocks from the node origin.
fn noise_value(node: u32, n: u32, local: vec3<f32>) -> f32 {
    let count = headers[node].counts[n];
    var sum = 0.0;
    for (var k = 0u; k < count; k++) {
        let slot = n * MAX_OCTAVES + k;
        let oc = params.octaves[slot];
        let p = headers[node].frac[slot].xyz + local * oc.xyz;
        let fl = floor(p);
        let cell = headers[node].cell[slot].xyz + vec3<i32>(fl);
        sum += oc.w * perlin(slot, cell, p - fl);
    }
    return sum + params.noise_offsets[n];
}

fn hash_voxel(v: vec3<i32>) -> f32 {
    var h = (u32(v.x) * 0x9E3779B1u) ^ (u32(v.y) * 0x85EBCA6Bu) ^ (u32(v.z) * 0xC2B2AE35u);
    h = h ^ (h >> 15u);
    h = h * 0x2C1B3C6Du;
    h = h ^ (h >> 12u);
    h = h * 0x297A2D39u;
    h = h ^ (h >> 15u);
    return f32(h >> 8u) / 16777216.0;
}

fn sign_of(v: f32) -> f32 {
    return select(select(0.0, -1.0, v < 0.0), 1.0, v > 0.0);
}

// The density at `local` blocks from the node origin, world height `yc`, in a column
// of the given height, depth noise and volatility: solid above 0.
fn density_at(node: u32, local: vec3<f32>, yc: f32, height: f32, depth: f32,
    base_volatility: f32) -> f32 {
    let selector = clamp(noise_value(node, 0u, local), 0.0, 1.0);
    let low = noise_value(node, 1u, local);
    let high = noise_value(node, 2u, local);
    let mixed = low + selector * (high - low);
    var vol_below = 1.0;
    if height > yc {
        vol_below = params.misc.x;
    }
    let vol = base_volatility * vol_below * params.height.z + params.height.w;
    return (mixed + depth) * vol + height - sign_of(vol) * yc;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let node = wg.x;
    let x = lid.x;
    let z = lid.y;
    let level = u32(headers[node].info.w);
    let s = f32(1u << level);
    let origin = headers[node].info.xyz;
    let col = z * 16u + x;
    let base_height = columns[node * 512u + col];
    let base_volatility = columns[node * 512u + 256u + col];
    let slot = headers[node].extra.y;
    let raw_biome = biomes[node * 256u + col];
    out[slot * SLOT_WORDS + NODE_VOXELS + col] = raw_biome;
    let biome = min(raw_biome, params.biome_count.x - 1u);
    let top_block = params.biomes[biome].x;
    let filler_block = params.biomes[biome].y;

    let lx = (f32(x) + 0.5) * s;
    let lz = (f32(z) + 0.5) * s;

    var depth = noise_value(node, 3u, vec3<f32>(lx, 0.0, lz));
    if depth < 0.0 {
        depth = depth * -0.3;
    }
    depth = clamp(depth * 3.0 - 2.0, -2.0, 1.0);
    depth = select(depth / 8.0, depth / 5.6, depth < 0.0);
    depth = depth * (0.2 * 17.0 / 64.0);

    let height = base_height * params.height.x + params.height.y;

    let extra = headers[node].extra.x;
    var jj = i32(15u + extra);
    var run = 0u;
    var seen_air = false;
    let origin_voxel = vec3<i32>(origin.x >> level, origin.y >> level, origin.z >> level);
    loop {
        if jj < 0 {
            break;
        }
        let ly = (f32(jj) + 0.5) * s;
        let yc = f32(origin.y) + ly;
        let local = vec3<f32>(lx, ly, lz);
        let density = density_at(node, local, yc, height, depth, base_volatility);
        let solid = density > 0.0;
        var block = params.ids.x;
        if !solid {
            let water_top = params.misc.z;
            if yc < water_top {
                block = params.ids.w;
            } else if jj < 16 && f32(origin.y) + f32(jj) * s < water_top {
                // The voxel holding the sea's surface, its centre above it: water
                // where the column is open at the sea (cube_gen.rs `water_at_sea`),
                // drawn up to the sea's surface only (cube_far.wgsl).
                let ys = water_top - 0.5;
                let at_sea = vec3<f32>(lx, ys - f32(origin.y), lz);
                if density_at(node, at_sea, ys, height, depth, base_volatility) <= 0.0 {
                    block = params.ids.w;
                }
            }
            run = 0u;
            seen_air = true;
        } else {
            var surface_depth = 1000000u;
            if seen_air {
                surface_depth = run;
            }
            if surface_depth == 0u {
                block = top_block;
            } else if f32(surface_depth) * s < params.misc.y {
                block = filler_block;
            } else {
                block = params.ids.y;
                if yc < params.misc.w {
                    block = params.ids.z;
                } else if yc < params.misc2.x {
                    let t = (params.misc2.x - yc) / (params.misc2.x - params.misc.w);
                    let voxel = vec3<i32>(
                        origin_voxel.x + i32(x),
                        origin_voxel.y + jj,
                        origin_voxel.z + i32(z),
                    );
                    if hash_voxel(voxel) < t {
                        block = params.ids.z;
                    }
                }
            }
            run = run + 1u;
        }
        if jj < 16 {
            out[slot * SLOT_WORDS + (u32(jj) << 8u | z << 4u | x)] = block;
        }
        jj = jj - 1;
    }
}
