// Culls the far cube nodes against the frustum and the camera's side of each
// direction, writes their instance data (camera-relative origin and fade code,
// as gfx/lod.rs `instance`) and compacts one indirect draw per (node, direction,
// pass): non-indexed, six vertices a quad, the node in `first_instance`.
//
// Hook for Hi-Z occlusion: gpu_occlusion.wgsl's test of the node's box against the
// previous frame's depth pyramid goes where `visible` is decided; not wired yet.

const NODE_WORDS: u32 = 32u;
const NOT_SET: u32 = 0xffffffffu;
const FADE_MS: f32 = 600.0;
const TILE_STEP: f32 = 0.03125;

struct Frame {
    planes: array<vec4<f32>, 6>,
    cam_int: vec4<i32>,
    cam_frac: vec4<f32>,
    now: u32,
    nodes: u32,
    pad0: u32,
    pad1: u32,
}
@group(0) @binding(0) var<uniform> frame: Frame;
// Per node: origin xyz (blocks), level; shown ms, dying ms, flags (bit 0 live, bit 1 alone),
// spare; twelve (first quad, quads) ranges, index dir * 2 + translucent.
@group(0) @binding(1) var<storage, read> nodes: array<u32>;
@group(0) @binding(2) var<storage, read_write> instances: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> solid_args: array<u32>;
@group(0) @binding(4) var<storage, read_write> translucent_args: array<u32>;
// Draws written per pass; the count draws read them.
@group(0) @binding(5) var<storage, read_write> counts: array<atomic<u32>, 2>;

fn elapsed(now: u32, since: u32) -> f32 {
    return f32(select(0u, now - since, now > since)) / FADE_MS;
}
fn fade_in(shown: u32, level: u32, now: u32) -> u32 {
    let step = u32(elapsed(now, shown) * 64.0) + 1u;
    if step >= 64u {
        return 0u;
    }
    return 1u + level * 129u + step;
}
fn fade_out(since: u32, level: u32, now: u32) -> u32 {
    return 1u + level * 129u + 64u + min(u32(elapsed(now, since) * 64.0) + 1u, 64u);
}

@compute @workgroup_size(64)
fn cull_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let node = id.x;
    if node >= frame.nodes {
        return;
    }
    let base = node * NODE_WORDS;
    if (nodes[base + 6u] & 1u) == 0u {
        return;
    }
    let shown = nodes[base + 4u];
    if shown == NOT_SET {
        return;
    }
    let level = nodes[base + 3u];
    let origin = vec3<i32>(bitcast<i32>(nodes[base]), bitcast<i32>(nodes[base + 1u]),
        bitcast<i32>(nodes[base + 2u]));
    let lo = vec3<f32>(origin - frame.cam_int.xyz) - frame.cam_frac.xyz;
    let hi = lo + f32(16u << level);
    var visible = true;
    for (var i = 0u; i < 6u; i++) {
        let pl = frame.planes[i];
        let far = select(lo, hi, pl.xyz >= vec3<f32>(0.0));
        if dot(pl.xyz, far) + pl.w < 0.0 {
            visible = false;
        }
    }
    if !visible {
        return;
    }
    var code = fade_in(shown, level, frame.now);
    let dying = nodes[base + 5u];
    if dying != NOT_SET && (nodes[base + 6u] & 2u) != 0u {
        code = fade_out(dying, level, frame.now);
    }
    // A node faded out entirely draws nothing.
    if code != 0u && (code - 1u) % 129u == 128u {
        return;
    }
    let w = select(-(f32(code) + TILE_STEP), TILE_STEP, code == 0u);
    instances[node * 2u] = vec4<f32>(lo, w);
    instances[node * 2u + 1u] = vec4<f32>(f32(1u << level), 0.0, 0.0, 0.0);
    // The camera sits at the origin: a direction shows only when it can face it.
    // Down, Up, North, South, West, East.
    var front = array<bool, 6>(
        0.0 < hi.y, 0.0 > lo.y, 0.0 < hi.z, 0.0 > lo.z, 0.0 < hi.x, 0.0 > lo.x);
    for (var dir = 0u; dir < 6u; dir++) {
        if !front[dir] {
            continue;
        }
        for (var pass_ = 0u; pass_ < 2u; pass_++) {
            let first = nodes[base + 8u + (dir * 2u + pass_) * 2u];
            let count = nodes[base + 9u + (dir * 2u + pass_) * 2u];
            if count == 0u {
                continue;
            }
            let draw = atomicAdd(&counts[pass_], 1u);
            let at = draw * 4u;
            if pass_ == 0u {
                solid_args[at] = count * 6u;
                solid_args[at + 1u] = 1u;
                solid_args[at + 2u] = first * 6u;
                solid_args[at + 3u] = node;
            } else {
                translucent_args[at] = count * 6u;
                translucent_args[at + 1u] = 1u;
                translucent_args[at + 2u] = first * 6u;
                translucent_args[at + 3u] = node;
            }
        }
    }
}
