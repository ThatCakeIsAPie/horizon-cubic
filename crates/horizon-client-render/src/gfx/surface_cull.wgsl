// Culls the surface tiles (gfx/surface_tile.rs) against the frustum, writes
// their instance data (camera-relative origin and fade code, as cube_cull.wgsl
// does for the far cube nodes) and compacts one indirect draw per tile and
// pass: non-indexed, the tile's pool slot in `first_instance`. A tile draws
// 256 columns x (its most spans) x 6 faces x 6 vertices; faces a column does
// not have collapse in the vertex shader (surface_draw.wgsl).

const TILE_WORDS: u32 = 3344u;
const TABLE_WORDS: u32 = 4u;
const NOT_SET: u32 = 0xffffffffu;
const FADE_MS: f32 = 600.0;
const TILE_STEP: f32 = 0.03125;

struct Frame {
    planes: array<vec4<f32>, 6>,
    cam_int: vec4<i32>,
    cam_frac: vec4<f32>,
    // now (ms), slots, unused, unused
    info: vec4<u32>,
    // sea level, lowest y, highest y, unused
    world: vec4<i32>,
    // water, stone, deepslate, unused
    ids: vec4<u32>,
}
@group(0) @binding(0) var<uniform> frame: Frame;
// Per slot: flags (bit 0 live, bit 1 fading out), shown ms, dying ms, spare.
@group(0) @binding(1) var<storage, read> table: array<u32>;
@group(0) @binding(2) var<storage, read> tiles: array<u32>;
// Per slot: camera-relative origin x, z and the instance word (f32 bits).
@group(0) @binding(3) var<storage, read_write> instances: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read_write> solid_args: array<u32>;
@group(0) @binding(5) var<storage, read_write> water_args: array<u32>;
@group(0) @binding(6) var<storage, read_write> counts: array<atomic<u32>, 2>;

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
    let slot = id.x;
    if slot >= frame.info.y {
        return;
    }
    let t = slot * TABLE_WORDS;
    let flags = table[t];
    if (flags & 1u) == 0u {
        return;
    }
    let shown = table[t + 1u];
    if shown == NOT_SET {
        return;
    }
    let h = slot * TILE_WORDS;
    let origin = vec2<i32>(bitcast<i32>(tiles[h]), bitcast<i32>(tiles[h + 1u]));
    let level = tiles[h + 2u];
    let spans = tiles[h + 3u];
    let highest = bitcast<i32>(tiles[h + 4u]);
    let skirt = bitcast<i32>(tiles[h + 5u]);
    let water = tiles[h + 7u];
    let size = f32(16u << level);
    let lo = vec3<f32>(
        f32(origin.x - frame.cam_int.x) - frame.cam_frac.x,
        f32(skirt - frame.cam_int.y) - frame.cam_frac.y,
        f32(origin.y - frame.cam_int.z) - frame.cam_frac.z,
    );
    let hi = vec3<f32>(lo.x + size, f32(highest + 1 - frame.cam_int.y) - frame.cam_frac.y, lo.z + size);
    for (var i = 0u; i < 6u; i++) {
        let pl = frame.planes[i];
        let far = select(lo, hi, pl.xyz >= vec3<f32>(0.0));
        if dot(pl.xyz, far) + pl.w < 0.0 {
            return;
        }
    }
    var code = fade_in(shown, level, frame.info.x);
    let dying = table[t + 2u];
    if dying != NOT_SET && (flags & 2u) != 0u {
        code = fade_out(dying, level, frame.info.x);
    }
    // A tile faded out entirely draws nothing.
    if code != 0u && (code - 1u) % 129u == 128u {
        return;
    }
    let w = select(-(f32(code) + TILE_STEP), TILE_STEP, code == 0u);
    instances[slot] = vec4<u32>(bitcast<u32>(lo.x), bitcast<u32>(lo.z), bitcast<u32>(w), 0u);
    if spans > 0u {
        let draw = atomicAdd(&counts[0], 1u);
        solid_args[draw * 4u] = 256u * spans * 36u;
        solid_args[draw * 4u + 1u] = 1u;
        solid_args[draw * 4u + 2u] = 0u;
        solid_args[draw * 4u + 3u] = slot;
    }
    if water > 0u {
        let draw = atomicAdd(&counts[1], 1u);
        water_args[draw * 4u] = 256u * 6u;
        water_args[draw * 4u + 1u] = 1u;
        water_args[draw * 4u + 2u] = 0u;
        water_args[draw * 4u + 3u] = slot;
    }
}
