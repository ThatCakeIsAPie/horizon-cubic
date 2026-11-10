// Surface tiles (gfx/surface_tile.rs) drawn by vertex pulling. Appended to
// terrain.wgsl and cube_far.wgsl: the fragment stages, light, fog, fade and the
// material lookup (`cf_face`, `cf_materials`, `cf_biomes`) are the far cube
// nodes'. `instance_index` is the tile's pool slot (surface_cull.wgsl writes
// it as the draw's first instance); `vertex_index` picks the column, the span,
// the face and the corner. Every span is a box over its column: its top, its
// bottom (none for the ground, which is open below) and four walls. The
// ground's walls reach down to the neighbouring column's ground, or to the
// tile's skirt at the tile's edge and where the neighbour has no ground; a
// floating span's walls cover it whole unless a neighbouring span does.

const ST_TILE_WORDS: u32 = 3344u;
const ST_HEADER: u32 = 16u;
const ST_COLUMN_WORDS: u32 = 13u;
const ST_OPEN_BOTTOM: i32 = -2147483647 - 1;
const ST_OPEN_TOP: i32 = 2147483647;

struct StFrame {
    planes: array<vec4<f32>, 6>,
    cam_int: vec4<i32>,
    cam_frac: vec4<f32>,
    // now (ms), slots, unused, unused
    info: vec4<u32>,
    // sea level, lowest y, highest y, deepslate below this y
    world: vec4<i32>,
    // water, stone, deepslate, unused
    ids: vec4<u32>,
}

@group(1) @binding(34) var<storage, read> st_tiles: array<u32>;
@group(1) @binding(35) var<storage, read> st_instances: array<vec4<u32>>;
@group(1) @binding(36) var<uniform> st_frame: StFrame;

fn st_y(y: i32) -> f32 {
    return f32(y - st_frame.cam_int.y) - st_frame.cam_frac.y;
}

fn st_column(base: u32, col: u32) -> u32 {
    return base + ST_HEADER + col * ST_COLUMN_WORDS;
}

// The top plane of a span's top block (the world's top for an open top).
fn st_top_plane(top: i32) -> i32 {
    return select(top + 1, st_frame.world.z + 1, top == ST_OPEN_TOP);
}

// Corner `c` (0..3, counter-clockwise seen from outside) of a column face:
// `dir` in Direction order (Down, Up, North, South, West, East), the column's
// footprint `x0..x1`, `z0..z1` (tile-relative blocks) and the camera-relative
// planes `ya..yb`. Returns the position and the UV in blocks.
fn st_corner(dir: u32, c: u32, x0: f32, x1: f32, z0: f32, z1: f32, ya: f32, yb: f32,
    blocks: f32) -> array<vec3<f32>, 2> {
    var p = vec3<f32>(0.0);
    var uv = vec2<f32>(0.0);
    let w = x1 - x0;
    let lu = c == 1u || c == 2u;
    let lv = c >= 2u;
    switch dir {
        case 1u: {
            // Up: (x0,z0) (x0,z1) (x1,z1) (x1,z0)
            p = vec3<f32>(select(x0, x1, lv), yb, select(z0, z1, lu));
            uv = vec2<f32>(select(0.0, w, lv), select(0.0, w, lu));
        }
        case 0u: {
            // Down: (x0,z0) (x1,z0) (x1,z1) (x0,z1)
            p = vec3<f32>(select(x0, x1, lu), ya, select(z0, z1, lv));
            uv = vec2<f32>(select(0.0, w, lu), select(0.0, w, lv));
        }
        case 5u: {
            // East, x = x1: (z0,ya) (z0,yb) (z1,yb) (z1,ya)
            p = vec3<f32>(x1, select(ya, yb, lu), select(z0, z1, lv));
            uv = vec2<f32>(select(w, 0.0, lv), select(blocks, 0.0, lu));
        }
        case 4u: {
            // West, x = x0: (z0,ya) (z1,ya) (z1,yb) (z0,yb)
            p = vec3<f32>(x0, select(ya, yb, lv), select(z0, z1, lu));
            uv = vec2<f32>(select(0.0, w, lu), select(blocks, 0.0, lv));
        }
        case 3u: {
            // South, z = z1: (x0,ya) (x1,ya) (x1,yb) (x0,yb)
            p = vec3<f32>(select(x0, x1, lu), select(ya, yb, lv), z1);
            uv = vec2<f32>(select(0.0, w, lu), select(blocks, 0.0, lv));
        }
        default: {
            // North, z = z0: (x0,ya) (x0,yb) (x1,yb) (x1,ya)
            p = vec3<f32>(select(x0, x1, lv), select(ya, yb, lu), z0);
            uv = vec2<f32>(select(w, 0.0, lv), select(blocks, 0.0, lu));
        }
    }
    return array<vec3<f32>, 2>(p, vec3<f32>(uv, 0.0));
}

fn st_corner_of(k: u32) -> u32 {
    // Two triangles: 0 1 2, 0 2 3.
    var c = k;
    if k >= 3u {
        c = select(0u, k - 2u, k > 3u);
    }
    return c;
}

// Whether a span of the neighbouring column covers `lo..hi` (block planes)
// whole, so a wall there is hidden.
fn st_covered(base: u32, nx: i32, nz: i32, lo: i32, hi: i32) -> bool {
    if nx < 0 || nx > 15 || nz < 0 || nz > 15 {
        return false;
    }
    let at = st_column(base, u32(nz * 16 + nx));
    let count = st_tiles[at] & 7u;
    for (var i = 0u; i < count; i++) {
        let top = st_top_plane(bitcast<i32>(st_tiles[at + 1u + i * 3u]));
        let bottom = bitcast<i32>(st_tiles[at + 2u + i * 3u]);
        if top >= hi && bottom <= lo {
            return true;
        }
    }
    return false;
}

// The top plane of the neighbouring column's ground (its open-bottomed last
// span), or `fallback` outside the tile and where it has none.
fn st_ground(base: u32, nx: i32, nz: i32, fallback: i32) -> i32 {
    if nx < 0 || nx > 15 || nz < 0 || nz > 15 {
        return fallback;
    }
    let at = st_column(base, u32(nz * 16 + nx));
    let count = st_tiles[at] & 7u;
    if count == 0u {
        return fallback;
    }
    let last = at + 1u + (count - 1u) * 3u;
    if bitcast<i32>(st_tiles[last + 1u]) != ST_OPEN_BOTTOM {
        return fallback;
    }
    return st_top_plane(bitcast<i32>(st_tiles[last]));
}

// What crosses to terrain.wgsl's fragment stages is `TerrainWire` (`to_wire`).
@vertex fn vs_surface(@builtin(vertex_index) vertex: u32,
    @builtin(instance_index) slot: u32) -> TerrainWire {
    return to_wire(st_surface_vertex(vertex, slot));
}
@vertex fn vs_surface_water(@builtin(vertex_index) vertex: u32,
    @builtin(instance_index) slot: u32) -> TerrainWire {
    return to_wire(st_water_vertex(vertex, slot));
}

fn st_surface_vertex(vertex: u32, slot: u32) -> TerrainOut {
    let base = slot * ST_TILE_WORDS;
    let level = st_tiles[base + 2u];
    let spans = max(st_tiles[base + 3u], 1u);
    let quad = vertex / 6u;
    let col = quad / (spans * 6u);
    let rest = quad % (spans * 6u);
    let span = rest / 6u;
    let face = rest % 6u;
    let at = st_column(base, col);
    let word = st_tiles[at];
    let count = word & 7u;
    if col >= 256u || span >= count {
        return left_out();
    }
    let s = at + 1u + span * 3u;
    let top = bitcast<i32>(st_tiles[s]);
    let bottom = bitcast<i32>(st_tiles[s + 1u]);
    let blocks_word = st_tiles[s + 2u];
    let biome = (word >> 8u) & 255u;
    let cx = i32(col & 15u);
    let cz = i32(col >> 4u);
    let yb = st_top_plane(top);
    // Faces 0 top, 1 bottom, then the walls North, South, West, East.
    var dirs = array<u32, 6>(1u, 0u, 2u, 3u, 4u, 5u);
    let dir = dirs[face];
    var state = blocks_word & 0x1ffffu;
    var ya = 0;
    if face == 0u {
        if top == ST_OPEN_TOP {
            return left_out();
        }
        ya = yb;
    } else if face == 1u {
        if bottom == ST_OPEN_BOTTOM {
            return left_out();
        }
        ya = bottom;
        state = select(st_frame.ids.y, st_frame.ids.z, (blocks_word >> 31u) == 1u);
    } else {
        var dx = array<i32, 6>(0, 0, 0, 0, -1, 1);
        var dz = array<i32, 6>(0, 0, -1, 1, 0, 0);
        let nx = cx + dx[dir];
        let nz = cz + dz[dir];
        if bottom == ST_OPEN_BOTTOM {
            ya = st_ground(base, nx, nz, bitcast<i32>(st_tiles[base + 5u]));
        } else {
            ya = bottom;
            if st_covered(base, nx, nz, ya, yb) {
                return left_out();
            }
        }
        if ya >= yb {
            return left_out();
        }
        // A step no steeper than a slope of one stands for a slope under the
        // top block; a steeper one is a cliff of rock, as the far cube
        // nodes' stone below the surface.
        if yb - ya > (1i << level) {
            let deep = (ya + (yb - ya) / 2) < st_frame.world.w;
            state = select(st_frame.ids.y, st_frame.ids.z, deep);
        }
    }
    let width = f32(1u << level);
    let x0 = f32(cx) * width;
    let z0 = f32(cz) * width;
    let corner = st_corner(dir, st_corner_of(vertex % 6u), x0, x0 + width, z0, z0 + width,
        st_y(ya), st_y(yb), f32(yb - ya));
    let f = cf_face(state, dir, 0u, biome);
    let instance = st_instances[slot];
    let origin = vec3<f32>(bitcast<f32>(instance.x), 0.0, bitcast<f32>(instance.y));
    // A span under another is roofed over: the sky light of a covered face.
    let light = vec2<u32>(select(240u, 64u, span > 0u), 0u);
    return cf_by_level(terrain_vertex(corner[0], corner[1].xy, f.layer, unorm8x4(f.color), light,
        origin, bitcast<f32>(instance.z)), level);
}

// The water over a column whose highest span lies below the sea, at the sea's
// top plane.
fn st_water_vertex(vertex: u32, slot: u32) -> TerrainOut {
    let base = slot * ST_TILE_WORDS;
    let level = st_tiles[base + 2u];
    let col = vertex / 6u;
    let word = st_tiles[st_column(base, col)];
    if col >= 256u || ((word >> 3u) & 1u) == 0u {
        return left_out();
    }
    let biome = (word >> 8u) & 255u;
    let width = f32(1u << level);
    let x0 = f32(col & 15u) * width;
    let z0 = f32(col >> 4u) * width;
    let y = st_y(st_frame.world.x + 1);
    let corner = st_corner(1u, st_corner_of(vertex % 6u), x0, x0 + width, z0, z0 + width, y, y,
        0.0);
    let f = cf_face(st_frame.ids.x, 1u, 0u, biome);
    let instance = st_instances[slot];
    let origin = vec3<f32>(bitcast<f32>(instance.x), 0.0, bitcast<f32>(instance.y));
    return cf_by_level(terrain_vertex(corner[0], corner[1].xy, f.layer, unorm8x4(f.color),
        vec2<u32>(240u, 0u), origin, bitcast<f32>(instance.z)), level);
}
