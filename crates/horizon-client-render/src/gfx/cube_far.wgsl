// Far cube nodes drawn from the GPU mesher's quads (gfx/cube_mesh.rs, cube_far.rs).
// Appended to terrain.wgsl, whose `terrain_vertex` and fragment stages shade them, so
// a node looks as the far-tile pipeline draws it: the same sprites, tints, light, fog,
// fade and emission. The vertex shader pulls a quad's record by `vertex_index`
// (six vertices a quad, no vertex buffer); `instance_index` is the node, written by
// the culling pass into the indirect draw's first instance.
@group(1) @binding(30) var<storage, read> cf_quads: array<vec2<u32>>;
// Per node two vec4: camera-relative origin and the instance word (gfx/lod.rs
// `instance`); then the voxel size in blocks and the sea's surface in node-local
// blocks.
@group(1) @binding(31) var<storage, read> cf_instances: array<vec4<f32>>;
@group(1) @binding(32) var<storage, read> cf_materials: array<u32>;
// Per biome: grass, foliage, dry foliage and water colours (0xRRGGBB).
@group(1) @binding(33) var<storage, read> cf_biomes: array<u32>;

// Coarser far terrain is drawn a hair farther, by level (depth is reversed:
// nearer is larger). A coarse node or tile stays until everything planned in
// its place is in, so it stands over the finer nodes that came in for as long
// as the rest takes (tens of seconds through a spyglass); their water and flat
// ground lie in its very planes, and the two fought. The finer now win. 2e-5
// of the distance a level: far over the depth buffer's precision, far under
// anything that shows where the levels only meet.
const CF_LEVEL_DEPTH: f32 = 2e-5;
fn cf_by_level(drawn: TerrainOut, level: u32) -> TerrainOut {
    var out = drawn;
    out.clip.z *= 1.0 - CF_LEVEL_DEPTH * f32(level);
    return out;
}

fn cf_shade(dir: u32) -> f32 {
    switch dir {
        case 0u: { return 0.5; }
        case 1u: { return 1.0; }
        case 2u, 3u: { return 0.8; }
        default: { return 0.6; }
    }
}

@vertex fn vs_cube_quads(@builtin(vertex_index) vertex: u32,
    @builtin(instance_index) node: u32) -> TerrainOut {
    let qi = vertex / 6u;
    let k = vertex % 6u;
    var corner = k;
    if k >= 3u {
        corner = select(0u, k - 2u, k > 3u);
    }
    let q = cf_quads[qi];
    let x = q.x & 15u;
    let y = (q.x >> 4u) & 15u;
    let z = (q.x >> 8u) & 15u;
    let w = ((q.x >> 12u) & 15u) + 1u;
    let h = ((q.x >> 16u) & 15u) + 1u;
    let dir = (q.x >> 20u) & 7u;
    let shaded = (q.x >> 23u) & 1u;
    let overlay = (q.x >> 24u) & 1u;
    let state = q.y & 0x1ffffu;
    let biome = (q.y >> 17u) & 255u;
    let instance = cf_instances[node * 2u];
    let extent = cf_instances[node * 2u + 1u];
    let scale = extent.x;

    let axis = dir >> 1u;
    let up = dir & 1u;
    var a0 = x;
    var b0 = z;
    var plane = y + up;
    if axis == 1u {
        b0 = y;
        plane = z + up;
    } else if axis == 2u {
        a0 = z;
        b0 = y;
        plane = x + up;
    }
    let u0 = f32(a0) * scale;
    let u1 = f32(a0 + w) * scale;
    let v0 = f32(b0) * scale;
    let v1 = f32(b0 + h) * scale;
    let p = f32(plane) * scale;
    var c = corner;
    if dir == 0u || dir == 3u || dir == 5u {
        c = 3u - c;
    }
    var hi_a = c == 1u || c == 2u;
    if axis == 2u {
        hi_a = c == 0u || c == 3u;
    }
    let hi_b = c <= 1u;
    let av = select(u0, u1, hi_a);
    let bv = select(v0, v1, hi_b);
    var position = vec3<f32>(av, p, bv);
    if axis == 1u {
        position = vec3<f32>(av, bv, p);
    } else if axis == 2u {
        position = vec3<f32>(p, bv, av);
    }
    // Water in a voxel that holds the sea's surface stops at the surface, not at the
    // voxel's top (cube_gen.rs `water_at_sea`): a 2,048-block voxel of sea is
    // drawn as sea, not as a 2,048-block slab of water.
    let surface = extent.y;
    let bottom = select(v0, f32(y) * scale, axis == 0u);
    if ((cf_materials[state * 32u] >> 2u) & 3u) != 0u && bottom < surface
        && position.y > surface {
        position.y = surface;
    }
    let uv = vec2<f32>(av - u0, v1 - bv);

    let f = cf_face(state, dir, overlay, biome);
    let light = vec2<u32>(select(240u, 64u, shaded == 1u), 0u);
    return cf_by_level(
        terrain_vertex(position, uv, f.layer, unorm8x4(f.color), light, instance.xyz, instance.w),
        countTrailingZeros(u32(scale)));
}

struct CfFace {
    layer: u32,
    // RGBA8: the tint shaded for the direction; alpha the HDR and fluid code.
    color: u32,
}

// A block face's sprite layer and vertex colour from the material table and
// the biome colours: shared with the surface tiles (surface_draw.wgsl).
fn cf_face(state: u32, dir: u32, overlay: u32, biome: u32) -> CfFace {
    let header = cf_materials[state * 32u];
    let face = 1u + 2u * (dir * 2u + overlay);
    let word = cf_materials[state * 32u + face];
    let layer = word & 0xffffu;
    let tint = (word >> 16u) & 7u;
    var rgb = 0xffffffu;
    if tint >= 1u && tint <= 4u {
        rgb = cf_biomes[biome * 4u + tint - 1u];
    } else if tint == 5u {
        rgb = cf_materials[state * 32u + face + 1u];
    }
    let shade = cf_shade(dir);
    let r = u32(f32((rgb >> 16u) & 255u) * shade);
    let g = u32(f32((rgb >> 8u) & 255u) * shade);
    let bl = u32(f32(rgb & 255u) * shade);
    let tier = (header >> 4u) & 255u;
    let level = min(tier & 7u, 4u);
    var code = level;
    if level > 0u && (tier & 8u) != 0u {
        code = level + 4u;
    }
    var high = 0xf0u;
    if ((header >> 2u) & 3u) != 0u {
        high = (8u | dir) << 4u;
    }
    let alpha = high | (15u - code);
    return CfFace(layer, r | (g << 8u) | (bl << 16u) | (alpha << 24u));
}
