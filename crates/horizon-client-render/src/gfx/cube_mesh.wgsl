// GPU mesher for far cube nodes (gfx/cube_mesh.rs documents the layouts).
//
// One workgroup per (job, direction): 256 threads walk the node's 16 slices
// perpendicular to the direction. Per slice every thread decides the exposed
// face of its cell (two overlays), the cells' keys sit in workgroup memory,
// and a thread that starts a maximal rectangle (a run along a, merged with
// the identical runs of the rows after it) counts it (count_main) or writes
// its quad (emit_main). The counts and the emit see the same keys, so the
// numbers agree.

const SLOT_WORDS: u32 = 4352u;
const BIOMES: u32 = 4096u;
const JOB_WORDS: u32 = 24u;
const NONE: u32 = 0xffffffffu;

struct Params {
    air: u32,
    state_count: u32,
    jobs: u32,
    pad: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> pool: array<u32>;
@group(0) @binding(2) var<storage, read> materials: array<u32>;
// Per job: slot, level, flags (bit 0: has cover, bit 1: the node's top is the sea's
// surface), 0, six neighbour slots
// (Down, Up, North, South, West, East; NONE), 2 spare, eight cover words
// (bit z << 4 | x), 4 spare.
@group(0) @binding(3) var<storage, read> jobs: array<u32>;
@group(0) @binding(4) var<storage, read_write> counts: array<u32>;
@group(0) @binding(5) var<storage, read> offsets: array<u32>;
@group(0) @binding(6) var<storage, read_write> quads: array<vec2<u32>>;

var<workgroup> keys: array<u32, 512>;
var<workgroup> lit: array<u32, 256>;
var<workgroup> tally: array<atomic<u32>, 2>;

fn header(state: u32) -> u32 {
    if state >= params.state_count {
        return 0u;
    }
    return materials[state * 32u];
}
fn is_opaque(state: u32) -> bool {
    return state != params.air && (header(state) & 2u) != 0u;
}
fn fluid_of(state: u32) -> u32 {
    return (header(state) >> 2u) & 3u;
}
fn face_word(state: u32, dir: u32, overlay: u32) -> u32 {
    if state >= params.state_count {
        return 0u;
    }
    return materials[state * 32u + 1u + 2u * (dir * 2u + overlay)];
}
fn state_at(slot: u32, x: u32, y: u32, z: u32) -> u32 {
    return pool[slot * SLOT_WORDS + ((y << 8u) | (z << 4u) | x)];
}

fn cell_pos(dir: u32, slice: u32, a: u32, b: u32) -> vec3<u32> {
    let axis = dir >> 1u;
    if axis == 0u {
        return vec3<u32>(a, slice, b);
    }
    if axis == 1u {
        return vec3<u32>(a, b, slice);
    }
    return vec3<u32>(slice, b, a);
}

fn step_of(dir: u32) -> vec3<i32> {
    switch dir {
        case 0u: { return vec3<i32>(0, -1, 0); }
        case 1u: { return vec3<i32>(0, 1, 0); }
        case 2u: { return vec3<i32>(0, 0, -1); }
        case 3u: { return vec3<i32>(0, 0, 1); }
        case 4u: { return vec3<i32>(-1, 0, 0); }
        default: { return vec3<i32>(1, 0, 0); }
    }
}

// The key of the face the voxel at p shows in direction dir, overlay o; 0: none.
// bit 0 valid, bit 1 shaded, bits 2..18 state, bits 19..26 biome (tinted faces only).
fn cell_key(base: u32, slot: u32, dir: u32, o: u32, p: vec3<u32>) -> u32 {
    let own = state_at(slot, p.x, p.y, p.z);
    if own == params.air {
        return 0u;
    }
    let fw = face_word(own, dir, o);
    if (fw & 0x100000u) == 0u {
        return 0u;
    }
    let q = vec3<i32>(p) + step_of(dir);
    let inside = all(q >= vec3<i32>(0)) && all(q < vec3<i32>(16));
    var next = params.air;
    var known = false;
    if inside {
        next = state_at(slot, u32(q.x), u32(q.y), u32(q.z));
        known = true;
    } else {
        // A fluid's faces on the node's edge lie inside the fluid the next node goes on
        // with, except its top on a node whose top is the sea's surface (job flags bit 1):
        // there the sea ends, and without it the sea showed its floor wherever its
        // surface fell on a node's top (levels 0 to 2).
        let sea_top = (jobs[base + 2u] & 2u) != 0u;
        if fluid_of(own) != 0u && !(dir == 1u && sea_top) {
            return 0u;
        }
        let neighbour = jobs[base + 4u + dir];
        if neighbour != NONE {
            let w = vec3<u32>(q & vec3<i32>(15));
            next = state_at(neighbour, w.x, w.y, w.z);
            known = true;
        }
    }
    if known {
        if next == own {
            return 0u;
        }
        if next != params.air {
            let f = fluid_of(next);
            if is_opaque(next) || (f != 0u && f == fluid_of(own)) {
                return 0u;
            }
        }
    }
    let col = (p.z << 4u) | p.x;
    // A face takes the light of the cell in front of it, as in Minecraft: a cliff's wall
    // is in the sun when the air beside it sees the sky, not only its top voxel. A top
    // on the node's top takes its own column's light there: open unless the column is
    // covered above the node.
    var front = (lit[col] >> 15u) & 1u;
    if inside {
        front = (lit[(u32(q.z) << 4u) | u32(q.x)] >> u32(q.y)) & 1u;
    } else if dir == 0u {
        // The underside at the node's bottom: shaded, as under an overhang.
        front = 0u;
    } else if dir >= 2u {
        // A wall on the node's side: its own column says nothing of the air in front
        // (a cliff rising past the node has rock over it, and its walls went dark
        // node by node). Open from the neighbour's top down to the cell in front where
        // the neighbour is known (what covers it above that node is not), else open:
        // a wall on a node's side mostly faces out, as a cliff's does.
        front = 1u;
        let neighbour = jobs[base + 4u + dir];
        if neighbour != NONE {
            let w = vec3<u32>(q & vec3<i32>(15));
            for (var y = 15u; y > w.y; y--) {
                if is_opaque(state_at(neighbour, w.x, y, w.z)) {
                    front = 0u;
                    break;
                }
            }
        }
    }
    let shaded = 1u - front;
    var biome = 0u;
    if ((fw >> 16u) & 7u) != 0u {
        biome = min(pool[slot * SLOT_WORDS + BIOMES + col] & 255u, 255u);
    }
    return 1u | (shaded << 1u) | (own << 2u) | (biome << 19u);
}

fn key_at(o: u32, a: u32, b: u32) -> u32 {
    return keys[o * 256u + b * 16u + a];
}
fn run_start(o: u32, a: u32, b: u32) -> bool {
    let k = key_at(o, a, b);
    return k != 0u && (a == 0u || key_at(o, a - 1u, b) != k);
}
fn run_len(o: u32, a: u32, b: u32) -> u32 {
    let k = key_at(o, a, b);
    var n = 1u;
    while a + n < 16u && key_at(o, a + n, b) == k {
        n += 1u;
    }
    return n;
}

fn mesh_group(group: u32, lid: u32, emit: bool) {
    let dir = group % 6u;
    let job = group / 6u;
    let base = job * JOB_WORDS;
    let slot = jobs[base];
    let has_cover = (jobs[base + 2u] & 1u) != 0u;
    let skipped = emit && offsets[job * 12u + dir * 2u] == NONE;

    // Sky light of this thread's column: open from the top unless covered.
    var open = true;
    if has_cover {
        open = ((jobs[base + 12u + (lid >> 5u)] >> (lid & 31u)) & 1u) == 0u;
    }
    var mask = 0u;
    for (var y = 15; y >= 0; y--) {
        if open {
            mask |= 1u << u32(y);
            if is_opaque(state_at(slot, lid & 15u, u32(y), lid >> 4u)) {
                open = false;
            }
        }
    }
    lit[lid] = mask;
    if lid < 2u {
        atomicStore(&tally[lid], 0u);
    }
    workgroupBarrier();

    let a = lid & 15u;
    let b = lid >> 4u;
    for (var slice = 0u; slice < 16u; slice++) {
        let p = cell_pos(dir, slice, a, b);
        for (var o = 0u; o < 2u; o++) {
            keys[o * 256u + lid] = cell_key(base, slot, dir, o, p);
        }
        workgroupBarrier();
        for (var o = 0u; o < 2u; o++) {
            let k = key_at(o, a, b);
            if k != 0u && run_start(o, a, b) {
                let len = run_len(o, a, b);
                var head = true;
                if b > 0u && key_at(o, a, b - 1u) == k && run_start(o, a, b - 1u)
                    && run_len(o, a, b - 1u) == len {
                    head = false;
                }
                if head {
                    var h = 1u;
                    while b + h < 16u && key_at(o, a, b + h) == k && run_start(o, a, b + h)
                        && run_len(o, a, b + h) == len {
                        h += 1u;
                    }
                    let state = (k >> 2u) & 0x1ffffu;
                    let translucent = (face_word(state, dir, o) >> 19u) & 1u;
                    let n = atomicAdd(&tally[translucent], 1u);
                    if emit && !skipped {
                        let w0 = p.x | (p.y << 4u) | (p.z << 8u) | ((len - 1u) << 12u)
                            | ((h - 1u) << 16u) | (dir << 20u) | (((k >> 1u) & 1u) << 23u)
                            | (o << 24u);
                        let w1 = state | (((k >> 19u) & 255u) << 17u);
                        quads[offsets[job * 12u + dir * 2u + translucent] + n] = vec2<u32>(w0, w1);
                    }
                }
            }
        }
        workgroupBarrier();
    }
    if !emit && lid < 2u {
        counts[job * 12u + dir * 2u + lid] = atomicLoad(&tally[lid]);
    }
}

@compute @workgroup_size(256)
fn count_main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let group = wg.y * 32768u + wg.x;
    if group < params.jobs * 6u {
        mesh_group(group, lid, false);
    }
}

@compute @workgroup_size(256)
fn emit_main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let group = wg.y * 32768u + wg.x;
    if group < params.jobs * 6u {
        mesh_group(group, lid, true);
    }
}
