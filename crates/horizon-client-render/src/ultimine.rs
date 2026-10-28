//! Ultimine, Horizon's vein miner (in the manner of FTB Ultimine): while its
//! key is held, breaking a block also breaks the blocks the current shape
//! selects, each as a normal player break. This module is the part without
//! the game: the shapes, which blocks count as the same kind, the selection
//! over a block lookup, the stop on a worn tool, the selection's outline, the
//! settings shared by the window and its server thread, and the payload a
//! client sends its server with the key's state.
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::RwLock;

/// The serverbound custom payload carrying a player's `Input`.
pub const CHANNEL: &str = "horizon:ultimine";

/// What the held key selects around the broken block.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Shape {
    /// Connected blocks of the same kind, diagonals included, nearest first.
    #[default]
    Shapeless,
    /// 1x1 straight into the clicked face.
    SmallTunnel,
    /// 3x3 straight into the clicked face.
    LargeTunnel,
    /// The 3x3 plane of the clicked face.
    SmallSquare,
    /// A 1 wide, 2 high staircase going down, away from the player.
    MiningTunnel,
    /// The same going up.
    EscapeTunnel,
}
impl Shape {
    pub const ALL: [Shape; 6] = [
        Shape::Shapeless,
        Shape::SmallTunnel,
        Shape::LargeTunnel,
        Shape::SmallSquare,
        Shape::MiningTunnel,
        Shape::EscapeTunnel,
    ];
    pub fn name(self) -> &'static str {
        match self {
            Shape::Shapeless => "Shapeless",
            Shape::SmallTunnel => "Small Tunnel",
            Shape::LargeTunnel => "Large Tunnel",
            Shape::SmallSquare => "Small Square",
            Shape::MiningTunnel => "Mining Tunnel",
            Shape::EscapeTunnel => "Escape Tunnel",
        }
    }
    pub fn id(self) -> u8 {
        Self::ALL.iter().position(|s| *s == self).unwrap_or(0) as u8
    }
    pub fn from_id(id: u8) -> Option<Shape> {
        Self::ALL.get(usize::from(id)).copied()
    }
    /// The next shape (`forward`) or the previous one, wrapping around.
    pub fn cycle(self, forward: bool) -> Shape {
        let n = Self::ALL.len();
        let i = usize::from(self.id());
        Self::ALL[if forward {
            (i + 1) % n
        } else {
            (i + n - 1) % n
        }]
    }
}

/// A player's Ultimine input: the key held, and the shape picked.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Input {
    pub held: bool,
    pub shape: Shape,
}
impl Input {
    pub fn encode(self) -> Vec<u8> {
        vec![u8::from(self.held), self.shape.id()]
    }
    pub fn decode(data: &[u8]) -> Option<Input> {
        match data {
            [held, shape] if *held <= 1 => Some(Input {
                held: *held == 1,
                shape: Shape::from_id(*shape)?,
            }),
            _ => None,
        }
    }
}

/// The server's limits, shared by the window and its server thread.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Config {
    /// Blocks one operation may take, the broken block included.
    pub max_blocks: u32,
    /// Each extra block costs vanilla's 0.005 exhaustion times this.
    pub exhaustion_per_block: f32,
    /// Stop while the tool has more than this many uses left (0: before it breaks).
    pub prevent_tool_break: u32,
}
impl Config {
    pub const DEFAULT: Config = Config {
        max_blocks: 64,
        exhaustion_per_block: 20.0,
        prevent_tool_break: 0,
    };
    /// The exhaustion one extra block costs in all.
    pub fn exhaustion(&self) -> f32 {
        0.005 * self.exhaustion_per_block
    }
}
impl Default for Config {
    fn default() -> Self {
        Self::DEFAULT
    }
}
static CONFIG: RwLock<Config> = RwLock::new(Config::DEFAULT);
pub fn config() -> Config {
    *CONFIG.read().unwrap_or_else(|e| e.into_inner())
}
pub fn set_config(config: Config) {
    *CONFIG.write().unwrap_or_else(|e| e.into_inner()) = config;
}

/// Which blocks count as one kind in the shapeless mode: the overworld's base
/// stones together, each `c:ores/<x>` tag together (or an ore with its
/// deepslate form by name), a wood's logs and wood together.
#[derive(Clone, Debug, Default)]
pub struct Groups {
    base_stone: HashSet<String>,
    logs: HashSet<String>,
    /// Block -> its `c:ores/<x>` tag.
    ores: HashMap<String, String>,
}
const BASE_STONE: &str = "minecraft:base_stone_overworld";
impl Groups {
    pub fn new(block_tags: &HashMap<String, HashSet<String>>) -> Groups {
        let tag = |name: &str| block_tags.get(name).cloned().unwrap_or_default();
        let mut ore_tags: Vec<&String> = block_tags
            .keys()
            .filter(|t| t.strip_prefix("c:ores/").is_some_and(|x| !x.is_empty()))
            .collect();
        ore_tags.sort();
        let mut ores = HashMap::new();
        for t in ore_tags {
            for block in &block_tags[t] {
                ores.entry(block.clone()).or_insert_with(|| t.clone());
            }
        }
        Groups {
            base_stone: tag(BASE_STONE),
            logs: tag("minecraft:logs"),
            ores,
        }
    }
    /// The kind `block` belongs to; blocks of one kind merge.
    pub fn key(&self, block: &str) -> String {
        if self.base_stone.contains(block) {
            return format!("#{BASE_STONE}");
        }
        if let Some(tag) = self.ores.get(block) {
            return format!("#{tag}");
        }
        let (namespace, path) = block.split_once(':').unwrap_or(("minecraft", block));
        if path.ends_with("_ore") {
            let path = path.strip_prefix("deepslate_").unwrap_or(path);
            return format!("ore:{namespace}:{path}");
        }
        if self.logs.contains(block) {
            for suffix in ["_log", "_wood", "_stem", "_hyphae"] {
                if let Some(wood) = path.strip_suffix(suffix) {
                    return format!("log:{namespace}:{wood}");
                }
            }
        }
        block.to_owned()
    }
}

/// `Direction.fromYRot`: the horizontal way a player with this yaw faces.
pub fn facing(yaw: f32) -> [i32; 3] {
    match ((yaw / 90.0 + 0.5).floor() as i32) & 3 {
        0 => [0, 0, 1],
        1 => [-1, 0, 0],
        2 => [0, 0, -1],
        _ => [1, 0, 0],
    }
}

fn add(a: [i32; 3], b: [i32; 3]) -> [i32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn scale(a: [i32; 3], k: i32) -> [i32; 3] {
    [a[0] * k, a[1] * k, a[2] * k]
}
fn dist2(a: [i32; 3], b: [i32; 3]) -> i64 {
    (0..3).map(|i| i64::from(a[i] - b[i]).pow(2)).sum()
}
/// Two axes across the face's normal.
fn across(face: [i32; 3]) -> ([i32; 3], [i32; 3]) {
    if face[1] != 0 {
        ([1, 0, 0], [0, 0, 1])
    } else if face[0] != 0 {
        ([0, 1, 0], [0, 0, 1])
    } else {
        ([1, 0, 0], [0, 1, 0])
    }
}
/// A 3x3 layer around its centre: the centre first, then the ring.
const LAYER: [(i32, i32); 9] = [
    (0, 0),
    (0, 1),
    (1, 0),
    (0, -1),
    (-1, 0),
    (1, 1),
    (1, -1),
    (-1, -1),
    (-1, 1),
];

/// The blocks one operation takes, the broken block `origin` first, at most
/// `max`. `face` is the clicked face's outward normal, `facing` the player's
/// horizontal facing (`facing`). `lookup` names a block's kind, or `None` for
/// one the operation may not take (air, fluids, unbreakable, not harvestable).
/// The shapeless mode takes connected blocks of `origin_key`'s kind; the
/// shapes take any block `lookup` allows.
pub fn select<K: PartialEq>(
    origin: [i32; 3],
    face: [i32; 3],
    facing: [i32; 3],
    shape: Shape,
    max: usize,
    origin_key: &K,
    mut lookup: impl FnMut([i32; 3]) -> Option<K>,
) -> Vec<[i32; 3]> {
    let mut out = vec![origin];
    if max <= 1 {
        return out;
    }
    let into = scale(face, -1);
    let up = [0, 1, 0];
    let mut candidates: Vec<[i32; 3]> = Vec::new();
    match shape {
        Shape::Shapeless => {
            let mut seen = HashSet::from([origin]);
            let mut heap = BinaryHeap::new();
            let mut visit =
                |p: [i32; 3], heap: &mut BinaryHeap<std::cmp::Reverse<(i64, [i32; 3])>>| {
                    for dx in -1..=1 {
                        for dy in -1..=1 {
                            for dz in -1..=1 {
                                let n = add(p, [dx, dy, dz]);
                                if seen.insert(n) && lookup(n).as_ref() == Some(origin_key) {
                                    heap.push(std::cmp::Reverse((dist2(n, origin), n)));
                                }
                            }
                        }
                    }
                };
            visit(origin, &mut heap);
            while out.len() < max {
                let Some(std::cmp::Reverse((_, p))) = heap.pop() else {
                    break;
                };
                out.push(p);
                if out.len() < max {
                    visit(p, &mut heap);
                }
            }
            return out;
        }
        Shape::SmallTunnel => {
            candidates.extend((1..=max as i32).map(|k| add(origin, scale(into, k))));
        }
        Shape::LargeTunnel | Shape::SmallSquare => {
            let (u, v) = across(face);
            let depth = if shape == Shape::SmallSquare {
                1
            } else {
                max as i32
            };
            for k in 0..depth {
                for (a, b) in LAYER {
                    let p = add(add(origin, scale(into, k)), add(scale(u, a), scale(v, b)));
                    if p != origin {
                        candidates.push(p);
                    }
                }
            }
        }
        Shape::MiningTunnel | Shape::EscapeTunnel => {
            let rise = if shape == Shape::MiningTunnel { -1 } else { 1 };
            for i in 0..max as i32 {
                let step = add(add(origin, scale(facing, i)), scale(up, rise * i));
                for p in [step, add(step, up)] {
                    if p != origin {
                        candidates.push(p);
                    }
                }
            }
        }
    }
    for p in candidates {
        if out.len() >= max {
            break;
        }
        if lookup(p).is_some() {
            out.push(p);
        }
    }
    out
}

/// Whether a tool with `remaining` uses (None: it doesn't wear) may take a
/// block costing `cost`, keeping more than `prevent` uses.
pub fn tool_allows(remaining: Option<u32>, cost: u32, prevent: u32) -> bool {
    remaining.is_none_or(|r| r > cost.saturating_add(prevent))
}

/// The outline of a selection as world-space segments: the edges of its
/// outer surface, not the seams between two of its blocks.
pub fn outline(blocks: &[[i32; 3]]) -> Vec<([f64; 3], [f64; 3])> {
    let cells: HashSet<[i32; 3]> = blocks.iter().copied().collect();
    let mut edges: HashSet<([i32; 3], usize)> = HashSet::new();
    let mut out = Vec::new();
    for &p in blocks {
        for axis in 0..3 {
            let (a, b) = ((axis + 1) % 3, (axis + 2) % 3);
            for (s, t) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                let mut q = p;
                q[a] += s;
                q[b] += t;
                if !edges.insert((q, axis)) {
                    continue;
                }
                // The four cells around the edge, in order around it.
                let around = [(-1, -1), (0, -1), (0, 0), (-1, 0)].map(|(i, j)| {
                    let mut c = q;
                    c[a] += i;
                    c[b] += j;
                    cells.contains(&c)
                });
                let count = around.iter().filter(|&&x| x).count();
                let diagonal = count == 2 && around[0] == around[2];
                if count == 1 || count == 3 || diagonal {
                    let start = q.map(f64::from);
                    let mut end = start;
                    end[axis] += 1.0;
                    out.push((start, end));
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A world of named blocks, air elsewhere.
    fn world(blocks: &[([i32; 3], &str)]) -> HashMap<[i32; 3], String> {
        blocks.iter().map(|(p, n)| (*p, (*n).to_owned())).collect()
    }
    fn groups() -> Groups {
        let mut tags: HashMap<String, HashSet<String>> = HashMap::new();
        let mut put = |tag: &str, blocks: &[&str]| {
            tags.insert(
                tag.into(),
                blocks.iter().map(|b| format!("minecraft:{b}")).collect(),
            );
        };
        put(
            BASE_STONE,
            &[
                "stone",
                "granite",
                "diorite",
                "andesite",
                "tuff",
                "deepslate",
            ],
        );
        put(
            "minecraft:logs",
            &[
                "oak_log",
                "oak_wood",
                "stripped_oak_log",
                "birch_log",
                "crimson_stem",
                "crimson_hyphae",
            ],
        );
        put("c:ores/copper", &["copper_ore", "deepslate_copper_ore"]);
        put("c:ores", &["copper_ore", "iron_ore", "gold_ore"]);
        Groups::new(&tags)
    }
    fn run(
        world: &HashMap<[i32; 3], String>,
        origin: [i32; 3],
        face: [i32; 3],
        facing: [i32; 3],
        shape: Shape,
        max: usize,
    ) -> Vec<[i32; 3]> {
        let g = groups();
        let key = g.key(&world[&origin]);
        select(origin, face, facing, shape, max, &key, |p| {
            world.get(&p).map(|n| g.key(n))
        })
    }

    #[test]
    fn kinds_merge_stones_ores_and_woods() {
        let g = groups();
        let k = |b: &str| g.key(&format!("minecraft:{b}"));
        assert_eq!(k("stone"), k("deepslate"));
        assert_eq!(k("granite"), k("tuff"));
        // A c:ores/<x> tag, and the name fallback without one.
        assert_eq!(k("copper_ore"), k("deepslate_copper_ore"));
        assert_eq!(k("iron_ore"), k("deepslate_iron_ore"));
        assert_ne!(k("iron_ore"), k("gold_ore"));
        assert_ne!(k("iron_ore"), k("copper_ore"));
        assert_eq!(k("oak_log"), k("oak_wood"));
        assert_eq!(k("crimson_stem"), k("crimson_hyphae"));
        assert_ne!(k("oak_log"), k("birch_log"));
        assert_ne!(k("oak_log"), k("stripped_oak_log"));
        assert_ne!(k("stone"), k("cobblestone"));
        assert_eq!(k("cobblestone"), "minecraft:cobblestone");
    }

    #[test]
    fn shapeless_follows_diagonals_and_merged_kinds_nearest_first() {
        let w = world(&[
            ([0, 0, 0], "minecraft:iron_ore"),
            // Only diagonally connected.
            ([1, 1, 0], "minecraft:deepslate_iron_ore"),
            ([2, 2, 1], "minecraft:iron_ore"),
            // Next to the ore but another kind: it neither joins nor connects.
            ([-1, 0, 0], "minecraft:stone"),
            ([-2, 0, 0], "minecraft:iron_ore"),
        ]);
        let got = run(&w, [0, 0, 0], [0, 1, 0], [0, 0, 1], Shape::Shapeless, 64);
        assert_eq!(got, [[0, 0, 0], [1, 1, 0], [2, 2, 1]]);
        // Nearest first: a ring of stone around the origin comes before the far end of a line.
        let mut blocks = vec![([0, 0, 0], "minecraft:stone")];
        for x in 1..6 {
            blocks.push(([x, 0, 0], "minecraft:andesite"));
        }
        blocks.push(([0, 0, 1], "minecraft:granite"));
        let w = world(&blocks);
        let got = run(&w, [0, 0, 0], [0, 1, 0], [0, 0, 1], Shape::Shapeless, 64);
        assert_eq!(got.len(), 7);
        assert_eq!(&got[..3], [[0, 0, 0], [0, 0, 1], [1, 0, 0]]);
        assert_eq!(got[6], [5, 0, 0]);
    }

    #[test]
    fn the_limit_counts_the_broken_block() {
        let w = solid();
        for shape in Shape::ALL {
            let got = run(&w, [0, 0, 0], [0, 0, -1], [0, 0, 1], shape, 64);
            assert!(got.len() <= 64, "{shape:?}");
            assert_eq!(got[0], [0, 0, 0]);
            let unique: HashSet<_> = got.iter().collect();
            assert_eq!(unique.len(), got.len(), "{shape:?} repeats a block");
        }
        for (shape, n) in [
            (Shape::Shapeless, 64),
            (Shape::LargeTunnel, 64),
            (Shape::SmallSquare, 9),
            (Shape::SmallTunnel, 13),
            (Shape::MiningTunnel, 26),
        ] {
            let got = run(&w, [0, 0, 0], [0, 0, -1], [0, 0, 1], shape, 64);
            assert_eq!(got.len(), n, "{shape:?}");
        }
        let got = run(&w, [0, 0, 0], [0, 0, -1], [0, 0, 1], Shape::Shapeless, 64);
        // Nearest first: everything within one block of the origin is in.
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    assert!(got.contains(&[dx, dy, dz]));
                }
            }
        }
        assert_eq!(
            run(&w, [0, 0, 0], [0, 0, -1], [0, 0, 1], Shape::Shapeless, 1),
            [[0, 0, 0]]
        );
        assert_eq!(
            run(&w, [0, 0, 0], [0, 0, -1], [0, 0, 1], Shape::Shapeless, 5).len(),
            5
        );
    }

    /// Solid stone around the origin, so shapes show their full geometry.
    fn solid() -> HashMap<[i32; 3], String> {
        let mut blocks = Vec::new();
        for x in -12..=12 {
            for y in -12..=12 {
                for z in -12..=12 {
                    blocks.push(([x, y, z], "minecraft:dirt"));
                }
            }
        }
        world(&blocks)
    }

    #[test]
    fn tunnels_go_straight_into_the_clicked_face() {
        let w = solid();
        // The north face (-z) clicked: the tunnel runs south (+z).
        let got = run(&w, [0, 0, 0], [0, 0, -1], [0, 0, 1], Shape::SmallTunnel, 5);
        assert_eq!(got, [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 0, 4]]);
        // The top face: straight down.
        let got = run(&w, [0, 0, 0], [0, 1, 0], [1, 0, 0], Shape::SmallTunnel, 3);
        assert_eq!(got, [[0, 0, 0], [0, -1, 0], [0, -2, 0]]);
        // A large tunnel into the east face (+x) runs west, 3x3 in y and z.
        let got = run(&w, [0, 0, 0], [1, 0, 0], [-1, 0, 0], Shape::LargeTunnel, 18);
        assert_eq!(got.len(), 18);
        for p in &got {
            assert!((-1..=0).contains(&p[0]), "{p:?}");
            assert!(
                (-1..=1).contains(&p[1]) && (-1..=1).contains(&p[2]),
                "{p:?}"
            );
        }
        assert!(got[..9].iter().all(|p| p[0] == 0));
        // Air in the way is skipped, not stopped at.
        let mut w = solid();
        w.remove(&[0, 0, 1]);
        let got = run(&w, [0, 0, 0], [0, 0, -1], [0, 0, 1], Shape::SmallTunnel, 3);
        assert_eq!(got, [[0, 0, 0], [0, 0, 2], [0, 0, 3]]);
    }

    #[test]
    fn a_small_square_is_the_clicked_face_plane() {
        let w = solid();
        let got = run(&w, [0, 0, 0], [0, 1, 0], [0, 0, 1], Shape::SmallSquare, 64);
        assert_eq!(got.len(), 9);
        assert!(
            got.iter()
                .all(|p| p[1] == 0 && p[0].abs() <= 1 && p[2].abs() <= 1)
        );
        let got = run(&w, [0, 0, 0], [0, 0, 1], [0, 0, -1], Shape::SmallSquare, 64);
        assert_eq!(got.len(), 9);
        assert!(
            got.iter()
                .all(|p| p[2] == 0 && p[0].abs() <= 1 && p[1].abs() <= 1)
        );
        let got = run(&w, [0, 0, 0], [-1, 0, 0], [1, 0, 0], Shape::SmallSquare, 4);
        assert_eq!(got.len(), 4);
        assert!(got.iter().all(|p| p[0] == 0));
    }

    #[test]
    fn staircases_follow_the_players_facing() {
        let w = solid();
        // Facing east: down one for each step east, two high.
        let got = run(&w, [0, 0, 0], [-1, 0, 0], [1, 0, 0], Shape::MiningTunnel, 7);
        assert_eq!(
            got,
            [
                [0, 0, 0],
                [0, 1, 0],
                [1, -1, 0],
                [1, 0, 0],
                [2, -2, 0],
                [2, -1, 0],
                [3, -3, 0]
            ]
        );
        // Facing north, clicking the floor: the facing decides, not the face.
        let got = run(&w, [0, 0, 0], [0, 1, 0], [0, 0, -1], Shape::EscapeTunnel, 6);
        assert_eq!(
            got,
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 1, -1],
                [0, 2, -1],
                [0, 2, -2],
                [0, 3, -2]
            ]
        );
    }

    #[test]
    fn shapes_take_only_what_the_lookup_allows() {
        let mut w = solid();
        w.insert([0, 0, 1], "minecraft:bedrock".into());
        let g = groups();
        let key = g.key("minecraft:dirt");
        let got = select(
            [0, 0, 0],
            [0, 0, -1],
            [0, 0, 1],
            Shape::SmallTunnel,
            3,
            &key,
            |p| {
                w.get(&p)
                    .filter(|n| *n != "minecraft:bedrock")
                    .map(|n| g.key(n))
            },
        );
        assert_eq!(got, [[0, 0, 0], [0, 0, 2], [0, 0, 3]]);
    }

    #[test]
    fn the_tool_stops_before_it_breaks() {
        assert!(tool_allows(None, 1, 0));
        assert!(tool_allows(Some(2), 1, 0));
        assert!(!tool_allows(Some(1), 1, 0));
        assert!(!tool_allows(Some(0), 0, 0));
        assert!(tool_allows(Some(1), 0, 0));
        assert!(!tool_allows(Some(11), 1, 10));
        assert!(tool_allows(Some(12), 1, 10));
        // A pickaxe with 5 uses left takes 4 blocks and keeps the last use.
        let mut remaining = 5;
        let mut taken = 0;
        while tool_allows(Some(remaining), 1, 0) {
            remaining -= 1;
            taken += 1;
        }
        assert_eq!((taken, remaining), (4, 1));
    }

    #[test]
    fn inputs_round_trip_and_shapes_cycle() {
        for shape in Shape::ALL {
            for held in [false, true] {
                let i = Input { held, shape };
                assert_eq!(Input::decode(&i.encode()), Some(i));
            }
            assert_eq!(shape.cycle(true).cycle(false), shape);
        }
        assert_eq!(Shape::EscapeTunnel.cycle(true), Shape::Shapeless);
        assert_eq!(Shape::Shapeless.cycle(false), Shape::EscapeTunnel);
        assert_eq!(Input::decode(&[1, 99]), None);
        assert_eq!(Input::decode(&[2, 0]), None);
        assert_eq!(Input::decode(&[1]), None);
    }

    #[test]
    fn facing_follows_vanilla_yaw() {
        assert_eq!(facing(0.0), [0, 0, 1]);
        assert_eq!(facing(90.0), [-1, 0, 0]);
        assert_eq!(facing(180.0), [0, 0, -1]);
        assert_eq!(facing(-90.0), [1, 0, 0]);
        assert_eq!(facing(44.0), [0, 0, 1]);
        assert_eq!(facing(46.0), [-1, 0, 0]);
    }

    #[test]
    fn the_outline_skips_inner_seams() {
        assert_eq!(outline(&[[0, 0, 0]]).len(), 12);
        // Two blocks side by side: a 2x1x1 box, 12 edges, 4 of them 2 long (8 segments).
        assert_eq!(outline(&[[0, 0, 0], [1, 0, 0]]).len(), 16);
        // A 2x2x1 slab: 12 box edges, 8 of them in two segments (20), no seams inside.
        let slab = outline(&[[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]);
        assert_eq!(slab.len(), 20);
        // Diagonal neighbours keep the edge they share (once).
        let d = outline(&[[0, 0, 0], [1, 1, 0]]);
        assert_eq!(d.len(), 23);
    }
}
