//! Helpers for tests and examples: a block registry holding the blocks of the
//! default settings, and a world of cubes in a map.

use std::collections::HashMap;

use super::populate::CubeWorld;
use crate::registry::BlockRegistry;

const ORES: [&str; 8] = [
    "coal", "iron", "copper", "gold", "redstone", "diamond", "lapis", "emerald",
];

/// Every block name the default settings use, `minecraft:air` first (id 0).
pub fn default_block_names() -> Vec<String> {
    let mut names: Vec<String> = [
        "air",
        "stone",
        "granite",
        "diorite",
        "andesite",
        "deepslate",
        "tuff",
        "dirt",
        "gravel",
        "grass_block",
        "coarse_dirt",
        "podzol",
        "sand",
        "red_sand",
        "sandstone",
        "red_sandstone",
        "terracotta",
        "orange_terracotta",
        "yellow_terracotta",
        "brown_terracotta",
        "red_terracotta",
        "white_terracotta",
        "light_gray_terracotta",
        "mycelium",
        "snow_block",
        "mud",
        "water",
        "lava",
        "lily_pad",
        "bedrock",
    ]
    .iter()
    .map(|n| format!("minecraft:{n}"))
    .collect();
    for o in ORES {
        names.push(format!("minecraft:{o}_ore"));
        names.push(format!("minecraft:deepslate_{o}_ore"));
    }
    names
}

/// A registry of property-less blocks (leaked, so it is `'static`).
pub fn registry_of(names: &[String]) -> &'static BlockRegistry {
    let rows: Vec<serde_json::Value> = names
        .iter()
        .map(|n| {
            serde_json::json!([
                n,
                n.ends_with(":air"),
                n.ends_with("water") || n.ends_with("lava"),
                "",
                {"id": n, "properties": {}}
            ])
        })
        .collect();
    let defaults: serde_json::Map<String, serde_json::Value> = names
        .iter()
        .enumerate()
        .map(|(i, n)| (n.clone(), serde_json::json!(i)))
        .collect();
    let facts = serde_json::json!({"states": rows, "defaults": defaults});
    let registry = BlockRegistry::from_facts_json(facts.to_string().as_bytes())
        .expect("a valid synthetic registry");
    Box::leak(Box::new(registry))
}

/// [`registry_of`] the default block names.
pub fn default_registry() -> &'static BlockRegistry {
    registry_of(&default_block_names())
}

/// A [`CubeWorld`] over a map of cubes; absent cubes read as `fill` and
/// ignore writes.
pub struct MapWorld {
    pub cubes: HashMap<[i32; 3], Box<[u32; 4096]>>,
    pub fill: u32,
}

impl MapWorld {
    pub fn new(fill: u32) -> Self {
        Self {
            cubes: HashMap::new(),
            fill,
        }
    }
    pub fn insert(&mut self, cube: [i32; 3], blocks: [u32; 4096]) {
        self.cubes.insert(cube, Box::new(blocks));
    }
    fn index(p: [i32; 3]) -> usize {
        (((p[1] & 15) << 8) | ((p[2] & 15) << 4) | (p[0] & 15)) as usize
    }
}

impl CubeWorld for MapWorld {
    fn block(&self, p: [i32; 3]) -> u32 {
        self.cubes
            .get(&[p[0] >> 4, p[1] >> 4, p[2] >> 4])
            .map_or(self.fill, |c| c[Self::index(p)])
    }
    fn set_block(&mut self, p: [i32; 3], state: u32) {
        if let Some(c) = self.cubes.get_mut(&[p[0] >> 4, p[1] >> 4, p[2] >> 4]) {
            c[Self::index(p)] = state;
        }
    }
}
