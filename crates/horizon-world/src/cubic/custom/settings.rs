//! `CustomGeneratorSettings`: CubicWorldGen's "Custom Cubic" settings object
//! (version 9) with its exact JSON field names, and the Flat Cubic settings.
//!
//! # Differences from CWG (1.12) that come from 26.3
//!
//! * `biome` is a biome name (`"plains"` is `minecraft:plains`). Empty, absent
//!   or the integer `-1` means the normal biome source. Other integers are
//!   1.12 biome ids and are translated.
//! * Biome names in every filter and list are normalised (`plains` ->
//!   `minecraft:plains`) and 1.12 names (`extreme_hills`, `mesa`,
//!   `redwood_taiga`, `mutated_*`, ...) are translated to their 26.3
//!   counterparts. `includeClass` / `excludeClass` filters keep CWG's Java
//!   class names; the classes are mapped to sets of 26.3 biomes
//!   ([`class_biomes`]).
//! * Block states are CWG's `BlockStateDesc` forms (`"stone"`, `"mod:block"`,
//!   `{"Name":..,"Properties":{..}}`). They are resolved against the block
//!   registry when the generator is built; a few 1.12 names that 26.3
//!   renamed are translated (see [`legacy_block`]).
//! * The default ores are 26.3's overworld ores, the default replacers add the
//!   deepslate gradient and use 26.3 biome names. `randomYGradient` has two
//!   Horizon extension fields (`filterBlocks`, `blockFilterType`) so the
//!   gradient can turn only stone into deepslate; CWG ignores them.
//! * There is no fixer chain: a `version` below 9 is accepted as it is and
//!   [`CustomGeneratorSettings::warnings`] says so.
//! * Structures generate (unlike CWG, which made only strongholds): 26.3's
//!   overworld structure sets, placed by vanilla's rules around the terrain's
//!   surface. CWG's toggles (`strongholds`, `villages`, `mineshafts`,
//!   `temples`, `oceanMonuments`, `woodlandMansions`) switch their groups of
//!   sets off; the Horizon extension `structureSets` switches and tunes each
//!   set ([`StructureSetTuning`], vanilla's numbers by default:
//!   [`OVERWORLD_STRUCTURE_SETS`]). `alternateStrongholdsPositions` is kept and
//!   unused. `riverSize` is kept and unused; `biomeSize` is applied by the
//!   integrator through [`CustomGeneratorSettings::biome_scale`].
//! * Missing `minY` / `maxY` of a replacer mean the whole range, and missing
//!   `minDensity` / `maxDensity` mean -inf / +inf (CWG's Java fields would be
//!   0, which no real preset relies on).
//! * `cubeAreas` boxes are inclusive cube coordinates; the first matching
//!   entry in list order wins (CWG used a hash map, so its order was
//!   undefined).

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::BTreeMap;

/// The settings version this module writes.
pub const LATEST_VERSION: i32 = 9;

/// Serde helpers for `f64` fields that CWG writes as bare `Infinity`,
/// `-Infinity` or `NaN`. [`CustomGeneratorSettings::from_json`] turns the bare
/// tokens into `1.7976931348623157e308`, `-1.7976931348623157e308` and `null`
/// before parsing; this module maps them back, and writes the strings
/// `"Infinity"`... which `to_json_*` turns into bare tokens again.
pub mod inf {
    use super::*;

    pub fn serialize<S: Serializer>(v: &f64, s: S) -> Result<S::Ok, S::Error> {
        if v.is_nan() {
            s.serialize_str("NaN")
        } else if *v == f64::INFINITY {
            s.serialize_str("Infinity")
        } else if *v == f64::NEG_INFINITY {
            s.serialize_str("-Infinity")
        } else {
            s.serialize_f64(*v)
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<f64, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Number {
            Null(()),
            Float(f64),
            Text(String),
        }
        Ok(match Number::deserialize(d)? {
            Number::Null(()) => f64::NAN,
            Number::Float(v) if v >= f64::MAX => f64::INFINITY,
            Number::Float(v) if v <= f64::MIN => f64::NEG_INFINITY,
            Number::Float(v) => v,
            Number::Text(t) => match t.trim() {
                "Infinity" | "+Infinity" => f64::INFINITY,
                "-Infinity" => f64::NEG_INFINITY,
                "NaN" => f64::NAN,
                other => other
                    .parse::<f64>()
                    .map_err(|_| serde::de::Error::custom(format!("not a number: {other}")))?,
            },
        })
    }
}

// ---------------------------------------------------------------------------
// Names
// ---------------------------------------------------------------------------

/// A biome, by namespaced name. Deserialises from `"plains"`,
/// `"minecraft:plains"` or a 1.12 name; serialises without `minecraft:`.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BiomeDesc(pub String);

impl BiomeDesc {
    pub fn new(name: &str) -> Self {
        Self(normalize_biome(name))
    }
    /// The namespaced name (`minecraft:plains`).
    pub fn name(&self) -> &str {
        &self.0
    }
}

impl Serialize for BiomeDesc {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self.0.strip_prefix("minecraft:").unwrap_or(&self.0))
    }
}
impl<'de> Deserialize<'de> for BiomeDesc {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        Ok(Self::new(&String::deserialize(d)?))
    }
}

/// `plains` -> `minecraft:plains`, 1.12 names translated.
pub fn normalize_biome(name: &str) -> String {
    let (ns, path) = match name.split_once(':') {
        Some((ns, path)) => (ns, path),
        None => ("minecraft", name),
    };
    if ns != "minecraft" {
        return format!("{ns}:{path}");
    }
    let path = match path {
        "extreme_hills" | "smaller_extreme_hills" | "extreme_hills_edge" => "windswept_hills",
        "extreme_hills_with_trees" | "mutated_extreme_hills_with_trees" => "windswept_forest",
        "mutated_extreme_hills" => "windswept_gravelly_hills",
        "swampland" | "mutated_swampland" => "swamp",
        "ice_flats" => "snowy_plains",
        "ice_mountains" => "snowy_slopes",
        "taiga_cold" | "taiga_cold_hills" => "snowy_taiga",
        "taiga_hills" | "mutated_taiga" => "taiga",
        "roofed_forest" | "mutated_roofed_forest" => "dark_forest",
        "mushroom_island" | "mushroom_island_shore" => "mushroom_fields",
        "beaches" => "beach",
        "cold_beach" => "snowy_beach",
        "stone_beach" => "stony_shore",
        "forest_hills" => "forest",
        "birch_forest_hills" => "birch_forest",
        "jungle_hills" | "mutated_jungle" => "jungle",
        "jungle_edge" | "mutated_jungle_edge" => "sparse_jungle",
        "desert_hills" | "mutated_desert" => "desert",
        "redwood_taiga" | "redwood_taiga_hills" => "old_growth_pine_taiga",
        "mutated_redwood_taiga" | "mutated_redwood_taiga_hills" => "old_growth_spruce_taiga",
        "savanna_rock" => "savanna_plateau",
        "mutated_savanna" | "mutated_savanna_rock" => "windswept_savanna",
        "mesa" | "mesa_clear_rock" => "badlands",
        "mesa_rock" => "wooded_badlands",
        "mutated_mesa" | "mutated_mesa_rock" | "mutated_mesa_clear_rock" => "eroded_badlands",
        "mutated_plains" => "sunflower_plains",
        "mutated_forest" => "flower_forest",
        "mutated_ice_flats" => "ice_spikes",
        "mutated_birch_forest" | "mutated_birch_forest_hills" => "old_growth_birch_forest",
        "hell" => "nether_wastes",
        "sky" => "the_end",
        other => other,
    };
    format!("minecraft:{path}")
}

/// A 1.12 biome id translated to a biome name (`-1` is handled by the caller).
pub fn legacy_biome_id(id: i64) -> Option<&'static str> {
    Some(match id {
        0 => "ocean",
        1 => "plains",
        2 => "desert",
        3 => "extreme_hills",
        4 => "forest",
        5 => "taiga",
        6 => "swampland",
        7 => "river",
        8 => "hell",
        9 => "sky",
        10 => "frozen_ocean",
        11 => "frozen_river",
        12 => "ice_flats",
        13 => "ice_mountains",
        14 => "mushroom_island",
        15 => "mushroom_island_shore",
        16 => "beaches",
        17 => "desert_hills",
        18 => "forest_hills",
        19 => "taiga_hills",
        20 => "smaller_extreme_hills",
        21 => "jungle",
        22 => "jungle_hills",
        23 => "jungle_edge",
        24 => "deep_ocean",
        25 => "stone_beach",
        26 => "cold_beach",
        27 => "birch_forest",
        28 => "birch_forest_hills",
        29 => "roofed_forest",
        30 => "taiga_cold",
        31 => "taiga_cold_hills",
        32 => "redwood_taiga",
        33 => "redwood_taiga_hills",
        34 => "extreme_hills_with_trees",
        35 => "savanna",
        36 => "savanna_rock",
        37 => "mesa",
        38 => "mesa_rock",
        39 => "mesa_clear_rock",
        127 => "void",
        129 => "mutated_plains",
        130 => "mutated_desert",
        131 => "mutated_extreme_hills",
        132 => "mutated_forest",
        133 => "mutated_taiga",
        134 => "mutated_swampland",
        140 => "mutated_ice_flats",
        149 => "mutated_jungle",
        151 => "mutated_jungle_edge",
        155 => "mutated_birch_forest",
        156 => "mutated_birch_forest_hills",
        157 => "mutated_roofed_forest",
        158 => "mutated_taiga_cold",
        160 => "mutated_redwood_taiga",
        161 => "mutated_redwood_taiga_hills",
        162 => "mutated_extreme_hills_with_trees",
        163 => "mutated_savanna",
        164 => "mutated_savanna_rock",
        165 => "mutated_mesa",
        166 => "mutated_mesa_rock",
        167 => "mutated_mesa_clear_rock",
        _ => return None,
    })
}

/// The 26.3 biomes a CWG Java biome class (`includeClass` / `excludeClass`
/// filters) stands for, subclasses included. Unknown classes match nothing.
pub fn class_biomes(class_name: &str) -> &'static [&'static str] {
    let simple = class_name.rsplit('.').next().unwrap_or(class_name);
    match simple {
        "BiomeMesa" => &[
            "minecraft:badlands",
            "minecraft:eroded_badlands",
            "minecraft:wooded_badlands",
        ],
        "BiomeSavannaMutated" => &["minecraft:windswept_savanna"],
        "BiomeSavanna" => &[
            "minecraft:savanna",
            "minecraft:savanna_plateau",
            "minecraft:windswept_savanna",
        ],
        "BiomeSwamp" => &["minecraft:swamp", "minecraft:mangrove_swamp"],
        "BiomeDesert" => &["minecraft:desert"],
        "BiomeHills" => &[
            "minecraft:windswept_hills",
            "minecraft:windswept_forest",
            "minecraft:windswept_gravelly_hills",
        ],
        "BiomeForest" => &[
            "minecraft:forest",
            "minecraft:flower_forest",
            "minecraft:birch_forest",
            "minecraft:old_growth_birch_forest",
            "minecraft:dark_forest",
            "minecraft:cherry_grove",
            "minecraft:pale_garden",
        ],
        "BiomeTaiga" => &[
            "minecraft:taiga",
            "minecraft:snowy_taiga",
            "minecraft:old_growth_pine_taiga",
            "minecraft:old_growth_spruce_taiga",
        ],
        "BiomeJungle" => &[
            "minecraft:jungle",
            "minecraft:sparse_jungle",
            "minecraft:bamboo_jungle",
        ],
        "BiomeSnow" => &[
            "minecraft:snowy_plains",
            "minecraft:ice_spikes",
            "minecraft:snowy_taiga",
        ],
        "BiomeOcean" => &[
            "minecraft:ocean",
            "minecraft:deep_ocean",
            "minecraft:cold_ocean",
            "minecraft:deep_cold_ocean",
            "minecraft:frozen_ocean",
            "minecraft:deep_frozen_ocean",
            "minecraft:lukewarm_ocean",
            "minecraft:deep_lukewarm_ocean",
            "minecraft:warm_ocean",
        ],
        "BiomeBeach" => &["minecraft:beach", "minecraft:snowy_beach"],
        "BiomeStoneBeach" => &["minecraft:stony_shore"],
        "BiomeRiver" => &["minecraft:river", "minecraft:frozen_river"],
        "BiomePlains" => &["minecraft:plains", "minecraft:sunflower_plains"],
        "BiomeMushroomIsland" => &["minecraft:mushroom_fields"],
        _ => &[],
    }
}

/// A block state in CWG's `BlockStateDesc` forms.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BlockStateDesc {
    /// Namespaced name (`minecraft:stone`).
    pub name: String,
    /// Non-default properties.
    pub properties: BTreeMap<String, String>,
}

impl BlockStateDesc {
    pub fn new(name: &str) -> Self {
        Self {
            name: namespaced(name),
            properties: BTreeMap::new(),
        }
    }
    pub fn with(name: &str, props: &[(&str, &str)]) -> Self {
        Self {
            name: namespaced(name),
            properties: props
                .iter()
                .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
                .collect(),
        }
    }
}

fn namespaced(name: &str) -> String {
    if name.contains(':') {
        name.to_owned()
    } else {
        format!("minecraft:{name}")
    }
}

impl Serialize for BlockStateDesc {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let short = self.name.strip_prefix("minecraft:").unwrap_or(&self.name);
        if self.properties.is_empty() {
            s.serialize_str(short)
        } else {
            use serde::ser::SerializeMap;
            let mut map = s.serialize_map(Some(2))?;
            map.serialize_entry("Name", short)?;
            map.serialize_entry("Properties", &self.properties)?;
            map.end()
        }
    }
}

impl<'de> Deserialize<'de> for BlockStateDesc {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Form {
            Name(String),
            Full {
                #[serde(rename = "Name")]
                name: String,
                #[serde(rename = "Properties", default)]
                properties: BTreeMap<String, serde_json::Value>,
            },
        }
        Ok(match Form::deserialize(d)? {
            Form::Name(name) => Self::new(&name),
            Form::Full { name, properties } => Self {
                name: namespaced(&name),
                properties: properties
                    .into_iter()
                    .map(|(k, v)| {
                        let v = match v {
                            serde_json::Value::String(s) => s,
                            other => other.to_string(),
                        };
                        (k, v)
                    })
                    .collect(),
            },
        })
    }
}

/// A 1.12 block (`name`, properties) translated to its 26.3 name and
/// properties. Unknown blocks pass through unchanged.
pub fn legacy_block(
    name: &str,
    props: &BTreeMap<String, String>,
) -> (String, BTreeMap<String, String>) {
    let path = name.strip_prefix("minecraft:").unwrap_or(name);
    if name.contains(':') && !name.starts_with("minecraft:") {
        return (name.to_owned(), props.clone());
    }
    let mut props = props.clone();
    let variant = props.get("variant").cloned();
    let color = props.get("color").cloned();
    let renamed: Option<String> = match path {
        "stone" => match variant.as_deref() {
            Some("granite") => Some("granite".into()),
            Some("smooth_granite") => Some("polished_granite".into()),
            Some("diorite") => Some("diorite".into()),
            Some("smooth_diorite") => Some("polished_diorite".into()),
            Some("andesite") => Some("andesite".into()),
            Some("smooth_andesite") => Some("polished_andesite".into()),
            _ => None,
        },
        "dirt" => match variant.as_deref() {
            Some("coarse_dirt") => Some("coarse_dirt".into()),
            Some("podzol") => Some("podzol".into()),
            _ => None,
        },
        "sand" if variant.as_deref() == Some("red_sand") => Some("red_sand".into()),
        "grass" => Some("grass_block".into()),
        "tallgrass" => Some("short_grass".into()),
        "waterlily" => Some("lily_pad".into()),
        "monster_egg" => Some("infested_stone".into()),
        "flowing_water" => Some("water".into()),
        "flowing_lava" => Some("lava".into()),
        "hardened_clay" => Some("terracotta".into()),
        "stained_hardened_clay" => Some(format!(
            "{}_terracotta",
            color
                .as_deref()
                .unwrap_or("white")
                .replace("silver", "light_gray")
        )),
        "snow_layer" => Some("snow".into()),
        "quartz_ore" => Some("nether_quartz_ore".into()),
        "lit_redstone_ore" => Some("redstone_ore".into()),
        "magma" => Some("magma_block".into()),
        "netherrack" | "gravel" | "clay" | "bedrock" => None,
        _ => None,
    };
    match renamed {
        Some(new_name) => {
            props.remove("variant");
            props.remove("color");
            (format!("minecraft:{new_name}"), props)
        }
        None => (format!("minecraft:{path}"), props),
    }
}

// ---------------------------------------------------------------------------
// Conditions and filters
// ---------------------------------------------------------------------------

/// `GenerationCondition`: when an ore may spawn or a block may be replaced.
#[derive(Clone, Debug, PartialEq)]
pub enum GenerationCondition {
    AnyOf(Vec<GenerationCondition>),
    AllOf(Vec<GenerationCondition>),
    NoneOf(Vec<GenerationCondition>),
    /// `{"rand": p}`: a fresh random number below `p`.
    Rand(f64),
    /// `{"posRand": p}`: a hash of the world seed and the position below `p`.
    PosRand(f64),
    /// `{"posRandSeed": p, "seed": n}`.
    PosRandSeed {
        chance: f64,
        seed: i64,
    },
    /// `{"x":..,"y":..,"z":..,"blocks": state | [states]}`: the block at the
    /// offset is one of `blocks`.
    BlockMatch {
        x: i32,
        y: i32,
        z: i32,
        blocks: Vec<BlockStateDesc>,
    },
}

impl GenerationCondition {
    /// Natural stone (CWG's default `placeBlockWhen` of veins).
    pub fn stone_only() -> Self {
        Self::BlockMatch {
            x: 0,
            y: 0,
            z: 0,
            blocks: vec![BlockStateDesc::new("stone")],
        }
    }
}

impl Serialize for GenerationCondition {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let mut map = s.serialize_map(None)?;
        match self {
            Self::AnyOf(v) => map.serialize_entry("anyOf", v)?,
            Self::AllOf(v) => map.serialize_entry("allOf", v)?,
            Self::NoneOf(v) => map.serialize_entry("noneOf", v)?,
            Self::Rand(p) => map.serialize_entry("rand", p)?,
            Self::PosRand(p) => map.serialize_entry("posRand", p)?,
            Self::PosRandSeed { chance, seed } => {
                map.serialize_entry("posRandSeed", chance)?;
                map.serialize_entry("seed", seed)?;
            }
            Self::BlockMatch { x, y, z, blocks } => {
                map.serialize_entry("x", x)?;
                map.serialize_entry("y", y)?;
                map.serialize_entry("z", z)?;
                if blocks.len() == 1 {
                    map.serialize_entry("blocks", &blocks[0])?;
                } else {
                    map.serialize_entry("blocks", blocks)?;
                }
            }
        }
        map.end()
    }
}

impl<'de> Deserialize<'de> for GenerationCondition {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        use serde::de::Error;
        let value = serde_json::Value::deserialize(d)?;
        let obj = value
            .as_object()
            .ok_or_else(|| D::Error::custom("a generation condition is an object"))?;
        let list = |key: &str| -> Result<Vec<GenerationCondition>, D::Error> {
            obj[key]
                .as_array()
                .ok_or_else(|| D::Error::custom(format!("{key} is an array")))?
                .iter()
                .map(|v| serde_json::from_value(v.clone()).map_err(D::Error::custom))
                .collect()
        };
        let num = |key: &str| -> Result<f64, D::Error> {
            obj[key]
                .as_f64()
                .ok_or_else(|| D::Error::custom(format!("{key} is a number")))
        };
        if obj.contains_key("anyOf") {
            Ok(Self::AnyOf(list("anyOf")?))
        } else if obj.contains_key("allOf") {
            Ok(Self::AllOf(list("allOf")?))
        } else if obj.contains_key("noneOf") {
            Ok(Self::NoneOf(list("noneOf")?))
        } else if obj.contains_key("rand") {
            Ok(Self::Rand(num("rand")?))
        } else if obj.contains_key("posRand") {
            Ok(Self::PosRand(num("posRand")?))
        } else if obj.contains_key("posRandSeed") {
            Ok(Self::PosRandSeed {
                chance: num("posRandSeed")?,
                seed: obj
                    .get("seed")
                    .and_then(|v| v.as_i64())
                    .ok_or_else(|| D::Error::custom("posRandSeed needs an integer seed"))?,
            })
        } else {
            let coord = |k: &str| obj.get(k).and_then(|v| v.as_i64()).unwrap_or(0) as i32;
            let blocks = match obj.get("blocks") {
                Some(serde_json::Value::Array(a)) => a
                    .iter()
                    .map(|v| serde_json::from_value(v.clone()).map_err(D::Error::custom))
                    .collect::<Result<Vec<_>, _>>()?,
                Some(v) => vec![serde_json::from_value(v.clone()).map_err(D::Error::custom)?],
                None => return Err(D::Error::custom("unknown generation condition")),
            };
            Ok(Self::BlockMatch {
                x: coord("x"),
                y: coord("y"),
                z: coord("z"),
                blocks,
            })
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum FilterType {
    Include,
    #[default]
    Exclude,
}

/// How `includeClass` / `excludeClass` match; kept for round trips (the
/// class mapping already includes subclasses).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum BiomeClassMatchType {
    #[default]
    RawExact,
    BiomeclassExact,
    RawWithSubclasses,
    BiomeclassWithSubclasses,
}

/// `BiomeFilter` of a replacer.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum BiomeFilter {
    Include {
        #[serde(default)]
        biomes: Vec<BiomeDesc>,
    },
    Exclude {
        #[serde(default)]
        biomes: Vec<BiomeDesc>,
    },
    #[serde(rename_all = "camelCase")]
    IncludeClass {
        #[serde(default)]
        match_type: BiomeClassMatchType,
        #[serde(default)]
        class_names: Vec<String>,
    },
    #[serde(rename_all = "camelCase")]
    ExcludeClass {
        #[serde(default)]
        match_type: BiomeClassMatchType,
        #[serde(default)]
        class_names: Vec<String>,
    },
    AllOf {
        #[serde(default)]
        filters: Vec<BiomeFilter>,
    },
    AnyOf {
        #[serde(default)]
        filters: Vec<BiomeFilter>,
    },
    NoneOf {
        #[serde(default)]
        filters: Vec<BiomeFilter>,
    },
}

impl BiomeFilter {
    pub fn include(names: &[&str]) -> Self {
        Self::Include {
            biomes: names.iter().map(|n| BiomeDesc::new(n)).collect(),
        }
    }
    pub fn exclude(names: &[&str]) -> Self {
        Self::Exclude {
            biomes: names.iter().map(|n| BiomeDesc::new(n)).collect(),
        }
    }
    /// Whether the biome (namespaced name) passes the filter.
    pub fn test(&self, biome: &str) -> bool {
        match self {
            Self::Include { biomes } => biomes.iter().any(|b| b.0 == biome),
            Self::Exclude { biomes } => !biomes.iter().any(|b| b.0 == biome),
            Self::IncludeClass { class_names, .. } => {
                class_names.iter().any(|c| class_biomes(c).contains(&biome))
            }
            Self::ExcludeClass { class_names, .. } => {
                !class_names.iter().any(|c| class_biomes(c).contains(&biome))
            }
            Self::AllOf { filters } => filters.iter().all(|f| f.test(biome)),
            Self::AnyOf { filters } => filters.iter().any(|f| f.test(biome)),
            Self::NoneOf { filters } => !filters.iter().any(|f| f.test(biome)),
        }
    }
}

// ---------------------------------------------------------------------------
// UserFunction
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct UserFunctionPoint {
    pub y: f64,
    pub v: f64,
}

/// A piecewise-linear function of Y: an array of `{"y":..,"v":..}` points.
/// Like CWG's it extrapolates linearly from the first or last segment.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
#[serde(transparent)]
pub struct UserFunction(pub Vec<UserFunctionPoint>);

impl UserFunction {
    pub fn new(points: &[(f64, f64)]) -> Self {
        let mut v: Vec<_> = points
            .iter()
            .map(|&(y, v)| UserFunctionPoint { y, v })
            .collect();
        v.sort_by(|a, b| a.y.total_cmp(&b.y));
        Self(v)
    }
    pub fn get(&self, y: f64) -> f64 {
        let p = &self.0;
        match p.len() {
            0 => 0.0,
            1 => p[0].v,
            _ => {
                let (mut e1, mut e2) = (p[0], p[1]);
                for i in 2..p.len() {
                    if p[i - 1].y < y {
                        e1 = e2;
                        e2 = p[i];
                    }
                }
                let t = if e2.y == e1.y {
                    0.0
                } else {
                    (y - e1.y) / (e2.y - e1.y)
                };
                e1.v + t * (e2.v - e1.v)
            }
        }
    }

    /// The smallest and largest value over the real interval `[y0, y1]`
    /// (piecewise linear, so at the ends and the breakpoints inside).
    pub fn range(&self, y0: i32, y1: i32) -> (f64, f64) {
        let (a, b) = (f64::from(y0), f64::from(y1));
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        let mut see = |y: f64| {
            let v = self.get(y);
            lo = lo.min(v);
            hi = hi.max(v);
        };
        see(a);
        see(b);
        for p in &self.0 {
            if p.y > a && p.y < b {
                see(p.y);
            }
        }
        if lo.is_nan() || hi.is_nan() {
            return (f64::NEG_INFINITY, f64::INFINITY);
        }
        (lo, hi)
    }
}

// ---------------------------------------------------------------------------
// Ores, lakes, caves
// ---------------------------------------------------------------------------

fn neg_inf() -> f64 {
    f64::NEG_INFINITY
}
fn pos_inf() -> f64 {
    f64::INFINITY
}
fn one() -> f64 {
    1.0
}

/// `StandardOreConfig`. Heights are normalised: `blockY = h *
/// expectedHeightVariation + expectedBaseHeight`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct StandardOreConfig {
    pub blockstate: BlockStateDesc,
    /// `None` = no biome restriction.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub biomes: Option<Vec<BiomeDesc>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub place_block_when: Option<GenerationCondition>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generate_when: Option<GenerationCondition>,
    pub spawn_size: i32,
    pub spawn_tries: i32,
    #[serde(default = "one")]
    pub spawn_probability: f64,
    #[serde(with = "inf", default = "neg_inf")]
    pub min_height: f64,
    #[serde(with = "inf", default = "pos_inf")]
    pub max_height: f64,
}

impl Default for StandardOreConfig {
    fn default() -> Self {
        Self {
            blockstate: BlockStateDesc::new("stone"),
            biomes: None,
            place_block_when: None,
            generate_when: None,
            spawn_size: 8,
            spawn_tries: 1,
            spawn_probability: 1.0,
            min_height: f64::NEG_INFINITY,
            max_height: f64::INFINITY,
        }
    }
}

/// `PeriodicGaussianOreConfig`: the vein count follows a bell curve of Y that
/// repeats every `heightSpacing`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct PeriodicGaussianOreConfig {
    pub blockstate: BlockStateDesc,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub biomes: Option<Vec<BiomeDesc>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub place_block_when: Option<GenerationCondition>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generate_when: Option<GenerationCondition>,
    pub spawn_size: i32,
    pub spawn_tries: i32,
    pub spawn_probability: f64,
    pub height_mean: f64,
    pub height_std_deviation: f64,
    pub height_spacing: f64,
    #[serde(with = "inf", default = "neg_inf")]
    pub min_height: f64,
    #[serde(with = "inf", default = "pos_inf")]
    pub max_height: f64,
}

impl Default for PeriodicGaussianOreConfig {
    fn default() -> Self {
        Self {
            blockstate: BlockStateDesc::new("stone"),
            biomes: None,
            place_block_when: None,
            generate_when: None,
            spawn_size: 8,
            spawn_tries: 1,
            spawn_probability: 1.0,
            height_mean: 0.0,
            height_std_deviation: 0.1,
            height_spacing: 1.0,
            min_height: f64::NEG_INFINITY,
            max_height: f64::INFINITY,
        }
    }
}

/// A lake. `block` is a block name (not a state): `"water"`, `"lava"`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct LakeConfig {
    pub block: String,
    pub biomes: Vec<BiomeDesc>,
    pub biome_select: FilterType,
    pub surface_probability: UserFunction,
    pub main_probability: UserFunction,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generate_when: Option<GenerationCondition>,
}

impl Default for LakeConfig {
    fn default() -> Self {
        Self {
            block: "water".into(),
            biomes: Vec::new(),
            biome_select: FilterType::Exclude,
            surface_probability: UserFunction::default(),
            main_probability: UserFunction::default(),
            generate_when: None,
        }
    }
}

/// `CaveConfig`, every field with CWG's default.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct CaveConfig {
    pub cave_block: BlockStateDesc,
    /// Lowest cube Y whose origin may start caves.
    pub cave_min_height: i32,
    pub cave_max_height: i32,
    pub cave_rarity: i32,
    pub max_init_nodes: i32,
    pub large_node_rarity: i32,
    pub large_node_max_branches: i32,
    pub big_cave_rarity: i32,
    pub cave_size_add: f64,
    pub steep_step_rarity: i32,
    pub flatten_factor: f64,
    pub steeper_flatten_factor: f64,
    pub direction_change_factor: f64,
    pub prev_horiz_direction_change_weight: f64,
    pub prev_vert_direction_change_weight: f64,
    pub max_add_direction_change_horiz: f64,
    pub max_add_direction_change_vert: f64,
    pub carve_step_rarity: i32,
    pub cave_floor_depth: f64,
    pub is_block_replaceable: Vec<BlockStateDesc>,
}

impl Default for CaveConfig {
    /// CWG's defaults; `isBlockReplaceable` is empty, as in Java (use
    /// [`CaveConfig::overworld`] for the preset's list).
    fn default() -> Self {
        Self {
            cave_block: BlockStateDesc::new("air"),
            cave_min_height: i32::MIN / 16,
            cave_max_height: i32::MAX / 16,
            cave_rarity: 16 * 7 / 8,
            max_init_nodes: 14,
            large_node_rarity: 4,
            large_node_max_branches: 4,
            big_cave_rarity: 10,
            cave_size_add: 1.5,
            steep_step_rarity: 6,
            flatten_factor: f64::from(0.7_f32),
            steeper_flatten_factor: f64::from(0.92_f32),
            direction_change_factor: f64::from(0.1_f32),
            prev_horiz_direction_change_weight: f64::from(0.75_f32),
            prev_vert_direction_change_weight: f64::from(0.9_f32),
            max_add_direction_change_horiz: 4.0,
            max_add_direction_change_vert: 2.0,
            carve_step_rarity: 4,
            cave_floor_depth: -0.7,
            is_block_replaceable: Vec::new(),
        }
    }
}

impl CaveConfig {
    /// The preset's cave: CWG's list (grass, dirt, stone) widened with the
    /// blocks 26.3's overworld has at the surface and below.
    pub fn overworld() -> Self {
        let names = [
            "grass_block",
            "dirt",
            "stone",
            "deepslate",
            "granite",
            "diorite",
            "andesite",
            "tuff",
            "coarse_dirt",
            "podzol",
            "sand",
            "red_sand",
            "sandstone",
            "red_sandstone",
            "gravel",
            "terracotta",
            "mycelium",
            "snow_block",
        ];
        Self {
            is_block_replaceable: names.iter().map(|n| BlockStateDesc::new(n)).collect(),
            ..Self::default()
        }
    }
}

// ---------------------------------------------------------------------------
// Replacers
// ---------------------------------------------------------------------------

fn min_y_default() -> i32 {
    i32::MIN
}
fn max_y_default() -> i32 {
    i32::MAX
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum NoiseType {
    PerlinFlowNoise,
    #[default]
    SimplexSpongeNoise,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum DecorationNoiseSource {
    TemperatureNoise,
    #[default]
    GrassColorNoise,
    FlowNoisePerlin,
    SpongeNoiseSimplex,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum MesaNoiseSource {
    #[default]
    MesaClayBandsOffsetNoise,
    MinecraftPerlinRaw,
    FlowNoisePerlin,
    SpongeNoiseSimplex,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct DensityRangeReplacerConfig {
    #[serde(default = "min_y_default")]
    pub min_y: i32,
    #[serde(default = "max_y_default")]
    pub max_y: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub biome_filter: Option<BiomeFilter>,
    /// Placed where the density is in range; `null` keeps the block.
    pub block_in_range: Option<BlockStateDesc>,
    pub block_out_of_range: Option<BlockStateDesc>,
    pub filter_blocks: Vec<BlockStateDesc>,
    pub block_filter_type: FilterType,
    #[serde(with = "inf", default = "neg_inf")]
    pub min_density: f64,
    #[serde(with = "inf", default = "pos_inf")]
    pub max_density: f64,
}

impl Default for DensityRangeReplacerConfig {
    fn default() -> Self {
        Self {
            min_y: i32::MIN,
            max_y: i32::MAX,
            biome_filter: None,
            block_in_range: None,
            block_out_of_range: None,
            filter_blocks: Vec::new(),
            block_filter_type: FilterType::Exclude,
            min_density: f64::NEG_INFINITY,
            max_density: f64::INFINITY,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct RandomYGradientReplacerConfig {
    #[serde(default = "min_y_default")]
    pub min_y: i32,
    #[serde(default = "max_y_default")]
    pub max_y: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub biome_filter: Option<BiomeFilter>,
    pub block_to_place: Option<BlockStateDesc>,
    pub probability_function: UserFunction,
    pub seed: i32,
    /// Horizon extension: only these blocks (or, with `EXCLUDE`, all but
    /// these) are replaced. Empty = every block, as in CWG.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub filter_blocks: Vec<BlockStateDesc>,
    /// Horizon extension; see `filterBlocks`.
    pub block_filter_type: FilterType,
}

impl Default for RandomYGradientReplacerConfig {
    fn default() -> Self {
        Self {
            min_y: i32::MIN,
            max_y: i32::MAX,
            biome_filter: None,
            block_to_place: None,
            probability_function: UserFunction::default(),
            seed: 0,
            filter_blocks: Vec::new(),
            block_filter_type: FilterType::Exclude,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct MainSurfaceReplacerConfig {
    #[serde(default = "min_y_default")]
    pub min_y: i32,
    #[serde(default = "max_y_default")]
    pub max_y: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub biome_filter: Option<BiomeFilter>,
    pub surface_depth_noise_type: NoiseType,
    pub surface_depth_noise_seed: i32,
    pub surface_depth_noise_frequency_x: f64,
    pub surface_depth_noise_frequency_y: f64,
    pub surface_depth_noise_frequency_z: f64,
    pub surface_depth_noise_octaves: i32,
    pub surface_depth_noise_factor: f64,
    pub surface_depth_noise_offset: f64,
    pub max_surface_depth: f64,
    pub horizontal_gradient_depth_decrease_weight: f64,
    pub ocean_level: f64,
    pub override_top: Option<BlockStateDesc>,
    pub override_filler: Option<BlockStateDesc>,
}

/// The surface depth noise of the preset: vanilla's `0.0625 / 2^(4-1)`
/// frequency, 4 octaves, `0.55 * 15 / 3` factor, offset 3.
pub const SURFACE_NOISE_FREQUENCY: f64 = 0.0625 / 8.0;
pub const SURFACE_NOISE_FACTOR: f64 = 0.55 * 15.0 / 3.0;

impl Default for MainSurfaceReplacerConfig {
    fn default() -> Self {
        Self {
            min_y: i32::MIN,
            max_y: i32::MAX,
            biome_filter: None,
            surface_depth_noise_type: NoiseType::SimplexSpongeNoise,
            surface_depth_noise_seed: 0,
            surface_depth_noise_frequency_x: SURFACE_NOISE_FREQUENCY,
            surface_depth_noise_frequency_y: 0.0,
            surface_depth_noise_frequency_z: SURFACE_NOISE_FREQUENCY,
            surface_depth_noise_octaves: 4,
            surface_depth_noise_factor: SURFACE_NOISE_FACTOR,
            surface_depth_noise_offset: 3.0,
            max_surface_depth: 9.0,
            horizontal_gradient_depth_decrease_weight: 1.0,
            ocean_level: 63.0,
            override_top: None,
            override_filler: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DepthEntry {
    pub y: f64,
    pub b: BlockStateDesc,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct DepthBasedSurfaceReplacerConfig {
    #[serde(default = "min_y_default")]
    pub min_y: i32,
    #[serde(default = "max_y_default")]
    pub max_y: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub biome_filter: Option<BiomeFilter>,
    pub surface_depth_noise_type: NoiseType,
    pub surface_depth_noise_seed: i32,
    pub surface_depth_noise_frequency_x: f64,
    pub surface_depth_noise_frequency_y: f64,
    pub surface_depth_noise_frequency_z: f64,
    pub surface_depth_noise_octaves: i32,
    pub surface_depth_noise_factor: f64,
    pub surface_depth_noise_offset: f64,
    pub max_surface_depth: f64,
    pub horizontal_gradient_depth_decrease_weight: f64,
    pub ocean_level: f64,
    pub override_top: Option<BlockStateDesc>,
    pub override_filler: Option<BlockStateDesc>,
    pub top_thresholds: Vec<DepthEntry>,
    pub filler_thresholds: Vec<DepthEntry>,
}

impl Default for DepthBasedSurfaceReplacerConfig {
    fn default() -> Self {
        let m = MainSurfaceReplacerConfig::default();
        Self {
            min_y: m.min_y,
            max_y: m.max_y,
            biome_filter: None,
            surface_depth_noise_type: m.surface_depth_noise_type,
            surface_depth_noise_seed: m.surface_depth_noise_seed,
            surface_depth_noise_frequency_x: m.surface_depth_noise_frequency_x,
            surface_depth_noise_frequency_y: m.surface_depth_noise_frequency_y,
            surface_depth_noise_frequency_z: m.surface_depth_noise_frequency_z,
            surface_depth_noise_octaves: m.surface_depth_noise_octaves,
            surface_depth_noise_factor: m.surface_depth_noise_factor,
            surface_depth_noise_offset: m.surface_depth_noise_offset,
            max_surface_depth: m.max_surface_depth,
            horizontal_gradient_depth_decrease_weight: m.horizontal_gradient_depth_decrease_weight,
            ocean_level: m.ocean_level,
            override_top: None,
            override_filler: None,
            top_thresholds: Vec::new(),
            filler_thresholds: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct MesaSurfaceReplacerConfig {
    #[serde(default = "min_y_default")]
    pub min_y: i32,
    #[serde(default = "max_y_default")]
    pub max_y: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub biome_filter: Option<BiomeFilter>,
    pub mesa_depth: f64,
    pub height_offset: f64,
    pub height_scale: f64,
    pub water_height: f64,
    pub surface_depth_noise_type: NoiseType,
    pub surface_depth_noise_seed: i32,
    pub surface_depth_noise_frequency_x: f64,
    pub surface_depth_noise_frequency_y: f64,
    pub surface_depth_noise_frequency_z: f64,
    pub surface_depth_noise_octaves: i32,
    pub surface_depth_noise_factor: f64,
    pub surface_depth_noise_offset: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub clay_bands_override: Option<Vec<BlockStateDesc>>,
    pub clay_bands_offset_noise_source: MesaNoiseSource,
    pub clay_bands_noise_frequency_x: f64,
    pub clay_bands_noise_frequency_y: f64,
    pub clay_bands_noise_frequency_z: f64,
    pub clay_bands_noise_factor: f64,
    pub clay_bands_noise_offset: f64,
    pub custom_clay_bands_noise_seed: i32,
    pub custom_clay_bands_noise_octaves: i32,
}

impl Default for MesaSurfaceReplacerConfig {
    fn default() -> Self {
        let m = MainSurfaceReplacerConfig::default();
        Self {
            min_y: i32::MIN,
            max_y: i32::MAX,
            biome_filter: None,
            mesa_depth: 16.0,
            height_offset: 64.0,
            height_scale: 64.0,
            water_height: 63.0,
            surface_depth_noise_type: m.surface_depth_noise_type,
            surface_depth_noise_seed: 0,
            surface_depth_noise_frequency_x: m.surface_depth_noise_frequency_x,
            surface_depth_noise_frequency_y: 0.0,
            surface_depth_noise_frequency_z: m.surface_depth_noise_frequency_z,
            surface_depth_noise_octaves: 4,
            surface_depth_noise_factor: m.surface_depth_noise_factor,
            surface_depth_noise_offset: 3.0,
            clay_bands_override: None,
            clay_bands_offset_noise_source: MesaNoiseSource::MesaClayBandsOffsetNoise,
            clay_bands_noise_frequency_x: 1.0 / 512.0,
            clay_bands_noise_frequency_y: 0.0,
            clay_bands_noise_frequency_z: 1.0 / 512.0,
            clay_bands_noise_factor: 1.0,
            clay_bands_noise_offset: 0.0,
            custom_clay_bands_noise_seed: 0,
            custom_clay_bands_noise_octaves: 4,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct NoiseBasedSurfaceDecorationConfig {
    #[serde(default = "min_y_default")]
    pub min_y: i32,
    #[serde(default = "max_y_default")]
    pub max_y: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub biome_filter: Option<BiomeFilter>,
    pub surface_density_threshold: f64,
    pub ground_block: Option<BlockStateDesc>,
    pub feature_block: Option<BlockStateDesc>,
    pub noise_source: DecorationNoiseSource,
    pub noise_freq_x: f64,
    pub noise_freq_y: f64,
    pub noise_freq_z: f64,
    pub noise_factor: f64,
    pub noise_offset: f64,
    pub custom_noise_seed: i32,
    pub custom_noise_octaves: i32,
    #[serde(with = "inf", default = "neg_inf")]
    pub feature_min_noise: f64,
    #[serde(with = "inf", default = "pos_inf")]
    pub feature_max_noise: f64,
    #[serde(with = "inf", default = "neg_inf")]
    pub ground_min_noise: f64,
    #[serde(with = "inf", default = "pos_inf")]
    pub ground_max_noise: f64,
}

impl Default for NoiseBasedSurfaceDecorationConfig {
    fn default() -> Self {
        Self {
            min_y: i32::MIN,
            max_y: i32::MAX,
            biome_filter: None,
            surface_density_threshold: 0.0,
            ground_block: None,
            feature_block: None,
            noise_source: DecorationNoiseSource::GrassColorNoise,
            noise_freq_x: 1.0,
            noise_freq_y: 0.0,
            noise_freq_z: 1.0,
            noise_factor: 1.0,
            noise_offset: 0.0,
            custom_noise_seed: 0,
            custom_noise_octaves: 4,
            feature_min_noise: f64::NEG_INFINITY,
            feature_max_noise: f64::INFINITY,
            ground_min_noise: f64::NEG_INFINITY,
            ground_max_noise: f64::INFINITY,
        }
    }
}

/// One replacer, tagged by `"type"` as in CWG.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ReplacerConfig {
    #[serde(rename = "densityRange")]
    DensityRange(DensityRangeReplacerConfig),
    #[serde(rename = "mainSurface")]
    MainSurface(MainSurfaceReplacerConfig),
    #[serde(rename = "depthBasedSurface")]
    DepthBasedSurface(DepthBasedSurfaceReplacerConfig),
    #[serde(rename = "mesaSurface")]
    MesaSurface(MesaSurfaceReplacerConfig),
    #[serde(rename = "noiseBasedSurfaceDecoration")]
    NoiseBasedSurfaceDecoration(NoiseBasedSurfaceDecorationConfig),
    #[serde(rename = "randomYGradient")]
    RandomYGradient(RandomYGradientReplacerConfig),
}

impl ReplacerConfig {
    pub fn min_y(&self) -> i32 {
        match self {
            Self::DensityRange(c) => c.min_y,
            Self::MainSurface(c) => c.min_y,
            Self::DepthBasedSurface(c) => c.min_y,
            Self::MesaSurface(c) => c.min_y,
            Self::NoiseBasedSurfaceDecoration(c) => c.min_y,
            Self::RandomYGradient(c) => c.min_y,
        }
    }
    pub fn max_y(&self) -> i32 {
        match self {
            Self::DensityRange(c) => c.max_y,
            Self::MainSurface(c) => c.max_y,
            Self::DepthBasedSurface(c) => c.max_y,
            Self::MesaSurface(c) => c.max_y,
            Self::NoiseBasedSurfaceDecoration(c) => c.max_y,
            Self::RandomYGradient(c) => c.max_y,
        }
    }
    pub fn biome_filter(&self) -> Option<&BiomeFilter> {
        match self {
            Self::DensityRange(c) => c.biome_filter.as_ref(),
            Self::MainSurface(c) => c.biome_filter.as_ref(),
            Self::DepthBasedSurface(c) => c.biome_filter.as_ref(),
            Self::MesaSurface(c) => c.biome_filter.as_ref(),
            Self::NoiseBasedSurfaceDecoration(c) => c.biome_filter.as_ref(),
            Self::RandomYGradient(c) => c.biome_filter.as_ref(),
        }
    }
}

// ---------------------------------------------------------------------------
// cubeAreas
// ---------------------------------------------------------------------------

/// A box of cubes (inclusive cube coordinates).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase", default)]
pub struct IntAabb {
    pub min_x: i32,
    pub min_y: i32,
    pub min_z: i32,
    pub max_x: i32,
    pub max_y: i32,
    pub max_z: i32,
}

impl IntAabb {
    pub fn contains(&self, x: i32, y: i32, z: i32) -> bool {
        x >= self.min_x
            && x <= self.max_x
            && y >= self.min_y
            && y <= self.max_y
            && z >= self.min_z
            && z <= self.max_z
    }
}

/// One `cubeAreas` entry: `[ box, settings ]`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    from = "(IntAabb, CustomGeneratorSettings)",
    into = "(IntAabb, CustomGeneratorSettings)"
)]
pub struct CubeArea {
    pub area: IntAabb,
    pub settings: Box<CustomGeneratorSettings>,
}

impl From<(IntAabb, CustomGeneratorSettings)> for CubeArea {
    fn from((area, settings): (IntAabb, CustomGeneratorSettings)) -> Self {
        Self {
            area,
            settings: Box::new(settings),
        }
    }
}
impl From<CubeArea> for (IntAabb, CustomGeneratorSettings) {
    fn from(a: CubeArea) -> Self {
        (a.area, *a.settings)
    }
}

// ---------------------------------------------------------------------------
// Structures (Horizon extension)
// ---------------------------------------------------------------------------

/// One structure set's entry in `structureSets`: a field left out is the
/// set's own (vanilla's) value.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct StructureSetTuning {
    /// `false` switches the set off.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    /// Random spread: the side of the grid cell, in chunks; each cell holds
    /// one start at most.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spacing: Option<i32>,
    /// Random spread: chunks at the far side of each cell no start takes, so
    /// starts are at least this far apart. Less than `spacing`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub separation: Option<i32>,
    /// The chance, 0 to 1, that a cell (a chunk, at spacing 1) that could
    /// have a start gets one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency: Option<f32>,
    /// Concentric rings (strongholds): the ring distance unit, in chunks (the
    /// first ring lies about 4 units out, each next one 6 units further).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub distance: Option<i32>,
    /// Concentric rings: starts in the first ring (later rings hold more).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spread: Option<i32>,
    /// Concentric rings: starts in all.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub count: Option<i32>,
}

/// How a structure set places its starts, with its numbers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum VanillaPlacement {
    /// `minecraft:random_spread`.
    Spread {
        spacing: i32,
        separation: i32,
        frequency: f32,
    },
    /// `minecraft:concentric_rings`.
    Rings {
        distance: i32,
        spread: i32,
        count: i32,
    },
}

/// The CubicWorldGen structure toggle a set belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CwgToggle {
    Villages,
    Temples,
    Mineshafts,
    OceanMonuments,
    WoodlandMansions,
    Strongholds,
}

/// A structure set of 26.3's overworld, with its vanilla placement.
#[derive(Clone, Copy, Debug)]
pub struct VanillaStructureSet {
    pub name: &'static str,
    pub toggle: Option<CwgToggle>,
    pub placement: VanillaPlacement,
}

const fn spread(
    name: &'static str,
    toggle: Option<CwgToggle>,
    spacing: i32,
    separation: i32,
    frequency: f32,
) -> VanillaStructureSet {
    VanillaStructureSet {
        name,
        toggle,
        placement: VanillaPlacement::Spread {
            spacing,
            separation,
            frequency,
        },
    }
}

/// 26.3's overworld structure sets (`data/minecraft/worldgen/structure_set`)
/// and their placements, in the order a settings screen lists them.
pub const OVERWORLD_STRUCTURE_SETS: &[VanillaStructureSet] = &[
    spread("minecraft:villages", Some(CwgToggle::Villages), 34, 8, 1.0),
    spread("minecraft:pillager_outposts", None, 32, 8, 0.2),
    spread(
        "minecraft:woodland_mansions",
        Some(CwgToggle::WoodlandMansions),
        80,
        20,
        1.0,
    ),
    spread("minecraft:desert_pyramids", Some(CwgToggle::Temples), 32, 8, 1.0),
    spread("minecraft:jungle_temples", Some(CwgToggle::Temples), 32, 8, 1.0),
    spread("minecraft:swamp_huts", Some(CwgToggle::Temples), 32, 8, 1.0),
    spread("minecraft:igloos", Some(CwgToggle::Temples), 32, 8, 1.0),
    spread("minecraft:abandoned_camp", None, 37, 8, 1.0),
    spread("minecraft:ruined_portals", None, 40, 15, 1.0),
    spread("minecraft:trail_ruins", None, 34, 8, 1.0),
    spread(
        "minecraft:ocean_monuments",
        Some(CwgToggle::OceanMonuments),
        32,
        5,
        1.0,
    ),
    spread("minecraft:ocean_ruins", None, 20, 8, 1.0),
    spread("minecraft:shipwrecks", None, 24, 4, 1.0),
    spread("minecraft:buried_treasures", None, 1, 0, 0.01),
    spread("minecraft:mineshafts", Some(CwgToggle::Mineshafts), 1, 0, 0.004),
    spread("minecraft:trial_chambers", None, 34, 12, 1.0),
    spread("minecraft:ancient_cities", None, 24, 8, 1.0),
    VanillaStructureSet {
        name: "minecraft:strongholds",
        toggle: Some(CwgToggle::Strongholds),
        placement: VanillaPlacement::Rings {
            distance: 32,
            spread: 3,
            count: 128,
        },
    },
];

/// A structure set's full name: `villages` is `minecraft:villages`.
pub fn structure_set_name(set: &str) -> String {
    if set.contains(':') {
        set.to_owned()
    } else {
        format!("minecraft:{set}")
    }
}

/// The overworld set named `set` (a full name).
pub fn vanilla_structure_set(set: &str) -> Option<&'static VanillaStructureSet> {
    OVERWORLD_STRUCTURE_SETS.iter().find(|v| v.name == set)
}

// ---------------------------------------------------------------------------
// The settings object
// ---------------------------------------------------------------------------

mod biome_field {
    use super::*;
    pub fn serialize<S: Serializer>(v: &str, s: S) -> Result<S::Ok, S::Error> {
        if v.is_empty() {
            s.serialize_i32(-1)
        } else {
            s.serialize_str(v.strip_prefix("minecraft:").unwrap_or(v))
        }
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<String, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum B {
            Id(i64),
            Name(String),
            Null(()),
        }
        match B::deserialize(d)? {
            B::Id(-1) | B::Null(()) => Ok(String::new()),
            B::Id(id) => legacy_biome_id(id)
                .map(normalize_biome)
                .ok_or_else(|| serde::de::Error::custom(format!("unknown 1.12 biome id {id}"))),
            B::Name(n) if n.is_empty() || n == "-1" => Ok(String::new()),
            B::Name(n) => Ok(normalize_biome(&n)),
        }
    }
}

/// CubicWorldGen's `CustomGeneratorSettings` (version 9).
///
/// `Default::default()` is what Java's `new CustomGeneratorSettings()` is: the
/// terrain numbers of the preset but empty ore, lake, cave and replacer lists
/// (a JSON object that leaves them out gets no ores, as in CWG). The preset
/// itself is [`CustomGeneratorSettings::defaults`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct CustomGeneratorSettings {
    pub strongholds: bool,
    pub alternate_strongholds_positions: bool,
    pub villages: bool,
    pub mineshafts: bool,
    pub temples: bool,
    pub ocean_monuments: bool,
    pub woodland_mansions: bool,
    /// Horizon extension (`structureSets`): each structure set's switch and
    /// placement, by the set's name (`minecraft:villages`). A set or field
    /// left out is vanilla's ([`OVERWORLD_STRUCTURE_SETS`]); CWG ignores it.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub structure_sets: BTreeMap<String, StructureSetTuning>,
    pub ravines: bool,
    pub dungeons: bool,
    pub dungeon_count: i32,
    pub lakes: Vec<LakeConfig>,
    pub caves: Vec<CaveConfig>,
    /// A biome name for single-biome mode; empty = the normal biome source.
    #[serde(with = "biome_field")]
    pub biome: String,
    pub biome_size: i32,
    pub river_size: i32,
    pub standard_ores: Vec<StandardOreConfig>,
    pub periodic_gaussian_ores: Vec<PeriodicGaussianOreConfig>,
    pub expected_base_height: f64,
    pub expected_height_variation: f64,
    pub actual_height: f64,
    pub height_variation_factor: f64,
    pub special_height_variation_factor_below_average_y: f64,
    pub height_variation_offset: f64,
    pub height_factor: f64,
    pub height_offset: f64,
    pub depth_noise_factor: f64,
    pub depth_noise_offset: f64,
    pub depth_noise_frequency_x: f64,
    pub depth_noise_frequency_z: f64,
    pub depth_noise_octaves: i32,
    pub selector_noise_factor: f64,
    pub selector_noise_offset: f64,
    pub selector_noise_frequency_x: f64,
    pub selector_noise_frequency_y: f64,
    pub selector_noise_frequency_z: f64,
    pub selector_noise_octaves: i32,
    pub low_noise_factor: f64,
    pub low_noise_offset: f64,
    pub low_noise_frequency_x: f64,
    pub low_noise_frequency_y: f64,
    pub low_noise_frequency_z: f64,
    pub low_noise_octaves: i32,
    pub high_noise_factor: f64,
    pub high_noise_offset: f64,
    pub high_noise_frequency_x: f64,
    pub high_noise_frequency_y: f64,
    pub high_noise_frequency_z: f64,
    pub high_noise_octaves: i32,
    pub replacers: Vec<ReplacerConfig>,
    pub noise_sample_size_x: i32,
    pub noise_sample_size_y: i32,
    pub noise_sample_size_z: i32,
    pub cube_areas: Vec<CubeArea>,
    pub version: i32,
}

/// `ConversionUtils.VANILLA_DEPTH_NOISE_FACTOR`: `1024 / 125 / 8`.
pub const VANILLA_DEPTH_NOISE_FACTOR: f64 = 1024.0 / 125.0 / 8.0;

/// `frequencyFromVanilla(f, octaves)`.
fn freq_from_vanilla(f: f64, octaves: i32) -> f64 {
    f / f64::from(1_u32 << (octaves - 1))
}

impl Default for CustomGeneratorSettings {
    fn default() -> Self {
        let depth_freq = freq_from_vanilla(200.0, 16) / 4.0;
        let selector_xz = freq_from_vanilla(684.412 / 80.0, 8) / 4.0;
        let selector_y = freq_from_vanilla(684.412 / 80.0, 8) / 8.0;
        let lowhigh_xz = freq_from_vanilla(684.412, 16) / 4.0;
        let lowhigh_y = freq_from_vanilla(684.412, 16) / 8.0;
        Self {
            strongholds: true,
            alternate_strongholds_positions: false,
            villages: true,
            mineshafts: true,
            temples: true,
            ocean_monuments: true,
            woodland_mansions: true,
            structure_sets: BTreeMap::new(),
            ravines: true,
            dungeons: true,
            dungeon_count: 7,
            lakes: Vec::new(),
            caves: Vec::new(),
            biome: String::new(),
            biome_size: 4,
            river_size: 4,
            standard_ores: Vec::new(),
            periodic_gaussian_ores: Vec::new(),
            expected_base_height: 64.0,
            expected_height_variation: 64.0,
            actual_height: 256.0,
            height_variation_factor: 64.0,
            special_height_variation_factor_below_average_y: f64::from(0.25_f32),
            height_variation_offset: 0.0,
            height_factor: 64.0,
            height_offset: 64.0,
            depth_noise_factor: VANILLA_DEPTH_NOISE_FACTOR,
            depth_noise_offset: 0.0,
            depth_noise_frequency_x: depth_freq,
            depth_noise_frequency_z: depth_freq,
            depth_noise_octaves: 16,
            selector_noise_factor: 12.75,
            selector_noise_offset: 0.5,
            selector_noise_frequency_x: selector_xz,
            selector_noise_frequency_y: selector_y,
            selector_noise_frequency_z: selector_xz,
            selector_noise_octaves: 8,
            low_noise_factor: 1.0,
            low_noise_offset: 0.0,
            low_noise_frequency_x: lowhigh_xz,
            low_noise_frequency_y: lowhigh_y,
            low_noise_frequency_z: lowhigh_xz,
            low_noise_octaves: 16,
            high_noise_factor: 1.0,
            high_noise_offset: 0.0,
            high_noise_frequency_x: lowhigh_xz,
            high_noise_frequency_y: lowhigh_y,
            high_noise_frequency_z: lowhigh_xz,
            high_noise_octaves: 16,
            replacers: Vec::new(),
            noise_sample_size_x: 4,
            noise_sample_size_y: 8,
            noise_sample_size_z: 4,
            cube_areas: Vec::new(),
            version: LATEST_VERSION,
        }
    }
}

// --- default ores (26.3 overworld) -----------------------------------------

/// A uniform-height ore in CWG's terms. `count` is veins per vanilla chunk
/// (all heights); `lo..=hi` are block Ys.
fn uniform_ore(
    block: &str,
    size: i32,
    count: f64,
    lo: f64,
    hi: f64,
    biomes: Option<&[&str]>,
) -> StandardOreConfig {
    // CWG: spawnTries * spawnProbability veins per cube in range
    // = count * 16 / range (vanilla's `256 / range / 16` idea).
    let per_cube = count * 16.0 / (hi - lo + 1.0);
    let tries = per_cube.ceil().max(1.0);
    StandardOreConfig {
        blockstate: BlockStateDesc::new(block),
        biomes: biomes.map(|b| b.iter().map(|n| BiomeDesc::new(n)).collect()),
        spawn_size: size,
        spawn_tries: tries as i32,
        spawn_probability: per_cube / tries,
        min_height: (lo - 64.0) / 64.0,
        max_height: (hi - 64.0) / 64.0,
        ..StandardOreConfig::default()
    }
}

/// A triangle / trapezoid-height ore as a periodic Gaussian ore whose spacing
/// (one million blocks) never repeats in practice. The bell's standard
/// deviation is the triangle's (`half width / sqrt 6`) and its peak keeps the
/// vein total: `tries * probability = 16 * count / (sd * sqrt(2 pi))`.
fn triangle_ore(
    block: &str,
    size: i32,
    count: f64,
    lo: f64,
    hi: f64,
    biomes: Option<&[&str]>,
) -> PeriodicGaussianOreConfig {
    let half = (hi - lo) / 2.0;
    let mean = (lo + hi) / 2.0;
    let sd = half / 6.0_f64.sqrt();
    let peak = 16.0 * count / (sd * (2.0 * std::f64::consts::PI).sqrt());
    let tries = peak.ceil().max(1.0);
    PeriodicGaussianOreConfig {
        blockstate: BlockStateDesc::new(block),
        biomes: biomes.map(|b| b.iter().map(|n| BiomeDesc::new(n)).collect()),
        spawn_size: size,
        spawn_tries: tries as i32,
        spawn_probability: (peak / tries).min(1.0),
        height_mean: (mean - 64.0) / 64.0,
        height_std_deviation: sd / 64.0,
        height_spacing: 1_000_000.0 / 64.0,
        min_height: (lo - 64.0) / 64.0,
        max_height: (hi - 64.0) / 64.0,
        ..PeriodicGaussianOreConfig::default()
    }
}

const MOUNTAIN_BIOMES: &[&str] = &[
    "windswept_hills",
    "windswept_forest",
    "windswept_gravelly_hills",
    "jagged_peaks",
    "frozen_peaks",
    "stony_peaks",
    "snowy_slopes",
    "grove",
];
const BADLANDS_BIOMES: &[&str] = &["badlands", "eroded_badlands", "wooded_badlands"];

impl CustomGeneratorSettings {
    /// The preset (CWG's `defaults()` with 26.3's ores, deepslate and biome
    /// names).
    pub fn defaults() -> Self {
        let mut s = Self::default();
        s.caves.push(CaveConfig::overworld());
        s.lakes = vec![
            LakeConfig {
                block: "lava".into(),
                biomes: Vec::new(),
                biome_select: FilterType::Exclude,
                main_probability: UserFunction::new(&[
                    (0.0, 4.0 / 263.0),
                    (7.0, 4.0 / 263.0),
                    (8.0, 247.0 / 16306.0),
                    (62.0, 193.0 / 16306.0),
                    (63.0, 48.0 / 40765.0),
                    (127.0, 32.0 / 40765.0),
                    (128.0, 32.0 / 40765.0),
                ]),
                surface_probability: UserFunction::new(&[
                    (-1.0, 19921.0 / 326120.0),
                    (0.0, 19921.0 / 326120.0),
                    (31.0, 1332.0 / 40765.0),
                    (63.0, 579.0 / 81530.0),
                    (95.0, 161.0 / 32612.0),
                    (127.0, 129.0 / 40765.0),
                    (128.0, 129.0 / 40765.0),
                ]),
                generate_when: None,
            },
            LakeConfig {
                block: "water".into(),
                biomes: vec![BiomeDesc::new("desert")],
                biome_select: FilterType::Exclude,
                main_probability: UserFunction::new(&[(0.0, 1.0 / 64.0)]),
                surface_probability: UserFunction::new(&[
                    (-1.0, 0.25),
                    (0.0, 0.25),
                    (128.0, 0.125),
                    (129.0, 0.125),
                ]),
                generate_when: None,
            },
        ];

        // Blobs. Granite, diorite and andesite use size 33 (vanilla 64) so a
        // vein stays inside the 2 x 2 x 2 cubes a populate writes; tuff gets
        // twice vanilla's count for the same reason.
        s.standard_ores = vec![
            uniform_ore("dirt", 33, 7.0, 0.0, 160.0, None),
            uniform_ore("gravel", 33, 14.0, -64.0, 320.0, None),
            uniform_ore("granite", 33, 1.0 / 6.0, 64.0, 128.0, None),
            uniform_ore("granite", 33, 2.0, 0.0, 60.0, None),
            uniform_ore("diorite", 33, 1.0 / 6.0, 64.0, 128.0, None),
            uniform_ore("diorite", 33, 2.0, 0.0, 60.0, None),
            uniform_ore("andesite", 33, 1.0 / 6.0, 64.0, 128.0, None),
            uniform_ore("andesite", 33, 2.0, 0.0, 60.0, None),
            uniform_ore("tuff", 33, 4.0, -64.0, 0.0, None),
            uniform_ore("coal_ore", 17, 30.0, 136.0, 320.0, None),
            uniform_ore("iron_ore", 4, 10.0, -64.0, 72.0, None),
            uniform_ore("gold_ore", 9, 0.5, -64.0, -48.0, None),
            uniform_ore("redstone_ore", 8, 4.0, -64.0, 15.0, None),
            uniform_ore("diamond_ore", 12, 1.0 / 9.0, -144.0, 16.0, None),
            uniform_ore("lapis_ore", 7, 4.0, -64.0, 64.0, None),
            uniform_ore("gold_ore", 9, 50.0, 32.0, 256.0, Some(BADLANDS_BIOMES)),
        ];
        // Diamond (small and buried) are triangles; the large one above is
        // uniform because its rarity makes the shape irrelevant.
        s.periodic_gaussian_ores = vec![
            triangle_ore("coal_ore", 17, 20.0, 0.0, 192.0, None),
            triangle_ore("iron_ore", 9, 90.0, 80.0, 384.0, None),
            triangle_ore("iron_ore", 9, 10.0, -24.0, 56.0, None),
            triangle_ore("copper_ore", 10, 16.0, -16.0, 112.0, None),
            triangle_ore("gold_ore", 9, 4.0, -96.0, -32.0, None),
            triangle_ore("redstone_ore", 8, 8.0, -96.0, -32.0, None),
            triangle_ore("diamond_ore", 4, 7.0, -144.0, 16.0, None),
            triangle_ore("diamond_ore", 8, 4.0, -144.0, 16.0, None),
            triangle_ore("lapis_ore", 7, 2.0, -96.0, -32.0, None),
            triangle_ore("emerald_ore", 3, 100.0, -16.0, 480.0, Some(MOUNTAIN_BIOMES)),
        ];

        let depth = |filter: Option<BiomeFilter>| MainSurfaceReplacerConfig {
            biome_filter: filter,
            ..MainSurfaceReplacerConfig::default()
        };
        let main_exclude = BiomeFilter::AllOf {
            filters: vec![BiomeFilter::exclude(&[
                "windswept_savanna",
                "badlands",
                "eroded_badlands",
                "wooded_badlands",
                "old_growth_pine_taiga",
                "old_growth_spruce_taiga",
            ])],
        };
        let savanna_m = DepthBasedSurfaceReplacerConfig {
            biome_filter: Some(BiomeFilter::include(&["windswept_savanna"])),
            top_thresholds: vec![
                DepthEntry {
                    y: -0.5 / 3.0 + 3.0,
                    b: BlockStateDesc::new("coarse_dirt"),
                },
                DepthEntry {
                    y: 1.75 / 3.0 + 3.0,
                    b: BlockStateDesc::new("stone"),
                },
            ],
            filler_thresholds: vec![DepthEntry {
                y: 1.75 / 3.0 + 3.0,
                b: BlockStateDesc::new("stone"),
            }],
            ..DepthBasedSurfaceReplacerConfig::default()
        };
        let redwood = DepthBasedSurfaceReplacerConfig {
            biome_filter: Some(BiomeFilter::include(&[
                "old_growth_pine_taiga",
                "old_growth_spruce_taiga",
            ])),
            top_thresholds: vec![
                DepthEntry {
                    y: -0.95 / 3.0 + 3.0,
                    b: BlockStateDesc::new("podzol"),
                },
                DepthEntry {
                    y: 1.75 / 3.0 + 3.0,
                    b: BlockStateDesc::new("coarse_dirt"),
                },
            ],
            ..DepthBasedSurfaceReplacerConfig::default()
        };
        s.replacers = vec![
            ReplacerConfig::DensityRange(DensityRangeReplacerConfig {
                min_density: 0.0,
                block_in_range: Some(BlockStateDesc::new("stone")),
                ..DensityRangeReplacerConfig::default()
            }),
            // 26.3's deepslate gradient: all deepslate at y <= 0, none at y >= 8.
            ReplacerConfig::RandomYGradient(RandomYGradientReplacerConfig {
                block_to_place: Some(BlockStateDesc::new("deepslate")),
                probability_function: UserFunction::new(&[(0.0, 1.0), (8.0, 0.0)]),
                seed: 0,
                filter_blocks: vec![BlockStateDesc::new("stone")],
                block_filter_type: FilterType::Include,
                ..RandomYGradientReplacerConfig::default()
            }),
            ReplacerConfig::MainSurface(depth(Some(main_exclude))),
            ReplacerConfig::DepthBasedSurface(savanna_m),
            ReplacerConfig::DepthBasedSurface(redwood),
            ReplacerConfig::MesaSurface(MesaSurfaceReplacerConfig {
                biome_filter: Some(BiomeFilter::include(&[
                    "badlands",
                    "eroded_badlands",
                    "wooded_badlands",
                ])),
                ..MesaSurfaceReplacerConfig::default()
            }),
            ReplacerConfig::DensityRange(DensityRangeReplacerConfig {
                max_y: 63,
                filter_blocks: vec![BlockStateDesc::new("air")],
                block_filter_type: FilterType::Include,
                block_in_range: Some(BlockStateDesc::new("water")),
                ..DensityRangeReplacerConfig::default()
            }),
            ReplacerConfig::NoiseBasedSurfaceDecoration(NoiseBasedSurfaceDecorationConfig {
                min_y: 63,
                max_y: 64,
                biome_filter: Some(BiomeFilter::include(&["swamp", "mangrove_swamp"])),
                feature_block: Some(BlockStateDesc::new("lily_pad")),
                ground_block: Some(BlockStateDesc::new("water")),
                ground_min_noise: 0.0,
                feature_min_noise: 0.0,
                feature_max_noise: 0.12,
                noise_source: DecorationNoiseSource::GrassColorNoise,
                noise_freq_x: 0.25,
                noise_freq_z: 0.25,
                ..NoiseBasedSurfaceDecorationConfig::default()
            }),
            ReplacerConfig::RandomYGradient(RandomYGradientReplacerConfig {
                min_y: i32::MIN / 2,
                max_y: i32::MIN / 2 + 5,
                block_to_place: Some(BlockStateDesc::new("bedrock")),
                probability_function: UserFunction::new(&[
                    (f64::from(i32::MIN >> 1), 1.0),
                    (f64::from((i32::MIN >> 1) + 5), 0.0),
                ]),
                ..RandomYGradientReplacerConfig::default()
            }),
        ];
        s
    }

    /// The preset with the structure toggles off, as CWG's hybrid overworld
    /// preset has them.
    pub fn defaults_without_structures() -> Self {
        let mut s = Self::defaults();
        s.strongholds = false;
        s.villages = false;
        s.mineshafts = false;
        s.temples = false;
        s.ocean_monuments = false;
        s.woodland_mansions = false;
        s
    }

    /// The horizontal scale `biomeSize` stands for: the integrator multiplies
    /// quart coordinates by `1 / biome_scale()` before asking its biome
    /// source, so 4 (the default) is vanilla's size, 8 doubles every biome,
    /// 1 shrinks them to a quarter.
    pub fn biome_scale(&self) -> f64 {
        f64::from(self.biome_size.clamp(1, 8)) / 4.0
    }

    fn toggle(&self, t: CwgToggle) -> bool {
        match t {
            CwgToggle::Villages => self.villages,
            CwgToggle::Temples => self.temples,
            CwgToggle::Mineshafts => self.mineshafts,
            CwgToggle::OceanMonuments => self.ocean_monuments,
            CwgToggle::WoodlandMansions => self.woodland_mansions,
            CwgToggle::Strongholds => self.strongholds,
        }
    }

    fn toggle_mut(&mut self, t: CwgToggle) -> &mut bool {
        match t {
            CwgToggle::Villages => &mut self.villages,
            CwgToggle::Temples => &mut self.temples,
            CwgToggle::Mineshafts => &mut self.mineshafts,
            CwgToggle::OceanMonuments => &mut self.ocean_monuments,
            CwgToggle::WoodlandMansions => &mut self.woodland_mansions,
            CwgToggle::Strongholds => &mut self.strongholds,
        }
    }

    /// Whether structure set `set` (`minecraft:villages`, or `villages`)
    /// generates: its CWG toggle, when it has one, and its own switch in
    /// `structureSets`. A set this module does not know (a mod's) generates
    /// unless switched off.
    pub fn structure_enabled(&self, set: &str) -> bool {
        let set = structure_set_name(set);
        let toggle = vanilla_structure_set(&set)
            .and_then(|v| v.toggle)
            .is_none_or(|t| self.toggle(t));
        toggle
            && self
                .structure_sets
                .get(&set)
                .and_then(|t| t.enabled)
                .unwrap_or(true)
    }

    /// Switches structure set `set` on or off. Switching on a set whose CWG
    /// toggle is off turns the toggle on and switches the toggle's other sets
    /// off, so only `set` comes back.
    pub fn set_structure_enabled(&mut self, set: &str, on: bool) {
        let set = structure_set_name(set);
        if on
            && let Some(toggle) = vanilla_structure_set(&set).and_then(|v| v.toggle)
            && !self.toggle(toggle)
        {
            *self.toggle_mut(toggle) = true;
            for other in OVERWORLD_STRUCTURE_SETS {
                if other.toggle == Some(toggle) && other.name != set {
                    self.structure_sets
                        .entry(other.name.to_owned())
                        .or_default()
                        .enabled = Some(false);
                }
            }
        }
        let entry = self.structure_sets.entry(set.clone()).or_default();
        entry.enabled = if on { None } else { Some(false) };
        self.prune_structure_set(&set);
    }

    /// The placement structure set `set` generates with: vanilla's
    /// ([`OVERWORLD_STRUCTURE_SETS`]) with the fields of its `structureSets`
    /// entry over it. `None` for a set this module does not know.
    pub fn structure_placement(&self, set: &str) -> Option<VanillaPlacement> {
        let set = structure_set_name(set);
        let vanilla = vanilla_structure_set(&set)?.placement;
        let t = self.structure_sets.get(&set);
        Some(match vanilla {
            VanillaPlacement::Spread {
                spacing,
                separation,
                frequency,
            } => VanillaPlacement::Spread {
                spacing: t.and_then(|t| t.spacing).unwrap_or(spacing),
                separation: t.and_then(|t| t.separation).unwrap_or(separation),
                frequency: t.and_then(|t| t.frequency).unwrap_or(frequency),
            },
            VanillaPlacement::Rings {
                distance,
                spread,
                count,
            } => VanillaPlacement::Rings {
                distance: t.and_then(|t| t.distance).unwrap_or(distance),
                spread: t.and_then(|t| t.spread).unwrap_or(spread),
                count: t.and_then(|t| t.count).unwrap_or(count),
            },
        })
    }

    /// Sets structure set `set`'s placement: the fields that differ from
    /// vanilla's are kept in its `structureSets` entry, the others dropped.
    /// Nothing for a set this module does not know.
    pub fn set_structure_placement(&mut self, set: &str, placement: VanillaPlacement) {
        let set = structure_set_name(set);
        let Some(vanilla) = vanilla_structure_set(&set).map(|v| v.placement) else {
            return;
        };
        let differs = |a: i32, b: i32| (a != b).then_some(a);
        let entry = self.structure_sets.entry(set.clone()).or_default();
        match (placement, vanilla) {
            (
                VanillaPlacement::Spread {
                    spacing,
                    separation,
                    frequency,
                },
                VanillaPlacement::Spread {
                    spacing: s0,
                    separation: p0,
                    frequency: f0,
                },
            ) => {
                entry.spacing = differs(spacing, s0);
                entry.separation = differs(separation, p0);
                entry.frequency = (frequency != f0).then_some(frequency);
            }
            (
                VanillaPlacement::Rings {
                    distance,
                    spread,
                    count,
                },
                VanillaPlacement::Rings {
                    distance: d0,
                    spread: s0,
                    count: c0,
                },
            ) => {
                entry.distance = differs(distance, d0);
                entry.spread = differs(spread, s0);
                entry.count = differs(count, c0);
            }
            _ => {}
        }
        self.prune_structure_set(&set);
    }

    /// Drops `set`'s entry when it says nothing.
    fn prune_structure_set(&mut self, set: &str) {
        if self
            .structure_sets
            .get(set)
            .is_some_and(|t| *t == StructureSetTuning::default())
        {
            self.structure_sets.remove(set);
        }
    }

    /// Notes about the settings that are not errors (an old `version`).
    pub fn warnings(&self) -> Vec<String> {
        let mut w = Vec::new();
        if self.version < LATEST_VERSION {
            w.push(format!(
                "settings are version {}; CWG would upgrade them to {LATEST_VERSION} with its fixer chain, which is not implemented, so they are used as they are",
                self.version
            ));
        } else if self.version > LATEST_VERSION {
            w.push(format!(
                "settings are version {}, newer than the {LATEST_VERSION} this generator knows",
                self.version
            ));
        }
        for area in &self.cube_areas {
            w.extend(area.settings.warnings());
        }
        w
    }

    /// Checks the numbers a generator cannot work with.
    pub fn validate(&self) -> Result<(), String> {
        for (name, t) in &self.structure_sets {
            let check = |field: &str, v: Option<i32>, range: std::ops::RangeInclusive<i32>| {
                match v {
                    Some(n) if !range.contains(&n) => Err(format!(
                        "structureSets.{name}.{field} must be {} to {}",
                        range.start(),
                        range.end()
                    )),
                    _ => Ok(()),
                }
            };
            check("spacing", t.spacing, 1..=4096)?;
            check("separation", t.separation, 0..=4095)?;
            check("distance", t.distance, 0..=1023)?;
            check("spread", t.spread, 0..=1023)?;
            check("count", t.count, 1..=4095)?;
            if let Some(f) = t.frequency
                && !(0.0..=1.0).contains(&f)
            {
                return Err(format!("structureSets.{name}.frequency must be 0 to 1"));
            }
            // A known set's spacing (its own or the entry's) must exceed its
            // separation, as vanilla's codec requires; for another set, the
            // entry's own pair.
            let pair = match self.structure_placement(name) {
                Some(VanillaPlacement::Spread {
                    spacing,
                    separation,
                    ..
                }) => Some((spacing, separation)),
                Some(VanillaPlacement::Rings { .. }) => None,
                None => t.spacing.zip(t.separation),
            };
            if let Some((spacing, separation)) = pair
                && separation >= spacing
                && !(spacing == 1 && separation == 0)
            {
                return Err(format!(
                    "structureSets.{name}: separation must be less than spacing"
                ));
            }
        }
        let ok = |n: i32| matches!(n, 1 | 2 | 4 | 8 | 16);
        for (name, n) in [
            ("noiseSampleSizeX", self.noise_sample_size_x),
            ("noiseSampleSizeY", self.noise_sample_size_y),
            ("noiseSampleSizeZ", self.noise_sample_size_z),
        ] {
            if !ok(n) {
                return Err(format!("{name} is {n}; it must be 1, 2, 4, 8 or 16"));
            }
        }
        if self.noise_sample_size_x != self.noise_sample_size_z {
            return Err("noiseSampleSizeX and noiseSampleSizeZ must be equal".into());
        }
        for (name, n) in [
            ("depthNoiseOctaves", self.depth_noise_octaves),
            ("selectorNoiseOctaves", self.selector_noise_octaves),
            ("lowNoiseOctaves", self.low_noise_octaves),
            ("highNoiseOctaves", self.high_noise_octaves),
        ] {
            if !(1..=32).contains(&n) {
                return Err(format!("{name} is {n}; it must be 1 to 32"));
            }
        }
        for (name, v) in [
            ("expectedBaseHeight", self.expected_base_height),
            ("expectedHeightVariation", self.expected_height_variation),
            ("heightFactor", self.height_factor),
            ("heightOffset", self.height_offset),
            ("heightVariationFactor", self.height_variation_factor),
            ("heightVariationOffset", self.height_variation_offset),
            ("depthNoiseFactor", self.depth_noise_factor),
            ("selectorNoiseFactor", self.selector_noise_factor),
            ("lowNoiseFactor", self.low_noise_factor),
            ("highNoiseFactor", self.high_noise_factor),
            ("depthNoiseFrequencyX", self.depth_noise_frequency_x),
            ("depthNoiseFrequencyZ", self.depth_noise_frequency_z),
            ("selectorNoiseFrequencyX", self.selector_noise_frequency_x),
            ("selectorNoiseFrequencyY", self.selector_noise_frequency_y),
            ("selectorNoiseFrequencyZ", self.selector_noise_frequency_z),
            ("lowNoiseFrequencyX", self.low_noise_frequency_x),
            ("lowNoiseFrequencyY", self.low_noise_frequency_y),
            ("lowNoiseFrequencyZ", self.low_noise_frequency_z),
            ("highNoiseFrequencyX", self.high_noise_frequency_x),
            ("highNoiseFrequencyY", self.high_noise_frequency_y),
            ("highNoiseFrequencyZ", self.high_noise_frequency_z),
        ] {
            if !v.is_finite() {
                return Err(format!("{name} must be a finite number"));
            }
        }
        if self.expected_height_variation == 0.0 {
            return Err("expectedHeightVariation must not be 0".into());
        }
        if !(0..=1000).contains(&self.dungeon_count) {
            return Err("dungeonCount must be 0 to 1000".into());
        }
        for (i, c) in self.caves.iter().enumerate() {
            for (name, n) in [
                ("caveRarity", c.cave_rarity),
                ("largeNodeRarity", c.large_node_rarity),
                ("bigCaveRarity", c.big_cave_rarity),
                ("steepStepRarity", c.steep_step_rarity),
                ("carveStepRarity", c.carve_step_rarity),
            ] {
                if n < 1 {
                    return Err(format!("caves[{i}].{name} must be at least 1"));
                }
            }
            if c.max_init_nodes < 0 || c.large_node_max_branches < 1 {
                return Err(format!(
                    "caves[{i}]: maxInitNodes must be 0 or more and largeNodeMaxBranches at least 1"
                ));
            }
        }
        for (i, o) in self.standard_ores.iter().enumerate() {
            if o.spawn_size < 1 || o.spawn_size > 64 || o.spawn_tries < 0 {
                return Err(format!(
                    "standardOres[{i}]: spawnSize must be 1 to 64 and spawnTries 0 or more"
                ));
            }
        }
        for (i, o) in self.periodic_gaussian_ores.iter().enumerate() {
            if o.spawn_size < 1 || o.spawn_size > 64 || o.spawn_tries < 0 {
                return Err(format!(
                    "periodicGaussianOres[{i}]: spawnSize must be 1 to 64 and spawnTries 0 or more"
                ));
            }
            if o.height_std_deviation <= 0.0 || !o.height_std_deviation.is_finite() {
                return Err(format!(
                    "periodicGaussianOres[{i}]: heightStdDeviation must be positive"
                ));
            }
        }
        for (i, r) in self.replacers.iter().enumerate() {
            match r {
                ReplacerConfig::MainSurface(c) => {
                    check_noise(i, c.surface_depth_noise_octaves)?;
                }
                ReplacerConfig::DepthBasedSurface(c) => {
                    check_noise(i, c.surface_depth_noise_octaves)?;
                }
                ReplacerConfig::MesaSurface(c) => {
                    check_noise(i, c.surface_depth_noise_octaves)?;
                }
                ReplacerConfig::NoiseBasedSurfaceDecoration(c) => {
                    check_noise(i, c.custom_noise_octaves)?;
                }
                _ => {}
            }
        }
        for (i, a) in self.cube_areas.iter().enumerate() {
            a.settings
                .validate()
                .map_err(|e| format!("cubeAreas[{i}]: {e}"))?;
        }
        Ok(())
    }

    // --- JSON --------------------------------------------------------------

    /// Parses CWG's lenient JSON (comments, trailing commas, unquoted keys,
    /// bare `Infinity` / `-Infinity` / `NaN`). An empty string is the preset.
    /// Unknown fields are ignored. The result is not validated.
    pub fn from_json(text: &str) -> Result<Self, String> {
        Self::from_json_with_warnings(text).map(|(s, _)| s)
    }

    /// [`Self::from_json`] plus [`Self::warnings`].
    pub fn from_json_with_warnings(text: &str) -> Result<(Self, Vec<String>), String> {
        if text.trim().is_empty() {
            return Ok((Self::defaults(), Vec::new()));
        }
        let strict = lenient_to_json(text)?;
        let s: Self =
            serde_json::from_str(&strict).map_err(|e| format!("invalid settings: {e}"))?;
        let w = s.warnings();
        Ok((s, w))
    }

    pub fn to_json_pretty(&self) -> String {
        bare_tokens(&serde_json::to_string_pretty(self).unwrap_or_default())
    }

    pub fn to_json_compact(&self) -> String {
        bare_tokens(&serde_json::to_string(self).unwrap_or_default())
    }
}

fn check_noise(i: usize, octaves: i32) -> Result<(), String> {
    if (1..=32).contains(&octaves) {
        Ok(())
    } else {
        Err(format!("replacers[{i}]: noise octaves must be 1 to 32"))
    }
}

/// `"Infinity"` strings written by [`inf`] become CWG's bare tokens.
fn bare_tokens(json: &str) -> String {
    json.replace("\"-Infinity\"", "-Infinity")
        .replace("\"Infinity\"", "Infinity")
        .replace("\"NaN\"", "NaN")
}

const BIG: &str = "1.7976931348623157e308";

/// Turns CWG's JSON5-ish text into strict JSON: strips `//` and `/* */`
/// comments and trailing commas, quotes bare keys, and replaces bare
/// `Infinity` / `-Infinity` / `NaN`.
pub fn lenient_to_json(text: &str) -> Result<String, String> {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        match c {
            '"' | '\'' => {
                // A string, copied with its escapes (single quotes become double).
                let quote = c;
                out.push('"');
                i += 1;
                while i < chars.len() && chars[i] != quote {
                    if chars[i] == '\\' && i + 1 < chars.len() {
                        out.push('\\');
                        out.push(chars[i + 1]);
                        i += 2;
                    } else {
                        if chars[i] == '"' {
                            out.push('\\');
                        }
                        out.push(chars[i]);
                        i += 1;
                    }
                }
                if i >= chars.len() {
                    return Err("invalid settings: unterminated string".into());
                }
                out.push('"');
                i += 1;
            }
            '/' if chars.get(i + 1) == Some(&'/') => {
                while i < chars.len() && chars[i] != '\n' {
                    i += 1;
                }
            }
            '/' if chars.get(i + 1) == Some(&'*') => {
                i += 2;
                while i + 1 < chars.len() && !(chars[i] == '*' && chars[i + 1] == '/') {
                    i += 1;
                }
                i += 2;
            }
            ',' => {
                // Drop a trailing comma: the next significant char closes.
                let mut j = i + 1;
                loop {
                    while j < chars.len() && chars[j].is_whitespace() {
                        j += 1;
                    }
                    if chars.get(j) == Some(&'/') && chars.get(j + 1) == Some(&'/') {
                        while j < chars.len() && chars[j] != '\n' {
                            j += 1;
                        }
                    } else if chars.get(j) == Some(&'/') && chars.get(j + 1) == Some(&'*') {
                        j += 2;
                        while j + 1 < chars.len() && !(chars[j] == '*' && chars[j + 1] == '/') {
                            j += 1;
                        }
                        j += 2;
                    } else {
                        break;
                    }
                }
                if !matches!(chars.get(j), Some(']') | Some('}')) {
                    out.push(',');
                }
                i += 1;
            }
            c if c.is_alphabetic() || c == '_' || c == '$' => {
                let start = i;
                while i < chars.len()
                    && (chars[i].is_alphanumeric() || chars[i] == '_' || chars[i] == '$')
                {
                    i += 1;
                }
                let word: String = chars[start..i].iter().collect();
                let mut j = i;
                while j < chars.len() && chars[j].is_whitespace() {
                    j += 1;
                }
                match word.as_str() {
                    "true" | "false" | "null" => out.push_str(&word),
                    "Infinity" => out.push_str(BIG),
                    "NaN" => out.push_str("null"),
                    _ if chars.get(j) == Some(&':') => {
                        out.push('"');
                        out.push_str(&word);
                        out.push('"');
                    }
                    _ => return Err(format!("invalid settings: unexpected word {word}")),
                }
            }
            '+' => {
                i += 1;
            }
            c => {
                out.push(c);
                i += 1;
            }
        }
    }
    // `-1.79e308` written as "-" + BIG stays a valid number.
    Ok(out)
}

// ---------------------------------------------------------------------------
// Flat Cubic
// ---------------------------------------------------------------------------

/// One layer of Flat Cubic: blocks from `fromY` up to (excluding) `toY`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct FlatLayer {
    pub from_y: i32,
    pub to_y: i32,
    /// Biome id of CWG (-1 = none); unused, Flat Cubic is single-biome Plains.
    pub biome: i32,
    pub block_state: BlockStateDesc,
}

impl Default for FlatLayer {
    fn default() -> Self {
        Self {
            from_y: 0,
            to_y: 1,
            biome: -1,
            block_state: BlockStateDesc::new("air"),
        }
    }
}

/// `FlatGeneratorSettings`: layers keyed by their start Y (the JSON keys are
/// decimal strings, as in CWG's `layers` object).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct FlatGeneratorSettings {
    pub layers: BTreeMap<i32, FlatLayer>,
    pub version: i32,
}

impl Default for FlatGeneratorSettings {
    /// CWG's: bedrock from the bottom to -8, stone to -1, dirt at -1, grass at 0.
    fn default() -> Self {
        let mut layers = BTreeMap::new();
        let mut put = |from: i32, to: i32, block: &str| {
            layers.insert(
                from,
                FlatLayer {
                    from_y: from,
                    to_y: to,
                    biome: -1,
                    block_state: BlockStateDesc::new(block),
                },
            );
        };
        put(i32::MIN + 1, -8, "bedrock");
        put(-8, -1, "stone");
        put(-1, 0, "dirt");
        put(0, 1, "grass_block");
        Self { layers, version: 1 }
    }
}

impl FlatGeneratorSettings {
    pub fn from_json(text: &str) -> Result<Self, String> {
        if text.trim().is_empty() {
            return Ok(Self::default());
        }
        let strict = lenient_to_json(text)?;
        serde_json::from_str(&strict).map_err(|e| format!("invalid flat settings: {e}"))
    }
    pub fn to_json_pretty(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }
    pub fn to_json_compact(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn structure_sets_are_vanillas_until_tuned() {
        let mut s = CustomGeneratorSettings::defaults();
        for v in OVERWORLD_STRUCTURE_SETS {
            assert!(s.structure_enabled(v.name), "{}", v.name);
            assert_eq!(s.structure_placement(v.name), Some(v.placement));
        }
        // Short names work, unknown sets generate untuned.
        assert!(s.structure_enabled("villages"));
        assert!(s.structure_enabled("somemod:castles"));
        assert_eq!(s.structure_placement("somemod:castles"), None);
        // Untouched settings write no `structureSets`.
        assert!(!s.to_json_compact().contains("structureSets"));

        s.set_structure_enabled("minecraft:igloos", false);
        s.set_structure_placement(
            "villages",
            VanillaPlacement::Spread {
                spacing: 20,
                separation: 8,
                frequency: 1.0,
            },
        );
        assert!(!s.structure_enabled("minecraft:igloos"));
        assert!(s.structure_enabled("minecraft:desert_pyramids"));
        let v = &s.structure_sets["minecraft:villages"];
        assert_eq!((v.spacing, v.separation, v.frequency), (Some(20), None, None));
        let text = s.to_json_compact();
        assert!(text.contains("structureSets"), "{text}");
        // (The whole preset does not round-trip exactly: serde_json reads
        // some of its lake probabilities one ulp off.)
        let back = CustomGeneratorSettings::from_json(&text).unwrap();
        assert_eq!(back.structure_sets, s.structure_sets);
        assert_eq!(back.temples, s.temples);
        back.validate().unwrap();

        // Back to vanilla's numbers and on: the entries go.
        s.set_structure_enabled("igloos", true);
        s.set_structure_placement(
            "villages",
            VanillaPlacement::Spread {
                spacing: 34,
                separation: 8,
                frequency: 1.0,
            },
        );
        assert!(s.structure_sets.is_empty(), "{:?}", s.structure_sets);
    }

    #[test]
    fn cwg_toggles_switch_their_groups() {
        let mut s = CustomGeneratorSettings::defaults_without_structures();
        for v in OVERWORLD_STRUCTURE_SETS {
            assert_eq!(s.structure_enabled(v.name), v.toggle.is_none(), "{}", v.name);
        }
        // One temple back: the toggle comes on, its other sets stay off.
        s.set_structure_enabled("minecraft:jungle_temples", true);
        assert!(s.temples);
        assert!(s.structure_enabled("minecraft:jungle_temples"));
        for other in ["desert_pyramids", "swamp_huts", "igloos"] {
            assert!(!s.structure_enabled(other), "{other}");
        }
        // CWG JSON switching villages off.
        let s = CustomGeneratorSettings::from_json(r#"{"villages": false}"#).unwrap();
        assert!(!s.structure_enabled("minecraft:villages"));
        assert!(s.structure_enabled("minecraft:pillager_outposts"));
    }

    #[test]
    fn structure_tuning_is_validated() {
        let bad = [
            r#"{"structureSets": {"minecraft:villages": {"spacing": 0}}}"#,
            r#"{"structureSets": {"minecraft:villages": {"separation": 34}}}"#,
            r#"{"structureSets": {"minecraft:villages": {"spacing": 8, "separation": 8}}}"#,
            r#"{"structureSets": {"minecraft:mineshafts": {"frequency": 1.5}}}"#,
            r#"{"structureSets": {"minecraft:strongholds": {"count": 0}}}"#,
            r#"{"structureSets": {"somemod:castles": {"spacing": 4, "separation": 4}}}"#,
        ];
        for text in bad {
            let s = CustomGeneratorSettings::from_json(text).unwrap();
            assert!(s.validate().is_err(), "{text}");
        }
        let good = [
            r#"{"structureSets": {"minecraft:villages": {"spacing": 9, "separation": 8}}}"#,
            r#"{"structureSets": {"minecraft:mineshafts": {"frequency": 0.05}}}"#,
            r#"{"structureSets": {"minecraft:strongholds": {"distance": 8, "spread": 1, "count": 3}}}"#,
            r#"{"structureSets": {"minecraft:buried_treasures": {"enabled": false}}}"#,
        ];
        for text in good {
            let s = CustomGeneratorSettings::from_json(text).unwrap();
            s.validate().unwrap_or_else(|e| panic!("{text}: {e}"));
        }
    }

    #[test]
    fn the_preset_survives_a_json_round_trip() {
        let a = CustomGeneratorSettings::defaults();
        for text in [a.to_json_pretty(), a.to_json_compact()] {
            let b = CustomGeneratorSettings::from_json(&text).unwrap();
            assert_eq!(a, b);
        }
        a.validate().unwrap();
    }

    #[test]
    fn cwg_style_json_with_comments_and_infinities_parses() {
        let text = r#"{
            // comment
            "version": 9, /* block */
            "biome": -1,
            "strongholds": false,
            standardOres: [
                { "blockstate": "coal_ore", "spawnSize": 17, "spawnTries": 20,
                  "spawnProbability": 0.125, "minHeight": -Infinity, "maxHeight": 1.0, },
                { "blockstate": {"Name": "mod:thing", "Properties": {"facing": "north"}},
                  "spawnSize": 3, "spawnTries": 1, "maxHeight": Infinity },
            ],
            "replacers": [
                {"type": "densityRange", "minDensity": 0, "maxDensity": Infinity,
                 "blockInRange": "stone", "blockFilterType": "EXCLUDE"}
            ],
            "unknownField": 5,
        }"#;
        let (s, warnings) = CustomGeneratorSettings::from_json_with_warnings(text).unwrap();
        assert!(warnings.is_empty());
        assert!(!s.strongholds);
        assert_eq!(s.standard_ores.len(), 2);
        assert_eq!(s.standard_ores[0].min_height, f64::NEG_INFINITY);
        assert_eq!(s.standard_ores[1].max_height, f64::INFINITY);
        assert_eq!(s.standard_ores[1].blockstate.name, "mod:thing");
        assert!(s.caves.is_empty(), "missing lists are empty, as in CWG");
        let ReplacerConfig::DensityRange(d) = &s.replacers[0] else {
            panic!("a density range");
        };
        assert_eq!(d.max_density, f64::INFINITY);
        assert_eq!(d.min_y, i32::MIN);
    }

    #[test]
    fn old_versions_warn_and_legacy_biome_ids_translate() {
        let (s, w) =
            CustomGeneratorSettings::from_json_with_warnings(r#"{"version": 5, "biome": 2}"#)
                .unwrap();
        assert_eq!(w.len(), 1);
        assert_eq!(s.biome, "minecraft:desert");
        let s = CustomGeneratorSettings::from_json(r#"{"biome": "plains"}"#).unwrap();
        assert_eq!(s.biome, "minecraft:plains");
    }

    #[test]
    fn validation_rejects_bad_sample_sizes() {
        let mut s = CustomGeneratorSettings::defaults();
        s.noise_sample_size_y = 3;
        assert!(s.validate().is_err());
        let mut s = CustomGeneratorSettings::defaults();
        s.noise_sample_size_x = 8;
        assert!(s.validate().is_err());
    }

    #[test]
    fn user_functions_interpolate_and_extrapolate() {
        let f = UserFunction::new(&[(0.0, 0.0), (10.0, 1.0), (20.0, 3.0)]);
        assert_eq!(f.get(5.0), 0.5);
        assert_eq!(f.get(15.0), 2.0);
        assert_eq!(f.get(30.0), 5.0);
        assert_eq!(f.get(-10.0), -1.0);
    }

    #[test]
    fn flat_settings_round_trip() {
        let a = FlatGeneratorSettings::default();
        let b = FlatGeneratorSettings::from_json(&a.to_json_pretty()).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn cube_areas_nest_settings() {
        let mut s = CustomGeneratorSettings::default();
        let inner = CustomGeneratorSettings {
            biome: "minecraft:desert".into(),
            ..CustomGeneratorSettings::default()
        };
        s.cube_areas.push(CubeArea::from((
            IntAabb {
                min_x: -1,
                min_y: -2,
                min_z: -3,
                max_x: 1,
                max_y: 2,
                max_z: 3,
            },
            inner,
        )));
        let t = CustomGeneratorSettings::from_json(&s.to_json_compact()).unwrap();
        assert_eq!(s, t);
        assert!(t.cube_areas[0].area.contains(0, 0, 0));
    }

    #[test]
    fn conditions_and_filters_round_trip() {
        let c = GenerationCondition::AllOf(vec![
            GenerationCondition::Rand(0.5),
            GenerationCondition::PosRandSeed {
                chance: 0.25,
                seed: 7,
            },
            GenerationCondition::stone_only(),
        ]);
        let text = serde_json::to_string(&c).unwrap();
        assert_eq!(
            serde_json::from_str::<GenerationCondition>(&text).unwrap(),
            c
        );
        let f = BiomeFilter::AllOf {
            filters: vec![
                BiomeFilter::exclude(&["mesa"]),
                BiomeFilter::IncludeClass {
                    match_type: BiomeClassMatchType::RawExact,
                    class_names: vec!["net.minecraft.world.biome.BiomeSwamp".into()],
                },
            ],
        };
        let text = serde_json::to_string(&f).unwrap();
        assert_eq!(serde_json::from_str::<BiomeFilter>(&text).unwrap(), f);
        assert!(!f.test("minecraft:badlands"));
        assert!(f.test("minecraft:swamp"));
    }
}
