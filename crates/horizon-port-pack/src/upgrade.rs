//! A 26.2 data pack's world generation, rewritten in 26.3's formats.
//!
//! 26.3 reworked the worldgen registries: configured carvers and features
//! became `carver` and `feature` with their `config` unwrapped, noise
//! parameters gained a base amplitude, density functions name their inputs
//! `input`/`left`/`right`, the noise settings moved aquifers, ore veins and
//! the surface rule out of the router into an `aquifers` object and a
//! `material_rule`, and biomes carry their mob spawns as attributes. Each
//! conversion here is the one vanilla's own data went through between 26.2
//! and 26.3, written so that the 26.3 pack generates what the 26.2 pack did:
//! the hard-coded parts of 26.2 (the beardifier added to the final density,
//! the deep dark aquifer exclusion, the ore veinifier, the interpolation
//! cell from `size_horizontal`/`size_vertical`) are spelled out as data.
//! Whatever has no faithful 26.3 form is kept as it is and listed in the
//! report.
use serde_json::{Map, Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::Read,
    path::Path,
};

/// 26.3's data pack format (`version.json`'s `pack_version.data_major`).
pub const PACK_FORMAT: u64 = 121;

/// What the upgrade did and what it could not do faithfully.
#[derive(Default, Debug)]
pub struct Report {
    /// Files rewritten, per registry.
    pub upgraded: BTreeMap<String, usize>,
    /// Files written that the 26.2 pack did not have (named density
    /// functions the noise settings share).
    pub added: Vec<String>,
    /// Parts that changed meaning or were dropped, with how often.
    pub lossy: BTreeMap<String, usize>,
    /// Types the upgrade does not know, left as they are.
    pub unknown: BTreeMap<String, usize>,
}

impl Report {
    fn lossy(&mut self, what: impl Into<String>) {
        *self.lossy.entry(what.into()).or_default() += 1;
    }
    fn unknown(&mut self, what: impl Into<String>) {
        *self.unknown.entry(what.into()).or_default() += 1;
    }
}

/// Every file of a pack, by its path inside the pack (`data/...`,
/// `pack.mcmeta`), upgraded. Files outside world generation are kept as
/// they are.
pub fn upgrade_pack(files: &BTreeMap<String, Vec<u8>>) -> (BTreeMap<String, Vec<u8>>, Report) {
    let mut report = Report::default();
    let mut json: BTreeMap<String, Value> = BTreeMap::new();
    let mut out: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    for (path, bytes) in files {
        let parsed = path
            .ends_with(".json")
            .then(|| serde_json::from_slice::<Value>(bytes).ok())
            .flatten();
        match parsed {
            Some(v) if path.starts_with("data/") => {
                json.insert(path.clone(), v);
            }
            _ => {
                out.insert(path.clone(), bytes.clone());
            }
        }
    }
    let mut ctx = Context::new(&json);
    let mut written: BTreeMap<String, Value> = BTreeMap::new();
    for (path, v) in &json {
        let Some(file) = PackFile::parse(path) else {
            written.insert(path.clone(), v.clone());
            continue;
        };
        let upgraded = match file.registry.as_str() {
            "worldgen/noise" => Some(noise(v, &mut report)),
            "worldgen/density_function" => Some(ctx.named_function(&file.id(), v, &mut report)),
            "worldgen/noise_settings" => Some(ctx.settings(&file.id(), v, &mut report)),
            "worldgen/configured_carver" => Some(carver(v, &mut report)),
            "worldgen/configured_feature" => Some(ctx.feature(v, &mut report)),
            "worldgen/placed_feature" => Some(ctx.placed(v, &mut report)),
            "worldgen/biome" => Some(biome(v, &mut report)),
            "worldgen/structure" => Some(structure(v)),
            "worldgen/world_preset" => Some(ctx.world_preset(v, &mut report)),
            "dimension" => Some(ctx.dimension(v, &mut report)),
            "dimension_type" => Some(dimension_type(v)),
            "worldgen/template_pool" | "worldgen/processor_list" => {
                let mut v = v.clone();
                states(&mut v);
                Some(v)
            }
            _ => None,
        };
        let path = match file.registry.as_str() {
            "worldgen/configured_carver" => file.with_registry("worldgen/carver"),
            "worldgen/configured_feature" => file.with_registry("worldgen/feature"),
            _ => path.clone(),
        };
        match upgraded {
            Some(u) => {
                *report.upgraded.entry(file.registry.clone()).or_default() += 1;
                written.insert(path, u);
            }
            None => {
                written.insert(path, v.clone());
            }
        }
    }
    for (id, v) in std::mem::take(&mut ctx.generated) {
        let path = registry_path("worldgen/density_function", &id);
        report.added.push(path.clone());
        written.insert(path, v);
    }
    for (id, v) in std::mem::take(&mut ctx.rules) {
        let path = registry_path("worldgen/material_rule", &id);
        report.added.push(path.clone());
        written.insert(path, v);
    }
    for (path, v) in written {
        out.insert(path, pretty(&v));
    }
    if let Some(meta) = out
        .get("pack.mcmeta")
        .and_then(|b| serde_json::from_slice(b).ok())
    {
        out.insert("pack.mcmeta".into(), pretty(&mcmeta(meta)));
    }
    (out, report)
}

fn pretty(v: &Value) -> Vec<u8> {
    let mut bytes = serde_json::to_vec_pretty(v).unwrap_or_default();
    bytes.push(b'\n');
    bytes
}

/// `pack.mcmeta` for 26.3: the pack format range is 26.3's alone.
fn mcmeta(mut meta: Value) -> Value {
    if let Some(pack) = meta.get_mut("pack").and_then(Value::as_object_mut) {
        pack.remove("supported_formats");
        pack.insert("pack_format".into(), json!(PACK_FORMAT));
        pack.insert("min_format".into(), json!(PACK_FORMAT));
        pack.insert("max_format".into(), json!(PACK_FORMAT));
        if let Some(Value::String(d)) = pack.get_mut("description") {
            d.push_str(" (upgraded to 26.3)");
        }
    }
    meta
}

/// `data/<ns>/<registry...>/<path>.json`, split by the registries the
/// upgrade knows (a registry name has one or two segments).
struct PackFile {
    namespace: String,
    registry: String,
    path: String,
}

impl PackFile {
    fn parse(path: &str) -> Option<Self> {
        let rest = path.strip_prefix("data/")?.strip_suffix(".json")?;
        let (namespace, rest) = rest.split_once('/')?;
        let (registry, path) = if let Some(r) = rest.strip_prefix("worldgen/") {
            let (kind, path) = r.split_once('/')?;
            (format!("worldgen/{kind}"), path)
        } else {
            rest.split_once('/').map(|(k, p)| (k.to_owned(), p))?
        };
        Some(Self {
            namespace: namespace.into(),
            registry,
            path: path.into(),
        })
    }
    fn id(&self) -> String {
        format!("{}:{}", self.namespace, self.path)
    }
    fn with_registry(&self, registry: &str) -> String {
        format!("data/{}/{registry}/{}.json", self.namespace, self.path)
    }
}

fn registry_path(registry: &str, id: &str) -> String {
    let (ns, path) = id.split_once(':').unwrap_or(("minecraft", id));
    format!("data/{ns}/{registry}/{path}.json")
}

/// A type name without its `minecraft:` prefix, as the codecs read both.
fn kind(v: &Value) -> Option<&str> {
    let t = v.get("type")?.as_str()?;
    Some(t.strip_prefix("minecraft:").unwrap_or(t))
}

/// An id with its namespace (`minecraft:` when it has none).
fn full_id(id: &str) -> String {
    if id.contains(':') {
        id.to_owned()
    } else {
        format!("minecraft:{id}")
    }
}

/// Whether an object of type `t` appears anywhere in `v`.
fn contains_type(v: &Value, t: &str) -> bool {
    match v {
        Value::Object(o) => kind(v) == Some(t) || o.values().any(|x| contains_type(x, t)),
        Value::Array(a) => a.iter().any(|x| contains_type(x, t)),
        _ => false,
    }
}

fn vanilla(name: &str) -> Value {
    Value::String(format!("minecraft:{name}"))
}

// ---------------------------------------------------------------------------
// Block states

/// 26.2 writes a block or fluid state as `{"Name": ..., "Properties":
/// {...}}`; 26.3 reads `{"id": ..., "properties": {...}}` (or the id alone
/// for the default state).
fn states(v: &mut Value) {
    match v {
        Value::Object(o) => {
            let state = o.get("Name").is_some_and(Value::is_string)
                && o.keys().all(|k| k == "Name" || k == "Properties");
            if state {
                let name = o.remove("Name").unwrap_or_default();
                match o.remove("Properties") {
                    Some(Value::Object(p)) if !p.is_empty() => {
                        o.insert("id".into(), name);
                        o.insert("properties".into(), Value::Object(p));
                    }
                    _ => *v = name,
                }
                return;
            }
            o.values_mut().for_each(states);
        }
        Value::Array(a) => a.iter_mut().for_each(states),
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Noise parameters

/// `{firstOctave, amplitudes}` to 26.3's `NormalNoise` parameters, as
/// vanilla's `NormalNoise.createParity` builds them: the same octaves and
/// the base amplitude that keeps 26.2's normalisation.
fn noise(v: &Value, report: &mut Report) -> Value {
    let (Some(first), Some(amps)) = (
        v.get("firstOctave").and_then(Value::as_i64),
        v.get("amplitudes").and_then(Value::as_array),
    ) else {
        return v.clone();
    };
    let amps: Vec<f64> = amps.iter().filter_map(Value::as_f64).collect();
    if amps.is_empty() || amps.len() > 32 || amps.iter().any(|a| *a < 0.0) {
        report.lossy("noise: more than 32 octaves or negative amplitudes (left as they are)");
        return v.clone();
    }
    let mut o = Map::new();
    let base = parity_base_amplitude(first as i32, &amps);
    if base != 1.0 {
        o.insert("base_amplitude".into(), json!(base));
    }
    o.insert("base_octave".into(), json!(first));
    if amps.len() != 1 {
        o.insert("octave_count".into(), json!(amps.len()));
    }
    if amps.iter().any(|a| *a != 1.0) {
        o.insert("amplitude_modifiers".into(), json!(amps));
    }
    Value::Object(o)
}

/// `NormalNoise.computeParityBaseAmplitude`.
pub fn parity_base_amplitude(base_octave: i32, amps: &[f64]) -> f64 {
    let octaves = build_octaves(base_octave, 1.0, amps.len(), amps);
    let target: f64 = java_sum(octaves.iter().map(|a| a.abs()));
    let new = normalization_factor(target, &octaves);
    if new == 0.0 {
        return 1.0;
    }
    parity_normalization_factor(1.0, amps) / new
}

/// `NormalNoise.buildOctaves` with normalisation on: the octaves' signed
/// amplitudes (zero-modifier octaves left out).
fn build_octaves(_base_octave: i32, base_amplitude: f64, count: usize, mods: &[f64]) -> Vec<f64> {
    let n = count as i32;
    let mut amplitude = base_amplitude * (0.5f64.powi(-(n - 1)) / (0.5f64.powi(-n) - 1.0));
    let mut out = Vec::new();
    for i in 0..count {
        let m = mods.get(i).copied().unwrap_or(1.0);
        if m != 0.0 {
            out.push(amplitude * m);
        }
        amplitude *= 0.5;
    }
    out
}

fn normalization_factor(target: f64, octaves: &[f64]) -> f64 {
    let mut variance = 0.0;
    for a in octaves {
        let layer = 0.2702247831245211 * a.abs();
        variance += layer * layer;
    }
    let deviation = variance.sqrt();
    if deviation == 0.0 {
        return 0.0;
    }
    target * 0.3333333333333333 / (deviation * 2f64.sqrt())
}

fn parity_normalization_factor(base_amplitude: f64, mods: &[f64]) -> f64 {
    let (mut min, mut max) = (i32::MAX, i32::MIN);
    for (i, m) in mods.iter().enumerate() {
        if *m != 0.0 {
            min = min.min(i as i32);
            max = max.max(i as i32);
        }
    }
    let span = max.wrapping_sub(min);
    base_amplitude * 0.5 * 0.3333333333333333 / (0.1 * (1.0 + 1.0 / f64::from(span + 1)))
}

/// `DoubleStream.sum()`: Kahan summation as `Collectors.sumWithCompensation`
/// and `computeFinalSum` do it.
fn java_sum(values: impl Iterator<Item = f64>) -> f64 {
    let (mut sum, mut c, mut simple) = (0.0f64, 0.0f64, 0.0f64);
    for v in values {
        simple += v;
        let tmp = v - c;
        let velvel = sum + tmp;
        c = (velvel - sum) - tmp;
        sum = velvel;
    }
    let tmp = sum - c;
    if tmp.is_nan() && simple.is_infinite() {
        simple
    } else {
        tmp
    }
}

/// A noise holder: an id, or inline parameters to upgrade.
fn noise_holder(v: &Value, report: &mut Report) -> Value {
    match v {
        Value::Object(_) => noise(v, report),
        other => other.clone(),
    }
}

// ---------------------------------------------------------------------------
// Density functions, noise settings, dimensions

/// Vanilla 26.2 density functions whose value does not depend on y, which
/// a pack can name without defining them.
const VANILLA_FLAT: [&str; 13] = [
    "minecraft:zero",
    "minecraft:shift_x",
    "minecraft:shift_z",
    "minecraft:overworld/continents",
    "minecraft:overworld/erosion",
    "minecraft:overworld/ridges",
    "minecraft:overworld/ridges_folded",
    "minecraft:overworld/offset",
    "minecraft:overworld/factor",
    "minecraft:overworld/jaggedness",
    "minecraft:overworld_large_biomes/continents",
    "minecraft:overworld_large_biomes/erosion",
    "minecraft:end/sloped_cheese",
];

struct Context<'a> {
    /// The pack's own density functions by id, in 26.2's form.
    functions: BTreeMap<String, &'a Value>,
    /// The interpolation cell (`size_horizontal * 4`, `size_vertical * 4`)
    /// of the noise settings being upgraded, which 26.3 writes on each
    /// `interpolated`.
    cell: (i64, i64),
    /// The cells of the pack's noise settings that reach each of its named
    /// density functions through their routers.
    cells: BTreeMap<String, BTreeSet<(i64, i64)>>,
    /// The cell of every noise settings in the pack.
    all_cells: BTreeSet<(i64, i64)>,
    /// Named density functions the upgrade adds, by id.
    generated: BTreeMap<String, Value>,
    /// Material rules the upgrade adds (one per noise settings), by id.
    rules: BTreeMap<String, Value>,
}

impl<'a> Context<'a> {
    fn new(files: &'a BTreeMap<String, Value>) -> Self {
        let mut functions = BTreeMap::new();
        let mut settings = Vec::new();
        for (path, v) in files {
            let Some(f) = PackFile::parse(path) else {
                continue;
            };
            match f.registry.as_str() {
                "worldgen/density_function" => {
                    functions.insert(f.id(), v);
                }
                "worldgen/noise_settings" => settings.push(v),
                _ => {}
            }
        }
        let mut ctx = Self {
            functions,
            cell: (4, 8),
            cells: BTreeMap::new(),
            all_cells: BTreeSet::new(),
            generated: BTreeMap::new(),
            rules: BTreeMap::new(),
        };
        for s in settings {
            let cell = cell_of(s.get("noise").unwrap_or(&Value::Null));
            ctx.all_cells.insert(cell);
            let mut reached = BTreeSet::new();
            ctx.reach(s.get("noise_router").unwrap_or(&Value::Null), &mut reached);
            for id in reached {
                ctx.cells.entry(id).or_default().insert(cell);
            }
        }
        ctx
    }

    /// Every pack density function `v` names, directly or through others.
    fn reach(&self, v: &Value, out: &mut BTreeSet<String>) {
        match v {
            Value::String(id) => {
                let id = full_id(id);
                if let Some(f) = self.functions.get(&id)
                    && out.insert(id)
                {
                    self.reach(f, out);
                }
            }
            Value::Object(o) => o.values().for_each(|x| self.reach(x, out)),
            Value::Array(a) => a.iter().for_each(|x| self.reach(x, out)),
            _ => {}
        }
    }

    /// A named density function file in 26.3's form, with the cell of the
    /// noise settings that use it: vanilla's overworld cell when none of the
    /// pack's do, the pack's own when it has one.
    fn named_function(&mut self, id: &str, v: &Value, report: &mut Report) -> Value {
        let cells = self.cells.get(id).cloned().unwrap_or_default();
        self.cell = match (cells.len(), self.all_cells.len()) {
            (1, _) => *cells.iter().next().unwrap_or(&(4, 8)),
            (0, 1) => *self.all_cells.iter().next().unwrap_or(&(4, 8)),
            (0, _) => (4, 8),
            _ => {
                if contains_type(v, "interpolated") {
                    report.lossy("density: an interpolated function shared by noise settings with different cells");
                }
                *cells.iter().next().unwrap_or(&(4, 8))
            }
        };
        self.density(v, report)
    }

    /// One density function (a number, an id, or an object) in 26.3's form.
    fn density(&mut self, v: &Value, report: &mut Report) -> Value {
        let Value::Object(o) = v else {
            return v.clone();
        };
        let Some(t) = kind(v) else {
            return v.clone();
        };
        let arg = |name: &str| o.get(name).cloned().unwrap_or(Value::Null);
        let unary = |this: &mut Self, report: &mut Report, to: &str| {
            let input = this.density(&arg("argument"), report);
            json!({"type": format!("minecraft:{to}"), "input": input})
        };
        match t {
            "abs" | "square" | "cube" | "half_negative" | "quarter_negative" | "squeeze"
            | "blend_density" => unary(self, report, t),
            "invert" => unary(self, report, "reciprocal"),
            "add" | "mul" | "min" | "max" => {
                let left = self.density(&arg("argument1"), report);
                let right = self.density(&arg("argument2"), report);
                json!({"type": format!("minecraft:{t}"), "left": left, "right": right})
            }
            "interpolated" => {
                let input = self.density(&arg("argument"), report);
                json!({"type": "minecraft:interpolated", "cell_size_xz": self.cell.0,
                    "cell_size_y": self.cell.1, "input": input})
            }
            // 26.2's flat cache holds the value at y = 0 of each column
            // inside the chunk; 26.3's cache only memoises.
            "flat_cache" => {
                let raw = arg("argument");
                let input = self.density(&raw, report);
                let input = if self.depends_on_y(&raw, &mut BTreeSet::new()) {
                    json!({"type": "minecraft:slice", "axis": "y", "coordinate": 0, "input": input})
                } else {
                    input
                };
                json!({"type": "minecraft:cache", "input": input})
            }
            "cache_2d" | "cache_once" | "cache_all_in_cell" => {
                let raw = arg("argument");
                if t == "cache_2d" && self.depends_on_y(&raw, &mut BTreeSet::new()) {
                    report.lossy("density: cache_2d around a function of y");
                }
                let input = self.density(&raw, report);
                json!({"type": "minecraft:cache", "input": input})
            }
            "y_clamped_gradient" => {
                let (from, to) = (arg("from_y"), arg("to_y"));
                if from == to {
                    // A step at that y: clampedMap's NaN at the step itself
                    // has no 26.3 form.
                    report.lossy("density: y_clamped_gradient with from_y == to_y");
                    let y = from.as_f64().unwrap_or_default();
                    return json!({"type": "minecraft:range_choice", "input": "minecraft:y",
                        "min_inclusive": -1_000_000.0, "max_exclusive": y,
                        "when_in_range": arg("from_value"), "when_out_of_range": arg("to_value")});
                }
                json!({"type": "minecraft:gradient", "axis": "y", "from_coordinate": from,
                    "to_coordinate": to, "from_value": arg("from_value"),
                    "to_value": arg("to_value")})
            }
            "noise" => {
                let mut n = o.clone();
                n.insert("noise".into(), noise_holder(&arg("noise"), report));
                Value::Object(n)
            }
            "shifted_noise" => {
                let mut n = Map::new();
                n.insert("type".into(), vanilla("noise"));
                n.insert("noise".into(), noise_holder(&arg("noise"), report));
                n.insert("xz_scale".into(), arg("xz_scale"));
                n.insert("y_scale".into(), arg("y_scale"));
                n.insert("shift_x".into(), self.density(&arg("shift_x"), report));
                let shift_y = arg("shift_y");
                if !shift_y.as_f64().is_some_and(|y| y == 0.0) {
                    n.insert("shift_y".into(), self.density(&shift_y, report));
                }
                n.insert("shift_z".into(), self.density(&arg("shift_z"), report));
                Value::Object(n)
            }
            "shift_a" | "shift_b" | "shift" => {
                json!({"type": format!("minecraft:{t}"), "noise": noise_holder(&arg("argument"), report)})
            }
            "end_islands" => json!({"type": "minecraft:end_outer_islands"}),
            "constant" => arg("argument"),
            "clamp" | "range_choice" | "find_top_surface" | "interval_select" => {
                let mut n = o.clone();
                for key in [
                    "input",
                    "when_in_range",
                    "when_out_of_range",
                    "density",
                    "upper_bound",
                ] {
                    if let Some(x) = o.get(key) {
                        n.insert(key.into(), self.density(x, report));
                    }
                }
                if let Some(Value::Array(fs)) = o.get("functions") {
                    let fs = fs.iter().map(|f| self.density(f, report)).collect();
                    n.insert("functions".into(), Value::Array(fs));
                }
                Value::Object(n)
            }
            "spline" => {
                let spline = self.spline(&arg("spline"), report);
                json!({"type": "minecraft:spline", "spline": spline})
            }
            "old_blended_noise" | "blend_alpha" | "blend_offset" | "beardifier" => v.clone(),
            other => {
                report.unknown(format!("density function {other}"));
                v.clone()
            }
        }
    }

    /// A cubic spline: a number, or a coordinate and points whose values are
    /// splines themselves.
    fn spline(&mut self, v: &Value, report: &mut Report) -> Value {
        let Value::Object(o) = v else {
            return v.clone();
        };
        let mut n = o.clone();
        if let Some(c) = o.get("coordinate") {
            n.insert("coordinate".into(), self.density(c, report));
        }
        if let Some(Value::Array(points)) = o.get("points") {
            let points = points
                .iter()
                .map(|p| {
                    let mut p = p.clone();
                    if let Some(value) = p.get("value").cloned() {
                        p["value"] = self.spline(&value, report);
                    }
                    p
                })
                .collect();
            n.insert("points".into(), Value::Array(points));
        }
        Value::Object(n)
    }

    /// Whether a 26.2 density function's value can change with y, through
    /// the pack's own functions; unknown ids count as depending on y.
    fn depends_on_y(&self, v: &Value, visiting: &mut BTreeSet<String>) -> bool {
        match v {
            Value::Number(_) => false,
            Value::String(id) => {
                let id = if id.contains(':') {
                    id.clone()
                } else {
                    format!("minecraft:{id}")
                };
                if id == "minecraft:y" {
                    return true;
                }
                if let Some(f) = self.functions.get(&id) {
                    return visiting.insert(id.clone()) && {
                        let r = self.depends_on_y(f, visiting);
                        visiting.remove(&id);
                        r
                    };
                }
                !VANILLA_FLAT.contains(&id.as_str())
            }
            Value::Object(o) => match kind(v) {
                Some("y_clamped_gradient" | "old_blended_noise" | "beardifier" | "shift") => true,
                Some("flat_cache" | "end_islands" | "find_top_surface" | "constant") => false,
                Some("blend_alpha" | "blend_offset" | "shift_a" | "shift_b") => false,
                Some("noise") => o.get("y_scale").and_then(Value::as_f64) != Some(0.0),
                Some("shifted_noise") => {
                    o.get("y_scale").and_then(Value::as_f64) != Some(0.0)
                        || ["shift_x", "shift_y", "shift_z"]
                            .iter()
                            .any(|k| o.get(*k).is_some_and(|x| self.depends_on_y(x, visiting)))
                }
                Some("spline") => self.spline_depends_on_y(&o["spline"], visiting),
                _ => o
                    .iter()
                    .filter(|(k, _)| *k != "type")
                    .any(|(_, x)| match x {
                        Value::Array(a) => a.iter().any(|x| self.depends_on_y(x, visiting)),
                        x => self.depends_on_y(x, visiting),
                    }),
            },
            _ => false,
        }
    }

    fn spline_depends_on_y(&self, v: &Value, visiting: &mut BTreeSet<String>) -> bool {
        let Value::Object(o) = v else {
            return false;
        };
        o.get("coordinate")
            .is_some_and(|c| self.depends_on_y(c, visiting))
            || o.get("points").and_then(Value::as_array).is_some_and(|ps| {
                ps.iter().any(|p| {
                    p.get("value")
                        .is_some_and(|x| self.spline_depends_on_y(x, visiting))
                })
            })
    }

    /// A density function the settings use in several places, as an id: its
    /// own id when it is one, otherwise a new named function under
    /// `<settings>/upgraded/<name>`.
    fn named(&mut self, settings: &str, name: &str, v: Value) -> Value {
        if v.is_string() {
            return v;
        }
        let (ns, path) = settings.split_once(':').unwrap_or(("minecraft", settings));
        let id = format!("{ns}:{path}/upgraded/{name}");
        self.generated.insert(id.clone(), v);
        Value::String(id)
    }

    /// 26.2 noise settings in 26.3's form.
    fn settings(&mut self, id: &str, v: &Value, report: &mut Report) -> Value {
        let Value::Object(o) = v else {
            return v.clone();
        };
        if o.contains_key("material_rule") {
            return v.clone();
        }
        let router = o.get("noise_router").cloned().unwrap_or_default();
        self.cell = cell_of(o.get("noise").unwrap_or(&Value::Null));
        let conv = |this: &mut Self, report: &mut Report, name: &str| {
            this.density(&router.get(name).cloned().unwrap_or(json!(0.0)), report)
        };
        // The climate functions are named: the spawn target and the aquifer
        // exclusion refer to them by id.
        let climate = |this: &mut Self, report: &mut Report, name: &str| {
            let f = conv(this, report, name);
            this.named(id, name, f)
        };
        let temperature = climate(self, report, "temperature");
        let vegetation = climate(self, report, "vegetation");
        let continents = climate(self, report, "continents");
        let erosion = climate(self, report, "erosion");
        let depth = climate(self, report, "depth");
        let ridges = climate(self, report, "ridges");
        let final_density = conv(self, report, "final_density");
        let surface = conv(self, report, "preliminary_surface_level");
        let surface = self.named(id, "preliminary_surface_level", surface);
        let mut n = Map::new();
        let mut noise = o.get("noise").cloned().unwrap_or_default();
        if let Some(noise) = noise.as_object_mut() {
            noise.remove("size_horizontal");
            noise.remove("size_vertical");
        }
        n.insert("noise".into(), noise);
        for key in ["default_block", "default_fluid"] {
            let mut s = o.get(key).cloned().unwrap_or_default();
            states(&mut s);
            n.insert(key.into(), s);
        }
        let aquifers = o
            .get("aquifers_enabled")
            .and_then(Value::as_bool)
            .unwrap_or(true);
        if aquifers {
            let barrier = conv(self, report, "barrier");
            let floodedness = conv(self, report, "fluid_level_floodedness");
            let spread = conv(self, report, "fluid_level_spread");
            let lava = conv(self, report, "lava");
            // 26.2's OverworldBiomeBuilder.isDeepDarkRegion: no aquifer
            // where erosion < -0.225 and depth > 0.9 (positive excludes).
            let exclusion = json!({"type": "minecraft:min",
                "left": {"type": "minecraft:sub", "left": -0.225, "right": erosion.clone()},
                "right": {"type": "minecraft:max",
                    "left": {"type": "minecraft:sub", "left": depth.clone(), "right": 0.9},
                    "right": 0.0}});
            n.insert(
                "aquifers".into(),
                json!({"barrier": barrier, "exclusion": exclusion,
                    "fluid_level_floodedness": floodedness, "fluid_level_spread": spread,
                    "lava": lava, "surface_level": surface.clone()}),
            );
        }
        let mut rule = o.get("surface_rule").cloned().unwrap_or(json!({
            "type": "minecraft:sequence", "sequence": []}));
        states(&mut rule);
        let veins = o
            .get("ore_veins_enabled")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let rule = if veins {
            let toggle = conv(self, report, "vein_toggle");
            let toggle = self.named(
                id,
                "vein_toggle",
                json!({"type": "minecraft:cache", "input": toggle}),
            );
            let ridged = conv(self, report, "vein_ridged");
            let ridged = self.named(id, "vein_ridged", ridged);
            let gap = conv(self, report, "vein_gap");
            let gap = self.named(id, "vein_gap", gap);
            // Ore veins come first: 26.2 placed them while filling the
            // noise, and its surface pass only replaced the default block.
            let mut sequence = ore_veins(&toggle, &ridged, &gap);
            sequence.push(rule);
            json!({"type": "minecraft:sequence", "sequence": sequence})
        } else {
            rule
        };
        n.insert(
            "noise_router".into(),
            json!({"temperature": temperature, "vegetation": vegetation,
                "continents": continents, "erosion": erosion,
                "depth": depth, "ridges": ridges,
                "chunk_surface_level": {"type": "minecraft:interpolated", "cell_size_xz": 16,
                    "cell_size_y": 1, "input": surface},
                "final_density": {"type": "minecraft:add", "left": final_density,
                    "right": {"type": "minecraft:beardifier"}}}),
        );
        // Named, as vanilla names its own (and as Horizon reads them), under
        // the settings' id so other vanilla settings keep their rule.
        let (ns, path) = id.split_once(':').unwrap_or(("minecraft", id));
        let rule_id = format!("{ns}:{path}/upgraded/material_rule");
        self.rules.insert(rule_id.clone(), rule);
        n.insert("material_rule".into(), Value::String(rule_id));
        let climate = [
            ("temperature", n["noise_router"]["temperature"].clone()),
            ("humidity", n["noise_router"]["vegetation"].clone()),
            ("continentalness", n["noise_router"]["continents"].clone()),
            ("erosion", n["noise_router"]["erosion"].clone()),
            ("weirdness", n["noise_router"]["ridges"].clone()),
        ];
        let spawn: Vec<Value> = o
            .get("spawn_target")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .map(|p| {
                for key in ["depth", "offset"] {
                    if p.get(key).is_some_and(|d| d.as_f64() != Some(0.0)) {
                        report.lossy(format!("noise settings: spawn target {key} other than 0"));
                    }
                }
                let mut point = Map::new();
                for (param, f) in &climate {
                    if let (Some(range), Some(f)) = (p.get(*param), f.as_str()) {
                        point.insert(f.into(), range.clone());
                    }
                }
                Value::Object(point)
            })
            .collect();
        n.insert("spawn_target".into(), Value::Array(spawn));
        for key in [
            "sea_level",
            "disable_mob_generation",
            "legacy_random_source",
        ] {
            if let Some(x) = o.get(key) {
                n.insert(key.into(), x.clone());
            }
        }
        Value::Object(n)
    }

    /// A dimension (`dimension/<id>.json` or a world preset's entry): an
    /// inline noise settings object in its generator is upgraded.
    fn dimension(&mut self, v: &Value, report: &mut Report) -> Value {
        let mut v = v.clone();
        let noise = v.get("generator").and_then(kind) == Some("noise");
        if let Some(settings) = v.pointer("/generator/settings").cloned()
            && settings.is_object()
            && noise
        {
            v["generator"]["settings"] = self.settings("minecraft:inline", &settings, report);
        }
        v
    }

    fn world_preset(&mut self, v: &Value, report: &mut Report) -> Value {
        let mut v = v.clone();
        if let Some(dims) = v.get_mut("dimensions").and_then(Value::as_object_mut) {
            for d in dims.values_mut() {
                *d = self.dimension(d, report);
            }
        }
        v
    }

    // -----------------------------------------------------------------------
    // Features

    /// A configured feature in 26.3's form: `config` unwrapped, block state
    /// providers renamed, nested features and placements upgraded.
    fn feature(&mut self, v: &Value, report: &mut Report) -> Value {
        let Value::Object(o) = v else {
            return v.clone();
        };
        let Some(t) = kind(v) else {
            return v.clone();
        };
        if REPLACED_FEATURES.contains(&t) {
            report.unknown(format!("feature {t} (written as data in 26.3)"));
        }
        let mut n = Map::new();
        n.insert("type".into(), o["type"].clone());
        if let Some(Value::Object(config)) = o.get("config") {
            for (k, x) in config {
                n.insert(k.clone(), x.clone());
            }
        }
        let mut v = Value::Object(n);
        states(&mut v);
        self.nested(&mut v, report);
        v
    }

    /// Inside a feature: block state providers, placed features and
    /// configured features held inline.
    fn nested(&mut self, v: &mut Value, report: &mut Report) {
        match v {
            Value::Object(o) => {
                if let Some(t) = kind(&Value::Object(o.clone())) {
                    if let Some(to) = provider(t) {
                        if t == "simple_state_provider" {
                            *v = o.get("state").cloned().unwrap_or_default();
                            return;
                        }
                        o.insert("type".into(), vanilla(to));
                    }
                }
                // An inline placed feature: {feature, placement}.
                if o.contains_key("placement") && o.contains_key("feature") {
                    let placed = self.placed(&Value::Object(o.clone()), report);
                    *v = placed;
                    return;
                }
                for x in o.values_mut() {
                    if x.get("config").is_some() && x.get("type").is_some() {
                        *x = self.feature(x, report);
                    } else {
                        self.nested(x, report);
                    }
                }
            }
            Value::Array(a) => a.iter_mut().for_each(|x| self.nested(x, report)),
            _ => {}
        }
    }

    /// A placed feature: its feature (an id or inline) and its placement
    /// modifiers.
    fn placed(&mut self, v: &Value, report: &mut Report) -> Value {
        let mut v = v.clone();
        states(&mut v);
        if let Some(f) = v.get("feature").cloned()
            && f.is_object()
        {
            v["feature"] = self.feature(&f, report);
        }
        if let Some(Value::Array(ms)) = v.get_mut("placement") {
            for m in ms.iter_mut() {
                placement(m);
            }
        }
        v
    }
}

/// The interpolation cell of 26.2 `noise` settings.
fn cell_of(noise: &Value) -> (i64, i64) {
    let size = |k: &str, d: i64| noise.get(k).and_then(Value::as_i64).unwrap_or(d) * 4;
    (size("size_horizontal", 1), size("size_vertical", 2))
}

/// 26.2's `OreVeinifier` as 26.3 material rules, one per vein type, from
/// the pack's `vein_toggle`, `vein_ridged` and `vein_gap`: a block between
/// the type's heights where the toggle (copper above zero, iron below)
/// clears 0.4 less the edge roundoff, and the ridged noise is negative, is
/// the filler with chance 0.7, and ore where the richness draw and the gap
/// noise allow.
fn ore_veins(toggle: &Value, ridged: &Value, gap: &Value) -> Vec<Value> {
    let types = [
        ("copper_ore", "raw_copper_block", "granite", 0, 50, false),
        (
            "deepslate_iron_ore",
            "raw_iron_block",
            "tuff",
            -60,
            -8,
            true,
        ),
    ];
    types
        .iter()
        .map(|(ore, raw, filler, min_y, max_y, negate)| {
            let veininess = if *negate {
                json!({"type": "minecraft:negate", "input": toggle})
            } else {
                toggle.clone()
            };
            // clampedMap(min(maxY - y, y - minY), 0, 20, -0.2, 0)
            let edge = json!({"type": "minecraft:add",
                "left": {"type": "minecraft:mul",
                    "left": {"type": "minecraft:clamp",
                        "input": {"type": "minecraft:min",
                            "left": {"type": "minecraft:sub", "left": f64::from(*max_y), "right": "minecraft:y"},
                            "right": {"type": "minecraft:sub", "left": "minecraft:y", "right": f64::from(*min_y)}},
                        "min": 0.0, "max": 20.0},
                    "right": 0.01},
                "right": -0.2});
            let density = json!({"type": "minecraft:range_choice", "input": "minecraft:y",
                "min_inclusive": f64::from(*min_y), "max_exclusive": f64::from(max_y + 1),
                "when_in_range": {"type": "minecraft:range_choice", "input": ridged,
                    "min_inclusive": -1_000_000.0, "max_exclusive": 0.0,
                    "when_in_range": {"type": "minecraft:range_choice",
                        "input": {"type": "minecraft:add",
                            "left": {"type": "minecraft:sub", "left": veininess, "right": 0.4},
                            "right": edge},
                        "min_inclusive": 0.0, "max_exclusive": 1_000_000.0,
                        "when_in_range": 0.7, "when_out_of_range": -1.0},
                    "when_out_of_range": -1.0},
                "when_out_of_range": -1.0});
            // clampedMap(|toggle|, 0.4, 0.6, 0.1, 0.3)
            let richness = json!({"type": "minecraft:add",
                "left": {"type": "minecraft:clamp", "input": {"type": "minecraft:abs", "input": toggle},
                    "min": 0.4, "max": 0.6},
                "right": -0.3});
            json!({"type": "minecraft:ore_vein", "ore_block": vanilla(ore),
                "raw_ore_block": vanilla(raw), "filler_block": vanilla(filler),
                "raw_ore_chance": 0.02, "density": density, "richness": richness,
                "filler_gap": {"type": "minecraft:sub", "left": -0.3, "right": gap}})
        })
        .collect()
}

/// Feature types 26.3 dropped for compositions of other features; a pack
/// using them needs vanilla's composition, which the upgrade does not
/// write.
const REPLACED_FEATURES: [&str; 12] = [
    "basalt_columns",
    "basalt_pillar",
    "coral_mushroom",
    "desert_well",
    "glowstone_blob",
    "kelp",
    "nether_forest_vegetation",
    "sea_pickle",
    "seagrass",
    "twisting_vines",
    "weeping_vines",
    "random_patch",
];

/// 26.2 block state provider types and their 26.3 names.
fn provider(t: &str) -> Option<&'static str> {
    Some(match t {
        "simple_state_provider" => "simple_state_provider",
        "weighted_state_provider" => "weighted",
        "rotated_block_provider" => "rotated",
        "randomized_int_state_provider" => "randomized_int",
        "rule_based_state_provider" => "rule_based",
        "noise_provider" => "noise",
        "dual_noise_provider" => "dual_noise",
        "noise_threshold_provider" => "noise_threshold",
        _ => return None,
    })
}

/// One placement modifier: `random_offset` became `offset` with one
/// provider per axis (x and z both from `xz_spread`, drawn in that order).
fn placement(m: &mut Value) {
    if kind(m) == Some("random_offset") {
        let xz = m.get("xz_spread").cloned().unwrap_or(json!(0));
        let y = m.get("y_spread").cloned().unwrap_or(json!(0));
        *m = json!({"type": "minecraft:offset", "x": xz.clone(), "y": y, "z": xz});
    }
}

// ---------------------------------------------------------------------------
// Carvers, biomes, structures

/// A configured carver in 26.3's form. 26.2's cave carvers hard-coded their
/// count, thickness and the room's vertical scale per type; 26.3 reads them.
fn carver(v: &Value, report: &mut Report) -> Value {
    let Some(t) = kind(v) else {
        return v.clone();
    };
    let mut c = v.get("config").cloned().unwrap_or_default();
    let Some(o) = c.as_object_mut() else {
        return v.clone();
    };
    o.remove("debug_settings");
    let lava = o.remove("lava_level");
    let replaceable = o.remove("replaceable");
    if lava.is_some_and(|l| l != json!({"above_bottom": 8}) && l != json!({"above_bottom": 10})) {
        report.lossy("carver: lava_level (26.3 takes it from the aquifer)");
    }
    if replaceable.is_some_and(|r| {
        r != json!("#minecraft:overworld_carver_replaceables")
            && r != json!("#minecraft:nether_carver_replaceables")
    }) {
        report.lossy("carver: replaceable (26.3 carves any stone)");
    }
    let y_scale = o.remove("yScale");
    let mut n = Map::new();
    match t {
        "cave" | "nether_cave" => {
            let nether = t == "nether_cave";
            n.insert("type".into(), vanilla("cave"));
            n.insert(
                "count".into(),
                json!({"type": "minecraft:very_biased_to_bottom", "min_inclusive": 0,
                    "max_inclusive": if nether { 9 } else { 14 }}),
            );
            n.insert(
                "thickness".into(),
                if nether {
                    json!({"type": "minecraft:trapezoid", "min": 0.0, "max": 6.0, "plateau": 2.0})
                } else {
                    json!({"type": "minecraft:trapezoid", "min": 0.0, "max": 3.0, "plateau": 1.0})
                },
            );
            if !nether {
                n.insert("weird_thickness_bias".into(), json!(true));
            } else {
                n.insert("start_vertical_radius_multiplier".into(), json!(5.0));
            }
            if let Some(s) = y_scale {
                n.insert("room_vertical_radius_multiplier".into(), s);
            }
        }
        "canyon" => {
            n.insert("type".into(), vanilla("canyon"));
            if let (Some(s), Some(shape)) =
                (y_scale, o.get_mut("shape").and_then(Value::as_object_mut))
            {
                shape.insert("y_scale".into(), s);
            }
        }
        other => {
            report.unknown(format!("carver {other}"));
            n.insert("type".into(), v["type"].clone());
        }
    }
    for (k, x) in o.iter() {
        n.insert(k.clone(), x.clone());
    }
    let mut v = Value::Object(n);
    states(&mut v);
    v
}

/// `{type, minCount, maxCount, weight}` to `{type, count, weight}`.
fn spawner(s: &Value) -> Value {
    let mut s = s.clone();
    if let Some(o) = s.as_object_mut() {
        let min = o.remove("minCount");
        let max = o.remove("maxCount");
        if let (Some(min), Some(max)) = (min, max) {
            let count = if min == max {
                min
            } else {
                json!({"type": "minecraft:uniform", "min_inclusive": min, "max_inclusive": max})
            };
            o.insert("count".into(), count);
        }
    }
    s
}

/// A biome in 26.3's form: mob spawns become the
/// `minecraft:gameplay/natural_mob_spawns` attribute (every category kept,
/// empty ones too, so nothing comes from the dimension's defaults), and the
/// creature spawn probability its own attribute.
fn biome(v: &Value, report: &mut Report) -> Value {
    let mut v = v.clone();
    states(&mut v);
    let Some(o) = v.as_object_mut() else {
        return v;
    };
    let spawners = o.remove("spawners");
    let costs = o.remove("spawn_costs");
    let probability = o.remove("creature_spawn_probability");
    let attributes = o
        .entry("attributes")
        .or_insert_with(|| json!({}))
        .as_object_mut();
    let Some(attributes) = attributes else {
        report.lossy("biome: attributes is not an object");
        return v;
    };
    if let Some(Value::Object(categories)) = spawners {
        let by_category: Map<String, Value> = categories
            .iter()
            .map(|(c, list)| {
                let list = list.as_array().into_iter().flatten().map(spawner).collect();
                (c.clone(), Value::Array(list))
            })
            .collect();
        attributes.insert(
            "minecraft:gameplay/natural_mob_spawns".into(),
            json!({"modifier": "overlay", "argument": {
                "spawn_costs": costs.unwrap_or(json!({})),
                "spawns_by_category": by_category}}),
        );
    }
    if let Some(p) = probability
        && p.as_f64().is_some_and(|p| (p as f32) != 0.1)
    {
        attributes.insert(
            "minecraft:gameplay/creature_world_gen_spawn_probability".into(),
            p,
        );
    }
    v
}

/// A dimension type: the bed rule's `explodes` is 26.3's `destroy_on_use`.
fn dimension_type(v: &Value) -> Value {
    let mut v = v.clone();
    if let Some(rule) = v
        .pointer_mut("/attributes/minecraft:gameplay~1bed_rule")
        .and_then(Value::as_object_mut)
        && let Some(explodes) = rule.remove("explodes")
    {
        rule.insert("destroy_on_use".into(), explodes);
    }
    v
}

/// A structure: its spawn overrides' counts in 26.3's form.
fn structure(v: &Value) -> Value {
    let mut v = v.clone();
    states(&mut v);
    if let Some(Value::Object(overrides)) = v.get_mut("spawn_overrides") {
        for o in overrides.values_mut() {
            if let Some(Value::Array(spawns)) = o.get_mut("spawns") {
                for s in spawns.iter_mut() {
                    *s = spawner(s);
                }
            }
        }
    }
    v
}

// ---------------------------------------------------------------------------
// Files

/// A pack's files by their path inside it, from a zip or a folder. A zip
/// whose `pack.mcmeta` sits in one top folder is read from that folder.
pub fn read_pack(path: &Path) -> Result<BTreeMap<String, Vec<u8>>, String> {
    let at = |e: &dyn std::fmt::Display| format!("{}: {e}", path.display());
    let mut files = BTreeMap::new();
    if path.is_dir() {
        let mut stack = vec![path.to_path_buf()];
        while let Some(dir) = stack.pop() {
            for entry in fs::read_dir(&dir).map_err(|e| at(&e))? {
                let p = entry.map_err(|e| at(&e))?.path();
                if p.is_dir() {
                    stack.push(p);
                } else if let Ok(rel) = p.strip_prefix(path) {
                    let rel = rel.to_string_lossy().replace('\\', "/");
                    files.insert(rel, fs::read(&p).map_err(|e| at(&e))?);
                }
            }
        }
    } else {
        let file = fs::File::open(path).map_err(|e| at(&e))?;
        let mut zip = zip::ZipArchive::new(file).map_err(|e| at(&e))?;
        for i in 0..zip.len() {
            let mut entry = zip.by_index(i).map_err(|e| at(&e))?;
            if entry.is_dir() {
                continue;
            }
            let mut bytes = Vec::new();
            entry.read_to_end(&mut bytes).map_err(|e| at(&e))?;
            files.insert(entry.name().replace('\\', "/"), bytes);
        }
    }
    if !files.contains_key("pack.mcmeta") {
        let top = files
            .keys()
            .find_map(|k| k.strip_suffix("/pack.mcmeta").filter(|t| !t.contains('/')))
            .map(|t| format!("{t}/"));
        if let Some(top) = top {
            files = files
                .into_iter()
                .filter_map(|(k, v)| k.strip_prefix(&top).map(|k| (k.to_owned(), v)))
                .collect();
        }
    }
    if !files.contains_key("pack.mcmeta") {
        return Err(format!("{}: no pack.mcmeta", path.display()));
    }
    Ok(files)
}

/// Upgrades the pack at `input` into the folder `out`, which must not
/// exist yet (a world's `datapacks/<name>`).
pub fn upgrade_into(input: &Path, out: &Path) -> Result<Report, String> {
    if out.exists() {
        return Err(format!("{} already exists", out.display()));
    }
    let files = read_pack(input)?;
    let (upgraded, report) = upgrade_pack(&files);
    for (path, bytes) in upgraded {
        let target = out.join(&path);
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
        }
        fs::write(&target, bytes).map_err(|e| format!("{}: {e}", target.display()))?;
    }
    Ok(report)
}

impl Report {
    /// The report as lines for the terminal.
    pub fn lines(&self) -> Vec<String> {
        let mut lines: Vec<String> = self
            .upgraded
            .iter()
            .map(|(r, n)| format!("upgraded {n} {r}"))
            .collect();
        if !self.added.is_empty() {
            lines.push(format!(
                "added {} shared density functions",
                self.added.len()
            ));
        }
        lines.extend(
            self.lossy
                .iter()
                .map(|(w, n)| format!("not faithful ({n}x): {w}")),
        );
        lines.extend(
            self.unknown
                .iter()
                .map(|(w, n)| format!("left as it is ({n}x): {w}")),
        );
        lines
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn up(path: &str, v: Value) -> (Value, Report) {
        let files = BTreeMap::from([(path.to_owned(), serde_json::to_vec(&v).unwrap())]);
        let (out, report) = upgrade_pack(&files);
        let (_, bytes) = out
            .into_iter()
            .find(|(p, _)| !p.contains("/upgraded/"))
            .unwrap();
        (serde_json::from_slice(&bytes).unwrap(), report)
    }

    #[test]
    fn noise_keeps_the_octaves_and_finds_the_parity_amplitude() {
        let (v, _) = up(
            "data/x/worldgen/noise/n.json",
            json!({"firstOctave": -7, "amplitudes": [1.0, 2.0, 1.0, 0.0, 0.0, 0.0]}),
        );
        // Vanilla 26.3's minecraft:ridge, which 26.2 wrote as above.
        assert_eq!(
            v,
            json!({"base_amplitude": 0.9147152149950137, "base_octave": -7, "octave_count": 6,
                "amplitude_modifiers": [1.0, 2.0, 1.0, 0.0, 0.0, 0.0]})
        );
        let (v, _) = up(
            "data/x/worldgen/noise/n.json",
            json!({"firstOctave": -3, "amplitudes": [1.0]}),
        );
        // Vanilla 26.3's minecraft:aquifer_barrier.
        assert_eq!(
            v,
            json!({"base_amplitude": 0.955388882960065, "base_octave": -3})
        );
    }

    #[test]
    fn density_arguments_and_removed_types() {
        let (v, _) = up(
            "data/x/worldgen/density_function/f.json",
            json!({"type": "minecraft:add", "argument1": {"type": "minecraft:constant", "argument": 2.0},
                "argument2": {"type": "minecraft:abs", "argument": {"type": "minecraft:y_clamped_gradient",
                    "from_y": -64, "to_y": 320, "from_value": 1.5, "to_value": -1.5}}}),
        );
        assert_eq!(
            v,
            json!({"type": "minecraft:add", "left": 2.0, "right": {"type": "minecraft:abs",
                "input": {"type": "minecraft:gradient", "axis": "y", "from_coordinate": -64,
                    "to_coordinate": 320, "from_value": 1.5, "to_value": -1.5}}})
        );
        let (v, _) = up(
            "data/x/worldgen/density_function/f.json",
            json!({"type": "minecraft:shifted_noise", "noise": "minecraft:erosion",
                "shift_x": "minecraft:shift_x", "shift_y": 0.0, "shift_z": "minecraft:shift_z",
                "xz_scale": 0.25, "y_scale": 0.0}),
        );
        assert_eq!(
            v,
            json!({"type": "minecraft:noise", "noise": "minecraft:erosion", "shift_x": "minecraft:shift_x",
                "shift_z": "minecraft:shift_z", "xz_scale": 0.25, "y_scale": 0.0})
        );
    }

    #[test]
    fn flat_cache_slices_at_y_zero_only_what_depends_on_y() {
        let flat = |argument: Value| {
            up(
                "data/x/worldgen/density_function/f.json",
                json!({"type": "minecraft:flat_cache", "argument": argument}),
            )
            .0
        };
        let n = json!({"type": "minecraft:noise", "noise": "x:n", "xz_scale": 1.0, "y_scale": 0.0});
        assert_eq!(
            flat(n.clone()),
            json!({"type": "minecraft:cache", "input": n})
        );
        let y = json!({"type": "minecraft:mul", "argument1": "minecraft:y", "argument2": 2.0});
        assert_eq!(
            flat(y),
            json!({"type": "minecraft:cache", "input": {"type": "minecraft:slice", "axis": "y",
                "coordinate": 0, "input": {"type": "minecraft:mul", "left": "minecraft:y", "right": 2.0}}})
        );
    }

    #[test]
    fn interpolation_cell_comes_from_the_settings() {
        let settings = json!({"noise": {"min_y": 0, "height": 128, "size_horizontal": 2, "size_vertical": 1},
            "noise_router": {}, "surface_rule": {"type": "minecraft:sequence", "sequence": []},
            "default_block": {"Name": "minecraft:stone"}, "default_fluid": {"Name": "minecraft:water",
                "Properties": {"level": "0"}}, "aquifers_enabled": false, "ore_veins_enabled": false,
            "sea_level": 32, "disable_mob_generation": false, "legacy_random_source": false,
            "spawn_target": []});
        let files = BTreeMap::from([
            (
                "data/x/worldgen/noise_settings/s.json".to_owned(),
                serde_json::to_vec(&settings).unwrap(),
            ),
            (
                "data/x/worldgen/density_function/f.json".to_owned(),
                serde_json::to_vec(&json!({"type": "minecraft:interpolated", "argument": 1.0}))
                    .unwrap(),
            ),
        ]);
        let (out, _) = upgrade_pack(&files);
        let f: Value =
            serde_json::from_slice(&out["data/x/worldgen/density_function/f.json"]).unwrap();
        assert_eq!(
            f,
            json!({"type": "minecraft:interpolated", "cell_size_xz": 8, "cell_size_y": 4, "input": 1.0})
        );
        let s: Value =
            serde_json::from_slice(&out["data/x/worldgen/noise_settings/s.json"]).unwrap();
        assert_eq!(s["noise"], json!({"min_y": 0, "height": 128}));
        assert_eq!(
            s["default_fluid"],
            json!({"id": "minecraft:water", "properties": {"level": "0"}})
        );
        assert_eq!(s["default_block"], json!("minecraft:stone"));
        assert!(s.get("aquifers").is_none());
        assert_eq!(
            s["noise_router"]["final_density"]["right"],
            json!({"type": "minecraft:beardifier"})
        );
    }

    #[test]
    fn settings_move_aquifers_veins_and_spawn_target() {
        let settings = json!({"noise": {"min_y": -64, "height": 384, "size_horizontal": 1, "size_vertical": 2},
            "noise_router": {"barrier": 1.0, "fluid_level_floodedness": 2.0, "fluid_level_spread": 3.0,
                "lava": 4.0, "temperature": "x:t", "vegetation": "x:v", "continents": "x:c",
                "erosion": "x:e", "depth": "x:d", "ridges": "x:r", "final_density": "x:f",
                "preliminary_surface_level": {"type": "minecraft:constant", "argument": 64.0},
                "vein_toggle": 0.5, "vein_ridged": -1.0, "vein_gap": 0.0},
            "surface_rule": {"type": "minecraft:block", "result_state": {"Name": "minecraft:dirt"}},
            "default_block": {"Name": "minecraft:stone"}, "default_fluid": {"Name": "minecraft:water"},
            "aquifers_enabled": true, "ore_veins_enabled": true, "sea_level": 63,
            "disable_mob_generation": false, "legacy_random_source": false,
            "spawn_target": [{"temperature": [-1, 1], "humidity": [-1, 1], "continentalness": [-0.11, 1],
                "erosion": [-1, 1], "weirdness": [-1, -0.16], "depth": 0, "offset": 0}]});
        let files = BTreeMap::from([(
            "data/x/worldgen/noise_settings/s.json".to_owned(),
            serde_json::to_vec(&settings).unwrap(),
        )]);
        let (out, _) = upgrade_pack(&files);
        let read = |p: &str| -> Value { serde_json::from_slice(&out[p]).unwrap() };
        let s = read("data/x/worldgen/noise_settings/s.json");
        assert_eq!(s["material_rule"], json!("x:s/upgraded/material_rule"));
        let rule = read("data/x/worldgen/material_rule/s/upgraded/material_rule.json");
        let aquifers = &s["aquifers"];
        assert_eq!(aquifers["barrier"], json!(1.0));
        assert_eq!(
            aquifers["surface_level"],
            json!("x:s/upgraded/preliminary_surface_level")
        );
        assert_eq!(aquifers["exclusion"]["left"]["right"], json!("x:e"));
        let rules = rule["sequence"].as_array().unwrap();
        assert_eq!(rules.len(), 3);
        assert_eq!(rules[0]["type"], json!("minecraft:ore_vein"));
        assert_eq!(
            rules[2],
            json!({"type": "minecraft:block", "result_state": "minecraft:dirt"})
        );
        assert_eq!(
            s["spawn_target"],
            json!([{"x:t": [-1, 1], "x:v": [-1, 1], "x:c": [-0.11, 1], "x:e": [-1, 1], "x:r": [-1, -0.16]}])
        );
        assert_eq!(
            s["noise_router"]["chunk_surface_level"],
            json!({"type": "minecraft:interpolated", "cell_size_xz": 16, "cell_size_y": 1,
                "input": "x:s/upgraded/preliminary_surface_level"})
        );
    }

    #[test]
    fn features_lose_their_config_and_placements_their_spread() {
        let (v, _) = up(
            "data/x/worldgen/configured_feature/f.json",
            json!({"type": "minecraft:disk", "config": {"state_provider": {"type":
                "minecraft:simple_state_provider", "state": {"Name": "minecraft:clay"}}, "radius": 2}}),
        );
        assert_eq!(
            v,
            json!({"type": "minecraft:disk", "state_provider": "minecraft:clay", "radius": 2})
        );
        let (v, _) = up(
            "data/x/worldgen/placed_feature/p.json",
            json!({"feature": "x:f", "placement": [{"type": "minecraft:random_offset",
                "xz_spread": 0, "y_spread": -1}]}),
        );
        assert_eq!(
            v,
            json!({"feature": "x:f", "placement": [{"type": "minecraft:offset", "x": 0, "y": -1, "z": 0}]})
        );
    }

    #[test]
    fn biome_spawns_become_attributes() {
        let (v, _) = up(
            "data/x/worldgen/biome/b.json",
            json!({"spawners": {"monster": [{"type": "minecraft:zombie", "minCount": 4, "maxCount": 4,
                "weight": 95}], "ambient": []}, "spawn_costs": {}, "creature_spawn_probability": 0.07,
                "attributes": {"minecraft:visual/sky_color": "#78a7ff"}}),
        );
        assert_eq!(
            v["attributes"]["minecraft:gameplay/natural_mob_spawns"],
            json!({"modifier": "overlay", "argument": {"spawn_costs": {}, "spawns_by_category": {
                "monster": [{"type": "minecraft:zombie", "count": 4, "weight": 95}], "ambient": []}}})
        );
        assert_eq!(
            v["attributes"]["minecraft:gameplay/creature_world_gen_spawn_probability"],
            json!(0.07)
        );
        assert!(v.get("spawners").is_none());
    }

    #[test]
    fn carvers_move_and_spell_out_their_hard_coded_shape() {
        let files = BTreeMap::from([(
            "data/x/worldgen/configured_carver/c.json".to_owned(),
            serde_json::to_vec(&json!({"type": "minecraft:canyon", "config": {"probability": 0.01,
                "yScale": 3.0, "lava_level": {"above_bottom": 8}, "shape": {"width_smoothness": 3}}}))
            .unwrap(),
        )]);
        let (out, _) = upgrade_pack(&files);
        let v: Value = serde_json::from_slice(&out["data/x/worldgen/carver/c.json"]).unwrap();
        assert_eq!(
            v,
            json!({"type": "minecraft:canyon", "probability": 0.01,
                "shape": {"width_smoothness": 3, "y_scale": 3.0}})
        );
    }
}
