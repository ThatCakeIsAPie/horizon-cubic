//! The world type of a cubic world and its generator settings, as Create World
//! chooses them and as a world folder keeps them (docs/cubic-v2-plan.md,
//! "Generation"):
//!
//! * `horizon_cubic.json` carries `"type": "custom" | "hybrid" | "flat"` beside
//!   the window the world last held (a world made before the types has no
//!   `type` and is [`CubicKind::Hybrid`], what v1 generated);
//! * `cubicgen.json` is the generator's settings, CubicWorldGen's v9 JSON for
//!   Custom Cubic and Hybrid, the layers for Flat Cubic.
use horizon_world::cubic::custom::{CustomGeneratorSettings, FlatGeneratorSettings};
use std::path::Path;

/// The file a world's cubic generator settings are kept in.
pub const SETTINGS_FILE: &str = "cubicgen.json";
/// The file that makes a world cubic (`cubic.rs` owns its other fields).
const WORLD_FILE: &str = "horizon_cubic.json";

/// How a cubic world is generated.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CubicKind {
    /// CubicWorldGen's generator in every cube.
    Custom,
    /// Vanilla 26.3 between -64 and 320, Custom Cubic outside that band.
    Hybrid,
    /// Endless superflat layers.
    Flat,
}
impl CubicKind {
    /// Every kind, in the order the Create World button cycles them.
    pub const ALL: [Self; 3] = [Self::Custom, Self::Hybrid, Self::Flat];
    /// The name `horizon_cubic.json` stores.
    pub fn id(self) -> &'static str {
        match self {
            Self::Custom => "custom",
            Self::Hybrid => "hybrid",
            Self::Flat => "flat",
        }
    }
    pub fn from_id(id: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| kind.id() == id)
    }
    /// The language key of the name the Create World button shows.
    pub fn label(self) -> &'static str {
        match self {
            Self::Custom => "horizon.cubic.type.custom",
            Self::Hybrid => "horizon.cubic.type.hybrid",
            Self::Flat => "horizon.cubic.type.flat",
        }
    }
    /// The kind after `self` in the cycle, with `None` (Off) between the last
    /// and the first: `step` 1 goes forward, -1 back.
    pub fn cycle(current: Option<Self>, step: i32) -> Option<Self> {
        // Off, Custom, Hybrid, Flat.
        let at = current.map_or(0, |kind| {
            Self::ALL.iter().position(|k| *k == kind).unwrap_or(0) as i32 + 1
        });
        let next = (at + step).rem_euclid(Self::ALL.len() as i32 + 1);
        (next > 0).then(|| Self::ALL[next as usize - 1])
    }
    /// Whether the settings are CubicWorldGen's (`CustomGeneratorSettings`).
    pub fn has_terrain_settings(self) -> bool {
        self != Self::Flat
    }
}

/// The settings a new world of `kind` starts from, as pretty JSON.
pub fn default_json(kind: CubicKind) -> String {
    match kind {
        CubicKind::Flat => FlatGeneratorSettings::default().to_json_pretty(),
        _ => CustomGeneratorSettings::defaults().to_json_pretty(),
    }
}

/// `text` read as the settings of `kind` (an empty text is the preset) and
/// written back as pretty JSON, or why it is not.
pub fn normalize_json(kind: CubicKind, text: &str) -> Result<String, String> {
    match kind {
        CubicKind::Flat => FlatGeneratorSettings::from_json(text).map(|s| s.to_json_pretty()),
        _ => {
            let settings = CustomGeneratorSettings::from_json(text)?;
            settings.validate()?;
            Ok(settings.to_json_pretty())
        }
    }
}

fn invalid(message: String) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, message)
}

/// Writes `cubicgen.json` (`json` as pretty JSON; empty means the kind's
/// defaults) and the `type` of the world in `dir`, beside the rest of
/// `horizon_cubic.json`, which `cubic::mark_cubic_world` made.
pub fn save_world_generation(dir: &Path, kind: CubicKind, json: &str) -> std::io::Result<()> {
    let pretty = normalize_json(kind, json).map_err(invalid)?;
    std::fs::create_dir_all(dir)?;
    write_atomically(&dir.join(SETTINGS_FILE), pretty.as_bytes())?;
    let mut world = read_world_file(dir);
    world["type"] = serde_json::Value::from(kind.id());
    world["cubic"] = serde_json::Value::Bool(true);
    write_atomically(&dir.join(WORLD_FILE), world.to_string().as_bytes())
}

/// The kind and the settings text of the cubic world in `dir`; `None` when it
/// is not a cubic world. A world with no `type` is [`CubicKind::Hybrid`]; one
/// with no `cubicgen.json` has the kind's defaults.
pub fn load_world_generation(dir: &Path) -> Option<(CubicKind, String)> {
    let text = std::fs::read_to_string(dir.join(WORLD_FILE)).ok()?;
    let world: serde_json::Value = serde_json::from_str(&text).unwrap_or_default();
    let kind = world
        .get("type")
        .and_then(serde_json::Value::as_str)
        .and_then(CubicKind::from_id)
        .unwrap_or(CubicKind::Hybrid);
    let json =
        std::fs::read_to_string(dir.join(SETTINGS_FILE)).unwrap_or_else(|_| default_json(kind));
    Some((kind, json))
}

/// The `type` already in `horizon_cubic.json`, for `cubic::save_window` to keep
/// when it rewrites the file.
pub fn stored_type(dir: &Path) -> Option<String> {
    read_world_file(dir)
        .get("type")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
}

fn read_world_file(dir: &Path) -> serde_json::Value {
    std::fs::read_to_string(dir.join(WORLD_FILE))
        .ok()
        .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
        .filter(serde_json::Value::is_object)
        .unwrap_or_else(|| serde_json::json!({}))
}

fn write_atomically(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let mut temporary = path.as_os_str().to_owned();
    temporary.push(".tmp");
    let temporary = std::path::PathBuf::from(temporary);
    std::fs::write(&temporary, bytes)?;
    std::fs::rename(temporary, path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "horizon-cubic-settings-{name}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn the_cycle_goes_off_custom_hybrid_flat_and_back() {
        let mut kind = None;
        let mut seen = Vec::new();
        for _ in 0..4 {
            kind = CubicKind::cycle(kind, 1);
            seen.push(kind);
        }
        assert_eq!(
            seen,
            [
                Some(CubicKind::Custom),
                Some(CubicKind::Hybrid),
                Some(CubicKind::Flat),
                None
            ]
        );
        assert_eq!(CubicKind::cycle(None, -1), Some(CubicKind::Flat));
        assert_eq!(
            CubicKind::cycle(Some(CubicKind::Custom), -1),
            None,
            "one step back from the first kind is Off"
        );
    }

    #[test]
    fn a_saved_world_reads_back_its_kind_and_pretty_settings() {
        let dir = scratch("roundtrip");
        // What `mark_cubic_world` writes first.
        std::fs::write(
            dir.join("horizon_cubic.json"),
            r#"{"cubic":true,"window_min_y":-96}"#,
        )
        .unwrap();
        let mut settings = CustomGeneratorSettings::defaults();
        settings.dungeon_count = 33;
        save_world_generation(&dir, CubicKind::Custom, &settings.to_json_compact()).unwrap();
        let (kind, text) = load_world_generation(&dir).unwrap();
        assert_eq!(kind, CubicKind::Custom);
        assert!(text.contains('\n'), "cubicgen.json is the pretty form");
        assert_eq!(
            CustomGeneratorSettings::from_json(&text)
                .unwrap()
                .dungeon_count,
            33
        );
        // The window the world kept is still in its file.
        let world: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("horizon_cubic.json")).unwrap())
                .unwrap();
        assert_eq!(world["window_min_y"], -96);
        assert_eq!(world["type"], "custom");
        assert_eq!(stored_type(&dir).as_deref(), Some("custom"));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_world_without_a_type_is_hybrid_and_a_plain_folder_is_not_cubic() {
        let dir = scratch("legacy");
        assert_eq!(load_world_generation(&dir), None);
        std::fs::write(
            dir.join("horizon_cubic.json"),
            r#"{"cubic":true,"window_min_y":0}"#,
        )
        .unwrap();
        let (kind, text) = load_world_generation(&dir).unwrap();
        assert_eq!(kind, CubicKind::Hybrid);
        assert_eq!(
            CustomGeneratorSettings::from_json(&text).unwrap(),
            CustomGeneratorSettings::defaults(),
            "no cubicgen.json means the preset"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn flat_settings_are_kept_as_layers_and_bad_json_is_refused() {
        let dir = scratch("flat");
        save_world_generation(&dir, CubicKind::Flat, "").unwrap();
        let (kind, text) = load_world_generation(&dir).unwrap();
        assert_eq!(kind, CubicKind::Flat);
        assert_eq!(
            FlatGeneratorSettings::from_json(&text).unwrap(),
            FlatGeneratorSettings::default()
        );
        assert!(save_world_generation(&dir, CubicKind::Custom, "{ not json").is_err());
        let _ = std::fs::remove_dir_all(dir);
    }
}
