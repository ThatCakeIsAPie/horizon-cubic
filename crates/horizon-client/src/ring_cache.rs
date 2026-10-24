//! Stronghold positions kept in the dimension's folder (`horizon_rings.json`),
//! so that loading a world skips their biome searches
//! (ConcentricRingsStructurePlacement: ~25 CPU-seconds with To The Max's
//! climate). Vanilla searches again on every start; the positions depend only
//! on the seed, the world generation data and the code, so a fingerprint of
//! those (as for baked far terrain, `lod::bake`) guards the file. A file that
//! does not match is searched again and replaced.
use crate::lod::bake::{exe_stamp, files_digest, world_packs_digest};
use horizon_core::position::ChunkPos;
use horizon_world::worldgen::{generator::OverworldGenerator, structure::start::Rings};
use serde_json::{Value, json};
use std::{fs, path::Path};

const FORMAT: &str = "horizon-rings 1";
const FILE: &str = "horizon_rings.json";

/// Gives `generator` (the dimension `name` of the world with `seed`, loaded
/// from the bundle `worldgen`) the positions kept in `dimension`'s folder, or
/// keeps them there once its first structure lookup has found them.
pub fn keep(
    generator: &OverworldGenerator,
    dimension: &Path,
    worldgen: &Path,
    seed: i64,
    name: &str,
) {
    let fingerprint = fingerprint(dimension, worldgen, seed, name);
    let path = dimension.join(FILE);
    if let Some(rings) = read(&path, &fingerprint) {
        if generator.preset_rings(rings) {
            log::info!("Horizon: stronghold positions read from {}", path.display());
            return;
        }
        log::info!("Horizon: the kept stronghold positions do not fit {name}; searching again");
    }
    generator.on_rings(move |rings| match write(&path, &fingerprint, rings) {
        Ok(()) => log::info!("Horizon: stronghold positions kept in {}", path.display()),
        Err(e) => log::warn!(
            "Horizon: stronghold positions not kept ({}): {e}",
            path.display()
        ),
    });
}

/// What the positions depend on besides the code's own version.
fn fingerprint(dimension: &Path, worldgen: &Path, seed: i64, name: &str) -> String {
    format!(
        "{FORMAT}\nversion {}\nexe {}\nseed {seed}\ndimension {name}\nworldgen {}\ndatapacks {}\n",
        env!("CARGO_PKG_VERSION"),
        exe_stamp(),
        files_digest(worldgen),
        world_packs_digest(dimension),
    )
}

/// The file's positions by set name, if it was written with `fingerprint`.
fn read(path: &Path, fingerprint: &str) -> Option<Rings> {
    let file: Value = serde_json::from_str(&fs::read_to_string(path).ok()?).ok()?;
    if file.get("fingerprint")?.as_str()? != fingerprint {
        log::info!("Horizon: the kept stronghold positions are out of date; searching again");
        return None;
    }
    file.get("rings")?
        .as_object()?
        .iter()
        .map(|(set, chunks)| {
            let chunks = chunks
                .as_array()?
                .iter()
                .map(|c| {
                    let [x, z] = c.as_array()?.as_slice() else {
                        return None;
                    };
                    let coord = |v: &Value| v.as_i64().and_then(|v| i32::try_from(v).ok());
                    Some(ChunkPos::new(coord(x)?, coord(z)?))
                })
                .collect::<Option<Vec<_>>>()?;
            Some((set.clone(), chunks))
        })
        .collect()
}

/// Writes the file whole, then moves it into place: a crash leaves the old
/// one or none.
fn write(path: &Path, fingerprint: &str, rings: &Rings) -> std::io::Result<()> {
    let rings: serde_json::Map<String, Value> = rings
        .iter()
        .map(|(set, chunks)| {
            let chunks = chunks.iter().map(|c| json!([c.x, c.z])).collect();
            (set.clone(), Value::Array(chunks))
        })
        .collect();
    let text = json!({ "fingerprint": fingerprint, "rings": rings }).to_string();
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
    }
    let partial = path.with_extension("json.partial");
    fs::write(&partial, text)?;
    fs::rename(&partial, path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positions_round_trip_under_their_fingerprint_only() {
        let dir = std::env::temp_dir().join(format!("horizon-rings-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let path = dir.join(FILE);
        let rings: Rings = [
            (
                "minecraft:strongholds".to_owned(),
                vec![ChunkPos::new(-81, 99), ChunkPos::new(120, -3)],
            ),
            ("empty".to_owned(), Vec::new()),
        ]
        .into();
        write(&path, "a\n", &rings).unwrap();
        assert_eq!(read(&path, "a\n"), Some(rings.clone()));
        assert_eq!(read(&path, "b\n"), None);
        write(&path, "b\n", &rings).unwrap();
        assert_eq!(read(&path, "b\n"), Some(rings));
        assert!(!path.with_extension("json.partial").exists());
        fs::write(&path, "{\"fingerprint\":\"b\\n\",\"rings\":{\"s\":[[1]]}}").unwrap();
        assert_eq!(read(&path, "b\n"), None);
        fs::remove_dir_all(&dir).unwrap();
    }
}
