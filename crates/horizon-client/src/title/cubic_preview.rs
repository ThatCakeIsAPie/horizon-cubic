//! The terrain preview of the Advanced terrain page (CWG's `UITerrainPreview`):
//! a vertical cross-section of the density, 256 blocks across and 128 up,
//! computed on a worker thread with `CustomCubicGenerator::block_at` (density
//! and replacers, no caves) and drawn as coloured runs.
//!
//! The screen marks the preview dirty when a setting changes; the first draw
//! 200 ms after the last change starts one job. A job's result is dropped when
//! a newer job was started.
use horizon_world::cubic::custom::{
    CustomCubicGenerator, CustomGeneratorSettings, FixedBiome,
    testing::{default_block_names, registry_of},
};
use std::sync::{
    Arc, Mutex, OnceLock,
    atomic::{AtomicU64, Ordering},
};
use std::time::{Duration, Instant};

/// Blocks across and up the section covers.
pub const WIDTH_BLOCKS: i32 = 256;
pub const HEIGHT_BLOCKS: i32 = 128;
/// Blocks per drawn cell.
pub const CELL: i32 = 2;
pub const COLUMNS: usize = (WIDTH_BLOCKS / CELL) as usize;
pub const ROWS: usize = (HEIGHT_BLOCKS / CELL) as usize;
/// The pause after the last change before a job starts.
const DEBOUNCE: Duration = Duration::from_millis(200);
/// The preview's own seed (the world's is not known to the screen).
const SEED: i64 = 0x5eed_c0de;

/// What a cell shows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cell {
    Air,
    Water,
    Lava,
    Grass,
    Dirt,
    Sand,
    Stone,
    Deepslate,
    Other,
}
impl Cell {
    /// The cell's colour (ARGB, opaque).
    pub fn color(self) -> u32 {
        match self {
            Self::Air => 0xFF8FB7FF,
            Self::Water => 0xFF3F6FE0,
            Self::Lava => 0xFFE8731A,
            Self::Grass => 0xFF5DAA3B,
            Self::Dirt => 0xFF7A5232,
            Self::Sand => 0xFFD9C98B,
            Self::Stone => 0xFF8A8A8A,
            Self::Deepslate => 0xFF4A4A52,
            Self::Other => 0xFFB05AA8,
        }
    }
}

/// A finished cross-section: `ROWS` rows from the top down, `COLUMNS` cells
/// each, and the block Y of the top row.
#[derive(Clone, Debug, PartialEq)]
pub struct Image {
    pub top_y: i32,
    pub cells: Vec<Cell>,
}
impl Image {
    pub fn row(&self, row: usize) -> &[Cell] {
        &self.cells[row * COLUMNS..(row + 1) * COLUMNS]
    }
}

#[derive(Default)]
struct Shared {
    latest: AtomicU64,
    done: Mutex<Option<(u64, Result<Image, String>)>>,
}

/// The preview of one Customize screen. Cloning shares the worker's result.
#[derive(Clone, Default)]
pub struct Preview {
    shared: Arc<Shared>,
    /// When the settings last changed, and nothing was started since.
    dirty: Arc<Mutex<Option<Instant>>>,
    image: Arc<Mutex<Option<Result<Image, String>>>>,
}
impl std::fmt::Debug for Preview {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Preview")
    }
}
impl Preview {
    /// A setting changed: a job starts after the pause.
    pub fn touch(&self) {
        if let Ok(mut dirty) = self.dirty.lock() {
            *dirty = Some(Instant::now());
        }
    }
    /// A preview that starts at once on its first poll.
    pub fn fresh() -> Self {
        let preview = Self::default();
        if let Ok(mut dirty) = preview.dirty.lock() {
            *dirty = Some(
                Instant::now()
                    .checked_sub(DEBOUNCE)
                    .unwrap_or_else(Instant::now),
            );
        }
        preview
    }
    /// Starts a job when a change has settled and takes a finished result in.
    /// `axis_z` shows the cross-section along Z instead of X.
    pub fn poll(&self, settings: &CustomGeneratorSettings, axis_z: bool) {
        let due = self
            .dirty
            .lock()
            .ok()
            .and_then(|mut dirty| match *dirty {
                Some(at) if at.elapsed() >= DEBOUNCE => dirty.take(),
                _ => None,
            })
            .is_some();
        if due {
            let id = self.shared.latest.fetch_add(1, Ordering::SeqCst) + 1;
            let shared = self.shared.clone();
            let settings = settings.clone();
            let spawned = std::thread::Builder::new()
                .name("cubic-preview".into())
                .spawn(move || {
                    let image = compute(&settings, axis_z);
                    if shared.latest.load(Ordering::SeqCst) == id
                        && let Ok(mut done) = shared.done.lock()
                    {
                        *done = Some((id, image));
                    }
                });
            if spawned.is_err() {
                self.touch();
            }
        }
        if let Ok(mut done) = self.shared.done.lock()
            && let Some((id, image)) = done.take()
            && id == self.shared.latest.load(Ordering::SeqCst)
            && let Ok(mut slot) = self.image.lock()
        {
            *slot = Some(image);
        }
    }
    /// The latest finished image, or the reason there is none.
    pub fn image(&self) -> Option<Result<Image, String>> {
        self.image.lock().ok().and_then(|slot| slot.clone())
    }
}

/// The registry the preview generates against: the blocks of the default
/// settings, whatever blocks the player's settings name.
fn registry() -> &'static horizon_world::registry::BlockRegistry {
    static REGISTRY: OnceLock<&'static horizon_world::registry::BlockRegistry> = OnceLock::new();
    REGISTRY.get_or_init(|| registry_of(&default_block_names()))
}

/// What the cross-section shows of the settings: the terrain, its replacers and
/// the biome, none of the ores, lakes or caves.
fn terrain_only(settings: &CustomGeneratorSettings) -> CustomGeneratorSettings {
    let mut s = settings.clone();
    s.standard_ores.clear();
    s.periodic_gaussian_ores.clear();
    s.lakes.clear();
    s.caves.clear();
    s.cube_areas.clear();
    s
}

/// The cross-section of `settings`: along X (Z fixed at 0), or along Z. The
/// top row is `expectedBaseHeight + 80`, so the usual surface is in view.
pub fn compute(settings: &CustomGeneratorSettings, axis_z: bool) -> Result<Image, String> {
    let registry = registry();
    let biomes = Arc::new(FixedBiome::new("plains"));
    let generator =
        CustomCubicGenerator::new(SEED, &terrain_only(settings), registry, biomes.clone())
            .or_else(|_| {
                // Replacers naming blocks this registry lacks: the bare terrain.
                let mut bare = terrain_only(settings);
                bare.replacers.clear();
                CustomCubicGenerator::new(SEED, &bare, registry, biomes)
            })?;
    let id = |name: &str| {
        registry
            .default_state(&format!("minecraft:{name}"))
            .map(|s| s.0)
    };
    let known = [
        (id("air"), Cell::Air),
        (id("water"), Cell::Water),
        (id("lava"), Cell::Lava),
        (id("grass_block"), Cell::Grass),
        (id("dirt"), Cell::Dirt),
        (id("coarse_dirt"), Cell::Dirt),
        (id("podzol"), Cell::Dirt),
        (id("sand"), Cell::Sand),
        (id("red_sand"), Cell::Sand),
        (id("sandstone"), Cell::Sand),
        (id("stone"), Cell::Stone),
        (id("gravel"), Cell::Stone),
        (id("granite"), Cell::Stone),
        (id("diorite"), Cell::Stone),
        (id("andesite"), Cell::Stone),
        (id("deepslate"), Cell::Deepslate),
        (id("tuff"), Cell::Deepslate),
    ];
    let top_y = (settings.expected_base_height + 80.0)
        .round()
        .clamp(-1.0e6, 1.0e6) as i32;
    let mut cells = Vec::with_capacity(COLUMNS * ROWS);
    for row in 0..ROWS as i32 {
        let y = top_y - row * CELL;
        for column in 0..COLUMNS as i32 {
            let along = column * CELL;
            let (x, z) = if axis_z { (0, along) } else { (along, 0) };
            let block = generator.block_at(x, y, z, CELL);
            cells.push(
                known
                    .iter()
                    .find(|(id, _)| *id == Some(block))
                    .map_or(Cell::Other, |(_, cell)| *cell),
            );
        }
    }
    Ok(Image { top_y, cells })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_preset_cross_section_has_sky_above_and_rock_below() {
        let image = compute(&CustomGeneratorSettings::defaults(), false).unwrap();
        assert_eq!(image.cells.len(), COLUMNS * ROWS);
        assert!(
            image.row(0).iter().all(|c| *c == Cell::Air),
            "the top row is sky"
        );
        let bottom = image.row(ROWS - 1);
        assert!(
            bottom
                .iter()
                .filter(|c| matches!(c, Cell::Stone | Cell::Deepslate))
                .count()
                > COLUMNS / 2,
            "the bottom row is mostly rock"
        );
    }

    #[test]
    fn a_taller_world_shows_more_rock_than_a_flat_one() {
        let solid = |settings: &CustomGeneratorSettings| {
            compute(settings, false)
                .unwrap()
                .cells
                .iter()
                .filter(|c| !matches!(c, Cell::Air | Cell::Water))
                .count()
        };
        let base = CustomGeneratorSettings::defaults();
        let mut high = base.clone();
        high.height_offset = 200.0;
        assert!(solid(&high) > solid(&base));
    }
}
