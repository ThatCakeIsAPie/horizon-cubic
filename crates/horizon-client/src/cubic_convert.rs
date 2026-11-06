//! Converts a vanilla world to a cubic **Hybrid** world (docs/cubic-chunks.md,
//! "Converting a world").
//!
//! The source world is never written. [`convert_with`] builds the copy in a
//! sibling folder `<dst>.converting` and renames it to `dst` when it is done,
//! so a world that fails half way leaves nothing behind.
//!
//! * Everything except the Overworld's chunk data is copied as it is:
//!   `level.dat`, players, advancements, statistics, `data/`, the Nether and
//!   the End, `icon.png`, data packs. The Overworld's `region/`, `entities/`
//!   and `poi/` are not (in 26.3 they are under `dimensions/minecraft/overworld/`;
//!   a world still in the old layout has them in its root).
//! * Each Overworld column is read as the game reads it (`SaveDir`: the
//!   data fixers bring an older record to the current version) and written
//!   as 24 cubes (-64..320) with [`crate::cubic::vanilla_column_records`],
//!   the code that saves a column the game generated: the marker cube with
//!   the structure starts and references and the surface heights, block
//!   entities, entities (`horizon:entities`, by the cube they are in), block
//!   and fluid ticks. Cubes of one block state go to the uniform sidecar.
//!   POI is not carried: the game rebuilds it from blocks.
//! * Vanilla's bedrock floor (y -64..-60) becomes deepslate, as in a Hybrid
//!   world the game generates, unless [`ConvertOptions::keep_bedrock`].
//! * `horizon_cubic.json` (`"type": "hybrid"`, a window around the spawn)
//!   and `cubicgen.json` (the Hybrid defaults) make it a cubic world.
//!   Columns the source never generated stay ungenerated: the game makes
//!   them with the vanilla generator, and the cubes outside -64..320 with
//!   CubicWorldGen's.
//!
//! A world saved by an older version is copied whole first and upgraded in
//! the copy (`save::upgrade_older_world`), then its Overworld is converted
//! from there and its chunk folders removed. A world with a taller or
//! shorter Overworld than vanilla's (the 4,096-high To The Max data pack)
//! is refused, as are flat and debug worlds.
use crate::biome::Biomes;
use crate::chunk::{ColumnLayout, LevelShape};
use crate::cubic::{
    DEFAULT_VERTICAL, replace_bedrock, split_by_cube, store_record, vanilla_column_records,
    window_cubes,
};
use crate::save::{SaveDir, SaveTables, decode_with, dimension_dir, number, read_dat, text};
use horizon_core::position::SectionPos;
use horizon_world::cubic::store::CubeStore;
use horizon_world::nbt::Tag;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{RecvTimeoutError, channel};
use std::time::{Duration, Instant};

pub type Result<T> = std::result::Result<T, String>;

/// The Overworld's dimension id.
const OVERWORLD: &str = "minecraft:overworld";
/// The folders of a dimension that hold its chunks.
const CHUNK_FOLDERS: [&str; 3] = ["region", "entities", "poi"];
/// Sections of a vanilla Overworld column: Y -4..=19.
const VANILLA_SECTIONS: std::ops::RangeInclusive<i64> = -4..=19;
/// The unreadable columns a [`Report`] lists.
const LISTED_UNREADABLE: usize = 32;

/// How a conversion runs.
#[derive(Clone, Debug)]
pub struct ConvertOptions {
    /// The prepared resources (`horizon-prepare`): the block registry and
    /// biomes records are read with.
    pub resources: PathBuf,
    /// Keep vanilla's bedrock floor at y -64..-60 instead of turning it into
    /// deepslate (a world that can go deeper has no floor).
    pub keep_bedrock: bool,
    /// Worker threads; 0 is the number of cores, at most 8.
    pub threads: usize,
}

impl ConvertOptions {
    pub fn new(resources: impl Into<PathBuf>) -> Self {
        Self {
            resources: resources.into(),
            keep_bedrock: false,
            threads: 0,
        }
    }
}

/// Where the game looks for its resources when nothing says otherwise.
pub fn default_resources() -> PathBuf {
    if let Some(dir) = std::env::var_os("HORIZON_TEST_RESOURCES") {
        return dir.into();
    }
    let sibling = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|dir| dir.join("resources")));
    match sibling {
        Some(dir) if dir.join("horizon-resources.json").exists() => dir,
        _ => Path::new(".local/horizon-client").join(horizon_core::VANILLA_VERSION),
    }
}

/// How far a conversion has come, for the caller's progress line.
#[derive(Clone, Debug, PartialEq)]
pub enum Progress {
    /// Checking the source world.
    Checking,
    /// Copying what is not Overworld chunk data.
    Copying { files: u64, total: u64 },
    /// Upgrading a world saved by an older version.
    Upgrading,
    /// Rewriting columns as cubes. `columns` counts every column looked at
    /// (converted or skipped).
    Converting {
        columns: u64,
        total: u64,
        skipped: u64,
        per_second: f64,
    },
    /// Writing the cubic markers and moving the copy into place.
    Finishing,
}

/// What a conversion did.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Report {
    /// Columns written as cubes.
    pub columns_converted: u64,
    /// Columns that would not read (damaged, or too old for the data
    /// fixers): not converted.
    pub columns_unreadable: u64,
    /// Columns saved before they were finished (vanilla's proto chunks at
    /// the edge of the explored world): not converted, the game generates
    /// them when they are visited.
    pub columns_unfinished: u64,
    /// The first unreadable columns (chunk x, z).
    pub unreadable: Vec<(i32, i32)>,
    /// Region files of the Overworld read.
    pub regions: u64,
    /// Cube records written (the uniform ones too).
    pub cubes_written: u64,
    /// Of those, the cubes of one block state (the sidecar's).
    pub cubes_uniform: u64,
    pub block_entities: u64,
    pub entities: u64,
    /// Bedrock blocks turned into deepslate.
    pub bedrock_replaced: u64,
    /// Files copied from the source.
    pub files_copied: u64,
    pub seconds: f64,
    pub columns_per_second: f64,
}

impl Report {
    /// The columns not converted.
    pub fn columns_skipped(&self) -> u64 {
        self.columns_unreadable + self.columns_unfinished
    }
}

/// `convert_with`, with the default options: bedrock becomes deepslate and
/// the resources are where the game keeps them ([`default_resources`]).
pub fn convert_to_cubic(
    src_world: &Path,
    dst_world: &Path,
    progress: impl FnMut(Progress),
) -> Result<Report> {
    convert_with(
        src_world,
        dst_world,
        &ConvertOptions::new(default_resources()),
        progress,
    )
}

/// The folder a conversion of `saves/<folder>` is written to by default:
/// `<folder> (Cubic)`, then `<folder> (Cubic 2)`... the first that is free.
pub fn destination_for(saves: &Path, folder: &str) -> PathBuf {
    let first = saves.join(format!("{folder} (Cubic)"));
    if !first.exists() {
        return first;
    }
    (2..)
        .map(|n| saves.join(format!("{folder} (Cubic {n})")))
        .find(|path| !path.exists())
        .expect("a free name")
}

/// Why `world` cannot be converted, if it cannot (without reading chunks).
pub fn check_source(world: &Path) -> Result<()> {
    if !world.join("level.dat").is_file() {
        return Err(format!("{} is not a world (no level.dat)", world.display()));
    }
    if crate::cubic::is_cubic_world(world) {
        return Err("this world is already a Cubic Chunks world".into());
    }
    if crate::save::SessionLock::is_locked(world) {
        return Err("this world is open in the game; close it first".into());
    }
    if crate::save::incompatible_version(world).is_some() {
        return Err("this world was saved by a newer version of the game".into());
    }
    if let Some(gen_world) = crate::save::WorldGen::read(world)
        && matches!(
            gen_world.preset,
            crate::save::WorldPreset::Flat | crate::save::WorldPreset::Debug
        )
    {
        return Err("flat and debug worlds cannot be converted".into());
    }
    if let Some(issue) = height_issue(world) {
        return Err(issue);
    }
    Ok(())
}

/// Converts the world in `src_world` into the new cubic world `dst_world`,
/// which must not exist. `progress` is called from this thread, a few
/// times a second while columns convert.
pub fn convert_with(
    src_world: &Path,
    dst_world: &Path,
    options: &ConvertOptions,
    mut progress: impl FnMut(Progress),
) -> Result<Report> {
    let started = Instant::now();
    progress(Progress::Checking);
    check_source(src_world)?;
    if dst_world.exists() {
        return Err(format!("{} already exists", dst_world.display()));
    }
    let work = {
        let mut name = dst_world.as_os_str().to_owned();
        name.push(".converting");
        PathBuf::from(name)
    };
    if work.exists() {
        // A run that did not finish: ours alone.
        std::fs::remove_dir_all(&work).map_err(|e| format!("{}: {e}", work.display()))?;
    }
    let result = convert_into(src_world, &work, options, &mut progress).and_then(|mut report| {
        progress(Progress::Finishing);
        std::fs::rename(&work, dst_world)
            .map_err(|e| format!("moving the copy to {}: {e}", dst_world.display()))?;
        report.seconds = started.elapsed().as_secs_f64();
        report.columns_per_second =
            report.columns_converted as f64 / report.seconds.max(f64::EPSILON);
        Ok(report)
    });
    if result.is_err() {
        let _ = std::fs::remove_dir_all(&work);
    }
    result
}

fn convert_into(
    src: &Path,
    work: &Path,
    options: &ConvertOptions,
    progress: &mut impl FnMut(Progress),
) -> Result<Report> {
    let mut report = Report::default();
    let old = saved_before_current(src);
    // 1. Everything but the Overworld's chunk data (a world to be upgraded is
    //    copied whole: the upgrade works on files in place).
    let skip = |relative: &Path| !old && is_overworld_chunk_data(relative);
    let mut total = 0;
    count_files(src, src, &skip, &mut total)?;
    let mut copied = 0;
    copy_tree(src, src, work, &skip, &mut copied, total, progress)?;
    report.files_copied = copied;
    // 2. An older world is upgraded in the copy; the chunks are read there.
    let chunk_world = if old {
        progress(Progress::Upgrading);
        crate::save::upgrade_older_world(work).map_err(|e| e.to_string())?;
        work
    } else {
        src
    };
    // 3. The Overworld's columns.
    convert_columns(chunk_world, work, options, progress, &mut report)?;
    if old {
        remove_chunk_folders(work);
    }
    // 4. The cubic markers, and a name that tells the copy from the source.
    progress(Progress::Finishing);
    let window = spawn_window(work);
    crate::cubic_settings::save_world_generation(
        work,
        crate::cubic_settings::CubicKind::Hybrid,
        &crate::cubic_settings::default_json(crate::cubic_settings::CubicKind::Hybrid),
    )
    .map_err(|e| format!("cubicgen.json: {e}"))?;
    crate::cubic::save_window(work, window.0, window.1)
        .map_err(|e| format!("horizon_cubic.json: {e}"))?;
    rename_level(work);
    Ok(report)
}

/// The window (bottom, height in blocks) the converted world opens with:
/// centred on the world spawn.
fn spawn_window(world: &Path) -> (i32, i32) {
    let spawn_y = crate::save::Level::read_summary(world)
        .and_then(|level| level.world_spawn.map(|s| s[1]))
        .unwrap_or(64);
    let v = i32::from(DEFAULT_VERTICAL);
    let centre = spawn_y.div_euclid(16);
    ((centre - v - 1) * 16, window_cubes(v) * 16)
}

/// Gives the copy's level.dat the name `<name> (Cubic)`.
fn rename_level(world: &Path) {
    let path = world.join("level.dat");
    let Some(mut root) = read_dat(&path) else {
        return;
    };
    let name = root
        .get("Data")
        .and_then(|d| text(d.get("LevelName")))
        .unwrap_or_default();
    if let Some(Tag::Compound(data)) = root.get_mut("Data") {
        data.insert(
            "LevelName".into(),
            Tag::String(format!("{name} (Cubic)").as_str().into()),
        );
        if let Err(e) = crate::save::write_dat(&path, root) {
            log::warn!("Horizon: the converted world keeps its name: {e}");
        }
    }
}

/// Whether the world's level.dat is from before this version's data
/// version (its files need the data fixers).
fn saved_before_current(world: &Path) -> bool {
    let Some(root) = read_dat(&world.join("level.dat")) else {
        return false;
    };
    let version = root
        .get("Data")
        .and_then(|d| number(d.get("DataVersion")))
        .unwrap_or(-1);
    version < i64::from(horizon_world::DATA_VERSION)
}

/// Whether `relative` (in the world folder) is Overworld chunk data.
fn is_overworld_chunk_data(relative: &Path) -> bool {
    let parts: Vec<_> = relative
        .components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect();
    let in_dimension = parts.len() == 4
        && parts[0] == "dimensions"
        && parts[1] == "minecraft"
        && parts[2] == "overworld"
        && CHUNK_FOLDERS.contains(&parts[3].as_str());
    let in_root = parts.len() == 1 && CHUNK_FOLDERS.contains(&parts[0].as_str());
    in_dimension || in_root
}

fn remove_chunk_folders(world: &Path) {
    for base in [dimension_dir(world, OVERWORLD), world.to_owned()] {
        for folder in CHUNK_FOLDERS {
            let _ = std::fs::remove_dir_all(base.join(folder));
        }
    }
}

fn count_files(
    root: &Path,
    dir: &Path,
    skip: &dyn Fn(&Path) -> bool,
    total: &mut u64,
) -> Result<()> {
    for entry in read_dir(dir)? {
        let path = entry.path();
        let relative = path.strip_prefix(root).unwrap_or(&path);
        if skip(relative) || is_session_lock(relative) {
            continue;
        }
        let kind = entry.file_type().map_err(|e| io_error(&path, e))?;
        if kind.is_dir() {
            count_files(root, &path, skip, total)?;
        } else if kind.is_file() {
            *total += 1;
        }
    }
    Ok(())
}

fn is_session_lock(relative: &Path) -> bool {
    relative == Path::new("session.lock")
}

fn copy_tree(
    root: &Path,
    dir: &Path,
    to: &Path,
    skip: &dyn Fn(&Path) -> bool,
    copied: &mut u64,
    total: u64,
    progress: &mut impl FnMut(Progress),
) -> Result<()> {
    std::fs::create_dir_all(to).map_err(|e| io_error(to, e))?;
    for entry in read_dir(dir)? {
        let path = entry.path();
        let relative = path.strip_prefix(root).unwrap_or(&path);
        if skip(relative) || is_session_lock(relative) {
            continue;
        }
        let target = to.join(entry.file_name());
        let kind = entry.file_type().map_err(|e| io_error(&path, e))?;
        if kind.is_symlink() {
            // The game refuses a world with links (`DirectoryValidator`).
            log::warn!("Horizon: not copying the symbolic link {}", path.display());
        } else if kind.is_dir() {
            copy_tree(root, &path, &target, skip, copied, total, progress)?;
        } else if kind.is_file() {
            std::fs::copy(&path, &target).map_err(|e| io_error(&path, e))?;
            *copied += 1;
            if copied.is_multiple_of(64) || *copied == total {
                progress(Progress::Copying {
                    files: *copied,
                    total,
                });
            }
        }
    }
    Ok(())
}

fn read_dir(dir: &Path) -> Result<Vec<std::fs::DirEntry>> {
    std::fs::read_dir(dir)
        .map_err(|e| io_error(dir, e))?
        .collect::<std::io::Result<Vec<_>>>()
        .map_err(|e| io_error(dir, e))
}

fn io_error(path: &Path, e: std::io::Error) -> String {
    format!("{}: {e}", path.display())
}

/// The folder holding the Overworld's `region/`: its dimension folder in 26.3,
/// else the world's root (the layout before dimensions had folders).
fn overworld_source(world: &Path) -> PathBuf {
    let dimension = dimension_dir(world, OVERWORLD);
    if dimension.join("region").is_dir() || !world.join("region").is_dir() {
        dimension
    } else {
        world.to_owned()
    }
}

/// A region file of the source: its position and the columns it lists.
struct SourceRegion {
    chunks: Vec<(i32, i32)>,
}

fn list_regions(dir: &Path) -> Vec<SourceRegion> {
    let Ok(entries) = std::fs::read_dir(dir.join("region")) else {
        return Vec::new();
    };
    let mut paths: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "mca"))
        .collect();
    paths.sort();
    paths
        .into_iter()
        .filter_map(|path| {
            let region = horizon_world::region::RegionFile::open(&path).ok()?;
            let chunks: Vec<(i32, i32)> = region.chunks().map(|(p, _)| (p.x, p.z)).collect();
            (!chunks.is_empty()).then_some(SourceRegion { chunks })
        })
        .collect()
}

/// What a worker found, added to the report as regions finish.
#[derive(Default)]
struct Counts {
    converted: AtomicU64,
    unreadable: AtomicU64,
    unfinished: AtomicU64,
    done: AtomicU64,
    cubes: AtomicU64,
    uniform: AtomicU64,
    block_entities: AtomicU64,
    entities: AtomicU64,
    bedrock: AtomicU64,
}

fn convert_columns(
    chunk_world: &Path,
    work: &Path,
    options: &ConvertOptions,
    progress: &mut impl FnMut(Progress),
    report: &mut Report,
) -> Result<()> {
    let source = overworld_source(chunk_world);
    let regions = list_regions(&source);
    let total: u64 = regions.iter().map(|r| r.chunks.len() as u64).sum();
    report.regions = regions.len() as u64;
    let assets = crate::assets::Assets::load(&options.resources)
        .map_err(|e| format!("resources {}: {e}", options.resources.display()))?;
    let biomes = Biomes::load(&options.resources)
        .map_err(|e| format!("biomes in {}: {e}", options.resources.display()))?;
    let tables = SaveTables::new(&assets);
    let save = SaveDir::new(source.clone(), LevelShape::OVERWORLD);
    let _ = save.tables.set(Arc::new(SaveTables::new(&assets)));
    let (Some(bedrock), Some(deepslate)) = (
        assets
            .registry
            .default_state("minecraft:bedrock")
            .map(|s| s.0),
        assets
            .registry
            .default_state("minecraft:deepslate")
            .map(|s| s.0),
    ) else {
        return Err("the resources have no bedrock or deepslate".into());
    };
    let level_dir = dimension_dir(work, OVERWORLD);
    std::fs::create_dir_all(&level_dir).map_err(|e| io_error(&level_dir, e))?;

    let threads = match options.threads {
        0 => std::thread::available_parallelism()
            .map_or(2, |n| n.get())
            .clamp(1, 8),
        n => n,
    }
    .min(regions.len().max(1));
    let counts = Counts::default();
    let next = AtomicUsize::new(0);
    let failed = AtomicBool::new(false);
    let unreadable = std::sync::Mutex::new(Vec::<(i32, i32)>::new());
    let error = std::sync::Mutex::new(None::<String>);
    let started = Instant::now();
    let (finished, wait) = channel::<()>();
    let job = Job {
        save: &save,
        assets: &assets,
        biomes: &biomes,
        tables: &tables,
        level_dir: &level_dir,
        keep_bedrock: options.keep_bedrock,
        bedrock,
        deepslate,
        counts: &counts,
        unreadable: &unreadable,
    };
    std::thread::scope(|scope| {
        for _ in 0..threads {
            let finished = finished.clone();
            let (job, regions, next, failed, error) = (&job, &regions, &next, &failed, &error);
            scope.spawn(move || {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(region) = regions.get(i) else {
                        break;
                    };
                    if failed.load(Ordering::Relaxed) {
                        break;
                    }
                    if let Err(e) = job.region(region, failed) {
                        failed.store(true, Ordering::Relaxed);
                        error.lock().unwrap().get_or_insert(e);
                        break;
                    }
                }
                let _ = finished.send(());
            });
        }
        drop(finished);
        let mut running = threads;
        while running > 0 {
            match wait.recv_timeout(Duration::from_millis(200)) {
                Ok(()) => running -= 1,
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => break,
            }
            let done = counts.done.load(Ordering::Relaxed);
            progress(Progress::Converting {
                columns: done,
                total,
                skipped: counts.unreadable.load(Ordering::Relaxed)
                    + counts.unfinished.load(Ordering::Relaxed),
                per_second: counts.converted.load(Ordering::Relaxed) as f64
                    / started.elapsed().as_secs_f64().max(0.001),
            });
        }
    });
    if let Some(e) = error.into_inner().unwrap() {
        return Err(e);
    }
    report.columns_converted = counts.converted.load(Ordering::Relaxed);
    report.columns_unreadable = counts.unreadable.load(Ordering::Relaxed);
    report.columns_unfinished = counts.unfinished.load(Ordering::Relaxed);
    report.cubes_written = counts.cubes.load(Ordering::Relaxed);
    report.cubes_uniform = counts.uniform.load(Ordering::Relaxed);
    report.block_entities = counts.block_entities.load(Ordering::Relaxed);
    report.entities = counts.entities.load(Ordering::Relaxed);
    report.bedrock_replaced = counts.bedrock.load(Ordering::Relaxed);
    let mut listed = unreadable.into_inner().unwrap();
    listed.sort_unstable();
    listed.truncate(LISTED_UNREADABLE);
    report.unreadable = listed;
    Ok(())
}

/// What the workers share.
struct Job<'a> {
    save: &'a SaveDir,
    assets: &'a crate::assets::Assets,
    biomes: &'a Biomes,
    tables: &'a SaveTables,
    level_dir: &'a Path,
    keep_bedrock: bool,
    bedrock: u32,
    deepslate: u32,
    counts: &'a Counts,
    unreadable: &'a std::sync::Mutex<Vec<(i32, i32)>>,
}

impl Job<'_> {
    /// One source region. It has a store of its own: the cube files it
    /// writes (one region per cube Y) belong to no other worker, and the
    /// store's open-file limit holds its 24 of them.
    fn region(&self, region: &SourceRegion, failed: &AtomicBool) -> Result<()> {
        let store = CubeStore::new(self.level_dir);
        for &(x, z) in &region.chunks {
            if failed.load(Ordering::Relaxed) {
                return Ok(());
            }
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.column(&store, x, z)
            }));
            match outcome {
                Ok(Ok(Column::Converted)) => {}
                Ok(Ok(Column::Unfinished)) => {
                    self.counts.unfinished.fetch_add(1, Ordering::Relaxed);
                }
                Ok(Ok(Column::Unreadable)) | Err(_) => self.unreadable(x, z),
                Ok(Err(e)) => return Err(e),
            }
            self.counts.done.fetch_add(1, Ordering::Relaxed);
        }
        // No fsync: the copy is renamed into place when everything is done and
        // thrown away if anything fails, so a crash costs a run, not a world.
        store
            .flush_uniform()
            .map_err(|e| format!("writing cubes {}: {e}", self.level_dir.display()))
    }

    fn unreadable(&self, x: i32, z: i32) {
        log::warn!("Horizon: column {x} {z} could not be read, not converted");
        self.counts.unreadable.fetch_add(1, Ordering::Relaxed);
        let mut listed = self.unreadable.lock().unwrap();
        if listed.len() < LISTED_UNREADABLE * 4 {
            listed.push((x, z));
        }
    }

    fn column(&self, store: &CubeStore, x: i32, z: i32) -> Result<Column> {
        let Some(root) = self.save.raw_chunk(x, z) else {
            return Ok(Column::Unreadable);
        };
        if text(root.tag.get("Status")).is_some_and(|s| s != "minecraft:full" && s != "full") {
            return Ok(Column::Unfinished);
        }
        if let Some(Tag::List(sections)) = root.tag.get("sections") {
            for section in sections {
                let y = number(section.get("Y")).unwrap_or(0);
                // Light-only sections (Y -5 and 20) hold no blocks.
                if !VANILLA_SECTIONS.contains(&y) && section.get("block_states").is_some() {
                    return Err(format!(
                        "column {x} {z} has blocks at section {y}: this world is taller than \
                         vanilla's, and only worlds of vanilla height can be converted"
                    ));
                }
            }
        }
        let Some(mut chunk) = decode_with(
            &root,
            self.assets.registry,
            self.biomes,
            Some(self.tables),
            ColumnLayout::VANILLA,
        ) else {
            return Ok(Column::Unreadable);
        };
        chunk.entities.extend(self.save.load_entities(x, z));
        if !self.keep_bedrock {
            let before = count_blocks(&chunk, self.bedrock);
            replace_bedrock(&mut chunk, self.bedrock, self.deepslate);
            self.counts.bedrock.fetch_add(
                before - count_blocks(&chunk, self.bedrock),
                Ordering::Relaxed,
            );
        }
        self.counts
            .block_entities
            .fetch_add(chunk.block_entities.len() as u64, Ordering::Relaxed);
        let entities = split_by_cube(std::mem::take(&mut chunk.entities));
        self.counts.entities.fetch_add(
            entities.values().map(|e| e.len() as u64).sum(),
            Ordering::Relaxed,
        );
        let records = vanilla_column_records(
            self.assets.registry,
            self.biomes,
            Some(self.tables),
            &chunk,
            &entities,
        );
        for cube in records.cubes {
            let Some(root) = cube.root else {
                continue;
            };
            let pos = SectionPos::new(x, cube.cy, z);
            store_record(store, pos, &root)
                .map_err(|e| format!("cube {x} {} {z}: {e}", cube.cy))?;
            self.counts.cubes.fetch_add(1, Ordering::Relaxed);
            if crate::cubic::is_uniform_record(&root) {
                self.counts.uniform.fetch_add(1, Ordering::Relaxed);
            }
        }
        self.counts.converted.fetch_add(1, Ordering::Relaxed);
        Ok(Column::Converted)
    }
}

enum Column {
    Converted,
    Unreadable,
    Unfinished,
}

/// Blocks of state `state` in the bedrock floor (y -64..-60).
fn count_blocks(chunk: &crate::chunk::Chunk, state: u32) -> u64 {
    let mut n = 0;
    for y in chunk.min_y()..chunk.min_y() + 5 {
        for z in 0..16 {
            for x in 0..16 {
                n += u64::from(chunk.get(x, y, z) == state);
            }
        }
    }
    n
}

/// A data pack that gives the Overworld another height than vanilla's
/// (`dimension_type/overworld.json`: To The Max's 4,096), or the world
/// saying its Overworld has a dimension type of its own.
fn height_issue(world: &Path) -> Option<String> {
    let refuse = |what: &str| {
        Some(format!(
            "this world's Overworld is not vanilla's height ({what}); only worlds of vanilla \
             height (-64 to 320) can be converted"
        ))
    };
    if let Some(settings) = crate::save::world_gen_settings(world)
        && let Some(kind) = settings
            .get("dimensions")
            .and_then(|d| d.get(OVERWORLD))
            .and_then(|o| text(o.get("type")))
        && kind != OVERWORLD
    {
        return refuse(&format!("dimension type {kind}"));
    }
    let packs = std::fs::read_dir(world.join("datapacks")).ok()?;
    for pack in packs.flatten() {
        let path = pack.path();
        let json = if path.is_dir() {
            std::fs::read_to_string(path.join("data/minecraft/dimension_type/overworld.json")).ok()
        } else {
            read_zip_entry(&path, "data/minecraft/dimension_type/overworld.json")
        };
        let Some(json) = json else {
            continue;
        };
        let Ok(value) = serde_json::from_str::<serde_json::Value>(&json) else {
            continue;
        };
        let min_y = value["min_y"].as_i64().unwrap_or(-64);
        let height = value["height"].as_i64().unwrap_or(384);
        if (min_y, height) != (-64, 384) {
            return refuse(&format!(
                "the data pack {} sets {min_y} + {height}",
                pack.file_name().to_string_lossy()
            ));
        }
    }
    None
}

fn read_zip_entry(path: &Path, name: &str) -> Option<String> {
    use std::io::Read;
    let mut archive = zip::ZipArchive::new(std::fs::File::open(path).ok()?).ok()?;
    let mut entry = archive.by_name(name).ok()?;
    let mut text = String::new();
    entry.read_to_string(&mut text).ok()?;
    Some(text)
}

/// The CLI's `--convert-cubic NAME`: converts `saves/NAME` to
/// `saves/NAME (Cubic)`, printing its progress, and returns the process's
/// exit code.
pub fn run_cli(saves: &Path, name: &str, options: &ConvertOptions) -> i32 {
    let src = saves.join(name);
    let dst = saves.join(format!("{name} (Cubic)"));
    println!("Converting {} to {}", src.display(), dst.display());
    let mut last = Instant::now() - Duration::from_secs(1);
    let result = convert_with(&src, &dst, options, |p| {
        // Converting reports often; print a line a second.
        if matches!(p, Progress::Converting { .. } | Progress::Copying { .. })
            && last.elapsed() < Duration::from_secs(1)
        {
            return;
        }
        last = Instant::now();
        println!("{}", describe(&p));
    });
    match result {
        Ok(report) => {
            println!("{}", summary(&report));
            0
        }
        Err(e) => {
            eprintln!("Could not convert {name}: {e}");
            1
        }
    }
}

/// A progress line in plain English (the CLI's; the game's own line comes
/// from the language file).
pub fn describe(progress: &Progress) -> String {
    match progress {
        Progress::Checking => "Checking the world".into(),
        Progress::Copying { files, total } => format!("Copying files: {files} / {total}"),
        Progress::Upgrading => "Upgrading the world to this version".into(),
        Progress::Converting {
            columns,
            total,
            skipped,
            per_second,
        } => format!(
            "Converting columns: {columns} / {total} ({per_second:.0} per second, {skipped} skipped)"
        ),
        Progress::Finishing => "Finishing".into(),
    }
}

/// A finished conversion's report as text.
pub fn summary(report: &Report) -> String {
    let mut text = format!(
        "Converted {} columns ({} cubes, {} of them uniform) in {:.1} s, {:.0} columns per second.\n\
         Carried over: {} block entities, {} entities, {} files. Bedrock turned into deepslate: {}.",
        report.columns_converted,
        report.cubes_written,
        report.cubes_uniform,
        report.seconds,
        report.columns_per_second,
        report.block_entities,
        report.entities,
        report.files_copied,
        report.bedrock_replaced,
    );
    if report.columns_unfinished > 0 {
        text += &format!(
            "\nNot converted, never finished by the source (the game generates them): {}.",
            report.columns_unfinished
        );
    }
    if report.columns_unreadable > 0 {
        text += &format!(
            "\nNot converted, unreadable: {} (first: {:?}).",
            report.columns_unreadable, report.unreadable
        );
    }
    text
}

#[cfg(test)]
#[path = "cubic_convert_tests.rs"]
mod tests;
