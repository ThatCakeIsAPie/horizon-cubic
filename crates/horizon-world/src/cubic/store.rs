//! Where a cubic level keeps its cubes: `region3d/y<cube y>/r.<x>.<z>.mca`
//! under the level's folder, beside vanilla's `region/`. Each cube Y has its
//! own folder of vanilla region files, one slot per cube of 32 x 32 columns
//! (Cubic Chunks' layout). Reusing [`RegionFile`] keeps vanilla's sector
//! allocation, oversized records in their own files, and its write order
//! (a cube's sectors are written before the header points to them).

use super::has_cube;
use crate::compression::Compression;
use crate::nbt::{Limits, NamedTag};
use crate::region::{ChunkRecord, RegionFile};
use horizon_core::position::{ChunkPos, SectionPos};
use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// The folder of a cubic level's cubes, in the level's folder.
pub const FOLDER: &str = "region3d";
/// Region files kept open at once; past it the least recently used closes.
const OPEN_LIMIT: usize = 64;

/// A region file's key: its cube Y and region X, Z.
type RegionKey = (i32, i32, i32);

/// The cubes of one cubic level on disk. Cubes are NBT compounds (the
/// cube codec is the caller's); a missing cube reads as `None`.
pub struct CubeStore {
    root: PathBuf,
    compression: Compression,
    open: Mutex<Open>,
}

#[derive(Default)]
struct Open {
    files: HashMap<RegionKey, (RegionFile, u64)>,
    clock: u64,
}

impl CubeStore {
    /// The store of the level whose folder is `level_dir` (its cubes in
    /// `level_dir/region3d`). Nothing is created until a cube is written.
    pub fn new(level_dir: &Path) -> Self {
        Self {
            root: level_dir.join(FOLDER),
            compression: Compression::Zlib,
            open: Mutex::default(),
        }
    }

    /// The `region3d` folder.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The region file holding `cube`.
    pub fn region_path(&self, cube: SectionPos) -> PathBuf {
        let (y, x, z) = key(cube);
        self.root
            .join(format!("y{y}"))
            .join(format!("r.{x}.{z}.mca"))
    }

    /// Cube `cube`'s NBT, or `None` when it was never saved.
    pub fn read(&self, cube: SectionPos) -> io::Result<Option<NamedTag>> {
        check(cube)?;
        let Some(record) = self.with_region(cube, false, |region| region.read(slot(cube)))? else {
            return Ok(None);
        };
        record?.map(|r| r.nbt(Limits::saved())).transpose()
    }

    /// Saves `root` as cube `cube`, replacing what it had.
    pub fn write(&self, cube: SectionPos, root: &NamedTag) -> io::Result<()> {
        check(cube)?;
        let record = ChunkRecord::from_nbt(root, self.compression, timestamp(), Limits::saved())?;
        self.with_region(cube, true, |region| region.write(slot(cube), &record))?
            .unwrap_or(Ok(()))
    }

    /// Forgets cube `cube`; false when it was not saved.
    pub fn remove(&self, cube: SectionPos) -> io::Result<bool> {
        check(cube)?;
        Ok(self
            .with_region(cube, false, |region| region.clear(slot(cube), timestamp()))?
            .transpose()?
            .unwrap_or(false))
    }

    /// The cube Ys saved for column (x, z), lowest first. It looks in every
    /// cube Y's folder: for tools and tests, not for every column load.
    pub fn saved_ys(&self, x: i32, z: i32) -> io::Result<Vec<i32>> {
        let mut ys = Vec::new();
        let entries = match std::fs::read_dir(&self.root) {
            Ok(entries) => entries,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(ys),
            Err(e) => return Err(e),
        };
        for entry in entries {
            let name = entry?.file_name();
            let Some(y) = name
                .to_str()
                .and_then(|n| n.strip_prefix('y'))
                .and_then(|n| n.parse::<i32>().ok())
            else {
                continue;
            };
            let cube = SectionPos::new(x, y, z);
            if !has_cube(cube) {
                continue;
            }
            let present = self.with_region(cube, false, |region| {
                region.chunks().any(|(chunk, _)| chunk == slot(cube))
            })?;
            if present == Some(true) {
                ys.push(y);
            }
        }
        ys.sort_unstable();
        Ok(ys)
    }

    /// Flushes every open region file to disk (`RegionFile::sync`).
    pub fn sync(&self) -> io::Result<()> {
        let open = self.open.lock().unwrap_or_else(|e| e.into_inner());
        for (region, _) in open.files.values() {
            region.sync()?;
        }
        Ok(())
    }

    /// Runs `f` on `cube`'s region file, opening it (and with `create`,
    /// making it and its folder). `None` when the file is not there and
    /// `create` is false.
    fn with_region<T>(
        &self,
        cube: SectionPos,
        create: bool,
        f: impl FnOnce(&mut RegionFile) -> T,
    ) -> io::Result<Option<T>> {
        let key = key(cube);
        let mut open = self.open.lock().unwrap_or_else(|e| e.into_inner());
        open.clock += 1;
        let now = open.clock;
        if let Some((region, used)) = open.files.get_mut(&key) {
            *used = now;
            return Ok(Some(f(region)));
        }
        let path = self.region_path(cube);
        if create {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
        } else if !path.exists() {
            return Ok(None);
        }
        let mut region = RegionFile::open_writable(&path, create)?;
        let result = f(&mut region);
        if open.files.len() >= OPEN_LIMIT
            && let Some(oldest) = open
                .files
                .iter()
                .min_by_key(|(_, (_, used))| *used)
                .map(|(k, _)| *k)
        {
            open.files.remove(&oldest);
        }
        open.files.insert(key, (region, now));
        Ok(Some(result))
    }
}

fn key(cube: SectionPos) -> RegionKey {
    (cube.y, cube.x >> 5, cube.z >> 5)
}

/// The cube's slot in its region file, as the chunk position the region
/// file keys it by.
fn slot(cube: SectionPos) -> ChunkPos {
    ChunkPos::new(cube.x, cube.z)
}

fn check(cube: SectionPos) -> io::Result<()> {
    if has_cube(cube) {
        Ok(())
    } else {
        Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("cube {cube:?} is outside a cubic level"),
        ))
    }
}

fn timestamp() -> u32 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs() as u32)
}

#[cfg(test)]
mod tests {
    use super::super::{MAX_CUBE_Y, MIN_CUBE_Y};
    use super::*;
    use crate::nbt::Tag;

    /// A scratch level folder that removes itself.
    struct Scratch(PathBuf);
    impl Scratch {
        fn new(name: &str) -> Self {
            let dir = std::env::temp_dir().join(format!(
                "horizon-cubic-{name}-{}-{:?}",
                std::process::id(),
                std::thread::current().id()
            ));
            let _ = std::fs::remove_dir_all(&dir);
            Self(dir)
        }
    }
    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn cube_nbt(cube: SectionPos, filler: i32) -> NamedTag {
        NamedTag {
            name: "".into(),
            tag: Tag::compound([
                ("x", Tag::Int(cube.x)),
                ("y", Tag::Int(cube.y)),
                ("z", Tag::Int(cube.z)),
                ("filler", Tag::IntArray(vec![filler; 64])),
            ]),
        }
    }

    #[test]
    fn cubes_round_trip_across_regions_and_cube_ys() {
        let scratch = Scratch::new("round-trip");
        let store = CubeStore::new(&scratch.0);
        let cubes = [
            SectionPos::new(0, 0, 0),
            SectionPos::new(-1, -1, -1),
            SectionPos::new(31, 5, -32),
            SectionPos::new(32, 5, -33),
            SectionPos::new(-1_875_000, MIN_CUBE_Y, 1_874_999),
            SectionPos::new(7, MAX_CUBE_Y, 7),
        ];
        assert_eq!(store.read(cubes[0]).unwrap(), None);
        for (i, &cube) in cubes.iter().enumerate() {
            store.write(cube, &cube_nbt(cube, i as i32)).unwrap();
        }
        // A cube in the same column of another region file, and a neighbour slot.
        assert_eq!(store.read(SectionPos::new(0, 1, 0)).unwrap(), None);
        assert_eq!(store.read(SectionPos::new(1, 0, 0)).unwrap(), None);
        store.sync().unwrap();
        // A fresh store reads what the first wrote.
        let reopened = CubeStore::new(&scratch.0);
        for (i, &cube) in cubes.iter().enumerate() {
            assert_eq!(reopened.read(cube).unwrap(), Some(cube_nbt(cube, i as i32)));
        }
        assert!(
            store
                .region_path(cubes[4])
                .ends_with(format!("y{MIN_CUBE_Y}/r.-58594.58593.mca"))
        );
    }

    #[test]
    fn rewriting_and_removing_a_cube() {
        let scratch = Scratch::new("rewrite");
        let store = CubeStore::new(&scratch.0);
        let cube = SectionPos::new(3, -20, 4);
        store.write(cube, &cube_nbt(cube, 1)).unwrap();
        store.write(cube, &cube_nbt(cube, 2)).unwrap();
        assert_eq!(store.read(cube).unwrap(), Some(cube_nbt(cube, 2)));
        assert!(store.remove(cube).unwrap());
        assert!(!store.remove(cube).unwrap());
        assert_eq!(store.read(cube).unwrap(), None);
        // Removing from a region never written makes no file.
        let elsewhere = SectionPos::new(1000, 1000, 1000);
        assert!(!store.remove(elsewhere).unwrap());
        assert!(!store.region_path(elsewhere).exists());
    }

    #[test]
    fn a_columns_saved_cube_ys() {
        let scratch = Scratch::new("column");
        let store = CubeStore::new(&scratch.0);
        assert!(store.saved_ys(0, 0).unwrap().is_empty());
        for y in [12, -3, 400_000, 0] {
            let cube = SectionPos::new(-40, y, 9);
            store.write(cube, &cube_nbt(cube, y)).unwrap();
        }
        // Another column of the same region files.
        store
            .write(
                SectionPos::new(-39, 77, 9),
                &cube_nbt(SectionPos::new(-39, 77, 9), 0),
            )
            .unwrap();
        assert_eq!(store.saved_ys(-40, 9).unwrap(), [-3, 0, 12, 400_000]);
        assert_eq!(store.saved_ys(-39, 9).unwrap(), [77]);
        assert!(store.saved_ys(-38, 9).unwrap().is_empty());
    }

    #[test]
    fn cubes_outside_the_level_are_refused() {
        let scratch = Scratch::new("outside");
        let store = CubeStore::new(&scratch.0);
        let cube = SectionPos::new(0, MAX_CUBE_Y + 1, 0);
        let error = store.write(cube, &cube_nbt(cube, 0)).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        assert!(store.read(SectionPos::new(0, MIN_CUBE_Y - 1, 0)).is_err());
        assert!(!scratch.0.exists());
    }

    #[test]
    fn more_regions_than_stay_open() {
        let scratch = Scratch::new("many");
        let store = CubeStore::new(&scratch.0);
        let cubes: Vec<_> = (0..OPEN_LIMIT as i32 + 10)
            .map(|i| SectionPos::new(i * 32, i, -i * 32))
            .collect();
        for (i, &cube) in cubes.iter().enumerate() {
            store.write(cube, &cube_nbt(cube, i as i32)).unwrap();
        }
        assert!(store.open.lock().unwrap().files.len() <= OPEN_LIMIT);
        for (i, &cube) in cubes.iter().enumerate() {
            assert_eq!(store.read(cube).unwrap(), Some(cube_nbt(cube, i as i32)));
        }
    }
}
