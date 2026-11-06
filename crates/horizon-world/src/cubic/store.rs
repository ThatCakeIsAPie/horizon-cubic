//! Where a cubic level keeps its cubes: `region3d/y<cube y>/r.<x>.<z>.mca`
//! under the level's folder, beside vanilla's `region/`. Each cube Y has its
//! own folder of vanilla region files, one slot per cube of 32 x 32 columns
//! (Cubic Chunks' layout). Reusing [`RegionFile`] keeps vanilla's sector
//! allocation, oversized records in their own files, and its write order
//! (a cube's sectors are written before the header points to them).
//!
//! A cube that is one block state throughout, with nothing else in it
//! (no block entities, entities or ticks: the caller says so with
//! [`CubeStore::write_uniform`]) would take a whole 4 KiB sector for ~100
//! bytes. Those go to a per-region sidecar instead, `y<cy>/u.<x>.<z>.dat`:
//! a table of the distinct record templates (the record without its
//! `xPos`/`zPos`/`yPos`) and, per slot, the index of its template. A
//! thousand cubes of deepslate cost about 6 bytes each.
//!
//! The sidecar is kept in memory and written whole to a temporary file that
//! is renamed over it (so a crash leaves the old file or the new, never a
//! torn one; a leftover temporary is ignored). It is persisted when 128
//! changes have gathered, by [`CubeStore::flush_uniform`] and
//! [`CubeStore::sync`], and when the store drops. Ordering keeps a cube
//! from being lost:
//! - a uniform write clears the cube's region slot only after the sidecar
//!   holding it is on disk, so a crash before leaves the older record;
//! - a record write over a uniform entry persists the entry's removal first;
//! - a cube found in both reads from the sidecar, which is only ever the
//!   newer of the two.
//!
//! Worlds saved before the sidecar existed have region files only, and read
//! as they did.

use super::has_cube;
use crate::compression::Compression;
use crate::nbt::{Limits, NamedTag, Tag};
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
/// Uniform sidecars kept in memory at once; past it clean ones are dropped.
const UNIFORM_LIMIT: usize = 256;
/// Changes to a sidecar that make `write_uniform` persist it.
const PERSIST_AFTER: u32 = 128;
const UNIFORM_MAGIC: &[u8; 4] = b"HZU1";

/// A region file's key: its cube Y and region X, Z.
type RegionKey = (i32, i32, i32);

/// The cubes of one cubic level on disk. Cubes are NBT compounds (the
/// cube codec is the caller's); a missing cube reads as `None`.
pub struct CubeStore {
    root: PathBuf,
    compression: Compression,
    open: Mutex<Open>,
    uniform: Mutex<HashMap<RegionKey, UniformRegion>>,
}

#[derive(Default)]
struct Open {
    files: HashMap<RegionKey, (RegionFile, u64)>,
    clock: u64,
}

/// One region's uniform cubes.
#[derive(Default)]
struct UniformRegion {
    /// Region slot (`x & 31 | (z & 31) << 5`) to its template's index.
    entries: HashMap<u16, u32>,
    /// Templates: serialized records without their coordinates.
    templates: Vec<Vec<u8>>,
    /// Changes not on disk yet.
    dirty: u32,
    /// Region-file slots to clear once the sidecar is on disk.
    clear: Vec<u16>,
}

impl CubeStore {
    /// The store of the level whose folder is `level_dir` (its cubes in
    /// `level_dir/region3d`). Nothing is created until a cube is written.
    pub fn new(level_dir: &Path) -> Self {
        Self {
            root: level_dir.join(FOLDER),
            compression: Compression::Zlib,
            open: Mutex::default(),
            uniform: Mutex::default(),
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

    /// The uniform sidecar that holds `cube` when it is uniform.
    pub fn uniform_path(&self, cube: SectionPos) -> PathBuf {
        let (y, x, z) = key(cube);
        self.root
            .join(format!("y{y}"))
            .join(format!("u.{x}.{z}.dat"))
    }

    /// Cube `cube`'s NBT, or `None` when it was never saved.
    pub fn read(&self, cube: SectionPos) -> io::Result<Option<NamedTag>> {
        check(cube)?;
        if let Some(root) = self.read_uniform(cube)? {
            return Ok(Some(root));
        }
        let Some(record) = self.with_region(cube, false, |region| region.read(slot(cube)))? else {
            return Ok(None);
        };
        record?.map(|r| r.nbt(Limits::saved())).transpose()
    }

    /// Saves `root` as cube `cube` in its region file, replacing what it had
    /// (a uniform entry too).
    pub fn write(&self, cube: SectionPos, root: &NamedTag) -> io::Result<()> {
        check(cube)?;
        self.drop_uniform(cube)?;
        let record = ChunkRecord::from_nbt(root, self.compression, timestamp(), Limits::saved())?;
        self.with_region(cube, true, |region| region.write(slot(cube), &record))?
            .unwrap_or(Ok(()))
    }

    /// Saves `root` as cube `cube` in the region's uniform sidecar, replacing
    /// what it had: for a cube with one block state and no block entities,
    /// entities or ticks, whose record is the same as any other uniform
    /// cube's but for its position (`xPos`, `zPos`, `yPos`, which must be
    /// the cube's). A record that does not fit that is refused with
    /// `InvalidInput`; save it with [`CubeStore::write`].
    pub fn write_uniform(&self, cube: SectionPos, root: &NamedTag) -> io::Result<()> {
        check(cube)?;
        let template = template_of(cube, root)?;
        let region_key = key(cube);
        let mut all = self.uniform.lock().unwrap_or_else(|e| e.into_inner());
        let region = self.uniform_region(&mut all, cube)?;
        let index = match region.templates.iter().position(|t| *t == template) {
            Some(i) => i as u32,
            None => {
                region.templates.push(template);
                region.templates.len() as u32 - 1
            }
        };
        let bit = slot_bit(cube);
        if region.entries.insert(bit, index) != Some(index) {
            region.dirty += 1;
        }
        // The cube's region slot is cleared once this is on disk.
        if !region.clear.contains(&bit) {
            region.clear.push(bit);
        }
        if region.dirty >= PERSIST_AFTER {
            self.persist(region_key, region)?;
        }
        Ok(())
    }

    /// Persists every uniform sidecar with changes not on disk.
    pub fn flush_uniform(&self) -> io::Result<()> {
        let mut all = self.uniform.lock().unwrap_or_else(|e| e.into_inner());
        let mut result = Ok(());
        for (k, region) in all.iter_mut() {
            if let Err(e) = self.persist(*k, region) {
                result = Err(e);
            }
        }
        result
    }

    /// Forgets cube `cube`; false when it was not saved.
    pub fn remove(&self, cube: SectionPos) -> io::Result<bool> {
        check(cube)?;
        let was_uniform = self.drop_uniform(cube)?;
        let was_record = self
            .with_region(cube, false, |region| region.clear(slot(cube), timestamp()))?
            .transpose()?
            .unwrap_or(false);
        Ok(was_uniform || was_record)
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
            let present = self.has_uniform(cube)?
                || self.with_region(cube, false, |region| {
                    region.chunks().any(|(chunk, _)| chunk == slot(cube))
                })? == Some(true);
            if present {
                ys.push(y);
            }
        }
        ys.sort_unstable();
        Ok(ys)
    }

    /// Flushes every open region file to disk (`RegionFile::sync`), the
    /// uniform sidecars first.
    pub fn sync(&self) -> io::Result<()> {
        self.flush_uniform()?;
        let open = self.open.lock().unwrap_or_else(|e| e.into_inner());
        for (region, _) in open.files.values() {
            region.sync()?;
        }
        Ok(())
    }

    /// The uniform entry of `cube`, as a record.
    fn read_uniform(&self, cube: SectionPos) -> io::Result<Option<NamedTag>> {
        let mut all = self.uniform.lock().unwrap_or_else(|e| e.into_inner());
        let region = self.uniform_region(&mut all, cube)?;
        let Some(&index) = region.entries.get(&slot_bit(cube)) else {
            return Ok(None);
        };
        let Some(bytes) = region.templates.get(index as usize) else {
            return Ok(None);
        };
        let (mut root, _) = crate::nbt::read_named(bytes, Limits::saved()).map_err(invalid)?;
        if let Tag::Compound(map) = &mut root.tag {
            map.insert("xPos".into(), Tag::Int(cube.x));
            map.insert("zPos".into(), Tag::Int(cube.z));
            map.insert("yPos".into(), Tag::Int(cube.y));
        }
        Ok(Some(root))
    }

    fn has_uniform(&self, cube: SectionPos) -> io::Result<bool> {
        let mut all = self.uniform.lock().unwrap_or_else(|e| e.into_inner());
        let region = self.uniform_region(&mut all, cube)?;
        Ok(region.entries.contains_key(&slot_bit(cube)))
    }

    /// Forgets `cube`'s uniform entry, persisting that at once (the record
    /// about to be written must not meet a stale entry after a crash).
    fn drop_uniform(&self, cube: SectionPos) -> io::Result<bool> {
        let mut all = self.uniform.lock().unwrap_or_else(|e| e.into_inner());
        let region = self.uniform_region(&mut all, cube)?;
        let bit = slot_bit(cube);
        region.clear.retain(|b| *b != bit);
        if region.entries.remove(&bit).is_none() {
            return Ok(false);
        }
        region.dirty += 1;
        self.persist(key(cube), region)?;
        Ok(true)
    }

    /// `cube`'s region of the uniform sidecars, loaded from its file (a
    /// missing or damaged file is an empty region).
    fn uniform_region<'a>(
        &self,
        all: &'a mut HashMap<RegionKey, UniformRegion>,
        cube: SectionPos,
    ) -> io::Result<&'a mut UniformRegion> {
        let k = key(cube);
        if !all.contains_key(&k) {
            if all.len() >= UNIFORM_LIMIT {
                all.retain(|_, r| r.dirty > 0 || !r.clear.is_empty());
            }
            let region = match std::fs::read(self.uniform_path(cube)) {
                Ok(bytes) => parse_uniform(&bytes).unwrap_or_else(|| {
                    log::warn!("Horizon: uniform cube file of {cube:?} is damaged; ignored");
                    UniformRegion::default()
                }),
                Err(e) if e.kind() == io::ErrorKind::NotFound => UniformRegion::default(),
                Err(e) => return Err(e),
            };
            all.insert(k, region);
        }
        Ok(all.get_mut(&k).expect("just inserted"))
    }

    /// Writes the region's sidecar (temporary file, then rename), then
    /// clears the region-file slots its new entries replace.
    fn persist(&self, k: RegionKey, region: &mut UniformRegion) -> io::Result<()> {
        if region.dirty == 0 && region.clear.is_empty() {
            return Ok(());
        }
        let (y, rx, rz) = k;
        let path = self
            .root
            .join(format!("y{y}"))
            .join(format!("u.{rx}.{rz}.dat"));
        // Only the templates still used, in order.
        let mut used: Vec<u32> = region.entries.values().copied().collect();
        used.sort_unstable();
        used.dedup();
        let remap: HashMap<u32, u32> = used
            .iter()
            .enumerate()
            .map(|(new, &old)| (old, new as u32))
            .collect();
        let templates: Vec<Vec<u8>> = used
            .iter()
            .map(|&i| std::mem::take(&mut region.templates[i as usize]))
            .collect();
        for entry in region.entries.values_mut() {
            *entry = remap[&*entry];
        }
        region.templates = templates;
        if region.entries.is_empty() {
            match std::fs::remove_file(&path) {
                Ok(()) => {}
                Err(e) if e.kind() == io::ErrorKind::NotFound => {}
                Err(e) => return Err(e),
            }
        } else {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let temporary = path.with_extension("dat.tmp");
            std::fs::write(&temporary, serialize_uniform(region))?;
            std::fs::rename(&temporary, &path)?;
        }
        region.dirty = 0;
        for bit in std::mem::take(&mut region.clear) {
            let cube = SectionPos::new(
                rx * 32 + i32::from(bit & 31),
                y,
                rz * 32 + i32::from(bit >> 5),
            );
            self.with_region(cube, false, |region| region.clear(slot(cube), timestamp()))?
                .transpose()?;
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

impl Drop for CubeStore {
    fn drop(&mut self) {
        if let Err(e) = self.flush_uniform() {
            log::warn!("Horizon: uniform cubes not saved: {e}");
        }
    }
}

fn invalid(e: impl std::fmt::Display) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, e.to_string())
}

/// A cube's slot in its region's sidecar.
fn slot_bit(cube: SectionPos) -> u16 {
    ((cube.x & 31) | (cube.z & 31) << 5) as u16
}

/// `root` without its position, serialized, for the sidecar's template table.
fn template_of(cube: SectionPos, root: &NamedTag) -> io::Result<Vec<u8>> {
    let Tag::Compound(map) = &root.tag else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "not a compound",
        ));
    };
    let mut map = map.clone();
    for (name, want) in [("xPos", cube.x), ("zPos", cube.z), ("yPos", cube.y)] {
        if map.remove(&name.into()) != Some(Tag::Int(want)) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("{name} is not the cube's"),
            ));
        }
    }
    let template = NamedTag {
        name: root.name.clone(),
        tag: Tag::Compound(map),
    };
    crate::nbt::write_named(&template, Limits::saved()).map_err(invalid)
}

fn checksum(bytes: &[u8]) -> u64 {
    let mut hash = 0xCBF2_9CE4_8422_2325u64;
    for &b in bytes {
        hash = (hash ^ u64::from(b)).wrapping_mul(0x0100_0000_01B3);
    }
    hash
}

fn serialize_uniform(region: &UniformRegion) -> Vec<u8> {
    let mut out = UNIFORM_MAGIC.to_vec();
    out.extend((region.templates.len() as u32).to_be_bytes());
    for t in &region.templates {
        out.extend((t.len() as u32).to_be_bytes());
        out.extend(t);
    }
    out.extend((region.entries.len() as u32).to_be_bytes());
    let mut entries: Vec<_> = region.entries.iter().collect();
    entries.sort_unstable();
    for (bit, index) in entries {
        out.extend(bit.to_be_bytes());
        out.extend(index.to_be_bytes());
    }
    let sum = checksum(&out);
    out.extend(sum.to_be_bytes());
    out
}

fn parse_uniform(bytes: &[u8]) -> Option<UniformRegion> {
    let (body, sum) = bytes.split_at_checked(bytes.len().checked_sub(8)?)?;
    if checksum(body) != u64::from_be_bytes(sum.try_into().ok()?) {
        return None;
    }
    let mut at = body.strip_prefix(UNIFORM_MAGIC.as_slice())?;
    let mut take = |n: usize| -> Option<&[u8]> {
        let (head, rest) = at.split_at_checked(n)?;
        at = rest;
        Some(head)
    };
    let word = |b: &[u8]| u32::from_be_bytes(b.try_into().unwrap());
    let mut region = UniformRegion::default();
    for _ in 0..word(take(4)?) {
        let len = word(take(4)?) as usize;
        region.templates.push(take(len)?.to_vec());
    }
    for _ in 0..word(take(4)?) {
        let bit = u16::from_be_bytes(take(2)?.try_into().ok()?);
        let index = word(take(4)?);
        if bit >= 1024 || index as usize >= region.templates.len() {
            return None;
        }
        region.entries.insert(bit, index);
    }
    Some(region)
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

    /// A uniform record of block `block`, positioned like a cube record.
    fn uniform_nbt(cube: SectionPos, block: &str) -> NamedTag {
        NamedTag {
            name: "".into(),
            tag: Tag::compound([
                ("xPos", Tag::Int(cube.x)),
                ("zPos", Tag::Int(cube.z)),
                ("yPos", Tag::Int(cube.y)),
                ("Status", Tag::String("minecraft:full".into())),
                (
                    "sections",
                    Tag::List(vec![Tag::compound([
                        ("Y", Tag::Int(cube.y)),
                        (
                            "block_states",
                            Tag::compound([(
                                "palette",
                                Tag::List(vec![Tag::compound([(
                                    "Name",
                                    Tag::String(block.into()),
                                )])]),
                            )]),
                        ),
                    ])]),
                ),
            ]),
        }
    }

    /// The bytes of the files under `dir`.
    fn disk_bytes(dir: &Path) -> u64 {
        let mut total = 0;
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let meta = entry.metadata().unwrap();
                total += if meta.is_dir() {
                    disk_bytes(&entry.path())
                } else {
                    meta.len()
                };
            }
        }
        total
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
        // Uniform cubes count too.
        let flat = SectionPos::new(-40, 7, 9);
        store
            .write_uniform(flat, &uniform_nbt(flat, "minecraft:stone"))
            .unwrap();
        // Another column of the same region files.
        store
            .write(
                SectionPos::new(-39, 77, 9),
                &cube_nbt(SectionPos::new(-39, 77, 9), 0),
            )
            .unwrap();
        assert_eq!(store.saved_ys(-40, 9).unwrap(), [-3, 0, 7, 12, 400_000]);
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

    #[test]
    fn uniform_cubes_round_trip_and_reopen() {
        let scratch = Scratch::new("uniform");
        let store = CubeStore::new(&scratch.0);
        let a = SectionPos::new(5, -30, 6);
        let b = SectionPos::new(-1, -30, 31);
        store
            .write_uniform(a, &uniform_nbt(a, "minecraft:deepslate"))
            .unwrap();
        store
            .write_uniform(b, &uniform_nbt(b, "minecraft:air"))
            .unwrap();
        assert_eq!(
            store.read(a).unwrap(),
            Some(uniform_nbt(a, "minecraft:deepslate"))
        );
        store.sync().unwrap();
        drop(store);
        let reopened = CubeStore::new(&scratch.0);
        assert_eq!(
            reopened.read(a).unwrap(),
            Some(uniform_nbt(a, "minecraft:deepslate"))
        );
        assert_eq!(
            reopened.read(b).unwrap(),
            Some(uniform_nbt(b, "minecraft:air"))
        );
        assert_eq!(reopened.read(SectionPos::new(6, -30, 6)).unwrap(), None);
        // No region file was made for them.
        assert!(!reopened.region_path(a).exists());
    }

    #[test]
    fn a_cube_goes_uniform_to_full_to_uniform() {
        let scratch = Scratch::new("overwrite");
        let store = CubeStore::new(&scratch.0);
        let cube = SectionPos::new(1, 2, 3);
        store
            .write_uniform(cube, &uniform_nbt(cube, "minecraft:stone"))
            .unwrap();
        store.write(cube, &cube_nbt(cube, 9)).unwrap();
        assert_eq!(store.read(cube).unwrap(), Some(cube_nbt(cube, 9)));
        // A fresh store agrees: the uniform entry is gone from the disk.
        assert_eq!(
            CubeStore::new(&scratch.0).read(cube).unwrap(),
            Some(cube_nbt(cube, 9))
        );
        store
            .write_uniform(cube, &uniform_nbt(cube, "minecraft:air"))
            .unwrap();
        assert_eq!(
            store.read(cube).unwrap(),
            Some(uniform_nbt(cube, "minecraft:air"))
        );
        store.sync().unwrap();
        // The record is cleared once the sidecar is on disk.
        let fresh = CubeStore::new(&scratch.0);
        assert_eq!(
            fresh.read(cube).unwrap(),
            Some(uniform_nbt(cube, "minecraft:air"))
        );
        let in_region = fresh
            .with_region(cube, false, |r| r.chunks().count())
            .unwrap();
        assert_eq!(in_region, Some(0));
        assert!(fresh.remove(cube).unwrap());
        assert_eq!(fresh.read(cube).unwrap(), None);
        assert!(fresh.saved_ys(1, 3).unwrap().is_empty());
    }

    #[test]
    fn a_crash_between_sidecar_and_region_keeps_the_newer_uniform() {
        let scratch = Scratch::new("both");
        let cube = SectionPos::new(2, 2, 2);
        {
            let store = CubeStore::new(&scratch.0);
            store.write(cube, &cube_nbt(cube, 1)).unwrap();
            // A uniform entry persisted, the slot not cleared (a crash
            // between the two): write the sidecar by hand.
            let mut region = UniformRegion::default();
            region
                .templates
                .push(template_of(cube, &uniform_nbt(cube, "minecraft:stone")).unwrap());
            region.entries.insert(slot_bit(cube), 0);
            std::fs::write(store.uniform_path(cube), serialize_uniform(&region)).unwrap();
        }
        let store = CubeStore::new(&scratch.0);
        assert_eq!(
            store.read(cube).unwrap(),
            Some(uniform_nbt(cube, "minecraft:stone"))
        );
    }

    #[test]
    fn a_leftover_temporary_and_a_damaged_sidecar_are_ignored() {
        let scratch = Scratch::new("crash");
        let cube = SectionPos::new(0, 0, 0);
        {
            let store = CubeStore::new(&scratch.0);
            store
                .write_uniform(cube, &uniform_nbt(cube, "minecraft:stone"))
                .unwrap();
            store.sync().unwrap();
            // A crash while writing the next version leaves a temporary.
            std::fs::write(store.uniform_path(cube).with_extension("dat.tmp"), b"torn").unwrap();
        }
        let store = CubeStore::new(&scratch.0);
        assert_eq!(
            store.read(cube).unwrap(),
            Some(uniform_nbt(cube, "minecraft:stone"))
        );
        // A truncated sidecar reads as empty rather than failing.
        let path = store.uniform_path(cube);
        let bytes = std::fs::read(&path).unwrap();
        std::fs::write(&path, &bytes[..bytes.len() - 3]).unwrap();
        let damaged = CubeStore::new(&scratch.0);
        assert_eq!(damaged.read(cube).unwrap(), None);
    }

    #[test]
    fn a_uniform_record_with_another_position_is_refused() {
        let scratch = Scratch::new("refused");
        let store = CubeStore::new(&scratch.0);
        let cube = SectionPos::new(0, 0, 0);
        let other = SectionPos::new(1, 0, 0);
        let error = store
            .write_uniform(cube, &uniform_nbt(other, "minecraft:stone"))
            .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn a_thousand_uniform_cubes_cost_bytes_not_sectors() {
        let scratch = Scratch::new("bytes");
        let store = CubeStore::new(&scratch.0);
        for i in 0..1000 {
            let cube = SectionPos::new(i % 32, 4 + i / 1024, i / 32);
            store
                .write_uniform(cube, &uniform_nbt(cube, "minecraft:deepslate"))
                .unwrap();
        }
        store.sync().unwrap();
        let bytes = disk_bytes(store.root());
        assert!(bytes < 1000 * 4096 / 100, "{bytes} bytes");
        for i in [0, 499, 999] {
            let cube = SectionPos::new(i % 32, 4 + i / 1024, i / 32);
            assert_eq!(
                store.read(cube).unwrap(),
                Some(uniform_nbt(cube, "minecraft:deepslate"))
            );
        }
    }
}
