//! Cube far terrain kept on disk, beside a dimension's save
//! (`<dimension>/horizon_cube_lod/`), so a world opened again shows what was
//! seen before: the player's edits and vanilla's decorated terrain built from
//! saved cubes, and the coarser levels made of them.
//!
//! One append-only file per bucket of 8 x 8 x 8 nodes of one level. A record is
//! `[u32 length][kind u8][level u8][x i32][y i32][z i32][body][u32 checksum]`
//! (`length` counts everything after itself): kind 0 stores a node, kind 1
//! erases it. A record cut short by a crash ends the file there when it is
//! opened; a damaged body fails its checksum and reads as missing. A file
//! mostly made of dead records is rewritten on open. The folder holds a
//! fingerprint of the format, the build, the seed and the block registry; any
//! change empties it.
//!
//! Only levels 1 to [`MAX_CACHED_LEVEL`] are stored: a level 0 node is the saved
//! cube itself. When a cube is saved again the owner of the save calls
//! [`CubeLodCache::invalidate`], which erases every stored node over it.
use super::cube::{Changes, CoverMask, CubeKey, CubeNode, MAX_CUBE_LEVEL, VOLUME, Voxel};
use crate::fast_hash::FxHashMap;
use flate2::{Compression, read::ZlibDecoder, write::ZlibEncoder};
use horizon_core::position::SectionPos;
use horizon_world::registry::BlockStateId;
use std::{
    collections::VecDeque,
    fs,
    io::{BufReader, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

/// The coarsest level kept on disk (nodes 4,096 blocks wide).
pub const MAX_CACHED_LEVEL: u8 = 8;
const FORMAT: u32 = 1;
const FINGERPRINT: &str = "fingerprint.txt";
/// Nodes along one side of a bucket file.
const BUCKET: i32 = 8;
/// Kind, level and the three coordinates, after the length.
const HEAD: usize = 14;
const TAIL: usize = 4;
const NODE: u8 = 0;
const ERASE: u8 = 1;
/// Cubes remembered for [`CubeLodCache::changes_since`] before it says "all".
const LOG: usize = 4096;
/// A bucket file this large with mostly dead records is rewritten on open.
const COMPACT_ABOVE: u64 = 1 << 20;
/// A node's raw voxels: a state and a biome each.
const RAW: usize = VOLUME * 6;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct BucketId(u8, i32, i32, i32);

impl BucketId {
    fn of(key: CubeKey) -> Self {
        Self(
            key.level,
            key.x.div_euclid(BUCKET),
            key.y.div_euclid(BUCKET),
            key.z.div_euclid(BUCKET),
        )
    }
    fn file(self) -> String {
        format!("n{}.{}.{}.{}.bin", self.0, self.1, self.2, self.3)
    }
}

fn checksum(bytes: &[u8]) -> u32 {
    // FNV-1a.
    bytes.iter().fold(0x811c_9dc5_u32, |h, &b| {
        (h ^ u32::from(b)).wrapping_mul(0x0100_0193)
    })
}

struct Bucket {
    file: fs::File,
    /// Each key's latest record: where its content (after the length) starts
    /// and how long it is. `None`: erased.
    index: FxHashMap<CubeKey, Option<(u64, u32)>>,
    end: u64,
}

impl Bucket {
    fn open(path: &Path) -> std::io::Result<Self> {
        let mut bucket = Self::scan(path)?;
        let live: u64 = bucket
            .index
            .values()
            .flatten()
            .map(|&(_, len)| 4 + u64::from(len))
            .sum();
        if bucket.end > COMPACT_ABOVE && live * 2 < bucket.end {
            // Rewritten beside it and swapped in with the file closed.
            let records = bucket.live_records()?;
            drop(bucket);
            let temporary = path.with_extension("tmp");
            fs::write(&temporary, &records)?;
            fs::rename(&temporary, path)?;
            return Self::scan(path);
        }
        Ok(bucket)
    }

    fn scan(path: &Path) -> std::io::Result<Self> {
        let mut file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)?;
        let len = file.metadata()?.len();
        let mut index = FxHashMap::default();
        let mut end = 0u64;
        {
            let mut reader = BufReader::new(&mut file);
            loop {
                let mut length = [0; 4];
                if reader.read_exact(&mut length).is_err() {
                    break;
                }
                let length = u32::from_le_bytes(length);
                let mut head = [0; HEAD];
                if (length as usize) < HEAD + TAIL || reader.read_exact(&mut head).is_err() {
                    break;
                }
                let content = end + 4;
                // A record cut short by a crash ends the file there.
                if content + u64::from(length) > len
                    || reader
                        .seek_relative(i64::from(length) - HEAD as i64)
                        .is_err()
                {
                    break;
                }
                let int = |at: usize| i32::from_le_bytes(head[at..at + 4].try_into().unwrap());
                let key = CubeKey::new(head[1], int(2), int(6), int(10));
                match head[0] {
                    NODE => index.insert(key, Some((content, length))),
                    ERASE => index.insert(key, None),
                    _ => break,
                };
                end = content + u64::from(length);
            }
        }
        if len != end {
            file.set_len(end)?;
        }
        Ok(Self { file, index, end })
    }

    /// The file's live records only, as a new file's bytes.
    fn live_records(&mut self) -> std::io::Result<Vec<u8>> {
        let mut keys: Vec<_> = self
            .index
            .iter()
            .filter_map(|(&key, at)| at.map(|at| (at.0, key, at.1)))
            .collect();
        keys.sort_unstable();
        let mut out = Vec::new();
        for (offset, _, length) in keys {
            let mut content = vec![0; length as usize];
            self.file.seek(SeekFrom::Start(offset))?;
            self.file.read_exact(&mut content)?;
            out.extend_from_slice(&length.to_le_bytes());
            out.extend_from_slice(&content);
        }
        Ok(out)
    }

    /// The body of `key`'s latest record, if it is intact.
    fn read(&mut self, key: CubeKey) -> Option<Vec<u8>> {
        let (offset, length) = (*self.index.get(&key)?)?;
        let mut content = vec![0; length as usize];
        self.file.seek(SeekFrom::Start(offset)).ok()?;
        self.file.read_exact(&mut content).ok()?;
        let (rest, tail) = content.split_at(content.len() - TAIL);
        if checksum(rest) != u32::from_le_bytes(tail.try_into().ok()?) {
            return None;
        }
        Some(rest[HEAD..].to_vec())
    }

    fn append(&mut self, kind: u8, key: CubeKey, body: &[u8]) -> std::io::Result<()> {
        let mut record = Vec::with_capacity(4 + HEAD + body.len() + TAIL);
        record.extend_from_slice(&[0; 4]);
        record.push(kind);
        record.push(key.level);
        for c in [key.x, key.y, key.z] {
            record.extend_from_slice(&c.to_le_bytes());
        }
        record.extend_from_slice(body);
        let sum = checksum(&record[4..]);
        record.extend_from_slice(&sum.to_le_bytes());
        let length = (record.len() - 4) as u32;
        record[..4].copy_from_slice(&length.to_le_bytes());
        self.file.seek(SeekFrom::Start(self.end))?;
        // One write, so a crash leaves at most a torn last record.
        if let Err(e) = self.file.write_all(&record) {
            let _ = self.file.set_len(self.end);
            return Err(e);
        }
        self.index
            .insert(key, (kind == NODE).then_some((self.end + 4, length)));
        self.end += record.len() as u64;
        Ok(())
    }
}

#[derive(Default)]
struct ChangeLog {
    /// Sequence number of the next change.
    next: u64,
    entries: VecDeque<(u64, SectionPos)>,
}

/// A dimension's cube far terrain on disk, shared by the sampling threads and
/// by the save that invalidates it.
pub struct CubeLodCache {
    dir: PathBuf,
    /// Open buckets; `None`: the file does not exist (or did not open).
    buckets: Mutex<FxHashMap<BucketId, Option<Bucket>>>,
    changes: Mutex<ChangeLog>,
}

impl CubeLodCache {
    /// Opens `dir`, emptying it when its fingerprint is not `fingerprint`
    /// (see [`fingerprint`]). `None` (no caching) on an error.
    pub fn open(dir: &Path, fingerprint: &str) -> Option<Arc<Self>> {
        let current = fs::read_to_string(dir.join(FINGERPRINT)).ok();
        if current.as_deref() != Some(fingerprint) {
            if current.is_some() {
                log::info!("Horizon: cube far terrain cache is out of date; building it again");
            }
            let reset = || -> std::io::Result<()> {
                if dir.exists() {
                    fs::remove_dir_all(dir)?;
                }
                fs::create_dir_all(dir)?;
                fs::write(dir.join(FINGERPRINT), fingerprint)
            };
            if let Err(e) = reset() {
                log::warn!(
                    "Horizon: cube far terrain will not be kept ({}): {e}",
                    dir.display()
                );
                return None;
            }
        }
        Some(Arc::new(Self {
            dir: dir.to_path_buf(),
            buckets: Mutex::default(),
            changes: Mutex::default(),
        }))
    }

    fn with_bucket<T>(
        &self,
        key: CubeKey,
        create: bool,
        f: impl FnOnce(&mut Bucket) -> Option<T>,
    ) -> Option<T> {
        let id = BucketId::of(key);
        let mut buckets = self.buckets.lock().ok()?;
        let open = match buckets.get(&id) {
            Some(Some(_)) => false,
            Some(None) => create,
            None => true,
        };
        if open {
            let path = self.dir.join(id.file());
            let bucket = if create || path.exists() {
                match Bucket::open(&path) {
                    Ok(bucket) => Some(bucket),
                    Err(e) => {
                        log::warn!("Horizon: cube far terrain {}: {e}", path.display());
                        None
                    }
                }
            } else {
                None
            };
            buckets.insert(id, bucket);
        }
        f(buckets.get_mut(&id)?.as_mut()?)
    }

    /// Whether `key` is stored.
    pub fn contains(&self, key: CubeKey) -> bool {
        key.level <= MAX_CACHED_LEVEL
            && self
                .with_bucket(key, false, |b| {
                    Some(matches!(b.index.get(&key), Some(Some(_))))
                })
                .unwrap_or(false)
    }

    /// The stored node `key`, if there is one and it is intact.
    pub fn get(&self, key: CubeKey) -> Option<CubeNode> {
        if key.level == 0 || key.level > MAX_CACHED_LEVEL {
            return None;
        }
        let body = self.with_bucket(key, false, |b| b.read(key))?;
        decode(key, &body)
    }

    /// Stores `node` (levels 1 to [`MAX_CACHED_LEVEL`]), replacing any older.
    pub fn put(&self, node: &CubeNode) {
        let key = node.key;
        if key.level == 0 || key.level > MAX_CACHED_LEVEL {
            return;
        }
        let body = encode(node);
        self.with_bucket(key, true, |b| {
            if let Err(e) = b.append(NODE, key, &body) {
                log::warn!("Horizon: keeping cube far terrain: {e}");
            }
            Some(())
        });
    }

    /// Cube `cube` was saved again (or for the first time): erases every
    /// stored node over it, and tells the samplers to drop theirs
    /// ([`Self::changes_since`]). Cheap: at most one lookup per level, and an
    /// appended record only for a node that is stored. The save calls this
    /// from any thread, when it queues the cube's write.
    pub fn invalidate(&self, cube: SectionPos) {
        for level in 1..=MAX_CACHED_LEVEL.min(MAX_CUBE_LEVEL) {
            let key = CubeKey::containing(level, cube);
            self.with_bucket(key, false, |b| {
                if matches!(b.index.get(&key), Some(Some(_)))
                    && let Err(e) = b.append(ERASE, key, &[])
                {
                    log::warn!("Horizon: dropping cube far terrain: {e}");
                }
                Some(())
            });
        }
        let Ok(mut log) = self.changes.lock() else {
            return;
        };
        let at = log.next;
        log.next += 1;
        log.entries.push_back((at, cube));
        if log.entries.len() > LOG {
            log.entries.pop_front();
        }
    }

    /// Where [`Self::changes_since`] starts for a new reader: now.
    pub fn cursor(&self) -> u64 {
        self.changes.lock().map_or(0, |log| log.next)
    }

    /// The cubes invalidated since `cursor`, which then moves to now.
    pub fn changes_since(&self, cursor: &mut u64) -> Changes {
        let Ok(log) = self.changes.lock() else {
            return Changes::All;
        };
        if *cursor == log.next {
            return Changes::None;
        }
        let first = log.entries.front().map_or(log.next, |e| e.0);
        let out = if *cursor < first {
            Changes::All
        } else {
            Changes::Cubes(
                log.entries
                    .iter()
                    .filter(|e| e.0 >= *cursor)
                    .map(|e| e.1)
                    .collect(),
            )
        };
        *cursor = log.next;
        out
    }
}

/// What the cache depends on besides the node data: the format, this build
/// (the generator it samples), the world's seed and the block registry.
pub fn fingerprint(seed: i64, block_states: usize) -> String {
    format!(
        "horizon-cube-lod {FORMAT}\nversion {}\nexe {}\nseed {seed}\nblock_states {block_states}\n",
        env!("CARGO_PKG_VERSION"),
        super::bake::exe_stamp(),
    )
}

fn encode(node: &CubeNode) -> Vec<u8> {
    let mut out = Vec::new();
    let uniform = node.as_uniform();
    let cover = node.covered_mask();
    out.push(u8::from(uniform.is_some()) | u8::from(cover.is_some()) << 1);
    if let Some(mask) = cover {
        for word in mask {
            out.extend_from_slice(&word.to_le_bytes());
        }
    }
    match uniform {
        Some(v) => {
            out.extend_from_slice(&v.state.0.to_le_bytes());
            out.extend_from_slice(&v.biome.to_le_bytes());
        }
        None => {
            let mut raw = Vec::with_capacity(RAW);
            for i in 0..VOLUME {
                raw.extend_from_slice(&node.at(i).state.0.to_le_bytes());
            }
            for i in 0..VOLUME {
                raw.extend_from_slice(&node.at(i).biome.to_le_bytes());
            }
            let mut encoder = ZlibEncoder::new(out, Compression::fast());
            let _ = encoder.write_all(&raw);
            return encoder.finish().unwrap_or_default();
        }
    }
    out
}

fn decode(key: CubeKey, body: &[u8]) -> Option<CubeNode> {
    let (&flags, mut rest) = body.split_first()?;
    let mut cover = None;
    if flags & 2 != 0 {
        let mut mask: CoverMask = [0; 4];
        let (words, tail) = rest.split_at_checked(32)?;
        for (word, bytes) in mask.iter_mut().zip(words.as_chunks::<8>().0) {
            *word = u64::from_le_bytes(*bytes);
        }
        cover = Some(Box::new(mask));
        rest = tail;
    }
    let node = if flags & 1 != 0 {
        let state = u32::from_le_bytes(rest.get(..4)?.try_into().ok()?);
        let biome = u16::from_le_bytes(rest.get(4..6)?.try_into().ok()?);
        CubeNode::uniform(
            key,
            Voxel {
                state: BlockStateId(state),
                biome,
            },
        )
    } else {
        let mut raw = Vec::with_capacity(RAW);
        ZlibDecoder::new(rest).read_to_end(&mut raw).ok()?;
        if raw.len() != RAW {
            return None;
        }
        let (states, biomes) = raw.split_at(VOLUME * 4);
        let mut dense = Box::new(
            [Voxel {
                state: BlockStateId(0),
                biome: 0,
            }; VOLUME],
        );
        for (i, voxel) in dense.iter_mut().enumerate() {
            voxel.state = BlockStateId(u32::from_le_bytes(
                states[i * 4..i * 4 + 4].try_into().ok()?,
            ));
            voxel.biome = u16::from_le_bytes(biomes[i * 2..i * 2 + 2].try_into().ok()?);
        }
        CubeNode::from_dense(key, dense)
    };
    Some(node.with_covered(cover))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("horizon-cube-lod-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir
    }
    fn dense(key: CubeKey, salt: u32) -> CubeNode {
        CubeNode::from_fn(key, |x, y, z| Voxel {
            state: BlockStateId(((x + 2 * y + 3 * z) as u32 + salt) % 5),
            biome: (x ^ z) as u16,
        })
    }
    fn same(a: &CubeNode, b: &CubeNode) {
        assert_eq!(a.key, b.key);
        assert_eq!(a.covered_mask(), b.covered_mask());
        for i in 0..VOLUME {
            assert_eq!(a.at(i), b.at(i), "voxel {i}");
        }
    }

    #[test]
    fn nodes_round_trip_across_sessions() {
        let dir = scratch("round");
        let key = CubeKey::new(2, -3, 5, 9);
        let uniform_key = CubeKey::new(3, 0, -1, 0);
        let mut mask = Box::new([0u64; 4]);
        mask[1] = 0xdead_beef;
        let node = dense(key, 1).with_covered(Some(mask));
        let uniform = CubeNode::uniform(
            uniform_key,
            Voxel {
                state: BlockStateId(7),
                biome: 3,
            },
        );
        {
            let cache = CubeLodCache::open(&dir, "a").unwrap();
            assert!(cache.get(key).is_none());
            cache.put(&node);
            cache.put(&uniform);
            assert!(cache.contains(key));
            same(&cache.get(key).unwrap(), &node);
            // Level 0 is the saved cube itself: never stored.
            cache.put(&dense(CubeKey::new(0, 0, 0, 0), 2));
            assert!(cache.get(CubeKey::new(0, 0, 0, 0)).is_none());
        }
        let cache = CubeLodCache::open(&dir, "a").unwrap();
        same(&cache.get(key).unwrap(), &node);
        same(&cache.get(uniform_key).unwrap(), &uniform);
        assert!(cache.get(CubeKey::new(2, -3, 5, 8)).is_none());
        // A newer node replaces the older.
        let newer = dense(key, 3);
        cache.put(&newer);
        drop(cache);
        let cache = CubeLodCache::open(&dir, "a").unwrap();
        same(&cache.get(key).unwrap(), &newer);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_saved_cube_erases_the_nodes_over_it_and_only_those() {
        let dir = scratch("invalidate");
        let cache = CubeLodCache::open(&dir, "a").unwrap();
        let cube = SectionPos::new(-5, 3, 20);
        let over: Vec<CubeKey> = (1..=MAX_CACHED_LEVEL)
            .map(|l| CubeKey::containing(l, cube))
            .collect();
        let beside = CubeKey::new(1, 100, 100, 100);
        for &key in over.iter().chain(&[beside]) {
            cache.put(&dense(key, 0));
        }
        let mut cursor = cache.cursor();
        cache.invalidate(cube);
        for &key in &over {
            assert!(!cache.contains(key), "{key:?} survived");
            assert!(cache.get(key).is_none());
        }
        assert!(cache.contains(beside));
        assert_eq!(cache.changes_since(&mut cursor), Changes::Cubes(vec![cube]));
        assert_eq!(cache.changes_since(&mut cursor), Changes::None);
        drop(cache);
        // The erasure is on disk too.
        let cache = CubeLodCache::open(&dir, "a").unwrap();
        assert!(over.iter().all(|&k| !cache.contains(k)));
        assert!(cache.contains(beside));
        // A node stored after the save is kept.
        cache.put(&dense(over[0], 4));
        assert!(cache.contains(over[0]));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_changed_fingerprint_empties_the_folder() {
        let dir = scratch("fingerprint");
        let key = CubeKey::new(1, 0, 0, 0);
        let cache = CubeLodCache::open(&dir, "build 1").unwrap();
        cache.put(&dense(key, 0));
        drop(cache);
        let cache = CubeLodCache::open(&dir, "build 1").unwrap();
        assert!(cache.contains(key));
        drop(cache);
        let cache = CubeLodCache::open(&dir, "build 2").unwrap();
        assert!(!cache.contains(key));
        assert!(cache.get(key).is_none());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_torn_last_record_is_cut_off_and_damage_reads_as_missing() {
        let dir = scratch("torn");
        let (a, b) = (CubeKey::new(1, 0, 0, 0), CubeKey::new(1, 1, 0, 0));
        let cache = CubeLodCache::open(&dir, "a").unwrap();
        cache.put(&dense(a, 0));
        cache.put(&dense(b, 1));
        drop(cache);
        let file = dir.join(BucketId::of(a).file());
        let bytes = fs::read(&file).unwrap();
        // The last record loses its end.
        fs::write(&file, &bytes[..bytes.len() - 10]).unwrap();
        let cache = CubeLodCache::open(&dir, "a").unwrap();
        assert!(cache.get(a).is_some());
        assert!(cache.get(b).is_none());
        // Appending after the cut works.
        cache.put(&dense(b, 1));
        assert!(cache.get(b).is_some());
        drop(cache);
        // A flipped byte inside a body fails the checksum.
        let mut bytes = fs::read(&file).unwrap();
        bytes[30] ^= 0xff;
        fs::write(&file, &bytes).unwrap();
        let cache = CubeLodCache::open(&dir, "a").unwrap();
        assert!(cache.get(a).is_none());
        assert!(cache.get(b).is_some());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn so_many_changes_that_the_log_overflows_say_all() {
        let dir = scratch("log");
        let cache = CubeLodCache::open(&dir, "a").unwrap();
        let mut cursor = cache.cursor();
        for i in 0..(LOG as i32 + 10) {
            cache.invalidate(SectionPos::new(i, 0, 0));
        }
        assert_eq!(cache.changes_since(&mut cursor), Changes::All);
        cache.invalidate(SectionPos::new(1, 2, 3));
        assert_eq!(
            cache.changes_since(&mut cursor),
            Changes::Cubes(vec![SectionPos::new(1, 2, 3)])
        );
        let _ = fs::remove_dir_all(&dir);
    }
}
