//! Cubic levels on the wire (docs/cubic-chunks.md). A cubic level's blocks reach
//! y = +-8.4 million, which a vanilla `BlockPos` (12 bits of Y) cannot hold, so
//! Horizon peers switch to wide positions ([`crate::buf::set_wide_positions`]).
//!
//! - [`MODE_CHANNEL`] (`horizon:cubic`), an empty payload in the *configuration*
//!   phase, clientbound: this server has a cubic level. The client turns wide
//!   positions on from here, before any play packet. Vanilla clients are refused.
//! - [`WINDOW_CHANNEL`] (`horizon:cubic_window`), in play, clientbound: the
//!   level the player is in is a window of `height` blocks from `min_y`. It
//!   follows the Login/Respawn packet and precedes the level's chunks, and is
//!   sent again whenever the window moves. The client lays its level out for
//!   the new window in place (the sections both windows hold stay), and the
//!   server sends only the cubes that entered, as [`CUBES_CHANNEL`].
//! - [`VIEW_CHANNEL`] (`horizon:cubic_view`), in play, serverbound: the
//!   client's vertical view distance in cubes. It drives the height of the
//!   window the server keeps around the player.
//! - [`CUBES_CHANNEL`] (`horizon:cubes`), in play, clientbound: cubes of one
//!   column the client already holds (the cubes that entered it when the
//!   window moved), with their light and block entities.
use crate::{
    Error, Result,
    buf::{Reader, Writer},
};

/// The configuration-phase announcement.
pub const MODE_CHANNEL: &str = "horizon:cubic";
/// The play-phase window payload.
pub const WINDOW_CHANNEL: &str = "horizon:cubic_window";
/// The vertical view distance a client asks for (serverbound).
pub const VIEW_CHANNEL: &str = "horizon:cubic_view";
/// Cubes of a column the client holds (clientbound).
pub const CUBES_CHANNEL: &str = "horizon:cubes";
/// Most cubes one [`Cubes`] payload holds (a window is at most 99 cubes tall).
pub const MAX_CUBES: usize = 128;
/// Most block entities one cube holds.
const MAX_BLOCK_ENTITIES: usize = 4096;
/// The largest section's bytes one cube may carry.
const MAX_SECTION_BYTES: usize = 1 << 17;

/// The vertical view distance a client asks for, in cubes
/// (`Graphics.cubic_vertical`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct View {
    pub vertical: u8,
}

impl View {
    pub fn encode(self) -> Vec<u8> {
        vec![self.vertical]
    }
    /// The view a payload names; an error unless it is exactly one byte.
    pub fn decode(data: &[u8]) -> Result<Self> {
        let mut r = Reader::new(data);
        let view = Self { vertical: r.u8()? };
        if r.remaining() != 0 {
            return Err(Error::new("cubic view: bad payload"));
        }
        Ok(view)
    }
}

/// A cube's light, as the server has it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CubeLight {
    /// One value everywhere: sky light in the high nibble, block light in the low one.
    Uniform(u8),
    /// 4096 values (`y << 8 | z << 4 | x`), each as `Uniform`'s.
    Full(Vec<u8>),
}

/// A block entity in a cube, as `BlockEntityInfo` has it.
#[derive(Clone, Debug, PartialEq)]
pub struct CubeBlockEntity {
    /// `x << 4 | z` in the column.
    pub packed_xz: u8,
    pub y: i32,
    pub kind: i32,
    /// The update tag, `Tag::End` for none.
    pub tag: horizon_world::nbt::Tag,
}

/// One cube of a [`Cubes`] payload.
#[derive(Clone, Debug, PartialEq)]
pub struct Cube {
    /// The cube's Y (block Y >> 4).
    pub cy: i32,
    /// `LevelChunkSection.write`'s bytes (blocks and biomes).
    pub section: Vec<u8>,
    pub light: CubeLight,
    pub block_entities: Vec<CubeBlockEntity>,
}

/// Cubes of column (`x`, `z`).
#[derive(Clone, Debug, PartialEq)]
pub struct Cubes {
    pub x: i32,
    pub z: i32,
    pub cubes: Vec<Cube>,
}

impl Cubes {
    pub fn encode(&self) -> Vec<u8> {
        let mut w = Writer::new();
        w.i32(self.x);
        w.i32(self.z);
        w.varint(self.cubes.len() as i32);
        for cube in &self.cubes {
            w.i32(cube.cy);
            w.byte_array(&cube.section);
            match &cube.light {
                CubeLight::Uniform(v) => {
                    w.u8(0);
                    w.u8(*v);
                }
                CubeLight::Full(values) => {
                    w.u8(1);
                    w.bytes(values);
                }
            }
            w.varint(cube.block_entities.len() as i32);
            for e in &cube.block_entities {
                w.u8(e.packed_xz);
                w.i32(e.y);
                w.varint(e.kind);
                w.nbt(&e.tag);
            }
        }
        w.data
    }

    /// The cubes a payload carries; an error for a short, oversized or
    /// trailing payload.
    pub fn decode(data: &[u8]) -> Result<Self> {
        let mut r = Reader::new(data);
        let (x, z) = (r.i32()?, r.i32()?);
        let n = r.count(MAX_CUBES)?;
        let mut cubes = Vec::with_capacity(n);
        for _ in 0..n {
            let cy = r.i32()?;
            let section = r.byte_array(MAX_SECTION_BYTES)?;
            let light = match r.u8()? {
                0 => CubeLight::Uniform(r.u8()?),
                1 => CubeLight::Full(r.bytes(4096)?.to_vec()),
                other => return Err(Error::new(format!("cubes: light kind {other}"))),
            };
            let m = r.count(MAX_BLOCK_ENTITIES)?;
            let mut block_entities = Vec::with_capacity(m);
            for _ in 0..m {
                block_entities.push(CubeBlockEntity {
                    packed_xz: r.u8()?,
                    y: r.i32()?,
                    kind: r.varint()?,
                    tag: r.nbt()?,
                });
            }
            cubes.push(Cube {
                cy,
                section,
                light,
                block_entities,
            });
        }
        if r.remaining() != 0 {
            return Err(Error::new("cubes: trailing bytes"));
        }
        Ok(Self { x, z, cubes })
    }
}

/// A cubic level's current window: whole sections.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Window {
    pub min_y: i32,
    pub height: i32,
}

impl Window {
    pub fn encode(self) -> Vec<u8> {
        let mut w = Writer::new();
        w.i32(self.min_y);
        w.i32(self.height);
        w.data
    }
    /// The window a payload names; an error for a short or oversized payload,
    /// or a height that is not positive.
    pub fn decode(data: &[u8]) -> Result<Self> {
        let mut r = Reader::new(data);
        let window = Self {
            min_y: r.i32()?,
            height: r.i32()?,
        };
        if r.remaining() != 0 || window.height <= 0 {
            return Err(Error::new("cubic window: bad payload"));
        }
        Ok(window)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_window_round_trips_and_bad_payloads_are_refused() {
        for window in [
            Window {
                min_y: 1_000_000,
                height: 512,
            },
            Window {
                min_y: -8_388_608,
                height: 256,
            },
        ] {
            assert_eq!(Window::decode(&window.encode()).unwrap(), window);
        }
        assert_eq!(
            Window {
                min_y: 1,
                height: 2
            }
            .encode(),
            [0, 0, 0, 1, 0, 0, 0, 2]
        );
        assert!(Window::decode(&[0; 7]).is_err());
        assert!(Window::decode(&[0; 9]).is_err());
        assert!(Window::decode(&[0; 8]).is_err(), "no height");
    }

    #[test]
    fn a_view_is_one_byte() {
        assert_eq!(
            View::decode(&View { vertical: 12 }.encode())
                .unwrap()
                .vertical,
            12
        );
        assert!(View::decode(&[]).is_err());
        assert!(View::decode(&[1, 2]).is_err());
    }

    #[test]
    fn cubes_round_trip_and_bad_payloads_are_refused() {
        let cubes = Cubes {
            x: -3,
            z: 7,
            cubes: vec![
                Cube {
                    cy: 62_500,
                    section: vec![1, 2, 3],
                    light: CubeLight::Uniform(0xF0),
                    block_entities: Vec::new(),
                },
                Cube {
                    cy: -62_501,
                    section: vec![9; 40],
                    light: CubeLight::Full((0..4096).map(|i| i as u8).collect()),
                    block_entities: vec![CubeBlockEntity {
                        packed_xz: 0x2A,
                        y: -1_000_000,
                        kind: 5,
                        tag: horizon_world::nbt::Tag::End,
                    }],
                },
            ],
        };
        let bytes = cubes.encode();
        assert_eq!(Cubes::decode(&bytes).unwrap(), cubes);
        assert!(Cubes::decode(&bytes[..bytes.len() - 1]).is_err());
        let mut trailing = bytes.clone();
        trailing.push(0);
        assert!(Cubes::decode(&trailing).is_err());
        let mut many = Writer::new();
        many.i32(0);
        many.i32(0);
        many.varint(MAX_CUBES as i32 + 1);
        assert!(Cubes::decode(&many.data).is_err());
    }
}
