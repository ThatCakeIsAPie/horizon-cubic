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
//!   sent again whenever the window moves; the client then drops its chunks and
//!   the server sends them all again.
use crate::{
    Error, Result,
    buf::{Reader, Writer},
};

/// The configuration-phase announcement.
pub const MODE_CHANNEL: &str = "horizon:cubic";
/// The play-phase window payload.
pub const WINDOW_CHANNEL: &str = "horizon:cubic_window";

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
}
