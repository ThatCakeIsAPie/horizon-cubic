//! A cubic level's window on the server (docs/cubic-chunks.md, "v1
//! architecture"): the level's chunks hold a window of cubes around its
//! players, and the window moves when a player nears its top or bottom.
//!
//! Moving it is a level switch into the same level: the level is saved
//! whole (its cubes and entities), opened again with the new window, and
//! swapped into place; the hub then tells its players, whose clients lay
//! out their level for the new window and get its chunks again.
use super::{Dim, Game};
use horizon_client::chunk::LevelShape;
use horizon_client::cubic;

impl Game<'_> {
    /// The cubic level of the server's players: the window moves when the
    /// first of them in it comes within `cubic::WINDOW_MARGIN` of its top
    /// or bottom.
    pub(super) fn follow_cubic_window(&mut self) -> horizon_client::Result<()> {
        if self.options.execution_role.is_client() || !self.level_shapes.overworld.cubic {
            return Ok(());
        }
        let Some(y) = self.lan.as_ref().and_then(|h| {
            h.hub
                .guests
                .iter()
                .find(|g| g.placed && g.dimension == Dim::Overworld.id())
                .map(|g| g.position[1])
        }) else {
            return Ok(());
        };
        if !cubic::needs_new_window(self.level_shapes.overworld, y) {
            return Ok(());
        }
        let window = cubic::window_around(self.level_shapes.overworld, y.floor() as i32);
        let back = self.dim;
        self.enter_level(None, Dim::Overworld)?;
        let moved = self.move_cubic_window(window);
        self.enter_level(None, back)?;
        moved
    }

    /// Replaces the cubic level in place with the same level holding
    /// `window`.
    pub(super) fn move_cubic_window(&mut self, window: LevelShape) -> horizon_client::Result<()> {
        let from = self.world.shape;
        log::info!(
            "Horizon: cubic window {}..{} -> {}..{}",
            from.min_y,
            from.max_y(),
            window.min_y,
            window.max_y()
        );
        // The old window's columns and entities saved. Not waited for: the
        // new window reads through the same saves, whose queued writes its
        // loads see.
        self.save_level(false)?;
        *self.level_shapes.get_mut(self.dim.id()) = window;
        if let Err(e) = cubic::save_window(&self.world_root, window.min_y) {
            log::warn!("Horizon: the cubic window was not recorded: {e}");
        }
        // The old window's level, saved above, goes; the new one keeps the
        // generator, its saves and the cube store (no new stronghold search).
        match self.save.cubic.get().cloned() {
            Some(level) => {
                level.set_window(window);
                match level.new_streaming() {
                    Some(streaming) => self.renew_level_in_place(streaming),
                    None => self.reopen_level_in_place()?,
                }
            }
            None => self.reopen_level_in_place()?,
        }
        // The players' clients lay out the new window and get its chunks.
        if let Some(h) = self.lan.as_mut() {
            h.hub.cubic_window_changed(self.dim.id(), window);
        }
        Ok(())
    }
}
