//! Ultimine in the game (`horizon_client::ultimine`): the key and its shape,
//! sent to the server when the window is a client; the extra blocks a break
//! takes, each through the normal player break (`break_block_as_player`); a
//! right click that harvests ripe crops or uses a hoe, shovel or axe on the
//! selection; and the preview the window draws (white outline, a HUD panel).
//!
//! Everything here waits for the key: with it up, a break costs one check of
//! `input.held`, and no selection or preview is ever computed.
use super::{Game, Hit, Screen};
use horizon_client::actions::Action;
use horizon_client::{
    extras::{Corner, MAX_ULTIMINE_BLOCKS},
    first_person::Hand,
    gizmos::Line,
    items::ItemStack,
    keybinds,
    text::{self, Text},
    ultimine::{self, Config, Groups, Input, Shape},
    world::World,
};
use horizon_protocol::packets::{configuration::CustomPayload, play::Serverbound as S};
use horizon_world::registry::{BlockRegistry, BlockStateId};
use std::sync::Arc;
use winit::{event::MouseButton, keyboard::KeyCode};

/// The id of the mod registry's Veinminer (`ultimine_yields`).
const VEINMINER: &str = "veinminer";

#[derive(Default)]
pub(super) struct UltimineState {
    /// The acting player's key and shape: the window's own player, or a
    /// guest's while it acts (`enter_guest` swaps it).
    pub(super) input: Input,
    /// The window's Ultimine key is down.
    key_down: bool,
    /// Which blocks merge, built from the block tags once.
    groups: Option<Arc<Groups>>,
    /// What the window's player would take now.
    preview: Option<Preview>,
    /// A native Veinminer is installed (`ultimine_yields`), once asked.
    yields: Option<bool>,
}

struct Preview {
    /// What the selection was computed for: a new one only when this changes.
    key: PreviewKey,
    count: usize,
    lines: Vec<([f64; 3], [f64; 3])>,
}
#[derive(PartialEq)]
struct PreviewKey {
    pos: [i32; 3],
    face: [i32; 3],
    state: BlockStateId,
    shape: Shape,
    facing: [i32; 3],
    held: Option<(Arc<str>, u32)>,
    creative: bool,
    max: u32,
}

/// The blocks around a position, for `BlockTransformer` conditions.
struct Column<'w> {
    world: &'w World,
    registry: &'w BlockRegistry,
    pos: [i32; 3],
}
impl horizon_client::block_transform::Around for Column<'_> {
    fn state(&self, o: [i32; 3]) -> horizon_world::chunk::BlockState {
        let id = self
            .world
            .block(self.pos[0] + o[0], self.pos[1] + o[1], self.pos[2] + o[2]);
        self.registry
            .state(id)
            .cloned()
            .unwrap_or_else(|| horizon_world::chunk::BlockState {
                name: "minecraft:air".into(),
                properties: Default::default(),
            })
    }
}

fn face_name(face: [i32; 3]) -> &'static str {
    match face {
        [0, -1, 0] => "down",
        [0, 1, 0] => "up",
        [0, 0, -1] => "north",
        [0, 0, 1] => "south",
        [-1, 0, 0] => "west",
        _ => "east",
    }
}

impl Game<'_> {
    /// The mod registry's Veinminer, a native mod, takes a break's connected
    /// blocks too: with it installed Ultimine stays off, so a break is not
    /// mined twice over. The native mods are built into the game: asked once.
    fn ultimine_yields(&mut self) -> bool {
        if let Some(yields) = self.ultimine.yields {
            return yields;
        }
        let mut mods = Vec::new();
        self.native_mods(&mut mods);
        let yields = mods.iter().any(|m| m.id == VEINMINER);
        self.ultimine.yields = Some(yields);
        yields
    }

    // The window's key, shape and preview.

    /// A key press or release (physical, before key binds).
    pub(super) fn ultimine_key(&mut self, key: KeyCode, pressed: bool) {
        if self.ultimine_yields() {
            return;
        }
        if keybinds::key_name(key).is_some_and(|n| n == self.settings.extras.ultimine_key) {
            self.ultimine.key_down = pressed;
        }
    }
    /// A mouse button bound as the Ultimine key.
    pub(super) fn ultimine_mouse(&mut self, button: MouseButton, pressed: bool) {
        if self.ultimine_yields() {
            return;
        }
        if keybinds::mouse_name(button) == self.settings.extras.ultimine_key {
            self.ultimine.key_down = pressed;
        }
    }
    /// Keys let go behind screens and while unfocused.
    pub(super) fn ultimine_release(&mut self) {
        self.ultimine.key_down = false;
    }
    /// The window's player holds the key, playing with no text open.
    fn ultimine_window_held(&self) -> bool {
        self.ultimine.key_down
            && self.screen == Screen::Playing
            && self.title.is_none()
            && self.sign_edit.is_none()
            && self.book.is_none()
            && self.player.is_some()
            && !self.spectator()
    }
    /// Sneak + wheel with the key held changes the shape; true when it took the steps.
    pub(super) fn ultimine_scroll(&mut self, steps: f32) -> bool {
        if steps == 0.0 || !self.ultimine_window_held() || !self.keys.held(Action::Sneak) {
            return false;
        }
        self.ultimine_cycle(steps < 0.0);
        true
    }
    /// Up and Down with the key held change the shape; true when it took the key.
    pub(super) fn ultimine_arrow(&mut self, key: KeyCode) -> bool {
        if !matches!(key, KeyCode::ArrowUp | KeyCode::ArrowDown) || !self.ultimine_window_held() {
            return false;
        }
        self.ultimine_cycle(key == KeyCode::ArrowDown);
        true
    }
    fn ultimine_cycle(&mut self, forward: bool) {
        self.ultimine.input.shape = self.ultimine.input.shape.cycle(forward);
        self.ultimine_update_preview();
    }
    /// Once a game tick in the window: the key's state for this tick's
    /// actions, sent to the server when it changed, and the preview.
    pub(super) fn ultimine_client_tick(&mut self) {
        if self.serving.is_some() || self.options.dedicated.is_some() {
            return;
        }
        let held = self.ultimine_window_held();
        self.ultimine.input.held = held;
        let input = self.ultimine.input;
        if let Some(r) = self.remote.as_mut()
            && r.joined
            && r.ultimine_sent != Some(input)
        {
            r.ultimine_sent = Some(input);
            self.send(S::CustomPayload(CustomPayload {
                channel: ultimine::CHANNEL.into(),
                data: input.encode(),
            }));
        }
        if held {
            self.ultimine_update_preview();
        } else {
            self.ultimine.preview = None;
        }
    }
    /// The selection the window's player aims at, computed again only when
    /// the target, its block, the shape or the tool changed.
    fn ultimine_update_preview(&mut self) {
        let Some(hit) = self.pick() else {
            self.ultimine.preview = None;
            return;
        };
        let key = PreviewKey {
            pos: hit.pos,
            face: hit.face,
            state: self.world.block(hit.pos[0], hit.pos[1], hit.pos[2]),
            shape: self.ultimine.input.shape,
            facing: ultimine::facing(self.yaw),
            held: self.held().map(|s| (s.item.clone(), s.damage)),
            creative: self.creative,
            max: ultimine::config().max_blocks,
        };
        if self.ultimine.preview.as_ref().is_some_and(|p| p.key == key) {
            return;
        }
        let blocks = self.ultimine_select(hit.pos, hit.face, key.shape);
        self.ultimine.preview = Some(Preview {
            key,
            count: blocks.len(),
            lines: ultimine::outline(&blocks),
        });
    }
    /// The preview's outline: solid where nothing hides it, faint through blocks.
    pub(super) fn ultimine_gizmos(&self, lines: &mut Vec<Line>, on_top: &mut Vec<Line>) {
        let Some(p) = &self.ultimine.preview else {
            return;
        };
        if !self.ultimine.input.held || p.count < 2 {
            return;
        }
        for &(start, end) in &p.lines {
            lines.push(Line {
                start,
                end,
                color: 0xFFFF_FFFF,
                width: 2.5,
            });
            on_top.push(Line {
                start,
                end,
                color: 0x40FF_FFFF,
                width: 1.5,
            });
        }
    }
    /// The HUD panel while the key is held: the shape and how many blocks it takes.
    pub(super) fn ultimine_hud(&self) -> Option<(Corner, Vec<(String, u32)>)> {
        if !self.ultimine.input.held {
            return None;
        }
        let shape = self.ultimine.input.shape.name();
        let count = self.ultimine.preview.as_ref().map_or(0, |p| p.count);
        let blocks = if count == 1 {
            "1 block".to_owned()
        } else {
            format!("{count} blocks")
        };
        Some((
            Corner::TopLeft,
            vec![(format!("Ultimine: {shape}"), 0xFFFFFF), (blocks, 0xAAAAAA)],
        ))
    }

    // The selection, for whoever acts.

    fn ultimine_groups(&mut self) -> Arc<Groups> {
        self.ultimine
            .groups
            .get_or_insert_with(|| Arc::new(Groups::new(&self.res.items().block_tags)))
            .clone()
    }
    /// The blocks a break of `origin` takes with `shape`, `origin` first: what
    /// the held tool can harvest (anything breakable in creative), within the
    /// world border; none when `origin` itself is not one of them.
    fn ultimine_select(&mut self, origin: [i32; 3], face: [i32; 3], shape: Shape) -> Vec<[i32; 3]> {
        let groups = self.ultimine_groups();
        let held = self.held().cloned();
        let creative = self.creative;
        let take = |p: [i32; 3]| -> Option<String> {
            let f = self.res.facts().get(self.world.block(p[0], p[1], p[2]));
            if f.outline.is_empty() || f.hardness < 0.0 {
                return None;
            }
            if !creative
                && f.requires_tool
                && !self.res.items().correct_for_drops(held.as_ref(), &f.name)
            {
                return None;
            }
            if !self
                .border
                .contains(f64::from(p[0]) + 0.5, f64::from(p[2]) + 0.5)
            {
                return None;
            }
            Some(groups.key(&f.name))
        };
        let Some(key) = take(origin) else {
            return Vec::new();
        };
        let max = ultimine::config().max_blocks as usize;
        ultimine::select(
            origin,
            face,
            ultimine::facing(self.yaw),
            shape,
            max,
            &key,
            &take,
        )
    }
    /// The face of `pos` the acting player aims at: the targeted one, else
    /// the side facing the player's look.
    fn ultimine_face(&self, pos: [i32; 3]) -> [i32; 3] {
        if let Some(hit) = self.pick()
            && hit.pos == pos
        {
            return hit.face;
        }
        let look = horizon_client::gizmos::view_vector(self.pitch, self.yaw);
        let axis = (0..3)
            .max_by(|&a, &b| look[a].abs().total_cmp(&look[b].abs()))
            .unwrap_or(1);
        let mut face = [0; 3];
        face[axis] = if look[axis] > 0.0 { -1 } else { 1 };
        face
    }
    /// The held tool's uses left, when mining wears it.
    fn ultimine_tool_uses(&self) -> Option<u32> {
        let held = self.held()?;
        let info = self.res.items().get(&held.item)?;
        (info.max_damage > 0).then(|| info.max_damage.saturating_sub(held.damage))
    }
    /// What breaking a block costs the held tool (`ItemStack.mineBlock`).
    fn ultimine_wear(&self, name: &str, hardness: f32) -> u32 {
        let Some(per_block) = self
            .held()
            .and_then(|s| self.res.items().get(&s.item))
            .and_then(|i| i.tool_damage_per_block)
        else {
            return 0;
        };
        let wears = if self.held().is_some_and(|s| &*s.item == "minecraft:shears") {
            !self.res.items().block_in_tag("minecraft:fire", name)
        } else {
            hardness != 0.0
        };
        if wears { per_block } else { 0 }
    }
    /// Whether the acting player may go on: food left and, out of creative,
    /// a tool that keeps more than `prevent_tool_break` uses after `cost`.
    fn ultimine_may_continue(&self, tool: &Option<Arc<str>>, cost: u32, config: &Config) -> bool {
        if self.creative {
            return true;
        }
        if self.held().map(|s| s.item.clone()) != *tool {
            return false;
        }
        if self
            .player
            .as_ref()
            .is_some_and(|p| p.health.food.food <= 0)
        {
            return false;
        }
        let uses = if cost == 0 {
            None
        } else {
            self.ultimine_tool_uses()
        };
        ultimine::tool_allows(uses, cost, config.prevent_tool_break)
    }
    fn ultimine_exhaust(&mut self, amount: f32) {
        if self.creative || amount <= 0.0 {
            return;
        }
        if let Some(p) = self.player.as_mut() {
            p.cause_food_exhaustion(amount);
        }
    }

    // Breaking.

    /// The blocks a player's break of `pos` takes besides `pos`, decided
    /// before `pos` goes: empty unless the acting player holds the key.
    pub(super) fn ultimine_plan_break(&mut self, pos: [i32; 3]) -> Vec<[i32; 3]> {
        if !self.ultimine.input.held
            || self.spectator()
            || !self.may_build()
            || self.ultimine_yields()
        {
            return Vec::new();
        }
        let face = self.ultimine_face(pos);
        let mut blocks = self.ultimine_select(pos, face, self.ultimine.input.shape);
        if !blocks.is_empty() {
            blocks.remove(0);
        }
        blocks
    }
    /// Each planned block as a normal player break, until the tool would
    /// break or the player is starving; each costs the extra exhaustion.
    pub(super) fn ultimine_break_extra(
        &mut self,
        blocks: &[[i32; 3]],
    ) -> horizon_client::Result<()> {
        if blocks.is_empty() {
            return Ok(());
        }
        let config = ultimine::config();
        let tool = self.held().map(|s| s.item.clone());
        for &pos in blocks {
            let f = self
                .res
                .facts()
                .get(self.world.block(pos[0], pos[1], pos[2]));
            // Gone since (a neighbour's break took it along).
            if f.outline.is_empty() {
                continue;
            }
            let (name, hardness) = (f.name.clone(), f.hardness);
            let cost = self.ultimine_wear(&name, hardness);
            if !self.ultimine_may_continue(&tool, cost, &config) {
                break;
            }
            self.break_block_as_player(pos)?;
            // Block.playerDestroy already caused vanilla's 0.005.
            self.ultimine_exhaust(config.exhaustion() - 0.005);
        }
        Ok(())
    }

    // Right click.

    /// A right click on a ripe crop with the key held harvests the ripe crops
    /// of its kind the shape selects and plants them again; true when it did.
    pub(super) fn ultimine_harvest(&mut self, hit: &Hit) -> horizon_client::Result<bool> {
        if !self.ultimine.input.held
            || self.ultimine_yields()
            || self.hand != Hand::Main
            || self.spectator()
            || !self.may_build()
        {
            return Ok(false);
        }
        let state = self.world.block(hit.pos[0], hit.pos[1], hit.pos[2]);
        let Some(crop) = self.ultimine_ripe(state) else {
            return Ok(false);
        };
        let take = |p: [i32; 3]| {
            self.ultimine_ripe(self.world.block(p[0], p[1], p[2]))
                .filter(|n| *n == crop)
        };
        let config = ultimine::config();
        let blocks = ultimine::select(
            hit.pos,
            hit.face,
            ultimine::facing(self.yaw),
            self.ultimine.input.shape,
            config.max_blocks as usize,
            &crop,
            &take,
        );
        let hoe = self.held().is_some_and(|s| {
            self.res
                .items()
                .item_tags
                .get("minecraft:hoes")
                .is_some_and(|t| t.contains(&*s.item))
        });
        let tool = self.held().map(|s| s.item.clone());
        for pos in blocks {
            if !self.ultimine_may_continue(&tool, u32::from(hoe), &config) {
                break;
            }
            self.ultimine_replant(pos)?;
            self.ultimine_exhaust(config.exhaustion());
            if hoe {
                self.damage_held(1);
            }
        }
        Ok(true)
    }
    /// The crop's name when `state` is a crop at its last age.
    fn ultimine_ripe(&self, state: BlockStateId) -> Option<String> {
        let registry = self.pack.assets.registry;
        let s = registry.state(state)?;
        let age: u32 = s.properties.get("age")?.parse().ok()?;
        let name = s.name.as_str();
        let crop = matches!(name, "minecraft:nether_wart" | "minecraft:cocoa")
            || (self.res.items().block_in_tag("minecraft:crops", name)
                && !name.ends_with("_stem")
                && !matches!(
                    name,
                    "minecraft:torchflower_crop" | "minecraft:pitcher_crop"
                ));
        if !crop {
            return None;
        }
        let mut older = s.clone();
        older.properties.insert("age".into(), (age + 1).to_string());
        registry.resolve(&older).is_err().then(|| s.name.clone())
    }
    /// One ripe crop: its loot dropped but for one of what plants it, and the
    /// crop again at age 0 (as it faced).
    fn ultimine_replant(&mut self, pos: [i32; 3]) -> horizon_client::Result<()> {
        let registry = self.pack.assets.registry;
        let state = self.world.block(pos[0], pos[1], pos[2]);
        let Some(mut young) = registry.state(state).cloned() else {
            return Ok(());
        };
        young.properties.insert("age".into(), "0".into());
        let Ok(young) = registry.resolve(&young) else {
            return Ok(());
        };
        let crop = self.res.facts().get(state).name.clone();
        let drops = if self.creative {
            Vec::new()
        } else {
            let held = self.held().cloned();
            // A crop has no block entity.
            self.block_loot_at(pos, state, held.as_ref(), None)
        };
        let mut drops: Vec<ItemStack> = drops
            .into_iter()
            .filter(|d| &*d.item != "minecraft:air")
            .collect();
        // The seed kept back to plant it again.
        if let Some(seed) = drops.iter().position(|d| {
            self.res
                .items()
                .get(&d.item)
                .and_then(|i| i.block)
                .is_some_and(|b| self.res.facts().get(b).name == crop)
        }) {
            drops[seed].count -= 1;
        }
        self.level_event_at(2001, pos, state.0 as i32);
        self.level_set(pos, young)?;
        for d in drops.into_iter().filter(|d| d.count > 0) {
            self.pop_resource(pos, d);
        }
        self.post_block_game_event(
            "minecraft:block_change",
            pos,
            self.sculk_actor(),
            Some(young),
        );
        Ok(())
    }
    /// The blocks besides `hit` a hoe, shovel or axe use on it takes: what
    /// its transformer changes from the clicked face (an axe: of the clicked
    /// block's kind too). Decided before the clicked block changes.
    pub(super) fn ultimine_transform_plan(
        &mut self,
        hit: &Hit,
        stack: &ItemStack,
    ) -> Vec<[i32; 3]> {
        if !self.ultimine.input.held
            || self.ultimine_yields()
            || self.hand != Hand::Main
            || self.spectator()
            || !self.may_build()
        {
            return Vec::new();
        }
        let Some(id) = self
            .res
            .items()
            .get(&stack.item)
            .and_then(|i| i.block_transformer.clone())
        else {
            return Vec::new();
        };
        let axe = self
            .res
            .items()
            .item_tags
            .get("minecraft:axes")
            .is_some_and(|t| t.contains(&*stack.item));
        let groups = self.ultimine_groups();
        let face = face_name(hit.face);
        let registry = self.pack.assets.registry;
        let kind = |p: [i32; 3]| {
            let name = &self
                .res
                .facts()
                .get(self.world.block(p[0], p[1], p[2]))
                .name;
            groups.key(name)
        };
        let origin = kind(hit.pos);
        let take = |p: [i32; 3]| -> Option<()> {
            let around = Column {
                world: &self.world,
                registry,
                pos: p,
            };
            self.res
                .transformers()
                .apply(self.res.items(), registry, &id, face, &around)?;
            (!axe || kind(p) == origin).then_some(())
        };
        if take(hit.pos).is_none() {
            return Vec::new();
        }
        let mut blocks = ultimine::select(
            hit.pos,
            hit.face,
            ultimine::facing(self.yaw),
            self.ultimine.input.shape,
            ultimine::config().max_blocks as usize,
            &(),
            &take,
        );
        blocks.remove(0);
        blocks
    }
    /// The planned blocks transformed in turn, each costing the tool its use
    /// and the exhaustion, stopping as a break would.
    pub(super) fn ultimine_transform(
        &mut self,
        blocks: &[[i32; 3]],
        face: [i32; 3],
    ) -> horizon_client::Result<()> {
        let config = ultimine::config();
        let tool = self.held().map(|s| s.item.clone());
        for &pos in blocks {
            let Some(stack) = self.held().cloned() else {
                break;
            };
            if !self.ultimine_may_continue(&tool, 1, &config) {
                break;
            }
            if self.use_block_transformer(&stack, pos, face)? {
                self.ultimine_exhaust(config.exhaustion());
            }
        }
        Ok(())
    }

    // The command.

    /// `/ultimine`, run by the window itself: shows or sets the limits that
    /// worlds this game runs (its own, its server thread, LAN) use.
    pub(super) fn ultimine_command(&mut self, args: &str) {
        let mut c = ultimine::config();
        let words: Vec<&str> = args.split_whitespace().collect();
        let usage = "Usage: /ultimine [reset | max_blocks <1-1024> | exhaustion_per_block <number> | prevent_tool_break <uses>]";
        let error = match words.as_slice() {
            [] => None,
            ["reset"] => {
                c = Config::DEFAULT;
                None
            }
            ["max_blocks", v] => match v.parse::<u32>() {
                Ok(n) if (1..=MAX_ULTIMINE_BLOCKS).contains(&n) => {
                    c.max_blocks = n;
                    None
                }
                _ => Some(format!(
                    "max_blocks is a whole number from 1 to {MAX_ULTIMINE_BLOCKS}"
                )),
            },
            ["exhaustion_per_block", v] => match v.parse::<f32>() {
                Ok(x) if x.is_finite() && x >= 0.0 => {
                    c.exhaustion_per_block = x;
                    None
                }
                _ => Some("exhaustion_per_block is a number, 0 or more".to_owned()),
            },
            ["prevent_tool_break", v] => match v.parse::<u32>() {
                Ok(n) => {
                    c.prevent_tool_break = n;
                    None
                }
                _ => Some("prevent_tool_break is a whole number, 0 or more".to_owned()),
            },
            _ => Some(usage.to_owned()),
        };
        let line = match error {
            Some(e) => Text::literal(e).with_color(text::RED),
            None if self.ultimine_yields() => Text::literal(
                "Ultimine is off: the Veinminer mod does the same, and the two would both mine",
            )
            .with_color(text::RED),
            None => {
                if c != ultimine::config() {
                    ultimine::set_config(c);
                    self.settings.extras.set_ultimine(c);
                    self.settings.save(&Self::options_path());
                }
                Text::literal(format!(
                    "Ultimine: max_blocks {}, exhaustion_per_block {}, prevent_tool_break {} (key: {})",
                    c.max_blocks,
                    c.exhaustion_per_block,
                    c.prevent_tool_break,
                    keybinds::display_name(&self.settings.extras.ultimine_key).plain(),
                ))
            }
        };
        let focused = self.screen == Screen::Chat;
        self.ui.chat.add(
            &self.gui.atlas,
            line,
            None,
            horizon_client::chat::ChatSource::Client,
            self.ticks,
            focused,
        );
    }
}
