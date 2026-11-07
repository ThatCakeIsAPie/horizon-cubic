//! The Customize screen of Create World's Cubic Chunks world types: a port of
//! CubicWorldGen's `CustomCubicGui`. Custom Cubic and Hybrid open its five
//! pages (Basic, Ores, Lakes, Caves, Advanced terrain, with Prev and Next) over
//! a [`CustomGeneratorSettings`]; Custom Cubic has a sixth after Basic,
//! Horizon's Structures, which switches and tunes vanilla's structure sets
//! (Hybrid's structures are vanilla's own, from its vanilla band). Flat Cubic
//! opens a layer editor. Both have Presets, an overlay that imports and exports
//! the settings as JSON.
//!
//! The controls of a page are rebuilt from the state each time they are drawn
//! or pressed ([`page`]): a [`Widget`] is its rectangle, its label and a
//! closure that writes the control's value back into the [`State`]. Done,
//! Cancel, Presets, Prev and Next are ordinary title [`Button`]s.
use super::{
    Button, Title,
    cubic_map::{self, Mapping, show},
    cubic_preview,
};
use crate::{
    cubic_settings::{self, CubicKind},
    hud::{HudAtlas, HudBatch, rgba},
    lang,
    text::Text,
    ui,
};
use horizon_world::cubic::custom::{
    CustomGeneratorSettings, FlatGeneratorSettings,
    settings::{
        BiomeDesc, BlockStateDesc, CaveConfig, FilterType, FlatLayer, LakeConfig,
        OVERWORLD_STRUCTURE_SETS, PeriodicGaussianOreConfig, StandardOreConfig, UserFunction,
        UserFunctionPoint, VanillaPlacement,
    },
};
use std::{cell::RefCell, collections::BTreeSet, path::Path, rc::Rc};

/// Writes into the ore `i` of the combined list (standard ores, then periodic
/// ones); `$body` sees it as `$o`, of either type.
macro_rules! ore {
    ($c:expr, $i:expr, |$o:ident| $body:expr) => {{
        let n = $c.standard_ores.len();
        if $i < n {
            let $o = &mut $c.standard_ores[$i];
            $body;
        } else if let Some($o) = $c.periodic_gaussian_ores.get_mut($i - n) {
            $body;
        }
    }};
}

/// The Presets overlay's JSON box and the picker's search box.
pub(super) const COMPACT: crate::clipboard::FieldId = ("cubic.presets-compact", 0);
pub(super) const SEARCH: crate::clipboard::FieldId = ("cubic.picker-search", 0);

/// Button ids of the screen. Done and Cancel mean the same on the page and in
/// an overlay: apply or drop what the open layer edits.
pub(super) const DONE: u8 = 100;
pub(super) const CANCEL: u8 = 101;
const PRESETS: u8 = 102;
const PREV: u8 = 103;
const NEXT: u8 = 104;
const COPY: u8 = 105;
const DEFAULTS: u8 = 106;
const ALL: u8 = 107;
const INVERT: u8 = 108;

/// A page of Custom Cubic and Hybrid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PageId {
    Basic,
    /// Horizon's: vanilla's structure sets, each switched and tuned.
    Structures,
    Ores,
    Lakes,
    Caves,
    Terrain,
}
impl PageId {
    /// The language key of the page's name.
    fn name(self) -> &'static str {
        match self {
            Self::Basic => "horizon.cubic.page.basic",
            Self::Structures => "horizon.cubic.page.structures",
            Self::Ores => "horizon.cubic.page.ores",
            Self::Lakes => "horizon.cubic.page.lakes",
            Self::Caves => "horizon.cubic.page.caves",
            Self::Terrain => "horizon.cubic.page.terrain",
        }
    }
}
/// Custom Cubic's pages: `CustomCubicGui`'s, in its order, with Structures
/// after Basic.
const CUSTOM_PAGES: &[PageId] = &[
    PageId::Basic,
    PageId::Structures,
    PageId::Ores,
    PageId::Lakes,
    PageId::Caves,
    PageId::Terrain,
];
/// Hybrid's: `CustomCubicGui`'s. Its structures come from its vanilla band,
/// with vanilla's settings, so it has no Structures page.
const HYBRID_PAGES: &[PageId] = &[
    PageId::Basic,
    PageId::Ores,
    PageId::Lakes,
    PageId::Caves,
    PageId::Terrain,
];

const HEADER: f32 = 24.0;
const FOOTER: f32 = 33.0;
/// The content column, two controls wide.
const WIDTH: f32 = 310.0;
const GAP: f32 = 10.0;
const ROW: f32 = 24.0;
/// The terrain preview's pinned strip under the header: the 128 x 64 pixels
/// of the cross-section and its margins.
const PINNED: f32 = 72.0;
const LIST_ROW: f32 = 20.0;
const LIST_WIDTH: f32 = 220.0;
/// The unbounded ends of a Flat Cubic layer.
const BOTTOM: i32 = i32::MIN + 1;
const TOP: i32 = i32::MAX;
const LAYER_RANGE: i32 = 256;
/// The Structures page's slider tracks, in chunks or starts (`(min, max)`):
/// a random spread's spacing (its separation runs from 0 to the spacing less
/// one), the strongholds' ring distance, first-ring spread and count. The
/// settings take more, typed into Presets.
const SPACING: (i32, i32) = (1, 256);
const RING_DISTANCE: (i32, i32) = (1, 128);
const RING_SPREAD: (i32, i32) = (1, 64);
const RING_COUNT: (i32, i32) = (1, 256);

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

/// What the screen hands back to Create World.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum Exit {
    Stay,
    Apply,
    Cancel,
    /// Copy this text to the clipboard (the pretty JSON).
    Copy(String),
}

/// The Customize screen's state.
#[derive(Clone, Debug)]
pub struct State {
    pub(super) kind: CubicKind,
    pub(super) settings: CustomGeneratorSettings,
    /// Flat Cubic's layers, bottom to top (the settings key them by `fromY`).
    pub(super) flat: Vec<FlatLayer>,
    pub(super) page: usize,
    pub(super) scroll: ui::Scroll,
    /// The Lock of the expected heights (Advanced terrain).
    pub(super) lock: bool,
    /// "Set period Z = X".
    pub(super) lock_xz: bool,
    pub(super) show_preview: bool,
    pub(super) axis_z: bool,
    pub(super) preview: cubic_preview::Preview,
    overlay: Overlay,
    /// The slider being dragged, by its place in the page's widgets.
    dragging: Option<usize>,
    /// Block ids and biome ids with their names, for the pickers.
    blocks: Vec<String>,
    biomes: Vec<(String, String)>,
}

#[derive(Clone, Debug)]
enum Overlay {
    None,
    Presets(Presets),
    Picker(Picker),
}

/// The Presets overlay: the JSON as the player edits it, and what it parses to.
#[derive(Clone, Debug)]
struct Presets {
    compact: String,
    focused: bool,
    /// The first pretty line shown.
    top_line: usize,
    /// `compact` and what [`cubic_settings::normalize_json`] made of it.
    view: RefCell<Option<(String, Result<String, String>)>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Target {
    Biome,
    OreBlock(usize),
    OreBiomes(usize),
    LakeBlock(usize),
    LakeBiomes(usize),
    CaveBlock(usize),
    CaveReplaceable(usize),
    LayerBlock(usize),
}

/// A search list of blocks or biomes, one choice or several.
#[derive(Clone, Debug)]
struct Picker {
    target: Target,
    multi: bool,
    /// `(id, name)`; a single choice may start with the row of id `""` ("All").
    rows: Vec<(String, String)>,
    selected: BTreeSet<String>,
    query: String,
    focused: bool,
    scroll: ui::Scroll,
}
impl Picker {
    fn visible(&self) -> Vec<usize> {
        let query = self.query.to_lowercase();
        (0..self.rows.len())
            .filter(|i| query.is_empty() || self.rows[*i].1.to_lowercase().contains(&query))
            .collect()
    }
}

/// Vanilla's overworld blocks the pickers offer when the prepared resources
/// have no block registry.
fn fallback_blocks() -> Vec<String> {
    horizon_world::cubic::custom::testing::default_block_names()
}
/// Biomes the biome picker offers when the prepared resources have none.
const FALLBACK_BIOMES: [&str; 22] = [
    "plains",
    "sunflower_plains",
    "forest",
    "birch_forest",
    "dark_forest",
    "taiga",
    "snowy_taiga",
    "swamp",
    "jungle",
    "savanna",
    "desert",
    "badlands",
    "meadow",
    "grove",
    "snowy_plains",
    "ice_spikes",
    "mushroom_fields",
    "stony_peaks",
    "ocean",
    "deep_ocean",
    "river",
    "beach",
];

fn tr(key: &str) -> String {
    Text::translate(key, &[]).plain()
}
/// "Label: value" (`options.generic_value`).
fn labelled(key: &str, value: impl Into<String>) -> String {
    Text::translate(
        "options.generic_value",
        &[Text::translate(key, &[]), Text::literal(value.into())],
    )
    .plain()
}
fn on_off(on: bool) -> String {
    tr(if on { "options.on" } else { "options.off" })
}
/// The name of a block id: the language's, else the id without `minecraft:`.
fn block_name(id: &str) -> String {
    let (namespace, path) = id.split_once(':').unwrap_or(("minecraft", id));
    let key = format!("block.{namespace}.{}", path.replace('/', "."));
    if lang::has(&key) {
        Text::translate(&key, &[]).plain()
    } else if namespace == "minecraft" {
        path.to_owned()
    } else {
        id.to_owned()
    }
}
fn biome_name(id: &str) -> String {
    let (namespace, path) = id.split_once(':').unwrap_or(("minecraft", id));
    let key = format!("biome.{namespace}.{}", path.replace('/', "."));
    if lang::has(&key) {
        Text::translate(&key, &[]).plain()
    } else if namespace == "minecraft" {
        path.to_owned()
    } else {
        id.to_owned()
    }
}
fn namespaced(id: &str) -> String {
    if id.contains(':') {
        id.to_owned()
    } else {
        format!("minecraft:{id}")
    }
}

impl State {
    /// The screen for a world of `kind` over the settings text `json` (empty
    /// is the defaults; text that does not parse is too). `root` is the
    /// prepared resource folder the block and biome lists come from.
    pub(super) fn new(root: &Path, kind: CubicKind, json: &str) -> Self {
        let mut blocks: BTreeSet<String> = super::flat::registry_ids(root, "minecraft:block");
        if blocks.is_empty() {
            blocks.extend(fallback_blocks());
        }
        let mut biomes: Vec<(String, String)> = super::buffet::biomes(root)
            .into_iter()
            .map(|b| (b.id, b.name))
            .collect();
        if biomes.is_empty() {
            biomes = FALLBACK_BIOMES
                .iter()
                .map(|n| (namespaced(n), biome_name(n)))
                .collect();
        }
        let mut state = Self {
            kind,
            settings: CustomGeneratorSettings::defaults(),
            flat: Vec::new(),
            page: 0,
            scroll: ui::Scroll::default(),
            lock: true,
            lock_xz: false,
            show_preview: true,
            axis_z: false,
            preview: cubic_preview::Preview::fresh(),
            overlay: Overlay::None,
            dragging: None,
            blocks: blocks.into_iter().collect(),
            biomes,
        };
        if state.load_json(json).is_err() {
            let _ = state.load_json("");
        }
        state
    }
    pub fn kind(&self) -> CubicKind {
        self.kind
    }
    pub fn page(&self) -> usize {
        self.page
    }
    /// The number of pages: six for Custom Cubic, five for Hybrid, one for
    /// Flat Cubic's layers.
    pub fn page_count(&self) -> usize {
        self.pages().len().max(1)
    }
    /// The pages of this world type; none for Flat Cubic, whose one page is
    /// its layers.
    fn pages(&self) -> &'static [PageId] {
        match self.kind {
            CubicKind::Custom => CUSTOM_PAGES,
            CubicKind::Hybrid => HYBRID_PAGES,
            CubicKind::Flat => &[],
        }
    }
    /// The open page; `None` on Flat Cubic's layers.
    fn page_id(&self) -> Option<PageId> {
        self.pages().get(self.page).copied()
    }
    /// Reads `text` as the settings of this world type, replacing the current
    /// ones; an error leaves them as they were.
    pub(super) fn load_json(&mut self, text: &str) -> Result<(), String> {
        if self.kind.has_terrain_settings() {
            let settings = CustomGeneratorSettings::from_json(text)?;
            settings.validate()?;
            self.lock = cubic_map::heights_locked(&settings);
            self.lock_xz = [Nz::Depth, Nz::Selector, Nz::Low, Nz::High]
                .iter()
                .all(|n| {
                    let v = noise(&settings, *n);
                    v.freq[0] == v.freq[2]
                });
            self.settings = settings;
        } else {
            let flat = FlatGeneratorSettings::from_json(text)?;
            self.flat = flat.layers.into_values().collect();
        }
        self.preview.touch();
        Ok(())
    }
    /// The settings as pretty JSON, what Done hands back and `cubicgen.json`
    /// holds.
    pub(super) fn json(&self) -> String {
        if self.kind.has_terrain_settings() {
            self.settings.to_json_pretty()
        } else {
            self.flat_settings().to_json_pretty()
        }
    }
    /// The layers as the generator takes them: keyed by `fromY`.
    pub(super) fn flat_settings(&self) -> FlatGeneratorSettings {
        let mut settings = FlatGeneratorSettings::default();
        settings.layers.clear();
        for layer in &self.flat {
            let mut layer = layer.clone();
            while settings.layers.contains_key(&layer.from_y) && layer.from_y < i32::MAX {
                layer.from_y += 1;
            }
            settings.layers.insert(layer.from_y, layer);
        }
        settings
    }
    /// Changes the settings; the Lock keeps the expected heights and the
    /// preview starts over.
    fn edit(&mut self, f: impl FnOnce(&mut CustomGeneratorSettings)) {
        f(&mut self.settings);
        if self.lock {
            cubic_map::lock_heights(&mut self.settings);
        }
        self.preview.touch();
    }
    fn switch_page(&mut self, page: usize) {
        self.page = page.min(self.page_count() - 1);
        self.scroll = ui::Scroll::default();
        self.dragging = None;
    }
    /// The Presets overlay is open.
    pub fn presets_open(&self) -> bool {
        matches!(self.overlay, Overlay::Presets(_))
    }
    pub fn overlay_open(&self) -> bool {
        !matches!(self.overlay, Overlay::None)
    }
    fn open_presets(&mut self) {
        self.dragging = None;
        self.overlay = Overlay::Presets(Presets {
            compact: if self.kind.has_terrain_settings() {
                self.settings.to_json_compact()
            } else {
                self.flat_settings().to_json_compact()
            },
            focused: true,
            top_line: 0,
            view: RefCell::new(None),
        });
    }
    fn open_picker(&mut self, target: Target) {
        self.dragging = None;
        let s = &self.settings;
        let blocks = || {
            self.blocks
                .iter()
                .map(|id| (id.clone(), block_name(id)))
                .collect::<Vec<_>>()
        };
        let ore_biomes = |i: usize| {
            let n = s.standard_ores.len();
            if i < n {
                s.standard_ores[i].biomes.clone()
            } else {
                s.periodic_gaussian_ores
                    .get(i - n)
                    .and_then(|o| o.biomes.clone())
            }
        };
        let ore_block = |i: usize| {
            let n = s.standard_ores.len();
            if i < n {
                s.standard_ores.get(i).map(|o| o.blockstate.name.clone())
            } else {
                s.periodic_gaussian_ores
                    .get(i - n)
                    .map(|o| o.blockstate.name.clone())
            }
        };
        let mut rows;
        let multi;
        let selected: BTreeSet<String> = match target {
            Target::Biome => {
                rows = vec![(String::new(), tr("horizon.cubic.biome.all"))];
                rows.extend(self.biomes.iter().cloned());
                multi = false;
                [s.biome.clone()].into_iter().collect()
            }
            Target::OreBiomes(i) => {
                rows = self.biomes.clone();
                multi = true;
                match ore_biomes(i) {
                    Some(list) => list.iter().map(|b| b.name().to_owned()).collect(),
                    None => self.biomes.iter().map(|b| b.0.clone()).collect(),
                }
            }
            Target::LakeBiomes(i) => {
                rows = self.biomes.clone();
                multi = true;
                s.lakes
                    .get(i)
                    .map(|l| l.biomes.iter().map(|b| b.name().to_owned()).collect())
                    .unwrap_or_default()
            }
            Target::OreBlock(i) => {
                rows = blocks();
                multi = false;
                ore_block(i).into_iter().collect()
            }
            Target::LakeBlock(i) => {
                rows = blocks();
                multi = false;
                s.lakes
                    .get(i)
                    .map(|l| namespaced(&l.block))
                    .into_iter()
                    .collect()
            }
            Target::CaveBlock(i) => {
                rows = blocks();
                multi = false;
                s.caves
                    .get(i)
                    .map(|c| c.cave_block.name.clone())
                    .into_iter()
                    .collect()
            }
            Target::CaveReplaceable(i) => {
                rows = blocks();
                multi = true;
                s.caves
                    .get(i)
                    .map(|c| {
                        c.is_block_replaceable
                            .iter()
                            .map(|b| b.name.clone())
                            .collect()
                    })
                    .unwrap_or_default()
            }
            Target::LayerBlock(i) => {
                rows = blocks();
                multi = false;
                self.flat
                    .get(i)
                    .map(|l| l.block_state.name.clone())
                    .into_iter()
                    .collect()
            }
        };
        // A value from the player's JSON the lists lack stays pickable.
        for id in &selected {
            if !id.is_empty() && !rows.iter().any(|r| &r.0 == id) {
                let name = if matches!(
                    target,
                    Target::Biome | Target::OreBiomes(_) | Target::LakeBiomes(_)
                ) {
                    biome_name(id)
                } else {
                    block_name(id)
                };
                rows.push((id.clone(), name));
            }
        }
        self.overlay = Overlay::Picker(Picker {
            target,
            multi,
            rows,
            selected,
            query: String::new(),
            focused: false,
            scroll: ui::Scroll::default(),
        });
    }
    /// Done in a picker: the choice goes into the settings.
    fn apply_picker(&mut self, picker: &Picker) {
        let one = || picker.selected.iter().next().cloned().unwrap_or_default();
        let all = self.biomes.iter().all(|b| picker.selected.contains(&b.0));
        let list: Vec<String> = picker.selected.iter().cloned().collect();
        match picker.target {
            Target::Biome => self.edit(|c| c.biome = one()),
            Target::OreBlock(i) => {
                let id = one();
                if id.is_empty() {
                    return;
                }
                self.edit(|c| ore!(c, i, |o| o.blockstate = BlockStateDesc::new(&id)));
            }
            Target::OreBiomes(i) => {
                let biomes =
                    (!all).then(|| list.iter().map(|b| BiomeDesc::new(b)).collect::<Vec<_>>());
                self.edit(|c| ore!(c, i, |o| o.biomes = biomes.clone()));
            }
            Target::LakeBlock(i) => {
                let id = one();
                if !id.is_empty() {
                    self.edit(|c| {
                        if let Some(lake) = c.lakes.get_mut(i) {
                            lake.block = id.strip_prefix("minecraft:").unwrap_or(&id).to_owned();
                        }
                    });
                }
            }
            Target::LakeBiomes(i) => self.edit(|c| {
                if let Some(lake) = c.lakes.get_mut(i) {
                    lake.biomes = list.iter().map(|b| BiomeDesc::new(b)).collect();
                }
            }),
            Target::CaveBlock(i) => {
                let id = one();
                if !id.is_empty() {
                    self.edit(|c| {
                        if let Some(cave) = c.caves.get_mut(i) {
                            cave.cave_block = BlockStateDesc::new(&id);
                        }
                    });
                }
            }
            Target::CaveReplaceable(i) => self.edit(|c| {
                if let Some(cave) = c.caves.get_mut(i) {
                    cave.is_block_replaceable =
                        list.iter().map(|b| BlockStateDesc::new(b)).collect();
                }
            }),
            Target::LayerBlock(i) => {
                let id = one();
                if let Some(layer) = self.flat.get_mut(i)
                    && !id.is_empty()
                {
                    layer.block_state = BlockStateDesc::new(&id);
                }
            }
        }
    }
    /// The text field that has the keys, for typing and the clipboard.
    pub(super) fn focused_text(
        &mut self,
    ) -> Option<(crate::clipboard::FieldId, &mut String, usize)> {
        match &mut self.overlay {
            Overlay::Presets(p) if p.focused => Some((COMPACT, &mut p.compact, 8_000_000)),
            Overlay::Picker(p) if p.focused => Some((SEARCH, &mut p.query, 50)),
            _ => None,
        }
    }
    /// The focused text field's box, for the pointer.
    pub(super) fn focused_box(&self, size: (f32, f32)) -> Option<[f32; 4]> {
        match &self.overlay {
            Overlay::Presets(p) if p.focused => Some(compact_box(size.0.ceil())),
            Overlay::Picker(p) if p.focused => Some(search_box(size.0.ceil())),
            _ => None,
        }
    }
    /// Escape: closes an overlay and says so; false when none was open.
    pub(super) fn escape(&mut self) -> bool {
        let open = self.overlay_open();
        self.overlay = Overlay::None;
        open
    }
    /// The Presets overlay's JSON, pretty, while it parses.
    fn presets_result(&self, presets: &Presets) -> Result<String, String> {
        let mut view = presets.view.borrow_mut();
        if view
            .as_ref()
            .is_none_or(|(text, _)| *text != presets.compact)
        {
            *view = Some((
                presets.compact.clone(),
                cubic_settings::normalize_json(self.kind, &presets.compact),
            ));
        }
        view.as_ref()
            .map(|v| v.1.clone())
            .unwrap_or(Err(String::new()))
    }
}

// ---------------------------------------------------------------------------
// Pages
// ---------------------------------------------------------------------------

type Set = Rc<dyn Fn(&mut State, f32)>;
#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Heading,
    Note,
    Button,
    Slider(f32),
}
struct Item {
    kind: Kind,
    label: String,
    enabled: bool,
    set: Option<Set>,
}
struct Widget {
    rect: [f32; 4],
    item: Item,
}
struct Page {
    widgets: Vec<Widget>,
    height: f32,
}

fn button(label: String, f: impl Fn(&mut State) + 'static) -> Item {
    Item {
        kind: Kind::Button,
        label,
        enabled: true,
        set: Some(Rc::new(move |s, _| f(s))),
    }
}
fn slider(label: String, pos: f32, f: impl Fn(&mut State, f32) + 'static) -> Item {
    Item {
        kind: Kind::Slider(pos),
        label,
        enabled: true,
        set: Some(Rc::new(f)),
    }
}
fn disabled(mut item: Item, when: bool) -> Item {
    item.enabled &= !when;
    item
}
/// A slider of `key`'s value under `mapping`, labelled "Name: value".
fn number(key: &str, mapping: Mapping, value: f64, f: impl Fn(&mut State, f64) + 'static) -> Item {
    number_shown(key, mapping, value, show(value), f)
}
fn number_shown(
    key: &str,
    mapping: Mapping,
    value: f64,
    shown: String,
    f: impl Fn(&mut State, f64) + 'static,
) -> Item {
    slider(labelled(key, shown), mapping.pos(value), move |s, p| {
        f(s, mapping.value(p));
    })
}
fn int(key: &str, min: i32, max: i32, value: i32, f: impl Fn(&mut State, i32) + 'static) -> Item {
    number(
        key,
        Mapping::Int { min, max },
        f64::from(value),
        move |s, v| {
            f(s, v as i32);
        },
    )
}
/// A button that switches `on`: "Name: ON".
fn toggle(label: &str, on: bool, f: impl Fn(&mut State, bool) + 'static) -> Item {
    button(labelled(label, on_off(on)), move |s| f(s, !on))
}

struct Builder {
    widgets: Vec<Widget>,
    y: f32,
}
impl Builder {
    fn new() -> Self {
        Self {
            widgets: Vec::new(),
            y: 0.0,
        }
    }
    fn text(&mut self, kind: Kind, label: String) {
        self.widgets.push(Widget {
            rect: [0.0, self.y, WIDTH, 10.0],
            item: Item {
                kind,
                label,
                enabled: true,
                set: None,
            },
        });
        self.y += 14.0;
    }
    fn heading(&mut self, label: String) {
        self.y += 4.0;
        self.text(Kind::Heading, label);
    }
    fn note(&mut self, label: String) {
        self.text(Kind::Note, label);
    }
    /// Empty room of `height` pixels.
    fn space(&mut self, height: f32) {
        self.y += height;
    }
    /// One, two or three controls across the column.
    fn row(&mut self, items: Vec<Item>) {
        let n = items.len().max(1) as f32;
        let width = (WIDTH - GAP * (n - 1.0)) / n;
        let sized = items.into_iter().map(|i| (width, i)).collect();
        self.sized(sized, GAP);
    }
    /// Controls of the given widths, `gap` apart.
    fn sized(&mut self, items: Vec<(f32, Item)>, gap: f32) {
        let mut x = 0.0;
        for (width, item) in items {
            self.widgets.push(Widget {
                rect: [x, self.y, width, 20.0],
                item,
            });
            x += width + gap;
        }
        self.y += ROW;
    }
    fn finish(self) -> Page {
        Page {
            widgets: self.widgets,
            height: self.y,
        }
    }
}

/// The controls of the open page.
fn page(state: &State) -> Page {
    let mut b = Builder::new();
    match state.page_id() {
        None => flat_page(&mut b, state),
        Some(PageId::Basic) => basic_page(&mut b, state),
        Some(PageId::Structures) => structures_page(&mut b, state),
        Some(PageId::Ores) => ores_page(&mut b, state),
        Some(PageId::Lakes) => lakes_page(&mut b, state),
        Some(PageId::Caves) => caves_page(&mut b, state),
        Some(PageId::Terrain) => terrain_page(&mut b, state),
    }
    b.finish()
}

fn basic_page(b: &mut Builder, state: &State) {
    let s = &state.settings;
    b.heading(tr("horizon.cubic.basic.features"));
    // `*`: nothing in Horizon generates it yet (CWG marks its own unfinished ones).
    // CWG's structure toggles are on the Structures page, one switch a set.
    b.row(vec![
        mark("horizon.cubic.flag.dungeons", true, s.dungeons, |c, on| {
            c.dungeons = on
        }),
        mark("horizon.cubic.flag.ravines", false, s.ravines, |c, on| {
            c.ravines = on
        }),
    ]);
    b.note(tr("horizon.cubic.basic.unimplemented"));
    b.heading(tr("horizon.cubic.basic.world"));
    let biome = if s.biome.is_empty() {
        tr("horizon.cubic.biome.all")
    } else {
        biome_name(&s.biome)
    };
    b.row(vec![
        button(labelled("horizon.cubic.biome", biome), |st| {
            st.open_picker(Target::Biome);
        }),
        int(
            "horizon.cubic.dungeon_count",
            1,
            100,
            s.dungeon_count.clamp(1, 100),
            |st, v| st.edit(|c| c.dungeon_count = v),
        ),
    ]);
    b.row(vec![
        int(
            "horizon.cubic.biome_size",
            1,
            8,
            s.biome_size.clamp(1, 8),
            |st, v| {
                st.edit(|c| c.biome_size = v);
            },
        ),
        int(
            "horizon.cubic.river_size_unused",
            1,
            5,
            s.river_size.clamp(1, 5),
            |st, v| st.edit(|c| c.river_size = v),
        ),
    ]);
}
/// A feature toggle; `star` marks one that is not generated yet.
fn mark(
    key: &str,
    star: bool,
    on: bool,
    f: impl Fn(&mut CustomGeneratorSettings, bool) + 'static,
) -> Item {
    let label = format!("{}{}", tr(key), if star { "*" } else { "" });
    button(format!("{label}: {}", on_off(on)), move |st| {
        st.edit(|c| f(c, !on));
    })
}

/// Custom Cubic's Structures page: 26.3's overworld structure sets in
/// [`OVERWORLD_STRUCTURE_SETS`]' order, each a block of its switch and the
/// sliders of its placement, which start at the numbers it generates with
/// (vanilla's unless changed). A random spread has Chance (the frequency),
/// Spacing and Separation (kept below the spacing, so the settings validate);
/// the strongholds' rings have Distance, Spread (the first ring's starts) and
/// Count. The sliders of a set that is off are greyed out. Reset to vanilla
/// heads the page; the list scrolls like the other pages'.
fn structures_page(b: &mut Builder, state: &State) {
    let s = &state.settings;
    b.note(tr("horizon.cubic.structure.help"));
    let vanilla = OVERWORLD_STRUCTURE_SETS
        .iter()
        .all(|v| s.structure_enabled(v.name) && s.structure_placement(v.name) == Some(v.placement));
    b.row(vec![disabled(
        button(tr("horizon.cubic.structure.vanilla"), |st| {
            st.edit(vanilla_structures);
        }),
        vanilla,
    )]);
    for set in OVERWORLD_STRUCTURE_SETS {
        let name = set.name;
        let Some(placement) = s.structure_placement(name) else {
            continue;
        };
        let on = s.structure_enabled(name);
        let switch = button(labelled(&set_key(name), on_off(on)), move |st| {
            st.edit(|c| c.set_structure_enabled(name, !on));
        });
        b.space(4.0);
        match placement {
            VanillaPlacement::Spread {
                spacing,
                separation,
                frequency,
            } => {
                let chance = f64::from(frequency);
                b.row(vec![
                    switch,
                    disabled(
                        number_shown(
                            "horizon.cubic.structure.chance",
                            Mapping::Cubic { max: 1.0 },
                            chance,
                            format!("{}%", show((chance * 10_000.0).round() / 100.0)),
                            move |st, v| spread_mut(st, name, |_, _, f| *f = chance_step(v)),
                        ),
                        !on,
                    ),
                ]);
                b.row(vec![
                    disabled(
                        int(
                            "horizon.cubic.structure.spacing",
                            SPACING.0,
                            SPACING.1,
                            spacing,
                            move |st, v| spread_mut(st, name, |sp, _, _| *sp = v),
                        ),
                        !on,
                    ),
                    disabled(
                        int(
                            "horizon.cubic.structure.separation",
                            0,
                            (spacing - 1).max(0),
                            separation,
                            move |st, v| spread_mut(st, name, |_, sep, _| *sep = v),
                        ),
                        !on || spacing <= 1,
                    ),
                ]);
            }
            VanillaPlacement::Rings {
                distance,
                spread,
                count,
            } => {
                let ring = |key: &str, (min, max): (i32, i32), value: i32, field: usize| {
                    disabled(
                        int(key, min, max, value, move |st, v| {
                            rings_mut(st, name, |r| r[field] = v);
                        }),
                        !on,
                    )
                };
                b.row(vec![
                    switch,
                    ring(
                        "horizon.cubic.structure.distance",
                        RING_DISTANCE,
                        distance,
                        0,
                    ),
                ]);
                b.row(vec![
                    ring("horizon.cubic.structure.spread", RING_SPREAD, spread, 1),
                    ring("horizon.cubic.structure.count", RING_COUNT, count, 2),
                ]);
            }
        }
    }
}
/// The language key of a structure set's name: `minecraft:villages` is
/// `horizon.cubic.set.villages`.
fn set_key(name: &str) -> String {
    let path = name.split_once(':').map_or(name, |(_, path)| path);
    format!("horizon.cubic.set.{path}")
}
/// A Chance slider's frequency: 0.1% steps below 5%, whole percents above,
/// so a drag can land on vanilla's 0.4%, 1% and 20%.
fn chance_step(v: f64) -> f32 {
    let steps = if v < 0.05 { 1000.0 } else { 100.0 };
    ((v * steps).round() / steps) as f32
}
/// Changes structure set `name`'s random spread: `f` gets its spacing,
/// separation and frequency. The separation is then pulled below the spacing,
/// so the settings still validate.
fn spread_mut(st: &mut State, name: &str, f: impl FnOnce(&mut i32, &mut i32, &mut f32)) {
    st.edit(|c| {
        let Some(VanillaPlacement::Spread {
            mut spacing,
            mut separation,
            mut frequency,
        }) = c.structure_placement(name)
        else {
            return;
        };
        f(&mut spacing, &mut separation, &mut frequency);
        let spacing = spacing.max(1);
        c.set_structure_placement(
            name,
            VanillaPlacement::Spread {
                spacing,
                separation: separation.clamp(0, spacing - 1),
                frequency: frequency.clamp(0.0, 1.0),
            },
        );
    });
}
/// Changes structure set `name`'s concentric rings: `f` gets its distance,
/// spread and count, in that order.
fn rings_mut(st: &mut State, name: &str, f: impl FnOnce(&mut [i32; 3])) {
    st.edit(|c| {
        let Some(VanillaPlacement::Rings {
            distance,
            spread,
            count,
        }) = c.structure_placement(name)
        else {
            return;
        };
        let mut r = [distance, spread, count];
        f(&mut r);
        let [distance, spread, count] = r;
        c.set_structure_placement(
            name,
            VanillaPlacement::Rings {
                distance,
                spread,
                count,
            },
        );
    });
}
/// Reset to vanilla: CWG's structure toggles on and every set the page lists
/// back to its own switch and placement. Entries of sets the page does not
/// list (a mod's) stay.
fn vanilla_structures(c: &mut CustomGeneratorSettings) {
    c.strongholds = true;
    c.villages = true;
    c.mineshafts = true;
    c.temples = true;
    c.ocean_monuments = true;
    c.woodland_mansions = true;
    for set in OVERWORLD_STRUCTURE_SETS {
        c.structure_sets.remove(set.name);
    }
}

/// One ore of the combined list, as the page shows it.
struct OreView {
    block: String,
    size: i32,
    tries: i32,
    probability: f64,
    min: f64,
    max: f64,
    biomes: Option<usize>,
    periodic: Option<(f64, f64, f64)>,
}
fn ore_views(s: &CustomGeneratorSettings) -> Vec<OreView> {
    let standard = s.standard_ores.iter().map(|o| OreView {
        block: o.blockstate.name.clone(),
        size: o.spawn_size,
        tries: o.spawn_tries,
        probability: o.spawn_probability,
        min: o.min_height,
        max: o.max_height,
        biomes: o.biomes.as_ref().map(Vec::len),
        periodic: None,
    });
    let periodic = s.periodic_gaussian_ores.iter().map(|o| OreView {
        block: o.blockstate.name.clone(),
        size: o.spawn_size,
        tries: o.spawn_tries,
        probability: o.spawn_probability,
        min: o.min_height,
        max: o.max_height,
        biomes: o.biomes.as_ref().map(Vec::len),
        periodic: Some((o.height_mean, o.height_std_deviation, o.height_spacing)),
    });
    standard.chain(periodic).collect()
}

fn ores_page(b: &mut Builder, state: &State) {
    let s = &state.settings;
    b.row(vec![
        button(tr("horizon.cubic.ore.add"), |st| {
            st.edit(|c| c.standard_ores.push(StandardOreConfig::default()));
        }),
        button(tr("horizon.cubic.ore.add_periodic"), |st| {
            st.edit(|c| {
                c.periodic_gaussian_ores
                    .push(PeriodicGaussianOreConfig::default());
            });
        }),
    ]);
    let variation = s.expected_height_variation;
    let blocks = |h: f64| show(cubic_map::ore_block_y(h, s).round());
    for (i, ore) in ore_views(s).into_iter().enumerate() {
        let kind = if ore.periodic.is_some() {
            "horizon.cubic.ore.periodic"
        } else {
            "horizon.cubic.ore.standard"
        };
        b.heading(format!(
            "{} {}: {}",
            tr(kind),
            i + 1,
            block_name(&ore.block)
        ));
        b.row(vec![
            button(
                labelled("horizon.cubic.block", block_name(&ore.block)),
                move |st| {
                    st.open_picker(Target::OreBlock(i));
                },
            ),
            button(tr("selectServer.delete"), move |st| {
                st.edit(|c| {
                    let n = c.standard_ores.len();
                    if i < n {
                        c.standard_ores.remove(i);
                    } else if i - n < c.periodic_gaussian_ores.len() {
                        c.periodic_gaussian_ores.remove(i - n);
                    }
                });
            }),
        ]);
        b.row(vec![
            int(
                "horizon.cubic.ore.size",
                1,
                50,
                ore.size.clamp(1, 50),
                move |st, v| {
                    st.edit(|c| ore!(c, i, |o| o.spawn_size = v));
                },
            ),
            int(
                "horizon.cubic.ore.tries",
                1,
                40,
                ore.tries.clamp(1, 40),
                move |st, v| {
                    st.edit(|c| ore!(c, i, |o| o.spawn_tries = v));
                },
            ),
        ]);
        let biomes = match ore.biomes {
            None => tr("horizon.cubic.biome.all"),
            Some(n) => n.to_string(),
        };
        b.row(vec![
            number(
                "horizon.cubic.ore.probability",
                Mapping::Linear { min: 0.0, max: 1.0 },
                ore.probability,
                move |st, v| st.edit(|c| ore!(c, i, |o| o.spawn_probability = v)),
            ),
            button(labelled("horizon.cubic.ore.biomes", biomes), move |st| {
                st.open_picker(Target::OreBiomes(i));
            }),
        ]);
        let height = |key: &str, value: f64, low: bool| {
            let shown = if value.is_finite() {
                format!("{} ({})", show(value), blocks(value))
            } else {
                tr("horizon.cubic.unbounded")
            };
            number_shown(
                key,
                Mapping::Linear {
                    min: -cubic_map::ORE_RANGE,
                    max: cubic_map::ORE_RANGE,
                },
                cubic_map::ore_slider_value(value),
                shown,
                move |st, v| {
                    let h = cubic_map::ore_from_slider(v);
                    st.edit(|c| {
                        if low {
                            ore!(c, i, |o| o.min_height = h);
                        } else {
                            ore!(c, i, |o| o.max_height = h);
                        }
                    });
                },
            )
        };
        b.row(vec![
            height("horizon.cubic.ore.min_height", ore.min, true),
            height("horizon.cubic.ore.max_height", ore.max, false),
        ]);
        if let Some((mean, deviation, spacing)) = ore.periodic {
            b.row(vec![
                number_shown(
                    "horizon.cubic.ore.mean",
                    Mapping::Linear {
                        min: -4.0,
                        max: 4.0,
                    },
                    mean,
                    format!("{} ({})", show(mean), blocks(mean)),
                    move |st, v| st.edit(|c| periodic(c, i, |o| o.height_mean = v)),
                ),
                number_shown(
                    "horizon.cubic.ore.deviation",
                    Mapping::Linear { min: 0.0, max: 1.0 },
                    deviation,
                    format!(
                        "{} ({})",
                        show(deviation),
                        show((deviation * variation).round())
                    ),
                    move |st, v| {
                        st.edit(|c| periodic(c, i, |o| o.height_std_deviation = v.max(0.001)));
                    },
                ),
            ]);
            b.row(vec![number_shown(
                "horizon.cubic.ore.spacing",
                Mapping::PosExp { lo: -1.0, hi: 6.0 },
                spacing,
                format!(
                    "{} ({})",
                    show(spacing),
                    show((spacing * variation).round())
                ),
                move |st, v| st.edit(|c| periodic(c, i, |o| o.height_spacing = v)),
            )]);
        }
    }
}
/// `f` on periodic ore `i` of the combined list, if it is one.
fn periodic(
    c: &mut CustomGeneratorSettings,
    i: usize,
    f: impl FnOnce(&mut PeriodicGaussianOreConfig),
) {
    let n = c.standard_ores.len();
    if i >= n
        && let Some(ore) = c.periodic_gaussian_ores.get_mut(i - n)
    {
        f(ore);
    }
}

fn lakes_page(b: &mut Builder, state: &State) {
    b.row(vec![button(tr("horizon.cubic.lake.add"), |st| {
        st.edit(|c| c.lakes.push(LakeConfig::default()));
    })]);
    for (i, lake) in state.settings.lakes.iter().enumerate() {
        b.heading(format!(
            "{} {}: {}",
            tr("horizon.cubic.lake"),
            i + 1,
            block_name(&lake.block)
        ));
        b.row(vec![
            button(
                labelled("horizon.cubic.block", block_name(&lake.block)),
                move |st| {
                    st.open_picker(Target::LakeBlock(i));
                },
            ),
            button(tr("selectServer.delete"), move |st| {
                st.edit(|c| {
                    if i < c.lakes.len() {
                        c.lakes.remove(i);
                    }
                });
            }),
        ]);
        let include = lake.biome_select == FilterType::Include;
        b.row(vec![
            button(
                labelled("horizon.cubic.lake.biomes", lake.biomes.len().to_string()),
                move |st| st.open_picker(Target::LakeBiomes(i)),
            ),
            button(
                labelled(
                    "horizon.cubic.lake.filter",
                    tr(if include {
                        "horizon.cubic.lake.include"
                    } else {
                        "horizon.cubic.lake.exclude"
                    }),
                ),
                move |st| {
                    st.edit(|c| {
                        if let Some(l) = c.lakes.get_mut(i) {
                            l.biome_select = if include {
                                FilterType::Exclude
                            } else {
                                FilterType::Include
                            };
                        }
                    });
                },
            ),
        ]);
        function_rows(
            b,
            i,
            false,
            "horizon.cubic.lake.surface",
            &lake.surface_probability,
        );
        function_rows(
            b,
            i,
            true,
            "horizon.cubic.lake.underground",
            &lake.main_probability,
        );
    }
}
/// The `(y, value)` points of one of a lake's probability functions: a Y
/// slider, a probability slider and a delete button each, then Add point.
fn function_rows(b: &mut Builder, lake: usize, main: bool, key: &str, f: &UserFunction) {
    b.note(tr(key));
    let y_map = Mapping::Linear {
        min: -256.0,
        max: 512.0,
    };
    let p_map = Mapping::Cubic { max: 1.0 };
    for (p, point) in f.0.iter().enumerate() {
        b.sized(
            vec![
                (
                    124.0,
                    number("horizon.cubic.point.y", y_map, point.y, move |st, v| {
                        let y = v.round();
                        st.edit(|c| point_mut(c, lake, main, p, |pt| pt.y = y));
                    }),
                ),
                (
                    124.0,
                    number_shown(
                        "horizon.cubic.point.value",
                        p_map,
                        point.v,
                        format!("{}%", show((point.v * 10_000.0).round() / 100.0)),
                        move |st, v| st.edit(|c| point_mut(c, lake, main, p, |pt| pt.v = v)),
                    ),
                ),
                (
                    24.0,
                    button("x".into(), move |st| {
                        st.edit(|c| {
                            if let Some(f) = function_mut(c, lake, main)
                                && p < f.0.len()
                            {
                                f.0.remove(p);
                            }
                        });
                    }),
                ),
            ],
            4.0,
        );
    }
    b.row(vec![button(tr("horizon.cubic.point.add"), move |st| {
        st.edit(|c| {
            if let Some(f) = function_mut(c, lake, main) {
                let last =
                    f.0.last()
                        .copied()
                        .unwrap_or(UserFunctionPoint { y: 0.0, v: 0.0 });
                f.0.push(UserFunctionPoint {
                    y: last.y + 16.0,
                    v: last.v,
                });
            }
        });
    })]);
}
fn function_mut(
    c: &mut CustomGeneratorSettings,
    lake: usize,
    main: bool,
) -> Option<&mut UserFunction> {
    let l = c.lakes.get_mut(lake)?;
    Some(if main {
        &mut l.main_probability
    } else {
        &mut l.surface_probability
    })
}
fn point_mut(
    c: &mut CustomGeneratorSettings,
    lake: usize,
    main: bool,
    point: usize,
    f: impl FnOnce(&mut UserFunctionPoint),
) {
    if let Some(p) = function_mut(c, lake, main).and_then(|f| f.0.get_mut(point)) {
        f(p);
    }
}

fn caves_page(b: &mut Builder, state: &State) {
    b.row(vec![button(tr("horizon.cubic.cave.add"), |st| {
        st.edit(|c| c.caves.push(CaveConfig::overworld()));
    })]);
    for (i, cave) in state.settings.caves.iter().enumerate() {
        b.heading(format!("{} {}", tr("horizon.cubic.cave"), i + 1));
        b.row(vec![
            button(
                labelled("horizon.cubic.block", block_name(&cave.cave_block.name)),
                move |st| {
                    st.open_picker(Target::CaveBlock(i));
                },
            ),
            button(tr("selectServer.delete"), move |st| {
                st.edit(|c| {
                    if i < c.caves.len() {
                        c.caves.remove(i);
                    }
                });
            }),
        ]);
        b.row(vec![button(
            labelled(
                "horizon.cubic.cave.replaceable",
                cave.is_block_replaceable.len().to_string(),
            ),
            move |st| st.open_picker(Target::CaveReplaceable(i)),
        )]);
        let cube = |key: &str, value: i32, lower: bool| {
            let at_end = if lower {
                value <= -cubic_map::CAVE_RANGE
            } else {
                value >= cubic_map::CAVE_RANGE
            };
            let shown = if at_end {
                tr("horizon.cubic.unbounded")
            } else {
                format!("{value} ({})", value * 16)
            };
            number_shown(
                key,
                Mapping::Int {
                    min: -cubic_map::CAVE_RANGE,
                    max: cubic_map::CAVE_RANGE,
                },
                f64::from(cubic_map::cave_slider_value(value)),
                shown,
                move |st, v| {
                    let y = cubic_map::cave_from_slider(v as i32, lower);
                    st.edit(|c| {
                        if let Some(cave) = c.caves.get_mut(i) {
                            if lower {
                                cave.cave_min_height = y;
                            } else {
                                cave.cave_max_height = y;
                            }
                        }
                    });
                },
            )
        };
        b.row(vec![
            cube("horizon.cubic.cave.min_cube", cave.cave_min_height, true),
            cube("horizon.cubic.cave.max_cube", cave.cave_max_height, false),
        ]);
        macro_rules! int_slider {
            ($key:literal, $max:literal, $field:ident) => {
                int($key, 1, $max, cave.$field.clamp(1, $max), move |st, v| {
                    st.edit(|c| {
                        if let Some(cave) = c.caves.get_mut(i) {
                            cave.$field = v;
                        }
                    });
                })
            };
        }
        macro_rules! float_slider {
            ($key:literal, $min:literal, $max:literal, $field:ident) => {
                number(
                    $key,
                    Mapping::Linear {
                        min: $min,
                        max: $max,
                    },
                    cave.$field,
                    move |st, v| {
                        let v = (v * 1000.0).round() / 1000.0;
                        st.edit(|c| {
                            if let Some(cave) = c.caves.get_mut(i) {
                                cave.$field = v;
                            }
                        });
                    },
                )
            };
        }
        b.row(vec![
            int_slider!("horizon.cubic.cave.rarity", 128, cave_rarity),
            int_slider!("horizon.cubic.cave.max_init_nodes", 128, max_init_nodes),
        ]);
        b.row(vec![
            int_slider!(
                "horizon.cubic.cave.large_node_rarity",
                128,
                large_node_rarity
            ),
            int_slider!(
                "horizon.cubic.cave.large_node_max_branches",
                256,
                large_node_max_branches
            ),
        ]);
        b.row(vec![
            int_slider!("horizon.cubic.cave.big_cave_rarity", 128, big_cave_rarity),
            float_slider!("horizon.cubic.cave.size_add", 0.0, 16.0, cave_size_add),
        ]);
        b.row(vec![
            int_slider!(
                "horizon.cubic.cave.steep_step_rarity",
                128,
                steep_step_rarity
            ),
            int_slider!(
                "horizon.cubic.cave.carve_step_rarity",
                128,
                carve_step_rarity
            ),
        ]);
        b.row(vec![
            float_slider!("horizon.cubic.cave.flatten", 0.0, 1.0, flatten_factor),
            float_slider!(
                "horizon.cubic.cave.steeper_flatten",
                0.0,
                1.0,
                steeper_flatten_factor
            ),
        ]);
        b.row(vec![
            float_slider!(
                "horizon.cubic.cave.direction_change",
                0.0,
                1.0,
                direction_change_factor
            ),
            float_slider!(
                "horizon.cubic.cave.floor_depth",
                -1.0,
                1.0,
                cave_floor_depth
            ),
        ]);
        b.row(vec![
            float_slider!(
                "horizon.cubic.cave.prev_horizontal",
                0.0,
                1.0,
                prev_horiz_direction_change_weight
            ),
            float_slider!(
                "horizon.cubic.cave.prev_vertical",
                0.0,
                1.0,
                prev_vert_direction_change_weight
            ),
        ]);
        b.row(vec![
            float_slider!(
                "horizon.cubic.cave.add_horizontal",
                0.0,
                128.0,
                max_add_direction_change_horiz
            ),
            float_slider!(
                "horizon.cubic.cave.add_vertical",
                0.0,
                128.0,
                max_add_direction_change_vert
            ),
        ]);
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Nz {
    Depth,
    Selector,
    Low,
    High,
}
/// One noise's settings as the page edits them: its frequencies (the depth
/// noise has no Y), octaves, factor and offset.
#[derive(Clone, Copy)]
struct Noise {
    freq: [f64; 3],
    octaves: i32,
    factor: f64,
    offset: f64,
}
fn noise(c: &CustomGeneratorSettings, n: Nz) -> Noise {
    match n {
        Nz::Depth => Noise {
            freq: [c.depth_noise_frequency_x, 0.0, c.depth_noise_frequency_z],
            octaves: c.depth_noise_octaves,
            factor: c.depth_noise_factor,
            offset: c.depth_noise_offset,
        },
        Nz::Selector => Noise {
            freq: [
                c.selector_noise_frequency_x,
                c.selector_noise_frequency_y,
                c.selector_noise_frequency_z,
            ],
            octaves: c.selector_noise_octaves,
            factor: c.selector_noise_factor,
            offset: c.selector_noise_offset,
        },
        Nz::Low => Noise {
            freq: [
                c.low_noise_frequency_x,
                c.low_noise_frequency_y,
                c.low_noise_frequency_z,
            ],
            octaves: c.low_noise_octaves,
            factor: c.low_noise_factor,
            offset: c.low_noise_offset,
        },
        Nz::High => Noise {
            freq: [
                c.high_noise_frequency_x,
                c.high_noise_frequency_y,
                c.high_noise_frequency_z,
            ],
            octaves: c.high_noise_octaves,
            factor: c.high_noise_factor,
            offset: c.high_noise_offset,
        },
    }
}
fn set_noise(c: &mut CustomGeneratorSettings, n: Nz, f: impl FnOnce(&mut Noise)) {
    let mut v = noise(c, n);
    f(&mut v);
    match n {
        Nz::Depth => {
            c.depth_noise_frequency_x = v.freq[0];
            c.depth_noise_frequency_z = v.freq[2];
            c.depth_noise_octaves = v.octaves;
            c.depth_noise_factor = v.factor;
            c.depth_noise_offset = v.offset;
        }
        Nz::Selector => {
            c.selector_noise_frequency_x = v.freq[0];
            c.selector_noise_frequency_y = v.freq[1];
            c.selector_noise_frequency_z = v.freq[2];
            c.selector_noise_octaves = v.octaves;
            c.selector_noise_factor = v.factor;
            c.selector_noise_offset = v.offset;
        }
        Nz::Low => {
            c.low_noise_frequency_x = v.freq[0];
            c.low_noise_frequency_y = v.freq[1];
            c.low_noise_frequency_z = v.freq[2];
            c.low_noise_octaves = v.octaves;
            c.low_noise_factor = v.factor;
            c.low_noise_offset = v.offset;
        }
        Nz::High => {
            c.high_noise_frequency_x = v.freq[0];
            c.high_noise_frequency_y = v.freq[1];
            c.high_noise_frequency_z = v.freq[2];
            c.high_noise_octaves = v.octaves;
            c.high_noise_factor = v.factor;
            c.high_noise_offset = v.offset;
        }
    }
}
/// A period slider's track: `2^2` to `2^13` blocks.
const PERIOD: Mapping = Mapping::Period { lo: 2.0, hi: 13.0 };

/// A noise's period sliders, then octaves, factor and offset.
fn noise_rows(
    b: &mut Builder,
    state: &State,
    n: Nz,
    group: &str,
    factor: Mapping,
    offset: Mapping,
) {
    let s = &state.settings;
    let v = noise(s, n);
    b.heading(tr(group));
    let axes: &[(usize, &str)] = if n == Nz::Depth {
        &[
            (0, "horizon.cubic.noise.period_x"),
            (2, "horizon.cubic.noise.period_z"),
        ]
    } else {
        &[
            (0, "horizon.cubic.noise.period_x"),
            (1, "horizon.cubic.noise.period_y"),
            (2, "horizon.cubic.noise.period_z"),
        ]
    };
    let lock_xz = state.lock_xz;
    let periods = axes
        .iter()
        .map(|&(axis, key)| {
            let period = cubic_map::period(v.freq[axis]);
            disabled(
                number(key, PERIOD, period, move |st, p| {
                    let f = cubic_map::frequency(p);
                    st.edit(|c| {
                        set_noise(c, n, |nz| {
                            nz.freq[axis] = f;
                            if lock_xz && axis == 0 {
                                nz.freq[2] = f;
                            }
                        });
                    });
                }),
                lock_xz && axis == 2,
            )
        })
        .collect();
    b.row(periods);
    b.row(vec![
        int(
            "horizon.cubic.noise.octaves",
            1,
            16,
            v.octaves.clamp(1, 16),
            move |st, o| {
                st.edit(|c| set_noise(c, n, |nz| nz.octaves = o));
            },
        ),
        number(
            "horizon.cubic.noise.factor",
            factor,
            v.factor,
            move |st, f| {
                st.edit(|c| set_noise(c, n, |nz| nz.factor = f));
            },
        ),
        number(
            "horizon.cubic.noise.offset",
            offset,
            v.offset,
            move |st, f| {
                st.edit(|c| set_noise(c, n, |nz| nz.offset = f));
            },
        ),
    ]);
}

fn terrain_page(b: &mut Builder, state: &State) {
    let s = &state.settings;
    let locked = state.lock;
    b.row(vec![
        toggle(
            "horizon.cubic.terrain.preview",
            state.show_preview,
            |st, on| {
                st.show_preview = on;
                st.preview.touch();
            },
        ),
        button(
            labelled(
                "horizon.cubic.terrain.axis",
                if state.axis_z { "Z" } else { "X" },
            ),
            |st| {
                st.axis_z = !st.axis_z;
                st.preview.touch();
            },
        ),
    ]);
    b.heading(tr("horizon.cubic.terrain.expected"));
    b.row(vec![
        toggle("horizon.cubic.terrain.lock", locked, |st, on| {
            st.lock = on;
            if on {
                st.edit(|_| {});
            }
        }),
        disabled(
            number(
                "horizon.cubic.terrain.actual_height",
                Mapping::SymExp { lo: 0.0, hi: 20.0 },
                s.actual_height,
                |st, v| st.edit(|c| c.actual_height = v),
            ),
            locked,
        ),
    ]);
    b.row(vec![
        disabled(
            number(
                "horizon.cubic.terrain.expected_base",
                Mapping::SymExp { lo: 0.0, hi: 20.0 },
                s.expected_base_height,
                |st, v| st.edit(|c| c.expected_base_height = v),
            ),
            locked,
        ),
        disabled(
            number(
                "horizon.cubic.terrain.expected_variation",
                Mapping::PosExp { lo: 0.0, hi: 20.0 },
                s.expected_height_variation,
                |st, v| st.edit(|c| c.expected_height_variation = v),
            ),
            locked,
        ),
    ]);
    b.heading(tr("horizon.cubic.terrain.variation"));
    b.row(vec![
        number(
            "horizon.cubic.terrain.variation_factor",
            Mapping::PosExp { lo: 0.0, hi: 20.0 },
            s.height_variation_factor,
            |st, v| st.edit(|c| c.height_variation_factor = v),
        ),
        number(
            "horizon.cubic.terrain.variation_special",
            Mapping::PosExp { lo: -6.0, hi: 6.0 },
            s.special_height_variation_factor_below_average_y,
            |st, v| st.edit(|c| c.special_height_variation_factor_below_average_y = v),
        ),
    ]);
    b.row(vec![number(
        "horizon.cubic.terrain.variation_offset",
        Mapping::SymExp { lo: 0.0, hi: 20.0 },
        s.height_variation_offset,
        |st, v| st.edit(|c| c.height_variation_offset = v),
    )]);
    b.heading(tr("horizon.cubic.terrain.height"));
    b.row(vec![
        number(
            "horizon.cubic.terrain.height_factor",
            Mapping::SymExp { lo: 1.0, hi: 20.0 },
            s.height_factor,
            |st, v| st.edit(|c| c.height_factor = v),
        ),
        number(
            "horizon.cubic.terrain.height_offset",
            Mapping::SymExp { lo: 1.0, hi: 20.0 },
            s.height_offset,
            |st, v| st.edit(|c| c.height_offset = v),
        ),
    ]);
    b.row(vec![toggle(
        "horizon.cubic.terrain.lock_xz",
        state.lock_xz,
        |st, on| {
            st.lock_xz = on;
            if on {
                st.edit(|c| {
                    for n in [Nz::Depth, Nz::Selector, Nz::Low, Nz::High] {
                        set_noise(c, n, |nz| nz.freq[2] = nz.freq[0]);
                    }
                });
            }
        },
    )]);
    noise_rows(
        b,
        state,
        Nz::Depth,
        "horizon.cubic.noise.depth",
        Mapping::PosExp { lo: 1.0, hi: 12.0 },
        Mapping::SymExp { lo: 1.0, hi: 12.0 },
    );
    noise_rows(
        b,
        state,
        Nz::Selector,
        "horizon.cubic.noise.selector",
        Mapping::PosExp { lo: 0.0, hi: 10.0 },
        Mapping::SymExp { lo: -5.0, hi: 5.0 },
    );
    for (n, group) in [
        (Nz::Low, "horizon.cubic.noise.low"),
        (Nz::High, "horizon.cubic.noise.high"),
    ] {
        noise_rows(
            b,
            state,
            n,
            group,
            Mapping::PosExp {
                lo: -10.0,
                hi: 10.0,
            },
            Mapping::SymExp { lo: -5.0, hi: 5.0 },
        );
    }
    b.heading(tr("horizon.cubic.terrain.samples"));
    let step = |n: i32| match n {
        1 => 2,
        2 => 4,
        4 => 8,
        8 => 16,
        _ => 1,
    };
    let (sx, sy) = (s.noise_sample_size_x, s.noise_sample_size_y);
    b.row(vec![
        button(
            labelled("horizon.cubic.terrain.sample_xz", sx.to_string()),
            move |st| {
                // X must equal Z.
                st.edit(|c| {
                    c.noise_sample_size_x = step(sx);
                    c.noise_sample_size_z = step(sx);
                });
            },
        ),
        button(
            labelled("horizon.cubic.terrain.sample_y", sy.to_string()),
            move |st| {
                st.edit(|c| c.noise_sample_size_y = step(sy));
            },
        ),
    ]);
}

fn flat_page(b: &mut Builder, state: &State) {
    b.note(tr("horizon.cubic.flat.help"));
    b.row(vec![button(tr("horizon.cubic.flat.add"), |st| {
        let from = st
            .flat
            .iter()
            .map(|l| l.to_y)
            .max()
            .filter(|y| *y < TOP)
            .unwrap_or(0);
        st.flat.push(FlatLayer {
            from_y: from,
            to_y: from.saturating_add(1),
            biome: -1,
            block_state: BlockStateDesc::new("stone"),
        });
    })]);
    let map = Mapping::Int {
        min: -LAYER_RANGE,
        max: LAYER_RANGE,
    };
    for (i, layer) in state.flat.iter().enumerate() {
        b.heading(format!(
            "{} {}: {}",
            tr("horizon.cubic.flat.layer"),
            i + 1,
            block_name(&layer.block_state.name)
        ));
        b.row(vec![
            button(
                labelled("horizon.cubic.block", block_name(&layer.block_state.name)),
                move |st| st.open_picker(Target::LayerBlock(i)),
            ),
            button(tr("selectServer.delete"), move |st| {
                if i < st.flat.len() {
                    st.flat.remove(i);
                }
            }),
        ]);
        let end = |key: &str, y: i32, from: bool| {
            let open = if from { y <= BOTTOM } else { y == TOP };
            let shown = if open {
                tr(if from {
                    "horizon.cubic.flat.bottom"
                } else {
                    "horizon.cubic.flat.top"
                })
            } else {
                y.to_string()
            };
            number_shown(
                key,
                map,
                f64::from(y.clamp(-LAYER_RANGE, LAYER_RANGE)),
                shown,
                move |st, v| {
                    let v = v as i32;
                    let y = if from && v <= -LAYER_RANGE {
                        BOTTOM
                    } else if !from && v >= LAYER_RANGE {
                        TOP
                    } else {
                        v
                    };
                    if let Some(layer) = st.flat.get_mut(i) {
                        if from {
                            layer.from_y = y;
                        } else {
                            layer.to_y = y;
                        }
                    }
                },
            )
        };
        b.row(vec![
            end("horizon.cubic.flat.from", layer.from_y, true),
            end("horizon.cubic.flat.to", layer.to_y, false),
        ]);
    }
}

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

/// The list of the open page: top, bottom and left of the column.
fn list_area(state: &State, (w, h): (f32, f32)) -> (f32, f32, f32) {
    let pinned = if state.page_id() == Some(PageId::Terrain) && state.show_preview {
        PINNED
    } else {
        0.0
    };
    (
        HEADER + pinned,
        h.ceil() - FOOTER,
        ((w.ceil() - WIDTH) * 0.5).floor(),
    )
}
fn content_height(page: &Page) -> f32 {
    page.height + 8.0
}
/// The Presets overlay's JSON box.
pub(super) fn compact_box(w: f32) -> [f32; 4] {
    [20.0, 38.0, (w - 40.0).max(1.0), 20.0]
}
/// The picker's search box.
pub(super) fn search_box(w: f32) -> [f32; 4] {
    [((w - 200.0) * 0.5).floor(), 25.0, 200.0, 15.0]
}
/// The pretty JSON's frame in the Presets overlay.
fn pretty_box(size: (f32, f32)) -> [f32; 4] {
    [
        20.0,
        84.0,
        (size.0.ceil() - 40.0).max(1.0),
        (size.1.ceil() - FOOTER - 92.0).max(9.0),
    ]
}
fn picker_area(picker: &Picker, (w, h): (f32, f32)) -> (f32, f32, f32, f32) {
    let rows = picker.visible().len();
    (
        44.0,
        h.ceil() - FOOTER,
        ((w.ceil() - LIST_WIDTH) * 0.5).floor(),
        rows as f32 * LIST_ROW + 4.0,
    )
}

/// The footer's buttons.
pub(super) fn buttons(w: f32, h: f32, state: &State) -> Vec<Button> {
    let y = h.ceil() - 27.0;
    let make = |x: f32, width: f32, key: &str, enabled: bool, id: u8| Button {
        x,
        y,
        w: width,
        label: tr(key),
        enabled,
        id,
        icon: None,
        trailing: None,
        action: None,
    };
    let row = |specs: Vec<(&str, bool, u8)>, width: f32| {
        let total = specs.len() as f32 * (width + 4.0) - 4.0;
        let x0 = ((w.ceil() - total) * 0.5).floor();
        specs
            .into_iter()
            .enumerate()
            .map(|(i, (key, enabled, id))| {
                let b = make(x0 + i as f32 * (width + 4.0), width, key, enabled, id);
                match id {
                    CANCEL => b.does(crate::widgets::Action::Cancel),
                    DONE => b.does(crate::widgets::Action::Done),
                    _ => b,
                }
            })
            .collect()
    };
    match &state.overlay {
        Overlay::Presets(presets) => {
            let ok = state.presets_result(presets).is_ok();
            row(
                vec![
                    ("horizon.cubic.presets.defaults", true, DEFAULTS),
                    ("horizon.cubic.presets.copy", ok, COPY),
                    ("gui.done", ok, DONE),
                    ("gui.cancel", true, CANCEL),
                ],
                74.0,
            )
        }
        Overlay::Picker(picker) => row(
            vec![
                ("horizon.cubic.picker.all", picker.multi, ALL),
                ("horizon.cubic.picker.invert", picker.multi, INVERT),
                (
                    "gui.done",
                    !picker.selected.is_empty() || picker.multi || picker.target == Target::Biome,
                    DONE,
                ),
                ("gui.cancel", true, CANCEL),
            ],
            74.0,
        ),
        Overlay::None => {
            let pages = state.page_count();
            let mut specs = Vec::new();
            if pages > 1 {
                specs.push(("horizon.cubic.page.prev", state.page > 0, PREV));
                specs.push(("horizon.cubic.page.next", state.page + 1 < pages, NEXT));
            }
            specs.push(("createWorld.customize.presets", true, PRESETS));
            specs.push(("gui.done", true, DONE));
            specs.push(("gui.cancel", true, CANCEL));
            row(specs, 60.0)
        }
    }
}

// ---------------------------------------------------------------------------
// Input
// ---------------------------------------------------------------------------

/// A footer button was pressed.
pub(super) fn press(state: &mut State, id: u8) -> Exit {
    match std::mem::replace(&mut state.overlay, Overlay::None) {
        Overlay::None => match id {
            DONE => return Exit::Apply,
            CANCEL => return Exit::Cancel,
            PRESETS => state.open_presets(),
            PREV => state.switch_page(state.page.saturating_sub(1)),
            NEXT => state.switch_page(state.page + 1),
            _ => {}
        },
        Overlay::Presets(presets) => match id {
            DONE => {
                if state.load_json(&presets.compact).is_err() {
                    state.overlay = Overlay::Presets(presets);
                } else {
                    state.scroll = ui::Scroll::default();
                }
            }
            CANCEL => {}
            COPY => {
                let text = state.presets_result(&presets).unwrap_or_default();
                state.overlay = Overlay::Presets(presets);
                return Exit::Copy(text);
            }
            DEFAULTS => {
                let mut presets = presets;
                presets.compact = if state.kind.has_terrain_settings() {
                    CustomGeneratorSettings::defaults().to_json_compact()
                } else {
                    FlatGeneratorSettings::default().to_json_compact()
                };
                presets.focused = true;
                state.overlay = Overlay::Presets(presets);
            }
            _ => state.overlay = Overlay::Presets(presets),
        },
        Overlay::Picker(mut picker) => match id {
            DONE => state.apply_picker(&picker),
            CANCEL => {}
            ALL => {
                picker.selected = picker.rows.iter().map(|r| r.0.clone()).collect();
                state.overlay = Overlay::Picker(picker);
            }
            INVERT => {
                picker.selected = picker
                    .rows
                    .iter()
                    .map(|r| r.0.clone())
                    .filter(|id| !picker.selected.contains(id))
                    .collect();
                state.overlay = Overlay::Picker(picker);
            }
            _ => state.overlay = Overlay::Picker(picker),
        },
    }
    Exit::Stay
}

/// A press that no footer button took: the overlay's boxes and rows, the
/// page's scrollbar and controls. Whether a control was hit (it clicks).
pub(super) fn click(state: &mut State, cursor: (f32, f32), size: (f32, f32)) -> bool {
    let w = size.0.ceil();
    if matches!(state.overlay, Overlay::Picker(_)) {
        return picker_click(state, cursor, size);
    }
    if let Overlay::Presets(presets) = &mut state.overlay {
        let field = compact_box(w);
        presets.focused = ui::hovered(cursor, field[0], field[1], field[2], field[3]);
        return false;
    }
    settle(state, size);
    let widgets = page(state);
    let (top, bottom, left) = list_area(state, size);
    let content = content_height(&widgets);
    if state
        .scroll
        .press(cursor, left + WIDTH + 6.0, top, bottom, content)
    {
        return false;
    }
    if cursor.1 < top || cursor.1 >= bottom {
        return false;
    }
    let Some((at, widget)) = widget_at(&widgets, state.scroll.amount.trunc(), cursor, (left, top))
    else {
        return false;
    };
    if !widget.item.enabled {
        return false;
    }
    let (x, _, wd, _) = absolute(widget.rect, state.scroll.amount.trunc(), (left, top));
    let set = widget.item.set.clone();
    match (widget.item.kind, set) {
        (Kind::Slider(_), Some(set)) => {
            state.dragging = Some(at);
            set(state, ui::slider_value(x, wd, cursor.0));
        }
        (Kind::Button, Some(set)) => set(state, 0.0),
        _ => return false,
    }
    true
}
/// A widget's rectangle on the screen, `offset` pixels scrolled.
fn absolute(rect: [f32; 4], offset: f32, (left, top): (f32, f32)) -> (f32, f32, f32, f32) {
    (
        left + rect[0],
        top + 2.0 + rect[1] - offset,
        rect[2],
        rect[3],
    )
}
/// Keeps the scroll amount inside the open page, which an edit may have
/// shortened.
fn settle(state: &mut State, size: (f32, f32)) {
    let (top, bottom, _) = list_area(state, size);
    let content = content_height(&page(state));
    state.scroll.clamp(top, bottom, content);
}
fn widget_at(
    page: &Page,
    offset: f32,
    cursor: (f32, f32),
    origin: (f32, f32),
) -> Option<(usize, &Widget)> {
    page.widgets.iter().enumerate().find(|(_, w)| {
        matches!(w.item.kind, Kind::Button | Kind::Slider(_)) && {
            let (x, y, wd, h) = absolute(w.rect, offset, origin);
            ui::hovered(cursor, x, y, wd, h)
        }
    })
}
fn picker_click(state: &mut State, cursor: (f32, f32), size: (f32, f32)) -> bool {
    let Overlay::Picker(picker) = &mut state.overlay else {
        return false;
    };
    let field = search_box(size.0.ceil());
    picker.focused = ui::hovered(cursor, field[0], field[1], field[2], field[3]);
    let (top, bottom, left, content) = picker_area(picker, size);
    if picker
        .scroll
        .press(cursor, left + LIST_WIDTH + 6.0, top, bottom, content)
    {
        return false;
    }
    if cursor.0 < left || cursor.0 >= left + LIST_WIDTH || cursor.1 < top || cursor.1 >= bottom {
        return false;
    }
    let row = ((cursor.1 - top - 2.0 + picker.scroll.amount.trunc()) / LIST_ROW).floor();
    let visible = picker.visible();
    let Some(index) = (row >= 0.0)
        .then(|| visible.get(row as usize).copied())
        .flatten()
    else {
        return false;
    };
    let id = picker.rows[index].0.clone();
    if picker.multi {
        if !picker.selected.remove(&id) {
            picker.selected.insert(id);
        }
    } else {
        picker.selected = [id].into_iter().collect();
    }
    true
}

pub(super) fn drag(state: &mut State, cursor: (f32, f32), size: (f32, f32)) {
    match &mut state.overlay {
        Overlay::Picker(picker) => {
            let (top, bottom, _, content) = picker_area(picker, size);
            picker.scroll.drag(cursor.1, top, bottom, content);
            return;
        }
        Overlay::Presets(_) => return,
        Overlay::None => {}
    }
    settle(state, size);
    let (top, bottom, left) = list_area(state, size);
    if state.scroll.held() {
        let content = content_height(&page(state));
        state.scroll.drag(cursor.1, top, bottom, content);
        return;
    }
    let Some(at) = state.dragging else { return };
    let widgets = page(state);
    let Some(widget) = widgets.widgets.get(at) else {
        return;
    };
    let (x, _, wd, _) = absolute(widget.rect, state.scroll.amount.trunc(), (left, top));
    if let Some(set) = widget.item.set.clone() {
        set(state, ui::slider_value(x, wd, cursor.0));
    }
}
pub(super) fn release(state: &mut State) {
    state.scroll.release();
    if let Overlay::Picker(picker) = &mut state.overlay {
        picker.scroll.release();
    }
    if state.dragging.take().is_some() {
        // A dragged probability point may have passed its neighbour: the
        // functions are piecewise linear in order of Y.
        for lake in &mut state.settings.lakes {
            for f in [&mut lake.surface_probability, &mut lake.main_probability] {
                f.0.sort_by(|a, b| a.y.total_cmp(&b.y));
            }
        }
        state.flat.sort_by_key(|l| l.from_y);
    }
}
pub(super) fn wheel(state: &mut State, delta: f32, cursor: (f32, f32), size: (f32, f32)) {
    match &mut state.overlay {
        Overlay::Picker(picker) => {
            let (top, bottom, _, content) = picker_area(picker, size);
            picker.scroll.wheel(delta, LIST_ROW, top, bottom, content);
        }
        Overlay::Presets(presets) => {
            let frame = pretty_box(size);
            if ui::hovered(cursor, frame[0], frame[1], frame[2], frame[3]) {
                let step = (-delta * 3.0) as i64;
                presets.top_line = (presets.top_line as i64 + step).max(0) as usize;
            }
        }
        Overlay::None => {
            let (top, bottom, _) = list_area(state, size);
            let content = content_height(&page(state));
            state.scroll.wheel(delta, 20.0, top, bottom, content);
        }
    }
}

/// The focus landed on a row of a single-choice picker (the controller): it is
/// chosen. A multi-choice row toggles only on a press, never on focus.
pub(super) fn focus_row(state: &mut State, cursor: (f32, f32), size: (f32, f32)) -> bool {
    matches!(&state.overlay, Overlay::Picker(p) if !p.multi) && picker_click(state, cursor, size)
}

// ---------------------------------------------------------------------------
// Drawing
// ---------------------------------------------------------------------------

const RED: u32 = 0xFFFF5555;
const GRAY: u32 = 0xFFAAAAAA;

pub(super) fn draw(
    title: &Title,
    atlas: &HudAtlas,
    batch: &mut HudBatch,
    cursor: (f32, f32),
    state: &State,
) {
    let (w, h) = (batch.width.ceil(), batch.height.ceil());
    ui::menu_background(batch, atlas, 0.0, 0.0, w, h, false);
    match &state.overlay {
        Overlay::Presets(presets) => {
            return draw_presets(title, atlas, batch, state, presets, (w, h));
        }
        Overlay::Picker(picker) => return draw_picker(title, atlas, batch, cursor, picker, (w, h)),
        Overlay::None => {}
    }
    let title_text = match state.page_id() {
        Some(id) => Text::translate(
            "horizon.cubic.page.title",
            &[
                Text::literal((state.page + 1).to_string()),
                Text::literal(state.page_count().to_string()),
                Text::translate(id.name(), &[]),
            ],
        )
        .plain(),
        None => tr("horizon.cubic.flat.title"),
    };
    super::centered(atlas, batch, &title_text, 8.0, u32::MAX);
    let preview_on = state.page_id() == Some(PageId::Terrain) && state.show_preview;
    if preview_on {
        draw_preview(atlas, batch, state, w);
    }
    let widgets = page(state);
    let (top, bottom, left) = list_area(state, (w, h));
    let content = content_height(&widgets);
    let mut scroll = state.scroll;
    scroll.clamp(top, bottom, content);
    let offset = scroll.amount.trunc();
    ui::list_background(batch, atlas, top, bottom, false);
    ui::separators(batch, atlas, top, bottom, false);
    let first = batch.vertices.len();
    let inside = cursor.1 >= top && cursor.1 < bottom;
    for (at, widget) in widgets.widgets.iter().enumerate() {
        let (x, y, wd, ht) = absolute(widget.rect, offset, (left, top));
        if y + ht < top || y > bottom {
            continue;
        }
        let item = &widget.item;
        let hover = inside && item.enabled && ui::hovered(cursor, x, y, wd, ht);
        match item.kind {
            Kind::Heading => {
                batch.text(atlas, &item.label, x, y + 1.0, u32::MAX, true);
            }
            Kind::Note => {
                batch.text(atlas, &item.label, x, y, GRAY, true);
            }
            Kind::Button => {
                ui::button(
                    batch,
                    atlas,
                    x,
                    y,
                    wd,
                    20.0,
                    &Text::literal(item.label.clone()),
                    item.enabled,
                    hover,
                );
            }
            Kind::Slider(pos) => {
                ui::slider(
                    batch,
                    atlas,
                    x,
                    y,
                    wd,
                    &Text::literal(item.label.clone()),
                    pos,
                    hover,
                    item.enabled,
                    state.dragging == Some(at),
                );
            }
        }
    }
    batch.clip_since(first, 0.0, top, w, bottom);
    ui::scrollbar(
        batch,
        atlas,
        left + WIDTH + 6.0,
        top,
        bottom,
        content,
        scroll.amount,
    );
}

/// The cross-section in its strip under the header, one pixel a cell, and a
/// note of what it shows.
fn draw_preview(atlas: &HudAtlas, batch: &mut HudBatch, state: &State, w: f32) {
    state.preview.poll(&state.settings, state.axis_z);
    let (x0, y0) = (
        ((w - cubic_preview::COLUMNS as f32) * 0.5).floor(),
        HEADER + 4.0,
    );
    let (pw, ph) = (cubic_preview::COLUMNS as f32, cubic_preview::ROWS as f32);
    batch.fill(
        atlas,
        x0 - 1.0,
        y0 - 1.0,
        pw + 2.0,
        ph + 2.0,
        rgba(0, 0, 0, 255),
    );
    match state.preview.image() {
        Some(Ok(image)) => {
            for row in 0..cubic_preview::ROWS {
                let cells = image.row(row);
                let mut start = 0;
                while start < cells.len() {
                    let cell = cells[start];
                    let mut end = start + 1;
                    while end < cells.len() && cells[end] == cell {
                        end += 1;
                    }
                    batch.fill(
                        atlas,
                        x0 + start as f32,
                        y0 + row as f32,
                        (end - start) as f32,
                        1.0,
                        cell.color(),
                    );
                    start = end;
                }
            }
            let note = Text::translate(
                "horizon.cubic.terrain.preview_note",
                &[
                    Text::literal(image.top_y.to_string()),
                    Text::literal((image.top_y - cubic_preview::HEIGHT_BLOCKS).to_string()),
                ],
            )
            .plain();
            batch.text(atlas, &note, x0 + pw + 6.0, y0, GRAY, true);
        }
        Some(Err(why)) => {
            batch.text(
                atlas,
                &tr("horizon.cubic.terrain.preview_failed"),
                x0 + 4.0,
                y0 + 4.0,
                RED,
                true,
            );
            batch.text(atlas, &why, x0 + pw + 6.0, y0, GRAY, true);
        }
        None => {
            batch.fill(atlas, x0, y0, pw, ph, rgba(40, 40, 40, 255));
            batch.text(
                atlas,
                &tr("horizon.cubic.terrain.preview_wait"),
                x0 + 4.0,
                y0 + 4.0,
                GRAY,
                true,
            );
        }
    }
}

fn draw_presets(
    title: &Title,
    atlas: &HudAtlas,
    batch: &mut HudBatch,
    state: &State,
    presets: &Presets,
    (w, h): (f32, f32),
) {
    super::centered(
        atlas,
        batch,
        &tr("horizon.cubic.presets.title"),
        8.0,
        u32::MAX,
    );
    batch.text(
        atlas,
        &tr("horizon.cubic.presets.compact"),
        20.0,
        26.0,
        GRAY,
        true,
    );
    let field = compact_box(w);
    title.caret.borrow_mut().draw(
        batch,
        atlas,
        (COMPACT, &presets.compact, 8_000_000),
        field,
        presets.focused,
        title.started,
    );
    crate::widgets::name_last_edit_box(|| tr("horizon.cubic.presets.compact"));
    let result = state.presets_result(presets);
    batch.text(
        atlas,
        &tr("horizon.cubic.presets.pretty"),
        20.0,
        70.0,
        GRAY,
        true,
    );
    if result.is_err() {
        let text = tr("horizon.cubic.presets.invalid");
        let width = HudBatch::text_width(atlas, &text);
        batch.text(atlas, &text, w - 20.0 - width, 70.0, RED, true);
    }
    let frame = pretty_box((w, h));
    batch.fill(
        atlas,
        frame[0] - 1.0,
        frame[1] - 1.0,
        frame[2] + 2.0,
        frame[3] + 2.0,
        rgba(160, 160, 160, 255),
    );
    batch.fill(
        atlas,
        frame[0],
        frame[1],
        frame[2],
        frame[3],
        rgba(0, 0, 0, 255),
    );
    if let Ok(pretty) = &result {
        let lines = ((frame[3] - 4.0) / 9.0).floor().max(1.0) as usize;
        let total = pretty.lines().count();
        let first = presets.top_line.min(total.saturating_sub(lines));
        let start = batch.vertices.len();
        for (i, line) in pretty.lines().skip(first).take(lines).enumerate() {
            batch.text(
                atlas,
                line,
                frame[0] + 3.0,
                frame[1] + 2.0 + i as f32 * 9.0,
                u32::MAX,
                false,
            );
        }
        batch.clip_since(
            start,
            frame[0],
            frame[1],
            frame[0] + frame[2],
            frame[1] + frame[3],
        );
    }
}

fn draw_picker(
    title: &Title,
    atlas: &HudAtlas,
    batch: &mut HudBatch,
    cursor: (f32, f32),
    picker: &Picker,
    (w, h): (f32, f32),
) {
    let key = match picker.target {
        Target::Biome | Target::OreBiomes(_) | Target::LakeBiomes(_) => {
            "horizon.cubic.picker.biome"
        }
        _ => "horizon.cubic.picker.block",
    };
    super::centered(atlas, batch, &tr(key), 8.0, u32::MAX);
    let field = search_box(w);
    title.caret.borrow_mut().draw(
        batch,
        atlas,
        (SEARCH, &picker.query, 50),
        field,
        picker.focused,
        title.started,
    );
    crate::widgets::name_last_edit_box(|| tr("createWorld.customize.buffet.search"));
    if picker.query.is_empty() && !picker.focused {
        let box_ = ui::EditBox::new(field[0], field[1], field[2], field[3]);
        batch.text_styled(
            atlas,
            &Text::translate("createWorld.customize.buffet.search", &[])
                .with_color(crate::text::DARK_GRAY),
            box_.text_x(),
            box_.text_y(),
            u32::MAX,
            true,
        );
    }
    let (top, bottom, left, content) = picker_area(picker, (w, h));
    ui::list_background(batch, atlas, top, bottom, false);
    ui::separators(batch, atlas, top, bottom, false);
    let first = batch.vertices.len();
    let offset = picker.scroll.amount.trunc();
    for (at, index) in picker.visible().into_iter().enumerate() {
        let (id, name) = &picker.rows[index];
        let y = top + 2.0 + at as f32 * LIST_ROW - offset;
        if y + LIST_ROW < top || y > bottom {
            continue;
        }
        let selected = picker.selected.contains(id);
        let over = cursor.1 >= top
            && cursor.1 < bottom
            && ui::hovered(cursor, left, y, LIST_WIDTH, LIST_ROW);
        if selected {
            batch.fill(
                atlas,
                left,
                y,
                LIST_WIDTH,
                LIST_ROW,
                rgba(255, 255, 255, 70),
            );
        } else if over {
            batch.fill(
                atlas,
                left,
                y,
                LIST_WIDTH,
                LIST_ROW,
                rgba(255, 255, 255, 25),
            );
        }
        if picker.multi {
            batch.fill(
                atlas,
                left + 4.0,
                y + 5.0,
                10.0,
                10.0,
                rgba(200, 200, 200, 255),
            );
            batch.fill(atlas, left + 5.0, y + 6.0, 8.0, 8.0, rgba(0, 0, 0, 255));
            if selected {
                batch.fill(atlas, left + 6.0, y + 7.0, 6.0, 6.0, rgba(85, 255, 85, 255));
            }
        }
        let x = left + if picker.multi { 20.0 } else { 5.0 };
        batch.text(atlas, name, x, y + 6.0, u32::MAX, true);
    }
    batch.clip_since(first, left, top, left + LIST_WIDTH, bottom);
    ui::scrollbar(
        batch,
        atlas,
        left + LIST_WIDTH + 6.0,
        top,
        bottom,
        content,
        picker.scroll.amount,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(kind: CubicKind, json: &str) -> State {
        State::new(Path::new("/nonexistent/horizon-test-resources"), kind, json)
    }
    const SIZE: (f32, f32) = (480.0, 270.0);

    fn sliders(state: &State) -> Vec<usize> {
        page(state)
            .widgets
            .iter()
            .enumerate()
            .filter(|(_, w)| matches!(w.item.kind, Kind::Slider(_)))
            .map(|(i, _)| i)
            .collect()
    }
    /// Sets widget `index` of the open page to `pos`, as a drag does.
    fn set(state: &mut State, index: usize, pos: f32) {
        let set = page(state).widgets[index].item.set.clone().unwrap();
        set(state, pos);
    }
    fn buttons_of(state: &State) -> Vec<Set> {
        page(state)
            .widgets
            .iter()
            .filter(|w| w.item.kind == Kind::Button)
            .filter_map(|w| w.item.set.clone())
            .collect()
    }
    fn presets_text(state: &mut State, text: &str) {
        if !state.presets_open() {
            assert_eq!(press(state, PRESETS), Exit::Stay);
        }
        if let Overlay::Presets(p) = &mut state.overlay {
            p.compact = text.to_owned();
        }
    }
    fn presets_result(state: &State) -> Result<String, String> {
        let Overlay::Presets(presets) = state.overlay.clone() else {
            panic!("the Presets overlay is open")
        };
        state.presets_result(&presets)
    }
    /// Opens page `id` of the world type's pages.
    fn open(state: &mut State, id: PageId) {
        let at = state
            .pages()
            .iter()
            .position(|p| *p == id)
            .unwrap_or_else(|| panic!("{:?} has a {id:?} page", state.kind));
        state.switch_page(at);
        assert_eq!(state.page_id(), Some(id));
    }
    /// The first control of the open page that, set to `pos`, changes what
    /// `look` reads of the settings.
    fn control<T: PartialEq>(
        state: &State,
        pos: f32,
        look: impl Fn(&CustomGeneratorSettings) -> T,
    ) -> Option<usize> {
        let before = look(&state.settings);
        let widgets = page(state).widgets;
        (0..widgets.len())
            .filter(|&i| widgets[i].item.set.is_some())
            .find(|&i| {
                let mut probe = state.clone();
                set(&mut probe, i, pos);
                look(&probe.settings) != before
            })
    }
    fn enabled(state: &State, index: usize) -> bool {
        page(state).widgets[index].item.enabled
    }
    fn spread_of(c: &CustomGeneratorSettings, name: &str) -> (i32, i32, f32) {
        match c.structure_placement(name) {
            Some(VanillaPlacement::Spread {
                spacing,
                separation,
                frequency,
            }) => (spacing, separation, frequency),
            other => panic!("{name} is a random spread: {other:?}"),
        }
    }
    fn rings_of(c: &CustomGeneratorSettings, name: &str) -> (i32, i32, i32) {
        match c.structure_placement(name) {
            Some(VanillaPlacement::Rings {
                distance,
                spread,
                count,
            }) => (distance, spread, count),
            other => panic!("{name} is concentric rings: {other:?}"),
        }
    }
    /// The settings validate and come back the same through `cubicgen.json`
    /// and through a new screen.
    fn assert_round_trips(s: &State) {
        s.settings.validate().unwrap();
        let json = s.json();
        assert_eq!(
            CustomGeneratorSettings::from_json(&json).unwrap(),
            s.settings
        );
        let again = state(s.kind, &json);
        assert_eq!(again.settings, s.settings, "the screen reopens on them");
    }

    #[test]
    fn presets_json_round_trips_through_the_overlay() {
        let mut s = state(CubicKind::Custom, "");
        let expected = CustomGeneratorSettings::defaults();
        assert_eq!(s.settings, expected);
        assert_eq!(press(&mut s, PRESETS), Exit::Stay);
        let Overlay::Presets(open) = &s.overlay else {
            panic!()
        };
        assert_eq!(open.compact, expected.to_json_compact());
        // Export: the pretty text is the settings as `cubicgen.json` keeps them.
        assert_eq!(presets_result(&s).unwrap(), expected.to_json_pretty());
        // Import: an edited preset applies on Done, in CWG's lenient syntax.
        presets_text(
            &mut s,
            "/* mine */ {\"dungeonCount\": 33, \"biomeSize\": 6,}",
        );
        assert_eq!(press(&mut s, DONE), Exit::Stay);
        assert!(!s.overlay_open());
        assert_eq!((s.settings.dungeon_count, s.settings.biome_size), (33, 6));
        // What was not mentioned is CWG's `new CustomGeneratorSettings()`: no ores.
        assert!(s.settings.standard_ores.is_empty());
    }

    #[test]
    fn invalid_json_is_flagged_and_done_keeps_the_overlay() {
        let mut s = state(CubicKind::Custom, "");
        presets_text(&mut s, "{ this is not json");
        assert!(presets_result(&s).is_err());
        let before = s.settings.clone();
        let done = buttons(480.0, 270.0, &s)
            .into_iter()
            .find(|b| b.id == DONE)
            .unwrap();
        assert!(!done.enabled, "Done waits for JSON that parses");
        press(&mut s, DONE);
        assert!(s.presets_open() && s.settings == before);
        // Fixing the text re-enables it, and Cancel drops edits.
        presets_text(&mut s, "{}");
        assert!(presets_result(&s).is_ok());
        press(&mut s, CANCEL);
        assert!(!s.overlay_open() && s.settings == before);
    }

    #[test]
    fn flat_layers_round_trip_through_presets() {
        let mut s = state(CubicKind::Flat, "");
        assert_eq!(s.flat_settings(), FlatGeneratorSettings::default());
        assert_eq!(s.page_count(), 1);
        presets_text(
            &mut s,
            r#"{"layers":{"0":{"fromY":0,"toY":3,"blockState":"sand"}},"version":1}"#,
        );
        press(&mut s, DONE);
        assert_eq!(s.flat.len(), 1);
        assert_eq!((s.flat[0].from_y, s.flat[0].to_y), (0, 3));
        assert_eq!(s.flat[0].block_state.name, "minecraft:sand");
        let again = FlatGeneratorSettings::from_json(&s.json()).unwrap();
        assert_eq!(again, s.flat_settings());
    }

    #[test]
    fn dragging_a_slider_writes_its_setting_and_the_track_maps_to_the_ends() {
        let mut s = state(CubicKind::Custom, "");
        // The first slider of the Basic page is the dungeon count, 1 to 100.
        let first = sliders(&s)[0];
        set(&mut s, first, 1.0);
        assert_eq!(s.settings.dungeon_count, 100);
        set(&mut s, first, 0.0);
        assert_eq!(s.settings.dungeon_count, 1);
        // A real press on the track: the right end of the widget.
        let widgets = page(&s);
        let w = &widgets.widgets[first];
        let (top, _, left) = list_area(&s, SIZE);
        let (x, y, wd, _) = absolute(w.rect, 0.0, (left, top));
        assert!(click(&mut s, (x + wd - 1.0, y + 10.0), SIZE));
        assert_eq!(s.settings.dungeon_count, 100);
        assert_eq!(s.dragging, Some(first));
        drag(&mut s, (x + 1.0, y + 10.0), SIZE);
        assert_eq!(s.settings.dungeon_count, 1);
        release(&mut s);
        assert_eq!(s.dragging, None);
    }

    #[test]
    fn a_period_slider_writes_the_inverse_as_the_frequency() {
        let mut s = state(CubicKind::Custom, "");
        open(&mut s, PageId::Terrain);
        let found = sliders(&s).into_iter().find(|&i| {
            let mut probe = s.clone();
            let before = probe.settings.depth_noise_frequency_x;
            set(&mut probe, i, 0.5);
            probe.settings.depth_noise_frequency_x != before
        });
        let i = found.expect("a slider sets the depth noise frequency");
        set(&mut s, i, 0.5);
        // The track runs 2^2 to 2^13 blocks: its middle is 2^7.5.
        let period = 2f64.powf(7.5);
        assert!((s.settings.depth_noise_frequency_x - 1.0 / period).abs() < 1e-12);
        assert!(s.lock_xz);
        assert_eq!(
            s.settings.depth_noise_frequency_z, s.settings.depth_noise_frequency_x,
            "Z follows X while the periods are tied"
        );
    }

    #[test]
    fn the_lock_ties_the_expected_heights_to_the_height_sliders() {
        let mut s = state(CubicKind::Custom, "");
        assert!(s.lock);
        s.edit(|c| c.height_offset = 100.0);
        s.edit(|c| c.height_factor = 30.0);
        assert_eq!(s.settings.expected_base_height, 100.0);
        assert_eq!(s.settings.expected_height_variation, 30.0);
        assert_eq!(
            s.settings.actual_height,
            cubic_map::actual_height(&s.settings)
        );
        s.lock = false;
        s.edit(|c| c.height_offset = 10.0);
        assert_eq!(
            s.settings.expected_base_height, 100.0,
            "unlocked: they stay"
        );
        // Preset JSON whose heights differ from the Lock's opens unlocked.
        let mut custom = CustomGeneratorSettings::defaults();
        custom.expected_base_height = 5.0;
        let s = state(CubicKind::Custom, &custom.to_json_compact());
        assert!(!s.lock);
    }

    #[test]
    fn a_picker_sets_the_biome_and_a_multi_picker_restricts_an_ore() {
        let mut s = state(CubicKind::Custom, "");
        s.open_picker(Target::Biome);
        let Overlay::Picker(p) = &mut s.overlay else {
            panic!()
        };
        assert_eq!(p.rows[0].0, "", "the first row is All");
        p.selected = ["minecraft:desert".to_owned()].into_iter().collect();
        press(&mut s, DONE);
        assert_eq!(s.settings.biome, "minecraft:desert");
        // An ore with no restriction starts with every biome chosen; leaving one
        // out restricts it to the rest.
        s.edit(|c| c.standard_ores.push(StandardOreConfig::default()));
        let last = s.settings.standard_ores.len() - 1;
        s.open_picker(Target::OreBiomes(last));
        let Overlay::Picker(p) = &mut s.overlay else {
            panic!()
        };
        assert!(p.multi && p.selected.len() == p.rows.len());
        p.selected.remove("minecraft:plains");
        press(&mut s, DONE);
        let restricted = s.settings.standard_ores[last].biomes.clone().unwrap();
        assert!(
            !restricted.is_empty() && !restricted.iter().any(|b| b.name() == "minecraft:plains")
        );
    }

    #[test]
    fn ore_height_sliders_have_open_ends_and_hundredth_steps() {
        let mut s = state(CubicKind::Custom, "");
        s.edit(|c| c.standard_ores.push(StandardOreConfig::default()));
        open(&mut s, PageId::Ores);
        let last = s.settings.standard_ores.len() - 1;
        // The new ore is unbounded both ways; a slider at 0.75 of the track
        // (-2 to 2) is height 1.0, and the end is open again.
        let min_slider = sliders(&s)
            .into_iter()
            .find(|&i| {
                let mut probe = s.clone();
                set(&mut probe, i, 0.75);
                probe.settings.standard_ores[last].min_height != f64::NEG_INFINITY
            })
            .expect("a slider sets the minimum height");
        set(&mut s, min_slider, 0.75);
        assert_eq!(s.settings.standard_ores[last].min_height, 1.0);
        set(&mut s, min_slider, 0.0);
        assert_eq!(s.settings.standard_ores[last].min_height, f64::NEG_INFINITY);
    }

    #[test]
    fn adding_and_deleting_entries_edits_the_lists() {
        let mut s = state(CubicKind::Custom, "");
        let (ores, lakes, caves) = (
            s.settings.standard_ores.len(),
            s.settings.lakes.len(),
            s.settings.caves.len(),
        );
        // Each page's first button is its Add.
        for id in [PageId::Ores, PageId::Lakes, PageId::Caves] {
            open(&mut s, id);
            let add = buttons_of(&s)[0].clone();
            add(&mut s, 0.0);
        }
        assert_eq!(s.settings.standard_ores.len(), ores + 1);
        assert_eq!(s.settings.lakes.len(), lakes + 1);
        assert_eq!(s.settings.caves.len(), caves + 1);
        // On the caves page the buttons are Add, then per cave its block,
        // Delete and Replaceable blocks: the new cave's Delete is the third.
        open(&mut s, PageId::Caves);
        let buttons = buttons_of(&s);
        buttons[2].clone()(&mut s, 0.0);
        assert_eq!(s.settings.caves.len(), caves);
    }

    #[test]
    fn the_footer_buttons_have_distinct_ids_on_every_page_and_overlay() {
        let mut s = state(CubicKind::Custom, "");
        let mut seen = Vec::new();
        for page in 0..s.page_count() {
            s.switch_page(page);
            seen.push(buttons(480.0, 270.0, &s));
        }
        assert_eq!(press(&mut s, PRESETS), Exit::Stay);
        seen.push(buttons(480.0, 270.0, &s));
        s.overlay = Overlay::None;
        s.open_picker(Target::Biome);
        seen.push(buttons(480.0, 270.0, &s));
        seen.push(buttons(480.0, 270.0, &state(CubicKind::Flat, "")));
        for set in seen {
            let mut ids: Vec<u8> = set.iter().map(|b| b.id).collect();
            let n = ids.len();
            ids.sort_unstable();
            ids.dedup();
            assert_eq!(ids.len(), n, "a button id repeats");
            for b in &set {
                assert!(b.x >= 0.0 && b.x + b.w <= 480.0, "a button fits the screen");
            }
        }
    }

    #[test]
    fn prev_and_next_walk_the_pages_and_stop_at_the_ends() {
        let enabled = |s: &State, id: u8| {
            buttons(480.0, 270.0, s)
                .into_iter()
                .find(|b| b.id == id)
                .map(|b| b.enabled)
        };
        // Hybrid has CWG's five pages, Custom Cubic Structures too.
        for (kind, pages) in [(CubicKind::Hybrid, 5), (CubicKind::Custom, 6)] {
            let mut s = state(kind, "");
            assert_eq!(s.page_count(), pages);
            assert_eq!(
                (enabled(&s, PREV), enabled(&s, NEXT)),
                (Some(false), Some(true))
            );
            for expected in 1..pages {
                press(&mut s, NEXT);
                assert_eq!(s.page, expected);
            }
            assert_eq!(
                s.page_id(),
                Some(PageId::Terrain),
                "{kind:?} ends on terrain"
            );
            assert_eq!(enabled(&s, NEXT), Some(false));
            press(&mut s, PREV);
            assert_eq!(s.page, pages - 2);
        }
        // A Flat Cubic world has the layer page only, with no Prev or Next.
        assert_eq!(enabled(&state(CubicKind::Flat, ""), PREV), None);
    }

    #[test]
    fn done_hands_back_the_settings_as_pretty_json() {
        let mut s = state(CubicKind::Custom, "");
        s.edit(|c| c.biome_size = 7);
        let json = s.json();
        assert!(json.contains('\n'));
        assert_eq!(
            CustomGeneratorSettings::from_json(&json)
                .unwrap()
                .biome_size,
            7
        );
        assert_eq!(press(&mut s, DONE), Exit::Apply);
        assert_eq!(press(&mut s, CANCEL), Exit::Cancel);
    }

    /// Reset to vanilla on the Structures page: after the help note.
    const RESET: usize = 1;
    /// Where structure set `name`'s block starts on the Structures page: after
    /// the help note and Reset to vanilla, a switch and three sliders a set.
    fn switch_of(name: &str) -> usize {
        let i = OVERWORLD_STRUCTURE_SETS
            .iter()
            .position(|v| v.name == name)
            .unwrap_or_else(|| panic!("{name} is an overworld set"));
        2 + 4 * i
    }
    fn slider_pos(state: &State, index: usize) -> f32 {
        match page(state).widgets[index].item.kind {
            Kind::Slider(pos) => pos,
            _ => panic!("widget {index} is a slider"),
        }
    }

    #[test]
    fn structures_is_custom_cubics_second_page_and_scrolls_through_every_set() {
        let mut s = state(CubicKind::Custom, "");
        press(&mut s, NEXT);
        assert_eq!(s.page_id(), Some(PageId::Structures));
        // A block a set, in the settings' order: its switch, then three sliders.
        let widgets = page(&s).widgets;
        assert_eq!(widgets.len(), 2 + 4 * OVERWORLD_STRUCTURE_SETS.len());
        assert!(widgets[RESET].item.kind == Kind::Button);
        for v in OVERWORLD_STRUCTURE_SETS {
            let at = switch_of(v.name);
            assert!(widgets[at].item.kind == Kind::Button, "{}", v.name);
            for slider in &widgets[at + 1..at + 4] {
                assert!(matches!(slider.item.kind, Kind::Slider(_)), "{}", v.name);
            }
        }
        // Eighteen blocks overflow the list, which scrolls; at its end a press
        // lands on the last set's switch.
        let (top, bottom, left) = list_area(&s, SIZE);
        assert!(content_height(&page(&s)) > 3.0 * (bottom - top));
        wheel(&mut s, -1000.0, (240.0, 100.0), SIZE);
        assert!(s.scroll.amount > 0.0);
        let at = switch_of("minecraft:strongholds");
        let (x, y, _, h) = absolute(
            page(&s).widgets[at].rect,
            s.scroll.amount.trunc(),
            (left, top),
        );
        assert!(y >= top && y + h <= bottom, "the last block is in view");
        assert!(click(&mut s, (x + 5.0, y + 5.0), SIZE));
        assert!(!s.settings.structure_enabled("minecraft:strongholds"));
        // Hybrid's structures are vanilla's, from its vanilla band: no page.
        let hybrid = state(CubicKind::Hybrid, "");
        assert!(!hybrid.pages().contains(&PageId::Structures));
        assert_eq!(hybrid.page_count(), 5);
    }

    #[test]
    fn a_structure_switch_turns_its_set_off_and_on() {
        let mut s = state(CubicKind::Custom, "");
        open(&mut s, PageId::Structures);
        for v in OVERWORLD_STRUCTURE_SETS {
            let at = switch_of(v.name);
            set(&mut s, at, 0.0);
            assert!(!s.settings.structure_enabled(v.name), "{} is off", v.name);
            assert!(
                OVERWORLD_STRUCTURE_SETS
                    .iter()
                    .filter(|o| o.name != v.name)
                    .all(|o| s.settings.structure_enabled(o.name)),
                "only {} is off",
                v.name
            );
            // Its sliders grey out; its switch and Reset to vanilla do not.
            assert!((1..4).all(|i| !enabled(&s, at + i)), "{}", v.name);
            assert!(enabled(&s, at) && enabled(&s, RESET));
            assert_round_trips(&s);
            set(&mut s, at, 0.0);
            assert!(s.settings.structure_enabled(v.name), "{} is back", v.name);
            assert!(enabled(&s, at + 1));
        }
        assert!(s.settings.structure_sets.is_empty(), "nothing is left over");
        assert!(!enabled(&s, RESET), "the settings are vanilla's");
        // A CWG toggle that is off shows its sets off; switching one of them on
        // brings back that one only.
        let mut s = state(CubicKind::Custom, r#"{"temples": false}"#);
        open(&mut s, PageId::Structures);
        let temples = [
            "minecraft:desert_pyramids",
            "minecraft:jungle_temples",
            "minecraft:swamp_huts",
            "minecraft:igloos",
        ];
        assert!(temples.iter().all(|t| !s.settings.structure_enabled(t)));
        set(&mut s, switch_of(temples[0]), 0.0);
        assert!(s.settings.temples && s.settings.structure_enabled(temples[0]));
        assert!(
            temples[1..]
                .iter()
                .all(|t| !s.settings.structure_enabled(t))
        );
        assert_round_trips(&s);
    }

    #[test]
    fn structure_sliders_write_the_placement_and_keep_separation_below_spacing() {
        let mut s = state(CubicKind::Custom, "");
        open(&mut s, PageId::Structures);
        let spacing_track = Mapping::Int {
            min: SPACING.0,
            max: SPACING.1,
        };
        let villages = "minecraft:villages";
        let at = switch_of(villages);
        let (chance, spacing, separation) = (at + 1, at + 2, at + 3);
        // The sliders start at vanilla's numbers.
        assert_eq!(slider_pos(&s, spacing), spacing_track.pos(34.0));
        assert_eq!(slider_pos(&s, chance), 1.0);
        // Only what differs from vanilla is stored.
        set(&mut s, spacing, 1.0);
        assert_eq!(spread_of(&s.settings, villages), (256, 8, 1.0));
        let entry = &s.settings.structure_sets[villages];
        assert_eq!(
            (entry.spacing, entry.separation, entry.frequency),
            (Some(256), None, None)
        );
        // The separation runs up to the spacing less one.
        set(&mut s, separation, 1.0);
        assert_eq!(spread_of(&s.settings, villages), (256, 255, 1.0));
        assert_round_trips(&s);
        // A smaller spacing pulls the separation below it; at 1 it is 0 and
        // its slider greys out.
        set(&mut s, spacing, spacing_track.pos(10.0));
        assert_eq!(spread_of(&s.settings, villages), (10, 9, 1.0));
        assert_round_trips(&s);
        set(&mut s, spacing, 0.0);
        assert_eq!(spread_of(&s.settings, villages), (1, 0, 1.0));
        assert!(!enabled(&s, separation) && enabled(&s, spacing));
        assert_round_trips(&s);
        // Chance is cubic: mineshafts' 0.4% sits well inside the track, which
        // steps by 0.1% below 5% and by whole percents above.
        let mineshafts = "minecraft:mineshafts";
        let chance = switch_of(mineshafts) + 1;
        assert!((slider_pos(&s, chance) - 0.004f32.cbrt()).abs() < 1e-6);
        set(&mut s, chance, 0.2);
        assert_eq!(spread_of(&s.settings, mineshafts).2, 0.008);
        set(&mut s, chance, 0.5);
        assert_eq!(spread_of(&s.settings, mineshafts).2, 0.13);
        assert_round_trips(&s);
        set(&mut s, chance, Mapping::Cubic { max: 1.0 }.pos(0.004));
        assert_eq!(spread_of(&s.settings, mineshafts).2, 0.004);
        assert!(
            !s.settings.structure_sets.contains_key(mineshafts),
            "back at vanilla's chance, nothing is stored"
        );
        // The strongholds' rings.
        let strongholds = "minecraft:strongholds";
        let at = switch_of(strongholds);
        assert_eq!(rings_of(&s.settings, strongholds), (32, 3, 128));
        set(&mut s, at + 1, 1.0);
        set(&mut s, at + 2, 0.0);
        set(&mut s, at + 3, 1.0);
        assert_eq!(
            rings_of(&s.settings, strongholds),
            (RING_DISTANCE.1, RING_SPREAD.0, RING_COUNT.1)
        );
        assert_round_trips(&s);
    }

    #[test]
    fn every_structure_slider_keeps_the_settings_valid_and_reset_restores_vanilla() {
        let mut s = state(CubicKind::Custom, "");
        open(&mut s, PageId::Structures);
        let theirs = horizon_world::cubic::custom::settings::StructureSetTuning {
            enabled: Some(false),
            ..Default::default()
        };
        s.settings
            .structure_sets
            .insert("mymod:towers".into(), theirs.clone());
        for pos in [0.0, 1.0, 0.03, 0.5, 0.97, 0.0] {
            for i in sliders(&s) {
                set(&mut s, i, pos);
                if let Err(e) = s.settings.validate() {
                    panic!("slider {i} at {pos}: {e}");
                }
            }
        }
        assert_round_trips(&s);
        s.edit(|c| c.villages = false);
        assert!(enabled(&s, RESET));
        set(&mut s, RESET, 0.0);
        for v in OVERWORLD_STRUCTURE_SETS {
            assert!(s.settings.structure_enabled(v.name), "{}", v.name);
            assert_eq!(s.settings.structure_placement(v.name), Some(v.placement));
        }
        assert!(!enabled(&s, RESET), "nothing left to reset");
        // A set the page does not list keeps its entry.
        assert_eq!(s.settings.structure_sets.len(), 1);
        assert_eq!(s.settings.structure_sets["mymod:towers"], theirs);
        assert_round_trips(&s);
    }

    #[test]
    fn the_basic_page_keeps_dungeons_and_ravines_but_no_structure_toggles() {
        for kind in [CubicKind::Custom, CubicKind::Hybrid] {
            let s = state(kind, "");
            assert_eq!(s.page_id(), Some(PageId::Basic));
            let structures = |c: &CustomGeneratorSettings| {
                (
                    (c.strongholds, c.villages, c.mineshafts, c.temples),
                    (c.ocean_monuments, c.woodland_mansions),
                    c.alternate_strongholds_positions,
                    c.structure_sets.clone(),
                )
            };
            for pos in [0.0, 1.0] {
                assert_eq!(control(&s, pos, structures), None, "{kind:?}");
            }
            assert!(control(&s, 0.0, |c| c.dungeons).is_some());
            assert!(control(&s, 0.0, |c| c.ravines).is_some());
        }
    }

    /// The vanilla strings the labels use, under Horizon's bundled ones.
    struct VanillaStrings;
    impl crate::lang::ResourceSource for VanillaStrings {
        fn read(&self, path: &str) -> Option<Vec<u8>> {
            (path == "assets/minecraft/lang/en_us.json").then(|| {
                br#"{"options.generic_value": "%s: %s", "options.on": "ON", "options.off": "OFF"}"#
                    .to_vec()
            })
        }
    }

    #[test]
    fn the_structures_page_shows_vanillas_numbers_in_english_and_portuguese() {
        use crate::lang::{Language, scoped};
        let mut s = state(CubicKind::Custom, "");
        open(&mut s, PageId::Structures);
        let labels = |s: &State| {
            page(s)
                .widgets
                .iter()
                .map(|w| w.item.label.clone())
                .collect::<Vec<_>>()
        };
        let (english, missed) = scoped(Language::load(&[&VanillaStrings], "en_us"), || labels(&s));
        assert!(missed.is_empty(), "{missed:?}");
        let block = |name: &str| english[switch_of(name)..switch_of(name) + 4].to_vec();
        assert_eq!(
            block("minecraft:villages"),
            [
                "Villages: ON",
                "Chance: 100%",
                "Spacing: 34",
                "Separation: 8"
            ]
        );
        assert_eq!(block("minecraft:pillager_outposts")[1], "Chance: 20%");
        assert_eq!(
            block("minecraft:buried_treasures")[..2],
            ["Buried Treasure: ON", "Chance: 1%"]
        );
        assert_eq!(
            block("minecraft:mineshafts")[..2],
            ["Mineshafts: ON", "Chance: 0.4%"]
        );
        assert_eq!(
            block("minecraft:strongholds"),
            ["Strongholds: ON", "Distance: 32", "Spread: 3", "Count: 128"]
        );
        let (portuguese, _) = scoped(Language::load(&[&VanillaStrings], "pt_br"), || labels(&s));
        assert!(portuguese[switch_of("minecraft:villages")].starts_with("Vilarejos"));
        // Every set's name and the page's own strings are in both files.
        let keys: Vec<String> = OVERWORLD_STRUCTURE_SETS
            .iter()
            .map(|v| set_key(v.name))
            .chain(
                [
                    PageId::Structures.name(),
                    "horizon.cubic.structure.help",
                    "horizon.cubic.structure.vanilla",
                    "horizon.cubic.structure.chance",
                    "horizon.cubic.structure.spacing",
                    "horizon.cubic.structure.separation",
                    "horizon.cubic.structure.distance",
                    "horizon.cubic.structure.spread",
                    "horizon.cubic.structure.count",
                ]
                .map(str::to_owned),
            )
            .collect();
        for (code, file) in [
            (
                "en_us",
                include_str!("../../assets/horizon/lang/en_us.json"),
            ),
            (
                "pt_br",
                include_str!("../../assets/horizon/lang/pt_br.json"),
            ),
        ] {
            let strings: std::collections::HashMap<String, String> =
                serde_json::from_str(file).unwrap();
            for key in &keys {
                assert!(strings.contains_key(key), "{code} names {key}");
            }
        }
    }
}
