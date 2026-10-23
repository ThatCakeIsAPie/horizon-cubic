# World height: from a fixed column to the dimension type's

Vanilla takes each level's build range from its dimension type: `min_y` and `height`, multiples of 16,
with `height >= 16` and `min_y + height <= 2032` (`DimensionType.MIN_Y = -2032`, `Y_SIZE = 4064`,
`MAX_Y = 2031`: `BlockPos` packs Y in 12 bits). `LevelHeightAccessor` (`getMinY`, `getHeight`,
`getSectionsCount`, `getSectionIndex`) derives everything else from those two values. Data packs that
reshape the Overworld make it taller: JJThunder To The Max sets `min_y -64, height 2096`, the largest
top vanilla allows, which is 131 sections against vanilla's 24.

Horizon's client fixes the height when the game is compiled. This page lists what depends on that, the
cost of a taller world, and the plan to make the height a runtime value. Vanilla worlds must come out
identical at every step: the same saves, packets and harness results.

## What is already dynamic

- **horizon-world's generator.** Proto chunks, heightmaps and decoration regions span the dimension
  type's build range (`Settings::level_min_y`, `level_height` in `worldgen/generator.rs`), and the noise
  spans the noise settings' own `min_y`/`height`. The build range itself still comes from a hard-coded
  table, `dimension_type_height` (Overworld `-64, 384`, Nether and End `0, 256`), not from the
  `dimension_type` data. That is the worldgen data-pack work's area (agent A), not this one.
- **Heightmaps** (`horizon_world::heightmap::Heightmap::new(kind, min_y, height)`) size their bit
  width from the height, as vanilla does: 9 bits at 384, 12 at 2096.
- **The protocol.** `horizon_protocol::registry::DimensionType` carries `min_y`, `height` and
  `has_skylight` from the server's registry data, and the light masks are `BitSet`s (`Vec<u8>`), so
  130+ sections fit.
- **Occlusion culling** (`gfx/occlusion.rs`) takes `(min_y, sections)` and keeps a `Box<[u64]>` of
  per-section visibility per column.
- **LOD columns** (`lod.rs`) and the client heightmap cache (`Chunk::surface`) store heights as `i16`.
- **Terrain vertices** are section-relative `f32`; a section's origin is an `i32`.

## What is fixed: one column window for every dimension

`crates/horizon-client/src/chunk.rs` defines

```rust
pub const MIN_Y: i32 = -64;
pub const HEIGHT: i32 = 384;
pub const SECTIONS: usize = (HEIGHT / 16) as usize;
```

and every client `Chunk` holds exactly `SECTIONS` sections from `MIN_Y`. Every dimension lives in that
window: the Nether and the End (`0..256`) use sections 4..19, with the others empty. Paths that receive
another level's layout map it into the window and drop what falls outside. `game/remote.rs` shifts a
server's chunk data by `(dimension.min_y - MIN_Y) / 16` and skips the sections beyond, and
`lan/chunks.rs`'s `set_dimension` encodes only the level's sub-range. So today a server with a taller
dimension is cut to `-64..320`, and so is a data pack's taller Overworld (`vanillagen.rs` clamps the
generator's output to the window).

The constants are used about 480 times in 65 files of `horizon-client`: 317 `MIN_Y`, 109 `SECTIONS`,
58 `HEIGHT`. The uses fall into two kinds, which the plan treats differently.

### Storage window uses (the column's layout)

| Subsystem | Files (references) |
| --- | --- |
| Chunk storage | `chunk.rs` (42): `Chunk::empty`, `get`, `set`, `highest`, heightmap cache |
| World and dirty sections | `world.rs` (15): `DirtySections`, `World::set`, light edit log |
| Light | `light.rs` (48): `light_chunk`'s flood-fill volume, `relight`, `light_from_layers`, `update_light_layers`, `lowest_sky_source` |
| Streaming and meshing | `streaming.rs` (6), `mesher.rs` (3), `gfx.rs` (5), `gfx/resident.rs` (1) |
| Saves | `save.rs` (36): section lists, `yPos`, light layers, heightmaps, LOD light |
| Network | `lan/chunks.rs` (21, the host's encoder), `game/remote.rs` (11, the client's decoder) |
| Per-section caches | `random_tick_sections.rs` (36), `portal_sections.rs` (12), `poi_sections.rs` (6) |
| Generation glue | `vanillagen.rs` (13), `vanillagen/lod.rs`, `flatgen.rs` (4) |
| LOD and minimap | `lod.rs` (9), `minimap.rs` (9) |

### Level build-range uses (gameplay)

These ask "where does this level begin and end": void damage, build limits, spawn and portal searches,
command bounds, pathfinding limits, mob AI. Many use the window's bounds for every dimension, and some
choose per dimension by hand. `Game`'s `LevelAccess::min_y`/`max_y` (`game.rs`) return `0..256` outside
the Overworld and `MIN_Y..MIN_Y + HEIGHT` in it, `game/render_tick.rs` picks `MIN_Y` or `0`,
`game/nether_portal.rs` hard-codes `128`/`384` logical heights, and `game.rs:2986` tests
`MIN_Y + 384`. Files: `game.rs` (24), `game/initial_spawn.rs` (10), `game/render_tick.rs` (8),
`game/commands*.rs` (21), `game/nether_portal.rs` (6), `game/tnt.rs`, `game/agent.rs`,
`game/natural_spawning.rs`, `game/creatures.rs`, `game/raids.rs`, `game/entity_culling.rs`,
`game/debug_keys.rs`, `game/sculk*.rs`, `game/chat.rs`, `game/creaking.rs`, `game/dispenser.rs`,
`game/wandering_trader.rs`, `game/wind.rs`, `arrows.rs` (8), `motion/*.rs`, and `mobs/*.rs` (`path.rs`,
`goals.rs`, `villager_brain.rs`, `nav.rs`, `raider.rs`, `world.rs` and six one-line uses). Vanilla
reads all of them from the level's `LevelHeightAccessor`.

The examples (`chunk_bench`, `light_parity`, `save_parity`, `section_mesh_fixture`, ...) and three
integration tests (`generation_saves`, `lan_bot`, `streaming_meshes`) use the constants too.

### Assumptions of at most 32 sections

`world.rs` asserts `SECTIONS <= 32` because per-column section sets are `u32` bitmasks:

- `world::DirtySections` (`columns: FxHashMap<(i32, i32), u32>`, `take_column`) and its consumer in
  `streaming.rs` (the `trailing_zeros` walk);
- `World::light_log` (`FxHashMap<(i32, i32), u32>`) and the edit log in `streaming.rs`;
- `random_tick_sections::Sections(u32)`, the sections that tick randomly;
- `portal_sections::PortalSections::get` (returns `u32`) and the `y_mask` it is tested against in
  `game/nether_portal.rs`;
- `gfx.rs`'s `detail_columns: HashMap<(i32, i32), u32>` and `COMPLETE_COLUMN`, the columns whose every
  section is meshed, which decide where distant terrain gives way to chunks.

Vanilla's tallest level has 254 sections, so a 256-bit set (`[u64; 4]`, 32 bytes, `Copy`) holds every
legal layout. `random_tick_sections.rs` also keeps a fixed `[Weak<Section>; SECTIONS]` per chunk, and
`save.rs` a `[_; SECTIONS]` light-layer array: both become `Vec`s or sized by the layout.

## What a 2096-high world costs

Numbers for To The Max's Overworld (131 sections) against vanilla's 24. Nothing here rules the height
out, but the empty-section overhead and the light volume should be dealt with early.

- **Empty sections.** `Chunk::empty` allocates an `Arc<Section>` per section: two single-value
  `PalettedContainer`s and a uniform `Light`, about 180 bytes with the `Arc` header and the `Vec` slot.
  That is about 4 KB per chunk at 24 sections and 24 KB at 131. At render distance 32 (4,225 chunks)
  it comes to about 100 MB of empty sections against 18 MB today. Sharing one empty
  `Arc<Section>` across the column (copy-on-write through `Arc::make_mut` already clones on the first
  edit) would reduce that to the 8-byte slots, about 4.4 MB. Whatever the window, Nether and End
  chunks pay the same, since they live in it.
- **Occupied sections.** A section with mixed blocks and light costs about 8 KB (blocks palette,
  4,096-byte light array). To The Max's mountains rise past y 1000 in places, so a column holds
  several times vanilla's 20-odd occupied sections where the terrain is high. Expect resident chunk
  memory to grow two to four times in mountain ranges, and GPU mesh memory to grow with the number of
  exposed faces (cliffs).
- **Light.** `light_chunk` floods a 48 x 48 x h volume, with h up to one section above the highest
  occupied section of the 3 x 3 neighbourhood, at 4 bytes per cell (2 for the cell, 1 each for sky and
  block light). 384 high: 3.5 MB and 885k cells. 2096 high: 19 MB and 4.8M cells per job, per worker,
  and about 5.5 times the work. The same flood fill runs for relighting and for every newly loaded
  chunk, so generation and streaming of mountains slow down by about that factor unless sky light
  starts at the region's top occupied section (cells above it are all 15). The per-section light
  layers of saves and packets (`vec![(None, None); SECTIONS]`, `SECTIONS * 4096` bytes in
  `save.rs`) grow linearly and stay small.
- **Meshing and streaming.** Loops `for sy in 0..SECTIONS` skip empty sections cheaply. Meshing cost
  follows the occupied sections, as above.
- **Rendering.** The occlusion graph walks sections in view. From a peak, many more sections are
  visible below the camera, and the walk and draw counts grow with the visible vertical extent. Fog and
  render distance are horizontal, so nothing else changes. Distant terrain (LOD) is height-agnostic.
- **Network.** A chunk packet carries every section of the level: 131 single-value sections add about
  1 KB. The heaviest case, 131 sections of direct-palette blocks, is about 1.1 MB, under vanilla's
  2 MB chunk buffer limit. Light masks are `BitSet`s.
- **Saves.** Region files store sections by Y with `yPos`; nothing in the format limits the height.
  Heightmaps are 12 bits per column at 2096.

## The model

1. **Column layout** (storage): the window, `min_y` and section count, becomes a runtime value that
   covers every dimension of the open world, so that for vanilla it is `-64, 24` as now. It is set
   once, before the first chunk, when a world is opened, a world is created, or a server's registries
   arrive, as `chunk::set_block_states` already sets the palette width. One world is open in a
   process at a time (the server thread shares its world), so a process-wide value is enough. The
   `Chunk` also records its own `min_y`, so its hot accessors read a field, not a global.
2. **Level build range** (gameplay): each level's `min_y`/`height` comes from its dimension type, read
   from the world's data (the `dimension_type` registry, data packs included) or a server's registry
   data, replacing the ad-hoc "Overworld or `0..256`" choices. For vanilla this gives the values in use
   today.

Holding every dimension in the union of their ranges keeps today's mapping code (`remote.rs`,
`lan/chunks.rs`) as it is. A per-dimension window would save the empty sections of Nether and End
chunks in a tall Overworld, which the shared empty section makes cheap anyway.

## Plan: small steps, vanilla unchanged at each

1. **Section sets past 32.** A `SectionMask` (256 bits) replaces the `u32` masks listed above, and
   the `SECTIONS <= 32` assertion goes. No behaviour change.
2. **The layout as a value.** `chunk::layout()` (`min_y`, `sections`, `height`, with `section_index`
   and `section_y` helpers) and `set_layout`, checked against vanilla's limits, with the default
   `-64, 24`. `Chunk` records `min_y`, and `Chunk::get`/`set`/`highest` read the chunk's own sections.
   The constants stay, documented as the default, until nothing uses them.
3. **Storage users move to the layout**, a subsystem per step: light; world, streaming, meshing and
   gfx; saves; network (`lan/chunks.rs`, `remote.rs`); per-section caches (`random_tick_sections`,
   `portal_sections`, `poi_sections`); generation glue, LOD and minimap.
4. **Level build range.** One function gives the current level's range from its dimension type. The
   gameplay users above move to it, and `LevelAccess::min_y`/`max_y` follow.
5. **Set it from the data.** World open and create read the dimension types (from the world's data
   packs, through the worldgen data-pack overlay) and call `set_layout` with their union, and a
   remote client does the same from the server's `minecraft:dimension_type` registry. The constants
   are removed. `vanillagen.rs` then stops clamping a taller generator's output.
6. **Cost.** One shared empty section per column. Sky light that starts at the region's top occupied
   section. Measure resident and GPU memory and generation speed with a 2096-high world.

Each step runs the harnesses its files reach (`python3 tools/parity.py --changed-files`), which for
client-library files is about a hundred, and a vanilla world in game. Once step 5 lands, a test world
with a tall dimension type (an MIT-licensed fixture data pack, not To The Max itself) checks saves,
light and LAN at the new height.
