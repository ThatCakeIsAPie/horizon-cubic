#!/usr/bin/env python3
"""Build the "Big Ore Deposits" data pack: ores come in a few huge deposits
instead of many small veins, like Cubic Chunks' custom ore settings.

Edit DEPOSITS below, then run:  python3 make_pack.py
It writes the pack folder big_ore_deposits/ next to this script. Copy that
folder into a world's datapacks/ folder; only chunks generated after that get
deposits (explored land keeps its old ores).

How a deposit is placed (vanilla 26.3 placement rules, so the pack also works
in Java Minecraft):
  rarity_filter 1 in `every` chunks -> one centre in the chunk -> its height
  (`depth` blocks under the local surface, or an absolute `y` range) -> `blobs`
  copies of the centre, each moved up to `spread` blocks -> an ore blob of
  `size` at each. Blobs overlap into one solid mass. Ores only replace stone
  and deepslate, so caves cut through deposits and show them off.
The pack replaces the overworld ore placements by id, so every biome (To The
Max's included) gets deposits and no ordinary veins.

Mekanism's ores (osmium, tin, lead, uranium, fluorite) come as deposits too,
when MEKANISM is True: they take over vanilla placement ids the pack switches
off anyway (MEKANISM_IDS), since a data pack can't add features to biomes.
They name mekanism:* blocks, so a world with this pack needs the Mekanism
content mod on (~/Work/mekanism/README.md); set MEKANISM = True  for a world
without it.
"""

import json
import shutil
from pathlib import Path

PACK = Path(__file__).resolve().parent / "big_ore_deposits"
PACK_FORMAT = 121  # Minecraft 26.3 (the same as the upgraded To The Max pack)
MEKANISM = False  # also make Mekanism's ore deposits (needs the Mekanism content mod)

# every:  one deposit in this many chunks (16x16 columns) on average
# blobs:  ore blobs per deposit; size: blocks per blob (vanilla max 64)
# spread: how far blobs sit from the centre (max 16)
# depth:  (min, max) blocks under the local surface, or y: (min, max) absolute
#   (deposits of one ore sit about 16 * sqrt(every) blocks apart)
DEPOSITS = {
    "coal":     dict(every=40,  blobs=64, size=64, spread=10, depth=(8, 40)),
    "iron":     dict(every=48,  blobs=64, size=56, spread=10, depth=(10, 44)),
    "copper":   dict(every=48,  blobs=56, size=56, spread=10, depth=(10, 44)),
    "gold":     dict(every=96,  blobs=48, size=48, spread=9,  y=(-64, 32)),
    "redstone": dict(every=80,  blobs=48, size=48, spread=9,  y=(-64, 16)),
    "lapis":    dict(every=96,  blobs=40, size=40, spread=8,  y=(-64, 64)),
    "diamond":  dict(every=160, blobs=32, size=32, spread=7,  y=(-64, 16)),
    "emerald":  dict(every=96,  blobs=24, size=24, spread=6,  depth=(6, 30)),
    # Mekanism (heights from its default ore config, mekanism-common.toml)
    "osmium":   dict(every=56,  blobs=56, size=56, spread=10, depth=(10, 44)),
    "tin":      dict(every=56,  blobs=48, size=48, spread=9,  y=(-32, 72)),
    "lead":     dict(every=80,  blobs=48, size=48, spread=9,  y=(-64, 48)),
    "uranium":  dict(every=128, blobs=40, size=40, spread=8,  y=(-64, 0)),
    "fluorite": dict(every=112, blobs=36, size=36, spread=8,  y=(-64, 16)),
}

# The placed-feature id each deposit takes over (biomes list these), and the
# other vanilla ids of that ore, which are switched off.
IDS = {
    "coal": ("ore_coal_upper", ["ore_coal_lower"]),
    "iron": ("ore_iron_upper", ["ore_iron_middle", "ore_iron_small"]),
    "copper": ("ore_copper", ["ore_copper_large"]),
    "gold": ("ore_gold", ["ore_gold_lower", "ore_gold_extra"]),
    "redstone": ("ore_redstone", ["ore_redstone_lower"]),
    "lapis": ("ore_lapis", ["ore_lapis_buried"]),
    "diamond": ("ore_diamond", ["ore_diamond_medium", "ore_diamond_large", "ore_diamond_buried"]),
    "emerald": ("ore_emerald", []),
}

# Mekanism's ores have no placement id of their own in biomes: each takes one
# that every overworld biome (To The Max's included) lists and that the pack
# switches off above.
MEKANISM_IDS = {
    "osmium": "ore_iron_middle",
    "tin": "ore_iron_small",
    "lead": "ore_gold_lower",
    "uranium": "ore_diamond_medium",
    "fluorite": "ore_diamond_large",
}


def targets(ore):
    ns = "mekanism" if ore in MEKANISM_IDS else "minecraft"
    deepslate = f"{ns}:deepslate_{ore}_ore"
    return [
        {"target": {"predicate_type": "minecraft:tag_match", "tag": "minecraft:stone_ore_replaceables"},
         "state": f"{ns}:{ore}_ore"},
        {"target": {"predicate_type": "minecraft:tag_match", "tag": "minecraft:deepslate_ore_replaceables"},
         "state": deepslate},
    ]


def uniform(lo, hi):
    return {"type": "minecraft:uniform", "min_inclusive": lo, "max_inclusive": hi}


def trapezoid(r):
    return {"type": "minecraft:trapezoid", "min": -r, "max": r, "plateau": 0}


def offsets_down(lo, hi):
    """`offset` moves at most 16 a step: split a (lo, hi) depth into steps."""
    steps = []
    while hi > 0:
        a, b = min(lo, 16), min(hi, 16)
        steps.append({"type": "minecraft:offset", "x": 0, "y": uniform(-b, -a), "z": 0})
        lo, hi = max(lo - 16, 0), hi - 16
    return steps


def placement(d):
    p = [{"type": "minecraft:rarity_filter", "chance": d["every"]}, {"type": "minecraft:in_square"}]
    if "depth" in d:
        p.append({"type": "minecraft:heightmap", "heightmap": "OCEAN_FLOOR_WG"})
        p += offsets_down(*d["depth"])
    else:
        lo, hi = d["y"]
        p.append({"type": "minecraft:height_range",
                  "height": {"type": "minecraft:uniform",
                             "min_inclusive": {"absolute": lo}, "max_inclusive": {"absolute": hi}}})
    p.append({"type": "minecraft:count", "count": d["blobs"]})
    s = d["spread"]
    p.append({"type": "minecraft:offset", "x": trapezoid(s), "y": trapezoid(s), "z": trapezoid(s)})
    return p


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    if PACK.exists():
        shutil.rmtree(PACK)
    write(PACK / "pack.mcmeta", {"pack": {
        "description": "Big Ore Deposits: few, huge ore deposits",
        "pack_format": PACK_FORMAT, "min_format": PACK_FORMAT, "max_format": PACK_FORMAT}})
    wg = PACK / "data"
    taken = set(MEKANISM_IDS.values()) if MEKANISM else set()
    for ore, d in DEPOSITS.items():
        assert 0 < d["size"] <= 64 and 0 < d["spread"] <= 16 and d["every"] >= 1
        if ore in MEKANISM_IDS and not MEKANISM:
            continue
        write(wg / "big_ore_deposits/worldgen/feature" / f"{ore}_deposit.json",
              {"type": "minecraft:ore", "size": d["size"],
               "discard_chance_on_air_exposure": 0.0, "targets": targets(ore)})
        main_id, off = IDS[ore] if ore in IDS else (MEKANISM_IDS[ore], [])
        write(wg / "minecraft/worldgen/placed_feature" / f"{main_id}.json",
              {"feature": f"big_ore_deposits:{ore}_deposit", "placement": placement(d)})
        for other in (o for o in off if o not in taken):
            write(wg / "minecraft/worldgen/placed_feature" / f"{other}.json",
                  {"feature": f"big_ore_deposits:{ore}_deposit",
                   "placement": [{"type": "minecraft:count", "count": 0}]})
    print(f"wrote {PACK}")


if __name__ == "__main__":
    main()
