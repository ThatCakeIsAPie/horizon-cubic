#!/usr/bin/env python3
"""Vertical shards end to end (docs/cubic-v2-plan.md, "Residency"): a host and a guest, both real games.

The host's cubic world runs on its integrated server and is open to LAN with commands allowed. The guest
joins, then teleports 20,000 blocks up. The server must give it a shard of its own and the checks are:

- split:    the guest is in a new shard: it stands at the new height with solid ground it builds, and the
            host (whose window is around y 64) cannot read that block;
- two:      the host still has its own terrain and the block it built, and neither player is told the
            other's cubes;
- merge:    the guest comes back down to the host: the shards merge, the guest sees the host's block and the
            host's world is unchanged;
- reopen:   the guest goes up again: the shard that closed saved the platform, and the new one loads it.

    python3 tools/cubic_shards_e2e.py --bin-dir D:/ClaudeData/horizon-cubic-shards-target/fast \
        --out D:/ClaudeData/cubic-play/shards/run1

Scratch folders (game dirs, HORIZON_HOME, logs) go under --scratch, never the real game folder.
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab  # noqa: E402
from lan_e2e import Failed, Game, pos_of  # noqa: E402

HIGH = 20_000


def block(game, x, y, z):
    return game.raw('block', x, y, z)['block']


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--bin-dir', required=True, help='prebuilt horizon-client and horizon-ctl (fast profile)')
    parser.add_argument('--scratch', default=r'D:\ClaudeData\cubic-play\shards', help='scratch folders go here')
    parser.add_argument('--out', default=None, help='the games\' logs go here')
    args = parser.parse_args()
    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(scratch)
    out = Path(args.out) if args.out else scratch / 'out'
    out.mkdir(parents=True, exist_ok=True)
    ctl = Path(args.bin_dir).resolve() / ('horizon-ctl.exe' if sys.platform == 'win32' else 'horizon-ctl')
    resources = lab.RESOURCES.resolve()
    host, guest = Game('host', ctl, resources), Game('guest', ctl, resources)
    results = {}
    try:
        host.pid = host.raw('launch', '--offscreen', '--no-pause', '--world', 'ShardsE2E', '--seed', 5,
                            '--creative', '--cubic', '--wait')['pid']
        host('wait_ready', seconds=600)
        host('command', command='time set day')
        host('command', command='gamerule spawn_mobs false')
        port = host.raw('lan', 'open', 0, 'commands')['port']
        guest.pid = guest.raw('launch', '--offscreen', '--no-pause')['pid']
        guest.raw('connect', f'127.0.0.1:{port}', 'Guest')
        guest.raw('wait-for', 'loaded', '--timeout', 300)
        guest.wait(60)
        gx, gy, gz = (int(v // 1) for v in pos_of(guest))
        hx, hy, hz = (int(v // 1) for v in pos_of(host))
        results['joined'] = {'host': [hx, hy, hz], 'guest': [gx, gy, gz]}
        host('command', command=f'setblock {hx} {hy - 3} {hz} minecraft:gold_block')
        guest.wait(20)
        assert block(guest, hx, hy - 3, hz) == 'minecraft:gold_block', 'the guest sees the host\'s block first'

        # split: the guest goes up.
        guest('command', command='gamemode creative')
        guest('command', command=f'tp @s {gx} {HIGH} {gz}')
        guest.until(lambda: guest('state'), lambda s: s.get('screen') == 'playing' and s['player']['y'] > HIGH - 5, ticks=400)
        guest.wait(120)
        guest('command', command=f'fill {gx - 2} {HIGH - 2} {gz - 2} {gx + 2} {HIGH - 2} {gz + 2} minecraft:diamond_block')
        guest.wait(20)
        up = block(guest, gx, HIGH - 2, gz)
        host_reads = block(host, gx, HIGH - 2, gz)
        results['split'] = {'guest_block': up, 'host_reads': host_reads}
        assert up == 'minecraft:diamond_block', results['split']
        assert host_reads == 'minecraft:air', results['split']
        # two: the host is untouched.
        assert block(host, hx, hy - 3, hz) == 'minecraft:gold_block'
        assert block(guest, hx, hy - 3, hz) == 'minecraft:air', 'the guest is not told the host cubes'
        results['two'] = {'host_block': block(host, hx, hy - 3, hz), 'host_pos': pos_of(host)}

        # merge: the guest comes back down.
        guest('command', command=f'tp @s {hx} {hy} {hz}')
        guest.until(lambda: guest('state'), lambda s: s.get('screen') == 'playing' and s['player']['y'] < hy + 5, ticks=400)
        guest.wait(120)
        results['merge'] = {'guest_sees': block(guest, hx, hy - 3, hz), 'guest_pos': pos_of(guest)}
        assert results['merge']['guest_sees'] == 'minecraft:gold_block', results['merge']

        # reopen: up again; the platform the closed shard saved is there.
        guest('command', command=f'tp @s {gx} {HIGH} {gz}')
        guest.until(lambda: guest('state'), lambda s: s.get('screen') == 'playing' and s['player']['y'] > HIGH - 5, ticks=400)
        guest.wait(120)
        results['reopen'] = {'guest_block': block(guest, gx, HIGH - 2, gz)}
        assert results['reopen']['guest_block'] == 'minecraft:diamond_block', results['reopen']
        results['pass'] = True
    except (Failed, AssertionError, KeyError, TypeError) as e:
        results['pass'] = False
        results['error'] = repr(e)
    finally:
        for g in (guest, host):
            (out / f'{g.name}.log').write_text(g.close())
    print(json.dumps(results, indent=1))
    sys.exit(0 if results.get('pass') else 1)


if __name__ == '__main__':
    main()
