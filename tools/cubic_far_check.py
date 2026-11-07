#!/usr/bin/env python3
"""Checks of a cubic world's far view that the far-fill benchmark does not make (docs/cubic-v2-plan.md).

    python3 tools/cubic_far_check.py --bin-dir D:/ClaudeData/horizon-cubic-target/fast \
        --resources <prepared 26.3 resources> --out <scratch folder> [--detail ultra] \
        [--hitch] [--spyglass] [--cavern]

Each check launches a cubic world offscreen at Whole World (as tools/cubic_far_bench.py does, with
its own HORIZON_HOME and control socket in --out) and prints `status`'s `cube_lod` counters:
- `--hitch`: teleports 5,000,000 blocks twice and prints the frame-time summary of the 4 s after each
  (`horizon-ctl stats`: the longest frame is the teleport's hitch), then waits for the fill.
- `--spyglass`: at y 300 facing the horizon (or `--spyglass-at Y:PITCH`), screenshots before and while
  looking through a spyglass (the far nodes in view refine: `planned` grows), and after letting go.
- `--watch S`: after the views, holds still for S seconds and samples the plan four times a second:
  how often it changed and how far `planned` and `budget` moved (a still camera's plan should not
  swing between sizes).
- `--cavern`: spectator at y -2,000 inside rock (screenshots four ways), then inside an open space: a
  natural one if a 96-block box around holds one, else a room carved with /fill. Prints the counters
  (no holes, complete) and the log's errors.
- `--views Y:PITCH ...`: at x --x, each height in turn, waits for the fill and screenshots
  (`view_y<Y>_p<PITCH>.png`), e.g. `--views 300:0 1000:60 10000:60 100000:60 1000000:60 5000000:60`.
- `--tall`: the world's CubicWorldGen heights raised (heightFactor 30,000, heightVariationFactor
  1,000: peaks of several thousand blocks) before the checks, as the tiles' own test did.
Screenshots go to --out.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time

EXE = '.exe' if os.name == 'nt' else ''
WHOLE_WORLD = 1 << 22


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bin-dir', required=True, help='folder with horizon-client and horizon-ctl')
    ap.add_argument('--resources', required=True, help='prepared 26.3 resources')
    ap.add_argument('--out', required=True, help='scratch folder: game, home, screenshots')
    ap.add_argument('--detail', choices=['low', 'normal', 'high', 'ultra', 'extreme'])
    ap.add_argument('--budget', type=int, help='the most far nodes a plan holds (HORIZON_FAR_NODES)')
    ap.add_argument('--seed', type=int, default=98765)
    ap.add_argument('--x', type=int, default=5_000_000, help='where the checks teleport to')
    ap.add_argument('--hitch', action='store_true')
    ap.add_argument('--spyglass', action='store_true')
    ap.add_argument('--spyglass-at', default='300:2', help='Y:PITCH the spyglass looks from')
    ap.add_argument('--size', default='1280x720', help='the window, WIDTHxHEIGHT')
    ap.add_argument('--watch', type=float, default=0.0, help='seconds to watch a still plan for swings')
    ap.add_argument('--cavern', action='store_true')
    ap.add_argument('--views', nargs='*', default=[], help='Y:PITCH views to screenshot after each fill')
    ap.add_argument('--tall', action='store_true', help='raise the terrain to peaks of thousands of blocks')
    ap.add_argument('--timeout', type=float, default=180.0, help='seconds a fill may take')
    args = ap.parse_args()
    bins = Path(args.bin_dir)
    ctl, game = bins / f'horizon-ctl{EXE}', bins / f'horizon-client{EXE}'
    scratch = Path(args.out)
    home, game_dir = scratch / 'home', scratch / 'game'
    for d in (home, game_dir):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    detail = f'cubicFarDetail:{args.detail}\n' if args.detail else ''
    (game_dir / 'horizon-graphics.txt').write_text(
        f'lodDistance:{WHOLE_WORLD}\nfarDetail:full\ncubicVertical:8\ngpuFarTerrain:auto\n' + detail)
    (game_dir / 'options.txt').write_text('inactivityFpsLimit:"minimized"\n')
    env = dict(os.environ, HORIZON_HOME=str(home), HORIZON_NO_AUDIO='1')
    if args.budget:
        env['HORIZON_FAR_NODES'] = str(args.budget)
    # A control socket of this run's own: another game on the machine never takes our commands.
    socket = scratch / 'control.json'

    def ctl_run(*argv, timeout=60):
        r = subprocess.run([str(ctl), '--socket', str(socket), *map(str, argv)], env=env, capture_output=True,
                           text=True, timeout=timeout)
        return r.stdout.strip()

    def status():
        try:
            return json.loads(ctl_run('status'))
        except (json.JSONDecodeError, subprocess.TimeoutExpired):
            return {}

    def counters(lod):
        keys = ['planned', 'ready', 'pending', 'holes', 'covering', 'complete', 'target_px', 'voxel_px',
                'limited', 'budget', 'cover_nodes', 'plan_ms', 'arena_bytes', 'surface', 'tiles', 'tiles_ready',
                'tiles_complex', 'tile_bytes']
        return ' '.join(f'{k}={lod.get(k)}' for k in keys)

    def settle(label, since, moved=True):
        """Waits until the plan made after `since` is complete; prints when it was covered and complete."""
        cover = None
        while time.monotonic() - since < args.timeout:
            s = status()
            lod = s.get('cube_lod') or {}
            elapsed = time.monotonic() - since
            changed = lod.get('plan_changed_ago_s')
            fresh = changed is not None and (changed <= elapsed + 0.05 or (not moved and elapsed > 1.0))
            if cover is None and fresh and lod.get('plan_covered_ago_s') is not None:
                cover = max(0.0, elapsed - lod['plan_covered_ago_s'])
            if fresh and lod.get('complete'):
                done = elapsed - (lod.get('plan_complete_ago_s') or 0)
                cover_text = 'n/a' if cover is None else f'{cover:.2f} s'
                print(f'{label}: covered {cover_text}, complete {done:.2f} s, fps {s.get("fps")}: {counters(lod)}',
                      flush=True)
                return lod
            time.sleep(0.2)
        lod = status().get('cube_lod') or {}
        print(f'{label}: not complete after {args.timeout} s: {counters(lod)}', flush=True)
        return lod

    def shots(prefix, views):
        for yaw, pitch, name in views:
            ctl_run('look', yaw, pitch)
            time.sleep(0.8)
            ctl_run('screenshot', scratch / f'{prefix}_{name}.png')

    def open_space(center):
        """A block of air near `center` in the loaded cubes: the box of 32^3 with the most air, or None."""
        best = None
        for dy in (-32, 0, 32):
            for dz in (-48, -16, 16):
                for dx in (-48, -16, 16):
                    lo = [center[0] + dx, center[1] + dy, center[2] + dz]
                    hi = [v + 31 for v in lo]
                    try:
                        box = json.loads(ctl_run('blocks', *lo, *hi, timeout=120))
                    except (json.JSONDecodeError, subprocess.TimeoutExpired):
                        continue
                    solid = {tuple(b[:3]) for b in box.get('blocks', []) if isinstance(b, list)}
                    if 'counts' not in box:
                        continue
                    air = 32 ** 3 - len(solid)
                    if air > 64 and (best is None or air > best[0]):
                        free = next((p for p in ((lo[0] + x, lo[1] + y, lo[2] + z)
                                                 for y in range(1, 31) for z in range(32) for x in range(32))
                                     if p not in solid and (p[0], p[1] - 1, p[2]) not in solid), None)
                        if free:
                            best = (air, free)
        return best

    def launch():
        print(ctl_run('launch', '--game', game, '--world', 'FarCheck', '--seed', args.seed, '--creative',
                      '--offscreen', '--wait', '--cubic', '--resources', args.resources, '--game-dir', game_dir,
                      '--width', args.size.split('x')[0], '--height', args.size.split('x')[1], timeout=600)[:120], flush=True)

    launch()
    if args.tall:
        ctl_run('quit')
        time.sleep(3)
        settings = game_dir / 'saves' / 'FarCheck' / 'cubicgen.json'
        data = json.loads(settings.read_text())
        data['heightFactor'] = 30000.0
        data['heightVariationFactor'] = 1000.0
        settings.write_text(json.dumps(data, indent=2))
        print('cubicgen.json: heightFactor 30000, heightVariationFactor 1000', flush=True)
        launch()
    try:
        ctl_run('command', 'gamemode spectator')
        ctl_run('look', 0, 0)
        settle('spawn', time.monotonic())
        for view in args.views:
            y, pitch = (float(v) for v in view.split(':'))
            ctl_run('tp', args.x, y, 0)
            ctl_run('look', 0, pitch)
            settle(f'view y {y:.0f}, pitch {pitch:.0f}', time.monotonic())
            time.sleep(1)
            ctl_run('screenshot', scratch / f'view_y{y:.0f}_p{pitch:.0f}.png')
        if args.watch > 0:
            # A still camera: count the plans made and how far their sizes move.
            planned, budgets, changes, last = [], [], 0, None
            end = time.monotonic() + args.watch
            while time.monotonic() < end:
                lod = status().get('cube_lod') or {}
                ago = lod.get('plan_changed_ago_s')
                if ago is not None and last is not None and ago + 0.05 < last:
                    changes += 1
                    # One line a plan: what moved.
                    print(f'  plan at {args.watch - (end - time.monotonic()):.1f} s: ' + ' '.join(
                        f'{k}={lod.get(k)}' for k in ['planned', 'tiles', 'tiles_complex', 'limited', 'voxel_px',
                                                     'tiles_limited', 'uniform_skipped', 'descent_runs', 'descent_uniform',
                                                     'descent_leaves', 'descent_overflow', 'holes', 'ready']), flush=True)
                last = ago
                if lod.get('planned') is not None:
                    planned.append(lod['planned'])
                if lod.get('budget') is not None:
                    budgets.append(lod['budget'])
                time.sleep(0.25)
            print(f'watch {args.watch:.0f} s: {changes} new plans, planned {min(planned, default=None)}..'
                  f'{max(planned, default=None)}, budget {min(budgets, default=None)}..{max(budgets, default=None)}',
                  flush=True)
        if args.hitch:
            for i in range(2):
                ctl_run('stats', 'reset')
                ctl_run('tp', args.x * (i + 1), 300, 0)
                time.sleep(4)
                try:
                    stats = json.loads(ctl_run('stats'))
                    print(f'teleport {i + 1}: longest frame {stats.get("max_ms"):.0f} ms, p99 {stats.get("p99_ms"):.0f} ms, '
                          f'p95 {stats.get("p95_ms"):.0f} ms, slow frames\' update {stats.get("slow_phases_avg_ms", {}).get("update", 0):.0f} ms',
                          flush=True)
                except (json.JSONDecodeError, TypeError, ValueError):
                    print(f'teleport {i + 1}: no stats', flush=True)
                settle(f'teleport {i + 1}', time.monotonic(), moved=False)
        if args.spyglass:
            spy_y, spy_pitch = (float(v) for v in args.spyglass_at.split(':'))
            ctl_run('tp', args.x, spy_y, 0)
            ctl_run('look', 0, spy_pitch)
            settle(f'y {spy_y:.0f}, pitch {spy_pitch:.0f}', time.monotonic())
            time.sleep(1)
            ctl_run('screenshot', scratch / 'spyglass_before.png')
            # Spectators cannot use items: creative, flying where it is.
            ctl_run('command', 'gamemode creative')
            ctl_run('command', 'give @s spyglass')
            ctl_run('select', 'spyglass')
            before = (status().get('cube_lod') or {}).get('planned')
            # Held for longer than the zoomed view can take to settle, so the
            # counters and the screenshot are taken while still looking through it.
            hold = int(args.timeout) + 30
            started = time.monotonic()
            holder = threading.Thread(
                target=lambda: ctl_run('use-item', '--hold', hold, timeout=hold + 60))
            holder.start()
            time.sleep(1.5)
            fov = (status().get('cube_lod') or {}).get('fov_deg')
            lod = settle('spyglass', time.monotonic())
            print(f'spyglass: planned {before} -> {lod.get("planned")} (fov {fov})', flush=True)
            if time.monotonic() - started < hold - 2:
                ctl_run('screenshot', scratch / 'spyglass_zoomed.png')
            else:
                print('spyglass: the hold ended before the view settled; no zoomed shot', flush=True)
            holder.join()
            t = time.monotonic()
            time.sleep(1.0)
            settle('let go', t)
            ctl_run('screenshot', scratch / 'spyglass_after.png')
            ctl_run('command', 'gamemode spectator')
        if args.cavern:
            x, y, z = args.x + 1000, -2000, 0
            ctl_run('tp', x, y, z)
            t = time.monotonic()
            ctl_run('look', 0, 0)
            settle('y -2000, in rock', t)
            shots('rock', [(0, 0, 'north'), (90, 0, 'west'), (0, 80, 'down'), (0, -80, 'up')])
            space = open_space([x, y, z])
            if space:
                print(f'open space: {space[0]} air blocks in a box, at {space[1]}', flush=True)
                p = space[1]
            else:
                p = (x, y, z)
                print(ctl_run('command', f'fill {x - 15} {y - 1} {z - 15} {x + 15} {y + 14} {z + 15} air')[:200])
                print('no open space found: carved a 31 x 16 x 31 room', flush=True)
            ctl_run('tp', p[0] + 0.5, p[1], p[2] + 0.5)
            settle('in the open space', time.monotonic(), moved=False)
            shots('cave', [(0, 0, 'north'), (180, 0, 'south'), (0, 60, 'down'), (0, -60, 'up')])
        log = ctl_run('logs')
        errors = [line for line in log.splitlines() if 'ERROR' in line or 'panic' in line.lower()]
        print(f'log errors: {len(errors)}', flush=True)
        for line in errors[:10]:
            print('  ', line[:200])
    finally:
        ctl_run('quit')


if __name__ == '__main__':
    main()
