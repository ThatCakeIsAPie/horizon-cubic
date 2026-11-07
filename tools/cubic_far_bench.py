#!/usr/bin/env python3
"""Far-fill benchmark of a cubic world: how fast distant terrain loads (docs/cubic-v2-plan.md).

    python3 tools/cubic_far_bench.py --bin-dir D:/ClaudeData/horizon-cubic-target/fast \
        --resources <prepared 26.3 resources>

The script launches a cubic world offscreen with LOD Distance at Whole World and the far terrain on the
GPU when the device allows. It waits for the far terrain around spawn to complete, then teleports
`--hops` times far away (5,000,000 blocks apart by default). Each time it measures two times:
- `cover_s`: until every part of the plan first has something drawn over it, the coarse coverage
  layer counting (`status`'s `cube_lod.plan_covered_ago_s`);
- `far_fill_s`: until the cube octree is complete again: every planned node is drawn or known empty,
  and nothing is waiting in the GPU mesher (`cube_lod.complete`).

Both come from `CubeLodStats` in `lod/cube.rs`. `--detail` picks the Cubic Far Detail step (the most
pixels a far voxel may span); `--shots` also takes the comparison screenshots (y 300 at the horizon,
y 1,000 and 10,000 looking down at 60 degrees).

The last line printed is a single JSON object, the metric for autoresearch loops:
    {"far_fill_s": <mean seconds per hop>, "cover_s": <mean>, "spawn_fill_s": ..., "hops": [...],
     "covers": [...], "fps_min": ..., "fps_steady": ..., ...}
The exit code is 1 when a fill timed out.

Each run uses a fresh scratch HORIZON_HOME and game dir, so the player's data is never touched.
Build first:
    cargo build --profile fast -p horizon-client -p horizon-client-lib --bin horizon-client --bin horizon-ctl
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

EXE = '.exe' if os.name == 'nt' else ''
# `crate::lod::MAX_CHUNKS`: the LOD Distance slider's Whole World stop.
WHOLE_WORLD = 1 << 22


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bin-dir', default='target/fast', help='folder with horizon-client and horizon-ctl')
    ap.add_argument('--resources', required=True, help='prepared 26.3 resources (the launcher\'s .local/horizon-client/26.3)')
    ap.add_argument('--out', help='keep the scratch folder here (screenshots, logs)')
    ap.add_argument('--seed', type=int, default=98765)
    ap.add_argument('--lod', type=int, default=WHOLE_WORLD, help='LOD Distance in chunks (default Whole World)')
    ap.add_argument('--vertical', type=int, default=8, help='Vertical Distance in cubes')
    ap.add_argument('--gpu', choices=['auto', 'off'], default='auto', help='Far Terrain on GPU')
    ap.add_argument('--hops', type=int, default=3)
    ap.add_argument('--hop', type=int, default=5_000_000, help='blocks between teleports')
    ap.add_argument('--y', type=int, default=300, help='height to look from')
    ap.add_argument('--timeout', type=float, default=180.0, help='seconds a fill may take')
    ap.add_argument('--trace', action='store_true', help='print the cube_lod counters every second while waiting')
    ap.add_argument('--trace-near', action='store_true',
                    help='print the near terrain streaming stages four times a second until it is in')
    ap.add_argument('--width', type=int, default=1280)
    ap.add_argument('--height', type=int, default=720)
    ap.add_argument('--detail', choices=['low', 'normal', 'high', 'ultra', 'extreme'],
                    help="Cubic Far Detail (default: the game's)")
    ap.add_argument('--budget', type=int, help='the most far nodes a plan holds (HORIZON_FAR_NODES)')
    ap.add_argument('--steady', type=float, default=3.0, help='seconds of fps sampled after the last fill')
    ap.add_argument('--shots', action='store_true', help='also take the comparison screenshots into --out')
    ap.add_argument('--shot-heights', type=int, nargs='*', default=[],
                    help='more heights for --shots, looking down at 60 degrees (e.g. 100000 1000000 5000000)')
    args = ap.parse_args()

    bins = Path(args.bin_dir)
    ctl, game = bins / f'horizon-ctl{EXE}', bins / f'horizon-client{EXE}'
    for b in (ctl, game):
        if not b.exists():
            sys.exit(f'missing {b}: build with --profile fast first')
    scratch = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix='cubic-far-bench-'))
    home, game_dir = scratch / 'home', scratch / 'game'
    for d in (home, game_dir):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    detail = f'cubicFarDetail:{args.detail}\n' if args.detail else ''
    (game_dir / 'horizon-graphics.txt').write_text(
        f'lodDistance:{args.lod}\nfarDetail:full\ncubicVertical:{args.vertical}\ngpuFarTerrain:{args.gpu}\n'
        + detail)
    # Vanilla's inactivity limit drops to 30 fps after a minute without input, which a
    # script never gives: keep it to minimized windows.
    (game_dir / 'options.txt').write_text('inactivityFpsLimit:"minimized"'+chr(10))
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

    peak = {'arena': 0, 'pool': 0, 'tiles': 0}

    near = {}

    def wait_fill(label, since, moved=True):
        """Seconds from `since` until the far terrain is first covered, and until it is complete again,
        with the fps seen meanwhile. Without `moved`, a plan already complete counts after a second.
        Also notes in `near[label]` when the near terrain was in: `terrain_loading` seen true, then false
        for a second (the time it went false)."""
        fps, last, cover = [], {}, None
        loading_seen, quiet_since = False, None
        while time.monotonic() - since < args.timeout:
            s = status()
            if args.trace_near and label not in near:
                stages = s.get('streaming') or {}
                print(f'  near t={time.monotonic() - since:5.2f} loading={s.get("terrain_loading")} '
                      f'chunks={s.get("chunks")} uploads={s.get("pending_uploads")} fps={s.get("fps")} '
                      + ' '.join(f'{k}={v}' for k, v in stages.items()), flush=True)
            if s.get('terrain_loading') is not None and label not in near:
                if s['terrain_loading']:
                    loading_seen, quiet_since = True, None
                elif loading_seen:
                    quiet_since = quiet_since or time.monotonic()
                    if time.monotonic() - quiet_since >= 2.0:
                        near[label] = quiet_since - since
            lod = s.get('cube_lod') or {}
            if s.get('fps') is not None:
                fps.append(float(s['fps']))
            last = lod
            if args.trace and int((time.monotonic() - since) * 10) % 10 == 0:
                print(f'  t={time.monotonic() - since:5.1f} ' + ' '.join(f'{k}={v}' for k, v in lod.items() if not isinstance(v, (dict, list))), flush=True)
            peak['arena'] = max(peak['arena'], lod.get('arena_bytes') or 0)
            peak['pool'] = max(peak['pool'], lod.get('pool_bytes') or 0)
            peak['tiles'] = max(peak['tiles'], lod.get('tile_bytes') or 0)
            # The plan changed after the teleport: this is the new view's fill.
            changed_ago = lod.get('plan_changed_ago_s')
            complete_ago = lod.get('plan_complete_ago_s')
            covered_ago = lod.get('plan_covered_ago_s')
            elapsed = time.monotonic() - since
            fresh = changed_ago is not None and (changed_ago <= elapsed + 0.05 or (not moved and elapsed > 1.0))
            if cover is None and fresh and covered_ago is not None:
                cover = max(0.0, elapsed - covered_ago)
            if lod.get('complete') and fresh and complete_ago is not None:
                done = elapsed - complete_ago
                if cover is None:
                    cover = done
                limited = ' (budget)' if lod.get('limited') else ''
                tiles = f', {lod.get("tiles")} of them tiles' if lod.get('surface') else ''
                print(f'{label}: covered in {cover:.2f}s, complete in {done:.2f}s, {lod.get("planned")} nodes '
                      f'planned{tiles}, {lod.get("uniform_skipped")} uniform, {lod.get("voxel_px")} px{limited}, '
                      f'gpu={lod.get("gpu_path")}', flush=True)
                return done, cover, fps, lod
            time.sleep(0.1)
        print(f'{label}: timed out after {args.timeout}s: {json.dumps(last)}', flush=True)
        return None, cover, fps, last

    def wait_near(label, since):
        """Waits on for the near terrain when the far terrain was complete first (see `wait_fill`)."""
        loading_seen, quiet_since = True, None
        while label not in near and time.monotonic() - since < args.timeout:
            s = status()
            if s.get('terrain_loading') is None:
                return
            if s['terrain_loading']:
                quiet_since = None
            else:
                quiet_since = quiet_since or time.monotonic()
                if time.monotonic() - quiet_since >= 2.0:
                    near[label] = quiet_since - since
            time.sleep(0.1)

    out = ctl_run('launch', '--game', game, '--world', 'FarBench', '--seed', args.seed, '--creative',
                  '--offscreen', '--wait', '--cubic', '--resources', args.resources, '--game-dir', game_dir,
                  '--width', args.width, '--height', args.height, timeout=600)
    print(out[:200], flush=True)
    result = {'ok': True}
    try:
        ctl_run('command', 'gamemode spectator')
        ctl_run('look', 0, 20)
        t0 = time.monotonic()
        spawn, spawn_cover, fps_all, lod = wait_fill('spawn', t0)
        wait_near('spawn', t0)
        result['spawn_fill_s'] = spawn
        result['spawn_cover_s'] = spawn_cover
        hops, covers = [], []
        for i in range(args.hops):
            x = (i + 1) * args.hop
            ctl_run('tp', x, args.y, 0)
            t = time.monotonic()
            fill, cover, fps, lod = wait_fill(f'hop {i + 1} (x {x})', t)
            wait_near(f'hop {i + 1} (x {x})', t)
            if f'hop {i + 1} (x {x})' in near:
                print(f'hop {i + 1}: near terrain in after {near[f"hop {i + 1} (x {x})"]:.2f}s', flush=True)
            fps_all += fps
            hops.append(fill)
            covers.append(cover)
            ctl_run('screenshot', scratch / f'hop{i + 1}.png')
        result['hops'] = hops
        result['covers'] = covers
        result['near'] = [near.get(k) for k in sorted(near, key=lambda k: (k != 'spawn', k))]
        good = [h for h in hops if h is not None]
        result['far_fill_s'] = statistics.mean(good) if len(good) == len(hops) and good else None
        good = [c for c in covers if c is not None]
        result['cover_s'] = statistics.mean(good) if len(good) == len(covers) and good else None
        result['ok'] = result['far_fill_s'] is not None and spawn is not None
        result['fps_min'] = min(fps_all) if fps_all else None
        result['fps_mean'] = statistics.mean(fps_all) if fps_all else None
        # The frame rate once the view is complete and nothing loads.
        steady = []
        end = time.monotonic() + args.steady
        while time.monotonic() < end:
            s = status()
            if s.get('fps') is not None:
                steady.append(float(s['fps']))
            time.sleep(0.2)
        result['fps_steady'] = statistics.median(steady) if steady else None
        for key in ('planned', 'gpu_path', 'target_px', 'voxel_px', 'budget', 'limited', 'plan_ms', 'surface',
                    'tiles', 'tiles_resident', 'tiles_complex', 'tiles_limited'):
            result[key] = lod.get(key)
        result['detail'] = args.detail
        result['arena_mb'] = (lod.get('arena_bytes', 0) + lod.get('pool_bytes', 0)) / 2**20
        result['peak_arena_mb'] = peak['arena'] / 2**20
        result['peak_pool_mb'] = peak['pool'] / 2**20
        result['peak_tile_mb'] = peak['tiles'] / 2**20
        if args.shots:
            x = args.hops * args.hop
            for name, y, pitch in [('horizon_y300', 300, 0), ('down_y1000', 1000, 60), ('down_y10000', 10000, 60)] + [
                    (f'down_y{y}', y, 60) for y in args.shot_heights]:
                ctl_run('tp', x, y, 0)
                ctl_run('look', 0, pitch)
                wait_fill(name, time.monotonic(), moved=False)
                time.sleep(1.0)
                ctl_run('screenshot', scratch / f'{name}.png')
    finally:
        ctl_run('quit')
    print(json.dumps(result), flush=True)
    sys.exit(0 if result['ok'] else 1)


if __name__ == '__main__':
    main()
