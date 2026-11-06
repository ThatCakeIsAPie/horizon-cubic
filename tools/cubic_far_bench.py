#!/usr/bin/env python3
"""Far-fill benchmark of a cubic world: how fast distant terrain loads (docs/cubic-v2-plan.md).

    python3 tools/cubic_far_bench.py --bin-dir D:/ClaudeData/horizon-cubic-target/fast \
        --resources <prepared 26.3 resources>

The script launches a cubic world offscreen with LOD Distance at Whole World and the far terrain on the
GPU when the device allows. It waits for the far terrain around spawn to complete, then teleports
`--hops` times far away (5,000,000 blocks apart by default). Each time it measures how long the
cube octree takes to become complete again:
- every planned node is drawn or known empty;
- nothing is waiting in the GPU mesher.

This is `status`'s `cube_lod.complete`, from `CubeLodStats` in `lod/cube.rs`.

The last line printed is a single JSON object, the metric for autoresearch loops:
    {"far_fill_s": <mean seconds per hop>, "spawn_fill_s": ..., "hops": [...], "fps_min": ..., ...}
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
    ap.add_argument('--width', type=int, default=1280)
    ap.add_argument('--height', type=int, default=720)
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
    (game_dir / 'horizon-graphics.txt').write_text(
        f'lodDistance:{args.lod}\nfarDetail:full\ncubicVertical:{args.vertical}\ngpuFarTerrain:{args.gpu}\n')
    # Vanilla's inactivity limit drops to 30 fps after a minute without input, which a
    # script never gives: keep it to minimized windows.
    (game_dir / 'options.txt').write_text('inactivityFpsLimit:"minimized"'+chr(10))
    env = dict(os.environ, HORIZON_HOME=str(home), HORIZON_NO_AUDIO='1')

    def ctl_run(*argv, timeout=60):
        r = subprocess.run([str(ctl), *map(str, argv)], env=env, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip()

    def status():
        try:
            return json.loads(ctl_run('status'))
        except (json.JSONDecodeError, subprocess.TimeoutExpired):
            return {}

    def wait_fill(label, since):
        """Seconds from `since` until the far terrain is complete again, with the fps seen meanwhile."""
        fps, last = [], {}
        while time.monotonic() - since < args.timeout:
            s = status()
            lod = s.get('cube_lod') or {}
            if s.get('fps') is not None:
                fps.append(float(s['fps']))
            last = lod
            # Complete, and the plan changed after the teleport: this is the new view's fill.
            changed_ago = lod.get('plan_changed_ago_s')
            complete_ago = lod.get('plan_complete_ago_s')
            elapsed = time.monotonic() - since
            if (lod.get('complete') and changed_ago is not None and complete_ago is not None
                    and changed_ago <= elapsed + 0.05):
                done = elapsed - complete_ago
                print(f'{label}: complete in {done:.2f}s, {lod.get("planned")} nodes planned, '
                      f'{lod.get("uniform_skipped")} uniform, gpu={lod.get("gpu_path")}', flush=True)
                return done, fps, lod
            time.sleep(0.1)
        print(f'{label}: timed out after {args.timeout}s: {json.dumps(last)}', flush=True)
        return None, fps, last

    out = ctl_run('launch', '--game', game, '--world', 'FarBench', '--seed', args.seed, '--creative',
                  '--offscreen', '--wait', '--cubic', '--resources', args.resources, '--game-dir', game_dir,
                  '--width', args.width, '--height', args.height, timeout=600)
    print(out[:200], flush=True)
    result = {'ok': True}
    try:
        ctl_run('command', 'gamemode spectator')
        ctl_run('look', 0, 20)
        t0 = time.monotonic()
        spawn, fps_all, lod = wait_fill('spawn', t0)
        result['spawn_fill_s'] = spawn
        hops = []
        for i in range(args.hops):
            x = (i + 1) * args.hop
            ctl_run('tp', x, args.y, 0)
            t = time.monotonic()
            fill, fps, lod = wait_fill(f'hop {i + 1} (x {x})', t)
            fps_all += fps
            hops.append(fill)
            ctl_run('screenshot', scratch / f'hop{i + 1}.png')
        result['hops'] = hops
        good = [h for h in hops if h is not None]
        result['far_fill_s'] = statistics.mean(good) if len(good) == len(hops) and good else None
        result['ok'] = result['far_fill_s'] is not None and spawn is not None
        result['fps_min'] = min(fps_all) if fps_all else None
        result['fps_mean'] = statistics.mean(fps_all) if fps_all else None
        result['planned'] = lod.get('planned')
        result['gpu_path'] = lod.get('gpu_path')
        result['arena_mb'] = (lod.get('arena_bytes', 0) + lod.get('pool_bytes', 0)) / 2**20
    finally:
        ctl_run('quit')
    print(json.dumps(result), flush=True)
    sys.exit(0 if result['ok'] else 1)


if __name__ == '__main__':
    main()
