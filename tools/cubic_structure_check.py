#!/usr/bin/env python3
"""Screenshots of a Custom Cubic world's structures (docs/cubic-v2-plan.md, "Structures in Custom Cubic").

    python3 tools/cubic_structure_check.py --bin-dir D:/ClaudeData/horizon-cubic-target/fast \\
        --resources <prepared 26.3 resources> --out <scratch folder> --seed 1 \\
        --view=-243,105,-870,180,30 --view=-243,170,-915,0,90

Launches a new Custom Cubic world offscreen with its own HORIZON_HOME, game dir and control socket in
--out (as tools/cubic_far_check.py does), in spectator mode. For each `--view X,Y,Z,YAW,PITCH` it
teleports there, waits until the near terrain has loaded and settled, and screenshots
`view_<n>.png`. `--blocks X0,Y0,Z0,X1,Y1,Z1` prints the block counts of a box after the views (what a
structure placed). Prints the log's errors at the end.

Far views: `--lod CHUNKS` sets the LOD Distance (far terrain on the GPU where it can be), and
`--far-wait S` waits up to S seconds at each view for the far view to complete and for the structures
the background search finds to be drawn into it (`cube_lod.late_refreshed` unchanged for 10 s).
`--command CMD` runs a command after the world opens and prints what it answered (`locate structure
minecraft:village_snowy`).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

EXE = '.exe' if sys.platform == 'win32' else ''


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bin-dir', required=True, help='folder with horizon-client and horizon-ctl')
    ap.add_argument('--resources', required=True, help='prepared 26.3 resources')
    ap.add_argument('--out', required=True, help='scratch folder: game, home, screenshots')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--view', action='append', default=[], help='X,Y,Z,YAW,PITCH to screenshot from')
    ap.add_argument('--blocks', action='append', default=[], help='X0,Y0,Z0,X1,Y1,Z1 box to count blocks in')
    ap.add_argument('--settings', help='a cubicgen.json to generate with')
    ap.add_argument('--size', default='1280x720', help='the window, WIDTHxHEIGHT')
    ap.add_argument('--timeout', type=float, default=120.0, help='seconds the near terrain may take')
    ap.add_argument('--lod', type=int, help='LOD Distance in chunks (far views)')
    ap.add_argument('--far-wait', type=float, default=0.0,
                    help='seconds to wait at each view for the far view and its structures')
    ap.add_argument('--command', action='append', default=[], help='a command to run, its answer printed')
    ap.add_argument('--no-clouds', action='store_true', help='clouds off (far views)')
    args = ap.parse_args()

    bins = Path(args.bin_dir)
    ctl, game = bins / f'horizon-ctl{EXE}', bins / f'horizon-client{EXE}'
    scratch = Path(args.out)
    home, game_dir = scratch / 'home', scratch / 'game'
    for d in (home, game_dir):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    (game_dir / 'options.txt').write_text(
        'inactivityFpsLimit:"minimized"\n' + ('renderClouds:"false"\n' if args.no_clouds else ''))
    if args.lod:
        (game_dir / 'horizon-graphics.txt').write_text(
            f'lodDistance:{args.lod}\nfarDetail:full\ngpuFarTerrain:auto\n')
    env = dict(os.environ, HORIZON_HOME=str(home), HORIZON_NO_AUDIO='1')
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

    def launch():
        w, h = args.size.split('x')
        print(ctl_run('launch', '--game', game, '--world', 'StructureCheck', '--seed', args.seed, '--creative',
                      '--offscreen', '--wait', '--cubic', '--resources', args.resources, '--game-dir', game_dir,
                      '--width', w, '--height', h, timeout=600)[:120], flush=True)

    def loaded(label, since):
        """Waits until the near terrain is loaded: after the move shows (3 s), four quiet polls in a row."""
        time.sleep(3)
        quiet = 0
        while time.monotonic() - since < args.timeout:
            s = status()
            quiet = quiet + 1 if s and not s.get('terrain_loading', True) else 0
            if quiet >= 4:
                print(f'{label}: near terrain in {time.monotonic() - since:.1f} s', flush=True)
                time.sleep(1)
                return
            time.sleep(0.5)
        print(f'{label}: near terrain still loading after {args.timeout} s', flush=True)

    def far_done(label, since):
        """Waits for the far view to complete and its structures to stop arriving."""
        last, steady = None, time.monotonic()
        while time.monotonic() - since < args.far_wait:
            lod = status().get('cube_lod') or {}
            late = lod.get('late_refreshed')
            if late != last:
                last, steady = late, time.monotonic()
            if lod.get('complete') and time.monotonic() - steady >= 10:
                break
            time.sleep(1)
        print(f'{label}: far view complete={lod.get("complete")}, nodes redrawn for structures '
              f'{last} after {time.monotonic() - since:.0f} s', flush=True)

    launch()
    if args.settings:
        ctl_run('quit')
        time.sleep(3)
        (game_dir / 'saves' / 'StructureCheck' / 'cubicgen.json').write_text(Path(args.settings).read_text())
        launch()
    try:
        ctl_run('command', 'gamemode spectator')
        for command in args.command:
            print(f'/{command}: {ctl_run("command", command, timeout=300)[:300]}', flush=True)
        for i, view in enumerate(args.view):
            x, y, z, yaw, pitch = (float(v) for v in view.split(','))
            ctl_run('tp', x, y, z)
            ctl_run('look', yaw, pitch)
            loaded(f'view {i} at {x:.0f} {y:.0f} {z:.0f}', time.monotonic())
            if args.far_wait:
                far_done(f'view {i}', time.monotonic())
            ctl_run('screenshot', scratch / f'view_{i}.png')
        for box in args.blocks:
            lo_hi = [int(v) for v in box.split(',')]
            try:
                counts = json.loads(ctl_run('blocks', *lo_hi, timeout=120)).get('counts', {})
            except (json.JSONDecodeError, subprocess.TimeoutExpired):
                counts = {}
            top = sorted(counts.items(), key=lambda kv: -kv[1])[:30]
            print(f'blocks {box}: ' + ', '.join(f'{k} {v}' for k, v in top), flush=True)
        log = ctl_run('logs')
        errors = [line for line in log.splitlines() if 'ERROR' in line or 'panic' in line.lower()]
        print(f'log errors: {len(errors)}', flush=True)
        for line in errors[:20]:
            print('  ' + line[:200])
        for line in log.splitlines():
            if 'cubic structures' in line:
                print(line[:300])
    finally:
        ctl_run('quit')
        time.sleep(2)


if __name__ == '__main__':
    main()
