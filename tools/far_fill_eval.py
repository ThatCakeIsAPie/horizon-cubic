#!/usr/bin/env python3
"""Far-view fill benchmark for the overnight loop's lane A (research/far-fill/instructions.md).

Builds the game (fast profile) unless --no-build, then runs tools/cubic_far_bench.py --launches times
(each a fresh offscreen game: spawn fill, then --hops teleports 5,000,000 blocks apart) and prints:
    SCORE: <median far_fill_s over every hop of every launch>   (seconds, lower is better)
    GUARD: PASS | FAIL <why>
The guards keep the far view's quality from being traded for speed. Against
research/far-fill/baseline.json (`--write-baseline` records it), every launch must:
- complete every fill (no timeout);
- plan within 3% as many nodes as the baseline, at a voxel size no coarser than 2% over it;
- keep its steady frame rate within 10%;
- keep its median cover time within 20% (+0.1 s).

    python tools/far_fill_eval.py --resources <prepared 26.3> [--target-dir D:/ClaudeData/horizon-cubic-target]
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / 'research' / 'far-fill' / 'baseline.json'
EXE = '.exe' if os.name == 'nt' else ''


def build(args):
    cargo = args.cargo.split() if args.cargo else (
        ['cargo', '+stable-x86_64-pc-windows-msvc'] if os.name == 'nt' else ['cargo'])
    cmd = cargo + ['build', '--profile', 'fast', '-p', 'horizon-client', '-p', 'horizon-client-lib',
                   '--bin', 'horizon-client', '--bin', 'horizon-ctl', '-j', str(args.jobs),
                   '--target-dir', args.target_dir]
    started = time.monotonic()
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    print(f'build: {time.monotonic() - started:.0f} s, exit {p.returncode}', flush=True)
    if p.returncode != 0:
        errors = [l for l in p.stderr.splitlines() if l.startswith('error') or l.lstrip().startswith('-->')]
        print('\n'.join(errors[:30]) or p.stderr[-3000:], file=sys.stderr)
        return False
    return True


def launch(args, bin_dir, i):
    out = Path(args.out) / f'launch{i}'
    cmd = [sys.executable, str(ROOT / 'tools' / 'cubic_far_bench.py'), '--bin-dir', str(bin_dir),
           '--resources', args.resources, '--detail', args.detail, '--hops', str(args.hops),
           '--timeout', str(args.timeout), '--out', str(out)]
    env = dict(os.environ)
    p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=args.timeout * (args.hops + 2) + 600)
    lines = [l for l in p.stdout.splitlines() if l.strip()]
    for line in lines:
        if line.startswith(('spawn', 'hop', 'teleport')):
            print('  ' + line[:200], flush=True)
    try:
        return json.loads(lines[-1])
    except (IndexError, json.JSONDecodeError):
        print(p.stdout[-2000:], p.stderr[-2000:], file=sys.stderr)
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--resources', required=True)
    ap.add_argument('--target-dir', default=str(ROOT.parent / 'horizon-cubic-target'))
    ap.add_argument('--bin-dir', help='use this build instead of <target-dir>/fast (with --no-build)')
    ap.add_argument('--no-build', action='store_true')
    ap.add_argument('--cargo')
    ap.add_argument('--jobs', type=int, default=os.cpu_count() or 8)
    ap.add_argument('--launches', type=int, default=2)
    ap.add_argument('--hops', type=int, default=3)
    ap.add_argument('--detail', default='high')
    ap.add_argument('--timeout', type=float, default=180.0)
    ap.add_argument('--out', default=str(ROOT / 'research' / 'far-fill' / 'tmp'))
    ap.add_argument('--write-baseline', action='store_true')
    args = ap.parse_args()

    if not args.no_build and not build(args):
        print('SCORE: crash')
        print('GUARD: FAIL build')
        sys.exit(1)
    bin_dir = Path(args.bin_dir) if args.bin_dir else Path(args.target_dir) / 'fast'

    runs = []
    for i in range(args.launches):
        r = launch(args, bin_dir, i)
        if r is None:
            print('SCORE: crash')
            print('GUARD: FAIL the benchmark crashed')
            sys.exit(1)
        print(f"launch {i + 1}: far_fill_s {r.get('far_fill_s')} hops {r.get('hops')} cover {r.get('covers')} "
              f"spawn {r.get('spawn_fill_s')} planned {r.get('planned')} voxel_px {r.get('voxel_px')} "
              f"fps {r.get('fps_steady')}/{r.get('fps_min')}", flush=True)
        runs.append(r)

    guard = 'PASS'
    if not all(r.get('ok') for r in runs):
        guard = 'FAIL a fill timed out'
    hops = [h for r in runs for h in (r.get('hops') or []) if h is not None]
    covers = [c for r in runs for c in (r.get('covers') or []) if c is not None]
    score = statistics.median(hops) if hops else None
    summary = {
        'far_fill_s': score,
        'hops': hops,
        'cover_s': statistics.median(covers) if covers else None,
        'spawn_fill_s': statistics.median([r['spawn_fill_s'] for r in runs if r.get('spawn_fill_s') is not None] or [0]),
        'planned': statistics.median([r.get('planned') or 0 for r in runs]),
        'voxel_px': max(r.get('voxel_px') or 0 for r in runs),
        'fps_steady': statistics.median([r.get('fps_steady') or 0 for r in runs]),
    }
    baseline = json.loads(BASELINE.read_text()) if BASELINE.exists() else None
    if args.write_baseline:
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(json.dumps(summary, indent=2) + '\n')
        print(f'baseline written: {BASELINE}')
    elif not baseline:
        guard = 'FAIL no baseline yet (run with --write-baseline on the unchanged code)'
    elif guard == 'PASS':
        b = baseline
        if abs(summary['planned'] - b['planned']) > 0.03 * b['planned']:
            guard = f"FAIL planned {summary['planned']} vs baseline {b['planned']} (more than 3% off)"
        elif summary['voxel_px'] > b['voxel_px'] * 1.02:
            guard = f"FAIL voxel_px {summary['voxel_px']} coarser than baseline {b['voxel_px']}"
        elif summary['fps_steady'] < 0.9 * b['fps_steady']:
            guard = f"FAIL fps_steady {summary['fps_steady']} under 90% of baseline {b['fps_steady']}"
        elif summary['cover_s'] is not None and b.get('cover_s') is not None \
                and summary['cover_s'] > 1.2 * b['cover_s'] + 0.1:
            guard = f"FAIL cover_s {summary['cover_s']} over baseline {b['cover_s']} by more than 20%"
    print(json.dumps(summary))
    print(f'SCORE: {score:.3f}' if score is not None else 'SCORE: crash')
    print(f'GUARD: {guard}')
    sys.exit(0 if guard == 'PASS' and score is not None else 1)


if __name__ == '__main__':
    main()
