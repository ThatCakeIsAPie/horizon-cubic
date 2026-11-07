#!/usr/bin/env python3
"""Screens HORIZON_FAR_* settings with one launch each (tools/far_fill_eval.py --no-build --launches 1).

    python research/far-fill/sweep.py --resources <26.3> --out research/far-fill/sweep1.txt \
        "control" "HORIZON_FAR_WORKERS=6" "HORIZON_FAR_GPU_BATCH=128 HORIZON_FAR_POOL_NEED=4" ...

One line per setting: its name, SCORE, GUARD and the eval's summary JSON.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--resources', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--launches', type=int, default=1)
    ap.add_argument('settings', nargs='+')
    args = ap.parse_args()
    for setting in args.settings:
        env = {k: v for k, v in os.environ.items() if not k.startswith('HORIZON_FAR_')}
        if setting != 'control':
            for pair in setting.split():
                k, v = pair.split('=', 1)
                env[k] = v
        p = subprocess.run([sys.executable, str(ROOT / 'tools' / 'far_fill_eval.py'), '--resources', args.resources,
                            '--no-build', '--launches', str(args.launches)],
                           cwd=ROOT, env=env, capture_output=True, text=True)
        lines = p.stdout.splitlines()
        score = next((l for l in lines if l.startswith('SCORE:')), 'SCORE: ?')
        guard = next((l for l in lines if l.startswith('GUARD:')), 'GUARD: ?')
        summary = next((l for l in lines if l.startswith('{')), '{}')
        launches = [l for l in lines if l.startswith('launch')]
        with open(args.out, 'a') as f:
            f.write(f'{setting} | {score} | {guard} | {summary}\n')
            for l in launches:
                f.write(f'    {l}\n')
        print(f'{setting} | {score} | {guard}', flush=True)


if __name__ == '__main__':
    main()
