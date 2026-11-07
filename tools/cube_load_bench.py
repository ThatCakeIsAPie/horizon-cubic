#!/usr/bin/env python3
"""Near-terrain loading benchmark of a cubic world: how fast cubes are generated and populated.

The overnight loop's lane B (research/cube-load/instructions.md). It builds the client library's
tests once, then runs the ignored test `cubic::tests::cube_load_bench` --runs times. That test makes
the cubes of a block around spawn in a fresh Custom world, as a teleport there loads them, and prints
the time per cube and a hash of every block.

    python3 tools/cube_load_bench.py --resources <prepared 26.3 resources> [--runs 5] [--jobs 4]

Prints, last:
    SCORE: <median ms per cube>        (lower is better)
    GUARD: PASS | FAIL <why>           (every run's hash equals research/cube-load/baseline.json's)
and exits 1 on a build failure, a crash or a failed guard. `--write-baseline` records this build's
hash (and its score) as the baseline; `--tests` also runs the cubic generator's unit tests.

The scratch levels go in `--tmp` (default: research/cube-load/tmp in the checkout), never /tmp:
on a machine whose /tmp is a RAM disk they would eat memory.
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
BASELINE = ROOT / 'research' / 'cube-load' / 'baseline.json'
TEST = 'cubic::tests::cube_load_bench'


def cargo_cmd(args):
    if args.cargo:
        return args.cargo.split()
    # The desktop's default toolchain is too old; the laptop has a system cargo.
    return ['cargo', '+stable-x86_64-pc-windows-msvc'] if os.name == 'nt' else ['cargo']


def build(args, env):
    """Builds the client library's tests; returns the test executable's path, or None."""
    cmd = cargo_cmd(args) + ['test', '-p', 'horizon-client-lib', '--lib', '--profile', args.profile,
                             '--no-run', '--message-format=json', '-j', str(args.jobs)]
    if args.target_dir:
        cmd += ['--target-dir', args.target_dir]
    started = time.monotonic()
    p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    exe = None
    for line in p.stdout.splitlines():
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if msg.get('reason') == 'compiler-artifact' and msg.get('executable') \
                and msg.get('target', {}).get('name') == 'horizon_client_lib':
            exe = msg['executable']
    print(f'build: {time.monotonic() - started:.0f} s, exit {p.returncode}', flush=True)
    if p.returncode != 0 or not exe:
        errors = [l for l in p.stderr.splitlines() if l.startswith('error') or l.lstrip().startswith('-->')]
        print('\n'.join(errors[:30]) or p.stderr[-3000:], file=sys.stderr)
        return None
    return exe


def run_once(exe, env):
    p = subprocess.run([exe, TEST, '--exact', '--ignored', '--nocapture', '--test-threads', '1'],
                       cwd=ROOT, env=env, capture_output=True, text=True, timeout=1800)
    for line in p.stdout.splitlines():
        # libtest prints the test's name on the same line before its output.
        at = line.find('CUBE_BENCH {')
        if at >= 0:
            return json.loads(line[at + len('CUBE_BENCH '):])
    print(p.stdout[-2000:], p.stderr[-2000:], file=sys.stderr)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--resources', required=True, help="prepared 26.3 resources (the launcher's .local/horizon-client/26.3)")
    ap.add_argument('--runs', type=int, default=5)
    ap.add_argument('--spec', default='4,-4,8,1', help='RADIUS,Y_LOW,Y_HIGH,THREADS in cubes (HORIZON_CUBE_BENCH)')
    ap.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 4) // 2), help='cargo -j')
    ap.add_argument('--profile', default='fast')
    ap.add_argument('--target-dir', help='cargo target dir (default: the checkout\'s)')
    ap.add_argument('--cargo', help='the cargo command, e.g. "cargo +stable"')
    ap.add_argument('--tmp', default=str(ROOT / 'research' / 'cube-load' / 'tmp'), help='scratch folder (TMPDIR)')
    ap.add_argument('--tests', action='store_true', help="also run the generator's unit tests (cubic, cubic_gen, horizon-world cubic)")
    ap.add_argument('--write-baseline', action='store_true')
    args = ap.parse_args()

    tmp = Path(args.tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, HORIZON_TEST_RESOURCES=args.resources, HORIZON_CUBE_BENCH=args.spec,
               TMPDIR=str(tmp), TMP=str(tmp), TEMP=str(tmp))
    exe = build(args, env)
    if not exe:
        print('SCORE: crash')
        print('GUARD: FAIL build')
        sys.exit(1)

    results = []
    for i in range(args.runs):
        r = run_once(exe, env)
        if r is None:
            print('SCORE: crash')
            print('GUARD: FAIL the benchmark crashed')
            sys.exit(1)
        print(f"run {i + 1}: {r['ms_per_cube']:.4f} ms/cube ({r['seconds']:.2f} s, {r['cubes']} cubes, "
              f"generate {r['generate_micros']:.0f} us x {r['generated']}, "
              f"populate {r['populate_micros']:.0f} us x {r['populations']}) hash {r['hash']}", flush=True)
        results.append(r)

    times = [r['ms_per_cube'] for r in results]
    median = statistics.median(times)
    hashes = {r['hash'] for r in results}
    threads = results[0]['threads']
    guard = 'PASS'
    if threads == 1 and len(hashes) != 1:
        guard = f'FAIL the hash changed between runs: {sorted(hashes)}'
    baseline = json.loads(BASELINE.read_text()) if BASELINE.exists() else None
    if args.write_baseline:
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(json.dumps({'spec': args.spec, 'hash': results[0]['hash'], 'ms_per_cube': median,
                                        'runs': times}, indent=2) + '\n')
        print(f'baseline written: {BASELINE}')
    elif baseline and threads == 1:
        if baseline.get('spec') != args.spec:
            guard = f"FAIL --spec {args.spec} is not the baseline's {baseline.get('spec')}"
        elif results[0]['hash'] != baseline['hash']:
            guard = f"FAIL the terrain changed: hash {results[0]['hash']}, baseline {baseline['hash']}"
    elif not baseline:
        guard = 'FAIL no baseline yet (run with --write-baseline on the unchanged code)'

    if args.tests and guard == 'PASS':
        for pkg, filt in (('horizon-client-lib', 'cubic'), ('horizon-world', 'cubic')):
            cmd = cargo_cmd(args) + ['test', '-p', pkg, '--lib', '--profile', args.profile, '-j', str(args.jobs)]
            if args.target_dir:
                cmd += ['--target-dir', args.target_dir]
            cmd += ['--', filt]
            p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
            summary = [l for l in p.stdout.splitlines() if l.startswith('test result')]
            print(f'tests {pkg} {filt}: exit {p.returncode} {summary[-1] if summary else ""}', flush=True)
            if p.returncode != 0:
                failed = [l for l in p.stdout.splitlines() if 'FAILED' in l or 'panicked' in l]
                print('\n'.join(failed[:20]), file=sys.stderr)
                guard = f'FAIL tests in {pkg}'
                break

    print(json.dumps({'ms_per_cube': median, 'min': min(times), 'max': max(times), 'runs': times,
                      'hash': results[0]['hash'], 'threads': threads,
                      'baseline_ms_per_cube': baseline and baseline.get('ms_per_cube')}))
    print(f'SCORE: {median:.4f}')
    print(f'GUARD: {guard}')
    sys.exit(0 if guard == 'PASS' else 1)


if __name__ == '__main__':
    main()
