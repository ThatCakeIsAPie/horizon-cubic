#!/usr/bin/env python3
"""Renders research/far-fill/autoresearch_dashboard.md from autoresearch.jsonl."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    rows = [json.loads(l) for l in (HERE / 'autoresearch.jsonl').read_text().splitlines() if l.strip()]
    config = next((r for r in rows if r.get('type') == 'config'), {})
    results = [r for r in rows if r.get('type') == 'result']
    baseline = config.get('baseline')
    kept = [r for r in results if r['status'] in ('keep', 'baseline')]
    best = min((r['score'] for r in results if r['status'] == 'keep' and r.get('score') is not None),
               default=baseline)
    counts = {s: sum(1 for r in results if r['status'] == s) for s in ('keep', 'discard', 'crash', 'guard_fail', 'screen')}
    lines = [
        '# Lane A: far view fill (desktop) - autoresearch dashboard',
        '',
        f"**Metric:** {config.get('metric')}",
        f"**Baseline:** {baseline} s | **Best:** {best} s"
        + (f" ({(best - baseline) / baseline * 100:+.1f}%)" if baseline and best else '')
        + f" | **Min-delta:** {config.get('min_delta_s')} s | **Iterations:** {max((r['iteration'] for r in results), default=0)}",
        f"**Guard:** {config.get('guard')}",
        '',
        f"**Kept:** {counts['keep']} | **Discarded:** {counts['discard']} | **Screened:** {counts['screen']} | "
        f"**Crashed:** {counts['crash']} | **Guard failures:** {counts['guard_fail']}",
        '',
        '| # | Score (s) | Delta | Status | Commit | Description |',
        '|---|---|---|---|---|---|',
    ]
    for r in results:
        score = '-' if r.get('score') is None else f"{r['score']:.3f}"
        lines.append(f"| {r['iteration']} | {score} | {r.get('delta') or ''} | {r['status']} | {r.get('commit') or ''} | "
                     f"{r.get('description', '').replace('|', '/')} |")
    (HERE / 'autoresearch_dashboard.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines[:8]))


if __name__ == '__main__':
    main()
