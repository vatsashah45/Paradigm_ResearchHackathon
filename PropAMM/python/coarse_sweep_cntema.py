#!/usr/bin/env python3
"""
Coarse sweep of CNT_EMA_FEE_K: cnt_ema-based fee boost.

Key insight from analysis:
  - cnt_ema correlates with norm_fee: r=0.662
  - When norm_fee is HIGH (Q5, 75 BPS): arbs prefer us 90.5% of time (our fee < norm_fee)
  - Avg missed retail revenue when norm_fee=75: 322 edge/sim
  - Raising fee_floor proportionally to cnt_ema:
      (a) reduces arb losses (arbs prefer normalizer as our fee rises toward norm_fee)
      (b) captures more retail revenue (charge closer to norm_fee)

Implementation:
  const CNT_EMA_FEE_K: u128 = K * BPS;  (BPS = SCALE/10000 = 100_000)
  After make_target calls:
    let cnt_boost = wmul(CNT_EMA_FEE_K, (cnt_ema as u128).min(SCALE));
    let max_target = target0.max(target1).max(target2) + cnt_boost;

  At cnt_ema = SCALE (max): boost = K * BPS = K fee units in BPS
  At cnt_ema = SCALE/2:      boost = K/2 BPS

Baseline: 501.62
"""

import subprocess
import re
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BINARY = os.path.join(_ROOT, 'target', 'release', 'prop-amm')
STRAT = os.path.join(_ROOT, 'strategy.rs')
TMP = '/tmp/strat_cntema.rs'
CARGO_BIN = os.path.expanduser('~/.cargo/bin')
BASELINE = 501.62

CONST_ANCHOR = 'const CNT_EMA_DECAY: u128 = 950_000_000;'
MAX_TARGET_OLD = ('    let max_target = target0.max(target1).max(target2);\n'
                  '    if max_target > fee_floor { fee_floor = max_target; }')
MAX_TARGET_NEW = ('    let cnt_boost = wmul(CNT_EMA_FEE_K, (cnt_ema as u128).min(SCALE));\n'
                  '    let max_target = target0.max(target1).max(target2) + cnt_boost;\n'
                  '    if max_target > fee_floor { fee_floor = max_target; }')


def patch_and_run(k_bps: int) -> float:
    with open(STRAT) as f:
        content = f.read()

    # Insert constant after CNT_EMA_DECAY
    content = content.replace(
        CONST_ANCHOR,
        CONST_ANCHOR + f'\nconst CNT_EMA_FEE_K: u128 = {k_bps} * BPS;'
    )
    # Modify max_target to include cnt_boost
    content = content.replace(MAX_TARGET_OLD, MAX_TARGET_NEW)

    with open(TMP, 'w') as f:
        f.write(content)

    env = os.environ.copy()
    env['RAYON_NUM_THREADS'] = '10'
    env['PATH'] = CARGO_BIN + ':' + env.get('PATH', '')

    r = subprocess.run(
        [BINARY, 'run', TMP, '--simulations', '1000',
         '--seed-start', '0', '--seed-stride', '1', '--workers', '10'],
        capture_output=True, text=True, env=env
    )
    out = r.stdout + r.stderr
    for line in out.splitlines():
        m = re.search(r'Avg edge:\s*(-?[\d.]+)', line)
        if m:
            return float(m.group(1))
    # fallback
    for line in out.splitlines():
        m = re.search(r'edge[:\s]+([\d.]+)', line, re.I)
        if m:
            return float(m.group(1))
    print(f"  STDOUT: {r.stdout[:400]}", file=sys.stderr)
    print(f"  STDERR: {r.stderr[:400]}", file=sys.stderr)
    raise RuntimeError("No edge in output")


def main():
    # Verify patch points
    with open(STRAT) as f:
        content = f.read()

    if CONST_ANCHOR not in content:
        print(f"ERROR: const anchor not in strategy.rs:\n  {CONST_ANCHOR!r}")
        return
    if MAX_TARGET_OLD not in content:
        print("ERROR: max_target pattern not in strategy.rs")
        for i, line in enumerate(content.splitlines()):
            if 'max_target' in line and 'target0' in line:
                print(f"  Line {i+1}: {line!r}")
        return
    print("Patch points verified OK.")
    print(f"Baseline: {BASELINE}\n")

    # K BPS boost at cnt_ema=SCALE (max arrival rate)
    # Typical cnt_ema: ~500M-1000M. At SCALE=1e9, the boost scales linearly.
    test_values = [5, 10, 15, 20, 30, 50]

    print(f"  {'K(BPS)':<10} {'Edge':<10} {'Delta':<10} Note")
    print('  ' + '-' * 52)

    best_k = None
    best_edge = BASELINE

    for k in test_values:
        try:
            edge = patch_and_run(k)
            delta = edge - BASELINE
            note = '+++ IMPROVED!' if delta > 1.0 else ('+ marginal' if delta > 0 else ('slight loss' if delta > -2 else '--- regression'))
            print(f"  K={k:<8} {edge:<10.2f} {delta:+.2f}      {note}")
            if edge > best_edge:
                best_edge = edge
                best_k = k
        except Exception as e:
            print(f"  K={k}: ERROR - {e}")

    print()
    if best_k is not None:
        print(f"Best: K={best_k} → edge={best_edge:.2f} (+{best_edge-BASELINE:.2f} vs baseline {BASELINE})")
    else:
        print("No improvement found. cnt_ema fee boost does not help.")


if __name__ == '__main__':
    main()
