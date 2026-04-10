#!/usr/bin/env python3
"""
Coarse sweep of profile switching constants that have never been optimized.
Tests one parameter at a time with a small grid to identify promising directions.

Baseline: 501.62 (seeds 0-999, 1000 sims)

Constants covered:
  SWITCH_ABS_GAP_DOWN, ROUTE_BONUS_W, BARRIER_K, SCORE_BIASES[2],
  SWITCH_ABS_GAP_UP, SCORE_DECAY_LO, BIAS_ADAPT_1, BIAS_ADAPT_2,
  BIAS_SHOCK_THRESH

Usage:
  python3 python/coarse_sweep_switching.py [--workers 10]
"""

import argparse
import os
import re
import subprocess
import sys
import time

STRATEGY_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "strategy.rs")
PROP_AMM_BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "target/release/prop-amm")
CARGO_BIN = os.path.expanduser("~/.cargo/bin")
TMP_PATH = "/tmp/strat_coarse_switch.rs"

BASELINE = 501.62

def run_sim(strategy_path, n_sims=1000, seed_start=0, workers=10):
    env = os.environ.copy()
    env["RAYON_NUM_THREADS"] = str(workers)
    env["PATH"] = CARGO_BIN + ":" + env.get("PATH", "")
    cmd = [PROP_AMM_BIN, "run", strategy_path,
           "--simulations", str(n_sims),
           "--seed-start", str(seed_start),
           "--seed-stride", "1",
           "--workers", str(workers)]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    out = result.stdout + result.stderr
    for line in out.split("\n"):
        m = re.search(r"Avg edge:\s+(-?[\d.]+)", line)
        if m:
            return float(m.group(1))
    print(f"  ERROR parsing output:\n{out[:400]}", file=sys.stderr)
    return None

def replace_const(content, name, val_str):
    """Replace a simple scalar const."""
    pattern = rf"(const {name}:\s*(?:u128|i64|u64)\s*=\s*)[^;]+(;)"
    new_content = re.sub(pattern, rf"\g<1>{val_str}\g<2>", content)
    if new_content == content:
        raise ValueError(f"Could not find/replace const {name}")
    return new_content

def replace_score_biases(content, b0, b1, b2):
    """Replace the SCORE_BIASES array."""
    pattern = r"(const SCORE_BIASES:\s*\[i64;\s*NUM_PROFILES\]\s*=\s*\[)[^\]]+(\];)"
    replacement = rf"\g<1>\n    {b0}, {b1}, {b2},\n\g<2>"
    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    if new_content == content:
        raise ValueError("Could not replace SCORE_BIASES")
    return new_content

def patch_and_run(patches, n_sims=1000, workers=10):
    """Apply patches (list of (fn, args)) to strategy.rs, run, return edge."""
    with open(STRATEGY_PATH) as f:
        content = f.read()
    for patch_fn, args in patches:
        content = patch_fn(content, *args)
    with open(TMP_PATH, "w") as f:
        f.write(content)
    return run_sim(TMP_PATH, n_sims=n_sims, workers=workers)

def sweep(name, patches_list, workers=10):
    """
    patches_list: list of (label, [(patch_fn, args), ...])
    Runs each and prints delta vs baseline.
    Returns (best_label, best_edge, best_delta).
    """
    print(f"\n{'='*60}")
    print(f"Hypothesis: {name}")
    print(f"{'='*60}")
    best_edge = BASELINE
    best_label = "baseline"
    best_patches = []
    for label, patches in patches_list:
        t0 = time.time()
        edge = patch_and_run(patches, workers=workers)
        elapsed = time.time() - t0
        if edge is None:
            print(f"  {label}: ERROR")
            continue
        delta = edge - BASELINE
        marker = " <-- BEST" if edge > best_edge else ""
        print(f"  {label}: {edge:.2f}  (delta={delta:+.2f})  [{elapsed:.0f}s]{marker}")
        sys.stdout.flush()
        if edge > best_edge:
            best_edge = edge
            best_label = label
            best_patches = patches
    print(f"  -> Best: {best_label} = {best_edge:.2f} (delta={best_edge-BASELINE:+.2f})")
    return best_label, best_edge, best_patches

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--sims", type=int, default=1000)
    args = parser.parse_args()

    print(f"Coarse sweep: switching constants")
    print(f"Baseline: {BASELINE}, sims={args.sims}, workers={args.workers}")
    print(f"Strategy: {STRATEGY_PATH}")

    results = []

    # ----------------------------------------------------------------
    # Hypothesis A: SWITCH_ABS_GAP_DOWN
    # Once in P2, needs this gap to return to P1/P0. Maybe 210M is too sticky.
    # ----------------------------------------------------------------
    label, edge, _ = sweep("A: SWITCH_ABS_GAP_DOWN (baseline=210M)", [
        ("80M",  [(replace_const, ("SWITCH_ABS_GAP_DOWN", "80_000_000"))]),
        ("130M", [(replace_const, ("SWITCH_ABS_GAP_DOWN", "130_000_000"))]),
        ("170M", [(replace_const, ("SWITCH_ABS_GAP_DOWN", "170_000_000"))]),
        ("280M", [(replace_const, ("SWITCH_ABS_GAP_DOWN", "280_000_000"))]),
        ("360M", [(replace_const, ("SWITCH_ABS_GAP_DOWN", "360_000_000"))]),
    ], workers=args.workers)
    results.append(("SWITCH_ABS_GAP_DOWN", label, edge))

    # ----------------------------------------------------------------
    # Hypothesis B: ROUTE_BONUS_W
    # Bonus added to score for counter-dir trades. May distort profile selection.
    # ----------------------------------------------------------------
    label, edge, _ = sweep("B: ROUTE_BONUS_W (baseline=145M)", [
        ("60M",  [(replace_const, ("ROUTE_BONUS_W", "60_000_000"))]),
        ("100M", [(replace_const, ("ROUTE_BONUS_W", "100_000_000"))]),
        ("110M", [(replace_const, ("ROUTE_BONUS_W", "110_000_000"))]),
        ("190M", [(replace_const, ("ROUTE_BONUS_W", "190_000_000"))]),
        ("240M", [(replace_const, ("ROUTE_BONUS_W", "240_000_000"))]),
    ], workers=args.workers)
    results.append(("ROUTE_BONUS_W", label, edge))

    # ----------------------------------------------------------------
    # Hypothesis C: BARRIER_K
    # Step-boundary bonus/penalty for realized vs sigma over/under-prediction.
    # ----------------------------------------------------------------
    label, edge, _ = sweep("C: BARRIER_K (baseline=65M)", [
        ("20M",  [(replace_const, ("BARRIER_K", "20_000_000"))]),
        ("40M",  [(replace_const, ("BARRIER_K", "40_000_000"))]),
        ("90M",  [(replace_const, ("BARRIER_K", "90_000_000"))]),
        ("130M", [(replace_const, ("BARRIER_K", "130_000_000"))]),
        ("200M", [(replace_const, ("BARRIER_K", "200_000_000"))]),
    ], workers=args.workers)
    results.append(("BARRIER_K", label, edge))

    # ----------------------------------------------------------------
    # Hypothesis D: SCORE_BIASES[2] (P2 entry difficulty)
    # Current -210M base bias for P2. Test harder/easier entry.
    # ----------------------------------------------------------------
    label, edge, _ = sweep("D: SCORE_BIASES[2] (baseline=-210M)", [
        ("-120M", [(replace_score_biases, (0, -120_000_000, -120_000_000))]),
        ("-160M", [(replace_score_biases, (0, -120_000_000, -160_000_000))]),
        ("-270M", [(replace_score_biases, (0, -120_000_000, -270_000_000))]),
        ("-330M", [(replace_score_biases, (0, -120_000_000, -330_000_000))]),
    ], workers=args.workers)
    results.append(("SCORE_BIASES[2]", label, edge))

    # ----------------------------------------------------------------
    # Hypothesis E: SWITCH_ABS_GAP_UP
    # Gap required to enter more defensive profile. 30M may be too low (noisy trigger).
    # ----------------------------------------------------------------
    label, edge, _ = sweep("E: SWITCH_ABS_GAP_UP (baseline=30M)", [
        ("15M",  [(replace_const, ("SWITCH_ABS_GAP_UP", "15_000_000"))]),
        ("50M",  [(replace_const, ("SWITCH_ABS_GAP_UP", "50_000_000"))]),
        ("80M",  [(replace_const, ("SWITCH_ABS_GAP_UP", "80_000_000"))]),
        ("120M", [(replace_const, ("SWITCH_ABS_GAP_UP", "120_000_000"))]),
    ], workers=args.workers)
    results.append(("SWITCH_ABS_GAP_UP", label, edge))

    # ----------------------------------------------------------------
    # Hypothesis F: SCORE_DECAY_LO
    # Score EMA decay in calm conditions (940M = slow). Faster decay = quicker adaptation.
    # ----------------------------------------------------------------
    label, edge, _ = sweep("F: SCORE_DECAY_LO (baseline=940M)", [
        ("860M", [(replace_const, ("SCORE_DECAY_LO", "860_000_000"))]),
        ("900M", [(replace_const, ("SCORE_DECAY_LO", "900_000_000"))]),
        ("910M", [(replace_const, ("SCORE_DECAY_LO", "910_000_000"))]),
        ("960M", [(replace_const, ("SCORE_DECAY_LO", "960_000_000"))]),
    ], workers=args.workers)
    results.append(("SCORE_DECAY_LO", label, edge))

    # ----------------------------------------------------------------
    # Hypothesis G: BIAS_ADAPT_1 (P1 volatile-regime release)
    # Current 60M. More boost = P1 wins more in volatile periods vs P2.
    # ----------------------------------------------------------------
    label, edge, _ = sweep("G: BIAS_ADAPT_1 (baseline=60M)", [
        ("20M",  [(replace_const, ("BIAS_ADAPT_1", "20_000_000"))]),
        ("40M",  [(replace_const, ("BIAS_ADAPT_1", "40_000_000"))]),
        ("90M",  [(replace_const, ("BIAS_ADAPT_1", "90_000_000"))]),
        ("130M", [(replace_const, ("BIAS_ADAPT_1", "130_000_000"))]),
    ], workers=args.workers)
    results.append(("BIAS_ADAPT_1", label, edge))

    # ----------------------------------------------------------------
    # Hypothesis H: BIAS_SHOCK_THRESH
    # Threshold where adaptive biases fully activate. Lower = more aggressive.
    # ----------------------------------------------------------------
    label, edge, _ = sweep("H: BIAS_SHOCK_THRESH (baseline=5M=0.5%)", [
        ("2M",  [(replace_const, ("BIAS_SHOCK_THRESH", "2_000_000"))]),
        ("3M",  [(replace_const, ("BIAS_SHOCK_THRESH", "3_000_000"))]),
        ("8M",  [(replace_const, ("BIAS_SHOCK_THRESH", "8_000_000"))]),
        ("12M", [(replace_const, ("BIAS_SHOCK_THRESH", "12_000_000"))]),
    ], workers=args.workers)
    results.append(("BIAS_SHOCK_THRESH", label, edge))

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print(f"\n{'='*60}")
    print("SUMMARY (baseline=501.62)")
    print(f"{'='*60}")
    any_improvement = False
    for param, best_label, best_edge in results:
        delta = best_edge - BASELINE
        flag = " ***" if delta > 1.0 else ""
        print(f"  {param}: best={best_label} -> {best_edge:.2f} ({delta:+.2f}){flag}")
        if delta > 1.0:
            any_improvement = True

    if any_improvement:
        print("\n*** Improvements found! Consider combining best values or running Optuna. ***")
    else:
        print("\nNo significant improvements found from single-parameter changes.")

if __name__ == "__main__":
    main()
