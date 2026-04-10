#!/usr/bin/env python3
"""
analyze_frontrun.py

Runs ARB model and NORMALIZER model in parallel on the same trade data.
Identifies front-running opportunities and quantifies expected edge improvement.

Usage:
  python3 python/analyze_frontrun.py [--gen-data] [--workers 10]

With --gen-data: regenerates CSV data (requires ~60s).
Without: uses existing csvs/ directory.
"""

import argparse
import os
import re
import subprocess
import sys
import time

import pandas as pd
import numpy as np

# Add python/ dir to path for model imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arb_model import ArbModel
from normalizer_model import NormalizerModel

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BINARY = os.path.join(PROJECT_ROOT, 'target/release/prop-amm')
STRATEGY = os.path.join(PROJECT_ROOT, 'strategy.rs')
CSV_DIR = os.path.join(PROJECT_ROOT, 'csvs')
CARGO_BIN = os.path.expanduser('~/.cargo/bin')


def generate_data(n_sims=500, workers=10):
    """Generate per-trade CSV data and per-sim summary."""
    os.makedirs(CSV_DIR, exist_ok=True)
    env = os.environ.copy()
    env['SWAP_CSV'] = '1'
    env['CSV_DUMP'] = os.path.join(CSV_DIR, 'summary.csv')
    env['RAYON_NUM_THREADS'] = str(workers)
    env['PATH'] = CARGO_BIN + ':' + env.get('PATH', '')
    cmd = [BINARY, 'run', STRATEGY, '--simulations', str(n_sims),
           '--seed-start', '0', '--seed-stride', '1', '--workers', str(workers)]
    print(f"Generating {n_sims} sims of CSV data...")
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    print(f"Done in {time.time()-t0:.0f}s")
    if r.returncode != 0:
        print(r.stderr[:500])


def run_all_models(summary_df: pd.DataFrame, swap_dir: str, verbose=False):
    """
    Run ARB and NORMALIZER models in parallel (per sim) across all CSV files.
    Returns merged per-sim analysis dataframe.
    """
    records = []
    available_seeds = {
        int(f.split('_')[2].split('.')[0])
        for f in os.listdir(swap_dir)
        if f.startswith('swap_log_') and f.endswith('.csv')
    }

    for _, sim_row in summary_df.iterrows():
        seed = int(sim_row['seed'])
        if seed not in available_seeds:
            continue

        norm_fee = sim_row['nfee'] / 10000.0
        arb_m = ArbModel(norm_fee, seed=seed)
        norm_m = NormalizerModel(norm_fee, seed=seed)

        # Load per-trade data
        swap_file = os.path.join(swap_dir, f'swap_log_{seed}.csv')
        trades = pd.read_csv(swap_file)

        # Run both models in parallel (same loop = same trades)
        for _, row in trades.iterrows():
            rd = row.to_dict()
            arb_m.process_trade(rd)
            norm_m.process_trade(rd)

        arb_s = arb_m.summary()
        norm_s = norm_m.summary()

        rec = {
            'seed': seed,
            'norm_fee_bps': sim_row['nfee'],
            'norm_fee': norm_fee,
            'sigma': sim_row['sigma'],
            'sim_edge': sim_row['edge'],
            'arb_edge': sim_row['arb_edge'],
            'retail_edge': sim_row['retail_edge'],
            'arb_cnt': sim_row['arb_cnt'],
            'retail_cnt': sim_row['retail_cnt'],
            'n_switches': sim_row['n_switches'],
            # ARB model outputs
            'arb_n_trades': arb_s.get('n_arb_trades', 0),
            'arb_prefers_us_pct': arb_s.get('arb_prefers_us_pct', float('nan')),
            'arb_fee_gap': arb_s.get('mean_fee_gap_when_prefer_us', float('nan')),
            'arb_total_loss': arb_s.get('total_arb_edge_loss', 0),
            # NORMALIZER model outputs
            'retail_n': norm_s.get('n_retail', 0),
            'retail_total_edge': norm_s.get('retail_edge_total', 0),
            'mean_stored_fee': norm_s.get('mean_stored_fee', float('nan')),
            'mean_fee_gap_retail': norm_s.get('mean_fee_gap_retail', float('nan')),
            'missed_revenue': norm_s.get('missed_revenue_total', 0),
            'pct_underpriced': norm_s.get('pct_retail_underpriced', float('nan')),
            'mean_cnt_ema': norm_s.get('mean_cnt_ema_retail', float('nan')),
            'pct_arb_prefers_us': norm_s.get('pct_arb_prefers_us', float('nan')),
        }
        records.append(rec)

    return pd.DataFrame(records)


def print_section(title):
    print(f"\n{'='*65}")
    print(f" {title}")
    print(f"{'='*65}")


def analyze_results(df: pd.DataFrame):
    """Generate actionable insights from combined model analysis."""

    print_section("DATASET OVERVIEW")
    print(f"  Simulations analyzed: {len(df)}")
    print(f"  norm_fee range: {df['norm_fee_bps'].min()}-{df['norm_fee_bps'].max()} BPS")
    print(f"  norm_fee distribution: {df['norm_fee_bps'].value_counts().sort_index().to_dict()}")
    print(f"  Baseline avg edge: {df['sim_edge'].mean():.2f}")
    print(f"  Mean arb edge/sim: {df['arb_edge'].mean():.2f}")
    print(f"  Mean retail edge/sim: {df['retail_edge'].mean():.2f}")

    # ----------------------------------------------------------------
    print_section("ARB MODEL: Does norm_fee predict arb behavior?")
    corr_arb_nf = df['norm_fee'].corr(df['arb_total_loss'])
    corr_arb_pct = df['norm_fee'].corr(df['arb_prefers_us_pct'])
    corr_arb_n = df['norm_fee'].corr(df['arb_n_trades'])
    print(f"  norm_fee vs arb_total_loss:      r={corr_arb_nf:.3f}")
    print(f"  norm_fee vs arb_prefers_us_pct:  r={corr_arb_pct:.3f}")
    print(f"  norm_fee vs n_arb_trades:        r={corr_arb_n:.3f}")

    # Breakdown by norm_fee quintile
    df['nf_quintile'] = pd.qcut(df['norm_fee'], 5, labels=['Q1(low)', 'Q2', 'Q3', 'Q4', 'Q5(high)'])
    print(f"\n  Arb behavior by norm_fee quintile:")
    print(f"  {'Quintile':<12} {'nfee(BPS)':<12} {'arb_pref_us%':<14} {'n_arb':<8} {'arb_loss':<10}")
    for q, g in df.groupby('nf_quintile', observed=True):
        print(f"  {str(q):<12} {g['norm_fee_bps'].mean():<12.0f} "
              f"{g['arb_prefers_us_pct'].mean():<14.1%} "
              f"{g['arb_n_trades'].mean():<8.0f} "
              f"{g['arb_total_loss'].mean():<10.1f}")

    # ----------------------------------------------------------------
    print_section("NORMALIZER MODEL: missed revenue analysis")
    print(f"  Mean stored_fee: {df['mean_stored_fee'].mean()*10000:.1f} BPS")
    print(f"  Mean norm_fee:   {df['norm_fee'].mean()*10000:.1f} BPS")
    print(f"  Mean fee gap (norm - stored): {df['mean_fee_gap_retail'].mean()*10000:.1f} BPS")
    print(f"  Pct trades underpriced vs norm: {df['pct_underpriced'].mean():.1%}")
    total_missed = df['missed_revenue'].sum()
    per_sim_missed = df['missed_revenue'].mean()
    print(f"  Total missed revenue (upper bound): {total_missed:.1f}")
    print(f"  Avg missed revenue/sim: {per_sim_missed:.2f}")
    print(f"  (If we could charge norm_fee for retail, edge would improve by ~{per_sim_missed:.2f}/sim)")

    print(f"\n  Missed revenue by norm_fee quintile:")
    print(f"  {'Quintile':<12} {'nfee(BPS)':<12} {'missed/sim':<12} {'n_retail':<10} {'sim_edge':<10}")
    for q, g in df.groupby('nf_quintile', observed=True):
        print(f"  {str(q):<12} {g['norm_fee_bps'].mean():<12.0f} "
              f"{g['missed_revenue'].mean():<12.1f} "
              f"{g['retail_n'].mean():<10.0f} "
              f"{g['sim_edge'].mean():<10.1f}")

    # ----------------------------------------------------------------
    print_section("CNT_EMA as norm_fee PROXY (key signal)")
    corr_cnt_nf = df['norm_fee'].corr(df['mean_cnt_ema'])
    corr_cnt_ret = df['norm_fee'].corr(df['retail_n'])
    corr_cnt_edge = df['norm_fee'].corr(df['retail_edge'])
    print(f"  norm_fee vs mean_cnt_ema: r={corr_cnt_nf:.3f}  "
          f"{'STRONG SIGNAL!' if abs(corr_cnt_nf) > 0.3 else 'weak signal'}")
    print(f"  norm_fee vs retail_count: r={corr_cnt_ret:.3f}")
    print(f"  norm_fee vs retail_edge:  r={corr_cnt_edge:.3f}")

    print(f"\n  cnt_ema vs norm_fee by quintile:")
    print(f"  {'Quintile':<12} {'nfee(BPS)':<12} {'cnt_ema':<10} {'sim_edge':<10}")
    for q, g in df.groupby('nf_quintile', observed=True):
        print(f"  {str(q):<12} {g['norm_fee_bps'].mean():<12.0f} "
              f"{g['mean_cnt_ema'].mean():<10.2f} "
              f"{g['sim_edge'].mean():<10.1f}")

    # ----------------------------------------------------------------
    print_section("SIGMA as norm_fee PROXY")
    corr_sig_nf = df['sigma'].corr(df['norm_fee'])
    print(f"  sigma vs norm_fee: r={corr_sig_nf:.3f}  "
          f"({'note: sigma predicts norm_fee' if abs(corr_sig_nf) > 0.15 else 'independent'})")
    # sigma vs edge
    corr_sig_edge = df['sigma'].corr(df['sim_edge'])
    print(f"  sigma vs sim_edge: r={corr_sig_edge:.3f}")

    # ----------------------------------------------------------------
    print_section("FRONT-RUNNING OPPORTUNITY QUANTIFICATION")

    # How much can we realistically gain?
    # Key: when norm_fee is high (>55 BPS), our stored_fee is way below it.
    # If we could charge norm_fee - 5 BPS for retail → how much extra edge?
    high_nf = df[df['norm_fee_bps'] > 55]
    low_nf = df[df['norm_fee_bps'] <= 55]

    print(f"\n  High norm_fee sims (>55 BPS): {len(high_nf)} sims")
    if len(high_nf) > 0:
        print(f"    Mean sim_edge:  {high_nf['sim_edge'].mean():.2f}")
        print(f"    Mean missed:    {high_nf['missed_revenue'].mean():.2f}/sim")
        print(f"    Mean cnt_ema:   {high_nf['mean_cnt_ema'].mean():.2f}")
        print(f"    Mean sigma:     {high_nf['sigma'].mean()*100:.3f}%")

    print(f"\n  Low norm_fee sims (<=55 BPS): {len(low_nf)} sims")
    if len(low_nf) > 0:
        print(f"    Mean sim_edge:  {low_nf['sim_edge'].mean():.2f}")
        print(f"    Mean missed:    {low_nf['missed_revenue'].mean():.2f}/sim")
        print(f"    Mean cnt_ema:   {low_nf['mean_cnt_ema'].mean():.2f}")
        print(f"    Mean sigma:     {low_nf['sigma'].mean()*100:.3f}%")

    # ----------------------------------------------------------------
    print_section("ACTIONABLE HYPOTHESES")

    # H1: Raise BASE_FEE when cnt_ema is high (high norm_fee likely)
    if abs(corr_cnt_nf) > 0.15:
        print(f"\n  H1: cnt_ema correlates with norm_fee (r={corr_cnt_nf:.2f})")
        print(f"      ACTIONABLE: add cnt_ema-based fee bump in quote_profile")
        print(f"      Estimated gain: ~{per_sim_missed*0.3:.1f}/sim (30% capture)")
        print(f"      (requires new Rust constant CNT_EMA_FEE_K)")
    else:
        print(f"\n  H1: cnt_ema does NOT predict norm_fee (r={corr_cnt_nf:.2f})")
        print(f"      cnt_ema is NOT useful as norm_fee proxy")

    # H2: TOX_READ_K (currently 0) - could tox in quote_profile improve arb detection?
    corr_tox_arb = df['arb_total_loss'].corr(df['sigma'])
    print(f"\n  H2: TOX_READ_K=0 (disabled) - tox in quote_profile unused")
    print(f"      sigma vs arb_loss corr: {corr_tox_arb:.2f}")
    print(f"      If tox predicts arbs: enabling TOX_READ_K might help")

    # H3: ROUTE_CORR constants (6 missing constants expected by convergence loop)
    print(f"\n  H3: ROUTE_CORR_P*_SHOCK / ROUTE_CORR_P*_TOX constants (missing from strategy.rs)")
    print(f"      These 6 i64 constants are expected by run_convergence_loop.py")
    print(f"      Adding them + running convergence loop might reveal new signal")

    return df


def propose_coarse_sweep(df: pd.DataFrame):
    """Propose specific coarse sweep tests based on model findings."""
    print_section("PROPOSED COARSE SWEEP TESTS")

    corr_cnt_nf = df['norm_fee'].corr(df['mean_cnt_ema'])
    if abs(corr_cnt_nf) > 0.1:
        print(f"\n  Test A: Can ROUTE_BONUS_W encode cnt_ema signal?")
        print(f"    ROUTE_BONUS_W currently: 145M (adds bonus to counter-dir score)")
        print(f"    Try: higher ROUTE_BONUS_W when cnt_ema is high")
        print(f"    (already swept individually; try in combo with other signals)")

    print(f"\n  Test B: TOX_READ_K (currently 0) - enable tox in quote_profile fee")
    print(f"    Try values: 50*BPS, 100*BPS, 200*BPS, 500*BPS")
    print(f"    This adds tox signal to real-time fee quote (not just stored_fee)")

    print(f"\n  Test C: Add ROUTE_CORR constants to strategy.rs (all=0)")
    print(f"    Then run: python3 python/run_convergence_loop.py --batches 3 --batch-size 200")
    print(f"    This ML approach adapts to routing patterns per profile/regime")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen-data', action='store_true', help='Regenerate CSV data')
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--sims', type=int, default=500)
    args = parser.parse_args()

    summary_path = os.path.join(CSV_DIR, 'summary.csv')

    if args.gen_data or not os.path.exists(summary_path):
        generate_data(n_sims=args.sims, workers=args.workers)

    summary_df = pd.read_csv(summary_path)
    print(f"Loaded {len(summary_df)} sims from {summary_path}")

    print("\nRunning ARB + NORMALIZER models in parallel on all trade data...")
    t0 = time.time()
    combined_df = run_all_models(summary_df, CSV_DIR)
    print(f"Models complete in {time.time()-t0:.1f}s ({len(combined_df)} sims)")

    combined_df = analyze_results(combined_df)
    propose_coarse_sweep(combined_df)

    # Save analysis
    out_path = os.path.join(CSV_DIR, 'frontrun_analysis.csv')
    combined_df.to_csv(out_path, index=False)
    print(f"\nAnalysis saved to {out_path}")


if __name__ == '__main__':
    main()
