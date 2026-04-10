#!/usr/bin/env python3
"""
ARB Model: tracks the arbitrage agent's decision-making from CSV trade data.

The arb agent trades when:
  price_deviation > fee_cost (simplified: deviation > our fee OR normalizer fee)

The arb ROUTES TO US when our arb-dir fee < normalizer fee.
The arb ROUTES TO NORMALIZER when normalizer fee < our arb-dir fee.

Key insight: when norm_fee is HIGH, arbs prefer us (our fee < their fee).
             when norm_fee is LOW, arbs prefer normalizer (cheaper).

Front-running the ARB means: detect when arb is about to trade, charge MORE
(up to norm_fee - epsilon) to extract maximum value while keeping arb routing to us.
"""

import pandas as pd
import numpy as np


class ArbModel:
    """Per-simulation arb agent state tracker."""

    def __init__(self, norm_fee_frac: float, seed: int = 0):
        """
        norm_fee_frac: normalizer fee as fraction (e.g., 0.0065 = 65 BPS)
        """
        self.norm_fee = norm_fee_frac
        self.seed = seed
        self.trades = []  # list of dicts for each arb trade

    def process_trade(self, row: dict):
        """Process a single CSV row. Classifies as arb if edge < 0."""
        edge = row['edge']
        stored_fee = row['stored_fee']
        deviation = row['deviation']
        cnt_ema = row['cnt_ema']
        sigma = row['sigma']
        shock_hat = row['shock_hat']
        tox_hat = row['tox_hat']

        is_arb = edge < -0.001

        if is_arb:
            # Arb direction: we lose edge
            # Arb routing preference: arb prefers us when stored_fee < norm_fee
            arb_prefers_us = stored_fee < self.norm_fee
            # If arb prefers us AND deviation > stored_fee: arb is profitable on us
            arb_profitable_on_us = deviation > stored_fee

            # How much EXTRA could we have charged before arb routes to normalizer?
            # Answer: up to norm_fee (arb indifferent at norm_fee, prefers normalizer above it)
            fee_gap = max(0.0, self.norm_fee - stored_fee)

            self.trades.append({
                'edge': edge,
                'stored_fee': stored_fee,
                'deviation': deviation,
                'cnt_ema': cnt_ema,
                'sigma': sigma,
                'shock_hat': shock_hat,
                'tox_hat': tox_hat,
                'arb_prefers_us': arb_prefers_us,
                'arb_profitable_on_us': arb_profitable_on_us,
                'fee_gap': fee_gap,
                'potential_extra_edge': fee_gap * abs(edge) / max(stored_fee, 1e-6) if stored_fee > 0 else 0,
            })

    def summary(self) -> dict:
        """Summarize arb agent behavior for this simulation."""
        if not self.trades:
            return {'n_arb_trades': 0}

        df = pd.DataFrame(self.trades)
        return {
            'n_arb_trades': len(df),
            'arb_prefers_us_pct': df['arb_prefers_us'].mean(),
            'mean_fee_gap_when_prefer_us': df.loc[df['arb_prefers_us'], 'fee_gap'].mean() if df['arb_prefers_us'].any() else 0,
            'total_arb_edge_loss': df['edge'].sum(),
            'mean_deviation': df['deviation'].mean(),
            'mean_stored_fee': df['stored_fee'].mean(),
            'norm_fee': self.norm_fee,
        }


def analyze_arb_behavior(summary_df: pd.DataFrame, swap_dir: str) -> pd.DataFrame:
    """
    Run arb model across all simulations. Returns per-sim arb analysis.

    summary_df: loaded from csvs/summary.csv
    swap_dir: directory containing swap_log_*.csv files
    """
    import os

    records = []
    for _, sim_row in summary_df.iterrows():
        seed = int(sim_row['seed'])
        norm_fee = sim_row['nfee'] / 10000.0  # convert BPS to fraction

        swap_file = os.path.join(swap_dir, f'swap_log_{seed}.csv')
        if not os.path.exists(swap_file):
            continue

        model = ArbModel(norm_fee, seed=seed)
        trades = pd.read_csv(swap_file)

        for _, row in trades.iterrows():
            model.process_trade(row.to_dict())

        s = model.summary()
        s['seed'] = seed
        s['sim_edge'] = sim_row['edge']
        s['sim_sigma'] = sim_row['sigma']
        s['sim_nliq'] = sim_row['nliq']
        s['sim_arb_edge'] = sim_row['arb_edge']
        s['sim_retail_edge'] = sim_row['retail_edge']
        records.append(s)

    return pd.DataFrame(records)


if __name__ == '__main__':
    import sys
    import os

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_dir = os.path.join(project_root, 'csvs')
    summary_path = os.path.join(csv_dir, 'summary.csv')

    summary_df = pd.read_csv(summary_path)
    print(f"Loaded summary: {len(summary_df)} sims, norm_fee range {summary_df['nfee'].min()}-{summary_df['nfee'].max()} BPS")

    arb_df = analyze_arb_behavior(summary_df, csv_dir)

    print(f"\n--- ARB MODEL SUMMARY ---")
    print(f"Mean arb trades/sim: {arb_df['n_arb_trades'].mean():.0f}")
    print(f"Mean arb prefers us: {arb_df['arb_prefers_us_pct'].mean():.1%}")
    print(f"Mean fee gap when arb prefers us: {arb_df['mean_fee_gap_when_prefer_us'].mean()*10000:.1f} BPS")
    print(f"Total arb edge loss (all sims): {arb_df['total_arb_edge_loss'].sum():.0f}")

    # Correlation: does norm_fee predict arb preference?
    corr = arb_df[['norm_fee', 'arb_prefers_us_pct', 'n_arb_trades', 'total_arb_edge_loss']].corr()
    print(f"\nCorrelations with norm_fee:")
    print(f"  arb_prefers_us_pct: {corr.loc['norm_fee','arb_prefers_us_pct']:.3f}")
    print(f"  n_arb_trades: {corr.loc['norm_fee','n_arb_trades']:.3f}")
    print(f"  total_arb_edge_loss: {corr.loc['norm_fee','total_arb_edge_loss']:.3f}")
