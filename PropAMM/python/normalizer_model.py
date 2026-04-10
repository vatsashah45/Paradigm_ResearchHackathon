#!/usr/bin/env python3
"""
Normalizer Model: tracks the normalizer AMM's competitive position.

The normalizer is a constant-product AMM with:
  - Fee = norm_fee BPS (fixed per simulation, drawn from U{30, 80})
  - Stateless (no storage, no after_swap)

Key competitive dynamics:
  - Retail (counter-dir) routes to us when our fee < norm_fee
  - Arbs route to us when our arb fee < norm_fee

Front-running the NORMALIZER means: estimate norm_fee from observable signals
(cnt_ema, sigma, retail arrival rate), then price our counter-dir fee just below
norm_fee to maximize revenue while still capturing retail flow.

The key insight: we currently charge ~20-28 BPS counter-dir (always < min norm_fee=30)
which guarantees retail capture but leaves 0-50 BPS of uncaptured revenue when
norm_fee is high (50-80 BPS).
"""

import pandas as pd
import numpy as np
import os


class NormalizerModel:
    """Per-simulation normalizer competitive tracker."""

    def __init__(self, norm_fee_frac: float, seed: int = 0):
        self.norm_fee = norm_fee_frac
        self.seed = seed
        self.retail_trades = []  # counter-dir trades (edge > 0)
        self.arb_trades = []     # arb-dir trades (edge < 0)

    def process_trade(self, row: dict):
        edge = row['edge']
        stored_fee = row['stored_fee']
        cnt_ema = row['cnt_ema']
        sigma = row['sigma']
        deviation = row['deviation']
        shock_hat = row['shock_hat']
        tox_hat = row['tox_hat']

        if edge > 0.001:  # retail / counter-dir
            # We captured this retail trade (it came to us → our fee < norm_fee at routing time)
            # Revenue we left on table: if we'd charged norm_fee instead of stored_fee
            # (This is an UPPER BOUND - actual counter-dir fee = stored_fee - rebate)
            missed_revenue = max(0.0, self.norm_fee - stored_fee) * abs(edge) / max(stored_fee, 1e-9)

            self.retail_trades.append({
                'edge': edge,
                'stored_fee': stored_fee,
                'norm_fee': self.norm_fee,
                'fee_gap': self.norm_fee - stored_fee,  # > 0: we undercharge relative to norm
                'missed_revenue_approx': missed_revenue,
                'cnt_ema': cnt_ema,
                'sigma': sigma,
                'shock_hat': shock_hat,
                'tox_hat': tox_hat,
                'deviation': deviation,
            })

        elif edge < -0.001:  # arb trade
            # Arb came to us → our fee ≤ norm_fee (arb prefers us when cheaper)
            fee_gap = self.norm_fee - stored_fee
            self.arb_trades.append({
                'edge': edge,
                'stored_fee': stored_fee,
                'norm_fee': self.norm_fee,
                'fee_gap': fee_gap,
                'cnt_ema': cnt_ema,
                'sigma': sigma,
            })

    def summary(self) -> dict:
        if not self.retail_trades and not self.arb_trades:
            return {'seed': self.seed}

        s = {'seed': self.seed, 'norm_fee': self.norm_fee}

        if self.retail_trades:
            ret_df = pd.DataFrame(self.retail_trades)
            s['n_retail'] = len(ret_df)
            s['retail_edge_total'] = ret_df['edge'].sum()
            s['mean_stored_fee'] = ret_df['stored_fee'].mean()
            s['mean_fee_gap_retail'] = ret_df['fee_gap'].mean()
            s['missed_revenue_total'] = ret_df['missed_revenue_approx'].sum()
            s['pct_retail_underpriced'] = (ret_df['fee_gap'] > 0).mean()  # we charge < norm_fee
            s['mean_cnt_ema_retail'] = ret_df['cnt_ema'].mean()
        else:
            s['n_retail'] = 0
            s['retail_edge_total'] = 0.0
            s['mean_stored_fee'] = float('nan')
            s['mean_fee_gap_retail'] = float('nan')
            s['missed_revenue_total'] = 0.0
            s['pct_retail_underpriced'] = float('nan')
            s['mean_cnt_ema_retail'] = float('nan')

        if self.arb_trades:
            arb_df = pd.DataFrame(self.arb_trades)
            s['n_arb'] = len(arb_df)
            s['arb_edge_total'] = arb_df['edge'].sum()
            s['pct_arb_prefers_us'] = (arb_df['fee_gap'] > 0).mean()
        else:
            s['n_arb'] = 0
            s['arb_edge_total'] = 0.0
            s['pct_arb_prefers_us'] = float('nan')

        return s


def analyze_normalizer_competition(summary_df: pd.DataFrame, swap_dir: str) -> pd.DataFrame:
    """
    Run normalizer model across all simulations.
    Returns per-sim analysis with competitive metrics.
    """
    records = []
    for _, sim_row in summary_df.iterrows():
        seed = int(sim_row['seed'])
        norm_fee = sim_row['nfee'] / 10000.0

        swap_file = os.path.join(swap_dir, f'swap_log_{seed}.csv')
        if not os.path.exists(swap_file):
            continue

        model = NormalizerModel(norm_fee, seed=seed)
        trades = pd.read_csv(swap_file)

        for _, row in trades.iterrows():
            model.process_trade(row.to_dict())

        s = model.summary()
        s['sim_edge'] = sim_row['edge']
        s['sim_sigma'] = sim_row['sigma']
        s['sim_nliq'] = sim_row['nliq']
        records.append(s)

    return pd.DataFrame(records)


if __name__ == '__main__':
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_dir = os.path.join(project_root, 'csvs')
    summary_path = os.path.join(csv_dir, 'summary.csv')

    summary_df = pd.read_csv(summary_path)
    print(f"Loaded summary: {len(summary_df)} sims")
    print(f"norm_fee distribution: {summary_df['nfee'].describe()}")

    norm_df = analyze_normalizer_competition(summary_df, csv_dir)

    print(f"\n--- NORMALIZER COMPETITION SUMMARY ---")
    print(f"Mean retail trades/sim: {norm_df['n_retail'].mean():.0f}")
    print(f"Mean arb trades/sim: {norm_df['n_arb'].mean():.0f}")
    print(f"Mean stored_fee vs norm_fee:")
    print(f"  mean stored_fee: {norm_df['mean_stored_fee'].mean()*10000:.1f} BPS")
    print(f"  mean norm_fee: {norm_df['norm_fee'].mean()*10000:.1f} BPS")
    print(f"  mean fee gap (norm - stored): {norm_df['mean_fee_gap_retail'].mean()*10000:.1f} BPS")
    print(f"  pct trades where we undercharge vs norm: {norm_df['pct_retail_underpriced'].mean():.1%}")
    print(f"  total missed revenue (upper bound): {norm_df['missed_revenue_total'].sum():.1f}")
    print(f"  avg missed revenue/sim: {norm_df['missed_revenue_total'].mean():.1f}")

    # Correlation: does cnt_ema predict norm_fee?
    print(f"\n--- CNT_EMA vs NORM_FEE correlation ---")
    corr_matrix = norm_df[['norm_fee', 'mean_cnt_ema_retail', 'n_retail', 'retail_edge_total']].corr()
    print(f"  cnt_ema vs norm_fee: {corr_matrix.loc['norm_fee','mean_cnt_ema_retail']:.3f}")
    print(f"  n_retail vs norm_fee: {corr_matrix.loc['norm_fee','n_retail']:.3f}")
    print(f"  retail_edge vs norm_fee: {corr_matrix.loc['norm_fee','retail_edge_total']:.3f}")

    # High vs low norm_fee breakdown
    high_nf = norm_df[norm_df['norm_fee'] > 0.0055]  # > 55 BPS
    low_nf = norm_df[norm_df['norm_fee'] <= 0.0055]
    print(f"\n--- HIGH vs LOW norm_fee breakdown ---")
    print(f"High norm_fee (>55 BPS): {len(high_nf)} sims")
    print(f"  mean sim_edge: {high_nf['sim_edge'].mean():.2f}")
    print(f"  mean missed_revenue: {high_nf['missed_revenue_total'].mean():.2f}")
    print(f"  mean cnt_ema (retail): {high_nf['mean_cnt_ema_retail'].mean():.2f}")
    print(f"Low norm_fee (<=55 BPS): {len(low_nf)} sims")
    print(f"  mean sim_edge: {low_nf['sim_edge'].mean():.2f}")
    print(f"  mean missed_revenue: {low_nf['missed_revenue_total'].mean():.2f}")
    print(f"  mean cnt_ema (retail): {low_nf['mean_cnt_ema_retail'].mean():.2f}")
