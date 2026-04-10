#!/usr/bin/env python3
"""Grid search over base_size parameter."""
import sys, os, tempfile
sys.path.insert(0, '/Users/vatsashah/prediction-market-challenge')
from orderbook_pm_challenge.runner import run_batch
from orderbook_pm_challenge.loader import load_strategy_factory

TEMPLATE = '''
from __future__ import annotations
from orderbook_pm_challenge.strategy import BaseStrategy
from orderbook_pm_challenge.types import CancelAll, PlaceOrder, Side, StepState

class Strategy(BaseStrategy):
    def __init__(self):
        self._fair_value = 50.0
        self._min_observed_spread = 99
        self._observations = 0
        self._is_wide = False
    def on_step(self, state):
        step = state.step
        steps_remaining = state.steps_remaining
        comp_bid = state.competitor_best_bid_ticks
        comp_ask = state.competitor_best_ask_ticks
        if comp_bid is not None and comp_ask is not None:
            comp_mid = (comp_bid + comp_ask) / 2.0
            comp_spread = comp_ask - comp_bid
            self._observations += 1
            if comp_spread < self._min_observed_spread:
                self._min_observed_spread = comp_spread
            if self._observations >= 3 and not self._is_wide:
                if self._min_observed_spread >= 7:
                    self._is_wide = True
        elif comp_bid is not None:
            comp_mid = comp_bid + 2.0
        elif comp_ask is not None:
            comp_mid = comp_ask - 2.0
        else:
            comp_mid = 50.0
        self._fair_value = max(3.0, min(97.0, comp_mid))
        net_inv = state.yes_inventory - state.no_inventory
        abs_inv = abs(net_inv)
        fair = self._fair_value
        fair_int = int(round(fair))
        actions = [CancelAll()]
        if not self._is_wide:
            return actions
        max_inv = {max_inv}
        inv_scale = max(0.05, 1.0 - abs_inv / (max_inv * 1.2))
        time_scale = 1.0
        if steps_remaining < 100:
            time_scale = max(0.1, steps_remaining / 100.0)
        skew = net_inv * 0.1
        inv_adj = min(2, int(abs_inv / 15.0))
        time_adj = 0
        if steps_remaining < 200:
            time_adj = 1
        if steps_remaining < 50:
            time_adj = 3
        hs = 2 + inv_adj + time_adj
        hs = max(2, min(10, hs))
        base_size = {base_size} * inv_scale * time_scale
        size = max(0.5, round(base_size, 2))
        bid_tick = max(1, min(98, int(round(fair - hs - skew))))
        ask_tick = max(2, min(99, int(round(fair + hs - skew))))
        if bid_tick >= ask_tick:
            bid_tick = max(1, ask_tick - 1)
        if steps_remaining > 8 and net_inv < max_inv:
            cost = bid_tick / 100.0 * size
            if state.free_cash > cost + 0.3:
                actions.append(PlaceOrder(side=Side.BUY, price_ticks=bid_tick, quantity=size))
        if steps_remaining > 8 and net_inv > -max_inv:
            avail = max(0.0, state.yes_inventory)
            cov = min(size, avail)
            unc = max(0.0, size - cov)
            cost = (1.0 - ask_tick / 100.0) * unc
            if state.free_cash > cost + 0.3 or cov >= size:
                actions.append(PlaceOrder(side=Side.SELL, price_ticks=ask_tick, quantity=size))
        return actions
'''

for base_size in [5, 6, 7, 8, 9, 10, 11, 12]:
    max_inv = base_size * 6
    code = TEMPLATE.format(base_size=base_size, max_inv=max_inv)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, dir='/tmp') as f:
        f.write(code)
        tmppath = f.name
    try:
        factory = load_strategy_factory(tmppath)
        batch = run_batch(
            factory,
            n_simulations=200,
        )
        print(f"size={base_size:2d} max_inv={max_inv:3d} | mean_edge={batch.mean_edge:+.4f} retail={batch.mean_retail_edge:+.4f} arb={batch.mean_arb_edge:+.4f}")
        sys.stdout.flush()
    finally:
        os.unlink(tmppath)
