from __future__ import annotations

from orderbook_pm_challenge.strategy import BaseStrategy
from orderbook_pm_challenge.types import CancelAll, PlaceOrder, Side, StepState


class Strategy(BaseStrategy):
    """
    comp_mid ± 2 strategy, active in spread_ticks 3-4 (comp_spread >= 5).
    Doubles the addressable sims vs spread_ticks=4 only.
    """

    def __init__(self) -> None:
        self._min_observed_spread: int = 99
        self._observations: int = 0
        self._regime_ok: bool = False

    def on_step(self, state: StepState):
        steps_remaining = state.steps_remaining
        comp_bid = state.competitor_best_bid_ticks
        comp_ask = state.competitor_best_ask_ticks

        actions = [CancelAll()]

        if comp_bid is None or comp_ask is None:
            return actions

        comp_spread = comp_ask - comp_bid
        comp_mid = (comp_bid + comp_ask) / 2.0
        self._observations += 1
        if comp_spread < self._min_observed_spread:
            self._min_observed_spread = comp_spread

        if self._observations >= 3 and not self._regime_ok:
            if self._min_observed_spread >= 5:
                self._regime_ok = True

        if not self._regime_ok:
            return actions

        fair = max(3.0, min(97.0, comp_mid))
        net_inv = state.yes_inventory - state.no_inventory
        abs_inv = abs(net_inv)
        max_inv = 50.0

        inv_scale = max(0.05, 1.0 - abs_inv / (max_inv * 1.2))
        time_scale = 1.0
        if steps_remaining < 100:
            time_scale = max(0.1, steps_remaining / 100.0)

        size = max(0.5, round(8.0 * inv_scale * time_scale, 2))
        skew = net_inv * 0.1

        hs = 2
        inv_adj = min(2, int(abs_inv / 15.0))
        time_adj = 0
        if steps_remaining < 200:
            time_adj = 1
        if steps_remaining < 50:
            time_adj = 3
        hs = hs + inv_adj + time_adj
        hs = max(2, min(10, hs))

        bid_tick = max(1, min(98, int(round(fair - hs - skew))))
        ask_tick = max(2, min(99, int(round(fair + hs - skew))))
        if bid_tick >= ask_tick:
            bid_tick = max(1, ask_tick - 1)

        if steps_remaining > 5 and net_inv < max_inv:
            cost = bid_tick / 100.0 * size
            if state.free_cash > cost + 0.3:
                actions.append(PlaceOrder(side=Side.BUY, price_ticks=bid_tick, quantity=size))

        if steps_remaining > 5 and net_inv > -max_inv:
            avail = max(0.0, state.yes_inventory)
            cov = min(size, avail)
            unc = max(0.0, size - cov)
            cost = (1.0 - ask_tick / 100.0) * unc
            if state.free_cash > cost + 0.3 or cov >= size:
                actions.append(PlaceOrder(side=Side.SELL, price_ticks=ask_tick, quantity=size))

        if steps_remaining < 6 and abs_inv > 0.5:
            fair_int = max(1, min(99, int(round(fair))))
            flat = max(0.5, min(abs_inv * 0.5, 5))
            flat = round(flat, 2)
            if net_inv > 0:
                if state.yes_inventory >= flat or state.free_cash > 0.3:
                    actions.append(PlaceOrder(side=Side.SELL, price_ticks=fair_int, quantity=flat))
            else:
                if state.free_cash > fair_int / 100.0 * flat:
                    actions.append(PlaceOrder(side=Side.BUY, price_ticks=fair_int, quantity=flat))

        return actions
