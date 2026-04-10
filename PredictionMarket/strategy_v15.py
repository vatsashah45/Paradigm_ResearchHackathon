from __future__ import annotations

from orderbook_pm_challenge.strategy import BaseStrategy
from orderbook_pm_challenge.types import CancelAll, PlaceOrder, Side, StepState


class Strategy(BaseStrategy):
    """v15: Tuned variant. base_size=8, no cooldown, simpler adj logic."""

    def __init__(self) -> None:
        self._fair_value: float = 50.0
        self._drift_ema: float = 0.0
        self._vol_ema: float = 0.0
        self._min_observed_spread: int = 99
        self._observations: int = 0
        self._is_wide: bool = False

    def on_step(self, state: StepState):
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

        buy_filled = state.buy_filled_quantity
        sell_filled = state.sell_filled_quantity
        alpha = 0.06

        if buy_filled > 0 or sell_filled > 0:
            net = sell_filled - buy_filled
            mag = buy_filled + sell_filled
            self._drift_ema = (1 - alpha) * self._drift_ema + alpha * net
            self._vol_ema = (1 - alpha) * self._vol_ema + alpha * mag
        else:
            self._drift_ema *= (1 - alpha * 0.3)
            self._vol_ema *= (1 - alpha * 0.2)

        self._fair_value = comp_mid + self._drift_ema * 1.5
        self._fair_value = max(3.0, min(97.0, self._fair_value))

        net_inv = state.yes_inventory - state.no_inventory
        abs_inv = abs(net_inv)
        fair = self._fair_value
        fair_int = int(round(fair))

        actions = [CancelAll()]

        if not self._is_wide:
            if steps_remaining < 6 and abs_inv > 0.5:
                self._flatten(actions, net_inv, abs_inv, fair_int, state)
            return actions

        max_inv = 50.0
        inv_scale = max(0.1, 1.0 - abs_inv / (max_inv * 1.3))
        time_scale = 1.0
        if steps_remaining < 80:
            time_scale = max(0.1, steps_remaining / 80.0)
        skew = net_inv * 0.1

        inv_adj = min(2, int(abs_inv / 20.0))
        time_adj = 0
        if steps_remaining < 200:
            time_adj = 1
        if steps_remaining < 50:
            time_adj = 3

        bid_hs = 2 + inv_adj + time_adj
        ask_hs = 2 + inv_adj + time_adj

        bid_hs = max(2, min(10, bid_hs))
        ask_hs = max(2, min(10, ask_hs))

        base_size = 8.0 * inv_scale * time_scale

        bid_size = max(0.5, round(base_size, 2))
        ask_size = max(0.5, round(base_size, 2))

        bid_tick = max(1, min(98, int(round(fair - bid_hs - skew))))
        ask_tick = max(2, min(99, int(round(fair + ask_hs - skew))))
        if bid_tick >= ask_tick:
            bid_tick = max(1, ask_tick - 1)

        if steps_remaining > 8 and net_inv < max_inv:
            cost = bid_tick / 100.0 * bid_size
            if state.free_cash > cost + 0.3:
                actions.append(PlaceOrder(side=Side.BUY, price_ticks=bid_tick, quantity=bid_size))

        if steps_remaining > 8 and net_inv > -max_inv:
            avail = max(0.0, state.yes_inventory)
            cov = min(ask_size, avail)
            unc = max(0.0, ask_size - cov)
            cost = (1.0 - ask_tick / 100.0) * unc
            if state.free_cash > cost + 0.3 or cov >= ask_size:
                actions.append(PlaceOrder(side=Side.SELL, price_ticks=ask_tick, quantity=ask_size))

        # Second level
        if steps_remaining > 20:
            l2_bid = max(1, bid_tick - 3)
            l2_ask = min(99, ask_tick + 3)
            l2_size = max(0.5, round(base_size * 0.5, 2))

            if net_inv < max_inv:
                cost = l2_bid / 100.0 * l2_size
                if state.free_cash > cost + bid_tick / 100.0 * bid_size + 0.3:
                    actions.append(PlaceOrder(side=Side.BUY, price_ticks=l2_bid, quantity=l2_size))

            if net_inv > -max_inv:
                avail = max(0.0, state.yes_inventory - ask_size)
                cov = min(l2_size, max(0, avail))
                unc = max(0.0, l2_size - cov)
                cost = (1.0 - l2_ask / 100.0) * unc
                if state.free_cash > cost + 0.3 or cov >= l2_size:
                    actions.append(PlaceOrder(side=Side.SELL, price_ticks=l2_ask, quantity=l2_size))

        # Third level
        if steps_remaining > 50:
            l3_bid = max(1, bid_tick - 6)
            l3_ask = min(99, ask_tick + 6)
            l3_size = max(0.3, round(base_size * 0.3, 2))

            if net_inv < max_inv:
                cost = l3_bid / 100.0 * l3_size
                if state.free_cash > cost + 0.5:
                    actions.append(PlaceOrder(side=Side.BUY, price_ticks=l3_bid, quantity=l3_size))

            if net_inv > -max_inv:
                avail = max(0.0, state.yes_inventory - ask_size - (l2_size if steps_remaining > 20 else 0))
                cov = min(l3_size, max(0, avail))
                unc = max(0.0, l3_size - cov)
                cost = (1.0 - l3_ask / 100.0) * unc
                if state.free_cash > cost + 0.5 or cov >= l3_size:
                    actions.append(PlaceOrder(side=Side.SELL, price_ticks=l3_ask, quantity=l3_size))

        if steps_remaining < 6 and abs_inv > 0.5:
            self._flatten(actions, net_inv, abs_inv, fair_int, state)

        return actions

    def _flatten(self, actions, net_inv, abs_inv, fair_int, state):
        flat = max(0.5, min(abs_inv * 0.5, 3))
        flat = round(flat, 2)
        if net_inv > 0:
            st = max(1, fair_int)
            if state.yes_inventory >= flat or state.free_cash > 0.3:
                actions.append(PlaceOrder(side=Side.SELL, price_ticks=st, quantity=flat))
        else:
            bt = min(99, fair_int)
            if state.free_cash > bt / 100.0 * flat:
                actions.append(PlaceOrder(side=Side.BUY, price_ticks=bt, quantity=flat))
