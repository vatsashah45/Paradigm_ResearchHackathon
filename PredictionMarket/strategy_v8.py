from __future__ import annotations

from orderbook_pm_challenge.strategy import BaseStrategy
from orderbook_pm_challenge.types import CancelAll, PlaceOrder, Side, StepState


class Strategy(BaseStrategy):
    """
    v8: Use visible spread threshold to isolate spread_ticks=4 regimes.
    
    spread_ticks=3: visible spread ≈ 6 (gap between levels is 2 + 2*(3-1) = 6)
    spread_ticks=4: visible spread ≈ 8 (gap is 2 + 2*(4-1) = 8)
    
    Use threshold of 7 to distinguish.
    """

    def __init__(self) -> None:
        self._initial_comp_mid: float | None = None
        self._fair_value: float = 50.0
        self._drift_ema: float = 0.0
        self._vol_ema: float = 0.0
        self._buy_fill_ema: float = 0.0
        self._sell_fill_ema: float = 0.0
        self._last_buy_fill_step: int = -100
        self._last_sell_fill_step: int = -100
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
            # Once we have enough observations, decide
            if self._observations >= 5 and not self._is_wide:
                # spread_ticks=4 → visible spread ~8+
                # spread_ticks=3 → visible spread ~6
                # Use 7 as threshold
                if self._min_observed_spread >= 7:
                    self._is_wide = True
        elif comp_bid is not None:
            comp_mid = comp_bid + 2.0
        elif comp_ask is not None:
            comp_mid = comp_ask - 2.0
        else:
            comp_mid = 50.0

        if self._initial_comp_mid is None:
            self._initial_comp_mid = comp_mid
            self._fair_value = comp_mid

        buy_filled = state.buy_filled_quantity
        sell_filled = state.sell_filled_quantity
        alpha = 0.06

        if buy_filled > 0:
            self._buy_fill_ema = (1 - alpha) * self._buy_fill_ema + alpha * buy_filled
            self._last_buy_fill_step = step
        else:
            self._buy_fill_ema *= (1 - alpha * 0.3)

        if sell_filled > 0:
            self._sell_fill_ema = (1 - alpha) * self._sell_fill_ema + alpha * sell_filled
            self._last_sell_fill_step = step
        else:
            self._sell_fill_ema *= (1 - alpha * 0.3)

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

        # ─── WIDE SPREAD: AGGRESSIVE QUOTING ─────────────────────────
        max_inv = 40.0
        inv_scale = max(0.1, 1.0 - abs_inv / (max_inv * 1.3))
        time_scale = 1.0
        if steps_remaining < 80:
            time_scale = max(0.1, steps_remaining / 80.0)
        skew = net_inv * 0.1

        buy_cooldown = max(0, 4 - (step - self._last_buy_fill_step))
        sell_cooldown = max(0, 4 - (step - self._last_sell_fill_step))

        vol_adj = min(3, int(self._vol_ema * 0.35))
        inv_adj = min(3, int(abs_inv / 15.0))
        time_adj = 0
        if steps_remaining < 200:
            time_adj = 1
        if steps_remaining < 50:
            time_adj = 3

        bid_hs = 2 + vol_adj + inv_adj + time_adj + buy_cooldown
        ask_hs = 2 + vol_adj + inv_adj + time_adj + sell_cooldown

        if self._buy_fill_ema > self._sell_fill_ema + 0.04:
            bid_hs += 2
        if self._sell_fill_ema > self._buy_fill_ema + 0.04:
            ask_hs += 2

        bid_hs = max(2, min(12, bid_hs))
        ask_hs = max(2, min(12, ask_hs))

        base_size = 3.0 * inv_scale * time_scale
        bid_factor = 1.0 if bid_hs >= 4 else 0.5
        ask_factor = 1.0 if ask_hs >= 4 else 0.5

        bid_size = max(0.5, round(base_size * bid_factor, 2))
        ask_size = max(0.5, round(base_size * ask_factor, 2))

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
            l2_size = max(0.5, round(base_size * 0.35, 2))

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
