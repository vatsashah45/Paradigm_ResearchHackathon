from __future__ import annotations

import math
from collections import deque
from orderbook_pm_challenge.strategy import BaseStrategy
from orderbook_pm_challenge.types import CancelAll, PlaceOrder, Side, StepState


class Strategy(BaseStrategy):
    """
    Hybrid market-making strategy combining two insights:

    1. TIGHT QUOTES for retail capture: Place tiny orders just inside the
       competitor to be first in queue for retail fills. The key metric is
       keeping size tiny so arb loss per fill is small, while retail edge
       per fill (spread) is significant.

    2. DIRECTIONAL ADAPTATION: After detecting drift (arb fills are
       directional), widen quotes on the vulnerable side and narrow on
       the safe side. If price is rising, our asks are safe (retail buys
       at our ask = positive edge) but bids are dangerous (arb sells
       through our bid = negative edge).

    3. POST-FILL WIDENING: After getting hit, the price has likely moved.
       Widen immediately and wait for signals before re-tightening.
    """

    def __init__(self) -> None:
        self._initial_comp_mid: float | None = None
        self._fair_value: float = 50.0

        # Drift tracking
        self._drift_ema: float = 0.0  # positive = price rising
        self._vol_ema: float = 0.0
        self._buy_fill_ema: float = 0.0  # EMA of buy fills
        self._sell_fill_ema: float = 0.0  # EMA of sell fills

        # Post-fill cooldown
        self._last_buy_fill_step: int = -100
        self._last_sell_fill_step: int = -100

    def on_step(self, state: StepState):
        step = state.step
        steps_remaining = state.steps_remaining
        total_steps = step + steps_remaining

        comp_bid = state.competitor_best_bid_ticks
        comp_ask = state.competitor_best_ask_ticks

        # Fair value from competitor
        if comp_bid is not None and comp_ask is not None:
            comp_mid = (comp_bid + comp_ask) / 2.0
        elif comp_bid is not None:
            comp_mid = comp_bid + 2.0
        elif comp_ask is not None:
            comp_mid = comp_ask - 2.0
        else:
            comp_mid = 50.0

        if self._initial_comp_mid is None:
            self._initial_comp_mid = comp_mid
            self._fair_value = comp_mid

        # Fill tracking
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

        # Update fair value
        self._fair_value = comp_mid + self._drift_ema * 1.5
        self._fair_value = max(3.0, min(97.0, self._fair_value))

        net_inv = state.yes_inventory - state.no_inventory
        abs_inv = abs(net_inv)
        fair = self._fair_value
        fair_int = int(round(fair))

        actions = [CancelAll()]

        # ─── QUOTE PARAMETERS ────────────────────────────────────────
        max_inv = 40.0
        inv_scale = max(0.1, 1.0 - abs_inv / (max_inv * 1.3))

        time_scale = 1.0
        if steps_remaining < 80:
            time_scale = max(0.1, steps_remaining / 80.0)

        # Inventory skew
        skew = net_inv * 0.1

        # ─── DETECT DRIFT DIRECTION ──────────────────────────────────
        # If our buys are getting filled → price dropped → our bids are getting
        # arb-swept → dangerous to have bids, safe to have asks
        # If our sells are getting filled → price rose → our asks are getting
        # arb-swept → dangerous to have asks, safe to have bids

        buy_pressure = self._buy_fill_ema  # how much our buys get hit
        sell_pressure = self._sell_fill_ema  # how much our sells get hit

        # Steps since last fill on each side
        steps_since_buy_fill = step - self._last_buy_fill_step
        steps_since_sell_fill = step - self._last_sell_fill_step

        # Cooldown: after a fill, widen that side for a few steps
        buy_cooldown = max(0, 3 - steps_since_buy_fill)  # 0-3 extra ticks
        sell_cooldown = max(0, 3 - steps_since_sell_fill)

        # Vol adjustment
        vol_adj = min(4, int(self._vol_ema * 0.4))
        inv_adj = min(3, int(abs_inv / 15.0))

        # Time: widen late
        time_adj = 0
        if steps_remaining < 200:
            time_adj = 1
        if steps_remaining < 50:
            time_adj = 3

        # ─── ASYMMETRIC SPREADS ──────────────────────────────────────
        # Base: tight enough to capture retail
        base_hs = 2

        # BID side half-spread: widen if buys are getting hit (price dropping)
        bid_hs = base_hs + vol_adj + inv_adj + time_adj + buy_cooldown
        # If price is dropping (our buys getting hit), widen bids extra
        if buy_pressure > sell_pressure + 0.05:
            bid_hs += 2

        # ASK side half-spread: widen if sells are getting hit (price rising)
        ask_hs = base_hs + vol_adj + inv_adj + time_adj + sell_cooldown
        if sell_pressure > buy_pressure + 0.05:
            ask_hs += 2

        bid_hs = max(2, min(12, bid_hs))
        ask_hs = max(2, min(12, ask_hs))

        # ─── QUOTE SIZING ────────────────────────────────────────────
        # Tiny size at tight levels, larger at wide levels
        # The tighter we quote, the smaller we should be (more arb risk)
        bid_size_factor = 1.0 if bid_hs >= 4 else 0.6
        ask_size_factor = 1.0 if ask_hs >= 4 else 0.6

        base_size = 2.0 * inv_scale * time_scale

        bid_size = max(0.5, round(base_size * bid_size_factor, 2))
        ask_size = max(0.5, round(base_size * ask_size_factor, 2))

        # ─── PLACE ORDERS ────────────────────────────────────────────
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

        # ─── SECOND LEVEL (wider) ────────────────────────────────────
        if steps_remaining > 20:
            l2_bid = max(1, bid_tick - 3)
            l2_ask = min(99, ask_tick + 3)
            l2_size = max(0.5, round(base_size * 0.4, 2))

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

        # ─── SETTLEMENT FLATTEN ──────────────────────────────────────
        if steps_remaining < 6 and abs_inv > 0.5:
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

        return actions
