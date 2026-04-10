from __future__ import annotations

from orderbook_pm_challenge.strategy import BaseStrategy
from orderbook_pm_challenge.types import CancelAll, PlaceOrder, Side, StepState


class Strategy(BaseStrategy):
    """
    Optimized market maker for spread_ticks=4 regimes.
    
    Key improvements over v25:
    - Fair value adjustment based on fill signals (tracks drift better)
    - Cooldown after large fills (avoids consecutive arb losses)
    - Ensure we stay inside the competitor spread dynamically
    """

    def __init__(self) -> None:
        self._min_observed_spread: int = 99
        self._observations: int = 0
        self._is_wide: bool = False
        self._fair_adj: float = 0.0
        self._last_posted_size: float = 8.0
        self._cooldown_bid: int = 0
        self._cooldown_ask: int = 0

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
        if self._observations >= 3 and not self._is_wide:
            if self._min_observed_spread >= 7:
                self._is_wide = True

        if not self._is_wide:
            return actions

        # --- Fair value tracking ---
        buy_filled = state.buy_filled_quantity
        sell_filled = state.sell_filled_quantity

        # Large fills suggest arb (price moved past our quote)
        arb_threshold = self._last_posted_size * 0.5
        if sell_filled > arb_threshold:
            # Price moved up past our ask. Adjust fair value up.
            self._fair_adj += 1.0
            self._cooldown_ask = 2
        if buy_filled > arb_threshold:
            # Price moved down past our bid. Adjust fair value down.
            self._fair_adj -= 1.0
            self._cooldown_bid = 2

        # Decay adjustment toward comp_mid (which also tracks)
        self._fair_adj *= 0.93

        # Decrement cooldowns
        if self._cooldown_bid > 0:
            self._cooldown_bid -= 1
        if self._cooldown_ask > 0:
            self._cooldown_ask -= 1

        fair = max(3.0, min(97.0, comp_mid + self._fair_adj))

        net_inv = state.yes_inventory - state.no_inventory
        abs_inv = abs(net_inv)
        max_inv = 50.0

        inv_scale = max(0.05, 1.0 - abs_inv / (max_inv * 1.2))
        time_scale = 1.0
        if steps_remaining < 100:
            time_scale = max(0.1, steps_remaining / 100.0)

        base_size = 8.0
        size = max(0.5, round(base_size * inv_scale * time_scale, 2))
        self._last_posted_size = size

        skew = net_inv * 0.1

        hs = 2
        inv_adj = min(2, int(abs_inv / 15.0))
        time_adj = 0
        if steps_remaining < 200:
            time_adj = 1
        if steps_remaining < 50:
            time_adj = 3
        hs = max(2, min(10, hs + inv_adj + time_adj))

        bid_tick = max(1, min(98, int(round(fair - hs - skew))))
        ask_tick = max(2, min(99, int(round(fair + hs - skew))))

        # Ensure we're inside the competitor spread
        bid_tick = min(bid_tick, comp_bid + 1) if comp_bid is not None else bid_tick
        ask_tick = max(ask_tick, comp_ask - 1) if comp_ask is not None else ask_tick

        if bid_tick >= ask_tick:
            bid_tick = max(1, ask_tick - 1)

        # Place bid (with cooldown check)
        if steps_remaining > 5 and net_inv < max_inv and self._cooldown_bid == 0:
            cost = bid_tick / 100.0 * size
            if state.free_cash > cost + 0.3:
                actions.append(PlaceOrder(side=Side.BUY, price_ticks=bid_tick, quantity=size))

        # Place ask (with cooldown check)
        if steps_remaining > 5 and net_inv > -max_inv and self._cooldown_ask == 0:
            avail = max(0.0, state.yes_inventory)
            cov = min(size, avail)
            unc = max(0.0, size - cov)
            cost = (1.0 - ask_tick / 100.0) * unc
            if state.free_cash > cost + 0.3 or cov >= size:
                actions.append(PlaceOrder(side=Side.SELL, price_ticks=ask_tick, quantity=size))

        # Flatten near end
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
