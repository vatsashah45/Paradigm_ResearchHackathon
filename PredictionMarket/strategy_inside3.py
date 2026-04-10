from __future__ import annotations

from orderbook_pm_challenge.strategy import BaseStrategy
from orderbook_pm_challenge.types import CancelAll, PlaceOrder, Side, StepState


class Strategy(BaseStrategy):
    """
    Inside-competitor v3: Skip spread_ticks<=2, optimize sizes for 3 and 4.
    Quote at (comp_bid+1, comp_ask-1).
    """

    def __init__(self) -> None:
        self._fair_adj: float = 0.0
        self._prev_comp_bid: int | None = None
        self._prev_comp_ask: int | None = None
        self._cooldown_bid: int = 0
        self._cooldown_ask: int = 0

    def on_step(self, state: StepState):
        comp_bid = state.competitor_best_bid_ticks
        comp_ask = state.competitor_best_ask_ticks

        actions = [CancelAll()]

        if comp_bid is None or comp_ask is None:
            return actions

        comp_spread = comp_ask - comp_bid

        # Only trade when comp_spread >= 5 (spread_ticks >= 3)
        if comp_spread < 5:
            return actions

        # Regime-adaptive base size
        if comp_spread >= 7:        # spread_ticks=4
            base_size = 25.0
            cooldown_len = 2
        else:                       # spread_ticks=3 (comp_spread 5-6)
            base_size = 12.0
            cooldown_len = 3

        # --- Infer price direction from competitor quote changes ---
        if self._prev_comp_ask is not None and self._prev_comp_bid is not None:
            ask_shift = comp_ask - self._prev_comp_ask
            bid_shift = comp_bid - self._prev_comp_bid
            if ask_shift > 0:
                self._fair_adj += 0.5 * ask_shift
            if bid_shift < 0:
                self._fair_adj += 0.5 * bid_shift
        self._prev_comp_bid = comp_bid
        self._prev_comp_ask = comp_ask

        self._fair_adj *= 0.90

        # --- Detect arb fills ---
        buy_filled = state.buy_filled_quantity
        sell_filled = state.sell_filled_quantity

        if sell_filled > 1.5:
            self._fair_adj += 0.8
            self._cooldown_ask = cooldown_len
        if buy_filled > 1.5:
            self._fair_adj -= 0.8
            self._cooldown_bid = cooldown_len

        if self._cooldown_bid > 0:
            self._cooldown_bid -= 1
        if self._cooldown_ask > 0:
            self._cooldown_ask -= 1

        # --- Compute quote ticks ---
        bid_tick = comp_bid + 1
        ask_tick = comp_ask - 1

        # Apply fair value adjustment
        adj_ticks = int(round(self._fair_adj))
        bid_tick += adj_ticks
        ask_tick += adj_ticks

        # Inventory skew
        net_inv = state.yes_inventory - state.no_inventory
        abs_inv = abs(net_inv)
        skew_ticks = int(round(net_inv * 0.05))
        bid_tick -= skew_ticks
        ask_tick -= skew_ticks

        # CLAMP: Must stay inside competitor
        bid_tick = min(bid_tick, comp_bid + 1)
        bid_tick = max(bid_tick, 1)
        ask_tick = max(ask_tick, comp_ask - 1)
        ask_tick = min(ask_tick, 99)

        if bid_tick >= ask_tick:
            mid = (bid_tick + ask_tick) // 2
            bid_tick = max(1, mid)
            ask_tick = min(99, mid + 1)

        # Scale size with inventory
        inv_scale = max(0.1, 1.0 - abs_inv / 80.0)
        size = max(1.0, round(base_size * inv_scale, 2))

        steps_remaining = state.steps_remaining

        if steps_remaining > 3 and self._cooldown_bid == 0:
            cost = bid_tick / 100.0 * size
            if state.free_cash > cost + 0.5:
                actions.append(PlaceOrder(side=Side.BUY, price_ticks=bid_tick, quantity=size))

        if steps_remaining > 3 and self._cooldown_ask == 0:
            avail_yes = max(0.0, state.yes_inventory)
            covered = min(size, avail_yes)
            uncovered = max(0.0, size - covered)
            cost = (1.0 - ask_tick / 100.0) * uncovered
            if state.free_cash > cost + 0.5 or covered >= size:
                actions.append(PlaceOrder(side=Side.SELL, price_ticks=ask_tick, quantity=size))

        return actions
