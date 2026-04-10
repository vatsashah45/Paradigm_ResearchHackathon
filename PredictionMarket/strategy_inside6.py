from __future__ import annotations

from orderbook_pm_challenge.strategy import BaseStrategy
from orderbook_pm_challenge.types import CancelAll, PlaceOrder, Side, StepState


class Strategy(BaseStrategy):
    """
    Inside-competitor v6: Fixed arb detection (proportional to posted size).
    - Quote at (comp_bid+1, comp_ask-1)
    - Arb detection: fill > 50% of posted size
    - Trade in spread_ticks >= 3 (comp_spread >= 5)
    """

    def __init__(self) -> None:
        self._fair_adj: float = 0.0
        self._prev_comp_bid: int | None = None
        self._prev_comp_ask: int | None = None
        self._cooldown_bid: int = 0
        self._cooldown_ask: int = 0
        self._last_bid_size: float = 0.0
        self._last_ask_size: float = 0.0

    def on_step(self, state: StepState):
        comp_bid = state.competitor_best_bid_ticks
        comp_ask = state.competitor_best_ask_ticks

        actions = [CancelAll()]

        if comp_bid is None or comp_ask is None:
            return actions

        comp_spread = comp_ask - comp_bid

        # Only trade when comp_spread >= 5 (spread_ticks >= 3)
        if comp_spread < 5:
            self._prev_comp_bid = comp_bid
            self._prev_comp_ask = comp_ask
            return actions

        # Regime-adaptive base size
        if comp_spread >= 7:
            base_size = 25.0
        else:
            base_size = 15.0

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

        # --- Detect arb fills (proportional threshold) ---
        buy_filled = state.buy_filled_quantity
        sell_filled = state.sell_filled_quantity

        bid_arb_thresh = self._last_bid_size * 0.5 if self._last_bid_size > 0 else 999
        ask_arb_thresh = self._last_ask_size * 0.5 if self._last_ask_size > 0 else 999

        if sell_filled > ask_arb_thresh:
            self._fair_adj += 0.8
            self._cooldown_ask = 2
        if buy_filled > bid_arb_thresh:
            self._fair_adj -= 0.8
            self._cooldown_bid = 2

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
            return actions

        # Scale size with inventory
        inv_scale = max(0.15, 1.0 - abs_inv / 100.0)
        size = max(1.0, round(base_size * inv_scale, 2))

        steps_remaining = state.steps_remaining
        self._last_bid_size = 0.0
        self._last_ask_size = 0.0

        if steps_remaining > 3 and self._cooldown_bid == 0:
            cost = bid_tick / 100.0 * size
            if state.free_cash > cost + 0.5:
                actions.append(PlaceOrder(side=Side.BUY, price_ticks=bid_tick, quantity=size))
                self._last_bid_size = size

        if steps_remaining > 3 and self._cooldown_ask == 0:
            avail_yes = max(0.0, state.yes_inventory)
            covered = min(size, avail_yes)
            uncovered = max(0.0, size - covered)
            cost = (1.0 - ask_tick / 100.0) * uncovered
            if state.free_cash > cost + 0.5 or covered >= size:
                actions.append(PlaceOrder(side=Side.SELL, price_ticks=ask_tick, quantity=size))
                self._last_ask_size = size

        return actions
