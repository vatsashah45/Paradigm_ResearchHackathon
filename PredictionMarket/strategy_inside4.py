from __future__ import annotations

from orderbook_pm_challenge.strategy import BaseStrategy
from orderbook_pm_challenge.types import CancelAll, PlaceOrder, Side, StepState


class Strategy(BaseStrategy):
    """
    Inside-competitor v4: Stripped down, no cooldown, no fair_adj.
    Pure (comp_bid+1, comp_ask-1) quoting with large size.
    Only trade when comp_spread >= 5.
    """

    def on_step(self, state: StepState):
        comp_bid = state.competitor_best_bid_ticks
        comp_ask = state.competitor_best_ask_ticks

        actions = [CancelAll()]

        if comp_bid is None or comp_ask is None:
            return actions

        comp_spread = comp_ask - comp_bid

        if comp_spread < 5:
            return actions

        # Size based on regime
        if comp_spread >= 7:
            size = 25.0
        else:
            size = 15.0

        # Simple inside-competitor quotes
        bid_tick = comp_bid + 1
        ask_tick = comp_ask - 1

        # Inventory skew
        net_inv = state.yes_inventory - state.no_inventory
        abs_inv = abs(net_inv)
        skew = int(round(net_inv * 0.04))
        bid_tick -= skew
        ask_tick -= skew

        # Clamp inside competitor
        bid_tick = min(bid_tick, comp_bid + 1)
        bid_tick = max(bid_tick, 1)
        ask_tick = max(ask_tick, comp_ask - 1)
        ask_tick = min(ask_tick, 99)

        if bid_tick >= ask_tick:
            return actions

        # Scale with inventory
        inv_scale = max(0.1, 1.0 - abs_inv / 100.0)
        size = max(1.0, round(size * inv_scale, 2))

        if state.steps_remaining > 3:
            cost = bid_tick / 100.0 * size
            if state.free_cash > cost + 0.5:
                actions.append(PlaceOrder(side=Side.BUY, price_ticks=bid_tick, quantity=size))

        if state.steps_remaining > 3:
            avail_yes = max(0.0, state.yes_inventory)
            covered = min(size, avail_yes)
            uncovered = max(0.0, size - covered)
            cost = (1.0 - ask_tick / 100.0) * uncovered
            if state.free_cash > cost + 0.5 or covered >= size:
                actions.append(PlaceOrder(side=Side.SELL, price_ticks=ask_tick, quantity=size))

        return actions
