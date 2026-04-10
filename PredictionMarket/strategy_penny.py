from __future__ import annotations

from orderbook_pm_challenge.strategy import BaseStrategy
from orderbook_pm_challenge.types import CancelAll, PlaceOrder, Side, StepState


class Strategy(BaseStrategy):
    """
    Penny-the-competitor strategy.

    Places orders 1 tick inside the competitor's best quotes, capturing
    ALL retail flow at maximum edge while minimizing arb loss. Active in
    all regimes where comp_spread >= 3 (spread_ticks >= 2).
    """

    def __init__(self) -> None:
        pass

    def on_step(self, state: StepState):
        steps_remaining = state.steps_remaining
        comp_bid = state.competitor_best_bid_ticks
        comp_ask = state.competitor_best_ask_ticks

        actions = [CancelAll()]

        if comp_bid is None or comp_ask is None:
            return actions

        comp_spread = comp_ask - comp_bid

        # Need at least 3 ticks of comp spread to fit inside
        if comp_spread < 3:
            return actions

        # Penny: 1 tick inside competitor on each side
        my_bid = comp_bid + 1
        my_ask = comp_ask - 1

        if my_ask <= my_bid:
            return actions

        net_inv = state.yes_inventory - state.no_inventory
        abs_inv = abs(net_inv)
        max_inv = 50.0

        inv_scale = max(0.05, 1.0 - abs_inv / (max_inv * 1.2))
        time_scale = 1.0
        if steps_remaining < 100:
            time_scale = max(0.1, steps_remaining / 100.0)

        size = max(0.5, round(10.0 * inv_scale * time_scale, 2))

        # Inventory skew: shift quotes to encourage rebalancing
        skew = int(round(net_inv * 0.05))
        my_bid = max(1, min(98, my_bid - skew))
        my_ask = max(2, min(99, my_ask - skew))
        if my_ask <= my_bid:
            my_bid = max(1, my_ask - 1)

        # Place bid
        if steps_remaining > 5 and net_inv < max_inv:
            cost = my_bid / 100.0 * size
            if state.free_cash > cost + 0.3:
                actions.append(PlaceOrder(side=Side.BUY, price_ticks=my_bid, quantity=size))

        # Place ask
        if steps_remaining > 5 and net_inv > -max_inv:
            avail = max(0.0, state.yes_inventory)
            cov = min(size, avail)
            unc = max(0.0, size - cov)
            cost = (1.0 - my_ask / 100.0) * unc
            if state.free_cash > cost + 0.3 or cov >= size:
                actions.append(PlaceOrder(side=Side.SELL, price_ticks=my_ask, quantity=size))

        # Flatten near end
        if steps_remaining < 6 and abs_inv > 0.5:
            comp_mid = (comp_bid + comp_ask) / 2.0
            fair_int = max(1, min(99, int(round(comp_mid))))
            flat = max(0.5, min(abs_inv * 0.5, 5))
            flat = round(flat, 2)
            if net_inv > 0:
                if state.yes_inventory >= flat or state.free_cash > 0.3:
                    actions.append(PlaceOrder(side=Side.SELL, price_ticks=fair_int, quantity=flat))
            else:
                if state.free_cash > fair_int / 100.0 * flat:
                    actions.append(PlaceOrder(side=Side.BUY, price_ticks=fair_int, quantity=flat))

        return actions
