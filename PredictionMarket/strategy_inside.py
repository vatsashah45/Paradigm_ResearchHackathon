from __future__ import annotations

from orderbook_pm_challenge.strategy import BaseStrategy
from orderbook_pm_challenge.types import CancelAll, PlaceOrder, Side, StepState


class Strategy(BaseStrategy):
    """
    Inside-competitor strategy: Always quote at (comp_bid+1, comp_ask-1).
    Works in ALL regimes where there's room inside the competitor spread.
    Key insight: we are the ONLY liquidity inside the competitor spread,
    so ALL retail flow must trade with us first.
    """

    def __init__(self) -> None:
        self._net_inv: float = 0.0
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

        comp_spread = comp_ask - comp_bid  # in ticks

        # Need at least 2 ticks of gap to fit inside (comp_bid+1 < comp_ask-1)
        # i.e. comp_spread >= 3
        if comp_spread < 3:
            return actions

        # --- Infer price direction from competitor quote changes ---
        if self._prev_comp_ask is not None:
            ask_shift = comp_ask - self._prev_comp_ask
            bid_shift = comp_bid - self._prev_comp_bid
            # Arb ate asks (price went up) -> asks shifted up
            if ask_shift > 0:
                self._fair_adj += 0.5 * ask_shift
            # Arb ate bids (price went down) -> bids shifted down
            if bid_shift < 0:
                self._fair_adj += 0.5 * bid_shift  # negative
        self._prev_comp_bid = comp_bid
        self._prev_comp_ask = comp_ask

        # Decay fair adjustment
        self._fair_adj *= 0.90

        # --- Detect arb fills (we got adversely filled) ---
        buy_filled = state.buy_filled_quantity
        sell_filled = state.sell_filled_quantity

        if sell_filled > 2.0:
            self._fair_adj += 0.8
            self._cooldown_ask = 2
        if buy_filled > 2.0:
            self._fair_adj -= 0.8
            self._cooldown_bid = 2

        if self._cooldown_bid > 0:
            self._cooldown_bid -= 1
        if self._cooldown_ask > 0:
            self._cooldown_ask -= 1

        # --- Compute quote ticks ---
        # Base: 1 tick inside competitor on each side
        bid_tick = comp_bid + 1
        ask_tick = comp_ask - 1

        # Apply fair value adjustment: shift both quotes
        adj_ticks = int(round(self._fair_adj))
        bid_tick += adj_ticks
        ask_tick += adj_ticks

        # Inventory skew: shift quotes to discourage building more inventory
        net_inv = state.yes_inventory - state.no_inventory
        abs_inv = abs(net_inv)
        skew_ticks = int(round(net_inv * 0.05))
        bid_tick -= skew_ticks
        ask_tick -= skew_ticks

        # CLAMP: Never go outside the competitor (would be pointless)
        # and never cross the competitor (arb bait)
        bid_tick = min(bid_tick, comp_bid + 1)
        bid_tick = max(bid_tick, 1)
        ask_tick = max(ask_tick, comp_ask - 1)
        ask_tick = min(ask_tick, 99)

        # Ensure bid < ask
        if bid_tick >= ask_tick:
            mid = (bid_tick + ask_tick) // 2
            bid_tick = max(1, mid)
            ask_tick = min(99, mid + 1)

        # --- Size: large enough to capture full retail orders ---
        # Retail mean notional = 2.64–6.34, mean ~$4.5
        # At mid prices: ~9 shares per order. Post 20 to capture everything.
        base_size = 20.0

        # Scale down with inventory to limit exposure
        inv_scale = max(0.1, 1.0 - abs_inv / 80.0)
        size = max(1.0, round(base_size * inv_scale, 2))

        # --- Place orders ---
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
