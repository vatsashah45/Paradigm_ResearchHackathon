# Paradigm Prediction Market Challenge — Strategy Attempts

## Challenge
Build a market-making strategy for a FIFO limit order book prediction market.
- **Framework**: `orderbook-pm-challenge` Python package
- **Simulation**: 2000 steps/sim, jump-diffusion price process
- **Scoring**: Mean edge across 200 simulations

## Leaderboard Context
- **#1**: $29.24
- **Our best submission**: $2.30 (#51) — `strategy_final.py`
- **Our best local**: $2.76 — `strategy_opt5.py`

---

## Strategy Evolution

### Phase 1: Initial Exploration (v1–v7)
| File | Mean Edge | Key Idea |
|------|-----------|----------|
| `strategy.py` | -19.5 | Asymmetric directional — massive arb losses |
| `strategy_v2.py` | ~-10 | Basic symmetric market-making |
| `strategy_v3.py` | ~-5 | Reduced size, wider spread |
| `strategy_v4.py` | ~-2 | Inventory management added |
| `strategy_v5.py` | ~0 | Better regime awareness |
| `strategy_v6.py` | ~0.3 | Spread-based regime detection |
| `strategy_v7.py` | ~0.68 | Only quote in wide-spread regimes |

### Phase 2: Regime Focus — spread_ticks=4 Only (v8–v16)
Key discovery: **Only profitable when competitor spread_ticks=4** (visual spread ≥ 7).

| File | Mean Edge | Key Idea |
|------|-----------|----------|
| `strategy_v8.py` | 0.81 | hs=2, size=3, threshold≥7 |
| `strategy_v9.py` | ~1.0 | Size tuning |
| `strategy_v10.py` | ~1.1 | Multi-level quotes |
| `strategy_v11.py` | 1.22 | hs=2, size=4.5, 3 levels |
| `strategy_v12.py` | 1.40 | size=6 |
| `strategy_v13.py` | 1.64 | size=8 |
| `strategy_v14.py` | ~1.8 | Reduced cooldown |
| `strategy_v15.py` | 2.02 | Simplified, no cooldown |
| `strategy_v16.py` | ~2.5 | size=10, no vol tracking |

### Phase 3: Simplification & Size (v17–v25)
| File | Mean Edge | Key Idea |
|------|-----------|----------|
| `strategy_v17.py` – `strategy_v20.py` | 1.8–2.3 | Multi-level experiments |
| `strategy_v21.py` – `strategy_v24.py` | 1.9–2.2 | Single level experiments |
| `strategy_v25.py` | 2.01 | Single level, size=8, clean (= `strategy_final.py`) |

### Phase 4: Penny Strategies
Tried quoting 1 tick inside competitor. Failed due to arb exposure.

| File | Mean Edge | Key Idea |
|------|-----------|----------|
| `strategy_penny.py` | -18.26 | Penny in all regimes — arb massacre |
| `strategy_penny2.py` | -5.59 | Penny only when comp_spread≥5 |
| `strategy_penny3.py` | -0.07 | Penny only in spread_ticks=4 |

### Phase 5: Hybrid & Multi-Regime
| File | Mean Edge | Key Idea |
|------|-----------|----------|
| `strategy_mid2_wide.py` | 1.38 | comp_mid±2, spread=5+ |
| `strategy_hybrid.py` | 1.52 | Size adapts by regime |

### Phase 6: Optimization Breakthrough (opt1–opt7)
Added fair-value tracking (arb fill detection → adjust fair ±1.0, 0.93 decay), cooldown after arb fills, inside-competitor clamp.

| File | Mean Edge | Key Idea |
|------|-----------|----------|
| `strategy_opt1.py` | 2.72 | fair_adj + cooldown(2) + inside-comp clamp |
| `strategy_opt2.py` | 2.61 | Stronger cooldown=3 |
| `strategy_opt3.py` | 2.68 | Milder cooldown=1 |
| `strategy_opt4.py` | 2.75 | No time_adj |
| **`strategy_opt5.py`** | **2.76** | **Pure hs=2, simplest: fair_adj+cooldown+clamp** |
| `strategy_opt6.py` | 1.32 | No inside-comp clamp (clamp is essential!) |
| `strategy_opt7.py` | 1.48 | No inventory skew |

---

## Key Findings

1. **Regime matters**: spread_ticks=4 (visual≥7) is the only profitable regime with simple approaches
2. **FIFO kills us**: CancelAll each step → lowest priority → must quote inside competitor
3. **Inside-competitor clamp is essential**: Keeps us inside competitor spread to capture retail
4. **Fair-value tracking helps**: Detecting arb fills adjusts where we think fair value is
5. **Cooldown after arb fills**: Pausing the side that got arb-swept reduces losses

## Fundamental Gap
We only profit in ~25% of sims (spread_ticks=4). Top strategies at $29 likely profit across ALL regimes. Need a fundamentally different approach.

---

## Utilities
- `run_test.py` — Test runner helper
- `analyze.py` — Basic analysis
- `analyze2.py` — Breakdown by spread_ticks regime (reads JSON from stdin)

## Running
```bash
cd /Users/vatsashah/prediction-market-challenge
uv run python -m orderbook_pm_challenge run /path/to/strategy.py --simulations 200
uv run python -m orderbook_pm_challenge run /path/to/strategy.py --simulations 200 --json | python analyze2.py
```
