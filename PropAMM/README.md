# Paradigm Optimization Arena — Prop AMM Challenge

**Author:** 0xVatsaShah  
**Competition:** [arena.paradigm.xyz](https://arena.paradigm.xyz)

## Competition Overview

Design a Rust AMM strategy compiled to BPF that controls the entire `compute_swap()` function. Given reserves (rx, ry) and an input amount, the strategy decides how much to output. Scored over 1000 simulations with randomized parameters (volatility σ 0.01%-0.70%, retail arrival 0.4-1.2, normalizer fee 30-80 bps, normalizer liquidity 0.4x-2.0x).

**Edge** = profit vs true market price. Higher edge = better AMM.

## Architecture

The final submission (`submission.rs`) implements a **3-profile momentum switcher**:

- **Profile 0 (balanced):** ARB_K=5200, COUNTER_K=1000, SIZE_K=2900, default=55bps
- **Profile 1 (max defense):** ARB_K=6600, COUNTER_K=200, SIZE_K=4350, default=78bps  
- **Profile 2 (loss minimizer):** ARB_K=27333, COUNTER_K=4600, SIZE_K=2700, default=32bps

Key features:
- Kalman-filtered VWAP price tracking (p_hat slow + fast)
- Shock/toxicity/volatility EMA tracking
- Momentum scoring with adaptive EMA decay
- Cooldown-gated profile switching with asymmetric gap thresholds
- Triple-barrier bonus at step boundaries
- Fee floor with regime-adaptive decay

## Experiment Log

| File | Description | Local (100) | Local (1000) | Server |
|------|-------------|-------------|--------------|--------|
| `00_starter.rs` | Competition starter (constant 30bps CFMM) | 174 | — | — |
| `00_reference_strategy.rs` | Repository reference (#1 leaderboard) | 488.31 | 501.62 | 617.75 |
| `01_v1_single_profile_adaptive` | First adaptive fee AMM | 495.66 | — | — |
| `02_v2_directional_size_k` | Added directional SIZE_K | 494.08 | — | — |
| `03_v3_exact_p0_profile` | Exact P0 match + proper storage | 498.27 | 498.27 | **514.06 (#29)** |
| `04a_v4a_elapsed_cap_5` | ELAPSED_CAP=5 | 487 | — | — |
| `04b_v4b_regime_blend` | Regime blend fees | 487 | — | — |
| `05_v5_calm_rebate` | Calm rebate bonus | 498.27 | — | — |
| `06_v6_flat_discount` | Flat fee discount | 487 | — | — |
| `07_v7_3profile_momentum_switcher` | **Full 3-profile system** | **488.31** | **501.62** | **516.96 (#24)** |
| `08_v8_aggressive_switching` | Lower cooldown + gap thresholds | 487.28 | — | — |
| `09_v9_inventory_tracking_wip` | Inventory tracking (WIP) | — | — | — |
| `10a_higher_counter_rebate` | P0 COUNTER_K 1000→1400 | 487.99 | — | — |
| `10b_lower_base_fee` | BASE_FEE 16→12 bps | 484.49 | — | — |
| `10c_faster_floor_decay` | FLOOR_DECAY 850M→800M | 487.76 | — | — |
| `10d_higher_arb_defense` | ARB_K 5200→6000 | 488.14 | — | — |
| `11_v11_per_profile_fee_floors` | Per-profile fee floor tracking | 474.85 | — | — |
| `12_v12_two_profiles_no_p2` | 2 profiles only (remove P2) | 486.31 | — | — |
| `13_v13_flow_adaptive_decay` | Flow-adaptive floor decay | 487.95 | 501.43 | — |

## Key Insights

1. **Local ≠ Server:** Local benchmark (seeds 0-999) poorly predicts server scores. The reference gets 501 locally but 617 on server — a +23% gap.
2. **Multi-profile is the key win:** Going from single-profile (v3: 514 server) to 3-profile (v7: 517 server) was the biggest structural improvement.
3. **Parameter tuning is noise:** All parameter variants clustered at 484-488 locally. The ~4 point variance makes optimization via local benchmarks unreliable.
4. **Momentum scoring matters:** The profile switcher uses counterfactual edge scoring with adaptive EMA decay — this is what separates top strategies.

## File Structure

```
Paradigm_ResearchHackathon/
├── README.md              # This file
├── submission.rs          # Current submission (= v7)
└── attempts/
    ├── 00_starter.rs              # Competition starter
    ├── 00_reference_strategy.rs   # #1 leaderboard reference
    ├── 01-13_*.rs                 # All attempts chronologically
    └── sweep_*.rs                 # Parameter sweep variants
```

## Tech Stack

- **Language:** Rust compiled to Solana BPF bytecode
- **Fixed-point:** 1e9 scale, u128 intermediates
- **Storage:** 1024 bytes persistent across trades within simulation
- **CLI:** `prop-amm run/validate <file>.rs --simulations N`
