# Strategy Experiment Learnings

Running edge baseline: **501.62** (1000 sims, seeds 0-999, RAYON_NUM_THREADS=10)

---

## Session: Kappa Market Response Model

### Context
The ML toxic classifier (P(toxic|trade)) was trained and embedded (AUC=0.8765), but ML_ARB_BUMP=0 because any arb-dir fee increase regresses. The user's key insight: we need to predict *E[future arb correction | state]* (kappa), not P(toxic|trade).

### What We Built
- **train_response_model.py**: trains linear regression predicting step-ending spot deviation from fair.
- **Label**: `y = |spot_last_in_step - fair_price| / fair_price` (end-of-step drift)
- **R² = 0.356** (on 819k step-level samples from 200 sims)
- **Key predictors**: shock_hat (+0.999), tox_hat (+0.291), sigma (+0.119)
- Kappa is well-calibrated: mean_kappa ≈ mean_y (0.00568), correlation 0.5991

### Applications Tested (all regressed from 501.62)

| Application | KAPPA param | Edge | Conclusion |
|---|---|---|---|
| Shadow (zero) | KAPPA_ARB_BOOST=0 | 501.62 | ✓ baseline |
| Additive arb fee | KAPPA_ARB_BOOST=1000 BPS | 499.03 | regression |
| Additive arb fee (smaller) | KAPPA_ARB_BOOST=500 BPS | 500.58 | regression |
| Multiplicative arb_k (full) | KAPPA_ARB_MULT=SCALE | 493.63 | large regression |
| Multiplicative arb_k (25%) | KAPPA_ARB_MULT=SCALE/4 | 500.60 | regression |
| Kappa counter dampening | KAPPA_COUNTER_DAMP=12.5M | 501.51 | small regression |
| Smooth kappa floor decay | (replaced binary shock threshold) | 499.89 | regression |

### Root Cause of Kappa Failure
1. Kappa features (shock_hat, tox_hat, sigma) are ALREADY used in the existing fee model
2. Any additional arb-dir fee regresses because arb-dir retail is net positive (mean_edge=0.006)
3. Counter-dir dampening changes also regress — shock-only dampening is already well-calibrated
4. Floor decay changes regress — the binary shock threshold is already effective

### Key Architecture Insight
kappa ≈ 0.0033 + 1.0×shock_hat + 0.29×tox_hat + 0.12×sigma. These features are already weighted
and applied in: SHOCK_READ_K×shock, TOX_READ_K×tox, vol_fee=VOL_MULT×sigma, shock_fee=SHOCK_QUAD×shock².
The kappa signal is **redundant** with the existing fee model.

---

## Earlier Session Learnings

### Dead Ends
- **ML_ARB_BUMP=30 BPS** → 497.25: arb-dir retail is net positive (60.6% profitable), fee routes them to normalizer
- **SIZE_COUNTER_K=0** → 495.12: removing size penalty for counter-dir reduces revenue per trade
- **Remove shock dampening from counter rebate** → 501.49: needed because p_ref (VWAP) lags fair in volatile conditions
- **Directional VWAP weighting** → 501.52: shifting p_ref toward counter-dir creates more arb opportunities

### Fundamental Constraints
- Any mechanism increasing arb-dir fees routes those trades to normalizer → lose 2% of retail edge
- Counter-dir = 98% of retail edge (mean_edge=0.214, 0.1% toxic)
- Arb-dir = 2% of retail edge (mean_edge=0.006, 39.4% toxic) — net positive even with toxicity
- Bad sims (61/1000 negative edge) are dominated by high-sigma (GBM vol), not controllable by fee adjustments
- Optuna has already swept the main tunable parameters; local optimum at 501.62 is very stable

### Load-bearing Features
- Shock dampening on counter_scale: `counter_scale = 1 - 5×shock_hat` (removing hurts 501.49)
- Floor decay threshold: binary at shock_hat > 15M (920M vs 850M)
- ML_ARB_BUMP = 0 (arb-dir retail net positive; any fee increase regresses)

---

## Flow-Momentum Signal Analysis (DOUBLE_ARB_K experiment, shadow mode 501.62 ✓)

**Finding**: flow_arb_dir (arb-dir AND same-flow-dir) is LESS toxic than arb-not-flow!
- flow_arb_dir (12.1% of arb-dir): 25.6% toxic, mean_edge=0.00335 (better!)
- arb-not-flow (87.9% of arb-dir): 42.7% toxic, mean_edge=0.00280 (worse)
- Reason: momentum-driven arb (flow + arb same dir) = VWAP lag, not genuine adverse selection
- Counter-flow arb (spot away from fair DESPITE opposing flow) = more genuine adverse selection

**Conclusion**: flow-momentum signal is NOT useful for arb-dir premium because:
1. flow_arb_dir is actually LESS toxic → charging more hurts profitable trades
2. arb-not-flow is 87.9% of arb-dir → charging more here ≈ charging all arb-dir → regression

**ofi_ema finding**: mean=0.991, std=0.027 → nearly always near 1.0 because retail arrival rate is small
(1-2 orders/step, usually all same direction → step_ofi≈1.0 → ofi_ema≈1.0 always)
ofi_ema is effectively constant and cannot be used as a fee signal.

---

## Session: Dynamic p_ref Blend Weight (train_pref_blend.py)

### What Was Tested
ML upgrade to p_ref blend: `w = min(shock, 0.3)` → piecewise rule based on (cnt_ema, shock_hat).

### Analysis Findings (train_pref_blend.py)
- **Current formula**: w = min(shock_hat, 0.3), actual w ≈ 0.003 avg (shock_hat rarely > 0.003)
- **Optimal w\* from data**: w\* ≈ 0.65 in ALL regimes (minimizes |p_ref - fair_price|)
- **Model R² ≈ 0.001**: features cannot predict WHEN to use more p_fast; it's a constant offset
- **Quadrant analysis**: differences between regimes are <0.04 in w\*, too small to matter

### Piecewise Rule Implemented
```
if shock_hat > 5M && cnt_ema > 1e9: w = 0.25  (trending+busy)
elif shock_hat > 5M:                 w = 0.08  (shock, quiet)
else:                                w = 0.12  (calm baseline)
```

### Result
**Edge: 501.01** (-0.61 regression). Reverted to baseline.

### Root Cause — Critical Insight
Minimizing `|p_ref - fair_price|` is NOT the right objective for p_ref.

**p_ref's actual role**: maximize the signal gap `(spot - p_ref)` during arb events to charge higher arb fees.

**Why p_slow beats p_fast as p_ref**:
- p_fast (ALPHA=0.65) tracks spot aggressively → when spot > fair (arb), p_fast also > fair
- Using more p_fast → p_ref rises toward spot → dev2 = (spot - p_ref)² SHRINKS → less arb fee
- p_slow (Kalman, lagging) stays near fair while spot diverges → LARGER dev2 → better arb detection

**Conclusion**: `w ≈ shock_hat ≈ 0.003` (essentially pure p_slow) is intentionally correct. The lagging nature of p_slow is a FEATURE, not a bug. p_ref blending is LOCKED.

---

## Session: Contextual Bandit for Mode Switcher

### What Was Tested
Replace global EMA scores (3 × i64) with per-context EMA scores (N_CTX × 3 × i64).
Context = regime bucket indexed by (shock_bin × tox_bin).

| Variant | N_CTX | Context update | Edge | Delta |
|---|---|---|---|---|
| Baseline (global) | 1 | — | 501.62 | — |
| 16 contexts | 16 | per-trade | 498.65 | -3.0 |
| 4 contexts | 4 | per-trade | 500.23 | -1.4 |
| 4 contexts | 4 | step-boundary cached | 500.81 | -0.8 |

### Root Cause of Failure

**More switching = more instability**: baseline 1.50 switches/sim, all variants 2.3–3.9 switches/sim.

**Why ctx scores cause more switching**: The SWITCH_ABS_GAP_UP/DOWN thresholds are calibrated for global score magnitudes. Per-context EMAs have the same long-run mean (EMA converges), but higher short-run variance (fewer samples per context). Noise crossings of the threshold trigger spurious switches.

**BIAS_ADAPT already does contextual scoring**: Inside `raw_edge_sample`, BIAS_ADAPT_0 (+55M calm), BIAS_ADAPT_1 (+60M volatile), BIAS_ADAPT_2 (quadratic) apply continuous regime weights. The contextual bandit duplicates this with coarser bins and more noise.

**Step-boundary caching partially fixes thrashing**: Caching ctx at step boundaries reduced switches from 2.6 to 2.3/sim and improved edge from 500.23 to 500.81 — but still a net regression. The remaining thrashing is from step-to-step context changes.

### What Would Make This Work
A proper contextual bandit would need:
1. Context updated only at step boundaries (not per-trade) ✓ tried, still regresses
2. Higher switching thresholds to compensate for noisier ctx scores (would require re-tuning)
3. OR: context scores as ADDITIVE correction to global scores (not replacement)
4. OR: many more steps per sim to build up per-context statistics

**Bottom line**: The profiles are already well-calibrated globally + BIAS_ADAPT provides regime sensing. Contextual bandit cannot add signal that justifies the instability cost.

---

---

## Session: Controlled Deviation Offset δ* (Signal-Generating MM)

### What Was Built
- **train_delta_model.py**: Grid search over δ ∈ [0, 0.005] per step, estimating edge change from fee sensitivity.
- Concept: `p_ref = p_slow - s * δ` where `s = sign(spot - p_slow)` — shifts p_ref AWAY from spot to amplify arb/counter deviation signal.

### Analysis Results
- Per-step optimal δ*: 55% of steps → δ*=0, 45% → δ*>0. Mean δ*=0.00223.
- **arb_frac = 44%** of retail trades are arb-direction (by count; edge contribution only 2%)
- Theoretical aggregate gain at fixed δ=0.001: **+0.27/sim** (monotone increasing curve)
- Theoretical max at δ=0.005: **+1.52/sim**
- R²=0.007 from state features → dominant predictor is arb_frac per step (unpredictable)

### Actual Sim Result
**δ=0.001: Edge 501.48 (-0.14 regression)**. Reverted to baseline.

### Root Cause — Same as ML_ARB_BUMP
δ increases arb-dir dev2 → higher arb fee → arb-dir retail routes to normalizer.
- Even 1% more arb-dir routing: 24k trades × 0.006 avg_edge = **-0.14** (exactly the observed regression)
- Arb-dir routing is highly elastic at the Optuna optimum: the current arb fee is already at the margin
- The theoretical model assumed routing unchanged; routing is the dominant effect

### Key Insight
**p_ref deviation is already optimally calibrated.** The existing Kalman lag + arb premium structure creates exactly the right dev2 for the fee equilibrium. Any shift in p_ref (which changes dev2 → changes arb fee) is equivalent to ML_ARB_BUMP acting through a different channel. Both hit the same routing sensitivity wall.

The "signal-generating MM" concept is correct as a description of what the AMM does, but the δ offset is not a free tuning knob — it's tightly constrained by arb-dir routing elasticity.

---

---

## Session: Dynamic Curvature (SIZE_LIVE_K state-dependent multiplier)

### Hypothesis (user insight)
- `size_penalty = k(state) × size_live_k × input² / sum`
- SIZE_LIVE_K is "second-order" (vanishes at q→0) → arb routing threshold should be unchanged
- Arb decides to route based on spot vs p_ref at q→0; retail decides at finite q
- Strategy: high-shock → higher k (protect); low-shock → lower k (attract retail volume)

### Variants Tested

| Variant | HIGH_MUL | LOW_MUL | Edge | Δ |
|---|---|---|---|---|
| Baseline | 1.0× | 1.0× | 501.62 | — |
| User direction (hi protect) | 1.1× | 0.9× | 500.44 | -1.18 |
| Opposite (hi flatten) | 0.9× | 1.1× | 501.36 | -0.26 |
| Low only (steepen calm) | 1.0× | 1.1× | 501.39 | -0.23 |
| High only (flatten shock) | 0.9× | 1.0× | 501.37 | -0.25 |

**ALL regress. All reverted.**

### Root Cause
The "second-order at q→0" argument is theoretically correct, but arb trades at FINITE q where size_penalty is non-zero. SIZE_LIVE_K affects effective arb execution cost → IS part of arb routing decision.
Additionally, the profile switcher already provides coarse curvature adaptation:
- P0 (balanced): size_live_k = 2900 BPS (used in calm regimes)
- P1 (max defense): size_live_k = 4350 BPS (used in volatile regimes, 1.5× P0)

Optuna calibrated both SIZE_LIVE_K and profile selection jointly. Any independent conditional multiplier breaks the joint calibration.

### Key Empirical Fact Added
**SIZE_LIVE_K is routing-elastic**: even ±10% state-conditional changes route some trades away.

This extends the routing-elasticity principle: not just arb-dir fee (first-order) but ALSO size penalty (second-order) is at the Optuna routing equilibrium. The equilibrium is even tighter than expected.

---

## Session: Flow Regime State Variable (train_flow_regime.py)

### Analysis Results
- **Lag-1 P(same direction) = 0.5389** (excess = 0.039) — statistically significant
- Signal decays to lag-2 = 0.511, lag-3+ ≈ 0.500 — memory is ≈1 trade deep
- Best alpha = 0.5 by directional accuracy
- Hi-flow sigma underestimation: +0.000528 vs lo-flow +0.000290 — flow predicts slight extra vol
- flow_abs correlation with next-step EDGE: **+0.0037** (essentially zero)

### Variants Tested

| Variant | TREND_VOL_MULT | Edge | Δ |
|---|---|---|---|
| Baseline | — | 501.62 | — |
| First attempt (multiply whole sigma) | 0.136 | 120.24 | -381 (catastrophic) |
| Inflate update term only | 0.136 | 499.93 | -1.69 |
| Inflate update term (small) | 0.01 | 501.42 | -0.20 |

### Root Cause of Failure
**sigma IS in the quote function** (vol_fee = VOL_MULT × sigma). ANY sigma change → fee change → routing shift at the knife-edge equilibrium. The "routing equilibrium preserved" argument fails because:
1. Quotes IMMEDIATELY reflect sigma changes (simulation routes based on current quote)
2. State estimator is not separable from Q(state) — sigma changes ARE Q changes
3. Even TREND_VOL_MULT=0.01 (1/14 of the calibrated value) still causes -0.20 regression

### Key Lesson
The "change state trajectory without changing Q(state)" insight is correct in theory, but sigma is not a pure state variable — it feeds directly into vol_fee every trade. The "invisible state change" would require a variable that is used INTERNALLY without any path to the quote function. No such variable exists in the current architecture (all state ultimately feeds into fees somehow).

---

## Session: Inventory Resistance (imbalance EMA + directional size_live_k inflation)

### Mechanism
Track `imbalance = EMA(signed trade_frac)` (dimensionless, buy+ / sell-). When `|imbalance| > threshold` and trade worsens imbalance, inflate size_live_k → arb golden-section optimal q shrinks → more arb routes to normalizer.

### Results

| Variant | Edge | Δ |
|---|---|---|
| Raw input (wrong units, ry>>rx) | 470.83 | -30.79 |
| Trade_frac (correct units, RESIST_MULT=4) | 485.43 | -16.19 |

### Root Cause — Architectural Failure

**In this 2-AMM simulator, our pool's spot price is corrected EXCLUSIVELY by trades through it.**

Routing arb to normalizer does NOT help our pool:
- Normalizer corrects its own reserves → its spot improves
- Our AMM reserves are unchanged → our spot stays diverged from fair
- Our diverged spot creates more arb opportunities
- Arb count INCREASES +21% (more steps where `|spot - fair| > 0.01%` threshold)
- Net: more arb events, each at lower loss/trade, but total worse

**Key observation**: At RESIST_MULT=0 (no resistance), arb = 1.97M events/1000 sims. At RESIST_MULT=4, arb = 2.40M events despite resistance — the pool becomes MORE adversely selected, not less.

### Why It Works in Real Markets But Not Here

Real market: external price discovery → arb using normalizer corrects absolute asset price → our pool naturally aligns without trading through it.

This simulator: no external price transmission. Pool spot = f(its own trade history). Must accept arb to stay priced correctly for retail.

**Conclusion**: Cannot selectively filter arb in a single-pool isolated pricing system. Arb is a NECESSARY COST of pool pricing quality. Arb resistance ≈ insurance premium payment required to maintain pool attractiveness for retail.

---

## Session: Signed Flow Price Estimation (train_price_estimator.py)

### Analysis
12-feature Ridge regression to predict `(fair_price - p_hat) / p_hat` from trade flow features.
- **Group A** (spot_dev, fast_dev, shock): R²=0.914 — mispricing state already captured
- **Group B** (signed flow: signed_trade_frac, ofi_ema, cnt_ema): R²=0.011 standalone
- **Group C** (within-step rolling: vwap_drift, dir_imbalance, vol_conc): R²=0.917 (collinear with A)
- **A+B**: R²=0.940 — signed_trade_frac adds +0.026 R² beyond spot deviation alone
- Key signal: `signed_trade_frac = direction × trade_frac`, coefficient ≈ +0.422 (B-only model)
- **R²=0.011 signal is real** (seed-split CV, no temporal leakage, significant above zero)

### Variants Tested

| Variant | PEST_SIGNED_TF | Edge | Δ |
|---|---|---|---|
| Baseline | 0 | 501.62 | — |
| Full model coefficient | 0.422 | 500.34 | -1.28 |
| 10× reduced | 0.042 | 501.52 | -0.10 |
| 52× reduced | 0.008 | 501.48 | -0.14 |

### Root Cause — Fourth Structural Constraint

**The p_ref lag IS the edge mechanism.** Making p_hat track fair_price more accurately REDUCES edge:
- `p_ref = wmul(p_slow, 1-w) + wmul(p_fast, w)` where p_slow = p_hat
- Deviation-based fees = `(spot - p_ref)² / ...` — the fee signal comes from THIS gap
- When p_hat tracks directional flow → p_ref moves toward spot → gap SHRINKS → lower arb fee
- This is exactly the same failure mode as the p_ref blend experiment (p_fast regression)
- The lagging Kalman (K≈0.001) is INTENTIONAL — it maximizes deviation gap during arb events

**Conclusion**: Price estimation accuracy is ANTI-correlated with edge in this AMM structure. Any improvement in p_hat tracking reduces the `|spot - p_ref|` gap during arb events, which is the primary fee driver. The 12-feature model cannot be deployed.

---

## Session: Inventory Skew (Avellaneda-Stoikov)

### Theory (user insight)
Inventory introduces a state variable orthogonal to price. OU error: `de_t = σS dW_t - K(e_t - αq_t)dt`.
Center shifts to `e_center = αq`, creating a family of equilibria parameterized by q. Suggested γ ≈ 0.15-0.25.

### Variants Tested

| Variant | Signal | GAMMA_INV | Edge | Δ | Root cause |
|---|---|---|---|---|---|
| rx_frac (spot-based) | rx×spot/(rx×spot+ry) | 200M (γ=0.20) | **501.62** | 0.0 | DEGENERATE: always 0.5 |
| inv_frac (p_hat-based) | rx×p_ref/(rx×p_ref+ry) | 200M (γ=0.20) | 497.73 | -3.89 | Linear arb premium, routing-elastic |
| Signed cross-step OFI EMA | EMA(sign × |net_flow|/vol) | 200M (γ=0.20) | **147.26** | -354.36 | Signal [0,SCALE], 2000 BPS max → 78% trade loss |

### Why rx_frac ≡ 0.5 (xy AMM invariant)
For any constant-product AMM: `spot = ry/rx`, so `rx × spot = ry` exactly.
Therefore: `rx_frac = rx×spot / (rx×spot + ry) = ry / (2ry) = 0.5` always.
ANY inventory signal using current spot and reserves is degenerate for xy AMM.

### Why p_hat-based inv_frac = linear arb premium
`inv_frac = rx×p_ref / (rx×p_ref + ry) = p_ref / (p_ref + spot)`
When spot > p_ref: `inv_frac < 0.5` → pool "short X at fair" → premium for side=0 (buy X).
This is exactly `is_arb_dir` with linear-in-deviation premium instead of `dev²`. Same routing failure.

### Why cross-step OFI EMA is catastrophic
Signed OFI EMA ∈ [0, SCALE]. At γ=0.20: max fee = 0.20 × SCALE = 2000 BPS. Base fee = 16-78 BPS.
With 50% OFI EMA: inv_fee = 1000 BPS → ALL arb and most retail routes to normalizer.
Catastrophic trade volume loss: 7.6M → 1.8M trades/1000 sims (-76%).

### Key Insight: A-S doesn't transfer to xy AMM
Avellaneda-Stoikov designed for limit order books where inventory accumulates independently of price.
For xy AMM:
1. Inventory IS price: pool position fully determined by xy=k + current price (via spot=ry/rx)
2. No truly orthogonal inventory signal exists — all directional signals correlate with deviation
3. Any directional fee asymmetry → routing elasticity → regression (same as all previous experiments)
4. The existing counter-direction rebate + arb-direction premium already implement the optimal form

### Fifth Structural Constraint
**Any directional fee asymmetry based on accumulated flow = routing-elastic.** The router responds to
directional fee asymmetry by routing the penalized direction to normalizer, eliminating the accumulated
signal and causing net trade loss. The A-S cross-step inventory concept cannot escape this in a single-pool
AMM because the router has full information about the fee function.

---

## Equilibrium Summary (All Dead Ends)

Every attempt to perturb the fee surface regresses:

| Mechanism | What changes | Min Δ |
|---|---|---|
| ML_ARB_BUMP | Arb fee (1st order) | -4.4 |
| δ offset (p_ref) | dev2 / arb fee (via 2nd order) | -0.14 |
| Dynamic SIZE_LIVE_K | Output curvature (2nd order) | -0.23 |
| Kappa fee boost | Arb fee boost | regress |
| Contextual bandit | Mode selection | -0.8 |
| p_ref blend | p_slow/p_fast ratio | -0.61 |
| Flow regime (sigma inflation) | Vol fee via sigma | -0.20 |
| Inventory resistance | Directional size_live_k | -16.19 |
| Signed flow p_hat correction | p_ref via p_hat | -1.28 |
| Inventory skew (rx_frac/spot) | Directional fee | 0.0 (degenerate) |
| Inventory skew (inv_frac/p_hat) | Directional fee | -3.89 |
| Inventory skew (signed OFI EMA) | Directional fee | -354.36 |
| ML routing (ROUTE_CORR P1 oracle) | Mode selection bias | -4.83 |

**The AMM is at a knife-edge routing equilibrium.** Every dimension (fee level, p_ref, curvature, mode selection, sigma trajectory) is already Optuna-calibrated. The harness is NOT "predicting a label" — it's maintaining an equilibrium where marginal perturbations to any fee-surface parameter route trades away.

**Structural constraints**:
1. Every state variable feeds into the quote function — no purely-internal state
2. Pool spot price is only corrected by trades through it — deflecting arb degrades pool quality
3. Cannot selectively filter arb without increasing pool mispricing → more arb total
4. **p_ref lag is the edge mechanism**: better p_hat tracking reduces deviation-based arb fee → less edge
5. **Any directional fee asymmetry is routing-elastic**: inventory skew fails because router responds
6. **Profile switching is routing-elastic**: higher-fee profiles repel retail even at step level; SCORE_BIASES already at Optuna optimum

---

---

## Session: ML Routing Pipeline (train_routing_pipeline.py + run_convergence_loop.py)

### What Was Built
- **train_routing_pipeline.py**: Python reimplementation of `quote_profile` (all 3 profiles), computes counterfactual edges per trade, aggregates to step level, runs Ridge regression → derives ROUTE_CORR constants
- **run_convergence_loop.py**: Sequential 1k-seed batch loop: run→train→apply→evaluate→keep/revert, convergence at |Δedge| < 0.3 for 2 rounds
- **strategy.rs additions**: 6 ROUTE_CORR constants (`P0/P1/P2 × SHOCK/TOX`), hook in `raw_edge_sample` applies `route_corr = rc_shock × shock / SCALE + rc_tox × tox / SCALE` after adaptive bias

### CSV Convention Discovery
- CSV `side=0`: rx INCREASES (AMM received X from user), ry DECREASES (AMM gave Y) — OPPOSITE to code convention
- `code_side = 1 - CSV_side`, `code_input = CSV_output` (what user sends to AMM)
- Pre-trade reserves for CSV_side=0: `rx_pre = reserve_x - CSV_output`, `ry_pre = reserve_y + CSV_input`

### Oracle Analysis (counterfactual, ignoring routing equilibrium)
- Mode dist in current sim: P0=85.2%, P1=6.8%, P2=8.0%
- Oracle optimal mode: P0=0%, **P1=73%**, P2=27%  ← P1 looks dramatically underused
- P1 mean advantage over P0: +0.019658/step (R²=0.454)
- Derived constants: ROUTE_CORR_P1_SHOCK=33_394_062_374, ROUTE_CORR_P2_SHOCK=11_043_707_547

### Simulation Tests (all configurations tested)

| Configuration | P1% | P2% | Edge | Δ |
|---|---|---|---|---|
| All zeros (baseline) | 6.8% | 8.0% | 501.62 | — |
| P1_SHOCK=33.4B (full oracle) | 89% | ~1% | 496.79 | -4.83 |
| P1_SHOCK=15B (half) | ~50% | ~14% | 498.75 | -2.87 |
| P1/P2 suppressed to -50B | 0.2% | 0.2% | 498.29 | -3.33 |
| P2_SHOCK=50B, P1=-100B | ~0% | 86% | 495.13 | -6.49 |

**ROUTE_CORR = 0 is the optimum.** All constants reverted to zero.

### Root Cause: Sixth Structural Constraint
**Profile switching is routing-elastic.** P1's higher fees (tox_read_k=98 vs 74 BPS, shock_read_k=1733 vs 733 BPS) repel retail in volatile regimes when P1 is selected more. The counterfactual oracle assumed routing unchanged; the routing response is the dominant effect.

**Why the oracle was wrong**: Per-step counterfactual edge gain from P1 ignores that P1's higher fees cause MORE retail to route to the normalizer. The oracle shows "if we had used P1 on all those steps, we would have had higher edge per trade" — but in practice the higher fee routes many of those trades away.

**Why the current mix (P0=85%, P1=7%, P2=8%) is optimal**: The Optuna-optimized SCORE_BIASES = [0, -120M, -210M] with adaptive bias were already calibrated for the routing equilibrium.

### Infrastructure Preserved
The ROUTE_CORR hook remains in `raw_edge_sample` as a future extension point (all zeros = no effect). The constants are at the top of strategy.rs grouped with other tunable parameters.

---

---

## Session: ARB + NORMALIZER Models + cnt_ema Fee Boost + Code Cleanup

### What Was Built
- **python/arb_model.py**: Per-sim ARB agent tracker (fee gap, routing preference)
- **python/normalizer_model.py**: Per-sim normalizer competition tracker (missed revenue analysis)
- **python/analyze_frontrun.py**: Combined analysis runner (correlations, quintile breakdowns)

### Key Analysis Findings
- cnt_ema → norm_fee: **r=0.662** (strong sim-level proxy)
- arb_prefers_us_pct → norm_fee: **r=0.828** (arbs come to us when norm_fee high, our fee lower)
- High norm_fee (Q5, 75 BPS): 322 edge/sim missed retail revenue (we undercharge by 19-45 BPS)
- Total avg missed revenue: 93.26 edge/sim (upper bound if we could perfectly charge norm_fee)
- tox_hat → norm_fee: **r=-0.360** (NEGATIVE: high tox when norm_fee low; many arbs in low-norm_fee sims)

### cnt_ema Fee Boost (fee_floor via make_target): ALL REGRESS

| K (BPS max boost) | Edge | Delta |
|---|---|---|
| 5 | 497.67 | -3.95 |
| 10 | 488.91 | -12.71 |
| 20 | 473.49 | -28.13 |
| 50 | 449.73 | -51.89 |

**Root Cause**: Adding cnt_boost to max_target updates fee_floor upward. fee_floor decays slowly (850M-920M). When cnt_ema spikes in high-activity sims, floor locks HIGH for ~4+ steps, elevating stored_fee for ALL profiles. In sims where norm_fee is moderate (35-45 BPS), our elevated stored_fee > norm_fee → retail routes to normalizer → large revenue loss. The downside (losing retail) dominates the upside (charging more when norm_fee is high). The r=0.662 correlation is not precise enough to avoid overshooting norm_fee.

### Code Cleanup (strategy.rs: 833 → 769 lines)
Removed confirmed dead code — baseline preserved at 501.62:
- **Dead global constants**: BASE_FEE, SPOT_BUMP_ARB_K, SPOT_BUMP_COUNTER_K, TOX_READ_K (global), SHOCK_READ_K (global), SIZE_LIVE_K (global), VOL_MULT (global), TOX_QUAD, TOX_CUBE, SHOCK_QUAD, SHOCK_CUBE (all superseded by per-profile P*_BASE_FEE etc.)
- **Dead function**: ceil_mul_div (never called)
- **Dead sigma_class dimension**: P0L=P0H, P1L=P1H, P2L=P2H (identical values → sigma_class has no effect on profile_params). Consolidated to P0_*, P1_*, P2_* constants.
- **Dead step accumulators**: step_shock_cum, step_tox_max, step_dir_vol (tracked but never used in any fee/mode decision; per comment "used in sweep variants" — currently dead)
- **Redundant make_target calls**: P0=P1=P2 target fee params → all profiles compute identical target. Simplified to single computation.

### Key Insight: Per-Profile Target Fees Are All Equal
P0_BASE_FEE=P1_BASE_FEE=P2_BASE_FEE=16 BPS, all VOL_MULT/TOX/SHOCK params identical.
→ target0=target1=target2 always → fee_floor is driven by one value.
→ S_FEE=S_FEE_1=S_FEE_2 always the same value.
Profiles only differentiate via quote_profile (spot_arb_k, spot_counter_k, shock_read_k, tox_read_k).

### Why Front-Running Normalizer Fails
The "front-running" concept (charge close to norm_fee) is theoretically sound but practically blocked:
1. norm_fee is NOT observable at runtime — only correlates with cnt_ema (r=0.662)
2. Any fee increase based on noisy proxy risks overshooting norm_fee → retail routes to normalizer
3. The fee-floor mechanism locks in elevated fees for multiple steps (slow decay)
4. The fee-floor is shared: raising it for high-activity moments hurts calm-period trades too
5. All fee-based approaches reduce to the same routing-elasticity wall documented in prior sessions

### Architecture Note: Switching Constants Exhausted
All 11 switching constants (SCORE_BIASES[0-2], BIAS_ADAPT_0/1/2, BIAS_SHOCK_THRESH, BARRIER_K,
SWITCH_COOLDOWN, SWITCH_ABS_GAP_UP/DOWN, SWITCH_REL_BPS, ROUTE_BONUS_W, SCORE_DECAY_LO/HI,
SHOCK_DECAY_THRESH) — all confirmed at Nash optimum by exhaustive coarse sweep.

---

## Next Directions to Explore
1. **Counter-dir fee structure**: 98% of edge comes here; improvements multiply 50x vs arb
2. **Profile parameter adjustments**: P0/P1/P2 params might have room beyond Optuna sweep
3. **Fee floor decay**: binary threshold at 15M shock (920M vs 850M) — finer gradient?
