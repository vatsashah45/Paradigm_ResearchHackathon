#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use core::convert::TryInto;
use pinocchio::{account_info::AccountInfo, entrypoint, pubkey::Pubkey, ProgramResult};
use prop_amm_submission_sdk::{set_return_data_bytes, set_return_data_u64, set_storage};

/// REQUIRED by the harness
pub const NAME: &str = "I'm not a quant I went to b-skool";
pub const MODEL_USED: &str = "Opus 4.6";

/// 1e9 fixed-point scale
const SCALE: u128 = 1_000_000_000;
const BPS: u128 = SCALE / 10_000; // 100_000

// Storage layout (1024 bytes total; we use first 232 bytes):
// [0..8)     stored_fee    (u64, 1e9)
// [8..16)    last_step     (u64)
// [16..24)   sigma_hat     (u64, 1e9)
// [24..32)   p_hat         (u64, 1e9)
// [32..40)   fee_floor     (u64, 1e9)
// [40..48)   cnt           (u64) : trades within current step
// [48..56)   init_flag     (u64) : 0=uninitialized
// [56..64)   tox_hat       (u64, 1e9) : smoothed toxicity EMA
// [64..72)   sum_w         (u64, 1e9) : VWAP weight accumulator
// [72..80)   sum_wp        (u64, 1e9) : VWAP weighted-price accumulator
// [80..88)   last_spot     (u64, 1e9) : previous spot (ry/rx)
// [88..96)   shock_hat     (u64, 1e9) : EMA(|spot_t - spot_{t-1}| / spot_{t-1})
// [96..104)  p_hat_fast    (u64, 1e9) : fast-tracking p_hat for gate/tox
// [104..112) mode          (u64) : active model index [0..2]
// [112..136) scores[0..3]  (i64 x 3) : momentum scores
// [136..144) last_sw       (u64) : step at last mode switch
// [144..152) cnt_ema       (u64) : EMA of trades-per-step

const S_FEE: usize = 0;
const S_STEP: usize = 8;
const S_SIGMA: usize = 16;
const S_PHAT: usize = 24;
const S_FLOOR: usize = 32;
const S_CNT: usize = 40;
const S_INIT: usize = 48;
const S_TOX: usize = 56;
const S_SUM_W: usize = 64;
const S_SUM_WP: usize = 72;
const S_LAST_SPOT: usize = 80;
const S_SHOCK: usize = 88;
const S_PHAT_FAST: usize = 96;
const S_MODE: usize = 104;
const S_SCORES: usize = 112; // 3 x i64 = 24 bytes
const S_LAST_SWITCH: usize = 136;
const S_CNT_EMA: usize = 144;
const S_STEP_VOL: usize = 152;
const S_STEP_DIR_VOL: usize = 160;
const S_STEP_SHOCK_CUM: usize = 168;
const S_STEP_TOX_MAX: usize = 176;
const S_FEE_1: usize = 184;   // stored_fee for profile 1
const S_FEE_2: usize = 192;   // stored_fee for profile 2
// ---- Per-profile stored-fee dynamics (after_swap) ----
// Profile 0 (balanced)
const P0_BASE_FEE:   u128 = 16 * BPS;
const P0_VOL_MULT:   u128 = 600_000_000;
const P0_TOX_QUAD:   u128 = 15_000 * BPS;
const P0_TOX_CUBE:   u128 = 55_000 * BPS;
const P0_SHOCK_QUAD: u128 = 120_000 * BPS;
const P0_SHOCK_CUBE: u128 = 95_000 * BPS;
// Profile 1 (max defense)
const P1_BASE_FEE:   u128 = 16 * BPS;
const P1_VOL_MULT:   u128 = 600_000_000;
const P1_TOX_QUAD:   u128 = 15_000 * BPS;
const P1_TOX_CUBE:   u128 = 55_000 * BPS;
const P1_SHOCK_QUAD: u128 = 120_000 * BPS;
const P1_SHOCK_CUBE: u128 = 95_000 * BPS;
// Profile 2 (loss minimizer)
const P2_BASE_FEE:   u128 = 16 * BPS;
const P2_VOL_MULT:   u128 = 600_000_000;
const P2_TOX_QUAD:   u128 = 15_000 * BPS;
const P2_TOX_CUBE:   u128 = 55_000 * BPS;
const P2_SHOCK_QUAD: u128 = 120_000 * BPS;
const P2_SHOCK_CUBE: u128 = 95_000 * BPS;

// ---- Tunable constants (compute_swap) ----
const FEE_MIN: u128 = BPS;
const FEE_MAX: u128 = 1000 * BPS;
const SIZE_FEE_K: u128 = 2800 * BPS;
const PRICE_DEV_K: u128 = 400 * BPS;
const DEFAULT_FEE: u128 = 25 * BPS;
const INITIAL_PRICE: u128 = 100 * SCALE;
const REBATE_RISK_MULT: u128 = 1_500_000_000;

// ---- Tunable constants (after_swap) ----
const ALPHA: u128 = 20_000_000;
const ALPHA_FAST: u128 = 650_000_000;
const VOL_DECAY: u128 = 998_000_000;
const VOL_DECAY_FAST: u128 = 880_000_000;
const VOL_CAP: u128 = 50_000_000;
const INIT_SIGMA: u128 = 3_000_000;
const FLOOR_DECAY: u128 = 850_000_000;    // calm-regime floor decay rate
const FLOOR_DECAY_HI: u128 = 920_000_000; // high-shock floor decay rate
const GATE_K: u128 = 4 * SCALE;
const GATE_FLOOR: u128 = 60_000_000;
const STEP_CAP: u64 = 16;
const ELAPSED_CAP: u64 = 3;
const EARLY_STEP_THRESH: u64 = 30;
const TOX_DECAY: u128 = 650_000_000;
const SHOCK_DECAY: u128 = 350_000_000;

// Momentum switcher params
// Adaptive score decay (#2): faster after shocks, stable in calm
const SCORE_DECAY_LO: u128 = 940_000_000;
const SCORE_DECAY_HI: u128 = 980_000_000;
const SHOCK_DECAY_THRESH: u128 = 20_000_000;

const SWITCH_COOLDOWN: u64 = 3;
const SWITCH_ABS_GAP_UP: i64 = 30_000_000;
const SWITCH_ABS_GAP_DOWN: i64 = 210_000_000;
const SWITCH_REL_BPS: i64 = 100;
const ROUTE_BONUS_W: u128 = 145_000_000;

const NUM_PROFILES: usize = 3;

// Score biases per profile (base values)
// P0=balanced, P1=max defense, P2=loss minimizer
const SCORE_BIASES: [i64; NUM_PROFILES] = [
    0, -120_000_000, -210_000_000,
];
// Adaptive biases
const BIAS_ADAPT_0: i64 = 55_000_000;  // boost balanced in calm
const BIAS_ADAPT_1: i64 = 60_000_000;  // release max-defense in volatile
const BIAS_ADAPT_2: i64 = 60_000_000; // release loss minimizer in volatile
const BIAS_SHOCK_THRESH: u128 = 5_000_000;

const CNT_EMA_DECAY: u128 = 950_000_000;

const BARRIER_K: i64 = 65_000_000;

// Kalman p_hat obs_var: controls how fast p_hat tracks VWAP (smaller = faster)
// Currently asymmetric: faster downward than upward
const KALMAN_OBS_VAR_DOWN: u128 = 20_000_000;  // when VWAP < p_hat (price fell)
const KALMAN_OBS_VAR_UP:   u128 = 50_000_000;  // when VWAP > p_hat (price rose)
const KALMAN_INNO_THRESH:  u128 = 10_000_000;  // 1% innovation gating


// ---- Profile params (quote_profile) ----
const SIGMA_CLASS_THRESH: u128 = 5_000_000;  // 0.5% (kept for sigma_class computation)

// Profile 0 (balanced)
const P0_ARB_K: u128 = 5_200 * BPS;
const P0_COUNTER_K: u128 = 1_000 * BPS;
const P0_SIZE_K: u128 = 2_900 * BPS;
const P0_SHOCK_K: u128 = 733 * BPS;
// Profile 1 (max defense)
const P1_ARB_K: u128 = 6_600 * BPS;
const P1_COUNTER_K: u128 = 200 * BPS;
const P1_SIZE_K: u128 = 4_350 * BPS;
const P1_SHOCK_K: u128 = 1_733 * BPS;
// Profile 2 (loss minimizer)
const P2_ARB_K: u128 = 27_333 * BPS;
const P2_COUNTER_K: u128 = 4_600 * BPS;
const P2_SIZE_K: u128 = 2_700 * BPS;
const P2_SHOCK_K: u128 = 1_600 * BPS;

pub fn get_model_used() -> &'static str {
    MODEL_USED
}

entrypoint!(process_instruction);

pub fn process_instruction(
    _program_id: &Pubkey,
    _accounts: &[AccountInfo],
    instruction_data: &[u8],
) -> ProgramResult {
    if instruction_data.is_empty() {
        return Ok(());
    }

    match instruction_data[0] {
        0 | 1 => {
            let out = compute_swap(instruction_data);
            set_return_data_u64(out);
        }

        2 => {
            handle_after_swap(instruction_data);
        }

        3 => set_return_data_bytes(NAME.as_bytes()),
        4 => set_return_data_bytes(MODEL_USED.as_bytes()),
        _ => {}
    }

    Ok(())
}

// -------------------- storage helpers --------------------

fn rd64(data: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(data[off..off + 8].try_into().unwrap())
}
fn wr64(data: &mut [u8], off: usize, v: u64) {
    data[off..off + 8].copy_from_slice(&v.to_le_bytes());
}
fn rd_i64(data: &[u8], off: usize) -> i64 {
    i64::from_le_bytes(data[off..off + 8].try_into().unwrap())
}
fn wr_i64(data: &mut [u8], off: usize, v: i64) {
    data[off..off + 8].copy_from_slice(&v.to_le_bytes());
}

// -------------------- math helpers (1e9 fixed-point) --------------------

fn abs_diff(a: u128, b: u128) -> u128 {
    if a >= b { a - b } else { b - a }
}
fn wmul(a: u128, b: u128) -> u128 {
    (a * b) / SCALE
}
fn wdiv(a: u128, b: u128) -> u128 {
    if b == 0 { return 0; }
    (a * SCALE) / b
}
fn floor_mul_div(a: u128, b: u128, c: u128) -> u128 {
    if c == 0 { return 0; }
    let q = a / c;
    let r = a % c;
    q * b + r * b / c
}
fn clamp_fee(fee: u128) -> u128 {
    if fee < FEE_MIN { FEE_MIN } else if fee > FEE_MAX { FEE_MAX } else { fee }
}
fn pow_wad(mut base: u128, mut exp: u64) -> u128 {
    let mut r = SCALE;
    while exp > 0 {
        if exp & 1 == 1 { r = wmul(r, base); }
        base = wmul(base, base);
        exp >>= 1;
    }
    r
}
fn to_u64_cap(x: u128) -> u64 {
    if x > u64::MAX as u128 { u64::MAX } else { x as u64 }
}
fn to_i64_cap_signed(x: i128) -> i64 {
    if x > i64::MAX as i128 {
        i64::MAX
    } else if x < i64::MIN as i128 {
        i64::MIN
    } else {
        x as i64
    }
}
fn iabs(x: i128) -> i128 {
    if x < 0 { -x } else { x }
}
fn ema_i64(prev: i64, sample: i64, decay: u128) -> i64 {
    let p = prev as i128;
    let s = sample as i128;
    let d = decay as i128;
    let num = p * d + s * (SCALE as i128 - d);
    to_i64_cap_signed(num / SCALE as i128)
}
fn add_i64_cap(a: i64, b: i64) -> i64 {
    to_i64_cap_signed(a as i128 + b as i128)
}

// -------------------- profile parameter lookup --------------------

// Returns (spot_arb_k, spot_counter_k, default_fee, tox_read_k, shock_read_k, size_live_k, risk_gate_counter)
#[inline(never)]
fn profile_params(profile: u8, _sigma_class: u8) -> (u128, u128, u128, u128, u128, u128, bool) {
    let (spot_arb_k, spot_counter_k, size_live_k, shock_read_k, default_fee, tox_read_k) = match profile {
        0 => (P0_ARB_K, P0_COUNTER_K, P0_SIZE_K, P0_SHOCK_K, 55 * BPS, 74 * BPS),
        1 => (P1_ARB_K, P1_COUNTER_K, P1_SIZE_K, P1_SHOCK_K, 78 * BPS, 98 * BPS),
        _ => (P2_ARB_K, P2_COUNTER_K, P2_SIZE_K, P2_SHOCK_K, 32 * BPS,  2 * BPS),
    };
    (spot_arb_k, spot_counter_k, default_fee, tox_read_k, shock_read_k, size_live_k, true)
}

// -------------------- quote_profile (forced out-of-line for BPF stack) --------------------

#[inline(never)]
fn quote_profile(
    side: u8,
    input: u128,
    rx: u128,
    ry: u128,
    init_flag: u64,
    spot: u128,
    p_ref: u128,
    stored_fee: u128,
    tox_hat: u128,
    shock_hat: u128,
    is_arb_dir: bool,
    profile: u8,
    sigma_class: u8,
) -> u128 {
    let (spot_arb_k, spot_counter_k, default_fee, tox_read_k, shock_read_k, size_live_k, risk_gate_counter) =
        profile_params(profile, sigma_class);

    let mut fee = if init_flag == 0 {
        let deviation0 = wdiv(abs_diff(spot, INITIAL_PRICE), INITIAL_PRICE);
        default_fee + wmul(PRICE_DEV_K, deviation0)
    } else {
        let deviation = if p_ref > 0 {
            wdiv(abs_diff(spot, p_ref), p_ref)
        } else {
            0
        };
        let dev2 = wmul(deviation, deviation);

        if risk_gate_counter {
            let mut f = stored_fee + wmul(tox_read_k, tox_hat) + wmul(shock_read_k, shock_hat);
            if is_arb_dir {
                f = f.saturating_add(wmul(spot_arb_k, dev2));
            } else {
                let risk = if tox_hat > shock_hat { tox_hat } else { shock_hat };
                let mut scale = SCALE.saturating_sub(wmul(REBATE_RISK_MULT, risk));
                if scale > SCALE { scale = SCALE; }
                let shock_risk = wmul(shock_hat, 5 * SCALE);
                let counter_scale = SCALE.saturating_sub(shock_risk).min(SCALE);
                let adapt_counter_k = wmul(spot_counter_k, counter_scale);
                let raw_rebate = wmul(adapt_counter_k, dev2);
                f = f.saturating_sub(wmul(raw_rebate, scale));
            }
            f
        } else {
            let bump = if is_arb_dir { wmul(spot_arb_k, dev2) } else { 0 };
            stored_fee + bump + wmul(tox_read_k, tox_hat) + wmul(shock_read_k, shock_hat)
        }
    };
    fee = clamp_fee(fee);

    let reserve_in = if side == 0 { ry } else { rx };
    let sum = input + reserve_in;
    let den_base = reserve_in * SCALE + input * (SCALE - fee);
    let ski = size_live_k * input;
    let size_penalty = match ski.checked_mul(input) {
        Some(num) => num / sum,
        None => (ski / sum) * input,
    };
    let den = den_base.saturating_sub(size_penalty);
    if den == 0 { return 0; }

    let k = rx * ry;
    if side == 0 {
        let new_rx = floor_mul_div(k, SCALE, den);
        rx.saturating_sub(new_rx)
    } else {
        let new_ry = floor_mul_div(k, SCALE, den);
        ry.saturating_sub(new_ry)
    }
}

// -------------------- scoring helper (forced out-of-line for BPF stack) --------------------

#[inline(never)]
fn raw_edge_sample(
    side: u8,
    input: u128,
    rx_pre: u128,
    ry_pre: u128,
    init_flag: u64,
    spot_pre: u128,
    p_ref_pre: u128,
    stored_fee_pre: u128,
    tox_pre: u128,
    shock_pre: u128,
    is_arb_pre: bool,
    profile: u8,
    sigma_class: u8,
) -> i64 {
    let q = quote_profile(side, input, rx_pre, ry_pre, init_flag, spot_pre, p_ref_pre, stored_fee_pre, tox_pre, shock_pre, is_arb_pre, profile, sigma_class);

    let edge = if side == 0 {
        input as i128 - wmul(q, p_ref_pre) as i128
    } else {
        wmul(input, p_ref_pre) as i128 - q as i128
    };

    let mut s = edge;

    if !is_arb_pre {
        let qq = if side == 0 { wmul(q, p_ref_pre) } else { q };
        s += wmul(ROUTE_BONUS_W, qq) as i128;
    }

    let base_bias = SCORE_BIASES[profile as usize];
    let vol_t = if shock_pre >= BIAS_SHOCK_THRESH { SCALE } else { wdiv(shock_pre, BIAS_SHOCK_THRESH) };
    let calm_t = SCALE.saturating_sub(vol_t);
    let adapt = match profile {
        0 => wmul(calm_t, BIAS_ADAPT_0 as u128) as i64,
        1 => wmul(vol_t, BIAS_ADAPT_1 as u128) as i64,
        2 => wmul(wmul(vol_t, vol_t), BIAS_ADAPT_2 as u128) as i64, // quadratic: extreme vol only
        _ => 0,
    };
    let bias = add_i64_cap(base_bias, adapt);
    add_i64_cap(to_i64_cap_signed(s), bias)
}

#[inline(never)]
fn score_one_profile(
    side: u8,
    input: u128,
    rx_pre: u128,
    ry_pre: u128,
    init_flag: u64,
    spot_pre: u128,
    p_ref_pre: u128,
    stored_fee_pre: u128,
    tox_pre: u128,
    shock_pre: u128,
    is_arb_pre: bool,
    profile: u8,
    sigma_class: u8,
    prev_score: i64,
) -> i64 {
    let e = raw_edge_sample(side, input, rx_pre, ry_pre, init_flag, spot_pre, p_ref_pre, stored_fee_pre, tox_pre, shock_pre, is_arb_pre, profile, sigma_class);
    // #2: Adaptive decay based on shock level
    let shock_t = if shock_pre >= SHOCK_DECAY_THRESH { SCALE } else { wdiv(shock_pre, SHOCK_DECAY_THRESH) };
    let adapt_decay = SCORE_DECAY_HI - wmul(shock_t, SCORE_DECAY_HI - SCORE_DECAY_LO);
    ema_i64(prev_score, e, adapt_decay)
}

// -------------------- tag 2 wrapper (isolates 1024-byte stack alloc) --------------------

#[inline(never)]
fn handle_after_swap(instruction_data: &[u8]) {
    if instruction_data.len() < 42 + 1024 {
        return;
    }
    let mut storage = [0u8; 1024];
    storage.copy_from_slice(&instruction_data[42..42 + 1024]);
    after_swap(instruction_data, &mut storage);
    let _ = set_storage(&storage);
}

// -------------------- tag 0/1: compute_swap --------------------

pub fn compute_swap(data: &[u8]) -> u64 {
    if data.len() < 25 + 1024 { return 0; }
    let side = data[0];
    let input = rd64(data, 1) as u128;
    let rx = rd64(data, 9) as u128;
    let ry = rd64(data, 17) as u128;
    let storage = &data[25..25 + 1024];
    if input == 0 || rx == 0 || ry == 0 { return 0; }

    let init_flag = rd64(storage, S_INIT);
    let spot = wdiv(ry, rx);
    let p_ref = if init_flag == 0 {
        INITIAL_PRICE
    } else {
        let p_slow = rd64(storage, S_PHAT) as u128;
        let p_fast = rd64(storage, S_PHAT_FAST) as u128;
        let sh = rd64(storage, S_SHOCK) as u128;
        let w = if sh > 300_000_000 { 300_000_000 } else { sh };
        wmul(p_slow, SCALE - w) + wmul(p_fast, w)
    };
    let is_arb_dir = (spot > p_ref && side == 0) || (spot < p_ref && side == 1);

    let tox_hat = rd64(storage, S_TOX) as u128;
    let shock_hat = rd64(storage, S_SHOCK) as u128;
    let sigma = rd64(storage, S_SIGMA) as u128;
    let sigma_class = if sigma >= SIGMA_CLASS_THRESH { 1u8 } else { 0u8 };

    // Use active mode's profile for quoting
    let mut mode = if init_flag == 0 { 0 } else { rd64(storage, S_MODE) as u8 };
    if mode as usize >= NUM_PROFILES { mode = 0; }

    let stored_fee = match mode {
        1 => rd64(storage, S_FEE_1) as u128,
        2 => rd64(storage, S_FEE_2) as u128,
        _ => rd64(storage, S_FEE) as u128,
    };

    let out = quote_profile(side, input, rx, ry, init_flag, spot, p_ref, stored_fee, tox_hat, shock_hat, is_arb_dir, mode, sigma_class);
    if out > u64::MAX as u128 { u64::MAX } else { out as u64 }
}

// -------------------- tag 2: after_swap --------------------
pub fn after_swap(data: &[u8], storage: &mut [u8]) {
    let side = data[1];
    let input = rd64(data, 2) as u128;
    let _output = rd64(data, 10) as u128;
    let rx = rd64(data, 18) as u128;
    let ry = rd64(data, 26) as u128;
    let step = rd64(data, 34);

    let last_step = rd64(storage, S_STEP);
    let mut p_hat = rd64(storage, S_PHAT) as u128;
    let mut sigma = rd64(storage, S_SIGMA) as u128;
    let mut fee_floor = rd64(storage, S_FLOOR) as u128;
    let mut cnt = rd64(storage, S_CNT);
    let init_flag = rd64(storage, S_INIT);

    let mut tox_hat = rd64(storage, S_TOX) as u128;
    let mut sum_w = rd64(storage, S_SUM_W) as u128;
    let mut sum_wp = rd64(storage, S_SUM_WP) as u128;

    let mut last_spot = rd64(storage, S_LAST_SPOT) as u128;
    let mut shock_hat = rd64(storage, S_SHOCK) as u128;
    let mut p_hat_fast = rd64(storage, S_PHAT_FAST) as u128;
    let mut mode = rd64(storage, S_MODE) as u8;
    if mode as usize >= NUM_PROFILES { mode = 0; }
    let mut cnt_ema = rd64(storage, S_CNT_EMA);
    let mut step_vol = rd64(storage, S_STEP_VOL) as u128;

    // Load scores into array
    let mut scores = [0i64; NUM_PROFILES];
    let mut last_switch = rd64(storage, S_LAST_SWITCH);

    // Initialize on first call
    if init_flag == 0 {
        p_hat = if rx > 0 { wdiv(ry, rx) } else { INITIAL_PRICE };
        sigma = INIT_SIGMA;
        fee_floor = DEFAULT_FEE;
        tox_hat = 0;
        cnt = 0;
        sum_w = 0;
        sum_wp = 0;
        last_spot = p_hat;
        shock_hat = 0;
        p_hat_fast = p_hat;
        mode = 0;
        last_switch = step;
        wr64(storage, S_FEE, DEFAULT_FEE as u64);
    } else {
        // Only load scores if initialized (otherwise they stay at 0)
        let mut p = 0usize;
        while p < NUM_PROFILES {
            scores[p] = rd_i64(storage, S_SCORES + p * 8);
            p += 1;
        }
    }

    // -------------------- momentum scoring (loop-based for BPF stack) --------------------
    if init_flag != 0 {
        let mut rx_pre = rx;
        let mut ry_pre = ry;
        let mut pre_ok = true;
        if side == 0 {
            rx_pre = rx.saturating_add(_output);
            if ry < input { pre_ok = false; } else { ry_pre = ry - input; }
        } else {
            if rx < input { pre_ok = false; } else { rx_pre = rx - input; }
            ry_pre = ry.saturating_add(_output);
        }

        if pre_ok && rx_pre > 0 && ry_pre > 0 {
            let spot_pre = wdiv(ry_pre, rx_pre);
            let p_slow_pre = rd64(storage, S_PHAT) as u128;
            let p_fast_pre = rd64(storage, S_PHAT_FAST) as u128;
            let sh_pre = rd64(storage, S_SHOCK) as u128;
            let w_pre = if sh_pre > 300_000_000 { 300_000_000 } else { sh_pre };
            let p_ref_pre = wmul(p_slow_pre, SCALE - w_pre) + wmul(p_fast_pre, w_pre);
            let is_arb_pre = (spot_pre > p_ref_pre && side == 0) || (spot_pre < p_ref_pre && side == 1);

            let stored_fee_pre = rd64(storage, S_FEE) as u128;
            let tox_pre = rd64(storage, S_TOX) as u128;
            let shock_pre = rd64(storage, S_SHOCK) as u128;
            let sigma_pre = rd64(storage, S_SIGMA) as u128;
            let sigma_class_pre = if sigma_pre >= SIGMA_CLASS_THRESH { 1u8 } else { 0u8 };

            // Score each profile one at a time (loop reuses stack frame)
            let mut p = 0u8;
            while p < NUM_PROFILES as u8 {
                scores[p as usize] = score_one_profile(
                    side, input, rx_pre, ry_pre, init_flag,
                    spot_pre, p_ref_pre, stored_fee_pre, tox_pre, shock_pre,
                    is_arb_pre, p, sigma_class_pre, scores[p as usize],
                );
                p += 1;
            }

            // Find best mode
            let mut best_mode = 0u8;
            let mut best_score = scores[0];
            let mut bp = 1u8;
            while bp < NUM_PROFILES as u8 {
                if scores[bp as usize] > best_score {
                    best_mode = bp;
                    best_score = scores[bp as usize];
                }
                bp += 1;
            }

            let cur_score = scores[mode as usize];

            if best_mode != mode && step.saturating_sub(last_switch) >= SWITCH_COOLDOWN {
                let rel_gap = iabs(best_score as i128) * SWITCH_REL_BPS as i128 / 10_000;
                let abs_gap = if best_mode > mode { SWITCH_ABS_GAP_UP as i128 } else { SWITCH_ABS_GAP_DOWN as i128 };
                let gap = if rel_gap > abs_gap { rel_gap } else { abs_gap };
                if (best_score as i128) > (cur_score as i128) + gap {
                    mode = best_mode;
                    last_switch = step;
                }
            }
        }
    }

    // Detect new time step
    let new_step = step > last_step || init_flag == 0;

    // Triple-barrier: at step boundary, adjust current mode score based on realized risk
    if new_step && init_flag != 0 && last_spot > 0 {
        let spot_now = if rx > 0 { wdiv(ry, rx) } else { last_spot };
        let realized = wdiv(abs_diff(spot_now, last_spot), last_spot);
        let barrier_bonus = if sigma > realized {
            ((sigma - realized) as i128 * BARRIER_K as i128 / SCALE as i128) as i64
        } else {
            -(((realized - sigma) as i128 * BARRIER_K as i128 / SCALE as i128) as i64)
        };
        if (mode as usize) < NUM_PROFILES {
            scores[mode as usize] = add_i64_cap(scores[mode as usize], barrier_bonus);
        }
    }

    if new_step {
        // Update cnt_ema before resetting cnt (cnt = trades in previous step)
        if init_flag != 0 {
            // cnt = trades from the last-traded step; step_gap empty steps since then
            let step_gap = step.saturating_sub(last_step);
            // 1) Real EMA update with stored cnt (from last traded step)
            let mut ema = wmul(cnt_ema as u128, CNT_EMA_DECAY)
                        + cnt as u128 * (SCALE - CNT_EMA_DECAY);
            // 2) Decay for empty steps (step_gap - 1 steps with 0 trades)
            if step_gap > 1 {
                let empty = (step_gap - 1).min(100);
                ema = wmul(ema, pow_wad(CNT_EMA_DECAY, empty));
            }
            cnt_ema = ema as u64;
        }

        if sum_w > 0 && init_flag != 0 {
            let vwap = wdiv(sum_wp, sum_w);
            let el = (step.saturating_sub(last_step)).max(1).min(50) as u128;
            let sig2 = wmul(sigma, sigma);
            let pred_var = wmul(sig2, el * 2 * SCALE);
            let inno = if p_hat > 0 { wdiv(abs_diff(vwap, p_hat), p_hat) } else { 0 };
            let obs_var = if vwap < p_hat && inno > KALMAN_INNO_THRESH {
                KALMAN_OBS_VAR_DOWN
            } else {
                KALMAN_OBS_VAR_UP
            };
            let gain = if pred_var + obs_var > 0 {
                wdiv(pred_var, pred_var + obs_var).min(SCALE)
            } else { ALPHA };
            let gain = gain.max(ALPHA).min(900_000_000);
            p_hat = wmul(p_hat, SCALE - gain) + wmul(vwap, gain);
            p_hat_fast = wmul(p_hat_fast, SCALE - ALPHA_FAST) + wmul(vwap, ALPHA_FAST);
        }

        sum_w = 0;
        sum_wp = 0;

        let elapsed = if init_flag == 0 { 1 } else { (step - last_step).min(ELAPSED_CAP) };
        let _fd = if shock_hat > 15_000_000 { FLOOR_DECAY_HI } else { FLOOR_DECAY };
        fee_floor = wmul(fee_floor, pow_wad(_fd, elapsed));
        cnt = 0;
        step_vol = 0;
    }

    let spot = if rx > 0 { wdiv(ry, rx) } else { p_hat };
    if p_hat == 0 { p_hat = spot; }

    // shock_hat update
    let delta_spot = if last_spot > 0 {
        wdiv(abs_diff(spot, last_spot), last_spot)
    } else {
        0
    };
    shock_hat = wmul(shock_hat, SHOCK_DECAY) + wmul(delta_spot, SCALE - SHOCK_DECAY);
    if shock_hat > SCALE / 2 { shock_hat = SCALE / 2; }

    // Fee-adjusted implied price
    let fee_used = rd64(storage, S_FEE) as u128;
    let gamma = if fee_used < SCALE { SCALE - fee_used } else { 0 };
    let p_impl = if gamma == 0 {
        spot
    } else if side == 0 {
        wmul(spot, gamma)
    } else {
        wdiv(spot, gamma)
    };

    let ret = if p_hat_fast > 0 { wdiv(abs_diff(p_impl, p_hat_fast), p_hat_fast) } else { 0 };
    let mut gate = wmul(sigma, GATE_K);
    if gate < GATE_FLOOR { gate = GATE_FLOOR; }

    if ret <= gate {
        let reserve_in = if side == 0 { ry } else { rx };
        let weight = if reserve_in > 0 {
            wdiv(input, reserve_in).min(SCALE)
        } else {
            0
        };
        sum_w = sum_w.saturating_add(weight);
        sum_wp = sum_wp.saturating_add(wmul(weight, p_impl));
    }

    let first = if new_step { true } else { cnt == 0 };
    if first {
        let ret_c = if ret > VOL_CAP { VOL_CAP } else { ret };
        let vd = if step < EARLY_STEP_THRESH { VOL_DECAY_FAST } else { VOL_DECAY };
        sigma = wmul(sigma, vd) + wmul(ret_c, SCALE - vd);
    }

    let tox = if p_hat_fast > 0 { wdiv(abs_diff(spot, p_hat_fast), p_hat_fast) } else { 0 };
    tox_hat = wmul(tox_hat, TOX_DECAY) + wmul(tox, SCALE - TOX_DECAY);
    if tox_hat > SCALE / 2 { tox_hat = SCALE / 2; }

    // Update step-level accumulators
    step_vol = step_vol.saturating_add(input);

    cnt += 1;
    if cnt > STEP_CAP { cnt = STEP_CAP; }

    // Compute per-profile target fees
    let reserve_in = if side == 0 { ry } else { rx };
    let trade_frac = if reserve_in > 0 { wdiv(input, reserve_in) } else { 0 };
    let size_fee = wmul(SIZE_FEE_K, trade_frac);
    let tox2 = wmul(tox, tox);
    let sh2 = wmul(shock_hat, shock_hat);
    let sh3 = wmul(shock_hat, sh2);
    let _sv_frac = if reserve_in > 0 { wdiv(step_vol, reserve_in) } else { 0 };
    let tox_shock_spike = shock_hat > 15_000_000 && tox > 15_000_000;
    let sv_bump = if _sv_frac > 300_000_000 { 10 * BPS }
                  else if _sv_frac > 150_000_000 { 5 * BPS }
                  else { 0 };

    // Helper closure: compute target_fee for given per-profile constants
    let make_target = |base: u128, vm: u128, tq: u128, tc: u128, sq: u128, sc: u128| -> u128 {
        let vol_fee   = wmul(vm, sigma);
        let tox_fee   = wmul(tq, tox2) + wmul(tc, wmul(tox, tox2));
        let shock_fee = wmul(sq, sh2)  + wmul(sc, sh3);
        let mut tf = base + vol_fee + tox_fee + size_fee + shock_fee;
        if tox_shock_spike { tf += 15 * BPS; }
        tf + sv_bump
    };

    // All three profiles share the same target fee formula (they differ only in quote_profile behavior)
    let target = make_target(P0_BASE_FEE, P0_VOL_MULT, P0_TOX_QUAD, P0_TOX_CUBE, P0_SHOCK_QUAD, P0_SHOCK_CUBE);

    if target > fee_floor { fee_floor = target; }

    let new_fee = clamp_fee(fee_floor) as u64;

    // Write all state
    wr64(storage, S_FEE,   new_fee);
    wr64(storage, S_FEE_1, new_fee);
    wr64(storage, S_FEE_2, new_fee);
    wr64(storage, S_STEP, step);
    wr64(storage, S_SIGMA, sigma as u64);
    wr64(storage, S_PHAT, p_hat as u64);
    wr64(storage, S_FLOOR, fee_floor as u64);
    wr64(storage, S_CNT, cnt);
    wr64(storage, S_INIT, 1);
    wr64(storage, S_TOX, tox_hat as u64);
    wr64(storage, S_SUM_W, to_u64_cap(sum_w));
    wr64(storage, S_SUM_WP, to_u64_cap(sum_wp));
    wr64(storage, S_LAST_SPOT, spot as u64);
    wr64(storage, S_SHOCK, shock_hat as u64);
    wr64(storage, S_PHAT_FAST, p_hat_fast as u64);
    wr64(storage, S_MODE, mode as u64);
    // Write scores
    {
        let mut p = 0usize;
        while p < NUM_PROFILES {
            wr_i64(storage, S_SCORES + p * 8, scores[p]);
            p += 1;
        }
    }
    wr64(storage, S_LAST_SWITCH, last_switch);
    wr64(storage, S_CNT_EMA, cnt_ema);
    wr64(storage, S_STEP_VOL, to_u64_cap(step_vol));
}
