#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use core::convert::TryInto;
use pinocchio::{account_info::AccountInfo, entrypoint, pubkey::Pubkey, ProgramResult};
use prop_amm_submission_sdk::{set_return_data_bytes, set_return_data_u64, set_storage};

pub const NAME: &str = "VatsaEdge v3";
pub const MODEL_USED: &str = "None";

const SCALE: u128 = 1_000_000_000;
const BPS: u128 = SCALE / 10_000;

// =====================================================================
// Storage layout — matches reference offsets exactly
// =====================================================================
const S_FEE:        usize = 0;    // u64 (1e9): stored fee (clamped)
const S_STEP:       usize = 8;
const S_SIGMA:      usize = 16;
const S_PHAT:       usize = 24;
const S_FLOOR:      usize = 32;   // u64 (1e9): fee floor (pre-clamp)
const S_CNT:        usize = 40;
const S_INIT:       usize = 48;
const S_TOX:        usize = 56;
const S_SUM_W:      usize = 64;
const S_SUM_WP:     usize = 72;
const S_LAST_SPOT:  usize = 80;
const S_SHOCK:      usize = 88;
const S_PHAT_FAST:  usize = 96;
const S_CNT_EMA:    usize = 104;
const S_STEP_VOL:   usize = 112;

// =====================================================================
// Fee parameters — exact P0 profile from reference
// =====================================================================
const FEE_MIN: u128 = 1 * BPS;
const FEE_MAX: u128 = 1000 * BPS;
const DEFAULT_FEE: u128 = 25 * BPS;
const INITIAL_PRICE: u128 = 100 * SCALE;

// Quote-time (compute_swap) — reference P0
const P0_DEFAULT_FEE: u128 = 55 * BPS;
const P0_ARB_K: u128 = 5_200 * BPS;
const P0_COUNTER_K: u128 = 1_000 * BPS;
const P0_SIZE_K: u128 = 2_900 * BPS;
const P0_SHOCK_K: u128 = 733 * BPS;
const P0_TOX_K: u128 = 74 * BPS;
const PRICE_DEV_K: u128 = 400 * BPS;
const REBATE_RISK_MULT: u128 = 1_500_000_000;

// After-swap fee floor formula
const BASE_FEE: u128 = 12 * BPS;
const VOL_MULT: u128 = 600_000_000;
const TOX_QUAD: u128 = 15_000 * BPS;
const TOX_CUBE: u128 = 55_000 * BPS;
const SHOCK_QUAD: u128 = 120_000 * BPS;
const SHOCK_CUBE: u128 = 95_000 * BPS;
const SIZE_FEE_K: u128 = 2_800 * BPS;

// Tracking
const VOL_DECAY: u128 = 998_000_000;
const VOL_DECAY_FAST: u128 = 880_000_000;
const VOL_CAP: u128 = 50_000_000;
const INIT_SIGMA: u128 = 3_000_000;
const FLOOR_DECAY: u128 = 850_000_000;
const FLOOR_DECAY_HI: u128 = 920_000_000;
const SHOCK_DECAY: u128 = 350_000_000;
const TOX_DECAY: u128 = 650_000_000;
const ALPHA_FAST: u128 = 650_000_000;
const ALPHA_SLOW: u128 = 20_000_000;
const GATE_K: u128 = 4 * SCALE;
const GATE_FLOOR: u128 = 60_000_000;
const KALMAN_OBS_VAR_DOWN: u128 = 20_000_000;
const KALMAN_OBS_VAR_UP: u128 = 50_000_000;
const KALMAN_INNO_THRESH: u128 = 10_000_000;
const EARLY_STEP_THRESH: u64 = 30;
const STEP_CAP: u64 = 16;
const ELAPSED_CAP: u64 = 3;
const CNT_EMA_DECAY: u128 = 950_000_000;

// =====================================================================
pub fn get_model_used() -> &'static str { MODEL_USED }

entrypoint!(process_instruction);

pub fn process_instruction(
    _program_id: &Pubkey,
    _accounts: &[AccountInfo],
    instruction_data: &[u8],
) -> ProgramResult {
    if instruction_data.is_empty() { return Ok(()); }
    match instruction_data[0] {
        0 | 1 => set_return_data_u64(compute_swap(instruction_data)),
        2 => handle_after_swap(instruction_data),
        3 => set_return_data_bytes(NAME.as_bytes()),
        4 => set_return_data_bytes(MODEL_USED.as_bytes()),
        _ => {}
    }
    Ok(())
}

// =====================================================================
// Helpers
// =====================================================================
#[inline]
fn rd64(d: &[u8], o: usize) -> u64 { u64::from_le_bytes(d[o..o+8].try_into().unwrap()) }
#[inline]
fn wr64(d: &mut [u8], o: usize, v: u64) { d[o..o+8].copy_from_slice(&v.to_le_bytes()); }
#[inline]
fn abs_diff(a: u128, b: u128) -> u128 { if a >= b { a - b } else { b - a } }
#[inline]
fn wmul(a: u128, b: u128) -> u128 { (a * b) / SCALE }
#[inline]
fn wdiv(a: u128, b: u128) -> u128 { if b == 0 { 0 } else { (a * SCALE) / b } }
#[inline]
fn floor_mul_div(a: u128, b: u128, c: u128) -> u128 {
    if c == 0 { return 0; }
    let q = a / c; let r = a % c;
    q * b + r * b / c
}
#[inline]
fn clamp_fee(f: u128) -> u128 { if f < FEE_MIN { FEE_MIN } else if f > FEE_MAX { FEE_MAX } else { f } }
fn pow_wad(mut base: u128, mut exp: u64) -> u128 {
    let mut r = SCALE;
    while exp > 0 {
        if exp & 1 == 1 { r = wmul(r, base); }
        base = wmul(base, base);
        exp >>= 1;
    }
    r
}
#[inline]
fn to_u64_cap(x: u128) -> u64 { if x > u64::MAX as u128 { u64::MAX } else { x as u64 } }

// =====================================================================
// compute_swap — exact P0 quote logic
// =====================================================================

#[inline(never)]
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
    let stored_fee = if init_flag == 0 { DEFAULT_FEE } else { rd64(storage, S_FEE) as u128 };
    let tox_hat = rd64(storage, S_TOX) as u128;
    let shock_hat = rd64(storage, S_SHOCK) as u128;

    let fee = if init_flag == 0 {
        let deviation = wdiv(abs_diff(spot, INITIAL_PRICE), INITIAL_PRICE);
        clamp_fee(P0_DEFAULT_FEE + wmul(PRICE_DEV_K, deviation))
    } else {
        let deviation = if p_ref > 0 { wdiv(abs_diff(spot, p_ref), p_ref) } else { 0 };
        let dev2 = wmul(deviation, deviation);

        let mut f = stored_fee + wmul(P0_TOX_K, tox_hat) + wmul(P0_SHOCK_K, shock_hat);

        if is_arb_dir {
            f = f.saturating_add(wmul(P0_ARB_K, dev2));
        } else {
            let risk = if tox_hat > shock_hat { tox_hat } else { shock_hat };
            let mut scale = SCALE.saturating_sub(wmul(REBATE_RISK_MULT, risk));
            if scale > SCALE { scale = SCALE; }
            let shock_risk = wmul(shock_hat, 5 * SCALE);
            let counter_scale = SCALE.saturating_sub(shock_risk).min(SCALE);
            let adapt_counter_k = wmul(P0_COUNTER_K, counter_scale);
            let raw_rebate = wmul(adapt_counter_k, dev2);
            f = f.saturating_sub(wmul(raw_rebate, scale));
        }
        clamp_fee(f)
    };

    let reserve_in = if side == 0 { ry } else { rx };
    let sum = input + reserve_in;
    let den_base = reserve_in * SCALE + input * (SCALE - fee);
    let ski = P0_SIZE_K * input;
    let size_penalty = match ski.checked_mul(input) {
        Some(num) => num / sum,
        None => (ski / sum) * input,
    };
    let den = den_base.saturating_sub(size_penalty);
    if den == 0 { return 0; }

    let k = rx * ry;
    let out = if side == 0 {
        rx.saturating_sub(floor_mul_div(k, SCALE, den))
    } else {
        ry.saturating_sub(floor_mul_div(k, SCALE, den))
    };
    if out > u64::MAX as u128 { u64::MAX } else { out as u64 }
}

// =====================================================================
// after_swap
// =====================================================================

#[inline(never)]
fn handle_after_swap(instruction_data: &[u8]) {
    if instruction_data.len() < 42 + 1024 { return; }
    let mut storage = [0u8; 1024];
    storage.copy_from_slice(&instruction_data[42..42 + 1024]);
    after_swap(instruction_data, &mut storage);
    let _ = set_storage(&storage);
}

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
    let mut cnt_ema = rd64(storage, S_CNT_EMA);
    let mut step_vol = rd64(storage, S_STEP_VOL) as u128;

    if init_flag == 0 {
        p_hat = if rx > 0 { wdiv(ry, rx) } else { INITIAL_PRICE };
        sigma = INIT_SIGMA;
        fee_floor = DEFAULT_FEE;
        tox_hat = 0; cnt = 0; sum_w = 0; sum_wp = 0;
        last_spot = p_hat; shock_hat = 0; p_hat_fast = p_hat;
        cnt_ema = 0; step_vol = 0;
        wr64(storage, S_FEE, DEFAULT_FEE as u64);
    }

    let new_step = step > last_step || init_flag == 0;

    if new_step {
        if init_flag != 0 {
            let step_gap = step.saturating_sub(last_step);
            let mut ema = wmul(cnt_ema as u128, CNT_EMA_DECAY)
                + cnt as u128 * (SCALE - CNT_EMA_DECAY);
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
            } else { ALPHA_SLOW };
            let gain = gain.max(ALPHA_SLOW).min(900_000_000);
            p_hat = wmul(p_hat, SCALE - gain) + wmul(vwap, gain);
            p_hat_fast = wmul(p_hat_fast, SCALE - ALPHA_FAST) + wmul(vwap, ALPHA_FAST);
        }

        sum_w = 0;
        sum_wp = 0;

        let elapsed = if init_flag == 0 { 1 } else { (step - last_step).min(ELAPSED_CAP) };
        let fd = if shock_hat > 15_000_000 { FLOOR_DECAY_HI } else { FLOOR_DECAY };
        fee_floor = wmul(fee_floor, pow_wad(fd, elapsed));
        cnt = 0;
        step_vol = 0;
    }

    let spot = if rx > 0 { wdiv(ry, rx) } else { p_hat };
    if p_hat == 0 { p_hat = spot; }

    // Shock
    let delta_spot = if last_spot > 0 { wdiv(abs_diff(spot, last_spot), last_spot) } else { 0 };
    shock_hat = wmul(shock_hat, SHOCK_DECAY) + wmul(delta_spot, SCALE - SHOCK_DECAY);
    if shock_hat > SCALE / 2 { shock_hat = SCALE / 2; }

    // VWAP: use STORED fee (not decayed fee_floor) for p_impl — matches reference
    let fee_used = rd64(storage, S_FEE) as u128;
    let gamma = if fee_used < SCALE { SCALE - fee_used } else { 0 };
    let p_impl = if gamma == 0 { spot }
        else if side == 0 { wmul(spot, gamma) }
        else { wdiv(spot, gamma) };

    let ret = if p_hat_fast > 0 { wdiv(abs_diff(p_impl, p_hat_fast), p_hat_fast) } else { 0 };
    let mut gate = wmul(sigma, GATE_K);
    if gate < GATE_FLOOR { gate = GATE_FLOOR; }
    if ret <= gate {
        let reserve_in = if side == 0 { ry } else { rx };
        let weight = if reserve_in > 0 { wdiv(input, reserve_in).min(SCALE) } else { 0 };
        sum_w = sum_w.saturating_add(weight);
        sum_wp = sum_wp.saturating_add(wmul(weight, p_impl));
    }

    // Sigma
    let first = new_step || cnt == 0;
    if first {
        let ret_c = if ret > VOL_CAP { VOL_CAP } else { ret };
        let vd = if step < EARLY_STEP_THRESH { VOL_DECAY_FAST } else { VOL_DECAY };
        sigma = wmul(sigma, vd) + wmul(ret_c, SCALE - vd);
    }

    // Tox
    let tox = if p_hat_fast > 0 { wdiv(abs_diff(spot, p_hat_fast), p_hat_fast) } else { 0 };
    tox_hat = wmul(tox_hat, TOX_DECAY) + wmul(tox, SCALE - TOX_DECAY);
    if tox_hat > SCALE / 2 { tox_hat = SCALE / 2; }

    step_vol = step_vol.saturating_add(input);
    cnt += 1;
    if cnt > STEP_CAP { cnt = STEP_CAP; }

    // Target fee
    let reserve_in = if side == 0 { ry } else { rx };
    let trade_frac = if reserve_in > 0 { wdiv(input, reserve_in) } else { 0 };
    let size_fee = wmul(SIZE_FEE_K, trade_frac);
    let tox2 = wmul(tox, tox);
    let sh2 = wmul(shock_hat, shock_hat);
    let sh3 = wmul(shock_hat, sh2);
    let sv_frac = if reserve_in > 0 { wdiv(step_vol, reserve_in) } else { 0 };
    let tox_shock_spike = shock_hat > 15_000_000 && tox > 15_000_000;
    let sv_bump = if sv_frac > 300_000_000 { 10 * BPS }
                  else if sv_frac > 150_000_000 { 5 * BPS }
                  else { 0 };

    let vol_fee = wmul(VOL_MULT, sigma);
    let tox_fee = wmul(TOX_QUAD, tox2) + wmul(TOX_CUBE, wmul(tox, tox2));
    let shock_fee = wmul(SHOCK_QUAD, sh2) + wmul(SHOCK_CUBE, sh3);
    let mut target = BASE_FEE + vol_fee + tox_fee + size_fee + shock_fee;
    if tox_shock_spike { target += 15 * BPS; }
    target += sv_bump;

    if target > fee_floor { fee_floor = target; }
    let new_fee = clamp_fee(fee_floor) as u64;

    // Write
    wr64(storage, S_FEE, new_fee);
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
    wr64(storage, S_CNT_EMA, cnt_ema);
    wr64(storage, S_STEP_VOL, to_u64_cap(step_vol));
}
