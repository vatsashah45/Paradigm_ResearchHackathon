use prop_amm_executor::{AfterSwapFn, BpfProgram, SwapFn};
use prop_amm_shared::config::SimulationConfig;
use prop_amm_shared::result::SimResult;

use crate::amm::BpfAmm;
use crate::arbitrageur::Arbitrageur;
use crate::price_process::GBMPriceProcess;
use crate::retail::RetailTrader;
use crate::router::OrderRouter;

/// Read a u64 from storage at the given byte offset (little-endian).
#[inline]
fn rd_storage(s: &[u8], off: usize) -> u64 {
    if off + 8 <= s.len() {
        u64::from_le_bytes(s[off..off + 8].try_into().unwrap_or([0; 8]))
    } else {
        0
    }
}

/// Read an i64 from storage at the given byte offset (little-endian).
#[inline]
fn rd_storage_i64(s: &[u8], off: usize) -> i64 {
    if off + 8 <= s.len() {
        i64::from_le_bytes(s[off..off + 8].try_into().unwrap_or([0; 8]))
    } else {
        0
    }
}

fn run_sim_inner(
    mut amm_sub: BpfAmm,
    mut amm_norm: BpfAmm,
    config: &SimulationConfig,
) -> anyhow::Result<SimResult> {
    let mut price = GBMPriceProcess::new(
        config.initial_price,
        config.gbm_mu,
        config.gbm_sigma,
        config.gbm_dt,
        config.seed,
    );
    let mut retail = RetailTrader::new(
        config.retail_arrival_rate,
        config.retail_mean_size,
        config.retail_size_sigma,
        config.retail_buy_prob,
        config.seed.wrapping_add(1),
    );
    let mut arb = Arbitrageur::new(
        config.min_arb_profit,
        config.retail_mean_size,
        config.retail_size_sigma,
        config.seed.wrapping_add(2),
    );
    let router = OrderRouter::new();

    let mut submission_edge = 0.0_f64;
    let mut arb_edge = 0.0_f64;
    let mut retail_edge = 0.0_f64;
    let mut retail_trade_count = 0u64;
    let mut arb_trade_count = 0u64;
    let mut retail_edge_buckets = [0u64; 8];
    let mut retail_edge_bucket_sums = [0.0f64; 8];
    let mut small_edge_buckets = [0u64; 5];
    let mut small_edge_sums = [0.0f64; 5];
    let mut small_loss_arb_dir_count = 0u64;
    let mut small_loss_counter_count = 0u64;
    let mut small_loss_size_sum = 0.0f64;
    let mut small_loss_count = 0u64;

    // Per-swap CSV logging (enabled by SWAP_CSV env var)
    let swap_csv_enabled = std::env::var("SWAP_CSV").is_ok();
    let mut csv_rows: Vec<String> = Vec::new();
    let mut prev_fair_price = config.initial_price;

    for step in 0..config.n_steps {
        amm_sub.set_current_step(step as u64);
        amm_norm.set_current_step(step as u64);
        let fair_price = price.step();

        if let Some(result) = arb.execute_arb(&mut amm_sub, fair_price) {
            submission_edge += result.edge;
            arb_edge += result.edge;
            arb_trade_count += 1;
        }
        arb.execute_arb(&mut amm_norm, fair_price);

        let orders = retail.generate_orders();
        for order in &orders {
            let trades = router.route_order(order, &mut amm_sub, &mut amm_norm, fair_price);
            for trade in trades {
                if trade.is_submission {
                    let trade_edge = if trade.amm_buys_x {
                        trade.amount_x * fair_price - trade.amount_y
                    } else {
                        trade.amount_y - trade.amount_x * fair_price
                    };
                    submission_edge += trade_edge;
                    retail_edge += trade_edge;
                    retail_trade_count += 1;
                    // Bucket: [<-1, -1..-0.1, -0.1..0, 0..0.01, 0.01..0.1, 0.1..1, 1..10, >=10]
                    let bi = if trade_edge < -1.0 { 0 }
                        else if trade_edge < -0.1 { 1 }
                        else if trade_edge < 0.0 { 2 }
                        else if trade_edge < 0.01 { 3 }
                        else if trade_edge < 0.1 { 4 }
                        else if trade_edge < 1.0 { 5 }
                        else if trade_edge < 10.0 { 6 }
                        else { 7 };
                    retail_edge_buckets[bi] += 1;
                    retail_edge_bucket_sums[bi] += trade_edge;

                    // Fine-grained small-edge tracking
                    if trade_edge >= -0.1 && trade_edge < 0.01 {
                        let si = if trade_edge < -0.01 { 0 }
                            else if trade_edge < 0.0 { 1 }
                            else if trade_edge < 0.001 { 2 }
                            else if trade_edge < 0.005 { 3 }
                            else { 4 };
                        small_edge_buckets[si] += 1;
                        small_edge_sums[si] += trade_edge;
                    }

                    // Track characteristics of small-loss trades (edge < 0.01)
                    if trade_edge < 0.01 {
                        small_loss_count += 1;
                        // Compute input size in token terms
                        let input_size = if trade.amm_buys_x {
                            trade.amount_y // Y tokens sent to AMM
                        } else {
                            trade.amount_x * fair_price // X tokens, converted to Y equiv
                        };
                        small_loss_size_sum += input_size;
                        // Determine if arb-direction: spot vs fair_price
                        let spot = amm_sub.spot_price();
                        let is_arb_dir = (spot > fair_price && trade.amm_buys_x)
                            || (spot < fair_price && !trade.amm_buys_x);
                        if is_arb_dir {
                            small_loss_arb_dir_count += 1;
                        } else {
                            small_loss_counter_count += 1;
                        }
                    }

                    // Per-swap CSV row
                    if swap_csv_enabled {
                        let st = amm_sub.storage();
                        let stored_fee  = rd_storage(st, 0) as f64 / 1e9;
                        let sigma       = rd_storage(st, 16) as f64 / 1e9;
                        let p_hat       = rd_storage(st, 24) as f64 / 1e9;
                        let fee_floor   = rd_storage(st, 32) as f64 / 1e9;
                        let tox_hat     = rd_storage(st, 56) as f64 / 1e9;
                        let shock_hat   = rd_storage(st, 88) as f64 / 1e9;
                        let p_hat_fast  = rd_storage(st, 96) as f64 / 1e9;
                        let mode        = rd_storage(st, 104);
                        let cnt_ema     = rd_storage(st, 144) as f64;
                        let cnt         = rd_storage(st, 40);

                        let spot_after = amm_sub.spot_price();
                        let side: u8 = if trade.amm_buys_x { 0 } else { 1 };
                        let input_val = if trade.amm_buys_x { trade.amount_y } else { trade.amount_x };
                        let output_val = if trade.amm_buys_x { trade.amount_x } else { trade.amount_y };
                        let reserve_x = amm_sub.reserve_x;
                        let reserve_y = amm_sub.reserve_y;
                        let trade_frac = input_val / (input_val + if side == 0 { reserve_y } else { reserve_x });
                        let deviation = if fair_price > 0.0 { (spot_after - fair_price).abs() / fair_price } else { 0.0 };

                        csv_rows.push(format!(
                            "{},{},{:.6},{:.6},{},{:.6},{:.6},{:.8},{:.6},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{},{:.0},{},{:.6},{:.6},{:.6},{:.6}",
                            config.seed, step, fair_price, spot_after, side,
                            input_val, output_val, trade_edge, deviation,
                            stored_fee, sigma, p_hat, tox_hat, shock_hat, p_hat_fast,
                            mode, cnt_ema, cnt,
                            trade_frac, reserve_x, reserve_y, prev_fair_price
                        ));
                    }
                }
            }
        }
        prev_fair_price = fair_price;
    }

    // Write per-swap CSV if enabled
    if swap_csv_enabled && !csv_rows.is_empty() {
        use std::io::Write;
        let path = format!("swap_log_{}.csv", config.seed);
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
            // Write header only if file is new/empty
            if f.metadata().map(|m| m.len() == 0).unwrap_or(true) {
                let _ = writeln!(f, "seed,step,fair_price,spot,side,input,output,edge,deviation,stored_fee,sigma,p_hat,tox_hat,shock_hat,p_hat_fast,mode,cnt_ema,cnt,trade_frac,reserve_x,reserve_y,prev_fair_price");
            }
            for row in &csv_rows {
                let _ = writeln!(f, "{}", row);
            }
        }
    }

    Ok(SimResult {
        seed: config.seed,
        submission_edge,
        mode_counts: amm_sub.mode_counts,
        mode_switches: amm_sub.mode_switches,
        gbm_sigma: config.gbm_sigma,
        norm_liquidity_mult: config.norm_liquidity_mult,
        retail_arrival_rate: config.retail_arrival_rate,
        retail_mean_size: config.retail_mean_size,
        norm_fee_bps: config.norm_fee_bps,
        arb_edge,
        retail_edge,
        retail_trade_count,
        arb_trade_count,
        retail_edge_buckets,
        retail_edge_bucket_sums,
        small_edge_buckets,
        small_edge_sums,
        small_loss_arb_dir_count,
        small_loss_counter_count,
        small_loss_size_sum,
        small_loss_count,
    })
}

/// Run simulation with BPF programs (slow, for validation)
pub fn run_simulation(
    submission_program: BpfProgram,
    normalizer_program: BpfProgram,
    config: &SimulationConfig,
) -> anyhow::Result<SimResult> {
    let amm_sub = BpfAmm::new(
        submission_program,
        config.initial_x,
        config.initial_y,
        "submission".to_string(),
    );
    let norm_x = config.initial_x * config.norm_liquidity_mult;
    let norm_y = config.initial_y * config.norm_liquidity_mult;
    let mut amm_norm = BpfAmm::new(
        normalizer_program,
        norm_x,
        norm_y,
        "normalizer".to_string(),
    );
    amm_norm.set_initial_storage(&config.norm_fee_bps.to_le_bytes());
    run_sim_inner(amm_sub, amm_norm, config)
}

/// Run simulation with native swap functions (fast, for production)
pub fn run_simulation_native(
    submission_fn: SwapFn,
    submission_after_swap: Option<AfterSwapFn>,
    normalizer_fn: SwapFn,
    normalizer_after_swap: Option<AfterSwapFn>,
    config: &SimulationConfig,
) -> anyhow::Result<SimResult> {
    let amm_sub = BpfAmm::new_native(
        submission_fn,
        submission_after_swap,
        config.initial_x,
        config.initial_y,
        "submission".to_string(),
    );
    let norm_x = config.initial_x * config.norm_liquidity_mult;
    let norm_y = config.initial_y * config.norm_liquidity_mult;
    let mut amm_norm = BpfAmm::new_native(
        normalizer_fn,
        normalizer_after_swap,
        norm_x,
        norm_y,
        "normalizer".to_string(),
    );
    amm_norm.set_initial_storage(&config.norm_fee_bps.to_le_bytes());
    run_sim_inner(amm_sub, amm_norm, config)
}

/// Run simulation with BPF submission + native normalizer (mixed mode)
pub fn run_simulation_mixed(
    submission_program: BpfProgram,
    normalizer_fn: SwapFn,
    normalizer_after_swap: Option<AfterSwapFn>,
    config: &SimulationConfig,
) -> anyhow::Result<SimResult> {
    let amm_sub = BpfAmm::new(
        submission_program,
        config.initial_x,
        config.initial_y,
        "submission".to_string(),
    );
    let norm_x = config.initial_x * config.norm_liquidity_mult;
    let norm_y = config.initial_y * config.norm_liquidity_mult;
    let mut amm_norm = BpfAmm::new_native(
        normalizer_fn,
        normalizer_after_swap,
        norm_x,
        norm_y,
        "normalizer".to_string(),
    );
    amm_norm.set_initial_storage(&config.norm_fee_bps.to_le_bytes());
    run_sim_inner(amm_sub, amm_norm, config)
}
