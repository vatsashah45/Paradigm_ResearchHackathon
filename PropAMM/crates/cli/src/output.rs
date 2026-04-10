use prop_amm_shared::result::BatchResult;
use std::time::Duration;

pub struct RunTimings {
    pub compile_or_load: Duration,
    pub simulation: Duration,
    pub total: Duration,
}

pub fn print_results(result: &BatchResult, timings: RunTimings) {
    let seed_range = result
        .results
        .iter()
        .map(|r| r.seed)
        .fold(None::<(u64, u64)>, |acc, seed| match acc {
            Some((lo, hi)) => Some((lo.min(seed), hi.max(seed))),
            None => Some((seed, seed)),
        });

    println!("\n========================================");
    println!("  Simulations: {}", result.n_sims());
    if let Some((seed_start, seed_end)) = seed_range {
        println!("  Seed range:  {}..={}", seed_start, seed_end);
    }
    println!(
        "  Compile/load:{:>8.2}s",
        timings.compile_or_load.as_secs_f64()
    );
    println!("  Simulation:  {:>8.2}s", timings.simulation.as_secs_f64());
    println!("  Total:       {:>8.2}s", timings.total.as_secs_f64());
    println!("  Avg edge:    {:.2}", result.avg_edge());
    println!("  Total edge:  {:.2}", result.total_edge);
    println!("========================================");

    // --- PER-TRADE EDGE ANALYSIS ---
    {
        let total_arb_edge: f64 = result.results.iter().map(|r| r.arb_edge).sum();
        let total_retail_edge: f64 = result.results.iter().map(|r| r.retail_edge).sum();
        let total_arb_trades: u64 = result.results.iter().map(|r| r.arb_trade_count).sum();
        let total_retail_trades: u64 = result.results.iter().map(|r| r.retail_trade_count).sum();

        let mut agg_buckets = [0u64; 8];
        let mut agg_bucket_sums = [0.0f64; 8];
        for r in &result.results {
            for i in 0..8 {
                agg_buckets[i] += r.retail_edge_buckets[i];
                agg_bucket_sums[i] += r.retail_edge_bucket_sums[i];
            }
        }

        println!("\n--- TRADE-LEVEL EDGE BREAKDOWN ---");
        println!("  Arb trades:    {:>10} total, edge={:>12.2} (avg {:>+.4}/trade)",
            total_arb_trades, total_arb_edge,
            if total_arb_trades > 0 { total_arb_edge / total_arb_trades as f64 } else { 0.0 });
        println!("  Retail trades: {:>10} total, edge={:>12.2} (avg {:>+.4}/trade)",
            total_retail_trades, total_retail_edge,
            if total_retail_trades > 0 { total_retail_edge / total_retail_trades as f64 } else { 0.0 });
        println!("  Avg retail trades/sim: {:.1}", total_retail_trades as f64 / result.n_sims() as f64);
        println!("  Avg arb trades/sim:    {:.1}", total_arb_trades as f64 / result.n_sims() as f64);

        let bucket_labels = ["<-1", "-1..-0.1", "-0.1..0", "0..0.01", "0.01..0.1", "0.1..1", "1..10", ">=10"];
        let total_trades: u64 = agg_buckets.iter().sum();
        println!("\n  Retail trade edge distribution:");
        println!("  {:>12}  {:>10}  {:>6}  {:>12}  {:>8}",
            "bucket", "count", "%", "sum_edge", "avg_edge");
        for i in 0..8 {
            let pct = if total_trades > 0 { agg_buckets[i] as f64 / total_trades as f64 * 100.0 } else { 0.0 };
            let avg = if agg_buckets[i] > 0 { agg_bucket_sums[i] / agg_buckets[i] as f64 } else { 0.0 };
            println!("  {:>12}  {:>10}  {:>5.1}%  {:>12.2}  {:>+8.4}",
                bucket_labels[i], agg_buckets[i], pct, agg_bucket_sums[i], avg);
        }

        // --- SMALL-LOSS TRADE ANALYSIS ---
        {
            let mut agg_small = [0u64; 5];
            let mut agg_small_sums = [0.0f64; 5];
            let mut total_sl_count = 0u64;
            let mut total_sl_arb = 0u64;
            let mut total_sl_counter = 0u64;
            let mut total_sl_size = 0.0f64;
            for r in &result.results {
                for i in 0..5 {
                    agg_small[i] += r.small_edge_buckets[i];
                    agg_small_sums[i] += r.small_edge_sums[i];
                }
                total_sl_count += r.small_loss_count;
                total_sl_arb += r.small_loss_arb_dir_count;
                total_sl_counter += r.small_loss_counter_count;
                total_sl_size += r.small_loss_size_sum;
            }
            let total_rtl = total_retail_trades;
            println!("\n  --- SMALL-LOSS TRADES (edge < 0.01) ---");
            println!("  Total: {} / {} retail trades ({:.1}%)",
                total_sl_count, total_rtl,
                if total_rtl > 0 { total_sl_count as f64 / total_rtl as f64 * 100.0 } else { 0.0 });
            println!("  Arb-direction:     {} ({:.1}%)", total_sl_arb,
                if total_sl_count > 0 { total_sl_arb as f64 / total_sl_count as f64 * 100.0 } else { 0.0 });
            println!("  Counter-direction: {} ({:.1}%)", total_sl_counter,
                if total_sl_count > 0 { total_sl_counter as f64 / total_sl_count as f64 * 100.0 } else { 0.0 });
            println!("  Avg input size:    {:.2}",
                if total_sl_count > 0 { total_sl_size / total_sl_count as f64 } else { 0.0 });
            let total_sl_edge: f64 = agg_small_sums.iter().sum::<f64>()
                + agg_bucket_sums[0] + agg_bucket_sums[1]; // add <-1 and -1..-0.1 buckets
            println!("  Total edge from small-loss trades: {:.2}", total_sl_edge);
            println!("  Avg edge/trade: {:.4}",
                if total_sl_count > 0 {
                    // Sum of all edge < 0.01 trades
                    let all_sl_edge: f64 = agg_small_sums.iter().sum::<f64>()
                        + agg_bucket_sums[0] + agg_bucket_sums[1];
                    all_sl_edge / total_sl_count as f64
                } else { 0.0 });

            let small_labels = ["<-0.01", "-0.01..0", "0..0.001", "0.001..0.005", "0.005..0.01"];
            println!("\n  Fine-grained edge distribution ([-0.1, 0.01] range):");
            println!("  {:>14}  {:>10}  {:>6}  {:>12}  {:>8}",
                "bucket", "count", "%", "sum_edge", "avg_edge");
            for i in 0..5 {
                let pct = if total_rtl > 0 { agg_small[i] as f64 / total_rtl as f64 * 100.0 } else { 0.0 };
                let avg = if agg_small[i] > 0 { agg_small_sums[i] / agg_small[i] as f64 } else { 0.0 };
                println!("  {:>14}  {:>10}  {:>5.1}%  {:>12.2}  {:>+8.5}",
                    small_labels[i], agg_small[i], pct, agg_small_sums[i], avg);
            }

            // Also show large-loss buckets for context
            println!("\n  Context: large-loss buckets");
            println!("  {:>14}  {:>10}  {:>5.1}%  {:>12.2}  {:>+8.4}",
                "<-1", agg_buckets[0],
                if total_rtl > 0 { agg_buckets[0] as f64 / total_rtl as f64 * 100.0 } else { 0.0 },
                agg_bucket_sums[0],
                if agg_buckets[0] > 0 { agg_bucket_sums[0] / agg_buckets[0] as f64 } else { 0.0 });
            println!("  {:>14}  {:>10}  {:>5.1}%  {:>12.2}  {:>+8.4}",
                "-1..-0.1", agg_buckets[1],
                if total_rtl > 0 { agg_buckets[1] as f64 / total_rtl as f64 * 100.0 } else { 0.0 },
                agg_bucket_sums[1],
                if agg_buckets[1] > 0 { agg_bucket_sums[1] / agg_buckets[1] as f64 } else { 0.0 });

            // Per-regime small-loss breakdown
            let regimes = result.bucket_by_regime();
            if regimes.len() > 1 {
                println!("\n  Small-loss by regime:");
                println!("  {:>10}  {:>8}  {:>8}  {:>8}  {:>10}  {:>8}",
                    "regime", "sl/sim", "sl%", "arb%", "avg_size", "sl_edge");
                for (label, br) in &regimes {
                    let n = br.n_sims() as f64;
                    let sl: u64 = br.results.iter().map(|r| r.small_loss_count).sum();
                    let rtl: u64 = br.results.iter().map(|r| r.retail_trade_count).sum();
                    let arb_d: u64 = br.results.iter().map(|r| r.small_loss_arb_dir_count).sum();
                    let sz: f64 = br.results.iter().map(|r| r.small_loss_size_sum).sum();
                    let sl_edge: f64 = br.results.iter().map(|r| {
                        r.small_edge_sums.iter().sum::<f64>()
                        + r.retail_edge_bucket_sums[0] + r.retail_edge_bucket_sums[1]
                    }).sum();
                    println!("  {:>10}  {:>8.1}  {:>7.1}%  {:>7.1}%  {:>10.2}  {:>8.2}",
                        label, sl as f64 / n,
                        if rtl > 0 { sl as f64 / rtl as f64 * 100.0 } else { 0.0 },
                        if sl > 0 { arb_d as f64 / sl as f64 * 100.0 } else { 0.0 },
                        if sl > 0 { sz / sl as f64 } else { 0.0 },
                        sl_edge / n);
                }
            }
        }

        // Per-regime trade breakdown
        let regimes = result.bucket_by_regime();
        if regimes.len() > 1 {
            println!("\n  Retail trades per regime:");
            println!("  {:>10}  {:>8}  {:>8}  {:>10}  {:>10}  {:>8}  {:>8}",
                "regime", "rtl/sim", "arb/sim", "rtl_edge", "arb_edge", "avg_rtl", "win_rate");
            for (label, br) in &regimes {
                let n = br.n_sims() as f64;
                let rtl: u64 = br.results.iter().map(|r| r.retail_trade_count).sum();
                let arb: u64 = br.results.iter().map(|r| r.arb_trade_count).sum();
                let rtl_e: f64 = br.results.iter().map(|r| r.retail_edge).sum();
                let arb_e: f64 = br.results.iter().map(|r| r.arb_edge).sum();
                let wins: u64 = br.results.iter().map(|r|
                    r.retail_edge_buckets[3] + r.retail_edge_buckets[4] +
                    r.retail_edge_buckets[5] + r.retail_edge_buckets[6] + r.retail_edge_buckets[7]
                ).sum();
                let wr = if rtl > 0 { wins as f64 / rtl as f64 * 100.0 } else { 0.0 };
                println!("  {:>10}  {:>8.1}  {:>8.1}  {:>10.2}  {:>10.2}  {:>+8.4}  {:>7.1}%",
                    label, rtl as f64 / n, arb as f64 / n,
                    rtl_e / n, arb_e / n,
                    if rtl > 0 { rtl_e / rtl as f64 } else { 0.0 }, wr);
            }
        }
    }

    // Aggregate mode usage across all sims
    let mut total_mode_counts = [0u64; 16];
    let mut total_switches = 0u64;
    let mut sims_with_switches = 0u64;
    for r in &result.results {
        for i in 0..16 {
            total_mode_counts[i] += r.mode_counts[i];
        }
        if !r.mode_switches.is_empty() {
            total_switches += r.mode_switches.len() as u64;
            sims_with_switches += 1;
        }
    }
    let total_swaps: u64 = total_mode_counts.iter().sum();
    if total_swaps > 0 {
        println!("\n--- PROFILE USAGE ---");
        for i in 0..16 {
            let count = total_mode_counts[i];
            if count > 0 {
                let pct = count as f64 / total_swaps as f64 * 100.0;
                println!("  Profile {:>2}: {:>10} swaps ({:>5.2}%)", i, count, pct);
            }
        }
        println!("  Total swaps: {}", total_swaps);
        println!("  Sims with switches: {}/{} ({:.1}%)",
            sims_with_switches, result.n_sims(),
            sims_with_switches as f64 / result.n_sims() as f64 * 100.0);
        println!("  Total switches: {} (avg {:.2}/sim)",
            total_switches,
            total_switches as f64 / result.n_sims() as f64);

        // Show first 20 switches from first sim that has them
        if let Some(r) = result.results.iter().find(|r| !r.mode_switches.is_empty()) {
            let n = r.mode_switches.len().min(20);
            println!("  Sample switches (seed={}):", r.seed);
            for &(step, from, to) in &r.mode_switches[..n] {
                println!("    step {:>5}: profile {} → {}", step, from, to);
            }
            if r.mode_switches.len() > 20 {
                println!("    ... and {} more", r.mode_switches.len() - 20);
            }
        }

        // Switch step distribution: when do profiles activate?
        {
            // Collect all switches across all sims
            let mut switch_to_steps: std::collections::HashMap<(u8, u8), Vec<u64>> = std::collections::HashMap::new();
            for r in &result.results {
                for &(step, from, to) in &r.mode_switches {
                    switch_to_steps.entry((from, to)).or_default().push(step);
                }
            }
            if !switch_to_steps.is_empty() {
                println!("\n--- SWITCH TIMING DISTRIBUTION ---");
                let mut keys: Vec<_> = switch_to_steps.keys().cloned().collect();
                keys.sort();
                for (from, to) in keys {
                    let steps = switch_to_steps.get(&(from, to)).unwrap();
                    let n = steps.len();
                    let mut sorted = steps.clone();
                    sorted.sort();
                    let min = sorted[0];
                    let p10 = sorted[n / 10];
                    let p25 = sorted[n / 4];
                    let median = sorted[n / 2];
                    let p75 = sorted[3 * n / 4];
                    let p90 = sorted[9 * n / 10];
                    let max = sorted[n - 1];
                    let mean = sorted.iter().sum::<u64>() as f64 / n as f64;
                    println!(
                        "  P{}→P{}: n={:>5}  mean={:>6.0}  min={:>5}  p10={:>5}  p25={:>5}  med={:>5}  p75={:>5}  p90={:>5}  max={:>5}",
                        from, to, n, mean, min, p10, p25, median, p75, p90, max
                    );
                }
            }

            // Per-profile: avg step of first activation
            println!("\n--- PROFILE FIRST ACTIVATION STEP ---");
            for profile in 0u8..3 {
                let mut first_steps: Vec<u64> = Vec::new();
                let mut never_count = 0u64;
                for r in &result.results {
                    if let Some(&(step, _, _)) = r.mode_switches.iter().find(|&&(_, _, to)| to == profile) {
                        first_steps.push(step);
                    } else if r.mode_counts[profile as usize] == 0 {
                        never_count += 1;
                    }
                }
                if !first_steps.is_empty() {
                    first_steps.sort();
                    let n = first_steps.len();
                    let mean = first_steps.iter().sum::<u64>() as f64 / n as f64;
                    let median = first_steps[n / 2];
                    println!(
                        "  P{}: activated in {:>4} sims ({:>5.1}%), never in {:>4} | first step: mean={:>6.0}  med={:>5}  min={:>5}  max={:>5}",
                        profile, n, n as f64 / result.n_sims() as f64 * 100.0, never_count,
                        mean, median, first_steps[0], first_steps[n - 1]
                    );
                } else {
                    println!("  P{}: never activated (via switch), never_count={}", profile, never_count);
                }
            }

            // Per-profile: average duration (steps spent in profile)
            println!("\n--- PROFILE AVG DURATION (steps per activation) ---");
            for profile in 0u8..3 {
                let total_swaps = total_mode_counts[profile as usize];
                // Count how many times we switched INTO this profile
                let mut activations = 0u64;
                for r in &result.results {
                    // If profile is the starting profile (first swap goes here), count that
                    if r.mode_counts[profile as usize] > 0 {
                        // Count switches TO this profile
                        let switches_to: u64 = r.mode_switches.iter().filter(|&&(_, _, to)| to == profile).count() as u64;
                        // If profile has swaps but no switch to it, it was the initial profile
                        if switches_to == 0 {
                            activations += 1;
                        } else {
                            activations += switches_to;
                        }
                    }
                }
                if activations > 0 {
                    println!("  P{}: {:>10} swaps / {:>5} activations = {:>7.1} swaps/activation",
                        profile, total_swaps, activations, total_swaps as f64 / activations as f64);
                }
            }
        }
    }

    // --- BY REGIME ---
    let regimes = result.bucket_by_regime();
    if regimes.len() > 1 {
        println!("\n--- BY REGIME ---");
        // Find which profiles are actually used
        let mut any_used = [false; 16];
        for r in &result.results {
            for i in 0..16 { if r.mode_counts[i] > 0 { any_used[i] = true; } }
        }
        let used_profiles: Vec<usize> = (0..16).filter(|&i| any_used[i]).collect();
        let mut header = format!("  {:>10}  {:>5}  {:>8}", "regime", "n", "avg_edge");
        for &i in &used_profiles { header.push_str(&format!("  P{:>1}%", i)); }
        header.push_str("  sw/sim");
        println!("{}", header);
        for (label, br) in &regimes {
            let mut mc = [0u64; 16];
            let mut sw = 0u64;
            for r in &br.results {
                for i in 0..16 { mc[i] += r.mode_counts[i]; }
                sw += r.mode_switches.len() as u64;
            }
            let ts: u64 = mc.iter().sum();
            let pct = |i: usize| if ts > 0 { mc[i] as f64 / ts as f64 * 100.0 } else { 0.0 };
            let mut line = format!(
                "  {:>10}  {:>5}  {:>8.2}",
                label, br.n_sims(), br.avg_edge(),
            );
            for &i in &used_profiles { line.push_str(&format!("  {:>4.1}", pct(i))); }
            line.push_str(&format!("  {:>6.2}", sw as f64 / br.n_sims() as f64));
            println!("{}", line);
        }
    }

    // --- PER-PARAMETER QUINTILES ---
    if result.n_sims() >= 20 {
        println!("\n--- EDGE BY PARAMETER QUINTILE ---");
        let mut sorted: Vec<_> = result.results.iter().collect();
        let n = sorted.len();
        let q = n / 5;
        if q > 0 {
            let params: Vec<(&str, Box<dyn Fn(&prop_amm_shared::result::SimResult) -> f64>)> = vec![
                ("sigma", Box::new(|r: &prop_amm_shared::result::SimResult| r.gbm_sigma)),
                ("norm_liq", Box::new(|r: &prop_amm_shared::result::SimResult| r.norm_liquidity_mult)),
                ("arrival", Box::new(|r: &prop_amm_shared::result::SimResult| r.retail_arrival_rate)),
                ("size", Box::new(|r: &prop_amm_shared::result::SimResult| r.retail_mean_size)),
                ("norm_fee", Box::new(|r: &prop_amm_shared::result::SimResult| r.norm_fee_bps as f64)),
            ];
            println!("  {:>10}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}",
                "param", "Q1(lo)", "Q2", "Q3", "Q4", "Q5(hi)", "spread");
            for (name, extractor) in &params {
                sorted.sort_by(|a, b| extractor(a).partial_cmp(&extractor(b)).unwrap());
                let mut q_edges = [0.0f64; 5];
                for qi in 0..5 {
                    let start = qi * q;
                    let end = if qi == 4 { n } else { (qi + 1) * q };
                    let sum: f64 = sorted[start..end].iter().map(|r| r.submission_edge).sum();
                    q_edges[qi] = sum / (end - start) as f64;
                }
                let spread = q_edges[4] - q_edges[0];
                println!(
                    "  {:>10}  {:>8.1}  {:>8.1}  {:>8.1}  {:>8.1}  {:>8.1}  {:>+8.1}",
                    name, q_edges[0], q_edges[1], q_edges[2], q_edges[3], q_edges[4], spread
                );
            }
        }
    }

    // --- LOSING SIMULATION ANALYSIS ---
    {
        let mut sorted_by_edge: Vec<_> = result.results.iter().collect();
        sorted_by_edge.sort_by(|a, b| a.submission_edge.partial_cmp(&b.submission_edge).unwrap());

        let n_negative = sorted_by_edge.iter().filter(|r| r.submission_edge < 0.0).count();
        let n_below_100 = sorted_by_edge.iter().filter(|r| r.submission_edge < 100.0).count();
        let bottom_10pct = result.n_sims() / 10;

        println!("\n--- LOSING SIMULATION ANALYSIS ---");
        println!("  Negative edge sims: {}/{} ({:.1}%)",
            n_negative, result.n_sims(), n_negative as f64 / result.n_sims() as f64 * 100.0);
        println!("  Below 100 edge sims: {}/{} ({:.1}%)",
            n_below_100, result.n_sims(), n_below_100 as f64 / result.n_sims() as f64 * 100.0);

        // Show worst 20 sims with arb/retail breakdown
        let show_n = 20.min(sorted_by_edge.len());
        println!("\n  Worst {} sims:", show_n);
        println!("  {:>6}  {:>8}  {:>8}  {:>8}  {:>7}  {:>7}  {:>7}  {:>7}  {:>5}  {:>8}  {:>6}",
            "seed", "edge", "arb_e", "rtl_e", "sigma", "nliq", "arrive", "size", "nfee", "regime", "P0%");
        for r in &sorted_by_edge[..show_n] {
            let regime = prop_amm_shared::result::regime_label(r);
            let total_swaps: u64 = r.mode_counts.iter().sum();
            let p0_pct = if total_swaps > 0 { r.mode_counts[0] as f64 / total_swaps as f64 * 100.0 } else { 0.0 };
            println!("  {:>6}  {:>8.1}  {:>8.1}  {:>8.1}  {:>7.4}  {:>7.2}  {:>7.2}  {:>7.1}  {:>5}  {:>8}  {:>5.1}",
                r.seed, r.submission_edge, r.arb_edge, r.retail_edge, r.gbm_sigma, r.norm_liquidity_mult,
                r.retail_arrival_rate, r.retail_mean_size, r.norm_fee_bps, regime, p0_pct);
        }

        // Aggregate stats for bottom 10%
        if bottom_10pct > 0 {
            let bottom = &sorted_by_edge[..bottom_10pct];
            let top = &sorted_by_edge[sorted_by_edge.len() - bottom_10pct..];
            let avg = |sims: &[&prop_amm_shared::result::SimResult], f: &dyn Fn(&prop_amm_shared::result::SimResult) -> f64| -> f64 {
                sims.iter().map(|r| f(r)).sum::<f64>() / sims.len() as f64
            };
            println!("\n  Bottom 10% vs Top 10% parameter averages:");
            println!("  {:>10}  {:>10}  {:>10}", "param", "bottom10%", "top10%");
            println!("  {:>10}  {:>10.1}  {:>10.1}", "edge", avg(bottom, &|r| r.submission_edge), avg(top, &|r| r.submission_edge));
            println!("  {:>10}  {:>10.5}  {:>10.5}", "sigma", avg(bottom, &|r| r.gbm_sigma), avg(top, &|r| r.gbm_sigma));
            println!("  {:>10}  {:>10.3}  {:>10.3}", "norm_liq", avg(bottom, &|r| r.norm_liquidity_mult), avg(top, &|r| r.norm_liquidity_mult));
            println!("  {:>10}  {:>10.3}  {:>10.3}", "arrival", avg(bottom, &|r| r.retail_arrival_rate), avg(top, &|r| r.retail_arrival_rate));
            println!("  {:>10}  {:>10.2}  {:>10.2}", "size", avg(bottom, &|r| r.retail_mean_size), avg(top, &|r| r.retail_mean_size));
            println!("  {:>10}  {:>10.1}  {:>10.1}", "norm_fee", avg(bottom, &|r| r.norm_fee_bps as f64), avg(top, &|r| r.norm_fee_bps as f64));
            println!("  {:>10}  {:>10.1}  {:>10.1}", "arb_edge", avg(bottom, &|r| r.arb_edge), avg(top, &|r| r.arb_edge));
            println!("  {:>10}  {:>10.1}  {:>10.1}", "rtl_edge", avg(bottom, &|r| r.retail_edge), avg(top, &|r| r.retail_edge));
            println!("  {:>10}  {:>10.1}  {:>10.1}", "arb_cnt", avg(bottom, &|r| r.arb_trade_count as f64), avg(top, &|r| r.arb_trade_count as f64));
            println!("  {:>10}  {:>10.1}  {:>10.1}", "rtl_cnt", avg(bottom, &|r| r.retail_trade_count as f64), avg(top, &|r| r.retail_trade_count as f64));
            // Left-tail retail trades in bottom 10%
            let bot_left: u64 = bottom.iter().map(|r| r.retail_edge_buckets[0] + r.retail_edge_buckets[1] + r.retail_edge_buckets[2]).sum();
            let bot_total_rtl: u64 = bottom.iter().map(|r| r.retail_trade_count).sum();
            let top_left: u64 = top.iter().map(|r| r.retail_edge_buckets[0] + r.retail_edge_buckets[1] + r.retail_edge_buckets[2]).sum();
            let top_total_rtl: u64 = top.iter().map(|r| r.retail_trade_count).sum();
            println!("  {:>10}  {:>9.1}%  {:>9.1}%", "rtl_loss%",
                if bot_total_rtl > 0 { bot_left as f64 / bot_total_rtl as f64 * 100.0 } else { 0.0 },
                if top_total_rtl > 0 { top_left as f64 / top_total_rtl as f64 * 100.0 } else { 0.0 });

            // Mode usage in bottom 10%
            let mut bot_modes = [0u64; 16];
            let mut top_modes = [0u64; 16];
            for r in bottom { for i in 0..16 { bot_modes[i] += r.mode_counts[i]; } }
            for r in top { for i in 0..16 { top_modes[i] += r.mode_counts[i]; } }
            let bot_total: u64 = bot_modes.iter().sum();
            let top_total: u64 = top_modes.iter().sum();
            if bot_total > 0 && top_total > 0 {
                println!("\n  Mode usage bottom10% vs top10%:");
                for i in 0..16 {
                    if bot_modes[i] > 0 || top_modes[i] > 0 {
                        println!("    P{}: {:>5.1}% vs {:>5.1}%", i,
                            bot_modes[i] as f64 / bot_total as f64 * 100.0,
                            top_modes[i] as f64 / top_total as f64 * 100.0);
                    }
                }
            }
        }
    }


    // --- CSV DUMP (for ML pipeline) ---
    if std::env::var("CSV_DUMP").is_ok() {
        let csv_path = std::env::var("CSV_DUMP").unwrap_or_else(|_| "sim_data.csv".to_string());
        let mut csv = String::new();
        csv.push_str("seed,edge,sigma,nliq,arrival,size,nfee,arb_edge,retail_edge,arb_cnt,retail_cnt,p0_pct,p1_pct,p2_pct,n_switches\n");
        for r in &result.results {
            let ts: u64 = r.mode_counts.iter().sum();
            let p0 = if ts > 0 { r.mode_counts[0] as f64 / ts as f64 } else { 0.0 };
            let p1 = if ts > 0 { r.mode_counts[1] as f64 / ts as f64 } else { 0.0 };
            let p2 = if ts > 0 { r.mode_counts[2] as f64 / ts as f64 } else { 0.0 };
            let nsw = r.mode_switches.len();
            csv.push_str(&format!("{},{:.4},{:.6},{:.4},{:.4},{:.2},{},{:.4},{:.4},{},{},{:.4},{:.4},{:.4},{}\n",
                r.seed, r.submission_edge, r.gbm_sigma, r.norm_liquidity_mult,
                r.retail_arrival_rate, r.retail_mean_size, r.norm_fee_bps,
                r.arb_edge, r.retail_edge, r.arb_trade_count, r.retail_trade_count,
                p0, p1, p2, nsw));
        }
        std::fs::write(&csv_path, &csv).ok();
        eprintln!("CSV written to {}", csv_path);
    }

        if let Some(stats) = prop_amm_sim::search_stats::snapshot_if_enabled() {
        let arb_calls = stats.arb_golden_calls.max(1);
        let router_calls = stats.router_calls.max(1);
        println!("\nSearch stats (PROP_AMM_SEARCH_STATS=1):");
        println!(
            "  Arb golden:  calls={} iters={} (avg {:.2}/call) evals={} (avg {:.2}/call) early_stop_amount_tol={}",
            stats.arb_golden_calls,
            stats.arb_golden_iters,
            stats.arb_golden_iters as f64 / arb_calls as f64,
            stats.arb_golden_evals,
            stats.arb_golden_evals as f64 / arb_calls as f64,
            stats.arb_early_stop_amount_tol,
        );
        println!(
            "  Arb bracket: calls={} evals={} (avg {:.2}/call)",
            stats.arb_bracket_calls,
            stats.arb_bracket_evals,
            stats.arb_bracket_evals as f64 / stats.arb_bracket_calls.max(1) as f64,
        );
        println!(
            "  Router:     calls={} iters={} (avg {:.2}/call) evals={} (avg {:.2}/call) early_stop_rel_gap={}",
            stats.router_calls,
            stats.router_golden_iters,
            stats.router_golden_iters as f64 / router_calls as f64,
            stats.router_evals,
            stats.router_evals as f64 / router_calls as f64,
            stats.router_early_stop_rel_gap,
        );
    }
}
