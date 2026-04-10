#[derive(Debug, Clone)]
pub struct SimResult {
    pub seed: u64,
    pub submission_edge: f64,
    pub mode_counts: [u64; 16],
    pub mode_switches: Vec<(u64, u8, u8)>,
    pub gbm_sigma: f64,
    pub norm_liquidity_mult: f64,
    pub retail_arrival_rate: f64,
    pub retail_mean_size: f64,
    pub norm_fee_bps: u16,
    // Per-trade edge tracking
    pub arb_edge: f64,
    pub retail_edge: f64,
    pub retail_trade_count: u64,
    pub arb_trade_count: u64,
    // Edge distribution buckets for retail trades
    // [<-1, -1..-0.1, -0.1..0, 0..0.01, 0.01..0.1, 0.1..1, 1..10, >=10]
    pub retail_edge_buckets: [u64; 8],
    pub retail_edge_bucket_sums: [f64; 8],
    // Fine-grained small-edge tracking (retail trades with edge in [-0.1, 0.01])
    // Buckets: [<-0.01, -0.01..0, 0..0.001, 0.001..0.005, 0.005..0.01]
    pub small_edge_buckets: [u64; 5],
    pub small_edge_sums: [f64; 5],
    // Characteristics of small-loss trades (edge < 0.01)
    pub small_loss_arb_dir_count: u64,    // arb-direction trades with edge < 0.01
    pub small_loss_counter_count: u64,    // counter-direction trades with edge < 0.01
    pub small_loss_size_sum: f64,         // sum of input sizes for edge < 0.01
    pub small_loss_count: u64,            // total trades with edge < 0.01
}

#[derive(Debug, Clone)]
pub struct BatchResult {
    pub results: Vec<SimResult>,
    pub total_edge: f64,
}

impl BatchResult {
    pub fn from_results(results: Vec<SimResult>) -> Self {
        let total_edge = results.iter().map(|r| r.submission_edge).sum();
        Self {
            results,
            total_edge,
        }
    }

    pub fn n_sims(&self) -> usize {
        self.results.len()
    }

    pub fn avg_edge(&self) -> f64 {
        if self.results.is_empty() {
            0.0
        } else {
            self.total_edge / self.results.len() as f64
        }
    }

    pub fn bucket_by_regime(&self) -> Vec<(&'static str, BatchResult)> {
        let mut buckets: std::collections::HashMap<&'static str, Vec<SimResult>> =
            std::collections::HashMap::new();
        for r in &self.results {
            let label = regime_label(r);
            buckets.entry(label).or_default().push(r.clone());
        }
        let mut out: Vec<_> = buckets
            .into_iter()
            .map(|(label, results)| (label, BatchResult::from_results(results)))
            .collect();
        out.sort_by(|(_, a), (_, b)| a.avg_edge().partial_cmp(&b.avg_edge()).unwrap().reverse());
        out
    }
}

pub fn regime_label(r: &SimResult) -> &'static str {
    // 3x3 grid: norm_liq (thin/mid/deep) x size (S/M/L)
    // norm_liq: U(0.4, 2.0) → thirds at ~0.93, ~1.47
    // size:     U(12, 28) → thirds at ~17.3, ~22.7
    let depth_level = if r.norm_liquidity_mult < 0.93 { 0 }
                      else if r.norm_liquidity_mult < 1.47 { 1 }
                      else { 2 };
    let size_level = if r.retail_mean_size < 17.3 { 0 }
                     else if r.retail_mean_size < 22.7 { 1 }
                     else { 2 };
    match (depth_level, size_level) {
        (0, 0) => "thin-S",
        (0, 1) => "thin-M",
        (0, 2) => "thin-L",
        (1, 0) => "mid-S",
        (1, 1) => "mid-M",
        (1, 2) => "mid-L",
        (2, 0) => "deep-S",
        (2, 1) => "deep-M",
        (_, _) => "deep-L",
    }
}
