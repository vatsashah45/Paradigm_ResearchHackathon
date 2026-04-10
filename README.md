# Paradigm Research Hackathon — Optimization Arena Submissions

[![Optimization Arena](https://img.shields.io/badge/Platform-Optimization%20Arena-blue)](https://www.optimizationarena.com/)

This repository contains our submissions and strategy iterations for the [Optimization Arena](https://www.optimizationarena.com/) hackathon challenges by [Paradigm](https://www.paradigm.xyz/).

Optimization Arena is an online platform where participants solve real-world challenges in trading, prediction markets, GPU optimization, and AI strategy by writing code that competes in robust simulated environments. Each challenge provides a sandboxed environment, a scoring function, and a leaderboard for iterative improvement.

## Challenges

| Challenge | Type | Best Score | Folder |
|-----------|------|------------|--------|
| [Attention Kernel](https://www.optimizationarena.com/) | GPU Optimization (Python/Triton) | 11.73 ms (#16) | [`AttentionKernel/`](./AttentionKernel/) |
| [Prediction Market](https://www.optimizationarena.com/prediction-market-challenge) | Trading Strategy (Python) | $2.30 (#51) | [`PredictionMarket/`](./PredictionMarket/) |
| [Prop AMM](https://www.optimizationarena.com/prop-amm) | AMM Strategy (Rust/Solana BPF) | 516.96 (#24) | [`PropAMM/`](./PropAMM/) |
| [Negotiation](https://www.optimizationarena.com/) | LLM Prompt Engineering | 83.8% | [`Negotiation_Prompts/`](./Negotiation_Prompts/) |
| [Sell This Pen](https://www.optimizationarena.com/persuasion) | Persuasion / Copywriting | 82.2% | [`Persuasion_Prompts/`](./Persuasion_Prompts/) |

## Challenge Summaries

### 🧠 [Attention Kernel](./AttentionKernel/)

Build the fastest numerically faithful **block-sparse attention** backend for NVIDIA H100 GPUs. Solutions are ranked by measured latency (lower is better) after correctness checks. Inputs are Q, K, V tensors in bf16 with sparse attention patterns in CSR format. Our best submission achieved **11.73 ms** using batched PyTorch with in-place operations.

- **Tech:** Python, PyTorch, Triton
- **GitHub:** [paradigmxyz/attention-kernel-challenge](https://github.com/paradigmxyz/attention-kernel-challenge)

### 📈 [Prediction Market](./PredictionMarket/)

Build a profitable **market-making strategy** for a FIFO limit order book prediction market. The challenge involves managing limit orders and trading a YES contract in a market with Gaussian drift and Poisson jumps. Strategies are scored by mean edge across 200 simulations. Our best submission achieved a **$2.30 mean edge**.

- **Tech:** Python, `orderbook-pm-challenge` package
- **Website:** [Prediction Market Challenge](https://www.optimizationarena.com/prediction-market-challenge)

### 💱 [Prop AMM](./PropAMM/)

Design a custom **Automated Market Maker** strategy in Rust, compiled to Solana BPF bytecode. Unlike standard AMMs, you control the entire `compute_swap()` function—implementing any pricing curve, dynamic fees, or novel logic. Scored by edge (profit vs true market price) over 1,000 simulations. Our best submission scored **516.96** with a 3-profile momentum switcher.

- **Tech:** Rust, Solana BPF
- **Website:** [Prop AMM Competition](https://www.optimizationarena.com/prop-amm/about)

### 🤝 [Negotiation](./Negotiation_Prompts/)

Write a **strategy prompt** for an AI agent that negotiates resource splits (books, hats, balls) against a baseline across 10 games. Each game involves two AI agents splitting a resource pool with hidden valuations. The strategy prompt is injected as system instructions into a Gemini model. Our best prompt achieved **83.8%** success rate.

- **Tech:** LLM Prompt Engineering (Gemini)
- **GitHub:** [paradigmxyz/negotiation-challenge](https://github.com/paradigmxyz/negotiation-challenge)

### 🖊️ [Sell This Pen](./Persuasion_Prompts/)

Write a **140-character sales pitch** for a pen to maximize median willingness-to-pay across 15 simulated buyers with diverse backgrounds—from college students to surgeons, ranchers to attorneys. Your score is the median price across all buyers.

- **Tech:** Persuasion / Copywriting
- **Website:** [Sell This Pen](https://www.optimizationarena.com/persuasion)

## Author

**@0xVatsaShah** — [GitHub](https://github.com/vatsashah45)

## Links

- 🌐 [Optimization Arena](https://www.optimizationarena.com/)
- 🏗️ [Paradigm](https://www.paradigm.xyz/)
