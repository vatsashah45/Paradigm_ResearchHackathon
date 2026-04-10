# Attention Kernel Challenge — Paradigm Automated Research Hackathon

**Challenge:** Build the fastest numerically faithful block-sparse attention backend for H100.  
**Author:** @0xVatsaShah  
**Best Score:** 11.73ms (v11, #16 on leaderboard)

## Challenge Details

- **Hardware:** 1x H100 SXM 80GB
- **Block size:** 128, Head dim: 128, Scale: 1/√128
- **Families:** sliding_window, sliding_window_global, sliding_window_retrieval
- **Sparsity format:** CSR (row_ptr, col_idx)
- **Inputs:** Q, K, V in bf16 (batch, heads, t_max, 128), seq_lens
- **Outputs:** o (bf16), lse (f32)
- **Tolerances:** output_atol=1e-3, output_rtol=1e-2, lse_atol=1e-5, lse_rtol=1e-5
- **Scoring:** geometric mean of per-family median latencies (lower = better)
- **Allowed imports:** torch, triton, numpy only

## Key Constraints Discovered

1. **bf16 matmul FAILS lse tolerance** (0.003 > 1e-5) — must use f32 for Q*K^T
2. **torch.compile FAILS in their sandbox** — submissions need fallback
3. **bf16 gather + f32 conversion is SLOWER** than direct f32 gather (extra kernel launch)
4. **logsumexp is SLOWER** than manual max+exp+sum for these tensor shapes on H100
5. **Triton kernels were consistently slower** than batched PyTorch approach (tested 4 times)
6. **In-place ops save ~0.6ms** by avoiding intermediate tensor allocations

## Attempt History

| Version | File | Score | Strategy |
|---------|------|-------|----------|
| v1 | `01_v1_per_qblock_loop.py` | 38.59ms | Python loop over q_blocks |
| v2 | `02_v2_fully_parallel.py` | 12.42ms | All q_blocks parallel, batched GEMMs |
| v3 | `03_v3_torch_compile.py` | REJECTED | torch.compile on mask+softmax (fails in sandbox) |
| v4 | `04_v4_compile_fallback.py` | 12.40ms | torch.compile with try/except fallback |
| v5 | `05_v5_bf16_gather_logsumexp.py` | N/A (not submitted as-is) | bf16 gather + logsumexp |
| v6 | `06_v6_triton_fused.py` | 14.64ms | Triton fused kernel (4 warps, register spill) |
| v7 | `07_v7_triton_online_softmax.py` | 16.55ms | Triton 8 warps + PyTorch online softmax loop |
| v8 | `08_v8_triton_auto_benchmark.py` | 12.44ms | Triton auto-benchmark, v4 PyTorch fallback |
| v9 | `09_v9_triton_bf16_tc.py` | 12.37ms | Triton bf16 tensor cores, fixed-bound loop |
| v10 | `10_v10_pytorch_bf16_logsumexp.py` | 13.37ms | PyTorch only, bf16 gather + logsumexp |
| v11 | `11_v11_inplace_ops.py` | **11.73ms** | v4 base + in-place ops (best!) |
