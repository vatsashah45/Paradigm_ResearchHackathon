"""
v9: Triton bf16 tensor cores, fixed-bound loop — 12.37ms (marginal improvement)

Triton kernel using bf16 tensor cores and a fixed iteration count
(MAX_KV_BLOCKS) to avoid dynamic loop bounds. Falls back to PyTorch
if Triton fails.

Result: 12.37ms — barely better than v4 (12.40ms). The Triton kernel
was selected by the auto-benchmark this time, but improvement was minimal.
"""

import math
import torch
import triton
import triton.language as tl

BLOCK_SIZE = 128
HEAD_DIM = 128
SCALE = 1.0 / math.sqrt(HEAD_DIM)
MAX_KV_BLOCKS = 64  # Upper bound on KV blocks per Q block

VARIANT_MANIFEST = [
    {"name": "default"},
]

_use_triton = True


@triton.jit
def _block_sparse_attn_v9(
    Q, K, V, Out, Lse,
    row_ptr, col_idx, seq_lens,
    stride_qb, stride_qh, stride_qt, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_ob, stride_oh, stride_ot, stride_od,
    stride_lb, stride_lh, stride_lt,
    stride_rp_b, stride_rp_i,
    stride_ci_b, stride_ci_i,
    num_heads: tl.constexpr,
    NQB: tl.constexpr,
    BS: tl.constexpr,
    HD: tl.constexpr,
    MAX_KV: tl.constexpr,
):
    pid = tl.program_id(0)
    bh = pid // NQB
    qb = pid % NQB
    b = bh // num_heads
    h = bh % num_heads

    seq_len = tl.load(seq_lens + b)
    q_start = qb * BS

    q_offs = tl.arange(0, BS)
    d_offs = tl.arange(0, HD)

    if q_start >= seq_len:
        # Write -inf to LSE for padding rows
        lse_ptrs = Lse + b * stride_lb + h * stride_lh + (q_start + q_offs) * stride_lt
        tl.store(lse_ptrs, float('-inf') + tl.zeros([BS], dtype=tl.float32))
        return

    # Load Q in bf16, accumulate in f32
    q_ptrs = Q + b * stride_qb + h * stride_qh + (q_start + q_offs[:, None]) * stride_qt + d_offs[None, :] * stride_qd
    q_block = tl.load(q_ptrs)  # bf16

    m_i = tl.full([BS], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BS], dtype=tl.float32)
    acc = tl.zeros([BS, HD], dtype=tl.float32)

    rs = tl.load(row_ptr + bh * stride_rp_b + qb * stride_rp_i)
    re = tl.load(row_ptr + bh * stride_rp_b + (qb + 1) * stride_rp_i)
    num_kv = re - rs

    # Fixed bound loop for compiler optimization
    for slot in range(MAX_KV):
        if slot >= num_kv:
            continue

        kb = tl.load(col_idx + bh * stride_ci_b + (rs + slot) * stride_ci_i)
        k_start = kb * BS
        k_offs = tl.arange(0, BS)

        # Load K/V in bf16
        k_ptrs = K + b * stride_kb + h * stride_kh + (k_start + k_offs[:, None]) * stride_kt + d_offs[None, :] * stride_kd
        k_block = tl.load(k_ptrs)  # bf16

        # bf16 tensor core matmul → f32 accumulation
        qk = tl.dot(q_block, tl.trans(k_block)).to(tl.float32) * SCALE

        # Masks
        if kb == qb:
            causal_mask = (q_start + q_offs[:, None]) >= (k_start + k_offs[None, :])
            qk = tl.where(causal_mask, qk, float('-inf'))
        k_valid = (k_start + k_offs) < seq_len
        qk = tl.where(k_valid[None, :], qk, float('-inf'))

        # Online softmax
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v_ptrs = V + b * stride_vb + h * stride_vh + (k_start + k_offs[:, None]) * stride_vt + d_offs[None, :] * stride_vd
        v_block = tl.load(v_ptrs)  # bf16
        # p is f32, v_block is bf16 — dot promotes to f32
        acc += tl.dot(p.to(tl.bfloat16), v_block).to(tl.float32)

        m_i = m_new

    q_valid_mask = (q_start + q_offs) < seq_len
    safe_l = tl.where(l_i > 0, l_i, 1.0)
    result = acc / safe_l[:, None]
    result = tl.where(q_valid_mask[:, None], result, 0.0)

    out_ptrs = Out + b * stride_ob + h * stride_oh + (q_start + q_offs[:, None]) * stride_ot + d_offs[None, :] * stride_od
    tl.store(out_ptrs, result.to(tl.bfloat16))

    lse_val = tl.where(q_valid_mask & (l_i > 0), m_i + tl.log(safe_l), float('-inf'))
    lse_ptrs = Lse + b * stride_lb + h * stride_lh + (q_start + q_offs) * stride_lt
    tl.store(lse_ptrs, lse_val)


def setup(suite_specs, device, variants):
    return None


def block_sparse_attn_fwd(q, k, v, row_ptr, col_idx, seq_lens):
    batch_size, num_heads, t_max, head_dim = q.shape
    device = q.device
    NQB = t_max // BLOCK_SIZE
    batch_heads = batch_size * num_heads

    output = torch.zeros(batch_size, num_heads, t_max, head_dim, device=device, dtype=torch.bfloat16)
    lse = torch.full((batch_size, num_heads, t_max), -torch.inf, device=device, dtype=torch.float32)

    row_ptr_2d = row_ptr.reshape(batch_heads, -1)
    col_idx_2d = col_idx.reshape(batch_heads, -1)

    grid = (batch_heads * NQB,)

    try:
        _block_sparse_attn_v9[grid](
            q, k, v, output, lse,
            row_ptr_2d, col_idx_2d, seq_lens,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            output.stride(0), output.stride(1), output.stride(2), output.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            row_ptr_2d.stride(0), row_ptr_2d.stride(1),
            col_idx_2d.stride(0), col_idx_2d.stride(1),
            num_heads=num_heads,
            NQB=NQB,
            BS=BLOCK_SIZE,
            HD=HEAD_DIM,
            MAX_KV=MAX_KV_BLOCKS,
            num_warps=8,
        )
        return output, lse
    except Exception:
        # Fallback to PyTorch v4
        return _pytorch_fallback(q, k, v, row_ptr, col_idx, seq_lens)


def _pytorch_fallback(q, k, v, row_ptr, col_idx, seq_lens):
    """v4 PyTorch fallback."""
    batch_size, num_heads, t_max, head_dim = q.shape
    device = q.device
    batch_heads = batch_size * num_heads
    NQB = t_max // BLOCK_SIZE
    N = batch_heads * NQB

    q_f = q.to(torch.float32).reshape(batch_heads, NQB, BLOCK_SIZE, head_dim)
    k_f = k.to(torch.float32).reshape(batch_heads, NQB, BLOCK_SIZE, head_dim)
    v_f = v.to(torch.float32).reshape(batch_heads, NQB, BLOCK_SIZE, head_dim)
    row_ptr_2d = row_ptr.reshape(batch_heads, NQB + 1).to(torch.int64)
    col_idx_2d = col_idx.reshape(batch_heads, -1).to(torch.int64)
    seq_lens_flat = seq_lens[:, None].expand(batch_size, num_heads).reshape(batch_heads).to(torch.int64)
    degrees_all = row_ptr_2d[:, 1:] - row_ptr_2d[:, :-1]
    D = int(degrees_all.max().item())
    if D <= 0:
        return (
            torch.zeros(batch_size, num_heads, t_max, head_dim, device=device, dtype=torch.bfloat16),
            torch.full((batch_size, num_heads, t_max), -torch.inf, device=device, dtype=torch.float32),
        )
    slot_offsets = torch.arange(D, device=device, dtype=torch.int64)
    row_start_all = row_ptr_2d[:, :-1]
    gather_pos = (row_start_all[:, :, None] + slot_offsets[None, None, :]).clamp(max=col_idx_2d.shape[1] - 1)
    slot_valid = slot_offsets[None, None, :] < degrees_all[:, :, None]
    k_block_idx = torch.gather(col_idx_2d, 1, gather_pos.reshape(batch_heads, -1)).reshape(batch_heads, NQB, D)
    k_block_idx = torch.where(slot_valid, k_block_idx, torch.zeros_like(k_block_idx))
    flat_k = k_f.reshape(batch_heads * NQB, BLOCK_SIZE, head_dim)
    flat_v = v_f.reshape(batch_heads * NQB, BLOCK_SIZE, head_dim)
    bh_base = torch.arange(batch_heads, device=device, dtype=torch.int64)[:, None, None] * NQB
    flat_gather = (bh_base + k_block_idx).reshape(-1)
    gathered_k = flat_k[flat_gather].reshape(N, D * BLOCK_SIZE, head_dim)
    gathered_v = flat_v[flat_gather].reshape(N, D * BLOCK_SIZE, head_dim)
    q_flat = q_f.reshape(N, BLOCK_SIZE, head_dim)
    scores = torch.bmm(q_flat, gathered_k.transpose(1, 2)) * SCALE
    block_tok = torch.arange(BLOCK_SIZE, device=device, dtype=torch.int64)
    qb_idx = torch.arange(NQB, device=device, dtype=torch.int64)
    seq_N = seq_lens_flat[:, None].expand(batch_heads, NQB).reshape(N)
    q_pos = (qb_idx[None, :, None] * BLOCK_SIZE + block_tok[None, None, :]).expand(batch_heads, NQB, BLOCK_SIZE).reshape(N, BLOCK_SIZE)
    q_valid = q_pos < seq_N[:, None]
    k_pos = (k_block_idx[:, :, :, None] * BLOCK_SIZE + block_tok[None, None, None, :]).reshape(batch_heads, NQB, D * BLOCK_SIZE).reshape(N, D * BLOCK_SIZE)
    k_valid = (slot_valid[:, :, :, None].expand(batch_heads, NQB, D, BLOCK_SIZE).reshape(N, D * BLOCK_SIZE)) & (k_pos < seq_N[:, None])
    is_diag = ((k_block_idx == qb_idx[None, :, None])[:, :, :, None].expand(batch_heads, NQB, D, BLOCK_SIZE).reshape(N, D * BLOCK_SIZE))
    causal_ok = k_pos[:, None, :] <= q_pos[:, :, None]
    mask = k_valid[:, None, :] & q_valid[:, :, None]
    mask = mask & ((~is_diag[:, None, :]) | causal_ok)
    scores = scores.masked_fill(~mask, -torch.inf)
    row_max = scores.max(dim=-1).values
    valid_rows = q_valid & torch.isfinite(row_max)
    row_max_safe = torch.where(valid_rows, row_max, torch.zeros_like(row_max))
    exp_scores = torch.exp(scores - row_max_safe[:, :, None]) * mask.to(torch.float32)
    denom = exp_scores.sum(dim=-1)
    denom_safe = torch.where(valid_rows, denom, torch.ones_like(denom))
    out = torch.bmm(exp_scores, gathered_v) / denom_safe[:, :, None]
    out = torch.where(valid_rows[:, :, None], out, torch.zeros_like(out))
    lse = torch.where(valid_rows, row_max_safe + torch.log(denom_safe), torch.full_like(row_max_safe, -torch.inf))
    return (
        out.reshape(batch_size, num_heads, t_max, head_dim).to(torch.bfloat16),
        lse.reshape(batch_size, num_heads, t_max),
    )
