"""
v7: Triton 8 warps + PyTorch online softmax fallback — 16.55ms (WORST)

Tried to fix v6's register pressure by using 8 warps (halves regs/thread).
Also added a PyTorch online softmax loop as fallback.

Problem: The PyTorch online softmax loop processes KV blocks sequentially
(~120 kernel launches for large cases), which is far worse than the
~40 kernel launches in the batched approach (v2/v4).

Key lesson: Online softmax only helps when fused into a single kernel.
As separate PyTorch ops, it's strictly worse than batched softmax.
"""

import math
import torch
import triton
import triton.language as tl

BLOCK_SIZE = 128
HEAD_DIM = 128
SCALE = 1.0 / math.sqrt(HEAD_DIM)

VARIANT_MANIFEST = [
    {"name": "default"},
]


def setup(suite_specs, device, variants):
    return None


@triton.jit
def _block_sparse_attn_kernel_v7(
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
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """Same kernel as v6 but with 8 warps to reduce register pressure."""
    pid = tl.program_id(0)
    bh = pid // NQB
    qb = pid % NQB
    b = bh // num_heads
    h = bh % num_heads

    seq_len = tl.load(seq_lens + b)
    q_start = qb * BLOCK_SIZE

    if q_start >= seq_len:
        q_offsets = tl.arange(0, BLOCK_SIZE)
        for qi in range(BLOCK_SIZE):
            tl.store(Lse + b * stride_lb + h * stride_lh + (q_start + qi) * stride_lt, float('-inf'))
        return

    q_offs = tl.arange(0, BLOCK_SIZE)
    d_offs = tl.arange(0, HEAD_DIM)
    q_ptrs = Q + b * stride_qb + h * stride_qh + (q_start + q_offs[:, None]) * stride_qt + d_offs[None, :] * stride_qd
    q_block = tl.load(q_ptrs).to(tl.float32)

    m_i = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    acc = tl.zeros([BLOCK_SIZE, HEAD_DIM], dtype=tl.float32)

    rs = tl.load(row_ptr + bh * stride_rp_b + qb * stride_rp_i)
    re = tl.load(row_ptr + bh * stride_rp_b + (qb + 1) * stride_rp_i)
    num_kv_blocks = re - rs

    for slot in range(num_kv_blocks):
        kb = tl.load(col_idx + bh * stride_ci_b + (rs + slot) * stride_ci_i)
        k_start = kb * BLOCK_SIZE
        k_offs = tl.arange(0, BLOCK_SIZE)
        k_ptrs = K + b * stride_kb + h * stride_kh + (k_start + k_offs[:, None]) * stride_kt + d_offs[None, :] * stride_kd
        k_block = tl.load(k_ptrs).to(tl.float32)

        qk = tl.dot(q_block, tl.trans(k_block)) * SCALE

        if kb == qb:
            q_pos = q_start + q_offs[:, None]
            k_pos = k_start + k_offs[None, :]
            causal_mask = q_pos >= k_pos
            qk = tl.where(causal_mask, qk, float('-inf'))

        k_pos_check = k_start + k_offs
        k_valid = k_pos_check < seq_len
        qk = tl.where(k_valid[None, :], qk, float('-inf'))

        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v_ptrs = V + b * stride_vb + h * stride_vh + (k_start + k_offs[:, None]) * stride_vt + d_offs[None, :] * stride_vd
        v_block = tl.load(v_ptrs).to(tl.float32)
        acc += tl.dot(p.to(tl.float32), v_block)

        m_i = m_new

    out_ptrs = Out + b * stride_ob + h * stride_oh + (q_start + q_offs[:, None]) * stride_ot + d_offs[None, :] * stride_od
    q_valid_mask = (q_start + q_offs) < seq_len
    safe_l = tl.where(l_i > 0, l_i, 1.0)
    result = acc / safe_l[:, None]
    result = tl.where(q_valid_mask[:, None], result, 0.0)
    tl.store(out_ptrs, result.to(tl.bfloat16))

    lse_val = tl.where(q_valid_mask & (l_i > 0), m_i + tl.log(safe_l), float('-inf'))
    lse_ptrs = Lse + b * stride_lb + h * stride_lh + (q_start + q_offs) * stride_lt
    tl.store(lse_ptrs, lse_val)


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

    _block_sparse_attn_kernel_v7[grid](
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
        BLOCK_SIZE=BLOCK_SIZE,
        HEAD_DIM=HEAD_DIM,
        num_warps=8,  # 8 warps to reduce register pressure
    )

    return output, lse
