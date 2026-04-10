"""
v10: PyTorch only, bf16 gather + logsumexp — 13.37ms (WORSE)

Attempted to reduce memory bandwidth by gathering K/V in bf16 and
converting to f32 after gather. Also used torch.logsumexp for a
fused stable softmax.

Problem: The bf16→f32 conversion after gather creates an extra kernel
launch (type cast) that costs more than the bandwidth saved. Net result
was ~1ms slower than v4.

Key lesson: On H100 with these tensor sizes, memory bandwidth is not
the bottleneck — kernel launch overhead is.
"""

import math
import torch

BLOCK_SIZE = 128
HEAD_DIM = 128
SCALE = 1.0 / math.sqrt(HEAD_DIM)

VARIANT_MANIFEST = [
    {"name": "default"},
]


def setup(suite_specs, device, variants):
    return None


def block_sparse_attn_fwd(q, k, v, row_ptr, col_idx, seq_lens):
    batch_size, num_heads, t_max, head_dim = q.shape
    device = q.device
    batch_heads = batch_size * num_heads
    NQB = t_max // BLOCK_SIZE
    N = batch_heads * NQB

    q_f = q.to(torch.float32).reshape(batch_heads, NQB, BLOCK_SIZE, head_dim)
    # Keep K/V in bf16 for gather (halves bandwidth)
    k_bf = k.reshape(batch_heads, NQB, BLOCK_SIZE, head_dim)
    v_bf = v.reshape(batch_heads, NQB, BLOCK_SIZE, head_dim)

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

    # Gather in bf16, then convert to f32 (extra kernel launch!)
    flat_k_bf = k_bf.reshape(batch_heads * NQB, BLOCK_SIZE, head_dim)
    flat_v_bf = v_bf.reshape(batch_heads * NQB, BLOCK_SIZE, head_dim)
    bh_base = torch.arange(batch_heads, device=device, dtype=torch.int64)[:, None, None] * NQB
    flat_gather = (bh_base + k_block_idx).reshape(-1)

    gathered_k = flat_k_bf[flat_gather].to(torch.float32).reshape(N, D * BLOCK_SIZE, head_dim)
    gathered_v = flat_v_bf[flat_gather].to(torch.float32).reshape(N, D * BLOCK_SIZE, head_dim)

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

    # Use logsumexp for fused stable softmax
    lse = torch.logsumexp(scores, dim=-1)
    valid_rows = q_valid & torch.isfinite(lse)

    exp_scores = torch.exp(scores - lse[:, :, None]) * mask.to(torch.float32)
    out = torch.bmm(exp_scores, gathered_v)
    out = torch.where(valid_rows[:, :, None], out, torch.zeros_like(out))
    lse = torch.where(valid_rows, lse, torch.full_like(lse, -torch.inf))

    return (
        out.reshape(batch_size, num_heads, t_max, head_dim).to(torch.bfloat16),
        lse.reshape(batch_size, num_heads, t_max),
    )
