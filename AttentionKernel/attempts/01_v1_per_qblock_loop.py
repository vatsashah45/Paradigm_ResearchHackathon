"""
v1: Per-q_block Python loop — 38.59ms (#18)

First attempt. Loops over each q_block in Python, processes one at a time.
Slow due to many small kernel launches (one GEMM per q_block).
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

    q_f = q.to(torch.float32).reshape(batch_heads, NQB, BLOCK_SIZE, head_dim)
    k_f = k.to(torch.float32).reshape(batch_heads, NQB, BLOCK_SIZE, head_dim)
    v_f = v.to(torch.float32).reshape(batch_heads, NQB, BLOCK_SIZE, head_dim)

    row_ptr_2d = row_ptr.reshape(batch_heads, NQB + 1).to(torch.int64)
    col_idx_2d = col_idx.reshape(batch_heads, -1).to(torch.int64)
    seq_lens_flat = seq_lens[:, None].expand(batch_size, num_heads).reshape(batch_heads).to(torch.int64)

    output = torch.zeros(batch_heads, NQB, BLOCK_SIZE, head_dim, device=device, dtype=torch.float32)
    lse = torch.full((batch_heads, NQB, BLOCK_SIZE), -torch.inf, device=device, dtype=torch.float32)

    block_tok = torch.arange(BLOCK_SIZE, device=device, dtype=torch.int64)

    for qb in range(NQB):
        q_start = qb * BLOCK_SIZE
        active = seq_lens_flat > q_start
        if not active.any():
            continue

        active_bh = torch.nonzero(active, as_tuple=False).squeeze(1)
        rs = row_ptr_2d[active_bh, qb]
        re = row_ptr_2d[active_bh, qb + 1]
        degrees = re - rs
        D = int(degrees.max().item())
        if D <= 0:
            continue

        slot_offsets = torch.arange(D, device=device, dtype=torch.int64)
        gather_pos = (rs[:, None] + slot_offsets[None, :]).clamp(max=col_idx_2d.shape[1] - 1)
        slot_valid = slot_offsets[None, :] < degrees[:, None]

        selected_col = col_idx_2d[active_bh]
        k_block_idx = torch.gather(selected_col, 1, gather_pos)
        k_block_idx = torch.where(slot_valid, k_block_idx, torch.zeros_like(k_block_idx))

        M = active_bh.shape[0]
        q_chunk = q_f[active_bh, qb]  # [M, BS, HD]

        flat_k = k_f[active_bh].reshape(M * NQB, BLOCK_SIZE, head_dim)
        flat_v = v_f[active_bh].reshape(M * NQB, BLOCK_SIZE, head_dim)
        bh_base = torch.arange(M, device=device, dtype=torch.int64)[:, None] * NQB
        flat_idx = (bh_base + k_block_idx).reshape(-1)

        gathered_k = flat_k[flat_idx].reshape(M, D * BLOCK_SIZE, head_dim)
        gathered_v = flat_v[flat_idx].reshape(M, D * BLOCK_SIZE, head_dim)

        scores = torch.bmm(q_chunk, gathered_k.transpose(1, 2)) * SCALE

        seq_active = seq_lens_flat[active_bh]
        q_pos = q_start + block_tok
        q_valid = q_pos[None, :] < seq_active[:, None]

        k_pos = (k_block_idx[:, :, None] * BLOCK_SIZE + block_tok[None, None, :]).reshape(M, D * BLOCK_SIZE)
        k_valid = slot_valid[:, :, None].expand(M, D, BLOCK_SIZE).reshape(M, D * BLOCK_SIZE) & (k_pos < seq_active[:, None])

        is_diag = (k_block_idx == qb)[:, :, None].expand(M, D, BLOCK_SIZE).reshape(M, D * BLOCK_SIZE)
        causal_ok = k_pos[:, None, :] <= q_pos[None, :, None]
        mask = k_valid[:, None, :] & q_valid[:, :, None]
        mask = mask & ((~is_diag[:, None, :]) | causal_ok)

        scores = scores.masked_fill(~mask, -torch.inf)
        row_max = scores.max(dim=-1).values
        vr = q_valid & torch.isfinite(row_max)
        rm_safe = torch.where(vr, row_max, torch.zeros_like(row_max))
        exp_s = torch.exp(scores - rm_safe[:, :, None]) * mask.to(torch.float32)
        denom = exp_s.sum(dim=-1)
        denom_safe = torch.where(vr, denom, torch.ones_like(denom))

        out_chunk = torch.bmm(exp_s, gathered_v) / denom_safe[:, :, None]
        out_chunk = torch.where(vr[:, :, None], out_chunk, torch.zeros_like(out_chunk))
        lse_chunk = torch.where(vr, rm_safe + torch.log(denom_safe), torch.full_like(rm_safe, -torch.inf))

        output[active_bh, qb] = out_chunk
        lse[active_bh, qb] = lse_chunk

    return (
        output.reshape(batch_size, num_heads, t_max, head_dim).to(torch.bfloat16),
        lse.reshape(batch_size, num_heads, t_max),
    )
