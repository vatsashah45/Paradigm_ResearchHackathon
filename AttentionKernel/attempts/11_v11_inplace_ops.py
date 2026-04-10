"""
v11: v4 base + in-place ops — 11.73ms (BEST! Personal record, #16)

Key insight: Replace out-of-place ops with in-place equivalents on the
large [N, 128, D×128] score tensor to eliminate ~3-4 intermediate tensor
allocations. This saved ~0.64ms.

Changes from v4:
- scores.masked_fill_(~mask, -torch.inf)  instead of  scores = scores.masked_fill(...)
- scores -= row_max_safe[:, :, None]       instead of  scores - row_max_safe (new tensor)
- scores.exp_()                            instead of  exp_scores = torch.exp(scores)
- scores *= mask.to(torch.float32)         instead of  * mask (new tensor)

These in-place ops avoid allocating new tensors of shape [N, 128, D*128]
which on H100 with large D can be significant.
"""

import math
import torch

BLOCK_SIZE = 128
HEAD_DIM = 128
SCALE = 1.0 / math.sqrt(HEAD_DIM)

VARIANT_MANIFEST = [
    {"name": "default"},
]

_graph_cache = {}


def setup(suite_specs, device, variants):
    return None


def block_sparse_attn_fwd(q, k, v, row_ptr, col_idx, seq_lens):
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

    gather_pos = (row_start_all[:, :, None] + slot_offsets[None, None, :]).clamp(
        max=col_idx_2d.shape[1] - 1
    )
    slot_valid = slot_offsets[None, None, :] < degrees_all[:, :, None]

    k_block_idx = torch.gather(
        col_idx_2d, 1, gather_pos.reshape(batch_heads, -1)
    ).reshape(batch_heads, NQB, D)
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

    q_pos = (qb_idx[None, :, None] * BLOCK_SIZE + block_tok[None, None, :]).expand(
        batch_heads, NQB, BLOCK_SIZE
    ).reshape(N, BLOCK_SIZE)
    q_valid = q_pos < seq_N[:, None]

    k_pos = (
        k_block_idx[:, :, :, None] * BLOCK_SIZE + block_tok[None, None, None, :]
    ).reshape(batch_heads, NQB, D * BLOCK_SIZE).reshape(N, D * BLOCK_SIZE)

    k_valid = (
        slot_valid[:, :, :, None]
        .expand(batch_heads, NQB, D, BLOCK_SIZE)
        .reshape(N, D * BLOCK_SIZE)
    ) & (k_pos < seq_N[:, None])

    is_diag = (
        (k_block_idx == qb_idx[None, :, None])[:, :, :, None]
        .expand(batch_heads, NQB, D, BLOCK_SIZE)
        .reshape(N, D * BLOCK_SIZE)
    )

    causal_ok = k_pos[:, None, :] <= q_pos[:, :, None]
    mask = k_valid[:, None, :] & q_valid[:, :, None]
    mask = mask & ((~is_diag[:, None, :]) | causal_ok)

    # === IN-PLACE OPS (the key optimization) ===
    scores.masked_fill_(~mask, -torch.inf)

    row_max = scores.max(dim=-1).values
    valid_rows = q_valid & torch.isfinite(row_max)
    row_max_safe = torch.where(valid_rows, row_max, torch.zeros_like(row_max))

    scores -= row_max_safe[:, :, None]
    scores.exp_()
    scores *= mask.to(torch.float32)
    denom = scores.sum(dim=-1)
    denom_safe = torch.where(valid_rows, denom, torch.ones_like(denom))

    out = torch.bmm(scores, gathered_v) / denom_safe[:, :, None]
    out = torch.where(valid_rows[:, :, None], out, torch.zeros_like(out))

    lse = torch.where(
        valid_rows,
        row_max_safe + torch.log(denom_safe),
        torch.full_like(row_max_safe, -torch.inf),
    )

    return (
        out.reshape(batch_size, num_heads, t_max, head_dim).to(torch.bfloat16),
        lse.reshape(batch_size, num_heads, t_max),
    )
