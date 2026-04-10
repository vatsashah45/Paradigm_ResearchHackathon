"""
v4: torch.compile with try/except fallback — 12.40ms (#6)

Same as v2 but tries torch.compile first; falls back to eager if it fails.
In the sandbox, compile always fails, so this effectively runs the same
as v2 but with try/except overhead removed at runtime.

Marginal improvement over v2 (12.40 vs 12.42ms) — within noise.
This became the "base" for later PyTorch-only iterations (v8, v11).
"""

import math
import torch

BLOCK_SIZE = 128
HEAD_DIM = 128
SCALE = 1.0 / math.sqrt(HEAD_DIM)

VARIANT_MANIFEST = [
    {"name": "default"},
]

_compiled_fn = None


def setup(suite_specs, device, variants):
    return None


def _core_attn(q_flat, gathered_k, gathered_v, mask, q_valid):
    scores = torch.bmm(q_flat, gathered_k.transpose(1, 2)) * SCALE
    scores = scores.masked_fill(~mask, -torch.inf)

    row_max = scores.max(dim=-1).values
    valid_rows = q_valid & torch.isfinite(row_max)
    row_max_safe = torch.where(valid_rows, row_max, torch.zeros_like(row_max))

    exp_scores = torch.exp(scores - row_max_safe[:, :, None]) * mask.to(torch.float32)
    denom = exp_scores.sum(dim=-1)
    denom_safe = torch.where(valid_rows, denom, torch.ones_like(denom))

    out = torch.bmm(exp_scores, gathered_v) / denom_safe[:, :, None]
    out = torch.where(valid_rows[:, :, None], out, torch.zeros_like(out))

    lse = torch.where(
        valid_rows,
        row_max_safe + torch.log(denom_safe),
        torch.full_like(row_max_safe, -torch.inf),
    )
    return out, lse


def block_sparse_attn_fwd(q, k, v, row_ptr, col_idx, seq_lens):
    global _compiled_fn

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

    # Try compile, fall back to eager
    if _compiled_fn is None:
        try:
            _compiled_fn = torch.compile(_core_attn, mode="reduce-overhead")
        except Exception:
            _compiled_fn = _core_attn

    try:
        out, lse = _compiled_fn(q_flat, gathered_k, gathered_v, mask, q_valid)
    except Exception:
        _compiled_fn = _core_attn
        out, lse = _compiled_fn(q_flat, gathered_k, gathered_v, mask, q_valid)

    return (
        out.reshape(batch_size, num_heads, t_max, head_dim).to(torch.bfloat16),
        lse.reshape(batch_size, num_heads, t_max),
    )
