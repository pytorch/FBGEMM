# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# TLX IKBO flash attention for Blackwell (sm_100 / sm_103): entry point.
#
# Two kernels:
#   - tlx_ikbo_fa_blackwell_2cta: 2-CTA FA4 pipeline, 512-row pair tiles (default).
#   - tlx_ikbo_fa_blackwell_1cta: 1-CTA FA4 pipeline, 256-row tiles, for small users.
# A user's n_seed * num_cands Q rows (per head) are cut into tiles and the last one is partial. A 2-CTA pair tile is
# ~10% faster than two 1-CTA tiles (it halves each SM's K/V SMEM traffic) but pads to 512 rows, so it loses when users
# have few candidates. The plan compares the padded tile cost of both tilings (GB300, B=2048, H=2, n_seed=64: 2-CTA
# wins from ~16 candidates per user, 1-CTA at <= 10).
#
# GB300 (1350 W board power limit), upstream benchmark inputs (B=2048, H=2, n_seed=64, D=128, 64/65 candidates per
# user), TFLOP/s at max_seq_len 512 / 1024 / 4096 / 16384:
#   tlx_flash_attn_ikbo_blackwell   1400 / 1621 / 1662 / 1696
#   triton_flash_attn_ikbo_tma       501 /  623 /  766 /  801
#   SDPA (cuDNN) on pre-broadcast K/V 307 / 343 / 377 / 383
# At long sequences this is the board's power-limited tensor throughput (cuDNN's dense kernel on the same FLOPs
# reaches the same TF / pJ per FLOP).

from typing import NamedTuple, Optional

import torch

from .tlx_ikbo_fa_blackwell_1cta import ikbo_fa_1cta, tile_rows
from .tlx_ikbo_fa_blackwell_2cta import ikbo_fa_2cta

_SPEEDUP_2CTA = 1.10  # 2-CTA full pair tile vs two 1-CTA full tiles (measured 1.07-1.11, seq 512..16384)
_HALF_COST_2CTA = 0.65  # 2-CTA half pair tile (both groups on 256 rows, each over half of the KV) vs a full pair tile
_HALF_COST_1CTA = 0.80  # 1-CTA half tile vs a full tile
_TAILS_LAST_MIN_KV_BLOCKS = (
    8  # tails-last order + late CLC claim from this many 128-row KV blocks
)
_TAILS_LAST_CLC_LEAD = 4


class IkboFaPlan(NamedTuple):
    use_2cta: bool
    tile_table: torch.Tensor  # int32 [T, 4] = (q_row0, kv_row0, valid_rows, head)
    clc_lead: int
    # the layout the plan was built for (checked at call time)
    num_cands: int
    num_users: int
    n_seed: int
    num_heads: int
    max_seq_len: int


def _tile_cost(num_cands_per_user, n_seed, block_m, half_cost):
    cost = 0.0
    for c in num_cands_per_user:
        nq = c * n_seed
        cost += nq // block_m
        r = nq % block_m
        if r:
            cost += half_cost if r <= block_m // 2 else 1.0
    return cost


def plan_flash_attn_ikbo_blackwell(
    cand_to_user_index: torch.Tensor,
    n_seed: int,
    num_heads: int,
    max_seq_len: int,
    num_cands_per_user: Optional[torch.Tensor] = None,
) -> IkboFaPlan:
    """Pick the kernel and build its tile table for a candidate layout: cand_to_user_index [B] int32, sorted
    ascending (each user's candidates contiguous, users in order). Host-side (one device-to-host copy plus the table
    upload); build it once per layout and pass it as `plan=`, like the Hopper kernel's cand_grid.
    """
    assert cand_to_user_index.dim() == 1 and cand_to_user_index.numel() > 0
    assert n_seed % 64 == 0 and num_heads > 0 and max_seq_len > 0
    idx = cand_to_user_index
    B = idx.numel()
    if num_cands_per_user is None:
        num_cands_per_user = torch.bincount(idx.long())
    unsorted = (idx[1:] < idx[:-1]).any().view(1)
    counts_and_flag = [num_cands_per_user.to(idx.device, torch.long), unsorted.long()]
    host = torch.cat(counts_and_flag).tolist()  # one device-to-host copy
    counts, unsorted = host[:-1], host[-1]
    assert (
        not unsorted
    ), "cand_to_user_index must be sorted ascending (candidates grouped by user, users in order)"
    assert sum(counts) == B, "num_cands_per_user does not match cand_to_user_index"
    layout = (B, len(counts), n_seed, num_heads, max_seq_len)

    cost_2cta = 2 * _tile_cost(counts, n_seed, 512, _HALF_COST_2CTA) / _SPEEDUP_2CTA
    use_2cta = cost_2cta < _tile_cost(counts, n_seed, 256, _HALF_COST_1CTA)
    rows = tile_rows(counts, n_seed, num_heads, max_seq_len, 512 if use_2cta else 256)
    clc_lead = 0
    if use_2cta and (max_seq_len + 127) // 128 >= _TAILS_LAST_MIN_KV_BLOCKS:
        full = [r for r in rows if r[2] > 256]
        if len(full) < len(rows):
            # Run the full pair tiles first and the cheaper half tiles last, and claim the next tile a few KV blocks
            # before the end of the current one (+1.3..2.6% at seq 1024..16384 on 64/65-candidate users; with
            # fewer KV blocks the late claim costs more than it saves).
            rows = full + [r for r in rows if r[2] <= 256]
            clc_lead = _TAILS_LAST_CLC_LEAD
    table = torch.tensor(rows, dtype=torch.int32, device=idx.device)
    return IkboFaPlan(use_2cta, table, clc_lead, *layout)


def tlx_flash_attn_ikbo_blackwell(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cand_to_user_index: torch.Tensor,
    n_seed: int,
    max_seq_len: int,
    plan: Optional[IkboFaPlan] = None,
) -> torch.Tensor:
    """IKBO flash attention on Blackwell.

    query [B * n_seed, H, 128], key / value [U * max_seq_len, H, 128] (all fp16 or all bf16, contiguous),
    cand_to_user_index [B] int32 sorted ascending (each user's candidates contiguous). Returns [B * n_seed, H, 128].
    n_seed must be a multiple of 64. Without `plan` the plan is rebuilt on every call (host sync). A plan is only
    valid for the layout it was built from: B, U, n_seed, H and max_seq_len are checked, the per-user counts are not.

    As in the Hopper kernels, K / V are read as one [U * max_seq_len] sequence: a user's last KV block reads the next
    user's leading rows with zero probability, so an Inf / NaN in those V rows propagates.
    """
    assert query.dim() == 3 and key.dim() == 3 and key.shape == value.shape
    H, D = query.shape[1], query.shape[2]
    assert (
        D == 128 and key.shape[1] == H and key.shape[2] == D
    ), "head_dim 128, same heads for Q and K/V"
    assert query.dtype in (torch.float16, torch.bfloat16), "fp16 / bf16 only"
    assert key.dtype == query.dtype and value.dtype == query.dtype, "mixed dtypes"
    assert (
        n_seed % 64 == 0
    ), "n_seed must be a multiple of 64 (partial tiles store whole 64-row candidates)"
    assert query.is_contiguous() and key.is_contiguous() and value.is_contiguous()
    B = cand_to_user_index.numel()
    assert query.shape[0] == B * n_seed, "query must have B * n_seed rows"
    if plan is None:
        plan = plan_flash_attn_ikbo_blackwell(
            cand_to_user_index, n_seed, H, max_seq_len
        )
    assert (plan.num_cands, plan.n_seed, plan.num_heads, plan.max_seq_len) == (
        B,
        n_seed,
        H,
        max_seq_len,
    ), "plan was built for a different layout"
    assert (
        key.shape[0] >= plan.num_users * max_seq_len
    ), "key / value have fewer than U * max_seq_len rows"
    assert all(
        t.device == query.device for t in (key, value, plan.tile_table)
    ), "query, key, value and the plan must be on the same device"
    # Triton launches on the current device's current stream, so make the inputs' device current.
    with torch.cuda.device(query.device.index):
        if plan.use_2cta:
            return ikbo_fa_2cta(
                query, key, value, max_seq_len, plan.tile_table, clc_lead=plan.clc_lead
            )
        return ikbo_fa_1cta(query, key, value, max_seq_len, plan.tile_table)
