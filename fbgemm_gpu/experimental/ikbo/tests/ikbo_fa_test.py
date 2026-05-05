# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import random
import sys

import pytest
import torch
from ikbo.ops.tlx_ikbo_fa_ws import tlx_flash_attn_ikbo_tma_persistent
from ikbo.ops.triton_ikbo_fa import triton_flash_attn_ikbo_tma

DEVICE = "cuda"
DTYPE = torch.float16
DEFAULT_CAND_TO_USER_RATIO = 64


def pytorch_sdpa(query, key, value):
    return torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
    )


def broadcast_sdpa(
    query, key, value, cand_to_user_index, n_seed, num_heads, d_head, max_seq_len
):
    # for accuracy check
    query_sdpa = query.view(-1, n_seed, num_heads, d_head).permute(0, 2, 1, 3)
    key_sdpa = key.view(-1, max_seq_len, num_heads, d_head)
    key_sdpa_broadcast = torch.index_select(
        key_sdpa, dim=0, index=cand_to_user_index
    ).permute(0, 2, 1, 3)
    value_sdpa = value.view(-1, max_seq_len, num_heads, d_head)
    value_sdpa_broadcast = torch.index_select(
        value_sdpa, dim=0, index=cand_to_user_index
    ).permute(0, 2, 1, 3)
    return pytorch_sdpa(query_sdpa, key_sdpa_broadcast, value_sdpa_broadcast)


def prepare_inputs_by_config(
    B: int,
    n_seed: int,
    num_heads: int,
    d_head: int,
    max_seq_len: int,
    low_num_cands_per_user: int = DEFAULT_CAND_TO_USER_RATIO,
    high_num_cands_per_user: int = DEFAULT_CAND_TO_USER_RATIO,
):
    def _generate_num_cands_per_user():
        res = []
        cum_sum = 0
        cand_grid = []
        while True:
            # Odd and even number of candidates per user got even chance
            cur = random.randint(
                low_num_cands_per_user, high_num_cands_per_user
            ) + random.randint(0, 1)
            for grid in range(cum_sum, min(cum_sum + cur, B), 2):
                cand_grid.append(grid)
            if cum_sum + cur >= B:
                res.append(B - cum_sum)
                break
            cum_sum += cur
            res.append(cur)
        return res, cand_grid

    res = _generate_num_cands_per_user()
    num_cands_per_user_tensor = torch.tensor(res[0])
    cand_grid = torch.tensor(res[1], dtype=torch.int32, device=DEVICE)

    cand_to_user_index = torch.repeat_interleave(
        torch.arange(num_cands_per_user_tensor.size(0)),
        num_cands_per_user_tensor,
    ).to(dtype=torch.int32, device=DEVICE)
    Bu = num_cands_per_user_tensor.size(0)

    query = torch.randn((B * n_seed, num_heads, d_head), device=DEVICE, dtype=DTYPE)
    key = torch.randn((Bu * max_seq_len, num_heads, d_head), device=DEVICE, dtype=DTYPE)
    value = torch.randn(
        (Bu * max_seq_len, num_heads, d_head), device=DEVICE, dtype=DTYPE
    )
    return (
        query,
        key,
        value,
        cand_to_user_index,
        cand_grid,
    )


@pytest.mark.parametrize("B", [512, 1024, 2048])
@pytest.mark.parametrize("n_seed", [64])
@pytest.mark.parametrize("num_heads", [1, 2, 4, 6])
@pytest.mark.parametrize("d_head", [128])
@pytest.mark.parametrize("max_seq_len", [500, 512, 1000, 1024, 2000, 2048])
@pytest.mark.parametrize("cand_to_user_ratio", [10, 70])
def test_triton_ikbo_fa(B, n_seed, num_heads, d_head, max_seq_len, cand_to_user_ratio):
    query, key, value, cand_to_user_index, cand_grid = prepare_inputs_by_config(
        B,
        n_seed,
        num_heads,
        d_head,
        max_seq_len,
        cand_to_user_ratio,
        cand_to_user_ratio,
    )
    triton_output = (
        triton_flash_attn_ikbo_tma(
            query, key, value, cand_to_user_index, n_seed, max_seq_len
        )
        .view(B, n_seed, num_heads, d_head)
        .permute(0, 2, 1, 3)
    )
    torch_output = broadcast_sdpa(
        query, key, value, cand_to_user_index, n_seed, num_heads, d_head, max_seq_len
    )
    torch.testing.assert_close(torch_output, triton_output, atol=1e-3, rtol=1e-4)


@pytest.mark.parametrize("B", [512, 1024, 2048])
@pytest.mark.parametrize("n_seed", [64])
@pytest.mark.parametrize("num_heads", [1, 2, 4, 6])
@pytest.mark.parametrize("d_head", [128])
@pytest.mark.parametrize("max_seq_len", [500, 512, 1000, 1024, 2000, 2048])
@pytest.mark.parametrize("cand_to_user_ratio", [10, 70])
def test_tlx_ikbo_fa(B, n_seed, num_heads, d_head, max_seq_len, cand_to_user_ratio):
    query, key, value, cand_to_user_index, cand_grid = prepare_inputs_by_config(
        B,
        n_seed,
        num_heads,
        d_head,
        max_seq_len,
        cand_to_user_ratio,
        cand_to_user_ratio,
    )
    tlx_output = (
        tlx_flash_attn_ikbo_tma_persistent(
            query, key, value, cand_to_user_index, n_seed, max_seq_len, cand_grid
        )
        .view(B, n_seed, num_heads, d_head)
        .permute(0, 2, 1, 3)
    )
    torch_output = broadcast_sdpa(
        query, key, value, cand_to_user_index, n_seed, num_heads, d_head, max_seq_len
    )
    torch.testing.assert_close(torch_output, tlx_output, atol=1e-3, rtol=1e-4)


def main():
    sys.exit(pytest.main([__file__, "-v"]))


if __name__ == "__main__":
    main()
