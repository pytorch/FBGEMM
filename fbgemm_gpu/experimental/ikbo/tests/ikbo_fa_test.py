# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import sys

import pytest
import torch
from ikbo.benchmarks.ikbo_fa_bench import broadcast_sdpa, prepare_inputs_by_config
from ikbo.ops.tlx_ikbo_fa_ws import tlx_flash_attn_ikbo_tma_persistent
from ikbo.ops.triton_ikbo_fa import triton_flash_attn_ikbo_tma

DEVICE = "cuda"
DTYPE = torch.float16


@pytest.mark.parametrize("B", [512, 102, 2048])
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


@pytest.mark.parametrize("B", [512, 102, 2048])
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
