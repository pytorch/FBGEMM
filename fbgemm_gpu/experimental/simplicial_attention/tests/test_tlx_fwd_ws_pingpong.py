# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


import sys

import pytest
import torch

from simplicial.ops.tlx.fwd_ws_pingpong import tlx_fwd_ws_pingpong
from simplicial.ops.triton.fwd import triton_fwd as triton_fwd

from simplicial.utils import assert_diff, compute_sqnr


@pytest.fixture(autouse=True)
def deterministic_seed():
    """Ensure reproducibility for every test."""
    torch.manual_seed(7)


@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("N", [512, 1024, 2028, 8192])
@pytest.mark.parametrize("H", [128])
@pytest.mark.parametrize("D", [128])
@pytest.mark.parametrize("w1", [16, 32])
@pytest.mark.parametrize("w2", [128, 256, 512])
def test_tlx_fwd_ws_pipelined(B, N, H, D, w1, w2):
    device = torch.accelerator.current_accelerator()

    Hkv = 1
    print(f"Testing B={B}, N={N}, H={H}, D={D}, w1={w1} w2={w2}")

    # Create normalized tensors as required
    Q = torch.randn(B, N, H, D, device=device, dtype=torch.bfloat16).normal_(
        mean=0.0, std=0.5
    )
    K1 = torch.randn(B, N, Hkv, D, device=device, dtype=torch.bfloat16).normal_(
        mean=0.0, std=0.5
    )
    K2 = torch.randn_like(K1).normal_(mean=0.0, std=0.5)
    V1 = torch.randn_like(K1).normal_(mean=0.0, std=0.5)
    V2 = torch.randn_like(K1).normal_(mean=0.0, std=0.5)

    # Normalize tensors
    Q = Q / torch.norm(Q, dim=-1, keepdim=True)
    K1 = K1 / torch.norm(K1, dim=-1, keepdim=True)
    K2 = K2 / torch.norm(K2, dim=-1, keepdim=True)
    V1 = V1 / torch.norm(V1, dim=-1, keepdim=True)
    V2 = V2 / torch.norm(V2, dim=-1, keepdim=True)

    # Test the pipelined tlx_fwd_ws_pingpong kernel
    out, m = tlx_fwd_ws_pingpong(Q, K1, K2, V1, V2, w1=w1, w2=w2)

    # Ground truth implementation
    out_ref, m_ref = triton_fwd(
        Q,
        K1,
        K2,
        V1,
        V2,
        w1=w1,
        w2=w2,
    )

    # Compute SQNR loss vs reference
    sqnr_ref = compute_sqnr(out, out_ref)
    assert (
        sqnr_ref > 40.0
    ), f"SQNR vs reference should be larger than 30.0. Got: {sqnr_ref}"
    assert_diff(out, out_ref)
    assert_diff(m, m_ref)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
