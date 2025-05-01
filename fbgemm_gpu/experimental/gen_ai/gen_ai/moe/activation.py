# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import torch
import triton
import triton.language as tl


# Function APIs
def silu_mul(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:

    assert x0.ndim == 2 and x0.stride(1) == 1
    assert x1.ndim == 2 and x1.stride(1) == 1
    assert x0.shape == x1.shape
    assert x0.dtype == x1.dtype

    T, D = x0.shape
    stride_0 = x0.stride(0)
    stride_1 = x1.stride(0)

    out = torch.empty((T, D), device="cuda", dtype=x0.dtype)

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    if T >= NUM_SMS:
        BLOCK_D_OUTER = D
        BLOCK_D_INNER = 1024
        assert D % BLOCK_D_INNER == 0
    else:
        BLOCK_D_OUTER = 512
        BLOCK_D_INNER = 256
        assert D % BLOCK_D_OUTER == 0
    grid = (T, D // BLOCK_D_OUTER)
    _fbgemm_silu_mul[grid](
        out,
        x0,
        x1,
        stride_0,
        stride_1,
        D,  # pyre-ignore
        BLOCK_D_OUTER,  # pyre-ignore
        BLOCK_D_INNER,  # pyre-ignore
    )
    return out


# Torch Custom Op Registrations
_SILU_MUL_OP_NAME = "fbgemm::silu_mul"

torch.library.define(
    "fbgemm::silu_mul",
    "(Tensor x0, Tensor x1) -> Tensor",
)


@torch.library.impl(_SILU_MUL_OP_NAME, "Meta")
def silu_mul_meta(x0, x1):
    return x0.new_empty(x0.shape)


@torch.library.impl(_SILU_MUL_OP_NAME, "CUDA")
def silu_mul_cuda(x0, x1):
    return silu_mul(x0, x1)


# Kernel Implementations
@triton.jit
def _fbgemm_silu_mul(
    y_ptr,
    x0_ptr,
    x1_ptr,
    stride_0,
    stride_1,
    D: tl.constexpr,
    BLOCK_D_OUTER: tl.constexpr,
    BLOCK_D_INNER: tl.constexpr,
) -> None:
    token_index = tl.program_id(0)
    feature_offset = tl.program_id(1) * BLOCK_D_OUTER + tl.arange(0, BLOCK_D_INNER)[:]

    for _ in range(0, BLOCK_D_OUTER // BLOCK_D_INNER):
        x0 = tl.load(
            x0_ptr + token_index * stride_0 + feature_offset,
            None,
            eviction_policy="evict_first",
        ).to(tl.float32)
        x1 = tl.load(
            x1_ptr + token_index * stride_1 + feature_offset,
            None,
            eviction_policy="evict_first",
        ).to(tl.float32)

        y = x0 * tl.sigmoid(x0) * x1

        tl.store(
            y_ptr + token_index * D + feature_offset,
            y,
            None,
        )
        feature_offset += BLOCK_D_INNER
