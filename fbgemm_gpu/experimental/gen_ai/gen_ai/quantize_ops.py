# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Optional, Tuple

import torch

"""
This file contains manual shape registrations for quantize custom operators.
These are needed for custom operators to be compatible with torch.compile.

In some cases, fake tensor handling can be done by registering a meta implementation
directly in cpp. However, for more complicated functions such as those that involve
cross device synchronization, pytorch requires a full fake implementation be registered
in python.
"""


@torch.library.register_fake("fbgemm::f8f8bf16_blockwise")
def f8f8bf16_blockwise_abstract(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    block_m: int = 128,
    block_n: int = 128,
    block_k: int = 128,
) -> torch.Tensor:
    M = XQ.shape[0]
    N = WQ.shape[0]
    return torch.empty(
        [M, N],
        dtype=torch.bfloat16,
        device=XQ.device,
    )


@torch.library.register_fake("fbgemm::f8f8bf16_tensorwise")
def f8f8bf16_tensorwise_abstract(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    scale: float,
    use_fast_accum: bool = True,
) -> torch.Tensor:
    M = XQ.shape[0]
    N = WQ.shape[0]
    return torch.empty(
        [M, N],
        dtype=torch.bfloat16,
        device=XQ.device,
    )


@torch.library.register_fake("fbgemm::f8f8bf16_rowwise")
def f8f8bf16_rowwise_abstract(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    use_fast_accum: bool = True,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    M = XQ.shape[0]
    N = WQ.shape[0]
    return torch.empty(
        [M, N],
        dtype=torch.bfloat16,
        device=XQ.device,
    )


@torch.library.register_fake("fbgemm::quantize_fp8_per_tensor")
def quantize_fp8_per_tensor_abstract(
    input: torch.Tensor,
    bs: Optional[torch.Tensor] = None,
    scale_ub: Optional[torch.Tensor] = None,
    stochastic_rounding: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if torch.version.hip:
        fp8_dtype = torch.float8_e4m3fnuz
    else:
        fp8_dtype = torch.float8_e4m3fn
    output = torch.empty_like(input, dtype=fp8_dtype, device=input.device)
    scale = torch.empty([], dtype=torch.bfloat16, device=input.device)
    return output, scale


@torch.library.register_fake("fbgemm::quantize_fp8_per_row")
def quantize_fp8_per_row_abstract(
    input: torch.Tensor,
    bs: Optional[torch.Tensor] = None,
    scale_ub: Optional[torch.Tensor] = None,
    output_dtype: Optional[torch.dtype] = None,
    stochastic_rounding: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if torch.version.hip:
        fp8_dtype = torch.float8_e4m3fnuz
    else:
        fp8_dtype = torch.float8_e4m3fn
    output = torch.empty_like(input, dtype=fp8_dtype, device=input.device)
    scale = torch.empty([], dtype=torch.bfloat16, device=input.device)
    return output, scale


@torch.library.register_fake("fbgemm::quantize_fp8_per_col")
def quantize_fp8_per_col_abstract(
    input: torch.Tensor,
    bs: Optional[torch.Tensor] = None,
    scale_ub: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if torch.version.hip:
        fp8_dtype = torch.float8_e4m3fnuz
    else:
        fp8_dtype = torch.float8_e4m3fn
    output = torch.empty_like(input, dtype=fp8_dtype, device=input.device)
    scale = torch.empty([], dtype=torch.bfloat16, device=input.device)
    return output, scale


# The following operators are not supported on AMD.
if not torch.version.hip:

    @torch.library.register_fake("fbgemm::i8i8bf16")
    def i8i8bf16_abstract(
        XQ: torch.Tensor,
        WQ: torch.Tensor,
        scale: float,
        split_k: int = 1,
    ) -> torch.Tensor:
        M = XQ.shape[0]
        N = WQ.shape[0]
        return torch.empty(
            [M, N],
            dtype=torch.bfloat16,
            device=XQ.device,
        )

    @torch.library.register_fake("fbgemm::f8f8bf16")
    def f8f8bf16_abstract(
        XQ: torch.Tensor,
        WQ: torch.Tensor,
        scale: torch.Tensor,
        use_fast_accum: bool = True,
    ) -> torch.Tensor:
        M = XQ.shape[0]
        N = WQ.shape[0]
        return torch.empty(
            [M, N],
            dtype=torch.bfloat16,
            device=XQ.device,
        )

    @torch.library.register_fake("fbgemm::f8f8bf16_cublas")
    def f8f8bf16_cublas_abstract(
        A: torch.Tensor,
        B: torch.Tensor,
        Ainvs: Optional[torch.Tensor] = None,
        Binvs: Optional[torch.Tensor] = None,
        use_fast_accum: bool = True,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        M = A.shape[0]
        N = B.shape[0]
        return torch.empty(
            [M, N],
            dtype=torch.bfloat16,
            device=A.device,
        )

    @torch.library.register_fake("fbgemm::f8f8bf16_rowwise_batched")
    def f8f8bf16_rowwise_batched_abstract(
        XQ: torch.Tensor,
        WQ: torch.Tensor,
        x_scale: torch.Tensor,
        w_scale: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        use_fast_accum: bool = True,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        M = XQ.shape[0]
        N = WQ.shape[0]
        return torch.empty(
            [M, N],
            dtype=torch.bfloat16,
            device=XQ.device,
        )

    @torch.library.register_fake("fbgemm::f8i4bf16_rowwise")
    def f8i4bf16_rowwise_abstract(
        XQ: torch.Tensor,
        WQ: torch.Tensor,
        x_scale: torch.Tensor,
        w_scale: torch.Tensor,
        w_zp: torch.Tensor,
    ) -> torch.Tensor:
        M = XQ.shape[0]
        N = WQ.shape[0]
        return torch.empty(
            [M, N],
            dtype=torch.bfloat16,
            device=XQ.device,
        )

    @torch.library.register_fake("fbgemm::bf16i4bf16_rowwise")
    def bf16i4bf16_rowwise_abstract(
        X: torch.Tensor,
        WQ: torch.Tensor,
        w_scale: torch.Tensor,
        w_zp: torch.Tensor,
    ) -> torch.Tensor:
        M = X.shape[0]
        N = WQ.shape[0]
        return torch.empty(
            [M, N],
            dtype=torch.bfloat16,
            device=X.device,
        )

    @torch.library.register_fake("fbgemm::bf16i4bf16_rowwise_batched")
    def bf16i4bf16_rowwise_batched_abstract(
        X: torch.Tensor,
        WQ: torch.Tensor,
        w_scale: torch.Tensor,
        w_zp: torch.Tensor,
    ) -> torch.Tensor:
        M = X.shape[0]
        N = WQ.shape[0]
        return torch.empty(
            [M, N],
            dtype=torch.bfloat16,
            device=X.device,
        )
