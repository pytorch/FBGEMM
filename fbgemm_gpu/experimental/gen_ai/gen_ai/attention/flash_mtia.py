# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors

from typing import Any, Iterable, Mapping, Optional, Set, Tuple

import torch

from .attn_bias import (
    BlockDiagonalCausalFromBottomRightMask,
    BlockDiagonalCausalLocalAttentionFromBottomRightMask,
    BlockDiagonalCausalLocalAttentionMask,
    BlockDiagonalCausalLocalAttentionPaddedKeysMask,
    BlockDiagonalCausalMask,
    BlockDiagonalCausalWithOffsetGappyKeysMask,
    BlockDiagonalCausalWithOffsetPaddedKeysMask,
    BlockDiagonalGappyKeysMask,
    BlockDiagonalLocalAttentionPaddedKeysMask,
    BlockDiagonalMask,
    BlockDiagonalPaddedKeysMask,
    LocalAttentionFromBottomRightMask,
    LowerTriangularFromBottomRightLocalAttentionMask,
    LowerTriangularFromBottomRightMask,
    LowerTriangularMask,
)

from .utils.op_utils import get_operator, register_operator

FLASH_VERSION = torch.nn.attention._get_flash_version()  # type: ignore
from .flash import BwOp as BwOpCUDA, FwOp as FwOpCUDA

try:
    import mtia.host_runtime.torch_mtia.dynamic_library  # noqa

    @torch.library.custom_op(
        "xformers_flash_mtia::flash_fwd",
        mutates_args=(),
        device_types=["mtia"],
    )
    def _flash_fwd(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens_q: Optional[torch.Tensor],
        cu_seqlens_k: Optional[torch.Tensor],
        seqused_k: Optional[torch.Tensor],
        max_seqlen_q: int,
        max_seqlen_k: int,
        p: float,
        softmax_scale: float,
        is_causal: bool,
        window_left: int,
        window_right: int,
        return_softmax: bool,
        block_tables: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ret = torch.ops.aten._flash_attention_forward(
            query,
            key,
            value,
            cu_seqlens_q,  # cum_seq_q
            cu_seqlens_k,  # cum_seq_k
            max_seqlen_q,  # max_q
            max_seqlen_k,  # max_k
            p,  # dropout_p
            is_causal,
            return_debug_mask=False,
            scale=softmax_scale,
            window_size_left=window_left,
            window_size_right=window_right,
            seqused_k=seqused_k,
            alibi_slopes=None,  # alibi_slopes
        )
        attention, logsumexp, rng_state, _, _ = ret
        return attention, logsumexp, rng_state

        @torch.library.register_fake("xformers_flash_mtia::flash_fwd")
        def _flash_fwd_abstract(
            query,
            key,
            value,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_k,
            max_seqlen_q,
            max_seqlen_k,
            p,
            softmax_scale,
            is_causal,
            window_left,
            window_right,
            return_softmax,
            block_tables,
        ):
            out = torch.empty_like(query)
            if cu_seqlens_q is None:
                B, M, H, K = query.shape
                lse_shape = [B, H, M]  # XXXX ?
            else:
                M, H, K = query.shape
                B = cu_seqlens_q.shape[0] - 1
                lse_shape = [H, M]
            softmax_lse = torch.empty(
                lse_shape, device=query.device, dtype=torch.float32
            )
            rng_state = torch.empty([2], device=query.device, dtype=torch.int64)
            return out, softmax_lse, rng_state

    @torch.library.custom_op(
        "xformers_flash_mtia::flash_bwd",
        mutates_args=(),
        device_types=["mtia"],
    )
    def _flash_bwd(
        grads_share_storage: bool,
        grad: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        out: torch.Tensor,
        lse: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        p: float,
        softmax_scale: float,
        is_causal: bool,
        window_left: int,
        window_right: int,
        rng_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rng_state0 = rng_state1 = rng_state
        dq, dk, dv = torch.ops.aten._flash_attention_backward(
            grad,
            query,
            key,
            value,
            out,
            lse,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            p,
            is_causal,
            rng_state0,
            rng_state1,
            scale=softmax_scale,
            window_size_left=window_left,
            window_size_right=window_right,
        )
        return dq, dk, dv

    @torch.library.register_fake("xformers_flash_mtia::flash_bwd")
    def _flash_bwd_abstract(
        grads_share_storage,
        grad,
        query,
        key,
        value,
        *args,
        **kwargs,
    ):
        return _create_dq_dk_dv(grads_share_storage, query, key, value)

except (ImportError, OSError):
    pass


def _create_dq_dk_dv(
    grads_share_storage: bool, query, key, value
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Create dq,dk,dv
    # If Q/K/V come from a single QKV tensor, let's put the gradient in the
    # right strides, so we can avoid a `cat`
    if grads_share_storage:
        chunk = torch.empty(
            (*query.shape[0:-2], 3, query.shape[-2], query.shape[-1]),
            dtype=query.dtype,
            device=query.device,
        )
        return chunk.select(-3, 0), chunk.select(-3, 1), chunk.select(-3, 2)
    return torch.empty_like(query), torch.empty_like(key), torch.empty_like(value)


@register_operator
class FwOp(FwOpCUDA):
    """Operator that computes memory-efficient attention using MTIA devicesa"""

    OPERATOR = get_operator("xformers_flash_mtia", "flash_fwd")
    SUPPORTED_DEVICES: Set[str] = {"mtia"}
    SUPPORTED_ATTN_BIAS_TYPES: Iterable[Any] = (
        type(None),
        LowerTriangularMask,
        LowerTriangularFromBottomRightMask,
        LowerTriangularFromBottomRightLocalAttentionMask,
        BlockDiagonalMask,
        BlockDiagonalCausalMask,
        BlockDiagonalCausalLocalAttentionMask,
        BlockDiagonalCausalLocalAttentionFromBottomRightMask,
        BlockDiagonalLocalAttentionPaddedKeysMask,
        BlockDiagonalCausalLocalAttentionPaddedKeysMask,
        BlockDiagonalCausalFromBottomRightMask,
        BlockDiagonalCausalWithOffsetGappyKeysMask,
        BlockDiagonalCausalWithOffsetPaddedKeysMask,
        BlockDiagonalGappyKeysMask,
        BlockDiagonalPaddedKeysMask,
        LocalAttentionFromBottomRightMask,
    )
    NAME = f"fa2F@{FLASH_VERSION}-mtia"

    ERROR_ATOL: Mapping[torch.dtype, float] = {
        torch.float: 3e-4,
        torch.half: 7e-3,
        torch.bfloat16: 2e-2,
    }
    ERROR_RTOL: Mapping[torch.dtype, float] = {
        torch.float: 2e-5,
        torch.half: 4e-4,
        torch.bfloat16: 5e-3,
    }
    REQUIRE_FBGEMM_KERNELS = False


@register_operator
class BwOp(BwOpCUDA):
    """Operator that computes memory-efficient attention using MTIA devicesa"""

    OPERATOR = get_operator("xformers_flash_mtia", "flash_bwd")
    SUPPORTED_DEVICES: Set[str] = {"mtia"}
    SUPPORTED_ATTN_BIAS_TYPES: Iterable[Any] = (
        type(None),
        LowerTriangularMask,
        LowerTriangularFromBottomRightMask,
        LowerTriangularFromBottomRightLocalAttentionMask,
        BlockDiagonalMask,
        BlockDiagonalCausalMask,
        BlockDiagonalCausalLocalAttentionMask,
        BlockDiagonalCausalLocalAttentionFromBottomRightMask,
        BlockDiagonalCausalFromBottomRightMask,
        LocalAttentionFromBottomRightMask,
    )
    NAME = f"fa2B@{FLASH_VERSION}-mtia"
    REQUIRE_FBGEMM_KERNELS = False
