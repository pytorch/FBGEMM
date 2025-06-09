#!/usr/bin/env python3
# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2024, NVIDIA Corporation & AFFILIATES.

# pyre-strict

from typing import Any, Optional, Tuple

import torch


class HstuAttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(  # pyre-ignore[14]
        ctx,  # pyre-ignore[2]
        q: torch.Tensor,  # need grad
        k: torch.Tensor,  # need grad
        v: torch.Tensor,  # need grad
        seq_offsets_q: torch.Tensor,
        seq_offsets_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        num_contexts: torch.Tensor,
        num_targets: torch.Tensor,
        target_group_size: int,
        window_size: Tuple[int, int] = (-1, -1),
        alpha: float = 1.0,
        rab: Optional[torch.Tensor] = None,  # need grad
        has_drab: bool = False,
        is_delta_q: bool = False,
        descale_q: Optional[torch.Tensor] = None,
        descale_k: Optional[torch.Tensor] = None,
        descale_v: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert q.dim() == 3, "q shape should be (L, num_heads, head_dim)"
        assert k.dim() == 3, "k shape should be (L, num_heads, head_dim)"
        assert v.dim() == 3, "v shape should be (L, num_heads, hidden_dim)"

        major_version = torch.cuda.get_device_capability()[0]
        assert major_version == 8 or major_version == 9, "Only support sm80 and sm90"
        if major_version == 8:
            out, rab_padded = torch.ops.fbgemm.hstu_varlen_fwd_80(
                q,
                k,
                v,
                seq_offsets_q,
                seq_offsets_k,
                max_seqlen_q,
                max_seqlen_k,
                num_contexts,
                num_targets,
                target_group_size,
                window_size[0],
                window_size[1],
                alpha,
                rab,
                is_delta_q,
            )
        else:
            out, rab_padded = torch.ops.fbgemm.hstu_varlen_fwd_90(
                q,
                k,
                v,
                seq_offsets_q,
                seq_offsets_k,
                max_seqlen_q,
                max_seqlen_k,
                num_contexts,
                num_targets,
                target_group_size,
                window_size[0],
                window_size[1],
                alpha,
                rab,
                is_delta_q,
                descale_q,
                descale_k,
                descale_v,
            )

        ctx.save_for_backward(
            q,
            k,
            v,
            seq_offsets_q,
            seq_offsets_k,
            num_contexts,
            num_targets,
            rab_padded,
        )
        ctx.major_version = major_version
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.target_group_size = target_group_size
        ctx.alpha = alpha
        ctx.window_size_left = window_size[0]
        ctx.window_size_right = window_size[1]
        ctx.has_drab = has_drab
        ctx.is_delta_q = is_delta_q
        return out

    @staticmethod
    def backward(  # pyre-ignore[14]
        ctx,  # pyre-ignore[2]
        dout: torch.Tensor,
        *args: Any,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        torch.Tensor,
        None,
        None,
        None,
        None,
        None,
    ]:
        (
            q,
            k,
            v,
            seq_offsets_q,
            seq_offsets_k,
            num_contexts,
            num_targets,
            rab_padded,
        ) = ctx.saved_tensors

        max_seqlen_q = ctx.max_seqlen_q
        max_seqlen_k = ctx.max_seqlen_k
        target_group_size = ctx.target_group_size
        window_size_left = ctx.window_size_left
        window_size_right = ctx.window_size_right
        alpha = ctx.alpha
        has_drab = ctx.has_drab
        is_delta_q = ctx.is_delta_q

        if ctx.major_version == 8:
            dq, dk, dv, dRab = torch.ops.fbgemm.hstu_varlen_bwd_80(
                dout,
                q,
                k,
                v,
                seq_offsets_q,
                seq_offsets_k,
                max_seqlen_q,
                max_seqlen_k,
                num_contexts,
                num_targets,
                target_group_size,
                window_size_left,
                window_size_right,
                alpha,
                rab_padded,
                has_drab,
                is_delta_q,
                False,  # deterministic
            )
        else:
            dq, dk, dv, dRab = torch.ops.fbgemm.hstu_varlen_bwd_90(
                dout,
                q,
                k,
                v,
                seq_offsets_q,
                seq_offsets_k,
                max_seqlen_q,
                max_seqlen_k,
                num_contexts,
                num_targets,
                target_group_size,
                window_size_left,
                window_size_right,
                alpha,
                rab_padded,
                has_drab,
                is_delta_q,
                False,  # deterministic
            )

        # q & k grad shape
        return (
            dq,
            dk,
            dv,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            dRab if ctx.has_drab else None,
            None,
            None,
            None,
            None,
            None,
        )


# pyre-ignore[3]
def hstu_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets_q: torch.Tensor,
    seq_offsets_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    num_contexts: torch.Tensor,
    num_targets: torch.Tensor,
    target_group_size: int = 1,
    window_size: Tuple[int, int] = (-1, -1),
    alpha: float = 1.0,
    rab: Optional[torch.Tensor] = None,
    has_drab: bool = False,
    is_delta_q: bool = False,
    descale_q: Optional[torch.Tensor] = None,
    descale_k: Optional[torch.Tensor] = None,
    descale_v: Optional[torch.Tensor] = None,
):
    """
    Arguments:
        q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
        k: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        v: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        seq_offsets_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into q.
        seq_offsets_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into kv.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        max_seqlen_k: int. Maximum key sequence length in the batch.
        num_contexts: (batch_size,). Number of context tokens in each batch.
        num_targets: (batch_size,). Number of target tokens in each batch.
        target_group_size: int. Number of target tokens in each group.
        window_size: (left, right). If not (-1, -1), implements sliding window local attention. If (-1, 0), implements causal attention.
        alpha: float. Scaling factor between add rab and silu.
        rab: (batch_size, max_seqlen_k, max_seqlen_k). Random access bias for the key.
        has_drab: bool. Whether to apply random access bias for the key.
        is_delta_q: bool. Whether to apply delta query.
        descale_q: (1,). Descaling factor for the query.
        descale_k: (1,). Descaling factor for the key.
        descale_v: (1,). Descaling factor for the value.
    Return:
        out: (total, nheads, headdim).
    """
    if has_drab and (rab is None):
        raise ValueError(
            "AssertError: rab is None, but has_drab is True, is not allowed in backward"
        )
    if num_contexts is not None and window_size != (-1, 0):
        raise ValueError(
            "AssertError: context is True and causal is not True, this is undefined behavior"
        )
    if num_targets is not None and window_size != (-1, 0):
        raise ValueError(
            "AssertError: target is True and causal is not True, this is undefined behavior"
        )
    if (num_contexts is not None and is_delta_q is True) or (
        num_targets is not None and is_delta_q is True
    ):
        raise ValueError(
            "AssertError: delta_q is True, but num_contexts or num_targets is not None, this is undefined behavior"
        )
    if num_targets is None and target_group_size < 1:
        raise ValueError(
            "AssertError: target_group_size should be greater than 0 when target is True"
        )
    if max_seqlen_q < max_seqlen_k and is_delta_q is False:
        raise ValueError(
            "AssertError: seq_len_q < seq_len_k, is_delta_q should be True, as is_delta_q represents mask behavior under the case"
        )
    if max_seqlen_q > max_seqlen_k:
        raise ValueError(
            "AssertError: seq_len_q >= seq_len_k, this is undefined behavior"
        )

    return HstuAttnVarlenFunc.apply(
        q,
        k,
        v,
        seq_offsets_q,
        seq_offsets_k,
        max_seqlen_q,
        max_seqlen_k,
        num_contexts,
        num_targets,
        target_group_size,
        window_size,
        alpha,
        rab,
        has_drab,
        is_delta_q,
        descale_q,
        descale_k,
        descale_v,
    )
