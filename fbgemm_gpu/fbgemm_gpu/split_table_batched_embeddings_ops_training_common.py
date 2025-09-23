#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Optional

import torch
from torch import Tensor

try:
    try:
        from torch.compiler import is_compiling

        def is_torchdynamo_compiling() -> bool:  # type: ignore[misc]
            # at least one test fails if we import is_compiling as a different name
            return is_compiling()

    except Exception:
        # torch.compiler.is_compiling is not available in torch 1.10
        from torch._dynamo import is_compiling as is_torchdynamo_compiling
except Exception:

    def is_torchdynamo_compiling() -> bool:  # type: ignore[misc]
        return False


# @manual=//deeplearning/fbgemm/fbgemm_gpu/codegen:split_embedding_codegen_lookup_invokers
import fbgemm_gpu.split_embedding_codegen_lookup_invokers as invokers
from fbgemm_gpu.split_table_batched_embeddings_ops_common import PoolingMode


def generate_vbe_metadata(
    offsets: Tensor,
    batch_size_per_feature_per_rank: Optional[list[list[int]]],
    pooling_mode: PoolingMode,
    feature_dims_cpu: Tensor,
    device: torch.device,
) -> invokers.lookup_args.VBEMetadata:
    """
    Generate VBE metadata based on batch_size_per_feature_per_rank.
    Metadata includes:
        1) B_offsets - A tensor that contains batch size offsets for each
                        feature
        2) output_offsets_feature_rank - A tensor that contains output
                                            offsets for each feature
        3) B_offsets_per_rank_per_feature - A tensor that contains batch
                                            size offsets for each feature
                                            and rank
        4) max_B - The maximum batch size for all features
        5) max_B_feature_rank - The maximum batch size for all ranks and
                                features
        6) output_size - The output size (number of elements)
    """
    if batch_size_per_feature_per_rank is not None:
        assert (
            pooling_mode != PoolingMode.NONE
        ), "Variable batch size TBE support is not enabled for PoolingMode.NONE"
        # TODO: Add input check
        zero_tensor = torch.zeros(1, device="cpu", dtype=torch.int32)

        # Create B offsets
        total_batch_size_per_feature = torch.tensor(
            batch_size_per_feature_per_rank, dtype=torch.int32, device="cpu"
        ).sum(dim=1)

        max_B = total_batch_size_per_feature.max().item()
        if not torch.jit.is_scripting() and is_torchdynamo_compiling():
            torch._check_is_size(max_B)
            torch._check(max_B < offsets.numel())

        Bs = torch.concat([zero_tensor, total_batch_size_per_feature])
        B_offsets = Bs.cumsum(dim=0).to(torch.int)

        # Create output offsets
        B_feature_rank = torch.tensor(
            batch_size_per_feature_per_rank,
            device="cpu",
            dtype=torch.int64,
        )
        max_B_feature_rank = B_feature_rank.max().item()
        if not torch.jit.is_scripting() and is_torchdynamo_compiling():
            torch._check_is_size(max_B_feature_rank)
            torch._check(max_B_feature_rank <= offsets.size(0))
        output_sizes_feature_rank = B_feature_rank.transpose(
            0, 1
        ) * feature_dims_cpu.view(1, -1)
        output_offsets_feature_rank = torch.concat(
            [
                zero_tensor.to(torch.int64),
                output_sizes_feature_rank.flatten().cumsum(dim=0),
            ]
        )
        output_size = output_offsets_feature_rank[-1].item()
        if not torch.jit.is_scripting() and is_torchdynamo_compiling():
            torch._check_is_size(output_size)

        # TODO: Support INT8 output
        # B_offsets_rank_per_feature is for rank and (b, t) mapping
        B_offsets_rank_per_feature = (
            torch.tensor(
                [
                    [0] + batch_size_per_feature
                    for batch_size_per_feature in batch_size_per_feature_per_rank
                ],
                device="cpu",
                dtype=torch.int32,
            )
            .cumsum(dim=1)
            .to(torch.int)
        )

        B_offsets = B_offsets.to(device, non_blocking=True)
        output_offsets_feature_rank = output_offsets_feature_rank.to(
            device, non_blocking=True
        )
        B_offsets_rank_per_feature = B_offsets_rank_per_feature.to(
            device, non_blocking=True
        )

        # TODO: Use int32 for B_offsets and int64 for output_offsets_feature_rank
        vbe_metadata = invokers.lookup_args.VBEMetadata(
            B_offsets=B_offsets,
            output_offsets_feature_rank=output_offsets_feature_rank,
            B_offsets_rank_per_feature=B_offsets_rank_per_feature,
            # pyre-ignore
            max_B=max_B,
            # pyre-ignore
            max_B_feature_rank=max_B_feature_rank,
            # pyre-ignore
            output_size=output_size,
        )
    else:
        vbe_metadata = invokers.lookup_args.VBEMetadata(
            B_offsets=None,
            output_offsets_feature_rank=None,
            B_offsets_rank_per_feature=None,
            max_B=-1,
            max_B_feature_rank=-1,
            output_size=-1,
        )
    return vbe_metadata
