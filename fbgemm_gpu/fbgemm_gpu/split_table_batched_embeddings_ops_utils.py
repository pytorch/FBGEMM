# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple

import fbgemm_gpu  # noqa: F401
import torch  # usort:skip

from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    DEFAULT_SCALE_BIAS_SIZE_IN_BYTES,
    EmbeddingLocation,
    round_up,
    SplitState,
)


def rounded_row_size_in_bytes(
    dim: int,
    weight_ty: SparseType,
    row_alignment: int,
    scale_bias_size_in_bytes: int = DEFAULT_SCALE_BIAS_SIZE_IN_BYTES,
) -> int:
    r = unpadded_row_size_in_bytes(dim, weight_ty, scale_bias_size_in_bytes)
    # align each row to 16-byte boundaries.
    return round_up(r, row_alignment)


def unpadded_row_size_in_bytes(
    dim: int,
    weight_ty: SparseType,
    scale_bias_size_in_bytes: int = DEFAULT_SCALE_BIAS_SIZE_IN_BYTES,
) -> int:
    r = {
        SparseType.FP32.value: dim * 4,
        SparseType.FP16.value: dim * 2,
        SparseType.FP8.value: dim,
        SparseType.INT8.value: dim + scale_bias_size_in_bytes,
        SparseType.INT4.value: dim // 2 + scale_bias_size_in_bytes,
        SparseType.INT2.value: dim // 4 + scale_bias_size_in_bytes,
    }[weight_ty.value]
    return r


def align_to_cacheline(a: int) -> int:
    # align each table to 128b cache line boundary.
    return round_up(a, 128)


def nbit_construct_split_state(
    embedding_specs: List[Tuple[str, int, int, SparseType, EmbeddingLocation]],
    cacheable: bool,
    row_alignment: int,
    scale_bias_size_in_bytes: int = DEFAULT_SCALE_BIAS_SIZE_IN_BYTES,
    cacheline_alignment: bool = True,
) -> SplitState:
    placements = torch.jit.annotate(List[EmbeddingLocation], [])
    offsets = torch.jit.annotate(List[int], [])
    dev_size = 0
    host_size = 0
    uvm_size = 0
    for _, num_embeddings, embedding_dim, weight_ty, location in embedding_specs:
        embedding_dim = rounded_row_size_in_bytes(
            embedding_dim, weight_ty, row_alignment, scale_bias_size_in_bytes
        )
        state_size = num_embeddings * embedding_dim
        if cacheline_alignment:
            state_size = align_to_cacheline(state_size)
        if location == EmbeddingLocation.HOST:
            placements.append(EmbeddingLocation.HOST)
            offsets.append(host_size)
            host_size += state_size
        elif location == EmbeddingLocation.DEVICE or location == EmbeddingLocation.MTIA:
            placements.append(location)
            offsets.append(dev_size)
            dev_size += state_size
        else:
            if cacheable and location == EmbeddingLocation.MANAGED_CACHING:
                placements.append(EmbeddingLocation.MANAGED_CACHING)
            else:
                placements.append(EmbeddingLocation.MANAGED)
            offsets.append(uvm_size)
            uvm_size += state_size
    assert len(placements) == len(offsets)
    return SplitState(
        dev_size=dev_size,
        host_size=host_size,
        uvm_size=uvm_size,
        placements=placements,
        offsets=offsets,
    )


def random_quant_scaled_tensor(
    shape: torch.Size,
    device: torch.device,
    output_tensor: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if output_tensor is not None:
        return torch.randint(
            0,
            255,
            size=shape,
            out=output_tensor,
            dtype=torch.uint8,
            device=device,
        )
    else:
        return torch.randint(
            0,
            255,
            size=shape,
            dtype=torch.uint8,
            device=device,
        )


@torch.fx.wrap
def inputs_to_device(
    indices: torch.Tensor,
    offsets: torch.Tensor,
    per_sample_weights: Optional[torch.Tensor],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if device.type == "meta":
        return indices, offsets, per_sample_weights

    non_blocking = device.type != "cpu"
    if indices.device != device:
        indices = indices.to(device, non_blocking=non_blocking)
    if offsets.device != device:
        offsets = offsets.to(device, non_blocking=non_blocking)
    if per_sample_weights is not None and per_sample_weights.device != device:
        per_sample_weights = per_sample_weights.to(device, non_blocking=non_blocking)
    return indices, offsets, per_sample_weights
