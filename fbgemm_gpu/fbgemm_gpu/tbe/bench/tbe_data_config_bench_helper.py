#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import List, Optional, Tuple

import torch

from fbgemm_gpu.tbe.bench.tbe_data_config import TBEDataConfig
from fbgemm_gpu.tbe.utils.common import get_device, round_up

from fbgemm_gpu.tbe.utils.requests import (
    generate_batch_sizes_from_stats,
    generate_pooling_factors_from_stats,
    get_table_batched_offsets_from_dense,
    maybe_to_dtype,
    TBERequest,
)

try:
    # pyre-ignore[21]
    from fbgemm_gpu import open_source  # noqa: F401
except Exception:
    torch.ops.load_library(
        "//deeplearning/fbgemm/fbgemm_gpu/src/tbe/eeg:indices_generator"
    )


def _generate_batch_sizes(
    tbe_data_config: TBEDataConfig,
) -> Tuple[List[int], Optional[List[List[int]]]]:
    if tbe_data_config.variable_B():
        assert (
            tbe_data_config.batch_params.vbe_num_ranks is not None
        ), "vbe_num_ranks must be set for varaible batch size generation"
        return generate_batch_sizes_from_stats(
            tbe_data_config.batch_params.B,
            tbe_data_config.T,
            # pyre-ignore [6]
            tbe_data_config.batch_params.sigma_B,
            tbe_data_config.batch_params.vbe_num_ranks,
            # pyre-ignore [6]
            tbe_data_config.batch_params.vbe_distribution,
        )

    else:
        return ([tbe_data_config.batch_params.B] * tbe_data_config.T, None)


def _generate_pooling_info(
    tbe_data_config: TBEDataConfig, iters: int, Bs: List[int]
) -> torch.Tensor:
    if tbe_data_config.variable_L():
        # Generate L from stats
        _, L_offsets = generate_pooling_factors_from_stats(
            iters,
            Bs,
            tbe_data_config.pooling_params.L,
            # pyre-ignore [6]
            tbe_data_config.pooling_params.sigma_L,
            # pyre-ignore [6]
            tbe_data_config.pooling_params.length_distribution,
        )
    else:
        Ls = [tbe_data_config.pooling_params.L] * (sum(Bs) * iters)
        L_offsets = torch.tensor([0] + Ls, dtype=torch.long).cumsum(0)

    return L_offsets


def _generate_indices(
    tbe_data_config: TBEDataConfig,
    iters: int,
    Bs: List[int],
    L_offsets: torch.Tensor,
) -> torch.Tensor:

    total_B = sum(Bs)
    L_offsets_list = L_offsets.tolist()
    indices_list = []
    for it in range(iters):
        # L_offsets is defined over the entire set of batches for a single iteration
        start_offset = L_offsets_list[it * total_B]
        end_offset = L_offsets_list[(it + 1) * total_B]

        indices_list.append(
            torch.ops.fbgemm.tbe_generate_indices_from_distribution(
                tbe_data_config.indices_params.heavy_hitters,
                tbe_data_config.indices_params.zipf_q,
                tbe_data_config.indices_params.zipf_s,
                # max_index = dimensions of the embedding table
                tbe_data_config.E,
                # num_indices = number of indices to generate
                end_offset - start_offset,
            )
        )

    return torch.cat(indices_list)


def _build_requests_jagged(
    tbe_data_config: TBEDataConfig,
    iters: int,
    Bs: List[int],
    Bs_feature_rank: Optional[List[List[int]]],
    L_offsets: torch.Tensor,
    all_indices: torch.Tensor,
) -> List[TBERequest]:
    total_B = sum(Bs)
    all_indices = all_indices.flatten()
    requests = []
    for it in range(iters):
        start_offset = L_offsets[it * total_B]
        it_L_offsets = torch.concat(
            [
                torch.zeros(1, dtype=L_offsets.dtype, device=L_offsets.device),
                L_offsets[it * total_B + 1 : (it + 1) * total_B + 1] - start_offset,
            ]
        )
        requests.append(
            TBERequest(
                maybe_to_dtype(
                    all_indices[start_offset : L_offsets[(it + 1) * total_B]],
                    tbe_data_config.indices_params.index_dtype,
                ),
                maybe_to_dtype(
                    it_L_offsets.to(get_device()),
                    tbe_data_config.indices_params.offset_dtype,
                ),
                tbe_data_config._new_weights(int(it_L_offsets[-1].item())),
                Bs_feature_rank if tbe_data_config.variable_B() else None,
            )
        )
    return requests


def _build_requests_dense(
    tbe_data_config: TBEDataConfig, iters: int, all_indices: torch.Tensor
) -> List[TBERequest]:
    # NOTE: We're using existing code from requests.py to build the
    # requests, and since the existing code requires 2D view of all_indices,
    # the existing all_indices must be reshaped
    all_indices = all_indices.reshape(iters, -1)

    requests = []
    for it in range(iters):
        indices, offsets = get_table_batched_offsets_from_dense(
            all_indices[it].view(
                tbe_data_config.T,
                tbe_data_config.batch_params.B,
                tbe_data_config.pooling_params.L,
            ),
            use_cpu=tbe_data_config.use_cpu,
        )
        requests.append(
            TBERequest(
                maybe_to_dtype(indices, tbe_data_config.indices_params.index_dtype),
                maybe_to_dtype(offsets, tbe_data_config.indices_params.offset_dtype),
                tbe_data_config._new_weights(
                    tbe_data_config.T
                    * tbe_data_config.batch_params.B
                    * tbe_data_config.pooling_params.L
                ),
            )
        )
    return requests


def generate_requests(
    tbe_data_config: TBEDataConfig,
    iters: int = 1,
) -> List[TBERequest]:
    # Generate batch sizes
    Bs, Bs_feature_rank = _generate_batch_sizes(tbe_data_config)

    # Generate pooling info
    L_offsets = _generate_pooling_info(tbe_data_config, iters, Bs)

    # Generate indices
    all_indices = _generate_indices(tbe_data_config, iters, Bs, L_offsets)

    # Build TBE requests
    if tbe_data_config.variable_B() or tbe_data_config.variable_L():
        return _build_requests_jagged(
            tbe_data_config, iters, Bs, Bs_feature_rank, L_offsets, all_indices
        )
    else:
        return _build_requests_dense(tbe_data_config, iters, all_indices)


def generate_embedding_dims(tbe_data_config: TBEDataConfig) -> Tuple[int, List[int]]:
    if tbe_data_config.mixed_dim:
        Ds = [
            round_up(
                int(
                    torch.randint(
                        low=int(0.5 * tbe_data_config.D),
                        high=int(1.5 * tbe_data_config.D),
                        size=(1,),
                    ).item()
                ),
                4,
            )
            for _ in range(tbe_data_config.T)
        ]
        return (sum(Ds) // len(Ds), Ds)
    else:
        return (tbe_data_config.D, [tbe_data_config.D] * tbe_data_config.T)


def generate_feature_requires_grad(
    tbe_data_config: TBEDataConfig, size: int
) -> torch.Tensor:
    assert (
        size <= tbe_data_config.T
    ), "size of feature_requires_grad must be less than T"
    weighted_requires_grad_tables = torch.randperm(tbe_data_config.T)[:size].tolist()
    return (
        torch.tensor(
            [
                1 if t in weighted_requires_grad_tables else 0
                for t in range(tbe_data_config.T)
            ]
        )
        .to(get_device())
        .int()
    )
