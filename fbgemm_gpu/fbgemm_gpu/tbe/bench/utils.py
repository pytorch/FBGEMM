# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import List, Tuple

import numpy as np
import torch

# fmt:skip
from fbgemm_gpu.split_embedding_configs import SparseType


def fill_random_scale_bias(
    emb: torch.nn.Module,
    T: int,
    weights_precision: SparseType,
) -> None:
    for t in range(T):
        # pyre-fixme[29]: `Union[Module, Tensor]` is not a function.
        weights, scale_shift = emb.split_embedding_weights()[t]
        if scale_shift is not None:
            E, R = scale_shift.shape
            assert R == 4
            scales = None
            shifts = None
            if weights_precision == SparseType.INT8:
                scales = np.random.uniform(0.001, 0.01, size=(E,)).astype(np.float16)
                shifts = np.random.normal(-2, 2, size=(E,)).astype(np.float16)
            elif weights_precision == SparseType.INT4:
                scales = np.random.uniform(0.01, 0.1, size=(E,)).astype(np.float16)
                shifts = np.random.normal(-2, 2, size=(E,)).astype(np.float16)
            elif weights_precision == SparseType.INT2:
                scales = np.random.uniform(0.1, 1, size=(E,)).astype(np.float16)
                shifts = np.random.normal(-2, 2, size=(E,)).astype(np.float16)
            scale_shift.copy_(
                torch.tensor(
                    np.stack([scales, shifts], axis=1)
                    .astype(np.float16)
                    .view(np.uint8),
                    device=scale_shift.device,
                )
            )


def check_oom(
    data_size: int,
) -> Tuple[bool, str]:
    free_memory, total_memory = torch.cuda.mem_get_info()
    if data_size > free_memory:
        warning = f"Expect to allocate {round(data_size / (1024 ** 3), 2)} GB, but available memory is {round(free_memory / (1024 ** 3), 2)} GB from {round(total_memory / (1024 ** 3), 2)} GB."
        return (True, warning)
    return (False, "")


def generate_batch_size_per_feature_per_rank(
    Bs: List[int], num_ranks: int
) -> List[List[int]]:
    """
    Generate batch size per feature per rank for VBE, assuming the batch size
    is evenly distributed across ranks.
    Args:
        Bs (List[int]): batch size per feature
        num_ranks (int): number of ranks
    Returns:
        List[List[int]]: batch size per feature per rank
    """
    b_per_feature_per_rank = []
    for B in Bs:
        b_per_feature = []
        for i in range(num_ranks):
            if i != num_ranks - 1:
                b_per_feature.append(int(B / num_ranks))
            else:
                b_per_feature.append(B - sum(b_per_feature))
        b_per_feature_per_rank.append(b_per_feature)
    return b_per_feature_per_rank


def generate_merged_output_and_offsets(
    Ds: List[int],
    Bs: List[int],
    output_dtype: torch.dtype,
    device: torch.device,
    num_ranks: int = 2,
    num_tbe_ops: int = 2,
) -> Tuple[List[List[int]], torch.Tensor, torch.Tensor]:
    """
    Generate merged vbe_output and vbe_output_offsets tensors for VBE.
    The vbe_output is a tensor that will contain forward output from all VBE TBE ops.
    The vbe_output_offsets is a tensor that will contain start offsets for the output to be written to.

    Args:
        Ds (List[int]): embedding dimension per feature
        Bs (List[int]): batch size per feature
        num_ranks (int): number of ranks
        num_tbe_ops (int): number of TBE ops
    Returns:
        Tuple[List[List[int]], torch.Tensor, torch.Tensor]: batch_size_per_feature_per_rank, merged vbe_output and vbe_output_offsets tensors
    """
    # The first embedding ops is the embedding op created in the benchmark
    emb_op = {}
    emb_op[0] = {}
    emb_op[0]["dim"] = Ds
    emb_op[0]["Bs"] = Bs
    emb_op[0]["output_size"] = sum([b * d for b, d in zip(Bs, Ds)])
    emb_op[0]["batch_size_per_feature_per_rank"] = (
        generate_batch_size_per_feature_per_rank(Bs, num_ranks)
    )
    num_features = len(Bs)
    # create other embedding ops to allocate output and offsets tensors
    # Using representative values for additional TBE ops in multi-op scenarios:
    # - batch_size=32000: typical large batch size for production workloads
    # - dim=512: common embedding dimension for large models
    for i in range(1, num_tbe_ops):
        emb_op[i] = {}
        emb_op[i]["batch_size_per_feature_per_rank"] = (
            generate_batch_size_per_feature_per_rank([32000], num_ranks)
        )
        emb_op[i]["Bs"] = [sum(B) for B in emb_op[i]["batch_size_per_feature_per_rank"]]
        emb_op[i]["dim"] = [512]
        emb_op[i]["output_size"] = sum(
            [b * d for b, d in zip(emb_op[i]["Bs"], emb_op[i]["dim"])]
        )
    total_output = 0
    ranks = [[] for _ in range(num_ranks)]
    for e in emb_op.values():
        b_per_rank_per_feature = list(zip(*e["batch_size_per_feature_per_rank"]))
        assert len(b_per_rank_per_feature) == num_ranks
        dims = e["dim"]
        for r, b_r in enumerate(b_per_rank_per_feature):
            for f, b in enumerate(b_r):
                output_size_per_batch = b * dims[f]
                ranks[r].append(output_size_per_batch)
                total_output += output_size_per_batch
    ranks[0].insert(0, 0)
    offsets_ranks: List[List[int]] = [[] for _ in range(num_ranks)]
    total_output_offsets = []
    start = 0
    for r in range(num_ranks):
        offsets_ranks[r] = [
            start + sum(ranks[r][: i + 1]) for i in range(len(ranks[r]))
        ]
        start = offsets_ranks[r][-1]
        total_output_offsets.extend(offsets_ranks[r])
    check_total_output_size = sum([e["output_size"] for e in emb_op.values()])
    assert (
        total_output == check_total_output_size
    ), f"{total_output} != {check_total_output_size}{[e['output_size'] for e in emb_op.values()]}"
    assert (
        total_output == total_output_offsets[-1]
    ), f"{total_output} != {total_output_offsets[-1]}"
    out = torch.empty(total_output, dtype=output_dtype, device=device)
    offsets = []
    offsets.append(offsets_ranks[0][:num_features])
    for r in range(1, num_ranks):
        start = [offsets_ranks[r - 1][-1]]
        the_rest = offsets_ranks[r][: num_features - 1] if num_features > 1 else []
        start.extend(the_rest)
        offsets.append(start)

    out_offsets = torch.tensor(
        offsets,
        dtype=torch.int64,
        device=device,
    )
    batch_size_per_feature_per_rank = emb_op[0]["batch_size_per_feature_per_rank"]
    return (batch_size_per_feature_per_rank, out, out_offsets)
