#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# pyre-ignore-all-errors[56]

import copy
import unittest
from typing import Any, Dict, List, Union

import numpy as np
import torch

from fbgemm_gpu.split_embedding_utils import round_up

from hypothesis import given, settings

from .backward_adagrad_common import (  # noqa
    adjust_mixed_B_st,
    common_settings,
    common_strategy,
    ComputeDevice,
    EmbeddingLocation,
    get_table_batched_offsets_from_dense,
    gpu_unavailable,
    optests,
    OptimType,
    PoolingMode,
    PoolingMode,
    skipIfRocm,
    SparseType,
    SplitTableBatchedEmbeddingBagsCodegen,
    st,
    WeightDecayMode,
)


# Set up test strategy
test_st: Dict[str, Any] = {
    "T": st.integers(min_value=1, max_value=5),
    "D": st.integers(min_value=2, max_value=128),
    "B": st.integers(min_value=1, max_value=128),
    "log_E": st.integers(min_value=3, max_value=5),
    "L": st.integers(min_value=0, max_value=20),
    "D_gradcheck": st.integers(min_value=1, max_value=2),
    "weighted": st.booleans(),
    "mixed": st.booleans(),
    "output_dtype": st.sampled_from(
        [SparseType.FP32, SparseType.FP16, SparseType.BF16]
    ),
}


def compare_output(
    output_ref: Union[torch.Tensor, List[torch.Tensor]],
    output: Union[torch.Tensor, List[torch.Tensor]],
    is_fp32: bool,
) -> None:
    tolerance = 1.0e-4 if is_fp32 else 1.0e-2

    torch.testing.assert_close(
        output_ref,
        output,
        atol=tolerance,
        rtol=tolerance,
    )


def apply_gwd_per_table(
    prev_iters: torch.Tensor,
    weights: torch.Tensor,
    learning_rate: float,
    weight_decay: float,
    step: int,
    current_device: torch.device,
    weights_dtype: torch.dtype,
    on_train_end: bool,
) -> torch.Tensor:
    x = 1
    if on_train_end:
        x = 0
    return (
        weights
        * (
            torch.pow(
                1 - learning_rate * weight_decay,
                step - prev_iters - x,
            )
            .reshape((-1, 1))
            .to(device=current_device)
        )
    ).to(dtype=weights_dtype)


def apply_gwd(
    T: int,
    B: int,
    emb: SplitTableBatchedEmbeddingBagsCodegen,
    prev_iter_dev: torch.Tensor,
    step: int,
    indices: torch.Tensor,
    offsets: torch.Tensor,
    weights_dtype: torch.dtype,
) -> None:
    weights = emb.split_embedding_weights()
    idx = 0
    for t in range(T):
        start_idx = offsets[idx]
        idx += B
        end_idx = offsets[idx]
        uniq_indices = indices[start_idx:end_idx].unique()
        # get prev_iter values
        prev_iter_indices = uniq_indices + emb.hash_size_cumsum[t]
        prev_iter_values = torch.index_select(
            prev_iter_dev, 0, prev_iter_indices.long()
        )
        # get weights and scale
        weights_values = torch.index_select(
            weights[emb.feature_table_map[t]], 0, uniq_indices.long()
        )
        # scale weights
        weights[emb.feature_table_map[t]].index_copy_(
            0,
            uniq_indices.long(),
            apply_gwd_per_table(
                prev_iter_values,
                weights_values,
                emb.optimizer_args.learning_rate,
                emb.optimizer_args.weight_decay,
                step,
                emb.current_device,
                weights_dtype,
                False,
            ),
        )
        start_idx = end_idx


def execute_global_weight_decay(  # noqa C901
    T: int,
    D: int,
    B: int,
    log_E: int,
    L: int,
    D_gradcheck: int,
    weights_precision: SparseType,
    weighted: bool,
    mixed: bool,
    output_dtype: SparseType,
    weight_decay: float,
) -> None:
    """
    Test global weight decay
    """
    weight_decay_mode = WeightDecayMode.DECOUPLE_GLOBAL
    E = int(10**log_E)
    D = D * 4
    pooling_mode = PoolingMode.SUM
    managed_option = EmbeddingLocation.DEVICE
    optimizer = OptimType.EXACT_ROWWISE_ADAGRAD

    is_fp32 = weights_precision == SparseType.FP32 and output_dtype == SparseType.FP32

    if not mixed:
        Ds = [D] * T
        Es = [E] * T
    else:
        Ds = [
            round_up(np.random.randint(low=int(0.25 * D), high=int(1.0 * D)), 4)
            for _ in range(T)
        ]
        Es = [np.random.randint(low=int(0.5 * E), high=int(2.0 * E)) for _ in range(T)]

    feature_table_map = list(range(T))
    num_features = len(feature_table_map)
    Bs = [B] * num_features
    Bs_rank_feature = None
    tbe = SplitTableBatchedEmbeddingBagsCodegen(
        embedding_specs=[
            (E, D, managed_option, ComputeDevice.CUDA) for (E, D) in zip(Es, Ds)
        ],
        optimizer=optimizer,
        learning_rate=0.1,
        eps=0.1,
        weights_precision=weights_precision,
        output_dtype=output_dtype,
        pooling_mode=pooling_mode,
        weight_decay=weight_decay,
        weight_decay_mode=weight_decay_mode,
    )
    device = torch.device("cuda")
    weights = tbe.split_embedding_weights()

    # Compare output of forward and weights between
    # 1) using DECOUPLE_GLOBAL mode (which gwd applies in forward
    # and weights updated in backward) with
    # 2) using DECOUPLE mode and apply gwd to update weights
    # before calling forward
    # the output should be the same
    tbe_ref = SplitTableBatchedEmbeddingBagsCodegen(
        embedding_specs=[
            (E, D, managed_option, ComputeDevice.CUDA) for (E, D) in zip(Es, Ds)
        ],
        optimizer=optimizer,
        learning_rate=0.1,
        eps=0.1,
        weights_precision=weights_precision,
        output_dtype=output_dtype,
        pooling_mode=pooling_mode,
        weight_decay=weight_decay,
        weight_decay_mode=WeightDecayMode.DECOUPLE,
    )
    weights_ref = tbe_ref.split_embedding_weights()

    # Initialize weights
    for t in range(T):
        weights[t] = torch.randn(size=(Es[t], D), dtype=weights_precision.as_dtype())
        weights_ref[t] = weights[t]

    xs = [
        torch.from_numpy(
            np.random.choice(range(Es[t]), size=(b, L), replace=True).astype(np.int64)
        ).to(device)
        for t, b in zip(feature_table_map, Bs)
    ]

    x = torch.cat([x.contiguous().flatten() for x in xs], dim=0)
    (indices, offsets) = get_table_batched_offsets_from_dense(x, L=L, total_B=sum(Bs))
    indices = indices.to(device)
    offsets = offsets.to(device)

    uniq_indices = indices.unique()  # indices are all 1's
    steps = [1, 2, 3, 10, 11, 15, 100, 1000, 10000]
    for i in steps:
        if uniq_indices.numel() > 0:
            # DECOUPLE MODE doesn't update prev_iter_dev (i.e., tbe_ref.prev_iter_dev won't be updated)
            apply_gwd(
                T,
                B,
                tbe_ref,
                tbe.prev_iter_dev,
                i,
                indices,
                offsets,
                weights_precision.as_dtype(),
            )
            if i != 1:
                tbe.step = i - 1  # step will be incremented when forward is called

            # Run forward pass
            output = tbe(
                indices, offsets, batch_size_per_feature_per_rank=Bs_rank_feature
            )
            # compare forward output
            output_ref = tbe_ref(
                indices, offsets, batch_size_per_feature_per_rank=Bs_rank_feature
            )
            compare_output(output_ref, output, is_fp32)
            # Run backward pass
            grad = torch.randn_like(output).to(device)
            grad_ref = copy.deepcopy(grad)
            output.backward(grad)
            # compare weights
            output_ref.backward(grad_ref)
            compare_output(
                tbe_ref.split_embedding_weights(),
                tbe.split_embedding_weights(),
                is_fp32,
            )


@optests.generate_opcheck_tests(fast=True)
class BackwardAdagradGlobalWeightDecay(unittest.TestCase):
    @unittest.skipIf(*gpu_unavailable)
    @given(
        weights_precision=st.sampled_from([SparseType.FP32, SparseType.FP16]),
        weight_decay=st.sampled_from([0.001, 0.01, 0.1]),
        **test_st,
    )
    @settings(**common_settings)
    def test_backward_adagrad_global_weight_decay(  # noqa C901
        self,
        weights_precision: SparseType,
        weight_decay: float,
        **kwargs: Any,
    ) -> None:
        """
        Test global weight decay with Rowwise Adagrad optimizers
        """
        execute_global_weight_decay(
            weights_precision=weights_precision,
            weight_decay=weight_decay,
            **kwargs,
        )


if __name__ == "__main__":
    unittest.main()
