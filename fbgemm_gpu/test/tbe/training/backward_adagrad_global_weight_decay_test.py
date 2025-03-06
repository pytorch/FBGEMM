#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

# pyre-ignore-all-errors[56]

import copy
import unittest
from typing import Any, Dict, List

import numpy as np
import torch

from fbgemm_gpu.split_embedding_utils import round_up
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    GlobalWeightDecayDefinition,
)

from hypothesis import given, settings

from .backward_adagrad_common import (  # noqa
    additional_decorators,
    adjust_mixed_B_st,
    common_settings,
    common_strategy,
    ComputeDevice,
    EmbeddingLocation,
    gen_mixed_B_batch_sizes,
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
    "pooling_mode": st.sampled_from([PoolingMode.SUM, PoolingMode.MEAN]),
    "start_iter": st.sampled_from([0, 10, 100, 1000]),
    "gwd_lower_bound": st.sampled_from([0, 0.01, 0.001]),
}


def compare_output(
    output_ref: torch.Tensor,
    output: torch.Tensor,
    is_fp32: bool,
) -> None:
    """
    This function compares two tensors and raise errors if they are not close, given
    a set tolerance value.
    Args:
        output_ref (Tensor): reference tensor
        output (Tensor): tensor to compare with
        is_fp32: whether the tensor is of type FP32
    Return:
        None
    """
    tolerance = 1.0e-4 if is_fp32 else 1.0e-2
    torch.testing.assert_close(
        output_ref.float(),
        output.float(),
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
    gwd_lower_bound: float,
    on_train_end: bool = False,
) -> torch.Tensor:
    """
    This function applies global weight decay (GWD) to the embedding weights of the
    given table and indices.
    Args:
        prev_iters (Tensor): the iteration in which each corresponding
            embedding row was last accessed
        weights (Tensor): embedding weights to be updated with global weight decay
        learning_rate (float): learning rate
        weight_decay (float): weight decay
        step (int): current iteration
        current_device (torch.device): current device
        weights_dtype (torch.dtype): weight dtype
        gwd_lower_bound (float): lower bound of global weight decay (i.e.,
            the global weight decay will not be lower than this value)
        on_train_end (bool): whether it is end of training
    """
    # GWD needed for the current iter is (1 - lr * weight_decay)^(N) where
    # N = step - prev_iters - 1
    # If on_train_end, we flush table with weight decay to compensate for
    # the missing iterations, N = step - prev_iters.
    extra_subtraction = 0 if on_train_end else 1
    gwd = (
        torch.pow(
            1 - learning_rate * weight_decay, step - prev_iters - extra_subtraction
        )
        .reshape((-1, 1))
        .to(device=current_device)
    )
    # Find indices for rows whose prev_iter = 0 (i.e., the first time the row is accessed)
    indices = (prev_iters == 0).nonzero(as_tuple=True)[0]
    # We set the GWD to 1 for these rows (i.e., GWD is not applied)
    gwd[indices] = 1

    # Ensure GWD does is not lower than gwd_lower_bound
    gwd[gwd < gwd_lower_bound] = gwd_lower_bound

    return (weights * gwd).to(dtype=weights_dtype)


def apply_gwd(
    T: int,
    Bs: List[int],
    emb: SplitTableBatchedEmbeddingBagsCodegen,
    prev_iter_dev: torch.Tensor,
    step: int,
    indices: torch.Tensor,
    offsets: torch.Tensor,
    weights_dtype: torch.dtype,
    gwd_lower_bound: float,
) -> None:
    """
    This function applies global weight decay for each embedding table.
    The corresponding embedding weights are updated in place.
    Args:
        T (int): number of tables
        Bs (List[int]): batch sizes for each table
        emb (SplitTableBatchedEmbeddingBagsCodegen): embedding table object
        prev_iter_dev (Tensor): the iteration in which each row was last accessed
        step (int): current iteration
        indices (Tensor): indices of the embedding table
        offsets (Tensor): offsets of the embedding table
        weights_dtype (torch.dtype): weight dtype
        gwd_lower_bound (float): lower bound of global weight decay (i.e.,
            the global weight decay will not be lower than this value)
    Return:
        None
    """
    weights = emb.split_embedding_weights()
    idx = 0
    for t in range(T):
        B = Bs[t]
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
                emb.optimizer_args.learning_rate_tensor.item(),
                emb.optimizer_args.weight_decay,
                step,
                emb.current_device,
                weights_dtype,
                gwd_lower_bound,
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
    mixed_B: bool,
    output_dtype: SparseType,
    weight_decay: float,
    start_iter: int,
    gwd_lower_bound: float,
    pooling_mode: PoolingMode,
) -> None:
    """
    Test global weight decay
    """
    weight_decay_mode = WeightDecayMode.DECOUPLE_GLOBAL
    E = int(10**log_E)
    D = D * 4
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
    if mixed_B:
        Bs_rank_feature, Bs = gen_mixed_B_batch_sizes(B, T)
    else:
        Bs = [B] * num_features
        Bs_rank_feature = None
    global_weight_decay = GlobalWeightDecayDefinition(
        start_iter=start_iter, lower_bound=gwd_lower_bound
    )
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
        global_weight_decay=global_weight_decay,
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
        weights[t].data.copy_(
            torch.randn(size=weights[t].shape, dtype=weights_precision.as_dtype())
        )
        weights_ref[t].data.copy_(weights[t].data)

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
        # Reset weights at each step to ensure the weights are the same for fair comparison
        # Otherwise the once-tolerable discrepencies can accumulate over iterations
        weights_ref = tbe_ref.split_embedding_weights()
        for t in range(T):
            weights_ref[t].data.copy_(weights[t].data)

        # Apply global weight decay for tbe_ref
        if uniq_indices.numel() > 0:
            # DECOUPLE MODE doesn't update prev_iter_dev (i.e., tbe_ref.prev_iter_dev won't be updated)
            if i >= start_iter:
                apply_gwd(
                    T,
                    Bs,
                    tbe_ref,
                    # pyre-fixme[6]: For 4th argument expected `Tensor` but got
                    #  `Union[Tensor, Module]`.
                    tbe.prev_iter_dev,
                    i,
                    indices,
                    offsets,
                    weights_precision.as_dtype(),
                    gwd_lower_bound,
                )
            if i != 1:
                tbe.iter_cpu.fill_(
                    i - 1
                )  # step will be incremented when forward is called
                tbe.iter.fill_(i - 1)

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
                # pyre-fixme[6]: For 1st argument expected `Tensor` but got
                #  `Union[Tensor, Module]`.
                tbe_ref.weights_dev,
                # pyre-fixme[6]: For 2nd argument expected `Tensor` but got
                #  `Union[Tensor, Module]`.
                tbe.weights_dev,
                is_fp32,
            )


@optests.generate_opcheck_tests(fast=True, additional_decorators=additional_decorators)
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
            mixed_B=False,
            **kwargs,
        )

    @unittest.skipIf(*gpu_unavailable)
    @given(
        weights_precision=st.sampled_from([SparseType.FP32, SparseType.FP16]),
        weight_decay=st.sampled_from([0.001, 0.01, 0.1]),
        **test_st,
    )
    @settings(**common_settings)
    def test_backward_adagrad_global_weight_decay_vbe(  # noqa C901
        self,
        weights_precision: SparseType,
        weight_decay: float,
        **kwargs: Any,
    ) -> None:
        """
        Test global weight decay with Rowwise Adagrad optimizers VBE
        """
        execute_global_weight_decay(
            weights_precision=weights_precision,
            weight_decay=weight_decay,
            mixed_B=True,
            **kwargs,
        )


if __name__ == "__main__":
    unittest.main()
