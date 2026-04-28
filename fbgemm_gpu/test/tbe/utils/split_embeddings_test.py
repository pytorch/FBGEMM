#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# pyre-ignore-all-errors[56]

import random
import unittest
from typing import Callable

import hypothesis.strategies as st
import numpy as np
import torch
from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType, SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import EmbeddingLocation
from fbgemm_gpu.split_table_batched_embeddings_ops_inference import (
    IntNBitTableBatchedEmbeddingBagsCodegen,
    rounded_row_size_in_bytes,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    _get_consumed_preallocated_keys,
    ComputeDevice,
    construct_split_state,
    INT8_EMB_ROW_DIM_OFFSET,
    SplitTableBatchedEmbeddingBagsCodegen,
)
from fbgemm_gpu.tbe.utils import (
    get_table_batched_offsets_from_dense,
    round_up,
    to_device,
)
from hypothesis import given, settings, Verbosity
from torch import Tensor

from .. import common  # noqa E402
from ..common import MAX_EXAMPLES, open_source

if open_source:
    # pyre-ignore[21]
    from test_utils import gpu_unavailable, optests, use_cpu_strategy
else:
    from fbgemm_gpu.test.test_utils import gpu_unavailable, optests, use_cpu_strategy


VERBOSITY: Verbosity = Verbosity.verbose


# pyre-ignore
additional_decorators: dict[str, list[Callable]] = {}


@optests.generate_opcheck_tests(fast=True, additional_decorators=additional_decorators)
class SplitTableBatchedEmbeddingsTest(unittest.TestCase):
    @unittest.skipIf(*gpu_unavailable)
    def test_split_embedding_codegen_forward(  # noqa C901
        self,
    ) -> None:
        # Dummy test in order to run generated opcheck tests on
        # split_embedding_codegen_forward_weighted_cuda and
        # split_embedding_codegen_forward_unweighted_cuda.
        # Sizes and values of int tensors were generated from running
        # one test instance of test_backward_adagrad_fp16_pmSUM and outputting
        # sizes/dtypes/values.
        def _do_test(weighted: bool) -> None:
            flatten_dev_weights = torch.rand(0, dtype=torch.half).cuda()
            uvm_weights = torch.rand(10456, dtype=torch.half).cuda()
            lxu_cache_weights = torch.rand(544, 4, dtype=torch.float).cuda()
            weights_placements = torch.tensor([2, 2, 2]).to(
                dtype=torch.int, device=torch.accelerator.current_accelerator()
            )
            weights_offsets = torch.tensor([0, 2784, 2784]).to(
                dtype=torch.long, device=torch.accelerator.current_accelerator()
            )
            D_offsets = torch.tensor([0, 4, 8]).to(
                dtype=torch.int, device=torch.accelerator.current_accelerator()
            )
            total_D = 12
            max_D = 4
            indices = torch.tensor(
                [
                    680,
                    213,
                    293,
                    439,
                    1004,
                    885,
                    986,
                    1162,
                    433,
                    1327,
                    187,
                    89,
                ]
            ).to(dtype=torch.long, device=torch.accelerator.current_accelerator())
            offsets = torch.tensor([0, 2, 4, 6, 8, 10, 12]).to(
                dtype=torch.long, device=torch.accelerator.current_accelerator()
            )
            pooling_mode = False
            indice_weights = torch.rand(12, dtype=torch.float).cuda()
            lxu_cache_locations = torch.tensor(
                [
                    224,
                    352,
                    353,
                    192,
                    384,
                    64,
                    288,
                    1,
                    96,
                    194,
                    0,
                    193,
                ]
            ).to(dtype=torch.int, device=torch.accelerator.current_accelerator())
            uvm_cache_stats = torch.tensor(
                [
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                ]
            ).to(dtype=torch.int, device=torch.accelerator.current_accelerator())

            output_dtype = 0  # SparseType.FP32
            is_experimental = False

            op_args = [
                flatten_dev_weights,
                uvm_weights,
                lxu_cache_weights,
                weights_placements,
                weights_offsets,
                D_offsets,
                total_D,
                max_D,
                indices,
                offsets,
                pooling_mode,
            ]
            if weighted:
                op_args += [indice_weights]
            op_args += [
                lxu_cache_locations,
                uvm_cache_stats,
                output_dtype,
                is_experimental,
            ]

            op_name = "split_embedding_codegen_forward"
            op_name += "_weighted" if weighted else "_unweighted"
            op_name += "_cuda"

            getattr(torch.ops.fbgemm, op_name)(*op_args)

        _do_test(True)
        _do_test(False)

    def test_get_table_name_for_logging(self) -> None:
        self.assertEqual(
            SplitTableBatchedEmbeddingBagsCodegen.get_table_name_for_logging(None),
            "<Unknown>",
        )
        self.assertEqual(
            SplitTableBatchedEmbeddingBagsCodegen.get_table_name_for_logging(["t1"]),
            "t1",
        )
        self.assertEqual(
            SplitTableBatchedEmbeddingBagsCodegen.get_table_name_for_logging(
                ["t1", "t1"]
            ),
            "t1",
        )
        self.assertEqual(
            SplitTableBatchedEmbeddingBagsCodegen.get_table_name_for_logging(
                ["t1", "t2"]
            ),
            "<2 tables>: ['t1', 't2']",
        )
        self.assertEqual(
            SplitTableBatchedEmbeddingBagsCodegen.get_table_name_for_logging(
                ["t1", "t2", "t1"]
            ),
            "<2 tables>: ['t1', 't2']",
        )
        self.assertEqual(
            SplitTableBatchedEmbeddingBagsCodegen.get_table_name_for_logging([]),
            "<0 tables>: []",
        )

    @unittest.skipIf(*gpu_unavailable)
    @given(N=st.integers(min_value=1, max_value=2))
    @settings(verbosity=VERBOSITY, max_examples=MAX_EXAMPLES, deadline=None)
    def test_stb_uvm_cache_stats(self, N: int) -> None:
        # Create an abstract split table
        D = 8
        T = 2
        E = 10**3
        Ds = [D] * T
        Es = [E] * T
        emb_op = SplitTableBatchedEmbeddingBagsCodegen
        cc = emb_op(
            embedding_specs=[
                (
                    E,
                    D,
                    EmbeddingLocation.MANAGED_CACHING,
                    ComputeDevice.CUDA,
                )
                for (E, D) in zip(Es, Ds)
            ],
            gather_uvm_cache_stats=True,
        )

        x = torch.Tensor([[[1], [1]], [[3], [4]]])
        x = to_device(torch.tensor(x, dtype=torch.int64), use_cpu=False)

        for _ in range(N):
            indices, offsets = get_table_batched_offsets_from_dense(x, use_cpu=False)
            cc.reset_cache_states()
            cc.reset_uvm_cache_stats()
            cc(indices, offsets)
            (
                n_calls,
                n_requested_indices,
                n_unique_indices,
                n_unique_misses,
                n_conflict_unique_misses,
                n_conflict_misses,
            ) = cc.get_uvm_cache_stats()
            self.assertEqual(n_calls, 1)
            self.assertEqual(n_requested_indices, len(indices))
            self.assertEqual(n_unique_indices, len(set(indices.tolist())))
            self.assertEqual(n_unique_misses, len(set(indices.tolist())))
            self.assertEqual(n_conflict_unique_misses, 0)
            self.assertEqual(n_conflict_misses, 0)

    @unittest.skipIf(*gpu_unavailable)
    @given(
        T=st.integers(min_value=1, max_value=5),
        D=st.integers(min_value=2, max_value=64),
        log_E=st.integers(min_value=2, max_value=3),
        N=st.integers(min_value=0, max_value=50),
        weights_ty=st.sampled_from(
            [
                SparseType.FP32,
                SparseType.FP16,
                SparseType.INT8,
                SparseType.INT4,
                SparseType.INT2,
            ]
        ),
        output_dtype=st.sampled_from(
            [
                SparseType.FP32,
                SparseType.FP16,
                SparseType.INT8,
            ]
        ),
        use_cpu=use_cpu_strategy(),
        test_internal=st.booleans(),
    )
    @settings(verbosity=VERBOSITY, max_examples=MAX_EXAMPLES, deadline=None)
    def test_embedding_inplace_update(
        self,
        T: int,  # num of embedding tables
        D: int,  # embedding dim
        log_E: int,  # embedding table row number
        N: int,  # num of update rows per table
        weights_ty: SparseType,
        output_dtype: SparseType,
        use_cpu: bool,
        test_internal: bool,  # test with OSS op or internal customized op
    ) -> None:
        D_alignment = max(weights_ty.align_size(), output_dtype.align_size())
        D = round_up(D, D_alignment)
        Ds = [
            round_up(
                np.random.randint(low=int(max(0.25 * D, 1)), high=int(1.0 * D)),
                D_alignment,
            )
            for _ in range(T)
        ]
        E = int(10**log_E)
        Es = [np.random.randint(low=int(0.5 * E), high=int(2.0 * E)) for _ in range(T)]
        row_alignment = 1 if use_cpu else 16
        current_device = "cpu" if use_cpu else torch.cuda.current_device()
        location = EmbeddingLocation.HOST if use_cpu else EmbeddingLocation.DEVICE

        weights_ty_list = [weights_ty] * T
        if open_source:
            test_internal = False

        # create two embedding bag op with random weights
        locations = [location] * T
        op = IntNBitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                ("", E, D, W_TY, L)
                for (E, D, W_TY, L) in zip(Es, Ds, weights_ty_list, locations)
            ],
            output_dtype=output_dtype,
            device=current_device,
        )
        op.fill_random_weights()
        op_ref = IntNBitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                ("", E, D, W_TY, L)
                for (E, D, W_TY, L) in zip(Es, Ds, weights_ty_list, locations)
            ],
            output_dtype=output_dtype,
            device=current_device,
        )
        op_ref.fill_random_weights()

        # randomly generate update table and row indices
        update_table_indices = []
        update_table_indices2 = []
        update_row_indices = []
        update_row_indices2 = []
        for t in range(T):
            n = np.random.randint(low=0, high=N) if N > 0 else 0
            if n == 0:
                continue
            update_table_indices.append(t)
            update_row_id_list = random.sample(range(Es[t]), n)
            update_row_indices.append(update_row_id_list)
            update_table_indices2.extend([t] * n)
            update_row_indices2.extend(update_row_id_list)

        # generate update tensor based on weights from "op_ref" embedding table
        update_weights_list = []
        ref_split_weights = op_ref.split_embedding_weights(split_scale_shifts=False)

        update_weight_size = sum(
            [
                rounded_row_size_in_bytes(
                    Ds[t],
                    weights_ty_list[t],
                    row_alignment,
                )
                for t in update_table_indices2
            ]
        )
        update_weights_tensor2 = torch.randint(
            low=0,
            high=255,
            size=(update_weight_size,),
            dtype=torch.uint8,
            device=current_device,
        )

        update_offsets = 0
        for i in range(len(update_table_indices)):
            table_idx = update_table_indices[i]
            ref_weights, _ = ref_split_weights[table_idx]

            D_bytes = rounded_row_size_in_bytes(
                Ds[table_idx], weights_ty_list[table_idx], row_alignment
            )

            update_weights = []
            for row_idx in update_row_indices[i]:
                update_weights.append(ref_weights[row_idx].tolist())
                # fmt: off
                update_weights_tensor2[update_offsets : update_offsets + D_bytes] = (
                    ref_weights[row_idx]
                )
                # fmt: on
                update_offsets += D_bytes

            update_weights_tensor = torch.tensor(
                update_weights,
                device=current_device,
                dtype=torch.uint8,
            )
            update_weights_list.append(update_weights_tensor)

        # run inplace update on "op" embedding table
        if not test_internal:
            # Test scatter_ based OSS solution
            op.embedding_inplace_update(
                update_table_indices,
                update_row_indices,
                update_weights_list,
            )
        else:
            # Test customized op
            op.embedding_inplace_update_internal(
                update_table_indices2,
                update_row_indices2,
                update_weights_tensor2,
            )

        # verify weights are equal with "op_ref" for the updated rows in "op"
        split_weights = op.split_embedding_weights(split_scale_shifts=False)
        for i in range(len(update_table_indices)):
            t = update_table_indices[i]
            for r in update_row_indices[i]:
                weights, _ = split_weights[t]
                ref_weights, _ = ref_split_weights[t]
                self.assertEqual(weights.size(), ref_weights.size())
                torch.testing.assert_close(
                    weights[r],
                    ref_weights[r],
                    rtol=1e-2,
                    atol=1e-2,
                    equal_nan=True,
                )

    @unittest.skipIf(*gpu_unavailable)
    @given(
        T=st.integers(min_value=1, max_value=5),
        D=st.integers(min_value=2, max_value=128),
        log_E=st.integers(min_value=2, max_value=3),
        weights_precision=st.sampled_from([SparseType.FP16, SparseType.FP32]),
        mixed=st.booleans(),
        use_cache=st.booleans(),
        output_dtype=st.sampled_from([SparseType.FP32, SparseType.FP16]),
        num_indices_per_table=st.integers(min_value=1, max_value=5),
    )
    @settings(
        verbosity=VERBOSITY,
        max_examples=MAX_EXAMPLES,
        deadline=None,
    )
    def test_reset_embedding_weight_momentum(
        self,
        T: int,
        D: int,
        log_E: int,
        weights_precision: SparseType,
        mixed: bool,
        use_cache: bool,
        output_dtype: SparseType,
        num_indices_per_table: int,
    ) -> None:
        emb_op = SplitTableBatchedEmbeddingBagsCodegen
        E = int(10**log_E)
        D = D * 4
        Ds: list[int] = []
        Es: list[int] = []
        if not mixed:
            Ds = [D] * T
            Es = [E] * T
        else:
            Ds = [
                round_up(np.random.randint(low=int(0.25 * D), high=int(1.0 * D)), 4)
                for _ in range(T)
            ]
            Es = [
                np.random.randint(low=int(0.5 * E), high=int(2.0 * E)) for _ in range(T)
            ]
        compute_device = ComputeDevice.CUDA
        if use_cache:
            managed = [EmbeddingLocation.MANAGED_CACHING] * T
            if mixed:
                average_D = sum(Ds) // T
                for t, d in enumerate(Ds):
                    managed[t] = (
                        EmbeddingLocation.DEVICE if d < average_D else managed[t]
                    )
        else:
            managed = [
                np.random.choice(
                    [
                        EmbeddingLocation.DEVICE,
                        EmbeddingLocation.MANAGED,
                    ]
                )
                for _ in range(T)
            ]
        optimizer = OptimType.EXACT_ROWWISE_ADAGRAD
        cc = emb_op(
            embedding_specs=[
                (E, D, M, compute_device) for (E, D, M) in zip(Es, Ds, managed)
            ],
            optimizer=optimizer,
            weights_precision=weights_precision,
            output_dtype=output_dtype,
        )

        pruned_indices: list[int] = []
        pruned_indices_offsets: list[int] = [0]
        logical_table_ids: list[int] = []
        buffer_ids: list[int] = []
        for i in range(len(Es)):
            indices = [
                np.random.randint(low=1, high=int(Es[i] - 2))
                for _ in range(num_indices_per_table)
            ]
            pruned_indices += indices
            pruned_indices_offsets.append(
                pruned_indices_offsets[i] + num_indices_per_table
            )
            logical_table_ids.append(i)
            buffer_ids.append(i)
        pruned_indices_tensor = to_device(
            torch.tensor(pruned_indices, dtype=torch.int64, requires_grad=False), False
        )
        pruned_indices_offsets_tensor = to_device(
            torch.tensor(
                pruned_indices_offsets, dtype=torch.int64, requires_grad=False
            ),
            False,
        )
        logical_table_ids_tensor = to_device(
            torch.tensor(logical_table_ids, dtype=torch.int32, requires_grad=False),
            False,
        )
        buffer_ids_tensor = to_device(
            torch.tensor(buffer_ids, dtype=torch.int32, requires_grad=False), False
        )

        momentum1: list[Tensor] = [
            s for (s,) in cc.split_optimizer_states()
        ]  # List[rows]
        weight: list[Tensor] = cc.split_embedding_weights()  # List[(rows, dim)]
        for t in range(T):
            momentum1[t].fill_(1)
            weight[t].fill_(1)

        def check_weight_momentum(v: int) -> None:
            for i in range(len(pruned_indices)):
                logical_id = i // num_indices_per_table
                table_momentum1 = momentum1[logical_id]
                table_weight = weight[logical_id]
                dim = Ds[logical_id]
                expected_row_momentum1 = to_device(
                    torch.tensor(v, dtype=torch.float32), False
                )
                expected_row_weight = to_device(
                    torch.tensor([v] * dim, dtype=weights_precision.as_dtype()),
                    False,
                )
                pruned_index = pruned_indices[i]
                row_weight = table_weight[pruned_index]
                if weights_precision == SparseType.INT8:
                    row_weight = row_weight[:-INT8_EMB_ROW_DIM_OFFSET]
                self.assertEqual(table_momentum1[pruned_index], expected_row_momentum1)
                torch.testing.assert_close(
                    row_weight,
                    expected_row_weight,
                    rtol=0,
                    atol=0,
                    equal_nan=True,
                )

        check_weight_momentum(1)

        cc.reset_embedding_weight_momentum(
            pruned_indices_tensor,
            pruned_indices_offsets_tensor,
            logical_table_ids_tensor,
            buffer_ids_tensor,
        )

        check_weight_momentum(0)

    @unittest.skipIf(*gpu_unavailable)
    def test_update_hyper_parameters(self) -> None:
        # Create an abstract split table
        D = 8
        T = 2
        E = 10**2
        Ds = [D] * T
        Es = [E] * T

        hyperparameters = {
            "eps": 0.1,
            "beta1": 0.9,
            "beta2": 0.999,
            "weight_decay": 0.0,
        }
        cc = SplitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                (
                    E,
                    D,
                    EmbeddingLocation.DEVICE,
                    ComputeDevice.CUDA,
                )
                for (E, D) in zip(Es, Ds)
            ],
            learning_rate=0.1,
            **hyperparameters,  # pyre-ignore[6]
        )

        # Update hyperparameters
        updated_parameters = {
            key: value + 1.0 for key, value in hyperparameters.items()
        } | {"lr": 1.0, "lower_bound": 2.0}
        cc.update_hyper_parameters(updated_parameters)
        self.assertAlmostEqual(cc.get_learning_rate(), updated_parameters["lr"])
        self.assertAlmostEqual(cc.optimizer_args.eps, updated_parameters["eps"])
        self.assertAlmostEqual(cc.optimizer_args.beta1, updated_parameters["beta1"])
        self.assertAlmostEqual(cc.optimizer_args.beta2, updated_parameters["beta2"])
        self.assertAlmostEqual(
            cc.optimizer_args.weight_decay, updated_parameters["weight_decay"]
        )
        self.assertAlmostEqual(cc.gwd_lower_bound, updated_parameters["lower_bound"])

        # Update hyperparameters with invalid parameter name
        invalid_parameter = "invalid_parameter"
        with self.assertRaisesRegex(
            NotImplementedError,
            f"Setting hyper-parameter {invalid_parameter} is not supported",
        ):
            cc.update_hyper_parameters({"invalid_parameter": 1.0})


class PreallocatedHostBuffersTest(unittest.TestCase):
    """Tests for the preallocated_host_buffers feature in SplitTableBatchedEmbeddingBagsCodegen."""

    def _make_cpu_embedding_specs(
        self,
        T: int = 3,
        D: int = 8,
        E: int = 100,
    ) -> list[tuple[int, int, EmbeddingLocation, ComputeDevice]]:
        return [(E, D, EmbeddingLocation.HOST, ComputeDevice.CPU) for _ in range(T)]

    def test_preallocated_host_buffers_data_ptr_identity(self) -> None:
        T, D, E = 3, 8, 100
        embedding_specs = self._make_cpu_embedding_specs(T, D, E)

        weight_split = construct_split_state(
            embedding_specs,
            rowwise=False,
            cacheable=True,
            precision=SparseType.FP32,
        )
        momentum1_split = construct_split_state(
            embedding_specs,
            rowwise=True,
            cacheable=False,
        )

        weights_buffer = torch.zeros(weight_split.host_size, dtype=torch.float32)
        momentum1_buffer = torch.zeros(momentum1_split.host_size, dtype=torch.float32)

        cc = SplitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=embedding_specs,
            optimizer=OptimType.EXACT_ROWWISE_ADAGRAD,
            weights_precision=SparseType.FP32,
            preallocated_host_buffers={
                "weights": weights_buffer,
                "momentum1": momentum1_buffer,
            },
        )

        weights_host = cc.weights_host
        momentum1_host = cc.momentum1_host
        assert isinstance(weights_host, torch.Tensor)
        assert isinstance(momentum1_host, torch.Tensor)
        self.assertEqual(weights_host.data_ptr(), weights_buffer.data_ptr())
        self.assertEqual(momentum1_host.data_ptr(), momentum1_buffer.data_ptr())

    def test_preallocated_host_buffers_none_falls_back_to_default(self) -> None:
        T, D, E = 2, 8, 50
        embedding_specs = self._make_cpu_embedding_specs(T, D, E)

        cc_default = SplitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=embedding_specs,
            optimizer=OptimType.EXACT_ROWWISE_ADAGRAD,
            weights_precision=SparseType.FP32,
        )

        cc_none = SplitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=embedding_specs,
            optimizer=OptimType.EXACT_ROWWISE_ADAGRAD,
            weights_precision=SparseType.FP32,
            preallocated_host_buffers=None,
        )

        default_weights_host = cc_default.weights_host
        none_weights_host = cc_none.weights_host
        default_momentum1_host = cc_default.momentum1_host
        none_momentum1_host = cc_none.momentum1_host
        assert isinstance(default_weights_host, torch.Tensor)
        assert isinstance(none_weights_host, torch.Tensor)
        assert isinstance(default_momentum1_host, torch.Tensor)
        assert isinstance(none_momentum1_host, torch.Tensor)
        self.assertEqual(default_weights_host.numel(), none_weights_host.numel())
        self.assertEqual(default_momentum1_host.numel(), none_momentum1_host.numel())

    def test_preallocated_host_buffers_size_mismatch_raises(self) -> None:
        T, D, E = 2, 8, 50
        embedding_specs = self._make_cpu_embedding_specs(T, D, E)

        wrong_size_buffer = torch.zeros(7, dtype=torch.float32)

        with self.assertRaises(AssertionError):
            SplitTableBatchedEmbeddingBagsCodegen(
                embedding_specs=embedding_specs,
                optimizer=OptimType.EXACT_ROWWISE_ADAGRAD,
                weights_precision=SparseType.FP32,
                preallocated_host_buffers={"weights": wrong_size_buffer},
            )

    def test_preallocated_host_buffers_forward_correctness(self) -> None:
        T, D, E = 3, 8, 100
        B, L = 4, 5
        embedding_specs = self._make_cpu_embedding_specs(T, D, E)

        weight_split = construct_split_state(
            embedding_specs,
            rowwise=False,
            cacheable=True,
            precision=SparseType.FP32,
        )
        momentum1_split = construct_split_state(
            embedding_specs,
            rowwise=True,
            cacheable=False,
        )

        weights_buffer = torch.zeros(weight_split.host_size, dtype=torch.float32)
        momentum1_buffer = torch.zeros(momentum1_split.host_size, dtype=torch.float32)

        cc_prealloc = SplitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=embedding_specs,
            optimizer=OptimType.EXACT_ROWWISE_ADAGRAD,
            weights_precision=SparseType.FP32,
            preallocated_host_buffers={
                "weights": weights_buffer,
                "momentum1": momentum1_buffer,
            },
        )
        cc_default = SplitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=embedding_specs,
            optimizer=OptimType.EXACT_ROWWISE_ADAGRAD,
            weights_precision=SparseType.FP32,
        )

        # Copy weights from default to preallocated so they match
        for t in range(T):
            w_prealloc = cc_prealloc.split_embedding_weights()[t]
            w_default = cc_default.split_embedding_weights()[t]
            w_prealloc.data.copy_(w_default.data)

        indices = torch.randint(0, E, (T * B * L,))
        offsets = torch.arange(0, T * B + 1) * L

        out_prealloc = cc_prealloc(indices, offsets)
        out_default = cc_default(indices, offsets)

        torch.testing.assert_close(out_prealloc, out_default, atol=0.0, rtol=0.0)

    def test_preallocated_host_buffers_backward_updates_buffer(self) -> None:
        T, D, E = 2, 8, 50
        B, L = 4, 3
        embedding_specs = self._make_cpu_embedding_specs(T, D, E)

        weight_split = construct_split_state(
            embedding_specs,
            rowwise=False,
            cacheable=True,
            precision=SparseType.FP32,
        )
        momentum1_split = construct_split_state(
            embedding_specs,
            rowwise=True,
            cacheable=False,
        )

        weights_buffer = torch.zeros(weight_split.host_size, dtype=torch.float32)
        momentum1_buffer = torch.zeros(momentum1_split.host_size, dtype=torch.float32)

        cc = SplitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=embedding_specs,
            optimizer=OptimType.EXACT_ROWWISE_ADAGRAD,
            weights_precision=SparseType.FP32,
            learning_rate=0.1,
            preallocated_host_buffers={
                "weights": weights_buffer,
                "momentum1": momentum1_buffer,
            },
        )

        # Initialize weights to non-zero so backward produces visible updates
        for t in range(T):
            w = cc.split_embedding_weights()[t]
            w.data.fill_(1.0)

        indices = torch.randint(0, E, (T * B * L,))
        offsets = torch.arange(0, T * B + 1) * L

        output = cc(indices, offsets)
        output.sum().backward()

        # After backward, momentum1 accumulates squared gradients → should be non-zero
        self.assertTrue(
            momentum1_buffer.any(),
            "momentum1 buffer should have been updated by backward pass",
        )

    def test_preallocated_host_buffers_partial_keys(self) -> None:
        T, D, E = 2, 8, 50
        embedding_specs = self._make_cpu_embedding_specs(T, D, E)

        weight_split = construct_split_state(
            embedding_specs,
            rowwise=False,
            cacheable=True,
            precision=SparseType.FP32,
        )

        weights_buffer = torch.zeros(weight_split.host_size, dtype=torch.float32)

        cc = SplitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=embedding_specs,
            optimizer=OptimType.EXACT_ROWWISE_ADAGRAD,
            weights_precision=SparseType.FP32,
            preallocated_host_buffers={"weights": weights_buffer},
        )

        weights_host = cc.weights_host
        momentum1_host = cc.momentum1_host
        assert isinstance(weights_host, torch.Tensor)
        assert isinstance(momentum1_host, torch.Tensor)
        self.assertEqual(weights_host.data_ptr(), weights_buffer.data_ptr())
        # momentum1 should still be allocated (not preallocated)
        self.assertGreater(momentum1_host.numel(), 0)
        self.assertNotEqual(momentum1_host.data_ptr(), weights_buffer.data_ptr())

    def test_preallocated_host_buffers_adagrad_per_element_momentum1(
        self,
    ) -> None:
        """Test preallocated buffers with EXACT_ADAGRAD (non-rowwise momentum1).

        Uses EXACT_ADAGRAD to exercise a different momentum1 allocation size
        (per-element, non-rowwise) compared to EXACT_ROWWISE_ADAGRAD (per-row).

        Note: the momentum2 preallocated path is structurally untestable here
        because the optimizers that allocate momentum2 (ADAM, PARTIAL_ROWWISE_ADAM,
        LAMB, PARTIAL_ROWWISE_LAMB, ENSEMBLE_ROWWISE_ADAGRAD) require CUDA, and
        FBGEMM forbids EmbeddingLocation.HOST on CUDA devices (assertion at
        split_table_batched_embeddings_ops_training.py:~915 — 'EmbeddingLocation.HOST
        doesn't work for CUDA device!'). This per-element momentum1 test exercises
        the same apply_split_helper plumbing that momentum2 would use — the only
        untested surface is the kwarg-routing wiring at the call site, which is
        a one-line `.get("momentum2")` mirroring the momentum1 routing.
        """
        T, D, E = 2, 8, 50
        embedding_specs = self._make_cpu_embedding_specs(T, D, E)

        weight_split = construct_split_state(
            embedding_specs,
            rowwise=False,
            cacheable=True,
            precision=SparseType.FP32,
        )
        # EXACT_ADAGRAD: non-rowwise momentum1 (per-element, same size as weights)
        momentum1_split = construct_split_state(
            embedding_specs,
            rowwise=False,
            cacheable=False,
        )

        weights_buffer = torch.zeros(weight_split.host_size, dtype=torch.float32)
        momentum1_buffer = torch.zeros(momentum1_split.host_size, dtype=torch.float32)

        cc = SplitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=embedding_specs,
            optimizer=OptimType.EXACT_ADAGRAD,
            weights_precision=SparseType.FP32,
            preallocated_host_buffers={
                "weights": weights_buffer,
                "momentum1": momentum1_buffer,
            },
        )

        weights_host = cc.weights_host
        momentum1_host = cc.momentum1_host
        assert isinstance(weights_host, torch.Tensor)
        assert isinstance(momentum1_host, torch.Tensor)
        self.assertEqual(weights_host.data_ptr(), weights_buffer.data_ptr())
        self.assertEqual(momentum1_host.data_ptr(), momentum1_buffer.data_ptr())
        # Non-rowwise momentum1 has same numel as weights (per-element)
        self.assertEqual(momentum1_host.numel(), weights_host.numel())

    def test_preallocated_host_buffers_wrong_dtype_raises(self) -> None:
        T, D, E = 2, 8, 50
        embedding_specs = self._make_cpu_embedding_specs(T, D, E)

        weight_split = construct_split_state(
            embedding_specs,
            rowwise=False,
            cacheable=True,
            precision=SparseType.FP32,
        )

        # float16 buffer when float32 expected
        wrong_dtype_buffer = torch.zeros(weight_split.host_size, dtype=torch.float16)

        with self.assertRaises(AssertionError):
            SplitTableBatchedEmbeddingBagsCodegen(
                embedding_specs=embedding_specs,
                optimizer=OptimType.EXACT_ROWWISE_ADAGRAD,
                weights_precision=SparseType.FP32,
                preallocated_host_buffers={"weights": wrong_dtype_buffer},
            )

    def test_preallocated_host_buffers_non_contiguous_raises(self) -> None:
        T, D, E = 2, 8, 50
        embedding_specs = self._make_cpu_embedding_specs(T, D, E)

        weight_split = construct_split_state(
            embedding_specs,
            rowwise=False,
            cacheable=True,
            precision=SparseType.FP32,
        )

        # Non-contiguous via stride-2 slicing
        non_contiguous_buffer = torch.zeros(
            2 * weight_split.host_size, dtype=torch.float32
        )[::2]

        with self.assertRaises(AssertionError):
            SplitTableBatchedEmbeddingBagsCodegen(
                embedding_specs=embedding_specs,
                optimizer=OptimType.EXACT_ROWWISE_ADAGRAD,
                weights_precision=SparseType.FP32,
                preallocated_host_buffers={"weights": non_contiguous_buffer},
            )

    def test_preallocated_host_buffers_2d_buffer_raises(self) -> None:
        T, D, E = 2, 8, 50
        embedding_specs = self._make_cpu_embedding_specs(T, D, E)

        weight_split = construct_split_state(
            embedding_specs,
            rowwise=False,
            cacheable=True,
            precision=SparseType.FP32,
        )

        # 2D buffer with matching numel
        buffer_2d = torch.zeros(E, D * T, dtype=torch.float32)
        # Ensure numel matches
        self.assertEqual(buffer_2d.numel(), weight_split.host_size)

        with self.assertRaises(AssertionError):
            SplitTableBatchedEmbeddingBagsCodegen(
                embedding_specs=embedding_specs,
                optimizer=OptimType.EXACT_ROWWISE_ADAGRAD,
                weights_precision=SparseType.FP32,
                preallocated_host_buffers={"weights": buffer_2d},
            )

    def test_preallocated_host_buffers_invalid_key_raises(self) -> None:
        T, D, E = 2, 8, 50
        embedding_specs = self._make_cpu_embedding_specs(T, D, E)

        weight_split = construct_split_state(
            embedding_specs,
            rowwise=False,
            cacheable=True,
            precision=SparseType.FP32,
        )

        buf = torch.zeros(weight_split.host_size, dtype=torch.float32)

        # "weight" is a typo for "weights"
        with self.assertRaises(AssertionError):
            SplitTableBatchedEmbeddingBagsCodegen(
                embedding_specs=embedding_specs,
                optimizer=OptimType.EXACT_ROWWISE_ADAGRAD,
                weights_precision=SparseType.FP32,
                preallocated_host_buffers={"weight": buf},
            )

    def test_preallocated_host_buffers_empty_dict_falls_back_to_default(self) -> None:
        T, D, E = 2, 8, 50
        embedding_specs = self._make_cpu_embedding_specs(T, D, E)

        cc_default = SplitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=embedding_specs,
            optimizer=OptimType.EXACT_ROWWISE_ADAGRAD,
            weights_precision=SparseType.FP32,
        )

        cc_empty = SplitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=embedding_specs,
            optimizer=OptimType.EXACT_ROWWISE_ADAGRAD,
            weights_precision=SparseType.FP32,
            preallocated_host_buffers={},
        )

        default_weights_host = cc_default.weights_host
        empty_weights_host = cc_empty.weights_host
        default_momentum1_host = cc_default.momentum1_host
        empty_momentum1_host = cc_empty.momentum1_host
        assert isinstance(default_weights_host, torch.Tensor)
        assert isinstance(empty_weights_host, torch.Tensor)
        assert isinstance(default_momentum1_host, torch.Tensor)
        assert isinstance(empty_momentum1_host, torch.Tensor)
        self.assertEqual(default_weights_host.numel(), empty_weights_host.numel())
        self.assertEqual(default_momentum1_host.numel(), empty_momentum1_host.numel())

    def test_consumed_keys_for_each_optimizer(self) -> None:
        """_get_consumed_preallocated_keys returns the documented set per OptimType.

        Locks down the optimizer→buffer-name mapping that the validation in
        SplitTableBatchedEmbeddingBagsCodegen.__init__ relies on. If a future
        optimizer is added to OptimType, this test will fail until the helper
        is updated to classify it.
        """
        # weights only (no momentum allocation paths in __init__).
        for optimizer in (OptimType.NONE, OptimType.EXACT_SGD):
            self.assertEqual(
                _get_consumed_preallocated_keys(optimizer),
                {"weights"},
                f"unexpected consumed keys for {optimizer}",
            )
        # weights + momentum1 only.
        for optimizer in (
            OptimType.EXACT_ADAGRAD,
            OptimType.EXACT_ROWWISE_ADAGRAD,
            OptimType.EMAINPLACE_ROWWISE_ADAGRAD,
        ):
            self.assertEqual(
                _get_consumed_preallocated_keys(optimizer),
                {"weights", "momentum1"},
                f"unexpected consumed keys for {optimizer}",
            )
        # weights + momentum1 + momentum2.
        for optimizer in (
            OptimType.ADAM,
            OptimType.PARTIAL_ROWWISE_ADAM,
            OptimType.LAMB,
            OptimType.PARTIAL_ROWWISE_LAMB,
            OptimType.ENSEMBLE_ROWWISE_ADAGRAD,
        ):
            self.assertEqual(
                _get_consumed_preallocated_keys(optimizer),
                {"weights", "momentum1", "momentum2"},
                f"unexpected consumed keys for {optimizer}",
            )

    def test_unconsumed_key_raises_for_sgd(self) -> None:
        """SGD with momentum1 buffer raises (SGD only consumes 'weights')."""
        T, D, E = 2, 8, 50
        embedding_specs = self._make_cpu_embedding_specs(T, D, E)
        weight_split = construct_split_state(
            embedding_specs,
            rowwise=False,
            cacheable=True,
            precision=SparseType.FP32,
        )
        weights_buffer = torch.zeros(weight_split.host_size, dtype=torch.float32)
        momentum_buffer = torch.zeros(100, dtype=torch.float32)

        with self.assertRaisesRegex(AssertionError, r"momentum1"):
            SplitTableBatchedEmbeddingBagsCodegen(
                embedding_specs=embedding_specs,
                optimizer=OptimType.EXACT_SGD,
                weights_precision=SparseType.FP32,
                preallocated_host_buffers={
                    "weights": weights_buffer,
                    "momentum1": momentum_buffer,
                },
            )

    def test_unconsumed_key_raises_for_adagrad_momentum2(self) -> None:
        """EXACT_ROWWISE_ADAGRAD with momentum2 buffer raises (Adagrad doesn't use momentum2)."""
        T, D, E = 2, 8, 50
        embedding_specs = self._make_cpu_embedding_specs(T, D, E)
        weight_split = construct_split_state(
            embedding_specs,
            rowwise=False,
            cacheable=True,
            precision=SparseType.FP32,
        )
        momentum1_split = construct_split_state(
            embedding_specs,
            rowwise=True,
            cacheable=False,
        )
        weights_buffer = torch.zeros(weight_split.host_size, dtype=torch.float32)
        momentum1_buffer = torch.zeros(momentum1_split.host_size, dtype=torch.float32)
        momentum2_buffer = torch.zeros(100, dtype=torch.float32)

        with self.assertRaisesRegex(AssertionError, r"momentum2"):
            SplitTableBatchedEmbeddingBagsCodegen(
                embedding_specs=embedding_specs,
                optimizer=OptimType.EXACT_ROWWISE_ADAGRAD,
                weights_precision=SparseType.FP32,
                preallocated_host_buffers={
                    "weights": weights_buffer,
                    "momentum1": momentum1_buffer,
                    "momentum2": momentum2_buffer,
                },
            )

    def test_consumed_keys_only_succeeds(self) -> None:
        """Providing exactly the consumed-key set constructs successfully (regression guard)."""
        T, D, E = 2, 8, 50
        embedding_specs = self._make_cpu_embedding_specs(T, D, E)
        weight_split = construct_split_state(
            embedding_specs,
            rowwise=False,
            cacheable=True,
            precision=SparseType.FP32,
        )
        momentum1_split = construct_split_state(
            embedding_specs,
            rowwise=True,
            cacheable=False,
        )
        weights_buffer = torch.zeros(weight_split.host_size, dtype=torch.float32)
        momentum1_buffer = torch.zeros(momentum1_split.host_size, dtype=torch.float32)

        cc = SplitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=embedding_specs,
            optimizer=OptimType.EXACT_ROWWISE_ADAGRAD,
            weights_precision=SparseType.FP32,
            preallocated_host_buffers={
                "weights": weights_buffer,
                "momentum1": momentum1_buffer,
            },
        )
        weights_host = cc.weights_host
        momentum1_host = cc.momentum1_host
        assert isinstance(weights_host, torch.Tensor)
        assert isinstance(momentum1_host, torch.Tensor)
        self.assertEqual(weights_host.data_ptr(), weights_buffer.data_ptr())
        self.assertEqual(momentum1_host.data_ptr(), momentum1_buffer.data_ptr())


if __name__ == "__main__":
    unittest.main()
