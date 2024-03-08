#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[56]

import itertools
import random
import unittest
from typing import List, Tuple

import hypothesis.strategies as st
import numpy as np
import torch
import torch._dynamo
from hypothesis import assume, given, settings, Verbosity

from .common import additional_decorators, open_source, torch_compiled

if open_source:
    # pyre-ignore[21]
    from test_utils import (
        gpu_available,
        gpu_unavailable,
        gradcheck,
        on_oss_clang,
        optests,
        symint_vector_unsupported,
        use_cpu_strategy,
    )
else:
    from fbgemm_gpu.test.test_utils import (
        gpu_available,
        gpu_unavailable,
        gradcheck,
        on_oss_clang,
        optests,
        symint_vector_unsupported,
        use_cpu_strategy,
    )


@optests.generate_opcheck_tests(additional_decorators=additional_decorators)
class JaggedTensorOpsTest(unittest.TestCase):
    def setUp(self) -> None:
        if symint_vector_unsupported()[0]:
            return

        assert hasattr(
            torch._dynamo.config, "assume_static_by_default"
        ), "Need to update the config as the dynamic/auto-dynamic setting has changed"
        # Turn off static assumption for auto-dynamic
        torch._dynamo.config.assume_static_by_default = False

    @staticmethod
    def expand_into_jagged_permute_ref_(
        permute: List[int],
        length: List[int],
    ) -> List[int]:
        offsets = [0] + list(itertools.accumulate(length))
        output_permute = []
        for r in permute:
            output_permute.extend(
                range(
                    offsets[r],
                    offsets[r + 1],
                )
            )
        return output_permute

    @unittest.skipIf(*on_oss_clang)
    @given(
        T=st.integers(min_value=10, max_value=20),
        W=st.integers(min_value=8, max_value=64),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=10, deadline=None)
    def test_expand_into_jagged_permute(
        self,
        T: int,
        W: int,
    ) -> None:
        length_per_w = [random.randint(5000, 10000) for i in range(W)]
        length_1d = list(
            itertools.chain.from_iterable(itertools.repeat(x, T) for x in length_per_w)
        )
        permute_list = list(range(T * W))
        random.shuffle(permute_list)
        permuted_length_1d = [length_1d[r] for r in permute_list]
        permute_tensor = torch.tensor(permute_list)

        # compute offsets
        offsets_1d = [0] + list(itertools.accumulate(length_1d))
        permuted_offsets_1d = [0] + list(itertools.accumulate(permuted_length_1d))
        offsets_1d_tensor = torch.tensor(offsets_1d)
        permuted_offsets_1d_tensor = torch.tensor(permuted_offsets_1d)

        # cpu op
        output_permute_cpu = torch.ops.fbgemm.expand_into_jagged_permute(
            permute_tensor,
            offsets_1d_tensor,
            permuted_offsets_1d_tensor,
            offsets_1d[-1],
        )

        # reference solution
        output_permute_ref = self.expand_into_jagged_permute_ref_(
            permute_list,
            length_1d,
        )
        output_permute_ref_tensor = torch.tensor(output_permute_ref)

        # assert cpu and gpu ops
        torch.testing.assert_close(output_permute_cpu, output_permute_ref_tensor)
        if gpu_available:
            # gpu op
            output_permute_gpu = torch.ops.fbgemm.expand_into_jagged_permute(
                permute_tensor.cuda(),
                offsets_1d_tensor.cuda(),
                permuted_offsets_1d_tensor.cuda(),
                offsets_1d[-1],
            )
            torch.testing.assert_close(
                output_permute_gpu.cpu(), output_permute_ref_tensor
            )

    def _to_padded_dense(
        self,
        values: torch.Tensor,
        offsets: List[torch.LongTensor],
        max_lengths: np.ndarray,
        padding_value: float = 0,
    ) -> torch.Tensor:
        outer_dense_size = len(offsets[0]) - 1
        # canonicalize by unsqueeze the last dim if the inner dense dimension
        # is 1 and folded.
        inner_dense_size = 1 if values.ndim == 1 else values.size(-1)
        dense = torch.empty(
            (outer_dense_size,) + tuple(max_lengths) + (inner_dense_size,),
            dtype=values.dtype,
            device=values.device,
        )
        for i in range(outer_dense_size):
            for jagged_coord in itertools.product(
                *(list(range(max_l)) for max_l in max_lengths)
            ):
                cur_offset = i
                is_zero = False
                for d in range(len(max_lengths)):
                    # pyre-fixme[6]: For 1st argument expected `Union[None, _NestedSe...
                    begin = offsets[d][cur_offset].item()
                    # pyre-fixme[6]: For 1st argument expected `Union[None, _NestedSe...
                    end = offsets[d][cur_offset + 1].item()
                    # pyre-fixme[6]: For 1st param expected `int` but got
                    #  `Union[bool, float, int]`.
                    if jagged_coord[d] >= end - begin:
                        is_zero = True
                        break
                    cur_offset = begin + jagged_coord[d]
                dense[(i,) + jagged_coord] = (
                    padding_value
                    if is_zero
                    # pyre-fixme[6]: For 1st argument expected `Union[None, _NestedSe...
                    else values[cur_offset]
                )
        return dense.squeeze(-1) if values.ndim == 1 else dense

    # TODO: reuse this code in test_(stacked)_jagged_1/2d
    def _generate_jagged_tensor(
        self,
        num_jagged_dim: int,
        outer_dense_size: int,
        inner_dense_size: int,
        dtype: torch.dtype,
        device: torch.device,
        fold_inner_dense: bool = False,
        # dynamo to mark the input as dynamic shape to make sure symbolic
        # shape is generated
        mark_dynamic: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.LongTensor], np.ndarray]:
        max_lengths = np.random.randint(low=1, high=10, size=(num_jagged_dim,))
        x_offsets: List[torch.LongTensor] = []
        num_lengths = outer_dense_size
        for d in range(num_jagged_dim):
            # Sometimes length[i] exceed max_L meaning jagged->dense will be
            # truncation vs. padding
            lengths = torch.randint(
                # PT2 specialize 0/1 dims as non-symbolic shape. So we need
                # to make it non 0/1 for testing. In real cases it'll likelyl
                # not 0/1 anyway (if so, they'll be recompiled)
                low=0 if not mark_dynamic else 1,
                high=max_lengths[d] * 2,
                # pyre-fixme[6]: For 3rd param expected `Union[List[int], Size,
                #  typing.Tuple[int, ...]]` but got `Tuple[Union[bool, float, int]]`.
                size=(num_lengths,),
                device=device,
            )
            offset = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
            if mark_dynamic:
                torch._dynamo.mark_dynamic(offset, 0)
            x_offsets.append(offset)
            num_lengths = x_offsets[-1][-1].item()

        x_values = torch.rand(
            # pyre-fixme[6]: For 1st param expected `Union[List[int], Size,
            #  typing.Tuple[int, ...]]` but got `Tensor`.
            x_offsets[-1][-1] * inner_dense_size,
            dtype=dtype,
            device=device,
        )
        if inner_dense_size != 1 or not fold_inner_dense:
            # pyre-fixme[6]: For 1st param expected `int` but got `Union[bool, float, int]`.
            x_values = x_values.reshape(x_offsets[-1][-1].item(), inner_dense_size)

        if mark_dynamic:
            for i in range(inner_dense_size):
                torch._dynamo.mark_dynamic(x_values, i)

        return x_values, x_offsets, max_lengths

    def _test_dense_to_jagged(
        self,
        num_jagged_dim: int,
        outer_dense_size: int,
        inner_dense_size: int,
        dtype: torch.dtype,
        device_type: str,
        precompute_total_L: bool,
    ) -> None:
        # Generate multi-dim jagged tensor
        device = torch.device(device_type)
        values_2d, offsets, max_lengths = self._generate_jagged_tensor(
            num_jagged_dim, outer_dense_size, inner_dense_size, dtype, device
        )
        values_2d = values_2d.clone().detach().requires_grad_(True)

        # jagged -> dense
        dense = torch.ops.fbgemm.jagged_to_padded_dense(values_2d, offsets, max_lengths)

        # dense -> jagged (op which is being tested)
        if precompute_total_L:
            total_L = values_2d.size(0)
            jagged_values, jagged_offsets = torch.ops.fbgemm.dense_to_jagged(
                dense, offsets, total_L
            )
        else:
            jagged_values, jagged_offsets = torch.ops.fbgemm.dense_to_jagged(
                dense, offsets
            )

        # jagged -> dense
        dense2 = torch.ops.fbgemm.jagged_to_padded_dense(
            jagged_values, jagged_offsets, max_lengths
        )

        # verify forward
        torch.testing.assert_close(dense, dense2)

        # verify backward
        dense.retain_grad()
        ref_output_values = jagged_values.clone().detach().requires_grad_(True)
        ref_values = dense.clone().detach().requires_grad_(True)
        jagged_values.backward(ref_output_values)
        torch.testing.assert_close(dense.grad, ref_values)

    @given(
        num_jagged_dim=st.integers(1, 5),
        outer_dense_size=st.integers(0, 5),
        inner_dense_size=st.integers(0, 5),
        dtype=st.sampled_from([torch.float, torch.half, torch.bfloat16]),
        device_type=(
            st.sampled_from(["cpu", "cuda"]) if gpu_available else st.just("cpu")
        ),
        precompute_total_L=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_dense_to_jagged(
        self,
        num_jagged_dim: int,
        outer_dense_size: int,
        inner_dense_size: int,
        dtype: torch.dtype,
        device_type: str,
        precompute_total_L: bool,
    ) -> None:
        self._test_dense_to_jagged(
            num_jagged_dim,
            outer_dense_size,
            inner_dense_size,
            dtype,
            device_type,
            precompute_total_L,
        )

    @unittest.skipIf(*gpu_unavailable)
    @given(
        num_jagged_dim=st.just(1),
        outer_dense_size=st.integers(0, 6000),
        inner_dense_size=st.sampled_from([8, 16, 23, 24, 48, 50, 64, 72, 96, 192]),
        dtype=st.just(torch.half),
        device_type=(
            st.sampled_from(["cpu", "cuda"]) if gpu_available else st.just("cpu")
        ),
        precompute_total_L=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_dense_to_jagged_opt(
        self,
        num_jagged_dim: int,
        outer_dense_size: int,
        inner_dense_size: int,
        dtype: torch.dtype,
        device_type: str,
        precompute_total_L: bool,
    ) -> None:
        self._test_dense_to_jagged(
            num_jagged_dim,
            outer_dense_size,
            inner_dense_size,
            dtype,
            device_type,
            precompute_total_L,
        )

    # (8000+1) * 8 (size of the element of LongTensor/int64_t offsets)
    # = ~62.5KB > 48KB default shared memory on V100/A100.
    @unittest.skipIf(*gpu_unavailable)
    @given(
        num_jagged_dim=st.just(1),
        outer_dense_size=st.just(8000),
        inner_dense_size=st.just(16),
        dtype=st.just(torch.half),
        device_type=(
            st.sampled_from(["cpu", "cuda"]) if gpu_available else st.just("cpu")
        ),
        precompute_total_L=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=1, deadline=None)
    def test_dense_to_jagged_opt_large_batch(
        self,
        num_jagged_dim: int,
        outer_dense_size: int,
        inner_dense_size: int,
        dtype: torch.dtype,
        device_type: str,
        precompute_total_L: bool,
    ) -> None:
        self._test_dense_to_jagged(
            num_jagged_dim,
            outer_dense_size,
            inner_dense_size,
            dtype,
            device_type,
            precompute_total_L,
        )

    @given(
        num_jagged_dim=st.integers(1, 5),
        outer_dense_size=st.integers(0, 5),
        inner_dense_size=st.integers(0, 5),
        dtype=st.sampled_from([torch.float, torch.half, torch.bfloat16]),
        device_type=st.sampled_from(["meta"]),
        precompute_total_L=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_dense_to_jagged_meta_backend(
        self,
        num_jagged_dim: int,
        outer_dense_size: int,
        inner_dense_size: int,
        dtype: torch.dtype,
        device_type: str,
        precompute_total_L: bool,
    ) -> None:
        device = torch.device("cpu")
        values_2d, offsets, max_lengths = self._generate_jagged_tensor(
            num_jagged_dim, outer_dense_size, inner_dense_size, dtype, device
        )
        values_2d = values_2d.clone().detach().requires_grad_(True)

        # jagged -> dense
        dense = torch.ops.fbgemm.jagged_to_padded_dense(values_2d, offsets, max_lengths)

        # dense -> jagged (op which is being tested)
        if precompute_total_L:
            total_L = values_2d.size(0)
            dense.to(device_type)
            jagged_values, jagged_offsets = torch.ops.fbgemm.dense_to_jagged(
                dense, offsets, total_L
            )
        else:
            dense.to(device_type)
            jagged_values, jagged_offsets = torch.ops.fbgemm.dense_to_jagged(
                dense, offsets
            )

        jagged_values.to(device_type)
        # jagged -> dense
        dense2 = torch.ops.fbgemm.jagged_to_padded_dense(
            jagged_values, jagged_offsets, max_lengths
        )

        # verify forward
        assert dense.size() == dense2.size()

    @optests.dontGenerateOpCheckTests("tests that call torch.compile are slow")
    @unittest.skipIf(*symint_vector_unsupported())
    @given(
        num_jagged_dim=st.integers(1, 5),
        # TODO: size = 0/1 will be incorrectly specialized
        outer_dense_size=st.integers(2, 5),
        inner_dense_size=st.integers(2, 5),
        dtype=st.sampled_from([torch.float, torch.half, torch.bfloat16]),
        device_type=(
            st.sampled_from(["cpu", "cuda"]) if gpu_available else st.just("cpu")
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_dense_to_jagged_dynamic_shape(
        self,
        num_jagged_dim: int,
        outer_dense_size: int,
        inner_dense_size: int,
        dtype: torch.dtype,
        device_type: str,
    ) -> None:
        # Start a fresh compile for each parameter of the test case
        torch._dynamo.reset()

        values_2d, offsets, max_lengths = self._generate_jagged_tensor(
            num_jagged_dim,
            outer_dense_size,
            inner_dense_size,
            dtype,
            torch.device(device_type),
            mark_dynamic=True,
        )
        values_2d = values_2d.clone().detach().requires_grad_(True)

        def jagged_to_dense(
            values: torch.Tensor,
            offsets: List[torch.LongTensor],
            max_lengths: List[int],
        ) -> torch.Tensor:
            return torch.ops.fbgemm.jagged_to_padded_dense(values, offsets, max_lengths)

        # jagged -> dense
        dense = jagged_to_dense(values_2d, offsets, max_lengths.tolist())

        # dense -> jagged, it is required to pre-compute totalL
        total_L = values_2d.size(0)
        dense = dense.clone().detach().to(device_type)

        torch._dynamo.mark_dynamic(dense, 0)
        torch._dynamo.mark_dynamic(dense, -1)

        def dense_to_jagged_withL(
            dense: torch.Tensor, offsets: List[torch.LongTensor], total_L: List[int]
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            return torch.ops.fbgemm.dense_to_jagged(dense, offsets, total_L)

        def dense_to_jagged_noL(
            dense: torch.Tensor, offsets: List[torch.LongTensor]
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            return torch.ops.fbgemm.dense_to_jagged(dense, offsets)

        jagged_values, jagged_offsets = dense_to_jagged_noL(dense, offsets)
        jagged_values, jagged_offsets = dense_to_jagged_withL(dense, offsets, total_L)

        jagged_values.to(device_type)
        # jagged -> dense
        dense2 = torch.ops.fbgemm.jagged_to_padded_dense(
            jagged_values, jagged_offsets, max_lengths
        )

        # verify forward
        assert dense.size() == dense2.size()

    @given(
        num_jagged_dim=st.integers(1, 5),
        outer_dense_size=st.integers(0, 5),
        inner_dense_size=st.integers(0, 5),
        fold_inner_dense=st.booleans(),
        padding_value=st.sampled_from([0, -1e-8]),
        dtype=st.sampled_from([torch.float, torch.half, torch.bfloat16]),
        device_type=(
            st.sampled_from(["cpu", "cuda"]) if gpu_available else st.just("cpu")
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_jagged_to_padded_dense(
        self,
        num_jagged_dim: int,
        outer_dense_size: int,
        inner_dense_size: int,
        fold_inner_dense: bool,
        padding_value: float,
        dtype: torch.dtype,
        device_type: str,
    ) -> None:
        # CPU doesn't support bfloat16
        assume(device_type != "cpu" or dtype != torch.bfloat16)
        assume(not fold_inner_dense or inner_dense_size == 1)

        # Testing with a basic crafted example.
        # dense representation is
        # [[[[0, 1], [ 0,  0], [0, 0]],
        #   [[2, 3], [ 4,  5], [6, 7]],
        #   [[0, 0], [ 0,  0], [0, 0]],
        #   [[0, 0], [ 0,  0], [0, 0]]],
        #  [[[0, 0], [ 0,  0], [0, 0]],
        #   [[0, 0], [ 0,  0], [0, 0]],
        #   [[0, 0], [ 0,  0], [0, 0]],
        #   [[0, 0], [ 0,  0], [0, 0]]],
        #  [[[8, 9], [10, 11], [0, 0]],
        #   [[0, 0], [ 0,  0], [0, 0]],
        #   [[0, 0], [ 0,  0], [0, 0]],
        #   [[0, 0], [ 0,  0], [0, 0]]]],
        # inner_dense_size = 2
        # x_offsets = [
        #     torch.LongTensor([0, 2, 2, 3]),  # lengths torch.Tensor([2, 0, 1]),
        #     torch.LongTensor([0, 1, 4, 6]),  # lengths torch.Tensor([1, 3, 2]),
        # ]
        # outer_dense_size = len(x_offsets[0]) - 1
        # max_lengths = [4, 3]

        device = torch.device(device_type)

        x_values, x_offsets, max_lengths = self._generate_jagged_tensor(
            num_jagged_dim,
            outer_dense_size,
            inner_dense_size,
            torch.float,
            device,
            fold_inner_dense,
        )

        output_ref = self._to_padded_dense(
            x_values, x_offsets, max_lengths, padding_value=padding_value
        )
        output = torch.ops.fbgemm.jagged_to_padded_dense(
            x_values,
            x_offsets,
            max_lengths,
            padding_value=padding_value,
        )

        torch.testing.assert_close(output, output_ref)

        gradcheck(
            torch.ops.fbgemm.jagged_to_padded_dense,
            (
                x_values.float().requires_grad_(True),
                x_offsets,
                max_lengths,
                padding_value,
            ),
            eps=1e-2,
            atol=1e-3,
            rtol=1e-3,
        )

    @given(
        num_jagged_dim=st.integers(1, 5),
        outer_dense_size=st.integers(0, 5),
        inner_dense_size=st.integers(0, 5),
        padding_value=st.just(0),
        dtype=st.sampled_from([torch.float, torch.half, torch.bfloat16]),
        device_type=st.just("meta"),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_jagged_to_padded_dense_meta_backend(
        self,
        num_jagged_dim: int,
        outer_dense_size: int,
        inner_dense_size: int,
        padding_value: float,
        dtype: torch.dtype,
        device_type: str,
    ) -> None:
        assume(device_type != "cpu" or dtype != torch.bfloat16)
        device = torch.device("cpu")

        x_values, x_offsets, max_lengths = self._generate_jagged_tensor(
            num_jagged_dim, outer_dense_size, inner_dense_size, torch.float, device
        )

        output_ref = self._to_padded_dense(
            x_values, x_offsets, max_lengths, padding_value=padding_value
        )
        x_values.to(device_type)
        output = torch.ops.fbgemm.jagged_to_padded_dense(
            x_values,
            x_offsets,
            max_lengths,
            padding_value=padding_value,
        )

        assert output.size() == output_ref.size()

    def _test_jagged_elementwise_binary(
        self,
        num_jagged_dim: int,
        outer_dense_size: int,
        inner_dense_size: int,
        operation: str,
        dtype: torch.dtype,
        device_type: str,
    ) -> None:
        device = torch.device(device_type)

        x_values, x_offsets, max_lengths = self._generate_jagged_tensor(
            num_jagged_dim, outer_dense_size, inner_dense_size, dtype, device
        )
        y = torch.rand(
            outer_dense_size * np.prod(max_lengths) * inner_dense_size,
            dtype=dtype,
            device=device,
        ).reshape((outer_dense_size,) + tuple(max_lengths) + (inner_dense_size,))

        x_padded = self._to_padded_dense(x_values, x_offsets, max_lengths)
        if operation == "add":
            output_ref = x_padded + y
            output = torch.ops.fbgemm.jagged_dense_elementwise_add(
                x_values, x_offsets, y
            )
        elif operation == "add_jagged_output":
            # create a jagged tensor and then densify
            y = self._to_padded_dense(
                torch.rand(
                    (
                        max(outer_dense_size * np.prod(max_lengths), x_values.size(0)),
                        inner_dense_size,
                    ),
                    dtype=dtype,
                    device=device,
                ),
                x_offsets,
                max_lengths,
            )
            output_ref = x_padded + y
            (
                output,
                output_offsets,
            ) = torch.ops.fbgemm.jagged_dense_elementwise_add_jagged_output(
                x_values, x_offsets, y
            )
            output = self._to_padded_dense(output, output_offsets, max_lengths)
        elif operation == "mul":
            output_ref = x_padded * y
            output, output_offsets = torch.ops.fbgemm.jagged_dense_elementwise_mul(
                x_values, x_offsets, y
            )
            output = self._to_padded_dense(output, output_offsets, max_lengths)
        else:
            raise AssertionError(f"Unknown operation {operation}")

        torch.testing.assert_close(output, output_ref)

        if operation == "add":
            f = torch.ops.fbgemm.jagged_dense_elementwise_add
        elif operation == "add_jagged_output":
            # pyre-fixme[2]: Parameter must be annotated.
            def add_jagged_output_func(*args) -> torch.Tensor:
                return torch.ops.fbgemm.jagged_dense_elementwise_add_jagged_output(
                    *args
                )[0]

            f = add_jagged_output_func
        else:
            assert operation == "mul"

            # pyre-fixme[2]: Parameter must be annotated.
            def mul_func(*args) -> torch.Tensor:
                return torch.ops.fbgemm.jagged_dense_elementwise_mul(*args)[0]

            f = mul_func

        gradcheck(
            f,
            (
                x_values.float().requires_grad_(True),
                x_offsets,
                y.float().requires_grad_(True),
            ),
            eps=1e-2,
            atol=1e-3,
            rtol=1e-3,
        )

    @given(
        num_jagged_dim=st.integers(1, 4),
        outer_dense_size=st.integers(0, 4),
        inner_dense_size=st.integers(0, 4),
        operation=st.sampled_from(["add", "add_jagged_output", "mul"]),
        dtype=st.sampled_from([torch.float, torch.half, torch.bfloat16]),
        device_type=(
            st.sampled_from(["cpu", "cuda"]) if gpu_available else st.just("cpu")
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_jagged_elementwise_binary(
        self,
        num_jagged_dim: int,
        outer_dense_size: int,
        inner_dense_size: int,
        operation: str,
        dtype: torch.dtype,
        device_type: str,
    ) -> None:
        self._test_jagged_elementwise_binary(
            num_jagged_dim,
            outer_dense_size,
            inner_dense_size,
            operation,
            dtype,
            device_type,
        )

    @unittest.skipIf(*gpu_unavailable)
    @given(
        num_jagged_dim=st.just(1),
        outer_dense_size=st.integers(0, 8),
        inner_dense_size=st.sampled_from([16, 64, 96, 192]),
        operation=st.sampled_from(["add_jagged_output", "mul"]),
        dtype=st.just(torch.half),
        device_type=(
            st.sampled_from(["cpu", "cuda"]) if gpu_available else st.just("cpu")
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=4, deadline=None)
    def test_jagged_elementwise_binary_opt(
        self,
        num_jagged_dim: int,
        outer_dense_size: int,
        inner_dense_size: int,
        operation: str,
        dtype: torch.dtype,
        device_type: str,
    ) -> None:
        self._test_jagged_elementwise_binary(
            num_jagged_dim,
            outer_dense_size,
            inner_dense_size,
            operation,
            dtype,
            device_type,
        )

    @optests.dontGenerateOpCheckTests("tests that call torch.compile are slow")
    @unittest.skipIf(*symint_vector_unsupported())
    @given(
        num_jagged_dim=st.integers(1, 5),
        outer_dense_size=st.integers(2, 5),
        inner_dense_size=st.integers(2, 5),
        operation=st.sampled_from(["add", "add_jagged_output", "mul"]),
        dtype=st.sampled_from([torch.float, torch.half, torch.bfloat16]),
        device_type=(
            st.sampled_from(["cpu", "cuda"]) if gpu_available else st.just("cpu")
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_jagged_elementwise_binary_dynamic_shape(
        self,
        num_jagged_dim: int,
        outer_dense_size: int,
        inner_dense_size: int,
        operation: str,
        dtype: torch.dtype,
        device_type: str,
    ) -> None:
        # Start a fresh compile for each parameter of the test case
        torch._dynamo.reset()

        device = torch.device(device_type)

        x_values, x_offsets, max_lengths = self._generate_jagged_tensor(
            num_jagged_dim,
            outer_dense_size,
            inner_dense_size,
            dtype,
            device,
            mark_dynamic=True,
        )
        y = torch.rand(
            outer_dense_size * np.prod(max_lengths) * inner_dense_size,
            dtype=dtype,
            device=device,
        ).reshape((outer_dense_size,) + tuple(max_lengths) + (inner_dense_size,))

        x_padded = self._to_padded_dense(x_values, x_offsets, max_lengths)

        def jagged_dense_elementwise_add(
            x_values: torch.Tensor, x_offsets: List[torch.LongTensor], y: torch.Tensor
        ) -> torch.Tensor:
            return torch.ops.fbgemm.jagged_dense_elementwise_add(x_values, x_offsets, y)

        def jagged_dense_elementwise_add_jagged_output(
            x_values: torch.Tensor, x_offsets: List[torch.LongTensor], y: torch.Tensor
        ) -> Tuple[torch.Tensor, List[torch.LongTensor]]:
            return torch.ops.fbgemm.jagged_dense_elementwise_add_jagged_output(
                x_values, x_offsets, y
            )

        def jagged_dense_elementwise_mul(
            x_values: torch.Tensor, x_offsets: List[torch.LongTensor], y: torch.Tensor
        ) -> Tuple[torch.Tensor, List[torch.LongTensor]]:
            return torch.ops.fbgemm.jagged_dense_elementwise_mul(x_values, x_offsets, y)

        if operation == "add":
            output_ref = x_padded + y
            output = jagged_dense_elementwise_add(x_values, x_offsets, y)

        elif operation == "add_jagged_output":
            # create a jagged tensor and then densify
            y = self._to_padded_dense(
                torch.rand(
                    (
                        max(outer_dense_size * np.prod(max_lengths), x_values.size(0)),
                        inner_dense_size,
                    ),
                    dtype=dtype,
                    device=device,
                ),
                x_offsets,
                max_lengths,
            )
            output_ref = x_padded + y
            (
                output,
                output_offsets,
            ) = jagged_dense_elementwise_add_jagged_output(x_values, x_offsets, y)
            output = self._to_padded_dense(output, output_offsets, max_lengths)

        elif operation == "mul":
            output_ref = x_padded * y
            output, output_offsets = jagged_dense_elementwise_mul(
                x_values, x_offsets, y
            )
            output = self._to_padded_dense(output, output_offsets, max_lengths)
        else:
            raise AssertionError(f"Unknown operation {operation}")

        assert output.size() == output_ref.size()

    def _test_jagged_dense_dense_elementwise_add_jagged_output(
        self,
        num_jagged_dim: int,
        outer_dense_size: int,
        inner_dense_size: int,
        dtype: torch.dtype,
        device_type: str,
    ) -> None:
        device = torch.device(device_type)

        x_values, x_offsets, max_lengths = self._generate_jagged_tensor(
            num_jagged_dim, outer_dense_size, inner_dense_size, dtype, device
        )

        x_padded = self._to_padded_dense(x_values, x_offsets, max_lengths)
        # create a jagged tensor and then densify
        y_0 = self._to_padded_dense(
            torch.rand(
                (
                    max(outer_dense_size * np.prod(max_lengths), x_values.size(0)),
                    inner_dense_size,
                ),
                dtype=dtype,
                device=device,
            ),
            x_offsets,
            max_lengths,
        )
        y_1 = self._to_padded_dense(
            torch.rand(
                (
                    max(outer_dense_size * np.prod(max_lengths), x_values.size(0)),
                    inner_dense_size,
                ),
                dtype=dtype,
                device=device,
            ),
            x_offsets,
            max_lengths,
        )
        output_ref = x_padded + y_0 + y_1
        (
            output,
            output_offsets,
        ) = torch.ops.fbgemm.jagged_dense_dense_elementwise_add_jagged_output(
            x_values, x_offsets, y_0, y_1
        )
        output = self._to_padded_dense(output, output_offsets, max_lengths)

        torch.testing.assert_close(output, output_ref)

        # pyre-fixme[2]: Parameter must be annotated.
        def add_jagged_output_func(*args) -> torch.Tensor:
            return torch.ops.fbgemm.jagged_dense_dense_elementwise_add_jagged_output(
                *args
            )[0]

        f = add_jagged_output_func

        gradcheck(
            f,
            (
                x_values.float().requires_grad_(True),
                x_offsets,
                y_0.float().requires_grad_(True),
                y_1.float().requires_grad_(True),
            ),
            eps=1e-2,
            atol=1e-3,
            rtol=1e-3,
        )

    @given(
        num_jagged_dim=st.integers(1, 4),
        outer_dense_size=st.integers(0, 4),
        inner_dense_size=st.integers(0, 4),
        dtype=st.sampled_from([torch.float, torch.half, torch.bfloat16]),
        device_type=(
            st.sampled_from(["cpu", "cuda"]) if gpu_available else st.just("cpu")
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_jagged_dense_dense_elementwise_add_jagged_output(
        self,
        num_jagged_dim: int,
        outer_dense_size: int,
        inner_dense_size: int,
        dtype: torch.dtype,
        device_type: str,
    ) -> None:
        self._test_jagged_dense_dense_elementwise_add_jagged_output(
            num_jagged_dim, outer_dense_size, inner_dense_size, dtype, device_type
        )

    @unittest.skipIf(*gpu_unavailable)
    @given(
        num_jagged_dim=st.just(1),
        outer_dense_size=st.integers(0, 8),
        inner_dense_size=st.sampled_from([16, 64, 96, 192]),
        dtype=st.just(torch.half),
        device_type=(
            st.sampled_from(["cpu", "cuda"]) if gpu_available else st.just("cpu")
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=4, deadline=None)
    def test_jagged_dense_dense_elementwise_add_jagged_output_opt(
        self,
        num_jagged_dim: int,
        outer_dense_size: int,
        inner_dense_size: int,
        dtype: torch.dtype,
        device_type: str,
    ) -> None:
        self._test_jagged_dense_dense_elementwise_add_jagged_output(
            num_jagged_dim, outer_dense_size, inner_dense_size, dtype, device_type
        )

    @given(
        num_jagged_dim=st.integers(1, 4),
        outer_dense_size=st.integers(0, 4),
        inner_dense_size=st.integers(0, 4),
        dtype=st.sampled_from([torch.float, torch.half, torch.bfloat16]),
        device_type=st.just("meta"),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_jagged_dense_dense_elementwise_add_jagged_output_meta_backend(
        self,
        num_jagged_dim: int,
        outer_dense_size: int,
        inner_dense_size: int,
        dtype: torch.dtype,
        device_type: str,
    ) -> None:
        device = torch.device("cpu")

        x_values, x_offsets, max_lengths = self._generate_jagged_tensor(
            num_jagged_dim, outer_dense_size, inner_dense_size, dtype, device
        )

        x_padded = self._to_padded_dense(x_values, x_offsets, max_lengths)
        # create a jagged tensor and then densify
        y_0 = self._to_padded_dense(
            torch.rand(
                (
                    max(outer_dense_size * np.prod(max_lengths), x_values.size(0)),
                    inner_dense_size,
                ),
                dtype=dtype,
                device=device,
            ),
            x_offsets,
            max_lengths,
        )
        y_1 = self._to_padded_dense(
            torch.rand(
                (
                    max(outer_dense_size * np.prod(max_lengths), x_values.size(0)),
                    inner_dense_size,
                ),
                dtype=dtype,
                device=device,
            ),
            x_offsets,
            max_lengths,
        )
        output_ref = x_padded + y_0 + y_1
        x_values.to(device_type)
        (
            output,
            output_offsets,
        ) = torch.ops.fbgemm.jagged_dense_dense_elementwise_add_jagged_output(
            x_values, x_offsets, y_0, y_1
        )
        output.to("cpu")
        output = self._to_padded_dense(output, output_offsets, max_lengths)

        assert output.size() == output_ref.size()

    @optests.dontGenerateOpCheckTests("tests that call torch.compile are slow")
    @unittest.skipIf(*symint_vector_unsupported())
    @given(
        num_jagged_dim=st.integers(1, 4),
        outer_dense_size=st.integers(2, 4),
        inner_dense_size=st.integers(2, 4),
        dtype=st.sampled_from([torch.float, torch.half, torch.bfloat16]),
        device_type=st.just("cpu"),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_jagged_dense_dense_elementwise_add_jagged_output_dynamic_shape(
        self,
        num_jagged_dim: int,
        outer_dense_size: int,
        inner_dense_size: int,
        dtype: torch.dtype,
        device_type: str,
    ) -> None:
        # Start a fresh compile for each parameter of the test case
        torch._dynamo.reset()

        x_values, x_offsets, max_lengths = self._generate_jagged_tensor(
            num_jagged_dim,
            outer_dense_size,
            inner_dense_size,
            dtype,
            torch.device(device_type),
            mark_dynamic=True,
        )

        x_padded = self._to_padded_dense(x_values, x_offsets, max_lengths)
        # create a jagged tensor and then densify
        y_0 = self._to_padded_dense(
            torch.rand(
                (
                    max(outer_dense_size * np.prod(max_lengths), x_values.size(0)),
                    inner_dense_size,
                ),
                dtype=dtype,
                device=torch.device(device_type),
            ),
            x_offsets,
            max_lengths,
        )
        y_1 = self._to_padded_dense(
            torch.rand(
                (
                    max(outer_dense_size * np.prod(max_lengths), x_values.size(0)),
                    inner_dense_size,
                ),
                dtype=dtype,
                device=torch.device(device_type),
            ),
            x_offsets,
            max_lengths,
        )
        output_ref = x_padded + y_0 + y_1
        x_values.to(device_type)
        (output, output_offsets) = torch_compiled(
            torch.ops.fbgemm.jagged_dense_dense_elementwise_add_jagged_output,
            fullgraph=True,
            dynamic=True,
        )(x_values, x_offsets, y_0, y_1)
        output.to("cpu")
        output = self._to_padded_dense(output, output_offsets, max_lengths)

        assert output.size() == output_ref.size()

    @settings(
        verbosity=Verbosity.verbose,
        max_examples=20,
        deadline=None,
    )
    @given(
        B=st.integers(0, 32),
        H=st.integers(1, 3),
        max_L=st.integers(1, 32),
        D=st.integers(0, 32),
        dtype=st.sampled_from([torch.float, torch.half, torch.bfloat16]),
        device_type=(
            st.sampled_from(["cpu", "cuda"]) if gpu_available else st.just("cpu")
        ),
    )
    def test_batched_dense_vec_jagged_2d_mul(
        self,
        B: int,
        H: int,
        max_L: int,
        D: int,
        dtype: torch.dtype,
        device_type: str,
    ) -> None:
        assume(H == 1 or B != 0)
        # CPU doesn't support bfloat16
        assume(device_type != "cpu" or dtype != torch.bfloat16)

        device = torch.device(device_type)
        torch.backends.cuda.matmul.allow_tf32 = False

        # Sometimes length[i] exceed max_L meaning jagged->dense will be
        # truncation vs. padding
        lengths = torch.randint(max_L * 2, size=(B,), device=device)
        offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
        values = torch.rand((offsets[-1], H * D), dtype=dtype, device=device)
        dense = torch.rand((B * H, max_L), dtype=dtype, device=device)
        padded_values = torch.ops.fbgemm.jagged_to_padded_dense(
            values,
            [offsets],
            [max_L],
        )  # [B, N, H * D]

        bmm_arg1 = dense.unsqueeze(1)
        bmm_arg2 = (
            padded_values.reshape(B, max_L, H, D)
            .transpose(1, 2)
            .reshape(B * H, max_L, D)
        )
        # torch.bmm not implemented for Half on CPU
        if dtype in [torch.half, torch.bfloat16] and device_type == "cpu":
            bmm_arg1 = bmm_arg1.float()
            bmm_arg2 = bmm_arg2.float()
        output_ref = torch.bmm(bmm_arg1, bmm_arg2).squeeze(
            1
        )  # [B H, 1, N] x [B H, N, D] = [B H, 1, D]
        if dtype in [torch.half, torch.bfloat16] and device_type == "cpu":
            output_ref = output_ref.to(dtype)
        output = torch.ops.fbgemm.batched_dense_vec_jagged_2d_mul(
            dense, values, offsets
        )
        torch.testing.assert_close(
            output,
            output_ref,
            rtol=1e-2 if dtype in [torch.half, torch.bfloat16] else None,
            atol=1e-2 if dtype in [torch.half, torch.bfloat16] else None,
        )

        gradcheck(
            torch.ops.fbgemm.batched_dense_vec_jagged_2d_mul,
            (
                dense.clone().detach().float().requires_grad_(True),
                values.clone().detach().float().requires_grad_(True),
                offsets,
            ),
            eps=1e-2,
            atol=1e-3,
            rtol=1e-3,
        )

    @settings(
        verbosity=Verbosity.verbose,
        max_examples=20,
        deadline=None,
    )
    @given(
        B=st.integers(0, 32),
        H=st.integers(1, 3),
        max_L=st.integers(1, 32),
        D=st.integers(0, 32),
        dtype=st.sampled_from([torch.float, torch.half, torch.bfloat16]),
        device_type=st.sampled_from(["meta"]),
    )
    def test_batched_dense_vec_jagged_2d_mul_meta_backend(
        self,
        B: int,
        H: int,
        max_L: int,
        D: int,
        dtype: torch.dtype,
        device_type: str,
    ) -> None:
        assume(H == 1 or B != 0)

        device = torch.device("cpu")
        torch.backends.cuda.matmul.allow_tf32 = False

        # Sometimes length[i] exceed max_L meaning jagged->dense will be
        # truncation vs. padding
        lengths = torch.randint(max_L * 2, size=(B,), device=device)
        offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
        values = torch.rand((offsets[-1], H * D), dtype=dtype, device=device)
        dense = torch.rand((B * H, max_L), dtype=dtype, device=device)
        padded_values = torch.ops.fbgemm.jagged_to_padded_dense(
            values,
            [offsets],
            [max_L],
        )  # [B, N, H * D]

        bmm_arg1 = dense.unsqueeze(1)
        bmm_arg2 = (
            padded_values.reshape(B, max_L, H, D)
            .transpose(1, 2)
            .reshape(B * H, max_L, D)
        )
        # torch.bmm not implemented for Half on CPU
        if dtype in [torch.half, torch.bfloat16]:
            bmm_arg1 = bmm_arg1.float()
            bmm_arg2 = bmm_arg2.float()
        output_ref = torch.bmm(bmm_arg1, bmm_arg2).squeeze(
            1
        )  # [B H, 1, N] x [B H, N, D] = [B H, 1, D]
        dense.to(device_type)
        values.to(device_type)
        output = torch.ops.fbgemm.batched_dense_vec_jagged_2d_mul(
            dense, values, offsets
        )
        assert output.size() == output_ref.size()

    @optests.dontGenerateOpCheckTests("tests that call torch.compile are slow")
    @unittest.skipIf(*symint_vector_unsupported())
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=20,
        deadline=None,
    )
    @given(
        B=st.integers(2, 32),
        H=st.integers(1, 3),
        max_L=st.integers(1, 32),
        D=st.integers(2, 32),
        dtype=st.sampled_from([torch.float, torch.half, torch.bfloat16]),
        device_type=st.just("cpu"),
    )
    def test_batched_dense_vec_jagged_2d_mul_dynamic_shape(
        self,
        B: int,
        H: int,
        max_L: int,
        D: int,
        dtype: torch.dtype,
        device_type: str,
    ) -> None:
        # Start a fresh compile for each parameter of the test case
        torch._dynamo.reset()

        assume(H == 1 or B != 0)

        device = torch.device(device_type)
        torch.backends.cuda.matmul.allow_tf32 = False

        # Sometimes length[i] exceed max_L meaning jagged->dense will be
        # truncation vs. padding
        lengths = torch.randint(low=1, high=max_L * 2, size=(B,), device=device)
        offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
        values = torch.rand((offsets[-1], H * D), dtype=dtype, device=device)
        dense = torch.rand((B * H, max_L), dtype=dtype, device=device)
        padded_values = torch.ops.fbgemm.jagged_to_padded_dense(
            values,
            [offsets],
            [max_L],
        )  # [B, N, H * D]

        bmm_arg1 = dense.unsqueeze(1)
        bmm_arg2 = (
            padded_values.reshape(B, max_L, H, D)
            .transpose(1, 2)
            .reshape(B * H, max_L, D)
        )
        # torch.bmm not implemented for Half on CPU
        if dtype in [torch.half, torch.bfloat16]:
            bmm_arg1 = bmm_arg1.float()
            bmm_arg2 = bmm_arg2.float()
        output_ref = torch.bmm(bmm_arg1, bmm_arg2).squeeze(
            1
        )  # [B H, 1, N] x [B H, N, D] = [B H, 1, D]
        dense.to(device_type)
        values.to(device_type)

        torch._dynamo.mark_dynamic(dense, 0)
        torch._dynamo.mark_dynamic(values, 0)
        torch._dynamo.mark_dynamic(values, 1)
        torch._dynamo.mark_dynamic(offsets, 0)

        output = torch_compiled(
            torch.ops.fbgemm.batched_dense_vec_jagged_2d_mul,
            fullgraph=True,
            dynamic=True,
        )(dense, values, offsets)
        assert output.size() == output_ref.size()

    @staticmethod
    def jagged_index_select_2d_ref(
        values: torch.Tensor,
        lengths: torch.Tensor,
        inverse_lookup: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        offsets = torch.ops.fbgemm.asynchronous_exclusive_cumsum(lengths)
        end_offsets = offsets + lengths
        full_start_offset = torch.index_select(offsets, 0, inverse_lookup)
        full_end_offset = torch.index_select(end_offsets, 0, inverse_lookup)
        index_ranges = torch.stack(
            (full_start_offset, full_end_offset), dim=0
        ).transpose(0, 1)

        to_be_merged_tensors = []
        for row in index_ranges:
            to_be_merged_tensors.append(torch.arange(row[0], row[1], device=device))
        all_indices = torch.cat(to_be_merged_tensors, dim=0)
        new_embeddings = torch.index_select(values, 0, all_indices)
        return new_embeddings

    @given(
        max_seq_length=st.integers(5, 10),
        batch_size=st.integers(1, 128),
        num_cols=st.integers(1, 128),
        num_jagged_tensor_rows=st.integers(1, 128),
        index_dtype=st.sampled_from([torch.int, torch.long]),
        jagged_tensor_dtype=st.sampled_from(
            [
                torch.float,
                torch.half,
                torch.int,
                torch.long,
            ]  # Disable torch.bfloat16 due to large error bound
        ),
        use_cpu=use_cpu_strategy(),
        check_non_contiguous=st.booleans(),
        known_shape=st.booleans(),
    )
    @settings(max_examples=20, deadline=None, verbosity=Verbosity.verbose)
    def test_jagged_index_select_2d(
        self,
        max_seq_length: int,
        batch_size: int,
        num_cols: int,
        num_jagged_tensor_rows: int,
        index_dtype: torch.dtype,
        jagged_tensor_dtype: torch.dtype,
        use_cpu: bool,
        check_non_contiguous: bool,
        known_shape: bool,
    ) -> None:
        device = torch.device("cpu" if use_cpu else "cuda")
        is_float = jagged_tensor_dtype in [torch.float, torch.half, torch.bfloat16]
        lengths = torch.randint(
            low=0,
            high=max_seq_length,
            size=(num_jagged_tensor_rows,),
            dtype=index_dtype,
            device=device,
        )
        indices, _ = torch.sort(
            torch.randint(
                low=0,
                high=num_jagged_tensor_rows,
                size=(batch_size,),
                dtype=index_dtype,
                device=device,
            )
        )
        if is_float:
            values = torch.rand(
                int(lengths.sum().item()),
                num_cols,
                dtype=jagged_tensor_dtype,
                device=device,
            )
        else:
            values = torch.randint(
                2**16,
                (int(lengths.sum().item()), num_cols),
                dtype=jagged_tensor_dtype,
                device=device,
            )
        values_ref = values.detach().clone()

        if check_non_contiguous:
            values = values.as_strided(values.shape, (1, values.shape[0]))
            values_ref = values_ref.as_strided(values.shape, (1, values.shape[0]))

        # Only float tensors can require grad
        if is_float:
            values.requires_grad = True
            values_ref.requires_grad = True

        if known_shape:
            with torch.no_grad():
                tmp_output, _ = torch.ops.fbgemm.jagged_index_select(
                    values, lengths, indices
                )
            num_dense_output_rows = tmp_output.shape[0]
            output, _ = torch.ops.fbgemm.jagged_index_select(
                values, lengths, indices, num_dense_output_rows
            )
        else:
            output, _ = torch.ops.fbgemm.jagged_index_select(values, lengths, indices)
        output_ref = self.jagged_index_select_2d_ref(
            values_ref, lengths, indices, device
        )

        assert torch.equal(output, output_ref)

        if not is_float:
            return

        grad = torch.rand_like(output)
        grad_ref = grad.detach().clone()

        if check_non_contiguous:
            grad = grad.as_strided(grad.shape, (1, grad.shape[0]))
            grad_ref = grad_ref.as_strided(grad.shape, (1, grad.shape[0]))

        output.backward(grad)
        output_ref.backward(grad_ref)

        torch.testing.assert_close(
            values.grad,
            values_ref.grad,
            rtol=1e-2 if jagged_tensor_dtype in [torch.half, torch.bfloat16] else None,
            atol=1e-2 if jagged_tensor_dtype in [torch.half, torch.bfloat16] else None,
        )

    @given(
        max_seq_length=st.integers(5, 10),
        batch_size=st.integers(1, 128),
        num_cols=st.integers(1, 128),
        num_jagged_tensor_rows=st.integers(1, 128),
        index_dtype=st.sampled_from([torch.int, torch.long]),
        jagged_tensor_dtype=st.sampled_from(
            [
                torch.float,
                torch.half,
                torch.int,
                torch.long,
            ]  # Disable torch.bfloat16 due to large error bound
        ),
        use_cpu=use_cpu_strategy(),
    )
    @settings(max_examples=20, deadline=None)
    def test_jagged_index_select_2d_in_inference(
        self,
        max_seq_length: int,
        batch_size: int,
        num_cols: int,
        num_jagged_tensor_rows: int,
        index_dtype: torch.dtype,
        jagged_tensor_dtype: torch.dtype,
        use_cpu: bool,
    ) -> None:
        device = torch.device("cpu" if use_cpu else "cuda")
        is_float = jagged_tensor_dtype in [torch.float, torch.half, torch.bfloat16]
        lengths = torch.randint(
            low=0,
            high=max_seq_length,
            size=(num_jagged_tensor_rows,),
            dtype=index_dtype,
            device=device,
        )
        indices, _ = torch.sort(
            torch.randint(
                low=0,
                high=num_jagged_tensor_rows,
                size=(batch_size,),
                dtype=index_dtype,
                device=device,
            )
        )
        if is_float:
            values = torch.rand(
                int(lengths.sum().item()),
                num_cols,
                dtype=jagged_tensor_dtype,
                device=device,
            )
        else:
            values = torch.randint(
                2**16,
                (int(lengths.sum().item()), num_cols),
                dtype=jagged_tensor_dtype,
                device=device,
            )
        values_ref = values.detach().clone()

        with torch.inference_mode():
            output, _ = torch.ops.fbgemm.jagged_index_select(values, lengths, indices)
            output_ref = self.jagged_index_select_2d_ref(
                values_ref, lengths, indices, device
            )
            assert torch.equal(output, output_ref)

    @given(
        batch_size=st.integers(1, 128),
        max_length=st.integers(0, 128),
        max_truncated_length=st.integers(1, 32),
        index_dtype=st.sampled_from([torch.int, torch.long]),
        jagged_tensor_dtype=st.sampled_from(
            [torch.float, torch.half, torch.bfloat16, torch.int, torch.long]
        ),
        use_cpu=st.just(True),
    )
    @settings(max_examples=20, deadline=None)
    def test_jagged_1d_to_truncated_values(
        self,
        max_length: int,
        batch_size: int,
        max_truncated_length: int,
        index_dtype: torch.dtype,
        jagged_tensor_dtype: torch.dtype,
        use_cpu: bool,
    ) -> None:
        device = "cpu" if use_cpu else "cuda"
        is_float = jagged_tensor_dtype in [torch.float, torch.half, torch.bfloat16]
        lengths = torch.randint(
            low=0,
            high=max_length + 1,
            size=(batch_size,),
            dtype=index_dtype,
            device=device,
        )
        n = int(lengths.sum().item())
        if is_float:
            values = torch.rand(
                (n,),
                dtype=jagged_tensor_dtype,
                device=device,
            )
        else:
            values = torch.randint(
                2**16,
                (n,),
                dtype=jagged_tensor_dtype,
                device=device,
            )

        truncated_values = torch.ops.fbgemm.jagged_1d_to_truncated_values(
            values,
            lengths,
            max_truncated_length,
        )
        dense_values = torch.ops.fbgemm.jagged_1d_to_dense(
            values=values,
            offsets=torch.ops.fbgemm.asynchronous_complete_cumsum(lengths),
            max_sequence_length=max_truncated_length,
            padding_value=0,
        )  # [B, N]
        truncated_lengths_ref = torch.clamp(lengths, max=max_truncated_length)
        mask2d = torch.arange(max_truncated_length, device=device).expand(
            batch_size, -1
        ) < truncated_lengths_ref.unsqueeze(-1)
        truncated_values_ref = dense_values[mask2d].view(-1)

        torch.testing.assert_close(truncated_values, truncated_values_ref)

    @given(
        batch_size=st.integers(1, 128),
        max_length=st.integers(0, 128),
        index_dtype=st.sampled_from([torch.int, torch.long]),
        jagged_tensor_dtype=st.sampled_from([torch.int, torch.long]),
        empty_lengths=st.booleans(),
        use_cpu=st.just(True),
    )
    @settings(max_examples=20, deadline=None)
    def test_masked_select_jagged_1d(
        self,
        max_length: int,
        batch_size: int,
        index_dtype: torch.dtype,
        jagged_tensor_dtype: torch.dtype,
        empty_lengths: bool,
        use_cpu: bool,
    ) -> None:
        device = "cpu" if use_cpu else "cuda"
        if empty_lengths:
            lengths = torch.zeros(batch_size, dtype=index_dtype, device=device)
        else:
            lengths = torch.randint(
                low=0,
                high=max_length + 1,
                size=(batch_size,),
                dtype=index_dtype,
                device=device,
            )
        lengths[batch_size // 2] = 0  # test a corner case
        n = int(lengths.sum().item())
        values = torch.randint(
            2**16,
            (n,),
            dtype=jagged_tensor_dtype,
            device=device,
        )
        mask = torch.randint(2, (n,)) > 0

        masked_values, masked_lengths = torch.ops.fbgemm.masked_select_jagged_1d(
            values,
            lengths,
            mask,
        )

        masked_values_ref = values[mask]
        cum_count = torch.cumsum(mask, 0)
        cum_count = torch.cat((cum_count, torch.tensor([0])))
        cum_length = cum_count[torch.cumsum(lengths, 0) - 1]
        cum_length_shift_right = torch.roll(cum_length, 1)
        cum_length_shift_right[0] = 0
        masked_lengths_ref = (cum_length - cum_length_shift_right).to(lengths.dtype)

        torch.testing.assert_close(masked_values, masked_values_ref)
        torch.testing.assert_close(masked_lengths, masked_lengths_ref)

    @unittest.skipIf(*gpu_unavailable)
    @given(
        max_seq_length=st.integers(5, 10),
        input_batch_size=st.integers(1, 128),
        output_batch_size=st.integers(1, 128),
        num_batches=st.integers(1, 3),
        index_dtype=st.sampled_from([torch.int, torch.long]),
        jagged_tensor_dtype=st.sampled_from(
            [
                torch.float,
                torch.half,
                torch.int,
                torch.long,
            ]  # Disable torch.bfloat16 due to large error bound
        ),
        has_weights=st.booleans(),
        check_non_contiguous=st.booleans(),
        use_selected_lengths_sum=st.booleans(),
    )
    @settings(max_examples=20, deadline=None)
    def test_keyed_jagged_index_select_dim1(
        self,
        max_seq_length: int,
        input_batch_size: int,
        output_batch_size: int,
        num_batches: int,
        index_dtype: torch.dtype,
        jagged_tensor_dtype: torch.dtype,
        has_weights: bool,
        check_non_contiguous: bool,
        use_selected_lengths_sum: bool,
    ) -> None:
        is_float = jagged_tensor_dtype in [torch.float, torch.half, torch.bfloat16]
        lengths = torch.randint(
            low=0,
            high=max_seq_length,
            size=(input_batch_size * num_batches,),
            dtype=index_dtype,
            device="cuda",
        )
        offsets = torch.concat(
            [torch.zeros(1, dtype=torch.long, device="cuda"), lengths.cumsum(0)]
        )
        indices = torch.randint(
            low=0,
            high=1,
            size=(output_batch_size,),
            dtype=index_dtype,
            device="cuda",
        )

        # If check_non_contiguous=True, create a tensor that is twice as big
        # and then select only odd indices to make it non contiguous
        values_numel = int(offsets[-1].item())
        values_numel = values_numel * 2 if check_non_contiguous else values_numel

        if is_float:
            values = torch.rand(
                values_numel,
                dtype=jagged_tensor_dtype,
                device="cuda",
            )
        else:
            values = torch.randint(
                2**16,
                (values_numel,),
                dtype=jagged_tensor_dtype,
                device="cuda",
            )
        values_ref = values.detach().clone()

        if check_non_contiguous:
            values = values[1::2]
            values_ref = values_ref[1::2]

        if has_weights:
            weights = torch.rand(
                int(offsets[-1].item()),
                dtype=random.choice([torch.float, torch.half]),
                device="cuda",
            )
        else:
            weights = None

        if use_selected_lengths_sum:
            length_indices = torch.cat(
                [indices + i * input_batch_size for i in range(num_batches)]
            )
            selected_lengths_sum = (
                torch.index_select(lengths, 0, length_indices).sum().item()
            )
        else:
            selected_lengths_sum = None

        # Only float tensors can require grad
        if is_float:
            values.requires_grad = True
            values_ref.requires_grad = True

        index_select_output = torch.ops.fbgemm.keyed_jagged_index_select_dim1(
            values,
            lengths,
            offsets,
            indices,
            input_batch_size,
            weights,
            selected_lengths_sum,
        )
        output = index_select_output[0]
        if has_weights:
            output_weights = index_select_output[2]

        output_ref = []
        output_weight_ref = []
        for k in range(num_batches):
            key_lengths = lengths[k * input_batch_size : (k + 1) * input_batch_size]
            start_offset = offsets[k * input_batch_size]
            end_offset = offsets[(k + 1) * input_batch_size]
            key_values = values_ref[start_offset:end_offset].view(-1, 1)
            output_ref.append(
                torch.ops.fbgemm.jagged_index_select(key_values, key_lengths, indices)[
                    0
                ].view(-1)
            )
            if has_weights:
                # pyre-ignore[16]
                key_weights = weights[start_offset:end_offset].view(-1, 1)
                output_weight_ref.append(
                    torch.ops.fbgemm.jagged_index_select(
                        key_weights, key_lengths, indices
                    )[0].view(-1)
                )

        output_ref = torch.concat(output_ref)
        assert torch.equal(output, output_ref)

        if has_weights:
            output_weight_ref = torch.concat(output_weight_ref)
            # pyre-ignore[61]
            assert torch.equal(output_weights, output_weight_ref)

        if not is_float:
            return

        # If check_non_contiguous=True, create a tensor that is twice as big
        # and then select only odd indices to make it non contiguous
        grad_numel = output.numel()
        grad_numel = grad_numel * 2 if check_non_contiguous else grad_numel

        grad = torch.rand(grad_numel, dtype=output.dtype, device=output.device)
        grad_ref = grad.detach().clone()

        if check_non_contiguous:
            grad = grad[1::2]
            grad_ref = grad_ref[1::2]

        output.backward(grad)
        output_ref.backward(grad_ref)

        torch.testing.assert_close(
            values.grad,
            values_ref.grad,
            rtol=1e-2 if jagged_tensor_dtype in [torch.half, torch.bfloat16] else None,
            atol=1e-2 if jagged_tensor_dtype in [torch.half, torch.bfloat16] else None,
        )

    @given(
        B=st.integers(1, 512),
        max_L=st.integers(1, 1000),
        D=st.integers(1, 32),
        dtype=st.sampled_from([torch.float]),
        device_type=(
            st.sampled_from(["cpu", "cuda"]) if gpu_available else st.just("cpu")
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_jagged_softmax(
        self,
        B: int,
        max_L: int,
        D: int,
        dtype: torch.dtype,
        device_type: str,
    ) -> None:
        device = torch.device(device_type)
        torch.backends.cuda.matmul.allow_tf32 = False
        lengths = torch.randint(max_L + 1, size=(B,), device=device)
        total_length = int(lengths.sum().item())
        offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
        values = torch.rand(
            (total_length, D), requires_grad=True, dtype=dtype, device=device
        )
        output, _ = torch.ops.fbgemm.jagged_softmax(
            values,
            offsets,
            max_L,
        )
        values_ref = values.detach().clone().requires_grad_(True)
        output_ref, _ = torch.ops.fbgemm.dense_to_jagged(
            torch.nn.functional.softmax(
                torch.ops.fbgemm.jagged_to_padded_dense(
                    values_ref,
                    [offsets],
                    max_lengths=[max_L],
                    padding_value=-5e7,
                ).transpose(1, 2),
                dim=-1,
            ).permute(0, 2, 1),
            [offsets],
            total_length,
        )

        # verify forward
        torch.testing.assert_close(output, output_ref)

        # verify backward
        grad_output = output.detach().clone().requires_grad_(True)

        output.backward(grad_output)
        output_ref.backward(grad_output)

        torch.testing.assert_close(values.grad, values_ref.grad)


if __name__ == "__main__":
    unittest.main()
