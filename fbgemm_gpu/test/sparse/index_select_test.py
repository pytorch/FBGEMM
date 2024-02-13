#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors[56]

import contextlib
import functools
import logging
import random
import unittest
from typing import Callable, Dict, List

import hypothesis.strategies as st
import numpy as np
import torch
from hypothesis import given, settings, Verbosity

from .common import extend_test_class, open_source

if open_source:
    # pyre-ignore[21]
    from test_utils import gpu_available
else:
    import fbgemm_gpu.sparse_ops  # noqa: F401, E402
    from fbgemm_gpu.test.test_utils import gpu_available


class IndexSelectTest(unittest.TestCase):
    @given(
        N=st.integers(1, 32),
        shape=st.one_of(
            st.lists(st.integers(1, 128), max_size=1),
            st.lists(st.integers(1, 16), min_size=2, max_size=2),
        ),
        dtype=st.sampled_from([torch.float, torch.half, torch.double]),
        use_cpu=st.booleans() if gpu_available else st.just(True),
        consecutive_indices=st.booleans(),
        skip_indices_sorting_fwd=st.booleans(),
        use_inference_mode=st.booleans(),
    )
    @settings(max_examples=20, deadline=None)
    def test_index_select_dim0(
        self,
        N: int,
        shape: List[int],
        dtype: torch.dtype,
        use_cpu: bool,
        consecutive_indices: bool,
        skip_indices_sorting_fwd: bool,
        use_inference_mode: bool,
    ) -> None:
        device = torch.device("cpu" if use_cpu else "cuda")
        U = random.randint(0, N + 1)

        kwargs = {}
        if consecutive_indices:
            start = np.random.randint(0, U)
            length = np.random.randint(1, U - start + 1)
            indices = list(range(start, start + length))
            np_arr = np.array(indices)
            for _ in range(N - U):
                indices.append(np.random.randint(start, start + length))
                np_arr = np.array(indices)
                np.random.shuffle(np_arr)
            indices = torch.from_numpy(np_arr).to(torch.int).to(device)
            kwargs["consecutive_range_start"] = start
            kwargs["consecutive_range_length"] = length
        else:
            indices = torch.randint(U, (N,), device=device)

        kwargs["skip_indices_sorting_fwd"] = skip_indices_sorting_fwd

        input = torch.rand((U,) + tuple(shape), dtype=dtype, device=device)

        with torch.inference_mode() if use_inference_mode else contextlib.nullcontext():
            output_ref = torch.ops.fbgemm.index_select_dim0(input, indices, **kwargs)
            output = torch.index_select(input, 0, indices)

            torch.testing.assert_close(output, output_ref)

        if not use_inference_mode:
            gradcheck_args = [
                input.clone().detach().double().requires_grad_(True),
                indices,
            ]
            for k in kwargs:
                gradcheck_args.append(kwargs[k])

            torch.autograd.gradcheck(torch.ops.fbgemm.index_select_dim0, gradcheck_args)

    @given(
        num_indices=st.integers(1, 32),
        max_num_input_rows=st.integers(1, 32),
        shape=st.lists(st.integers(1, 32), min_size=1, max_size=2),
        dtype=st.sampled_from([torch.float, torch.half, torch.double]),
        use_cpu=st.booleans() if gpu_available else st.just(True),
        num_groups=st.integers(1, 32),
        use_var_cols=st.booleans(),
        use_var_num_input_rows=st.booleans(),
        check_non_contiguous=st.booleans(),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=20,
        deadline=None,
    )
    def test_group_index_select_dim0(
        self,
        num_indices: int,
        max_num_input_rows: int,
        shape: List[int],
        dtype: torch.dtype,
        use_cpu: bool,
        num_groups: int,
        use_var_cols: bool,
        use_var_num_input_rows: bool,
        check_non_contiguous: bool,
    ) -> None:
        device = torch.device("cpu" if use_cpu else "cuda")

        input_group: List[torch.Tensor] = []
        input_ref_group: List[torch.Tensor] = []
        indices_group: List[torch.Tensor] = []
        grad_group: List[torch.Tensor] = []
        for _ in range(num_groups):
            if use_var_num_input_rows:
                num_input_rows = (
                    random.randint(1, max_num_input_rows)
                    if max_num_input_rows > 1
                    else 1
                )
            else:
                num_input_rows = max_num_input_rows
            indices = torch.randint(num_input_rows, (num_indices,), device=device)
            assert indices.max() < num_input_rows

            if use_var_cols:
                var_dim = random.randint(0, len(shape) - 1)
                new_shape = random.randint(1, 32)
                shape[var_dim] = new_shape
            indices_group.append(indices)
            input = torch.rand(
                (num_input_rows,) + tuple(shape), dtype=dtype, device=device
            )
            input_ref = input.clone().detach()

            input.requires_grad = True
            input_ref.requires_grad = True

            input_group.append(input)
            input_ref_group.append(input_ref)

            grad = torch.rand((num_indices,) + tuple(shape), dtype=dtype, device=device)
            grad_group.append(grad)

        # Test forward
        output_ref_group = []
        for input, indices in zip(input_ref_group, indices_group):
            output_ref_group.append(torch.index_select(input, 0, indices))

        output_group = torch.ops.fbgemm.group_index_select_dim0(
            input_group, indices_group
        )

        # Test backward
        for out, grad in zip(output_ref_group, grad_group):
            out.backward(grad)

        cat_output = torch.concat(
            [
                (
                    # Transpose is likely going to make the tensor
                    # noncontiguous
                    output.transpose(1, 0).flatten()
                    if check_non_contiguous
                    else output.flatten()
                )
                for output in output_group
            ]
        )

        cat_grad = torch.concat(
            [
                (
                    # Transpose is likely going to make the tensor
                    # noncontiguous
                    grad.transpose(1, 0).flatten()
                    if check_non_contiguous
                    else grad.flatten()
                )
                for grad in grad_group
            ]
        )
        cat_output.backward(cat_grad)

        def compare_tensor_groups(
            test_group: List[torch.Tensor],
            ref_group: List[torch.Tensor],
            tensor_type: str,
            tols: Dict["str", float],
        ) -> None:
            passed = True
            failure_count = 0
            for i, (test, ref) in enumerate(zip(test_group, ref_group)):
                # pyre-ignore [6]
                if not torch.allclose(test, ref, **tols):
                    passed = False
                    failure_count += 1
                    print(
                        f"FAILED: group {i} {tensor_type} ({dtype}), "
                        f"input shape {input_group[i].shape}, indices "
                        f"{indices_group[i]}, test {test}, ref {ref}"
                    )
            assert (
                passed
            ), f"{failure_count}/{num_groups} groups of {tensor_type} failed"

        compare_tensor_groups(
            output_group, output_ref_group, "activation", {"rtol": 0, "atol": 0}
        )
        compare_tensor_groups(
            # pyre-ignore [6]
            [i.grad for i in input_group],
            # pyre-ignore [6]
            [i.grad for i in input_ref_group],
            "gradient",
            {"rtol": 1e-02, "atol": 1e-02} if dtype == torch.half else {},
        )

    @given(
        num_inputs=st.integers(0, 100),
        max_input_rows=st.integers(2, 32),
        max_cols_factor=st.integers(2, 256),
        max_output_rows=st.integers(2, 32),
        permute_output_dim_0_1=st.booleans(),
        dtype=st.sampled_from([torch.float, torch.half]),
        use_cpu=st.booleans() if gpu_available else st.just(True),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_batch_index_select_dim0(  # noqa: C901
        self,
        num_inputs: int,
        max_input_rows: int,
        max_cols_factor: int,
        max_output_rows: int,
        permute_output_dim_0_1: bool,
        dtype: torch.dtype,
        use_cpu: bool,
    ) -> None:
        device = "cpu" if use_cpu else "cuda"
        input_rows = torch.randint(
            low=1, high=max_input_rows, size=(num_inputs,)
        ).tolist()
        input_columns = (
            torch.randint(low=1, high=max_cols_factor, size=(num_inputs,)) * 4
        ).tolist()
        if permute_output_dim_0_1:
            # All num_indices must be the same if permute_output_dim_0_1 is
            # True
            num_indices = torch.randint(low=1, high=max_output_rows, size=(1,)).item()
            input_num_indices = [num_indices] * num_inputs
        else:
            input_num_indices = torch.randint(
                low=1, high=max_output_rows, size=(num_inputs,)
            ).tolist()

        def validate(
            test_list: List[torch.Tensor],
            ref_list: List[torch.Tensor],
            rows: List[int],
            val_fn: Callable[[torch.Tensor, torch.Tensor], bool],
            name: str,
        ) -> None:
            test_passed_all = True
            error_msg = ""
            for i, (test, ref) in enumerate(zip(test_list, ref_list)):
                test = test.float()
                ref = ref.float()
                test_passed = val_fn(test, ref)
                test_passed_all = test_passed & test_passed_all
                if not test_passed:
                    test = test.reshape(rows[i], -1)
                    ref = ref.reshape(rows[i], -1)
                    for r in range(rows[i]):
                        test_row = test[r]
                        ref_row = ref[r]
                        if not val_fn(test_row, ref_row):
                            error_msg += f"ERROR: {name} {i} row {r} are different, test {test_row}, ref {ref_row}\n"
            assert test_passed_all, error_msg
            logging.info(f"{name} test passed")

        if num_inputs == 0:
            inputs = [torch.empty(0, dtype=dtype, device=device)]
            indices = [torch.empty(0, dtype=torch.long, device=device)]
        else:
            inputs = [
                torch.rand(rows, cols, dtype=dtype, device=device)
                for rows, cols in zip(input_rows, input_columns)
            ]
            indices = [
                torch.randint(
                    low=0, high=rows, size=(num,), dtype=torch.long, device=device
                )
                for num, rows in zip(input_num_indices, input_rows)
            ]

        for i in range(len(inputs)):
            inputs[i].requires_grad = True

        output_ref = [
            input.index_select(dim=0, index=index).flatten()
            for input, index in zip(inputs, indices)
        ]

        concat_inputs = torch.concat(
            [input.flatten().clone().detach() for input in inputs]
        )
        concat_indices = torch.concat(indices)

        concat_inputs.requires_grad = True

        output_test = torch.ops.fbgemm.batch_index_select_dim0(
            concat_inputs,
            concat_indices,
            input_num_indices,
            input_rows,
            input_columns,
            permute_output_dim_0_1,
        )

        if permute_output_dim_0_1 and num_inputs > 0:
            output_list = output_test.view(input_num_indices[0], -1).split(
                input_columns,
                dim=1,
            )
            output_list = [out.flatten() for out in output_list]
        else:
            output_list = output_test.split(
                [rows * cols for rows, cols in zip(input_num_indices, input_columns)]
            )

        validate(output_list, output_ref, input_num_indices, torch.equal, "output")

        if num_inputs == 0:
            grads = [torch.empty(0, dtype=dtype, device=device)]
        else:
            grads = [torch.rand_like(output) for output in output_ref]
        for out_ref, grad in zip(output_ref, grads):
            out_ref.backward(grad)

        if permute_output_dim_0_1 and num_inputs > 0:
            concat_grads = torch.concat(
                [grad.view(input_num_indices[0], -1) for grad in grads], dim=1
            ).flatten()
        else:
            concat_grads = torch.concat(grads)

        assert concat_grads.shape == output_test.shape
        output_test.backward(concat_grads)

        assert concat_inputs.grad is not None
        grad_list = concat_inputs.grad.split(
            [rows * cols for rows, cols in zip(input_rows, input_columns)]
        )

        grad_ref = []
        for input in inputs:
            assert input.grad is not None
            grad_ref.append(input.grad.flatten())

        tol = 1.0e-4 if dtype == torch.float else 1.0e-2

        validate(
            grad_list,
            grad_ref,
            input_rows,
            functools.partial(torch.allclose, atol=tol, rtol=tol),
            "grad",
        )


# e.g. "test_faketensor__test_cumsum": [unittest.expectedFailure]
# Please avoid putting tests here, you should put operator-specific
# skips and failures in deeplearning/fbgemm/fbgemm_gpu/test/failures_dict.json
# pyre-ignore[24]: Generic type `Callable` expects 2 type parameters.
additional_decorators: Dict[str, List[Callable]] = {
    "test_aot_dispatch_dynamic__test_index_select_dim0": [unittest.skip("hangs")],
    "test_aot_dispatch_static__test_index_select_dim0": [unittest.skip("hangs")],
    "test_faketensor__test_index_select_dim0": [unittest.skip("hangs")],
    "test_autograd_registration__test_index_select_dim0": [unittest.skip("hangs")],
    "test_schema__test_index_select_dim0": [unittest.skip("hangs")],
}

extend_test_class(IndexSelectTest, additional_decorators)

if __name__ == "__main__":
    unittest.main()
