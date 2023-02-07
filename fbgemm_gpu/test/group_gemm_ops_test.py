#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import unittest

import fbgemm_gpu
import fbgemm_gpu.group_linear_ops
import hypothesis.strategies as st
import torch
from hypothesis import given, settings

try:
    # pyre-ignore[21]
    from fbgemm_gpu import open_source  # noqa: F401

    # pyre-ignore[21]
    from test_utils import gpu_available

except Exception:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:group_gemm_ops")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:group_gemm_ops_cpu")
    from fbgemm_gpu.test.test_utils import gpu_available


def check_allclose(tensor_a: torch.Tensor, tensor_b: torch.Tensor, msg: str) -> bool:
    if not torch.allclose(tensor_a, tensor_b, rtol=1e-3, atol=1e-3):
        logging.info(f"FAILED: {msg}")
        return False
    return True


# cuda is not compatible with asan. To make sure that cuda tests
# run, use one of the non-asan modes, e.g., @mode/dev-nosan or
# @mode/opt. It goes without saying that cuda tests will run
# on GPU-enabled machines only (e.g., devgpu-s)
class GroupGemmOpsTest(unittest.TestCase):
    # pyre-ignore [56]: Invalid decoration, was not able to infer the type of argument
    @given(
        num_groups=st.integers(1, 5),
        max_m=st.integers(1, 5),
        max_n=st.integers(1, 5),
        max_k=st.integers(1, 5),
        transpose_b=st.booleans(),
        dtype=st.sampled_from(
            [
                torch.double,
                torch.float32,
                torch.float16,
            ]
        ),
        device=st.sampled_from(["cuda", "cpu"] if gpu_available else ["cpu"]),
        beta=st.integers(0, 1),
    )
    @settings(deadline=10000)
    def test_gmm(
        self,
        num_groups: int,
        max_m: int,
        max_n: int,
        max_k: int,
        transpose_b: bool,
        dtype: torch.dtype,
        device: str,
        # beta can be 0 or 1. If beta = 1, add another 2D tensor to the mm
        # product for each group. If beta = 0, just return the mm product for
        # each group.
        beta: int,
    ) -> None:
        beta = 1
        if device == "cpu" and dtype == torch.float16:
            logging.info("CPU op does not support half. Force dtype to float.")
            dtype = torch.float

        ms = torch.randint(1, max_m + 1, (num_groups,), dtype=torch.int)
        ns = torch.randint(1, max_n + 1, (num_groups,), dtype=torch.int) * 8
        ks = torch.randint(1, max_k + 1, (num_groups,), dtype=torch.int) * 8

        a_group = []
        b_group = []
        c_group = []
        for m, n, k in zip(ms, ns, ks):
            a_group.append(torch.rand((m, k), dtype=dtype, device=device))
            b = torch.rand((k, n), dtype=dtype, device=device)
            if transpose_b:
                # B becomes n x k tensor
                b = torch.as_strided(b, (k, n), (1, k))
            b_group.append(b)
            if beta == 1:
                # Test c_group as 2D-tensors (1D-tensors will be tested in test_group_linear with bias)
                c_group.append(torch.rand((m, n), dtype=dtype, device=device))

        output = torch.ops.fbgemm.gmm(a_group, b_group, c_group if beta == 1 else None)

        output_ref = []
        for i, (a, b) in enumerate(zip(a_group, b_group)):
            out_ref = torch.mm(a, b)
            if beta == 1:
                out_ref += c_group[i]
            output_ref.append(out_ref)

        passed = True
        for i, (test, ref) in enumerate(zip(output, output_ref)):
            passed = passed and check_allclose(
                test, ref, f"group {i}, m {ms[i]}, n {ns[i]}, k {ks[i]}"
            )
        assert passed

    # pyre-ignore [56]: Invalid decoration, was not able to infer the type of argument
    @given(
        num_groups=st.integers(1, 5),
        max_m=st.integers(1, 5),
        max_n=st.integers(1, 5),
        max_k=st.integers(1, 5),
        dtype=st.sampled_from(
            [
                torch.double,
                torch.float32,
                torch.float16,
            ]
        ),
        device=st.sampled_from(["cuda", "cpu"] if gpu_available else ["cpu"]),
        bias=st.booleans(),
    )
    @settings(deadline=10000)
    def test_group_linear(
        self,
        num_groups: int,
        max_m: int,
        max_n: int,
        max_k: int,
        dtype: torch.dtype,
        device: str,
        bias: bool,
    ) -> None:
        if device == "cpu" and dtype == torch.float16:
            logging.info("CPU op does not support half. Force dtype to float.")
            dtype = torch.float

        ms = torch.randint(1, max_m + 1, (num_groups,), dtype=torch.int)
        ns = torch.randint(1, max_n + 1, (num_groups,), dtype=torch.int) * 8
        ks = torch.randint(1, max_k + 1, (num_groups,), dtype=torch.int) * 8

        linears = []
        inputs = []
        for m, n, k in zip(ms, ns, ks):
            input = torch.rand((m, k), dtype=dtype, device=device)
            inputs.append(input)

            linear = torch.nn.Linear(k, n, bias=bias).to(device).to(dtype)
            linears.append(linear)

        output_ref = []
        for in_tensor, linear in zip(inputs, linears):
            output = linear(in_tensor)
            output_ref.append(output)

        group_linear = (
            fbgemm_gpu.group_linear_ops.GroupLinear(
                [(k, n) for k, n in zip(ks, ns)], bias=bias
            )
            .to(device)
            .to(dtype)
        )
        for i, linear in enumerate(linears):
            group_linear.gmm[i].weight.data.copy_(linear.weight.data)
            if bias:
                group_linear.gmm[i].bias.data.copy_(linear.bias.data)

        output = group_linear(inputs)

        passed = True
        for i in range(num_groups):
            size = f"m {ms[i]}, n {ns[i]}, k {ks[i]}"
            passed = passed and check_allclose(
                output[i], output_ref[i], f"group {i} activation, {size}"
            )
        assert passed


if __name__ == "__main__":
    unittest.main()
