# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
import unittest
from typing import Callable, Dict, List

import hypothesis.strategies as st
import torch
from fbgemm_gpu.split_embedding_configs import SparseType
from hypothesis import given, settings, Verbosity


try:
    # pyre-ignore[21]
    from fbgemm_gpu import open_source  # noqa: F401

    # pyre-ignore[21]
    from test_utils import (  # noqa: F401
        gpu_unavailable,
        optests,
        symint_vector_unsupported,
    )
except Exception:
    if torch.version.hip:
        torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_hip")
    else:
        torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")

    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")
    from fbgemm_gpu.test.test_utils import (
        gpu_unavailable,
        optests,
        symint_vector_unsupported,
    )


# e.g. "test_faketensor__test_cumsum": [unittest.expectedFailure]
# Please avoid putting tests here, you should put operator-specific
# skips and failures in deeplearning/fbgemm/fbgemm_gpu/test/failures_dict.json
# pyre-ignore[24]: Generic type `Callable` expects 2 type parameters.
additional_decorators: Dict[str, List[Callable]] = {
    "test_pt2_compliant_tag_fbgemm_jagged_dense_elementwise_add": [
        # This operator has been grandfathered in. We need to fix this test failure.
        unittest.expectedFailure,
    ],
    "test_pt2_compliant_tag_fbgemm_jagged_dense_elementwise_add_jagged_output": [
        # This operator has been grandfathered in. We need to fix this test failure.
        unittest.expectedFailure,
    ],
}


@optests.generate_opcheck_tests(additional_decorators=additional_decorators)
class TestFP8RowwiseQuantizationConversion(unittest.TestCase):
    enable_logging: bool = False
    max_examples: int = 40

    def setUp(self) -> None:
        self.enable_logging = bool(os.getenv("FBGEMM_GPU_ENABLE_LOGGING", 0))
        if self.enable_logging:
            logging.info("Enabled logging for TestFP8RowwiseQuantizationConversion")

        torch._dynamo.config.cache_size_limit = self.max_examples
        logging.info(
            f"Setting torch._dynamo.config.cache_size_limit = {self.max_examples}"
        )

    @unittest.skipIf(*gpu_unavailable)
    # pyre-fixme[56]:
    @given(
        batched=st.booleans(),
        bs=st.integers(min_value=1, max_value=100),
        m=st.integers(min_value=0, max_value=100),
        n=st.integers(min_value=0, max_value=100),
        forward=st.booleans(),
        given_last_dim=st.booleans(),
        dtype=st.sampled_from(
            [
                torch.float,
                torch.half,
                torch.bfloat16,
            ],
        ),
        # if before PT 2.1, we don't support symint_vector, so turn it off
        test_compile=st.booleans() if symint_vector_unsupported() else st.just(False),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=max_examples, deadline=None)
    def test_quantize_and_dequantize_op_fp8_rowwise(
        self,
        batched: bool,
        bs: int,
        m: int,
        n: int,
        forward: bool,
        given_last_dim: bool,
        dtype: torch.dtype,
        test_compile: bool,
    ) -> None:
        n = n * 4  # need (n % 4 == 0)
        input_data = (
            torch.rand(bs, m, n, dtype=dtype)
            if batched
            else torch.rand(bs * m, n, dtype=dtype)
        )

        input_data_gpu = input_data.cuda()
        quantized_data_gpu = torch.ops.fbgemm.FloatToFP8RowwiseQuantized(
            input_data_gpu, forward=forward
        )
        quantize_func = (
            torch.compile(
                torch.ops.fbgemm.FP8RowwiseQuantizedToFloat,
                dynamic=True,
                fullgraph=True,
            )
            if test_compile and sys.version_info < (3, 12, 0)
            else torch.ops.fbgemm.FP8RowwiseQuantizedToFloat
        )

        if test_compile:
            torch._dynamo.mark_dynamic(quantized_data_gpu, 0)
            torch._dynamo.mark_dynamic(quantized_data_gpu, 1)

        output_dtype = {
            torch.float: SparseType.FP32,
            torch.half: SparseType.FP16,
            torch.bfloat16: SparseType.BF16,
        }[dtype].as_int()

        dequantized_data_gpu = quantize_func(
            quantized_data_gpu,
            forward=forward,
            output_dtype=output_dtype,
        )

        if m == 0 or n == 0:
            assert dequantized_data_gpu.numel() == 0
            return

        assert (
            dequantized_data_gpu.dtype == dtype
        ), "Result is {dequantized_data_gpu.dtype} type, but expected {dtype}"

        qref = input_data_gpu.float()
        dq = dequantized_data_gpu.float()

        assert not torch.isnan(dq).any(), "Results contain nan"

        if self.enable_logging:
            # Logging quantization errors
            errors = (qref - dq) / (qref + 1e-5)
            logging.info(f"max relative error {errors.abs().max()}")
            val, idx = torch.topk(errors.flatten().abs(), k=min(10, errors.shape[-1]))
            logging.info(f"top-10 errors {val}")
            logging.info(f"ref data {input_data_gpu.flatten()}")
            logging.info(f"dequantized data {dequantized_data_gpu.flatten()}")
            logging.info(f"max relative error {errors.flatten()[idx]}")

        torch.testing.assert_close(qref.cpu(), dq.cpu(), rtol=0.1, atol=0.05)

    @unittest.skipIf(*gpu_unavailable)
    # pyre-fixme[56]:
    @given(
        m=st.integers(min_value=1, max_value=1000),
        n1=st.integers(min_value=1, max_value=1000),
        n2=st.integers(min_value=1, max_value=1000),
        n3=st.integers(min_value=1, max_value=1000),
        row_dim=st.integers(min_value=1, max_value=2048),
        forward=st.booleans(),
        given_last_dim=st.booleans(),
        dtype=st.sampled_from(
            [
                torch.float,
                torch.half,
                torch.bfloat16,
            ],
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=max_examples, deadline=None)
    def test_quantize_and_dequantize_op_padded_fp8_rowwise(
        self,
        m: int,
        n1: int,
        n2: int,
        n3: int,
        row_dim: int,
        forward: bool,
        given_last_dim: bool,
        dtype: torch.dtype,
    ) -> None:
        row_dim = row_dim * 4
        device = "cuda"
        input1 = torch.rand(m, n1, device=device, dtype=dtype)
        input2 = torch.rand(m, n2, device=device, dtype=dtype)
        input3 = torch.rand(m, n3, device=device, dtype=dtype)
        output_dtype = {
            torch.float: SparseType.FP32,
            torch.half: SparseType.FP16,
            torch.bfloat16: SparseType.BF16,
        }[dtype].as_int()

        q1 = torch.ops.fbgemm.FloatToPaddedFP8RowwiseQuantized(
            input1, forward=forward, row_dim=row_dim
        )
        q2 = torch.ops.fbgemm.FloatToPaddedFP8RowwiseQuantized(
            input2, forward=forward, row_dim=row_dim
        )
        q3 = torch.ops.fbgemm.FloatToPaddedFP8RowwiseQuantized(
            input3, forward=forward, row_dim=row_dim
        )
        qcat = torch.cat([q1, q3, q2], dim=-1)
        if given_last_dim:
            d_qcat = torch.ops.fbgemm.PaddedFP8RowwiseQuantizedToFloat(
                qcat,
                forward=forward,
                row_dim=row_dim,
                output_last_dim=n1 + n2 + n3,
                output_dtype=output_dtype,
            )
        else:
            d_qcat = torch.ops.fbgemm.PaddedFP8RowwiseQuantizedToFloat(
                qcat,
                forward=forward,
                row_dim=row_dim,
                output_dtype=output_dtype,
            )

        assert (
            d_qcat.dtype == dtype
        ), "Result is {d_qcat.dtype} type, but expected {dtype}"
        qref = torch.cat([input1, input3, input2], dim=-1).cpu().float()
        dqcat = d_qcat.cpu().float()

        assert not torch.isnan(dqcat).any(), "Results contain nan"

        if self.enable_logging:
            # Logging quantization errors
            errors = (dqcat - qref) / (qref + 1e-5)
            assert not torch.isnan(errors).any()
            val, idx = torch.topk(errors.abs(), k=min(10, errors.shape[-1]))
            logging.info(f"top-10 errors {val}")
            logging.info(f"qref {torch.gather(qref, dim=1, index=idx)}")
            logging.info(f"dqcat {torch.gather(dqcat, dim=1, index=idx)}")
            logging.info(
                f"relative error: max: {errors.abs().max()*100:.1f}%, "
                f"median: {errors.abs().median()*100:.1f}%, "
                f"mean: {errors.abs().mean()*100:.1f}%"
            )

        torch.testing.assert_allclose(dqcat, qref, rtol=0.1, atol=0.05)


if __name__ == "__main__":
    unittest.main()
