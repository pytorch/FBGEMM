# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import random
import unittest

import hypothesis.strategies as st
import torch
from fbgemm_gpu.split_embedding_configs import SparseType
from hypothesis import given, HealthCheck, settings

from . import common  # noqa E402

# pyre-fixme[21]: Could not find name `open_source` in
#  `deeplearning.fbgemm.fbgemm_gpu.test.quantize.common`.
from .common import open_source

# pyre-fixme[16]: Module `common` has no attribute `open_source`.
if open_source:
    # pyre-ignore[21]
    from test_utils import gpu_unavailable
else:
    from fbgemm_gpu.test.test_utils import gpu_unavailable


class TestMixedDimInt8DequantizationConversion(unittest.TestCase):
    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    # Pyre was not able to infer the type of argument `not torch.cuda.is_available()`
    # to decorator factory `unittest.skipIf`.
    @unittest.skipIf(*gpu_unavailable)
    def test_mixed_dim_8bit_dequantize_op_empty(self) -> None:
        # assert that kernel return empty tensor and not failing with cuda error
        input_refs = torch.empty((0, 0), dtype=torch.uint8).cuda()
        D_offsets = torch.tensor([0]).cuda()
        mixed_dim_dequant_output = (
            torch.ops.fbgemm.Fused8BitRowwiseQuantizedToFloatMixedDim(
                input_refs, D_offsets, SparseType.FP32.as_int()
            )
        )
        assert mixed_dim_dequant_output.numel() == 0

    @unittest.skipIf(*gpu_unavailable)
    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    #  `hypothesis.strategies.integers($parameter$min_value = 0, $parameter$max_value =
    #  100)` to decorator factory `hypothesis.given`.
    @given(
        B=st.integers(min_value=1, max_value=100),
        T=st.integers(min_value=1, max_value=100),
        output_dtype=st.sampled_from([SparseType.FP32, SparseType.FP16]),
        min_dim=st.just(1),
        max_dim=st.just(100),
    )
    @settings(deadline=10000, suppress_health_check=[HealthCheck.filter_too_much])
    def test_mixed_dim_8bit_dequantize_op(
        self,
        B: int,
        T: int,
        output_dtype: SparseType,
        min_dim: int,
        max_dim: int,
    ) -> None:
        self.run_mixed_dim_8bit_dequantize_op_test(B, T, output_dtype, min_dim, max_dim)

    @unittest.skipIf(*gpu_unavailable)
    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    #  `hypothesis.strategies.integers($parameter$min_value = 0, $parameter$max_value =
    #  100)` to decorator factory `hypothesis.given`.
    @given(
        B=st.integers(min_value=1, max_value=100),
        T=st.integers(min_value=1, max_value=100),
        output_dtype=st.sampled_from([SparseType.FP32, SparseType.FP16]),
        min_dim=st.just(100),
        max_dim=st.just(1000),
    )
    @settings(deadline=10000, suppress_health_check=[HealthCheck.filter_too_much])
    def test_mixed_dim_8bit_dequantize_op_large_dims(
        self,
        B: int,
        T: int,
        output_dtype: SparseType,
        min_dim: int,
        max_dim: int,
    ) -> None:
        self.run_mixed_dim_8bit_dequantize_op_test(B, T, output_dtype, min_dim, max_dim)

    @unittest.skipIf(*gpu_unavailable)
    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    #  `hypothesis.strategies.integers($parameter$min_value = 0, $parameter$max_value =
    #  100)` to decorator factory `hypothesis.given`.
    @given(
        B=st.just(65540),
        T=st.just(5),
        output_dtype=st.just(SparseType.FP32),
        min_dim=st.just(1),
        max_dim=st.just(100),
    )
    @settings(deadline=10000, suppress_health_check=[HealthCheck.filter_too_much])
    def test_mixed_dim_8bit_dequantize_op_large_rows(
        self,
        B: int,
        T: int,
        output_dtype: SparseType,
        min_dim: int,
        max_dim: int,
    ) -> None:
        self.run_mixed_dim_8bit_dequantize_op_test(B, T, output_dtype, min_dim, max_dim)

    def run_mixed_dim_8bit_dequantize_op_test(
        self,
        B: int,
        T: int,
        output_dtype: SparseType,
        min_dim: int,
        max_dim: int,
    ) -> None:
        table_dims = [
            random.randint(min_dim, max_dim) * 8 for _ in range(T)
        ]  # assume table dimensions are multiples of 8
        table_dims_with_qparams = [d + 8 for d in table_dims]
        D_offsets = (
            torch.cumsum(torch.tensor([0] + table_dims_with_qparams), dim=0)
            .to(torch.int)
            .cuda()
        )
        input_refs = [torch.randn((B, d)).cuda() for d in table_dims]
        input_refs_int8 = [
            torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(t) for t in input_refs
        ]
        input_data = torch.concat(input_refs_int8, dim=1).contiguous()
        mixed_dim_dequant_output = (
            torch.ops.fbgemm.Fused8BitRowwiseQuantizedToFloatMixedDim(
                input_data, D_offsets, output_dtype.as_int()
            )
        )

        table_output_split = [t + 8 for t in table_dims]
        output_ref = []

        for output_i8 in torch.split(input_data, table_output_split, dim=1):
            output_ref.append(
                torch.ops.fbgemm.Fused8BitRowwiseQuantizedToFloat(
                    output_i8.contiguous()
                )
            )
        output_ref_concat = torch.cat(output_ref, dim=1)
        if output_dtype == SparseType.FP16:
            output_ref_concat = output_ref_concat.half()

        torch.testing.assert_close(output_ref_concat, mixed_dim_dequant_output)


if __name__ == "__main__":
    unittest.main()
