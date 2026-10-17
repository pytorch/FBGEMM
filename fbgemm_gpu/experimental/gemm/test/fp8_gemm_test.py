# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import itertools
import unittest
from unittest import mock

import torch
import triton
from fbgemm_gpu.experimental.gemm.triton_gemm.fp8_gemm import (
    MATMUL_CONFIGS,
    MINIMUM_BLOCK_K,
    MINIMUM_BLOCK_M,
    MINIMUM_BLOCK_N,
    prune_configs,
    prune_configs_h100_static,
    USED_MATMUL_NON_PERSISTENT_CONFIGS,
)
from torch._dynamo.source import LocalSource
from torch._dynamo.testing import CompileCounterWithBackend
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv

if torch.cuda.is_available():
    from fbgemm_gpu.experimental.gemm.triton_gemm.fp8_gemm import (
        dequantize_fp8_block,
        dequantize_fp8_packed_row,
        dequantize_fp8_row,
        matmul_fp8_block,
        matmul_fp8_row,
        quantize_fp8_block,
        quantize_fp8_group,
        # packed_row unpacks the values, packed_row_raw returns just the packed tensor
        quantize_fp8_packed_row,
        quantize_fp8_packed_row_raw,
        quantize_fp8_row,
        quantize_fp8_row_meta,
        scale_fp8_row,
    )
    from fbgemm_gpu.experimental.gemm.triton_gemm.matmul_perf_model import (
        estimate_matmul_time,
    )


@unittest.skipIf(
    not torch.cuda.is_available()
    or torch.cuda.get_device_properties(torch.cuda.current_device()).major < 9,
    "Skip when H100 is not available",
)
class TestFp8Matmul(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    def test_quantize_fp8_row(self) -> None:
        def _test_quantize_fp8_row(
            shape: tuple[int, ...],
            use_triton: bool,
            device: torch.device,
            output_device: torch.device | None = None,
            use_jagged: bool = False,
            use_scale_ub: bool = False,
            transpose_inputs: bool = False,
            align_rows_to: int | None = None,
            expected_padded_size: int | None = None,  # only set with align_rows_to
        ) -> None:
            a = torch.randn(shape, dtype=torch.bfloat16, device=device)
            inputs = [a]
            # if transpose_inputs is true, get all possible dimension combinations
            # of the input tensor and transposes each pair
            if transpose_inputs:
                dims = range(a.ndim)
                for dim1, dim2 in itertools.combinations(dims, 2):
                    dims_list = list(dims)
                    dims_list[dim1], dims_list[dim2] = dims_list[dim2], dims_list[dim1]
                    inputs.append(a.clone().permute(dims_list))
            scale_ub = (
                torch.tensor([1200], dtype=torch.float, device=device)
                if use_scale_ub
                else None
            )
            for input_a in inputs:
                # Apply sparsification if specified.
                zero_start_index_M = None
                if use_jagged:
                    # View input as [G, M, K] where G is the number of groups.
                    grouped_input = input_a.view(
                        -1, input_a.shape[-2], input_a.shape[-1]
                    )
                    m_vals = torch.randint(
                        0, grouped_input.shape[1] + 1, (grouped_input.shape[0],)
                    )
                    mask = torch.arange(grouped_input.shape[-2]).expand(
                        (grouped_input.shape[0], grouped_input.shape[1])
                    ) >= m_vals.unsqueeze(-1)
                    # Set corresponding values to 0.
                    grouped_input[mask] = 0.0
                    # Generate nonzero tensor in same layout as input.
                    zero_start_index_M = torch.count_nonzero(
                        torch.sum(grouped_input, dim=-1), dim=-1
                    )

                a_fp8, a_scale = quantize_fp8_row(
                    input_a,
                    scale_ub=scale_ub,
                    zero_start_index_M=zero_start_index_M,
                    use_triton=use_triton,
                    output_device=output_device,
                    align_rows_to=align_rows_to,
                )

                a_fp8_meta, a_scale_meta = quantize_fp8_row_meta(
                    input_a,
                    scale_ub=scale_ub,
                    zero_start_index_M=zero_start_index_M,
                    use_triton=use_triton,
                    output_device=output_device,
                    align_rows_to=align_rows_to,
                )

                self.assertEqual(a_fp8.dtype, a_fp8_meta.dtype)
                self.assertEqual(a_fp8.shape, a_fp8_meta.shape)
                self.assertEqual(a_scale.dtype, a_scale_meta.dtype)
                self.assertEqual(a_scale.shape, a_scale_meta.shape)

                # Undo scaling.
                a_torch = a_fp8.to(torch.bfloat16)
                broadcast_shape = list(a_torch.shape[:-1]) + [-1]

                assert a_scale.shape == a_torch.shape[:-1]

                a_torch *= a_scale.view(broadcast_shape)

                if align_rows_to is not None:
                    # Pad input_a's row dimension to expected_padded_size if specified.
                    assert expected_padded_size is not None
                    pad_rows = expected_padded_size - input_a.shape[-1]
                    if pad_rows > 0:
                        pad_shape = list(input_a.shape)
                        pad_shape[-1] = pad_rows
                        pad_tensor = torch.zeros(
                            pad_shape,
                            dtype=input_a.dtype,
                            device=input_a.device,
                        )
                        input_a = torch.cat([input_a, pad_tensor], dim=-1)

                self.assertTrue(
                    torch.allclose(
                        input_a.to(device=output_device),
                        a_torch,
                        atol=2e-1,
                        rtol=1e-1,
                    )
                )

        for n_col in range(1, 9000, 100):
            _test_quantize_fp8_row((2, n_col), True, torch.device("cuda"))

            # Test with padding. These go up to 9000 (larger than max BLOCK_SIZE)

            # Calculate expected_padded_size from align_rows_to=8.
            # Using a different math here, just to make tests different from implementation.
            align_rows_to = 8
            trailing_beyond_alignment = n_col % align_rows_to
            padding_size = (
                align_rows_to - trailing_beyond_alignment
                if trailing_beyond_alignment > 0
                else 0
            )
            expected_padded_size = n_col + padding_size
            _test_quantize_fp8_row(
                (2, n_col),
                True,
                torch.device("cuda"),
                align_rows_to=align_rows_to,
                expected_padded_size=expected_padded_size,
            )

        # Test with batched input.
        _test_quantize_fp8_row((4, 2, 3), True, torch.device("cuda"))
        _test_quantize_fp8_row(  # simple padding case
            (4, 2, 3),
            True,
            torch.device("cuda"),
            align_rows_to=8,
            expected_padded_size=8,
        )
        _test_quantize_fp8_row(  # multiple padding case
            (4, 2, 13),
            True,
            torch.device("cuda"),
            align_rows_to=8,
            expected_padded_size=16,
        )
        _test_quantize_fp8_row(  # 0 padding case
            (4, 2, 8),
            True,
            torch.device("cuda"),
            align_rows_to=8,
            expected_padded_size=8,
        )
        _test_quantize_fp8_row((6, 4, 2, 3), True, torch.device("cuda"))
        # Test with non-contiguous input
        _test_quantize_fp8_row(
            (4, 2, 3), True, torch.device("cuda"), transpose_inputs=True
        )
        _test_quantize_fp8_row(
            (6, 4, 2, 3), True, torch.device("cuda"), transpose_inputs=True
        )
        _test_quantize_fp8_row((2, 3), True, torch.device("cuda"), use_scale_ub=True)
        # Test with cpu
        _test_quantize_fp8_row((2, 3), False, torch.device("cpu"), torch.device("cuda"))
        _test_quantize_fp8_row(
            (2, 3), False, torch.device("cpu"), torch.device("cuda"), use_scale_ub=True
        )
        _test_quantize_fp8_row((4, 2, 3), True, torch.device("cpu"))
        _test_quantize_fp8_row((6, 4, 2, 3), True, torch.device("cpu"))
        # Test with zero_start_index_M
        _test_quantize_fp8_row((20, 30), True, torch.device("cuda"), use_jagged=True)
        _test_quantize_fp8_row(
            (20, 30),
            True,
            torch.device("cuda"),
            use_jagged=True,
            align_rows_to=16,
            expected_padded_size=32,
        )
        _test_quantize_fp8_row(
            (6, 4, 2, 3), True, torch.device("cuda"), use_jagged=True
        )
        _test_quantize_fp8_row(
            (4, 2, 3),
            True,
            torch.device("cuda"),
            transpose_inputs=True,
            use_jagged=True,
        )

    def test_quantize_fp8_packed_row(self) -> None:
        def _test_quantize_fp8_packed_row(
            shape: tuple[int, ...],
            use_triton: bool,
            device: torch.device,
            output_device: torch.device | None = None,
            use_jagged: bool = False,
            use_scale_ub: bool = False,
            transpose_inputs: bool = False,
        ) -> None:
            a = torch.randn(shape, dtype=torch.bfloat16, device=device)
            inputs = [a]
            # if transpose_inputs is true, get all possible dimension combinations
            # of the input tensor and transposes each pair
            if transpose_inputs:
                dims = range(a.ndim)
                for dim1, dim2 in itertools.combinations(dims, 2):
                    dims_list = list(dims)
                    dims_list[dim1], dims_list[dim2] = dims_list[dim2], dims_list[dim1]
                    inputs.append(a.clone().permute(dims_list))
            scale_ub = (
                torch.tensor([1200], dtype=torch.float, device=device)
                if use_scale_ub
                else None
            )
            for input_a in inputs:
                # Apply sparsification if specified.
                zero_start_index_M = None
                if use_jagged:
                    # View input as [G, M, K] where G is the number of groups.
                    grouped_input = input_a.view(
                        -1, input_a.shape[-2], input_a.shape[-1]
                    )
                    m_vals = torch.randint(
                        0, grouped_input.shape[1] + 1, (grouped_input.shape[0],)
                    )
                    mask = torch.arange(grouped_input.shape[-2]).expand(
                        (grouped_input.shape[0], grouped_input.shape[1])
                    ) >= m_vals.unsqueeze(-1)
                    # Set corresponding values to 0.
                    grouped_input[mask] = 0.0
                    # Generate nonzero tensor in same layout as input.
                    zero_start_index_M = torch.count_nonzero(
                        torch.sum(grouped_input, dim=-1), dim=-1
                    )

                a_fp8, a_scale = quantize_fp8_packed_row(
                    input_a,
                    scale_ub=scale_ub,
                    zero_start_index_M=zero_start_index_M,
                    use_triton=use_triton,
                    output_device=output_device,
                )

                # Undo scaling.
                a_torch = a_fp8.to(torch.bfloat16)
                broadcast_shape = list(a_torch.shape[:-1]) + [-1]

                assert a_scale.shape == a_torch.shape[:-1]

                a_torch *= a_scale.view(broadcast_shape)

                self.assertTrue(
                    torch.allclose(
                        input_a.to(device=output_device),
                        a_torch,
                        atol=2e-1,
                        rtol=1e-1,
                    )
                )

        for n_col in range(1, 9000, 100):
            _test_quantize_fp8_packed_row((2, n_col), True, torch.device("cuda"))
        # Test with batched input.
        _test_quantize_fp8_packed_row((4, 2, 3), True, torch.device("cuda"))
        _test_quantize_fp8_packed_row((6, 4, 2, 3), True, torch.device("cuda"))
        # Test with non-contiguous input
        _test_quantize_fp8_packed_row(
            (4, 2, 3), True, torch.device("cuda"), transpose_inputs=True
        )
        _test_quantize_fp8_packed_row(
            (6, 4, 2, 3), True, torch.device("cuda"), transpose_inputs=True
        )
        _test_quantize_fp8_packed_row(
            (2, 3), True, torch.device("cuda"), use_scale_ub=True
        )
        # Test with cpu
        _test_quantize_fp8_packed_row(
            (2, 3), False, torch.device("cpu"), torch.device("cuda")
        )
        _test_quantize_fp8_packed_row(
            (2, 3), False, torch.device("cpu"), torch.device("cuda"), use_scale_ub=True
        )
        _test_quantize_fp8_packed_row((4, 2, 3), True, torch.device("cpu"))
        _test_quantize_fp8_packed_row((6, 4, 2, 3), True, torch.device("cpu"))
        # Test with zero_start_index_M
        _test_quantize_fp8_packed_row(
            (20, 30), True, torch.device("cuda"), use_jagged=True
        )
        _test_quantize_fp8_packed_row(
            (6, 4, 2, 3), True, torch.device("cuda"), use_jagged=True
        )
        _test_quantize_fp8_packed_row(
            (4, 2, 3),
            True,
            torch.device("cuda"),
            transpose_inputs=True,
            use_jagged=True,
        )

    def test_dequantize_fp8_row(self) -> None:
        def _test_dequantize_fp8_row(
            shape: tuple[int, ...],
        ) -> None:
            a = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
            a_fp8, a_scale = quantize_fp8_row(
                a,
                use_triton=True,
            )

            # Undo scaling.
            a_bf16 = dequantize_fp8_row(a_fp8, a_scale)

            ms = triton.testing.do_bench(
                lambda: dequantize_fp8_row(a_fp8, a_scale),
            )
            print(f"Shape: {a.shape} MS: {ms}")
            torch.testing.assert_close(a_bf16, a, atol=2e-1, rtol=1e-1)
            self.assertTrue(
                torch.allclose(
                    a,
                    a_bf16,
                    atol=2e-1,
                    rtol=1e-1,
                )
            )

        for n_col in [1, 100, 1000]:
            _test_dequantize_fp8_row((2, n_col))
        # Test with batched input.
        _test_dequantize_fp8_row((4, 2, 3))
        shapes = [(4, 2, 3), (6, 4, 2, 3), (2, 3), (20, 30)]
        for shape in shapes:
            _test_dequantize_fp8_row(shape)

    def test_dequantize_fp8_packed_row(self) -> None:
        def _test_dequantize_fp8_packed_row(
            shape: tuple[int, ...],
        ) -> None:
            a = torch.randn(shape, dtype=torch.bfloat16, device="cuda")

            packed_values = quantize_fp8_packed_row_raw(
                a,
                use_triton=True,
            )

            # Undo scaling.
            a_bf16 = dequantize_fp8_packed_row(packed_values)

            ms = triton.testing.do_bench(
                lambda: dequantize_fp8_packed_row(packed_values),
            )
            print(f"Shape: {a.shape} MS: {ms}")

            torch.testing.assert_close(a_bf16, a, atol=2e-1, rtol=1e-1)

            self.assertTrue(
                torch.allclose(
                    a,
                    a_bf16,
                    atol=2e-1,
                    rtol=1e-1,
                )
            )

        for n_col in [1, 100, 1000]:
            _test_dequantize_fp8_packed_row((2, n_col))
        # Test with batched input.
        _test_dequantize_fp8_packed_row((4, 2, 3))
        shapes = [(4, 2, 3), (6, 4, 2, 3), (2, 3), (20, 30)]
        for shape in shapes:
            _test_dequantize_fp8_packed_row(shape)

    def test_scale_fp8_row(self) -> None:
        def _test_scale_fp8_row(
            shape: tuple[int, int],
            device: torch.device,
        ) -> None:
            M, K = shape
            a = torch.randn(M, K, dtype=torch.bfloat16, device=device)

            x_scale = torch.randn(M, dtype=torch.bfloat16, device=device)
            w_scale = torch.randn(K, dtype=torch.bfloat16, device=device)

            scaled_out = scale_fp8_row(a, x_scale, w_scale)

            # Compare with reference value.
            scaled_out_torch = a * x_scale[:, None] * w_scale[None, :]

            self.assertTrue(
                torch.allclose(
                    scaled_out,
                    scaled_out_torch,
                    atol=2e-1,
                    rtol=1e-1,
                )
            )

        _test_scale_fp8_row((2, 3), torch.device("cuda"))
        _test_scale_fp8_row((2, 3), torch.device("cpu"))
        # Shapes whose last block of columns is partial, including ones that
        # take more than one iteration of the kernel's column loop.
        for shape in [(4, 513), (3, 1000), (2, 1025)]:
            # lint-fixme: TorchDeviceCuda, TorchFunctionCallCudaDevice
            # CUDA specifically required: only the CUDA path reaches the Triton
            # kernel whose partial trailing column block these shapes exercise.
            _test_scale_fp8_row(shape, torch.device("cuda"))

    def test_matmul_fp8_row(self) -> None:
        def _test_matmul_fp8_row(
            shape: tuple[int, int, int],
            device: torch.device,
            fp8_fast_accum: bool,
            use_bias: bool = False,
            transpose_input: bool = False,
            compile: bool = False,
        ) -> None:
            M, N, K = shape
            a = torch.randn(M, K, dtype=torch.bfloat16, device=device)
            # Make a non-contiguous tensor and check that we still get proper results.
            if transpose_input:
                a = a.t()
            b = torch.randn(N, K, dtype=torch.bfloat16, device=device)
            bias = (
                torch.randn(N, dtype=torch.float32, device=device) if use_bias else None
            )

            # Test that we can compile the full fp8 matmul operation.
            if compile:

                @torch.compile(fullgraph=True)
                def _quantize_matmul_fp8(
                    a: torch.Tensor,
                    b: torch.Tensor,
                    bias: torch.Tensor | None,
                    fp8_fast_accum: bool,
                ) -> torch.Tensor:
                    a_fp8, a_scale = quantize_fp8_row(a)
                    b_fp8, b_scale = quantize_fp8_row(b)
                    return matmul_fp8_row(
                        a_fp8,
                        b_fp8,
                        a_scale,
                        b_scale,
                        bias=bias,
                        fp8_fast_accum=fp8_fast_accum,
                    )

                result = _quantize_matmul_fp8(a, b, bias, fp8_fast_accum)
            # Otherwise run normally.
            else:
                # Quantize inputs.
                a_fp8, a_scale = quantize_fp8_row(a)
                b_fp8, b_scale = quantize_fp8_row(b)

                result = matmul_fp8_row(
                    a_fp8,
                    b_fp8,
                    a_scale,
                    b_scale,
                    bias=bias,
                    fp8_fast_accum=fp8_fast_accum,
                )
            self.assertTrue(result.shape == (M, N))

            expected_result = a @ b.T
            if use_bias:
                # pyre-fixme[6]: For 1st argument expected `Union[bool, complex,
                #  float, int, Tensor]` but got `Tensor | None`.
                expected_result += bias
            self.assertTrue(
                torch.allclose(result, expected_result, atol=2e-1, rtol=5e-2)
            )

        _test_matmul_fp8_row((3, 4, 5), torch.device("cuda"), True)
        _test_matmul_fp8_row((3, 4, 5), torch.device("cuda"), True, compile=True)
        _test_matmul_fp8_row(
            (5, 4, 5), torch.device("cuda"), True, transpose_input=True
        )
        _test_matmul_fp8_row((3, 4, 5), torch.device("cuda"), True, True)
        _test_matmul_fp8_row((3, 4, 5), torch.device("cuda"), False)
        _test_matmul_fp8_row((3, 4, 5), torch.device("cuda"), False, True)
        _test_matmul_fp8_row((3, 4, 5), torch.device("cpu"), False)
        _test_matmul_fp8_row((3, 4, 5), torch.device("cpu"), False, True)

    @unittest.skipIf(
        torch.version.hip is None,
        "Non-persistent kernel configs use AMD-only options.",
    )
    def test_matmul_fp8_row_dynamic_rows(self) -> None:
        counter = CompileCounterWithBackend("inductor")

        @torch.compile(backend=counter, fullgraph=True)
        def matmul(
            a: torch.Tensor,
            b: torch.Tensor,
            a_scale: torch.Tensor,
            b_scale: torch.Tensor,
        ) -> torch.Tensor:
            return matmul_fp8_row(a, b, a_scale, b_scale, no_use_persistent=True)

        device = torch.accelerator.current_accelerator()
        a, a_scale = quantize_fp8_row(
            torch.randn(64, 3072, device=device, dtype=torch.bfloat16)
        )
        b, b_scale = quantize_fp8_row(
            torch.randn(1280, 3072, device=device, dtype=torch.bfloat16)
        )
        torch._dynamo.mark_dynamic(a, 0, min=1, max=64)
        torch._dynamo.mark_dynamic(a_scale, 0, min=1, max=64)
        for rows in (64, 63, 33, 32, 17, 10, 6, 2, 1):
            with self.subTest(rows=rows):
                input_a = a if rows == 64 else a[:rows].clone()
                input_scale = a_scale if rows == 64 else a_scale[:rows].clone()
                result = matmul(input_a, b, input_scale, b_scale)
                self.assertEqual(result.shape, (rows, 1280))
                expected = input_a.float() @ b.float().T
                expected *= input_scale[:, None] * b_scale[None, :]
                torch.testing.assert_close(
                    result.float(), expected, atol=0.5, rtol=0.02
                )
                if rows != 1:
                    self.assertEqual(counter.frame_count, 1)

    def test_matmul_fp8_row_skip_scaling(self) -> None:
        def _fp8_clamp(x: torch.Tensor) -> torch.Tensor:
            fp8_dtype = torch.float8_e4m3fn
            fp8_max = torch.finfo(fp8_dtype).max
            xq = torch.clamp(x, min=-1 * fp8_max, max=fp8_max).to(fp8_dtype)
            return xq

        def _test_matmul_fp8_row_skip_scaling(
            shape: tuple[int, int, int],
            device: torch.device,
            use_bias: bool = True,
            transpose_input: bool = False,
            compile: bool = False,
        ) -> None:
            M, N, K = shape
            a = torch.randn(M, K, dtype=torch.bfloat16, device=device)
            # Make a non-contiguous tensor and check that we still get proper results.
            if transpose_input:
                a = a.t()
            b = torch.randn(N, K, dtype=torch.bfloat16, device=device)
            bias = (
                torch.randn(N, dtype=torch.float32, device=device) if use_bias else None
            )

            # Test that we can compile the full fp8 matmul operation.
            if compile:

                @torch.compile(fullgraph=True)
                def _quantize_matmul_fp8(
                    a: torch.Tensor,
                    b: torch.Tensor,
                    bias: torch.Tensor | None,
                ) -> torch.Tensor:
                    a_fp8, a_scale = _fp8_clamp(a), None
                    b_fp8, b_scale = quantize_fp8_row(b)
                    return matmul_fp8_row(
                        a_fp8,
                        b_fp8,
                        a_scale,
                        b_scale,
                        bias=bias,
                        fp8_fast_accum=True,
                        imprecise_acc=False,
                        tma_persistent=False,
                        no_use_persistent=False,
                        use_warp_specialization=False,
                    )

                result = _quantize_matmul_fp8(a, b, bias)
            # Otherwise run normally.
            else:
                # Quantize inputs.
                a_fp8, a_scale = _fp8_clamp(a), None
                b_fp8, b_scale = quantize_fp8_row(b)

                result = matmul_fp8_row(
                    a_fp8,
                    b_fp8,
                    a_scale,
                    b_scale,
                    bias=bias,
                    fp8_fast_accum=True,
                    imprecise_acc=False,
                    tma_persistent=False,
                    no_use_persistent=False,
                    use_warp_specialization=False,
                )
            self.assertTrue(result.shape == (M, N))

            expected_result = a @ b.T
            if use_bias:
                # pyre-fixme[6]: For 1st argument expected `Union[bool, complex,
                #  float, int, Tensor]` but got `Tensor | None`.
                expected_result += bias
            self.assertTrue(
                torch.allclose(result, expected_result, atol=2e-1, rtol=5e-2)
            )

        _test_matmul_fp8_row_skip_scaling((3, 4, 5), torch.device("cuda"))
        _test_matmul_fp8_row_skip_scaling((3, 4, 5), torch.device("cuda"), compile=True)
        _test_matmul_fp8_row_skip_scaling(
            (5, 4, 5), torch.device("cuda"), transpose_input=True
        )
        _test_matmul_fp8_row_skip_scaling(
            (3, 4, 5), torch.device("cuda"), use_bias=False
        )

    def test_matmul_fp8_row_cpu_preserves_batch_dimensions(self) -> None:
        a = torch.randn(2, 3, 5, dtype=torch.bfloat16)
        b = torch.randn(4, 5, dtype=torch.bfloat16)
        a_fp8, a_scale = quantize_fp8_row(a)
        b_fp8, b_scale = quantize_fp8_row(b)

        result = matmul_fp8_row(a_fp8, b_fp8, a_scale, b_scale)

        self.assertEqual(result.shape, torch.Size((2, 3, 4)))
        self.assertTrue(torch.allclose(result, a @ b.T, atol=2e-1, rtol=5e-2))

    def test_fp8_matmul_fake_preserves_batch_dimensions(self) -> None:
        with FakeTensorMode():
            device = torch.accelerator.current_accelerator()
            a = torch.empty(2, 3, 5, device=device, dtype=torch.float8_e4m3fn)
            b = torch.empty(4, 5, device=device, dtype=torch.float8_e4m3fn)
            row_scale_a = torch.empty(2, 3, device=device)
            row_scale_b = torch.empty(4, device=device)
            block_scale_a = torch.empty(1, 1, device=device)
            block_scale_b = torch.empty(1, 1, device=device)

            row_result = matmul_fp8_row(
                a,
                b,
                row_scale_a,
                row_scale_b,
                use_persistent=False,
                # Non-default last arg forces every schema arg into the fake.
                use_warp_specialization=True,
            )
            block_result = matmul_fp8_block(
                a,
                b,
                block_scale_a,
                block_scale_b,
            )

        self.assertEqual(row_result.shape, torch.Size((2, 3, 4)))
        self.assertEqual(block_result.shape, torch.Size((2, 3, 4)))

    def test_fp8_matmul_empty_uses_requested_output_dtype(self) -> None:
        device = torch.accelerator.current_accelerator()
        a = torch.empty(0, 5, device=device, dtype=torch.float8_e4m3fn)
        b = torch.empty(4, 5, device=device, dtype=torch.float8_e4m3fn)
        row_scale_a = torch.empty(0, device=device)
        row_scale_b = torch.empty(4, device=device)
        block_scale_a = torch.empty(0, 1, device=device)
        block_scale_b = torch.empty(1, 1, device=device)

        row_result = matmul_fp8_row(
            a, b, row_scale_a, row_scale_b, dot_out_dtype=torch.float16
        )
        block_result = matmul_fp8_block(
            a,
            b,
            block_scale_a,
            block_scale_b,
            dot_out_dtype=torch.float16,
        )

        with FakeTensorMode():
            fake_a = torch.empty(0, 5, device=device, dtype=torch.float8_e4m3fn)
            fake_b = torch.empty(4, 5, device=device, dtype=torch.float8_e4m3fn)
            fake_row_result = matmul_fp8_row(
                fake_a,
                fake_b,
                torch.empty(0, device=device),
                torch.empty(4, device=device),
                dot_out_dtype=torch.float16,
            )
            fake_block_result = matmul_fp8_block(
                fake_a,
                fake_b,
                torch.empty(0, 1, device=device),
                torch.empty(1, 1, device=device),
                dot_out_dtype=torch.float16,
            )

        self.assertEqual(row_result.dtype, torch.float16)
        self.assertEqual(block_result.dtype, torch.float16)
        self.assertEqual(fake_row_result.dtype, torch.float16)
        self.assertEqual(fake_block_result.dtype, torch.float16)

    def test_quantize_fp8_group(self) -> None:
        def _test_quantize_fp8_group(
            shape: tuple[int, int],
            group_size: int,
            use_scale_ub: bool = False,
        ) -> None:
            M, K = shape
            a = torch.randn(M, K, dtype=torch.float, device="cuda")

            scale_ub = (
                torch.tensor([1200], dtype=torch.float, device="cuda")
                if use_scale_ub
                else None
            )

            a_fp8, a_scale = quantize_fp8_group(a, group_size, scale_ub=scale_ub)

            a_torch = a_fp8.to(torch.float)

            # Undo scaling.
            a_torch = a_torch.view(-1, K // group_size, group_size) * a_scale.unsqueeze(
                -1
            )
            a_torch = a_torch.view(M, K)

            self.assertTrue(torch.allclose(a, a_torch, atol=2e-1, rtol=5e-2))

        _test_quantize_fp8_group((128, 128), 128)
        _test_quantize_fp8_group((1, 256), 64)
        _test_quantize_fp8_group((2, 384), 128, use_scale_ub=True)

    def test_quantize_fp8_block(self) -> None:
        def _test_quantize_fp8_block(
            shape: tuple[int, int],
            block_shape: tuple[int, int],
            use_scale_ub: bool = False,
        ) -> None:
            M, K = shape
            BLOCK_M, BLOCK_K = block_shape
            a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")

            scale_ub = (
                torch.tensor([1200], dtype=torch.float, device="cuda")
                if use_scale_ub
                else None
            )

            a_fp8, a_scale = quantize_fp8_block(a, BLOCK_M, BLOCK_K, scale_ub=scale_ub)

            a_torch = a_fp8.to(torch.bfloat16)

            # Undo scaling.
            for i in range(0, M, BLOCK_M):
                for j in range(0, K, BLOCK_K):
                    block = a_torch[i : i + BLOCK_M, j : j + BLOCK_K]
                    scaling = a_scale[i // BLOCK_M, j // BLOCK_K]
                    scaled_block = block * scaling
                    a_torch[i : i + BLOCK_M, j : j + BLOCK_K] = scaled_block

            self.assertTrue(torch.allclose(a, a_torch, atol=2e-1, rtol=5e-2))

        _test_quantize_fp8_block((2, 4), (1, 2))
        _test_quantize_fp8_block((3, 6), (2, 8))
        _test_quantize_fp8_block((3, 6), (2, 8), use_scale_ub=True)

    def test_dequantize_fp8_block(self) -> None:
        def _test_dequantize_fp8_block(
            shape: tuple[int, int],
            block_shape: tuple[int, int],
            use_scale_ub: bool = False,
        ) -> None:
            M, K = shape
            BLOCK_M, BLOCK_K = block_shape
            a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")

            scale_ub = (
                torch.tensor([1200], dtype=torch.float, device="cuda")
                if use_scale_ub
                else None
            )

            a_fp8, a_scale = quantize_fp8_block(
                a, block_m=BLOCK_M, block_k=BLOCK_K, scale_ub=scale_ub
            )
            a_dequant = dequantize_fp8_block(
                a_fp8, a_scale, block_m=BLOCK_M, block_k=BLOCK_K
            )
            self.assertTrue(torch.allclose(a, a_dequant, atol=2e-1, rtol=5e-2))

        _test_dequantize_fp8_block((3, 1024), (1, 256))
        _test_dequantize_fp8_block((11, 128), (1, 128))
        _test_dequantize_fp8_block((11, 256), (1, 256), use_scale_ub=True)

    def test_matmul_fp8_block(self) -> None:
        def _test_matmul_fp8_block(
            shape: tuple[int, int, int],
            block_shape: tuple[int, int, int],
            fp8_fast_accum: bool,
            transpose_input: bool = False,
            device: str = "cuda",
        ) -> None:
            M, N, K = shape
            BLOCK_M, BLOCK_N, BLOCK_K = block_shape
            a = torch.randn(M, K, dtype=torch.bfloat16, device=device)
            # Make a non-contiguous tensor and check that we still get proper results.
            if transpose_input:
                a = a.t()
            b = torch.randn(N, K, dtype=torch.bfloat16, device=device)

            # Quantize inputs.
            a_fp8, a_scale = quantize_fp8_block(
                a, BLOCK_M, BLOCK_K, output_device=torch.device("cuda")
            )
            b_fp8, b_scale = quantize_fp8_block(
                b, BLOCK_N, BLOCK_K, output_device=torch.device("cuda")
            )

            result = matmul_fp8_block(
                a_fp8,
                b_fp8,
                a_scale,
                b_scale,
                BLOCK_M,
                BLOCK_N,
                BLOCK_K,
                fp8_fast_accum=fp8_fast_accum,
            )
            self.assertTrue(result.shape == (M, N))

            expected_result = (a @ b.T).to("cuda")

            self.assertTrue(
                torch.allclose(result, expected_result, atol=1e2, rtol=5e-2)
            )

        _test_matmul_fp8_block((3, 4, 5), (256, 256, 256), True)
        _test_matmul_fp8_block((5, 4, 5), (256, 256, 256), True, transpose_input=True)
        _test_matmul_fp8_block((1024, 2048, 4096), (256, 512, 1024), True)
        _test_matmul_fp8_block((1024, 2048, 4096), (256, 512, 1024), False)
        _test_matmul_fp8_block((3, 4, 5), (256, 256, 256), False)
        _test_matmul_fp8_block((3, 4, 5), (256, 256, 256), True, device="cpu")
        _test_matmul_fp8_block((1024, 2048, 4096), (256, 512, 1024), True, device="cpu")
        # K values that leave a partial trailing scale block while still making
        # EVEN_K true, so the last iteration is reached with
        # k_remaining == BLOCK_K * SPLIT_K. The K=5 shapes above make EVEN_K
        # false, which is why they already took the tail branch and never
        # covered this case. scale_block_m/n are kept at M/N so that no tile can
        # span two M/N scale blocks whichever BLOCK_M/BLOCK_N autotuning picks.
        _test_matmul_fp8_block((256, 256, 640), (256, 256, 256), True)
        _test_matmul_fp8_block((256, 256, 64), (256, 256, 256), True)
        _test_matmul_fp8_block((256, 256, 192), (256, 256, 128), True)


class TestPruneConfigsNonPersistent(unittest.TestCase):
    def test_pruning_preserves_symbolic_dimensions(self) -> None:
        for dimension, hint in (("M", 64), ("N", 1280), ("K", 3072)):
            with self.subTest(dimension=dimension):
                shape_env = ShapeEnv()
                source = LocalSource(dimension)
                symbol = shape_env.create_symbol(
                    hint,
                    source,
                    dynamic_dim=DimDynamic.DYNAMIC,
                    do_not_specialize_zero_one=True,
                )
                shape_env.constrain_symbol_range(symbol, 1, hint)
                symbolic_size = shape_env.create_symintnode(
                    symbol, hint=hint, source=source
                )
                args: dict[str, torch.Tensor | int | torch.SymInt] = {
                    "M": 64,
                    "N": 1280,
                    "K": 3072,
                    "A": torch.empty(
                        64, 3072, dtype=torch.float8_e4m3fn, device="meta"
                    ),
                    "B": torch.empty(
                        1280, 3072, dtype=torch.float8_e4m3fn, device="meta"
                    ),
                }
                expected = prune_configs(
                    USED_MATMUL_NON_PERSISTENT_CONFIGS, args, USE_BIAS=False
                )
                args[dimension] = symbolic_size
                actual = prune_configs(
                    USED_MATMUL_NON_PERSISTENT_CONFIGS, args, USE_BIAS=False
                )
                self.assertEqual(actual, expected)
                self.assertIs(args[dimension], symbolic_size)
                bounds = shape_env.bound_sympy(symbol)
                self.assertEqual(bounds.lower, 1)
                self.assertEqual(bounds.upper, hint)
                self.assertEqual(shape_env.guards, [])

    def test_pruning_keeps_small_shape_and_bias_constraints(self) -> None:
        configs = [
            triton.Config(
                {
                    "BLOCK_M": block_m,
                    "BLOCK_N": 16,
                    "BLOCK_K": 64,
                    "GROUP_M": 1,
                    "SPLIT_K": split_k,
                    "matrix_instr_nonkdim": 16,
                }
            )
            for block_m, split_k in ((16, 1), (32, 1), (16, 2))
        ]
        args = {
            "M": 1,
            "N": 1,
            "K": 3072,
            "A": torch.empty(1, 3072, dtype=torch.float8_e4m3fn, device="meta"),
            "B": torch.empty(1, 3072, dtype=torch.float8_e4m3fn, device="meta"),
        }
        self.assertEqual(
            prune_configs(configs, args, USE_BIAS=False), [configs[0], configs[2]]
        )
        self.assertEqual(prune_configs(configs, args, USE_BIAS=True), [configs[0]])


class _FakeConfig:
    """Minimal stand-in for triton.Config (only the fields the prune reads)."""

    def __init__(self, bm: int, bn: int, bk: int, ns: int, nw: int) -> None:
        self.kwargs = {
            "BLOCK_M": bm,
            "BLOCK_N": bn,
            "BLOCK_K": bk,
            "SPLIT_K": 1,
        }
        self.num_stages = ns
        self.num_warps = nw


class TestPruneConfigsH100Static(unittest.TestCase):
    """CPU-runnable coverage for the H100 static autotune prune.

    Uses lightweight fake configs so no GPU or Triton launch is needed;
    the Hopper gate is faked via mock.
    """

    def _make_configs(self) -> list[_FakeConfig]:
        configs = []
        for bm, bn, bk, ns, nw in [
            (128, 256, 32, 3, 8),
            (256, 128, 32, 3, 8),
            (64, 128, 256, 4, 4),
            (256, 256, 32, 4, 4),
            (64, 32, 32, 5, 2),
            (64, 256, 32, 4, 4),
        ]:
            configs.append(_FakeConfig(bm, bn, bk, ns, nw))
        for ns in [2, 3, 4, 5, 6]:
            configs.append(_FakeConfig(16, 64, 32, ns, 2))
        return configs

    def _prune(
        self,
        m: int,
        n: int,
        k: int,
        hopper: bool = True,
        configs: list[_FakeConfig] | None = None,
    ) -> list[_FakeConfig]:
        with mock.patch(
            "fbgemm_gpu.experimental.gemm.triton_gemm.fp8_gemm.is_hopper",
            return_value=hopper,
        ):
            return prune_configs_h100_static(
                configs if configs is not None else self._make_configs(),
                {"M": m, "N": n, "K": k},
            )

    @staticmethod
    def _keys(kept: list[_FakeConfig]) -> set[tuple[int, int, int]]:
        return {
            (c.kwargs["BLOCK_M"], c.kwargs["BLOCK_N"], c.kwargs["BLOCK_K"])
            for c in kept
        }

    def test_pool_minima_match_config_space(self) -> None:
        self.assertEqual(
            MINIMUM_BLOCK_M, min(c.kwargs["BLOCK_M"] for c in MATMUL_CONFIGS)
        )
        self.assertEqual(
            MINIMUM_BLOCK_N, min(c.kwargs["BLOCK_N"] for c in MATMUL_CONFIGS)
        )
        self.assertEqual(
            MINIMUM_BLOCK_K, min(c.kwargs["BLOCK_K"] for c in MATMUL_CONFIGS)
        )

    def test_small_m_drops_large_block_m(self) -> None:
        kept = self._prune(1, 1024, 1024)
        bms = {c.kwargs["BLOCK_M"] for c in kept}
        self.assertTrue(bms <= {16, 32, 64})

    def test_small_n_drops_large_block_n(self) -> None:
        kept = self._prune(512, 32, 256)
        keys = self._keys(kept)
        self.assertIn((64, 32, 32), keys)
        self.assertNotIn((128, 256, 32), keys)
        self.assertNotIn((64, 256, 32), keys)

    def test_small_k_drops_large_block_k(self) -> None:
        kept = self._prune(512, 1024, 32)
        keys = self._keys(kept)
        self.assertNotIn((64, 128, 256), keys)
        self.assertIn((64, 256, 32), keys)

    def test_large_k_keeps_large_block_k(self) -> None:
        kept = self._prune(512, 1024, 19712)
        self.assertIn((64, 128, 256), self._keys(kept))

    def test_bk_floor_keeps_32_below_threshold(self) -> None:
        configs = [
            _FakeConfig(64, 64, 32, 4, 4),
            _FakeConfig(64, 64, 64, 4, 4),
        ]
        kept = self._prune(1024, 1024, 256, configs=configs)
        self.assertEqual(self._keys(kept), {(64, 64, 32), (64, 64, 64)})

    def test_bk_floor_drops_32_at_large_k(self) -> None:
        configs = [
            _FakeConfig(64, 64, 32, 4, 4),
            _FakeConfig(64, 64, 64, 4, 4),
        ]
        kept = self._prune(1024, 1024, 1024, configs=configs)
        self.assertEqual(self._keys(kept), {(64, 64, 64)})

    def test_bk_floor_drops_64_at_huge_k(self) -> None:
        configs = [
            _FakeConfig(64, 64, 64, 4, 4),
            _FakeConfig(64, 64, 128, 4, 4),
        ]
        kept = self._prune(1024, 1024, 6656, configs=configs)
        self.assertEqual(self._keys(kept), {(64, 64, 128)})

    def test_bm_floor_drops_small_bm_at_big_m(self) -> None:
        configs = [
            _FakeConfig(16, 64, 64, 4, 4),
            _FakeConfig(32, 64, 64, 4, 4),
            _FakeConfig(64, 64, 64, 4, 4),
        ]
        kept = self._prune(512, 1024, 1024, configs=configs)
        self.assertEqual(self._keys(kept), {(64, 64, 64)})

    def test_bm_floor_keeps_small_bm_at_small_k(self) -> None:
        configs = [
            _FakeConfig(16, 64, 32, 4, 4),
            _FakeConfig(32, 64, 32, 4, 4),
            _FakeConfig(64, 64, 32, 4, 4),
        ]
        kept = self._prune(512, 1024, 16, configs=configs)
        self.assertEqual(self._keys(kept), {(16, 64, 32), (32, 64, 32), (64, 64, 32)})

    def test_bn_floor_drops_32_at_big_n(self) -> None:
        configs = [
            _FakeConfig(64, 32, 64, 4, 4),
            _FakeConfig(64, 64, 64, 4, 4),
        ]
        kept = self._prune(512, 1024, 1024, configs=configs)
        self.assertEqual(self._keys(kept), {(64, 64, 64)})

    def test_bn_floor_drops_64_at_huge_n(self) -> None:
        configs = [
            _FakeConfig(64, 64, 128, 4, 4),
            _FakeConfig(64, 128, 128, 4, 4),
        ]
        kept = self._prune(512, 8192, 2048, configs=configs)
        self.assertEqual(self._keys(kept), {(64, 128, 128)})

    def test_bmn_floor_keeps_small_tiles_at_small_problem(self) -> None:
        configs = [
            _FakeConfig(16, 32, 32, 4, 2),
            _FakeConfig(32, 32, 32, 4, 2),
        ]
        kept = self._prune(64, 256, 256, configs=configs)
        self.assertEqual(self._keys(kept), {(16, 32, 32), (32, 32, 32)})

    def test_register_cap_drops_spiller(self) -> None:
        configs = [
            _FakeConfig(256, 256, 64, 4, 4),
            _FakeConfig(128, 256, 128, 3, 8),
        ]
        kept = self._prune(8192, 8192, 2048, configs=configs)
        self.assertEqual(self._keys(kept), {(128, 256, 128)})

    def test_register_cap_boundary(self) -> None:
        configs = [
            _FakeConfig(255, 128, 64, 4, 4),
            _FakeConfig(256, 128, 64, 4, 4),
        ]
        kept = self._prune(8192, 8192, 2048, configs=configs)
        self.assertEqual(self._keys(kept), {(255, 128, 64)})

    def test_wgmma_floors_keep_minimum_tiles(self) -> None:
        configs = [
            _FakeConfig(64, 32, 32, 4, 4),
            _FakeConfig(128, 32, 32, 4, 4),
            _FakeConfig(64, 64, 32, 4, 4),
            _FakeConfig(64, 32, 64, 4, 4),
        ]
        kept = self._prune(1, 1, 1, configs=configs)
        self.assertEqual(self._keys(kept), {(64, 32, 32)})

    def test_tiny_n_keeps_bn32_winner(self) -> None:
        configs = [
            _FakeConfig(16, 32, 32, 3, 2),
            _FakeConfig(64, 64, 32, 4, 4),
            _FakeConfig(16, 128, 32, 3, 2),
        ]
        kept = self._prune(3, 4, 5, configs=configs)
        keys = self._keys(kept)
        self.assertIn((16, 32, 32), keys)
        self.assertNotIn((16, 128, 32), keys)
        self.assertLess(len(kept), len(configs))

    def test_ns_floor_drops_ns2_when_ns3_fits(self) -> None:
        configs = [_FakeConfig(32, 256, 64, ns, 4) for ns in (2, 3, 4, 5, 6)]
        configs += [_FakeConfig(64, 128, 32, ns, 4) for ns in (2, 4)]
        kept = self._prune(512, 1024, 512, configs=configs)
        io = sorted(c.num_stages for c in kept if c.kwargs["BLOCK_M"] == 32)
        compute = sorted(c.num_stages for c in kept if c.kwargs["BLOCK_M"] == 64)
        self.assertEqual(io, [3, 4, 5, 6])
        self.assertEqual(compute, [4])

    def test_ns_floor_keeps_ns2_when_ns3_exceeds_smem(self) -> None:
        configs = [
            _FakeConfig(128, 256, 256, 2, 8),
            _FakeConfig(128, 256, 256, 3, 8),
            _FakeConfig(64, 128, 128, 4, 4),
        ]
        kept = self._prune(512, 1024, 512, configs=configs)
        huge = sorted(c.num_stages for c in kept if c.kwargs["BLOCK_K"] == 256)
        self.assertEqual(huge, [2])
        self.assertIn((64, 128, 128), self._keys(kept))

    def test_bm_product_drops_small_bm_at_rich_grid(self) -> None:
        configs = [
            _FakeConfig(32, 256, 64, 4, 4),
            _FakeConfig(64, 256, 64, 4, 4),
        ]
        kept = self._prune(2048, 4096, 512, configs=configs)
        self.assertEqual(self._keys(kept), {(64, 256, 64)})

    def test_bm_product_keeps_small_bm_at_starved_grid(self) -> None:
        configs = [
            _FakeConfig(32, 64, 64, 4, 2),
            _FakeConfig(64, 64, 64, 4, 2),
        ]
        kept = self._prune(1024, 64, 512, configs=configs)
        self.assertEqual(self._keys(kept), {(32, 64, 64), (64, 64, 64)})

    def test_bn_product_drops_bn32_at_rich_grid(self) -> None:
        configs = [
            _FakeConfig(32, 32, 64, 4, 2),
            _FakeConfig(32, 64, 64, 4, 2),
        ]
        kept = self._prune(64, 6144, 2048, configs=configs)
        self.assertEqual(self._keys(kept), {(32, 64, 64)})

    def test_bn_product_keeps_bn32_below_threshold(self) -> None:
        configs = [
            _FakeConfig(16, 32, 64, 4, 2),
            _FakeConfig(16, 64, 64, 4, 2),
        ]
        kept = self._prune(64, 4096, 2048, configs=configs)
        self.assertEqual(self._keys(kept), {(16, 32, 64), (16, 64, 64)})

    def test_mid_m_drops_block_m_256(self) -> None:
        kept = self._prune(64, 1024, 1024)
        bms = {c.kwargs["BLOCK_M"] for c in kept}
        self.assertNotIn(256, bms)

    def test_large_m_keeps_block_m_256(self) -> None:
        kept = self._prune(8192, 1024, 256)
        keys = self._keys(kept)
        self.assertIn((256, 128, 32), keys)
        self.assertNotIn((256, 256, 32), keys)

    def test_stages_kept_when_pipeline_saturates(self) -> None:
        for m in (1, 256):
            kept = self._prune(m, 1024, 256)
            stages = sorted(c.num_stages for c in kept if c.kwargs["BLOCK_M"] == 16)
            self.assertEqual(stages, [3, 4, 5, 6])

    def test_smem_fit_keeps_tightest_config(self) -> None:
        kept = self._prune(512, 1024, 19712)
        self.assertIn((64, 128, 256), self._keys(kept))

    def test_smem_drops_infeasible_config(self) -> None:
        configs = [
            _FakeConfig(128, 256, 256, 4, 8),
            _FakeConfig(64, 64, 128, 5, 2),
        ]
        kept = self._prune(512, 1024, 19712, configs=configs)
        self.assertEqual(self._keys(kept), {(64, 64, 128)})

    def test_pipeline_drops_unsaturable_stages(self) -> None:
        configs = [_FakeConfig(16, 32, 64, ns, 2) for ns in (2, 3, 4, 5, 6)]
        kept = self._prune(64, 256, 256, configs=configs)
        stages = sorted(c.num_stages for c in kept)
        self.assertEqual(stages, [3, 4, 5])

    def test_pipeline_keeps_all_at_single_k_tile(self) -> None:
        configs = [_FakeConfig(16, 32, 32, ns, 2) for ns in (2, 3, 4, 5, 6)]
        kept = self._prune(512, 1024, 16, configs=configs)
        stages = sorted(c.num_stages for c in kept)
        self.assertEqual(stages, [3, 4, 5, 6])

    def test_pipeline_ignores_compute_configs(self) -> None:
        configs = [_FakeConfig(64, 32, 32, 5, 2)]
        kept = self._prune(512, 1024, 64, configs=configs)
        self.assertEqual(self._keys(kept), {(64, 32, 32)})

    def test_fallback_returns_full_space_when_all_pruned(self) -> None:
        configs = [c for c in self._make_configs() if c.kwargs["BLOCK_M"] >= 128]
        self.assertTrue(len(configs) > 0)
        kept = self._prune(1, 1024, 1024, configs=configs)
        self.assertEqual(kept, configs)

    def test_non_hopper_is_noop(self) -> None:
        configs = self._make_configs()
        kept = self._prune(1, 1024, 1024, hopper=False, configs=configs)
        self.assertEqual(kept, configs)


@unittest.skipIf(not torch.cuda.is_available(), "Skip when CUDA is not available")
class TestPerfModelFp8(unittest.TestCase):
    """estimate_matmul_time must accept torch fp8 dtypes.

    Regression test: Triton's peak-flops table has no torch fp8 entries, so
    unmapped dtypes crash with "dtype not supported" (this also fixes a latent
    crash for block kernels whenever early pruning keeps more than top_k).
    """

    def test_fp8_dtypes_accepted(self) -> None:
        device = torch.accelerator.current_accelerator()
        for dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            a = torch.empty(64, 256, dtype=dtype, device=device)
            b = torch.empty(256, 256, dtype=dtype, device=device)
            c = torch.empty(64, 256, dtype=torch.bfloat16, device=device)
            est = estimate_matmul_time(4, 4, a, b, c, 64, 256, 256, 64, 128, 32, 1)
            self.assertTrue(est > 0)
