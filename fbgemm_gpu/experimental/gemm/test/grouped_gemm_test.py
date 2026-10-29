# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[53]

import unittest

import fbgemm_gpu
import torch
from hypothesis import example, given, settings, strategies as st, Verbosity

open_source: bool = bool(getattr(fbgemm_gpu, "open_source", False))

if open_source:
    gpu_unavailable: tuple[bool, str] = (
        not torch.cuda.is_available() or torch.cuda.device_count() == 0,
        "CUDA is not available or no GPUs detected",
    )
else:
    from fbgemm_gpu.test.test_utils import gpu_unavailable

if torch.cuda.is_available():
    from fbgemm_gpu.experimental.gemm.triton_gemm.fp8_gemm import quantize_fp8_row
    from fbgemm_gpu.experimental.gemm.triton_gemm.grouped_gemm import (
        _HAS_WS_SUPPORT,
        grouped_gemm,
        grouped_gemm_fp8_rowwise,
    )

_MAX_SAMPLES = 32


@unittest.skipIf(
    not torch.cuda.is_available(),
    "Skip when CUDA is not available",
)
class TestGroupedGEMM(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    @unittest.skipIf(bool(gpu_unavailable[0]), str(gpu_unavailable[1]))
    @unittest.skipIf(
        (not torch.cuda.is_available())
        or (torch.version.hip is None)
        and (torch.cuda.get_device_properties(0).major < 9),
        "Skip FP8 test on architectures before SM90.",
    )
    def test_grouped_gemm_fp8_k_tail(self) -> None:
        device = torch.accelerator.current_accelerator()
        m_sizes = torch.tensor([33, 0, 36], dtype=torch.int32, device=device)
        M, N, G = 69, 80, 3
        for K in [64, 192]:
            a = torch.randn(M, K, dtype=torch.bfloat16, device=device)
            b = torch.randn(G * N, K, dtype=torch.bfloat16, device=device)
            a_fp8, a_scale = quantize_fp8_row(a)
            b_fp8, b_scale = quantize_fp8_row(b)
            expected = torch.empty(M, N, dtype=torch.bfloat16, device=device)
            start = 0
            for g, size in enumerate([33, 0, 36]):
                end = start + size
                expected[start:end] = (
                    a_fp8[start:end].float()
                    @ b_fp8[g * N : (g + 1) * N].float().T
                    * a_scale[start:end, None]
                    * b_scale[None, g * N : (g + 1) * N]
                ).to(torch.bfloat16)
                start = end

            for scatter_add in [False, True]:
                with self.subTest(K=K, scatter_add=scatter_add):
                    target = (
                        torch.randn(M, N, dtype=torch.bfloat16, device=device)
                        if scatter_add
                        else None
                    )
                    indices = (
                        torch.randperm(M, device=device).to(torch.int32)
                        if scatter_add
                        else None
                    )
                    reference = expected
                    if scatter_add:
                        assert target is not None and indices is not None
                        reference = target.clone().scatter_add_(
                            0, indices.long()[:, None].expand(M, N), expected
                        )
                    result = grouped_gemm_fp8_rowwise(
                        a_fp8,
                        b_fp8,
                        m_sizes,
                        a_scale,
                        b_scale,
                        _use_warp_specialization=False,
                        _output_tensor=target,
                        _scatter_add_indices=indices,
                    )
                    torch.testing.assert_close(
                        result, reference, atol=2e-2, rtol=1.6e-2
                    )

    @given(
        G=st.sampled_from([1, 4, 16, 128]),
        M=st.sampled_from([0, 128, 2048, 16384]),
        N=st.sampled_from([256]),
        K=st.sampled_from([64, 128, 256]),
        fast_accumulation=st.sampled_from([True, False]),
        warp_specialization=st.sampled_from(
            [False]
            # TODO(T224502057): Re-enable the test after fixing WS hanging issue.
            # [True, False] if _HAS_WS_SUPPORT else [False]
        ),
        fuse_scatter_add=st.sampled_from([True, False]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=_MAX_SAMPLES, deadline=None)
    @unittest.skipIf(  # pyre-ignore [56]
        (not torch.cuda.is_available())
        or (torch.version.hip is None)
        and (torch.cuda.get_device_properties(0).major < 9),
        "Skip FP8 test on architectures before SM90.",
    )
    def test_grouped_gemm_fp8_rowwise(
        self,
        G: int,
        M: int,
        N: int,
        K: int,
        fast_accumulation: bool,
        warp_specialization: bool,
        fuse_scatter_add: bool,
    ) -> None:
        torch.manual_seed(0)

        device = torch.accelerator.current_accelerator("cuda")
        a = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        b = torch.randn(N * G, K, dtype=torch.bfloat16, device=device)
        m_ends, _ = torch.sort(
            torch.randint(low=0, high=M, size=[G - 1], device=device, dtype=torch.int32)
            if M > 0
            else torch.zeros([G - 1], device=device, dtype=torch.int32)
        )
        m_ends = m_ends.tolist()
        m_starts = [0] + m_ends
        m_ends = m_ends + [M]
        m_sizes = torch.tensor(
            [m_ends[i] - m_starts[i] for i in range(G)], device=device
        ).to(torch.int32)

        a_fp8, a_scale = quantize_fp8_row(a)
        b_fp8, b_scale = quantize_fp8_row(b)

        if fuse_scatter_add:
            scatter_add_target = torch.randn(M, N, dtype=torch.bfloat16, device=device)
            scatter_add_indices = torch.randperm(M, device=device).to(torch.int32)
            scatter_add_target_clone = scatter_add_target.clone()
        else:
            scatter_add_target = None
            scatter_add_indices = None
            scatter_add_target_clone = None

        result = grouped_gemm_fp8_rowwise(
            a_fp8,
            b_fp8,
            m_sizes,
            a_scale,
            b_scale,
            use_fast_accum=fast_accumulation,
            _use_warp_specialization=warp_specialization,
            _output_tensor=scatter_add_target,
            _scatter_add_indices=scatter_add_indices,
        )
        self.assertTrue(result.shape == (M, N))

        if M == 0:
            return

        expected_result = torch.zeros(M, N, dtype=torch.bfloat16, device=device)
        # Running baseline with quantization to exclude quantization error from the test as it has nothing to do with the correctness of the kernel implementation.
        for g in range(G):
            m_start = m_starts[g]
            m_end = m_ends[g]
            n_start = g * N
            n_end = (g + 1) * N

            expected_result[m_start:m_end, :] = (
                a_fp8[m_start:m_end, :].to(torch.float32)
                @ b_fp8[n_start:n_end, :].to(torch.float32).T
                * a_scale[m_start:m_end][:, None]
                * b_scale[n_start:n_end][None, :]
            ).to(torch.bfloat16)

        if fuse_scatter_add:
            assert scatter_add_target_clone is not None
            assert scatter_add_indices is not None
            scatter_add_target_clone.scatter_add_(
                0,
                scatter_add_indices.view(M, 1).expand(M, N).to(torch.int64),
                expected_result,
            )
            expected_result = scatter_add_target_clone

        def msg(s: str) -> str:
            return f"{G=}, {M=}, {N=}, {K=}, {fast_accumulation=}, {warp_specialization=}, {fuse_scatter_add=}, {s}"

        if M >= 16384:
            torch.testing.assert_close(
                result,
                expected_result,
                atol=5e-2,
                rtol=1.6e-2,
                msg=msg,
            )
        else:
            torch.testing.assert_close(
                result,
                expected_result,
                atol=2e-2,
                rtol=1.6e-2,
                msg=msg,
            )

    @example(
        G=1, M=2048, N=256, K=256, warp_specialization=False, fuse_scatter_add=True
    )
    @example(
        G=4, M=2048, N=256, K=256, warp_specialization=False, fuse_scatter_add=True
    )
    @given(
        G=st.sampled_from([1, 4, 16, 128]),
        M=st.sampled_from([0, 128, 2048, 16384]),
        N=st.sampled_from([256, 451]),
        K=st.sampled_from([100, 256, 257]),
        warp_specialization=st.sampled_from(
            [True, False] if torch.cuda.is_available() and _HAS_WS_SUPPORT else [False]
        ),
        fuse_scatter_add=st.sampled_from([True, False]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=_MAX_SAMPLES, deadline=None)
    # TODO(shikaili): Re-enable the test for SM80 after fixing TMA issues.
    @unittest.skipIf(  # pyre-ignore [56]
        (not torch.cuda.is_available())
        or (torch.version.hip is None)
        and (torch.cuda.get_device_properties(0).major < 9),
        "Skip BF16 test on architectures before SM90.",
    )
    def test_grouped_gemm_bf16(
        self,
        G: int,
        M: int,
        N: int,
        K: int,
        warp_specialization: bool,
        fuse_scatter_add: bool,
    ) -> None:
        # TODO: fix crashes for K divisible by 8 but not divisible by 128, like 264.
        if K % 8 != 0 or N % 8 != 0:
            # When K or N are not a multiple of 8, using fuse_scatter_add has large numerical discrepancy,
            # possibly due to atomic add in scatter_add or larger rounding error with negative numbers.
            fuse_scatter_add = False

        torch.manual_seed(0)

        device = torch.accelerator.current_accelerator()
        a = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        b = torch.randn(N * G, K, dtype=torch.bfloat16, device=device)
        m_ends, _ = torch.sort(
            torch.randint(low=0, high=M, size=[G - 1], device=device, dtype=torch.int32)
            if M > 0
            else torch.zeros([G - 1], device=device, dtype=torch.int32)
        )
        m_ends = m_ends.tolist()
        m_starts = [0] + m_ends
        m_ends = m_ends + [M]
        m_sizes = torch.tensor(
            [m_ends[i] - m_starts[i] for i in range(G)], device=device
        ).to(torch.int32)

        if fuse_scatter_add:
            scatter_add_target = torch.randn(M, N, dtype=torch.bfloat16, device=device)
            scatter_add_indices = torch.randperm(M, device=device).to(torch.int32)
            scatter_add_target_clone = scatter_add_target.clone()
        else:
            scatter_add_target = None
            scatter_add_indices = None
            scatter_add_target_clone = None

        result = grouped_gemm(
            a,
            b,
            m_sizes,
            _use_warp_specialization=warp_specialization,
            _output_tensor=scatter_add_target,
            _scatter_add_indices=scatter_add_indices,
        )
        self.assertTrue(result.shape == (M, N))

        if M == 0:
            return

        expected_result = torch.zeros(M, N, dtype=torch.bfloat16, device=device)
        for g in range(G):
            m_start = m_starts[g]
            m_end = m_ends[g]
            expected_result[m_start:m_end, :] = (
                a[m_start:m_end, :] @ b[g * N : (g + 1) * N, :].T
            )

        def msg(s: str) -> str:
            return f"{G=}, {M=}, {N=}, {K=}, {warp_specialization=}, {fuse_scatter_add=}, {s}"

        if fuse_scatter_add:
            unfused_result = grouped_gemm(
                a, b, m_sizes, _use_warp_specialization=warp_specialization
            )
            torch.testing.assert_close(
                unfused_result, expected_result, atol=1e-5, rtol=1.6e-2, msg=msg
            )
            expected_result = unfused_result
            assert scatter_add_target_clone is not None
            assert scatter_add_indices is not None
            scatter_add_target_clone.scatter_add_(
                0,
                scatter_add_indices.view(M, 1).expand(M, N).to(torch.int64),
                expected_result,
            )
            expected_result = scatter_add_target_clone

        torch.testing.assert_close(
            result, expected_result, atol=1e-5, rtol=1.6e-2, msg=msg
        )

    def test_grouped_gemm_bf16_ws_request(self) -> None:
        device = torch.accelerator.current_accelerator()
        a = torch.randn(128, 256, dtype=torch.bfloat16, device=device)
        b = torch.randn(256, 256, dtype=torch.bfloat16, device=device)
        sizes = torch.tensor([128], dtype=torch.int32, device=device)
        result = grouped_gemm(a, b, sizes, _use_warp_specialization=True)
        torch.testing.assert_close(result, a @ b.T, atol=1e-5, rtol=1.6e-2)
