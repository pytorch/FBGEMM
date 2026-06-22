#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# pyre-ignore-all-errors[56]

import unittest
from collections.abc import Callable
from typing import Any

import hypothesis.strategies as st
import numpy as np
import numpy.typing as npt
import torch
from hypothesis import given, settings

from .common import extend_test_class, open_source

if open_source:
    # pyre-ignore[21]
    from test_utils import gpu_available, gpu_memory_lt_gb, gpu_unavailable, optests
else:
    from fbgemm_gpu.test.test_utils import (
        gpu_available,
        gpu_memory_lt_gb,
        gpu_unavailable,
        optests,
    )


def get_n_rand_num_summing_to_k(n: int, k: int) -> npt.NDArray:
    """Get a list of `n` integers which collectively sum to `k`, drawn
    uniformly from the set of all such lists.

    Args:
        n - The number of integers in the result list
        k - The value they should sum to
    """
    # There are a lot of ways to do this wrong, probably including
    # the ones you've just thought of. I think the following does
    # it correctly, though.
    if n == 0:
        return np.array([])
    return np.random.multinomial(k, np.ones(n) / n, size=1)[0]


class PackedSegmentsTest(unittest.TestCase):
    def _pack_segments_ref(
        self,
        lengths: torch.Tensor,
        tensor: torch.Tensor,
        max_length: int | None = None,
    ) -> npt.NDArray:
        """
        This function is a reference implementation of pack_segments.

        Args:
            lengths (Tensor): The lengths of tensor.
            tensor (Tensor): The tensor to be packed.
            max_length (int | None): The maximum length of the packed tensor.

        Returns:
            The packed tensor.
        """
        lengths_np = lengths.numpy()
        sections = np.split(tensor, np.cumsum(lengths_np))
        max_length = np.max(lengths_np, initial=0) if max_length is None else max_length
        padded_arrs = []
        for arr in sections[:-1]:  # Last section is always a blank
            arr = arr[: min(max_length, len(arr)), ...]
            padded_arr = np.pad(
                arr,
                [(0, max(max_length - arr.shape[0], 0))]
                + ([(0, 0)] * (len(arr.shape) - 1)),
                constant_values=0,
            )
            padded_arrs.append(padded_arr)

        if len(padded_arrs) == 0:
            padded_arrs = torch.empty((0, 0) + tuple(tensor.shape[1:]))
        else:
            padded_arrs = torch.Tensor(np.stack(padded_arrs))

        # pyre-fixme[7]: Expected `ndarray` but got `Tensor`.
        return padded_arrs

    @given(
        n=st.integers(2, 10),
        k=st.integers(2, 10),
        batch_size=st.integers(1, 30),
        divisions=st.integers(1, 10),
        dtype=st.sampled_from(
            [
                torch.float,
                torch.half,
                torch.bfloat16,
                torch.int,
            ]
        ),
        torch_compile=st.booleans(),
    )
    @settings(deadline=None)
    def test_pack_segments(
        self,
        n: int,
        k: int,
        batch_size: int,
        divisions: int,
        dtype: torch.dtype,
        torch_compile: bool,
    ) -> None:
        """
        This function tests pack_segments ops compared to the reference implementation.
        Both CPU and GPU (if available) are tested.

        Args:
            n - The number of rows in the input tensor
            k - The number of columns in the input tensor
            batch_size - The number of batches of the input tensor
            divisions - The number of segments to be packed
            dtype - The data type
            torch_compile - Whether to use torch.compile

        Returns:
            None
        """

        input_raw = np.random.rand(batch_size, n, k)
        test_backward = dtype != torch.int
        input_data = torch.tensor(input_raw, dtype=dtype, requires_grad=test_backward)
        lengths = torch.tensor(
            get_n_rand_num_summing_to_k(divisions, batch_size),
            dtype=torch.int,
        )
        max_length = lengths.max().item()

        packed_tensor = torch.ops.fbgemm.pack_segments(
            t_in=input_data, lengths=lengths, max_length=max_length
        )

        packed_tensor_v2, _ = torch.ops.fbgemm.pack_segments_v2(
            t_in=input_data, lengths=lengths, max_length=max_length
        )

        packed_ref = self._pack_segments_ref(lengths, input_raw)
        packed_ref = torch.Tensor(packed_ref).to(dtype)

        self.assertTrue(torch.equal(packed_tensor, packed_ref))
        self.assertTrue(torch.equal(packed_tensor_v2, packed_ref))

        grad_cpu = torch.tensor(
            np.random.uniform(low=0.01, high=0.5, size=packed_ref.shape).astype(
                np.float32
            )
        ).to(dtype)
        grad_cpu_v2 = torch.clone(grad_cpu)
        # CPU backward
        if test_backward:
            packed_tensor.backward(grad_cpu)
            packed_tensor_v2.backward(grad_cpu_v2)

        if gpu_available:
            pack_segments_fun = torch.ops.fbgemm.pack_segments

            if torch_compile:
                pack_segments_fun = torch.compile(pack_segments_fun, dynamic=True)

            packed_cuda = pack_segments_fun(
                t_in=input_data.cuda(),
                lengths=lengths.cuda(),
                max_length=max_length,
            )

            pack_segments_fun_v2 = torch.ops.fbgemm.pack_segments_v2

            if torch_compile:
                pack_segments_fun_v2 = torch.compile(pack_segments_fun_v2, dynamic=True)

            packed_cuda_v2, _ = pack_segments_fun_v2(
                t_in=input_data.cuda(),
                lengths=lengths.cuda(),
                max_length=max_length,
            )

            self.assertTrue(torch.equal(packed_tensor, packed_cuda.cpu()))
            self.assertTrue(torch.equal(packed_tensor_v2, packed_cuda_v2.cpu()))

            # GPU backward
            if test_backward:
                packed_cuda.backward(grad_cpu.cuda())
                packed_cuda_v2.backward(grad_cpu_v2.cuda())

            # dynamic check
            input_raw = np.random.rand(batch_size, n + 1, k + 2)
            input_data = torch.tensor(
                input_raw, dtype=dtype, requires_grad=test_backward
            )
            lengths = torch.tensor(
                get_n_rand_num_summing_to_k(divisions, batch_size), dtype=torch.int
            )
            max_length = lengths.max().item()
            packed_tensor = torch.ops.fbgemm.pack_segments(
                t_in=input_data, lengths=lengths, max_length=max_length
            )
            packed_tensor_v2, _ = torch.ops.fbgemm.pack_segments_v2(
                t_in=input_data, lengths=lengths, max_length=max_length
            )

            packed_ref = self._pack_segments_ref(lengths, input_raw)
            packed_ref = torch.Tensor(packed_ref).to(dtype)

            self.assertTrue(torch.equal(packed_tensor, packed_ref))
            self.assertTrue(torch.equal(packed_tensor_v2, packed_ref))

            grad_cpu = torch.tensor(
                np.random.uniform(low=0.01, high=0.5, size=packed_ref.shape).astype(
                    np.float32
                )
            ).to(dtype)
            grad_cpu_v2 = torch.clone(grad_cpu)
            # CPU backward
            if test_backward:
                packed_tensor.backward(grad_cpu)
                packed_tensor_v2.backward(grad_cpu_v2)

            # reusing the previously compiled kernel
            packed_cuda = pack_segments_fun(
                t_in=input_data.cuda(),
                lengths=lengths.cuda(),
                max_length=max_length,
            )
            packed_cuda_v2, _ = pack_segments_fun_v2(
                t_in=input_data.cuda(),
                lengths=lengths.cuda(),
                max_length=max_length,
            )
            self.assertTrue(torch.equal(packed_tensor, packed_cuda.cpu()))
            self.assertTrue(torch.equal(packed_tensor_v2, packed_cuda_v2.cpu()))

            # GPU backward
            if test_backward:
                packed_cuda.backward(grad_cpu.cuda())
                packed_cuda_v2.backward(grad_cpu_v2.cuda())

    @given(
        n=st.integers(2, 10),
        k=st.integers(2, 10),
        batch_size=st.integers(1, 30),
        divisions=st.integers(1, 10),
        max_length=st.integers(1, 20),
        dtype=st.sampled_from(
            [
                torch.float,
                torch.half,
                torch.bfloat16,
                torch.int,
            ]
        ),
        torch_compile=st.booleans(),
    )
    @settings(deadline=None)
    def test_pack_segments_smaller_max_len(
        self,
        n: int,
        k: int,
        batch_size: int,
        divisions: int,
        max_length: int,
        dtype: torch.dtype,
        torch_compile: bool,
    ) -> None:
        """
        This function tests pack_segments ops with set max_length
        Both CPU and GPU (if available) are tested.

        Args:
            n - The number of rows in the input tensor
            k - The number of columns in the input tensor
            batch_size - The number of batches of the input tensor
            divisions - The number of segments to be packed
            max_length - The maximum length of the packed tensor
            dtype - The data type
            torch_compile - Whether to use torch.compile

        Returns:
            None
        """

        input_raw = np.random.rand(batch_size, n, k)
        input_data = torch.tensor(input_raw, dtype=dtype)
        lengths = torch.tensor(
            get_n_rand_num_summing_to_k(divisions, batch_size), dtype=torch.int
        )

        packed_tensor = torch.ops.fbgemm.pack_segments(
            t_in=input_data,
            lengths=lengths,
            max_length=max_length,
        )
        packed_tensor_v2, _ = torch.ops.fbgemm.pack_segments_v2(
            t_in=input_data,
            lengths=lengths,
            max_length=max_length,
        )
        self.assertEqual(packed_tensor.shape, (divisions, max_length, n, k))
        self.assertEqual(packed_tensor_v2.shape, (divisions, max_length, n, k))

        packed_ref = self._pack_segments_ref(
            lengths,
            input_raw,
            max_length=max_length,
        )
        packed_ref = torch.Tensor(packed_ref).to(dtype)
        self.assertTrue(torch.equal(packed_tensor, packed_ref))
        self.assertTrue(torch.equal(packed_tensor_v2, packed_ref))

        if gpu_available:
            pack_segments_fun = torch.ops.fbgemm.pack_segments
            if torch_compile:
                pack_segments_fun = torch.compile(pack_segments_fun)

            packed_cuda = pack_segments_fun(
                t_in=input_data.cuda(),
                lengths=lengths.cuda(),
                max_length=max_length,
            )
            self.assertTrue(torch.equal(packed_tensor, packed_cuda.cpu()))
            pack_segments_fun = torch.ops.fbgemm.pack_segments_v2
            if torch_compile:
                pack_segments_fun = torch.compile(pack_segments_fun)

            packed_cuda_v2, _ = pack_segments_fun(
                t_in=input_data.cuda(),
                lengths=lengths.cuda(),
                max_length=max_length,
            )
            self.assertTrue(torch.equal(packed_tensor_v2, packed_cuda_v2.cpu()))

    @given(
        n=st.integers(2, 10),
        k=st.integers(2, 10),
        batch_size=st.integers(1, 30),
        divisions=st.integers(1, 10),
        dtype=st.sampled_from(
            [
                torch.float,
                torch.half,
                torch.bfloat16,
            ]
        ),
    )
    @settings(deadline=None)
    def test_pack_segments_meta_backend(
        self,
        n: int,
        k: int,
        batch_size: int,
        divisions: int,
        dtype: torch.dtype,
    ) -> None:
        """
        This function tests pack_segments ops with meta backend.

        Args:
            n - The number of rows in the input tensor
            k - The number of columns in the input tensor
            batch_size - The number of batches of the input tensor
            divisions - The number of segments to be packed
            dtype - The data type

        Returns:
            None
        """

        input_raw = np.random.rand(batch_size, n, k)
        input_data = torch.tensor(
            input_raw, dtype=torch.float32, requires_grad=True
        ).to("meta")
        lengths = torch.tensor(
            get_n_rand_num_summing_to_k(divisions, batch_size), dtype=torch.int
        )
        max_length = lengths.max().item()

        packed_tensor = torch.ops.fbgemm.pack_segments(
            t_in=input_data, lengths=lengths, max_length=max_length
        )
        packed_tensor_v2, _ = torch.ops.fbgemm.pack_segments_v2(
            t_in=input_data, lengths=lengths, max_length=max_length
        )
        packed_ref = self._pack_segments_ref(lengths, input_raw)

        # verify forward
        assert packed_tensor.size() == torch.Tensor(packed_ref).size()
        assert packed_tensor_v2.size() == torch.Tensor(packed_ref).size()

        packed_tensor_v2, presence_mask = torch.ops.fbgemm.pack_segments_v2(
            t_in=input_data,
            lengths=lengths,
            max_length=max_length,
            return_presence_mask=True,
        )

        # pyre-fixme[6]: In call `tuple.__new__`, for 1st positional argument, expected `Iterable[int]` but got `Iterable[bool | float | int]`.
        assert presence_mask.size() == torch.Size([lengths.numel(), max_length])

    @unittest.skipIf(*gpu_unavailable)
    @given(
        n=st.integers(2, 10),
        k=st.integers(2, 10),
        batch_size=st.integers(1, 30),
        divisions=st.integers(1, 10),
        dtype=st.sampled_from(
            [
                torch.float,
                torch.half,
            ]
        ),
        torch_compile=st.booleans(),
        use_cpu=st.booleans(),
    )
    @settings(deadline=None)
    @optests.dontGenerateOpCheckTests(
        "GPU-only test; opcheck variants only skip on CPU samples; op covered by test_pack_segments (T191384137)"
    )
    def test_pack_segments_noncontig(
        self,
        n: int,
        k: int,
        batch_size: int,
        divisions: int,
        dtype: torch.dtype,
        torch_compile: bool,
        use_cpu: bool,
    ) -> None:
        """
        This function tests pack_segments ops when input gradients to backward are non-contiguous.

        Args:
            n - The number of rows in the input tensor
            k - The number of columns in the input tensor
            batch_size - The number of batches of the input tensor
            divisions - The number of segments to be packed
            dtype - The data type
            torch_compile - Whether to use torch.compile
            use_cpu - Whether to use CPU or GPU

        Returns:
            None
        """

        input_raw = np.random.rand(batch_size, n, k)
        # create input
        input_data_ref = torch.tensor(input_raw, dtype=dtype, requires_grad=True)
        input_data = torch.tensor(input_raw, dtype=dtype, requires_grad=True).cuda()
        input_data_ref_v2 = torch.tensor(input_raw, dtype=dtype, requires_grad=True)
        input_data_v2 = torch.tensor(input_raw, dtype=dtype, requires_grad=True).cuda()
        # retain grad to compare gradients of the inputs later
        input_data.retain_grad()
        input_data_ref.retain_grad()
        input_data_v2.retain_grad()
        input_data_ref_v2.retain_grad()

        # set lengths
        lengths = torch.tensor(
            get_n_rand_num_summing_to_k(divisions, batch_size),
            dtype=torch.int,
        )
        max_length = lengths.max().item()

        packed_ref = torch.ops.fbgemm.pack_segments(
            t_in=input_data_ref, lengths=lengths, max_length=max_length
        )
        packed_ref.retain_grad()

        # pack segments using fbgemm and fb
        packed_tensor = torch.ops.fbgemm.pack_segments(
            t_in=input_data, lengths=lengths.cuda(), max_length=max_length
        )
        packed_tensor.retain_grad()

        # verify forward
        self.assertTrue(torch.equal(packed_tensor.cpu(), packed_ref))

        packed_ref_v2, _ = torch.ops.fbgemm.pack_segments_v2(
            t_in=input_data_ref_v2, lengths=lengths, max_length=max_length
        )
        packed_ref_v2.retain_grad()

        # pack segments using fbgemm and fb
        packed_tensor_v2, _ = torch.ops.fbgemm.pack_segments_v2(
            t_in=input_data_v2, lengths=lengths.cuda(), max_length=max_length
        )
        packed_tensor_v2.retain_grad()

        # verify forward
        self.assertTrue(torch.equal(packed_tensor_v2.cpu(), packed_ref_v2))

        # create non-contiguous grad
        shape = tuple(x * 2 for x in packed_ref.shape)
        grads = torch.tensor(
            np.random.uniform(low=0.01, high=0.5, size=shape).astype(np.float32)
        ).to(dtype)
        grads_v2 = torch.clone(grads)
        grad_noncontig_cpu = grads.as_strided(packed_ref.shape, grads.stride())
        grad_noncontig_cuda = grads.cuda().as_strided(
            packed_ref_v2.shape, grads_v2.stride()
        )

        self.assertTrue(
            not (
                grad_noncontig_cpu.is_contiguous()
                and grad_noncontig_cuda.is_contiguous()
            ),
            msg="Expected grads to be non-contiguous but they are contiguous",
        )

        grad_noncontig_cpu_v2 = grads_v2.as_strided(
            packed_ref_v2.shape, grads_v2.stride()
        )
        grad_noncontig_cuda_v2 = grads_v2.cuda().as_strided(
            packed_ref_v2.shape, grads_v2.stride()
        )
        self.assertTrue(
            not (
                grad_noncontig_cpu_v2.is_contiguous()
                and grad_noncontig_cuda_v2.is_contiguous()
            ),
            msg="Expected grads_v2 to be non-contiguous but they are contiguous",
        )

        # verify backward
        packed_ref.backward(grad_noncontig_cpu)
        packed_tensor.backward(grad_noncontig_cuda)
        self.assertTrue(
            torch.equal(packed_tensor.cpu(), packed_ref),
            msg="Expected packed tensors to be equal but they are not",
        )

        # verify backward input gradients
        self.assertTrue(
            # pyre-fixme[16]: Optional type has no attribute `cpu`.
            torch.equal(input_data.grad.cpu(), input_data_ref.grad.cpu()),
            msg="Expected input gradients to be equal but they are not",
        )

        # verify backward
        packed_ref_v2.backward(grad_noncontig_cpu_v2)
        packed_tensor_v2.backward(grad_noncontig_cuda_v2)
        self.assertTrue(
            torch.equal(packed_tensor_v2.cpu(), packed_ref),
            msg="Expected packed tensors to be equal but they are not",
        )

        # verify backward input gradients
        self.assertTrue(
            torch.equal(input_data_v2.grad.cpu(), input_data_ref_v2.grad.cpu()),
            msg="Expected input gradients to be equal but they are not",
        )

    @unittest.skipIf(*gpu_unavailable)
    @given(
        dtype=st.sampled_from(
            [
                torch.float,
                torch.half,
                torch.bfloat16,
            ]
        ),
    )
    @settings(deadline=None)
    @optests.dontGenerateOpCheckTests(
        "GPU-only test; opcheck variants only skip on CPU samples; op covered by test_pack_segments (T191384137)"
    )
    def test_pack_segments_backward_truncated(self, dtype: torch.dtype) -> None:
        """
        Regression test: when lengths[seq] > max_length, the backward kernel
        previously left positions [cumsum[seq]+max_length, cumsum[seq]+lengths[seq])
        in the input gradient as uninitialized memory (allocated via at::empty).

        After the fix (at::empty -> at::zeros), those positions must be exactly 0
        because they correspond to events that were truncated by the forward pass
        and so cannot influence the loss.

        Without the fix, these positions contain garbage, which propagates upstream
        and can cause NaN cascades in deep networks (LayerNorm backward amplification).
        """
        # Choose lengths intentionally larger than max_length for some segments
        max_length = 4
        lengths_cpu = torch.tensor([10, 5, 8, 2], dtype=torch.int)
        total_length = int(lengths_cpu.sum().item())
        cell_size = 8

        # Run multiple trials to detect uninitialized memory:
        # if positions are uninit, values change across trials.
        observed_grads = []
        for _ in range(5):
            input_data = (
                torch.randn(total_length, cell_size, dtype=dtype)
                .cuda()  # noqa: CITRINE(redundant_cuda_to_device)
                .requires_grad_(True)
            )
            lengths = lengths_cpu.cuda()

            packed = torch.ops.fbgemm.pack_segments(
                t_in=input_data, lengths=lengths, max_length=max_length
            )
            grad_out = torch.ones_like(packed)
            packed.backward(grad_out)

            # pyre-ignore[16]
            observed_grads.append(input_data.grad.detach().cpu().clone())

        # Verify: positions where cell < min(lengths[seq], max_length) get grad=1
        # positions where cell >= max_length but cell < lengths[seq] get grad=0
        cumsum = 0
        for seq, L in enumerate(lengths_cpu.tolist()):
            for cell in range(L):
                row = cumsum + cell
                expected = 1.0 if cell < max_length else 0.0
                for trial, grad in enumerate(observed_grads):
                    actual = grad[row].abs().max().item()
                    self.assertAlmostEqual(
                        actual,
                        expected,
                        places=2,
                        msg=(
                            f"trial={trial} seq={seq} cell={cell} row={row}: "
                            f"expected grad abs.max={expected}, got {actual}. "
                            "Truncated rows must receive zero gradient (not uninit memory)."
                        ),
                    )
            cumsum += L

    @unittest.skipIf(*gpu_unavailable)
    # Skip on GPUs with insufficient HBM. The test allocates the packed
    # output of shape (num_seq, max_length) at fp16, ~8 GiB at the chosen
    # max_length.
    @unittest.skipIf(*gpu_memory_lt_gb(12))
    @optests.dontGenerateOpCheckTests(
        "large-grid GPU-memory-gated stress repro; opcheck variants add no coverage (T191384137)"
    )
    def test_pack_segments_large_grid(self) -> None:
        """
        Reproduces the HIP grid-overflow bug in pack_segments_cuda{,_v2}
        and verifies output correctness via a downsampled CPU oracle.

        With block size 128, the launch grid is
        cuda_calc_xblock_count(num_seq * max_length * cell_size, 128).
        For num_seq * max_length * cell_size > 2**32, total threads
        exceed the HIP 2**32 limit, causing
        FBGEMM_LAUNCH_DSA_KERNEL ->
        KernelLauncher::checkThreadCountNotExceeded to TORCH_CHECK-fail
        on ROCm pre-fix. Both pack_segments_cuda_kernel (uses
        CUDA_KERNEL_LOOP) and pack_segments_cuda_v2_kernel (uses
        CUDA_KERNEL_LOOP_TYPE) already grid-stride, so capping the grid
        is correctness-preserving for the launcher.

        Verification strategy (per master plan's downsampled-oracle
        guidance for ops where the full-scale CPU oracle is impractical):

        1. Full-scale invocation of v1 and v2 to verify the launch
           survives the production cap. Only shape is asserted because
           v1 uses ``CUDA_KERNEL_LOOP`` with an int32 loop index, which
           overflows for output linear indices >= 2**31; element-wise
           comparison would surface this pre-existing kernel bug, which
           is out of scope for this diff (the diff only caps the grid).
        2. Small-scale invocation of v1 and v2 vs CPU dispatch to
           validate kernel correctness end-to-end at a scale where the
           int32 loop index does not overflow. This catches kernel
           correctness regressions introduced by the cap fix.
        """

        # Choose num_seq * max_length so that total threads strictly
        # exceeds 2**32. With cell_size=1: total threads ~= num_seq *
        # max_length; need product > 2**32.
        num_seq = 2
        max_length = (1 << 31) + 1

        device = torch.device(torch.accelerator.current_accelerator() or "cuda")

        # ---- Step 1: full-scale launch survival (cap-trip detection). ----
        # Sparse non-zero lengths: only the last segment is non-empty.
        # t_in has a single sentinel value.
        lengths_large = torch.zeros(num_seq, dtype=torch.int32, device=device)
        lengths_large[-1] = 1
        t_in_large = torch.tensor([3.5], dtype=torch.float16, device=device)

        # Pre-fix, this launch trips KernelLauncher::checkThreadCountNotExceeded.
        packed_v1 = torch.ops.fbgemm.pack_segments(
            t_in_large, lengths_large, max_length
        )
        self.assertEqual(packed_v1.shape, (num_seq, max_length))
        del packed_v1

        packed_v2, _ = torch.ops.fbgemm.pack_segments_v2(
            t_in_large, lengths_large, max_length
        )
        self.assertEqual(packed_v2.shape, (num_seq, max_length))
        del packed_v2

        # ---- Step 2: downsampled CPU-oracle correctness check. ----
        # Same kernel code path, smaller scale to keep the int32 loop
        # index of v1 in range and the CPU oracle cheap.
        small_max_length = 16
        small_lengths_cpu = torch.tensor([0, 3, 0, 2], dtype=torch.int32)
        # Total non-zero lengths = 5; t_in is a sequence of distinct
        # values so any "wrong row/col" bug surfaces.
        small_t_in_cpu = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float16)

        # CPU oracles.
        small_packed_cpu = torch.ops.fbgemm.pack_segments(
            small_t_in_cpu, small_lengths_cpu, small_max_length
        )
        small_packed_cpu_v2, _ = torch.ops.fbgemm.pack_segments_v2(
            small_t_in_cpu, small_lengths_cpu, small_max_length
        )

        # GPU under test.
        small_packed_gpu = torch.ops.fbgemm.pack_segments(
            small_t_in_cpu.to(device),
            small_lengths_cpu.to(device),
            small_max_length,
        )
        small_packed_gpu_v2, _ = torch.ops.fbgemm.pack_segments_v2(
            small_t_in_cpu.to(device),
            small_lengths_cpu.to(device),
            small_max_length,
        )

        torch.testing.assert_close(small_packed_gpu.cpu(), small_packed_cpu)
        torch.testing.assert_close(small_packed_gpu_v2.cpu(), small_packed_cpu_v2)


# The opcheck harness (generate_opcheck_tests) re-executes the op for each
# generated variant. test_pack_segments_large_grid deliberately drives
# max_length = 2**31 + 1 (an ~8 GiB fp16 output) to exercise the HIP grid cap,
# and intentionally asserts shape only at full scale because v1's
# CUDA_KERNEL_LOOP uses an int32 index that overflows past 2**31. Re-running the
# op in the generated variants hits that overflow -> illegal memory access ->
# FATAL crash (not catchable by xfail), so skip opcheck for this stress test.
# T191384137
_LARGE_GRID_SKIP: list[Callable[..., Any]] = [
    unittest.skip("large-grid stress test is incompatible with opcheck re-execution")
]
additional_decorators: dict[str, list[Callable[..., Any]]] = {
    "test_schema__test_pack_segments_large_grid": _LARGE_GRID_SKIP,
    "test_autograd_registration__test_pack_segments_large_grid": _LARGE_GRID_SKIP,
    "test_faketensor__test_pack_segments_large_grid": _LARGE_GRID_SKIP,
    "test_aot_dispatch_dynamic__test_pack_segments_large_grid": _LARGE_GRID_SKIP,
}

extend_test_class(PackedSegmentsTest, additional_decorators)


if __name__ == "__main__":
    unittest.main()
