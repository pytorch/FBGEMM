#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors[56]

import unittest

import hypothesis.strategies as st
import torch
import torch._dynamo
from hypothesis import given, settings, Verbosity

from .common import additional_decorators, open_source

if open_source:
    # pyre-ignore[21]
    from test_utils import (
        cpu_and_maybe_gpu,
        gpu_unavailable,
        optests,
        symint_vector_unsupported,
    )
else:
    from fbgemm_gpu.test.test_utils import (
        cpu_and_maybe_gpu,
        gpu_unavailable,
        optests,
        symint_vector_unsupported,
    )


def repeat_arange_ref(lengths: torch.Tensor) -> torch.Tensor:
    """
    Reference implementation using PyTorch ops.

    This is the original implementation from vista_utils.py that uses:
    - asynchronous_complete_cumsum (1 kernel)
    - arange (1 kernel)
    - repeat_interleave (1 kernel)
    - subtraction (1 kernel)
    Total: 4 kernels with multiple intermediate allocations
    """
    offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
    offsets_without_last, max_len = offsets[:-1], int(offsets[-1])
    global_indices = torch.arange(max_len, device=lengths.device, dtype=lengths.dtype)
    repeated_offsets = torch.repeat_interleave(offsets_without_last, lengths.int())
    return global_indices - repeated_offsets


@optests.generate_opcheck_tests(additional_decorators=additional_decorators)
class RepeatArangeTest(unittest.TestCase):
    @given(
        batch_size=st.integers(0, 100),
        max_length=st.integers(0, 50),
        dtype=st.sampled_from([torch.int32, torch.int64]),
        device=cpu_and_maybe_gpu(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_repeat_arange_correctness(
        self,
        batch_size: int,
        max_length: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Test that our implementation matches the reference PyTorch implementation."""
        lengths = torch.randint(
            low=0,
            high=max_length + 1,
            size=(batch_size,),
            dtype=dtype,
            device=device,
        )

        # Our optimized implementation (1 kernel)
        result = torch.ops.fbgemm.repeat_arange(lengths)

        # Reference implementation (4+ kernels)
        if batch_size > 0 and lengths.sum().item() > 0:
            expected = repeat_arange_ref(lengths)
            torch.testing.assert_close(result, expected)
        else:
            # Empty case
            self.assertEqual(result.numel(), 0)
            self.assertEqual(result.device, lengths.device)
            self.assertEqual(result.dtype, lengths.dtype)

        torch.library.opcheck(
            torch.ops.fbgemm.repeat_arange,
            (lengths,),
        )

    def test_repeat_arange_example(self) -> None:
        """Test the example from the docstring."""
        lengths = torch.tensor([3, 5, 2])
        result = torch.ops.fbgemm.repeat_arange(lengths)
        expected = torch.tensor([0, 1, 2, 0, 1, 2, 3, 4, 0, 1])
        torch.testing.assert_close(result, expected)

    @given(
        batch_size=st.integers(1, 20),
        dtype=st.sampled_from([torch.int32, torch.int64]),
        device=cpu_and_maybe_gpu(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=10, deadline=None)
    def test_repeat_arange_single_length(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Test with uniform lengths."""
        length = 10
        lengths = torch.full((batch_size,), length, dtype=dtype, device=device)

        result = torch.ops.fbgemm.repeat_arange(lengths)
        expected = repeat_arange_ref(lengths)

        torch.testing.assert_close(result, expected)
        self.assertEqual(result.shape[0], batch_size * length)

    def test_repeat_arange_empty_batch(self) -> None:
        """Test with empty batch."""
        lengths = torch.tensor([], dtype=torch.int64)
        result = torch.ops.fbgemm.repeat_arange(lengths)

        self.assertEqual(result.numel(), 0)
        self.assertEqual(result.dtype, torch.int64)

    def test_repeat_arange_zero_lengths(self) -> None:
        """Test with all zero lengths."""
        lengths = torch.zeros(10, dtype=torch.int64)
        result = torch.ops.fbgemm.repeat_arange(lengths)

        self.assertEqual(result.numel(), 0)
        self.assertEqual(result.dtype, torch.int64)

    def test_repeat_arange_mixed_zero_nonzero(self) -> None:
        """Test with mix of zero and non-zero lengths."""
        lengths = torch.tensor([0, 3, 0, 5, 0, 2])
        result = torch.ops.fbgemm.repeat_arange(lengths)
        expected = torch.tensor([0, 1, 2, 0, 1, 2, 3, 4, 0, 1])

        torch.testing.assert_close(result, expected)

    @unittest.skipIf(*gpu_unavailable)
    @given(
        batch_size=st.integers(1, 1000),
        max_length=st.integers(1, 100),
        dtype=st.sampled_from([torch.int32, torch.int64]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=10, deadline=None)
    def test_repeat_arange_large_batch(
        self,
        batch_size: int,
        max_length: int,
        dtype: torch.dtype,
    ) -> None:
        """Test with large batch sizes on GPU."""
        device = torch.accelerator.current_accelerator()
        lengths = torch.randint(
            low=0,
            high=max_length + 1,
            size=(batch_size,),
            dtype=dtype,
            device=device,
        )

        result = torch.ops.fbgemm.repeat_arange(lengths)
        expected = repeat_arange_ref(lengths)

        torch.testing.assert_close(result, expected)

    @given(
        batch_size=st.integers(0, 100),
        max_length=st.integers(0, 50),
        dtype=st.sampled_from([torch.int32, torch.int64]),
        device=st.sampled_from([torch.device("meta")]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_repeat_arange_meta_backend(
        self,
        batch_size: int,
        max_length: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Test meta tensor support for torch.compile."""
        lengths = torch.randint(
            low=0,
            high=max_length + 1,
            size=(batch_size,),
            dtype=dtype,
            device=device,
        )

        result = torch.ops.fbgemm.repeat_arange(lengths)

        # For meta tensors with data-dependent output size, we return a placeholder
        # tensor with size 0. This is expected behavior since meta tensors cannot
        # represent data-dependent shapes.
        self.assertEqual(result.device, device)
        self.assertEqual(result.dtype, dtype)
        self.assertEqual(result.numel(), 0)  # Placeholder size for meta tensors

    @optests.dontGenerateOpCheckTests("tests that call torch.compile are slow")
    @unittest.skipIf(*symint_vector_unsupported())
    @given(
        batch_size=st.integers(2, 20),
        max_length=st.integers(2, 30),
        dtype=st.sampled_from([torch.int32, torch.int64]),
        device=cpu_and_maybe_gpu(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=10, deadline=None)
    def test_repeat_arange_dynamic_shape(
        self,
        batch_size: int,
        max_length: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Test with torch.compile and dynamic shapes."""
        # Start a fresh compile for each parameter of the test case
        torch._dynamo.reset()

        lengths = torch.randint(
            low=1,  # Avoid 0/1 which get specialized
            high=max_length + 1,
            size=(batch_size,),
            dtype=dtype,
            device=device,
        )

        # Mark as dynamic
        torch._dynamo.mark_dynamic(lengths, 0)

        def repeat_arange_fn(lengths: torch.Tensor) -> torch.Tensor:
            return torch.ops.fbgemm.repeat_arange(lengths)

        # Test with compiled version
        result = repeat_arange_fn(lengths)
        expected = repeat_arange_ref(lengths)

        torch.testing.assert_close(result, expected)

    def test_repeat_arange_dtypes(self) -> None:
        """Test with different dtypes."""
        for dtype in [torch.int32, torch.int64]:
            lengths = torch.tensor([3, 5, 2], dtype=dtype)
            result = torch.ops.fbgemm.repeat_arange(lengths)
            expected = torch.tensor([0, 1, 2, 0, 1, 2, 3, 4, 0, 1], dtype=dtype)
            torch.testing.assert_close(result, expected)
            self.assertEqual(result.dtype, dtype)

    @unittest.skipIf(*gpu_unavailable)
    def test_repeat_arange_cpu_gpu_consistency(self) -> None:
        """Test that CPU and GPU implementations produce same results."""
        lengths_cpu = torch.tensor([3, 5, 2, 7, 1, 0, 4])
        lengths_gpu = lengths_cpu.cuda()

        result_cpu = torch.ops.fbgemm.repeat_arange(lengths_cpu)
        result_gpu = torch.ops.fbgemm.repeat_arange(lengths_gpu)

        torch.testing.assert_close(result_cpu, result_gpu.cpu())

    def test_repeat_arange_large_single_length(self) -> None:
        """Test with a single very large length."""
        lengths = torch.tensor([10000])
        result = torch.ops.fbgemm.repeat_arange(lengths)

        # Check shape
        self.assertEqual(result.shape[0], 10000)

        # Check first and last few values
        self.assertEqual(result[0].item(), 0)
        self.assertEqual(result[9999].item(), 9999)

        # Spot check some values
        for i in [0, 100, 500, 1000, 5000, 9999]:
            self.assertEqual(result[i].item(), i)


if __name__ == "__main__":
    unittest.main()
