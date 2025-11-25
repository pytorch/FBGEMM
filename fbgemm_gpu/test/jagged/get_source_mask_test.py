#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[56]

# buck test @mode/opt deeplearning/fbgemm/fbgemm_gpu/test/jagged:get_source_mask
import unittest

import hypothesis.strategies as st
import torch
from hypothesis import given, settings, Verbosity

from .common import additional_decorators, open_source

if open_source:
    # pyre-ignore[21]
    from test_utils import gpu_unavailable, optests
else:
    from fbgemm_gpu.test.test_utils import gpu_unavailable, optests


def get_source_mask_reference(
    num_sources: torch.Tensor, num_targets: torch.Tensor
) -> torch.Tensor:
    """Reference implementation using PyTorch ops."""
    batch_size = num_sources.shape[0]
    device = num_sources.device
    skeleton = (
        torch.tensor([[True, False]], device=device).expand(batch_size, 2).flatten()
    )
    repeats = torch.stack([num_sources, num_targets], dim=1).flatten()
    return skeleton.repeat_interleave(repeats)


@optests.generate_opcheck_tests(additional_decorators=additional_decorators)
class GetSourceMaskTest(unittest.TestCase):
    @unittest.skipIf(*gpu_unavailable)
    @given(
        batch_size=st.integers(min_value=1, max_value=100),
        max_sources=st.integers(min_value=1, max_value=20),
        max_targets=st.integers(min_value=1, max_value=20),
        dtype=st.sampled_from([torch.int32, torch.int64]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_get_source_mask_basic(
        self,
        batch_size: int,
        max_sources: int,
        max_targets: int,
        dtype: torch.dtype,
    ) -> None:
        device = torch.accelerator.current_accelerator()

        num_sources = torch.randint(
            0, max_sources + 1, (batch_size,), dtype=dtype, device=device
        )
        num_targets = torch.randint(
            0, max_targets + 1, (batch_size,), dtype=dtype, device=device
        )

        # Compute output_size for C++ op
        output_size = int((num_sources + num_targets).sum().item())
        result = torch.ops.fbgemm.get_source_mask(num_sources, num_targets, output_size)
        expected = get_source_mask_reference(num_sources, num_targets)

        self.assertTrue(torch.equal(result, expected))
        self.assertEqual(result.dtype, torch.bool)

    @unittest.skipIf(*gpu_unavailable)
    def test_get_source_mask_simple_example(self) -> None:
        device = torch.accelerator.current_accelerator()

        num_sources = torch.tensor([2, 3], dtype=torch.int64, device=device)
        num_targets = torch.tensor([1, 2], dtype=torch.int64, device=device)

        # Test with C++ op, must provide output_size
        output_size = int((num_sources + num_targets).sum().item())
        result = torch.ops.fbgemm.get_source_mask(num_sources, num_targets, output_size)

        expected = torch.tensor(
            [True, True, False, True, True, True, False, False],
            dtype=torch.bool,
            device=device,
        )

        self.assertTrue(torch.equal(result, expected))

        # Test with output_size parameter
        result_with_size = torch.ops.fbgemm.get_source_mask(num_sources, num_targets, 8)
        self.assertTrue(torch.equal(result_with_size, expected))

    @unittest.skipIf(*gpu_unavailable)
    def test_get_source_mask_single_batch(self) -> None:
        device = torch.accelerator.current_accelerator()

        num_sources = torch.tensor([5], dtype=torch.int64, device=device)
        num_targets = torch.tensor([3], dtype=torch.int64, device=device)

        # Must provide output_size for C++ op
        result = torch.ops.fbgemm.get_source_mask(num_sources, num_targets, 8)

        expected = torch.tensor(
            [True, True, True, True, True, False, False, False],
            dtype=torch.bool,
            device=device,
        )

        self.assertTrue(torch.equal(result, expected))

    @unittest.skipIf(*gpu_unavailable)
    def test_get_source_mask_zeros(self) -> None:
        device = torch.accelerator.current_accelerator()

        num_sources = torch.tensor([0, 2, 0], dtype=torch.int64, device=device)
        num_targets = torch.tensor([3, 0, 1], dtype=torch.int64, device=device)

        # Must provide output_size for C++ op
        output_size = int((num_sources + num_targets).sum().item())
        result = torch.ops.fbgemm.get_source_mask(num_sources, num_targets, output_size)

        expected = torch.tensor(
            [False, False, False, True, True, False],
            dtype=torch.bool,
            device=device,
        )

        self.assertTrue(torch.equal(result, expected))

    @unittest.skipIf(*gpu_unavailable)
    @given(
        batch_size=st.integers(min_value=10, max_value=1000),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=5, deadline=None)
    def test_get_source_mask_large_batch(self, batch_size: int) -> None:
        device = torch.accelerator.current_accelerator()

        num_sources = torch.randint(
            0, 50, (batch_size,), dtype=torch.int64, device=device
        )
        num_targets = torch.randint(
            0, 50, (batch_size,), dtype=torch.int64, device=device
        )

        # Must provide output_size for C++ op
        output_size = int((num_sources + num_targets).sum().item())
        result = torch.ops.fbgemm.get_source_mask(num_sources, num_targets, output_size)
        expected = get_source_mask_reference(num_sources, num_targets)

        self.assertTrue(torch.equal(result, expected))

    @unittest.skipIf(*gpu_unavailable)
    def test_get_source_mask_all_zeros(self) -> None:
        device = torch.accelerator.current_accelerator()

        num_sources = torch.zeros(5, dtype=torch.int64, device=device)
        num_targets = torch.zeros(5, dtype=torch.int64, device=device)

        # Must provide output_size for C++ op (all zeros = output_size 0)
        result = torch.ops.fbgemm.get_source_mask(num_sources, num_targets, 0)

        expected = torch.tensor([], dtype=torch.bool, device=device)

        self.assertTrue(torch.equal(result, expected))
        self.assertEqual(result.numel(), 0)

    def test_get_source_mask_meta_output_shape(self) -> None:
        """Test that meta implementation returns correct output shape when output_size provided."""
        num_sources = torch.tensor([2, 3, 1], dtype=torch.int64, device="meta")
        num_targets = torch.tensor([1, 2, 3], dtype=torch.int64, device="meta")

        # Expected total size: (2+1) + (3+2) + (1+3) = 3 + 5 + 4 = 12
        result = torch.ops.fbgemm.get_source_mask(num_sources, num_targets, 12)

        self.assertEqual(result.device.type, "meta")
        self.assertEqual(result.dtype, torch.bool)
        self.assertEqual(result.dim(), 1)
        self.assertEqual(result.shape[0], 12)

    def test_get_source_mask_meta_batch_size_mismatch(self) -> None:
        """Test that meta implementation validates batch size mismatch."""
        num_sources = torch.tensor([2, 3], dtype=torch.int64, device="meta")
        num_targets = torch.tensor([1, 2, 3], dtype=torch.int64, device="meta")

        with self.assertRaisesRegex(
            RuntimeError, "num_sources and num_targets must have the same batch size"
        ):
            torch.ops.fbgemm.get_source_mask(num_sources, num_targets, 0)

    def test_get_source_mask_meta_empty_batch(self) -> None:
        """Test meta implementation with empty batch."""
        num_sources = torch.tensor([], dtype=torch.int64, device="meta")
        num_targets = torch.tensor([], dtype=torch.int64, device="meta")

        # With empty batch and output_size=0, meta implementation should work
        result = torch.ops.fbgemm.get_source_mask(num_sources, num_targets, 0)

        self.assertEqual(result.device.type, "meta")
        self.assertEqual(result.dtype, torch.bool)
        self.assertEqual(result.dim(), 1)
        self.assertEqual(result.shape[0], 0)

    def test_get_source_mask_meta_single_batch(self) -> None:
        """Test meta implementation with single batch element."""
        num_sources = torch.tensor([5], dtype=torch.int64, device="meta")
        num_targets = torch.tensor([3], dtype=torch.int64, device="meta")

        # Must provide output_size for meta tensors
        result = torch.ops.fbgemm.get_source_mask(num_sources, num_targets, 8)

        # Expected total size: 5 + 3 = 8
        self.assertEqual(result.device.type, "meta")
        self.assertEqual(result.dtype, torch.bool)
        self.assertEqual(result.dim(), 1)
        self.assertEqual(result.shape[0], 8)

    def test_get_source_mask_meta_dtype_preservation(self) -> None:
        """Test that meta implementation works with different input dtypes."""
        for dtype in [torch.int32, torch.int64]:
            num_sources = torch.tensor([2, 3], dtype=dtype, device="meta")
            num_targets = torch.tensor([1, 2], dtype=dtype, device="meta")

            # Must provide output_size for meta tensors: (2+1) + (3+2) = 8
            result = torch.ops.fbgemm.get_source_mask(num_sources, num_targets, 8)

            # Output should always be bool regardless of input dtype
            self.assertEqual(result.dtype, torch.bool)
            self.assertEqual(result.device.type, "meta")

    # Tests for Python wrapper from sparse_ops.py
    @unittest.skipIf(*gpu_unavailable)
    def test_python_wrapper_basic_functionality(self) -> None:
        """Test that Python wrapper computes output_size automatically."""
        # Import the Python wrapper
        from fbgemm_gpu.sparse_ops import get_source_mask

        device = torch.accelerator.current_accelerator()
        num_sources = torch.tensor([2, 3], dtype=torch.int64, device=device)
        num_targets = torch.tensor([1, 2], dtype=torch.int64, device=device)

        # Call without output_size - wrapper should compute it
        result = get_source_mask(num_sources, num_targets)

        expected = torch.tensor(
            [True, True, False, True, True, True, False, False],
            dtype=torch.bool,
            device=device,
        )
        self.assertTrue(torch.equal(result, expected))

    @unittest.skipIf(*gpu_unavailable)
    def test_python_wrapper_with_explicit_output_size(self) -> None:
        """Test that Python wrapper works when output_size is provided."""
        from fbgemm_gpu.sparse_ops import get_source_mask

        device = torch.accelerator.current_accelerator()
        num_sources = torch.tensor([2, 3], dtype=torch.int64, device=device)
        num_targets = torch.tensor([1, 2], dtype=torch.int64, device=device)

        # Call with explicit output_size
        result = get_source_mask(num_sources, num_targets, output_size=8)

        expected = torch.tensor(
            [True, True, False, True, True, True, False, False],
            dtype=torch.bool,
            device=device,
        )
        self.assertTrue(torch.equal(result, expected))

    def test_python_wrapper_meta_with_output_size(self) -> None:
        """Test that Python wrapper works with meta tensors when output_size provided."""
        from fbgemm_gpu.sparse_ops import get_source_mask

        num_sources = torch.tensor([2, 3], dtype=torch.int64, device="meta")
        num_targets = torch.tensor([1, 2], dtype=torch.int64, device="meta")

        # Call with output_size for meta tensors
        result = get_source_mask(num_sources, num_targets, output_size=8)

        self.assertEqual(result.device.type, "meta")
        self.assertEqual(result.dtype, torch.bool)
        self.assertEqual(result.shape[0], 8)

    @unittest.skipIf(*gpu_unavailable)
    def test_python_wrapper_fx_tracing_and_jit_scripting(self) -> None:
        """
        Test that the Python wrapper from sparse_ops (used in test_python_wrapper_basic_functionality)
        works under fx.tracing and jit.scripting when output_size is explicitly provided.

        Note: fx.tracing and jit.scripting cannot handle dynamic .item() calls,
        so we must provide output_size explicitly to make the function traceable/scriptable.
        """
        from fbgemm_gpu.sparse_ops import get_source_mask

        device = torch.accelerator.current_accelerator()

        # Test data
        num_sources = torch.tensor([2, 3], dtype=torch.int64, device=device)
        num_targets = torch.tensor([1, 2], dtype=torch.int64, device=device)

        # Expected output
        expected = torch.tensor(
            [True, True, False, True, True, True, False, False],
            dtype=torch.bool,
            device=device,
        )

        # Test 1: FX Tracing with explicit output_size
        # FX tracing requires output_size to be explicitly provided to avoid .item() calls
        traced_model = torch.fx.symbolic_trace(
            lambda src, tgt: get_source_mask(src, tgt, output_size=8)
        )
        fx_result = traced_model(num_sources, num_targets)
        self.assertTrue(torch.equal(fx_result, expected))

        # Test 2: JIT Scripting with explicit output_size
        # JIT scripting also requires explicit output_size
        @torch.jit.script
        def scripted_wrapper(src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
            return get_source_mask(src, tgt, output_size=8)

        jit_result = scripted_wrapper(num_sources, num_targets)
        self.assertTrue(torch.equal(jit_result, expected))


if __name__ == "__main__":
    unittest.main()
