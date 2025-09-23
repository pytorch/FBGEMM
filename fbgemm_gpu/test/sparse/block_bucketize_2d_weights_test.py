#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# pyre-ignore-all-errors[56]

import unittest

import hypothesis.strategies as st
import torch
from hypothesis import given, settings, Verbosity

from .common import extend_test_class, open_source

if open_source:
    # pyre-ignore[21]
    from test_utils import gpu_available, skipIfRocm
else:
    from fbgemm_gpu.test.test_utils import gpu_available, skipIfRocm

ROCM_FAILURE_MESSAGE = "Test is causing HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION"


class BlockBucketize2DWeightsTest(unittest.TestCase):
    def validate_out_of_order_output(
        self,
        expected: torch.Tensor,
        actual: torch.Tensor,
        lengths: torch.Tensor,
        is_int: bool = True,
    ) -> None:
        self.assertEqual(actual.numel(), expected.numel())
        self.assertEqual(torch.sum(lengths).item(), actual.numel())
        expected_list = expected.tolist()
        actual_list = actual.tolist()
        offset_list = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths).tolist()

        for i in range(len(offset_list) - 1):
            expected_sample = sorted(expected_list[offset_list[i] : offset_list[i + 1]])
            actual_sample = sorted(actual_list[offset_list[i] : offset_list[i + 1]])
            if is_int:
                self.assertEqual(expected_sample, actual_sample)
            else:
                for left, right in zip(expected_sample, actual_sample):
                    self.assertAlmostEqual(left, right)
        return

    @skipIfRocm(ROCM_FAILURE_MESSAGE)
    @given(
        index_type=st.sampled_from([torch.int, torch.long]),
        bucketize_pos=st.booleans(),
        sequence=st.booleans(),
        weights_dim=st.sampled_from([2, 3, 4]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=16, deadline=None)
    def test_block_bucketize_sparse_features_2d_weights(
        self,
        index_type: type[torch.dtype],
        bucketize_pos: bool,
        sequence: bool,
        weights_dim: int,
    ) -> None:
        """Test block bucketize sparse features with 2D weights.

        Tests both CPU and GPU implementations for 2D weights.
        """
        # pyre-ignore [6]
        lengths = torch.tensor([0, 2, 1, 3, 2, 3, 3, 1], dtype=index_type)
        indices = torch.tensor(
            [3, 4, 15, 11, 28, 29, 1, 10, 11, 12, 13, 11, 22, 20, 20],
            # pyre-ignore [6]
            dtype=index_type,
        )
        # Create 2D weights with shape [indices.numel(), weights_dim]
        weights = torch.rand(
            (indices.numel(), weights_dim),
            dtype=torch.float,
        )
        # pyre-ignore [6]
        block_sizes = torch.tensor([5, 15, 10, 20], dtype=index_type)
        my_size = 2

        new_lengths_ref = torch.tensor(
            [0, 2, 0, 1, 1, 0, 1, 0, 0, 0, 1, 2, 1, 3, 2, 1],
            # pyre-ignore [6]
            dtype=index_type,
        )
        new_indices_ref = torch.tensor(
            [3, 4, 11, 1, 11, 0, 13, 14, 0, 1, 2, 3, 2, 0, 0],
            # pyre-ignore [6]
            dtype=index_type,
        )

        # CPU implementation
        (
            new_lengths_cpu,
            new_indices_cpu,
            new_weights_cpu,
            new_pos_cpu,
            unbucketize_permute,
        ) = torch.ops.fbgemm.block_bucketize_sparse_features_2d_weights(
            lengths,
            indices,
            bucketize_pos,
            sequence,
            block_sizes,
            my_size,
            weights,
            weights_dim,
        )

        # Verify output shapes and types
        torch.testing.assert_close(new_lengths_cpu, new_lengths_ref, rtol=0, atol=0)
        torch.testing.assert_close(new_indices_cpu, new_indices_ref, rtol=0, atol=0)
        self.assertEqual(new_weights_cpu.shape, (indices.numel(), weights_dim))
        self.assertEqual(new_weights_cpu.dtype, weights.dtype)

        # Test GPU implementation if available
        if gpu_available:
            (
                new_lengths_gpu,
                new_indices_gpu,
                new_weights_gpu,
                new_pos_gpu,
                unbucketize_permute_gpu,
            ) = torch.ops.fbgemm.block_bucketize_sparse_features_2d_weights(
                lengths.cuda(),
                indices.cuda(),
                bucketize_pos,
                sequence,
                block_sizes.cuda(),
                my_size,
                weights.cuda(),
                weights_dim,
            )

            torch.testing.assert_close(
                new_lengths_gpu.cpu(), new_lengths_ref, rtol=0, atol=0
            )

            # Verify output shapes and types
            self.assertEqual(new_weights_gpu.shape, (indices.numel(), weights_dim))
            self.assertEqual(new_weights_gpu.dtype, weights.dtype)

            if sequence:
                torch.testing.assert_close(
                    new_indices_gpu.cpu(), new_indices_ref, rtol=0, atol=0
                )

                # For sequence mode, weights should be in the same order
                for d in range(weights_dim):
                    weights_cpu_d = new_weights_cpu[:, d]
                    weights_gpu_d = new_weights_gpu.cpu()[:, d]
                    torch.testing.assert_close(weights_gpu_d, weights_cpu_d)

                if bucketize_pos:
                    torch.testing.assert_close(new_pos_gpu.cpu(), new_pos_cpu)
            else:
                # For non-sequence mode, indices may be in different order
                self.validate_out_of_order_output(
                    new_indices_ref, new_indices_gpu.cpu(), new_lengths_ref
                )

                # For non-sequence mode, we can't directly compare weights
                # but we can verify that all dimensions are preserved
                self.assertEqual(new_weights_gpu.shape[1], weights_dim)

    @skipIfRocm(ROCM_FAILURE_MESSAGE)
    @given(
        index_type=st.sampled_from([torch.int, torch.long]),
        bucketize_pos=st.booleans(),
        sequence=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=16, deadline=None)
    def test_block_bucketize_sparse_features_2d_weights_vs_original(
        self,
        index_type: type[torch.dtype],
        bucketize_pos: bool,
        sequence: bool,
    ) -> None:
        """Test that block_bucketize_sparse_features_2d_weights with weights_dim=1
        produces the same results as block_bucketize_sparse_features.
        """
        # pyre-ignore [6]
        lengths = torch.tensor([0, 2, 1, 3, 2, 3, 3, 1], dtype=index_type)
        indices = torch.tensor(
            [3, 4, 15, 11, 28, 29, 1, 10, 11, 12, 13, 11, 22, 20, 20],
            # pyre-ignore [6]
            dtype=index_type,
        )

        # Create 1D weights for original operator
        weights_1d = torch.rand(
            indices.numel(),
            dtype=torch.float,
        )

        # Create 2D weights with only one column for new operator
        weights_2d = weights_1d.clone().view(-1, 1)

        # pyre-ignore [6]
        block_sizes = torch.tensor([5, 15, 10, 20], dtype=index_type)
        my_size = 2

        # Call original operator
        (
            lengths_1d,
            indices_1d,
            weights_1d_out,
            pos_1d,
            unbucketize_permute_1d,
        ) = torch.ops.fbgemm.block_bucketize_sparse_features(
            lengths,
            indices,
            bucketize_pos,
            sequence,
            block_sizes,
            my_size,
            weights_1d,
        )

        # Call new operator with weights_dim=1
        (
            lengths_2d,
            indices_2d,
            weights_2d_out,
            pos_2d,
            unbucketize_permute_2d,
        ) = torch.ops.fbgemm.block_bucketize_sparse_features_2d_weights(
            lengths,
            indices,
            bucketize_pos,
            sequence,
            block_sizes,
            my_size,
            weights_2d,
            weights_dim=1,
        )

        # Verify outputs are the same
        torch.testing.assert_close(lengths_1d, lengths_2d, rtol=0, atol=0)
        torch.testing.assert_close(indices_1d, indices_2d, rtol=0, atol=0)

        # Compare weights - need to reshape 2D weights to 1D for comparison
        torch.testing.assert_close(weights_1d_out, weights_2d_out.view(-1))

        if bucketize_pos:
            torch.testing.assert_close(pos_1d, pos_2d, rtol=0, atol=0)

        if unbucketize_permute_1d is not None and unbucketize_permute_2d is not None:
            torch.testing.assert_close(
                unbucketize_permute_1d, unbucketize_permute_2d, rtol=0, atol=0
            )

        # Test on GPU if available
        if gpu_available:
            # Call original operator on GPU
            (
                lengths_1d_gpu,
                indices_1d_gpu,
                weights_1d_out_gpu,
                pos_1d_gpu,
                unbucketize_permute_1d_gpu,
            ) = torch.ops.fbgemm.block_bucketize_sparse_features(
                lengths.cuda(),
                indices.cuda(),
                bucketize_pos,
                sequence,
                block_sizes.cuda(),
                my_size,
                weights_1d.cuda(),
            )

            # Call new operator on GPU
            (
                lengths_2d_gpu,
                indices_2d_gpu,
                weights_2d_out_gpu,
                pos_2d_gpu,
                unbucketize_permute_2d_gpu,
            ) = torch.ops.fbgemm.block_bucketize_sparse_features_2d_weights(
                lengths.cuda(),
                indices.cuda(),
                bucketize_pos,
                sequence,
                block_sizes.cuda(),
                my_size,
                weights_2d.cuda(),
                weights_dim=1,
            )

            # Verify GPU outputs are the same
            torch.testing.assert_close(lengths_1d_gpu, lengths_2d_gpu, rtol=0, atol=0)

            if sequence:
                torch.testing.assert_close(
                    indices_1d_gpu, indices_2d_gpu, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    weights_1d_out_gpu, weights_2d_out_gpu.view(-1)
                )
                if bucketize_pos:
                    torch.testing.assert_close(pos_1d_gpu, pos_2d_gpu, rtol=0, atol=0)
            else:
                # For non-sequence mode, indices may be in different order
                # but should contain the same values
                self.validate_out_of_order_output(
                    indices_1d_gpu, indices_2d_gpu, lengths_1d_gpu
                )

                # For weights, we need to compare by sample since order may differ
                self.validate_out_of_order_output(
                    weights_1d_out_gpu,
                    weights_2d_out_gpu.view(-1),
                    lengths_1d_gpu,
                    is_int=False,
                )

                if bucketize_pos:
                    self.validate_out_of_order_output(
                        pos_1d_gpu, pos_2d_gpu, lengths_1d_gpu
                    )

    @skipIfRocm(ROCM_FAILURE_MESSAGE)
    @given(
        index_type=st.sampled_from([torch.int, torch.long]),
        bucketize_pos=st.booleans(),
        weights_dtype=st.sampled_from([torch.float, torch.double]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=16, deadline=None)
    def test_block_bucketize_sparse_features_2d_weights_pooled_vs_sequence(
        self,
        index_type: type[torch.dtype],
        bucketize_pos: bool,
        weights_dtype: torch.dtype,
    ) -> None:
        """Test block bucketize sparse features with 2D weights in both pooled and sequence modes.

        This test explicitly compares the behavior between pooled (sequence=False) and
        sequence (sequence=True) modes with the same inputs.
        """
        weights_dim = 3
        # pyre-ignore [6]
        lengths = torch.tensor([0, 2, 1, 3, 2, 3, 3, 1], dtype=index_type)
        indices = torch.tensor(
            [3, 4, 15, 11, 28, 29, 1, 10, 11, 12, 13, 11, 22, 20, 20],
            # pyre-ignore [6]
            dtype=index_type,
        )

        # Create 2D weights with specified dtype
        weights = torch.rand(
            (indices.numel(), weights_dim),
            dtype=weights_dtype,
        )

        # pyre-ignore [6]
        block_sizes = torch.tensor([5, 15, 10, 20], dtype=index_type)
        my_size = 2

        # Run with sequence=True (order matters)
        (
            lengths_seq,
            indices_seq,
            weights_seq,
            pos_seq,
            unbucketize_permute_seq,
        ) = torch.ops.fbgemm.block_bucketize_sparse_features_2d_weights(
            lengths,
            indices,
            bucketize_pos,
            True,  # sequence=True
            block_sizes,
            my_size,
            weights,
            weights_dim,
        )

        # Run with sequence=False (order doesn't matter)
        (
            lengths_pooled,
            indices_pooled,
            weights_pooled,
            pos_pooled,
            unbucketize_permute_pooled,
        ) = torch.ops.fbgemm.block_bucketize_sparse_features_2d_weights(
            lengths,
            indices,
            bucketize_pos,
            False,  # sequence=False
            block_sizes,
            my_size,
            weights,
            weights_dim,
        )

        # Verify lengths are the same regardless of sequence mode
        torch.testing.assert_close(lengths_seq, lengths_pooled, rtol=0, atol=0)

        # Verify weights have the same shape and dtype
        self.assertEqual(weights_seq.shape, weights_pooled.shape)
        self.assertEqual(weights_seq.dtype, weights_dtype)
        self.assertEqual(weights_pooled.dtype, weights_dtype)

        # In pooled mode, indices may be in different order but should contain the same values
        self.validate_out_of_order_output(indices_seq, indices_pooled, lengths_seq)

        # For weights, we need to compare by sample since order may differ
        for d in range(weights_dim):
            self.validate_out_of_order_output(
                weights_seq[:, d], weights_pooled[:, d], lengths_seq, is_int=False
            )

        if bucketize_pos:
            self.assertTrue(pos_seq is not None)
            self.assertTrue(pos_pooled is not None)
            self.validate_out_of_order_output(pos_seq, pos_pooled, lengths_seq)

        # Test on GPU if available
        if gpu_available:
            # Run with sequence=True on GPU
            (
                lengths_seq_gpu,
                indices_seq_gpu,
                weights_seq_gpu,
                pos_seq_gpu,
                unbucketize_permute_seq_gpu,
            ) = torch.ops.fbgemm.block_bucketize_sparse_features_2d_weights(
                lengths.cuda(),
                indices.cuda(),
                bucketize_pos,
                True,  # sequence=True
                block_sizes.cuda(),
                my_size,
                weights.cuda(),
                weights_dim,
            )

            # Run with sequence=False on GPU
            (
                lengths_pooled_gpu,
                indices_pooled_gpu,
                weights_pooled_gpu,
                pos_pooled_gpu,
                unbucketize_permute_pooled_gpu,
            ) = torch.ops.fbgemm.block_bucketize_sparse_features_2d_weights(
                lengths.cuda(),
                indices.cuda(),
                bucketize_pos,
                False,  # sequence=False
                block_sizes.cuda(),
                my_size,
                weights.cuda(),
                weights_dim,
            )

            # Verify GPU results match between modes
            torch.testing.assert_close(
                lengths_seq_gpu, lengths_pooled_gpu, rtol=0, atol=0
            )

            # Verify weights have the same shape and dtype
            self.assertEqual(weights_seq_gpu.shape, weights_pooled_gpu.shape)
            self.assertEqual(weights_seq_gpu.dtype, weights_dtype)
            self.assertEqual(weights_pooled_gpu.dtype, weights_dtype)

            # In pooled mode, indices may be in different order
            self.validate_out_of_order_output(
                indices_seq_gpu.cpu(), indices_pooled_gpu.cpu(), lengths_seq_gpu.cpu()
            )

    @skipIfRocm(ROCM_FAILURE_MESSAGE)
    @given(
        index_type=st.sampled_from([torch.int, torch.long]),
        bucketize_pos=st.booleans(),
        sequence=st.booleans(),
        keep_orig_idx=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=16, deadline=None)
    def test_block_bucketize_sparse_features_2d_weights_keep_orig_idx(
        self,
        index_type: type[torch.dtype],
        bucketize_pos: bool,
        sequence: bool,
        keep_orig_idx: bool,
    ) -> None:
        """Test block bucketize sparse features with 2D weights and keep_orig_idx parameter."""
        weights_dim = 2
        # pyre-ignore [6]
        lengths = torch.tensor([0, 2, 1, 3, 2, 3, 3, 1], dtype=index_type)
        indices = torch.tensor(
            [3, 4, 15, 11, 28, 29, 1, 10, 11, 12, 13, 11, 22, 20, 20],
            # pyre-ignore [6]
            dtype=index_type,
        )

        # Create 2D weights
        weights = torch.rand(
            (indices.numel(), weights_dim),
            dtype=torch.float,
        )

        # pyre-ignore [6]
        block_sizes = torch.tensor([5, 15, 10, 20], dtype=index_type)
        my_size = 2

        # Run with keep_orig_idx=True/False
        (
            lengths_out,
            indices_out,
            weights_out,
            pos_out,
            unbucketize_permute_out,
        ) = torch.ops.fbgemm.block_bucketize_sparse_features_2d_weights(
            lengths,
            indices,
            bucketize_pos,
            sequence,
            block_sizes,
            my_size,
            weights,
            weights_dim,
            keep_orig_idx=keep_orig_idx,
        )

        # Verify output shapes
        self.assertEqual(weights_out.shape, (indices.numel(), weights_dim))

        # If keep_orig_idx is True, indices should match the original indices
        if keep_orig_idx and sequence:
            # In sequence mode with keep_orig_idx=True, indices should be preserved exactly
            # We need to use the unbucketize_permute to reorder the indices back to original order
            if unbucketize_permute_out is not None:
                reordered_indices = torch.zeros_like(indices_out)
                for i in range(indices.numel()):
                    reordered_indices[i] = indices_out[unbucketize_permute_out[i]]

                # Check that reordered indices match original indices
                torch.testing.assert_close(reordered_indices, indices, rtol=0, atol=0)

        # Test on GPU if available
        if gpu_available:
            (
                lengths_out_gpu,
                indices_out_gpu,
                weights_out_gpu,
                pos_out_gpu,
                unbucketize_permute_out_gpu,
            ) = torch.ops.fbgemm.block_bucketize_sparse_features_2d_weights(
                lengths.cuda(),
                indices.cuda(),
                bucketize_pos,
                sequence,
                block_sizes.cuda(),
                my_size,
                weights.cuda(),
                weights_dim,
                keep_orig_idx=keep_orig_idx,
            )

            # Verify output shapes
            self.assertEqual(weights_out_gpu.shape, (indices.numel(), weights_dim))

            # Check that GPU results match CPU results
            torch.testing.assert_close(
                lengths_out_gpu.cpu(), lengths_out, rtol=0, atol=0
            )

            if sequence:
                torch.testing.assert_close(
                    indices_out_gpu.cpu(), indices_out, rtol=0, atol=0
                )

                for d in range(weights_dim):
                    torch.testing.assert_close(
                        weights_out_gpu.cpu()[:, d], weights_out[:, d]
                    )
            else:
                # For non-sequence mode, indices may be in different order
                self.validate_out_of_order_output(
                    indices_out, indices_out_gpu.cpu(), lengths_out
                )

    @skipIfRocm(ROCM_FAILURE_MESSAGE)
    @given(
        index_type=st.sampled_from([torch.int, torch.long]),
        bucketize_pos=st.booleans(),
        sequence=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=16, deadline=None)
    def test_block_bucketize_sparse_features_2d_weights_keep_orig_idx_per_feature(
        self,
        index_type: type[torch.dtype],
        bucketize_pos: bool,
        sequence: bool,
    ) -> None:
        """Test block bucketize sparse features with 2D weights and keep_orig_idx_per_feature parameter."""
        weights_dim = 2
        # pyre-ignore [6]
        lengths = torch.tensor([0, 2, 1, 3, 2, 3, 3, 1], dtype=index_type)
        indices = torch.tensor(
            [3, 4, 15, 11, 28, 29, 1, 10, 11, 12, 13, 11, 22, 20, 20],
            # pyre-ignore [6]
            dtype=index_type,
        )

        # Create 2D weights
        weights = torch.rand(
            (indices.numel(), weights_dim),
            dtype=torch.float,
        )

        # pyre-ignore [6]
        block_sizes = torch.tensor([5, 15, 10, 20], dtype=index_type)
        my_size = 2

        # Create keep_orig_idx_per_feature tensor
        # First and third features keep original indices, others don't
        keep_orig_idx_per_feature = torch.tensor(
            [True, False, True, False], dtype=torch.bool
        )

        # Run with keep_orig_idx_per_feature
        (
            lengths_out,
            indices_out,
            weights_out,
            pos_out,
            unbucketize_permute_out,
        ) = torch.ops.fbgemm.block_bucketize_sparse_features_2d_weights(
            lengths,
            indices,
            bucketize_pos,
            sequence,
            block_sizes,
            my_size,
            weights,
            weights_dim,
            keep_orig_idx=False,  # Global setting is False
            keep_orig_idx_per_feature=keep_orig_idx_per_feature,  # Per-feature setting
        )

        # Verify output shapes
        self.assertEqual(weights_out.shape, (indices.numel(), weights_dim))

        # Test on GPU if available
        if gpu_available:
            (
                lengths_out_gpu,
                indices_out_gpu,
                weights_out_gpu,
                pos_out_gpu,
                unbucketize_permute_out_gpu,
            ) = torch.ops.fbgemm.block_bucketize_sparse_features_2d_weights(
                lengths.cuda(),
                indices.cuda(),
                bucketize_pos,
                sequence,
                block_sizes.cuda(),
                my_size,
                weights.cuda(),
                weights_dim,
                keep_orig_idx=False,
                keep_orig_idx_per_feature=keep_orig_idx_per_feature.cuda(),
            )

            # Verify output shapes
            self.assertEqual(weights_out_gpu.shape, (indices.numel(), weights_dim))

            # Check that GPU results match CPU results
            torch.testing.assert_close(
                lengths_out_gpu.cpu(), lengths_out, rtol=0, atol=0
            )

            if sequence:
                torch.testing.assert_close(
                    indices_out_gpu.cpu(), indices_out, rtol=0, atol=0
                )

                for d in range(weights_dim):
                    torch.testing.assert_close(
                        weights_out_gpu.cpu()[:, d], weights_out[:, d]
                    )
            else:
                # For non-sequence mode, indices may be in different order
                self.validate_out_of_order_output(
                    indices_out, indices_out_gpu.cpu(), lengths_out
                )

    @skipIfRocm(ROCM_FAILURE_MESSAGE)
    @given(
        index_type=st.sampled_from([torch.int, torch.long]),
        bucketize_pos=st.booleans(),
        sequence=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=16, deadline=None)
    def test_block_bucketize_sparse_features_2d_weights_total_num_blocks(
        self,
        index_type: type[torch.dtype],
        bucketize_pos: bool,
        sequence: bool,
    ) -> None:
        """Test block bucketize sparse features with 2D weights and total_num_blocks parameter."""
        weights_dim = 2
        # pyre-ignore [6]
        lengths = torch.tensor([0, 2, 1, 3, 2, 3, 3, 1], dtype=index_type)
        indices = torch.tensor(
            [3, 4, 15, 11, 28, 29, 1, 10, 11, 12, 13, 11, 22, 20, 20],
            # pyre-ignore [6]
            dtype=index_type,
        )

        # Create 2D weights
        weights = torch.rand(
            (indices.numel(), weights_dim),
            dtype=torch.float,
        )

        # pyre-ignore [6]
        block_sizes = torch.tensor([5, 15, 10, 20], dtype=index_type)
        my_size = 2

        # Create total_num_blocks tensor
        # pyre-ignore [6]
        total_num_blocks = torch.tensor([6, 6, 6, 6], dtype=index_type)

        # Run with total_num_blocks
        (
            lengths_out,
            indices_out,
            weights_out,
            pos_out,
            unbucketize_permute_out,
        ) = torch.ops.fbgemm.block_bucketize_sparse_features_2d_weights(
            lengths,
            indices,
            bucketize_pos,
            sequence,
            block_sizes,
            my_size,
            weights,
            weights_dim,
            total_num_blocks=total_num_blocks,
        )

        # Verify output shapes
        self.assertEqual(weights_out.shape, (indices.numel(), weights_dim))

        # Test on GPU if available
        if gpu_available:
            (
                lengths_out_gpu,
                indices_out_gpu,
                weights_out_gpu,
                pos_out_gpu,
                unbucketize_permute_out_gpu,
            ) = torch.ops.fbgemm.block_bucketize_sparse_features_2d_weights(
                lengths.cuda(),
                indices.cuda(),
                bucketize_pos,
                sequence,
                block_sizes.cuda(),
                my_size,
                weights.cuda(),
                weights_dim,
                total_num_blocks=total_num_blocks.cuda(),
            )

            # Verify output shapes
            self.assertEqual(weights_out_gpu.shape, (indices.numel(), weights_dim))

            # Check that GPU results match CPU results
            torch.testing.assert_close(
                lengths_out_gpu.cpu(), lengths_out, rtol=0, atol=0
            )

            if sequence:
                torch.testing.assert_close(
                    indices_out_gpu.cpu(), indices_out, rtol=0, atol=0
                )

                for d in range(weights_dim):
                    torch.testing.assert_close(
                        weights_out_gpu.cpu()[:, d], weights_out[:, d]
                    )
            else:
                # For non-sequence mode, indices may be in different order
                self.validate_out_of_order_output(
                    indices_out, indices_out_gpu.cpu(), lengths_out
                )

    @skipIfRocm(ROCM_FAILURE_MESSAGE)
    @given(
        index_type=st.sampled_from([torch.int, torch.long]),
        bucketize_pos=st.booleans(),
        sequence=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=16, deadline=None)
    def test_block_bucketize_sparse_features_2d_weights_block_bucketize_pos(
        self,
        index_type: type[torch.dtype],
        bucketize_pos: bool,
        sequence: bool,
    ) -> None:
        """Test block bucketize sparse features with 2D weights and block_bucketize_pos parameter."""
        weights_dim = 2
        # pyre-ignore [6]
        lengths = torch.tensor([2, 1, 1, 2, 0, 2], dtype=index_type)
        indices = torch.tensor(
            [1, 8, 5, 6, 7, 8, 8, 4],
            # pyre-ignore [6]
            dtype=index_type,
        )

        # Create 2D weights
        weights = torch.rand(
            (indices.numel(), weights_dim),
            dtype=torch.float,
        )

        # pyre-ignore [6]
        block_sizes = torch.tensor([5, 10, 8], dtype=index_type)
        my_size = 2

        # Create block_bucketize_pos
        block_bucketize_pos = [
            # pyre-ignore [6]
            torch.tensor([0, 2, 8], dtype=index_type),
            # pyre-ignore [6]
            torch.tensor([0, 5, 10], dtype=index_type),
            # pyre-ignore [6]
            torch.tensor([0, 7, 12], dtype=index_type),
        ]

        # Run with block_bucketize_pos
        (
            lengths_out,
            indices_out,
            weights_out,
            pos_out,
            unbucketize_permute_out,
        ) = torch.ops.fbgemm.block_bucketize_sparse_features_2d_weights(
            lengths,
            indices,
            bucketize_pos,
            sequence,
            block_sizes,
            my_size,
            weights,
            weights_dim,
            block_bucketize_pos=block_bucketize_pos,
        )

        # Verify output shapes
        self.assertEqual(weights_out.shape, (indices.numel(), weights_dim))

        # Test on GPU if available
        if gpu_available:
            block_bucketize_pos_gpu = [t.cuda() for t in block_bucketize_pos]

            (
                lengths_out_gpu,
                indices_out_gpu,
                weights_out_gpu,
                pos_out_gpu,
                unbucketize_permute_out_gpu,
            ) = torch.ops.fbgemm.block_bucketize_sparse_features_2d_weights(
                lengths.cuda(),
                indices.cuda(),
                bucketize_pos,
                sequence,
                block_sizes.cuda(),
                my_size,
                weights.cuda(),
                weights_dim,
                block_bucketize_pos=block_bucketize_pos_gpu,
            )

            # Verify output shapes
            self.assertEqual(weights_out_gpu.shape, (indices.numel(), weights_dim))

            # Check that GPU results match CPU results
            torch.testing.assert_close(
                lengths_out_gpu.cpu(), lengths_out, rtol=0, atol=0
            )

            if sequence:
                torch.testing.assert_close(
                    indices_out_gpu.cpu(), indices_out, rtol=0, atol=0
                )

                for d in range(weights_dim):
                    torch.testing.assert_close(
                        weights_out_gpu.cpu()[:, d], weights_out[:, d]
                    )
            else:
                # For non-sequence mode, indices may be in different order
                self.validate_out_of_order_output(
                    indices_out, indices_out_gpu.cpu(), lengths_out
                )

    @skipIfRocm(ROCM_FAILURE_MESSAGE)
    @given(
        index_type=st.sampled_from([torch.int, torch.long]),
        bucketize_pos=st.booleans(),
        sequence=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=16, deadline=None)
    def test_block_bucketize_sparse_features_2d_weights_with_variable_batch_sizes(
        self,
        index_type: type[torch.dtype],
        bucketize_pos: bool,
        sequence: bool,
    ) -> None:
        """Test block bucketize sparse features with 2D weights and variable batch sizes."""
        weights_dim = 3
        # pyre-ignore [6]
        lengths = torch.tensor([2, 1, 1, 2, 0, 2], dtype=index_type)
        indices = torch.tensor(
            [1, 8, 5, 6, 7, 8, 8, 4],
            # pyre-ignore [6]
            dtype=index_type,
        )
        # pyre-ignore [6]
        batch_sizes = torch.tensor([3, 1, 2], dtype=index_type)

        # Create 2D weights with shape [indices.numel(), weights_dim]
        weights = torch.rand(
            (indices.numel(), weights_dim),
            dtype=torch.float,
        )

        # pyre-ignore [6]
        block_sizes = torch.tensor([5, 10, 8], dtype=index_type)
        my_size = 2
        max_B = batch_sizes.max().item()

        new_lengths_ref = torch.tensor(
            [1, 0, 0, 2, 0, 1, 1, 1, 1, 0, 0, 1],
            # pyre-ignore [6]
            dtype=index_type,
        )
        new_indices_ref = torch.tensor(
            [1, 7, 8, 4, 3, 0, 1, 0],
            # pyre-ignore [6]
            dtype=index_type,
        )

        (
            new_lengths_cpu,
            new_indices_cpu,
            new_weights_cpu,
            new_pos_cpu,
            unbucketize_permute,
        ) = torch.ops.fbgemm.block_bucketize_sparse_features_2d_weights(
            lengths,
            indices,
            bucketize_pos,
            sequence,
            block_sizes,
            my_size,
            weights,
            weights_dim,
            batch_sizes,
            max_B,
        )

        torch.testing.assert_close(new_lengths_cpu, new_lengths_ref, rtol=0, atol=0)
        torch.testing.assert_close(new_indices_cpu, new_indices_ref, rtol=0, atol=0)
        self.assertEqual(new_weights_cpu.shape, (indices.numel(), weights_dim))

        if gpu_available:
            (
                new_lengths_gpu,
                new_indices_gpu,
                new_weights_gpu,
                new_pos_gpu,
                unbucketize_permute_gpu,
            ) = torch.ops.fbgemm.block_bucketize_sparse_features_2d_weights(
                lengths.cuda(),
                indices.cuda(),
                bucketize_pos,
                sequence,
                block_sizes.cuda(),
                my_size,
                weights.cuda(),
                weights_dim,
                batch_sizes.cuda(),
                max_B,
            )

            torch.testing.assert_close(
                new_lengths_gpu.cpu(), new_lengths_ref, rtol=0, atol=0
            )
            self.assertEqual(new_weights_gpu.shape, (indices.numel(), weights_dim))

            if sequence:
                torch.testing.assert_close(
                    new_indices_gpu.cpu(), new_indices_ref, rtol=0, atol=0
                )
            else:
                self.validate_out_of_order_output(
                    new_indices_ref, new_indices_gpu.cpu(), new_lengths_ref
                )


extend_test_class(BlockBucketize2DWeightsTest)

if __name__ == "__main__":
    unittest.main()
