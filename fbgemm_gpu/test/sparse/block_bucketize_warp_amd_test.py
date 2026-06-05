# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# AMD/ROCm wavefront64 regression guard for the warp-parallel
# populate_bucketized_permute kernel.
#
# This is a deliberately *minimal* test module: it does NOT call
# generate_opcheck_tests / extend_test_class, so no faketensor / aot_dispatch
# opcheck variants are generated. That keeps this target free of the
# pre-existing abstract-kernel/failures_dict coverage gaps and lets it gate
# purely on numerical correctness of the warp kernel on wavefront64 hardware.
#
# The owning BUCK target forces the warp-parallel kernel on via
# FBGEMM_NO_JK=1 + FBGEMM_BUCKETIZED_PERMUTE_WARP_KERNEL=1 and runs on the
# amd-gpu (MI300) platform. With my_size <= kWarpSize and sequences longer than
# 32 elements, lanes 32-63 of the 64-bit ballot are exercised; the pre-fix
# 32-bit __popc truncated them, producing colliding output offsets. The fix
# (width-correct ballot_sync + __popcll) makes the GPU warp kernel match the
# CPU reference exactly.

import unittest

import fbgemm_gpu  # noqa: F401  # registers torch.ops.fbgemm operators
import torch


class BlockBucketizeWarpAmdTest(unittest.TestCase):
    def test_warp_parallel_vs_reference_wavefront64(self) -> None:
        """Warp-parallel populate_bucketized_permute must match the CPU
        reference for sequences longer than 32 elements, which is the only
        regime that exercises lanes 32-63 on wavefront64 (MI300/MI350)."""
        if not torch.cuda.is_available():
            self.skipTest("No GPU available")
        torch.manual_seed(42)

        # (my_size, block_size, lengths). Every config includes at least one
        # sequence with length > 32 so lanes 32-63 are active; lengths > 64
        # additionally exercise multi-chunk cumulative counting.
        test_configs = [
            # Canonical serving case: my_size=2, single length-64 sequence
            # (one full wavefront, lanes 0-63 active) plus multi-chunk 128.
            (2, 50, torch.tensor([64, 128, 33, 0, 1, 40])),
            # Moderate bucket count, varied lengths spanning the boundary.
            (8, 10, torch.tensor([33, 64, 65, 100, 0, 32])),
            # Larger bucket count, multi-chunk.
            (16, 5, torch.tensor([48, 96, 0, 17, 65])),
            # Boundary my_size for the warp path, lengths past 32 and 64.
            (32, 1, torch.tensor([33, 64, 65, 0, 128])),
            # Many sequences to stress grid-level parallelism, all > 32.
            (4, 10, torch.randint(33, 129, (256,))),
        ]

        for index_type in (torch.int, torch.long):
            for my_size, block_size, lengths in test_configs:
                lengths = lengths.to(index_type)
                block_sizes = torch.tensor([block_size], dtype=index_type)
                total = int(lengths.sum().item())
                indices = torch.randint(
                    0, my_size * block_size, (total,), dtype=index_type
                )

                # Ground truth + CPU reference (serial kernel).
                (
                    new_lengths_cpu,
                    _,
                    _,
                    _,
                    unbucketize_permute_ref,
                    bucket_mapping_cpu,
                ) = torch.ops.fbgemm.block_bucketize_sparse_features_inference(
                    lengths,
                    indices,
                    False,
                    True,
                    block_sizes,
                    my_size,
                    None,
                    return_bucket_mapping=True,
                )
                permute_cpu = torch.ops.fbgemm.populate_bucketized_permute(
                    lengths,
                    new_lengths_cpu,
                    bucket_mapping_cpu,
                )

                # GPU warp-parallel kernel (forced on via the target's env).
                (
                    new_lengths_gpu,
                    _,
                    _,
                    _,
                    _,
                    bucket_mapping_gpu,
                ) = torch.ops.fbgemm.block_bucketize_sparse_features_inference(
                    lengths.to(torch.accelerator.current_accelerator()),
                    indices.to(torch.accelerator.current_accelerator()),
                    False,
                    True,
                    block_sizes.to(torch.accelerator.current_accelerator()),
                    my_size,
                    None,
                    return_bucket_mapping=True,
                )
                permute_gpu = torch.ops.fbgemm.populate_bucketized_permute(
                    lengths.to(torch.accelerator.current_accelerator()),
                    new_lengths_gpu,
                    bucket_mapping_gpu,
                ).cpu()

                msg = (
                    f"warp-parallel populate_bucketized_permute mismatch for "
                    f"my_size={my_size}, block_size={block_size}, "
                    f"index_type={index_type}, lengths={lengths.tolist()}"
                )
                # Output must be a permutation identical to the CPU paths.
                torch.testing.assert_close(
                    permute_gpu, permute_cpu, rtol=0, atol=0, msg=msg
                )
                torch.testing.assert_close(
                    permute_gpu, unbucketize_permute_ref, rtol=0, atol=0, msg=msg
                )
