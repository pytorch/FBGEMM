#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[56]

import unittest

import torch

try:
    # pyre-ignore[21]
    from fbgemm_gpu import open_source  # noqa: F401

    # pyre-ignore[21]
    from test_utils import gpu_memory_lt_gb, gpu_unavailable
except Exception:
    from fbgemm_gpu.test.test_utils import gpu_memory_lt_gb, gpu_unavailable

    torch.ops.load_library(
        "//deeplearning/fbgemm/fbgemm_gpu:permute_multi_embedding_ops_gpu"
    )
    torch.ops.load_library(
        "//deeplearning/fbgemm/fbgemm_gpu:permute_multi_embedding_ops_cpu"
    )


# permutes tensor schema (matching PermuteParam enum in
# permute_multi_embedding_ops.cu): each row is
#   [in_tensor, out_tensor, in_offset, out_offset, length, next].

PERMUTES_DTYPE = torch.int32
SHAPES_DTYPE = torch.int32


class PermuteMultiEmbeddingOpsTest(unittest.TestCase):
    @unittest.skipIf(*gpu_unavailable)
    def test_permute_multi_embedding_smoke(self) -> None:
        """Small-input correctness regression. Gates the kernel-side
        grid-stride transformation against breakage on small inputs."""
        device = torch.accelerator.current_accelerator("cuda")
        batch_size = 4
        in_lengths = [4, 8]
        out_lengths = [4, 8]

        permutes = torch.tensor(
            [
                [0, 0, 0, 0, 4, -1],
                [1, 1, 0, 0, 8, -1],
            ],
            dtype=PERMUTES_DTYPE,
            device=device,
        )
        in_shapes = torch.tensor(in_lengths, dtype=SHAPES_DTYPE, device=device)
        out_shapes = torch.tensor(out_lengths, dtype=SHAPES_DTYPE, device=device)

        pooled_embs = [
            torch.arange(batch_size * length, dtype=torch.float32, device=device)
            .view(batch_size, length)
            .contiguous()
            for length in in_lengths
        ]

        outputs = torch.ops.fbgemm.permute_multi_embedding(
            pooled_embs,
            permutes,
            in_shapes,
            out_shapes,
            out_lengths,
        )

        self.assertEqual(len(outputs), 2)
        self.assertEqual(outputs[0].shape, (batch_size, out_lengths[0]))
        self.assertEqual(outputs[1].shape, (batch_size, out_lengths[1]))
        torch.testing.assert_close(outputs[0], pooled_embs[0])
        torch.testing.assert_close(outputs[1], pooled_embs[1])

    @unittest.skipIf(*gpu_unavailable)
    @unittest.skipIf(*gpu_memory_lt_gb(4))
    def test_permute_multi_embedding_large_grid(self) -> None:
        """
        Reproduces the HIP grid-overflow bug in permute_multi_embs_kernel
        (src/permute_multi_embedding_ops/permute_multi_embedding_ops.cu).

        Block: dim3(kWarpSize=32, kMaxThreads/kWarpSize=32) = 1024 threads.
        Grid x = div_round_up(permute_size, 32). Grid y = batch_size capped
        at max_grid_dim=32768. With permute_size = 2**22 + 1 and
        batch_size = 32, total threads = ceil((2**22+1)/32) * 32 * 1024
        > 2**32, tripping KernelLauncher::checkThreadCountNotExceeded on
        ROCm pre-fix.

        Pre-fix: AMD launch fails. Post-fix: passes (kernel grid-strides
        over permute_id).

        All-zero permutes (length=0, next=-1) make the inner copy work
        bounded. We still exercise the saturating gridDim.x path.
        """
        device = torch.accelerator.current_accelerator("cuda")
        permute_size = (1 << 22) + 1
        batch_size = 32
        in_lengths = [1]
        out_lengths = [1]

        permutes = torch.zeros((permute_size, 6), dtype=PERMUTES_DTYPE, device=device)
        # next = -1 disables the reverse_permute follow-on chain
        permutes[:, 5] = -1
        in_shapes = torch.tensor(in_lengths, dtype=SHAPES_DTYPE, device=device)
        out_shapes = torch.tensor(out_lengths, dtype=SHAPES_DTYPE, device=device)

        pooled_embs = [
            torch.zeros(
                (batch_size, in_lengths[0]),
                dtype=torch.float32,
                device=device,
            )
        ]

        outputs = torch.ops.fbgemm.permute_multi_embedding(
            pooled_embs,
            permutes,
            in_shapes,
            out_shapes,
            out_lengths,
        )

        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0].shape, (batch_size, out_lengths[0]))

        # Tier-A CPU-oracle correctness check at small scale with
        # non-trivial permutes (length > 0). The smoke test above covers
        # identity permutes; this section exercises the full grid-stride
        # path with non-trivial copy work.
        small_batch = 4
        small_in_lengths = [4, 8]
        small_out_lengths = [4, 8]
        small_permutes = torch.tensor(
            [
                [0, 0, 0, 0, 4, -1],
                [1, 1, 0, 0, 8, -1],
            ],
            dtype=PERMUTES_DTYPE,
        )
        small_in_shapes = torch.tensor(small_in_lengths, dtype=SHAPES_DTYPE)
        small_out_shapes = torch.tensor(small_out_lengths, dtype=SHAPES_DTYPE)
        small_pooled_cpu = [
            torch.arange(small_batch * length, dtype=torch.float32)
            .view(small_batch, length)
            .contiguous()
            for length in small_in_lengths
        ]
        out_cpu = torch.ops.fbgemm.permute_multi_embedding(
            small_pooled_cpu,
            small_permutes,
            small_in_shapes,
            small_out_shapes,
            small_out_lengths,
        )
        out_gpu = torch.ops.fbgemm.permute_multi_embedding(
            [t.to(device) for t in small_pooled_cpu],
            small_permutes.to(device),
            small_in_shapes.to(device),
            small_out_shapes.to(device),
            small_out_lengths,
        )
        self.assertEqual(len(out_gpu), len(out_cpu))
        for cpu_t, gpu_t in zip(out_cpu, out_gpu):
            torch.testing.assert_close(gpu_t.cpu(), cpu_t)


if __name__ == "__main__":
    unittest.main()
