#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# pyre-ignore-all-errors[56]


import random
import unittest
from math import sqrt

import fbgemm_gpu.batched_unary_embeddings_ops as batched_unary_embeddings_ops
import numpy as np
import torch

try:
    # pyre-ignore[21]
    from fbgemm_gpu import open_source  # noqa: F401

    # pyre-ignore[21]
    from test_utils import gpu_memory_lt_gb, gpu_unavailable

except Exception:
    from fbgemm_gpu.test.test_utils import gpu_memory_lt_gb, gpu_unavailable

    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")


# Relative tolerances
# pyre-fixme[5]: Global expression must be annotated.
TOLERANCE_REL = {
    torch.float32: 1e-4,
    torch.float16: 1e-2,
    torch.bfloat16: 0.1,
}

# Absolute tolerances
# pyre-fixme[5]: Global expression must be annotated.
TOLERANCE_ABS = {
    torch.float32: 1e-4,
    torch.float16: 1e-2,
    torch.bfloat16: 1e-2,
}


class TableBatchedEmbeddingsTest(unittest.TestCase):
    class RefEmb(torch.nn.Module):
        def __init__(self, num_tasks: int, hash_sizes: list[int]) -> None:
            super().__init__()
            self.num_tasks = num_tasks
            self.hash_sizes = hash_sizes
            self.emb_modules = torch.nn.ModuleList()
            for _ in range(num_tasks):
                for h in self.hash_sizes:
                    emb = torch.nn.EmbeddingBag(
                        num_embeddings=h,
                        embedding_dim=1,
                        mode="sum",
                        sparse=False,
                        include_last_offset=True,
                    )
                    emb.weight = torch.nn.Parameter(
                        torch.empty([h, 1]).uniform_(-sqrt(1 / h), sqrt(1 / h))
                    )
                    self.emb_modules.append(emb)

        def forward(
            self, offsets: list[torch.Tensor], indices: list[torch.Tensor]
        ) -> torch.Tensor:
            tt_list = []
            for n in range(self.num_tasks):
                t_list = []
                for i in range(len(self.hash_sizes)):
                    t = self.emb_modules[n * len(self.hash_sizes) + i](
                        offsets=offsets[i].long(), input=indices[i].long()
                    )
                    t_list.append(t)
                tt = torch.cat(t_list, dim=1)
                tt_list.append(tt)
            return torch.cat(tt_list).view(self.num_tasks, -1, len(self.hash_sizes))

    def _generate_unary_features(
        self,
        batch_size: int,
        num_embeddings: int,
        # pyre-fixme[24]: Generic type `list` expects 1 type parameter, use
        #  `typing.List[<element type>]` to avoid runtime subscripting errors.
    ) -> tuple[list, list, list]:
        lengths = []
        offsets = []
        indices = []
        offset = 0
        for _ in range(batch_size):
            n_indices = 1
            # pyre-fixme[6]: For 1st argument expected `Iterable[typing.Any]` but
            #  got `float`.
            indices += np.round(
                np.random.random(n_indices) * (num_embeddings - 1)
            ).tolist()
            offsets.append(offset)
            offset += 1
            lengths.append(n_indices)
        offsets.append(offset)
        return (lengths, offsets, indices)

    def _test_main(
        self,
        gpu_infer: bool,
        torch_compile: bool = False,
    ) -> None:
        if gpu_infer:
            device = torch.device("cuda:0")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")
        batch_size = 128
        hash_sizes = [100, 200]
        num_tasks = 3
        emb_dtype = random.choice([torch.float, torch.half, torch.bfloat16])
        # generate unary features
        lengths = []
        offsets = []
        indices = []
        for h in hash_sizes:
            l, o, i = self._generate_unary_features(batch_size, h)
            lengths.append(torch.IntTensor(l).to(device))
            offsets.append(torch.IntTensor(o).to(device))
            indices.append(torch.IntTensor(i).to(device))
        lengths_tensor = torch.cat(lengths)
        indices_tensor = torch.cat(indices)
        offsets_tensor = torch.zeros(
            lengths_tensor.numel() + 1,
            dtype=lengths_tensor.dtype,
            device=lengths_tensor.device,
        )
        offsets_tensor[1:] = torch.ops.fbgemm.asynchronous_inclusive_cumsum(
            lengths_tensor.view(-1)
        )
        # forward with int_32
        ref_emb = self.RefEmb(num_tasks, hash_sizes).to(device).to(emb_dtype)
        unary_emb = (
            batched_unary_embeddings_ops.BatchedUnaryEmbeddingBag(num_tasks, hash_sizes)
            .to(device)
            .to(emb_dtype)
        )
        for i, param in enumerate(unary_emb.split_embedding_weights()):
            param.detach().copy_(ref_emb.emb_modules[i].weight)
        output_ref = ref_emb(offsets, indices)
        if torch_compile:
            unary_emb = torch.compile(unary_emb, dynamic=True, fullgraph=True)
        output = unary_emb(offsets_tensor, indices_tensor)
        torch.testing.assert_close(
            output_ref,
            output,
            atol=TOLERANCE_ABS[emb_dtype],
            rtol=TOLERANCE_REL[emb_dtype],
        )

        # forward with int_64
        ref_emb = self.RefEmb(num_tasks, hash_sizes).to(device).to(emb_dtype)
        unary_emb = (
            batched_unary_embeddings_ops.BatchedUnaryEmbeddingBag(
                num_tasks=num_tasks, hash_sizes=hash_sizes, long_index=True
            )
            .to(device)
            .to(emb_dtype)
        )
        for i, param in enumerate(unary_emb.split_embedding_weights()):
            param.detach().copy_(ref_emb.emb_modules[i].weight)
        output_ref = ref_emb(offsets, indices)
        if torch_compile:
            unary_emb = torch.compile(unary_emb, dynamic=True, fullgraph=True)
        output = unary_emb(offsets_tensor.long(), indices_tensor.long())
        torch.testing.assert_close(
            output_ref,
            output,
            atol=TOLERANCE_ABS[emb_dtype],
            rtol=TOLERANCE_REL[emb_dtype],
        )

        # No implementation for CPU backprop yet
        if not gpu_infer:
            return
        # FIXME: the following doesn't work
        # with torch.compile-d unary_emb
        if torch_compile:
            return

        d_output = (
            torch.randn([num_tasks, batch_size, len(hash_sizes)]).to(device) * 0.1
        )
        output_ref.backward(d_output)
        output.backward(d_output)
        d_weight_ref = []
        for emb in ref_emb.emb_modules:
            d_weight_ref.append(emb.weight.grad)
        d_weight_ref = torch.cat(d_weight_ref).view(num_tasks, sum(hash_sizes), -1)
        d_weight = unary_emb.weight.grad  # pyre-ignore[16]
        torch.testing.assert_close(
            d_weight_ref,
            d_weight,
            atol=TOLERANCE_ABS[emb_dtype],
            rtol=TOLERANCE_REL[emb_dtype],
        )

        # Testing the case where we add permute operation, which produces
        # in contiguous grad tensor, this should also work
        unary_embedding_module = batched_unary_embeddings_ops.BatchedUnaryEmbeddingBag(
            num_tasks=3,
            hash_sizes=[71, 107],
            long_index=True,
        ).to(device)
        offsets = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.long).to(device)
        values = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.long).to(device)
        for _ in range(10):
            output = unary_embedding_module(offsets, values).transpose(1, 0)
            output = output[1:]
            output.sum().backward()

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    #  `test_utils.gpu_unavailable` to decorator factory `unittest.skipIf`.
    @unittest.skipIf(*gpu_unavailable)
    def test_gpu(self) -> None:
        self._test_main(gpu_infer=True)

    # the test below fails with CUDA error in the OSS CI
    # likely to the CUDA IMA issues in the test_gpu above
    # commenting out for now
    # @unittest.skipIf(*gpu_unavailable)
    # def test_gpu_torch_compile(self) -> None:
    #     self._test_main(gpu_infer=True, torch_compile=True)

    def test_cpu(self) -> None:
        self._test_main(gpu_infer=False)

    @unittest.skipIf(*gpu_unavailable)
    # This test exercises the HIP launch-side limit and requires a large
    # output tensor (~17 GiB) plus offsets (~4 GiB) — total ~22 GiB GPU
    # peak. Most CI machines without 24+ GiB of HBM will skip this. ROCm
    # developer machines (MI300/MI350, 192/256 GiB HBM) run it.
    @unittest.skipIf(*gpu_memory_lt_gb(24))
    def test_batched_unary_embeddings_forward_large_grid(self) -> None:
        """
        Reproduces the HIP grid-overflow bug in
        batched_unary_embeddings_forward_kernel and verifies output
        correctness via a downsampled CPU oracle.

        With block size up to 512 and grid (cuda_calc_xblock_count(B, 512),
        T, N), total threads ~= B * T * N. For B*T*N > 2**32, total threads
        exceed the HIP 2**32 limit, causing FBGEMM_LAUNCH_KERNEL ->
        KernelLauncher::checkThreadCountNotExceeded to TORCH_CHECK-fail on
        ROCm pre-fix.

        Verification strategy (per master plan's downsampled-oracle
        guidance for ops where the full-scale CPU oracle is impractical):

        1. Full-scale invocation to verify launch survival at the
           cap-trip scale. Uses zero-length offsets so the kernel does no
           per-segment work; output shape and zero-fill are asserted.
        2. Small-scale invocation vs CPU dispatch to validate kernel
           correctness end-to-end at a scale where the CPU oracle output
           (~MB range) is cheap to compute and compare element-wise.
        """

        # The production cap is `blocks_x_capped = min(blocks_x_uncapped,
        # get_max_thread_blocks(stream))` where `get_max_thread_blocks =
        # 64 * #SMs ~= 16384` on MI300/MI350. For the cap to actually help,
        # we need:
        #   (a) blocks_x_uncapped > 16384 (so cap applies) -> B > 16384*512.
        #   (b) blocks_x_uncapped * T * N * 512 > 2**32 (pre-fix trips).
        #   (c) 16384 * T * N * 512 < 2**32 (post-fix passes), i.e. T*N < 512.
        # Choose T=4, N=8 (T*N=32) and B = (1<<27)+1 = 2**27+1 so blocks_x =
        # ceil(B/512) = 2**18 + 1 (>> 16384), pre-fix total ~= 2**32 + 2**14
        # (trips), post-fix total = 16384*32*512 = 2**28 (well under 2**32).
        B = (1 << 27) + 1
        T = 4
        N = 8

        device = torch.device(torch.accelerator.current_accelerator() or "cuda")

        # ---- Step 1: full-scale launch survival (cap-trip detection). ----
        # weight[N, sum_E] where sum_E = T (each table has E=1). ~16 MB.
        weight_large = torch.zeros((N, T), dtype=torch.float32, device=device)
        table_offsets_large = torch.arange(T + 1, dtype=torch.int64, device=device)
        # offsets[T*B+1] = all-zero so each (t,b) has L=0; no per-segment work.
        offsets_large = torch.zeros(T * B + 1, dtype=torch.int64, device=device)
        indices_large = torch.empty(0, dtype=torch.int64, device=device)

        output_large = torch.ops.fbgemm.batched_unary_embeddings(
            weight_large, table_offsets_large, offsets_large, indices_large
        )

        self.assertEqual(output_large.shape, (N, B, T))
        self.assertTrue(torch.all(output_large == 0).item())
        del output_large

        # ---- Step 2: downsampled CPU-oracle correctness check. ----
        # Same kernel code path, smaller scale to keep the CPU oracle cheap.
        small_B = 4
        small_T = 3

        # Two embedding tables, each with E=2 rows. sum_E = 2*small_T = 6.
        # N=2 (rows of weight). Distinct values per (n, row) so any "kernel
        # addressed wrong row" bug surfaces.
        small_weight_cpu = torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            ],
            dtype=torch.float32,
        )
        # table_offsets[t]: start row of table t in weight columns.
        small_table_offsets_cpu = torch.tensor([0, 2, 4, 6], dtype=torch.int64)
        # offsets[t*B+b+1] - offsets[t*B+b] = length of (t,b) segment.
        # Use 1 index per (t,b) cell. Total indices = T*B = 12.
        small_offsets_cpu = torch.arange(small_T * small_B + 1, dtype=torch.int64)
        # Each (t,b) picks row 0 of its table.
        small_indices_cpu = torch.zeros(small_T * small_B, dtype=torch.int64)

        # CPU reference oracle — same op, different dispatch.
        small_output_cpu = torch.ops.fbgemm.batched_unary_embeddings(
            small_weight_cpu,
            small_table_offsets_cpu,
            small_offsets_cpu,
            small_indices_cpu,
        )

        # GPU under test.
        small_output_gpu = torch.ops.fbgemm.batched_unary_embeddings(
            small_weight_cpu.to(device),
            small_table_offsets_cpu.to(device),
            small_offsets_cpu.to(device),
            small_indices_cpu.to(device),
        )

        torch.testing.assert_close(small_output_gpu.cpu(), small_output_cpu)


if __name__ == "__main__":
    unittest.main()
