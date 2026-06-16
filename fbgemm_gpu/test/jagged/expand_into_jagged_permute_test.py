#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[56]

import itertools
import random
import unittest

import hypothesis.strategies as st
import torch
import torch._dynamo
from hypothesis import given, settings, Verbosity

from .common import additional_decorators, open_source

if open_source:
    # pyre-ignore[21]
    from test_utils import (
        gpu_available,
        gpu_memory_lt_gb,
        gpu_unavailable,
        on_oss_clang,
        optests,
    )
else:
    from fbgemm_gpu.test.test_utils import (
        gpu_available,
        gpu_memory_lt_gb,
        gpu_unavailable,
        on_oss_clang,
        optests,
    )


@optests.generate_opcheck_tests(additional_decorators=additional_decorators)
class ExpandIntoJaggedPermuteTest(unittest.TestCase):
    @staticmethod
    def expand_into_jagged_permute_ref_(
        permute: list[int],
        length: list[int],
    ) -> list[int]:
        offsets = [0] + list(itertools.accumulate(length))
        output_permute = []
        for r in permute:
            output_permute.extend(
                range(
                    offsets[r],
                    offsets[r + 1],
                )
            )
        return output_permute

    @unittest.skipIf(*on_oss_clang)
    @given(
        T=st.integers(min_value=10, max_value=20),
        W=st.integers(min_value=8, max_value=64),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=10, deadline=None)
    def test_expand_into_jagged_permute(
        self,
        T: int,
        W: int,
    ) -> None:
        length_per_w = [random.randint(5000, 10000) for i in range(W)]
        length_1d = list(
            itertools.chain.from_iterable(itertools.repeat(x, T) for x in length_per_w)
        )
        permute_list = list(range(T * W))
        random.shuffle(permute_list)
        permuted_length_1d = [length_1d[r] for r in permute_list]
        permute_tensor = torch.tensor(permute_list)

        # compute offsets
        offsets_1d = [0] + list(itertools.accumulate(length_1d))
        permuted_offsets_1d = [0] + list(itertools.accumulate(permuted_length_1d))
        offsets_1d_tensor = torch.tensor(offsets_1d)
        permuted_offsets_1d_tensor = torch.tensor(permuted_offsets_1d)

        # cpu op
        output_permute_cpu = torch.ops.fbgemm.expand_into_jagged_permute(
            permute_tensor,
            offsets_1d_tensor,
            permuted_offsets_1d_tensor,
            offsets_1d[-1],
        )

        # reference solution
        output_permute_ref = self.expand_into_jagged_permute_ref_(
            permute_list,
            length_1d,
        )
        output_permute_ref_tensor = torch.tensor(output_permute_ref)

        # assert cpu and gpu ops
        torch.testing.assert_close(output_permute_cpu, output_permute_ref_tensor)
        if gpu_available:
            # gpu op
            output_permute_gpu = torch.ops.fbgemm.expand_into_jagged_permute(
                permute_tensor.cuda(),
                offsets_1d_tensor.cuda(),
                permuted_offsets_1d_tensor.cuda(),
                offsets_1d[-1],
            )
            torch.testing.assert_close(
                output_permute_gpu.cpu(), output_permute_ref_tensor
            )

    @unittest.skipIf(*gpu_unavailable)
    # Skip on GPUs with insufficient HBM. Inputs are int32 of size
    # ~permute_size, totaling ~1.5 GiB at the chosen permute_size.
    @unittest.skipIf(*gpu_memory_lt_gb(4))
    def test_expand_into_jagged_permute_large_grid(self) -> None:
        """
        Reproduces the HIP grid-overflow bug in expand_into_jagged_permute_cuda
        and verifies output correctness at the same scale.

        With T_blocks = kMaxThreads/kWarpSize = 32 and dim3(32, 32)
        (block size 1024), the launch grid is
        cuda_calc_xblock_count(permute_size, 32). For
        permute_size > 2**27, total threads exceed the HIP 2**32
        limit, causing FBGEMM_LAUNCH_KERNEL ->
        KernelLauncher::checkThreadCountNotExceeded to TORCH_CHECK-fail
        on ROCm pre-fix. With the production fix in place, this test
        additionally validates output correctness against the CPU
        dispatch of the same op — the GPU output must match the CPU
        reference element-for-element.

        ``length`` (per segment) is sparse: all zero except for three
        known non-zero positions (start / middle / end of the logical
        range), so HBM usage stays bounded (~1.6 GiB int32) while the
        permutation expansion logic is still exercised. ``permute`` is
        a deterministic non-identity circular shift (``perm[i] != i``
        everywhere), so any "kernel computed identity instead of
        permutation" bug surfaces in the assertion below.
        """

        # Choose permute_size so that total threads strictly exceeds 2**32:
        # cuda_calc_xblock_count(permute_size, 32) * 1024 ~= permute_size * 32;
        # need permute_size > 2**27.
        permute_size = (1 << 27) + 1

        device = torch.device(torch.accelerator.current_accelerator() or "cuda")

        # Sparse non-zero lengths at start / middle / end. Total = 10.
        lengths_cpu = torch.zeros(permute_size, dtype=torch.int32)
        lengths_cpu[0] = 3
        lengths_cpu[permute_size // 2] = 5
        lengths_cpu[permute_size - 1] = 2

        # Deterministic non-identity permute: circular shift by +1.
        # perm_cpu[0] == permute_size - 1 and perm_cpu[i] == i - 1 for
        # i >= 1, so perm_cpu[i] != i for every i.
        perm_cpu = torch.roll(torch.arange(permute_size, dtype=torch.int32), 1)

        # Build offsets (size permute_size + 1) on CPU.
        input_offsets_cpu = torch.zeros(permute_size + 1, dtype=torch.int32)
        input_offsets_cpu[1:] = torch.cumsum(lengths_cpu, dim=0).to(torch.int32)

        permuted_lengths_cpu = lengths_cpu[perm_cpu.long()]
        output_offsets_cpu = torch.zeros(permute_size + 1, dtype=torch.int32)
        output_offsets_cpu[1:] = torch.cumsum(permuted_lengths_cpu, dim=0).to(
            torch.int32
        )

        total_length = int(input_offsets_cpu[-1].item())

        # CPU reference oracle — same op, different dispatch.
        output_permute_cpu = torch.ops.fbgemm.expand_into_jagged_permute(
            perm_cpu, input_offsets_cpu, output_offsets_cpu, total_length
        )

        # GPU op under test. Pre-fix, this launch trips
        # KernelLauncher::checkThreadCountNotExceeded on ROCm.
        output_permute_gpu = torch.ops.fbgemm.expand_into_jagged_permute(
            perm_cpu.to(device),
            input_offsets_cpu.to(device),
            output_offsets_cpu.to(device),
            total_length,
        )

        torch.testing.assert_close(output_permute_gpu.cpu(), output_permute_cpu)


if __name__ == "__main__":
    unittest.main()
