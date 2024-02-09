#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors[56]

import unittest
from typing import List

import fbgemm_gpu
import hypothesis.strategies as st
import torch
from hypothesis import given, settings, Verbosity

# pyre-fixme[16]: Module `fbgemm_gpu` has no attribute `open_source`.
open_source: bool = getattr(fbgemm_gpu, "open_source", False)

if open_source:
    # pyre-ignore[21]
    from test_utils import gpu_available, gpu_unavailable, skipIfRocm
else:
    from fbgemm_gpu.test.test_utils import gpu_available, gpu_unavailable, skipIfRocm

if gpu_available:
    import fbgemm_gpu.uvm

MAX_EXAMPLES = 40


class CopyTest(unittest.TestCase):
    @unittest.skipIf(*gpu_unavailable)
    @given(
        sizes=st.lists(st.integers(min_value=1, max_value=8), min_size=1, max_size=4),
        uvm_op=st.sampled_from(
            [
                torch.ops.fbgemm.new_unified_tensor,
                torch.ops.fbgemm.new_managed_tensor,
                torch.ops.fbgemm.new_vanilla_managed_tensor,
            ]
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    # pyre-fixme[2]: Parameter must be annotated.
    def test_uvm_to_cpu(self, sizes: List[int], uvm_op) -> None:
        if uvm_op is torch.ops.fbgemm.new_unified_tensor:
            is_host_mapped = False
            uvm_t = uvm_op(
                torch.empty(0, device="cuda:0", dtype=torch.float),
                sizes,
                is_host_mapped,
            )
        else:
            uvm_t = uvm_op(torch.empty(0, device="cuda:0", dtype=torch.float), sizes)

        cpu_t = torch.ops.fbgemm.uvm_to_cpu(uvm_t)
        assert not torch.ops.fbgemm.is_uvm_tensor(cpu_t)
        assert torch.ops.fbgemm.uvm_storage(cpu_t)

        uvm_t.copy_(cpu_t)
        assert torch.ops.fbgemm.is_uvm_tensor(uvm_t)
        assert torch.ops.fbgemm.uvm_storage(uvm_t)

        # Test use of cpu tensor after freeing the uvm tensor
        del uvm_t
        cpu_t.mul_(42)

    @skipIfRocm()
    @unittest.skipIf(
        not torch.cuda.is_available() or torch.cuda.device_count() < 2,
        "Skip unless two CUDA devices are detected",
    )
    @given(
        sizes=st.lists(
            st.integers(min_value=1, max_value=(1024)), min_size=1, max_size=4
        ),
        uvm_op=st.sampled_from(
            [
                torch.ops.fbgemm.new_unified_tensor,
                torch.ops.fbgemm.new_managed_tensor,
                torch.ops.fbgemm.new_vanilla_managed_tensor,
            ]
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    # pyre-fixme[2]: Parameter must be annotated.
    def test_uvm_to_device(self, sizes: List[int], uvm_op) -> None:
        if uvm_op is torch.ops.fbgemm.new_unified_tensor:
            is_host_mapped = False
            uvm_t = uvm_op(
                torch.empty(0, device="cuda:0", dtype=torch.float),
                sizes,
                is_host_mapped,
            )
        else:
            uvm_t = uvm_op(torch.empty(0, device="cuda:0", dtype=torch.float), sizes)

        assert torch.ops.fbgemm.is_uvm_tensor(uvm_t)
        assert torch.ops.fbgemm.uvm_storage(uvm_t)

        # Reference uvm tensor from second cuda device
        try:
            device_prototype = torch.empty(0, device="cuda:1")
        except RuntimeError:
            # Skip the tests if there is no "cuda:1" device
            return

        second_t = torch.ops.fbgemm.uvm_to_device(uvm_t, device_prototype)

        assert torch.ops.fbgemm.is_uvm_tensor(second_t)
        assert torch.ops.fbgemm.uvm_storage(second_t)
        assert second_t.device == device_prototype.device

    @unittest.skipIf(*gpu_unavailable)
    @given(
        sizes=st.lists(
            st.integers(min_value=1, max_value=(512)), min_size=1, max_size=3
        ),
        uvm_op=st.sampled_from(
            [
                torch.ops.fbgemm.new_unified_tensor,
                torch.ops.fbgemm.new_managed_tensor,
                torch.ops.fbgemm.new_vanilla_managed_tensor,
            ]
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    # pyre-fixme[2]: Parameter must be annotated.
    def test_uvm_to_cpu_clone(self, sizes: List[int], uvm_op) -> None:
        if uvm_op is torch.ops.fbgemm.new_unified_tensor:
            is_host_mapped = False
            uvm_t = uvm_op(
                torch.empty(0, device="cuda:0", dtype=torch.float),
                sizes,
                is_host_mapped,
            )
        else:
            uvm_t = uvm_op(torch.empty(0, device="cuda:0", dtype=torch.float), sizes)

        assert torch.ops.fbgemm.is_uvm_tensor(uvm_t)
        assert torch.ops.fbgemm.uvm_storage(uvm_t)

        cpu_clone = torch.ops.fbgemm.uvm_to_cpu_clone(uvm_t)

        assert not torch.ops.fbgemm.is_uvm_tensor(cpu_clone)
        assert not torch.ops.fbgemm.uvm_storage(cpu_clone)


if __name__ == "__main__":
    unittest.main()
