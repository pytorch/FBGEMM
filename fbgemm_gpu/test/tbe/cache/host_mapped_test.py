#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[56]

"""
Regression tests for `uvm_to_cpu`, `uvm_to_device`, and `uvm_get_guard_index`
on *host-mapped* UVM tensors (created via `new_host_mapped_tensor` or
`new_unified_tensor(..., is_host_mapped=True)`).

These exercise the `CUDAHostMappedContext` branch added to memory_utils.cu.
Before that fix, these ops unconditionally cast the storage context to
`CUDAManagedIndirectContext`, which returns nullptr for host-mapped tensors
and triggers `TORCH_CHECK(tcontext != nullptr)` — surfacing as
"Expected tcontext != nullptr" during DCP checkpoint loading.

The existing `copy_test.py` only covers the `CUDAManagedIndirectContext`
path (`new_managed_tensor` / `new_vanilla_managed_tensor` /
`new_unified_tensor(is_host_mapped=False)`); this file fills the host-mapped
gap and also regresses the managed path.
"""

import unittest

import fbgemm_gpu
import hypothesis.strategies as st
import torch
from hypothesis import given, settings, Verbosity

# pyre-fixme[16]: Module `fbgemm_gpu` has no attribute `open_source`.
open_source: bool = getattr(fbgemm_gpu, "open_source", False)

if open_source:
    # pyre-ignore[21]
    from test_utils import gpu_available, gpu_unavailable
else:
    from fbgemm_gpu.test.test_utils import gpu_available, gpu_unavailable

if gpu_available:
    import fbgemm_gpu.tbe.cache.uvm  # noqa: F401


MAX_EXAMPLES = 20


def _make_host_mapped(sizes: list[int]) -> torch.Tensor:
    """Allocate a host-mapped UVM tensor on cuda:0."""
    return torch.ops.fbgemm.new_host_mapped_tensor(
        torch.empty(0, device="cuda:0", dtype=torch.float),
        sizes,
    )


def _make_unified_host_mapped(sizes: list[int]) -> torch.Tensor:
    """Allocate a host-mapped unified tensor on cuda:0."""
    return torch.ops.fbgemm.new_unified_tensor(
        torch.empty(0, device="cuda:0", dtype=torch.float),
        sizes,
        True,  # is_host_mapped
    )


def _make_managed(sizes: list[int]) -> torch.Tensor:
    """Allocate a (managed) UVM tensor on cuda:0."""
    return torch.ops.fbgemm.new_managed_tensor(
        torch.empty(0, device="cuda:0", dtype=torch.float),
        sizes,
    )


class HostMappedUvmOpsTest(unittest.TestCase):
    """Tests for uvm_to_cpu / uvm_to_device / uvm_get_guard_index on
    host-mapped tensors, plus regression coverage of the managed path."""

    # ---------- uvm_to_cpu ----------

    @unittest.skipIf(*gpu_unavailable)
    @given(
        sizes=st.lists(st.integers(min_value=1, max_value=8), min_size=1, max_size=3),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_uvm_to_cpu_host_mapped(self, sizes: list[int]) -> None:
        uvm_t = _make_host_mapped(sizes)
        self.assertTrue(torch.ops.fbgemm.is_uvm_tensor(uvm_t))
        self.assertTrue(torch.ops.fbgemm.uvm_storage(uvm_t))

        # The bug pre-fix raised "Expected tcontext != nullptr" here.
        cpu_t = torch.ops.fbgemm.uvm_to_cpu(uvm_t)

        self.assertEqual(cpu_t.device.type, "cpu")
        self.assertEqual(list(cpu_t.shape), sizes)
        self.assertFalse(torch.ops.fbgemm.is_uvm_tensor(cpu_t))
        # cpu_t should still report uvm_storage=True since it shares storage.
        self.assertTrue(torch.ops.fbgemm.uvm_storage(cpu_t))

        # Mutate via the CPU view; the change must be visible on the GPU view,
        # confirming shared storage (no deep copy).
        cpu_t.fill_(7.0)
        torch.cuda.synchronize()
        self.assertTrue(torch.all(uvm_t.cpu() == 7.0).item())

        # CPU view must outlive the original UVM tensor (storage refcount).
        del uvm_t
        cpu_t.mul_(2)  # would segfault if storage was freed
        self.assertTrue(torch.all(cpu_t == 14.0).item())

    @unittest.skipIf(*gpu_unavailable)
    @given(
        sizes=st.lists(st.integers(min_value=1, max_value=8), min_size=1, max_size=3),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_uvm_to_cpu_unified_host_mapped(self, sizes: list[int]) -> None:
        uvm_t = _make_unified_host_mapped(sizes)
        self.assertTrue(torch.ops.fbgemm.is_uvm_tensor(uvm_t))
        cpu_t = torch.ops.fbgemm.uvm_to_cpu(uvm_t)
        self.assertEqual(cpu_t.device.type, "cpu")
        self.assertEqual(list(cpu_t.shape), sizes)

    @unittest.skipIf(*gpu_unavailable)
    def test_uvm_to_cpu_managed_regression(self) -> None:
        """Regression: ensure the managed path still works unchanged."""
        uvm_t = _make_managed([4, 8])
        cpu_t = torch.ops.fbgemm.uvm_to_cpu(uvm_t)
        self.assertEqual(cpu_t.device.type, "cpu")
        self.assertEqual(list(cpu_t.shape), [4, 8])
        self.assertFalse(torch.ops.fbgemm.is_uvm_tensor(cpu_t))
        self.assertTrue(torch.ops.fbgemm.uvm_storage(cpu_t))

        cpu_t.fill_(3.0)
        torch.cuda.synchronize()
        self.assertTrue(torch.all(uvm_t.cpu() == 3.0).item())

    # ---------- uvm_to_device ----------

    @unittest.skipIf(*gpu_unavailable)
    def test_uvm_to_device_host_mapped_same_device(self) -> None:
        """Re-wrap a host-mapped tensor for the same CUDA device."""
        uvm_t = _make_host_mapped([4, 4])
        target_device = torch.device("cuda:0")
        rewrapped = torch.ops.fbgemm.uvm_to_device_d(uvm_t, target_device)
        self.assertEqual(rewrapped.device, target_device)
        self.assertEqual(list(rewrapped.shape), [4, 4])
        # Should still be considered a UVM tensor.
        self.assertTrue(torch.ops.fbgemm.is_uvm_tensor(rewrapped))
        self.assertTrue(torch.ops.fbgemm.uvm_storage(rewrapped))

        # Shared storage: mutations propagate.
        rewrapped.fill_(5.0)
        torch.cuda.synchronize()
        self.assertTrue(torch.all(uvm_t.cpu() == 5.0).item())

    @unittest.skipIf(
        not torch.cuda.is_available() or torch.cuda.device_count() < 2,
        "Skip unless two CUDA devices are detected",
    )
    def test_uvm_to_device_host_mapped_cross_device(self) -> None:
        """Re-wrap a host-mapped tensor for a *different* CUDA device."""
        uvm_t = _make_host_mapped([8, 8])
        prototype = torch.empty(0, device="cuda:1")
        second_t = torch.ops.fbgemm.uvm_to_device(uvm_t, prototype)
        self.assertEqual(second_t.device, prototype.device)
        self.assertTrue(torch.ops.fbgemm.is_uvm_tensor(second_t))
        self.assertTrue(torch.ops.fbgemm.uvm_storage(second_t))

    @unittest.skipIf(*gpu_unavailable)
    def test_uvm_to_device_managed_regression(self) -> None:
        uvm_t = _make_managed([4, 4])
        target_device = torch.device("cuda:0")
        rewrapped = torch.ops.fbgemm.uvm_to_device_d(uvm_t, target_device)
        self.assertEqual(rewrapped.device, target_device)
        self.assertTrue(torch.ops.fbgemm.is_uvm_tensor(rewrapped))

    @unittest.skipIf(*gpu_unavailable)
    def test_uvm_to_device_host_mapped_chained(self) -> None:
        """Chained call: the result of uvm_to_device_d is itself a uvm tensor
        (deleter = CUDAHostMappedIndirectContext::release). A second
        uvm_to_device_d must recognize that and not fall through to the
        managed-memory path, where cast_context<CUDAManagedIndirectContext>
        would return nullptr and trigger TORCH_CHECK."""
        uvm_t = _make_host_mapped([8, 8])
        target_device = torch.device("cuda:0")
        first = torch.ops.fbgemm.uvm_to_device_d(uvm_t, target_device)
        self.assertTrue(torch.ops.fbgemm.is_uvm_tensor(first))
        # Must not raise.
        second = torch.ops.fbgemm.uvm_to_device_d(first, target_device)
        self.assertEqual(second.device, target_device)
        self.assertTrue(torch.ops.fbgemm.is_uvm_tensor(second))

        # Storage is still shared end-to-end.
        second.fill_(9.0)
        torch.cuda.synchronize()
        self.assertTrue(torch.all(uvm_t.cpu() == 9.0).item())

    @unittest.skipIf(*gpu_unavailable)
    def test_uvm_to_cpu_after_uvm_to_device_host_mapped(self) -> None:
        """uvm_to_cpu of an already-rewrapped (indirect-context) host-mapped
        CUDA tensor must take the host-mapped branch, not fall through."""
        uvm_t = _make_host_mapped([8, 8])
        rewrapped = torch.ops.fbgemm.uvm_to_device_d(uvm_t, torch.device("cuda:0"))
        cpu_t = torch.ops.fbgemm.uvm_to_cpu(rewrapped)
        self.assertEqual(cpu_t.device.type, "cpu")
        cpu_t.fill_(2.0)
        torch.cuda.synchronize()
        self.assertTrue(torch.all(uvm_t.cpu() == 2.0).item())

    # ---------- uvm_get_guard_index (exercised via cuda_mem_advise) ----------

    @unittest.skipIf(*gpu_unavailable)
    def test_uvm_get_guard_index_host_mapped_cuda_view(self) -> None:
        """CUDA-side path uses t.get_device(); host-mapped tensor's device
        index should resolve correctly."""
        uvm_t = _make_host_mapped([16, 16])
        # Direct call on the cuda tensor goes through the t.is_cuda() branch,
        # which is unchanged but still must not regress.
        torch.ops.fbgemm.cuda_mem_advise(uvm_t, 5)

    @unittest.skipIf(*gpu_unavailable)
    def test_uvm_get_guard_index_managed_regression(self) -> None:
        uvm_t = _make_managed([16, 16])
        cpu_view = torch.ops.fbgemm.uvm_to_cpu(uvm_t)
        torch.ops.fbgemm.cuda_mem_advise(cpu_view, 5)
        torch.ops.fbgemm.cuda_mem_advise(uvm_t, 5)


if __name__ == "__main__":
    unittest.main()
