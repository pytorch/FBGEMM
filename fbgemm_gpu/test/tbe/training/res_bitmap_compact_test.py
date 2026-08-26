#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import torch
from fbgemm_gpu.split_table_batched_embeddings_ops_training import res_bitmap_compact
from torch.utils._python_dispatch import TorchDispatchMode


class _HostSync(AssertionError):
    pass


class _NoHostSync(TorchDispatchMode):
    """Raises on any op whose output size depends on tensor DATA.

    Those are exactly the ops that must read the device before they can
    allocate, which is the sync `res_bitmap_compact` exists to avoid. Works on
    CPU because interception is at the dispatcher, not at the device.
    """

    BANNED = {"aten::nonzero", "aten::masked_select", "aten::_local_scalar_dense"}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        schema = getattr(func, "_schema", None)
        name = schema.name if schema is not None else str(func)
        if name in self.BANNED:
            raise _HostSync(name)
        return func(*args, **(kwargs or {}))


def _reference(selected: torch.Tensor) -> list[int]:
    """Ascending positions where `selected` is True, computed the obvious way.

    Deliberately uses `nonzero()`, which `res_bitmap_compact` avoids because it
    syncs. Correctness is the same; only the sync behaviour differs.
    """
    return selected.nonzero().flatten().tolist()


class ResBitmapCompactTest(unittest.TestCase):
    def _check(self, selected: torch.Tensor) -> None:
        rows, count = res_bitmap_compact(selected)
        expected = _reference(selected)
        self.assertEqual(int(count.item()), len(expected))
        self.assertEqual(rows[: len(expected)].tolist(), expected)
        # One longer than the input: the extra slot is where the Falses go.
        self.assertEqual(rows.numel(), selected.numel() + 1)

    def test_compaction_does_not_host_sync(self) -> None:
        # The property the helper exists for, and the only test that pins it:
        # a `nonzero()` implementation keeping the n+1 buffer and the 0-dim
        # count passes every other test in this file.
        selected = torch.tensor([False, True, False, True, True])
        with _NoHostSync():
            res_bitmap_compact(selected)

        # Positive control, same invocation: `_reference` really does sync, so
        # an inert mode fails here rather than passing the assertions above.
        with self.assertRaises(_HostSync):
            with _NoHostSync():
                _reference(selected)

    def test_mixed_selection_returns_ascending_positions(self) -> None:
        selected = torch.tensor([False, True, False, True, True])
        self._check(selected)

    def test_no_positions_selected_yields_zero_count(self) -> None:
        selected = torch.zeros(8, dtype=torch.bool)
        rows, count = res_bitmap_compact(selected)
        self.assertEqual(int(count.item()), 0)
        self.assertEqual(rows.numel(), 9)

    def test_every_position_selected_yields_identity(self) -> None:
        selected = torch.ones(6, dtype=torch.bool)
        rows, count = res_bitmap_compact(selected)
        self.assertEqual(int(count.item()), 6)
        self.assertEqual(rows[:6].tolist(), list(range(6)))

    def test_first_and_last_positions_are_not_dropped(self) -> None:
        # Boundary positions are where an off-by-one in the cumsum rank shows up.
        selected = torch.tensor([True, False, False, False, True])
        self._check(selected)

    def test_result_is_independent_of_selection_density(self) -> None:
        # Anti-vacuity: a wrong implementation that returned arange() or an
        # empty slice would pass a single-density test. Vary the density and
        # require the reference to disagree between cases.
        sparse = torch.zeros(32, dtype=torch.bool)
        sparse[7] = True
        dense = torch.ones(32, dtype=torch.bool)
        dense[7] = False
        self.assertNotEqual(_reference(sparse), _reference(dense))
        self._check(sparse)
        self._check(dense)

    def test_input_mask_is_not_mutated(self) -> None:
        # The caller reuses the presence mask across drains, so compaction must
        # leave it untouched.
        selected = torch.tensor([False, True, True, False])
        before = selected.clone()
        res_bitmap_compact(selected)
        self.assertEqual(selected.tolist(), before.tolist())
