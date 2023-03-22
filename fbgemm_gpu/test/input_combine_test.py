#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import unittest
from typing import List, Optional, Tuple

import torch
from hypothesis import given, settings

try:
    # pyre-ignore[21]
    from fbgemm_gpu import open_source  # noqa: F401

    # pyre-ignore[21]
    from test_utils import cpu_and_maybe_gpu
except Exception:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:input_combine")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:input_combine_cpu")
    from fbgemm_gpu.test.test_utils import cpu_and_maybe_gpu

DEFAULT_DEVICE = torch.device("cpu")


class TBEInputPrepareReference(torch.nn.Module):
    def __init__(self, include_last_offsets: List[bool]) -> None:
        super().__init__()
        self.include_last_offsets = include_last_offsets

    def forward(
        self,
        indices_list: List[torch.Tensor],
        offsets_list: List[torch.Tensor],
        per_sample_weights_list: List[torch.Tensor],
        batch_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        size = 0
        assert len(indices_list) > 0
        assert len(indices_list) == len(offsets_list)
        assert len(indices_list) == len(per_sample_weights_list)
        assert len(indices_list) == len(self.include_last_offsets)
        for i in range(len(self.include_last_offsets)):
            size += indices_list[i].size(0)
            assert indices_list[i].dim() == 1
            assert offsets_list[i].dim() == 1
            if per_sample_weights_list[i].numel() > 0:
                assert per_sample_weights_list[i].dim() == 1
                assert indices_list[i].numel() == per_sample_weights_list[i].numel()
        combined_indices = torch.empty(
            size,
            dtype=torch.int32,
            device=indices_list[0].device,
        )
        torch.cat(indices_list, out=combined_indices)
        offsets_starts = torch.zeros(
            [len(offsets_list) + 1],
            dtype=offsets_list[0].dtype,
            device=offsets_list[0].device,
        )
        offsets_accs = torch.zeros(
            [len(offsets_list) + 1],
            dtype=offsets_list[0].dtype,
            device=offsets_list[0].device,
        )

        for i, include_last_offset in enumerate(self.include_last_offsets):
            if include_last_offset:
                offsets_starts[i + 1] = offsets_starts[i] + offsets_list[i].size(0) - 1
            else:
                offsets_starts[i + 1] = offsets_starts[i] + offsets_list[i].size(0)
            offsets_accs[i + 1] = offsets_accs[i] + indices_list[i].size(0)

        assert offsets_accs[-1] == combined_indices.size(0)
        combined_offsets_size: List[int] = (
            [int(offsets_starts[-1].item()) + 1]
            if batch_size is None
            else [batch_size * len(offsets_list) + 1]
        )
        combined_offsets = torch.zeros(
            combined_offsets_size,
            dtype=torch.int32,
            device=offsets_list[0].device,
        )
        if batch_size is None:
            for i in range(len(self.include_last_offsets)):
                combined_offsets[offsets_starts[i] : offsets_starts[i + 1]] = (
                    offsets_list[i][: offsets_starts[i + 1] - offsets_starts[i]]
                    + offsets_accs[i]
                )
        else:
            for i in range(len(self.include_last_offsets)):
                cur_start = batch_size * i
                combined_offsets[
                    cur_start : cur_start + offsets_starts[i + 1] - offsets_starts[i]
                ] = (
                    offsets_list[i][: offsets_starts[i + 1] - offsets_starts[i]]
                    + offsets_accs[i]
                )
                cur_start = cur_start + offsets_starts[i + 1] - offsets_starts[i]
                for j in range(batch_size - offsets_starts[i + 1] + offsets_starts[i]):
                    combined_offsets[cur_start + j] = (
                        indices_list[i].numel() + offsets_accs[i]
                    )
        combined_offsets[-1] = offsets_accs[-1]
        per_sample_weights: Optional[torch.Tensor] = None
        for i in range(len(self.include_last_offsets)):
            if per_sample_weights_list[i].size(0) > 0:
                per_sample_weights = torch.ones(
                    combined_indices.size(0),
                    dtype=per_sample_weights_list[i].dtype,
                    device=per_sample_weights_list[i].device,
                )
                break
        if per_sample_weights is not None:
            for i in range(len(self.include_last_offsets)):
                if per_sample_weights_list[i].size(0) > 0:
                    per_sample_weights[
                        offsets_accs[i] : offsets_accs[i + 1]
                    ] = per_sample_weights_list[i][:]

        # indices and offsets are required to be int32 for TBE
        return combined_indices, combined_offsets, per_sample_weights


class InputCombineTest(unittest.TestCase):
    def _get_inputs(self, dtypes, device=DEFAULT_DEVICE):
        indices_list = [
            torch.tensor([1, 2, 3], dtype=dtypes[0], device=device),
            torch.tensor([1, 2, 3, 4], dtype=dtypes[1], device=device),
        ]
        offsets_list = [
            torch.tensor([0, 2], dtype=dtypes[0], device=device),
            torch.tensor([0, 1, 4], dtype=dtypes[1], device=device),
        ]
        include_last_offsets = [False, True]
        per_sample_weights = [
            torch.tensor([1, 2, 1], dtype=torch.float, device=device),
            torch.tensor([1, 2, 1, 3], dtype=torch.float, device=device),
        ]
        empty_per_sample_weights = [
            torch.tensor([], dtype=torch.float, device=device),
            torch.tensor([], dtype=torch.float, device=device),
        ]
        return (
            indices_list,
            offsets_list,
            per_sample_weights,
            empty_per_sample_weights,
            include_last_offsets,
        )

    def _run_test(self, dtypes) -> None:
        (
            indices_list,
            offsets_list,
            per_sample_weights,
            empty_per_sample_weights,
            include_last_offsets,
        ) = self._get_inputs(dtypes)
        ref_mod = TBEInputPrepareReference(include_last_offsets)

        outputs = torch.ops.fbgemm.tbe_input_combine(
            indices_list,
            offsets_list,
            per_sample_weights,
            torch.BoolTensor(include_last_offsets),
        )
        ref_outputs = ref_mod(indices_list, offsets_list, per_sample_weights)
        for i, j in zip(outputs, ref_outputs):
            torch.testing.assert_close(i, j)
        self.assertTrue(outputs[0].dtype == torch.int32)
        self.assertTrue(outputs[1].dtype == torch.int32)

        outputs = torch.ops.fbgemm.tbe_input_combine(
            indices_list,
            offsets_list,
            empty_per_sample_weights,
            torch.BoolTensor(include_last_offsets),
        )
        ref_outputs = ref_mod(indices_list, offsets_list, empty_per_sample_weights)
        for i, j in zip(outputs[:-1], ref_outputs[:-1]):
            torch.testing.assert_close(i, j)
            self.assertTrue(j.dtype == torch.int32)

        self.assertTrue(outputs[0].dtype == torch.int32)
        self.assertTrue(outputs[1].dtype == torch.int32)
        self.assertTrue(outputs[-1].size(0) == 0)

    def _run_padding_fused_test(self, dtypes, batch_size) -> None:
        (
            indices_list,
            offsets_list,
            per_sample_weights,
            empty_per_sample_weights,
            include_last_offsets,
        ) = self._get_inputs(dtypes)
        ref_mod = TBEInputPrepareReference(include_last_offsets)

        outputs = torch.ops.fbgemm.padding_fused_tbe_input_combine(
            indices_list,
            offsets_list,
            per_sample_weights,
            torch.BoolTensor(include_last_offsets),
            batch_size,
        )
        ref_outputs = ref_mod(
            indices_list, offsets_list, per_sample_weights, batch_size
        )
        for i, j in zip(outputs, ref_outputs):
            torch.testing.assert_close(i, j)
        self.assertTrue(outputs[0].dtype == torch.int32)
        self.assertTrue(outputs[1].dtype == torch.int32)

        outputs = torch.ops.fbgemm.padding_fused_tbe_input_combine(
            indices_list,
            offsets_list,
            empty_per_sample_weights,
            torch.BoolTensor(include_last_offsets),
            batch_size,
        )
        ref_outputs = ref_mod(
            indices_list, offsets_list, empty_per_sample_weights, batch_size
        )
        for i, j in zip(outputs[:-1], ref_outputs[:-1]):
            torch.testing.assert_close(i, j)
            self.assertTrue(j.dtype == torch.int32)

        self.assertTrue(outputs[0].dtype == torch.int32)
        self.assertTrue(outputs[1].dtype == torch.int32)
        self.assertTrue(outputs[-1].size(0) == 0)

    def _offsets_to_lengths(
        self, offsets, indices, include_last_offsets, device=DEFAULT_DEVICE
    ):
        if include_last_offsets:
            offsets_complete = offsets
        else:
            offsets_complete = torch.cat(
                [
                    offsets,
                    torch.tensor([indices.numel()], dtype=offsets.dtype, device=device),
                ]
            )
        return offsets_complete[1:] - offsets_complete[:-1]

    def _run_test_with_length(self, dtypes, device=DEFAULT_DEVICE) -> None:
        (
            indices_list,
            offsets_list,
            per_sample_weights,
            empty_per_sample_weights,
            include_last_offsets,
        ) = self._get_inputs(dtypes, device=device)
        ref_mod = TBEInputPrepareReference(include_last_offsets)

        lengths_list = [
            self._offsets_to_lengths(
                offsets, indices, include_last_offsets, device=device
            )
            for offsets, indices, include_last_offsets in zip(
                offsets_list, indices_list, include_last_offsets
            )
        ]
        outputs = torch.ops.fbgemm.tbe_input_combine_with_length(
            indices_list, lengths_list, per_sample_weights
        )

        ref_outputs = ref_mod(indices_list, offsets_list, per_sample_weights)
        # indices
        self.assertTrue(ref_outputs[0].allclose(outputs[0]))
        # per sample weights
        self.assertTrue(ref_outputs[2].allclose(outputs[2]))

        ref_lengths = self._offsets_to_lengths(ref_outputs[1], ref_outputs[0], True)
        self.assertTrue(ref_lengths.allclose(outputs[1]))

    def _run_padding_fused_test_with_length(self, dtypes, batch_size) -> None:
        (
            indices_list,
            offsets_list,
            per_sample_weights,
            empty_per_sample_weights,
            include_last_offsets,
        ) = self._get_inputs(dtypes)
        ref_mod = TBEInputPrepareReference(include_last_offsets)

        lengths_list = [
            self._offsets_to_lengths(offsets, indices, include_last_offsets)
            for offsets, indices, include_last_offsets in zip(
                offsets_list, indices_list, include_last_offsets
            )
        ]
        outputs = torch.ops.fbgemm.padding_fused_tbe_input_combine_with_length(
            indices_list,
            lengths_list,
            per_sample_weights,
            batch_size,
        )

        ref_outputs = ref_mod(
            indices_list, offsets_list, per_sample_weights, batch_size
        )
        # indices
        self.assertTrue(ref_outputs[0].allclose(outputs[0]))
        # per sample weights
        self.assertTrue(ref_outputs[2].allclose(outputs[2]))

        ref_lengths = self._offsets_to_lengths(ref_outputs[1], ref_outputs[0], True)
        self.assertTrue(ref_lengths.allclose(outputs[1]))

    def test_input_combine_int64(self) -> None:
        self._run_test((torch.int64, torch.int64))

    def test_input_combine_int32(self) -> None:
        self._run_test((torch.int64, torch.int64))

    def test_input_combined_mix(self) -> None:
        self._run_test((torch.int64, torch.int32))

    @given(device=cpu_and_maybe_gpu())
    @settings(deadline=None)
    def test_input_combine_int64_with_length(self, device: torch.device) -> None:
        self._run_test_with_length((torch.int64, torch.int64), device=device)

    @given(device=cpu_and_maybe_gpu())
    @settings(deadline=None)
    def test_input_combine_int32_with_length(self, device: torch.device) -> None:
        self._run_test_with_length((torch.int32, torch.int32), device=device)

    @given(device=cpu_and_maybe_gpu())
    @settings(deadline=None)
    def test_input_combine_mix_with_length(self, device: torch.device) -> None:
        self._run_test_with_length((torch.int64, torch.int32), device=device)

    def test_padding_fused_input_combine_int64(self) -> None:
        self._run_padding_fused_test((torch.int64, torch.int64), 64)

    def test_padding_fused_input_combine_int32(self) -> None:
        self._run_padding_fused_test((torch.int32, torch.int32), 64)

    def test_padding_fused_input_combined_mix(self) -> None:
        self._run_padding_fused_test((torch.int64, torch.int32), 64)

    def test_padding_fused_input_combine_int64_with_length(self) -> None:
        self._run_padding_fused_test_with_length((torch.int64, torch.int64), 64)

    def test_padding_fused_input_combine_int32_with_length(self) -> None:
        self._run_padding_fused_test_with_length((torch.int32, torch.int32), 64)

    def test_padding_fused_input_combined_mix_with_length(self) -> None:
        self._run_padding_fused_test_with_length((torch.int64, torch.int32), 64)


if __name__ == "__main__":
    unittest.main()
