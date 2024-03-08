#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import unittest

import torch
from fbgemm_gpu import sparse_ops  # noqa: F401
from hypothesis import given, settings

from .common import open_source, TBEInputPrepareReference

if open_source:
    # pyre-ignore[21]
    from test_utils import cpu_and_maybe_gpu, optests
else:
    from fbgemm_gpu.test.test_utils import cpu_and_maybe_gpu, optests

DEFAULT_DEVICE = torch.device("cpu")


@optests.generate_opcheck_tests()
class InputCombineTest(unittest.TestCase):
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
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

    # pyre-fixme[2]: Parameter must be annotated.
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

    # pyre-fixme[2]: Parameter must be annotated.
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

    # pyre-fixme[3]: Return type must be annotated.
    def _offsets_to_lengths(
        self,
        # pyre-fixme[2]: Parameter must be annotated.
        offsets,
        # pyre-fixme[2]: Parameter must be annotated.
        indices,
        # pyre-fixme[2]: Parameter must be annotated.
        include_last_offsets,
        # pyre-fixme[2]: Parameter must be annotated.
        device=DEFAULT_DEVICE,
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

    # pyre-fixme[2]: Parameter must be annotated.
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

    # pyre-fixme[2]: Parameter must be annotated.
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

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    #  `test_utils.cpu_and_maybe_gpu()` to decorator factory `hypothesis.given`.
    @given(device=cpu_and_maybe_gpu())
    @settings(deadline=None)
    def test_input_combine_int64_with_length(self, device: torch.device) -> None:
        self._run_test_with_length((torch.int64, torch.int64), device=device)

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    #  `test_utils.cpu_and_maybe_gpu()` to decorator factory `hypothesis.given`.
    @given(device=cpu_and_maybe_gpu())
    @settings(deadline=None)
    def test_input_combine_int32_with_length(self, device: torch.device) -> None:
        self._run_test_with_length((torch.int32, torch.int32), device=device)

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    #  `test_utils.cpu_and_maybe_gpu()` to decorator factory `hypothesis.given`.
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
