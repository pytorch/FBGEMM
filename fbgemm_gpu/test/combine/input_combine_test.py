#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import unittest

import fbgemm_gpu  # noqa: F401
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

    def _get_prepadded_inputs(
        self,
        dtypes: list[torch.dtype],
        device: torch._C.device = DEFAULT_DEVICE,
        include_last: bool = True,
    ) -> tuple[
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
        list[bool],
    ]:
        indices_list = [
            torch.tensor([1, 2, 3, 123, 123, 123], dtype=dtypes[0], device=device),
            torch.tensor([1, 2, 3, 4, 456, 456, 456], dtype=dtypes[1], device=device),
        ]
        offsets_list = [
            torch.tensor([0, 2, 3], dtype=dtypes[0], device=device),
            torch.tensor([0, 1, 4], dtype=dtypes[1], device=device),
        ]
        # One of the offsets tensor is always with the last offset
        include_last_offsets = [True, include_last]
        per_sample_weights = [
            torch.tensor([1, 2, 1, 0, 0, 0], dtype=torch.float, device=device),
            torch.tensor([1, 2, 1, 3, 0, 0, 0], dtype=torch.float, device=device),
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

    def _run_test_with_prepadded_indices_weights(self) -> None:
        """
        When indices tensors are having paddings and the offsets tensors are all
        with the last offset, we should expect the outputs will have values in
        the front with paddings in the end.
        """
        dtypes = [torch.int64, torch.int64]
        (
            indices_list,
            offsets_list,
            per_sample_weights,
            empty_per_sample_weights,
            include_last_offsets,
        ) = self._get_prepadded_inputs(dtypes, include_last=True)

        outputs = torch.ops.fbgemm.tbe_input_combine(
            indices_list,
            offsets_list,
            per_sample_weights,
            torch.BoolTensor(include_last_offsets),
        )
        expected_outputs = [
            torch.tensor(
                [1, 2, 3, 1, 2, 3, 4, 123, 123, 123, 456, 456, 456], dtype=torch.int32
            ),
            torch.tensor([0, 2, 3, 4, 7], dtype=torch.int32),
            torch.tensor(
                [1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            ),
        ]
        for i, j in zip(outputs, expected_outputs):
            torch.testing.assert_close(i, j)
        self.assertTrue(outputs[0].dtype == torch.int32)
        self.assertTrue(outputs[1].dtype == torch.int32)

        outputs = torch.ops.fbgemm.tbe_input_combine(
            indices_list,
            offsets_list,
            empty_per_sample_weights,
            torch.BoolTensor(include_last_offsets),
        )
        expected_outputs = [
            torch.tensor(
                [1, 2, 3, 1, 2, 3, 4, 123, 123, 123, 456, 456, 456], dtype=torch.int32
            ),
            torch.tensor([0, 2, 3, 4, 7], dtype=torch.int32),
            torch.empty(0),
        ]
        for i, j in zip(outputs, expected_outputs):
            torch.testing.assert_close(i, j)
        self.assertTrue(outputs[0].dtype == torch.int32)
        self.assertTrue(outputs[1].dtype == torch.int32)
        self.assertTrue(outputs[2].size(0) == 0)

    def _run_test_with_prepadded_indices_weights_without_last_offsets(self) -> None:
        """
        When indices tensors are having paddings and there is at least one offsets
        tensor doesn't have the last offset, we should expect the outputs will be as
        previously.
        """
        dtypes = [torch.int64, torch.int64]
        (
            indices_list,
            offsets_list,
            per_sample_weights,
            empty_per_sample_weights,
            include_last_offsets,
        ) = self._get_prepadded_inputs(dtypes, include_last=False)
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
        ref_outputs = ref_mod(indices_list, offsets_list, per_sample_weights)
        for i, j in zip(outputs[:-1], ref_outputs[:-1]):
            torch.testing.assert_close(i, j)
        self.assertTrue(outputs[0].dtype == torch.int32)
        self.assertTrue(outputs[1].dtype == torch.int32)
        self.assertTrue(outputs[2].size(0) == 0)

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

    def test_tbe_input_combine_cpu_with_padded_indices(self) -> None:
        self._run_test_with_prepadded_indices_weights()

    def test_tbe_input_combine_cpu_with_padded_indices_without_last_offsets(
        self,
    ) -> None:
        self._run_test_with_prepadded_indices_weights_without_last_offsets()

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

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    #  `torch.cuda.is_available()` to decorator factory `unittest.skipUnless`.
    @unittest.skipUnless(torch.cuda.is_available(), "GPU not available")
    def test_tbe_input_combine_with_length_correctness_large(self) -> None:
        """
        Correctness check for the refactored
        ``tbe_input_combine_with_length_kernel`` (D105204171), which
        lacked its own test method when landed.

        The production fix wraps the kernel body in a grid-stride loop
        over ``(list_id, warp_id)`` pairs, adds the ``total_warp_work``
        parameter, and caps ``grid_x`` on ROCm via ``cap_grid_dim_x``.
        This test exercises a multi-list, multi-block configuration and
        compares GPU output against the CPU dispatch, guarding the new
        kernel signature and the grid-stride index math.

        Note: at this scale the grid is small (a handful of blocks), so
        the ROCm ``OverflowOnly`` cap does not engage and the grid-stride
        outer loop executes once. Actually tripping the 2^32-thread cap
        would require ``num_lists * max_list_size >= ~2^37`` (a multi-GB
        allocation), which is infeasible in a unit test; the cap/overflow
        boundary is therefore covered by the host-side clamp logic, not
        this test.
        """
        # Many small lists so total_warp_work is non-trivial but
        # memory stays modest. Each list contributes one warp of work.
        num_lists = 256
        list_len = 64
        batch_size = 32

        indices_list = [
            torch.arange(list_len, dtype=torch.int32) for _ in range(num_lists)
        ]
        # Equal-length offsets per list.
        lengths_list = [
            torch.full(
                (batch_size,),
                list_len // batch_size,
                dtype=torch.int32,
            )
            for _ in range(num_lists)
        ]
        # Trim the last segment to match list_len.
        for lengths in lengths_list:
            lengths[-1] = list_len - (list_len // batch_size) * (batch_size - 1)
        # Non-empty per-sample weights (numel must match each list's indices)
        # so the kernel's weights-copy branch is exercised too.
        per_sample_weights = [
            torch.ones(list_len, dtype=torch.float) for _ in range(num_lists)
        ]

        # Reference is the CPU dispatch of the SAME op. Only
        # ``tbe_input_combine_with_length`` has a CUDA kernel (the one this
        # diff modified); ``padding_fused_*`` is CPU-only, so the GPU dispatch
        # must go through the non-fused op.
        out_cpu = torch.ops.fbgemm.tbe_input_combine_with_length(
            indices_list,
            lengths_list,
            per_sample_weights,
        )

        # GPU op under test.
        device = torch.device(torch.accelerator.current_accelerator() or "cuda")
        out_gpu = torch.ops.fbgemm.tbe_input_combine_with_length(
            [t.to(device) for t in indices_list],
            [t.to(device) for t in lengths_list],
            [t.to(device) for t in per_sample_weights],
        )

        # Compare combined indices, lengths, and per-sample weights.
        self.assertEqual(len(out_cpu), len(out_gpu))
        for cpu_t, gpu_t in zip(out_cpu, out_gpu):
            torch.testing.assert_close(gpu_t.cpu(), cpu_t)


if __name__ == "__main__":
    unittest.main()
