#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from typing import List, Optional, Tuple

import fbgemm_gpu
import torch

# pyre-fixme[16]: Module `fbgemm_gpu` has no attribute `open_source`.
open_source: bool = getattr(fbgemm_gpu, "open_source", False)

if not open_source:
    if torch.version.hip:
        torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:input_combine_hip")
    else:
        torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:input_combine")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:input_combine_cpu")


class TBEInputPrepareReference(torch.nn.Module):
    def __init__(self, include_last_offsets: List[bool]) -> None:
        super().__init__()
        self.include_last_offsets = include_last_offsets

    def forward(  # noqa C901
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
                    # fmt: off
                    per_sample_weights[offsets_accs[i] : offsets_accs[i + 1]] = (
                        per_sample_weights_list[i][:]
                    )
                    # fmt: on

        # indices and offsets are required to be int32 for TBE
        return combined_indices, combined_offsets, per_sample_weights
