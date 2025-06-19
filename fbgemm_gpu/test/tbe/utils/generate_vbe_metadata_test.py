#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import unittest
from itertools import accumulate

import torch
from fbgemm_gpu.split_table_batched_embeddings_ops_common import PoolingMode
from fbgemm_gpu.split_table_batched_embeddings_ops_training_common import (
    generate_vbe_metadata,
)

from ..common import open_source

if open_source:
    # pyre-ignore[21]
    from test_utils import gpu_unavailable
else:
    from fbgemm_gpu.test.test_utils import gpu_unavailable

from typing import List


class GenerateVBEMetadataTest(unittest.TestCase):
    def generate_vbe_metadata_ref(
        self,
        offsets: torch.Tensor,
        info_B_num_bits: int,
        batch_size_per_feature_per_rank: List[List[int]],
        output_offset_feature_rank: torch.Tensor,
        feature_dims: torch.Tensor,
    ):
        b_t_map = torch.empty(offsets.numel() - 1, dtype=torch.int)
        row_output_offsets = torch.empty(*b_t_map.shape, dtype=torch.long)
        output_offset_feature_rank_cpu = output_offset_feature_rank.cpu()

        b_t = 0
        num_features = len(batch_size_per_feature_per_rank)
        num_ranks = len(batch_size_per_feature_per_rank[0])
        for t in range(num_features):
            b_offset = 0
            D = feature_dims[t]
            for r in range(num_ranks):
                batch_size = batch_size_per_feature_per_rank[t][r]
                for b in range(batch_size):
                    b_t_map[b_t] = (t << info_B_num_bits) | (b + b_offset)
                    row_output_offsets[b_t] = output_offset_feature_rank_cpu[
                        r * num_features + t
                    ] + (b * D)
                    b_t += 1
                b_offset += batch_size
        return row_output_offsets, b_t_map

    def execute_generate_vbe_metadata_kernel(
        self,
        num_ranks: int,
        num_features: int,
        max_B: int,
    ) -> None:
        # Randomize batch sizes
        batch_size_per_feature_per_rank = torch.randint(
            low=1, high=max_B + 1, size=(num_features, num_ranks)
        ).tolist()
        # Max supported embedding dim is 2048 (which is 512 * 4)
        feature_dims = torch.randint(low=1, high=513, size=(num_features,)) * 4
        # Generate offsets as a sparse tensor because we only need its shape
        total_B = sum(
            [sum(batch_size_per_feature_per_rank[t]) for t in range(num_features)]
        )
        # pyre-ignore: Missing argument [20]
        dummy_offsets = torch.sparse_coo_tensor(
            (total_B + 1,),
            dtype=torch.long,
        )

        # Prepare inputs
        assert dummy_offsets.numel() == total_B + 1
        vbe_metadata = generate_vbe_metadata(
            dummy_offsets,
            batch_size_per_feature_per_rank,
            PoolingMode.SUM,
            feature_dims,
            torch.device("cuda"),
        )
        assert vbe_metadata.max_B_feature_rank > 0
        B_offsets = vbe_metadata.B_offsets
        assert isinstance(B_offsets, torch.Tensor)
        info_B_num_bits, info_B_mask = torch.ops.fbgemm.get_infos_metadata(
            vbe_metadata.B_offsets,
            vbe_metadata.max_B,
            B_offsets.numel() - 1,  # T
        )

        # Run test
        row_output_offsets, b_t_map = torch.ops.fbgemm.generate_vbe_metadata(
            B_offsets,
            vbe_metadata.B_offsets_rank_per_feature,
            vbe_metadata.output_offsets_feature_rank,
            D_offsets=torch.tensor(
                [0] + list(accumulate(feature_dims.tolist())),
                device=torch.device("cuda"),
                dtype=torch.int,
            ),
            D=-1,
            nobag=False,
            max_B_feature_rank=vbe_metadata.max_B_feature_rank,
            info_B_num_bits=info_B_num_bits,
            total_B=dummy_offsets.numel() - 1,
        )

        output_offsets_feature_rank = vbe_metadata.output_offsets_feature_rank
        assert isinstance(output_offsets_feature_rank, torch.Tensor)

        # Run ref
        row_output_offsets_ref, b_t_map_ref = self.generate_vbe_metadata_ref(
            dummy_offsets,
            info_B_num_bits,
            batch_size_per_feature_per_rank,
            output_offsets_feature_rank,
            feature_dims,
        )
        # Compare results
        assert torch.equal(row_output_offsets.cpu(), row_output_offsets_ref)
        assert torch.equal(b_t_map.cpu(), b_t_map_ref)

        # Run CPU test
        assert (
            vbe_metadata.B_offsets_rank_per_feature is not None
            and vbe_metadata.output_offsets_feature_rank is not None
        )
        row_output_offsets_cpu, b_t_map_cpu = torch.ops.fbgemm.generate_vbe_metadata(
            B_offsets.cpu(),
            vbe_metadata.B_offsets_rank_per_feature.cpu(),
            vbe_metadata.output_offsets_feature_rank.cpu(),
            D_offsets=torch.tensor(
                [0] + list(accumulate(feature_dims.tolist())),
                device=torch.device("cpu"),
                dtype=torch.int,
            ),
            D=-1,
            nobag=False,
            max_B_feature_rank=vbe_metadata.max_B_feature_rank,
            info_B_num_bits=info_B_num_bits,
            total_B=dummy_offsets.numel() - 1,
        )
        # Compare results
        assert torch.equal(row_output_offsets.cpu(), row_output_offsets_cpu)
        assert torch.equal(b_t_map.cpu(), b_t_map_cpu)

    @unittest.skipIf(*gpu_unavailable)
    def test_generate_vbe_metadata_kernel(self):
        self.execute_generate_vbe_metadata_kernel(
            num_ranks=128,
            num_features=4,
            max_B=32,
        )

    @unittest.skipIf(*gpu_unavailable)
    def test_generate_vbe_metadata_kernel_large(self):
        self.execute_generate_vbe_metadata_kernel(
            num_ranks=1024,
            num_features=4,
            max_B=128,
        )
