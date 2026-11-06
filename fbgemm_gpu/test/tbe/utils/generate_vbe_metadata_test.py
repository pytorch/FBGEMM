#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import unittest
from itertools import accumulate

import hypothesis.strategies as st
import torch
from fbgemm_gpu.split_table_batched_embeddings_ops_training_common import (
    generate_vbe_metadata,
)
from fbgemm_gpu.tbe.config.embedding_config import PoolingMode
from hypothesis import given, settings

from ..common import MAX_EXAMPLES, open_source

if open_source:
    # pyre-ignore[21]
    from test_utils import gpu_unavailable
else:
    from fbgemm_gpu.test.test_utils import gpu_unavailable


class GenerateVBEMetadataTest(unittest.TestCase):
    def generate_vbe_metadata_ref(
        self,
        offsets: torch.Tensor,
        info_B_num_bits: int,
        batch_size_per_feature_per_rank: list[list[int]],
        output_offset_feature_rank: torch.Tensor,
        feature_dims: torch.Tensor,
    ):
        b_t_map = torch.empty(offsets.numel() - 1, dtype=torch.uint32)
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
        compute_row_metadata: bool = False,
        batch_size_per_feature_per_rank: list[list[int]] | None = None,
        feature_dims: torch.Tensor | None = None,
        info_B_num_bits: int | None = None,
    ) -> None:
        """
        Check `generate_vbe_metadata` on CUDA and CPU against the reference.
        With `compute_row_metadata`, the CUDA outputs come from the helper's
        `VBEMetadata`, otherwise from the op directly. `info_B_num_bits`
        defaults to the value from `get_infos_metadata`.
        """
        device = torch.accelerator.current_accelerator()
        assert device is not None
        # Randomize batch sizes unless given
        batch_size_per_feature_per_rank = (
            torch.randint(
                low=1, high=max_B + 1, size=(num_features, num_ranks)
            ).tolist()
            if batch_size_per_feature_per_rank is None
            else batch_size_per_feature_per_rank
        )
        # Max supported embedding dim is 2048 (which is 512 * 4)
        feature_dims = (
            torch.randint(low=1, high=513, size=(num_features,)) * 4
            if feature_dims is None
            else feature_dims
        )
        # Generate offsets as a sparse tensor because we only need its shape
        total_B = sum(
            [sum(batch_size_per_feature_per_rank[t]) for t in range(num_features)]
        )
        # pyre-ignore: Missing argument [20]
        dummy_offsets = torch.sparse_coo_tensor(
            # pyrefly: ignore [bad-argument-type]
            (total_B + 1,),
            dtype=torch.long,
        )

        # Prepare inputs
        assert dummy_offsets.numel() == total_B + 1
        D_offsets = torch.tensor(
            [0] + list(accumulate(feature_dims.tolist())),
            device=device,
            dtype=torch.int,
        )
        info_B_num_bits = (
            torch.ops.fbgemm.get_infos_metadata(
                D_offsets,  # unused
                max(map(sum, batch_size_per_feature_per_rank)),  # max_B
                num_features,  # T
            )[0]
            if info_B_num_bits is None
            else info_B_num_bits
        )

        # Run test
        vbe_metadata = generate_vbe_metadata(
            dummy_offsets,
            batch_size_per_feature_per_rank,
            PoolingMode.SUM,
            feature_dims,
            device,
            compute_row_metadata=compute_row_metadata,
            D_offsets=D_offsets,
            info_B_num_bits=info_B_num_bits,
        )
        assert vbe_metadata.max_B_feature_rank > 0
        B_offsets = vbe_metadata.B_offsets
        assert isinstance(B_offsets, torch.Tensor)
        row_output_offsets = vbe_metadata.row_output_offsets
        b_t_map = vbe_metadata.b_t_map
        if not compute_row_metadata:
            # The helper leaves both fields None when the bool is off
            assert row_output_offsets is None and b_t_map is None
            row_output_offsets, b_t_map = torch.ops.fbgemm.generate_vbe_metadata(
                B_offsets,
                vbe_metadata.B_offsets_rank_per_feature,
                vbe_metadata.output_offsets_feature_rank,
                D_offsets=D_offsets,
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
        # Compare results. The op stores the packed uint32 values in an int32
        # tensor, so a value with bit 31 set reads back as a negative int32.
        assert row_output_offsets is not None and b_t_map is not None
        assert b_t_map.dtype == torch.int32
        assert torch.equal(row_output_offsets.cpu(), row_output_offsets_ref)
        assert torch.equal(b_t_map.cpu().view(torch.uint32), b_t_map_ref)

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
        assert b_t_map_cpu.dtype == torch.int32
        assert torch.equal(row_output_offsets.cpu(), row_output_offsets_cpu)
        assert torch.equal(b_t_map.cpu(), b_t_map_cpu)

    # pyrefly: ignore [bad-argument-type]
    @unittest.skipIf(*gpu_unavailable)
    @given(
        num_ranks=st.integers(min_value=1, max_value=64),
        num_features=st.integers(min_value=1, max_value=8),
        max_B=st.integers(min_value=1, max_value=32),
        # Max supported embedding dim is 2048 (which is 512 * 4)
        D=st.integers(min_value=1, max_value=512),
        mixed_D=st.booleans(),
        compute_row_metadata=st.booleans(),
        data=st.data(),
    )
    @settings(max_examples=MAX_EXAMPLES, deadline=None)
    def test_generate_vbe_metadata_kernel(
        self,
        num_ranks: int,
        num_features: int,
        max_B: int,
        D: int,
        mixed_D: bool,
        compute_row_metadata: bool,
        data: st.DataObject,
    ) -> None:
        batch_size_per_feature_per_rank = data.draw(
            st.lists(
                st.lists(
                    st.integers(min_value=1, max_value=max_B),
                    min_size=num_ranks,
                    max_size=num_ranks,
                ),
                min_size=num_features,
                max_size=num_features,
            )
        )
        Ds = (
            data.draw(
                st.lists(
                    st.integers(min_value=1, max_value=D),
                    min_size=num_features,
                    max_size=num_features,
                )
            )
            if mixed_D
            else [D] * num_features
        )
        feature_dims = torch.tensor(Ds) * 4
        self.execute_generate_vbe_metadata_kernel(
            num_ranks=num_ranks,
            num_features=num_features,
            max_B=max_B,
            batch_size_per_feature_per_rank=batch_size_per_feature_per_rank,
            feature_dims=feature_dims,
            compute_row_metadata=compute_row_metadata,
        )

    # pyrefly: ignore [bad-argument-type]
    @unittest.skipIf(*gpu_unavailable)
    def test_generate_vbe_metadata_kernel_large(self):
        self.execute_generate_vbe_metadata_kernel(
            num_ranks=1024,
            num_features=4,
            max_B=128,
        )

    # pyrefly: ignore [bad-argument-type]
    @unittest.skipIf(*gpu_unavailable)
    def test_generate_vbe_metadata_kernel_full_32bit_b_t_map(self) -> None:
        """
        Check `b_t_map` when `t` and `b` together fill all 32 bits. With
        `info_B_num_bits = 17` there are 2**15 features, and the last one has a
        batch of 2**17, so its last row packs t = 2**15 - 1 into the top 15 bits
        and b = 2**17 - 1 into the low 17, i.e. 0xFFFFFFFF.

        `info_B_num_bits` is passed in because `get_infos_metadata` would keep
        more bits for t with this many features. T = 2**15 stays within the
        CUDA grid's z-dimension limit (one block per feature).
        """
        info_B_num_bits = 17
        num_features = 1 << (32 - info_B_num_bits)
        max_B = 1 << info_B_num_bits
        for compute_row_metadata in (False, True):
            with self.subTest(compute_row_metadata=compute_row_metadata):
                self.execute_generate_vbe_metadata_kernel(
                    num_ranks=1,
                    num_features=num_features,
                    max_B=max_B,
                    batch_size_per_feature_per_rank=[[1]] * (num_features - 1)
                    + [[max_B]],
                    feature_dims=torch.full((num_features,), 4),
                    compute_row_metadata=compute_row_metadata,
                    info_B_num_bits=info_B_num_bits,
                )
