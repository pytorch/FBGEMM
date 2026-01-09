#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

# pyre-ignore-all-errors[56]


import tempfile
import unittest
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from fbgemm_gpu.tbe.ssd import SSDTableBatchedEmbeddingBags
from fbgemm_gpu.tbe.utils import (
    # b_indices,
    get_table_batched_offsets_from_dense,
    round_up,
    # to_device,
)
from hypothesis import assume, given, settings

from .backward_adagrad_common import (  # noqa
    additional_decorators,
    common_settings,
    ComputeDevice,
    EmbeddingLocation,
    gen_mixed_B_batch_sizes,
    gpu_unavailable,
    open_source,
    optests,
    OptimType,
    PoolingMode,
    skipIfRocm,
    SparseType,
    SplitTableBatchedEmbeddingBagsCodegen,
    st,
    TEST_WITH_ROCM,
    WeightDecayMode,
)

if open_source:
    # pyre-ignore[21]
    from test_utils import running_in_oss
else:
    from fbgemm_gpu.test.test_utils import running_in_oss

# Set up test strategy
test_st: Dict[str, Any] = {
    "num_embeddings": st.integers(min_value=2, max_value=5),
    "num_ranks": st.integers(min_value=2, max_value=8),
    "T": st.integers(min_value=1, max_value=5),
    "D": st.integers(min_value=2, max_value=128),
    "B": st.integers(min_value=1, max_value=128),
    "log_E": st.integers(min_value=3, max_value=5),
    "L": st.integers(min_value=0, max_value=20),
    # "weighted": st.booleans(),
    "mixed_D": st.booleans(),
    "pooling_mode": st.sampled_from([PoolingMode.SUM, PoolingMode.MEAN]),
}


class EmbeddingBag:
    T: int
    rows: List[int]
    dims: List[int]
    is_ssd: bool
    op: Union[SplitTableBatchedEmbeddingBagsCodegen, SSDTableBatchedEmbeddingBags]

    def __init__(
        self,
        embedding_specs: Union[
            List[Tuple[int, int, EmbeddingLocation, ComputeDevice]],
            List[Tuple[int, int]],
        ],
        optimizer: OptimType,
        output_dtype: SparseType,
        pooling_mode: PoolingMode,
        weights_precision: SparseType,
        feature_table_map: Optional[List[int]],
        is_ssd: bool = False,
    ):
        self.T = (
            len(embedding_specs)
            if feature_table_map is None
            else len(feature_table_map)
        )
        self.is_ssd = is_ssd
        kwargs: Dict[str, Any] = {}
        if is_ssd:
            emb_op = SSDTableBatchedEmbeddingBags
            kwargs["cache_sets"] = 1
            kwargs["ssd_storage_directory"] = tempfile.mkdtemp()
            (self.rows, self.dims) = zip(*embedding_specs)
        else:
            emb_op = SplitTableBatchedEmbeddingBagsCodegen
            (self.rows, self.dims, _, _) = zip(*embedding_specs)

        self.op = emb_op(
            # pyre-fixme[6]
            embedding_specs=embedding_specs,
            optimizer=optimizer,
            weights_precision=weights_precision,
            output_dtype=output_dtype,
            pooling_mode=pooling_mode,
            feature_table_map=feature_table_map,
            **kwargs,
        )
        if is_ssd:
            # Initialize TBE SSD weights
            for t in range(self.T):
                E = self.rows[t]
                D = self.dims[t]
                ref = torch.rand((E, D), dtype=weights_precision.as_dtype())
                self.op.ssd_db.set_cuda(
                    torch.arange(t * E, (t + 1) * E).to(torch.int64),
                    ref,
                    torch.as_tensor([E]),
                    t,
                )
            # Ensure that `set` (invoked by `set_cuda`) is done
            torch.cuda.synchronize()
        else:
            # pyre-fixme[29]
            self.op.init_embedding_weights_uniform(-1, 1)


def _merge_variable_batch_embeddings(
    embeddings: List[torch.Tensor],
    features: List[int],
    splits: List[List[int]],
    num_ranks: int,
) -> torch.Tensor:
    assert (
        len(embeddings) > 1
    ), f"Expected num_embeddings > 1 but got {len(embeddings)}."
    assert len(splits) > 1, f"Expected number of splits > 1 but got {len(splits)}. "
    split_embs = [e.split(s) for e, s in zip(embeddings, splits)]
    combined_embs = [
        emb
        for rank in range(num_ranks)
        for n, embs in zip(features, split_embs)
        for emb in embs[n * rank : n * rank + n]
    ]
    return torch.cat(combined_embs)


def create_embedding(
    T: int,
    log_E: int,
    D: int,
    mixed_D: bool,
    is_ssd: bool,
    optimizer: OptimType,
    weights_precision: SparseType,
    output_dtype: SparseType,
    pooling_mode: PoolingMode,
    feature_table_map: Optional[List[int]] = None,
    use_cpu: bool = False,
    use_cache: bool = False,
    mixed_uvm: bool = False,
) -> EmbeddingBag:
    rows = [int(10**log_E)] * T
    dims = (
        [
            round_up(np.random.randint(low=int(0.25 * D), high=int(1.0 * D)), 4)
            for _ in range(T)
        ]
        if mixed_D and not is_ssd
        else [D] * T
    )
    if is_ssd:
        embedding_specs = [(E, D) for (E, D) in zip(rows, dims)]
    else:
        compute_device = ComputeDevice.CUDA
        if use_cpu:
            managed = [EmbeddingLocation.HOST] * T
            compute_device = ComputeDevice.CPU
        elif TEST_WITH_ROCM:
            # ROCm managed memory allocation is under development
            managed = [EmbeddingLocation.DEVICE] * T
        elif use_cache:
            managed = [EmbeddingLocation.MANAGED_CACHING] * T
            if mixed_D:
                average_D = sum(dims) // T
                for t, d in enumerate(dims):
                    managed[t] = (
                        EmbeddingLocation.DEVICE if d < average_D else managed[t]
                    )
        elif mixed_uvm:
            managed = [
                np.random.choice(
                    [
                        EmbeddingLocation.DEVICE,
                        EmbeddingLocation.MANAGED,
                    ]
                )
                for _ in range(T)
            ]
        else:
            managed = [EmbeddingLocation.DEVICE] * T
        embedding_specs = [
            (E, D, M, compute_device) for (E, D, M) in zip(rows, dims, managed)
        ]
    return EmbeddingBag(
        embedding_specs=embedding_specs,
        optimizer=optimizer,
        weights_precision=weights_precision,
        output_dtype=output_dtype,
        pooling_mode=pooling_mode,
        feature_table_map=feature_table_map,
        is_ssd=is_ssd,
    )


# @optests.generate_opcheck_tests(fast=True, additional_decorators=additional_decorators)
class MergeVBETest(unittest.TestCase):

    def get_vbe_splits_and_offsets(
        self,
        embedding_list: List[EmbeddingBag],
        Bs_feature_rank_list: List[List[List[int]]],
        num_ranks: int,
    ) -> Tuple[int, List[List[int]], List[List[List[int]]]]:
        """
        Calculate split size list, total VBE output size and offsets.

        Args:
            embedding_list (List[EmbeddingBag]):
                List of TBEs/SSD TBEs
            Bs_feature_rank_list (List[List[List[int]]]):
                List of batch_size_per_feature_per_rank where Bs_feature_rank_list[i]
                is batch_size_per_feature_per_rank of the ith embedding table
            num_ranks (int):
                number of ranks (i.e., world_size)
        Return:
            total_output_size (int):
                total VBE output size for all embeddings
            splits_list (List[List[int]]):
                list of split size where splits_list[i] contains output size for each
                batch in embedding table i
            vbe_output_offset_list List[List[int]]):
                list of vbe_output_offset where vbe_output_offset_list[i] contains
                offsets for embedding table i. vbe_output_offset has a shape of
                [num_ranks, num_features].
        """
        splits_list = []
        vbe_offsets_list = []
        total_output_size = 0
        # get output size for each batch
        for i, emb in enumerate(embedding_list):
            batch_size_per_feature_per_rank = Bs_feature_rank_list[i]
            batch_size_per_rank_per_feature = list(
                zip(*batch_size_per_feature_per_rank)
            )
            output_size_per_rank = [
                b * dim
                for batch_size_per_rank in batch_size_per_rank_per_feature
                for b, dim in zip(batch_size_per_rank, emb.dims)
            ]
            total_output_size += sum(output_size_per_rank)
            splits_list.append(output_size_per_rank)
            vbe_offsets_list.append([])

        # calculate offsets
        global_offset = 0
        for rank_id in range(num_ranks):
            for i, emb in enumerate(embedding_list):
                output_sizes = splits_list[i]
                # offset within embedding table
                local_offset = 0
                output_offsets = []
                local_output_sizes = output_sizes[
                    emb.T * rank_id : emb.T * rank_id + emb.T
                ]
                for size in local_output_sizes:
                    output_offsets.append(local_offset + global_offset)
                    local_offset += size
                vbe_offsets_list[i].append(output_offsets)
                global_offset += sum(local_output_sizes)

        return total_output_size, splits_list, vbe_offsets_list

    def execute_merge_vbe(
        self,
        num_embeddings: int,
        num_ssd: int,
        num_ranks: int,
        T: int,
        log_E: int,
        D: int,
        B: int,
        L: int,
        mixed_D: bool,
        optimizer: OptimType,
        weights_precision: SparseType,
        output_dtype: SparseType,
        pooling_mode: PoolingMode,
        use_cpu: bool = False,
        features_list: Optional[list[int]] = None,
        dims_list: Optional[list[int]] = None,
        Bs_feature_rank_list: Optional[List[List[List[int]]]] = None,
    ):
        """
        Execute merge VBE

        Args:

        Return:
            None
        """
        assume(num_ssd <= num_embeddings)
        assert not use_cpu, "CPU is not supported"

        # SSD does not support BF16
        assume(num_ssd == 0 or weights_precision != SparseType.BF16)
        assume(num_ssd == 0 or output_dtype != SparseType.BF16)

        # TODO: use torch.accelerator.current_accelerator() to include MTIA
        device = (
            torch.device("cpu") if use_cpu else torch.accelerator.current_accelerator()
        )
        if features_list is None:
            features_list = [
                np.random.randint(low=1, high=T + 1) for _ in range(num_embeddings)
            ]
        assert (
            len(features_list) == num_embeddings
        ), f"Expected the size of features_list to be {num_embeddings} but got {len(features_list)}"
        if dims_list is None:
            dims_list = [
                np.random.randint(low=1, high=D + 1) * 4 for _ in range(num_embeddings)
            ]
        assert (
            len(dims_list) == num_embeddings
        ), f"Expected the size of dims_list to be {num_embeddings} but got {len(dims_list)}"

        # create a list of embedding tables
        embedding_list = []
        ssd_counter = num_ssd
        for i in range(num_embeddings):
            is_ssd = False
            if ssd_counter > 0:
                is_ssd = True
                ssd_counter -= 1
            embedding_list.append(
                create_embedding(
                    T=features_list[i],
                    log_E=log_E,
                    D=dims_list[i],
                    mixed_D=mixed_D,
                    is_ssd=is_ssd,
                    use_cpu=use_cpu,
                    optimizer=optimizer,
                    weights_precision=weights_precision,
                    output_dtype=output_dtype,
                    pooling_mode=pooling_mode,
                )
            )
        # get batch size per feature per rank for each TBE
        Bs_list = []
        if Bs_feature_rank_list is None:
            Bs_feature_rank_list = []
            for i in range(num_embeddings):
                Bs_feature_rank, Bs = gen_mixed_B_batch_sizes(
                    B, features_list[i], num_ranks=num_ranks
                )
                Bs_feature_rank_list.append(Bs_feature_rank)
                Bs_list.append(Bs)
        else:
            assert (
                len(Bs_feature_rank_list) == num_embeddings
            ), f"Expected the size of Bs_feature_rank_list to be {num_embeddings} but got {len(Bs_feature_rank_list)}"
            for i, Bs_feature_rank in enumerate(Bs_feature_rank_list):
                assert (
                    len(Bs_feature_rank) == features_list[i]
                ), f"Expected dim0 of Bs_feature_rank to be number of features = {features_list[i]} but got {len(Bs_feature_rank[0])}. {Bs_feature_rank[0]} "
                assert (
                    len(Bs_feature_rank[0]) == num_ranks
                ), f"Expected dim1 of Bs_feature_rank to be number of ranks = {num_ranks} but got {len(Bs_feature_rank[1])}. {Bs_feature_rank[1]} "
                Bs = [sum(Bs_feature) for Bs_feature in Bs_feature_rank]
                Bs_list.append(Bs)

        # get splits and offsets
        total_output_size, splits_list, vbe_offsets_list = (
            self.get_vbe_splits_and_offsets(
                embedding_list,
                Bs_feature_rank_list,
                num_ranks,
            )
        )

        # create vbe_output
        vbe_output = torch.empty(
            [total_output_size], device=device, dtype=output_dtype.as_dtype()
        )

        # For each embedding, perform TBE forward
        output_list = []
        output_ref_list = []

        for i, emb in enumerate(embedding_list):
            # generate indices and offsets
            xs = [
                torch.from_numpy(
                    np.random.choice(
                        range(emb.rows[t]), size=(b, L), replace=True
                    ).astype(np.int64)
                )
                for t, b in zip(emb.op.feature_table_map, Bs_list[i])
            ]
            x = torch.cat([x.contiguous().flatten() for x in xs], dim=0)
            (indices, offsets) = get_table_batched_offsets_from_dense(
                x, L, sum(Bs_list[i])
            )
            indices = indices.to(device)
            offsets = offsets.to(device)

            # call TBE forward with vbe output and offsets.
            emb_output = emb.op(
                indices=indices,
                offsets=offsets,
                batch_size_per_feature_per_rank=Bs_feature_rank_list[i],
                vbe_output=vbe_output,
                vbe_output_offsets=torch.tensor(vbe_offsets_list[i], device=device),
            )

            # call regular TBE forward as reference implementation
            emb_output_ref = emb.op(
                indices=indices,
                offsets=offsets,
                batch_size_per_feature_per_rank=Bs_feature_rank_list[i],
            )
            output_list.append(emb_output)
            output_ref_list.append(emb_output_ref)

        # merge outputs for the reference
        vbe_output_ref = _merge_variable_batch_embeddings(
            output_ref_list, features_list, splits_list, num_ranks
        )
        vbe_output = vbe_output.flatten()

        # the final output and the merged output of the reference needs to be the same
        if not torch.equal(vbe_output, vbe_output_ref):
            diff = torch.nonzero(vbe_output != vbe_output_ref, as_tuple=True)[
                0
            ].unique()
            print(
                f"Mistmatches: {diff.shape} ({(diff.numel() / vbe_output.numel()) * 100}%)\n"
            )
        assert torch.equal(vbe_output, vbe_output_ref)

    @unittest.skipIf(*gpu_unavailable)
    @unittest.skipIf(*running_in_oss)
    @given(
        num_ssd=st.integers(min_value=1, max_value=5),
        weights_precision=st.sampled_from([SparseType.FP32, SparseType.FP16]),
        output_dtype=st.sampled_from([SparseType.FP32, SparseType.FP16]),
        **test_st,
    )
    @settings(**common_settings)
    def test_merge_VBE_SSD(  # noqa C901
        self,
        num_ssd: int,
        weights_precision: SparseType,
        output_dtype: SparseType,
        **kwargs: Any,
    ) -> None:
        """
        Test merge VBE between TBEs and SSD TBEs
        """
        self.execute_merge_vbe(
            num_ssd=num_ssd,
            optimizer=OptimType.EXACT_ROWWISE_ADAGRAD,
            output_dtype=output_dtype,
            weights_precision=weights_precision,
            features_list=None,
            dims_list=None,
            Bs_feature_rank_list=None,
            **kwargs,
        )

    @unittest.skipIf(*gpu_unavailable)
    @given(
        optimizer=st.sampled_from(
            [
                OptimType.EXACT_ROWWISE_ADAGRAD,
                OptimType.EXACT_SGD,
                OptimType.ADAM,
            ]
        ),
        weights_precision=st.sampled_from([SparseType.FP32, SparseType.FP16]),
        output_dtype=st.sampled_from(
            [SparseType.FP32, SparseType.FP16, SparseType.BF16]
        ),
        **test_st,
    )
    @settings(**common_settings)
    def test_merge_VBE_nonSSD(  # noqa C901
        self,
        optimizer,
        weights_precision,
        output_dtype,
        **kwargs: Any,
    ) -> None:
        """
        Test merge VBE between non-SSD TBEs
        """
        self.execute_merge_vbe(
            num_ssd=0,
            optimizer=optimizer,
            weights_precision=weights_precision,
            output_dtype=output_dtype,
            features_list=None,
            dims_list=None,
            Bs_feature_rank_list=None,
            **kwargs,
        )

    @unittest.skipIf(*gpu_unavailable)
    @unittest.skipIf(*running_in_oss)
    @given(
        num_ssd=st.integers(min_value=0, max_value=2),
    )
    @settings(**common_settings)
    def test_merge_VBE_simple(  # noqa C901
        self,
        num_ssd,
    ) -> None:
        """
        Test merge VBE simple case
        """
        num_embeddings = 2
        num_ranks = 2
        features_list = [2, 1]
        dims_list = [16, 40]
        Bs_feature_rank_list = [[[2, 2], [2, 2]], [[7, 6]]]
        weights_precision = SparseType.FP32
        optimizer = OptimType.EXACT_ROWWISE_ADAGRAD
        output_dtype = SparseType.FP32
        pooling_mode = PoolingMode.SUM
        self.execute_merge_vbe(
            num_embeddings,
            num_ssd,
            num_ranks,
            T=1,
            log_E=1,
            D=1,
            B=1,
            L=1,
            mixed_D=False,
            optimizer=optimizer,
            weights_precision=weights_precision,
            output_dtype=output_dtype,
            pooling_mode=pooling_mode,
            features_list=features_list,
            dims_list=dims_list,
            Bs_feature_rank_list=Bs_feature_rank_list,
        )


if __name__ == "__main__":
    unittest.main()
