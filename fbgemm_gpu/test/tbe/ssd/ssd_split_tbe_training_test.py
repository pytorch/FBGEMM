# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[3,6,56]

import unittest
from enum import Enum

from typing import Any, Dict, List, Optional, Tuple

import hypothesis.strategies as st
import numpy as np
import torch
from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType, SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    BoundsCheckMode,
    PoolingMode,
)
from fbgemm_gpu.tbe.ssd import SSDTableBatchedEmbeddingBags
from fbgemm_gpu.tbe.utils import b_indices, get_table_batched_offsets_from_dense
from hypothesis import assume, given, settings, Verbosity

from .. import common  # noqa E402
from ..common import gen_mixed_B_batch_sizes, gpu_unavailable, running_in_oss


MAX_EXAMPLES = 40
MAX_PIPELINE_EXAMPLES = 10

default_st: Dict["str", Any] = {
    "T": st.integers(min_value=1, max_value=10),
    "D": st.integers(min_value=2, max_value=128),
    "B": st.integers(min_value=1, max_value=128),
    "log_E": st.integers(min_value=3, max_value=5),
    "L": st.integers(min_value=0, max_value=20),
    "weighted": st.booleans(),
    "cache_set_scale": st.sampled_from([0.0, 0.005, 1]),
    "pooling_mode": st.sampled_from(
        [PoolingMode.NONE, PoolingMode.SUM, PoolingMode.MEAN]
    ),
    "weights_precision": st.sampled_from([SparseType.FP32, SparseType.FP16]),
    "output_dtype": st.sampled_from([SparseType.FP32, SparseType.FP16]),
    "share_table": st.booleans(),
    "trigger_bounds_check": st.booleans(),
    "mixed_B": st.booleans(),
}


class PrefetchLocation(Enum):
    BEFORE_FWD = 1
    BETWEEN_FWD_BWD = 2


class FlushLocation(Enum):
    AFTER_FWD = 1
    AFTER_BWD = 2
    BEFORE_TRAINING = 3
    ALL = 4


@unittest.skipIf(*running_in_oss)
@unittest.skipIf(*gpu_unavailable)
class SSDSplitTableBatchedEmbeddingsTest(unittest.TestCase):
    def get_physical_table_arg_indices_(self, feature_table_map: List[int]):
        """
        Get the physical table arg indices for the reference and TBE.  The
        first element in each tuple is for accessing the reference embedding
        list.  The second element is for accessing TBE data.

        Example:
            feature_table_map = [0, 1, 2, 2, 3, 4]
            This function returns [(0, 0), (1, 1), (2, 2), (4, 3), (5, 4)]
        """
        ref_arg_indices = []
        test_arg_indices = []
        prev_t = -1
        for f, t in enumerate(feature_table_map):
            # Only get the physical tables
            if prev_t != t:
                prev_t = t
                ref_arg_indices.append(f)
                test_arg_indices.append(t)
        return zip(ref_arg_indices, test_arg_indices)

    @given(
        weights_precision=st.sampled_from([SparseType.FP32, SparseType.FP16]),
        indice_int64_t=st.sampled_from([True, False]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_ssd(self, indice_int64_t: bool, weights_precision: SparseType) -> None:
        import tempfile

        E = int(1e4)
        D = 128
        N = 100
        if indice_int64_t:
            indices = torch.as_tensor(
                np.random.choice(E, replace=False, size=(N,)), dtype=torch.int64
            )
        else:
            indices = torch.as_tensor(
                np.random.choice(E, replace=False, size=(N,)), dtype=torch.int32
            )
        weights = torch.randn(N, D, dtype=weights_precision.as_dtype())
        output_weights = torch.empty_like(weights)
        count = torch.tensor([N])

        feature_table_map = list(range(1))
        emb = SSDTableBatchedEmbeddingBags(
            embedding_specs=[(E, D)],
            feature_table_map=feature_table_map,
            ssd_storage_directory=tempfile.mkdtemp(),
            cache_sets=1,
            ssd_uniform_init_lower=-0.1,
            ssd_uniform_init_upper=0.1,
            weights_precision=weights_precision,
            l2_cache_size=8,
        )
        emb.ssd_db.get_cuda(indices, output_weights, count)
        torch.cuda.synchronize()
        assert (output_weights <= 0.1).all().item()
        assert (output_weights >= -0.1).all().item()

        emb.ssd_db.set_cuda(indices, weights, count, 1)
        emb.ssd_db.get_cuda(indices, output_weights, count)
        torch.cuda.synchronize()
        torch.testing.assert_close(weights, output_weights)

    def generate_inputs_(
        self,
        B: int,
        L: int,
        Es: List[int],
        feature_table_map: List[int],
        weights_precision: SparseType = SparseType.FP32,
        trigger_bounds_check: bool = False,
        mixed_B: bool = False,
    ) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[List[List[int]]],
    ]:
        """
        Generate indices and per sample weights
        """
        T = len(feature_table_map)

        Bs = [B] * T
        Bs_rank_feature = [[0]]
        if mixed_B:
            Bs_rank_feature, Bs = gen_mixed_B_batch_sizes(B, T)

        # Generate random indices and per sample weights
        indices_list = [
            torch.randint(
                low=0, high=Es[t] * (2 if trigger_bounds_check else 1), size=(b, L)
            ).cuda()
            for (b, t) in zip(Bs, feature_table_map)
        ]
        per_sample_weights_list = [torch.randn(size=(b, L)).cuda() for b in Bs]

        # Concat inputs for SSD TBE
        indices = torch.cat(
            [indices.flatten() for indices in indices_list],
            dim=0,
        )
        per_sample_weights: torch.Tensor = torch.cat(
            [
                per_sample_weights.flatten()
                for per_sample_weights in per_sample_weights_list
            ],
            dim=0,
        )
        (indices, offsets) = get_table_batched_offsets_from_dense(indices, L, sum(Bs))

        batch_size_per_feature_per_rank = Bs_rank_feature if mixed_B else None

        if trigger_bounds_check:
            # Manual bounds check
            for f, (t, indices_ref) in enumerate(zip(feature_table_map, indices_list)):
                indices_ref[indices_ref >= Es[t]] = 0
                indices_list[f] = indices_ref

        return (
            indices_list,
            per_sample_weights_list,
            indices.cuda(),
            offsets.cuda(),
            per_sample_weights.contiguous().view(-1).cuda(),
            batch_size_per_feature_per_rank,
        )

    def generate_ssd_tbes(
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        weighted: bool,
        lr: float = 0.01,  # from SSDTableBatchedEmbeddingBags
        eps: float = 1.0e-8,  # from SSDTableBatchedEmbeddingBags
        ssd_shards: int = 1,  # from SSDTableBatchedEmbeddingBags
        optimizer: OptimType = OptimType.EXACT_ROWWISE_ADAGRAD,
        cache_set_scale: float = 1.0,
        # pyre-fixme[9]: pooling_mode has type `bool`; used as `PoolingMode`.
        pooling_mode: bool = PoolingMode.SUM,
        weights_precision: SparseType = SparseType.FP32,
        output_dtype: SparseType = SparseType.FP32,
        stochastic_rounding: bool = True,
        share_table: bool = False,
        prefetch_pipeline: bool = False,
        bulk_init_chunk_size: int = 0,
    ) -> Tuple[SSDTableBatchedEmbeddingBags, List[torch.nn.EmbeddingBag]]:
        """
        Generate embedding modules (i,e., SSDTableBatchedEmbeddingBags and
        torch.nn.EmbeddingBags)
        """
        import tempfile

        torch.manual_seed(42)
        E = int(10**log_E)
        D = D * 4
        Ds = [D] * T
        Es = [E] * T

        if pooling_mode == PoolingMode.SUM:
            mode = "sum"
            do_pooling = True
        elif pooling_mode == PoolingMode.MEAN:
            mode = "mean"
            do_pooling = True
        elif pooling_mode == PoolingMode.NONE:
            mode = "sum"
            do_pooling = False
        else:
            # This proves that we have exhaustively checked all PoolingModes
            raise RuntimeError("Unknown PoolingMode!")

        # Generate torch EmbeddingBag
        if do_pooling:
            emb_ref = [
                torch.nn.EmbeddingBag(E, D, mode=mode, sparse=True).cuda()
                for (E, D) in zip(Es, Ds)
            ]
        else:
            emb_ref = [
                torch.nn.Embedding(E, D, sparse=True).cuda() for (E, D) in zip(Es, Ds)
            ]

        # Cast type
        if weights_precision == SparseType.FP16:
            emb_ref = [emb.half() for emb in emb_ref]

        # Init weights
        [emb.weight.data.uniform_(-2.0, 2.0) for emb in emb_ref]

        # Construct feature_table_map
        feature_table_map = list(range(T))
        table_to_replicate = -1
        if share_table:
            # autograd with shared embedding only works for exact
            table_to_replicate = T // 2
            # pyre-fixme[6]: For 2nd param expected `Embedding` but got
            #  `Union[Embedding, EmbeddingBag]`.
            feature_table_map.insert(table_to_replicate, table_to_replicate)
            emb_ref.insert(table_to_replicate, emb_ref[table_to_replicate])

        cache_sets = max(int(max(T * B * L, 1) * cache_set_scale), 1)

        # Generate TBE SSD
        emb = SSDTableBatchedEmbeddingBags(
            embedding_specs=[(E, D) for (E, D) in zip(Es, Ds)],
            feature_table_map=feature_table_map,
            ssd_storage_directory=tempfile.mkdtemp(),
            cache_sets=cache_sets,
            ssd_uniform_init_lower=-0.1,
            ssd_uniform_init_upper=0.1,
            learning_rate=lr,
            eps=eps,
            ssd_rocksdb_shards=ssd_shards,
            optimizer=optimizer,
            pooling_mode=pooling_mode,
            weights_precision=weights_precision,
            output_dtype=output_dtype,
            stochastic_rounding=stochastic_rounding,
            prefetch_pipeline=prefetch_pipeline,
            bounds_check_mode=BoundsCheckMode.WARNING,
            l2_cache_size=8,
            bulk_init_chunk_size=bulk_init_chunk_size,
        ).cuda()

        if bulk_init_chunk_size > 0:
            self.assertIsNotNone(
                emb.lazy_init_thread,
                "if bulk_init_chunk_size > 0, lazy_init_thread must be set and it should not be force-synchronized yet",
            )

        # By doing the check for ssd_db being None below, we also access the getter property of ssd_db, which will
        # force the synchronization of lazy_init_thread, and then reset it to None.
        if emb.ssd_db is not None:
            self.assertIsNone(emb.lazy_init_thread)

        # A list to keep the CPU tensor alive until `set` (called inside
        # `set_cuda`) is complete. Note that `set_cuda` is non-blocking
        # asynchronous
        emb_ref_cpu = []

        # Initialize TBE SSD weights
        for f, t in self.get_physical_table_arg_indices_(emb.feature_table_map):
            emb_ref_ = emb_ref[f].weight.clone().detach().cpu()
            emb.ssd_db.set_cuda(
                torch.arange(t * E, (t + 1) * E).to(torch.int64),
                emb_ref_,
                torch.as_tensor([E]),
                t,
            )
            emb_ref_cpu.append(emb_ref_)

        # Ensure that `set` (invoked by `set_cuda`) is done
        torch.cuda.synchronize()

        # Convert back to float (to make sure that accumulation is done
        # in FP32 -- like TBE)
        if weights_precision == SparseType.FP16:
            emb_ref = [emb.float() for emb in emb_ref]

        # pyre-fixme[7]: Expected `Tuple[SSDTableBatchedEmbeddingBags,
        #  List[EmbeddingBag]]` but got `Tuple[SSDTableBatchedEmbeddingBags,
        #  Union[List[Union[Embedding, EmbeddingBag]], List[Embedding],
        #  List[EmbeddingBag]]]`.
        return emb, emb_ref

    def concat_ref_tensors(
        self,
        tensors: List[torch.Tensor],
        do_pooling: bool,
        B: int,
        D: int,
    ) -> torch.Tensor:
        if do_pooling:
            return torch.cat([t.view(B, -1) for t in tensors], dim=1)
        return torch.cat(tensors, dim=0).view(-1, D)

    def concat_ref_tensors_vbe(
        self,
        tensors: List[torch.Tensor],
        batch_size_per_feature_per_rank: List[List[int]],
    ) -> torch.Tensor:
        """
        rearrange tensors into VBE format and concat them into one tensor

        Parameters:
            tensors (List[torch.Tensor]): List of tensors
            batch_size_per_feature_per_rank (List[List[int]]): List of batch sizes per feature per rank

        Returns:
            concatenated tensor in VBE output format
        """
        output = []
        ranks = len(batch_size_per_feature_per_rank[0])
        T = len(batch_size_per_feature_per_rank)
        # for each rank
        start = [0] * T
        for r in range(ranks):
            # for each feature
            for t in range(T):
                b = batch_size_per_feature_per_rank[t][r]
                output.append((tensors[t][start[t] : start[t] + b]).flatten())
                start[t] += b
        return torch.cat(output, dim=0)

    def execute_ssd_forward_(
        self,
        emb: SSDTableBatchedEmbeddingBags,
        emb_ref: List[torch.nn.EmbeddingBag],
        indices_list: List[torch.Tensor],
        per_sample_weights_list: List[torch.Tensor],
        indices: torch.Tensor,
        offsets: torch.Tensor,
        per_sample_weights: torch.Tensor,
        B: int,
        L: int,
        weighted: bool,
        tolerance: Optional[float] = None,
        it: int = -1,
        batch_size_per_feature_per_rank: Optional[List[List[int]]] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Execute the forward functions of SSDTableBatchedEmbeddingBags and
        torch.nn.EmbeddingBag and compare outputs
        """
        assert len(emb_ref) == len(indices_list)
        do_pooling = emb.pooling_mode != PoolingMode.NONE
        # Execute torch EmbeddingBag forward
        output_ref_list = (
            [
                b_indices(emb_, indices, do_pooling=do_pooling)
                for (emb_, indices) in zip(emb_ref, indices_list)
            ]
            if not weighted
            else [
                b_indices(
                    emb_,
                    indices,
                    per_sample_weights=per_sample_weights.view(-1),
                    do_pooling=do_pooling,
                )
                for (emb_, indices, per_sample_weights) in zip(
                    emb_ref, indices_list, per_sample_weights_list
                )
            ]
        )

        # Execute TBE SSD forward
        output = (
            emb(
                indices,
                offsets,
                batch_size_per_feature_per_rank=batch_size_per_feature_per_rank,
            )
            if not weighted
            else emb(
                indices,
                offsets,
                per_sample_weights=per_sample_weights,
                batch_size_per_feature_per_rank=batch_size_per_feature_per_rank,
            )
        )
        if batch_size_per_feature_per_rank is not None:
            output_ref = self.concat_ref_tensors_vbe(
                output_ref_list, batch_size_per_feature_per_rank
            )
        else:
            output_ref = self.concat_ref_tensors(
                output_ref_list,
                do_pooling,
                B,
                emb.embedding_specs[0][1],
            )

        out_dtype = output.dtype
        # Cast the ref output type the output types do not match between ref
        # and test
        if output_ref.dtype != out_dtype:
            output_ref_list = [out.to(out_dtype) for out in output_ref_list]
            output_ref = output_ref.to(out_dtype)

        # Set tolerance
        tolerance = (
            (
                1.0e-5
                if emb_ref[0].weight.dtype == torch.float and out_dtype == torch.float
                else 8.0e-3
            )
            if tolerance is None
            else tolerance
        )

        # Compare outputs
        torch.testing.assert_close(
            output.float(),
            output_ref.float(),
            atol=tolerance,
            rtol=tolerance,
        )
        return output_ref_list, output

    @given(**default_st)
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_ssd_forward(
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        weighted: bool,
        cache_set_scale: float,
        pooling_mode: PoolingMode,
        weights_precision: SparseType,
        output_dtype: SparseType,
        share_table: bool,
        trigger_bounds_check: bool,
        mixed_B: bool,
    ) -> None:

        assume(not weighted or pooling_mode == PoolingMode.SUM)
        assume(not mixed_B or pooling_mode != PoolingMode.NONE)
        # Generate embedding modules
        (
            emb,
            emb_ref,
        ) = self.generate_ssd_tbes(
            T,
            D,
            B,
            log_E,
            L,
            weighted,
            cache_set_scale=cache_set_scale,
            pooling_mode=pooling_mode,
            weights_precision=weights_precision,
            output_dtype=output_dtype,
            share_table=share_table,
        )

        # Generate inputs
        Es = [emb.embedding_specs[t][0] for t in range(T)]
        (
            indices_list,
            per_sample_weights_list,
            indices,
            offsets,
            per_sample_weights,
            batch_size_per_feature_per_rank,
        ) = self.generate_inputs_(
            B,
            L,
            Es,
            emb.feature_table_map,
            weights_precision=weights_precision,
            trigger_bounds_check=trigger_bounds_check,
            mixed_B=mixed_B,
        )

        # Execute forward
        self.execute_ssd_forward_(
            emb,
            emb_ref,
            indices_list,
            per_sample_weights_list,
            indices,
            offsets,
            per_sample_weights,
            B,
            L,
            weighted,
            batch_size_per_feature_per_rank=batch_size_per_feature_per_rank,
        )

    @given(**default_st)
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_ssd_backward_adagrad(
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        weighted: bool,
        cache_set_scale: float,
        pooling_mode: PoolingMode,
        weights_precision: SparseType,
        output_dtype: SparseType,
        share_table: bool,
        trigger_bounds_check: bool,
        mixed_B: bool,
    ) -> None:
        assume(not weighted or pooling_mode == PoolingMode.SUM)
        assume(not mixed_B or pooling_mode != PoolingMode.NONE)

        # Constants
        lr = 0.5
        eps = 0.2
        ssd_shards = 2

        # Generate embedding modules and inputs
        (
            emb,
            emb_ref,
        ) = self.generate_ssd_tbes(
            T,
            D,
            B,
            log_E,
            L,
            weighted,
            lr=lr,
            eps=eps,
            ssd_shards=ssd_shards,
            cache_set_scale=cache_set_scale,
            pooling_mode=pooling_mode,
            weights_precision=weights_precision,
            output_dtype=output_dtype,
            share_table=share_table,
        )

        Es = [emb.embedding_specs[t][0] for t in range(T)]
        (
            indices_list,
            per_sample_weights_list,
            indices,
            offsets,
            per_sample_weights,
            batch_size_per_feature_per_rank,
        ) = self.generate_inputs_(
            B,
            L,
            Es,
            emb.feature_table_map,
            weights_precision=weights_precision,
            trigger_bounds_check=trigger_bounds_check,
            mixed_B=mixed_B,
        )

        # Execute forward
        output_ref_list, output = self.execute_ssd_forward_(
            emb,
            emb_ref,
            indices_list,
            per_sample_weights_list,
            indices,
            offsets,
            per_sample_weights,
            B,
            L,
            weighted,
            batch_size_per_feature_per_rank=batch_size_per_feature_per_rank,
        )

        # Generate output gradient
        output_grad_list = [torch.randn_like(out) for out in output_ref_list]

        # Execute torch EmbeddingBag backward
        [out.backward(grad) for (out, grad) in zip(output_ref_list, output_grad_list)]

        do_pooling = pooling_mode != PoolingMode.NONE
        if batch_size_per_feature_per_rank is not None:
            grad_test = self.concat_ref_tensors_vbe(
                output_grad_list, batch_size_per_feature_per_rank
            )
        else:
            grad_test = self.concat_ref_tensors(
                output_grad_list,
                do_pooling,
                B,
                D * 4,
            )

        # Execute TBE SSD backward
        output.backward(grad_test)

        tolerance = (
            1.0e-4
            if weights_precision == SparseType.FP32 and output_dtype == SparseType.FP32
            else 1.0e-2
        )

        # Compare optimizer states
        split_optimizer_states = [s for (s,) in emb.debug_split_optimizer_states()]
        for f, t in self.get_physical_table_arg_indices_(emb.feature_table_map):
            # pyre-fixme[16]: Optional type has no attribute `float`.
            ref_optimizer_state = emb_ref[f].weight.grad.float().to_dense().pow(2)
            torch.testing.assert_close(
                split_optimizer_states[t].float(),
                ref_optimizer_state.mean(dim=1),
                atol=tolerance,
                rtol=tolerance,
            )

        # Compare weights
        emb.flush()

        emb_test = emb.debug_split_embedding_weights()
        for f, t in self.get_physical_table_arg_indices_(emb.feature_table_map):
            emb_r = emb_ref[f]
            new_ref_weight = torch.addcdiv(
                emb_r.weight.float(),
                value=-lr,
                tensor1=emb_r.weight.grad.float().to_dense(),
                tensor2=split_optimizer_states[t]
                .float()
                .sqrt_()
                .add_(eps)
                .view(Es[t], 1),
            )

            if weights_precision == SparseType.FP16:
                # Round the reference weight the same way that TBE does
                new_ref_weight = new_ref_weight.half().float()

            torch.testing.assert_close(
                emb_test[t].float().cuda(),
                new_ref_weight,
                atol=tolerance,
                rtol=tolerance,
            )

    @given(
        bulk_init_chunk_size=st.sampled_from([0, 100]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_ssd_emb_state_dict(self, bulk_init_chunk_size: int) -> None:
        # Constants
        lr = 0.5
        eps = 0.2
        ssd_shards = 2

        T = 4
        B = 10
        D = 128
        L = 10
        log_E = 4
        weights_precision = SparseType.FP32
        output_dtype = SparseType.FP32

        # Generate embedding modules and inputs
        (
            emb,
            emb_ref,
        ) = self.generate_ssd_tbes(
            T,
            D,
            B,
            log_E,
            L,
            False,  # weighted
            lr=lr,
            eps=eps,
            ssd_shards=ssd_shards,
            cache_set_scale=0.2,
            pooling_mode=PoolingMode.SUM,
            weights_precision=weights_precision,
            output_dtype=output_dtype,
            share_table=True,
            bulk_init_chunk_size=bulk_init_chunk_size,
        )

        Es = [emb.embedding_specs[t][0] for t in range(T)]
        (
            indices_list,
            per_sample_weights_list,
            indices,
            offsets,
            per_sample_weights,
            batch_size_per_feature_per_rank,
        ) = self.generate_inputs_(
            B,
            L,
            Es,
            emb.feature_table_map,
            weights_precision=weights_precision,
            trigger_bounds_check=True,
            mixed_B=True,
        )

        # Execute forward
        output_ref_list, output = self.execute_ssd_forward_(
            emb,
            emb_ref,
            indices_list,
            per_sample_weights_list,
            indices,
            offsets,
            per_sample_weights,
            B,
            L,
            False,
            batch_size_per_feature_per_rank=batch_size_per_feature_per_rank,
        )

        # Generate output gradient
        output_grad_list = [torch.randn_like(out) for out in output_ref_list]

        # Execute torch EmbeddingBag backward
        [out.backward(grad) for (out, grad) in zip(output_ref_list, output_grad_list)]
        if batch_size_per_feature_per_rank is not None:
            grad_test = self.concat_ref_tensors_vbe(
                output_grad_list, batch_size_per_feature_per_rank
            )
        else:
            grad_test = self.concat_ref_tensors(
                output_grad_list,
                True,  # do_pooling
                B,
                D * 4,
            )

        # Execute TBE SSD backward
        output.backward(grad_test)

        tolerance = (
            1.0e-4
            if weights_precision == SparseType.FP32 and output_dtype == SparseType.FP32
            else 1.0e-2
        )

        split_optimizer_states = [s for (s,) in emb.debug_split_optimizer_states()]
        emb.flush()

        # Compare emb state dict with expected values from nn.EmbeddingBag
        emb_state_dict = emb.split_embedding_weights(no_snapshot=False)
        for feature_index, table_index in self.get_physical_table_arg_indices_(
            emb.feature_table_map
        ):
            emb_r = emb_ref[feature_index]
            self.assertLess(table_index, len(emb_state_dict))
            new_ref_weight = torch.addcdiv(
                emb_r.weight.float(),
                value=-lr,
                tensor1=emb_r.weight.grad.float().to_dense(),  # pyre-ignore[16]
                tensor2=split_optimizer_states[table_index]
                .float()
                .sqrt_()
                .add_(eps)
                .view(Es[table_index], 1),
            ).cpu()

            torch.testing.assert_close(
                emb_state_dict[table_index].full_tensor().float(),
                new_ref_weight,
                atol=tolerance,
                rtol=tolerance,
            )

    def execute_ssd_cache_pipeline_(  # noqa C901
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        weighted: bool,
        cache_set_scale: float,
        pooling_mode: PoolingMode,
        weights_precision: SparseType,
        output_dtype: SparseType,
        share_table: bool,
        prefetch_pipeline: bool,
        # If True, prefetch will be invoked by the user.
        explicit_prefetch: bool,
        prefetch_location: Optional[PrefetchLocation],
        use_prefetch_stream: bool,
        flush_location: Optional[FlushLocation],
        trigger_bounds_check: bool,
        mixed_B: bool = False,
    ) -> None:
        # If using pipeline prefetching, explicit prefetching must be True
        assert not prefetch_pipeline or explicit_prefetch

        lr = 0.5
        eps = 0.2
        ssd_shards = 2
        torch.manual_seed(42)

        # Generate embedding modules
        (
            emb,
            emb_ref,
        ) = self.generate_ssd_tbes(
            T,
            D,
            B,
            log_E,
            L,
            weighted,
            lr=lr,
            eps=eps,
            ssd_shards=ssd_shards,
            cache_set_scale=cache_set_scale,
            pooling_mode=pooling_mode,
            weights_precision=weights_precision,
            output_dtype=output_dtype,
            # Disable stochastic rounding because error is too large when
            # running for many iterations. This should be OK for testing the
            # functionality of the cache
            stochastic_rounding=False,
            share_table=share_table,
            prefetch_pipeline=prefetch_pipeline,
        )

        optimizer_states_ref = [
            s.clone().float() for (s,) in emb.debug_split_optimizer_states()
        ]

        Es = [emb.embedding_specs[t][0] for t in range(T)]

        tolerance = (
            1.0e-4
            if weights_precision == SparseType.FP32 and output_dtype == SparseType.FP32
            else 1.0e-2
        )

        batches = []
        for it in range(10):
            batches.append(
                self.generate_inputs_(
                    B,
                    L,
                    Es,
                    emb.feature_table_map,
                    weights_precision=weights_precision,
                    trigger_bounds_check=trigger_bounds_check,
                    mixed_B=mixed_B,
                )
            )

        prefetch_stream = (
            torch.cuda.Stream() if use_prefetch_stream else torch.cuda.current_stream()
        )
        forward_stream = torch.cuda.current_stream() if use_prefetch_stream else None

        iters = 10

        force_flush = flush_location == FlushLocation.ALL

        if force_flush or flush_location == FlushLocation.BEFORE_TRAINING:
            emb.flush()

        # pyre-ignore[53]
        def _prefetch(b_it: int) -> int:
            if not explicit_prefetch or b_it >= iters:
                return b_it + 1

            (
                _,
                _,
                indices,
                offsets,
                _,
                batch_size_per_feature_per_rank,
            ) = batches[b_it]
            with torch.cuda.stream(prefetch_stream):
                emb.prefetch(
                    indices,
                    offsets,
                    forward_stream=forward_stream,
                    batch_size_per_feature_per_rank=batch_size_per_feature_per_rank,
                )
            return b_it + 1

        if prefetch_pipeline:
            # Prefetch the first iteration
            _prefetch(0)
            b_it = 1
        else:
            b_it = 0

        for it in range(iters):
            (
                indices_list,
                per_sample_weights_list,
                indices,
                offsets,
                per_sample_weights,
                batch_size_per_feature_per_rank,
            ) = batches[it]

            # Ensure that prefetch i is done before forward i
            if use_prefetch_stream:
                assert forward_stream is not None
                forward_stream.wait_stream(prefetch_stream)

            # Prefetch before forward
            if (
                not prefetch_pipeline
                or prefetch_location == PrefetchLocation.BEFORE_FWD
            ):
                b_it = _prefetch(b_it)

            # Execute forward
            output_ref_list, output = self.execute_ssd_forward_(
                emb,
                emb_ref,
                indices_list,
                per_sample_weights_list,
                indices,
                offsets,
                per_sample_weights,
                B,
                L,
                weighted,
                tolerance=tolerance,
                it=it,
                batch_size_per_feature_per_rank=batch_size_per_feature_per_rank,
            )

            if force_flush or flush_location == FlushLocation.AFTER_FWD:
                emb.flush()

            # Generate output gradient
            output_grad_list = [torch.randn_like(out) for out in output_ref_list]

            # Zero out weight grad
            for f, _ in self.get_physical_table_arg_indices_(emb.feature_table_map):
                emb_ref[f].weight.grad = None

            # Execute torch EmbeddingBag backward
            for out, grad in zip(output_ref_list, output_grad_list):
                out.backward(grad)

            do_pooling = pooling_mode != PoolingMode.NONE
            if batch_size_per_feature_per_rank is not None:
                grad_test = self.concat_ref_tensors_vbe(
                    output_grad_list, batch_size_per_feature_per_rank
                )
            else:
                grad_test = self.concat_ref_tensors(
                    output_grad_list,
                    do_pooling,
                    B,
                    D * 4,
                )

            # Prefetch between forward and backward
            if (
                prefetch_pipeline
                and prefetch_location == PrefetchLocation.BETWEEN_FWD_BWD
            ):
                b_it = _prefetch(b_it)

            # Execute TBE SSD backward
            output.backward(grad_test)

            if force_flush or flush_location == FlushLocation.AFTER_BWD:
                emb.flush()

            # Compare optimizer states
            split_optimizer_states = [s for (s,) in emb.debug_split_optimizer_states()]
            for f, t in self.get_physical_table_arg_indices_(emb.feature_table_map):
                optim_state_r = optimizer_states_ref[t]
                optim_state_t = split_optimizer_states[t]
                emb_r = emb_ref[f]

                # pyre-fixme[16]: Optional type has no attribute `float`.
                optim_state_r.add_(
                    # pyre-fixme[16]: `Optional` has no attribute `float`.
                    emb_r.weight.grad.float()
                    .to_dense()
                    .pow(2)
                    .mean(dim=1)
                )
                torch.testing.assert_close(
                    optim_state_t.float(),
                    optim_state_r,
                    atol=tolerance,
                    rtol=tolerance,
                )

                new_ref_weight = torch.addcdiv(
                    emb_r.weight.float(),
                    value=-lr,
                    tensor1=emb_r.weight.grad.float().to_dense(),
                    tensor2=optim_state_t.float().sqrt().add(eps).view(Es[t], 1),
                )

                if weights_precision == SparseType.FP16:
                    # Round the reference weight the same way that
                    # TBE does
                    new_ref_weight = new_ref_weight.half().float()
                    assert new_ref_weight.dtype == emb_r.weight.dtype

                emb_r.weight.data.copy_(new_ref_weight)

        # Compare weights
        emb.flush()
        for f, t in self.get_physical_table_arg_indices_(emb.feature_table_map):
            weight_r = emb_ref[f].weight.float()
            weight_t = emb.debug_split_embedding_weights()[t].float().cuda()
            torch.testing.assert_close(
                weight_t,
                weight_r,
                atol=tolerance,
                rtol=tolerance,
            )

    @given(
        flush_location=st.sampled_from(FlushLocation),
        explicit_prefetch=st.booleans(),
        prefetch_location=st.sampled_from(PrefetchLocation),
        use_prefetch_stream=st.booleans(),
        **default_st,
    )
    @settings(
        verbosity=Verbosity.verbose, max_examples=MAX_PIPELINE_EXAMPLES, deadline=None
    )
    def test_ssd_cache_flush(self, **kwargs: Any):
        """
        Test the correctness of the SSD cache prefetch workflow with
        excessive flushing
        """
        kwargs["prefetch_pipeline"] = False
        if kwargs["explicit_prefetch"] or not kwargs["use_prefetch_stream"]:
            kwargs["prefetch_pipeline"] = True

        assume(not kwargs["weighted"] or kwargs["pooling_mode"] == PoolingMode.SUM)
        assume(kwargs["prefetch_pipeline"] and kwargs["explicit_prefetch"])
        assume(not kwargs["use_prefetch_stream"] or kwargs["prefetch_pipeline"])
        assume(not kwargs["mixed_B"] or kwargs["pooling_mode"] != PoolingMode.NONE)
        self.execute_ssd_cache_pipeline_(
            **kwargs,
        )

    @given(**default_st)
    @settings(
        verbosity=Verbosity.verbose, max_examples=MAX_PIPELINE_EXAMPLES, deadline=None
    )
    def test_ssd_cache_implicit_prefetch(self, **kwargs: Any):
        """
        Test the correctness of the SSD cache prefetch workflow
        without pipeline prefetching and with implicit prefetching.
        Implicit prefetching relies on TBE forward to invoke prefetch.
        """
        assume(not kwargs["weighted"] or kwargs["pooling_mode"] == PoolingMode.SUM)
        assume(not kwargs["mixed_B"] or kwargs["pooling_mode"] != PoolingMode.NONE)
        self.execute_ssd_cache_pipeline_(
            prefetch_pipeline=False,
            explicit_prefetch=False,
            prefetch_location=None,
            use_prefetch_stream=False,
            flush_location=None,
            **kwargs,
        )

    @given(**default_st)
    @settings(
        verbosity=Verbosity.verbose, max_examples=MAX_PIPELINE_EXAMPLES, deadline=None
    )
    def test_ssd_cache_explicit_prefetch(self, **kwargs: Any):
        """
        Test the correctness of the SSD cache prefetch workflow
        without pipeline prefetching and with explicit prefetching
        (the user explicitly invokes prefetch).  Each prefetch invoked
        before a forward TBE fetches data for that specific iteration.

        For example:

        ------------------------- Timeline ------------------------>
        pf(i) -> fwd(i) -> ... -> pf(i+1) -> fwd(i+1) -> ...

        Note:
        - pf(i) = prefetch of iteration i
        - fwd(i) = forward TBE of iteration i
        """
        assume(not kwargs["weighted"] or kwargs["pooling_mode"] == PoolingMode.SUM)
        assume(not kwargs["mixed_B"] or kwargs["pooling_mode"] != PoolingMode.NONE)
        self.execute_ssd_cache_pipeline_(
            prefetch_pipeline=False,
            explicit_prefetch=True,
            prefetch_location=None,
            use_prefetch_stream=False,
            flush_location=None,
            **kwargs,
        )

    @given(use_prefetch_stream=st.booleans(), **default_st)
    @settings(
        verbosity=Verbosity.verbose, max_examples=MAX_PIPELINE_EXAMPLES, deadline=None
    )
    def test_ssd_cache_pipeline_before_fwd(self, **kwargs: Any):
        """
        Test the correctness of the SSD cache prefetch workflow with
        pipeline prefetching when cache prefetching of the next
        iteration is invoked before the forward TBE of the current
        iteration.

        For example:

        ------------------------- Timeline ------------------------>
        pf(i+1) -> fwd(i) -> ... -> pf(i+2) -> fwd(i+1) -> ...

        Note:
        - pf(i) = prefetch of iteration i
        - fwd(i) = forward TBE of iteration i
        """
        assume(not kwargs["weighted"] or kwargs["pooling_mode"] == PoolingMode.SUM)
        assume(not kwargs["mixed_B"] or kwargs["pooling_mode"] != PoolingMode.NONE)
        self.execute_ssd_cache_pipeline_(
            prefetch_pipeline=True,
            explicit_prefetch=True,
            prefetch_location=PrefetchLocation.BEFORE_FWD,
            flush_location=None,
            **kwargs,
        )

    @given(use_prefetch_stream=st.booleans(), **default_st)
    @settings(
        verbosity=Verbosity.verbose, max_examples=MAX_PIPELINE_EXAMPLES, deadline=None
    )
    def test_ssd_cache_pipeline_between_fwd_bwd(self, **kwargs: Any):
        """
        Test the correctness of the SSD cache prefetch workflow with
        pipeline prefetching when cache prefetching of the next
        iteration is invoked after the forward TBE and before the
        backward TBE of the current iteration.

        For example:

        ------------------------- Timeline ------------------------>
        fwd(i) -> pf(i+1) -> bwd(i) -> ... -> fwd(i+1) -> pf(i+2) -> bwd(i+1) -> ...

        Note:
        - pf(i) = prefetch of iteration i
        - fwd(i) = forward TBE of iteration i
        - bwd(i) = backward TBE of iteration i
        """

        assume(not kwargs["weighted"] or kwargs["pooling_mode"] == PoolingMode.SUM)
        assume(not kwargs["mixed_B"] or kwargs["pooling_mode"] != PoolingMode.NONE)
        self.execute_ssd_cache_pipeline_(
            prefetch_pipeline=True,
            explicit_prefetch=True,
            prefetch_location=PrefetchLocation.BETWEEN_FWD_BWD,
            flush_location=None,
            **kwargs,
        )
