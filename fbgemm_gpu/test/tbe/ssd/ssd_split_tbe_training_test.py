# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[3,6,56]

import math
import tempfile
import unittest
from enum import Enum

from typing import Any, Dict, List, Optional, Tuple, Union

import hypothesis.strategies as st
import numpy as np
import torch
from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType, SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    BackendType,
    BoundsCheckMode,
    EvictionPolicy,
    KVZCHParams,
    PoolingMode,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_training import RESParams
from fbgemm_gpu.tbe.ssd import SSDTableBatchedEmbeddingBags
from fbgemm_gpu.tbe.utils import b_indices, get_table_batched_offsets_from_dense
from hypothesis import assume, given, settings, Verbosity
from torch import distributed as dist

from .. import common  # noqa E402
from ..common import gen_mixed_B_batch_sizes, gpu_unavailable, running_in_oss


def find_different_rows(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    atol: float = 1.0e-4,
    rtol: float = 1.0e-4,
    return_values: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Find the indices of rows that are different between two tensors.

    Args:
        tensor1: First tensor
        tensor2: Second tensor
        atol: Absolute tolerance for comparison
        rtol: Relative tolerance for comparison
        return_values: If True, also return the values of the different rows

    Returns:
        If return_values is False:
            indices: Indices of rows that are different
        If return_values is True:
            (indices, tensor1_values, tensor2_values): Tuple containing indices and corresponding values
    """
    # Convert to float and CPU for consistent comparison
    t1 = tensor1.float().cpu()
    t2 = tensor2.float().cpu()

    # Check if shapes match
    if t1.shape != t2.shape:
        raise ValueError(f"Tensor shapes don't match: {t1.shape} vs {t2.shape}")

    # Calculate absolute and relative differences
    abs_diff = torch.abs(t1 - t2)
    rel_diff = abs_diff / torch.max(torch.abs(t2), torch.tensor(1e-8))

    # Find rows where either absolute or relative difference exceeds tolerance
    # First check if each element in a row exceeds the tolerance
    abs_mask = abs_diff > atol
    rel_mask = rel_diff > rtol
    element_mask = abs_mask & rel_mask

    # Then check if any element in a row exceeds the tolerance
    if len(t1.shape) > 1:
        # For 2D+ tensors, check if any element in each row differs
        row_mask = element_mask.any(dim=tuple(range(1, len(t1.shape))))
    else:
        # For 1D tensors, each element is a "row"
        row_mask = element_mask

    # Get indices of different rows
    diff_indices = torch.nonzero(row_mask).flatten()

    if return_values:
        # Return indices and the corresponding rows from both tensors
        return diff_indices, t1[diff_indices], t2[diff_indices]
    else:
        return diff_indices


def print_different_rows(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    atol: float = 1.0e-4,
    rtol: float = 1.0e-4,
    max_rows: int = 10,
    name1: str = "tensor1",
    name2: str = "tensor2",
) -> None:
    """
    Print the indices and values of rows that are different between two tensors.

    Args:
        tensor1: First tensor
        tensor2: Second tensor
        atol: Absolute tolerance for comparison
        rtol: Relative tolerance for comparison
        max_rows: Maximum number of different rows to print
        name1: Name of the first tensor for display
        name2: Name of the second tensor for display
    """
    indices, values1, values2 = find_different_rows(
        tensor1, tensor2, atol, rtol, return_values=True
    )

    num_diff = len(indices)
    print(f"Found {num_diff} different rows out of {tensor1.shape[0]} total rows")

    if num_diff == 0:
        return

    # Limit the number of rows to display
    display_count = min(num_diff, max_rows)
    if num_diff > max_rows:
        print(f"Showing first {display_count} differences:")

    # Calculate max absolute and relative differences for each row
    t1 = tensor1.float().cpu()
    t2 = tensor2.float().cpu()

    for i in range(display_count):
        idx = indices[i].item()
        row1 = t1[idx]
        row2 = t2[idx]

        abs_diff = torch.abs(row1 - row2)
        rel_diff = abs_diff / torch.max(torch.abs(row2), torch.tensor(1e-8))

        max_abs_diff = torch.max(abs_diff).item()
        max_rel_diff = torch.max(rel_diff).item()

        print(f"Row {idx}:")
        print(f"  {name1}: {row1}")
        print(f"  {name2}: {row2}")
        print(f"  Max absolute difference: {max_abs_diff}")
        print(f"  Max relative difference: {max_rel_diff}")
        print()


MAX_EXAMPLES = 40
MAX_PIPELINE_EXAMPLES = 10
KV_WORLD_SIZE = 4
VIRTUAL_TABLE_ROWS = int(
    2**18
)  # relatively large for now given optimizer is still pre-allocated

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

        weights = torch.randn(N, emb.cache_row_dim, dtype=weights_precision.as_dtype())
        output_weights = torch.empty_like(weights)

        emb.ssd_db.get_cuda(indices, output_weights, count)
        torch.cuda.synchronize()
        assert (output_weights <= 0.1).all().item()
        assert (output_weights >= -0.1).all().item()

        emb.ssd_db.set_cuda(indices, weights, count, 1)
        emb.ssd_db.get_cuda(indices, output_weights, count)
        torch.cuda.synchronize()
        torch.testing.assert_close(weights, output_weights)

    def generate_in_bucket_indices(
        self,
        hash_mode: int,
        bucket_id_range: Tuple[int, int],
        bucket_size: int,
        # max height in ref_emb, the logical id high, physically id in kv is a shift from [0,h) to [table_offset, table_offset+h]
        high: int,
        size: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Generate indices in embedding bucket, this is guarantee on the torchrec input_dist
        """
        assert hash_mode == 0, "only support hash_mode=0, aka chunk-based hashing"

        # hash mode is chunk-based hashing
        # STEP 1: generate all the eligible indices in the given range
        bucket_id_start = bucket_id_range[0]
        bucket_id_end = bucket_id_range[1]
        rank_input_range = (bucket_id_end - bucket_id_start) * bucket_size
        rank_input_range = min(rank_input_range, high)
        indices = torch.as_tensor(
            np.random.choice(rank_input_range, replace=False, size=(rank_input_range,)),
            dtype=torch.int64,
        )
        indices += bucket_id_start * bucket_size

        # STEP 2: generate random indices with the given shape from the eligible indices above                      # 想要的输出形状
        idx = torch.randint(0, indices.numel(), size)
        random_indices = indices[idx]
        return random_indices

    def generate_inputs_(
        self,
        B: int,
        L: int,
        Es: List[int],
        feature_table_map: List[int],
        weights_precision: SparseType = SparseType.FP32,
        trigger_bounds_check: bool = False,
        mixed_B: bool = False,
        is_kv_tbes: bool = False,
        bucket_offsets: Optional[List[Tuple[int, int]]] = None,
        bucket_sizes: Optional[List[int]] = None,
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
        if is_kv_tbes:
            assert len(bucket_offsets) == len(bucket_sizes)
            assert len(bucket_offsets) <= len(feature_table_map)
            indices_list = [
                self.generate_in_bucket_indices(
                    0,
                    # pyre-ignore
                    bucket_offsets[t],
                    # pyre-ignore
                    bucket_sizes[t],
                    Es[t] * (2 if trigger_bounds_check else 1),
                    size=(b, L),
                ).cuda()
                for (b, t) in zip(Bs, feature_table_map)
            ]
        else:
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

    def generate_kvzch_tbes(
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        weighted: bool,
        lr: float = 0.01,  # from SSDTableBatchedEmbeddingBags
        eps: float = 1.0e-8,  # from SSDTableBatchedEmbeddingBags
        weight_decay: float = 0.0,  # used by LARS-SGD, LAMB, ADAM, and Rowwise Adagrad
        beta1: float = 0.9,  # used by Partial Rowwise Adam
        beta2: float = 0.999,  # used by Partial Rowwise Adam
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
        backend_type: BackendType = BackendType.SSD,
        num_buckets: int = 10,
        mixed: bool = False,
        enable_optimizer_offloading: bool = False,
        backend_return_whole_row: bool = False,
        optimizer_state_dtypes: Dict[OptimType, SparseType] = {},
    ) -> Tuple[
        SSDTableBatchedEmbeddingBags,
        List[torch.nn.EmbeddingBag],
        List[int],
        List[int],
        List[Tuple[int, int]],
        List[int],
    ]:
        """
        Generate embedding modules (i,e., SSDTableBatchedEmbeddingBags and
        torch.nn.EmbeddingBags)

        Idea in this UT, originally we have a ref_emb using EmbeddingBag/Embedding with the same size in emb,
        to stimulate the lookup result from different pooling, weighted, etc..., and doing lookup on both
        ref_emb and emb and compare the result.

        However when we test with kv zch embedding lookup, we are using virtual space which can not be preallocated
        in ref_emb using EmbeddingBag/Embeddings.

        The little trick we do here is we still pre-allocate ref_emb with the given table size,
        but when we create SSD TBE, we passed in the virtual table size.

        For example if the given table size is [100, 256], we have ref_emb with [100, 256], and SSD TBE with [2^25, 256]
        the input id will always be in the range of [0, 100)
        """
        import tempfile

        torch.manual_seed(42)
        E = int(10**log_E)
        virtual_E = VIRTUAL_TABLE_ROWS
        D = D * 4

        bucket_sizes = []
        bucket_offsets = []
        for _ in range(T):
            bucket_sizes.append(math.ceil(virtual_E / num_buckets))
            bucket_start = (
                0  # since ref_emb is dense format, we need to start from 0th bucket
            )
            bucket_end = min(math.ceil(num_buckets / KV_WORLD_SIZE), num_buckets)
            bucket_offsets.append((bucket_start, bucket_end))

        # In reality this will be populated with _populate_zero_collision_tbe_params
        # from virtual_table_eviction_policy. For UT, we need to explicitly populate it
        kv_zch_param = KVZCHParams(
            bucket_offsets=bucket_offsets,
            bucket_sizes=bucket_sizes,
            enable_optimizer_offloading=enable_optimizer_offloading,
            backend_return_whole_row=backend_return_whole_row,
            eviction_policy=EvictionPolicy(
                meta_header_lens=([16 // (weights_precision.bit_rate() // 8)] * T)
            ),
        )

        E = min(E, (bucket_offsets[0][1] - bucket_offsets[0][0]) * bucket_sizes[0])

        if not mixed:
            Ds = [D] * T
            Es = [E] * T
        else:
            # Ds = [
            #     round_up(np.random.randint(low=int(0.25 * D), high=int(1.0 * D)), 4)
            #     for _ in range(T)
            # ]
            Ds = [D] * T
            Es = [np.random.randint(low=int(0.5 * E), high=int(E)) for _ in range(T)]

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
        # in kv tbes, we still maintain a small emb in EmbeddingBag or Embedding as a reference for expected outcome,
        # but the virual space passed into TBE will be super large, e.g. 2^50
        # NOTE we will use a relative large virtual size for now, given that optimizer is still pre-allocated
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
            feature_table_map.insert(table_to_replicate, table_to_replicate)
            emb_ref.insert(table_to_replicate, emb_ref[table_to_replicate])

        cache_sets = max(int(max(T * B * L, 1) * cache_set_scale), 1)

        # Generate TBE SSD
        emb = SSDTableBatchedEmbeddingBags(
            embedding_specs=[(virtual_E, D) for D in Ds],
            feature_table_map=feature_table_map,
            ssd_storage_directory=tempfile.mkdtemp(),
            cache_sets=cache_sets,
            ssd_uniform_init_lower=-0.1,
            ssd_uniform_init_upper=0.1,
            learning_rate=lr,
            eps=eps,
            weight_decay=weight_decay,
            beta1=beta1,
            beta2=beta2,
            ssd_rocksdb_shards=ssd_shards,
            optimizer=optimizer,
            pooling_mode=pooling_mode,
            weights_precision=weights_precision,
            output_dtype=output_dtype,
            stochastic_rounding=stochastic_rounding,
            prefetch_pipeline=prefetch_pipeline,
            bounds_check_mode=BoundsCheckMode.WARNING,
            l2_cache_size=8,
            backend_type=backend_type,
            kv_zch_params=kv_zch_param,
            optimizer_state_dtypes=optimizer_state_dtypes,
        ).cuda()

        if backend_type == BackendType.SSD:
            self.assertTrue(emb.ssd_db.is_auto_compaction_enabled())

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
            pad_opt_width = emb.cache_row_dim - emb_ref_.size(1)
            pad_opt = torch.zeros(emb_ref_.size(0), pad_opt_width, dtype=emb_ref_.dtype)
            emb_opt_ref = torch.cat((emb_ref_, pad_opt), dim=1)
            emb.ssd_db.set_cuda(
                torch.arange(t * virtual_E, t * virtual_E + Es[t]).to(torch.int64),
                emb_opt_ref,
                torch.as_tensor([Es[t]]),
                t,
            )
            emb_ref_cpu.append(emb_ref_)

        # Ensure that `set` (invoked by `set_cuda`) is done
        torch.cuda.synchronize()

        # Convert back to float (to make sure that accumulation is done
        # in FP32 -- like TBE)
        if weights_precision == SparseType.FP16:
            emb_ref = [emb.float() for emb in emb_ref]

        # pyre-fixme[7]
        return emb, emb_ref, Es, Ds, bucket_offsets, bucket_sizes

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
        weight_decay: float = 0.0,  # used by LARS-SGD, LAMB, ADAM, and Rowwise Adagrad
        beta1: float = 0.9,  # used by Partial Rowwise Adam
        beta2: float = 0.999,  # used by Partial Rowwise Adam
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
        lazy_bulk_init_enabled: bool = False,
        backend_type: BackendType = BackendType.SSD,
        enable_raw_embedding_streaming: bool = False,
        optimizer_state_dtypes: Dict[str, SparseType] = {},  # noqa: B006
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
            feature_table_map.insert(table_to_replicate, table_to_replicate)
            emb_ref.insert(table_to_replicate, emb_ref[table_to_replicate])

        cache_sets = max(int(max(T * B * L, 1) * cache_set_scale), 1)
        res_params: Optional[RESParams] = None
        if enable_raw_embedding_streaming:
            res_params = RESParams(
                res_server_port=0,
                res_store_shards=1,
                table_names=["t" + str(x) for x in range(0, T)],
                table_offsets=[0] * T,
            )

        with unittest.mock.patch.object(dist, "get_rank", return_value=0):
            # Generate TBE SSD
            emb = SSDTableBatchedEmbeddingBags(
                embedding_specs=[(E, D) for (E, D) in zip(Es, Ds)],
                feature_table_map=feature_table_map,
                ssd_storage_directory=f"{tempfile.mkdtemp()},{tempfile.mkdtemp()}",
                cache_sets=cache_sets,
                ssd_uniform_init_lower=-0.1,
                ssd_uniform_init_upper=0.1,
                learning_rate=lr,
                eps=eps,
                weight_decay=weight_decay,
                beta1=beta1,
                beta2=beta2,
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
                lazy_bulk_init_enabled=lazy_bulk_init_enabled,
                enable_raw_embedding_streaming=enable_raw_embedding_streaming,
                backend_type=backend_type,
                res_params=res_params,
                optimizer_state_dtypes=optimizer_state_dtypes,
            ).cuda()

        if bulk_init_chunk_size > 0 and lazy_bulk_init_enabled:
            self.assertIsNotNone(
                emb.lazy_init_thread,
                "if bulk_init_chunk_size > 0, lazy_init_thread must be set and it should not be force-synchronized yet",
            )
        if backend_type == BackendType.SSD:
            self.assertTrue(emb.ssd_db.is_auto_compaction_enabled())

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

    def execute_ssd_backward_(
        self,
        output_ref_list: List[torch.Tensor],
        output: torch.Tensor,
        B: int,
        D: int,
        pooling_mode: PoolingMode,
        batch_size_per_feature_per_rank: Optional[List[List[int]]] = None,
    ) -> None:
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
                pooling_mode != PoolingMode.NONE,
                B,
                D * 4,
            )

        # Execute SSD TBE backward
        output.backward(grad_test)

    def split_optimizer_states_(
        self, emb: SSDTableBatchedEmbeddingBags
    ) -> List[List[torch.Tensor]]:
        _, bucket_asc_ids_list, _, _ = emb.split_embedding_weights(
            no_snapshot=False, should_flush=True
        )

        return emb.split_optimizer_states(
            bucket_asc_ids_list, no_snapshot=False, should_flush=True
        )

    def assert_close_(self, test: torch.Tensor, ref: torch.Tensor) -> None:
        tolerance = 1.0e-3 if test.dtype == torch.float else 1.0e-2

        torch.testing.assert_close(
            test.float().cpu(),
            ref.float().cpu(),
            atol=tolerance,
            rtol=tolerance,
        )

    @given(
        **default_st, backend_type=st.sampled_from([BackendType.SSD, BackendType.DRAM])
    )
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
        backend_type: BackendType,
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
            backend_type=backend_type,
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

    @given(
        **default_st, backend_type=st.sampled_from([BackendType.SSD, BackendType.DRAM])
    )
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
        backend_type: BackendType,
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
            backend_type=backend_type,
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

        # Execute backward
        self.execute_ssd_backward_(
            output_ref_list,
            output,
            B,
            D,
            pooling_mode,
            batch_size_per_feature_per_rank,
        )

        tolerance = (
            1.0e-4
            if weights_precision == SparseType.FP32 and output_dtype == SparseType.FP32
            else 1.0e-2
        )

        # Compare optimizer states
        split_optimizer_states = self.split_optimizer_states_(emb)
        for f, t in self.get_physical_table_arg_indices_(emb.feature_table_map):
            # pyre-fixme[16]: Optional type has no attribute `float`.
            ref_optimizer_state = emb_ref[f].weight.grad.float().to_dense().pow(2)
            torch.testing.assert_close(
                split_optimizer_states[t][0].float(),
                ref_optimizer_state.mean(dim=1),
                atol=tolerance,
                rtol=tolerance,
            )

        # Compare weights
        emb.flush()

        emb_test = emb.debug_split_embedding_weights()
        for f, t in self.get_physical_table_arg_indices_(emb.feature_table_map):
            emb_r = emb_ref[f]
            (m1,) = split_optimizer_states[t]
            new_ref_weight = torch.addcdiv(
                emb_r.weight.float(),
                value=-lr,
                tensor1=emb_r.weight.grad.float().to_dense(),
                tensor2=m1.float().sqrt_().add_(eps).view(Es[t], 1),
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
        **default_st,
        backend_type=st.sampled_from([BackendType.SSD, BackendType.DRAM]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_ssd_backward_partial_rowwise_adam(
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
        backend_type: BackendType,
    ) -> None:
        assume(not weighted or pooling_mode == PoolingMode.SUM)
        # VBE is currently not supported for PARTIAL_ROWWISE_ADAM optimizer
        assume(not mixed_B)

        # Constants
        lr = 0.5
        eps = 0.2
        ssd_shards = 2
        beta1 = 0.9
        beta2 = 0.99
        weight_decay = 0.01

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
            weight_decay=weight_decay,
            beta1=beta1,
            beta2=beta2,
            ssd_shards=ssd_shards,
            optimizer=OptimType.PARTIAL_ROWWISE_ADAM,
            cache_set_scale=cache_set_scale,
            pooling_mode=pooling_mode,
            weights_precision=weights_precision,
            output_dtype=output_dtype,
            share_table=share_table,
            backend_type=backend_type,
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

        # Execute backward
        self.execute_ssd_backward_(
            output_ref_list,
            output,
            B,
            D,
            pooling_mode,
            batch_size_per_feature_per_rank,
        )

        emb.flush()

        tolerance = 1.0e-2

        emb_test_weights = emb.debug_split_embedding_weights()
        split_optimizer_states = self.split_optimizer_states_(emb)

        for f, t in self.get_physical_table_arg_indices_(emb.feature_table_map):
            (m1, m2) = split_optimizer_states[t]
            # Some optimizers have non-float momentum values
            # pyre-ignore[16]
            ref_grad = emb_ref[f].weight.grad.cpu().to_dense()
            ref_weights = emb_ref[f].weight.cpu()

            # Compare momentum2 values: (1 - beta2) * dL^2
            m2_ref = (ref_grad.pow(2).mean(dim=1)) * (1.0 - beta2)
            self.assert_close_(m2, m2_ref)

            # Compare momentum1 values: (1 - beta1) * dL
            m1_ref = ref_grad * (1.0 - beta1)
            self.assert_close_(m1, m1_ref)

            # Bias corrections
            iter_ = emb.iter.item()
            v_hat_t = m2_ref / (1 - beta2**iter_)
            v_hat_t = v_hat_t.view(v_hat_t.numel(), 1)
            m_hat_t = m1_ref / (1 - beta1**iter_)

            # Weight update
            ref_weights_updated = (
                torch.addcdiv(
                    ref_weights,
                    value=-lr,
                    tensor1=m_hat_t,
                    tensor2=v_hat_t.sqrt_().add_(eps),
                )
                - lr * weight_decay * ref_weights
            )

            if weights_precision == SparseType.FP16:
                # Round the reference weight the same way that TBE does
                ref_weights_updated = ref_weights_updated.half().float()

            # Compare weights
            torch.testing.assert_close(
                emb_test_weights[t].float().cuda(),
                ref_weights_updated.cuda(),
                atol=tolerance,
                rtol=tolerance,
            )

    @given(
        **default_st,
        backend_type=st.sampled_from([BackendType.SSD, BackendType.DRAM]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_ssd_backward_adam(
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
        backend_type: BackendType,
    ) -> None:
        assume(not weighted or pooling_mode == PoolingMode.SUM)
        # VBE is currently not supported for PARTIAL_ROWWISE_ADAM optimizer
        assume(not mixed_B)

        # Constants
        lr = 0.5
        eps = 0.2
        ssd_shards = 2
        beta1 = 0.9
        beta2 = 0.99
        weight_decay = 0.01

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
            weight_decay=weight_decay,
            beta1=beta1,
            beta2=beta2,
            ssd_shards=ssd_shards,
            optimizer=OptimType.ADAM,
            cache_set_scale=cache_set_scale,
            pooling_mode=pooling_mode,
            weights_precision=weights_precision,
            output_dtype=output_dtype,
            share_table=share_table,
            backend_type=backend_type,
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

        # Execute backward
        self.execute_ssd_backward_(
            output_ref_list,
            output,
            B,
            D,
            pooling_mode,
            batch_size_per_feature_per_rank,
        )

        emb.flush()

        tolerance = 1.0e-2

        emb_test_weights = emb.debug_split_embedding_weights()
        split_optimizer_states = self.split_optimizer_states_(emb)

        for f, t in self.get_physical_table_arg_indices_(emb.feature_table_map):
            (m1, m2) = split_optimizer_states[t]
            # Some optimizers have non-float momentum values
            # pyre-ignore[16]
            ref_grad = emb_ref[f].weight.grad.cpu().to_dense()
            ref_weights = emb_ref[f].weight.cpu()

            # Compare momentum2 values: (1 - beta2) * dL^2
            m2_ref = ref_grad.pow(2) * (1.0 - beta2)
            self.assert_close_(m2, m2_ref)

            # Compare momentum1 values: (1 - beta1) * dL
            m1_ref = ref_grad * (1.0 - beta1)
            self.assert_close_(m1, m1_ref)

            # Bias corrections
            iter_ = emb.iter.item()
            v_hat_t = m2_ref / (1 - beta2**iter_)
            m_hat_t = m1_ref / (1 - beta1**iter_)

            # Weight update
            ref_weights_updated = (
                torch.addcdiv(
                    ref_weights,
                    value=-lr,
                    tensor1=m_hat_t,
                    tensor2=v_hat_t.sqrt_().add_(eps),
                )
                - lr * weight_decay * ref_weights
            )

            if weights_precision == SparseType.FP16:
                # Round the reference weight the same way that TBE does
                ref_weights_updated = ref_weights_updated.half().float()

            # Compare weights
            torch.testing.assert_close(
                emb_test_weights[t].float().cuda(),
                ref_weights_updated.cuda(),
                atol=tolerance,
                rtol=tolerance,
            )

    @given(
        bulk_init_chunk_size=st.sampled_from([0, 204800]),
        lazy_bulk_init_enabled=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_ssd_emb_state_dict(
        self, bulk_init_chunk_size: int, lazy_bulk_init_enabled: bool
    ) -> None:
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
        pooling_mode = PoolingMode.SUM

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
            pooling_mode=pooling_mode,
            weights_precision=weights_precision,
            output_dtype=output_dtype,
            share_table=True,
            bulk_init_chunk_size=bulk_init_chunk_size,
            lazy_bulk_init_enabled=lazy_bulk_init_enabled,
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

        # Execute backward
        self.execute_ssd_backward_(
            output_ref_list,
            output,
            B,
            D,
            pooling_mode,
            batch_size_per_feature_per_rank,
        )

        emb.flush()

        tolerance = (
            1.0e-4
            if weights_precision == SparseType.FP32 and output_dtype == SparseType.FP32
            else 1.0e-2
        )

        split_optimizer_states = self.split_optimizer_states_(emb)

        # Compare emb state dict with expected values from nn.EmbeddingBag
        emb_state_dict, _, _, _ = emb.split_embedding_weights(no_snapshot=False)
        for feature_index, table_index in self.get_physical_table_arg_indices_(
            emb.feature_table_map
        ):
            (m1,) = split_optimizer_states[table_index]
            emb_r = emb_ref[feature_index]
            self.assertLess(table_index, len(emb_state_dict))
            new_ref_weight = torch.addcdiv(
                emb_r.weight.float(),
                value=-lr,
                tensor1=emb_r.weight.grad.float().to_dense(),  # pyre-ignore[16]
                tensor2=m1.float().sqrt_().add_(eps).view(Es[table_index], 1),
            ).cpu()

            torch.testing.assert_close(
                # pyre-fixme[16]: Undefined attribute: Item `torch._tensor.Tensor` of `typing.Uni...
                emb_state_dict[table_index].full_tensor().float(),
                new_ref_weight,
                atol=tolerance,
                rtol=tolerance,
            )

    @given(
        **default_st,
        **{
            "m1_dtype": st.sampled_from([SparseType.BF16, SparseType.FP32]),
            "m2_dtype": st.sampled_from([SparseType.BF16, SparseType.FP32]),
        },
        num_buckets=st.integers(min_value=10, max_value=15),
        enable_optimizer_offloading=st.booleans(),
        backend_type=st.sampled_from([BackendType.SSD, BackendType.DRAM]),
        bulk_init_chunk_size=st.sampled_from([0, 204800]),
        lazy_bulk_init_enabled=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_ssd_emb_state_dict_partial_rowwise_adam(
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        weighted: bool,
        pooling_mode: PoolingMode,
        weights_precision: SparseType,
        output_dtype: SparseType,
        m1_dtype: SparseType,
        m2_dtype: SparseType,
        bulk_init_chunk_size: int,
        lazy_bulk_init_enabled: bool,
        # pyre-ignore[2]
        **kwargs,
    ) -> None:
        # Constants
        lr = 0.5
        eps = 0.2
        ssd_shards = 2
        beta1 = 0.9
        beta2 = 0.99
        weight_decay = 0.01

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
            weight_decay=weight_decay,
            beta1=beta1,
            beta2=beta2,
            ssd_shards=ssd_shards,
            optimizer=OptimType.PARTIAL_ROWWISE_ADAM,
            cache_set_scale=0.2,
            pooling_mode=pooling_mode,
            weights_precision=weights_precision,
            output_dtype=output_dtype,
            share_table=True,
            bulk_init_chunk_size=bulk_init_chunk_size,
            lazy_bulk_init_enabled=lazy_bulk_init_enabled,
            optimizer_state_dtypes={
                "momentum1": m1_dtype,
                "momentum2": m2_dtype,
            },
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

        # Execute backward
        self.execute_ssd_backward_(
            output_ref_list,
            output,
            B,
            D,
            pooling_mode,
            batch_size_per_feature_per_rank,
        )

        emb.flush()

        tolerance = 1.0e-2

        split_optimizer_states = self.split_optimizer_states_(emb)

        # Compare emb state dict with expected values from nn.EmbeddingBag
        emb_state_dict, _, _, _ = emb.split_embedding_weights(no_snapshot=False)
        for f, t in self.get_physical_table_arg_indices_(emb.feature_table_map):
            (
                m1,
                m2,
            ) = split_optimizer_states[t]
            # Some optimizers have non-float momentum values
            # pyre-ignore[16]
            ref_grad = emb_ref[f].weight.grad.cpu().to_dense()
            ref_weights = emb_ref[f].weight.cpu()

            # Compare momentum2 values: (1 - beta2) * dL^2
            m2_ref = (ref_grad.pow(2).mean(dim=1)) * (1.0 - beta2)
            self.assert_close_(m2, m2_ref)

            # Compare momentum1 values: (1 - beta1) * dL
            m1_ref = ref_grad * (1.0 - beta1)
            self.assert_close_(m1, m1_ref)

            # Bias corrections
            iter_ = emb.iter.item()
            v_hat_t = m2_ref / (1 - beta2**iter_)
            v_hat_t = v_hat_t.view(v_hat_t.numel(), 1)
            m_hat_t = m1_ref / (1 - beta1**iter_)

            # Weight update
            ref_weights_updated = (
                torch.addcdiv(
                    ref_weights,
                    value=-lr,
                    tensor1=m_hat_t,
                    tensor2=v_hat_t.sqrt_().add_(eps),
                )
                - lr * weight_decay * ref_weights
            )

            # Compare weights
            torch.testing.assert_close(
                # pyre-fixme [16]
                emb_state_dict[t].full_tensor().float(),
                ref_weights_updated,
                atol=tolerance,
                rtol=tolerance,
            )

    @given(
        **default_st,
        **{
            "m1_dtype": st.sampled_from([SparseType.BF16, SparseType.FP32]),
            "m2_dtype": st.sampled_from([SparseType.BF16, SparseType.FP32]),
        },
        num_buckets=st.integers(min_value=10, max_value=15),
        enable_optimizer_offloading=st.booleans(),
        backend_type=st.sampled_from([BackendType.SSD, BackendType.DRAM]),
        bulk_init_chunk_size=st.sampled_from([0, 204800]),
        lazy_bulk_init_enabled=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_ssd_emb_state_dict_adam(
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        weighted: bool,
        pooling_mode: PoolingMode,
        weights_precision: SparseType,
        output_dtype: SparseType,
        m1_dtype: SparseType,
        m2_dtype: SparseType,
        bulk_init_chunk_size: int,
        lazy_bulk_init_enabled: bool,
        # pyre-ignore[2]
        **kwargs,
    ) -> None:
        # Constants
        lr = 0.5
        eps = 0.2
        ssd_shards = 2
        beta1 = 0.9
        beta2 = 0.99
        weight_decay = 0.01

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
            weight_decay=weight_decay,
            beta1=beta1,
            beta2=beta2,
            ssd_shards=ssd_shards,
            optimizer=OptimType.ADAM,
            cache_set_scale=0.2,
            pooling_mode=pooling_mode,
            weights_precision=weights_precision,
            output_dtype=output_dtype,
            share_table=True,
            bulk_init_chunk_size=bulk_init_chunk_size,
            lazy_bulk_init_enabled=lazy_bulk_init_enabled,
            optimizer_state_dtypes={
                "momentum1": m1_dtype,
                "momentum2": m2_dtype,
            },
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

        # Execute backward
        self.execute_ssd_backward_(
            output_ref_list,
            output,
            B,
            D,
            pooling_mode,
            batch_size_per_feature_per_rank,
        )

        emb.flush()

        tolerance = 1.0e-2

        split_optimizer_states = self.split_optimizer_states_(emb)

        # Compare emb state dict with expected values from nn.EmbeddingBag
        emb_state_dict, _, _, _ = emb.split_embedding_weights(no_snapshot=False)
        for f, t in self.get_physical_table_arg_indices_(emb.feature_table_map):
            (m1, m2) = split_optimizer_states[t]
            # Some optimizers have non-float momentum values
            # pyre-ignore[16]
            ref_grad = emb_ref[f].weight.grad.cpu().to_dense()
            ref_weights = emb_ref[f].weight.cpu()

            # Compare momentum2 values: (1 - beta2) * dL^2
            m2_ref = ref_grad.pow(2) * (1.0 - beta2)
            self.assert_close_(m2, m2_ref)

            # Compare momentum1 values: (1 - beta1) * dL
            m1_ref = ref_grad * (1.0 - beta1)
            self.assert_close_(m1, m1_ref)

            # Bias corrections
            iter_ = emb.iter.item()
            v_hat_t = m2_ref / (1 - beta2**iter_)
            m_hat_t = m1_ref / (1 - beta1**iter_)

            # Weight update
            ref_weights_updated = (
                torch.addcdiv(
                    ref_weights,
                    value=-lr,
                    tensor1=m_hat_t,
                    tensor2=v_hat_t.sqrt_().add_(eps),
                )
                - lr * weight_decay * ref_weights
            )

            # Compare weights
            torch.testing.assert_close(
                # pyre-fixme [16]
                emb_state_dict[t].full_tensor().float(),
                ref_weights_updated,
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
        enable_raw_embedding_streaming: bool = False,
        num_iterations: int = 10,
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
            enable_raw_embedding_streaming=enable_raw_embedding_streaming,
        )

        optimizer_states_ref = [
            [s.clone().float() for s in states]
            for states in self.split_optimizer_states_(emb)
        ]

        Es = [emb.embedding_specs[t][0] for t in range(T)]

        tolerance = (
            1.0e-4
            if weights_precision == SparseType.FP32 and output_dtype == SparseType.FP32
            else 1.0e-2
        )

        batches = []
        for _it in range(num_iterations):
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

        force_flush = flush_location == FlushLocation.ALL

        if force_flush or flush_location == FlushLocation.BEFORE_TRAINING:
            emb.flush()

        # pyre-ignore[53]
        def _prefetch(b_it: int) -> int:
            if not explicit_prefetch or b_it >= num_iterations:
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

        for it in range(num_iterations):
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

            # Prefetch between forward and backward
            if (
                prefetch_pipeline
                and prefetch_location == PrefetchLocation.BETWEEN_FWD_BWD
            ):
                b_it = _prefetch(b_it)

            # Zero out weight grad
            for f, _ in self.get_physical_table_arg_indices_(emb.feature_table_map):
                emb_ref[f].weight.grad = None

            # Execute backward
            self.execute_ssd_backward_(
                output_ref_list,
                output,
                B,
                D,
                pooling_mode,
                batch_size_per_feature_per_rank,
            )

            if force_flush or flush_location == FlushLocation.AFTER_BWD:
                emb.flush()

            # Compare optimizer states
            split_optimizer_states = self.split_optimizer_states_(emb)
            for f, t in self.get_physical_table_arg_indices_(emb.feature_table_map):
                (optim_state_r,) = optimizer_states_ref[t]
                (optim_state_t,) = split_optimizer_states[t]
                emb_r = emb_ref[f]

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

    @given(
        **default_st,
        num_buckets=st.integers(min_value=10, max_value=15),
        backend_type=st.sampled_from([BackendType.SSD, BackendType.DRAM]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_kv_db_forward(
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
        num_buckets: int,
        backend_type: BackendType,
    ) -> None:
        trigger_bounds_check = False  # don't stimulate boundary check cases
        assume(not weighted or pooling_mode == PoolingMode.SUM)
        assume(not mixed_B or pooling_mode != PoolingMode.NONE)
        # Generate embedding modules
        (
            emb,
            emb_ref,
            Es,
            _,
            bucket_offsets,
            bucket_sizes,
        ) = self.generate_kvzch_tbes(
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
            num_buckets=num_buckets,
            backend_type=backend_type,
        )

        # Generate inputs
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
            bucket_offsets=bucket_offsets,
            bucket_sizes=bucket_sizes,
            is_kv_tbes=True,
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

    @given(
        **default_st,
        num_buckets=st.integers(min_value=10, max_value=15),
        enable_optimizer_offloading=st.booleans(),
        backend_type=st.sampled_from([BackendType.SSD, BackendType.DRAM]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_kv_emb_state_dict(
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
        num_buckets: int,
        enable_optimizer_offloading: bool,
        backend_type: BackendType,
    ) -> None:
        # Constants
        lr = 0.5
        eps = 0.2
        ssd_shards = 2

        trigger_bounds_check = False  # don't stimulate boundary check cases
        assume(not weighted or pooling_mode == PoolingMode.SUM)
        assume(not mixed_B or pooling_mode != PoolingMode.NONE)

        # Generate embedding modules and inputs
        (
            emb,
            emb_ref,
            Es,
            _,
            bucket_offsets,
            bucket_sizes,
        ) = self.generate_kvzch_tbes(
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
            num_buckets=num_buckets,
            enable_optimizer_offloading=enable_optimizer_offloading,
            backend_type=backend_type,
            mixed=True,
        )

        # Generate inputs
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
            bucket_offsets=bucket_offsets,
            bucket_sizes=bucket_sizes,
            is_kv_tbes=True,
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

        # Execute backward
        self.execute_ssd_backward_(
            output_ref_list,
            output,
            B,
            D,
            pooling_mode,
            batch_size_per_feature_per_rank,
        )

        emb.flush()

        tolerance = (
            1.0e-4
            if weights_precision == SparseType.FP32 and output_dtype == SparseType.FP32
            else 1.0e-2
        )

        split_optimizer_states = []

        # Compare emb state dict with expected values from nn.EmbeddingBag
        (
            emb_state_dict_list,
            bucket_asc_ids_list,
            num_active_id_per_bucket_list,
            metadata_list,
        ) = emb.split_embedding_weights(no_snapshot=False, should_flush=True)

        for s in emb.split_optimizer_states(
            bucket_asc_ids_list, no_snapshot=False, should_flush=True
        ):
            split_optimizer_states.append(s)

        # Compare optimizer states
        for f, t in self.get_physical_table_arg_indices_(emb.feature_table_map):
            # pyre-fixme[16]: Optional type has no attribute `float`
            ref_optimizer_state = emb_ref[f].weight.grad.float().to_dense().pow(2)

            # pyre-fixme[16]: Undefined attribute: `Optional` has no attribute `__getitem__`.
            ref_opt_mean = ref_optimizer_state[bucket_asc_ids_list[t].view(-1)].mean(
                dim=1
            )
            torch.testing.assert_close(
                # NOTE: The [0] index is a hack since the test is fixed to use
                # EXACT_ROWWISE_ADAGRAD optimizer.  The test in general should
                # be upgraded in the future to support multiple optimizers
                split_optimizer_states[t][0].float(),
                ref_opt_mean.cpu(),
                atol=tolerance,
                rtol=tolerance,
            )

        for feature_index, table_index in self.get_physical_table_arg_indices_(
            emb.feature_table_map
        ):
            """
            validate bucket_asc_ids_list and num_active_id_per_bucket_list
            """
            bucket_asc_id = bucket_asc_ids_list[table_index]
            num_active_id_per_bucket = num_active_id_per_bucket_list[table_index]

            bucket_id_start = bucket_offsets[table_index][0]
            bucket_id_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
                num_active_id_per_bucket.view(-1)
            )
            for bucket_idx, id_count in enumerate(num_active_id_per_bucket):
                bucket_id = bucket_idx + bucket_id_start
                active_id_cnt = 0
                for idx in range(
                    bucket_id_offsets[bucket_idx],
                    bucket_id_offsets[bucket_idx + 1],
                ):
                    # for chunk-based hashing
                    self.assertEqual(
                        bucket_id, bucket_asc_id[idx] // bucket_sizes[table_index]
                    )
                    active_id_cnt += 1
                self.assertEqual(active_id_cnt, id_count)

            """
            validate embeddings
            """
            num_ids = len(bucket_asc_ids_list[table_index])
            emb_r_w = emb_ref[feature_index].weight[
                bucket_asc_ids_list[table_index].view(-1)
            ]
            emb_r_w_g = (
                emb_ref[feature_index]
                .weight.grad.float()
                .to_dense()[bucket_asc_ids_list[table_index].view(-1)]
            )
            self.assertLess(table_index, len(emb_state_dict_list))
            assert len(split_optimizer_states[table_index][0]) == num_ids
            (m1,) = split_optimizer_states[table_index]
            new_ref_weight = torch.addcdiv(
                emb_r_w.float(),
                value=-lr,
                tensor1=emb_r_w_g,
                tensor2=m1.float().sqrt_().add_(eps).view(num_ids, 1).cuda(),
            ).cpu()

            emb_w = (
                emb_state_dict_list[table_index]
                .narrow(0, 0, bucket_asc_ids_list[table_index].size(0))
                .float()
            )
            torch.testing.assert_close(
                emb_w,
                new_ref_weight,
                atol=tolerance,
                rtol=tolerance,
            )

            self.assertTrue(len(metadata_list[table_index].size()) == 2)

    @given(
        **default_st,
        **{
            "m1_dtype": st.sampled_from([SparseType.BF16, SparseType.FP32]),
            "m2_dtype": st.sampled_from([SparseType.BF16, SparseType.FP32]),
        },
        num_buckets=st.integers(min_value=10, max_value=15),
        enable_optimizer_offloading=st.booleans(),
        backend_type=st.sampled_from([BackendType.SSD, BackendType.DRAM]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_kv_emb_state_dict_partial_rowwise_adam(
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
        m1_dtype: SparseType,
        m2_dtype: SparseType,
        share_table: bool,
        trigger_bounds_check: bool,
        mixed_B: bool,
        num_buckets: int,
        enable_optimizer_offloading: bool,
        backend_type: BackendType,
    ) -> None:
        assume(not weighted or pooling_mode == PoolingMode.SUM)
        # VBE is currently not supported for PARTIAL_ROWWISE_ADAM optimizer
        assume(not mixed_B)
        # Don't stimulate boundary check cases
        trigger_bounds_check = False

        # Constants
        lr = 0.5
        eps = 0.2
        ssd_shards = 2
        beta1 = 0.9
        beta2 = 0.99
        weight_decay = 0.01

        # Generate embedding modules and inputs
        (
            emb,
            emb_ref,
            Es,
            _,
            bucket_offsets,
            bucket_sizes,
        ) = self.generate_kvzch_tbes(
            T,
            D,
            B,
            log_E,
            L,
            weighted,
            lr=lr,
            eps=eps,
            weight_decay=weight_decay,
            beta1=beta1,
            beta2=beta2,
            ssd_shards=ssd_shards,
            optimizer=OptimType.PARTIAL_ROWWISE_ADAM,
            cache_set_scale=cache_set_scale,
            pooling_mode=pooling_mode,
            weights_precision=weights_precision,
            output_dtype=output_dtype,
            share_table=share_table,
            num_buckets=num_buckets,
            enable_optimizer_offloading=enable_optimizer_offloading,
            backend_type=backend_type,
            optimizer_state_dtypes={
                "momentum1": m1_dtype,
                "momentum2": m2_dtype,
            },
        )

        # Generate inputs
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
            bucket_offsets=bucket_offsets,
            bucket_sizes=bucket_sizes,
            is_kv_tbes=True,
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

        # Execute backward
        self.execute_ssd_backward_(
            output_ref_list,
            output,
            B,
            D,
            pooling_mode,
            batch_size_per_feature_per_rank,
        )

        emb.flush()

        split_optimizer_states = []

        # Compare emb state dict with expected values from nn.EmbeddingBag
        emb_state_dict_list, bucket_asc_ids_list, num_active_id_per_bucket_list, _ = (
            emb.split_embedding_weights(no_snapshot=False, should_flush=True)
        )

        for s in emb.split_optimizer_states(
            bucket_asc_ids_list, no_snapshot=False, should_flush=True
        ):
            split_optimizer_states.append(s)

        # Compare optimizer states
        for f, t in self.get_physical_table_arg_indices_(emb.feature_table_map):
            (m1, m2) = split_optimizer_states[t]
            # Some optimizers have non-float momentum values
            # pyre-ignore[16]
            ref_grad = emb_ref[f].weight.grad.cpu().to_dense()
            ref_weights = emb_ref[f].weight.cpu()

            # Compare momentum2 values: (1 - beta2) * dL^2
            m2_ref = (ref_grad.pow(2).mean(dim=1)) * (1.0 - beta2)
            # Get only the subset of rows based on bucket_asc_ids_list[t]
            # pyre-ignore [16]
            m2_ref = m2_ref[bucket_asc_ids_list[t].view(-1)]
            self.assert_close_(m2, m2_ref)

            # Compare momentum1 values: (1 - beta1) * dL
            m1_ref = ref_grad * (1.0 - beta1)
            # Get only the subset of rows based on bucket_asc_ids_list[t]
            m1_ref = m1_ref[bucket_asc_ids_list[t].view(-1)]

            # Print which rows are different between m1 and m1_ref
            print_different_rows(
                m1, m1_ref, atol=1.0e-2, rtol=1.0e-2, name1="m1", name2="m1_ref"
            )

            self.assert_close_(m1, m1_ref)
            ####################################################################
            # Compare weight values
            ####################################################################

            # Re-index the weights according to bucket ids
            ref_weights = ref_weights[bucket_asc_ids_list[t].view(-1)]

            # Bias corrections
            iter_ = emb.iter.item()
            v_hat_t = m2_ref / (1 - beta2**iter_)
            v_hat_t = v_hat_t.view(v_hat_t.numel(), 1)
            m_hat_t = m1_ref / (1 - beta1**iter_)

            # Manually update the ref weights
            ref_weights_updated = (
                torch.addcdiv(
                    ref_weights,
                    value=-lr,
                    tensor1=m_hat_t,
                    tensor2=v_hat_t.sqrt_().add_(eps),
                )
                - lr * weight_decay * ref_weights
            )

            # Fetch the updated weights from SSDTableBatchedEmbeddingBags
            emb_w = (
                emb_state_dict_list[t]
                .narrow(0, 0, bucket_asc_ids_list[t].size(0))
                .float()
            )

            # Compare weights
            tolerance = 1.0e-2
            torch.testing.assert_close(
                emb_w,
                ref_weights_updated,
                atol=tolerance,
                rtol=tolerance,
            )

    @given(
        **default_st,
        **{
            "m1_dtype": st.sampled_from([SparseType.BF16, SparseType.FP32]),
            "m2_dtype": st.sampled_from([SparseType.BF16, SparseType.FP32]),
        },
        num_buckets=st.integers(min_value=10, max_value=15),
        enable_optimizer_offloading=st.booleans(),
        backend_type=st.sampled_from([BackendType.SSD, BackendType.DRAM]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_kv_emb_state_dict_adam(
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
        m1_dtype: SparseType,
        m2_dtype: SparseType,
        share_table: bool,
        trigger_bounds_check: bool,
        mixed_B: bool,
        num_buckets: int,
        enable_optimizer_offloading: bool,
        backend_type: BackendType,
    ) -> None:
        assume(not weighted or pooling_mode == PoolingMode.SUM)
        # VBE is currently not supported for PARTIAL_ROWWISE_ADAM optimizer
        assume(not mixed_B)
        # Don't stimulate boundary check cases
        trigger_bounds_check = False

        # Constants
        lr = 0.5
        eps = 0.2
        ssd_shards = 2
        beta1 = 0.9
        beta2 = 0.99
        weight_decay = 0.01

        # Generate embedding modules and inputs
        (
            emb,
            emb_ref,
            Es,
            _,
            bucket_offsets,
            bucket_sizes,
        ) = self.generate_kvzch_tbes(
            T,
            D,
            B,
            log_E,
            L,
            weighted,
            lr=lr,
            eps=eps,
            weight_decay=weight_decay,
            beta1=beta1,
            beta2=beta2,
            ssd_shards=ssd_shards,
            optimizer=OptimType.ADAM,
            cache_set_scale=cache_set_scale,
            pooling_mode=pooling_mode,
            weights_precision=weights_precision,
            output_dtype=output_dtype,
            share_table=share_table,
            num_buckets=num_buckets,
            enable_optimizer_offloading=enable_optimizer_offloading,
            backend_type=backend_type,
            optimizer_state_dtypes={
                "momentum1": m1_dtype,
                "momentum2": m2_dtype,
            },
        )

        # Generate inputs
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
            bucket_offsets=bucket_offsets,
            bucket_sizes=bucket_sizes,
            is_kv_tbes=True,
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

        # Execute backward
        self.execute_ssd_backward_(
            output_ref_list,
            output,
            B,
            D,
            pooling_mode,
            batch_size_per_feature_per_rank,
        )

        emb.flush()

        split_optimizer_states = []

        # Compare emb state dict with expected values from nn.EmbeddingBag
        emb_state_dict_list, bucket_asc_ids_list, num_active_id_per_bucket_list, _ = (
            emb.split_embedding_weights(no_snapshot=False, should_flush=True)
        )

        for s in emb.split_optimizer_states(
            bucket_asc_ids_list, no_snapshot=False, should_flush=True
        ):
            split_optimizer_states.append(s)

        # Compare optimizer states
        for f, t in self.get_physical_table_arg_indices_(emb.feature_table_map):
            (m1, m2) = split_optimizer_states[t]
            # Some optimizers have non-float momentum values
            # pyre-ignore[16]
            ref_grad = emb_ref[f].weight.grad.cpu().to_dense()
            ref_weights = emb_ref[f].weight.cpu()

            # Compare momentum2 values: (1 - beta2) * dL^2
            m2_ref = ref_grad.pow(2) * (1.0 - beta2)
            # Get only the subset of rows based on bucket_asc_ids_list[t]
            # pyre-ignore [16]
            m2_ref = m2_ref[bucket_asc_ids_list[t].view(-1)]
            self.assert_close_(m2, m2_ref)

            # Compare momentum1 values: (1 - beta1) * dL
            m1_ref = ref_grad * (1.0 - beta1)
            # Get only the subset of rows based on bucket_asc_ids_list[t]
            m1_ref = m1_ref[bucket_asc_ids_list[t].view(-1)]
            # Print which rows are different between m1 and m1_ref
            print_different_rows(
                m1, m1_ref, atol=1.0e-2, rtol=1.0e-2, name1="m1", name2="m1_ref"
            )
            self.assert_close_(m1, m1_ref)

            ####################################################################
            # Compare weight values
            ####################################################################

            # Re-index the weights according to bucket ids
            ref_weights = ref_weights[bucket_asc_ids_list[t].view(-1)]

            # Bias corrections
            iter_ = emb.iter.item()
            v_hat_t = m2_ref / (1 - beta2**iter_)
            m_hat_t = m1_ref / (1 - beta1**iter_)

            # Manually update the ref weights
            ref_weights_updated = (
                torch.addcdiv(
                    ref_weights,
                    value=-lr,
                    tensor1=m_hat_t,
                    tensor2=v_hat_t.sqrt_().add_(eps),
                )
                - lr * weight_decay * ref_weights
            )

            # Fetch the updated weights from SSDTableBatchedEmbeddingBags
            emb_w = (
                emb_state_dict_list[t]
                .narrow(0, 0, bucket_asc_ids_list[t].size(0))
                .float()
            )

            # Compare weights
            tolerance = 1.0e-2
            torch.testing.assert_close(
                emb_w,
                ref_weights_updated,
                atol=tolerance,
                rtol=tolerance,
            )

    @given(
        **default_st,
        **{
            "m1_dtype": st.sampled_from([SparseType.BF16, SparseType.FP32]),
            "m2_dtype": st.sampled_from([SparseType.BF16, SparseType.FP32]),
        },
        num_buckets=st.integers(min_value=10, max_value=15),
        enable_optimizer_offloading=st.just(True),
        backend_type=st.sampled_from([BackendType.DRAM]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_kv_emb_state_dict_partial_rowwise_adam_whole_row(
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
        m1_dtype: SparseType,
        m2_dtype: SparseType,
        share_table: bool,
        trigger_bounds_check: bool,
        mixed_B: bool,
        num_buckets: int,
        enable_optimizer_offloading: bool,
        backend_type: BackendType,
    ) -> None:
        assume(not weighted or pooling_mode == PoolingMode.SUM)
        # VBE is currently not supported for PARTIAL_ROWWISE_ADAM optimizer
        assume(not mixed_B)
        # Don't stimulate boundary check cases
        trigger_bounds_check = False

        # Constants
        lr = 0.5
        eps = 0.2
        ssd_shards = 2
        beta1 = 0.9
        beta2 = 0.99
        weight_decay = 0.01
        metaheader_dim = 16 // (
            weights_precision.as_dtype().itemsize
        )  # 16-byte metaheader

        # Generate embedding modules and inputs
        (
            emb,
            emb_ref,
            Es,
            _,
            bucket_offsets,
            bucket_sizes,
        ) = self.generate_kvzch_tbes(
            T,
            D,
            B,
            log_E,
            L,
            weighted,
            lr=lr,
            eps=eps,
            weight_decay=weight_decay,
            beta1=beta1,
            beta2=beta2,
            ssd_shards=ssd_shards,
            optimizer=OptimType.PARTIAL_ROWWISE_ADAM,
            cache_set_scale=cache_set_scale,
            pooling_mode=pooling_mode,
            weights_precision=weights_precision,
            output_dtype=output_dtype,
            share_table=share_table,
            num_buckets=num_buckets,
            enable_optimizer_offloading=enable_optimizer_offloading,
            backend_type=backend_type,
            backend_return_whole_row=True,
            optimizer_state_dtypes={
                "momentum1": m1_dtype,
                "momentum2": m2_dtype,
            },
        )

        # Generate inputs
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
            bucket_offsets=bucket_offsets,
            bucket_sizes=bucket_sizes,
            is_kv_tbes=True,
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

        # Execute backward
        self.execute_ssd_backward_(
            output_ref_list,
            output,
            B,
            D,
            pooling_mode,
            batch_size_per_feature_per_rank,
        )

        emb.flush()

        split_optimizer_states = []

        # Compare emb state dict with expected values from nn.EmbeddingBag
        emb_state_dict_list, bucket_asc_ids_list, _, _ = emb.split_embedding_weights(
            no_snapshot=False, should_flush=True
        )

        for s in emb.split_optimizer_states(
            bucket_asc_ids_list, no_snapshot=False, should_flush=True
        ):
            split_optimizer_states.append(s)

        # Compare optimizer states
        table_offset = 0
        for f, t in self.get_physical_table_arg_indices_(emb.feature_table_map):
            (m1, m2) = split_optimizer_states[t]
            # Some optimizers have non-float momentum values
            # pyre-ignore[16]
            ref_grad = emb_ref[f].weight.grad.cpu().to_dense()
            ref_weights = emb_ref[f].weight.cpu()

            # Compare momentum2 values: (1 - beta2) * dL^2
            m2_ref = (ref_grad.pow(2).mean(dim=1)) * (1.0 - beta2)
            # Get only the subset of rows based on bucket_asc_ids_list[t]
            # pyre-ignore [16]
            m2_ref = m2_ref[bucket_asc_ids_list[t].view(-1)]
            m2_tensor = m2.full_tensor().view(m2_dtype.as_dtype())
            if m2_tensor.shape[1] > 1:
                # HACK: the KVTensorWrapper does not support dtypes other than weight dtype,
                # so when the optimizer state dtype is smaller than weight dtype, we'll save more
                # data than needed. Before we support the optimizer dtype from backend, we workaround the
                # verification in test:
                # the m2 dtype could be smaller than weight dtype, so we need to get the first column as the
                # m2 value, and then convert it to the same dtype as m2_ref and then compare with it.
                m2_tensor = m2_tensor[:, 0]
            # Compare the weight part
            self.assert_close_(m2_tensor.view(-1), m2_ref)

            # Compare momentum1 values: (1 - beta1) * dL
            m1_ref = ref_grad * (1.0 - beta1)
            # Get only the subset of rows based on bucket_asc_ids_list[t]
            m1_ref = m1_ref[bucket_asc_ids_list[t].view(-1)]
            m1_tensor = m1.full_tensor().view(m1_dtype.as_dtype())

            # Print which rows are different between m1 and m1_ref
            print_different_rows(
                m1_tensor, m1_ref, atol=1.0e-2, rtol=1.0e-2, name1="m1", name2="m1_ref"
            )

            self.assert_close_(m1_tensor, m1_ref)

            ####################################################################
            # Compare weight values
            ####################################################################
            # Re-index the weights according to bucket ids
            ref_weights = ref_weights[bucket_asc_ids_list[t].view(-1)]
            iter_ = emb.iter.item()
            v_hat_t = m2_ref / (1 - beta2**iter_)
            v_hat_t = v_hat_t.view(v_hat_t.numel(), 1)
            m_hat_t = m1_ref / (1 - beta1**iter_)

            # Manually update the ref weights
            ref_weights_updated = (
                torch.addcdiv(
                    ref_weights,
                    value=-lr,
                    tensor1=m_hat_t,
                    tensor2=v_hat_t.sqrt_().add_(eps),
                )
                - lr * weight_decay * ref_weights
            )

            """
            validate the whole embeddings rows (metaheader + weight + m2 + m1)
            """

            emb_w = emb_state_dict_list[t].narrow(0, 0, bucket_asc_ids_list[t].size(0))
            emb_d = emb.embedding_specs[t][1]

            # Lookup the byte offsets for each optimizer state
            optimizer_state_byte_offsets = emb.optimizer.byte_offsets_along_row(
                emb_d,
                emb.weights_precision,
                emb.optimizer_state_dtypes,
            )

            m2_start, m2_end = optimizer_state_byte_offsets["momentum2"]
            m1_start, m1_end = optimizer_state_byte_offsets["momentum1"]
            meta_bytes = metaheader_dim * weights_precision.as_dtype().itemsize
            m2_from_emb_w = (
                emb_w.view(dtype=torch.uint8)[
                    :,
                    meta_bytes + m2_start : meta_bytes + m2_end,
                ]
                .view(m2_dtype.as_dtype())
                .view(-1)
            )
            m1_from_emb_w = emb_w.view(dtype=torch.uint8)[
                :,
                meta_bytes + m1_start : meta_bytes + m1_end,
            ].view(m1_dtype.as_dtype())
            self.assert_close_(m2_from_emb_w, m2_ref)
            self.assert_close_(m1_from_emb_w, m1_ref)

            # Copmare the id part
            id_extracted_from_emb_w = (
                emb_w[:, 0 : metaheader_dim // 2].view(torch.int64).view(-1)
            )
            self.assert_close_(
                id_extracted_from_emb_w, bucket_asc_ids_list[t].view(-1) + table_offset
            )
            # Compare the weight part
            tolerance = 1.0e-2
            torch.testing.assert_close(
                emb_w[:, metaheader_dim : metaheader_dim + emb_d].float(),
                ref_weights_updated,
                atol=tolerance,
                rtol=tolerance,
            )

            table_offset += VIRTUAL_TABLE_ROWS

    @given(
        **default_st,
        **{
            "m1_dtype": st.sampled_from([SparseType.BF16, SparseType.FP32]),
            "m2_dtype": st.sampled_from([SparseType.BF16, SparseType.FP32]),
        },
        num_buckets=st.integers(min_value=10, max_value=15),
        enable_optimizer_offloading=st.just(True),
        backend_type=st.sampled_from([BackendType.DRAM]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_kv_emb_state_dict_adam_whole_row(
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
        m1_dtype: SparseType,
        m2_dtype: SparseType,
        share_table: bool,
        trigger_bounds_check: bool,
        mixed_B: bool,
        num_buckets: int,
        enable_optimizer_offloading: bool,
        backend_type: BackendType,
    ) -> None:
        assume(not weighted or pooling_mode == PoolingMode.SUM)
        # VBE is currently not supported for PARTIAL_ROWWISE_ADAM optimizer
        assume(not mixed_B)
        # Don't stimulate boundary check cases
        trigger_bounds_check = False

        # Constants
        lr = 0.5
        eps = 0.2
        ssd_shards = 2
        beta1 = 0.9
        beta2 = 0.99
        weight_decay = 0.01
        metaheader_dim = 16 // (
            weights_precision.as_dtype().itemsize
        )  # 16-byte metaheader

        # Generate embedding modules and inputs
        (
            emb,
            emb_ref,
            Es,
            _,
            bucket_offsets,
            bucket_sizes,
        ) = self.generate_kvzch_tbes(
            T,
            D,
            B,
            log_E,
            L,
            weighted,
            lr=lr,
            eps=eps,
            weight_decay=weight_decay,
            beta1=beta1,
            beta2=beta2,
            ssd_shards=ssd_shards,
            optimizer=OptimType.ADAM,
            cache_set_scale=cache_set_scale,
            pooling_mode=pooling_mode,
            weights_precision=weights_precision,
            output_dtype=output_dtype,
            share_table=share_table,
            num_buckets=num_buckets,
            enable_optimizer_offloading=enable_optimizer_offloading,
            backend_type=backend_type,
            backend_return_whole_row=True,
            optimizer_state_dtypes={
                "momentum1": m1_dtype,
                "momentum2": m2_dtype,
            },
        )

        # Generate inputs
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
            bucket_offsets=bucket_offsets,
            bucket_sizes=bucket_sizes,
            is_kv_tbes=True,
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

        # Execute backward
        self.execute_ssd_backward_(
            output_ref_list,
            output,
            B,
            D,
            pooling_mode,
            batch_size_per_feature_per_rank,
        )

        emb.flush()

        split_optimizer_states = []

        # Compare emb state dict with expected values from nn.EmbeddingBag
        emb_state_dict_list, bucket_asc_ids_list, _, _ = emb.split_embedding_weights(
            no_snapshot=False, should_flush=True
        )

        for s in emb.split_optimizer_states(
            bucket_asc_ids_list, no_snapshot=False, should_flush=True
        ):
            split_optimizer_states.append(s)

        # Compare optimizer states
        table_offset = 0
        for f, t in self.get_physical_table_arg_indices_(emb.feature_table_map):
            ####################################################################
            # Compare optimizer states
            ####################################################################

            (m1, m2) = split_optimizer_states[t]
            # Some optimizers have non-float momentum values
            # pyre-ignore[16]
            ref_grad = emb_ref[f].weight.grad.cpu().to_dense()
            ref_weights = emb_ref[f].weight.cpu()

            # Compare momentum2 values: (1 - beta2) * dL^2
            m2_ref = ref_grad.pow(2) * (1.0 - beta2)
            # Get only the subset of rows based on bucket_asc_ids_list[t]
            # pyre-ignore[16]
            m2_ref = m2_ref[bucket_asc_ids_list[t].view(-1)]
            m2 = m2.full_tensor().view(m2_dtype.as_dtype())
            self.assert_close_(m2, m2_ref)

            # Compare momentum1 values: (1 - beta1) * dL
            m1_ref = ref_grad * (1.0 - beta1)
            # Get only the subset of rows based on bucket_asc_ids_list[t]
            m1_ref = m1_ref[bucket_asc_ids_list[t].view(-1)]
            m1 = m1.full_tensor().view(m1_dtype.as_dtype())
            # Print which rows are different between m1 and m1_ref
            print_different_rows(
                m1, m1_ref, atol=1.0e-2, rtol=1.0e-2, name1="m1", name2="m1_ref"
            )
            # self.assert_close_(m1_tensor, m1_ref)
            self.assert_close_(m1, m1_ref)

            ####################################################################
            # Compare weight values
            ####################################################################

            # Re-index the weights according to bucket ids
            ref_weights = ref_weights[bucket_asc_ids_list[t].view(-1)]

            # Bias corrections
            iter_ = emb.iter.item()
            v_hat_t = m2_ref / (1 - beta2**iter_)
            # v_hat_t = v_hat_t.view(v_hat_t.numel(), 1)
            m_hat_t = m1_ref / (1 - beta1**iter_)

            # Manually update the ref weights
            ref_weights_updated = (
                torch.addcdiv(
                    ref_weights,
                    value=-lr,
                    tensor1=m_hat_t,
                    tensor2=v_hat_t.sqrt_().add_(eps),
                )
                - lr * weight_decay * ref_weights
            )

            emb_w = emb_state_dict_list[t].narrow(0, 0, bucket_asc_ids_list[t].size(0))
            emb_d = emb.embedding_specs[t][1]

            # Lookup the byte offsets for each optimizer state
            optimizer_state_byte_offsets = emb.optimizer.byte_offsets_along_row(
                emb_d,
                emb.weights_precision,
                emb.optimizer_state_dtypes,
            )

            m1_start, m1_end = optimizer_state_byte_offsets["momentum1"]
            m2_start, m2_end = optimizer_state_byte_offsets["momentum2"]
            meta_bytes = metaheader_dim * weights_precision.as_dtype().itemsize
            m2_from_emb_w = emb_w.view(dtype=torch.uint8)[
                :,
                meta_bytes + m2_start : meta_bytes + m2_end,
            ].view(m2_dtype.as_dtype())
            m1_from_emb_w = emb_w.view(dtype=torch.uint8)[
                :,
                meta_bytes + m1_start : meta_bytes + m1_end,
            ].view(m1_dtype.as_dtype())
            self.assert_close_(m2_from_emb_w, m2_ref)
            self.assert_close_(m1_from_emb_w, m1_ref)

            # Copmare the id part
            id_extracted_from_emb_w = (
                emb_w[:, 0 : metaheader_dim // 2].view(torch.int64).view(-1)
            )
            self.assert_close_(
                id_extracted_from_emb_w, bucket_asc_ids_list[t].view(-1) + table_offset
            )
            # Compare the weight part
            tolerance = 1.0e-2
            torch.testing.assert_close(
                emb_w[:, metaheader_dim : metaheader_dim + emb_d].float(),
                ref_weights_updated,
                atol=tolerance,
                rtol=tolerance,
            )

            table_offset += VIRTUAL_TABLE_ROWS

    @given(
        **default_st,
        num_buckets=st.integers(min_value=10, max_value=15),
        enable_optimizer_offloading=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_kv_opt_state_w_offloading(
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
        num_buckets: int,
        enable_optimizer_offloading: bool,
    ) -> None:
        # Constants
        lr = 0.5
        eps = 0.2
        ssd_shards = 2

        trigger_bounds_check = False  # don't stimulate boundary check cases
        assume(not weighted or pooling_mode == PoolingMode.SUM)
        assume(not mixed_B or pooling_mode != PoolingMode.NONE)

        # TODO: check split_optimizer_states when optimizer offloading is ready
        # Generate embedding modules and inputs
        (
            emb,
            emb_ref,
            Es,
            _,
            bucket_offsets,
            bucket_sizes,
        ) = self.generate_kvzch_tbes(
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
            num_buckets=num_buckets,
            enable_optimizer_offloading=enable_optimizer_offloading,
        )

        # Generate inputs
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
            bucket_offsets=bucket_offsets,
            bucket_sizes=bucket_sizes,
            is_kv_tbes=True,
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

        # Execute backward
        self.execute_ssd_backward_(
            output_ref_list,
            output,
            B,
            D,
            pooling_mode,
            batch_size_per_feature_per_rank,
        )

        emb.flush()

        tolerance = (
            1.0e-4
            if weights_precision == SparseType.FP32 and output_dtype == SparseType.FP32
            else 1.0e-2
        )

        # Compare emb state dict with expected values from nn.EmbeddingBag
        emb_state_dict_list, bucket_asc_ids_list, num_active_id_per_bucket_list, _ = (
            emb.split_embedding_weights(no_snapshot=False, should_flush=True)
        )
        split_optimizer_states = emb.split_optimizer_states(
            bucket_asc_ids_list, no_snapshot=False
        )
        table_input_id_range = []
        for t, row in enumerate(Es):
            bucket_id_start = bucket_offsets[t][0]
            bucket_id_end = bucket_offsets[t][1]
            bucket_size = bucket_sizes[t]
            table_input_id_range.append(
                (
                    min(bucket_id_start * bucket_size, row),
                    min(bucket_id_end * bucket_size, row),
                )
            )
            # since we use ref_emb in dense format, the rows start from id 0
            self.assertEqual(table_input_id_range[-1][0], 0)

        # Compare optimizer states
        for f, t in self.get_physical_table_arg_indices_(emb.feature_table_map):
            # pyre-fixme[16]: Optional type has no attribute `float`.
            ref_emb = emb_ref[f].weight.grad.float().to_dense().pow(2).cpu()
            ref_optimizer_state = ref_emb.mean(dim=1)[
                table_input_id_range[t][0] : min(
                    table_input_id_range[t][1], emb_ref[f].weight.size(0)
                )
            ]
            # pyre-fixme[16]: Undefined attribute: `Optional` has no attribute `__getitem__`.
            ref_kv_opt = ref_optimizer_state[bucket_asc_ids_list[t]].view(-1)
            torch.testing.assert_close(
                # NOTE: The [0] index is a hack since the test is fixed to use
                # EXACT_ROWWISE_ADAGRAD optimizer.  The test in general should
                # be upgraded in the future to support multiple optimizers
                split_optimizer_states[t][0].float(),
                ref_kv_opt,
                atol=tolerance,
                rtol=tolerance,
            )

        for feature_index, table_index in self.get_physical_table_arg_indices_(
            emb.feature_table_map
        ):
            """
            validate bucket_asc_ids_list and num_active_id_per_bucket_list
            """
            bucket_asc_id = bucket_asc_ids_list[table_index]
            num_active_id_per_bucket = num_active_id_per_bucket_list[table_index]

            bucket_id_start = bucket_offsets[table_index][0]
            bucket_id_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
                num_active_id_per_bucket.view(-1)
            )
            for bucket_idx, id_count in enumerate(num_active_id_per_bucket):
                bucket_id = bucket_idx + bucket_id_start
                active_id_cnt = 0
                for idx in range(
                    bucket_id_offsets[bucket_idx],
                    bucket_id_offsets[bucket_idx + 1],
                ):
                    # for chunk-based hashing
                    self.assertEqual(
                        bucket_id, bucket_asc_id[idx] // bucket_sizes[table_index]
                    )
                    active_id_cnt += 1
                self.assertEqual(active_id_cnt, id_count)

            """
            validate embeddings
            """
            num_ids = len(bucket_asc_ids_list[table_index])
            emb_r_w = emb_ref[feature_index].weight[
                bucket_asc_ids_list[table_index].view(-1)
            ]
            emb_r_w_g = (
                emb_ref[feature_index]
                .weight.grad.float()
                .to_dense()[bucket_asc_ids_list[table_index].view(-1)]
            )
            self.assertLess(table_index, len(emb_state_dict_list))
            # NOTE: The [0] index is a hack since the test is fixed to use
            # EXACT_ROWWISE_ADAGRAD optimizer.  The test in general should
            # be upgraded in the future to support multiple optimizers
            assert len(split_optimizer_states[table_index][0]) == num_ids
            opt = split_optimizer_states[table_index][0]
            new_ref_weight = torch.addcdiv(
                emb_r_w.float(),
                value=-lr,
                tensor1=emb_r_w_g,
                tensor2=opt.float()
                .sqrt_()
                .add_(eps)
                .view(
                    num_ids,
                    1,
                )
                .cuda(),
            ).cpu()

            emb_w = (
                emb_state_dict_list[table_index]
                .narrow(0, 0, bucket_asc_ids_list[table_index].size(0))
                .float()
            )
            torch.testing.assert_close(
                emb_w,
                new_ref_weight,
                atol=tolerance,
                rtol=tolerance,
            )

    @given(
        **default_st,
        num_buckets=st.integers(min_value=10, max_value=15),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_kv_state_dict_w_backend_return_whole_row(
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
        num_buckets: int,
    ) -> None:
        # Constants
        lr = 0.5
        eps = 0.2
        ssd_shards = 2
        metaheader_dim = 16 // (weights_precision.bit_rate() // 8)  # 8-byte metaheader
        opt_dim = 4 // (weights_precision.bit_rate() // 8)  # 4-byte optimizer state

        trigger_bounds_check = False  # don't stimulate boundary check cases
        assume(not weighted or pooling_mode == PoolingMode.SUM)
        assume(not mixed_B or pooling_mode != PoolingMode.NONE)

        # Generate embedding modules and inputs
        (
            emb,
            emb_ref,
            Es,
            _,
            bucket_offsets,
            bucket_sizes,
        ) = self.generate_kvzch_tbes(
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
            num_buckets=num_buckets,
            backend_type=BackendType.DRAM,
            enable_optimizer_offloading=True,
            backend_return_whole_row=True,
        )

        # Generate inputs
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
            bucket_offsets=bucket_offsets,
            bucket_sizes=bucket_sizes,
            is_kv_tbes=True,
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

        # Execute backward
        self.execute_ssd_backward_(
            output_ref_list,
            output,
            B,
            D,
            pooling_mode,
            batch_size_per_feature_per_rank,
        )

        emb.flush()

        tolerance = (
            1.0e-4
            if weights_precision == SparseType.FP32 and output_dtype == SparseType.FP32
            else 1.0e-2
        )

        # Compare emb state dict with expected values from nn.EmbeddingBag
        emb_state_dict_list, bucket_asc_ids_list, num_active_id_per_bucket_list, _ = (
            emb.split_embedding_weights(no_snapshot=False, should_flush=True)
        )
        split_optimizer_states = emb.split_optimizer_states(
            bucket_asc_ids_list, no_snapshot=False
        )
        table_input_id_range = []
        for t, row in enumerate(Es):
            bucket_id_start = bucket_offsets[t][0]
            bucket_id_end = bucket_offsets[t][1]
            bucket_size = bucket_sizes[t]
            table_input_id_range.append(
                (
                    min(bucket_id_start * bucket_size, row),
                    min(bucket_id_end * bucket_size, row),
                )
            )
            # since we use ref_emb in dense format, the rows start from id 0
            self.assertEqual(table_input_id_range[-1][0], 0)

        """
        validate optimizer states
        """
        opt_validated = []
        for f, t in self.get_physical_table_arg_indices_(emb.feature_table_map):
            # pyre-fixme[16]: Optional type has no attribute `float`.
            ref_emb = emb_ref[f].weight.grad.float().to_dense().pow(2).cpu()
            ref_optimizer_state = ref_emb.mean(dim=1)[
                table_input_id_range[t][0] : min(
                    table_input_id_range[t][1], emb_ref[f].weight.size(0)
                )
            ]
            # pyre-fixme[16]: Undefined attribute: `Optional` has no attribute `__getitem__`.
            ref_kv_opt = ref_optimizer_state[bucket_asc_ids_list[t]].view(-1)
            opt = (
                # NOTE: The [0] index is a hack since the test is fixed to use
                # EXACT_ROWWISE_ADAGRAD optimizer.  The test in general should
                # be upgraded in the future to support multiple optimizers
                split_optimizer_states[t][0]
                .narrow(0, 0, bucket_asc_ids_list[t].size(0))
                .view(-1)
                .view(torch.float32)
                .float()
            )
            opt_validated.append(opt.clone().detach())
            torch.testing.assert_close(
                opt,
                ref_kv_opt,
                atol=tolerance,
                rtol=tolerance,
            )

        table_offset = 0
        for feature_index, table_index in self.get_physical_table_arg_indices_(
            emb.feature_table_map
        ):
            """
            validate bucket_asc_ids_list and num_active_id_per_bucket_list
            """
            bucket_asc_id = bucket_asc_ids_list[table_index]
            num_active_id_per_bucket = num_active_id_per_bucket_list[table_index]

            bucket_id_start = bucket_offsets[table_index][0]
            bucket_id_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
                num_active_id_per_bucket.view(-1)
            )
            for bucket_idx, id_count in enumerate(num_active_id_per_bucket):
                bucket_id = bucket_idx + bucket_id_start
                active_id_cnt = 0
                for idx in range(
                    bucket_id_offsets[bucket_idx],
                    bucket_id_offsets[bucket_idx + 1],
                ):
                    # for chunk-based hashing
                    self.assertEqual(
                        bucket_id, bucket_asc_id[idx] // bucket_sizes[table_index]
                    )
                    active_id_cnt += 1
                self.assertEqual(active_id_cnt, id_count)

            """
            validate the whole embeddings rows (metaheader + weight + opt)
            """
            num_ids = len(bucket_asc_ids_list[table_index])
            emb_r_w = emb_ref[feature_index].weight[
                bucket_asc_ids_list[table_index].view(-1)
            ]
            emb_r_w_g = (
                emb_ref[feature_index]
                .weight.grad.float()
                .to_dense()[bucket_asc_ids_list[table_index].view(-1)]
            )
            self.assertLess(table_index, len(emb_state_dict_list))
            # NOTE: The [0] index is a hack since the test is fixed to use
            # EXACT_ROWWISE_ADAGRAD optimizer.  The test in general should
            # be upgraded in the future to support multiple optimizers
            assert split_optimizer_states[table_index][0].size(0) == num_ids
            new_ref_weight = torch.addcdiv(
                emb_r_w.float(),
                value=-lr,
                tensor1=emb_r_w_g,
                tensor2=opt_validated[table_index]
                .clone()
                .sqrt_()
                .add_(eps)
                .view(
                    num_ids,
                    1,
                )
                .cuda(),
            ).cpu()

            emb_w = emb_state_dict_list[table_index].narrow(
                0, 0, bucket_asc_ids_list[table_index].size(0)
            )
            # Compare the opt part
            opt_extracted_from_emb_w = (
                emb_w[:, (metaheader_dim + D * 4) : (metaheader_dim + D * 4) + opt_dim]
                .view(torch.float32)
                .view(-1)
            )
            torch.testing.assert_close(
                opt_extracted_from_emb_w,
                opt_validated[table_index],
                atol=tolerance,
                rtol=tolerance,
            )

            # Copmare the id part
            id_extracted_from_emb_w = (
                emb_w[:, 0 : metaheader_dim // 2].view(torch.int64).view(-1)
            )
            torch.testing.assert_close(
                id_extracted_from_emb_w,
                bucket_asc_ids_list[table_index].view(-1) + table_offset,
                atol=tolerance,
                rtol=tolerance,
            )

            # Compare the weight part
            torch.testing.assert_close(
                emb_w[:, metaheader_dim : metaheader_dim + D * 4].float(),
                new_ref_weight,
                atol=tolerance,
                rtol=tolerance,
            )

            table_offset += VIRTUAL_TABLE_ROWS

    @given(
        **default_st,
        num_buckets=st.integers(min_value=10, max_value=15),
        backend_type=st.sampled_from([BackendType.SSD, BackendType.DRAM]),
        enable_optimizer_offloading=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_apply_kv_state_dict(
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
        num_buckets: int,
        backend_type: BackendType,
        enable_optimizer_offloading: bool,
    ) -> None:
        # Constants
        lr = 0.5
        eps = 0.2
        ssd_shards = 2

        trigger_bounds_check = False  # don't stimulate boundary check cases
        assume(not weighted or pooling_mode == PoolingMode.SUM)
        assume(not mixed_B or pooling_mode != PoolingMode.NONE)

        # TODO: check split_optimizer_states when optimizer offloading is ready
        # Generate embedding modules and inputs
        (
            emb,
            emb_ref,
            Es,
            _,
            bucket_offsets,
            bucket_sizes,
        ) = self.generate_kvzch_tbes(
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
            num_buckets=num_buckets,
            enable_optimizer_offloading=enable_optimizer_offloading,
            backend_type=backend_type,
        )

        # Generate inputs
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
            bucket_offsets=bucket_offsets,
            bucket_sizes=bucket_sizes,
            is_kv_tbes=True,
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

        # Execute backward
        self.execute_ssd_backward_(
            output_ref_list,
            output,
            B,
            D,
            pooling_mode,
            batch_size_per_feature_per_rank,
        )

        emb.flush()

        tolerance = (
            1.0e-4
            if weights_precision == SparseType.FP32 and output_dtype == SparseType.FP32
            else 1.0e-2
        )

        # Compare emb state dict with expected values from nn.EmbeddingBag
        emb_state_dict_list, bucket_asc_ids_list, num_active_id_per_bucket_list, _ = (
            emb.split_embedding_weights(no_snapshot=False, should_flush=True)
        )
        split_optimizer_states = emb.split_optimizer_states(
            bucket_asc_ids_list, no_snapshot=False, should_flush=True
        )

        # create an empty emb with same parameters
        # Construct feature_table_map

        cache_sets = max(int(max(T * B * L, 1) * cache_set_scale), 1)
        emb2 = SSDTableBatchedEmbeddingBags(
            embedding_specs=emb.embedding_specs,
            feature_table_map=emb.feature_table_map,
            ssd_storage_directory=tempfile.mkdtemp(),
            cache_sets=cache_sets,
            ssd_uniform_init_lower=-0.1,
            ssd_uniform_init_upper=0.1,
            learning_rate=lr,
            eps=eps,
            ssd_rocksdb_shards=ssd_shards,
            optimizer=OptimType.EXACT_ROWWISE_ADAGRAD,
            pooling_mode=pooling_mode,
            weights_precision=weights_precision,
            output_dtype=output_dtype,
            stochastic_rounding=True,
            prefetch_pipeline=False,
            bounds_check_mode=BoundsCheckMode.WARNING,
            l2_cache_size=8,
            backend_type=backend_type,
            kv_zch_params=emb.kv_zch_params,
        ).cuda()

        # pyre-fixme[16]: Undefined attribute: `Optional` has no attribute `__iter__`.
        emb2.local_weight_counts = [ids.numel() for ids in bucket_asc_ids_list]
        emb2.enable_load_state_dict_mode()
        self.assertIsNotNone(emb2._cached_kvzch_data)
        for i, _ in enumerate(emb.embedding_specs):
            # pyre-ignore [16]
            emb2._cached_kvzch_data.cached_weight_tensor_per_table[i].copy_(
                # pyre-fixme[16]: Undefined attribute: Item `torch._tensor.Tensor` of `typing.Uni...
                emb_state_dict_list[i].full_tensor()
            )
            # NOTE: The [0] index is a hack since the test is fixed to use
            # EXACT_ROWWISE_ADAGRAD optimizer.  The test in general should
            # be upgraded in the future to support multiple optimizers
            # pyre-ignore [16]
            emb2._cached_kvzch_data.cached_optimizer_states_per_table[i][0].copy_(
                split_optimizer_states[i][0]
            )
            # pyre-ignore [16]
            emb2._cached_kvzch_data.cached_id_tensor_per_table[i].copy_(
                # pyre-fixme[16]: Undefined attribute: `Optional` has no attribute `__getitem__`.
                bucket_asc_ids_list[i]
            )
            # pyre-ignore [16]
            emb2._cached_kvzch_data.cached_bucket_splits[i].copy_(
                num_active_id_per_bucket_list[i]
            )

        emb2.apply_state_dict()

        emb2.flush(True)
        # Compare emb state dict with expected values from nn.EmbeddingBag
        (
            emb_state_dict_list2,
            bucket_asc_ids_list2,
            num_active_id_per_bucket_list2,
            _,
        ) = emb2.split_embedding_weights(no_snapshot=False, should_flush=True)
        split_optimizer_states2 = emb2.split_optimizer_states(
            bucket_asc_ids_list2, no_snapshot=False, should_flush=True
        )

        for t in range(len(emb.embedding_specs)):
            sorted_ids = torch.sort(bucket_asc_ids_list[t].flatten())
            sorted_ids2 = torch.sort(bucket_asc_ids_list2[t].flatten())
            torch.testing.assert_close(
                sorted_ids.values,
                sorted_ids2.values,
                atol=tolerance,
                rtol=tolerance,
            )

            torch.testing.assert_close(
                # pyre-ignore [16]
                emb_state_dict_list[t].full_tensor()[sorted_ids.indices],
                # pyre-ignore [16]
                emb_state_dict_list2[t].full_tensor()[sorted_ids2.indices],
                atol=tolerance,
                rtol=tolerance,
            )
            torch.testing.assert_close(
                split_optimizer_states[t][0][sorted_ids.indices],
                split_optimizer_states2[t][0][sorted_ids2.indices],
                atol=tolerance,
                rtol=tolerance,
            )
            torch.testing.assert_close(
                num_active_id_per_bucket_list[t],
                num_active_id_per_bucket_list2[t],
                atol=tolerance,
                rtol=tolerance,
            )

    def _check_raw_embedding_stream_call_counts(
        self,
        mock_raw_embedding_stream: unittest.mock.Mock,
        mock_raw_embedding_stream_sync: unittest.mock.Mock,
        num_iterations: int,
        prefetch_pipeline: bool,
        L: int,
    ) -> None:
        offset = 2 if prefetch_pipeline else 1
        self.assertEqual(
            mock_raw_embedding_stream.call_count,
            num_iterations * 2 - offset if L > 0 else num_iterations - offset,
        )
        self.assertEqual(
            mock_raw_embedding_stream_sync.call_count, num_iterations - offset
        )

    @staticmethod
    def _record_event_mock(
        stream: torch.cuda.Stream,
        pre_event: Optional[torch.cuda.Event],
        post_event: Optional[torch.cuda.Event],
        **kwargs_: Any,
    ) -> None:
        with torch.cuda.stream(stream):
            if pre_event is not None:
                stream.wait_event(pre_event)

            if post_event is not None:
                stream.record_event(post_event)

    @given(
        use_prefetch_stream=st.booleans(),
        prefetch_location=st.sampled_from(PrefetchLocation),
        **default_st,
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=MAX_PIPELINE_EXAMPLES,
        deadline=None,
    )
    def test_raw_embedding_streaming(
        self,
        **kwargs: Any,
    ):
        assume(not kwargs["weighted"] or kwargs["pooling_mode"] == PoolingMode.SUM)
        assume(not kwargs["mixed_B"] or kwargs["pooling_mode"] != PoolingMode.NONE)
        num_iterations = 10
        prefetch_pipeline = False
        with unittest.mock.patch.object(
            SSDTableBatchedEmbeddingBags,
            "raw_embedding_stream",
            side_effect=self._record_event_mock,
        ) as mock_raw_embedding_stream, unittest.mock.patch.object(
            SSDTableBatchedEmbeddingBags,
            "raw_embedding_stream_sync",
            side_effect=self._record_event_mock,
        ) as mock_raw_embedding_stream_sync:
            self.execute_ssd_cache_pipeline_(
                prefetch_pipeline=prefetch_pipeline,
                explicit_prefetch=prefetch_pipeline,
                enable_raw_embedding_streaming=True,
                flush_location=None,
                num_iterations=num_iterations,
                **kwargs,
            )
            self._check_raw_embedding_stream_call_counts(
                mock_raw_embedding_stream=mock_raw_embedding_stream,
                mock_raw_embedding_stream_sync=mock_raw_embedding_stream_sync,
                num_iterations=num_iterations,
                prefetch_pipeline=prefetch_pipeline,
                L=kwargs["L"],
            )

    @given(
        use_prefetch_stream=st.booleans(),
        prefetch_location=st.sampled_from(PrefetchLocation),
        **default_st,
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=MAX_PIPELINE_EXAMPLES,
        deadline=None,
    )
    def test_raw_embedding_streaming_prefetch_pipeline(
        self,
        **kwargs: Any,
    ):
        assume(not kwargs["weighted"] or kwargs["pooling_mode"] == PoolingMode.SUM)
        assume(not kwargs["mixed_B"] or kwargs["pooling_mode"] != PoolingMode.NONE)
        num_iterations = 10
        prefetch_pipeline = True
        with unittest.mock.patch.object(
            SSDTableBatchedEmbeddingBags,
            "raw_embedding_stream",
            side_effect=self._record_event_mock,
        ) as mock_raw_embedding_stream, unittest.mock.patch.object(
            SSDTableBatchedEmbeddingBags,
            "raw_embedding_stream_sync",
            side_effect=self._record_event_mock,
        ) as mock_raw_embedding_stream_sync:
            self.execute_ssd_cache_pipeline_(
                prefetch_pipeline=prefetch_pipeline,
                explicit_prefetch=prefetch_pipeline,
                enable_raw_embedding_streaming=True,
                flush_location=None,
                num_iterations=num_iterations,
                **kwargs,
            )
            self._check_raw_embedding_stream_call_counts(
                mock_raw_embedding_stream=mock_raw_embedding_stream,
                mock_raw_embedding_stream_sync=mock_raw_embedding_stream_sync,
                num_iterations=num_iterations,
                prefetch_pipeline=prefetch_pipeline,
                L=kwargs["L"],
            )

    @given(**default_st)
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_ssd_fetch_from_l1_sp_w_row_ids_weight(
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

        updated_weights = torch.zeros(
            indices.numel(),
            emb.max_D,
            device=emb.current_device,
            dtype=emb.weights_precision.as_dtype(),
        )
        linearized_indices = []
        for f, idxes in enumerate(indices_list):
            linearized_indices.append(idxes.flatten().add(emb.hash_size_cumsum[f]))
        linearized_indices_tensor = torch.cat(linearized_indices)

        def copy_weights_hook(
            _grad: torch.Tensor,
            updated_weights: torch.Tensor,
            emb: SSDTableBatchedEmbeddingBags,
            row_ids: torch.Tensor,
        ) -> None:
            if row_ids.numel() != 0:
                updates_list, _mask = emb.fetch_from_l1_sp_w_row_ids(row_ids=row_ids)
                # The function now returns a list of tensors, but for weights we expect only one tensor
                updated_weights.copy_(updates_list[0])

        emb.register_backward_hook_before_eviction(
            lambda grad: copy_weights_hook(
                grad,
                updated_weights,
                emb,
                linearized_indices_tensor,
            )
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

        # Execute backward
        self.execute_ssd_backward_(
            output_ref_list,
            output,
            B,
            D,
            pooling_mode,
            batch_size_per_feature_per_rank,
        )

        tolerance = (
            1.0e-4
            if weights_precision == SparseType.FP32 and output_dtype == SparseType.FP32
            else 1.0e-2
        )

        # Compare optimizer states
        split_optimizer_states = emb.split_optimizer_states()
        for f, t in self.get_physical_table_arg_indices_(emb.feature_table_map):
            # pyre-fixme[16]: Optional type has no attribute `float`.
            ref_optimizer_state = emb_ref[f].weight.grad.float().to_dense().pow(2)
            torch.testing.assert_close(
                # NOTE: The [0] index is a hack since the test is fixed to use
                # EXACT_ROWWISE_ADAGRAD optimizer.  The test in general should
                # be upgraded in the future to support multiple optimizers
                split_optimizer_states[t][0].float(),
                ref_optimizer_state.mean(dim=1),
                atol=tolerance,
                rtol=tolerance,
            )

        # Compare weights
        emb.flush()

        cursor = 0
        emb_test = emb.debug_split_embedding_weights()
        for f, t in enumerate(emb.feature_table_map):
            row_idxes = indices_list[f]
            local_idxes = row_idxes.flatten()
            weights_per_tb = updated_weights[cursor : cursor + local_idxes.numel()]
            cursor += local_idxes.numel()

            if weights_precision == SparseType.FP16:
                # Round the reference weight the same way that TBE does
                weights_per_tb = weights_per_tb.half().float()

            # check only the updated rows
            torch.testing.assert_close(
                emb_test[t][local_idxes.cpu()].float().cuda(),
                weights_per_tb.float().cuda(),
                atol=tolerance,
                rtol=tolerance,
            )

    @given(**default_st)
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_ssd_fetch_from_l1_sp_w_row_ids_opt_only(
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

        # Constants
        lr = 0.5
        eps = 0.2
        ssd_shards = 2

        trigger_bounds_check = False  # don't stimulate boundary check cases
        assume(not weighted or pooling_mode == PoolingMode.SUM)
        assume(not mixed_B or pooling_mode != PoolingMode.NONE)

        # Generate embedding modules and inputs
        (
            emb,
            emb_ref,
            Es,
            _,
            bucket_offsets,
            bucket_sizes,
        ) = self.generate_kvzch_tbes(
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
            enable_optimizer_offloading=True,
        )

        # Generate inputs
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
            bucket_offsets=bucket_offsets,
            bucket_sizes=bucket_sizes,
            is_kv_tbes=True,
        )

        updated_opt_states = torch.zeros(
            indices.numel(),
            1,
            device=emb.current_device,
            # NOTE: This is a hack to keep fetch_from_l1_sp_w_row_ids unit test
            # working until it is upgraded to support optimizers with multiple
            # states and dtypes
            dtype=torch.float32,
        )
        linearized_indices = []
        for f, idxes in enumerate(indices_list):
            linearized_indices.append(idxes.flatten().add(emb.hash_size_cumsum[f]))
        linearized_indices_tensor = torch.cat(linearized_indices)

        def copy_opt_states_hook(
            _grad: torch.Tensor,
            updated_opt_states: torch.Tensor,
            emb: SSDTableBatchedEmbeddingBags,
            row_ids: torch.Tensor,
        ) -> None:
            if row_ids.numel() != 0:
                updates_list, _mask = emb.fetch_from_l1_sp_w_row_ids(
                    row_ids=row_ids, only_get_optimizer_states=True
                )
                # The function now returns a list of tensors
                # for EXACT_ROWWISE_ADAGRAD optimizer states we take the first one
                updated_opt_states.copy_(updates_list[0])

        emb.register_backward_hook_before_eviction(
            lambda grad: copy_opt_states_hook(
                grad,
                updated_opt_states,
                emb,
                linearized_indices_tensor,
            )
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

        # Execute backward
        self.execute_ssd_backward_(
            output_ref_list,
            output,
            B,
            D,
            pooling_mode,
            batch_size_per_feature_per_rank,
        )

        emb.flush()

        # Compare emb state dict with expected values from nn.EmbeddingBag
        _emb_state_dict_list, bucket_asc_ids_list, _num_active_id_per_bucket_list, _ = (
            emb.split_embedding_weights(no_snapshot=False, should_flush=True)
        )
        assert bucket_asc_ids_list is not None
        split_optimizer_states = emb.split_optimizer_states(
            bucket_asc_ids_list, no_snapshot=False
        )
        table_input_id_range = []
        for t, row in enumerate(Es):
            bucket_id_start = bucket_offsets[t][0]
            bucket_id_end = bucket_offsets[t][1]
            bucket_size = bucket_sizes[t]
            table_input_id_range.append(
                (
                    min(bucket_id_start * bucket_size, row),
                    min(bucket_id_end * bucket_size, row),
                )
            )
            # since we use ref_emb in dense format, the rows start from id 0
            self.assertEqual(table_input_id_range[-1][0], 0)

        cursor = 0
        tolerance = 1.0e-4
        # Compare optimizer states
        for f, t in enumerate(emb.feature_table_map):
            row_idxes = indices_list[f]
            local_idxes = row_idxes.flatten()
            value_to_index = {
                v.item(): i for i, v in enumerate(bucket_asc_ids_list[t].flatten())
            }
            indices = torch.tensor([value_to_index[v.item()] for v in local_idxes])
            opt_states_per_tb = updated_opt_states[
                cursor : cursor + local_idxes.numel()
            ].flatten()
            if local_idxes.numel() == 0:
                continue
            cursor += local_idxes.numel()

            torch.testing.assert_close(
                # NOTE: The [0] index is a hack since the test is fixed to use
                # EXACT_ROWWISE_ADAGRAD optimizer.  The test in general should
                # be upgraded in the future to support multiple optimizers
                split_optimizer_states[t][0][indices].float(),
                opt_states_per_tb.cpu().float(),
                atol=tolerance,
                rtol=tolerance,
            )

    @given(**default_st)
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_ssd_fetch_from_l1_sp_w_row_ids_partial_rowwise_adam(
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

        # Constants
        lr = 0.5
        eps = 0.2
        ssd_shards = 2
        beta1 = 0.9
        beta2 = 0.999
        weight_decay = 0.01

        trigger_bounds_check = False  # don't stimulate boundary check cases
        assume(not mixed_B)
        assume(not weighted or pooling_mode == PoolingMode.SUM)

        # Generate embedding modules and inputs
        (
            emb,
            emb_ref,
            Es,
            _,
            bucket_offsets,
            bucket_sizes,
        ) = self.generate_kvzch_tbes(
            T,
            D,
            B,
            log_E,
            L,
            weighted,
            lr=lr,
            eps=eps,
            weight_decay=weight_decay,
            beta1=beta1,
            beta2=beta2,
            ssd_shards=ssd_shards,
            optimizer=OptimType.PARTIAL_ROWWISE_ADAM,
            cache_set_scale=cache_set_scale,
            pooling_mode=pooling_mode,
            weights_precision=weights_precision,
            output_dtype=output_dtype,
            share_table=share_table,
            enable_optimizer_offloading=True,
        )

        # Generate inputs
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
            mixed_B=False,
            bucket_offsets=bucket_offsets,
            bucket_sizes=bucket_sizes,
            is_kv_tbes=True,
        )

        # For PARTIAL_ROWWISE_ADAM, we need to handle two optimizer states (m1 and m2)
        updated_m1 = torch.zeros(
            indices.numel(),
            emb.max_D,
            device=emb.current_device,
            dtype=torch.float32,
        )

        updated_m2 = torch.zeros(
            indices.numel(),
            1,
            device=emb.current_device,
            dtype=torch.float32,
        )

        linearized_indices = []
        for f, idxes in enumerate(indices_list):
            linearized_indices.append(idxes.flatten().add(emb.hash_size_cumsum[f]))
        linearized_indices_tensor = torch.cat(linearized_indices)

        def copy_opt_states_hook(
            _grad: torch.Tensor,
            updated_m1: torch.Tensor,
            updated_m2: torch.Tensor,
            emb: SSDTableBatchedEmbeddingBags,
            row_ids: torch.Tensor,
        ) -> None:
            if row_ids.numel() != 0:
                updates_list, _mask = emb.fetch_from_l1_sp_w_row_ids(
                    row_ids=row_ids, only_get_optimizer_states=True
                )
                # The function now returns a list of tensors for optimizer states
                # For PARTIAL_ROWWISE_ADAM, we expect two tensors: m1 and m2
                updated_m1.copy_(updates_list[1])
                updated_m2.copy_(updates_list[0])

        emb.register_backward_hook_before_eviction(
            lambda grad: copy_opt_states_hook(
                grad,
                updated_m1,
                updated_m2,
                emb,
                linearized_indices_tensor,
            )
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

        # Execute backward
        self.execute_ssd_backward_(
            output_ref_list,
            output,
            B,
            D,
            pooling_mode,
            batch_size_per_feature_per_rank,
        )

        emb.flush()

        # Compare emb state dict with expected values from nn.EmbeddingBag
        _emb_state_dict_list, bucket_asc_ids_list, _num_active_id_per_bucket_list, _ = (
            emb.split_embedding_weights(no_snapshot=False, should_flush=True)
        )
        assert bucket_asc_ids_list is not None
        split_optimizer_states = emb.split_optimizer_states(
            bucket_asc_ids_list, no_snapshot=False
        )
        table_input_id_range = []
        for t, row in enumerate(Es):
            bucket_id_start = bucket_offsets[t][0]
            bucket_id_end = bucket_offsets[t][1]
            bucket_size = bucket_sizes[t]
            table_input_id_range.append(
                (
                    min(bucket_id_start * bucket_size, row),
                    min(bucket_id_end * bucket_size, row),
                )
            )
            # since we use ref_emb in dense format, the rows start from id 0
            self.assertEqual(table_input_id_range[-1][0], 0)

        cursor = 0
        tolerance = 1.0e-4
        # Compare optimizer states
        for f, t in enumerate(emb.feature_table_map):
            row_idxes = indices_list[f]
            local_idxes = row_idxes.flatten()
            value_to_index = {
                v.item(): i for i, v in enumerate(bucket_asc_ids_list[t].flatten())
            }
            indices = torch.tensor([value_to_index[v.item()] for v in local_idxes])
            m1_states_per_tb = updated_m1[cursor : cursor + local_idxes.numel()]
            m2_states_per_tb = updated_m2[cursor : cursor + local_idxes.numel()]

            if local_idxes.numel() == 0:
                continue
            cursor += local_idxes.numel()

            # For PARTIAL_ROWWISE_ADAM, we have two optimizer states: m1 and m2
            torch.testing.assert_close(
                split_optimizer_states[t][0][indices].float(),
                m1_states_per_tb.cpu().float(),
                atol=tolerance,
                rtol=tolerance,
            )

            torch.testing.assert_close(
                split_optimizer_states[t][1][indices].float(),
                m2_states_per_tb.cpu().flatten().float(),
                atol=tolerance,
                rtol=tolerance,
            )
