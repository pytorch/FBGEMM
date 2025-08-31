# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[3,6,56]

import math
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
from torch import distributed as dist

from .. import common  # noqa E402
from ..common import gen_mixed_B_batch_sizes


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

default_strategies: Dict["str", Any] = {
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


class SSDSplitTableBatchedEmbeddingsTestCommon(unittest.TestCase):
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
        optimizer_state_dtypes: Dict[OptimType, SparseType] = {},  # noqa: B006
        embedding_cache_mode: bool = False,
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
            embedding_cache_mode=embedding_cache_mode,
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
