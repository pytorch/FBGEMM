# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[3,6,56]

import logging
import math
import tempfile

import threading
import time
import unittest

from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import patch

import hypothesis.strategies as st
import numpy as np
import torch
from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.tbe.ssd import SSDTableBatchedEmbeddingBags
from fbgemm_gpu.tbe.ssd.training import KVZCHParams
from fbgemm_gpu.tbe.utils import round_up
from hypothesis import given, settings, Verbosity

from .. import common  # noqa E402
from ..common import gpu_unavailable, running_in_oss

MAX_EXAMPLES = 100
WORLD_SIZE = 4
default_st: Dict[str, Any] = {
    "T": st.integers(min_value=1, max_value=10),
    "D": st.integers(min_value=2, max_value=128),
    "log_E": st.integers(min_value=2, max_value=3),
    "mixed": st.booleans(),
    "weights_precision": st.sampled_from([SparseType.FP32, SparseType.FP16]),
}

default_settings: Dict[str, Any] = {
    "verbosity": Verbosity.verbose,
    "max_examples": MAX_EXAMPLES,
    "deadline": None,
}


@unittest.skipIf(*running_in_oss)
@unittest.skipIf(*gpu_unavailable)
class SSDCheckpointTest(unittest.TestCase):
    def generate_fbgemm_ssd_tbe(
        self,
        T: int,
        D: int,
        log_E: int,
        weights_precision: SparseType,
        mixed: bool,
        enable_l2: bool = True,
        ssd_rocksdb_shards: int = 1,
        kv_zch_params: Optional[KVZCHParams] = None,
    ) -> Tuple[SSDTableBatchedEmbeddingBags, List[int], List[int]]:
        E = int(10**log_E)
        D = D * 4
        if not mixed:
            Ds = [D] * T
            Es = [E] * T
        else:
            Ds = [
                round_up(np.random.randint(low=int(0.25 * D), high=int(1.0 * D)), 4)
                for _ in range(T)
            ]
            Es = [
                np.random.randint(low=int(0.5 * E), high=int(2.0 * E)) for _ in range(T)
            ]

        feature_table_map = list(range(T))
        emb = SSDTableBatchedEmbeddingBags(
            embedding_specs=[(E, D) for (E, D) in zip(Es, Ds)],
            feature_table_map=feature_table_map,
            ssd_storage_directory=tempfile.mkdtemp(),
            cache_sets=1,
            ssd_uniform_init_lower=-0.1,
            ssd_uniform_init_upper=0.1,
            weights_precision=weights_precision,
            l2_cache_size=1 if enable_l2 else 0,
            ssd_rocksdb_shards=ssd_rocksdb_shards,
            kv_zch_params=kv_zch_params,
        )
        return emb, Es, Ds

    @given(**default_st, do_flush=st.sampled_from([True, False]))
    @settings(**default_settings)
    def test_l2_flush(
        self,
        T: int,
        D: int,
        log_E: int,
        mixed: bool,
        weights_precision: SparseType,
        do_flush: bool,
    ) -> None:
        emb, Es, _ = self.generate_fbgemm_ssd_tbe(T, D, log_E, weights_precision, mixed)
        indices = torch.arange(start=0, end=sum(Es))
        weights = torch.randn(
            indices.numel(), emb.cache_row_dim, dtype=weights_precision.as_dtype()
        )
        weights_from_l2 = torch.empty_like(weights)
        count = torch.as_tensor([indices.numel()])
        emb.ssd_db.set_cuda(indices, weights, count, 1)
        emb.ssd_db.get_cuda(indices.clone(), weights_from_l2, count)

        torch.cuda.synchronize()
        assert torch.equal(weights, weights_from_l2)

        weights_from_ssd = torch.empty_like(weights)
        if do_flush:
            emb.ssd_db.flush()
        emb.ssd_db.reset_l2_cache()
        emb.ssd_db.get_cuda(indices, weights_from_ssd, count)
        torch.cuda.synchronize()
        if do_flush:
            assert torch.equal(weights, weights_from_ssd)
        else:
            assert not torch.equal(weights, weights_from_ssd)

    @given(**default_st, enable_l2=st.sampled_from([True, False]))
    @settings(**default_settings)
    def test_l2_io(
        self,
        T: int,
        D: int,
        log_E: int,
        mixed: bool,
        weights_precision: SparseType,
        enable_l2: bool,
    ) -> None:
        emb, _, _ = self.generate_fbgemm_ssd_tbe(
            T, D, log_E, weights_precision, mixed, enable_l2
        )
        E = int(10**log_E)
        num_rounds = 10
        N = E
        total_indices = torch.tensor([])

        indices = torch.as_tensor(
            np.random.choice(E, replace=False, size=(N,)), dtype=torch.int64
        )
        weights = torch.randn(
            indices.numel(), emb.cache_row_dim, dtype=weights_precision.as_dtype()
        )
        sub_N = N // num_rounds

        for _ in range(num_rounds):
            sub_indices = torch.as_tensor(
                np.random.choice(E, replace=False, size=(sub_N,)), dtype=torch.int64
            )
            sub_weights = weights[sub_indices, :]
            sub_weights_out = torch.empty_like(sub_weights)
            count = torch.as_tensor([sub_indices.numel()])
            emb.ssd_db.set_cuda(sub_indices, sub_weights, count, 1)
            emb.ssd_db.get_cuda(sub_indices.clone(), sub_weights_out, count)
            torch.cuda.synchronize()
            assert torch.equal(sub_weights, sub_weights_out)
            total_indices = torch.cat((total_indices, sub_indices))
        # dedup
        used_unique_indices = torch.tensor(
            list(set(total_indices.tolist())), dtype=torch.int64
        )
        stored_weights = weights[used_unique_indices, :]
        weights_out = torch.empty_like(stored_weights)
        count = torch.as_tensor([used_unique_indices.numel()])
        emb.ssd_db.get_cuda(used_unique_indices.clone(), weights_out, count)
        torch.cuda.synchronize()
        assert torch.equal(stored_weights, weights_out)

        emb.ssd_db.flush()
        emb.ssd_db.reset_l2_cache()
        weights_out = torch.empty_like(stored_weights)
        count = torch.as_tensor([used_unique_indices.numel()])
        emb.ssd_db.get_cuda(used_unique_indices.clone(), weights_out, count)
        torch.cuda.synchronize()
        assert torch.equal(stored_weights, weights_out)

    @given(**default_st)
    @settings(**default_settings)
    def test_l2_prefetch_compatibility(
        self,
        T: int,
        D: int,
        log_E: int,
        mixed: bool,
        weights_precision: SparseType,
    ) -> None:
        emb, _, _ = self.generate_fbgemm_ssd_tbe(T, D, log_E, weights_precision, mixed)
        E = int(10**log_E)
        N = E
        indices = torch.as_tensor(
            np.random.choice(E, replace=False, size=(N,)), dtype=torch.int64
        )
        weights = torch.randn(N, emb.cache_row_dim, dtype=weights_precision.as_dtype())
        new_weights = weights + 1
        weights_out = torch.empty_like(weights)
        count = torch.as_tensor([E])
        emb.ssd_db.set(indices, weights, count)
        emb.ssd_db.wait_util_filling_work_done()

        event = threading.Event()
        get_sleep_ms = 50

        # pyre-ignore
        def trigger_get() -> None:
            event.set()
            emb.ssd_db.get(indices.clone(), weights_out, count, get_sleep_ms)

        # pyre-ignore
        def trigger_set() -> None:
            event.wait()
            time.sleep(
                get_sleep_ms / 1000.0 / 2
            )  # sleep half of the sleep time in get, making sure set is trigger after get but before get is done
            emb.ssd_db.set(indices, new_weights, count)

        thread1 = threading.Thread(target=trigger_get)
        thread2 = threading.Thread(target=trigger_set)
        thread1.start()
        thread2.start()
        thread1.join()
        thread2.join()
        assert torch.equal(weights, weights_out)
        emb.ssd_db.get(indices.clone(), weights_out, count)
        assert torch.equal(new_weights, weights_out)

    @given(**default_st)
    @settings(**default_settings)
    def test_l2_multiple_flush_at_same_train_iter(
        self,
        T: int,
        D: int,
        log_E: int,
        mixed: bool,
        weights_precision: SparseType,
    ) -> None:
        emb, _, _ = self.generate_fbgemm_ssd_tbe(T, D, log_E, weights_precision, mixed)

        with patch.object(torch.cuda, "synchronize") as mock_calls:
            mock_calls.side_effect = None
            emb.flush()
            emb.flush()
            mock_calls.assert_called_once()
            mock_calls.reset_mock()

            emb.step += 1
            emb.flush()
            emb.step += 1
            emb.flush()
            self.assertEqual(mock_calls.call_count, 2)

    @given(**default_st)
    @settings(**default_settings)
    def test_rocksdb_get_discrete_ids(
        self,
        T: int,
        D: int,
        log_E: int,
        mixed: bool,
        weights_precision: SparseType,
    ) -> None:
        emb, Es, _ = self.generate_fbgemm_ssd_tbe(
            T, D, log_E, weights_precision, mixed, False, 8
        )
        E = int(10**log_E)
        N = int(E / 10)

        offset = 1000
        indices = torch.as_tensor(
            np.random.choice(E, replace=False, size=(N,)), dtype=torch.int64
        )
        new_indices = torch.as_tensor(
            np.random.choice(E, replace=False, size=(N,)), dtype=torch.int64
        )
        weights = torch.arange(N * D, dtype=weights_precision.as_dtype()).view(N, D)
        new_weights_after_snapshot = torch.randn(
            N, D, dtype=weights_precision.as_dtype()
        )

        # use kvtensor to directly set KVs into rocksdb wrapper
        tensor_wrapper = torch.classes.fbgemm.KVTensorWrapper([E, D], weights.dtype, 0)
        tensor_wrapper.set_embedding_rocks_dp_wrapper(emb.ssd_db)
        tensor_wrapper.set_weights_and_ids(weights, indices + offset)

        snapshot = emb.ssd_db.create_snapshot()
        tensor_wrapper.set_weights_and_ids(
            new_weights_after_snapshot, new_indices + offset
        )
        start_id = 0
        end_id = int(E / 2)

        mask = (indices >= start_id) & (indices < end_id)
        ids_in_range = indices[mask]
        id_tensor = emb.ssd_db.get_keys_in_range_by_snapshot(
            start_id + offset, end_id + offset, offset, snapshot
        )
        ids_in_range_ordered, _ = torch.sort(ids_in_range)
        id_tensor_ordered, _ = torch.sort(id_tensor.view(-1))

        assert torch.equal(ids_in_range_ordered, id_tensor_ordered)

    @given(
        E=st.integers(min_value=1000, max_value=10000),
        num_buckets=st.integers(min_value=10, max_value=15),
        my_rank=st.integers(min_value=0, max_value=WORLD_SIZE),
        hash_mode=st.sampled_from([0, 1, 2]),
    )
    @settings(**default_settings)
    def test_get_bucket_sorted_indices(
        self,
        E: int,
        num_buckets: int,
        my_rank: int,
        hash_mode: int,
    ) -> None:
        bucket_size = math.ceil(E / num_buckets)
        bucket_start = min(math.ceil(num_buckets / WORLD_SIZE) * my_rank, num_buckets)
        bucket_end = min(
            math.ceil(num_buckets / WORLD_SIZE) * (my_rank + 1), num_buckets
        )
        rank_range = (bucket_end - bucket_start) * bucket_size

        indices = torch.empty(0, dtype=torch.int64)

        # construct indices, the indices's bucket ids have to fall into the range of bucket start and end
        if hash_mode == 0:
            if rank_range == 0:
                indices = torch.empty(0, dtype=torch.int64)
            else:
                indices = torch.as_tensor(
                    np.random.choice(rank_range, replace=False, size=(rank_range,)),
                    dtype=torch.int64,
                )
                indices += bucket_start * bucket_size
        elif hash_mode == 1:
            for i in range(bucket_start, bucket_end):
                sub_indices = torch.arange(start=0, end=bucket_size, dtype=torch.int64)
                sub_indices *= num_buckets
                sub_indices += i
                indices = torch.cat((indices, sub_indices), dim=0)

        # test get_bucket_sorted_indices_and_bucket_tensor
        if hash_mode == 0:
            # meaning it is mod based hashing
            [bucket_sorted_ids, bucket_t] = (
                torch.ops.fbgemm.get_bucket_sorted_indices_and_bucket_tensor(
                    indices,
                    hash_mode,
                    bucket_start,
                    bucket_end,
                    bucket_size,
                )
            )
        elif hash_mode == 1:
            # meaning it is interleaved based hashing
            [bucket_sorted_ids, bucket_t] = (
                torch.ops.fbgemm.get_bucket_sorted_indices_and_bucket_tensor(
                    indices,
                    hash_mode,
                    bucket_start,
                    bucket_end,
                    None,
                    num_buckets,
                )
            )
        else:
            # test failure
            with self.assertRaisesRegex(
                RuntimeError,
                "only support hash by chunk-based or interleaved-based hashing for now",
            ):
                torch.ops.fbgemm.get_bucket_sorted_indices_and_bucket_tensor(
                    indices,
                    hash_mode,
                    bucket_start,
                    bucket_end,
                    bucket_size,
                )
            return
        last_bucket_id = 0
        for i in range(bucket_sorted_ids.numel()):
            self.assertTrue(hash_mode >= 0 and hash_mode <= 1)
            cur_bucket_id = -1
            if hash_mode == 0:
                cur_bucket_id = bucket_sorted_ids[i] // bucket_size
            elif hash_mode == 1:
                cur_bucket_id = bucket_sorted_ids[i] % num_buckets

            self.assertGreaterEqual(cur_bucket_id, last_bucket_id)
            last_bucket_id = cur_bucket_id
        # Calculate expected tensor output
        expected_bucket_tensor = torch.zeros(
            bucket_end - bucket_start, dtype=torch.int64
        )
        for index in indices:
            self.assertTrue(hash_mode >= 0 and hash_mode <= 1)
            bucket_id = -1
            if hash_mode == 0:
                bucket_id = index // bucket_size
            elif hash_mode == 1:
                bucket_id = index % num_buckets
            expected_bucket_tensor[bucket_id - bucket_start] += 1

        # Compare actual and expected tensor outputs
        self.assertTrue(torch.equal(bucket_t.view(-1), expected_bucket_tensor))

    @given(
        T=st.integers(min_value=2, max_value=10),
        D=st.integers(min_value=2, max_value=128),
        log_E=st.integers(min_value=2, max_value=3),
        weights_precision=st.sampled_from([SparseType.FP32, SparseType.FP16]),
        enable_l2=st.sampled_from([True, False]),
    )
    @settings(**default_settings)
    def test_all_zero_opt_state_offloading(
        self,
        T: int,
        D: int,
        log_E: int,
        weights_precision: SparseType,
        enable_l2: bool,
    ) -> None:
        kv_zch_params = KVZCHParams(
            enable_optimizer_offloading=True,
        )
        emb, Es, Ds = self.generate_fbgemm_ssd_tbe(
            T,
            D,
            log_E,
            weights_precision,
            mixed=True,
            enable_l2=enable_l2,
            kv_zch_params=kv_zch_params,
        )
        dtype = weights_precision.as_dtype()
        opt_dim = int(math.ceil(4 / dtype.itemsize))
        snapshot = emb.ssd_db.create_snapshot()
        offsets = 0
        max_D = max(Ds)
        for E, D in zip(Es, Ds):
            if D + opt_dim > max_D:
                # skip this case, we need enough space to simulate optimizer state query
                offsets += E
                continue

            tensor_wrapper = torch.classes.fbgemm.KVTensorWrapper(
                [E, max_D], dtype, offsets, snapshot
            )
            tensor_wrapper.set_embedding_rocks_dp_wrapper(emb.ssd_db)
            weight_opt = tensor_wrapper.narrow(0, 0, 1)
            pad4_d = (D + 3) & ~3
            torch.testing.assert_close(
                weight_opt[:, pad4_d:], torch.zeros(1, max_D - pad4_d, dtype=dtype)
            )
            offsets += E

    @given(
        T=st.integers(min_value=2, max_value=10),
        D=st.integers(min_value=2, max_value=128),
        log_E=st.integers(min_value=2, max_value=3),
        weights_precision=st.sampled_from([SparseType.FP32, SparseType.FP16]),
        enable_l2=st.sampled_from([True, False]),
    )
    @settings(**default_settings)
    def test_contiguous_narrow_w_offloading(
        self,
        T: int,
        D: int,
        log_E: int,
        weights_precision: SparseType,
        enable_l2: bool,
    ) -> None:
        kv_zch_params = KVZCHParams(
            enable_optimizer_offloading=True,
        )
        emb, Es, Ds = self.generate_fbgemm_ssd_tbe(
            T,
            D,
            log_E,
            weights_precision,
            mixed=True,
            enable_l2=enable_l2,
            kv_zch_params=kv_zch_params,
        )
        dtype = weights_precision.as_dtype()
        opt_dim = int(math.ceil(4 / dtype.itemsize))
        snapshot = emb.ssd_db.create_snapshot()
        offsets = 0
        max_D = max(Ds)
        for E, D in zip(Es, Ds):
            if D + opt_dim > max_D:
                # skip this case, we need enough space to simulate optimizer state query
                offsets += E
                continue

            tensor_wrapper = torch.classes.fbgemm.KVTensorWrapper(
                [E, D], dtype, offsets, snapshot
            )
            tensor_wrapper.set_embedding_rocks_dp_wrapper(emb.ssd_db)
            weight_opt = tensor_wrapper.narrow(0, 0, 1)
            self.assertTrue(weight_opt.is_contiguous())
            offsets += E
