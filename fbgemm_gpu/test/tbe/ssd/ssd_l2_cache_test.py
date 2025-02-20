# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[3,6,56]

import tempfile

import threading
import time
import unittest

from typing import Any, Dict, List, Tuple

import hypothesis.strategies as st
import numpy as np
import torch
from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.tbe.ssd import SSDTableBatchedEmbeddingBags
from fbgemm_gpu.tbe.utils import round_up
from hypothesis import given, settings, Verbosity

from .. import common  # noqa E402
from ..common import gpu_unavailable, running_in_oss

MAX_EXAMPLES = 20
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
    ) -> Tuple[SSDTableBatchedEmbeddingBags, List[int], List[int], int]:
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
        )
        return emb, Es, Ds, max(Ds)

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
        emb, Es, Ds, max_D = self.generate_fbgemm_ssd_tbe(
            T, D, log_E, weights_precision, mixed
        )
        indices = torch.arange(start=0, end=sum(Es))
        weights = torch.randn(
            indices.numel(), max_D, dtype=weights_precision.as_dtype()
        )
        weights_from_l2 = torch.empty_like(weights)
        count = torch.as_tensor([indices.numel()])
        emb.ssd_db.set_cuda(indices, weights, count, 1)
        emb.ssd_db.get_cuda(indices.clone(), weights_from_l2, count)

        torch.cuda.synchronize()
        assert torch.equal(weights, weights_from_l2)
        import logging

        logging.info(f"wgqtest {do_flush=}")
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
        emb, Es, Ds, max_D = self.generate_fbgemm_ssd_tbe(
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
            indices.numel(), max_D, dtype=weights_precision.as_dtype()
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
        weights_precision: SparseType = SparseType.FP32
        emb, Es, Ds, max_D = self.generate_fbgemm_ssd_tbe(
            T, D, log_E, weights_precision, mixed
        )
        E = int(10**log_E)
        N = E
        indices = torch.as_tensor(
            np.random.choice(E, replace=False, size=(N,)), dtype=torch.int64
        )
        weights = torch.randn(N, max_D, dtype=weights_precision.as_dtype())
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
