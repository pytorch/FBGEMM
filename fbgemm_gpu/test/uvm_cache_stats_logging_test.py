# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest
from math import floor
from unittest.mock import MagicMock, patch

import torch

from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType, SparseType
from fbgemm_gpu.split_embedding_utils import to_device
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    CacheAlgorithm,
    EmbeddingLocation,
    PoolingMode,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    ComputeDevice,
    SplitTableBatchedEmbeddingBagsCodegen,
)

from rfe.scubadata.scubadata_py3 import Sample


class UvmCacheStatsLoggingTest(unittest.TestCase):
    # Used for UVMCacheStatsManager checking env for job id and rank
    @patch.dict(
        os.environ,
        {"MAST_HPC_JOB_NAME": "test-mast-job", "RANK": "0"},
    )
    def setUp(self) -> None:
        self.tbe = self.construct_simple_tbe()
        # Make uvm cache stats manager magic mock
        self.tbe.uvm_cache_stats_manager.scuba_data = MagicMock()
        self.tbe.gather_uvm_cache_stats_interval = 2
        # For purpose of test, gather stats 2x more than logging interval
        self.tbe.log_uvm_cache_stats_interval = 4

    def test_simulate_tbe_forwards(self) -> None:
        # Only simulate the part of fwd that updates uvm cache stats
        # After 1.5x the logging interval, should only have logged once
        iters = 6
        test_requests = [2,5,3,7,3,9]
        test_unique_indices = [1,4,2,6,2,5]
        test_unique_misses = [1,2,1,4,1,3]
        test_conflict_unique_misses = [1,2,1,4,2,6]
        test_conflict_misses = [1,3,1,5,3,7]
        for i in range(iters):
            self.tbe.step += 1  # Manually increment step
            # Ignore N called bc we don't log it
            self.tbe.local_uvm_cache_stats[1] += test_requests[i]
            self.tbe.local_uvm_cache_stats[2] += test_unique_indices[i]
            self.tbe.local_uvm_cache_stats[3] += test_unique_misses[i]
            self.tbe.local_uvm_cache_stats[4] += test_conflict_unique_misses[i]
            self.tbe.local_uvm_cache_stats[5] += test_conflict_misses[i]
            self.tbe.update_uvm_cache_stats()

        test_sample = self.make_test_sample()
        self.tbe.uvm_cache_stats_manager.scuba_data.addSample.assert_called_once_with(
            test_sample
        )

    def make_test_sample(self) -> Sample:
        # This is the sample we expect to be logged given the simulated forward
        # and the logic in UVMCacheStatsManager
        sample = Sample()
        sample.addNormalValue("job_id", "test-mast-job")
        sample.addNormalValue("rank", "0")
        sample.addDoubleValue("requests", 5.0)
        sample.addDoubleValue("unique_indices", 4.0)
        sample.addDoubleValue("unique_misses", 2.5)
        sample.addDoubleValue("conflict_unique_misses", 2.5)
        sample.addDoubleValue("conflict_misses", 3.0)
        sample.addDoubleValue("unique_miss_rate", 0.5)
        sample.addDoubleValue("conflict_unique_miss_rate", 0.5)
        return sample

    def test_jit_compatible(self) -> None:
        tbe = self.construct_simple_tbe()
        tbe = torch.jit.script(tbe)  # Will fail if it's not jit compatible

    def construct_simple_tbe(self) -> SplitTableBatchedEmbeddingBagsCodegen:
        emb_op = SplitTableBatchedEmbeddingBagsCodegen
        T = 5
        D = 128
        E = 10**4
        Ds = [D] * T
        Es = [E] * T
        bs = [
            to_device(
                torch.nn.EmbeddingBag(E, D, mode="sum", sparse=True), use_cpu=False
            )
            for (E, D) in zip(Es, Ds)
        ]
        compute_device = ComputeDevice.CUDA
        managed = [EmbeddingLocation.DEVICE] * T
        tbe = emb_op(
            embedding_specs=[
                (
                    E,
                    D,
                    EmbeddingLocation(M),
                    compute_device,
                )
                for (E, D, M) in zip(Es, Ds, managed)
            ],
            weights_precision=SparseType.FP32,
            optimizer=OptimType.EXACT_SGD,
            learning_rate=0.05,
            cache_algorithm=CacheAlgorithm.LRU,
            pooling_mode=PoolingMode.SUM,
            output_dtype=SparseType.FP32,
            use_experimental_tbe=False,
            gather_uvm_cache_stats=True,
        )
        for t in range(T):
            tbe.split_embedding_weights()[t].data.copy_(bs[t].weight)
        return tbe
