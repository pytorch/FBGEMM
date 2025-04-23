# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[3,6,56]

import logging
import unittest

from typing import Any, Dict, List, Tuple

import hypothesis.strategies as st
import numpy as np
import torch
from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.tbe.ssd import SSDTableBatchedEmbeddingBags
from fbgemm_gpu.tbe.utils import round_up
from fbgemm_gpu.utils.loader import load_torch_module
from hypothesis import Verbosity

from ..common import gpu_unavailable, open_source, running_in_oss

if not open_source:
    load_torch_module(
        "//deeplearning/fbgemm/fbgemm_gpu:ssd_split_table_batched_embeddings",
    )

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
class DRAMKVTest(unittest.TestCase):
    def generate_fbgemm_kv_tbe(
        self,
        T: int,
        D: int,
        log_E: int,
        weights_precision: SparseType,
        mixed: bool,
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
        max_D = max(Ds)
        dram_kv = torch.classes.fbgemm.DramKVEmbeddingCacheWrapper(
            max_D=max_D,
            uniform_init_lower=-0.1,
            uniform_init_upper=0.1,
            num_shards=8,
            num_threads=32,
            row_storage_bitwidth=weights_precision.bit_rate(),
        )
        return dram_kv, Es, Ds, max_D

    def test_basics(
        self,
    ) -> None:

        T = 3
        D = 1
        log_E = 1
        mixed = True
        weights_precision: SparseType = SparseType.FP32

        dram_kv, Es, Ds, max_D = self.generate_fbgemm_kv_tbe(
            T, D, log_E, weights_precision, mixed
        )
        E = int(10**log_E)
        N = E

        indices = torch.as_tensor(
            np.random.choice(E, replace=False, size=(N,)), dtype=torch.int64
        )
        weights = torch.randn(
            indices.numel(), max_D, dtype=weights_precision.as_dtype()
        )
        count = torch.as_tensor([indices.numel()])

        weights_out = torch.empty_like(weights)

        # pyre-ignore[29]
        dram_kv.set_cuda(indices, weights, count, 1)
        # clone is needed as indices will be updated on get
        # pyre-ignore[29]
        dram_kv.get_cuda(indices.clone(), weights_out, count)
        torch.cuda.synchronize()

        logging.info(f"weights expected {weights=}")
        logging.info(f"weights actual {weights_out=}")
        assert torch.equal(weights, weights_out)

    def test_initializer(
        self,
    ):
        weights_precision: SparseType = SparseType.FP32
        emb_dim = 128
        uniform_init_lower = -1
        uniform_init_upper = 1
        dram_kv = torch.classes.fbgemm.DramKVEmbeddingCacheWrapper(
            max_D=emb_dim,
            uniform_init_lower=uniform_init_lower,
            uniform_init_upper=uniform_init_upper,
            num_shards=8,
            num_threads=32,
            row_storage_bitwidth=weights_precision.bit_rate(),
        )

        E = int(10**2)
        N = E

        indices = torch.as_tensor(
            np.random.choice(E, replace=False, size=(N,)), dtype=torch.int64
        )
        weights_out = torch.full(
            (indices.numel(), emb_dim), -2, dtype=weights_precision.as_dtype()
        )
        logging.info(f"weights initial {weights_out=}")

        count = torch.as_tensor([indices.numel()])
        # weights_out = torch.empty_like(weights)
        dram_kv.get_cuda(indices.clone(), weights_out, count)
        torch.cuda.synchronize()

        assert weights_out.size(0) == indices.numel()
        assert weights_out.size(1) == emb_dim
        for i in range(weights_out.shape[0]):
            assert torch.all(
                (weights_out[i] >= uniform_init_lower)
                & (weights_out[i] <= uniform_init_upper)
            ).item()
        logging.info(f"weights initialized {weights_out=}")

    def test_get_ids(self):
        T = 3
        D = 1
        log_E = 1
        mixed = True
        weights_precision: SparseType = SparseType.FP32

        dram_kv, Es, Ds, max_D = self.generate_fbgemm_kv_tbe(
            T, D, log_E, weights_precision, mixed
        )
        E = int(10**log_E)
        N = E

        indices = torch.as_tensor(
            np.random.choice(E, replace=False, size=(N,)), dtype=torch.int64
        )
        weights = torch.randn(
            indices.numel(), max_D, dtype=weights_precision.as_dtype()
        )
        count = torch.as_tensor([indices.numel()])
        # pyre-ignore[29]
        dram_kv.set_cuda(indices, weights, count, 1)
        torch.cuda.synchronize()

        median = torch.median(indices)
        max_val = torch.max(indices)
        filtered_indices_input = indices[(indices >= median) & (indices < max_val)]
        # pyre-ignore[29]
        ids_in_range = torch.sort(dram_kv.get_keys_in_range(median, max_val)).values

        expected_indicies = torch.sort(filtered_indices_input).values
        logging.info(f"indicies expected: {expected_indicies=}")
        logging.info(f"indicies actual: {ids_in_range=}")
        assert torch.equal(ids_in_range, expected_indicies)
