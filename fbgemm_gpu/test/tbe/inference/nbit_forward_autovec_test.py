#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[56]

import os
import random
import unittest
from typing import Optional

import hypothesis.strategies as st
import torch
from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    CacheAlgorithm,
    PoolingMode,
)
from hypothesis import given, settings, Verbosity

from ..common import MAX_EXAMPLES, TEST_WITH_ROCM
from .common import get_nbit_weights_ty, NBitFowardTestCommon

VERBOSITY: Verbosity = Verbosity.verbose


class NBitFowardAutovecTest(NBitFowardTestCommon):
    @unittest.skipIf(
        TEST_WITH_ROCM,
        "Test appears to be unreliable on ROCm",
    )
    @given(
        nbit_weights_ty=get_nbit_weights_ty(),
        pooling_mode=st.sampled_from(
            [PoolingMode.SUM, PoolingMode.MEAN, PoolingMode.NONE]
        ),
        indices_dtype=st.sampled_from([torch.int32, torch.int64]),
        output_dtype=st.sampled_from(
            [SparseType.FP32, SparseType.FP16, SparseType.BF16]
        ),
    )
    @settings(
        verbosity=VERBOSITY,
        max_examples=MAX_EXAMPLES,
        deadline=None,
    )
    def test_nbit_forward_cpu_autovec(
        self,
        nbit_weights_ty: Optional[SparseType],
        pooling_mode: PoolingMode,
        indices_dtype: torch.dtype,
        output_dtype: SparseType,
    ) -> None:
        use_cpu = True
        T = random.randint(1, 50)
        B = random.randint(0, 128)
        L = random.randint(0, 32)
        D = random.randint(2, 2048)
        log_E = random.randint(2, 4)

        use_cache = False
        # cache_algorithm is don't care as we don't use cache.
        cache_algorithm = CacheAlgorithm.LRU

        mixed = random.choice([True, False])
        if pooling_mode == PoolingMode.SUM:
            weighted = random.choice([True, False])
        else:
            weighted = False

        if nbit_weights_ty is None:
            # don't care when mixed type is used.
            weights_ty: SparseType = SparseType.INT8
            mixed_weights_ty = True
        else:
            weights_ty: SparseType = nbit_weights_ty
            mixed_weights_ty = False

        os.environ["FBGEMM_FORCE_AUTOVEC"] = "1"
        os.environ["FBGEMM_NO_ASMJIT"] = "1"

        self.execute_nbit_forward_(
            T,
            D,
            B,
            log_E,
            L,
            weighted,
            mixed,
            pooling_mode,
            weights_ty,
            use_cache,
            cache_algorithm,
            use_cpu,
            False,
            False,
            mixed_weights_ty,
            indices_dtype,
            output_dtype,
        )

        del os.environ["FBGEMM_FORCE_AUTOVEC"]
        del os.environ["FBGEMM_NO_ASMJIT"]


if __name__ == "__main__":
    unittest.main()
