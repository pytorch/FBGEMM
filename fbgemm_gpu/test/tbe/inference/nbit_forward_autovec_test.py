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

import hypothesis.strategies as st
import torch
from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_inference import (
    IntNBitTableBatchedEmbeddingBagsCodegen,
)
from fbgemm_gpu.tbe.cache.cache_config import CacheAlgorithm
from fbgemm_gpu.tbe.config.embedding_config import EmbeddingLocation, PoolingMode
from fbgemm_gpu.tbe.utils import quantize_embs
from hypothesis import given, settings, Verbosity

from ..common import MAX_EXAMPLES, TEST_WITH_ROCM
from .common import get_nbit_weights_ty, NBitFowardTestCommon

# Force the autovec CPU kernels for this whole test process. The backend
# selection reads FBGEMM_FORCE_AUTOVEC / FBGEMM_NO_ASMJIT exactly once (a
# function-local static in Utils.cc), so the vars must be set before the first
# kernel dispatch. Setting them at import time is the earliest reliable point
# and pins the backend regardless of test execution order.
os.environ["FBGEMM_FORCE_AUTOVEC"] = "1"
os.environ["FBGEMM_NO_ASMJIT"] = "1"

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
        nbit_weights_ty: SparseType | None,
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

    def test_nbit_forward_cpu_multi_table_pooled_pruned_autovec(self) -> None:
        """Autovec CPU SUM-pooled nbit forward zeroes empty/fully-pruned bags
        for FLOAT (FP16) weights.

        This is the regression guard for the float autovec/reference SpMDM `-1`
        handling. The base FP32/FP16/BF16 SpMDM kernel is reached with
        ``scale_bias_last == false`` (CPU TBE), where ``-1`` marks a pruned row
        that must contribute 0. Previously the float autovec kernel returned
        ``false`` on ``-1`` (unlike the int8/nbit/asmjit paths that skip it),
        leaving whole pooled tables unwritten -- observed as an all-zero table
        on ARM/OSS once the redundant ``output.fill_(0)`` pre-zero is dropped.

        The test forces the autovec kernel (via FBGEMM_FORCE_AUTOVEC, set at
        module import) and pins the invariant directly: every embedding row is
        1.0, so each output column must equal the count of non-pruned indices in
        that (table, bag); empty (L == 0) and fully-pruned bags must be exactly
        0. Without the `-1` skip this fails; with it, it passes.
        """
        E = 100
        Ds = [16, 24, 40]  # distinct per-table widths -> non-trivial D_offsets
        T = len(Ds)
        # Ragged per-(table, bag) lengths; include empty bags (L == 0).
        Ls = [
            [0, 2, 5, 0, 3, 4],
            [1, 0, 4, 2, 0, 6],
            [3, 3, 0, 1, 5, 0],
        ]
        B = len(Ls[0])
        fully_pruned = [4, 2, 3]  # per table: a non-empty bag whose indices are all -1

        for output_dtype in (SparseType.FP32, SparseType.FP16, SparseType.BF16):
            with self.subTest(output_dtype=output_dtype):
                op = IntNBitTableBatchedEmbeddingBagsCodegen(
                    embedding_specs=[
                        # FP16 weights -> the float (base) SpMDM kernel path.
                        ("", E, Ds[t], SparseType.FP16, EmbeddingLocation.HOST)
                        for t in range(T)
                    ],
                    output_dtype=output_dtype,
                    pooling_mode=PoolingMode.SUM,
                    device="cpu",
                )
                op.fill_random_weights()

                # Overwrite every row of every table with 1.0 (exact in fp16).
                for t in range(T):
                    quant_weights, quant_scale_shift = quantize_embs(
                        torch.ones(E, Ds[t]), SparseType.FP16
                    )
                    weights, scale_shift = op.split_embedding_weights()[t]
                    weights.copy_(quant_weights)
                    if quant_scale_shift is not None:
                        self.assertIsNotNone(scale_shift)
                        scale_shift.copy_(quant_scale_shift)

                # Build indices/offsets in (table, bag) order, pruning ~1/3 of
                # indices plus the designated fully-pruned bag per table.
                all_indices: list[int] = []
                offsets: list[int] = [0]
                expected = torch.zeros(B, sum(Ds), dtype=torch.float)
                d_prefix = [0]
                for d in Ds:
                    d_prefix.append(d_prefix[-1] + d)
                running = 0
                for t in range(T):
                    for b in range(B):
                        kept = 0
                        for _ in range(Ls[t][b]):
                            prune = (b == fully_pruned[t]) or (running % 3 == 0)
                            all_indices.append(-1 if prune else (running % E))
                            kept += 0 if prune else 1
                            running += 1
                        offsets.append(len(all_indices))
                        expected[b, d_prefix[t] : d_prefix[t + 1]] = float(kept)

                output = op(
                    indices=torch.tensor(all_indices, dtype=torch.int),
                    offsets=torch.tensor(offsets, dtype=torch.int),
                )

                self.assertEqual(tuple(output.shape), (B, sum(Ds)))
                self.assertTrue(torch.isfinite(output.float()).all().item())
                torch.testing.assert_close(
                    output.float().cpu(),
                    expected,
                    atol=1.0e-2,
                    rtol=1.0e-2,
                    equal_nan=False,  # leaked uninitialized data (NaN) must fail
                )


if __name__ == "__main__":
    unittest.main()
