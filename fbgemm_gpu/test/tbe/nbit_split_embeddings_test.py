#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors[56]

import random
import unittest

import fbgemm_gpu
import hypothesis.strategies as st
import numpy as np
import torch

from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.split_embedding_utils import generate_requests, round_up
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    CacheAlgorithm,
    EmbeddingLocation,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_inference import (
    IntNBitTableBatchedEmbeddingBagsCodegen,
)

from hypothesis import assume, given, HealthCheck, settings, Verbosity

from . import common  # noqa E402,F401
from .common import MAX_EXAMPLES, MAX_EXAMPLES_LONG_RUNNING  # noqa E402

# pyre-fixme[16]: Module `fbgemm_gpu` has no attribute `open_source`.
open_source: bool = getattr(fbgemm_gpu, "open_source", False)

if open_source:
    # pyre-ignore[21]
    from test_utils import gpu_unavailable, optests, TEST_WITH_ROCM
else:
    from fbgemm_gpu.test.test_utils import gpu_unavailable, optests, TEST_WITH_ROCM


VERBOSITY: Verbosity = Verbosity.verbose


@optests.generate_opcheck_tests(fast=True)
class NBitSplitEmbeddingsTest(unittest.TestCase):
    @unittest.skipIf(*gpu_unavailable)
    @given(
        T=st.integers(min_value=1, max_value=10),
        D=st.integers(min_value=2, max_value=128),
        B=st.integers(min_value=1, max_value=128),
        log_E=st.integers(min_value=3, max_value=5),
        L=st.integers(min_value=0, max_value=20),
        weights_ty=st.sampled_from(
            [
                SparseType.FP32,
                SparseType.FP16,
                SparseType.INT8,
                SparseType.INT4,
                SparseType.INT2,
            ]
        ),
        output_dtype=st.sampled_from(
            [
                SparseType.FP16,
                SparseType.BF16,
                SparseType.INT8,
            ]
        )
        if not TEST_WITH_ROCM
        else st.sampled_from(
            [
                SparseType.FP16,
                # The counterparts of __nv_bfloat16 and __nv_bfloat162 are not supported on ROCm
                SparseType.INT8,
            ]
        ),
    )
    @settings(
        verbosity=VERBOSITY,
        max_examples=MAX_EXAMPLES_LONG_RUNNING,
        deadline=None,
        suppress_health_check=[HealthCheck.filter_too_much],
    )
    def test_nbit_split_embedding_weights_with_scale_and_bias(
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        weights_ty: SparseType,
        output_dtype: SparseType,
    ) -> None:
        D_alignment = max(weights_ty.align_size() for t in range(T))
        D_alignment = max(D_alignment, output_dtype.align_size())
        D = round_up(D, D_alignment)
        # BF16 output only works for CUDA device sm80+ (e.g., A100)
        assume(
            torch.cuda.is_available()
            and torch.cuda.get_device_capability() >= (8, 0)
            or not output_dtype == SparseType.BF16
        )
        Ds = [
            round_up(
                np.random.randint(low=int(max(0.25 * D, 1)), high=int(1.0 * D)),
                D_alignment,
            )
            for _ in range(T)
        ]
        Ds = [D] * T
        E = int(10**log_E)
        Es = [np.random.randint(low=int(0.5 * E), high=int(2.0 * E)) for _ in range(T)]

        weights_ty_list = [weights_ty] * T
        managed = [EmbeddingLocation.DEVICE] * T
        op = IntNBitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                (
                    "",
                    E,
                    D,
                    W_TY,
                    EmbeddingLocation(M),
                )
                for (E, D, M, W_TY) in zip(Es, Ds, managed, weights_ty_list)
            ],
            output_dtype=output_dtype,
            device=torch.cuda.current_device(),
        )
        # Initialize the random weights for int nbit table split embedding bag
        op.fill_random_weights()

        # sync weights between two ops
        split_weights = op.split_embedding_weights()
        split_weights_with_scale_bias = op.split_embedding_weights_with_scale_bias(
            split_scale_bias_mode=2
        )
        for t in range(T):
            (weights, scale_bias) = split_weights[t]
            (weights2, scale, bias) = split_weights_with_scale_bias[t]
            torch.testing.assert_close(weights2, weights)
            if scale is None:
                self.assertIsNone(scale_bias)
                self.assertIsNone(bias)
            else:
                torch.testing.assert_close(
                    scale.cpu(),
                    torch.tensor(
                        scale_bias[:, : scale_bias.size(1) // 2]
                        .contiguous()
                        .cpu()
                        .numpy()
                        .view(np.float16)
                    ),
                )
                torch.testing.assert_close(
                    bias.cpu(),
                    torch.tensor(
                        scale_bias[:, scale_bias.size(1) // 2 :]
                        .contiguous()
                        .cpu()
                        .numpy()
                        .view(np.float16)
                    ),
                )

    @unittest.skipIf(*gpu_unavailable)
    @given(
        weights_ty=st.sampled_from(
            [
                SparseType.FP32,
                SparseType.FP16,
                SparseType.INT8,
                SparseType.INT4,
                SparseType.INT2,
            ]
        ),
        emulate_pruning=st.booleans(),
    )
    @settings(verbosity=VERBOSITY, max_examples=MAX_EXAMPLES, deadline=None)
    def test_int_nbit_split_embedding_uvm_caching_codegen_lookup_function(
        self,
        weights_ty: SparseType,
        emulate_pruning: bool,
    ) -> None:
        # TODO: support direct-mapped in int_nbit_split_embedding_uvm_caching_codegen_lookup_function
        # This test is for int_nbit_split_embedding_uvm_caching_codegen_lookup_function.
        # We run IntNBitTableBatchedEmbeddingBagsCodegen with UVM_CACHING, and then
        # run int_nbit_split_embedding_uvm_caching_codegen_lookup_function with the
        # exact same cache configuration. As both use the same logic, the result
        # as well as cache state should match.

        # Currently, int_nbit_split_embedding_uvm_caching_codegen_lookup_function supports only LRU.
        cache_algorithm = CacheAlgorithm.LRU
        associativity = 32  # Currently, hard-coded 32-way set associative.
        current_device: torch.device = torch.device(torch.cuda.current_device())

        T = random.randint(1, 5)
        B = random.randint(1, 128)
        L = random.randint(1, 20)
        D = random.randint(2, 256)
        log_E = random.randint(3, 5)

        iters = 3
        E = int(10**log_E)

        D_alignment = (
            1 if weights_ty.bit_rate() % 8 == 0 else int(8 / weights_ty.bit_rate())
        )
        D = round_up(D, D_alignment)

        # Currently, int_nbit_split_embedding_uvm_caching_codegen_lookup_function supports only all UVM or all UVM_CACHING.
        Ds = [D] * T
        Es = [E] * T
        managed_caching = [EmbeddingLocation.MANAGED_CACHING] * T

        # Note both cc_ref and cc use caching.
        cc_ref = IntNBitTableBatchedEmbeddingBagsCodegen(
            [("", E, D, weights_ty, M) for (E, D, M) in zip(Es, Ds, managed_caching)],
            cache_algorithm=cache_algorithm,
        )
        cc_ref.fill_random_weights()

        # cc is only for cache states; we test int_nbit_split_embedding_uvm_caching_codegen_lookup_function directly;
        # hence, no need to synchronize cc's weights with cc_ref's.
        cc = IntNBitTableBatchedEmbeddingBagsCodegen(
            [("", E, D, weights_ty, M) for (E, D, M) in zip(Es, Ds, managed_caching)],
            cache_algorithm=cache_algorithm,
        )
        cc.fill_random_weights()

        # weights_placement for all UVM case.
        managed_uvm = [EmbeddingLocation.MANAGED] * T
        placement_uvm = torch.tensor(
            managed_uvm, device=current_device, dtype=torch.int32
        )

        # zero size HBM cache for UVM case.
        zero_size_cache_weights = torch.zeros(
            0, 0, device=current_device, dtype=torch.uint8
        )

        requests = generate_requests(
            iters, B, T, L, min(Es), reuse=0.1, emulate_pruning=emulate_pruning
        )
        for indices, offsets, _ in requests:
            indices = indices.int()
            offsets = offsets.int()
            output_ref = cc_ref(indices, offsets)

            # int_nbit_split_embedding_uvm_caching_codegen_lookup_function for UVM_CACHING.
            # using weights and other params from cc_ref, but
            # cache states from cc.
            output_uvm_caching = torch.ops.fbgemm.int_nbit_split_embedding_uvm_caching_codegen_lookup_function(
                dev_weights=cc_ref.weights_host
                if cc_ref.host_size > 0
                else cc_ref.weights_dev,
                uvm_weights=cc_ref.weights_uvm,
                weights_placements=cc_ref.weights_placements,
                weights_offsets=cc_ref.weights_offsets,
                weights_tys=cc_ref.weights_tys,
                D_offsets=cc_ref.D_offsets,
                total_D=cc_ref.total_D,
                max_int2_D=cc_ref.max_int2_D,
                max_int4_D=cc_ref.max_int4_D,
                max_int8_D=cc_ref.max_int8_D,
                max_float16_D=cc_ref.max_float16_D,
                max_float32_D=cc_ref.max_float32_D,
                indices=indices,
                offsets=offsets,
                pooling_mode=int(cc_ref.pooling_mode),
                indice_weights=None,
                output_dtype=cc_ref.output_dtype,
                lxu_cache_weights=cc.lxu_cache_weights,  # cc, not cc_ref.
                lxu_cache_locations=torch.empty(0, dtype=torch.int32).fill_(-1),
                row_alignment=cc_ref.row_alignment,
                max_float8_D=cc_ref.max_float8_D,
                fp8_exponent_bits=cc_ref.fp8_exponent_bits,
                fp8_exponent_bias=cc_ref.fp8_exponent_bias,
                # Additional args for UVM_CACHING: using cc, not cc_ref.
                cache_hash_size_cumsum=cc.cache_hash_size_cumsum,
                total_cache_hash_size=cc.total_cache_hash_size,
                cache_index_table_map=cc.cache_index_table_map,
                lxu_cache_state=cc.lxu_cache_state,
                lxu_state=cc.lxu_state,
            )
            torch.testing.assert_close(output_uvm_caching, output_ref, equal_nan=True)
            # cache status; we use the exact same logic, but still assigning ways in a associative cache can be
            # arbitrary. We compare sum along ways in each set, instead of expecting exact tensor match.
            cache_weights_ref = torch.reshape(
                cc_ref.lxu_cache_weights,
                [-1, associativity],
            )
            cache_weights = torch.reshape(cc.lxu_cache_weights, [-1, associativity])
            torch.testing.assert_close(
                torch.sum(cache_weights_ref, 1),
                torch.sum(cache_weights, 1),
                equal_nan=True,
            )
            torch.testing.assert_close(
                torch.sum(cc.lxu_cache_state, 1),
                torch.sum(cc_ref.lxu_cache_state, 1),
                equal_nan=True,
            )
            # lxu_state can be different as time_stamp values can be different.
            # we check the entries with max value.
            max_timestamp_ref = torch.max(cc_ref.lxu_state)
            max_timestamp_uvm_caching = torch.max(cc.lxu_state)
            x = cc_ref.lxu_state == max_timestamp_ref
            y = cc.lxu_state == max_timestamp_uvm_caching
            torch.testing.assert_close(torch.sum(x, 1), torch.sum(y, 1))

            # int_nbit_split_embedding_uvm_caching_codegen_lookup_function for UVM.
            output_uvm = torch.ops.fbgemm.int_nbit_split_embedding_uvm_caching_codegen_lookup_function(
                dev_weights=cc_ref.weights_host
                if cc_ref.host_size > 0
                else cc_ref.weights_dev,
                uvm_weights=cc_ref.weights_uvm,
                weights_placements=placement_uvm,  # all UVM weights placement.
                weights_offsets=cc_ref.weights_offsets,
                weights_tys=cc_ref.weights_tys,
                D_offsets=cc_ref.D_offsets,
                total_D=cc_ref.total_D,
                max_int2_D=cc_ref.max_int2_D,
                max_int4_D=cc_ref.max_int4_D,
                max_int8_D=cc_ref.max_int8_D,
                max_float16_D=cc_ref.max_float16_D,
                max_float32_D=cc_ref.max_float32_D,
                indices=indices,
                offsets=offsets,
                pooling_mode=int(cc_ref.pooling_mode),
                indice_weights=None,
                output_dtype=cc_ref.output_dtype,
                lxu_cache_weights=zero_size_cache_weights,  # empty HBM cache.
                lxu_cache_locations=torch.empty(0, dtype=torch.int32).fill_(-1),
                row_alignment=cc_ref.row_alignment,
                max_float8_D=cc_ref.max_float8_D,
                fp8_exponent_bits=cc_ref.fp8_exponent_bits,
                fp8_exponent_bias=cc_ref.fp8_exponent_bias,
                # Additional args for UVM_CACHING; not needed for UVM.
                cache_hash_size_cumsum=None,
                total_cache_hash_size=None,
                cache_index_table_map=None,
                lxu_cache_state=None,
                lxu_state=None,
            )
            torch.testing.assert_close(output_uvm, output_ref, equal_nan=True)


if __name__ == "__main__":
    unittest.main()
