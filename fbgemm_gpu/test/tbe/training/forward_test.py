#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# pyre-ignore-all-errors[56]

import math
import random
import unittest
from unittest.mock import MagicMock, patch

import hypothesis.strategies as st
import numpy as np
import torch
from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType, SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    CacheAlgorithm,
    EmbeddingLocation,
    PoolingMode,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    ComputeDevice,
    RESParams,
    SplitTableBatchedEmbeddingBagsCodegen,
)
from fbgemm_gpu.tbe.utils import (
    b_indices,
    generate_requests,
    get_table_batched_offsets_from_dense,
    round_up,
    to_device,
)
from hypothesis import assume, given, HealthCheck, settings, Verbosity

from .. import common  # noqa E402
from ..common import (
    format_ref_tensors_in_mixed_B_layout,
    FORWARD_MAX_THREADS,
    gen_mixed_B_batch_sizes,
    get_max_thread_blocks,
    MAX_EXAMPLES_LONG_RUNNING,
    open_source,
)

if open_source:
    # pyre-ignore[21]
    from test_utils import (
        additional_decorators,
        gpu_unavailable,
        is_nvidia_device,
        optests,
        running_in_oss,
        TEST_WITH_ROCM,
    )
else:
    from fbgemm_gpu.test.test_utils import (
        additional_decorators,
        gpu_unavailable,
        is_nvidia_device,
        optests,
        running_in_oss,
        TEST_WITH_ROCM,
    )

VERBOSITY: Verbosity = Verbosity.verbose

fp8_dtype: torch.dtype = (
    torch.float8_e4m3fnuz if torch.version.hip is not None else torch.float8_e4m3fn
)

# pyre-ignore
additional_decorators.update(
    {
        # TODO: Implement the operator registrations later
        "test_faketensor__test_forward_cpu_int8": [
            unittest.skip("Operator not implemented for Meta tensors"),
        ],
        "test_faketensor__test_forward_fused_pooled_emb_quant": [
            unittest.skip("Operator not implemented for Meta tensors"),
        ],
        "test_faketensor__test_forward_gpu_no_cache_int8": [
            unittest.skip("Operator not implemented for Meta tensors"),
        ],
        "test_faketensor__test_forward_gpu_uvm_cache_int8": [
            unittest.skip("Operator not implemented for Meta tensors"),
        ],
        "test_faketensor__test_forward_cpu_fp32": [
            unittest.skip("Operator not implemented for Meta tensors"),
        ],
        # TODO: Make it compatible with opcheck tests
        "test_schema__test_forward_gpu_uvm_cache_fp16": [
            unittest.skip(
                "Failed with Argument lxu_cache_locations_output is not defined to alias output but was aliasing"
            ),
        ],
        "test_schema__test_forward_gpu_uvm_cache_fp32": [
            unittest.skip(
                "Failed with Argument lxu_cache_locations_output is not defined to alias output but was aliasing"
            ),
        ],
        # learning rate tensor needs to be on CPU to avoid D->H sync point since it will be used as float in the kernel
        # this fails fake_tensor test as the test expects all tensors to be on the same device
        "test_pt2_compliant_tag_fbgemm_split_embedding_codegen_lookup_rowwise_adagrad_function": [
            unittest.skip(
                "Operator failed on FakeTensor test since learning rate tensor is always on CPU regardless of other tensors"
            ),
        ],
    }
)


@optests.generate_opcheck_tests(fast=True, additional_decorators=additional_decorators)
class ForwardTest(unittest.TestCase):
    def execute_forward_(  # noqa C901
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        weights_precision: SparseType,
        weighted: bool,
        mixed: bool,
        mixed_B: bool,
        use_cache: bool,
        cache_algorithm: CacheAlgorithm,
        pooling_mode: PoolingMode,
        use_cpu: bool,
        output_dtype: SparseType,
        use_experimental_tbe: bool,
        enable_raw_embedding_streaming: bool = False,
        prefetch_pipeline: bool = False,
    ) -> None:
        # NOTE: cache is not applicable to CPU version.
        assume(not use_cpu or not use_cache)
        # NOTE: limit (T * B * L * D) to avoid timeout for CPU version!
        assume(not use_cpu or T * B * L * D <= 2048)
        # NOTE: CPU does not support FP16.
        assume(not (use_cpu and weights_precision == SparseType.FP16))

        # NOTE: weighted operation can be done only for SUM.
        assume(pooling_mode == PoolingMode.SUM or not weighted)
        # NOTE: No bag ops, no mixed
        assume(not mixed or pooling_mode != PoolingMode.NONE)
        # NOTE: No bag CPU doesn't supprot INT8
        assume(
            not (
                use_cpu
                and weights_precision == SparseType.INT8
                and pooling_mode == PoolingMode.NONE
            )
        )
        # TODO: Support these cases
        assume(
            not mixed_B
            or (
                weights_precision != SparseType.INT8
                and output_dtype != SparseType.INT8
                and pooling_mode != PoolingMode.NONE
            )
        )
        # NOTE: Raw embedding streaming requires UVM cache
        assume(not enable_raw_embedding_streaming or use_cache)
        # NOTE: Raw embedding streaming not supported on CPU
        assume(not enable_raw_embedding_streaming or not use_cpu)

        emb_op = SplitTableBatchedEmbeddingBagsCodegen
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

        E = int(10**log_E)
        if use_cpu:
            D = (D + 15) // 16 * 4
        else:
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

        if not mixed_B:
            Bs = [B] * T
            Bs_rank_feature = [[0]]
        else:
            Bs_rank_feature, Bs = gen_mixed_B_batch_sizes(B, T)

        compute_device = ComputeDevice.CUDA
        if use_cpu:
            managed = [EmbeddingLocation.HOST] * T
            compute_device = ComputeDevice.CPU
        elif TEST_WITH_ROCM:
            # ROCm managed memory allocation is under development
            managed = [EmbeddingLocation.DEVICE] * T
        elif use_cache:
            managed = [EmbeddingLocation.MANAGED_CACHING] * T
            if mixed:
                average_D = sum(Ds) // T
                for t, d in enumerate(Ds):
                    managed[t] = (
                        EmbeddingLocation.DEVICE if d < average_D else managed[t]
                    )
        else:
            managed = [
                np.random.choice(
                    [
                        EmbeddingLocation.DEVICE,
                        EmbeddingLocation.MANAGED,
                    ]
                )
                for _ in range(T)
            ]
        if do_pooling:
            bs = [
                to_device(torch.nn.EmbeddingBag(E, D, mode=mode, sparse=True), use_cpu)
                for (E, D) in zip(Es, Ds)
            ]
        else:
            bs = [
                to_device(torch.nn.Embedding(E, D, sparse=True), use_cpu)
                for (E, D) in zip(Es, Ds)
            ]
        if weights_precision == SparseType.INT8:
            for t in range(T):
                bs[t].weight.data.copy_(
                    torch.ops.fbgemm.Fused8BitRowwiseQuantizedToFloat(
                        torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(
                            bs[t].weight.data
                        )
                    )
                )

        if weights_precision == SparseType.NFP8:
            for t in range(T):
                bs[t].weight.data.copy_(bs[t].weight.data.to(fp8_dtype).to(torch.float))

        if weights_precision == SparseType.FP16:
            bs = [b.half() for b in bs]

        # Generate indices
        xs = [
            to_device(torch.randint(low=0, high=e, size=(b, L)), use_cpu)
            for e, b in zip(Es, Bs)
        ]
        # Generate positional weights
        xws = [to_device(torch.randn(size=(b, L)), use_cpu) for b in Bs]
        if weights_precision == SparseType.FP16:
            xws = [xw.half() for xw in xws]

        # Run baseline
        fs = (
            [
                b_indices(b, x, use_cpu=use_cpu, do_pooling=do_pooling)
                for (b, x) in zip(bs, xs)
            ]
            if not weighted
            else [
                b_indices(
                    b,
                    x,
                    per_sample_weights=xw.view(-1),
                    use_cpu=use_cpu,
                    do_pooling=do_pooling,
                )
                for (b, x, xw) in zip(bs, xs, xws)
            ]
        )

        if do_pooling:
            if mixed_B:
                f = format_ref_tensors_in_mixed_B_layout(fs, Bs_rank_feature)
            else:
                f = torch.cat([f.view(B, -1) for f in fs], dim=1)
        else:
            f = torch.cat(fs, dim=0).view(-1, D)

        # Create RES parameters if raw embedding streaming is enabled
        res_params = None
        if enable_raw_embedding_streaming:
            res_params = RESParams(
                res_store_shards=1,
                table_names=[f"table_{i}" for i in range(T)],
                table_offsets=[sum(Es[:i]) for i in range(T + 1)],
                table_sizes=Es,
            )

        # Create a TBE op
        cc = emb_op(
            embedding_specs=[
                (
                    E,
                    D,
                    EmbeddingLocation(M),
                    compute_device,
                )
                for (E, D, M) in zip(Es, Ds, managed)
            ],
            weights_precision=weights_precision,
            optimizer=(
                OptimType.EXACT_ROWWISE_ADAGRAD if mixed_B else OptimType.EXACT_SGD
            ),
            learning_rate=0.05,
            cache_algorithm=cache_algorithm,
            pooling_mode=pooling_mode,
            output_dtype=output_dtype,
            use_experimental_tbe=use_experimental_tbe,
            prefetch_pipeline=prefetch_pipeline,
            enable_raw_embedding_streaming=enable_raw_embedding_streaming,
            res_params=res_params,
        )
        # Test torch JIT script compatibility
        if not use_cpu:
            cc = torch.jit.script(cc)

        for t in range(T):
            if weights_precision == SparseType.INT8:
                b_weight = torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(
                    bs[t].weight
                )
            elif weights_precision == SparseType.NFP8:
                b_weight = bs[t].weight.to(fp8_dtype)
            else:
                b_weight = bs[t].weight
            cc.split_embedding_weights()[t].data.copy_(b_weight)

        x = torch.cat([x.contiguous().flatten() for x in xs], dim=0)
        xw = torch.cat([xw.contiguous().flatten() for xw in xws], dim=0)

        indices, offsets = get_table_batched_offsets_from_dense(x, L, sum(Bs), use_cpu)

        batch_size_per_feature_per_rank = Bs_rank_feature if mixed_B else None

        hash_zch_identities = to_device(
            torch.randint(
                low=0,
                high=E * 10,  # The upper-bound doesn't matter
                size=(B * T * L,),  # Matches dimension 0 of indices
                dtype=torch.int64,
            ),
            use_cpu,
        )
        hash_zch_runtime_meta = to_device(
            torch.randint(
                low=0,
                high=E * 10,  # The upper-bound doesn't matter
                size=(B * T * L,),  # Matches dimension 0 of indices
                dtype=torch.int64,
            ),
            use_cpu,
        )
        # Run TBE
        fc2 = (
            cc(
                indices,
                offsets,
                batch_size_per_feature_per_rank=batch_size_per_feature_per_rank,
                hash_zch_identities=hash_zch_identities,
                hash_zch_runtime_meta=hash_zch_runtime_meta,
            )
            if not weighted
            else cc(
                indices,
                offsets,
                to_device(xw.contiguous().view(-1), use_cpu),
                batch_size_per_feature_per_rank=batch_size_per_feature_per_rank,
                hash_zch_identities=hash_zch_identities,
                hash_zch_runtime_meta=hash_zch_runtime_meta,
            )
        )

        # Compare results: f = baseline, fc2 = TBE
        tolerance = (
            1.0e-5
            if weights_precision == SparseType.FP32 and output_dtype == SparseType.FP32
            else 8.0e-3
        )
        torch.testing.assert_close(
            fc2.float(), f.float(), atol=tolerance, rtol=tolerance
        )

    def test_forward_cpu_int8(
        self,
    ) -> None:
        weights_precision = SparseType.INT8
        use_cpu = True
        T = random.randint(1, 10)
        D = random.randint(2, min(256, int(2048 / T)))
        B = random.randint(1, min(128, int(2048 / T / D)))
        L = random.randint(0, min(20, int(2048 / T / D / B)))
        log_E = random.randint(3, 5)

        use_cache = False
        # cache_algorithm is don't care as we don't use cache.
        cache_algorithm = CacheAlgorithm.LRU

        pooling_mode = random.choice(
            [
                PoolingMode.SUM,
                PoolingMode.MEAN,
            ]
        )
        mixed = False
        mixed_B = False
        if pooling_mode == PoolingMode.SUM:
            weighted = random.choice([True, False])
        else:
            weighted = False
        self.execute_forward_(
            T,
            D,
            B,
            log_E,
            L,
            weights_precision,
            weighted,
            mixed,
            mixed_B,
            use_cache,
            cache_algorithm,
            pooling_mode,
            use_cpu,
            SparseType.FP32,
            False,  # use_experimental_tbe
        )

    def test_forward_cpu_fp32(
        self,
    ) -> None:
        weights_precision = SparseType.FP32
        use_cpu = True
        T = random.randint(1, 10)
        D = random.randint(2, min(256, int(2048 / T)))
        B = random.randint(1, min(128, int(2048 / T / D)))
        L = random.randint(0, min(20, int(2048 / T / D / B)))
        log_E = random.randint(3, 5)

        use_cache = False
        # cache_algorithm is don't care as we don't use cache.
        cache_algorithm = CacheAlgorithm.LRU

        pooling_mode = random.choice(
            [
                PoolingMode.SUM,
                PoolingMode.MEAN,
            ]
        )
        mixed = False
        mixed_B = random.choice([False, True])
        if pooling_mode == PoolingMode.SUM:
            weighted = random.choice([True, False])
        else:
            weighted = False
        self.execute_forward_(
            T,
            D,
            B,
            log_E,
            L,
            weights_precision,
            weighted,
            mixed,
            mixed_B,
            use_cache,
            cache_algorithm,
            pooling_mode,
            use_cpu,
            SparseType.FP32,
            False,  # use_experimental_tbe
        )

    def test_forward_cpu_fp32_nobag(
        self,
    ) -> None:
        weights_precision = SparseType.FP32
        use_cpu = True
        T = random.randint(1, 10)
        D = random.randint(2, min(256, int(2048 / T)))
        B = random.randint(1, min(128, int(2048 / T / D)))
        L = random.randint(0, min(20, int(2048 / T / D / B)))
        log_E = random.randint(3, 5)

        use_cache = False
        # cache_algorithm is don't care as we don't use cache.
        cache_algorithm = CacheAlgorithm.LRU

        pooling_mode = PoolingMode.NONE
        mixed = False
        mixed_B = False
        weighted = False
        self.execute_forward_(
            T,
            D,
            B,
            log_E,
            L,
            weights_precision,
            weighted,
            mixed,
            mixed_B,
            use_cache,
            cache_algorithm,
            pooling_mode,
            use_cpu,
            SparseType.FP32,
            False,  # use_experimental_tbe
        )

    @unittest.skipIf(True, "INT8 support is disabled")
    def test_forward_gpu_no_cache_int8(
        self,
    ) -> None:
        weights_precision = SparseType.INT8
        use_cpu = False
        T = random.randint(1, 10)
        D = random.randint(2, 256)
        B = random.randint(1, 128)
        L = random.randint(0, 20)
        log_E = random.randint(3, 5)

        use_cache = False
        # cache_algorithm is don't care as we don't use cache.
        cache_algorithm = CacheAlgorithm.LRU

        pooling_mode = random.choice(
            [
                PoolingMode.SUM,
                PoolingMode.MEAN,
                PoolingMode.NONE,
            ]
        )
        if pooling_mode == PoolingMode.NONE:
            mixed = False
        else:
            mixed = random.choice([True, False])
        mixed_B = False
        if pooling_mode == PoolingMode.SUM:
            weighted = random.choice([True, False])
        else:
            weighted = False
        self.execute_forward_(
            T,
            D,
            B,
            log_E,
            L,
            weights_precision,
            weighted,
            mixed,
            mixed_B,
            use_cache,
            cache_algorithm,
            pooling_mode,
            use_cpu,
            SparseType.FP32,
            False,  # use_experimental_tbe
        )

    def _test_forward_gpu_no_cache_fp16_impl(
        self,
        T: int,
        B: int,
        L: int,
        use_experimental_tbe: bool,
    ) -> None:
        weights_precision = SparseType.FP16
        use_cpu = False
        D = random.randint(2, 256)
        log_E = random.randint(3, 5)

        use_cache = False
        # cache_algorithm is don't care as we don't use cache.
        cache_algorithm = CacheAlgorithm.LRU

        pooling_mode = random.choice(
            [
                PoolingMode.SUM,
                PoolingMode.MEAN,
            ]
            + ([PoolingMode.NONE] if not use_experimental_tbe else [])
        )
        if pooling_mode == PoolingMode.NONE:
            mixed = False
            mixed_B = False
        else:
            mixed = random.choice([True, False])
            mixed_B = (
                random.choice([True, False]) if not use_experimental_tbe else False
            )
        if pooling_mode == PoolingMode.SUM:
            weighted = random.choice([True, False])
        else:
            weighted = False
        self.execute_forward_(
            T,
            D,
            B,
            log_E,
            L,
            weights_precision,
            weighted,
            mixed,
            mixed_B,
            use_cache,
            cache_algorithm,
            pooling_mode,
            use_cpu,
            SparseType.FP32,
            use_experimental_tbe,
        )

    @unittest.skipIf(*gpu_unavailable)
    @given(
        use_experimental_tbe=st.booleans() if not TEST_WITH_ROCM else st.just(False),
    )
    @settings(
        verbosity=VERBOSITY,
        max_examples=MAX_EXAMPLES_LONG_RUNNING,
        deadline=None,
        suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.data_too_large],
    )
    def test_forward_gpu_no_cache_fp16(
        self,
        use_experimental_tbe: bool,
    ) -> None:
        return self._test_forward_gpu_no_cache_fp16_impl(
            random.randint(1, 10),
            random.randint(1, 128),
            random.randint(0, 20),
            use_experimental_tbe,
        )

    @unittest.skipIf(*gpu_unavailable)
    @given(
        use_experimental_tbe=st.booleans() if not TEST_WITH_ROCM else st.just(False),
    )
    @settings(
        verbosity=VERBOSITY,
        max_examples=MAX_EXAMPLES_LONG_RUNNING,
        deadline=None,
        suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.data_too_large],
    )
    def test_forward_gpu_no_cache_fp16_large(
        self,
        use_experimental_tbe: bool,
    ) -> None:
        torch.cuda.empty_cache()

        max_num_threads = FORWARD_MAX_THREADS * get_max_thread_blocks(
            torch.cuda.current_stream()
        )
        # NOTE: L is arbitrarily chosen here
        L = 10
        # NOTE: Fix to the smallest value B such that (B x L) = (number of
        # indices) > (allowed grid size x block size)
        # Add 256 to B just in case B * L == max_num_threads
        B = 2 ** (math.ceil(math.log2(max_num_threads / L))) + 256
        # NOTE: T is chosen to be small enough to avoid OOM errors given that
        # B x L must be large enough
        T = 3

        assert (
            B * L > max_num_threads
        ), "Should be testing the case where B * L is larger than max_num_threads"

        return self._test_forward_gpu_no_cache_fp16_impl(
            T,
            B,
            L,
            use_experimental_tbe,
        )

    @optests.dontGenerateOpCheckTests("FP8 compute requires custom op support.")
    @unittest.skipIf(*gpu_unavailable)
    def test_forward_gpu_no_cache_fp8(
        self,
        use_experimental_tbe: bool = False,  # TODO This does not yet work when True.
    ) -> None:
        # Skip on rocm as fp8 is not supported for all versions.
        if not is_nvidia_device:
            return

        weights_precision = SparseType.NFP8
        use_cpu = False
        T = random.randint(1, 10)
        D = random.randint(2, 256)
        B = random.randint(1, 128)
        L = random.randint(0, 20)
        log_E = random.randint(3, 5)

        use_cache = False
        # cache_algorithm is don't care as we don't use cache.
        cache_algorithm = CacheAlgorithm.LRU

        pooling_mode = random.choice(
            [
                PoolingMode.SUM,
                PoolingMode.MEAN,
            ]
            + ([PoolingMode.NONE] if not use_experimental_tbe else [])
        )
        if pooling_mode == PoolingMode.NONE:
            mixed = False
            mixed_B = False
        else:
            mixed = random.choice([True, False])
            mixed_B = (
                random.choice([True, False]) if not use_experimental_tbe else False
            )
        if pooling_mode == PoolingMode.SUM:
            weighted = random.choice([True, False])
        else:
            weighted = False
        self.execute_forward_(
            T,
            D,
            B,
            log_E,
            L,
            weights_precision,
            weighted,
            mixed,
            mixed_B,
            use_cache,
            cache_algorithm,
            pooling_mode,
            use_cpu,
            SparseType.FP32,
            use_experimental_tbe,
        )

    @unittest.skipIf(*gpu_unavailable)
    @given(
        use_experimental_tbe=st.booleans(),
    )
    @settings(
        verbosity=VERBOSITY,
        max_examples=MAX_EXAMPLES_LONG_RUNNING,
        deadline=None,
        suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.data_too_large],
    )
    def test_forward_gpu_no_cache_fp32(
        self,
        use_experimental_tbe: bool,
    ) -> None:
        weights_precision = SparseType.FP32
        use_cpu = False
        T = random.randint(1, 10)
        D = random.randint(2, 256)
        B = random.randint(1, 128)
        L = random.randint(0, 20)
        log_E = random.randint(3, 5)

        use_cache = False
        # cache_algorithm is don't care as we don't use cache.
        cache_algorithm = CacheAlgorithm.LRU

        pooling_mode = random.choice(
            [
                PoolingMode.SUM,
                PoolingMode.MEAN,
            ]
            + ([PoolingMode.NONE] if not use_experimental_tbe else [])
        )
        if pooling_mode == PoolingMode.NONE:
            mixed = False
            mixed_B = False
        else:
            mixed = random.choice([True, False])
            mixed_B = (
                random.choice([True, False]) if not use_experimental_tbe else False
            )
        if pooling_mode == PoolingMode.SUM:
            weighted = random.choice([True, False])
        else:
            weighted = False
        self.execute_forward_(
            T,
            D,
            B,
            log_E,
            L,
            weights_precision,
            weighted,
            mixed,
            mixed_B,
            use_cache,
            cache_algorithm,
            pooling_mode,
            use_cpu,
            SparseType.FP32,
            use_experimental_tbe,
        )

    @unittest.skipIf(True, "INT8 support is disabled")
    @given(
        cache_algorithm=st.sampled_from(CacheAlgorithm),
    )
    @settings(
        verbosity=VERBOSITY,
        max_examples=MAX_EXAMPLES_LONG_RUNNING,
        deadline=None,
        suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.data_too_large],
    )
    def test_forward_gpu_uvm_cache_int8(
        self,
        cache_algorithm: CacheAlgorithm,
    ) -> None:
        weights_precision = SparseType.INT8
        use_cpu = False
        T = random.randint(1, 10)
        D = random.randint(2, 256)
        B = random.randint(1, 128)
        L = random.randint(0, 20)
        log_E = random.randint(3, 5)

        use_cache = True

        pooling_mode = random.choice(
            [
                PoolingMode.SUM,
                PoolingMode.MEAN,
                PoolingMode.NONE,
            ]
        )
        output_dtype = random.choice(
            [
                SparseType.FP32,
                SparseType.FP16,
                SparseType.BF16,
            ]
        )
        if pooling_mode == PoolingMode.NONE:
            mixed = False
        else:
            mixed = random.choice([True, False])
        mixed_B = False
        if pooling_mode == PoolingMode.SUM:
            weighted = random.choice([True, False])
        else:
            weighted = False
        self.execute_forward_(
            T,
            D,
            B,
            log_E,
            L,
            weights_precision,
            weighted,
            mixed,
            mixed_B,
            use_cache,
            cache_algorithm,
            pooling_mode,
            use_cpu,
            output_dtype,
            False,  # use_experimental_tbe
        )

    @unittest.skipIf(*gpu_unavailable)
    @optests.dontGenerateOpCheckTests("FP8 compute requires custom op support.")
    @given(
        cache_algorithm=st.sampled_from(CacheAlgorithm),
    )
    @settings(
        verbosity=VERBOSITY,
        max_examples=MAX_EXAMPLES_LONG_RUNNING,
        deadline=None,
        suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.data_too_large],
    )
    def test_forward_gpu_uvm_cache_fp8(
        self,
        cache_algorithm: CacheAlgorithm,
    ) -> None:
        # Skip tests on rocm since it does not work for all versions.
        if not is_nvidia_device:
            return

        weights_precision = SparseType.NFP8
        use_cpu = False
        T = random.randint(1, 10)
        D = random.randint(2, 256)
        B = random.randint(1, 128)
        L = random.randint(0, 20)
        log_E = random.randint(3, 5)

        use_cache = True

        pooling_mode = random.choice(
            [
                PoolingMode.SUM,
                PoolingMode.MEAN,
                PoolingMode.NONE,
            ]
        )
        output_dtype = random.choice(
            [
                SparseType.FP32,
                SparseType.FP16,
                SparseType.BF16,
            ]
        )
        if pooling_mode == PoolingMode.NONE:
            mixed = False
        else:
            mixed = random.choice([True, False])
        mixed_B = False
        if pooling_mode == PoolingMode.SUM:
            weighted = random.choice([True, False])
        else:
            weighted = False
        self.execute_forward_(
            T,
            D,
            B,
            log_E,
            L,
            weights_precision,
            weighted,
            mixed,
            mixed_B,
            use_cache,
            cache_algorithm,
            pooling_mode,
            use_cpu,
            output_dtype,
            False,  # use_experimental_tbe
        )

    @unittest.skipIf(*gpu_unavailable)
    @given(
        cache_algorithm=st.sampled_from(CacheAlgorithm),
        use_experimental_tbe=st.booleans(),
    )
    @settings(
        verbosity=VERBOSITY,
        max_examples=MAX_EXAMPLES_LONG_RUNNING,
        deadline=None,
        suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.data_too_large],
    )
    def test_forward_gpu_uvm_cache_fp16(
        self,
        cache_algorithm: CacheAlgorithm,
        use_experimental_tbe: bool,
    ) -> None:
        weights_precision = SparseType.FP16
        use_cpu = False
        T = random.randint(1, 10)
        D = random.randint(2, 256)
        B = random.randint(1, 128)
        L = random.randint(0, 20)
        log_E = random.randint(3, 5)

        use_cache = True

        pooling_mode = random.choice(
            [
                PoolingMode.SUM,
                PoolingMode.MEAN,
            ]
            + ([PoolingMode.NONE] if not use_experimental_tbe else [])
        )
        output_dtype = random.choice(
            [
                SparseType.FP32,
                SparseType.FP16,
                SparseType.BF16,
            ]
        )
        if pooling_mode == PoolingMode.NONE:
            mixed = False
            mixed_B = False
        else:
            mixed = random.choice([True, False])
            mixed_B = (
                random.choice([True, False]) if not use_experimental_tbe else False
            )
        if pooling_mode == PoolingMode.SUM:
            weighted = random.choice([True, False])
        else:
            weighted = False
        self.execute_forward_(
            T,
            D,
            B,
            log_E,
            L,
            weights_precision,
            weighted,
            mixed,
            mixed_B,
            use_cache,
            cache_algorithm,
            pooling_mode,
            use_cpu,
            output_dtype,
            use_experimental_tbe,
        )

    @unittest.skipIf(*gpu_unavailable)
    @given(
        cache_algorithm=st.sampled_from(CacheAlgorithm),
        use_experimental_tbe=st.booleans(),
    )
    @settings(
        verbosity=VERBOSITY,
        max_examples=MAX_EXAMPLES_LONG_RUNNING,
        deadline=None,
        suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.data_too_large],
    )
    def test_forward_gpu_uvm_cache_fp32(
        self,
        cache_algorithm: CacheAlgorithm,
        use_experimental_tbe: bool,
    ) -> None:
        weights_precision = SparseType.FP32
        use_cpu = False
        T = random.randint(1, 10)
        D = random.randint(2, 256)
        B = random.randint(1, 128)
        L = random.randint(0, 20)
        log_E = random.randint(3, 5)

        use_cache = True

        pooling_mode = random.choice(
            [
                PoolingMode.SUM,
                PoolingMode.MEAN,
            ]
            + ([PoolingMode.NONE] if not use_experimental_tbe else [])
        )
        output_dtype = random.choice(
            [
                SparseType.FP32,
                SparseType.FP16,
                SparseType.BF16,
            ]
        )
        if pooling_mode == PoolingMode.NONE:
            mixed = False
            mixed_B = False
        else:
            mixed = random.choice([True, False])
            mixed_B = (
                random.choice([True, False]) if not use_experimental_tbe else False
            )
        if pooling_mode == PoolingMode.SUM:
            weighted = random.choice([True, False])
        else:
            weighted = False
        self.execute_forward_(
            T,
            D,
            B,
            log_E,
            L,
            weights_precision,
            weighted,
            mixed,
            mixed_B,
            use_cache,
            cache_algorithm,
            pooling_mode,
            use_cpu,
            output_dtype,
            use_experimental_tbe,
        )

    @unittest.skipIf(*gpu_unavailable)
    @given(
        T=st.integers(min_value=1, max_value=10),
        D=st.integers(min_value=2, max_value=128),
        B=st.integers(min_value=1, max_value=128),
        log_E=st.integers(min_value=3, max_value=5),
        L=st.integers(min_value=0, max_value=20),
        output_dtype=st.sampled_from([SparseType.FP16]),
    )
    @settings(
        verbosity=VERBOSITY,
        max_examples=MAX_EXAMPLES_LONG_RUNNING,
        deadline=None,
        suppress_health_check=[HealthCheck.filter_too_much],
    )
    def test_forward_fused_pooled_emb_quant(
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        output_dtype: SparseType,
    ) -> None:
        Ds = [
            round_up(np.random.randint(low=int(max(0.25 * D, 1)), high=int(1.0 * D)), 4)
            for _ in range(T)
        ]
        E = int(10**log_E)
        Es = [np.random.randint(low=int(0.5 * E), high=int(2.0 * E)) for _ in range(T)]

        op = SplitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                (
                    E,
                    D,
                    EmbeddingLocation.DEVICE,
                    ComputeDevice.CUDA,
                )
                for (E, D) in zip(Es, Ds)
            ],
            output_dtype=output_dtype,
            device=torch.cuda.current_device(),
        )
        op_ref = SplitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                (
                    E,
                    D,
                    EmbeddingLocation.DEVICE,
                    ComputeDevice.CUDA,
                )
                for (E, D) in zip(Es, Ds)
            ],
            output_dtype=SparseType.FP32,
            device=torch.cuda.current_device(),
        )
        # sync weights between two ops
        split_weights = op.split_embedding_weights()
        ref_split_weights = op_ref.split_embedding_weights()
        for t in range(T):
            split_weights[t].data.copy_(ref_split_weights[t])

        requests = generate_requests(2, B, T, L, min(Es), reuse=0.1)

        for req in requests:
            indices, offsets = req.unpack_2()
            lowp_pooled_output = op(
                indices=indices,
                offsets=offsets,
            )
            fp32_pooled_output = op_ref(
                indices=indices,
                offsets=offsets,
            )
            lowp_pooled_emb_split = [
                d + 8 if output_dtype == SparseType.INT8 else d for d in op.dims
            ]
            lowp_pooled_output_per_table = torch.split(
                lowp_pooled_output, lowp_pooled_emb_split, dim=1
            )
            deq_lowp_pooled_output_per_table = [
                (
                    torch.ops.fbgemm.Fused8BitRowwiseQuantizedToFloat(t.contiguous())
                    if output_dtype == SparseType.INT8
                    else t.float()
                )
                for t in lowp_pooled_output_per_table
            ]
            fp32_pooled_output_per_table = torch.split(
                fp32_pooled_output, op.dims, dim=1
            )
            dq_fp32_pooled_output_per_table = [
                (
                    torch.ops.fbgemm.Fused8BitRowwiseQuantizedToFloat(
                        torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(
                            t.contiguous()
                        ).contiguous()
                    )
                    if output_dtype == SparseType.INT8
                    else t.half().float()
                )
                for t in fp32_pooled_output_per_table
            ]
            cat_deq_lowp_pooled_output = torch.cat(
                deq_lowp_pooled_output_per_table, dim=1
            )
            cat_dq_fp32_pooled_output = torch.cat(
                dq_fp32_pooled_output_per_table, dim=1
            )
            torch.testing.assert_close(
                cat_deq_lowp_pooled_output, cat_dq_fp32_pooled_output
            )

    def _check_raw_embedding_stream_call_counts(
        self,
        mock_raw_embedding_stream: unittest.mock.Mock,
        num_iterations: int,
        prefetch_pipeline: bool,
        L: int,
    ) -> None:
        # For TBE (not SSD), raw_embedding_stream is called once per prefetch
        # when there's data to stream
        expected_calls = num_iterations if L > 0 else 0
        if prefetch_pipeline:
            # With prefetch pipeline, there might be fewer calls initially
            expected_calls = max(0, expected_calls - 1)

        self.assertGreaterEqual(mock_raw_embedding_stream.call_count, 0)
        # Allow some flexibility in call count due to caching behavior
        self.assertLessEqual(mock_raw_embedding_stream.call_count, expected_calls + 2)

    @unittest.skipIf(*gpu_unavailable)
    @given(
        T=st.integers(min_value=1, max_value=5),
        D=st.integers(min_value=2, max_value=64),
        B=st.integers(min_value=1, max_value=32),
        log_E=st.integers(min_value=3, max_value=4),
        L=st.integers(min_value=1, max_value=10),
        weights_precision=st.sampled_from([SparseType.FP32, SparseType.FP16]),
        cache_algorithm=st.sampled_from(CacheAlgorithm),
        pooling_mode=st.sampled_from([PoolingMode.SUM, PoolingMode.MEAN]),
        weighted=st.booleans(),
        mixed=st.booleans(),
        prefetch_pipeline=st.booleans(),
    )
    @settings(
        verbosity=VERBOSITY,
        max_examples=MAX_EXAMPLES_LONG_RUNNING,
        deadline=None,
        suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.data_too_large],
    )
    def test_forward_raw_embedding_streaming(
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        weights_precision: SparseType,
        cache_algorithm: CacheAlgorithm,
        pooling_mode: PoolingMode,
        weighted: bool,
        mixed: bool,
        prefetch_pipeline: bool,
    ) -> None:
        """Test raw embedding streaming functionality integrated with forward pass."""
        num_iterations = 5
        # only LRU supports prefetch_pipeline
        assume(not prefetch_pipeline or cache_algorithm == CacheAlgorithm.LRU)

        with patch(
            "fbgemm_gpu.split_table_batched_embeddings_ops_training.torch.classes.fbgemm.RawEmbeddingStreamer"
        ) as mock_streamer_class:
            # Mock the RawEmbeddingStreamer class
            mock_streamer_instance = MagicMock()
            mock_streamer_class.return_value = mock_streamer_instance

            # Run multiple iterations to test streaming behavior
            for _ in range(num_iterations):
                self.execute_forward_(
                    T=T,
                    D=D,
                    B=B,
                    log_E=log_E,
                    L=L,
                    weights_precision=weights_precision,
                    weighted=weighted,
                    mixed=mixed,
                    mixed_B=False,  # Keep simple for streaming tests
                    use_cache=True,  # Required for streaming
                    cache_algorithm=cache_algorithm,
                    pooling_mode=pooling_mode,
                    use_cpu=False,  # Streaming not supported on CPU
                    output_dtype=SparseType.FP32,
                    use_experimental_tbe=False,
                    enable_raw_embedding_streaming=True,
                    prefetch_pipeline=prefetch_pipeline,
                )

            self._check_raw_embedding_stream_call_counts(
                mock_streamer_instance, num_iterations, prefetch_pipeline, L
            )

    @unittest.skipIf(*gpu_unavailable)
    @unittest.skipIf(*running_in_oss)
    @given(
        T=st.integers(min_value=2, max_value=8),
        D=st.integers(min_value=2, max_value=128),
        B=st.integers(min_value=1, max_value=64),
        log_E=st.integers(min_value=3, max_value=4),
        L=st.integers(min_value=1, max_value=10),
        weights_precision=st.sampled_from([SparseType.FP32, SparseType.FP16]),
        pooling_mode=st.sampled_from([PoolingMode.SUM, PoolingMode.MEAN]),
        weighted=st.booleans(),
    )
    @settings(
        verbosity=VERBOSITY,
        max_examples=MAX_EXAMPLES_LONG_RUNNING,
        deadline=None,
        suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.data_too_large],
    )
    def test_forward_mixed_cache_non_cache_tables_w_raw_embedding_streaming(
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        weights_precision: SparseType,
        pooling_mode: PoolingMode,
        weighted: bool,
    ) -> None:
        """
        Test forward pass with mixed cache and non-cache tables.
        This validates that total_cache_hash_size < total_hash_size works correctly.
        """
        # weighted operation can be done only for SUM
        assume(pooling_mode == PoolingMode.SUM or not weighted)

        E = int(10**log_E)
        D = D * 4

        # Create mixed configuration: some tables with cache, some without
        Ds = [D] * T
        Es = [E] * T

        # Mix cache and non-cache tables
        managed = []
        for t in range(T):
            if t % 2 == 0:
                # Even tables: use cache (MANAGED_CACHING)
                managed.append(EmbeddingLocation.MANAGED_CACHING)
            else:
                # Odd tables: no cache (DEVICE or MANAGED)
                managed.append(
                    np.random.choice(
                        [EmbeddingLocation.DEVICE, EmbeddingLocation.MANAGED]
                    )
                )

        compute_device = ComputeDevice.CUDA
        mode = "sum" if pooling_mode == PoolingMode.SUM else "mean"
        do_pooling = True

        # Create baseline embeddings
        bs = [
            torch.nn.EmbeddingBag(E, D, mode=mode, sparse=True).cuda()
            for E, D in zip(Es, Ds)
        ]

        if weights_precision == SparseType.FP16:
            bs = [b.half() for b in bs]

        # Create RES parameters for raw embedding streaming
        res_params = RESParams(
            res_store_shards=1,
            table_names=[f"table_{i}" for i in range(T)],
            table_offsets=[sum(Es[:i]) for i in range(T + 1)],
            table_sizes=Es,
        )

        with patch(
            "fbgemm_gpu.split_table_batched_embeddings_ops_training.torch.classes.fbgemm.RawEmbeddingStreamer"
        ) as mock_streamer_class:
            # Mock the RawEmbeddingStreamer class
            mock_streamer_instance = MagicMock()
            mock_streamer_class.return_value = mock_streamer_instance

            # Create TBE with mixed cache/non-cache configuration and raw embedding streaming
            cc = SplitTableBatchedEmbeddingBagsCodegen(
                embedding_specs=[
                    (E, D, EmbeddingLocation(M), compute_device)
                    for E, D, M in zip(Es, Ds, managed)
                ],
                weights_precision=weights_precision,
                optimizer=OptimType.EXACT_SGD,
                learning_rate=0.05,
                cache_algorithm=CacheAlgorithm.LRU,
                pooling_mode=pooling_mode,
                output_dtype=SparseType.FP32,
                enable_raw_embedding_streaming=True,
                res_params=res_params,
            )

            # Verify that total_cache_hash_size < total_hash_size
            num_cache_tables = sum(
                1 for m in managed if m == EmbeddingLocation.MANAGED_CACHING
            )
            self.assertGreater(
                num_cache_tables, 0, "Should have at least one cached table"
            )
            self.assertLess(
                num_cache_tables, T, "Should have at least one non-cached table"
            )
            self.assertLess(
                int(cc.total_cache_hash_size),
                int(cc.total_hash_size),
                "total_cache_hash_size should be less than total_hash_size for mixed config",
            )

            # Copy weights
            for t in range(T):
                if weights_precision == SparseType.FP16:
                    cc.split_embedding_weights()[t].data.copy_(bs[t].weight)
                else:
                    cc.split_embedding_weights()[t].data.copy_(bs[t].weight)

            # Run multiple forward iterations to test streaming behavior
            num_iterations = 5
            tolerance = 1.0e-5 if weights_precision == SparseType.FP32 else 8.0e-3
            previous_linearized_cache_indices = None

            for iteration in range(num_iterations):
                # Generate inputs for this iteration
                xs_iter = [torch.randint(low=0, high=e, size=(B, L)).cuda() for e in Es]
                xws_iter = [torch.randn(size=(B, L)).cuda() for _ in range(T)]
                if weights_precision == SparseType.FP16:
                    xws_iter = [xw.half() for xw in xws_iter]

                # Run baseline for this iteration
                fs_iter = (
                    [
                        b_indices(b, x, use_cpu=False, do_pooling=do_pooling)
                        for b, x in zip(bs, xs_iter)
                    ]
                    if not weighted
                    else [
                        b_indices(
                            b,
                            x,
                            per_sample_weights=xw.view(-1),
                            use_cpu=False,
                            do_pooling=do_pooling,
                        )
                        for b, x, xw in zip(bs, xs_iter, xws_iter)
                    ]
                )
                f_iter = torch.cat([f.view(B, -1) for f in fs_iter], dim=1)

                # Prepare inputs
                x = torch.cat([x.contiguous().flatten() for x in xs_iter], dim=0)
                xw = torch.cat([xw.contiguous().flatten() for xw in xws_iter], dim=0)
                indices, offsets = get_table_batched_offsets_from_dense(
                    x, L, B * T, False
                )

                # Run TBE forward
                fc2 = (
                    cc(indices, offsets)
                    if not weighted
                    else cc(indices, offsets, xw.contiguous().view(-1))
                )

                # Compare results
                torch.testing.assert_close(
                    fc2.float(), f_iter.float(), atol=tolerance, rtol=tolerance
                )

                # Check streaming calls after each iteration
                if mock_streamer_instance.stream.call_count != 0:
                    last_call_args = mock_streamer_instance.stream.call_args
                    self.assertIsNotNone(
                        last_call_args, "stream() should be called with args"
                    )

                    call_args = last_call_args[0] if last_call_args[0] else []
                    self.assertGreaterEqual(
                        len(call_args),
                        5,
                        "stream() should be called with at least 5 args",
                    )

                    streamed_indices = call_args[0]
                    streamed_weights = call_args[1]
                    # call_args[2] is hash_zch_identities (optional)
                    # call_args[3] is hash_zch_runtime_meta (optional)
                    count = call_args[4]  # Number of valid entries

                    self.assertIsInstance(streamed_indices, torch.Tensor)
                    self.assertIsInstance(streamed_weights, torch.Tensor)
                    self.assertIsInstance(count, torch.Tensor)

                    # Get the actual count of valid entries
                    valid_count = count.item()

                    # Only validate the first `valid_count` entries
                    valid_streamed_indices = streamed_indices[:valid_count].to(
                        torch.cuda.current_device()
                    )
                    valid_streamed_weights = streamed_weights[:valid_count].to(
                        torch.cuda.current_device()
                    )

                    # Check that there is at most 1 -1 entry
                    num_negative_ones = (valid_streamed_indices == -1).sum().item()
                    self.assertLessEqual(
                        num_negative_ones,
                        1,
                        f"Iteration {iteration}: Found {num_negative_ones} -1 entries in valid_streamed_indices, expected at most 1",
                    )

                    # Filter out -1 entries for validation
                    valid_mask = valid_streamed_indices != -1
                    filtered_indices = valid_streamed_indices[valid_mask]
                    filtered_weights = valid_streamed_weights[valid_mask]

                    # Validate indices are within bounds
                    self.assertGreaterEqual(filtered_indices.min().item(), 0)
                    self.assertLess(
                        filtered_indices.max().item(),
                        int(cc.total_hash_size),
                    )

                    # Compare filtered_indices (no -1s) with previous iteration's linearized cache indices
                    if previous_linearized_cache_indices is not None:
                        # streamed_indices should match the previous iteration's linearized cache indices
                        # Both are linearized indices into the embedding table
                        current_streamed_set = set(
                            filtered_indices.flatten().cpu().tolist()
                        )
                        previous_indices_set = set(
                            previous_linearized_cache_indices.flatten().cpu().tolist()
                        )

                        # Streamed indices should be exactly the same as previous iteration's indices
                        self.assertEqual(
                            current_streamed_set,
                            previous_indices_set,
                            f"Iteration {iteration}: Streamed indices should exactly match previous iteration's linearized indices. "
                            f"Streamed: {len(current_streamed_set)}, Previous: {len(previous_indices_set)}, "
                            f"Only in streamed: {current_streamed_set - previous_indices_set}, "
                            f"Only in previous: {previous_indices_set - current_streamed_set}",
                        )

                        # Get all embedding weights concatenated
                        all_weights = torch.cat(
                            cc.split_embedding_weights(),
                            dim=0,
                        )

                        selected_weights = torch.index_select(
                            all_weights,
                            0,
                            previous_linearized_cache_indices.flatten().long(),
                        )

                        # Compare filtered_weights with selected_weights
                        torch.testing.assert_close(
                            filtered_weights.float(),
                            selected_weights.float(),
                            atol=tolerance,
                            rtol=tolerance,
                            msg=f"Iteration {iteration}: Streamed weights should match split_embedding_weights at previous_linearized_cache_indices",
                        )
                # Compute linearized cache indices for current iteration
                indices_per_table = torch.split(indices, [B * L] * T)

                linear_cache_indices = torch.ops.fbgemm.linearize_cache_indices(
                    cc.cache_hash_size_cumsum,
                    indices,
                    offsets,
                )

                # Perform cache lookup to filter out indices that are not in cache
                lxu_cache_locations = torch.ops.fbgemm.lxu_cache_lookup(
                    linear_cache_indices,
                    cc.lxu_cache_state,
                    cc.total_cache_hash_size,
                )

                # pyre-fixme[9]: Comparison returns Tensor, not bool
                cache_hit_mask: torch.Tensor = lxu_cache_locations >= 0

                # Collect cache hit indices from MANAGED_CACHING tables
                current_linearized_managed_cache_indices = []
                start_idx = 0
                for t, location in enumerate(managed):
                    end_idx = start_idx + B * L
                    if location == EmbeddingLocation.MANAGED_CACHING:
                        # Filter to cache hits only
                        table_indices = indices_per_table[t]
                        table_mask = cache_hit_mask[start_idx:end_idx]
                        cache_hit_table_indices = table_indices[table_mask]
                        if len(cache_hit_table_indices) > 0:
                            # Linearize: add table offset to convert table-local indices to global indices
                            table_offset = sum(Es[:t])
                            linearized = cache_hit_table_indices + table_offset
                            current_linearized_managed_cache_indices.append(linearized)

                    start_idx = end_idx

                # Concatenate and get unique indices
                if current_linearized_managed_cache_indices:
                    all_cache_hit_indices = torch.cat(
                        current_linearized_managed_cache_indices, dim=0
                    )
                    previous_linearized_cache_indices = torch.unique(
                        all_cache_hit_indices
                    ).to(torch.cuda.current_device())
                else:
                    previous_linearized_cache_indices = None


if __name__ == "__main__":
    unittest.main()
