#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors[56]

import copy
import pickle
import random
import unittest

from itertools import accumulate
from typing import Callable, Dict, List, Optional, Tuple

import fbgemm_gpu
import hypothesis.strategies as st
import numpy as np
import torch

from fbgemm_gpu.split_embedding_configs import (
    EmbOptimType as OptimType,
    FP8QuantizationConfig,
    SparseType,
)
from fbgemm_gpu.split_embedding_utils import (
    b_indices,
    fake_quantize_embs,
    generate_requests,
    get_table_batched_offsets_from_dense,
    quantize_embs,
    round_up,
    to_device,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    BoundsCheckMode,
    CacheAlgorithm,
    EmbeddingLocation,
    PoolingMode,
    RecordCacheMetrics,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_inference import (
    IntNBitTableBatchedEmbeddingBagsCodegen,
    rounded_row_size_in_bytes,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    ComputeDevice,
    DEFAULT_ASSOC,
    INT8_EMB_ROW_DIM_OFFSET,
    SplitTableBatchedEmbeddingBagsCodegen,
)

from hypothesis import assume, given, HealthCheck, settings, Verbosity
from hypothesis.strategies import composite
from torch import Tensor

torch.ops.import_module("fbgemm_gpu.sparse_ops")

# pyre-fixme[16]: Module `fbgemm_gpu` has no attribute `open_source`.
open_source: bool = getattr(fbgemm_gpu, "open_source", False)

if open_source:
    # pyre-ignore[21]
    from test_utils import gpu_available, gpu_unavailable, optests, TEST_WITH_ROCM
else:
    from fbgemm_gpu.test.test_utils import (
        gpu_available,
        gpu_unavailable,
        optests,
        TEST_WITH_ROCM,
    )


MAX_EXAMPLES = 40

# For long running tests reduce the number of iterations to reduce timeout errors.
MAX_EXAMPLES_LONG_RUNNING = 15


VERBOSITY: Verbosity = Verbosity.verbose


settings.register_profile("derandomize", derandomize=True)
settings.load_profile("derandomize")


@composite
# pyre-ignore
def get_nbit_weights_ty(draw) -> Optional[SparseType]:
    """
    Returns None if mixed weights ty should be used, otherwise, returns specific SparseType.
    """
    mixed_weights_ty = draw(st.booleans())
    if mixed_weights_ty:
        return None
    return draw(
        st.sampled_from(
            [
                SparseType.FP32,
                SparseType.FP16,
                SparseType.FP8,
                SparseType.INT8,
                SparseType.INT4,
                SparseType.INT2,
            ]
        )
    )


def gen_mixed_B_batch_sizes(B: int, T: int) -> Tuple[List[List[int]], List[int]]:
    num_ranks = np.random.randint(low=1, high=4)
    low = max(int(0.25 * B), 1)
    high = int(B)
    if low == high:
        Bs_rank_feature = [[B] * num_ranks for _ in range(T)]
    else:
        Bs_rank_feature = [
            np.random.randint(low=low, high=high, size=num_ranks).tolist()
            for _ in range(T)
        ]
    Bs = [sum(Bs_feature) for Bs_feature in Bs_rank_feature]
    return Bs_rank_feature, Bs


def format_ref_tensors_in_mixed_B_layout(
    ref_tensors: List[torch.Tensor], Bs_rank_feature: List[List[int]]
) -> torch.Tensor:
    # Relayout the reference tensor
    # Jagged dimension: (rank, table, local batch)
    num_ranks = len(Bs_rank_feature[0])
    split_tensors = [[] for _ in range(num_ranks)]  # shape (rank, table)
    for t, ref_tensor in enumerate(ref_tensors):
        assert ref_tensor.shape[0] == sum(Bs_rank_feature[t])
        tensors = ref_tensor.split(Bs_rank_feature[t])
        for r, tensor in enumerate(tensors):
            split_tensors[r].append(tensor.flatten())
    concat_list = []
    for r in range(num_ranks):
        concat_list += split_tensors[r]
    return torch.cat(concat_list, dim=0)


# pyre-ignore
additional_decorators: Dict[str, List[Callable]] = {
    "test_schema__test_backward_none_with_rowwise_adagrad": [
        unittest.skip("Cannot access data pointer of Tensor that doesn't have storage")
    ],
    "test_faketensor__test_backward_none_with_rowwise_adagrad": [
        unittest.skip("Cannot access data pointer of Tensor that doesn't have storage")
    ],
    "test_autograd_registration__test_backward_none_with_rowwise_adagrad": [
        unittest.skip("Cannot access data pointer of Tensor that doesn't have storage")
    ],
    "test_faketensor__test_nbit_forward_uvm_cache": [
        unittest.skip("CUDA Assert"),
    ],
    "test_faketensor__test_nbit_uvm_cache_stats": [
        unittest.skip("very slow"),
    ],
    "test_faketensor__test_nbit_direct_mapped_uvm_cache_stats": [
        unittest.skip("very slow"),
    ],
    # Implement the operator registrations later
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
    "test_faketensor__test_nbit_forward_cpu": [
        unittest.skip("Operator not implemented for Meta tensors"),
    ],
    "test_faketensor__test_nbit_forward_fused_pooled_emb_quant": [
        unittest.skip("Operator not implemented for Meta tensors"),
    ],
    "test_faketensor__test_nbit_forward_gpu_no_cache": [
        unittest.skip("Operator not implemented for Meta tensors"),
    ],
    "test_faketensor__test_nbit_forward_gpu_no_cache_fp8_2048": [
        unittest.skip("Operator not implemented for Meta tensors"),
    ],
}


@optests.generate_opcheck_tests(fast=True, additional_decorators=additional_decorators)
class SplitTableBatchedEmbeddingsTest(unittest.TestCase):
    _do_cuda_memory_leak_check = True

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
                # FIXME: INT2 caused big numerical error for this test
                # SparseType.INT2,
            ]
        ),
        output_dtype=st.sampled_from(
            [
                SparseType.FP16,
                SparseType.BF16,
                SparseType.INT8,
                # SparseType.INT4,
            ]
        )
        if not TEST_WITH_ROCM
        else st.sampled_from(
            [
                SparseType.FP16,
                # The counterparts of __nv_bfloat16 and __nv_bfloat162 are not supported on ROCm
                SparseType.INT8,
                # SparseType.INT4,
            ]
        ),
    )
    @settings(
        verbosity=VERBOSITY,
        max_examples=MAX_EXAMPLES_LONG_RUNNING,
        deadline=None,
        suppress_health_check=[HealthCheck.filter_too_much],
    )
    def test_nbit_forward_fused_pooled_emb_quant(
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

        op_ref = IntNBitTableBatchedEmbeddingBagsCodegen(
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
            output_dtype=SparseType.FP32,
            device=torch.cuda.current_device(),
        )
        # Initialize the random weights for int nbit table split embedding bag
        op_ref.fill_random_weights()

        # sync weights between two ops
        split_weights = op.split_embedding_weights()
        ref_split_weights = op_ref.split_embedding_weights()
        for t in range(T):
            (weights, scale_shift) = split_weights[t]
            (ref_weights, ref_scale_shift) = ref_split_weights[t]
            self.assertEqual(weights.size(), ref_weights.size())
            element_size = weights_ty_list[t].bit_rate() / 8.0
            rand_tensor = torch.rand(
                ref_weights.shape[0], int(ref_weights.shape[1] / element_size)
            )
            rand_weights, rand_scale_shift = quantize_embs(
                rand_tensor, weights_ty_list[t]
            )
            ref_weights.copy_(rand_weights)
            weights.copy_(ref_weights)
            if rand_scale_shift is not None:
                self.assertIsNotNone(scale_shift)
                self.assertIsNotNone(ref_scale_shift)
                ref_scale_shift.copy_(rand_scale_shift)
                scale_shift.copy_(ref_scale_shift)

        requests = generate_requests(1, B, T, L, min(Es), reuse=0.1)
        for indices, offsets, _ in requests:
            lowp_pooled_output = op(
                indices=indices.int(),
                offsets=offsets.int(),
            )
            fp32_pooled_output = op_ref(
                indices=indices.int(),
                offsets=offsets.int(),
            )
            lowp_pooled_emb_split = [
                d + 8 if output_dtype == SparseType.INT8 else d for d in Ds
            ]
            lowp_pooled_output_per_table = torch.split(
                lowp_pooled_output, lowp_pooled_emb_split, dim=1
            )
            deq_lowp_pooled_output_per_table = [
                torch.ops.fbgemm.Fused8BitRowwiseQuantizedToFloat(t.contiguous())
                if output_dtype == SparseType.INT8
                else t.float()
                for t in lowp_pooled_output_per_table
            ]
            fp32_pooled_output_per_table = torch.split(fp32_pooled_output, Ds, dim=1)
            dq_fp32_pooled_output_per_table = [
                torch.ops.fbgemm.Fused8BitRowwiseQuantizedToFloat(
                    torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(
                        t.contiguous()
                    ).contiguous()
                ).contiguous()
                if output_dtype == SparseType.INT8
                else t.half().float()
                for t in fp32_pooled_output_per_table
            ]
            cat_deq_lowp_pooled_output = torch.cat(
                deq_lowp_pooled_output_per_table, dim=1
            )
            cat_dq_fp32_pooled_output = torch.cat(
                dq_fp32_pooled_output_per_table, dim=1
            )
            torch.testing.assert_close(
                cat_deq_lowp_pooled_output,
                cat_dq_fp32_pooled_output,
                rtol=1e-2,
                atol=1e-2,
                equal_nan=True,
            )

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

    def _generate_cache_tbes(
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        mixed: bool,
        cache_algorithm: CacheAlgorithm = CacheAlgorithm.LRU,
        prefetch_pipeline: bool = False,
        use_int_weight: bool = False,
    ) -> Tuple[
        SplitTableBatchedEmbeddingBagsCodegen,
        SplitTableBatchedEmbeddingBagsCodegen,
        int,
        int,
    ]:
        lr = 1.0 if use_int_weight else 0.02
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
        managed = [EmbeddingLocation.MANAGED_CACHING] * T
        if mixed:
            average_D = sum(Ds) // T
            for t, d in enumerate(Ds):
                managed[t] = EmbeddingLocation.DEVICE if d < average_D else managed[t]
        cc_ref = SplitTableBatchedEmbeddingBagsCodegen(
            [
                (
                    E,
                    D,
                    EmbeddingLocation.DEVICE,
                    ComputeDevice.CUDA,
                )
                for (E, D) in zip(Es, Ds)
            ],
            stochastic_rounding=False,
            prefetch_pipeline=False,
            learning_rate=lr,
        )
        cc = SplitTableBatchedEmbeddingBagsCodegen(
            [(E, D, M, ComputeDevice.CUDA) for (E, D, M) in zip(Es, Ds, managed)],
            cache_algorithm=cache_algorithm,
            stochastic_rounding=False,
            prefetch_pipeline=prefetch_pipeline,
            learning_rate=lr,
        )

        if use_int_weight:
            min_val = -20
            max_val = +20
            for param in cc_ref.split_embedding_weights():
                p = torch.randint(
                    int(min_val),
                    int(max_val) + 1,
                    size=param.shape,
                    device=param.device,
                )
                param.data.copy_(p)

        for t in range(T):
            self.assertEqual(
                cc.split_embedding_weights()[t].size(),
                cc_ref.split_embedding_weights()[t].size(),
            )
            cc.split_embedding_weights()[t].data.copy_(
                cc_ref.split_embedding_weights()[t]
            )

        return (cc, cc_ref, min(Es), sum(Ds))

    @optests.dontGenerateOpCheckTests("Serial OOM")
    @unittest.skipIf(*gpu_unavailable)
    @given(
        T=st.integers(min_value=1, max_value=5),
        D=st.integers(min_value=2, max_value=256),
        B=st.integers(min_value=1, max_value=128),
        log_E=st.integers(min_value=3, max_value=5),
        L=st.integers(min_value=1, max_value=20),
        mixed=st.booleans(),
        cache_algorithm=st.sampled_from(CacheAlgorithm),
    )
    @settings(verbosity=VERBOSITY, max_examples=MAX_EXAMPLES, deadline=None)
    def test_cache_pipeline(
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        mixed: bool,
        cache_algorithm: CacheAlgorithm,
    ) -> None:
        cc, cc_ref, min_Es, sum_Ds = self._generate_cache_tbes(
            T, D, B, log_E, L, mixed, cache_algorithm
        )
        iters = 3
        requests = generate_requests(iters, B, T, L, min_Es, reuse=0.1)
        grad_output = torch.randn(B, sum_Ds).cuda()

        for indices, offsets, _ in requests:
            output = cc(indices, offsets)
            output_ref = cc_ref(indices, offsets)
            torch.testing.assert_close(output, output_ref)
            output.backward(grad_output)
            output_ref.backward(grad_output)
        cc.flush()
        for t in range(T):
            torch.testing.assert_close(
                cc.split_embedding_weights()[t], cc_ref.split_embedding_weights()[t]
            )

    def _test_cache_prefetch_pipeline(  # noqa C901
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        mixed: bool,
        prefetch_location: str,
        prefetch_stream: Optional[torch.cuda.Stream],
    ) -> None:
        """
        test cache prefetch pipeline with prefetch_pipeline=True.
        prefetch_location can be "before_fwd" or "between_fwd_bwd",
        where the TBE prefetch(batch_{i+1}) is called before forward(batch_i)
        or in between of forward(batch_i) and backward(batch_i), respectively.
        If prefetch_stream is not None, the TBE prefetch function will use this stream.
        In addition, we make the TBE weights initialized as integer values, learning_rate
        as integer value, and gradients as integer values so that the test is more stable.
        """

        assert prefetch_location in ["before_fwd", "between_fwd_bwd"]
        cc, cc_ref, min_Es, sum_Ds = self._generate_cache_tbes(
            T, D, B, log_E, L, mixed, CacheAlgorithm.LRU, True, True
        )
        iters = 5
        requests = generate_requests(iters, B, T, L, min_Es, reuse=0.1)
        grad_output = (
            torch.randint(
                low=-10,
                high=10,
                size=(B, sum_Ds),
            )
            .float()
            .cuda()
        )
        torch.cuda.synchronize()  # make sure TBEs and inputs are ready
        self.assertTrue(torch.all(cc.lxu_cache_locking_counter == 0))

        cur_stream: torch.cuda.Stream = torch.cuda.current_stream()

        req_iter = iter(requests)
        batch_i = next(req_iter)
        batch_ip1 = None
        output, output_ref = None, None

        def _prefetch(
            cc: SplitTableBatchedEmbeddingBagsCodegen,
            batch: Optional[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]],
        ) -> None:
            if not batch:
                return
            context_stream = prefetch_stream if prefetch_stream else cur_stream
            stream = cur_stream if prefetch_stream else None
            indices, offsets, _ = batch
            with torch.cuda.stream(context_stream):
                cc.prefetch(indices, offsets, stream)

        _prefetch(cc, batch_i)
        while batch_i:
            indices, offsets, _ = batch_i
            batch_ip1 = next(req_iter, None)
            if prefetch_stream:
                cur_stream.wait_stream(prefetch_stream)
            if prefetch_location == "before_fwd":
                _prefetch(cc, batch_ip1)
            output = cc(indices, offsets)
            if prefetch_location == "between_fwd_bwd":
                _prefetch(cc, batch_ip1)
            output.backward(grad_output)
            batch_i = batch_ip1
            batch_ip1 = None
        cc.flush()

        for indices, offsets, _ in requests:
            output_ref = cc_ref(indices, offsets)
            output_ref.backward(grad_output)

        for t in range(T):
            torch.testing.assert_close(
                cc.split_embedding_weights()[t], cc_ref.split_embedding_weights()[t]
            )

        torch.testing.assert_close(output, output_ref)
        self.assertTrue(torch.all(cc.lxu_cache_locking_counter == 0))

    @optests.dontGenerateOpCheckTests("Serial OOM")
    @unittest.skipIf(*gpu_unavailable)
    @given(
        T=st.integers(min_value=1, max_value=5),
        D=st.integers(min_value=2, max_value=256),
        B=st.integers(min_value=1, max_value=128),
        log_E=st.integers(min_value=3, max_value=5),
        L=st.integers(min_value=1, max_value=20),
        mixed=st.booleans(),
        prefetch_location=st.sampled_from(["before_fwd", "between_fwd_bwd"]),
    )
    @settings(verbosity=VERBOSITY, max_examples=MAX_EXAMPLES, deadline=None)
    def test_cache_prefetch_pipeline(
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        mixed: bool,
        prefetch_location: str,
    ) -> None:
        self._test_cache_prefetch_pipeline(
            T,
            D,
            B,
            log_E,
            L,
            mixed,
            prefetch_location,
            prefetch_stream=None,
        )

    @optests.dontGenerateOpCheckTests("Serial OOM")
    @unittest.skipIf(*gpu_unavailable)
    @given(
        T=st.integers(min_value=1, max_value=5),
        D=st.integers(min_value=2, max_value=256),
        B=st.integers(min_value=1, max_value=128),
        log_E=st.integers(min_value=3, max_value=5),
        L=st.integers(min_value=1, max_value=20),
        mixed=st.booleans(),
    )
    @settings(verbosity=VERBOSITY, max_examples=MAX_EXAMPLES, deadline=None)
    def test_cache_prefetch_pipeline_stream_1(
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        mixed: bool,
    ) -> None:
        self._test_cache_prefetch_pipeline(
            T,
            D,
            B,
            log_E,
            L,
            mixed,
            prefetch_location="before_fwd",
            prefetch_stream=torch.cuda.Stream(),
        )

    @optests.dontGenerateOpCheckTests("Serial OOM")
    @unittest.skipIf(*gpu_unavailable)
    @given(
        T=st.integers(min_value=1, max_value=5),
        D=st.integers(min_value=2, max_value=256),
        B=st.integers(min_value=1, max_value=128),
        log_E=st.integers(min_value=3, max_value=5),
        L=st.integers(min_value=1, max_value=20),
        mixed=st.booleans(),
    )
    @settings(verbosity=VERBOSITY, max_examples=MAX_EXAMPLES, deadline=None)
    def test_cache_prefetch_pipeline_stream_2(
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        mixed: bool,
    ) -> None:
        self._test_cache_prefetch_pipeline(
            T,
            D,
            B,
            log_E,
            L,
            mixed,
            prefetch_location="between_fwd_bwd",
            prefetch_stream=torch.cuda.Stream(),
        )

    @unittest.skipIf(*gpu_unavailable)
    def test_split_embedding_codegen_forward(  # noqa C901
        self,
    ) -> None:
        # Dummy test in order to run generated opcheck tests on
        # split_embedding_codegen_forward_weighted_cuda and
        # split_embedding_codegen_forward_unweighted_cuda.
        # Sizes and values of int tensors were generated from running
        # one test instance of test_backward_adagrad_fp16_pmSUM and outputting
        # sizes/dtypes/values.
        def _do_test(weighted: bool) -> None:
            flatten_dev_weights = torch.rand(0, dtype=torch.half).cuda()
            uvm_weights = torch.rand(10456, dtype=torch.half).cuda()
            lxu_cache_weights = torch.rand(544, 4, dtype=torch.float).cuda()
            weights_placements = torch.tensor([2, 2, 2]).to(
                dtype=torch.int, device="cuda"
            )
            weights_offsets = torch.tensor([0, 2784, 2784]).to(
                dtype=torch.long, device="cuda"
            )
            D_offsets = torch.tensor([0, 4, 8]).to(dtype=torch.int, device="cuda")
            total_D = 12
            max_D = 4
            indices = torch.tensor(
                [
                    680,
                    213,
                    293,
                    439,
                    1004,
                    885,
                    986,
                    1162,
                    433,
                    1327,
                    187,
                    89,
                ]
            ).to(dtype=torch.long, device="cuda")
            offsets = torch.tensor([0, 2, 4, 6, 8, 10, 12]).to(
                dtype=torch.long, device="cuda"
            )
            pooling_mode = False
            indice_weights = torch.rand(12, dtype=torch.float).cuda()
            lxu_cache_locations = torch.tensor(
                [
                    224,
                    352,
                    353,
                    192,
                    384,
                    64,
                    288,
                    1,
                    96,
                    194,
                    0,
                    193,
                ]
            ).to(dtype=torch.int, device="cuda")
            uvm_cache_stats = torch.tensor(
                [
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                ]
            ).to(dtype=torch.int, device="cuda")

            output_dtype = 0  # SparseType.FP32
            is_experimental = False

            op_args = [
                flatten_dev_weights,
                uvm_weights,
                lxu_cache_weights,
                weights_placements,
                weights_offsets,
                D_offsets,
                total_D,
                max_D,
                indices,
                offsets,
                pooling_mode,
            ]
            if weighted:
                op_args += [indice_weights]
            op_args += [
                lxu_cache_locations,
                uvm_cache_stats,
                output_dtype,
                is_experimental,
            ]

            op_name = "split_embedding_codegen_forward"
            op_name += "_weighted" if weighted else "_unweighted"
            op_name += "_cuda"

            getattr(torch.ops.fbgemm, op_name)(*op_args)

        _do_test(True)
        _do_test(False)

    def execute_nbit_forward_(  # noqa C901
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        weighted: bool,
        mixed: bool,
        pooling_mode: PoolingMode,
        weights_ty: SparseType,
        use_cache: bool,
        cache_algorithm: CacheAlgorithm,
        use_cpu: bool,
        use_array_for_index_remapping: bool,
        do_pruning: bool,
        mixed_weights_ty: bool,
        output_dtype: SparseType,
    ) -> None:
        # NOTE: weighted operation can be done only for SUM.
        assume(pooling_mode == PoolingMode.SUM or not weighted)
        assume(not mixed or pooling_mode != PoolingMode.NONE)

        mode = "sum"
        do_pooling = True
        if pooling_mode == PoolingMode.SUM:
            mode = "sum"
        elif pooling_mode == PoolingMode.MEAN:
            mode = "mean"
        else:
            mode = "sum"
            do_pooling = False
        E = int(10**log_E)

        if not mixed_weights_ty:
            weights_ty_list = [weights_ty] * T
        else:
            weights_ty_list = [
                np.random.choice(
                    [
                        SparseType.FP32,
                        SparseType.FP16,
                        SparseType.FP8,
                        SparseType.INT8,
                        SparseType.INT4,
                        SparseType.INT2,
                    ]
                )
                for _ in range(T)
            ]

        D_alignment = max(
            1 if ty.bit_rate() % 8 == 0 else int(8 / ty.bit_rate())
            for ty in weights_ty_list
        )
        D = round_up(D, D_alignment)

        if not mixed:
            Ds = [D] * T
            Es = [E] * T
        else:
            Ds = [
                round_up(
                    np.random.randint(low=int(max(0.25 * D, 1)), high=int(1.0 * D)),
                    D_alignment,
                )
                for _ in range(T)
            ]
            Ds = [min(D, 128) for D in Ds]
            Es = [
                np.random.randint(low=int(0.5 * E), high=int(2.0 * E)) for _ in range(T)
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

        if use_cpu:
            managed = [EmbeddingLocation.HOST] * T
        elif use_cache:
            managed = [
                EmbeddingLocation.MANAGED_CACHING,
            ] * T
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

        # Fix exponent bias to 7 for now (TODO: Randomize it from a range of integers)
        if SparseType.FP8 in weights_ty_list:
            fp8_config = FP8QuantizationConfig(random.choice([4, 5]), 7)
            has_fp8_weight = True
        else:
            has_fp8_weight = False

        xs = [to_device(torch.randint(low=0, high=e, size=(B, L)), use_cpu) for e in Es]
        xws = [to_device(torch.randn(size=(B, L)), use_cpu) for _ in range(T)]

        xws_acc_type = copy.deepcopy(xws)

        if do_pruning:
            x = torch.cat([x.view(1, B, L) for x in xs], dim=0)
            xw = torch.cat([xw.view(1, B, L) for xw in xws_acc_type], dim=0)

            (indices, offsets) = get_table_batched_offsets_from_dense(
                x, use_cpu=use_cpu
            )

            # generate index_remapping
            dense_indices = torch.randint(low=0, high=E, size=(T, B, L)).view(-1).int()

            original_E = E
            current_device = "cpu" if use_cpu else torch.cuda.current_device()

            indices = indices.view(-1).int()
            offsets = offsets.view(-1).int()

            # generate index_remapping done
            # Initialize and insert Array index remapping based data structure
            index_remappings_array = []
            for t in range(T):
                indice_t = (indices.view(T, B, L))[t].long().view(-1).to(current_device)
                dense_indice_t = (
                    (dense_indices.view(T, B, L))[t].view(-1).to(current_device)
                )
                index_remappings_array_t = torch.tensor(
                    [-1] * original_E,
                    dtype=torch.int32,
                    device=current_device,
                )
                index_remappings_array_t[indice_t] = dense_indice_t
                index_remappings_array.append(index_remappings_array_t.cpu())
        else:
            index_remappings_array = [torch.arange(E, dtype=torch.int32) for E in Es]
            x = torch.cat([x.view(1, B, L) for x in xs], dim=0)
            xw = torch.cat([xw.view(1, B, L) for xw in xws_acc_type], dim=0)
            (indices, offsets) = get_table_batched_offsets_from_dense(
                x, use_cpu=use_cpu
            )

        cc = IntNBitTableBatchedEmbeddingBagsCodegen(
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
            pooling_mode=pooling_mode,
            index_remapping=index_remappings_array if B != 0 else None,
            device="cpu" if use_cpu else torch.cuda.current_device(),
            cache_algorithm=cache_algorithm,
            use_array_for_index_remapping=use_array_for_index_remapping,
            output_dtype=output_dtype,
            fp8_exponent_bits=fp8_config.get("exponent_bits")
            if has_fp8_weight
            else None,
            fp8_exponent_bias=fp8_config.get("exponent_bias")
            if has_fp8_weight
            else None,
        )
        # Initialize the random weights for int nbit table split embedding bag
        cc.fill_random_weights()
        # NOTE: test TorchScript-compatible!
        cc = torch.jit.script(cc)

        for t in range(T):
            (weights, scale_shift) = cc.split_embedding_weights()[t]
            if scale_shift is not None:
                (E, R) = scale_shift.shape
                self.assertEqual(R, 4)
                if weights_ty_list[t] == SparseType.INT2:
                    scales = np.random.uniform(0.1, 1, size=(E,)).astype(np.float16)
                    shifts = np.random.uniform(-2, 2, size=(E,)).astype(np.float16)
                if weights_ty_list[t] == SparseType.INT4:
                    scales = np.random.uniform(0.01, 0.1, size=(E,)).astype(np.float16)
                    shifts = np.random.uniform(-2, 2, size=(E,)).astype(np.float16)
                if weights_ty_list[t] == SparseType.INT8:
                    scales = np.random.uniform(0.001, 0.01, size=(E,)).astype(
                        np.float16
                    )
                    shifts = np.random.uniform(-2, 2, size=(E,)).astype(np.float16)

                scale_shift[:, :] = torch.tensor(
                    # pyre-fixme[61]: `scales` is undefined, or not always defined.
                    # pyre-fixme[61]: `shifts` is undefined, or not always defined.
                    np.stack([scales, shifts], axis=1)
                    .astype(np.float16)
                    .view(np.uint8)
                )

            fake_quantize_embs(
                weights,
                scale_shift,
                bs[t].weight.detach(),
                weights_ty_list[t],
                use_cpu=False,
                # pyre-fixme[61]: `fp8_config` is undefined, or not always defined.
                fp8_config=fp8_config if has_fp8_weight else None,
            )

        if not use_cpu:
            fc2 = (
                cc(indices.int(), offsets.int())
                if not weighted
                else cc(indices.int(), offsets.int(), xw.contiguous().view(-1))
            )
        else:
            cc = cc.cpu()
            indices, offsets = indices.cpu(), offsets.cpu()
            fc2 = (
                cc(indices.int(), offsets.int())
                if not weighted
                else cc(indices.int(), offsets.int(), xw.contiguous().view(-1).cpu())
            )

        if do_pooling and B == 0:
            self.assertEqual(fc2.size(), (0, cc.total_D))
            return

        new_indices = []
        for t in range(T):
            new_indices_t = torch.zeros([B, L], dtype=torch.int32)
            for i in range(B):
                for j in range(L):
                    old_index = xs[t][i, j]
                    new_index = index_remappings_array[t][old_index]
                    new_indices_t[i][j] = new_index
            new_indices.append(new_indices_t)

        fs = (
            [
                b_indices(b, x, use_cpu=use_cpu, do_pooling=do_pooling)
                for (b, x) in zip(bs, new_indices)
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
                for (b, x, xw) in zip(bs, new_indices, xws)
            ]
        )
        if do_pooling:
            f = torch.cat([f.view(B, -1) for f in fs], dim=1)
        else:
            f = torch.cat(fs, dim=0).view(-1, D)
        torch.testing.assert_close(
            fc2.float().cpu(),
            f.float().cpu(),
            atol=1.0e-2,
            rtol=1.0e-2,
        )

    @given(
        nbit_weights_ty=get_nbit_weights_ty(),
        use_array_for_index_remapping=st.booleans(),
        do_pruning=st.booleans(),
        pooling_mode=st.sampled_from(
            [PoolingMode.SUM, PoolingMode.NONE, PoolingMode.MEAN]
        ),
        output_dtype=st.sampled_from(
            [SparseType.FP32, SparseType.FP16, SparseType.BF16]
        ),
    )
    @settings(
        verbosity=VERBOSITY,
        max_examples=MAX_EXAMPLES_LONG_RUNNING,
        deadline=None,
    )
    def test_nbit_forward_cpu(
        self,
        nbit_weights_ty: Optional[SparseType],
        use_array_for_index_remapping: bool,
        do_pruning: bool,
        pooling_mode: PoolingMode,
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
            use_array_for_index_remapping,
            do_pruning,
            mixed_weights_ty,
            output_dtype,
        )

    @unittest.skipIf(*gpu_unavailable)
    def test_nbit_forward_gpu_no_cache_fp8_2048(self) -> None:
        # Test the case of FB8 table with 128B*8 < D <= 128B*16
        self.execute_nbit_forward_(
            T=1,
            D=2048,  # 128B*8 < D <= 128B*16
            B=128,
            log_E=2,
            L=4,
            weighted=False,
            mixed=False,
            pooling_mode=PoolingMode.SUM,
            weights_ty=SparseType.FP8,  # FP8 table
            use_cache=False,
            cache_algorithm=CacheAlgorithm.LRU,
            use_cpu=False,
            use_array_for_index_remapping=True,
            do_pruning=False,
            mixed_weights_ty=False,
            output_dtype=SparseType.FP16,
        )

    @unittest.skipIf(*gpu_unavailable)
    @given(
        nbit_weights_ty=get_nbit_weights_ty(),
        use_array_for_index_remapping=st.booleans(),
        do_pruning=st.booleans(),
    )
    @settings(
        verbosity=VERBOSITY,
        max_examples=MAX_EXAMPLES_LONG_RUNNING,
        deadline=None,
    )
    def test_nbit_forward_gpu_no_cache(
        self,
        nbit_weights_ty: Optional[SparseType],
        use_array_for_index_remapping: bool,
        do_pruning: bool,
    ) -> None:
        use_cpu = False
        T = random.randint(1, 50)
        B = random.randint(0, 128)
        L = random.randint(0, 32)
        D = random.randint(2, 2048)
        log_E = random.randint(2, 4)

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
        output_dtype = random.choice(
            [SparseType.FP32, SparseType.FP16, SparseType.BF16]
        )
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
            use_array_for_index_remapping,
            do_pruning,
            mixed_weights_ty,
            output_dtype,
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
        cache_algorithm=st.sampled_from(CacheAlgorithm),
        associativity=st.sampled_from([1, DEFAULT_ASSOC]),
        do_pruning=st.booleans(),
        use_array_for_index_remapping=st.booleans(),
    )
    @settings(verbosity=VERBOSITY, max_examples=MAX_EXAMPLES, deadline=None)
    def test_nbit_forward_uvm_cache(
        self,
        weights_ty: SparseType,
        cache_algorithm: CacheAlgorithm,
        associativity: int,
        do_pruning: bool,
        use_array_for_index_remapping: bool,
    ) -> None:
        assume(cache_algorithm == CacheAlgorithm.LRU or associativity != 1)

        T = random.randint(1, 5)
        B = random.randint(1, 128)
        L = random.randint(1, 20)
        D = random.randint(2, 256)
        log_E = random.randint(3, 5)
        mixed = random.choice([True, False])

        iters = 3
        E = int(10**log_E)

        D_alignment = (
            1 if weights_ty.bit_rate() % 8 == 0 else int(8 / weights_ty.bit_rate())
        )
        D = round_up(D, D_alignment)

        if not mixed:
            Ds = [D] * T
            Es = [E] * T
        else:
            Ds = [
                round_up(
                    np.random.randint(low=int(max(0.25 * D, 1)), high=int(1.0 * D)),
                    D_alignment,
                )
                for _ in range(T)
            ]
            Es = [
                np.random.randint(low=int(0.5 * E), high=int(2.0 * E)) for _ in range(T)
            ]
        managed = [EmbeddingLocation.MANAGED_CACHING] * T
        if mixed:
            average_D = sum(Ds) // T
            for t, d in enumerate(Ds):
                managed[t] = EmbeddingLocation.DEVICE if d < average_D else managed[t]
        index_remapping = None
        pruning_hash_load_factor = 0.5
        if do_pruning:
            current_device = torch.cuda.current_device()
            index_remapping = []
            for E in Es:
                # For each table, keep the first half of rows as is, but
                # the rest is treated as pruned (-1).
                remapping = list(range(0, E // 2)) + [-1] * (E - E // 2)
                remapping_t = torch.tensor(
                    remapping,
                    dtype=torch.int32,
                    device=current_device,
                )
                index_remapping.append(remapping_t)
        cc_ref = IntNBitTableBatchedEmbeddingBagsCodegen(
            [
                (
                    "",
                    E,
                    D,
                    weights_ty,
                    EmbeddingLocation.DEVICE,
                )
                for (E, D) in zip(Es, Ds)
            ],
            index_remapping=index_remapping,
            use_array_for_index_remapping=use_array_for_index_remapping,
            pruning_hash_load_factor=pruning_hash_load_factor,
        )
        cc_ref.fill_random_weights()
        cc = IntNBitTableBatchedEmbeddingBagsCodegen(
            [("", E, D, weights_ty, M) for (E, D, M) in zip(Es, Ds, managed)],
            cache_algorithm=cache_algorithm,
            cache_assoc=associativity,
            index_remapping=index_remapping,
            use_array_for_index_remapping=use_array_for_index_remapping,
            pruning_hash_load_factor=pruning_hash_load_factor,
        )
        cc.fill_random_weights()

        split_weights = cc.split_embedding_weights()
        ref_split_weights = cc_ref.split_embedding_weights()
        for t in range(T):
            (weights, scale_shift) = split_weights[t]
            (ref_weights, ref_scale_shift) = ref_split_weights[t]
            self.assertEqual(weights.size(), ref_weights.size())
            weights.copy_(ref_weights)
            if ref_scale_shift is not None:
                scale_shift.copy_(ref_scale_shift)

        requests = generate_requests(iters, B, T, L, min(Es), reuse=0.1)

        for indices, offsets, _ in requests:
            indices = indices.int()
            offsets = offsets.int()
            output = cc(indices, offsets)
            output_ref = cc_ref(indices, offsets)
            torch.testing.assert_close(output, output_ref, equal_nan=True)

    @given(
        T=st.integers(min_value=1, max_value=5),
        B=st.integers(min_value=1, max_value=8),
        L=st.integers(min_value=0, max_value=8),
        use_cpu=st.booleans()
        if (gpu_available and not TEST_WITH_ROCM)
        else st.just(False)
        if (gpu_available and TEST_WITH_ROCM)
        else st.just(True),
        use_cpu_hashtable=st.booleans(),
        use_array_for_index_remapping=st.booleans(),
    )
    @settings(verbosity=VERBOSITY, max_examples=MAX_EXAMPLES, deadline=None)
    def test_pruning(
        self,
        T: int,
        B: int,
        L: int,
        use_cpu: bool,
        use_cpu_hashtable: bool,
        use_array_for_index_remapping: bool,
    ) -> None:
        E = int(1000)
        LOAD_FACTOR = 0.8
        pruning_ratio = 0.5

        capacities = [int(B * L / LOAD_FACTOR) + 1 for _ in range(T)]
        original_E = int(E / (1.0 - pruning_ratio))

        # Enforce the size of original_E/B/L to get the unique indices
        assume(original_E > B * L)

        current_device = "cpu" if use_cpu else torch.cuda.current_device()

        if use_cpu_hashtable:
            assume(use_cpu)

        indices = torch.randint(low=0, high=original_E, size=(T, B, L))
        for t in range(T):
            while (
                torch.unique(
                    indices[t], return_counts=False, return_inverse=False
                ).numel()
                != indices[t].numel()
            ):
                indices[t] = torch.randint(low=0, high=original_E, size=(B, L))

        indices = indices.view(-1).int()
        dense_indices = torch.randint(low=0, high=E, size=(T, B, L)).view(-1).int()
        offsets = torch.tensor([L * b_t for b_t in range(B * T + 1)]).int()

        # Initialize and insert Hashmap index remapping based data structure
        hash_table = torch.empty(
            (sum(capacities), 2),
            dtype=torch.int32,
        )
        hash_table[:, :] = -1
        hash_table_offsets = torch.tensor([0] + np.cumsum(capacities).tolist()).long()

        torch.ops.fbgemm.pruned_hashmap_insert(
            indices, dense_indices, offsets, hash_table, hash_table_offsets
        )

        if use_cpu_hashtable:
            ht = torch.classes.fbgemm.PrunedMapCPU()
            ht.insert(indices, dense_indices, offsets, T)

        # Initialize and insert Array index remapping based data structure
        index_remappings_array = torch.tensor(
            [-1] * original_E * T,
            dtype=torch.int32,
            device=current_device,
        )
        index_remappings_array_offsets = torch.empty(
            T + 1,
            dtype=torch.int64,
            device=current_device,
        )
        index_remappings_array_offsets[0] = 0
        for t in range(T):
            indice_t = (indices.view(T, B, L))[t].long().view(-1).to(current_device)
            dense_indice_t = (
                (dense_indices.view(T, B, L))[t].view(-1).to(current_device)
            )
            selected_indices = torch.add(indice_t, t * original_E)[:E]
            index_remappings_array[selected_indices] = dense_indice_t
            index_remappings_array_offsets[t + 1] = (
                index_remappings_array_offsets[t] + original_E
            )

        # Move data when using device
        if not use_cpu:
            (
                indices,
                dense_indices,
                offsets,
                hash_table,
                hash_table_offsets,
                index_remappings_array,
                index_remappings_array_offsets,
            ) = (
                indices.to(current_device),
                dense_indices.to(current_device),
                offsets.to(current_device),
                hash_table.to(current_device),
                hash_table_offsets.to(current_device),
                index_remappings_array.to(current_device),
                index_remappings_array_offsets.to(current_device),
            )

        # Lookup
        if use_cpu_hashtable:
            dense_indices_ = ht.lookup(indices, offsets)
        elif not use_array_for_index_remapping:  # hashmap based pruning
            dense_indices_ = torch.ops.fbgemm.pruned_hashmap_lookup(
                indices, offsets, hash_table, hash_table_offsets
            )
        else:  # array based pruning
            dense_indices_ = torch.ops.fbgemm.pruned_array_lookup(
                indices,
                offsets,
                index_remappings_array,
                index_remappings_array_offsets,
            )

        # Validate the lookup result
        torch.testing.assert_close(dense_indices, dense_indices_)

        # For array based pruning, it will be out-of-boundary for arbitrarily
        # large indices. We will rely on bound checker to make sure indices
        # are within the boundary.
        if not use_array_for_index_remapping:
            # now, use a value that does not exist in the original set of indices
            # and so should be pruned out.
            indices[:] = np.iinfo(np.int32).max

            if use_cpu_hashtable:
                dense_indices_ = ht.lookup(indices, offsets)
            elif not use_array_for_index_remapping:  # hashmap based pruning
                dense_indices_ = torch.ops.fbgemm.pruned_hashmap_lookup(
                    indices, offsets, hash_table, hash_table_offsets
                )
            else:  # array based pruning
                dense_indices_ = torch.ops.fbgemm.pruned_array_lookup(
                    indices,
                    offsets,
                    index_remappings_array,
                    index_remappings_array_offsets,
                )
            torch.testing.assert_close(dense_indices.clone().fill_(-1), dense_indices_)

    @given(
        L=st.integers(min_value=0, max_value=16),
        H=st.integers(min_value=512, max_value=1024),
        S=st.integers(min_value=0, max_value=128),
    )
    @settings(verbosity=VERBOSITY, max_examples=MAX_EXAMPLES, deadline=None)
    def test_cache_update_function(self, L: int, H: int, S: int) -> None:
        # Generate synthetic data
        linear_cache_indices_cpu = torch.randint(L, H, (S,))
        lxu_cache_locations_cpu = torch.clone(linear_cache_indices_cpu)

        indices = [True if np.random.rand() < 0.5 else False for _ in range(S)]
        lxu_cache_locations_cpu[indices] = -1

        cache_miss_ids = torch.clone(linear_cache_indices_cpu)
        cache_miss_ids[lxu_cache_locations_cpu != -1] = -2

        # Calculate the correct output
        unique_cache_miss_ids = torch.unique(cache_miss_ids)
        expect_out = sum(unique_cache_miss_ids >= 0)
        linear_cache_indices = to_device(
            torch.tensor(linear_cache_indices_cpu, dtype=torch.int64), use_cpu=False
        )
        lxu_cache_locations = to_device(
            torch.tensor(lxu_cache_locations_cpu, dtype=torch.int32), use_cpu=False
        )

        # Create an abstract split table
        D = 8
        T = 2
        E = 10**3
        Ds = [D] * T
        Es = [E] * T
        emb_op = SplitTableBatchedEmbeddingBagsCodegen
        cc = emb_op(
            embedding_specs=[
                (
                    E,
                    D,
                    EmbeddingLocation.MANAGED_CACHING,
                    ComputeDevice.CUDA,
                )
                for (E, D) in zip(Es, Ds)
            ],
            record_cache_metrics=RecordCacheMetrics(True, False),
        )
        cc._update_cache_miss_counter(lxu_cache_locations, linear_cache_indices)
        (
            cache_miss_forward_count,
            unique_cache_miss_count,
        ) = cc.get_cache_miss_counter().cpu()

        self.assertEqual(unique_cache_miss_count, expect_out)
        self.assertLessEqual(cache_miss_forward_count, unique_cache_miss_count)

    @given(N=st.integers(min_value=1, max_value=8))
    @settings(verbosity=VERBOSITY, max_examples=MAX_EXAMPLES, deadline=None)
    def test_cache_miss_counter(self, N: int) -> None:
        # Create an abstract split table
        D = 8
        T = 2
        E = 10**3
        Ds = [D] * T
        Es = [E] * T
        emb_op = SplitTableBatchedEmbeddingBagsCodegen
        cc = emb_op(
            embedding_specs=[
                (
                    E,
                    D,
                    EmbeddingLocation.MANAGED_CACHING,
                    ComputeDevice.CUDA,
                )
                for (E, D) in zip(Es, Ds)
            ],
            record_cache_metrics=RecordCacheMetrics(True, True),
        )

        # Create fake input data and the target output
        xs = []
        x1 = torch.Tensor([[[1], [1]], [[3], [4]]])
        x1 = to_device(torch.tensor(x1, dtype=torch.int64), use_cpu=False)

        x2 = torch.Tensor([[[2], [1]], [[3], [4]]])
        x2 = to_device(torch.tensor(x2, dtype=torch.int64), use_cpu=False)

        x3 = torch.Tensor([[[5], [6]], [[7], [8]]])
        x3 = to_device(torch.tensor(x3, dtype=torch.int64), use_cpu=False)

        xs.append(x1)
        xs.append(x2)
        xs.append(x3)

        target_counter_list = [[1, 3], [2, 4], [3, 8]]
        target_tablewise_cache_miss_list = [[1, 2], [2, 2], [4, 4]]
        for x, t_counter, t_tablewise_cache_miss in zip(
            xs, target_counter_list, target_tablewise_cache_miss_list
        ):
            (indices, offsets) = get_table_batched_offsets_from_dense(x, use_cpu=False)
            for _ in range(N):
                cc(indices, offsets)
                (
                    cache_miss_forward_count,
                    unique_cache_miss_count,
                ) = cc.get_cache_miss_counter().cpu()
                tablewise_cache_miss = cc.get_table_wise_cache_miss().cpu()
                self.assertEqual(cache_miss_forward_count, t_counter[0])
                self.assertEqual(unique_cache_miss_count, t_counter[1])
                for i in range(len(tablewise_cache_miss)):
                    self.assertEqual(tablewise_cache_miss[i], t_tablewise_cache_miss[i])

    @given(N=st.integers(min_value=1, max_value=2))
    @settings(verbosity=VERBOSITY, max_examples=MAX_EXAMPLES, deadline=None)
    def test_stb_uvm_cache_stats(self, N: int) -> None:
        # Create an abstract split table
        D = 8
        T = 2
        E = 10**3
        Ds = [D] * T
        Es = [E] * T
        emb_op = SplitTableBatchedEmbeddingBagsCodegen
        cc = emb_op(
            embedding_specs=[
                (
                    E,
                    D,
                    EmbeddingLocation.MANAGED_CACHING,
                    ComputeDevice.CUDA,
                )
                for (E, D) in zip(Es, Ds)
            ],
            gather_uvm_cache_stats=True,
        )

        x = torch.Tensor([[[1], [1]], [[3], [4]]])
        x = to_device(torch.tensor(x, dtype=torch.int64), use_cpu=False)

        for _ in range(N):
            indices, offsets = get_table_batched_offsets_from_dense(x, use_cpu=False)
            cc.reset_cache_states()
            cc.reset_uvm_cache_stats()
            cc(indices, offsets)
            (
                n_calls,
                n_requested_indices,
                n_unique_indices,
                n_unique_misses,
                n_conflict_unique_misses,
                n_conflict_misses,
            ) = cc.get_uvm_cache_stats()
            self.assertEqual(n_calls, 1)
            self.assertEqual(n_requested_indices, len(indices))
            self.assertEqual(n_unique_indices, len(set(indices.tolist())))
            self.assertEqual(n_unique_misses, len(set(indices.tolist())))
            self.assertEqual(n_conflict_unique_misses, 0)
            self.assertEqual(n_conflict_misses, 0)

    @unittest.skipIf(*gpu_unavailable)
    @given(
        L=st.integers(min_value=0, max_value=16),
        H=st.integers(min_value=512, max_value=1024),
        S=st.integers(min_value=0, max_value=128),
    )
    @settings(verbosity=VERBOSITY, max_examples=MAX_EXAMPLES, deadline=None)
    def test_nbit_cache_update_function(self, L: int, H: int, S: int) -> None:
        # Generate synthetic data
        linear_cache_indices_cpu = torch.randint(L, H, (S,))
        lxu_cache_locations_cpu = torch.clone(linear_cache_indices_cpu)

        indices = [True if np.random.rand() < 0.5 else False for _ in range(S)]
        lxu_cache_locations_cpu[indices] = -1

        cache_miss_ids = torch.clone(linear_cache_indices_cpu)
        cache_miss_ids[lxu_cache_locations_cpu != -1] = -2

        # Calculate the correct output
        unique_cache_miss_ids = torch.unique(cache_miss_ids)
        expect_out = sum(unique_cache_miss_ids >= 0)
        linear_cache_indices = linear_cache_indices_cpu.to(torch.int32).cuda()
        lxu_cache_locations = lxu_cache_locations_cpu.to(torch.int32).cuda()
        expected_unique_access = len(torch.unique(linear_cache_indices_cpu))
        expected_total_access = len(linear_cache_indices_cpu)

        # Create an abstract split table
        D = 8
        T = 2
        E = 10**3
        Ds = [D] * T
        Es = [E] * T
        cc = IntNBitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                (
                    "",
                    E,
                    D,
                    SparseType.INT8,
                    EmbeddingLocation.MANAGED_CACHING,
                )
                for (E, D) in zip(Es, Ds)
            ],
            device=torch.cuda.current_device(),
            record_cache_metrics=RecordCacheMetrics(True, False),
        )
        cc.fill_random_weights()

        cc._update_cache_miss_counter(lxu_cache_locations, linear_cache_indices)
        (
            cache_miss_forward_count,
            unique_cache_miss_count,
            unique_access_count,
            total_access_count,
        ) = cc.get_cache_miss_counter().cpu()

        self.assertEqual(unique_cache_miss_count, expect_out)
        self.assertLessEqual(cache_miss_forward_count, unique_cache_miss_count)
        self.assertEqual(unique_access_count, expected_unique_access)
        self.assertEqual(total_access_count, expected_total_access)

    @unittest.skipIf(*gpu_unavailable)
    @given(N=st.integers(min_value=1, max_value=8))
    @settings(verbosity=VERBOSITY, max_examples=MAX_EXAMPLES, deadline=None)
    def test_nbit_cache_miss_counter(self, N: int) -> None:
        # Create an abstract split table
        D = 8
        T = 2
        E = 10**3
        Ds = [D] * T
        Es = [E] * T
        cc = IntNBitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                (
                    "",
                    E,
                    D,
                    SparseType.INT8,
                    EmbeddingLocation.MANAGED_CACHING,
                )
                for (E, D) in zip(Es, Ds)
            ],
            device=torch.cuda.current_device(),
            record_cache_metrics=RecordCacheMetrics(True, True),
        )
        cc.fill_random_weights()

        # Create fake input data and the target output
        x1 = torch.Tensor([[[1], [1]], [[3], [4]]]).cuda()
        x2 = torch.Tensor([[[2], [1]], [[3], [4]]]).cuda()
        x3 = torch.Tensor([[[5], [6]], [[7], [8]]]).cuda()

        xs = [x1, x2, x3]
        target_counter_list = [[1, 3], [2, 4], [3, 8]]
        target_tablewise_cache_miss_list = [[1, 2], [2, 2], [4, 4]]
        for x, t_counter, t_tablewise_cache_miss in zip(
            xs, target_counter_list, target_tablewise_cache_miss_list
        ):
            (indices, offsets) = get_table_batched_offsets_from_dense(x, use_cpu=False)
            for _ in range(N):
                cc(indices.int(), offsets.int())
                (
                    cache_miss_forward_count,
                    unique_cache_miss_count,
                    _,
                    _,
                ) = cc.get_cache_miss_counter().cpu()
                tablewise_cache_miss = cc.get_table_wise_cache_miss().cpu()
                self.assertEqual(cache_miss_forward_count, t_counter[0])
                self.assertEqual(unique_cache_miss_count, t_counter[1])
                for i in range(len(tablewise_cache_miss)):
                    self.assertEqual(tablewise_cache_miss[i], t_tablewise_cache_miss[i])

    @unittest.skipIf(*gpu_unavailable)
    @given(
        N=st.integers(min_value=1, max_value=8),
        dtype=st.sampled_from([SparseType.INT8, SparseType.INT4, SparseType.INT2]),
    )
    @settings(verbosity=VERBOSITY, max_examples=MAX_EXAMPLES, deadline=None)
    def test_nbit_uvm_cache_stats(self, N: int, dtype: SparseType) -> None:
        # Create an abstract split table
        D = 8
        T = 2
        E = 10**3
        Ds = [D] * T
        Es = [E] * T
        cc = IntNBitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                (
                    "",
                    E,
                    D,
                    dtype,
                    EmbeddingLocation.MANAGED_CACHING,
                )
                for (E, D) in zip(Es, Ds)
            ],
            device=torch.cuda.current_device(),
            gather_uvm_cache_stats=True,
        )
        cc.fill_random_weights()

        # Create fake input data and the target output
        x1 = torch.Tensor([[[1], [1]], [[3], [4]]]).cuda()
        x2 = torch.Tensor([[[2], [1]], [[3], [4]]]).cuda()
        x3 = torch.Tensor([[[5], [6]], [[7], [8]]]).cuda()

        xs = [x1, x2, x3]
        # num_unique_indices, num_unique_misses
        # note that these are cumulative over calls; and also "unique" is per batch.
        target_counter_list = [[3, 3], [4, 4], [4, 8]]
        num_calls_expected = 0
        num_indices_expcted = 0
        num_unique_indices_expected = 0
        for x, t_counter in zip(xs, target_counter_list):
            (indices, offsets) = get_table_batched_offsets_from_dense(x, use_cpu=False)
            for _ in range(N):
                num_calls_expected = num_calls_expected + 1
                num_indices_expcted = num_indices_expcted + len(indices)
                cc(indices.int(), offsets.int())
                (
                    num_calls,
                    num_indices,
                    num_unique_indices,
                    num_unique_misses,
                    num_conflict_unique_miss,
                    num_conflict_miss,
                ) = cc.get_uvm_cache_stats().cpu()
                # Note num_unique_indices is cumulative stats.
                num_unique_indices_expected = num_unique_indices_expected + t_counter[0]
                self.assertEqual(num_calls, num_calls_expected)
                self.assertEqual(num_indices, num_indices_expcted)
                self.assertEqual(num_unique_indices, num_unique_indices_expected)
                self.assertEqual(num_unique_misses, t_counter[1])
                self.assertEqual(num_conflict_unique_miss, 0)
                self.assertEqual(num_conflict_miss, 0)

        T = 1  # for simplicity
        Ds = [D] * T
        Es = [E] * T
        cc1 = IntNBitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                (
                    "",
                    E,
                    D,
                    SparseType.INT8,
                    EmbeddingLocation.MANAGED_CACHING,
                )
                for (E, D) in zip(Es, Ds)
            ],
            device=torch.cuda.current_device(),
            gather_uvm_cache_stats=True,
            cache_sets=1,  # Only one set.
        )
        cc1.fill_random_weights()

        associativty = DEFAULT_ASSOC  # 32 for NVidia / 64 for AMD.
        repetition = 17
        indices1 = torch.Tensor(
            [[list(range(0, associativty))] * repetition]
        ).cuda()  # 0, 1, ..., 31.
        indices2 = torch.Tensor(
            [[list(range(0, associativty + 1))] * repetition]
        ).cuda()  # 0, 1, ..., 31, 32.
        indices3 = torch.Tensor(
            [[list(range(0, associativty + 10))] * repetition]
        ).cuda()  # 0, 1, ..., 31, 32, ..., 41.

        # num_conflict_unique_miss, num_conflict_miss
        expected = [[0, 0], [1, 17], [10, 170]]

        for x, e in zip((indices1, indices2, indices3), expected):
            (indices, offsets) = get_table_batched_offsets_from_dense(x, use_cpu=False)
            for _ in range(N):
                cc1(indices.int(), offsets.int())
                (
                    _,
                    _,
                    _,
                    _,
                    num_conflict_unique_miss,
                    num_conflict_miss,
                ) = cc1.get_uvm_cache_stats().cpu()
                self.assertEqual(num_conflict_unique_miss, e[0])
                self.assertEqual(num_conflict_miss, e[1])
                cc1.reset_uvm_cache_stats()

    @unittest.skipIf(*gpu_unavailable)
    @given(
        N=st.integers(min_value=1, max_value=8),
        dtype=st.sampled_from([SparseType.INT8, SparseType.INT4, SparseType.INT2]),
    )
    @settings(verbosity=VERBOSITY, max_examples=MAX_EXAMPLES, deadline=None)
    def test_nbit_direct_mapped_uvm_cache_stats(
        self, N: int, dtype: SparseType
    ) -> None:
        # Create an abstract split table
        D = 8
        T = 2
        E = 10**3
        Ds = [D] * T
        Es = [E] * T
        cc = IntNBitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                (
                    "",
                    E,
                    D,
                    dtype,
                    EmbeddingLocation.MANAGED_CACHING,
                )
                for (E, D) in zip(Es, Ds)
            ],
            device=torch.cuda.current_device(),
            gather_uvm_cache_stats=True,
            cache_assoc=1,  # Direct Mapped
        )
        cc.fill_random_weights()

        # Create fake input data and the target output
        x1 = torch.Tensor([[[1], [1]], [[3], [4]]]).cuda()
        x2 = torch.Tensor([[[2], [1]], [[3], [4]]]).cuda()
        x3 = torch.Tensor([[[5], [6]], [[7], [8]]]).cuda()

        xs = [x1, x2, x3]
        # num_unique_indices, num_unique_misses
        # note that these are cumulative over calls; and also "unique" is per batch.
        target_counter_list = [[3, 3], [4, 4], [4, 8]]
        num_calls_expected = 0
        num_indices_expcted = 0
        num_unique_indices_expected = 0
        for x, t_counter in zip(xs, target_counter_list):
            (indices, offsets) = get_table_batched_offsets_from_dense(x, use_cpu=False)
            for _ in range(N):
                num_calls_expected = num_calls_expected + 1
                num_indices_expcted = num_indices_expcted + len(indices)
                cc(indices.int(), offsets.int())
                (
                    num_calls,
                    num_indices,
                    num_unique_indices,
                    num_unique_misses,
                    num_conflict_unique_miss,
                    num_conflict_miss,
                ) = cc.get_uvm_cache_stats().cpu()
                # Note num_unique_indices is cumulative stats.
                num_unique_indices_expected = num_unique_indices_expected + t_counter[0]
                self.assertEqual(num_calls, num_calls_expected)
                self.assertEqual(num_indices, num_indices_expcted)
                self.assertEqual(num_unique_indices, 0)  # N/A for Direct Mapped
                self.assertEqual(num_unique_misses, 0)  # N/A for Direct Mapped
                self.assertEqual(
                    num_conflict_unique_miss, t_counter[1]
                )  # number of actually inserted rows for Direct Mapped
                self.assertEqual(num_conflict_miss, 0)

        T = 1  # for simplicity
        Ds = [D] * T
        Es = [E] * T
        cc1 = IntNBitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                (
                    "",
                    E,
                    D,
                    SparseType.INT8,
                    EmbeddingLocation.MANAGED_CACHING,
                )
                for (E, D) in zip(Es, Ds)
            ],
            device=torch.cuda.current_device(),
            gather_uvm_cache_stats=True,
            cache_sets=1,  # Only one set.
            cache_assoc=1,  # Direct Mapped
        )
        cc1.fill_random_weights()

        associativty = 1  # Direct-Mapped
        repetition = 17
        indices1 = torch.Tensor(
            [[list(range(0, associativty))] * repetition]
        ).cuda()  # no conflict miss
        indices2 = torch.Tensor(
            [[list(range(0, associativty + 1))] * repetition]
        ).cuda()  # 1 * 17 conflict miss per request
        indices3 = torch.Tensor(
            [[list(range(0, associativty + 10))] * repetition]
        ).cuda()  # 10 * 17 conflict misses per request

        # num_conflict_unique_miss, num_conflict_miss
        expected = [[1, 0], [1, 17], [1, 170]]

        accum_num_conflict_miss = 0
        for x, e in zip((indices1, indices2, indices3), expected):
            (indices, offsets) = get_table_batched_offsets_from_dense(x, use_cpu=False)
            for _ in range(N):
                cc1(indices.int(), offsets.int())
                (
                    _,
                    _,
                    _,
                    _,
                    num_conflict_unique_miss,
                    num_conflict_miss,
                ) = cc1.get_uvm_cache_stats().cpu()
                # for DM this represents number of actually inserted rows
                self.assertEqual(num_conflict_unique_miss, e[0])
                accum_num_conflict_miss += e[1]
                self.assertEqual(num_conflict_miss, accum_num_conflict_miss)

    @given(
        T=st.integers(min_value=1, max_value=64),
        B=st.integers(min_value=1, max_value=64),
        max_L=st.integers(min_value=1, max_value=64),
        bounds_check_mode=st.sampled_from(
            [
                BoundsCheckMode.FATAL,
                BoundsCheckMode.WARNING,
                BoundsCheckMode.IGNORE,
            ]
        ),
        use_cpu=st.booleans()
        if (gpu_available and not TEST_WITH_ROCM)
        else st.just(False)
        if (gpu_available and TEST_WITH_ROCM)
        else st.just(True),
        weighted=st.booleans(),
        dtype=st.sampled_from(
            [
                torch.int64,
                torch.int32,
            ]
        ),
        mixed_B=st.booleans(),
    )
    @settings(verbosity=VERBOSITY, max_examples=MAX_EXAMPLES, deadline=None)
    def test_bounds_check(  # noqa C901
        self,
        T: int,
        B: int,
        max_L: int,
        bounds_check_mode: BoundsCheckMode,
        use_cpu: bool,
        weighted: bool,
        dtype: torch.dtype,
        mixed_B: bool,
    ) -> None:
        # use_cpu does not support mixed_B
        if use_cpu and mixed_B:
            mixed_B = False
        rows_per_table = torch.tensor(
            np.random.randint(low=1, high=1000, size=(T,))
        ).long()
        if not mixed_B:
            Bs = [B] * T
        else:
            low = max(int(0.25 * B), 1)
            high = int(B)
            if low == high:
                Bs = [B] * T
            else:
                Bs = [np.random.randint(low=low, high=high) for _ in range(T)]
        B_offsets = [0] + list(accumulate(Bs))
        Ls = np.random.randint(low=0, high=max_L, size=(B_offsets[-1],))
        indices = [
            np.random.randint(
                low=0,
                high=rows_per_table[t],
                size=sum(Ls[B_offsets[t] : B_offsets[t + 1]]),
            )
            for t in range(T)
        ]
        indices = torch.tensor(np.concatenate(indices, axis=0)).to(dtype)
        weights = (
            torch.rand(indices.shape, dtype=torch.float, device=indices.device)
            if weighted
            else None
        )
        offsets = torch.tensor([0] + np.cumsum(Ls.flatten()).tolist()).to(dtype)
        warning = torch.tensor([0]).long()

        if mixed_B:
            B_offsets = torch.tensor(B_offsets, device="cuda", dtype=torch.int32)
            max_B = max(Bs)
        else:
            B_offsets = None
            max_B = -1

        self.assertEqual(indices.numel(), np.sum(Ls).item())
        self.assertEqual(offsets[-1], np.sum(Ls).item())
        if not use_cpu:
            indices, offsets, rows_per_table, warning = (
                indices.cuda(),
                offsets.cuda(),
                rows_per_table.cuda(),
                warning.cuda(),
            )
            if weighted:
                # pyre-fixme[16]: `Optional` has no attribute `cuda`.
                weights = weights.cuda()
        indices_copy = indices.clone()
        offsets_copy = offsets.clone()
        torch.ops.fbgemm.bounds_check_indices(
            rows_per_table,
            indices,
            offsets,
            bounds_check_mode,
            warning,
            weights,
            B_offsets=B_offsets,
            max_B=max_B,
        )
        # we don't modify when we are in-bounds.
        torch.testing.assert_close(indices_copy, indices)
        indices[:] = torch.iinfo(dtype).max
        if bounds_check_mode != BoundsCheckMode.FATAL:
            torch.ops.fbgemm.bounds_check_indices(
                rows_per_table,
                indices,
                offsets,
                bounds_check_mode,
                warning,
                weights,
                B_offsets=B_offsets,
                max_B=max_B,
            )
            torch.testing.assert_close(indices, torch.zeros_like(indices))
            if bounds_check_mode == BoundsCheckMode.WARNING:
                self.assertEqual(warning.item(), indices.numel())
        else:
            if use_cpu and indices.numel():
                with self.assertRaises(RuntimeError):
                    torch.ops.fbgemm.bounds_check_indices(
                        rows_per_table,
                        indices,
                        offsets,
                        bounds_check_mode,
                        warning,
                        weights,
                        B_offsets=B_offsets,
                        max_B=max_B,
                    )
            # It would be nice to test the CUDA implementation of BoundsCheckMode==FATAL,
            # but the device assert kills the CUDA context and requires a process restart,
            # which is a bit inconvenient.

        # test offsets bound errors
        indices = indices_copy.clone()
        offsets = offsets_copy.clone()
        if offsets.numel() > 0:
            offsets[0] = -100
        if offsets.numel() > 1:
            offsets[-1] += 100
        if bounds_check_mode != BoundsCheckMode.FATAL:
            torch.ops.fbgemm.bounds_check_indices(
                rows_per_table,
                indices,
                offsets,
                bounds_check_mode,
                warning,
                weights,
                B_offsets=B_offsets,
                max_B=max_B,
            )
            if offsets.numel() > 0:
                self.assertEqual(offsets[0].item(), 0)
            if offsets.numel() > 1:
                self.assertEqual(offsets[-1].item(), indices.numel())
            if bounds_check_mode == BoundsCheckMode.WARNING:
                # -1 because when we have 2 elements in offsets, we have only 1
                # warning for the pair.
                self.assertGreaterEqual(warning.item(), min(2, offsets.numel() - 1))
        else:
            if use_cpu and indices.numel():
                with self.assertRaises(RuntimeError):
                    torch.ops.fbgemm.bounds_check_indices(
                        rows_per_table,
                        indices,
                        offsets,
                        bounds_check_mode,
                        warning,
                        weights,
                    )

        # test offsets.size(0) ! = B * T + 1 case. Here we test with T >= 2 case.
        # If T == 1, we will always get the even division.
        # (does not apply to mixed_B = True)
        if not mixed_B and T >= 2:
            indices = indices_copy.clone()
            offsets = offsets_copy.clone()
            offsets = torch.cat(
                (
                    offsets,
                    torch.tensor(
                        [indices.numel()] * (T - 1),
                        dtype=offsets.dtype,
                        device=offsets.device,
                    ),
                ),
                dim=0,
            )
            with self.assertRaises(RuntimeError):
                torch.ops.fbgemm.bounds_check_indices(
                    rows_per_table,
                    indices,
                    offsets,
                    bounds_check_mode,
                    warning,
                    weights,
                )

        # test weights.size(0) != indices.size(0) case
        weights = torch.rand(
            (indices.size(0) + 1,), dtype=torch.float, device=indices.device
        )
        with self.assertRaises(RuntimeError):
            torch.ops.fbgemm.bounds_check_indices(
                rows_per_table,
                indices,
                offsets,
                bounds_check_mode,
                warning,
                weights,
                B_offsets=B_offsets,
                max_B=max_B,
            )

    def test_pickle(self) -> None:
        tensor_queue = torch.classes.fbgemm.TensorQueue(torch.empty(0))
        pickled = pickle.dumps(tensor_queue)
        unpickled = pickle.loads(pickled)  # noqa: F841

    @unittest.skipIf(*gpu_unavailable)
    def test_linearize_cache_indices(self) -> None:
        indices = torch.tensor(
            [10, 2, 3, 7, 1, 4, 5, 9, 2, 7, 6, 8, 5, 1, 0, 4],
            dtype=torch.int,
            device="cuda",
        )
        pruned_indices = torch.tensor(
            [10, -1, 3, 7, 1, 4, -1, 9, 2, -1, 6, 8, 5, 1, -1, 4],
            dtype=torch.int,
            device="cuda",
        )
        equal_offsets = torch.tensor([0, 4, 8, 12, 16], dtype=torch.int, device="cuda")
        varying_offsets = torch.tensor(
            [0, 1, 3, 6, 8, 10, 14, 15, 16], dtype=torch.int, device="cuda"
        )

        # Testing equal sized tables.
        cache_hash_size_cumsum_0 = torch.tensor([0, 12, 24, 36, 48]).cuda()
        linear_cache_indices_0 = torch.ops.fbgemm.linearize_cache_indices(
            cache_hash_size_cumsum_0, indices, equal_offsets
        )
        self.assertTrue(
            torch.equal(
                linear_cache_indices_0.cpu(),
                torch.tensor(
                    [10, 2, 3, 7, 13, 16, 17, 21, 26, 31, 30, 32, 41, 37, 36, 40],
                    dtype=torch.int,
                ),
            )
        )

        # Testing partially cached tables.
        cache_hash_size_cumsum_1 = torch.tensor([0, 12, -1, 24, 36]).cuda()
        linear_cache_indices_1 = torch.ops.fbgemm.linearize_cache_indices(
            cache_hash_size_cumsum_1, indices, equal_offsets
        )
        self.assertTrue(
            torch.equal(
                linear_cache_indices_1.cpu(),
                torch.tensor(
                    [10, 2, 3, 7, 13, 16, 17, 21, 36, 36, 36, 36, 29, 25, 24, 28],
                    dtype=torch.int,
                ),
            )
        )

        # Testing batched with varying pooling factor.
        cache_hash_size_cumsum_2 = torch.tensor([0, 12, -1, 24, 36]).cuda()
        linear_cache_indices_2 = torch.ops.fbgemm.linearize_cache_indices(
            cache_hash_size_cumsum_2, indices, varying_offsets
        )
        self.assertTrue(
            torch.equal(
                linear_cache_indices_2.cpu(),
                torch.tensor(
                    [10, 2, 3, 19, 13, 16, 17, 21, 36, 36, 36, 36, 36, 36, 24, 28],
                    dtype=torch.int,
                ),
            )
        )

        # Testing when multiple features share the same table.
        cache_hash_size_cumsum_3 = torch.tensor([0, 0, 12, 12, 24]).cuda()
        linear_cache_indices_3 = torch.ops.fbgemm.linearize_cache_indices(
            cache_hash_size_cumsum_3, indices, varying_offsets
        )
        self.assertTrue(
            torch.equal(
                linear_cache_indices_3.cpu(),
                torch.tensor(
                    [10, 2, 3, 7, 1, 4, 5, 9, 14, 19, 18, 20, 17, 13, 12, 16],
                    dtype=torch.int,
                ),
            )
        )

        # Testing equal sized tables + pruned indices
        cache_hash_size_cumsum_4 = torch.tensor([0, 12, 24, 36, 48]).cuda()
        linear_cache_indices_4 = torch.ops.fbgemm.linearize_cache_indices(
            cache_hash_size_cumsum_4, pruned_indices, equal_offsets
        )
        self.assertTrue(
            torch.equal(
                linear_cache_indices_4.cpu(),
                torch.tensor(
                    [10, 48, 3, 7, 13, 16, 48, 21, 26, 48, 30, 32, 41, 37, 48, 40],
                    dtype=torch.int,
                ),
            )
        )

        # Testing batched with varying pooling factor + pruned indices
        cache_hash_size_cumsum_5 = torch.tensor([0, 12, -1, 24, 36]).cuda()
        linear_cache_indices_5 = torch.ops.fbgemm.linearize_cache_indices(
            cache_hash_size_cumsum_5, pruned_indices, varying_offsets
        )
        self.assertTrue(
            torch.equal(
                linear_cache_indices_5.cpu(),
                torch.tensor(
                    [10, 36, 3, 19, 13, 16, 36, 21, 36, 36, 36, 36, 36, 36, 36, 28],
                    dtype=torch.int,
                ),
            )
        )

    @unittest.skipIf(*gpu_unavailable)
    def test_linearize_cache_indices_from_row_idx(self) -> None:
        update_row_indices = torch.tensor(
            [10, 2, 3, 7, 1, 4, 5, 9, 2, 7, 6, 8, 5, 1, 0, 4],
            dtype=torch.int,
            device="cuda",
        )
        update_table_indices = torch.tensor(
            [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
            dtype=torch.int,
            device="cuda",
        )
        varying_update_table_indices = torch.tensor(
            [0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3],
            dtype=torch.int,
            device="cuda",
        )

        # Testing equal sized tables.
        cache_hash_size_cumsum_0 = torch.tensor([0, 12, 24, 36, 48]).cuda()
        linear_cache_indices_0 = torch.ops.fbgemm.linearize_cache_indices_from_row_idx(
            cache_hash_size_cumsum_0,
            update_table_indices,
            update_row_indices,
        )
        self.assertTrue(
            torch.equal(
                linear_cache_indices_0.cpu(),
                torch.tensor(
                    [10, 2, 3, 7, 13, 16, 17, 21, 26, 31, 30, 32, 41, 37, 36, 40],
                    dtype=torch.int,
                ),
            )
        )

        # Testing partially cached tables.
        cache_hash_size_cumsum_1 = torch.tensor([0, 12, -1, 24, 36]).cuda()
        linear_cache_indices_1 = torch.ops.fbgemm.linearize_cache_indices_from_row_idx(
            cache_hash_size_cumsum_1,
            update_table_indices,
            update_row_indices,
        )
        self.assertTrue(
            torch.equal(
                linear_cache_indices_1.cpu(),
                torch.tensor(
                    [10, 2, 3, 7, 13, 16, 17, 21, 36, 36, 36, 36, 29, 25, 24, 28],
                    dtype=torch.int,
                ),
            )
        )

        # Testing batched with varying pooling factor.
        cache_hash_size_cumsum_2 = torch.tensor([0, 12, -1, 24, 36]).cuda()
        linear_cache_indices_2 = torch.ops.fbgemm.linearize_cache_indices_from_row_idx(
            cache_hash_size_cumsum_2,
            varying_update_table_indices,
            update_row_indices,
        )
        self.assertTrue(
            torch.equal(
                linear_cache_indices_2.cpu(),
                torch.tensor(
                    [10, 2, 3, 19, 13, 16, 17, 21, 36, 36, 36, 36, 36, 36, 24, 28],
                    dtype=torch.int,
                ),
            )
        )

    @unittest.skipIf(*gpu_unavailable)
    @given(
        associativity=st.sampled_from([1, DEFAULT_ASSOC]),
    )
    @settings(deadline=None)
    def test_lxu_cache_lookup(self, associativity: int) -> None:
        max_index: int = 8000
        # Use single cache set to avoid dealing with cache set hash algorithm.
        lxu_cache_state_gpu = (
            torch.arange(associativity, dtype=torch.int64).unsqueeze(0).cuda()
        )

        # Testing all miss.
        linear_cache_indices_0 = (
            torch.tensor([32, 33, 34, 35, 36, 100, 1000, 1725])
            if associativity <= 32
            else torch.tensor([64, 65, 66, 67, 68, 100, 1000, 1725])
        ).cuda()
        lxu_locations = torch.ops.fbgemm.lxu_cache_lookup(
            linear_cache_indices_0, lxu_cache_state_gpu, max_index
        )
        torch.testing.assert_close(
            lxu_locations,
            torch.full_like(lxu_locations, -1),
        )

        # Testing all hits.
        cache_indices_1 = torch.randint(0, associativity, (associativity,))
        linear_cache_indices_1 = cache_indices_1.cuda()
        lxu_locations = torch.ops.fbgemm.lxu_cache_lookup(
            linear_cache_indices_1, lxu_cache_state_gpu, max_index
        )
        torch.testing.assert_close(
            lxu_locations.cpu(),
            cache_indices_1.int(),
        )

        # Testing mixture.
        miss_cache_indices_0 = torch.randint(associativity, max_index // 2, (10,))
        hit_cache_indices_0 = torch.randint(0, associativity, (8,))
        miss_cache_indices_1 = torch.randint(max_index // 2, max_index, (16,))
        hit_cache_indices_1 = torch.randint(0, associativity, (8,))
        linear_cache_indices_2 = torch.cat(
            [
                miss_cache_indices_0,
                hit_cache_indices_0,
                miss_cache_indices_1,
                hit_cache_indices_1,
            ]
        ).cuda()
        lxu_locations = torch.ops.fbgemm.lxu_cache_lookup(
            linear_cache_indices_2, lxu_cache_state_gpu, max_index
        )

        expected_result = torch.cat(
            [
                torch.full_like(miss_cache_indices_0, -1),
                hit_cache_indices_0,
                torch.full_like(miss_cache_indices_1, -1),
                hit_cache_indices_1,
            ]
        ).int()
        torch.testing.assert_close(
            lxu_locations.cpu(),
            expected_result,
        )

    @unittest.skipIf(*gpu_unavailable)
    @given(
        cache_sets=st.integers(min_value=10, max_value=300),
    )
    @settings(verbosity=VERBOSITY, max_examples=MAX_EXAMPLES, deadline=None)
    def test_lxu_cache_locking_counter_decrement(
        self,
        cache_sets: int,
    ) -> None:
        warp_size = DEFAULT_ASSOC
        N = cache_sets * warp_size
        lxu_cache_locking_counter = torch.randint(
            low=1,
            high=3,
            size=[cache_sets, warp_size],
            device="cuda",
            dtype=torch.int32,
        )
        counter_ref = lxu_cache_locking_counter.tolist()
        lxu_cache_locations_list = []
        lxu_cache_locations_set = set()
        for _ in range(3 * N):
            location = random.randrange(-1, N)
            lxu_cache_locations_list.append(location)
            lxu_cache_locations_set.add(location)

        for idx in lxu_cache_locations_set:
            if idx >= 0:
                q, r = idx // warp_size, idx % warp_size
                counter_ref[q][r] -= 1

        counter_ref = torch.tensor(counter_ref, device="cuda", dtype=torch.int32)
        lxu_cache_locations = torch.tensor(
            lxu_cache_locations_list, device="cuda", dtype=torch.int32
        )
        torch.ops.fbgemm.lxu_cache_locking_counter_decrement(
            lxu_cache_locking_counter, lxu_cache_locations
        )
        self.assertTrue(torch.equal(lxu_cache_locking_counter, counter_ref))

    @unittest.skipIf(*gpu_unavailable)
    @given(
        T=st.integers(min_value=1, max_value=5),
        D=st.integers(min_value=2, max_value=64),
        log_E=st.integers(min_value=2, max_value=3),
        N=st.integers(min_value=0, max_value=50),
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
                SparseType.FP32,
                SparseType.FP16,
                SparseType.INT8,
            ]
        ),
        use_cpu=st.booleans()
        if (gpu_available and not TEST_WITH_ROCM)
        else st.just(False)
        if (gpu_available and TEST_WITH_ROCM)
        else st.just(True),
        test_internal=st.booleans(),
    )
    @settings(verbosity=VERBOSITY, max_examples=MAX_EXAMPLES, deadline=None)
    def test_embedding_inplace_update(
        self,
        T: int,  # num of embedding tables
        D: int,  # embedding dim
        log_E: int,  # embedding table row number
        N: int,  # num of update rows per table
        weights_ty: SparseType,
        output_dtype: SparseType,
        use_cpu: bool,
        test_internal: bool,  # test with OSS op or internal customized op
    ) -> None:
        D_alignment = max(weights_ty.align_size(), output_dtype.align_size())
        D = round_up(D, D_alignment)
        Ds = [
            round_up(
                np.random.randint(low=int(max(0.25 * D, 1)), high=int(1.0 * D)),
                D_alignment,
            )
            for _ in range(T)
        ]
        E = int(10**log_E)
        Es = [np.random.randint(low=int(0.5 * E), high=int(2.0 * E)) for _ in range(T)]
        row_alignment = 1 if use_cpu else 16
        current_device = "cpu" if use_cpu else torch.cuda.current_device()
        location = EmbeddingLocation.HOST if use_cpu else EmbeddingLocation.DEVICE

        weights_ty_list = [weights_ty] * T
        if open_source:
            test_internal = False

        # create two embedding bag op with random weights
        locations = [location] * T
        op = IntNBitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                ("", E, D, W_TY, L)
                for (E, D, W_TY, L) in zip(Es, Ds, weights_ty_list, locations)
            ],
            output_dtype=output_dtype,
            device=current_device,
        )
        op.fill_random_weights()
        op_ref = IntNBitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                ("", E, D, W_TY, L)
                for (E, D, W_TY, L) in zip(Es, Ds, weights_ty_list, locations)
            ],
            output_dtype=output_dtype,
            device=current_device,
        )
        op_ref.fill_random_weights()

        # randomly generate update table and row indices
        update_table_indices = []
        update_table_indices2 = []
        update_row_indices = []
        update_row_indices2 = []
        for t in range(T):
            n = np.random.randint(low=0, high=N) if N > 0 else 0
            if n == 0:
                continue
            update_table_indices.append(t)
            update_row_id_list = random.sample(range(Es[t]), n)
            update_row_indices.append(update_row_id_list)
            update_table_indices2.extend([t] * n)
            update_row_indices2.extend(update_row_id_list)

        # generate update tensor based on weights from "op_ref" embedding table
        update_weights_list = []
        ref_split_weights = op_ref.split_embedding_weights(split_scale_shifts=False)

        update_weight_size = sum(
            [
                rounded_row_size_in_bytes(
                    Ds[t],
                    weights_ty_list[t],
                    row_alignment,
                )
                for t in update_table_indices2
            ]
        )
        update_weights_tensor2 = torch.randint(
            low=0,
            high=255,
            size=(update_weight_size,),
            dtype=torch.uint8,
            device=current_device,
        )

        update_offsets = 0
        for i in range(len(update_table_indices)):
            table_idx = update_table_indices[i]
            (ref_weights, _) = ref_split_weights[table_idx]

            D_bytes = rounded_row_size_in_bytes(
                Ds[table_idx], weights_ty_list[table_idx], row_alignment
            )

            update_weights = []
            for row_idx in update_row_indices[i]:
                update_weights.append(ref_weights[row_idx].tolist())
                update_weights_tensor2[
                    update_offsets : update_offsets + D_bytes
                ] = ref_weights[row_idx]
                update_offsets += D_bytes

            update_weights_tensor = torch.tensor(
                update_weights,
                device=current_device,
                dtype=torch.uint8,
            )
            update_weights_list.append(update_weights_tensor)

        # run inplace update on "op" embedding table
        if not test_internal:
            # Test scatter_ based OSS solution
            op.embedding_inplace_update(
                update_table_indices,
                update_row_indices,
                update_weights_list,
            )
        else:
            # Test customized op
            op.embedding_inplace_update_internal(
                update_table_indices2,
                update_row_indices2,
                update_weights_tensor2,
            )

        # verify weights are equal with "op_ref" for the updated rows in "op"
        split_weights = op.split_embedding_weights(split_scale_shifts=False)
        for i in range(len(update_table_indices)):
            t = update_table_indices[i]
            for r in update_row_indices[i]:
                (weights, _) = split_weights[t]
                (ref_weights, _) = ref_split_weights[t]
                self.assertEqual(weights.size(), ref_weights.size())
                torch.testing.assert_close(
                    weights[r],
                    ref_weights[r],
                    rtol=1e-2,
                    atol=1e-2,
                    equal_nan=True,
                )

    @given(
        T=st.integers(min_value=1, max_value=5),
        D=st.integers(min_value=2, max_value=128),
        log_E=st.integers(min_value=2, max_value=3),
        weights_precision=st.sampled_from(
            [SparseType.FP16, SparseType.FP32, SparseType.INT8]
        ),
        mixed=st.booleans(),
        use_cache=st.booleans(),
        output_dtype=st.sampled_from([SparseType.FP32, SparseType.FP16]),
        num_indices_per_table=st.integers(min_value=1, max_value=5),
    )
    @settings(
        verbosity=VERBOSITY,
        max_examples=MAX_EXAMPLES,
        deadline=None,
    )
    def test_reset_embedding_weight_momentum(
        self,
        T: int,
        D: int,
        log_E: int,
        weights_precision: SparseType,
        mixed: bool,
        use_cache: bool,
        output_dtype: SparseType,
        num_indices_per_table: int,
    ) -> None:
        emb_op = SplitTableBatchedEmbeddingBagsCodegen
        E = int(10**log_E)
        D = D * 4
        Ds: List[int] = []
        Es: List[int] = []
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
        compute_device = ComputeDevice.CUDA
        if use_cache:
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
        optimizer = OptimType.EXACT_ROWWISE_ADAGRAD
        cc = emb_op(
            embedding_specs=[
                (E, D, M, compute_device) for (E, D, M) in zip(Es, Ds, managed)
            ],
            optimizer=optimizer,
            weights_precision=weights_precision,
            output_dtype=output_dtype,
        )

        pruned_indices: List[int] = []
        pruned_indices_offsets: List[int] = [0]
        logical_table_ids: List[int] = []
        buffer_ids: List[int] = []
        for i in range(len(Es)):
            indices = [
                np.random.randint(low=1, high=int(Es[i] - 2))
                for _ in range(num_indices_per_table)
            ]
            pruned_indices += indices
            pruned_indices_offsets.append(
                pruned_indices_offsets[i] + num_indices_per_table
            )
            logical_table_ids.append(i)
            buffer_ids.append(i)
        pruned_indices_tensor = to_device(
            torch.tensor(pruned_indices, dtype=torch.int64, requires_grad=False), False
        )
        pruned_indices_offsets_tensor = to_device(
            torch.tensor(
                pruned_indices_offsets, dtype=torch.int64, requires_grad=False
            ),
            False,
        )
        logical_table_ids_tensor = to_device(
            torch.tensor(logical_table_ids, dtype=torch.int32, requires_grad=False),
            False,
        )
        buffer_ids_tensor = to_device(
            torch.tensor(buffer_ids, dtype=torch.int32, requires_grad=False), False
        )

        momentum1: List[Tensor] = [
            s for (s,) in cc.split_optimizer_states()
        ]  # List[rows]
        weight: List[Tensor] = cc.split_embedding_weights()  # List[(rows, dim)]
        for t in range(T):
            momentum1[t].fill_(1)
            weight[t].fill_(1)

        def check_weight_momentum(v: int) -> None:
            for i in range(len(pruned_indices)):
                logical_id = i // num_indices_per_table
                table_momentum1 = momentum1[logical_id]
                table_weight = weight[logical_id]
                dim = Ds[logical_id]
                expected_row_momentum1 = to_device(
                    torch.tensor(v, dtype=torch.float32), False
                )
                expected_row_weight = to_device(
                    torch.tensor([v] * dim, dtype=weights_precision.as_dtype()),
                    False,
                )
                pruned_index = pruned_indices[i]
                row_weight = table_weight[pruned_index]
                if weights_precision == SparseType.INT8:
                    row_weight = row_weight[:-INT8_EMB_ROW_DIM_OFFSET]
                self.assertEqual(table_momentum1[pruned_index], expected_row_momentum1)
                torch.testing.assert_close(
                    row_weight,
                    expected_row_weight,
                    rtol=0,
                    atol=0,
                    equal_nan=True,
                )

        check_weight_momentum(1)

        cc.reset_embedding_weight_momentum(
            pruned_indices_tensor,
            pruned_indices_offsets_tensor,
            logical_table_ids_tensor,
            buffer_ids_tensor,
        )

        check_weight_momentum(0)

    @unittest.skipIf(*gpu_unavailable)
    @given(
        T=st.integers(min_value=1, max_value=10),
        D=st.integers(min_value=2, max_value=128),
        B=st.integers(min_value=1, max_value=128),
        log_E=st.integers(min_value=3, max_value=5),
        L=st.integers(min_value=0, max_value=20),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_unique_lxu_cache_lookup(
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
    ) -> None:
        E = int(10**log_E)

        indices = to_device(
            torch.randint(low=0, high=E, size=(T * L * B,)),
            use_cpu=False,
        ).long()
        offsets = to_device(
            torch.tensor([0] + list(accumulate([L] * (T * L)))),
            use_cpu=False,
        ).long()

        def unique_lookup(
            indices: Tensor,
            offsets: Tensor,
            cache_hash_size_cumsum: Tensor,
            total_cache_hash_size: int,
        ) -> Tuple[Tensor, Tensor]:
            linear_cache_indices = torch.ops.fbgemm.linearize_cache_indices(
                cache_hash_size_cumsum,
                indices,
                offsets,
            )

            uniq_indices, uniq_indices_length, _ = torch.ops.fbgemm.get_unique_indices(
                linear_cache_indices, total_cache_hash_size, compute_count=False
            )

            uniq_lxu_cache_locations = torch.ops.fbgemm.lxu_cache_lookup(
                uniq_indices,
                lxu_cache_state,
                total_cache_hash_size,
                gather_cache_stats=False,
                num_uniq_cache_indices=uniq_indices_length,
            )

            return uniq_lxu_cache_locations, uniq_indices_length

        def duplicate_lookup(
            indices: Tensor,
            offsets: Tensor,
            cache_hash_size_cumsum: Tensor,
            total_cache_hash_size: int,
        ) -> Tensor:
            linear_cache_indices = torch.ops.fbgemm.linearize_cache_indices(
                cache_hash_size_cumsum,
                indices,
                offsets,
            )

            lxu_cache_locations = torch.ops.fbgemm.lxu_cache_lookup(
                linear_cache_indices,
                lxu_cache_state,
                total_cache_hash_size,
                gather_cache_stats=False,
            )
            return lxu_cache_locations

        cache_sets = int((E * T) * 0.2)
        lxu_cache_state = torch.zeros(
            cache_sets,
            DEFAULT_ASSOC,
            device="cuda",
            dtype=torch.int64,
        ).fill_(-1)

        hash_sizes = torch.tensor([E] * T, dtype=torch.long, device="cuda")
        cache_hash_size_cumsum = torch.ops.fbgemm.asynchronous_complete_cumsum(
            hash_sizes
        )
        total_cache_hash_size = cache_hash_size_cumsum[-1].item()

        linear_cache_indices = torch.ops.fbgemm.linearize_cache_indices(
            cache_hash_size_cumsum,
            indices,
            offsets,
        )

        # Emulate cache population
        uniq_indices_cpu = linear_cache_indices.unique().cpu()
        index_cache_set_map = uniq_indices_cpu.clone()
        index_cache_set_map.apply_(
            lambda x: torch.ops.fbgemm.lxu_cache_slot(x, cache_sets)
        )
        index_cache_set_map = index_cache_set_map.tolist()
        uniq_indices_cpu = uniq_indices_cpu.tolist()

        slots = {}
        for idx, c in zip(uniq_indices_cpu, index_cache_set_map):
            if c not in slots:
                slots[c] = 0
            slot = slots[c]
            if slot < DEFAULT_ASSOC:
                lxu_cache_state[c][slot] = idx
            slots[c] = slot + 1

        # Run unique lookup
        uniq_lookup_output, uniq_indices_length = unique_lookup(
            indices, offsets, cache_hash_size_cumsum, total_cache_hash_size
        )

        # Run duplicate lookup
        duplicate_lookup_output = duplicate_lookup(
            indices, offsets, cache_hash_size_cumsum, total_cache_hash_size
        )

        # Start running validation

        # Compute unique indices using PyTorch ops
        sorted_linear_cache_indices, inverse_sorted_cache_indices = torch.sort(
            linear_cache_indices
        )
        ref_uniq_cache_indices, cache_indices_counts = torch.unique_consecutive(
            sorted_linear_cache_indices, return_inverse=False, return_counts=True
        )

        # Convert to lists
        cache_indices_counts = cache_indices_counts.cpu().tolist()
        uniq_lookup_output = uniq_lookup_output.cpu().tolist()

        # Validate the number of unique cache indices
        ref_num_uniq_indices = ref_uniq_cache_indices.numel()
        assert ref_num_uniq_indices == uniq_indices_length.item()

        # Expand
        reshaped_uniq_lookup_output = uniq_lookup_output[:ref_num_uniq_indices]
        sorted_lxu_cache_locations = to_device(
            torch.tensor(
                np.repeat(reshaped_uniq_lookup_output, cache_indices_counts),
                dtype=duplicate_lookup_output.dtype,
            ),
            use_cpu=False,
        )

        _, cache_location_indices = torch.sort(inverse_sorted_cache_indices)

        expanded_lxu_cache_locations = torch.index_select(
            sorted_lxu_cache_locations, 0, cache_location_indices
        )

        assert torch.equal(expanded_lxu_cache_locations, duplicate_lookup_output)


if __name__ == "__main__":
    unittest.main()
