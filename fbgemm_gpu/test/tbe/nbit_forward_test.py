#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors[56]

import copy
import random
import unittest

from typing import Callable, Dict, List, Optional

import fbgemm_gpu
import hypothesis.strategies as st
import numpy as np
import torch

from fbgemm_gpu.split_embedding_configs import FP8QuantizationConfig, SparseType
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
    CacheAlgorithm,
    EmbeddingLocation,
    PoolingMode,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_inference import (
    IntNBitTableBatchedEmbeddingBagsCodegen,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_training import DEFAULT_ASSOC

from hypothesis import assume, given, HealthCheck, settings, Verbosity
from hypothesis.strategies import composite

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


# pyre-ignore
additional_decorators: Dict[str, List[Callable]] = {
    "test_faketensor__test_nbit_forward_uvm_cache": [
        unittest.skip("CUDA Assert"),
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
class NBitFowardTest(unittest.TestCase):
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

        if not use_cpu:
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
                use_cpu=use_cpu,
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


if __name__ == "__main__":
    unittest.main()
