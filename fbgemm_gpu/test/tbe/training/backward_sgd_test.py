#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# pyre-ignore-all-errors[56]

import unittest

import hypothesis.strategies as st
import numpy as np
import torch
from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType, SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    ComputeDevice,
    SplitTableBatchedEmbeddingBagsCodegen,
    UserEnabledConfigDefinition,
)
from fbgemm_gpu.tbe.cache.cache_config import CacheAlgorithm
from fbgemm_gpu.tbe.config.embedding_config import EmbeddingLocation, PoolingMode
from fbgemm_gpu.tbe.utils import (
    b_indices,
    get_table_batched_offsets_from_dense,
    round_up,
    to_device,
)
from hypothesis import given, settings, Verbosity

from .. import common  # noqa E402
from ..common import (
    format_ref_tensors_in_mixed_B_layout,
    gen_mixed_B_batch_sizes,
    MAX_EXAMPLES,
    MAX_EXAMPLES_LONG_RUNNING,
    open_source,
    v1_lookup,
)

if open_source:
    # pyre-ignore[21]
    from test_utils import (
        additional_decorators,
        gpu_unavailable,
        optests,
        running_on_github,
        TEST_WITH_ROCM,
        use_cpu_strategy,
    )
else:
    from fbgemm_gpu.test.test_utils import (
        additional_decorators,
        gpu_unavailable,
        optests,
        running_on_github,
        TEST_WITH_ROCM,
        use_cpu_strategy,
    )

VERBOSITY: Verbosity = Verbosity.verbose


@optests.generate_opcheck_tests(fast=True, additional_decorators=additional_decorators)
class BackwardSGDTest(unittest.TestCase):
    def execute_backward_sgd_(  # noqa C901
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
        long_segments: bool,
        pooling_mode: PoolingMode,
        use_cpu: bool,
        output_dtype: SparseType,
        use_writeback_bwd_prehook: bool = False,
        enable_writeback_bwd_prehook_first_feature_only: bool = False,
        use_api_v1: bool = False,
        use_preproc_bwd: bool = False,
    ) -> None:
        # The preproc-consume backward is only wired for the common path (bagged/SUM,
        # non-VBE, CUDA, FP32, no cache/writeback). Skip other combos so the arm is
        # only exercised where it is valid -- same early-return style as below.
        if use_preproc_bwd and (
            use_cpu
            or use_cache
            or mixed_B
            or pooling_mode != PoolingMode.SUM
            or use_writeback_bwd_prehook
            or weights_precision != SparseType.FP32
            or output_dtype != SparseType.FP32
        ):
            return
        # NOTE: cache is not applicable to CPU version.
        if use_cpu and use_cache:
            return
        # NOTE: limit (T * B * L * D) to avoid timeout for CPU version!
        if use_cpu and T * B * L * D > 2048:
            return
        if use_cpu and weights_precision == SparseType.FP16:
            return
        # V1 API doesn't support nobag on CPU (only PT2 path does)
        if use_cpu and pooling_mode == PoolingMode.NONE and use_api_v1:
            return
        if mixed and pooling_mode == PoolingMode.NONE:
            return
        if weighted and pooling_mode == PoolingMode.NONE:
            return
        if pooling_mode != PoolingMode.SUM and weighted:
            return
        # TODO: Support these cases
        if mixed_B and (
            weights_precision == SparseType.INT8
            or output_dtype == SparseType.INT8
            or use_cpu
            or pooling_mode == PoolingMode.NONE
        ):
            return

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
        else:
            low = max(int(0.25 * B), 1)
            high = int(B)
            if low == high:
                Bs = [B] * T
            else:
                Bs = [np.random.randint(low=low, high=high) for _ in range(T)]

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

        if weights_precision == SparseType.FP16:
            bs = [b.half() for b in bs]

        feature_table_map = list(range(T))
        # Skip the duplicate feature test for use_writeback_bwd_prehook=True case
        table_to_replicate = 0 if use_writeback_bwd_prehook else T // 2
        if not use_writeback_bwd_prehook:
            # pyre-fixme[6]: For 2nd param expected `Embedding` but got
            #  `Union[Embedding, EmbeddingBag]`.
            bs.insert(table_to_replicate, bs[table_to_replicate])
            feature_table_map.insert(table_to_replicate, table_to_replicate)

        num_features = len(feature_table_map)
        if not mixed_B:
            Bs = [B] * num_features
            Bs_rank_feature = [[0]]
        else:
            Bs_rank_feature, Bs = gen_mixed_B_batch_sizes(B, num_features)

        # Generate indices
        xs = [
            to_device(
                torch.from_numpy(
                    np.random.choice(range(Es[t]), size=(b, L), replace=True).astype(
                        np.int64
                    )
                ),
                use_cpu,
            )
            for t, b in zip(feature_table_map, Bs)
        ]

        if long_segments and L > 0:
            for x in xs:
                x[:, 0] = 0

        # Generate positional weights
        xws = [to_device(torch.randn(size=(b, L)), use_cpu) for b in Bs]

        if weights_precision == SparseType.FP16:
            xws = [xw.half() for xw in xws]

        # Run baseline's forward
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

        # Cast output type to output_dtype
        if weights_precision != output_dtype:
            fs = [f.to(output_dtype.as_dtype()) for f in fs]

        # Generate gradients
        if use_writeback_bwd_prehook:
            # require constant grad for the same entity for writeback purpose
            if enable_writeback_bwd_prehook_first_feature_only:
                gos = [
                    torch.ones_like(f) if index == 0 else torch.zeros_like(f)
                    for index, f in enumerate(fs)
                ]
            else:
                gos = [torch.ones_like(f) for f in fs]
        else:
            gos = [torch.randn_like(f) for f in fs]
            del bs[table_to_replicate]
        # Run baseline's backward
        [f.backward(go) for (f, go) in zip(fs, gos)]
        # do SGD update
        lr = 0.05
        if use_writeback_bwd_prehook:

            new_weights = []
            for b, x in zip(bs, xs):
                # pyre-ignore[16]
                grad = b.weight.grad.coalesce()
                indices = grad.indices()[0]
                values = grad.values()
                raw_indices = x
                unique_indices, counts = torch.unique(raw_indices, return_counts=True)
                index_to_count = {
                    index.item(): count.item()
                    for index, count in zip(unique_indices, counts)
                }
                # Create a tensor of counts corresponding to the input indices
                counts_tensor = torch.tensor(
                    [index_to_count[index.item()] for index in indices]
                ).to(values.device)
                new_grad_value = values / counts_tensor.unsqueeze(1)
                new_grad = torch.sparse_coo_tensor(
                    indices.unsqueeze(0), new_grad_value, grad.shape
                )

                new_weights.append(b.weight - lr * new_grad)

        else:
            # pyre-ignore[58]
            new_weights = [(b.weight - b.weight.grad * lr) for b in bs]

        # Create a TBE op
        cc = emb_op(
            embedding_specs=[
                (E, D, M, compute_device) for (E, D, M) in zip(Es, Ds, managed)
            ],
            optimizer=OptimType.EXACT_SGD,
            feature_table_map=feature_table_map,
            learning_rate=lr,
            weights_precision=weights_precision,
            cache_algorithm=cache_algorithm,
            pooling_mode=pooling_mode,
            output_dtype=output_dtype,
            extra_optimizer_config=UserEnabledConfigDefinition(
                use_writeback_bwd_prehook=use_writeback_bwd_prehook,
                writeback_first_feature_only=enable_writeback_bwd_prehook_first_feature_only,
            ),
        )
        for t in range(T):
            cc.split_embedding_weights()[t].data.copy_(bs[t].weight)

        x = torch.cat([x.contiguous().flatten() for x in xs], dim=0)
        xw = torch.cat([xw.contiguous().flatten() for xw in xws], dim=0)

        indices, offsets = get_table_batched_offsets_from_dense(
            x, L, sum(Bs), use_cpu=use_cpu
        )

        # The preproc op's find_long_segments kernel cannot launch a 0-width grid, so
        # the preproc arm needs at least one index. The normal backward handles empty.
        if use_preproc_bwd and indices.numel() == 0:
            return

        batch_size_per_feature_per_rank = Bs_rank_feature if mixed_B else None

        # Run TBE's forward
        per_sample_weights = (
            to_device(xw.contiguous().view(-1), use_cpu) if weighted else None
        )
        if use_api_v1:
            fc2 = v1_lookup(
                cc,
                indices,
                offsets,
                use_cpu=use_cpu,
                per_sample_weights=per_sample_weights,
                batch_size_per_feature_per_rank=batch_size_per_feature_per_rank,
            )
        else:
            fc2 = cc(
                indices,
                offsets,
                per_sample_weights=per_sample_weights,
                batch_size_per_feature_per_rank=batch_size_per_feature_per_rank,
            )
        # Generate gradients
        if do_pooling:
            if mixed_B:
                goc = format_ref_tensors_in_mixed_B_layout(gos, Bs_rank_feature)
            else:
                goc = torch.cat([go.view(B, -1) for go in gos], dim=1)
        else:
            goc = torch.cat(gos, dim=0)

        # Run TBE's backward
        if use_preproc_bwd:
            # Drive the fused SGD backward through the backend-dispatched
            # *_pt2_wrapper op with a hoisted index-preproc bundle instead of the
            # normal autograd backward. The in-place weight update must match the
            # baseline all the same -- see _run_preproc_backward.
            self._run_preproc_backward(
                cc,
                indices,
                offsets,
                goc,
                B,
                num_features,
                weighted,
                per_sample_weights,
            )
        else:
            fc2.backward(goc)

        if use_cache:
            cc.flush()
        for t in range(T):
            torch.testing.assert_close(
                cc.split_embedding_weights()[t],
                (
                    new_weights[t].half()
                    if weights_precision == SparseType.FP16 and not use_cpu
                    else new_weights[t]
                ),
                atol=(
                    1.0e-2
                    if long_segments
                    else (5.0e-3 if weights_precision == SparseType.FP16 else 1.0e-5)
                ),
                rtol=(
                    1.0e-1
                    if long_segments
                    else (2.0e-2 if weights_precision == SparseType.FP16 else 1.0e-5)
                ),
            )

    def _run_preproc_backward(
        self,
        cc: SplitTableBatchedEmbeddingBagsCodegen,
        indices: torch.Tensor,
        offsets: torch.Tensor,
        grad_output: torch.Tensor,
        B: int,
        num_features: int,
        weighted: bool,
        per_sample_weights: torch.Tensor | None,
    ) -> None:
        """Feed a hoisted index-preproc bundle into the fused SGD backward op.

        Instead of driving the normal autograd backward, compute the index-preproc
        (``tbe_bwd_indices_preproc``) up front and hand it to the backward driver via
        the trailing ``preproc_tensors`` arg. The op skips the inline
        ``transpose_embedding_input`` + ``find_long_segments`` and consumes the bundle
        instead; the in-place fused SGD weight update is identical, so the caller's
        existing weight assertion validates numerical correctness.

        The call goes through the backend-dispatched ``*_pt2_wrapper`` op (NOT the
        CUDA-only ``*_exact_cuda`` op), so the path stays portable across backends
        (CPU / Meta / MTIA), which is the whole point of routing preproc through the
        wrapper.
        """
        ind, off, _, _ = cc.prepare_inputs(
            indices, offsets, None, None, force_cast_input_types=True
        )
        info_B_num_bits, info_B_mask = torch.ops.fbgemm.get_infos_metadata(
            cc.hash_size_cumsum, B, num_features
        )
        preproc = list(
            torch.ops.fbgemm.tbe_bwd_indices_preproc(
                cc.hash_size_cumsum,
                cc.total_hash_size_bits,
                ind,
                off,
                nobag=False,
                vbe_b_t_map=None,
                info_B_num_bits=info_B_num_bits,
                info_B_mask=info_B_mask,
                total_unique_indices=-1,
            )
        )
        wdesc = "weighted" if weighted else "unweighted"
        backward_op = getattr(
            torch.ops.fbgemm,
            f"split_embedding_backward_codegen_sgd_{wdesc}_pt2_wrapper",
        )
        # Bagged ops always take an indice_weights arg; unweighted passes an empty one.
        indice_weights = (
            per_sample_weights
            if weighted
            else torch.empty(0, dtype=torch.float, device=cc.weights_dev.device)
        )
        # Post-D113869217 the *_pt2_wrapper op takes a packed `weights` TensorList
        # and a packed `aux_tensor_bwd` TensorList instead of loose tensors. GPU
        # (non-VBE) layout: weights = [dev, placements, offsets, uvm, lxu_cache];
        # aux_tensor_bwd = [lxu_cache_locations] (slot 0 only, size 1).
        weights = [
            cc.weights_dev,
            cc.weights_placements,
            cc.weights_offsets,
            cc.weights_uvm,
            cc.lxu_cache_weights,
        ]
        aux_tensor_bwd = [cc.lxu_cache_locations_empty]
        backward_op(
            grad_output.contiguous(),
            weights,
            cc.D_offsets,
            cc.max_D,
            cc.mixed_D,
            cc.hash_size_cumsum,
            cc.total_hash_size_bits,
            ind,
            off,
            cc.pooling_mode,
            indice_weights,
            aux_tensor_bwd,
            0,  # BT_block_size (unused)
            32,  # max_segment_length_per_warp
            cc.stochastic_rounding,
            info_B_num_bits,
            info_B_mask,
            False,  # use_uniq_cache_locations
            False,  # use_homogeneous_placements
            cc.learning_rate_tensor,
            cc.output_dtype,
            preproc,
        )

    @given(
        T=st.integers(min_value=1, max_value=5),
        D=st.integers(min_value=2, max_value=256),
        B=st.integers(min_value=1, max_value=128),
        log_E=st.integers(min_value=3, max_value=5),
        L=st.integers(min_value=0, max_value=20),
        weights_precision=st.sampled_from([SparseType.FP16, SparseType.FP32]),
        weighted=st.booleans(),
        mixed=st.booleans(),
        mixed_B=st.booleans(),
        use_cache=st.booleans(),
        cache_algorithm=st.sampled_from(CacheAlgorithm),
        long_segments=st.booleans(),
        pooling_mode=st.sampled_from(
            [
                PoolingMode.SUM,
                PoolingMode.MEAN,
                PoolingMode.NONE,
            ]
        ),
        use_cpu=use_cpu_strategy(),
    )
    @settings(
        verbosity=VERBOSITY,
        max_examples=MAX_EXAMPLES,
        deadline=None,
    )
    def test_backward_sgd(  # noqa C901
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
        long_segments: bool,
        pooling_mode: PoolingMode,
        use_cpu: bool,
    ) -> None:
        self.execute_backward_sgd_(
            T,
            D,
            B,
            log_E,
            L,
            weights_precision,
            weighted,
            mixed,
            mixed_B if not use_cpu else False,
            use_cache,
            cache_algorithm,
            long_segments,
            pooling_mode,
            use_cpu,
            SparseType.FP32,  # output_dtype
        )

    @unittest.skipIf(*gpu_unavailable)
    @given(
        T=st.integers(min_value=1, max_value=5),
        D=st.integers(min_value=2, max_value=256),
        B=st.integers(min_value=1, max_value=128),
        log_E=st.integers(min_value=3, max_value=5),
        L=st.integers(min_value=1, max_value=20),
        weighted=st.booleans(),
        mixed=st.booleans(),
        long_segments=st.booleans(),
    )
    @settings(
        verbosity=VERBOSITY,
        max_examples=MAX_EXAMPLES,
        deadline=None,
    )
    def test_backward_sgd_preproc(
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        weighted: bool,
        mixed: bool,
        long_segments: bool,
    ) -> None:
        """Same fused SGD backward as ``test_backward_sgd``, but the backward is driven
        through the ``*_pt2_wrapper`` op fed a hoisted index-preproc bundle
        (``use_preproc_bwd=True``). Consuming the preproc must produce the identical
        weight update, so this reuses the harness assertion. Constrained to the common
        path the preproc consume is wired for: bagged/SUM, CUDA, FP32, no cache."""
        self.execute_backward_sgd_(
            T,
            D,
            B,
            log_E,
            L,
            weights_precision=SparseType.FP32,
            weighted=weighted,
            mixed=mixed,
            mixed_B=False,
            use_cache=False,
            cache_algorithm=CacheAlgorithm.LRU,
            long_segments=long_segments,
            pooling_mode=PoolingMode.SUM,
            use_cpu=False,
            output_dtype=SparseType.FP32,
            use_preproc_bwd=True,
        )

    @given(
        T=st.integers(min_value=1, max_value=3),
        D=st.sampled_from([2, 4, 128, 256]),
        B=st.integers(min_value=1, max_value=10),
        log_E=st.integers(min_value=3, max_value=5),
        L=st.integers(min_value=1, max_value=20),
        long_segments=st.booleans(),
    )
    @settings(
        verbosity=VERBOSITY,
        max_examples=MAX_EXAMPLES,
        deadline=None,
    )
    def test_backward_sgd_fp32_pmNONE_cpu(
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        long_segments: bool,
    ) -> None:
        self.execute_backward_sgd_(
            T,
            D,
            B,
            log_E,
            L,
            weights_precision=SparseType.FP32,
            weighted=False,
            mixed=False,
            mixed_B=False,
            use_cache=False,
            cache_algorithm=CacheAlgorithm.LRU,
            long_segments=long_segments,
            pooling_mode=PoolingMode.NONE,
            use_cpu=True,
            output_dtype=SparseType.FP32,
        )

    @given(
        T=st.integers(min_value=1, max_value=5),
        D=st.integers(min_value=2, max_value=256),
        B=st.integers(min_value=1, max_value=128),
        log_E=st.integers(min_value=3, max_value=5),
        L=st.integers(min_value=0, max_value=20),
        weights_precision=st.sampled_from([SparseType.FP16, SparseType.FP32]),
        weighted=st.booleans(),
        mixed=st.booleans(),
        use_cache=st.booleans(),
        cache_algorithm=st.sampled_from(CacheAlgorithm),
        long_segments=st.booleans(),
        pooling_mode=st.sampled_from(
            [
                PoolingMode.SUM,
                PoolingMode.MEAN,
            ]
        ),
    )
    @settings(
        verbosity=VERBOSITY,
        max_examples=MAX_EXAMPLES,
        deadline=None,
    )
    def test_backward_sgd_vbe_cpu(  # noqa C901
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        weights_precision: SparseType,
        weighted: bool,
        mixed: bool,
        use_cache: bool,
        cache_algorithm: CacheAlgorithm,
        long_segments: bool,
        pooling_mode: PoolingMode,
    ) -> None:
        use_cpu = True
        mixed_B = True
        self.execute_backward_sgd_(
            T,
            D,
            B,
            log_E,
            L,
            weights_precision,
            weighted,
            mixed,
            mixed_B if not use_cpu else False,
            use_cache,
            cache_algorithm,
            long_segments,
            pooling_mode,
            use_cpu,
            SparseType.FP32,  # output_dtype
        )

    @given(
        D=st.integers(min_value=2, max_value=10),
        # 128 * 1024 is to exercise a case num_ctas_for_run needs to be capped
        # at the number of SMs (H100 SXM5 has 132 SMs and the default seglen
        # per CTA is 1024)
        B=st.sampled_from([1152, 256 * 1024]),
        L=st.integers(min_value=1, max_value=4),
        weighted=st.booleans(),
        mixed=st.booleans(),
        mixed_B=st.booleans(),
        use_cache=st.booleans(),
        cache_algorithm=st.sampled_from(CacheAlgorithm),
    )
    @settings(
        verbosity=VERBOSITY,
        max_examples=MAX_EXAMPLES_LONG_RUNNING,
        deadline=None,
    )
    @unittest.skipIf(*gpu_unavailable)
    def test_backward_sgd_really_long_segments(  # noqa C901
        self,
        D: int,
        B: int,
        L: int,
        weighted: bool,
        mixed: bool,
        mixed_B: bool,
        use_cache: bool,
        cache_algorithm: CacheAlgorithm,
    ) -> None:
        self.execute_backward_sgd_(
            2,  # T
            D,
            B,
            1,  # log_E,
            L,
            SparseType.FP32,  # weights_precision
            weighted,
            mixed,
            mixed_B,
            use_cache,
            cache_algorithm,
            True,  # long_segments
            PoolingMode.SUM,  # pooling_mode
            False,  # use_cpu
            SparseType.FP32,  # output_dtype
        )

    @unittest.skipIf(
        running_on_github and torch.version.hip is not None,
        "Test is flaky on GitHub + ROCm",
    )
    @given(
        T=st.integers(min_value=1, max_value=3),
        D=st.integers(min_value=2, max_value=256),
        B=st.integers(min_value=16, max_value=20),
        log_E=st.integers(min_value=2, max_value=5),
        L=st.integers(min_value=0, max_value=1),
        weights_precision=st.sampled_from([SparseType.FP16, SparseType.FP32]),
        weighted=st.booleans(),
        mixed=st.booleans(),
        mixed_B=st.booleans(),
        use_cache=st.booleans(),
        cache_algorithm=st.sampled_from(CacheAlgorithm),
        long_segments=st.booleans(),
        use_cpu=use_cpu_strategy(),
    )
    @settings(
        verbosity=VERBOSITY,
        max_examples=MAX_EXAMPLES,
        deadline=None,
    )
    def test_backward_sgd_writeback(  # noqa C901
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
        long_segments: bool,
        use_cpu: bool,
    ) -> None:
        """
        This function test writeback functionality on EXACT SGD optimizer, most arguments are the same as other tests, while following arguments are different:
        Args:
            L (int): number of indices per sample, this is always set to 1 for writeback features
            extra_optimizer_config (UserEnabledConfigDefinition): Set use_writeback_bwd_prehook to True to enable this functionality.

        Return:
            None
        """
        self.execute_backward_sgd_(
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
            long_segments,
            PoolingMode.NONE,
            use_cpu,
            SparseType.FP32,  # output_dtype
            use_writeback_bwd_prehook=True,
        )

    @unittest.skipIf(
        running_on_github and torch.version.hip is not None,
        "Test is flaky on GitHub + ROCm",
    )
    @given(
        T=st.integers(min_value=1, max_value=3),
        D=st.integers(min_value=2, max_value=256),
        B=st.integers(min_value=16, max_value=20),
        log_E=st.integers(min_value=2, max_value=5),
        L=st.integers(min_value=0, max_value=1),
        weights_precision=st.sampled_from([SparseType.FP16, SparseType.FP32]),
        weighted=st.booleans(),
        mixed=st.booleans(),
        mixed_B=st.booleans(),
        use_cache=st.booleans(),
        cache_algorithm=st.sampled_from(CacheAlgorithm),
        long_segments=st.booleans(),
        use_cpu=use_cpu_strategy(),
    )
    @settings(
        verbosity=VERBOSITY,
        max_examples=MAX_EXAMPLES,
        deadline=None,
    )
    def test_backward_sgd_writeback_first_feature_only(  # noqa C901
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
        long_segments: bool,
        use_cpu: bool,
    ) -> None:
        """
        This function test writeback functionality on EXACT SGD optimizer, most arguments are the same as other tests, while following arguments are different:
        Args:
            L (int): number of indices per sample, this is always set to 1 for writeback features
            extra_optimizer_config (UserEnabledConfigDefinition): Set use_writeback_bwd_prehook to True to enable this functionality.

        Return:
            None
        """
        self.execute_backward_sgd_(
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
            long_segments,
            PoolingMode.NONE,
            use_cpu,
            SparseType.FP32,  # output_dtype
            use_writeback_bwd_prehook=True,
            enable_writeback_bwd_prehook_first_feature_only=True,
        )

    @given(
        T=st.integers(min_value=1, max_value=3),
        D=st.sampled_from([2, 4, 128, 256]),
        B=st.integers(min_value=1, max_value=10),
        L=st.sampled_from([1, 20, 50]),
        weights_precision=st.sampled_from([SparseType.FP16, SparseType.FP32]),
        weighted=st.booleans(),
        mixed=st.booleans(),
        mixed_B=st.booleans(),
        use_cache=st.booleans(),
        cache_algorithm=st.sampled_from(CacheAlgorithm),
        long_segments=st.booleans(),
        pooling_mode=st.sampled_from(
            [
                PoolingMode.SUM,
                PoolingMode.MEAN,
                PoolingMode.NONE,
            ]
        ),
        use_cpu=use_cpu_strategy(),
    )
    @settings(
        verbosity=VERBOSITY,
        max_examples=MAX_EXAMPLES,
        deadline=None,
    )
    def test_backward_sgd_v1(  # noqa C901
        self,
        T: int,
        D: int,
        B: int,
        L: int,
        weights_precision: SparseType,
        weighted: bool,
        mixed: bool,
        mixed_B: bool,
        use_cache: bool,
        cache_algorithm: CacheAlgorithm,
        long_segments: bool,
        pooling_mode: PoolingMode,
        use_cpu: bool,
    ) -> None:
        self.execute_backward_sgd_(
            T,
            D,
            B,
            3,  # log_E,
            L,
            weights_precision,
            weighted,
            mixed,
            mixed_B if not use_cpu else False,
            use_cache,
            cache_algorithm,
            long_segments,
            pooling_mode,
            use_cpu,
            SparseType.FP32,  # output_dtype
            use_api_v1=True,
        )

    @unittest.skipIf(
        running_on_github and torch.version.hip is not None,
        "Test is flaky on GitHub + ROCm",
    )
    @given(
        T=st.integers(min_value=1, max_value=3),
        D=st.integers(min_value=2, max_value=256),
        B=st.integers(min_value=16, max_value=20),
        log_E=st.integers(min_value=2, max_value=5),
        L=st.integers(min_value=0, max_value=1),
        weights_precision=st.sampled_from([SparseType.FP16, SparseType.FP32]),
        weighted=st.booleans(),
        mixed=st.booleans(),
        mixed_B=st.booleans(),
        use_cache=st.booleans(),
        cache_algorithm=st.sampled_from(CacheAlgorithm),
        long_segments=st.booleans(),
        use_cpu=use_cpu_strategy(),
    )
    @settings(
        verbosity=VERBOSITY,
        max_examples=MAX_EXAMPLES,
        deadline=None,
    )
    def test_backward_sgd_writeback_nobag(  # noqa C901
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
        long_segments: bool,
        use_cpu: bool,
    ) -> None:
        """
        This function test writeback functionality on EXACT SGD optimizer, most arguments are the same as other tests, while following arguments are different:
        Args:
            L (int): number of indices per sample, this is always set to 1 for writeback features
            extra_optimizer_config (UserEnabledConfigDefinition): Set use_writeback_bwd_prehook to True to enable this functionality.

        Return:
            None
        """
        self.execute_backward_sgd_(
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
            long_segments,
            PoolingMode.NONE,
            use_cpu,
            SparseType.FP32,  # output_dtype
            use_writeback_bwd_prehook=True,
            enable_writeback_bwd_prehook_first_feature_only=False,  # update all features in EC.
        )


if __name__ == "__main__":
    unittest.main()
