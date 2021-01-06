#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest
from typing import Any, Callable, List, Optional, Tuple

import hypothesis.strategies as st
import numpy as np
import split_table_batched_embeddings_ops
import torch
from hypothesis import Verbosity, assume, given, settings
from split_table_batched_embeddings_ops import OptimType


MAX_EXAMPLES = 40


def div_round_up(a: int, b: int) -> int:
    return int((a + b - 1) // b) * b


def get_offsets_from_dense(indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    (B, L) = indices.size()
    return (
        indices.contiguous().view(-1),
        torch.tensor(
            np.cumsum(np.asarray([0] + [L for _ in range(B)])[:-1]).astype(np.int64)
        ),
    )


def to_device(t: torch.Tensor, use_cpu: bool):
    return t.cpu() if use_cpu else t.cuda()


def b_indices(
    b: Callable, x: torch.Tensor, per_sample_weights=None, use_cpu=False
) -> Any:
    (indices, offsets) = get_offsets_from_dense(x)
    return b(
        to_device(indices, use_cpu),
        to_device(offsets, use_cpu),
        per_sample_weights=per_sample_weights,
    )


def get_table_batched_offsets_from_dense(
    merged_indices: torch.Tensor, use_cpu=False
) -> Tuple[torch.Tensor, torch.Tensor]:
    (T, B, L) = merged_indices.size()
    lengths = np.ones((T, B)) * L
    flat_lengths = lengths.flatten()
    return (
        to_device(merged_indices.contiguous().view(-1), use_cpu),
        to_device(
            torch.tensor(([0] + np.cumsum(flat_lengths).tolist())).long(),
            use_cpu,
        ),
    )


def generate_requests(
    iters: int,
    B: int,
    T: int,
    L: int,
    E: int,
    # inter-batch indices reuse rate
    reuse: float = 0.0,
    # alpha <= 1.0: use uniform distribution
    # alpha > 1.0: use zjpf distribution
    alpha: float = 1.0,
    fp16: bool = False,
    weighted: bool = False,
) -> List[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
    if alpha <= 1.0:
        all_indices = torch.randint(
            low=0,
            high=E,
            size=(iters, T, B * L),
            device=torch.cuda.current_device(),
            dtype=torch.int32,
        )
    else:
        all_indices = (
            torch.as_tensor(np.random.zipf(a=alpha, size=(iters, T, B * L)))
            .to(torch.cuda.current_device())
            .int()
            % E
        )
    for it in range(iters - 1):
        for t in range(T):
            reused_indices = torch.randperm(B * L, device=torch.cuda.current_device())[
                : int(B * L * reuse)
            ]
            all_indices[it + 1, t, reused_indices] = all_indices[it, t, reused_indices]

    rs = [
        get_table_batched_offsets_from_dense(all_indices[it].view(T, B, L)) + (
            torch.randn(
                T * B * L,
                device=torch.cuda.current_device(),
                dtype=torch.float16 if fp16 else torch.float32,
            )
            if weighted
            else None,
        )
        for it in range(iters)
    ]
    # pyre-fixme[7]
    return rs


@unittest.skipIf(not torch.cuda.is_available(), "Skip when CUDA is not available")
class SplitTableBatchedEmbeddingsTest(unittest.TestCase):
    @given(
        T=st.integers(min_value=1, max_value=10),
        D=st.integers(min_value=2, max_value=128),
        B=st.integers(min_value=1, max_value=128),
        log_E=st.integers(min_value=3, max_value=5),
        L=st.integers(min_value=0, max_value=20),
        fp16=st.booleans(),
        weighted=st.booleans(),
        mixed=st.booleans(),
        use_cache=st.booleans(),
        cache_algorithm=st.sampled_from(
            split_table_batched_embeddings_ops.CacheAlgorithm
        ),
        pooling_mode=st.sampled_from(
            split_table_batched_embeddings_ops.PoolingMode
        ),
        use_cpu=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_forward(
        self,
        T,
        D,
        B,
        log_E,
        L,
        fp16,
        weighted,
        mixed,
        use_cache,
        cache_algorithm,
        pooling_mode,
        use_cpu,
    ):
        # NOTE: cache is not applicable to CPU version.
        assume(not use_cpu or not use_cache)
        # NOTE: limit (T * B * L * D) to avoid timeout for CPU version!
        assume(not use_cpu or T * B * L * D <= 2048)

        assume(
            pooling_mode == split_table_batched_embeddings_ops.PoolingMode.SUM
            or not weighted
        )
        mode = (
            "sum"
            if pooling_mode == split_table_batched_embeddings_ops.PoolingMode.SUM
            else "mean"
        )

        E = int(10 ** log_E)
        if use_cpu:
            D = (D + 15) // 16 * 4
        else:
            D = D * 4
        if not mixed:
            Ds = [D] * T
            Es = [int(1e4)] * T
        else:
            Ds = [
                div_round_up(np.random.randint(low=int(0.5 * D), high=int(1.5 * D)), 4)
                for _ in range(T)
            ]
            Es = [
                np.random.randint(low=int(0.5 * E), high=int(2.0 * E)) for _ in range(T)
            ]
        compute_device = split_table_batched_embeddings_ops.ComputeDevice.CUDA
        if use_cpu:
            managed = [
                split_table_batched_embeddings_ops.EmbeddingLocation.HOST
            ] * T
            compute_device = split_table_batched_embeddings_ops.ComputeDevice.CPU
        elif use_cache:
            managed = [
                split_table_batched_embeddings_ops.EmbeddingLocation.MANAGED_CACHING
            ] * T
            if mixed:
                average_D = sum(Ds) // T
                for t, d in enumerate(Ds):
                    managed[t] = (
                        split_table_batched_embeddings_ops.EmbeddingLocation.DEVICE
                        if d < average_D
                        else managed[t]
                    )
        else:
            managed = [
                np.random.choice(
                    [
                        split_table_batched_embeddings_ops.EmbeddingLocation.DEVICE,
                        split_table_batched_embeddings_ops.EmbeddingLocation.MANAGED,
                    ]
                )
                for _ in range(T)
            ]
        bs = [
            to_device(torch.nn.EmbeddingBag(E, D, mode=mode, sparse=True), use_cpu)
            for (E, D) in zip(Es, Ds)
        ]
        if fp16 and not use_cpu:
            # NOTE: CPU version of torch.nn.EmbeddingBag doesn't support fp16.
            bs = [b.half() for b in bs]

        xs = [to_device(torch.randint(low=0, high=e, size=(B, L)), use_cpu) for e in Es]
        xws = [to_device(torch.randn(size=(B, L)), use_cpu) for _ in range(T)]
        xws_acc_type = copy.deepcopy(xws)

        if fp16 and not use_cpu:
            # NOTE: CPU version of torch.nn.EmbeddingBag doesn't support fp16.
            xws = [xw.half() for xw in xws]

        fs = (
            [b_indices(b, x, use_cpu=use_cpu) for (b, x) in zip(bs, xs)]
            if not weighted
            else [
                b_indices(b, x, per_sample_weights=xw.view(-1), use_cpu=use_cpu)
                for (b, x, xw) in zip(bs, xs, xws)
            ]
        )
        f = torch.cat([f.view(B, -1) for f in fs], dim=1)

        cc = split_table_batched_embeddings_ops.SplitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                (
                    E,
                    D,
                    split_table_batched_embeddings_ops.EmbeddingLocation(M),
                    compute_device,
                )
                for (E, D, M) in zip(Es, Ds, managed)
            ],
            fp16=fp16,
            optimizer=OptimType.EXACT_SGD,
            learning_rate=0.05,
            cache_algorithm=cache_algorithm,
            pooling_mode=pooling_mode,
        )
        # NOTE: test TorchScript-compatible!
        cc = torch.jit.script(cc)

        for t in range(T):
            cc.split_embedding_weights()[t].data.copy_(bs[t].weight)

        x = torch.cat([x.view(1, B, L) for x in xs], dim=0)
        xw = torch.cat([xw.view(1, B, L) for xw in xws_acc_type], dim=0)

        (indices, offsets) = get_table_batched_offsets_from_dense(x, use_cpu)
        fc2 = (
            cc(indices, offsets)
            if not weighted
            else cc(indices, offsets, to_device(xw.contiguous().view(-1), use_cpu))
        )
        torch.testing.assert_allclose(
            fc2.float(),
            f.float(),
            atol=8.0e-3 if fp16 else 1.0e-5,
            rtol=8.0e-3 if fp16 else 1.0e-5,
        )

    @given(
        T=st.integers(min_value=1, max_value=3),
        D=st.integers(min_value=2, max_value=128),
        B=st.integers(min_value=1, max_value=32),
        log_E=st.integers(min_value=3, max_value=5),
        L=st.integers(min_value=0, max_value=10),
        fp16=st.booleans(),
        weighted=st.booleans(),
        mixed=st.booleans(),
        long_segments=st.booleans(),
        pooling_mode=st.sampled_from(
            split_table_batched_embeddings_ops.PoolingMode
        ),
        use_cpu=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=10, deadline=None)
    def test_backward_dense(
        self,
        T,
        D,
        B,
        log_E,
        L,
        fp16,
        weighted,
        mixed,
        long_segments,
        pooling_mode,
        use_cpu,
    ):
        # NOTE: torch.autograd.gradcheck() is too time-consuming for CPU version
        #       so we have to limit (T * B * L * D)!
        assume(not use_cpu or T * B * L * D <= 2048)
        assume(
            pooling_mode == split_table_batched_embeddings_ops.PoolingMode.SUM
            or not weighted
        )
        mode = (
            "sum"
            if pooling_mode == split_table_batched_embeddings_ops.PoolingMode.SUM
            else "mean"
        )

        E = int(10 ** log_E)
        if use_cpu:
            D = (D + 15) // 16 * 4
        else:
            D = D * 4
        if not mixed:
            Ds = [D] * T
            Es = [E] * T
        else:
            Ds = [
                div_round_up(np.random.randint(low=int(0.5 * D), high=int(1.5 * D)), 4)
                for _ in range(T)
            ]
            Es = [
                np.random.randint(low=int(0.5 * E), high=int(2 * E)) for _ in range(T)
            ]
        bs = [
            to_device(torch.nn.EmbeddingBag(E, D, mode=mode, sparse=False), use_cpu)
            for (E, D) in zip(Es, Ds)
        ]

        if fp16 and not use_cpu:
            # NOTE: CPU version of torch.nn.EmbeddingBag doesn't support fp16.
            bs = [b.half() for b in bs]

        xs = [
            to_device(
                torch.from_numpy(
                    np.random.choice(range(e), size=(B, L), replace=True).astype(
                        np.int64
                    )
                ),
                use_cpu,
            )
            for e in Es
        ]
        if long_segments and L > 0 and not fp16:
            for x in xs:
                x[:, 0] = 0

        xws = [to_device(torch.randn(size=(B, L)), use_cpu) for _ in range(T)]
        xws_acc_type = copy.deepcopy(xws)

        if fp16 and not use_cpu:
            # NOTE: CPU version of torch.nn.EmbeddingBag doesn't support fp16.
            xws = [xw.half() for xw in xws]

        fs = (
            [b_indices(b, x, use_cpu=use_cpu) for (b, x) in zip(bs, xs)]
            if not weighted
            else [
                b_indices(b, x, per_sample_weights=xw.view(-1), use_cpu=use_cpu)
                for (b, x, xw) in zip(bs, xs, xws)
            ]
        )
        gos = [torch.randn_like(f) for f in fs]
        [f.backward(go) for (f, go) in zip(fs, gos)]

        grad_weights = torch.cat([b.weight.grad.view(-1) for b in bs])
        if fp16:
            grad_weights = grad_weights.half()

        cc = split_table_batched_embeddings_ops.DenseTableBatchedEmbeddingBagsCodegen(
            [(E, D) for (E, D) in zip(Es, Ds)],
            pooling_mode=pooling_mode,
            use_cpu=use_cpu,
        )
        if fp16:
            cc = cc.half()
        # NOTE: test TorchScript-compatible!
        cc = torch.jit.script(cc)

        for t in range(T):
            cc.split_embedding_weights()[t].data.copy_(bs[t].weight)

        x = torch.cat([x.view(1, B, L) for x in xs], dim=0)
        xw = torch.cat([xw.view(1, B, L) for xw in xws_acc_type], dim=0)

        (indices, offsets) = get_table_batched_offsets_from_dense(x, use_cpu)
        fc2 = (
            cc(indices, offsets)
            if not weighted
            else cc(indices, offsets, to_device(xw.contiguous().view(-1), use_cpu))
        )
        f = torch.cat([f.view(B, -1) for f in fs], dim=1)
        torch.testing.assert_allclose(
            fc2.float(),
            f.float(),
            atol=5.0e-3 if fp16 else 1.0e-5,
            rtol=5.0e-3 if fp16 else 1.0e-5,
        )

        goc = torch.cat([go.view(B, -1) for go in gos], dim=1).contiguous()
        fc2.backward(goc)
        torch.testing.assert_allclose(
            cc.weights.grad,
            grad_weights,
            atol=5.0e-3 if fp16 else 1.0e-4,
            rtol=5.0e-3 if fp16 else 1.0e-4,
        )

        cc = split_table_batched_embeddings_ops.DenseTableBatchedEmbeddingBagsCodegen(
            [(E, D) for (E, D) in zip(Es, Ds)],
            # NOTE: only SUM pooling can work with per_sample_weights!
            pooling_mode=split_table_batched_embeddings_ops.PoolingMode.SUM,
            use_cpu=use_cpu,
        ).double()
        per_sample_weights = to_device(xw.contiguous().view(-1), use_cpu).double()
        per_sample_weights.requires_grad = True
        indices.requires_grad = False
        offsets.requires_grad = False
        for param in cc.parameters():
            param.requires_grad = False
        torch.autograd.gradcheck(cc, (indices, offsets, per_sample_weights))

    @given(
        T=st.integers(min_value=1, max_value=5),
        D=st.integers(min_value=2, max_value=128),
        B=st.integers(min_value=1, max_value=128),
        log_E=st.integers(min_value=3, max_value=5),
        L=st.integers(min_value=0, max_value=20),
        fp16=st.booleans(),
        weighted=st.booleans(),
        exact=st.booleans(),
        mixed=st.booleans(),
        use_cache=st.booleans(),
        cache_algorithm=st.sampled_from(
            split_table_batched_embeddings_ops.CacheAlgorithm
        ),
        long_segments=st.booleans(),
        pooling_mode=st.sampled_from(
            split_table_batched_embeddings_ops.PoolingMode
        ),
        use_cpu=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_backward_sgd(  # noqa C901
        self,
        T,
        D,
        B,
        log_E,
        L,
        fp16,
        weighted,
        exact,
        mixed,
        use_cache,
        cache_algorithm,
        long_segments,
        pooling_mode,
        use_cpu,
    ):
        # NOTE: cache is not applicable to CPU version.
        assume(not use_cpu or not use_cache)
        # NOTE: limit (T * B * L * D) to avoid timeout for CPU version!
        assume(not use_cpu or T * B * L * D <= 2048)

        assume(
            pooling_mode == split_table_batched_embeddings_ops.PoolingMode.SUM
            or not weighted
        )
        mode = (
            "sum"
            if pooling_mode == split_table_batched_embeddings_ops.PoolingMode.SUM
            else "mean"
        )

        # only non-exact supports caching
        assume(not exact or not use_cache)
        E = int(10 ** log_E)
        if use_cpu:
            D = (D + 15) // 16 * 4
        else:
            D = D * 4
        if not mixed:
            Ds = [D] * T
            Es = [E] * T
        else:
            Ds = [
                div_round_up(np.random.randint(low=int(0.5 * D), high=int(1.5 * D)), 4)
                for _ in range(T)
            ]
            Es = [
                np.random.randint(low=int(0.5 * E), high=int(2.0 * E)) for _ in range(T)
            ]
        compute_device = split_table_batched_embeddings_ops.ComputeDevice.CUDA
        if use_cpu:
            managed = [
                split_table_batched_embeddings_ops.EmbeddingLocation.HOST
            ] * T
            compute_device = split_table_batched_embeddings_ops.ComputeDevice.CPU
        elif use_cache:
            managed = [
                split_table_batched_embeddings_ops.EmbeddingLocation.MANAGED_CACHING
            ] * T
            if mixed:
                average_D = sum(Ds) // T
                for t, d in enumerate(Ds):
                    managed[t] = (
                        split_table_batched_embeddings_ops.EmbeddingLocation.DEVICE
                        if d < average_D
                        else managed[t]
                    )
        else:
            managed = [
                np.random.choice(
                    [
                        split_table_batched_embeddings_ops.EmbeddingLocation.DEVICE,
                        split_table_batched_embeddings_ops.EmbeddingLocation.MANAGED,
                    ]
                )
                for _ in range(T)
            ]
        bs = [
            to_device(torch.nn.EmbeddingBag(E, D, mode=mode, sparse=True), use_cpu)
            for (E, D) in zip(Es, Ds)
        ]

        if fp16 and not use_cpu:
            # NOTE: CPU version of torch.nn.EmbeddingBag doesn't support fp16.
            bs = [b.half() for b in bs]

        feature_table_map = list(range(T))
        table_to_replicate = T // 2
        bs.insert(table_to_replicate, bs[table_to_replicate])
        feature_table_map.insert(table_to_replicate, table_to_replicate)

        xs = [
            to_device(torch.from_numpy(
                np.random.choice(range(Es[t]), size=(B, L), replace=True).astype(
                    np.int64
                )
            ), use_cpu)
            for t in feature_table_map
        ]

        if long_segments and L > 0:
            for x in xs:
                x[:, 0] = 0

        xws = [to_device(torch.randn(size=(B, L)), use_cpu) for _ in range(len(xs))]
        xws_acc_type = copy.deepcopy(xws)

        if fp16 and not use_cpu:
            # NOTE: CPU version of torch.nn.EmbeddingBag doesn't support fp16.
            xws = [xw.half() for xw in xws]

        fs = (
            [b_indices(b, x, use_cpu=use_cpu) for (b, x) in zip(bs, xs)]
            if not weighted
            else [
                b_indices(b, x, per_sample_weights=xw.view(-1), use_cpu=use_cpu)
                for (b, x, xw) in zip(bs, xs, xws)
            ]
        )
        gos = [torch.randn_like(f) for f in fs]
        [f.backward(go) for (f, go) in zip(fs, gos)]
        # do SGD update
        lr = 0.05
        del bs[table_to_replicate]
        new_weights = [(b.weight - b.weight.grad * lr) for b in bs]

        cc = split_table_batched_embeddings_ops.SplitTableBatchedEmbeddingBagsCodegen(
            [(E, D, M, compute_device) for (E, D, M) in zip(Es, Ds, managed)],
            optimizer=OptimType.EXACT_SGD,
            feature_table_map=feature_table_map,
            learning_rate=0.05,
            fp16=fp16,
            cache_algorithm=cache_algorithm,
            pooling_mode=pooling_mode,
        )

        for t in range(T):
            cc.split_embedding_weights()[t].data.copy_(bs[t].weight)

        x = torch.cat([x.view(1, B, L) for x in xs], dim=0)
        xw = torch.cat([xw.view(1, B, L) for xw in xws_acc_type], dim=0)

        (indices, offsets) = get_table_batched_offsets_from_dense(x, use_cpu)
        fc2 = (
            cc(indices, offsets)
            if not weighted
            else cc(indices, offsets, to_device(xw.contiguous().view(-1), use_cpu))
        )
        goc = torch.cat([go.view(B, -1) for go in gos], dim=1).contiguous()
        fc2.backward(goc)
        if use_cache:
            cc.flush()
        for t in range(T):
            torch.testing.assert_allclose(
                cc.split_embedding_weights()[t],
                new_weights[t].half() if fp16 and use_cpu else new_weights[t],
                atol=(1.0e-2 if long_segments else 5.0e-3) if fp16 else 1.0e-5,
                rtol=(1.0e-2 if long_segments else 5.0e-3) if fp16 else 1.0e-5,
            )

    @given(
        T=st.integers(min_value=1, max_value=5),
        D=st.integers(min_value=2, max_value=128),
        B=st.integers(min_value=1, max_value=128),
        log_E=st.integers(min_value=3, max_value=5),
        L=st.integers(min_value=0, max_value=20),
        D_gradcheck=st.integers(min_value=1, max_value=2),
        fp16=st.booleans(),
        stochastic_rounding=st.booleans(),
        weighted=st.booleans(),
        row_wise=st.booleans(),
        exact=st.booleans(),
        mixed=st.booleans(),
        use_cache=st.booleans(),
        cache_algorithm=st.sampled_from(
            split_table_batched_embeddings_ops.CacheAlgorithm
        ),
        pooling_mode=st.sampled_from(
            split_table_batched_embeddings_ops.PoolingMode
        ),
        use_cpu=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_backward_adagrad(  # noqa C901
        self,
        T,
        D,
        B,
        log_E,
        L,
        D_gradcheck,
        fp16,
        stochastic_rounding,
        weighted,
        row_wise,
        exact,
        mixed,
        use_cache,
        cache_algorithm,
        pooling_mode,
        use_cpu,
    ):
        # NOTE: cache is not applicable to CPU version.
        assume(not use_cpu or not use_cache)

        # NOTE: torch.autograd.gradcheck() is too time-consuming for CPU version
        #       so we have to limit (T * B * L * D)!
        assume(not use_cpu or T * B * L * D <= 1024)

        assume(
            pooling_mode == split_table_batched_embeddings_ops.PoolingMode.SUM
            or not weighted
        )
        mode = (
            "sum"
            if pooling_mode == split_table_batched_embeddings_ops.PoolingMode.SUM
            else "mean"
        )

        # stochastic rounding only implemented for rowwise
        assume(not stochastic_rounding or row_wise)
        # exact only implemented for rowwise non-weighted
        assume(not exact or (row_wise and not weighted))
        # need unique indices for non-exact tests
        assume(exact or int(10 ** log_E) > int(2.1 * B * L))
        # only row-wise supports caching
        assume(row_wise or not use_cache)

        E = int(10 ** log_E)
        if use_cpu:
            D = (D + 15) // 16 * 4
        else:
            D = D * 4
        if not mixed:
            Ds = [D] * T
            Es = [E] * T
        else:
            Ds = [
                div_round_up(np.random.randint(low=int(0.5 * D), high=int(1.5 * D)), 4)
                for _ in range(T)
            ]
            Es = [
                np.random.randint(low=int(0.5 * E), high=int(2.0 * E)) for _ in range(T)
            ]
        compute_device = split_table_batched_embeddings_ops.ComputeDevice.CUDA
        if use_cpu:
            managed = [
                split_table_batched_embeddings_ops.EmbeddingLocation.HOST
            ] * T
            compute_device = split_table_batched_embeddings_ops.ComputeDevice.CPU
        elif use_cache:
            managed = [
                split_table_batched_embeddings_ops.EmbeddingLocation.MANAGED_CACHING
            ] * T
            if mixed:
                average_D = sum(Ds) // T
                for t, d in enumerate(Ds):
                    managed[t] = (
                        split_table_batched_embeddings_ops.EmbeddingLocation.DEVICE
                        if d < average_D
                        else managed[t]
                    )
        else:
            managed = [
                np.random.choice(
                    [
                        split_table_batched_embeddings_ops.EmbeddingLocation.DEVICE,
                        split_table_batched_embeddings_ops.EmbeddingLocation.MANAGED,
                    ]
                )
                for _ in range(T)
            ]
        bs = [
            to_device(torch.nn.EmbeddingBag(E, D, mode=mode, sparse=True), use_cpu)
            for (E, D) in zip(Es, Ds)
        ]

        if fp16 and not use_cpu:
            # NOTE: CPU version of torch.nn.EmbeddingBag doesn't support fp16.
            bs = [b.half() for b in bs]

        feature_table_map = list(range(T))
        if exact:
            # autograd with shared embedding only works for exact
            table_to_replicate = T // 2
            bs.insert(table_to_replicate, bs[table_to_replicate])
            feature_table_map.insert(table_to_replicate, table_to_replicate)

        xs = [
            to_device(torch.from_numpy(
                np.random.choice(range(Es[t]), size=(B, L), replace=exact).astype(
                    np.int64
                )
            ), use_cpu)
            for t in feature_table_map
        ]
        xws = [to_device(torch.randn(size=(B, L)), use_cpu) for _ in range(len(xs))]
        xws_acc_type = copy.deepcopy(xws)

        if fp16 and not use_cpu:
            # NOTE: CPU version of torch.nn.EmbeddingBag doesn't support fp16.
            xws = [xw.half() for xw in xws]

        fs = (
            [b_indices(b, x, use_cpu=use_cpu) for (b, x) in zip(bs, xs)]
            if not weighted
            else [
                b_indices(b, x, per_sample_weights=xw.view(-1), use_cpu=use_cpu)
                for (b, x, xw) in zip(bs, xs, xws)
            ]
        )
        gos = [torch.randn_like(f) for f in fs]
        [f.backward(go) for (f, go) in zip(fs, gos)]
        # do SGD update
        lr = 0.5
        eps = 0.2

        cc = split_table_batched_embeddings_ops.SplitTableBatchedEmbeddingBagsCodegen(
            [(E, D, M, compute_device) for (E, D, M) in zip(Es, Ds, managed)],
            feature_table_map=feature_table_map,
            optimizer=OptimType.EXACT_ROWWISE_ADAGRAD
            if row_wise
            else OptimType.EXACT_ADAGRAD,
            learning_rate=lr,
            eps=eps,
            fp16=fp16,
            stochastic_rounding=stochastic_rounding,
            pooling_mode=pooling_mode,
        )

        if exact:
            del bs[table_to_replicate]
        for t in range(T):
            cc.split_embedding_weights()[t].data.copy_(bs[t].weight)

        x = torch.cat([x.view(1, B, L) for x in xs], dim=0)
        xw = torch.cat([xw.view(1, B, L) for xw in xws_acc_type], dim=0)

        (indices, offsets) = get_table_batched_offsets_from_dense(x, use_cpu)
        fc2 = (
            cc(indices, offsets)
            if not weighted
            else cc(indices, offsets, to_device(xw.contiguous().view(-1), use_cpu))
        )
        fc2.backward(torch.cat([go.view(B, -1) for go in gos], dim=1))
        cc.flush()

        split_optimizer_states = [s for (s,) in cc.split_optimizer_states()]
        for t in range(T):
            ref_optimizer_state = bs[t].weight.grad.float().to_dense().pow(2)
            torch.testing.assert_allclose(
                split_optimizer_states[t].float(),
                ref_optimizer_state.mean(dim=1) if row_wise else ref_optimizer_state,
                atol=5.0e-3 if fp16 else 1.0e-4,
                rtol=5.0e-3 if fp16 else 1.0e-4,
            )

        for t in range(T):
            # optimizer_state = squares (no row-wise) or sum squares (row-wise)
            torch.testing.assert_allclose(
                cc.split_embedding_weights()[t].float(),
                torch.addcdiv(
                    bs[t].weight.float(),
                    value=-lr,
                    tensor1=bs[t].weight.grad.float().to_dense(),
                    tensor2=split_optimizer_states[t]
                    .float()
                    .sqrt_()
                    .add_(eps)
                    .view(Es[t], 1 if row_wise else Ds[t]),
                ),
                atol=5.0e-3 if fp16 else 1.0e-4,
                rtol=5.0e-3 if fp16 else 1.0e-4,
            )
        if use_cpu:
            D_gradcheck = (D_gradcheck + 15) // 16 * 4
        else:
            D_gradcheck = D_gradcheck * 4
        cc = split_table_batched_embeddings_ops.SplitTableBatchedEmbeddingBagsCodegen(
            [(E, D_gradcheck, M, compute_device) for (E, M) in zip(Es, managed)],
            feature_table_map=feature_table_map,
            optimizer=OptimType.EXACT_ROWWISE_ADAGRAD
            if row_wise
            else OptimType.EXACT_ADAGRAD,
            learning_rate=0.0,
            eps=eps,
            fp16=fp16,
            stochastic_rounding=stochastic_rounding,
            # NOTE: only SUM pooling can work with per_sample_weights!
            pooling_mode=split_table_batched_embeddings_ops.PoolingMode.SUM,
        )
        if use_cpu:
            # NOTE: GPU version of SplitTableBatchedEmbeddingBagsCodegen doesn't support double.
            cc = cc.double()

        per_sample_weights = to_device(xw.contiguous().view(-1), use_cpu)
        if use_cpu:
            per_sample_weights = per_sample_weights.double()
        per_sample_weights.requires_grad = True
        indices.requires_grad = False
        offsets.requires_grad = False
        for param in cc.parameters():
            param.requires_grad = False
        torch.autograd.gradcheck(cc, (indices, offsets, per_sample_weights))

        per_sample_weights = to_device(xw.contiguous().view(-1), use_cpu)
        if use_cpu:
            per_sample_weights = per_sample_weights.double()
        per_sample_weights.requires_grad = True
        indices.requires_grad = False
        offsets.requires_grad = False
        for param in cc.parameters():
            param.requires_grad = False
        y = cc(indices, offsets, per_sample_weights)
        y.sum().backward()
        indice_weight_grad_all = per_sample_weights.grad.clone().cpu()
        T_ = len(xws)
        feature_requires_grad = to_device(
            torch.tensor(np.random.choice([0, 1], replace=True, size=(T_,))).int(), use_cpu
        )
        per_sample_weights = per_sample_weights.detach().clone()
        per_sample_weights.requires_grad = True
        y = cc(
            indices,
            offsets,
            per_sample_weights,
            feature_requires_grad=feature_requires_grad,
        )
        y.sum().backward()
        indice_weight_grad_mask = per_sample_weights.grad.clone().cpu()
        for t in range(T_):
            if feature_requires_grad[t]:
                torch.testing.assert_allclose(
                    indice_weight_grad_mask.view(T_, B, L)[t],
                    indice_weight_grad_all.view(T_, B, L)[t],
                )
            else:
                torch.testing.assert_allclose(
                    indice_weight_grad_mask.view(T_, B, L)[t],
                    torch.zeros_like(indice_weight_grad_mask.view(T_, B, L)[t]),
                )

    @given(
        T=st.integers(min_value=1, max_value=5),
        D=st.integers(min_value=2, max_value=64),
        B=st.integers(min_value=1, max_value=128),
        log_E=st.integers(min_value=3, max_value=5),
        L=st.integers(min_value=1, max_value=20),
        mixed=st.booleans(),
        cache_algorithm=st.sampled_from(
            split_table_batched_embeddings_ops.CacheAlgorithm
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_cache_pipeline(
        self, T, D, B, log_E, L, mixed, cache_algorithm
    ):
        iters = 3
        E = int(10 ** log_E)
        D = D * 4
        if not mixed:
            Ds = [D] * T
            Es = [E] * T
        else:
            Ds = [
                div_round_up(np.random.randint(low=int(0.5 * D), high=int(1.5 * D)), 4)
                for _ in range(T)
            ]
            Es = [
                np.random.randint(low=int(0.5 * E), high=int(2.0 * E)) for _ in range(T)
            ]
        managed = [
            split_table_batched_embeddings_ops.EmbeddingLocation.MANAGED_CACHING
        ] * T
        if mixed:
            average_D = sum(Ds) // T
            for t, d in enumerate(Ds):
                managed[t] = (
                    split_table_batched_embeddings_ops.EmbeddingLocation.DEVICE
                    if d < average_D
                    else managed[t]
                )
        cc_ref = (
            split_table_batched_embeddings_ops.SplitTableBatchedEmbeddingBagsCodegen(
                [
                    (
                        E,
                        D,
                        split_table_batched_embeddings_ops.EmbeddingLocation.DEVICE,
                        split_table_batched_embeddings_ops.ComputeDevice.CUDA,
                    )
                    for (E, D) in zip(Es, Ds)
                ],
            )
        )
        cc = split_table_batched_embeddings_ops.SplitTableBatchedEmbeddingBagsCodegen(
            [(E, D, managed, split_table_batched_embeddings_ops.ComputeDevice.CUDA) for (E, D) in zip(Es, Ds)],
            cache_algorithm=cache_algorithm,
        )
        for t in range(T):
            assert (
                cc.split_embedding_weights()[t].size()
                == cc_ref.split_embedding_weights()[t].size()
            )
            cc.split_embedding_weights()[t].data.copy_(
                cc_ref.split_embedding_weights()[t]
            )

        requests = generate_requests(iters, B, T, L, min(Es), reuse=0.1)
        grad_output = torch.randn(B, sum(Ds)).cuda()

        for indices, offsets, _ in requests:
            output = cc(indices, offsets)
            output_ref = cc_ref(indices, offsets)
            torch.testing.assert_allclose(output, output_ref)
            output.backward(grad_output)
            output_ref.backward(grad_output)
        cc.flush()
        for t in range(T):
            torch.testing.assert_allclose(
                cc.split_embedding_weights()[t], cc_ref.split_embedding_weights()[t]
            )

    @given(
        T=st.integers(min_value=1, max_value=5),
        D=st.integers(min_value=2, max_value=128),
        B=st.integers(min_value=1, max_value=128),
        log_E=st.integers(min_value=3, max_value=5),
        L=st.integers(min_value=0, max_value=20),
        stochastic_rounding=st.booleans(),
        weighted=st.booleans(),
        mixed=st.booleans(),
        optimizer=st.sampled_from(
            [
                OptimType.ADAM,
                OptimType.EXACT_ADAGRAD,
                OptimType.EXACT_ROWWISE_ADAGRAD,
                OptimType.LAMB,
                OptimType.LARS_SGD,
                OptimType.PARTIAL_ROWWISE_ADAM,
                OptimType.PARTIAL_ROWWISE_LAMB,
            ]
        ),
        long_segments=st.booleans(),
        pooling_mode=st.sampled_from(
            split_table_batched_embeddings_ops.PoolingMode
        ),
        use_cpu=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_backward_optimizers(  # noqa C901
        self,
        T,
        D,
        B,
        log_E,
        L,
        stochastic_rounding,
        weighted,
        mixed,
        optimizer,
        long_segments,
        pooling_mode,
        use_cpu,
    ):
        # NOTE: limit (T * B * L * D) to avoid timeout for CPU version!
        assume(not use_cpu or T * B * L * D <= 2048)

        assume(
            pooling_mode == split_table_batched_embeddings_ops.PoolingMode.SUM
            or not weighted
        )
        mode = (
            "sum"
            if pooling_mode == split_table_batched_embeddings_ops.PoolingMode.SUM
            else "mean"
        )

        E = int(10 ** log_E)
        if use_cpu:
            D = (D + 15) // 16 * 4
        else:
            D = D * 4
        if not mixed:
            Ds = [D] * T
            Es = [E] * T
        else:
            Ds = [
                div_round_up(np.random.randint(low=int(0.5 * D), high=int(1.5 * D)), 4)
                for _ in range(T)
            ]
            Es = [
                np.random.randint(low=int(0.5 * E), high=int(2.0 * E)) for _ in range(T)
            ]
        compute_device = split_table_batched_embeddings_ops.ComputeDevice.CUDA
        if use_cpu:
            managed = [
                split_table_batched_embeddings_ops.EmbeddingLocation.HOST
            ] * T
            compute_device = split_table_batched_embeddings_ops.ComputeDevice.CPU
        else:
            managed = [
                np.random.choice(
                    [
                        split_table_batched_embeddings_ops.EmbeddingLocation.DEVICE,
                        split_table_batched_embeddings_ops.EmbeddingLocation.MANAGED,
                    ]
                )
                for _ in range(T)
            ]
        bs = [
            to_device(torch.nn.EmbeddingBag(E, D, mode=mode, sparse=True), use_cpu)
            for (E, D) in zip(Es, Ds)
        ]

        xs = [
            to_device(torch.from_numpy(
                np.random.choice(range(e), size=(B, L), replace=True).astype(np.int64)
            ), use_cpu)
            for e in Es
        ]
        if long_segments and L > 0:
            for x in xs:
                x[:, 0] = 0

        xws = [to_device(torch.randn(size=(B, L)), use_cpu) for _ in range(T)]
        xws_acc_type = copy.deepcopy(xws)

        fs = (
            [b_indices(b, x, use_cpu=use_cpu) for (b, x) in zip(bs, xs)]
            if not weighted
            else [
                b_indices(b, x, per_sample_weights=xw.view(-1), use_cpu=use_cpu)
                for (b, x, xw) in zip(bs, xs, xws)
            ]
        )
        gos = [torch.randn_like(f) for f in fs]
        [f.backward(go) for (f, go) in zip(fs, gos)]
        # do SGD update

        optimizer_kwargs = {"learning_rate": 0.5}
        (lr, eps, beta1, beta2, weight_decay, momentum, eta) = (
            0.5,
            1e-4,
            0.9,
            0.99,
            0.01,
            0.9,
            0.01,
        )
        if optimizer in (OptimType.EXACT_ROWWISE_ADAGRAD, OptimType.EXACT_ADAGRAD):
            optimizer_kwargs["eps"] = eps

        if optimizer in (OptimType.PARTIAL_ROWWISE_ADAM, OptimType.ADAM):
            optimizer_kwargs["eps"] = eps
            optimizer_kwargs["beta1"] = beta1
            optimizer_kwargs["beta2"] = beta2
            optimizer_kwargs["weight_decay"] = weight_decay

        if optimizer in (OptimType.PARTIAL_ROWWISE_LAMB, OptimType.LAMB):
            optimizer_kwargs["eps"] = eps
            optimizer_kwargs["beta1"] = beta1
            optimizer_kwargs["beta2"] = beta2
            optimizer_kwargs["weight_decay"] = weight_decay

        if optimizer == OptimType.LARS_SGD:
            optimizer_kwargs["weight_decay"] = weight_decay
            optimizer_kwargs["momentum"] = momentum
            optimizer_kwargs["eta"] = eta

        cc = split_table_batched_embeddings_ops.SplitTableBatchedEmbeddingBagsCodegen(
            [(E, D, M, compute_device) for (E, D, M) in zip(Es, Ds, managed)],
            optimizer=optimizer,
            stochastic_rounding=stochastic_rounding,
            pooling_mode=pooling_mode,
            **optimizer_kwargs,
        )

        for t in range(T):
            cc.split_embedding_weights()[t].data.copy_(bs[t].weight)

        x = torch.cat([x.view(1, B, L) for x in xs], dim=0)
        xw = torch.cat([xw.view(1, B, L) for xw in xws_acc_type], dim=0)

        (indices, offsets) = get_table_batched_offsets_from_dense(x, use_cpu)
        fc2 = (
            cc(indices, offsets)
            if not weighted
            else cc(indices, offsets, to_device(xw.contiguous().view(-1), use_cpu))
        )
        fc2.backward(torch.cat([go.view(B, -1) for go in gos], dim=1))
        cc.flush()

        split_optimizer_states = cc.split_optimizer_states()
        assert len(split_optimizer_states) == T
        split_weights = cc.split_embedding_weights()

        if optimizer in (OptimType.EXACT_ROWWISE_ADAGRAD, OptimType.EXACT_ADAGRAD):
            rowwise = optimizer == OptimType.EXACT_ROWWISE_ADAGRAD
            for t in range(T):
                (m1,) = split_optimizer_states[t]
                m1_ref = (
                    bs[t].weight.grad.to_dense().pow(2)
                    if not rowwise
                    else bs[t].weight.grad.to_dense().pow(2).mean(dim=1)
                )
                torch.testing.assert_allclose(
                    m1.float(), m1_ref.float(), atol=1.0e-4, rtol=1.0e-4
                )
                weights_new = split_weights[t]
                weights_ref = bs[t].weight - lr * bs[t].weight.grad.to_dense() / (
                    torch.sqrt(
                        m1_ref if not rowwise else m1_ref.view(m1_ref.numel(), 1)
                    )
                    + eps
                )
                # TODO: why is tolerance off here?
                torch.testing.assert_allclose(
                    weights_new.float(), weights_ref.float(), atol=1.0e-2, rtol=1.0e-2
                )

        if optimizer in (OptimType.PARTIAL_ROWWISE_ADAM, OptimType.ADAM):
            rowwise = optimizer == OptimType.PARTIAL_ROWWISE_ADAM
            for t in range(T):
                (m1, m2) = split_optimizer_states[t]
                m2_ref = (
                    bs[t].weight.grad.to_dense().pow(2)
                    if not rowwise
                    else bs[t].weight.grad.to_dense().pow(2).mean(dim=1)
                ) * (1.0 - beta2)
                torch.testing.assert_allclose(m2, m2_ref, atol=1.0e-4, rtol=1.0e-4)
                m1_ref = bs[t].weight.grad.to_dense() * (1.0 - beta1)
                torch.testing.assert_allclose(m1, m1_ref, atol=1.0e-4, rtol=1.0e-4)
                iter_ = cc.iter.item()
                v_hat_t = m2_ref / (1 - beta2 ** iter_)
                v_hat_t = v_hat_t if not rowwise else v_hat_t.view(v_hat_t.numel(), 1)
                m_hat_t = m1_ref / (1 - beta1 ** iter_)
                weights_new = split_weights[t]
                weights_ref = (
                    torch.addcdiv(
                        bs[t].weight,
                        value=-lr,
                        tensor1=m_hat_t,
                        tensor2=v_hat_t.sqrt_().add_(eps),
                    )
                    - lr * weight_decay * bs[t].weight
                )
                torch.testing.assert_allclose(
                    weights_new.index_select(dim=0, index=x[t].view(-1)),
                    weights_ref.index_select(dim=0, index=x[t].view(-1)),
                    atol=1.0e-3,
                    rtol=1.0e-3,
                )

        if optimizer in (OptimType.PARTIAL_ROWWISE_LAMB, OptimType.LAMB):
            rowwise = optimizer == OptimType.PARTIAL_ROWWISE_LAMB
            for t in range(T):
                (m1, m2) = split_optimizer_states[t]
                m2_ref = (
                    bs[t].weight.grad.to_dense().pow(2)
                    if not rowwise
                    else bs[t].weight.grad.to_dense().pow(2).mean(dim=1)
                ) * (1.0 - beta2)
                torch.testing.assert_allclose(m2, m2_ref, atol=1.0e-4, rtol=1.0e-4)
                m1_ref = bs[t].weight.grad.to_dense() * (1.0 - beta1)
                torch.testing.assert_allclose(m1, m1_ref, atol=1.0e-4, rtol=1.0e-4)
                iter_ = cc.iter.item()
                v_hat_t = m2_ref / (1 - beta2 ** iter_)
                v_hat_t = v_hat_t if not rowwise else v_hat_t.view(v_hat_t.numel(), 1)
                m_hat_t = m1_ref / (1 - beta1 ** iter_)
                rtw = (m_hat_t / (torch.sqrt(v_hat_t) + eps)) + weight_decay * bs[
                    t
                ].weight
                true_ratio = torch.linalg.norm(bs[t].weight, dim=1, ord=2).view(
                    m1.shape[0], 1
                ) / torch.linalg.norm(rtw, dim=1, ord=2).view(m1.shape[0], 1)
                weights_new = split_weights[t]
                weights_ref = bs[t].weight - lr * true_ratio * rtw
                torch.testing.assert_allclose(
                    weights_new.index_select(dim=0, index=x[t].view(-1)),
                    weights_ref.index_select(dim=0, index=x[t].view(-1)),
                    atol=1.0e-3,
                    rtol=1.0e-3,
                )

        if optimizer == OptimType.LARS_SGD:
            for t in range(T):
                (m1,) = split_optimizer_states[t]
                weight_norm = torch.linalg.norm(bs[t].weight, dim=1, ord=2).view(
                    m1.shape[0], 1
                )
                grad_norm = torch.linalg.norm(
                    bs[t].weight.grad.to_dense(), dim=1, ord=2
                ).view(m1.shape[0], 1)
                adjusted_lr = (
                    lr * eta * weight_norm / (grad_norm + weight_decay * weight_norm)
                )
                m1_ref = adjusted_lr * (
                    bs[t].weight.grad.to_dense() + weight_decay * bs[t].weight
                )

                torch.testing.assert_allclose(
                    m1.index_select(dim=0, index=x[t].view(-1)),
                    m1_ref.index_select(dim=0, index=x[t].view(-1)),
                    atol=1.0e-4,
                    rtol=1.0e-4,
                )
                weights_new = split_weights[t]
                weights_ref = bs[t].weight - m1_ref
                torch.testing.assert_allclose(
                    weights_new.index_select(dim=0, index=x[t].view(-1)),
                    weights_ref.index_select(dim=0, index=x[t].view(-1)),
                    atol=1.0e-4,
                    rtol=1.0e-4,
                )


if __name__ == "__main__":
    unittest.main()
