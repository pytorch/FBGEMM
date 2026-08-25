#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[56]

import random
import unittest

import hypothesis.strategies as st
import numpy as np
import torch
from fbgemm_gpu.split_embedding_configs import FP8QuantizationConfig, SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_inference import (
    IntNBitTableBatchedEmbeddingBagsCodegen,
)
from fbgemm_gpu.tbe.cache.cache_config import CacheAlgorithm
from fbgemm_gpu.tbe.config.embedding_config import EmbeddingLocation, PoolingMode
from fbgemm_gpu.tbe.utils import (
    b_indices,
    fake_quantize_embs,
    get_table_batched_offsets_from_dense,
    quantize_embs,
    round_up,
    to_device,
)
from hypothesis import assume
from hypothesis.strategies import composite


@composite
# pyre-ignore
def get_nbit_weights_ty(draw) -> SparseType | None:
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


FP8_EXPONENT_BITS: int = 4
FP8_EXPONENT_BIAS: int = 7


def nbit_output_to_float(output: torch.Tensor) -> torch.Tensor:
    """Bring a TBE output tensor into a dtype that compares elementwise."""
    if output.dtype == torch.quint4x2:
        return torch.ops.fbgemm.FusedNBitRowwiseQuantizedSBHalfFrontToFloatOrHalf(
            output.cpu(), bit_rate=4, output_dtype=0
        )
    return output.cpu().float()


class NBitFowardTestCommon(unittest.TestCase):
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
        indices_dtype: torch.dtype,
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
                    ]
                    + (
                        [
                            SparseType.INT2,
                        ]
                        if output_dtype != SparseType.FP32
                        else []
                    )
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

        if do_pruning:
            x = torch.cat([x.view(1, B, L) for x in xs], dim=0)
            xw = torch.cat([xw.view(1, B, L) for xw in xws], dim=0)

            indices, offsets = get_table_batched_offsets_from_dense(x, use_cpu=use_cpu)

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
                indice_t = (
                    (indices.view(T, B, L))[t]
                    .view(-1)
                    .to(dtype=indices_dtype, device=current_device)
                )
                dense_indice_t = (
                    (dense_indices.view(T, B, L))[t]
                    .view(-1)
                    .to(dtype=indices_dtype, device=current_device)
                )
                index_remappings_array_t = torch.tensor(
                    [-1] * original_E,
                    dtype=indices_dtype,
                    device=current_device,
                )
                index_remappings_array_t[indice_t] = dense_indice_t
                index_remappings_array.append(index_remappings_array_t.cpu())
        else:
            index_remappings_array = [torch.arange(E, dtype=indices_dtype) for E in Es]
            x = torch.cat([x.view(1, B, L) for x in xs], dim=0)
            xw = torch.cat([xw.view(1, B, L) for xw in xws], dim=0)
            indices, offsets = get_table_batched_offsets_from_dense(x, use_cpu=use_cpu)

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
            fp8_exponent_bits=(
                fp8_config.get("exponent_bits") if has_fp8_weight else None
            ),
            fp8_exponent_bias=(
                fp8_config.get("exponent_bias") if has_fp8_weight else None
            ),
            indices_dtype=indices_dtype,
        )
        # Initialize the random weights for int nbit table split embedding bag
        cc.fill_random_weights()

        if not use_cpu:
            # NOTE: test TorchScript-compatible!
            cc = torch.jit.script(cc)

        for t in range(T):
            weights, scale_shift = cc.split_embedding_weights()[t]
            if scale_shift is not None:
                E, R = scale_shift.shape
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

        indices = indices.to(dtype=indices_dtype)
        offsets = offsets.to(dtype=indices_dtype)

        if not use_cpu:
            fc2 = (
                cc(indices, offsets)
                if not weighted
                else cc(indices, offsets, xw.contiguous().view(-1))
            )
        else:
            cc = cc.cpu()
            indices, offsets = indices.cpu(), offsets.cpu()
            fc2 = (
                cc(indices, offsets)
                if not weighted
                else cc(indices, offsets, xw.contiguous().view(-1).cpu())
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

        if fc2.dtype == torch.quint4x2:
            fc2_float = (
                torch.ops.fbgemm.FusedNBitRowwiseQuantizedSBHalfFrontToFloatOrHalf(
                    fc2.cpu(), bit_rate=4, output_dtype=0
                )
            )
        else:
            fc2_float = fc2.float()

        torch.testing.assert_close(
            fc2_float.cpu(),
            f.float().cpu(),
            atol=1.0e-2,
            rtol=1.0e-2,
        )

    def _make_cpu_seq_op(
        self,
        weights_ty: SparseType,
        output_dtype: SparseType,
        T: int,
        E: int,
        D: int,
    ) -> IntNBitTableBatchedEmbeddingBagsCodegen:
        """A CPU sequence-mode TBE whose rows hold finite, reproducible values.

        ``fill_random_weights`` writes random *bytes*, and for the float weight
        types those decode to NaN often enough that an exact output comparison
        is meaningless. Quantizing a fixed pseudo-random float matrix instead
        keeps every row finite.
        """
        fp8_config = (
            FP8QuantizationConfig(FP8_EXPONENT_BITS, FP8_EXPONENT_BIAS)
            if weights_ty == SparseType.FP8
            else None
        )
        op = IntNBitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                ("", E, D, weights_ty, EmbeddingLocation.HOST) for _ in range(T)
            ],
            pooling_mode=PoolingMode.NONE,
            output_dtype=output_dtype,
            device="cpu",
            fp8_exponent_bits=FP8_EXPONENT_BITS if fp8_config is not None else None,
            fp8_exponent_bias=FP8_EXPONENT_BIAS if fp8_config is not None else None,
        )
        op.fill_random_weights()

        generator = torch.Generator().manual_seed(0)
        for t in range(T):
            quant_weights, quant_scale_shift = quantize_embs(
                torch.rand((E, D), generator=generator) * 2 - 1,
                weights_ty,
                fp8_config,
            )
            weights, scale_shift = op.split_embedding_weights()[t]
            weights.copy_(quant_weights)
            if quant_scale_shift is not None:
                self.assertIsNotNone(scale_shift)
                scale_shift.copy_(quant_scale_shift)
        return op

    def _check_nbit_forward_cpu_seq_ragged_matches_flat(
        self,
        weights_ty: SparseType,
        output_dtype: SparseType,
    ) -> None:
        """A sequence forward depends only on the flat index sequence.

        How those indices are grouped into bags must not change the result. The
        CPU nobag path hands some kernels the real per-bag offsets (their no_bag
        fast path never dereferences offsets) and others a virtual offsets
        tensor whose lengths are all ones. Feeding the same indices first as
        ragged bags and then one index per bag makes those two tensors differ in
        length and in content, so a kernel handed the wrong one cannot agree
        with itself across the two groupings.
        """
        T, E, D = 3, 64, 32
        # Ragged, including empty bags. Every table must see the same number of
        # indices, otherwise the two groupings would not slice the flat index
        # sequence into the same per-table ranges.
        lengths = [[0, 3, 1, 4], [2, 0, 1, 5], [4, 4, 0, 0]]
        indices_per_table = sum(lengths[0])
        self.assertTrue(all(sum(row) == indices_per_table for row in lengths))

        op = self._make_cpu_seq_op(weights_ty, output_dtype, T, E, D)

        indices = torch.tensor(
            [(i * 7 + 3) % E for i in range(T * indices_per_table)], dtype=torch.int
        )
        ragged_offsets: list[int] = [0]
        for table_lengths in lengths:
            for length in table_lengths:
                ragged_offsets.append(ragged_offsets[-1] + length)
        self.assertEqual(ragged_offsets[-1], indices.numel())

        ragged_output = op(indices, torch.tensor(ragged_offsets, dtype=torch.int))
        flat_output = op(indices, torch.arange(indices.numel() + 1, dtype=torch.int))

        self.assertEqual(ragged_output.shape, flat_output.shape)
        torch.testing.assert_close(
            nbit_output_to_float(ragged_output),
            nbit_output_to_float(flat_output),
            rtol=0,
            atol=0,
            equal_nan=False,
        )

    def _check_nbit_forward_cpu_seq_no_indices(
        self,
        weights_ty: SparseType,
        output_dtype: SparseType,
    ) -> None:
        """A sequence forward whose bags are all empty returns an empty output.

        Every table has an index range of length zero here, which is the
        boundary at which the nobag path decides whether to build virtual
        offsets at all.
        """
        T, E, D, B = 2, 64, 32, 4

        op = self._make_cpu_seq_op(weights_ty, output_dtype, T, E, D)

        output = op(
            torch.empty(0, dtype=torch.int), torch.zeros(T * B + 1, dtype=torch.int)
        )

        self.assertEqual(output.shape[0], 0)
