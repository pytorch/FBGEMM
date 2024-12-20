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
from typing import List, Optional, Tuple

import hypothesis.strategies as st
import numpy as np
import torch
from fbgemm_gpu.split_embedding_configs import FP8QuantizationConfig, SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    CacheAlgorithm,
    DEFAULT_SCALE_BIAS_SIZE_IN_BYTES,
    EmbeddingLocation,
    PoolingMode,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_inference import (
    IntNBitTableBatchedEmbeddingBagsCodegen,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_utils import (
    random_quant_scaled_tensor,
    rounded_row_size_in_bytes,
    unpadded_row_size_in_bytes,
)
from fbgemm_gpu.ssd_prefetcher import SSDPrefetcher
from fbgemm_gpu.tbe.utils import (
    b_indices,
    fake_quantize_embs,
    get_table_batched_offsets_from_dense,
    round_up,
    to_device,
)
from hypothesis import assume
from hypothesis.strategies import composite
from torch import Tensor


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
        test_ssd_prefetcher: bool = False,
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
            (indices, offsets) = get_table_batched_offsets_from_dense(
                x, use_cpu=use_cpu
            )

        embedding_specs = [
            (
                "",
                E,
                D,
                W_TY,
                EmbeddingLocation(M),
            )
            for (E, D, M, W_TY) in zip(Es, Ds, managed, weights_ty_list)
        ]
        if test_ssd_prefetcher:
            ssd_prefetcher = DummyPrefetcher(embedding_specs)
            ssd_table_placements = []
            for t in range(T):
                ssd_table_placements.extend([t] * B * L)
        else:
            ssd_prefetcher = None
            ssd_table_placements = None

        cc = IntNBitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=embedding_specs,
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
            ssd_prefetcher=ssd_prefetcher,
            ssd_placements=ssd_table_placements,
        )
        # Initialize the random weights for int nbit table split embedding bag
        if not test_ssd_prefetcher:
            cc.fill_random_weights()

        if not use_cpu:
            # NOTE: test TorchScript-compatible!
            cc = torch.jit.script(cc)

        for t in range(T):
            if ssd_prefetcher is None:
                (weights, scale_shift) = cc.split_embedding_weights()[t]
            else:
                (weights, scale_shift) = ssd_prefetcher.split_embedding_weights()[t]
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
            fc2_float = torch.ops.fbgemm.FusedNBitRowwiseQuantizedSBHalfFrontToFloat(
                fc2.cpu(), bit_rate=4
            )
        else:
            fc2_float = fc2.float()

        torch.testing.assert_close(
            fc2_float.cpu(),
            f.float().cpu(),
            atol=1.0e-2,
            rtol=1.0e-2,
        )


class DummyPrefetcher(SSDPrefetcher):
    """
    A dummy fake SSD prefetcher intended to test if the TBE module can work with
    the SSD prefetcher interface. It simply takes a range of embedding table shapes
    and generate random weights based on the shapes during initialization, and
    returns the picked rows when the prefetch() function is called.
    """

    def _generate_weights(
        self, embedding_specs: List[Tuple[str, int, int, SparseType, EmbeddingLocation]]
    ) -> None:
        for _, rows, dim, weight_ty, _ in embedding_specs:
            weights = random_quant_scaled_tensor(
                torch.Size(
                    (
                        rows,
                        rounded_row_size_in_bytes(
                            dim,
                            weight_ty,
                            self.row_alignment,
                            self.scale_bias_size_in_bytes,
                        ),
                    ),
                ),
                device=torch.device("cpu"),
            )
            self.weights.append(weights)
            self.nrows.append(self.nrows[-1] + rows)

    def split_embedding_weights(
        self, split_scale_shifts: bool = True
    ) -> List[Tuple[Tensor, Optional[Tensor]]]:
        splits: List[Tuple[Tensor, Optional[Tensor]]] = []
        for i, (_, _, dim, weight_ty, _) in enumerate(self.embedding_specs):
            weights_shifts = self.weights[i].detach()
            if split_scale_shifts:
                # remove the padding at the end of each row.
                weights_shifts = weights_shifts[
                    :,
                    : unpadded_row_size_in_bytes(
                        dim, weight_ty, self.scale_bias_size_in_bytes
                    ),
                ]
                if (
                    weight_ty == SparseType.INT8
                    or weight_ty == SparseType.INT4
                    or weight_ty == SparseType.INT2
                ):
                    splits.append(
                        (
                            weights_shifts[:, self.scale_bias_size_in_bytes :],
                            weights_shifts[:, : self.scale_bias_size_in_bytes],
                        )
                    )
                else:
                    assert (
                        weight_ty == SparseType.FP8
                        or weight_ty == SparseType.FP16
                        or weight_ty == SparseType.FP32
                    )
                    splits.append(
                        (
                            weights_shifts,
                            None,
                        )
                    )
            else:
                splits.append((weights_shifts, None))

        return splits

    def __init__(
        self,
        embedding_specs: List[Tuple[str, int, int, SparseType, EmbeddingLocation]],
        row_alignment: int = 1,
        scale_bias_size_in_bytes: int = DEFAULT_SCALE_BIAS_SIZE_IN_BYTES,
    ) -> None:
        self.embedding_specs = embedding_specs
        self.row_alignment = row_alignment
        self.scale_bias_size_in_bytes = scale_bias_size_in_bytes
        self.weights: List[torch.Tensor] = []
        self.nrows: List[int] = [0]
        self._generate_weights(embedding_specs)

    def prefetch(
        self,
        indices: torch.Tensor,
        offsets: torch.Tensor,
        table_placement: torch.Tensor,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if indices.numel() == 0:
            return (torch.tensor([], dtype=torch.uint8, device="cpu"), None, None)
        assert indices.numel() == table_placement.numel()
        fetched_rows = []
        new_indices = []
        weights_offsets = [0]
        prev_table_id = None
        idx = 0
        bytes_added = 0
        for i, tbl in enumerate(table_placement):
            if prev_table_id is None:
                prev_table_id = tbl
            if prev_table_id != tbl:
                idx = 0
                prev_table_id = tbl
                weights_offsets.append(bytes_added)
            fetched_rows.append(self.weights[tbl][indices[i]])
            new_indices.append(idx)
            idx += 1
            bytes_added += self.weights[tbl][indices[i]].numel()

        return (
            torch.cat(fetched_rows),
            torch.tensor(new_indices, dtype=indices.dtype, device="cpu"),
            torch.tensor(weights_offsets, dtype=torch.int64, device="cpu"),
        )
