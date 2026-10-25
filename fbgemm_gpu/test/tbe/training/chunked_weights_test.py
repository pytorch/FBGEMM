#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
import unittest

import torch
from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType, SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    chunk_div_magic,
    SplitTableBatchedEmbeddingBagsCodegen,
)
from fbgemm_gpu.tbe.config.embedding_config import (
    ComputeDevice,
    EmbeddingLocation,
    PoolingMode,
)

from ..common import open_source

if open_source:
    # pyre-ignore[21]
    from test_utils import gpu_unavailable, running_on_rocm
else:
    from fbgemm_gpu.test.test_utils import gpu_unavailable, running_on_rocm


B = 16
MAX_L = 10
# Small enough that every table spans several chunks with an odd row count
ROWS_PER_CHUNK = 7

DEVICE = EmbeddingLocation.DEVICE
MANAGED = EmbeddingLocation.MANAGED


def _specs(
    tables: list[tuple[int, int, EmbeddingLocation]],
) -> list[tuple[int, int, EmbeddingLocation, ComputeDevice]]:
    return [(rows, dim, loc, ComputeDevice.CUDA) for rows, dim, loc in tables]


def _make_tbe(
    tables: list[tuple[int, int, EmbeddingLocation]],
    chunk_bytes: int | None,
    optimizer: OptimType = OptimType.EXACT_ROWWISE_ADAGRAD,
    pooling_mode: PoolingMode = PoolingMode.SUM,
    precision: SparseType = SparseType.FP32,
    use_experimental_tbe: bool = False,
    chunk_optimizer_state: bool = False,
) -> SplitTableBatchedEmbeddingBagsCodegen:
    return SplitTableBatchedEmbeddingBagsCodegen(
        embedding_specs=_specs(tables),
        optimizer=optimizer,
        learning_rate=0.1,
        weights_precision=precision,
        output_dtype=SparseType.FP32,
        pooling_mode=pooling_mode,
        stochastic_rounding=False,
        weights_chunk_size_bytes=chunk_bytes,
        use_experimental_tbe=use_experimental_tbe,
        chunk_optimizer_state=chunk_optimizer_state,
    )


def _make_pair(
    tables: list[tuple[int, int, EmbeddingLocation]],
    optimizer: OptimType = OptimType.EXACT_ROWWISE_ADAGRAD,
    pooling_mode: PoolingMode = PoolingMode.SUM,
    precision: SparseType = SparseType.FP32,
    use_experimental_tbe: bool = False,
    chunk_optimizer_state: bool = False,
) -> tuple[
    SplitTableBatchedEmbeddingBagsCodegen, SplitTableBatchedEmbeddingBagsCodegen
]:
    """
    Builds a flat TBE and a chunked TBE with identical weights. The chunk size
    is picked so the widest table gets ROWS_PER_CHUNK rows per chunk.
    """
    elem_size = 2 if precision == SparseType.FP16 else 4
    max_dim = max(dim for _, dim, _ in tables)
    chunk_bytes = ROWS_PER_CHUNK * max_dim * elem_size

    flat = _make_tbe(
        tables, None, optimizer, pooling_mode, precision, use_experimental_tbe
    )
    chunked = _make_tbe(
        tables,
        chunk_bytes,
        optimizer,
        pooling_mode,
        precision,
        use_experimental_tbe,
        chunk_optimizer_state,
    )
    flat.init_embedding_weights_uniform(-1.0, 1.0)
    for flat_w, chunks in zip(
        flat.split_embedding_weights(), chunked.split_embedding_weights_chunked()
    ):
        start = 0
        for chunk in chunks:
            chunk.copy_(flat_w[start : start + chunk.shape[0]])
            start += chunk.shape[0]
    return flat, chunked


def _gathered_weights(tbe: SplitTableBatchedEmbeddingBagsCodegen) -> list[torch.Tensor]:
    return [torch.cat(chunks) for chunks in tbe.split_embedding_weights_chunked()]


def _make_inputs(
    tables: list[tuple[int, int, EmbeddingLocation]],
    weighted: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Random bags per (table, sample). Every table also looks up the rows on
    either side of each chunk boundary plus its last row.
    """
    indices, lengths = [], []
    for rows, _, _ in tables:
        boundary_rows = [
            r
            for start in range(ROWS_PER_CHUNK, rows, ROWS_PER_CHUNK)
            for r in (start - 1, start)
        ] + [rows - 1]
        for b in range(B):
            L = int(torch.randint(0, MAX_L + 1, (1,)))
            bag = torch.randint(0, rows, (L,)).tolist()
            if b == 0:
                bag += boundary_rows
            indices += bag
            lengths.append(len(bag))
    offsets = torch.tensor(
        [0] + lengths, device=torch.accelerator.current_accelerator()
    ).cumsum(0)
    per_sample_weights = (
        torch.rand(len(indices), device=torch.accelerator.current_accelerator())
        if weighted
        else None
    )
    return (
        torch.tensor(indices, device=torch.accelerator.current_accelerator()),
        offsets,
        per_sample_weights,
    )


# Chunked weights are CUDA only
# pyrefly: ignore [bad-argument-type]
@unittest.skipIf(*running_on_rocm)
class ChunkedWeightsTest(unittest.TestCase):
    # pyrefly: ignore [bad-argument-type]
    @unittest.skipIf(*gpu_unavailable)
    def test_chunk_layout(self) -> None:
        # 100 rows per chunk for the D=64 table, so 10 chunks; the D=32 table
        # fits in one chunk; the MANAGED table stays flat
        tables = [(1000, 64, DEVICE), (10, 32, DEVICE), (50, 16, MANAGED)]
        chunk_bytes = 100 * 64 * 4
        tbe = _make_tbe(tables, chunk_bytes)

        chunks = tbe.split_embedding_weights_chunked()
        self.assertEqual([len(c) for c in chunks], [10, 1, 1])
        self.assertEqual([c.shape for c in chunks[0]], [torch.Size([100, 64])] * 10)
        self.assertEqual(chunks[1][0].shape, torch.Size([10, 32]))
        for c in tbe.weights_dev_chunks[0] + tbe.weights_dev_chunks[1]:
            self.assertLessEqual(c.numel() * c.element_size(), chunk_bytes)
        self.assertEqual(tbe.weights_dev_chunks[2], [])

        # Chunked tables point at a chunk header, the flat table keeps its offset
        weights_offsets = tbe.weights_offsets
        assert isinstance(weights_offsets, torch.Tensor)
        offsets = weights_offsets.tolist()
        self.assertLess(offsets[0], 0)
        self.assertLess(offsets[1], 0)
        self.assertGreaterEqual(offsets[2], 0)

        # The single-chunk table still has a plain view; multi-chunk does not
        with self.assertRaises(NotImplementedError):
            tbe.split_embedding_weights()

    def test_chunk_div_magic(self) -> None:
        # The kernels' multiply-shift must match floor division for any row
        # index a chunked table can have
        for rows_per_chunk in (1, 3, 7, 100, 4095, 4096, 4097, 2**31 - 1):
            magic, shift = chunk_div_magic(rows_per_chunk)
            for idx in (0, rows_per_chunk - 1, rows_per_chunk, 12345, 2**31 - 1):
                self.assertEqual((idx * magic) >> shift, idx // rows_per_chunk)

    # pyrefly: ignore [bad-argument-type]
    @unittest.skipIf(*gpu_unavailable)
    def test_init_embedding_weights_uniform(self) -> None:
        tbe = _make_tbe([(100, 16, DEVICE)], chunk_bytes=10 * 16 * 4)
        tbe.init_embedding_weights_uniform(0.5, 1.0)
        weights = _gathered_weights(tbe)[0]
        self.assertTrue(bool(((weights >= 0.5) & (weights <= 1.0)).all()))

    # pyrefly: ignore [bad-argument-type]
    @unittest.skipIf(*gpu_unavailable)
    def test_forward_parity(self) -> None:
        pooled_tables = [(60, 16, DEVICE), (45, 32, DEVICE), (30, 8, MANAGED)]
        nobag_tables = [(60, 16, DEVICE), (45, 16, DEVICE), (30, 16, MANAGED)]
        cases = [
            (pooling, weighted, precision, experimental)
            for pooling in (PoolingMode.SUM, PoolingMode.MEAN, PoolingMode.NONE)
            for weighted in (False, True)
            for precision in (SparseType.FP32, SparseType.FP16)
            for experimental in (False, True)
            if not (pooling == PoolingMode.NONE and weighted)
        ]
        for pooling, weighted, precision, experimental in cases:
            with self.subTest(
                pooling=pooling,
                weighted=weighted,
                precision=precision,
                experimental=experimental,
            ):
                tables = nobag_tables if pooling == PoolingMode.NONE else pooled_tables
                flat, chunked = _make_pair(
                    tables,
                    pooling_mode=pooling,
                    precision=precision,
                    use_experimental_tbe=experimental,
                )
                indices, offsets, psw = _make_inputs(tables, weighted)
                torch.testing.assert_close(
                    chunked(indices, offsets, psw),
                    flat(indices, offsets, psw),
                    rtol=0,
                    atol=0,
                )

    # pyrefly: ignore [bad-argument-type]
    @unittest.skipIf(*gpu_unavailable)
    def test_backward_parity(self) -> None:
        tables = [(60, 16, DEVICE), (45, 32, DEVICE), (30, 8, MANAGED)]
        cases = [
            (optimizer, precision, weighted)
            for optimizer in (
                OptimType.EXACT_ROWWISE_ADAGRAD,
                OptimType.EXACT_SGD,
                OptimType.EXACT_ADAGRAD,
                OptimType.ADAM,
                OptimType.PARTIAL_ROWWISE_ADAM,
                OptimType.LAMB,
                OptimType.PARTIAL_ROWWISE_LAMB,
            )
            for precision in (SparseType.FP32, SparseType.FP16)
            for weighted in (False, True)
            # Flat TBE rejects ADAM with FP16 weights too (row_counter dtype)
            if not (optimizer == OptimType.ADAM and precision == SparseType.FP16)
        ]
        for optimizer, precision, weighted in cases:
            with self.subTest(
                optimizer=optimizer, precision=precision, weighted=weighted
            ):
                flat, chunked = _make_pair(
                    tables,
                    optimizer=optimizer,
                    precision=precision,
                    chunk_optimizer_state=True,
                )
                for _ in range(3):
                    indices, offsets, psw = _make_inputs(tables, weighted)
                    out_flat = flat(indices, offsets, psw)
                    out_chunked = chunked(indices, offsets, psw)
                    torch.testing.assert_close(out_chunked, out_flat, rtol=0, atol=0)
                    grad = torch.randn_like(out_flat)
                    out_flat.backward(grad)
                    out_chunked.backward(grad)

                for w_chunked, w_flat in zip(
                    _gathered_weights(chunked), flat.split_embedding_weights()
                ):
                    torch.testing.assert_close(w_chunked, w_flat, rtol=0, atol=0)
                if optimizer == OptimType.EXACT_SGD:
                    continue
                for states_chunked, states_flat in zip(
                    chunked.split_optimizer_states_chunked(),
                    flat.split_optimizer_states(),
                ):
                    for state_chunked, state_flat in zip(states_chunked, states_flat):
                        if isinstance(state_chunked, list):
                            state_chunked = torch.cat(state_chunked)
                        torch.testing.assert_close(
                            state_chunked, state_flat, rtol=0, atol=0
                        )

    # pyrefly: ignore [bad-argument-type]
    @unittest.skipIf(*gpu_unavailable)
    def test_optimizer_state_layout(self) -> None:
        # Adam's pointwise momentum is chunked like the weights; the 16-wide
        # table gets 4 rows of fp32 state per 256 byte chunk
        tables = [(10, 16, DEVICE), (20, 16, MANAGED)]
        # Optimizer state stays flat unless asked for
        weights_only = _make_tbe(tables, chunk_bytes=256, optimizer=OptimType.ADAM)
        self.assertEqual(weights_only.momentum1_dev_chunks, [])
        tbe = _make_tbe(
            tables,
            chunk_bytes=256,
            optimizer=OptimType.ADAM,
            chunk_optimizer_state=True,
        )
        self.assertEqual([c.numel() for c in tbe.momentum1_dev_chunks[0]], [64, 64, 32])
        self.assertEqual(tbe.momentum1_dev_chunks[1], [])
        momentum1_offsets = tbe.momentum1_offsets
        assert isinstance(momentum1_offsets, torch.Tensor)
        self.assertLess(momentum1_offsets.tolist()[0], 0)

        # Same state names as the flat accessor
        flat = _make_tbe(tables, None, optimizer=OptimType.ADAM)
        self.assertEqual(
            [d.keys() for d in tbe.get_optimizer_state_chunked()],
            [d.keys() for d in flat.get_optimizer_state()],
        )
        with self.assertRaises(NotImplementedError):
            tbe.get_optimizer_state()

    # pyrefly: ignore [bad-argument-type]
    @unittest.skipIf(*gpu_unavailable)
    def test_torchscript_parity(self) -> None:
        # Scripted modules only keep attributes TorchScript can type, so the
        # weight accessors must still work after scripting
        tables = [(60, 16, DEVICE), (30, 8, MANAGED)]
        flat, chunked = _make_pair(tables)
        scripted_flat = torch.jit.script(flat)
        scripted_chunked = torch.jit.script(chunked)
        indices, offsets, _ = _make_inputs(tables, weighted=False)

        torch.testing.assert_close(
            scripted_chunked(indices, offsets),
            scripted_flat(indices, offsets),
            rtol=0,
            atol=0,
        )
        for chunks, w_flat in zip(
            scripted_chunked.split_embedding_weights_chunked(),
            scripted_flat.split_embedding_weights(),
        ):
            torch.testing.assert_close(torch.cat(chunks), w_flat, rtol=0, atol=0)

    # pyrefly: ignore [bad-argument-type]
    @unittest.skipIf(*gpu_unavailable)
    def test_relocation(self) -> None:
        # Chunk headers hold raw addresses, so copies and device moves have to
        # rewrite them
        tables = [(60, 16, DEVICE), (30, 8, MANAGED)]
        flat, chunked = _make_pair(tables, chunk_optimizer_state=True)
        indices, offsets, _ = _make_inputs(tables, weighted=False)
        expected = flat(indices, offsets)

        copied = copy.deepcopy(chunked)
        copied_offsets, original_offsets = (
            copied.weights_offsets,
            chunked.weights_offsets,
        )
        assert isinstance(copied_offsets, torch.Tensor)
        assert isinstance(original_offsets, torch.Tensor)
        self.assertNotEqual(copied_offsets.tolist(), original_offsets.tolist())
        del chunked
        torch.testing.assert_close(copied(indices, offsets), expected, rtol=0, atol=0)

        # TBE can't run forward after a cross-device move (its cache buffers
        # stay put), but the chunks and headers should follow the module
        if torch.cuda.device_count() < 2:
            return
        device = torch.device("cuda:1")
        moved = copied.to(device)
        header_table = moved._chunk_tables["weights"]
        self.assertEqual(header_table.device, device)
        self.assertTrue(all(c.device == device for c in moved.weights_dev_chunks[0]))
        weights_offsets = moved.weights_offsets
        assert isinstance(weights_offsets, torch.Tensor)
        self.assertEqual(-int(weights_offsets[0]), header_table.data_ptr())

    # pyrefly: ignore [bad-argument-type]
    @unittest.skipIf(*gpu_unavailable)
    def test_unsupported_configs_raise(self) -> None:
        tables = [(10, 16, DEVICE)]
        with self.assertRaises(AssertionError):
            _make_tbe(tables, chunk_bytes=0)
        with self.assertRaises(AssertionError):
            _make_tbe(tables, chunk_bytes=1024, optimizer=OptimType.NONE)


if __name__ == "__main__":
    unittest.main()
