# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# Worker for nbit_forward_threading_test: builds a deterministic CPU int-nbit TBE op
# and writes its forward output to the path given as argv[1]. The driver runs this
# under different FBGEMM_TBE_MAX_NUM_THREADS / FBGEMM_TBE_MIN_TABLES_PER_THREAD env values (read once,
# at the first kernel call, hence a separate process per setting) and checks the
# outputs are bitwise identical -- i.e. table-threading does not change the result.
import sys

import torch
from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_inference import (
    IntNBitTableBatchedEmbeddingBagsCodegen,
)
from fbgemm_gpu.tbe.config.embedding_config import EmbeddingLocation, PoolingMode


def _pooled() -> torch.Tensor:
    # T=40 > the default threading onset (2*G = 32 at G=16), so even the
    # default-granularity arm (FBGEMM_TBE_MAX_NUM_THREADS=2, no FBGEMM_TBE_MIN_TABLES_PER_THREAD)
    # genuinely spawns threads rather than falling back to the serial path.
    T, E, D, B, L = 40, 1000, 16, 8, 6

    # Deterministic weights: same seed + same torch build => identical across the
    # worker processes the driver spawns, so the only variable is the thread count.
    torch.manual_seed(0)
    cc = IntNBitTableBatchedEmbeddingBagsCodegen(
        embedding_specs=[("", E, D, SparseType.INT8, EmbeddingLocation.HOST)] * T,
        pooling_mode=PoolingMode.SUM,
        device="cpu",
        output_dtype=SparseType.FP16,
    )
    cc.fill_random_weights()

    # Deterministic indices/offsets (no RNG): T*B bags, each pooling L indices.
    indices = (torch.arange(T * B * L) % E).to(torch.int32)
    offsets = (torch.arange(T * B + 1) * L).to(torch.int32)
    return cc(indices, offsets)


def _nobag_skewed(
    output_dtype: SparseType = SparseType.INT4,
    index_dtype: torch.dtype = torch.int32,
) -> torch.Tensor:
    """INT4-weight sequence (NOBAG) TBE where one table carries ~85% of the
    lookups.

    This is the shape row chunking exists for: parallelising over tables caps
    the speedup at sum(rows)/max(rows) ~= 1.17 here, so the scheduler must split
    table 0's row range. Counts are well above MIN_CHUNK_ROWS (1024) and
    MIN_ROWS_PER_THREAD (4096) so chunking actually engages rather than falling
    back to the serial path.

    `output_dtype` and `index_dtype` are parametrised because the row-chunk
    scheduler runs for every NOBAG output except INT8: the INT4 fast path
    ignores offsets, but a floating output (FP16) consumes the synthesised
    unit-stride chunk offsets, and int64 indices/offsets hit the other
    generated kernel instantiation. All must stay bitwise identical to serial.
    """
    T, E, D, B = 8, 4096, 64, 4
    counts = [20000] + [500] * (T - 1)

    torch.manual_seed(0)
    cc = IntNBitTableBatchedEmbeddingBagsCodegen(
        embedding_specs=[("", E, D, SparseType.INT4, EmbeddingLocation.HOST)] * T,
        pooling_mode=PoolingMode.NONE,
        output_dtype=output_dtype,
        device="cpu",
        # int4 NOBAG output is only well-defined at row_alignment=1
        row_alignment=1,
    )
    cc.fill_random_weights()

    # Deterministic indices (no RNG); a stride coprime with E scatters them.
    idx_parts, all_lengths = [], []
    for c in counts:
        idx_parts.append(((torch.arange(c) * 7919) % E).to(index_dtype))
        base, rem = divmod(c, B)
        lengths = torch.full((B,), base, dtype=index_dtype)
        lengths[:rem] += 1
        all_lengths.append(lengths)
    indices = torch.cat(idx_parts)
    offsets = torch.cat(
        [
            torch.zeros(1, dtype=index_dtype),
            torch.cumsum(torch.cat(all_lengths), 0).to(index_dtype),
        ]
    )
    return cc(indices, offsets)


def _serializable(t: torch.Tensor) -> torch.Tensor:
    """INT4 output comes back as quint4x2, which cannot be pickled (its
    quantizer is UnknownQuantizer). Hand back a flat uint8 view of the same
    bytes -- which is exactly what a bitwise comparison wants anyway."""
    t = t.cpu()
    if t.dtype in (torch.quint4x2, torch.quint2x4):
        return torch.empty(0, dtype=torch.uint8).set_(t.untyped_storage()).clone()
    return t


_MODES = {
    "pooled": _pooled,
    "nobag_skewed": lambda: _nobag_skewed(SparseType.INT4, torch.int32),
    # Floating output consumes the synthesised unit-stride chunk offsets, so it
    # exercises the offset/index/output rebasing that the INT4 fast path skips.
    "nobag_fp16": lambda: _nobag_skewed(SparseType.FP16, torch.int32),
    # int64 indices/offsets hit the other generated kernel instantiation.
    "nobag_fp16_int64": lambda: _nobag_skewed(SparseType.FP16, torch.int64),
}


def main() -> None:
    out_path = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "pooled"
    out = _MODES[mode]()
    torch.save(_serializable(out), out_path)


if __name__ == "__main__":
    main()
