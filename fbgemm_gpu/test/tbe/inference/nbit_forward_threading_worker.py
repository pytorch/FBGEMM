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
import os
import sys
from typing import Callable

import torch
from fbgemm_gpu.config import FeatureGate, FeatureGateName
from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_inference import (
    IntNBitTableBatchedEmbeddingBagsCodegen,
)
from fbgemm_gpu.tbe.config.embedding_config import EmbeddingLocation, PoolingMode
from torch.autograd.profiler_util import FunctionEvent


def _fill_random_weights(cc: IntNBitTableBatchedEmbeddingBagsCodegen) -> None:
    """fill_random_weights() only fills the quantized row bytes; the per-row
    fp16 scale/bias stay zero, which makes every forward output zero and every
    bitwise comparison vacuous. Give them small deterministic non-zero values."""
    cc.fill_random_weights()
    for _, scale_bias in cc.split_embedding_weights():
        if scale_bias is not None:
            scale_bias.view(torch.float16).copy_(
                (torch.rand(scale_bias.shape[0], 2) * 0.01 + 0.001).half()
            )


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
    _fill_random_weights(cc)

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
    _fill_random_weights(cc)

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


def _pooled_bags(
    pooling_mode: PoolingMode = PoolingMode.SUM,
    weighted: bool = False,
    T: int = 4,
    B: int = 1023,
    pad_total_D: int = 0,
    output_dtype: SparseType = SparseType.FP16,
    index_dtype: torch.dtype = torch.int32,
) -> tuple[torch.Tensor, bool]:
    # Few tables, many bags: table-level parallelism alone tops out at T=4
    # threads here, so anything beyond that comes from the bag-chunked
    # scheduler. _pooled() cannot cover this -- at B=8 the chunk grain exceeds
    # the batch, every table stays one work item and bag splitting never runs.
    #
    # B=1023: every table's last chunk is ragged (1/3/7 bags at 2/4/8 threads).
    #
    # Bag lengths are deliberately NON-UNIFORM and include empty bags. With a
    # constant L a wrong per-chunk index_size can still land on the right
    # answer, because offsets[n_bags] - offsets[0] and n_bags * L agree; varying
    # them makes the offset rebasing load bearing. Empty bags must still come
    # out as zeros on the path that zeroes the output per chunk.
    #
    # `weighted` passes per-sample weights, which each chunk reads from its own
    # rebased offset. B=1 with many tables is the read-only shape that must stay
    # serial when bag parallelism is on.
    #
    # `pad_total_D` widens total_D past D_offsets[T]; only the up-front fill
    # zeroes those columns. Also returns whether that fill ran.
    E, D = 1000, 16

    torch.manual_seed(0)
    cc = IntNBitTableBatchedEmbeddingBagsCodegen(
        embedding_specs=[("", E, D, SparseType.INT8, EmbeddingLocation.HOST)] * T,
        pooling_mode=pooling_mode,
        device="cpu",
        output_dtype=output_dtype,
    )
    _fill_random_weights(cc)
    cc.total_D += pad_total_D

    lengths = (torch.arange(T * B) % 9).to(index_dtype)
    offsets = torch.cat(
        [torch.zeros(1, dtype=index_dtype), lengths.cumsum(0).to(index_dtype)]
    )
    indices = (torch.arange(int(offsets[-1].item())) * 7 % E).to(index_dtype)
    per_sample_weights = (
        torch.rand(indices.numel(), dtype=torch.float32) if weighted else None
    )
    return _runs_up_front_fill(lambda: cc(indices, offsets, per_sample_weights))


def _runs_up_front_fill(
    forward: Callable[[], torch.Tensor],
) -> tuple[torch.Tensor, bool]:
    """Run `forward` under the profiler; report whether the lookup op ran
    aten::fill_. Float-output pooled lookups skip it only on the flattened
    path, so this shows which path ran."""
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as prof:
        out = forward()

    def inside_lookup(e: FunctionEvent) -> bool:
        p = e.cpu_parent
        while p is not None:
            if "int_nbit_split_embedding" in p.name:
                return True
            p = p.cpu_parent
        return False

    filled = any(e.name == "aten::fill_" and inside_lookup(e) for e in prof.events())
    return out, filled


_MODES = {
    "pooled": _pooled,
    "pooled_bags": _pooled_bags,
    # MEAN divides each bag by its own length, so this is the arm that catches a
    # chunk being handed the wrong bag count rather than the wrong bag offset.
    "pooled_bags_mean": lambda: _pooled_bags(PoolingMode.MEAN),
    "pooled_bags_weighted": lambda: _pooled_bags(weighted=True),
    "pooled_b1": lambda: _pooled_bags(T=64, B=1),
    "pooled_bags_padded_total_d": lambda: _pooled_bags(pad_total_D=16),
    # int64 indices/offsets hit the other generated kernel instantiation.
    "pooled_bags_int64": lambda: _pooled_bags(index_dtype=torch.int64),
    # 4-byte output_t for the per-chunk memset.
    "pooled_bags_fp32": lambda: _pooled_bags(output_dtype=SparseType.FP32),
    "nobag_skewed": lambda: _nobag_skewed(SparseType.INT4, torch.int32),
    # Floating output consumes the synthesised unit-stride chunk offsets, so it
    # exercises the offset/index/output rebasing that the INT4 fast path skips.
    "nobag_fp16": lambda: _nobag_skewed(SparseType.FP16, torch.int32),
    # int64 indices/offsets hit the other generated kernel instantiation.
    "nobag_fp16_int64": lambda: _nobag_skewed(SparseType.FP16, torch.int64),
}


def _check_row_parallelism_gate() -> None:
    """Fail loudly if the row-chunk gate did not resolve to what the driver asked
    for. Without this the NOBAG arms can silently fall through to the old
    whole-table scheduler and still pass, since the gate defaults to a JK lookup
    in fbcode and to an unset (false) env var in OSS. The driver pins it with
    FBGEMM_NO_JK=1, which is what makes the env var authoritative here."""
    if os.environ.get("FBGEMM_NO_JK") != "1":
        return
    expected = os.environ.get("FBGEMM_TBE_NOBAG_ROW_PARALLELISM") == "1"
    actual = FeatureGate.is_enabled(FeatureGateName.TBE_NOBAG_ROW_PARALLELISM)
    if actual != expected:
        raise RuntimeError(
            f"TBE_NOBAG_ROW_PARALLELISM resolved to {actual}, expected {expected}"
        )


def _compare(ref_path: str, other_paths: list[str]) -> None:
    """Compare saved outputs by VALUE. Needed because torch.save is not
    byte-reproducible -- two identical serial runs of this worker produce
    different file bytes and even different file sizes -- so diffing or hashing
    the .pt files says nothing about whether the tensors agree."""
    ref = torch.load(ref_path)
    f = ref.to(torch.float32)
    print(
        f"REF {ref_path}: shape={tuple(ref.shape)} dtype={ref.dtype} "
        f"nonzero={int((ref != 0).sum())}/{ref.numel()} "
        f"min={float(f.min()):.6g} max={float(f.max()):.6g} sum={float(f.sum()):.6g}"
    )
    mismatches = 0
    for path in other_paths:
        out = torch.load(path)
        if out.shape != ref.shape:
            print(f"MISMATCH {path}: shape {tuple(out.shape)} vs {tuple(ref.shape)}")
            mismatches += 1
        elif not torch.equal(out, ref):
            delta = (out.to(torch.float32) - ref.to(torch.float32)).abs()
            print(
                f"MISMATCH {path}: ndiff={int((out != ref).sum())} "
                f"of {out.numel()} maxabs={float(delta.max()):.6g}"
            )
            mismatches += 1
        else:
            print(f"OK {path}: bitwise identical")
    if mismatches:
        raise SystemExit(f"{mismatches} mismatch(es) against {ref_path}")


def main() -> None:
    out_path = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "pooled"
    if mode == "compare":
        _compare(out_path, sys.argv[3:])
        return
    _check_row_parallelism_gate()
    result = _MODES[mode]()
    out, filled = result if isinstance(result, tuple) else (result, None)
    torch.save(_serializable(out), out_path)
    if filled is not None:
        # Read by the driver's path assertions.
        with open(out_path + ".fill", "w") as f:
            f.write("1" if filled else "0")


if __name__ == "__main__":
    main()
