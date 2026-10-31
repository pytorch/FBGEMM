# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for FBGEMM_TBE_ROCM_VBE_HALFWAVE_INT32 (ON in this process): the ROCm VBE forward's half-wave branch for
int32 indices.

The gate is read once per process, so this module sets it before the first TBE call; the gate-OFF case is
vbe_halfwave_int32_off_test.py (its own process). With the gate on, an int32-indexed VBE forward with FP16 weights,
FP32 output, no cache and MAX_D <= 128 must take the half-wave kernel (kSubwarpDivisor 2) and stay bit-identical:
each lane still sums its columns over the bag's ids in order. With MAX_D > 128 the branch does not apply.
"""

import os
import unittest

os.environ.setdefault("FBGEMM_NO_JK", "1")
os.environ["FBGEMM_TBE_ROCM_VBE_HALFWAVE_INT32"] = "1"

import torch  # noqa: E402
from fbgemm_gpu.split_embedding_configs import (  # noqa: E402
    EmbOptimType as OptimType,
    SparseType,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (  # noqa: E402
    BoundsCheckMode,
    EmbeddingLocation,
    PoolingMode,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (  # noqa: E402
    ComputeDevice,
    SplitTableBatchedEmbeddingBagsCodegen,
)

DIMS = [64, 64, 128]  # MAX_D 128: the half-wave branch's limit
WIDE_DIMS = [64, 64, 192]  # MAX_D 192 > 128: never half-wave
ROWS = [5000, 7000, 3000]
B = [[37, 5, 120], [64, 64, 1], [9, 300, 17]]  # feature x rank, deliberately uneven


def _rocm() -> bool:
    return torch.cuda.is_available() and torch.version.hip is not None


def _build(
    index_dtype: torch.dtype, dims: list[int]
) -> SplitTableBatchedEmbeddingBagsCodegen:
    tbe = SplitTableBatchedEmbeddingBagsCodegen(
        embedding_specs=[
            (r, d, EmbeddingLocation.DEVICE, ComputeDevice.CUDA)
            for r, d in zip(ROWS, dims)
        ],
        optimizer=OptimType.EXACT_ROWWISE_ADAGRAD,
        weights_precision=SparseType.FP16,
        output_dtype=SparseType.FP32,
        pooling_mode=PoolingMode.SUM,
        bounds_check_mode=BoundsCheckMode.FATAL,
        embedding_table_index_type=index_dtype,
        embedding_table_offset_type=index_dtype,
    ).cuda()
    with torch.no_grad():
        for t, w in enumerate(tbe.split_embedding_weights()):
            g = torch.Generator(device="cuda").manual_seed(1234 + t)
            w.copy_((torch.rand(w.shape, generator=g, device="cuda") - 0.5).to(w.dtype))
    return tbe


def _inputs(
    batch: list[list[int]],
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    """Per (feature, rank): bag lengths 0-5 with some 40-long bags, and uniform ids; concatenated feature-major."""
    g = torch.Generator().manual_seed(7)
    lens, ids = [], []
    for f in range(len(batch)):
        for r in range(len(batch[f])):
            L = torch.randint(0, 6, (batch[f][r],), generator=g)
            L[::11] = 40  # some long bags
            lens.append(L)
            ids.append(torch.randint(0, ROWS[f], (int(L.sum()),), generator=g))
    lengths = torch.cat(lens)
    offsets = torch.zeros(lengths.numel() + 1, dtype=torch.int64)
    offsets[1:] = torch.cumsum(lengths, 0)
    return torch.cat(ids), offsets, lens, ids


def _reference(
    tbe: SplitTableBatchedEmbeddingBagsCodegen,
    dims: list[int],
    lens: list[torch.Tensor],
    ids: list[torch.Tensor],
) -> torch.Tensor:
    """FP32 sum of FP16 rows in id order, laid out rank-major then feature then bag (FBGEMM's VBE output order)."""
    weights = [w.float().cpu() for w in tbe.split_embedding_weights()]
    T, R = len(dims), len(B[0])
    blocks = {}
    k = 0
    for f in range(T):
        for r in range(R):
            L, idx = lens[k], ids[k]
            out = torch.zeros(B[f][r], dims[f], dtype=torch.float32)
            start = 0
            for b in range(B[f][r]):
                acc = torch.zeros(dims[f], dtype=torch.float32)
                for j in range(int(L[b])):
                    acc = acc + weights[f][int(idx[start + j])]
                out[b] = acc
                start += int(L[b])
            blocks[(f, r)] = out.flatten()
            k += 1
    return torch.cat([blocks[(f, r)] for r in range(R) for f in range(T)])


def vbe_forward(
    index_dtype: torch.dtype, dims: list[int]
) -> tuple[torch.Tensor, list[str], torch.Tensor]:
    """One VBE forward; returns the output, the VBE forward kernel names the profiler saw, and the id-order reference."""
    tbe = _build(index_dtype, dims)
    idx, off, lens, ids = _inputs(B)
    with torch.no_grad():
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]
        ) as p:
            out = tbe(
                idx.to(index_dtype).cuda(),
                off.to(index_dtype).cuda(),
                batch_size_per_feature_per_rank=B,
            )
            torch.cuda.synchronize()
    names = sorted(
        {e.name for e in p.events() if "vbe" in e.name and "forward" in e.name}
    )
    return out.cpu(), names, _reference(tbe, dims, lens, ids)


def divisor(name: str) -> str:
    """The VBE forward kernel's last template argument (kSubwarpDivisor), e.g. "2ul"."""
    return name.split("(")[0].rstrip(">").split(",")[-1].strip()


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max()) if a.shape == b.shape else float("inf")


@unittest.skipUnless(_rocm(), "ROCm only")
class VbeHalfwaveInt32Test(unittest.TestCase):
    """Gate ON: route (kernel instantiation) and exact values of the int32 VBE forward."""

    def test_int32_equals_int64(self) -> None:
        """int32 and int64 indices (both half-wave now) give the same output bit for bit."""
        out32, _, _ = vbe_forward(torch.int32, DIMS)
        out64, _, _ = vbe_forward(torch.int64, DIMS)
        self.assertTrue(torch.equal(out32, out64), f"max abs {max_abs(out32, out64)}")

    def test_int32_equals_id_order_reference(self) -> None:
        """The int32 output equals an FP32 id-order sum of the FP16 rows, bit for bit."""
        out32, _, ref = vbe_forward(torch.int32, DIMS)
        self.assertTrue(torch.equal(out32, ref), f"max abs {max_abs(out32, ref)}")

    def test_int32_takes_half_wave_kernel(self) -> None:
        """MAX_D 128: the VBE forward kernel runs with kSubwarpDivisor 2 (two 32-lane groups per wave)."""
        _, names, _ = vbe_forward(torch.int32, DIMS)
        self.assertTrue(names, "no VBE forward kernel ran")
        self.assertTrue(all(divisor(n) == "2ul" for n in names), names)

    def test_max_d_above_128_keeps_full_wave(self) -> None:
        """MAX_D 192: the half-wave branch does not apply; kSubwarpDivisor 1, and the output is still exact."""
        out32, names, ref = vbe_forward(torch.int32, WIDE_DIMS)
        self.assertTrue(names, "no VBE forward kernel ran")
        self.assertTrue(all(divisor(n) == "1ul" for n in names), names)
        self.assertTrue(torch.equal(out32, ref), f"max abs {max_abs(out32, ref)}")


if __name__ == "__main__":
    unittest.main()
