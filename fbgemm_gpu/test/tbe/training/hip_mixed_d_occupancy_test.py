# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Tests for the ROCm hip_mixed_d warp backward occupancy instantiation (TBE_ROCM_HIP_MIXED_D_OCCUPANCY, default off).

With the HIP TBE backward and the eighth-wave route on (TBE_ROCM_HIP_MIXED_D_EIGHTH_WAVE=1), the gate makes tables
with max_D <= 96 take a <..., 3, 8, false> instantiation: three vectors per lane over 8-lane groups (8 * 3 * 4 = 96
columns), built with __launch_bounds__(512, 4) so that at least 4 waves fit per SIMD. Tables with max_D > 96 keep
<..., 4, 8, false>; max_D > 128 keeps its previous instantiation.

Every case checks the update against an exact reference. Gradients are multiples of 1/8, so their per-row sums are
exact in FP32, and the rowwise-Adagrad step recomputed in FP32 must agree:
  - every touched row is updated (weights or momentum) and every untouched row is bit-identical;
  - weights within 1 FP16 ulp with nearest rounding, or 1.25 ulp with stochastic rounding (an SR result may land
    on either FP16 neighbour of the FP32 reference, plus a quarter ulp of margin for the reference's own rounding);
  - momentum within 1e-6 relative.
The lane-to-column mapping changes, so the order of the rowwise sum of squares across lanes changes: the rounding
points move and the result is not bit-identical to <..., 4, 8, false>, but it stays within the bounds above.

The gates are read once per process (function-local statics on the first backward call), so this module sets them
before the first TBE call and all its tests run with them ON. The gate-OFF and "eighth wave off" cases are separate
test modules (separate processes): hip_mixed_d_eighth_wave_test.py (occupancy unset: max_D 96 stays on <4, 8>) and
hip_mixed_d_occupancy_gate_test.py (occupancy set without the eighth wave: nothing changes).
"""

import os
import unittest

os.environ.setdefault("FBGEMM_NO_JK", "1")
os.environ["FBGEMM_TBE_ROCM_HIP_BACKWARD_KERNEL"] = "1"
os.environ["TBE_ROCM_HIP_MIXED_D_EIGHTH_WAVE"] = "1"
os.environ["TBE_ROCM_HIP_MIXED_D_UNIFORM"] = "1"
os.environ["TBE_ROCM_HIP_MIXED_D_OCCUPANCY"] = "1"

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

ROWS = [4000, 3000, 2000, 1000]
B = 512
LR, EPS = 0.021, 1e-4
EIGHTH_WAVE = ", 4, 8, false>"
OCCUPANCY = ", 3, 8, false>"


def _rocm() -> bool:
    return torch.cuda.is_available() and torch.version.hip is not None


def _build(dims: list[int], sr: bool) -> SplitTableBatchedEmbeddingBagsCodegen:
    tbe = SplitTableBatchedEmbeddingBagsCodegen(
        embedding_specs=[
            (r, d, EmbeddingLocation.DEVICE, ComputeDevice.CUDA)
            for r, d in zip(ROWS, dims)
        ],
        optimizer=OptimType.EXACT_ROWWISE_ADAGRAD,
        weights_precision=SparseType.FP16,
        output_dtype=SparseType.FP32,
        stochastic_rounding=sr,
        pooling_mode=PoolingMode.SUM,
        bounds_check_mode=BoundsCheckMode.FATAL,
        learning_rate=LR,
        eps=EPS,
        embedding_table_index_type=torch.int32,
        embedding_table_offset_type=torch.int32,
    ).cuda()
    with torch.no_grad():
        for t, w in enumerate(tbe.split_embedding_weights()):
            g = torch.Generator(device="cuda").manual_seed(1234 + t)
            w.copy_(
                (torch.rand(w.shape, generator=g, device="cuda") * 0.1 - 0.05).to(
                    w.dtype
                )
            )
    return tbe


def _inputs(
    dims: list[int], batch: list[int], low_dup: bool
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """Per feature f: bag lengths, ids and a bag gradient of shape (batch[f], dims[f]) whose entries are multiples of
    1/8 in [-2, 2]. Heavy duplication: lengths 0-8, every 13th bag 48 ids, ids skewed to low rows (u^3). Low
    duplication: lengths 0-3 and uniform ids, so most touched rows get a single id."""
    g = torch.Generator().manual_seed(7)
    lens, ids, grads = [], [], []
    for f in range(len(dims)):
        if low_dup:
            L = torch.randint(0, 4, (batch[f],), generator=g)
            u = torch.rand(int(L.sum()), generator=g, dtype=torch.float64)
        else:
            L = torch.randint(0, 9, (batch[f],), generator=g)
            L[::13] = 48
            u = torch.rand(int(L.sum()), generator=g, dtype=torch.float64) ** 3
        lens.append(L)
        ids.append((u * ROWS[f]).long().clamp_(max=ROWS[f] - 1))
        grads.append(
            torch.randint(-16, 17, (batch[f], dims[f]), generator=g).float() / 8.0
        )
    return lens, ids, grads


def run_and_check(
    tc: unittest.TestCase,
    dims: list[int],
    *,
    sr: bool = False,
    low_dup: bool = False,
    vbe: bool = False,
) -> list[str]:
    """Run one forward + backward, check coverage and values against the exact reference (module docstring), and
    return the backward kernel names the profiler saw. With vbe=True the features get different batch sizes and
    the TBE runs in VBE mode on one rank, whose output is the features' (batch[f], dims[f]) blocks in feature order.
    """
    batch = [B, B // 2, B // 4, B] if vbe else [B] * len(dims)
    tbe = _build(dims, sr)
    lens, ids, grads = _inputs(dims, batch, low_dup)
    w0 = [w.detach().float().clone() for w in tbe.split_embedding_weights()]
    lengths = torch.cat(lens)
    offsets = torch.zeros(lengths.numel() + 1, dtype=torch.int64)
    offsets[1:] = torch.cumsum(lengths, 0)
    idx, off = torch.cat(ids).int().cuda(), offsets.int().cuda()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
        if vbe:
            out = tbe(idx, off, batch_size_per_feature_per_rank=[[b] for b in batch])
            out.backward(torch.cat([x.flatten() for x in grads]).cuda())
        else:
            out = tbe(idx, off)
            out.backward(torch.cat(grads, dim=1).cuda())
        torch.cuda.synchronize()
    names = sorted(
        {e.name for e in p.events() if "backward" in e.name and "kernel" in e.name}
    )

    tol = 1.25 if sr else 1.0
    for t, (w, m) in enumerate(
        zip(tbe.split_embedding_weights(), tbe.split_optimizer_states())
    ):
        G = torch.zeros(ROWS[t], dims[t])
        bag_of_id = torch.repeat_interleave(torch.arange(batch[t]), lens[t])
        G.index_add_(0, ids[t], grads[t][bag_of_id])
        touched = G.abs().sum(1) > 0
        m_ref = (G**2).mean(1)
        w_ref = w0[t].cpu() - (LR / (m_ref.sqrt() + EPS))[:, None] * G
        wf, mf, w0t = w.float().cpu(), m[0].float().view(-1).cpu(), w0[t].cpu()
        where = f"table {t} (D={dims[t]}, rows={ROWS[t]}, sr={sr}, low_dup={low_dup}, vbe={vbe})"
        changed = (wf != w0t).any(1) | (mf != 0)
        stray = int((changed & ~touched).sum())
        tc.assertEqual(stray, 0, f"{where}: {stray} untouched rows changed")
        missed = int((mf[touched] == 0).sum())
        tc.assertEqual(missed, 0, f"{where}: {missed} touched rows not updated")
        ulp = torch.exp2(torch.floor(torch.log2(w_ref.abs().clamp_min(2**-14))) - 10)
        err = float(((wf - w_ref).abs() / ulp)[touched].max())
        tc.assertLessEqual(
            err, tol, f"{where}: weight error {err:.3f} FP16 ulp > {tol}"
        )
        rel = float(((mf - m_ref).abs() / m_ref.clamp_min(1e-30))[touched].max())
        tc.assertLessEqual(
            rel, 1e-6, f"{where}: momentum relative error {rel:.2e} > 1e-6"
        )
    return names


@unittest.skipUnless(_rocm(), "ROCm only")
class HipMixedDOccupancyTest(unittest.TestCase):
    """Occupancy and eighth-wave gates ON (this module's process): which warp-backward instantiation each table shape
    gets, and that each one updates exactly as the reference does."""

    def _assert_occupancy(self, names: list[str]) -> None:
        self.assertTrue(
            any(OCCUPANCY in n for n in names),
            f"expected hip_mixed_d <..., 3, 8, false>, saw {names}",
        )
        self.assertFalse(
            any(EIGHTH_WAVE in n for n in names),
            f"unexpected <..., 4, 8, false> in {names}",
        )

    def test_mixed_d_max96_takes_occupancy_instantiation(self) -> None:
        """Mixed D 96/96/64/32 (max_D 96): the backward runs <..., 3, 8, false> and matches the exact reference."""
        self._assert_occupancy(run_and_check(self, [96, 96, 64, 32]))

    def test_low_duplication_takes_occupancy_instantiation(self) -> None:
        """About 1.2 ids per touched row (the low-duplication branch production inputs take): same instantiation,
        exact update."""
        self._assert_occupancy(run_and_check(self, [96, 96, 64, 32], low_dup=True))

    def test_vbe_takes_occupancy_instantiation(self) -> None:
        """A VBE TBE (features with batch sizes B, B/2, B/4, B): the VBE variant of <..., 3, 8, false> runs, exact
        update."""
        names = run_and_check(self, [96, 96, 64, 32], vbe=True)
        self._assert_occupancy(names)
        self.assertTrue(
            any("vbe" in n for n in names),
            f"expected the VBE kernel variant, saw {names}",
        )

    def test_stochastic_rounding_takes_occupancy_instantiation(self) -> None:
        """Stochastic rounding on: same instantiation; weights within 1.25 FP16 ulp of the exact reference."""
        self._assert_occupancy(run_and_check(self, [96, 96, 64, 32], sr=True))

    def test_uniform_d96_takes_occupancy_instantiation(self) -> None:
        """Uniform D=96 (not served by the specialized HIP kernel) with the uniform route on: <..., 3, 8, false>."""
        self._assert_occupancy(run_and_check(self, [96, 96, 96, 96]))

    def test_max_d_128_keeps_four_vectors(self) -> None:
        """max_D 128 > 96: the occupancy instantiation cannot hold a 128-wide row, so <..., 4, 8, false> runs."""
        names = run_and_check(self, [128, 96, 64, 32])
        self.assertTrue(
            any(EIGHTH_WAVE in n for n in names),
            f"expected <..., 4, 8, false>, saw {names}",
        )
        self.assertFalse(
            any(OCCUPANCY in n for n in names),
            f"unexpected <..., 3, 8, false> in {names}",
        )

    def test_max_d_above_128_keeps_its_previous_instantiation(self) -> None:
        """max_D 256: neither eighth-wave instantiation applies; the previous hip_mixed_d instantiation runs."""
        names = run_and_check(self, [256, 96, 64, 32])
        self.assertTrue(
            any("hip_mixed_d" in n for n in names), f"expected hip_mixed_d, saw {names}"
        )
        self.assertFalse(
            any(OCCUPANCY in n or EIGHTH_WAVE in n for n in names),
            f"unexpected eighth-wave kernel in {names}",
        )


if __name__ == "__main__":
    unittest.main()
