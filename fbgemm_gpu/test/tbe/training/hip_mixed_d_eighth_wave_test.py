# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Tests for the ROCm hip_mixed_d warp backward gates, with both gates ON:

  TBE_ROCM_HIP_MIXED_D_EIGHTH_WAVE=1  for max_D <= 128, launch the <..., 4, 8, false> instantiation (8-lane thread
                                      groups x 4 vectors per lane, eight rows per wave64) instead of <..., 1, 2, false>
                                      (32-lane groups x 1 vector, two rows per wave64);
  TBE_ROCM_HIP_MIXED_D_UNIFORM=1      send uniform-D non-VBE TBEs that the specialized HIP kernel does not take (for
                                      example D=96) to hip_mixed_d instead of the generic warp_per_row kernel.

Both gates, and FBGEMM_TBE_ROCM_HIP_BACKWARD_KERNEL, are read once per process (function-local statics at the first
TBE backward), so this module sets them before the first TBE call and every test here runs with them ON. The
gate-OFF and "route must not change" cases live in hip_mixed_d_routes_test.py, a separate test module and therefore
a separate process.

The eighth-wave optimization is wave64-only. Wave32 devices retain the full-warp <..., 1, 1, false> instantiation
and run the same coverage and numerical checks with the gates ON.

Every case runs one forward + backward of a TBE with FP16 weights, rowwise Adagrad (EXACT_ROWWISE_ADAGRAD), int32
indices and offsets and SUM pooling, and checks:
  1. Route: the kernel names the profiler saw prove which instantiation ran. On libraries without the change this
     assertion fails (<..., 1, 2, false> or the generic kernel), which is what makes it a regression test.
  2. Coverage: every row that received a gradient is updated (momentum != 0), and every row that did not is
     bit-identical to its initial value (weights and momentum).
  3. Values: weights and momentum match an exact reference.
     - Gradients are multiples of 1/8 in [-2, 2], so each row's gradient sum and the mean of its squares are exact
       in FP32 whatever the summation order. The kernel's sum-of-squares reduction order does change with the lane
       layout; on these inputs that cannot show, and in general it moves momentum only in its last bits.
     - The reference recomputes the rowwise Adagrad step in FP32 from the initial FP16 weights. With nearest rounding
       the only rounding left is the final FP16 store, hence 1 FP16 ulp. With stochastic rounding the store picks one
       of the two FP16 neighbours of the FP32 result, so the error stays below one FP16 spacing; 1.25 ulp leaves room
       for the reference's own FP32 rounding at binade edges. Momentum is FP32 and gets 1e-6 relative.

The input shapes cover the hip_mixed_d paths that matter: heavy duplication (ids skewed to low rows, bags of 0-8 ids
plus 48-id bags, about 2.6 ids per touched row), low duplication (uniform ids, bags of 0-3 ids, about 1.2 ids per
touched row, close to production embedding traffic), a VBE TBE (different batch size per feature), and stochastic
rounding.
"""

import os
import unittest

os.environ.setdefault("FBGEMM_NO_JK", "1")
os.environ["FBGEMM_TBE_ROCM_HIP_BACKWARD_KERNEL"] = "1"
os.environ["TBE_ROCM_HIP_MIXED_D_EIGHTH_WAVE"] = "1"
os.environ["TBE_ROCM_HIP_MIXED_D_UNIFORM"] = "1"

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
    return the HIP backward kernel names the profiler saw. With vbe=True the features get different batch sizes and
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
        {e.name for e in p.events() if "hip_" in e.name and "backward" in e.name}
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
class HipMixedDEighthWaveTest(unittest.TestCase):
    def _assert_eighth_wave(self, names: list[str]) -> None:
        warp_size = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).warp_size
        self.assertIn(warp_size, (32, 64))
        expected = EIGHTH_WAVE if warp_size == 64 else ", 1, 1, false>"
        self.assertTrue(
            any("hip_mixed_d" in n and expected in n for n in names),
            f"expected hip_mixed_d <...{expected} on wave{warp_size}, saw {names}",
        )
        if warp_size == 32:
            self.assertFalse(any(EIGHTH_WAVE in n for n in names), names)

    def test_mixed_d_takes_eighth_wave_and_matches_exact_reference(self) -> None:
        """Mixed D 96/96/64/32 (max_D 96 <= 128), heavy duplication: the warp backward must run as
        hip_mixed_d <..., 4, 8, false>, and the rowwise Adagrad update must match the exact reference.
        """
        self._assert_eighth_wave(run_and_check(self, [96, 96, 64, 32]))

    def test_low_duplication_takes_eighth_wave(self) -> None:
        """Mixed D 96/96/64/32 with low duplication (about 1.2 ids per touched row, as in production traffic): same
        route and the same exact update on the kernel's low-duplication path."""
        self._assert_eighth_wave(run_and_check(self, [96, 96, 64, 32], low_dup=True))

    def test_vbe_takes_eighth_wave(self) -> None:
        """VBE TBE (batch 512/256/128/512 per feature), mixed D 96/96/64/32: the VBE hip_mixed_d kernel must run as
        <..., 4, 8, false>, with the exact update; the gate covers VBE as well as non-VBE.
        """
        names = run_and_check(self, [96, 96, 64, 32], vbe=True)
        self._assert_eighth_wave(names)
        self.assertTrue(
            any("vbe" in n for n in names), f"expected a VBE kernel, saw {names}"
        )

    def test_stochastic_rounding_takes_eighth_wave(self) -> None:
        """Mixed D 96/96/64/32 with stochastic rounding: same route; every touched row updated, untouched rows
        bit-identical, and weights within 1.25 FP16 ulp of the exact FP32 result (one of its two FP16 neighbours).
        """
        self._assert_eighth_wave(run_and_check(self, [96, 96, 64, 32], sr=True))

    def test_uniform_d96_takes_hip_mixed_d_eighth_wave(self) -> None:
        """Uniform D=96 (not taken by the specialized HIP kernel) with UNIFORM=1 and EIGHTH_WAVE=1: the table must
        leave the generic warp_per_row kernel for hip_mixed_d <..., 4, 8, false>, with an exact update.
        """
        self._assert_eighth_wave(run_and_check(self, [96, 96, 96, 96]))

    def test_max_d_above_128_keeps_its_previous_instantiation(self) -> None:
        """Mixed D 256/96/64/32 (max_D 256 > 128): the gate only acts inside the max_D <= 128 branch, so the warp
        backward must NOT be the eighth-wave instantiation; the update must still match the exact reference.
        """
        names = run_and_check(self, [256, 96, 64, 32])
        self.assertTrue(
            any("hip_mixed_d" in n for n in names), f"expected hip_mixed_d, saw {names}"
        )
        self.assertFalse(
            any(EIGHTH_WAVE in n for n in names),
            f"unexpected <..., 4, 8, false> in {names}",
        )


if __name__ == "__main__":
    unittest.main()
