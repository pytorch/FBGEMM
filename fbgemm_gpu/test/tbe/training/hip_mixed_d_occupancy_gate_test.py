# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Route test for TBE_ROCM_HIP_MIXED_D_OCCUPANCY without the eighth-wave route: nothing may change.

The occupancy instantiation is reached only inside the eighth-wave branch, so with TBE_ROCM_HIP_MIXED_D_OCCUPANCY=1
but TBE_ROCM_HIP_MIXED_D_EIGHTH_WAVE (and _UNIFORM) unset, a mixed-D table with max_D 96 must still run the existing
<..., 1, 2, false> hip_mixed_d instantiation. The gates are read once per process, hence a separate module. The
update is checked against the same exact reference as hip_mixed_d_occupancy_test.py, whose helpers are copied here
(importing that module would set the gates in this process).
"""

import os
import unittest

os.environ.setdefault("FBGEMM_NO_JK", "1")
os.environ["FBGEMM_TBE_ROCM_HIP_BACKWARD_KERNEL"] = "1"
os.environ["TBE_ROCM_HIP_MIXED_D_OCCUPANCY"] = "1"
os.environ.pop("TBE_ROCM_HIP_MIXED_D_EIGHTH_WAVE", None)
os.environ.pop("TBE_ROCM_HIP_MIXED_D_UNIFORM", None)

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
class HipMixedDOccupancyGateTest(unittest.TestCase):
    """Occupancy set, eighth wave unset (this module's process): the warp backward keeps its previous kernel."""

    def test_occupancy_without_eighth_wave_keeps_32_lane_kernel(self) -> None:
        """Mixed D 96/96/64/32: hip_mixed_d <..., 1, 2, false> runs, neither eighth-wave instantiation does, and the
        update matches the exact reference."""
        names = run_and_check(self, [96, 96, 64, 32])
        self.assertTrue(
            any("hip_mixed_d" in n and ", 1, 2, false>" in n for n in names),
            f"expected hip_mixed_d <..., 1, 2, false>, saw {names}",
        )
        self.assertFalse(
            any(OCCUPANCY in n or EIGHTH_WAVE in n for n in names),
            f"unexpected eighth-wave kernel in {names}",
        )


if __name__ == "__main__":
    unittest.main()
