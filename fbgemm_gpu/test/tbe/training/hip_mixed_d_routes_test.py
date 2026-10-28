# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Route tests for the ROCm hip_mixed_d warp backward gates: what must NOT change.

This module runs with the HIP backward and TBE_ROCM_HIP_MIXED_D_UNIFORM=1 ON, but TBE_ROCM_HIP_MIXED_D_EIGHTH_WAVE
OFF. The gates are read once per process, so these cases cannot share a process with hip_mixed_d_eighth_wave_test.py
(which turns EIGHTH_WAVE on); a separate test module gives them their own process.

  - With EIGHTH_WAVE off, a mixed-D table with max_D <= 128 runs 32-lane groups: <..., 1, 2, false> on wave64 and
    <..., 1, 1, false> on wave32.
  - With UNIFORM on, a uniform D that the specialized HIP kernel covers (D=128) still runs that specialized kernel,
    not hip_mixed_d, on supported CDNA devices. Other architectures retain hip_mixed_d.

Each case also checks the update against the same exact reference as hip_mixed_d_eighth_wave_test.py: touched rows
updated, untouched rows bit-identical, weights within 1 FP16 ulp, momentum within 1e-6 relative.
"""

import os
import unittest

os.environ.setdefault("FBGEMM_NO_JK", "1")
os.environ["FBGEMM_TBE_ROCM_HIP_BACKWARD_KERNEL"] = "1"
os.environ.pop("TBE_ROCM_HIP_MIXED_D_EIGHTH_WAVE", None)
os.environ["TBE_ROCM_HIP_MIXED_D_UNIFORM"] = "1"

import torch  # noqa: E402

# The helpers below are a copy of hip_mixed_d_eighth_wave_test.py's: importing that module would set
# TBE_ROCM_HIP_MIXED_D_EIGHTH_WAVE=1 in this process, which is exactly what these route cases must not have.
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


def _rocm() -> bool:
    return torch.cuda.is_available() and torch.version.hip is not None


def _build(dims: list[int]) -> SplitTableBatchedEmbeddingBagsCodegen:
    tbe = SplitTableBatchedEmbeddingBagsCodegen(
        embedding_specs=[
            (r, d, EmbeddingLocation.DEVICE, ComputeDevice.CUDA)
            for r, d in zip(ROWS, dims)
        ],
        optimizer=OptimType.EXACT_ROWWISE_ADAGRAD,
        weights_precision=SparseType.FP16,
        output_dtype=SparseType.FP32,
        stochastic_rounding=False,
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
    dims: list[int],
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """Per feature: bag lengths 0-8 with every 13th bag at 48 ids, ids skewed to low rows (u^3), and a bag gradient
    whose entries are multiples of 1/8 in [-2, 2]."""
    g = torch.Generator().manual_seed(7)
    lens, ids, grads = [], [], []
    for f in range(len(dims)):
        L = torch.randint(0, 9, (B,), generator=g)
        L[::13] = 48
        u = torch.rand(int(L.sum()), generator=g, dtype=torch.float64)
        lens.append(L)
        ids.append((u**3 * ROWS[f]).long().clamp_(max=ROWS[f] - 1))
        grads.append(torch.randint(-16, 17, (B, dims[f]), generator=g).float() / 8.0)
    return lens, ids, grads


def run_and_check(tc: unittest.TestCase, dims: list[int]) -> list[str]:
    """Run one forward + backward of a TBE with `dims`, check coverage and values against the exact reference
    (module docstring), and return the hip_mixed_d / HIP kernel names the profiler saw.
    """
    tbe = _build(dims)
    lens, ids, grads = _inputs(dims)
    w0 = [w.detach().float().clone() for w in tbe.split_embedding_weights()]
    lengths = torch.cat(lens)
    offsets = torch.zeros(lengths.numel() + 1, dtype=torch.int64)
    offsets[1:] = torch.cumsum(lengths, 0)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
        out = tbe(torch.cat(ids).int().cuda(), offsets.int().cuda())
        out.backward(torch.cat(grads, dim=1).cuda())
        torch.cuda.synchronize()
    names = sorted(
        {e.name for e in p.events() if "hip_" in e.name and "backward" in e.name}
    )

    for t, (w, m) in enumerate(
        zip(tbe.split_embedding_weights(), tbe.split_optimizer_states())
    ):
        G = torch.zeros(ROWS[t], dims[t])
        G.index_add_(
            0, ids[t], grads[t][torch.repeat_interleave(torch.arange(B), lens[t])]
        )
        touched = G.abs().sum(1) > 0
        m_ref = (G**2).mean(1)
        w_ref = w0[t].cpu() - (LR / (m_ref.sqrt() + EPS))[:, None] * G
        wf, mf, w0t = w.float().cpu(), m[0].float().view(-1).cpu(), w0[t].cpu()
        where = f"table {t} (D={dims[t]}, rows={ROWS[t]})"
        changed = (wf != w0t).any(1) | (mf != 0)
        stray = int((changed & ~touched).sum())
        tc.assertEqual(stray, 0, f"{where}: {stray} untouched rows changed")
        missed = int((mf[touched] == 0).sum())
        tc.assertEqual(missed, 0, f"{where}: {missed} touched rows not updated")
        ulp = torch.exp2(torch.floor(torch.log2(w_ref.abs().clamp_min(2**-14))) - 10)
        err = float(((wf - w_ref).abs() / ulp)[touched].max())
        tc.assertLessEqual(err, 1.0, f"{where}: weight error {err:.3f} FP16 ulp > 1")
        rel = float(((mf - m_ref).abs() / m_ref.clamp_min(1e-30))[touched].max())
        tc.assertLessEqual(
            rel, 1e-6, f"{where}: momentum relative error {rel:.2e} > 1e-6"
        )
    return names


@unittest.skipUnless(_rocm(), "ROCm only")
class HipMixedDRoutesTest(unittest.TestCase):
    def test_gate_off_mixed_d_keeps_32_lane_kernel(self) -> None:
        """EIGHTH_WAVE unset, mixed D 96/96/64/32: retain 32-lane groups and exclude the eighth-wave instantiation."""
        names = run_and_check(self, [96, 96, 64, 32])
        warp_size = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).warp_size
        self.assertIn(warp_size, (32, 64))
        expected = f", 1, {warp_size // 32}, false>"
        self.assertTrue(any("hip_mixed_d" in n and expected in n for n in names), names)
        self.assertFalse(any(", 4, 8, false>" in n for n in names), names)

    def test_uniform_d_in_specialized_list_keeps_specialized_kernel(self) -> None:
        """UNIFORM=1, D=128: supported CDNA devices retain the specialized kernel; others retain hip_mixed_d."""
        names = run_and_check(self, [128, 128, 128, 128])
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        arch = props.gcnArchName.split(":")[0]
        if arch in {"gfx90a", "gfx942", "gfx950"}:
            self.assertTrue(any("hip_split_embedding" in n for n in names), names)
            self.assertFalse(any("hip_mixed_d" in n for n in names), names)
        else:
            self.assertIn(props.warp_size, (32, 64))
            expected = f", 1, {props.warp_size // 32}, false>"
            self.assertTrue(
                any("hip_mixed_d" in n and expected in n for n in names), names
            )
            self.assertFalse(any("hip_split_embedding" in n for n in names), names)


if __name__ == "__main__":
    unittest.main()
