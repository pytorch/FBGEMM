# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Tests for TBE_ROCM_LINEARIZE_GROUP=8: narrow lane groups in linearize_index_kernel's bag loop.

Stock, each wave walks its 64 bags one at a time with all 64 lanes on one bag. With the gate (8, 16 or 32), a wave
works on 64/G bags at a time with G lanes each. Every output element is still written exactly once, with the same
value, so the outputs must not change at all.

The gate is read once per process, so this module sets it (8) before the first call; invalid values are tested in
linearize_lane_group_invalid_test.py, a separate process. The lane group is a runtime kernel argument, so the profiler
cannot show which loop ran; these tests guard bit-identity, and the speed-up at production-like sizes (in the diff
summary) is the evidence that the narrow loop runs.
"""

import math
import os
import unittest

os.environ["TBE_ROCM_LINEARIZE_GROUP"] = "8"

import torch  # noqa: E402
from fbgemm_gpu.split_embedding_configs import (  # noqa: E402
    EmbOptimType as OptimType,
    SparseType,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (  # noqa: E402
    EmbeddingLocation,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (  # noqa: E402
    ComputeDevice,
    SplitTableBatchedEmbeddingBagsCodegen,
)


def _rocm() -> bool:
    return torch.cuda.is_available() and torch.version.hip is not None


def _diff(what: str, got: torch.Tensor, want: torch.Tensor) -> str:
    """Assertion message: the number of differing elements and the first one (or the shapes, if they differ)."""
    if got.shape != want.shape:
        return f"{what}: shape {tuple(got.shape)} != {tuple(want.shape)}"
    bad = (got != want).nonzero()
    if bad.numel() == 0:
        return f"{what}: equal"
    i = tuple(bad[0].tolist())
    return (
        f"{what}: {bad.shape[0]} of {want.numel()} elements differ; first at {i}: "
        f"got {got[i].item()}, want {want[i].item()}"
    )


def check_transpose(tc: unittest.TestCase, seed: int, T: int, B: int) -> None:
    """transpose_embedding_input on T non-VBE tables x B bags (0-24 ids, every 37th bag 150 ids, int32) must equal
    the reference exactly: linear indices = hash_size_cumsum[t] + index, infos = (t << info_B_num_bits) | b, their
    stable sort by linear index, and the run-length encoding (only the defined prefix of `run`, up to num_runs).
    """
    g = torch.Generator().manual_seed(seed)
    hash_sizes = torch.randint(50, 5000, (T,), generator=g)
    lens = torch.randint(0, 25, (T * B,), generator=g)
    lens[::37] = 150
    offsets = torch.zeros(T * B + 1, dtype=torch.int64)
    offsets[1:] = torch.cumsum(lens, 0)
    t_of_id = torch.repeat_interleave(torch.arange(T), lens.view(T, B).sum(1))
    b_of_id = torch.repeat_interleave(torch.arange(T * B) % B, lens)
    indices = (torch.rand(int(lens.sum()), generator=g) * hash_sizes[t_of_id]).long()
    hsc = torch.zeros(T + 1, dtype=torch.int64)
    hsc[1:] = torch.cumsum(hash_sizes, 0)
    bits = int(math.log2(int(hsc[-1])) + 1)
    info_B_num_bits, info_B_mask = torch.ops.fbgemm.get_infos_metadata(hsc.cuda(), B, T)
    linear, linear_sorted, infos_sorted, run, run_lengths, num_runs, _ = (
        torch.ops.fbgemm.transpose_embedding_input(
            hsc.cuda(),
            bits,
            indices.int().cuda(),
            offsets.int().cuda(),
            info_B_num_bits=info_B_num_bits,
            info_B_mask=info_B_mask,
        )
    )
    linear_ref = (hsc[t_of_id] + indices).int()
    infos_ref = ((t_of_id << info_B_num_bits) | b_of_id).int()
    order = torch.sort(linear_ref.long(), stable=True).indices
    sorted_ref = linear_ref[order]
    run_ref, run_lengths_ref = torch.unique_consecutive(sorted_ref, return_counts=True)
    pairs = [
        ("linear indices", linear.cpu(), linear_ref),
        ("sorted linear indices", linear_sorted.cpu(), sorted_ref),
        ("sorted infos", infos_sorted.cpu().int(), infos_ref[order]),
    ]
    for what, got, want in pairs:
        tc.assertTrue(torch.equal(got, want), _diff(what, got, want))
    n = int(num_runs.cpu()[0])
    tc.assertEqual(n, run_ref.numel(), "num_runs")
    got_run, got_len = run.cpu()[:n], run_lengths.cpu()[:n].long()
    tc.assertTrue(torch.equal(got_run, run_ref), _diff("run", got_run, run_ref))
    tc.assertTrue(
        torch.equal(got_len, run_lengths_ref),
        _diff("run lengths", got_len, run_lengths_ref),
    )


@unittest.skipUnless(_rocm(), "ROCm only")
class LinearizeLaneGroupTest(unittest.TestCase):
    def test_outputs_match_reference(self) -> None:
        """6 non-VBE tables x 4,096 bags with lane group 8: every transpose_embedding_input output equals the
        reference exactly."""
        check_transpose(self, seed=5, T=6, B=4096)

    def test_small_batch_matches_reference(self) -> None:
        """3 tables x 37 bags, so the last wave is partly empty: the narrow loop must not read or write past the
        last bag."""
        check_transpose(self, seed=9, T=3, B=37)

    def test_vbe_backward_updates_exactly(self) -> None:
        """VBE TBE (batch 512/256/128 per feature, D 64/32/16, FP32 weights, rowwise Adagrad, int32) with lane
        group 8: linearize also runs on the VBE path, so one backward must update every touched row exactly as the
        FP32 reference does (gradients are multiples of 1/8, so the per-row sums are exact; tolerance 1e-6 relative)
        and leave every untouched row bit-identical."""
        rows, dims, batch = [3000, 2000, 1000], [64, 32, 16], [512, 256, 128]
        lr, eps = 0.021, 1e-4
        tbe = SplitTableBatchedEmbeddingBagsCodegen(
            [
                (r, d, EmbeddingLocation.DEVICE, ComputeDevice.CUDA)
                for r, d in zip(rows, dims)
            ],
            optimizer=OptimType.EXACT_ROWWISE_ADAGRAD,
            weights_precision=SparseType.FP32,
            learning_rate=lr,
            eps=eps,
        ).cuda()
        g = torch.Generator().manual_seed(13)
        lens, ids, grads = [], [], []
        for f in range(3):
            L = torch.randint(0, 9, (batch[f],), generator=g)
            lens.append(L)
            ids.append(torch.randint(0, rows[f], (int(L.sum()),), generator=g))
            grads.append(
                torch.randint(-16, 17, (batch[f], dims[f]), generator=g).float() / 8.0
            )
        w0 = [w.detach().clone().cpu() for w in tbe.split_embedding_weights()]
        lengths = torch.cat(lens)
        offsets = torch.zeros(lengths.numel() + 1, dtype=torch.int64)
        offsets[1:] = torch.cumsum(lengths, 0)
        out = tbe(
            torch.cat(ids).int().cuda(),
            offsets.int().cuda(),
            batch_size_per_feature_per_rank=[[b] for b in batch],
        )
        out.backward(torch.cat([x.flatten() for x in grads]).cuda())
        torch.cuda.synchronize()
        for t, w in enumerate(tbe.split_embedding_weights()):
            G = torch.zeros(rows[t], dims[t])
            G.index_add_(
                0,
                ids[t],
                grads[t][torch.repeat_interleave(torch.arange(batch[t]), lens[t])],
            )
            touched = G.abs().sum(1) > 0
            w_ref = w0[t] - (lr / ((G**2).mean(1).sqrt() + eps))[:, None] * G
            wc = w.detach().cpu()
            stray = int(((wc != w0[t]).any(1) & ~touched).sum())
            self.assertEqual(stray, 0, f"table {t}: {stray} untouched rows changed")
            rel = float(
                ((wc - w_ref).abs() / w_ref.abs().clamp_min(1e-6))[touched].max()
            )
            self.assertLessEqual(
                rel,
                1e-6,
                f"table {t} (D={dims[t]}): weight relative error {rel:.2e} > 1e-6",
            )


if __name__ == "__main__":
    unittest.main()
