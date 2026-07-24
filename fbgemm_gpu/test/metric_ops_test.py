# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[56]

import unittest

import fbgemm_gpu.metrics
import hypothesis.strategies as st
import torch
from hypothesis import given, settings

try:
    # pyre-ignore[21]
    from fbgemm_gpu import open_source  # noqa: F401

    # pyre-ignore[21]
    from test_utils import gpu_memory_lt_gb, gpu_unavailable
except Exception:
    from fbgemm_gpu.test.test_utils import gpu_memory_lt_gb, gpu_unavailable

    try:
        torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:metric_ops")
    except Exception:
        if torch.version.hip:
            torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:metric_ops_hip")
        else:
            torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:metric_ops")


class MetricOpsTest(unittest.TestCase):
    @unittest.skipIf(
        True,
        "Test is sometimes failed due to issues with Flaky. Skipping until the issues are resolved. ",
    )
    # pyre-ignore [56]
    @given(
        n_tasks=st.integers(1, 5),
        batch_size=st.integers(1, 1024),
        dtype=st.sampled_from([torch.half, torch.float, torch.double]),
    )
    @settings(max_examples=20, deadline=None)
    def test_auc(self, n_tasks: int, batch_size: int, dtype: torch.dtype) -> None:
        predictions = torch.randint(0, 1000, (n_tasks, batch_size)).to(dtype).cuda()
        labels = torch.randint(0, 1000, (n_tasks, batch_size)).to(dtype).cuda() / 1000.0
        weights = torch.rand(n_tasks, batch_size).to(dtype).cuda()

        compute_auc = fbgemm_gpu.metrics.Auc()
        output_ref = compute_auc(n_tasks, predictions, labels, weights)
        output = fbgemm_gpu.metrics.auc(n_tasks, predictions, labels, weights)

        # Explicitly convert type based on output_ref's dtype
        output = output.to(output_ref.dtype)

        # Test correctness only if output_ref does not product nan or inf
        if not (torch.isnan(output_ref).any() or torch.isinf(output_ref).any()):
            torch.testing.assert_close(
                output_ref,
                output,
                rtol=1e-2 if dtype == torch.half else None,
                atol=1e-2 if dtype == torch.half else None,
            )

    @unittest.skipIf(*gpu_unavailable)
    @unittest.skipIf(*gpu_memory_lt_gb(2))
    def test_batch_auc_large_grid_torch_check(self) -> None:
        """
        Asserts the host-side TORCH_CHECK in batch_auc fires when the
        auc_kernel launch would exceed the HIP 2^32 thread-per-launch
        limit.

        auc_kernel cannot be converted to a grid-stride loop without
        redesigning its inter-block prefix-sum scan (block_flags +
        spinwait coordination). Tier-3 follow-up.

        Sizing: grid_size = num_blocks * num_tasks. With num_entries = 1,
        num_blocks = 1, so grid_size = num_tasks. Pick num_tasks =
        (1 << 24) + 1; total threads = (2**24 + 1) * 256 > 2**32.
        """
        num_tasks = (1 << 24) + 1
        num_entries = 1
        device = torch.accelerator.current_accelerator()
        predictions = torch.zeros(
            (num_tasks, num_entries),
            dtype=torch.float32,
            device=device,
        )
        labels = torch.zeros(
            (num_tasks, num_entries),
            dtype=torch.float32,
            device=device,
        )
        weights = torch.ones(
            (num_tasks, num_entries),
            dtype=torch.float32,
            device=device,
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "auc_kernel launch would exceed HIP 2\\^32 thread-per-launch limit",
        ):
            _ = fbgemm_gpu.metrics.auc(num_tasks, predictions, labels, weights)

    @unittest.skipIf(*gpu_unavailable)
    def test_batch_auc_small_scale_correctness(self) -> None:
        """
        Tier-B small-scale correctness check for the AUC kernel,
        complementing the cap-trip Tier-C assertRaisesRegex above. This
        verifies that the kernel still produces correct AUC values when
        the cap is not hit.

        Python reference: per-task AUC = U / (P * N) where U is the
        Mann-Whitney U statistic with weights, computed via a sort-and-
        count-pairs over (positive, negative) pairs.
        """
        device = torch.accelerator.current_accelerator()
        num_tasks = 2
        num_entries = 8
        # Sentinel values: distinct predictions so AUC is well-defined.
        predictions_cpu = torch.tensor(
            [
                [0.1, 0.4, 0.35, 0.8, 0.05, 0.7, 0.2, 0.9],
                [0.5, 0.6, 0.1, 0.3, 0.9, 0.45, 0.55, 0.25],
            ],
            dtype=torch.float32,
        )
        labels_cpu = torch.tensor(
            [
                [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        weights_cpu = torch.ones((num_tasks, num_entries), dtype=torch.float32)

        result = fbgemm_gpu.metrics.auc(
            num_tasks,
            predictions_cpu.to(device),
            labels_cpu.to(device),
            weights_cpu.to(device),
        )

        # Python reference: per-task AUC via Mann-Whitney U.
        ref_aucs = []
        for t in range(num_tasks):
            preds = predictions_cpu[t]
            lbls = labels_cpu[t]
            wts = weights_cpu[t]
            pos_idx = (lbls > 0).nonzero(as_tuple=True)[0]
            neg_idx = (lbls == 0).nonzero(as_tuple=True)[0]
            num = 0.0
            den = 0.0
            for p in pos_idx:
                for n in neg_idx:
                    w = float(wts[p].item()) * float(wts[n].item())
                    if preds[p] > preds[n]:
                        num += w
                    elif preds[p] == preds[n]:
                        num += 0.5 * w
                    den += w
            ref_aucs.append(num / den if den > 0 else 0.0)
        ref = torch.tensor(ref_aucs, dtype=torch.float32)
        torch.testing.assert_close(result.cpu().view(-1), ref, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
