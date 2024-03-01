# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import fbgemm_gpu.metrics
import hypothesis.strategies as st
import torch
from hypothesis import given, settings

try:
    # pyre-ignore[21]
    from fbgemm_gpu import open_source  # noqa: F401

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


if __name__ == "__main__":
    unittest.main()
