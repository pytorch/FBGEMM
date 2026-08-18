#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# pyre-ignore-all-errors[56]

import unittest

import torch
from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    ComputeDevice,
    EmbeddingLocation,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    SplitTableBatchedEmbeddingBagsCodegen,
)

try:
    # pyre-ignore[21]
    from test_utils import gpu_unavailable
except Exception:
    from fbgemm_gpu.test.test_utils import gpu_unavailable


@unittest.skipIf(*gpu_unavailable)
class NFP8EncodingTest(unittest.TestCase):
    """Pins the invariant that the NFP8 encoding the device kernel uses matches
    the encoding the host tensor's dtype label advertises.

    This matters because FP8 e4m3 has two encodings that share a bit layout but
    differ in exponent bias, so the same byte decodes to a 2x different value:

        byte 0x7A -> 320.0 as float8_e4m3fn   (bias 7, max finite 448)
        byte 0x7A -> 160.0 as float8_e4m3fnuz (bias 8, max finite 240)

    On ROCm the label is arch-dependent (getNFP8ScalarType: fnuz on gfx94x and
    gfx90a, OCP fn on gfx950) while only the fnuz kernel variant is instantiated,
    so host tensors are relabeled to fnuz at the dispatch boundary by
    relabel_nfp8_for_dispatch. That relabel must NOT change the numerics: the
    physical encoding is bound per-arch in device code via the __nv_fp8_e4m3
    alias. This test fails if the relabel ever starts leaking into the math.

    Written arch-agnostically -- it compares the device round-trip against a host
    round-trip through the tensor's own dtype, so it is a valid check on gfx942
    (fnuz), gfx950 (fn) and CUDA (fn) alike.
    """

    def test_device_encoding_matches_host_label(self) -> None:
        num_embeddings, embedding_dim = 64, 32

        tbe = SplitTableBatchedEmbeddingBagsCodegen(
            [
                (
                    num_embeddings,
                    embedding_dim,
                    EmbeddingLocation.DEVICE,
                    ComputeDevice.CUDA,
                )
            ],
            weights_precision=SparseType.NFP8,
            output_dtype=SparseType.FP32,
            device=torch.cuda.current_device(),
        )
        weights = tbe.split_embedding_weights()[0]

        # The weight buffer must carry an FP8 e4m3 label, and it must be the
        # arch-correct one -- never silently relabeled for the caller.
        self.assertIn(
            weights.dtype,
            (torch.float8_e4m3fn, torch.float8_e4m3fnuz),
            "NFP8 weights must be labeled with an FP8 e4m3 dtype",
        )

        # 224.0 = 1.75 * 2^7 is exactly representable in BOTH encodings (fn max
        # 448, fnuz max 240), so the probe is arch-safe, while still exercising
        # the top of the exponent range where a bias-7 vs bias-8 mismatch is
        # glaring. Do not use an fn-only value such as 320.0 here: it overflows
        # fnuz to NaN on gfx942 and assert_close treats NaN != NaN.
        probe = torch.tensor([0.5, 1.0, 2.0, 224.0], device=torch.cuda.current_device())
        row = probe.repeat(embedding_dim // probe.numel())
        weights.copy_(row.expand(num_embeddings, embedding_dim).to(weights.dtype))

        # Reference: round trip on the host through the SAME dtype the tensor
        # advertises. If the device honors the label, it must agree.
        expected = row.to(weights.dtype).float()

        indices = torch.zeros(1, device=torch.cuda.current_device(), dtype=torch.int64)
        offsets = torch.tensor(
            [0, 1], device=torch.cuda.current_device(), dtype=torch.int64
        )
        actual = tbe(indices, offsets)[0]

        torch.testing.assert_close(
            actual,
            expected,
            rtol=0,
            atol=0,
            msg=lambda m: (
                f"Device FP8 encoding disagrees with the host dtype label "
                f"({weights.dtype}). This usually means a kernel is reading the "
                f"weights with the other e4m3 bias. {m}"
            ),
        )
