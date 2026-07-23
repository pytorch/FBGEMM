# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import hypothesis.strategies as st
import torch
from hypothesis import given, HealthCheck, settings

from . import common  # noqa E402

# pyre-fixme[21]: Could not find name `open_source` in
#  `deeplearning.fbgemm.fbgemm_gpu.test.quantize.common`.
from .common import open_source

# pyre-fixme[16]: Module `common` has no attribute `open_source`.
if open_source:
    # pyre-ignore[21]
    from test_utils import gpu_memory_lt_gb, gpu_unavailable
else:
    from fbgemm_gpu.test.test_utils import gpu_memory_lt_gb, gpu_unavailable


class TestMSFPQuantizationConversion(unittest.TestCase):
    # pyrefly: ignore [bad-argument-type]
    @unittest.skipIf(*gpu_unavailable)
    # pyre-ignore [56]: Invalid decoration, was not able to infer the type of argument
    @given(
        nrows=st.integers(min_value=0, max_value=100),
        ncols=st.integers(min_value=0, max_value=100),
    )
    @settings(deadline=None, suppress_health_check=[HealthCheck.filter_too_much])
    def test_quantize_op(self, nrows: int, ncols: int) -> None:
        ebits = 8
        mbits = 7
        bias = 127
        max_pos = (1 << ((1 << ebits) - 2 - bias)) * (2 - 2 ** (-mbits))
        min_pos = 2 ** (1 - bias - mbits)
        bounding_box_size = 16
        print("MSFP parameters", bounding_box_size, ebits, mbits, bias)
        input_data = torch.rand(nrows, ncols).float()
        quantized_data = torch.ops.fbgemm.FloatToMSFPQuantized(
            input_data.cuda(),
            bounding_box_size,
            ebits,
            mbits,
            bias,
            min_pos,
            max_pos,
        )
        dequantized_data = torch.ops.fbgemm.MSFPQuantizedToFloat(
            quantized_data.cuda(), ebits, mbits, bias
        )
        torch.testing.assert_close(dequantized_data.cpu(), input_data, rtol=1, atol=0)

    # pyrefly: ignore [bad-argument-type]
    @unittest.skipIf(*gpu_unavailable)
    # pyre-ignore[56]: Pyre cannot infer the type of `gpu_memory_lt_gb`
    # through the open-source / non-open-source import branch above.
    @unittest.skipIf(*gpu_memory_lt_gb(2))
    def test_float_to_msfp_large_grid(self) -> None:
        """
        Reproduces the HIP grid-overflow bug in
        `_compute_msfp_shared_exponent_cuda_kernel`
        (quantize_ops/quantize_msfp.cu line ~161).

        Block: dim3(blockDim_x, threads_per_block / blockDim_x), with
        threads_per_block = 256 and blockDim_x = min(ncols, 256).
        gridDim.y is already manually capped at 65535. gridDim.x =
        ceil(ncols / blockDim.x) is uncapped pre-fix. Total threads ~=
        gridDim.x * gridDim.y * threads_per_block. For ncols >= ~65536
        with nrows large enough to saturate gridDim.y at 65535, total
        threads can exceed HIP's 2**32 cap and trip
        `KernelLauncher::checkThreadCountNotExceeded`.

        The kernel grid-strides on both dims (line 87 over row, line 90
        over col), so the cap is correctness-preserving.

        Note: at nrows=1024, ncols=65792 the input, output, and internal
        shared_exponents tensors are each nrows*ncols*4 bytes (~257 MiB),
        for a peak of ~0.8 GiB (well within the gpu_memory_lt_gb(2)
        guard). This scale does not itself trip the 2**32 thread limit --
        that requires nrows and ncols both near their caps (tens of GiB) --
        it exercises the gridDim.x cap / grid-stride code path, which is
        hygienic for grid-striding kernels. FloatToMSFPQuantized is a
        CUDA-only op (no CPU/Meta dispatch), so this is a launch-success
        regression test with no CPU oracle.
        """
        nrows = 1024
        ncols = 65792  # > 256 * 256 + 256: forces gridDim.x >= 257
        device = torch.device(torch.accelerator.current_accelerator() or "cuda")
        input_data = torch.zeros((nrows, ncols), dtype=torch.float32, device=device)
        ebits = 8
        mbits = 7
        bias = 127
        max_pos = (1 << ((1 << ebits) - 2 - bias)) * (2 - 2 ** (-mbits))
        min_pos = 2 ** (1 - bias - mbits)
        bounding_box_size = 16
        output = torch.ops.fbgemm.FloatToMSFPQuantized(
            input_data,
            bounding_box_size,
            ebits,
            mbits,
            bias,
            min_pos,
            max_pos,
        )
        # If we reach this point, the cap path is in place and the
        # kernel launched successfully.
        self.assertEqual(output.shape, input_data.shape)


if __name__ == "__main__":
    unittest.main()
