# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[56]

import logging
import unittest

import fbgemm_gpu.experimental.gen_ai  # noqa: F401

import torch
import triton  # noqa: F401

logger: logging.Logger = logging.getLogger()
logger.setLevel(logging.INFO)


@unittest.skipIf(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0),
    "Skip when no Hopper GPU is available. This test is only for Hopper GPU.",
)
class GatherScatterTests(unittest.TestCase):
    """Test Gathers."""

    def test_gather_along_first_dim(self) -> None:
        def _test_gather_along_first_dim(
            M: int, N: int, K: int, compile: bool = False
        ) -> None:
            logger.info(f"Running test_gather_along_first_dim: {M=}, {N=}, {K=}")
            src = torch.randn([M, K], device="cuda", dtype=torch.bfloat16).abs()
            if M == N:
                indices = torch.randperm(N, device="cuda", dtype=torch.int32)
            else:
                indices = torch.randint(0, M, [N], device="cuda", dtype=torch.int32)

            def fn():
                op = torch.ops.fbgemm.gather_along_first_dim
                if compile:
                    op = torch.compile(op, backend="inductor", fullgraph=True)
                return op(src, indices)

            def ref_fn():
                return torch.index_select(src, 0, indices)

            logger.info("Running FBGMM")
            dst = fn()
            logger.info("Running PyTorch")
            ref_dst = ref_fn()

            self.assertTrue((dst == ref_dst).all().item())

            # Load src, store dst. x2.
            data_size_in_terabytes = N * K * 2 * 2 / 1e12

            time_in_us = triton.testing.do_bench(fn) * 1e3
            time_in_second = time_in_us / 1e6
            terabytes_per_second = data_size_in_terabytes / time_in_second

            ref_time_in_us = triton.testing.do_bench(ref_fn) * 1e3
            ref_time_in_second = ref_time_in_us / 1e6
            ref_terabytes_per_second = data_size_in_terabytes / ref_time_in_second

            logger.info(
                f"FBGEMM time: {time_in_us:.2f} us. Bandwidth: {terabytes_per_second:.2f} TB/s"
            )
            logger.info(
                f"PyTorch time: {ref_time_in_us:.2f} us. Bandwidth: {ref_terabytes_per_second:.2f} TB/s"
            )

        _test_gather_along_first_dim(127, 257, 1023)
        _test_gather_along_first_dim(127, 257, 1024)
        _test_gather_along_first_dim(255, 129, 2049)
        _test_gather_along_first_dim(255, 129, 2048)
        _test_gather_along_first_dim(1024, 1024, 1024)
        _test_gather_along_first_dim(1024, 1024, 1024, compile=True)

        _test_gather_along_first_dim(1, 1, 5120)
        _test_gather_along_first_dim(128, 128, 5120)
        _test_gather_along_first_dim(2048, 2048, 5120)
        _test_gather_along_first_dim(4096, 4096, 5120)
        _test_gather_along_first_dim(8192, 8192, 5120)

    def test_scatter_add_along_first_dim(self) -> None:
        def _test_scatter_add_along_first_dim(
            M: int, N: int, K: int, compile: bool = False
        ) -> None:
            logger.info(f"Running test_scatter_add_along_first_dim: {M=}, {N=}, {K=}")
            src = torch.randn([M, K], device="cuda", dtype=torch.bfloat16).abs()
            dst = torch.randn([N, K], device="cuda", dtype=torch.bfloat16).abs()
            if M == N:
                indices_1d = torch.randperm(N, device="cuda", dtype=torch.int64)
            else:
                indices_1d = torch.randint(0, N, [M], device="cuda", dtype=torch.int64)

            indices_2d = indices_1d.to(torch.int64).unsqueeze(1).expand(-1, K)

            test_dst = dst.clone()
            ref_dst = dst.clone()

            logger.info("Running FBGMM")
            torch.ops.fbgemm.scatter_add_along_first_dim(test_dst, src, indices_1d)

            logger.info("Running PyTorch")
            ref_dst.scatter_add_(0, indices_2d, src)

            torch.testing.assert_close(test_dst, ref_dst, atol=1e-3, rtol=2.1e-2)

            def fn():
                op = torch.ops.fbgemm.scatter_add_along_first_dim
                if compile:
                    op = torch.compile(op, backend="inductor", fullgraph=True)
                op(test_dst, src, indices_1d)

            def ref_fn():
                ref_dst.scatter_add_(0, indices_2d, src)

            # Load src, load dst, store dst. x3.
            data_size_in_terabytes = N * K * 2 * 3 / 1e12

            time_in_us = triton.testing.do_bench(fn) * 1e3
            time_in_second = time_in_us / 1e6
            terabytes_per_second = data_size_in_terabytes / time_in_second

            ref_time_in_us = triton.testing.do_bench(ref_fn) * 1e3
            ref_time_in_second = ref_time_in_us / 1e6
            ref_terabytes_per_second = data_size_in_terabytes / ref_time_in_second

            logger.info(
                f"FBGEMM time: {time_in_us:.2f} us. Bandwidth: {terabytes_per_second:.2f} TB/s"
            )
            logger.info(
                f"PyTorch time: {ref_time_in_us:.2f} us. Bandwidth: {ref_terabytes_per_second:.2f} TB/s"
            )

        _test_scatter_add_along_first_dim(127, 257, 1023)
        _test_scatter_add_along_first_dim(127, 257, 1024)
        _test_scatter_add_along_first_dim(255, 129, 2049)
        _test_scatter_add_along_first_dim(255, 129, 2048)
        _test_scatter_add_along_first_dim(1024, 1024, 1024)
        _test_scatter_add_along_first_dim(1024, 1024, 1024, compile=True)

        _test_scatter_add_along_first_dim(1, 1, 5120)
        _test_scatter_add_along_first_dim(128, 128, 5120)
        _test_scatter_add_along_first_dim(2048, 2048, 5120)
        _test_scatter_add_along_first_dim(4096, 4096, 5120)
        _test_scatter_add_along_first_dim(8192, 8192, 5120)


if __name__ == "__main__":
    unittest.main()
