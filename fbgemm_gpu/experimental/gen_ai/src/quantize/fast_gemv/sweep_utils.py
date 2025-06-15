# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import fbgemm_gpu.experimental.gen_ai  # noqa: F401
import torch
from torch._inductor.utils import do_bench_using_profiling


class SweepHeuristics:
    def __init__(self):
        self.ms = [1, 2, 3, 4]
        self.block_dims = [
            (32, 1),
            (32, 4),
            (32, 8),
            (32, 16),
            (32, 32),
            (64, 1),
            (64, 2),
            (64, 4),
            (64, 8),
            (64, 16),
            (128, 1),
            (128, 2),
            (128, 4),
            (128, 8),
            (256, 1),
            (256, 2),
            (256, 4),
            (512, 1),
            (512, 2),
            (1024, 1),
        ]
        self.block_dim_xs = [
            (32, 1),
            (64, 1),
            (128, 1),
            (256, 1),
            (512, 1),
            (1024, 1),
        ]
        self.nks = [(1280, 8192), (8192, 1024), (7168, 8192), (8192, 3584)]
        # 17bx128e dense model shapes grabbed from here:
        # https://www.internalfb.com/code/fbsource/[26a75e239633]/fbcode/accelerators/workloads/microbench/bench_gemm.py?lines=269
        self.nks_l4 = [(4096, 5120), (5120, 2048), (896, 5120), (5120, 640)]

    def sweep_heuristics(
        self,
        fn,
        quantize_w: bool = False,
        quantize_x: bool = False,
        llama_4: bool = False,
    ) -> None:
        """Sweep heuristics for a given kernel"""
        block_dims = self.block_dims
        if llama_4:
            nks = self.nks_l4
            block_dims = self.block_dim_xs
        else:
            nks = self.nks
            if quantize_w and quantize_x:
                block_dims = self.block_dim_xs
        for m in self.ms:
            for n, k in nks:
                x = torch.randn(size=(m, k), dtype=torch.bfloat16, device="cuda")
                w = torch.randn(size=(n, k), dtype=torch.bfloat16, device="cuda")

                best_elapsed_time, best_block_dim_x, best_block_dim_y = None, None, None

                for block_dim_x, block_dim_y in block_dims:
                    if (k % block_dim_x != 0) or (n % block_dim_x != 0):
                        continue
                    res = 0.0
                    # Currently this requires manual changes to the kernel code as listed below:
                    # 1. update for "testing purpose" the pytorch custom gemv op to accept additional params block_dim_x and block_dim_y
                    # 2. modify the corresponding `{precision}_fast_gemv.cu` kernel signature to reflect the block_dim_x and block_dim_y heuristics
                    # e.g. https://www.internalfb.com/code/fbsource/[208a27f25373]/fbcode/deeplearning/fbgemm/fbgemm_gpu/experimental/gen_ai/bench/quantize_ops.py?lines=375
                    if quantize_w and quantize_x:
                        xq, x_scale = torch.ops.fbgemm.quantize_fp8_per_tensor(x)
                        wq, w_scale = torch.ops.fbgemm.quantize_fp8_per_tensor(w)
                        res = do_bench_using_profiling(
                            lambda func=fn, x=xq, w=wq, x_scale=x_scale, w_scale=w_scale, block_dim_x=block_dim_x: func(
                                x, w, x_scale, w_scale, block_dim_x
                            )
                        )
                    elif quantize_w:
                        wq, w_scale = torch.ops.fbgemm.quantize_fp8_per_tensor(w)
                        res = do_bench_using_profiling(
                            lambda func=fn, x=x, w=wq, scale=w_scale, block_dim_x=block_dim_x, block_dim_y=block_dim_y: func(
                                x, w, scale, block_dim_x, block_dim_y
                            )
                        )
                    else:
                        res = do_bench_using_profiling(
                            lambda func=fn, x=x, w=w, block_dim_x=block_dim_x, block_dim_y=block_dim_y: func(
                                x, w, block_dim_x, block_dim_y
                            )
                        )

                    if best_elapsed_time is None or res < best_elapsed_time:
                        best_elapsed_time, best_block_dim_x, best_block_dim_y = (
                            res,
                            block_dim_x,
                            block_dim_y,
                        )
                if best_elapsed_time is None:
                    print("Error: No valid elapsed time found. Exiting the function.")
                    return
                if fn == torch.ops.fbgemm.bf16_fast_gemv:
                    bw = (
                        (m * k * 2 + n * k * 2 + m * n * 2)
                        / (best_elapsed_time / 1000)
                        / (1024**3)
                    )
                elif fn == torch.ops.fbgemm.bf16fp8bf16_fast_gemv:
                    bw = (
                        (m * k * 2 + n * k + m * n * 2)
                        / (best_elapsed_time / 1000)
                        / (1024**3)
                    )
                else:  # Assuming fn is torch.ops.fbgemm.fp8fp8bf16_fast_gemv
                    bw = (
                        (m * k + n * k + m * n * 2)
                        / (best_elapsed_time / 1000)
                        / (1024**3)
                    )
                print(f"m: {m}, n: {n}, k: {k}")
                print(f"tuning heuristics for kernel: {fn.__name__}")
                print(f"best elapsed time: {best_elapsed_time} ms")
                print(f"best block_dim_x: {best_block_dim_x}")
                print(f"best block_dim_y: {best_block_dim_y}")
                print(f"best bw: {bw} GB/s")


sweep_instance = SweepHeuristics()
sweep_instance.sweep_heuristics(fn=torch.ops.fbgemm.bf16_fast_gemv)
sweep_instance.sweep_heuristics(
    fn=torch.ops.fbgemm.bf16fp8bf16_fast_gemv, quantize_w=True
)
sweep_instance.sweep_heuristics(
    fn=torch.ops.fbgemm.fp8fp8bf16_fast_gemv, quantize_w=True, quantize_x=True
)

# 17bx128e dense l4 model shapes
sweep_instance.sweep_heuristics(
    fn=torch.ops.fbgemm.fp8fp8bf16_fast_gemv,
    quantize_w=True,
    quantize_x=True,
    llama_4=True,
)
