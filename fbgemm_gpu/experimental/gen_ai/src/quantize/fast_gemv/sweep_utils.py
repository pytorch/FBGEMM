# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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
        self.nks = [(1280, 8192), (8192, 1024), (7168, 8192), (8192, 3584)]

    def sweep_heuristics(self, fn, quantize_w=False, quantize_x=False) -> None:
        for m in self.ms:
            for n, k in self.nks:
                x = torch.randn(size=(m, k), dtype=torch.bfloat16, device="cuda")
                w = torch.randn(size=(n, k), dtype=torch.bfloat16, device="cuda")

                best_elapsed_time, best_block_dim_x, best_block_dim_y = None, None, None

                for block_dim_x, block_dim_y in self.block_dims:
                    if (
                        (k % block_dim_x != 0)
                        or (n % block_dim_x != 0)
                        or ((k / block_dim_x) % 8 != 0)
                    ):
                        continue

                    res = 0.0
                    if quantize_w and quantize_x:
                        xq, x_scale = torch.ops.fbgemm.quantize_fp8_per_tensor(x)
                        wq, w_scale = torch.ops.fbgemm.quantize_fp8_per_tensor(w)
                        res = do_bench_using_profiling(
                            lambda func=fn, x=xq, w=wq, scale=x_scale * w_scale, block_dim_x=block_dim_x, block_dim_y=block_dim_y: func(
                                x, w, scale, block_dim_x, block_dim_y
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
                        # This requires
                        # 1. update for "testing purpose" the pytorch custom gemv op to accept additional params block_dim_x and block_dim_y
                        # 2. modify the corresponding `{precision}_fast_gemv.cu` kernel signature to reflect the block_dim_x and block_dim_y heuristics
                        # e.g. https://www.internalfb.com/code/fbsource/[208a27f25373]/fbcode/deeplearning/fbgemm/fbgemm_gpu/experimental/gen_ai/bench/quantize_ops.py?lines=375
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
