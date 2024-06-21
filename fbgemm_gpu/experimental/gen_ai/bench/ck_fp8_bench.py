# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
from typing import Any, Tuple

import fbgemm_gpu.experimental.gen_ai  # noqa: F401

import pandas as pd

import torch
import triton  # @manual=//triton:triton
from fbgemm_gpu.experimental.gemm.triton_gemm.fp8_gemm import (
    quantize_fp8_block,
    quantize_fp8_row,
)

E4M3_MAX_POS: float = torch.finfo(torch.float8_e4m3fnuz).max
EPS = 1e-12


def set_amd_env_vars() -> None:
    print("Setting environment variables for AMD GPU performance")
    os.environ["DISABLE_ADDMM_HIP_LT"] = "0"
    os.environ["HIP_FORCE_DEV_KERNARG"] = "1"
    os.environ["PYTORCH_TUNABLEOP_VERBOSE"] = "0"
    os.environ["PYTORCH_TUNABLEOP_ENABLED"] = "1"
    os.environ["PYTORCH_TUNABLEOP_TUNING"] = "1"
    os.environ["PYTORCH_TUNABLEOP_FILENAME"] = "hipblas_tuning_pt_llama.csv"
    os.environ["PYTORCH_TUNABLEOP_MAX_TUNING_DURATION_MS"] = "30"
    os.environ["PYTORCH_TUNABLEOP_MAX_WARMUP_DURATION_MS"] = "30"


def fp8_quantize(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    f8_max = torch.tensor(E4M3_MAX_POS, device=x.device)
    x_amax = torch.max(torch.abs(x))
    scale = f8_max / torch.clamp(x_amax, min=EPS)
    scaled_x = torch.clamp(x * scale, min=-1 * f8_max, max=f8_max)
    return scaled_x.to(torch.float8_e4m3fnuz), scale.reciprocal().to(torch.float32)


class FPMatMul(torch.nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.matmul(a, b)


class BaselineMatmul(torch.nn.Module):
    def forward(
        self,
        qa: torch.Tensor,
        qb: torch.Tensor,
        a_scale: torch.Tensor,
        b_scale: torch.Tensor,
    ) -> torch.Tensor:
        output, _ = torch._scaled_mm(
            qa,
            qb,
            bias=None,
            out_dtype=torch.bfloat16,
            scale_a=a_scale,
            scale_b=b_scale,
            scale_result=None,
            use_fast_accum=True,
        )
        return output


class CKTensorMatmul(torch.nn.Module):
    def forward(
        self, a: torch.Tensor, b: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        return torch.ops.fbgemm.f8f8bf16_tensorwise(a, b, scale)


class CKRowMatmul(torch.nn.Module):
    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        a_scale: torch.Tensor,
        b_scale: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ops.fbgemm.f8f8bf16_rowwise(a, b, a_scale, b_scale)


class CKBlockMatmul(torch.nn.Module):
    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        a_scale: torch.Tensor,
        b_scale: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ops.fbgemm.f8f8bf16_blockwise(a, b, a_scale, b_scale)


@torch.no_grad()
def evaluate_impl(M: int, N: int, K: int) -> Tuple[Any, ...]:
    print(f"Evaluating {M=}, {N=}, {K=}")
    A = torch.randn(M, K).to(dtype=torch.bfloat16, device="cuda")
    QA, a_scale = fp8_quantize(A)
    B = torch.randn(N, K).to(dtype=torch.bfloat16, device="cuda")
    QB, b_scale = fp8_quantize(B)
    QA_row, a_scale_row = quantize_fp8_row(A)
    QB_row, b_scale_row = quantize_fp8_row(B)
    QA_block, a_scale_block = quantize_fp8_block(A)
    QB_block, b_scale_block = quantize_fp8_block(B)

    # Create modules for benchmarking.
    fp_func = FPMatMul()
    baseline_func = BaselineMatmul()
    ck_tensor_func = CKTensorMatmul()
    ck_row_func = CKRowMatmul()
    ck_block_func = CKBlockMatmul()

    # Check accuracy.
    out_ref = fp_func(A.to(torch.float32), B.t().to(torch.float32))

    baseline_out = baseline_func(QA, QB.t(), a_scale, b_scale)
    baseline_sim = torch.mean(torch.pow(torch.abs(baseline_out - out_ref), 2))
    print(f"Baseline accuracy: {baseline_sim}")

    ck_tensor_out = ck_tensor_func(QA, QB, a_scale * b_scale)
    ck_tensor_sim = torch.mean(torch.pow(torch.abs(ck_tensor_out - out_ref), 2))
    print(f"CK tensorwise accuracy: {ck_tensor_sim}")

    ck_row_out = ck_row_func(QA_row, QB_row, a_scale_row, b_scale_row)
    ck_row_sim = torch.mean(torch.pow(torch.abs(ck_row_out - out_ref), 2))
    print(f"CK rowwise accuracy: {ck_row_sim}")

    ck_block_out = ck_block_func(QA_block, QB_block, a_scale_block, b_scale_block)
    ck_block_sim = torch.mean(torch.pow(torch.abs(ck_block_out - out_ref), 2))
    print(f"CK blockwise accuracy: {ck_block_sim}")

    # Benchmark runtimes.
    ms_ref: float = triton.testing.do_bench(lambda: fp_func(A, B.t()))
    print(f"BF16 runtime: {ms_ref} ms")

    ms_baseline: float = triton.testing.do_bench(
        lambda: baseline_func(QA, QB.t(), a_scale, b_scale)
    )
    print(f"Baseline runtime: {ms_baseline} ms")

    ms_tensor_ck: float = triton.testing.do_bench(
        lambda: ck_tensor_func(QA, QB, a_scale * b_scale)
    )
    print(f"CK tensorwise runtime: {ms_tensor_ck} ms")

    ms_row_ck: float = triton.testing.do_bench(
        lambda: ck_row_func(QA_row, QB_row, a_scale_row, b_scale_row)
    )
    print(f"CK rowwise runtime: {ms_row_ck} ms")

    ms_block_ck: float = triton.testing.do_bench(
        lambda: ck_block_func(QA_block, QB_block, a_scale_block, b_scale_block)
    )
    print(f"CK blockwise runtime: {ms_block_ck} ms")

    return (
        float(baseline_sim.item()),
        float(ck_tensor_sim.item()),
        float(ck_row_sim.item()),
        ms_baseline,
        ms_tensor_ck,
        ms_row_ck,
        ms_ref,
    )


def main(args: Any) -> None:
    if args.enable_amd_env_vars:
        set_amd_env_vars()

    with torch.no_grad():
        # Create a list of results.
        benchmark_results = []

        # Test over a bunch of shapes.
        M = [1024, 2048, 4096]  # [1, 4, 8, 16, 32, 64, 128, 2048, 4096]
        N = [1280, 2304, 7168, 8192]
        K = [1024, 3584, 8192]

        for m in M:
            for n in N:
                for k in K:
                    (
                        baseline_sim,
                        ck_tensor_sim,
                        ck_row_sim,
                        ms_baseline,
                        ms_tensor_ck,
                        ms_row_ck,
                        ms_bf16,
                    ) = evaluate_impl(m, n, k)
                    benchmark_results.append(
                        {
                            "M": m,
                            "N": n,
                            "K": k,
                            "baseline_sim": baseline_sim,
                            "ck_tensor_sim": ck_tensor_sim,
                            "ck_row_sim": ck_row_sim,
                            "ms_baseline": ms_baseline,
                            "ms_tensor_ck": ms_tensor_ck,
                            "ms_row_ck": ms_row_ck,
                            "ms_bf16": ms_bf16,
                        }
                    )
        if args.export_csv:
            benchmark_results_df = pd.DataFrame(benchmark_results)
            benchmark_results_df.to_csv("ck_bench.csv", index=False)


def invoke_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--export_csv",
        action="store_true",
        help="Export results to a CSV file.",
    )
    parser.add_argument(
        "--enable_amd_env_vars",
        default=False,
        action="store_true",
        help="Enable a set of environment variables for AMD GPU performance",
    )

    args = parser.parse_args()
    main(args)
