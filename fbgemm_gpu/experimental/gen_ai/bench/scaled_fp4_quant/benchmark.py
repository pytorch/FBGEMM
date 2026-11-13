# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run against the working SM100a baseline and the amended CUDA build."""

import argparse
import functools
import json
import statistics

import fbgemm_gpu.experimental.gen_ai  # noqa: F401
import torch
from fbgemm_gpu.experimental.gemm.triton_gemm.fp4_quantize import (
    triton_scale_nvfp4_quant,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    if not torch.version.cuda or torch.cuda.get_device_capability()[0] != 10:
        raise RuntimeError("This benchmark requires an SM100-family NVIDIA GPU")
    device = torch.accelerator.current_accelerator()
    torch.manual_seed(0)
    print(
        json.dumps(
            {
                "label": args.label,
                "gpu": torch.cuda.get_device_name(),
                "capability": torch.cuda.get_device_capability(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "method": "50 isolated CUDA profiler kernel durations after 20 warmups",
            }
        )
    )
    for dtype in (torch.float16, torch.bfloat16):
        for rows, cols in ((128, 128), (8192, 8192)):
            x = torch.randn((rows, cols), dtype=dtype, device=device)
            global_scale = 2688.0 / x.abs().amax().float()
            xq = torch.full((rows, cols // 2), 255, dtype=torch.uint8, device=device)
            scales = torch.full(
                (rows, cols // 64), -1, dtype=torch.int32, device=device
            )
            expected_q, expected_scales = triton_scale_nvfp4_quant(x, global_scale)
            torch.cuda.synchronize()

            quantize = functools.partial(
                torch.ops.fbgemm.scaled_fp4_quant, xq, x, scales, global_scale
            )

            quantize()
            torch.cuda.synchronize()
            torch.testing.assert_close(
                scales.view(torch.uint8).flatten(),
                expected_scales.view(torch.uint8).flatten(),
                rtol=0,
                atol=0,
            )
            agreement = (xq == expected_q.view(torch.uint8)).float().mean().item()
            if agreement < 0.99:
                raise AssertionError(f"Packed-byte agreement is only {agreement}")
            for _ in range(20):
                quantize()
            torch.cuda.synchronize()
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA]
            ) as prof:
                for _ in range(50):
                    quantize()
                    torch.cuda.synchronize()
            # Read kernel intervals, not events on the caller stream: the old
            # working baseline launches on a different pool stream.
            times = [
                event.time_range.elapsed_us()
                for event in prof.events()
                if event.device_type == torch.autograd.DeviceType.CUDA
                and "cvt_fp16_to_fp4" in event.name
            ]
            if len(times) != 50:
                raise AssertionError(
                    f"Expected 50 quantization kernels, got {len(times)}"
                )
            print(
                json.dumps(
                    {
                        "label": args.label,
                        "dtype": str(dtype),
                        "shape": [rows, cols],
                        "packed_byte_agreement": agreement,
                        "kernels": len(times),
                        "median_us": statistics.median(times),
                        "min_us": min(times),
                        "max_us": max(times),
                    }
                )
            )


if __name__ == "__main__":
    main()
