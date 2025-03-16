# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import torch

# Load the custom op library
torch.ops.load_library(
    "//deeplearning/fbgemm/fbgemm_gpu/experimental/gen_ai/src/quantize/ck_extensions:ck_fused_moe"
)


def main():
    # Set dimensions
    tokens = 128
    hidden_size = 8192
    experts = 32
    intermediate_size = 8192
    topk = 5

    print("Running fused MoE kernel...")

    # Create input tensors on GPU
    input = torch.randn(tokens, hidden_size, dtype=torch.bfloat16, device="cuda")

    gate_up_weight = torch.randn(
        experts, intermediate_size, hidden_size, dtype=torch.bfloat16, device="cuda"
    )

    down_weight = torch.randn(
        experts, hidden_size, intermediate_size, dtype=torch.bfloat16, device="cuda"
    )

    topk_ids = torch.randint(
        0, experts, (tokens, topk), dtype=torch.int32, device="cuda"
    )

    topk_weights = torch.randn(tokens, topk, dtype=torch.float32, device="cuda")

    # Optional scale tensors for quantization
    input_scales = torch.randn(tokens, dtype=torch.float32, device="cuda")

    gate_up_scales = torch.randn(intermediate_size, dtype=torch.float32, device="cuda")

    down_scales = torch.randn(intermediate_size, dtype=torch.float32, device="cuda")

    smooth_scales = torch.randn(intermediate_size, dtype=torch.float32, device="cuda")

    # Call the fused MoE operation
    output = torch.ops.fbgemm.fused_moe(
        input=input,
        gate_up_weight=gate_up_weight,
        down_weight=down_weight,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        input_scales=input_scales,
        gate_up_scales=gate_up_scales,
        down_scales=down_scales,
        smooth_scales=smooth_scales,
        block_m=32,
        gate_only=True,
        fused_quant=1,
    )

    print(f"Output shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")


if __name__ == "__main__":
    main()
