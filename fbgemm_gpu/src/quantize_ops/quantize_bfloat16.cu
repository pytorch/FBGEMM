/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "common.cuh"

using Tensor = at::Tensor;

namespace fbgemm_gpu {

/// @ingroup quantize-ops-cuda
///
/// Converts a tensor of `float` values into a tensor of Brain Floating Point
/// (`bfloat16`) values.
///
/// @param input A tensor of `float` values
///
/// @return A new tensor with values from the input tensor converted to
/// `bfloat16`.
DLL_PUBLIC at::Tensor _float_to_bfloat16_gpu(const at::Tensor& input) {
  CUDA_DEVICE_GUARD(input);

  // TODO: replace Half by BFloat16, after BFloat16 is supported by Nvidia
  // NCCL input.options().dtype(at::kBFloat16)); // at::kBFloat16
  auto output = at::empty({}, input.options().dtype(at::kHalf));
  output.resize_(0);

  auto iter = at::TensorIteratorConfig()
                  .check_all_same_dtype(false)
                  .add_output(output)
                  .add_input(input)
                  .build();
  at::native::gpu_kernel(iter, [] GPU_LAMBDA(float in) -> at::Half {
    fbgemm_gpu::fint32 temp;
    temp.F = in;
    return at::Half((temp.I + (1 << 15)) >> 16, at::Half::from_bits());
  });

  return output;
}

/// @ingroup quantize-ops-cuda
///
/// Converts a tensor of Brain Floating Point (`bfloat16`) values into a tensor
/// of `float` values.
///
/// @param input A tensor of `bfloat16` values
///
/// @return A new tensor with values from the input tensor converted to `float`.
DLL_PUBLIC at::Tensor _bfloat16_to_float_gpu(const at::Tensor& input) {
  CUDA_DEVICE_GUARD(input);

  auto output = at::empty({}, input.options().dtype(at::kFloat));
  output.resize_(0);
  auto iter = at::TensorIteratorConfig()
                  .check_all_same_dtype(false)
                  .add_output(output)
                  .add_input(input)
                  .build();

  at::native::gpu_kernel(iter, [] GPU_LAMBDA(at::Half in) -> float {
    fbgemm_gpu::fint32 temp;
    temp.I = in.x << 16;
    return temp.F;
  });
  return output;
}

} // namespace fbgemm_gpu

FBGEMM_OP_DISPATCH(
    CUDA,
    "Bfloat16QuantizedToFloat",
    fbgemm_gpu::_bfloat16_to_float_gpu);
FBGEMM_OP_DISPATCH(
    CUDA,
    "FloatToBfloat16Quantized",
    fbgemm_gpu::_float_to_bfloat16_gpu);
