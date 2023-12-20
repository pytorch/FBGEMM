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
/// Converts a tensor of `float` values into a tensor of Hybrid 8-bit Floating
/// Point (`hfp8`) values.
DLL_PUBLIC at::Tensor _float_to_hfp8_gpu(
    const at::Tensor& input,
    const int64_t ebits,
    const int64_t exponent_bias,
    const double max_pos) {
  TORCH_CHECK(ebits > 0);
  TORCH_CHECK(exponent_bias > 0);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(input.get_device());

  auto output = at::empty({}, input.options().dtype(at::kByte));
  output.resize_(0);

  auto iter = at::TensorIteratorConfig()
                  .check_all_same_dtype(false)
                  .add_output(output)
                  .add_input(input)
                  .build();

  at::native::gpu_kernel(iter, [=] GPU_LAMBDA(float in) -> uint8_t {
    return float_to_hfp8(in, ebits, exponent_bias, max_pos);
  });

  return output;
}

/// @ingroup quantize-ops-cuda
/// Converts a tensor of Hybrid 8-bit Floating Point (`hfp8`) values into a
/// tensor of `float` values.
DLL_PUBLIC at::Tensor _hfp8_to_float_gpu(
    const at::Tensor& input,
    const int64_t ebits,
    const int64_t exponent_bias) {
  TORCH_CHECK(ebits > 0);
  TORCH_CHECK(exponent_bias > 0);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(input.get_device());

  auto output = at::empty({}, input.options().dtype(at::kFloat));
  output.resize_(0);

  auto iter = at::TensorIteratorConfig()
                  .check_all_same_dtype(false)
                  .add_output(output)
                  .add_input(input)
                  .build();

  at::native::gpu_kernel(iter, [=] GPU_LAMBDA(uint8_t in) -> float {
    return hfp8_to_float(in, ebits, exponent_bias);
  });

  return output;
}

} // namespace fbgemm_gpu

FBGEMM_OP_DISPATCH(
    CUDA,
    "FloatToHFP8Quantized",
    fbgemm_gpu::_float_to_hfp8_gpu);
FBGEMM_OP_DISPATCH(
    CUDA,
    "HFP8QuantizedToFloat",
    fbgemm_gpu::_hfp8_to_float_gpu);
