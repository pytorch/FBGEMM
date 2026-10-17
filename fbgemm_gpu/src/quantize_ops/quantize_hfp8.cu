/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "common.cuh"
#include "fbgemm/Types.h"

using Tensor = at::Tensor;

namespace fbgemm_gpu {

namespace {
template <typename Format>
struct Hfp8Parameters {
  static constexpr int exponent_bits = Format::exponent_bits;
  static constexpr int exponent_bias = Format::exponent_bias;
};

template <>
struct Hfp8Parameters<void> {
  int exponent_bits;
  int exponent_bias;
};

template <typename Format>
struct EncodeHfp8 : Format {
  double max_pos;

  C10_HOST_DEVICE uint8_t operator()(float input) const {
    return float_to_hfp8(
        input, this->exponent_bits, this->exponent_bias, max_pos);
  }
};

template <typename Format>
struct DecodeHfp8 : Format {
  C10_HOST_DEVICE float operator()(uint8_t input) const {
    return hfp8_to_float(input, this->exponent_bits, this->exponent_bias);
  }
};

template <typename Launch>
void dispatch_hfp8_format(int64_t ebits, int64_t exponent_bias, Launch launch) {
  if (ebits == 4 && exponent_bias == 7) {
    launch(Hfp8Parameters<fbgemm::hfp8_e4m3>{});
  } else if (ebits == 5 && exponent_bias == 7) {
    launch(Hfp8Parameters<fbgemm::hfp8<5, 7>>{});
  } else if (ebits == 5 && exponent_bias == 15) {
    launch(Hfp8Parameters<fbgemm::hfp8_e5m2>{});
  } else {
    launch(
        Hfp8Parameters<void>{
            static_cast<int>(ebits), static_cast<int>(exponent_bias)});
  }
}
} // namespace

/// @ingroup quantize-ops-cuda
/// Converts a tensor of `float` values into a tensor of Hybrid 8-bit Floating
/// Point (`hfp8`) values.
///
/// @param input A tensor of `float` values
/// @param ebits
/// @param exponent_bias
/// @param max_pos
///
/// @return A new tensor with values from the input tensor converted to `hfp8`.
///
/// @throw c10::Error if `ebits > 0` or `exponent_bias > 0`.
DLL_PUBLIC at::Tensor _float_to_hfp8_gpu(
    const at::Tensor& input,
    const int64_t ebits,
    const int64_t exponent_bias,
    const double max_pos) {
  TORCH_CHECK(ebits > 0);
  TORCH_CHECK(exponent_bias > 0);
  CUDA_DEVICE_GUARD(input);

  auto output = at::empty({}, input.options().dtype(at::kByte));
  output.resize_(0);

  auto iter = at::TensorIteratorConfig()
                  .check_all_same_dtype(false)
                  .add_output(output)
                  .add_input(input)
                  .build();

  dispatch_hfp8_format(ebits, exponent_bias, [&](auto format) {
    at::native::gpu_kernel(iter, EncodeHfp8<decltype(format)>{format, max_pos});
  });

  return output;
}

/// @ingroup quantize-ops-cuda
/// Converts a tensor of Hybrid 8-bit Floating Point (`hfp8`) values into a
/// tensor of `float` values.
///
/// @param input A tensor of `hfp8` values
/// @param ebits
/// @param exponent_bias
///
/// @return A new tensor with values from the input tensor converted to `float`.
///
/// @throw c10::Error if `ebits > 0` or `exponent_bias > 0`.
DLL_PUBLIC at::Tensor _hfp8_to_float_gpu(
    const at::Tensor& input,
    const int64_t ebits,
    const int64_t exponent_bias) {
  TORCH_CHECK(ebits > 0);
  TORCH_CHECK(exponent_bias > 0);
  CUDA_DEVICE_GUARD(input);

  auto output = at::empty({}, input.options().dtype(at::kFloat));
  output.resize_(0);

  auto iter = at::TensorIteratorConfig()
                  .check_all_same_dtype(false)
                  .add_output(output)
                  .add_input(input)
                  .build();

  dispatch_hfp8_format(ebits, exponent_bias, [&](auto format) {
    at::native::gpu_kernel(iter, DecodeHfp8<decltype(format)>{format});
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
