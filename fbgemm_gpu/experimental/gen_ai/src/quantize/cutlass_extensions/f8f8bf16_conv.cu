/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/packed_stride.hpp>

// clang-format off
// The fixed ordering of the headers is required for CUTLASS 3.2+
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/conv/collective/collective_builder.hpp>
#include <cutlass/conv/convnd_problem_shape.hpp>
#include <cutlass/conv/convolution.h>
#include <cutlass/conv/device/conv_universal_adapter.hpp>
#include <cutlass/conv/dispatch_policy.hpp>
#include <cutlass/conv/kernel/conv_universal.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
// clang-format on

#include "f8f8bf16_conv/f8f8bf16_conv_manifest.cuh"

namespace fbgemm_gpu {

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

struct ProblemSize {
  std::vector<int64_t> activation_shape; // [N, D, H, W, C]
  std::vector<int64_t> filter_shape; // [K, T, R, S, C]
  std::vector<int64_t> padding;
  std::vector<int64_t> stride;
  std::vector<int64_t> dilation;
  bool operator==(const ProblemSize& ps) const {
    return activation_shape == ps.activation_shape &&
        filter_shape == ps.filter_shape && padding == ps.padding &&
        stride == ps.stride && dilation == ps.dilation;
  }
};

inline void hash_combine(std::size_t& seed, std::size_t value) {
  seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

// Hash function for ProblemSize for use in unordered_map
struct ProblemSizeHash {
  std::size_t operator()(const ProblemSize& ps) const {
    std::size_t seed = 0;
    auto vec_hash = [](const std::vector<int64_t>& v) {
      std::size_t h = 0;
      for (auto x : v)
        hash_combine(h, std::hash<int64_t>{}(x));
      return h;
    };
    hash_combine(seed, vec_hash(ps.activation_shape));
    hash_combine(seed, vec_hash(ps.filter_shape));
    hash_combine(seed, vec_hash(ps.padding));
    hash_combine(seed, vec_hash(ps.stride));
    hash_combine(seed, vec_hash(ps.dilation));
    return seed;
  }
};

// clang-format off
std::unordered_map<ProblemSize, Kernel_f8f8bf16_conv, ProblemSizeHash> kernel_map = {
    {{{1,6,32,48,48}, {48,1,1,1,48}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_128x128x128_1x1x1},
    {{{1,3,34,50,48}, {1024,3,3,3,48}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_256x256x128_4x1x1},
    {{{1,3,34,50,1024}, {1024,3,3,3,1024}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_128x256x128_2x1x1},
    {{{1,3,66,98,1024}, {1024,3,3,3,1024}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_256x256x128_4x1x1},
    {{{1,3,130,194,1024}, {512,3,3,3,1024}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_512x256x128_4x1x1},
    {{{1,3,130,194,512}, {512,3,3,3,512}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_256x512x128_2x2x1},
    {{{1,1,128,192,1024}, {512,1,1,1,1024}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_128x256x128_2x1x1},
    {{{1,3,258,386,512}, {256,3,3,3,512}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_512x256x128_4x1x1},
    {{{1,3,258,386,256}, {256,3,3,3,256}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_256x256x128_2x1x1},
    {{{1,1,256,384,512}, {256,1,1,1,512}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_256x256x128_2x1x1},
  //{{{1,3,258,386,256}, {12,3,3,3,256}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_invalid},
    {{{1,3,32,48,1024}, {2048,3,1,1,1024}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_128x256x128_2x2x1},
    {{{1,4,66,98,1024}, {1024,3,3,3,1024}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_256x512x128_2x2x1},
    {{{1,4,64,96,1024}, {2048,3,1,1,1024}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_256x256x128_2x1x1},
    {{{1,6,130,194,1024}, {512,3,3,3,1024}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_256x256x128_2x1x1},
    {{{1,6,130,194,512}, {512,3,3,3,512}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_256x256x128_2x1x1},
    {{{1,4,128,192,1024}, {512,1,1,1,1024}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_128x256x128_2x1x1},
    {{{1,6,258,386,512}, {256,3,3,3,512}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_256x256x128_2x1x1},
    {{{1,6,258,386,256}, {256,3,3,3,256}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_256x256x128_2x1x1},
    {{{1,4,256,384,512}, {256,1,1,1,512}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_512x256x128_4x1x1},
  //{{{1,6,258,386,256}, {12,3,3,3,256}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_invalid},
    {{{1,1,64,96,1024}, {1024,1,3,3,1024}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_256x256x128_2x1x1},
    {{{1,1,128,192,1024}, {1024,1,3,3,1024}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_256x256x128_2x1x1},
    {{{1,1,256,384,512}, {512,1,3,3,512}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_256x256x128_2x1x1},
    {{{2,1,64,96,1024}, {1024,1,3,3,1024}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_256x256x128_2x1x1},
    {{{4,1,128,192,1024}, {1024,1,3,3,1024}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_256x256x128_2x1x1},
    {{{4,1,256,384,512}, {512,1,3,3,512}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}}, f8f8bf16_conv_256x256x128_2x1x1}
};
// clang-format on

Kernel_f8f8bf16_conv get_kernel_via_heuristic(
    at::Tensor activation,
    at::Tensor filter,
    at::Tensor scale,
    std::vector<int64_t> padding,
    std::vector<int64_t> stride,
    std::vector<int64_t> dilation) {
  ProblemSize ps = {
      activation.sizes().vec(),
      filter.sizes().vec(),
      padding,
      stride,
      dilation};
  auto it = kernel_map.find(ps);
  if (it != kernel_map.end()) {
    return it->second;
  }
  // Fallback kernel
  return f8f8bf16_conv_256x256x128_2x1x1;
}

at::Tensor f8f8bf16_conv(
    at::Tensor activation, // FP8 - NDHWC layout
    at::Tensor filter, // FP8 - KTRSC layout
    at::Tensor scale,
    std::vector<int64_t> padding, // [pad_d, pad_h, pad_w]
    std::vector<int64_t> stride, // [stride_d, stride_h, stride_w]
    std::vector<int64_t> dilation) { // [dilation_d, dilation_h, dilation_w]

  // Select kernel to run via heuristics or tuning.
  auto kernel = [&]() {
    return get_kernel_via_heuristic(
        activation, filter, scale, padding, stride, dilation);
  }();

  return kernel(activation, filter, scale, padding, stride, dilation);
}

#else

at::Tensor f8f8bf16_conv(
    at::Tensor activation,
    at::Tensor filter,
    at::Tensor scale,
    std::vector<int64_t> padding,
    std::vector<int64_t> stride,
    std::vector<int64_t> dilation) {
  throw std::runtime_error(
      "SM100 (Blackwell) architecture not supported. Requires CUTLASS 3.x with SM100 support.");
}

#endif

} // namespace fbgemm_gpu
