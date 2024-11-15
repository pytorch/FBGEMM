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
#include <cutlass/gemm/collective/collective_builder.hpp>     // @manual
#include <cutlass/gemm/device/gemm_universal_adapter.h>       // @manual
#include <cutlass/epilogue/collective/collective_builder.hpp> // @manual
// clang-format on

namespace fbgemm_gpu {

constexpr int kNumSMsForH100 = 132;

inline int ceildiv(int a, int b) {
  return (a + b - 1) / b;
}

inline int round_up_to_nearest_multiple(int a, int b) {
  return ceildiv(a, b) * b;
}

#define FOR_FLOAT_TYPES(INSTANTIATE_F)                                    \
  INSTANTIATE_F(cutlass::float_e5m2_t, true, true, cutlass::bfloat16_t);  \
  INSTANTIATE_F(cutlass::float_e4m3_t, true, true, cutlass::bfloat16_t);  \
  INSTANTIATE_F(cutlass::float_e5m2_t, false, true, cutlass::bfloat16_t); \
  INSTANTIATE_F(cutlass::float_e4m3_t, false, true, cutlass::bfloat16_t); \
  INSTANTIATE_F(cutlass::float_e5m2_t, true, true, float);                \
  INSTANTIATE_F(cutlass::float_e4m3_t, true, true, float);                \
  INSTANTIATE_F(cutlass::float_e5m2_t, false, true, float);               \
  INSTANTIATE_F(cutlass::float_e4m3_t, false, true, float);               \
  INSTANTIATE_F(cutlass::float_e5m2_t, true, false, float);               \
  INSTANTIATE_F(cutlass::float_e4m3_t, true, false, float);               \
  INSTANTIATE_F(cutlass::float_e5m2_t, false, false, float);              \
  INSTANTIATE_F(cutlass::float_e4m3_t, false, false, float);

} // namespace fbgemm_gpu
