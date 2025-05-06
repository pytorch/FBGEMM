/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <cuda.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <torch/types.h> // @manual=//caffe2:torch-cpp-cpu

#include <cmath>

#include "fbgemm_gpu/utils/kernel_launcher.cuh"
#include "fbgemm_gpu/utils/weight_row.cuh"

namespace fbgemm_gpu::utils {

////////////////////////////////////////////////////////////////////////////////
// Example Optimizer State Structs
//
// NOTE: The size of a struct is generally padded by the compiler to fit the
// byte-alignment of the largest-sized member value for efficient memory access.
// This means that it is dependent on the ** order ** of the member field
// declarations in the struct definition.
//
// TLDR: !!! The optimizer state value must be set with the correct alignment
// with respect to the tensor row on the PyTorch side to align with the struct
// definition and order of field declarations on the C++ side !!!
////////////////////////////////////////////////////////////////////////////////

struct ExampleOptimizerState1 {
  float momentum1; // 4 byte value
  // 4 bytes padding
  double momentum2; // 8 byte value
  // 0 byte padding
  uint8_t momentum3; // 1 byte value
  // 0 byte padding
  uint8_t momentum4; // 1 byte value
  // 6 bytes padding

  // TOTAL = sizeof(ExampleOptimizerState1) = 24 bytes
};

struct ExampleOptimizerState2 {
  float momentum; // 4 byte value
  // 0 bytes padding

  // TOTAL = sizeof(ExampleOptimizerState2) = 4 bytes
};

constexpr auto kMomentum1 = 2.71f;
constexpr double kMomentum2 = 3.1415926535;
constexpr uint8_t kMomentum3 = 42;
constexpr uint8_t kMomentum4 = 88;

////////////////////////////////////////////////////////////////////////////////
// Test Kernels
////////////////////////////////////////////////////////////////////////////////

template <typename emb_t, typename cache_t>
__global__ void set_optimizer_state_kernel1(
    const pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> weights,
    pta::PackedTensorAccessor64<cache_t, 1, at::RestrictPtrTraits> cache,
    const size_t rows,
    const uint32_t D,
    const size_t nOptimizerSizeBytes) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < rows) {
    // Start with D_emb set to D + padding if needed
    auto D_emb = utils::pad4(D);

    if constexpr (std::is_same_v<emb_t, uint8_t>) {
      // Add to D_emb the size of the qparams
      D_emb += kINT8QparamsBytes;
    }

    // Add to D_emb the size of the optimizer state in terms of size of emb_t,
    // rounded to the nearest 4
    D_emb += utils::pad4(
        static_cast<int32_t>(ceil(nOptimizerSizeBytes / sizeof(emb_t))));

    const auto weight_row =
        WeightRow<emb_t, cache_t, at::acc_type<cache_t, true>>(
            weights.data() + idx * D_emb, cache.data() + idx * D_emb, D);

    auto* state =
        weight_row.template optimizer_state_ptr<ExampleOptimizerState1>();

    state->momentum1 = kMomentum1;
    state->momentum2 = kMomentum2;
    state->momentum3 = kMomentum3;
    state->momentum4 = kMomentum4;
  }
}

template <typename row_t, typename dst_t>
__global__ void set_optimizer_state_kernel2(
    pta::PackedTensorAccessor64<row_t, 1, at::RestrictPtrTraits> row,
    const size_t rows,
    const uint32_t D,
    const size_t nOptimizerSizeBytes) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < rows) {
    auto D_emb = utils::pad4(D);

    if constexpr (std::is_same_v<row_t, uint8_t>) {
      D_emb += kINT8QparamsBytes;
    }

    D_emb += utils::pad4(
        static_cast<int32_t>(ceil(nOptimizerSizeBytes / sizeof(row_t))));

    const auto weight_row =
        WeightRowAccessor<row_t, dst_t>(row.data() + idx * D_emb, D);

    auto* state =
        weight_row.template optimizer_state_ptr<ExampleOptimizerState2>();

    state->momentum = kMomentum1;
  }
}

////////////////////////////////////////////////////////////////////////////////
// Padding Tests
////////////////////////////////////////////////////////////////////////////////

TEST(Padding, pad4) {
  EXPECT_EQ(utils::pad4(0), 0);
  EXPECT_EQ(utils::pad4(3), 4);
  EXPECT_EQ(utils::pad4(4), 4);
  EXPECT_EQ(utils::pad4(5), 8);
  EXPECT_EQ(utils::pad4(6), 8);
  EXPECT_EQ(utils::pad4(7), 8);
  EXPECT_EQ(utils::pad4(8), 8);
  EXPECT_EQ(utils::pad4(-5), -4);
}

////////////////////////////////////////////////////////////////////////////////
// Weight Row Tests
////////////////////////////////////////////////////////////////////////////////

TEST(WeightRow, update_optimizer_state) {
  const auto D = 32u;
  const auto rows = 4;
  const auto cols = utils::pad4(D);
  const auto extra = pad4(static_cast<int32_t>(
      ceil(sizeof(ExampleOptimizerState1) / sizeof(float))));

  const auto fW = 21.0f;

  auto cache = torch::full(
                   {rows, cols + extra},
                   fW,
                   torch::dtype(torch::kFloat32)
                       .device(torch::kCUDA, at::cuda::current_device()))
                   .contiguous();

  // Clear the area in memory used to store the optimizer state
  cache.slice(1, cols, cols + extra).fill_(0);

  const auto weights = torch::zeros_like(cache);

  FBGEMM_LAUNCH_KERNEL(
      (set_optimizer_state_kernel1<float, float>),
      1,
      1024,
      0,
      at::cuda::getCurrentCUDAStream(),
      PTA_B(weights.flatten(), float, 1, 64),
      PTA_B(cache.flatten(), float, 1, 64),
      rows,
      D,
      sizeof(ExampleOptimizerState1));

  // The weights are not mutated
  EXPECT_TRUE(torch::all(cache.slice(1, 0, cols) == fW).item<bool>());

  // momentum1 is set correctly
  EXPECT_TRUE(
      torch::all(cache.slice(1, cols, cols + 1) == kMomentum1).item<bool>());

  // momentum2 is set correctly
  {
    // Define reference to hold the value and prevent ASAN false negatives
    auto momentum2 =
        cache.slice(1, cols + 2, cols + 4).cpu().contiguous().flatten();
    auto* doubles = reinterpret_cast<double*>(momentum2.data_ptr<float>());

    for (auto i = 0; i < rows; i++) {
      EXPECT_EQ(doubles[i], kMomentum2);
    }
  }

  // momentum3 and momentum4 are set correctly
  {
    // Define reference to hold the value and prevent ASAN false negatives
    auto momentum34 =
        cache.slice(1, cols + 4, cols + 5).cpu().contiguous().flatten();
    auto* uints = reinterpret_cast<uint8_t*>(momentum34.data_ptr<float>());

    for (auto i = 0; i < rows; i++) {
      EXPECT_EQ(uints[i * 4], kMomentum3);
      EXPECT_EQ(uints[i * 4 + 1], kMomentum4);
    }
  }
}

TEST(WeightRowAccessor, update_optimizer_state) {
  const auto D = 32u;
  const auto rows = 4;
  const auto cols = utils::pad4(D);
  const auto extra = pad4(static_cast<int32_t>(
      ceil(sizeof(ExampleOptimizerState2) / sizeof(float))));

  const auto fW = 13.7f;

  auto weights = torch::full(
                     {rows, cols + extra},
                     fW,
                     torch::dtype(torch::kFloat32)
                         .device(torch::kCUDA, at::cuda::current_device()))
                     .contiguous();

  // Clear the area in memory used to store the optimizer state
  weights.slice(1, cols, cols + extra).fill_(0);

  FBGEMM_LAUNCH_KERNEL(
      (set_optimizer_state_kernel2<float, float>),
      1,
      1024,
      0,
      at::cuda::getCurrentCUDAStream(),
      PTA_B(weights.flatten(), float, 1, 64),
      rows,
      D,
      sizeof(ExampleOptimizerState2));

  // The weights are not mutated
  EXPECT_TRUE(torch::all(weights.slice(1, 0, cols) == fW).item<bool>());

  // momentum is set correctly
  EXPECT_TRUE(
      torch::all(weights.slice(1, cols, cols + 1) == kMomentum1).item<bool>());
}

} // namespace fbgemm_gpu::utils
