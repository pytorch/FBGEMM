/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <torch/types.h> // @manual=//caffe2:torch-cpp-cpu

#include <cmath>

#include "fbgemm_gpu/utils/kernel_launcher.cuh"
#include "fbgemm_gpu/utils/vec4.cuh"
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
  const auto extra = pad4(
      static_cast<int32_t>(
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
  const auto extra = pad4(
      static_cast<int32_t>(
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

namespace {

template <typename T>
__global__ void half_vector_memory_order_kernel(at::Half* input, float* output) {
  const Vec4T<T> before(input);
  // Four FP16 encodings of 11, written as one aligned vector.
  *reinterpret_cast<uint2*>(input) = make_uint2(0x49804980, 0x49804980);
  const Vec4T<T> after(input);
  *reinterpret_cast<float4*>(output) = before.acc;
  *reinterpret_cast<float4*>(output + 4) = after.acc;
}

template <typename T>
void checkHalfVectorMemoryOrder() {
  at::Tensor input = at::full(
      {4}, 3.f, at::TensorOptions().dtype(at::kHalf).device(at::kCUDA));
  at::Tensor output = at::empty({8}, input.options().dtype(at::kFloat));
  FBGEMM_LAUNCH_KERNEL(
      (half_vector_memory_order_kernel<T>),
      1,
      1,
      0,
      at::cuda::getCurrentCUDAStream(),
      input.template data_ptr<at::Half>(),
      output.template data_ptr<float>());
  EXPECT_TRUE(at::all(output.slice(0, 0, 4) == 3.f).template item<bool>());
  EXPECT_TRUE(at::all(output.slice(0, 4, 8) == 11.f).template item<bool>());
}

TEST(WeightRow, half_vector_memory_order) {
  checkHalfVectorMemoryOrder<float>();
  checkHalfVectorMemoryOrder<at::Half>();
  checkHalfVectorMemoryOrder<at::BFloat16>();
}

template <typename T>
__global__ void unaligned_optimizer_state_kernel(
    const T* input,
    const float* values,
    float* loaded,
    T* stored,
    T* reference,
    int count) {
  const int i = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (i < count) {
    *reinterpret_cast<float4*>(loaded + i) = vec4_load_unaligned(input + i).acc;
    const Vec4T<T> value(values + i);
    vec4_store_unaligned(value, stored + i);
    // Use the established aligned global-memory store as the reference.
    value.store(reference + i);
  }
}

template <typename T>
void checkUnalignedOptimizerState() {
  const auto dtype = c10::CppTypeToScalarType<T>::value;
  SCOPED_TRACE(c10::toString(dtype));
  // Signed zero, subnormals, overflow, rounding ties, infinities and NaNs.
  const auto values = at::tensor(
                          {0.f,
                           -0.f,
                           1.f,
                           -1.f,
                           0x1p-24f,
                           0x1p-133f,
                           65504.f,
                           65520.f,
                           1.f + 0x1p-11f,
                           1.f + 0x1.8p-10f,
                           1.f + 0x1p-8f,
                           1.f + 0x1.8p-7f,
                           INFINITY,
                           -INFINITY,
                           NAN,
                           -NAN},
                          at::TensorOptions().dtype(at::kFloat))
                          .to(at::kCUDA);
  const int count = values.numel();
  const auto input = values.repeat({2}).to(dtype);
  const auto reference_load = input.to(at::kFloat);
  auto loaded = at::empty({count}, values.options());
  auto reference_store = at::empty({count}, input.options());
  auto stored = at::empty({count + 6}, input.options());
  auto expected = at::empty_like(stored);
  const auto stream = at::cuda::getCurrentCUDAStream();

  for (int offset = 0; offset < 4; ++offset) {
    SCOPED_TRACE(offset);
    stored.fill_(7);
    FBGEMM_LAUNCH_KERNEL(
        (unaligned_optimizer_state_kernel<T>),
        1,
        32,
        0,
        stream,
        input.template data_ptr<T>() + offset,
        values.template data_ptr<float>(),
        loaded.template data_ptr<float>(),
        stored.template data_ptr<T>() + offset,
        reference_store.template data_ptr<T>(),
        count);
    // Compare encodings so NaNs and signed zero are checked too.
    EXPECT_TRUE(
        at::equal(
            loaded.view(at::kByte),
            reference_load.slice(0, offset, offset + count).view(at::kByte)));
    expected.fill_(7);
    expected.slice(0, offset, offset + count).copy_(reference_store);
    // One bytewise comparison checks both stored values and untouched guards.
    EXPECT_TRUE(at::equal(stored.view(at::kByte), expected.view(at::kByte)));
  }
}

TEST(WeightRow, unaligned_optimizer_state_vectors) {
  checkUnalignedOptimizerState<float>();
  checkUnalignedOptimizerState<at::Half>();
  checkUnalignedOptimizerState<at::BFloat16>();
}

} // namespace

} // namespace fbgemm_gpu::utils
