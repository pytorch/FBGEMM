/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <gtest/gtest.h>

#include <bit>

#include "fbgemm_gpu/utils/kernel_launcher.cuh"
#include "fbgemm_gpu/utils/vec4.cuh"

namespace fbgemm_gpu {
namespace {

template <typename T>
__global__ void load_unaligned_kernel(
    const T* input, float* output, int count) {
  const int i = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (i < count) {
    const auto value = vec4_load_unaligned(input + i);
    *reinterpret_cast<float4*>(output + i) = value.acc;
  }
}

template <typename T>
__global__ void store_unaligned_kernel(
    const float* input, T* output, T* reference, int count) {
  const int i = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (i < count) {
    const Vec4T<T> value(input + i);
    vec4_store_unaligned(value, output + i);
    // The established, aligned global-memory path is the bitwise reference.
    value.store(reference + i);
  }
}

template <typename T>
void checkUnalignedAccess() {
  constexpr int count = 4 * 65536;
  const auto dtype = c10::CppTypeToScalarType<T>::value;
  const auto options = at::TensorOptions().dtype(dtype).device(at::kCPU);
  auto host_input = at::empty({count + 3}, options);
  auto* data = host_input.template data_ptr<T>();
  for (int i = 0; i < count + 3; ++i) {
    if constexpr (std::is_same_v<T, float>) {
      data[i] = (i % 65536 - 32768) * 0.25f;
    } else {
      // All encodings, including signed zero, subnormals and NaN payloads.
      data[i] = T(static_cast<uint16_t>(i), T::from_bits());
    }
  }
  const auto input = host_input.to(at::kCUDA);
  // PyTorch's device conversion provides the independent scalar reference.
  const auto reference_load = input.to(at::kFloat);
  auto loaded = at::empty({count}, input.options().dtype(at::kFloat));

  auto host_values = at::empty({count}, options.dtype(at::kFloat));
  auto* values = host_values.template data_ptr<float>();
  constexpr uint32_t tails[] = {0u, 0x7fffu, 0x8000u, 0xffffu};
  for (uint32_t i = 0; i < count; ++i) {
    values[i] = std::bit_cast<float>(((i % 65536) << 16) | tails[i / 65536]);
  }
  const auto store_input = host_values.to(at::kCUDA);
  auto reference_store = at::empty({count}, input.options());
  auto stored = at::empty({count + 6}, input.options());
  const auto stream = at::cuda::getCurrentCUDAStream();

  for (int offset = 0; offset < 4; ++offset) {
    SCOPED_TRACE(offset);
    FBGEMM_LAUNCH_KERNEL((load_unaligned_kernel<T>), count / (4 * 256), 256, 0,
        stream, input.template data_ptr<T>() + offset,
        loaded.template data_ptr<float>(), count);
    // Compare encodings so NaNs and signed zero are checked too.
    EXPECT_TRUE(at::equal(loaded.view(at::kByte),
        reference_load.slice(0, offset, offset + count).view(at::kByte)));

    stored.fill_(7);
    FBGEMM_LAUNCH_KERNEL((store_unaligned_kernel<T>), count / (4 * 256), 256, 0,
        stream, store_input.template data_ptr<float>(),
        stored.template data_ptr<T>() + offset,
        reference_store.template data_ptr<T>(), count);
    EXPECT_TRUE(
        at::equal(stored.slice(0, offset, offset + count).view(at::kByte),
            reference_store.view(at::kByte)));
    EXPECT_TRUE(stored.slice(0, 0, offset).eq(7).all().template item<bool>());
    EXPECT_TRUE(
        stored.slice(0, offset + count).eq(7).all().template item<bool>());
  }
}

TEST(Vec4UnalignedTest, Float) { checkUnalignedAccess<float>(); }

TEST(Vec4UnalignedTest, Half) { checkUnalignedAccess<at::Half>(); }

TEST(Vec4UnalignedTest, BFloat16) { checkUnalignedAccess<at::BFloat16>(); }

} // namespace
} // namespace fbgemm_gpu
