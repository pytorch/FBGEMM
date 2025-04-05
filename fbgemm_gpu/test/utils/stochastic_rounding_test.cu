/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <gtest/gtest.h>

#include "fbgemm_gpu/utils/host_device_buffer_pair.cuh"
#include "fbgemm_gpu/utils/stochastic_rounding.cuh"

namespace fbgemm_gpu::utils {

////////////////////////////////////////////////////////////////////////////////
// FBGEMM Stochastic Rounding Kernel
////////////////////////////////////////////////////////////////////////////////

__global__ void convert_float_to_half_fbgemm_rand(
    half* dst,
    const float* src,
    int size,
    at::PhiloxCudaState philox_args) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size) {
    auto random_bits = StochasticRoundingRNGState(philox_args, idx).rand4();
    dst[idx] = stochastic_rounding_scalar(src[idx], random_bits.x);
  }
}

////////////////////////////////////////////////////////////////////////////////
// Rounding Up Kernel
////////////////////////////////////////////////////////////////////////////////

template <int rounding_choice>
__global__ void
convert_float_to_half_deterministic(half* dst, const float* src, int size) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    if constexpr (rounding_choice > 0) {
      dst[idx] = __float2half_ru(src[idx]);
    } else if constexpr (rounding_choice < 0) {
      dst[idx] = __float2half_rd(src[idx]);
    } else {
      dst[idx] = __float2half_rz(src[idx]);
    }
  }
}

half float2half_ru(float x) {
#ifdef USE_ROCM
  auto f16 = utils::HostDeviceBufferPair<half>(1);
  auto f32 = utils::HostDeviceBufferPair<float>(1, x);

  convert_float_to_half_deterministic<1><<<1, 32>>>(f16.device, f32.device, 1);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  f16.syncToHost();
  return f16[0];

#else
  return __float2half_ru(x);
#endif
}

////////////////////////////////////////////////////////////////////////////////
// Benchmarking
////////////////////////////////////////////////////////////////////////////////

inline at::PhiloxCudaState philox_rng(long seed) {
  at::manual_seed(seed);
  const auto gen = at::cuda::detail::getDefaultCUDAGenerator();
  return at::check_generator<at::CUDAGeneratorImpl>(gen)->philox_cuda_state(4);
}

inline bool half_equal(const half& a, const half& b) {
  // https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/common/float16.h
  return reinterpret_cast<__half_raw*>(const_cast<half*>(&a))->x ==
      reinterpret_cast<__half_raw*>(const_cast<half*>(&b))->x;
}

void test_stochastic_rounding(float test_value, int num_samples = 10000000) {
  // Expected FP16 values and their FP32 representation
  const half h_floor = __float2half_rz(test_value);
  const half h_ceil = float2half_ru(test_value);
  const float f_floor = __half2float(h_floor);
  const float f_ceil = __half2float(h_ceil);

  // Expected probability of rounding upwards
  const float expected_probability =
      (test_value - f_floor) / (f_ceil - f_floor);

  printf(
      "\n"
      "Testing FP32 value  : %.11f\n"
      "FP16 floor          : %.11f (0x%04x)\n"
      "FP16 ceil           : %.11f (0x%04x)\n",
      test_value,
      __half2float(h_floor),
      *reinterpret_cast<const uint16_t*>(&h_floor),
      __half2float(h_ceil),
      *reinterpret_cast<const uint16_t*>(&h_ceil));

  constexpr int block_size = 128;
  const int num_blocks = (num_samples + block_size - 1) / block_size;

  // Set up buffers with the test value
  auto f32 = utils::HostDeviceBufferPair<float>(num_samples, test_value);
  auto f16 = utils::HostDeviceBufferPair<half>(num_samples);
  const auto rng_input = philox_rng(1234567890L);

  // Convert FP32 to FP16 using stochastic rounding
  convert_float_to_half_fbgemm_rand<<<num_blocks, block_size>>>(
      f16.device, f32.device, num_samples, rng_input);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // Sync buffer back to host to compare
  f16.syncToHost();

  // Compare values and count number of round-ups
  int round_up_count = 0;
  for (const auto x : f16.host) {
    if (half_equal(x, h_ceil)) {
      round_up_count++;
    }
  }

  // Calculate actual probability of rounding up and difference from expected
  const float actual_probability =
      static_cast<float>(round_up_count) / num_samples;
  const float difference = std::abs(actual_probability - expected_probability);

  printf(
      "Results:\n"
      "Number of samples    : %d\n"
      "Round-up Count       : %d\n"
      "Expected probability : %.11f\n"
      "Actual probability   : %.11f\n"
      "Difference           : %.11f\n",
      num_samples,
      round_up_count,
      expected_probability,
      actual_probability,
      difference);

  EXPECT_TRUE(difference < 1e-4f)
      << "Expected difference in probability of rounding up with stochastic rounding should less than 1e-4f";
}

TEST(StochasticRoundingTest, stochastic_rounding) {
  test_stochastic_rounding(1.1f);
  test_stochastic_rounding(2.7f);
}

} // namespace fbgemm_gpu::utils
