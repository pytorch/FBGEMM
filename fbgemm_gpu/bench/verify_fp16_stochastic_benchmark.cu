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
#include <curand.h>
#include <curand_kernel.h>

#include <unistd.h>
#include <iostream>

#include "fbgemm_gpu/utils/device_cache_flusher.cuh"
#include "fbgemm_gpu/utils/host_device_buffer_pair.cuh"
#include "fbgemm_gpu/utils/stochastic_rounding.cuh"

namespace fbgemm_gpu {

////////////////////////////////////////////////////////////////////////////////
// FBGEMM Stochastic Rounding Kernels
////////////////////////////////////////////////////////////////////////////////

DEVICE_INLINE half
float_to_sto_half_fbgemm_rand(float x, StochasticRoundingRNGState& state) {
  const auto random_bits = state.rand4();
  uint32_t random_value = random_bits.x;
  uint32_t w_int = __float_as_uint(x);
  unsigned assembles = (w_int & 0xff800000) | (random_value >> 19);
  unsigned subtract = (w_int & 0xff800000);
  float assemble_float = __uint_as_float(assembles) - __uint_as_float(subtract);
  return __float2half_rz(x + assemble_float);
}

__global__ void convert_float_to_half_fbgemm_rand(
    half* dst,
    const float* src,
    int size,
    at::PhiloxCudaState philox_args) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;

  auto state = StochasticRoundingRNGState(philox_args, idx);

  if (idx < size) {
    dst[idx] = float_to_sto_half_fbgemm_rand(src[idx], state);
  }
}

////////////////////////////////////////////////////////////////////////////////
// Direct Stochastic Rounding Kernels
////////////////////////////////////////////////////////////////////////////////

DEVICE_INLINE half float_to_sto_half_direct(float w) {
  curandState_t state;
  curand_init((unsigned long long)(w * 100), 0, 0, &state);
  half up = __float2half_ru(w);
  half down = __float2half_rd(w);
  const float up_f32 = __half2float(up);
  const float down_f32 = __half2float(down);
  // 1 - (w - w_down) / (w_up - w_down) = (w_up - w) / (w_up - w_down) = n / m
  const float m = (up_f32 - down_f32);
  const float rand = curand_uniform(&state);
  if (__float_as_uint(m) == 0) {
    return up;
  }
  const float n = (up_f32 - w);
  return rand > n / m ? up : down;
}

__global__ void
convert_float_to_half_direct(half* dst, const float* src, int size) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    dst[idx] = float_to_sto_half_direct(src[idx]);
  }
}

////////////////////////////////////////////////////////////////////////////////
// Bitcarry Stochastic Rounding Kernels
////////////////////////////////////////////////////////////////////////////////

DEVICE_INLINE float two_to_e(float X) {
  const float Y = 16777216 * X; // 2^24
  const float U = ((Y + X) - Y) * 0.5;
  return U == 0 ? X : U;
}

DEVICE_INLINE half float_to_sto_half_bitcarry(float w) {
  curandState_t state;
  curand_init((unsigned long long)(w * 100), 0, 0, &state);
  float rand = curand_uniform(&state);
  float rand_match_w = two_to_e(w) * rand * 0.0009765625; // 2^(-10)
  float Z = w + rand_match_w;
  return __float2half_rz(Z);
}

__global__ void
convert_float_to_half_bitcarry(half* dst, const float* src, int size) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    dst[idx] = float_to_sto_half_bitcarry(src[idx]);
  }
}

////////////////////////////////////////////////////////////////////////////////
// Shortrand Stochastic Rounding Kernels
////////////////////////////////////////////////////////////////////////////////

DEVICE_INLINE half float_to_sto_half_shortrand(float w, uint8_t rand) {
  const unsigned w_int = __float_as_uint(w);
  const unsigned w_new = w_int + (rand << 5);
  return __float2half_rz(__uint_as_float(w_new));
}

__global__ void convert_float_to_half_shortrand(
    half* dst,
    const float* src,
    const uint8_t* r,
    int size) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    dst[idx] = float_to_sto_half_shortrand(src[idx], r[idx]);
  }
}

////////////////////////////////////////////////////////////////////////////////
// AssembleFloat Stochastic Rounding Kernels
////////////////////////////////////////////////////////////////////////////////

DEVICE_INLINE half float_to_sto_half_assemblefloat(float w, uint8_t rand) {
  const unsigned w_int = __float_as_uint(w);
  const unsigned assembles = (w_int & 0xff800000) | (rand << 5);
  const unsigned subtract = (w_int & 0xff800000);
  const float assemble_float =
      __uint_as_float(assembles) - __uint_as_float(subtract);
  return __float2half_rz(w + assemble_float);
}

__global__ void convert_float_to_half_assemblefloat(
    half* dst,
    const float* src,
    const uint8_t* r,
    int size) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    dst[idx] = float_to_sto_half_assemblefloat(src[idx], r[idx]);
  }
}

////////////////////////////////////////////////////////////////////////////////
// Benchmarking
////////////////////////////////////////////////////////////////////////////////

template <typename KernelFunc, typename... Args>
void time_kernel_run(
    const std::string& description,
    const KernelFunc& kernel,
    dim3 grid,
    dim3 block,
    Args&&... args) {
  std::cout << "[" << description << "] starting kernel run ..." << std::endl;

  // Create CUDA events to time the kernel
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Execute the kernel, while recording the start and end times
  cudaEventRecord(start);
  kernel<<<grid, block>>>(std::forward<Args>(args)...);
  cudaEventRecord(stop);

  // Synchronize to ensure that the kernel has completed
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  cudaEventSynchronize(stop);

  // Check for kernel execution errors
  const auto e = cudaGetLastError();
  if (e != cudaSuccess) {
    std::cout << "[" << description
              << "] CUDA Failure: " << cudaGetErrorString(e) << std::endl;
    std::exit(-1);
  }

  // Calculate the elapsed time in milliseconds
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  std::cout << "[" << description << "] " << milliseconds << " ms\n"
            << std::endl;
  return;
}

} // namespace fbgemm_gpu

using namespace fbgemm_gpu;

int main(int argc, char* argv[]) {
  int test_size = 10;
  bool verbose = false;
  int opt;
  while ((opt = getopt(argc, argv, "n:v")) != -1) {
    switch (opt) {
      case 'n':
        test_size = atoi(optarg);
        break;
      case 'v':
        verbose = true;
        break;
    }
  }

  std::cout << "Start stochastic algorithm tests with test_size = " << test_size
            << "\n"
            << std::endl;

  // Initialize buffers
  float value = 1.00048828125f; // Replace with your desired value
  auto f32 = utils::HostDeviceBufferPair<float>(test_size, value);

  utils::HostDeviceBufferPair<half> f16_direct(test_size),
      f16_bitcarry(test_size), f16_shortrand(test_size),
      f16_assemblefloat(test_size), f16_fbgemmrand(test_size);

  // Random bits
  auto random = utils::HostDeviceBufferPair<uint8_t>(test_size);
  random.deviceRandInit(1234ULL);

  const auto flusher = utils::DeviceCacheFlusher();

  // RNG input (for FBGEMM stochastic rounding)
  const auto gen = at::cuda::detail::getDefaultCUDAGenerator();
  const auto rng_input =
      at::check_generator<at::CUDAGeneratorImpl>(gen)->philox_cuda_state(4);

  constexpr int block_size = 128;
  const int num_blocks = (test_size + block_size - 1) / block_size;

  flusher.flush();
  time_kernel_run(
      "Direct Stochastic Algorithm",
      convert_float_to_half_direct,
      num_blocks,
      block_size,
      f16_direct.device,
      f32.device,
      test_size);

  flusher.flush();
  time_kernel_run(
      "Bitcarry Stochastic Algorithm",
      convert_float_to_half_bitcarry,
      num_blocks,
      block_size,
      f16_bitcarry.device,
      f32.device,
      test_size);

  flusher.flush();
  time_kernel_run(
      "Shortrand Stochastic Algorithm",
      convert_float_to_half_shortrand,
      num_blocks,
      block_size,
      f16_shortrand.device,
      f32.device,
      random.device,
      test_size);

  flusher.flush();
  time_kernel_run(
      "AssembleFloat Stochastic Algorithm",
      convert_float_to_half_assemblefloat,
      num_blocks,
      block_size,
      f16_assemblefloat.device,
      f32.device,
      random.device,
      test_size);

  flusher.flush();
  time_kernel_run(
      "FBGEMM Stochastic Algorithm",
      convert_float_to_half_fbgemm_rand,
      num_blocks,
      block_size,
      f16_fbgemmrand.device,
      f32.device,
      test_size,
      rng_input);

  if (verbose) {
    f32.syncToHost();
    f16_direct.syncToHost();
    f16_bitcarry.syncToHost();
    f16_shortrand.syncToHost();
    f16_assemblefloat.syncToHost();
    f16_fbgemmrand.syncToHost();

    for (int i = 0; i < test_size; i++) {
      // std::cout << std::hexfloat << f32[i] << ":\t(up:" << std::hexfloat
      //           << __half2float(__float2half_ru(f32[i]))
      //           << "\tdown:" << std::hexfloat
      //           << __half2float(__float2half_rd(f32[i]))
      //           << ") \tdirect: " << std::hexfloat
      //           << __half2float(f16_direct[i])
      //           << "\tbitcarry: " << std::hexfloat
      //           << __half2float(f16_bitcarry[i])
      //           << " \tshortrand: " << std::hexfloat
      //           << __half2float(f16_shortrand[i])
      //           << " \tassemblefloat: " << std::hexfloat
      //           << __half2float(f16_assemblefloat[i]) << std::endl;

      printf(
          "%.11f:\t(up:%.11f\tdown:%.11f) \tdirect:%.14f \tshortrand:%.11f \tfbgemmrand:%.11f \n",
          f32[i],
          __half2float(__float2half_ru(f32[i])),
          __half2float(__float2half_rd(f32[i])),
          __half2float(f16_direct[i]),
          // __half2float(f16_bitcarry[i]),
          __half2float(f16_shortrand[i]),
          // __half2float(f16_assemblefloat[i]),
          __half2float(f16_fbgemmrand[i]));
    }
  }

  return 0;
}
