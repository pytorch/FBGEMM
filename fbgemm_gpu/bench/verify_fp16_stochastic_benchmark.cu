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
#include <chrono>
#include <iostream>

////////////////////////////////////////////////////////////////////////////////
// Direct Stochastic Rounding Kernels
////////////////////////////////////////////////////////////////////////////////

__device__ half float_to_sto_half_direct(float w) {
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
__global__ void convert_float_to_half_direct(half* dst, float* src, int size) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    dst[idx] = float_to_sto_half_direct(src[idx]);
  }
}

////////////////////////////////////////////////////////////////////////////////
// Bitcarry Stochastic Rounding Kernels
////////////////////////////////////////////////////////////////////////////////

__inline__ __device__ float two_to_e(float X) {
  const float Y = 16777216 * X; // 2^24
  const float U = ((Y + X) - Y) * 0.5;
  return U == 0 ? X : U;
}

__device__ half float_to_sto_half_bitcarry(float w) {
  curandState_t state;
  curand_init((unsigned long long)(w * 100), 0, 0, &state);
  float rand = curand_uniform(&state);
  float rand_match_w = two_to_e(w) * rand * 0.0009765625; // 2^(-10)
  float Z = w + rand_match_w;
  return __float2half_rz(Z);
}

__global__ void
convert_float_to_half_bitcarry(half* dst, float* src, int size) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    dst[idx] = float_to_sto_half_bitcarry(src[idx]);
  }
}

////////////////////////////////////////////////////////////////////////////////
// Shortrand Stochastic Rounding Kernels
////////////////////////////////////////////////////////////////////////////////

__device__ half float_to_sto_half_shortrand(float w, uint8_t rand) {
  const unsigned w_int = __float_as_uint(w);
  const unsigned w_new = w_int + (rand << 5);
  return __float2half_rz(__uint_as_float(w_new));
}

__global__ void
convert_float_to_half_shortrand(half* dst, float* src, uint8_t* r, int size) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    dst[idx] = float_to_sto_half_shortrand(src[idx], r[idx]);
  }
}

////////////////////////////////////////////////////////////////////////////////
// AssembleFloat Stochastic Rounding Kernels
////////////////////////////////////////////////////////////////////////////////

__device__ half float_to_sto_half_assemblefloat(float w, uint8_t rand) {
  const unsigned w_int = __float_as_uint(w);
  const unsigned assembles = (w_int & 0xff800000) | (rand << 5);
  const unsigned subtract = (w_int & 0xff800000);
  const float assemble_float =
      __uint_as_float(assembles) - __uint_as_float(subtract);
  return __float2half_rz(w + assemble_float);
}

__global__ void convert_float_to_half_assemblefloat(
    half* dst,
    float* src,
    uint8_t* r,
    int size) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    dst[idx] = float_to_sto_half_assemblefloat(src[idx], r[idx]);
  }
}

////////////////////////////////////////////////////////////////////////////////
// Random Data Generation
////////////////////////////////////////////////////////////////////////////////

void gen_data(float* d_f32_array, int test_size) {
  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MT19937);
  curandSetPseudoRandomGeneratorSeed(gen, 1234ULL); // Random seed
  curandGenerateUniform(gen, d_f32_array, test_size);
  curandDestroyGenerator(gen);
  cudaDeviceSynchronize();
}

// generate 64bit random number and then copy back to 8bit memory
void gen_8bit_random(uint8_t* d_random_number, int test_size) {
  curandGenerator_t gen;
  unsigned* d_random_number_f32;
  cudaMalloc(
      &d_random_number_f32,
      (test_size / sizeof(unsigned) + 1) * sizeof(unsigned));
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MT19937);
  curandSetPseudoRandomGeneratorSeed(gen, 5678ULL); // Random seed
  curandGenerate(gen, d_random_number_f32, (test_size / sizeof(unsigned) + 1));
  cudaMemcpy(
      d_random_number,
      d_random_number_f32,
      test_size * sizeof(uint8_t),
      cudaMemcpyDeviceToDevice);
  curandDestroyGenerator(gen);
  cudaFree(d_random_number_f32);
}

////////////////////////////////////////////////////////////////////////////////
// Cache Flusher
////////////////////////////////////////////////////////////////////////////////

__global__ void flush_gpu(char* d_flush, char* d_flush2, bool do_write) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  const char val = d_flush[idx];
  if (do_write * val) {
    d_flush2[idx] = val;
  }
}

class CacheFlusher {
  size_t cache_size;

  std::vector<char> h_flush;
  char* d_flush;
  char* d_flush2;

 public:
  CacheFlusher() {
    // Use the first device to determine L2 cache size
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, 0);
    cache_size = properties.l2CacheSize;

    h_flush.assign(cache_size, 255);
    cudaMalloc(&d_flush, cache_size * sizeof(char));
    cudaMalloc(&d_flush2, cache_size * sizeof(char));
  }

  inline void flush(bool do_write = false) const {
    // Force a copy from host to data1, and from data1 to data2 buffer, to flush
    // the L2 cache
    cudaMemcpy(d_flush, h_flush.data(), cache_size, cudaMemcpyHostToDevice);
    const unsigned num_blocks = cache_size / 512;
    flush_gpu<<<num_blocks, 512>>>(d_flush, d_flush2, do_write);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    cudaDeviceSynchronize();
  }

  ~CacheFlusher() {
    if (d_flush) {
      cudaFree(d_flush);
    }

    if (d_flush2) {
      cudaFree(d_flush2);
    }
  }
};

////////////////////////////////////////////////////////////////////////////////
// Host-Device Buffer Pair
////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct BufferPair {
  std::vector<T> host;
  T* device = nullptr;

  BufferPair(size_t size) {
    init(size);
  }

  void init(size_t size) {
    free();
    host.reserve(size);
    cudaMalloc(&device, size * sizeof(T));
  }

  void free() {
    if (device) {
      cudaFree(device);
    }
  }

  void syncToDevice() {
    cudaMemcpy(
        device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice);
  }

  void syncToHost() {
    cudaMemcpy(
        host.data(), device, host.size() * sizeof(T), cudaMemcpyDeviceToHost);
  }

  ~BufferPair() {
    free();
  }
};

template <typename KernelFunc, typename... Args>
void time_kernel_run(
    const std::string& description,
    const KernelFunc& kernel,
    dim3 grid,
    dim3 block,
    Args&&... args) {
  std::cout << "[" << description << "] starting kernel run ..." << std::endl;

  const auto start = std::chrono::high_resolution_clock::now();
  kernel<<<grid, block>>>(std::forward<Args>(args)...);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  cudaDeviceSynchronize();
  const auto end = std::chrono::high_resolution_clock::now();

  const auto e = cudaGetLastError();
  if (e != cudaSuccess) {
    std::cout << "[" << description
              << "] CUDA Failure: " << cudaGetErrorString(e) << std::endl;
    std::exit(-1);
  }

  const std::chrono::duration<double> time = end - start;
  std::cout << "[" << description << "] " << time.count() << " sec(s)\n"
            << std::endl;
  return;
}

int main(int argc, char* argv[]) {
  uint8_t* d_random_number;

  const auto flusher = CacheFlusher();

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

  BufferPair<float> f32(test_size);
  BufferPair<half> f16_direct(test_size), f16_bitcarry(test_size),
      f16_shortrand(test_size), f16_assemblefloat(test_size);

  cudaMalloc(&d_random_number, test_size * sizeof(uint8_t));

  // Generate random data
  gen_data(f32.device, test_size);
  gen_8bit_random(d_random_number, test_size);

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
      d_random_number,
      test_size);

  flusher.flush();
  time_kernel_run(
      "AssembleFloat Stochastic Algorithm",
      convert_float_to_half_assemblefloat,
      num_blocks,
      block_size,
      f16_assemblefloat.device,
      f32.device,
      d_random_number,
      test_size);

  if (verbose) {
    f32.syncToHost();
    f16_direct.syncToHost();
    f16_bitcarry.syncToHost();
    f16_shortrand.syncToHost();
    f16_assemblefloat.syncToHost();

    for (int i = 0; i < test_size; i++) {
      std::cout << std::hexfloat << f32.host[i] << ":\t(up:" << std::hexfloat
                << __half2float(__float2half_ru(f32.host[i]))
                << "\tdown:" << std::hexfloat
                << __half2float(__float2half_rd(f32.host[i]))
                << ") \tdirect: " << std::hexfloat
                << __half2float(f16_direct.host[i])
                << "\tbitcarry: " << std::hexfloat
                << __half2float(f16_bitcarry.host[i])
                << " \tshortrand: " << std::hexfloat
                << __half2float(f16_shortrand.host[i])
                << " \tassemblefloat: " << std::hexfloat
                << __half2float(f16_assemblefloat.host[i]) << std::endl;
    }
  }

  return 0;
}
