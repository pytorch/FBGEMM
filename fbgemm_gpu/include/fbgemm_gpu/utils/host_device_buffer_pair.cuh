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

namespace fbgemm_gpu::utils {

////////////////////////////////////////////////////////////////////////////////
// Host-Device Buffer Pair
//
// This utility class provides a simple abstraction over a host-side and
// device-side buffer pair.  It is used mainly for testing and bencharking
// purposes.
////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct HostDeviceBufferPair {
  // Host-side buffer
  std::vector<T> host;
  // Device-side buffer
  T* device = nullptr;

  HostDeviceBufferPair(size_t size, T value = T()) {
    assign(size, value);
  }

  inline void assign(size_t size, T value = T()) {
    free();
    host.assign(size, value);
    cudaMalloc(&device, size * sizeof(T));
    syncToDevice();
  }

  void deviceRandInit(unsigned long long seed = 1234ULL);

  inline T& operator[](size_t index) {
    return host[index];
  }

  inline const T& operator[](size_t index) const {
    return host[index];
  }

  inline size_t size() const {
    return host.size();
  }

  inline void syncToDevice() {
    cudaMemcpy(
        device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice);
  }

  inline void syncToHost() {
    cudaMemcpy(
        host.data(), device, host.size() * sizeof(T), cudaMemcpyDeviceToHost);
  }

  inline void free() {
    if (device) {
      cudaFree(device);
      device = nullptr;
    }
  }

  ~HostDeviceBufferPair() {
    free();
  }
};

////////////////////////////////////////////////////////////////////////////////
// Random Data Generation
////////////////////////////////////////////////////////////////////////////////

template <>
void HostDeviceBufferPair<float>::deviceRandInit(unsigned long long seed) {
  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MT19937);
  curandSetPseudoRandomGeneratorSeed(gen, seed);
  curandGenerateUniform(gen, device, host.size());
  curandDestroyGenerator(gen);
  cudaDeviceSynchronize();
}

template <>
void HostDeviceBufferPair<uint8_t>::deviceRandInit(unsigned long long seed) {
  // Allocate f32 buffer
  unsigned* d_tmpbuffer_f32;
  cudaMalloc(
      &d_tmpbuffer_f32,
      (host.size() / sizeof(unsigned) + 1) * sizeof(unsigned));

  // Populate the tmp buffer with random f32 values
  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MT19937);
  curandSetPseudoRandomGeneratorSeed(gen, seed);
  curandGenerate(gen, d_tmpbuffer_f32, (host.size() / sizeof(unsigned) + 1));

  // Copy back to 8bit memory
  cudaMemcpy(
      device,
      d_tmpbuffer_f32,
      host.size() * sizeof(uint8_t),
      cudaMemcpyDeviceToDevice);

  // Cleanup
  curandDestroyGenerator(gen);
  cudaFree(d_tmpbuffer_f32);
  cudaDeviceSynchronize();
}

} // namespace fbgemm_gpu::utils
