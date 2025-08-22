/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include "fbgemm_gpu/utils/kernel_launcher.cuh"

namespace fbgemm_gpu::utils {

////////////////////////////////////////////////////////////////////////////////
// GPU Cache Flusher
//
// This utility class is used to flush the L2 cache synchronously on the GPU.
// It is used mainly for benchmarking purposes.
//
////////////////////////////////////////////////////////////////////////////////

__global__ inline void
flush_gpu_cache(char* d_flush, char* d_flush2, bool do_write) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  const char val = d_flush[idx];
  if (do_write * val) {
    d_flush2[idx] = val;
  }
}

class DeviceCacheFlusher {
  size_t cache_size = 40; // A100 40MB L2 cache

  std::vector<char> h_flush;
  char* d_flush1 = nullptr;
  char* d_flush2 = nullptr;

 public:
  DeviceCacheFlusher() {
    // Use the first device to determine L2 cache size
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, 0);
    cache_size = properties.l2CacheSize;

    h_flush.assign(cache_size, 0xFF);
    cudaMalloc(&d_flush1, cache_size * sizeof(char));
    cudaMalloc(&d_flush2, cache_size * sizeof(char));
  }

  DeviceCacheFlusher(const DeviceCacheFlusher&) = delete;

  inline void flush(bool do_write = false) const {
    const unsigned num_blocks = cache_size / 512;

    // Force a copy from host to data1, and from data1 to data2 buffer, in order
    // to flush the L2 cache
    C10_CUDA_CHECK(cudaMemcpy(
        d_flush1, h_flush.data(), cache_size, cudaMemcpyHostToDevice));
    FBGEMM_LAUNCH_KERNEL(
        (flush_gpu_cache),
        num_blocks,
        512,
        0,
        at::cuda::getCurrentCUDAStream(),
        d_flush1,
        d_flush2,
        do_write);
  }

  ~DeviceCacheFlusher() {
    if (d_flush1) {
      cudaFree(d_flush1);
      d_flush1 = nullptr;
    }

    if (d_flush2) {
      cudaFree(d_flush2);
      d_flush2 = nullptr;
    }
  }
};

} // namespace fbgemm_gpu::utils
