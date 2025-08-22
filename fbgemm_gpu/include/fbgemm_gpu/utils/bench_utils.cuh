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
#include <curand.h>
#include <curand_kernel.h>
#include <vector>

#include "fbgemm_gpu/utils/cuda_prelude.cuh"
#include "fbgemm_gpu/utils/device_cache_flusher.cuh"
#include "fbgemm_gpu/utils/kernel_launcher.cuh"

namespace fbgemm_gpu {

void generate_random_table(float* d_f32_table, unsigned size) {
  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen, 1234ULL); // Random seed
  curandGenerateUniform(gen, d_f32_table, size);
  C10_CUDA_CHECK(cudaGetLastError());
  curandDestroyGenerator(gen);
}

template <typename Lambda>
float benchmark_function(int iters, Lambda&& f) {
  const auto flusher = utils::DeviceCacheFlusher();
  float elapsed = 0;
  cudaEvent_t start, stop;
  C10_CUDA_CHECK(cudaEventCreate(&start));
  C10_CUDA_CHECK(cudaEventCreate(&stop));
  for (int i = 0; i < iters; i++) {
    float local_elapsed = 0;
    flusher.flush(); // Flush L1 cache
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaEventRecord(start, 0));

    // kernel launch here
    f();
    C10_CUDA_CHECK(cudaEventRecord(stop, 0));
    C10_CUDA_CHECK(cudaEventSynchronize(stop));

    C10_CUDA_CHECK(cudaEventElapsedTime(&local_elapsed, start, stop));
    C10_CUDA_CHECK(cudaGetLastError());
    elapsed += local_elapsed;
  }
  C10_CUDA_CHECK(cudaEventDestroy(start));
  C10_CUDA_CHECK(cudaEventDestroy(stop));
  return elapsed / iters;
}

} // namespace fbgemm_gpu
