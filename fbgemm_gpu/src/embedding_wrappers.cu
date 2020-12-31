/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// Copyright 2004-present Facebook. All Rights Reserved.
#include <cuda.h>
#include <algorithm>
#include <cassert>
#include "fbgemm_gpu/batched_unary_embeddings.cuh"
#include "fbgemm_gpu/cuda_utils.cuh"
#include "fbgemm_gpu/embedding_wrappers.cuh"

void fbgemm_gpu_test::batched_unary_embeddings_forward(
    const int32_t N,
    const int32_t B,
    const int32_t T,
    const float* __restrict__ weight,
    const long* __restrict__ table_offsets,
    const long* __restrict__ offsets,
    const long* __restrict__ indices,
    float* __restrict__ output) {
  int32_t threads = std::min<int32_t>(B, 512);
  dim3 blocks((B + threads - 1) / threads, T, N);
  assert(T <= 65535);
  assert(N <= 65535);
  batched_unary_embeddings_forward_kernel<float><<<blocks, threads>>>(
      N, B, T, weight, table_offsets, offsets, indices, output);
  CUDA_CHECK(cudaGetLastError());
}

void fbgemm_gpu_test::batched_unary_embeddings_backward(
    const int32_t N,
    const int32_t B,
    const int32_t T,
    const float* __restrict__ grad_output,
    const long* __restrict__ table_offsets,
    const long* __restrict__ offsets,
    const long* __restrict__ indices,
    float* __restrict__ grad_weight) {
  int threads = std::min<int32_t>(N * T, 512);
  dim3 blocks((N * T + threads - 1) / threads);
  batched_unary_embeddings_backward_kernel<float><<<blocks, threads>>>(
      N, B, T, grad_output, table_offsets, offsets, indices, grad_weight);
  CUDA_CHECK(cudaGetLastError());
}
