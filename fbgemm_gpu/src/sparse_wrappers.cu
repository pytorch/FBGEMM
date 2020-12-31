/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// Copyright 2004-present Facebook. All Rights Reserved.

#include "cub/device/device_scan.cuh"
#include "fbgemm_gpu/cuda_utils.cuh"
#include "fbgemm_gpu/sparse_ops.cuh"
#include "fbgemm_gpu/sparse_wrappers.cuh"

void fbgemm_gpu_test::permute_sparse_features(
    int weights_size,
    int T,
    int B,
    const int* permute,
    const long* lengths,
    const long* indices,
    const float* weights,
    long* permuted_lengths,
    long* permuted_indices,
    float* permuted_weights) {
  int threads_per_block_1 = 256;
  int num_blocks_1 = (T * B + threads_per_block_1 - 1) / threads_per_block_1;
  permute_lengths_kernel<long><<<num_blocks_1, threads_per_block_1>>>(
      T, B, lengths, permute, permuted_lengths);
  CUDA_CHECK(cudaGetLastError());

  size_t temp_storage_bytes = 0;
  long* input_offsets_gpu;
  long* output_offsets_gpu;
  long* temp_storage_gpu;

  CUDA_CHECK(cudaMalloc((void**)&input_offsets_gpu, T * B * sizeof(long)));
  CUDA_CHECK(cudaMalloc((void**)&output_offsets_gpu, T * B * sizeof(long)));

  cub::DeviceScan::ExclusiveSum(
      nullptr, temp_storage_bytes, lengths, input_offsets_gpu, T * B);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(
      cudaMalloc((void**)&temp_storage_gpu, temp_storage_bytes * sizeof(long)));
  cub::DeviceScan::ExclusiveSum(
      temp_storage_gpu, temp_storage_bytes, lengths, input_offsets_gpu, T * B);
  CUDA_CHECK(cudaGetLastError());

  cub::DeviceScan::ExclusiveSum(
      temp_storage_gpu,
      temp_storage_bytes,
      permuted_lengths,
      output_offsets_gpu,
      T * B);
  CUDA_CHECK(cudaGetLastError());

  int BT_blocks = 32;
  dim3 threads_per_block_2(32, BT_blocks);
  int num_blocks_2 = (B * T + BT_blocks - 1) / BT_blocks;
  permute_indices_weights_kernel<true, long, float>
      <<<num_blocks_2, threads_per_block_2>>>(
          weights_size,
          T,
          B,
          indices,
          weights,
          permute,
          input_offsets_gpu,
          output_offsets_gpu,
          permuted_indices,
          permuted_weights);
  CUDA_CHECK(cudaGetLastError());
  cudaFree(temp_storage_gpu);
  cudaFree(input_offsets_gpu);
  cudaFree(output_offsets_gpu);
}

void fbgemm_gpu_test::bucketize_sparse_features(
    int lengths_size,
    int my_size,
    const long* lengths,
    const long* indices,
    const float* weights,
    long* bucketized_lengths,
    long* bucketized_indices,
    float* bucketized_weights,
    long* bucketized_pos) {
  int threads_per_block = 256;
  int num_blocks = (lengths_size + threads_per_block - 1) / threads_per_block;
  size_t temp_storage_bytes = 0;

  // gpu ptrs
  long* offsets_ptr_gpu;
  long* bucketized_offsets_ptr_gpu;
  long* temp_storage_gpu;

  CUDA_CHECK(cudaMalloc((void**)&offsets_ptr_gpu, lengths_size * sizeof(long)));
  CUDA_CHECK(cudaMalloc(
      (void**)&bucketized_offsets_ptr_gpu,
      my_size * lengths_size * sizeof(long)));
  // compute offsets
  cub::DeviceScan::InclusiveSum(
      nullptr, temp_storage_bytes, lengths, offsets_ptr_gpu, lengths_size);
  CUDA_CHECK(
      cudaMalloc((void**)&temp_storage_gpu, temp_storage_bytes * sizeof(long)));
  cub::DeviceScan::InclusiveSum(
      temp_storage_gpu,
      temp_storage_bytes,
      lengths,
      offsets_ptr_gpu,
      lengths_size);
  CUDA_CHECK(cudaGetLastError());
  cudaFree(temp_storage_gpu);

  // kernel 1
  _bucketize_sparse_features_cuda_kernel1<<<num_blocks, threads_per_block>>>(
      lengths_size, my_size, offsets_ptr_gpu, indices, bucketized_lengths);
  CUDA_CHECK(cudaGetLastError());
  // compute bucketized offsets
  temp_storage_bytes = 0;
  cub::DeviceScan::ExclusiveSum(
      nullptr,
      temp_storage_bytes,
      bucketized_lengths,
      bucketized_offsets_ptr_gpu,
      my_size * lengths_size);
  CUDA_CHECK(
      cudaMalloc((void**)&temp_storage_gpu, temp_storage_bytes * sizeof(long)));
  cub::DeviceScan::ExclusiveSum(
      temp_storage_gpu,
      temp_storage_bytes,
      bucketized_lengths,
      bucketized_offsets_ptr_gpu,
      my_size * lengths_size);
  CUDA_CHECK(cudaGetLastError());
  cudaFree(temp_storage_gpu);

  // kernel 2
  _bucketize_sparse_features_cuda_kernel2<true, true, long, float>
      <<<num_blocks, threads_per_block>>>(
          lengths_size,
          my_size,
          offsets_ptr_gpu,
          indices,
          weights,
          bucketized_offsets_ptr_gpu,
          bucketized_indices,
          bucketized_weights,
          bucketized_pos);
  CUDA_CHECK(cudaGetLastError());
  cudaFree(offsets_ptr_gpu);
  cudaFree(bucketized_offsets_ptr_gpu);
}
