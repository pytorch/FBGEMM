/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <gtest/gtest.h>
#include "fbgemm_gpu/batched_unary_embedding_wrappers.cuh"
#include "fbgemm_gpu/cuda_utils.cuh"

class FBGEMMGPUBatchUnaryEmbeddingTest : public ::testing::Test {
 protected:
  static long* indices;
  static long* offsets;
  static float* weights;
  static int* hash_sizes;
  static float* output_ref;
  static long* table_offsets;
  static float* grad_weight_ref;
  static float* grad_output_ref;

  constexpr static int indices_size = 6;
  constexpr static int T = 2;
  constexpr static int num_task = 2;
  constexpr static int B = 3;

  static void SetUpTestCase() {
    hash_sizes = new int[T]{2, 3};
    table_offsets = new long[T + 1]{0, 2, 5};
    offsets = new long[indices_size + 1]{0, 1, 2, 3, 4, 5, 6};
    indices = new long[indices_size]{1, 1, 1, 0, 2, 1};
    weights = new float[(hash_sizes[0] + hash_sizes[1]) * num_task]{-0.1264,
                                                                    0.2836,
                                                                    -0.5619,
                                                                    0.5717,
                                                                    0.1725,
                                                                    -0.2929,
                                                                    0.2342,
                                                                    0.0099,
                                                                    -0.5364,
                                                                    -0.4393};
    output_ref = new float[T * num_task * B]{0.2836,
                                             -0.5619,
                                             0.2836,
                                             0.1725,
                                             0.2836,
                                             0.5717,
                                             0.2342,
                                             0.0099,
                                             0.2342,
                                             -0.4393,
                                             0.2342,
                                             -0.5364};
    grad_output_ref = new float[T * num_task * B]{-0.1434,
                                                  0.0576,
                                                  -0.0076,
                                                  -0.1141,
                                                  -0.0708,
                                                  -0.0876,
                                                  -0.0649,
                                                  -0.0071,
                                                  -0.0240,
                                                  0.1031,
                                                  0.2131,
                                                  -0.1412};
    grad_weight_ref =
        new float[(hash_sizes[0] + hash_sizes[1]) * num_task]{-0.0000,
                                                              -0.2218,
                                                              0.0576,
                                                              -0.0876,
                                                              -0.1141,
                                                              0.0000,
                                                              0.1243,
                                                              -0.0071,
                                                              -0.1412,
                                                              0.1031};
  }

  static void TearDownTestCase() {
    delete[] indices;
    delete[] offsets;
    delete[] weights;
    delete[] hash_sizes;
    delete[] output_ref;
    delete[] table_offsets;
    delete[] grad_weight_ref;
    delete[] grad_output_ref;
  }
};
long* FBGEMMGPUBatchUnaryEmbeddingTest::indices;
long* FBGEMMGPUBatchUnaryEmbeddingTest::offsets;
int* FBGEMMGPUBatchUnaryEmbeddingTest::hash_sizes;
float* FBGEMMGPUBatchUnaryEmbeddingTest::weights;
float* FBGEMMGPUBatchUnaryEmbeddingTest::output_ref;
long* FBGEMMGPUBatchUnaryEmbeddingTest::table_offsets;
float* FBGEMMGPUBatchUnaryEmbeddingTest::grad_weight_ref;
float* FBGEMMGPUBatchUnaryEmbeddingTest::grad_output_ref;

TEST_F(FBGEMMGPUBatchUnaryEmbeddingTest, forward_test) {
  int device_cnt;
  cudaGetDeviceCount(&device_cnt);
  if (device_cnt == 0) {
    GTEST_SKIP();
  }
  // gpu ptrs
  long* offsets_gpu_ptr;
  long* indices_gpu_ptr;
  long* table_offsets_gpu_ptr;
  float* embedding_table_gpu_ptr;
  float* output_gpu_ptr;

  // cpu ptrs
  float* output_cpu_ptr = new float[T * num_task * B];
  CUDA_CHECK(
      cudaMalloc((void**)&offsets_gpu_ptr, (indices_size + 1) * sizeof(long)));
  CUDA_CHECK(cudaMalloc((void**)&indices_gpu_ptr, indices_size * sizeof(long)));
  CUDA_CHECK(cudaMalloc(
      (void**)&embedding_table_gpu_ptr,
      (hash_sizes[0] + hash_sizes[1]) * num_task * sizeof(float)));
  CUDA_CHECK(
      cudaMalloc((void**)&table_offsets_gpu_ptr, (T + 1) * sizeof(long)));
  CUDA_CHECK(
      cudaMalloc((void**)&output_gpu_ptr, T * num_task * B * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(
      offsets_gpu_ptr,
      offsets,
      (indices_size + 1) * sizeof(long),
      cudaMemcpyKind::cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(
      indices_gpu_ptr,
      indices,
      indices_size * sizeof(long),
      cudaMemcpyKind::cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(
      embedding_table_gpu_ptr,
      weights,
      (hash_sizes[0] + hash_sizes[1]) * num_task * sizeof(float),
      cudaMemcpyKind::cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(
      table_offsets_gpu_ptr,
      table_offsets,
      (T + 1) * sizeof(long),
      cudaMemcpyKind::cudaMemcpyHostToDevice));

  fbgemm_gpu_test::batched_unary_embeddings_forward(
      num_task,
      B,
      T,
      embedding_table_gpu_ptr,
      table_offsets_gpu_ptr,
      offsets_gpu_ptr,
      indices_gpu_ptr,
      output_gpu_ptr);

  CUDA_CHECK(cudaMemcpy(
      output_cpu_ptr,
      output_gpu_ptr,
      T * num_task * B * sizeof(float),
      cudaMemcpyKind::cudaMemcpyDeviceToHost));
  cudaFree(offsets_gpu_ptr);
  cudaFree(indices_gpu_ptr);
  cudaFree(embedding_table_gpu_ptr);
  cudaFree(table_offsets_gpu_ptr);
  cudaFree(output_gpu_ptr);
  for (int i = 0; i < T * num_task * B; i++) {
    ASSERT_FLOAT_EQ(output_cpu_ptr[i], output_ref[i]);
  }
  delete[] output_cpu_ptr;
}

TEST_F(FBGEMMGPUBatchUnaryEmbeddingTest, backward_test) {
  int device_cnt;
  cudaGetDeviceCount(&device_cnt);
  if (device_cnt == 0) {
    GTEST_SKIP();
  }
  // gpu ptrs
  long* offsets_gpu_ptr;
  long* indices_gpu_ptr;
  long* table_offsets_gpu_ptr;
  float* grad_output_gpu_ptr;
  float* grad_weight_gpu_ptr;

  // cpu ptrs
  float* grad_weight_cpu_ptr =
      new float[(hash_sizes[0] + hash_sizes[1]) * num_task];
  CUDA_CHECK(
      cudaMalloc((void**)&offsets_gpu_ptr, (indices_size + 1) * sizeof(long)));
  CUDA_CHECK(cudaMalloc((void**)&indices_gpu_ptr, indices_size * sizeof(long)));
  CUDA_CHECK(cudaMalloc(
      (void**)&grad_output_gpu_ptr, T * num_task * B * sizeof(float)));
  CUDA_CHECK(
      cudaMalloc((void**)&table_offsets_gpu_ptr, (T + 1) * sizeof(long)));
  CUDA_CHECK(cudaMalloc(
      (void**)&grad_weight_gpu_ptr,
      (hash_sizes[0] + hash_sizes[1]) * num_task * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(
      offsets_gpu_ptr,
      offsets,
      (indices_size + 1) * sizeof(long),
      cudaMemcpyKind::cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(
      indices_gpu_ptr,
      indices,
      indices_size * sizeof(long),
      cudaMemcpyKind::cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(
      grad_output_gpu_ptr,
      grad_output_ref,
      T * num_task * B * sizeof(float),
      cudaMemcpyKind::cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(
      table_offsets_gpu_ptr,
      table_offsets,
      (T + 1) * sizeof(long),
      cudaMemcpyKind::cudaMemcpyHostToDevice));

  fbgemm_gpu_test::batched_unary_embeddings_backward(
      num_task,
      B,
      T,
      grad_output_gpu_ptr,
      table_offsets_gpu_ptr,
      offsets_gpu_ptr,
      indices_gpu_ptr,
      grad_weight_gpu_ptr);

  CUDA_CHECK(cudaMemcpy(
      grad_weight_cpu_ptr,
      grad_weight_gpu_ptr,
      (hash_sizes[0] + hash_sizes[1]) * num_task * sizeof(float),
      cudaMemcpyKind::cudaMemcpyDeviceToHost));
  cudaFree(offsets_gpu_ptr);
  cudaFree(indices_gpu_ptr);
  cudaFree(grad_output_gpu_ptr);
  cudaFree(table_offsets_gpu_ptr);
  cudaFree(grad_weight_gpu_ptr);
  for (int i = 0; i < (hash_sizes[0] + hash_sizes[1]) * num_task; i++) {
    ASSERT_NEAR(grad_weight_cpu_ptr[i], grad_weight_ref[i], 0.0002);
  }
  delete[] grad_weight_cpu_ptr;
}
