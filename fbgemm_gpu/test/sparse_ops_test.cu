/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <gtest/gtest.h>
#include "fbgemm_gpu/cuda_utils.cuh"
#include "fbgemm_gpu/sparse_wrappers.cuh"

class FBGEMMGPUPermuteSparseFeaturesTest : public ::testing::Test {
 protected:
  static long* lengths;
  static long* indices;
  static float* weights;
  static long* permuted_lengths_ref;
  static long* permuted_indices_ref;
  static float* permuted_weights_ref;
  constexpr static int T = 3;
  constexpr static int B = 2;
  constexpr static int indices_size = 9;
  constexpr static int weights_size = 9;

  static void SetUpTestCase() {
    lengths = new long[6]{0, 3, 2, 0, 1, 3};
    indices = new long[indices_size]{0, 1, 2, 3, 4, 5, 6, 7, 8};
    weights = new float[indices_size]{0, 10, 20, 30, 40, 50, 60, 70, 80};
    permuted_lengths_ref = new long[6]{1, 3, 0, 3, 2, 0};
    permuted_indices_ref = new long[indices_size]{5, 6, 7, 8, 0, 1, 2, 3, 4};
    permuted_weights_ref =
        new float[weights_size]{50, 60, 70, 80, 0, 10, 20, 30, 40};
  }

  static void TearDownTestCase() {
    delete[] lengths;
    delete[] indices;
    delete[] weights;
    delete[] permuted_lengths_ref;
    delete[] permuted_indices_ref;
    delete[] permuted_weights_ref;
  }
};
long* FBGEMMGPUPermuteSparseFeaturesTest::lengths;
long* FBGEMMGPUPermuteSparseFeaturesTest::indices;
float* FBGEMMGPUPermuteSparseFeaturesTest::weights;
long* FBGEMMGPUPermuteSparseFeaturesTest::permuted_lengths_ref;
long* FBGEMMGPUPermuteSparseFeaturesTest::permuted_indices_ref;
float* FBGEMMGPUPermuteSparseFeaturesTest::permuted_weights_ref;

TEST_F(FBGEMMGPUPermuteSparseFeaturesTest, permute_sparse_features_test) {
  int device_cnt;
  cudaGetDeviceCount(&device_cnt);
  if (device_cnt == 0) {
    GTEST_SKIP();
  }
  // permute indices
  std::vector<int> permute{2, 0, 1};
  // gpu input ptrs
  int* permute_ptr_gpu;
  long* lengths_ptr_gpu;
  long* indices_ptr_gpu;
  float* weights_ptr_gpu;

  // gpu output ptrs
  long* permuted_lengths_gpu;
  long* permuted_indices_gpu;
  float* permuted_weights_gpu;

  // cpu ptrs
  long* lengths_ptr_cpu = lengths;
  int* permute_ptr_cpu = permute.data();
  long* permuted_lengths_cpu = new long[T * B];
  long* permuted_indices_cpu = new long[indices_size];
  float* permuted_weights_cpu = new float[weights_size];

  CUDA_CHECK(cudaMalloc((void**)&lengths_ptr_gpu, T * B * sizeof(long)));
  CUDA_CHECK(cudaMalloc((void**)&indices_ptr_gpu, indices_size * sizeof(long)));
  CUDA_CHECK(
      cudaMalloc((void**)&weights_ptr_gpu, weights_size * sizeof(float)));

  CUDA_CHECK(cudaMalloc((void**)&permute_ptr_gpu, T * sizeof(int)));
  CUDA_CHECK(cudaMalloc((void**)&permuted_lengths_gpu, T * B * sizeof(long)));
  CUDA_CHECK(
      cudaMalloc((void**)&permuted_indices_gpu, indices_size * sizeof(long)));
  CUDA_CHECK(
      cudaMalloc((void**)&permuted_weights_gpu, weights_size * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(
      permute_ptr_gpu,
      permute_ptr_cpu,
      T * sizeof(int),
      cudaMemcpyKind::cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(
      lengths_ptr_gpu,
      lengths_ptr_cpu,
      T * B * sizeof(long),
      cudaMemcpyKind::cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(
      indices_ptr_gpu,
      indices,
      indices_size * sizeof(long),
      cudaMemcpyKind::cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(
      weights_ptr_gpu,
      weights,
      weights_size * sizeof(float),
      cudaMemcpyKind::cudaMemcpyHostToDevice));

  fbgemm_gpu_test::permute_sparse_features(
      weights_size,
      T,
      B,
      permute_ptr_gpu,
      lengths_ptr_gpu,
      indices_ptr_gpu,
      weights_ptr_gpu,
      permuted_lengths_gpu,
      permuted_indices_gpu,
      permuted_weights_gpu);

  CUDA_CHECK(cudaMemcpy(
      permuted_lengths_cpu,
      permuted_lengths_gpu,
      T * B * sizeof(long),
      cudaMemcpyKind::cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(
      permuted_indices_cpu,
      permuted_indices_gpu,
      indices_size * sizeof(long),
      cudaMemcpyKind::cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(
      permuted_weights_cpu,
      permuted_weights_gpu,
      weights_size * sizeof(float),
      cudaMemcpyKind::cudaMemcpyDeviceToHost));

  cudaFree(permute_ptr_gpu);
  cudaFree(lengths_ptr_gpu);
  cudaFree(indices_ptr_gpu);
  cudaFree(weights_ptr_gpu);
  cudaFree(permuted_lengths_gpu);
  cudaFree(permuted_indices_gpu);
  cudaFree(permuted_weights_gpu);

  for (int i = 0; i < T * B; i++)
    ASSERT_EQ(permuted_lengths_cpu[i], permuted_lengths_ref[i]);
  for (int i = 0; i < indices_size; i++) {
    ASSERT_EQ(permuted_weights_cpu[i], permuted_weights_ref[i]);
    ASSERT_EQ(permuted_indices_cpu[i], permuted_indices_ref[i]);
  }
  delete[] permuted_lengths_cpu;
  delete[] permuted_indices_cpu;
  delete[] permuted_weights_cpu;
}

class FBGEMMGPUBucketizeSparseFeaturesTest : public ::testing::Test {
 protected:
  static long* lengths;
  static long* indices;
  static float* weights;
  static long* bucketized_lengths_ref;
  static long* bucketized_indices_ref;
  static float* bucketized_weights_ref;
  static long* bucketized_pos_ref;
  constexpr static int my_size = 2;
  constexpr static int lengths_size = 4;
  constexpr static int indices_size = 6;
  constexpr static int weights_size = 6;

  static void SetUpTestCase() {
    lengths = new long[lengths_size]{0, 2, 1, 3};
    indices = new long[indices_size]{10, 10, 15, 20, 25, 30};
    weights = new float[weights_size]{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    bucketized_lengths_ref = new long[8]{0, 2, 0, 2, 0, 0, 1, 1};
    bucketized_indices_ref = new long[indices_size]{5, 5, 10, 15, 7, 12};
    bucketized_weights_ref =
        new float[weights_size]{1.0, 2.0, 4.0, 6.0, 3.0, 5.0};
    bucketized_pos_ref = new long[indices_size]{0, 1, 0, 2, 0, 1};
  }

  static void TearDownTestCase() {
    delete[] lengths;
    delete[] indices;
    delete[] weights;
    delete[] bucketized_lengths_ref;
    delete[] bucketized_indices_ref;
    delete[] bucketized_weights_ref;
    delete[] bucketized_pos_ref;
  }
};
long* FBGEMMGPUBucketizeSparseFeaturesTest::lengths;
long* FBGEMMGPUBucketizeSparseFeaturesTest::indices;
float* FBGEMMGPUBucketizeSparseFeaturesTest::weights;
long* FBGEMMGPUBucketizeSparseFeaturesTest::bucketized_lengths_ref;
long* FBGEMMGPUBucketizeSparseFeaturesTest::bucketized_indices_ref;
float* FBGEMMGPUBucketizeSparseFeaturesTest::bucketized_weights_ref;
long* FBGEMMGPUBucketizeSparseFeaturesTest::bucketized_pos_ref;

TEST_F(FBGEMMGPUBucketizeSparseFeaturesTest, bucketize_sparse_features_test) {
  int device_cnt;
  cudaGetDeviceCount(&device_cnt);
  if (device_cnt == 0) {
    GTEST_SKIP();
  }
  long* lengths_ptr_gpu;
  long* indices_ptr_gpu;
  float* weights_ptr_gpu;

  long* bucketized_lengths_ptr_gpu;
  long* bucketized_indices_ptr_gpu;
  float* bucketized_weights_ptr_gpu;
  long* bucketized_pos_ptr_gpu;

  long* bucketized_lengths_ptr_cpu = new long[8];
  long* bucketized_indices_ptr_cpu = new long[indices_size];
  long* bucketized_pos_ptr_cpu = new long[indices_size];
  float* bucketized_weights_ptr_cpu = new float[weights_size];
  CUDA_CHECK(cudaMalloc((void**)&lengths_ptr_gpu, lengths_size * sizeof(long)));
  CUDA_CHECK(cudaMalloc((void**)&indices_ptr_gpu, indices_size * sizeof(long)));
  CUDA_CHECK(
      cudaMalloc((void**)&weights_ptr_gpu, weights_size * sizeof(float)));

  CUDA_CHECK(cudaMalloc(
      (void**)&bucketized_lengths_ptr_gpu,
      my_size * lengths_size * sizeof(long)));
  CUDA_CHECK(cudaMalloc(
      (void**)&bucketized_indices_ptr_gpu, indices_size * sizeof(long)));
  CUDA_CHECK(cudaMalloc(
      (void**)&bucketized_weights_ptr_gpu, weights_size * sizeof(float)));
  CUDA_CHECK(
      cudaMalloc((void**)&bucketized_pos_ptr_gpu, indices_size * sizeof(long)));

  CUDA_CHECK(cudaMemcpy(
      lengths_ptr_gpu,
      lengths,
      lengths_size * sizeof(long),
      cudaMemcpyKind::cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(
      indices_ptr_gpu,
      indices,
      indices_size * sizeof(long),
      cudaMemcpyKind::cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(
      weights_ptr_gpu,
      weights,
      weights_size * sizeof(float),
      cudaMemcpyKind::cudaMemcpyHostToDevice));

  fbgemm_gpu_test::bucketize_sparse_features(
      lengths_size,
      my_size,
      lengths_ptr_gpu,
      indices_ptr_gpu,
      weights_ptr_gpu,
      bucketized_lengths_ptr_gpu,
      bucketized_indices_ptr_gpu,
      bucketized_weights_ptr_gpu,
      bucketized_pos_ptr_gpu);

  CUDA_CHECK(cudaMemcpy(
      bucketized_lengths_ptr_cpu,
      bucketized_lengths_ptr_gpu,
      my_size * lengths_size * sizeof(long),
      cudaMemcpyKind::cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(
      bucketized_indices_ptr_cpu,
      bucketized_indices_ptr_gpu,
      indices_size * sizeof(long),
      cudaMemcpyKind::cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(
      bucketized_pos_ptr_cpu,
      bucketized_pos_ptr_gpu,
      indices_size * sizeof(long),
      cudaMemcpyKind::cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(
      bucketized_weights_ptr_cpu,
      bucketized_weights_ptr_gpu,
      weights_size * sizeof(float),
      cudaMemcpyKind::cudaMemcpyDeviceToHost));

  for (int i = 0; i < my_size * lengths_size; i++)
    ASSERT_EQ(bucketized_lengths_ptr_cpu[i], bucketized_lengths_ref[i]);
  for (int i = 0; i < indices_size; i++) {
    ASSERT_EQ(bucketized_indices_ptr_cpu[i], bucketized_indices_ref[i]);
    ASSERT_EQ(bucketized_pos_ptr_cpu[i], bucketized_pos_ref[i]);
    ASSERT_EQ(bucketized_weights_ptr_cpu[i], bucketized_weights_ref[i]);
  }

  cudaFree(bucketized_lengths_ptr_gpu);
  cudaFree(bucketized_indices_ptr_gpu);
  cudaFree(bucketized_weights_ptr_gpu);
  cudaFree(bucketized_pos_ptr_gpu);
  cudaFree(lengths_ptr_gpu);
  cudaFree(indices_ptr_gpu);
  cudaFree(weights_ptr_gpu);
  delete[] bucketized_lengths_ptr_cpu;
  delete[] bucketized_indices_ptr_cpu;
  delete[] bucketized_pos_ptr_cpu;
  delete[] bucketized_weights_ptr_cpu;
}
