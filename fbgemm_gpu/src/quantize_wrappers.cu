/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// Copyright 2004-present Facebook. All Rights Reserved.
#include <cuda.h>
#include <cassert>
#include "fbgemm_gpu/cuda_utils.cuh"
#include "fbgemm_gpu/quantize_ops.cuh"
#include "fbgemm_gpu/quantize_wrappers.cuh"

void fbgemm_gpu_test::FloatToFused8BitRowwiseQuantized(
    const int32_t nrows,
    const int32_t ncols,
    const float* __restrict__ input,
    uint8_t* __restrict__ output) {
  int threads_per_block = 256;
  int num_blocks = (nrows + threads_per_block - 1) / threads_per_block;

  if (nrows <= 20) {
    _float_to_fused8bitrowwise_cuda_kernel<<<num_blocks, threads_per_block>>>(
        input, nrows, ncols, output);
  } else {
    float* range_tensor;
    CUDA_CHECK(cudaMalloc((void**)&range_tensor, (nrows) * sizeof(float)));
    _get_8bit_qparam_cuda_kernel<<<num_blocks, threads_per_block>>>(
        input, nrows, ncols, output, range_tensor);

    int blockDim_x = std::min(ncols, threads_per_block);
    dim3 blockDim(blockDim_x, threads_per_block / blockDim_x);
    int gridDim_x = (ncols + blockDim.x - 1) / blockDim.x;
    int gridDim_y = std::min((nrows + blockDim.y - 1) / blockDim.y, 65535u);
    dim3 gridDim(gridDim_x, gridDim_y);

    _compute_8bit_quantize_cuda_kernel<<<gridDim, blockDim>>>(
        input, range_tensor, nrows, ncols, output);
    CUDA_CHECK(cudaFree(range_tensor));
  }
  CUDA_CHECK(cudaGetLastError());
}

void fbgemm_gpu_test::Fused8BitRowwiseQuantizedToFloat(
    const int32_t nrows,
    const int32_t ncols,
    const uint8_t* __restrict__ input,
    float* __restrict__ output) {
  int ncols_aligned = (ncols + 4 - 1) / 4 * 4;
  int output_columns = ncols_aligned - 2 * sizeof(float);

  int threads_per_block = 256;

  int blockDim_x = std::min(threads_per_block, output_columns);
  dim3 blockDim(blockDim_x, threads_per_block / blockDim_x);

  int gridDim_x = (output_columns + blockDim.x - 1) / blockDim.x;
  int gridDim_y = std::min((nrows + blockDim.y - 1) / blockDim.y, 65535u);
  dim3 gridDim(gridDim_x, gridDim_y);

  _fused8bitrowwise_to_float_cuda_kernel<<<gridDim, blockDim>>>(
      input, nrows, ncols, output);

  CUDA_CHECK(cudaGetLastError());
}

void fbgemm_gpu_test::FloatToFusedNBitRowwiseQuantizedSBHalf(
    const int32_t nrows,
    const int32_t ncols,
    const int32_t bit_rate,
    const float* __restrict__ input,
    uint8_t* __restrict__ output) {
  assert(
      ncols % (2 * (8 / bit_rate)) == 0 &&
      "ncols needs to be multiple of 2 Bytes (half type size) to make the address aligned");

  int threads_per_block = 256;
  int num_blocks = (nrows + threads_per_block - 1) / threads_per_block;
  // think unsigned as we use 0, 255

  _float_to_fusednbitrowwise_cuda_kernel<<<num_blocks, threads_per_block>>>(
      bit_rate, input, nrows, ncols, output);

  CUDA_CHECK(cudaGetLastError());
}

void fbgemm_gpu_test::FusedNBitRowwiseQuantizedSBHalfToFloat(
    const int32_t nrows,
    const int32_t ncols,
    const int32_t bit_rate,
    const uint8_t* __restrict__ input,
    float* __restrict__ output) {
  int num_elem_per_byte = 8 / bit_rate;
  int output_columns = (ncols - 2 * sizeof(__half)) * num_elem_per_byte;

  int threads_per_block = 256;

  int blockDim_x = std::min(output_columns, threads_per_block);
  dim3 blockDim(blockDim_x, threads_per_block / blockDim_x);
  int gridDim_x = (output_columns + blockDim.x - 1) / blockDim.x;
  int gridDim_y = std::min((nrows + blockDim.y - 1) / blockDim.y, 65535u);
  dim3 gridDim(gridDim_x, gridDim_y);

  _fusednbitrowwise_to_float_cuda_kernel<<<gridDim, blockDim>>>(
      bit_rate, input, nrows, ncols, output);

  CUDA_CHECK(cudaGetLastError());
}
