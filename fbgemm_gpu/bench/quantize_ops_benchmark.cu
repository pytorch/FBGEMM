/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <cuda.h>
#include <iostream>
#include <random>
#include <vector>

#include "fbgemm_gpu/bench_utils.cuh"
#include "fbgemm_gpu/cuda_utils.cuh"
#include "fbgemm_gpu/quantize_wrappers.cuh"

int main(int argc, char* argv[]) {
  std::vector<int> nrows = {10, 20, 100, 256, 512, 5120};
  std::vector<int> ncols = {128, 256, 512, 1024};
  std::vector<int> bit_rates = {2, 4, 8};
  int iters = 100;
  // gpu ptrs
  float* float_input_ptr;
  uint8_t* quantize_ptr;
  float* dequantize_ptr;

  for (auto const& nrow : nrows) {
    for (auto const& ncol : ncols) {
      for (auto const& bit_rate : bit_rates) {
        CUDA_CHECK(cudaMalloc(&float_input_ptr, nrow * ncol * sizeof(float)));
        generate_random_table(float_input_ptr, nrow * ncol);
        CUDA_CHECK(cudaDeviceSynchronize());
        int output_columns;
        float quantize_time;
        float dequantize_time;
        if (bit_rate == 8) {
          int ncols_aligned = (ncol + 4 - 1) / 4 * 4;
          output_columns = ncols_aligned - 2 * sizeof(float);
        } else {
          int num_elem_per_byte = 8 / bit_rate;
          output_columns =
              (ncol + num_elem_per_byte - 1) / num_elem_per_byte + 2 * 2;
        }
        CUDA_CHECK(
            cudaMalloc(&quantize_ptr, nrow * output_columns * sizeof(char)));
        CUDA_CHECK(cudaMalloc(&dequantize_ptr, nrow * ncol * sizeof(float)));

        if (bit_rate != 8) {
          quantize_time = benchmark_function(iters, [&]() {
            fbgemm_gpu_test::FloatToFusedNBitRowwiseQuantizedSBHalf(
                nrow, ncol, bit_rate, float_input_ptr, quantize_ptr);
          });

          dequantize_time = benchmark_function(iters, [&]() {
            fbgemm_gpu_test::FusedNBitRowwiseQuantizedSBHalfToFloat(
                nrow, output_columns, bit_rate, quantize_ptr, dequantize_ptr);
          });
        } else {
          quantize_time = benchmark_function(iters, [&]() {
            fbgemm_gpu_test::FloatToFused8BitRowwiseQuantized(
                nrow, ncol, float_input_ptr, quantize_ptr);
          });

          dequantize_time = benchmark_function(iters, [&]() {
            fbgemm_gpu_test::Fused8BitRowwiseQuantizedToFloat(
                nrow, output_columns, quantize_ptr, dequantize_ptr);
          });
        }
        std::cout << "nrow: " << nrow << " ncol: " << ncol
                  << " bit rate: " << bit_rate
                  << " quantize time per iter: " << quantize_time << " ms,"
                  << " dequantize time per iter: " << dequantize_time << " ms"
                  << std::endl;
        CUDA_CHECK(cudaFree(float_input_ptr));
        CUDA_CHECK(cudaFree(quantize_ptr));
        CUDA_CHECK(cudaFree(dequantize_ptr));
        CUDA_CHECK(cudaDeviceSynchronize());
      }
    }
  }
}
