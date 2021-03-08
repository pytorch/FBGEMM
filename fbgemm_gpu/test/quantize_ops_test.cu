/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <gtest/gtest.h>
#include "fbgemm_gpu/cuda_utils.cuh"
#include "fbgemm_gpu/quantize_wrappers.cuh"

class FBGEMMGPUQuantizeOpsTest : public ::testing::Test {
 protected:
  static float* input;
  static uint8_t* quantize_fuse8bit_output_ref;
  static float* dequantize_fuse8bit_output_ref;
  static uint8_t* quantize_fusenbit_output_ref;
  static float* dequantize_fusenbit_output_ref;
  static constexpr int nrows = 2;
  static constexpr int ncols = 8;
  static constexpr int bit_rate = 4;
  // ncol of quant buffer for 8bit quantization
  static constexpr int output_columns =
      (ncols + 4 - 1) / 4 * 4 + 2 * sizeof(float);

  static void SetUpTestCase() {
    input = new float[nrows * ncols]{
        0.4962565899,
        0.7682217956,
        0.0884774327,
        0.1320304871,
        0.3074228168,
        0.6340786815,
        0.4900934100,
        0.8964447379,
        0.4556279778,
        0.6323062778,
        0.3488934636,
        0.4017173052,
        0.0223257542,
        0.1688589454,
        0.2938884497,
        0.5185217857};
    quantize_fuse8bit_output_ref = new uint8_t[nrows * output_columns]{
        129, 215, 0,   14,  69, 172, 127, 255, 153, 166, 79,
        59,  168, 51,  181, 61, 181, 255, 137, 159, 0,   61,
        114, 207, 115, 196, 28, 59,  128, 228, 182, 60};
    dequantize_fuse8bit_output_ref = new float[nrows * ncols]{
        0.4972138405,
        0.7697047591,
        0.0884774327,
        0.1328364164,
        0.3071038723,
        0.6334593296,
        0.4908768535,
        0.8964447379,
        0.4552923143,
        0.6323062778,
        0.3500407636,
        0.4026665390,
        0.0223257542,
        0.1682426631,
        0.2950229049,
        0.5174863935};
    quantize_fusenbit_output_ref = new uint8_t[nrows * ncols]{
        216,
        16,
        164,
        247,
        229,
        42,
        170,
        45,
        251,
        152,
        64,
        199,
        53,
        41,
        183,
        37};
    dequantize_fusenbit_output_ref = new float[nrows * ncols]{
        0.5194091797,
        0.7887268066,
        0.0885009766,
        0.1423645020,
        0.3039550781,
        0.6271362305,
        0.4655456543,
        0.8964538574,
        0.4698028564,
        0.6325225830,
        0.3477630615,
        0.3884429932,
        0.0223236084,
        0.1850433350,
        0.3070831299,
        0.5104827881};
  }

  static void TearDownTestCase() {
    delete[] input;
    delete[] quantize_fuse8bit_output_ref;
    delete[] dequantize_fuse8bit_output_ref;
    delete[] quantize_fusenbit_output_ref;
    delete[] dequantize_fusenbit_output_ref;
  }
};
float* FBGEMMGPUQuantizeOpsTest::input;
uint8_t* FBGEMMGPUQuantizeOpsTest::quantize_fuse8bit_output_ref;
float* FBGEMMGPUQuantizeOpsTest::dequantize_fuse8bit_output_ref;
uint8_t* FBGEMMGPUQuantizeOpsTest::quantize_fusenbit_output_ref;
float* FBGEMMGPUQuantizeOpsTest::dequantize_fusenbit_output_ref;

TEST_F(FBGEMMGPUQuantizeOpsTest, quantize_dequantize_8bit_test) {
  int device_cnt;
  cudaGetDeviceCount(&device_cnt);
  if (device_cnt == 0) {
    GTEST_SKIP();
  }
  // gpu ptrs
  float* input_gpu_ptr;
  uint8_t* quantize_fuse8bit_output_gpu_ptr;
  float* dequantize_fuse8bit_output_gpu_ptr;

  // cpu ptrs
  uint8_t* quantize_fuse8bit_output_cpu_ptr =
      new uint8_t[nrows * output_columns];
  float* dequantize_fuse8bit_output_cpu_ptr = new float[nrows * ncols];
  CUDA_CHECK(
      cudaMalloc((void**)&input_gpu_ptr, (nrows * ncols) * sizeof(float)));
  CUDA_CHECK(cudaMalloc(
      (void**)&quantize_fuse8bit_output_gpu_ptr,
      nrows * output_columns * sizeof(uint8_t)));
  CUDA_CHECK(cudaMalloc(
      (void**)&dequantize_fuse8bit_output_gpu_ptr,
      (nrows * ncols) * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(
      input_gpu_ptr,
      input,
      (nrows * ncols) * sizeof(float),
      cudaMemcpyKind::cudaMemcpyHostToDevice));

  fbgemm_gpu_test::FloatToFused8BitRowwiseQuantized(
      nrows, ncols, input_gpu_ptr, quantize_fuse8bit_output_gpu_ptr);
  CUDA_CHECK(cudaMemcpy(
      quantize_fuse8bit_output_cpu_ptr,
      quantize_fuse8bit_output_gpu_ptr,
      nrows * output_columns * sizeof(uint8_t),
      cudaMemcpyKind::cudaMemcpyDeviceToHost));
  for (int i = 0; i < nrows * output_columns; i++) {
    ASSERT_EQ(
        quantize_fuse8bit_output_cpu_ptr[i], quantize_fuse8bit_output_ref[i]);
  }
  fbgemm_gpu_test::Fused8BitRowwiseQuantizedToFloat(
      nrows,
      output_columns,
      quantize_fuse8bit_output_gpu_ptr,
      dequantize_fuse8bit_output_gpu_ptr);
  CUDA_CHECK(cudaMemcpy(
      dequantize_fuse8bit_output_cpu_ptr,
      dequantize_fuse8bit_output_gpu_ptr,
      nrows * ncols * sizeof(float),
      cudaMemcpyKind::cudaMemcpyDeviceToHost));
  for (int i = 0; i < nrows * ncols; i++) {
    ASSERT_FLOAT_EQ(
        dequantize_fuse8bit_output_cpu_ptr[i],
        dequantize_fuse8bit_output_ref[i]);
  }
  cudaFree(input_gpu_ptr);
  cudaFree(quantize_fuse8bit_output_gpu_ptr);
  cudaFree(dequantize_fuse8bit_output_gpu_ptr);
  delete[] quantize_fuse8bit_output_cpu_ptr;
  delete[] dequantize_fuse8bit_output_cpu_ptr;
}

TEST_F(FBGEMMGPUQuantizeOpsTest, quantize_dequantize_nbit_test) {
  int device_cnt;
  cudaGetDeviceCount(&device_cnt);
  if (device_cnt == 0) {
    GTEST_SKIP();
  }
  // gpu ptrs
  float* input_gpu_ptr;
  uint8_t* quantize_fusenbit_output_gpu_ptr;
  float* dequantize_fusenbit_output_gpu_ptr;

  // cpu ptrs
  uint8_t* quantize_fusenbit_output_cpu_ptr = new uint8_t[nrows * ncols];
  float* dequantize_fusenbit_output_cpu_ptr = new float[nrows * ncols];
  CUDA_CHECK(
      cudaMalloc((void**)&input_gpu_ptr, (nrows * ncols) * sizeof(float)));
  CUDA_CHECK(cudaMalloc(
      (void**)&quantize_fusenbit_output_gpu_ptr,
      nrows * ncols * sizeof(uint8_t)));
  CUDA_CHECK(cudaMalloc(
      (void**)&dequantize_fusenbit_output_gpu_ptr,
      (nrows * ncols) * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(
      input_gpu_ptr,
      input,
      (nrows * ncols) * sizeof(float),
      cudaMemcpyKind::cudaMemcpyHostToDevice));

  fbgemm_gpu_test::FloatToFusedNBitRowwiseQuantizedSBHalf(
      nrows, ncols, bit_rate, input_gpu_ptr, quantize_fusenbit_output_gpu_ptr);
  CUDA_CHECK(cudaMemcpy(
      quantize_fusenbit_output_cpu_ptr,
      quantize_fusenbit_output_gpu_ptr,
      nrows * ncols * sizeof(uint8_t),
      cudaMemcpyKind::cudaMemcpyDeviceToHost));
  for (int i = 0; i < nrows * ncols; i++) {
    ASSERT_FLOAT_EQ(
        quantize_fusenbit_output_cpu_ptr[i], quantize_fusenbit_output_ref[i]);
  }
  fbgemm_gpu_test::FusedNBitRowwiseQuantizedSBHalfToFloat(
      nrows,
      ncols,
      bit_rate,
      quantize_fusenbit_output_gpu_ptr,
      dequantize_fusenbit_output_gpu_ptr);
  CUDA_CHECK(cudaMemcpy(
      dequantize_fusenbit_output_cpu_ptr,
      dequantize_fusenbit_output_gpu_ptr,
      nrows * ncols * sizeof(float),
      cudaMemcpyKind::cudaMemcpyDeviceToHost));
  for (int i = 0; i < nrows * ncols; i++) {
    ASSERT_NEAR(
        dequantize_fusenbit_output_cpu_ptr[i],
        dequantize_fusenbit_output_ref[i],
        0.0001);
  }
  cudaFree(input_gpu_ptr);
  cudaFree(quantize_fusenbit_output_gpu_ptr);
  cudaFree(dequantize_fusenbit_output_gpu_ptr);
  delete[] quantize_fusenbit_output_cpu_ptr;
  delete[] dequantize_fusenbit_output_cpu_ptr;
}
