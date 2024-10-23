/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAGuard.h>

#include "cublas_utils.h"

namespace fbgemm_gpu {

#if CUDART_VERSION >= 12000

at::Tensor f8f8bf16_cublas(
    at::Tensor A, // FP8
    at::Tensor B, // FP8
    std::optional<at::Tensor> Ainvs = std::nullopt,
    std::optional<at::Tensor> Binvs = std::nullopt,
    bool use_fast_accum = true,
    std::optional<at::Tensor> output = std::nullopt) {
  auto m = A.size(0);
  auto n = B.size(0);
  auto k = A.size(1);
  size_t workspaceSize = CUBLAS_WORKSPACE_SIZE;
  const int8_t fastAccuMode = use_fast_accum ? 1 : 0;

  TORCH_CHECK(A.is_cuda() && A.is_contiguous());
  TORCH_CHECK(B.is_cuda() && B.is_contiguous());

  cublasLtHandle_t ltHandle;
  checkCublasStatus(cublasLtCreate(&ltHandle));
  auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
  auto workspace = allocator.allocate(workspaceSize);
  if (output.has_value()) {
    auto output_tensor = output.value();
    TORCH_CHECK(output_tensor.is_cuda());
    TORCH_CHECK(output_tensor.is_contiguous());
    TORCH_CHECK(
        output_tensor.numel() == m * n,
        "output_tensor.numel=",
        output_tensor.numel(),
        ", m=",
        m,
        ", n=",
        n);
    TORCH_CHECK(output_tensor.options().dtype() == at::kBFloat16);
  }

  const cudaDataType_t A_type = CUDA_R_8F_E4M3;
  const cudaDataType_t B_type = CUDA_R_8F_E4M3;
  const cudaDataType_t D_type = CUDA_R_16BF;

  float one = 1.0;
  float zero = 0.0;

  cublasOperation_t transa = CUBLAS_OP_T;
  cublasOperation_t transb = CUBLAS_OP_N;

  cublasLtMatmulDesc_t operationDesc = nullptr;
  cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Ddesc = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;
  int returnedResults = 0;
  cublasLtMatmulHeuristicResult_t heuristicResult = {};
  cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;

  cublasComputeType_t gemm_compute_type = CUBLAS_COMPUTE_32F;
  // Create matrix descriptors. Not setting any extra attributes.

  auto lda = k;
  auto ldb = k;
  auto ldd = n;
  checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, A_type, k, m, lda));
  checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, B_type, k, n, ldb));
  checkCublasStatus(cublasLtMatrixLayoutCreate(&Ddesc, D_type, n, m, ldd));

  checkCublasStatus(
      cublasLtMatmulDescCreate(&operationDesc, gemm_compute_type, CUDA_R_32F));
  checkCublasStatus(cublasLtMatmulDescSetAttribute(
      operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
  checkCublasStatus(cublasLtMatmulDescSetAttribute(
      operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

  checkCublasStatus(cublasLtMatmulDescSetAttribute(
      operationDesc,
      CUBLASLT_MATMUL_DESC_FAST_ACCUM,
      &fastAccuMode,
      sizeof(fastAccuMode)));

  if (Ainvs.has_value()) {
    const float* Ainvs_pt = Ainvs.value().data_ptr<float>();
    checkCublasStatus(cublasLtMatmulDescSetAttribute(
        operationDesc,
        CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
        &Ainvs_pt,
        sizeof(Ainvs_pt)));
  }

  if (Binvs.has_value()) {
    const float* Binvs_pt = Binvs.value().data_ptr<float>();
    checkCublasStatus(cublasLtMatmulDescSetAttribute(
        operationDesc,
        CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
        &Binvs_pt,
        sizeof(Binvs_pt)));
  }

  checkCublasStatus(cublasLtMatmulDescSetAttribute(
      operationDesc,
      CUBLASLT_MATMUL_DESC_EPILOGUE,
      &epilogue,
      sizeof(epilogue)));

  checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference));

  checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(
      preference,
      CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &workspaceSize,
      sizeof(workspaceSize)));

  checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(
      ltHandle,
      operationDesc,
      Bdesc,
      Adesc,
      Ddesc,
      Ddesc,
      preference,
      1,
      &heuristicResult,
      &returnedResults));

  if (returnedResults == 0)
    throw std::runtime_error("Unable to find any suitable algorithms");

  // D = alpha * (A * B) + beta * C
  // Warmup
  auto Y = output.value_or(at::empty({m, n}, A.options().dtype(at::kBFloat16)));
  checkCublasStatus(cublasLtMatmul(
      ltHandle,
      operationDesc,
      static_cast<const void*>(&one), /* alpha */
      B.data_ptr(), /* B */
      Bdesc,
      A.data_ptr(), /* A */
      Adesc,
      static_cast<const void*>(&zero), /* beta */
      nullptr, /* C */
      Ddesc,
      Y.data_ptr(), /* D */
      Ddesc,
      &heuristicResult.algo, /* algo */
      workspace.mutable_get(), /* workspace */
      workspaceSize,
      at::cuda::getCurrentCUDAStream())); /* stream */
  return Y;
}

#else

at::Tensor f8f8bf16_cublas(
    at::Tensor A, // FP8
    at::Tensor B, // FP8
    std::optional<at::Tensor> Ainvs = std::nullopt,
    std::optional<at::Tensor> Binvs = std::nullopt,
    bool use_fast_accum = true,
    std::optional<at::Tensor> output = std::nullopt) {
  throw std::runtime_error(
      "CUDA version is older than 12.0"); // requires CUDA>=12
}

#endif

} // namespace fbgemm_gpu
