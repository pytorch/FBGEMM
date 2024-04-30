/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <cutlass/gemm/device/gemm.h>
#include <torch/library.h>

namespace fbgemm_gpu::experimental {

at::Tensor sgemm_float_cuda(
    const double alpha_,
    const at::Tensor& TA,
    const at::Tensor& TB,
    const double beta_,
    const at::Tensor& TC) {
  TORCH_CHECK_EQ(TA.dim(), 2);
  TORCH_CHECK_EQ(TB.dim(), 2);
  TORCH_CHECK_EQ(TC.dim(), 2);

  const auto M = static_cast<int>(TA.size(0));
  const auto K = static_cast<int>(TA.size(1));
  const auto N = static_cast<int>(TB.size(1));

  TORCH_CHECK_EQ(TB.size(0), K);
  TORCH_CHECK_EQ(TC.size(0), M);
  TORCH_CHECK_EQ(TC.size(1), N);

  // Compute leading dimensions for each matrix
  const auto lda = K;
  const auto ldb = N;
  const auto ldc = N;

  const auto* A = TA.data_ptr<float>();
  const auto* B = TB.data_ptr<float>();
  const auto* C = TC.data_ptr<float>();

  const auto alpha = static_cast<float>(alpha_);
  const auto beta = static_cast<float>(beta_);

  // Create result tensor
  auto TD = at::zeros({M, N}, TC.options());
  auto* D = TD.data_ptr<float>();

  // PyTorch tensors are stored in row-major format
  using Layout = cutlass::layout::RowMajor;

  // Define type definition for single-precision CUTLASS GEMM with row-major
  // input matrices and 128x128x8 threadblock tile size (chosen by default)
  using CutlassGemm = cutlass::gemm::device::Gemm<
      float, // Data-type of A matrix
      Layout, // Layout of A matrix
      float, // Data-type of B matrix
      Layout, // Layout of B matrix
      float, // Data-type of C matrix
      Layout>; // Layout of C matrix

  // Construct the CUTLASS GEMM arguments object
  CutlassGemm::Arguments args(
      {M, N, K}, // GEMM problem dimensions
      {A, lda}, // Tensor-ref for source matrix A
      {B, ldb}, // Tensor-ref for source matrix B
      {C, ldc}, // Tensor-ref for source matrix C
      {D, ldc}, // Tensor-ref for destination matrix D (may be different memory
                // than source C matrix)
      {alpha, beta}); // Scalars used in the epilogue

  // Create and launch the CUTLASS GEMM kernel
  // D = alpha * A x B + beta * C
  const auto status = CutlassGemm()(args);

  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error(
        std::string("CUTLASS GEMM kernel failed: ") +
        std::string(cudaGetErrorString(cudaErrorUnknown)));
  }

  return TD;
}

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.def(
      "sgemm_float(float alpha, Tensor TA, Tensor TB, float beta, Tensor TC) -> Tensor");
}

TORCH_LIBRARY_IMPL(fbgemm, CUDA, m) {
  m.impl(
      "sgemm_float",
      torch::dispatch(
          c10::DispatchKey::CUDA,
          TORCH_FN(fbgemm_gpu::experimental::sgemm_float_cuda)));
}

} // namespace fbgemm_gpu::experimental
