/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include "cutlass_extensions/include/threadblock.h"

namespace fbgemm_gpu {

template <int TB_M, int TB_N, int TB_K, int W_M, int W_N, int W_K>
at::Tensor i8i8bf16_dynamic_impl(
    at::Tensor XQ, // INT8
    at::Tensor WQ, // INT8
    at::Tensor scale,
    int64_t split_k) {
  auto M = XQ.size(0);
  auto N = WQ.size(0);
  auto K = XQ.size(1);

  TORCH_CHECK(XQ.is_cuda() && XQ.is_contiguous());
  TORCH_CHECK(WQ.is_cuda() && WQ.is_contiguous());

  auto Y = at::empty({M, N}, XQ.options().dtype(at::kBFloat16));

  using ElementOutput = cutlass::bfloat16_t;
  using ElementAccumulator = int32_t;
  using ElementComputeEpilogue = float;
  using ElementInputA = int8_t; // <- data type of elements in input matrix A
  using ElementInputB = int8_t; // <- data type of elements in input matrix B

  // The code section below describes matrix layout of input and output
  // matrices. Column Major for Matrix A, Row Major for Matrix B and Row Major
  // for Matrix C
  using LayoutInputA = cutlass::layout::RowMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::RowMajor;

  using Gemm = cutlass::gemm::device::Gemm<
      int8_t,
      cutlass::layout::RowMajor,
      int8_t,
      cutlass::layout::ColumnMajor,
      ElementOutput,
      cutlass::layout::RowMajor,
      ElementAccumulator,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<TB_M, TB_N, TB_K>, // ThreadBlockShape
      cutlass::gemm::GemmShape<W_M, W_N, W_K>, // WarpShape
      cutlass::gemm::GemmShape<16, 8, 32>, // InstructionShape
      cutlass::epilogue::thread::LinearCombinationOnDevice<
          ElementOutput,
          128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator,
          ElementComputeEpilogue>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
      3,
      16,
      16,
      true>;

  auto input_size = cutlass::MatrixCoord(M, K);
  auto weight_size = cutlass::MatrixCoord(K, N);
  auto output_size = cutlass::MatrixCoord(M, N);

  // constexpr int kSparse = Gemm::kSparse;
  // How many elements of A are covered per ElementE
  // constexpr int kElementsPerElementE = Gemm::kElementsPerElementE;
  // The size of individual meta data
  // constexpr int kMetaSizeInBits = Gemm::kMetaSizeInBits;
  cutlass::gemm::GemmCoord problem_size(M, N, K);

  cutlass::TensorRef<ElementInputA, LayoutInputA> input_ref(
      XQ.data_ptr<ElementInputA>(), LayoutInputA::packed(input_size));
  cutlass::TensorRef<ElementInputB, LayoutInputB> weight_ref(
      WQ.data_ptr<ElementInputB>(), LayoutInputB::packed(weight_size));
  cutlass::TensorRef<ElementOutput, LayoutOutput> out_ref(
      (ElementOutput*)Y.data_ptr<at::BFloat16>(),
      LayoutOutput::packed(output_size));

  typename Gemm::Arguments arguments{
      problem_size,
      input_ref,
      weight_ref,
      out_ref,
      out_ref,
      {scale.data_ptr<float>()},
      int(split_k)};
  Gemm gemm_op;

  // Using the arguments, query for extra workspace required for matrix
  // multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  auto workspace =
      at::empty({int64_t(workspace_size)}, Y.options().dtype(at::kChar));

  // Check the problem size is supported or not
  cutlass::Status status = gemm_op.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot implement");
  }

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm_op.initialize(
      arguments, workspace.data_ptr(), at::cuda::getCurrentCUDAStream());
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot initialize");
  }

  status = gemm_op(at::cuda::getCurrentCUDAStream());
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error(
        std::string("cutlass cannot run") +
        cutlass::cutlassGetStatusString(status));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return Y;
}

at::Tensor i8i8bf16_dynamic(
    at::Tensor XQ, // INT8
    at::Tensor WQ, // INT8
    at::Tensor scale,
    int64_t split_k) {
  auto M = XQ.size(0);
  auto N = WQ.size(0);
  auto K = XQ.size(1);
  if (M <= 128 && N >= K) {
    return i8i8bf16_dynamic_impl<64, 128, 64, 32, 64, 64>(
        XQ, WQ, scale, split_k);
  } else if (M <= 128 && N < K) {
    return i8i8bf16_dynamic_impl<64, 64, 128, 32, 32, 128>(
        XQ, WQ, scale, split_k);
  } else {
    return i8i8bf16_dynamic_impl<256, 128, 64, 64, 64, 64>(
        XQ, WQ, scale, split_k);
  }
}

} // namespace fbgemm_gpu
