/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/packed_stride.hpp>

// clang-format off
// The fixed ordering of the headers is required for CUTLASS 3.2+
#include <cute/tensor.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>     // @manual
#include <cutlass/gemm/device/gemm_universal_adapter.h>       // @manual
#include <cutlass/epilogue/collective/collective_builder.hpp> // @manual
// clang-format on

#include "cutlass_extensions/include/fp8_blockwise_cutlass_helpers.h"
#include "cutlass_extensions/include/kernel_mode.h"

namespace {

int64_t ceil_div(int64_t a, int64_t b) {
  return (a + b - 1) / b;
}

} // namespace

namespace fbgemm_gpu {

#if CUDART_VERSION >= 12000

// Cutlass blockwise kernel
template <
    int TB_M,
    int TB_N,
    int TB_K,
    int TBS_M,
    int TBS_N,
    int TBS_K>
at::Tensor f8f8bf16_blockwise_impl(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    at::Tensor x_scale,
    at::Tensor w_scale,
    int64_t block_m,
    int64_t block_n,
    int64_t block_k) {
  // XQ: M x K
  // WQ: N x K
  // output: M x N
  int M = size_to_dim_(XQ.dim() - 1, XQ.sizes());
  int N = WQ.size(0);
  int K = WQ.size(1);
  // 1. If the input tensor is {M, K}, the output tensor is {M, N}.
  // 2. If the input tensor is {b, M, K}, the output tensor is {b, M, N}.
  auto out_sizes = XQ.sizes().vec();
  out_sizes.back() = N;
  // Handle case where input shapes are empty.
  if (M == 0 || N == 0 || K == 0) {
    // Return a zero tensor in case K is 0.
    return at::zeros(out_sizes, XQ.options().dtype(at::kBFloat16));
  }

  TORCH_CHECK(WQ.size(1) == K);
  TORCH_CHECK(XQ.stride(-1) == 1);
  TORCH_CHECK(WQ.stride(0) == K);
  TORCH_CHECK(WQ.stride(1) == 1);

  TORCH_CHECK(block_m % TB_N == 0);
  TORCH_CHECK(block_n % TB_M == 0);
  TORCH_CHECK(block_k % TB_K == 0);

  TORCH_CHECK(x_scale.dim() == 2);
  TORCH_CHECK(w_scale.dim() == 2);
  TORCH_CHECK(x_scale.size(0) == ceil_div(M, block_m));
  TORCH_CHECK(x_scale.size(1) == ceil_div(K, block_k));
  TORCH_CHECK(w_scale.size(0) == ceil_div(N, block_n));
  TORCH_CHECK(w_scale.size(1) == ceil_div(K, block_k));
  TORCH_CHECK(x_scale.stride(0) == ceil_div(K, block_k));
  TORCH_CHECK(x_scale.stride(1) == 1);
  TORCH_CHECK(w_scale.stride(0) == ceil_div(K, block_k));
  TORCH_CHECK(w_scale.stride(1) == 1);

  TORCH_CHECK(XQ.dtype() == at::kFloat8_e4m3fn);
  TORCH_CHECK(WQ.dtype() == at::kFloat8_e4m3fn);
  TORCH_CHECK(XQ.is_cuda());
  TORCH_CHECK(WQ.is_cuda());
  TORCH_CHECK(XQ.device().index() == WQ.device().index());
  TORCH_CHECK(x_scale.dtype() == at::kFloat);
  TORCH_CHECK(w_scale.dtype() == at::kFloat);
  TORCH_CHECK(x_scale.is_cuda());
  TORCH_CHECK(w_scale.is_cuda());
  TORCH_CHECK(x_scale.device().index() == XQ.device().index());
  TORCH_CHECK(w_scale.device().index() == XQ.device().index());

  auto Y = at::empty(out_sizes, XQ.options().dtype(at::kBFloat16));

  using ElementInputA = cutlass::float_e4m3_t;
  using LayoutInputA = cutlass::layout::RowMajor;
  constexpr int AlignmentInputA = 16 / sizeof(ElementInputA);

  using ElementInputB = cutlass::float_e4m3_t;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  constexpr int AlignmentInputB = 16 / sizeof(ElementInputB);

  using ElementOutput = cutlass::bfloat16_t;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  constexpr int AlignmentOutput = 16 / sizeof(ElementOutput);

  using ElementAccumulator = float;
  using ElementComputeEpilogue = float;
  using ArchTag = cutlass::arch::Sm90; // Tag indicating the minimum SM that
                                       // supports the intended feature
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using TileShape = cute::Shape<
      cute::Int<TB_M>,
      cute::Int<TB_N>,
      cute::Int<TB_K>>; // Threadblock-level
                        // tile size
  using ClusterShape = cute::Shape<
      cute::Int<TBS_M>,
      cute::Int<TBS_N>,
      cute::Int<TBS_K>>; // Shape of the
                         // threadblocks in a
                         // cluster

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag,
          OperatorClass,
          TileShape,
          ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAccumulator,
          ElementComputeEpilogue,
          ElementOutput,
          LayoutOutput,
          AlignmentOutput,
          ElementOutput,
          LayoutOutput,
          AlignmentOutput,
          cutlass::epilogue::TmaWarpSpecializedCooperative>::CollectiveOp;

  using MainLoopSchedule =
      cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8BlockScaling;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag,
          OperatorClass,
          ElementInputA,
          LayoutInputA,
          AlignmentInputA,
          ElementInputB,
          LayoutInputB,
          AlignmentInputB,
          ElementAccumulator,
          TileShape,
          ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          MainLoopSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideInputA = typename Gemm::GemmKernel::StrideA;
  using StrideInputB = typename Gemm::GemmKernel::StrideB;
  using StrideOutput = typename Gemm::GemmKernel::StrideD;

  StrideInputA stride_a = cutlass::make_cute_packed_stride(
      StrideInputA{}, cute::make_shape(N, K, 1));
  StrideInputB stride_b = cutlass::make_cute_packed_stride(
      StrideInputB{}, cute::make_shape(M, K, 1));
  StrideOutput stride_output = cutlass::make_cute_packed_stride(
      StrideOutput{}, cute::make_shape(N, M, 1));

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {N, M, K},
      {reinterpret_cast<cutlass::float_e4m3_t*>(WQ.data_ptr()),
       stride_a,
       reinterpret_cast<cutlass::float_e4m3_t*>(XQ.data_ptr()),
       stride_b,
       w_scale.data_ptr<float>(),
       x_scale.data_ptr<float>(),
       static_cast<uint8_t>(block_n / TB_M),
       static_cast<uint8_t>(block_m / TB_N),
       static_cast<uint8_t>(block_k / TB_K)},
      {{},
       (cutlass::bfloat16_t*)Y.data_ptr<at::BFloat16>(),
       stride_output,
       (cutlass::bfloat16_t*)Y.data_ptr<at::BFloat16>(),
       stride_output},
  };

  Gemm gemm;

  // Using the arguments, query for extra workspace required for matrix
  // multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Check the problem size is supported or not
  cutlass::Status status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot implement");
  }

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm.initialize(arguments, workspace.get());
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot initialize");
  }

  status = gemm(at::cuda::getCurrentCUDAStream());
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error(
        std::string("cutlass cannot run") +
        cutlass::cutlassGetStatusString(status));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return Y;
}

// FP8 blockwise Cutlass kernel dispatch.
at::Tensor dispatch_fp8_blockwise_kernel(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    int64_t block_m,
    int64_t block_n,
    int64_t block_k) {
  KernelMode kernel = get_kernel_mode(XQ, WQ);
  if (kernel == KernelMode::Small) {
    return f8f8bf16_blockwise_impl<128, 128, 128, 2, 1, 1>(
        XQ, WQ, x_scale, w_scale, block_m, block_n, block_k);
  } else if (kernel == KernelMode::Large) {
    return f8f8bf16_blockwise_impl<128, 128, 128, 2, 1, 1>(
        XQ, WQ, x_scale, w_scale, block_m, block_n, block_k);
  } else {
    return f8f8bf16_blockwise_impl<128, 128, 128, 1, 2, 1>(
        XQ, WQ, x_scale, w_scale, block_m, block_n, block_k);
  }
}

at::Tensor f8f8bf16_blockwise(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    at::Tensor x_scale, // FP32
    at::Tensor w_scale, // FP32
    int64_t block_m = 256,
    int64_t block_n = 256,
    int64_t block_k = 256) {
  // Check datatypes.
  TORCH_CHECK(
      x_scale.dtype() == at::kFloat && w_scale.dtype() == at::kFloat,
      "Scale tensors must be float32.");

  return dispatch_fp8_blockwise_kernel(
      XQ, WQ, x_scale, w_scale, block_m, block_n, block_k);
}

#else

at::Tensor f8f8bf16_blockwise(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    at::Tensor x_scale,
    at::Tensor w_scale,
    int64_t block_m = 256,
    int64_t block_n = 256,
    int64_t block_k = 256) {
  throw std::runtime_error(
      "CUDA version is older than 12.0"); // requires CUDA>=12
}

#endif

} // namespace fbgemm_gpu
