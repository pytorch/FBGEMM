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

namespace fbgemm_gpu {

#if CUDART_VERSION >= 12000

// Cutlass groupwise kernel
template <
    int TB_M,
    int TB_N,
    int TB_K,
    int TBS_M,
    int TBS_N,
    int TBS_K,
    int ARCH,
    bool PONG,
    bool TRANSPOSE,
    int SWIZZLE>
at::Tensor f8f8bf16_groupwise_impl(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    at::Tensor x_scale,
    at::Tensor w_scale) {
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
  // Handle case where there is a zero dimension, we simply return an empty
  // tensor.
  if (M == 0 || N == 0 || K == 0) {
    // Use zeros instead of empty for special case where K=0.
    return at::zeros(out_sizes, XQ.options().dtype(at::kBFloat16));
  }

  TORCH_CHECK(XQ.size(-1) == K);
  TORCH_CHECK(XQ.is_cuda() && XQ.is_contiguous());
  TORCH_CHECK(WQ.is_cuda() && WQ.is_contiguous());

  at::Tensor Y = at::empty(out_sizes, XQ.options().dtype(at::kBFloat16));

  using ElementA = cutlass::float_e4m3_t;
  using LayoutA = cutlass::layout::RowMajor;

  using ElementB = cutlass::float_e4m3_t;
  using LayoutB = cutlass::layout::ColumnMajor;

  // Use implicit transpose to enable more flexible configurations.
  using LayoutA_Transpose =
      typename cutlass::layout::LayoutTranspose<LayoutA>::type;
  using LayoutB_Transpose =
      typename cutlass::layout::LayoutTranspose<LayoutB>::type;

  using ElementOutput = cutlass::bfloat16_t;
  using LayoutOutput = cutlass::layout::RowMajor;
  constexpr int AlignmentOutput = 16 / sizeof(ElementOutput);

  using LayoutOutput_Transpose =
      typename cutlass::layout::LayoutTranspose<LayoutOutput>::type;

  using ElementAccumulator = float;
  using ElementComputeEpilogue = float;
  using ArchTag = cutlass::arch::Sm90;
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
  // Set the size of scale blocks. We could consider making this a template arg.
  constexpr int ScaleGranularityM = 1;
  constexpr int ScaleGranularityN = 128;
  constexpr int ScaleGranularityK = 128;
  using ScaleConfig = cutlass::detail::Sm90BlockwiseScaleConfig<
      ScaleGranularityM,
      ScaleGranularityN,
      ScaleGranularityK>;
  using LayoutSFA =
      decltype(ScaleConfig::deduce_layoutSFA()); // Layout type for SFA matrix
                                                 // operand
  using LayoutSFB =
      decltype(ScaleConfig::deduce_layoutSFB()); // Layout type for SFB matrix
                                                 // operand

  // Set appropriate inputs depending on whether we are doing implicit
  // transpose.
  using ElementLHS = cute::conditional_t<TRANSPOSE, ElementB, ElementA>;
  using LayoutLHS = cute::conditional_t<TRANSPOSE, LayoutB_Transpose, LayoutA>;
  constexpr int AlignmentInputLHS = 16 / sizeof(ElementLHS);
  using ElementRHS = cute::conditional_t<TRANSPOSE, ElementA, ElementB>;
  using LayoutRHS = cute::conditional_t<TRANSPOSE, LayoutA_Transpose, LayoutB>;
  constexpr int AlignmentInputRHS = 16 / sizeof(ElementRHS);
  using LayoutOut =
      cute::conditional_t<TRANSPOSE, LayoutOutput_Transpose, LayoutOutput>;
  using LayoutScaleLHS = cute::conditional_t<TRANSPOSE, LayoutSFB, LayoutSFA>;
  using LayoutScaleRHS = cute::conditional_t<TRANSPOSE, LayoutSFA, LayoutSFB>;
  // Set data pointers.
  ElementLHS* LHS_data;
  ElementRHS* RHS_data;
  ElementComputeEpilogue* LHS_scale_data;
  ElementComputeEpilogue* RHS_scale_data;
  if constexpr (TRANSPOSE) {
    LHS_data = reinterpret_cast<ElementLHS*>(WQ.data_ptr());
    RHS_data = reinterpret_cast<ElementRHS*>(XQ.data_ptr());
    LHS_scale_data =
        reinterpret_cast<ElementComputeEpilogue*>(w_scale.data_ptr());
    RHS_scale_data =
        reinterpret_cast<ElementComputeEpilogue*>(x_scale.data_ptr());
  } else {
    LHS_data = reinterpret_cast<ElementLHS*>(XQ.data_ptr());
    RHS_data = reinterpret_cast<ElementRHS*>(WQ.data_ptr());
    LHS_scale_data =
        reinterpret_cast<ElementComputeEpilogue*>(x_scale.data_ptr());
    RHS_scale_data =
        reinterpret_cast<ElementComputeEpilogue*>(w_scale.data_ptr());
  }

  using KernelSchedule = cute::conditional_t<
      PONG,
      cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8BlockScaledAccum,
      cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8BlockScaledAccum>;

  using EpilogueSchedule = cute::conditional_t<
      PONG,
      cutlass::epilogue::TmaWarpSpecialized,
      cutlass::epilogue::TmaWarpSpecializedCooperative>;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag,
          OperatorClass,
          TileShape,
          ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAccumulator,
          ElementComputeEpilogue,
          void,
          LayoutOut,
          AlignmentOutput,
          ElementOutput,
          LayoutOut,
          AlignmentOutput,
          EpilogueSchedule>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag,
          OperatorClass,
          ElementB,
          cute::tuple<LayoutLHS, LayoutScaleLHS>,
          AlignmentInputLHS,
          ElementA,
          cute::tuple<LayoutRHS, LayoutScaleRHS>,
          AlignmentInputRHS,
          ElementAccumulator,
          TileShape,
          ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          KernelSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideInputA = typename Gemm::GemmKernel::StrideA;
  using StrideInputB = typename Gemm::GemmKernel::StrideB;
  using StrideInputLHS =
      std::conditional_t<TRANSPOSE, StrideInputB, StrideInputA>;
  using StrideInputRHS =
      std::conditional_t<TRANSPOSE, StrideInputA, StrideInputB>;
  using StrideOutput = typename Gemm::GemmKernel::StrideC;

  StrideInputA stride_a = cutlass::make_cute_packed_stride(
      StrideInputA{}, cute::make_shape(M, K, 1));
  StrideInputB stride_b = cutlass::make_cute_packed_stride(
      StrideInputB{}, cute::make_shape(N, K, 1));

  LayoutSFA layout_SFA =
      ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
  LayoutSFB layout_SFB =
      ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));

  StrideInputLHS stride_lhs;
  StrideInputRHS stride_rhs;
  LayoutScaleLHS layout_scale_lhs;
  LayoutScaleRHS layout_scale_rhs;
  StrideOutput stride_output;
  cute::Shape<int, int, int, int> problem_shape;
  if constexpr (TRANSPOSE) {
    problem_shape = {N, M, K, 1};
    stride_lhs = stride_b;
    stride_rhs = stride_a;
    layout_scale_lhs = layout_SFB;
    layout_scale_rhs = layout_SFA;
    stride_output = cutlass::make_cute_packed_stride(
        StrideOutput{}, cute::make_shape(N, M, 1));
  } else {
    problem_shape = {M, N, K, 1};
    stride_lhs = stride_a;
    stride_rhs = stride_b;
    layout_scale_lhs = layout_SFA;
    layout_scale_rhs = layout_SFB;
    stride_output = cutlass::make_cute_packed_stride(
        StrideOutput{}, cute::make_shape(M, N, 1));
  }

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_shape,
      {LHS_data,
       stride_lhs,
       RHS_data,
       stride_rhs,
       LHS_scale_data,
       layout_scale_lhs,
       RHS_scale_data,
       layout_scale_rhs},
      {{}, // Epilogue thread we populate below.
       nullptr,
       stride_output,
       (ElementOutput*)Y.data_ptr<at::BFloat16>(),
       stride_output}};

  // Set tile swizzling limit, helps in compute bound cases.
  if constexpr (SWIZZLE > 0) {
    arguments.scheduler.max_swizzle_size = SWIZZLE;
  }

  Gemm gemm;

  // Using the arguments, query for extra workspace required for matrix
  // multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  at::Tensor workspace =
      at::empty(workspace_size, XQ.options().dtype(at::kByte));

  // Check the problem size is supported or not
  // cutlass::Status status = gemm.can_implement(arguments);
  // if (status != cutlass::Status::kSuccess) {
  //   throw std::runtime_error("cutlass cannot implement");
  // }

  // Initialize CUTLASS kernel with arguments and workspace pointer
  cutlass::Status status = gemm.initialize(arguments, workspace.data_ptr());
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

template <
    int TB_M,
    int TB_N,
    int TB_K,
    int TBS_M,
    int TBS_N,
    int TBS_K,
    int ARCH,
    bool PONG,
    bool TRANSPOSE,
    int SWIZZLE>
at::Tensor f8f8bf16_groupwise_wrapper(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale) {
  // Check datatypes.
  TORCH_CHECK(
      x_scale.dtype() == at::kFloat && w_scale.dtype() == at::kFloat,
      "Scale tensors must be float32.");

  return f8f8bf16_groupwise_impl<
      TB_M,
      TB_N,
      TB_K,
      TBS_M,
      TBS_N,
      TBS_K,
      ARCH,
      PONG,
      TRANSPOSE,
      SWIZZLE>(XQ, WQ, x_scale, w_scale);
}

#else

template <
    int TB_M,
    int TB_N,
    int TB_K,
    int TBS_M,
    int TBS_N,
    int TBS_K,
    bool PONG,
    bool TRANSPOSE,
    int SWIZZLE>
at::Tensor f8f8bf16_groupwise_wrapper(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale) {
  throw std::runtime_error(
      "CUDA version is older than 12.0"); // requires CUDA>=12
}

#endif

} // namespace fbgemm_gpu
