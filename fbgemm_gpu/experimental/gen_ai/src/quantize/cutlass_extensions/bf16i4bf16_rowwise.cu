/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cutlass/util/device_memory.h>
#include <cutlass/util/packed_stride.hpp>

// clang-format off
// The fixed ordering of the headers is required for CUTLASS 3.2+
#include <cute/tensor.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>     // @manual
#include <cutlass/gemm/device/gemm_universal_adapter.h>       // @manual
#include <cutlass/epilogue/collective/collective_builder.hpp> // @manual
// clang-format on

#include "cutlass_extensions/include/kernel_mode.h"

namespace fbgemm_gpu {

#if CUDART_VERSION >= 12000

template <
    int TB_M,
    int TB_N,
    int TB_K,
    int TBS_M,
    int TBS_N,
    int TBS_K,
    bool PONG,
    typename WEIGHT_SCALE_DTYPE>
at::Tensor bf16i4bf16_rowwise_impl(
    at::Tensor X, // BF16
    at::Tensor WQ, // INT4
    at::Tensor w_scale,
    at::Tensor w_zp) {
  int M = X.size(0);
  int N = WQ.size(0);
  int K = X.size(1);

  int num_groups = w_scale.size(0);

  TORCH_CHECK(X.is_cuda() && X.is_contiguous());
  TORCH_CHECK(WQ.is_cuda() && WQ.is_contiguous());
  TORCH_CHECK(w_scale.is_cuda() && w_scale.is_contiguous());
  TORCH_CHECK(w_zp.is_cuda() && w_zp.is_contiguous());
  TORCH_CHECK(K >= num_groups && K % num_groups == 0);

  int group_size = K / num_groups;

  auto Y = at::empty({M, N}, X.options().dtype(at::kBFloat16));

  using ElementInputA = cutlass::bfloat16_t;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  constexpr int AlignmentInputA =
      128 /
      cutlass::sizeof_bits<
          ElementInputA>::value; // Memory access granularity/alignment of A
                                 // matrix in units of elements (up to 16 bytes)

  using ElementInputB = cutlass::int4b_t;
  using LayoutInputB = cutlass::layout::RowMajor;
  constexpr int AlignmentInputB =
      128 /
      cutlass::sizeof_bits<
          ElementInputB>::value; // Memory access granularity/alignment of B
                                 // matrix in units of elements (up to 16 bytes)

  using ElementScale = WEIGHT_SCALE_DTYPE;
  using ElementZeroPoint = WEIGHT_SCALE_DTYPE;
  using ElementComputeEpilogue = float;
  using ElementAccumulator = float;

  using ElementOutput = cutlass::bfloat16_t;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  constexpr int AlignmentOutput =
      128 /
      cutlass::sizeof_bits<
          ElementOutput>::value; // Memory access granularity/alignment of C
                                 // matrix in units of elements (up to 16 bytes)

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
  using CooperativeSchedule =
      cutlass::gemm::KernelTmaWarpSpecializedCooperative;
  using PongSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
  using CooperativeEpilogueSchedule =
      cutlass::epilogue::TmaWarpSpecializedCooperative;
  using PongEpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized;
  using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;
  using MainLoopSchedule =
      cute::conditional_t<PONG, PongSchedule, CooperativeSchedule>;
  using EpilogueSchedule = cute::
      conditional_t<PONG, PongEpilogueSchedule, CooperativeEpilogueSchedule>;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm90,
          cutlass::arch::OpClassTensorOp,
          TileShape,
          ClusterShape,
          EpilogueTileType,
          ElementAccumulator,
          ElementAccumulator,
          ElementOutput,
          LayoutOutput,
          AlignmentOutput,
          ElementOutput,
          LayoutOutput,
          AlignmentOutput,
          EpilogueSchedule>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag,
          OperatorClass,
          cute::tuple<ElementInputB, ElementScale, ElementZeroPoint>,
          LayoutInputB,
          AlignmentInputB,
          ElementInputA,
          LayoutInputA,
          AlignmentInputA,
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
  using StrideOutput = typename Gemm::GemmKernel::StrideC;
  using StrideS = typename CollectiveMainloop::StrideScale;

  StrideInputA stride_a = cutlass::make_cute_packed_stride(
      StrideInputA{}, cute::make_shape(M, K, 1));
  StrideInputB stride_b = cutlass::make_cute_packed_stride(
      StrideInputB{}, cute::make_shape(N, K, 1));
  StrideOutput stride_output = cutlass::make_cute_packed_stride(
      StrideOutput{}, cute::make_shape(N, M, 1));
  StrideS stride_S = cutlass::make_cute_packed_stride(
      StrideS{}, cute::make_shape(N, num_groups, 1));

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {N, M, K},
      {reinterpret_cast<ElementInputB*>(WQ.data_ptr()),
       stride_b,
       reinterpret_cast<ElementInputA*>(X.data_ptr()),
       stride_a,
       reinterpret_cast<ElementScale*>(w_scale.data_ptr()),
       stride_S,
       group_size,
       reinterpret_cast<ElementZeroPoint*>(w_zp.data_ptr())},
      {{1.0, 0.0},
       (ElementOutput*)Y.data_ptr<at::BFloat16>(),
       stride_output,
       (ElementOutput*)Y.data_ptr<at::BFloat16>(),
       stride_output}};

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

template <typename WEIGHT_SCALE_DTYPE>
at::Tensor dispatch_bf16i4bf16_rowwise_kernel(
    at::Tensor X, // BF16
    at::Tensor WQ, // INT4
    at::Tensor w_scale,
    at::Tensor w_zp) {
  KernelMode kernel = get_kernel_mode(X, WQ);
  if (kernel == KernelMode::Small) {
    return bf16i4bf16_rowwise_impl<
        64,
        128,
        128,
        2,
        1,
        1,
        true,
        WEIGHT_SCALE_DTYPE>(X, WQ, w_scale, w_zp);
  } else if (kernel == KernelMode::Large) {
    return bf16i4bf16_rowwise_impl<
        128,
        256,
        64,
        2,
        1,
        1,
        false,
        WEIGHT_SCALE_DTYPE>(X, WQ, w_scale, w_zp);
  } else {
    return bf16i4bf16_rowwise_impl<
        128,
        256,
        64,
        2,
        1,
        1,
        false,
        WEIGHT_SCALE_DTYPE>(X, WQ, w_scale, w_zp);
  }
}

at::Tensor bf16i4bf16_rowwise(
    at::Tensor X, // BF16
    at::Tensor WQ, // INT4
    at::Tensor w_scale,
    at::Tensor w_zp) {
  // Check datatypes.
  TORCH_CHECK(
      (w_scale.dtype() == at::kFloat && w_zp.dtype() == at::kFloat) ||
          (w_scale.dtype() == at::kHalf && w_zp.dtype() == at::kHalf) ||
          (w_scale.dtype() == at::kBFloat16 && w_zp.dtype() == at::kBFloat16),
      "Weight scale and zero point tensors must be float32, bfloat16, or float16, and dtype of weight scale and zero point tensors must be the same .");

  if (w_scale.dtype() == at::kFloat) {
    return dispatch_bf16i4bf16_rowwise_kernel<float>(X, WQ, w_scale, w_zp);
  } else if (w_scale.dtype() == at::kHalf) {
    return dispatch_bf16i4bf16_rowwise_kernel<cutlass::half_t>(
        X, WQ, w_scale, w_zp);
  } else if (w_scale.dtype() == at::kBFloat16) {
    return dispatch_bf16i4bf16_rowwise_kernel<cutlass::bfloat16_t>(
        X, WQ, w_scale, w_zp);
  } else {
    throw std::runtime_error(
        "Weight scale and zero point data type not supported in bf16i4bf16_rowwise");
  }
}

#else

at::Tensor bf16i4bf16_rowwise(
    at::Tensor X, // BF16
    at::Tensor WQ, // INT4
    at::Tensor w_scale,
    at::Tensor w_zp) {
  throw std::runtime_error(
      "CUDA version is older than 12.0"); // requires CUDA>=12
}

#endif

} // namespace fbgemm_gpu
