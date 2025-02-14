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
    bool PONG>
at::Tensor bf16i4bf16_rowwise_impl(
    at::Tensor X, // BF16
    at::Tensor WQ, // INT4
    at::Tensor w_scale,
    at::Tensor w_zp) {
  int M = X.size(0);
  int N = WQ.size(0);
  int K = X.size(1);
  int scale_k = w_scale.size(1);

  TORCH_CHECK(X.is_cuda() && X.is_contiguous());
  TORCH_CHECK(WQ.is_cuda() && WQ.is_contiguous());
  TORCH_CHECK(w_scale.is_cuda() && w_scale.is_contiguous());
  TORCH_CHECK(w_zp.is_cuda() && w_zp.is_contiguous());
  TORCH_CHECK(K >= scale_k && K % scale_k == 0);

  int group_size = K / scale_k;

  auto Y = at::empty({M, N}, X.options().dtype(at::kBFloat16));

  using MmaType = cutlass::bfloat16_t;
  using QuantType = cutlass::int4b_t;
  // TODO Is this really needed?
  constexpr int TileShapeK = 128 * 8 / cutlass::sizeof_bits<MmaType>::value;

  using ElementA = MmaType;
  using LayoutA = cutlass::layout::RowMajor;
  constexpr int AlignmentA =
      128 /
      cutlass::sizeof_bits<
          ElementA>::value; // Memory access granularity/alignment of A
                            // matrix in units of elements (up to 16 bytes)

  using ElementB = QuantType;
  using LayoutB = cutlass::layout::ColumnMajor;
  constexpr int AlignmentB =
      128 /
      cutlass::sizeof_bits<
          ElementB>::value; // Memory access granularity/alignment of B
                            // matrix in units of elements (up to 16 bytes)

  // We transpose and swap inputs.
  using LayoutA_Transpose =
      typename cutlass::layout::LayoutTranspose<LayoutA>::type;
  using LayoutB_Transpose =
      typename cutlass::layout::LayoutTranspose<LayoutB>::type;

  using LayoutScale = cutlass::layout::RowMajor;

  using ElementScale = MmaType;
  using ElementZero = MmaType;
  using ElementCompute = float;
  using ElementAccumulator = float;

  using ElementOutput = cutlass::bfloat16_t;
  using LayoutOutput = cutlass::layout::RowMajor;
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
      cute::Int<TileShapeK>>; // Threadblock-level
                              // tile size
  using ClusterShape = cute::Shape<
      cute::Int<TBS_M>,
      cute::Int<TBS_N>,
      cute::Int<TBS_K>>; // Shape of the
                         // threadblocks in a
                         // cluster
  using DefaultSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
  using PongSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
  using DefaultEpiSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
  using PongEpiSchedule = cutlass::epilogue::TmaWarpSpecialized;
  using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;
  using KernelSchedule =
      cute::conditional_t<PONG, PongSchedule, DefaultSchedule>;
  // TODO Possible that only cooperative schedule works.
  using EpilogueSchedule = DefaultEpiSchedule;
  // using EpilogueSchedule =
  //     cute::conditional_t<PONG, PongEpiSchedule, DefaultEpiSchedule>;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm90,
          cutlass::arch::OpClassTensorOp,
          TileShape,
          ClusterShape,
          EpilogueTileType,
          ElementAccumulator,
          ElementAccumulator,
          // Transpose layout of D here since we use explicit swap + transpose
          // the void type for C tells the builder to allocate 0 smem for the C
          // matrix. We can enable this if beta == 0 by changing ElementC to
          // void below.
          ElementOutput,
          typename cutlass::layout::LayoutTranspose<LayoutOutput>::type,
          AlignmentOutput,
          ElementOutput,
          typename cutlass::layout::LayoutTranspose<LayoutOutput>::type,
          AlignmentOutput,
          EpilogueSchedule // This is the only epi supporting the required swap
                           // + transpose.
          >::CollectiveOp;

  using CollectiveMainloopScaleWithZeroPoint =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag,
          OperatorClass,
          cute::tuple<ElementB, ElementScale, ElementZero>,
          LayoutB_Transpose,
          AlignmentB,
          ElementA,
          LayoutA_Transpose,
          AlignmentA,
          ElementAccumulator,
          TileShape,
          ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          KernelSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int, int>, // Indicates ProblemShape
      CollectiveMainloopScaleWithZeroPoint,
      CollectiveEpilogue>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideOutput = typename Gemm::GemmKernel::StrideC;
  using StrideS = typename CollectiveMainloopScaleWithZeroPoint::StrideScale;

  StrideA stride_A =
      cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  StrideB stride_B =
      cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
  StrideOutput stride_output = cutlass::make_cute_packed_stride(
      StrideOutput{}, cute::make_shape(N, M, 1));
  StrideS stride_S = cutlass::make_cute_packed_stride(
      StrideS{}, cute::make_shape(N, scale_k, 1));

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {N, M, K, 1},
      {reinterpret_cast<ElementB*>(WQ.data_ptr()),
       stride_B,
       reinterpret_cast<ElementA*>(X.data_ptr()),
       stride_A,
       reinterpret_cast<ElementScale*>(w_scale.data_ptr()),
       stride_S,
       group_size,
       reinterpret_cast<ElementZero*>(w_zp.data_ptr())},
      {{},
       reinterpret_cast<ElementOutput*>(Y.data_ptr()),
       stride_output,
       reinterpret_cast<ElementOutput*>(Y.data_ptr()),
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

at::Tensor dispatch_bf16i4bf16_rowwise_kernel(
    at::Tensor X, // BF16
    at::Tensor WQ, // INT4
    at::Tensor w_scale,
    at::Tensor w_zp) {
  KernelMode kernel = get_kernel_mode(X, WQ);
  if (kernel == KernelMode::Small) {
    return bf16i4bf16_rowwise_impl<64, 128, 128, 2, 1, 1, true>(
        X, WQ, w_scale, w_zp);
  } else if (kernel == KernelMode::Large) {
    return bf16i4bf16_rowwise_impl<128, 128, 128, 2, 1, 1, true>(
        X, WQ, w_scale, w_zp);
  } else {
    return bf16i4bf16_rowwise_impl<128, 128, 128, 2, 1, 1, false>(
        X, WQ, w_scale, w_zp);
  }
}

at::Tensor bf16i4bf16_rowwise(
    at::Tensor X, // BF16
    at::Tensor WQ, // INT4
    at::Tensor w_scale,
    at::Tensor w_zp) {
  // Check datatypes.
  TORCH_CHECK(
      (w_scale.dtype() == at::kBFloat16 && w_zp.dtype() == at::kBFloat16),
      "Weight scale and zero point tensors must be bfloat16 and dtype of weight scale and zero point tensors must be the same.");

  return dispatch_bf16i4bf16_rowwise_kernel(X, WQ, w_scale, w_zp);
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
