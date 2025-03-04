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
    bool FAST_ACCUM>
at::Tensor f8f8bf16_tensorwise_impl(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    double scale) {
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
  // Handle case where inputs are empty.
  if (M == 0 || N == 0 || K == 0) {
    return at::zeros(out_sizes, XQ.options().dtype(at::kBFloat16));
  }

  TORCH_CHECK(XQ.is_cuda() && XQ.is_contiguous());
  TORCH_CHECK(WQ.is_cuda() && WQ.is_contiguous());

  auto Y = at::empty(out_sizes, XQ.options().dtype(at::kBFloat16));

  using ElementInputA = cutlass::float_e4m3_t;
  using LayoutInputA = cutlass::layout::RowMajor;
  constexpr int AlignmentInputA = 16 /
      sizeof(ElementInputA); // Memory access granularity/alignment of A
                             // matrix in units of elements (up to 16 bytes)

  using ElementInputB = cutlass::float_e4m3_t;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  constexpr int AlignmentInputB = 16 /
      sizeof(ElementInputB); // Memory access granularity/alignment of B
                             // matrix in units of elements (up to 16 bytes)

  using ElementOutput = cutlass::bfloat16_t;
  using LayoutOutput = cutlass::layout::RowMajor;
  constexpr int AlignmentOutput = 16 /
      sizeof(ElementOutput); // Memory access granularity/alignment of C
                             // matrix in units of elements (up to 16 bytes)

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
  using StageCountType =
      cutlass::gemm::collective::StageCountAuto; // Stage count maximized
                                                 // based on the tile size
  using KernelSchedule = cutlass::gemm::collective::
      KernelScheduleAuto; // Kernel to launch based on the default setting in
                          // the Collective Builder

  using DefaultSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
  using PongSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
  using FastDefaultSchedule =
      cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8FastAccum;
  using FastPongSchedule =
      cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum;
  using SlowAccum = cute::conditional_t<PONG, PongSchedule, DefaultSchedule>;
  using FastAccum =
      cute::conditional_t<PONG, FastPongSchedule, FastDefaultSchedule>;
  using CooperativeEpilogueSchedule =
      cutlass::epilogue::TmaWarpSpecializedCooperative;
  using PongEpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized;
  using MainLoopSchedule =
      cute::conditional_t<FAST_ACCUM, FastAccum, SlowAccum>;
  using EpilogueSchedule = cute::
      conditional_t<PONG, PongEpilogueSchedule, CooperativeEpilogueSchedule>;

  using Scale_ =
      cutlass::epilogue::fusion::Sm90ScalarBroadcast<ElementComputeEpilogue>;

  using Accum = cutlass::epilogue::fusion::Sm90AccFetch;

  using Compute0 = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies,
      ElementOutput, // First stage output type.
      ElementComputeEpilogue, // First stage input types.
      cutlass::FloatRoundStyle::round_to_nearest>;

  using EpilogueEVT =
      cutlass::epilogue::fusion::Sm90EVT<Compute0, Scale_, Accum>;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm90,
          cutlass::arch::OpClassTensorOp,
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
          EpilogueSchedule,
          EpilogueEVT>::CollectiveOp;

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
  using StrideOutput = typename Gemm::GemmKernel::StrideC;

  StrideInputA stride_a = cutlass::make_cute_packed_stride(
      StrideInputA{}, cute::make_shape(M, K, 1));
  StrideInputB stride_b = cutlass::make_cute_packed_stride(
      StrideInputB{}, cute::make_shape(N, K, 1));
  StrideOutput stride_output = cutlass::make_cute_packed_stride(
      StrideOutput{}, cute::make_shape(M, N, 1));

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K},
      {reinterpret_cast<ElementInputA*>(XQ.data_ptr()),
       stride_a,
       reinterpret_cast<ElementInputB*>(WQ.data_ptr()),
       stride_b},
      {{},
       (ElementOutput*)Y.data_ptr<at::BFloat16>(),
       stride_output,
       (ElementOutput*)Y.data_ptr<at::BFloat16>(),
       stride_output}};

  arguments.epilogue.thread = {
      {float(scale)}, // scale
      {}, // Accumulator
      {}, // Multiplies
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

at::Tensor f8f8bf16_tensorwise(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    double scale,
    bool use_fast_accum) {
  KernelMode kernel = get_kernel_mode(XQ, WQ);
  if (kernel == KernelMode::Small) {
    return f8f8bf16_tensorwise_impl<64, 128, 128, 2, 1, 1, true, true>(
        XQ, WQ, scale);
  } else if (kernel == KernelMode::Large) {
    return f8f8bf16_tensorwise_impl<128, 256, 128, 2, 1, 1, false, true>(
        XQ, WQ, scale);
  } else {
    return f8f8bf16_tensorwise_impl<128, 128, 128, 1, 2, 1, true, true>(
        XQ, WQ, scale);
  }
}

#else

at::Tensor f8f8bf16_tensorwise(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    double scale,
    bool use_fast_accum) {
  throw std::runtime_error(
      "CUDA version is older than 12.0"); // requires CUDA>=12
}

#endif

} // namespace fbgemm_gpu
