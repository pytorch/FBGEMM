/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include "cutlass_extensions/include/kernel_mode.h"
#include "cutlass_extensions/include/threadblock.h"

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
    typename INPUT_DTYPE,
    typename WEIGHT_SCALE_DTYPE>
at::Tensor f8i4bf16_rowwise_impl(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // INT4
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor w_zp) {
  int M = XQ.size(0);
  int N = WQ.size(0);
  int K = XQ.size(1);

  int num_groups = w_scale.size(0);

  TORCH_CHECK(XQ.is_cuda() && XQ.is_contiguous());
  TORCH_CHECK(WQ.is_cuda() && WQ.is_contiguous());
  TORCH_CHECK(x_scale.is_cuda() && x_scale.is_contiguous());
  TORCH_CHECK(w_scale.is_cuda() && w_scale.is_contiguous());
  TORCH_CHECK(w_zp.is_cuda() && w_zp.is_contiguous());
  TORCH_CHECK(K >= num_groups && K % num_groups == 0);

  int group_size = K / num_groups;

  auto Y = at::empty({M, N}, XQ.options().dtype(at::kBFloat16));

  using ElementInputA = INPUT_DTYPE;
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
      cutlass::gemm::KernelTmaWarpSpecializedCooperativeMixedInput;
  using PongSchedule =
      cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput;
  using CooperativeEpilogueSchedule =
      cutlass::epilogue::TmaWarpSpecializedCooperative;
  using PongEpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized;
  using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;
  using MainLoopSchedule =
      cute::conditional_t<PONG, PongSchedule, CooperativeSchedule>;
  using EpilogueSchedule = cute::
      conditional_t<PONG, PongEpilogueSchedule, CooperativeEpilogueSchedule>;

  // Implement rowwise scaling epilogue for x
  using XScale = cutlass::epilogue::fusion::Sm90RowBroadcast<
      0,
      TileShape,
      ElementComputeEpilogue,
      ElementComputeEpilogue,
      cute::Stride<cute::Int<0>, cute::Int<1>, cute::Int<0>>>;

  using Accum = cutlass::epilogue::fusion::Sm90AccFetch;

  using Compute0 = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies,
      ElementOutput, // First stage output type.
      ElementComputeEpilogue, // First stage input types.
      cutlass::FloatRoundStyle::round_to_nearest>;

  using EpilogueEVT =
      cutlass::epilogue::fusion::Sm90EVT<Compute0, XScale, Accum>;

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
          EpilogueSchedule,
          EpilogueEVT>::CollectiveOp;

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
       reinterpret_cast<ElementInputA*>(XQ.data_ptr()),
       stride_a,
       reinterpret_cast<ElementScale*>(w_scale.data_ptr()),
       stride_S,
       group_size,
       reinterpret_cast<ElementZeroPoint*>(w_zp.data_ptr())},
      {{},
       (ElementOutput*)Y.data_ptr<at::BFloat16>(),
       stride_output,
       (ElementOutput*)Y.data_ptr<at::BFloat16>(),
       stride_output}};

  arguments.epilogue.thread = {
      {reinterpret_cast<ElementComputeEpilogue*>(
          x_scale.data_ptr())}, // x_scale
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

template <typename InputDType, typename WEIGHT_SCALE_DTYPE>
at::Tensor dispatch_f8i4bf16_rowwise_kernel(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor w_zp) {
  KernelMode kernel = get_kernel_mode(XQ, WQ);
  if (kernel == KernelMode::Small) {
    return f8i4bf16_rowwise_impl<
        64,
        128,
        128,
        2,
        1,
        1,
        true,
        InputDType,
        WEIGHT_SCALE_DTYPE>(XQ, WQ, x_scale, w_scale, w_zp);
  } else if (kernel == KernelMode::Large) {
    return f8i4bf16_rowwise_impl<
        128,
        256,
        64,
        2,
        1,
        1,
        false,
        InputDType,
        WEIGHT_SCALE_DTYPE>(XQ, WQ, x_scale, w_scale, w_zp);
  } else {
    return f8i4bf16_rowwise_impl<
        128,
        256,
        64,
        2,
        1,
        1,
        false,
        InputDType,
        WEIGHT_SCALE_DTYPE>(XQ, WQ, x_scale, w_scale, w_zp);
  }
}

at::Tensor f8i4bf16_rowwise(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // INT4
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor w_zp) {
  // Check datatypes.
  TORCH_CHECK(
      x_scale.dtype() == at::kFloat, "Input scale tensor must be float32.");
  TORCH_CHECK(
      (w_scale.dtype() == at::kFloat && w_zp.dtype() == at::kFloat) ||
          (w_scale.dtype() == at::kHalf && w_zp.dtype() == at::kHalf) ||
          (w_scale.dtype() == at::kBFloat16 && w_zp.dtype() == at::kBFloat16),
      "Weight scale and zero point tensors must be float32, bfloat16, or float16, and dtype of weight scale and zero point tensors must be the same .");

  // Templatize based on input and weight scale/zero point dtype.
  bool use_e5m2 = XQ.dtype() == at::kFloat8_e5m2;

  if (w_scale.dtype() == at::kFloat) {
    if (use_e5m2) {
      return dispatch_f8i4bf16_rowwise_kernel<cutlass::float_e5m2_t, float>(
          XQ, WQ, x_scale, w_scale, w_zp);
    } else {
      return dispatch_f8i4bf16_rowwise_kernel<cutlass::float_e4m3_t, float>(
          XQ, WQ, x_scale, w_scale, w_zp);
    }
  } else if (w_scale.dtype() == at::kHalf) {
    if (use_e5m2) {
      return dispatch_f8i4bf16_rowwise_kernel<
          cutlass::float_e5m2_t,
          cutlass::half_t>(XQ, WQ, x_scale, w_scale, w_zp);
    } else {
      return dispatch_f8i4bf16_rowwise_kernel<
          cutlass::float_e4m3_t,
          cutlass::half_t>(XQ, WQ, x_scale, w_scale, w_zp);
    }
  } else if (w_scale.dtype() == at::kBFloat16) {
    if (use_e5m2) {
      return dispatch_f8i4bf16_rowwise_kernel<
          cutlass::float_e5m2_t,
          cutlass::bfloat16_t>(XQ, WQ, x_scale, w_scale, w_zp);
    } else {
      return dispatch_f8i4bf16_rowwise_kernel<
          cutlass::float_e4m3_t,
          cutlass::bfloat16_t>(XQ, WQ, x_scale, w_scale, w_zp);
    }
  } else {
    throw std::runtime_error(
        "Weight scale and zero point data type not supported in f8i4bf16_rowwise");
  }
}

#else

at::Tensor f8i4bf16_rowwise(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // INT4
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor w_zp) {
  throw std::runtime_error(
      "CUDA version is older than 12.0"); // requires CUDA>=12
}

#endif

} // namespace fbgemm_gpu
