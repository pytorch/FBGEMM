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

// Cutlass rowwise kernel
template <
    int TB_M,
    int TB_N,
    int TB_K,
    int TBS_M,
    int TBS_N,
    int TBS_K,
    bool PONG,
    bool COOP,
    bool FAST_ACCUM,
    bool USE_BIAS,
    typename INPUT_DTYPE,
    typename BIAS_DTYPE>
at::Tensor f8f8bf16_rowwise_impl(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    std::optional<at::Tensor> output) {
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

  at::Tensor Y;
  if (output.has_value()) {
    Y = output.value();
    // Make sure the provided output has the proper shape and dtype.
    TORCH_CHECK(Y.sizes().vec() == out_sizes);
    TORCH_CHECK(Y.dtype() == at::kBFloat16);
  } else {
    Y = at::empty(out_sizes, XQ.options().dtype(at::kBFloat16));
  }

  using ElementA = INPUT_DTYPE;
  using LayoutA = cutlass::layout::RowMajor;
  constexpr int AlignmentInputA = 16 / sizeof(ElementA);

  using ElementB = cutlass::float_e4m3_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  constexpr int AlignmentInputB = 16 / sizeof(ElementB);

  // Use implicit transpose to enable more flexible configurations.
  using LayoutA_Transpose =
      typename cutlass::layout::LayoutTranspose<LayoutA>::type;
  using LayoutB_Transpose =
      typename cutlass::layout::LayoutTranspose<LayoutB>::type;

  using ElementBias = BIAS_DTYPE;

  using ElementOutput = cutlass::bfloat16_t;
  using LayoutOutput = cutlass::layout::RowMajor;
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
  using StageCountType =
      cutlass::gemm::collective::StageCountAuto; // Stage count maximized
                                                 // based on the tile size
  using KernelSchedule = cutlass::gemm::collective::
      KernelScheduleAuto; // Kernel to launch based on the default setting in
                          // the Collective Builder

  // Implement rowwise scaling epilogue.
  using XScale = cutlass::epilogue::fusion::Sm90ColBroadcast<
      0,
      TileShape,
      ElementComputeEpilogue,
      ElementComputeEpilogue,
      cute::Stride<cute::Int<1>, cute::Int<0>, cute::Int<0>>>;

  using WScale = cutlass::epilogue::fusion::Sm90RowBroadcast<
      0,
      TileShape,
      ElementComputeEpilogue,
      ElementComputeEpilogue,
      cute::Stride<cute::Int<0>, cute::Int<1>, cute::Int<0>>>;

  using Bias = cutlass::epilogue::fusion::Sm90ColBroadcast<
      0,
      TileShape,
      ElementBias,
      ElementBias,
      cute::Stride<cute::Int<1>, cute::Int<0>, cute::Int<0>>>;

  using Accum = cutlass::epilogue::fusion::Sm90AccFetch;

  using Compute0 = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies,
      ElementComputeEpilogue, // First stage output type.
      ElementComputeEpilogue, // First stage input types.
      cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTCompute0 =
      cutlass::epilogue::fusion::Sm90EVT<Compute0, WScale, Accum>;

  using Compute1 = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies,
      cute::conditional_t< // Second stage output type.
          USE_BIAS,
          ElementBias,
          ElementOutput>,
      ElementComputeEpilogue, // Second stage input types.
      cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTCompute1 =
      cutlass::epilogue::fusion::Sm90EVT<Compute1, XScale, EVTCompute0>;

  using ComputeBias = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::plus,
      ElementOutput, // Final (optional) stage output type.
      ElementBias, // Final stage input types.
      cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTComputeBias =
      cutlass::epilogue::fusion::Sm90EVT<ComputeBias, Bias, EVTCompute1>;

  using EpilogueEVT =
      cute::conditional_t<USE_BIAS, EVTComputeBias, EVTCompute1>;

  using DefaultSchedule = cutlass::gemm::KernelTmaWarpSpecialized;
  using PongSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
  using SlowAccum = cute::conditional_t<PONG, PongSchedule, DefaultSchedule>;
  using FastAccum = cute::conditional_t<
      COOP,
      cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8FastAccum,
      cute::conditional_t<
          PONG,
          cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum,
          cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum>>;
  using MainLoopSchedule =
      cute::conditional_t<FAST_ACCUM, FastAccum, SlowAccum>;
  using EpilogueSchedule = cute::conditional_t<
      COOP,
      cutlass::epilogue::TmaWarpSpecializedCooperative,
      cutlass::epilogue::TmaWarpSpecialized>;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm90,
          cutlass::arch::OpClassTensorOp,
          TileShape,
          ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAccumulator,
          ElementComputeEpilogue,
          void,
          typename cutlass::layout::LayoutTranspose<LayoutOutput>::type,
          AlignmentOutput,
          ElementOutput,
          typename cutlass::layout::LayoutTranspose<LayoutOutput>::type,
          AlignmentOutput,
          EpilogueSchedule,
          EpilogueEVT>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag,
          OperatorClass,
          ElementB,
          LayoutB_Transpose,
          AlignmentInputA,
          ElementA,
          LayoutA_Transpose,
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
      StrideOutput{}, cute::make_shape(N, M, 1));

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {N, M, K},
      {reinterpret_cast<ElementB*>(WQ.data_ptr()),
       stride_b,
       reinterpret_cast<ElementA*>(XQ.data_ptr()),
       stride_a},
      {{}, // Epilogue thread we populate below.
       nullptr,
       stride_output,
       (ElementOutput*)Y.data_ptr<at::BFloat16>(),
       stride_output}};

  if constexpr (USE_BIAS) {
    arguments.epilogue.thread = {
        {reinterpret_cast<ElementBias*>(bias.value().data_ptr())}, // bias
        // compute_1
        {
            {reinterpret_cast<ElementComputeEpilogue*>(
                w_scale.data_ptr())}, // x_scale
            // compute_0
            {
                {reinterpret_cast<ElementComputeEpilogue*>(
                    x_scale.data_ptr())}, // w_scale
                {}, // Accumulator
                {} // Multiplies
            },
            {}, // Multiplies
        },
        {}, // Plus
    };
  } else {
    arguments.epilogue.thread = {
        {reinterpret_cast<ElementComputeEpilogue*>(
            w_scale.data_ptr())}, // x_scale
        // compute_0
        {
            {reinterpret_cast<ElementComputeEpilogue*>(
                x_scale.data_ptr())}, // w_scale
            {}, // Accumulator
            {} // Multiplies
        },
        {}, // Multiplies
    };
  }

  Gemm gemm;

  // Using the arguments, query for extra workspace required for matrix
  // multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  at::Tensor workspace =
      at::empty(workspace_size, XQ.options().dtype(at::kByte));

  // Check the problem size is supported or not
  cutlass::Status status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot implement");
  }

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm.initialize(arguments, workspace.data_ptr());
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
    bool PONG,
    bool COOP>
at::Tensor f8f8bf16_rowwise_wrapper(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    bool use_fast_accum,
    std::optional<at::Tensor> bias,
    std::optional<at::Tensor> output) {
  // Check datatypes.
  TORCH_CHECK(
      x_scale.dtype() == at::kFloat && w_scale.dtype() == at::kFloat,
      "Scale tensors must be float32.");
  if (bias.has_value()) {
    TORCH_CHECK(
        bias.value().dtype() == at::kFloat ||
            bias.value().dtype() == at::kBFloat16,
        "Bias type must be bfloat16 or float32 if provided.");
  }
  bool use_bias = bias.has_value();
  bool bf16_bias = use_bias && bias.value().dtype() == at::kBFloat16;

  // Templatize based on input dtype.
  bool use_e5m2 = XQ.dtype() == at::kFloat8_e5m2;

  if (use_bias) {
    if (bf16_bias) {
      if (use_fast_accum) {
        if (use_e5m2) {
          return f8f8bf16_rowwise_impl<
              TB_M,
              TB_N,
              TB_K,
              TBS_M,
              TBS_N,
              TBS_K,
              PONG,
              COOP,
              true,
              true,
              cutlass::float_e5m2_t,
              cutlass::bfloat16_t>(XQ, WQ, x_scale, w_scale, bias, output);
        } else {
          return f8f8bf16_rowwise_impl<
              TB_M,
              TB_N,
              TB_K,
              TBS_M,
              TBS_N,
              TBS_K,
              PONG,
              COOP,
              true,
              true,
              cutlass::float_e4m3_t,
              cutlass::bfloat16_t>(XQ, WQ, x_scale, w_scale, bias, output);
        }
      } else {
        if (use_e5m2) {
          return f8f8bf16_rowwise_impl<
              TB_M,
              TB_N,
              TB_K,
              TBS_M,
              TBS_N,
              TBS_K,
              PONG,
              COOP,
              false,
              true,
              cutlass::float_e5m2_t,
              cutlass::bfloat16_t>(XQ, WQ, x_scale, w_scale, bias, output);
        } else {
          return f8f8bf16_rowwise_impl<
              TB_M,
              TB_N,
              TB_K,
              TBS_M,
              TBS_N,
              TBS_K,
              PONG,
              COOP,
              false,
              true,
              cutlass::float_e4m3_t,
              cutlass::bfloat16_t>(XQ, WQ, x_scale, w_scale, bias, output);
        }
      }
    } else {
      if (use_fast_accum) {
        if (use_e5m2) {
          return f8f8bf16_rowwise_impl<
              TB_M,
              TB_N,
              TB_K,
              TBS_M,
              TBS_N,
              TBS_K,
              PONG,
              COOP,
              true,
              true,
              cutlass::float_e5m2_t,
              float>(XQ, WQ, x_scale, w_scale, bias, output);
        } else {
          return f8f8bf16_rowwise_impl<
              TB_M,
              TB_N,
              TB_K,
              TBS_M,
              TBS_N,
              TBS_K,
              PONG,
              COOP,
              true,
              true,
              cutlass::float_e4m3_t,
              float>(XQ, WQ, x_scale, w_scale, bias, output);
        }
      } else {
        if (use_e5m2) {
          return f8f8bf16_rowwise_impl<
              TB_M,
              TB_N,
              TB_K,
              TBS_M,
              TBS_N,
              TBS_K,
              PONG,
              COOP,
              false,
              true,
              cutlass::float_e5m2_t,
              float>(XQ, WQ, x_scale, w_scale, bias, output);
        } else {
          return f8f8bf16_rowwise_impl<
              TB_M,
              TB_N,
              TB_K,
              TBS_M,
              TBS_N,
              TBS_K,
              PONG,
              COOP,
              false,
              true,
              cutlass::float_e4m3_t,
              float>(XQ, WQ, x_scale, w_scale, bias, output);
        }
      }
    }
  } else {
    if (use_fast_accum) {
      if (use_e5m2) {
        return f8f8bf16_rowwise_impl<
            TB_M,
            TB_N,
            TB_K,
            TBS_M,
            TBS_N,
            TBS_K,
            PONG,
            COOP,
            true,
            false,
            cutlass::float_e5m2_t,
            float>(XQ, WQ, x_scale, w_scale, bias, output);
      } else {
        return f8f8bf16_rowwise_impl<
            TB_M,
            TB_N,
            TB_K,
            TBS_M,
            TBS_N,
            TBS_K,
            PONG,
            COOP,
            true,
            false,
            cutlass::float_e4m3_t,
            float>(XQ, WQ, x_scale, w_scale, bias, output);
      }
    } else {
      if (use_e5m2) {
        return f8f8bf16_rowwise_impl<
            TB_M,
            TB_N,
            TB_K,
            TBS_M,
            TBS_N,
            TBS_K,
            PONG,
            COOP,
            false,
            false,
            cutlass::float_e5m2_t,
            float>(XQ, WQ, x_scale, w_scale, bias, output);
      } else {
        return f8f8bf16_rowwise_impl<
            TB_M,
            TB_N,
            TB_K,
            TBS_M,
            TBS_N,
            TBS_K,
            PONG,
            COOP,
            false,
            false,
            cutlass::float_e4m3_t,
            float>(XQ, WQ, x_scale, w_scale, bias, output);
      }
    }
  }
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
    bool COOP>
at::Tensor f8f8bf16_rowwise_wrapper(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    bool use_fast_accum,
    std::optional<at::Tensor> bias,
    std::optional<at::Tensor> output) {
  throw std::runtime_error(
      "CUDA version is older than 12.0"); // requires CUDA>=12
}

#endif

} // namespace fbgemm_gpu
