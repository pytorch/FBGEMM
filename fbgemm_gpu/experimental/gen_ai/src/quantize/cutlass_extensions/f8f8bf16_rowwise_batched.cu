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

constexpr int kNumSMsForH100 = 132;

int ceildiv(int a, int b) {
  return (a + b - 1) / b;
}

int round_up_to_nearest_multiple(int a, int b) {
  return ceildiv(a, b) * b;
}

#if CUDART_VERSION >= 12000

// Cutlass rowwise batched kernel
template <
    int TB_M,
    int TB_N,
    int TB_K,
    int TBS_M,
    int TBS_N,
    int TBS_K,
    bool PONG,
    bool FAST_ACCUM,
    bool USE_BIAS,
    bool Transposed,
    typename INPUT_DTYPE,
    typename BIAS_DTYPE>
at::Tensor f8f8bf16_rowwise_batched_impl(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    std::optional<at::Tensor> output) {
  int B, M, N, K, padded_N;
  if constexpr (Transposed) {
    B = XQ.size(0);
    M = XQ.size(2);
    N = WQ.size(2);
    K = WQ.size(1);
    padded_N = N;
    if (N % 256 > 0) {
      padded_N = ((N - 1) / 256 + 1) * 256;
      // Create a new tensor with the padded shape
      at::Tensor padded_w_scale =
          w_scale.new_empty({w_scale.size(0), padded_N});
      // Copy the original tensor into the new tensor
      padded_w_scale.slice(/*dim=*/1, /*start=*/0, /*end=*/N).copy_(w_scale);
      // Update w_scale to the new padded tensor
      w_scale = std::move(padded_w_scale);
    }
  } else {
    B = XQ.size(0);
    M = XQ.size(1);
    N = WQ.size(1);
    K = WQ.size(2);
    padded_N = N;
  }

  at::Tensor Y;
  if (output.has_value()) {
    Y = output.value();
    TORCH_CHECK(Y.dtype() == at::kBFloat16);
  } else {
    Y = at::empty({B, M, N}, XQ.options().dtype(at::kBFloat16));
  }

  using ElementInputA = INPUT_DTYPE;
  using LayoutInputA = cutlass::layout::RowMajor;
  constexpr int AlignmentInputA = 16 / sizeof(ElementInputA);

  using ElementInputB = cutlass::float_e4m3_t;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  constexpr int AlignmentInputB = 16 / sizeof(ElementInputB);

  using ElementBias = BIAS_DTYPE;

  using ElementOutput = cutlass::bfloat16_t;
  using LayoutOutput = std::conditional_t<
      Transposed,
      cutlass::layout::ColumnMajor,
      cutlass::layout::RowMajor>;
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
      cute::Stride<cute::Int<1>, cute::Int<0>, int32_t>>;

  using WScale = cutlass::epilogue::fusion::Sm90RowBroadcast<
      0,
      TileShape,
      ElementComputeEpilogue,
      ElementComputeEpilogue,
      cute::Stride<cute::Int<0>, cute::Int<1>, int32_t>>;

  using Bias = cutlass::epilogue::fusion::Sm90RowBroadcast<
      0,
      TileShape,
      ElementBias,
      ElementBias,
      cute::Stride<cute::Int<0>, cute::Int<1>, int32_t>>;

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
          cutlass::epilogue::TmaWarpSpecialized,
          EpilogueEVT>::CollectiveOp;

  using DefaultSchedule = cutlass::gemm::KernelTmaWarpSpecialized;
  using PongSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
  using FastDefaultSchedule =
      cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum;
  using FastPongSchedule =
      cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum;
  using SlowAccum = cute::conditional_t<PONG, PongSchedule, DefaultSchedule>;
  using FastAccum =
      cute::conditional_t<PONG, FastPongSchedule, FastDefaultSchedule>;
  using MainLoopSchedule =
      cute::conditional_t<FAST_ACCUM, FastAccum, SlowAccum>;

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
      cute::Shape<int, int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideInputA = typename Gemm::GemmKernel::StrideA;
  using StrideInputB = typename Gemm::GemmKernel::StrideB;
  using StrideOutput = typename Gemm::GemmKernel::StrideC;

  StrideInputA stride_a = cutlass::make_cute_packed_stride(
      StrideInputA{}, cute::make_shape(M, K, B));
  StrideInputB stride_b = cutlass::make_cute_packed_stride(
      StrideInputB{}, cute::make_shape(N, K, B));
  StrideOutput stride_output = cutlass::make_cute_packed_stride(
      StrideOutput{}, cute::make_shape(M, N, B));

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, B},
      {reinterpret_cast<ElementInputA*>(XQ.data_ptr()),
       stride_a,
       reinterpret_cast<ElementInputB*>(WQ.data_ptr()),
       stride_b},
      {{}, // Epilogue thread we populate below.
       (ElementOutput*)Y.data_ptr<at::BFloat16>(),
       stride_output,
       (ElementOutput*)Y.data_ptr<at::BFloat16>(),
       stride_output}};

  if constexpr (USE_BIAS) {
    arguments.epilogue.thread = {
        {reinterpret_cast<ElementBias*>(bias.value().data_ptr()),
         ElementBias(0),
         {cute::Int<0>(), cute::Int<1>(), int32_t(padded_N)}}, // bias
        // compute_1
        {
            {reinterpret_cast<ElementComputeEpilogue*>(x_scale.data_ptr()),
             ElementComputeEpilogue(0),
             {cute::Int<1>(), cute::Int<0>(), int32_t(M)}}, // x_scale
            // compute_0
            {
                {reinterpret_cast<ElementComputeEpilogue*>(w_scale.data_ptr()),
                 ElementComputeEpilogue(0),
                 {cute::Int<0>(),
                  cute::Int<1>(),
                  int32_t(padded_N)}}, // w_scale
                {}, // Accumulator
                {} // Multiplies
            },
            {}, // Multiplies
        },
        {}, // Plus
    };
  } else {
    arguments.epilogue.thread = {
        {reinterpret_cast<ElementComputeEpilogue*>(x_scale.data_ptr()),
         ElementComputeEpilogue(0),
         {cute::Int<1>(), cute::Int<0>(), int32_t(M)}}, // x_scale
        // compute_0
        {
            {reinterpret_cast<ElementComputeEpilogue*>(w_scale.data_ptr()),
             ElementComputeEpilogue(0),
             {cute::Int<0>(), cute::Int<1>(), int32_t(padded_N)}}, // w_scale
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

template <
    int TBS_M,
    int TBS_N,
    int TBS_K,
    bool FAST_ACCUM,
    bool USE_BIAS,
    bool Transposed,
    typename INPUT_DTYPE,
    typename BIAS_DTYPE>
at::Tensor dispatch_fp8_rowwise_batched_kernel_on_tile_size(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    std::optional<at::Tensor> out) {
  int M, N;
  if constexpr (Transposed) {
    M = XQ.size(2);
    N = WQ.size(2);
  } else {
    M = XQ.size(1);
    N = WQ.size(1);
  }

  if ((ceildiv(M, 64 * TBS_M) * ceildiv(N, 128 * TBS_N)) <= kNumSMsForH100 /
          cute::size(cute::Shape<
                     cute::Int<TBS_M>,
                     cute::Int<TBS_N>,
                     cute::Int<TBS_K>>{})) {
    return f8f8bf16_rowwise_batched_impl<
        64,
        128,
        128,
        TBS_M,
        TBS_N,
        TBS_K,
        false,
        FAST_ACCUM,
        USE_BIAS,
        Transposed,
        INPUT_DTYPE,
        BIAS_DTYPE>(XQ, WQ, x_scale, w_scale, bias, out);
  } else {
    return f8f8bf16_rowwise_batched_impl<
        128,
        128,
        128,
        TBS_M,
        TBS_N,
        TBS_K,
        true,
        FAST_ACCUM,
        USE_BIAS,
        Transposed,
        INPUT_DTYPE,
        BIAS_DTYPE>(XQ, WQ, x_scale, w_scale, bias, out);
  }
}

template <
    int TBS_M,
    int TBS_N,
    int TBS_K,
    bool FAST_ACCUM,
    bool USE_BIAS,
    bool Transposed,
    typename INPUT_DTYPE,
    typename BIAS_DTYPE>
at::Tensor handle_transposition(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    std::optional<at::Tensor> out) {
  if constexpr (!Transposed) {
    return dispatch_fp8_rowwise_batched_kernel_on_tile_size<
        TBS_M,
        TBS_N,
        TBS_K,
        FAST_ACCUM,
        USE_BIAS,
        Transposed,
        INPUT_DTYPE,
        BIAS_DTYPE>(XQ, WQ, x_scale, w_scale, bias, out);
  } else {
    at::Tensor out_;
    if (out.has_value()) {
      out_ = dispatch_fp8_rowwise_batched_kernel_on_tile_size<
          TBS_M,
          TBS_N,
          TBS_K,
          FAST_ACCUM,
          USE_BIAS,
          Transposed,
          INPUT_DTYPE,
          BIAS_DTYPE>(
          WQ.transpose(1, 2),
          XQ.transpose(1, 2),
          w_scale,
          x_scale,
          bias,
          out.value().transpose(1, 2));
    } else {
      out_ = dispatch_fp8_rowwise_batched_kernel_on_tile_size<
          TBS_M,
          TBS_N,
          TBS_K,
          FAST_ACCUM,
          USE_BIAS,
          Transposed,
          INPUT_DTYPE,
          BIAS_DTYPE>(
          WQ.transpose(1, 2), XQ.transpose(1, 2), w_scale, x_scale, bias, out);
    }
    return out_.transpose(1, 2).contiguous();
  }
}

// FP8 Rowwise batched Cutlass kernel dispatch.
template <typename InputDType, bool FastAccum, bool UseBias, typename BiasDType>
at::Tensor dispatch_fp8_rowwise_batched_kernel_on_cluster_size_and_transpose(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    std::optional<at::Tensor> output) {
  TORCH_CHECK(
      (XQ.dim() == 3 && WQ.dim() == 3),
      "FP8 rowwise batched GEMM only supports 3D inputs");
  int M, N;
  M = XQ.size(1);
  N = WQ.size(1);
  // All the tiles we use have sizes which are multiples of 64, hence any
  // non-multiple of 64 will get padded anyways. Let's round up to simplify.
  M = round_up_to_nearest_multiple(M, 64);
  N = round_up_to_nearest_multiple(N, 64);

  // Small/skinny shapes with odd multiples of 64.
  if (M == 64 && N >= 3072) {
    return handle_transposition<
        1,
        2,
        1,
        FastAccum,
        UseBias,
        false,
        InputDType,
        BiasDType>(XQ, WQ, x_scale, w_scale, bias, output);
  }

  if (N == 64 && M >= 3072) {
    return handle_transposition<
        1,
        2,
        1,
        FastAccum,
        UseBias,
        false,
        InputDType,
        BiasDType>(XQ, WQ, x_scale, w_scale, bias, output);
  }

  if (M == 192 && N >= 4096) {
    return handle_transposition<
        1,
        2,
        1,
        FastAccum,
        UseBias,
        false,
        InputDType,
        BiasDType>(XQ, WQ, x_scale, w_scale, bias, output);
  }

  if (N == 192 && M >= 4096) {
    return handle_transposition<
        1,
        2,
        1,
        FastAccum,
        UseBias,
        false,
        InputDType,
        BiasDType>(XQ, WQ, x_scale, w_scale, bias, output);
  }

  // Now to odd multiples of 128 (but only if not too large).
  if (M * N <= 4096 * 4096) {
    if (M % 256 > 0 && N % 256 == 0) {
      return handle_transposition<
          2,
          1,
          1,
          FastAccum,
          UseBias,
          false,
          InputDType,
          BiasDType>(XQ, WQ, x_scale, w_scale, bias, output);
    }
    if (N % 256 > 0 && M % 256 == 0) {
      return handle_transposition<
          2,
          1,
          1,
          FastAccum,
          UseBias,
          false,
          InputDType,
          BiasDType>(XQ, WQ, x_scale, w_scale, bias, output);
    }
  }
  if (M % 256 > 0 && N % 256 > 0) {
    if ((M <= N) ^ (M * N <= 1024 * 1024)) {
      return handle_transposition<
          2,
          1,
          1,
          FastAccum,
          UseBias,
          false,
          InputDType,
          BiasDType>(XQ, WQ, x_scale, w_scale, bias, output);
    } else {
      return handle_transposition<
          2,
          1,
          1,
          FastAccum,
          UseBias,
          false,
          InputDType,
          BiasDType>(XQ, WQ, x_scale, w_scale, bias, output);
    }
  }

  // General case for large tensors.
  if (M >= 1024 && N >= 1024) {
    return handle_transposition<
        2,
        1,
        1,
        FastAccum,
        UseBias,
        false,
        InputDType,
        BiasDType>(XQ, WQ, x_scale, w_scale, bias, output);
  } else {
    return handle_transposition<
        1,
        2,
        1,
        FastAccum,
        UseBias,
        false,
        InputDType,
        BiasDType>(XQ, WQ, x_scale, w_scale, bias, output);
  }
}

at::Tensor f8f8bf16_rowwise_batched(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    at::Tensor x_scale, // FP32
    at::Tensor w_scale, // FP32
    std::optional<at::Tensor> bias = std::nullopt,
    bool use_fast_accum = true,
    std::optional<at::Tensor> output = std::nullopt) {
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
          return dispatch_fp8_rowwise_batched_kernel_on_cluster_size_and_transpose<
              cutlass::float_e5m2_t,
              true,
              true,
              cutlass::bfloat16_t>(XQ, WQ, x_scale, w_scale, bias, output);
        } else {
          return dispatch_fp8_rowwise_batched_kernel_on_cluster_size_and_transpose<
              cutlass::float_e4m3_t,
              true,
              true,
              cutlass::bfloat16_t>(XQ, WQ, x_scale, w_scale, bias, output);
        }
      } else {
        if (use_e5m2) {
          return dispatch_fp8_rowwise_batched_kernel_on_cluster_size_and_transpose<
              cutlass::float_e5m2_t,
              false,
              true,
              cutlass::bfloat16_t>(XQ, WQ, x_scale, w_scale, bias, output);
        } else {
          return dispatch_fp8_rowwise_batched_kernel_on_cluster_size_and_transpose<
              cutlass::float_e4m3_t,
              false,
              true,
              cutlass::bfloat16_t>(XQ, WQ, x_scale, w_scale, bias, output);
        }
      }
    } else {
      if (use_fast_accum) {
        if (use_e5m2) {
          return dispatch_fp8_rowwise_batched_kernel_on_cluster_size_and_transpose<
              cutlass::float_e5m2_t,
              true,
              true,
              float>(XQ, WQ, x_scale, w_scale, bias, output);
        } else {
          return dispatch_fp8_rowwise_batched_kernel_on_cluster_size_and_transpose<
              cutlass::float_e4m3_t,
              true,
              true,
              float>(XQ, WQ, x_scale, w_scale, bias, output);
        }
      } else {
        if (use_e5m2) {
          return dispatch_fp8_rowwise_batched_kernel_on_cluster_size_and_transpose<
              cutlass::float_e5m2_t,
              false,
              true,
              float>(XQ, WQ, x_scale, w_scale, bias, output);
        } else {
          return dispatch_fp8_rowwise_batched_kernel_on_cluster_size_and_transpose<
              cutlass::float_e4m3_t,
              false,
              true,
              float>(XQ, WQ, x_scale, w_scale, bias, output);
        }
      }
    }
  } else {
    if (use_fast_accum) {
      if (use_e5m2) {
        return dispatch_fp8_rowwise_batched_kernel_on_cluster_size_and_transpose<
            cutlass::float_e5m2_t,
            true,
            false,
            float>(XQ, WQ, x_scale, w_scale, bias, output);
      } else {
        return dispatch_fp8_rowwise_batched_kernel_on_cluster_size_and_transpose<
            cutlass::float_e4m3_t,
            true,
            false,
            float>(XQ, WQ, x_scale, w_scale, bias, output);
      }
    } else {
      if (use_e5m2) {
        return dispatch_fp8_rowwise_batched_kernel_on_cluster_size_and_transpose<
            cutlass::float_e5m2_t,
            false,
            false,
            float>(XQ, WQ, x_scale, w_scale, bias, output);
      } else {
        return dispatch_fp8_rowwise_batched_kernel_on_cluster_size_and_transpose<
            cutlass::float_e4m3_t,
            false,
            false,
            float>(XQ, WQ, x_scale, w_scale, bias, output);
      }
    }
  }
}

#else

at::Tensor f8f8bf16_rowwise_batched(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias = std::nullopt,
    bool use_fast_accum = true,
    std::optional<at::Tensor> output = std::nullopt) {
  throw std::runtime_error(
      "CUDA version is older than 12.0"); // requires CUDA>=12
}

#endif

} // namespace fbgemm_gpu
