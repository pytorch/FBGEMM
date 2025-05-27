/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "common.cuh"

namespace fbgemm_gpu {

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

/*
  The cartesian product instantiations are derived from the call stack from
  f8f8bf16_rowwise_batched down to f8f8bf16_rowwise_batched_impl.  They need to
  be updated accordingly if the cartesian product changes.
*/

#define INSTANTIATE_FUNC_0(                          \
    TB_M,                                            \
    TB_N,                                            \
    TB_K,                                            \
    TBS_M,                                           \
    TBS_N,                                           \
    TBS_K,                                           \
    PONG,                                            \
    FAST_ACCUM,                                      \
    USE_BIAS,                                        \
    Transposed,                                      \
    INPUT_DTYPE,                                     \
    BIAS_DTYPE)                                      \
  template at::Tensor f8f8bf16_rowwise_batched_impl< \
      TB_M,                                          \
      TB_N,                                          \
      TB_K,                                          \
      TBS_M,                                         \
      TBS_N,                                         \
      TBS_K,                                         \
      PONG,                                          \
      FAST_ACCUM,                                    \
      USE_BIAS,                                      \
      Transposed,                                    \
      INPUT_DTYPE,                                   \
      BIAS_DTYPE>(                                   \
      at::Tensor XQ,                                 \
      at::Tensor WQ,                                 \
      at::Tensor x_scale,                            \
      at::Tensor w_scale,                            \
      std::optional<at::Tensor> bias,                \
      std::optional<at::Tensor> output);

#define INSTANTIATE_FUNC_1( \
    TBS_M,                  \
    TBS_N,                  \
    TBS_K,                  \
    FAST_ACCUM,             \
    USE_BIAS,               \
    Transposed,             \
    INPUT_DTYPE,            \
    BIAS_DTYPE)             \
  INSTANTIATE_FUNC_0(       \
      64,                   \
      128,                  \
      128,                  \
      TBS_M,                \
      TBS_N,                \
      TBS_K,                \
      false,                \
      FAST_ACCUM,           \
      USE_BIAS,             \
      Transposed,           \
      INPUT_DTYPE,          \
      BIAS_DTYPE);          \
  INSTANTIATE_FUNC_0(       \
      128,                  \
      128,                  \
      128,                  \
      TBS_M,                \
      TBS_N,                \
      TBS_K,                \
      true,                 \
      FAST_ACCUM,           \
      USE_BIAS,             \
      Transposed,           \
      INPUT_DTYPE,          \
      BIAS_DTYPE);

#define INSTANTIATE_FUNC_2(InputDType, FastAccum, UseBias, BiasDType) \
  INSTANTIATE_FUNC_1(                                                 \
      1, 2, 1, FastAccum, UseBias, false, InputDType, BiasDType);     \
  INSTANTIATE_FUNC_1(2, 1, 1, FastAccum, UseBias, false, InputDType, BiasDType);

#if CUDART_VERSION >= 12000

// Create instantiations for the cartesian product of input dtypes, bias dtypes,
// fast-accum options, and use-bias options
FOR_FLOAT_TYPES(INSTANTIATE_FUNC_2);

#endif

#undef INSTANTIATE_FUNC_2
#undef INSTANTIATE_FUNC_1
#undef INSTANTIATE_FUNC_0

} // namespace fbgemm_gpu
