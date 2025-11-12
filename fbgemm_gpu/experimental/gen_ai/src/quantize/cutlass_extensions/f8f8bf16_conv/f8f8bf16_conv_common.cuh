/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/packed_stride.hpp>

// clang-format off
// The fixed ordering of the headers is required for CUTLASS 3.2+
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/conv/collective/collective_builder.hpp>
#include <cutlass/conv/convnd_problem_shape.hpp>
#include <cutlass/conv/convolution.h>
#include <cutlass/conv/device/conv_universal_adapter.hpp>
#include <cutlass/conv/dispatch_policy.hpp>
#include <cutlass/conv/kernel/conv_universal.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
// clang-format on

namespace fbgemm_gpu {

// Cutlass FP8 convolution kernel for SM100 (Blackwell architecture)
template <
    int TB_M,
    int TB_N,
    int TB_K,
    int TBS_M,
    int TBS_N,
    int TBS_K,
    class KernelScheduleType = cutlass::conv::collective::KernelScheduleAuto>
at::Tensor f8f8bf16_conv_impl(
    at::Tensor activation, // FP8 - NDHWC layout
    at::Tensor filter, // FP8 - KTRSC layout
    at::Tensor scale,
    std::vector<int64_t> padding, // [pad_d, pad_h, pad_w]
    std::vector<int64_t> stride, // [stride_d, stride_h, stride_w]
    std::vector<int64_t> dilation) { // [dilation_d, dilation_h, dilation_w]
  c10::cuda::CUDAGuard deviceGuard(activation.device());

  // Extract dimensions from activation (NDHWC)
  TORCH_CHECK(activation.dim() == 5, "Activation must be 5D tensor (NDHWC)");
  TORCH_CHECK(filter.dim() == 5, "Filter must be 5D tensor (KTRSC)");

  int n = activation.size(0);
  int d = activation.size(1);
  int h = activation.size(2);
  int w = activation.size(3);
  int c = activation.size(4);

  // Extract dimensions from filter (KTRSC)
  int k = filter.size(0);
  int t = filter.size(1);
  int r = filter.size(2);
  int s = filter.size(3);

  TORCH_CHECK(
      filter.size(4) == c, "Filter channels must match activation channels");

  // Extract padding, stride, dilation
  TORCH_CHECK(padding.size() == 3, "Padding must have 3 elements");
  TORCH_CHECK(stride.size() == 3, "Stride must have 3 elements");
  TORCH_CHECK(dilation.size() == 3, "Dilation must have 3 elements");

  int pad_d = padding[0];
  int pad_h = padding[1];
  int pad_w = padding[2];

  int stride_d = stride[0];
  int stride_h = stride[1];
  int stride_w = stride[2];

  int dilation_d = dilation[0];
  int dilation_h = dilation[1];
  int dilation_w = dilation[2];

  // Calculate output dimensions
  int z = 1 + (d + 2 * pad_d - ((t - 1) * dilation_d + 1)) / stride_d;
  int p = 1 + (h + 2 * pad_h - ((r - 1) * dilation_h + 1)) / stride_h;
  int q = 1 + (w + 2 * pad_w - ((s - 1) * dilation_w + 1)) / stride_w;

  TORCH_CHECK(activation.is_cuda() && activation.is_contiguous());
  TORCH_CHECK(filter.is_cuda() && filter.is_contiguous());

  auto output =
      at::empty({n, z, p, q, k}, activation.options().dtype(at::kBFloat16));

  using ElementAct = cutlass::float_e4m3_t;
  using LayoutA = cutlass::layout::TensorNDHWC;
  constexpr int AlignmentAct = 128 / cutlass::sizeof_bits<ElementAct>::value;

  using ElementFlt = cutlass::float_e4m3_t;
  using LayoutB = cutlass::layout::TensorNDHWC;
  constexpr int AlignmentFlt = 128 / cutlass::sizeof_bits<ElementFlt>::value;

  using ElementOutput = cutlass::bfloat16_t;
  using LayoutC = cutlass::layout::TensorNDHWC;
  constexpr int AlignmentOutput =
      128 / cutlass::sizeof_bits<ElementOutput>::value;

  using ElementAccumulator = float;
  using ElementCompute = float;
  using ArchTag = cutlass::arch::Sm100;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  constexpr cutlass::conv::Operator ConvOp = cutlass::conv::Operator::kFprop;

  using TileShape = cute::
      Shape<cute::Int<TB_M>, cute::Int<TB_N>, cute::Shape<cute::Int<TB_K>>>;
  using ClusterShape =
      cute::Shape<cute::Int<TBS_M>, cute::Int<TBS_N>, cute::Int<TBS_K>>;

  // Define Scale EVT.
  using Scale_ =
      cutlass::epilogue::fusion::Sm90ScalarBroadcast<ElementAccumulator>;

  using Accum = cutlass::epilogue::fusion::Sm90AccFetch;

  using EpilogueCompute = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies,
      ElementOutput,
      ElementAccumulator,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using EpilogueEVT =
      cutlass::epilogue::fusion::Sm90EVT<EpilogueCompute, Scale_, Accum>;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag,
          OperatorClass,
          TileShape,
          ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAccumulator,
          ElementCompute,
          ElementOutput,
          LayoutC,
          AlignmentOutput,
          ElementOutput,
          LayoutC,
          AlignmentOutput,
          cutlass::epilogue::collective::EpilogueScheduleAuto,
          EpilogueEVT>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::conv::collective::CollectiveBuilder<
          ArchTag,
          OperatorClass,
          ConvOp,
          ElementAct,
          LayoutA,
          AlignmentAct,
          ElementFlt,
          LayoutB,
          AlignmentFlt,
          ElementAccumulator,
          TileShape,
          ClusterShape,
          cutlass::conv::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          KernelScheduleType>::CollectiveOp;

  using ProblemShape = cutlass::conv::ConvProblemShape<
      ConvOp,
      CollectiveMainloop::DispatchPolicy::NumSpatialDimensions>;
  using ConvKernel = cutlass::conv::kernel::
      ConvUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue>;

  using Conv = cutlass::conv::device::ConvUniversalAdapter<ConvKernel>;

  using StrideC = typename Conv::ConvKernel::StrideC;
  using StrideD = typename Conv::ConvKernel::StrideD;

  ProblemShape problem_shape(
      cutlass::conv::Mode::kCrossCorrelation,
      {n, d, h, w, c},
      {k, t, r, s, c},
      {pad_d, pad_h, pad_w},
      {pad_d, pad_h, pad_w},
      {stride_d, stride_h, stride_w},
      {dilation_d, dilation_h, dilation_w},
      1 // group
  );

  StrideC stride_C;
  StrideD stride_D;

  cute::for_each(cute::make_seq<cute::rank<0>(StrideC{})>{}, [&](auto i) {
    cute::get<0, i>(stride_C) =
        problem_shape.stride_C[ProblemShape::RankT - 2 - i];
  });
  cute::for_each(cute::make_seq<cute::rank<0>(StrideD{})>{}, [&](auto i) {
    cute::get<0, i>(stride_D) =
        problem_shape.stride_C[ProblemShape::RankT - 2 - i];
  });

  typename Conv::Arguments arguments{
      problem_shape,
      {reinterpret_cast<ElementAct*>(activation.data_ptr()),
       reinterpret_cast<ElementFlt*>(filter.data_ptr())},
      {{},
       reinterpret_cast<ElementOutput*>(output.data_ptr<at::BFloat16>()),
       stride_C,
       reinterpret_cast<ElementOutput*>(output.data_ptr<at::BFloat16>()),
       stride_D}};

  arguments.epilogue.thread = {
      {{}, {reinterpret_cast<ElementAccumulator*>(scale.data_ptr())}},
      {}, // Accumulator
      {}, // Multiplies
  };

  Conv conv;

  size_t workspace_size = Conv::get_workspace_size(arguments);
  at::Tensor workspace =
      at::empty(workspace_size, activation.options().dtype(at::kByte));

  cutlass::Status status = conv.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot implement convolution");
  }

  status = conv.initialize(arguments, workspace.data_ptr());
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot initialize convolution");
  }

  status = conv(at::cuda::getCurrentCUDAStream());
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error(
        std::string("cutlass cannot run convolution: ") +
        cutlass::cutlassGetStatusString(status));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return output;
}

} // namespace fbgemm_gpu
