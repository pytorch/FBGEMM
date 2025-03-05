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

namespace GroupedGemmArgs {
using ProblemShape =
    cutlass::gemm::GroupProblemShape<cute::Shape<int, int, int>>;
using ElementInputA = cutlass::float_e4m3_t;
using ElementInputB = cutlass::float_e4m3_t;
using ElementOutput = cutlass::bfloat16_t;
using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::RowMajor;
using ElementAccumulator = float;
using ElementComputeEpilogue = float;
using ArchTag = cutlass::arch::Sm90;
using OperatorClass = cutlass::arch::OpClassTensorOp;
using StageCountType = cutlass::gemm::collective::StageCountAuto;
// Template structure to encapsulate configurations
template <
    int TB_M,
    int TB_N,
    int TB_K,
    int TBS_M,
    int TBS_N,
    int TBS_K,
    bool PONG>
struct GroupedGemmConfigs {
  using TileShape =
      cute::Shape<cute::Int<TB_M>, cute::Int<TB_N>, cute::Int<TB_K>>;
  using ClusterShape =
      cute::Shape<cute::Int<TBS_M>, cute::Int<TBS_N>, cute::Int<TBS_K>>;
  using CooperativeSchedule =
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperativeFP8FastAccum;
  using PongSchedule =
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpongFP8FastAccum;
  using CooperativeEpilogueSchedule =
      cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
  using PongEpilogueSchedule =
      cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
  using KernelSchedule =
      cute::conditional_t<PONG, PongSchedule, CooperativeSchedule>;
  using EpilogueSchedule = cute::
      conditional_t<PONG, PongEpilogueSchedule, CooperativeEpilogueSchedule>;

  // Implement rowwise scaling epilogue.
  using XScale = cutlass::epilogue::fusion::Sm90ColBroadcast<
      0,
      TileShape,
      ElementComputeEpilogue*,
      ElementComputeEpilogue,
      cute::Stride<cute::Int<1>, cute::Int<0>, cute::Int<0>>>;

  using WScale = cutlass::epilogue::fusion::Sm90RowBroadcast<
      0,
      TileShape,
      ElementComputeEpilogue*,
      ElementComputeEpilogue,
      cute::Stride<cute::Int<0>, cute::Int<1>, cute::Int<0>>>;

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
      ElementOutput,
      ElementComputeEpilogue, // Second stage input types.
      cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTCompute1 =
      cutlass::epilogue::fusion::Sm90EVT<Compute1, XScale, EVTCompute0>;

  using EpilogueEVT = EVTCompute1;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm90,
          cutlass::arch::OpClassTensorOp,
          TileShape,
          ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAccumulator,
          ElementAccumulator,
          ElementOutput,
          LayoutOutput*,
          128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementOutput,
          LayoutOutput*,
          128 / cutlass::sizeof_bits<ElementOutput>::value,
          EpilogueSchedule,
          EpilogueEVT>::CollectiveOp;
  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag,
          OperatorClass,
          ElementInputA,
          LayoutInputA*,
          128 / cutlass::sizeof_bits<ElementInputA>::value,
          ElementInputB,
          LayoutInputB*,
          128 / cutlass::sizeof_bits<ElementInputB>::value,
          ElementAccumulator,
          TileShape,
          ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          KernelSchedule>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::
      GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideInputA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideInputB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideOutput = typename Gemm::GemmKernel::InternalStrideD;
};
} // namespace GroupedGemmArgs

__global__ void set_kernel_args_kernel(
    int64_t xq_ptr,
    int64_t wq_ptr,
    int64_t x_scale_ptr,
    int64_t w_scale_ptr,
    int64_t* input_args_ptr,
    int64_t* output_args_ptr,
    at::BFloat16* output_data,
    int output_offset,
    int xq_ptr_offset,
    int wq_ptr_offset,
    int x_scale_ptr_offset,
    int w_scale_ptr_offset,
    int problem_shape_buf_offset,
    int stride_buf_offset,
    int stride_size,
    int group_count,
    int problem_shape_size,
    int group_index,
    int M,
    int N,
    int K) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // Each kernel annoyingly can only set the kernel args for one group.
  // This could only be avoided with complicated memory management.
  if (idx == 0) {
    int64_t* xq_ptr_ = input_args_ptr + xq_ptr_offset;
    int64_t* wq_ptr_ = input_args_ptr + wq_ptr_offset;
    int64_t* x_scale_ptr_ = input_args_ptr + x_scale_ptr_offset;
    int64_t* w_scale_ptr_ = input_args_ptr + w_scale_ptr_offset;
    uint8_t* problem_shape_buf =
        reinterpret_cast<uint8_t*>(input_args_ptr + problem_shape_buf_offset);
    uint8_t* stride_buf =
        reinterpret_cast<uint8_t*>(input_args_ptr + stride_buf_offset);

    GroupedGemmArgs::ProblemShape::UnderlyingProblemShape* problem_shape_ptr =
        reinterpret_cast<
            GroupedGemmArgs::ProblemShape::UnderlyingProblemShape*>(
            problem_shape_buf);
    // Pass dummy configs to get Stride structure
    GroupedGemmArgs::GroupedGemmConfigs<128, 256, 128, 2, 1, 1, false>::
        StrideInputA* stride_input_A_ptr = reinterpret_cast<
            GroupedGemmArgs::GroupedGemmConfigs<128, 256, 128, 2, 1, 1, false>::
                StrideInputA*>(stride_buf);
    GroupedGemmArgs::GroupedGemmConfigs<128, 256, 128, 2, 1, 1, false>::
        StrideInputB* stride_input_B_ptr = reinterpret_cast<
            GroupedGemmArgs::GroupedGemmConfigs<128, 256, 128, 2, 1, 1, false>::
                StrideInputB*>(stride_buf + stride_size);
    GroupedGemmArgs::GroupedGemmConfigs<128, 256, 128, 2, 1, 1, false>::
        StrideOutput* stride_output_ptr = reinterpret_cast<
            GroupedGemmArgs::GroupedGemmConfigs<128, 256, 128, 2, 1, 1, false>::
                StrideOutput*>(stride_buf + (stride_size * 2));

    output_args_ptr[group_index] =
        reinterpret_cast<int64_t>(output_data + output_offset);

    // Write kernel arguments directly to memory.
    xq_ptr_[group_index] = xq_ptr;
    wq_ptr_[group_index] = wq_ptr;
    x_scale_ptr_[group_index] = x_scale_ptr;
    w_scale_ptr_[group_index] = w_scale_ptr;
    problem_shape_ptr[group_index] =
        GroupedGemmArgs::ProblemShape::UnderlyingProblemShape(M, N, K);
    stride_input_A_ptr[group_index] = cutlass::make_cute_packed_stride(
        typename GroupedGemmArgs::
            GroupedGemmConfigs<128, 256, 128, 2, 1, 1, false>::StrideInputA{},
        {M, K, 1});
    stride_input_B_ptr[group_index] = cutlass::make_cute_packed_stride(
        typename GroupedGemmArgs::
            GroupedGemmConfigs<128, 256, 128, 2, 1, 1, false>::StrideInputB{},
        {N, K, 1});
    stride_output_ptr[group_index] = cutlass::make_cute_packed_stride(
        typename GroupedGemmArgs::
            GroupedGemmConfigs<128, 256, 128, 2, 1, 1, false>::StrideOutput{},
        {M, N, 1});
  }
}

__global__ void set_dynamic_kernel_args_kernel(
    int64_t xq_ptr,
    int64_t wq_ptr,
    int64_t x_scale_ptr,
    int64_t w_scale_ptr,
    int64_t* input_args_ptr,
    int64_t* output_args_ptr,
    at::BFloat16* output_data,
    int output_offset,
    int xq_ptr_offset,
    int wq_ptr_offset,
    int x_scale_ptr_offset,
    int w_scale_ptr_offset,
    int problem_shape_buf_offset,
    int stride_buf_offset,
    int stride_size,
    int group_count,
    int problem_shape_size,
    int group_index,
    int64_t* zero_start_index_M,
    int N,
    int K) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // Each kernel annoyingly can only set the kernel args for one group.
  // This could only be avoided with complicated memory management.
  if (idx == 0) {
    int64_t* xq_ptr_ = input_args_ptr + xq_ptr_offset;
    int64_t* wq_ptr_ = input_args_ptr + wq_ptr_offset;
    int64_t* x_scale_ptr_ = input_args_ptr + x_scale_ptr_offset;
    int64_t* w_scale_ptr_ = input_args_ptr + w_scale_ptr_offset;
    uint8_t* problem_shape_buf =
        reinterpret_cast<uint8_t*>(input_args_ptr + problem_shape_buf_offset);
    uint8_t* stride_buf =
        reinterpret_cast<uint8_t*>(input_args_ptr + stride_buf_offset);

    GroupedGemmArgs::ProblemShape::UnderlyingProblemShape* problem_shape_ptr =
        reinterpret_cast<
            GroupedGemmArgs::ProblemShape::UnderlyingProblemShape*>(
            problem_shape_buf);
    // Pass dummy configs to get Stride structure
    GroupedGemmArgs::GroupedGemmConfigs<128, 256, 128, 2, 1, 1, false>::
        StrideInputA* stride_input_A_ptr = reinterpret_cast<
            GroupedGemmArgs::GroupedGemmConfigs<128, 256, 128, 2, 1, 1, false>::
                StrideInputA*>(stride_buf);
    GroupedGemmArgs::GroupedGemmConfigs<128, 256, 128, 2, 1, 1, false>::
        StrideInputB* stride_input_B_ptr = reinterpret_cast<
            GroupedGemmArgs::GroupedGemmConfigs<128, 256, 128, 2, 1, 1, false>::
                StrideInputB*>(stride_buf + stride_size);
    GroupedGemmArgs::GroupedGemmConfigs<128, 256, 128, 2, 1, 1, false>::
        StrideOutput* stride_output_ptr = reinterpret_cast<
            GroupedGemmArgs::GroupedGemmConfigs<128, 256, 128, 2, 1, 1, false>::
                StrideOutput*>(stride_buf + (stride_size * 2));

    output_args_ptr[group_index] =
        reinterpret_cast<int64_t>(output_data + output_offset);

    // Write kernel arguments directly to memory.
    xq_ptr_[group_index] = xq_ptr;
    wq_ptr_[group_index] = wq_ptr;
    x_scale_ptr_[group_index] = x_scale_ptr;
    w_scale_ptr_[group_index] = w_scale_ptr;
    problem_shape_ptr[group_index] =
        GroupedGemmArgs::ProblemShape::UnderlyingProblemShape(
            zero_start_index_M[group_index], N, K);
    stride_input_A_ptr[group_index] = cutlass::make_cute_packed_stride(
        typename GroupedGemmArgs::
            GroupedGemmConfigs<128, 256, 128, 2, 1, 1, false>::StrideInputA{},
        {zero_start_index_M[group_index], K, 1});
    stride_input_B_ptr[group_index] = cutlass::make_cute_packed_stride(
        typename GroupedGemmArgs::
            GroupedGemmConfigs<128, 256, 128, 2, 1, 1, false>::StrideInputB{},
        {N, K, 1});
    stride_output_ptr[group_index] = cutlass::make_cute_packed_stride(
        typename GroupedGemmArgs::
            GroupedGemmConfigs<128, 256, 128, 2, 1, 1, false>::StrideOutput{},
        {zero_start_index_M[group_index], N, 1});
  }
}

template <
    typename InputType,
    int TB_M,
    int TB_N,
    int TB_K,
    int TBS_M,
    int TBS_N,
    int TBS_K,
    bool PONG>
at::Tensor f8f8bf16_rowwise_grouped_impl(
    InputType XQ, // FP8
    InputType WQ, // FP8
    InputType x_scale,
    InputType w_scale,
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M) {
  int group_count;
  at::TensorOptions options;
  if constexpr (std::is_same_v<InputType, at::TensorList>) {
    group_count = XQ.size();
    options = XQ[0].options();
    TORCH_CHECK(WQ.size() == group_count);
  } else {
    group_count = XQ.size(0);
    options = XQ.options();
    TORCH_CHECK(WQ.size(0) == group_count);
  }
  if (group_count == 0) {
    return at::Tensor();
  }
  using GroupedGemmConfigs = GroupedGemmArgs::
      GroupedGemmConfigs<TB_M, TB_N, TB_K, TBS_M, TBS_N, TBS_K, PONG>;

  int64_t total_output_size = 0;
  std::vector<int64_t> output_sizes;
  output_sizes.reserve(group_count);
  at::Tensor output_args = at::empty({group_count}, options.dtype(at::kLong));

  const int64_t problem_shape_size = group_count *
      ((int64_t)sizeof(GroupedGemmArgs::ProblemShape::UnderlyingProblemShape));
  const int64_t stride_size = group_count *
      ((int64_t)sizeof(typename GroupedGemmConfigs::StrideInputA));

  // TODO: Though pointer buffer with 1000 is suitable all of our usecases, we
  // should refactor pointer buffer with better general strategy to avoid this
  // number
  at::Tensor input_args = at::empty(
      {group_count * 4 + problem_shape_size + stride_size * 3 + 1000},
      options.dtype(at::kLong));

  int xq_ptr_offset = 0;
  int wq_ptr_offset = group_count * sizeof(int64_t);
  int x_scale_ptr_offset = group_count * 2 * sizeof(int64_t);
  int w_scale_ptr_offset = group_count * 3 * sizeof(int64_t);
  int problem_shape_buf_offset = group_count * 4 * sizeof(int64_t);
  int stride_buf_offset =
      group_count * 4 * sizeof(int64_t) + problem_shape_size;

  if constexpr (std::is_same_v<InputType, at::TensorList>) {
    for (int i = 0; i < group_count; ++i) {
      const int64_t output_size = XQ[i].size(0) * WQ[i].size(0);
      total_output_size += output_size;
      output_sizes.push_back(output_size);
    }
  } else {
    // When inputs are pregrouped, output has G * M * N elements.
    total_output_size = group_count * XQ.size(1) * WQ.size(1);
    for (int i = 0; i < group_count; ++i) {
      output_sizes.push_back(XQ.size(1) * WQ.size(1));
    }
  }

  int blockSize = 256;
  int numBlocks = 1;
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  int64_t output_offset = 0;

  TORCH_CHECK(
      !zero_start_index_M.has_value() ||
          zero_start_index_M->dtype() == at::kLong,
      "zero_start_index_M must be int64.");

  // Set arguments
  for (int i = 0; i < group_count; ++i) {
    int M, N, K;
    int64_t xq_ptr, wq_ptr, x_scale_ptr, w_scale_ptr;
    // Compute buffer pointers based on input type.
    if constexpr (std::is_same_v<InputType, at::TensorList>) {
      M = XQ[i].size(0);
      N = WQ[i].size(0);
      K = XQ[i].size(1);
      TORCH_CHECK_EQ(WQ[i].size(1), K);
      // Calculate data pointer for this group.
      xq_ptr = reinterpret_cast<int64_t>(XQ[i].data_ptr());
      wq_ptr = reinterpret_cast<int64_t>(WQ[i].data_ptr());
      x_scale_ptr = reinterpret_cast<int64_t>(x_scale[i].data_ptr());
      w_scale_ptr = reinterpret_cast<int64_t>(w_scale[i].data_ptr());
    } else {
      M = XQ.size(1);
      N = WQ.size(1);
      K = XQ.size(2);
      // Calculate data pointer for this group.
      xq_ptr = reinterpret_cast<int64_t>(XQ.data_ptr()) +
          (i * M * K * sizeof(GroupedGemmArgs::ElementInputA));
      wq_ptr = reinterpret_cast<int64_t>(WQ.data_ptr()) +
          (i * N * K * sizeof(GroupedGemmArgs::ElementInputB));
      x_scale_ptr = reinterpret_cast<int64_t>(x_scale.data_ptr()) +
          (i * M * sizeof(GroupedGemmArgs::ElementAccumulator));
      w_scale_ptr = reinterpret_cast<int64_t>(w_scale.data_ptr()) +
          (i * N * sizeof(GroupedGemmArgs::ElementAccumulator));
    }
    if (zero_start_index_M.has_value() == true) {
      set_dynamic_kernel_args_kernel<<<numBlocks, blockSize, 0, stream>>>(
          xq_ptr,
          wq_ptr,
          x_scale_ptr,
          w_scale_ptr,
          input_args.data_ptr<int64_t>(),
          output_args.data_ptr<int64_t>(),
          output.data_ptr<at::BFloat16>(),
          output_offset,
          xq_ptr_offset,
          wq_ptr_offset,
          x_scale_ptr_offset,
          w_scale_ptr_offset,
          problem_shape_buf_offset,
          stride_buf_offset,
          stride_size,
          group_count,
          problem_shape_size,
          i,
          reinterpret_cast<int64_t*>(zero_start_index_M.value().data_ptr()),
          N,
          K);
    } else {
      set_kernel_args_kernel<<<numBlocks, blockSize, 0, stream>>>(
          xq_ptr,
          wq_ptr,
          x_scale_ptr,
          w_scale_ptr,
          input_args.data_ptr<int64_t>(),
          output_args.data_ptr<int64_t>(),
          output.data_ptr<at::BFloat16>(),
          output_offset,
          xq_ptr_offset,
          wq_ptr_offset,
          x_scale_ptr_offset,
          w_scale_ptr_offset,
          problem_shape_buf_offset,
          stride_buf_offset,
          stride_size,
          group_count,
          problem_shape_size,
          i,
          M,
          N,
          K);
    }
    output_offset += output_sizes[i];
  }

  int64_t* output_ptr = output_args.data_ptr<int64_t>();
  int64_t* xq_ptr = input_args.data_ptr<int64_t>() + xq_ptr_offset;
  int64_t* wq_ptr = input_args.data_ptr<int64_t>() + wq_ptr_offset;
  int64_t* x_scale_ptr = input_args.data_ptr<int64_t>() + x_scale_ptr_offset;
  int64_t* w_scale_ptr = input_args.data_ptr<int64_t>() + w_scale_ptr_offset;
  uint8_t* problem_shape_buf = reinterpret_cast<uint8_t*>(
      input_args.data_ptr<int64_t>() + problem_shape_buf_offset);
  uint8_t* stride_buf = reinterpret_cast<uint8_t*>(
      input_args.data_ptr<int64_t>() + stride_buf_offset);

  GroupedGemmArgs::ProblemShape::UnderlyingProblemShape* problem_shape_ptr =
      reinterpret_cast<GroupedGemmArgs::ProblemShape::UnderlyingProblemShape*>(
          problem_shape_buf);
  typename GroupedGemmConfigs::StrideInputA* stride_input_A_ptr =
      reinterpret_cast<typename GroupedGemmConfigs::StrideInputA*>(stride_buf);
  typename GroupedGemmConfigs::StrideInputB* stride_input_B_ptr =
      reinterpret_cast<typename GroupedGemmConfigs::StrideInputB*>(
          stride_buf + stride_size);
  typename GroupedGemmConfigs::StrideOutput* stride_output_ptr =
      reinterpret_cast<typename GroupedGemmConfigs::StrideOutput*>(
          stride_buf + (stride_size * 2));

  typename GroupedGemmConfigs::Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {group_count, problem_shape_ptr, nullptr},
      {reinterpret_cast<const GroupedGemmArgs::ElementInputA**>(xq_ptr),
       stride_input_A_ptr,
       reinterpret_cast<const GroupedGemmArgs::ElementInputB**>(wq_ptr),
       stride_input_B_ptr},
      {{},
       reinterpret_cast<const GroupedGemmArgs::ElementOutput**>(output_ptr),
       stride_output_ptr,
       reinterpret_cast<GroupedGemmArgs::ElementOutput**>(output_ptr),
       stride_output_ptr}};

  arguments.epilogue.thread = {
      {reinterpret_cast<const GroupedGemmArgs::ElementComputeEpilogue**>(
          x_scale_ptr)}, // x_scale
      // compute_0
      {
          {reinterpret_cast<const GroupedGemmArgs::ElementComputeEpilogue**>(
              w_scale_ptr)}, // w_scale
          {}, // Accumulator
          {} // Multiplies
      },
      {}, // Multiplies
  };

  typename GroupedGemmConfigs::Gemm gemm;

  // Using the arguments, query for extra workspace required for matrix
  // multiplication computation
  size_t workspace_size =
      GroupedGemmConfigs::Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  at::Tensor workspace =
      at::empty(workspace_size, XQ[0].options().dtype(at::kByte));

  // Check the problem size is supported or not
  cutlass::Status status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot implement");
  }

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm.initialize(
      arguments, reinterpret_cast<uint8_t*>(workspace.data_ptr()));
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

  return output;
}

// FP8 Tensorwise grouped cutlass kernel dispatch.
template <typename InputType>
at::Tensor dispatch_fp8_grouped_kernel(
    InputType XQ, // FP8
    InputType WQ, // FP8
    InputType x_scale,
    InputType w_scale,
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M) {
  KernelMode kernel = get_grouped_kernel_mode(XQ, WQ);
  if (kernel == KernelMode::Small) {
    return f8f8bf16_rowwise_grouped_impl<
        InputType,
        64,
        128,
        128,
        2,
        1,
        1,
        true>(XQ, WQ, x_scale, w_scale, output, zero_start_index_M);
  } else if (kernel == KernelMode::Large) {
    return f8f8bf16_rowwise_grouped_impl<
        InputType,
        128,
        256,
        128,
        2,
        1,
        1,
        false>(XQ, WQ, x_scale, w_scale, output, zero_start_index_M);
  } else {
    return f8f8bf16_rowwise_grouped_impl<
        InputType,
        128,
        256,
        64,
        2,
        1,
        1,
        false>(XQ, WQ, x_scale, w_scale, output, zero_start_index_M);
  }
}

template <typename OutputType>
OutputType _f8f8bf16_rowwise_grouped(
    at::TensorList XQ, // FP8
    at::TensorList WQ, // FP8
    at::TensorList x_scale,
    at::TensorList w_scale,
    std::optional<OutputType> output = std::nullopt) {
  at::Tensor Y;
  int group_count = XQ.size();
  std::vector<int64_t> output_sizes;
  if (output.has_value()) {
    // Handle initialization check for output list.
    if constexpr (std::is_same_v<OutputType, std::vector<at::Tensor>>) {
      std::vector<at::Tensor> output_;
      output_ = output.value();
      TORCH_CHECK(
          output_.size() == group_count,
          "Output and input must have same number of groups.");
      // Check that output shapes are correct.
      for (int i = 0; i < group_count; i++) {
        int M = XQ[i].size(0);
        int N = WQ[i].size(0);
        int out_M = output_[i].size(0);
        int out_N = output_[i].size(1);
        TORCH_CHECK(
            M == out_M && N == out_N,
            "Output tensors do not have the expected shape.");
        TORCH_CHECK(
            output_[i].dtype() == at::kBFloat16,
            "Output dtype must be bfloat16.");
        output_sizes.push_back(out_M * out_N);
      }
      Y = at::stack(output.value(), 0);
      // Handle initialization check for output tensor.
    } else {
      at::Tensor output_ = output.value();
      // Check that output shapes are correct.
      int N = WQ[0].size(0);
      for (int i = 0; i < group_count; i++) {
        TORCH_CHECK(
            WQ[i].size(0) == N,
            "N must be static for single output tensor mode.");
        TORCH_CHECK(
            output_.size(0) == group_count &&
                output_.size(1) == XQ[i].size(0) && output_.size(2) == N,
            "Output must be shape [G, M, N]");
      }
      TORCH_CHECK(output_.dtype() == at::kBFloat16, "Output must be bfloat16.")
      Y = output_.view(-1);
    }
  } else {
    int64_t total_output_size = 0;
    for (int i = 0; i < group_count; ++i) {
      const int64_t output_size = XQ[i].size(0) * WQ[i].size(0);
      total_output_size += output_size;
      output_sizes.push_back(output_size);
    }
    Y = at::empty(total_output_size, XQ[0].options().dtype(at::kBFloat16));
  }

  // Run kernel.
  at::Tensor g_out = dispatch_fp8_grouped_kernel<at::TensorList>(
      XQ, WQ, x_scale, w_scale, Y, std::nullopt);

  // Return proper output type.
  if constexpr (std::is_same_v<OutputType, std::vector<at::Tensor>>) {
    // Return grouped view of output.
    std::vector<at::Tensor> output_group = g_out.split(output_sizes);
    for (int i = 0; i < group_count; ++i) {
      output_group[i] = output_group[i].view({XQ[i].size(0), WQ[i].size(0)});
    }
    return output_group;
  } else {
    // When stacked we assume N is fixed.
    return g_out.view({-1, WQ[0].size(0)});
  }
}

std::vector<at::Tensor> f8f8bf16_rowwise_grouped(
    at::TensorList XQ, // FP8
    at::TensorList WQ, // FP8
    at::TensorList x_scale,
    at::TensorList w_scale,
    std::optional<std::vector<at::Tensor>> output = std::nullopt) {
  return _f8f8bf16_rowwise_grouped<std::vector<at::Tensor>>(
      XQ, WQ, x_scale, w_scale);
}

at::Tensor f8f8bf16_rowwise_grouped_stacked(
    at::TensorList XQ, // FP8
    at::TensorList WQ, // FP8
    at::TensorList x_scale,
    at::TensorList w_scale,
    std::optional<at::Tensor> output = std::nullopt) {
  return _f8f8bf16_rowwise_grouped<at::Tensor>(XQ, WQ, x_scale, w_scale);
}

at::Tensor f8f8bf16_rowwise_grouped_dynamic(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor zero_start_index_M,
    bool zeroing_output_tensor = true) {
  at::Tensor Y;
  int group_count = XQ.size(0);
  int M = XQ.size(1);
  int N = WQ.size(1);
  int K = XQ.size(0);
  int total_output_size = group_count * M * N;
  if (zeroing_output_tensor) {
    Y = at::zeros(total_output_size, XQ.options().dtype(at::kBFloat16));
  } else {
    Y = at::empty(total_output_size, XQ.options().dtype(at::kBFloat16));
  }

  // Return continuous view of output.
  at::Tensor output = dispatch_fp8_grouped_kernel<at::Tensor>(
      XQ, WQ, x_scale, w_scale, Y, zero_start_index_M);
  // View as proper shape.
  return output.view({-1, M, N});
}

#else

std::vector<at::Tensor> f8f8bf16_rowwise_grouped(
    at::TensorList XQ, // FP8
    at::TensorList WQ, // FP8
    at::TensorList x_scale,
    at::TensorList w_scale,
    std::optional<std::vector<at::Tensor>> output = std::nullopt) {
  throw std::runtime_error(
      "CUDA version is older than 12.0"); // requires CUDA>=12
}

at::Tensor f8f8bf16_rowwise_grouped_stacked(
    at::TensorList XQ, // FP8
    at::TensorList WQ, // FP8
    at::TensorList x_scale,
    at::TensorList w_scale,
    std::optional<at::Tensor> output = std::nullopt) {
  throw std::runtime_error(
      "CUDA version is older than 12.0"); // requires CUDA>=12
}

at::Tensor f8f8bf16_rowwise_grouped_dynamic(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor zero_start_index_M,
    bool zeroing_output_tensor = true) {
  throw std::runtime_error(
      "CUDA version is older than 12.0"); // requires CUDA>=12
}

#endif

} // namespace fbgemm_gpu
