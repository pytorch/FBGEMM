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
          cutlass::epilogue::fusion::LinearCombination<
              ElementOutput,
              ElementAccumulator>>::CollectiveOp;
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

template <
    int TB_M,
    int TB_N,
    int TB_K,
    int TBS_M,
    int TBS_N,
    int TBS_K,
    bool PONG,
    bool FAST_ACCUM>
std::vector<at::Tensor> f8f8bf16_grouped_impl(
    const std::vector<at::Tensor>& XQ, // FP8
    const std::vector<at::Tensor>& WQ, // FP8
    const std::vector<at::Tensor>& scale) {
  int problem_count = XQ.size();
  TORCH_CHECK(WQ.size() == problem_count);
  if (problem_count == 0) {
    return std::vector<at::Tensor>();
  }
  using GroupedGemmConfigs = GroupedGemmArgs::
      GroupedGemmConfigs<TB_M, TB_N, TB_K, TBS_M, TBS_N, TBS_K, PONG>;

  constexpr int AlignmentA =
      128 /
      cutlass::sizeof_bits<
          GroupedGemmArgs::ElementInputA>::value; // Alignment of A matrix
                                                  // in units of elements
                                                  // (up to 16 bytes)

  constexpr int AlignmentB =
      128 /
      cutlass::sizeof_bits<
          GroupedGemmArgs::ElementInputB>::value; // Alignment of B matrix
                                                  // in units of elements
                                                  // (up to 16 bytes)

  constexpr int AlignmentD =
      128 /
      cutlass::sizeof_bits<
          GroupedGemmArgs::ElementOutput>::value; // Alignment of C matrix
                                                  // in units of elements
                                                  // (up to 16 bytes)

  int64_t output_offset = 0;
  int64_t total_output_size = 0;
  std::vector<int64_t> output_sizes;
  output_sizes.reserve(problem_count);
  at::Tensor output_args = at::empty(
      {problem_count},
      at::TensorOptions().dtype(at::kLong).pinned_memory(true));
  int64_t* output_ptr = output_args.data_ptr<int64_t>();

  const int64_t problem_shape_size = problem_count *
      ((int64_t)sizeof(GroupedGemmArgs::ProblemShape::UnderlyingProblemShape));
  const int64_t stride_size = problem_count *
      ((int64_t)sizeof(typename GroupedGemmConfigs::StrideInputA));

  at::Tensor input_args = at::empty(
      {problem_count * 3 + problem_shape_size + stride_size * 3},
      at::TensorOptions().dtype(at::kLong).pinned_memory(true));

  int64_t* xq_ptr = input_args.data_ptr<int64_t>();
  int64_t* wq_ptr =
      input_args.data_ptr<int64_t>() + (problem_count * sizeof(int64_t));
  int64_t* scale_ptr =
      input_args.data_ptr<int64_t>() + (problem_count * 2 * sizeof(int64_t));
  uint8_t* problem_shape_buf = reinterpret_cast<uint8_t*>(
      input_args.data_ptr<int64_t>() + (problem_count * 3 * sizeof(int64_t)));
  uint8_t* stride_buf = reinterpret_cast<uint8_t*>(
      input_args.data_ptr<int64_t>() + (problem_count * 3 * sizeof(int64_t)) +
      problem_shape_size);

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

  for (int i = 0; i < problem_count; ++i) {
    const int64_t output_size = XQ[i].size(0) * WQ[i].size(0);
    total_output_size += output_size;
    output_sizes.push_back(output_size);
  }

  at::Tensor output_tensor =
      at::empty(total_output_size, XQ[0].options().dtype(at::kBFloat16));
  at::BFloat16* output_data = output_tensor.data_ptr<at::BFloat16>();

  // Set arguments
  for (int i = 0; i < problem_count; ++i) {
    int m = XQ[i].size(0);
    int n = WQ[i].size(0);
    int k = XQ[i].size(1);
    TORCH_CHECK_EQ(WQ[i].size(1), k);

    output_ptr[i] = reinterpret_cast<int64_t>(output_data + output_offset);
    output_offset += output_sizes[i];

    xq_ptr[i] = reinterpret_cast<int64_t>(XQ[i].data_ptr<at::Float8_e4m3fn>());
    wq_ptr[i] = reinterpret_cast<int64_t>(WQ[i].data_ptr<at::Float8_e4m3fn>());
    scale_ptr[i] = reinterpret_cast<int64_t>(
        scale[i].data_ptr<GroupedGemmArgs::ElementAccumulator>());
    problem_shape_ptr[i] =
        GroupedGemmArgs::ProblemShape::UnderlyingProblemShape(m, n, k);
    stride_input_A_ptr[i] = cutlass::make_cute_packed_stride(
        typename GroupedGemmConfigs::StrideInputA{}, {m, k, 1});
    stride_input_B_ptr[i] = cutlass::make_cute_packed_stride(
        typename GroupedGemmConfigs::StrideInputB{}, {n, k, 1});
    stride_output_ptr[i] = cutlass::make_cute_packed_stride(
        typename GroupedGemmConfigs::StrideOutput{}, {m, n, 1});
  }

  // Allocate input args memory on the GPU
  size_t input_args_size = input_args.numel() * sizeof(int64_t);
  at::Tensor d_input_args = at::empty(
      {problem_count * 3 + problem_shape_size + stride_size * 3},
      at::TensorOptions().dtype(at::kLong).device(at::kCUDA));

  // Allocate output args memory on the GPU
  size_t output_args_size = output_args.numel() * sizeof(int64_t);
  at::Tensor d_output_args = at::empty(
      {problem_count}, at::TensorOptions().dtype(at::kLong).device(at::kCUDA));

  // Copy data from CPU to GPU asynchronously
  cudaMemcpyAsync(
      d_input_args.data_ptr(),
      input_args.data_ptr<int64_t>(),
      input_args_size,
      cudaMemcpyHostToDevice,
      at::cuda::getCurrentCUDAStream());

  cudaMemcpyAsync(
      d_output_args.data_ptr(),
      output_args.data_ptr<int64_t>(),
      output_args_size,
      cudaMemcpyHostToDevice,
      at::cuda::getCurrentCUDAStream());

  output_ptr = output_args.data_ptr<int64_t>();
  xq_ptr = input_args.data_ptr<int64_t>();
  wq_ptr = input_args.data_ptr<int64_t>() + (problem_count * sizeof(int64_t));
  scale_ptr =
      input_args.data_ptr<int64_t>() + (problem_count * 2 * sizeof(int64_t));

  problem_shape_buf = reinterpret_cast<uint8_t*>(
      input_args.data_ptr<int64_t>() + (problem_count * 3 * sizeof(int64_t)));
  problem_shape_ptr =
      reinterpret_cast<GroupedGemmArgs::ProblemShape::UnderlyingProblemShape*>(
          problem_shape_buf);
  stride_buf = reinterpret_cast<uint8_t*>(
      input_args.data_ptr<int64_t>() + (problem_count * 3 * sizeof(int64_t)) +
      problem_shape_size);
  stride_input_A_ptr =
      reinterpret_cast<typename GroupedGemmConfigs::StrideInputA*>(stride_buf);
  stride_input_B_ptr =
      reinterpret_cast<typename GroupedGemmConfigs::StrideInputB*>(
          stride_buf + stride_size);
  stride_output_ptr =
      reinterpret_cast<typename GroupedGemmConfigs::StrideOutput*>(
          stride_buf + (stride_size * 2));

  typename GroupedGemmConfigs::Gemm::Arguments arguments;
  decltype(arguments.epilogue.thread) fusion_args;
  fusion_args.alpha_ptr_array =
      reinterpret_cast<const GroupedGemmArgs::ElementAccumulator**>(scale_ptr);
  fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 1};

  arguments = typename GroupedGemmConfigs::Gemm::Arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {problem_count, problem_shape_ptr, nullptr},
      {reinterpret_cast<const GroupedGemmArgs::ElementInputA**>(xq_ptr),
       stride_input_A_ptr,
       reinterpret_cast<const GroupedGemmArgs::ElementInputB**>(wq_ptr),
       stride_input_B_ptr},
      {fusion_args,
       reinterpret_cast<const GroupedGemmArgs::ElementOutput**>(output_ptr),
       stride_output_ptr,
       reinterpret_cast<GroupedGemmArgs::ElementOutput**>(output_ptr),
       stride_output_ptr}};

  typename GroupedGemmConfigs::Gemm gemm;

  // Using the arguments, query for extra workspace required for matrix
  // multiplication computation
  size_t workspace_size =
      GroupedGemmConfigs::Gemm::get_workspace_size(arguments);

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

  std::vector<at::Tensor> output_group = output_tensor.split(output_sizes);
  for (int i = 0; i < problem_count; ++i) {
    output_group[i] = output_group[i].view({XQ[i].size(0), WQ[i].size(0)});
  }
  return output_group;
}

// FP8 Tensorwise grouped cutlass kernel dispatch.
template <bool FastAccum>
std::vector<at::Tensor> dispatch_fp8_grouped_kernel(
    const std::vector<at::Tensor>& xq_group, // FP8
    const std::vector<at::Tensor>& wq_group, // FP8
    const std::vector<at::Tensor>& scale) {
  KernelMode kernel = get_grouped_kernel_mode(xq_group, wq_group);
  if (kernel == KernelMode::Small) {
    return f8f8bf16_grouped_impl<64, 128, 128, 2, 1, 1, true, FastAccum>(
        xq_group, wq_group, scale);
  } else if (kernel == KernelMode::Large) {
    return f8f8bf16_grouped_impl<128, 128, 128, 2, 1, 1, true, FastAccum>(
        xq_group, wq_group, scale);
  } else {
    return f8f8bf16_grouped_impl<128, 128, 128, 1, 2, 1, true, FastAccum>(
        xq_group, wq_group, scale);
  }
}

std::vector<at::Tensor> f8f8bf16_grouped(
    const std::vector<at::Tensor>& xq_group, // FP8
    const std::vector<at::Tensor>& wq_group, // FP8
    const std::vector<at::Tensor>& scale,
    bool use_fast_accum) {
  if (use_fast_accum) {
    return dispatch_fp8_grouped_kernel<true>(xq_group, wq_group, scale);
  } else {
    return dispatch_fp8_grouped_kernel<false>(xq_group, wq_group, scale);
  }
}

#else

std::vector<at::Tensor> f8f8bf16_grouped(
    const std::vector<at::Tensor>& xq_group, // FP8
    const std::vector<at::Tensor>& wq_group, // FP8
    const std::vector<at::Tensor>& scale,
    bool use_fast_accum) {
  throw std::runtime_error(
      "CUDA version is older than 12.0"); // requires CUDA>=12
}

#endif

} // namespace fbgemm_gpu
