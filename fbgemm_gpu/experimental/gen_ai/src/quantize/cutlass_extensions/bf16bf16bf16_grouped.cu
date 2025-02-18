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

namespace GroupedGemmBF16Args {
using ProblemShape =
    cutlass::gemm::GroupProblemShape<cute::Shape<int, int, int>>;
using ElementInputA = cutlass::bfloat16_t;
using ElementInputB = cutlass::bfloat16_t;
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
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative;
  using PongSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong;
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
} // namespace GroupedGemmBF16Args

__global__ void set_dynamic_kernel_args_kernel(
    GroupedGemmBF16Args::ElementInputA* x_ptr,
    GroupedGemmBF16Args::ElementInputB* w_ptr,
    int64_t* input_args_ptr,
    int64_t* output_args_ptr,
    GroupedGemmBF16Args::ElementOutput* output_data,
    int x_ptr_offset,
    int w_ptr_offset,
    int problem_shape_buf_offset,
    int stride_buf_offset,
    int stride_size,
    int problem_count,
    int problem_shape_size,
    int64_t* zero_start_index_M,
    int M,
    int N,
    int K) {
  int group_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (group_index < problem_count) {
    int64_t* x_ptr_ = input_args_ptr + x_ptr_offset;
    int64_t* w_ptr_ = input_args_ptr + w_ptr_offset;
    uint8_t* problem_shape_buf =
        reinterpret_cast<uint8_t*>(input_args_ptr + problem_shape_buf_offset);
    uint8_t* stride_buf =
        reinterpret_cast<uint8_t*>(input_args_ptr + stride_buf_offset);

    GroupedGemmBF16Args::ProblemShape::UnderlyingProblemShape*
        problem_shape_ptr = reinterpret_cast<
            GroupedGemmBF16Args::ProblemShape::UnderlyingProblemShape*>(
            problem_shape_buf);
    // Pass dummy configs to get Stride structure
    GroupedGemmBF16Args::GroupedGemmConfigs<128, 256, 64, 2, 1, 1, false>::
        StrideInputA* stride_input_A_ptr = reinterpret_cast<
            GroupedGemmBF16Args::
                GroupedGemmConfigs<128, 256, 64, 2, 1, 1, false>::
                    StrideInputA*>(stride_buf);
    GroupedGemmBF16Args::GroupedGemmConfigs<128, 256, 64, 2, 1, 1, false>::
        StrideInputB* stride_input_B_ptr = reinterpret_cast<
            GroupedGemmBF16Args::
                GroupedGemmConfigs<128, 256, 64, 2, 1, 1, false>::
                    StrideInputB*>(stride_buf + stride_size);
    GroupedGemmBF16Args::GroupedGemmConfigs<128, 256, 64, 2, 1, 1, false>::
        StrideOutput* stride_output_ptr = reinterpret_cast<
            GroupedGemmBF16Args::
                GroupedGemmConfigs<128, 256, 64, 2, 1, 1, false>::
                    StrideOutput*>(stride_buf + (stride_size * 2));

    output_args_ptr[group_index] =
        reinterpret_cast<int64_t>(output_data + (group_index * M * N));

    // Write kernel arguments directly to memory.
    x_ptr_[group_index] =
        reinterpret_cast<int64_t>(x_ptr + (group_index * M * K));
    w_ptr_[group_index] =
        reinterpret_cast<int64_t>(w_ptr + (group_index * N * K));
    problem_shape_ptr[group_index] =
        GroupedGemmBF16Args::ProblemShape::UnderlyingProblemShape(
            zero_start_index_M[group_index], N, K);
    stride_input_A_ptr[group_index] = cutlass::make_cute_packed_stride(
        typename GroupedGemmBF16Args::
            GroupedGemmConfigs<128, 256, 64, 2, 1, 1, false>::StrideInputA{},
        {zero_start_index_M[group_index], K, 1});
    stride_input_B_ptr[group_index] = cutlass::make_cute_packed_stride(
        typename GroupedGemmBF16Args::
            GroupedGemmConfigs<128, 256, 64, 2, 1, 1, false>::StrideInputB{},
        {N, K, 1});
    stride_output_ptr[group_index] = cutlass::make_cute_packed_stride(
        typename GroupedGemmBF16Args::
            GroupedGemmConfigs<128, 256, 64, 2, 1, 1, false>::StrideOutput{},
        {zero_start_index_M[group_index], N, 1});
  }
}

__global__ void set_static_kernel_args_kernel(
    GroupedGemmBF16Args::ElementInputA* x_ptr,
    GroupedGemmBF16Args::ElementInputB* w_ptr,
    int64_t* input_args_ptr,
    int64_t* output_args_ptr,
    GroupedGemmBF16Args::ElementOutput* output_data,
    int x_ptr_offset,
    int w_ptr_offset,
    int problem_shape_buf_offset,
    int stride_buf_offset,
    int stride_size,
    int problem_count,
    int problem_shape_size,
    int group_index,
    int M,
    int N,
    int K) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // We only set one group's information per kernel launch.
  if (idx == 0) {
    int64_t* x_ptr_ = input_args_ptr + x_ptr_offset;
    int64_t* w_ptr_ = input_args_ptr + w_ptr_offset;
    uint8_t* problem_shape_buf =
        reinterpret_cast<uint8_t*>(input_args_ptr + problem_shape_buf_offset);
    uint8_t* stride_buf =
        reinterpret_cast<uint8_t*>(input_args_ptr + stride_buf_offset);

    GroupedGemmBF16Args::ProblemShape::UnderlyingProblemShape*
        problem_shape_ptr = reinterpret_cast<
            GroupedGemmBF16Args::ProblemShape::UnderlyingProblemShape*>(
            problem_shape_buf);
    // Pass dummy configs to get Stride structure
    GroupedGemmBF16Args::GroupedGemmConfigs<128, 256, 64, 2, 1, 1, false>::
        StrideInputA* stride_input_A_ptr = reinterpret_cast<
            GroupedGemmBF16Args::
                GroupedGemmConfigs<128, 256, 64, 2, 1, 1, false>::
                    StrideInputA*>(stride_buf);
    GroupedGemmBF16Args::GroupedGemmConfigs<128, 256, 64, 2, 1, 1, false>::
        StrideInputB* stride_input_B_ptr = reinterpret_cast<
            GroupedGemmBF16Args::
                GroupedGemmConfigs<128, 256, 64, 2, 1, 1, false>::
                    StrideInputB*>(stride_buf + stride_size);
    GroupedGemmBF16Args::GroupedGemmConfigs<128, 256, 64, 2, 1, 1, false>::
        StrideOutput* stride_output_ptr = reinterpret_cast<
            GroupedGemmBF16Args::
                GroupedGemmConfigs<128, 256, 64, 2, 1, 1, false>::
                    StrideOutput*>(stride_buf + (stride_size * 2));

    output_args_ptr[group_index] = reinterpret_cast<int64_t>(output_data);

    // Write kernel arguments directly to memory.
    x_ptr_[group_index] = reinterpret_cast<int64_t>(x_ptr);
    w_ptr_[group_index] = reinterpret_cast<int64_t>(w_ptr);
    problem_shape_ptr[group_index] =
        GroupedGemmBF16Args::ProblemShape::UnderlyingProblemShape(M, N, K);
    stride_input_A_ptr[group_index] = cutlass::make_cute_packed_stride(
        typename GroupedGemmBF16Args::
            GroupedGemmConfigs<128, 256, 64, 2, 1, 1, false>::StrideInputA{},
        {M, K, 1});
    stride_input_B_ptr[group_index] = cutlass::make_cute_packed_stride(
        typename GroupedGemmBF16Args::
            GroupedGemmConfigs<128, 256, 64, 2, 1, 1, false>::StrideInputB{},
        {N, K, 1});
    stride_output_ptr[group_index] = cutlass::make_cute_packed_stride(
        typename GroupedGemmBF16Args::
            GroupedGemmConfigs<128, 256, 64, 2, 1, 1, false>::StrideOutput{},
        {M, N, 1});
  }
}

template <
    int TB_M,
    int TB_N,
    int TB_K,
    int TBS_M,
    int TBS_N,
    int TBS_K,
    bool PONG>
std::vector<at::Tensor> bf16bf16bf16_grouped_impl(
    at::TensorList X, // BF16
    at::TensorList W, // BF16
    std::vector<at::Tensor> output_tensor,
    std::optional<at::Tensor> zero_start_index_M) {
  int problem_count = X.size();
  TORCH_CHECK(W.size() == problem_count);
  TORCH_CHECK(
      !zero_start_index_M.has_value() ||
      zero_start_index_M->size(0) == problem_count);
  if (problem_count == 0) {
    return {};
  }
  using GroupedGemmConfigs = GroupedGemmBF16Args::
      GroupedGemmConfigs<TB_M, TB_N, TB_K, TBS_M, TBS_N, TBS_K, PONG>;

  at::Tensor output_args =
      at::empty({problem_count}, X[0].options().dtype(at::kLong));

  const int64_t problem_shape_size = problem_count *
      ((int64_t)sizeof(
          GroupedGemmBF16Args::ProblemShape::UnderlyingProblemShape));
  const int64_t stride_size = problem_count *
      ((int64_t)sizeof(typename GroupedGemmConfigs::StrideInputA));

  at::Tensor input_args = at::empty(
      {problem_count * 3 + problem_shape_size + stride_size * 3},
      X[0].options().dtype(at::kLong));

  int x_ptr_offset = 0;
  int w_ptr_offset = problem_count * sizeof(int64_t);
  int problem_shape_buf_offset = problem_count * 2 * sizeof(int64_t);
  int stride_buf_offset =
      problem_count * 2 * sizeof(int64_t) + problem_shape_size;

  TORCH_CHECK(
      !zero_start_index_M.has_value() ||
          zero_start_index_M->dtype() == at::kLong,
      "zero_start_index_M must be int64.");

  // Set arguments
  // Here we support two methods for initializing arguments. If
  // zero_start_index_M is provided, we assume that M is dynamic and N and K are
  // fixed. This is useful for cuda graphs. If it is not provided we run in
  // eager mode.
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  if (zero_start_index_M.has_value()) {
    // When zero_start_M is provided, we run in dynamic shape mode.
    int M = X[0].size(0);
    int N = W[0].size(0);
    int K = X[0].size(1);
    // Make sure that inputs are allocated in sequential memory as required by
    // this mode.
    for (int i = 1; i < problem_count; i++) {
      // Check that all inputs are allocated directly following preceding input.
      TORCH_CHECK(
          X[i].data_ptr() ==
              (reinterpret_cast<GroupedGemmBF16Args::ElementInputA*>(
                   X[i - 1].data_ptr()) +
               (M * K)),
          "Inputs must be sequential in memory to support dynamic M, but X is not.");
      TORCH_CHECK(
          W[i].data_ptr() ==
              (reinterpret_cast<GroupedGemmBF16Args::ElementInputB*>(
                   W[i - 1].data_ptr()) +
               (N * K)),
          "Inputs must be sequential in memory to support dynamic M, but W is not.");
      TORCH_CHECK(
          output_tensor[i].data_ptr() ==
              (reinterpret_cast<GroupedGemmBF16Args::ElementOutput*>(
                   output_tensor[i - 1].data_ptr()) +
               (M * N)),
          "Inputs must be sequential in memory to support dynamic M, but output is not.");
    }
    int blockSize = std::min(1024, problem_count);
    int numBlocks = (problem_count + blockSize - 1) / blockSize;
    set_dynamic_kernel_args_kernel<<<numBlocks, blockSize, 0, stream>>>(
        reinterpret_cast<GroupedGemmBF16Args::ElementInputA*>(X[0].data_ptr()),
        reinterpret_cast<GroupedGemmBF16Args::ElementInputB*>(W[0].data_ptr()),
        input_args.data_ptr<int64_t>(),
        output_args.data_ptr<int64_t>(),
        reinterpret_cast<GroupedGemmBF16Args::ElementOutput*>(
            output_tensor[0].data_ptr()),
        x_ptr_offset,
        w_ptr_offset,
        problem_shape_buf_offset,
        stride_buf_offset,
        stride_size,
        problem_count,
        problem_shape_size,
        reinterpret_cast<int64_t*>(zero_start_index_M.value().data_ptr()),
        M,
        N,
        K);
  } else {
    // Otherwise run in static mode, which can support arbitrary N and K.
    int blockSize = 256;
    int numBlocks = 1;
    // Iterate over groups and launch one kernel to set each up.
    for (int i = 0; i < problem_count; i++) {
      int M = X[i].size(0);
      int N = W[i].size(0);
      int K = X[i].size(1);
      set_static_kernel_args_kernel<<<numBlocks, blockSize, 0, stream>>>(
          reinterpret_cast<GroupedGemmBF16Args::ElementInputA*>(
              X[i].data_ptr()),
          reinterpret_cast<GroupedGemmBF16Args::ElementInputB*>(
              W[i].data_ptr()),
          input_args.data_ptr<int64_t>(),
          output_args.data_ptr<int64_t>(),
          reinterpret_cast<GroupedGemmBF16Args::ElementOutput*>(
              output_tensor[i].data_ptr()),
          x_ptr_offset,
          w_ptr_offset,
          problem_shape_buf_offset,
          stride_buf_offset,
          stride_size,
          problem_count,
          problem_shape_size,
          i,
          M,
          N,
          K);
    }
  }

  // Get appropriate pointers to each component of the kernel args.
  int64_t* output_ptr = output_args.data_ptr<int64_t>();
  int64_t* x_ptr = input_args.data_ptr<int64_t>() + x_ptr_offset;
  int64_t* w_ptr = input_args.data_ptr<int64_t>() + w_ptr_offset;
  uint8_t* problem_shape_buf = reinterpret_cast<uint8_t*>(
      input_args.data_ptr<int64_t>() + problem_shape_buf_offset);
  uint8_t* stride_buf = reinterpret_cast<uint8_t*>(
      input_args.data_ptr<int64_t>() + stride_buf_offset);

  GroupedGemmBF16Args::ProblemShape::UnderlyingProblemShape* problem_shape_ptr =
      reinterpret_cast<
          GroupedGemmBF16Args::ProblemShape::UnderlyingProblemShape*>(
          problem_shape_buf);
  typename GroupedGemmConfigs::StrideInputA* stride_input_A_ptr =
      reinterpret_cast<typename GroupedGemmConfigs::StrideInputA*>(stride_buf);
  typename GroupedGemmConfigs::StrideInputB* stride_input_B_ptr =
      reinterpret_cast<typename GroupedGemmConfigs::StrideInputB*>(
          stride_buf + stride_size);
  typename GroupedGemmConfigs::StrideOutput* stride_output_ptr =
      reinterpret_cast<typename GroupedGemmConfigs::StrideOutput*>(
          stride_buf + (stride_size * 2));

  typename GroupedGemmConfigs::Gemm::Arguments arguments;
  decltype(arguments.epilogue.thread) fusion_args;
  fusion_args.alpha = 1.0;
  fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 0};

  arguments = typename GroupedGemmConfigs::Gemm::Arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {problem_count, problem_shape_ptr, nullptr},
      {reinterpret_cast<const GroupedGemmBF16Args::ElementInputA**>(x_ptr),
       stride_input_A_ptr,
       reinterpret_cast<const GroupedGemmBF16Args::ElementInputB**>(w_ptr),
       stride_input_B_ptr},
      {fusion_args,
       reinterpret_cast<const GroupedGemmBF16Args::ElementOutput**>(output_ptr),
       stride_output_ptr,
       reinterpret_cast<GroupedGemmBF16Args::ElementOutput**>(output_ptr),
       stride_output_ptr}};

  typename GroupedGemmConfigs::Gemm gemm;

  // Using the arguments, query for extra workspace required for matrix
  // multiplication computation
  size_t workspace_size =
      GroupedGemmConfigs::Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  at::Tensor workspace =
      at::empty(workspace_size, X[0].options().dtype(at::kByte));

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

  return output_tensor;
}

std::vector<at::Tensor> dispatch_bf16_grouped_kernel(
    at::TensorList x_group, // BF16
    at::TensorList w_group, // BF16
    std::vector<at::Tensor> output_tensor,
    std::optional<at::Tensor> zero_start_index_M) {
  KernelMode kernel = get_grouped_kernel_mode(x_group, w_group);
  if (kernel == KernelMode::Small) {
    return bf16bf16bf16_grouped_impl<64, 128, 128, 2, 1, 1, true>(
        x_group, w_group, output_tensor, zero_start_index_M);
  } else if (kernel == KernelMode::Large) {
    return bf16bf16bf16_grouped_impl<128, 256, 64, 2, 1, 1, false>(
        x_group, w_group, output_tensor, zero_start_index_M);
  } else {
    return bf16bf16bf16_grouped_impl<128, 256, 64, 2, 1, 1, false>(
        x_group, w_group, output_tensor, zero_start_index_M);
  }
}

std::vector<at::Tensor> bf16bf16bf16_grouped(
    at::TensorList x_group, // BF16
    at::TensorList w_group, // BF16
    std::optional<std::vector<at::Tensor>> output = std::nullopt) {
  TORCH_CHECK(!output.has_value(), "Preallocated output not yet supported.");
  // Initialize output tensor.
  int problem_count = x_group.size();
  std::vector<at::Tensor> output_tensor;
  for (int i = 0; i < problem_count; i++) {
    int M = x_group[i].size(0);
    int N = w_group[i].size(0);
    output_tensor.push_back(
        at::empty({M, N}, x_group[i].options().dtype(at::kBFloat16)));
  }
  return dispatch_bf16_grouped_kernel(
      x_group, w_group, output_tensor, std::nullopt);
}

at::Tensor bf16bf16bf16_grouped_dynamic(
    at::TensorList x_group, // BF16
    at::TensorList w_group, // BF16
    std::optional<at::Tensor> zero_start_index_M = std::nullopt) {
  std::vector<at::Tensor> output_groups;
  at::Tensor output_full;
  int problem_count = x_group.size();
  int N = w_group[0].size(0);
  int K = x_group[0].size(1);
  if (zero_start_index_M.has_value()) {
    int M = x_group[0].size(0);
    // Fill output with zeros to simplify integration. This prevents nans from
    // showing up in the tensor.
    output_full = at::zeros(
        {problem_count, M, N}, x_group[0].options().dtype(at::kBFloat16));
    // Split the output into groups.
    output_groups = at::unbind(output_full, 0);
  } else {
    // If not provided, we try to allocate a single blob that can store each
    // group.
    int total_M = 0;
    std::vector<int> group_sizes = {};
    for (int i = 0; i < problem_count; i++) {
      TORCH_CHECK(
          x_group[i].size(1) == K && w_group[i].size(0) == N,
          "Dynamic grouped gemm requires fixed N and K.");
      int group_M = x_group[i].size(0);
      total_M += group_M;
      group_sizes.push_back(group_M);
    }
    // Allocate a contiguous array for all groups.
    output_full =
        at::empty({total_M, N}, x_group[0].options().dtype(at::kBFloat16));
    // Split the full array into appropriate groups.
    // We do this with narrow to make sure there are no extra copies.
    int offset = 0;
    for (int size : group_sizes) {
      output_groups.push_back(output_full.narrow(0, offset, size));
      offset += size;
    }
  }
  // Run kernel to populate output tensor.
  dispatch_bf16_grouped_kernel(
      x_group, w_group, output_groups, zero_start_index_M);
  // Return coalesced view of output.
  return output_full;
}

#else

std::vector<at::Tensor> bf16bf16bf16_grouped(
    at::TensorList /* x_group */, // BF16
    at::TensorList /* w_group */, // BF16
    std::optional<std::vector<at::Tensor>> /* output */) {
  throw std::runtime_error(
      "CUDA version is older than 12.0"); // requires CUDA>=12
}

at::Tensor bf16bf16bf16_grouped_dynamic(
    at::TensorList /* x_group */, // BF16
    at::TensorList /* w_group */, // BF16
    std::optional<at::Tensor> /* zero_start_index_M */) {
  throw std::runtime_error(
      "CUDA version is older than 12.0"); // requires CUDA>=12
}

#endif

} // namespace fbgemm_gpu
