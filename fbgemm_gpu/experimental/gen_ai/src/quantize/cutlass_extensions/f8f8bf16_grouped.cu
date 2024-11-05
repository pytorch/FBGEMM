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
std::vector<at::Tensor> f8f8bf16_grouped_impl(
    const std::vector<at::Tensor>& XQ, // FP8
    const std::vector<at::Tensor>& WQ, // FP8
    const std::vector<at::Tensor>& scale) {
  int problem_count = XQ.size();
  TORCH_CHECK(WQ.size() == problem_count);
  if (problem_count == 0) {
    return std::vector<at::Tensor>();
  }

  using ProblemShape =
      cutlass::gemm::GroupProblemShape<cute::Shape<int, int, int>>; // <M,N,K>
                                                                    // per group
  using ElementInputA =
      cutlass::float_e4m3_t; // Element type for A matrix operand
  using ElementInputB =
      cutlass::float_e4m3_t; // Element type for B matrix operand
  using ElementOutput =
      cutlass::bfloat16_t; // Element type for C and D matrix operands

  using LayoutInputA =
      cutlass::layout::RowMajor; // Layout type for A matrix operand
  constexpr int AlignmentA =
      128 / cutlass::sizeof_bits<ElementInputA>::value; // Alignment of A matrix
                                                        // in units of elements
                                                        // (up to 16 bytes)

  using LayoutInputB =
      cutlass::layout::ColumnMajor; // Layout type for B matrix operand
  constexpr int AlignmentB =
      128 / cutlass::sizeof_bits<ElementInputB>::value; // Alignment of B matrix
                                                        // in units of elements
                                                        // (up to 16 bytes)

  using LayoutOutput =
      cutlass::layout::RowMajor; // Layout type for C and D matrix operands
  constexpr int AlignmentD =
      128 / cutlass::sizeof_bits<ElementOutput>::value; // Alignment of C matrix
                                                        // in units of elements
                                                        // (up to 16 bytes)
  // Core kernel configurations
  using ElementAccumulator = float; // Element type for internal accumulation
  using ArchTag = cutlass::arch::Sm90; // Tag indicating the minimum SM that
                                       // supports the intended feature
  using OperatorClass = cutlass::arch::OpClassTensorOp; // Operator class tag
  using StageCountType =
      cutlass::gemm::collective::StageCountAuto; // Stage count maximized based
                                                 // on the tile size

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
  using TileShape =
      cute::Shape<cute::Int<TB_M>, cute::Int<TB_N>, cute::Int<TB_K>>;
  using ClusterShape =
      cute::Shape<cute::Int<TBS_M>, cute::Int<TBS_N>, cute::Int<TBS_K>>;

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
          AlignmentD,
          ElementOutput,
          LayoutOutput*,
          AlignmentD,
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
          AlignmentA,
          ElementInputB,
          LayoutInputB*,
          AlignmentB,
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

  int64_t output_offset = 0;
  int64_t total_output_size = 0;
  std::vector<int64_t> output_sizes;
  output_sizes.reserve(problem_count);
  at::Tensor output_args = at::empty(
      {problem_count},
      at::TensorOptions().dtype(at::kLong).pinned_memory(true));
  int64_t* output_ptr = output_args.data_ptr<int64_t>();

  const int64_t problem_shape_size =
      problem_count * ((int64_t)sizeof(ProblemShape::UnderlyingProblemShape));
  const int64_t stride_size = problem_count * ((int64_t)sizeof(StrideInputA));

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

  ProblemShape::UnderlyingProblemShape* problem_shape_ptr =
      reinterpret_cast<ProblemShape::UnderlyingProblemShape*>(
          problem_shape_buf);
  StrideInputA* stride_input_A_ptr =
      reinterpret_cast<StrideInputA*>(stride_buf);
  StrideInputB* stride_input_B_ptr =
      reinterpret_cast<StrideInputB*>(stride_buf + stride_size);
  StrideOutput* stride_output_ptr =
      reinterpret_cast<StrideOutput*>(stride_buf + (stride_size * 2));

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
    scale_ptr[i] =
        reinterpret_cast<int64_t>(scale[i].data_ptr<ElementAccumulator>());
    problem_shape_ptr[i] = ProblemShape::UnderlyingProblemShape(m, n, k);
    stride_input_A_ptr[i] =
        cutlass::make_cute_packed_stride(StrideInputA{}, {m, k, 1});
    stride_input_B_ptr[i] =
        cutlass::make_cute_packed_stride(StrideInputB{}, {n, k, 1});
    stride_output_ptr[i] =
        cutlass::make_cute_packed_stride(StrideOutput{}, {m, n, 1});
  }

  const auto device = XQ[0].device();
  input_args = input_args.to(device, /*non_blocking=*/true);
  output_args = output_args.to(device, /*non_blocking=*/true);

  output_ptr = output_args.data_ptr<int64_t>();
  xq_ptr = input_args.data_ptr<int64_t>();
  wq_ptr = input_args.data_ptr<int64_t>() + (problem_count * sizeof(int64_t));
  scale_ptr =
      input_args.data_ptr<int64_t>() + (problem_count * 2 * sizeof(int64_t));

  problem_shape_buf = reinterpret_cast<uint8_t*>(
      input_args.data_ptr<int64_t>() + (problem_count * 3 * sizeof(int64_t)));
  problem_shape_ptr =
      reinterpret_cast<typename ProblemShape::UnderlyingProblemShape*>(
          problem_shape_buf);
  stride_buf = reinterpret_cast<uint8_t*>(
      input_args.data_ptr<int64_t>() + (problem_count * 3 * sizeof(int64_t)) +
      problem_shape_size);
  stride_input_A_ptr = reinterpret_cast<StrideInputA*>(stride_buf);
  stride_input_B_ptr =
      reinterpret_cast<StrideInputB*>(stride_buf + stride_size);
  stride_output_ptr =
      reinterpret_cast<StrideOutput*>(stride_buf + (stride_size * 2));

  typename Gemm::Arguments arguments;
  decltype(arguments.epilogue.thread) fusion_args;
  fusion_args.alpha_ptr_array =
      reinterpret_cast<const ElementAccumulator**>(scale_ptr);
  fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 1};

  arguments = typename Gemm::Arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {problem_count, problem_shape_ptr, nullptr},
      {reinterpret_cast<const ElementInputA**>(xq_ptr),
       stride_input_A_ptr,
       reinterpret_cast<const ElementInputB**>(wq_ptr),
       stride_input_B_ptr},
      {fusion_args,
       reinterpret_cast<const ElementOutput**>(output_ptr),
       stride_output_ptr,
       reinterpret_cast<ElementOutput**>(output_ptr),
       stride_output_ptr}};

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
