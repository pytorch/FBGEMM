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

namespace fbgemm_gpu {

inline int64_t _byte_align(int64_t offset) {
  int64_t remainder = offset % 16;
  if (remainder != 0) {
    offset += (16 - remainder);
  }
  return offset;
}

template <
    typename ProblemShape,
    typename ElementA,
    typename ElementB,
    typename ElementC,
    typename StrideA,
    typename StrideB,
    typename StrideC>
__global__ void set_stacked_kernel_args_kernel(
    int64_t G,
    int64_t N,
    int64_t K,
    ProblemShape* problem_shape_ptr,
    ElementA* x,
    const ElementA** x_ptr,
    ElementB* w,
    const ElementB** w_ptr,
    ElementC* output,
    ElementC** output_ptr,
    StrideA* stride_a_ptr,
    StrideB* stride_b_ptr,
    StrideC* stride_c_ptr,
    int64_t* M_sizes) {
  uint32_t group_index = blockIdx.x * blockDim.x + threadIdx.x;
  // If this thread corresponds to a valid group, write kernel args to device
  // memory.
  if (group_index < G) {
    // Its possible that we're only writing a subset of the groups to
    // kernel args. To do this, we need to set all groups initially to empty.
    // and keep a problem counter for the number of non-empty groups.
    __shared__ int non_zero_counter;
    // Initialize counter in first group.
    if (group_index == 0) {
      non_zero_counter = 0;
    }
    // Set problem shapes to empty by default.
    problem_shape_ptr[group_index] = ProblemShape(0, 0, 0);
    // Sync threads to get consistent state in the block.
    __syncthreads();

    // Compute shape for this group.
    // M for this group is pulled directly from M_sizes.
    int M = M_sizes[group_index];
    // Only proceed to writing kernel args if this group is non-empty.
    if (M > 0) {
      // Get the index for this group atomically.
      int non_zero_idx = atomicAdd(&non_zero_counter, 1);
      // We compute the offset by getting the cumulative sum over
      // prior groups.
      int64_t offset_M = 0;
      for (int i = 0; i < group_index; i++) {
        offset_M += M_sizes[i];
      }
      // Set the problem shape for this group.
      problem_shape_ptr[non_zero_idx] = ProblemShape(int(M), int(N), int(K));
      // Set input pointers.
      x_ptr[non_zero_idx] = x + (offset_M * K);
      w_ptr[non_zero_idx] = w + (group_index * N * K);
      output_ptr[non_zero_idx] = output + (offset_M * N);
      stride_a_ptr[non_zero_idx] = cutlass::make_cute_packed_stride(
          StrideA{}, cute::make_shape(int(M), int(K), 1));
      stride_b_ptr[non_zero_idx] = cutlass::make_cute_packed_stride(
          StrideB{}, cute::make_shape(int(N), int(K), 1));
      stride_c_ptr[non_zero_idx] = cutlass::make_cute_packed_stride(
          StrideC{}, cute::make_shape(int(M), int(N), 1));
    }
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
at::Tensor bf16bf16bf16_grouped_grad_impl(
    at::Tensor X,
    at::Tensor W,
    at::Tensor output,
    std::optional<at::Tensor> M_sizes) {
  int64_t G;
  at::TensorOptions options;
  G = W.size(0);
  options = X.options();

  // The number of groups the kernel uses may vary.
  int kernel_groups = int(G);
  // Return early if there are no elements in the output.
  if (output.numel() == 0) {
    return output;
  }

  // Define gemm configuration.
  using ProblemShape =
      cutlass::gemm::GroupProblemShape<cute::Shape<int, int, int>>;
  using ElementA = cutlass::bfloat16_t;
  using ElementB = cutlass::bfloat16_t;
  using ElementC = cutlass::bfloat16_t;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using ElementAccumulator = float;
  using ArchTag = cutlass::arch::Sm90; // Tag indicating the minimum SM that
                                       // supports the intended feature
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using StageCountType = cutlass::gemm::collective::StageCountAuto;
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

  using MainLoopSchedule =
      cute::conditional_t<PONG, PongSchedule, CooperativeSchedule>;
  using EpilogueSchedule = cute::
      conditional_t<PONG, PongEpilogueSchedule, CooperativeEpilogueSchedule>;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag,
          cutlass::arch::OpClassTensorOp,
          TileShape,
          ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAccumulator,
          ElementAccumulator,
          void, // Indicate there is no beta scaling to save register space.
          LayoutC*,
          128 / cutlass::sizeof_bits<ElementC>::value,
          ElementC,
          LayoutC*,
          128 / cutlass::sizeof_bits<ElementC>::value,
          EpilogueSchedule>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag,
          OperatorClass,
          ElementA,
          LayoutA*,
          128 / cutlass::sizeof_bits<ElementA>::value,
          ElementB,
          LayoutB*,
          128 / cutlass::sizeof_bits<ElementB>::value,
          ElementAccumulator,
          TileShape,
          ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          MainLoopSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::
      GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideC = typename Gemm::GemmKernel::InternalStrideD;

  // Create a buffer for kernel arguments. We do this by first figuring out
  // how much space each sub-argument requires and setting up corresponding
  // pointers.
  const int64_t problem_size_offset = 0;
  int64_t problem_size_buffer =
      _byte_align(G * sizeof(ProblemShape::UnderlyingProblemShape));

  // Next create space for X pointers.
  const int64_t x_offset = problem_size_offset + problem_size_buffer;
  int64_t x_size_buffer = _byte_align(G * sizeof(ElementA**));

  // W Pointers.
  const int64_t w_offset = x_offset + x_size_buffer;
  int64_t w_size_buffer = _byte_align(G * sizeof(ElementB**));

  // Outputs.
  const int64_t output_offset = w_offset + w_size_buffer;
  int64_t output_buffer = _byte_align(G * sizeof(ElementC**));

  // A stride.
  const int64_t stride_a_offset = output_offset + output_buffer;
  int64_t stride_a_buffer = _byte_align(G * sizeof(StrideA));

  // B stride;
  const int64_t stride_b_offset = stride_a_offset + stride_a_buffer;
  int64_t stride_b_buffer = _byte_align(G * sizeof(StrideB));

  // C stride;
  const int64_t stride_c_offset = stride_b_offset + stride_b_buffer;
  int64_t stride_c_buffer = _byte_align(G * sizeof(StrideC));

  // Compute total buffer size
  int64_t total_buffer_size = stride_c_offset + stride_c_buffer;

  // Allocate space for gemm information.
  at::Tensor kernel_args =
      at::empty({total_buffer_size}, options.dtype(at::kByte));

  // Get byte pointer to underlying data.
  char* kernel_args_ptr = reinterpret_cast<char*>(kernel_args.data_ptr());

  // Now use offsets to get appropriately typed pointers.
  ProblemShape::UnderlyingProblemShape* problem_shape_ptr =
      reinterpret_cast<ProblemShape::UnderlyingProblemShape*>(
          kernel_args_ptr + problem_size_offset);
  const ElementA** x_ptr =
      reinterpret_cast<const ElementA**>(kernel_args_ptr + x_offset);
  const ElementB** w_ptr =
      reinterpret_cast<const ElementB**>(kernel_args_ptr + w_offset);
  ElementC** output_ptr =
      reinterpret_cast<ElementC**>(kernel_args_ptr + output_offset);
  StrideA* stride_a_ptr =
      reinterpret_cast<StrideA*>(kernel_args_ptr + stride_a_offset);
  StrideB* stride_b_ptr =
      reinterpret_cast<StrideB*>(kernel_args_ptr + stride_b_offset);
  StrideC* stride_c_ptr =
      reinterpret_cast<StrideC*>(kernel_args_ptr + stride_c_offset);

  auto stream = at::cuda::getCurrentCUDAStream().stream();

  TORCH_CHECK(
      !M_sizes.has_value() || M_sizes->dtype() == at::kLong,
      "M_sizes must be int64.");
  int64_t M = X.size(X.dim() - 2);
  int64_t N = W.size(1);
  int64_t K = W.size(2);

  int64_t* M_sizes_ptr = reinterpret_cast<int64_t*>(M_sizes.value().data_ptr());
  set_stacked_kernel_args_kernel<<<1, G, 0, stream>>>(
      G,
      N,
      K,
      problem_shape_ptr,
      reinterpret_cast<ElementA*>(X.data_ptr()),
      x_ptr,
      reinterpret_cast<ElementB*>(W.data_ptr()),
      w_ptr,
      reinterpret_cast<ElementC*>(output.data_ptr()),
      output_ptr,
      stride_a_ptr,
      stride_b_ptr,
      stride_c_ptr,
      M_sizes_ptr);
  // Set the number of groups to the kernel to be at most the number of
  // non-zero rows.
  kernel_groups = int(std::min(M, G));

  cutlass::KernelHardwareInfo hw_info;
  // Change device_id to another value if you are running on a machine with
  // multiple GPUs and wish to use a GPU other than that with device ID 0.
  hw_info.device_id = 0;
  hw_info.sm_count =
      min(cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
              hw_info.device_id),
          2147483647); // INT_MAX

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {kernel_groups, problem_shape_ptr, nullptr},
      {x_ptr, stride_a_ptr, w_ptr, stride_b_ptr},
      {{}, nullptr, stride_c_ptr, output_ptr, stride_c_ptr},
      hw_info};

  Gemm gemm;

  // Using the arguments, query for extra workspace required for matrix
  // multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  at::Tensor workspace = at::empty(workspace_size, options.dtype(at::kByte));

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

#if CUDART_VERSION >= 12080
template <
    int TB_M,
    int TB_N,
    int TB_K,
    int TBS_M,
    int TBS_N,
    int TBS_K,
    bool PONG>
at::Tensor bf16bf16bf16_grouped_grad_sm100_impl(
    at::Tensor X,
    at::Tensor W,
    at::Tensor output,
    std::optional<at::Tensor> M_sizes) {
  int64_t G;
  at::TensorOptions options;
  G = W.size(0);
  options = X.options();

  // The number of groups the kernel uses may vary.
  int kernel_groups = int(G);
  // Return early if there are no elements in the output.
  if (output.numel() == 0) {
    return output;
  }

  // Define gemm configuration.
  using ProblemShape =
      cutlass::gemm::GroupProblemShape<cute::Shape<int, int, int>>;
  using ElementA = cutlass::bfloat16_t;
  using ElementB = cutlass::bfloat16_t;
  using ElementC = cutlass::bfloat16_t;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using ElementAccumulator = float;
  using ArchTag = cutlass::arch::Sm100; // Tag indicating the minimum SM that
                                        // supports the intended feature
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using StageCountType = cutlass::gemm::collective::StageCountAuto;
  using TileShape =
      cute::Shape<cute::Int<TB_M>, cute::Int<TB_N>, cute::Int<TB_K>>;
  using ClusterShape =
      cute::Shape<cute::Int<TBS_M>, cute::Int<TBS_N>, cute::Int<TBS_K>>;

  using MainLoopSchedule = cute::conditional_t<
      (TBS_M % 2 == 0) || (TB_M == 256),
      cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmSm100,
      cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmSm100>;
  using EpilogueSchedule = cute::conditional_t<
      (TBS_M % 2 == 0) || (TB_M == 256),
      cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm,
      cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm>;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag,
          cutlass::arch::OpClassTensorOp,
          TileShape,
          ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAccumulator,
          ElementAccumulator,
          void, // Indicate there is no beta scaling to save register space.
          LayoutC*,
          128 / cutlass::sizeof_bits<ElementC>::value,
          ElementC,
          LayoutC*,
          128 / cutlass::sizeof_bits<ElementC>::value,
          EpilogueSchedule>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag,
          OperatorClass,
          ElementA,
          LayoutA*,
          128 / cutlass::sizeof_bits<ElementA>::value,
          ElementB,
          LayoutB*,
          128 / cutlass::sizeof_bits<ElementB>::value,
          ElementAccumulator,
          TileShape,
          ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          MainLoopSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::
      GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideC = typename Gemm::GemmKernel::InternalStrideD;

  // Create a buffer for kernel arguments. We do this by first figuring out
  // how much space each sub-argument requires and setting up corresponding
  // pointers.
  const int64_t problem_size_offset = 0;
  int64_t problem_size_buffer =
      _byte_align(G * sizeof(ProblemShape::UnderlyingProblemShape));

  // Next create space for X pointers.
  const int64_t x_offset = problem_size_offset + problem_size_buffer;
  int64_t x_size_buffer = _byte_align(G * sizeof(ElementA**));

  // W Pointers.
  const int64_t w_offset = x_offset + x_size_buffer;
  int64_t w_size_buffer = _byte_align(G * sizeof(ElementB**));

  // Outputs.
  const int64_t output_offset = w_offset + w_size_buffer;
  int64_t output_buffer = _byte_align(G * sizeof(ElementC**));

  // A stride.
  const int64_t stride_a_offset = output_offset + output_buffer;
  int64_t stride_a_buffer = _byte_align(G * sizeof(StrideA));

  // B stride;
  const int64_t stride_b_offset = stride_a_offset + stride_a_buffer;
  int64_t stride_b_buffer = _byte_align(G * sizeof(StrideB));

  // C stride;
  const int64_t stride_c_offset = stride_b_offset + stride_b_buffer;
  int64_t stride_c_buffer = _byte_align(G * sizeof(StrideC));

  // Compute total buffer size
  int64_t total_buffer_size = stride_c_offset + stride_c_buffer;

  // Allocate space for gemm information.
  at::Tensor kernel_args =
      at::empty({total_buffer_size}, options.dtype(at::kByte));

  // Get byte pointer to underlying data.
  char* kernel_args_ptr = reinterpret_cast<char*>(kernel_args.data_ptr());

  // Now use offsets to get appropriately typed pointers.
  ProblemShape::UnderlyingProblemShape* problem_shape_ptr =
      reinterpret_cast<ProblemShape::UnderlyingProblemShape*>(
          kernel_args_ptr + problem_size_offset);
  const ElementA** x_ptr =
      reinterpret_cast<const ElementA**>(kernel_args_ptr + x_offset);
  const ElementB** w_ptr =
      reinterpret_cast<const ElementB**>(kernel_args_ptr + w_offset);
  ElementC** output_ptr =
      reinterpret_cast<ElementC**>(kernel_args_ptr + output_offset);
  StrideA* stride_a_ptr =
      reinterpret_cast<StrideA*>(kernel_args_ptr + stride_a_offset);
  StrideB* stride_b_ptr =
      reinterpret_cast<StrideB*>(kernel_args_ptr + stride_b_offset);
  StrideC* stride_c_ptr =
      reinterpret_cast<StrideC*>(kernel_args_ptr + stride_c_offset);

  auto stream = at::cuda::getCurrentCUDAStream().stream();

  TORCH_CHECK(
      !M_sizes.has_value() || M_sizes->dtype() == at::kLong,
      "M_sizes must be int64.");
  int64_t M = X.size(X.dim() - 2);
  int64_t N = W.size(1);
  int64_t K = W.size(2);

  int64_t* M_sizes_ptr = reinterpret_cast<int64_t*>(M_sizes.value().data_ptr());
  set_stacked_kernel_args_kernel<<<1, G, 0, stream>>>(
      G,
      N,
      K,
      problem_shape_ptr,
      reinterpret_cast<ElementA*>(X.data_ptr()),
      x_ptr,
      reinterpret_cast<ElementB*>(W.data_ptr()),
      w_ptr,
      reinterpret_cast<ElementC*>(output.data_ptr()),
      output_ptr,
      stride_a_ptr,
      stride_b_ptr,
      stride_c_ptr,
      M_sizes_ptr);
  // Set the number of groups to the kernel to be at most the number of
  // non-zero rows.
  kernel_groups = int(std::min(M, G));

  cutlass::KernelHardwareInfo hw_info;
  // Change device_id to another value if you are running on a machine with
  // multiple GPUs and wish to use a GPU other than that with device ID 0.
  hw_info.device_id = 0;
  hw_info.sm_count =
      min(cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
              hw_info.device_id),
          2147483647); // INT_MAX

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {kernel_groups, problem_shape_ptr, nullptr},
      {x_ptr, stride_a_ptr, w_ptr, stride_b_ptr},
      {{}, nullptr, stride_c_ptr, output_ptr, stride_c_ptr},
      hw_info};

  Gemm gemm;

  // Using the arguments, query for extra workspace required for matrix
  // multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  at::Tensor workspace = at::empty(workspace_size, options.dtype(at::kByte));

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

#else

template <
    int TB_M,
    int TB_N,
    int TB_K,
    int TBS_M,
    int TBS_N,
    int TBS_K,
    bool PONG>
at::Tensor bf16bf16bf16_grouped_grad_sm100_impl(
    at::Tensor X,
    at::Tensor W,
    at::Tensor output,
    std::optional<at::Tensor> M_sizes) {
  return output;
}
#endif

} // namespace fbgemm_gpu
