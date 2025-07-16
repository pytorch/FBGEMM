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

#if CUDART_VERSION >= 12000

inline int64_t _byte_align(int64_t offset) {
  int64_t remainder = offset % 16;
  if (remainder != 0) {
    offset += (16 - remainder);
  }
  return offset;
}

template <
    int ScaleGranularityN,
    int ScaleGranularityK,
    typename ScaleConfig,
    typename ProblemShape,
    typename ElementA,
    typename ElementB,
    typename ElementC,
    typename ElementComputeEpilogue,
    typename StrideA,
    typename StrideB,
    typename StrideC,
    typename LayoutSFA,
    typename LayoutSFB>
__global__ void set_stacked_kernel_args_kernel(
    int64_t G,
    int64_t N,
    int64_t K,
    ProblemShape* problem_shape_ptr,
    ElementA* xq,
    const ElementA** xq_ptr,
    ElementB* wq,
    const ElementB** wq_ptr,
    ElementComputeEpilogue* x_scale,
    const ElementComputeEpilogue** x_scale_ptr,
    ElementComputeEpilogue* w_scale,
    const ElementComputeEpilogue** w_scale_ptr,
    ElementC* output,
    ElementC** output_ptr,
    StrideA* stride_a_ptr,
    StrideB* stride_b_ptr,
    StrideC* stride_c_ptr,
    LayoutSFA* stride_xs_ptr,
    LayoutSFB* stride_ws_ptr,
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
      problem_shape_ptr[non_zero_idx] = ProblemShape(N, M, K);
      // Set input pointers.
      xq_ptr[non_zero_idx] = xq + (offset_M * K);
      wq_ptr[non_zero_idx] = wq + (group_index * N * K);
      x_scale_ptr[non_zero_idx] =
          x_scale + (offset_M * (K / ScaleGranularityK));
      w_scale_ptr[non_zero_idx] = w_scale +
          (group_index * (N / ScaleGranularityN) * (K / ScaleGranularityK));
      output_ptr[non_zero_idx] = output + (offset_M * N);
      stride_a_ptr[non_zero_idx] = cutlass::make_cute_packed_stride(
          StrideA{}, cute::make_shape(int(M), int(K), 1));
      stride_b_ptr[non_zero_idx] = cutlass::make_cute_packed_stride(
          StrideB{}, cute::make_shape(int(N), int(K), 1));
      stride_c_ptr[non_zero_idx] = cutlass::make_cute_packed_stride(
          StrideC{}, cute::make_shape(int(N), int(M), 1));
      LayoutSFA layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(
          cute::make_shape(M, int(N), int(K), 1));
      stride_xs_ptr[non_zero_idx] = layout_SFA;
      LayoutSFB layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(
          cute::make_shape(M, int(N), int(K), 1));
      stride_ws_ptr[non_zero_idx] = layout_SFB;
    }
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
    int ARCH,
    bool PONG>
at::Tensor f8f8bf16_groupwise_grouped_impl(
    InputType XQ, // FP8
    InputType WQ, // FP8
    InputType x_scale,
    InputType w_scale,
    at::Tensor output,
    at::Tensor M_sizes) {
  int64_t G;
  at::TensorOptions options;
  static_assert(
      std::is_same_v<InputType, at::Tensor>,
      "Only stacked API is currently supported.");
  G = WQ.size(0);
  options = XQ.options();

  // The number of groups the kernel uses may vary.
  int kernel_groups = G;
  // Return early if there are no elements in the output.
  if (output.numel() == 0) {
    return output;
  }

  // Define gemm configuration.
  using ProblemShape =
      cutlass::gemm::GroupProblemShape<cute::Shape<int, int, int>>;
  using ElementA = cutlass::float_e4m3_t;
  using ElementB = cutlass::float_e4m3_t;
  using ElementC = cutlass::bfloat16_t;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutA_Transpose =
      typename cutlass::layout::LayoutTranspose<LayoutA>::type;
  using LayoutB_Transpose =
      typename cutlass::layout::LayoutTranspose<LayoutB>::type;
  using ElementAccumulator = float;
  using ElementComputeEpilogue = float;

  // Configure scale settings.
  constexpr int ScaleGranularityM = 1;
  constexpr int ScaleGranularityN = 128;
  constexpr int ScaleGranularityK = 128;
  using ScaleConfig = cutlass::detail::Sm90BlockwiseScaleConfig<
      ScaleGranularityM,
      ScaleGranularityN,
      ScaleGranularityK>;
  using LayoutSFA =
      decltype(ScaleConfig::deduce_layoutSFA()); // Layout type for SFA matrix
                                                 // operand
  using LayoutSFB =
      decltype(ScaleConfig::deduce_layoutSFB()); // Layout type for SFB matrix
                                                 // operand

  using ArchTag = cutlass::arch::Sm90; // Tag indicating the minimum SM that
                                       // supports the intended feature
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using StageCountType = cutlass::gemm::collective::StageCountAuto;
  using TileShape =
      cute::Shape<cute::Int<TB_M>, cute::Int<TB_N>, cute::Int<TB_K>>;
  using ClusterShape =
      cute::Shape<cute::Int<TBS_M>, cute::Int<TBS_N>, cute::Int<TBS_K>>;

  // Define kernel schedules.
  using CooperativeSchedule = cutlass::gemm::
      KernelPtrArrayTmaWarpSpecializedCooperativeFP8BlockScaledAccum;
  using PongSchedule = cutlass::gemm::
      KernelPtrArrayTmaWarpSpecializedPingpongFP8BlockScaledAccum;
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
          typename cutlass::layout::LayoutTranspose<LayoutC>::type*,
          128 / cutlass::sizeof_bits<ElementC>::value,
          ElementC,
          typename cutlass::layout::LayoutTranspose<LayoutC>::type*,
          128 / cutlass::sizeof_bits<ElementC>::value,
          EpilogueSchedule>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag,
          OperatorClass,
          ElementB,
          cute::tuple<LayoutB_Transpose*, LayoutSFB*>,
          128 / cutlass::sizeof_bits<ElementA>::value,
          ElementA,
          cute::tuple<LayoutA_Transpose*, LayoutSFA*>,
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

  // Next create space for XQ pointers.
  const int64_t xq_offset = problem_size_offset + problem_size_buffer;
  int64_t xq_size_buffer = _byte_align(G * sizeof(ElementA**));

  // WQ Pointers.
  const int64_t wq_offset = xq_offset + xq_size_buffer;
  int64_t wq_size_buffer = _byte_align(G * sizeof(ElementB**));

  // X group scales.
  const int64_t x_scale_offset = wq_offset + wq_size_buffer;
  int64_t x_scale_buffer = _byte_align(G * sizeof(ElementComputeEpilogue**));

  // W group scales.
  const int64_t w_scale_offset = x_scale_offset + x_scale_buffer;
  int64_t w_scale_buffer = _byte_align(G * sizeof(ElementComputeEpilogue**));

  // Outputs.
  const int64_t output_offset = w_scale_offset + w_scale_buffer;
  int64_t output_buffer = _byte_align(G * sizeof(ElementC**));

  // A stride.
  const int64_t stride_a_offset = output_offset + output_buffer;
  int64_t stride_a_buffer = _byte_align(G * sizeof(StrideA));

  // B stride.
  const int64_t stride_b_offset = stride_a_offset + stride_a_buffer;
  int64_t stride_b_buffer = _byte_align(G * sizeof(StrideB));

  // C stride.
  const int64_t stride_c_offset = stride_b_offset + stride_b_buffer;
  int64_t stride_c_buffer = _byte_align(G * sizeof(StrideC));

  // X group scales stride.
  const int64_t stride_x_scale_offset = stride_c_offset + stride_c_buffer;
  int64_t stride_x_scale_buffer = _byte_align(G * sizeof(LayoutSFA));

  // W group scales stride.
  const int64_t stride_w_scale_offset =
      stride_x_scale_offset + stride_x_scale_buffer;
  int64_t stride_w_scale_buffer = _byte_align(G * sizeof(LayoutSFB));

  // Compute total buffer size
  int64_t total_buffer_size = stride_w_scale_offset + stride_w_scale_buffer;

  // Allocate space for gemm information.
  at::Tensor kernel_args =
      at::empty({total_buffer_size}, options.dtype(at::kByte));

  // Get byte pointer to underlying data.
  char* kernel_args_ptr = reinterpret_cast<char*>(kernel_args.data_ptr());

  // Now use offsets to get appropriately typed pointers.
  ProblemShape::UnderlyingProblemShape* problem_shape_ptr =
      reinterpret_cast<ProblemShape::UnderlyingProblemShape*>(
          kernel_args_ptr + problem_size_offset);
  const ElementA** xq_ptr =
      reinterpret_cast<const ElementA**>(kernel_args_ptr + xq_offset);
  const ElementB** wq_ptr =
      reinterpret_cast<const ElementB**>(kernel_args_ptr + wq_offset);
  const ElementComputeEpilogue** x_scale_ptr =
      reinterpret_cast<const ElementComputeEpilogue**>(
          kernel_args_ptr + x_scale_offset);
  const ElementComputeEpilogue** w_scale_ptr =
      reinterpret_cast<const ElementComputeEpilogue**>(
          kernel_args_ptr + w_scale_offset);
  ElementC** output_ptr =
      reinterpret_cast<ElementC**>(kernel_args_ptr + output_offset);
  StrideA* stride_a_ptr =
      reinterpret_cast<StrideA*>(kernel_args_ptr + stride_a_offset);
  StrideB* stride_b_ptr =
      reinterpret_cast<StrideB*>(kernel_args_ptr + stride_b_offset);
  StrideC* stride_c_ptr =
      reinterpret_cast<StrideC*>(kernel_args_ptr + stride_c_offset);
  LayoutSFA* stride_x_scale_ptr =
      reinterpret_cast<LayoutSFA*>(kernel_args_ptr + stride_x_scale_offset);
  LayoutSFB* stride_w_scale_ptr =
      reinterpret_cast<LayoutSFB*>(kernel_args_ptr + stride_w_scale_offset);

  auto stream = at::cuda::getCurrentCUDAStream().stream();
  TORCH_CHECK(M_sizes.dtype() == at::kLong, "M_sizes must be int64.");
  // When m_offsets is used, XQ is shape [total_M, K].
  int64_t M = XQ.size(0);
  int64_t N = WQ.size(1);
  int64_t K = WQ.size(2);
  int64_t* M_sizes_ptr = reinterpret_cast<int64_t*>(M_sizes.data_ptr());
  set_stacked_kernel_args_kernel<
      ScaleGranularityN,
      ScaleGranularityK,
      ScaleConfig><<<1, G, 0, stream>>>(
      G,
      N,
      K,
      problem_shape_ptr,
      reinterpret_cast<ElementA*>(XQ.data_ptr()),
      xq_ptr,
      reinterpret_cast<ElementB*>(WQ.data_ptr()),
      wq_ptr,
      reinterpret_cast<ElementComputeEpilogue*>(x_scale.data_ptr()),
      x_scale_ptr,
      reinterpret_cast<ElementComputeEpilogue*>(w_scale.data_ptr()),
      w_scale_ptr,
      reinterpret_cast<ElementC*>(output.data_ptr()),
      output_ptr,
      stride_a_ptr,
      stride_b_ptr,
      stride_c_ptr,
      stride_x_scale_ptr,
      stride_w_scale_ptr,
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
      {wq_ptr,
       stride_b_ptr,
       xq_ptr,
       stride_a_ptr,
       w_scale_ptr,
       stride_w_scale_ptr,
       x_scale_ptr,
       stride_x_scale_ptr},
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

#endif

} // namespace fbgemm_gpu
