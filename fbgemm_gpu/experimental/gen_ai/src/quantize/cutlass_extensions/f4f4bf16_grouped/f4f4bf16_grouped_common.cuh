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

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12080)

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
    typename ElementScale,
    typename StrideA,
    typename StrideB,
    typename StrideC,
    typename LayoutSFA,
    typename LayoutSFB,
    typename ElementGlobalScale,
    typename Sm1xxBlkScaledConfig>
__global__ void set_stacked_kernel_args_kernel(
    int64_t G,
    int64_t N,
    int64_t K,
    ProblemShape* problem_shape_ptr,
    ElementA* xq,
    const ElementA** xq_ptr,
    ElementB* wq,
    const ElementB** wq_ptr,
    ElementScale* x_scale,
    const ElementScale** x_scale_ptr,
    ElementScale* w_scale,
    const ElementScale** w_scale_ptr,
    ElementC* output,
    ElementC** output_ptr,
    StrideA* stride_a_ptr,
    StrideB* stride_b_ptr,
    StrideC* stride_c_ptr,
    int64_t* M_sizes,
    LayoutSFA* layout_SFA,
    LayoutSFB* layout_SFB,
    ElementGlobalScale* global_scale,
    const ElementGlobalScale** global_scale_ptr,
    int64_t* starting_row_after_padding) {
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
    auto round_up = [](int64_t x, int64_t y) { return ((x + y - 1) / y) * y; };

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
      int64_t accumulated_x_scale = 0;
      int64_t accumulated_w_scale = 0;
      int ele_per_quantize_group = 16;
      if (global_scale == nullptr) {
        ele_per_quantize_group = 32;
      }
      for (int i = 0; i < group_index; i++) {
        offset_M += M_sizes[i];
        /* It's calculated this way since the scales are at least padded to
           multiples of (128, 4), and there is a group of 16 elements per scale.
        */
        accumulated_w_scale +=
            round_up(N, 128) * round_up(K, 4) / ele_per_quantize_group;
      }
      accumulated_x_scale =
          starting_row_after_padding[group_index] * K / ele_per_quantize_group;
      // Set the problem shape for this group.
      problem_shape_ptr[non_zero_idx] = ProblemShape(N, M, K);
      // Set input pointers.
      xq_ptr[non_zero_idx] = xq + (offset_M * K / 2);
      wq_ptr[non_zero_idx] = wq + (group_index * N * K / 2);
      x_scale_ptr[non_zero_idx] = x_scale + accumulated_x_scale;
      w_scale_ptr[non_zero_idx] = w_scale + accumulated_w_scale;
      output_ptr[non_zero_idx] = output + (offset_M * N);
      stride_a_ptr[non_zero_idx] = cutlass::make_cute_packed_stride(
          StrideA{}, cute::make_shape(int(M), int(K), 1));
      stride_b_ptr[non_zero_idx] = cutlass::make_cute_packed_stride(
          StrideB{}, cute::make_shape(int(N), int(K), 1));
      stride_c_ptr[non_zero_idx] = cutlass::make_cute_packed_stride(
          StrideC{}, cute::make_shape(int(N), int(M), 1));
      layout_SFA[non_zero_idx] = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
          cute::make_shape(int(M), int(N), int(K), 1));
      layout_SFB[non_zero_idx] = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
          cute::make_shape(int(M), int(N), int(K), 1));
      if (global_scale != nullptr) {
        global_scale_ptr[non_zero_idx] = global_scale + group_index;
      }
    }
  }
}

template <
    typename InputQuantType,
    int TB_M,
    int TB_N,
    int TB_K,
    int TBS_M,
    int TBS_N,
    int TBS_K>
at::Tensor f4f4bf16_grouped_impl(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    std::optional<at::Tensor> M_sizes,
    std::optional<at::Tensor> global_scale,
    std::optional<at::Tensor> starting_row_after_padding) {
  // The number of groups the kernel uses may vary.
  int64_t G;
  int kernel_groups;
  if (M_sizes) {
    G = M_sizes->size(0);
    kernel_groups = M_sizes->size(0);
  }

  at::TensorOptions options;
  options = XQ.options();

  // Return early if there are no elements in the output.
  if (output.numel() == 0) {
    return output;
  }

  // NVFP4 uses global scale
  constexpr bool is_nvfp4 = std::
      is_same_v<InputQuantType, cutlass::nv_float4_t<cutlass::float_e2m1_t>>;
  TORCH_CHECK(
      is_nvfp4 == global_scale.has_value(),
      "global_scale must be set for nvfp4 inputs.");

  // Define gemm configuration.
  using ProblemShape =
      cutlass::gemm::GroupProblemShape<cute::Shape<int, int, int>>;
  using ElementA = InputQuantType;
  using ElementB = InputQuantType;
  using ElementC = cutlass::bfloat16_t;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;

  using ElementAUnderlying = typename ElementA::DataType;
  using ElementBUnderlying = typename ElementB::DataType;

  constexpr int AlignmentA =
      128 / cutlass::sizeof_bits<ElementAUnderlying>::value;
  constexpr int AlignmentB =
      128 / cutlass::sizeof_bits<ElementBUnderlying>::value;
  constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

  using LayoutA_Transpose =
      typename cutlass::layout::LayoutTranspose<LayoutA>::type;
  using LayoutB_Transpose =
      typename cutlass::layout::LayoutTranspose<LayoutB>::type;

  using ElementGlobalScale = float;
  using ElementAccumulator = float;
  using ArchTag = cutlass::arch::Sm100;
  using StageCountType = cutlass::gemm::collective::StageCountAuto;
  using TileShape =
      cute::Shape<cute::Int<TB_M>, cute::Int<TB_N>, cute::Int<TB_K>>;
  using ClusterShape =
      cute::Shape<cute::Int<TBS_M>, cute::Int<TBS_N>, cute::Int<TBS_K>>;

  using KernelSchedule = cute::conditional_t<
      is_nvfp4,
      cute::conditional_t<
          (TB_M == 256) && (TBS_M % 2 == 0),
          cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmNvf4Sm100,
          cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmNvf4Sm100>,
      cute::conditional_t<
          (TB_M == 256) && (TBS_M % 2 == 0),
          cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmMxf4Sm100,
          cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmMxf4Sm100>>;
  using EpilogueSchedule = cute::conditional_t<
      (TB_M == 256) && (TBS_M % 2 == 0),
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
          //   void, // Indicate there is no beta scaling to save register
          //   space.
          ElementC,
          typename cutlass::layout::LayoutTranspose<LayoutC>::type*,
          AlignmentC,
          ElementC,
          typename cutlass::layout::LayoutTranspose<LayoutC>::type*,
          AlignmentC,
          EpilogueSchedule>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag,
          cutlass::arch::OpClassBlockScaledTensorOp,
          ElementB,
          LayoutB_Transpose*,
          AlignmentB,
          ElementA,
          LayoutA_Transpose*,
          AlignmentA,
          ElementAccumulator,
          TileShape,
          ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          KernelSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::
      GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideC = typename Gemm::GemmKernel::InternalStrideD;

  using ElementScale = typename ElementA::ScaleFactorType;

  using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::
      InternalLayoutSFA; // Scale Factor tensors have an interleaved layout.
                         // Bring Layout instead of stride.
  using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::
      InternalLayoutSFB; // Scale Factor tensors have an interleaved layout.
                         // Bring Layout instead of stride.
  // For SFA and SFB tensors layouts
  using Sm1xxBlkScaledConfig =
      typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

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

  // X block scales.
  const int64_t x_scale_offset = wq_offset + wq_size_buffer;
  int64_t x_scale_buffer = _byte_align(G * sizeof(ElementScale**));

  // W block scales.
  const int64_t w_scale_offset = x_scale_offset + x_scale_buffer;
  int64_t w_scale_buffer = _byte_align(G * sizeof(ElementScale**));

  // Outputs.
  const int64_t output_offset = w_scale_offset + w_scale_buffer;
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

  // SFA layout
  const int64_t layout_SFA_offset = stride_c_offset + stride_c_buffer;
  int64_t layout_SFA_buffer = _byte_align(G * sizeof(LayoutSFA));

  // SFB layout
  const int64_t layout_SFB_offset = layout_SFA_offset + layout_SFA_buffer;
  int64_t layout_SFB_buffer = _byte_align(G * sizeof(LayoutSFB));

  // Global scale
  const int64_t global_scale_offset = layout_SFB_offset + layout_SFB_buffer;
  int64_t global_scale_buffer = _byte_align(G * sizeof(ElementGlobalScale**));

  // Compute total buffer size
  int64_t total_buffer_size = global_scale_offset + global_scale_buffer;

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
  const ElementScale** x_scale_ptr =
      reinterpret_cast<const ElementScale**>(kernel_args_ptr + x_scale_offset);
  const ElementScale** w_scale_ptr =
      reinterpret_cast<const ElementScale**>(kernel_args_ptr + w_scale_offset);
  ElementC** output_ptr =
      reinterpret_cast<ElementC**>(kernel_args_ptr + output_offset);
  StrideA* stride_a_ptr =
      reinterpret_cast<StrideA*>(kernel_args_ptr + stride_a_offset);
  StrideB* stride_b_ptr =
      reinterpret_cast<StrideB*>(kernel_args_ptr + stride_b_offset);
  StrideC* stride_c_ptr =
      reinterpret_cast<StrideC*>(kernel_args_ptr + stride_c_offset);
  LayoutSFA* layout_SFA =
      reinterpret_cast<LayoutSFA*>(kernel_args_ptr + layout_SFA_offset);
  LayoutSFB* layout_SFB =
      reinterpret_cast<LayoutSFB*>(kernel_args_ptr + layout_SFB_offset);
  const ElementGlobalScale** global_scale_ptr =
      reinterpret_cast<const ElementGlobalScale**>(
          kernel_args_ptr + global_scale_offset);

  auto stream = at::cuda::getCurrentCUDAStream().stream();

  TORCH_CHECK(
      M_sizes.has_value() && M_sizes->dtype() == at::kLong,
      "M_sizes must be int64.");
  // When m_offsets is used, XQ is shape [total_M, K].
  int64_t M = XQ.size(XQ.dim() - 2);
  int64_t N = WQ.size(1);
  int64_t K = WQ.size(2) * 2; // Since K is packed

  set_stacked_kernel_args_kernel<
      ProblemShape::UnderlyingProblemShape,
      ElementA,
      ElementB,
      ElementC,
      ElementScale,
      StrideA,
      StrideB,
      StrideC,
      LayoutSFA,
      LayoutSFB,
      ElementGlobalScale,
      Sm1xxBlkScaledConfig><<<1, G, 0, stream>>>(
      G,
      N,
      K,
      problem_shape_ptr,
      reinterpret_cast<ElementA*>(XQ.data_ptr()),
      xq_ptr,
      reinterpret_cast<ElementB*>(WQ.data_ptr()),
      wq_ptr,
      reinterpret_cast<ElementScale*>(x_scale.data_ptr()),
      x_scale_ptr,
      reinterpret_cast<ElementScale*>(w_scale.data_ptr()),
      w_scale_ptr,
      reinterpret_cast<ElementC*>(output.data_ptr()),
      output_ptr,
      stride_a_ptr,
      stride_b_ptr,
      stride_c_ptr,
      reinterpret_cast<int64_t*>(M_sizes.value().data_ptr()),
      layout_SFA,
      layout_SFB,
      is_nvfp4 ? reinterpret_cast<ElementGlobalScale*>(
                     global_scale.value().data_ptr())
               : nullptr,
      is_nvfp4 ? global_scale_ptr : nullptr,
      reinterpret_cast<int64_t*>(
          starting_row_after_padding.value().data_ptr()));
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
      {reinterpret_cast<const ElementBUnderlying**>(wq_ptr),
       stride_b_ptr,
       reinterpret_cast<const ElementAUnderlying**>(xq_ptr),
       stride_a_ptr,
       reinterpret_cast<const ElementScale**>(w_scale_ptr),
       layout_SFB,
       reinterpret_cast<const ElementScale**>(x_scale_ptr),
       layout_SFA},
      {{}, nullptr, stride_c_ptr, output_ptr, stride_c_ptr},
      hw_info};

  if constexpr (is_nvfp4) {
    auto& fusion_args = arguments.epilogue.thread;
    fusion_args.alpha = 0;
    fusion_args.alpha_ptr_array = global_scale_ptr;
    fusion_args.dAlpha = {cute::Int<0>{}, cute::Int<0>{}, 1};
  }

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

} // namespace fbgemm_gpu

#endif
