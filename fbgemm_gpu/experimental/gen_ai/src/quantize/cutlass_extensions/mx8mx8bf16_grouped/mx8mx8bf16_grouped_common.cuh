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

enum GroupedGemmInputType {
  // K dynamic
  _2D2D,
  // M dynamic (MoE style)
  _2D3D
};

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
    typename ScaleDtype,
    typename StrideA,
    typename StrideB,
    typename StrideC,
    typename LayoutSFA,
    typename LayoutSFB,
    typename Sm1xxBlkScaledConfig>
__global__ void set_stacked_kernel_args_kernel(
    int64_t G,
    int64_t M,
    int64_t N,
    int64_t K,
    ProblemShape* problem_shape_ptr,
    ElementA* xq,
    const ElementA** xq_ptr,
    ElementB* wq,
    const ElementB** wq_ptr,
    ScaleDtype* x_scale,
    const ScaleDtype** x_scale_ptr,
    ScaleDtype* w_scale,
    const ScaleDtype** w_scale_ptr,
    ElementC* output,
    ElementC** output_ptr,
    StrideA* stride_a_ptr,
    StrideB* stride_b_ptr,
    StrideC* stride_c_ptr,
    int32_t* offsets, // Group end offsets
    LayoutSFA* layout_SFA,
    LayoutSFB* layout_SFB,
    GroupedGemmInputType gemm_type) {
  uint32_t group_index = blockIdx.x * blockDim.x + threadIdx.x;
  // If this thread corresponds to a valid group, write kernel args to device
  // memory.
  if (group_index < G) {
    // Set problem shapes to empty by default.
    problem_shape_ptr[group_index] = ProblemShape(0, 0, 0);

    // Offsets for this group.
    int64_t xq_offset = 0;
    int64_t wq_offset = 0;
    int64_t output_offset = 0;
    int64_t x_scale_offset = 0;
    int64_t w_scale_offset = 0;

    auto round_up = [](int64_t x, int64_t y) { return ((x + y - 1) / y) * y; };

    // Pre-compute common rounded values to minimize round_up calls
    const int64_t N_rounded = round_up(N, 128);
    const int64_t M_rounded = round_up(M, 128);

    // Handle offsets API (torch compliant API for 2D-2D and 2D-3D inputs from
    // mx8mx8bf16_grouped)
    CUDA_KERNEL_ASSERT(
        offsets != nullptr &&
        "offsets must be set for 2d-2d and 2d-3d grouped GEMMs");
    switch (gemm_type) {
      // In the 2d-2d case, contraction dim (total_K) has variable group
      // sizes. XQ = (M, total_K) WQ = (N, total_K) Main loop defined with WQ
      // @ XQ^T = (N, M) for each group. out = (G, N, M)
      case GroupedGemmInputType::_2D2D: {
        // `offsets` contains end index of each group.
        const int32_t prev_group_end_offset =
            (group_index == 0) ? 0 : offsets[group_index - 1];
        const int32_t curr_group_end_offset = offsets[group_index];
        const int32_t K_group_size =
            curr_group_end_offset - prev_group_end_offset;

        // Validate group offsets.
        int align = 128 / cutlass::sizeof_bits<ElementA>::value;
        CUDA_KERNEL_ASSERT(
            K_group_size % align == 0 &&
            "for 2d-2d grouped gemm, group sizes along K dim must be non-negative multiple of 16\n");
        CUDA_KERNEL_ASSERT(
            curr_group_end_offset <= K &&
            "for 2d-2d grouped gemm, group end offsets must be non-negative and must be <= K\n");

        // Set starting input offsets for this group.
        // XQ is shape (M,K) with strides (K, 1) and group offsets are along
        // the K dim, so: xq_offset -> prev_group_end_offset * 1
        xq_offset = prev_group_end_offset;

        // WQ is shape (N,K) with strides (K, 1) and group offsets are along
        // the K dim, so: wq_offset -> prev_group_end_offset * 1
        wq_offset = prev_group_end_offset;

        // Output for 2d-2d grouped GEMM is shape (G, M, N)
        // output_offset -> group_index rows with stride of M * N
        output_offset = group_index * M * N;

        // Group sizes are variable and converted to blocked/padded format, so
        // to calculate the starting offset of this group's scales, we do the
        // following: For each previous group
        // - Calculate the expected size of its blocked formatted scales
        // - Increment the scale offsets by that size
        // x_scale shape (M_rounded, total_K_padded_per_group).
        // w_scale has shape (N_rounded, total_K_padded_per_group).
        for (int i = 0; i < group_index; i++) {
          int group_i_size = i == 0 ? offsets[i] : offsets[i] - offsets[i - 1];
          int scale_cols_for_group_i_padded = round_up(group_i_size / 32, 4);
          x_scale_offset += M_rounded * scale_cols_for_group_i_padded;
          w_scale_offset += N_rounded * scale_cols_for_group_i_padded;
        }

        // Only write kernel args if this group is non-empty
        if (K_group_size > 0) {
          // Get index automatically for this group
          int total_K = K; // Name alias for clarity/readability.

          // Set problem shape.
          // Main loop passes inputs in B,A order, so we have: (N, K_group) @
          // (M, K_group)^T = (N, M) for each group.
          problem_shape_ptr[group_index] = ProblemShape(N, M, K_group_size);

          // Set pointers for this group.
          xq_ptr[group_index] = xq + xq_offset;
          wq_ptr[group_index] = wq + wq_offset;
          x_scale_ptr[group_index] = x_scale + x_scale_offset;
          w_scale_ptr[group_index] = w_scale + w_scale_offset;
          output_ptr[group_index] = output + output_offset;

          // Set strides.
          // TODO: make strides configurable to handle all NT/TN/NN/NT layouts
          // that Blackwell supports. For XQ, the group processes a slice (M,
          // K_group_size) but it's part of a larger tensor (M, total_K). The
          // stride needs to reflect that rows are separated by total_K
          // elements in the original tensor.
          stride_a_ptr[group_index] = cutlass::make_cute_packed_stride(
              StrideA{}, cute::make_shape(int(M), int(total_K), 1));

          // For WQ, the group processes a slice (N, K_group_size) but it's
          // part of a larger tensor (N, total_K). The stride needs to reflect
          // that rows are separated by total_K elements in the original
          // tensor.
          stride_b_ptr[group_index] = cutlass::make_cute_packed_stride(
              StrideB{}, cute::make_shape(int(N), int(total_K), 1));

          // For output of this group, (M, K_group_size) @ (N, K_group_size)^T
          // = (M, N)
          stride_c_ptr[group_index] = cutlass::make_cute_packed_stride(
              StrideC{}, cute::make_shape(int(N), int(M), 1));

          // Set layouts for scale factors.
          // Groups of variable size are along the K dim, so we need to
          // calculate the size of the blocked group scale factor here.
          layout_SFA[group_index] =
              Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
                  cute::make_shape(int(M), int(N), int(K_group_size), 1));
          layout_SFB[group_index] =
              Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
                  cute::make_shape(int(M), int(N), int(K_group_size), 1));
        }
        break;
      }
      case GroupedGemmInputType::_2D3D: {
        // `offsets` contains end index of each group.
        const int32_t prev_group_end_offset =
            (group_index == 0) ? 0 : offsets[group_index - 1];
        const int32_t curr_group_end_offset = offsets[group_index];
        const int32_t M_group_size =
            curr_group_end_offset - prev_group_end_offset;

        if (M_group_size > 0) {
          // Validate group offsets.
          CUDA_KERNEL_ASSERT(
              curr_group_end_offset <= M &&
              "for 2d-3d grouped gemm, group end offsets must be non-negative and must be <= M\n");

          // Compute starting offset for this group when M_group size > 0
          int64_t group_offset_M =
              group_index == 0 ? 0 : offsets[group_index - 1];
          int64_t scale_group_offset_M = 0;
          for (int i = 0; i < group_index; i++) {
            // Group offset on XQ along total_M dim is the sum of all previous
            // group sizes.
            int group_i_size =
                i == 0 ? offsets[i] : offsets[i] - offsets[i - 1];

            // Scale group offset on x_scale is sum of all previous scale
            // group sizes.
            int scale_group_rows_padded = round_up(group_i_size, 128);
            scale_group_offset_M += scale_group_rows_padded;
          }

          // wq_offset -> group_offset_M rows with stride of K
          xq_offset = group_offset_M * K;

          // wq_offset -> group_index rows with stride of N * K (3d tensor)
          wq_offset = group_index * N * K;

          // output_offset -> group_offset_M rows with stride of N
          output_offset = group_offset_M * N;

          // x_scale offset -> sum of all padded group sizes (rows) * rounded
          // scale group cols
          const int64_t K_div_32_rounded = round_up(K / 32, 4);
          x_scale_offset = scale_group_offset_M * K_div_32_rounded;

          // w_scale_offset -> group_index rows with stride of (N rounded to
          // nearest multiple of 128 * K rounded to nearest multiple of 4)
          w_scale_offset = group_index * N_rounded * K_div_32_rounded;

          // Set problem shape
          problem_shape_ptr[group_index] = ProblemShape(N, M_group_size, K);

          // Set pointers
          xq_ptr[group_index] = xq + xq_offset;
          wq_ptr[group_index] = wq + wq_offset;
          x_scale_ptr[group_index] = x_scale + x_scale_offset;
          w_scale_ptr[group_index] = w_scale + w_scale_offset;
          output_ptr[group_index] = output + output_offset;

          // Set strides
          stride_a_ptr[group_index] = cutlass::make_cute_packed_stride(
              StrideA{}, cute::make_shape(int(M_group_size), int(K), 1));
          stride_b_ptr[group_index] = cutlass::make_cute_packed_stride(
              StrideB{}, cute::make_shape(int(N), int(K), 1));
          stride_c_ptr[group_index] = cutlass::make_cute_packed_stride(
              StrideC{}, cute::make_shape(int(N), int(M_group_size), 1));

          // Set layouts for scale factors
          layout_SFA[group_index] =
              Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
                  cute::make_shape(int(M_group_size), int(N), int(K), 1));
          layout_SFB[group_index] =
              Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
                  cute::make_shape(int(M_group_size), int(N), int(K), 1));
        }
        break;
      }
    }
  }
}

/*
    MXFP8 grouped GEMM that performs, which handles 2 cases:

    1. XQ 2d, WQ 3d:
       XQ shape = (total_M, K) where groups are along the M dimension
       WQ shape = (N, K)
       out shape = (total_M, N)

    2. XQ 2d, WQ 2d:
       XQ shape = (M, total_K) where groups are along the K dimension
       WQ shape = (N, total_K) where groups are along the K dimension
       out shape = (num_groups, M, N)
*/
template <
    typename InputType,
    int TB_M,
    int TB_N,
    int TB_K,
    int TBS_M,
    int TBS_N,
    int TBS_K>
at::Tensor mx8mx8bf16_grouped_impl(
    InputType XQ, // FP8
    InputType WQ, // FP8
    InputType x_scale,
    InputType w_scale,
    at::Tensor output,
    int64_t G,
    at::Tensor offsets) {
  // The number of groups the kernel uses may vary.
  int kernel_groups = G;

  at::TensorOptions options;
  options = XQ.options();

  // Return early if there are no elements in the output.
  if (output.numel() == 0) {
    return output;
  }

  // WQ is shape (K,N) or (E,K,N) in column major layout, to align with
  // torch._scaled_grouped_mm. We transpose here to match cutlass kernel
  // requirements.
  InputType WQ_contig = WQ.transpose(-2, -1);

  // Define gemm configuration.
  using ProblemShape =
      cutlass::gemm::GroupProblemShape<cute::Shape<int, int, int>>;
  using ElementA = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
  using ElementB = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
  using ElementC = cutlass::bfloat16_t;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutA_Transpose =
      typename cutlass::layout::LayoutTranspose<LayoutA>::type;
  using LayoutB_Transpose =
      typename cutlass::layout::LayoutTranspose<LayoutB>::type;
  constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
  constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
  using ElementAccumulator = float;
  using ArchTag = cutlass::arch::Sm100;
  using StageCountType = cutlass::gemm::collective::StageCountAuto;
  using TileShape =
      cute::Shape<cute::Int<TB_M>, cute::Int<TB_N>, cute::Int<TB_K>>;
  using ClusterShape =
      cute::Shape<cute::Int<TBS_M>, cute::Int<TBS_N>, cute::Int<TBS_K>>;

  using KernelSchedule = cute::conditional_t<
      (TB_M == 256) && (TBS_M % 2 == 0),
      cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmMxf8f6f4Sm100,
      cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmMxf8f6f4Sm100>;
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
          128 / cutlass::sizeof_bits<ElementC>::value,
          ElementC,
          typename cutlass::layout::LayoutTranspose<LayoutC>::type*,
          128 / cutlass::sizeof_bits<ElementC>::value,
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

  using ScaleDtype = typename ElementA::ScaleFactorType;

  using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::
      InternalLayoutSFA; // Scale Factor tensors have an interleaved layout.
                         // Bring Layout instead of stride.
  using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::
      InternalLayoutSFB; // Scale Factor tensors have an interleaved layout.
                         // Bring Layout instead of stride.

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
  int64_t x_scale_buffer = _byte_align(G * sizeof(ScaleDtype**));

  // W block scales.
  const int64_t w_scale_offset = x_scale_offset + x_scale_buffer;
  int64_t w_scale_buffer = _byte_align(G * sizeof(ScaleDtype**));

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

  // Compute total buffer size
  int64_t total_buffer_size = layout_SFB_offset + layout_SFB_buffer;

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
  const ScaleDtype** x_scale_ptr =
      reinterpret_cast<const ScaleDtype**>(kernel_args_ptr + x_scale_offset);
  const ScaleDtype** w_scale_ptr =
      reinterpret_cast<const ScaleDtype**>(kernel_args_ptr + w_scale_offset);
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

  auto stream = at::cuda::getCurrentCUDAStream().stream();

  // For SFA and SFB tensors layouts
  using Sm1xxBlkScaledConfig =
      typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  TORCH_CHECK(x_scale.dim() == 2, "x_scale must be a 2D tensor");
  TORCH_CHECK(
      w_scale.dim() == 2 || w_scale.dim() == 3,
      "w_scale must be either 2D or 3D tensor");

  int64_t M = XQ.size(0);
  int64_t N = WQ_contig.size(-2);
  int64_t K = WQ_contig.size(-1);
  int32_t* offsets_ptr = reinterpret_cast<int32_t*>(offsets.data_ptr());

  // Determine gemm type.
  GroupedGemmInputType gemm_type;
  if (XQ.dim() == 2 && WQ_contig.dim() == 2) {
    gemm_type = GroupedGemmInputType::_2D2D;
  } else if (XQ.dim() == 2 && WQ_contig.dim() == 3) {
    gemm_type = GroupedGemmInputType::_2D3D;
  } else {
    TORCH_CHECK(
        false,
        "Invalid input dimensions. MXFP8 grouped GEMM currently only supports 2D-2D and 2D-3D inputs.");
  }

  // Execute kernel to dynamically set kernel arguments for each group.
  set_stacked_kernel_args_kernel<
      ProblemShape::UnderlyingProblemShape,
      ElementA,
      ElementB,
      ElementC,
      ScaleDtype,
      StrideA,
      StrideB,
      StrideC,
      LayoutSFA,
      LayoutSFB,
      Sm1xxBlkScaledConfig><<<1, G, 0, stream>>>(
      G,
      M,
      N,
      K,
      problem_shape_ptr,
      reinterpret_cast<ElementA*>(XQ.data_ptr()),
      xq_ptr,
      reinterpret_cast<ElementB*>(WQ_contig.data_ptr()),
      wq_ptr,
      reinterpret_cast<ScaleDtype*>(x_scale.data_ptr()),
      x_scale_ptr,
      reinterpret_cast<ScaleDtype*>(w_scale.data_ptr()),
      w_scale_ptr,
      reinterpret_cast<ElementC*>(output.data_ptr()),
      output_ptr,
      stride_a_ptr,
      stride_b_ptr,
      stride_c_ptr,
      offsets_ptr,
      layout_SFA,
      layout_SFB,
      gemm_type);

  cutlass::KernelHardwareInfo hw_info;
  // Change device_id to another value if you are running on a machine with
  // multiple GPUs and wish to use a GPU other than that with device ID 0.
  hw_info.device_id = 0;
  hw_info.sm_count =
      min(cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
              hw_info.device_id),
          2147483647); // INT_MAX

  using DataTypeA = typename ElementA::DataType;
  using DataTypeB = typename ElementB::DataType;

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {kernel_groups, problem_shape_ptr, nullptr},
      {
          reinterpret_cast<const DataTypeB**>(wq_ptr),
          stride_b_ptr,
          reinterpret_cast<const DataTypeA**>(xq_ptr),
          stride_a_ptr,
          reinterpret_cast<const ScaleDtype**>(w_scale_ptr),
          layout_SFB,
          reinterpret_cast<const ScaleDtype**>(x_scale_ptr),
          layout_SFA,
      },
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
