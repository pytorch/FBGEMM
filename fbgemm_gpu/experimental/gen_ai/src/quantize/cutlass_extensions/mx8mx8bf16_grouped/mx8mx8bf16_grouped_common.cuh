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

#include "cutlass_extensions/include/grouped_common.cuh"

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12080)

namespace fbgemm_gpu {

inline int64_t _byte_align(int64_t offset) {
  int64_t remainder = offset % 16;
  if (remainder != 0) {
    offset += (16 - remainder);
  }
  return offset;
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
  set_grouped_gemm_args_kernel<
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

} // namespace fbgemm_gpu

#endif
