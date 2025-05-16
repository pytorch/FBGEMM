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

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12080)

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
    typename ElementComputeEpilogue,
    typename StrideA,
    typename StrideB,
    typename StrideC,
    typename LayoutSFA,
    typename LayoutSFB,
    typename ElementGlobalScale,
    int SFVecSize>
__global__ void set_kernel_args_kernel(
    int i, // Group index
    int64_t G, // Total groups.
    int64_t M,
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
    LayoutSFA* layout_SFA,
    LayoutSFB* layout_SFB,
    ElementGlobalScale* global_scale,
    const ElementGlobalScale** global_scale_ptr) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  // Each kernel annoyingly can only set the kernel args for one group.
  // This could only be avoided with complicated memory management.
  if (idx == 0) {
    problem_shape_ptr[i] = ProblemShape(N, M, K);
    xq_ptr[i] = xq;
    wq_ptr[i] = wq;
    x_scale_ptr[i] = x_scale;
    w_scale_ptr[i] = w_scale;
    output_ptr[i] = output;
    stride_a_ptr[i] = cutlass::make_cute_packed_stride(
        StrideA{}, cute::make_shape(int(M), int(K), 1));
    stride_b_ptr[i] = cutlass::make_cute_packed_stride(
        StrideB{}, cute::make_shape(int(N), int(K), 1));
    stride_c_ptr[i] = cutlass::make_cute_packed_stride(
        StrideC{}, cute::make_shape(int(N), int(M), 1));
    layout_SFA[i] = cutlass::detail::Sm100BlockScaledConfig<SFVecSize>::
        tile_atom_to_shape_SFA(cute::make_shape(int(M), int(N), int(K), 1));
    layout_SFB[i] = cutlass::detail::Sm100BlockScaledConfig<SFVecSize>::
        tile_atom_to_shape_SFB(cute::make_shape(int(M), int(N), int(K), 1));
    if (global_scale != nullptr) {
      global_scale_ptr[i] = global_scale;
    }
  }
}

template <
    typename ProblemShape,
    typename ElementA,
    typename ElementB,
    typename ElementC,
    typename ElementComputeEpilogue,
    typename StrideA,
    typename StrideB,
    typename StrideC,
    typename LayoutSFA,
    typename LayoutSFB,
    typename ElementGlobalScale,
    int SFVecSize>
__global__ void set_stacked_kernel_args_kernel(
    int64_t G,
    int64_t N,
    int64_t K,
    int64_t num_x_scale_per_group,
    int64_t num_w_scale_per_group,
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
    int64_t* M_sizes,
    LayoutSFA* layout_SFA,
    LayoutSFB* layout_SFB,
    ElementGlobalScale* global_scale,
    const ElementGlobalScale** global_scale_ptr) {
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
      xq_ptr[non_zero_idx] = xq + (offset_M * K / 2);
      wq_ptr[non_zero_idx] = wq + (group_index * N * K / 2);
      x_scale_ptr[non_zero_idx] =
          x_scale + (group_index * num_x_scale_per_group);
      w_scale_ptr[non_zero_idx] =
          w_scale + (group_index * num_w_scale_per_group);
      output_ptr[non_zero_idx] = output + (offset_M * N);
      stride_a_ptr[non_zero_idx] = cutlass::make_cute_packed_stride(
          StrideA{}, cute::make_shape(int(M), int(K), 1));
      stride_b_ptr[non_zero_idx] = cutlass::make_cute_packed_stride(
          StrideB{}, cute::make_shape(int(N), int(K), 1));
      stride_c_ptr[non_zero_idx] = cutlass::make_cute_packed_stride(
          StrideC{}, cute::make_shape(int(N), int(M), 1));
      layout_SFA[non_zero_idx] = cutlass::detail::
          Sm100BlockScaledConfig<SFVecSize>::tile_atom_to_shape_SFA(
              cute::make_shape(int(M), int(N), int(K), 1));
      layout_SFB[non_zero_idx] = cutlass::detail::
          Sm100BlockScaledConfig<SFVecSize>::tile_atom_to_shape_SFB(
              cute::make_shape(int(M), int(N), int(K), 1));
      if (global_scale != nullptr) {
        global_scale_ptr[non_zero_idx] = global_scale + group_index;
      }
    }
  }
}

template <
    typename InputType,
    typename InputQuantType,
    int TB_M,
    int TB_N,
    int TB_K,
    int TBS_M,
    int TBS_N,
    int TBS_K,
    int SFVecSize>
at::Tensor f4f4bf16_grouped_impl(
    InputType XQ, // FP4
    InputType WQ, // FP4
    InputType x_scale,
    InputType w_scale,
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes,
    std::optional<InputType> global_scale) {
  int64_t G;
  at::TensorOptions options;
  if constexpr (std::is_same_v<InputType, at::TensorList>) {
    G = XQ.size();
    options = XQ[0].options();
    TORCH_CHECK(WQ.size() == G);
  } else {
    TORCH_CHECK(
        zero_start_index_M.has_value() != M_sizes.has_value(),
        "One of zero_start_index_M or M_sizes must be provided.");
    G = WQ.size(0);
    options = XQ.options();
  }

  // The number of groups the kernel uses may vary.
  int kernel_groups = G;
  // Return early if there are no elements in the output.
  if (output.numel() == 0) {
    return output;
  }

  // Define gemm configuration.
  using ProblemShape =
      cutlass::gemm::GroupProblemShape<cute::Shape<int, int, int>>;
  using ElementA = InputQuantType;
  using ElementB = InputQuantType;
  using ElementC = cutlass::bfloat16_t;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutA_Transpose =
      typename cutlass::layout::LayoutTranspose<LayoutA>::type;
  using LayoutB_Transpose =
      typename cutlass::layout::LayoutTranspose<LayoutB>::type;
  constexpr int AlignmentA = 32;
  constexpr int AlignmentB = 32;
  using ElementGlobalScale = float;
  using ElementAccumulator = float;
  using ArchTag = cutlass::arch::Sm100;
  using StageCountType = cutlass::gemm::collective::StageCountAuto;
  using TileShape =
      cute::Shape<cute::Int<TB_M>, cute::Int<TB_N>, cute::Int<TB_K>>;
  using ClusterShape =
      cute::Shape<cute::Int<TBS_M>, cute::Int<TBS_N>, cute::Int<TBS_K>>;

  using KernelSchedule = cute::conditional_t<
      std::is_same_v<
          InputQuantType,
          cutlass::nv_float4_t<cutlass::float_e2m1_t>>,
      cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmNvf4Sm100,
      cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmMxf4Sm100>;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm;

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

  using ElementComputeEpilogue = typename ElementA::ScaleFactorType;

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
  int64_t x_scale_buffer = _byte_align(G * sizeof(ElementComputeEpilogue**));

  // W block scales.
  const int64_t w_scale_offset = x_scale_offset + x_scale_buffer;
  int64_t w_scale_buffer = _byte_align(G * sizeof(ElementComputeEpilogue**));

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
  int64_t global_scale_buffer = _byte_align(G * sizeof(ElementGlobalScale));

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
  LayoutSFA* layout_SFA =
      reinterpret_cast<LayoutSFA*>(kernel_args_ptr + layout_SFA_offset);
  LayoutSFB* layout_SFB =
      reinterpret_cast<LayoutSFB*>(kernel_args_ptr + layout_SFB_offset);
  const ElementGlobalScale** global_scale_ptr =
      reinterpret_cast<const ElementGlobalScale**>(
          kernel_args_ptr + global_scale_offset);

  auto stream = at::cuda::getCurrentCUDAStream().stream();

  if constexpr (std::is_same_v<
                    InputQuantType,
                    cutlass::nv_float4_t<cutlass::float_e2m1_t>>) {
    TORCH_CHECK(global_scale.has_value(), "global_scale is required in nvfp4.");
  }

  // Set kernel arguments for tensor list inputs.
  // The strategy here is to iterate over each group and set the corresponding
  // device memory separately. This is the best way to allow true dynamic
  // shapes.
  if constexpr (std::is_same_v<InputType, at::TensorList>) {
    int64_t _output_offset = 0;
    for (int i = 0; i < G; ++i) {
      // Compute buffer pointers based on input type.
      int64_t M = XQ[i].size(0);
      int64_t N = WQ[i].size(0);
      int64_t K = XQ[i].size(1) * 2; // Since K is packed

      // Launch a kernel to set one group's arguments.
      // NVFP4 args kernel
      if constexpr (std::is_same_v<
                        InputQuantType,
                        cutlass::nv_float4_t<cutlass::float_e2m1_t>>) {
        set_kernel_args_kernel<
            ProblemShape::UnderlyingProblemShape,
            ElementA,
            ElementB,
            ElementC,
            ElementComputeEpilogue,
            StrideA,
            StrideB,
            StrideC,
            LayoutSFA,
            LayoutSFB,
            ElementGlobalScale,
            SFVecSize><<<1, 1, 0, stream>>>(
            i,
            G,
            M,
            N,
            K,
            problem_shape_ptr,
            reinterpret_cast<ElementA*>(XQ[i].data_ptr()),
            xq_ptr,
            reinterpret_cast<ElementB*>(WQ[i].data_ptr()),
            wq_ptr,
            reinterpret_cast<ElementComputeEpilogue*>(x_scale[i].data_ptr()),
            x_scale_ptr,
            reinterpret_cast<ElementComputeEpilogue*>(w_scale[i].data_ptr()),
            w_scale_ptr,
            (reinterpret_cast<ElementC*>(output.data_ptr()) + _output_offset),
            output_ptr,
            stride_a_ptr,
            stride_b_ptr,
            stride_c_ptr,
            layout_SFA,
            layout_SFB,
            reinterpret_cast<ElementGlobalScale*>(
                global_scale.value()[i].data_ptr()),
            global_scale_ptr);
        _output_offset += M * N;
      }
      // MXFP4 args kernel
      else {
        set_kernel_args_kernel<
            ProblemShape::UnderlyingProblemShape,
            ElementA,
            ElementB,
            ElementC,
            ElementComputeEpilogue,
            StrideA,
            StrideB,
            StrideC,
            LayoutSFA,
            LayoutSFB,
            ElementGlobalScale,
            SFVecSize><<<1, 1, 0, stream>>>(
            i,
            G,
            M,
            N,
            K,
            problem_shape_ptr,
            reinterpret_cast<ElementA*>(XQ[i].data_ptr()),
            xq_ptr,
            reinterpret_cast<ElementB*>(WQ[i].data_ptr()),
            wq_ptr,
            reinterpret_cast<ElementComputeEpilogue*>(x_scale[i].data_ptr()),
            x_scale_ptr,
            reinterpret_cast<ElementComputeEpilogue*>(w_scale[i].data_ptr()),
            w_scale_ptr,
            (reinterpret_cast<ElementC*>(output.data_ptr()) + _output_offset),
            output_ptr,
            stride_a_ptr,
            stride_b_ptr,
            stride_c_ptr,
            layout_SFA,
            layout_SFB,
            nullptr,
            nullptr);
        _output_offset += M * N;
      }
    }
  } else {
    // For Tensor inputs, we can set all group arguments in a single kernel
    // launch.
    TORCH_CHECK(
        !zero_start_index_M.has_value() ||
            zero_start_index_M->dtype() == at::kLong,
        "zero_start_index_M must be int64.");

    TORCH_CHECK(
        !M_sizes.has_value() || M_sizes->dtype() == at::kLong,
        "M_sizes must be int64.");
    // When m_offsets is used, XQ is shape [total_M, K]. When zero_start_index_M
    // is used, shape is [G, M, K].
    int64_t M = XQ.size(XQ.dim() - 2);
    int64_t N = WQ.size(1);
    int64_t K = WQ.size(2) * 2; // Since K is packed

    // Calculate the number of scale elements per group
    int64_t num_x_scale_per_group;
    int64_t num_w_scale_per_group;
    TORCH_CHECK(
        x_scale.dim() == 2 || x_scale.dim() == 3,
        "x_scale must be either 2D or 3D tensor")
    if (x_scale.dim() == 3) {
      num_x_scale_per_group = x_scale.size(1) * x_scale.size(2);
    } else {
      num_x_scale_per_group = x_scale.size(1);
    }
    TORCH_CHECK(
        w_scale.dim() == 2 || w_scale.dim() == 3,
        "w_scale must be either 2D or 3D tensor")
    if (w_scale.dim() == 3) {
      num_w_scale_per_group = w_scale.size(1) * w_scale.size(2);
    } else {
      num_w_scale_per_group = w_scale.size(1);
    }

    int64_t* M_sizes_ptr =
        reinterpret_cast<int64_t*>(M_sizes.value().data_ptr());
    // NVFP4 stacked args kernel
    if constexpr (std::is_same_v<
                      InputQuantType,
                      cutlass::nv_float4_t<cutlass::float_e2m1_t>>) {
      set_stacked_kernel_args_kernel<
          ProblemShape::UnderlyingProblemShape,
          ElementA,
          ElementB,
          ElementC,
          ElementComputeEpilogue,
          StrideA,
          StrideB,
          StrideC,
          LayoutSFA,
          LayoutSFB,
          ElementGlobalScale,
          SFVecSize><<<1, G, 0, stream>>>(
          G,
          N,
          K,
          num_x_scale_per_group,
          num_w_scale_per_group,
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
          M_sizes_ptr,
          layout_SFA,
          layout_SFB,
          reinterpret_cast<ElementGlobalScale*>(
              global_scale.value().data_ptr()),
          global_scale_ptr);
    }
    // MXFP4 stacked args kernel
    else {
      set_stacked_kernel_args_kernel<
          ProblemShape::UnderlyingProblemShape,
          ElementA,
          ElementB,
          ElementC,
          ElementComputeEpilogue,
          StrideA,
          StrideB,
          StrideC,
          LayoutSFA,
          LayoutSFB,
          ElementGlobalScale,
          SFVecSize><<<1, G, 0, stream>>>(
          G,
          N,
          K,
          num_x_scale_per_group,
          num_w_scale_per_group,
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
          M_sizes_ptr,
          layout_SFA,
          layout_SFB,
          nullptr,
          nullptr);
    }
    // Set the number of groups to the kernel to be at most the number of
    // non-zero rows.
    kernel_groups = int(std::min(M, G));
  }

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
      {reinterpret_cast<const DataTypeB**>(wq_ptr),
       stride_b_ptr,
       reinterpret_cast<const DataTypeA**>(xq_ptr),
       stride_a_ptr,
       reinterpret_cast<const ElementComputeEpilogue**>(w_scale_ptr),
       layout_SFB,
       reinterpret_cast<const ElementComputeEpilogue**>(x_scale_ptr),
       layout_SFA},
      {{}, nullptr, stride_c_ptr, output_ptr, stride_c_ptr},
      hw_info};

  if constexpr (std::is_same_v<
                    InputQuantType,
                    cutlass::nv_float4_t<cutlass::float_e2m1_t>>) {
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

template <typename InputType>
at::Tensor dispatch_fp4_grouped_kernel(
    int total_M,
    InputType XQ, // FP4
    InputType WQ, // FP4
    InputType x_scale,
    InputType w_scale,
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M = std::nullopt,
    std::optional<at::Tensor> M_sizes = std::nullopt,
    std::optional<InputType> global_scale = std::nullopt,
    bool use_mx = true) {
  // MXFP4
  if (use_mx) {
    if (total_M <= 1024) {
      return f4f4bf16_grouped_impl<
          InputType,
          cutlass::mx_float4_t<cutlass::float_e2m1_t>,
          256,
          128,
          256,
          2,
          1,
          1,
          32>(
          XQ,
          WQ,
          x_scale,
          w_scale,
          output,
          zero_start_index_M,
          M_sizes,
          global_scale);
    } else {
      return f4f4bf16_grouped_impl<
          InputType,
          cutlass::mx_float4_t<cutlass::float_e2m1_t>,
          256,
          256,
          256,
          2,
          1,
          1,
          32>(
          XQ,
          WQ,
          x_scale,
          w_scale,
          output,
          zero_start_index_M,
          M_sizes,
          global_scale);
    }
  } // NVFP4
  else {
    if (total_M <= 1024) {
      return f4f4bf16_grouped_impl<
          InputType,
          cutlass::nv_float4_t<cutlass::float_e2m1_t>,
          256,
          256,
          256,
          2,
          1,
          1,
          16>(
          XQ,
          WQ,
          x_scale,
          w_scale,
          output,
          zero_start_index_M,
          M_sizes,
          global_scale);
    } else {
      return f4f4bf16_grouped_impl<
          InputType,
          cutlass::nv_float4_t<cutlass::float_e2m1_t>,
          256,
          256,
          256,
          2,
          1,
          1,
          16>(
          XQ,
          WQ,
          x_scale,
          w_scale,
          output,
          zero_start_index_M,
          M_sizes,
          global_scale);
    }
  }
}

template <typename OutputType>
OutputType _f4f4bf16_grouped(
    at::TensorList XQ, // FP4
    at::TensorList WQ, // FP4
    at::TensorList x_scale,
    at::TensorList w_scale,
    std::optional<at::TensorList> global_scale,
    bool use_mx) {
  at::Tensor Y;
  int64_t total_M = 0;
  int64_t G = XQ.size();

  // Allocate output tensor.
  std::vector<int64_t> output_sizes;
  int64_t total_output_size = 0;
  for (int i = 0; i < G; ++i) {
    int64_t M = XQ[i].size(0);
    int64_t N = WQ[i].size(0);
    total_M += M;
    const int64_t output_size = M * N;
    total_output_size += output_size;
    output_sizes.push_back(output_size);
  }
  Y = at::empty(total_output_size, XQ[0].options().dtype(at::kBFloat16));

  // Run kernel.
  at::Tensor g_out = dispatch_fp4_grouped_kernel<at::TensorList>(
      total_M,
      XQ,
      WQ,
      x_scale,
      w_scale,
      Y,
      std::nullopt,
      std::nullopt,
      global_scale,
      use_mx);

  // Return appropriate output type.
  if constexpr (std::is_same_v<OutputType, at::Tensor>) {
    int64_t N = WQ[0].size(0);
    return g_out.view({total_M, N});
  } else {
    // Return grouped view of output.
    std::vector<at::Tensor> output_group = g_out.split(output_sizes);
    for (int i = 0; i < G; ++i) {
      output_group[i] = output_group[i].view({XQ[i].size(0), WQ[i].size(0)});
    }
    return output_group;
  }
}

std::vector<at::Tensor> f4f4bf16_grouped(
    at::TensorList XQ, // FP4
    at::TensorList WQ, // FP4
    at::TensorList x_scale,
    at::TensorList w_scale,
    std::optional<at::TensorList> global_scale = std::nullopt,
    bool use_mx = true) {
  return _f4f4bf16_grouped<std::vector<at::Tensor>>(
      XQ, WQ, x_scale, w_scale, global_scale, use_mx);
}

at::Tensor f4f4bf16_grouped_stacked(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor M_sizes,
    std::optional<at::Tensor> global_scale = std::nullopt,
    bool use_mx = true) {
  int64_t total_M = XQ.size(0);
  int64_t N = WQ.size(1);
  int64_t G = M_sizes.size(0);
  TORCH_CHECK(
      M_sizes.device() == XQ.device(),
      "M_sizes must be on same device as inputs.");
  TORCH_CHECK(
      WQ.dim() == 3 && WQ.size(0) == G, "Weights should be shape [G, N, K].")
  at::Tensor Y = at::empty({total_M, N}, XQ.options().dtype(at::kBFloat16));
  // Early exit for empty inputs.
  if (total_M == 0) {
    return Y;
  }
  // Return continuous view of output.
  return dispatch_fp4_grouped_kernel<at::Tensor>(
      total_M,
      XQ,
      WQ,
      x_scale,
      w_scale,
      Y,
      std::nullopt,
      M_sizes,
      global_scale,
      use_mx);
}

#else

std::vector<at::Tensor> f4f4bf16_grouped(
    at::TensorList XQ, // FP4
    at::TensorList WQ, // FP4
    at::TensorList x_scale,
    at::TensorList w_scale,
    std::optional<at::TensorList> global_scale = std::nullopt,
    bool use_mx = true) {
  throw std::runtime_error(
      "CUDA version is older than 12.8"); // requires CUDA>=12.8
}

at::Tensor f4f4bf16_grouped_stacked(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor M_sizes,
    std::optional<at::Tensor> global_scale = std::nullopt,
    bool use_mx = true) {
  throw std::runtime_error(
      "CUDA version is older than 12.8"); // requires CUDA>=12.8
}
#endif

} // namespace fbgemm_gpu
