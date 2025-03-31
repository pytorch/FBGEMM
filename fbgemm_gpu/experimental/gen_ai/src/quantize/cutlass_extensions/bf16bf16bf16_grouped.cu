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
    typename ProblemShape,
    typename ElementA,
    typename ElementB,
    typename ElementC,
    typename StrideA,
    typename StrideB,
    typename StrideC>
__global__ void set_kernel_args_kernel(
    int i, // Group index
    int G, // Total groups.
    int M,
    int N,
    int K,
    ProblemShape* problem_shape_ptr,
    ElementA* x,
    const ElementA** x_ptr,
    ElementB* w,
    const ElementB** w_ptr,
    ElementC* output,
    ElementC** output_ptr,
    StrideA* stride_a_ptr,
    StrideB* stride_b_ptr,
    StrideC* stride_c_ptr) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  // Each kernel annoyingly can only set the kernel args for one group.
  // This could only be avoided with complicated memory management.
  if (idx == 0) {
    problem_shape_ptr[i] = ProblemShape(N, M, K);
    x_ptr[i] = x;
    w_ptr[i] = w;
    output_ptr[i] = output;
    stride_a_ptr[i] =
        cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    stride_b_ptr[i] =
        cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
    stride_c_ptr[i] =
        cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(N, M, 1));
  }
}

template <
    typename ProblemShape,
    typename ElementA,
    typename ElementB,
    typename ElementC,
    typename StrideA,
    typename StrideB,
    typename StrideC>
__global__ void set_dynamic_kernel_args_kernel(
    int G,
    int M,
    int N,
    int K,
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
    int64_t* zero_start_index_M) {
  uint32_t group_index = blockIdx.x * blockDim.x + threadIdx.x;
  // If this thread corresponds to a valid group, write kernel args to device
  // memory.
  if (group_index < G) {
    // Compute shape for this group.
    int kernel_M = zero_start_index_M[group_index];
    int offset_M = group_index * M;
    // Set the problem shape for this group.
    problem_shape_ptr[group_index] = ProblemShape(N, kernel_M, K);
    // Set input pointers.
    x_ptr[group_index] = x + (offset_M * K);
    w_ptr[group_index] = w + (group_index * N * K);
    output_ptr[group_index] = output + (offset_M * N);
    stride_a_ptr[group_index] = cutlass::make_cute_packed_stride(
        StrideA{}, cute::make_shape(kernel_M, K, 1));
    stride_b_ptr[group_index] =
        cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
    stride_c_ptr[group_index] = cutlass::make_cute_packed_stride(
        StrideC{}, cute::make_shape(N, kernel_M, 1));
  }
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
    int G,
    int N,
    int K,
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
      int offset_M = 0;
      for (int i = 0; i < group_index; i++) {
        offset_M += M_sizes[i];
      }
      // Set the problem shape for this group.
      problem_shape_ptr[non_zero_idx] = ProblemShape(N, M, K);
      // Set input pointers.
      x_ptr[non_zero_idx] = x + (offset_M * K);
      w_ptr[non_zero_idx] = w + (group_index * N * K);
      output_ptr[non_zero_idx] = output + (offset_M * N);
      stride_a_ptr[non_zero_idx] = cutlass::make_cute_packed_stride(
          StrideA{}, cute::make_shape(M, K, 1));
      stride_b_ptr[non_zero_idx] = cutlass::make_cute_packed_stride(
          StrideB{}, cute::make_shape(N, K, 1));
      stride_c_ptr[non_zero_idx] = cutlass::make_cute_packed_stride(
          StrideC{}, cute::make_shape(N, M, 1));
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
    bool PONG>
at::Tensor bf16bf16bf16_grouped_impl(
    InputType X, // FP8
    InputType W, // FP8
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes) {
  int G;
  at::TensorOptions options;
  if constexpr (std::is_same_v<InputType, at::TensorList>) {
    G = X.size();
    options = X[0].options();
    TORCH_CHECK(W.size() == G);
  } else {
    TORCH_CHECK(
        zero_start_index_M.has_value() != M_sizes.has_value(),
        "One of zero_start_index_M or M_sizes must be provided.");
    G = W.size(0);
    options = X.options();
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
  using ElementA = cutlass::bfloat16_t;
  using ElementB = cutlass::bfloat16_t;
  using ElementC = cutlass::bfloat16_t;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutA_Transpose =
      typename cutlass::layout::LayoutTranspose<LayoutA>::type;
  using LayoutB_Transpose =
      typename cutlass::layout::LayoutTranspose<LayoutB>::type;
  using ElementAccumulator = float;
  using ArchTag = cutlass::arch::Sm90;
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
          LayoutB_Transpose*,
          128 / cutlass::sizeof_bits<ElementA>::value,
          ElementA,
          LayoutA_Transpose*,
          128 / cutlass::sizeof_bits<ElementB>::value,
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
  // Set kernel arguments for tensor list inputs.
  // The strategy here is to iterate over each group and set the corresponding
  // device memory separately. This is the best way to allow true dynamic
  // shapes.
  if constexpr (std::is_same_v<InputType, at::TensorList>) {
    int64_t output_offset = 0;
    for (int i = 0; i < G; ++i) {
      // Compute buffer pointers based on input type.
      int M = X[i].size(0);
      int N = W[i].size(0);
      int K = X[i].size(1);
      TORCH_CHECK_EQ(W[i].size(1), K);
      // Launch a kernel to set one group's arguments.
      set_kernel_args_kernel<<<1, 1, 0, stream>>>(
          i,
          G,
          M,
          N,
          K,
          problem_shape_ptr,
          reinterpret_cast<ElementA*>(X[i].data_ptr()),
          x_ptr,
          reinterpret_cast<ElementB*>(W[i].data_ptr()),
          w_ptr,
          (reinterpret_cast<ElementC*>(output.data_ptr()) + output_offset),
          output_ptr,
          stride_a_ptr,
          stride_b_ptr,
          stride_c_ptr);
      output_offset += M * N;
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
    // When m_offsets is used, X is shape [total_M, K]. When zero_start_index_M
    // is used, shape is [G, M, K].
    int M = X.size(X.dim() - 2);
    int N = W.size(1);
    int K = W.size(2);
    if (zero_start_index_M.has_value()) {
      int64_t* zero_start_index_M_ptr =
          reinterpret_cast<int64_t*>(zero_start_index_M.value().data_ptr());
      set_dynamic_kernel_args_kernel<<<1, G, 0, stream>>>(
          G,
          M,
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
          zero_start_index_M_ptr);
    } else {
      int64_t* M_sizes_ptr =
          reinterpret_cast<int64_t*>(M_sizes.value().data_ptr());
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
      kernel_groups = std::min(M, G);
    }
  }

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {kernel_groups, problem_shape_ptr, nullptr},
      {w_ptr, stride_b_ptr, x_ptr, stride_a_ptr},
      {{}, nullptr, stride_c_ptr, output_ptr, stride_c_ptr}};

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

// FP8 Tensorwise grouped cutlass kernel dispatch.
template <typename InputType>
at::Tensor dispatch_bf16_grouped_kernel(
    int total_M,
    InputType X, // BF16
    InputType W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M = std::nullopt,
    std::optional<at::Tensor> M_sizes = std::nullopt) {
  // Use heuristics to pick best kernel implementation.
  if (total_M <= 16) {
    return bf16bf16bf16_grouped_impl<InputType, 128, 16, 128, 1, 1, 1, false>(
        X, W, output, zero_start_index_M, M_sizes);
  } else if (total_M <= 32) {
    return bf16bf16bf16_grouped_impl<InputType, 128, 32, 128, 1, 1, 1, false>(
        X, W, output, zero_start_index_M, M_sizes);
  } else if (total_M <= 64) {
    return bf16bf16bf16_grouped_impl<InputType, 128, 64, 128, 1, 1, 1, false>(
        X, W, output, zero_start_index_M, M_sizes);
  } else if (total_M <= 128) {
    return bf16bf16bf16_grouped_impl<InputType, 128, 128, 128, 1, 1, 1, false>(
        X, W, output, zero_start_index_M, M_sizes);
  } else if (total_M <= 512) {
    return bf16bf16bf16_grouped_impl<InputType, 256, 128, 128, 2, 1, 1, false>(
        X, W, output, zero_start_index_M, M_sizes);
  } else {
    return bf16bf16bf16_grouped_impl<InputType, 128, 256, 128, 2, 1, 1, false>(
        X, W, output, zero_start_index_M, M_sizes);
  }
}

template <typename OutputType>
OutputType _bf16bf16bf16_grouped(at::TensorList X, at::TensorList W) {
  at::Tensor Y;
  int total_M = 0;
  int G = X.size();

  // Allocate output tensor.
  std::vector<int64_t> output_sizes;
  int64_t total_output_size = 0;
  for (int i = 0; i < G; ++i) {
    int M = X[i].size(0);
    int N = W[i].size(0);
    total_M += M;
    const int64_t output_size = M * N;
    total_output_size += output_size;
    output_sizes.push_back(output_size);
  }
  Y = at::empty(total_output_size, X[0].options().dtype(at::kBFloat16));

  // Run kernel.
  at::Tensor g_out =
      dispatch_bf16_grouped_kernel<at::TensorList>(total_M, X, W, Y);

  // Return appropriate output type.
  if constexpr (std::is_same_v<OutputType, at::Tensor>) {
    int N = W[0].size(0);
    return g_out.view({total_M, N});
  } else {
    // Return grouped view of output.
    std::vector<at::Tensor> output_group = g_out.split(output_sizes);
    for (int i = 0; i < G; ++i) {
      output_group[i] = output_group[i].view({X[i].size(0), W[i].size(0)});
    }
    return output_group;
  }
}

std::vector<at::Tensor> bf16bf16bf16_grouped(
    at::TensorList X,
    at::TensorList W) {
  return _bf16bf16bf16_grouped<std::vector<at::Tensor>>(X, W);
}

at::Tensor bf16bf16bf16_grouped_cat(at::TensorList X, at::TensorList W) {
  return _bf16bf16bf16_grouped<at::Tensor>(X, W);
}

at::Tensor
bf16bf16bf16_grouped_stacked(at::Tensor X, at::Tensor W, at::Tensor M_sizes) {
  int total_M = X.size(0);
  int N = W.size(1);
  int G = M_sizes.size(0);
  TORCH_CHECK(
      M_sizes.device() == X.device(),
      "M_sizes must be on same device as inputs.");
  TORCH_CHECK(
      W.dim() == 3 && W.size(0) == G, "Weights should be shape [G, N, K].")
  at::Tensor Y = at::empty(total_M * N, X.options().dtype(at::kBFloat16));
  // Early exit for empty inputs.
  if (total_M == 0) {
    return Y.view({total_M, N});
  }
  // Return continuous view of output.
  at::Tensor out = dispatch_bf16_grouped_kernel<at::Tensor>(
      total_M, X, W, Y, std::nullopt, M_sizes);
  return out.view({total_M, N});
}

at::Tensor bf16bf16bf16_grouped_dynamic(
    at::Tensor X,
    at::Tensor W,
    at::Tensor zero_start_index_M) {
  TORCH_CHECK(
      zero_start_index_M.device() == X.device(),
      "zero_start_index_M must be on same device as inputs.");
  int G = X.size(0);
  int M = X.size(1);
  int N = W.size(1);
  int K = X.size(0);
  int total_output_size = G * M * N;
  at::Tensor Y;
  Y = at::zeros(total_output_size, X.options().dtype(at::kBFloat16));

  // Return continuous view of output.
  at::Tensor output = dispatch_bf16_grouped_kernel<at::Tensor>(
      G * M, X, W, Y, zero_start_index_M);
  // View as proper shape.
  return output.view({G, M, N});
}

#else

std::vector<at::Tensor> bf16bf16bf16_grouped(
    at::TensorList X,
    at::TensorList W) {
  throw std::runtime_error(
      "CUDA version is older than 12.0"); // requires CUDA>=12
}

at::Tensor bf16bf16bf16_grouped_cat(at::TensorList X, at::TensorList W) {
  throw std::runtime_error(
      "CUDA version is older than 12.0"); // requires CUDA>=12
}

at::Tensor bf16bf16bf16_grouped_dynamic(
    at::Tensor X,
    at::Tensor W,
    at::Tensor zero_start_index_M) {
  throw std::runtime_error(
      "CUDA version is older than 12.0"); // requires CUDA>=12
}

at::Tensor bf16bf16bf16_grouped_stacked(at::Tensor, at::Tensor, at::Tensor) {
  throw std::runtime_error(
      "CUDA version is older than 12.0"); // requires CUDA>=12
}

#endif

} // namespace fbgemm_gpu
