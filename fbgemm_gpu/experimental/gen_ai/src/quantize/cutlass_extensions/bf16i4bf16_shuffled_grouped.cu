/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

#include "cutlass/util/mixed_dtype_utils.hpp"
#include "cutlass/util/packed_stride.hpp"

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
    typename ElementOutput,
    typename ElementScale,
    typename ElementZero,
    typename StrideA,
    typename StrideB,
    typename StrideC,
    typename StrideS,
    typename LayoutAtomQuant>
__global__ void set_kernel_args(
    int G,
    int N,
    int K,
    int num_scale_groups,
    int32_t* M_sizes,
    ProblemShape* problem_shape_ptr,
    ElementA* xq,
    const ElementA** xq_ptr,
    ElementB* wq,
    const ElementB** wq_ptr,
    ElementScale* w_scale_group,
    const ElementScale** w_scale_group_ptr,
    ElementZero* w_zero_group,
    const ElementZero** w_zero_group_ptr,
    ElementOutput* output,
    ElementOutput** output_ptr,
    StrideA* stride_a_ptr,
    StrideB* stride_b_ptr,
    StrideC* stride_c_ptr,
    StrideS* stride_s_ptr) {
  // Get the group corresponding to this thread.
  auto group_index = blockIdx.x * blockDim.x + threadIdx.x;
  // If this is a valid group, write kernel args to device.
  if (group_index < G) {
    // Since we are only writing a subset of the groups to kernel args,
    // we need to start by initializing a counter and setting other groups
    // to empty problems.
    __shared__ int non_zero_counter;
    // Initialize counter and problem memory for this group.
    if (group_index == 0) {
      non_zero_counter = 0;
    }
    // We set the problem shapes to empty by default to skip over
    // these groups.
    problem_shape_ptr[group_index] = ProblemShape(0, 0, 0);
    // Sync threads to make sure state is shared across the block.
    __syncthreads();

    // Now check if this is a non-zero group.
    int M = M_sizes[group_index];
    // Only proceed if so.
    if (M > 0) {
      // Get the non-zero index for this group atomically.
      int non_zero_idx = atomicAdd(&non_zero_counter, 1);
      // Compute offset into tensor where this group begins.
      int offset_M = 0;
      // Compute cumulative sum of prior groups to find offset.
      for (int i = 0; i < group_index; i++) {
        offset_M += M_sizes[i];
      }
      // Set the problem shape for this group.
      problem_shape_ptr[non_zero_idx] = ProblemShape(N, M, K);
      // Set pointer to xq.
      xq_ptr[non_zero_idx] = xq + (offset_M * K);
      // Set pointer to wq, dividing by two as wq is packed into bytes.
      wq_ptr[non_zero_idx] = wq + (group_index * N * K / 2);
      // Set scale pointers.
      w_scale_group_ptr[non_zero_idx] =
          w_scale_group + (group_index * N * num_scale_groups);
      w_zero_group_ptr[non_zero_idx] =
          w_zero_group + (group_index * N * num_scale_groups);
      // Set output pointer.
      output_ptr[non_zero_idx] = output + (offset_M * N);
      // Set stride pointers.
      stride_a_ptr[non_zero_idx] = cutlass::make_cute_packed_stride(
          StrideA{}, cute::make_shape(M, K, 1));
      stride_b_ptr[non_zero_idx] =
          cute::tile_to_shape(LayoutAtomQuant{}, cute::make_shape(N, K, 1));
      stride_c_ptr[non_zero_idx] = cutlass::make_cute_packed_stride(
          StrideC{}, cute::make_shape(N, M, 1));
      stride_s_ptr[non_zero_idx] = cutlass::make_cute_packed_stride(
          StrideS{}, cute::make_shape(N, num_scale_groups, 1));
    }
  }
}

template <
    typename SCALE_TYPE,
    int TB_M,
    int TB_N,
    int TBS_M,
    int TBS_N,
    int TBS_K>
void _bf16i4bf16_shuffled_grouped(
    at::Tensor X,
    at::Tensor WQ,
    at::Tensor w_scale_group,
    at::Tensor w_zero_group,
    at::Tensor M_sizes,
    at::Tensor Y) {
  // Get basic shape information.
  int G = M_sizes.size(0);
  // X is shape [total_M, K]
  int total_M = X.size(0);
  int kernel_groups = std::min(G, total_M);
  int K = X.size(-1);
  // WQ is shape [G, N, K/2]
  int N = WQ.size(1);
  // Group scales should have shape [G, num_scale_groups, 8, N]
  int num_scale_groups = w_scale_group.size(1);
  int group_size = K / num_scale_groups;
  // Define cutlass types.
  using ProblemShape = cutlass::gemm::GroupProblemShape<
      cute::Shape<int, int, int>>; // <M,N,K> per group.
  using MmaType = cutlass::bfloat16_t;
  using QuantType = cutlass::int4b_t;
  using ElementAccumulator = float;
  // K Tile size is fixed for preshuffled Gemm.
  constexpr int TileShapeK = 128 * 8 / cute::sizeof_bits<MmaType>::value;

  // A matrix configuration.
  using ElementA = MmaType;
  using LayoutA = cutlass::layout::RowMajor;
  constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

  // B matrix configuration.
  using ElementB = QuantType;
  using LayoutB = cutlass::layout::ColumnMajor;
  constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

  // Explicitly swap and transdpose inputs as thinner dtype needs to be first
  // gemm arg.
  using LayoutA_Transpose =
      typename cutlass::layout::LayoutTranspose<LayoutA>::type;
  using LayoutB_Transpose =
      typename cutlass::layout::LayoutTranspose<LayoutB>::type;

  // Need to pass a pointer type to make the 3rd dimension of Stride be _0
  using StrideA =
      cute::remove_pointer_t<cutlass::detail::TagToStrideA_t<LayoutA*>>;
  using StrideB =
      cute::remove_pointer_t<cutlass::detail::TagToStrideB_t<LayoutB*>>;

  // Define layout for shuffled weight tensor.
  using ValueShuffle = cute::Layout<
      cute::Shape<cute::_2, cute::_4>,
      cute::Stride<cute::_4, cute::_1>>; // order [0,2,4,6,1,3,5,7]
  int constexpr NumShuffleAtoms = 1;
  using MmaAtomShape =
      cute::Layout<cute::Shape<cute::_1, cute::Int<NumShuffleAtoms>>>;
  using LayoutAtomQuant = decltype(cutlass::compute_memory_reordering_atom<
                                   MmaType,
                                   MmaAtomShape,
                                   ValueShuffle>());
  using LayoutB_Reordered = decltype(cute::tile_to_shape(
      LayoutAtomQuant{}, cute::Layout<cute::Shape<int, int, int>, StrideB>{}));

  using ElementScale = SCALE_TYPE;
  using ElementZero = ElementScale;

  // Output Matrix configuration.
  using ElementC = cutlass::bfloat16_t;
  using LayoutC = cutlass::layout::RowMajor;
  constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

  // Core kernel configurations
  using ArchTag = cutlass::arch::Sm90;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using TileShape =
      cute::Shape<cute::Int<TB_M>, cute::Int<TB_N>, cute::Int<TileShapeK>>;
  using ClusterShape =
      cute::Shape<cute::Int<TBS_M>, cute::Int<TBS_N>, cute::Int<TBS_K>>;
  using KernelSchedule =
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative;
  using EpilogueSchedule =
      cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
  using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm90,
          cutlass::arch::OpClassTensorOp,
          TileShape,
          ClusterShape,
          EpilogueTileType,
          ElementAccumulator,
          ElementAccumulator,
          void, // Indicate there is no beta scaling.
          typename cutlass::layout::LayoutTranspose<LayoutC>::type*,
          AlignmentC,
          ElementC,
          typename cutlass::layout::LayoutTranspose<LayoutC>::type*,
          AlignmentC,
          EpilogueSchedule>::CollectiveOp;

  using CollectiveMainloopShuffled =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag,
          OperatorClass,
          cute::tuple<ElementB, ElementScale>,
          LayoutB_Reordered*,
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

  using GemmKernelShuffled = cutlass::gemm::kernel::GemmUniversal<
      ProblemShape,
      CollectiveMainloopShuffled,
      CollectiveEpilogue>;

  using GemmShuffled =
      cutlass::gemm::device::GemmUniversalAdapter<GemmKernelShuffled>;

  using StrideC = typename GemmKernelShuffled::InternalStrideC;
  using StrideS = typename CollectiveMainloopShuffled::StrideScale;

  // Determine how much device memory is needed and create pointers.
  // Each buffer is aligned with 16 bytes.
  // Start with space needed for the problem size.
  const int64_t problem_size_offset = 0;
  int64_t problem_size_buffer =
      _byte_align(G * sizeof(ProblemShape::UnderlyingProblemShape));

  // Next create space for X pointers.
  const int64_t xq_offset = problem_size_offset + problem_size_buffer;
  int64_t xq_size_buffer = _byte_align(G * sizeof(ElementA**));

  // WQ Pointers.
  const int64_t wq_offset = xq_offset + xq_size_buffer;
  int64_t wq_size_buffer = _byte_align(G * sizeof(ElementB**));

  // W group scales.
  const int64_t w_scale_group_offset = wq_offset + wq_size_buffer;
  int64_t w_scale_group_buffer = _byte_align(G * sizeof(ElementScale**));

  // W group zeros.
  const int64_t w_zero_group_offset =
      w_scale_group_offset + w_scale_group_buffer;
  int64_t w_zero_group_buffer = _byte_align(G * sizeof(ElementZero**));

  // Outputs.
  const int64_t output_offset = w_zero_group_offset + w_zero_group_buffer;
  int64_t output_buffer = _byte_align(G * sizeof(ElementC**));

  // A stride.
  const int64_t stride_a_offset = output_offset + output_buffer;
  int64_t stride_a_buffer = _byte_align(G * sizeof(StrideA));

  // B stride;
  const int64_t stride_b_offset = stride_a_offset + stride_a_buffer;
  int64_t stride_b_buffer = _byte_align(G * sizeof(LayoutB_Reordered));

  // C stride;
  const int64_t stride_c_offset = stride_b_offset + stride_b_buffer;
  int64_t stride_c_buffer = _byte_align(G * sizeof(StrideC));

  // Scale stride;
  const int64_t stride_s_offset = stride_c_offset + stride_c_buffer;
  int64_t stride_s_buffer = _byte_align(G * sizeof(StrideS));

  // Compute total buffer size
  int64_t total_buffer_size = stride_s_offset + stride_s_buffer;

  // Allocate space for gemm information.
  at::Tensor kernel_args =
      at::empty({total_buffer_size}, X.options().dtype(at::kByte));

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
  const ElementScale** w_scale_group_ptr =
      reinterpret_cast<const ElementScale**>(
          kernel_args_ptr + w_scale_group_offset);
  const ElementZero** w_zero_group_ptr = reinterpret_cast<const ElementZero**>(
      kernel_args_ptr + w_zero_group_offset);
  ElementC** output_ptr =
      reinterpret_cast<ElementC**>(kernel_args_ptr + output_offset);
  StrideA* stride_a_ptr =
      reinterpret_cast<StrideA*>(kernel_args_ptr + stride_a_offset);
  LayoutB_Reordered* stride_b_ptr =
      reinterpret_cast<LayoutB_Reordered*>(kernel_args_ptr + stride_b_offset);
  StrideC* stride_c_ptr =
      reinterpret_cast<StrideC*>(kernel_args_ptr + stride_c_offset);
  StrideS* stride_s_ptr =
      reinterpret_cast<StrideS*>(kernel_args_ptr + stride_s_offset);

  // Invoke kernel to set device memory specifying grouped gemm configuration.
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  set_kernel_args<
      ProblemShape::UnderlyingProblemShape,
      ElementA,
      ElementB,
      ElementC,
      ElementScale,
      ElementZero,
      StrideA,
      LayoutB_Reordered,
      StrideC,
      StrideS,
      LayoutAtomQuant><<<1, G, 0, stream>>>(
      G,
      N,
      K,
      num_scale_groups,
      reinterpret_cast<int32_t*>(M_sizes.data_ptr()),
      problem_shape_ptr,
      reinterpret_cast<ElementA*>(X.data_ptr()),
      xq_ptr,
      reinterpret_cast<ElementB*>(WQ.data_ptr()),
      wq_ptr,
      reinterpret_cast<ElementScale*>(w_scale_group.data_ptr()),
      w_scale_group_ptr,
      reinterpret_cast<ElementZero*>(w_zero_group.data_ptr()),
      w_zero_group_ptr,
      reinterpret_cast<ElementC*>(Y.data_ptr()),
      output_ptr,
      stride_a_ptr,
      stride_b_ptr,
      stride_c_ptr,
      stride_s_ptr);

  // Define GEMM arguments.
  typename GemmShuffled::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {kernel_groups, problem_shape_ptr, nullptr},
      {wq_ptr,
       stride_b_ptr,
       xq_ptr,
       stride_a_ptr,
       w_scale_group_ptr,
       stride_s_ptr,
       group_size},
      {{}, nullptr, stride_c_ptr, output_ptr, stride_c_ptr}};

  // Launch the workload.
  GemmShuffled gemm;
  int workspace_size = GemmShuffled::get_workspace_size(arguments);

  // Allocate empty workspace memory.
  at::Tensor workspace =
      at::empty(workspace_size, X.options().dtype(at::kByte));

  // Check the problem size is supported or not
  cutlass::Status status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot implement");
  }

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm.initialize(arguments, workspace.data_ptr());
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
}

template <typename SCALE_TYPE>
at::Tensor bf16i4bf16_shuffled_grouped_dispatch(
    at::Tensor X,
    at::Tensor WQ,
    at::Tensor w_scale_group,
    at::Tensor w_zero_group,
    at::Tensor M_sizes) {
  // X should be shape [total_M, K], W should be shape [G, N, K/2]
  int total_M = X.size(0);
  int K = X.size(1);
  int N = WQ.size(1);
  int group_count = M_sizes.size(0);
  TORCH_CHECK(
      M_sizes.device() == X.device() && M_sizes.dtype() == at::kInt,
      "M_sizes must be int32 and on the same device as inputs.");
  TORCH_CHECK(
      WQ.dim() == 3 && WQ.size(0) == group_count && WQ.size(2) == K / 2,
      "Weights should be shape [G, N, K / 2]");
  // Allocate output.
  at::Tensor Y = at::empty({total_M, N}, X.options().dtype(at::kBFloat16));
  // Handle empty input by skipping kernel launch.
  if (total_M > 0) {
    // Use heuristics to pick best kernel implementation.
    if (total_M <= 16) {
      _bf16i4bf16_shuffled_grouped<SCALE_TYPE, 128, 16, 1, 1, 1>(
          X, WQ, w_scale_group, w_zero_group, M_sizes, Y);
    } else if (total_M <= 32) {
      _bf16i4bf16_shuffled_grouped<SCALE_TYPE, 128, 32, 1, 1, 1>(
          X, WQ, w_scale_group, w_zero_group, M_sizes, Y);
    } else if (total_M <= 64) {
      _bf16i4bf16_shuffled_grouped<SCALE_TYPE, 128, 64, 1, 1, 1>(
          X, WQ, w_scale_group, w_zero_group, M_sizes, Y);
    } else if (total_M <= 128) {
      _bf16i4bf16_shuffled_grouped<SCALE_TYPE, 128, 128, 1, 1, 1>(
          X, WQ, w_scale_group, w_zero_group, M_sizes, Y);
    } else if (total_M <= 512) {
      _bf16i4bf16_shuffled_grouped<SCALE_TYPE, 256, 128, 2, 1, 1>(
          X, WQ, w_scale_group, w_zero_group, M_sizes, Y);
    } else {
      _bf16i4bf16_shuffled_grouped<SCALE_TYPE, 128, 256, 2, 1, 1>(
          X, WQ, w_scale_group, w_zero_group, M_sizes, Y);
    }
  }
  return Y;
}

at::Tensor bf16i4bf16_shuffled_grouped(
    at::Tensor X,
    at::Tensor WQ,
    at::Tensor w_scale_group,
    at::Tensor w_zero_group,
    at::Tensor M_sizes) {
  if (w_scale_group.dtype() == at::kBFloat16) {
    return bf16i4bf16_shuffled_grouped_dispatch<cutlass::bfloat16_t>(
        X, WQ, w_scale_group, w_zero_group, M_sizes);
  } else if (w_scale_group.dtype() == at::kFloat) {
    return bf16i4bf16_shuffled_grouped_dispatch<float>(
        X, WQ, w_scale_group, w_zero_group, M_sizes);
  } else {
    TORCH_CHECK(
        false, "Only bf16 and fp32 scales and zeros currently supported.");
  }
}

#else

at::Tensor bf16i4bf16_shuffled_grouped(
    at::Tensor X,
    at::Tensor WQ,
    at::Tensor w_scale_group,
    at::Tensor w_zero_group,
    at::Tensor M_sizes) {
  throw std::runtime_error(
      "CUDA version is older than 12.0"); // requires CUDA>=12
}

#endif

} // namespace fbgemm_gpu
