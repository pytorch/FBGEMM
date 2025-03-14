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
    typename ElementAccumulator,
    typename ElementPackedScale,
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
    int64_t* M_offsets,
    ProblemShape* problem_shape_ptr,
    ElementA* xq,
    const ElementA** xq_ptr,
    ElementB* wq,
    const ElementB** wq_ptr,
    ElementAccumulator* x_scale,
    const ElementAccumulator** x_scale_ptr,
    ElementAccumulator* w_scale,
    const ElementAccumulator** w_scale_ptr,
    ElementPackedScale* w_scale_group,
    const ElementPackedScale** w_scale_group_ptr,
    ElementOutput* output,
    ElementOutput** output_ptr,
    StrideA* stride_a_ptr,
    StrideB* stride_b_ptr,
    StrideC* stride_c_ptr,
    StrideS* stride_s_ptr) {
  // Get the group corresponding to this thread.
  int group_index = blockIdx.x * blockDim.x + threadIdx.x;
  // If this is a valid group, write kernel args to device.
  if (group_index < G) {
    // First get the M value for this group.
    int offset_M;
    int M;
    if (group_index == 0) {
      offset_M = 0;
      M = M_offsets[group_index];
    } else {
      offset_M = M_offsets[group_index - 1];
      M = M_offsets[group_index] - offset_M;
    }
    // Set the problem shape for this group.
    problem_shape_ptr[group_index] = ProblemShape(N, M, K);
    // Set pointer to xq.
    xq_ptr[group_index] = xq + (offset_M * K);
    // Set pointer to wq, dividing by two as wq is packed into bytes.
    wq_ptr[group_index] = wq + (group_index * N * K / 2);
    // Set scale pointers.
    x_scale_ptr[group_index] = x_scale + offset_M;
    w_scale_ptr[group_index] = w_scale + (group_index * N);
    w_scale_group_ptr[group_index] =
        w_scale_group + (group_index * N * num_scale_groups);
    // Set output pointer.
    output_ptr[group_index] = output + (offset_M * N);
    // Set stride pointers.
    stride_a_ptr[group_index] =
        cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    stride_b_ptr[group_index] = cute::tile_to_shape(
        LayoutAtomQuant{}, cute::make_shape(N, K, cute::Int<1>{}));
    stride_c_ptr[group_index] =
        cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(N, M, 1));
    stride_s_ptr[group_index] = cutlass::make_cute_packed_stride(
        StrideS{}, cute::make_shape(N, num_scale_groups, 1));
  }
}

template <int TB_M, int TB_N, int TBS_M, int TBS_N, int TBS_K>
void _f8i4bf16_shuffled_grouped(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor w_scale_group,
    at::Tensor M_offsets,
    at::Tensor Y) {
  // Get basic shape information.
  int G = M_offsets.size(0);
  // XQ is shape [total_M, K]
  int K = XQ.size(-1);
  // WQ is shape [G, N, K/2]
  int N = WQ.size(1);
  // Group scales should have shape [G, num_scale_groups, 8, N]
  int num_scale_groups = w_scale_group.size(1);
  int group_size = K / num_scale_groups;
  TORCH_CHECK(
      num_scale_groups == 1,
      "Mixed dtype grouped gemm only supports rowwise scaling currently (group_size=K).");

  // Define cutlass types.
  using ProblemShape = cutlass::gemm::GroupProblemShape<
      cute::Shape<int, int, int>>; // <M,N,K> per group.
  using MmaType = cutlass::float_e4m3_t;
  using QuantType = cutlass::int4b_t;
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
  using LayoutAtomQuant =
      decltype(cutlass::compute_memory_reordering_atom<MmaType>());
  using LayoutB_Reordered = decltype(cute::tile_to_shape(
      LayoutAtomQuant{},
      cute::Layout<cute::Shape<int, int, cute::Int<1>>, StrideB>{}));

  using ElementScale = MmaType;
  using ElementPackedScale = cutlass::Array<ElementScale, 8>;

  // Output Matrix configuration.
  using ElementC = cutlass::bfloat16_t;
  using LayoutC = cutlass::layout::RowMajor;
  constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

  // Core kernel configurations
  using ElementAccumulator = float;
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

  // Define EVT for rowwise scaling.
  using XScale = cutlass::epilogue::fusion::Sm90ColBroadcast<
      0,
      TileShape,
      ElementAccumulator*,
      ElementAccumulator,
      cute::Stride<cute::Int<1>, cute::Int<0>, cute::Int<0>>>;

  using WScale = cutlass::epilogue::fusion::Sm90RowBroadcast<
      0,
      TileShape,
      ElementAccumulator*,
      ElementAccumulator,
      cute::Stride<cute::Int<0>, cute::Int<1>, cute::Int<0>>>;

  using Accum = cutlass::epilogue::fusion::Sm90AccFetch;

  using Compute0 = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies,
      ElementAccumulator, // First stage output type.
      ElementAccumulator, // First stage input types.
      cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTCompute0 =
      cutlass::epilogue::fusion::Sm90EVT<Compute0, WScale, Accum>;

  using Compute1 = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies,
      ElementC,
      ElementAccumulator, // Second stage input types.
      cutlass::FloatRoundStyle::round_to_nearest>;

  using EpilogueEVT =
      cutlass::epilogue::fusion::Sm90EVT<Compute1, XScale, EVTCompute0>;

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
          EpilogueSchedule,
          EpilogueEVT>::CollectiveOp;

  using CollectiveMainloopShuffled =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag,
          OperatorClass,
          cute::tuple<ElementB, ElementPackedScale>,
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

  // Next create space for XQ pointers.
  const int64_t xq_offset = problem_size_offset + problem_size_buffer;
  int64_t xq_size_buffer = _byte_align(G * sizeof(ElementA**));

  // WQ Pointers.
  const int64_t wq_offset = xq_offset + xq_size_buffer;
  int64_t wq_size_buffer = _byte_align(G * sizeof(ElementB**));

  // X row scales.
  const int64_t x_scale_offset = wq_offset + wq_size_buffer;
  int64_t x_scale_buffer = _byte_align(G * sizeof(ElementAccumulator**));

  // W row scales.
  const int64_t w_scale_offset = x_scale_offset + x_scale_buffer;
  int64_t w_scale_buffer = _byte_align(G * sizeof(ElementAccumulator**));

  // W group scales.
  const int64_t w_scale_group_offset = w_scale_offset + w_scale_buffer;
  int64_t w_scale_group_buffer = _byte_align(G * sizeof(ElementPackedScale**));

  // Outputs.
  const int64_t output_offset = w_scale_group_offset + w_scale_group_buffer;
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
      at::empty({total_buffer_size}, XQ.options().dtype(at::kByte));

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
  const ElementAccumulator** x_scale_ptr =
      reinterpret_cast<const ElementAccumulator**>(
          kernel_args_ptr + x_scale_offset);
  const ElementAccumulator** w_scale_ptr =
      reinterpret_cast<const ElementAccumulator**>(
          kernel_args_ptr + w_scale_offset);
  const ElementPackedScale** w_scale_group_ptr =
      reinterpret_cast<const ElementPackedScale**>(
          kernel_args_ptr + w_scale_group_offset);
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
      ElementAccumulator,
      ElementPackedScale,
      StrideA,
      LayoutB_Reordered,
      StrideC,
      StrideS,
      LayoutAtomQuant><<<1, G, 0, stream>>>(
      G,
      N,
      K,
      num_scale_groups,
      reinterpret_cast<int64_t*>(M_offsets.data_ptr()),
      problem_shape_ptr,
      reinterpret_cast<ElementA*>(XQ.data_ptr()),
      xq_ptr,
      reinterpret_cast<ElementB*>(WQ.data_ptr()),
      wq_ptr,
      reinterpret_cast<ElementAccumulator*>(x_scale.data_ptr()),
      x_scale_ptr,
      reinterpret_cast<ElementAccumulator*>(w_scale.data_ptr()),
      w_scale_ptr,
      reinterpret_cast<ElementPackedScale*>(w_scale_group.data_ptr()),
      w_scale_group_ptr,
      reinterpret_cast<ElementC*>(Y.data_ptr()),
      output_ptr,
      stride_a_ptr,
      stride_b_ptr,
      stride_c_ptr,
      stride_s_ptr);

  // Define GEMM arguments.
  typename GemmShuffled::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {G, problem_shape_ptr, nullptr},
      {wq_ptr,
       stride_b_ptr,
       xq_ptr,
       stride_a_ptr,
       w_scale_group_ptr,
       stride_s_ptr,
       group_size},
      {{}, nullptr, stride_c_ptr, output_ptr, stride_c_ptr}};

  arguments.epilogue.thread = {
      {w_scale_ptr}, // w_scale
      // compute_0
      {
          {x_scale_ptr}, // x_scale
          {}, // Accumulator
          {} // Multiplies
      },
      {}, // Multiplies
  };

  // Launch the workload.
  GemmShuffled gemm;
  int workspace_size = GemmShuffled::get_workspace_size(arguments);

  // Allocate empty workspace memory.
  at::Tensor workspace =
      at::empty(workspace_size, XQ.options().dtype(at::kByte));

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

// Define kernel type signature
using Kernel = std::function<void(
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor)>;

at::Tensor f8i4bf16_shuffled_grouped(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor w_scale_group,
    at::Tensor M_offsets) {
  // X should be shape [total_M, K], W should be shape [G, N, K/2]
  int total_M = XQ.size(0);
  int K = XQ.size(1);
  int N = WQ.size(1);
  int group_count = M_offsets.size(0);
  TORCH_CHECK(
      M_offsets.device() == XQ.device() && M_offsets.dtype() == at::kLong,
      "M_offsets must be int64 and on the same device as inputs.");
  TORCH_CHECK(
      WQ.dim() == 3 && WQ.size(0) == group_count && WQ.size(2) == K / 2,
      "Weights should be shape [G, N, K / 2]");
  // Allocate output.
  at::Tensor Y = at::empty({total_M, N}, XQ.options().dtype(at::kBFloat16));
  // Handle empty input by skipping kernel launch.
  if (total_M > 0) {
    // Use heuristics to pick best kernel implementation.
    if (total_M <= 16) {
      _f8i4bf16_shuffled_grouped<128, 16, 1, 1, 1>(
          XQ, WQ, x_scale, w_scale, w_scale_group, M_offsets, Y);
    } else if (total_M <= 32) {
      _f8i4bf16_shuffled_grouped<128, 32, 1, 1, 1>(
          XQ, WQ, x_scale, w_scale, w_scale_group, M_offsets, Y);
    } else if (total_M <= 64) {
      _f8i4bf16_shuffled_grouped<128, 64, 1, 1, 1>(
          XQ, WQ, x_scale, w_scale, w_scale_group, M_offsets, Y);
    } else if (total_M <= 128) {
      _f8i4bf16_shuffled_grouped<128, 128, 1, 1, 1>(
          XQ, WQ, x_scale, w_scale, w_scale_group, M_offsets, Y);
    } else if (total_M <= 512) {
      _f8i4bf16_shuffled_grouped<256, 128, 2, 1, 1>(
          XQ, WQ, x_scale, w_scale, w_scale_group, M_offsets, Y);
    } else {
      _f8i4bf16_shuffled_grouped<128, 256, 2, 1, 1>(
          XQ, WQ, x_scale, w_scale, w_scale_group, M_offsets, Y);
    }
  }
  return Y;
}

#else

at::Tensor f8i4bf16_shuffled_grouped(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor w_scale_group,
    at::Tensor M_offsets) {
  throw std::runtime_error(
      "CUDA version is older than 12.0"); // requires CUDA>=12
}

#endif

} // namespace fbgemm_gpu
