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

#include "cutlass_extensions/include/kernel_mode.h"

namespace fbgemm_gpu {

#if CUDART_VERSION >= 12000

template <int TB_M, int TB_N, int TBS_M, int TBS_N, int TBS_K, bool COOP>
at::Tensor _f8i4bf16_shuffled(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale) {
  // Get shape information from input tensors.
  int M = XQ.size(0);
  int K = XQ.size(1);
  int N = WQ.size(0);
  // Make sure w_scale is in proper format.
  TORCH_CHECK(
      w_scale.size(1) == 8,
      "Weights and scales must be prepacked with preshuffle_i4.");
  int num_groups = w_scale.size(0);
  int group_size = K / num_groups;
  // Allocate output.
  at::Tensor Y = at::empty({M, N}, XQ.options().dtype(at::kBFloat16));

  // Define input types.
  using MmaType = cutlass::float_e4m3_t;
  using QuantType = cutlass::int4b_t;
  constexpr int TileShapeK = 128 * 8 / cute::sizeof_bits<MmaType>::value;

  // A Matrix configuration.
  using ElementA = MmaType;
  using LayoutA = cutlass::layout::RowMajor;
  constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

  // B Matrix Configuration.
  using ElementB = QuantType;
  using LayoutB = cutlass::layout::ColumnMajor;
  constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

  // We need to manually swap and transpose inputs. Unclear how required this is
  // though.
  using LayoutA_Transpose =
      typename cutlass::layout::LayoutTranspose<LayoutA>::type;
  using LayoutB_Transpose =
      typename cutlass::layout::LayoutTranspose<LayoutB>::type;

  using StrideA = cutlass::detail::TagToStrideA_t<LayoutA>;
  using StrideB = cutlass::detail::TagToStrideB_t<LayoutB>;

  // Define layout for shuffled weight tensor.
  using LayoutAtomQuant =
      decltype(cutlass::compute_memory_reordering_atom<MmaType>());
  using LayoutB_Reordered = decltype(cute::tile_to_shape(
      LayoutAtomQuant{}, cute::Layout<cute::Shape<int, int, int>, StrideB>{}));

  using ElementScale = MmaType;

  // Output Matrix configuration.
  using ElementC = cutlass::bfloat16_t;
  using LayoutC = cutlass::layout::RowMajor;
  constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

  // Core kernel configurations
  using ElementAccumulator = float;
  using ElementCompute = float;
  using ArchTag = cutlass::arch::Sm90;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  // TODO tune these shapes.
  using TileShape =
      cute::Shape<cute::Int<TB_M>, cute::Int<TB_N>, cute::Int<TileShapeK>>;
  using ClusterShape =
      cute::Shape<cute::Int<TBS_M>, cute::Int<TBS_N>, cute::Int<TBS_K>>;
  // TODO Should we use fast accum here?
  using KernelSchedule = cute::conditional_t<
      COOP,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative,
      cutlass::gemm::KernelTmaWarpSpecialized>;
  // Might be the only epilogue schedule that supports swap + transpose.
  using EpilogueSchedule = cute::conditional_t<
      COOP,
      cutlass::epilogue::TmaWarpSpecializedCooperative,
      cutlass::epilogue::TmaWarpSpecialized>;
  using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;

  // Define EVT for rowwise scaling.
  using XScale = cutlass::epilogue::fusion::Sm90RowBroadcast<
      0,
      TileShape,
      ElementAccumulator,
      ElementAccumulator,
      cute::Stride<cute::Int<0>, cute::Int<1>, cute::Int<0>>>;

  using Accum = cutlass::epilogue::fusion::Sm90AccFetch;

  using Compute0 = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies,
      ElementC, // First stage output type.
      ElementAccumulator, // First stage input types.
      cutlass::FloatRoundStyle::round_to_nearest>;

  using EpilogueEVT =
      cutlass::epilogue::fusion::Sm90EVT<Compute0, XScale, Accum>;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm90,
          cutlass::arch::OpClassTensorOp,
          TileShape,
          ClusterShape,
          EpilogueTileType,
          ElementAccumulator,
          ElementAccumulator,
          ElementC,
          typename cutlass::layout::LayoutTranspose<LayoutC>::type,
          AlignmentC,
          ElementC,
          typename cutlass::layout::LayoutTranspose<LayoutC>::type,
          AlignmentC,
          EpilogueSchedule,
          EpilogueEVT>::CollectiveOp;

  using CollectiveMainloopShuffled =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag,
          OperatorClass,
          cute::tuple<ElementB, cutlass::Array<ElementScale, 8>>,
          LayoutB_Reordered,
          AlignmentB,
          ElementA,
          LayoutA_Transpose,
          AlignmentA,
          ElementAccumulator,
          TileShape,
          ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          KernelSchedule>::CollectiveOp;

  using GemmKernelShuffled = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int, int>,
      CollectiveMainloopShuffled,
      CollectiveEpilogue>;

  using GemmShuffled =
      cutlass::gemm::device::GemmUniversalAdapter<GemmKernelShuffled>;

  using StrideC = typename GemmKernelShuffled::StrideC;

  /// Initialization
  auto shape_B = cute::make_shape(N, K, 1);
  StrideA stride_A =
      cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, shape_B);
  StrideC stride_C =
      cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(N, M, 1));
  LayoutB_Reordered layout_B_reordered =
      cute::tile_to_shape(LayoutAtomQuant{}, shape_B);
  using StrideS = typename CollectiveMainloopShuffled::StrideScale;
  StrideS stride_S = cutlass::make_cute_packed_stride(
      StrideS{}, cute::make_shape(N, num_groups, 1));

  // Define Gemm arguments.
  typename GemmShuffled::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {N, M, K, 1},
      {reinterpret_cast<ElementB*>(WQ.data_ptr()),
       layout_B_reordered,
       reinterpret_cast<ElementA*>(XQ.data_ptr()),
       stride_A,
       reinterpret_cast<cutlass::Array<ElementScale, 8>*>(w_scale.data_ptr()),
       stride_S,
       group_size},
      {{},
       reinterpret_cast<ElementC*>(Y.data_ptr()),
       stride_C,
       reinterpret_cast<ElementC*>(Y.data_ptr()),
       stride_C}};

  arguments.epilogue.thread = {
      {reinterpret_cast<ElementAccumulator*>(x_scale.data_ptr())}, // x_scale
      {}, // Accumulator
      {}, // Multiplies
  };

  // Launch the workload.
  GemmShuffled gemm;

  // Using the arguments, query for extra workspace required for matrix
  // multiplication computation
  size_t workspace_size = GemmShuffled::get_workspace_size(arguments);

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

  return Y;
}

at::Tensor f8i4bf16_shuffled(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale) {
  int M = XQ.size(0);
  int K = XQ.size(1);
  int N = WQ.size(0);
  // Use shape heuristics to dispatch to optimized kernel configuration.
  if (M <= 16) {
    return _f8i4bf16_shuffled<64, 16, 2, 1, 1, false>(XQ, WQ, x_scale, w_scale);
  } else if (M <= 32) {
    return _f8i4bf16_shuffled<64, 32, 2, 1, 1, false>(XQ, WQ, x_scale, w_scale);
  } else if (M <= 64) {
    return _f8i4bf16_shuffled<64, 64, 2, 1, 1, false>(XQ, WQ, x_scale, w_scale);
  } else if (M <= 128) {
    return _f8i4bf16_shuffled<64, 128, 2, 1, 1, false>(
        XQ, WQ, x_scale, w_scale);
  } else if (M <= 256) {
    if (N <= 4096) {
      return _f8i4bf16_shuffled<64, 128, 2, 1, 1, false>(
          XQ, WQ, x_scale, w_scale);
    } else {
      return _f8i4bf16_shuffled<64, 256, 1, 1, 1, false>(
          XQ, WQ, x_scale, w_scale);
    }
  } else if (M <= 512) {
    if (N <= 4096) {
      return _f8i4bf16_shuffled<64, 256, 2, 1, 1, false>(
          XQ, WQ, x_scale, w_scale);
    } else {
      return _f8i4bf16_shuffled<128, 256, 2, 1, 1, true>(
          XQ, WQ, x_scale, w_scale);
    }
  } else if (M <= 1024) {
    if (N <= 1024) {
      return _f8i4bf16_shuffled<64, 128, 2, 1, 1, false>(
          XQ, WQ, x_scale, w_scale);
    } else if (N <= 2048) {
      return _f8i4bf16_shuffled<64, 256, 2, 1, 1, false>(
          XQ, WQ, x_scale, w_scale);
    } else {
      return _f8i4bf16_shuffled<128, 256, 2, 1, 1, true>(
          XQ, WQ, x_scale, w_scale);
    }
  } else {
    if (N <= 1024) {
      return _f8i4bf16_shuffled<64, 256, 2, 1, 1, false>(
          XQ, WQ, x_scale, w_scale);
    } else {
      return _f8i4bf16_shuffled<128, 256, 2, 1, 1, true>(
          XQ, WQ, x_scale, w_scale);
    }
  }
}

#else

at::Tensor f8i4bf16_shuffled(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale) {
  throw std::runtime_error(
      "CUDA version is older than 12.0"); // requires CUDA>=12
}

#endif

} // namespace fbgemm_gpu
