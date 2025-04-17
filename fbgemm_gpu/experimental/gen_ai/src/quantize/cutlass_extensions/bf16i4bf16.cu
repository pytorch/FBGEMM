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

template <
    bool SHUFFLE,
    typename SCALE_TYPE,
    int TB_M,
    int TB_N,
    int TBS_M,
    int TBS_N,
    int TBS_K,
    bool COOP>
at::Tensor _bf16i4bf16(
    at::Tensor X,
    at::Tensor W,
    at::Tensor w_scale_group,
    at::Tensor w_zero_group,
    at::Tensor Y) {
  // Get shape information from input tensors.
  int M = size_to_dim_(X.dim() - 1, X.sizes());
  int K = X.size(-1);
  int N = size_to_dim_(W.dim() - 1, W.sizes());
  int num_groups = w_scale_group.size(0);
  TORCH_CHECK(
      w_zero_group.size(0) == num_groups,
      "Scales and zeros must be the same shape.");
  int group_size = K / num_groups;

  // Define input types.
  using MmaType = cutlass::bfloat16_t;
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

  using B_Layout =
      cute::conditional_t<SHUFFLE, LayoutB_Reordered, LayoutB_Transpose>;

  using ElementScale = SCALE_TYPE;
  using ElementZero = ElementScale;

  // Output Matrix configuration.
  using ElementC = cutlass::bfloat16_t;
  using LayoutC = cutlass::layout::RowMajor;
  constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

  // Core kernel configurations
  using ElementAccumulator = float;
  using ElementCompute = float;
  using ArchTag = cutlass::arch::Sm90;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using TileShape =
      cute::Shape<cute::Int<TB_M>, cute::Int<TB_N>, cute::Int<TileShapeK>>;
  using ClusterShape =
      cute::Shape<cute::Int<TBS_M>, cute::Int<TBS_N>, cute::Int<TBS_K>>;
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
          typename cutlass::layout::LayoutTranspose<LayoutC>::type,
          AlignmentC,
          ElementC,
          typename cutlass::layout::LayoutTranspose<LayoutC>::type,
          AlignmentC,
          EpilogueSchedule>::CollectiveOp;

  using CollectiveMainloopShuffled =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag,
          OperatorClass,
          cute::tuple<ElementB, ElementScale, ElementZero>,
          B_Layout,
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
  StrideB stride_B =
      cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
  StrideC stride_C =
      cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(N, M, 1));
  LayoutB_Reordered layout_B_reordered =
      cute::tile_to_shape(LayoutAtomQuant{}, shape_B);

  using stride_type = cute::conditional_t<SHUFFLE, LayoutB_Reordered, StrideB>;
  stride_type B_stride;
  if constexpr (SHUFFLE) {
    B_stride = layout_B_reordered;
  } else {
    B_stride = stride_B;
  }

  using StrideS = typename CollectiveMainloopShuffled::StrideScale;
  StrideS stride_S = cutlass::make_cute_packed_stride(
      StrideS{}, cute::make_shape(N, num_groups, 1));

  // Define Gemm arguments.
  typename GemmShuffled::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {N, M, K, 1},
      {reinterpret_cast<ElementB*>(W.data_ptr()),
       B_stride,
       reinterpret_cast<ElementA*>(X.data_ptr()),
       stride_A,
       reinterpret_cast<ElementScale*>(w_scale_group.data_ptr()),
       stride_S,
       group_size,
       reinterpret_cast<ElementZero*>(w_zero_group.data_ptr())},
      {{},
       nullptr,
       stride_C,
       reinterpret_cast<ElementC*>(Y.data_ptr()),
       stride_C}};

  // Launch the workload.
  GemmShuffled gemm;

  // Using the arguments, query for extra workspace required for matrix
  // multiplication computation
  int workspace_size = GemmShuffled::get_workspace_size(arguments);

  // Allocate workspace memory
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

  return Y;
}

template <bool SHUFFLE, typename SCALE_TYPE>
at::Tensor bf16i4bf16_dispatch(
    at::Tensor X,
    at::Tensor W,
    at::Tensor w_scale_group,
    at::Tensor w_zero_group) {
  int M = size_to_dim_(X.dim() - 1, X.sizes());
  int K = X.size(-1);
  int N = size_to_dim_(W.dim() - 1, W.sizes());
  // Check input types and shapes.
  TORCH_CHECK(
      X.is_cuda() && X.is_contiguous() && X.dtype() == at::kBFloat16,
      "X must be BF16 and contiguous on GPU.");
  TORCH_CHECK(
      W.size(-1) == K / 2 && W.is_cuda() && W.is_contiguous() &&
          W.dtype() == at::kChar,
      "W should be int8 (which represent two int4 values), have shape [..., N, K/2], "
      "and be contiguous on GPU.");
  // Make sure group scales and zeros are in proper format.
  TORCH_CHECK(
      w_scale_group.dim() == 2 && w_scale_group.size(1) == N,
      "Group scales are expected to have shape [num_groups, N].");

  // Allocate output or return an empty tensor if input is empty.
  if (M == 0 || N == 0 || K == 0) {
    return at::zeros({M, N}, X.options().dtype(at::kBFloat16));
  }
  at::Tensor Y = at::empty({M, N}, X.options().dtype(at::kBFloat16));

  // Use shape heuristics to dispatch to optimized kernel configuration.
  if (M <= 16) {
    return _bf16i4bf16<SHUFFLE, SCALE_TYPE, 64, 16, 1, 1, 1, false>(
        X, W, w_scale_group, w_zero_group, Y);
  } else if (M <= 32) {
    if (N <= 4096) {
      return _bf16i4bf16<SHUFFLE, SCALE_TYPE, 64, 16, 1, 1, 1, false>(
          X, W, w_scale_group, w_zero_group, Y);
    } else {
      return _bf16i4bf16<SHUFFLE, SCALE_TYPE, 64, 32, 2, 1, 1, false>(
          X, W, w_scale_group, w_zero_group, Y);
    }
  } else if (M <= 64) {
    if (N <= 2048) {
      return _bf16i4bf16<SHUFFLE, SCALE_TYPE, 64, 16, 1, 1, 1, false>(
          X, W, w_scale_group, w_zero_group, Y);
    } else if (N <= 4096) {
      return _bf16i4bf16<SHUFFLE, SCALE_TYPE, 64, 32, 2, 1, 1, false>(
          X, W, w_scale_group, w_zero_group, Y);
    } else {
      return _bf16i4bf16<SHUFFLE, SCALE_TYPE, 64, 64, 2, 1, 1, false>(
          X, W, w_scale_group, w_zero_group, Y);
    }
  } else if (M <= 128) {
    if (N <= 1024) {
      return _bf16i4bf16<SHUFFLE, SCALE_TYPE, 64, 16, 1, 1, 1, false>(
          X, W, w_scale_group, w_zero_group, Y);
    } else if (N <= 2048) {
      return _bf16i4bf16<SHUFFLE, SCALE_TYPE, 64, 32, 2, 1, 1, false>(
          X, W, w_scale_group, w_zero_group, Y);
    } else if (N <= 4096) {
      return _bf16i4bf16<SHUFFLE, SCALE_TYPE, 64, 64, 2, 1, 1, false>(
          X, W, w_scale_group, w_zero_group, Y);
    } else {
      return _bf16i4bf16<SHUFFLE, SCALE_TYPE, 64, 128, 1, 1, 1, false>(
          X, W, w_scale_group, w_zero_group, Y);
    }
  } else if (M <= 256) {
    if (N <= 1024) {
      return _bf16i4bf16<SHUFFLE, SCALE_TYPE, 64, 32, 2, 1, 1, false>(
          X, W, w_scale_group, w_zero_group, Y);
    } else if (N <= 2048) {
      return _bf16i4bf16<SHUFFLE, SCALE_TYPE, 64, 64, 2, 1, 1, false>(
          X, W, w_scale_group, w_zero_group, Y);
    } else if (N <= 4096) {
      return _bf16i4bf16<SHUFFLE, SCALE_TYPE, 64, 128, 1, 1, 1, false>(
          X, W, w_scale_group, w_zero_group, Y);
    } else {
      return _bf16i4bf16<SHUFFLE, SCALE_TYPE, 64, 256, 1, 1, 1, false>(
          X, W, w_scale_group, w_zero_group, Y);
    }
  } else if (M <= 512) {
    if (N <= 1024) {
      return _bf16i4bf16<SHUFFLE, SCALE_TYPE, 64, 64, 2, 1, 1, false>(
          X, W, w_scale_group, w_zero_group, Y);
    } else if (N <= 2048) {
      return _bf16i4bf16<SHUFFLE, SCALE_TYPE, 64, 128, 1, 1, 1, false>(
          X, W, w_scale_group, w_zero_group, Y);
    } else if (N <= 4096) {
      return _bf16i4bf16<SHUFFLE, SCALE_TYPE, 64, 256, 1, 1, 1, false>(
          X, W, w_scale_group, w_zero_group, Y);
    } else {
      return _bf16i4bf16<SHUFFLE, SCALE_TYPE, 128, 256, 2, 1, 1, true>(
          X, W, w_scale_group, w_zero_group, Y);
    }
  } else if (M <= 1024) {
    if (N <= 1024) {
      return _bf16i4bf16<SHUFFLE, SCALE_TYPE, 64, 128, 1, 1, 1, false>(
          X, W, w_scale_group, w_zero_group, Y);
    } else if (N <= 2048) {
      return _bf16i4bf16<SHUFFLE, SCALE_TYPE, 64, 256, 1, 1, 1, false>(
          X, W, w_scale_group, w_zero_group, Y);
    } else {
      return _bf16i4bf16<SHUFFLE, SCALE_TYPE, 128, 256, 1, 1, 1, true>(
          X, W, w_scale_group, w_zero_group, Y);
    }
  } else {
    if (M <= 2048 && N <= 1024) {
      return _bf16i4bf16<SHUFFLE, SCALE_TYPE, 64, 256, 2, 1, 1, false>(
          X, W, w_scale_group, w_zero_group, Y);
    } else {
      return _bf16i4bf16<SHUFFLE, SCALE_TYPE, 128, 256, 2, 1, 1, true>(
          X, W, w_scale_group, w_zero_group, Y);
    }
  }
}

at::Tensor bf16i4bf16_shuffled(
    at::Tensor X,
    at::Tensor W,
    at::Tensor w_scale_group,
    at::Tensor w_zero_group) {
  if (w_scale_group.dtype() == at::kFloat) {
    return bf16i4bf16_dispatch<true, float>(X, W, w_scale_group, w_zero_group);
  } else if (w_scale_group.dtype() == at::kBFloat16) {
    return bf16i4bf16_dispatch<true, cutlass::bfloat16_t>(
        X, W, w_scale_group, w_zero_group);
  } else {
    TORCH_CHECK(false, "Only fp32 an bf16 scales supported.")
  }
}

at::Tensor bf16i4bf16_rowwise(
    at::Tensor X, // BF16
    at::Tensor W, // INT4
    at::Tensor w_scale_group,
    at::Tensor w_zero_group) {
  if (w_scale_group.dtype() == at::kFloat) {
    return bf16i4bf16_dispatch<false, float>(X, W, w_scale_group, w_zero_group);
  } else if (w_scale_group.dtype() == at::kBFloat16) {
    return bf16i4bf16_dispatch<false, cutlass::bfloat16_t>(
        X, W, w_scale_group, w_zero_group);
  } else {
    TORCH_CHECK(false, "Only fp32 an bf16 scales supported.")
  }
}

#else

at::Tensor bf16i4bf16_shuffled(
    at::Tensor X,
    at::Tensor W,
    at::Tensor w_scale_group,
    at::Tensor w_zero_group) {
  throw std::runtime_error(
      "CUDA version is older than 12.0"); // requires CUDA>=12
}

at::Tensor bf16i4bf16_rowwise(
    at::Tensor X, // BF16
    at::Tensor W, // INT4
    at::Tensor w_scale_group,
    at::Tensor w_zero_group) {
  throw std::runtime_error(
      "CUDA version is older than 12.0"); // requires CUDA>=12
}

#endif

} // namespace fbgemm_gpu
