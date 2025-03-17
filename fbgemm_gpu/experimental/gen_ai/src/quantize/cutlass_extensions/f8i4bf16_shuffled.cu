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

template <int TB_M, int TB_N, int TBS_M, int TBS_N, int TBS_K, bool COOP>
at::Tensor _f8i4bf16_shuffled(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor w_scale_group,
    at::Tensor Y) {
  // Get shape information from input tensors.
  int M = size_to_dim_(XQ.dim() - 1, XQ.sizes());
  int K = XQ.size(-1);
  int N = size_to_dim_(WQ.dim() - 1, WQ.sizes());
  int num_groups = w_scale_group.size(0);
  int group_size = K / num_groups;

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
  // Implement rowwise scaling epilogue.
  using XScale = cutlass::epilogue::fusion::Sm90ColBroadcast<
      0,
      TileShape,
      ElementAccumulator,
      ElementAccumulator,
      cute::Stride<cute::Int<1>, cute::Int<0>, cute::Int<0>>>;

  using WScale = cutlass::epilogue::fusion::Sm90RowBroadcast<
      0,
      TileShape,
      ElementAccumulator,
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
       reinterpret_cast<cutlass::Array<ElementScale, 8>*>(
           w_scale_group.data_ptr()),
       stride_S,
       group_size},
      {{},
       reinterpret_cast<ElementC*>(Y.data_ptr()),
       stride_C,
       reinterpret_cast<ElementC*>(Y.data_ptr()),
       stride_C}};

  arguments.epilogue.thread = {
      {reinterpret_cast<ElementAccumulator*>(w_scale.data_ptr())}, // w_scale
      // compute_0
      {
          {reinterpret_cast<ElementAccumulator*>(
              x_scale.data_ptr())}, // w_scale
          {}, // Accumulator
          {} // Multiplies
      },
      {}, // Multiplies
  };

  // Launch the workload.
  GemmShuffled gemm;

  // Using the arguments, query for extra workspace required for matrix
  // multiplication computation
  int workspace_size = GemmShuffled::get_workspace_size(arguments);

  // Allocate workspace memory
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

  return Y;
}

at::Tensor f8i4bf16_shuffled(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor w_scale_group) {
  int M = size_to_dim_(XQ.dim() - 1, XQ.sizes());
  int K = XQ.size(-1);
  int N = size_to_dim_(WQ.dim() - 1, WQ.sizes());
  // Check input types and shapes.
  TORCH_CHECK(
      XQ.is_cuda() && XQ.is_contiguous() && XQ.dtype() == at::kFloat8_e4m3fn,
      "XQ must be FP8 and contiguous on GPU.");
  TORCH_CHECK(
      WQ.size(-1) == K / 2 && WQ.is_cuda() && WQ.is_contiguous() &&
          WQ.dtype() == at::kChar,
      "WQ should be int8 (which represent two int4 values), have shape [..., N, K/2], "
      "and be contiguous on GPU.");
  TORCH_CHECK(
      x_scale.numel() == M && x_scale.dtype() == at::kFloat &&
          x_scale.is_cuda(),
      "x_scale must be fp32 and have M total elements.");
  TORCH_CHECK(
      w_scale.numel() == N && w_scale.dtype() == at::kFloat &&
          w_scale.is_cuda(),
      "Weight row scale should have N elements and be on GPU.");
  // Make sure w_scale_group is in proper format.
  TORCH_CHECK(
      w_scale_group.dtype() == at::kFloat8_e4m3fn && w_scale_group.dim() == 3 &&
          w_scale_group.size(1) == 8 && w_scale_group.size(2) == N,
      "Weights and group scales must be prepacked with preshuffle_i4. "
      "Group scales are expected to be FP8 and have shape [num_groups, 8, N].");

  // Allocate output or return an empty tensor if input is empty.
  if (M == 0 || N == 0 || K == 0) {
    return at::zeros({M, N}, XQ.options().dtype(at::kBFloat16));
  }
  at::Tensor Y = at::empty({M, N}, XQ.options().dtype(at::kBFloat16));

  // Use shape heuristics to dispatch to optimized kernel configuration.
  if (M <= 16) {
    return _f8i4bf16_shuffled<64, 16, 1, 1, 1, false>(
        XQ, WQ, x_scale, w_scale, w_scale_group, Y);
  } else if (M <= 32) {
    if (N <= 4096) {
      return _f8i4bf16_shuffled<64, 16, 1, 1, 1, false>(
          XQ, WQ, x_scale, w_scale, w_scale_group, Y);
    } else {
      return _f8i4bf16_shuffled<64, 32, 2, 1, 1, false>(
          XQ, WQ, x_scale, w_scale, w_scale_group, Y);
    }
  } else if (M <= 64) {
    if (N <= 2048) {
      return _f8i4bf16_shuffled<64, 16, 1, 1, 1, false>(
          XQ, WQ, x_scale, w_scale, w_scale_group, Y);
    } else if (N <= 4096) {
      return _f8i4bf16_shuffled<64, 32, 2, 1, 1, false>(
          XQ, WQ, x_scale, w_scale, w_scale_group, Y);
    } else {
      return _f8i4bf16_shuffled<64, 64, 2, 1, 1, false>(
          XQ, WQ, x_scale, w_scale, w_scale_group, Y);
    }
  } else if (M <= 128) {
    if (N <= 1024) {
      return _f8i4bf16_shuffled<64, 16, 1, 1, 1, false>(
          XQ, WQ, x_scale, w_scale, w_scale_group, Y);
    } else if (N <= 2048) {
      return _f8i4bf16_shuffled<64, 32, 2, 1, 1, false>(
          XQ, WQ, x_scale, w_scale, w_scale_group, Y);
    } else if (N <= 4096) {
      return _f8i4bf16_shuffled<64, 64, 2, 1, 1, false>(
          XQ, WQ, x_scale, w_scale, w_scale_group, Y);
    } else {
      return _f8i4bf16_shuffled<64, 128, 1, 1, 1, false>(
          XQ, WQ, x_scale, w_scale, w_scale_group, Y);
    }
  } else if (M <= 256) {
    if (N <= 1024) {
      return _f8i4bf16_shuffled<64, 32, 2, 1, 1, false>(
          XQ, WQ, x_scale, w_scale, w_scale_group, Y);
    } else if (N <= 2048) {
      return _f8i4bf16_shuffled<64, 64, 2, 1, 1, false>(
          XQ, WQ, x_scale, w_scale, w_scale_group, Y);
    } else if (N <= 4096) {
      return _f8i4bf16_shuffled<64, 128, 1, 1, 1, false>(
          XQ, WQ, x_scale, w_scale, w_scale_group, Y);
    } else {
      return _f8i4bf16_shuffled<64, 256, 1, 1, 1, false>(
          XQ, WQ, x_scale, w_scale, w_scale_group, Y);
    }
  } else if (M <= 512) {
    if (N <= 1024) {
      return _f8i4bf16_shuffled<64, 64, 2, 1, 1, false>(
          XQ, WQ, x_scale, w_scale, w_scale_group, Y);
    } else if (N <= 2048) {
      return _f8i4bf16_shuffled<64, 128, 1, 1, 1, false>(
          XQ, WQ, x_scale, w_scale, w_scale_group, Y);
    } else if (N <= 4096) {
      return _f8i4bf16_shuffled<64, 256, 1, 1, 1, false>(
          XQ, WQ, x_scale, w_scale, w_scale_group, Y);
    } else {
      return _f8i4bf16_shuffled<128, 256, 2, 1, 1, true>(
          XQ, WQ, x_scale, w_scale, w_scale_group, Y);
    }
  } else if (M <= 1024) {
    if (N <= 1024) {
      return _f8i4bf16_shuffled<64, 128, 1, 1, 1, false>(
          XQ, WQ, x_scale, w_scale, w_scale_group, Y);
    } else if (N <= 2048) {
      return _f8i4bf16_shuffled<64, 256, 1, 1, 1, false>(
          XQ, WQ, x_scale, w_scale, w_scale_group, Y);
    } else {
      return _f8i4bf16_shuffled<128, 256, 1, 1, 1, true>(
          XQ, WQ, x_scale, w_scale, w_scale_group, Y);
    }
  } else {
    if (M <= 2048 && N <= 1024) {
      return _f8i4bf16_shuffled<64, 256, 2, 1, 1, false>(
          XQ, WQ, x_scale, w_scale, w_scale_group, Y);
    } else {
      return _f8i4bf16_shuffled<128, 256, 2, 1, 1, true>(
          XQ, WQ, x_scale, w_scale, w_scale_group, Y);
    }
  }
}

#else

at::Tensor f8i4bf16_shuffled(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor w_scale_group) {
  throw std::runtime_error(
      "CUDA version is older than 12.0"); // requires CUDA>=12
}

#endif

} // namespace fbgemm_gpu
