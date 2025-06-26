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

#include "cutlass/cutlass.h"

// clang-format off
// The fixed ordering of the headers is required for CUTLASS 3.2+
#include <cute/tensor.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>     // @manual
#include <cutlass/gemm/device/gemm_universal_adapter.h>       // @manual
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include <cutlass/epilogue/collective/collective_builder.hpp> // @manual
#include "cutlass/epilogue/collective/default_epilogue.hpp"

#include "cutlass/util/mixed_dtype_utils.hpp"
#include "cutlass/util/packed_stride.hpp"
// clang-format on

#include "cutlass_extensions/include/kernel_mode.h"

namespace fbgemm_gpu {

#if CUDART_VERSION >= 12000

template <
    bool SHUFFLE,
    int TB_M,
    int TB_N,
    int TB_K,
    int TBS_M,
    int TBS_N,
    int TBS_K,
    bool PONG,
    typename WEIGHT_SCALE_DTYPE>
at::Tensor bf16i4bf16_rowwise_batched_impl(
    at::Tensor X, // BF16
    at::Tensor WQ, // INT4
    at::Tensor w_scale,
    at::Tensor w_zp) {
  // XQ: B x M x K
  // WQ: B x N x K
  // output: B x M x N
  int B = X.size(0);
  int M = X.size(1);
  int N = WQ.size(1);
  int K = X.size(2);

  int num_groups = w_scale.size(0) / B;

  using MmaType = cutlass::bfloat16_t;
  using QuantType = cutlass::int4b_t;

  TORCH_CHECK(X.is_cuda() && X.is_contiguous());
  TORCH_CHECK(WQ.is_cuda() && WQ.is_contiguous());
  TORCH_CHECK(w_scale.is_cuda() && w_scale.is_contiguous());
  TORCH_CHECK(w_zp.is_cuda() && w_zp.is_contiguous());
  TORCH_CHECK(K >= num_groups && K % num_groups == 0);

  int group_size = K / num_groups;

  auto Y = at::empty({B, M, N}, X.options().dtype(at::kBFloat16));

  using ElementA = MmaType;
  using LayoutA = cutlass::layout::RowMajor;
  constexpr int AlignmentA =
      128 /
      cutlass::sizeof_bits<
          ElementA>::value; // Memory access granularity/alignment of A
                            // matrix in units of elements (up to 16 bytes)

  using ElementB = QuantType;
  using LayoutB = cutlass::layout::ColumnMajor;
  constexpr int AlignmentB =
      128 / cutlass::sizeof_bits<ElementB>::value; // Memory access
                                                   // granularity/alignment of B
  // matrix in units of elements (up to 16 bytes)

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

  using ElementScale = WEIGHT_SCALE_DTYPE;
  using ElementZeroPoint = WEIGHT_SCALE_DTYPE;
  using ElementComputeEpilogue = float;
  using ElementAccumulator = float;

  using ElementC = cutlass::bfloat16_t;

  using LayoutC = cutlass::layout::RowMajor;
  constexpr int AlignmentC =
      128 /
      cutlass::sizeof_bits<
          ElementC>::value; // Memory access granularity/alignment of C
                            // matrix in units of elements (up to 16 bytes)

  using ArchTag = cutlass::arch::Sm90; // Tag indicating the minimum SM that
                                       // supports the intended feature
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using TileShape = cute::Shape<
      cute::Int<TB_M>,
      cute::Int<TB_N>,
      cute::Int<TB_K>>; // Threadblock-level
                        // tile size
  using ClusterShape = cute::Shape<
      cute::Int<TBS_M>,
      cute::Int<TBS_N>,
      cute::Int<TBS_K>>; // Shape of the
                         // threadblocks in a
                         // cluster
  using CooperativeSchedule =
      cutlass::gemm::KernelTmaWarpSpecializedCooperative;
  using PongSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
  using CooperativeEpilogueSchedule =
      cutlass::epilogue::TmaWarpSpecializedCooperative;
  using PongEpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized;
  using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;
  using MainLoopSchedule =
      cute::conditional_t<PONG, PongSchedule, CooperativeSchedule>;
  using EpilogueSchedule = cute::
      conditional_t<PONG, PongEpilogueSchedule, CooperativeEpilogueSchedule>;

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
          EpilogueSchedule>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag,
          OperatorClass,
          cute::tuple<ElementB, ElementScale, ElementZeroPoint>,
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
          MainLoopSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideC = typename Gemm::GemmKernel::StrideC;

  auto shape_b = cute::make_shape(N, K, B);
  StrideA stride_a =
      cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, B));
  StrideB stride_b =
      cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, B));
  StrideC stride_c =
      cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(N, M, B));

  LayoutB_Reordered layout_b_reordered =
      cute::tile_to_shape(LayoutAtomQuant{}, shape_b);

  using stride_type = cute::conditional_t<SHUFFLE, LayoutB_Reordered, StrideB>;
  stride_type b_stride;
  if constexpr (SHUFFLE) {
    b_stride = layout_b_reordered;
  } else {
    b_stride = stride_b;
  }

  using StrideS = typename CollectiveMainloop::StrideScale;
  // Note we can support non contiguous strides by actually using the
  // strides of the scales here. This applies to both scale and zeros.
  int32_t scale_stride = w_scale.stride(0);
  StrideS stride_S = cutlass::make_cute_packed_stride(
      StrideS{}, cute::make_shape(scale_stride, num_groups, B));

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {N, M, K, B},
      {reinterpret_cast<ElementB*>(WQ.data_ptr()),
       b_stride,
       reinterpret_cast<ElementA*>(X.data_ptr()),
       stride_a,
       reinterpret_cast<ElementScale*>(w_scale.data_ptr()),
       stride_S,
       group_size,
       reinterpret_cast<ElementZeroPoint*>(w_zp.data_ptr())},
      {{1.0, 0.0},
       (ElementC*)Y.data_ptr<at::BFloat16>(),
       stride_c,
       (ElementC*)Y.data_ptr<at::BFloat16>(),
       stride_c}};

  Gemm gemm;

  // Using the arguments, query for extra workspace required for matrix
  // multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

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

template <bool SHUFFLE, typename WEIGHT_SCALE_DTYPE>
at::Tensor dispatch_bf16i4bf16_rowwise_batched_kernel(
    at::Tensor X, // BF16
    at::Tensor WQ, // INT4
    at::Tensor w_scale,
    at::Tensor w_zp) {
  KernelMode kernel = get_batched_kernel_mode(X, WQ);
  if (kernel == KernelMode::Small) {
    return bf16i4bf16_rowwise_batched_impl<
        SHUFFLE,
        64,
        128,
        64,
        2,
        1,
        1,
        true,
        WEIGHT_SCALE_DTYPE>(X, WQ, w_scale, w_zp);
  } else if (kernel == KernelMode::Large) {
    return bf16i4bf16_rowwise_batched_impl<
        SHUFFLE,
        128,
        256,
        64,
        2,
        1,
        1,
        false,
        WEIGHT_SCALE_DTYPE>(X, WQ, w_scale, w_zp);
  } else {
    return bf16i4bf16_rowwise_batched_impl<
        SHUFFLE,
        128,
        256,
        64,
        2,
        1,
        1,
        false,
        WEIGHT_SCALE_DTYPE>(X, WQ, w_scale, w_zp);
  }
}

at::Tensor bf16i4bf16_shuffled_batched(
    at::Tensor X, // BF16
    at::Tensor WQ, // INT4
    at::Tensor w_scale,
    at::Tensor w_zp) {
  // Check datatypes.
  TORCH_CHECK(
      (w_scale.dtype() == at::kFloat && w_zp.dtype() == at::kFloat) ||
          (w_scale.dtype() == at::kHalf && w_zp.dtype() == at::kHalf) ||
          (w_scale.dtype() == at::kBFloat16 && w_zp.dtype() == at::kBFloat16),
      "Weight scale and zero point tensors must be float32, bfloat16, or float16, and dtype of weight scale and zero point tensors must be the same .");

  if (w_scale.dtype() == at::kFloat) {
    return dispatch_bf16i4bf16_rowwise_batched_kernel<true, float>(
        X, WQ, w_scale, w_zp);
  } else if (w_scale.dtype() == at::kHalf) {
    return dispatch_bf16i4bf16_rowwise_batched_kernel<true, cutlass::half_t>(
        X, WQ, w_scale, w_zp);
  } else if (w_scale.dtype() == at::kBFloat16) {
    return dispatch_bf16i4bf16_rowwise_batched_kernel<
        true,
        cutlass::bfloat16_t>(X, WQ, w_scale, w_zp);
  } else {
    throw std::runtime_error(
        "Weight scale and zero point data type not supported in bf16i4bf16_rowwise_batched");
  }
}

at::Tensor bf16i4bf16_rowwise_batched(
    at::Tensor X, // BF16
    at::Tensor WQ, // INT4
    at::Tensor w_scale,
    at::Tensor w_zp) {
  // Check datatypes.
  TORCH_CHECK(
      (w_scale.dtype() == at::kFloat && w_zp.dtype() == at::kFloat) ||
          (w_scale.dtype() == at::kHalf && w_zp.dtype() == at::kHalf) ||
          (w_scale.dtype() == at::kBFloat16 && w_zp.dtype() == at::kBFloat16),
      "Weight scale and zero point tensors must be float32, bfloat16, or float16, and dtype of weight scale and zero point tensors must be the same .");

  if (w_scale.dtype() == at::kFloat) {
    return dispatch_bf16i4bf16_rowwise_batched_kernel<false, float>(
        X, WQ, w_scale, w_zp);
  } else if (w_scale.dtype() == at::kHalf) {
    return dispatch_bf16i4bf16_rowwise_batched_kernel<false, cutlass::half_t>(
        X, WQ, w_scale, w_zp);
  } else if (w_scale.dtype() == at::kBFloat16) {
    return dispatch_bf16i4bf16_rowwise_batched_kernel<
        false,
        cutlass::bfloat16_t>(X, WQ, w_scale, w_zp);
  } else {
    throw std::runtime_error(
        "Weight scale and zero point data type not supported in bf16i4bf16_rowwise_batched");
  }
}

#else

at::Tensor bf16i4bf16_shuffled_batched(
    at::Tensor X, // BF16
    at::Tensor WQ, // INT4
    at::Tensor w_scale,
    at::Tensor w_zp) {
  throw std::runtime_error(
      "CUDA version is older than 12.0"); // requires CUDA>=12
}

at::Tensor bf16i4bf16_rowwise_batched(
    at::Tensor X, // BF16
    at::Tensor WQ, // INT4
    at::Tensor w_scale,
    at::Tensor w_zp) {
  throw std::runtime_error(
      "CUDA version is older than 12.0"); // requires CUDA>=12
}

#endif

} // namespace fbgemm_gpu
