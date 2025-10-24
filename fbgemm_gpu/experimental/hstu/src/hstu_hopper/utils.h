/*
 * Copyright (c) 2024, Tri Dao.
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <tuple>

#include <cuda_fp16.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#include <cuda_bf16.h>
#endif

#include <cute/arch/cluster_sm90.hpp> // For cute::elect_one_sync()
#include <cute/tensor.hpp>

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#define NO_FP8_COLUMN_PERMUTE

#ifndef EPSILON
#define EPSILON 1e-6
#endif

#define CHECK_CUDA(call)                \
  do {                                  \
    cudaError_t status_ = call;         \
    if (status_ != cudaSuccess) {       \
      fprintf(                          \
          stderr,                       \
          "CUDA error (%s:%d): %s\n",   \
          __FILE__,                     \
          __LINE__,                     \
          cudaGetErrorString(status_)); \
      exit(1);                          \
    }                                   \
  } while (0)

#define CHECK_CUDA_KERNEL_LAUNCH() CHECK_CUDA(cudaGetLastError())

namespace flash {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

static __device__ float tanh_fast(float x) {
  float res;
  asm volatile("{ tanh.approx.f32 %0, %1; }\n" : "=f"(res) : "f"(x));
  return res;
}

static __device__ float sigmoid_fast(float x) {
  return 0.5f * tanh_fast(0.5f * x) + 0.5f;
}

template <typename Engine, typename Layout>
CUTLASS_DEVICE void silu(Tensor<Engine, Layout>& t) {
  using ValT = typename Engine::value_type;
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < size(t); ++i) {
    float v = static_cast<float>(t(i));
    float sigmoid_v = sigmoid_fast(v);
    float out = v * sigmoid_v;
    float silu_out = v > -10.0f ? out : 0.f;
    t(i) = static_cast<ValT>(silu_out);
  }
}

template <typename Engine, typename Layout>
CUTLASS_DEVICE void fast_silu(Tensor<Engine, Layout>& t) {
  using ValT = typename Engine::value_type;
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < size(t); ++i) {
    float v = static_cast<float>(t(i)) * 0.5f;
    float tanh_v = tanh_fast(v);
    t(i) = v > -10.0f ? __fmaf_rn(v, tanh_v, v) : 0.f;
  }
}

template <
    typename Engine0,
    typename Layout0,
    typename Engine1,
    typename Layout1>
CUTLASS_DEVICE void silu_bwd(
    Tensor<Engine0, Layout0>& x,
    Tensor<Engine1, Layout1>& y) {
  static_assert(decltype(size(x))::value == decltype(size(y))::value);
  using ValT0 = typename Engine0::value_type;
  using ValT1 = typename Engine1::value_type;
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < size(x); ++i) {
    float v = static_cast<float>(x(i));
    float sigmoid_v = sigmoid_fast(v);
    float out = v * sigmoid_v;
    float temp = sigmoid_v * (1 + v * (1 - sigmoid_v));
    float dsilu_temp = v > -10.0f ? temp : 0.f;
    float silu_out = v > -10.0f ? out : 0.f;
    x(i) = static_cast<ValT0>(dsilu_temp);
    y(i) = static_cast<ValT1>(silu_out);
  }
}

template <typename Tensor0, typename Tensor1>
CUTLASS_DEVICE void dsilu_bwd(Tensor0& dy, Tensor1& x) {
  static_assert(decltype(size(dy))::value == decltype(size(x))::value);
  using ValT = typename Tensor0::value_type;
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < size(dy); ++i) {
    float dsilu_temp = static_cast<float>(x(i));
    float dyv = static_cast<float>(dy(i));
    float out = dyv * dsilu_temp;
    float dsilu_out = out;
    dy(i) = static_cast<ValT>(dsilu_out);
  }
}

template<typename T>
struct MaxOp {
  __device__ __forceinline__ T operator()(const T& a, const T& b) const {
    return max(a, b);
  }
};

template<typename T>
struct MinOp {
  __device__ __forceinline__ T operator()(const T& a, const T& b) const {
    return min(a, b);
  }
};

template<typename T, typename Op>
__inline__ __device__ void warpReduce(T& val, Op op) {
  CUTLASS_PRAGMA_UNROLL
  for (int mask = 16; mask > 0; mask >>= 1)
    val = op(val, __shfl_xor_sync(0xffffffff, val, mask, 32));
}
//
////////////////////////////////////////////////////////////////////////////////////////////////////

// For SM90, convert acc_layout from ((2, 2, N / 8), MMA_M, MMA_N) to ((2, 2,
// 2), MMA_M, (N / 16, MMA_N))
template <typename MMA_traits, typename Layout>
CUTLASS_DEVICE auto convert_layout_acc_Aregs(Layout acc_layout) {
  using X = Underscore;
  if constexpr (decltype(rank<0>(acc_layout))::value == 3) {
    static_assert(decltype(size<0, 0>(acc_layout))::value == 2);
    static_assert(decltype(size<0, 1>(acc_layout))::value == 2);
    static_assert(decltype(rank(acc_layout))::value == 3);
    static_assert(decltype(rank(get<0>(acc_layout)))::value == 3);
    auto l = logical_divide(
        get<0>(acc_layout), Shape<X, X, _2>{}); // (2, 2, (2, N / 16)))
    return make_layout(
        make_layout(get<0>(l), get<1>(l), get<2, 0>(l)),
        get<1>(acc_layout),
        make_layout(get<2, 1>(l), get<2>(acc_layout)));
  } else {
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    constexpr int mma_shape_K = get<2>(typename MMA_traits::Shape_MNK{});
    static_assert(mma_shape_K == 8 || mma_shape_K == 16);
    if constexpr (mma_shape_K == 8) {
      return acc_layout;
    } else {
      auto l = logical_divide(
          acc_layout, Shape<X, X, _2>{}); // (4, MMA_M, (2, MMA_N / 2)))
      return make_layout(
          make_layout(get<0>(l), get<2, 0>(l)), get<1>(l), get<2, 1>(l));
    }
  }
};

// Convert acc_layout from ((2, 2, N / 8), MMA_M, MMA_N) to ((4, 2, 2), MMA_M,
// (N / 32, MMA_N))
template <typename Layout>
CUTLASS_DEVICE auto convert_layout_acc_Aregs_fp8(Layout acc_layout) {
  using X = Underscore;
  static_assert(decltype(size<0, 0>(acc_layout))::value == 2);
  static_assert(decltype(size<0, 1>(acc_layout))::value == 2);
  static_assert(decltype(rank(acc_layout))::value == 3);
  static_assert(decltype(rank(get<0>(acc_layout)))::value == 3);
  auto l = logical_divide(
      get<0>(acc_layout), Shape<X, X, _4>{}); // (2, 2, (2, N / 32)))
  return make_layout(
      make_layout(Shape<_4, _2, _2>{}),
      get<1>(acc_layout),
      make_layout(get<2, 1>(l), get<2>(acc_layout)));
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Byte permute for fp8 kernel
template <bool Transpose = false, typename Fragment>
CUTLASS_DEVICE void permute_regs_C_to_A(Fragment& accum) {
  static constexpr int upper_map[4] = {0, 3, 1, 2};
  static constexpr int lower_map[4] = {1, 2, 0, 3};

  int quad_idx = threadIdx.x % 4;
  bool lane_03 = quad_idx == 0 || quad_idx == 3;

  Tensor frag_64b = recast<uint2>(accum);
  uint32_t upper, lower;
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < size(frag_64b); ++i) {
    upper = frag_64b[i].x;
    lower = frag_64b[i].y;
    if constexpr (!Transpose) {
      uint32_t upper0 = lane_03 ? upper : lower;
      uint32_t lower0 = lane_03 ? lower : upper;
      upper0 = __shfl_sync(uint32_t(-1), upper0, upper_map[quad_idx], 4);
      lower0 = __shfl_sync(uint32_t(-1), lower0, lower_map[quad_idx], 4);
      upper = lane_03 ? upper0 : lower0;
      lower = lane_03 ? lower0 : upper0;
    }
    frag_64b[i].x = __byte_perm(upper, lower, 0x5410);
    frag_64b[i].y = __byte_perm(upper, lower, 0x7632);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename To_type, typename Engine, typename Layout>
CUTLASS_DEVICE auto convert_type(Tensor<Engine, Layout> const& tensor) {
  using From_type = typename Engine::value_type;
  constexpr int numel = decltype(size(tensor))::value;
  cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
  // HACK: this requires tensor to be "contiguous"
  auto frag =
      convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel>*>(
          tensor.data()));
  return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Engine, typename Layout, typename EngineOut>
CUTLASS_DEVICE void convert_type_safe(
    Tensor<Engine, Layout> const& tensor,
    Tensor<EngineOut, Layout>& out) {
  using From_type = typename Engine::value_type;
  using To_type = typename EngineOut::value_type;
  constexpr int numel = decltype(size(tensor))::value;
  cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
  // HACK: this requires tensor to be "contiguous"
  auto frag =
      convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel>*>(
          tensor.data()));
  cute::copy(make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout()), out);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    bool zero_init = false,
    int wg_wait = 0,
    bool SwapAB = false,
    int M_slice = -1,
    typename Tensor0,
    typename Tensor1,
    typename Tensor2,
    typename TiledMma>
CUTLASS_DEVICE void gemm(
    TiledMma& tiled_mma,
    Tensor0 const& tCrA,
    Tensor1 const& tCrB,
    Tensor2& tCrC) {
  if constexpr (M_slice >= 0) {
    static constexpr int MMA_M = decltype(size<1>(tCrC))::value;
    static_assert(M_slice < MMA_M);
    // After logical_divide, C has shape ((2,2,V), (MMA_M, 1), MMA_N)
    Tensor tCrC_slice =
        cute::logical_divide(tCrC, Shape<cute::Underscore, Int<MMA_M>>{})(
            _, make_coord(Int<M_slice>{}, _), _);
    if constexpr (!SwapAB) {
      Tensor tCrA_slice =
          cute::logical_divide(tCrA, Shape<cute::Underscore, Int<MMA_M>>{})(
              _, make_coord(Int<M_slice>{}, _), _);
      gemm<zero_init, wg_wait, SwapAB, /*M_slice=*/-1>(
          tiled_mma, tCrA_slice, tCrB, tCrC_slice);
    } else {
      Tensor tCrB_slice =
          cute::logical_divide(tCrB, Shape<cute::Underscore, Int<MMA_M>>{})(
              _, make_coord(Int<M_slice>{}, _), _);
      gemm<zero_init, wg_wait, SwapAB, /*M_slice=*/-1>(
          tiled_mma, tCrA, tCrB_slice, tCrC_slice);
    }
  } else {
    constexpr bool Is_RS = !cute::is_base_of<
        cute::GMMA::DescriptorIterator,
        typename TiledMma::FrgTypeA>::value;
    // Need to cast away const on tCrA since warpgroup_fence_operand doesn't
    // take const
    if constexpr (Is_RS) {
      if constexpr (!SwapAB) {
        warpgroup_fence_operand(const_cast<Tensor0&>(tCrA));
      } else {
        warpgroup_fence_operand(const_cast<Tensor1&>(tCrB));
      }
    }
    warpgroup_fence_operand(tCrC);
    warpgroup_arrive();
    if constexpr (zero_init) {
      tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
    }
    // Unroll the K mode manually to set scale D to 1
    CUTLASS_PRAGMA_UNROLL
    for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
      if constexpr (!SwapAB) {
        cute::gemm(tiled_mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tCrC);
      } else {
        cute::gemm(tiled_mma, tCrB(_, _, k_block), tCrA(_, _, k_block), tCrC);
      }
      tiled_mma.accumulate_ = GMMA::ScaleOut::One;
    }
    warpgroup_commit_batch();
    if constexpr (wg_wait >= 0) {
      warpgroup_wait<wg_wait>();
    }
    warpgroup_fence_operand(tCrC);
    if constexpr (Is_RS) {
      if constexpr (!SwapAB) {
        warpgroup_fence_operand(const_cast<Tensor0&>(tCrA));
      } else {
        warpgroup_fence_operand(const_cast<Tensor1&>(tCrB));
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    bool Is_even_MN = true,
    bool Is_even_K = true,
    bool Clear_OOB_MN = false,
    bool Clear_OOB_K = true,
    typename TiledCopy,
    typename Engine0,
    typename Layout0,
    typename Engine1,
    typename Layout1,
    typename Engine2,
    typename Layout2,
    typename Engine3,
    typename Layout3>
CUTLASS_DEVICE void copy(
    TiledCopy tiled_copy,
    Tensor<Engine0, Layout0> const& S,
    Tensor<Engine1, Layout1>& D,
    Tensor<Engine2, Layout2> const& identity_MN,
    Tensor<Engine3, Layout3> const& predicate_K,
    const int max_MN = 0) {
  CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
  CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
  CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D)); // MMA
  CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D)); // MMA_M
  CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D)); // MMA_K
  // There's no case where !Clear_OOB_K && Clear_OOB_MN
  static_assert(!(Clear_OOB_MN && !Clear_OOB_K));
  CUTLASS_PRAGMA_UNROLL
  for (int m = 0; m < size<1>(S); ++m) {
    if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN) {
      CUTLASS_PRAGMA_UNROLL
      for (int k = 0; k < size<2>(S); ++k) {
        if (Is_even_K || predicate_K(k)) {
          cute::copy(tiled_copy, S(_, m, k), D(_, m, k));
        } else if (Clear_OOB_K) {
          cute::clear(D(_, m, k));
        }
      }
    } else if (Clear_OOB_MN) {
      cute::clear(D(_, m, _));
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    int NumCopyThreads,
    typename ElemO,
    typename TiledCopyO,
    typename LayoutO,
    typename TileShapeO,
    typename SMemO,
    typename SeqLenTraits>
CUTLASS_DEVICE void write_tiled(
    ElemO* O,
    const TiledCopyO& tiled_copy_O,
    const LayoutO& layout_O,
    const TileShapeO& tile_shape_O,
    const SMemO& sO,
    int m_block,
    int bidh,
    int bidb,
    const SeqLenTraits& seqlen_traits_o) {
  Tensor mO = make_tensor(make_gmem_ptr(O), layout_O);
  Tensor gO = seqlen_traits_o.get_local_tile_tensor(
      mO, tile_shape_O, bidh, bidb)(_, _, m_block); // (M, K)

  ThrCopy thr_copy_O = tiled_copy_O.get_slice(threadIdx.x - NumCopyThreads);
  Tensor tOgO = thr_copy_O.partition_D(gO); // (CPY,CPY_M,CPY_K,k)
  Tensor tOsO = thr_copy_O.partition_S(sO); // (CPY,CPY_M,CPY_K)

  // Prepare for TiledCopy.
  // Grouping is needed because cute::copy_if() does group_modes<1, R> for src
  // and dst. After grouping, the first dim is number of elements to read
  // together.
  Tensor tOsOFlatten = cute::flatten(tOsO);
  Tensor tOsOGroup = cute::group_modes<1, rank(tOsOFlatten)>(tOsOFlatten);
  Tensor tOgOFlatten = cute::flatten(tOgO);
  Tensor tOgOGroup = cute::group_modes<1, rank(tOgOFlatten)>(tOgOFlatten);

  // Get thread coords to global index mapping.
  Tensor gOCounting = cute::make_identity_tensor(gO.shape());
  Tensor tSgOCounting = thr_copy_O.partition_D(gOCounting);
  Tensor tSgOCountingFlatten = cute::flatten(tSgOCounting);
  Tensor tSgOCountingGrouped =
      cute::group_modes<1, rank(tSgOCountingFlatten)>(tSgOCountingFlatten);

  // Write out to GMEM.
  const int kNumMsPerTile = get<0>(tile_shape_O);
  int cta_m = std::min(
      seqlen_traits_o.actual_seq_len - m_block * kNumMsPerTile, kNumMsPerTile);
  if (cta_m == kNumMsPerTile) {
    copy(tiled_copy_O, tOsOGroup, tOgOGroup);
  } else {
    auto predicate_fn = [&](auto coords) {
      auto s_coords = tSgOCountingGrouped(_0{}, coords);
      return elem_less(get<0>(s_coords), cta_m);
    };
    copy_if(tiled_copy_O, predicate_fn, tOsOGroup, tOgOGroup);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// {kBlockM, kBlockN, kNWarps}
template <int Headdim, bool Has_rab, bool Is_fp8>
constexpr std::tuple<int, int, int> get_tile_size_fwd() {
  if constexpr (Is_fp8) {
    if constexpr (Has_rab) {
      if constexpr (Headdim == 64) {
        return {128, 128, 12};
      } else {
        return {128, 64, 12};
      }
    } else {
      if constexpr (Headdim == 64) {
        return {128, 128, 12};
      } else if constexpr (Headdim == 128) {
        return {128, 128, 12};
      } else {
        return {128, 64, 12};
      }
    }
  } else {
    if constexpr (Has_rab) {
      if constexpr (Headdim == 32) {
        return {192, 128, 16};
      } else if constexpr (Headdim == 64) {
        return {128, 128, 12};
      } else {
        return {128, 64, 12};
      }
    } else {
      if constexpr (Headdim <= 64) {
        return {192, 128, 16};
      } else if constexpr (Headdim == 128) {
        return {128, 128, 12};
      } else {
        return {128, 64, 12};
      }
    }
  }
}

// {kBlockM, kBlockN, kNWarpGroups}
template <int Headdim, bool Has_rab, bool Is_fp8>
constexpr std::tuple<int, int, int> get_tile_size_bwd() {
  if constexpr (Is_fp8) {
    return {64, 128, 3};
  } else {
    if constexpr (Has_rab) {
      if constexpr (Headdim <= 64) {
        return {64, 128, 3};
      } else {
        return {64, 64, 3};
      }
    } else {
      if constexpr (Headdim <= 64) {
        return {128, 128, 3};
      } else if constexpr (Headdim <= 128) {
        return {64, 128, 3};
      } else {
        return {64, 64, 3};
      }
    }
  }
}

} // namespace flash
