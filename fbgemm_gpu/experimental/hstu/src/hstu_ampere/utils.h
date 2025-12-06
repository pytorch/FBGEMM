/*
 * Copyright (c) 2023, Tri Dao.
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <assert.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdlib.h>
#include <tuple>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#include <cuda_bf16.h>
#endif

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include <cute/tensor.hpp>

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef HSTU_DEBUG_ENABLED
#define HSTU_DEBUG_ENABLED 0
#endif

#if HSTU_DEBUG_ENABLED
#define HSTU_DEBUG_INFO(cond, ...)                 \
 do {                                              \
   if (cond) {                                     \
     printf(__VA_ARGS__);                          \
   }                                               \
 } while (0)
#else
#define HSTU_DEBUG_INFO(cond, ...) do {} while (0)
#endif

namespace flash {

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
inline __device__ void silu(Tensor<Engine, Layout>& t) {
  using ValT = typename Engine::value_type;
#pragma unroll
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
inline __device__ void silu_bwd(
    Tensor<Engine0, Layout0>& x,
    Tensor<Engine1, Layout1>& y) {
  static_assert(decltype(size(x))::value == decltype(size(y))::value);
  using ValT0 = typename Engine0::value_type;
  using ValT1 = typename Engine1::value_type;
#pragma unroll
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
inline __device__ void dsilu_bwd(Tensor0& dy, Tensor1& x) {
  static_assert(decltype(size(dy))::value == decltype(size(x))::value);
  using ValT = typename Tensor0::value_type;
#pragma unroll
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
  #pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    val = op(val, __shfl_xor_sync(0xffffffff, val, mask, 32));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Engine, typename Layout, typename EngineOut>
CUTLASS_DEVICE void convert_type_safe(
    Tensor<Engine, Layout> const& tensor,
    Tensor<EngineOut, Layout>& out) {
  // Somehow if we allocate out inside this function and return it, e2e is
  // slower and the output can be wrong.
  using From_type = typename Engine::value_type;
  using To_type = typename EngineOut::value_type;
  static constexpr int FragmentSize = std::max(
      sizeof(From_type) / sizeof(To_type), sizeof(To_type) / sizeof(From_type));
  static_assert(
      CUTE_STATIC_V(size(tensor)) % FragmentSize == 0,
      "Fragment size does not vectorize properly");
  Tensor frag = recast<cutlass::Array<From_type, FragmentSize> const>(tensor);
  Tensor out_frg = recast<cutlass::Array<To_type, FragmentSize>>(out);
  static_assert(size(frag) == size(out_frg));
  cutlass::NumericArrayConverter<To_type, From_type, FragmentSize> convert_op;
#pragma unroll
  for (int i = 0; i < size(frag); ++i) {
    out_frg[i] = convert_op(frag[i]);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    bool A_in_regs = false,
    bool B_in_regs = false,
    typename Tensor0,
    typename Tensor1,
    typename Tensor2,
    typename Tensor3,
    typename Tensor4,
    typename TiledMma,
    typename TiledCopyA,
    typename TiledCopyB,
    typename ThrCopyA,
    typename ThrCopyB>
__forceinline__ __device__ void gemm(
    Tensor0& acc,
    Tensor1& tCrA,
    Tensor2& tCrB,
    Tensor3 const& tCsA,
    Tensor4 const& tCsB,
    TiledMma tiled_mma,
    TiledCopyA smem_tiled_copy_A,
    TiledCopyB smem_tiled_copy_B,
    ThrCopyA smem_thr_copy_A,
    ThrCopyB smem_thr_copy_B) {
  CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc)); // MMA_M
  CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc)); // MMA_N
  CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB)); // MMA_K
  Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
  CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view)); // M
  Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
  CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view)); // N
  if (!A_in_regs) {
    cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{}));
  }
  if (!B_in_regs) {
    cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
  }
#pragma unroll
  for (int i = 0; i < size<2>(tCsA); ++i) {
    if (i < size<2>(tCsA) - 1) {
      if (!A_in_regs) {
        cute::copy(
            smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1));
      }
      if (!B_in_regs) {
        cute::copy(
            smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
      }
    }
    cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    typename Tensor0,
    typename Tensor1,
    typename Tensor2,
    typename Tensor3,
    typename TiledMma,
    typename TiledCopy,
    typename ThrCopy>
__forceinline__ __device__ void gemm_rs(
    Tensor0& acc,
    Tensor1& tCrA,
    Tensor2& tCrB,
    Tensor3 const& tCsB,
    TiledMma tiled_mma,
    TiledCopy smem_tiled_copy_B,
    ThrCopy smem_thr_copy_B) {
  CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc)); // MMA_M
  CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc)); // MMA_N
  CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB)); // MMA_K
  Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
  CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view)); // N
  cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
#pragma unroll
  for (int i = 0; i < size<2>(tCrA); ++i) {
    if (i < size<2>(tCrA) - 1) {
      cute::copy(
          smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
    }
    cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Convert acc_layout from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
// if using m16n8k16, or to (4, MMA_M, MMA_N) if using m16n8k8.
template <typename MMA_traits, typename Layout>
__forceinline__ __device__ auto convert_layout_acc_Aregs(Layout acc_layout) {
  using X = Underscore;
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
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Blocks until all but N previous cp.async.commit_group operations have
// committed. This differs from cute::cp_async_wait in that when N = 0 we don't
// call cp.async.wait_all (which is equivalent to commit_group then wait_group
// 0). Instead we just call cp.async.wait_group 0, which is slightly faster.
// https://github.com/NVIDIA/cutlass/blob/master/include/cute/arch/copy_sm80.hpp#L113
template <int N>
CUTE_HOST_DEVICE void cp_async_wait() {
#if defined(CUTE_ARCH_CP_ASYNC_SM80_ENABLED)
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    bool Is_even_MN,
    bool Clear_OOB_MN = false,
    bool Clear_OOB_K = true,
    typename TiledCopy,
    typename Engine0,
    typename Layout0,
    typename Engine1,
    typename Layout1,
    typename Engine2,
    typename Layout2>
__forceinline__ __device__ void copy(
    TiledCopy tiled_copy,
    Tensor<Engine0, Layout0> const& S,
    Tensor<Engine1, Layout1>& D,
    Tensor<Engine2, Layout2> const& identity_MN,
    const int max_MN = 0) {
  CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
  CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
  CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D)); // MMA
  CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D)); // MMA_M
  CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D)); // MMA_K
  // There's no case where !Clear_OOB_K && Clear_OOB_MN
  static_assert(!(Clear_OOB_MN && !Clear_OOB_K));
#pragma unroll
  for (int m = 0; m < size<1>(S); ++m) {
    if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN) {
#pragma unroll
      for (int k = 0; k < size<2>(S); ++k) {
        cute::copy(tiled_copy, S(_, m, k), D(_, m, k));
      }
    } else if (Clear_OOB_MN) {
      cute::clear(D(_, m, _));
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// {kBlockM, kBlockN, kNWarps}
template <int Arch, int kHeadDim, bool Has_rab>
constexpr std::tuple<int, int, int> get_tile_size_fwd() {
  if constexpr (Arch == 80) {
    if constexpr (Has_rab) {
      return {128, 64, 8};
    } else {
      if constexpr (kHeadDim <= 64) {
        return {128, 96, 4};
      } else if constexpr (kHeadDim <= 128) {
        return {128, 96, 8};
      } else {
        return {128, 96, 8};
      }
    }
  } else {
    if constexpr (Has_rab) {
      if constexpr (kHeadDim <= 128) {
        return {128, 64, 4};
      } else {
        return {64, 32, 4};
      }
    } else {
      if constexpr (kHeadDim <= 128) {
        return {128, 96, 4};
      } else {
        return {64, 64, 4};
      }
    }
  }
}

// {kBlockM, kBlockN, kNWarps}
template <int kHeadDim, bool Has_rab>
constexpr std::tuple<int, int, int> get_tile_size_bwd() {
  if constexpr (kHeadDim <= 128) {
    return {64, 128, 8};
  } else {
    return {64, 64, 8};
  }
}

} // namespace flash
