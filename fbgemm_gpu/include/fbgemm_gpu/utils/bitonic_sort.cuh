/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "fbgemm_gpu/utils/cuda_prelude.cuh"

namespace fbgemm_gpu {

/// Warp bitonic K/V sorting code
template <typename T>
struct Comparator {
  __device__ static inline bool lt(T a, T b) {
    return a < b;
  }
  __device__ static inline bool gt(T a, T b) {
    return a > b;
  }
};

template <typename T>
inline __device__ void assign(bool assign, T& x, T y) {
  x = assign ? y : x;
}

template <
    typename K,
    typename V,
    int32_t L,
    bool Dir,
    typename Comp,
    bool IsBitonic>
inline __device__ void warpBitonicMergeLE16(K& k, V& v) {
  static_assert(
      L <= fbgemm_gpu::kWarpSize / 2, "merge list size must be <= 16");
  int32_t laneId = threadIdx.x;

  if (!IsBitonic) {
    // Reverse the first comparison stage.
    // For example, merging a list of size 8 has the exchanges:
    // 0 <-> 15, 1 <-> 14, ...
    K otherK = fbgemm_gpu::shfl_xor(k, 2 * L - 1);
    V otherV = fbgemm_gpu::shfl_xor(v, 2 * L - 1);

    // Whether we are the lesser thread in the exchange
    bool small = !(laneId & L);

    if (Dir) {
      // See the comment above how performing both of these
      // comparisons in the warp seems to win out over the
      // alternatives in practice
      bool s = small ? Comp::gt(k, otherK) : Comp::lt(k, otherK);
      assign(s, k, otherK);
      assign(s, v, otherV);

    } else {
      bool s = small ? Comp::lt(k, otherK) : Comp::gt(k, otherK);
      assign(s, k, otherK);
      assign(s, v, otherV);
    }
  }

#pragma unroll
  for (int32_t stride = IsBitonic ? L : L / 2; stride > 0; stride /= 2) {
    K otherK = fbgemm_gpu::shfl_xor(k, stride);
    V otherV = fbgemm_gpu::shfl_xor(v, stride);

    // Whether we are the lesser thread in the exchange
    bool small = !(laneId & stride);

    if (Dir) {
      bool s = small ? Comp::gt(k, otherK) : Comp::lt(k, otherK);
      assign(s, k, otherK);
      assign(s, v, otherV);

    } else {
      bool s = small ? Comp::lt(k, otherK) : Comp::gt(k, otherK);
      assign(s, k, otherK);
      assign(s, v, otherV);
    }
  }
}

template <typename K, typename V, bool Dir, typename Comp>
struct BitonicSort {
  static inline __device__ void sort(K k[1], V v[1]) {
#ifdef USE_ROCM
    static_assert(fbgemm_gpu::kWarpSize == 64, "unexpected warp size");
#else
    static_assert(fbgemm_gpu::kWarpSize == 32, "unexpected warp size");
#endif
    warpBitonicMergeLE16<K, V, 1, Dir, Comp, false>(k[0], v[0]);
    warpBitonicMergeLE16<K, V, 2, Dir, Comp, false>(k[0], v[0]);
    warpBitonicMergeLE16<K, V, 4, Dir, Comp, false>(k[0], v[0]);
    warpBitonicMergeLE16<K, V, 8, Dir, Comp, false>(k[0], v[0]);
    warpBitonicMergeLE16<K, V, 16, Dir, Comp, false>(k[0], v[0]);
  }
};

} // namespace fbgemm_gpu
