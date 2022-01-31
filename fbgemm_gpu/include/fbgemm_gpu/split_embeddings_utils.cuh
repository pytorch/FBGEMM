/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <ATen/ATen.h>
#include "fbgemm_gpu/fbgemm_cuda_utils.cuh"

// Warp bitonic K/V sorting code from @jhj
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
#ifdef __HIP_PLATFORM_HCC__
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

std::tuple<at::Tensor, at::Tensor, c10::optional<at::Tensor>>
get_unique_indices_cuda(
    at::Tensor linear_indices,
    int64_t max_indices,
    bool compute_count);

std::pair<at::Tensor, at::Tensor> lru_cache_find_uncached_cuda(
    at::Tensor unique_indices,
    at::Tensor unique_indices_length,
    int64_t max_indices,
    at::Tensor lxu_cache_state,
    int64_t time_stamp,
    at::Tensor lru_state);

/**
 * "Transpose" embedding inputs by sorting indices by their values.
 * Logically this transpose compressed sparse row (CSR) representation
 * stored in indices and offsets to compressed sparse column (CSC).
 */
std::tuple<
    at::Tensor /*linear_indices*/,
    at::Tensor /*linear_indices_sorted*/,
    at::Tensor /*infos_sorted*/,
    at::Tensor /*sorted_linear_indices_run*/,
    at::Tensor /*sorted_linear_indices_run_lengths*/,
    at::Tensor /*sorted_linear_indices_num_runs*/,
    at::Tensor /*sorted_linear_indices_cumulative_run_lengths*/>
transpose_embedding_input(
    at::Tensor hash_size_cumsum,
    int64_t total_hash_size_bits,
    at::Tensor indices,
    at::Tensor offsets,
    bool nobag = false);

// Use these functions instead of directly calling cub functions
// to reduce code size and compilation time.
// Arguments are the same as cub::DeviceRadixSort::SortPairs
#define DECL_RADIX_SORT_PAIRS_FN(KeyT, ValueT) \
  cudaError_t radix_sort_pairs(                \
      void* d_temp_storage,                    \
      size_t& temp_storage_bytes,              \
      const KeyT* d_keys_in,                   \
      KeyT* d_keys_out,                        \
      const ValueT* d_values_in,               \
      ValueT* d_values_out,                    \
      int num_items,                           \
      int begin_bit = 0,                       \
      int end_bit = sizeof(KeyT) * 8,          \
      cudaStream_t stream = 0,                 \
      bool debug_synchronous = false)

DECL_RADIX_SORT_PAIRS_FN(int64_t, float);
DECL_RADIX_SORT_PAIRS_FN(int64_t, double);
DECL_RADIX_SORT_PAIRS_FN(int64_t, int64_t);
DECL_RADIX_SORT_PAIRS_FN(int64_t, int32_t);

#undef DECL_RADIX_SORT_PAIRS_FN
