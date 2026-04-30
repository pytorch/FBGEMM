/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/core/ScalarType.h>
#include <cuda.h>
#include <gtest/gtest.h>
#include <torch/types.h> // @manual=//caffe2:torch-cpp-cpu

#include "fbgemm_gpu/utils/bitonic_sort.cuh"
#include "fbgemm_gpu/utils/kernel_launcher.cuh"

namespace fbgemm_gpu::utils {

////////////////////////////////////////////////////////////////////////////////
// Warp-wide single-register bitonic sort kernel.
//
// Each lane seeds a single K/V pair; after BitonicSort::sort, lane `l` holds
// the `l`-th element in sorted order across the warp. The kernel writes
// each lane's post-sort value to out_k / out_v for host-side verification.
//
// When launched with blockDim.x == at::cuda::warp_size(), this exercises the
// full bitonic sort network: the L=16 stage covers warpSize 32 archs, and
// the warpSize 64 (gfx9xx) path additionally runs the L=32 merge stage
// required to fully sort all 64 elements.
////////////////////////////////////////////////////////////////////////////////

template <typename K, typename V, bool Dir>
__global__ void
warp_bitonic_sort_kernel(K* out_k, V* out_v, K* seed_keys, V* seed_vals) {
  const auto lane = threadIdx.x;

  K k[1];
  V v[1];
  k[0] = seed_keys[lane];
  v[0] = seed_vals[lane];

  fbgemm_gpu::BitonicSort<K, V, Dir, fbgemm_gpu::Comparator<K>>::sort(k, v);

  out_k[lane] = k[0];
  out_v[lane] = v[0];
}

////////////////////////////////////////////////////////////////////////////////
// Helpers
////////////////////////////////////////////////////////////////////////////////

// Run one warp-wide sort and verify the result. Seeds lane `l` with
// k = (W - 1 - l), v = l. Expected post-sort (ascending, Dir=true):
//   k[l] == l and v[l] == (W - 1 - l).
// Descending (Dir=false) yields the original reversed seed, unchanged.
template <typename K, typename V, bool Dir>
void run_and_check_sort() {
  const int32_t W = at::cuda::warp_size();
  const auto device = torch::Device(torch::kCUDA, at::cuda::current_device());

  // Seed lane l with k = (W - 1 - l), v = l on host, then push to device.
  auto seed_k_cpu =
      torch::empty({W}, torch::dtype(c10::CppTypeToScalarType<K>::value));
  auto seed_v_cpu =
      torch::empty({W}, torch::dtype(c10::CppTypeToScalarType<V>::value));
  auto* seed_k_host = seed_k_cpu.template data_ptr<K>();
  auto* seed_v_host = seed_v_cpu.template data_ptr<V>();
  for (int32_t i = 0; i < W; ++i) {
    seed_k_host[i] = static_cast<K>(W - 1 - i);
    seed_v_host[i] = static_cast<V>(i);
  }
  const auto seed_k = seed_k_cpu.to(device);
  const auto seed_v = seed_v_cpu.to(device);

  auto out_k = torch::zeros(
      {W}, torch::dtype(c10::CppTypeToScalarType<K>::value).device(device));
  auto out_v = torch::zeros(
      {W}, torch::dtype(c10::CppTypeToScalarType<V>::value).device(device));

  FBGEMM_LAUNCH_KERNEL(
      (warp_bitonic_sort_kernel<K, V, Dir>),
      1,
      W,
      0,
      at::cuda::getCurrentCUDAStream(),
      out_k.template data_ptr<K>(),
      out_v.template data_ptr<V>(),
      seed_k.template data_ptr<K>(),
      seed_v.template data_ptr<V>());

  const auto out_k_cpu = out_k.cpu();
  const auto out_v_cpu = out_v.cpu();
  const auto* out_k_host = out_k_cpu.template data_ptr<K>();
  const auto* out_v_host = out_v_cpu.template data_ptr<V>();

  for (int32_t i = 0; i < W; ++i) {
    if (Dir) {
      // Ascending: k[i] == i, v[i] == W - 1 - i.
      EXPECT_EQ(out_k_host[i], static_cast<K>(i)) << "lane=" << i;
      EXPECT_EQ(out_v_host[i], static_cast<V>(W - 1 - i)) << "lane=" << i;
    } else {
      // Descending: input was already reverse-sorted, so output is unchanged.
      EXPECT_EQ(out_k_host[i], static_cast<K>(W - 1 - i)) << "lane=" << i;
      EXPECT_EQ(out_v_host[i], static_cast<V>(i)) << "lane=" << i;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// Tests
////////////////////////////////////////////////////////////////////////////////

// Covers the LRU/LFU populate instantiation
// (src/split_embeddings_cache/lru_cache_populate.cu,
// lfu_cache_populate.cu). Key/val types match the production callsites.
TEST(BitonicSortTest, int64_uint32_ascending) {
  run_and_check_sort<int64_t, uint32_t, /*Dir=*/true>();
}

TEST(BitonicSortTest, int64_uint32_descending) {
  run_and_check_sort<int64_t, uint32_t, /*Dir=*/false>();
}

// Covers the SSD cache instantiation
// (src/ssd_split_embeddings_cache/ssd_split_embeddings_cache_cuda.cu).
TEST(BitonicSortTest, int64_int64_ascending) {
  run_and_check_sort<int64_t, int64_t, /*Dir=*/true>();
}

TEST(BitonicSortTest, int64_int64_descending) {
  run_and_check_sort<int64_t, int64_t, /*Dir=*/false>();
}

} // namespace fbgemm_gpu::utils
