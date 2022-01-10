/*
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// clang-format off
#include "fbgemm_gpu/cub_namespace_prefix.cuh"
#include "cub/device/device_radix_sort.cuh"
#include "cub/device/device_run_length_encode.cuh"
#include "cub/device/device_select.cuh"
#include "fbgemm_gpu/cub_namespace_postfix.cuh"
// clang-format on

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/CUDAGeneratorImpl.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/TensorAccessor.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <limits>
#include <mutex>

#include "fbgemm_gpu/dispatch_macros.h"
#include "fbgemm_gpu/embedding_common.h"
#include "fbgemm_gpu/fbgemm_cuda_utils.cuh"
#include "fbgemm_gpu/sparse_ops_utils.h"
#include "fbgemm_gpu/split_embeddings_utils.cuh"

constexpr size_t kCacheMaxThreads = 512;

using Tensor = at::Tensor;

using namespace fbgemm_gpu;

namespace {
__forceinline__ __host__ __device__ uint32_t round_up(uint32_t a, uint32_t b) {
  return ((a + b - 1) / b) * b;
}
__host__ __device__ inline int32_t unpadded_row_size_in_bytes(
    int32_t dim,
    SparseType weight_ty) {
  if (weight_ty == SparseType::FP32) {
    return dim * 4;
  }
  if (weight_ty == SparseType::FP16) {
    return dim * 2;
  }
  if (weight_ty == SparseType::INT8) {
    return dim + 4;
  }
  if (weight_ty == SparseType::INT4) {
    return dim / 2 + 4;
  }
  if (weight_ty == SparseType::INT2) {
    return dim / 4 + 4;
  }
  return 0;
}

__host__ __device__ inline int32_t padded_row_size_in_bytes(
    int32_t dim,
    SparseType weight_ty) {
  auto r = unpadded_row_size_in_bytes(dim, weight_ty);
  return round_up(r, 16);
}
} // namespace

// // TODO: do we care about 64-bit indices? Currently we just ignore.
// __host__ DEVICE_INLINE uint32_t cache_slot(int32_t h_in, int32_t C) {
//   // MurmorHash3 32-bit mixing function.
//   uint32_t h = (uint32_t)h_in;
//   h ^= h >> 16;
//   h *= 0x85ebca6b;
//   h ^= h >> 13;
//   h *= 0xc2b2ae35;
//   h ^= h >> 16;
//   //
//   https://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
//   return ((uint64_t)h * (uint64_t)C) >> 32;
// }

__host__ DEVICE_INLINE uint32_t cache_slot(int64_t h_in, int32_t C) {
  // MurmurHash3 64-bit mixing function.
  uint64_t h = (uint64_t)h_in;
  h ^= h >> 33;
  h *= 0xff51afd7ed558ccd;
  h ^= h >> 33;
  h *= 0xc4ceb9fe1a85ec53;
  h ^= h >> 33;

  return h % (uint32_t)C;
}

int64_t host_lxu_cache_slot(int64_t h_in, int64_t C) {
  return static_cast<int64_t>(cache_slot(h_in, static_cast<int32_t>(C)));
}

constexpr int32_t kCacheLocationMissing = -1;
constexpr int64_t kCacheStateInvalid = -1;

template <typename emb_t, typename cache_t>
__global__ __launch_bounds__(kMaxThreads) void lxu_cache_flush_kernel(
    at::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> weights,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        cache_hash_size_cumsum,
    const at::PackedTensorAccessor64<int32_t, 1, at::RestrictPtrTraits>
        cache_index_table_map,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        weights_offsets,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        D_offsets,
    const at::PackedTensorAccessor32<int64_t, 2, at::RestrictPtrTraits>
        lxu_cache_state,
    at::PackedTensorAccessor64<cache_t, 2, at::RestrictPtrTraits>
        lxu_cache_weights,
    bool stochastic_rounding,
    at::PhiloxCudaState stochastic_rounding_philox_args) {
  int32_t B = lxu_cache_weights.size(0);
  int32_t b = blockIdx.x * blockDim.y + threadIdx.y;
  if (b >= B) {
    return;
  }
  int32_t slot = b % kWarpSize;
  int32_t cache_set = b / kWarpSize;
  int64_t current_idx = lxu_cache_state[cache_set][slot];
  if (current_idx != static_cast<int64_t>(kCacheStateInvalid)) {
    // evict from slot to backing storage
    int32_t t_current = cache_index_table_map[current_idx];
    int64_t idx_current = current_idx - cache_hash_size_cumsum[t_current];
    int64_t weights_offset_current = weights_offsets[t_current];
    int32_t D_start_current = D_offsets[t_current];
    int32_t D_end_current = D_offsets[t_current + 1];
    int32_t D_current = D_end_current - D_start_current;

    int32_t D_emb = D_current;
    if (std::is_same<emb_t, uint8_t>::value) {
      D_emb += kINT8QparamsBytes;
    }
    auto weight_row = WeightRow<emb_t, cache_t, at::acc_type<cache_t, true>>(
        &weights[weights_offset_current + idx_current * D_emb + 0],
        &lxu_cache_weights[b][0],
        D_current,
        nullptr);
    if (!std::is_same<emb_t, float>::value && stochastic_rounding) {
      StochasticRoundingRNGState state;
      // different for every *run* and every *thread*.
      auto stochastic_rounding_seeds =
          at::cuda::philox::unpack(stochastic_rounding_philox_args);
      stochastic_rounding_init(
          std::get<0>(stochastic_rounding_seeds) ^
              std::get<1>(stochastic_rounding_seeds),
          blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
              threadIdx.x,
          &state);
      weight_row.set_stoc_state(&state);
    }

    float2 qparams;
    if (std::is_same<emb_t, uint8_t>::value) {
      qparams =
          thrust_find_qparams<cache_t>(&lxu_cache_weights[b][0], D_current);
      if (threadIdx.x == 0) {
        weight_row.store_qparams(qparams);
      }
    }
    for (int32_t d = threadIdx.x; d * 4 < D_current; d += blockDim.x) {
      Vec4T<at::acc_type<cache_t, true>> cache_weights_vec =
          weight_row.load(d * 4, qparams);
      weight_row.evict(cache_weights_vec, d * 4, qparams);
    }
  }
}

void lxu_cache_flush_cuda(
    Tensor uvm_weights,
    Tensor cache_hash_size_cumsum,
    Tensor cache_index_table_map,
    Tensor weights_offsets,
    Tensor D_offsets,
    int64_t total_D,
    Tensor lxu_cache_state,
    Tensor lxu_cache_weights,
    bool stochastic_rounding) {
  TENSOR_ON_CUDA_GPU(uvm_weights);
  TENSOR_ON_CUDA_GPU(cache_hash_size_cumsum);
  TENSOR_ON_CUDA_GPU(cache_index_table_map);
  TENSOR_ON_CUDA_GPU(weights_offsets);
  TENSOR_ON_CUDA_GPU(D_offsets);
  TENSOR_ON_CUDA_GPU(lxu_cache_state);
  TENSOR_ON_CUDA_GPU(lxu_cache_weights);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(lxu_cache_weights.get_device());

  int32_t T = D_offsets.numel() - 1;
  int32_t S = lxu_cache_weights.size(0);
  int32_t tx = std::min<int32_t>(total_D / 4 / T, kMaxThreads);
  dim3 threads(tx, kMaxThreads / tx);
  dim3 blocks(div_round_up(S, kMaxThreads / tx));

  DISPATCH_EMB_CACHE_TYPES(
      uvm_weights.type(),
      lxu_cache_weights.type(),
      "lxu_cache_flush_kernel_2",
      ([&] {
        at::PhiloxCudaState rng_engine_inputs;
        if (stochastic_rounding && std::is_same<emb_t, at::Half>::value) {
          auto gen = at::cuda::detail::getDefaultCUDAGenerator();
          std::lock_guard<std::mutex> lock(gen.mutex());
          rng_engine_inputs = at::check_generator<at::CUDAGeneratorImpl>(gen)
                                  ->philox_cuda_state(4);
        }
        lxu_cache_flush_kernel<emb_t, cache_t>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                uvm_weights
                    .packed_accessor64<emb_t, 1, at::RestrictPtrTraits>(),
                cache_hash_size_cumsum
                    .packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                cache_index_table_map
                    .packed_accessor64<int32_t, 1, at::RestrictPtrTraits>(),
                weights_offsets
                    .packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                D_offsets
                    .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                lxu_cache_state
                    .packed_accessor32<int64_t, 2, at::RestrictPtrTraits>(),
                lxu_cache_weights
                    .packed_accessor64<cache_t, 2, at::RestrictPtrTraits>(),
                stochastic_rounding,
                rng_engine_inputs);
      }));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return;
}

template <typename index_t>
__global__ __launch_bounds__(kMaxThreads) void linearize_cache_indices_kernel(
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        cache_hash_size_cumsum,
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> indices,
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> offsets,
    at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        linear_cache_indices) {
  int32_t T = cache_hash_size_cumsum.size(0) - 1;
  int64_t total_cache_hash_size = cache_hash_size_cumsum[T];
  int32_t B = (offsets.size(0) - 1) / T;
  int32_t b_t = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t b = b_t % B;
  int32_t t = b_t / B;
  bool valid = t < T;

  int64_t hash_offset = valid ? cache_hash_size_cumsum[t] : -1;
  auto indices_start = valid ? offsets[t * B + b] : -1;
  int32_t L = valid ? offsets[t * B + b + 1] - indices_start : 0;
  int32_t lane_id = threadIdx.x % kWarpSize;

  // hash_offset < 0 for non-caching tables
  for (int32_t j = 0; j < kWarpSize; ++j) {
    auto indices_start_warp = shfl_sync(indices_start, j);
    int32_t L_warp = shfl_sync(L, j);
    int64_t hash_offset_warp = shfl_sync(hash_offset, j);
    if (hash_offset_warp >= 0) {
      for (int32_t i = lane_id; i < L_warp; i += kWarpSize) {
        auto idx = __ldg(&indices[indices_start_warp + i]);
        linear_cache_indices[indices_start_warp + i] = hash_offset_warp + idx;
      }
    } else {
      for (int32_t i = lane_id; i < L_warp; i += kWarpSize) {
        linear_cache_indices[indices_start_warp + i] = total_cache_hash_size;
      }
    }
  }
}

Tensor linearize_cache_indices_cuda(
    Tensor cache_hash_size_cumsum,
    Tensor indices,
    Tensor offsets) {
  TENSOR_ON_CUDA_GPU(cache_hash_size_cumsum);
  TENSOR_ON_CUDA_GPU(indices);
  TENSOR_ON_CUDA_GPU(offsets);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(cache_hash_size_cumsum.get_device());

  auto T = cache_hash_size_cumsum.size(0) - 1;
  TORCH_CHECK(T > 0);
  // offsets = [B x T  + 1]
  auto B = (offsets.size(0) - 1) / T;
  TORCH_CHECK(B >= 0);

  auto linear_cache_indices = at::empty_like(indices);
  if (B == 0) {
    return linear_cache_indices;
  }
  AT_DISPATCH_INDEX_TYPES(
      indices.scalar_type(), "linearize_cache_indices_kernel", [&]() {
        linearize_cache_indices_kernel<<<
            div_round_up(B * T, kMaxThreads),
            kMaxThreads,
            0,
            at::cuda::getCurrentCUDAStream()>>>(
            cache_hash_size_cumsum
                .packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
            indices.packed_accessor32<index_t, 1, at::RestrictPtrTraits>(),
            offsets.packed_accessor32<index_t, 1, at::RestrictPtrTraits>(),
            linear_cache_indices
                .packed_accessor32<index_t, 1, at::RestrictPtrTraits>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return linear_cache_indices;
}

std::tuple<Tensor, Tensor, c10::optional<Tensor>> get_unique_indices_cuda(
    Tensor linear_indices,
    int64_t max_indices,
    bool compute_count) {
  TENSOR_ON_CUDA_GPU(linear_indices);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(linear_indices.get_device());

  TORCH_CHECK(linear_indices.numel() < std::numeric_limits<int32_t>::max());
  int32_t N = linear_indices.numel();
  auto sorted_indices = at::empty_like(linear_indices);
  auto unique_indices = at::empty_like(linear_indices);
  auto unique_indices_length =
      at::empty({1}, linear_indices.options().dtype(at::kInt));
  c10::optional<Tensor> unique_indices_count = c10::nullopt;
  if (compute_count) {
    unique_indices_count = at::empty(
        {linear_indices.numel()}, linear_indices.options().dtype(at::kInt));
  }
  AT_DISPATCH_INDEX_TYPES(
      linear_indices.scalar_type(), "get_unique_indices_cuda", [&]() {
        // sort indices
        size_t temp_storage_bytes_0 = 0;
        AT_CUDA_CHECK(FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceRadixSort::SortKeys(
            nullptr,
            temp_storage_bytes_0,
            linear_indices.data_ptr<index_t>(),
            sorted_indices.data_ptr<index_t>(),
            N,
            0,
            int(log2(float(max_indices + 1)) + 1),
            at::cuda::getCurrentCUDAStream(),
            false));
        auto temp_storage_0 = at::empty(
            {static_cast<index_t>(temp_storage_bytes_0)},
            linear_indices.options().dtype(at::kByte));
        AT_CUDA_CHECK(FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceRadixSort::SortKeys(
            temp_storage_0.data_ptr(),
            temp_storage_bytes_0,
            linear_indices.data_ptr<index_t>(),
            sorted_indices.data_ptr<index_t>(),
            N,
            0,
            int(log2(float(max_indices + 1)) + 1),
            at::cuda::getCurrentCUDAStream(),
            false));
        // get unique indices
        if (compute_count) {
          size_t temp_storage_bytes_1 = 0;
          AT_CUDA_CHECK(
              FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceRunLengthEncode::Encode(
                  nullptr,
                  temp_storage_bytes_1,
                  sorted_indices.data_ptr<index_t>(),
                  unique_indices.data_ptr<index_t>(),
                  unique_indices_count->data_ptr<int32_t>(),
                  unique_indices_length.data_ptr<int32_t>(),
                  N,
                  at::cuda::getCurrentCUDAStream(),
                  false));
          auto temp_storage_1 = at::empty(
              {static_cast<index_t>(temp_storage_bytes_1)},
              linear_indices.options().dtype(at::kByte));
          AT_CUDA_CHECK(
              FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceRunLengthEncode::Encode(
                  temp_storage_1.data_ptr(),
                  temp_storage_bytes_1,
                  sorted_indices.data_ptr<index_t>(),
                  unique_indices.data_ptr<index_t>(),
                  unique_indices_count->data_ptr<int32_t>(),
                  unique_indices_length.data_ptr<int32_t>(),
                  N,
                  at::cuda::getCurrentCUDAStream(),
                  false));
        } else {
          size_t temp_storage_bytes_1 = 0;
          AT_CUDA_CHECK(FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceSelect::Unique(
              nullptr,
              temp_storage_bytes_1,
              sorted_indices.data_ptr<index_t>(),
              unique_indices.data_ptr<index_t>(),
              unique_indices_length.data_ptr<int32_t>(),
              N,
              at::cuda::getCurrentCUDAStream(),
              false));
          auto temp_storage_1 = at::empty(
              {static_cast<index_t>(temp_storage_bytes_1)},
              linear_indices.options().dtype(at::kByte));
          AT_CUDA_CHECK(FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceSelect::Unique(
              temp_storage_1.data_ptr(),
              temp_storage_bytes_1,
              sorted_indices.data_ptr<index_t>(),
              unique_indices.data_ptr<index_t>(),
              unique_indices_length.data_ptr<int32_t>(),
              N,
              at::cuda::getCurrentCUDAStream(),
              false));
        }
      });
  return std::make_tuple(
      unique_indices, unique_indices_length, unique_indices_count);
}

template <typename index_t>
__global__ __launch_bounds__(kMaxThreads) void lru_cache_find_uncached_kernel(
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        unique_indices,
    const int32_t* __restrict__ N_unique,
    int64_t max_indices,
    const at::PackedTensorAccessor32<int64_t, 2, at::RestrictPtrTraits>
        lxu_cache_state,
    at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> cache_sets,
    int64_t time_stamp,
    at::PackedTensorAccessor32<int64_t, 2, at::RestrictPtrTraits> lru_state) {
  int32_t N = unique_indices.size(0);
  int32_t C = lxu_cache_state.size(0);

  int32_t n = blockIdx.x * blockDim.y + threadIdx.y;
  if (n >= N) {
    return;
  }
  if (n >= *N_unique) {
    if (threadIdx.x == 0) {
      cache_sets[n] = C; // invalid index, used as sentinel
    }
    return;
  }
  int64_t idx = unique_indices[n];
  if (idx == max_indices) {
    if (threadIdx.x == 0) {
      cache_sets[n] = C; // invalid index, used as sentinel
    }
    return;
  }
  int32_t cache_set = cache_slot(idx, C);

  auto slot = threadIdx.x;
  bool found = __ldg((&lxu_cache_state[cache_set][0]) + slot) == idx;
  if (found) {
    // mark it as existing.
    cache_sets[n] = C; // invalid index, used as sentinel
    // mark it as recently accessed so we don't evict.
    lru_state[cache_set][slot] = time_stamp;
  }

#ifdef __HIP_PLATFORM_HCC__
  // FIXME: __any_sync with mask isn't supported by HIP yet.
  // See https://fburl.com/fvy7j0lq for the similar context.
  // assert false here with https://fburl.com/pfm7enw2
  assert(false);
  if (!__any(found)) {
#else
  if (!__any_sync(0xFFFFFFFF, found)) {
#endif
    if (threadIdx.x == 0) {
      cache_sets[n] = cache_set;
    }
  }
}

std::pair<Tensor, Tensor> lru_cache_find_uncached_cuda(
    Tensor unique_indices,
    Tensor unique_indices_length,
    int64_t max_indices,
    Tensor lxu_cache_state,
    int64_t time_stamp,
    Tensor lru_state) {
  TENSOR_ON_CUDA_GPU(unique_indices);
  TENSOR_ON_CUDA_GPU(unique_indices_length);
  TENSOR_ON_CUDA_GPU(lxu_cache_state);
  TENSOR_ON_CUDA_GPU(lru_state);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(unique_indices.get_device());

  auto cache_sets =
      empty_like(unique_indices, unique_indices.options().dtype(at::kInt));
  int32_t N = unique_indices.numel();
  auto sorted_cache_sets = empty_like(cache_sets);
  auto cache_set_sorted_unique_indices = empty_like(unique_indices);

  AT_DISPATCH_INDEX_TYPES(
      unique_indices.scalar_type(), "lru_cache_find_uncached_cuda", [&]() {
        // Find uncached indices
        lru_cache_find_uncached_kernel<<<
            div_round_up(N, kMaxThreads / kWarpSize),
            dim3(kWarpSize, kMaxThreads / kWarpSize),
            0,
            at::cuda::getCurrentCUDAStream()>>>(
            unique_indices
                .packed_accessor32<index_t, 1, at::RestrictPtrTraits>(),
            unique_indices_length.data_ptr<int32_t>(),
            max_indices,
            lxu_cache_state
                .packed_accessor32<int64_t, 2, at::RestrictPtrTraits>(),
            cache_sets.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
            time_stamp,
            lru_state.packed_accessor32<int64_t, 2, at::RestrictPtrTraits>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        // Sort the cache sets and ids
        size_t temp_storage_bytes = 0;
        AT_CUDA_CHECK(FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceRadixSort::SortPairs(
            nullptr,
            temp_storage_bytes,
            cache_sets.data_ptr<int32_t>(),
            sorted_cache_sets.data_ptr<int32_t>(),
            unique_indices.data_ptr<index_t>(),
            cache_set_sorted_unique_indices.data_ptr<index_t>(),
            N,
            0,
            int(log2(float(lxu_cache_state.size(0) + 1)) + 1),
            at::cuda::getCurrentCUDAStream(),
            false));
        auto temp_storage = at::empty(
            {static_cast<index_t>(temp_storage_bytes)},
            unique_indices.options().dtype(at::kByte));
        AT_CUDA_CHECK(FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceRadixSort::SortPairs(
            temp_storage.data_ptr(),
            temp_storage_bytes,
            cache_sets.data_ptr<int32_t>(),
            sorted_cache_sets.data_ptr<int32_t>(),
            unique_indices.data_ptr<index_t>(),
            cache_set_sorted_unique_indices.data_ptr<index_t>(),
            N,
            0,
            int(log2(float(lxu_cache_state.size(0) + 1)) + 1),
            at::cuda::getCurrentCUDAStream(),
            false));
      });
  return {sorted_cache_sets, cache_set_sorted_unique_indices};
}

template <typename emb_t, typename cache_t>
__global__ __launch_bounds__(kMaxThreads) void lru_cache_insert_kernel(
    at::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> weights,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        cache_hash_size_cumsum,
    const at::PackedTensorAccessor64<int32_t, 1, at::RestrictPtrTraits>
        cache_index_table_map,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        weights_offsets,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        D_offsets,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        sorted_cache_sets,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        cache_set_sorted_indices,
    const int32_t* __restrict__ N_unique,
    at::PackedTensorAccessor32<int64_t, 2, at::RestrictPtrTraits>
        lxu_cache_state,
    at::PackedTensorAccessor64<cache_t, 2, at::RestrictPtrTraits>
        lxu_cache_weights,
    int64_t time_stamp,
    at::PackedTensorAccessor32<int64_t, 2, at::RestrictPtrTraits> lru_state,
    bool stochastic_rounding,
    at::PhiloxCudaState stochastic_rounding_philox_args) {
  int32_t C = lxu_cache_state.size(0);
  int32_t n = blockIdx.x * blockDim.y + threadIdx.y;
  if (n >= *N_unique) {
    return;
  }
  // check if this warp is responsible for this whole segment.
  bool segment_start =
      (n == 0 || sorted_cache_sets[n - 1] != sorted_cache_sets[n]);

  if (!segment_start) {
    // don't have *warp* divergence since we launch full warps in blockDim.x,
    // so we can just exit this warp entirely.
    return;
  }
  int32_t cache_set = sorted_cache_sets[n];
  if (cache_set == C) {
    // ignore the already-existing elements
    return;
  }

  int32_t SL = 1;
  while (n + SL < *N_unique && sorted_cache_sets[n + SL] == cache_set) {
    SL += 1;
  }

  // now, we need to insert the (unique!) values in indices[n:n + SL] into
  // our slots.
  int32_t slot = threadIdx.x;
  int64_t slot_time = lru_state[cache_set][slot];
  int64_t costs[1] = {slot_time};
  int32_t slots[1] = {slot};

  BitonicSort<int64_t, int32_t, 1, Comparator<int64_t>>::sort(costs, slots);
  int32_t sorted_slot = slots[0];
  int64_t sorted_lru_cost = costs[0];

  for (int32_t l = 0; l < min(SL, kWarpSize); ++l) {
    int32_t insert_slot = shfl_sync(sorted_slot, l);
    int64_t insert_current_lru_cost = shfl_sync(sorted_lru_cost, l);
    if (insert_current_lru_cost == time_stamp) {
      return;
    }
    int64_t insert_idx = cache_set_sorted_indices[n + l];
    int32_t t_insert = cache_index_table_map[insert_idx];
    int64_t idx_insert = insert_idx - cache_hash_size_cumsum[t_insert];
    int64_t weights_offset_insert = weights_offsets[t_insert];
    int32_t D_start_insert = D_offsets[t_insert];
    int32_t D_end_insert = D_offsets[t_insert + 1];
    int32_t D_insert = D_end_insert - D_start_insert;

    // ensure that threadIdx.x is the only thread reading/writing to
    // lxu_cache_state
    int64_t current_idx =
        threadIdx.x == 0 ? lxu_cache_state[cache_set][insert_slot] : 0;
    current_idx = shfl_sync(current_idx, 0);

    // not empty
    if (current_idx != static_cast<int64_t>(kCacheStateInvalid)) {
      // evict from slot to backing storage
      int32_t t_current = cache_index_table_map[current_idx];
      int64_t idx_current = current_idx - cache_hash_size_cumsum[t_current];
      int64_t weights_offset_current = weights_offsets[t_current];
      int32_t D_start_current = D_offsets[t_current];
      int32_t D_end_current = D_offsets[t_current + 1];
      int32_t D_current = D_end_current - D_start_current;
      int32_t D_emb = D_current;
      if (std::is_same<emb_t, uint8_t>::value) {
        D_emb += kINT8QparamsBytes;
      }
      auto weight_row = WeightRow<emb_t, cache_t, cache_t>(
          &weights[weights_offset_current + idx_current * D_emb + 0],
          &lxu_cache_weights[cache_set * kWarpSize + insert_slot][0],
          D_current,
          nullptr);
      if (!std::is_same<emb_t, float>::value && stochastic_rounding) {
        StochasticRoundingRNGState state;
        // different for every *run* and every *thread*.
        auto stochastic_rounding_seeds =
            at::cuda::philox::unpack(stochastic_rounding_philox_args);
        stochastic_rounding_init(
            std::get<0>(stochastic_rounding_seeds) ^
                std::get<1>(stochastic_rounding_seeds),
            (blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
             threadIdx.x) *
                    kWarpSize +
                l,
            &state);
        weight_row.set_stoc_state(&state);
      }
      float2 qparams;
      at::acc_type<cache_t, true> local_min =
          std::numeric_limits<at::acc_type<cache_t, true>>::max();
      at::acc_type<cache_t, true> local_max =
          std::numeric_limits<at::acc_type<cache_t, true>>::lowest();
      if (std::is_same<emb_t, uint8_t>::value) {
        for (int32_t d = threadIdx.x; d * 4 < D_current; d += blockDim.x) {
          Vec4T<cache_t> cache_weights_vec =
              weight_row.load(d * 4, qparams); // qparams not used
          local_max = max(local_max, vec4_max(cache_weights_vec));
          local_min = min(local_min, vec4_min(cache_weights_vec));
        }
        qparams = warp_find_qparams(local_min, local_max);
        if (threadIdx.x == 0) {
          weight_row.store_qparams(qparams);
        }
      }
      for (int32_t d = threadIdx.x; d * 4 < D_current; d += blockDim.x) {
        Vec4T<cache_t> cache_weights_vec = weight_row.load(d * 4, qparams);
        weight_row.evict(
            cache_weights_vec, d * 4, qparams); // FP32 -> FP16/FP32
      }
    }
    int32_t D_emb = D_insert;
    if (std::is_same<emb_t, uint8_t>::value) {
      D_emb += kINT8QparamsBytes;
    }
    // insert into cache
    auto weight_row_cache = WeightRow<emb_t, cache_t, cache_t>(
        &weights[weights_offset_insert + idx_insert * D_emb + 0],
        &lxu_cache_weights[cache_set * kWarpSize + insert_slot][0],
        D_insert,
        nullptr);

    auto weight_row_emb = WeightRow<emb_t, cache_t, cache_t>(
        &weights[weights_offset_insert + idx_insert * D_emb + 0],
        nullptr,
        D_insert,
        nullptr);

    float2 qparams;
    if (std::is_same<emb_t, uint8_t>::value) {
      qparams = weight_row_emb.load_qparams();
    }
    for (int32_t d = threadIdx.x; d * 4 < D_insert; d += blockDim.x) {
      auto row = weight_row_emb.load(d * 4, qparams);
      weight_row_cache.store(row, d * 4, qparams);
    }
    if (threadIdx.x == 0) {
      lxu_cache_state[cache_set][insert_slot] = insert_idx;
      lru_state[cache_set][insert_slot] = time_stamp;
    }
  }
}

void lru_cache_insert_cuda(
    Tensor weights,
    Tensor cache_hash_size_cumsum,
    Tensor cache_index_table_map,
    Tensor weights_offsets,
    Tensor D_offsets,
    Tensor sorted_cache_sets,
    Tensor cache_set_sorted_unique_indices,
    Tensor unique_indices_length,
    Tensor lxu_cache_state,
    Tensor lxu_cache_weights,
    int64_t time_stamp,
    Tensor lru_state,
    bool stochastic_rounding) {
  TENSOR_ON_CUDA_GPU(weights);
  TENSOR_ON_CUDA_GPU(cache_hash_size_cumsum);
  TENSOR_ON_CUDA_GPU(cache_index_table_map);
  TENSOR_ON_CUDA_GPU(weights_offsets);
  TENSOR_ON_CUDA_GPU(D_offsets);
  TENSOR_ON_CUDA_GPU(sorted_cache_sets);
  TENSOR_ON_CUDA_GPU(cache_set_sorted_unique_indices);
  TENSOR_ON_CUDA_GPU(unique_indices_length);
  TENSOR_ON_CUDA_GPU(lxu_cache_state);
  TENSOR_ON_CUDA_GPU(lxu_cache_weights);
  TENSOR_ON_CUDA_GPU(lru_state);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(weights.get_device());

  int32_t N = cache_set_sorted_unique_indices.numel();

  DISPATCH_EMB_CACHE_TYPES(
      weights.type(),
      lxu_cache_weights.type(),
      "lru_cache_insert_kernel_2",
      ([&] {
        at::PhiloxCudaState rng_engine_inputs;
        if (stochastic_rounding && !std::is_same<emb_t, float>::value) {
          auto gen = at::cuda::detail::getDefaultCUDAGenerator();
          std::lock_guard<std::mutex> lock(gen.mutex());
          rng_engine_inputs = at::check_generator<at::CUDAGeneratorImpl>(gen)
                                  ->philox_cuda_state(4);
        }

        lru_cache_insert_kernel<emb_t, cache_t>
            <<<div_round_up(N, kMaxThreads / kWarpSize),
               dim3(kWarpSize, kMaxThreads / kWarpSize),
               0,
               at::cuda::getCurrentCUDAStream()>>>(
                weights.packed_accessor64<emb_t, 1, at::RestrictPtrTraits>(),
                cache_hash_size_cumsum
                    .packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                cache_index_table_map
                    .packed_accessor64<int32_t, 1, at::RestrictPtrTraits>(),
                weights_offsets
                    .packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                D_offsets
                    .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                sorted_cache_sets
                    .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                cache_set_sorted_unique_indices
                    .packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                unique_indices_length.data_ptr<int32_t>(),
                lxu_cache_state
                    .packed_accessor32<int64_t, 2, at::RestrictPtrTraits>(),
                lxu_cache_weights
                    .packed_accessor64<cache_t, 2, at::RestrictPtrTraits>(),
                time_stamp,
                lru_state
                    .packed_accessor32<int64_t, 2, at::RestrictPtrTraits>(),
                stochastic_rounding,
                rng_engine_inputs);
      }));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void lru_cache_populate_cuda(
    Tensor weights,
    Tensor cache_hash_size_cumsum,
    int64_t total_cache_hash_size,
    Tensor cache_index_table_map,
    Tensor weights_offsets,
    Tensor D_offsets,
    Tensor linear_cache_indices,
    Tensor lxu_cache_state,
    Tensor lxu_cache_weights,
    int64_t time_stamp,
    Tensor lru_state,
    bool stochastic_rounding) {
  TENSOR_ON_CUDA_GPU(weights);
  TENSOR_ON_CUDA_GPU(cache_hash_size_cumsum);
  TENSOR_ON_CUDA_GPU(cache_index_table_map);
  TENSOR_ON_CUDA_GPU(weights_offsets);
  TENSOR_ON_CUDA_GPU(D_offsets);
  TENSOR_ON_CUDA_GPU(linear_cache_indices);
  TENSOR_ON_CUDA_GPU(lxu_cache_state);
  TENSOR_ON_CUDA_GPU(lxu_cache_weights);
  TENSOR_ON_CUDA_GPU(lru_state);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(weights.get_device());

  TORCH_CHECK(
      linear_cache_indices.numel() < std::numeric_limits<int32_t>::max());
  if (linear_cache_indices.numel() == 0) {
    // nothing to do
    return;
  }

  // Get unqiue indices
  Tensor unique_indices;
  Tensor unique_indices_length;
  c10::optional<Tensor> unique_indices_count;
  std::tie(unique_indices, unique_indices_length, unique_indices_count) =
      get_unique_indices_cuda(
          linear_cache_indices, total_cache_hash_size, false);

  // Find uncached indices
  auto cache_sets_and_unique_indices = lru_cache_find_uncached_cuda(
      unique_indices,
      unique_indices_length,
      total_cache_hash_size,
      lxu_cache_state,
      time_stamp,
      lru_state);
  auto sorted_cache_sets = cache_sets_and_unique_indices.first;
  auto cache_set_sorted_unique_indices = cache_sets_and_unique_indices.second;

  // insert caching weights
  lru_cache_insert_cuda(
      weights,
      cache_hash_size_cumsum,
      cache_index_table_map,
      weights_offsets,
      D_offsets,
      sorted_cache_sets,
      cache_set_sorted_unique_indices,
      unique_indices_length,
      lxu_cache_state,
      lxu_cache_weights,
      time_stamp,
      lru_state,
      stochastic_rounding);
}

template <typename index_t>
__global__ __launch_bounds__(kMaxThreads) void lru_cache_insert_byte_kernel(
    at::PackedTensorAccessor64<uint8_t, 1, at::RestrictPtrTraits> weights,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        cache_hash_size_cumsum,
    const at::PackedTensorAccessor64<int32_t, 1, at::RestrictPtrTraits>
        cache_index_table_map,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        weights_offsets,
    const at::PackedTensorAccessor32<uint8_t, 1, at::RestrictPtrTraits>
        weights_tys,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        D_offsets,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        sorted_cache_sets,
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        cache_set_sorted_indices,
    const int32_t* __restrict__ N_unique,
    at::PackedTensorAccessor32<int64_t, 2, at::RestrictPtrTraits>
        lxu_cache_state,
    at::PackedTensorAccessor64<uint8_t, 2, at::RestrictPtrTraits>
        lxu_cache_weights,
    int64_t time_stamp,
    at::PackedTensorAccessor32<int64_t, 2, at::RestrictPtrTraits> lru_state) {
  int32_t C = lxu_cache_state.size(0);
  int32_t n = blockIdx.x * blockDim.y + threadIdx.y;
  if (n >= *N_unique) {
    return;
  }
  // check if this warp is responsible for this whole segment.
  bool segment_start =
      (n == 0 || sorted_cache_sets[n - 1] != sorted_cache_sets[n]);

  if (!segment_start) {
    // don't have *warp* divergence since we launch full warps in blockDim.x,
    // so we can just exit this warp entirely.
    return;
  }
  int32_t cache_set = sorted_cache_sets[n];
  if (cache_set == C) {
    // ignore the already-existing elements
    return;
  }

  int32_t SL = 1;
  while (n + SL < *N_unique && sorted_cache_sets[n + SL] == cache_set) {
    SL += 1;
  }

  // now, we need to insert the (unique!) values in indices[n:n + SL] into
  // our slots.
  int32_t slot = threadIdx.x;
  int64_t slot_time = lru_state[cache_set][slot];
  int64_t costs[1] = {slot_time};
  int32_t slots[1] = {slot};

  BitonicSort<int64_t, int32_t, 1, Comparator<int64_t>>::sort(costs, slots);
  int32_t sorted_slot = slots[0];
  int64_t sorted_lru_cost = costs[0];

  for (int32_t l = 0; l < min(SL, kWarpSize); ++l) {
    int32_t insert_slot = shfl_sync(sorted_slot, l);
    int64_t insert_current_lru_cost = shfl_sync(sorted_lru_cost, l);
    if (insert_current_lru_cost == time_stamp) {
      return;
    }
    index_t insert_idx = cache_set_sorted_indices[n + l];
    int32_t t_insert = cache_index_table_map[insert_idx];
    SparseType weight_ty_insert =
        static_cast<SparseType>(weights_tys[t_insert]);
    int64_t idx_insert = insert_idx - cache_hash_size_cumsum[t_insert];
    int64_t weights_offset_insert = weights_offsets[t_insert];
    int32_t D_start_insert = D_offsets[t_insert];
    int32_t D_end_insert = D_offsets[t_insert + 1];
    int32_t D_insert = D_end_insert - D_start_insert;

    const int32_t D_insert_bytes =
        padded_row_size_in_bytes(D_insert, weight_ty_insert);

    // ensure that threadIdx.x is the only thread reading/writing to
    // lxu_cache_state
    int64_t current_idx =
        threadIdx.x == 0 ? lxu_cache_state[cache_set][insert_slot] : 0;
    current_idx = shfl_sync(current_idx, 0);

    // not empty
    if (current_idx != static_cast<int64_t>(kCacheStateInvalid)) {
      // evict from slot to backing storage
      int32_t t_current = cache_index_table_map[current_idx];
      SparseType weight_ty_current =
          static_cast<SparseType>(weights_tys[t_current]);
      int64_t idx_current = current_idx - cache_hash_size_cumsum[t_current];
      int64_t weights_offset_current = weights_offsets[t_current];
      int32_t D_start_current = D_offsets[t_current];
      int32_t D_end_current = D_offsets[t_current + 1];
      int32_t D_current = D_end_current - D_start_current;

      const int32_t D_current_bytes =
          padded_row_size_in_bytes(D_current, weight_ty_current);

      auto row =
          &weights[weights_offset_current + idx_current * D_current_bytes + 0];
      auto cache_row =
          &lxu_cache_weights[cache_set * kWarpSize + insert_slot][0];
      // Evict the cache
      for (int32_t d = threadIdx.x; d < D_current_bytes; d += blockDim.x) {
        row[d] = cache_row[d]; // uint8_t access
      }
    }
    auto row =
        &weights[weights_offset_insert + idx_insert * D_insert_bytes + 0];
    auto cache_row = &lxu_cache_weights[cache_set * kWarpSize + insert_slot][0];
    for (int32_t d = threadIdx.x; d < D_insert_bytes; d += blockDim.x) {
      cache_row[d] = row[d];
    }

    if (threadIdx.x == 0) {
      lxu_cache_state[cache_set][insert_slot] = insert_idx;
      lru_state[cache_set][insert_slot] = time_stamp;
    }
  }
}

void lru_cache_insert_byte_cuda(
    Tensor weights,
    Tensor cache_hash_size_cumsum,
    Tensor cache_index_table_map,
    Tensor weights_offsets,
    Tensor weights_tys,
    Tensor D_offsets,
    Tensor sorted_cache_sets,
    Tensor cache_set_sorted_unique_indices,
    Tensor unique_indices_length,
    Tensor lxu_cache_state,
    Tensor lxu_cache_weights,
    int64_t time_stamp,
    Tensor lru_state) {
  TENSOR_ON_CUDA_GPU(weights);
  TENSOR_ON_CUDA_GPU(cache_hash_size_cumsum);
  TENSOR_ON_CUDA_GPU(cache_index_table_map);
  TENSOR_ON_CUDA_GPU(weights_offsets);
  TENSOR_ON_CUDA_GPU(weights_tys);
  TENSOR_ON_CUDA_GPU(D_offsets);
  TENSOR_ON_CUDA_GPU(sorted_cache_sets);
  TENSOR_ON_CUDA_GPU(cache_set_sorted_unique_indices);
  TENSOR_ON_CUDA_GPU(unique_indices_length);
  TENSOR_ON_CUDA_GPU(lxu_cache_state);
  TENSOR_ON_CUDA_GPU(lxu_cache_weights);
  TENSOR_ON_CUDA_GPU(lru_state);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(weights.get_device());

  int32_t N = cache_set_sorted_unique_indices.numel();

  AT_DISPATCH_INDEX_TYPES(
      cache_set_sorted_unique_indices.scalar_type(),
      "lru_cache_insert_byte_cuda",
      [&]() {
        lru_cache_insert_byte_kernel<<<
            div_round_up(N, kMaxThreads / kWarpSize),
            dim3(kWarpSize, kMaxThreads / kWarpSize),
            0,
            at::cuda::getCurrentCUDAStream()>>>(
            weights.packed_accessor64<uint8_t, 1, at::RestrictPtrTraits>(),
            cache_hash_size_cumsum
                .packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
            cache_index_table_map
                .packed_accessor64<int32_t, 1, at::RestrictPtrTraits>(),
            weights_offsets
                .packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
            weights_tys.packed_accessor32<uint8_t, 1, at::RestrictPtrTraits>(),
            D_offsets.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
            sorted_cache_sets
                .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
            cache_set_sorted_unique_indices
                .packed_accessor32<index_t, 1, at::RestrictPtrTraits>(),
            unique_indices_length.data_ptr<int32_t>(),
            lxu_cache_state
                .packed_accessor32<int64_t, 2, at::RestrictPtrTraits>(),
            lxu_cache_weights
                .packed_accessor64<uint8_t, 2, at::RestrictPtrTraits>(),
            time_stamp,
            lru_state.packed_accessor32<int64_t, 2, at::RestrictPtrTraits>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
}

void lru_cache_populate_byte_cuda(
    Tensor weights,
    Tensor cache_hash_size_cumsum,
    int64_t total_cache_hash_size,
    Tensor cache_index_table_map,
    Tensor weights_offsets,
    Tensor weights_tys,
    Tensor D_offsets,
    Tensor linear_cache_indices,
    Tensor lxu_cache_state,
    Tensor lxu_cache_weights,
    int64_t time_stamp,
    Tensor lru_state) {
  TENSOR_ON_CUDA_GPU(weights);
  TENSOR_ON_CUDA_GPU(cache_hash_size_cumsum);
  TENSOR_ON_CUDA_GPU(cache_index_table_map);
  TENSOR_ON_CUDA_GPU(weights_offsets);
  TENSOR_ON_CUDA_GPU(weights_tys);
  TENSOR_ON_CUDA_GPU(D_offsets);
  TENSOR_ON_CUDA_GPU(linear_cache_indices);
  TENSOR_ON_CUDA_GPU(lxu_cache_state);
  TENSOR_ON_CUDA_GPU(lxu_cache_weights);
  TENSOR_ON_CUDA_GPU(lru_state);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(weights.get_device());

  TORCH_CHECK(
      linear_cache_indices.numel() < std::numeric_limits<int32_t>::max());
  if (linear_cache_indices.numel() == 0) {
    // nothing to do
    return;
  }

  // Get unqiue indices
  Tensor unique_indices;
  Tensor unique_indices_length;
  c10::optional<Tensor> unique_indices_count;
  std::tie(unique_indices, unique_indices_length, unique_indices_count) =
      get_unique_indices_cuda(
          linear_cache_indices, total_cache_hash_size, false);

  // Find uncached indices
  auto cache_sets_and_unique_indices = lru_cache_find_uncached_cuda(
      unique_indices,
      unique_indices_length,
      total_cache_hash_size,
      lxu_cache_state,
      time_stamp,
      lru_state);
  auto sorted_cache_sets = cache_sets_and_unique_indices.first;
  auto cache_set_sorted_unique_indices = cache_sets_and_unique_indices.second;

  // insert caching weights
  lru_cache_insert_byte_cuda(
      weights,
      cache_hash_size_cumsum,
      cache_index_table_map,
      weights_offsets,
      weights_tys,
      D_offsets,
      sorted_cache_sets,
      cache_set_sorted_unique_indices,
      unique_indices_length,
      lxu_cache_state,
      lxu_cache_weights,
      time_stamp,
      lru_state);
}

template <typename index_t>
__global__ __launch_bounds__(kMaxThreads) void lfu_update_counts_kernel(
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        unique_indices,
    const int32_t* __restrict__ N_unique,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        unique_indices_count,
    at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> lfu_state) {
  int32_t n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= *N_unique) {
    return;
  }
  auto idx = unique_indices[n];
  lfu_state[idx] += unique_indices_count[n];
}

void lfu_update_counts_cuda(
    Tensor unique_indices,
    Tensor unique_indices_length,
    Tensor unique_indices_count,
    Tensor lfu_state) {
  TENSOR_ON_CUDA_GPU(unique_indices);
  TENSOR_ON_CUDA_GPU(unique_indices_length);
  TENSOR_ON_CUDA_GPU(unique_indices_count);
  TENSOR_ON_CUDA_GPU(lfu_state);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(unique_indices.get_device());

  int32_t N = unique_indices.size(0);
  AT_DISPATCH_INDEX_TYPES(
      unique_indices.scalar_type(), "lfu_update_counts_cuda", [&]() {
        lfu_update_counts_kernel<<<
            div_round_up(N, kMaxThreads),
            kMaxThreads,
            0,
            at::cuda::getCurrentCUDAStream()>>>(
            unique_indices
                .packed_accessor32<index_t, 1, at::RestrictPtrTraits>(),
            unique_indices_length.data_ptr<int32_t>(),
            unique_indices_count
                .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
            lfu_state.packed_accessor64<int64_t, 1, at::RestrictPtrTraits>());
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

constexpr int32_t kCacheSetBits = 24;
constexpr int32_t kLFUCounterBits = 40;
static_assert(kCacheSetBits + kLFUCounterBits == 8 * sizeof(int64_t), "");

template <typename index_t>
__global__ __launch_bounds__(kMaxThreads) void lfu_cache_find_uncached_kernel(
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        unique_indices,
    const int32_t* __restrict__ N_unique,
    int64_t max_indices,
    const at::PackedTensorAccessor32<int64_t, 2, at::RestrictPtrTraits>
        lxu_cache_state,
    uint64_t* __restrict__ cache_sets,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits>
        lfu_state) {
  int32_t N = unique_indices.size(0);
  int32_t C = lxu_cache_state.size(0);
  int32_t n = blockIdx.x * blockDim.y + threadIdx.y;
  if (n >= N) {
    return;
  }
  if (n >= *N_unique) {
    if (threadIdx.x == 0) {
      cache_sets[n] =
          (static_cast<uint64_t>(C)
           << kLFUCounterBits); // invalid index, used as sentinel
    }
    return;
  }
  int64_t idx = unique_indices[n];
  if (idx == max_indices) {
    if (threadIdx.x == 0) {
      cache_sets[n] =
          (static_cast<uint64_t>(C)
           << kLFUCounterBits); // invalid index, used as sentinel
    }
    return;
  }
  uint32_t cache_set = cache_slot(idx, C);

  auto slot = threadIdx.x;
  bool found = __ldg((&lxu_cache_state[cache_set][0]) + slot) == idx;
  if (found) {
    // mark it as existing.
    cache_sets[n] =
        (static_cast<uint64_t>(C)
         << kLFUCounterBits); // invalid index, used as sentinel
  }

#ifdef __HIP_PLATFORM_HCC__
  // FIXME: __any_sync with mask isn't supported by HIP yet.
  // See https://fburl.com/fvy7j0lq for the similar context.
  // assert false here with https://fburl.com/pfm7enw2
  assert(false);
  if (!__any(found)) {
#else
  if (!__any_sync(0xFFFFFFFF, found)) {
#endif
    if (threadIdx.x == 0) {
      // sort so the highest LFUs come first in the segment.
      // assume lfu_state[idx] <= 2^40 - 1 and cache_set < 2^24 -1
      cache_sets[n] = ((static_cast<uint64_t>(cache_set) << kLFUCounterBits)) |
          ((static_cast<uint64_t>(1) << kLFUCounterBits) - 1 - lfu_state[idx]);
    }
  }
}

std::pair<Tensor, Tensor> lfu_cache_find_uncached_cuda(
    Tensor unique_indices,
    Tensor unique_indices_length,
    int64_t max_indices,
    Tensor lxu_cache_state,
    Tensor lfu_state) {
  TENSOR_ON_CUDA_GPU(unique_indices);
  TENSOR_ON_CUDA_GPU(unique_indices_length);
  TENSOR_ON_CUDA_GPU(lxu_cache_state);
  TENSOR_ON_CUDA_GPU(lfu_state);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(unique_indices.get_device());

  auto cache_sets =
      empty_like(unique_indices, unique_indices.options().dtype(at::kLong));
  int32_t N = unique_indices.numel();
  auto sorted_cache_sets = empty_like(cache_sets);
  auto cache_set_sorted_unique_indices = empty_like(unique_indices);

  AT_DISPATCH_INDEX_TYPES(
      unique_indices.scalar_type(), "lfu_cache_find_uncached_cuda", [&]() {
        // Find uncached indices
        lfu_cache_find_uncached_kernel<<<
            div_round_up(N, kMaxThreads / kWarpSize),
            dim3(kWarpSize, kMaxThreads / kWarpSize),
            0,
            at::cuda::getCurrentCUDAStream()>>>(
            unique_indices
                .packed_accessor32<index_t, 1, at::RestrictPtrTraits>(),
            unique_indices_length.data_ptr<int32_t>(),
            max_indices,
            lxu_cache_state
                .packed_accessor32<int64_t, 2, at::RestrictPtrTraits>(),
            (uint64_t*)cache_sets.data_ptr<int64_t>(),
            lfu_state.packed_accessor64<int64_t, 1, at::RestrictPtrTraits>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        // Sort the cache sets and ids
        size_t temp_storage_bytes = 0;
        AT_CUDA_CHECK(FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceRadixSort::SortPairs(
            nullptr,
            temp_storage_bytes,
            (uint64_t*)cache_sets.data_ptr<int64_t>(),
            (uint64_t*)sorted_cache_sets.data_ptr<int64_t>(),
            unique_indices.data_ptr<index_t>(),
            cache_set_sorted_unique_indices.data_ptr<index_t>(),
            N,
            0,
            int(log2(float(lxu_cache_state.size(0) + 1)) + 1) + kLFUCounterBits,
            at::cuda::getCurrentCUDAStream(),
            false));
        auto temp_storage = at::empty(
            {static_cast<int64_t>(temp_storage_bytes)},
            unique_indices.options().dtype(at::kByte));
        AT_CUDA_CHECK(FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceRadixSort::SortPairs(
            temp_storage.data_ptr(),
            temp_storage_bytes,
            (uint64_t*)cache_sets.data_ptr<int64_t>(),
            (uint64_t*)sorted_cache_sets.data_ptr<int64_t>(),
            unique_indices.data_ptr<index_t>(),
            cache_set_sorted_unique_indices.data_ptr<index_t>(),
            N,
            0,
            int(log2(float(lxu_cache_state.size(0) + 1)) + 1) + kLFUCounterBits,
            at::cuda::getCurrentCUDAStream(),
            false));
      });
  return {sorted_cache_sets, cache_set_sorted_unique_indices};
}

template <typename emb_t, typename cache_t>
__global__ __launch_bounds__(kCacheMaxThreads) void lfu_cache_insert_kernel(
    at::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> weights,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        cache_hash_size_cumsum,
    const at::PackedTensorAccessor64<int32_t, 1, at::RestrictPtrTraits>
        cache_index_table_map,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        weights_offsets,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        D_offsets,
    const uint64_t* __restrict__ sorted_cache_sets,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        cache_set_sorted_indices,
    const int32_t* __restrict__ N_unique,
    at::PackedTensorAccessor32<int64_t, 2, at::RestrictPtrTraits>
        lxu_cache_state,
    at::PackedTensorAccessor64<cache_t, 2, at::RestrictPtrTraits>
        lxu_cache_weights,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits>
        lfu_state,
    bool stochastic_rounding,
    at::PhiloxCudaState stochastic_rounding_philox_args) {
  int32_t C = lxu_cache_state.size(0);
  int32_t n = blockIdx.x * blockDim.y + threadIdx.y;
  if (n >= *N_unique) {
    return;
  }
  // check if this warp is responsible for this whole segment.
  bool segment_start =
      (n == 0 ||
       (sorted_cache_sets[n - 1] >> kLFUCounterBits) !=
           (sorted_cache_sets[n] >> kLFUCounterBits));

  if (!segment_start) {
    // don't have *warp* divergence since we launch full warps in blockDim.x,
    // so we can just exit this warp entirely.
    return;
  }
  uint32_t cache_set = (sorted_cache_sets[n] >> kLFUCounterBits);
  if (cache_set == C) {
    // ignore the already-existing elements
    return;
  }

  int32_t SL = 1;
  while (n + SL < *N_unique &&
         (sorted_cache_sets[n + SL] >> kLFUCounterBits) == cache_set) {
    SL += 1;
  }

  // now, we need to insert the (unique!) values in indices[n:n + SL] into
  // our slots.
  int32_t slot = threadIdx.x;
  int64_t current_idx = lxu_cache_state[cache_set][slot];
  int64_t current_lfu_cost =
      (current_idx != static_cast<int64_t>(kCacheStateInvalid))
      ? lfu_state[current_idx]
      : -1;
  int64_t costs[1] = {current_lfu_cost};
  int32_t slots[1] = {slot};

  BitonicSort<int64_t, int32_t, 1, Comparator<int64_t>>::sort(costs, slots);
  int32_t sorted_slot = slots[0];
  int64_t sorted_lfu_cost = costs[0];

  for (int32_t l = 0; l < min(SL, kWarpSize); ++l) {
    int32_t insert_slot = shfl_sync(sorted_slot, l);
    int64_t insert_current_lfu_cost = shfl_sync(sorted_lfu_cost, l);
    int64_t insert_idx = cache_set_sorted_indices[n + l];
    int64_t insert_lfu_cost = lfu_state[insert_idx];

    if (insert_current_lfu_cost > insert_lfu_cost) {
      // don't insert.
      // all subsequent `current_lfu_cost` values are greater, and all
      // subsequent `insert_lfu_cost` values are smaller, so we can exit
      // early here.
      return;
    }
    int32_t t_insert = cache_index_table_map[insert_idx];
    int64_t idx_insert = insert_idx - cache_hash_size_cumsum[t_insert];
    int64_t weights_offset_insert = weights_offsets[t_insert];
    int32_t D_start_insert = D_offsets[t_insert];
    int32_t D_end_insert = D_offsets[t_insert + 1];
    int32_t D_insert = D_end_insert - D_start_insert;

    // not empty
    if (insert_current_lfu_cost != -1) {
      // ensure that threadIdx.x is the only thread reading/writing to
      // lxu_cache_state
      int64_t current_idx =
          threadIdx.x == 0 ? lxu_cache_state[cache_set][insert_slot] : 0;
      current_idx = shfl_sync(current_idx, 0);
      int32_t t_current = cache_index_table_map[current_idx];
      int64_t idx_current = current_idx - cache_hash_size_cumsum[t_current];
      int64_t weights_offset_current = weights_offsets[t_current];
      int32_t D_start_current = D_offsets[t_current];
      int32_t D_end_current = D_offsets[t_current + 1];
      int32_t D_current = D_end_current - D_start_current;

      int32_t D_emb = D_current;
      if (std::is_same<emb_t, uint8_t>::value) {
        D_emb += kINT8QparamsBytes;
      }
      auto weight_row = WeightRow<emb_t, cache_t, cache_t>(
          &weights[weights_offset_current + idx_current * D_emb + 0],
          &lxu_cache_weights[cache_set * kWarpSize + insert_slot][0],
          D_current,
          nullptr);
      if (!std::is_same<emb_t, float>::value && stochastic_rounding) {
        StochasticRoundingRNGState state;
        // different for every *run* and every *thread*.
        auto stochastic_rounding_seeds =
            at::cuda::philox::unpack(stochastic_rounding_philox_args);
        stochastic_rounding_init(
            std::get<0>(stochastic_rounding_seeds) ^
                std::get<1>(stochastic_rounding_seeds),
            (blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
             threadIdx.x) *
                    kWarpSize +
                l,
            &state);
        weight_row.set_stoc_state(&state);
      }

      float2 qparams;
      at::acc_type<cache_t, true> local_min =
          std::numeric_limits<at::acc_type<cache_t, true>>::max();
      at::acc_type<cache_t, true> local_max =
          std::numeric_limits<at::acc_type<cache_t, true>>::lowest();
      if (std::is_same<emb_t, uint8_t>::value) {
        for (int32_t d = threadIdx.x; d * 4 < D_current; d += blockDim.x) {
          Vec4T<cache_t> cache_weights_vec =
              weight_row.load(d * 4, qparams); // qparams not used
          local_max = max(local_max, vec4_max(cache_weights_vec));
          local_min = min(local_min, vec4_min(cache_weights_vec));
        }
        qparams = warp_find_qparams(local_min, local_max);
        if (threadIdx.x == 0) {
          weight_row.store_qparams(qparams);
        }
      }
      for (int32_t d = threadIdx.x; d * 4 < D_current; d += blockDim.x) {
        Vec4T<cache_t> cache_weights_vec = weight_row.load(d * 4, qparams);
        weight_row.evict(cache_weights_vec, d * 4, qparams);
      }
    }
    // insert into cache
    int32_t D_emb = D_insert;
    if (std::is_same<emb_t, uint8_t>::value) {
      D_emb += kINT8QparamsBytes;
    }
    auto weight_row_cache = WeightRow<emb_t, cache_t, cache_t>(
        &weights[weights_offset_insert + idx_insert * D_emb + 0],
        &lxu_cache_weights[cache_set * kWarpSize + insert_slot][0],
        D_insert,
        nullptr);

    auto weight_row_emb = WeightRow<emb_t, cache_t, cache_t>(
        &weights[weights_offset_insert + idx_insert * D_emb + 0],
        nullptr,
        D_insert,
        nullptr);

    float2 qparams;
    if (std::is_same<emb_t, uint8_t>::value) {
      qparams = weight_row_emb.load_qparams();
    }
    for (int32_t d = threadIdx.x; d * 4 < D_insert; d += blockDim.x) {
      auto row = weight_row_emb.load(d * 4, qparams);
      weight_row_cache.store(row, d * 4, qparams);
    }
    if (threadIdx.x == 0) {
      lxu_cache_state[cache_set][insert_slot] = insert_idx;
    }
  }
}

void lfu_cache_insert_cuda(
    Tensor weights,
    Tensor cache_hash_size_cumsum,
    Tensor cache_index_table_map,
    Tensor weights_offsets,
    Tensor D_offsets,
    Tensor sorted_cache_sets,
    Tensor cache_set_sorted_unique_indices,
    Tensor unique_indices_length,
    Tensor lxu_cache_state,
    Tensor lxu_cache_weights,
    Tensor lfu_state,
    bool stochastic_rounding) {
  TENSOR_ON_CUDA_GPU(weights);
  TENSOR_ON_CUDA_GPU(cache_hash_size_cumsum);
  TENSOR_ON_CUDA_GPU(cache_index_table_map);
  TENSOR_ON_CUDA_GPU(weights_offsets);
  TENSOR_ON_CUDA_GPU(D_offsets);
  TENSOR_ON_CUDA_GPU(sorted_cache_sets);
  TENSOR_ON_CUDA_GPU(cache_set_sorted_unique_indices);
  TENSOR_ON_CUDA_GPU(unique_indices_length);
  TENSOR_ON_CUDA_GPU(lxu_cache_state);
  TENSOR_ON_CUDA_GPU(lxu_cache_weights);
  TENSOR_ON_CUDA_GPU(lfu_state);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(weights.get_device());

  int32_t N = cache_set_sorted_unique_indices.numel();

  DISPATCH_EMB_CACHE_TYPES(
      weights.type(),
      lxu_cache_weights.type(),
      "lfu_cache_insert_kernel_2",
      ([&] {
        at::PhiloxCudaState rng_engine_inputs;
        if (stochastic_rounding && !std::is_same<emb_t, float>::value) {
          auto gen = at::cuda::detail::getDefaultCUDAGenerator();
          std::lock_guard<std::mutex> lock(gen.mutex());
          rng_engine_inputs = at::check_generator<at::CUDAGeneratorImpl>(gen)
                                  ->philox_cuda_state(4);
        }

        lfu_cache_insert_kernel<emb_t, cache_t>
            <<<div_round_up(N, kCacheMaxThreads / kWarpSize),
               dim3(kWarpSize, kCacheMaxThreads / kWarpSize),
               0,
               at::cuda::getCurrentCUDAStream()>>>(
                weights.packed_accessor64<emb_t, 1, at::RestrictPtrTraits>(),
                cache_hash_size_cumsum
                    .packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                cache_index_table_map
                    .packed_accessor64<int32_t, 1, at::RestrictPtrTraits>(),
                weights_offsets
                    .packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                D_offsets
                    .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                (uint64_t*)sorted_cache_sets.data_ptr<int64_t>(),
                cache_set_sorted_unique_indices
                    .packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                unique_indices_length.data_ptr<int32_t>(),
                lxu_cache_state
                    .packed_accessor32<int64_t, 2, at::RestrictPtrTraits>(),
                lxu_cache_weights
                    .packed_accessor64<cache_t, 2, at::RestrictPtrTraits>(),
                lfu_state
                    .packed_accessor64<int64_t, 1, at::RestrictPtrTraits>(),
                stochastic_rounding,
                rng_engine_inputs);
      }));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void lfu_cache_populate_cuda(
    Tensor weights,
    Tensor cache_hash_size_cumsum,
    int64_t total_cache_hash_size,
    Tensor cache_index_table_map,
    Tensor weights_offsets,
    Tensor D_offsets,
    Tensor linear_cache_indices,
    Tensor lxu_cache_state,
    Tensor lxu_cache_weights,
    Tensor lfu_state,
    bool stochastic_rounding) {
  TENSOR_ON_CUDA_GPU(weights);
  TENSOR_ON_CUDA_GPU(cache_hash_size_cumsum);
  TENSOR_ON_CUDA_GPU(cache_index_table_map);
  TENSOR_ON_CUDA_GPU(weights_offsets);
  TENSOR_ON_CUDA_GPU(D_offsets);
  TENSOR_ON_CUDA_GPU(linear_cache_indices);
  TENSOR_ON_CUDA_GPU(lxu_cache_state);
  TENSOR_ON_CUDA_GPU(lxu_cache_weights);
  TENSOR_ON_CUDA_GPU(lfu_state);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(weights.get_device());

  TORCH_CHECK(
      linear_cache_indices.numel() < std::numeric_limits<int32_t>::max());
  if (linear_cache_indices.numel() == 0) {
    // nothing to do
    return;
  }

  // get unqiue indices
  Tensor unique_indices;
  Tensor unique_indices_length;
  c10::optional<Tensor> unique_indices_count;
  std::tie(unique_indices, unique_indices_length, unique_indices_count) =
      get_unique_indices_cuda(
          linear_cache_indices, total_cache_hash_size, true);

  // update lfu counts
  lfu_update_counts_cuda(
      unique_indices, unique_indices_length, *unique_indices_count, lfu_state);

  // find uncached indices
  auto cache_sets_and_unique_indices = lfu_cache_find_uncached_cuda(
      unique_indices,
      unique_indices_length,
      total_cache_hash_size,
      lxu_cache_state,
      lfu_state);
  auto sorted_cache_sets = cache_sets_and_unique_indices.first;
  auto cache_set_sorted_unique_indices = cache_sets_and_unique_indices.second;

  // insert caching weights
  lfu_cache_insert_cuda(
      weights,
      cache_hash_size_cumsum,
      cache_index_table_map,
      weights_offsets,
      D_offsets,
      sorted_cache_sets,
      cache_set_sorted_unique_indices,
      unique_indices_length,
      lxu_cache_state,
      lxu_cache_weights,
      lfu_state,
      stochastic_rounding);
}

// In `lfu_cache_insert_kernel`, we use `emb_t` and `cache_t` for the
// high-precision cache implementation, where we can have {FP32, FP16, INT8}
// for embedding precision (data types), and {FP32, FP16} for cache precision
// (data types).
//
// In `lfu_cache_insert_byte_kernel`, we only use uint8_t for the both embedding
// and cache data type (conforming to the inference TBE kernel logics).
// - We pass in `weights_tys` to denote the real data types for the embeddings:
// {FP32, FP16, INT8, INT4}. For example, FP32 is 4 byte element in the byte
// tensor, and INT4 is half byte element in the byte tensor.
// - We only assume that the embedding and cache have the same precisions (the
// real "precision" is determined by `weights_tys` although the data types are
// uint8_t only). Basically no "high-precision cache" support for now.
// - The insert/evict of embedding row from the cache are done in a byte-by-byte
// manner.
template <typename index_t>
__global__
__launch_bounds__(kCacheMaxThreads) void lfu_cache_insert_byte_kernel(
    at::PackedTensorAccessor64<uint8_t, 1, at::RestrictPtrTraits> weights,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        cache_hash_size_cumsum,
    const at::PackedTensorAccessor64<int32_t, 1, at::RestrictPtrTraits>
        cache_index_table_map,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        weights_offsets,
    const at::PackedTensorAccessor32<uint8_t, 1, at::RestrictPtrTraits>
        weights_tys,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        D_offsets,
    const uint64_t* __restrict__ sorted_cache_sets,
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        cache_set_sorted_indices,
    const int32_t* __restrict__ N_unique,
    at::PackedTensorAccessor32<int64_t, 2, at::RestrictPtrTraits>
        lxu_cache_state,
    at::PackedTensorAccessor64<uint8_t, 2, at::RestrictPtrTraits>
        lxu_cache_weights,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits>
        lfu_state) {
  int32_t C = lxu_cache_state.size(0);
  int32_t n = blockIdx.x * blockDim.y + threadIdx.y;
  if (n >= *N_unique) {
    return;
  }
  // check if this warp is responsible for this whole segment.
  bool segment_start =
      (n == 0 ||
       (sorted_cache_sets[n - 1] >> kLFUCounterBits) !=
           (sorted_cache_sets[n] >> kLFUCounterBits));

  if (!segment_start) {
    // don't have *warp* divergence since we launch full warps in blockDim.x,
    // so we can just exit this warp entirely.
    return;
  }
  uint32_t cache_set = (sorted_cache_sets[n] >> kLFUCounterBits);
  if (cache_set == C) {
    // ignore the already-existing elements
    return;
  }

  int32_t SL = 1;
  while (n + SL < *N_unique &&
         (sorted_cache_sets[n + SL] >> kLFUCounterBits) == cache_set) {
    SL += 1;
  }

  // now, we need to insert the (unique!) values in indices[n:n + SL] into
  // our slots.
  int32_t slot = threadIdx.x;
  int64_t current_idx = lxu_cache_state[cache_set][slot];
  int64_t current_lfu_cost =
      (current_idx != static_cast<int64_t>(kCacheStateInvalid))
      ? lfu_state[current_idx]
      : -1;
  int64_t costs[1] = {current_lfu_cost};
  int32_t slots[1] = {slot};

  BitonicSort<int64_t, int32_t, 1, Comparator<int64_t>>::sort(costs, slots);
  int32_t sorted_slot = slots[0];
  int64_t sorted_lfu_cost = costs[0];

  for (int32_t l = 0; l < min(SL, kWarpSize); ++l) {
    int32_t insert_slot = shfl_sync(sorted_slot, l);
    int64_t insert_current_lfu_cost = shfl_sync(sorted_lfu_cost, l);
    index_t insert_idx = cache_set_sorted_indices[n + l];
    int64_t insert_lfu_cost = lfu_state[insert_idx];

    if (insert_current_lfu_cost > insert_lfu_cost) {
      // don't insert.
      // all subsequent `current_lfu_cost` values are greater, and all
      // subsequent `insert_lfu_cost` values are smaller, so we can exit
      // early here.
      return;
    }
    int32_t t_insert = cache_index_table_map[insert_idx];
    SparseType weight_ty_insert =
        static_cast<SparseType>(weights_tys[t_insert]);
    int64_t idx_insert = insert_idx - cache_hash_size_cumsum[t_insert];
    int64_t weights_offset_insert = weights_offsets[t_insert];
    int32_t D_start_insert = D_offsets[t_insert];
    int32_t D_end_insert = D_offsets[t_insert + 1];
    int32_t D_insert = D_end_insert - D_start_insert;

    const int32_t D_insert_bytes =
        padded_row_size_in_bytes(D_insert, weight_ty_insert);

    // not empty
    if (insert_current_lfu_cost != -1) {
      // ensure that threadIdx.x is the only thread reading/writing to
      // lxu_cache_state
      int64_t current_idx =
          threadIdx.x == 0 ? lxu_cache_state[cache_set][insert_slot] : 0;
      current_idx = shfl_sync(current_idx, 0);
      int32_t t_current = cache_index_table_map[current_idx];
      SparseType weight_ty_current =
          static_cast<SparseType>(weights_tys[t_current]);
      int64_t idx_current = current_idx - cache_hash_size_cumsum[t_current];
      int64_t weights_offset_current = weights_offsets[t_current];
      int32_t D_start_current = D_offsets[t_current];
      int32_t D_end_current = D_offsets[t_current + 1];
      int32_t D_current = D_end_current - D_start_current;

      const int32_t D_current_bytes =
          padded_row_size_in_bytes(D_current, weight_ty_current);

      auto row =
          &weights[weights_offset_current + idx_current * D_current_bytes + 0];
      auto cache_row =
          &lxu_cache_weights[cache_set * kWarpSize + insert_slot][0];
      // Evict the cache
      for (int32_t d = threadIdx.x; d < D_current_bytes; d += blockDim.x) {
        row[d] = cache_row[d]; // uint8_t access
      }
    }
    // insert into cache
    auto row =
        &weights[weights_offset_insert + idx_insert * D_insert_bytes + 0];
    auto cache_row = &lxu_cache_weights[cache_set * kWarpSize + insert_slot][0];
    for (int32_t d = threadIdx.x; d < D_insert_bytes; d += blockDim.x) {
      cache_row[d] = row[d];
    }
    if (threadIdx.x == 0) {
      lxu_cache_state[cache_set][insert_slot] = insert_idx;
    }
  }
}

void lfu_cache_insert_byte_cuda(
    Tensor weights,
    Tensor cache_hash_size_cumsum,
    Tensor cache_index_table_map,
    Tensor weights_offsets,
    Tensor weights_tys,
    Tensor D_offsets,
    Tensor sorted_cache_sets,
    Tensor cache_set_sorted_unique_indices,
    Tensor unique_indices_length,
    Tensor lxu_cache_state,
    Tensor lxu_cache_weights,
    Tensor lfu_state) {
  TENSOR_ON_CUDA_GPU(weights);
  TENSOR_ON_CUDA_GPU(cache_hash_size_cumsum);
  TENSOR_ON_CUDA_GPU(cache_index_table_map);
  TENSOR_ON_CUDA_GPU(weights_offsets);
  TENSOR_ON_CUDA_GPU(weights_tys)
  TENSOR_ON_CUDA_GPU(D_offsets);
  TENSOR_ON_CUDA_GPU(sorted_cache_sets);
  TENSOR_ON_CUDA_GPU(cache_set_sorted_unique_indices);
  TENSOR_ON_CUDA_GPU(unique_indices_length);
  TENSOR_ON_CUDA_GPU(lxu_cache_state);
  TENSOR_ON_CUDA_GPU(lxu_cache_weights);
  TENSOR_ON_CUDA_GPU(lfu_state);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(weights.get_device());

  int32_t N = cache_set_sorted_unique_indices.numel();

  AT_DISPATCH_INDEX_TYPES(
      cache_set_sorted_unique_indices.scalar_type(),
      "lfu_cache_insert_byte_cuda",
      [&]() {
        lfu_cache_insert_byte_kernel<<<
            div_round_up(N, kCacheMaxThreads / kWarpSize),
            dim3(kWarpSize, kCacheMaxThreads / kWarpSize),
            0,
            at::cuda::getCurrentCUDAStream()>>>(
            weights.packed_accessor64<uint8_t, 1, at::RestrictPtrTraits>(),
            cache_hash_size_cumsum
                .packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
            cache_index_table_map
                .packed_accessor64<int32_t, 1, at::RestrictPtrTraits>(),
            weights_offsets
                .packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
            weights_tys.packed_accessor32<uint8_t, 1, at::RestrictPtrTraits>(),
            D_offsets.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
            (uint64_t*)sorted_cache_sets.data_ptr<int64_t>(),
            cache_set_sorted_unique_indices
                .packed_accessor32<index_t, 1, at::RestrictPtrTraits>(),
            unique_indices_length.data_ptr<int32_t>(),
            lxu_cache_state
                .packed_accessor32<int64_t, 2, at::RestrictPtrTraits>(),
            lxu_cache_weights
                .packed_accessor64<uint8_t, 2, at::RestrictPtrTraits>(),
            lfu_state.packed_accessor64<int64_t, 1, at::RestrictPtrTraits>());
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void lfu_cache_populate_byte_cuda(
    Tensor weights,
    Tensor cache_hash_size_cumsum,
    int64_t total_cache_hash_size,
    Tensor cache_index_table_map,
    Tensor weights_offsets,
    Tensor weights_tys,
    Tensor D_offsets,
    Tensor linear_cache_indices,
    Tensor lxu_cache_state,
    Tensor lxu_cache_weights,
    Tensor lfu_state) {
  TENSOR_ON_CUDA_GPU(weights);
  TENSOR_ON_CUDA_GPU(cache_hash_size_cumsum);
  TENSOR_ON_CUDA_GPU(cache_index_table_map);
  TENSOR_ON_CUDA_GPU(weights_offsets);
  TENSOR_ON_CUDA_GPU(weights_tys)
  TENSOR_ON_CUDA_GPU(D_offsets);
  TENSOR_ON_CUDA_GPU(linear_cache_indices);
  TENSOR_ON_CUDA_GPU(lxu_cache_state);
  TENSOR_ON_CUDA_GPU(lxu_cache_weights);
  TENSOR_ON_CUDA_GPU(lfu_state);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(weights.get_device());

  TORCH_CHECK(
      linear_cache_indices.numel() < std::numeric_limits<int32_t>::max());
  if (linear_cache_indices.numel() == 0) {
    // nothing to do
    return;
  }

  // get unqiue indices
  Tensor unique_indices;
  Tensor unique_indices_length;
  c10::optional<Tensor> unique_indices_count;
  std::tie(unique_indices, unique_indices_length, unique_indices_count) =
      get_unique_indices_cuda(
          linear_cache_indices, total_cache_hash_size, true);

  // update lfu counts
  lfu_update_counts_cuda(
      unique_indices, unique_indices_length, *unique_indices_count, lfu_state);

  // find uncached indices
  auto cache_sets_and_unique_indices = lfu_cache_find_uncached_cuda(
      unique_indices,
      unique_indices_length,
      total_cache_hash_size,
      lxu_cache_state,
      lfu_state);
  auto sorted_cache_sets = cache_sets_and_unique_indices.first;
  auto cache_set_sorted_unique_indices = cache_sets_and_unique_indices.second;

  // insert caching weights
  lfu_cache_insert_byte_cuda(
      weights,
      cache_hash_size_cumsum,
      cache_index_table_map,
      weights_offsets,
      weights_tys,
      D_offsets,
      sorted_cache_sets,
      cache_set_sorted_unique_indices,
      unique_indices_length,
      lxu_cache_state,
      lxu_cache_weights,
      lfu_state);
}

template <typename index_t>
__global__ __launch_bounds__(kMaxThreads) void lxu_cache_lookup_kernel(
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        linear_cache_indices,
    const at::PackedTensorAccessor32<int64_t, 2, at::RestrictPtrTraits>
        lxu_cache_state,
    at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        lxu_cache_locations) {
  const int32_t C = lxu_cache_state.size(0);
  const int32_t N = linear_cache_indices.size(0);
  int32_t n = blockIdx.x * blockDim.y + threadIdx.y;
  if (n >= N) {
    return;
  }
  int64_t idx = linear_cache_indices[n];
  int32_t cache_set = cache_slot(idx, C);
  auto slot = threadIdx.x;
  bool found = (__ldg((&lxu_cache_state[cache_set][0]) + slot) == idx);
  if (found) {
    lxu_cache_locations[n] = cache_set * kWarpSize + slot;
  }
#ifdef __HIP_PLATFORM_HCC__
  // FIXME: __any_sync with mask isn't supported by HIP yet.
  // See https://fburl.com/fvy7j0lq for the similar context.
  // assert false here with https://fburl.com/pfm7enw2
  assert(false);
  if (!__any(found)) {
#else
  if (!__any_sync(0xFFFFFFFF, found)) {
#endif
    if (threadIdx.x == 0) {
      lxu_cache_locations[n] = kCacheLocationMissing;
    }
  }
}

Tensor lxu_cache_lookup_cuda(
    Tensor linear_cache_indices,
    Tensor lxu_cache_state) {
  TENSOR_ON_CUDA_GPU(linear_cache_indices);
  TENSOR_ON_CUDA_GPU(lxu_cache_state);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(linear_cache_indices.get_device());

  const auto N = linear_cache_indices.numel();
  auto lxu_cache_locations = empty_like(
      linear_cache_indices, linear_cache_indices.options().dtype(at::kInt));
  if (linear_cache_indices.numel() == 0) {
    // nothing to do
    return lxu_cache_locations;
  }

  const dim3 threads(kWarpSize, kMaxThreads / kWarpSize);
  const dim3 blocks(div_round_up(N, kMaxThreads / kWarpSize));

  AT_DISPATCH_INDEX_TYPES(
      linear_cache_indices.scalar_type(), "lxu_cache_lookup_cuda", [&]() {
        lxu_cache_lookup_kernel<<<
            blocks,
            threads,
            0,
            at::cuda::getCurrentCUDAStream()>>>(
            linear_cache_indices
                .packed_accessor32<index_t, 1, at::RestrictPtrTraits>(),
            lxu_cache_state
                .packed_accessor32<int64_t, 2, at::RestrictPtrTraits>(),
            lxu_cache_locations
                .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });

  return lxu_cache_locations;
}
