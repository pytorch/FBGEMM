/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/native/cuda/SortingCommon.cuh>
#include <limits>
#include "common.cuh"
#include "fbgemm_gpu/utils/cuda_utilities.cuh"

using Tensor = at::Tensor;
using namespace fbgemm_gpu;

namespace {

// Tag of empty and invalidated ways; key -1 never matches or is admitted.
constexpr int64_t kKvHbmEmptyTag = -1;
// Ways of a set probed by one warp.
constexpr int64_t kKvHbmMaxWays = 32;
// Admissions of at most this many keys, or with at most this many absent
// keys, are left to the caller's host path.
constexpr int64_t kKvAdmitMinFusedKeys = 32;

constexpr int64_t kKvDirectHit = -1;
constexpr int64_t kKvDirectBadMapping = -2;

// Row of the packed input rows for direct output position `index`;
// kKvDirectHit for a hit, and kKvDirectBadMapping for an out-of-range
// miss-slot or rank mapping, which the wrapper never passes.
__device__ __forceinline__ int64_t kv_hbm_direct_source_row(
    const int32_t* __restrict__ miss_slots,
    const int32_t* __restrict__ miss_rank,
    int64_t index,
    int64_t num_ranks,
    int64_t num_inputs) {
  const int64_t slot = miss_slots[index];
  if (slot < 0) {
    return kKvDirectHit;
  }
  if (slot >= num_ranks) {
    return kKvDirectBadMapping;
  }
  const int64_t rank = miss_rank[slot];
  return rank >= 0 && rank < num_inputs ? rank : kKvDirectBadMapping;
}

__device__ __forceinline__ int64_t
kv_hbm_cache_set(int64_t key, int64_t num_sets) {
  uint64_t mixed = static_cast<uint64_t>(key ^ (key >> 31));
  mixed *= 0x9E3779B97F4A7C15ULL;
  mixed ^= static_cast<uint64_t>(static_cast<int64_t>(mixed) >> 29);
  const auto remainder = static_cast<int64_t>(mixed) % num_sets;
  return remainder < 0 ? remainder + num_sets : remainder;
}

__global__ __launch_bounds__(kMaxThreads) void kv_hbm_cache_lookup_kernel(
    const int64_t* __restrict__ keys,
    const int64_t* __restrict__ tags,
    const uint8_t* __restrict__ cache_rows,
    int64_t* __restrict__ lru,
    uint8_t* __restrict__ output_rows,
    bool* __restrict__ hit_mask,
    int64_t num_keys,
    int64_t num_sets,
    int64_t assoc,
    int64_t row_bytes,
    int64_t tick) {
  const auto lane = threadIdx.x;
  const int64_t first =
      static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.y;
  for (int64_t index = first; index < num_keys; index += stride) {
    const auto key = keys[index];
    const auto set = kv_hbm_cache_set(key, num_sets);
    const bool found = key != kKvHbmEmptyTag && lane < assoc &&
        tags[set * assoc + lane] == key;
    const auto bitmap = fbgemm_gpu::ballot_sync(found);
    const auto way = bitmap ? __ffsll(static_cast<long long>(bitmap)) - 1 : -1;
    if (lane == 0) {
      hit_mask[index] = way >= 0;
      if (way >= 0) {
        lru[set * assoc + way] = tick;
      }
    }
    if (way < 0) {
      continue;
    }
    const auto* source = cache_rows + (set * assoc + way) * row_bytes;
    auto* destination = output_rows + index * row_bytes;
    constexpr int64_t kVectorBytes = sizeof(uint4);
    if (row_bytes % kVectorBytes == 0 &&
        reinterpret_cast<uintptr_t>(source) % kVectorBytes == 0 &&
        reinterpret_cast<uintptr_t>(destination) % kVectorBytes == 0) {
      const auto* source_vectors = reinterpret_cast<const uint4*>(source);
      auto* destination_vectors = reinterpret_cast<uint4*>(destination);
      for (int64_t chunk = lane; chunk < row_bytes / kVectorBytes;
           chunk += blockDim.x) {
        destination_vectors[chunk] = source_vectors[chunk];
      }
    } else {
      for (int64_t byte = lane; byte < row_bytes; byte += blockDim.x) {
        destination[byte] = source[byte];
      }
    }
  }
}

constexpr int kKvAdmitWarps = 8;
constexpr int kKvDirectThreads = 256;
constexpr int64_t kKvDirectMaxKeys = 512 * 1024;
constexpr int64_t kKvDirectRowBytes = 80;
constexpr int64_t kKvPackedRowBytes = 72;

__device__ __forceinline__ uint64_t kv_hbm_direct_hash(int64_t key) {
  auto mixed = static_cast<uint64_t>(key ^ (key >> 31));
  mixed *= 0x9E3779B97F4A7C15ULL;
  return mixed ^ static_cast<uint64_t>(static_cast<int64_t>(mixed) >> 29);
}

__global__
__launch_bounds__(kKvDirectThreads) void kv_hbm_cache_direct_probe_kernel(
    const int64_t* __restrict__ keys,
    const int64_t* __restrict__ tags,
    const uint8_t* __restrict__ cache_rows,
    int64_t* __restrict__ lru,
    uint8_t* __restrict__ output_rows,
    int32_t* __restrict__ miss_table,
    int32_t* __restrict__ miss_slots,
    int32_t* __restrict__ block_hits,
    int64_t num_keys,
    int64_t num_sets,
    int64_t assoc,
    int64_t table_mask,
    int64_t tick) {
  const auto lane = threadIdx.x % 32;
  const auto warp = threadIdx.x / 32;
  constexpr int kWarps = kKvDirectThreads / 32;
  int hits = 0;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * kWarps + warp;
       index < num_keys;
       index += static_cast<int64_t>(gridDim.x) * kWarps) {
    const auto key = keys[index];
    const auto set = kv_hbm_cache_set(key, num_sets);
    const auto bitmap = fbgemm_gpu::ballot_sync(
        key != kKvHbmEmptyTag && lane < assoc &&
        tags[set * assoc + lane] == key);
    const auto way = bitmap ? __ffsll(static_cast<long long>(bitmap)) - 1 : -1;
    if (way >= 0) {
      if (lane == 0) {
        auto* stamp =
            reinterpret_cast<unsigned long long*>(lru + set * assoc + way);
        hits += atomicExch(stamp, static_cast<unsigned long long>(tick)) !=
            static_cast<unsigned long long>(tick);
        miss_slots[index] = -1;
      }
      const auto* source = reinterpret_cast<const uint4*>(
          cache_rows + (set * assoc + way) * kKvDirectRowBytes);
      auto* destination =
          reinterpret_cast<uint4*>(output_rows + index * kKvDirectRowBytes);
      for (int chunk = lane; chunk < kKvDirectRowBytes / sizeof(uint4);
           chunk += 32) {
        destination[chunk] = source[chunk];
      }
    } else if (lane == 0) {
      auto slot = static_cast<int32_t>(kv_hbm_direct_hash(key) & table_mask);
      while (true) {
        const auto owner =
            atomicCAS(miss_table + slot, -1, static_cast<int32_t>(index));
        if (owner == -1 || keys[owner] == key) {
          miss_slots[index] = slot;
          break;
        }
        slot = static_cast<int32_t>((slot + 1) & table_mask);
      }
    }
  }
  using Reduce =
      FBGEMM_GPU_CUB_NS_PREFIX cub::BlockReduce<int, kKvDirectThreads>;
  __shared__ typename Reduce::TempStorage temp;
  const auto total = Reduce(temp).Sum(hits);
  if (threadIdx.x == 0) {
    block_hits[blockIdx.x] = total;
  }
}

__global__
__launch_bounds__(kKvDirectThreads) void kv_hbm_cache_direct_compact_kernel(
    const int64_t* __restrict__ keys,
    const int32_t* __restrict__ miss_table,
    const int32_t* __restrict__ block_hits,
    int64_t* __restrict__ miss_keys,
    int32_t* __restrict__ compact_slots,
    int32_t* __restrict__ status,
    int64_t table_capacity,
    int64_t num_probe_blocks) {
  using Reduce =
      FBGEMM_GPU_CUB_NS_PREFIX cub::BlockReduce<int, kKvDirectThreads>;
  using Scan = FBGEMM_GPU_CUB_NS_PREFIX cub::BlockScan<int, kKvDirectThreads>;
  __shared__ union {
    typename Reduce::TempStorage reduce;
    typename Scan::TempStorage scan;
  } temp;
  __shared__ int base;
  int hits = 0;
  for (int64_t index =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < num_probe_blocks;
       index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    hits += block_hits[index];
  }
  const auto total_hits = Reduce(temp.reduce).Sum(hits);
  if (threadIdx.x == 0 && total_hits != 0) {
    atomicAdd(status, total_hits);
  }
  __syncthreads();
  for (int64_t tile = static_cast<int64_t>(blockIdx.x) * blockDim.x;
       tile < table_capacity;
       tile += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const auto slot = tile + threadIdx.x;
    const auto owner = slot < table_capacity ? miss_table[slot] : -1;
    int offset;
    int count;
    Scan(temp.scan).ExclusiveSum(owner >= 0 ? 1 : 0, offset, count);
    if (threadIdx.x == 0) {
      base = count == 0 ? 0 : atomicAdd(status + 1, count);
    }
    __syncthreads();
    if (owner >= 0) {
      miss_keys[base + offset] = keys[owner];
      compact_slots[base + offset] = static_cast<int32_t>(slot);
    }
    __syncthreads();
  }
}

__global__
__launch_bounds__(kKvDirectThreads) void kv_hbm_cache_direct_rank_kernel(
    const int32_t* __restrict__ sorted_slots,
    int32_t* __restrict__ miss_rank,
    int64_t num_misses) {
  for (int64_t index =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < num_misses;
       index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    miss_rank[sorted_slots[index]] = static_cast<int32_t>(index);
  }
}

__device__ __forceinline__ void kv_hbm_copy_packed_row(
    const uint8_t* source,
    uint8_t* destination,
    int lane,
    int lanes) {
  if (reinterpret_cast<uintptr_t>(source) % sizeof(uint2) == 0 &&
      reinterpret_cast<uintptr_t>(destination) % sizeof(uint2) == 0) {
    const auto* source_vectors = reinterpret_cast<const uint2*>(source);
    auto* destination_vectors = reinterpret_cast<uint2*>(destination);
    for (int chunk = lane; chunk < kKvDirectRowBytes / sizeof(uint2);
         chunk += lanes) {
      destination_vectors[chunk] = chunk < kKvPackedRowBytes / sizeof(uint2)
          ? source_vectors[chunk]
          : make_uint2(0, 0);
    }
  } else {
    for (int byte = lane; byte < kKvDirectRowBytes; byte += lanes) {
      destination[byte] = byte < kKvPackedRowBytes ? source[byte] : 0;
    }
  }
}

// Fills direct output row `index` with its miss row, unless it is a hit. A
// bad mapping gets a zero row, as for an absent id.
__device__ __forceinline__ void kv_hbm_write_direct_output(
    const uint8_t* __restrict__ input_rows,
    const int32_t* __restrict__ miss_slots,
    const int32_t* __restrict__ miss_rank,
    uint8_t* __restrict__ output_rows,
    int64_t index,
    int64_t num_ranks,
    int64_t num_inputs,
    int lane,
    int lanes) {
  const auto source = kv_hbm_direct_source_row(
      miss_slots, miss_rank, index, num_ranks, num_inputs);
  auto* destination = output_rows + index * kKvDirectRowBytes;
  if (source >= 0) {
    kv_hbm_copy_packed_row(
        input_rows + source * kKvPackedRowBytes, destination, lane, lanes);
  } else if (source == kKvDirectBadMapping) {
    for (int byte = lane; byte < kKvDirectRowBytes; byte += lanes) {
      destination[byte] = 0;
    }
  }
}

__global__
__launch_bounds__(32 * kKvAdmitWarps) void kv_hbm_cache_pad_output_kernel(
    const uint8_t* __restrict__ input_rows,
    const int32_t* __restrict__ miss_slots,
    const int32_t* __restrict__ miss_rank,
    uint8_t* __restrict__ padded_rows,
    uint8_t* __restrict__ output_rows,
    int64_t num_misses,
    int64_t num_outputs,
    int64_t num_ranks,
    int64_t padding_blocks) {
  const auto lane = threadIdx.x;
  const auto warp = threadIdx.y;
  if (blockIdx.x < padding_blocks) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.y + warp;
         index < num_misses;
         index += padding_blocks * blockDim.y) {
      kv_hbm_copy_packed_row(
          input_rows + index * kKvPackedRowBytes,
          padded_rows + index * kKvDirectRowBytes,
          lane,
          blockDim.x);
    }
  } else {
    const auto output_blocks = gridDim.x - padding_blocks;
    for (int64_t index =
             (static_cast<int64_t>(blockIdx.x) - padding_blocks) * blockDim.y +
             warp;
         index < num_outputs;
         index += output_blocks * blockDim.y) {
      kv_hbm_write_direct_output(
          input_rows,
          miss_slots,
          miss_rank,
          output_rows,
          index,
          num_ranks,
          num_misses,
          lane,
          blockDim.x);
    }
  }
}

constexpr int kKvAdmitRows = 4;
constexpr int kKvAdmitCandidates = 128;
constexpr int kKvAdmitAbsent = 0;
constexpr int kKvAdmitTouched = 1;
constexpr int kKvAdmitAccepted = 2;
constexpr int kKvAdmitOverflow = 3;

__device__ __forceinline__ void kv_hbm_lru_compare_swap(
    int64_t* recency,
    int64_t* ways,
    bool* valid,
    unsigned int first,
    unsigned int second,
    bool direction) {
  const auto less = at::native::LTOp<int64_t, true>();
  const bool swap =
      (less(recency[first], recency[second]) && valid[first]) || !valid[second];
  if (swap == direction) {
    const auto key = recency[first];
    recency[first] = recency[second];
    recency[second] = key;
    const auto way = ways[first];
    ways[first] = ways[second];
    ways[second] = way;
    const auto is_valid = valid[first];
    valid[first] = valid[second];
    valid[second] = is_valid;
  }
}

// Matches ATen's 32-element bitonic compare-exchange network.
__device__ __forceinline__ void
kv_hbm_sort_lru(int64_t* recency, int64_t* ways, bool* valid) {
#pragma unroll
  for (unsigned int size = 2; size < 32; size *= 2) {
    const bool direction = (threadIdx.x & (size / 2)) != 0;
#pragma unroll
    for (unsigned int stride = size / 2; stride > 0; stride /= 2) {
      __syncthreads();
      const unsigned int first = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
      kv_hbm_lru_compare_swap(
          recency, ways, valid, first, first + stride, direction);
    }
  }
#pragma unroll
  for (unsigned int stride = 16; stride > 0; stride /= 2) {
    __syncthreads();
    const unsigned int first = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
    kv_hbm_lru_compare_swap(recency, ways, valid, first, first + stride, false);
  }
  __syncthreads();
}

__device__ bool kv_hbm_prepare_admission_set(
    int64_t set,
    int64_t generation,
    int64_t* epochs,
    int32_t* heads,
    int32_t* counts) {
  auto* epoch = reinterpret_cast<unsigned long long*>(epochs + set);
  const auto busy = static_cast<unsigned long long>(generation) * 2;
  const auto ready = busy + 1;
  auto observed = atomicAdd(epoch, 0ULL);
  while (observed != ready) {
    if (observed != busy && atomicCAS(epoch, observed, busy) == observed) {
      heads[set] = -1;
      counts[set] = 0;
      __threadfence();
      atomicExch(epoch, ready);
      return true;
    }
    observed = atomicAdd(epoch, 0ULL);
  }
  return false;
}

__global__
__launch_bounds__(32 * kKvAdmitWarps) void kv_hbm_cache_admission_probe_kernel(
    const int64_t* __restrict__ keys,
    const int64_t* __restrict__ tags,
    int64_t* __restrict__ epochs,
    int32_t* __restrict__ heads,
    int32_t* __restrict__ counts,
    int32_t* __restrict__ next,
    int32_t* __restrict__ touched,
    int32_t* __restrict__ status,
    int64_t num_keys,
    int64_t num_sets,
    int64_t assoc,
    int64_t generation) {
  __shared__ int first[kKvAdmitWarps];
  __shared__ int absent[kKvAdmitWarps];
  __shared__ int accepted[kKvAdmitWarps];
  __shared__ int overflow[kKvAdmitWarps];
  __shared__ int touched_base;
  const auto lane = threadIdx.x;
  const auto warp = threadIdx.y;
  const int64_t num_tiles = div_round_up(num_keys, kKvAdmitWarps);
  for (int64_t tile = blockIdx.x; tile < num_tiles; tile += gridDim.x) {
    const int64_t index = tile * kKvAdmitWarps + warp;
    const bool in_range = index < num_keys;
    const auto key = in_range ? keys[index] : 0;
    const bool valid = in_range && key != kKvHbmEmptyTag;
    const auto set = kv_hbm_cache_set(key, num_sets);
    const bool found = valid && lane < assoc && tags[set * assoc + lane] == key;
    const bool missing = valid && fbgemm_gpu::ballot_sync(found) == 0;
    if (lane == 0) {
      first[warp] = 0;
      absent[warp] = 0;
      accepted[warp] = 0;
      overflow[warp] = 0;
      if (missing) {
        first[warp] = kv_hbm_prepare_admission_set(
            set, generation, epochs, heads, counts);
        const auto rank = atomicAdd(counts + set, 1);
        const auto predecessor =
            atomicExch(heads + set, static_cast<int32_t>(index));
        next[index] = predecessor;
        absent[warp] = 1;
        accepted[warp] = rank < assoc;
        overflow[warp] = rank >= kKvAdmitCandidates;
      }
    }
    __syncthreads();
    if (lane == 0 && warp == 0) {
      int first_total = 0;
      int absent_total = 0;
      int accepted_total = 0;
      int overflow_total = 0;
      for (int i = 0; i < kKvAdmitWarps; ++i) {
        first_total += first[i];
        absent_total += absent[i];
        accepted_total += accepted[i];
        overflow_total |= overflow[i];
      }
      touched_base = atomicAdd(status + kKvAdmitTouched, first_total);
      atomicAdd(status + kKvAdmitAbsent, absent_total);
      atomicAdd(status + kKvAdmitAccepted, accepted_total);
      if (overflow_total) {
        atomicExch(status + kKvAdmitOverflow, 1);
      }
    }
    __syncthreads();
    if (lane == 0 && first[warp]) {
      int offset = 0;
      for (int i = 0; i < warp; ++i) {
        offset += first[i];
      }
      touched[touched_base + offset] = static_cast<int32_t>(set);
    }
    __syncthreads();
  }
}

template <bool DirectOutput>
__global__
__launch_bounds__(16 * kKvAdmitRows) void kv_hbm_cache_admission_write_kernel(
    const int64_t* __restrict__ keys,
    const uint8_t* __restrict__ input_rows,
    int64_t* __restrict__ tags,
    uint8_t* __restrict__ cache_rows,
    int64_t* __restrict__ lru,
    const int32_t* __restrict__ heads,
    const int32_t* __restrict__ next,
    const int32_t* __restrict__ touched,
    int64_t num_touched,
    int64_t assoc,
    int64_t row_bytes,
    int64_t tick,
    const int32_t* __restrict__ miss_slots,
    const int32_t* __restrict__ miss_rank,
    uint8_t* __restrict__ output_rows,
    int64_t num_outputs,
    int64_t num_ranks,
    int64_t num_inputs,
    int64_t cache_blocks) {
  __shared__ int32_t candidates[kKvAdmitRows][32];
  __shared__ int candidate_count[kKvAdmitRows];
  __shared__ int64_t recency[kKvAdmitRows][32];
  __shared__ int64_t ways[kKvAdmitRows][32];
  __shared__ bool valid_ways[kKvAdmitRows][32];
  const auto lane = threadIdx.x;
  const auto row = threadIdx.y;
  if constexpr (DirectOutput) {
    if (blockIdx.x >= cache_blocks) {
      const auto output_blocks = gridDim.x - cache_blocks;
      for (int64_t index =
               (static_cast<int64_t>(blockIdx.x) - cache_blocks) * blockDim.y +
               row;
           index < num_outputs;
           index += output_blocks * blockDim.y) {
        kv_hbm_write_direct_output(
            input_rows,
            miss_slots,
            miss_rank,
            output_rows,
            index,
            num_ranks,
            num_inputs,
            lane,
            blockDim.x);
      }
      return;
    }
  }
  const int64_t num_tiles = div_round_up(num_touched, kKvAdmitRows);
  for (int64_t tile = blockIdx.x; tile < num_tiles; tile += cache_blocks) {
    const int64_t index = tile * kKvAdmitRows + row;
    const bool row_valid = index < num_touched;
    const int64_t set = row_valid ? touched[index] : 0;
    if (lane == 0) {
      int count = 0;
      auto node = row_valid ? heads[set] : -1;
      for (int visited = 0; node >= 0 && visited < kKvAdmitCandidates;
           ++visited) {
        if (count < assoc || node < candidates[row][assoc - 1]) {
          int position = count < assoc ? count++ : assoc - 1;
          while (position > 0 && candidates[row][position - 1] > node) {
            candidates[row][position] = candidates[row][position - 1];
            --position;
          }
          candidates[row][position] = node;
        }
        node = next[node];
      }
      candidate_count[row] = count;
    }
    for (int i = 0; i < 2; ++i) {
      const auto way = lane + i * blockDim.x;
      const bool valid = row_valid && way < assoc;
      recency[row][way] = valid ? lru[set * assoc + way] : 0;
      ways[row][way] = valid ? way : 0;
      valid_ways[row][way] = valid;
    }
    kv_hbm_sort_lru(recency[row], ways[row], valid_ways[row]);
    for (int rank = 0; rank < candidate_count[row]; ++rank) {
      const auto source_index = candidates[row][rank];
      const auto slot = set * assoc + ways[row][rank];
      const auto* source = input_rows +
          static_cast<int64_t>(source_index) *
              (DirectOutput ? kKvPackedRowBytes : row_bytes);
      auto* destination = cache_rows + slot * row_bytes;
      constexpr int64_t kVectorBytes = sizeof(uint4);
      if constexpr (DirectOutput) {
        kv_hbm_copy_packed_row(source, destination, lane, blockDim.x);
      } else {
        if (row_bytes % kVectorBytes == 0 &&
            reinterpret_cast<uintptr_t>(source) % kVectorBytes == 0 &&
            reinterpret_cast<uintptr_t>(destination) % kVectorBytes == 0) {
          const auto* source_vectors = reinterpret_cast<const uint4*>(source);
          auto* destination_vectors = reinterpret_cast<uint4*>(destination);
          for (int64_t chunk = lane; chunk < row_bytes / kVectorBytes;
               chunk += blockDim.x) {
            destination_vectors[chunk] = source_vectors[chunk];
          }
        } else {
          for (int64_t byte = lane; byte < row_bytes; byte += blockDim.x) {
            destination[byte] = source[byte];
          }
        }
      }
      if (lane == 0) {
        tags[slot] = keys[source_index];
        lru[slot] = tick;
      }
    }
    __syncthreads();
  }
}

} // namespace

DLL_PUBLIC std::tuple<Tensor, Tensor> kv_hbm_cache_lookup_cuda(
    const Tensor& unique_keys,
    const Tensor& tags,
    const Tensor& cache_rows,
    const Tensor& lru,
    int64_t tick) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(unique_keys, tags, cache_rows, lru);
  TORCH_CHECK(
      unique_keys.dim() == 1 && unique_keys.scalar_type() == at::kLong &&
          unique_keys.is_contiguous(),
      "KV HBM keys must be contiguous one-dimensional int64 data");
  TORCH_CHECK(
      tags.dim() == 2 && tags.scalar_type() == at::kLong &&
          tags.is_contiguous() && tags.size(0) > 0 && tags.size(1) > 0 &&
          tags.size(1) <= kKvHbmMaxWays,
      "KV HBM tags must be contiguous int64 data with 1 to 32 ways");
  TORCH_CHECK(
      lru.sizes() == tags.sizes() && lru.scalar_type() == at::kLong &&
          lru.is_contiguous(),
      "KV HBM LRU data must match the tags");
  TORCH_CHECK(
      cache_rows.dim() == 2 && cache_rows.scalar_type() == at::kByte &&
          cache_rows.is_contiguous() &&
          cache_rows.size(0) == tags.size(0) * tags.size(1),
      "KV HBM rows must be contiguous uint8 data matching the cache capacity");
  CUDA_DEVICE_GUARD(unique_keys);
  const auto num_keys = unique_keys.numel();
  const auto row_bytes = cache_rows.size(1);
  auto output_rows = at::empty({num_keys, row_bytes}, cache_rows.options());
  auto hit_mask = at::empty({num_keys}, unique_keys.options().dtype(at::kBool));
  if (num_keys == 0) {
    return {output_rows, hit_mask};
  }
  const auto warps_per_block = kMaxThreads / kWarpSizeHost();
  const dim3 blocks(
      std::min<int64_t>(
          div_round_up(num_keys, warps_per_block),
          get_max_thread_blocks_for_cache_kernels_()));
  FBGEMM_LAUNCH_KERNEL(
      kv_hbm_cache_lookup_kernel,
      blocks,
      dim3(kWarpSizeHost(), warps_per_block),
      0,
      at::cuda::getCurrentCUDAStream(),
      unique_keys.const_data_ptr<int64_t>(),
      tags.const_data_ptr<int64_t>(),
      cache_rows.const_data_ptr<uint8_t>(),
      lru.data_ptr<int64_t>(),
      output_rows.data_ptr<uint8_t>(),
      hit_mask.data_ptr<bool>(),
      num_keys,
      tags.size(0),
      tags.size(1),
      row_bytes,
      tick);
  return {output_rows, hit_mask};
}

DLL_PUBLIC std::tuple<Tensor, Tensor, Tensor, Tensor, int64_t>
kv_hbm_cache_lookup_direct_cuda(
    const Tensor& keys,
    const Tensor& tags,
    const Tensor& cache_rows,
    const Tensor& lru,
    int64_t tick) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(keys, tags, cache_rows, lru);
  TORCH_CHECK(
      keys.dim() == 1 && keys.scalar_type() == at::kLong &&
          keys.is_contiguous() && keys.numel() <= kKvDirectMaxKeys,
      "KV HBM direct keys must be contiguous int64 data with at most 512K keys");
  TORCH_CHECK(
      tags.dim() == 2 && tags.scalar_type() == at::kLong &&
          tags.is_contiguous() && tags.size(0) > 0 && tags.size(1) > 0 &&
          tags.size(1) <= kKvHbmMaxWays,
      "KV HBM direct tags must be contiguous int64 data with 1 to 32 ways");
  TORCH_CHECK(
      lru.sizes() == tags.sizes() && lru.scalar_type() == at::kLong &&
          lru.is_contiguous(),
      "KV HBM direct LRU data must match the tags");
  TORCH_CHECK(
      cache_rows.dim() == 2 && cache_rows.scalar_type() == at::kByte &&
          cache_rows.is_contiguous() && cache_rows.size(0) == tags.numel() &&
          cache_rows.size(1) == kKvDirectRowBytes &&
          reinterpret_cast<uintptr_t>(cache_rows.const_data_ptr<uint8_t>()) %
                  sizeof(uint4) ==
              0,
      "KV HBM direct rows must be 16-byte-aligned contiguous uint8 80-byte rows");
#ifdef USE_ROCM
  TORCH_CHECK(false, "KV HBM direct lookup requires CUDA with 32-thread warps");
#endif
  CUDA_DEVICE_GUARD(keys);
  const auto num_keys = keys.numel();
  auto output_rows =
      at::empty({num_keys, kKvDirectRowBytes}, cache_rows.options());
  auto miss_slots = at::empty({num_keys}, keys.options().dtype(at::kInt));
  if (num_keys == 0) {
    return {
        output_rows,
        at::empty({0}, keys.options()),
        miss_slots,
        at::empty({0}, keys.options().dtype(at::kInt)),
        0};
  }
  int64_t table_capacity = 1;
  while (table_capacity < 2 * num_keys) {
    table_capacity *= 2;
  }
  const auto probe_blocks = std::min<int64_t>(
      div_round_up(num_keys, kKvDirectThreads / 32),
      get_max_thread_blocks_for_cache_kernels_());
  const auto stream = at::cuda::getCurrentCUDAStream();
  auto miss_keys_in = at::empty({num_keys}, keys.options());
  auto miss_keys_out = at::empty({num_keys}, keys.options());
  auto compact_slots = at::empty({num_keys}, keys.options().dtype(at::kInt));
  auto sorted_slots = at::empty({num_keys}, keys.options().dtype(at::kInt));
  size_t sort_bytes = 0;
  AT_CUDA_CHECK(
      FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceRadixSort::SortPairs(
          nullptr,
          sort_bytes,
          miss_keys_in.const_data_ptr<int64_t>(),
          miss_keys_out.data_ptr<int64_t>(),
          compact_slots.const_data_ptr<int32_t>(),
          sorted_slots.data_ptr<int32_t>(),
          num_keys,
          0,
          64,
          stream));
  constexpr int64_t kMaxScratchBytes = 64 * 1024 * 1024;
  TORCH_CHECK(
      sort_bytes + 8 * table_capacity + 28 * num_keys + 4 * probe_blocks + 8 <=
          kMaxScratchBytes,
      "KV HBM direct workspace exceeds 64 MiB");
  auto sort_temp = at::empty(
      {static_cast<int64_t>(sort_bytes)}, keys.options().dtype(at::kByte));
  auto miss_table =
      at::full({table_capacity}, -1, keys.options().dtype(at::kInt));
  auto miss_rank = at::empty({table_capacity}, keys.options().dtype(at::kInt));
  auto block_hits = at::empty({probe_blocks}, keys.options().dtype(at::kInt));
  auto status = at::zeros({2}, keys.options().dtype(at::kInt));
  auto status_cpu = at::empty(
      {2},
      at::TensorOptions().device(at::kCPU).dtype(at::kInt).pinned_memory(true));
  FBGEMM_LAUNCH_KERNEL(
      kv_hbm_cache_direct_probe_kernel,
      dim3(probe_blocks),
      dim3(kKvDirectThreads),
      0,
      stream,
      keys.const_data_ptr<int64_t>(),
      tags.const_data_ptr<int64_t>(),
      cache_rows.const_data_ptr<uint8_t>(),
      lru.data_ptr<int64_t>(),
      output_rows.data_ptr<uint8_t>(),
      miss_table.data_ptr<int32_t>(),
      miss_slots.data_ptr<int32_t>(),
      block_hits.data_ptr<int32_t>(),
      num_keys,
      tags.size(0),
      tags.size(1),
      table_capacity - 1,
      tick);
  const auto compact_blocks = std::min<int64_t>(
      div_round_up(table_capacity, kKvDirectThreads),
      get_max_thread_blocks_for_cache_kernels_());
  FBGEMM_LAUNCH_KERNEL(
      kv_hbm_cache_direct_compact_kernel,
      dim3(compact_blocks),
      dim3(kKvDirectThreads),
      0,
      stream,
      keys.const_data_ptr<int64_t>(),
      miss_table.const_data_ptr<int32_t>(),
      block_hits.const_data_ptr<int32_t>(),
      miss_keys_in.data_ptr<int64_t>(),
      compact_slots.data_ptr<int32_t>(),
      status.data_ptr<int32_t>(),
      table_capacity,
      probe_blocks);
  AT_CUDA_CHECK(cudaMemcpyAsync(
      status_cpu.data_ptr<int32_t>(),
      status.const_data_ptr<int32_t>(),
      2 * sizeof(int32_t),
      cudaMemcpyDeviceToHost,
      stream));
  AT_CUDA_CHECK(cudaStreamSynchronize(stream));
  const auto* observed = status_cpu.const_data_ptr<int32_t>();
  const auto num_misses = observed[1];
  if (num_misses > 0) {
    AT_CUDA_CHECK(
        FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceRadixSort::SortPairs(
            sort_temp.data_ptr<uint8_t>(),
            sort_bytes,
            miss_keys_in.const_data_ptr<int64_t>(),
            miss_keys_out.data_ptr<int64_t>(),
            compact_slots.const_data_ptr<int32_t>(),
            sorted_slots.data_ptr<int32_t>(),
            num_misses,
            0,
            64,
            stream));
    FBGEMM_LAUNCH_KERNEL(
        kv_hbm_cache_direct_rank_kernel,
        dim3(
            std::min<int64_t>(
                div_round_up(num_misses, kKvDirectThreads),
                get_max_thread_blocks_for_cache_kernels_())),
        dim3(kKvDirectThreads),
        0,
        stream,
        sorted_slots.const_data_ptr<int32_t>(),
        miss_rank.data_ptr<int32_t>(),
        num_misses);
  }
  return {
      output_rows,
      miss_keys_out.narrow(0, 0, num_misses),
      miss_slots,
      miss_rank,
      observed[0]};
}

namespace {

void kv_hbm_check_direct_output(
    const Tensor& input_rows,
    const Tensor& miss_slots,
    const Tensor& miss_rank,
    const Tensor& output_rows) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      input_rows, miss_slots, miss_rank, output_rows);
  TORCH_CHECK(
      input_rows.dim() == 2 && input_rows.scalar_type() == at::kByte &&
          input_rows.is_contiguous() &&
          input_rows.size(1) == kKvPackedRowBytes && output_rows.dim() == 2 &&
          output_rows.scalar_type() == at::kByte &&
          output_rows.is_contiguous() &&
          output_rows.size(1) == kKvDirectRowBytes &&
          output_rows.size(0) >= input_rows.size(0) && miss_slots.dim() == 1 &&
          miss_slots.scalar_type() == at::kInt && miss_slots.is_contiguous() &&
          miss_slots.numel() == output_rows.size(0) && miss_rank.dim() == 1 &&
          miss_rank.scalar_type() == at::kInt && miss_rank.is_contiguous() &&
          miss_rank.numel() >= input_rows.size(0),
      "KV HBM direct output requires packed 72-byte rows, 80-byte output rows, "
      "and int32 miss-slot and rank mappings");
}

} // namespace

DLL_PUBLIC Tensor kv_hbm_cache_pad_output_cuda(
    const Tensor& input_rows,
    const Tensor& miss_slots,
    const Tensor& miss_rank,
    const Tensor& output_rows) {
  kv_hbm_check_direct_output(input_rows, miss_slots, miss_rank, output_rows);
  CUDA_DEVICE_GUARD(input_rows);
  const auto num_misses = input_rows.size(0);
  auto padded_rows =
      at::empty({num_misses, kKvDirectRowBytes}, input_rows.options());
  if (num_misses == 0) {
    return padded_rows;
  }
  const auto padding_blocks = std::min<int64_t>(
      div_round_up(num_misses, kKvAdmitWarps),
      get_max_thread_blocks_for_cache_kernels_());
  const auto output_blocks = std::min<int64_t>(
      div_round_up(output_rows.size(0), kKvAdmitWarps),
      get_max_thread_blocks_for_cache_kernels_());
  FBGEMM_LAUNCH_KERNEL(
      kv_hbm_cache_pad_output_kernel,
      dim3(padding_blocks + output_blocks),
      dim3(32, kKvAdmitWarps),
      0,
      at::cuda::getCurrentCUDAStream(),
      input_rows.const_data_ptr<uint8_t>(),
      miss_slots.const_data_ptr<int32_t>(),
      miss_rank.const_data_ptr<int32_t>(),
      padded_rows.data_ptr<uint8_t>(),
      output_rows.data_ptr<uint8_t>(),
      num_misses,
      output_rows.size(0),
      miss_rank.numel(),
      padding_blocks);
  return padded_rows;
}

namespace {

std::tuple<int64_t, int64_t, bool, bool, bool, bool> kv_hbm_cache_admit_impl(
    const Tensor& keys,
    const Tensor& input_rows,
    const Tensor& tags,
    const Tensor& cache_rows,
    const Tensor& lru,
    const Tensor& epochs,
    const Tensor& heads,
    const Tensor& counts,
    const Tensor& next,
    const Tensor& touched,
    const Tensor* miss_slots,
    const Tensor* miss_rank,
    const Tensor* output_rows,
    int64_t tick,
    int64_t generation) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      keys,
      input_rows,
      tags,
      cache_rows,
      lru,
      epochs,
      heads,
      counts,
      next,
      touched);
  const bool direct = output_rows != nullptr;
  if (direct) {
    kv_hbm_check_direct_output(
        input_rows, *miss_slots, *miss_rank, *output_rows);
  }
  TORCH_CHECK(
      keys.dim() == 1 && keys.scalar_type() == at::kLong &&
          keys.is_contiguous(),
      "KV HBM admission keys must be contiguous one-dimensional int64 data");
  TORCH_CHECK(
      tags.dim() == 2 && tags.scalar_type() == at::kLong &&
          tags.is_contiguous() && tags.size(0) > 0 && tags.size(1) > 0,
      "KV HBM admission tags must be contiguous int64 data");
  TORCH_CHECK(
      lru.sizes() == tags.sizes() && lru.scalar_type() == at::kLong &&
          lru.is_contiguous(),
      "KV HBM admission LRU data must match the tags");
  TORCH_CHECK(
      cache_rows.dim() == 2 && cache_rows.scalar_type() == at::kByte &&
          cache_rows.is_contiguous() && cache_rows.size(0) == tags.numel() &&
          input_rows.dim() == 2 && input_rows.scalar_type() == at::kByte &&
          input_rows.is_contiguous() && input_rows.size(0) == keys.numel() &&
          input_rows.size(1) ==
              (direct ? kKvPackedRowBytes : cache_rows.size(1)) &&
          (!direct || cache_rows.size(1) == kKvDirectRowBytes),
      "KV HBM admission rows must match the input and cache capacity");
  CUDA_DEVICE_GUARD(keys);
  const auto num_keys = keys.numel();
  const auto num_sets = tags.size(0);
  const auto assoc = tags.size(1);
  if (num_keys == 0) {
    return {0, 0, false, false, false, false};
  }
  if (generation <= 0 || generation > std::numeric_limits<int64_t>::max() / 2) {
    return {0, 0, false, false, true, false};
  }
#ifdef USE_ROCM
  return {0, 0, false, false, false, true};
#endif
  if (assoc > kKvHbmMaxWays || num_keys > std::numeric_limits<int32_t>::max() ||
      num_sets > std::numeric_limits<int32_t>::max()) {
    return {0, 0, false, false, false, true};
  }
  if (num_keys <= kKvAdmitMinFusedKeys) {
    return {0, 0, true, false, false, false};
  }
  TORCH_CHECK(
      epochs.dim() == 1 && epochs.numel() == num_sets &&
          epochs.scalar_type() == at::kLong && epochs.is_contiguous() &&
          heads.dim() == 1 && heads.numel() == num_sets &&
          heads.scalar_type() == at::kInt && heads.is_contiguous() &&
          counts.dim() == 1 && counts.numel() == num_sets &&
          counts.scalar_type() == at::kInt && counts.is_contiguous() &&
          next.dim() == 1 && next.numel() == num_keys &&
          next.scalar_type() == at::kInt && next.is_contiguous() &&
          touched.dim() == 1 && touched.numel() == num_keys &&
          touched.scalar_type() == at::kInt && touched.is_contiguous(),
      "KV HBM admission scratch must match the sets and input capacity");
  auto status = at::zeros({4}, keys.options().dtype(at::kInt));
  auto status_cpu = at::empty(
      {4},
      at::TensorOptions().device(at::kCPU).dtype(at::kInt).pinned_memory(true));
  const auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 probe_blocks(
      std::min<int64_t>(
          div_round_up(num_keys, kKvAdmitWarps),
          get_max_thread_blocks_for_cache_kernels_()));
  FBGEMM_LAUNCH_KERNEL(
      kv_hbm_cache_admission_probe_kernel,
      probe_blocks,
      dim3(32, kKvAdmitWarps),
      0,
      stream,
      keys.const_data_ptr<int64_t>(),
      tags.const_data_ptr<int64_t>(),
      epochs.data_ptr<int64_t>(),
      heads.data_ptr<int32_t>(),
      counts.data_ptr<int32_t>(),
      next.data_ptr<int32_t>(),
      touched.data_ptr<int32_t>(),
      status.data_ptr<int32_t>(),
      num_keys,
      num_sets,
      assoc,
      generation);
  C10_CUDA_CHECK(cudaMemcpyAsync(
      status_cpu.data_ptr<int32_t>(),
      status.const_data_ptr<int32_t>(),
      4 * sizeof(int32_t),
      cudaMemcpyDeviceToHost,
      stream));
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  const auto* result = status_cpu.const_data_ptr<int32_t>();
  const int64_t absent = result[kKvAdmitAbsent];
  if (absent == 0 && !direct) {
    return {0, 0, false, false, false, false};
  }
  if (absent > 0 && absent <= kKvAdmitMinFusedKeys) {
    return {0, 0, true, false, false, false};
  }
  if (result[kKvAdmitOverflow]) {
    return {0, 0, false, true, false, false};
  }
  const int64_t num_touched = result[kKvAdmitTouched];
  const auto cache_blocks = std::min<int64_t>(
      div_round_up(num_touched, kKvAdmitRows),
      get_max_thread_blocks_for_cache_kernels_());
  if (direct) {
    const auto output_blocks = std::min<int64_t>(
        div_round_up(output_rows->size(0), kKvAdmitRows),
        get_max_thread_blocks_for_cache_kernels_());
    FBGEMM_LAUNCH_KERNEL(
        kv_hbm_cache_admission_write_kernel<true>,
        dim3(cache_blocks + output_blocks),
        dim3(16, kKvAdmitRows),
        0,
        stream,
        keys.const_data_ptr<int64_t>(),
        input_rows.const_data_ptr<uint8_t>(),
        tags.data_ptr<int64_t>(),
        cache_rows.data_ptr<uint8_t>(),
        lru.data_ptr<int64_t>(),
        heads.const_data_ptr<int32_t>(),
        next.const_data_ptr<int32_t>(),
        touched.const_data_ptr<int32_t>(),
        num_touched,
        assoc,
        cache_rows.size(1),
        tick,
        miss_slots->const_data_ptr<int32_t>(),
        miss_rank->const_data_ptr<int32_t>(),
        output_rows->data_ptr<uint8_t>(),
        output_rows->size(0),
        miss_rank->numel(),
        num_keys,
        cache_blocks);
  } else {
    FBGEMM_LAUNCH_KERNEL(
        kv_hbm_cache_admission_write_kernel<false>,
        dim3(cache_blocks),
        dim3(16, kKvAdmitRows),
        0,
        stream,
        keys.const_data_ptr<int64_t>(),
        input_rows.const_data_ptr<uint8_t>(),
        tags.data_ptr<int64_t>(),
        cache_rows.data_ptr<uint8_t>(),
        lru.data_ptr<int64_t>(),
        heads.const_data_ptr<int32_t>(),
        next.const_data_ptr<int32_t>(),
        touched.const_data_ptr<int32_t>(),
        num_touched,
        assoc,
        cache_rows.size(1),
        tick,
        nullptr,
        nullptr,
        nullptr,
        0,
        0,
        0,
        cache_blocks);
  }
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  const int64_t accepted = result[kKvAdmitAccepted];
  return {accepted, absent - accepted, false, false, false, false};
}

} // namespace

DLL_PUBLIC std::tuple<int64_t, int64_t, bool, bool, bool, bool>
kv_hbm_cache_admit_cuda(
    const Tensor& keys,
    const Tensor& input_rows,
    const Tensor& tags,
    const Tensor& cache_rows,
    const Tensor& lru,
    const Tensor& epochs,
    const Tensor& heads,
    const Tensor& counts,
    const Tensor& next,
    const Tensor& touched,
    int64_t tick,
    int64_t generation) {
  return kv_hbm_cache_admit_impl(
      keys,
      input_rows,
      tags,
      cache_rows,
      lru,
      epochs,
      heads,
      counts,
      next,
      touched,
      nullptr,
      nullptr,
      nullptr,
      tick,
      generation);
}

DLL_PUBLIC std::tuple<int64_t, int64_t, bool, bool, bool, bool>
kv_hbm_cache_admit_direct_cuda(
    const Tensor& keys,
    const Tensor& input_rows,
    const Tensor& tags,
    const Tensor& cache_rows,
    const Tensor& lru,
    const Tensor& epochs,
    const Tensor& heads,
    const Tensor& counts,
    const Tensor& next,
    const Tensor& touched,
    const Tensor& miss_slots,
    const Tensor& miss_rank,
    const Tensor& output_rows,
    int64_t tick,
    int64_t generation) {
  return kv_hbm_cache_admit_impl(
      keys,
      input_rows,
      tags,
      cache_rows,
      lru,
      epochs,
      heads,
      counts,
      next,
      touched,
      &miss_slots,
      &miss_rank,
      &output_rows,
      tick,
      generation);
}

DLL_PUBLIC int64_t host_lxu_cache_slot(int64_t h_in, int64_t C) {
  return static_cast<int64_t>(cache_slot(h_in, static_cast<int32_t>(C)));
}

namespace {

template <typename emb_t, typename cache_t>
__global__ __launch_bounds__(kMaxThreads) void lxu_cache_flush_kernel(
    pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> weights,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        cache_hash_size_cumsum,
    const pta::PackedTensorAccessor64<int32_t, 1, at::RestrictPtrTraits>
        cache_index_table_map,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        weights_offsets,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        D_offsets,
    const pta::PackedTensorAccessor32<int64_t, 2, at::RestrictPtrTraits>
        lxu_cache_state,
    pta::PackedTensorAccessor64<cache_t, 2, at::RestrictPtrTraits>
        lxu_cache_weights,
    bool stochastic_rounding,
    at::PhiloxCudaState stochastic_rounding_philox_args) {
  const int64_t B = lxu_cache_weights.size(0);
  for (int64_t b = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
       b < B;
       b += static_cast<int64_t>(gridDim.x) * blockDim.y) {
    const int32_t slot = static_cast<int32_t>(b % kWarpSize);
    const int32_t cache_set = static_cast<int32_t>(b / kWarpSize);
    const int64_t current_idx = lxu_cache_state[cache_set][slot];
    if (current_idx != static_cast<int64_t>(kCacheStateInvalid)) {
      // evict from slot to backing storage
      const int32_t t_current = cache_index_table_map[current_idx];
      const int64_t idx_current =
          current_idx - cache_hash_size_cumsum[t_current];
      const int64_t weights_offset_current = weights_offsets[t_current];
      const int32_t D_start_current = D_offsets[t_current];
      const int32_t D_end_current = D_offsets[t_current + 1];
      const int32_t D_current = D_end_current - D_start_current;

      int32_t D_emb = D_current;
      if constexpr (std::is_same_v<emb_t, uint8_t>) {
        D_emb += kINT8QparamsBytes;
      }

      auto weight_row = WeightRow<emb_t, cache_t, at::acc_type<cache_t, true>>(
          &weights[weights_offset_current + idx_current * D_emb + 0],
          &lxu_cache_weights[b][0],
          D_current,
          stochastic_rounding,
          &stochastic_rounding_philox_args,
          static_cast<uint64_t>(b) * blockDim.x + threadIdx.x);

      float2 qparams;
      if constexpr (std::is_same_v<emb_t, uint8_t>) {
        qparams =
            thrust_find_qparams<cache_t>(&lxu_cache_weights[b][0], D_current);
        if (threadIdx.x == 0) {
          weight_row.store_qparams(qparams);
        }
      }
      for (int64_t d = static_cast<int64_t>(threadIdx.x) * 4; d < D_current;
           d += blockDim.x * 4) {
        weight_row.evict_cache(d, qparams);
      }
    }
  }
}

} // namespace

DLL_PUBLIC void lxu_cache_flush_cuda(
    Tensor uvm_weights,
    Tensor cache_hash_size_cumsum,
    Tensor cache_index_table_map,
    Tensor weights_offsets,
    Tensor D_offsets,
    int64_t total_D,
    Tensor lxu_cache_state,
    Tensor lxu_cache_weights,
    bool stochastic_rounding) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      uvm_weights,
      cache_hash_size_cumsum,
      cache_index_table_map,
      weights_offsets,
      D_offsets,
      lxu_cache_state,
      lxu_cache_weights);

  CUDA_DEVICE_GUARD(lxu_cache_weights);

  const auto dispatch_uvm_weights =
      fbgemm_gpu::relabel_nfp8_for_dispatch(uvm_weights);

  const int32_t T = D_offsets.numel() - 1;
  TORCH_CHECK(
      T > 0,
      "lxu_cache_flush_cuda: D_offsets must contain at least 2 elements (got ",
      D_offsets.numel(),
      ")");
  const int64_t S = lxu_cache_weights.size(0);
  const int32_t tx = std::min<int32_t>(total_D / 4 / T, kMaxThreads);
  TORCH_CHECK(
      tx > 0,
      "lxu_cache_flush_cuda: total_D (",
      total_D,
      ") is too small relative to T (",
      T,
      "); need total_D >= 4");
  const dim3 threads(tx, kMaxThreads / tx);
  const uint64_t max_grid_x = static_cast<uint64_t>(
      at::cuda::getCurrentDeviceProperties()->maxGridSize[0]);
  const int64_t blocks_per_row = kMaxThreads / tx;
  const uint64_t requested_blocks =
      static_cast<uint64_t>((S + blocks_per_row - 1) / blocks_per_row);
  const uint64_t capped_blocks = std::min<uint64_t>(
      requested_blocks,
      std::min<uint64_t>(
          max_grid_x,
          static_cast<uint64_t>(get_max_thread_blocks_for_cache_kernels_())));
  const dim3 blocks(static_cast<uint32_t>(capped_blocks));

  fbgemm_gpu::dispatch_emb_cache_types(
      dispatch_uvm_weights.scalar_type(),
      lxu_cache_weights.scalar_type(),
      "lxu_cache_flush_kernel_2",
      ([&]<typename emb_t, typename cache_t>() {
        // Stochastic rounding is required only when emb_t and cache_t are
        // not the same type and emb_t is not float
        const bool stochastic_rounding_ = stochastic_rounding &&
            !std::is_same_v<emb_t, float> && !std::is_same_v<emb_t, cache_t>;

        at::PhiloxCudaState rng_engine_inputs;
        if (stochastic_rounding_) {
          auto gen = at::cuda::detail::getDefaultCUDAGenerator();
          std::lock_guard<std::mutex> lock(gen.mutex());
          rng_engine_inputs = at::check_generator<at::CUDAGeneratorImpl>(gen)
                                  ->philox_cuda_state(4);
        }

        FBGEMM_LAUNCH_KERNEL(
            (lxu_cache_flush_kernel<emb_t, cache_t>),
            blocks,
            threads,
            0,
            at::cuda::getCurrentCUDAStream(),
            PTA_B(dispatch_uvm_weights, emb_t, 1, 64),
            PTA_B(cache_hash_size_cumsum, int64_t, 1, 32),
            PTA_B(cache_index_table_map, int32_t, 1, 64),
            PTA_B(weights_offsets, int64_t, 1, 32),
            PTA_B(D_offsets, int32_t, 1, 32),
            PTA_B(lxu_cache_state, int64_t, 2, 32),
            PTA_B(lxu_cache_weights, cache_t, 2, 64),
            stochastic_rounding_,
            rng_engine_inputs);
      }));
}

namespace {

// count the number of times that a cache_slot appears in lxu_cache_locations
// we actually only care about whether the number is 0 or > 0.
__global__ __launch_bounds__(kMaxThreads) void lxu_cache_locations_count_kernel(
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        lxu_cache_locations,
    pta::PackedTensorAccessor32<int32_t, 2, at::RestrictPtrTraits> count,
    FixedDivisor fd) {
  const int32_t N = lxu_cache_locations.size(0);
  CUDA_KERNEL_LOOP(n, N) {
    if (lxu_cache_locations[n] >= 0) {
      int32_t cache_set;
      int32_t slot;
      fd.DivMod(lxu_cache_locations[n], &cache_set, &slot);
      atomicAdd(&count[cache_set][slot], 1);
    }
  }
}

// if a cache_slot is in lxu_cache_locations (count > 0),
// decrement the counter of that cache_slot.
__global__
__launch_bounds__(kMaxThreads) void lxu_cache_locking_counter_decrement_kernel(
    pta::PackedTensorAccessor32<int32_t, 2, at::RestrictPtrTraits>
        lxu_cache_locking_counter,
    pta::PackedTensorAccessor32<int32_t, 2, at::RestrictPtrTraits> count) {
  const int32_t C = lxu_cache_locking_counter.size(0);
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
       i < C;
       i += gridDim.x * blockDim.y) {
    const auto j = threadIdx.x;
    if (count[i][j] > 0) {
      atomicAdd(&lxu_cache_locking_counter[i][j], -1);
    }
  }
}

} // namespace

// for any cache_slot in lxu_cache_locations,
// decrement the counter of that cache_slot.
// duplicate cache_slot only decrement once.
void lxu_cache_locking_counter_decrement_cuda(
    at::Tensor lxu_cache_locking_counter,
    at::Tensor lxu_cache_locations) {
  TENSOR_ON_CUDA_GPU(lxu_cache_locking_counter);
  TENSOR_ON_CUDA_GPU(lxu_cache_locations);

  CUDA_DEVICE_GUARD(lxu_cache_locations);

  const auto N = lxu_cache_locations.numel();
  if (N == 0) {
    return;
  }

  auto count = at::zeros_like(lxu_cache_locking_counter);
  const int32_t C = lxu_cache_locking_counter.size(0);
  TORCH_CHECK(lxu_cache_locking_counter.size(1) == kWarpSizeHost());
  auto fd = FixedDivisor(kWarpSizeHost());

  const dim3 blocks(
      std::min(
          div_round_up(N, kMaxThreads),
          get_max_thread_blocks_for_cache_kernels_()));

  FBGEMM_LAUNCH_KERNEL(
      lxu_cache_locations_count_kernel,
      blocks,
      kMaxThreads,
      0,
      at::cuda::getCurrentCUDAStream(),
      PTA_B(lxu_cache_locations, int32_t, 1, 32),
      PTA_B(count, int32_t, 2, 32),
      fd);

  FBGEMM_LAUNCH_KERNEL(
      lxu_cache_locking_counter_decrement_kernel,
      std::min(
          div_round_up(C, kMaxThreads / kWarpSizeHost()),
          get_max_thread_blocks_for_cache_kernels_()),
      dim3(kWarpSizeHost(), kMaxThreads / kWarpSizeHost()),
      0,
      at::cuda::getCurrentCUDAStream(),
      PTA_B(lxu_cache_locking_counter, int32_t, 2, 32),
      PTA_B(count, int32_t, 2, 32));
}

namespace {

template <typename index_t>
__global__ __launch_bounds__(kMaxThreads) void lxu_cache_lookup_kernel(
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        linear_cache_indices,
    const pta::PackedTensorAccessor32<int64_t, 2, at::RestrictPtrTraits>
        lxu_cache_state,
    int64_t invalid_index,
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        lxu_cache_locations,
    const bool gather_cache_stats,
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        uvm_cache_stats,
    const int32_t* N_unique) {
  const int32_t C = lxu_cache_state.size(0);
  const int64_t N =
      N_unique == nullptr ? linear_cache_indices.size(0) : *N_unique;
  // On ROCm the launch caps the grid to stay within the HIP 2^32
  // threads-per-launch limit, so we grid-stride to cover the full workload.
  // On CUDA the grid is not capped and the loop body runs once per warp.
  const int64_t block_width = blockDim.x;
  const int64_t n_stride =
      static_cast<int64_t>(gridDim.x) * blockDim.y * block_width;
#ifdef USE_ROCM
  for (int64_t n0 =
           static_cast<int64_t>(blockIdx.x) * blockDim.y * block_width +
           static_cast<int64_t>(threadIdx.y) * block_width;
       n0 < N;
       n0 += n_stride) {
#else
  const int64_t n0 =
      static_cast<int64_t>(blockIdx.x) * blockDim.y * block_width +
      static_cast<int64_t>(threadIdx.y) * block_width;
  if (n0 >= N) {
    return;
  }
#endif

    int32_t cache_location = kCacheLocationMissing;
    int32_t n_indices = 0;
    int32_t n_hits = 0;
    const auto slot = threadIdx.x;
    for (int64_t i = 0; i < block_width; ++i) {
      const auto n = n0 + i;
      if (n >= N) {
        continue;
      }
      const int64_t idx = linear_cache_indices[n];
      if (idx == invalid_index) {
        continue;
      }
      const int32_t cache_set = cache_slot(idx, C);
      n_indices++;
      const bool found =
          (::__ldg((&lxu_cache_state[cache_set][0]) + slot) == idx);
      // fbgemm_gpu::ballot_sync wraps __ballot (ROCm, unsynchronized) vs
      // __ballot_sync (CUDA). Return type is uint64_t on ROCm and uint32_t
      // on CUDA, so __ffsll handles both via implicit promotion.
      const auto bitmap = fbgemm_gpu::ballot_sync(found);
      if (bitmap) {
        // LSB == 1 hence we need to subtract one to get lane ID.
        const auto way = __ffsll(static_cast<long long>(bitmap)) - 1;
        if (i == threadIdx.x) {
          cache_location = cache_set * kWarpSize + way;
        }
        n_hits++;
      }
    }

    const auto n = n0 + threadIdx.x;
    if (n < N) {
      lxu_cache_locations[n] = cache_location;
    }
    if (gather_cache_stats && threadIdx.x == 0 && n_indices > n_hits) {
      atomicAdd(
          &uvm_cache_stats[uvm_cache_stats_index::num_conflict_misses],
          (n_indices - n_hits));
    }
#ifdef USE_ROCM
  } // for n0 (grid-stride loop, ROCm only)
#endif
}

template <typename index_t>
__global__
__launch_bounds__(kMaxThreads) void direct_mapped_lxu_cache_lookup_kernel(
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        linear_cache_indices,
    const pta::PackedTensorAccessor32<int64_t, 2, at::RestrictPtrTraits>
        lxu_cache_state,
    int64_t invalid_index,
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        lxu_cache_locations,
    const bool gather_cache_stats,
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        uvm_cache_stats) {
  const int32_t C = lxu_cache_state.size(0);
  const int32_t N = linear_cache_indices.size(0);

  int32_t n_indices = 0;
  int32_t n_hits = 0;

  CUDA_KERNEL_LOOP(n, N) {
    int32_t cache_location = kCacheLocationMissing;
    const auto slot = 0;

    const int64_t idx = linear_cache_indices[n];
    if (idx == invalid_index) {
      lxu_cache_locations[n] = kCacheLocationMissing;
      continue;
    }

    const int32_t cache_set = cache_slot(idx, C);
    n_indices++;
    const bool found =
        (::__ldg((&lxu_cache_state[cache_set][0]) + slot) == idx);
    if (found) {
      cache_location = cache_set;
      n_hits++;
    }
    lxu_cache_locations[n] = cache_location;
  }

  if (gather_cache_stats) {
    typedef cub::BlockReduce<int32_t, kMaxThreads> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;

    const int32_t conflict_miss = n_indices - n_hits;
    const int32_t conflict_miss_sum = BlockReduce(temp).Sum(conflict_miss);

    if (threadIdx.x == 0) {
      atomicAdd(
          &uvm_cache_stats[uvm_cache_stats_index::num_conflict_misses],
          conflict_miss_sum);
    }
  }
}

} // namespace

/// Lookup the cache locations for each linear cache indices in
/// linear_cache_indices and return lxu_cache_locations
///
/// lxu_cache_locations A 1D tensor with the same length as
///                     linear_cache_indices.  It contains the cache locations
///                     (the row indices in the cache) of the corresponding
///                     indices in linear_cache_indices, i.e.,
///                     lxu_cache_locations[i] is the cache location for
///                     linear_cache_indices[i], where 0 <= i <
///                     linear_cache_indices.numel().
///
/// @param linear_cache_indices        Linear cache indices tensor (1D)
/// @param lxu_cache_state             LXU cache state tensor (2D tensor of
///                                    shape (# of cache sets, # of cache
///                                    slots per set)).  It contains linear
///                                    indices of rows that are in the
///                                    corresponding cache slots. If the cache
///                                    slot is empty, a sentinel value is
///                                    stored.
/// @param invalid_index               A sentinel value for linear cache
///                                    indices.  A cache index is skipped if it
///                                    is a sentinel value.
/// @param gather_cache_stats          A flag to enable/disable cache stats
///                                    collection.
/// @param uvm_cache_stats             A tensor for storing cache stats.
/// @param num_uniq_cache_indices      An optional GPU tensor that contains the
///                                    number of unique cache indices.  If this
///                                    tensor is passed, the kernel will only
///                                    lookup num_uniq_cache_indices number of
///                                    indices instead of looking up the entire
///                                    linear_cache_indices.
/// @param lxu_cache_locations_output  An optional output tensor.  If the
///                                    tensor is passed, the operator will not
///                                    allocate a new output tensor and use
///                                    this tensor as an output tensor.
DLL_PUBLIC Tensor lxu_cache_lookup_cuda(
    const Tensor linear_cache_indices,
    const Tensor lxu_cache_state,
    const int64_t invalid_index,
    const bool gather_cache_stats,
    const std::optional<Tensor> uvm_cache_stats,
    const std::optional<Tensor> num_uniq_cache_indices,
    const std::optional<Tensor> lxu_cache_locations_output) {
  const auto uniq_lookup = num_uniq_cache_indices.has_value();
  // TODO: Support gather_cache_stats=true when uniq_lookup=true
  TORCH_CHECK(
      !uniq_lookup || !gather_cache_stats,
      "Unique lxu_cache_locations generation does not support gather_cache_stats=true");
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      linear_cache_indices, lxu_cache_state, num_uniq_cache_indices);
  Tensor uvm_cache_stats_ =
      at::empty({0}, linear_cache_indices.options().dtype(at::kInt));
  if (gather_cache_stats) {
    TORCH_CHECK(uvm_cache_stats.has_value());
    uvm_cache_stats_ = uvm_cache_stats.value();
  }

  CUDA_DEVICE_GUARD(linear_cache_indices);

  const auto lxu_cache_locations =
      lxu_cache_locations_output.value_or(empty_like(
          linear_cache_indices,
          linear_cache_indices.options().dtype(at::kInt)));

  const auto N = linear_cache_indices.numel();
  if (linear_cache_indices.numel() == 0) {
    // nothing to do
    return lxu_cache_locations;
  }

  const dim3 threads(kWarpSizeHost(), kMaxThreads / kWarpSizeHost());
  // HIP enforces a hard limit of 2^32 total threads per launch;
  // lxu_cache_lookup_kernel grid-strides over n0, so capping is
  // correctness-preserving. See: https://github.com/ROCm/hip/issues/2253
  const dim3 blocks(
      utils::cuda::cap_grid_dim_x_from_workload(
          N, kMaxThreads, at::cuda::getCurrentCUDAStream()));

  AT_DISPATCH_INDEX_TYPES(
      linear_cache_indices.scalar_type(), "lxu_cache_lookup_cuda", [&] {
        FBGEMM_LAUNCH_KERNEL(
            (lxu_cache_lookup_kernel<index_t>),
            blocks,
            threads,
            0,
            at::cuda::getCurrentCUDAStream(),
            PTA_B(linear_cache_indices, index_t, 1, 32),
            PTA_B(lxu_cache_state, int64_t, 2, 32),
            invalid_index,
            PTA_B(lxu_cache_locations, int32_t, 1, 32),
            gather_cache_stats,
            PTA_B(uvm_cache_stats_, int32_t, 1, 32),
            num_uniq_cache_indices.has_value()
                ? num_uniq_cache_indices.value().data_ptr<int32_t>()
                : nullptr);
      });
  return lxu_cache_locations;
}

DLL_PUBLIC Tensor direct_mapped_lxu_cache_lookup_cuda(
    Tensor linear_cache_indices,
    Tensor lxu_cache_state,
    int64_t invalid_index,
    bool gather_cache_stats,
    std::optional<Tensor> uvm_cache_stats) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      linear_cache_indices, lxu_cache_state);
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(uvm_cache_stats, lxu_cache_state);

  if (gather_cache_stats) {
    TORCH_CHECK(uvm_cache_stats.has_value());
  }
  auto uvm_cache_stats_ = uvm_cache_stats.value_or(
      at::empty({0}, linear_cache_indices.options().dtype(at::kInt)));

  CUDA_DEVICE_GUARD(linear_cache_indices);

  const auto N = linear_cache_indices.numel();
  auto lxu_cache_locations = empty_like(
      linear_cache_indices, linear_cache_indices.options().dtype(at::kInt));
  if (linear_cache_indices.numel() == 0) {
    // nothing to do
    return lxu_cache_locations;
  }

  // HIP enforces a hard limit of 2^32 total threads per launch.
  // direct_mapped_lxu_cache_lookup_kernel uses CUDA_KERNEL_LOOP, which
  // already grid-strides over n, so capping is correctness-preserving.
  // See: https://github.com/ROCm/hip/issues/2253
  const dim3 blocks(
      utils::cuda::cap_grid_dim_x_from_workload(
          N, kMaxThreads, at::cuda::getCurrentCUDAStream()));

  AT_DISPATCH_INDEX_TYPES(
      linear_cache_indices.scalar_type(),
      "direct_mapped_lxu_cache_lookup_cuda",
      [&] {
        FBGEMM_LAUNCH_KERNEL(
            (direct_mapped_lxu_cache_lookup_kernel<index_t>),
            blocks,
            kMaxThreads,
            0,
            at::cuda::getCurrentCUDAStream(),
            PTA_B(linear_cache_indices, index_t, 1, 32),
            PTA_B(lxu_cache_state, int64_t, 2, 32),
            invalid_index,
            PTA_B(lxu_cache_locations, int32_t, 1, 32),
            gather_cache_stats,
            PTA_B(uvm_cache_stats_, int32_t, 1, 32));
      });

  return lxu_cache_locations;
}

namespace {

__global__
__launch_bounds__(kMaxThreads) void lxu_cache_locations_update_kernel(
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        lxu_cache_locations,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        lxu_cache_locations_new,
    const int32_t* N_unique) {
  const auto N = N_unique == nullptr ? lxu_cache_locations.size(0) : *N_unique;
  CUDA_KERNEL_LOOP(n, N) {
    if (N_unique != nullptr ||
        (lxu_cache_locations[n] == kCacheLocationMissing &&
         lxu_cache_locations_new[n] >= 0)) {
      lxu_cache_locations[n] = lxu_cache_locations_new[n];
    }
  }
}

} // namespace

DLL_PUBLIC void lxu_cache_locations_update_cuda(
    Tensor lxu_cache_locations,
    Tensor lxu_cache_locations_new,
    std::optional<Tensor> num_uniq_cache_indices) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      lxu_cache_locations, lxu_cache_locations_new, num_uniq_cache_indices);

  CUDA_DEVICE_GUARD(lxu_cache_locations);

  const auto N = lxu_cache_locations.numel();

  if (N == 0) {
    return;
  }

  const dim3 blocks(
      std::min(
          div_round_up(N, kMaxThreads),
          get_max_thread_blocks_for_cache_kernels_()));

  FBGEMM_LAUNCH_KERNEL(
      lxu_cache_locations_update_kernel,
      blocks,
      kMaxThreads,
      0,
      at::cuda::getCurrentCUDAStream(),
      PTA_B(lxu_cache_locations, int32_t, 1, 32),
      PTA_B(lxu_cache_locations_new, int32_t, 1, 32),
      num_uniq_cache_indices.has_value()
          ? num_uniq_cache_indices.value().data_ptr<int32_t>()
          : nullptr);

  return;
}
