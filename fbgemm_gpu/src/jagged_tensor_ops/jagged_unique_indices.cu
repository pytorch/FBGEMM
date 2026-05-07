/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "common.cuh"

// clang-format off
#include "fbgemm_gpu/utils/cub_namespace_prefix.cuh"
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_scan.cuh>
#include "fbgemm_gpu/utils/cub_namespace_postfix.cuh"
// clang-format on

#include "fbgemm_gpu/split_embeddings_utils.cuh"

#include <climits>
#include <cmath>

using Tensor = at::Tensor;

namespace fbgemm_gpu {

// Flat-grid linearize: one block per (t, b) sample. Replaces the old
// warp-cooperative linearize_index_wo_infos_kernel which launched only
// ⌈total_B/kMaxThreads⌉ blocks, severely underutilizing the SMs on
// large-N workloads. Writes the linearized index directly in the target
// key_t (int32 or int64), saving a subsequent cast pass.
template <typename index_t, typename key_t>
__global__ __launch_bounds__(256) void linearize_index_flat_kernel(
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        hash_size_cumsum,
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        indices,
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        offsets,
    pta::PackedTensorAccessor32<key_t, 1, at::RestrictPtrTraits> linear_indices,
    FixedDivisor fd) {
  const auto b_t = blockIdx.x;
  int32_t b;
  int32_t t;
  fd.DivMod(static_cast<int32_t>(b_t), &t, &b);

  const auto indices_start = offsets[b_t];
  const auto L = offsets[b_t + 1] - indices_start;
  if (L == 0) {
    return;
  }
  const auto hash_offset = hash_size_cumsum[t];

  for (auto i = threadIdx.x; i < L; i += blockDim.x) {
    const auto idx = __ldg(&indices[indices_start + i]);
    linear_indices[indices_start + i] = static_cast<key_t>(hash_offset + idx);
  }
}

// Adjacent-difference over sorted keys. Mirrors
// caffe2/aten/src/ATen/native/cuda/UniqueCub.cu:24-31.
// out[0] = 0; out[i] = (sorted[i] != sorted[i-1]) ? 1 : 0 for i > 0.
template <typename key_t>
__global__ __launch_bounds__(256) void jagged_unique_adjacent_diff_kernel(
    const pta::PackedTensorAccessor32<key_t, 1, at::RestrictPtrTraits>
        sorted_keys,
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> out) {
  const auto n = sorted_keys.size(0);
  const auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < static_cast<uint64_t>(n)) {
    out[i] = (i > 0 && sorted_keys[i] != sorted_keys[i - 1]) ? 1 : 0;
  }
}

// Scatter inv_loc_out into reverse_index via sorted_positions. Mirrors
// caffe2/aten/src/ATen/native/cuda/UniqueCub.cu:33-41. Output is int64
// to preserve the op's public contract (at::_unique returns int64 inverse).
__global__ __launch_bounds__(256) void jagged_unique_scatter_kernel(
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        sorted_positions,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        inv_loc_out,
    pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        reverse_index) {
  const auto n = sorted_positions.size(0);
  const auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < static_cast<uint64_t>(n)) {
    reverse_index[sorted_positions[i]] = static_cast<int64_t>(inv_loc_out[i]);
  }
}

// Device-side lower_bound over a PackedTensorAccessor32<index_t, 1>.
// Returns the first position whose value is >= `value`. Equivalent to
// std::lower_bound.
template <typename index_t>
__device__ __forceinline__ int32_t device_lower_bound(
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>& arr,
    const index_t value) {
  int32_t lo = 0;
  int32_t hi = static_cast<int32_t>(arr.size(0));
  while (lo < hi) {
    const int32_t mid = lo + ((hi - lo) >> 1);
    if (arr[mid] < value) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

// Feature-lookup delinearize: iterates over num_unique (~millions) instead
// of total_indices (~tens of millions). Each thread handles one sorted
// unique key and recovers its feature-local index via binary search over
// hash_size_cumsum. Replaces the old delinearize_unique_index_kernel's
// total_indices-sized scatter.
template <typename index_t, typename key_t>
__global__ __launch_bounds__(256) void delinearize_unique_from_sorted_kernel(
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        hash_size_cumsum,
    const pta::PackedTensorAccessor32<key_t, 1, at::RestrictPtrTraits>
        unique_keys,
    pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        unique_indices) {
  const auto num_unique = unique_keys.size(0);
  const auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= static_cast<uint64_t>(num_unique)) {
    return;
  }
  const index_t v = static_cast<index_t>(unique_keys[i]);
  // Largest t such that hash_size_cumsum[t] <= v, i.e. the feature that
  // owns this linearized value.
  const int32_t t = device_lower_bound<index_t>(hash_size_cumsum, v + 1) - 1;
  unique_indices[i] = v - hash_size_cumsum[t];
}

// Compute the lengths for each feature in the unique indices.
//
// Invariant leveraged: linearize_index_wo_infos_kernel prefixes every index
// of feature t with hash_size_cumsum[t], so linearized values of feature t
// lie in [hash_size_cumsum[t], hash_size_cumsum[t+1]). at::_unique is called
// with sorted=true, therefore entries of feature t occupy a single
// contiguous slice of `unique_indices`, namely
//   [lower_bound(unique_indices, hash_size_cumsum[t]),
//    lower_bound(unique_indices, hash_size_cumsum[t+1]))
// and the slice length equals the previous kernel's (max - min + 1) over
// reverse_index. This replaces an O(N) reduction over reverse_index with
// two O(log U) binary searches per feature group.
template <typename index_t>
__global__ __launch_bounds__(256) void unique_indices_length_kernel(
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        hash_size_offsets,
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        hash_size_cumsum,
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        unique_indices,
    pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> lengths,
    const int32_t batch_size) {
  const auto tid = threadIdx.x;
  const auto bid = blockIdx.x;

  const auto hash_begin = hash_size_offsets[bid];
  const auto hash_end = hash_size_offsets[bid + 1];
  const auto offset_begin = hash_begin * batch_size;
  const auto offset_end = hash_end * batch_size;
  const auto num_lengths = offset_end - offset_begin;
  if (num_lengths == 0) {
    return;
  }

  __shared__ index_t s_div_length;
  __shared__ index_t s_r_length;
  if (tid == 0) {
    const auto low = hash_size_cumsum[hash_begin];
    const auto high = hash_size_cumsum[hash_end];
    if (low == high) {
      // Empty feature group. Output is pre-zeroed; nothing to do.
      s_div_length = 0;
      s_r_length = 0;
    } else {
      const int32_t lo_pos = device_lower_bound<index_t>(unique_indices, low);
      const int32_t hi_pos = device_lower_bound<index_t>(unique_indices, high);
      const index_t total_length = static_cast<index_t>(hi_pos - lo_pos);
      s_div_length = total_length / static_cast<index_t>(num_lengths);
      s_r_length = total_length % static_cast<index_t>(num_lengths);
    }
  }
  __syncthreads();

  const index_t div_length = s_div_length;
  const index_t r_length = s_r_length;
  if (div_length == 0 && r_length == 0) {
    return;
  }
  for (int32_t i = tid; i < num_lengths; i += blockDim.x) {
    const index_t seg_length =
        (static_cast<index_t>(i) < r_length) ? (div_length + 1) : div_length;
    lengths[offset_begin + i] = seg_length;
  }
}

// Custom cub pipeline that replaces at::_unique: radix sort pairs (keys +
// arange positions), run-length encode, adjacent-diff + inclusive-scan +
// scatter to build the inverse index, then a feature-lookup delinearize
// over num_unique. Exposes linear_unique_indices as index_t so that
// unique_indices_length_kernel can be reused unchanged.
template <typename index_t, typename key_t>
static void jagged_unique_indices_pipeline(
    const Tensor& hash_size_cumsum,
    const Tensor& offsets,
    const Tensor& indices,
    const int total_hash_size_bits,
    const int64_t total_B,
    const int64_t T,
    Tensor& linear_unique_indices,
    Tensor& unique_indices,
    Tensor& reverse_index) {
  const int64_t N = indices.numel();
  auto stream = at::cuda::getCurrentCUDAStream();

  const auto key_opts = indices.options().dtype(
      std::is_same<key_t, int32_t>::value ? at::kInt : at::kLong);
  const auto int32_opts = indices.options().dtype(at::kInt);
  const auto int64_opts = indices.options().dtype(at::kLong);
  const auto byte_opts = indices.options().dtype(at::kByte);

  // --- Step F: flat-grid linearize, writes key_t directly ---
  Tensor linear_indices = at::empty({N}, key_opts);
  FBGEMM_LAUNCH_KERNEL(
      (linearize_index_flat_kernel<index_t, key_t>),
      total_B,
      256,
      0,
      stream,
      PTA_B(hash_size_cumsum, index_t, 1, 32),
      PTA_B(indices, index_t, 1, 32),
      PTA_B(offsets, index_t, 1, 32),
      PTA_B(linear_indices, key_t, 1, 32),
      FixedDivisor(static_cast<int32_t>(total_B / T)));

  // --- Step 3a: cub radix sort pairs with trimmed end_bit ---
  Tensor sorted_keys = at::empty({N}, key_opts);
  Tensor positions = at::arange(N, int32_opts);
  Tensor sorted_positions = at::empty({N}, int32_opts);
  {
    size_t temp_storage_bytes = 0;
    AT_CUDA_CHECK(radix_sort_pairs(
        nullptr,
        temp_storage_bytes,
        linear_indices.const_data_ptr<key_t>(),
        sorted_keys.data_ptr<key_t>(),
        positions.const_data_ptr<int32_t>(),
        sorted_positions.data_ptr<int32_t>(),
        static_cast<int>(N),
        0,
        total_hash_size_bits,
        stream));
    Tensor temp_storage =
        at::empty({static_cast<int64_t>(temp_storage_bytes)}, byte_opts);
    AT_CUDA_CHECK(radix_sort_pairs(
        temp_storage.data_ptr(),
        temp_storage_bytes,
        linear_indices.const_data_ptr<key_t>(),
        sorted_keys.data_ptr<key_t>(),
        positions.const_data_ptr<int32_t>(),
        sorted_positions.data_ptr<int32_t>(),
        static_cast<int>(N),
        0,
        total_hash_size_bits,
        stream));
  }

  // --- Step 3b: cub run-length encode to extract unique keys ---
  Tensor unique_keys = at::empty({N}, key_opts);
  Tensor run_lengths = at::empty({N}, int32_opts); // required but unused
  Tensor num_unique_d = at::empty({1}, int32_opts);
  {
    size_t temp_storage_bytes = 0;
    AT_CUDA_CHECK(
        FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceRunLengthEncode::Encode(
            nullptr,
            temp_storage_bytes,
            sorted_keys.const_data_ptr<key_t>(),
            unique_keys.data_ptr<key_t>(),
            run_lengths.data_ptr<int32_t>(),
            num_unique_d.data_ptr<int32_t>(),
            static_cast<int>(N),
            stream));
    Tensor temp_storage =
        at::empty({static_cast<int64_t>(temp_storage_bytes)}, byte_opts);
    AT_CUDA_CHECK(
        FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceRunLengthEncode::Encode(
            temp_storage.data_ptr(),
            temp_storage_bytes,
            sorted_keys.const_data_ptr<key_t>(),
            unique_keys.data_ptr<key_t>(),
            run_lengths.data_ptr<int32_t>(),
            num_unique_d.data_ptr<int32_t>(),
            static_cast<int>(N),
            stream));
  }
  const int32_t num_unique = num_unique_d.item<int32_t>();
  unique_keys = unique_keys.narrow(0, 0, num_unique);

  // --- Step 3c: inverse-index construction
  //     (adjacent_diff + inclusive_sum + scatter) ---
  Tensor inv_loc = at::empty({N}, int32_opts);
  FBGEMM_LAUNCH_KERNEL(
      (jagged_unique_adjacent_diff_kernel<key_t>),
      div_round_up(static_cast<int32_t>(N), 256),
      256,
      0,
      stream,
      PTA_B(sorted_keys, key_t, 1, 32),
      PTA_B(inv_loc, int32_t, 1, 32));

  Tensor inv_loc_out = at::empty({N}, int32_opts);
  {
    size_t temp_storage_bytes = 0;
    AT_CUDA_CHECK(
        FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceScan::InclusiveSum(
            nullptr,
            temp_storage_bytes,
            inv_loc.const_data_ptr<int32_t>(),
            inv_loc_out.data_ptr<int32_t>(),
            static_cast<int>(N),
            stream));
    Tensor temp_storage =
        at::empty({static_cast<int64_t>(temp_storage_bytes)}, byte_opts);
    AT_CUDA_CHECK(
        FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceScan::InclusiveSum(
            temp_storage.data_ptr(),
            temp_storage_bytes,
            inv_loc.const_data_ptr<int32_t>(),
            inv_loc_out.data_ptr<int32_t>(),
            static_cast<int>(N),
            stream));
  }

  reverse_index = at::empty({N}, int64_opts);
  FBGEMM_LAUNCH_KERNEL(
      (jagged_unique_scatter_kernel),
      div_round_up(static_cast<int32_t>(N), 256),
      256,
      0,
      stream,
      PTA_B(sorted_positions, int32_t, 1, 32),
      PTA_B(inv_loc_out, int32_t, 1, 32),
      PTA_B(reverse_index, int64_t, 1, 32));

  // --- Step E: feature-lookup delinearize over num_unique ---
  unique_indices = at::empty({num_unique}, indices.options());
  if (num_unique > 0) {
    FBGEMM_LAUNCH_KERNEL(
        (delinearize_unique_from_sorted_kernel<index_t, key_t>),
        div_round_up(num_unique, 256),
        256,
        0,
        stream,
        PTA_B(hash_size_cumsum, index_t, 1, 32),
        PTA_B(unique_keys, key_t, 1, 32),
        PTA_B(unique_indices, index_t, 1, 32));
  }

  // Provide linear_unique_indices as index_t for unique_indices_length_kernel.
  if (std::is_same<key_t, index_t>::value) {
    linear_unique_indices = unique_keys;
  } else {
    linear_unique_indices = unique_keys.to(indices.scalar_type());
  }
}

// Preconditions enforced by the TORCH_CHECKs at the top of
// jagged_unique_indices_cuda; trace each back to its downstream consumer.
//
//  - hash_size_cumsum.dtype() == indices.dtype()
//      Required so AT_DISPATCH_INDEX_TYPES on indices.scalar_type() can
//      bind a single index_t for both tensors inside the pipeline.
//
//  - hash_size_cumsum.numel() >= 2 (i.e. T = numel - 1 >= 1)
//      T feeds three consumers that assume T >= 1:
//        1. total_B / T — integer division below; UB if T == 0.
//        2. FixedDivisor(total_B / T) — passed to
//           linearize_index_flat_kernel to recover (t, b) from the flat
//           block index; FixedDivisor with divisor 0 is undefined.
//        3. unique_indices_length_kernel launches T blocks and reads
//           hash_size_cumsum[bid+1], so it requires numel >= T + 1 = 2.
//      We also dereference hash_size_cumsum[numel-1] just below to read
//      total_hash_size, which independently needs numel >= 1.
//
//  - N = indices.numel() <= INT32_MAX
//      N feeds three int32-typed consumers inside
//      jagged_unique_indices_pipeline:
//        1. positions = at::arange(N, int32_opts) — int32 value payload
//           of the radix sort, must enumerate [0, N) in int32.
//        2. sorted_positions / inv_loc / inv_loc_out — int32 scratch
//           tensors sized N, indexed in [0, N).
//        3. static_cast<int>(N) passed as num_items to
//           cub::DeviceRadixSort, DeviceRunLengthEncode, and DeviceScan
//           (FBGEMM's radix_sort_pairs wrapper takes `int num_items`;
//           CUB's APIs likewise).
//      If N exceeds INT32_MAX we would silently wrap and produce garbage
//      sorts/uniques rather than fail.
//
//  - total_hash_size >= 0 and total_hash_size_bits <= 63
//      Bounds the radix-sort end_bit; >63 would overflow CUB's pass count.
std::tuple<Tensor, Tensor, Tensor, Tensor> jagged_unique_indices_cuda(
    const Tensor& hash_size_cumsum,
    const Tensor& hash_size_offsets,
    const Tensor& offsets,
    const Tensor& indices) {
  TORCH_CHECK(hash_size_cumsum.dtype() == indices.dtype());
  TORCH_CHECK(
      hash_size_cumsum.numel() >= 2,
      "jagged_unique_indices: hash_size_cumsum must have at least 2 entries "
      "(T >= 1), got ",
      hash_size_cumsum.numel());

  const auto total_B = offsets.size(0) - 1;
  const auto T = hash_size_cumsum.size(0) - 1;
  const auto N = indices.numel();
  TORCH_CHECK(
      N <= static_cast<int64_t>(INT32_MAX),
      "jagged_unique_indices: indices.numel() (",
      N,
      ") exceeds INT32_MAX");

  // One D→H sync to get total_hash_size for radix-sort end_bit trimming.
  // at::_unique historically already did an implicit sync here for output
  // allocation, so this is net-neutral.
  const int64_t total_hash_size =
      hash_size_cumsum[hash_size_cumsum.numel() - 1].item().toLong();
  TORCH_CHECK(total_hash_size >= 0);
  const int total_hash_size_bits = (total_hash_size > 0)
      ? static_cast<int>(
            std::log2(static_cast<double>(total_hash_size + 1)) + 1)
      : 1;
  TORCH_CHECK(total_hash_size_bits <= 63);
  // int32 key path is safe when total_hash_size fits in int32: linearized
  // values are bounded by total_hash_size - 1 < INT32_MAX.
  const bool use_int32_keys = total_hash_size < static_cast<int64_t>(INT32_MAX);

  Tensor unique_indices;
  Tensor reverse_index;
  Tensor linear_unique_indices;

  if (N == 0) {
    unique_indices = at::empty({0}, indices.options());
    reverse_index = at::empty({0}, indices.options().dtype(at::kLong));
    linear_unique_indices = at::empty({0}, indices.options());
  } else {
    AT_DISPATCH_INDEX_TYPES(
        indices.scalar_type(), "jagged_unique_indices_pipeline", ([&] {
          if (use_int32_keys) {
            jagged_unique_indices_pipeline<index_t, int32_t>(
                hash_size_cumsum,
                offsets,
                indices,
                total_hash_size_bits,
                total_B,
                T,
                linear_unique_indices,
                unique_indices,
                reverse_index);
          } else {
            jagged_unique_indices_pipeline<index_t, index_t>(
                hash_size_cumsum,
                offsets,
                indices,
                total_hash_size_bits,
                total_B,
                T,
                linear_unique_indices,
                unique_indices,
                reverse_index);
          }
        }));
  }

  Tensor output_lengths = at::zeros({total_B}, offsets.options());
  AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "unique_indices_length", ([&] {
                            FBGEMM_LAUNCH_KERNEL(
                                (unique_indices_length_kernel<index_t>),
                                T,
                                256,
                                0,
                                at::cuda::getCurrentCUDAStream(),
                                PTA_B(hash_size_offsets, index_t, 1, 32),
                                PTA_B(hash_size_cumsum, index_t, 1, 32),
                                PTA_B(linear_unique_indices, index_t, 1, 32),
                                PTA_B(output_lengths, index_t, 1, 32),
                                static_cast<int32_t>(total_B / T));
                          }));

  Tensor output_offsets = asynchronous_complete_cumsum_gpu(output_lengths);
  return {output_lengths, output_offsets, unique_indices, reverse_index};
}

// Compute hash size for each key using the max value of indices per key.
template <typename index_t, auto min_value>
__global__ __launch_bounds__(kMaxThreads) void compute_hash_size_kernel(
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        offsets,
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        indices,
    const int64_t batch_size,
    pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> hash_size) {
  typedef cub::BlockReduce<index_t, kMaxThreads> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage_max;

  const auto tid = threadIdx.x;
  const auto bid = blockIdx.x;

  const auto offset_begin = bid * batch_size;
  const auto offset_end = (bid + 1) * batch_size;
  const auto index_begin = offsets[offset_begin];
  const auto index_end = offsets[offset_end];

  if (index_begin == index_end) {
    return;
  }

  index_t t_max = min_value;
  for (index_t i = (index_begin + tid); i < index_end; i += kMaxThreads) {
    const index_t value = indices[i];
    t_max = (value > t_max) ? value : t_max;
  }

  index_t block_max =
      BlockReduce(temp_storage_max).Reduce(t_max, Max<index_t>());
  if (tid == 0) {
    hash_size[bid] = block_max + 1;
  }
}

std::tuple<Tensor, Tensor> jagged_hash_size_cumsum_cuda(
    const Tensor& offsets,
    const Tensor& indices,
    const int64_t batch_size) {
  const auto T = (offsets.size(0) - 1) / batch_size;
  Tensor hash_size = at::zeros({T}, offsets.options());

  using at::RestrictPtrTraits;

  AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "compute_hash_size", ([&] {
                            FBGEMM_LAUNCH_KERNEL(
                                (compute_hash_size_kernel<
                                    index_t,
                                    std::numeric_limits<index_t>::min()>),
                                T,
                                kMaxThreads,
                                0,
                                at::cuda::getCurrentCUDAStream(),
                                PTA_B(offsets, index_t, 1, 32),
                                PTA_B(indices, index_t, 1, 32),
                                batch_size,
                                PTA_B(hash_size, index_t, 1, 32));
                          }));

  Tensor hash_size_cumsum;
  hash_size_cumsum = asynchronous_complete_cumsum_gpu(hash_size);

  Tensor hash_size_lengths = at::ones_like(hash_size);
  Tensor hash_size_offsets;
  hash_size_offsets = asynchronous_complete_cumsum_gpu(hash_size_lengths);
  return {hash_size_cumsum, hash_size_offsets};
}

// Optimized atomic kernel with better memory access patterns
template <typename index_t, typename scalar_t>
__global__
__launch_bounds__(kMaxThreads) void accumulate_weights_and_counts_kernel(
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        reverse_indices,
    const pta::PackedTensorAccessor32<scalar_t, 1, at::RestrictPtrTraits>
        weights,
    pta::PackedTensorAccessor32<float, 2, at::RestrictPtrTraits>
        accumulated_data) {
  const auto tid = threadIdx.x;
  const auto bid = blockIdx.x;
  const auto total_elements = weights.size(0);

  // Process elements with stride of kMaxThreads for better memory
  // bandwidth utilization
  for (int i = bid * kMaxThreads + tid; i < total_elements;
       i += kMaxThreads * gridDim.x) {
    const index_t unique_idx = reverse_indices[i];
    const scalar_t weight_val = weights[i];

    // Use fast atomic operations
    atomicAdd(&accumulated_data[unique_idx][0], static_cast<float>(weight_val));
    atomicAdd(&accumulated_data[unique_idx][1], 1.0f);
  }
}

// Optimized function to accumulate weights and counts using atomic operations
// Simplified approach that focuses on memory bandwidth and atomic efficiency
Tensor jagged_acc_weights_and_counts_cu(
    const Tensor& weights,
    const Tensor& reverse_indices,
    int64_t num_unique_indices) {
  // Create 2D tensor: [num_unique_indices, 2] where dim 0 = accumulated
  // weights, dim 1 = counts
  Tensor accumulated_data = at::zeros(
      {num_unique_indices, 2},
      at::TensorOptions().dtype(at::kFloat).device(weights.device()));

  const auto total_elements = weights.size(0);

  // Use optimized atomic approach - simpler and often faster than segmented
  // reduction for this use case due to reduced overhead
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      weights.scalar_type(), "accumulate_weights_and_counts", ([&] {
        AT_DISPATCH_INDEX_TYPES(
            reverse_indices.scalar_type(),
            "accumulate_weights_and_counts_idx",
            ([&] {
              // Calculate number of blocks based on total elements
              const int num_blocks = div_round_up(total_elements, kMaxThreads);

              FBGEMM_LAUNCH_KERNEL(
                  (accumulate_weights_and_counts_kernel<index_t, scalar_t>),
                  num_blocks,
                  kMaxThreads,
                  0,
                  at::cuda::getCurrentCUDAStream(),
                  PTA_B(reverse_indices, index_t, 1, 32),
                  PTA_B(weights, scalar_t, 1, 32),
                  PTA_B(accumulated_data, float, 2, 32));
            }));
      }));

  return accumulated_data;
}

} // namespace fbgemm_gpu

FBGEMM_OP_DISPATCH(
    CUDA,
    "jagged_unique_indices",
    fbgemm_gpu::jagged_unique_indices_cuda);

FBGEMM_OP_DISPATCH(
    CUDA,
    "jagged_hash_size_cumsum",
    fbgemm_gpu::jagged_hash_size_cumsum_cuda);

FBGEMM_OP_DISPATCH(
    CUDA,
    "jagged_acc_weights_and_counts",
    fbgemm_gpu::jagged_acc_weights_and_counts_cu);
