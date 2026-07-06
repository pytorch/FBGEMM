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

using Tensor = at::Tensor;

namespace fbgemm_gpu {

// Block size for the flat-grid kernels in this file. 256 = 8 warps, chosen
// so the SM scheduler can pack ~4-8 blocks per SM on H100 given the
// flat-grid launches with grid = total_B (or num_unique).
static constexpr int32_t kFlatBlockSize = 256;

// Linearize the index with the cumsum of hash size so that linearized indices
// can be sorted together. Flat-grid: one block per (t, b) sample.
//
// Replaces the prior warp-cooperative kernel which was launched as
//   grid = ceil(total_B / kMaxThreads)
// On production shapes total_B is in the low thousands and kMaxThreads = 1024,
// so the prior launch consumed only ~5 SMs out of 132 on H100 with each warp
// shuffling work between lanes. The flat grid uses one block per sample,
// dispatching all SMs and removing the intra-warp shuffle dance.
//
// Contract: indices[i] must satisfy `idx < per_feature_hash_size[t]` for any
// feature `t < T - 1` (intermediate features). Violating this causes
// `hash_offset + idx` to overflow into feature `t + 1`'s hash range, which
// silently mis-attributes the value's count in unique_indices_length_kernel
// (counts leak from t to t+1; total is preserved). The last feature is
// explicitly exempt: its OOB tail is supported via the clamp in the length
// kernel and the scatter-form delinearize. The assert below catches the
// intermediate-feature case in debug builds; CUDA_KERNEL_ASSERT also fires
// in release builds (by design — see torch/headeronly/macros/Macros.h).
template <typename index_t>
__global__ __launch_bounds__(kFlatBlockSize) void linearize_index_flat_kernel(
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        hash_size_cumsum,
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        indices,
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        offsets,
    pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        linear_indices,
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
  // OOB check exemptions:
  //   - is_last_feature: the last feature's OOB tail is explicitly supported.
  //   - is_merged_feature (per_feature_hash == 0): the caller is grouping
  //     this feature with another via the hash_size_offsets indirection;
  //     consecutive-zero entries in hash_size_cumsum mean the two features
  //     share the same hash space, so any local index value is valid.
  const bool is_last_feature = (t == hash_size_cumsum.size(0) - 2);
  const auto per_feature_hash =
      is_last_feature ? index_t{0} : (hash_size_cumsum[t + 1] - hash_offset);
  const bool is_merged_feature = !is_last_feature && (per_feature_hash == 0);

  for (auto i = threadIdx.x; i < L; i += blockDim.x) {
    const auto idx = __ldg(&indices[indices_start + i]);
    CUDA_KERNEL_ASSERT(
        is_last_feature || is_merged_feature || idx < per_feature_hash);
    linear_indices[indices_start + i] = hash_offset + idx;
  }
}

// Adjacent-difference over sorted keys. out[0] = 0; out[i > 0] = 1 if
// sorted[i] != sorted[i-1] else 0. Mirrors
// caffe2/aten/src/ATen/native/cuda/UniqueCub.cu.
template <typename index_t>
__global__
__launch_bounds__(kFlatBlockSize) void jagged_unique_adjacent_diff_kernel(
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        sorted_keys,
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> out) {
  const auto n = sorted_keys.size(0);
  const auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < static_cast<uint64_t>(n)) {
    out[i] = (i > 0 && sorted_keys[i] != sorted_keys[i - 1]) ? 1 : 0;
  }
}

// Scatter inv_loc_out[i] -> reverse_index[sorted_positions[i]] to recover the
// inverse-index in the original input order. Output is int64 to preserve the
// public contract of jagged_unique_indices (matches at::_unique's historical
// inverse-index dtype).
__global__ __launch_bounds__(kFlatBlockSize) void jagged_unique_scatter_kernel(
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

// Cub-based replacement for at::_unique(linear_indices, sorted=true,
// return_inverse=true). Returns sorted unique linearized keys and an
// int64 inverse-index in the original input order.
//
// Pipeline:
//   1. cub::DeviceRadixSort::SortPairs on (linear_indices, arange-positions)
//      with end_bit trimmed to the bit-width of the maximum linearized key,
//      cutting radix-sort passes from ceil(64/8)=8 to ceil(bit_width/8) (e.g.
//      3 passes for hash_size=1M). end_bit must cover the real max key (not
//      just total_hash_size) so last-feature OOB keys are fully ordered; see
//      the caller for why.
//   2. cub::DeviceRunLengthEncode::Encode to extract the sorted unique keys.
//   3. adjacent_diff -> inclusive_scan -> scatter to recover the inverse
//      index in the original (pre-sort) input order, mirroring pytorch's
//      UniqueCub.cu pattern.
template <typename index_t>
static void jagged_unique_indices_pipeline(
    const Tensor& linear_indices,
    const int end_bit,
    Tensor& linear_unique_indices,
    Tensor& reverse_index) {
  const int64_t N = linear_indices.numel();
  auto stream = at::cuda::getCurrentCUDAStream();

  const auto int32_opts = linear_indices.options().dtype(at::kInt);
  const auto int64_opts = linear_indices.options().dtype(at::kLong);
  const auto byte_opts = linear_indices.options().dtype(at::kByte);

  // --- Step 1: radix sort pairs ---
  Tensor sorted_keys = at::empty({N}, linear_indices.options());
  Tensor positions = at::arange(N, int32_opts);
  Tensor sorted_positions = at::empty({N}, int32_opts);
  {
    size_t temp_storage_bytes = 0;
    AT_CUDA_CHECK(radix_sort_pairs(
        nullptr,
        temp_storage_bytes,
        linear_indices.const_data_ptr<index_t>(),
        sorted_keys.data_ptr<index_t>(),
        positions.const_data_ptr<int32_t>(),
        sorted_positions.data_ptr<int32_t>(),
        static_cast<int>(N),
        0,
        end_bit,
        stream));
    Tensor temp_storage =
        at::empty({static_cast<int64_t>(temp_storage_bytes)}, byte_opts);
    AT_CUDA_CHECK(radix_sort_pairs(
        temp_storage.data_ptr(),
        temp_storage_bytes,
        linear_indices.const_data_ptr<index_t>(),
        sorted_keys.data_ptr<index_t>(),
        positions.const_data_ptr<int32_t>(),
        sorted_positions.data_ptr<int32_t>(),
        static_cast<int>(N),
        0,
        end_bit,
        stream));
  }

  // --- Step 2: run-length encode to extract sorted unique keys ---
  Tensor unique_keys = at::empty({N}, linear_indices.options());
  Tensor run_lengths = at::empty({N}, int32_opts);
  Tensor num_unique_d = at::empty({1}, int32_opts);
  {
    size_t temp_storage_bytes = 0;
    AT_CUDA_CHECK(
        FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceRunLengthEncode::Encode(
            nullptr,
            temp_storage_bytes,
            sorted_keys.const_data_ptr<index_t>(),
            unique_keys.data_ptr<index_t>(),
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
            sorted_keys.const_data_ptr<index_t>(),
            unique_keys.data_ptr<index_t>(),
            run_lengths.data_ptr<int32_t>(),
            num_unique_d.data_ptr<int32_t>(),
            static_cast<int>(N),
            stream));
  }
  const int32_t num_unique = num_unique_d.item<int32_t>();
  linear_unique_indices = unique_keys.narrow(0, 0, num_unique);

  // --- Step 3: build inverse-index (adjacent_diff + inclusive_scan + scatter)
  // ---
  Tensor inv_loc = at::empty({N}, int32_opts);
  FBGEMM_LAUNCH_KERNEL(
      (jagged_unique_adjacent_diff_kernel<index_t>),
      static_cast<int32_t>((N + kFlatBlockSize - 1) / kFlatBlockSize),
      kFlatBlockSize,
      0,
      stream,
      PTA_B(sorted_keys, index_t, 1, 32),
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
      static_cast<int32_t>((N + kFlatBlockSize - 1) / kFlatBlockSize),
      kFlatBlockSize,
      0,
      stream,
      PTA_B(sorted_positions, int32_t, 1, 32),
      PTA_B(inv_loc_out, int32_t, 1, 32),
      PTA_B(reverse_index, int64_t, 1, 32));
}

// Device-side lower_bound over a PackedTensorAccessor32<index_t, 1>.
// Returns the first position whose value is >= `value`, equivalent to
// std::lower_bound on the underlying sorted array.
template <typename index_t>
__device__ __forceinline__ int32_t device_lower_bound(
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>& arr,
    const index_t value) {
  int32_t lo = 0;
  int32_t hi = arr.size(0);
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

// Delinearize the unique indices via scatter over the original input indices.
// For each input position i, write the original (pre-linearization) per-feature
// index value into unique_indices at the position designated by reverse_index.
//
// This is index-bound-agnostic by construction: the original indices[i] value
// is preserved exactly, regardless of whether it exceeds the per-feature
// hash_size. A gather form that recovers the per-feature index from the
// linearized key (v - hash_size_cumsum[t]) misattributes out-of-bound values
// whose linearized form exceeds hash_size_cumsum[T] to a phantom feature
// index and loses the original value.
template <typename index_t>
__global__
__launch_bounds__(kFlatBlockSize) void delinearize_unique_index_kernel(
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        indices,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        reverse_index,
    pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        unique_indices) {
  const auto total_indices = indices.size(0);
  const auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < static_cast<uint64_t>(total_indices)) {
    unique_indices[reverse_index[i]] = indices[i];
  }
}

// Compute the per-(feature, batch) lengths for the unique indices.
//
// Caller-provided invariant (see jagged_unique_indices_cuda for the
// pipeline contract that establishes it): `linear_unique_indices` is
// sorted ascending, and feature t's values occupy a contiguous slice
//   [lower_bound(linear_unique_indices, hash_size_cumsum[t]),
//    lower_bound(linear_unique_indices, hash_size_cumsum[t+1])).
// The slice length equals num_unique_t for feature t.
//
// Out-of-bound carve-out: when the last feature has indices exceeding its
// per-feature hash_size, their linearized form `hash_size_cumsum[T-1] + idx`
// can exceed hash_size_cumsum[T] and sort past the upper-bound binary
// search. For the last feature group only, clamp hi_pos to
// linear_unique_indices.size(0) so those tail values are still counted.
// This also covers the degenerate case where the last feature has size 0
// (consecutive equal entries at the tail of hash_size_cumsum): low == high
// would otherwise short-circuit to length=0 and drop the OOB tail.
// Without these carve-outs, sum(output_lengths) < unique_indices.numel(),
// which produces an inconsistent KJT and crashes downstream
// all_to_all_single with a "Split sizes doesn't match total dim 0 size".
template <typename index_t>
__global__ __launch_bounds__(kFlatBlockSize) void unique_indices_length_kernel(
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        hash_size_offsets,
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        hash_size_cumsum,
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        linear_unique_indices,
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
    const bool is_last_group = (hash_end == hash_size_cumsum.size(0) - 1);
    if (low == high && !is_last_group) {
      // Empty intermediate feature group. Output is pre-zeroed by
      // at::zeros at the launch site; nothing to write.
      s_div_length = 0;
      s_r_length = 0;
    } else {
      const int32_t lo_pos =
          device_lower_bound<index_t>(linear_unique_indices, low);
      const int32_t hi_pos = is_last_group
          ? linear_unique_indices.size(0)
          : device_lower_bound<index_t>(linear_unique_indices, high);
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

// Pipeline (cross-kernel data flow that ties the steps together):
//
// Caller contract on indices:
//   - For intermediate features (t < T - 1) with per_feature_hash > 0:
//     indices[i] < per_feature_hash[t]. Violations are detected by a
//     CUDA_KERNEL_ASSERT in linearize_index_flat_kernel. The op cannot
//     produce correct per-feature output_lengths under intermediate-feature
//     OOB (counts leak into the next feature group), but the total count and
//     unique_indices values are still correct.
//   - For merged/masked features (per_feature_hash == 0, i.e. consecutive
//     equal entries in hash_size_cumsum): any indices[i] is valid. The
//     feature shares hash space with another via the hash_size_offsets
//     indirection.
//   - For the last feature (t == T - 1): indices may exceed per_feature_hash.
//     Supported via the hi_pos clamp in unique_indices_length_kernel and the
//     scatter-form delinearize_unique_index_kernel.
//
//   1. linearize_index_flat_kernel writes
//        linear_indices[i] = hash_size_cumsum[t] + indices[i]
//      so feature t's linearized values lie in
//        [hash_size_cumsum[t], hash_size_cumsum[t+1])
//      under the contract above (last-feature OOB extends the upper end).
//
//   2. jagged_unique_indices_pipeline replaces at::_unique with an
//      explicit cub radix sort + RLE + inverse-index scatter, returning
//      (linear_unique_indices, reverse_index). linear_unique_indices is
//      sorted ascending. Combined with (1), this means feature t's unique
//      linearized values occupy a contiguous slice of linear_unique_indices.
//      The radix sort end_bit is trimmed to bit_width(max linearized key),
//      which on production shapes (hash_size ~1M) reduces 8 radix passes
//      to ~3. Deriving end_bit from the real max (rather than
//      total_hash_size) keeps last-feature OOB keys fully ordered.
//
//   3. delinearize_unique_index_kernel scatters the original
//      (pre-linearization) per-feature index values back into
//      unique_indices via reverse_index. This is index-bound-agnostic:
//      out-of-bound indices on any feature round-trip exactly, because
//      the scatter writes the original indices[i] value rather than
//      recovering it from the linearized key.
//
//   4. unique_indices_length_kernel relies on (1)+(2) to compute
//      num_unique per feature group via two binary searches over
//      linear_unique_indices, instead of an O(N) reduction over
//      reverse_index. See the kernel's docstring for the local form of
//      the invariant.
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
  // The cub pipeline uses int32 positions and an int32 num_items argument
  // for radix sort, RLE, and scan, so N must fit in int32. In practice this
  // is enforced upstream by the int32 PackedTensorAccessor on indices, but
  // we make it explicit here to fail loudly rather than silently truncating.
  TORCH_CHECK(
      N < static_cast<int64_t>(INT32_MAX),
      "jagged_unique_indices: indices.numel() (",
      N,
      ") exceeds INT32_MAX");

  Tensor linear_indices = at::empty_like(indices);

  using at::RestrictPtrTraits;

  AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "linearize_index", ([&] {
                            FBGEMM_LAUNCH_KERNEL(
                                (linearize_index_flat_kernel<index_t>),
                                total_B,
                                kFlatBlockSize,
                                0,
                                at::cuda::getCurrentCUDAStream(),
                                PTA_B(hash_size_cumsum, index_t, 1, 32),
                                PTA_B(indices, index_t, 1, 32),
                                PTA_B(offsets, index_t, 1, 32),
                                PTA_B(linear_indices, index_t, 1, 32),
                                FixedDivisor(total_B / T));
                          }));

  Tensor linear_unique_indices;
  Tensor reverse_index;
  if (N == 0) {
    linear_unique_indices = at::empty({0}, indices.options());
    reverse_index = at::empty({0}, indices.options().dtype(at::kLong));
  } else {
    // Derive the radix-sort end_bit from the actual maximum linearized key,
    // not from total_hash_size = hash_size_cumsum[T]. cub::DeviceRadixSort
    // ignores key bits >= end_bit, and this op supports last-feature OOB
    // indices whose linearized key `hash_size_cumsum[T-1] + idx` can exceed
    // total_hash_size. Trimming end_bit to bit_width(total_hash_size) would
    // drop those high bits, leaving linear_unique_indices sorted only by its
    // low bits -- which silently mis-attributes per-feature output_lengths
    // (the length kernel's binary searches assume a fully sorted array) and
    // can over-count num_unique (the full-value RLE dedup sees equal keys
    // separated by low-bit-colliding distinct keys). Reducing over the real
    // max keeps the radix-pass win in the common in-bound case (max ~
    // total_hash_size) while staying correct under OOB. The max-reduction
    // replaces at::_unique's historical implicit D->H sync, so the single
    // sync below is net-neutral (plus one cheap reduce kernel over N).
    const int64_t max_linear_index = linear_indices.max().item().toLong();
    TORCH_CHECK(
        max_linear_index >= 0,
        "jagged_unique_indices: linearized indices must be non-negative, got "
        "max ",
        max_linear_index);
    // Bit-width of max_linear_index via integer math. __builtin_clzll is
    // defined for any positive uint64_t, so this expression is total over
    // the entire [0, INT64_MAX] range and never invokes UB. (Unlike the
    // float-log2 formulation, which produces NaN at INT64_MAX due to
    // signed overflow in `max_linear_index + 1`.)
    AT_DISPATCH_INDEX_TYPES(
        indices.scalar_type(), "jagged_unique_indices_pipeline", ([&] {
          const int max_bits = static_cast<int>(sizeof(index_t) * 8);
          int end_bit;
          if (max_linear_index <= 0) {
            end_bit = 1;
          } else {
            end_bit =
                64 - __builtin_clzll(static_cast<uint64_t>(max_linear_index));
          }
          end_bit = std::min(end_bit, max_bits);
          TORCH_CHECK(
              end_bit >= 1 && end_bit <= max_bits,
              "jagged_unique_indices: bad end_bit=",
              end_bit);
          jagged_unique_indices_pipeline<index_t>(
              linear_indices, end_bit, linear_unique_indices, reverse_index);
        }));
  }

  const auto total_indices = indices.size(0);
  Tensor unique_indices = at::empty_like(linear_unique_indices);

  if (total_indices > 0 && unique_indices.numel() > 0) {
    AT_DISPATCH_INDEX_TYPES(
        indices.scalar_type(), "delinearize_unique_index", ([&] {
          FBGEMM_LAUNCH_KERNEL(
              (delinearize_unique_index_kernel<index_t>),
              static_cast<int32_t>(
                  (total_indices + kFlatBlockSize - 1) / kFlatBlockSize),
              kFlatBlockSize,
              0,
              at::cuda::getCurrentCUDAStream(),
              PTA_B(indices, index_t, 1, 32),
              PTA_B(reverse_index, int64_t, 1, 32),
              PTA_B(unique_indices, index_t, 1, 32));
        }));
  }

  Tensor output_lengths = at::zeros({total_B}, offsets.options());
  AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "unique_indices_length", ([&] {
                            FBGEMM_LAUNCH_KERNEL(
                                (unique_indices_length_kernel<index_t>),
                                T,
                                kFlatBlockSize,
                                0,
                                at::cuda::getCurrentCUDAStream(),
                                PTA_B(hash_size_offsets, index_t, 1, 32),
                                PTA_B(hash_size_cumsum, index_t, 1, 32),
                                PTA_B(linear_unique_indices, index_t, 1, 32),
                                PTA_B(output_lengths, index_t, 1, 32),
                                static_cast<int32_t>(total_B / T));
                          }));

  Tensor output_offsets;
  output_offsets = asynchronous_complete_cumsum_gpu(output_lengths);
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
              // HIP enforces a hard limit of 2^32 total threads per launch
              // (unlike CUDA, which silently wraps).
              // accumulate_weights_and_counts_kernel grid-strides over `i`
              // (`for (int i = bid * kMaxThreads + tid; i < total_elements;
              // i += kMaxThreads * gridDim.x)`), so capping is
              // correctness-preserving.
              // See: https://github.com/ROCm/hip/issues/2253
              const auto num_blocks = utils::cuda::cap_grid_dim_x_from_workload(
                  total_elements,
                  kMaxThreads,
                  at::cuda::getCurrentCUDAStream());

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
