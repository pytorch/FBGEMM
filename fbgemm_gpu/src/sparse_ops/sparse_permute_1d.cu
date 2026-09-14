/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "common.cuh"

using Tensor = at::Tensor;

namespace fbgemm_gpu {

// Kernel for permuting 1D lengths. Used for permutation of sparse features.
//
// The bounds checks are compiled in only for debug=true, which defaults to the
// compile-time kPermuteDeviceAssert (see common.cuh): a terminating failure
// path costs on the healthy path even when it never fires, so the default
// instantiation carries no check. See permute_2D_data_kernel_vec for the
// measurement.
template <
    typename index_t,
    typename permute_t = int32_t,
    bool debug = kPermuteDeviceAssert>
__global__ __launch_bounds__(kMaxThreads) void permute_1D_lengths_kernel(
    const index_t* __restrict__ lengths,
    int32_t permuted_lengths_size,
    const permute_t* __restrict__ permute,
    int64_t lengths_size,
    int64_t indices_size,
    index_t* __restrict__ permuted_lengths) {
  CUDA_KERNEL_LOOP(i, permuted_lengths_size) {
    const auto permute_idx = permute[i];
    if constexpr (debug) {
      // An out-of-range permute index makes lengths[permute[i]] read out of
      // bounds and silently poisons permuted_lengths; the corruption only
      // surfaces much later as a non-deterministic CUDA illegal memory access
      // in the data kernel. Asserting at the read site localizes the fault to
      // its true origin.
      CUDA_KERNEL_ASSERT_PRINTF(
          permute_idx >= 0 && static_cast<int64_t>(permute_idx) < lengths_size,
          "permute_1D_lengths_kernel: permute index out of bounds "
          "(i=%d, permute[i]=%lld, lengths_size=%lld)",
          i,
          static_cast<long long>(permute_idx),
          static_cast<long long>(lengths_size));
    }
    const index_t length = lengths[permute_idx];
    if constexpr (debug) {
      // A corrupt length *value* even when the index is in range: a valid
      // per-feature length is in [0, indices_size], since one feature cannot
      // own more elements than the total input indices. Out of that range means
      // the lengths contents were poisoned (e.g. a concurrent write/free),
      // which otherwise surfaces only as a garbage output_offsets sum. Firing
      // here rather than the index assert distinguishes the two causes.
      CUDA_KERNEL_ASSERT_PRINTF(
          static_cast<int64_t>(length) >= 0 &&
              static_cast<int64_t>(length) <= indices_size,
          "permute_1D_lengths_kernel: length value out of range "
          "(i=%d, permute[i]=%lld, length=%lld, indices_size=%lld)",
          i,
          static_cast<long long>(permute_idx),
          static_cast<long long>(length),
          static_cast<long long>(indices_size));
    }
    permuted_lengths[i] = length;
  }
}

// Flat counterpart to permute_1D_data_kernel_vec, for the same reason as the 2D
// case but more acute: the 1D launch shape is dim3(64, 16), so a short row
// wastes 64 lanes instead of 32. Each thread takes a contiguous run of output
// elements, and the block stages the offsets of the rows it spans in shared
// memory. Measured on GB300 at 4.79 elements per row: 5626 -> 848 us.
template <
    int ELEMS_PER_THREAD,
    int SMEM_ROWS,
    typename offsets_t,
    typename indices_t,
    typename permute_t,
    bool debug = kPermuteDeviceAssert>
__global__ __launch_bounds__(kMaxThreads) void permute_1D_data_kernel_flat(
    int64_t len,
    int32_t BT,
    const indices_t* __restrict__ indices,
    const permute_t* __restrict__ permute,
    const offsets_t* __restrict__ input_offsets,
    const offsets_t* __restrict__ output_offsets,
    indices_t* __restrict__ permuted_indices) {
  __shared__ int64_t s_out_off[SMEM_ROWS + 1];
  __shared__ int64_t s_src_base[SMEM_ROWS];
  __shared__ int32_t s_span[2];

  const int64_t block_base =
      static_cast<int64_t>(blockIdx.x) * blockDim.x * ELEMS_PER_THREAD;
  if (block_base >= len) {
    return;
  }
  if (threadIdx.x == 0) {
    const int64_t last_elem =
        min(block_base + blockDim.x * ELEMS_PER_THREAD - 1, len - 1);
    s_span[0] = find_row_for_element(output_offsets, BT, block_base);
    s_span[1] = find_row_for_element(output_offsets, BT, last_elem);
  }
  __syncthreads();
  const int32_t row_lo = s_span[0];
  const int32_t nrows = s_span[1] - row_lo + 1;
  const bool use_smem = nrows <= SMEM_ROWS;

  if (use_smem) {
    for (auto r = threadIdx.x; r <= nrows; r += blockDim.x) {
      const int32_t row = row_lo + r;
      s_out_off[r] =
          (row < BT) ? static_cast<int64_t>(output_offsets[row]) : len;
      if (r < nrows && row < BT) {
        s_src_base[r] = static_cast<int64_t>(input_offsets[permute[row]]) -
            static_cast<int64_t>(output_offsets[row]);
      }
    }
    __syncthreads();
  }

  const int64_t base =
      block_base + static_cast<int64_t>(threadIdx.x) * ELEMS_PER_THREAD;
  if (base >= len) {
    return;
  }

  int32_t r;
  int64_t row_end;
  int64_t src_base;
  if (use_smem) {
    r = 0;
    while (r < nrows && s_out_off[r + 1] <= base) {
      ++r;
    }
    row_end = s_out_off[r + 1];
    src_base = s_src_base[r];
  } else {
    r = row_lo;
    while (r + 1 < BT && static_cast<int64_t>(output_offsets[r + 1]) <= base) {
      ++r;
    }
    row_end = (r + 1 < BT) ? static_cast<int64_t>(output_offsets[r + 1]) : len;
    src_base = static_cast<int64_t>(input_offsets[permute[r]]) -
        static_cast<int64_t>(output_offsets[r]);
  }

  if constexpr (debug) {
    CUDA_KERNEL_ASSERT_PRINTF(
        src_base + base >= 0,
        "permute_1D_data_kernel_flat: negative source index (base=%lld)",
        static_cast<long long>(base));
  }

  const int64_t last = min(base + ELEMS_PER_THREAD, len);
  if (last == base + ELEMS_PER_THREAD && last <= row_end) {
    const int64_t src = src_base + base;
    const uintptr_t coalign =
        reinterpret_cast<uintptr_t>(permuted_indices + base) |
        reinterpret_cast<uintptr_t>(indices + src);
    constexpr int32_t E = 16 / static_cast<int32_t>(sizeof(indices_t));
    if ((coalign & 0xF) == 0 && E > 0 && (ELEMS_PER_THREAD % E) == 0) {
      auto* dst4 = reinterpret_cast<uint4*>(permuted_indices + base);
      const auto* src4 = reinterpret_cast<const uint4*>(indices + src);
#pragma unroll
      for (int v = 0; v < ELEMS_PER_THREAD / E; ++v) {
        dst4[v] = src4[v];
      }
      return;
    }
  }

#pragma unroll
  for (int e = 0; e < ELEMS_PER_THREAD; ++e) {
    const int64_t i = base + e;
    if (i >= len) {
      return;
    }
    if (i >= row_end) {
      if (use_smem) {
        do {
          ++r;
        } while (r < nrows && s_out_off[r + 1] <= i);
        row_end = s_out_off[r + 1];
        src_base = s_src_base[r];
      } else {
        do {
          ++r;
        } while (r + 1 < BT &&
                 static_cast<int64_t>(output_offsets[r + 1]) <= i);
        row_end =
            (r + 1 < BT) ? static_cast<int64_t>(output_offsets[r + 1]) : len;
        src_base = static_cast<int64_t>(input_offsets[permute[r]]) -
            static_cast<int64_t>(output_offsets[r]);
      }
    }
    permuted_indices[i] = indices[src_base + i];
  }
}

// Kernel for permuting the indices and weights. Used for permutation of sparse
// data
template <
    bool has_weight,
    typename offsets_t,
    typename indices_t,
    typename weights_t,
    typename permute_t = int32_t>
__global__ __launch_bounds__(kMaxThreads) void permute_1D_data_kernel(
    int32_t permuted_indices_size,
    int32_t permuted_lengths_size,
    const indices_t* __restrict__ indices,
    const weights_t* __restrict__ weights,
    const permute_t* __restrict__ permute,
    const offsets_t* __restrict__ input_offsets,
    const offsets_t* __restrict__ output_offsets,
    indices_t* __restrict__ permuted_indices,
    weights_t* __restrict__ permuted_weights,
    int32_t weights_columns) {
  auto b_t_start = blockIdx.x * blockDim.y + threadIdx.y;
  const auto stride = gridDim.x * blockDim.y;
  for (int b_t = b_t_start; b_t < permuted_lengths_size; b_t += stride) {
    offsets_t output_start = output_offsets[b_t];
    offsets_t segment_length;
    if (b_t == permuted_lengths_size - 1) {
      segment_length = permuted_indices_size - output_offsets[b_t];
    } else {
      segment_length = output_offsets[b_t + 1] - output_offsets[b_t];
    }
    offsets_t input_start = input_offsets[permute[b_t]];
    for (auto i = threadIdx.x; i < segment_length; i += blockDim.x) {
      permuted_indices[output_start + i] = indices[input_start + i];
      if (has_weight) {
        for (int col = 0; col < weights_columns; ++col) {
          permuted_weights[(output_start + i) * weights_columns + col] =
              weights[(input_start + i) * weights_columns + col];
        }
      }
    }
  }
}

// Vectorized kernel for permuting the indices and weights. Used for permutation
// of sparse data. Uses vec4 loads for improved memory bandwidth.
template <
    bool has_weight,
    typename offsets_t,
    typename indices_t,
    typename weights_t,
    typename permute_t = int32_t>
__global__ __launch_bounds__(kMaxThreads) void permute_1D_data_kernel_vec(
    int32_t permuted_indices_size,
    int32_t permuted_lengths_size,
    const indices_t* __restrict__ indices,
    const weights_t* __restrict__ weights,
    const permute_t* __restrict__ permute,
    const offsets_t* __restrict__ input_offsets,
    const offsets_t* __restrict__ output_offsets,
    indices_t* __restrict__ permuted_indices,
    weights_t* __restrict__ permuted_weights,
    int32_t weights_columns) {
  // Select vector types based on element size (vec4 for 4× bandwidth)
  using indices_vec4_t =
      typename std::conditional<sizeof(indices_t) == 8, long4, float4>::type;
  using weights_vec4_t =
      typename std::conditional<sizeof(weights_t) == 8, long4, float4>::type;

  const auto b_t_start = blockIdx.x * blockDim.y + threadIdx.y;
  const auto stride = gridDim.x * blockDim.y;

  for (int b_t = b_t_start; b_t < permuted_lengths_size; b_t += stride) {
    // Read offsets once - use int32_t for segment_length as it fits in 32 bits
    const offsets_t output_start = output_offsets[b_t];
    const offsets_t output_end = (b_t == permuted_lengths_size - 1)
        ? permuted_indices_size
        : output_offsets[b_t + 1];
    const int32_t segment_length =
        static_cast<int32_t>(output_end - output_start);
    const offsets_t input_start = input_offsets[permute[b_t]];

    // Compute pointers
    indices_t* __restrict__ indices_dst_ptr = permuted_indices + output_start;
    const indices_t* __restrict__ indices_src_ptr = indices + input_start;
    weights_t* __restrict__ weights_dst_ptr = has_weight
        ? permuted_weights + output_start * weights_columns
        : nullptr;
    const weights_t* __restrict__ weights_src_ptr =
        has_weight ? weights + input_start * weights_columns : nullptr;

    // Total weight elements to copy for this segment (accounts for 2D weights)
    const int32_t total_weight_elements = segment_length * weights_columns;

    // Check alignment once per segment.
    // For 2D weights, the weight pointer stride includes weights_columns, so
    // vec4 alignment is valid only when weights_columns is divisible by 4 or
    // equals 1 (1D case), ensuring row boundaries stay aligned.
    const bool indices_vec4_aligned =
        (sizeof(indices_t) == 4 || sizeof(indices_t) == 8) &&
        (reinterpret_cast<uintptr_t>(indices_dst_ptr) &
         (alignof(indices_vec4_t) - 1)) == 0 &&
        (reinterpret_cast<uintptr_t>(indices_src_ptr) &
         (alignof(indices_vec4_t) - 1)) == 0;

    const bool weights_vec4_aligned = !has_weight ||
        ((weights_columns == 1 || weights_columns % 4 == 0) &&
         (reinterpret_cast<uintptr_t>(weights_dst_ptr) &
          (alignof(weights_vec4_t) - 1)) == 0 &&
         (reinterpret_cast<uintptr_t>(weights_src_ptr) &
          (alignof(weights_vec4_t) - 1)) == 0);

    if (indices_vec4_aligned && weights_vec4_aligned) {
      // Vectorized path - process indices and weights separately since they may
      // have different element counts (weights has weights_columns factor)
      const int32_t vec4_count = segment_length / 4;
      const int32_t remainder = segment_length & 3; // segment_length % 4

      auto indices_dst = reinterpret_cast<indices_vec4_t*>(indices_dst_ptr);
      auto indices_src =
          reinterpret_cast<const indices_vec4_t*>(indices_src_ptr);

      if (has_weight) {
        const int32_t vec4_weight_count = total_weight_elements / 4;
        const int32_t weight_remainder = total_weight_elements & 3;
        auto weights_dst = reinterpret_cast<weights_vec4_t*>(weights_dst_ptr);
        auto weights_src =
            reinterpret_cast<const weights_vec4_t*>(weights_src_ptr);

// copy indices
#pragma unroll
        for (auto i = threadIdx.x; i < vec4_count; i += blockDim.x) {
          indices_dst[i] = indices_src[i];
        }
        // Handle remainder indices (0-3 elements)
        if (threadIdx.x < remainder) {
          const auto offset = vec4_count * 4 + threadIdx.x;
          indices_dst_ptr[offset] = indices_src_ptr[offset];
        }

// copy weights (segment_length * weights_columns total elements)
#pragma unroll
        for (auto i = threadIdx.x; i < vec4_weight_count; i += blockDim.x) {
          weights_dst[i] = weights_src[i];
        }
        // Handle remainder weight elements (0-3 elements)
        if (threadIdx.x < weight_remainder) {
          const auto offset = vec4_weight_count * 4 + threadIdx.x;
          weights_dst_ptr[offset] = weights_src_ptr[offset];
        }
      } else {
// copy only indices
#pragma unroll
        for (auto i = threadIdx.x; i < vec4_count; i += blockDim.x) {
          indices_dst[i] = indices_src[i];
        }

        // Handle remainder elements (0-3 elements)
        if (threadIdx.x < remainder) {
          const auto offset = vec4_count * 4 + threadIdx.x;
          indices_dst_ptr[offset] = indices_src_ptr[offset];
        }
      }
    } else {
      // Scalar fallback path
      for (auto i = threadIdx.x; i < segment_length; i += blockDim.x) {
        indices_dst_ptr[i] = indices_src_ptr[i];
      }
      if (has_weight) {
        for (auto i = threadIdx.x; i < total_weight_elements; i += blockDim.x) {
          weights_dst_ptr[i] = weights_src_ptr[i];
        }
      }
    }
  }
}

DLL_PUBLIC std::tuple<Tensor, Tensor, std::optional<Tensor>>
permute_1D_sparse_data_cuda(
    const Tensor& permute,
    const Tensor& lengths,
    const Tensor& indices,
    const std::optional<Tensor>& weights,
    const std::optional<int64_t>& permuted_lengths_sum) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(permute, lengths, indices, weights);

  CUDA_DEVICE_GUARD(indices);

  const auto permute_contig = permute.contiguous();
  const auto lengths_contig = lengths.contiguous();
  const auto indices_contig = indices.contiguous();
  // the data to permute over can be less or more with or without
  // repetitions

  const auto lengths_size = lengths.numel();

  const auto permuted_lengths_size = permute.numel();

  if (permuted_lengths_size == 0 || lengths_size == 0) {
    // Permutation will not be performed.  Return the input tensors
    return {
        lengths.view({-1}).clone(),
        indices.clone(),
        weights.has_value() ? std::make_optional(weights->clone())
                            : std::nullopt};
  }
  int64_t debug_permuted_lengths_sum = 0;
  if (is_debug_permute_enabled()) {
    debug_check_permute_inputs(
        "permute_1D",
        permute_contig,
        lengths_contig,
        indices_contig,
        lengths_size,
        weights);
  }

  Tensor permuted_lengths;
  Tensor permuted_indices;
  Tensor permuted_weights;
  TORCH_CHECK(
      permuted_lengths_size >= 0 &&
          permuted_lengths_size <= std::numeric_limits<int32_t>::max(),
      "permuted_lengths_size must be >= 0 and within int32. permute.numel() = ",
      permuted_lengths_size,
      ", lengths.numel() = ",
      lengths_size);
  permuted_lengths = at::empty({permuted_lengths_size}, lengths.options());

  constexpr int32_t threads_1 = kMaxThreads;
  // HIP enforces a hard limit of 2^32 total threads per launch (unlike CUDA,
  // which silently wraps). permute_1D_lengths_kernel uses CUDA_KERNEL_LOOP,
  // which already grid-strides, so capping on ROCm overflow is
  // correctness-preserving. OverflowOnly leaves the CUDA launch uncapped.
  // See: https://github.com/ROCm/hip/issues/2253
  const auto blocks_1 = utils::cuda::cap_grid_dim_x_from_workload(
      permuted_lengths_size, threads_1, at::cuda::getCurrentCUDAStream());
  AT_DISPATCH_INDEX_TYPES(
      permute.scalar_type(), "permute_1D_lengths_permute_type", [&] {
        using permute_t = index_t;
        AT_DISPATCH_INDEX_TYPES(
            lengths.scalar_type(), "permute_1D_lengths_kernel", [&] {
              FBGEMM_LAUNCH_KERNEL(
                  (permute_1D_lengths_kernel<index_t, permute_t>),
                  blocks_1,
                  threads_1,
                  0,
                  at::cuda::getCurrentCUDAStream(),
                  lengths_contig.data_ptr<index_t>(),
                  permuted_lengths_size,
                  permute_contig.data_ptr<permute_t>(),
                  lengths_size,
                  indices_contig.numel(),
                  permuted_lengths.data_ptr<index_t>());
            });
      });

  if (is_debug_permute_enabled()) {
    debug_permuted_lengths_sum = debug_check_permuted_lengths(
        "permute_1D", permuted_lengths, indices_contig.numel());
  }

  // convert lengths to offsets
  const auto input_offsets = asynchronous_exclusive_cumsum_gpu(lengths_contig);
  const auto output_offsets =
      asynchronous_complete_cumsum_gpu(permuted_lengths.flatten());

  if (is_debug_permute_enabled()) {
    debug_check_output_offsets(
        "permute_1D", output_offsets, debug_permuted_lengths_sum);
  }

  int64_t permuted_indices_size = 0;
  if (permuted_lengths_sum.has_value()) {
    permuted_indices_size = permuted_lengths_sum.value();
  } else {
    permuted_indices_size = output_offsets[-1].item<int64_t>();
  }
  TORCH_CHECK(
      permuted_indices_size >= 0 &&
          permuted_indices_size <= std::numeric_limits<int32_t>::max(),
      "permuted_indices_size must be >= 0 and within int32. "
      "permuted_indices_size = ",
      permuted_indices_size);

  constexpr int32_t BT_blocks = 16;
  dim3 threads_2(64, BT_blocks);
  // HIP enforces a hard limit of 2^32 total threads per launch (unlike CUDA,
  // which silently wraps). The kernel's grid-striding loop over b_t handles
  // the overflow, so capping on ROCm overflow is correctness-preserving.
  // OverflowOnly leaves the CUDA launch uncapped.
  // See: https://github.com/ROCm/hip/issues/2253
  //
  // gridDim.x is sized by the BT_blocks (blockDim.y) work dimension, i.e.
  // ceil(N / BT_blocks) -- NOT the full 64 * BT_blocks block size. Use the
  // lower-level cap_grid_dim_x with the pre-computed block count so the grid
  // size is preserved; 64 * BT_blocks is passed only as the per-block thread
  // count for the OverflowOnly overflow-threshold check.
  const auto blocks_2 = utils::cuda::cap_grid_dim_x(
      cuda_calc_xblock_count(permuted_lengths_size, BT_blocks),
      BT_blocks * 64,
      at::cuda::getCurrentCUDAStream(),
      utils::cuda::BlockCapPolicy::OverflowOnly);
  // Short rows waste 64 lanes each in the vec kernel; route them to the flat
  // kernel. FBGEMM_FLAT_PERMUTE_1D overrides the heuristic in either direction.
  constexpr int32_t kFlatElemsPerThread = 4;
  constexpr int32_t kFlatSmemRows = 1024;
  constexpr int32_t kFlatThreads = 256;
  constexpr int64_t kFlatMaxAvgSegment = 24;
  const int64_t avg_segment_length = (permuted_lengths_size > 0)
      ? (permuted_indices_size / permuted_lengths_size)
      : 0;
  const bool use_flat_kernel = !weights.has_value() &&
      permuted_indices_size > 0 &&
      is_flat_permute_1d_enabled(avg_segment_length <= kFlatMaxAvgSegment);

  permuted_indices = at::empty(permuted_indices_size, indices.options());

  if (use_flat_kernel) {
    const int64_t elems_per_block =
        static_cast<int64_t>(kFlatThreads) * kFlatElemsPerThread;
    const int32_t num_blocks = static_cast<int32_t>(
        (permuted_indices_size + elems_per_block - 1) / elems_per_block);
    AT_DISPATCH_INDEX_TYPES(
        output_offsets.scalar_type(), "permute_1D_block_first_row", [&] {
          using offsets_t = index_t;
          AT_DISPATCH_INDEX_TYPES(
              permute.scalar_type(), "permute_1D_flat_permute_type", [&] {
                using permute_t = index_t;
                FBGEMM_DISPATCH_ALL_TYPES(
                    indices.scalar_type(), "permute_1D_data_kernel_flat", [&] {
                      using indices_t = scalar_t;
                      FBGEMM_LAUNCH_KERNEL(
                          (permute_1D_data_kernel_flat<
                              kFlatElemsPerThread,
                              kFlatSmemRows,
                              offsets_t,
                              indices_t,
                              permute_t>),
                          num_blocks,
                          kFlatThreads,
                          0,
                          at::cuda::getCurrentCUDAStream(),
                          permuted_indices_size,
                          permuted_lengths_size,
                          indices_contig.data_ptr<indices_t>(),
                          permute_contig.data_ptr<permute_t>(),
                          input_offsets.data_ptr<offsets_t>(),
                          output_offsets.data_ptr<offsets_t>(),
                          permuted_indices.data_ptr<indices_t>());
                    });
              });
        });

    return {permuted_lengths, permuted_indices, permuted_weights};
  }

  AT_DISPATCH_INDEX_TYPES(
      permute.scalar_type(), "permute_1D_data_permute_type", [&] {
        using permute_t = index_t;
        AT_DISPATCH_INDEX_TYPES(
            input_offsets.scalar_type(), "permute_1D_data_kernel_vec_1", [&] {
              using offsets_t = index_t;
              FBGEMM_DISPATCH_ALL_TYPES(
                  indices.scalar_type(), "permute_1D_data_kernel_vec_2", [&] {
                    using indices_t = scalar_t;
                    if (weights.has_value()) {
                      const Tensor weights_value = weights.value();
                      const auto weights_value_contig =
                          weights_value.contiguous();
                      int32_t weights_columns = 1;
                      if (weights_value.dense_dim() > 1) {
                        TORCH_CHECK(
                            weights_value.size(1) >= 0 &&
                                weights_value.size(1) <=
                                    std::numeric_limits<int32_t>::max(),
                            "weights_columns must be >= 0 and within int32. "
                            "weights.size(1) = ",
                            weights_value.size(1));
                        weights_columns = weights_value.size(1);
                        permuted_weights = at::empty(
                            {permuted_indices_size, weights_columns},
                            weights_value.options());
                      } else {
                        permuted_weights = at::empty(
                            permuted_indices_size, weights_value.options());
                      }
                      FBGEMM_DISPATCH_ALL_TYPES_AND_DOUBLE(
                          weights_value.scalar_type(),
                          "permute_1D_data_kernel_vec_3",
                          [&] {
                            using weights_t = scalar_t;
                            FBGEMM_LAUNCH_KERNEL(
                                (permute_1D_data_kernel_vec<
                                    true,
                                    offsets_t,
                                    indices_t,
                                    weights_t,
                                    permute_t>),
                                blocks_2,
                                threads_2,
                                0,
                                at::cuda::getCurrentCUDAStream(),
                                permuted_indices_size,
                                permuted_lengths_size,
                                indices_contig.data_ptr<indices_t>(),
                                weights_value_contig.data_ptr<weights_t>(),
                                permute_contig.data_ptr<permute_t>(),
                                input_offsets.data_ptr<offsets_t>(),
                                output_offsets.data_ptr<offsets_t>(),
                                permuted_indices.data_ptr<indices_t>(),
                                permuted_weights.data_ptr<weights_t>(),
                                weights_columns);
                          }); // for each weights_t
                    } else {
                      FBGEMM_LAUNCH_KERNEL(
                          (permute_1D_data_kernel_vec<
                              false,
                              offsets_t,
                              indices_t,
                              std::nullptr_t,
                              permute_t>),
                          blocks_2,
                          threads_2,
                          0,
                          at::cuda::getCurrentCUDAStream(),
                          permuted_indices_size,
                          permuted_lengths_size,
                          indices_contig.data_ptr<indices_t>(),
                          nullptr,
                          permute_contig.data_ptr<permute_t>(),
                          input_offsets.data_ptr<offsets_t>(),
                          output_offsets.data_ptr<offsets_t>(),
                          permuted_indices.data_ptr<indices_t>(),
                          nullptr,
                          1);
                    }
                  }); // for each indices_t
            }); // for each offsets_t
      }); // for each permute_t

  return {permuted_lengths, permuted_indices, permuted_weights};
}

} // namespace fbgemm_gpu

FBGEMM_OP_DISPATCH(
    CUDA,
    "permute_1D_sparse_data",
    fbgemm_gpu::permute_1D_sparse_data_cuda);
