/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "fbgemm_gpu/utils/embedding_bounds_check_common.cuh"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>

namespace {

// FBGEMM_FLAT_BOUNDS_CHECK: opt-in, since the flat kernels were only tuned on
// GB300. Unset keeps v1, `auto` leaves the heuristic in charge, truthy forces
// the flat kernels on, falsy forces them off (rollback without a rebuild).
enum class FlatOverride { kHeuristic, kOn, kOff };

FlatOverride read_flat_override() {
  const char* v = std::getenv("FBGEMM_FLAT_BOUNDS_CHECK");
  if (v == nullptr || v[0] == '\0') {
    return FlatOverride::kOff;
  }
  std::string val(v);
  std::transform(val.begin(), val.end(), val.begin(), [](unsigned char c) {
    return std::tolower(c);
  });
  if (val == "auto" || val == "heuristic") {
    return FlatOverride::kHeuristic;
  }
  if (val == "1" || val == "true" || val == "yes" || val == "on") {
    return FlatOverride::kOn;
  }
  if (val == "0" || val == "false" || val == "no" || val == "off") {
    return FlatOverride::kOff;
  }
  TORCH_WARN(
      "[fbgemm] FBGEMM_FLAT_BOUNDS_CHECK=\"",
      v,
      "\" is not recognized; keeping the existing kernel.");
  return FlatOverride::kOff;
}

bool use_flat_bounds_check(bool heuristic) {
  static const FlatOverride override = read_flat_override();
  switch (override) {
    case FlatOverride::kOn:
      return true;
    case FlatOverride::kOff:
      return false;
    default:
      return heuristic;
  }
}

} // namespace

// Validates and clamps the per-row offsets, one thread per row. v1 spends a
// whole warp per row on this and then loops the row's indices with that same
// warp; splitting the two lets the index pass use a mapping that does not
// depend on how many indices a row happens to hold.
template <typename index_t>
__global__ void bounds_check_offsets_kernel(
    int32_t total_B,
    index_t num_indices,
    index_t* __restrict__ offsets,
    int64_t* __restrict__ warning,
    bool warn) {
  const auto b_t = blockIdx.x * blockDim.x + threadIdx.x;
  if (b_t >= total_B) {
    return;
  }
  index_t start = offsets[b_t];
  index_t end = offsets[b_t + 1];
  if (start < 0 || start > end || end > num_indices) {
    if (warn && gpuAtomicIncrement(warning) == 0) {
      printf(
          "EmbeddingBoundsCheck: (at least one) Out of bounds access for "
          "b_t: %d, indices_start: %lld, indices_end: %lld, num_indices: %lld. "
          "Setting indices_start and indices_end within the range.\n",
          static_cast<int32_t>(b_t),
          static_cast<int64_t>(start),
          static_cast<int64_t>(end),
          static_cast<int64_t>(num_indices));
    }
    start = std::max(static_cast<index_t>(0), std::min(start, num_indices));
    end = std::max(start, std::min(end, num_indices));
    offsets[b_t] = start;
    offsets[b_t + 1] = end;
  }
}

namespace {
// Locate the row owning `target` in a monotone offsets array of `n` rows.
template <typename offsets_t>
__device__ __forceinline__ int32_t
find_row_for_element(const offsets_t* offsets, int32_t n, int64_t target) {
  int32_t lo = 0;
  int32_t hi = n - 1;
  while (lo < hi) {
    const int32_t mid = (lo + hi + 1) >> 1;
    if (static_cast<int64_t>(offsets[mid]) <= target) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  return lo;
}

} // namespace

// Flat index check: each thread owns a contiguous run of indices rather than a
// row, and the block stages the row boundaries and each row's table bound in
// shared memory. At the production pooling of about one index per row the v1
// mapping leaves 31 of 32 lanes idle and reaches 73 GB/s; a streaming read of
// the same data runs at 6642 GB/s on GB300.
template <int ELEMS_PER_THREAD, int SMEM_ROWS, typename index_t>
__global__ __launch_bounds__(kMaxThreads) void bounds_check_indices_kernel_flat(
    const int64_t* __restrict__ rows_per_table,
    index_t* __restrict__ indices,
    const index_t* __restrict__ offsets,
    int32_t total_B,
    int32_t B,
    index_t num_indices,
    BoundsCheckMode bounds_check_mode,
    int64_t* __restrict__ warning) {
  __shared__ int64_t s_off[SMEM_ROWS + 1];
  __shared__ int64_t s_bound[SMEM_ROWS];
  __shared__ int32_t s_span[2];

  const int64_t block_base =
      static_cast<int64_t>(blockIdx.x) * blockDim.x * ELEMS_PER_THREAD;
  if (block_base >= static_cast<int64_t>(num_indices)) {
    return;
  }
  // Resolve the block's row span here rather than in a separate pass: two
  // binary searches per block is cheaper than an extra launch plus the
  // allocation its output would need.
  if (threadIdx.x == 0) {
    const int64_t last_elem =
        min(block_base + blockDim.x * ELEMS_PER_THREAD - 1,
            static_cast<int64_t>(num_indices) - 1);
#pragma unroll 1
    for (int which = 0; which < 2; ++which) {
      const int64_t target = which == 0 ? block_base : last_elem;
      int32_t lo = 0;
      int32_t hi = total_B - 1;
      while (lo < hi) {
        const int32_t mid = (lo + hi + 1) >> 1;
        if (static_cast<int64_t>(offsets[mid]) <= target) {
          lo = mid;
        } else {
          hi = mid - 1;
        }
      }
      s_span[which] = lo;
    }
  }
  __syncthreads();
  const int32_t row_lo = s_span[0];
  const int32_t nrows = s_span[1] - row_lo + 1;
  const bool use_smem = nrows <= SMEM_ROWS;

  if (use_smem) {
    for (auto r = static_cast<int32_t>(threadIdx.x); r <= nrows;
         r += static_cast<int32_t>(blockDim.x)) {
      const int32_t row = row_lo + r;
      s_off[r] = (row <= total_B) ? static_cast<int64_t>(offsets[row])
                                  : static_cast<int64_t>(num_indices);
      if (r < nrows && row < total_B) {
        s_bound[r] = rows_per_table[row / B];
      }
    }
    __syncthreads();
  }

  const int64_t base =
      block_base + static_cast<int64_t>(threadIdx.x) * ELEMS_PER_THREAD;
  if (base >= static_cast<int64_t>(num_indices)) {
    return;
  }

  int32_t r;
  int64_t row_end;
  int64_t bound;
  if (use_smem) {
    r = find_row_for_element(s_off, nrows, base);
    row_end = s_off[r + 1];
    bound = s_bound[r];
  } else {
    r = find_row_for_element(offsets, total_B, base);
    row_end = (r + 1 <= total_B) ? static_cast<int64_t>(offsets[r + 1])
                                 : static_cast<int64_t>(num_indices);
    bound = rows_per_table[r / B];
  }

#pragma unroll
  for (int e = 0; e < ELEMS_PER_THREAD; ++e) {
    const int64_t i = base + e;
    if (i >= static_cast<int64_t>(num_indices)) {
      return;
    }
    if (i >= row_end) {
      // Binary search rather than stepping row by row: a run of empty rows
      // between two elements is unbounded, so a walk here costs O(rows) for a
      // single thread whenever the length distribution is skewed.
      if (use_smem) {
        r = find_row_for_element(s_off, nrows, i);
        row_end = s_off[r + 1];
        bound = s_bound[r];
      } else {
        r = find_row_for_element(offsets, total_B, i);
        row_end = (r + 1 <= total_B) ? static_cast<int64_t>(offsets[r + 1])
                                     : static_cast<int64_t>(num_indices);
        bound = rows_per_table[r / B];
      }
    }
    const auto idx = indices[i];
    // -1 marks a pruned row and is left alone, as in v1.
    if (idx == -1) {
      continue;
    }
    if (idx < 0 || static_cast<int64_t>(idx) >= bound) {
      if (bounds_check_mode == BoundsCheckMode::WARNING &&
          gpuAtomicIncrement(warning) == 0) {
        printf(
            "EmbeddingBoundsCheck: (at least one) Out of bounds access for "
            "element %lld, idx: %lld, num_rows: %lld. Setting idx to zero.\n",
            static_cast<int64_t>(i),
            static_cast<int64_t>(idx),
            static_cast<int64_t>(bound));
      }
      indices[i] = 0;
    }
  }
}

template <typename index_t, bool vbe>
__global__ __launch_bounds__(kMaxThreads) void bounds_check_indices_kernel_v1(
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        rows_per_table,
    pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> indices,
    pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> offsets,
    const int32_t* const B_offsets, // Use a raw pointer to avoid creating a
                                    // dummy PackedTensorAccessor
    BoundsCheckMode bounds_check_mode,
    pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> warning,
    FixedDivisor fd,
    const bool disable_offsets_adjustment,
    TORCH_DSA_KERNEL_ARGS) {
  int32_t T = rows_per_table.size(0);
  int32_t total_B = offsets.size(0) - 1;
  // On ROCm the launch caps the grid to stay within the HIP 2^32
  // threads-per-launch limit, so we grid-stride to cover the full workload.
  // On CUDA the grid is not capped and the loop body runs once per warp.
#ifdef USE_ROCM
  for (auto bt0 = blockIdx.x * blockDim.y + threadIdx.y; bt0 < fd.D() * T;
       bt0 += blockDim.y * gridDim.x) {
#else
  auto bt0 = blockIdx.x * blockDim.y + threadIdx.y;
  if (bt0 >= fd.D() * T) {
    return;
  }
#endif
    auto b_t = bt0;
    int32_t b;
    int32_t t;
    int32_t B = 0;

    if (!vbe && b_t >= total_B) {
#ifdef USE_ROCM
      continue;
#else
    return;
#endif
    }

    fd.DivMod(b_t, &t, &b);

    if (vbe) {
      // Check if t is valid
      if (t >= T) {
#ifdef USE_ROCM
        continue;
#else
      return;
#endif
      }
      const auto B_start = B_offsets[t];
      B = B_offsets[t + 1] - B_start;
      // Check if b is valid
      if (b >= B) {
#ifdef USE_ROCM
        continue;
#else
      return;
#endif
      }
      // Update b_t value
      b_t = B_start + b;
    } else {
      B = total_B / T;
    }

    const auto num_rows = rows_per_table[t];
    auto indices_start = offsets[b_t];
    auto indices_end = offsets[b_t + 1];
    const index_t num_indices = indices.size(0);

    if (disable_offsets_adjustment ||
        bounds_check_mode == BoundsCheckMode::FATAL) {
      CUDA_KERNEL_ASSERT(
          indices_start >= 0 && "indices_start must be non-negative");
      CUDA_KERNEL_ASSERT(
          indices_start <= indices_end &&
          "indices_start must not exceed indices_end");
      CUDA_KERNEL_ASSERT(
          indices_end <= num_indices &&
          "indices_end must not exceed num_indices");
    } else if (
        indices_start < 0 || indices_start > indices_end ||
        indices_end > num_indices) {
      if (bounds_check_mode == BoundsCheckMode::WARNING) {
        if (threadIdx.x == 0 && gpuAtomicIncrement(&warning[0]) == 0) {
          printf(
              "EmbeddingBoundsCheck (VBE %s): (at least one) Out of bounds access for "
              "batch: %d, table: %d, indices_start: %lld, indices_end: %lld,"
              " num_indices: %lld. Setting indices_start and indices_end within "
              "the range.\n",
              vbe ? "true" : "false",
              b,
              t,
              static_cast<int64_t>(indices_start),
              static_cast<int64_t>(indices_end),
              static_cast<int64_t>(num_indices));
        }
      }
      indices_start = std::max(
          static_cast<index_t>(0), std::min(indices_start, num_indices));
      indices_end = std::max(indices_start, std::min(indices_end, num_indices));
      // Only thread 0 writes back the adjusted offsets to avoid the intra-warp
      // race; no sync needed since offsets are not re-read in this kernel.
      if (threadIdx.x == 0) {
        offsets[b_t] = indices_start;
        offsets[b_t + 1] = indices_end;
      }
    }

    const auto L = indices_end - indices_start;
    for (index_t i = static_cast<index_t>(threadIdx.x); i < L;
         i += static_cast<index_t>(fbgemm_gpu::kWarpSize)) {
      const auto idx = indices[indices_start + i];
      if (idx == -1) {
        // -1 indicates pruned rows.
        continue;
      }
      if (bounds_check_mode == BoundsCheckMode::FATAL) {
        CUDA_KERNEL_ASSERT(
            idx >= 0 && "Failed idx >= 0 in bounds_check_indices");
        CUDA_KERNEL_ASSERT(
            idx < num_rows && "Failed idx < num_rows in bounds_check_indices");
      } else if (bounds_check_mode == BoundsCheckMode::WARNING) {
        if (idx < 0 || idx >= num_rows) {
          if (gpuAtomicIncrement(&warning[0]) == 0) {
            printf(
                "EmbeddingBoundsCheck (VBE %s): (at least one) Out of bounds access for batch: %d, table: %d, bag element: %lld, idx: %lld, num_rows: %lld, indices_start: %lld, indices_end: %lld, T: %d, B: %d, b_t: %d. Setting idx to zero.\n",
                vbe ? "true" : "false",
                b,
                t,
                static_cast<int64_t>(i),
                static_cast<int64_t>(idx),
                num_rows,
                static_cast<int64_t>(indices_start),
                static_cast<int64_t>(indices_end),
                T,
                B,
                b_t);
          }
          indices[indices_start + i] = 0;
        }
      } else if (bounds_check_mode == BoundsCheckMode::IGNORE) {
        if (idx < 0 || idx >= num_rows) {
          indices[indices_start + i] = 0;
        }
      }
    }

    if (disable_offsets_adjustment ||
        bounds_check_mode == BoundsCheckMode::FATAL) {
      if (b_t == 0 && threadIdx.x == 0) {
        CUDA_KERNEL_ASSERT(
            num_indices == offsets[total_B] &&
            "num_indices must match the last element in offsets");
      }
    } else if (num_indices != offsets[total_B]) {
      // The last-element check is a single global condition; one thread handles
      // the warning and the correction (for both WARNING and IGNORE).
      if (b_t == 0 && threadIdx.x == 0) {
        if (bounds_check_mode == BoundsCheckMode::WARNING) {
          if (gpuAtomicIncrement(&warning[0]) == 0) {
            printf(
                "EmbeddingBoundsCheck (VBE %s): the last element in offsets is incorrect for "
                "total batch size %s: %d, total table num T: %d, "
                " last element in offsets: %lld, indices size: %lld. "
                " Setting the last element in offsets to be indices size.\n",
                vbe ? "true" : "false",
                vbe ? "total_B" : "B",
                vbe ? total_B : B,
                T,
                static_cast<int64_t>(offsets[total_B]),
                static_cast<int64_t>(num_indices));
          }
        }
        offsets[total_B] = num_indices;
      }
    }
#ifdef USE_ROCM
  } // for bt0 (grid-stride loop, ROCm only)
#endif
}

void _bounds_check_indices_cuda_v1(
    Tensor& rows_per_table,
    Tensor& indices,
    Tensor& offsets,
    BoundsCheckMode bounds_check_mode,
    Tensor& warning,
    const std::optional<Tensor>& /*weights*/,
    const std::optional<Tensor>& B_offsets,
    int64_t max_B,
    const std::optional<Tensor>& /*b_t_map*/,
    int32_t /*info_b_num_bits*/,
    uint32_t /*info_B_mask*/,
    int64_t T,
    int64_t B,
    int64_t /*total_B*/,
    bool vbe,
    bool prefetch_pipeline,
    bool disable_offsets_adjustment) {
  TORCH_CHECK(
      !prefetch_pipeline,
      "bounds_check_indices_v1 does not support prefetch_pipeline=true")

  CUDA_DEVICE_GUARD(rows_per_table);

  if (bounds_check_mode == BoundsCheckMode::WARNING) {
    warning.zero_();
  }

  constexpr size_t kNumThreads = 256;
  const auto max_B_ = vbe ? max_B : B;

  // Flat path: one thread owns a contiguous run of indices instead of a warp
  // owning a row, so the cost tracks indices rather than rows. Restricted to
  // the plain configuration -- VBE needs the B_offsets row mapping, FATAL wants
  // v1's asserts, and disable_offsets_adjustment changes the offsets contract.
  const auto total_B_rows = static_cast<int32_t>(offsets.size(0) - 1);
  const auto num_indices_host = indices.size(0);
  const int64_t avg_segment_length =
      (total_B_rows > 0) ? (num_indices_host / total_B_rows) : 0;
  constexpr int64_t kFlatMaxAvgSegment = 24;
  const bool flat_eligible = !vbe && !disable_offsets_adjustment &&
      bounds_check_mode != BoundsCheckMode::FATAL && total_B_rows > 0 &&
      num_indices_host > 0 && B > 0;
  constexpr int32_t kFlatElemsPerThread = 4;
  constexpr int32_t kFlatSmemRows = 1024;
  constexpr int32_t kFlatThreads = 256;
  if (flat_eligible &&
      use_flat_bounds_check(avg_segment_length <= kFlatMaxAvgSegment)) {
    const int64_t elems_per_block =
        static_cast<int64_t>(kFlatThreads) * kFlatElemsPerThread;
    const int32_t num_blocks = static_cast<int32_t>(
        (num_indices_host + elems_per_block - 1) / elems_per_block);
    const bool warn = bounds_check_mode == BoundsCheckMode::WARNING;

    AT_DISPATCH_INDEX_TYPES(
        indices.scalar_type(), "bounds_check_indices_flat", [&] {
          FBGEMM_LAUNCH_KERNEL(
              (bounds_check_offsets_kernel<index_t>),
              div_round_up(total_B_rows, 256),
              256,
              0,
              at::cuda::getCurrentCUDAStream(),
              total_B_rows,
              static_cast<index_t>(num_indices_host),
              offsets.data_ptr<index_t>(),
              warning.data_ptr<int64_t>(),
              warn);

          FBGEMM_LAUNCH_KERNEL(
              (bounds_check_indices_kernel_flat<
                  kFlatElemsPerThread,
                  kFlatSmemRows,
                  index_t>),
              num_blocks,
              kFlatThreads,
              0,
              at::cuda::getCurrentCUDAStream(),
              rows_per_table.data_ptr<int64_t>(),
              indices.data_ptr<index_t>(),
              offsets.data_ptr<index_t>(),
              total_B_rows,
              static_cast<int32_t>(B),
              static_cast<index_t>(num_indices_host),
              bounds_check_mode,
              warning.data_ptr<int64_t>());
        });
    return;
  }

  AT_DISPATCH_INDEX_TYPES(
      indices.scalar_type(), "bounds_check_indices_cuda_v1", [&] {
        const auto bounds_check_kernel =
            (vbe ? bounds_check_indices_kernel_v1<index_t, true>
                 : bounds_check_indices_kernel_v1<index_t, false>);
        FBGEMM_LAUNCH_DSA_KERNEL(
            bounds_check_kernel,
            utils::cuda::cap_grid_dim_x(
                div_round_up(
                    max_B_ * T, kNumThreads / fbgemm_gpu::kWarpSizeHost()),
                kNumThreads,
                at::cuda::getCurrentCUDAStream()),
            dim3(
                fbgemm_gpu::kWarpSizeHost(),
                kNumThreads / fbgemm_gpu::kWarpSizeHost()),
            0,
            at::cuda::getCurrentCUDAStream(),
            PTA_B(rows_per_table, int64_t, 1, 32),
            PTA_B(indices, index_t, 1, 32),
            PTA_B(offsets, index_t, 1, 32),
            vbe ? B_offsets.value().data_ptr<int32_t>() : nullptr,
            bounds_check_mode,
            PTA_B(warning, int64_t, 1, 32),
            FixedDivisor(max_B_),
            disable_offsets_adjustment);
      });
}
