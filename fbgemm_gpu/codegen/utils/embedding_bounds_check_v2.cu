/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <cstdlib>
#include "fbgemm_gpu/config/feature_gates.h"
#include "fbgemm_gpu/utils/embedding_bounds_check_common.cuh"

using namespace fbgemm_gpu;

template <typename index_t, bool vbe, BoundsCheckMode bounds_check_mode>
__global__ __launch_bounds__(kMaxThreads) void bounds_check_indices_kernel_v2(
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        rows_per_table,
    pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> indices,
    pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> offsets,
    const int32_t* const B_offsets, // Use a raw pointer to avoid creating a
                                    // dummy PackedTensorAccessor
    pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> warning,
    FixedDivisor fd,
    const int32_t* const b_t_map,
    const int32_t info_B_num_bits,
    const int32_t info_B_mask,
    const bool disable_offsets_adjustment,
    TORCH_DSA_KERNEL_ARGS) {
  int32_t T = rows_per_table.size(0);
  int32_t total_B = offsets.size(0) - 1;
  int32_t B = vbe ? 0 : (total_B / T);

  const index_t num_indices = indices.size(0);
  const int64_t b_t_start =
      static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
  constexpr bool is_warning_mode =
      bounds_check_mode == BoundsCheckMode::WARNING ||
      bounds_check_mode == BoundsCheckMode::WARNING_ALLOW_TRAILING_INDICES;
#ifdef USE_ROCM
  // Deliberately ROCm-only: NVIDIA emits the WARNING printf inline in the loop,
  // so allocating this state there would cost registers the kernel never uses,
  // which at 1024 threads/block is enough to drop occupancy to 1 block/SM.
  index_t invalid_i = -1, invalid_idx = -1;
  int32_t invalid_b_t = -1;
  int64_t warning_inc = 0;
#endif

  // Last-element check; one thread only.
  if (b_t_start == 0 && threadIdx.x == 0) {
    const bool invalid_final_offset = offsets[total_B] > num_indices ||
        (bounds_check_mode != BoundsCheckMode::WARNING_ALLOW_TRAILING_INDICES &&
         offsets[total_B] != num_indices);
    if (disable_offsets_adjustment ||
        bounds_check_mode == BoundsCheckMode::FATAL) {
      CUDA_KERNEL_ASSERT(
          !invalid_final_offset &&
          "num_indices must match the last element in offsets");
    } else if (invalid_final_offset) {
      if (is_warning_mode) {
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

  // blockDim.x is the group width the launch actually chose -- kWarpSizeHost()
  // on the warp-wide path, the adaptive width otherwise -- so stride by it
  // rather than re-deriving the host's decision from a predicate that has to
  // be kept in sync with the launch. A predicate that drifted out of sync
  // (blockDim.x < stride) would silently skip indices and drop their bounds
  // check, with nothing to catch it at compile or run time.
  const auto index_stride = static_cast<index_t>(blockDim.x);

  for (int64_t b_t =
           static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
       b_t < total_B;
       b_t += static_cast<int64_t>(blockDim.y) * gridDim.x) {
    // Compute b and t
    int32_t b;
    int32_t t;
    if (vbe) {
      const auto info = *reinterpret_cast<const uint32_t*>(&b_t_map[b_t]);
      *reinterpret_cast<uint32_t*>(&t) = info >> info_B_num_bits;
      *reinterpret_cast<uint32_t*>(&b) = info & info_B_mask;
    } else {
      fd.DivMod(static_cast<int32_t>(b_t), &t, &b);
    }

    const auto num_rows = rows_per_table[t];
    auto indices_start = offsets[b_t];
    auto indices_end = offsets[b_t + 1];

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
      if (is_warning_mode) {
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
         i += index_stride) {
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
      } else if (is_warning_mode) {
        if (idx < 0 || idx >= num_rows) {
#ifdef USE_ROCM
          // Record only; the print slot is claimed once per block after the
          // loop. Printing here would let every wavefront race for it.
          invalid_i = i;
          invalid_idx = idx;
          invalid_b_t = b_t;
          warning_inc += 1;
#else
          // The fused increment-and-test is what limits this to one print per
          // launch: only the thread that observes 0 may print.
          if (gpuAtomicIncrement(&warning[0]) == 0) {
            const int32_t B_print =
                vbe ? (B_offsets[t + 1] - B_offsets[t]) : (total_B / T);
            printf(
                "EmbeddingBoundsCheck (VBE %s): (at least one) Out of bounds access for "
                "batch: %d, table: %d, bag element: %lld, idx: %lld, num_rows: %lld, "
                "indices_start: %lld, indices_end: %lld, T: %d, B: %d, b_t: %d. "
                "Setting idx to zero.\n",
                vbe ? "true" : "false",
                b,
                t,
                static_cast<int64_t>(i),
                static_cast<int64_t>(idx),
                rows_per_table[t],
                static_cast<int64_t>(indices_start),
                static_cast<int64_t>(indices_end),
                T,
                B_print,
                b_t);
          }
#endif
          indices[indices_start + i] = 0;
        }
      } else if (bounds_check_mode == BoundsCheckMode::IGNORE) {
        if (idx < 0 || idx >= num_rows) {
          indices[indices_start + i] = 0;
        }
      }
    }
  } // for b_t

#ifdef USE_ROCM
  // The unsynchronized read of warning[0] is a filter, not the decision: it
  // only skips the atomic once some block has already claimed the slot. The
  // gpuAtomicIncrement still decides, so a stale read costs an extra atomic,
  // never a second print.
  bool print_warning = false;
  if (is_warning_mode && warning_inc > 0 && warning[0] == 0) {
    print_warning = (gpuAtomicIncrement(&warning[0]) == 0);
  }

  if (print_warning) {
    int32_t b;
    int32_t t;

    // Must decode the same way the loop above did. fd is built from the
    // non-VBE B, so using it under vbe yields a t/b pair that indexes
    // rows_per_table[t] and B_offsets[t] out of range.
    if constexpr (vbe) {
      const auto info =
          *reinterpret_cast<const uint32_t*>(&b_t_map[invalid_b_t]);
      *reinterpret_cast<uint32_t*>(&t) = info >> info_B_num_bits;
      *reinterpret_cast<uint32_t*>(&b) = info & info_B_mask;
    } else {
      fd.DivMod(invalid_b_t, &t, &b);
    }

    const int32_t batch_size =
        vbe ? (B_offsets[t + 1] - B_offsets[t]) : (total_B / T);

    printf(
        "EmbeddingBoundsCheck (VBE %s): (at least one) Out of bounds access for "
        "batch: %d, table: %d, bag element: %lld, idx: %lld, num_rows: %lld, "
        "indices_start: %lld, indices_end: %lld, T: %d, B: %d, b_t: %d. "
        "Setting idx to zero.\n",
        vbe ? "true" : "false",
        b,
        t,
        static_cast<int64_t>(invalid_i),
        static_cast<int64_t>(invalid_idx),
        rows_per_table[t],
        static_cast<int64_t>(offsets[invalid_b_t]),
        static_cast<int64_t>(offsets[invalid_b_t + 1]),
        T,
        batch_size,
        invalid_b_t);
  }
#endif
}

// Thread-per-bag kernel: each thread handles one bag (b_t) independently.
// Better than warp-per-b_t when avg L is small: a warp per b_t leaves lanes
// idle whenever L is below the warp size (31 of 32 at L = 1).
//
// Offsets are no longer adjusted: invalid offsets assert in every mode, as in
// the default behavior of both bounds check kernels (gated by
// DISABLE_OFFSETS_ADJUSTMENT, which is true by default).
// WARNING and IGNORE only adjust out-of-range indices.
template <typename index_t, bool vbe, BoundsCheckMode bounds_check_mode>
__global__
__launch_bounds__(kMaxThreads) void bounds_check_indices_kernel_thread_per_bag(
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        rows_per_table,
    pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> indices,
    pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> offsets,
    const int32_t* const B_offsets,
    pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> warning,
    FixedDivisor fd,
    const int32_t* const b_t_map,
    const int32_t info_B_num_bits,
    const int32_t info_B_mask,
    TORCH_DSA_KERNEL_ARGS) {
  const int32_t T = rows_per_table.size(0);
  const int32_t total_B = offsets.size(0) - 1;
  const index_t num_indices = indices.size(0);

  if (blockIdx.x == 0 && threadIdx.x == 0) {
    CUDA_KERNEL_ASSERT(
        num_indices == offsets[total_B] &&
        "num_indices must match the last element in offsets");
  }

  for (int64_t b_t =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       b_t < total_B;
       b_t += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    int32_t b;
    int32_t t;
    if (vbe) {
      const auto info = *reinterpret_cast<const uint32_t*>(&b_t_map[b_t]);
      *reinterpret_cast<uint32_t*>(&t) = info >> info_B_num_bits;
      *reinterpret_cast<uint32_t*>(&b) = info & info_B_mask;
    } else {
      fd.DivMod(static_cast<int32_t>(b_t), &t, &b);
    }

    const auto num_rows = rows_per_table[t];
    const auto indices_start = offsets[b_t];
    const auto indices_end = offsets[b_t + 1];

    CUDA_KERNEL_ASSERT(
        indices_start >= 0 && "indices_start must be non-negative");
    CUDA_KERNEL_ASSERT(
        indices_start <= indices_end &&
        "indices_start must not exceed indices_end");
    CUDA_KERNEL_ASSERT(
        indices_end <= num_indices &&
        "indices_end must not exceed num_indices");

    const auto L = indices_end - indices_start;
    for (index_t i = 0; i < L; i++) {
      const auto idx = indices[indices_start + i];
      if (idx == -1) {
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
            const int32_t B_print =
                vbe ? (B_offsets[t + 1] - B_offsets[t]) : (total_B / T);
            printf(
                "EmbeddingBoundsCheck (VBE %s): (at least one) Out of bounds access for "
                "batch: %d, table: %d, bag element: %lld, idx: %lld, num_rows: %lld, "
                "indices_start: %lld, indices_end: %lld, T: %d, B: %d, b_t: %d. "
                "Setting idx to zero.\n",
                vbe ? "true" : "false",
                b,
                t,
                static_cast<int64_t>(i),
                static_cast<int64_t>(idx),
                rows_per_table[t],
                static_cast<int64_t>(indices_start),
                static_cast<int64_t>(indices_end),
                T,
                B_print,
                static_cast<int32_t>(b_t));
          }
          indices[indices_start + i] = 0;
        }
      } else if (bounds_check_mode == BoundsCheckMode::IGNORE) {
        if (idx < 0 || idx >= num_rows) {
          indices[indices_start + i] = 0;
        }
      }
    }
  }
}

void _bounds_check_indices_cuda_v2(
    Tensor& rows_per_table,
    Tensor& indices,
    Tensor& offsets,
    BoundsCheckMode bounds_check_mode,
    Tensor& warning,
    const std::optional<Tensor>& /*weights*/,
    const std::optional<Tensor>& B_offsets,
    int64_t /*max_B*/,
    const std::optional<Tensor>& b_t_map,
    int32_t info_B_num_bits,
    uint32_t info_B_mask,
    int64_t /*T*/,
    int64_t B,
    int64_t total_B,
    bool vbe,
    bool prefetch_pipeline,
    bool disable_offsets_adjustment) {
  if (vbe) {
    TORCH_CHECK(b_t_map.has_value());
    TENSOR_NDIM_EQUALS(b_t_map.value(), 1);
  }

  CUDA_DEVICE_GUARD(rows_per_table);

  if (bounds_check_mode == BoundsCheckMode::WARNING ||
      bounds_check_mode == BoundsCheckMode::WARNING_ALLOW_TRAILING_INDICES) {
    warning.zero_();
  }

#ifndef USE_ROCM
  // NVIDIA only; ROCm keeps the launch below. The thread-per-bag kernel does
  // not implement the trailing-indices mode.
  //
  // Experimentally on H100, thread-per-bag is used for avg L <= 2, and for
  // avg L <= 72 when there are enough bags. With few bags, each thread's L
  // serial loads are not hidden, while the warp kernel covers a bag in
  // ceil(L / 32) passes; the bag floor scales with the warp kernel's busy lanes
  // per pass, L / ceil(L / 32). When every bag has the same L and spans a
  // multiple of 64 bytes, the lanes of a warp read the same sector position of
  // different cache lines at once and each load waits about twice as long, so
  // such inputs use a higher floor.
  constexpr int32_t kThreadPerBagAlwaysMaxAvgL = 2;
  constexpr int32_t kThreadPerBagMaxAvgL = 72;
  constexpr int32_t kThreadPerBagMinBagsPerLane = 2048;
  constexpr int32_t kThreadPerBagMinBagsPerLaneAligned = 4096;
  const int32_t avg_L =
      total_B > 0 ? static_cast<int32_t>(indices.numel() / total_B) : 0;
  const bool aligned_bags = total_B > 0 && indices.numel() % total_B == 0 &&
      (avg_L * indices.element_size()) % 64 == 0;
  static const bool thread_per_bag_enabled =
      fbgemm_gpu::config::is_feature_enabled(
          fbgemm_gpu::config::FeatureGateName::BOUNDS_CHECK_THREAD_PER_BAG);
  const bool use_thread_per_bag_kernel = thread_per_bag_enabled &&
      bounds_check_mode != BoundsCheckMode::WARNING_ALLOW_TRAILING_INDICES &&
      (avg_L <= kThreadPerBagAlwaysMaxAvgL ||
       (avg_L <= kThreadPerBagMaxAvgL &&
        total_B >= (aligned_bags ? kThreadPerBagMinBagsPerLaneAligned
                                 : kThreadPerBagMinBagsPerLane) *
                (avg_L / div_round_up(avg_L, fbgemm_gpu::kWarpSizeHost()))));
  if (use_thread_per_bag_kernel) {
    constexpr size_t kThreadPerBagThreads = 1024;
    auto thread_per_bag_grid_dim = fbgemm_gpu::utils::cuda::cap_grid_dim_x(
        cuda_calc_xblock_count(total_B, kThreadPerBagThreads),
        kThreadPerBagThreads,
        at::cuda::getCurrentCUDAStream(),
        fbgemm_gpu::utils::cuda::BlockCapPolicy::Always);
    if (prefetch_pipeline) {
      constexpr int PREFETCH_KERNEL_MAX_BLOCKS = 8;
      thread_per_bag_grid_dim = std::min<uint32_t>(
          thread_per_bag_grid_dim, PREFETCH_KERNEL_MAX_BLOCKS);
    }

#define INVOKE_THREAD_PER_BAG_BOUNDS_CHECK(MODE)                     \
  if (bounds_check_mode == MODE) {                                   \
    AT_DISPATCH_INDEX_TYPES(                                         \
        indices.scalar_type(), "bounds_check_indices_cuda", [&] {    \
          [[maybe_unused]] const auto func_name =                    \
              "bounds_check_indices_cuda_thread_per_bag";            \
          const auto bounds_check_kernel =                           \
              (vbe ? bounds_check_indices_kernel_thread_per_bag<     \
                         index_t,                                    \
                         true,                                       \
                         MODE>                                       \
                   : bounds_check_indices_kernel_thread_per_bag<     \
                         index_t,                                    \
                         false,                                      \
                         MODE>);                                     \
          FBGEMM_LAUNCH_DSA_KERNEL(                                  \
              bounds_check_kernel,                                   \
              thread_per_bag_grid_dim,                               \
              dim3(kThreadPerBagThreads),                            \
              0,                                                     \
              at::cuda::getCurrentCUDAStream(),                      \
              PTA_B(rows_per_table, int64_t, 1, 32),                 \
              PTA_B(indices, index_t, 1, 32),                        \
              PTA_B(offsets, index_t, 1, 32),                        \
              vbe ? B_offsets.value().data_ptr<int32_t>() : nullptr, \
              PTA_B(warning, int64_t, 1, 32),                        \
              FixedDivisor(B),                                       \
              vbe ? b_t_map.value().data_ptr<int32_t>() : nullptr,   \
              info_B_num_bits,                                       \
              info_B_mask);                                          \
        });                                                          \
  }

    INVOKE_THREAD_PER_BAG_BOUNDS_CHECK(BoundsCheckMode::FATAL)
    INVOKE_THREAD_PER_BAG_BOUNDS_CHECK(BoundsCheckMode::WARNING)
    INVOKE_THREAD_PER_BAG_BOUNDS_CHECK(BoundsCheckMode::IGNORE)

#undef INVOKE_THREAD_PER_BAG_BOUNDS_CHECK
    return;
  }
#endif

  // The WARNING template is the register-heaviest of the three; at 1024
  // threads/block that is enough to cap occupancy at one block per SM.
  const bool is_warning_mode = bounds_check_mode == BoundsCheckMode::WARNING ||
      bounds_check_mode == BoundsCheckMode::WARNING_ALLOW_TRAILING_INDICES;
  const size_t kNumThreads = is_warning_mode ? 256 : 1024;
  auto num_threads = kNumThreads;
  auto thread_group_size = fbgemm_gpu::kWarpSizeHost();
#ifdef USE_ROCM
  // rocm vbe uses adaptive threading by default
  constexpr bool adaptive_vbe = true;
  static const bool adaptive_all = [] {
    const char* v = std::getenv("TBE_ROCM_BOUNDS_CHECK_ADAPTIVE_ALL");
    return v != nullptr && v[0] == '1' && v[1] == '\0';
  }();
#else
  // for nvidia we gate all adaptive threading behind flags
  static const bool adaptive_vbe = [] {
    const char* v = std::getenv("TBE_BOUNDS_CHECK_ADAPTIVE_THREADING_VBE");
    return v != nullptr && v[0] == '1' && v[1] == '\0';
  }();
  static const bool adaptive_all = [] {
    const char* v = std::getenv("TBE_BOUNDS_CHECK_ADAPTIVE_THREADING_ALL");
    return v != nullptr && v[0] == '1' && v[1] == '\0';
  }();
#endif
  // By default, we use a whole warp (nvda) or wavefront (amd) to process a
  // single bag. This can be quite inefficient if the bag sizes are very small
  // on average. Adaptive threading allows kernel to dynamically allocate less
  // threads when the average bag sizes are small, and vice versa.
  bool enable_adaptive_threading = (vbe && adaptive_vbe) || adaptive_all;
  if (enable_adaptive_threading &&
      (bounds_check_mode == BoundsCheckMode::WARNING ||
       bounds_check_mode == BoundsCheckMode::IGNORE)) {
    // Average pooling factor. Integer division, so sparse batches
    // (numel < total_B) and empty ones yield 0 and take the narrowest group.
    const auto average_L = total_B > 0 ? indices.numel() / total_B : 0;
    // Past this point each bag already fills a wavefront and the original
    // warp-wide 1024-thread launch wins on occupancy, so leave it alone.
    constexpr int64_t kMaxAdaptiveL = 512;
    if (average_L <= kMaxAdaptiveL) {
      // The prefetch path caps the grid at eight blocks, so keep full blocks.
      // Everywhere else drop to 256: the widths below put many groups in one
      // block, and a smaller block spreads those groups over more CUs than a
      // 1024-thread block does. Chosen in the same MI350 sweep as the widths.
      num_threads = prefetch_pipeline ? kNumThreads : 256;
      // No wave-level collectives are used, so one warp or wave can process
      // multiple bags. That holds for WARNING too: its post-loop work is a
      // lone gpuAtomicIncrement, and the kernel's block-wide reduction is
      // compile-time dead in that specialization, so nothing there is tied to
      // the logical group width. Narrow groups waste fewer lanes on the ragged
      // tail of each bag; wide groups supply more resident threads. These
      // widths were picked by sweeping group size against pooling factor on
      // MI350.
      if (average_L <= 4) {
        thread_group_size = 1;
      } else if (average_L <= 8) {
        thread_group_size = 2;
      } else if (average_L <= 16) {
        thread_group_size = 4;
      } else if (average_L <= 128) {
        thread_group_size = 8;
      } else if (average_L <= 256) {
        thread_group_size = 16;
      } else {
        thread_group_size = 32;
      }
      // average_L is a mean, so a batch of mostly-empty bags with a few long
      // ones picks a group too narrow for the long ones and walks them nearly
      // serially. Widening wastes lanes on short bags, but only once the
      // narrow launch already fills the device: at 256Ki bags a one-lane
      // launch is ~1024 blocks, so on a 256-CU MI350 the extra lanes were idle
      // regardless. Measured free below this point (1.0-1.1x on uniform bags,
      // 1.3-2.7x faster on skewed ones) and costly above it (3.1x slower on
      // the 22.7M-bag trace shape), so larger batches keep the ladder as is.
      constexpr int64_t kMaxUnsaturatedB = 256 * 1024;
      if (total_B <= kMaxUnsaturatedB) {
        thread_group_size = std::max(thread_group_size, 4);
      }
    }
  }
  auto grid_dim = fbgemm_gpu::utils::cuda::cap_grid_dim_x(
      cuda_calc_xblock_count(total_B, num_threads / thread_group_size),
      num_threads,
      at::cuda::getCurrentCUDAStream(),
      fbgemm_gpu::utils::cuda::BlockCapPolicy::Always);
  if (prefetch_pipeline) {
    // Limit the grid size to PREFETCH_KERNEL_MAX_BLOCKS if running this kernel
    // on the prefetch stream
    constexpr int PREFETCH_KERNEL_MAX_BLOCKS = 8;
    grid_dim = std::min<uint32_t>(grid_dim, PREFETCH_KERNEL_MAX_BLOCKS);
  }

#define INVOKE_BOUNDS_CHECK_INDICES(MODE)                                   \
  if (bounds_check_mode == MODE) {                                          \
    AT_DISPATCH_INDEX_TYPES(                                                \
        indices.scalar_type(), "bounds_check_indices_cuda", [&] {           \
          [[maybe_unused]] const auto func_name =                           \
              "bounds_check_indices_cuda_v2";                               \
          const auto bounds_check_kernel =                                  \
              (vbe ? bounds_check_indices_kernel_v2<index_t, true, MODE>    \
                   : bounds_check_indices_kernel_v2<index_t, false, MODE>); \
          FBGEMM_LAUNCH_DSA_KERNEL(                                         \
              bounds_check_kernel,                                          \
              grid_dim,                                                     \
              dim3(thread_group_size, num_threads / thread_group_size),     \
              0,                                                            \
              at::cuda::getCurrentCUDAStream(),                             \
              PTA_B(rows_per_table, int64_t, 1, 32),                        \
              PTA_B(indices, index_t, 1, 32),                               \
              PTA_B(offsets, index_t, 1, 32),                               \
              vbe ? B_offsets.value().data_ptr<int32_t>() : nullptr,        \
              PTA_B(warning, int64_t, 1, 32),                               \
              FixedDivisor(B),                                              \
              vbe ? b_t_map.value().data_ptr<int32_t>() : nullptr,          \
              info_B_num_bits,                                              \
              info_B_mask,                                                  \
              disable_offsets_adjustment);                                  \
        });                                                                 \
  }

  INVOKE_BOUNDS_CHECK_INDICES(BoundsCheckMode::FATAL)
  INVOKE_BOUNDS_CHECK_INDICES(BoundsCheckMode::WARNING)
  INVOKE_BOUNDS_CHECK_INDICES(BoundsCheckMode::WARNING_ALLOW_TRAILING_INDICES)
  INVOKE_BOUNDS_CHECK_INDICES(BoundsCheckMode::IGNORE)

#undef INVOKE_BOUNDS_CHECK_INDICES
}
