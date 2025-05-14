/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "fbgemm_gpu/utils/embedding_bounds_check_common.cuh"

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
    TORCH_DSA_KERNEL_ARGS) {
  int32_t T = rows_per_table.size(0);
  int32_t total_B = offsets.size(0) - 1;
  int32_t B = vbe ? 0 : (total_B / T);

  const index_t num_indices = indices.size(0);
  const auto b_t_start = blockIdx.x * blockDim.y + threadIdx.y;
  index_t invalid_i = -1, invalid_idx = -1;
  int32_t invalid_b_t = -1;

  // Check the last element
  if (b_t_start == 0 && threadIdx.x == 0) {
    if (bounds_check_mode == BoundsCheckMode::FATAL) {
      CUDA_KERNEL_ASSERT2(num_indices == offsets[total_B]);
    } else if (bounds_check_mode == BoundsCheckMode::WARNING) {
      if (num_indices != offsets[total_B]) {
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
        offsets[total_B] = num_indices;
      }
    } else if (bounds_check_mode == BoundsCheckMode::IGNORE) {
      if (num_indices != offsets[total_B]) {
        offsets[total_B] = num_indices;
      }
    }
  }

  for (auto b_t = blockIdx.x * blockDim.y + threadIdx.y; b_t < total_B;
       b_t += blockDim.y * gridDim.x) {
    // Compute b and t
    int32_t b;
    int32_t t;
    if (vbe) {
      const auto info = *reinterpret_cast<const uint32_t*>(&b_t_map[b_t]);
      *reinterpret_cast<uint32_t*>(&t) = info >> info_B_num_bits;
      *reinterpret_cast<uint32_t*>(&b) = info & info_B_mask;
    } else {
      fd.DivMod(b_t, &t, &b);
    }

    const auto num_rows = rows_per_table[t];
    auto indices_start = offsets[b_t];
    auto indices_end = offsets[b_t + 1];

    if (bounds_check_mode == BoundsCheckMode::FATAL) {
      CUDA_KERNEL_ASSERT2(indices_start >= 0);
      CUDA_KERNEL_ASSERT2(indices_start <= indices_end);
      CUDA_KERNEL_ASSERT2(indices_end <= num_indices);
    } else if (bounds_check_mode == BoundsCheckMode::WARNING) {
      if (indices_start < 0 || indices_start > indices_end ||
          indices_end > num_indices) {
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
        adjust_offset_kernel(
            indices_start,
            indices_end,
            num_indices,
            &offsets[b_t],
            &offsets[b_t + 1]);
      }
    } else if (bounds_check_mode == BoundsCheckMode::IGNORE) {
      adjust_offset_kernel(
          indices_start,
          indices_end,
          num_indices,
          &offsets[b_t],
          &offsets[b_t + 1]);
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
        CUDA_KERNEL_ASSERT2(
            idx >= 0 && "Failed idx >= 0 in bounds_check_indices");
        CUDA_KERNEL_ASSERT2(
            idx < num_rows && "Failed idx < num_rows in bounds_check_indices");
      } else if (bounds_check_mode == BoundsCheckMode::WARNING) {
        if (idx < 0 || idx >= num_rows) {
          invalid_i = i;
          invalid_idx = idx;
          invalid_b_t = b_t;
          indices[indices_start + i] = 0;
          // Count warnings to keep the unit tests happy
          gpuAtomicIncrement(&warning[0]);
        }
      } else if (bounds_check_mode == BoundsCheckMode::IGNORE) {
        if (idx < 0 || idx >= num_rows) {
          indices[indices_start + i] = 0;
        }
      }
    }
  } // for b_t

  if (bounds_check_mode == BoundsCheckMode::WARNING && invalid_i != -1 &&
      static_cast<int64_t>(atomicAdd(
          reinterpret_cast<unsigned long long int*>(&warning[0]), 0)) == 0) {
    int32_t b;
    int32_t t;

    fd.DivMod(invalid_b_t, &t, &b);

    int32_t B = vbe ? (B_offsets[t + 1] - B_offsets[t]) : (total_B / T);

    printf(
        "EmbeddingBoundsCheck (VBE %s): (at least one) Out of bounds access for batch: %d, table: %d, bag element: %lld, idx: %lld, num_rows: %lld, indices_start: %lld, indices_end: %lld, T: %d, B: %d, b_t: %d. Setting idx to zero.\n",
        vbe ? "true" : "false",
        b,
        t,
        static_cast<int64_t>(invalid_i),
        static_cast<int64_t>(invalid_idx),
        rows_per_table[t],
        static_cast<int64_t>(offsets[invalid_b_t]),
        static_cast<int64_t>(offsets[invalid_b_t + 1]),
        T,
        B,
        invalid_b_t);
  }
}

void _bounds_check_indices_cuda_v2(
    Tensor& rows_per_table,
    Tensor& indices,
    Tensor& offsets,
    BoundsCheckMode bounds_check_mode,
    Tensor& warning,
    const std::optional<Tensor>& weights,
    const std::optional<Tensor>& B_offsets,
    int64_t /*max_B*/,
    const std::optional<Tensor>& b_t_map,
    int32_t info_B_num_bits,
    uint32_t info_B_mask,
    int64_t /*T*/,
    int64_t B,
    int64_t total_B,
    bool vbe,
    bool prefetch_pipeline) {
  if (vbe) {
    TORCH_CHECK(b_t_map.has_value());
    TENSOR_NDIM_EQUALS(b_t_map.value(), 1);
  }

  CUDA_DEVICE_GUARD(rows_per_table);

  if (bounds_check_mode == BoundsCheckMode::WARNING) {
    warning.zero_();
  }

  constexpr size_t kNumThreads = 1024;
  auto grid_dim =
      min(div_round_up(total_B, kNumThreads / fbgemm_gpu::kWarpSize),
          get_max_thread_blocks_());
  if (prefetch_pipeline) {
    // Limit the grid size to PREFETCH_KERNEL_MAX_BLOCKS if running this kernel
    // on the prefetch stream
    constexpr int PREFETCH_KERNEL_MAX_BLOCKS = 8;
    grid_dim = min(grid_dim, PREFETCH_KERNEL_MAX_BLOCKS);
  }

#define INVOKE_BOUNDS_CHECK_INDICES(MODE)                                      \
  if (bounds_check_mode == MODE) {                                             \
    AT_DISPATCH_INDEX_TYPES(                                                   \
        indices.scalar_type(), "bounds_check_indices_cuda", [&] {              \
          [[maybe_unused]] const auto func_name =                              \
              "bounds_check_indices_cuda_v2";                                  \
          const auto bounds_check_kernel =                                     \
              (vbe ? bounds_check_indices_kernel_v2<index_t, true, MODE>       \
                   : bounds_check_indices_kernel_v2<index_t, false, MODE>);    \
          FBGEMM_LAUNCH_DSA_KERNEL(                                            \
              bounds_check_kernel,                                             \
              grid_dim,                                                        \
              dim3(                                                            \
                  fbgemm_gpu::kWarpSize, kNumThreads / fbgemm_gpu::kWarpSize), \
              0,                                                               \
              at::cuda::getCurrentCUDAStream(),                                \
              PTA_B(rows_per_table, int64_t, 1, 32),                           \
              PTA_B(indices, index_t, 1, 32),                                  \
              PTA_B(offsets, index_t, 1, 32),                                  \
              vbe ? B_offsets.value().data_ptr<int32_t>() : nullptr,           \
              PTA_B(warning, int64_t, 1, 32),                                  \
              FixedDivisor(B),                                                 \
              vbe ? b_t_map.value().data_ptr<int32_t>() : nullptr,             \
              info_B_num_bits,                                                 \
              info_B_mask);                                                    \
        });                                                                    \
  }

  INVOKE_BOUNDS_CHECK_INDICES(BoundsCheckMode::FATAL)
  INVOKE_BOUNDS_CHECK_INDICES(BoundsCheckMode::WARNING)
  INVOKE_BOUNDS_CHECK_INDICES(BoundsCheckMode::IGNORE)

#undef INVOKE_BOUNDS_CHECK_INDICES
}
