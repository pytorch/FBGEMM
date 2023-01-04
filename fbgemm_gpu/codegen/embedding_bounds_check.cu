/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "fbgemm_gpu/embedding_backward_template_helpers.cuh"

using Tensor = at::Tensor;
using namespace fbgemm_gpu;

template <typename index_t>
__device__ void adjust_offset_kernel(
    index_t& indices_start,
    index_t& indices_end,
    const index_t num_indices,
    index_t* const offset_acc_start,
    index_t* const offset_acc_end) {
  indices_start =
      std::max(static_cast<index_t>(0), std::min(indices_start, num_indices));
  indices_end = std::max(indices_start, std::min(indices_end, num_indices));
  *offset_acc_start = indices_start;
  *offset_acc_end = indices_end;
}

template <typename index_t>
__global__ __launch_bounds__(kMaxThreads) void bounds_check_indices_kernel(
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        rows_per_table,
    at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> indices,
    at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> offsets,
    const int64_t bounds_check_mode_,
    at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> warning,
    FixedDivisor fd) {
  int32_t T = rows_per_table.size(0);
  int32_t B = (offsets.size(0) - 1) / T;

  int32_t b_t = blockIdx.x * blockDim.y + threadIdx.y;
  int32_t b; // = b_t % B;
  int32_t t; // = b_t / B;
  fd.DivMod(b_t, &t, &b);
  if (t >= T) {
    return;
  }
  auto bounds_check_mode = static_cast<BoundsCheckMode>(bounds_check_mode_);

  auto num_rows = rows_per_table[t];
  auto indices_start = offsets[t * B + b];
  auto indices_end = offsets[t * B + b + 1];
  index_t num_indices = indices.size(0);

  if (bounds_check_mode == BoundsCheckMode::FATAL) {
    CUDA_KERNEL_ASSERT(indices_start >= 0);
    CUDA_KERNEL_ASSERT(indices_start <= indices_end);
    CUDA_KERNEL_ASSERT(indices_end <= num_indices);
  } else if (bounds_check_mode == BoundsCheckMode::WARNING) {
    if (indices_start < 0 || indices_start > indices_end ||
        indices_end > num_indices) {
      if (gpuAtomicIncrement(&warning[0]) == 0) {
        printf(
            "EmbeddingBoundsCheck: (at least one) Out of bounds access for "
            "batch: %lld, table: %lld, indices_start: %lld, indices_end: %lld,"
            " num_indices: %lld. Setting indices_start and indices_end within "
            "the range.\n",
            static_cast<int64_t>(b),
            static_cast<int64_t>(t),
            static_cast<int64_t>(indices_start),
            static_cast<int64_t>(indices_end),
            static_cast<int64_t>(num_indices));
      }
      adjust_offset_kernel(
          indices_start,
          indices_end,
          num_indices,
          &offsets[t * B + b],
          &offsets[t * B + b + 1]);
    }
  } else if (bounds_check_mode == BoundsCheckMode::IGNORE) {
    adjust_offset_kernel(
        indices_start,
        indices_end,
        num_indices,
        &offsets[t * B + b],
        &offsets[t * B + b + 1]);
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
      CUDA_KERNEL_ASSERT(idx >= 0 && "Failed idx >= 0 in bounds_check_indices");
      CUDA_KERNEL_ASSERT(
          idx < num_rows && "Failed idx < num_rows in bounds_check_indices");
    } else if (bounds_check_mode == BoundsCheckMode::WARNING) {
      if (idx < 0 || idx >= num_rows) {
        if (gpuAtomicIncrement(&warning[0]) == 0) {
          printf(
              "EmbeddingBoundsCheck: (at least one) Out of bounds access for batch: %lld, table: %lld, bag element: %lld, idx: %lld, num_rows: %lld, indices_start: %lld, indices_end: %lld, T: %d, B: %d, b_t: %d. Setting idx to zero.\n",
              static_cast<int64_t>(b),
              static_cast<int64_t>(t),
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

  if (bounds_check_mode == BoundsCheckMode::FATAL) {
    CUDA_KERNEL_ASSERT(num_indices == offsets[B * T]);
  } else if (bounds_check_mode == BoundsCheckMode::WARNING) {
    if (num_indices != offsets[B * T]) {
      if (gpuAtomicIncrement(&warning[0]) == 0) {
        printf(
            "EmbeddingBoundsCheck: the last element in offsets is incorrect for "
            "total batch size B: %lld, total table num T: %lld, "
            " last element in offsets: %lld, indices size: %lld. "
            " Setting the last element in offsets to be indices size.\n",
            static_cast<int64_t>(B),
            static_cast<int64_t>(T),
            static_cast<int64_t>(offsets[B * T]),
            static_cast<int64_t>(num_indices));
      }
      offsets[B * T] = num_indices;
    }
  } else if (bounds_check_mode == BoundsCheckMode::IGNORE) {
    if (num_indices != offsets[B * T]) {
      offsets[B * T] = num_indices;
    }
  }
}

void bounds_check_indices_cuda(
    Tensor& rows_per_table,
    Tensor& indices,
    Tensor& offsets,
    int64_t bounds_check_mode_,
    Tensor& warning,
    c10::optional<Tensor> weights) {
  TENSOR_ON_CUDA_GPU(rows_per_table);
  TENSOR_ON_CUDA_GPU(indices);
  TENSOR_ON_CUDA_GPU(offsets);
  TENSOR_ON_CUDA_GPU(warning);
  TENSOR_EMPTY_OR_ON_CUDA_GPU(weights);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(rows_per_table.get_device());

  const int32_t T = rows_per_table.size(0);
  const int32_t B = (offsets.size(0) - 1) / T;
  if (B == 0 || T == 0) {
    return;
  }
  const auto bounds_check_mode =
      static_cast<BoundsCheckMode>(bounds_check_mode_);
  if (bounds_check_mode == BoundsCheckMode::WARNING) {
    warning.zero_();
  }
  const int64_t num_indices = indices.size(0);

  TORCH_CHECK(
      offsets.size(0) == B * T + 1,
      "offsets size " + std::to_string(offsets.size(0)) +
          " is not equal to B (" + std::to_string(B) + ") * T (" +
          std::to_string(T) + ") + 1");
  if (weights.has_value()) {
    TORCH_CHECK(
        weights.value().size(0) == num_indices,
        "weights size " + std::to_string(weights.value().size(0)) +
            " is not equal to indices size " + std::to_string(num_indices));
  }

  constexpr size_t kNumThreads = 256;

  AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "bounds_check_indices", [&] {
    bounds_check_indices_kernel<index_t>
        <<<div_round_up(B * T, kNumThreads / fbgemm_gpu::kWarpSize),
           dim3(fbgemm_gpu::kWarpSize, kNumThreads / fbgemm_gpu::kWarpSize),
           0,
           at::cuda::getCurrentCUDAStream()>>>(
            rows_per_table
                .packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
            indices.packed_accessor32<index_t, 1, at::RestrictPtrTraits>(),
            offsets.packed_accessor32<index_t, 1, at::RestrictPtrTraits>(),
            bounds_check_mode_,
            warning.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
            FixedDivisor(B));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename index_t>
__global__ __launch_bounds__(kMaxThreads) void bounds_check_row_indices_kernel(
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        rows_per_table,
    at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        update_row_indices,
    at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        update_table_indices,
    const int64_t bounds_check_mode_,
    at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> warning) {
  int32_t T = rows_per_table.size(0);

  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= update_row_indices.size(0)) {
    return;
  }
  const int table_idx = update_table_indices[idx];
  const auto row_idx = update_row_indices[idx];

  auto bounds_check_mode = static_cast<BoundsCheckMode>(bounds_check_mode_);

  if (bounds_check_mode == BoundsCheckMode::FATAL) {
    CUDA_KERNEL_ASSERT(table_idx >= 0);
    CUDA_KERNEL_ASSERT(table_idx < T);
  } else if (bounds_check_mode == BoundsCheckMode::WARNING) {
    if (table_idx < 0 || table_idx >= T) {
      if (gpuAtomicIncrement(&warning[0]) == 0) {
        printf(
            "EmbeddingBoundsCheck: (at least one) Out of bounds access for idx: %lld, table_idx: %lld. Setting table_idx to zero.\n",
            static_cast<int64_t>(idx),
            static_cast<int64_t>(table_idx));
      }
      update_table_indices[idx] = 0;
    }
  } else if (bounds_check_mode == BoundsCheckMode::IGNORE) {
    update_table_indices[idx] = 0;
  }

  auto num_rows = rows_per_table[table_idx];

  if (row_idx == -1) {
    // -1 indicates pruned rows.
    return;
  }
  if (bounds_check_mode == BoundsCheckMode::FATAL) {
    CUDA_KERNEL_ASSERT(
        row_idx >= 0 && "Failed row_idx >= 0 in bounds_check_indices");
    CUDA_KERNEL_ASSERT(
        row_idx < num_rows &&
        "Failed row_idx < num_rows in bounds_check_indices");
  } else if (bounds_check_mode == BoundsCheckMode::WARNING) {
    if (row_idx < 0 || row_idx >= num_rows) {
      if (gpuAtomicIncrement(&warning[0]) == 0) {
        printf(
            "EmbeddingBoundsCheck: (at least one) Out of bounds access for idx: %lld, table_idx: %lld, row_idx: %lld, num_rows: %lld. Setting row_idx to zero.\n",
            static_cast<int64_t>(idx),
            static_cast<int64_t>(table_idx),
            static_cast<int64_t>(row_idx),
            num_rows);
      }
      update_row_indices[idx] = 0;
    }
  } else if (bounds_check_mode == BoundsCheckMode::IGNORE) {
    if (idx < 0 || idx >= num_rows) {
      update_row_indices[idx] = 0;
    }
  }
}

void bounds_check_row_indices_cuda(
    Tensor& rows_per_table,
    Tensor& update_row_indices,
    Tensor& update_table_indices,
    int64_t bounds_check_mode_,
    Tensor& warning) {
  TENSOR_ON_CUDA_GPU(rows_per_table);
  TENSOR_ON_CUDA_GPU(update_row_indices);
  TENSOR_ON_CUDA_GPU(update_table_indices);
  TENSOR_ON_CUDA_GPU(warning);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(rows_per_table.get_device());

  const int32_t T = rows_per_table.size(0);
  if (T == 0) {
    return;
  }
  const auto bounds_check_mode =
      static_cast<BoundsCheckMode>(bounds_check_mode_);
  if (bounds_check_mode == BoundsCheckMode::WARNING) {
    warning.zero_();
  }
  const int64_t num_indices = update_row_indices.size(0);

  constexpr size_t kNumThreads = 256;

  AT_DISPATCH_INDEX_TYPES(
      update_row_indices.scalar_type(), "bounds_check_indices", [&] {
        bounds_check_row_indices_kernel<index_t>
            <<<div_round_up(num_indices, kNumThreads),
               kNumThreads,
               0,
               at::cuda::getCurrentCUDAStream()>>>(
                rows_per_table
                    .packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                update_row_indices
                    .packed_accessor32<index_t, 1, at::RestrictPtrTraits>(),
                update_table_indices
                    .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                bounds_check_mode_,
                warning.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
}
