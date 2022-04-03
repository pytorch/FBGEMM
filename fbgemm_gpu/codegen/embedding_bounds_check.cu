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
__global__ __launch_bounds__(kMaxThreads) void bounds_check_indices_kernel(
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        rows_per_table,
    at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> indices,
    at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> offsets,
    int64_t bounds_check_mode_,
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
            int64_t(b),
            int64_t(t),
            int64_t(indices_start),
            int64_t(indices_end),
            int64_t(num_indices));
      }
      indices_start = std::max(
          static_cast<index_t>(0), std::min(indices_start, num_indices));
      indices_end = std::max(indices_start, std::min(indices_end, num_indices));
      offsets[t * B + b] = indices_start;
      offsets[t * B + b + 1] = indices_end;
    }
  } else if (bounds_check_mode == BoundsCheckMode::IGNORE) {
    indices_start =
        std::max(static_cast<index_t>(0), std::min(indices_start, num_indices));
    indices_end = std::max(indices_start, std::min(indices_end, num_indices));
    offsets[t * B + b] = indices_start;
    offsets[t * B + b + 1] = indices_end;
  }

  auto L = indices_end - indices_start;
  for (index_t i = (index_t)threadIdx.x; i < L;
       i += (index_t)fbgemm_gpu::kWarpSize) {
    auto idx = indices[indices_start + i];
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
              "EmbeddingBoundsCheck: (at least one) Out of bounds access for batch: %lld, table: %lld, bag element: %lld, idx: %lld, num_rows: %lld, indices_start: %lld, T: %d, B: %d, b_t: %d. Setting idx to zero.\n",
              int64_t(b),
              int64_t(t),
              int64_t(i),
              int64_t(idx),
              num_rows,
              int64_t(indices_start),
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
}

void bounds_check_indices_cuda(
    Tensor rows_per_table,
    Tensor indices,
    Tensor offsets,
    int64_t bounds_check_mode_,
    Tensor warning) {
  TENSOR_ON_CUDA_GPU(rows_per_table);
  TENSOR_ON_CUDA_GPU(indices);
  TENSOR_ON_CUDA_GPU(offsets);
  TENSOR_ON_CUDA_GPU(warning);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(rows_per_table.get_device());

  int32_t T = rows_per_table.size(0);
  int32_t B = (offsets.size(0) - 1) / T;
  if (B == 0 || T == 0) {
    return;
  }
  auto bounds_check_mode = static_cast<BoundsCheckMode>(bounds_check_mode_);
  if (bounds_check_mode == BoundsCheckMode::WARNING) {
    warning.zero_();
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
