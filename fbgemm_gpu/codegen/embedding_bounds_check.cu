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

template <typename index_t, bool vbe>
__global__ __launch_bounds__(kMaxThreads) void bounds_check_indices_kernel(
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        rows_per_table,
    at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> indices,
    at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> offsets,
    const int32_t* const B_offsets, // Use a raw pointer to avoid creating a
                                    // dummy PackedTensorAccessor
    const int64_t bounds_check_mode_,
    at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> warning,
    FixedDivisor fd) {
  int32_t T = rows_per_table.size(0);
  int32_t b_t = blockIdx.x * blockDim.y + threadIdx.y;
  int32_t b;
  int32_t t;
  int32_t B = 0;
  int32_t total_B = offsets.size(0) - 1;

  if (!vbe && b_t >= total_B) {
    return;
  }

  fd.DivMod(b_t, &t, &b);

  if (vbe) {
    // Check if t is valid
    if (t >= T) {
      return;
    }
    const auto B_start = B_offsets[t];
    B = B_offsets[t + 1] - B_start;
    // Check if b is valid
    if (b >= B) {
      return;
    }
    // Update b_t value
    b_t = B_start + b;
  } else {
    B = total_B / T;
  }

  const auto bounds_check_mode =
      static_cast<BoundsCheckMode>(bounds_check_mode_);
  const auto num_rows = rows_per_table[t];
  auto indices_start = offsets[b_t];
  auto indices_end = offsets[b_t + 1];
  const index_t num_indices = indices.size(0);

  if (bounds_check_mode == BoundsCheckMode::FATAL) {
    CUDA_KERNEL_ASSERT(indices_start >= 0);
    CUDA_KERNEL_ASSERT(indices_start <= indices_end);
    CUDA_KERNEL_ASSERT(indices_end <= num_indices);
  } else if (bounds_check_mode == BoundsCheckMode::WARNING) {
    if (indices_start < 0 || indices_start > indices_end ||
        indices_end > num_indices) {
      if (gpuAtomicIncrement(&warning[0]) == 0) {
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
      CUDA_KERNEL_ASSERT(idx >= 0 && "Failed idx >= 0 in bounds_check_indices");
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

  if (bounds_check_mode == BoundsCheckMode::FATAL) {
    CUDA_KERNEL_ASSERT(num_indices == offsets[total_B]);
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

void bounds_check_indices_cuda(
    Tensor& rows_per_table,
    Tensor& indices,
    Tensor& offsets,
    int64_t bounds_check_mode_,
    Tensor& warning,
    const c10::optional<Tensor>& weights,
    const c10::optional<Tensor>& B_offsets,
    const int64_t max_B) {
  TENSOR_ON_CUDA_GPU(rows_per_table);
  TENSOR_ON_CUDA_GPU(indices);
  TENSOR_ON_CUDA_GPU(offsets);
  TENSOR_ON_CUDA_GPU(warning);
  TENSOR_EMPTY_OR_ON_CUDA_GPU(weights);
  TENSOR_EMPTY_OR_ON_CUDA_GPU(B_offsets);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(rows_per_table.get_device());

  const int32_t T = rows_per_table.size(0);
  const int32_t total_B = offsets.size(0) - 1;
  const int32_t B = (total_B) / T;
  if (total_B == 0 || T == 0) {
    return;
  }
  const auto bounds_check_mode =
      static_cast<BoundsCheckMode>(bounds_check_mode_);
  if (bounds_check_mode == BoundsCheckMode::WARNING) {
    warning.zero_();
  }
  const int64_t num_indices = indices.size(0);
  const auto vbe = B_offsets.has_value();

  if (vbe) {
    TORCH_CHECK(max_B >= 0);
  } else {
    TORCH_CHECK(
        offsets.size(0) == B * T + 1,
        "offsets size " + std::to_string(offsets.size(0)) +
            " is not equal to B (" + std::to_string(B) + ") * T (" +
            std::to_string(T) + ") + 1");
  }
  if (weights.has_value()) {
    TORCH_CHECK(
        weights.value().size(0) == num_indices,
        "weights size " + std::to_string(weights.value().size(0)) +
            " is not equal to indices size " + std::to_string(num_indices));
  }

  constexpr size_t kNumThreads = 256;
  const auto max_B_ = vbe ? max_B : B;

  AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "bounds_check_indices", [&] {
    const auto bounds_check_kernel =
        (vbe ? bounds_check_indices_kernel<index_t, true>
             : bounds_check_indices_kernel<index_t, false>);
    bounds_check_kernel<<<
        div_round_up(max_B_ * T, kNumThreads / fbgemm_gpu::kWarpSize),
        dim3(fbgemm_gpu::kWarpSize, kNumThreads / fbgemm_gpu::kWarpSize),
        0,
        at::cuda::getCurrentCUDAStream()>>>(
        rows_per_table.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
        indices.packed_accessor32<index_t, 1, at::RestrictPtrTraits>(),
        offsets.packed_accessor32<index_t, 1, at::RestrictPtrTraits>(),
        vbe ? B_offsets.value().data_ptr<int32_t>() : nullptr,
        bounds_check_mode_,
        warning.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
        FixedDivisor(max_B_));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
}
