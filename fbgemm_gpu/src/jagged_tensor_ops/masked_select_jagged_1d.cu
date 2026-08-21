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

/**
 * Per-row count of surviving elements.
 *
 * The exclusive prefix count of set mask bits is already materialized in
 * `mask_prefix`, so a row's output length is the difference of the prefix at
 * its two boundaries - no per-row scan needed.
 */
template <typename index_t>
__global__
__launch_bounds__(kMaxThreads) void masked_select_jagged_1d_lengths_kernel(
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        input_offsets,
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        mask_prefix,
    pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        masked_lengths) {
  const index_t B = masked_lengths.size(0);
  for (index_t b = blockIdx.x * blockDim.x + threadIdx.x; b < B;
       b += gridDim.x * blockDim.x) {
    masked_lengths[b] =
        mask_prefix[input_offsets[b + 1]] - mask_prefix[input_offsets[b]];
  }
}

/**
 * Compact the surviving values.
 *
 * Rows tile the input contiguously, so the number of survivors before element
 * `i` within its own row plus the number in all preceding rows is exactly
 * `mask_prefix[i]`. The destination index is therefore the prefix itself, with
 * no per-row correction, which keeps this a single coalesced pass with one
 * thread per element and preserves input order within each row.
 */
template <typename index_t, typename scalar_t>
__global__
__launch_bounds__(kMaxThreads) void masked_select_jagged_1d_compact_kernel(
    const pta::PackedTensorAccessor32<scalar_t, 1, at::RestrictPtrTraits>
        values,
    const pta::PackedTensorAccessor32<bool, 1, at::RestrictPtrTraits> mask,
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        input_offsets,
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        mask_prefix,
    pta::PackedTensorAccessor32<scalar_t, 1, at::RestrictPtrTraits>
        masked_values) {
  // Elements past the last row are ignored, matching the CPU op, which walks
  // rows rather than the mask and so never reads a trailing mask overhang.
  const index_t total = input_offsets[input_offsets.size(0) - 1];
  for (index_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total;
       i += gridDim.x * blockDim.x) {
    if (mask[i]) {
      masked_values[mask_prefix[i]] = values[i];
    }
  }
}

std::tuple<Tensor, Tensor> masked_select_jagged_1d_cuda(
    const Tensor& values,
    const Tensor& lengths,
    const Tensor& mask,
    const std::optional<bool> check_length) {
  TENSOR_ON_CUDA_GPU(values);
  TENSOR_ON_CUDA_GPU(lengths);
  TENSOR_ON_CUDA_GPU(mask);
  TORCH_CHECK(values.dim() == 1);
  TORCH_CHECK(lengths.dim() == 1);
  TORCH_CHECK(mask.dim() == 1);

  // Kept optional to mirror the CPU op, which still tolerates callsites that
  // pass a mask shorter or longer than values.
  if (check_length.has_value() && check_length.value()) {
    TORCH_CHECK(
        mask.numel() == values.numel(),
        "mask and values should have the same numel, but got mask numel: ",
        mask.numel(),
        " values numel: ",
        values.numel());
  }

  at::cuda::OptionalCUDAGuard device_guard(device_of(values));

  const auto B = lengths.numel();
  Tensor masked_lengths = at::empty_like(lengths);
  if (B == 0) {
    return {at::empty({0}, values.options()), masked_lengths};
  }

  const auto values_contiguous = values.expect_contiguous();
  const auto lengths_contiguous = lengths.expect_contiguous();
  const auto mask_contiguous = mask.expect_contiguous();

  const Tensor input_offsets =
      asynchronous_complete_cumsum_gpu(*lengths_contiguous);
  // Counting in the lengths dtype keeps the prefix, the offsets and the output
  // lengths in one index type, so the kernels need a single template parameter.
  const Tensor mask_prefix = asynchronous_complete_cumsum_gpu(
      mask_contiguous->to(lengths.scalar_type()));

  Tensor masked_values;
  AT_DISPATCH_INDEX_TYPES(
      lengths.scalar_type(), "masked_select_jagged_1d_cuda_index", [&] {
        // The output length is data-dependent, so one host sync is
        // unavoidable. Reading the tail of the prefix costs the same as the
        // CPU op's mask.sum() and keeps the allocation size identical to it.
        //
        // Deliberately indexed at mask.numel(), not input_offsets[B]: the CPU
        // op sizes its output from std::count over the WHOLE mask while filling
        // only the row range, so a mask longer than sum(lengths) leaves an
        // unwritten tail there too. Matching that keeps the two ops' output
        // shapes identical, which is what the parity gate compares. Narrowing
        // this to the row range would be a behaviour change from the CPU
        // reference, not a fix.
        index_t num_outputs = 0;
        C10_CUDA_CHECK(cudaMemcpyAsync(
            &num_outputs,
            mask_prefix.data_ptr<index_t>() + mask.numel(),
            sizeof(index_t),
            cudaMemcpyDeviceToHost,
            at::cuda::getCurrentCUDAStream()));
        C10_CUDA_CHECK(cudaStreamSynchronize(at::cuda::getCurrentCUDAStream()));

        masked_values =
            at::empty({static_cast<int64_t>(num_outputs)}, values.options());

        FBGEMM_LAUNCH_KERNEL(
            (masked_select_jagged_1d_lengths_kernel<index_t>),
            utils::cuda::cap_grid_dim_x(
                div_round_up(B, kMaxThreads),
                kMaxThreads,
                at::cuda::getCurrentCUDAStream()),
            kMaxThreads,
            0,
            at::cuda::getCurrentCUDAStream(),
            PTA_B(input_offsets, index_t, 1, 32),
            PTA_B(mask_prefix, index_t, 1, 32),
            PTA_B(masked_lengths, index_t, 1, 32));

        if (num_outputs == 0) {
          return;
        }

        // The compact kernel iterates over rows -- input_offsets[B], i.e.
        // sum(lengths) -- not over the mask, and `check_length` is optional so
        // mask.numel() can be smaller than that. Sizing the grid on the mask
        // would then launch too few threads and silently drop the tail
        // elements, so size it on the iteration space the kernel actually
        // walks. Grid-stride means an oversized grid is harmless.
        const int64_t compact_bound =
            values.numel() > mask.numel() ? values.numel() : mask.numel();

        FBGEMM_DISPATCH_ALL_TYPES(
            values.scalar_type(), "masked_select_jagged_1d_cuda_value", [&] {
              FBGEMM_LAUNCH_KERNEL(
                  (masked_select_jagged_1d_compact_kernel<index_t, scalar_t>),
                  utils::cuda::cap_grid_dim_x(
                      div_round_up(compact_bound, kMaxThreads),
                      kMaxThreads,
                      at::cuda::getCurrentCUDAStream()),
                  kMaxThreads,
                  0,
                  at::cuda::getCurrentCUDAStream(),
                  PTA_B(*values_contiguous, scalar_t, 1, 32),
                  PTA_B(*mask_contiguous, bool, 1, 32),
                  PTA_B(input_offsets, index_t, 1, 32),
                  PTA_B(mask_prefix, index_t, 1, 32),
                  PTA_B(masked_values, scalar_t, 1, 32));
            });
      });

  return {masked_values, masked_lengths};
}

} // namespace fbgemm_gpu

FBGEMM_OP_DISPATCH(
    CUDA,
    "masked_select_jagged_1d",
    fbgemm_gpu::masked_select_jagged_1d_cuda);
