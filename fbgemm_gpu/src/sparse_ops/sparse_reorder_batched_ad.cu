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

template <typename Dtype>
__global__
__launch_bounds__(kMaxThreads) void reorder_batched_ad_lengths_kernel(
    // reorder lengths from (ragged) [B  x T x #num_ads_b)] to
    // [T][B][#num_ads_b], i.e. [T][sum(#num_ads_b)].
    const at::PackedTensorAccessor32<Dtype, 1, at::RestrictPtrTraits>
        cat_ad_lengths,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        batch_offsets,
    at::PackedTensorAccessor32<Dtype, 1, at::RestrictPtrTraits>
        reordered_cat_ad_lengths,
    const int32_t T,
    const bool broadcast_lengths) {
  const int32_t B = batch_offsets.size(0) - 1;

  const int32_t num_ads_in_batch = batch_offsets[B];
  // warp-per-segment.
  const int32_t b_t = blockIdx.x * blockDim.y + threadIdx.y;
  const int32_t b = b_t % B;
  const int32_t t = b_t / B;
  if (t >= T) {
    return;
  }

  const int32_t num_ads_b = batch_offsets[b + 1] - batch_offsets[b];
  const int32_t input_segment_start =
      broadcast_lengths ? T * b + t : T * batch_offsets[b] + t * num_ads_b;
  const int32_t output_segment_start = t * num_ads_in_batch + batch_offsets[b];

  for (int32_t i = threadIdx.x; i < num_ads_b; i += blockDim.x) {
    reordered_cat_ad_lengths[output_segment_start + i] = broadcast_lengths
        ? cat_ad_lengths[input_segment_start]
        : cat_ad_lengths[input_segment_start + i];
  }
}

DLL_PUBLIC Tensor reorder_batched_ad_lengths_gpu(
    const Tensor& cat_ad_lengths,
    const Tensor& batch_offsets,
    const int64_t num_ads_in_batch,
    const bool broadcast_lengths) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(cat_ad_lengths, batch_offsets);

  CUDA_DEVICE_GUARD(cat_ad_lengths);

  const int64_t B = batch_offsets.numel() - 1;
  const int64_t T = broadcast_lengths
      ? cat_ad_lengths.numel() / B
      : cat_ad_lengths.numel() / num_ads_in_batch;

  Tensor reordered_cat_ad_lengths = broadcast_lengths
      ? at::empty({T * num_ads_in_batch}, cat_ad_lengths.options())
      : at::empty_like(cat_ad_lengths);

  const dim3 threads(32, 32);
  const dim3 blocks((B * T + 32 - 1) / 32);

  AT_DISPATCH_ALL_TYPES(
      cat_ad_lengths.scalar_type(),
      "reorder_batched_ad_lengths_gpu_kernel",
      [&] {
        reorder_batched_ad_lengths_kernel<scalar_t>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                cat_ad_lengths
                    .packed_accessor32<scalar_t, 1, at::RestrictPtrTraits>(),
                batch_offsets
                    .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                reordered_cat_ad_lengths
                    .packed_accessor32<scalar_t, 1, at::RestrictPtrTraits>(),
                T,
                broadcast_lengths);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return reordered_cat_ad_lengths;
}

template <typename Dtype, typename index_t = int32_t>
__global__ __launch_bounds__(kMaxThreads) void narrow_broadcast_indices_kernel(
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        cat_ad_offsets,
    const at::PackedTensorAccessor32<Dtype, 1, at::RestrictPtrTraits>
        cat_ad_indices,
    at::PackedTensorAccessor32<Dtype, 1, at::RestrictPtrTraits>
        reordered_cat_ad_indices,
    const int num_ads_in_batch,
    const int reordered_cat_ad_batches) {
  const auto lane_id = threadIdx.x % kWarpSize;
  const auto warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / kWarpSize;
  const auto table_idx = warp_id / num_ads_in_batch;
  const auto ads_idx = warp_id % num_ads_in_batch;
  const auto start_offset = cat_ad_offsets[table_idx];
  const auto end_offset = cat_ad_offsets[table_idx + 1];
  const auto num_ads = end_offset - start_offset;
  if (warp_id < reordered_cat_ad_batches) {
    for (auto i = lane_id; i < num_ads; i += kWarpSize) {
      reordered_cat_ad_indices
          [start_offset * num_ads_in_batch + ads_idx * num_ads + i] =
              cat_ad_indices[start_offset + i];
    }
  }
}

template <typename Dtype, typename index_t = int32_t>
__global__
__launch_bounds__(kMaxThreads) void reorder_batched_ad_indices_kernel(
    // reorder indices from (ragged) [B  x T x #num_ads_b x length_{b, t, a})]
    // to [T][B][#num_ads_b][length_{b, t, a}], i.e. [sum(length_{b, t, a})],
    // laid out as [T][B][A][L] (if all lengths were equal).

    // if broadcast_indices is enabled, all the indices will be copies of the
    // first batch of the cat_ad_indices, this is useful for request-only
    // broadcast
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        cat_ad_offsets,
    const at::PackedTensorAccessor32<Dtype, 1, at::RestrictPtrTraits>
        cat_ad_indices,
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        reordered_cat_ad_offsets,
    at::PackedTensorAccessor32<Dtype, 1, at::RestrictPtrTraits>
        reordered_cat_ad_indices,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        batch_offsets,
    const int32_t T,
    const bool broadcast_indices) {
  const int32_t B = batch_offsets.size(0) - 1;
  const int32_t num_ads_in_batch = batch_offsets[B];
  // warp-per-segment.
  const int32_t b_t = blockIdx.x * blockDim.y + threadIdx.y;
  const int32_t b = b_t % B;
  const int32_t t = b_t / B;
  if (t >= T) {
    return;
  }

  const auto num_ads_b = batch_offsets[b + 1] - batch_offsets[b];
  const auto output_segment_offset_start =
      t * num_ads_in_batch + batch_offsets[b];
  const auto output_segment_start =
      reordered_cat_ad_offsets[output_segment_offset_start];
  const int32_t input_segment_offset_start =
      broadcast_indices ? T * b + t : T * batch_offsets[b] + t * num_ads_b;
  const int32_t input_segment_offset_end = broadcast_indices
      ? input_segment_offset_start + 1
      : input_segment_offset_start + num_ads_b;
  const auto input_segment_start = cat_ad_offsets[input_segment_offset_start];
  const auto input_segment_end = cat_ad_offsets[input_segment_offset_end];
  const auto num_elements = input_segment_end - input_segment_start;

  if (broadcast_indices) {
    for (int32_t i = threadIdx.x; i < num_ads_b * num_elements;
         i += blockDim.x) {
      reordered_cat_ad_indices[output_segment_start + i] =
          cat_ad_indices[input_segment_start + i % num_elements];
    }
  } else {
    // Idea: we want to copy the entire segment of size sum_a(length_{b, t, a})
    // from starting point (given by cat_ad_offsets[b, t])
    // to end point (given by reordered_cat_ad_indices[t][b])
    for (int32_t i = threadIdx.x; i < input_segment_end - input_segment_start;
         i += blockDim.x) {
      reordered_cat_ad_indices[output_segment_start + i] =
          cat_ad_indices[input_segment_start + i];
    }
  }
}

DLL_PUBLIC Tensor reorder_batched_ad_indices_gpu(
    const Tensor& cat_ad_offsets,
    const Tensor& cat_ad_indices,
    const Tensor& reordered_cat_ad_offsets,
    const Tensor& batch_offsets,
    const int64_t num_ads_in_batch,
    const bool broadcast_indices,
    const int64_t num_indices_after_broadcast) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      cat_ad_offsets, cat_ad_indices, reordered_cat_ad_offsets, batch_offsets);

  CUDA_DEVICE_GUARD(cat_ad_offsets);

  const int64_t B = batch_offsets.numel() - 1;
  const int64_t T = (reordered_cat_ad_offsets.numel() - 1) / num_ads_in_batch;
  Tensor reordered_cat_ad_indices;
  if (broadcast_indices) {
    TORCH_CHECK_GE(num_indices_after_broadcast, 0);
    reordered_cat_ad_indices =
        at::empty({num_indices_after_broadcast}, cat_ad_indices.options());
  } else {
    reordered_cat_ad_indices = at::empty_like(cat_ad_indices);
  }

  if (broadcast_indices && B == 1 && T <= 320) {
    // for B = 1 broadcast case
    TORCH_CHECK(num_ads_in_batch * T == reordered_cat_ad_offsets.numel() - 1);
    constexpr auto NUM_WARPS = 16;
    const dim3 threads(NUM_WARPS * kWarpSize); //  16 x 32
    const dim3 blocks(cuda_calc_xblock_count(
        reordered_cat_ad_offsets.numel() - 1,
        NUM_WARPS)); // one warp per sample
    AT_DISPATCH_ALL_TYPES(
        cat_ad_indices.scalar_type(), "narrow_broadcast_indices_kernel_1", [&] {
          AT_DISPATCH_INDEX_TYPES(
              cat_ad_offsets.scalar_type(),
              "narrow_broadcast_indices_kernel_2",
              [&] {
                narrow_broadcast_indices_kernel<scalar_t, index_t>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                        cat_ad_offsets.packed_accessor32<
                            index_t,
                            1,
                            at::RestrictPtrTraits>(),
                        cat_ad_indices.packed_accessor32<
                            scalar_t,
                            1,
                            at::RestrictPtrTraits>(),
                        reordered_cat_ad_indices.packed_accessor32<
                            scalar_t,
                            1,
                            at::RestrictPtrTraits>(),
                        num_ads_in_batch,
                        reordered_cat_ad_offsets.numel() - 1);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
              });
        });
    return reordered_cat_ad_indices;
  }

  const dim3 threads(32, 32);
  const dim3 blocks((B * T + 32 - 1) / 32);

  AT_DISPATCH_ALL_TYPES_AND(
      at::ScalarType::BFloat16,
      cat_ad_indices.scalar_type(),
      "reorder_batched_ad_indices_gpu_kernel_1",
      [&] {
        AT_DISPATCH_INDEX_TYPES(
            cat_ad_offsets.scalar_type(),
            "reorder_batched_ad_indices_gpu_kernel_2",
            [&] {
              reorder_batched_ad_indices_kernel<scalar_t, index_t><<<
                  blocks,
                  threads,
                  0,
                  at::cuda::getCurrentCUDAStream()>>>(
                  cat_ad_offsets
                      .packed_accessor32<index_t, 1, at::RestrictPtrTraits>(),
                  cat_ad_indices
                      .packed_accessor32<scalar_t, 1, at::RestrictPtrTraits>(),
                  reordered_cat_ad_offsets
                      .packed_accessor32<index_t, 1, at::RestrictPtrTraits>(),
                  reordered_cat_ad_indices
                      .packed_accessor32<scalar_t, 1, at::RestrictPtrTraits>(),
                  batch_offsets
                      .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                  T,
                  broadcast_indices);
              C10_CUDA_KERNEL_LAUNCH_CHECK();
            });
      });
  return reordered_cat_ad_indices;
}

template <typename Dtype, typename index_t = int32_t>
__global__
__launch_bounds__(kMaxThreads) void reorder_batched_sequence_embeddings_kernel(
    // reorder embeddings from (ragged) [B x T x #num_ads_B_{i} x length_{B_{i},
    // t, a})x D] to [T][B][#num_ads_b][length_{b, t, a}][D], i.e.
    // [sum(length_{B_{i}, t, a}), D]
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        cat_sequence_embeddings_offsets,
    const at::PackedTensorAccessor32<Dtype, 2, at::RestrictPtrTraits>
        cat_sequence_embeddings,
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        reordered_cat_sequence_embeddings_offsets,
    at::PackedTensorAccessor32<Dtype, 2, at::RestrictPtrTraits>
        reordered_cat_sequence_embeddings,
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        batch_offsets,
    const int32_t T,
    const int32_t D) {
  const int32_t B = batch_offsets.size(0) - 1;
  const int32_t num_items_in_batch = batch_offsets[B];
  // warp-per-segment.
  const int32_t b_t = blockIdx.x * blockDim.y + threadIdx.y;
  const int32_t b = b_t % B;
  const int32_t t = b_t / B;
  if (t >= T) {
    return;
  }

  const auto num_ads_b = batch_offsets[b + 1] - batch_offsets[b];
  const auto output_segment_offset_start =
      t * num_items_in_batch + batch_offsets[b];
  const auto output_segment_start =
      reordered_cat_sequence_embeddings_offsets[output_segment_offset_start];
  const int32_t input_segment_offset_start =
      T * batch_offsets[b] + t * num_ads_b;
  const int32_t input_segment_offset_end =
      input_segment_offset_start + num_ads_b;
  const auto input_segment_start =
      cat_sequence_embeddings_offsets[input_segment_offset_start];
  const auto input_segment_end =
      cat_sequence_embeddings_offsets[input_segment_offset_end];
  const auto num_elements = input_segment_end - input_segment_start;

  for (size_t i = 0; i < input_segment_end - input_segment_start; i++) {
    const auto output_offset = output_segment_start + i;
    const auto input_offset = input_segment_start + i;
    for (int32_t d = threadIdx.x; d < D; d += blockDim.x) {
      reordered_cat_sequence_embeddings[output_offset][d] =
          cat_sequence_embeddings[input_offset][d];
    }
  }
}

DLL_PUBLIC Tensor reorder_batched_sequence_embeddings_gpu(
    const Tensor& cat_sequence_embeddings_offsets,
    const Tensor& cat_sequence_embeddings,
    const Tensor& reordered_cat_sequence_embeddings_offsets,
    const Tensor& batch_offsets,
    const int64_t num_items_in_batch) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      cat_sequence_embeddings_offsets,
      cat_sequence_embeddings,
      reordered_cat_sequence_embeddings_offsets,
      batch_offsets);
  const auto cat_sequence_embeddings_contig =
      cat_sequence_embeddings.expect_contiguous();

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(cat_sequence_embeddings_offsets.get_device());

  const int64_t B = batch_offsets.numel() - 1;
  const int64_t T = (reordered_cat_sequence_embeddings_offsets.numel() - 1) /
      num_items_in_batch;
  const int64_t D = cat_sequence_embeddings.size(1);
  Tensor reordered_cat_sequence_embeddings =
      at::empty_like(cat_sequence_embeddings);

  const dim3 threads(32, 32);
  const dim3 blocks((B * T + 32 - 1) / 32);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      cat_sequence_embeddings.scalar_type(),
      "reorder_batched_sequence_embeddings_gpu_kernel_1",
      [&] {
        AT_DISPATCH_INDEX_TYPES(
            cat_sequence_embeddings_offsets.scalar_type(),
            "reorder_batched_sequence_embeddings_gpu_kernel_2",
            [&] {
              reorder_batched_sequence_embeddings_kernel<scalar_t, index_t><<<
                  blocks,
                  threads,
                  0,
                  at::cuda::getCurrentCUDAStream()>>>(
                  cat_sequence_embeddings_offsets
                      .packed_accessor32<index_t, 1, at::RestrictPtrTraits>(),
                  cat_sequence_embeddings_contig
                      ->packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
                  reordered_cat_sequence_embeddings_offsets
                      .packed_accessor32<index_t, 1, at::RestrictPtrTraits>(),
                  reordered_cat_sequence_embeddings
                      .packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
                  batch_offsets
                      .packed_accessor32<index_t, 1, at::RestrictPtrTraits>(),
                  T,
                  D);
              C10_CUDA_KERNEL_LAUNCH_CHECK();
            });
      });
  return reordered_cat_sequence_embeddings;
}

} // namespace fbgemm_gpu
