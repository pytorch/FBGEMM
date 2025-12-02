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
template <typename index_t>
__global__ __launch_bounds__(kMaxThreads) void permute_1D_lengths_kernel(
    const index_t* __restrict__ lengths,
    int32_t permuted_lengths_size,
    const int32_t* __restrict__ permute,
    index_t* __restrict__ permuted_lengths) {
  CUDA_KERNEL_LOOP(i, permuted_lengths_size) {
    permuted_lengths[i] = lengths[permute[i]];
  }
}

// Kernel for permuting the indices and weights. Used for permutation of sparse
// data
template <
    bool has_weight,
    typename offsets_t,
    typename indices_t,
    typename weights_t>
__global__ __launch_bounds__(kMaxThreads) void permute_1D_data_kernel(
    int32_t permuted_indices_size,
    int32_t permuted_lengths_size,
    const indices_t* __restrict__ indices,
    const weights_t* __restrict__ weights,
    const int32_t* __restrict__ permute,
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
    typename weights_t>
__global__ __launch_bounds__(kMaxThreads) void permute_1D_data_kernel_vec(
    int32_t permuted_indices_size,
    int32_t permuted_lengths_size,
    const indices_t* __restrict__ indices,
    const weights_t* __restrict__ weights,
    const int32_t* __restrict__ permute,
    const offsets_t* __restrict__ input_offsets,
    const offsets_t* __restrict__ output_offsets,
    indices_t* __restrict__ permuted_indices,
    weights_t* __restrict__ permuted_weights) {
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
    weights_t* __restrict__ weights_dst_ptr =
        has_weight ? permuted_weights + output_start : nullptr;
    const weights_t* __restrict__ weights_src_ptr =
        has_weight ? weights + input_start : nullptr;

    // Check alignment once per segment
    const bool indices_vec4_aligned =
        (sizeof(indices_t) == 4 || sizeof(indices_t) == 8) &&
        (reinterpret_cast<uintptr_t>(indices_dst_ptr) &
         (alignof(indices_vec4_t) - 1)) == 0 &&
        (reinterpret_cast<uintptr_t>(indices_src_ptr) &
         (alignof(indices_vec4_t) - 1)) == 0;

    const bool weights_vec4_aligned = !has_weight ||
        ((reinterpret_cast<uintptr_t>(weights_dst_ptr) &
          (alignof(weights_vec4_t) - 1)) == 0 &&
         (reinterpret_cast<uintptr_t>(weights_src_ptr) &
          (alignof(weights_vec4_t) - 1)) == 0);

    if (indices_vec4_aligned && weights_vec4_aligned) {
      // Vectorized path - process both indices and weights together
      const int32_t vec4_count = segment_length / 4;
      const int32_t remainder = segment_length & 3; // segment_length % 4

      auto indices_dst = reinterpret_cast<indices_vec4_t*>(indices_dst_ptr);
      auto indices_src =
          reinterpret_cast<const indices_vec4_t*>(indices_src_ptr);

      if (has_weight) {
        auto weights_dst = reinterpret_cast<weights_vec4_t*>(weights_dst_ptr);
        auto weights_src =
            reinterpret_cast<const weights_vec4_t*>(weights_src_ptr);

// copy both indices and weights
#pragma unroll
        for (auto i = threadIdx.x; i < vec4_count; i += blockDim.x) {
          indices_dst[i] = indices_src[i];
          weights_dst[i] = weights_src[i];
        }
        // Handle remainder elements (0-3 elements)
        if (threadIdx.x < remainder) {
          const auto offset = vec4_count * 4 + threadIdx.x;
          indices_dst_ptr[offset] = indices_src_ptr[offset];
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
        if (has_weight) {
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

  Tensor permuted_lengths;
  Tensor permuted_indices;
  Tensor permuted_weights;
  permuted_lengths = at::empty({permuted_lengths_size}, lengths.options());

  constexpr int32_t threads_1 = kMaxThreads;
  const auto blocks_1 =
      cuda_calc_xblock_count(permuted_lengths_size, threads_1);
  AT_DISPATCH_INDEX_TYPES(
      lengths.scalar_type(), "permute_1D_lengths_kernel", [&] {
        FBGEMM_LAUNCH_KERNEL(
            (permute_1D_lengths_kernel<index_t>),
            blocks_1,
            threads_1,
            0,
            at::cuda::getCurrentCUDAStream(),
            lengths_contig.data_ptr<index_t>(),
            permuted_lengths_size,
            permute_contig.data_ptr<int32_t>(),
            permuted_lengths.data_ptr<index_t>());
      });

  // convert lengths to offsets
  const auto input_offsets = asynchronous_exclusive_cumsum_gpu(lengths_contig);
  const auto output_offsets =
      asynchronous_complete_cumsum_gpu(permuted_lengths.flatten());
  int64_t permuted_indices_size = 0;
  if (permuted_lengths_sum.has_value()) {
    permuted_indices_size = permuted_lengths_sum.value();
  } else {
    permuted_indices_size = output_offsets[-1].item<int64_t>();
  }

  constexpr int32_t BT_blocks = 16;
  dim3 threads_2(64, BT_blocks);
  const auto blocks_2 =
      cuda_calc_xblock_count(permuted_lengths_size, BT_blocks);
  permuted_indices = at::empty(permuted_indices_size, indices.options());

  AT_DISPATCH_INDEX_TYPES(
      input_offsets.scalar_type(), "permute_1D_data_kernel_vec_1", [&] {
        using offsets_t = index_t;
        FBGEMM_DISPATCH_ALL_TYPES(
            indices.scalar_type(), "permute_1D_data_kernel_vec_2", [&] {
              using indices_t = scalar_t;
              if (weights.has_value()) {
                const Tensor weights_value = weights.value();
                const auto weights_value_contig = weights_value.contiguous();
                int32_t weights_columns = 1;
                if (weights_value.dense_dim() > 1) {
                  weights_columns = weights_value.size(1);
                  permuted_weights = at::empty(
                      {permuted_indices_size, weights_columns},
                      weights_value.options());
                } else {
                  permuted_weights =
                      at::empty(permuted_indices_size, weights_value.options());
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
                              weights_t>),
                          blocks_2,
                          threads_2,
                          0,
                          at::cuda::getCurrentCUDAStream(),
                          permuted_indices_size,
                          permuted_lengths_size,
                          indices_contig.data_ptr<indices_t>(),
                          weights_value_contig.data_ptr<weights_t>(),
                          permute_contig.data_ptr<int32_t>(),
                          input_offsets.data_ptr<offsets_t>(),
                          output_offsets.data_ptr<offsets_t>(),
                          permuted_indices.data_ptr<indices_t>(),
                          permuted_weights.data_ptr<weights_t>());
                    }); // for each weights_t
              } else {
                FBGEMM_LAUNCH_KERNEL(
                    (permute_1D_data_kernel_vec<
                        false,
                        offsets_t,
                        indices_t,
                        std::nullptr_t>),
                    blocks_2,
                    threads_2,
                    0,
                    at::cuda::getCurrentCUDAStream(),
                    permuted_indices_size,
                    permuted_lengths_size,
                    indices_contig.data_ptr<indices_t>(),
                    nullptr,
                    permute_contig.data_ptr<int32_t>(),
                    input_offsets.data_ptr<offsets_t>(),
                    output_offsets.data_ptr<offsets_t>(),
                    permuted_indices.data_ptr<indices_t>(),
                    nullptr);
              }
            }); // for each indices_t
      }); // for each offsets_t

  return {permuted_lengths, permuted_indices, permuted_weights};
}

} // namespace fbgemm_gpu

FBGEMM_OP_DISPATCH(
    CUDA,
    "permute_1D_sparse_data",
    fbgemm_gpu::permute_1D_sparse_data_cuda);
