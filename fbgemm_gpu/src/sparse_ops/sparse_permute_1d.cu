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
    weights_t* __restrict__ permuted_weights) {
  int32_t b_t_start = blockIdx.x * blockDim.y + threadIdx.y;
  const int stride = gridDim.x * blockDim.y;
  for (int b_t = b_t_start; b_t < permuted_lengths_size; b_t += stride) {
    offsets_t output_start = output_offsets[b_t];
    offsets_t segment_length;
    if (b_t == permuted_lengths_size - 1) {
      segment_length = permuted_indices_size - output_offsets[b_t];
    } else {
      segment_length = output_offsets[b_t + 1] - output_offsets[b_t];
    }
    offsets_t input_start = input_offsets[permute[b_t]];
    for (int32_t i = threadIdx.x; i < segment_length; i += blockDim.x) {
      permuted_indices[output_start + i] = indices[input_start + i];
      if (has_weight) {
        permuted_weights[output_start + i] = weights[input_start + i];
      }
    }
  }
}

DLL_PUBLIC std::tuple<Tensor, Tensor, c10::optional<Tensor>>
permute_1D_sparse_data_cuda(
    const Tensor& permute,
    const Tensor& lengths,
    const Tensor& indices,
    const c10::optional<Tensor>& weights,
    const c10::optional<int64_t>& permuted_lengths_sum) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(permute, lengths, indices, weights);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(indices.get_device());

  const auto permute_contig = permute.contiguous();
  const auto lengths_contig = lengths.contiguous();
  const auto indices_contig = indices.contiguous();
  // the data to permute over can be less or more with or without
  // repetitions

  const auto lengths_size = lengths.numel();

  const auto permuted_lengths_size = permute.numel();
  Tensor permuted_lengths;
  Tensor permuted_indices;
  Tensor permuted_weights;
  permuted_lengths = at::empty({permuted_lengths_size}, lengths.options());

  constexpr int32_t threads_1 = kMaxThreads;
  const auto blocks_1 =
      cuda_calc_xblock_count(permuted_lengths_size, threads_1);
  AT_DISPATCH_INDEX_TYPES(
      lengths.scalar_type(), "permute_1D_lengths_kernel", [&] {
        permute_1D_lengths_kernel<index_t>
            <<<blocks_1, threads_1, 0, at::cuda::getCurrentCUDAStream()>>>(
                lengths_contig.data_ptr<index_t>(),
                permuted_lengths_size,
                permute_contig.data_ptr<int32_t>(),
                permuted_lengths.data_ptr<index_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
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

  constexpr int32_t BT_blocks = 32;
  dim3 threads_2(32, BT_blocks);
  const auto blocks_2 =
      cuda_calc_xblock_count(permuted_lengths_size, BT_blocks);
  permuted_indices = at::empty(permuted_indices_size, indices.options());

  AT_DISPATCH_INDEX_TYPES(
      input_offsets.scalar_type(), "permute_1D_data_kernel_1", [&] {
        using offsets_t = index_t;
        AT_DISPATCH_ALL_TYPES_AND(
            at::ScalarType::Half,
            indices.scalar_type(),
            "permute_1D_data_kernel_2",
            [&] {
              using indices_t = scalar_t;
              if (weights.has_value()) {
                const Tensor weights_value = weights.value();
                const auto weights_value_contig = weights_value.contiguous();
                permuted_weights =
                    at::empty(permuted_indices_size, weights_value.options());
                AT_DISPATCH_ALL_TYPES_AND(
                    at::ScalarType::Half,
                    weights_value.scalar_type(),
                    "permute_1D_data_kernel_3",
                    [&] {
                      using weights_t = scalar_t;
                      permute_1D_data_kernel<
                          true,
                          offsets_t,
                          indices_t,
                          weights_t>
                          <<<blocks_2,
                             threads_2,
                             0,
                             at::cuda::getCurrentCUDAStream()>>>(
                              permuted_indices_size,
                              permuted_lengths_size,
                              indices_contig.data_ptr<indices_t>(),
                              weights_value_contig.data_ptr<weights_t>(),
                              permute_contig.data_ptr<int32_t>(),
                              input_offsets.data_ptr<offsets_t>(),
                              output_offsets.data_ptr<offsets_t>(),
                              permuted_indices.data_ptr<indices_t>(),
                              permuted_weights.data_ptr<weights_t>());
                      C10_CUDA_KERNEL_LAUNCH_CHECK();
                    }); // for each weights_t
              } else {
                permute_1D_data_kernel<
                    false,
                    offsets_t,
                    indices_t,
                    std::nullptr_t>
                    <<<blocks_2,
                       threads_2,
                       0,
                       at::cuda::getCurrentCUDAStream()>>>(
                        permuted_indices_size,
                        permuted_lengths_size,
                        indices_contig.data_ptr<indices_t>(),
                        nullptr,
                        permute_contig.data_ptr<int32_t>(),
                        input_offsets.data_ptr<offsets_t>(),
                        output_offsets.data_ptr<offsets_t>(),
                        permuted_indices.data_ptr<indices_t>(),
                        nullptr);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
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
