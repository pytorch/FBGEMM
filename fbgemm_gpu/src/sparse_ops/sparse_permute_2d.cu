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

// Kernel for permuting the indices and weights. Used for permutation of sparse
// data
template <
    bool has_weight,
    typename offsets_t,
    typename indices_t,
    typename weights_t>
__global__ __launch_bounds__(kMaxThreads) void permute_2D_data_kernel(
    int32_t len,
    int32_t T,
    int32_t B,
    const indices_t* __restrict__ indices,
    const weights_t* __restrict__ weights,
    const int32_t* __restrict__ permute,
    const offsets_t* __restrict__ input_offsets,
    const offsets_t* __restrict__ output_offsets,
    indices_t* __restrict__ permuted_indices,
    weights_t* __restrict__ permuted_weights) {
  int32_t b_t_start = blockIdx.x * blockDim.y + threadIdx.y;
  const int stride = gridDim.x * blockDim.y;
  for (int b_t = b_t_start; b_t < B * T; b_t += stride) {
    int32_t b = b_t % B;
    int32_t t = b_t / B;
    offsets_t output_start = output_offsets[b_t];
    offsets_t segment_length;
    if (b_t == B * T - 1) {
      segment_length = len - output_offsets[b_t];
    } else {
      segment_length = output_offsets[b_t + 1] - output_offsets[b_t];
    }
    offsets_t input_start = input_offsets[permute[t] * B + b];
    for (int32_t i = threadIdx.x; i < segment_length; i += blockDim.x) {
      permuted_indices[output_start + i] = indices[input_start + i];
      if (has_weight) {
        permuted_weights[output_start + i] = weights[input_start + i];
      }
    }
  }
}

// Kernel for permuting the lengths. Used for permutation of sparse features.
template <typename index_t>
__global__ __launch_bounds__(kMaxThreads) void permute_2D_lengths_kernel(
    int32_t T,
    int32_t B,
    const index_t* __restrict__ lengths,
    const int32_t* __restrict__ permute,
    index_t* __restrict__ permuted_lengths) {
  CUDA_KERNEL_LOOP(b_t, B * T) {
    int32_t b = b_t % B;
    int32_t t = b_t / B;
    permuted_lengths[b_t] = lengths[permute[t] * B + b];
  }
}

DLL_PUBLIC std::tuple<Tensor, Tensor, c10::optional<Tensor>>
permute_2D_sparse_data_cuda(
    const Tensor& permute,
    const Tensor& lengths,
    const Tensor& indices,
    const c10::optional<Tensor>& weights,
    const c10::optional<int64_t>& permuted_lengths_sum) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(permute, lengths, indices, weights);
  TORCH_CHECK(lengths.dim() == 2);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(indices.get_device());

  const auto permute_contig = permute.contiguous();
  const auto lengths_contig = lengths.contiguous();
  const auto indices_contig = indices.contiguous();
  // the data to permute over can be less or more with or without
  // repetitions
  const auto T = permute.numel();
  const auto B = lengths.size(1);

  Tensor permuted_lengths;
  Tensor permuted_indices;
  Tensor permuted_weights;

  permuted_lengths = at::empty({T, B}, lengths.options());

  constexpr int32_t threads_1 = 256;
  const auto blocks_1 = cuda_calc_xblock_count(B * T, threads_1);
  AT_DISPATCH_INDEX_TYPES(
      lengths.scalar_type(), "permute_2D_lengths_kernel", [&] {
        permute_2D_lengths_kernel<index_t>
            <<<blocks_1, threads_1, 0, at::cuda::getCurrentCUDAStream()>>>(
                T,
                B,
                lengths_contig.data_ptr<index_t>(),
                permute.data_ptr<int32_t>(),
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
  const auto blocks_2 = cuda_calc_xblock_count(B * T, BT_blocks);
  permuted_indices = at::empty(permuted_indices_size, indices.options());

  AT_DISPATCH_INDEX_TYPES(
      input_offsets.scalar_type(), "permute_2D_data_kernel_1", [&] {
        using offsets_t = index_t;
        AT_DISPATCH_ALL_TYPES_AND(
            at::ScalarType::Half,
            indices.scalar_type(),
            "permute_2D_data_kernel_2",
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
                    "permute_2D_data_kernel_3",
                    [&] {
                      using weights_t = scalar_t;
                      permute_2D_data_kernel<
                          true,
                          offsets_t,
                          indices_t,
                          weights_t>
                          <<<blocks_2,
                             threads_2,
                             0,
                             at::cuda::getCurrentCUDAStream()>>>(
                              permuted_indices_size,
                              T,
                              B,
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
                permute_2D_data_kernel<
                    false,
                    offsets_t,
                    indices_t,
                    std::nullptr_t>
                    <<<blocks_2,
                       threads_2,
                       0,
                       at::cuda::getCurrentCUDAStream()>>>(
                        permuted_indices_size,
                        T,
                        B,
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

// Kernel for permuting the indices and weights. Used for permutation of
// sparse features
template <bool has_weight, typename index_t, typename scalar_t>
__global__ __launch_bounds__(kMaxThreads) void permute_indices_weights_kernel(
    int32_t T,
    int32_t B,
    const index_t* __restrict__ indices,
    const scalar_t* __restrict__ weights,
    const int32_t* __restrict__ permute,
    const index_t* __restrict__ input_offsets,
    const index_t* __restrict__ output_offsets,
    index_t* __restrict__ permuted_indices,
    scalar_t* __restrict__ permuted_weights) {
  int32_t b_t_start = blockIdx.x * blockDim.y + threadIdx.y;
  const int stride = gridDim.x * blockDim.y;
  for (int b_t = b_t_start; b_t < B * T; b_t += stride) {
    int32_t b = b_t % B;
    int32_t t = b_t / B;
    index_t output_start = output_offsets[b_t];
    index_t segment_length = output_offsets[b_t + 1] - output_offsets[b_t];
    index_t input_start = input_offsets[permute[t] * B + b];
    for (int32_t i = threadIdx.x; i < segment_length; i += blockDim.x) {
      permuted_indices[output_start + i] = indices[input_start + i];
      if (has_weight) {
        permuted_weights[output_start + i] = weights[input_start + i];
      }
    }
  }
}

DLL_PUBLIC std::tuple<Tensor, Tensor, c10::optional<Tensor>>
permute_sparse_features_cuda(
    const Tensor& permute,
    const Tensor& lengths,
    const Tensor& indices,
    const c10::optional<Tensor>& weights) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(permute, lengths, indices, weights);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(indices.get_device());

  // the following implementation requires lengths and indices has the same
  // dtype if usecase comes up that requires different dtype (e.g. int32 for
  // lengths and int64 for indices, this will give a better error msg for
  // debugging
  TENSORS_HAVE_SAME_TYPE(lengths, indices);

  TORCH_CHECK(
      lengths.dim() == 2,
      "The dimension of lengths tensor should be equal to 2 to correctly infer number of features and batch size.")

  const auto permute_contig = permute.contiguous();
  const auto lengths_contig = lengths.contiguous();
  const auto indices_contig = indices.contiguous();
  // the features to permute over can be less or more with or without
  // repetitions
  const auto num_output_features = permute.numel();
  const auto num_features = lengths.size(0);
  const auto B = lengths.size(1);

  Tensor permuted_lengths;
  Tensor permuted_indices;
  Tensor permuted_weights;

  permuted_lengths = at::empty({num_output_features, B}, lengths.options());

  constexpr int32_t threads_1 = 256;
  const auto blocks_1 =
      cuda_calc_xblock_count(B * num_output_features, threads_1);
  AT_DISPATCH_INDEX_TYPES(
      lengths.scalar_type(), "permute_2D_lengths_kernel", [&] {
        fbgemm_gpu::permute_2D_lengths_kernel<index_t>
            <<<blocks_1, threads_1, 0, at::cuda::getCurrentCUDAStream()>>>(
                num_output_features,
                B,
                lengths_contig.data_ptr<index_t>(),
                permute.data_ptr<int32_t>(),
                permuted_lengths.data_ptr<index_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });

  // convert lengths to offsets
  const auto input_offsets =
      fbgemm_gpu::asynchronous_exclusive_cumsum_gpu(lengths_contig);
  const auto output_offsets =
      fbgemm_gpu::asynchronous_complete_cumsum_gpu(permuted_lengths.flatten());
  int64_t permuted_lengths_sum = indices.numel();

  /* TODO: Remove the condition protecting the slow path because even when the
   * condition below is true permuted_lengths.sum() could still be needed. For
   * instance if there are three features with indices `[0, 1, 2]`, `permute`
   * can be `[0, 1, 1]` for which permuted lengths sum would be needed to
   * create permuted_{indices, weights} and `permuted_lengths_sum =
   * indices.numel() or weights.numdel() would be incorrect.
   */
  if (num_features != num_output_features) {
    permuted_lengths_sum = output_offsets[-1].item<int64_t>();
  }

  constexpr int32_t BT_blocks = 32;
  dim3 threads_2(32, BT_blocks);
  const auto blocks_2 =
      cuda_calc_xblock_count(B * num_output_features, BT_blocks);
  permuted_indices = at::empty(permuted_lengths_sum, indices.options());
  if (weights.has_value()) {
    const Tensor weights_value = weights.value();
    const auto weights_value_contig = weights_value.contiguous();
    permuted_weights = at::empty(permuted_lengths_sum, weights_value.options());
    AT_DISPATCH_INDEX_TYPES(
        input_offsets.scalar_type(), "permute_indices_weights_kernel_1", [&] {
          AT_DISPATCH_FLOATING_TYPES_AND2(
              at::ScalarType::Int,
              at::ScalarType::BFloat16,
              weights_value.scalar_type(),
              "permute_indices_weights_kernel_2",
              [&] {
                permute_indices_weights_kernel<true, index_t, scalar_t>
                    <<<blocks_2,
                       threads_2,
                       0,
                       at::cuda::getCurrentCUDAStream()>>>(
                        num_output_features,
                        B,
                        indices_contig.data_ptr<index_t>(),
                        weights_value_contig.data_ptr<scalar_t>(),
                        permute_contig.data_ptr<int32_t>(),
                        input_offsets.data_ptr<index_t>(),
                        output_offsets.data_ptr<index_t>(),
                        permuted_indices.data_ptr<index_t>(),
                        permuted_weights.data_ptr<scalar_t>());
                C10_CUDA_KERNEL_LAUNCH_CHECK();
              });
        });
  } else {
    AT_DISPATCH_INDEX_TYPES(
        indices.scalar_type(), "permute_indices_kernel", [&] {
          permute_indices_weights_kernel<false, index_t, std::nullptr_t>
              <<<blocks_2, threads_2, 0, at::cuda::getCurrentCUDAStream()>>>(
                  num_output_features,
                  B,
                  indices_contig.data_ptr<index_t>(),
                  nullptr,
                  permute_contig.data_ptr<int32_t>(),
                  input_offsets.data_ptr<index_t>(),
                  output_offsets.data_ptr<index_t>(),
                  permuted_indices.data_ptr<index_t>(),
                  nullptr);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
  }
  return {permuted_lengths, permuted_indices, permuted_weights};
}

} // namespace fbgemm_gpu

FBGEMM_OP_DISPATCH(
    CUDA,
    "permute_sparse_data",
    fbgemm_gpu::permute_2D_sparse_data_cuda);
FBGEMM_OP_DISPATCH(
    CUDA,
    "permute_2D_sparse_data",
    fbgemm_gpu::permute_2D_sparse_data_cuda);
FBGEMM_OP_DISPATCH(
    CUDA,
    "permute_sparse_features",
    fbgemm_gpu::permute_sparse_features_cuda);
