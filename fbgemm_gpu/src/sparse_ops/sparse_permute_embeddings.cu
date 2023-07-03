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

// Kernel for permuting the indices and weights. Used for permutation of
// table-wise partitioned sequence embeddings

template <typename index_t, typename scalar_t>
__global__ void permute_embeddings_kernel(
    int32_t len,
    int32_t T,
    int32_t B,
    const scalar_t* __restrict__ embeddings,
    // bag level permute
    const int32_t* __restrict__ permute,
    const index_t* __restrict__ input_offsets,
    const index_t* __restrict__ output_offsets,
    scalar_t* __restrict__ permuted_embeddings) {
  int32_t b_t_start = blockIdx.x * blockDim.y + threadIdx.y;
  const int stride = gridDim.x * blockDim.y;
  for (int b_t = b_t_start; b_t < B * T; b_t += stride) {
    int32_t b = b_t % B;
    int32_t t = b_t / B;
    index_t output_start = output_offsets[b_t];
    index_t segment_length;
    if (b_t == B * T - 1) {
      segment_length = len - output_offsets[b_t];
    } else {
      segment_length = output_offsets[b_t + 1] - output_offsets[b_t];
    }
    index_t input_start = input_offsets[permute[t] * B + b];
    for (int32_t i = threadIdx.x; i < segment_length; i += blockDim.x) {
      permuted_embeddings[output_start + i] = embeddings[input_start + i];
    }
  }
}

DLL_PUBLIC std::tuple<Tensor, Tensor> permute_sequence_embeddings_cuda(
    const Tensor& permute,
    const Tensor& lengths,
    const Tensor& embeddings) {
  // wrapper for permute_2D_sparse_data_cuda, kept for BC
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(permute, lengths, embeddings);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(embeddings.get_device());

  TORCH_CHECK(
      lengths.dim() == 2,
      "The dimension of lengths tensor should be equal to 2"
      "to correctly infer number of features and batch size.")

  Tensor permuted_lengths;
  Tensor permuted_embeddings;
  c10::optional<Tensor> weights_dummy;
  c10::optional<int64_t> permuted_lengths_sum_dummy;

  const auto T = permute.numel();
  const auto B = lengths.size(1);

  permuted_lengths = at::empty({T, B}, lengths.options());

  // ignore the third element in the tuple
  std::tie(permuted_lengths, permuted_embeddings, std::ignore) =
      fbgemm_gpu::permute_2D_sparse_data_cuda(
          permute,
          lengths,
          embeddings,
          weights_dummy,
          permuted_lengths_sum_dummy);

  return {permuted_lengths, permuted_embeddings};
}

} // namespace fbgemm_gpu

FBGEMM_OP_DISPATCH(
    CUDA,
    "permute_sequence_embeddings",
    fbgemm_gpu::permute_sequence_embeddings_cuda);
