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

// Kernel for generate 1D data permute from dimension permute index.
// Used for permutation of sparse features.
template <typename index_t, typename offsets_t>
__global__
__launch_bounds__(kMaxThreads) void expand_into_jagged_permute_kernel(
    const offsets_t* __restrict__ input_offsets,
    const offsets_t* __restrict__ output_offsets,
    int32_t input_size,
    const index_t* __restrict__ permute,
    index_t* __restrict__ output_permute) {
  const auto t_start = blockIdx.x * blockDim.y + threadIdx.y;
  const auto stride = gridDim.x * blockDim.y;
  for (int t = t_start; t < input_size; t += stride) {
    const offsets_t output_start = output_offsets[t];
    const offsets_t segment_length = output_offsets[t + 1] - output_offsets[t];
    const offsets_t input_start = input_offsets[permute[t]];
    for (auto i = threadIdx.x; i < segment_length; i += blockDim.x) {
      output_permute[output_start + i] = input_start + i;
    }
  }
}

DLL_PUBLIC Tensor expand_into_jagged_permute_cuda(
    const Tensor& permute,
    const Tensor& input_offsets,
    const Tensor& output_offsets,
    int64_t output_size) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      permute, input_offsets, output_offsets);

  TORCH_CHECK(permute.numel() > 0);
  TORCH_CHECK(permute.numel() == input_offsets.numel() - 1);
  TORCH_CHECK(permute.numel() == output_offsets.numel() - 1);

  CUDA_DEVICE_GUARD(permute);

  const auto permute_size = permute.numel();

  Tensor output_permute = at::empty({output_size}, permute.options());

  // number of table per block
  constexpr int32_t T_blocks = kMaxThreads / kWarpSize;
  dim3 threads(kWarpSize, T_blocks);
  // HIP enforces a hard limit of 2^32 total threads per launch (unlike CUDA,
  // which silently wraps). expand_into_jagged_permute_kernel grid-strides
  // over t (line 27), so capping is correctness-preserving.
  // See: https://github.com/ROCm/hip/issues/2253
  const auto blocks = utils::cuda::cap_grid_dim_x(
      cuda_calc_xblock_count(permute_size, T_blocks),
      kMaxThreads,
      at::cuda::getCurrentCUDAStream());
  AT_DISPATCH_INDEX_TYPES(
      permute.scalar_type(), "expand_into_jagged_permute_kernel", [&] {
        using offsets_t = index_t;
        FBGEMM_LAUNCH_KERNEL(
            (expand_into_jagged_permute_kernel<index_t, offsets_t>),
            blocks,
            threads,
            0,
            at::cuda::getCurrentCUDAStream(),
            input_offsets.data_ptr<offsets_t>(),
            output_offsets.data_ptr<offsets_t>(),
            permute_size,
            permute.data_ptr<index_t>(),
            output_permute.data_ptr<index_t>());
      });

  return output_permute;
}

} // namespace fbgemm_gpu

FBGEMM_OP_DISPATCH(
    CUDA,
    "expand_into_jagged_permute",
    fbgemm_gpu::expand_into_jagged_permute_cuda);
