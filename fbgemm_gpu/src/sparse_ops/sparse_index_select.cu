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

template <
    typename index_t,
    typename scalar_t,
    int UNROLL_FACTOR,
    bool indices_sorted>
__global__ __launch_bounds__(kMaxThreads) void index_select_2d_kernel(
    const at::PackedTensorAccessor64<scalar_t, 2, at::RestrictPtrTraits> input,
    const at::PackedTensorAccessor64<index_t, 1, at::RestrictPtrTraits> indices,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits>
        orig_indices,
    at::PackedTensorAccessor64<scalar_t, 2> output,
    TORCH_DSA_KERNEL_ARGS) {
  const int N = indices.size(0);
  const int input_size = input.size(0);
  const int D = input.size(1);
  CUDA_KERNEL_ASSERT2(output.size(0) == N);

  for (int row = blockIdx.x; row < N; row += gridDim.x) {
    const index_t src_idx = indices[row];
    const int64_t dst_idx = indices_sorted ? orig_indices[row] : row;
    CUDA_KERNEL_ASSERT2(src_idx < input_size);
    int col;
    for (col = threadIdx.x * UNROLL_FACTOR;
         col < D / UNROLL_FACTOR * UNROLL_FACTOR;
         col += blockDim.x * UNROLL_FACTOR) {
#pragma unroll
      for (int i = 0; i < UNROLL_FACTOR; i++) {
        output[dst_idx][col + i] = LDG(&input[src_idx][col + i]);
      }
    }
    for (; col < D; ++col) {
      output[dst_idx][col] = LDG(&input[src_idx][col]);
    }
  }
}

DLL_PUBLIC Tensor index_select_cuda(
    const Tensor& input,
    const Tensor& indices,
    const Tensor& orig_indices,
    const bool indices_sorted) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(input.get_device());

  const int N = indices.size(0);
  auto output_shape = input.sizes().vec();
  output_shape[0] = N;

  if (input.numel() == 0 || N == 0) {
    return at::empty(output_shape, input.options());
  }

  Tensor input_reshaped = input.reshape({input.size(0), -1});
  const int D = input_reshaped.size(1);

  Tensor output = at::empty({N, D}, input_reshaped.options());

  const int UNROLL_FACTOR = 2;

#define LAUNCH_INDEX_SELECT(INDICES_SORTED)                                   \
  TORCH_DSA_KERNEL_LAUNCH(                                                    \
      (index_select_2d_kernel<                                                \
          index_t,                                                            \
          scalar_t,                                                           \
          UNROLL_FACTOR,                                                      \
          INDICES_SORTED>),                                                   \
      cuda_calc_xblock_count(N, 1),                                           \
      std::min(div_round_up(D, UNROLL_FACTOR), kMaxThreads),                  \
      0,                                                                      \
      at::cuda::getCurrentCUDAStream(),                                       \
      input_reshaped.packed_accessor64<scalar_t, 2, at::RestrictPtrTraits>(), \
      indices.packed_accessor64<index_t, 1, at::RestrictPtrTraits>(),         \
      INDICES_SORTED                                                          \
          ? orig_indices                                                      \
                .packed_accessor64<int64_t, 1, at::RestrictPtrTraits>()       \
          : dummy_packed_accessor64<int64_t, 1, at::RestrictPtrTraits>(),     \
      output.packed_accessor64<scalar_t, 2>());

  AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "index_add_2d_kernel_1", [&] {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input_reshaped.scalar_type(), "index_add_2d_kernel_2", [&] {
          if (indices_sorted) {
            LAUNCH_INDEX_SELECT(true)
          } else {
            LAUNCH_INDEX_SELECT(false)
          }
        });
  });

#undef LAUNCH_INDEX_SELECT

  return output.reshape(output_shape);
}

} // namespace fbgemm_gpu
