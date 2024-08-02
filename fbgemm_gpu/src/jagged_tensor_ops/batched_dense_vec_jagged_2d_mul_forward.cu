/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// clang-format off
#include "common.cuh"

using Tensor = at::Tensor;

namespace fbgemm_gpu {

template <typename index_t, typename scalar_t>
__global__ __launch_bounds__(kMaxThreads) void dense_vec_jagged_2d_bmm(
    const pta::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> v,
    const pta::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> a_values,
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> a_offsets,
    pta::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> output) {
  const int B = a_offsets.size(0) - 1;
  const int H = v.size(0) / B;
  const int max_L = v.size(1);
  const int D = output.size(1);

  const int b_h_begin = blockIdx.x * blockDim.y + threadIdx.y;
  const int b_h_step = gridDim.x * blockDim.y;
  for (int b_h = b_h_begin; b_h < B * H; b_h += b_h_step) {
    const int b = b_h / H;
    const int h = b_h % H;

    const int row_start = a_offsets[b];
    const int row_end = a_offsets[b + 1];
    const int length = std::min(row_end - row_start, max_L);
    if (length == 0) {
      for (int d = threadIdx.x; d < D; d += blockDim.x) {
        output[b_h][d] = 0;
      }
    } else {
      // TODO: use shared memory
      for (int d = threadIdx.x; d < D; d += blockDim.x) {
        at::acc_type<scalar_t, true> acc =
            v[b_h][0] * a_values[row_start][h * D + d];
        for (int l = 1; l < length; ++l) {
          acc += v[b_h][l] * a_values[row_start + l][h * D + d];
        }
        output[b_h][d] = acc;
      }
    }
  }
}

Tensor batched_dense_vec_jagged_2d_mul_forward(
    const Tensor& v,
    const Tensor& a_values,
    const Tensor& a_offsets) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(v, a_values, a_offsets);
  CUDA_DEVICE_GUARD(v);

  const int B = a_offsets.numel() - 1;
  TORCH_CHECK(
      B == 0 || v.size(0) % B == 0,
      "B, ",
      B,
      " doesn't divide v.size(0), ",
      v.size(0));
  const int H = (B == 0) ? 1 : v.size(0) / B;
  const int D = a_values.size(-1) / H;
  auto output = at::empty({B * H, D}, v.options());

  if (B > 0 && D > 0) {
    const int block_dim_x =
        std::min(div_round_up(D, kWarpSize) * kWarpSize, kMaxThreads);
    const int block_dim_y = kMaxThreads / block_dim_x;

    AT_DISPATCH_INDEX_TYPES(
        a_offsets.scalar_type(), "dense_vec_jagged_2d_bmm_kernel_1", [&] {
          FBGEMM_DISPATCH_FLOATING_TYPES(
              a_values.scalar_type(),
              "dense_vec_jagged_2d_bmm_kernel_2",
              [&] {

#ifdef FBGEMM_GPU_MEMCHECK
                const auto func_name1 = "dense_vec_jagged_2d_bmm";
#endif
                dense_vec_jagged_2d_bmm<index_t, scalar_t>
                    <<<div_round_up(B * H, block_dim_y),
                       dim3(block_dim_x, block_dim_y),
                       0,
                       at::cuda::getCurrentCUDAStream()>>>(
                        MAKE_PTA_WITH_NAME(func_name1, v, scalar_t, 2, 32),
                        MAKE_PTA_WITH_NAME(func_name1, a_values, scalar_t, 2, 32),
                        MAKE_PTA_WITH_NAME(func_name1, a_offsets, index_t, 1, 32),
                        MAKE_PTA_WITH_NAME(func_name1, output, scalar_t, 2, 32));
                C10_CUDA_KERNEL_LAUNCH_CHECK();
              });
        });
  }

  return output;
}

} // namespace fbgemm_gpu

FBGEMM_OP_DISPATCH(
    CUDA,
    "batched_dense_vec_jagged_2d_mul_forward",
    fbgemm_gpu::batched_dense_vec_jagged_2d_mul_forward);
