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
__global__ __launch_bounds__(kMaxThreads) void outer_prod_jagged_2d_output(
    const pta::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> x,
    const pta::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> y,
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> offsets,
    pta::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> output_values) {
  const int B = offsets.size(0) - 1;
  const int H = x.size(0) / B;
  const int max_L = x.size(1);
  const int D = y.size(1);

  const int b_h_l_begin = blockIdx.x * blockDim.y + threadIdx.y;
  const int b_h_l_step = gridDim.x * blockDim.y;
  for (int b_h_l = b_h_l_begin; b_h_l < B * H * max_L; b_h_l += b_h_l_step) {
    const int b_h = b_h_l / max_L;
    const int b = b_h / H;
    const int h = b_h % H;
    const int l = b_h_l % max_L;

    const int row_start = offsets[b];
    const int row_end = offsets[b + 1];
    const int length = row_end - row_start;
    if (l < length) {
      for (int d = threadIdx.x; d < D; d += blockDim.x) {
        output_values[row_start + l][h * D + d] = x[b_h][l] * y[b_h][d];
      }
    }
  }
}

template <typename index_t, typename scalar_t>
__global__
__launch_bounds__(kMaxThreads) void dense_vec_jagged_2d_transposed_bmm(
    const pta::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> v,
    const pta::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> a_values,
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> a_offsets,
    pta::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> output) {
  const int B = a_offsets.size(0) - 1;
  const int H = v.size(0) / B;
  const int max_L = output.size(1);
  const int D = v.size(1);

  const int b_h_begin = blockIdx.x * blockDim.y + threadIdx.y;
  const int b_h_step = gridDim.x * blockDim.y;
  for (int b_h = b_h_begin; b_h < B * H; b_h += b_h_step) {
    const int b = b_h / H;
    const int h = b_h % H;

    const int row_start = a_offsets[b];
    const int row_end = a_offsets[b + 1];
    const int length = std::min(row_end - row_start, max_L);
    if (D == 0) {
      for (int l = threadIdx.x; l < max_L; ++l) {
        output[b_h][l] = 0;
      }
    } else {
      int l;
      for (l = threadIdx.x; l < length; l += blockDim.x) {
        at::acc_type<scalar_t, true> acc =
            v[b_h][0] * a_values[row_start + l][h * D];
        for (int d = 1; d < D; ++d) {
          acc += v[b_h][d] * a_values[row_start + l][h * D + d];
        }
        output[b_h][l] = acc;
      }
      for (; l < max_L; l += blockDim.x) {
        output[b_h][l] = 0;
      }
    }
  }
}

std::tuple<Tensor, Tensor> batched_dense_vec_jagged_2d_mul_backward(
    const Tensor& grad_output,
    const Tensor& v,
    const Tensor& a_values,
    const Tensor& a_offsets) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(grad_output, a_values, a_offsets, v);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(grad_output.get_device());

  const int B = a_offsets.numel() - 1;
  const int D = grad_output.size(-1);

  Tensor a_values_grad = at::zeros_like(a_values);
  Tensor v_grad = at::empty_like(v);

  if (B > 0 && D > 0) {
    TORCH_CHECK(
        v.size(0) % B == 0, "B, ", B, " doesn't divide v.size(0), ", v.size(0));
    const int H = v.size(0) / B;
    const int max_L = v.size(-1);

    AT_DISPATCH_INDEX_TYPES(
        a_offsets.scalar_type(),
        "dense_vec_jagged_2d_bmm_backward_kernel_1",
        [&] {
          AT_DISPATCH_FLOATING_TYPES_AND2(
              at::ScalarType::Half,
              at::ScalarType::BFloat16,
              grad_output.scalar_type(),
              "dense_vec_jagged_2d_bmm_backward_kernel_2",
              [&] {
                int block_dim_x = std::min(
                    div_round_up(max_L, kWarpSize) * kWarpSize, kMaxThreads);
                int block_dim_y = kMaxThreads / block_dim_x;

#ifdef FBGEMM_GPU_MEMCHECK
                const auto func_name1 = "dense_vec_jagged_2d_transposed_bmm";
#endif

                dense_vec_jagged_2d_transposed_bmm<index_t, scalar_t>
                    <<<div_round_up(B * H, block_dim_y),
                       dim3(block_dim_x, block_dim_y),
                       0,
                       at::cuda::getCurrentCUDAStream()>>>(
                        MAKE_PTA_WITH_NAME(func_name1, grad_output, scalar_t, 2, 32),
                        MAKE_PTA_WITH_NAME(func_name1, a_values, scalar_t, 2, 32),
                        MAKE_PTA_WITH_NAME(func_name1, a_offsets, index_t, 1, 32),
                        MAKE_PTA_WITH_NAME(func_name1, v_grad, scalar_t, 2, 32)
                        );
                C10_CUDA_KERNEL_LAUNCH_CHECK();

                block_dim_x = std::min(
                    div_round_up(D, kWarpSize) * kWarpSize, kMaxThreads);
                block_dim_y = kMaxThreads / block_dim_x;

#ifdef FBGEMM_GPU_MEMCHECK
                const auto func_name2 = "outer_prod_jagged_2d_output";
#endif

                outer_prod_jagged_2d_output<index_t, scalar_t>
                    <<<div_round_up(B * H * max_L, block_dim_y),
                       dim3(block_dim_x, block_dim_y),
                       0,
                       at::cuda::getCurrentCUDAStream()>>>(
                        MAKE_PTA_WITH_NAME(func_name2, v, scalar_t, 2, 32),
                        MAKE_PTA_WITH_NAME(func_name2, grad_output, scalar_t, 2, 32),
                        MAKE_PTA_WITH_NAME(func_name2, a_offsets, index_t, 1, 32),
                        MAKE_PTA_WITH_NAME(func_name2, a_values_grad, scalar_t, 2, 32)
                        );
                C10_CUDA_KERNEL_LAUNCH_CHECK();
              });
        });
  } else {
    v_grad.zero_();
  }

  return {v_grad, a_values_grad};
}

} // namespace fbgemm_gpu

FBGEMM_OP_DISPATCH(
    CUDA,
    "batched_dense_vec_jagged_2d_mul_backward",
    fbgemm_gpu::batched_dense_vec_jagged_2d_mul_backward);
