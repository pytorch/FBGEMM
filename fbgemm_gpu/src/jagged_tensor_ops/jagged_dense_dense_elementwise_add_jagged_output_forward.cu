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

#define INVOKE_KERNEL_WITH_DIM(NUM_JAGGED_DIM)                                 \
  {                                                                            \
    dim3 threads, blocks;                                                      \
    StackArray<int64_t> jagged_dims_tensor;                                    \
    std::tie(threads, blocks, jagged_dims_tensor) =                            \
        check_shape_and_partition_(x_values, x_offsets, y_0);                  \
    blocks.x = div_round_up(x_values.size(0), threads.y);                      \
    std::vector<Tensor> x_offsets_contig;                                      \
    x_offsets_contig.resize(num_jagged_dim);                                   \
    StackArray<index_t*> x_offset_ptrs;                                        \
    x_offset_ptrs.ndim = num_jagged_dim;                                       \
    StackArray<int64_t> x_offset_sizes;                                        \
    x_offset_sizes.ndim = num_jagged_dim;                                      \
    for (int d = 0; d < num_jagged_dim; ++d) {                                 \
      x_offsets_contig[d] = x_offsets[d].contiguous();                         \
      x_offset_ptrs.vals[d] =                                                  \
          x_offsets_contig[d].template data_ptr<index_t>();                    \
      x_offset_sizes.vals[d] = x_offsets[d].numel();                           \
    }                                                                          \
    jagged_dense_dense_elementwise_jagged_output_kernel_<                      \
        NUM_JAGGED_DIM,                                                        \
        index_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(    \
        x_values.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),      \
        x_offset_ptrs,                                                         \
        x_offset_sizes,                                                        \
        y_0_reshaped.packed_accessor32<scalar_t, 3, at::RestrictPtrTraits>(),  \
        y_1_reshaped.packed_accessor32<scalar_t, 3, at::RestrictPtrTraits>(),  \
        output_values.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(), \
        jagged_dims_tensor,                                                    \
        f);                                                                    \
  }

template <typename scalar_t, typename F>
void jagged_dense_dense_elementwise_jagged_output_opt_(
    const Tensor& x_values,
    const std::vector<Tensor>& x_offsets,
    const Tensor& y_0,
    const Tensor& y_1,
    const Tensor& output_values,
    F f) {
  TENSOR_ON_CUDA_GPU(x_values);
  for (auto& x_offset : x_offsets) {
    TENSOR_ON_CUDA_GPU(x_offset);
  }

  const int num_jagged_dim = y_0.dim() - 2;
  TORCH_CHECK_EQ(x_offsets.size(), static_cast<size_t>(num_jagged_dim));

  if (y_0.numel() == 0 || x_values.numel() == 0) {
    return;
  }

  // Canonicalize y to 3D, collapsing jagged dimensions.
  const Tensor y_0_reshaped = y_0.view({y_0.size(0), -1, y_0.size(-1)});
  const Tensor y_1_reshaped = y_1.view({y_1.size(0), -1, y_1.size(-1)});

  if (jagged_dense_dense_elementwise_jagged_output_matches_opt(
          num_jagged_dim,
          x_values,
          x_offsets,
          y_0_reshaped,
          y_1_reshaped,
          output_values)) {
    AT_DISPATCH_INDEX_TYPES(
        x_offsets[0].scalar_type(),
        "jagged_dense_dense_indices_fast_path",
        [=] {
          auto nnz = output_values.size(0);
          auto B = y_0_reshaped.size(0);
          auto E = y_0_reshaped.size(2);
          Tensor t_rows_after_bs = at::empty(
              {nnz},
              at::TensorOptions().dtype(at::kInt).device(
                  at::kCUDA, at::cuda::current_device()));
          Tensor t_cols_after_bs = at::empty(
              {nnz},
              at::TensorOptions().dtype(at::kInt).device(
                  at::kCUDA, at::cuda::current_device()));

          // Binary search
          size_t dynamic_smem_size = (B + 1) * sizeof(index_t);
          auto cur_max_shared_bytes =
              at::cuda::getCurrentDeviceProperties()->sharedMemPerBlock;
          if (dynamic_smem_size > cur_max_shared_bytes) {
            int max_shared_bytes;
#ifndef __HIP_PLATFORM_HCC__
            C10_CUDA_CHECK(cudaDeviceGetAttribute(
                &max_shared_bytes,
                cudaDevAttrMaxSharedMemoryPerBlockOptin,
                y_0_reshaped.get_device()));
#else
            // MI100 has 64 KB local memory (shared memory) per workgroup
            max_shared_bytes = 64 << 10;
#endif
            int shared_kb = max_shared_bytes >> 10;
#ifndef __HIP_PLATFORM_HCC__
            // Use 2/3 of the available GPU shared mem; leave rooms for L1$.
            int used_shared_kb = round_down(shared_kb * 2 / 3, 16);
            TORCH_CHECK_GT(used_shared_kb, 0);
#else
            // MI100 has independent shared mem and L1
            int used_shared_kb = shared_kb;
#endif
            int used_shared_bytes = used_shared_kb << 10;
#ifndef __HIP_PLATFORM_HCC__
            C10_CUDA_CHECK(cudaFuncSetAttribute(
                jagged_dense_dense_elementwise_jagged_output_opt_search_kernel_<
                    index_t>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                used_shared_bytes)); // V100: 64 KB; A100: 96 KB.
#endif
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            TORCH_CHECK_LE(dynamic_smem_size, used_shared_bytes);
          }
          dim3 threads_bs = dim3(1024, 1, 1);
          dim3 blocks_bs = dim3(div_round_up(nnz, threads_bs.x), 1, 1);
          jagged_dense_dense_elementwise_jagged_output_opt_search_kernel_<
              index_t>
              <<<blocks_bs,
                 threads_bs,
                 dynamic_smem_size,
                 at::cuda::getCurrentCUDAStream()>>>(
                  x_offsets[0]
                      .packed_accessor32<index_t, 1, at::RestrictPtrTraits>(),
                  t_rows_after_bs
                      .packed_accessor32<int, 1, at::RestrictPtrTraits>(),
                  t_cols_after_bs
                      .packed_accessor32<int, 1, at::RestrictPtrTraits>(),
                  nnz,
                  B);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
          // Gather kernel
          dim3 threads = dim3(16, 16, 1);
          dim3 blocks = dim3(1, div_round_up(nnz, threads.y), 1);
          if (blocks.y > 65535) {
            blocks.y = 65535;
          }
          jagged_dense_dense_elementwise_jagged_output_opt_gather_kernel_<
              index_t>
              <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                  output_values
                      .packed_accessor32<c10::Half, 2, at::RestrictPtrTraits>(),
                  x_values
                      .packed_accessor32<c10::Half, 2, at::RestrictPtrTraits>(),
                  y_0_reshaped
                      .packed_accessor32<c10::Half, 3, at::RestrictPtrTraits>(),
                  y_1_reshaped
                      .packed_accessor32<c10::Half, 3, at::RestrictPtrTraits>(),
                  t_rows_after_bs
                      .packed_accessor32<int, 1, at::RestrictPtrTraits>(),
                  t_cols_after_bs
                      .packed_accessor32<int, 1, at::RestrictPtrTraits>(),
                  nnz,
                  E,
                  [f] __device__(__half x, __half y0, __half y1) -> __half {
                    return f(x, y0, y1);
                  });
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        }); // AT_DISPATCH
  } else {
    JAGGED_TENSOR_DISPATCH_DIMS();
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

template <typename scalar_t, typename F>
void jagged_dense_dense_elementwise_jagged_output_(
    const Tensor& x_values,
    const std::vector<Tensor>& x_offsets,
    const Tensor& y_0,
    const Tensor& y_1,
    const Tensor& output_values,
    F f) {
  TENSOR_ON_CUDA_GPU(x_values);
  for (auto& x_offset : x_offsets) {
    TENSOR_ON_CUDA_GPU(x_offset);
  }

  const int num_jagged_dim = y_0.dim() - 2;
  TORCH_CHECK(
      x_offsets.size() == static_cast<size_t>(num_jagged_dim),
      "x_offsets.size(), ",
      x_offsets.size(),
      " != num_jagged_dim, ",
      num_jagged_dim);

  if (y_0.numel() == 0 || x_values.numel() == 0) {
    return;
  }

  // Canonicalize y to 3D, collapsing jagged dimensions.
  const Tensor y_0_reshaped = y_0.view({y_0.size(0), -1, y_0.size(-1)});
  const Tensor y_1_reshaped = y_1.view({y_1.size(0), -1, y_1.size(-1)});

  JAGGED_TENSOR_DISPATCH_DIMS();
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

Tensor jagged_dense_dense_elementwise_add_jagged_output_forward(
    const Tensor& x_values,
    const std::vector<Tensor>& offsets,
    const Tensor& dense_0,
    const Tensor& dense_1) {
  TORCH_CHECK_EQ(dense_0.sizes(), dense_1.sizes());
  auto output = at::empty_like(x_values);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(dense_0.get_device());

  if (x_values.scalar_type() == at::ScalarType::BFloat16 &&
      dense_0.scalar_type() == at::ScalarType::BFloat16 &&
      dense_1.scalar_type() == at::ScalarType::Float) {
    AT_DISPATCH_SWITCH(
        x_values.scalar_type(),
        "jagged_dense_dense_elementwise_jagged_output_forward",
        AT_DISPATCH_CASE_FLOATING_TYPES_AND(
            at::ScalarType::BFloat16,
            [&] {
              jagged_dense_dense_elementwise_jagged_output_<scalar_t>(
                  x_values,
                  offsets,
                  dense_0,
                  dense_1.to(at::ScalarType::BFloat16),
                  output,
                  [] __device__(scalar_t x, scalar_t y_0, scalar_t y_1)
                      -> scalar_t { return x + y_0 + y_1; });
            } // lambda
            ) // AT_DISPATCH_CASE_FLOATING_TYPES_AND
    ); // SWITCH
  } else {
    AT_DISPATCH_SWITCH(
        x_values.scalar_type(),
        "jagged_dense_dense_elementwise_jagged_output_forward",
        AT_DISPATCH_CASE(
            at::ScalarType::Half,
            [&] {
              jagged_dense_dense_elementwise_jagged_output_opt_<scalar_t>(
                  x_values,
                  offsets,
                  dense_0,
                  dense_1,
                  output,
                  [] __device__(scalar_t x, scalar_t y_0, scalar_t y_1)
                      -> scalar_t { return x + y_0 + y_1; });
            } // lambda
            ) // CASE
        AT_DISPATCH_CASE_FLOATING_TYPES_AND(
            at::ScalarType::BFloat16,
            [&] {
              jagged_dense_dense_elementwise_jagged_output_<scalar_t>(
                  x_values,
                  offsets,
                  dense_0,
                  dense_1,
                  output,
                  [] __device__(scalar_t x, scalar_t y_0, scalar_t y_1)
                      -> scalar_t { return x + y_0 + y_1; });
            } // lambda
            ) // CASE_FLOATING_TYPES_AND
    ); // SWITCH
  }

  return output;
}

} // namespace fbgemm_gpu

FBGEMM_OP_DISPATCH(
    CUDA,
    "jagged_dense_dense_elementwise_add_jagged_output_forward",
    fbgemm_gpu::jagged_dense_dense_elementwise_add_jagged_output_forward);
