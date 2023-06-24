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
    const int BLOCK_TILE_M, // tile height of C that each thread block
                            // calculates
    const int BLOCK_TILE_N, // tile width of C that each thread block
                            // calculates
    const int BLOCK_TILE_K, // tile width of A that each thread block calculates
    const int THREAD_TILE_M, // tile height of C that each thread
                             // calculates
    const int THREAD_TILE_N, // tile width of C that each thread calcualtes
    typename index_t,
    typename scalar_t>
__global__ __launch_bounds__(kMaxThreads) void jagged_dense_bmm_kernel(
    const pta::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits>
        x_values,
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        x_offsets,
    const pta::PackedTensorAccessor32<scalar_t, 3, at::RestrictPtrTraits> y,
    pta::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> output,
    const int max_L) {
  const int B = x_offsets.size(0) - 1;
  const int K = x_values.size(1);
  const int N = y.size(2);

  const auto block_row = blockIdx.y;
  const auto block_col = blockIdx.x;

  const int THREADS_X_PER_BLOCK = BLOCK_TILE_N / THREAD_TILE_N;
  const int THREADS_Y_PER_BLOCK = BLOCK_TILE_M / THREAD_TILE_M;
  const int THREADS_PER_BLOCK = THREADS_X_PER_BLOCK * THREADS_Y_PER_BLOCK;
  const auto thread_row = threadIdx.x / THREADS_X_PER_BLOCK;
  const auto thread_col = threadIdx.x % THREADS_X_PER_BLOCK;
  const auto NUM_K_BLOCKS = (K + BLOCK_TILE_K - 1) / BLOCK_TILE_K;

  __shared__ scalar_t As[BLOCK_TILE_M][BLOCK_TILE_K];
  __shared__ scalar_t Bs[BLOCK_TILE_K][BLOCK_TILE_N];

  // Once we remove ROCm<=5.3 support, we should replace uint32_t with auto.
  // See #1655
  for (uint32_t b = blockIdx.z; b < B; b += gridDim.z) {
    const index_t row_start = x_offsets[b];
    const index_t row_end = x_offsets[b + 1];
    const auto length = min(row_end - row_start, (index_t)max_L);

    // the indices that this current will load into shared mem
    const auto inner_row_a = threadIdx.x / BLOCK_TILE_K;
    const auto inner_col_a = threadIdx.x % BLOCK_TILE_K;
    // the number of rows of As that will be loaded per step by a thread block
    const auto A_TILE_ROW_STRIDE = THREADS_PER_BLOCK / BLOCK_TILE_K;

    const auto inner_row_b = threadIdx.x / BLOCK_TILE_N;
    const auto inner_col_b = threadIdx.x % BLOCK_TILE_N;
    const auto B_TILE_ROW_STRIDE = THREADS_PER_BLOCK / BLOCK_TILE_N;

    // registers for C
    scalar_t accum[THREAD_TILE_M][THREAD_TILE_N] = {0};

    // registers for As and Bs
    scalar_t fragment_a[THREAD_TILE_M] = {0};
    scalar_t fragment_b[THREAD_TILE_N] = {0};

    // loop for block tiles in K dimension
    for (auto block = 0; block < NUM_K_BLOCKS; block++) {
// load a block of x_values from global memory to shared memory
// apply tiling for threads in a block
#pragma unroll
      for (auto offset = 0; offset < BLOCK_TILE_M;
           offset += A_TILE_ROW_STRIDE) {
        auto x_row_offset = block_row * BLOCK_TILE_M + inner_row_a + offset;
        auto x_col_offset = block * BLOCK_TILE_K + inner_col_a;
        if ((x_row_offset < length) && (x_col_offset < K)) {
          As[inner_row_a + offset][inner_col_a] =
              x_values[row_start + x_row_offset][x_col_offset];
        } else {
          As[inner_row_a + offset][inner_col_a] = 0;
        }
      }

// load a block of y from global memory to shared memory
// apply tiling for threads in a block
#pragma unroll
      for (auto offset = 0; offset < BLOCK_TILE_K;
           offset += B_TILE_ROW_STRIDE) {
        auto y_row_offset = block * BLOCK_TILE_K + inner_row_b + offset;
        auto y_col_offset = block_col * BLOCK_TILE_N + inner_col_b;
        if ((y_row_offset < K) && (y_col_offset < N)) {
          Bs[inner_row_b + offset][inner_col_b] =
              y[b][y_row_offset][y_col_offset];
        } else {
          Bs[inner_row_b + offset][inner_col_b] = 0;
        }
      }

      __syncthreads();

// calculate the results per thread
#pragma unroll
      for (auto k = 0; k < BLOCK_TILE_K; k++) {
        // load values from shared memory to registers for x_values
        for (auto row = 0; row < THREAD_TILE_M; row++) {
          fragment_a[row] = As[thread_row * THREAD_TILE_M + row][k];
        }

// load values from shared memory to registers for y
#pragma unroll
        for (auto col = 0; col < THREAD_TILE_N; col++) {
          fragment_b[col] = Bs[k][thread_col * THREAD_TILE_N + col];
        }

// each thread calcualtes THREAD_TILE_M * THREAD_TILE_N elements
#pragma unroll
        for (auto row = 0; row < THREAD_TILE_M; row++) {
#pragma unroll
          for (auto col = 0; col < THREAD_TILE_N; col++) {
            accum[row][col] += fragment_a[row] * fragment_b[col];
          }
        }
      }

      __syncthreads();
    }

// write the result to the output
#pragma unroll
    for (auto row = 0; row < THREAD_TILE_M; row++) {
#pragma unroll
      for (auto col = 0; col < THREAD_TILE_N; col++) {
        auto out_row_offset =
            block_row * BLOCK_TILE_M + thread_row * THREAD_TILE_M + row;
        auto out_col_offset =
            block_col * BLOCK_TILE_N + thread_col * THREAD_TILE_N + col;
        if ((out_row_offset < length) && (out_col_offset < N)) {
          output[row_start + out_row_offset][out_col_offset] = accum[row][col];
        }
      }
    }
  }
}

Tensor jagged_dense_bmm_forward_cuda(
    const Tensor& x_values,
    const Tensor& x_offsets,
    const Tensor& y,
    const int64_t max_L) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(x_values, x_offsets, y);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(x_values.get_device());

  const int B = x_offsets.numel() - 1;
  const int M = x_values.size(-1);
  const int N = y.size(-1);
  const int total_L = x_values.size(0);
  auto output = at::zeros({total_L, N}, x_values.options());
  if (B > 0 && M > 0 && N > 0) {
    // The shared memory size is (BLOCK_TILE_M + BLOCK_TILE_N) * BLOCK_TILE_K
    // BLOCK_TILE_M needs to be multiple of THREAD_TILE_M, and
    // BLOCK_TILE_N needs to be multiple of THREAD_TILE_N
    // The setting of these parameters needs to balance the hardware's shared
    // memory size limit and occupancy
    // TODO: autotune these parameters based on max_L and input and output
    // tensor sizes

    constexpr int BLOCK_TILE_K = 8;
    constexpr int THREAD_TILE_M = 2;
    constexpr int THREAD_TILE_N = 2;

    if (M > N) {
      constexpr int BLOCK_TILE_M = 32;
      constexpr int BLOCK_TILE_N = 8;

      const dim3 block(
          (BLOCK_TILE_M * BLOCK_TILE_N) / (THREAD_TILE_M * THREAD_TILE_N));
      const auto grid_dim_x = div_round_up(N, BLOCK_TILE_N);
      const auto grid_dim_y = div_round_up(max_L, BLOCK_TILE_M);
      TORCH_CHECK(
          grid_dim_y <= kMaxBlockYDim,
          "max_L cannot be larger than",
          kMaxBlockYDim * BLOCK_TILE_M + 1 - BLOCK_TILE_M);
      const auto grid_dim_z = std::min(B, kMaxBlockZDim);
      const dim3 grid(grid_dim_x, grid_dim_y, grid_dim_z);

      AT_DISPATCH_INDEX_TYPES(
          x_offsets.scalar_type(), "jagged_dense_bmm_kernel_1", [&] {
            AT_DISPATCH_FLOATING_TYPES_AND2(
                at::ScalarType::Half,
                at::ScalarType::BFloat16,
                x_values.scalar_type(),
                "jagged_dense_bmm_kernel_2",
                [&] {

#ifdef FBGEMM_GPU_MEMCHECK
                  const auto func_name1 = "jagged_dense_bmm_kernel";
#endif

                  jagged_dense_bmm_kernel<
                      BLOCK_TILE_M,
                      BLOCK_TILE_N,
                      BLOCK_TILE_K,
                      THREAD_TILE_M,
                      THREAD_TILE_N,
                      index_t,
                      scalar_t>
                      <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                          MAKE_PTA_WITH_NAME(
                              func_name1, x_values, scalar_t, 2, 32),
                          MAKE_PTA_WITH_NAME(
                              func_name1, x_offsets, index_t, 1, 32),
                          MAKE_PTA_WITH_NAME(func_name1, y, scalar_t, 3, 32),
                          MAKE_PTA_WITH_NAME(
                              func_name1, output, scalar_t, 2, 32),
                          (int)max_L);
                  C10_CUDA_KERNEL_LAUNCH_CHECK();
                });
          });
    } else {
      constexpr int BLOCK_TILE_M = 8;
      constexpr int BLOCK_TILE_N = 32;
      const dim3 block(
          (BLOCK_TILE_M * BLOCK_TILE_N) / (THREAD_TILE_M * THREAD_TILE_N));
      const auto grid_dim_x = div_round_up(N, BLOCK_TILE_N);
      const auto grid_dim_y = div_round_up(max_L, BLOCK_TILE_M);
      TORCH_CHECK(
          grid_dim_y <= kMaxBlockYDim,
          "max_L cannot be larger than",
          kMaxBlockYDim * BLOCK_TILE_M + 1 - BLOCK_TILE_M);
      const auto grid_dim_z = std::min(B, kMaxBlockZDim);
      const dim3 grid(grid_dim_x, grid_dim_y, grid_dim_z);

      AT_DISPATCH_INDEX_TYPES(
          x_offsets.scalar_type(), "jagged_dense_bmm_kernel_1", [&] {
            AT_DISPATCH_FLOATING_TYPES_AND2(
                at::ScalarType::Half,
                at::ScalarType::BFloat16,
                x_values.scalar_type(),
                "jagged_dense_bmm_kernel_2",
                [&] {

#ifdef FBGEMM_GPU_MEMCHECK
                  const auto func_name2 = "jagged_dense_bmm_kernel";
#endif

                  jagged_dense_bmm_kernel<
                      BLOCK_TILE_M,
                      BLOCK_TILE_N,
                      BLOCK_TILE_K,
                      THREAD_TILE_M,
                      THREAD_TILE_N,
                      index_t,
                      scalar_t>
                      <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                          MAKE_PTA_WITH_NAME(
                              func_name2, x_values, scalar_t, 2, 32),
                          MAKE_PTA_WITH_NAME(
                              func_name2, x_offsets, index_t, 1, 32),
                          MAKE_PTA_WITH_NAME(func_name2, y, scalar_t, 3, 32),
                          MAKE_PTA_WITH_NAME(
                              func_name2, output, scalar_t, 2, 32),
                          (int)max_L);
                  C10_CUDA_KERNEL_LAUNCH_CHECK();
                });
          });
    }
  }

  return output;
}

} // namespace fbgemm_gpu

JAGGED_TENSOR_OPS_CUDA_DISPATCH(
    "jagged_dense_bmm_forward",
    fbgemm_gpu::jagged_dense_bmm_forward_cuda);
