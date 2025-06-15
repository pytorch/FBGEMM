/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

#include "include/common_utils.h"
#include "include/fast_gemv.cuh"

namespace fbgemm_gpu {

using SizeType32 = std::size_t;

// The heuristics are derived by sweeping over 4 different
// problem sizes we care about and selected the best elapsed time/bw
// combination. See more in
// deeplearning/fbgemm/fbgemm_gpu/experimental/gen_ai/src/quantize/fast_gemv/sweep_utils.py
namespace {
dim3 get_best_block_dim(int m, int n, int k) {
  if (m == 1 && n == 1280 && k == 8192) {
    return dim3(256, 1);
  } else if (m == 1 && n == 8192 && k == 1024) {
    return dim3(128, 1);
  } else if (m == 1 && n == 7168 && k == 8192) {
    return dim3(256, 1);
  } else if (m == 1 && n == 8192 && k == 3584) {
    return dim3(128, 1);
  } else if (m == 2 && n == 1280 && k == 8192) {
    return dim3(128, 1);
  } else if (m == 2 && n == 8192 && k == 1024) {
    return dim3(64, 1);
  } else if (m == 2 && n == 7168 && k == 8192) {
    return dim3(256, 1);
  } else if (m == 2 && n == 8192 && k == 3584) {
    return dim3(128, 1);
  } else if (m == 3 && n == 1280 && k == 8192) {
    return dim3(128, 1);
  } else if (m == 3 && n == 8192 && k == 1024) {
    return dim3(64, 1);
  } else if (m == 3 && n == 7168 && k == 8192) {
    return dim3(128, 1);
  } else if (m == 3 && n == 8192 && k == 3584) {
    return dim3(128, 1);
  } else if (m == 4 && n == 1280 && k == 8192) {
    return dim3(128, 1);
  } else if (m == 4 && n == 8192 && k == 1024) {
    return dim3(64, 1);
  } else if (m == 4 && n == 7168 && k == 8192) {
    return dim3(128, 1);
  } else if (m == 4 && n == 8192 && k == 3584) {
    return dim3(128, 1);
  } // below four cases are grabbed from llama4 17b_128e
    // model FFN/Attn_linear layers:
    // https://www.internalfb.com/code/fbsource/[26a75e239633]/fbcode/accelerators/workloads/microbench/bench_gemm.py?lines=269
  else if (m == 1 && n == 4096 && k == 5120) {
    return dim3(128, 1);
  } else if (m == 1 && n == 5120 && k == 2048) {
    return dim3(64, 1);
  } else if (m == 1 && n == 896 && k == 5120) {
    return dim3(64, 1);
  } else if (m == 1 && n == 5120 && k == 640) {
    return dim3(64, 1);
  } else if (m == 2 && n == 4096 && k == 5120) {
    return dim3(256, 1);
  } else if (m == 2 && n == 5120 && k == 2048) {
    return dim3(256, 1);
  } else if (m == 2 && n == 896 && k == 5120) {
    return dim3(64, 1);
  } else if (m == 2 && n == 5120 && k == 640) {
    return dim3(32, 1);
  } else {
    // Default block dimensions
    return dim3(32, 1);
  }
}
} // namespace

template <SizeType32 TILE_M, SizeType32 TILE_N>
void fp8fp8FastGemvKernel(
    cutlass::float_e4m3_t* mat,
    cutlass::float_e4m3_t* vec,
    __nv_bfloat16* res,
    const unsigned int b,
    const unsigned int k,
    const unsigned int m,
    const unsigned int n,
    float const* mat_scale,
    float const* vec_scale) {
  // each threadblock handles TILE_M * TILE_N dot products in the resulting
  // matrix.
  // block_size is represented as (block_dim.x, block_dim.y).
  // grid_dim is accordingly calculated based on the number of threadblocks
  // needed to cover the given problem size
  dim3 block_dim = get_best_block_dim(m, n, k);
  dim3 grid_dim(b, m / TILE_M, n / TILE_N * block_dim.y);
  // total number of memory loads needed per thread
  unsigned int num_iter_per_thread = ((k >> 4) + block_dim.x - 1) / block_dim.x;

  check_if_valid_input_dimensions_fp8fp8(m, n, k, TILE_N, block_dim);

  auto stream = at::cuda::getCurrentCUDAStream();

  if (block_dim.x == 128) {
    gemv_quantized_fp8_fp8<TILE_M, TILE_N, 128>
        <<<grid_dim, block_dim, 0, stream>>>(
            mat,
            vec,
            res,
            b,
            k,
            m,
            n,
            mat_scale,
            vec_scale,
            num_iter_per_thread);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else if (block_dim.x == 64) {
    gemv_quantized_fp8_fp8<TILE_M, TILE_N, 64>
        <<<grid_dim, block_dim, 0, stream>>>(
            mat,
            vec,
            res,
            b,
            k,
            m,
            n,
            mat_scale,
            vec_scale,
            num_iter_per_thread);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else if (block_dim.x == 256) {
    gemv_quantized_fp8_fp8<TILE_M, TILE_N, 256>
        <<<grid_dim, block_dim, 0, stream>>>(
            mat,
            vec,
            res,
            b,
            k,
            m,
            n,
            mat_scale,
            vec_scale,
            num_iter_per_thread);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    gemv_quantized_fp8_fp8<TILE_M, TILE_N, 32>
        <<<grid_dim, block_dim, 0, stream>>>(
            mat,
            vec,
            res,
            b,
            k,
            m,
            n,
            mat_scale,
            vec_scale,
            num_iter_per_thread);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

template <SizeType32 TILE_M, SizeType32 TILE_N>
bool fastGemvTemplateCaller(
    cutlass::float_e4m3_t* mat,
    cutlass::float_e4m3_t* vec,
    __nv_bfloat16* res,
    const unsigned int b,
    const unsigned int k,
    const unsigned int m,
    const unsigned int n,
    float const* mat_scale,
    float const* vec_scale) {
  if (m == TILE_M) {
    fp8fp8FastGemvKernel<TILE_M, TILE_N>(
        mat, vec, res, b, k, m, n, mat_scale, vec_scale);
    return true;
  }

  if constexpr (TILE_M < MAX_M_SIZE) {
    return fastGemvTemplateCaller<TILE_M + 1, TILE_N>(
        mat, vec, res, b, k, m, n, mat_scale, vec_scale);
  }
  return false;
}

bool fastGemvLauncher(
    cutlass::float_e4m3_t* mat,
    cutlass::float_e4m3_t* vec,
    __nv_bfloat16* res,
    const unsigned int b,
    const unsigned int k,
    const unsigned int m,
    const unsigned int n,
    float const* mat_scale,
    float const* vec_scale) {
  // Note: based on sweeping result, heuristic TILE_N = 2 here gives best
  // performance over larger TILE_N value. this is potentially because smaller
  // tile_n leads to more threadblocks and thus increase the block concurrency.
  return fastGemvTemplateCaller</* TILE_M=*/1, /* TILE_N=*/2>(
      mat, vec, res, b, k, m, n, mat_scale, vec_scale);
}

at::Tensor fp8fp8bf16_fast_gemv(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    bool is_batched) {
  unsigned int b, m, n, k;
  if (is_batched) {
    TORCH_CHECK(XQ.dim() == 3 && WQ.dim() == 3);
    b = XQ.size(0);
    m = XQ.size(1);
    n = WQ.size(1);
    k = WQ.size(2);
  } else {
    m = XQ.size(0);
    n = WQ.size(0);
    k = WQ.size(1);
    TORCH_CHECK(XQ.size(-1) == k);
  }
  TORCH_CHECK(XQ.is_cuda() && XQ.is_contiguous());
  TORCH_CHECK(WQ.is_cuda() && WQ.is_contiguous());
  std::vector<int64_t> shape =
      is_batched ? std::vector<int64_t>{b, m, n} : std::vector<int64_t>{m, n};
  auto Y = at::empty(shape, XQ.options().dtype(at::kBFloat16));

  if (!is_batched) {
    b = 1;
  }

  bool dispatched = fastGemvLauncher(
      reinterpret_cast<cutlass::float_e4m3_t*>(WQ.data_ptr()), // mat
      reinterpret_cast<cutlass::float_e4m3_t*>(XQ.data_ptr()), // vec
      reinterpret_cast<__nv_bfloat16*>(Y.data_ptr()), // res
      b,
      k,
      m,
      n,
      reinterpret_cast<float const*>(w_scale.data_ptr()),
      reinterpret_cast<float const*>(x_scale.data_ptr()));

  if (!dispatched) {
    throw std::runtime_error("f8f8bf16_fast_gemv cannot run.");
  }

  return Y;
}

} // namespace fbgemm_gpu
