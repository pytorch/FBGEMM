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

#include "include/fast_gemv.cuh"

namespace fbgemm_gpu {

// The heuristics are derived by sweeping over 4 different
// problem sizes we care about and selected the best elapsed time/bw
// combination. See more in
// deeplearning/fbgemm/fbgemm_gpu/experimental/gen_ai/src/quantize/fast_gemv/sweep_utils.py
dim3 get_best_block_dim(int m, int n, int k) {
  if (m == 1 && n == 1280 && k == 8192) {
    return dim3(128, 4);
  } else if (m == 1 && n == 8192 && k == 1024) {
    return dim3(32, 8);
  } else if (m == 1 && n == 7168 && k == 8192) {
    return dim3(256, 1);
  } else if (m == 1 && n == 8192 && k == 3584) {
    return dim3(64, 2);
  } else {
    // Default block dimensions
    return dim3(32, 4);
  }
}

at::Tensor bf16_fast_gemv(at::Tensor X, at::Tensor W) {
  // X: M x K
  // W: N x K
  auto m = X.size(0);
  auto n = W.size(0);
  auto k = W.size(1);

  TORCH_CHECK(X.is_cuda() && X.is_contiguous());
  TORCH_CHECK(W.is_cuda() && W.is_contiguous());

  dim3 block_dim = get_best_block_dim(m, n, k);

  TORCH_CHECK(
      n % block_dim.y == 0,
      "Invalid block dimensions: n (",
      n,
      ") must be divisible by block_dim.y (",
      block_dim.y,
      "). Received n: ",
      n,
      ", block_dim.y: ",
      block_dim.y,
      " Please either use a `n` which is divisible by `block_dim.y`, or update "
      "`get_best_block_dim()` heuristics to choose another `block_dim.y`. "
      " All current params - m: ",
      m,
      ", n: ",
      n,
      ", k: ",
      k,
      ", block_dim.x: ",
      block_dim.x,
      ", block_dim.y: ",
      block_dim.y,
      ".");
  TORCH_CHECK(
      k % block_dim.x == 0,
      "Invalid block dimensions: k (",
      k,
      ") must be divisible by block_dim.x (",
      block_dim.x,
      "). Received k: ",
      k,
      ", block_dim.x: ",
      block_dim.x,
      " Please either use a `k` which is divisible by `block_dim.x`, or update "
      "`get_best_block_dim()` heuristics to choose another `block_dim.x`."
      " All current params - m: ",
      m,
      ", n: ",
      n,
      ", k: ",
      k,
      ", block_dim.x: ",
      block_dim.x,
      ", block_dim.y: ",
      block_dim.y,
      ".");
  TORCH_CHECK(
      (k / block_dim.x) % 8 == 0,
      "Invalid num_per_thread: (",
      k / block_dim.x,
      ") must be divisible by 8.",
      " Received k: ",
      k,
      ", block_dim.x: ",
      block_dim.x,
      " Please either use a `k` that `k / block_dim.x` that is divisble by 8, or update "
      "`get_best_block_dim()` heuristics to choose another `block_dim.x`."
      " All current params - m: ",
      m,
      ", n: ",
      n,
      ", k: ",
      k,
      ", block_dim.x: ",
      block_dim.x,
      ", block_dim.y: ",
      block_dim.y,
      ".");

  dim3 grid_dim(1, n / block_dim.y);
  unsigned int num_per_thread = k / block_dim.x;

  auto stream = at::cuda::getCurrentCUDAStream();

  auto Y = at::empty({m, n}, X.options().dtype(at::kBFloat16));

  gemv_bf16<<<grid_dim, block_dim, 0, stream>>>(
      reinterpret_cast<__nv_bfloat16*>(W.data_ptr()), // mat
      reinterpret_cast<__nv_bfloat16*>(X.data_ptr()), // vec
      reinterpret_cast<__nv_bfloat16*>(Y.data_ptr()), // res
      k,
      num_per_thread);

  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return Y;
}

} // namespace fbgemm_gpu
