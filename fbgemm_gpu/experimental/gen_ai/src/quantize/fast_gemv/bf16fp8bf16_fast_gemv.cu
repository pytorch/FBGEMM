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

// The heuristics are derived by sweeping over 4 different
// problem sizes we care about and selected the best elapsed time/bw
// combination. See more in
// deeplearning/fbgemm/fbgemm_gpu/experimental/gen_ai/src/quantize/fast_gemv/sweep_utils.py
namespace {
dim3 get_best_block_dim(int m, int n, int k) {
  if (m == 1 && n == 1280 && k == 8192) {
    return dim3(128, 1);
  } else if (m == 1 && n == 8192 && k == 1024) {
    return dim3(32, 4);
  } else if (m == 1 && n == 7168 && k == 8192) {
    return dim3(128, 1);
  } else if (m == 1 && n == 8192 && k == 3584) {
    return dim3(64, 2);
  } else {
    // Default block dimensions
    return dim3(32, 4);
  }
}
} // namespace

at::Tensor
bf16fp8bf16_fast_gemv(at::Tensor X, at::Tensor W, at::Tensor w_scale) {
  // X: M x K
  // W: N x K
  auto m = X.size(0);
  auto n = W.size(0);
  auto k = W.size(1);

  TORCH_CHECK(X.is_cuda() && X.is_contiguous());
  TORCH_CHECK(W.is_cuda() && W.is_contiguous());

  dim3 block_dim = get_best_block_dim(m, n, k);

  check_if_valid_block_dimensions(m, n, k, block_dim);

  dim3 grid_dim(1, n / block_dim.y);
  unsigned int num_per_thread = k / block_dim.x;

  auto stream = at::cuda::getCurrentCUDAStream();

  auto Y = at::empty({m, n}, X.options().dtype(at::kBFloat16));

  gemv_quantized_bf16_fp8<<<grid_dim, block_dim, 0, stream>>>(
      reinterpret_cast<cutlass::float_e4m3_t*>(W.data_ptr()), // mat
      reinterpret_cast<__nv_bfloat16*>(X.data_ptr()), // vec
      reinterpret_cast<__nv_bfloat16*>(Y.data_ptr()), // res
      k,
      reinterpret_cast<float const*>(w_scale.data_ptr()),
      num_per_thread);

  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return Y;
}

} // namespace fbgemm_gpu
