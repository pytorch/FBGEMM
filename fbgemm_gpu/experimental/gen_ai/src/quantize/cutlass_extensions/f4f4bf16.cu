/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cutlass/util/device_memory.h>
// clang-format on

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12080)
#include "f4f4bf16/f4f4bf16_manifest.cuh"
#endif

namespace fbgemm_gpu {

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12080)

at::Tensor dispatch_f4f4bf16_kernel(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> global_scale,
    bool use_mx = true) {
  auto M = XQ.size(0);
  auto K = XQ.size(1);
  auto N = WQ.size(0);
  auto BLOCK_SIZE = 16;
  TORCH_CHECK(
      N % BLOCK_SIZE == 0 && K % BLOCK_SIZE == 0,
      "Weight dimensions N and K must be multiples of block size 16");

  // MXFP4
  if (use_mx) {
    if (M <= 128) {
      if (N <= 1024) {
        return f4f4bf16_256_128_2_4_1_t(XQ, WQ, x_scale, w_scale, global_scale);
      } else if (N <= 2048) {
        return f4f4bf16_256_192_4_1_1_t(XQ, WQ, x_scale, w_scale, global_scale);
      } else {
        return f4f4bf16_128_128_4_1_1_t(XQ, WQ, x_scale, w_scale, global_scale);
      }
    } else if (M <= 2048) {
      if (N <= 2048) {
        return f4f4bf16_256_128_2_2_1_t(XQ, WQ, x_scale, w_scale, global_scale);
      } else if (N <= 8192) {
        return f4f4bf16_128_256_2_1_1_t(XQ, WQ, x_scale, w_scale, global_scale);
      } else {
        return f4f4bf16_256_256_2_1_1_t(XQ, WQ, x_scale, w_scale, global_scale);
      }
    } else if (M <= 4096) {
      if (N <= 4096) {
        return f4f4bf16_256_256_4_1_1_t(XQ, WQ, x_scale, w_scale, global_scale);
      } else if (N <= 8192) {
        return f4f4bf16_256_256_2_1_1_t(XQ, WQ, x_scale, w_scale, global_scale);
      } else {
        return f4f4bf16_256_128_2_4_1_t(XQ, WQ, x_scale, w_scale, global_scale);
      }
    } else if (M <= 8192) {
      if (N <= 4096) {
        return f4f4bf16_256_256_2_2_1_t(XQ, WQ, x_scale, w_scale, global_scale);
      } else if (N <= 8192) {
        return f4f4bf16_256_256_2_4_1_t(XQ, WQ, x_scale, w_scale, global_scale);
      } else {
        return f4f4bf16_128_256_2_1_1_t(XQ, WQ, x_scale, w_scale, global_scale);
      }
    } else if (M <= 16384) {
      if (N <= 2048) {
        return f4f4bf16_256_256_2_4_1_t(XQ, WQ, x_scale, w_scale, global_scale);
      } else if (N <= 8192) {
        return f4f4bf16_128_192_2_2_1_t(XQ, WQ, x_scale, w_scale, global_scale);
      } else {
        return f4f4bf16_128_256_2_1_1_t(XQ, WQ, x_scale, w_scale, global_scale);
      }
    } else if (M <= 32768) {
      if (N <= 1024) {
        return f4f4bf16_256_256_2_1_1_t(XQ, WQ, x_scale, w_scale, global_scale);
      } else if (N <= 4096) {
        return f4f4bf16_128_192_2_2_1_t(XQ, WQ, x_scale, w_scale, global_scale);
      } else {
        return f4f4bf16_256_192_4_1_1_t(XQ, WQ, x_scale, w_scale, global_scale);
      }
    } else if (M <= 65536) {
      if (N <= 2048) {
        return f4f4bf16_256_192_2_4_1_t(XQ, WQ, x_scale, w_scale, global_scale);
      } else if (N <= 4096) {
        return f4f4bf16_256_192_2_2_1_t(XQ, WQ, x_scale, w_scale, global_scale);
      } else {
        return f4f4bf16_256_256_2_1_1_t(XQ, WQ, x_scale, w_scale, global_scale);
      }
    } else {
      if (N <= 1024) {
        return f4f4bf16_256_192_2_4_1_t(XQ, WQ, x_scale, w_scale, global_scale);
      } else {
        return f4f4bf16_256_256_2_2_1_t(XQ, WQ, x_scale, w_scale, global_scale);
      }
    }
  }
  // NVFP4
  else {
    if (M <= 128) {
      if (N <= 1024) {
        return f4f4bf16_256_128_2_4_1_f(XQ, WQ, x_scale, w_scale, global_scale);
      } else if (N <= 2048) {
        return f4f4bf16_256_192_4_1_1_f(XQ, WQ, x_scale, w_scale, global_scale);
      } else {
        return f4f4bf16_128_128_4_1_1_f(XQ, WQ, x_scale, w_scale, global_scale);
      }
    } else if (M <= 2048) {
      if (N <= 2048) {
        return f4f4bf16_256_128_2_2_1_f(XQ, WQ, x_scale, w_scale, global_scale);
      } else if (N <= 8192) {
        return f4f4bf16_128_256_2_1_1_f(XQ, WQ, x_scale, w_scale, global_scale);
      } else {
        return f4f4bf16_256_256_2_1_1_f(XQ, WQ, x_scale, w_scale, global_scale);
      }
    } else if (M <= 4096) {
      if (N <= 4096) {
        return f4f4bf16_256_256_4_1_1_f(XQ, WQ, x_scale, w_scale, global_scale);
      } else if (N <= 8192) {
        return f4f4bf16_256_256_2_1_1_f(XQ, WQ, x_scale, w_scale, global_scale);
      } else {
        return f4f4bf16_256_128_2_4_1_f(XQ, WQ, x_scale, w_scale, global_scale);
      }
    } else if (M <= 8192) {
      if (N <= 4096) {
        return f4f4bf16_256_256_2_2_1_f(XQ, WQ, x_scale, w_scale, global_scale);
      } else if (N <= 8192) {
        return f4f4bf16_256_256_2_4_1_f(XQ, WQ, x_scale, w_scale, global_scale);
      } else {
        return f4f4bf16_128_256_2_1_1_f(XQ, WQ, x_scale, w_scale, global_scale);
      }
    } else if (M <= 16384) {
      if (N <= 2048) {
        return f4f4bf16_256_256_2_4_1_f(XQ, WQ, x_scale, w_scale, global_scale);
      } else if (N <= 8192) {
        return f4f4bf16_128_192_2_2_1_f(XQ, WQ, x_scale, w_scale, global_scale);
      } else {
        return f4f4bf16_128_256_2_1_1_f(XQ, WQ, x_scale, w_scale, global_scale);
      }
    } else if (M <= 32768) {
      if (N <= 1024) {
        return f4f4bf16_256_256_2_1_1_f(XQ, WQ, x_scale, w_scale, global_scale);
      } else if (N <= 4096) {
        return f4f4bf16_128_192_2_2_1_f(XQ, WQ, x_scale, w_scale, global_scale);
      } else {
        return f4f4bf16_256_192_4_1_1_f(XQ, WQ, x_scale, w_scale, global_scale);
      }
    } else if (M <= 65536) {
      if (N <= 2048) {
        return f4f4bf16_256_192_2_4_1_f(XQ, WQ, x_scale, w_scale, global_scale);
      } else if (N <= 4096) {
        return f4f4bf16_256_192_2_2_1_f(XQ, WQ, x_scale, w_scale, global_scale);
      } else {
        return f4f4bf16_256_256_2_1_1_f(XQ, WQ, x_scale, w_scale, global_scale);
      }
    } else {
      if (N <= 1024) {
        return f4f4bf16_256_192_2_4_1_f(XQ, WQ, x_scale, w_scale, global_scale);
      } else {
        return f4f4bf16_256_256_2_2_1_f(XQ, WQ, x_scale, w_scale, global_scale);
      }
    }
  }
}

at::Tensor f4f4bf16(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> global_scale,
    bool use_mx = true) {
  return dispatch_f4f4bf16_kernel(
      XQ, WQ, x_scale, w_scale, global_scale, use_mx);
}

#else

at::Tensor f4f4bf16(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> global_scale,
    bool use_mx = true) {
  throw std::runtime_error(
      "CUDA version is older than 12.8"); // requires CUDA>=12.8
}

#endif

} // namespace fbgemm_gpu
