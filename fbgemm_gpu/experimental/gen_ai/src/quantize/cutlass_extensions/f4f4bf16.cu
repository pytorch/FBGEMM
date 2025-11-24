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

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12080)
#include "f4f4bf16/f4f4bf16_manifest.cuh"
#endif

namespace fbgemm_gpu {

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12080)

Kernel_f4f4bf16 get_kernel_via_heuristics(int M, int N, int K, bool use_mx) {
  // MXFP4
  if (use_mx) {
    if (M <= 128) {
      if (N <= 1024) {
        return f4f4bf16_256_128_2_4_1_t;
      } else if (N <= 2048) {
        return f4f4bf16_256_192_4_1_1_t;
      } else {
        return f4f4bf16_128_128_4_1_1_t;
      }
    } else if (M <= 2048) {
      if (M <= 256) {
        if (N == 896) {
          return f4f4bf16_128_128_2_2_1_t;
        } else if (N == 5120) {
          if (K == 640 || K == 5120) {
            return f4f4bf16_128_128_4_1_1_t;
          } else if ((K == 8192) || (K == 16384)) {
            return f4f4bf16_256_128_2_2_1_t;
          }
        } else if (N == 5632) {
          return f4f4bf16_128_192_2_2_1_t;
        } else if (N == 8192) {
          return f4f4bf16_256_128_2_2_1_t;
        }
      } else if (M <= 512) {
        if (N == 896) {
          return f4f4bf16_128_128_2_2_1_t;
        } else if (N == 5120) {
          return f4f4bf16_256_192_4_1_1_t;
        } else if (N == 5632) {
          return f4f4bf16_256_128_2_4_1_t;
        } else if (N == 8192) {
          return f4f4bf16_256_128_2_2_1_t;
        }
      } else if (M <= 1024) {
        if (N == 896) {
          return f4f4bf16_256_128_2_4_1_t;
        } else if (N == 5120) {
          if (K == 640) {
            return f4f4bf16_128_128_1_4_1_t;
          } else if (K == 5120) {
            return f4f4bf16_128_192_4_2_1_t;
          } else if (K == 5120 || K == 16384) {
            return f4f4bf16_256_128_2_4_1_t;
          }
        } else if (N == 5632) {
          return f4f4bf16_256_128_2_4_1_t;
        } else if (N == 8192) {
          return f4f4bf16_256_256_4_1_1_t;
        }
      }
      if (N <= 2048) {
        return f4f4bf16_256_128_2_2_1_t;
      } else if (N <= 8192) {
        return f4f4bf16_128_256_2_1_1_t;
      } else {
        return f4f4bf16_256_256_2_1_1_t;
      }
    } else if (M <= 4096) {
      if (N <= 4096) {
        return f4f4bf16_256_256_4_1_1_t;
      } else if (N <= 8192) {
        return f4f4bf16_256_256_2_1_1_t;
      } else {
        return f4f4bf16_256_128_2_4_1_t;
      }
    } else if (M <= 8192) {
      if (N <= 4096) {
        return f4f4bf16_256_256_2_2_1_t;
      } else if (N <= 8192) {
        return f4f4bf16_256_256_2_4_1_t;
      } else {
        return f4f4bf16_128_256_2_1_1_t;
      }
    } else if (M <= 16384) {
      if (N <= 2048) {
        return f4f4bf16_256_256_2_4_1_t;
      } else if (N <= 8192) {
        return f4f4bf16_128_192_2_2_1_t;
      } else {
        return f4f4bf16_128_256_2_1_1_t;
      }
    } else if (M <= 32768) {
      if (N <= 1024) {
        return f4f4bf16_256_256_2_1_1_t;
      } else if (N <= 4096) {
        return f4f4bf16_128_192_2_2_1_t;
      } else {
        return f4f4bf16_256_192_4_1_1_t;
      }
    } else if (M <= 65536) {
      if (N <= 2048) {
        return f4f4bf16_256_192_2_4_1_t;
      } else if (N <= 4096) {
        return f4f4bf16_256_192_2_2_1_t;
      } else {
        return f4f4bf16_256_256_2_1_1_t;
      }
    } else {
      if (N <= 1024) {
        return f4f4bf16_256_192_2_4_1_t;
      } else {
        return f4f4bf16_256_256_2_2_1_t;
      }
    }
  }
  // NVFP4
  else {
    if (M <= 128) {
      if (N <= 1024) {
        return f4f4bf16_256_128_2_4_1_f;
      } else if (N <= 2048) {
        return f4f4bf16_256_192_4_1_1_f;
      } else {
        return f4f4bf16_128_128_4_1_1_f;
      }
    } else if (M <= 2048) {
      if (M <= 256) {
        if (N == 896) {
          return f4f4bf16_128_128_2_2_1_f;
        } else if (N == 5120) {
          if (K == 640 || K == 5120) {
            return f4f4bf16_128_128_4_1_1_f;
          } else if ((K == 8192) || (K == 16384)) {
            return f4f4bf16_256_128_2_2_1_f;
          }
        } else if (N == 5632) {
          return f4f4bf16_128_192_2_2_1_f;
        } else if (N == 8192 || N == 16384) {
          return f4f4bf16_256_128_2_2_1_f;
        }
      } else if (M <= 512) {
        if (N == 896) {
          return f4f4bf16_128_128_2_2_1_f;
        } else if (N == 5120) {
          return f4f4bf16_256_192_4_1_1_f;
        } else if (N == 5632) {
          return f4f4bf16_256_128_2_4_1_f;
        } else if (N == 8192) {
          return f4f4bf16_256_128_2_2_1_f;
        }
      } else if (M <= 1024) {
        if (N == 896) {
          return f4f4bf16_256_128_2_4_1_f;
        } else if (N == 5120) {
          if (K == 640) {
            return f4f4bf16_128_128_1_4_1_f;
          } else if (K == 5120) {
            return f4f4bf16_128_192_4_2_1_f;
          } else if (K == 5120 || K == 16384) {
            return f4f4bf16_256_128_2_4_1_f;
          }
        } else if (N == 5632) {
          return f4f4bf16_256_128_2_4_1_f;
        } else if (N == 8192) {
          return f4f4bf16_256_256_4_1_1_f;
        }
      }
      if (N <= 2048) {
        return f4f4bf16_256_128_2_2_1_f;
      } else if (N <= 8192) {
        return f4f4bf16_128_256_2_1_1_f;
      } else {
        return f4f4bf16_256_256_2_1_1_f;
      }
    } else if (M <= 4096) {
      if (N <= 4096) {
        return f4f4bf16_256_256_4_1_1_f;
      } else if (N <= 8192) {
        return f4f4bf16_256_256_2_1_1_f;
      } else {
        return f4f4bf16_256_128_2_4_1_f;
      }
    } else if (M <= 8192) {
      if (N <= 4096) {
        return f4f4bf16_256_256_2_2_1_f;
      } else if (N <= 8192) {
        return f4f4bf16_256_256_2_4_1_f;
      } else {
        return f4f4bf16_128_256_2_1_1_f;
      }
    } else if (M <= 16384) {
      if (N <= 2048) {
        return f4f4bf16_256_256_2_4_1_f;
      } else if (N <= 8192) {
        return f4f4bf16_128_192_2_2_1_f;
      } else {
        return f4f4bf16_128_256_2_1_1_f;
      }
    } else if (M <= 32768) {
      if (N <= 1024) {
        return f4f4bf16_256_256_2_1_1_f;
      } else if (N <= 4096) {
        return f4f4bf16_128_192_2_2_1_f;
      } else {
        return f4f4bf16_256_192_4_1_1_f;
      }
    } else if (M <= 65536) {
      if (N <= 2048) {
        return f4f4bf16_256_192_2_4_1_f;
      } else if (N <= 4096) {
        return f4f4bf16_256_192_2_2_1_f;
      } else {
        return f4f4bf16_256_256_2_1_1_f;
      }
    } else {
      if (N <= 1024) {
        return f4f4bf16_256_192_2_4_1_f;
      } else {
        return f4f4bf16_256_256_2_2_1_f;
      }
    }
  }
}

at::Tensor f4f4bf16(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> output,
    std::optional<at::Tensor> global_scale) {
  TORCH_CHECK(XQ.is_cuda() && XQ.is_contiguous());
  TORCH_CHECK(WQ.is_cuda() && WQ.is_contiguous());
  TORCH_CHECK(x_scale.is_cuda() && x_scale.is_contiguous());
  TORCH_CHECK(w_scale.is_cuda() && w_scale.is_contiguous());

  const auto M = XQ.size(0);
  const auto N = WQ.size(0);
  const auto K = XQ.size(1) * 2; // Since K is packed
  constexpr auto BLOCK_SIZE = 16;
  TORCH_CHECK(
      N % BLOCK_SIZE == 0 && K % BLOCK_SIZE == 0,
      "Weight dimensions N and K must be multiples of block size 16");

  if (M == 0 || N == 0 || K == 0) {
    // Use zeros instead of empty for special case where K=0.
    return at::zeros({M, N}, XQ.options().dtype(at::kBFloat16));
  }

  at::Tensor out = output.has_value()
      ? output.value()
      : at::empty({M, N}, XQ.options().dtype(at::kBFloat16));

  auto kernel = get_kernel_via_heuristics(M, N, K, !global_scale.has_value());
  return kernel(XQ, WQ, x_scale, w_scale, out, global_scale);
}

#else

at::Tensor f4f4bf16(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> output,
    std::optional<at::Tensor> global_scale) {
  throw std::runtime_error(
      "CUDA version is older than 12.8"); // requires CUDA>=12.8
}

#endif

} // namespace fbgemm_gpu
