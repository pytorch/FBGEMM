/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/library.h>

#include "bf16bf16bf16_grouped_wgrad/bf16bf16bf16_grouped_wgrad_manifest.cuh"
#include "fbgemm_gpu/quantize/tuning_cache.cuh"
#include "fbgemm_gpu/quantize/utils.h"
#include "fbgemm_gpu/quantize/utils_gpu.h"

namespace fbgemm_gpu {

#if CUDART_VERSION >= 12000

namespace {
TuningCache& getTuningCache() {
  static TuningCache cache("bf16bf16bf16_grouped_wgrad");
  return cache;
}
} // namespace

Kernel_bf16bf16bf16_grouped_wgrad
get_wgrad_kernel_via_heuristic(int arch, int G, int total_M, int N, int K) {
  // Use heuristics to pick best kernel implementation.
  if (arch == 10) {
    // Llama4 shapes
    if ((N == 5120 && K == 1024) || (N == 2048 && K == 5120)) {
      if (total_M <= 256) {
        return bf16bf16bf16_grouped_wgrad_256_32_128_2_1_1_10_f;
      } else if (total_M <= 512) {
        return bf16bf16bf16_grouped_wgrad_256_64_128_2_1_1_10_f;
      } else if (total_M <= 1024) {
        return bf16bf16bf16_grouped_wgrad_256_128_128_2_1_1_10_f;
      } else {
        return bf16bf16bf16_grouped_wgrad_256_256_128_2_1_1_10_f;
      }
    }

    // Fallback to legacy heuristic.
    if (total_M <= 64 || (total_M <= 256 and N <= 1024)) {
      if (K <= 4096) {
        return bf16bf16bf16_grouped_wgrad_256_32_128_2_1_1_10_f;
      } else {
        return bf16bf16bf16_grouped_wgrad_128_32_128_2_1_1_10_f;
      }
    } else if (total_M <= 512) {
      if (N <= 1024) {
        return bf16bf16bf16_grouped_wgrad_128_64_128_2_1_1_10_f;
      } else if (N <= 8192) {
        if (K <= 2048) {
          return bf16bf16bf16_grouped_wgrad_256_32_128_2_1_1_10_f;
        } else if (K <= 4096) {
          return bf16bf16bf16_grouped_wgrad_128_32_128_2_1_1_10_f;
        } else {
          return bf16bf16bf16_grouped_wgrad_128_64_128_2_1_1_10_f;
        }
      }
    } else if (total_M <= 1024) {
      if (N <= 1024) {
        return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_10_f;
      } else if (N <= 8192) {
        if (K <= 2048) {
          return bf16bf16bf16_grouped_wgrad_256_64_128_2_1_1_10_f;
        } else if (K <= 4096) {
          return bf16bf16bf16_grouped_wgrad_128_64_128_2_1_1_10_f;
        } else {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_10_f;
        }
      }
    } else if (total_M <= 2048) {
      if (N <= 1024) {
        return bf16bf16bf16_grouped_wgrad_256_256_128_2_1_1_10_f;
      } else if (N <= 8192) {
        if (K <= 2048) {
          return bf16bf16bf16_grouped_wgrad_256_128_128_2_1_1_10_f;
        } else if (K <= 4096) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_10_f;
        }
      }
    }
    return bf16bf16bf16_grouped_wgrad_256_256_128_2_1_1_10_f;
  } else { // arch == 9
    // Llama4.x pretraining
    if (total_M == 8192) {
      if (N == 2560) {
        if (K == 1280) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_4_2_1_9_t;
        } else if (K == 5120) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_4_1_9_t;
        }
      } else if (N == 3072) {
        if (K == 1536) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_4_1_9_t;
        } else if (K == 6144) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        }
      } else if (N == 5120) {
        return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
      } else if (N == 6144) {
        if (K == 1536) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_4_1_9_t;
        } else if (K == 6144) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        }
      }
    }

    if (total_M == 16384) {
      if (N == 2560 || N == 3072) {
        return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
      } else if (N == 5120) {
        if (K == 1280) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        } else if (K == 5120) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_4_1_9_t;
        }
      } else if (N == 6144) {
        if (K == 1536) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_4_1_9_t;
        } else if (K == 6144) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_4_1_9_t;
        }
      }
    }

    if (total_M == 65536) {
      if (N <= 512) {
        if (K <= 256) {
          return bf16bf16bf16_grouped_wgrad_128_64_128_1_2_1_9_f;
        } else if (K <= 512) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_4_1_9_t;
        }
      } else if (N <= 768) {
        if (K <= 384) {
          return bf16bf16bf16_grouped_wgrad_128_64_128_1_2_1_9_f;
        } else if (K <= 768) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        }
      } else if (N <= 1024) {
        if (K <= 1024) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        }
      } else if (N <= 1280) {
        if (K <= 640) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_t;
        } else if (K <= 1280) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_2_1_9_t;
        }
      } else if (N <= 1536) {
        if (K <= 768) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        } else if (K <= 1536) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        }
      } else if (N <= 1792) {
        if (K <= 1792) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_4_1_9_t;
        }
      } else if (N <= 2048) {
        if (K <= 2048) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        }
      } else if (N <= 2560) {
        if (K <= 1280) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_2_1_9_t;
        } else if (K <= 2560) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        }
      } else if (N <= 3072) {
        if (K <= 1536) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        } else if (K <= 3072) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_2_1_9_t;
        }
      }
    }

    // Fallback to legacy heuristic
    if (total_M <= 128) {
      if (N <= 128) {
        if (K <= 128) {
          return bf16bf16bf16_grouped_wgrad_128_32_128_1_4_1_9_f;
        } else if (K <= 256) {
          return bf16bf16bf16_grouped_wgrad_128_32_128_1_1_1_9_f;
        } else if (K <= 512) {
          return bf16bf16bf16_grouped_wgrad_128_64_128_1_1_1_9_f;
        } else if (K <= 1024) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_1_1_9_f;
        } else if (K <= 2048) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_1_1_9_t;
        } else {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_f;
        }
      } else if (N <= 256) {
        if (K <= 128) {
          return bf16bf16bf16_grouped_wgrad_128_32_128_1_1_1_9_f;
        } else if (K <= 256) {
          return bf16bf16bf16_grouped_wgrad_128_64_128_1_1_1_9_f;
        } else if (K <= 512) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_1_1_9_f;
        } else if (K <= 1024) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        } else if (K <= 4096) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_f;
        } else if (K <= 8192) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_f;
        } else {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_f;
        }
      } else if (N <= 512) {
        if (K <= 128) {
          return bf16bf16bf16_grouped_wgrad_128_64_128_1_1_1_9_f;
        } else if (K <= 2048) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_f;
        } else if (K <= 4096) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_f;
        } else {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_f;
        }
      } else if (N <= 1024) {
        if (K <= 128) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_1_1_9_f;
        } else if (K <= 256) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_f;
        } else if (K <= 512) {
          return bf16bf16bf16_grouped_wgrad_256_64_128_1_4_1_9_f;
        } else if (K <= 4096) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_f;
        } else if (K <= 8192) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_f;
        } else {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_f;
        }
      } else if (N <= 2048) {
        if (K <= 512) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_f;
        } else if (K <= 1024) {
          return bf16bf16bf16_grouped_wgrad_256_64_128_1_2_1_9_f;
        } else if (K <= 8192) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_f;
        } else {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_f;
        }
      } else if (N <= 4096) {
        if (K <= 128) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_f;
        } else if (K <= 256) {
          return bf16bf16bf16_grouped_wgrad_256_64_128_1_2_1_9_f;
        } else if (K <= 8192) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_f;
        } else {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        }
      } else if (N <= 8192) {
        if (K <= 8192) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_f;
        } else {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        }
      } else {
        if (K <= 4096) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_f;
        } else if (K <= 8192) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        } else {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_t;
        }
      }
    } else if (total_M <= 256) {
      if (N <= 128) {
        if (K <= 256) {
          return bf16bf16bf16_grouped_wgrad_128_32_128_1_1_1_9_f;
        } else if (K <= 512) {
          return bf16bf16bf16_grouped_wgrad_128_64_128_1_1_1_9_f;
        } else if (K <= 1024) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_f;
        } else if (K <= 2048) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_1_1_9_f;
        } else {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_f;
        }
      } else if (N <= 256) {
        if (K <= 128) {
          return bf16bf16bf16_grouped_wgrad_128_32_128_1_1_1_9_f;
        } else if (K <= 256) {
          return bf16bf16bf16_grouped_wgrad_128_64_128_1_1_1_9_f;
        } else if (K <= 512) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_1_1_9_f;
        } else if (K <= 2048) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_f;
        } else {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_f;
        }
      } else if (N <= 512) {
        if (K <= 128) {
          return bf16bf16bf16_grouped_wgrad_128_64_128_1_1_1_9_f;
        } else if (K <= 512) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_1_1_9_f;
        } else if (K <= 1024) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        } else if (K <= 2048) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_f;
        } else {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_f;
        }
      } else if (N <= 1024) {
        if (K <= 512) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_f;
        } else if (K <= 1024) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_1_1_9_f;
        } else if (K <= 2048) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_f;
        } else if (K <= 4096) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_f;
        } else if (K <= 8192) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_f;
        } else {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_f;
        }
      } else if (N <= 2048) {
        if (K <= 256) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_f;
        } else if (K <= 512) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_f;
        } else if (K <= 4096) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_f;
        } else {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_f;
        }
      } else if (N <= 4096) {
        if (K <= 256) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_f;
        } else if (K <= 512) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_f;
        } else if (K <= 8192) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_f;
        } else {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        }
      } else if (N <= 8192) {
        if (K <= 128) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_f;
        } else if (K <= 256) {
          return bf16bf16bf16_grouped_wgrad_256_64_128_1_2_1_9_f;
        } else if (K <= 4096) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_f;
        } else {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        }
      } else {
        if (K <= 4096) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_f;
        } else {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_t;
        }
      }
    } else if (total_M <= 512) {
      if (N <= 128) {
        if (K <= 256) {
          return bf16bf16bf16_grouped_wgrad_128_32_128_1_1_1_9_f;
        } else if (K <= 512) {
          return bf16bf16bf16_grouped_wgrad_128_64_128_1_1_1_9_f;
        } else if (K <= 2048) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_1_1_9_f;
        } else {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_f;
        }
      } else if (N <= 256) {
        if (K <= 128) {
          return bf16bf16bf16_grouped_wgrad_128_32_128_1_1_1_9_f;
        } else if (K <= 256) {
          return bf16bf16bf16_grouped_wgrad_128_64_128_1_1_1_9_f;
        } else if (K <= 512) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_f;
        } else if (K <= 1024) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_1_1_9_f;
        } else if (K <= 2048) {
          return bf16bf16bf16_grouped_wgrad_256_64_128_1_2_1_9_f;
        } else if (K <= 8192) {
          return bf16bf16bf16_grouped_wgrad_256_64_128_1_1_1_9_f;
        } else {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_f;
        }
      } else if (N <= 512) {
        if (K <= 128) {
          return bf16bf16bf16_grouped_wgrad_128_64_128_1_1_1_9_f;
        } else if (K <= 256) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_1_1_9_f;
        } else if (K <= 2048) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_f;
        } else if (K <= 8192) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_f;
        } else {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_f;
        }
      } else if (N <= 1024) {
        if (K <= 128) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_f;
        } else if (K <= 256) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        } else if (K <= 2048) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_f;
        } else if (K <= 4096) {
          return bf16bf16bf16_grouped_wgrad_256_64_128_1_2_1_9_f;
        } else {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_f;
        }
      } else if (N <= 2048) {
        if (K <= 128) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_f;
        } else if (K <= 512) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        } else if (K <= 1024) {
          return bf16bf16bf16_grouped_wgrad_256_64_128_1_2_1_9_f;
        } else if (K <= 8192) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_f;
        } else {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_f;
        }
      } else if (N <= 4096) {
        if (K <= 128) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_f;
        } else if (K <= 256) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_f;
        } else if (K <= 512) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_f;
        } else if (K <= 1024) {
          return bf16bf16bf16_grouped_wgrad_256_64_128_1_2_1_9_f;
        } else if (K <= 8192) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_f;
        } else {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_f;
        }
      } else if (N <= 8192) {
        if (K <= 128) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_f;
        } else if (K <= 256) {
          return bf16bf16bf16_grouped_wgrad_256_64_128_1_2_1_9_f;
        } else if (K <= 1024) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_f;
        } else if (K <= 2048) {
          return bf16bf16bf16_grouped_wgrad_256_64_128_1_2_1_9_f;
        } else if (K <= 8192) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_f;
        } else {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        }
      } else {
        if (K <= 512) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_f;
        } else if (K <= 1024) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        } else if (K <= 2048) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_f;
        } else if (K <= 4096) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_f;
        } else {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_t;
        }
      }
    } else if (total_M <= 1024) {
      if (N <= 128) {
        if (K <= 128) {
          return bf16bf16bf16_grouped_wgrad_128_32_128_1_4_1_9_f;
        } else if (K <= 256) {
          return bf16bf16bf16_grouped_wgrad_128_32_128_1_1_1_9_f;
        } else if (K <= 512) {
          return bf16bf16bf16_grouped_wgrad_128_64_128_1_1_1_9_f;
        } else if (K <= 1024) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_f;
        } else if (K <= 2048) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_1_1_9_f;
        } else {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_f;
        }
      } else if (N <= 256) {
        if (K <= 128) {
          return bf16bf16bf16_grouped_wgrad_128_32_128_1_1_1_9_f;
        } else if (K <= 256) {
          return bf16bf16bf16_grouped_wgrad_128_64_128_1_1_1_9_f;
        } else if (K <= 512) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_1_1_9_f;
        } else if (K <= 2048) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        } else if (K <= 4096) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_f;
        } else {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_f;
        }
      } else if (N <= 512) {
        if (K <= 128) {
          return bf16bf16bf16_grouped_wgrad_128_64_128_1_1_1_9_f;
        } else if (K <= 256) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_f;
        } else if (K <= 1024) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_f;
        } else if (K <= 2048) {
          return bf16bf16bf16_grouped_wgrad_256_64_128_1_2_1_9_f;
        } else if (K <= 8192) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_f;
        } else {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_1_1_9_f;
        }
      } else if (N <= 1024) {
        if (K <= 1024) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_f;
        } else if (K <= 2048) {
          return bf16bf16bf16_grouped_wgrad_256_64_128_1_2_1_9_f;
        } else if (K <= 4096) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_f;
        } else if (K <= 8192) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_f;
        } else {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_f;
        }
      } else if (N <= 2048) {
        if (K <= 256) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_f;
        } else if (K <= 512) {
          return bf16bf16bf16_grouped_wgrad_256_64_128_1_2_1_9_f;
        } else if (K <= 4096) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_f;
        } else if (K <= 8192) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_f;
        } else {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_f;
        }
      } else if (N <= 4096) {
        if (K <= 128) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_f;
        } else if (K <= 256) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_t;
        } else if (K <= 512) {
          return bf16bf16bf16_grouped_wgrad_256_64_128_1_2_1_9_f;
        } else if (K <= 8192) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_f;
        } else {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_f;
        }
      } else if (N <= 8192) {
        if (K <= 128) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_f;
        } else if (K <= 256) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_f;
        } else if (K <= 512) {
          return bf16bf16bf16_grouped_wgrad_256_64_128_1_2_1_9_f;
        } else if (K <= 1024) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_f;
        } else if (K <= 2048) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_1_1_9_f;
        } else if (K <= 4096) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        } else if (K <= 8192) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_f;
        } else {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        }
      } else {
        if (K <= 128) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_t;
        } else if (K <= 256) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_f;
        } else if (K <= 1024) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_1_1_9_f;
        } else if (K <= 2048) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_f;
        } else if (K <= 8192) {
          return bf16bf16bf16_grouped_wgrad_256_64_128_1_2_1_9_f;
        } else {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_f;
        }
      }
    } else if (total_M <= 2048) {
      if (N <= 128) {
        if (K <= 128) {
          return bf16bf16bf16_grouped_wgrad_128_32_128_1_4_1_9_f;
        } else if (K <= 256) {
          return bf16bf16bf16_grouped_wgrad_128_32_128_1_1_1_9_f;
        } else if (K <= 512) {
          return bf16bf16bf16_grouped_wgrad_128_64_128_1_2_1_9_f;
        } else if (K <= 1024) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_1_1_9_f;
        } else if (K <= 8192) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        } else {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_4_1_9_f;
        }
      } else if (N <= 256) {
        if (K <= 128) {
          return bf16bf16bf16_grouped_wgrad_128_32_128_1_1_1_9_f;
        } else if (K <= 256) {
          return bf16bf16bf16_grouped_wgrad_128_64_128_1_1_1_9_f;
        } else if (K <= 512) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_1_1_9_f;
        } else if (K <= 2048) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        } else if (K <= 4096) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_t;
        } else if (K <= 8192) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_1_1_9_f;
        } else {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        }
      } else if (N <= 512) {
        if (K <= 128) {
          return bf16bf16bf16_grouped_wgrad_128_64_128_1_2_1_9_f;
        } else if (K <= 256) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_f;
        } else if (K <= 512) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_t;
        } else if (K <= 1024) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        } else if (K <= 2048) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_f;
        } else if (K <= 4096) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_t;
        } else if (K <= 8192) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_f;
        } else {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        }
      } else if (N <= 1024) {
        if (K <= 128) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_1_1_9_f;
        } else if (K <= 2048) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        } else if (K <= 4096) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_f;
        } else {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        }
      } else if (N <= 2048) {
        if (K <= 128) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_t;
        } else if (K <= 256) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        } else if (K <= 512) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_f;
        } else if (K <= 1024) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        } else if (K <= 2048) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_f;
        } else {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        }
      } else if (N <= 4096) {
        if (K <= 128) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_t;
        } else if (K <= 512) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        } else if (K <= 1024) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_f;
        } else if (K <= 4096) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        } else {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_t;
        }
      } else if (N <= 8192) {
        if (K <= 128) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_f;
        } else if (K <= 256) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        } else if (K <= 512) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_f;
        } else if (K <= 2048) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        } else {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_t;
        }
      } else {
        if (K <= 128) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_t;
        } else if (K <= 256) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        } else if (K <= 512) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_f;
        } else if (K <= 1024) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        } else if (K <= 8192) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_t;
        } else {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        }
      }
    } else if (total_M <= 4096) {
      if (N <= 128) {
        if (K <= 128) {
          return bf16bf16bf16_grouped_wgrad_128_32_128_1_4_1_9_f;
        } else if (K <= 256) {
          return bf16bf16bf16_grouped_wgrad_128_32_128_1_1_1_9_f;
        } else if (K <= 512) {
          return bf16bf16bf16_grouped_wgrad_128_64_128_1_1_1_9_f;
        } else if (K <= 1024) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_1_1_9_f;
        } else {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        }
      } else if (N <= 256) {
        if (K <= 128) {
          return bf16bf16bf16_grouped_wgrad_128_32_128_1_1_1_9_f;
        } else if (K <= 256) {
          return bf16bf16bf16_grouped_wgrad_128_64_128_1_1_1_9_f;
        } else if (K <= 512) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_1_1_9_f;
        } else if (K <= 2048) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        } else if (K <= 4096) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_t;
        } else if (K <= 8192) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_4_1_9_t;
        } else {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_f;
        }
      } else if (N <= 512) {
        if (K <= 128) {
          return bf16bf16bf16_grouped_wgrad_128_64_128_1_1_1_9_f;
        } else if (K <= 256) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_1_1_9_f;
        } else if (K <= 512) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        } else if (K <= 1024) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_t;
        } else {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        }
      } else if (N <= 1024) {
        if (K <= 128) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_f;
        } else if (K <= 1024) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        } else if (K <= 2048) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_t;
        } else {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        }
      } else if (N <= 2048) {
        if (K <= 128) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_t;
        } else {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        }
      } else if (N <= 4096) {
        if (K <= 128) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_t;
        } else if (K <= 1024) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        } else if (K <= 2048) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_t;
        } else if (K <= 8192) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        } else {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_t;
        }
      } else if (N <= 8192) {
        if (K <= 128) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_t;
        } else if (K <= 2048) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        } else if (K <= 8192) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_t;
        } else {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        }
      } else {
        if (K <= 128) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_4_1_1_9_t;
        } else if (K <= 1024) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        } else if (K <= 8192) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_t;
        } else {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        }
      }
    } else if (total_M <= 8192) {
      if (N <= 128) {
        if (K <= 128) {
          return bf16bf16bf16_grouped_wgrad_128_32_128_1_4_1_9_f;
        } else if (K <= 256) {
          return bf16bf16bf16_grouped_wgrad_128_32_128_1_1_1_9_f;
        } else if (K <= 512) {
          return bf16bf16bf16_grouped_wgrad_128_64_128_1_1_1_9_f;
        } else if (K <= 1024) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_1_1_9_f;
        } else if (K <= 2048) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_4_1_9_t;
        } else if (K <= 4096) {
          return bf16bf16bf16_grouped_wgrad_128_64_128_1_1_1_9_f;
        } else {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        }
      } else if (N <= 256) {
        if (K <= 128) {
          return bf16bf16bf16_grouped_wgrad_128_32_128_1_1_1_9_f;
        } else if (K <= 256) {
          return bf16bf16bf16_grouped_wgrad_128_64_128_1_1_1_9_f;
        } else if (K <= 512) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_f;
        } else if (K <= 1024) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_1_1_9_t;
        } else if (K <= 2048) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_t;
        } else if (K <= 4096) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        } else if (K <= 8192) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_t;
        } else {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        }
      } else if (N <= 512) {
        if (K <= 128) {
          return bf16bf16bf16_grouped_wgrad_128_64_128_1_2_1_9_f;
        } else if (K <= 256) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_1_1_9_f;
        } else if (K <= 512) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        } else if (K <= 1024) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_4_4_1_9_t;
        } else {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        }
      } else if (N <= 1024) {
        if (K <= 128) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_f;
        } else if (K <= 256) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_4_2_1_9_t;
        } else if (K <= 512) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        } else if (K <= 1024) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_t;
        } else {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        }
      } else if (N <= 2048) {
        if (K <= 128) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_t;
        } else if (K <= 256) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        } else if (K <= 512) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_t;
        } else {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        }
      } else if (N <= 4096) {
        if (K <= 128) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_f;
        } else {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        }
      } else if (N <= 8192) {
        if (K <= 128) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_t;
        } else if (K <= 2048) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        } else if (K <= 8192) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_t;
        } else {
          return bf16bf16bf16_grouped_wgrad_128_128_128_4_4_1_9_t;
        }
      } else {
        if (K <= 128) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_f;
        } else if (K <= 256) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        } else if (K <= 512) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_4_4_1_9_t;
        } else if (K <= 1024) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        } else if (K <= 2048) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_4_4_1_9_t;
        } else if (K <= 8192) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        } else {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_t;
        }
      }
    } else {
      if (N <= 128) {
        if (K <= 128) {
          return bf16bf16bf16_grouped_wgrad_128_32_128_1_4_1_9_f;
        } else if (K <= 256) {
          return bf16bf16bf16_grouped_wgrad_128_32_128_1_2_1_9_f;
        } else if (K <= 512) {
          return bf16bf16bf16_grouped_wgrad_128_64_128_1_1_1_9_f;
        } else if (K <= 1024) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_1_1_9_f;
        } else if (K <= 2048) {
          return bf16bf16bf16_grouped_wgrad_128_64_128_1_2_1_9_f;
        } else if (K <= 4096) {
          return bf16bf16bf16_grouped_wgrad_128_64_128_1_1_1_9_f;
        } else {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        }
      } else if (N <= 256) {
        if (K <= 128) {
          return bf16bf16bf16_grouped_wgrad_128_32_128_1_2_1_9_f;
        } else if (K <= 256) {
          return bf16bf16bf16_grouped_wgrad_128_64_128_1_2_1_9_f;
        } else if (K <= 512) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_f;
        } else if (K <= 1024) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_1_1_9_f;
        } else if (K <= 2048) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_2_1_9_t;
        } else {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        }
      } else if (N <= 512) {
        if (K <= 128) {
          return bf16bf16bf16_grouped_wgrad_128_64_128_1_2_1_9_f;
        } else if (K <= 256) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_f;
        } else if (K <= 512) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_1_1_9_t;
        } else {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        }
      } else if (N <= 1024) {
        if (K <= 128) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_1_1_9_f;
        } else if (K <= 256) {
          return bf16bf16bf16_grouped_wgrad_128_64_128_1_2_1_9_f;
        } else {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        }
      } else if (N <= 2048) {
        if (K <= 128) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_f;
        } else if (K <= 4096) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        } else if (K <= 8192) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_t;
        } else {
          return bf16bf16bf16_grouped_wgrad_128_128_128_4_4_1_9_t;
        }
      } else if (N <= 4096) {
        if (K <= 128) {
          return bf16bf16bf16_grouped_wgrad_128_64_128_1_1_1_9_f;
        } else if (K <= 256) {
          return bf16bf16bf16_grouped_wgrad_128_64_128_1_2_1_9_f;
        } else if (K <= 8192) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        } else {
          return bf16bf16bf16_grouped_wgrad_128_128_128_4_4_1_9_t;
        }
      } else if (N <= 8192) {
        if (K <= 128) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_f;
        } else if (K <= 2048) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        } else if (K <= 4096) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_t;
        } else if (K <= 8192) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        } else {
          return bf16bf16bf16_grouped_wgrad_128_128_128_4_4_1_9_t;
        }
      } else {
        if (K <= 128) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_2_1_1_9_f;
        } else if (K <= 256) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        } else if (K <= 2048) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_4_1_9_t;
        } else if (K <= 4096) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_2_1_9_t;
        } else if (K <= 8192) {
          return bf16bf16bf16_grouped_wgrad_128_128_128_4_4_1_9_t;
        } else {
          return bf16bf16bf16_grouped_wgrad_128_128_128_1_4_1_9_t;
        }
      }
    }
  }
}

Kernel_bf16bf16bf16_grouped_wgrad get_kernel_via_tuning(
    int arch,
    int G,
    int total_M,
    int N,
    int K,
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor M_sizes,
    at::Tensor output,
    bool output_accum) {
  auto& cache = getTuningCache();

  // Reducing amount of auto tuning by rounding up total_m to next power of 2.
  total_M = nextPowerOf2(total_M);
  // Use (total_M, N, K, G) shape as the key.
  const std::string shape_key = std::to_string(total_M) + "_" +
      std::to_string(N) + "_" + std::to_string(K) + "_" + std::to_string(G);
  const auto& kernels = get_bf16bf16bf16_grouped_wgrad_kernels(arch);
  auto kernel = cache.findBestKernelMaybeAutotune(
      shape_key, kernels, X, W, M_sizes, output, output_accum);

  return kernel;
}

// BF16 grouped cutlass kernel dispatch.
at::Tensor dispatch_bf16_grouped_kernel(
    int G,
    int total_M,
    int N,
    int K,
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor M_sizes,
    at::Tensor output,
    bool output_accum) {
  const int arch = getDeviceArch();

  // Select kernel to run via heuristics or tuning.
  auto kernel = [&]() {
    if (std::getenv("FBGEMM_AUTOTUNE_ENABLE")) {
      return get_kernel_via_tuning(
          arch, G, total_M, N, K, X, W, M_sizes, output, output_accum);
    } else {
      return get_wgrad_kernel_via_heuristic(arch, G, total_M, N, K);
    }
  }();
  // Invoke kernel
  return kernel(X, W, M_sizes, output, output_accum);
}

at::Tensor bf16bf16bf16_grouped_wgrad(
    at::Tensor X,
    at::Tensor W,
    at::Tensor M_sizes,
    std::optional<at::Tensor> output,
    bool output_accum) {
  int64_t total_M = X.size(0);
  int64_t N = X.size(1);
  int64_t K = W.size(1);
  int64_t G = M_sizes.size(0);
  TORCH_CHECK(
      M_sizes.device() == X.device(),
      "M_sizes must be on same device as inputs.");
  TORCH_CHECK(
      X.dim() == 2 && W.dim() == 2 && W.size(0) == total_M,
      "Activations should be shape [GM, N] and weights should be shape [GM, K]")

  if (output_accum) {
    TORCH_CHECK(
        output.has_value(), "Must provide output tensor for output_accum=True");
  }

  at::Tensor Y;
  if (output.has_value()) {
    Y = output.value();
    if (output_accum) {
      TORCH_CHECK(
          Y.dtype() == at::kFloat,
          "Output tensor must be Float32 when output_accum=True");
    } else {
      TORCH_CHECK(
          Y.dtype() == at::kBFloat16,
          "Output tensor must be BFloat16 when output_accum=False");
    }
  } else {
    Y = at::empty(G * N * K, X.options().dtype(at::kBFloat16));
  }

  // Early exit for empty inputs.
  if (total_M == 0) {
    return Y.view({G, N, K});
  }
  // Return continuous view of output.
  at::Tensor out = dispatch_bf16_grouped_kernel(
      G, total_M, N, K, X, W, M_sizes, Y, output_accum);
  return out.view({G, N, K});
}

#else

at::Tensor bf16bf16bf16_grouped_wgrad(
    at::Tensor,
    at::Tensor,
    at::Tensor,
    std::optional<at::Tensor>,
    bool) {
  throw std::runtime_error(
      "CUDA version is older than 12.0"); // requires CUDA>=12
}

#endif

at::Tensor bf16bf16bf16_grouped_wgrad_meta(
    at::Tensor X,
    at::Tensor W,
    at::Tensor M_sizes,
    std::optional<at::Tensor> /* output = std::nullopt */,
    bool /* output_accum = false */) {
  const at::SymInt G = M_sizes.size(0);
  const at::SymInt N = X.sym_size(1);
  const at::SymInt K = W.sym_size(1);
  at::Tensor Y = at::empty_symint({G, N, K}, X.options().dtype(at::kBFloat16));
  return Y;
}

TORCH_LIBRARY_IMPL(fbgemm, CUDA, m) {
  m.impl("bf16bf16bf16_grouped_wgrad", bf16bf16bf16_grouped_wgrad);
}

TORCH_LIBRARY_IMPL(fbgemm, Meta, m) {
  m.impl("bf16bf16bf16_grouped_wgrad", bf16bf16bf16_grouped_wgrad_meta);
}

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.def(
      "bf16bf16bf16_grouped_wgrad(Tensor X, Tensor W, Tensor M_sizes, Tensor(a!)? output=None, bool output_accum=False) -> Tensor");
}

} // namespace fbgemm_gpu
