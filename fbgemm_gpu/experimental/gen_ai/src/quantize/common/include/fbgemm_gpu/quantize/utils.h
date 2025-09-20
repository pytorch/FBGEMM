/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <climits>
#include <cstdint>

#include <ATen/cuda/CUDAContext.h>

namespace fbgemm_gpu {

constexpr int64_t nextPowerOf2(int64_t num) {
  if (num <= 1)
    return 1;
  return 1 << (CHAR_BIT * sizeof(num) - __builtin_clz(num - 1));
}

inline int getDeviceArch() {
  static int arch = []() {
    const int majorVersion =
        at::cuda::getDeviceProperties(at::cuda::current_device())->major;
    if (majorVersion >= 10) {
      int runtimeVersion = 0;
      C10_CUDA_CHECK(cudaRuntimeGetVersion(&runtimeVersion));
      TORCH_CHECK(
          runtimeVersion >= 12080, "SM100a+ kernels require cuda >= 12.8");
    }
    return majorVersion;
  }();
  return arch;
}

} // namespace fbgemm_gpu
