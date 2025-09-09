/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "fbgemm_gpu/quantize/utils.h" // @manual

#include <ATen/ATen.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

namespace fbgemm_gpu {

int getDeviceArch() {
  static int arch = []() {
    // Avoid expensive cudaGetDeviceProperties call.
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    if (prop.major >= 10) {
      int runtimeVersion = 0;
      C10_CUDA_CHECK(cudaRuntimeGetVersion(&runtimeVersion));
      TORCH_CHECK(
          runtimeVersion >= 12080, "SM100a+ kernels require cuda >= 12.8");
    }

    return prop.major;
  }();
  return arch;
}
} // namespace fbgemm_gpu
