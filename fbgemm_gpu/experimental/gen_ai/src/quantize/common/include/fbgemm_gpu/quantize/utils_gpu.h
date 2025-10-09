/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/cuda/CUDAContext.h>

namespace fbgemm_gpu {

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

inline int64_t getSMCount(int device_index, std::optional<int64_t> num_sms) {
  if (num_sms.has_value()) {
    return num_sms.value();
  }

  static int64_t cached_sm_count = []() {
    int64_t sm_count = at::cuda::getDeviceProperties(0)->multiProcessorCount;
    if (at::globalContext()._SMCarveout_EXPERIMENTAL().has_value()) {
      sm_count -= at::globalContext()._SMCarveout_EXPERIMENTAL().value();
    }
    return sm_count;
  }();
  return cached_sm_count;
}

} // namespace fbgemm_gpu
