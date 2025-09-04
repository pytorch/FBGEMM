/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
namespace fbgemm_gpu {

#if (defined(USE_ROCM) && ROCM_VERSION >= 60200) || \
    (defined(CUDA_VERSION) && CUDA_VERSION >= 12000)
class FP8_E4M3_MAX {
 public:
#ifndef USE_ROCM
  static constexpr float value = 448.0;
#else
  static constexpr float value = 240.0;
#endif
};
class FP8_E5M2_MAX {
 public:
  static constexpr float value = 57344.0;
};
#endif

} // namespace fbgemm_gpu
