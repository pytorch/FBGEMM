/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <c10/macros/Macros.h>
#include <cstdlib>
#include <string_view>

////////////////////////////////////////////////////////////////////////////////
//
// Note: Device Side Assertion (DSA) is currently only supported on CUDA.
// We undefine TORCH_USE_CUDA_DSA for ROCm builds to disable the DSA code path.
//
// TODO: Enable DSA for ROCm after
// https://github.com/pytorch/pytorch/pull/172679 lands
//
////////////////////////////////////////////////////////////////////////////////

#if defined(TORCH_USE_CUDA_DSA) && !defined(USE_ROCM)
#include <c10/cuda/CUDADeviceAssertion.h>
#else
#undef TORCH_USE_CUDA_DSA
#endif

namespace fbgemm_gpu::utils {

////////////////////////////////////////////////////////////////////////////////
//
// FBGEMM_KERNEL_ASSERT Macro
//
// Unlike PyTorch CUDA_KERNEL_ASSERT2, FBGEMM_KERNEL_ASSERT guarantees an error
// on condition failure during kernel execution, regardless of whether
// TORCH_USE_CUDA_DSA is defined.
//
// When TORCH_USE_CUDA_DSA (compile time flag) is defined:
//   If PYTORCH_USE_CUDA_DSA is set at runtime (assertions_data != nullptr):
//     Uses CUDA_KERNEL_ASSERT2 for Device Side Assertion support with
//     detailed error reporting.
//   If PYTORCH_USE_CUDA_DSA is NOT set at runtime (assertions_data == nullptr):
//     Falls back to CUDA_KERNEL_ASSERT.
//
// When TORCH_USE_CUDA_DSA is NOT defined:
//   Falls back to CUDA_KERNEL_ASSERT.
//
////////////////////////////////////////////////////////////////////////////////

#ifdef TORCH_USE_CUDA_DSA

#define FBGEMM_KERNEL_ASSERT(condition)                                \
  do {                                                                 \
    if (C10_UNLIKELY(!(condition))) {                                  \
      if (assertions_data) {                                           \
        /* Runtime DSA is enabled, use CUDA_KERNEL_ASSERT2 behavior */ \
        c10::cuda::dsa_add_new_assertion_failure(                      \
            assertions_data,                                           \
            C10_STRINGIZE(condition),                                  \
            __FILE__,                                                  \
            __FUNCTION__,                                              \
            __LINE__,                                                  \
            assertion_caller_id,                                       \
            blockIdx,                                                  \
            threadIdx);                                                \
        return;                                                        \
      } else {                                                         \
        /* Runtime DSA is disabled, fall back to printf + trap */      \
        CUDA_KERNEL_ASSERT(condition);                                 \
      }                                                                \
    }                                                                  \
  } while (0)

#else

#define FBGEMM_KERNEL_ASSERT(condition) CUDA_KERNEL_ASSERT(condition)

#endif // TORCH_USE_CUDA_DSA

inline bool isPytorchDsaEnabled() {
  static auto result = [] {
    const auto* env_val = std::getenv("PYTORCH_USE_CUDA_DSA");
    return env_val != nullptr && std::string_view(env_val) == "1";
  }();
  return result;
}

} // namespace fbgemm_gpu::utils
