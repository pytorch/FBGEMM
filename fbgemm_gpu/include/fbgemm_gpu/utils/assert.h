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
// Helper Macro: Abort kernel with message
//
// ROCm disables kernel assert by default for performance considerations. Though
// ROCm supports __assert_fail, it uses kernel printf which has a non-negligible
// performance impact even if the assert condition is never triggered. We choose
// to use abort() instead which will still terminate the application.
//
////////////////////////////////////////////////////////////////////////////////

#if defined(USE_ROCM)
#define __FBGEMM_KERNEL_ABORT(message) \
  do {                                 \
    abort();                           \
  } while (0)
#else // CUDA
#define __FBGEMM_KERNEL_ABORT(message)                                     \
  do {                                                                     \
    __assert_fail(                                                         \
        message, __FILE__, static_cast<unsigned int>(__LINE__), __func__); \
  } while (0)
#endif

////////////////////////////////////////////////////////////////////////////////
//
// Helper Macro: Prints assertion failure message (if print_error is true) and
// traps.
//
// This is used as a fallback when DSA is not available or disabled at
// runtime.
//
////////////////////////////////////////////////////////////////////////////////

#define __FBGEMM_KERNEL_ASSERT_FAIL(print_error, condition_str) \
  do {                                                          \
    if constexpr (print_error) {                                \
      printf(                                                   \
          "[FBGEMM] %s:%d ASSERT FAILED: `%s` "                 \
          "block: [%u,%u,%u], thread: [%u,%u,%u]\n",            \
          __FILE__,                                             \
          __LINE__,                                             \
          condition_str,                                        \
          blockIdx.x,                                           \
          blockIdx.y,                                           \
          blockIdx.z,                                           \
          threadIdx.x,                                          \
          threadIdx.y,                                          \
          threadIdx.z);                                         \
    }                                                           \
    __FBGEMM_KERNEL_ABORT(condition_str);                       \
  } while (0)

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
//     Falls back to printf + trap to guarantee the error is visible.
//
// When TORCH_USE_CUDA_DSA is NOT defined:
//   - CUDA: Uses __assert_fail() which prints the condition and terminates.
//   - ROCm: Uses printf + __trap() to guarantee the condition is printed.
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
        __FBGEMM_KERNEL_ASSERT_FAIL(true, #condition);                 \
      }                                                                \
    }                                                                  \
  } while (0)

#else

#define FBGEMM_KERNEL_ASSERT(condition)               \
  do {                                                \
    if (C10_UNLIKELY(!(condition))) {                 \
      __FBGEMM_KERNEL_ASSERT_FAIL(false, #condition); \
    }                                                 \
  } while (0)

#endif // TORCH_USE_CUDA_DSA

inline bool isPytorchDsaEnabled() {
  static auto result = [] {
    const auto* env_val = std::getenv("PYTORCH_USE_CUDA_DSA");
    return env_val != nullptr && std::string_view(env_val) == "1";
  }();
  return result;
}

} // namespace fbgemm_gpu::utils
