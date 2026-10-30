/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#define FBGEMM_EXPORTS
#include "fbgemm/FbgemmConvert.h"
#include "fbgemm/Utils.h"

#include <algorithm>
#include <bit>
#include <stdexcept>

#include <cpuinfo.h>

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
double naive_malloc_time = 0.0;
double naive_A_bf16_to_fp32_time = 0.0;
double naive_B_bf16_to_fp32_time = 0.0;
double naive_C_bf16_to_fp32_time = 0.0;
double naive_computing_time = 0.0;
double naive_C_fp32_to_bf16_time = 0.0;
double naive_run_time = 0.0;
#endif

namespace fbgemm {

void FloatToBfloat16_simd(const float* src, bfloat16* dst, size_t size) {
  // Run time CPU detection
  if (cpuinfo_initialize()) {
#if !defined(__aarch64__)
    const bool has_avx512 = fbgemmHasAvx512Support();
#endif
#if defined(FBGEMM_HAVE_NATIVE_BF16) && \
    (defined(__aarch64__) || defined(__x86_64__) || defined(_M_X64))
#if defined(__aarch64__)
    const bool native_bf16 = cpuinfo_has_arm_bf16();
#else
    const bool native_bf16 = has_avx512 && fbgemmHasAvx512Bf16Support();
#endif
    if (native_bf16) {
      [](const float* input, bfloat16* output, size_t length)
#if defined(__aarch64__)
          __attribute__((target("+bf16")))
#else
          __attribute__((target("avx512bf16,avx512vl")))
#endif
      {
#if defined(__aarch64__)
        uint64_t fpcr;
        asm volatile("mrs %0, fpcr" : "=r"(fpcr));
        // Fall back for nondefault rounding/denormal handling or FP traps.
        if (fpcr & ((3ull << 22) | (1ull << 24) | 0x9f03ull)) {
          FloatToBfloat16_ref(input, output, length);
          return;
        }
#endif
        for (size_t i = 0; i < length;) {
          const size_t count = std::min(length - i, size_t{32});
          bool needs_reference = false;
          for (size_t j = 0; j < count; ++j) {
            const uint32_t bits = std::bit_cast<uint32_t>(input[i + j]) & 0x7fffffffu;
            needs_reference |= bits > 0x7f800000u ||
                (bits != 0 && bits < 0x00800000u);
          }
          // Preserve NaN behavior and subnormals, which x86 BF16 flushes.
          if (needs_reference) {
            FloatToBfloat16_ref(input + i, output + i, count);
          } else {
            for (size_t j = 0; j < count; ++j) {
              output[i + j].val =
                  std::bit_cast<uint16_t>(static_cast<__bf16>(input[i + j]));
            }
          }
          i += count;
        }
      }(src, dst, size);
      return;
    }
#endif
#if !defined(__aarch64__)
    if (has_avx512) {
      FloatToBfloat16_avx512(src, dst, size);
    } else if (fbgemmHasAvx2Support()) {
      FloatToBfloat16_avx2(src, dst, size);
    } else
#endif
    {
      FloatToBfloat16_ref(src, dst, size);
      return;
    }
  } else {
    throw std::runtime_error("Failed to initialize cpuinfo!");
  }
}

void Bfloat16ToFloat_simd(const bfloat16* src, float* dst, size_t size) {
  // Run time CPU detection
  if (cpuinfo_initialize()) {
#if !defined(__aarch64__)
    if (fbgemmHasAvx512Support()) {
      Bfloat16ToFloat_avx512(src, dst, size);
    } else if (fbgemmHasAvx2Support()) {
      Bfloat16ToFloat_avx2(src, dst, size);
    } else
#endif
    {
      Bfloat16ToFloat_ref(src, dst, size);
      return;
    }
  } else {
    throw std::runtime_error("Failed to initialize cpuinfo!");
  }
}

} // namespace fbgemm
