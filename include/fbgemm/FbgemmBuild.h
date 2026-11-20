/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// For details about dllexport/dllimport, checkout the following SO question
// https://stackoverflow.com/questions/57999/what-is-the-difference-between-dllexport-and-dllimport
#if !defined(FBGEMM_API)
#if defined(FBGEMM_STATIC)
#define FBGEMM_API
#define FBGEMM_ENUM_CLASS_API
#elif defined _WIN32 || defined __CYGWIN__
#if (__GNUC__ || __clang__) && !(__MINGW64__ || __MINGW32__)
#if defined(FBGEMM_EXPORTS)
#define FBGEMM_API __attribute__((__dllexport__))
#else
#define FBGEMM_API __attribute__((__dllimport__))
#endif
#else
#if defined(FBGEMM_EXPORTS)
#define FBGEMM_API __declspec(dllexport)
#else
#define FBGEMM_API __declspec(dllimport)
#endif
#endif
#define FBGEMM_ENUM_CLASS_API
#else
#if __clang__ || __GNUC__ || __INTEL_COMPILER
#define FBGEMM_API __attribute__((__visibility__("default")))
#else
#define FBGEMM_API
#endif
// Currently, enum classes need to be declaredly explicitly for shared build on
// macos
#if __clang__
#define FBGEMM_ENUM_CLASS_API __attribute__((__visibility__("default")))
#else
#define FBGEMM_ENUM_CLASS_API
#endif
#endif
#endif

// Defined on x86_64 when the F16C VCVTPH2PS / VCVTPS2PH instructions are
// available.
// - GCC/Clang: __F16C__ is defined by -mf16c or -march=haswell+
// - MSVC: never defines __F16C__, but /arch:AVX2 implies F16C hardware
#if !defined(__CUDACC__) && \
    (defined(__F16C__) || (defined(_MSC_VER) && defined(__AVX2__)))
#define FBGEMM_HAS_X86F16C
#endif

// _Float16 with hardware float <-> _Float16 conversions. clang only: GCC and
// MSVC don't vectorize loops over these conversions.
#if defined(__clang__) && defined(__FLT16_MANT_DIG__) && \
    (defined(__aarch64__) || defined(FBGEMM_HAS_X86F16C))
#define FBGEMM_HAS_FLOAT16
#endif

// Namespace helpers for the runtime SIMD dispatch mechanism.
// FBGEMM_SIMD_NAMESPACE - namespace used for the platform kernels.
// FBGEMM_ABI_NAMESPACE - inline namespace for common code we want to use in
// platform specific code.
// Both of these help avoid ODR violations.
#ifdef FBGEMM_SIMD_AVX512_BF16
#define FBGEMM_SIMD_NAMESPACE avx512_bf16
#define FBGEMM_ABI_NAMESPACE abi_avx512_bf16
#elif defined(FBGEMM_SIMD_AVX512)
#define FBGEMM_SIMD_NAMESPACE avx512
#define FBGEMM_ABI_NAMESPACE abi_avx512
#elif defined(FBGEMM_SIMD_AVX512_256)
#define FBGEMM_SIMD_NAMESPACE avx512_256
#define FBGEMM_ABI_NAMESPACE abi_avx512_256
#elif defined(FBGEMM_SIMD_AVX512_256_BF16)
#define FBGEMM_SIMD_NAMESPACE avx512_256_bf16
#define FBGEMM_ABI_NAMESPACE abi_avx512_256_bf16
#elif defined(FBGEMM_SIMD_AVX2)
#define FBGEMM_SIMD_NAMESPACE avx2
#define FBGEMM_ABI_NAMESPACE abi_avx2
#elif defined(FBGEMM_SIMD_SVE128)
#define FBGEMM_SIMD_NAMESPACE sve128
#define FBGEMM_ABI_NAMESPACE abi_sve128
#elif defined(FBGEMM_SIMD_NEON)
#define FBGEMM_SIMD_NAMESPACE neon
#define FBGEMM_ABI_NAMESPACE abi_neon
#elif defined(FBGEMM_SIMD_REF)
#define FBGEMM_SIMD_NAMESPACE ref
#define FBGEMM_ABI_NAMESPACE abi_ref
#else
#define FBGEMM_ABI_NAMESPACE abi_generic
#endif

// Use this to indicate to not inline functions
#if __clang__ || __GNUC__ || __INTEL_COMPILER
#define NOINLINE __attribute__((noinline))
#elif defined(_MSC_VER) && _MSC_VER
#define NOINLINE __declspec(noinline)
#else
#define NOINLINE
#endif

// Use this to indicate always inline functions
#if __clang__ || __GNUC__ || __INTEL_COMPILER
#define ALWAYS_INLINE inline __attribute__((__always_inline__))
#elif defined(_MSC_VER) && _MSC_VER
// commenting out because __forceinline takes too long time in MSVC
#define ALWAYS_INLINE // __forceinline
#else
#define ALWAYS_INLINE inline
#endif

// alignas is standard since C++11
#define ALIGNAS(byte_alignment) alignas(byte_alignment)

// Sanitizers annotations
#if defined(__has_attribute)
#if __has_attribute(no_sanitize)
#define NO_SANITIZE(what) __attribute__((no_sanitize(what)))
#endif
#endif
#if !defined(NO_SANITIZE)
#define NO_SANITIZE(what)
#endif

// Ignore __builtin_assume() when not supported by compiler.
#ifndef __has_builtin
#define __has_builtin(x) 0
#endif
#if !__has_builtin(__builtin_assume)
#define __builtin_assume(x) (static_cast<void>(0))
#endif

// Macro for silencing warnings
#if __clang__ || __GNUC__
// clang-format off
#define FBGEMM_PUSH_WARNING _Pragma("GCC diagnostic push")
#define FBGEMM_DISABLE_WARNING_INTERNAL2(warningName) #warningName
#define FBGEMM_DISABLE_WARNING(warningName) \
  _Pragma(                                     \
      FBGEMM_DISABLE_WARNING_INTERNAL2(GCC diagnostic ignored warningName))
#define FBGEMM_PUSH_WARNING_AND_DISABLE(warningName) \
  _Pragma("GCC diagnostic push") \
  _Pragma(                                     \
      FBGEMM_DISABLE_WARNING_INTERNAL2(GCC diagnostic ignored warningName))
#define FBGEMM_POP_WARNING _Pragma("GCC diagnostic pop")
// clang-format on
#else
#define FBGEMM_PUSH_WARNING
#define FBGEMM_DISABLE_WARNING(NAME)
#define FBGEMM_PUSH_WARNING_AND_DISABLE(NAME)
#define FBGEMM_POP_WARNING
#endif
