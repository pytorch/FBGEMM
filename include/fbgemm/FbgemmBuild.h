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

// Use this to indicate to not inline functions
#if __clang__ || __GNUC__ || __INTEL_COMPILER
#define NOINLINE __attribute__((noinline))
#elif _MSC_VER
#define NOINLINE __declspec(noinline)
#else
#define NOINLINE
#endif

// Use this to indicate always inline functions
#if __clang__ || __GNUC__ || __INTEL_COMPILER
#define ALWAYS_INLINE inline __attribute__((__always_inline__))
#elif _MSC_VER
// commenting out because __forceinline takes too long time in MSVC
#define ALWAYS_INLINE // __forceinline
#else
#define ALWAYS_INLINE inline
#endif

// Use the C++11 keyword "alignas" if you can
#if _MSC_VER
#define ALIGNAS(byte_alignment) __declspec(align(byte_alignment))
#else
#define ALIGNAS(byte_alignment) __attribute__((aligned(byte_alignment)))
#endif

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
