/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#if !defined(FBGEMM_API)
  #if defined(FBGEMM_STATIC)
    #define FBGEMM_API
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
  #else
    #if __clang__ || __GNUC__ >=4 || __INTEL_COMPILER
      #define FBGEMM_API __attribute__((__visibility__("default")))
    #else
      #define FBGEMM_API
    #endif
  #endif
#endif

// Use this to indicate to not inline functions
#if __clang__ || __GNUC__ >= 4 || __INTEL_COMPILER
#define NOINLINE __attribute__((noinline))
#elif _MSC_VER
#define NOINLINE __declspec(noinline)
#else
#define NOINLINE
#endif

// Use this to indicate always inline functions
#if __clang__ || __GNUC__ >= 4 || __INTEL_COMPILER
#define ALWAYS_INLINE inline __attribute__((__always_inline__))
#elif _MSC_VER
#define ALWAYS_INLINE __forceinline
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
