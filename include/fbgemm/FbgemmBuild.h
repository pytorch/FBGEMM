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
