/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cuda.h>

////////////////////////////////////////////////////////////////////////////////
// Kernel Barrier Isolation
//
// The kernel barrier isolation macro is a performance profiling tool that
// isolates kernel execution from other GPU processes that might otherwise have
// been running concurrently.  This is used in conjunction with trace inspection
// to determine whether a kernel's regression might be due to other GPU
// processes competing for memory bandwidth that is causing the kernel slowdown,
// which can be especially relevant when data accessed by the kernel is in UVM.
////////////////////////////////////////////////////////////////////////////////

#ifdef FBGEMM_GPU_KERNEL_DEBUG

#define DEBUG_KERNEL_BARRIER_ISOLATE(...) \
  do {                                    \
    cudaDeviceSynchronize();              \
    __VA_ARGS__();                        \
    cudaDeviceSynchronize();              \
  } while (0);

#else

#define DEBUG_KERNEL_BARRIER_ISOLATE(...) \
  do {                                    \
    __VA_ARGS__();                        \
  } while (0);

#endif
