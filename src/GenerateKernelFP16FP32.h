/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "fbgemm/FbgemmFPCommon.h"
#include "fbgemm/Utils.h"

namespace fbgemm {

/// Generate a JIT-compiled GEMM micro-kernel for the given ISA and data type.
/// Returns a function pointer matching funcptr_t<T>.
/// Thread-safe: uses static local for lazy one-time generation.
template <typename T, inst_set_t instSet>
funcptr_t<T> generateGemmKernel(int kernel_nrows);

/// Build a kernel_array_t by JIT-generating kernels for rows [from, to].
template <typename T, inst_set_t instSet>
kernel_array_t<T> makeKernelArray(int from, int to) {
  kernel_array_t<T> k{};
  for (int n = from; n <= to; n++)
    k[n] = generateGemmKernel<T, instSet>(n);
  return k;
}

} // namespace fbgemm
