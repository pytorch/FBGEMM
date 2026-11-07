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
/// Each call generates a new kernel; installation in the JIT runtime is
/// serialized.
template <typename T, inst_set_t instSet>
FBGEMM_API funcptr_t<T> generateGemmKernel(int kernel_nrows);

/// Generate each ISA's dispatch table on first use.
template <typename T, inst_set_t instSet>
const kernel_array_t<T>& getGemmKernelArray() {
  static const kernel_array_t<T> kernels = [] {
    kernel_array_t<T> result{};
    int first_row = 1;
    if constexpr (instSet == inst_set_t::avx512_ymm) {
      result = getGemmKernelArray<T, inst_set_t::avx2>();
      first_row = 7;
    }
    constexpr int last_row = instSet == inst_set_t::avx2 ? 6 : 14;
    for (int row = first_row; row <= last_row; ++row) {
      result[row] = generateGemmKernel<T, instSet>(row);
    }
    return result;
  }();
  return kernels;
}

}  // namespace fbgemm
