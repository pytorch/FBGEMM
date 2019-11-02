/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <asmjit/asmjit.h>

#include <cstdint>
#include <functional>

#include "FbgemmBuild.h"

namespace fbgemm {

namespace internal {

/**
 * Map ACC_T (type used for accumulation) to the types of A and B
 */
template <typename ACC_T>
class SpMMTypeTrait {};

template <>
class SpMMTypeTrait<float> {
 public:
  using a_type = float;
  using b_type = float;
  // Type used to store values in the data section
  using a_temp_type = float;

  using microkernel_function_type =
      void (*)(b_type const*, float*, std::uint64_t);
};

template <>
class SpMMTypeTrait<std::int32_t> {
 public:
  using a_type = std::int8_t;
  using b_type = std::uint8_t;
  // Type used to store values in the data section
  using a_temp_type = std::uint32_t;

  using microkernel_function_type =
      void (*)(b_type const*, std::int32_t*, std::uint64_t);
};

} // namespace internal

/**
 * NOTE: this is an experimental feature
 *
 * Generate a kernel that computes C = A * B with specialization for sparse
 * matrix A's structure and values.
 * @params flags 1 means we're accumulating to CData in the generated kernel.
 *
 * Note on matrix order and layout:
 *   Unlike other fbgemm functions that follow PyTorch convention where A
 * matrix is activation (so in uint8_t for quantized FC/Conv) and B matrix is
 * weight (so in int8_t for quantized FC/Conv), here A is weight matrix (so in
 * int8_t can be seen from SpMMTypeTrait<int32_t>. This is because we mostly
 * target sparsity in weights and for row-major layout it's more efficient to
 * have A as a sparse matrix: for each non-zero of A at ith row and kth column,
 * we can access kth row of B, whose elements are contiguous in memory. If B
 * matrix was sparse, for each non-zero of B at kth row and jth column, we
 * would've needed to access kth column of A, whose elements are not contiguous
 * in memory with C/C++'s row-major layout.
 *   Alternatively, we can call this function as if we're computing
 * C^T = B^T * A^T while maintaining PyTorch's convention that the lefthand
 * side matrix B is activation. If B matrix is in column-major layout, we don't
 * need to do an extra transposition. The C matrix will be output in
 * column-major layout, so if we have a back-to-back SpMMs, B matrices of
 * subsequent matrices will be already in column-major layout. Refer to
 * SpMMFP32Test.cc for an example.
 *   The generated kernel assumes, for ACC_T == int32_t case, AcBr % 4 == 0 and
 * B should be in C/4 N c4 layout where 4 input channels are interleaved
 * (because of 4-way horizontal reduction in x86 SIMD int8 instructions).
 * Refer to SpMMI8Test.cc for an example.
 *
 * Note on specialization for n.
 *   When A matrix is weight and B is activation, n corresponds to batch
 * dimension, which may not be known beforehand. If this turns out to be an
 * issue, we can have another JIT kernel that doesn't need to know n dimension
 * beforehand.
 */
template <typename ACC_T>
FBGEMM_API std::function<void(
    const typename internal::SpMMTypeTrait<ACC_T>::b_type* BData,
    ACC_T* CData,
    std::uint64_t flags)>
generateSpMM(
    int m,
    int n,
    int k,
    const typename internal::SpMMTypeTrait<ACC_T>::a_type* AData,
    int lda,
    int ldb,
    int ldc);

} // namespace fbgemm
