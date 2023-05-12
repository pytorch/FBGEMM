/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "./Utils.h"

#include <asmjit/asmjit.h>

namespace fbgemm {
/**
 * @brief Some commonly used variables for different instruction sets
 */
template <inst_set_t inst_set>
struct simd_info;

template <>
struct simd_info<inst_set_t::avx2> {
  static constexpr int WIDTH_BITS = 256;
  static constexpr int WIDTH_BYTES = 32;
  static constexpr int WIDTH_32BIT_ELEMS = 8;
  static constexpr int NUM_VEC_REGS = 16;

  using vec_reg_t = asmjit::x86::Ymm;
};

template <>
struct simd_info<inst_set_t::avx512> {
  static constexpr int WIDTH_BITS = 512;
  static constexpr int WIDTH_BYTES = 64;
  static constexpr int WIDTH_32BIT_ELEMS = 16;
  static constexpr int NUM_VEC_REGS = 32;

  using vec_reg_t = asmjit::x86::Zmm;
};

template <>
struct simd_info<inst_set_t::avx512_vnni>
    : public simd_info<inst_set_t::avx512> {};

template <>
struct simd_info<inst_set_t::avx512_ymm> {
  static constexpr int WIDTH_BITS = 256;
  static constexpr int WIDTH_BYTES = 32;
  static constexpr int WIDTH_32BIT_ELEMS = 8;
  static constexpr int NUM_VEC_REGS = 32;

  using vec_reg_t = asmjit::x86::Ymm;
};

template <>
struct simd_info<inst_set_t::avx512_vnni_ymm>
    : public simd_info<inst_set_t::avx512_ymm> {};

} // namespace fbgemm
