/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "./Utils.h" // @manual

#include <asmjit/core.h> // @manual
#include <asmjit/x86.h> // @manual

namespace fbgemm {

#if ASMJIT_LIBRARY_VERSION >= ASMJIT_LIBRARY_MAKE_VERSION(1, 17, 0)
//! 128-bit XMM register (SSE+).
class Xmm : public asmjit::x86::Vec {
 public:
  using Vec::Vec;
  using Vec::operator=;
  Xmm(uint32_t regId) : Vec(asmjit::x86::Vec::make_xmm(regId)) {}
  //! Casts this register to a register that has half the size (XMM).
  ASMJIT_INLINE_NODEBUG Xmm half() const noexcept {
    return Xmm(id());
  }
};

//! 256-bit YMM register (AVX+).
class Ymm : public asmjit::x86::Vec {
 public:
  using Vec::Vec;
  using Vec::operator=;
  Ymm(uint32_t regId) : Vec(asmjit::x86::Vec::make_ymm(regId)) {}
  //! Casts this register to a register that has half the size (XMM).
  ASMJIT_INLINE_NODEBUG Xmm half() const noexcept {
    return Xmm(id());
  }
};

//! 512-bit ZMM register (AVX512+).
class Zmm : public asmjit::x86::Vec {
 public:
  using Vec::Vec;
  using Vec::operator=;
  Zmm(uint32_t regId) : Vec(asmjit::x86::Vec::make_zmm(regId)) {}
  //! Casts this register to a register that has half the size (YMM).
  ASMJIT_INLINE_NODEBUG Ymm half() const noexcept {
    return Ymm(id());
  }
};
#else
using Xmm = asmjit::x86::Xmm;
using Ymm = asmjit::x86::Ymm;
using Zmm = asmjit::x86::Zmm;
#endif

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

  using vec_reg_t = Ymm;
};

template <>
struct simd_info<inst_set_t::sve> {
  // Implementation is unrolled to match params used on avx2
  static constexpr int WIDTH_BITS = 256;
  static constexpr int WIDTH_BYTES = 32;
  static constexpr int WIDTH_32BIT_ELEMS = 8;
  static constexpr int NUM_VEC_REGS = 32;
};

template <>
struct simd_info<inst_set_t::avx512> {
  static constexpr int WIDTH_BITS = 512;
  static constexpr int WIDTH_BYTES = 64;
  static constexpr int WIDTH_32BIT_ELEMS = 16;
  static constexpr int NUM_VEC_REGS = 32;

  using vec_reg_t = Zmm;
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

  using vec_reg_t = Ymm;
};

template <>
struct simd_info<inst_set_t::avx512_vnni_ymm>
    : public simd_info<inst_set_t::avx512_ymm> {};

} // namespace fbgemm
