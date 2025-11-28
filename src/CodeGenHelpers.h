/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <asmjit/x86.h> // @manual
#include "fbgemm/SimdUtils.h"
#include "fbgemm/Utils.h"

namespace fbgemm {

namespace x86 = asmjit::x86;

class FileLoggerWithClose : public asmjit::FileLogger {
  explicit FileLoggerWithClose(FILE* f)
    : FileLogger(f) {}

  ~FileLoggerWithClose() noexcept override {
    if (_file) {
      fclose(_file);
    }
  }
};

/**
 * @brief Emit an instruction that fills a vector register with
 *        all ones.
 *
 * @param dest Destination register to fill
 */
template <inst_set_t instSet>
void emitVecFillWithOnes(x86::Emitter* a, const x86::Vec& dest) {
  if constexpr (instSet == inst_set_t::avx2) {
    a->vpcmpeqb(dest, dest, dest);
  } else {
    a->vpternlogd(dest, dest, dest, 0xff);
  }
}

/**
 * @brief Create instruction sequence to generate 8-bit 1s
 *
 * @param dest Once the instruction sequence is executed,
 *             dest[0:7] will have 0x01, dest[8:15]
 *             will have 0x01 and so on
 */
template <inst_set_t instSet>
void gen8BitVectorOne(x86::Emitter* a, const x86::Vec& dest) {
  emitVecFillWithOnes<instSet>(a, dest);
  a->vpabsb(dest, dest);
}

/**
 * @brief Create instruction sequence to generate 16-bit 1s
 *
 * @param dest Once the instruction sequence is executed,
 *             dest[0:15] will have 0x0001, dest[16:31]
 *             will have 0x0001 and so on
 */
template <inst_set_t instSet>
void gen16BitVectorOne(x86::Emitter* a, const x86::Vec& dest) {
  emitVecFillWithOnes<instSet>(a, dest);
  a->vpsrlw(dest, dest, 15);
}

/**
 * @brief Emit instruction do load 32-bit integer. AVX512 has
 *        different instruction to load registers with index >= 16
 *
 * @param dest Destination vector register
 */
template <inst_set_t instSet, typename Dest, typename Src>
void emitVecMove(x86::Emitter* a, const Dest& dest, const Src& src) {
  if constexpr (instSet == inst_set_t::avx2) {
    a->vmovdqa(dest, src);
  } else {
    a->vmovdqa32(dest, src);
  }
}

template <inst_set_t instSet, typename Dest, typename Src1, typename Src2>
void emitVecOr(x86::Emitter* a, const Dest& dest, const Src1& src1, const Src2& src2) {
  if constexpr (instSet == inst_set_t::avx2) {
    a->vpor(dest, src1, src2);
  } else {
    a->vpord(dest, src1, src2);
  }
}

template <inst_set_t instSet, typename Dest, typename Src1, typename Src2>
void emitVecXor(x86::Emitter* a, const Dest& dest, const Src1& src1, const Src2& src2) {
  if constexpr (instSet == inst_set_t::avx2) {
    a->vpxor(dest, src1, src2);
  } else {
    a->vpxord(dest, src1, src2);
  }
}

/**
 * @brief Emit partial extract from Wide regiter to Half Register, eg.
 *        Zmm -> Ymm or Ymm -> Xmm
 * @tparam instSet instruction set to be used
 *
 * @param half Destination (half) vector register
 * @param vec Source (full) vector register
 * @param idx Index of of the half vector 0 or 1
 */

template <inst_set_t instSet>
void emitExtractHalfVector(
    x86::Emitter* a,
    const x86::Vec& half,
    const x86::Vec& vec,
    int idx) {
  if constexpr (instSet == inst_set_t::avx2) {
    a->vextracti128(half, vec, idx);
  } else {
    if (vec.is_vec512()) {
      a->vextracti32x8(half, vec, idx);
    } else {
      a->vextracti32x4(half, vec, idx);
    }
  }
}

/**
 * @brief Generates instruction sequence to compute s32 += U8 * I8
 * @tparam T Register type of destination, e.g., Ymm or Zmm
 *
 * @param cReg contains result
 *
 */

template <
    inst_set_t INST_SET,
    std::enable_if_t<
        INST_SET == inst_set_t::avx2 || INST_SET == inst_set_t::avx512,
        int> = 0>
void genU8I8S32FMA(
    x86::Emitter* a,
    const x86::Vec& aReg,
    const x86::Vec& bReg,
    const x86::Vec& cReg,
    const x86::Vec& oneReg16Bit,
    const x86::Vec& tmpReg) {
  a->vpmaddubsw(tmpReg, aReg, bReg);
  a->vpmaddwd(tmpReg, oneReg16Bit, tmpReg);
  a->vpaddd(cReg, tmpReg, cReg);
}

template <
    inst_set_t INST_SET,
    std::enable_if_t<INST_SET == inst_set_t::avx512_vnni, int> = 0>
void genU8I8S32FMA(
    x86::Emitter* a,
    const x86::Vec& aReg,
    const x86::Vec& bReg,
    const x86::Vec& cReg,
    const x86::Vec& /*oneReg16Bit*/,
    const x86::Vec& /*tmpReg*/) {
  a->vpdpbusd(cReg, aReg, bReg);
}

/**
 * @brief Add 4 consecutive numbers of type uint8
 *        and emit their sum as 32-bit numbers.
 *        i.e., dest[0:31] contains
 *        src[0:7] + src[8:15] + src[16:23] + src[24:31]
 * @tparam T Register type of destination, e.g., Ymm or Zmm
 *
 * @param dest contains result
 *
 */
template <
    inst_set_t INST_SET,
    std::enable_if_t<
        INST_SET == inst_set_t::avx2 || INST_SET == inst_set_t::avx512,
        int> = 0>
void genU8Sum4(
    x86::Emitter* a,
    const x86::Vec& src,
    const x86::Vec& dest,
    const x86::Vec& oneReg16Bit,
    const x86::Vec& tmpReg) {
  gen8BitVectorOne<INST_SET>(a, tmpReg);
  a->vpmaddubsw(tmpReg, src, tmpReg);
  a->vpmaddwd(tmpReg, tmpReg, oneReg16Bit);
  a->vpaddd(dest, tmpReg, dest);
  /*a->vxorps(tmpReg, tmpReg, tmpReg);*/
  /*a->vmpsadbw(tmpReg, src, tmpReg, static_cast<asmjit::Imm>(0));*/
  /*a->vpermilps(tmpReg, tmpReg, static_cast<asmjit::Imm>(4));*/
  /*a->vpmovzxwd(tmpReg, tmpReg.half());*/
  /*a->vpaddd(dest, tmpReg, dest);*/
}

template <
    inst_set_t INST_SET,
    std::enable_if_t<INST_SET == inst_set_t::avx512_vnni, int> = 0>
void genU8Sum4(
    x86::Emitter* a,
    const x86::Vec& src,
    const x86::Vec& dest,
    const x86::Vec& /*oneReg16Bit*/,
    const x86::Vec& tmpReg) {
  gen8BitVectorOne<INST_SET>(a, tmpReg);
  a->vpdpbusd(dest, src, tmpReg);
}

/**
 * @brief Add 8 consecutive numbers of type uint8
 *        and emit their sum as 16-bit numbers.
 *        i.e., dest[0:15] contains
 *        src[0:7] + src[8:15] + src[16:23] + src[24:31]
 *        src[32:39] + src[40:47] + src[48:55] + src[56:63]
 *
 *        and
 *
 *        dest[64:79] contains
 *        src[64:71] + src[71:79] + src[80:87] + src[88:95]
 *        src[96:103] + src[104:111] + src[112:119] + src[120:127]
 *
 *        so on
 *
 * @tparam T Register type of destination, e.g., Ymm or Zmm
 *
 * @param dest contains result
 *
 */
template <typename T>
void genU8Sum8(x86::Emitter* a, T src, T dest, T tmpReg) {
  a->vxorps(tmpReg, tmpReg, tmpReg);
  a->vpsadbw(tmpReg, src, tmpReg);
  a->vpaddd(dest, tmpReg, dest);
}

/**
 * @brief Broadcast lower 8-bits of src to destination vector
 *        register.
 */
template <typename T>
void broadcast8Bit(x86::Emitter* a, x86::Gp src, T dest) {
  // move src to dest
  auto xmm = dest.xmm();
  a->vmovq(xmm, src);
  a->vpbroadcastb(dest, xmm);
}

template <inst_set_t instSet>
void emitReduceAddF32(x86::Emitter* a, x86::Vec v, const x86::Vec& tmp) {
  if constexpr (instSet != inst_set_t::avx2) {
    if (v.is_vec512()) {
      a->vextractf32x8(tmp.ymm(), v, 1);
      v = v.ymm();
      a->vaddps(v.ymm(), v, tmp);
    }
  }

  if (v.is_vec256()) {
    if constexpr (instSet == inst_set_t::avx2) {
      a->vextractf128(tmp.xmm(), v, 1);
    } else {
      a->vextractf32x4(tmp.xmm(), v, 1);
    }
    v = v.xmm();
    a->vaddps(v, v, tmp.xmm());
  }

  a->vhaddps(v, v, v);
  a->vhaddps(v, v, v);
}

} // namespace fbgemm
