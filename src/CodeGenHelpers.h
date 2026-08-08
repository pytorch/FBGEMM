/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <asmjit/x86.h> // @manual
#include "AsmjitCompat.h" // @manual
#include "fbgemm/SimdUtils.h"
#include "fbgemm/Utils.h"

namespace fbgemm {

namespace x86 = asmjit::x86;

/// True for any AVX512 variant (avx512, avx512_ymm, avx512_vnni,
/// avx512_vnni_ymm).
template <inst_set_t instSet>
inline constexpr bool is_avx512_any_v = instSet == inst_set_t::avx512 ||
    instSet == inst_set_t::avx512_ymm || instSet == inst_set_t::avx512_vnni ||
    instSet == inst_set_t::avx512_vnni_ymm;

/// True for VNNI variants (avx512_vnni, avx512_vnni_ymm).
template <inst_set_t instSet>
inline constexpr bool is_avx512_vnni_v = instSet == inst_set_t::avx512_vnni ||
    instSet == inst_set_t::avx512_vnni_ymm;

/**
 * @brief Create instruction sequence to generate 16-bit 1s
 * @tparam T Register type of destination, e.g., Ymm or Zmm
 *
 * @param dest Once the instruction sequence is executed,
 *             dest[0:15] will have 0x0001, dest[16:31]
 *             will have 0x0001 and so on
 */
template <inst_set_t instSet, typename T>
void gen16BitVectorOne(x86::Emitter* a, T dest) {
  if constexpr (instSet == inst_set_t::avx2) {
    a->vpcmpeqw(dest, dest, dest);
  } else {
    static_assert(
        is_avx512_any_v<instSet>,
        "unsupported instruction set for gen16BitVectorOne");
    a->vpternlogd(dest, dest, dest, 0xff);
  }
  a->vpsrlw(dest, dest, 15);
}

/**
 * @brief Emit instruction to load 32-bit integer. AVX512 has
 *        different instruction to load registers with index >= 16
 * @tparam T Register type of destination, e.g., Ymm or Zmm
 *
 * @param dest Destination vector register
 */
template <inst_set_t instSet, typename T>
void emitLoadDWord(x86::Emitter* a, T dest, const x86::Mem& ptr) {
  if constexpr (instSet == inst_set_t::avx2) {
    a->vmovdqa(dest, ptr);
  } else {
    static_assert(
        is_avx512_any_v<instSet>,
        "unsupported instruction set for emitLoadDWord");
    a->vmovdqa32(dest, ptr);
  }
}

/**
 * @brief Emit partial extract from Wide register to Half Register, eg.
 *        Zmm -> Ymm or Ymm -> Xmm
 * @tparam instSet instruction set to be used
 *
 * @param half Destination (half) vector register
 * @param vec Source (full) vector register
 * @param idx Index of the half vector 0 or 1
 */
template <inst_set_t instSet>
void emitExtractHalfVector(
    x86::Emitter* a,
    const Ymm& half,
    const Zmm& vec,
    int idx) {
  static_assert(is_avx512_any_v<instSet>, "Zmm->Ymm extract requires AVX512");
  a->vextracti32x8(half, vec, idx);
}

template <inst_set_t instSet>
void emitExtractHalfVector(
    x86::Emitter* a,
    const Xmm& half,
    const Ymm& vec,
    int idx) {
  if constexpr (instSet == inst_set_t::avx2) {
    a->vextracti128(half, vec, idx);
  } else {
    static_assert(
        is_avx512_any_v<instSet>,
        "unsupported instruction set for emitExtractHalfVector");
    a->vextracti32x4(half, vec, idx);
  }
}

/**
 * @brief Create instruction sequence to generate 8-bit 1s
 * @tparam T Register type of destination, e.g., Ymm or Zmm
 *
 * @param dest Once the instruction sequence is executed,
 *             dest[0:7] will have 0x01, dest[8:15]
 *             will have 0x01 and so on
 */
template <typename T>
void gen8BitVectorOne(x86::Emitter* a, T dest) {
  if constexpr (std::is_same_v<T, Ymm>) {
    a->vpcmpeqw(dest, dest, dest);
  } else {
    static_assert(std::is_same_v<T, Zmm>, "T must be Ymm or Zmm");
    a->vpternlogd(dest, dest, dest, 0xff);
  }
  a->vpabsb(dest, dest);
}

/**
 * @brief Generates instruction sequence to compute s32 += U8 * I8
 *
 * @param cReg contains result
 */
template <inst_set_t INST_SET>
void genU8I8S32FMA(
    x86::Emitter* a,
    typename simd_info<INST_SET>::vec_reg_t aReg,
    typename simd_info<INST_SET>::vec_reg_t bReg,
    typename simd_info<INST_SET>::vec_reg_t cReg,
    [[maybe_unused]] typename simd_info<INST_SET>::vec_reg_t oneReg16Bit,
    [[maybe_unused]] typename simd_info<INST_SET>::vec_reg_t tmpReg) {
  if constexpr (is_avx512_vnni_v<INST_SET>) {
    a->vpdpbusd(cReg, aReg, bReg);
  } else {
    a->vpmaddubsw(tmpReg, aReg, bReg);
    a->vpmaddwd(tmpReg, oneReg16Bit, tmpReg);
    a->vpaddd(cReg, tmpReg, cReg);
  }
}

/**
 * @brief Add 4 consecutive numbers of type uint8
 *        and emit their sum as 32-bit numbers.
 *        i.e., dest[0:31] contains
 *        src[0:7] + src[8:15] + src[16:23] + src[24:31]
 *
 * @param dest contains result
 */
template <inst_set_t INST_SET>
void genU8Sum4(
    x86::Emitter* a,
    typename simd_info<INST_SET>::vec_reg_t src,
    typename simd_info<INST_SET>::vec_reg_t dest,
    [[maybe_unused]] typename simd_info<INST_SET>::vec_reg_t oneReg16Bit,
    typename simd_info<INST_SET>::vec_reg_t tmpReg) {
  gen8BitVectorOne(a, tmpReg);
  if constexpr (is_avx512_vnni_v<INST_SET>) {
    a->vpdpbusd(dest, src, tmpReg);
  } else {
    a->vpmaddubsw(tmpReg, src, tmpReg);
    a->vpmaddwd(tmpReg, tmpReg, oneReg16Bit);
    a->vpaddd(dest, tmpReg, dest);
  }
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
 * @param dest contains result
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
  auto xmm = dest.xmm();
  a->movq(xmm, src);
  a->vpbroadcastb(dest, xmm);
}

} // namespace fbgemm
