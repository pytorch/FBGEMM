/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <iostream>
#include "./GenerateKernel.h"

namespace fbgemm {

namespace x86 = asmjit::x86;

/**
 * Generate AVX512 instructions for initializing the C registers to 0 in 16-bit
 * Accumulation kernel.
 */
template <>
template <>
void CodeGenBase<uint8_t, int8_t, int32_t, int16_t>::initCRegs<
    inst_set_t::avx512_vnni>(x86::Emitter* a, int rowRegs, int colRegs) {
  assert(0 && "Accumulation to int16_t is not available for VNNI!");

  // For AVX512VNNI, redirect to int32_t accumulation.
  CodeGenBase<uint8_t, int8_t, int32_t, int32_t> codeObj;
  codeObj.initCRegs<inst_set_t::avx512_vnni>(a, rowRegs, colRegs);
}

/**
 * Generate AVX512 instructions for computing block in the rank-k update of
 * 16-bit Accmulation kernel.
 */
template <>
template <>
void CodeGenBase<uint8_t, int8_t, int32_t, int16_t>::genComputeBlock<
    inst_set_t::avx512_vnni>(
    x86::Emitter* a,
    x86::Gp buffer_A,
    x86::Gp buffer_B,
    x86::Gp /* unused (reserved for prefetching)*/,
    int rowRegs,
    int colRegs,
    int lda) {
  assert(0 && "Accumulation to int16_t is not available for VNNI!");

  // For AVX512VNNI, redirect to int32_t accumulation.
  CodeGenBase<uint8_t, int8_t, int32_t, int32_t> codeObj;
  codeObj.genComputeBlock<inst_set_t::avx512_vnni>(
      a, buffer_A, buffer_B, buffer_B, rowRegs, colRegs, lda);
}

/**
 * Generate AVX512 instructions for storing the C registers back to the memory
 * in 16-bit Accumulation kernel.
 */
template <>
template <>
void CodeGenBase<uint8_t, int8_t, int32_t, int16_t>::storeCRegs<
    inst_set_t::avx512_vnni>(
    x86::Emitter* a,
    int rowRegs,
    int colRegs,
    x86::Gp C_Offset,
    x86::Gp ldcReg,
    bool accum) {
  assert(0 && "Accumulation to int16_t is not available for VNNI!");

  // For AVX512VNNI, redirect to int32_t accumulation.
  CodeGenBase<uint8_t, int8_t, int32_t, int32_t> codeObj;
  codeObj.storeCRegs<inst_set_t::avx512_vnni>(
      a, rowRegs, colRegs, C_Offset, ldcReg, accum);
}

/**
 * Get or Create the AVX512 instructions for 16-bit Accumulation macro-kernel.
 *
 */
template <>
template <>
CodeGenBase<uint8_t, int8_t, int32_t, int16_t>::jit_micro_kernel_fp
CodeGenBase<uint8_t, int8_t, int32_t, int16_t>::getOrCreate<
    inst_set_t::avx512_vnni>(bool accum, int32_t mc, int32_t nc, int32_t kc) {
  assert(0 && "Accumulation to int16_t is not available for VNNI!");

  // For AVX512VNNI, redirect to int32_t accumulation.
  CodeGenBase<uint8_t, int8_t, int32_t, int32_t> codeObj;
  return codeObj.getOrCreate<inst_set_t::avx512_vnni>(accum, mc, nc, kc);
}

} // namespace fbgemm
