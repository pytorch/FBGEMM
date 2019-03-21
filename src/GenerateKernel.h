/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once
#include <asmjit/asmjit.h>
#include <cpuinfo.h>
#include <map>
#include <string>
#include <tuple>
#include "fbgemm/Fbgemm.h"
/*#define FBGEMM_LOG_CODE 1*/

namespace fbgemm {

namespace x86 = asmjit::x86;

/**
 * @brief AVX2/AVX512 JIT assembly code generator.
 * @tparam TA Type of matrix A.
 * @tparam TB Type of matrix B.
 * @tparam TC Type of matrix C.
 * @tparam accT Accumulation type, currently we support 16-bit (std::int16_t) or
 * 32-bit (std::int32_t) accumulation.
 */
template <typename TA, typename TB, typename TC, typename accT>
class CodeGenBase {
 public:
  using jit_micro_kernel_fp = void (*)(
      TA* bufferA,
      TB* bufferB,
      TB* b_pf,
      TC* bufferC,
      int kc,
      int ldc);

  /**
   * @brief Constructor for initializing AVX2/AVX512 registers.
   */
  CodeGenBase()
      : CRegs_avx2_{x86::ymm0,
                    x86::ymm1,
                    x86::ymm2,
                    x86::ymm3,
                    x86::ymm4,
                    x86::ymm5,
                    x86::ymm6,
                    x86::ymm7,
                    x86::ymm8,
                    x86::ymm9,
                    x86::ymm10,
                    x86::ymm11},
        CRegs_avx512_{
            x86::zmm0,  x86::zmm1,  x86::zmm2,  x86::zmm3,  x86::zmm4,
            x86::zmm5,  x86::zmm6,  x86::zmm7,  x86::zmm8,  x86::zmm9,
            x86::zmm10, x86::zmm11, x86::zmm12, x86::zmm13, x86::zmm14,
            x86::zmm15, x86::zmm16, x86::zmm17, x86::zmm18, x86::zmm19,
            x86::zmm20, x86::zmm21, x86::zmm22, x86::zmm23, x86::zmm24,
            x86::zmm25, x86::zmm26, x86::zmm27,
        },
        AllRegs_avx512_{x86::zmm0,  x86::zmm1,  x86::zmm2,  x86::zmm3,
                        x86::zmm4,  x86::zmm5,  x86::zmm6,  x86::zmm7,
                        x86::zmm8,  x86::zmm9,  x86::zmm10, x86::zmm11,
                        x86::zmm12, x86::zmm13, x86::zmm14, x86::zmm15,
                        x86::zmm16, x86::zmm17, x86::zmm18, x86::zmm19,
                        x86::zmm20, x86::zmm21, x86::zmm22, x86::zmm23,
                        x86::zmm24, x86::zmm25, x86::zmm26, x86::zmm27,
                        x86::zmm28, x86::zmm29, x86::zmm30, x86::zmm31} {
    // vector width in bits
    if (cpuinfo_initialize()) {
      if (fbgemmHasAvx512Support()) {
        vectorWidth_ = 512;
      } else if (fbgemmHasAvx2Support()) {
        vectorWidth_ = 256;
      } else {
        // TODO: Have default path
        assert(0 && "unsupported architecture");
        return;
      }
    } else {
      throw std::runtime_error("Failed to initialize cpuinfo!");
    }
    // vector width in elements
    VLEN_ = vectorWidth_ / 8 * sizeof(TA);
  }

  /**
   * @brief Get or Create the instructions for macro-kernel.
   *
   * If the problem size (mc, nc) and accumulation flag (accum) can be found in
   * the code cache (a hash map), then get the macro-kernel instructions
   * directly from it. Otherwise, create the instructions for macro-kernel, and
   * store that into the code cache.
   */
  template <inst_set_t instSet>
  jit_micro_kernel_fp
  getOrCreate(bool accum, int32_t mc, int32_t nc, int32_t kc, int32_t ldc);

  /**
   * @brief Generate instructions for initializing the C registers to 0.
   */
  template <inst_set_t instSet>
  void initCRegs(
      asmjit::X86Emitter* a,
      int rowRegs,
      int colRegs,
      int leadingDimCRegAssign = 4);

  /**
   * @brief Generate instructions for computing block in the rank-k update.
   */
  template <inst_set_t instSet>
  void genComputeBlock(
      asmjit::X86Emitter* a,
      asmjit::X86Gp buffer_A,
      asmjit::X86Gp buffer_B,
      asmjit::X86Gp B_pf,
      int rowRegs,
      int colRegs,
      int lda,
      int leadingDimCRegAssign = 4);

  /**
   * @brief Generate instructions for storing the C registers back to the
   * memory.
   */
  template <inst_set_t instSet>
  void storeCRegs(
      asmjit::X86Emitter* a,
      int rowRegs,
      int colRegs,
      asmjit::X86Gp C_Offset,
      asmjit::X86Gp ldcReg,
      bool accum,
      int leadingDimCRegAssign = 4);

  /**
   * @brief Generate filename to dump generated code
   * (debug-only)
   */
  template <inst_set_t instSet>
  std::string getCodeLoggingFile(bool accum, int mc, int nc) {
    std::string fileName = "gemm_";
    if (std::is_same<accT, std::int16_t>::value) {
      fileName += "acc16_";
    } else if (std::is_same<accT, std::int32_t>::value) {
      fileName += "acc32_";
    } else {
      fileName += "unknown_";
    }
    fileName += "accum-" + std::to_string(accum);
    fileName += "_MC-" + std::to_string(mc);
    fileName += "_NC-" + std::to_string(nc);
    if (instSet == inst_set_t::avx512) {
      fileName += "_avx512";
    } else if (instSet == inst_set_t::avx2) {
      fileName += "_avx2";
    }
    fileName += ".txt";
    return fileName;
  }

 private:
  asmjit::X86Ymm
      CRegs_avx2_[12]; ///< AVX2 ymm registers for C in the micro-kernel.
  asmjit::X86Zmm
      CRegs_avx512_[28]; ///< AVX512 zmm registers for C in the micro-kernel.
  asmjit::X86Zmm
      AllRegs_avx512_[32]; ///< all AVX512 zmm registers.

  int vectorWidth_; ///< Vector width in bits.
  int VLEN_; ///< Vector width in elements.
  static thread_local asmjit::JitRuntime rt_; ///< JIT Runtime for asmjit.
  static thread_local asmjit::CodeHolder code_; ///< JIT Code Holder for asmjit.
  static thread_local std::map<std::tuple<bool, int, int>, jit_micro_kernel_fp>
      codeCache_; ///< JIT Code Cache for reuse.
};

} // namespace fbgemm
