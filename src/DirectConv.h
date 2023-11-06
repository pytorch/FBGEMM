/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <asmjit/asmjit.h>
#include <cpuinfo.h>
#include <cassert>
#include <cstdint>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include "./CodeCache.h"
#include "fbgemm/ConvUtils.h"
#include "fbgemm/Fbgemm.h"
#include "fbgemm/Utils.h"
/*#define FBGEMM_LOG_CODE 1*/

namespace fbgemm {

namespace x86 = asmjit::x86;

/**
 * @brief Generate instructions for initializing the C registers to 0.
 */
void initCRegs(x86::Emitter* a, int rowRegs, int colRegs);

template <typename TA, typename TB, typename TC, typename accT>
class DirectConvCodeGenBase {
 public:
  using jit_micro_kernel_fp = void (*)(
      const TA* bufferA,
      const TB* bufferB,
      const TB* b_pf,
      TC* bufferC,
      int kc,
      int ldc);

  // microkernel signature for transposed direct conv
  // ic: input channel
  // ldcReg: leading dimension of output, a.k.a OC
  // o1Xoc: output width multiply output channel:
  // OUT_DIM[1] x OC
  using jit_micro_kernel_fp_convT = void (*)(
      const TA* bufferA,
      const TB* bufferB,
      TC* bufferC,
      int ic,
      int ldcReg,
      int o1Xoc,
      int i1);

  static std::mutex rtMutex_; ///< Control access to runtime;

  // The hash depends on accumulate, mc, nc, ncb, kcb, nr, mr
  static CodeCache<
      std::tuple<bool, int, int, int, int, int, int>,
      jit_micro_kernel_fp>
      codeCache_; ///< JIT Code Cache for reuse.

  // The hash depends on accumulate, stride, mr, nr
  static CodeCache<
      std::tuple<bool, int, int, int>,
      jit_micro_kernel_fp_convT>
      codeCacheT_; ///< JIT Code Cache for reuse.

  /**
   * @brief Generate instructions for storing the C registers back to the
   * memory.
   */
  template <inst_set_t instSet>
  void storeCRegs(
      x86::Emitter* a,
      int rowRegs,
      int colRegs,
      x86::Gp C_Offset,
      x86::Gp ldcReg,
      bool accum);

  /**
   * @brief Generate instructions for storing the C registers back to the
   * memory.
   */
  template <inst_set_t instSet>
  void storeCRegsTrans(
      x86::Emitter* a,
      int rowRegs,
      int colRegs,
      x86::Gp C_offset,
      x86::Gp o1XocReg,
      x86::Gp ldcReg,
      bool accum);

  /**
   * @brief Generate filename to dump generated code
   * (debug-only)
   */
  template <inst_set_t instSet>
  static std::string getCodeLoggingFile(
      bool accum,
      int mc,
      int nc,
      int NCB,
      int KCB,
      int MR,
      int NR) {
    std::ostringstream oss;
    oss << "directconv_";
    if (std::is_same<accT, std::int16_t>::value) {
      oss << "acc16_";
    } else if (std::is_same<accT, std::int32_t>::value) {
      oss << "acc32_";
    } else {
      oss << "unknown_";
    }
    oss << "accum-" + std::to_string(accum) << "_MC-" + std::to_string(mc)
        << "_NC-" + std::to_string(nc) << "_NCB-" + std::to_string(NCB)
        << "_KCB-" + std::to_string(KCB) << "_MR-" + std::to_string(MR)
        << "_NR-" + std::to_string(NR);
    if (instSet == inst_set_t::avx512_vnni) {
      oss << "_avx512vnni";
    } else if (instSet == inst_set_t::avx512) {
      oss << "_avx512";
    } else if (instSet == inst_set_t::avx512_ymm) {
      oss << "_avx512_ymm";
    } else if (instSet == inst_set_t::avx2) {
      oss << "_avx2";
    }
    oss << ".txt";
    return oss.str();
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
  getOrCreateDirectConv(bool accum, int32_t mc, int32_t nc, int32_t kc);

  /**
   * @brief Get or Create the instructions for macro-kernel.
   *
   * If the problem size (mc, nc) and accumulation flag (accum) can be found in
   * the code cache (a hash map), then get the macro-kernel instructions
   * directly from it. Otherwise, create the instructions for macro-kernel, and
   * store that into the code cache.
   */
  template <inst_set_t instSet>
  jit_micro_kernel_fp_convT
  getOrCreateDirectConvTrans(bool accum, int32_t stride, int32_t numColRegs);

  /**
   * @brief Generate instructions for computing block in the rank-k update.
   */
  template <inst_set_t instSet>
  void genComputeBlock(
      x86::Emitter* a,
      x86::Gp buffer_A,
      x86::Gp buffer_B,
      x86::Gp B_pf,
      int rowRegs,
      int colRegs,
      int lda);
  /**
   * @brief Generate instructions for computing block in the rank-k update.
   */
  template <inst_set_t instSet>
  void genComputeBlockDirectConv(
      x86::Emitter* a,
      x86::Gp buffer_A,
      x86::Gp buffer_B,
      x86::Gp B_pf,
      int rowRegs,
      int colRegs,
      int strideXich);

  /**
   * @brief Generate instructions for computing block in the rank-k update.
   */
  template <inst_set_t instSet>
  void genComputeBlockDirectConvTrans(
      x86::Emitter* a,
      x86::Gp buffer_A,
      x86::Gp buffer_B,
      x86::Gp icReg,
      x86::Gp C_offset,
      int rowRegs,
      int colRegs);

 private:
  static asmjit::JitRuntime& runtime() {
    static asmjit::JitRuntime rt; //< JIT Runtime for asmjit,
                                  // depents on other static
                                  // variables.  Required to prevent
                                  // initialization order fiasco
    return rt;
  }
};

template <typename TA, typename TB, typename TC, typename accT>
std::mutex DirectConvCodeGenBase<TA, TB, TC, accT>::rtMutex_;

template <typename TA, typename TB, typename TC, typename accT>
CodeCache<
    std::tuple<bool, int, int, int, int, int, int>,
    typename DirectConvCodeGenBase<TA, TB, TC, accT>::jit_micro_kernel_fp>
    DirectConvCodeGenBase<TA, TB, TC, accT>::codeCache_;

template <typename TA, typename TB, typename TC, typename accT>
CodeCache<
    std::tuple<bool, int, int, int>,
    typename DirectConvCodeGenBase<TA, TB, TC, accT>::jit_micro_kernel_fp_convT>
    DirectConvCodeGenBase<TA, TB, TC, accT>::codeCacheT_;

}; // namespace fbgemm
