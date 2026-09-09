/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <cpuinfo.h>
#include <mutex>
#include <sstream>
#include <string>
#include <tuple>
#include "./CodeCache.h" // @manual
#include "./JitPerfMap.h" // @manual
#include "fbgemm/Fbgemm.h"
#include "fbgemm/SimdUtils.h"
// #define FBGEMM_LOG_CODE 1

namespace fbgemm {

namespace x86 = asmjit::x86;

/**
 * @brief Generate instructions for initializing the C registers to 0.
 */
void initCRegs(x86::Emitter* a, int rowRegs, int colRegs);

/**
 * @brief AVX2/AVX512/AVX512VNNI JIT assembly code generator.
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
      const TA* bufferA,
      const TB* bufferB,
      const TB* b_pf,
      TC* bufferC,
      int kc,
      int ldc);

  /**
   * @brief Constructor for initializing AVX2/AVX512 registers.
   */
  CodeGenBase(const BlockingFactors* params = nullptr)
      : blocking_params(params) {}

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
  getOrCreate(bool accum, int32_t mc, int32_t nc, int32_t kc);

  /**
   * @brief Generate instructions for computing block in the rank-k update.
   */
  template <inst_set_t instSet>
  void genComputeBlock(
      x86::Emitter* a,
      const x86::Gp& buffer_A,
      const x86::Gp& buffer_B,
      const x86::Gp& B_pf,
      int rowRegs,
      int colRegs,
      int lda);

  /**
   * @brief Generate instructions for storing the C registers back to the
   * memory.
   */
  template <inst_set_t instSet>
  void storeCRegs(
      x86::Emitter* a,
      int rowRegs,
      int colRegs,
      const x86::Gp& C_Offset,
      const x86::Gp& ldcReg,
      bool accum);

  const BlockingFactors* blocking_params;
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
    return getKernelName<instSet>(accum, mc, nc, NCB, KCB, MR, NR) + ".txt";
  }

  /**
   * @brief The perf-map symbol for this kernel.
   *
   * Shares getKernelName() with the debug dump filename so the two cannot
   * drift, and keeps the "fbgemm::" prefix here rather than repeating it in
   * every generator.
   */
  template <inst_set_t instSet>
  static std::string getKernelSymbol(
      bool accum,
      int mc,
      int nc,
      int NCB,
      int KCB,
      int MR,
      int NR) {
    return "fbgemm::" + getKernelName<instSet>(accum, mc, nc, NCB, KCB, MR, NR);
  }

  // The code-logging filename without its extension, so the same descriptive
  // shape can be reused as a JIT symbol name. Deliberately unprefixed: this
  // also feeds getCodeLoggingFile(), and a "fbgemm::" prefix would put a colon
  // in the dump filename, which is not legal on Windows. Callers registering a
  // perf-map symbol prepend the namespace themselves.
  template <inst_set_t instSet>
  static std::string
  getKernelName(bool accum, int mc, int nc, int NCB, int KCB, int MR, int NR) {
    std::ostringstream oss;
    oss << "gemm_";
    if constexpr (std::is_same_v<accT, std::int16_t>) {
      oss << "acc16_";
    } else if constexpr (std::is_same_v<accT, std::int32_t>) {
      oss << "acc32_";
    } else if constexpr (std::is_same_v<accT, std::int64_t>) {
      oss << "acc64_";
    } else {
      oss << "unknown_";
    }
    oss << "accum-" + std::to_string(accum) << "_MC-" + std::to_string(mc)
        << "_NC-" + std::to_string(nc) << "_NCB-" + std::to_string(NCB)
        << "_KCB-" + std::to_string(KCB) << "_MR-" + std::to_string(MR)
        << "_NR-" + std::to_string(NR);
    // instSetName() rather than a local chain: it is the same spelling the
    // embedding kernels use, and it has a fallback, so an instSet nobody
    // thought to list here still names itself instead of silently producing a
    // symbol with no ISA at all.
    oss << "_" << instSetName<instSet>();
    return oss.str();
  }

 private:
  static asmjit::JitRuntime& runtime() {
    static asmjit::JitRuntime rt; //< JIT Runtime for asmjit,
                                  // depents on other static
                                  // variables.  Required to prevent
                                  // initialization order fiasco
    return rt;
  }

  inline static std::mutex rtMutex_; ///< Controll access to runtime;

  // The hash depends on accumulate, mc, nc, ncb, kcb, nr, mr
  inline static CodeCache<
      std::tuple<bool, int, int, int, int, int, int>,
      jit_micro_kernel_fp>
      codeCache_; ///< JIT Code Cache for reuse.
};

} // namespace fbgemm
