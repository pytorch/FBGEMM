/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once
#include <asmjit/asmjit.h>
#include <cpuinfo.h>
#include <cassert>
#include <cstdint>
#include <map>
#include <string>
#include <tuple>
#include "fbgemm/ConvUtils.h"
#include "fbgemm/Fbgemm.h"
#include "fbgemm/Utils.h"
/*#define FBGEMM_LOG_CODE 1*/

namespace fbgemm {

namespace x86 = asmjit::x86;

using jit_conv_kernel_fp = void (*)(
    const uint8_t* in_acts,
    int8_t* wghts,
    int32_t* out_acts,
    int32_t a_zero_pt,
    int32_t height,
    int32_t width);

using jit_rowoffset_kernel_fp = void (*)(
    const uint8_t* in_acts,
    int32_t a_zero_pt,
    int32_t height,
    int32_t width,
    int32_t* row_offset);

template <typename accT = int32_t>
class GenConvKernel {
 public:
  GenConvKernel(const conv_param_t<>& conv_param, std::int32_t a_zero_point)
      : WRegs_avx2_{x86::ymm0,
                    x86::ymm1,
                    x86::ymm2,
                    x86::ymm3,
                    x86::ymm4,
                    x86::ymm5,
                    x86::ymm6,
                    x86::ymm7,
                    x86::ymm8} {
    // vector width in bits
    if (cpuinfo_initialize()) {
      if (cpuinfo_has_x86_avx512f()) {
        vectorWidth_ = 512;
      } else if (cpuinfo_has_x86_avx2()) {
        vectorWidth_ = 256;
      } else {
        // TODO: Have default path
        assert(0 && "unsupported architecture");
        return;
      }
    } else {
      throw std::runtime_error("Failed to initialize cpuinfo!");
    }
    zeroPTRegAvx2_ = x86::ymm9;
    oneReg8BitAvx2_ = x86::ymm10;
    tmpReg1Avx2_ = x86::ymm11;
    stPermRegAvx2_ = x86::ymm12;
    actRegAvx2_ = x86::ymm13;
    resultRegAvx2_ = x86::ymm14;
    oneReg16BitAvx2_ = x86::ymm15;

    // vector width in elements; Each element is int8 or uint8
    VLEN_ = vectorWidth_ / 8;

    isAZeroPointZero_ = a_zero_point == 0;

    G_ = conv_param.G;
    K_per_G_ = conv_param.OC / conv_param.G;
    K_ = conv_param.OC;
    C_per_G_ = conv_param.IC / conv_param.G;
    C_ = conv_param.IC;
    R_ = conv_param.K[0];
    S_ = conv_param.K[1];
    H_ = conv_param.OUT_DIM[0];
    W_ = conv_param.OUT_DIM[1];
    H_PAD_ = conv_param.pad[0];
    W_PAD_ = conv_param.pad[1];

    assert(fbgemmOptimizedGConv(conv_param));
  }

  template <inst_set_t instSet>
  std::string getCodeLoggingFile(bool rowOffsetKernel = false) {
    std::string fileName = "conv_";
    fileName += "G-" + std::to_string(G_);
    fileName += "_K-" + std::to_string(K_);
    fileName += "_C-" + std::to_string(C_);
    fileName += "_R-" + std::to_string(R_);
    fileName += "_S-" + std::to_string(S_);
    fileName += "_PADH-" + std::to_string(H_PAD_);
    fileName += "_PADW-" + std::to_string(W_PAD_);
    fileName += "_isZeroPointZero-" + std::to_string(isAZeroPointZero_);
    if (rowOffsetKernel) {
      fileName += "_rowOffset";
    }

    if (instSet == inst_set_t::avx512) {
      fileName += "_avx512";
    } else if (instSet == inst_set_t::avx2) {
      fileName += "_avx2";
    }
    fileName += ".txt";
    return fileName;
  }

  ~GenConvKernel() {}

  template <inst_set_t instSet>
  jit_conv_kernel_fp getOrCreate(const conv_param_t<>& conv_param);

  template <inst_set_t instSet>
  jit_rowoffset_kernel_fp getOrCreateRowOffset(
      const conv_param_t<>& conv_param);

  template <inst_set_t instSet>
  void createVector16BitOne(asmjit::X86Emitter* a);

  template <inst_set_t instSet>
  void createVector8BitOne(asmjit::X86Emitter* a);

  template <inst_set_t instSet>
  void setToZeroPt(asmjit::X86Emitter* a, asmjit::X86Ymm destReg);

  template <inst_set_t instSet>
  void
  gen8bitFMA(asmjit::X86Emitter* a, asmjit::X86Ymm aReg, asmjit::X86Ymm wReg);

  template <inst_set_t instSet>
  void genForLoadingWeights(asmjit::X86Emitter* a, int c_offset);

  template <inst_set_t instSet>
  void genConstForPermutations(asmjit::X86Emitter* a);

  template <inst_set_t instSet>
  void genForTopEdge(asmjit::X86Emitter* a, int c_offset);

  template <inst_set_t instSet>
  void genForLeftEdge(asmjit::X86Emitter* a, int c_offset);

  template <inst_set_t instSet>
  void genForRightEdge(asmjit::X86Emitter* a, int c_offset);

  template <inst_set_t instSet>
  void genForBottomEdge(asmjit::X86Emitter* a, int c_offset);

  template <inst_set_t instSet>
  void genCoreInsts(asmjit::X86Emitter* a, int c_offset);

  template <inst_set_t instSet>
  void storeResult(asmjit::X86Emitter* a);

  // for Rowoffset kernel
  // Add 4 consecutive numbers of 32 uint8 and emit 8 32-bit
  template <inst_set_t instSet>
  void gen8BitSumX4(asmjit::X86Emitter* a, asmjit::X86Ymm aReg);

  // Add 8 consecutive numbers of 64 uint8 and emit 8 32-bit
  template <inst_set_t instSet>
  void
  gen8BitSumX8(asmjit::X86Emitter* a, asmjit::X86Ymm aReg, asmjit::X86Ymm bReg);

  // Add 16 consecutive numbers of 128 uint8 and emit 8 32-bit
  template <inst_set_t instSet>
  void gen8BitSumX16(
      asmjit::X86Emitter* a,
      asmjit::X86Ymm aReg,
      asmjit::X86Ymm bReg,
      asmjit::X86Ymm cReg,
      asmjit::X86Ymm dReg);

  // Generate instruction sequence that loads 8-bit values and sum them up.
  // Depending on C_per_G_, this function dispatches to gen8BitSumX4/8/16
  // This function assumes in_acts_R_ has the base pointer to activation,
  // scratchReg1_ has a variable offset, and act_offset has the final immediate
  // offset.
  // Internally, actRegAvx2_, stPermRegAvx2_, WRegs_avx2_[0, 1], tmpReg1Avx2_,
  // and resultRegAvx2_ are used.
  template <inst_set_t instSet>
  void gen8BitSum(
      asmjit::X86Emitter* a,
      int act_offset,
      bool use_scratch_reg1 = true);

  // Use scratchReg1_ and tmpReg1Avx2_ internally
  template <inst_set_t instSet>
  void genZeroPtSum(asmjit::X86Emitter* a, int multiplier);

  template <inst_set_t instSet>
  void genForTopEdgeRowoffset(asmjit::X86Emitter* a);

  template <inst_set_t instSet>
  void genForLeftEdgeRowoffset(asmjit::X86Emitter* a);

  template <inst_set_t instSet>
  void genForRightEdgeRowoffset(asmjit::X86Emitter* a);

  template <inst_set_t instSet>
  void genForBottomEdgeRowoffset(asmjit::X86Emitter* a);

  template <inst_set_t instSet>
  void genRowoffsetCorners(asmjit::X86Emitter* a);

  template <inst_set_t instSet>
  void genRowoffsetCore(asmjit::X86Emitter* a);

  template <inst_set_t instSet>
  void storeResultRowoffset(asmjit::X86Emitter* a, int offset = 0);

  static thread_local asmjit::JitRuntime rt_; ///< JIT Runtime for asmjit.
  static thread_local asmjit::CodeHolder code_; ///< JIT Code Holder for asmjit.
  static thread_local std::
      map<std::tuple<bool, int, int, int>, jit_conv_kernel_fp>
          codeCache_; ///< JIT Code Cache for reuse.
  static thread_local std::
      map<std::tuple<bool, int, int, int>, jit_rowoffset_kernel_fp>
          codeCacheRowOffset_; ///< JIT Code Cache for row offset kernel.

 private:
  int vectorWidth_; ///< Vector width in bits.
  int VLEN_; ///< Vector width in elements.
  // avx2 specific
  asmjit::X86Ymm
      WRegs_avx2_[9]; ///< AVX2 ymm registers for weights in the micro-kernel.
  asmjit::X86Ymm zeroPTRegAvx2_;
  asmjit::X86Ymm tmpReg1Avx2_;
  asmjit::X86Ymm stPermRegAvx2_;
  asmjit::X86Ymm actRegAvx2_;
  asmjit::X86Ymm resultRegAvx2_;
  asmjit::X86Ymm oneReg8BitAvx2_;
  asmjit::X86Ymm oneReg16BitAvx2_;

  // arguments to the function created
  asmjit::X86Gp in_acts_R_;
  asmjit::X86Gp wghts_R_;
  asmjit::X86Gp out_acts_R_;
  asmjit::X86Gp a_zero_pt_R_;
  asmjit::X86Gp H_R_;
  asmjit::X86Gp W_R_;
  asmjit::X86Gp row_offset_R_;

  // Used registers
  asmjit::X86Gp loopR1_;
  asmjit::X86Gp loopR2_;
  asmjit::X86Gp scratchReg1_;
  asmjit::X86Gp scratchReg2_;

  // Other parameters
  bool isAZeroPointZero_;

  // current conv parameters
  int G_; ///< Number of groups
  int K_; ///< Number of output channels
  int K_per_G_; ///< Number of output channels per group
  int C_; ///< Number of input channels
  int C_per_G_; ///< Number of input channels per group
  int R_; ///< Filter/Kernel height
  int S_; ///< Filter/Kernel width
  int H_;
  int W_;
  int H_PAD_; ///< Padding for height (top and bottom)
  int W_PAD_; ///< Padding for width (left and right)
};

} // namespace fbgemm
