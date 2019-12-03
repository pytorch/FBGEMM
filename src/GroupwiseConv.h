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
#include <mutex>
#include <string>
#include <tuple>
#include "CodeCache.h"
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

template <int SPATIAL_DIM = 2, typename accT = int32_t>
class GenConvKernel {
 public:
  GenConvKernel(
      const conv_param_t<SPATIAL_DIM>& conv_param,
      std::int32_t a_zero_point)
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
      if (fbgemmHasAvx512Support()) {
        // TODO: change this to 512 once we have avx512f version
        vectorWidth_ = 256;
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
  jit_conv_kernel_fp getOrCreate(const conv_param_t<SPATIAL_DIM>& conv_param);

  template <inst_set_t instSet>
  jit_rowoffset_kernel_fp getOrCreateRowOffset(
      const conv_param_t<SPATIAL_DIM>& conv_param);

  template <inst_set_t instSet>
  void createVector16BitOne(x86::Emitter* a);

  template <inst_set_t instSet>
  void createVector8BitOne(x86::Emitter* a);

  template <inst_set_t instSet>
  void setToZeroPt(x86::Emitter* a, x86::Ymm destReg);

  template <inst_set_t instSet>
  void gen8bitFMA(x86::Emitter* a, x86::Ymm aReg, x86::Ymm wReg);

  template <inst_set_t instSet>
  void genForLoadingWeights(x86::Emitter* a, int c_offset);

  template <inst_set_t instSet>
  void genConstForPermutations(x86::Emitter* a);

  template <inst_set_t instSet>
  void genForTopEdge(x86::Emitter* a, int c_offset);

  template <inst_set_t instSet>
  void genForLeftEdge(x86::Emitter* a, int c_offset);

  template <inst_set_t instSet>
  void genForRightEdge(x86::Emitter* a, int c_offset);

  template <inst_set_t instSet>
  void genForBottomEdge(x86::Emitter* a, int c_offset);

  template <inst_set_t instSet>
  void genCoreInsts(x86::Emitter* a, int c_offset);

  template <inst_set_t instSet>
  void storeResult(x86::Emitter* a);

  // for Rowoffset kernel
  // Add 4 consecutive numbers of 32 uint8 and emit 8 32-bit
  template <inst_set_t instSet>
  void gen8BitSumX4(x86::Emitter* a, x86::Ymm aReg);

  // Add 8 consecutive numbers of 64 uint8 and emit 8 32-bit
  template <inst_set_t instSet>
  void gen8BitSumX8(x86::Emitter* a, x86::Ymm aReg, x86::Ymm bReg);

  // Add 16 consecutive numbers of 128 uint8 and emit 8 32-bit
  template <inst_set_t instSet>
  void gen8BitSumX16(
      x86::Emitter* a,
      x86::Ymm aReg,
      x86::Ymm bReg,
      x86::Ymm cReg,
      x86::Ymm dReg);

  // Generate instruction sequence that loads 8-bit values and sum them up.
  // Depending on C_per_G_, this function dispatches to gen8BitSumX4/8/16
  // This function assumes in_acts_R_ has the base pointer to activation,
  // scratchReg1_ has a variable offset, and act_offset has the final immediate
  // offset.
  // Internally, actRegAvx2_, stPermRegAvx2_, WRegs_avx2_[0, 1], tmpReg1Avx2_,
  // and resultRegAvx2_ are used.
  template <inst_set_t instSet>
  void
  gen8BitSum(x86::Emitter* a, int act_offset, bool use_scratch_reg1 = true);

  // Use scratchReg1_ and tmpReg1Avx2_ internally
  template <inst_set_t instSet>
  void genZeroPtSum(x86::Emitter* a, int multiplier);

  template <inst_set_t instSet>
  void genForTopEdgeRowoffset(x86::Emitter* a);

  template <inst_set_t instSet>
  void genForLeftEdgeRowoffset(x86::Emitter* a);

  template <inst_set_t instSet>
  void genForRightEdgeRowoffset(x86::Emitter* a);

  template <inst_set_t instSet>
  void genForBottomEdgeRowoffset(x86::Emitter* a);

  template <inst_set_t instSet>
  void genRowoffsetCorners(x86::Emitter* a);

  template <inst_set_t instSet>
  void genRowoffsetCore(x86::Emitter* a);

  template <inst_set_t instSet>
  void storeResultRowoffset(x86::Emitter* a, int offset = 0);


  static asmjit::JitRuntime &runtime() {
    static asmjit::JitRuntime rt; //< JIT Runtime for asmjit,
                                  // depents on other static
                                  // variables.  Required to prevent
                                  // initialization order fiasco
    return rt;
  }

  static std::mutex rtMutex_; ///< Controll access to runtime;

  static CodeCache<std::tuple<bool, int, int, int>, jit_conv_kernel_fp>
      codeCache_; ///< JIT Code Cache for reuse.
  static CodeCache<std::tuple<bool, int, int, int>, jit_rowoffset_kernel_fp>
      codeCacheRowOffset_; ///< JIT Code Cache for row offset kernel.

private:
  int vectorWidth_; ///< Vector width in bits.
  int VLEN_; ///< Vector width in elements.
  // avx2 specific
  x86::Ymm
      WRegs_avx2_[9]; ///< AVX2 ymm registers for weights in the micro-kernel.
  x86::Ymm zeroPTRegAvx2_;
  x86::Ymm tmpReg1Avx2_;
  x86::Ymm stPermRegAvx2_;
  x86::Ymm actRegAvx2_;
  x86::Ymm resultRegAvx2_;
  x86::Ymm oneReg8BitAvx2_;
  x86::Ymm oneReg16BitAvx2_;

  // arguments to the function created
  x86::Gp in_acts_R_;
  x86::Gp wghts_R_;
  x86::Gp out_acts_R_;
  x86::Gp a_zero_pt_R_;
  x86::Gp H_R_;
  x86::Gp W_R_;
  x86::Gp row_offset_R_;

  // Used registers
  x86::Gp loopR1_;
  x86::Gp loopR2_;
  x86::Gp scratchReg1_;
  x86::Gp scratchReg2_;

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

template <int SPATIAL_DIM, typename accT>
std::mutex GenConvKernel<SPATIAL_DIM, accT>::rtMutex_;

template <int SPATIAL_DIM, typename accT>
CodeCache<std::tuple<bool, int, int, int>, jit_conv_kernel_fp>
    GenConvKernel<SPATIAL_DIM, accT>::codeCache_;

template <int SPATIAL_DIM, typename accT>
CodeCache<std::tuple<bool, int, int, int>, jit_rowoffset_kernel_fp>
    GenConvKernel<SPATIAL_DIM, accT>::codeCacheRowOffset_;

} // namespace fbgemm
