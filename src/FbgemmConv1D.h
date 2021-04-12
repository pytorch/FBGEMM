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
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include "./CodeCache.h"
#include "fbgemm/ConvUtils.h"
#include "fbgemm/Fbgemm.h"
#include "fbgemm/Utils.h"

/*#define FBGEMM_LOG_CODE 1*/

#define INST_AVX2_HEADER                \
  template <inst_set_t ISET = INST_SET> \
  typename std::enable_if<ISET == inst_set_t::avx2, void>::type

#define INST_AVX512_AND_VNNI_HEADER                                  \
  template <inst_set_t ISET = INST_SET>                              \
  typename std::enable_if<                                           \
      ISET == inst_set_t::avx512 || ISET == inst_set_t::avx512_vnni, \
      void>::type

#define INST_DEF_AVX2_HEADER     \
  template <inst_set_t INST_SET> \
  template <inst_set_t ISET>     \
  typename std::enable_if<ISET == inst_set_t::avx2, void>::type

#define INST_DEF_AVX512_AND_VNNI_HEADER                              \
  template <inst_set_t INST_SET>                                     \
  template <inst_set_t ISET>                                         \
  typename std::enable_if<                                           \
      ISET == inst_set_t::avx512 || ISET == inst_set_t::avx512_vnni, \
      void>::type

namespace fbgemm {

namespace x86 = asmjit::x86;

using jit_conv1d_kernel_fp = void (*)(
    const uint8_t* in_acts,
    int8_t* wghts,
    int32_t* out_acts,
    int32_t a_zero_pt,
    int32_t ow_start,
    int32_t ow_end,
    int32_t dummy,
    int32_t* row_offset);

using kernel1d_sig_t = std::tuple<
    bool, /* is A zero point 0 */
    bool, /* should row offset be calculated */
    bool, /* accumulate rowoffsets and output instead of overwrite? */
    int, /* groups */
    int, /* stride */
    int, /* number of input channels per group */
    int, /* number of output channels per group */
    int>; /* number of input */

// Common code in a base class
template <inst_set_t INST_SET>
class GenConv1DKernelBase {
 public:
  GenConv1DKernelBase(
      const conv_param_t<1>& conv_param,
      std::int32_t a_zero_point,
      bool needRowOffset,
      bool accum) {
    assert(take1DFastPath(conv_param));

    isAZeroPointZero_ = a_zero_point == 0;
    needRowOffset_ = needRowOffset;
    accum_ = accum;

    G_ = conv_param.G;
    K_per_G_ = conv_param.OC / conv_param.G;
    K_ = conv_param.OC;
    C_per_G_ = conv_param.IC / conv_param.G;
    C_ = conv_param.IC;

    // Strides are assumed to be the same in all directions
    STRIDE_ = conv_param.stride[0];
    S_ = conv_param.K[0];
    OW_ = conv_param.OUT_DIM[0];
    IW_ = conv_param.IN_DIM[0];
    W_PAD_ = conv_param.pad[0];
    COW_ = 0;

    use_right_padding_ = !(STRIDE_ > 1 && conv_param.IN_DIM[0] % 2 == 0);
  }

  ~GenConv1DKernelBase() {}

  static std::string getCodeLoggingFile(kernel1d_sig_t kernel_sig) {
    std::ostringstream oss;
    oss << "conv";
    oss << "_G-" << std::get<3>(kernel_sig);
    oss << "_stride-" << std::get<4>(kernel_sig);
    oss << "_IC_per_G-" << std::get<5>(kernel_sig);
    oss << "_OC_per_G-" << std::get<6>(kernel_sig);
    oss << "_IW-" << std::get<7>(kernel_sig);
    oss << "_isZeroPointZero-" << std::get<0>(kernel_sig);
    oss << "_rowoffset-" << std::get<1>(kernel_sig);
    oss << "_accum-" << std::get<2>(kernel_sig);

    if (INST_SET == inst_set_t::avx512) {
      oss << "_avx512";
    } else if (INST_SET == inst_set_t::avx2) {
      oss << "_avx2";
    } else {
      oss << "_unknown";
    }

    oss << ".txt";
    return oss.str();
  }

  static asmjit::JitRuntime& runtime() {
    static asmjit::JitRuntime rt; /* library-local */
                                  //< JIT Runtime for asmjit,
                                  // depents on other static
                                  // variables.  Required to prevent
                                  // initialization order fiasco
    return rt;
  }

  static std::mutex rtMutex_; ///< Control access to runtime;

  static CodeCache<
      kernel1d_sig_t,
      jit_conv1d_kernel_fp>
      codeCache_; ///< JIT Code Cache for reuse.

 protected:
  // current conv parameters
  int G_; ///< Number of groups
  int K_; ///< Number of output channels
  int K_per_G_; ///< Number of output channels per group
  int C_; ///< Number of input channels
  int STRIDE_; ///< Stride in either direction
  int C_per_G_; ///< Number of input channels per group
  int S_; ///< Filter/Kernel width
  int OW_; ///< output width
  int IW_; //
  int W_PAD_; ///< Padding for width (left and right)
  int COW_;

  // Other parameters
  bool isAZeroPointZero_;
  bool needRowOffset_;
  bool accum_;
  // For 3x3 kernels with pad == 1: If stride is 2 and image height/width are
  // even, the right or bottom paddings are not used. This variables is set to
  // false if paddings on the left and bottom are not used and kernel generation
  // takes care to not generate code with paddings on the right and bottom side.
  bool use_right_padding_;
};

// Generic class
template <inst_set_t INST_SET>
class FBGEMM_API GenConv1DKernel : public GenConv1DKernelBase<INST_SET> {
  typedef typename simd_info<INST_SET>::vec_reg_t vec_reg_t;

 public:
  GenConv1DKernel(
      const conv_param_t<1>& conv_param,
      int nreg,
      std::int32_t a_zero_point,
      bool needRowoffset,
      bool accum)
      : GenConv1DKernelBase<INST_SET>(
            conv_param,
            a_zero_point,
            needRowoffset,
            accum) {
    constexpr int SIMD_WIDTH = simd_info<INST_SET>::WIDTH_BYTES;

    zeroPTReg_V_ = vec_reg_t(0);
    tmpReg1_V_ = vec_reg_t(1);
    actReg_V_ = vec_reg_t(2);
    actReg_V1_ = vec_reg_t(3);
    actReg_V2_ = vec_reg_t(4);
    rowOffsetReg_V_ = vec_reg_t(5);
    rowOffsetReg_V1_ = vec_reg_t(6);
    rowOffsetReg_V2_ = vec_reg_t(7);
    oneReg16Bit_V_ = vec_reg_t(8);

    vsize_ = SIMD_WIDTH;
    int vsized4 = vsize_ / 4;

    cinLoopIters_ = (this->C_per_G_ + SIMD_WIDTH - 1) / SIMD_WIDTH;
    cinLoopRemainder_ = this->C_per_G_ % SIMD_WIDTH;
    coutLoopIters_ = (this->K_per_G_ + vsized4 - 1) / vsized4;
    coutLoopRemainder_ = this->K_per_G_ % vsized4;

    paddedICPerG_ = (this->C_per_G_ + 3) / 4 * 4;
    paddedOCPerG_ = (this->K_per_G_ + vsized4 - 1) / vsized4 * vsized4;

    nreg_ = nreg;
    kLoopIters_ = std::min(paddedICPerG_ * paddedOCPerG_ / SIMD_WIDTH, nreg_);
  }

  jit_conv1d_kernel_fp getOrCreate();

  void genCoreInsts(x86::Emitter* a);

  INST_AVX2_HEADER storeOffset(x86::Emitter* a);
  INST_AVX2_HEADER initResultRegs(x86::Emitter* a, int nw);
  INST_AVX2_HEADER
  genForSingleFilterPoint(x86::Emitter* a, int nw, int use_zero_flag);

  INST_AVX512_AND_VNNI_HEADER storeOffset(x86::Emitter* a);
  INST_AVX512_AND_VNNI_HEADER initResultRegs(x86::Emitter* a, int nw);
  INST_AVX512_AND_VNNI_HEADER
  genForSingleFilterPoint(x86::Emitter* a, int nw, int use_zero_flag);

  void genForSingleOutput(x86::Emitter* a, int nw);

  INST_AVX512_AND_VNNI_HEADER
  mAdd1(x86::Emitter* a, int b, int len, int nr, int use_zero_flag);

  INST_AVX512_AND_VNNI_HEADER
  mAdd2(x86::Emitter* a, int b, int len, int nr, int use_zero_flag);

  INST_AVX512_AND_VNNI_HEADER
  mAdd3(x86::Emitter* a, int b, int len, int nr, int use_zero_flag);

 private:
  int kLoopIters_;
  int cinLoopIters_;
  int coutLoopIters_;
  int cinLoopRemainder_;
  int coutLoopRemainder_;
  asmjit::FuncDetail func_;
  asmjit::FuncFrame frame_;
  vec_reg_t zeroPTReg_V_;
  vec_reg_t tmpReg1_V_;
  vec_reg_t stPermReg_V_;
  vec_reg_t actReg_V_;
  vec_reg_t actReg_V1_;
  vec_reg_t actReg_V2_;
  vec_reg_t resultReg_V_;
  vec_reg_t oneReg8Bit_V_;
  vec_reg_t oneReg16Bit_V_;
  vec_reg_t rowOffsetReg_V_;
  vec_reg_t rowOffsetReg_V1_;
  vec_reg_t rowOffsetReg_V2_;

  // arguments to the function created
  x86::Gp in_acts_R_;
  x86::Gp wghts_R_;
  x86::Gp out_acts_R_;
  x86::Gp a_zero_pt_R_;
  x86::Gp W_start_R_;
  x86::Gp W_end_R_;
  x86::Gp W_R_;
  x86::Gp row_offset_R_;

  // Used registers
  x86::Gp loopR1_;
  x86::Gp loopR2_;
  x86::Gp scratchReg1_;
  x86::Gp scratchReg2_;

  int nreg_;
  int vsize_;
  int paddedICPerG_;
  int paddedOCPerG_;

  const int wrIdx = 9;
};

template <inst_set_t INST_SET>
std::mutex GenConv1DKernelBase<INST_SET>::rtMutex_;

template <inst_set_t INST_SET>
CodeCache<kernel1d_sig_t, jit_conv1d_kernel_fp>
    GenConv1DKernelBase<INST_SET>::codeCache_;

void kernel_compute_1d(
    const conv_param_t<1>& conv_p,
    const uint8_t* in_acts,
    int8_t* wghts,
    int32_t* out_acts,
    int32_t a_zero_pt,
    int32_t ow_start,
    int32_t ow_end,
    int32_t* rowOffset,
    int32_t vsize,
    int32_t nreg,
    bool accum);

kernel1d_sig_t getKernel1DSig(
    const conv_param_t<1>& conv_param,
    bool isAZeroPointZero,
    bool needRowOffset,
    bool accum);

jit_conv1d_kernel_fp getOrCreateConv1DKernel(
    const conv_param_t<1>& conv_param,
    int nreg,
    int a_zero_point,
    bool needRowOffset,
    bool accum);

} // namespace fbgemm
