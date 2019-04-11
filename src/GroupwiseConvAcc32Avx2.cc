/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <asmjit/asmjit.h>
#include <cpuinfo.h>
#include <immintrin.h>
#include <array>
#include <iostream>
#include <map>
#include <stdexcept>
#include <tuple>
#include "GroupwiseConv.h"
#include "RefImplementations.h"
#include "TransposeUtils.h"
#include "fbgemm/Fbgemm.h"

namespace fbgemm {

using namespace std;

template <typename accT>
thread_local asmjit::JitRuntime GenConvKernel<accT>::rt_;

template <typename accT>
thread_local asmjit::CodeHolder GenConvKernel<accT>::code_;

template <typename accT>
thread_local std::
    map<std::tuple<bool, int, int, int, int, int, int>, jit_conv_kernel_fp>
        GenConvKernel<accT>::codeCache_;

template <typename accT>
thread_local std::
    map<std::tuple<bool, int, int, int, int, int, int>, jit_rowoffset_kernel_fp>
        GenConvKernel<accT>::codeCacheRowOffset_;

namespace x86 = asmjit::x86;

void calculateRowOffsets(
    const conv_param_t<>& conv_param,
    const uint8_t* activations,
    int32_t* rowOffsetBuf,
    int32_t a_zero_point,
    int groupNum) {
  int H_IN = conv_param.IN_DIM[0];
  int W_IN = conv_param.IN_DIM[1];
  int H_OUT = conv_param.OUT_DIM[0];
  int W_OUT = conv_param.OUT_DIM[1];

  int G = conv_param.G;
  int C_per_G = conv_param.IC / conv_param.G;
  int H_PAD = conv_param.pad[0];
  int W_PAD = conv_param.pad[1];
  // calculate row offset
  for (int h = 0; h < H_OUT; ++h) {
    for (int w = 0; w < W_OUT; ++w) {
      int32_t sum = 0;
      for (int r = 0; r < conv_param.K[0]; ++r) {
        int h_in = -H_PAD + h * conv_param.stride[0] + r;
        for (int s = 0; s < conv_param.K[1]; ++s) {
          int w_in = -W_PAD + w * conv_param.stride[1] + s;
          for (int c = 0; c < C_per_G; ++c) {
            if (h_in < 0 || h_in >= H_IN || w_in < 0 || w_in >= W_IN) {
              sum += a_zero_point;
            } else {
              sum += activations
                  [((h_in * W_IN + w_in) * G + groupNum) * C_per_G + c];
            }
          }
        }
      }
      rowOffsetBuf[h * W_OUT + w] = sum;
    }
  }
}

tuple<bool, int, int, int, int, int, int> getKernelSig(
    const conv_param_t<>& conv_param,
    bool isAZeroPointZero) {
  int C_per_G = conv_param.IC / conv_param.G;
  int K_per_G = conv_param.OC / conv_param.G;
  int bottom_edge_width = internal::getBottomEdgeWidth_(conv_param);
  int right_edge_width = internal::getRightEdgeWidth_(conv_param);
  auto kernelSig = std::make_tuple(
      isAZeroPointZero,
      conv_param.G,
      C_per_G,
      K_per_G,
      conv_param.stride[0],
      bottom_edge_width,
      right_edge_width);
  return kernelSig;
}

template <typename accT = int32_t>
jit_conv_kernel_fp getOrCreateConvKernel(
    const conv_param_t<>& conv_param,
    int a_zero_point) {
  // Note: Wrong code is generated if it's not one of the supported convolution
  assert(fbgemmOptimizedGConv<2>(conv_param));
  auto kernelSig = getKernelSig(conv_param, a_zero_point == 0);
  if (GenConvKernel<accT>::codeCache_.find(kernelSig) !=
      GenConvKernel<accT>::codeCache_.end()) {
    return GenConvKernel<accT>::codeCache_[kernelSig];
  } else {
    auto genObj = GenConvKernel<accT>(conv_param, a_zero_point);
    // TODO: Instruction set based dispatch
    return genObj.template getOrCreate<inst_set_t::avx2>(conv_param);
  }
}

template <>
template <>
void GenConvKernel<int32_t>::createVector8BitOne<inst_set_t::avx2>(
    asmjit::X86Emitter* a) {
  // create 8-bit 1s
  // i.e., oneReg16BitAvx2_[0:7] contains 0x01, oneReg8BitAvx2_[8:15] contains
  // 0x01 and so on
  a->vpcmpeqw(oneReg8BitAvx2_, oneReg8BitAvx2_, oneReg8BitAvx2_);
  a->vpabsb(oneReg8BitAvx2_, oneReg8BitAvx2_);
}

template <>
template <>
void GenConvKernel<int32_t>::createVector16BitOne<inst_set_t::avx2>(
    asmjit::X86Emitter* a) {
  // create 16-bit 1s
  // i.e., oneReg16BitAvx2_[0:15] contains 0x0001, oneReg16BitAvx2_[16:31]
  // contains 0x0001 and so on
  a->vpcmpeqw(oneReg16BitAvx2_, oneReg16BitAvx2_, oneReg16BitAvx2_);
  a->vpsrlw(oneReg16BitAvx2_, oneReg16BitAvx2_, 15);
}
template <>
template <>
void GenConvKernel<int32_t>::setToZeroPt<inst_set_t::avx2>(
    asmjit::X86Emitter* a,
    asmjit::X86Ymm destReg) {
  // make destReg all zeros
  a->vxorps(destReg, destReg, destReg);
  asmjit::X86Xmm const_reg_xmm = x86::xmm10;
  // move zero point to xmm10
  a->movq(const_reg_xmm, a_zero_pt_R_);
  // make copies of zero point
  a->vbroadcastsd(x86::ymm10, const_reg_xmm);
  // shuffle
  // overall impact is that destReg contains 32 8-bit values equal to the lower
  // 8-bits of a_zero_pt_R_
  a->vpshufb(destReg, x86::ymm10, destReg);
}

template <>
template <>
void GenConvKernel<int32_t>::genConstForPermutations<inst_set_t::avx2>(
    asmjit::X86Emitter* a) {
  asmjit::X86Gp permute_const_reg = a->gpzRef(12);
  asmjit::X86Xmm const_reg_xmm = x86::xmm10;
  // We have 1st group in even lanes and 2nd group in odd lanes.
  // Permute to put 1st group to lower 128-bit and 2nd group in upper
  // 128-bit.
  // load 7, 5, 3, 1, 6, 4, 2, 0 in a 64-bit reg
  a->mov(permute_const_reg, static_cast<asmjit::Imm>(0x0705030106040200));
  a->movq(const_reg_xmm, permute_const_reg);
  // Zero extend 8 packed 8-bit integers in the low 8 bytes of const_reg_xmm to
  // 8 packed 32-bit integers in stPermRegAvx2_
  a->vpmovzxbd(stPermRegAvx2_, const_reg_xmm);
}

template <>
template <>
void GenConvKernel<int32_t>::storeResult<inst_set_t::avx2>(
    asmjit::X86Emitter* a) {
  if (C_per_G_ == 4) {
    // store with permutation
    a->vpermd(resultRegAvx2_, stPermRegAvx2_, resultRegAvx2_);
  }
  a->vmovups(x86::dword_ptr(out_acts_R_), resultRegAvx2_);
}

template <>
template <>
void GenConvKernel<int32_t>::storeResultRowoffset<inst_set_t::avx2>(
    asmjit::X86Emitter* a,
    int offset) {
  // store
  if (C_per_G_ == 4) {
    a->vmovups(x86::dword_ptr(row_offset_R_, offset), resultRegAvx2_);
  } else if (C_per_G_ == 8) {
    // need to permute because vphaddd is used in gen8BitSumX8
    // 11 01 10 00 = 0xd8
    a->vpermq(resultRegAvx2_, resultRegAvx2_, static_cast<asmjit::Imm>(0xd8));
    a->vmovups(x86::dword_ptr(row_offset_R_, offset), resultRegAvx2_);
  } else {
    assert(C_per_G_ == 16);
    // need to permute because vphaddd is used in gen8BitSumX16
    // a[0:4] = a[0] + ... + a[15], a[4:8] = b[0] + ... + b[15]
    // a[8:12] = a[16] + ... + a[31], a[12:16] = b[16] + ... + b[31]
    a->vpermq(resultRegAvx2_, resultRegAvx2_, static_cast<asmjit::Imm>(0xd8));
    // 11 01 10 00 = 0xd8
    // a[0:4] = a[0] + ... + a[15], a[4:8] = a[16] + ... + a[31]
    // a[8:12] = b[0] + ... + b[16], a[12:16] = b[16] + ... + b[31]
    a->vpshufd(resultRegAvx2_, resultRegAvx2_, static_cast<asmjit::Imm>(0xd8));
    a->vmovups(x86::dword_ptr(row_offset_R_, offset), resultRegAvx2_);
  }
}

template <>
template <>
void GenConvKernel<int32_t>::genForLoadingWeights<inst_set_t::avx2>(
    asmjit::X86Emitter* a, int c_offset) {
  // load weights
  for (int r = 0; r < R_; ++r) {
    for (int s = 0; s < S_; ++s) {
      if (C_per_G_ == 4) {
        a->vmovaps(
            WRegs_avx2_[r * S_ + s],
            x86::dword_ptr(
                wghts_R_,
                (r * S_ + s) * 2 * K_per_G_ * C_per_G_ * sizeof(int8_t)));
      } else {
        // C_per_G == 8 or 16
        a->vmovaps(
            WRegs_avx2_[r * S_ + s],
            x86::dword_ptr(
                wghts_R_,
                (((c_offset / 4) * R_ + r) * S_ + s) * K_per_G_ * 4 *
                    sizeof(int8_t)));
      }
    }
  }
}

template <>
template <>
void GenConvKernel<int32_t>::gen8bitFMA<inst_set_t::avx2>(
    asmjit::X86Emitter* a,
    asmjit::X86Ymm aReg,
    asmjit::X86Ymm wReg) {
  a->vpmaddubsw(tmpReg1Avx2_, aReg, wReg);
  a->vpmaddwd(tmpReg1Avx2_, oneReg16BitAvx2_, tmpReg1Avx2_);
  a->vpaddd(resultRegAvx2_, tmpReg1Avx2_, resultRegAvx2_);
}

template <>
template <>
void GenConvKernel<int32_t>::gen8BitSumX4<inst_set_t::avx2>(
    asmjit::X86Emitter* a,
    asmjit::X86Ymm aReg) {
  a->vpmaddubsw(tmpReg1Avx2_, aReg, oneReg8BitAvx2_);
  a->vpmaddwd(tmpReg1Avx2_, tmpReg1Avx2_, oneReg16BitAvx2_);
  a->vpaddd(resultRegAvx2_, tmpReg1Avx2_, resultRegAvx2_);
}

template <>
template <>
void GenConvKernel<int32_t>::gen8BitSumX8<inst_set_t::avx2>(
    asmjit::X86Emitter* a,
    asmjit::X86Ymm aReg,
    asmjit::X86Ymm bReg) {
  a->vxorps(tmpReg1Avx2_, tmpReg1Avx2_, tmpReg1Avx2_);
  // Let a[0] denote 0th (LSB) 8-bit of aReg
  // After vpsadbw, a[0:2] = a[0] + ... + a[7]
  // a[8:10] = a[8] + ... + a[16]
  // a[16:18] = a[16] + ... + a[24]
  // a[24:26] = a[24] + ... + a[32]
  a->vpsadbw(aReg, aReg, tmpReg1Avx2_);
  a->vpsadbw(bReg, bReg, tmpReg1Avx2_);
  // After vphadd, a[0:4] = a[0] + ... + a[7], a[4:8] = a[8] + ... + b[15]
  // a[8:12] = b[0] + ... + b[7], a[12:16] = b[8] + ... + b[15]
  // ...
  a->vphaddd(aReg, aReg, bReg);
  a->vpaddd(resultRegAvx2_, aReg, resultRegAvx2_);
}

template <>
template <>
void GenConvKernel<int32_t>::gen8BitSumX16<inst_set_t::avx2>(
    asmjit::X86Emitter* a,
    asmjit::X86Ymm aReg,
    asmjit::X86Ymm bReg,
    asmjit::X86Ymm cReg,
    asmjit::X86Ymm dReg) {
  a->vxorps(tmpReg1Avx2_, tmpReg1Avx2_, tmpReg1Avx2_);
  // After vpsadbw, a[0:2] = a[0] + ... + a[7]
  // a[8:10] = a[8] + ... + a[15]
  // a[16:18] = a[16] + ... + a[23]
  // a[24:26] = a[24] + ... + a[31]
  a->vpsadbw(aReg, aReg, tmpReg1Avx2_);
  // 11 01 10 00 = 0xd8
  // a[0:4] = a[0] + ... + a[7], a[4:8] = a[8] + ... + a[15]
  // a[8:16] = zeros
  a->vpshufd(aReg, aReg, static_cast<asmjit::Imm>(0xd8));
  a->vpsadbw(bReg, bReg, tmpReg1Avx2_);
  // 10 00 11 01 = 0x8d
  // b[0:8] = zeros
  // b[8:12] = b[0] + ... + b[7], b[12:16] = b[8] + ... + b[15]
  a->vpshufd(bReg, bReg, static_cast<asmjit::Imm>(0x8d));
  // a[0:4] = a[0] + ... + a[7], a[4:8] = a[8] + ... + a[15]
  // a[8:12] = b[0] + ... + b[7], a[12:16] + b[8] + ... + b[15]
  a->vpaddd(aReg, aReg, bReg);

  // After vpsadbw, c[0:4] = c[0] + ... + c[7]
  // c[8:12] = c[8] + ... + c[15]
  // c[16:20] = c[16] + ... + c[23]
  // c[24:28] = c[24] + ... + c[31]
  a->vpsadbw(cReg, cReg, tmpReg1Avx2_);
  // 11 01 10 00 = 0xd8
  // c[0:4] = c[0] + ... + c[7], c[4:8] = c[8] + ... + c[15]
  // c[8:16] = zeros
  a->vpshufd(cReg, cReg, static_cast<asmjit::Imm>(0xd8));
  a->vpsadbw(dReg, dReg, tmpReg1Avx2_);
  // 10 00 11 01 = 0x8d
  // d[0:8] = zeros
  // d[8:12] = d[0] + ... + d[7], d[12:16] = d[8] + ... + d[15]
  a->vpshufd(dReg, dReg, static_cast<asmjit::Imm>(0x8d));
  // c[0:4] = c[0] + ... + c[7], c[4:8] = c[8] + ... + c[15]
  // c[8:12] = d[0] + ... + d[7], c[12:16] + d[8] + ... + d[15]
  a->vpaddd(cReg, cReg, dReg);

  // a[0:4] = a[0] + ... + a[15], a[4:8] = b[0] + ... + b[15]
  // a[8:12] = c[0] + ... + c[15], a[12:16] = d[0] + ... + d[15]
  a->vphaddd(aReg, aReg, cReg);

  a->vpaddd(resultRegAvx2_, aReg, resultRegAvx2_);
}

template <>
template <>
void GenConvKernel<int32_t>::gen8BitSum<inst_set_t::avx2>(
    asmjit::X86Emitter* a,
    int act_offset,
    bool use_scratch_reg1 /*=true*/) {
  if (use_scratch_reg1) {
    a->vmovups(
        actRegAvx2_,
        x86::dword_ptr(
            in_acts_R_, scratchReg1_, 0, act_offset * sizeof(uint8_t)));
  } else {
    a->vmovups(
        actRegAvx2_, x86::dword_ptr(in_acts_R_, act_offset * sizeof(uint8_t)));
  }
  if (C_per_G_ == 4) {
    gen8BitSumX4<inst_set_t::avx2>(a, actRegAvx2_);
  } else {
    if (use_scratch_reg1) {
      a->vmovups(
          stPermRegAvx2_,
          x86::dword_ptr(
              in_acts_R_,
              scratchReg1_,
              0,
              (act_offset + VLEN_) * sizeof(uint8_t)));
    } else {
      a->vmovups(
          stPermRegAvx2_,
          x86::dword_ptr(in_acts_R_, (act_offset + VLEN_) * sizeof(uint8_t)));
    }
    if (C_per_G_ == 8) {
      gen8BitSumX8<inst_set_t::avx2>(a, actRegAvx2_, stPermRegAvx2_);
    } else {
      assert(C_per_G_ == 16);
      if (use_scratch_reg1) {
        a->vmovups(
            WRegs_avx2_[0],
            x86::dword_ptr(
                in_acts_R_,
                scratchReg1_,
                0,
                (act_offset + 2 * VLEN_) * sizeof(uint8_t)));
        a->vmovups(
            WRegs_avx2_[1],
            x86::dword_ptr(
                in_acts_R_,
                scratchReg1_,
                0,
                (act_offset + 3 * VLEN_) * sizeof(uint8_t)));
      } else {
        a->vmovups(
            WRegs_avx2_[0],
            x86::dword_ptr(
                in_acts_R_, (act_offset + 2 * VLEN_) * sizeof(uint8_t)));
        a->vmovups(
            WRegs_avx2_[1],
            x86::dword_ptr(
                in_acts_R_, (act_offset + 3 * VLEN_) * sizeof(uint8_t)));
      }
      gen8BitSumX16<inst_set_t::avx2>(
          a, actRegAvx2_, stPermRegAvx2_, WRegs_avx2_[0], WRegs_avx2_[1]);
    } // C_per_G_ != 8
  } // C_per_G_ != 4
}

template <>
template <>
void GenConvKernel<int32_t>::genZeroPtSum<inst_set_t::avx2>(
    asmjit::X86Emitter* a,
    int multiplier) {
  a->mov(scratchReg1_, static_cast<asmjit::Imm>(multiplier));
  // tmpReg1Avx2_ also uses xmm11
  asmjit::X86Xmm const_reg_xmm = x86::xmm11;
  a->movq(const_reg_xmm, scratchReg1_);
  a->vpbroadcastd(tmpReg1Avx2_, const_reg_xmm);
  a->vpmulld(tmpReg1Avx2_, zeroPTRegAvx2_, tmpReg1Avx2_);
  a->vpaddd(resultRegAvx2_, tmpReg1Avx2_, resultRegAvx2_);
}

template <>
template <>
void GenConvKernel<int32_t>::genForTopEdge<inst_set_t::avx2>(
    asmjit::X86Emitter* a, int c_offset) {
  // Save the original in_acts_R_ in row_offset_R_
  a->mov(row_offset_R_, in_acts_R_);

  // top-left corner code
  if (c_offset == 0) {
    // zero out the results register
    a->vxorps(resultRegAvx2_, resultRegAvx2_, resultRegAvx2_);
  } else {
    a->vmovups(resultRegAvx2_, x86::dword_ptr(out_acts_R_));
  }
  for (int r = 0; r < R_; ++r) {
    int h_in = -H_PAD_ + r;
    if (h_in >= 0) {
      a->imul(
          scratchReg1_,
          W_in_R_,
          static_cast<asmjit::Imm>(h_in * C_ * sizeof(uint8_t)));
    }
    for (int s = 0; s < S_; ++s) {
      int w_in = -W_PAD_ + s;
      if (h_in >= 0 && w_in >= 0) {
        if (C_per_G_ == 4) {
          a->vbroadcastsd(
              actRegAvx2_,
              x86::dword_ptr(
                  in_acts_R_, scratchReg1_, 0, w_in * C_ * sizeof(uint8_t)));
        } else {
          a->vbroadcastss(
              actRegAvx2_,
              x86::word_ptr(
                  in_acts_R_,
                  scratchReg1_,
                  0,
                  (w_in * C_ + c_offset) * sizeof(uint8_t)));
        }
        gen8bitFMA<inst_set_t::avx2>(a, actRegAvx2_, WRegs_avx2_[r * S_ + s]);
      } else {
        if (!isAZeroPointZero_) {
          gen8bitFMA<inst_set_t::avx2>(
              a, zeroPTRegAvx2_, WRegs_avx2_[r * S_ + s]);
        }
      }
    }
  }
  storeResult<inst_set_t::avx2>(a);

  a->add(out_acts_R_, static_cast<asmjit::Imm>(K_ * sizeof(int32_t)));

  // top edge excluding corners
  a->imul(scratchReg2_, W_in_R_, C_);
  asmjit::Label LoopTopEdge = a->newLabel();
  a->mov(loopR1_, W_out_R_);
  a->sub(loopR1_, left_edge_width_ + right_edge_width_);
  a->bind(LoopTopEdge);
  // zero out
  if (c_offset == 0) {
    a->vxorps(resultRegAvx2_, resultRegAvx2_, resultRegAvx2_);
  } else {
    a->vmovups(resultRegAvx2_, x86::dword_ptr(out_acts_R_));
  }
  if (!isAZeroPointZero_) {
    for (int r = 0; r < H_PAD_; ++r) {
      for (int s = 0; s < S_; ++s) {
        gen8bitFMA<inst_set_t::avx2>(a, zeroPTRegAvx2_, WRegs_avx2_[s]);
      }
    }
  }
  a->mov(scratchReg1_, 0);
  for (int r = H_PAD_; r < R_; ++r) {
    for (int s = 0; s < S_; ++s) {
      if (C_per_G_ == 4) {
        a->vbroadcastsd(
            actRegAvx2_,
            x86::dword_ptr(
                in_acts_R_, scratchReg1_, 0, (stride_ - W_PAD_ + s) * C_));
      } else {
        a->vbroadcastss(
            actRegAvx2_,
            x86::word_ptr(
                in_acts_R_,
                scratchReg1_,
                0,
                (stride_ - W_PAD_ + s) * C_ + c_offset));
      }
      gen8bitFMA<inst_set_t::avx2>(a, actRegAvx2_, WRegs_avx2_[r * S_ + s]);
    }
    a->add(scratchReg1_, scratchReg2_);
  }
  a->add(in_acts_R_, static_cast<asmjit::Imm>(stride_ * C_ * sizeof(uint8_t)));

  storeResult<inst_set_t::avx2>(a);

  a->add(out_acts_R_, static_cast<asmjit::Imm>(K_ * sizeof(int32_t)));
  a->dec(loopR1_);
  a->jg(LoopTopEdge);
  // Restore the original in_acts_R_
  a->mov(in_acts_R_, row_offset_R_);

  // top-right corner code
  if (right_edge_width_ > 0) {
    // zero out
    if (c_offset == 0) {
      a->vxorps(resultRegAvx2_, resultRegAvx2_, resultRegAvx2_);
    } else {
      a->vmovups(resultRegAvx2_, x86::dword_ptr(out_acts_R_));
    }
    if (!isAZeroPointZero_) {
      for (int r = 0; r < H_PAD_; ++r) {
        for (int s = 0; s < S_; ++s) {
          gen8bitFMA<inst_set_t::avx2>(
              a, zeroPTRegAvx2_, WRegs_avx2_[r * S_ + s]);
        }
      }
    }
    for (int r = H_PAD_; r < R_; ++r) {
      int h_in = -H_PAD_ + r;
      a->imul(scratchReg1_, W_in_R_, (h_in + 1) * C_);
      for (int s = 0; s < S_ - W_PAD_; ++s) {
        if (C_per_G_ == 4) {
          a->vbroadcastsd(
              actRegAvx2_,
              x86::dword_ptr(
                  in_acts_R_, scratchReg1_, 0, (W_PAD_ + s - S_) * C_));
        } else {
          a->vbroadcastss(
              actRegAvx2_,
              x86::word_ptr(
                  in_acts_R_,
                  scratchReg1_,
                  0,
                  (W_PAD_ + s - S_) * C_ + c_offset));
        }
        gen8bitFMA<inst_set_t::avx2>(a, actRegAvx2_, WRegs_avx2_[r * S_ + s]);
      }
      if (!isAZeroPointZero_) {
        for (int s = S_ - W_PAD_; s < S_; ++s) {
          gen8bitFMA<inst_set_t::avx2>(
              a, zeroPTRegAvx2_, WRegs_avx2_[r * S_ + s]);
        }
      }
    }
    storeResult<inst_set_t::avx2>(a);
    a->add(out_acts_R_, static_cast<asmjit::Imm>(K_ * sizeof(int32_t)));
  }

  // reset output activation pointer
  a->imul(scratchReg1_, W_out_R_, K_ * sizeof(int32_t));
  a->sub(out_acts_R_, scratchReg1_);
}

template <>
template <>
void GenConvKernel<int32_t>::genForLeftEdge<inst_set_t::avx2>(
    asmjit::X86Emitter* a, int c_offset) {
  // Save the original in_acts_R_ in row_offset_R_
  a->mov(row_offset_R_, in_acts_R_);

  a->imul(scratchReg1_, W_in_R_, (top_edge_width_ * stride_ - H_PAD_) * C_);
  a->add(in_acts_R_, scratchReg1_);

  a->imul(scratchReg2_, W_in_R_, C_);

  // left edge excluding corners
  asmjit::Label LoopLeftEdge = a->newLabel();
  // loopR1_ corresponds to the output row
  a->mov(loopR1_, H_out_R_);
  a->sub(loopR1_, top_edge_width_ + bottom_edge_width_);
  a->imul(loopR2_, W_out_R_, K_ * sizeof(int32_t));
  a->bind(LoopLeftEdge);
  a->add(out_acts_R_, loopR2_);
  if (c_offset == 0) {
    // zero out
    a->vxorps(resultRegAvx2_, resultRegAvx2_, resultRegAvx2_);
  } else {
    a->vmovups(resultRegAvx2_, x86::dword_ptr(out_acts_R_));
  }
  a->mov(scratchReg1_, 0);
  for (int r = 0; r < R_; ++r) {
    if (!isAZeroPointZero_) {
      for (int s = 0; s < W_PAD_; ++s) {
        gen8bitFMA<inst_set_t::avx2>(
            a, zeroPTRegAvx2_, WRegs_avx2_[r * S_ + s]);
      }
    }
    for (int s = W_PAD_; s < S_; ++s) {
      if (C_per_G_ == 4) {
        a->vbroadcastsd(
            actRegAvx2_,
            x86::dword_ptr(
                in_acts_R_,
                scratchReg1_,
                0,
                (s - W_PAD_) * C_ * sizeof(uint8_t)));
      } else {
        a->vbroadcastss(
            actRegAvx2_,
            x86::word_ptr(
                in_acts_R_,
                scratchReg1_,
                0,
                ((s - W_PAD_) * C_ + c_offset) * sizeof(uint8_t)));
      }
      gen8bitFMA<inst_set_t::avx2>(a, actRegAvx2_, WRegs_avx2_[r * S_ + s]);
    }
    a->add(scratchReg1_, scratchReg2_);
  }
  for (int i = 0; i < stride_; ++i) {
    a->add(in_acts_R_, scratchReg2_);
  }
  storeResult<inst_set_t::avx2>(a);

  a->dec(loopR1_);
  a->jg(LoopLeftEdge);

  // reset output activation pointer
  a->mov(scratchReg2_, H_out_R_);
  a->sub(scratchReg2_, top_edge_width_ + bottom_edge_width_);
  a->imul(scratchReg2_, W_out_R_);
  a->imul(scratchReg2_, static_cast<asmjit::Imm>(K_ * sizeof(int32_t)));
  a->sub(out_acts_R_, scratchReg2_);

  // reset input activation pointer
  a->mov(in_acts_R_, row_offset_R_);
}

template <>
template <>
void GenConvKernel<int32_t>::genForRightEdge<inst_set_t::avx2>(
    asmjit::X86Emitter* a, int c_offset) {
  // Save the original in_acts_R_ in row_offset_R_
  a->mov(row_offset_R_, in_acts_R_);

  a->imul(scratchReg1_, W_in_R_, (top_edge_width_ * stride_ - H_PAD_) * C_);
  a->imul(scratchReg2_, W_out_R_, stride_ * C_);
  a->add(scratchReg1_, scratchReg2_);
  a->add(in_acts_R_, scratchReg1_);

  // right edge excluding corners
  asmjit::Label LoopRightEdge = a->newLabel();

  // move output pointer to the top right edge
  // (W_ + W_ - 1)*K_*sizeof(int32_t)
  a->mov(scratchReg2_, W_out_R_);
  a->imul(scratchReg2_, top_edge_width_ + 1);
  a->sub(scratchReg2_, right_edge_width_);
  a->imul(scratchReg2_, static_cast<asmjit::Imm>(K_ * sizeof(int32_t)));
  a->add(out_acts_R_, scratchReg2_);

  a->imul(scratchReg2_, W_in_R_, C_);

  // loopR1_ corresponds to the output row
  a->mov(loopR1_, H_out_R_);
  a->sub(loopR1_, top_edge_width_ + bottom_edge_width_);
  a->imul(loopR2_, W_out_R_, K_ * sizeof(int32_t));
  a->bind(LoopRightEdge);
  if (c_offset == 0) {
    // zero out
    a->vxorps(resultRegAvx2_, resultRegAvx2_, resultRegAvx2_);
  } else {
    a->vmovups(resultRegAvx2_, x86::dword_ptr(out_acts_R_));
  }

  a->mov(scratchReg1_, 0);
  for (int r = 0; r < R_; ++r) {
    for (int s = 0; s < S_ - W_PAD_; ++s) {
      if (C_per_G_ == 4) {
        a->vbroadcastsd(
            actRegAvx2_,
            x86::dword_ptr(
                in_acts_R_,
                scratchReg1_,
                0,
                (s - right_edge_width_ * stride_ - W_PAD_) * C_));
      } else {
        a->vbroadcastss(
            actRegAvx2_,
            x86::word_ptr(
                in_acts_R_,
                scratchReg1_,
                0,
                (s - right_edge_width_ * stride_ - W_PAD_) * C_ + c_offset));
      }
      gen8bitFMA<inst_set_t::avx2>(a, actRegAvx2_, WRegs_avx2_[r * S_ + s]);
    }
    if (!isAZeroPointZero_) {
      for (int s = S_ - W_PAD_; s < S_; ++s) {
        gen8bitFMA<inst_set_t::avx2>(
            a, zeroPTRegAvx2_, WRegs_avx2_[r * S_ + s]);
      }
    }

    a->add(scratchReg1_, scratchReg2_);
  }
  for (int i = 0; i < stride_; ++i) {
    a->add(in_acts_R_, scratchReg2_);
  }

  // storeResult<inst_set_t::avx2>(a, (W_+W_-1)*K_*sizeof(int32_t));
  storeResult<inst_set_t::avx2>(a);

  a->add(out_acts_R_, loopR2_);

  a->dec(loopR1_);
  a->jg(LoopRightEdge);

  // reset base
  a->mov(scratchReg2_, W_out_R_);
  a->imul(scratchReg2_, top_edge_width_ + 1);
  a->sub(scratchReg2_, right_edge_width_);
  a->imul(scratchReg2_, static_cast<asmjit::Imm>(K_ * sizeof(int32_t)));
  a->sub(out_acts_R_, scratchReg2_);

  //(H_out - 2*H_PAD_)*W_*K_*sizeof(int32_t)
  a->mov(scratchReg2_, H_out_R_);
  a->sub(scratchReg2_, top_edge_width_ + bottom_edge_width_);
  a->imul(scratchReg2_, W_out_R_);
  a->imul(scratchReg2_, static_cast<asmjit::Imm>(K_ * sizeof(int32_t)));
  a->sub(out_acts_R_, scratchReg2_);
  // a->sub(out_acts_R_, (H_out - 2*H_PAD_)*W_out*K_*sizeof(int32_t));

  // reset input activation pointer
  a->mov(in_acts_R_, row_offset_R_);
}

template <>
template <>
void GenConvKernel<int32_t>::genForBottomEdge<inst_set_t::avx2>(
    asmjit::X86Emitter* a, int c_offset) {
  // bottom-left corner
  // we are updating the last row
  a->mov(scratchReg2_, H_out_R_);
  a->sub(scratchReg2_, bottom_edge_width_);
  a->imul(scratchReg1_, scratchReg2_, K_ * sizeof(int32_t));
  a->imul(scratchReg1_, W_out_R_);
  a->add(out_acts_R_, scratchReg1_);
  if (c_offset == 0) {
    // zero out
    a->vxorps(resultRegAvx2_, resultRegAvx2_, resultRegAvx2_);
  } else {
    a->vmovups(resultRegAvx2_, x86::dword_ptr(out_acts_R_));
  }

  // adjust input pointer
  a->imul(scratchReg1_, scratchReg2_, stride_);
  a->sub(scratchReg1_, H_PAD_);
  a->imul(scratchReg1_, W_in_R_);
  a->imul(scratchReg1_, static_cast<asmjit::Imm>(C_ * sizeof(uint8_t)));
  a->add(in_acts_R_, scratchReg1_);
  // Save the original in_acts_R_ after adjustment in row_offset_R_
  a->mov(row_offset_R_, in_acts_R_);

  for (int r = 0; r < R_ - H_PAD_; ++r) {
    a->imul(scratchReg1_, W_in_R_, static_cast<asmjit::Imm>(r * C_));
    if (!isAZeroPointZero_) {
      for (int s = 0; s < W_PAD_; ++s) {
        gen8bitFMA<inst_set_t::avx2>(
            a, zeroPTRegAvx2_, WRegs_avx2_[r * S_ + s]);
      }
    }
    for (int s = W_PAD_; s < S_; ++s) {
      if (C_per_G_ == 4) {
        a->vbroadcastsd(
            actRegAvx2_,
            x86::dword_ptr(
                in_acts_R_,
                scratchReg1_,
                0,
                (s - W_PAD_) * C_ * sizeof(uint8_t)));
      } else {
        a->vbroadcastss(
            actRegAvx2_,
            x86::word_ptr(
                in_acts_R_,
                scratchReg1_,
                0,
                ((s - W_PAD_) * C_ + c_offset) * sizeof(uint8_t)));
      }
      gen8bitFMA<inst_set_t::avx2>(a, actRegAvx2_, WRegs_avx2_[r * S_ + s]);
    }
  }
  if (!isAZeroPointZero_) {
    for (int r = R_ - H_PAD_; r < R_; ++r) {
      for (int s = 0; s < S_; ++s) {
        gen8bitFMA<inst_set_t::avx2>(
            a, zeroPTRegAvx2_, WRegs_avx2_[r * S_ + s]);
      }
    }
  }

  // storeResult<inst_set_t::avx2>(a, (H_-1)*W_*K_*sizeof(int32_t));
  storeResult<inst_set_t::avx2>(a);
  a->add(out_acts_R_, static_cast<asmjit::Imm>(K_ * sizeof(int32_t)));

  // bottom edge excluding corners
  a->imul(scratchReg2_, W_in_R_, static_cast<asmjit::Imm>(C_));
  asmjit::Label LoopBottomEdge = a->newLabel();
  a->mov(loopR1_, W_out_R_);
  a->sub(loopR1_, left_edge_width_ + right_edge_width_);
  a->bind(LoopBottomEdge);
  if (c_offset == 0) {
    // zero out
    a->vxorps(resultRegAvx2_, resultRegAvx2_, resultRegAvx2_);
  } else {
    a->vmovups(resultRegAvx2_, x86::dword_ptr(out_acts_R_));
  }
  a->mov(scratchReg1_, 0);
  for (int r = 0; r < R_ - H_PAD_; ++r) {
    // int h_in = H_-2*H_PAD_ + r;
    for (int s = 0; s < S_; ++s) {
      if (C_per_G_ == 4) {
        a->vbroadcastsd(
            actRegAvx2_,
            x86::dword_ptr(
                in_acts_R_, scratchReg1_, 0, (stride_ - W_PAD_ + s) * C_));
      } else {
        a->vbroadcastss(
            actRegAvx2_,
            x86::word_ptr(
                in_acts_R_,
                scratchReg1_,
                0,
                (stride_ - W_PAD_ + s) * C_ + c_offset));
      }
      gen8bitFMA<inst_set_t::avx2>(a, actRegAvx2_, WRegs_avx2_[r * S_ + s]);
    }
    a->add(scratchReg1_, scratchReg2_);
  }

  if (!isAZeroPointZero_) {
    for (int r = R_ - H_PAD_; r < R_; ++r) {
      for (int s = 0; s < S_; ++s) {
        gen8bitFMA<inst_set_t::avx2>(
            a, zeroPTRegAvx2_, WRegs_avx2_[r * S_ + s]);
      }
    }
  }

  a->add(in_acts_R_, static_cast<asmjit::Imm>(stride_ * C_ * sizeof(uint8_t)));
  // storeResult<inst_set_t::avx2>(a, ((H_-1)*W_+1)*K_*sizeof(int32_t));
  storeResult<inst_set_t::avx2>(a);

  a->add(out_acts_R_, static_cast<asmjit::Imm>(K_ * sizeof(int32_t)));
  a->dec(loopR1_);
  a->jg(LoopBottomEdge);
  // Restore the original in_acts_R_
  a->mov(in_acts_R_, row_offset_R_);

  // bottom-right corner
  if (right_edge_width_ > 0) {
    if (c_offset == 0) {
      // zero out
      a->vxorps(resultRegAvx2_, resultRegAvx2_, resultRegAvx2_);
    } else {
      a->vmovups(resultRegAvx2_, x86::dword_ptr(out_acts_R_));
    }
    // input start point
    a->mov(scratchReg1_, W_out_R_);
    a->imul(scratchReg1_, stride_);
    a->imul(scratchReg1_, static_cast<asmjit::Imm>(C_ * sizeof(uint8_t)));
    a->imul(scratchReg2_, W_in_R_, C_ * sizeof(uint8_t));

    for (int r = 0; r < R_ - H_PAD_; ++r) {
      for (int s = 0; s < S_ - W_PAD_; ++s) {
        if (C_per_G_ == 4) {
          a->vbroadcastsd(
              actRegAvx2_,
              x86::dword_ptr(
                  in_acts_R_,
                  scratchReg1_,
                  0,
                  (s - right_edge_width_ * stride_ - W_PAD_) * C_));
        } else {
          a->vbroadcastss(
              actRegAvx2_,
              x86::word_ptr(
                  in_acts_R_,
                  scratchReg1_,
                  0,
                  (s - right_edge_width_ * stride_ - W_PAD_) * C_ + c_offset));
        }
        gen8bitFMA<inst_set_t::avx2>(a, actRegAvx2_, WRegs_avx2_[r * S_ + s]);
      }
      a->add(scratchReg1_, scratchReg2_);
      if (!isAZeroPointZero_) {
        for (int s = S_ - W_PAD_; s < S_; ++s) {
          gen8bitFMA<inst_set_t::avx2>(
              a, zeroPTRegAvx2_, WRegs_avx2_[r * S_ + s]);
        }
      }
    }

    if (!isAZeroPointZero_) {
      for (int r = R_ - H_PAD_; r < R_; ++r) {
        for (int s = 0; s < S_; ++s) {
          gen8bitFMA<inst_set_t::avx2>(
              a, zeroPTRegAvx2_, WRegs_avx2_[r * S_ + s]);
        }
      }
    }

    storeResult<inst_set_t::avx2>(a);
    // storeResult<inst_set_t::avx2>(a, ((H_-1)*W_+W_-1)*K_*sizeof(int32_t));
    a->add(out_acts_R_, static_cast<asmjit::Imm>(K_ * sizeof(int32_t)));
  } // right_edge_width_ > 0

  // reset input pointer
  a->mov(scratchReg2_, H_out_R_);
  a->sub(scratchReg2_, bottom_edge_width_);
  a->imul(scratchReg1_, scratchReg2_, stride_);
  a->sub(scratchReg1_, H_PAD_);
  a->imul(scratchReg1_, W_in_R_);
  a->imul(scratchReg1_, static_cast<asmjit::Imm>(C_ * sizeof(uint8_t)));
  a->sub(in_acts_R_, scratchReg1_);

  // reset output pointer
  a->imul(scratchReg2_, W_out_R_);
  a->add(scratchReg2_, W_out_R_);
  a->imul(scratchReg2_, static_cast<asmjit::Imm>(K_ * sizeof(int32_t)));
  a->sub(out_acts_R_, scratchReg2_);
}

template <>
template <>
void GenConvKernel<int32_t>::genCoreInsts<inst_set_t::avx2>(
    asmjit::X86Emitter* a, int c_offset) {
  // main compute
  asmjit::Label LoopH = a->newLabel();
  asmjit::Label LoopW = a->newLabel();
  // base for output
  a->imul(scratchReg2_, W_out_R_, top_edge_width_);
  a->add(scratchReg2_, left_edge_width_);
  a->imul(scratchReg2_, K_ * sizeof(int32_t));
  a->add(out_acts_R_, scratchReg2_);

  // base for input
  if (stride_ > 1) {
    a->imul(scratchReg1_, W_in_R_, top_edge_width_ * stride_ - H_PAD_);
    a->imul(scratchReg1_, C_);
    a->add(in_acts_R_, scratchReg1_);

    // Save the original in_acts_R_ in row_offset_R_
    a->mov(row_offset_R_, in_acts_R_);
  }

  a->imul(scratchReg2_, W_in_R_, C_ * sizeof(uint8_t));

  // H loop w.r.t. output image
  a->mov(loopR1_, H_out_R_);
  a->sub(loopR1_, top_edge_width_ + bottom_edge_width_);
  a->bind(LoopH);
  // W loop w.r.t. output image
  a->mov(loopR2_, W_out_R_);
  a->sub(loopR2_, left_edge_width_ + right_edge_width_);
  a->bind(LoopW);
  if (c_offset == 0) {
    // zero out
    a->vxorps(resultRegAvx2_, resultRegAvx2_, resultRegAvx2_);
  } else {
    a->vmovups(resultRegAvx2_, x86::dword_ptr(out_acts_R_));
  }
  // compute on all filters
  for (int r = 0; r < R_; ++r) {
    for (int s = 0; s < S_; ++s) {
      if (C_per_G_ == 4) {
        a->vbroadcastsd(
            actRegAvx2_,
            x86::dword_ptr(
                in_acts_R_, (left_edge_width_ * stride_ - W_PAD_ + s) * C_));
      } else {
        a->vbroadcastss(
            actRegAvx2_,
            x86::word_ptr(
                in_acts_R_,
                ((left_edge_width_ * stride_ - W_PAD_ + s) * C_ + c_offset)));
      }
      gen8bitFMA<inst_set_t::avx2>(a, actRegAvx2_, WRegs_avx2_[r * S_ + s]);
    }
    // advance input pointer by one row
    a->add(in_acts_R_, scratchReg2_);
  }
  // rewind input pointer
  a->imul(scratchReg1_, W_in_R_, R_ * C_ * sizeof(uint8_t));
  a->sub(in_acts_R_, scratchReg1_);
  // a->add(scratchReg1_, C_*sizeof(uint8_t));
  // advance input pointer by one output pixel
  a->add(in_acts_R_, static_cast<asmjit::Imm>(stride_ * C_ * sizeof(uint8_t)));

  // storeResult<inst_set_t::avx2>(a, (W_+1)*K_*sizeof(int32_t));
  storeResult<inst_set_t::avx2>(a);

  // advance output pointer by one pixel
  a->add(out_acts_R_, static_cast<asmjit::Imm>(K_ * sizeof(int32_t)));

  a->dec(loopR2_);
  a->jg(LoopW);
  if (stride_ == 1) {
    a->add(in_acts_R_, (left_edge_width_ + bottom_edge_width_) * C_);
  } else {
    a->imul(scratchReg1_, W_in_R_, stride_ * C_);
    a->add(row_offset_R_, scratchReg1_);
    a->mov(in_acts_R_, row_offset_R_);
  }
  // a->sub(in_acts_R_, (W_ - 2*W_PAD_)*C_*sizeof(uint8_t));
  // a->add(in_acts_R_, W_*C_*sizeof(uint8_t));
  // advance output pointer by padding size
  a->add(
      out_acts_R_,
      (left_edge_width_ + right_edge_width_) * K_ * sizeof(int32_t));
  // a->sub(out_acts_R_, (W_ - 2*W_PAD_)*K_*sizeof(int32_t));
  // a->add(out_acts_R_, W_*K_*sizeof(int32_t));

  a->dec(loopR1_);
  a->jg(LoopH);

  // Now, loopR1 has the number of output rows we processed
  if (c_offset + 4 < C_per_G_) {
    a->mov(loopR1_, H_out_R_);
    a->sub(loopR1_, top_edge_width_ + bottom_edge_width_);

    // reset input pointer
    a->imul(scratchReg2_, loopR1_, stride_ * C_);
    // NOTE: 3-operand form a->imul(scratchReg2_, scratchReg2_, W_in_R_) doesn't
    // work because it requires the 3rd operand to be an immediate.
    a->imul(scratchReg2_, W_in_R_);
    if (stride_ == 1) {
      a->sub(in_acts_R_, scratchReg2_);
    } else {
      a->sub(row_offset_R_, scratchReg2_);
      a->mov(in_acts_R_, row_offset_R_);

      a->imul(scratchReg1_, W_in_R_, top_edge_width_ * stride_ - H_PAD_);
      a->imul(scratchReg1_, C_);
      a->sub(in_acts_R_, scratchReg1_);
    }

    // reset output pointer
    assert(K_ == C_);
    if (stride_ == 1 && 2 * H_PAD_ - R_ == -1) {
      a->imul(scratchReg2_, static_cast<asmjit::Imm>(sizeof(int32_t)));
    } else {
      assert(stride_ == 2);
      a->imul(scratchReg2_, loopR1_, C_ * sizeof(int32_t));
      a->imul(scratchReg2_, W_out_R_);
    }
    a->sub(out_acts_R_, scratchReg2_);

    a->imul(scratchReg2_, W_out_R_, top_edge_width_);
    a->add(scratchReg2_, left_edge_width_);
    a->imul(scratchReg2_, K_ * sizeof(int32_t));
    a->sub(out_acts_R_, scratchReg2_);
  }
}

template <>
template <>
jit_conv_kernel_fp GenConvKernel<int32_t>::getOrCreate<inst_set_t::avx2>(
    const conv_param_t<>& conv_param) {
  code_.reset(false);
  code_.init(rt_.getCodeInfo());
  asmjit::X86Assembler assembler(&code_);
  asmjit::X86Emitter* a = assembler.asEmitter();

#if defined(FBGEMM_LOG_CODE)
  // log code to a file
  FILE* codeLogfile =
      fopen(getCodeLoggingFile<inst_set_t::avx2>(false).c_str(), "w");
  asmjit::FileLogger* codeLogger = new asmjit::FileLogger(codeLogfile);
  if (codeLogger) {
    code_.setLogger(codeLogger);
  }
#endif

  // arguments to the function created
  in_acts_R_ = a->zdi();
  wghts_R_ = a->zsi();
  out_acts_R_ = a->zdx();
  a_zero_pt_R_ = a->zcx();
  H_out_R_ = a->gpzRef(8); // We get H_in but will convert to H_out soon
  W_in_R_ = a->gpzRef(9);
  row_offset_R_ = a->gpzRef(10);
  W_out_R_ = a->gpzRef(11);

  // register for temporary use
  scratchReg1_ = a->gpzRef(12);
  scratchReg2_ = a->gpzRef(13);

  asmjit::FuncDetail func;
  func.init(asmjit::FuncSignature6<
            void,
            uint8_t*,
            int8_t*,
            int32_t*,
            int32_t,
            int32_t,
            int32_t>(asmjit::CallConv::kIdHost));

  asmjit::FuncFrameInfo ffi;
  ffi.setDirtyRegs(
      asmjit::X86Reg::kKindVec,
      asmjit::Utils::mask(0, 1, 2, 3, 4, 5, 6, 7) |
          asmjit::Utils::mask(8, 9, 10, 11, 12, 13, 14, 15));
  ffi.setDirtyRegs(
      asmjit::X86Reg::kKindGp, asmjit::Utils::mask(10, 11, 12, 13, 14, 15));

  asmjit::FuncArgsMapper args(&func);
  args.assignAll(
      in_acts_R_, wghts_R_, out_acts_R_, a_zero_pt_R_, H_out_R_, W_in_R_);

  args.updateFrameInfo(ffi);

  asmjit::FuncFrameLayout layout;
  layout.init(func, ffi);

  asmjit::FuncUtils::emitProlog(a, layout);
  asmjit::FuncUtils::allocArgs(a, layout, args);

  createVector16BitOne<inst_set_t::avx2>(a);

  loopR1_ = a->gpzRef(14);
  loopR2_ = a->gpzRef(15);

  if (!isAZeroPointZero_) {
    setToZeroPt<inst_set_t::avx2>(a, zeroPTRegAvx2_);
  }

  genConstForPermutations<inst_set_t::avx2>(a);

  // compute H_out_R_ and W_out_R_
  // h_out = (h_in + 2*pad - kernel) / stride + 1
  a->mov(W_out_R_, W_in_R_);
  if (stride_ > 1 || 2 * W_PAD_ - S_ != -1) {
    assert(stride_ == 2);
    a->add(H_out_R_, 2 * H_PAD_ - R_);
    a->shr(H_out_R_, 1);
    a->add(H_out_R_, 1);

    a->add(W_out_R_, 2 * W_PAD_ - S_);
    a->shr(W_out_R_, 1);
    a->add(W_out_R_, 1);
  }

  // The invariants of gen* functions used in the following c-loop is that
  // they should preserve that both the input and output pointer points to the
  // first pixel of input and output image.

  // Work on 4 input channels at a time.
  // The minimum unit should be 4 because instruction sequence in gen8bitFMA
  // reduces 4 inputs.
  // We can't work on more than 4 input channels because of we can't put too
  // many weights in register (we need R S K 4 / 32 registers to store
  // weights for 4 input channels).
  for (int c = 0; c < C_per_G_; c += 4) {
    genForLoadingWeights<inst_set_t::avx2>(a, c);

    genForTopEdge<inst_set_t::avx2>(a, c);
    genForLeftEdge<inst_set_t::avx2>(a, c);
    if (right_edge_width_ > 0) {
      genForRightEdge<inst_set_t::avx2>(a, c);
    }
    if (bottom_edge_width_ > 0) {
      genForBottomEdge<inst_set_t::avx2>(a, c);
    }

    genCoreInsts<inst_set_t::avx2>(a, c);
  }

  asmjit::FuncUtils::emitEpilog(a, layout);

  jit_conv_kernel_fp fn;
  asmjit::Error err = rt_.add(&fn, &code_);
  if (err) {
    std::cout << "Error: in fn add" << std::endl;
    return nullptr;
  }
  auto kernelSig = getKernelSig(conv_param, isAZeroPointZero_);
  codeCache_[kernelSig] = fn;

#if defined(FBGEMM_LOG_CODE)
  fclose(codeLogfile);
  delete codeLogger;
#endif

  return fn;
}

template <>
template <>
void GenConvKernel<int32_t>::genForTopEdgeRowoffset<inst_set_t::avx2>(
    asmjit::X86Emitter* a) {
  // Save the original in_acts_R_ in wghts_R_
  a->mov(wghts_R_, in_acts_R_);

  // top-left corner code
  // zero out the results register
  a->vxorps(resultRegAvx2_, resultRegAvx2_, resultRegAvx2_);
  if (!isAZeroPointZero_) {
    genZeroPtSum<inst_set_t::avx2>(a, R_ * S_ - (R_ - H_PAD_) * (S_ - W_PAD_));
  }
  for (int r = H_PAD_; r < R_; ++r) {
    int h_in = -H_PAD_ + r;
    a->imul(
        scratchReg1_,
        W_in_R_,
        static_cast<asmjit::Imm>(h_in * C_ * sizeof(uint8_t)));
    for (int s = W_PAD_; s < S_; ++s) {
      int w_in = -W_PAD_ + s;
      gen8BitSum<inst_set_t::avx2>(a, w_in * C_);
    }
  }

  // store results
  storeResultRowoffset<inst_set_t::avx2>(a);

  // for C_per_G == 4 and K_per_G == 4, 8 groups processed at a time
  a->add(row_offset_R_, static_cast<asmjit::Imm>(8 * sizeof(int32_t)));

  // top edge excluding corners
  a->imul(scratchReg2_, W_in_R_, C_);
  asmjit::Label LoopTopEdge = a->newLabel();
  a->mov(loopR1_, W_out_R_);
  a->sub(loopR1_, left_edge_width_ + right_edge_width_);
  a->bind(LoopTopEdge);
  // zero out
  a->vxorps(resultRegAvx2_, resultRegAvx2_, resultRegAvx2_);
  if (!isAZeroPointZero_) {
    genZeroPtSum<inst_set_t::avx2>(a, H_PAD_ * S_);
  }
  a->mov(scratchReg1_, 0);
  for (int r = H_PAD_; r < R_; ++r) {
    for (int s = 0; s < S_; ++s) {
      gen8BitSum<inst_set_t::avx2>(a, (stride_ - W_PAD_ + s) * C_);
    }
    a->add(scratchReg1_, scratchReg2_);
  }
  a->add(in_acts_R_, static_cast<asmjit::Imm>(stride_ * C_ * sizeof(uint8_t)));

  // store results
  storeResultRowoffset<inst_set_t::avx2>(a);

  a->add(row_offset_R_, static_cast<asmjit::Imm>(8 * sizeof(int32_t)));
  a->dec(loopR1_);
  a->jg(LoopTopEdge);
  // Restore the original in_acts_R_
  a->mov(in_acts_R_, wghts_R_);

  // top-right corner code
  if (right_edge_width_ > 0) {
    // zero out
    a->vxorps(resultRegAvx2_, resultRegAvx2_, resultRegAvx2_);
    if (!isAZeroPointZero_) {
      genZeroPtSum<inst_set_t::avx2>(
          a, R_ * S_ - (R_ - H_PAD_) * (S_ - W_PAD_));
    }
    for (int r = H_PAD_; r < R_; ++r) {
      int h_in = -H_PAD_ + r;
      a->imul(scratchReg1_, W_in_R_, (h_in + 1) * C_);
      for (int s = 0; s < S_ - W_PAD_; ++s) {
        gen8BitSum<inst_set_t::avx2>(a, (W_PAD_ + s - S_) * C_);
      }
    }

    // store results
    storeResultRowoffset<inst_set_t::avx2>(a);
    a->add(row_offset_R_, static_cast<asmjit::Imm>(8 * sizeof(int32_t)));
  }

  // reset output pointer
  a->imul(scratchReg1_, W_out_R_, 8 * sizeof(int32_t));
  a->sub(row_offset_R_, scratchReg1_);
}

template <>
template <>
void GenConvKernel<int32_t>::genForLeftEdgeRowoffset<inst_set_t::avx2>(
    asmjit::X86Emitter* a) {
  // Save the original in_acts_R_ in wghts_R_
  a->mov(wghts_R_, in_acts_R_);

  a->imul(scratchReg1_, W_in_R_, (top_edge_width_ * stride_ - H_PAD_) * C_);
  a->add(in_acts_R_, scratchReg1_);

  a->imul(scratchReg2_, W_in_R_, C_);

  // left edge excluding corners
  asmjit::Label LoopLeftEdge = a->newLabel();
  // loopR1_ corresponds to the output row
  a->mov(loopR1_, H_out_R_);
  a->sub(loopR1_, top_edge_width_ + bottom_edge_width_);
  a->imul(loopR2_, W_out_R_, 8 * sizeof(int32_t));
  a->bind(LoopLeftEdge);
  // zero out
  a->vxorps(resultRegAvx2_, resultRegAvx2_, resultRegAvx2_);
  if (!isAZeroPointZero_) {
    genZeroPtSum<inst_set_t::avx2>(a, R_ * W_PAD_);
  }
  a->mov(scratchReg1_, 0);
  for (int r = 0; r < R_; ++r) {
    for (int s = W_PAD_; s < S_; ++s) {
      gen8BitSum<inst_set_t::avx2>(a, (s - W_PAD_) * C_);
    }
    a->imul(scratchReg2_, W_in_R_, C_);
    a->add(scratchReg1_, scratchReg2_);
  }
  for (int i = 0; i < stride_; ++i) {
    a->add(in_acts_R_, scratchReg2_);
  }

  a->add(row_offset_R_, loopR2_);
  storeResultRowoffset<inst_set_t::avx2>(a);

  a->dec(loopR1_);
  a->jg(LoopLeftEdge);

  // reset output pointer
  a->mov(scratchReg2_, H_out_R_);
  a->sub(scratchReg2_, bottom_edge_width_ + top_edge_width_);
  a->imul(scratchReg2_, W_out_R_);
  a->imul(scratchReg2_, static_cast<asmjit::Imm>(8 * sizeof(int32_t)));
  a->sub(row_offset_R_, scratchReg2_);

  // reset input pointer
  a->mov(in_acts_R_, wghts_R_);
}

template <>
template <>
void GenConvKernel<int32_t>::genForRightEdgeRowoffset<inst_set_t::avx2>(
    asmjit::X86Emitter* a) {
  // Save the original in_acts_R_ in wghts_R_
  a->mov(wghts_R_, in_acts_R_);

  a->imul(scratchReg1_, W_in_R_, (top_edge_width_ * stride_ - H_PAD_) * C_);
  a->imul(scratchReg2_, W_out_R_, stride_ * C_);
  a->add(scratchReg1_, scratchReg2_);
  a->add(in_acts_R_, scratchReg1_);

  // right edge excluding corners
  asmjit::Label LoopRightEdge = a->newLabel();

  // output pointer to the right edge
  // (W_ + W_ - 1)*8*sizeof(int32_t)
  a->mov(scratchReg2_, W_out_R_);
  a->imul(scratchReg2_, top_edge_width_ + 1);
  a->sub(scratchReg2_, right_edge_width_);
  a->imul(scratchReg2_, static_cast<asmjit::Imm>(8 * sizeof(int32_t)));
  a->add(row_offset_R_, scratchReg2_);

  a->imul(scratchReg2_, W_in_R_, C_ * sizeof(uint8_t));

  // loopR1_ corresponds to the output row
  a->mov(loopR1_, H_out_R_);
  a->sub(loopR1_, top_edge_width_ + bottom_edge_width_);
  a->imul(loopR2_, W_out_R_, 8 * sizeof(int32_t));
  a->bind(LoopRightEdge);
  // zero out
  a->vxorps(resultRegAvx2_, resultRegAvx2_, resultRegAvx2_);
  if (!isAZeroPointZero_) {
    genZeroPtSum<inst_set_t::avx2>(a, R_ * W_PAD_);
  }

  a->mov(scratchReg1_, 0);
  for (int r = 0; r < R_; ++r) {
    for (int s = 0; s < S_ - W_PAD_; ++s) {
      gen8BitSum<inst_set_t::avx2>(
          a, (s - right_edge_width_ * stride_ - W_PAD_) * C_);
    }

    a->add(scratchReg1_, scratchReg2_);
  }
  for (int i = 0; i < stride_; ++i) {
    a->add(in_acts_R_, scratchReg2_);
  }

  storeResultRowoffset<inst_set_t::avx2>(a);

  a->add(row_offset_R_, loopR2_);
  a->dec(loopR1_);
  a->jg(LoopRightEdge);

  // reset base
  a->mov(scratchReg2_, W_out_R_);
  a->imul(scratchReg2_, top_edge_width_ + 1);
  a->sub(scratchReg2_, right_edge_width_);
  a->imul(scratchReg2_, static_cast<asmjit::Imm>(8 * sizeof(int32_t)));
  a->sub(row_offset_R_, scratchReg2_);

  // reset increments done in the loop
  //(H_ - 2*H_PAD_)*W_*8*sizeof(int32_t)
  a->mov(scratchReg2_, H_out_R_);
  a->sub(scratchReg2_, top_edge_width_ + bottom_edge_width_);
  a->imul(scratchReg2_, W_out_R_);
  a->imul(scratchReg2_, static_cast<asmjit::Imm>(8 * sizeof(int32_t)));
  a->sub(row_offset_R_, scratchReg2_);

  // reset input pointer
  a->mov(in_acts_R_, wghts_R_);
}

template <>
template <>
void GenConvKernel<int32_t>::genForBottomEdgeRowoffset<inst_set_t::avx2>(
    asmjit::X86Emitter* a) {
  // bottom-left corner
  // we updating the last row
  a->mov(scratchReg2_, H_out_R_);
  a->sub(scratchReg2_, bottom_edge_width_);
  a->imul(scratchReg1_, scratchReg2_, 8 * sizeof(int32_t));
  a->imul(scratchReg1_, W_out_R_);
  a->add(row_offset_R_, scratchReg1_);

  // zero out
  a->vxorps(resultRegAvx2_, resultRegAvx2_, resultRegAvx2_);
  if (!isAZeroPointZero_) {
    genZeroPtSum<inst_set_t::avx2>(a, R_ * S_ - (R_ - H_PAD_) * (S_ - W_PAD_));
  }
  // adjust input pointer
  a->imul(scratchReg1_, scratchReg2_, stride_);
  a->sub(scratchReg1_, H_PAD_);
  a->imul(scratchReg1_, W_in_R_);
  a->imul(scratchReg1_, static_cast<asmjit::Imm>(C_ * sizeof(uint8_t)));
  a->add(in_acts_R_, scratchReg1_);
  // Save the original in_acts_R_ after adjustment in wghts_R_
  a->mov(wghts_R_, in_acts_R_);

  for (int r = 0; r < R_ - H_PAD_; ++r) {
    a->imul(scratchReg1_, W_in_R_, static_cast<asmjit::Imm>(r * C_));
    for (int s = W_PAD_; s < S_; ++s) {
      gen8BitSum<inst_set_t::avx2>(a, (s - W_PAD_) * C_);
    }
  }

  storeResultRowoffset<inst_set_t::avx2>(a);
  a->add(row_offset_R_, static_cast<asmjit::Imm>(8 * sizeof(int32_t)));

  // bottom edge excluding corners
  a->imul(scratchReg2_, W_in_R_, static_cast<asmjit::Imm>(C_));
  asmjit::Label LoopBottomEdge = a->newLabel();
  a->mov(loopR1_, W_out_R_);
  a->sub(loopR1_, left_edge_width_ + right_edge_width_);
  a->bind(LoopBottomEdge);
  // zero out
  a->vxorps(resultRegAvx2_, resultRegAvx2_, resultRegAvx2_);
  if (!isAZeroPointZero_) {
    genZeroPtSum<inst_set_t::avx2>(a, H_PAD_ * S_);
  }
  a->mov(scratchReg1_, 0);
  for (int r = 0; r < R_ - H_PAD_; ++r) {
    // int h_in = H_-2*H_PAD_ + r;
    for (int s = 0; s < S_; ++s) {
      gen8BitSum<inst_set_t::avx2>(a, (stride_ - W_PAD_ + s) * C_);
    }
    a->add(scratchReg1_, scratchReg2_);
  }

  a->add(in_acts_R_, static_cast<asmjit::Imm>(stride_ * C_ * sizeof(uint8_t)));
  // storeResult<inst_set_t::avx2>(a, ((H_-1)*W_+1)*8*sizeof(int32_t));
  storeResultRowoffset<inst_set_t::avx2>(a);

  a->add(row_offset_R_, static_cast<asmjit::Imm>(8 * sizeof(int32_t)));
  a->dec(loopR1_);
  a->jg(LoopBottomEdge);
  // Restore the original in_acts_R_
  a->mov(in_acts_R_, wghts_R_);

  // bottom-right corner
  if (right_edge_width_ > 0) {
  // zero out
  a->vxorps(resultRegAvx2_, resultRegAvx2_, resultRegAvx2_);
  if (!isAZeroPointZero_) {
    genZeroPtSum<inst_set_t::avx2>(
        a, R_ * S_ - (R_ - H_PAD_) * (S_ - W_PAD_));
  }
  // input start point
  a->mov(scratchReg1_, W_out_R_);
  a->imul(scratchReg1_, stride_);
  a->imul(scratchReg1_, static_cast<asmjit::Imm>(C_ * sizeof(uint8_t)));
  a->imul(scratchReg2_, W_in_R_, C_ * sizeof(uint8_t));

  for (int r = 0; r < R_ - H_PAD_; ++r) {
    for (int s = 0; s < S_ - W_PAD_; ++s) {
      gen8BitSum<inst_set_t::avx2>(
          a, (s - right_edge_width_ * stride_ - W_PAD_) * C_);
    }
    a->add(scratchReg1_, scratchReg2_);
  }

  storeResultRowoffset<inst_set_t::avx2>(a);
  a->add(row_offset_R_, static_cast<asmjit::Imm>(8 * sizeof(int32_t)));
  } // right_edge_width_ > 0

  // reset input pointer
  a->mov(scratchReg2_, H_out_R_);
  a->sub(scratchReg2_, bottom_edge_width_);
  a->imul(scratchReg1_, scratchReg2_, stride_);
  a->sub(scratchReg1_, H_PAD_);
  a->imul(scratchReg1_, W_in_R_);
  a->imul(scratchReg1_, static_cast<asmjit::Imm>(C_ * sizeof(uint8_t)));
  a->sub(in_acts_R_, scratchReg1_);

  // reset output pointer
  a->imul(scratchReg2_, W_out_R_);
  a->add(scratchReg2_, W_out_R_);
  a->imul(scratchReg2_, static_cast<asmjit::Imm>(8 * sizeof(int32_t)));
  a->sub(row_offset_R_, scratchReg2_);
}

template <>
template <>
void GenConvKernel<int32_t>::genRowoffsetCore<inst_set_t::avx2>(
    asmjit::X86Emitter* a) {
  // number of uint8 elements in input channels should be a multiple of 32
  assert(C_ % 32 == 0);

  asmjit::Label LoopH = a->newLabel();
  asmjit::Label LoopW = a->newLabel();
  // base for output
  a->imul(scratchReg2_, W_out_R_, top_edge_width_);
  a->add(scratchReg2_, left_edge_width_);
  a->imul(scratchReg2_, 8 * sizeof(int32_t));
  a->add(row_offset_R_, scratchReg2_);

  // base for input
  if (stride_ > 1) {
    a->imul(scratchReg1_, W_in_R_, top_edge_width_ * stride_ - H_PAD_);
    a->add(scratchReg1_, left_edge_width_ * stride_ - W_PAD_);
    a->imul(scratchReg1_, C_);
    a->add(in_acts_R_, scratchReg1_);

    // Save the original in_acts_R_ in wghts_R_
    a->mov(wghts_R_, in_acts_R_);
  }

  a->imul(scratchReg2_, W_in_R_, C_ * sizeof(uint8_t));

  // H loop
  a->mov(loopR1_, H_out_R_);
  a->sub(loopR1_, top_edge_width_ + bottom_edge_width_);
  a->bind(LoopH);
  // W loop
  a->mov(loopR2_, W_out_R_);
  a->sub(loopR2_, left_edge_width_ + right_edge_width_);
  a->bind(LoopW);

  // zero out
  a->vxorps(resultRegAvx2_, resultRegAvx2_, resultRegAvx2_);
  for (int r = 0; r < R_; ++r) {
    for (int s = 0; s < S_; ++s) {
      gen8BitSum<inst_set_t::avx2>(a, s * C_, false /*use_scratch_reg1*/);
    }
    a->add(in_acts_R_, scratchReg2_);
  }
  a->imul(scratchReg1_, W_in_R_, R_ * C_ * sizeof(uint8_t));
  a->sub(in_acts_R_, scratchReg1_);
  // store results
  storeResultRowoffset<inst_set_t::avx2>(a);

  a->add(in_acts_R_, static_cast<asmjit::Imm>(stride_ * C_ * sizeof(uint8_t)));
  a->add(row_offset_R_, static_cast<asmjit::Imm>(8 * sizeof(int32_t)));

  a->dec(loopR2_);
  a->jg(LoopW);
  if (stride_ == 1) {
    a->add(in_acts_R_, (left_edge_width_ + bottom_edge_width_) * C_);
  } else {
    a->imul(scratchReg1_, W_in_R_, stride_ * C_);
    a->add(wghts_R_, scratchReg1_);
    a->mov(in_acts_R_, wghts_R_);
  }
  a->add(
      row_offset_R_,
      (left_edge_width_ + right_edge_width_) * 8 * sizeof(int32_t));
  a->dec(loopR1_);
  a->jg(LoopH);
}

template <>
template <>
jit_rowoffset_kernel_fp
GenConvKernel<int32_t>::getOrCreateRowOffset<inst_set_t::avx2>(
    const conv_param_t<>& conv_param) {
  code_.reset(false);
  code_.init(rt_.getCodeInfo());
  asmjit::X86Assembler assembler(&code_);
  asmjit::X86Emitter* a = assembler.asEmitter();

#if defined(FBGEMM_LOG_CODE)
  // log code to a file
  FILE* codeLogfile =
      fopen(getCodeLoggingFile<inst_set_t::avx2>(true).c_str(), "w");
  asmjit::FileLogger* codeLogger = new asmjit::FileLogger(codeLogfile);
  if (codeLogger) {
    code_.setLogger(codeLogger);
  }
#endif

  // arguments to the function created
  in_acts_R_ = a->zdi();
  a_zero_pt_R_ = a->zsi();
  H_out_R_ = a->zdx(); // we get H_in initially but will convert to H_out soon
  W_in_R_ = a->zcx();
  row_offset_R_ = a->gpzRef(8);
  wghts_R_ = a->gpzRef(9);
  W_out_R_ = a->gpzRef(10);

  // register for temporary use
  scratchReg1_ = a->gpzRef(12);
  scratchReg2_ = a->gpzRef(13);

  loopR1_ = a->gpzRef(14);
  loopR2_ = a->gpzRef(15);

  asmjit::FuncDetail func;
  func.init(
      asmjit::
          FuncSignature5<void, uint8_t*, int32_t, int32_t, int32_t, int32_t*>(
              asmjit::CallConv::kIdHost));

  asmjit::FuncFrameInfo ffi;
  ffi.setDirtyRegs(
      asmjit::X86Reg::kKindVec,
      asmjit::Utils::mask(0, 1, 2, 3, 4, 5, 6, 7) |
          asmjit::Utils::mask(8, 9, 10, 11, 12, 13, 14, 15));
  ffi.setDirtyRegs(
      asmjit::X86Reg::kKindGp, asmjit::Utils::mask(10, 11, 12, 13, 14, 15));

  asmjit::FuncArgsMapper args(&func);
  args.assignAll(in_acts_R_, a_zero_pt_R_, H_out_R_, W_in_R_, row_offset_R_);

  args.updateFrameInfo(ffi);

  asmjit::FuncFrameLayout layout;
  layout.init(func, ffi);

  asmjit::FuncUtils::emitProlog(a, layout);
  asmjit::FuncUtils::allocArgs(a, layout, args);

  // This uses xmm10 register temporarily. Should come before
  // createVector8BitOne
  if (!isAZeroPointZero_) {
    // we can use xmm11 because ymm11 is used by tmpReg1Avx2_
    asmjit::X86Xmm const_reg_xmm = x86::xmm11;
    a->movq(const_reg_xmm, a_zero_pt_R_);
    a->vpbroadcastd(zeroPTRegAvx2_, const_reg_xmm);

    a->mov(scratchReg1_, static_cast<asmjit::Imm>(C_per_G_));
    a->movq(const_reg_xmm, scratchReg1_);
    a->vpbroadcastd(tmpReg1Avx2_, const_reg_xmm);
    a->vpmulld(zeroPTRegAvx2_, zeroPTRegAvx2_, tmpReg1Avx2_);
  }

  createVector16BitOne<inst_set_t::avx2>(a);
  // we set ymm10 to contain 8-bit 1s
  createVector8BitOne<inst_set_t::avx2>(a);

  // compute H_out_R_ and W_out_R_
  // h_out = (h_in + 2*pad - kernel) / stride + 1
  a->mov(W_out_R_, W_in_R_);
  if (stride_ > 1 || 2 * W_PAD_ - S_ != -1) {
    assert(stride_ == 2);
    a->add(H_out_R_, 2 * H_PAD_ - R_);
    a->shr(H_out_R_, 1);
    a->add(H_out_R_, 1);

    a->add(W_out_R_, 2 * W_PAD_ - S_);
    a->shr(W_out_R_, 1);
    a->add(W_out_R_, 1);
  }

  genForTopEdgeRowoffset<inst_set_t::avx2>(a);
  genForLeftEdgeRowoffset<inst_set_t::avx2>(a);

  if (right_edge_width_ > 0) {
    genForRightEdgeRowoffset<inst_set_t::avx2>(a);
  }
  if (bottom_edge_width_ > 0) {
    genForBottomEdgeRowoffset<inst_set_t::avx2>(a);
  }

  genRowoffsetCore<inst_set_t::avx2>(a);

  asmjit::FuncUtils::emitEpilog(a, layout);

  jit_rowoffset_kernel_fp fn;
  asmjit::Error err = rt_.add(&fn, &code_);
  if (err) {
    std::cout << "Error: in fn add" << std::endl;
    return nullptr;
  }
  auto kernelSig = getKernelSig(conv_param, isAZeroPointZero_);
  codeCacheRowOffset_[kernelSig] = fn;

#if defined(FBGEMM_LOG_CODE)
  delete codeLogger;
  fclose(codeLogfile);
#endif

  return fn;
}

namespace {

template <
    typename packed_W,
    typename outType,
    typename processOutputType,
    int SPATIAL_DIM>
void fbgemmGroupwiseConvBase_(
    const conv_param_t<SPATIAL_DIM>& conv_param,
    const std::uint8_t* activations,
    std::int32_t a_zero_point,
    std::int32_t* rowOffsetBuf,
    packed_W& packed_weights,
    outType* out,
    int32_t* outBuffer,
    const processOutputType& outProcess,
    int thread_id,
    int num_threads) {

  int MB = conv_param.MB;
  int H_in = conv_param.IN_DIM[0];
  int W_in = conv_param.IN_DIM[1];
  int G = conv_param.G;
  int K_per_G = conv_param.OC / G;
  int C_per_G = conv_param.IC / G;
  int oh_ow = conv_param.OUT_DIM[0] * conv_param.OUT_DIM[1];
  int ih_iw = H_in * W_in;

  static_assert(SPATIAL_DIM == 2, "3D conv not supported yet");

  if (fbgemmOptimizedGConv<SPATIAL_DIM>(conv_param)) {
    int32_t* rowOffsetTrDest =
        rowOffsetBuf ? rowOffsetBuf + 8 * oh_ow : nullptr;
    assert(G % 8 == 0);
    // generate convolution kernel
    jit_conv_kernel_fp fpConv =
        getOrCreateConvKernel<>(conv_param, a_zero_point);
    // generate row offset kernel
    jit_rowoffset_kernel_fp fpRowoffset =
        getOrCreateRowOffsetKernel(conv_param, a_zero_point);
    for (int i = 0; i < MB; ++i) {
      const uint8_t* actStartBatch = activations + i * ih_iw * conv_param.IC;
      for (int gOuter = 0; gOuter < G; gOuter += 8) {
        // row offset is calcualted for 8 groups at a time.
        // The result is row offsets in the format IH*IW x G
        if (rowOffsetBuf) {
          fpRowoffset(
              actStartBatch + gOuter * C_per_G,
              a_zero_point,
              H_in,
              W_in,
              rowOffsetBuf);
          // Transpose to get row offsets in the format G x IH*IW
          internal::transpose_8x8(
              oh_ow,
              8,
              reinterpret_cast<const float*>(rowOffsetBuf),
              8,
              reinterpret_cast<float*>(rowOffsetTrDest),
              oh_ow);
        }
        int gLimit = gOuter + 8;
        // Work on 8 output channels at a time (8 * sizeof(int32_t) == 32B VLEN
        // of AVX2), and we need multiple groups if a group has not enough
        // number of channels.
        int gDelta = std::max(8 / C_per_G, 1);
        for (int g = gOuter; g < gLimit; g += gDelta) {
          int32_t* currOutBuf =
              outBuffer + i * oh_ow * conv_param.OC + g * K_per_G;
          const uint8_t* actStartGroup = actStartBatch + g * C_per_G;
          for (int k = 0; k < K_per_G; k += 8) {
            // Don't be confused with k above which refers to output channels.
            // k0 and k1 are filter dimensions (commonly 3 and 3)
            int k0 = conv_param.K[0];
            int k1 = conv_param.K[1];
            fpConv(
                actStartGroup,
                // packed weight is in G (C/4) R S K 4 layout for IC_per_G >= 8
                // in (G/2) R S K (2C) for IC_per_G == 4
                packed_weights.getBuf() +
                    (g * (C_per_G / 4) * k0 * k1 * K_per_G + k) * 4,
                currOutBuf + k,
                a_zero_point,
                H_in,
                W_in);
          } // k loop

          // Output processing should be called for each group
          for (int j = 0; j < gDelta; ++j) {
            // calculateRowOffsets(
            // conv_param, actStartGroup, rowOffsetBuf, a_zero_point, j);
            int32_t* rowOffsetForCurG =
                rowOffsetTrDest
                ? rowOffsetTrDest + ((g - gOuter) + j) * oh_ow
                : nullptr;
            // compare_buffers(rowOffsetBuf, rowOffsetForCurG,
            // conv_param.IN_DIM[0]*conv_param.IN_DIM[1], 1, 1, 100);

            // outProcess expects rowOffsetBuf to contain row offsets for the
            // current group
            memcpy(rowOffsetBuf, rowOffsetForCurG, oh_ow * sizeof(int32_t));

            if (fbgemmHasAvx512Support()) {
              // Currently use avx2 code
              outProcess.template f<inst_set_t::avx2>(
                  out,
                  currOutBuf + j * K_per_G,
                  {i * oh_ow, oh_ow, (g + j) * K_per_G, K_per_G},
                  K_per_G * G,
                  K_per_G * G);
            } else if (fbgemmHasAvx2Support()) {
              outProcess.template f<inst_set_t::avx2>(
                  out,
                  currOutBuf + j * K_per_G,
                  {i * oh_ow, oh_ow, (g + j) * K_per_G, K_per_G},
                  K_per_G * G,
                  K_per_G * G);
            } else {
              // TODO: Have default slower path
              assert(0 && "unsupported architecure");
            }
          } // j loop
        } // g loop
      } // gOuter loop
    } // i loop
  } else {
    // for the not supported cases, just execute the naive C implementation
    conv_ref(
        conv_param,
        activations,
        a_zero_point,
        packed_weights.getBuf(),
        outBuffer);
    for (int i = 0; i < conv_param.MB; ++i) {
      for (int g = 0; g < conv_param.G; ++g) {
        if (rowOffsetBuf) {
          calculateRowOffsets(
              conv_param,
              activations +
                  i * conv_param.IN_DIM[0] * conv_param.IN_DIM[1] *
                      conv_param.IC,
              rowOffsetBuf,
              a_zero_point,
              g);
        }
        outProcess.template f<inst_set_t::anyarch>(
            out,
            outBuffer + i * oh_ow * conv_param.OC + g * K_per_G,
            {i * oh_ow, oh_ow, g * K_per_G, K_per_G},
            K_per_G * G,
            K_per_G * G);
      }
    }
  }
}

}

template <
    typename packed_W,
    typename outType,
    typename processOutputType,
    int SPATIAL_DIM>
void fbgemmGroupwiseConv(
    const conv_param_t<SPATIAL_DIM>& conv_param,
    const std::uint8_t* activations,
    std::int32_t a_zero_point,
    std::int32_t* rowOffsetBuf,
    packed_W& packed_weights,
    outType* out,
    int32_t* outBuffer,
    const processOutputType& outProcess,
    int thread_id,
    int num_threads) {
  return fbgemmGroupwiseConvBase_<
      packed_W,
      outType,
      processOutputType,
      SPATIAL_DIM>(
      conv_param,
      activations,
      a_zero_point,
      rowOffsetBuf,
      packed_weights,
      out,
      outBuffer,
      outProcess,
      thread_id,
      num_threads);
}

template <
    typename packed_W,
    typename outType,
    bool FUSE_RELU,
    QuantizationGranularity Q_GRAN,
    int SPATIAL_DIM>
void fbgemmGroupwiseConv(
    const conv_param_t<SPATIAL_DIM>& conv_param,
    const std::uint8_t* activations,
    std::int32_t a_zero_point,
    std::int32_t* rowOffsetBuf,
    packed_W& packed_weights,
    outType* out,
    int32_t* outBuffer,
    const ReQuantizeOutput<FUSE_RELU, Q_GRAN>& outProcess,
    int thread_id,
    int num_threads) {
  typedef ReQuantizeOutput<FUSE_RELU, Q_GRAN> processOutputType;

  if (!fbgemmOptimizedGConv<SPATIAL_DIM>(conv_param) ||
      (!fbgemmHasAvx512Support() && !fbgemmHasAvx2Support())) {
    return fbgemmGroupwiseConvBase_<
        packed_W,
        outType,
        processOutputType,
        SPATIAL_DIM>(
        conv_param,
        activations,
        a_zero_point,
        rowOffsetBuf,
        packed_weights,
        out,
        outBuffer,
        outProcess,
        thread_id,
        num_threads);
  }

  int MB = conv_param.MB;
  int H_in = conv_param.IN_DIM[0];
  int W_in = conv_param.IN_DIM[1];
  int G = conv_param.G;
  int K_per_G = conv_param.OC / G;
  int C_per_G = conv_param.IC / G;
  int oh_ow = conv_param.OUT_DIM[0] * conv_param.OUT_DIM[1];
  int ih_iw = H_in * W_in;

  static_assert(SPATIAL_DIM == 2, "3D conv not supported yet");

  int32_t* rowOffsetTrDest = rowOffsetBuf ? rowOffsetBuf + 8 * oh_ow : nullptr;
  assert(G % 8 == 0);
  // generate convolution kernel
  jit_conv_kernel_fp fpConv =
      getOrCreateConvKernel<>(conv_param, a_zero_point);
  // generate row offset kernel
  jit_rowoffset_kernel_fp fpRowoffset =
      getOrCreateRowOffsetKernel(conv_param, a_zero_point);
  for (int i = 0; i < MB; ++i) {
    const uint8_t* actStartBatch = activations + i * ih_iw * conv_param.IC;
    for (int gOuter = 0; gOuter < G; gOuter += 8) {
      if (rowOffsetBuf && outProcess.getBZeroPoint()) {
        // row offset is calcualted for 8 groups at a time.
        // The result is row offsets in the format IH*IW x G
        fpRowoffset(
            actStartBatch + gOuter * C_per_G,
            a_zero_point,
            H_in,
            W_in,
            rowOffsetBuf);
        // Transpose to get row offsets in the format G x IH*IW
        internal::transpose_8x8(
            oh_ow,
            8,
            reinterpret_cast<const float*>(rowOffsetBuf),
            8,
            reinterpret_cast<float*>(rowOffsetTrDest),
            oh_ow);
      }
      int gLimit = gOuter + 8;
      // Work on 8 output channels at a time (8 * sizeof(int32_t) == 32B VLEN
      // of AVX2), and we need multiple groups if a group has not enough
      // number of channels.
      int gDelta = std::max(8 / C_per_G, 1);
      for (int g = gOuter; g < gLimit; g += gDelta) {
        // Reusing the same region of outBuffer multiple times for locality
        int32_t* currOutBuf = outBuffer + (g - gOuter) * K_per_G;
        const uint8_t* actStartGroup = actStartBatch + g * C_per_G;
        for (int k = 0; k < K_per_G; k += 8) {
          // Don't be confused with k above which refers to output channels.
          // k0 and k1 are filter dimensions (commonly 3 and 3)
          int k0 = conv_param.K[0];
          int k1 = conv_param.K[1];
          fpConv(
              actStartGroup,
              // packed weight is in G (C/4) R S K 4 layout for IC_per_G >= 8
              // in (G/2) R S K (2C) for IC_per_G == 4
              packed_weights.getBuf() +
                  (g * (C_per_G / 4) * k0 * k1 * K_per_G + k) * 4,
              currOutBuf + k,
              a_zero_point,
              H_in,
              W_in);
        } // k loop
      } // g loop

      bool b_symmetric = (Q_GRAN == QuantizationGranularity::TENSOR &&
                          outProcess.getBZeroPoint()[0] == 0) ||
          rowOffsetBuf == nullptr;

      requantizationParams_t r = {a_zero_point,
                                  outProcess.getBZeroPoint(),
                                  outProcess.getCZeroPoint(),
                                  outProcess.getCMultiplier(),
                                  rowOffsetBuf,
                                  outProcess.getColOffsets(),
                                  outProcess.getBias(),
                                  outProcess.getNCols(),
                                  G};

      const std::int32_t* inp = outBuffer;
      block_type_t block{i * oh_ow, oh_ow, gOuter * K_per_G, 8 * K_per_G};
      int ld_out = K_per_G * G;
      int ld_in = K_per_G * G;

      if (C_per_G == 4) {
        if (a_zero_point == 0) {
          if (b_symmetric) {
            if (outProcess.getBias() == nullptr) {
              requantizeOutputProcessingGConvAvx2<
                  true,
                  true,
                  Q_GRAN,
                  false,
                  FUSE_RELU,
                  4>(out, inp, block, ld_out, ld_in, r);
            } else {
              requantizeOutputProcessingGConvAvx2<
                  true,
                  true,
                  Q_GRAN,
                  true,
                  FUSE_RELU,
                  4>(out, inp, block, ld_out, ld_in, r);
            }
          } else {
            if (outProcess.getBias() == nullptr) {
              requantizeOutputProcessingGConvAvx2<
                  true,
                  false,
                  Q_GRAN,
                  false,
                  FUSE_RELU,
                  4>(out, inp, block, ld_out, ld_in, r);
            } else {
              requantizeOutputProcessingGConvAvx2<
                  true,
                  false,
                  Q_GRAN,
                  true,
                  FUSE_RELU,
                  4>(out, inp, block, ld_out, ld_in, r);
            }
          }
        } else {
          if (b_symmetric) {
            if (outProcess.getBias() == nullptr) {
              requantizeOutputProcessingGConvAvx2<
                  false,
                  true,
                  Q_GRAN,
                  false,
                  FUSE_RELU,
                  4>(out, inp, block, ld_out, ld_in, r);
            } else {
              requantizeOutputProcessingGConvAvx2<
                  false,
                  true,
                  Q_GRAN,
                  true,
                  FUSE_RELU,
                  4>(out, inp, block, ld_out, ld_in, r);
            }
          } else {
            if (outProcess.getBias() == nullptr) {
              requantizeOutputProcessingGConvAvx2<
                  false,
                  false,
                  Q_GRAN,
                  false,
                  FUSE_RELU,
                  4>(out, inp, block, ld_out, ld_in, r);
            } else {
              requantizeOutputProcessingGConvAvx2<
                  false,
                  false,
                  Q_GRAN,
                  true,
                  FUSE_RELU,
                  4>(out, inp, block, ld_out, ld_in, r);
            }
          }
        }
      } else if (C_per_G == 8) {
        if (a_zero_point == 0) {
          if (b_symmetric) {
            if (outProcess.getBias() == nullptr) {
              requantizeOutputProcessingGConvAvx2<
                  true,
                  true,
                  Q_GRAN,
                  false,
                  FUSE_RELU,
                  8>(out, inp, block, ld_out, ld_in, r);
            } else {
              requantizeOutputProcessingGConvAvx2<
                  true,
                  true,
                  Q_GRAN,
                  true,
                  FUSE_RELU,
                  8>(out, inp, block, ld_out, ld_in, r);
            }
          } else {
            if (outProcess.getBias() == nullptr) {
              requantizeOutputProcessingGConvAvx2<
                  true,
                  false,
                  Q_GRAN,
                  false,
                  FUSE_RELU,
                  8>(out, inp, block, ld_out, ld_in, r);
            } else {
              requantizeOutputProcessingGConvAvx2<
                  true,
                  false,
                  Q_GRAN,
                  true,
                  FUSE_RELU,
                  8>(out, inp, block, ld_out, ld_in, r);
            }
          }
        } else {
          if (b_symmetric) {
            if (outProcess.getBias() == nullptr) {
              requantizeOutputProcessingGConvAvx2<
                  false,
                  true,
                  Q_GRAN,
                  false,
                  FUSE_RELU,
                  8>(out, inp, block, ld_out, ld_in, r);
            } else {
              requantizeOutputProcessingGConvAvx2<
                  false,
                  true,
                  Q_GRAN,
                  true,
                  FUSE_RELU,
                  8>(out, inp, block, ld_out, ld_in, r);
            }
          } else {
            if (outProcess.getBias() == nullptr) {
              requantizeOutputProcessingGConvAvx2<
                  false,
                  false,
                  Q_GRAN,
                  false,
                  FUSE_RELU,
                  8>(out, inp, block, ld_out, ld_in, r);
            } else {
              requantizeOutputProcessingGConvAvx2<
                  false,
                  false,
                  Q_GRAN,
                  true,
                  FUSE_RELU,
                  8>(out, inp, block, ld_out, ld_in, r);
            }
          }
        }
      } else {
        if (a_zero_point == 0) {
          if (b_symmetric) {
            if (outProcess.getBias() == nullptr) {
              requantizeOutputProcessingGConvAvx2<
                  true,
                  true,
                  Q_GRAN,
                  false,
                  FUSE_RELU,
                  16>(out, inp, block, ld_out, ld_in, r);
            } else {
              requantizeOutputProcessingGConvAvx2<
                  true,
                  true,
                  Q_GRAN,
                  true,
                  FUSE_RELU,
                  16>(out, inp, block, ld_out, ld_in, r);
            }
          } else {
            if (outProcess.getBias() == nullptr) {
              requantizeOutputProcessingGConvAvx2<
                  true,
                  false,
                  Q_GRAN,
                  false,
                  FUSE_RELU,
                  16>(out, inp, block, ld_out, ld_in, r);
            } else {
              requantizeOutputProcessingGConvAvx2<
                  true,
                  false,
                  Q_GRAN,
                  true,
                  FUSE_RELU,
                  16>(out, inp, block, ld_out, ld_in, r);
            }
          }
        } else {
          if (b_symmetric) {
            if (outProcess.getBias() == nullptr) {
              requantizeOutputProcessingGConvAvx2<
                  false,
                  true,
                  Q_GRAN,
                  false,
                  FUSE_RELU,
                  16>(out, inp, block, ld_out, ld_in, r);
            } else {
              requantizeOutputProcessingGConvAvx2<
                  false,
                  true,
                  Q_GRAN,
                  true,
                  FUSE_RELU,
                  16>(out, inp, block, ld_out, ld_in, r);
            }
          } else {
            if (outProcess.getBias() == nullptr) {
              requantizeOutputProcessingGConvAvx2<
                  false,
                  false,
                  Q_GRAN,
                  false,
                  FUSE_RELU,
                  16>(out, inp, block, ld_out, ld_in, r);
            } else {
              requantizeOutputProcessingGConvAvx2<
                  false,
                  false,
                  Q_GRAN,
                  true,
                  FUSE_RELU,
                  16>(out, inp, block, ld_out, ld_in, r);
            }
          }
        }
      }
    } // gOuter loop
  } // i loop
}

jit_rowoffset_kernel_fp getOrCreateRowOffsetKernel(
    const conv_param_t<>& conv_param,
    int a_zero_point) {
  // Note: Wrong code is generated if it's not one of the supported convolution
  assert(fbgemmOptimizedGConv<2>(conv_param));
  auto kernelSig = getKernelSig(conv_param, a_zero_point == 0);
  if (GenConvKernel<int32_t>::codeCacheRowOffset_.find(kernelSig) !=
      GenConvKernel<int32_t>::codeCacheRowOffset_.end()) {
    return GenConvKernel<int32_t>::codeCacheRowOffset_[kernelSig];
  } else {
    auto genObj = GenConvKernel<int32_t>(conv_param, a_zero_point);
    // TODO: Instruction set based dispatch
    return genObj.template getOrCreateRowOffset<inst_set_t::avx2>(conv_param);
  }
}

template <int SPATIAL_DIM>
int rowOffsetBufferSizeGConv(const conv_param_t<SPATIAL_DIM>& conv_param) {
  // row offset buffer should be a able to hold row offsets for however
  // number of groups we process at a time.
  if (cpuinfo_initialize()) {
    if (fbgemmHasAvx512Support()) {
      int bufferSize = conv_param.OUT_DIM[0] * conv_param.OUT_DIM[1];
      int C_per_G = conv_param.IC / conv_param.G;
      int K_per_G = conv_param.OC / conv_param.G;
      if (C_per_G == K_per_G &&
          (C_per_G == 4 || C_per_G == 8 || C_per_G == 16)) {
        return 2 * 8 * bufferSize;
      } else {
        return conv_param.G * bufferSize;
      }
    } else if (fbgemmHasAvx2Support()) {
      int bufferSize = conv_param.OUT_DIM[0] * conv_param.OUT_DIM[1];
      int C_per_G = conv_param.IC / conv_param.G;
      int K_per_G = conv_param.OC / conv_param.G;
      if (C_per_G == K_per_G &&
          (C_per_G == 4 || C_per_G == 8 || C_per_G == 16)) {
        // row offset is calculated for 8 groups at a time
        // 2x is needed for transposing
        return 2 * 8 * bufferSize;
      } else {
        return conv_param.G * bufferSize;
      }
    } else {
      // TODO: Have default slower path
      assert(0 && "unsupported architecture");
      return -1;
    }
  } else {
    throw std::runtime_error("Failed to initialize cpuinfo!");
  }
}

template int rowOffsetBufferSizeGConv<2>(const conv_param_t<2>& conv_param);

#define INSTANTIATE_BASE(RELU, Q_GRAN)                              \
  template void fbgemmGroupwiseConv(                                \
      const conv_param_t<2>& conv_param,                            \
      const uint8_t* activations,                                   \
      int32_t a_zero_point,                                         \
      std::int32_t* rowOffsetBuf,                                   \
      PackWeightMatrixForGConv<int8_t, int32_t, 2>& packed_weights, \
      uint8_t* out,                                                 \
      int32_t* outBuffer,                                           \
      const ReQuantizeOutput<RELU, Q_GRAN>& outProcess,             \
      int thread_id,                                                \
      int num_threads);

#define INSTANTIATE_Q_GRANS(RELU)                          \
  INSTANTIATE_BASE(RELU, QuantizationGranularity::TENSOR); \
  INSTANTIATE_BASE(RELU, QuantizationGranularity::GROUP);  \
  INSTANTIATE_BASE(RELU, QuantizationGranularity::OUT_CHANNEL);

INSTANTIATE_Q_GRANS(false);
INSTANTIATE_Q_GRANS(true);

#undef INSTANTIATE_Q_GRANS
#undef INSTANTIATE_BASE

template void fbgemmGroupwiseConv(
    const conv_param_t<2>& conv_param,
    const uint8_t* activations,
    int32_t a_zero_point,
    std::int32_t* rowOffsetBuf,
    PackWeightMatrixForGConv<int8_t, int32_t, 2>& packed_weights,
    int32_t* out,
    int32_t* outBuffer,
    const DoNothing<int32_t, int32_t>& outProcess,
    int thread_id,
    int num_threads);

} // namespace fbgemm
