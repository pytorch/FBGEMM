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
thread_local std::map<std::tuple<bool, int, int, int>, jit_conv_kernel_fp>
    GenConvKernel<accT>::codeCache_;

template <typename accT>
thread_local std::map<std::tuple<bool, int, int, int>, jit_rowoffset_kernel_fp>
    GenConvKernel<accT>::codeCacheRowOffset_;

namespace x86 = asmjit::x86;

void calculateRowOffsets(
    const conv_param_t<>& conv_param,
    const uint8_t* activations,
    int32_t* rowOffsetBuf,
    int32_t a_zero_point,
    int groupNum) {
  int H = conv_param.OUT_DIM[0];
  int W = conv_param.OUT_DIM[1];
  int G = conv_param.G;
  int C_per_G = conv_param.IC / conv_param.G;
  int H_PAD = conv_param.pad[0];
  int W_PAD = conv_param.pad[1];
  // calculate row offset
  for (int h = 0; h < H; ++h) {
    for (int w = 0; w < W; ++w) {
      int32_t sum = 0;
      for (int r = 0; r < conv_param.K[0]; ++r) {
        int h_in = -H_PAD + h + r;
        for (int s = 0; s < conv_param.K[1]; ++s) {
          int w_in = -W_PAD + w + s;
          for (int c = 0; c < C_per_G; ++c) {
            if (h_in < 0 || h_in >= H || w_in < 0 || w_in >= W) {
              sum += a_zero_point;
            } else {
              sum +=
                  activations[((h_in * W + w_in) * G + groupNum) * C_per_G + c];
            }
          }
        }
      }
      rowOffsetBuf[h * W + w] = sum;
    }
  }
}

tuple<bool, int, int, int> getKernelSig(
    const conv_param_t<>& conv_param,
    bool isZeroPointZero) {
  int C_per_G = conv_param.IC / conv_param.G;
  int K_per_G = conv_param.OC / conv_param.G;
  auto kernelSig =
      std::make_tuple(isZeroPointZero, conv_param.G, C_per_G, K_per_G);
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
    asmjit::X86Emitter* a,
    int offset) {
  // store with permutation
  a->vpermd(resultRegAvx2_, stPermRegAvx2_, resultRegAvx2_);
  a->vmovups(x86::dword_ptr(out_acts_R_, offset), resultRegAvx2_);
}

template <>
template <>
void GenConvKernel<int32_t>::storeResultRowoffset<inst_set_t::avx2>(
    asmjit::X86Emitter* a,
    int offset) {
  // store
  a->vmovups(x86::dword_ptr(row_offset_R_, offset), resultRegAvx2_);
}

template <>
template <>
void GenConvKernel<int32_t>::genForLoadingWeights<inst_set_t::avx2>(
    asmjit::X86Emitter* a) {
  // load weights
  for (int r = 0; r < R_; ++r) {
    for (int s = 0; s < S_; ++s) {
      a->vmovaps(
          WRegs_avx2_[r * S_ + s],
          x86::dword_ptr(
              wghts_R_,
              (r * S_ + s) * G_ * K_per_G_ * C_per_G_ * sizeof(int8_t)));
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
void GenConvKernel<int32_t>::gen8BitSum<inst_set_t::avx2>(
    asmjit::X86Emitter* a,
    asmjit::X86Ymm aReg) {
  a->vpmaddubsw(tmpReg1Avx2_, aReg, oneReg8BitAvx2_);
  a->vpmaddwd(tmpReg1Avx2_, tmpReg1Avx2_, oneReg16BitAvx2_);
  a->vpaddd(resultRegAvx2_, tmpReg1Avx2_, resultRegAvx2_);
}

template <>
template <>
void GenConvKernel<int32_t>::genForTopEdge<inst_set_t::avx2>(
    asmjit::X86Emitter* a) {
  // top-left corner code
  // zero out the results register
  a->vxorps(resultRegAvx2_, resultRegAvx2_, resultRegAvx2_);
  for (int r = 0; r < R_; ++r) {
    int h_in = -H_PAD_ + r;
    if (h_in >= 0) {
      a->imul(
          scratchReg1_,
          W_R_,
          static_cast<asmjit::Imm>(h_in * C_ * sizeof(uint8_t)));
    }
    for (int s = 0; s < S_; ++s) {
      int w_in = -W_PAD_ + s;
      if (h_in >= 0 && w_in >= 0) {
        a->vbroadcastsd(
            actRegAvx2_,
            x86::dword_ptr(
                in_acts_R_, scratchReg1_, 0, w_in * C_ * sizeof(uint8_t)));
        gen8bitFMA<inst_set_t::avx2>(a, actRegAvx2_, WRegs_avx2_[r * S_ + s]);
      } else {
        if (!isZeroPointZero_) {
          gen8bitFMA<inst_set_t::avx2>(
              a, zeroPTRegAvx2_, WRegs_avx2_[r * S_ + s]);
        }
      }
    }
  }
  storeResult<inst_set_t::avx2>(a);

  a->add(out_acts_R_, static_cast<asmjit::Imm>(K_ * sizeof(int32_t)));

  // top edge excluding corners
  asmjit::Label LoopTopEdge = a->newLabel();
  a->mov(loopR2_, static_cast<asmjit::Imm>(W_PAD_));
  a->bind(LoopTopEdge);
  // zero out
  a->vxorps(resultRegAvx2_, resultRegAvx2_, resultRegAvx2_);
  if (!isZeroPointZero_) {
    for (int r = 0; r < H_PAD_; ++r) {
      for (int s = 0; s < S_; ++s) {
        gen8bitFMA<inst_set_t::avx2>(a, zeroPTRegAvx2_, WRegs_avx2_[s]);
      }
    }
  }
  for (int r = H_PAD_; r < R_; ++r) {
    int h_in = -H_PAD_ + r;
    a->imul(
        scratchReg1_,
        W_R_,
        static_cast<asmjit::Imm>(h_in * C_ * sizeof(uint8_t)));
    for (int s = 0; s < S_; ++s) {
      a->vbroadcastsd(
          actRegAvx2_,
          x86::dword_ptr(
              in_acts_R_, scratchReg1_, 0, s * C_ * sizeof(uint8_t)));
      gen8bitFMA<inst_set_t::avx2>(a, actRegAvx2_, WRegs_avx2_[r * S_ + s]);
    }
  }
  a->add(in_acts_R_, static_cast<asmjit::Imm>(C_ * sizeof(uint8_t)));

  storeResult<inst_set_t::avx2>(a);

  a->add(out_acts_R_, static_cast<asmjit::Imm>(K_ * sizeof(int32_t)));
  a->mov(loopR1_, W_R_);
  a->sub(loopR1_, static_cast<asmjit::Imm>(W_PAD_));
  a->inc(loopR2_);
  a->cmp(loopR2_, loopR1_);
  a->jl(LoopTopEdge);
  a->mov(scratchReg2_, W_R_);
  a->imul(scratchReg2_, static_cast<asmjit::Imm>(C_ * sizeof(uint8_t)));
  a->sub(
      scratchReg2_,
      static_cast<asmjit::Imm>(2 * W_PAD_ * C_ * sizeof(uint8_t)));
  a->sub(in_acts_R_, scratchReg2_);

  // top-right corner code

  // zero out
  a->vxorps(resultRegAvx2_, resultRegAvx2_, resultRegAvx2_);
  if (!isZeroPointZero_) {
    for (int r = 0; r < H_PAD_; ++r) {
      for (int s = 0; s < S_; ++s) {
        gen8bitFMA<inst_set_t::avx2>(
            a, zeroPTRegAvx2_, WRegs_avx2_[r * S_ + s]);
      }
    }
  }
  for (int r = H_PAD_; r < R_; ++r) {
    int h_in = -H_PAD_ + r;
    for (int s = 0; s < S_ - W_PAD_; ++s) {
      a->imul(
          scratchReg1_,
          W_R_,
          static_cast<asmjit::Imm>(h_in * C_ * sizeof(uint8_t)));
      a->mov(scratchReg2_, W_R_);
      a->sub(scratchReg2_, static_cast<asmjit::Imm>(R_ - W_PAD_ - s));
      a->imul(scratchReg2_, static_cast<asmjit::Imm>(C_ * sizeof(uint8_t)));
      a->add(scratchReg1_, scratchReg2_);
      a->vbroadcastsd(actRegAvx2_, x86::dword_ptr(in_acts_R_, scratchReg1_));
      gen8bitFMA<inst_set_t::avx2>(a, actRegAvx2_, WRegs_avx2_[r * S_ + s]);
    }
    if (!isZeroPointZero_) {
      for (int s = S_ - W_PAD_; s < S_; ++s) {
        gen8bitFMA<inst_set_t::avx2>(
            a, zeroPTRegAvx2_, WRegs_avx2_[r * S_ + s]);
      }
    }
  }
  storeResult<inst_set_t::avx2>(a);
  a->add(out_acts_R_, static_cast<asmjit::Imm>(K_ * sizeof(int32_t)));

  // reset output activation pointer
  a->imul(scratchReg1_, W_R_, static_cast<asmjit::Imm>(K_ * sizeof(int32_t)));
  a->sub(out_acts_R_, scratchReg1_);
}

template <>
template <>
void GenConvKernel<int32_t>::genForLeftEdge<inst_set_t::avx2>(
    asmjit::X86Emitter* a) {
  // left edge excluding corners
  asmjit::Label LoopLeftEdge = a->newLabel();
  a->mov(loopR1_, static_cast<asmjit::Imm>(H_PAD_));
  a->bind(LoopLeftEdge);
  // zero out
  a->vxorps(resultRegAvx2_, resultRegAvx2_, resultRegAvx2_);
  a->mov(scratchReg1_, loopR1_);
  a->sub(scratchReg1_, static_cast<asmjit::Imm>(H_PAD_));
  a->imul(scratchReg1_, W_R_);
  a->imul(scratchReg1_, static_cast<asmjit::Imm>(C_ * sizeof(uint8_t)));
  for (int r = 0; r < R_; ++r) {
    if (!isZeroPointZero_) {
      for (int s = 0; s < W_PAD_; ++s) {
        gen8bitFMA<inst_set_t::avx2>(
            a, zeroPTRegAvx2_, WRegs_avx2_[r * S_ + s]);
      }
    }
    for (int s = W_PAD_; s < S_; ++s) {
      a->vbroadcastsd(
          actRegAvx2_,
          x86::dword_ptr(
              in_acts_R_,
              scratchReg1_,
              0,
              (s - W_PAD_) * C_ * sizeof(uint8_t)));
      gen8bitFMA<inst_set_t::avx2>(a, actRegAvx2_, WRegs_avx2_[r * S_ + s]);
    }
    a->imul(scratchReg2_, W_R_, static_cast<asmjit::Imm>(C_ * sizeof(uint8_t)));
    a->add(scratchReg1_, scratchReg2_);
  }

  a->imul(scratchReg2_, W_R_, static_cast<asmjit::Imm>(K_ * sizeof(int32_t)));
  a->add(out_acts_R_, scratchReg2_);
  storeResult<inst_set_t::avx2>(a);

  a->inc(loopR1_);
  a->mov(loopR2_, H_R_);
  a->sub(loopR2_, static_cast<asmjit::Imm>(H_PAD_));
  a->cmp(loopR1_, loopR2_);
  a->jl(LoopLeftEdge);

  // reset output activation pointer
  a->mov(scratchReg2_, H_R_);
  a->sub(scratchReg2_, static_cast<asmjit::Imm>(2 * H_PAD_));
  a->imul(scratchReg2_, W_R_);
  a->imul(scratchReg2_, static_cast<asmjit::Imm>(K_ * sizeof(int32_t)));
  a->sub(out_acts_R_, scratchReg2_);
}

template <>
template <>
void GenConvKernel<int32_t>::genForRightEdge<inst_set_t::avx2>(
    asmjit::X86Emitter* a) {
  // right edge excluding corners
  asmjit::Label LoopRightEdge = a->newLabel();

  // output pointer to the right edge
  // (W_ + W_ - 1)*K_*sizeof(int32_t)
  a->mov(scratchReg2_, W_R_);
  a->imul(scratchReg2_, 2);
  a->sub(scratchReg2_, 1);
  a->imul(scratchReg2_, static_cast<asmjit::Imm>(K_ * sizeof(int32_t)));
  a->add(out_acts_R_, scratchReg2_);

  a->mov(loopR1_, static_cast<asmjit::Imm>(H_PAD_));
  a->bind(LoopRightEdge);
  // zero out
  a->vxorps(resultRegAvx2_, resultRegAvx2_, resultRegAvx2_);
  a->mov(scratchReg1_, loopR1_);
  a->sub(scratchReg1_, static_cast<asmjit::Imm>(H_PAD_));
  a->imul(scratchReg1_, W_R_);
  a->imul(scratchReg1_, static_cast<asmjit::Imm>(C_ * sizeof(uint8_t)));

  a->mov(scratchReg2_, W_R_);
  a->sub(scratchReg2_, static_cast<asmjit::Imm>(2 * W_PAD_));
  a->imul(scratchReg2_, static_cast<asmjit::Imm>(C_ * sizeof(uint8_t)));
  a->add(scratchReg1_, scratchReg2_);
  for (int r = 0; r < R_; ++r) {
    for (int s = 0; s < S_ - W_PAD_; ++s) {
      a->vbroadcastsd(actRegAvx2_, x86::dword_ptr(in_acts_R_, scratchReg1_));
      gen8bitFMA<inst_set_t::avx2>(a, actRegAvx2_, WRegs_avx2_[r * S_ + s]);
      a->add(scratchReg1_, static_cast<asmjit::Imm>(C_ * sizeof(uint8_t)));
    }
    if (!isZeroPointZero_) {
      for (int s = S_ - W_PAD_; s < S_; ++s) {
        gen8bitFMA<inst_set_t::avx2>(
            a, zeroPTRegAvx2_, WRegs_avx2_[r * S_ + s]);
      }
    }

    a->sub(
        scratchReg1_,
        static_cast<asmjit::Imm>((S_ - W_PAD_) * C_ * sizeof(uint8_t)));
    a->imul(scratchReg2_, W_R_, static_cast<asmjit::Imm>(C_ * sizeof(uint8_t)));
    a->add(scratchReg1_, scratchReg2_);
  }

  // storeResult<inst_set_t::avx2>(a, (W_+W_-1)*K_*sizeof(int32_t));
  storeResult<inst_set_t::avx2>(a);

  a->imul(scratchReg2_, W_R_, static_cast<asmjit::Imm>(K_ * sizeof(int32_t)));
  a->add(out_acts_R_, scratchReg2_);
  a->mov(loopR2_, H_R_);
  a->sub(loopR2_, static_cast<asmjit::Imm>(H_PAD_));
  a->inc(loopR1_);
  a->cmp(loopR1_, loopR2_);
  a->jl(LoopRightEdge);

  // reset base
  a->mov(scratchReg2_, W_R_);
  a->imul(scratchReg2_, 2);
  a->sub(scratchReg2_, 1);
  a->imul(scratchReg2_, static_cast<asmjit::Imm>(K_ * sizeof(int32_t)));
  a->sub(out_acts_R_, scratchReg2_);

  // reset loop increments
  //(H_ - 2*H_PAD_)*W_*K_*sizeof(int32_t)
  a->mov(scratchReg2_, H_R_);
  a->sub(scratchReg2_, static_cast<asmjit::Imm>(2 * H_PAD_));
  a->imul(scratchReg2_, W_R_);
  a->imul(scratchReg2_, static_cast<asmjit::Imm>(K_ * sizeof(int32_t)));
  a->sub(out_acts_R_, scratchReg2_);
  // a->sub(out_acts_R_, (H_ - 2*H_PAD_)*W_*K_*sizeof(int32_t));
}

template <>
template <>
void GenConvKernel<int32_t>::genForBottomEdge<inst_set_t::avx2>(
    asmjit::X86Emitter* a) {
  // bottom-left corner
  // zero out
  a->vxorps(resultRegAvx2_, resultRegAvx2_, resultRegAvx2_);
  a->mov(scratchReg1_, H_R_);
  a->sub(scratchReg1_, static_cast<asmjit::Imm>(2 * H_PAD_));
  a->imul(scratchReg1_, W_R_);
  a->imul(scratchReg1_, static_cast<asmjit::Imm>(C_ * sizeof(uint8_t)));
  for (int r = 0; r < R_ - H_PAD_; ++r) {
    if (!isZeroPointZero_) {
      for (int s = 0; s < W_PAD_; ++s) {
        gen8bitFMA<inst_set_t::avx2>(
            a, zeroPTRegAvx2_, WRegs_avx2_[r * S_ + s]);
      }
    }
    for (int s = W_PAD_; s < S_; ++s) {
      a->vbroadcastsd(
          actRegAvx2_,
          x86::dword_ptr(
              in_acts_R_,
              scratchReg1_,
              0,
              (s - W_PAD_) * C_ * sizeof(uint8_t)));
      gen8bitFMA<inst_set_t::avx2>(a, actRegAvx2_, WRegs_avx2_[r * S_ + s]);
    }
    a->imul(scratchReg2_, W_R_, static_cast<asmjit::Imm>(C_ * sizeof(uint8_t)));
    a->add(scratchReg1_, scratchReg2_);
  }
  if (!isZeroPointZero_) {
    for (int r = R_ - H_PAD_; r < R_; ++r) {
      for (int s = 0; s < S_; ++s) {
        gen8bitFMA<inst_set_t::avx2>(
            a, zeroPTRegAvx2_, WRegs_avx2_[r * S_ + s]);
      }
    }
  }

  // we updating the last row
  a->mov(scratchReg1_, H_R_);
  a->sub(scratchReg1_, 1);
  a->imul(scratchReg1_, W_R_);
  a->imul(scratchReg1_, static_cast<asmjit::Imm>(K_ * sizeof(int32_t)));
  a->add(out_acts_R_, scratchReg1_);
  // storeResult<inst_set_t::avx2>(a, (H_-1)*W_*K_*sizeof(int32_t));
  storeResult<inst_set_t::avx2>(a);
  a->add(out_acts_R_, static_cast<asmjit::Imm>(K_ * sizeof(int32_t)));

  // bottom edge excluding corners
  asmjit::Label LoopBottomEdge = a->newLabel();
  a->mov(loopR2_, static_cast<asmjit::Imm>(W_PAD_));
  a->bind(LoopBottomEdge);
  // zero out
  a->vxorps(resultRegAvx2_, resultRegAvx2_, resultRegAvx2_);
  a->mov(scratchReg1_, H_R_);
  a->sub(scratchReg1_, static_cast<asmjit::Imm>(2 * H_PAD_));
  a->imul(scratchReg1_, W_R_);
  a->imul(scratchReg1_, static_cast<asmjit::Imm>(C_ * sizeof(uint8_t)));
  for (int r = 0; r < R_ - W_PAD_; ++r) {
    // int h_in = H_-2*H_PAD_ + r;
    for (int s = 0; s < S_; ++s) {
      a->vbroadcastsd(
          actRegAvx2_,
          x86::dword_ptr(
              in_acts_R_, scratchReg1_, 0, s * C_ * sizeof(uint8_t)));
      gen8bitFMA<inst_set_t::avx2>(a, actRegAvx2_, WRegs_avx2_[r * S_ + s]);
    }
    a->imul(scratchReg2_, W_R_, static_cast<asmjit::Imm>(C_ * sizeof(uint8_t)));
    a->add(scratchReg1_, scratchReg2_);
  }

  if (!isZeroPointZero_) {
    for (int r = R_ - W_PAD_; r < R_; ++r) {
      for (int s = 0; s < S_; ++s) {
        gen8bitFMA<inst_set_t::avx2>(
            a, zeroPTRegAvx2_, WRegs_avx2_[r * S_ + s]);
      }
    }
  }

  a->add(in_acts_R_, static_cast<asmjit::Imm>(C_ * sizeof(uint8_t)));
  // storeResult<inst_set_t::avx2>(a, ((H_-1)*W_+1)*K_*sizeof(int32_t));
  storeResult<inst_set_t::avx2>(a);

  a->add(out_acts_R_, static_cast<asmjit::Imm>(K_ * sizeof(int32_t)));
  a->inc(loopR2_);
  a->mov(loopR1_, W_R_);
  a->sub(loopR1_, static_cast<asmjit::Imm>(W_PAD_));
  a->cmp(loopR2_, loopR1_);
  a->jl(LoopBottomEdge);
  a->mov(scratchReg1_, W_R_);
  a->sub(scratchReg1_, static_cast<asmjit::Imm>(2 * W_PAD_));
  a->imul(scratchReg1_, static_cast<asmjit::Imm>(C_ * sizeof(uint8_t)));
  a->sub(in_acts_R_, scratchReg1_);
  // a->sub(in_acts_R_, (W_ - 2*W_PAD_)*C_*sizeof(uint8_t));
  // a->sub(out_acts_R_, (W_ - 2*W_PAD_)*K_*sizeof(int32_t));

  // bottom-right corner
  // zero out
  a->vxorps(resultRegAvx2_, resultRegAvx2_, resultRegAvx2_);
  // input start point
  // ((H_-(R_-H_PAD_))*W_+(W_-(S_-W_PAD_)))*C_*sizeof(uint8_t)
  a->mov(scratchReg1_, H_R_);
  a->sub(scratchReg1_, static_cast<asmjit::Imm>(R_ - H_PAD_));
  a->imul(scratchReg1_, W_R_);
  a->add(scratchReg1_, W_R_);
  a->sub(scratchReg1_, static_cast<asmjit::Imm>(S_ - W_PAD_));
  a->imul(scratchReg1_, static_cast<asmjit::Imm>(C_ * sizeof(uint8_t)));
  for (int r = 0; r < R_ - H_PAD_; ++r) {
    for (int s = 0; s < S_ - W_PAD_; ++s) {
      a->vbroadcastsd(
          actRegAvx2_,
          x86::dword_ptr(
              in_acts_R_, scratchReg1_, 0, s * C_ * sizeof(uint8_t)));
      gen8bitFMA<inst_set_t::avx2>(a, actRegAvx2_, WRegs_avx2_[r * S_ + s]);
    }
    a->imul(scratchReg2_, W_R_, static_cast<asmjit::Imm>(C_ * sizeof(uint8_t)));
    a->add(scratchReg1_, scratchReg2_);
    if (!isZeroPointZero_) {
      for (int s = S_ - W_PAD_; s < S_; ++s) {
        gen8bitFMA<inst_set_t::avx2>(
            a, zeroPTRegAvx2_, WRegs_avx2_[r * S_ + s]);
      }
    }
  }

  if (!isZeroPointZero_) {
    for (int r = R_ - H_PAD_; r < R_; ++r) {
      for (int s = 0; s < S_; ++s) {
        gen8bitFMA<inst_set_t::avx2>(
            a, zeroPTRegAvx2_, WRegs_avx2_[r * S_ + s]);
      }
    }
  }

  storeResult<inst_set_t::avx2>(a);
  // storeResult<inst_set_t::avx2>(a, ((H_-1)*W_+W_-1)*K_*sizeof(int32_t));
  // reset output pointer
  a->mov(scratchReg1_, H_R_);
  a->sub(scratchReg1_, 1);
  a->imul(scratchReg1_, W_R_);
  a->add(scratchReg1_, W_R_);
  a->sub(scratchReg1_, 1);
  a->imul(scratchReg1_, static_cast<asmjit::Imm>(K_ * sizeof(int32_t)));
  a->sub(out_acts_R_, scratchReg1_);
}

template <>
template <>
void GenConvKernel<int32_t>::genCoreInsts<inst_set_t::avx2>(
    asmjit::X86Emitter* a) {
  // main compute
  asmjit::Label LoopH = a->newLabel();
  asmjit::Label LoopW = a->newLabel();
  // base for output
  a->mov(scratchReg2_, static_cast<asmjit::Imm>(H_PAD_));
  a->imul(scratchReg2_, W_R_);
  a->add(scratchReg2_, static_cast<asmjit::Imm>(W_PAD_));
  a->imul(scratchReg2_, static_cast<asmjit::Imm>(K_ * sizeof(int32_t)));
  a->add(out_acts_R_, scratchReg2_);

  a->mov(scratchReg1_, W_R_);
  a->sub(scratchReg1_, static_cast<asmjit::Imm>(W_PAD_));

  // H loop
  a->mov(loopR1_, static_cast<asmjit::Imm>(H_PAD_));
  a->bind(LoopH);
  // W loop
  a->mov(loopR2_, static_cast<asmjit::Imm>(W_PAD_));
  a->bind(LoopW);
  // zero out
  a->vxorps(resultRegAvx2_, resultRegAvx2_, resultRegAvx2_);
  // compute on all filters
  for (int r = 0; r < R_; ++r) {
    for (int s = 0; s < S_; ++s) {
      a->vbroadcastsd(
          actRegAvx2_, x86::dword_ptr(in_acts_R_, s * C_ * sizeof(uint8_t)));
      gen8bitFMA<inst_set_t::avx2>(a, actRegAvx2_, WRegs_avx2_[r * S_ + s]);
    }
    a->imul(scratchReg2_, W_R_, static_cast<asmjit::Imm>(C_ * sizeof(uint8_t)));
    a->add(in_acts_R_, scratchReg2_);
  }
  a->imul(
      scratchReg2_, W_R_, static_cast<asmjit::Imm>(R_ * C_ * sizeof(uint8_t)));
  a->sub(in_acts_R_, scratchReg2_);
  // a->add(scratchReg1_, C_*sizeof(uint8_t));
  a->add(in_acts_R_, static_cast<asmjit::Imm>(C_ * sizeof(uint8_t)));

  // storeResult<inst_set_t::avx2>(a, (W_+1)*K_*sizeof(int32_t));
  storeResult<inst_set_t::avx2>(a);

  a->add(out_acts_R_, static_cast<asmjit::Imm>(K_ * sizeof(int32_t)));

  a->inc(loopR2_);
  a->cmp(loopR2_, scratchReg1_);
  a->jl(LoopW);
  // add (W_ - 2*W_PAD_)*C_*sizeof(uint8_t) and subtract W_*C_*sizeof(uint8_t)
  a->add(
      in_acts_R_, static_cast<asmjit::Imm>(2 * W_PAD_ * C_ * sizeof(uint8_t)));
  // a->sub(in_acts_R_, (W_ - 2*W_PAD_)*C_*sizeof(uint8_t));
  // a->add(in_acts_R_, W_*C_*sizeof(uint8_t));
  a->add(
      out_acts_R_, static_cast<asmjit::Imm>(2 * W_PAD_ * K_ * sizeof(int32_t)));
  // a->sub(out_acts_R_, (W_ - 2*W_PAD_)*K_*sizeof(int32_t));
  // a->add(out_acts_R_, W_*K_*sizeof(int32_t));

  a->inc(loopR1_);
  a->mov(scratchReg2_, H_R_);
  a->sub(scratchReg2_, static_cast<asmjit::Imm>(H_PAD_));
  a->cmp(loopR1_, scratchReg2_);
  a->jl(LoopH);
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
  H_R_ = a->gpzRef(8);
  W_R_ = a->gpzRef(9);
  row_offset_R_ = a->gpzRef(10);

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
  args.assignAll(in_acts_R_, wghts_R_, out_acts_R_, a_zero_pt_R_, H_R_, W_R_);

  args.updateFrameInfo(ffi);

  asmjit::FuncFrameLayout layout;
  layout.init(func, ffi);

  asmjit::FuncUtils::emitProlog(a, layout);
  asmjit::FuncUtils::allocArgs(a, layout, args);

  createVector16BitOne<inst_set_t::avx2>(a);

  loopR1_ = a->gpzRef(14);
  loopR2_ = a->gpzRef(15);

  if (!isZeroPointZero_) {
    setToZeroPt<inst_set_t::avx2>(a, zeroPTRegAvx2_);
  }

  genForLoadingWeights<inst_set_t::avx2>(a);

  genConstForPermutations<inst_set_t::avx2>(a);

  genForTopEdge<inst_set_t::avx2>(a);
  genForLeftEdge<inst_set_t::avx2>(a);
  genForRightEdge<inst_set_t::avx2>(a);
  genForBottomEdge<inst_set_t::avx2>(a);

  genCoreInsts<inst_set_t::avx2>(a);

  asmjit::FuncUtils::emitEpilog(a, layout);

  jit_conv_kernel_fp fn;
  asmjit::Error err = rt_.add(&fn, &code_);
  if (err) {
    std::cout << "Error: in fn add" << std::endl;
    return nullptr;
  }
  auto kernelSig = getKernelSig(conv_param, isZeroPointZero_);
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
  // top-left corner code
  // zero out the results register
  a->vxorps(resultRegAvx2_, resultRegAvx2_, resultRegAvx2_);
  for (int r = 0; r < R_; ++r) {
    int h_in = -H_PAD_ + r;
    if (h_in >= 0) {
      a->imul(
          scratchReg1_,
          W_R_,
          static_cast<asmjit::Imm>(h_in * C_ * sizeof(uint8_t)));
    }
    for (int s = 0; s < S_; ++s) {
      int w_in = -W_PAD_ + s;
      if (h_in >= 0 && w_in >= 0) {
        a->vmovaps(
            actRegAvx2_,
            x86::dword_ptr(
                in_acts_R_, scratchReg1_, 0, w_in * C_ * sizeof(uint8_t)));
        gen8BitSum<inst_set_t::avx2>(a, actRegAvx2_);
      } else {
        if (!isZeroPointZero_) {
          gen8BitSum<inst_set_t::avx2>(a, zeroPTRegAvx2_);
        }
      }
    }
  }
  // store results
  storeResultRowoffset<inst_set_t::avx2>(a);

  // for C_per_G == 4 and K_per_G == 4, 8 groups processed at a time
  a->add(row_offset_R_, static_cast<asmjit::Imm>(8 * sizeof(int32_t)));

  // top edge excluding corners
  asmjit::Label LoopTopEdge = a->newLabel();
  a->mov(loopR2_, static_cast<asmjit::Imm>(W_PAD_));
  a->bind(LoopTopEdge);
  // zero out
  a->vxorps(resultRegAvx2_, resultRegAvx2_, resultRegAvx2_);
  if (!isZeroPointZero_) {
    for (int r = 0; r < H_PAD_; ++r) {
      for (int s = 0; s < S_; ++s) {
        gen8BitSum<inst_set_t::avx2>(a, zeroPTRegAvx2_);
      }
    }
  }
  for (int r = H_PAD_; r < R_; ++r) {
    int h_in = -H_PAD_ + r;
    a->imul(
        scratchReg1_,
        W_R_,
        static_cast<asmjit::Imm>(h_in * C_ * sizeof(uint8_t)));
    for (int s = 0; s < S_; ++s) {
      a->vmovaps(
          actRegAvx2_,
          x86::dword_ptr(
              in_acts_R_, scratchReg1_, 0, s * C_ * sizeof(uint8_t)));
      gen8BitSum<inst_set_t::avx2>(a, actRegAvx2_);
    }
  }
  a->add(in_acts_R_, static_cast<asmjit::Imm>(C_ * sizeof(uint8_t)));

  // store results
  storeResultRowoffset<inst_set_t::avx2>(a);

  a->add(row_offset_R_, static_cast<asmjit::Imm>(8 * sizeof(int32_t)));
  a->mov(loopR1_, W_R_);
  a->sub(loopR1_, static_cast<asmjit::Imm>(W_PAD_));
  a->inc(loopR2_);
  a->cmp(loopR2_, loopR1_);
  a->jl(LoopTopEdge);
  a->mov(scratchReg2_, W_R_);
  a->imul(scratchReg2_, static_cast<asmjit::Imm>(C_ * sizeof(uint8_t)));
  a->sub(
      scratchReg2_,
      static_cast<asmjit::Imm>(2 * W_PAD_ * C_ * sizeof(uint8_t)));
  a->sub(in_acts_R_, scratchReg2_);

  // top-right corner code
  // zero out
  a->vxorps(resultRegAvx2_, resultRegAvx2_, resultRegAvx2_);
  if (!isZeroPointZero_) {
    for (int r = 0; r < H_PAD_; ++r) {
      for (int s = 0; s < S_; ++s) {
        gen8BitSum<inst_set_t::avx2>(a, zeroPTRegAvx2_);
      }
    }
  }
  for (int r = H_PAD_; r < R_; ++r) {
    int h_in = -H_PAD_ + r;
    for (int s = 0; s < S_ - W_PAD_; ++s) {
      a->imul(
          scratchReg1_,
          W_R_,
          static_cast<asmjit::Imm>(h_in * C_ * sizeof(uint8_t)));
      a->mov(scratchReg2_, W_R_);
      a->sub(scratchReg2_, static_cast<asmjit::Imm>(R_ - W_PAD_ - s));
      a->imul(scratchReg2_, static_cast<asmjit::Imm>(C_ * sizeof(uint8_t)));
      a->add(scratchReg1_, scratchReg2_);
      a->vmovaps(actRegAvx2_, x86::dword_ptr(in_acts_R_, scratchReg1_));
      gen8BitSum<inst_set_t::avx2>(a, actRegAvx2_);
    }
    if (!isZeroPointZero_) {
      for (int s = S_ - W_PAD_; s < S_; ++s) {
        gen8BitSum<inst_set_t::avx2>(a, zeroPTRegAvx2_);
      }
    }
  }

  // store results
  storeResultRowoffset<inst_set_t::avx2>(a);

  a->add(row_offset_R_, static_cast<asmjit::Imm>(8 * sizeof(int32_t)));

  // reset output pointer
  a->imul(scratchReg1_, W_R_, static_cast<asmjit::Imm>(8 * sizeof(int32_t)));
  a->sub(row_offset_R_, scratchReg1_);
}

template <>
template <>
void GenConvKernel<int32_t>::genForLeftEdgeRowoffset<inst_set_t::avx2>(
    asmjit::X86Emitter* a) {
  // left edge excluding corners
  asmjit::Label LoopLeftEdge = a->newLabel();
  a->mov(loopR1_, static_cast<asmjit::Imm>(H_PAD_));
  a->bind(LoopLeftEdge);
  // zero out
  a->vxorps(resultRegAvx2_, resultRegAvx2_, resultRegAvx2_);
  a->mov(scratchReg1_, loopR1_);
  a->sub(scratchReg1_, static_cast<asmjit::Imm>(H_PAD_));
  a->imul(scratchReg1_, W_R_);
  a->imul(scratchReg1_, static_cast<asmjit::Imm>(C_ * sizeof(uint8_t)));
  for (int r = 0; r < R_; ++r) {
    if (!isZeroPointZero_) {
      for (int s = 0; s < W_PAD_; ++s) {
        gen8BitSum<inst_set_t::avx2>(a, zeroPTRegAvx2_);
      }
    }
    for (int s = W_PAD_; s < S_; ++s) {
      a->vmovaps(
          actRegAvx2_,
          x86::dword_ptr(
              in_acts_R_,
              scratchReg1_,
              0,
              (s - W_PAD_) * C_ * sizeof(uint8_t)));
      gen8BitSum<inst_set_t::avx2>(a, actRegAvx2_);
    }
    a->imul(scratchReg2_, W_R_, static_cast<asmjit::Imm>(C_ * sizeof(uint8_t)));
    a->add(scratchReg1_, scratchReg2_);
  }

  a->imul(scratchReg2_, W_R_, static_cast<asmjit::Imm>(8 * sizeof(int32_t)));
  a->add(row_offset_R_, scratchReg2_);
  storeResultRowoffset<inst_set_t::avx2>(a);

  a->inc(loopR1_);
  a->mov(loopR2_, H_R_);
  a->sub(loopR2_, static_cast<asmjit::Imm>(H_PAD_));
  a->cmp(loopR1_, loopR2_);
  a->jl(LoopLeftEdge);

  // reset output pointer
  a->mov(scratchReg2_, H_R_);
  a->sub(scratchReg2_, static_cast<asmjit::Imm>(2 * H_PAD_));
  a->imul(scratchReg2_, W_R_);
  a->imul(scratchReg2_, static_cast<asmjit::Imm>(8 * sizeof(int32_t)));
  a->sub(row_offset_R_, scratchReg2_);
}

template <>
template <>
void GenConvKernel<int32_t>::genForRightEdgeRowoffset<inst_set_t::avx2>(
    asmjit::X86Emitter* a) {
  // right edge excluding corners
  asmjit::Label LoopRightEdge = a->newLabel();

  // output pointer to the right edge
  // (W_ + W_ - 1)*8*sizeof(int32_t)
  a->mov(scratchReg2_, W_R_);
  a->imul(scratchReg2_, 2);
  a->sub(scratchReg2_, 1);
  a->imul(scratchReg2_, static_cast<asmjit::Imm>(8 * sizeof(int32_t)));
  a->add(row_offset_R_, scratchReg2_);

  a->mov(loopR1_, static_cast<asmjit::Imm>(H_PAD_));
  a->bind(LoopRightEdge);
  // zero out
  a->vxorps(resultRegAvx2_, resultRegAvx2_, resultRegAvx2_);
  a->mov(scratchReg1_, loopR1_);
  a->sub(scratchReg1_, static_cast<asmjit::Imm>(H_PAD_));
  a->imul(scratchReg1_, W_R_);
  a->imul(scratchReg1_, static_cast<asmjit::Imm>(C_ * sizeof(uint8_t)));

  a->mov(scratchReg2_, W_R_);
  a->sub(scratchReg2_, static_cast<asmjit::Imm>(2 * W_PAD_));
  a->imul(scratchReg2_, static_cast<asmjit::Imm>(C_ * sizeof(uint8_t)));
  a->add(scratchReg1_, scratchReg2_);
  for (int r = 0; r < R_; ++r) {
    for (int s = 0; s < S_ - W_PAD_; ++s) {
      a->vbroadcastsd(actRegAvx2_, x86::dword_ptr(in_acts_R_, scratchReg1_));
      a->vmovaps(actRegAvx2_, x86::dword_ptr(in_acts_R_, scratchReg1_));
      gen8BitSum<inst_set_t::avx2>(a, actRegAvx2_);
      a->add(scratchReg1_, static_cast<asmjit::Imm>(C_ * sizeof(uint8_t)));
    }
    if (!isZeroPointZero_) {
      for (int s = S_ - W_PAD_; s < S_; ++s) {
        gen8BitSum<inst_set_t::avx2>(a, zeroPTRegAvx2_);
      }
    }

    a->sub(
        scratchReg1_,
        static_cast<asmjit::Imm>((S_ - W_PAD_) * C_ * sizeof(uint8_t)));
    a->imul(scratchReg2_, W_R_, static_cast<asmjit::Imm>(C_ * sizeof(uint8_t)));
    a->add(scratchReg1_, scratchReg2_);
  }

  storeResultRowoffset<inst_set_t::avx2>(a);

  a->imul(scratchReg2_, W_R_, static_cast<asmjit::Imm>(8 * sizeof(int32_t)));
  a->add(row_offset_R_, scratchReg2_);
  a->mov(loopR2_, H_R_);
  a->sub(loopR2_, static_cast<asmjit::Imm>(H_PAD_));
  a->inc(loopR1_);
  a->cmp(loopR1_, loopR2_);
  a->jl(LoopRightEdge);

  // reset base
  a->mov(scratchReg2_, W_R_);
  a->imul(scratchReg2_, 2);
  a->sub(scratchReg2_, 1);
  a->imul(scratchReg2_, static_cast<asmjit::Imm>(8 * sizeof(int32_t)));
  a->sub(row_offset_R_, scratchReg2_);

  // reset increments done in the loop
  //(H_ - 2*H_PAD_)*W_*8*sizeof(int32_t)
  a->mov(scratchReg2_, H_R_);
  a->sub(scratchReg2_, static_cast<asmjit::Imm>(2 * H_PAD_));
  a->imul(scratchReg2_, W_R_);
  a->imul(scratchReg2_, static_cast<asmjit::Imm>(8 * sizeof(int32_t)));
  a->sub(row_offset_R_, scratchReg2_);
}

template <>
template <>
void GenConvKernel<int32_t>::genForBottomEdgeRowoffset<inst_set_t::avx2>(
    asmjit::X86Emitter* a) {
  // bottom-left corner
  // zero out
  a->vxorps(resultRegAvx2_, resultRegAvx2_, resultRegAvx2_);
  a->mov(scratchReg1_, H_R_);
  a->sub(scratchReg1_, static_cast<asmjit::Imm>(2 * H_PAD_));
  a->imul(scratchReg1_, W_R_);
  a->imul(scratchReg1_, static_cast<asmjit::Imm>(C_ * sizeof(uint8_t)));
  for (int r = 0; r < R_ - H_PAD_; ++r) {
    if (!isZeroPointZero_) {
      for (int s = 0; s < W_PAD_; ++s) {
        gen8BitSum<inst_set_t::avx2>(a, zeroPTRegAvx2_);
      }
    }
    for (int s = W_PAD_; s < S_; ++s) {
      a->vmovaps(
          actRegAvx2_,
          x86::dword_ptr(
              in_acts_R_,
              scratchReg1_,
              0,
              (s - W_PAD_) * C_ * sizeof(uint8_t)));
      gen8BitSum<inst_set_t::avx2>(a, actRegAvx2_);
    }
    a->imul(scratchReg2_, W_R_, static_cast<asmjit::Imm>(C_ * sizeof(uint8_t)));
    a->add(scratchReg1_, scratchReg2_);
  }
  if (!isZeroPointZero_) {
    for (int r = R_ - H_PAD_; r < R_; ++r) {
      for (int s = 0; s < S_; ++s) {
        gen8BitSum<inst_set_t::avx2>(a, zeroPTRegAvx2_);
      }
    }
  }

  // we updating the last row
  a->mov(scratchReg1_, H_R_);
  a->sub(scratchReg1_, 1);
  a->imul(scratchReg1_, W_R_);
  a->imul(scratchReg1_, static_cast<asmjit::Imm>(8 * sizeof(int32_t)));
  a->add(row_offset_R_, scratchReg1_);
  storeResultRowoffset<inst_set_t::avx2>(a);
  a->add(row_offset_R_, static_cast<asmjit::Imm>(8 * sizeof(int32_t)));

  // bottom edge excluding corners
  asmjit::Label LoopBottomEdge = a->newLabel();
  a->mov(loopR2_, static_cast<asmjit::Imm>(W_PAD_));
  a->bind(LoopBottomEdge);
  // zero out
  a->vxorps(resultRegAvx2_, resultRegAvx2_, resultRegAvx2_);
  a->mov(scratchReg1_, H_R_);
  a->sub(scratchReg1_, static_cast<asmjit::Imm>(2 * H_PAD_));
  a->imul(scratchReg1_, W_R_);
  a->imul(scratchReg1_, static_cast<asmjit::Imm>(C_ * sizeof(uint8_t)));
  for (int r = 0; r < R_ - W_PAD_; ++r) {
    // int h_in = H_-2*H_PAD_ + r;
    for (int s = 0; s < S_; ++s) {
      a->vmovaps(
          actRegAvx2_,
          x86::dword_ptr(
              in_acts_R_, scratchReg1_, 0, s * C_ * sizeof(uint8_t)));
      gen8BitSum<inst_set_t::avx2>(a, actRegAvx2_);
    }
    a->imul(scratchReg2_, W_R_, static_cast<asmjit::Imm>(C_ * sizeof(uint8_t)));
    a->add(scratchReg1_, scratchReg2_);
  }

  if (!isZeroPointZero_) {
    for (int r = R_ - W_PAD_; r < R_; ++r) {
      for (int s = 0; s < S_; ++s) {
        gen8BitSum<inst_set_t::avx2>(a, zeroPTRegAvx2_);
      }
    }
  }

  a->add(in_acts_R_, static_cast<asmjit::Imm>(C_ * sizeof(uint8_t)));
  // storeResult<inst_set_t::avx2>(a, ((H_-1)*W_+1)*8*sizeof(int32_t));
  storeResultRowoffset<inst_set_t::avx2>(a);

  a->add(row_offset_R_, static_cast<asmjit::Imm>(8 * sizeof(int32_t)));
  a->inc(loopR2_);
  a->mov(loopR1_, W_R_);
  a->sub(loopR1_, static_cast<asmjit::Imm>(W_PAD_));
  a->cmp(loopR2_, loopR1_);
  a->jl(LoopBottomEdge);
  a->mov(scratchReg1_, W_R_);
  a->sub(scratchReg1_, static_cast<asmjit::Imm>(2 * W_PAD_));
  a->imul(scratchReg1_, static_cast<asmjit::Imm>(C_ * sizeof(uint8_t)));
  a->sub(in_acts_R_, scratchReg1_);
  // a->sub(in_acts_R_, (W_ - 2*W_PAD_)*C_*sizeof(uint8_t));
  // a->sub(out_acts_R_, (W_ - 2*W_PAD_)*8*sizeof(int32_t));

  // bottom-right corner
  // zero out
  a->vxorps(resultRegAvx2_, resultRegAvx2_, resultRegAvx2_);
  // input start point
  // ((H_-(R_-H_PAD_))*W_+(W_-(S_-W_PAD_)))*C_*sizeof(uint8_t)
  a->mov(scratchReg1_, H_R_);
  a->sub(scratchReg1_, static_cast<asmjit::Imm>(R_ - H_PAD_));
  a->imul(scratchReg1_, W_R_);
  a->add(scratchReg1_, W_R_);
  a->sub(scratchReg1_, static_cast<asmjit::Imm>(S_ - W_PAD_));
  a->imul(scratchReg1_, static_cast<asmjit::Imm>(C_ * sizeof(uint8_t)));
  for (int r = 0; r < R_ - H_PAD_; ++r) {
    for (int s = 0; s < S_ - W_PAD_; ++s) {
      a->vmovaps(
          actRegAvx2_,
          x86::dword_ptr(
              in_acts_R_, scratchReg1_, 0, s * C_ * sizeof(uint8_t)));
      gen8BitSum<inst_set_t::avx2>(a, actRegAvx2_);
    }
    a->imul(scratchReg2_, W_R_, static_cast<asmjit::Imm>(C_ * sizeof(uint8_t)));
    a->add(scratchReg1_, scratchReg2_);
    if (!isZeroPointZero_) {
      for (int s = S_ - W_PAD_; s < S_; ++s) {
        gen8BitSum<inst_set_t::avx2>(a, zeroPTRegAvx2_);
      }
    }
  }

  if (!isZeroPointZero_) {
    for (int r = R_ - H_PAD_; r < R_; ++r) {
      for (int s = 0; s < S_; ++s) {
        gen8BitSum<inst_set_t::avx2>(a, zeroPTRegAvx2_);
      }
    }
  }

  storeResultRowoffset<inst_set_t::avx2>(a);
  // reset output pointer
  a->mov(scratchReg1_, H_R_);
  a->sub(scratchReg1_, 1);
  a->imul(scratchReg1_, W_R_);
  a->add(scratchReg1_, W_R_);
  a->sub(scratchReg1_, 1);
  a->imul(scratchReg1_, static_cast<asmjit::Imm>(8 * sizeof(int32_t)));
  a->sub(row_offset_R_, scratchReg1_);
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
  a->mov(scratchReg2_, static_cast<asmjit::Imm>(H_PAD_));
  a->imul(scratchReg2_, W_R_);
  a->add(scratchReg2_, static_cast<asmjit::Imm>(W_PAD_));
  a->imul(scratchReg2_, static_cast<asmjit::Imm>(8 * sizeof(int32_t)));
  a->add(row_offset_R_, scratchReg2_);

  a->mov(scratchReg1_, W_R_);
  a->sub(scratchReg1_, static_cast<asmjit::Imm>(W_PAD_));

  // H loop
  a->mov(loopR1_, static_cast<asmjit::Imm>(H_PAD_));
  a->bind(LoopH);
  // W loop
  a->mov(loopR2_, static_cast<asmjit::Imm>(W_PAD_));
  a->bind(LoopW);

  // zero out
  a->vxorps(resultRegAvx2_, resultRegAvx2_, resultRegAvx2_);
  for (int r = 0; r < R_; ++r) {
    for (int s = 0; s < S_; ++s) {
      a->vmovaps(
          actRegAvx2_, x86::dword_ptr(in_acts_R_, s * C_ * sizeof(uint8_t)));
      gen8BitSum<inst_set_t::avx2>(a, actRegAvx2_);
    }
    a->imul(scratchReg2_, W_R_, static_cast<asmjit::Imm>(C_ * sizeof(uint8_t)));
    a->add(in_acts_R_, scratchReg2_);
  }
  a->imul(
      scratchReg2_, W_R_, static_cast<asmjit::Imm>(R_ * C_ * sizeof(uint8_t)));
  a->sub(in_acts_R_, scratchReg2_);
  // store results
  storeResultRowoffset<inst_set_t::avx2>(a);

  a->add(in_acts_R_, static_cast<asmjit::Imm>(C_ * sizeof(uint8_t)));
  a->add(row_offset_R_, static_cast<asmjit::Imm>(8 * sizeof(int32_t)));

  a->inc(loopR2_);
  a->cmp(loopR2_, scratchReg1_);
  a->jl(LoopW);
  a->add(
      in_acts_R_, static_cast<asmjit::Imm>(2 * W_PAD_ * C_ * sizeof(uint8_t)));
  a->add(
      row_offset_R_,
      static_cast<asmjit::Imm>(2 * W_PAD_ * 8 * sizeof(int32_t)));
  a->inc(loopR1_);
  a->mov(scratchReg2_, H_R_);
  a->sub(scratchReg2_, static_cast<asmjit::Imm>(H_PAD_));
  a->cmp(loopR1_, scratchReg2_);
  a->jl(LoopH);
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
  H_R_ = a->zdx();
  W_R_ = a->zcx();
  row_offset_R_ = a->gpzRef(8);

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
  args.assignAll(in_acts_R_, a_zero_pt_R_, H_R_, W_R_, row_offset_R_);

  args.updateFrameInfo(ffi);

  asmjit::FuncFrameLayout layout;
  layout.init(func, ffi);

  asmjit::FuncUtils::emitProlog(a, layout);
  asmjit::FuncUtils::allocArgs(a, layout, args);

  // This uses xmm10 register temporarily. Should come before
  // createVector8BitOne
  if (!isZeroPointZero_) {
    setToZeroPt<inst_set_t::avx2>(a, zeroPTRegAvx2_);
  }

  createVector16BitOne<inst_set_t::avx2>(a);
  // we set ymm10 to contain 8-bit 1s
  createVector8BitOne<inst_set_t::avx2>(a);

  genForTopEdgeRowoffset<inst_set_t::avx2>(a);
  genForLeftEdgeRowoffset<inst_set_t::avx2>(a);
  genForRightEdgeRowoffset<inst_set_t::avx2>(a);
  genForBottomEdgeRowoffset<inst_set_t::avx2>(a);

  genRowoffsetCore<inst_set_t::avx2>(a);

  asmjit::FuncUtils::emitEpilog(a, layout);

  jit_rowoffset_kernel_fp fn;
  asmjit::Error err = rt_.add(&fn, &code_);
  if (err) {
    std::cout << "Error: in fn add" << std::endl;
    return nullptr;
  }
  auto kernelSig = getKernelSig(conv_param, isZeroPointZero_);
  codeCacheRowOffset_[kernelSig] = fn;

#if defined(FBGEMM_LOG_CODE)
  delete codeLogger;
  fclose(codeLogfile);
#endif

  return fn;
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

  int MB = conv_param.MB;
  int H = conv_param.OUT_DIM[0];
  int W = conv_param.OUT_DIM[1];
  int G = conv_param.G;
  int K_per_G = conv_param.OC / G;
  int C_per_G = conv_param.IC / G;
  int oh_ow = conv_param.OUT_DIM[0] * conv_param.OUT_DIM[1];

  static_assert(SPATIAL_DIM == 2, "3D conv not supported yet");

  int32_t* rowOffsetTrDest =
      rowOffsetBuf + 8 * conv_param.IN_DIM[0] * conv_param.IN_DIM[1];
  if (fbgemmOptimizedGConv<SPATIAL_DIM>(conv_param)) {
    assert(G % 8 == 0);
    // generate convolution kernel
    jit_conv_kernel_fp fpConv =
        getOrCreateConvKernel<>(conv_param, a_zero_point);
    // generate row offset kernel
    jit_rowoffset_kernel_fp fpRowoffset =
        getOrCreateRowOffsetKernel(conv_param, a_zero_point);
    for (int i = 0; i < MB; ++i) {
      const uint8_t* actStartBatch = activations +
          i * conv_param.IN_DIM[0] * conv_param.IN_DIM[1] * conv_param.IC;
      for (int gOuter = 0; gOuter < G; gOuter += 8) {
        // for C_per_G == 4 and K_per_G == 4, row offset is calcualted for 8
        // groups at a time The result is row offsets in the format IH*IW x G
        fpRowoffset(
            actStartBatch + gOuter * C_per_G,
            a_zero_point,
            H,
            W,
            rowOffsetBuf);
        // Transpose to get row offsets in the format G x IH*IW
        internal::transpose_8x8(
            conv_param.IN_DIM[0] * conv_param.IN_DIM[1],
            8,
            (const float*)rowOffsetBuf,
            8,
            (float*)rowOffsetTrDest,
            conv_param.IN_DIM[0] * conv_param.IN_DIM[1]);
        int gLimit = gOuter + 8;
        for (int g = gOuter; g < gLimit; g += 2) {
          int32_t* currOutBuf =
              outBuffer + i * oh_ow * conv_param.OC + g * K_per_G;
          const uint8_t* actStartGroup = actStartBatch + g * C_per_G;

          fpConv(
              actStartGroup,
              packed_weights.getBuf() + g * K_per_G * C_per_G,
              currOutBuf,
              a_zero_point,
              H,
              W);

          // Output processing should be called for each group
          for (int j = 0; j < 2; ++j) {
            // calculateRowOffsets(
            // conv_param, actStartGroup, rowOffsetBuf, a_zero_point, j);
            int32_t* rowOffsetForCurG = rowOffsetTrDest +
                ((g - gOuter) + j) * conv_param.IN_DIM[0] *
                    conv_param.IN_DIM[1];
            // compare_buffers(rowOffsetBuf, rowOffsetForCurG,
            // conv_param.IN_DIM[0]*conv_param.IN_DIM[1], 1, 1, 100);

            // outProcess expects rowOffsetBuf to contain row offsets for the
            // current group
            memcpy(
                rowOffsetBuf,
                rowOffsetForCurG,
                conv_param.IN_DIM[0] * conv_param.IN_DIM[1] * sizeof(int32_t));

            if (cpuinfo_has_x86_avx512f()) {
              // Currently use avx2 code
              outProcess.template f<inst_set_t::avx2>(
                  out,
                  currOutBuf + j * K_per_G,
                  {i * oh_ow, oh_ow, (g + j) * K_per_G, K_per_G},
                  K_per_G * G,
                  K_per_G * G);
            } else if (cpuinfo_has_x86_avx2()) {
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
        }
      }
    }
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
        calculateRowOffsets(
            conv_param,
            activations +
                i * conv_param.IN_DIM[0] * conv_param.IN_DIM[1] * conv_param.IC,
            rowOffsetBuf,
            a_zero_point,
            g);
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
    if (cpuinfo_has_x86_avx512f()) {
      int bufferSize = conv_param.OUT_DIM[0] * conv_param.OUT_DIM[1];
      int C_per_G = conv_param.IC / conv_param.G;
      int K_per_G = conv_param.OC / conv_param.G;
      if (C_per_G == 4 && K_per_G == 4) {
        return 2 * 8 * bufferSize;
      } else {
        return conv_param.G * bufferSize;
      }
    } else if (cpuinfo_has_x86_avx2()) {
      int bufferSize = conv_param.OUT_DIM[0] * conv_param.OUT_DIM[1];
      int C_per_G = conv_param.IC / conv_param.G;
      int K_per_G = conv_param.OC / conv_param.G;
      if (C_per_G == 4 && K_per_G == 4) {
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
