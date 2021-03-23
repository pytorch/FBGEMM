/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#define FBGEMM_EXPORTS
#include <asmjit/asmjit.h>
#include "./CodeGenHelpers.h"
#include "./FbgemmConv1D.h"
#include "fbgemm/Fbgemm.h"

namespace fbgemm {

using namespace std;

namespace x86 = asmjit::x86;

INST_DEF_AVX2_HEADER GenConv1DKernel<INST_SET>::initResultRegs(
    x86::Emitter* a, int nw) {
  for (int k = 0; k < this->nreg_ * nw; ++k) {
    a->vpxor(x86::Xmm(15 - k), x86::Xmm(15 - k), x86::Xmm(15 - k));
  }
}

INST_DEF_AVX2_HEADER GenConv1DKernel<INST_SET>::storeOffset(
    x86::Emitter* a) {
  a->vpermq(tmpReg1_V_, rowOffsetReg_V_, static_cast<asmjit::Imm>(0x4e));
  a->vphaddd(rowOffsetReg_V_, tmpReg1_V_, rowOffsetReg_V_);
  a->vphaddd(rowOffsetReg_V_, rowOffsetReg_V_, rowOffsetReg_V_);
  a->vphaddd(rowOffsetReg_V_, rowOffsetReg_V_, rowOffsetReg_V_);
  a->vmovd(x86::dword_ptr(row_offset_R_), rowOffsetReg_V_.half());
}

INST_DEF_AVX2_HEADER GenConv1DKernel<INST_SET>::genForSingleFilterPoint(
    x86::Emitter* a,
    int nw,
    int use_zero_flag) {
  using WRegs = x86::Ymm;

  bool use_zero_reg = use_zero_flag & 0x1;

  // asmjit::Label LoopCout = a->newLabel();
  // asmjit::Label LoopCin  = a->newLabel();
  // asmjit::Label LoopCoutRemainder  = a->newLabel();
  // asmjit::Label LoopCinRemainder  = a->newLabel();

  int vsized4 = this->vsize_ / 4;
  int bsize = vsized4 * this->nreg_;
  int nblock = this->K_per_G_ / bsize;

  for (int b = 0; b < nblock; b++) {
    initResultRegs(a, nw);
    for (int ic = 0; ic < this->C_per_G_; ic += 4) {
      // get a
      if (use_zero_reg) {
        a->vmovapd(actReg_V_, zeroPTReg_V_);
      } else {
        a->vpbroadcastd(
            actReg_V_, x86::dword_ptr(in_acts_R_, (ic) * sizeof(uint8_t)));
      }

      // get b
      int idx = this->paddedICPerG_ * bsize * b + ic * bsize;

      for (int r = 0; r < this->nreg_; r++) {
        a->vmovaps(
            WRegs(this->wrIdx),
            x86::dword_ptr(
                wghts_R_, (idx + r * this->vsize_) * sizeof(int8_t)));

        genU8I8S32FMA<INST_SET>(
            a,
            actReg_V_,
            WRegs(this->wrIdx),
            x86::Ymm(15 - r),
            oneReg16Bit_V_,
            tmpReg1_V_);
      }
    }

    for (int r = 0; r < this->nreg_; r++) {
      int offset = (b * bsize + r * vsized4) * sizeof(int32_t);
      if (this->accum_) {
        a->vpaddd(
            x86::Ymm(15 - r),
            x86::Ymm(15 - r),
            x86::dword_ptr(out_acts_R_, offset));
      }
      a->vmovups(x86::dword_ptr(out_acts_R_, offset), x86::Ymm(15 - r));
    }
  }

  if (this->K_per_G_ > nblock * bsize) {
    initResultRegs(a, nw);
    int nr = (this->K_per_G_ % bsize + vsized4 - 1) / vsized4;
    for (int ic = 0; ic < this->C_per_G_; ic += 4) {
      // get a
      if (use_zero_reg) {
        a->vmovapd(actReg_V_, zeroPTReg_V_);
      } else {
        a->vpbroadcastd(
            actReg_V_,
            x86::dword_ptr(
                in_acts_R_, (ic) * sizeof(uint8_t))); // TODO add G effect
      }

      // get b
      int idx = this->paddedICPerG_ * bsize * nblock +
          ic * (this->paddedOCPerG_ - bsize * nblock);

      for (int r = 0; r < nr; r++) {
        a->vmovaps(
            WRegs(this->wrIdx),
            x86::dword_ptr(
                wghts_R_, (idx + r * this->vsize_) * sizeof(int8_t)));

        genU8I8S32FMA<INST_SET>(
            a,
            actReg_V_,
            WRegs(this->wrIdx),
            x86::Ymm(15 - r),
            oneReg16Bit_V_,
            tmpReg1_V_);
      }
    }

    int rem = this->K_per_G_ % vsized4;
    if (rem) {
      nr--;
    }

    for (int r = 0; r < nr; r++) {
      int offset = (nblock * bsize + r * vsized4) *
          sizeof(int32_t); // TODO : should be in bytes ?
      if (this->accum_) {
        a->vpaddd(
            x86::Ymm(15 - r),
            x86::Ymm(15 - r),
            x86::dword_ptr(out_acts_R_, offset));
      }
      a->vmovups(x86::dword_ptr(out_acts_R_, offset), x86::Ymm(15 - r));
    }

    // handle the last out channel
    // check x86::r13d : using scratchReg2_ ?
    // use vpmaskmovd ?

    int wr = 15 - nr;
    int offset = (nblock * bsize + nr * vsized4) * sizeof(int32_t);
    if (this->accum_) {
      a->vpaddd(
          x86::Ymm(wr), x86::Ymm(wr), x86::dword_ptr(out_acts_R_, offset));
    }

    switch (this->K_per_G_ % 8) {
      case 0:
        break;
      case 1:
        a->pextrd(x86::r13d, x86::Xmm(wr), static_cast<asmjit::Imm>(0x0));
        a->mov(x86::dword_ptr(out_acts_R_, offset), x86::r13d);
        break;

      case 2:
        a->pextrq(x86::r13, x86::Xmm(wr), static_cast<asmjit::Imm>(0x0));
        a->mov(x86::dword_ptr(out_acts_R_, offset), x86::r13);
        break;

      case 3:
        a->pextrq(x86::r13, x86::Xmm(wr), static_cast<asmjit::Imm>(0x0));
        a->mov(x86::dword_ptr(out_acts_R_, offset), x86::r13);

        a->pextrd(x86::r13d, x86::Xmm(wr), static_cast<asmjit::Imm>(0x2));
        a->mov(x86::dword_ptr(out_acts_R_, offset + 8), x86::r13d);
        break;

      case 4:
        a->movups(
            x86::dword_ptr(out_acts_R_, offset),
            x86::Xmm(wr)); // TODO: check here, use 9 before
        break;

      case 5:
        a->movups(x86::dword_ptr(out_acts_R_, offset), x86::Xmm(wr));
        a->pextrd(x86::r13d, x86::Xmm(wr), static_cast<asmjit::Imm>(0x4));
        a->mov(x86::dword_ptr(out_acts_R_, offset + 16), x86::r13d);
        break;

      case 6:
        a->movups(x86::dword_ptr(out_acts_R_, offset), x86::Xmm(wr));
        a->pextrq(x86::r13, x86::Xmm(wr), static_cast<asmjit::Imm>(0x4));
        a->mov(x86::dword_ptr(out_acts_R_, offset + 16), x86::r13);
        break;

      case 7:
        a->movups(x86::dword_ptr(out_acts_R_, offset), x86::Xmm(wr));
        a->pextrq(x86::r13, x86::Xmm(wr), static_cast<asmjit::Imm>(0x4));
        a->mov(x86::dword_ptr(out_acts_R_, offset + 16), x86::r13);
        a->pextrd(x86::r13d, x86::Xmm(wr), static_cast<asmjit::Imm>(0x6));
        a->mov(x86::dword_ptr(out_acts_R_, offset + 24), x86::r13d);
        break;

      default:
        assert(0 && "not supported case for Cout");
    }
  }

  // handle offsets
  if (this->needRowOffset_) {
    for (int j = 0; j < this->cinLoopIters_; j++) {
      if (use_zero_reg) {
        a->vmovapd(actReg_V_, zeroPTReg_V_);
      } else {
        a->vmovupd(
            actReg_V_, x86::dword_ptr(in_acts_R_, (j * 32) * sizeof(uint8_t)));
      }
      if (this->cinLoopRemainder_ > 0 && j == this->cinLoopIters_ - 1) {
        if (this->cinLoopRemainder_ > 16) {
          a->vpslldq(
              tmpReg1_V_,
              actReg_V_,
              static_cast<asmjit::Imm>(16 - this->cinLoopRemainder_ % 16));
          a->vpsrldq(
              tmpReg1_V_,
              tmpReg1_V_,
              static_cast<asmjit::Imm>(16 - this->cinLoopRemainder_ % 16));
          a->vpblendd(
              actReg_V_, actReg_V_, tmpReg1_V_, static_cast<asmjit::Imm>(0xF0));
        } else {
          if (this->cinLoopRemainder_ < 16) {
            a->vpslldq(
                actReg_V_,
                actReg_V_,
                static_cast<asmjit::Imm>(16 - this->cinLoopRemainder_ % 16));
            a->vpsrldq(
                actReg_V_,
                actReg_V_,
                static_cast<asmjit::Imm>(16 - this->cinLoopRemainder_ % 16));
          }
          a->vpxor(tmpReg1_V_, tmpReg1_V_, tmpReg1_V_);
          a->vpblendd(
              actReg_V_, actReg_V_, tmpReg1_V_, static_cast<asmjit::Imm>(0xF0));
        }
      }

      genU8Sum8(a, actReg_V_, rowOffsetReg_V_, tmpReg1_V_);
    }
  }
}

#define GENCONVKERNEL_FUNCS(IN)                                   \
  template void GenConv1DKernel<IN>::genForSingleFilterPoint<IN>( \
      x86::Emitter * a, int nw, int use_zero_flag);               \
  template void GenConv1DKernel<IN>::storeOffset<IN>(             \
      x86::Emitter * a);                                          \
  template void GenConv1DKernel<IN>::initResultRegs<IN>(x86::Emitter * a, int nw);

GENCONVKERNEL_FUNCS(inst_set_t::avx2)
#undef GENCONVKERNEL_FUNCS

template class GenConv1DKernel<inst_set_t::avx2>;

} // namespace fbgemm
