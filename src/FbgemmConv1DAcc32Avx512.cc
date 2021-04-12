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
// #include "fbgemm/Fbgemm.h"

namespace fbgemm {

using namespace std;

namespace x86 = asmjit::x86;

INST_DEF_AVX512_AND_VNNI_HEADER
GenConv1DKernel<INST_SET>::initResultRegs(x86::Emitter* a, int nw) {
  // assert(this->nreg_ * 3 < 31 - this->wrIdx && "Too many result registers");

  for (int k = 0; k < this->nreg_ * nw; ++k) {
    a->vpxor(x86::Xmm(31 - k), x86::Xmm(31 - k), x86::Xmm(31 - k));
  }
}

INST_DEF_AVX512_AND_VNNI_HEADER
GenConv1DKernel<INST_SET>::storeOffset(x86::Emitter* a) {
  auto rowOffsetReg_V_Ymm = rowOffsetReg_V_.half();

  a->vextracti32x8(tmpReg1_V_.ymm(), rowOffsetReg_V_, 1);
  a->vpaddd(rowOffsetReg_V_Ymm, tmpReg1_V_.ymm(), rowOffsetReg_V_Ymm);
  a->vpermq(
      tmpReg1_V_.ymm(), rowOffsetReg_V_Ymm, static_cast<asmjit::Imm>(0x4e));

  a->vphaddd(rowOffsetReg_V_Ymm, tmpReg1_V_.ymm(), rowOffsetReg_V_Ymm);
  a->vphaddd(rowOffsetReg_V_Ymm, rowOffsetReg_V_Ymm, rowOffsetReg_V_Ymm);
  a->vphaddd(rowOffsetReg_V_Ymm, rowOffsetReg_V_Ymm, rowOffsetReg_V_Ymm);

  a->vmovd(x86::dword_ptr(row_offset_R_), rowOffsetReg_V_.half().half());
}

INST_DEF_AVX512_AND_VNNI_HEADER
GenConv1DKernel<INST_SET>::mAdd1(
    x86::Emitter* a,
    int b,
    int len,
    int nr,
    int use_zero_flag) {
  using WRegs = x86::Zmm;
  int vsized4 = this->vsize_ / 4;
  int bsize = vsized4 * this->nreg_;

  bool use_zero_reg = use_zero_flag & 0x11;

  a->push(in_acts_R_);

  asmjit::Label LoopIC = a->newLabel();
  asmjit::Label LoopICDone = a->newLabel();
  a->xor_(this->W_R_, this->W_R_);
  a->mov(
      scratchReg1_, static_cast<asmjit::Imm>(this->paddedICPerG_ * bsize * b));
  a->bind(LoopIC);
  a->cmp(W_R_, static_cast<asmjit::Imm>(this->C_per_G_));
  a->jge(LoopICDone);

  if (use_zero_reg) {
    a->vmovapd(actReg_V_, zeroPTReg_V_); // 64 * 8 bit zero points
  } else {
    a->vpbroadcastd(actReg_V_, x86::dword_ptr(in_acts_R_, this->W_R_, 0));
  }

  for (int r = 0; r < nr; r++) {
    a->vmovaps(
        WRegs(this->wrIdx),
        x86::zmmword_ptr(
            wghts_R_, scratchReg1_, 0, (r * this->vsize_) * sizeof(int8_t)));

    genU8I8S32FMA<INST_SET>(
        a,
        actReg_V_,
        WRegs(this->wrIdx),
        x86::Zmm(31 - r),
        oneReg16Bit_V_,
        tmpReg1_V_);
  }

  a->add(W_R_, 4);
  a->add(scratchReg1_, static_cast<asmjit::Imm>(len * 4));
  a->jmp(LoopIC);
  a->bind(LoopICDone);

  a->pop(in_acts_R_);
}

INST_DEF_AVX512_AND_VNNI_HEADER
GenConv1DKernel<INST_SET>::mAdd2(
    x86::Emitter* a,
    int b,
    int len,
    int nr,
    int use_zero_flag) {
  using WRegs = x86::Zmm;
  int vsized4 = this->vsize_ / 4;
  int bsize = vsized4 * this->nreg_;

  bool use_zero_reg = use_zero_flag & 0x1;
  bool use_zero_reg1 = use_zero_flag & 0x2;

  a->push(in_acts_R_);

  asmjit::Label LoopIC = a->newLabel();
  asmjit::Label LoopICDone = a->newLabel();
  a->xor_(this->W_R_, this->W_R_);
  a->mov(
      scratchReg1_, static_cast<asmjit::Imm>(this->paddedICPerG_ * bsize * b));
  a->bind(LoopIC);
  a->cmp(W_R_, static_cast<asmjit::Imm>(this->C_per_G_));
  a->jge(LoopICDone);

  if (use_zero_reg) {
    a->vmovapd(actReg_V_, zeroPTReg_V_); // 64 * 8 bit zero points
    a->vpbroadcastd(actReg_V1_, x86::dword_ptr(in_acts_R_, this->W_R_, 0));
  } else {
    a->vpbroadcastd(actReg_V_, x86::dword_ptr(in_acts_R_, this->W_R_, 0));
    if (use_zero_reg1) {
      a->vmovapd(actReg_V1_, zeroPTReg_V_);
    } else {
      a->vpbroadcastd(
          actReg_V1_, x86::dword_ptr(in_acts_R_, this->W_R_, 0, this->C_));
    }
  }

  for (int r = 0; r < nr; r++) {
    a->vmovaps(
        WRegs(this->wrIdx),
        x86::zmmword_ptr(
            wghts_R_, scratchReg1_, 0, (r * this->vsize_) * sizeof(int8_t)));

    genU8I8S32FMA<INST_SET>(
        a,
        actReg_V_,
        WRegs(this->wrIdx),
        x86::Zmm(31 - r),
        oneReg16Bit_V_,
        tmpReg1_V_);
    genU8I8S32FMA<INST_SET>(
        a,
        actReg_V1_,
        WRegs(this->wrIdx),
        x86::Zmm(31 - r - this->nreg_),
        oneReg16Bit_V_,
        tmpReg1_V_);
  }

  a->add(W_R_, 4);
  a->add(scratchReg1_, static_cast<asmjit::Imm>(len * 4));
  a->jmp(LoopIC);
  a->bind(LoopICDone);

  a->pop(in_acts_R_);
}

INST_DEF_AVX512_AND_VNNI_HEADER
GenConv1DKernel<INST_SET>::mAdd3(
    x86::Emitter* a,
    int b,
    int len,
    int nr,
    int use_zero_flag) {
  using WRegs = x86::Zmm;
  int vsized4 = this->vsize_ / 4;
  int bsize = vsized4 * this->nreg_;

  bool use_zero_reg = use_zero_flag & 0x1;
  bool use_zero_reg1 = use_zero_flag & 0x2;

  a->push(in_acts_R_);

  asmjit::Label LoopIC = a->newLabel();
  asmjit::Label LoopICDone = a->newLabel();
  a->xor_(this->W_R_, this->W_R_);
  a->mov(
      scratchReg1_, static_cast<asmjit::Imm>(this->paddedICPerG_ * bsize * b));
  a->bind(LoopIC);
  a->cmp(W_R_, static_cast<asmjit::Imm>(this->C_per_G_));
  a->jge(LoopICDone);

  if (use_zero_reg) {
    a->vmovapd(actReg_V_, zeroPTReg_V_); // 64 * 8 bit zero points
    a->vpbroadcastd(actReg_V1_, x86::dword_ptr(in_acts_R_, this->W_R_, 0));
    if (use_zero_reg1) {
      a->vmovapd(actReg_V2_, zeroPTReg_V_);
    } else {
      a->vpbroadcastd(
          actReg_V2_, x86::dword_ptr(in_acts_R_, this->W_R_, 0, this->C_));
    }

  } else {
    a->vpbroadcastd(actReg_V_, x86::dword_ptr(in_acts_R_, this->W_R_));
    a->vpbroadcastd(
        actReg_V1_, x86::dword_ptr(in_acts_R_, this->W_R_, 0, this->C_));
    if (use_zero_reg1) {
      a->vmovapd(actReg_V2_, zeroPTReg_V_);
    } else {
      a->vpbroadcastd(
          actReg_V2_, x86::dword_ptr(in_acts_R_, this->W_R_, 0, this->C_ * 2));
    }
  }

  for (int r = 0; r < nr; r++) {
    a->vmovaps(
        WRegs(this->wrIdx),
        x86::zmmword_ptr(
            wghts_R_, scratchReg1_, 0, (r * this->vsize_) * sizeof(int8_t)));

    genU8I8S32FMA<INST_SET>(
        a,
        actReg_V_,
        WRegs(this->wrIdx),
        x86::Zmm(31 - r),
        oneReg16Bit_V_,
        tmpReg1_V_);
    genU8I8S32FMA<INST_SET>(
        a,
        actReg_V1_,
        WRegs(this->wrIdx),
        x86::Zmm(31 - r - this->nreg_),
        oneReg16Bit_V_,
        tmpReg1_V_);
    genU8I8S32FMA<INST_SET>(
        a,
        actReg_V2_,
        WRegs(this->wrIdx),
        x86::Zmm(31 - r - this->nreg_ * 2),
        oneReg16Bit_V_,
        tmpReg1_V_);
  }

  a->add(W_R_, 4);
  a->add(scratchReg1_, static_cast<asmjit::Imm>(len * 4));
  a->jmp(LoopIC);
  a->bind(LoopICDone);

  a->pop(in_acts_R_);
}

INST_DEF_AVX512_AND_VNNI_HEADER
GenConv1DKernel<INST_SET>::genForSingleFilterPoint(
    x86::Emitter* a,
    int nw,
    int use_zero_flag) {

  int vsized4 = this->vsize_ / 4;
  int bsize = vsized4 * this->nreg_;
  int nblock = this->K_per_G_ / bsize;

  for (int b = 0; b < nblock; b++) {
    initResultRegs(a, nw);

    if (nw == 1) {
      mAdd1(a, b, bsize, this->nreg_, use_zero_flag);
    } else {
      if (nw == 2) {
        mAdd2(a, b, bsize, this->nreg_, use_zero_flag);
      } else {
        mAdd3(a, b, bsize, this->nreg_, use_zero_flag);
      }
    }

    for (int n = 0; n < nw; n++) {
      for (int r = 0; r < this->nreg_; r++) {
        int offset = (b * bsize + r * vsized4) * sizeof(int32_t);
        if (this->accum_) {
          a->vpaddd(
              x86::Zmm(31 - r - this->nreg_ * n),
              x86::Zmm(31 - r - this->nreg_ * n),
              x86::zmmword_ptr(
                  out_acts_R_, offset + n * this->K_ * sizeof(int32_t)));
        }
        a->vmovups(
            x86::zmmword_ptr(
                out_acts_R_, offset + n * this->K_ * sizeof(int32_t)),
            x86::Zmm(31 - r - this->nreg_ * n));
      }
    }
  }

  if (this->K_per_G_ > nblock * bsize) {
    initResultRegs(a, nw);
    int nr = (this->K_per_G_ % bsize + vsized4 - 1) / vsized4;

    if (nw == 1) {
      mAdd1(a, nblock, this->paddedOCPerG_ - bsize * nblock, nr, use_zero_flag);
    } else {
      if (nw == 2) {
        mAdd2(
            a, nblock, this->paddedOCPerG_ - bsize * nblock, nr, use_zero_flag);
      } else {
        mAdd3(
            a, nblock, this->paddedOCPerG_ - bsize * nblock, nr, use_zero_flag);
      }
    }

    int rem = this->K_per_G_ % vsized4;
    if (rem) {
      nr--;
    }

    for (int n = 0; n < nw; n++) {
      for (int r = 0; r < nr; r++) {
        int offset = (nblock * bsize + r * vsized4) * sizeof(int32_t);
        if (this->accum_) {
          a->vpaddd(
              x86::Zmm(31 - r - this->nreg_ * n),
              x86::Zmm(31 - r - this->nreg_ * n),
              x86::zmmword_ptr(
                  out_acts_R_, offset + n * this->K_ * sizeof(int32_t)));
        }
        a->vmovups(
            x86::zmmword_ptr(
                out_acts_R_, offset + n * this->K_ * sizeof(int32_t)),
            x86::Zmm(31 - r - this->nreg_ * n));
      }
    }

    // handle the last out channel
    // using mask
    if (rem) {
      a->mov(scratchReg1_, (1 << rem) - 1);
      a->kmovw(x86::k(1), scratchReg1_);
      int offset = (nblock * bsize + nr * vsized4) * sizeof(int32_t);

      for (int n = 0; n < nw; n++) {
        if (this->accum_) {
          a->vpaddd(
              x86::Zmm(31 - nr - this->nreg_ * n),
              x86::Zmm(31 - nr - this->nreg_ * n),
              x86::zmmword_ptr(
                  out_acts_R_, offset + n * this->K_ * sizeof(int32_t)));
        }
        a->k(x86::k(1)).vmovups(
            x86::zmmword_ptr(
                out_acts_R_, offset + n * this->K_ * sizeof(int32_t)),
            x86::Zmm(31 - nr - this->nreg_ * n));
      }
    }
  }

}

#define GENCONVKERNEL_FUNCS(IN)                                     \
  template void GenConv1DKernel<IN>::genForSingleFilterPoint<IN>(   \
      x86::Emitter * a, int nw, int use_zero_flag);                 \
  template void GenConv1DKernel<IN>::mAdd1<IN>(                     \
      x86::Emitter * a, int b, int len, int nr, int use_zero_flag); \
  template void GenConv1DKernel<IN>::mAdd2<IN>(                     \
      x86::Emitter * a, int b, int len, int nr, int use_zero_flag); \
  template void GenConv1DKernel<IN>::mAdd3<IN>(                     \
      x86::Emitter * a, int b, int len, int nr, int use_zero_flag); \
  template void GenConv1DKernel<IN>::storeOffset<IN>(               \
      x86::Emitter * a);                                            \
  template void GenConv1DKernel<IN>::initResultRegs<IN>(            \
      x86::Emitter * a, int nw);

GENCONVKERNEL_FUNCS(inst_set_t::avx512)
GENCONVKERNEL_FUNCS(inst_set_t::avx512_vnni)
#undef GENCONVKERNEL_FUNCS

template class GenConv1DKernel<inst_set_t::avx512>;
template class GenConv1DKernel<inst_set_t::avx512_vnni>;

} // namespace fbgemm
