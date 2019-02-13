/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <iostream>
#include "GenerateKernel.h"

namespace fbgemm {

namespace x86 = asmjit::x86;

/**
 * Generate AVX512 instructions for initializing the C registers to 0 in 32-bit
 * Accumulation kernel.
 */
template <>
template <>
void CodeGenBase<uint8_t, int8_t, int32_t, int32_t>::initCRegs<
    inst_set_t::avx512>(
    asmjit::X86Emitter* a,
    int rowRegs,
    int colRegs,
    int leadingDimCReg) {
  for (int i = 0; i < rowRegs; ++i) {
    for (int j = 0; j < colRegs; ++j) {
      a->vxorps(
          CRegs_avx512_[i * leadingDimCReg + j],
          CRegs_avx512_[i * leadingDimCReg + j],
          CRegs_avx512_[i * leadingDimCReg + j]);
    }
  }
}

/**
 * Generate AVX512 instructions for computing block in the rank-k update of
 * 32-bit Accmulation kernel.
 */
template <>
template <>
void CodeGenBase<uint8_t, int8_t, int32_t, int32_t>::genComputeBlock<
    inst_set_t::avx512>(
    asmjit::X86Emitter* a,
    asmjit::X86Gp buffer_A,
    asmjit::X86Gp buffer_B,
    asmjit::X86Gp B_pf,
    int rowRegs,
    int colRegs,
    int lda,
    int leadingDimCRegAssign) {
  // used for matrix A
  asmjit::X86Zmm AReg = x86::zmm31;

  // used for matrix B
  asmjit::X86Zmm BReg = x86::zmm30;

  // Contains 16-bit 1s
  asmjit::X86Zmm oneReg = x86::zmm29;

  // temporary register
  asmjit::X86Zmm res1 = x86::zmm28;

  for (int j = 0; j < colRegs; ++j) {
    // load B
    a->vmovaps(BReg, x86::dword_ptr(buffer_B, j * VLEN_ * sizeof(int8_t)));
    // load A, broadcast and fmas
    for (int i = 0; i < rowRegs; ++i) {
      a->vpbroadcastd(
          AReg, x86::dword_ptr(buffer_A, (i * lda) * sizeof(uint8_t)));
      a->vpmaddubsw(res1, AReg, BReg);
      a->vpmaddwd(res1, oneReg, res1);
      a->vpaddd(
          CRegs_avx512_[i * leadingDimCRegAssign + j],
          res1,
          CRegs_avx512_[i * leadingDimCRegAssign + j]);
    }
    a->prefetcht0(x86::dword_ptr(B_pf, j * VLEN_ * sizeof(int8_t)));
  }
}

/**
 * Generate AVX512 instructions for storing the C registers back to the memory
 * in 32-bit Accumulation kernel.
 */
template <>
template <>
void CodeGenBase<uint8_t, int8_t, int32_t, int32_t>::storeCRegs<
    inst_set_t::avx512>(
    asmjit::X86Emitter* a,
    int rowRegs,
    int colRegs,
    asmjit::X86Gp C_Offset,
    asmjit::X86Gp ldcReg,
    bool accum,
    int leadingDimCRegAssign) {
  // temp register
  asmjit::X86Zmm tmpReg = x86::zmm28;

  for (int i = 0; i < rowRegs; ++i) {
    if (i != 0) {
      a->add(C_Offset, ldcReg);
    }
    for (int j = 0; j < colRegs; ++j) {
      if (accum) {
        a->vpaddd(
            CRegs_avx512_[i * leadingDimCRegAssign + j],
            CRegs_avx512_[i * leadingDimCRegAssign + j],
            x86::dword_ptr(a->zcx(), C_Offset, 0, j * 16 * sizeof(int32_t)));
      }
      a->vmovups(
          x86::dword_ptr(a->zcx(), C_Offset, 0, j * 16 * sizeof(int32_t)),
          CRegs_avx512_[i * leadingDimCRegAssign + j]);
    }
  }
}

/**
 * Get or Create the AVX512 instructions for 32-bit Accumulation macro-kernel.
 *
 */
template <>
template <>
CodeGenBase<uint8_t, int8_t, int32_t, int32_t>::jit_micro_kernel_fp
CodeGenBase<uint8_t, int8_t, int32_t, int32_t>::getOrCreate<inst_set_t::avx512>(
    bool accum,
    int32_t mc,
    int32_t nc,
    int32_t kc,
    int32_t /* unused */) {
  auto kernelSig = std::make_tuple(accum, mc, nc);
  if (codeCache_.find(kernelSig) != codeCache_.end()) {
    return codeCache_[kernelSig];
  }

  code_.reset(false);
  code_.init(rt_.getCodeInfo());
  asmjit::X86Assembler assembler(&code_);
  asmjit::X86Emitter* a = assembler.asEmitter();
  // ToDo: Dump in a file for debugging
  // code dumping/logging
  // asmjit::FileLogger logger(stderr);
  // code_.setLogger(&logger);

  constexpr int kBlock =
      PackingTraits<int8_t, int32_t, inst_set_t::avx512>::KCB;
  constexpr int nBlock =
      PackingTraits<int8_t, int32_t, inst_set_t::avx512>::NCB;
  constexpr int mRegBlockSize =
      PackingTraits<int8_t, int32_t, inst_set_t::avx512>::MR;
  constexpr int row_interleave =
      PackingTraits<int8_t, int32_t, inst_set_t::avx512>::ROW_INTERLEAVE;
  assert(kc % row_interleave == 0 && "kc must be a multiple of row_interleave");
  // assert(mc <= 12 && "mc must be <= 12 (available registers constraint)");
  int mRegBlocks = mc / mRegBlockSize;
  int mRegBlocksRem = mc % mRegBlockSize;

  // arguments to the function created
  asmjit::X86Gp buffer_A = a->zdi();
  asmjit::X86Gp buffer_B = a->zsi();
  asmjit::X86Gp B_pf = a->zdx();
  asmjit::X86Gp CBase = a->zcx();
  asmjit::X86Gp kSize = a->gpzRef(8);
  asmjit::X86Gp ldcReg = a->gpzRef(9);

  asmjit::FuncDetail func;
  func.init(
      asmjit::
          FuncSignature6<void, uint8_t*, int8_t*, int8_t*, int32_t*, int, int>(
              asmjit::CallConv::kIdHost));

  asmjit::FuncFrameInfo ffi;
  ffi.setDirtyRegs(
      asmjit::X86Reg::kKindVec,
      asmjit::Utils::mask(0, 1, 2, 3, 4, 5, 6, 7) |
          asmjit::Utils::mask(8, 9, 10, 11, 12, 13, 14, 15));
  ffi.setDirtyRegs(
      asmjit::X86Reg::kKindGp, asmjit::Utils::mask(8, 9, 10, 11, 12, 13, 14));

  asmjit::FuncArgsMapper args(&func);
  args.assignAll(buffer_A, buffer_B, B_pf, CBase, kSize, ldcReg);

  args.updateFrameInfo(ffi);

  asmjit::FuncFrameLayout layout;
  layout.init(func, ffi);

  asmjit::FuncUtils::emitProlog(a, layout);
  asmjit::FuncUtils::allocArgs(a, layout, args);

  asmjit::Label Loopk = a->newLabel();
  asmjit::Label LoopMBlocks = a->newLabel();

  asmjit::X86Gp buffer_B_saved = a->gpzRef(10);
  asmjit::X86Gp C_Offset = a->gpzRef(11);
  asmjit::X86Gp B_pf_saved = a->gpzRef(12);
  asmjit::X86Gp iIdx = a->gpzRef(13);
  asmjit::X86Gp kIdx = a->gpzRef(14);
  // asmjit::X86Gp B_pf = a->gpzRef(8);

  asmjit::X86Zmm oneReg = x86::zmm29;
  // create 16-bit 1s
  // i.e., oneReg[0:15] contains 0x0001, oneReg[16:31] contains 0x0001
  // and so on
  // a->vpcmpeqw(oneReg, oneReg, oneReg);
  a->vpternlogd(oneReg, oneReg, oneReg, 0xff);
  a->vpsrlw(oneReg, oneReg, 15);
  a->imul(ldcReg, ldcReg, static_cast<asmjit::Imm>(sizeof(int32_t)));
  a->mov(C_Offset, 0);

  int colRegs = nc * row_interleave * sizeof(int8_t) / VLEN_;
  if (mRegBlocks > 0) {
    // move 0 to iteration variables
    a->mov(iIdx, 0);

    // save B_buffer address
    a->mov(buffer_B_saved, buffer_B);
    a->mov(B_pf_saved, B_pf);

    a->bind(LoopMBlocks);
    a->inc(iIdx);

    int rowRegs = mRegBlockSize;

    // init C registers
    initCRegs<inst_set_t::avx512>(a, rowRegs, colRegs, colRegs);

    // init k loop index
    a->mov(kIdx, 0);
    a->bind(Loopk);

    // k is incremented by row_interleave
    a->add(kIdx, static_cast<asmjit::Imm>(row_interleave));

    genComputeBlock<inst_set_t::avx512>(
        a, buffer_A, buffer_B, B_pf, rowRegs, colRegs, kBlock, colRegs);

    // update buffer_A address for next k iteration
    a->add(
        buffer_A, static_cast<asmjit::Imm>(row_interleave * sizeof(uint8_t)));

    // update buffer_B address for next k iteration
    a->add(
        buffer_B,
        static_cast<asmjit::Imm>(nBlock * row_interleave * sizeof(int8_t)));
    a->add(
        B_pf,
        static_cast<asmjit::Imm>(nBlock * row_interleave * sizeof(int8_t)));

    // a->add(B_pf, static_cast<asmjit::Imm>(32*sizeof(float)));

    a->cmp(kIdx, kSize);
    a->jl(Loopk);

    // store C matrix
    storeCRegs<inst_set_t::avx512>(
        a, rowRegs, colRegs, C_Offset, ldcReg, accum, colRegs);

    // increment A for next block
    a->sub(buffer_A, kSize);
    a->add(
        buffer_A, static_cast<asmjit::Imm>((rowRegs)*kBlock * sizeof(uint8_t)));

    // increment C for next block
    a->imul(C_Offset, ldcReg, static_cast<asmjit::Imm>(rowRegs));
    a->add(CBase, C_Offset);
    a->mov(C_Offset, 0);

    // reset B
    a->mov(buffer_B, buffer_B_saved);
    a->mov(B_pf, B_pf_saved);
    a->cmp(iIdx, mRegBlocks);
    a->jl(LoopMBlocks);
  }
  // generate code for remainder
  if (mRegBlocksRem > 0) {
    asmjit::Label LoopkRem = a->newLabel();
    int rowRegs = mRegBlocksRem;

    // init C registers
    initCRegs<inst_set_t::avx512>(a, rowRegs, colRegs, colRegs);

    // init k loop index
    a->mov(kIdx, 0);
    a->bind(LoopkRem);

    // k is incremented by row_interleave
    a->add(kIdx, static_cast<asmjit::Imm>(row_interleave));

    genComputeBlock<inst_set_t::avx512>(
        a, buffer_A, buffer_B, B_pf, rowRegs, colRegs, kBlock, colRegs);

    // update buffer_A address for next k iteration
    a->add(
        buffer_A, static_cast<asmjit::Imm>(row_interleave * sizeof(uint8_t)));

    // update buffer_B address for next k iteration
    a->add(
        buffer_B,
        static_cast<asmjit::Imm>(nBlock * row_interleave * sizeof(int8_t)));
    a->add(
        B_pf,
        static_cast<asmjit::Imm>(nBlock * row_interleave * sizeof(int8_t)));

    a->cmp(kIdx, kSize);
    a->jl(LoopkRem);

    // store C matrix
    storeCRegs<inst_set_t::avx512>(
        a, rowRegs, colRegs, C_Offset, ldcReg, accum, colRegs);
  }

  asmjit::FuncUtils::emitEpilog(a, layout);

  jit_micro_kernel_fp fn;
  asmjit::Error err = rt_.add(&fn, &code_);
  if (err) {
    std::cout << "Error: in fn add" << std::endl;
    return nullptr;
  }
  codeCache_[kernelSig] = fn;
  return fn;
}

} // namespace fbgemm
