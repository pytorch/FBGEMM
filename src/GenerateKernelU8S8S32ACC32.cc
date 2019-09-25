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
 * Generate AVX2 instructions for initializing the C registers to 0 in 32-bit
 * Accumulation kernel.
 */
template <>
template <>
void CodeGenBase<uint8_t, int8_t, int32_t, int32_t>::initCRegs<
    inst_set_t::avx2>(
    x86::Emitter* a,
    int rowRegs,
    int colRegs,
    int leadingDimCReg) {
  using CRegs = x86::Ymm;
  for (int i = 0; i < rowRegs; ++i) {
    for (int j = 0; j < colRegs; ++j) {
      a->vxorps(
          CRegs(i * leadingDimCReg + j),
          CRegs(i * leadingDimCReg + j),
          CRegs(i * leadingDimCReg + j));
    }
  }
}

/**
 * Generate AVX2 instructions for computing block in the rank-k update of 32-bit
 * Accmulation kernel.
 */
template <>
template <>
void CodeGenBase<uint8_t, int8_t, int32_t, int32_t>::genComputeBlock<
    inst_set_t::avx2>(
    x86::Emitter* a,
    x86::Gp buffer_A,
    x86::Gp buffer_B,
    x86::Gp B_pf,
    int rowRegs,
    int colRegs,
    int lda,
    int leadingDimCReg) {
  // used for matrix A
  x86::Ymm AReg = x86::ymm12;

  // used for matrix B
  x86::Ymm BReg = x86::ymm13;

  // Contains 16-bit 1s
  x86::Ymm oneReg = x86::ymm15;

  // temporary register
  x86::Ymm res1 = x86::ymm14;

  using CRegs = x86::Ymm;

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
          CRegs(i * leadingDimCReg + j),
          res1,
          CRegs(i * leadingDimCReg + j));
    }
    a->prefetcht0(x86::dword_ptr(B_pf, j * VLEN_ * sizeof(int8_t)));
  }
}

/**
 * Generate AVX2 instructions for storing the C registers back to the memory in
 * 32-bit Accumulation kernel.
 */
template <>
template <>
void CodeGenBase<uint8_t, int8_t, int32_t, int32_t>::storeCRegs<
    inst_set_t::avx2>(
    x86::Emitter* a,
    int rowRegs,
    int colRegs,
    x86::Gp C_Offset,
    x86::Gp ldcReg,
    bool accum,
    int leadingDimCReg) {
  using CRegs = x86::Ymm;
  for (int i = 0; i < rowRegs; ++i) {
    if (i != 0) {
      a->add(C_Offset, ldcReg);
    }
    for (int j = 0; j < colRegs; ++j) {
      if (accum) {
        a->vpaddd(
            CRegs(i * leadingDimCReg + j),
            CRegs(i * leadingDimCReg + j),
#ifdef _MSC_VER
            x86::dword_ptr(a->gpz(9), C_Offset, 0, j * 8 * sizeof(int32_t)));
#else
            x86::dword_ptr(a->zcx(), C_Offset, 0, j * 8 * sizeof(int32_t)));
#endif
      }
      a->vmovups(
#ifdef _MSC_VER
          x86::dword_ptr(a->gpz(9), C_Offset, 0, j * 8 * sizeof(int32_t)),
#else
          x86::dword_ptr(a->zcx(), C_Offset, 0, j * 8 * sizeof(int32_t)),
#endif
          CRegs(i * leadingDimCReg + j));
    }
  }
}

/**
 * Get or Create the AVX2 instructions for 32-bit Accumulation macro-kernel.
 *
 */
template <>
template <>
CodeGenBase<uint8_t, int8_t, int32_t, int32_t>::jit_micro_kernel_fp
CodeGenBase<uint8_t, int8_t, int32_t, int32_t>::getOrCreate<inst_set_t::avx2>(
    bool accum,
    int32_t mc,
    int32_t nc,
    int32_t kc,
    int32_t /* unused */) {
  std::tuple<bool, int, int, int, int, int, int, int> kernelSig;
  int kBlock;
  int nBlock;
  int mRegBlockSize;
  int nRegBlockSize;
  int nRegBlockSizeMin;
  int row_interleave;

  if (blocking_params) {
    kBlock = blocking_params->KCB;
    nBlock = blocking_params->NCB;
    mRegBlockSize = blocking_params->MR;
    nRegBlockSize = blocking_params->NR;
    nRegBlockSizeMin = blocking_params->NR_MIN;
    row_interleave = blocking_params->ROW_INTERLEAVE;
  } else {
    kBlock = PackingTraits<uint8_t, int32_t, inst_set_t::avx2>::KCB;
    nBlock = PackingTraits<uint8_t, int32_t, inst_set_t::avx2>::NCB;
    mRegBlockSize = PackingTraits<uint8_t, int32_t, inst_set_t::avx2>::MR;
    nRegBlockSize = PackingTraits<uint8_t, int32_t, inst_set_t::avx2>::NR;
    nRegBlockSizeMin =
        PackingTraits<uint8_t, int32_t, inst_set_t::avx2>::NR_MIN;
    row_interleave =
        PackingTraits<uint8_t, int32_t, inst_set_t::avx2>::ROW_INTERLEAVE;
  }

  kernelSig = std::make_tuple(
      accum,
      mc,
      nc,
      nBlock,
      kBlock,
      mRegBlockSize,
      nRegBlockSize,
      nRegBlockSizeMin);

  return codeCache_.getOrCreate(kernelSig, [&]() -> jit_micro_kernel_fp {
    asmjit::CodeHolder code;
    code.init(runtime().codeInfo());
    x86::Assembler assembler(&code);
    x86::Emitter *a = assembler.as<x86::Emitter>();
#if defined(FBGEMM_LOG_CODE)
    // generated code logging
    FILE *codeLogfile = fopen(getCodeLoggingFile<inst_set_t::avx2>(
                                  accum, mc, nc, nBlock, kBlock, mRegBlockSize,
                                  nRegBlockSize, nRegBlockSizeMin)
                                  .c_str(),
                              "w");
    asmjit::FileLogger *codeLogger = new asmjit::FileLogger(codeLogfile);
    if (codeLogger) {
      code.setLogger(codeLogger);
    }
#endif

    // assert(mc <= 12 && "mc must be <= 12 (available registers constraint)");
    int mRegBlocks = mc / mRegBlockSize;
    int mRegBlocksRem = mc % mRegBlockSize;

    // arguments to the function created
#ifdef _MSC_VER
    x86::Gp buffer_A = a->zcx();
    x86::Gp buffer_B = a->zdx();
    x86::Gp B_pf = a->gpz(8);
    x86::Gp CBase = a->gpz(9);
    x86::Gp kSize = a->zdi();
    x86::Gp ldcReg = a->zsi();
#else
    x86::Gp buffer_A = a->zdi();
    x86::Gp buffer_B = a->zsi();
    x86::Gp B_pf = a->zdx();
    x86::Gp CBase = a->zcx();
    x86::Gp kSize = a->gpz(8);
    x86::Gp ldcReg = a->gpz(9);
#endif

    asmjit::FuncDetail func;
    func.init(
        asmjit::FuncSignatureT<void, uint8_t *, int8_t *, int8_t *, int32_t *,
                               int, int>(asmjit::CallConv::kIdHost));

    asmjit::FuncFrame frame;
    frame.init(func);
    frame.setDirtyRegs(
        x86::Reg::kGroupVec,
        asmjit::Support::bitMask(0, 1, 2, 3, 4, 5, 6, 7) |
            asmjit::Support::bitMask(8, 9, 10, 11, 12, 13, 14, 15));
    frame.setDirtyRegs(x86::Reg::kGroupGp,
                       asmjit::Support::bitMask(8, 9, 10, 11, 12, 13, 14));

    asmjit::FuncArgsAssignment args(&func);
    args.assignAll(buffer_A, buffer_B, B_pf, CBase, kSize, ldcReg);

    args.updateFuncFrame(frame);
    frame.finalize();

    a->emitProlog(frame);
    a->emitArgsAssignment(frame, args);

    asmjit::Label LoopMBlocks = a->newLabel();

    x86::Gp buffer_B_saved = a->gpz(10);
    x86::Gp C_Offset = a->gpz(11);
    x86::Gp B_pf_saved = a->gpz(12);
    x86::Gp iIdx = a->gpz(13);
    x86::Gp kIdx = a->gpz(14);
    // x86::Gp B_pf = a->gpz(8);

    x86::Ymm oneReg = x86::ymm15;
    // create 16-bit 1s
    // i.e., oneReg[0:15] contains 0x0001, oneReg[16:31] contains 0x0001
    // and so on
    a->vpcmpeqw(oneReg, oneReg, oneReg);
    a->vpsrlw(oneReg, oneReg, 15);
    a->imul(ldcReg, ldcReg, static_cast<asmjit::Imm>(sizeof(int32_t)));
    a->mov(C_Offset, 0);

    int colRegs = nc * row_interleave * sizeof(int8_t) / VLEN_;

    auto issueLoopOverK = [&](int rowRegs) {
      asmjit::Label LoopKLabel = a->newLabel();

      // Init C (result) vector registers
      initCRegs<inst_set_t::avx2>(a, rowRegs, colRegs, colRegs);

      // Loops over K
      a->mov(kIdx, 0);
      a->bind(LoopKLabel);

      // k is incremented by row_interleave
      a->add(kIdx, static_cast<asmjit::Imm>(row_interleave));

      genComputeBlock<inst_set_t::avx2>(a, buffer_A, buffer_B, B_pf, rowRegs,
                                        colRegs, kBlock, colRegs);

      // update buffer_A address for next k iteration
      a->add(buffer_A,
             static_cast<asmjit::Imm>(row_interleave * sizeof(uint8_t)));

      // update buffer_B address for next k iteration
      a->add(buffer_B, static_cast<asmjit::Imm>(nBlock * row_interleave *
                                                sizeof(int8_t)));
      a->add(B_pf, static_cast<asmjit::Imm>(nBlock * row_interleave *
                                            sizeof(int8_t)));

      a->cmp(kIdx, kSize);
      a->jl(LoopKLabel);

      // store C matrix
      storeCRegs<inst_set_t::avx2>(a, rowRegs, colRegs, C_Offset, ldcReg, accum,
                                   colRegs);
    };

    if (mRegBlocks > 0) {
      // move 0 to iteration variables
      a->mov(iIdx, 0);

      // save B_buffer address
      a->mov(buffer_B_saved, buffer_B);
      a->mov(B_pf_saved, B_pf);

      a->bind(LoopMBlocks);
      a->inc(iIdx);

      issueLoopOverK(mRegBlockSize);

      int rowRegs = mRegBlockSize;
      // increment A for next block
      a->sub(buffer_A, kSize);
      a->add(buffer_A,
             static_cast<asmjit::Imm>((rowRegs)*kBlock * sizeof(uint8_t)));

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
        issueLoopOverK(mRegBlocksRem);
    }

    a->emitEpilog(frame);

    jit_micro_kernel_fp fn;
    asmjit::Error err;
    {
      std::unique_lock<std::mutex> lock(rtMutex_);
      err = runtime().add(&fn, &code);
    }
    if (err) {
      std::cout << "Error: in fn add" << std::endl;
      return nullptr;
    }

#if defined(FBGEMM_LOG_CODE)
    fclose(codeLogfile);
    delete codeLogger;
#endif

    return fn;
  });
}

} // namespace fbgemm
