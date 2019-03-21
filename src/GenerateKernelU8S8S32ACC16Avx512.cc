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
 * Generate AVX512 instructions for initializing the C registers to 0 in 16-bit
 * Accumulation kernel.
 */
template <>
template <>
void CodeGenBase<uint8_t, int8_t, int32_t, int16_t>::initCRegs<
    inst_set_t::avx512>(
    asmjit::X86Emitter* a,
    int rowRegs,
    int colRegs,
    int leadingDimCRegAssign) {
  for (int i = 0; i < rowRegs; ++i) {
    for (int j = 0; j < colRegs; ++j) {
      a->vxorps(
          CRegs_avx512_[i * leadingDimCRegAssign + j],
          CRegs_avx512_[i * leadingDimCRegAssign + j],
          CRegs_avx512_[i * leadingDimCRegAssign + j]);
    }
  }
}

/**
 * Generate AVX512 instructions for computing block in the rank-k update of
 * 16-bit Accmulation kernel.
 */
template <>
template <>
void CodeGenBase<uint8_t, int8_t, int32_t, int16_t>::genComputeBlock<
    inst_set_t::avx512>(
    asmjit::X86Emitter* a,
    asmjit::X86Gp buffer_A,
    asmjit::X86Gp buffer_B,
    asmjit::X86Gp /* unused (reserved for prefetching)*/,
    int rowRegs,
    int colRegs,
    int lda,
    int leadingDimCRegAssign) {
  // used for matrix A
  asmjit::X86Zmm AReg = x86::zmm29;

  asmjit::X86Zmm tmpReg = x86::zmm30;

  // We start allocating BRegs from zmm27 and then allocate zmm26 and so on.
  for (int j = 0; j < colRegs; ++j) {
    a->vmovups(
        AllRegs_avx512_[27 - j],
        x86::dword_ptr(buffer_B, j * VLEN_ * sizeof(int8_t)));
  }

  for (int i = 0; i < rowRegs; ++i) {
    // broadcast A
    a->vpbroadcastw(
        AReg, x86::dword_ptr(buffer_A, (i * lda) * sizeof(uint8_t)));
    for (int j = 0; j < colRegs; ++j) {
      a->vpmaddubsw(
          tmpReg, AReg, AllRegs_avx512_[27-j]);
      a->vpaddsw(
          CRegs_avx512_[i * leadingDimCRegAssign + j],
          tmpReg,
          CRegs_avx512_[i * leadingDimCRegAssign + j]);
      // Prefetching is hurting performance in some cases
      // because prefetch instructions itself consumes a slot
      // in pipeline issue thus slowing down the kernel.
      // if((i == rowRegs - 1) && j % 2 == 0){
      // a->prefetcht0(x86::dword_ptr(B_pf, j*VLEN_*sizeof(int8_t)));
      //}
    }
  }
}

/**
 * Generate AVX512 instructions for storing the C registers back to the memory
 * in 16-bit Accumulation kernel.
 */
template <>
template <>
void CodeGenBase<uint8_t, int8_t, int32_t, int16_t>::storeCRegs<
    inst_set_t::avx512>(
    asmjit::X86Emitter* a,
    int rowRegs,
    int colRegs,
    asmjit::X86Gp C_Offset,
    asmjit::X86Gp ldcReg,
    bool accum,
    int leadingDimCRegAssign) {
  asmjit::X86Ymm extractDest256 = x86::ymm31;
  asmjit::X86Zmm extractDest512 = x86::zmm31;

  for (int i = 0; i < rowRegs; ++i) {
    a->imul(C_Offset, ldcReg, static_cast<asmjit::Imm>(i * sizeof(int32_t)));
    for (int j = 0; j < colRegs; ++j) {
      for (int idx = 0; idx < 2; ++idx) {
        a->vextracti32x8(
            extractDest256, CRegs_avx512_[i * leadingDimCRegAssign + j], idx);
        a->vpmovsxwd(extractDest512, extractDest256);
        asmjit::X86Mem destAddr = x86::dword_ptr(
            a->zcx(), C_Offset, 0, (j * 2 + idx) * 16 * sizeof(int32_t));
        if (accum) {
          a->vpaddd(extractDest512, extractDest512, destAddr);
        }
        a->vmovups(destAddr, extractDest512);
      }
    }
  }
}

/**
 * Get or Create the AVX512 instructions for 16-bit Accumulation macro-kernel.
 *
 */
template <>
template <>
CodeGenBase<uint8_t, int8_t, int32_t, int16_t>::jit_micro_kernel_fp
CodeGenBase<uint8_t, int8_t, int32_t, int16_t>::getOrCreate<inst_set_t::avx512>(
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

#if defined(FBGEMM_LOG_CODE)
  // generated code logging
  FILE* codeLogfile =
      fopen(getCodeLoggingFile<inst_set_t::avx512>(accum, mc, nc).c_str(), "w");
  asmjit::FileLogger* codeLogger = new asmjit::FileLogger(codeLogfile);
  if (codeLogger) {
    code_.setLogger(codeLogger);
  }
#endif

  constexpr int kBlock =
      PackingTraits<int8_t, int16_t, inst_set_t::avx512>::KCB;
  constexpr int nBlock =
      PackingTraits<int8_t, int16_t, inst_set_t::avx512>::NCB;
  constexpr int mRegBlockSize =
      PackingTraits<int8_t, int16_t, inst_set_t::avx512>::MR;
  constexpr int nRegBlockSize =
      PackingTraits<int8_t, int16_t, inst_set_t::avx512>::NR;
  constexpr int row_interleave =
      PackingTraits<int8_t, int16_t, inst_set_t::avx512>::ROW_INTERLEAVE;

  assert(kc % row_interleave == 0 && "kc must be a multiple of row_interleave");
  assert(nc % nRegBlockSize == 0 && "nc must be a multiple of NR");
  int maxMRegs = mRegBlockSize;
  int maxNRegs = nRegBlockSize * row_interleave / VLEN_;
  assert(
      maxMRegs * maxNRegs <= 24 &&
      "MR*(NR*ROW_INTERLEAVE*8/512) \
      must be <= 24(available registers constraint)");

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
      asmjit::X86Reg::kKindGp,
      asmjit::Utils::mask(8, 9, 10, 11, 12, 13, 14, 15));

  asmjit::FuncArgsMapper args(&func);
  args.assignAll(buffer_A, buffer_B, B_pf, CBase, kSize, ldcReg);

  args.updateFrameInfo(ffi);

  asmjit::FuncFrameLayout layout;
  layout.init(func, ffi);

  asmjit::FuncUtils::emitProlog(a, layout);
  asmjit::FuncUtils::allocArgs(a, layout, args);

  asmjit::Label LoopMBlocks = a->newLabel();
  asmjit::Label LoopNBlocks = a->newLabel();
  asmjit::Label Loopk = a->newLabel();

  asmjit::X86Gp buffer_B_saved = a->gpzRef(10);
  asmjit::X86Gp C_Offset = a->gpzRef(11);
  // asmjit::X86Gp B_pf_saved = a->gpzRef(12);
  asmjit::X86Gp iIdx = a->gpzRef(13);
  asmjit::X86Gp jIdx = a->gpzRef(14);
  asmjit::X86Gp kIdx = a->gpzRef(15);

  // save B_buffer address
  a->mov(buffer_B_saved, buffer_B);
  // a->mov(B_pf_saved, B_pf);

  int currColRegs = nc * row_interleave * sizeof(int8_t) / VLEN_;
  int colRegs = std::min(currColRegs, maxNRegs);
  if (mRegBlocks > 0) {
    // move 0 to iteration variables
    a->mov(iIdx, 0);

    a->bind(LoopMBlocks);
    a->inc(iIdx);
    a->mov(jIdx, 0);

    a->bind(LoopNBlocks);
    a->inc(jIdx);

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
    // a->add(B_pf, static_cast<asmjit::Imm>(nBlock * row_interleave *
    // sizeof(int8_t)));

    a->cmp(kIdx, kSize);
    a->jl(Loopk);

    // store C matrix
    storeCRegs<inst_set_t::avx512>(
        a, rowRegs, colRegs, C_Offset, ldcReg, accum, colRegs);

    // reset A
    a->sub(buffer_A, kSize);

    // B for next block
    a->mov(buffer_B, buffer_B_saved);
    // using C_Offset as temp reg
    a->imul(
        C_Offset,
        jIdx,
        static_cast<asmjit::Imm>(
            nRegBlockSize * row_interleave * sizeof(int8_t)));
    a->add(buffer_B, C_Offset);

    // increment C for next block
    a->add(CBase, static_cast<asmjit::Imm>(nRegBlockSize * sizeof(int32_t)));

    int jLoopTrips = currColRegs / maxNRegs;
    a->cmp(jIdx, jLoopTrips);
    a->jl(LoopNBlocks);

    // increment A for next block
    a->add(
        buffer_A, static_cast<asmjit::Imm>((rowRegs)*kBlock * sizeof(uint8_t)));

    // increment C for next A block
    a->sub(
        CBase,
        static_cast<asmjit::Imm>(jLoopTrips * nRegBlockSize * sizeof(int32_t)));
    a->imul(
        C_Offset, ldcReg, static_cast<asmjit::Imm>(rowRegs * sizeof(int32_t)));
    a->add(CBase, C_Offset);

    // reset B
    a->mov(buffer_B, buffer_B_saved);
    // a->mov(B_pf, B_pf_saved);

    a->cmp(iIdx, mRegBlocks);
    a->jl(LoopMBlocks);
  }
  // generate code for remainder
  if (mRegBlocksRem > 0) {
    asmjit::Label LoopNRem = a->newLabel();
    asmjit::Label LoopkRem = a->newLabel();
    int rowRegs = mRegBlocksRem;

    a->mov(jIdx, 0);
    a->bind(LoopNRem);
    a->inc(jIdx);

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
    // a->add(B_pf, static_cast<asmjit::Imm>(nBlock * row_interleave *
    // sizeof(int8_t)));

    a->cmp(kIdx, kSize);
    a->jl(LoopkRem);

    // reset A
    a->sub(buffer_A, kSize);

    // B for next block
    a->mov(buffer_B, buffer_B_saved);
    // using C_Offset as temp reg
    a->imul(
        C_Offset,
        jIdx,
        static_cast<asmjit::Imm>(
            nRegBlockSize * row_interleave * sizeof(int8_t)));
    a->add(buffer_B, C_Offset);

    // store C matrix
    storeCRegs<inst_set_t::avx512>(
        a, rowRegs, colRegs, C_Offset, ldcReg, accum, colRegs);

    // increment C for next block
    a->add(CBase, static_cast<asmjit::Imm>(nRegBlockSize * sizeof(int32_t)));

    int jLoopTrips = currColRegs / maxNRegs;
    a->cmp(jIdx, jLoopTrips);
    a->jl(LoopNRem);
  }

  asmjit::FuncUtils::emitEpilog(a, layout);

  jit_micro_kernel_fp fn;
  asmjit::Error err = rt_.add(&fn, &code_);
  if (err) {
    std::cout << "Error: in fn add" << std::endl;
    return nullptr;
  }
  codeCache_[kernelSig] = fn;

#if defined(FBGEMM_LOG_CODE)
  fclose(codeLogfile);
  delete codeLogger;
#endif

  return fn;
}

} // namespace fbgemm
