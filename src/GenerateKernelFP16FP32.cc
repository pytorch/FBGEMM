/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "./GenerateKernelFP16FP32.h" // @manual

#include <asmjit/core.h> // @manual
#include <asmjit/x86.h> // @manual
#include <cstddef>
#include <mutex>
#include <type_traits>

#include "fbgemm/SimdUtils.h"
#include "fbgemm/Types.h"

namespace fbgemm {

namespace {

using namespace asmjit;

JitRuntime& runtime() {
  static JitRuntime rt;
  return rt;
}

static std::mutex rtMutex_;

// Create a vector register of the appropriate width for this ISA.
template <inst_set_t instSet>
x86::Vec makeVec(uint32_t id) {
  constexpr int VEC_BYTES = simd_info<instSet>::WIDTH_BYTES;
  if constexpr (VEC_BYTES == 64) {
    return x86::Vec::make_v512(id);
  } else {
    return x86::Vec::make_v256(id);
  }
}

// Emit a B-load instruction: FP16 uses vcvtph2ps, FP32 uses vmovups.
// offset is in bytes from basePtr.
template <typename T, inst_set_t instSet>
void emitBLoad(
    x86::Emitter* a,
    x86::Vec dest,
    const x86::Gp& basePtr,
    int offset) {
  constexpr int VEC_BYTES = simd_info<instSet>::WIDTH_BYTES;
  if constexpr (std::is_same_v<T, float16>) {
    if constexpr (VEC_BYTES == 32) {
      // vcvtph2ps ymm, xmmword [mem] -- load 128-bit FP16 -> 256-bit FP32
      a->vcvtph2ps(dest, x86::xmmword_ptr(basePtr, offset));
    } else {
      // vcvtph2ps zmm, ymmword [mem] -- load 256-bit FP16 -> 512-bit FP32
      a->vcvtph2ps(dest, x86::ymmword_ptr(basePtr, offset));
    }
  } else {
    if constexpr (VEC_BYTES == 32) {
      a->vmovups(dest, x86::ymmword_ptr(basePtr, offset));
    } else {
      a->vmovups(dest, x86::zmmword_ptr(basePtr, offset));
    }
  }
}

template <typename T, inst_set_t instSet>
funcptr_t<T> generateGemmKernelImpl(int kernel_nrows) {
  constexpr int NUM_REGS = simd_info<instSet>::NUM_VEC_REGS;
  constexpr int VEC_BYTES = simd_info<instSet>::WIDTH_BYTES;
  constexpr int VEC_ELEMS = simd_info<instSet>::WIDTH_32BIT_ELEMS;
  constexpr bool is_fp16 = std::is_same_v<T, float16>;
  // B stride in bytes per k-iteration: 2 vectors worth of T elements
  constexpr int B_STRIDE_BYTES = 2 * VEC_ELEMS * static_cast<int>(sizeof(T));
  // C stride in bytes to advance to next block column: 2 vectors of floats
  constexpr int C_STRIDE_BYTES = 2 * VEC_BYTES;
  // B half-vector size in bytes (for second vector load offset)
  constexpr int B_VEC_BYTES = is_fp16 ? (VEC_BYTES / 2) : VEC_BYTES;

  // Vector register assignment
  // Accumulators: vec[0 .. 2*nrows-1]
  const int aRegId = 2 * kernel_nrows;
  const int bReg0Id = 2 * kernel_nrows + 1;
  const int bReg1Id = 2 * kernel_nrows + 2;
  const int betaRegId = NUM_REGS - 1;

  x86::Vec aReg = makeVec<instSet>(aRegId);
  x86::Vec bReg0 = makeVec<instSet>(bReg0Id);
  x86::Vec bReg1 = makeVec<instSet>(bReg1Id);
  x86::Vec betaReg = makeVec<instSet>(betaRegId);

  CodeHolder code;
  code.init(runtime().environment());
  x86::Assembler assembler(&code);
  x86::Emitter* a = assembler.as<x86::Emitter>();

  // Function signature: void(GemmParams<T>*)
  FuncDetail func;
  func.init(
      FuncSignature::build<void, GemmParams<T>*>(), a->environment());

  FuncFrame frame;
  frame.init(func);

  // Mark all used vector regs as dirty
  uint32_t dirtyVecRegs =
      Support::bitMask(0, 1, 2, 3, 4, 5, 6, 7) |
      Support::bitMask(8, 9, 10, 11, 12, 13, 14, 15);
  if constexpr (NUM_REGS > 16) {
    dirtyVecRegs |=
        Support::bitMask(16, 17, 18, 19, 20, 21, 22, 23) |
        Support::bitMask(24, 25, 26, 27, 28, 29, 30, 31);
  }
  frame.setDirtyRegs(RegGroup::kVec, dirtyVecRegs);

  // GP registers we use (callee-saved ones need to be declared dirty)
  frame.setDirtyRegs(
      RegGroup::kGp,
      Support::bitMask(0, 1, 2, 3, 6, 7) | // rax, rcx, rdx, rbx, rsi, rdi
          Support::bitMask(8, 9, 10, 11, 12, 13, 14, 15));

  // Function argument: GemmParams<T>* in a GP register
  const x86::Gp gpReg = a->gpz(0); // rax temporarily, reassigned by asmjit
  FuncArgsAssignment args(&func);
  args.assignAll(gpReg);
  args.updateFuncFrame(frame);
  frame.finalize();

  a->emitProlog(frame);
  a->emitArgsAssignment(frame, args);

  // GP register allocation
  const x86::Gp& kReg = a->gpz(8); // r8: k
  const x86::Gp& APtr = a->gpz(9); // r9: A pointer
  const x86::Gp& BPtr = a->gpz(10); // r10: B pointer
  const x86::Gp& CPtr = a->gpz(12); // r12: C pointer
  const x86::Gp& ldcReg = a->gpz(13); // r13: ldc (in bytes)
  const x86::Gp& betaPtr = a->gpz(15); // r15: &beta
  const x86::Gp& bColsReg = a->gpz(7); // rdi: b_block_cols
  const x86::Gp& ASaved = a->gpz(1); // rcx: saved A base
  const x86::Gp& CSaved = a->gpz(2); // rdx: saved C base
  const x86::Gp& outerIdx = a->gpz(3); // rbx: outer loop counter
  const x86::Gp& innerIdx = a->gpz(14); // r14: inner loop counter

  // Load GemmParams fields from struct pointer (gpReg = rax)
  a->mov(kReg, x86::qword_ptr(gpReg, offsetof(GemmParams<T>, k)));
  a->mov(APtr, x86::qword_ptr(gpReg, offsetof(GemmParams<T>, A)));
  a->mov(BPtr, x86::qword_ptr(gpReg, offsetof(GemmParams<T>, B)));
  a->lea(betaPtr, x86::ptr(gpReg, offsetof(GemmParams<T>, beta)));
  a->mov(CPtr, x86::qword_ptr(gpReg, offsetof(GemmParams<T>, C)));
  a->mov(ldcReg, x86::qword_ptr(gpReg, offsetof(GemmParams<T>, ldc)));
  a->mov(
      bColsReg,
      x86::qword_ptr(gpReg, offsetof(GemmParams<T>, b_block_cols)));

  // Save base pointers
  a->mov(ASaved, APtr);
  a->mov(CSaved, CPtr);

  // ====== Outer loop (over b_block_cols) ======
  a->xor_(outerIdx.r32(), outerIdx.r32());

  Label loopOuter = a->newLabel();
  Label loopEnd = a->newLabel();

  a->bind(loopOuter);
  a->cmp(outerIdx, bColsReg);
  a->jge(loopEnd);

  // Reset A for each column block
  a->mov(APtr, ASaved);

  // innerIdx = k - 1 (peel first iteration)
  a->mov(innerIdx, kReg);
  a->dec(innerIdx);

  // Broadcast beta
  a->vbroadcastss(betaReg, x86::dword_ptr(betaPtr));

  // Load first B pair
  emitBLoad<T, instSet>(a, bReg0, BPtr, 0);
  emitBLoad<T, instSet>(a, bReg1, BPtr, B_VEC_BYTES);

  // Check if beta == 0
  x86::Vec xmmZero = x86::Vec::make_v128(aRegId);
  a->vxorps(xmmZero, xmmZero, xmmZero);
  a->vucomiss(x86::Vec::make_v128(betaRegId), xmmZero);

  Label zeroRegs = a->newLabel();
  Label innerLoopEntry = a->newLabel();
  Label innerLoop = a->newLabel();
  Label dumpC = a->newLabel();

  a->je(zeroRegs);

  // ------ First k iteration, beta != 0 ------
  // Load C rows and multiply by beta, then FMA with A*B
  {
    x86::Gp tempC = a->gpz(11); // r11 as temp for C row pointer
    a->mov(tempC, CPtr);
    for (int jj = 0; jj < kernel_nrows; jj++) {
      // C_row0 = beta * C[jj*ldc + 0..VEC_ELEMS-1]
      a->vmulps(makeVec<instSet>(2 * jj), betaReg, x86::ptr(tempC, 0));
      // C_row1 = beta * C[jj*ldc + VEC_ELEMS..2*VEC_ELEMS-1]
      a->vmulps(
          makeVec<instSet>(2 * jj + 1),
          betaReg,
          x86::ptr(tempC, VEC_BYTES));
      if (jj < kernel_nrows - 1) {
        a->add(tempC, ldcReg);
      }
    }
    // FMA: accum += A[jj] * B
    for (int jj = 0; jj < kernel_nrows; jj++) {
      a->vbroadcastss(
          aReg, x86::dword_ptr(APtr, jj * static_cast<int>(sizeof(float))));
      a->vfmadd231ps(makeVec<instSet>(2 * jj), aReg, bReg0);
      a->vfmadd231ps(makeVec<instSet>(2 * jj + 1), aReg, bReg1);
    }
    a->add(APtr, kernel_nrows * static_cast<int>(sizeof(float)));
    a->add(BPtr, B_STRIDE_BYTES);
  }

  // Check if k > 1
  a->test(innerIdx, innerIdx);
  a->jnz(innerLoopEntry);
  a->jmp(dumpC);

  // ------ First k iteration, beta == 0 ------
  a->bind(zeroRegs);
  {
    for (int jj = 0; jj < kernel_nrows; jj++) {
      a->vbroadcastss(
          aReg, x86::dword_ptr(APtr, jj * static_cast<int>(sizeof(float))));
      a->vmulps(makeVec<instSet>(2 * jj), aReg, bReg0);
      a->vmulps(makeVec<instSet>(2 * jj + 1), aReg, bReg1);
    }
    a->add(APtr, kernel_nrows * static_cast<int>(sizeof(float)));
    a->add(BPtr, B_STRIDE_BYTES);
  }

  a->test(innerIdx, innerIdx);
  a->jz(dumpC);

  // ------ Inner loop (k iterations 2..k) ------
  a->bind(innerLoopEntry);
  a->bind(innerLoop);
  {
    // Load next B pair
    emitBLoad<T, instSet>(a, bReg0, BPtr, 0);
    emitBLoad<T, instSet>(a, bReg1, BPtr, B_VEC_BYTES);

    // FMA: accum += A[jj] * B
    for (int jj = 0; jj < kernel_nrows; jj++) {
      a->vbroadcastss(
          aReg, x86::dword_ptr(APtr, jj * static_cast<int>(sizeof(float))));
      a->vfmadd231ps(makeVec<instSet>(2 * jj), aReg, bReg0);
      a->vfmadd231ps(makeVec<instSet>(2 * jj + 1), aReg, bReg1);
    }
    a->add(APtr, kernel_nrows * static_cast<int>(sizeof(float)));
    a->add(BPtr, B_STRIDE_BYTES);

    a->dec(innerIdx);
    a->jnz(innerLoop);
  }

  // ------ Store C ------
  a->bind(dumpC);
  {
    x86::Gp tempC = a->gpz(11); // r11
    a->mov(tempC, CPtr);
    for (int jj = 0; jj < kernel_nrows; jj++) {
      a->vmovups(x86::ptr(tempC, 0), makeVec<instSet>(2 * jj));
      a->vmovups(x86::ptr(tempC, VEC_BYTES), makeVec<instSet>(2 * jj + 1));
      if (jj < kernel_nrows - 1) {
        a->add(tempC, ldcReg);
      }
    }
  }

  // Advance C to next block column
  a->add(CPtr, C_STRIDE_BYTES);

  // Increment outer counter
  a->inc(outerIdx);
  a->jmp(loopOuter);

  // ====== End ======
  a->bind(loopEnd);
  a->emitEpilog(frame);

  funcptr_t<T> fn = nullptr;
  {
    std::lock_guard<std::mutex> lock(rtMutex_);
    Error err = runtime().add(&fn, &code);
    if (err) {
      return nullptr;
    }
  }
  return fn;
}

} // anonymous namespace

template <typename T, inst_set_t instSet>
funcptr_t<T> generateGemmKernel(int kernel_nrows) {
  return generateGemmKernelImpl<T, instSet>(kernel_nrows);
}

// Explicit template instantiations
// FP16
template funcptr_t<float16>
generateGemmKernel<float16, inst_set_t::avx2>(int);
template funcptr_t<float16>
generateGemmKernel<float16, inst_set_t::avx512>(int);
template funcptr_t<float16>
generateGemmKernel<float16, inst_set_t::avx512_ymm>(int);

// FP32
template funcptr_t<float> generateGemmKernel<float, inst_set_t::avx2>(int);
template funcptr_t<float> generateGemmKernel<float, inst_set_t::avx512>(int);
template funcptr_t<float>
generateGemmKernel<float, inst_set_t::avx512_ymm>(int);

} // namespace fbgemm
