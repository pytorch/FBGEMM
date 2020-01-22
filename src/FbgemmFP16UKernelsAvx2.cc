/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "./FbgemmFP16UKernelsAvx2.h"
#ifdef _MSC_VER
#include <immintrin.h>
#endif

namespace fbgemm {

#ifndef _MSC_VER
void NOINLINE
gemmkernel_1x2_Avx2_fp16_fA0fB0fC0(GemmParamsFP16* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // k
      "mov r8, [r14 + 0]\t\n"
      "dec r8\t\n"
      // A
      "mov r9, [r14 + 8]\t\n"
      // B
      "mov r10, [r14 + 16]\t\n"
      // beta
      "lea r15, [r14 + 24]\t\n"
      // C
      "mov r12, [r14 + 32]\t\n"
      // ldc
      "mov r13, [r14 + 40]\t\n"
      // b_block_cols
      "mov rdi, [r14 + 48]\t\n"
      // b_block_size
      "mov rsi, [r14 + 56]\t\n"

      // Make copies of A and C
      "mov rax, r9\t\n"
      "mov rcx, r12\t\n"

      "mov rbx, 0\t\n"
      "loop_outter%=:\t\n"
      "mov r14, r8\t\n"
      "vbroadcastss ymm15,DWORD PTR [r15]\t\n"
      "vcvtph2ps ymm3,XMMWORD PTR [r10 + 0]\t\n"
      "vcvtph2ps ymm4,XMMWORD PTR [r10 + 16]\t\n"
      "vxorps xmm0, xmm0, xmm0\t\n"
      "vcomiss xmm15, xmm0\t\n"
      "jz zero_regs%=\t\n"

      // Setup values with beta multiplication
      "vmulps ymm0, ymm15, [r12 + 0]\t\n"
      "vmulps ymm1, ymm15, [r12 + 32]\t\n"
      "test r14,r14\t\n"
      "jz skip_preload%=\t\n"
      "vcvtph2ps ymm15,XMMWORD PTR [r10 + 32]\t\n"
      "skip_preload%=:\t\n"
      "vbroadcastss ymm2,DWORD PTR [r9+0]\t\n"
      "vfmadd231ps ymm0,ymm3,ymm2\t\n"
      "vfmadd231ps ymm1,ymm4,ymm2\t\n"
      "test r14,r14\t\n"
      "jnz next_inner%=\t\n"
      "add r10,32\t\n"
      "jmp dump_C%=\t\n"

      "zero_regs%=:\t\n"

      "test r14,r14\t\n"
      "jz skip_preload_b_zero%=\t\n"
      "vcvtph2ps ymm15,XMMWORD PTR [r10 + 32]\t\n"
      "skip_preload_b_zero%=:\t\n"
      "vbroadcastss ymm2,DWORD PTR [r9+0]\t\n"
      "vmulps ymm0,ymm3,ymm2\t\n"
      "vmulps ymm1,ymm4,ymm2\t\n"
      "test r14,r14\t\n"
      "jnz next_inner%=\t\n"
      "add r10,32\t\n"
      "jmp dump_C%=\t\n"

      "loop_inner%=:\t\n"

      "vmovaps ymm3,ymm15\t\n"
      "vcvtph2ps ymm4,XMMWORD PTR [r10 + 16]\t\n"
      "vcvtph2ps ymm15,XMMWORD PTR [r10 + 32]\t\n"
      "vbroadcastss ymm2,DWORD PTR [r9+0]\t\n"
      "vfmadd231ps ymm0,ymm3,ymm2\t\n"
      "vfmadd231ps ymm1,ymm4,ymm2\t\n"

      "next_inner%=:\t\n"
      "add r9,4\t\n"
      "add r10,32\t\n"
      "dec r14\t\n"
      "jnz loop_inner%=\t\n"

      "vmovaps ymm3,ymm15\t\n"
      "vcvtph2ps ymm4,XMMWORD PTR [r10 + 16]\t\n"
      "vbroadcastss ymm2,DWORD PTR [r9+0]\t\n"
      "vfmadd231ps ymm0,ymm3,ymm2\t\n"
      "vfmadd231ps ymm1,ymm4,ymm2\t\n"
      "add r9,4\t\n"
      "add r10,32\t\n"
      // Dump C
      "dump_C%=:\t\n"
      "vmovups ymmword PTR [r12 + 0], ymm0\t\n"
      "vmovups ymmword PTR [r12 + 32], ymm1\t\n"

      // next outer iteration
      "add rcx, 64\t\n"
      "mov r12, rcx\t\n"
      "mov r9, rax\t\n"
      "inc rbx\t\n"
      "cmp rbx, rdi\t\n"
      "jl loop_outter%=\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r13",
        "r14",
        "rax",
        "rcx",
        "rsi",
        "rdi",
        "rbx",
        "r12",
        "r15",
        "memory");
}
void NOINLINE
gemmkernel_2x2_Avx2_fp16_fA0fB0fC0(GemmParamsFP16* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // k
      "mov r8, [r14 + 0]\t\n"
      "dec r8\t\n"
      // A
      "mov r9, [r14 + 8]\t\n"
      // B
      "mov r10, [r14 + 16]\t\n"
      // beta
      "lea r15, [r14 + 24]\t\n"
      // C
      "mov r12, [r14 + 32]\t\n"
      // ldc
      "mov r13, [r14 + 40]\t\n"
      // b_block_cols
      "mov rdi, [r14 + 48]\t\n"
      // b_block_size
      "mov rsi, [r14 + 56]\t\n"

      // Make copies of A and C
      "mov rax, r9\t\n"
      "mov rcx, r12\t\n"

      "mov rbx, 0\t\n"
      "loop_outter%=:\t\n"
      "mov r14, r8\t\n"
      "vbroadcastss ymm15,DWORD PTR [r15]\t\n"
      "vcvtph2ps ymm5,XMMWORD PTR [r10 + 0]\t\n"
      "vcvtph2ps ymm6,XMMWORD PTR [r10 + 16]\t\n"
      "vxorps xmm0, xmm0, xmm0\t\n"
      "vcomiss xmm15, xmm0\t\n"
      "jz zero_regs%=\t\n"

      // Setup values with beta multiplication
      "vmulps ymm0, ymm15, [r12 + 0]\t\n"
      "vmulps ymm1, ymm15, [r12 + 32]\t\n"
      "add r12, r13\t\n"
      "vmulps ymm2, ymm15, [r12 + 0]\t\n"
      "vmulps ymm3, ymm15, [r12 + 32]\t\n"
      "test r14,r14\t\n"
      "jz skip_preload%=\t\n"
      "vcvtph2ps ymm15,XMMWORD PTR [r10 + 32]\t\n"
      "skip_preload%=:\t\n"
      "vbroadcastss ymm4,DWORD PTR [r9+0]\t\n"
      "vfmadd231ps ymm0,ymm5,ymm4\t\n"
      "vfmadd231ps ymm1,ymm6,ymm4\t\n"
      "vbroadcastss ymm4,DWORD PTR [r9+4]\t\n"
      "vfmadd231ps ymm2,ymm5,ymm4\t\n"
      "vfmadd231ps ymm3,ymm6,ymm4\t\n"
      "mov r12, rcx\t\n"
      "test r14,r14\t\n"
      "jnz next_inner%=\t\n"
      "add r10,32\t\n"
      "jmp dump_C%=\t\n"

      "zero_regs%=:\t\n"

      "test r14,r14\t\n"
      "jz skip_preload_b_zero%=\t\n"
      "vcvtph2ps ymm15,XMMWORD PTR [r10 + 32]\t\n"
      "skip_preload_b_zero%=:\t\n"
      "vbroadcastss ymm4,DWORD PTR [r9+0]\t\n"
      "vmulps ymm0,ymm5,ymm4\t\n"
      "vmulps ymm1,ymm6,ymm4\t\n"
      "add r12, r13\t\n"
      "vbroadcastss ymm4,DWORD PTR [r9+4]\t\n"
      "vmulps ymm2,ymm5,ymm4\t\n"
      "vmulps ymm3,ymm6,ymm4\t\n"
      "mov r12, rcx\t\n"
      "test r14,r14\t\n"
      "jnz next_inner%=\t\n"
      "add r10,32\t\n"
      "jmp dump_C%=\t\n"

      "loop_inner%=:\t\n"

      "vmovaps ymm5,ymm15\t\n"
      "vcvtph2ps ymm6,XMMWORD PTR [r10 + 16]\t\n"
      "vcvtph2ps ymm15,XMMWORD PTR [r10 + 32]\t\n"
      "vbroadcastss ymm4,DWORD PTR [r9+0]\t\n"
      "vfmadd231ps ymm0,ymm5,ymm4\t\n"
      "vfmadd231ps ymm1,ymm6,ymm4\t\n"
      "vbroadcastss ymm4,DWORD PTR [r9+4]\t\n"
      "vfmadd231ps ymm2,ymm5,ymm4\t\n"
      "vfmadd231ps ymm3,ymm6,ymm4\t\n"

      "next_inner%=:\t\n"
      "add r9,8\t\n"
      "add r10,32\t\n"
      "dec r14\t\n"
      "jnz loop_inner%=\t\n"

      "vmovaps ymm5,ymm15\t\n"
      "vcvtph2ps ymm6,XMMWORD PTR [r10 + 16]\t\n"
      "vbroadcastss ymm4,DWORD PTR [r9+0]\t\n"
      "vfmadd231ps ymm0,ymm5,ymm4\t\n"
      "vfmadd231ps ymm1,ymm6,ymm4\t\n"
      "vbroadcastss ymm4,DWORD PTR [r9+4]\t\n"
      "vfmadd231ps ymm2,ymm5,ymm4\t\n"
      "vfmadd231ps ymm3,ymm6,ymm4\t\n"
      "add r9,8\t\n"
      "add r10,32\t\n"
      // Dump C
      "dump_C%=:\t\n"
      "vmovups ymmword PTR [r12 + 0], ymm0\t\n"
      "vmovups ymmword PTR [r12 + 32], ymm1\t\n"
      "add r12, r13\t\n"
      "vmovups ymmword PTR [r12 + 0], ymm2\t\n"
      "vmovups ymmword PTR [r12 + 32], ymm3\t\n"

      // next outer iteration
      "add rcx, 64\t\n"
      "mov r12, rcx\t\n"
      "mov r9, rax\t\n"
      "inc rbx\t\n"
      "cmp rbx, rdi\t\n"
      "jl loop_outter%=\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r13",
        "r14",
        "rax",
        "rcx",
        "rsi",
        "rdi",
        "rbx",
        "r12",
        "r15",
        "memory");
}
void NOINLINE
gemmkernel_3x2_Avx2_fp16_fA0fB0fC0(GemmParamsFP16* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // k
      "mov r8, [r14 + 0]\t\n"
      "dec r8\t\n"
      // A
      "mov r9, [r14 + 8]\t\n"
      // B
      "mov r10, [r14 + 16]\t\n"
      // beta
      "lea r15, [r14 + 24]\t\n"
      // C
      "mov r12, [r14 + 32]\t\n"
      // ldc
      "mov r13, [r14 + 40]\t\n"
      // b_block_cols
      "mov rdi, [r14 + 48]\t\n"
      // b_block_size
      "mov rsi, [r14 + 56]\t\n"

      // Make copies of A and C
      "mov rax, r9\t\n"
      "mov rcx, r12\t\n"

      "mov rbx, 0\t\n"
      "loop_outter%=:\t\n"
      "mov r14, r8\t\n"
      "vbroadcastss ymm15,DWORD PTR [r15]\t\n"
      "vcvtph2ps ymm7,XMMWORD PTR [r10 + 0]\t\n"
      "vcvtph2ps ymm8,XMMWORD PTR [r10 + 16]\t\n"
      "vxorps xmm0, xmm0, xmm0\t\n"
      "vcomiss xmm15, xmm0\t\n"
      "jz zero_regs%=\t\n"

      // Setup values with beta multiplication
      "vmulps ymm0, ymm15, [r12 + 0]\t\n"
      "vmulps ymm1, ymm15, [r12 + 32]\t\n"
      "add r12, r13\t\n"
      "vmulps ymm2, ymm15, [r12 + 0]\t\n"
      "vmulps ymm3, ymm15, [r12 + 32]\t\n"
      "add r12, r13\t\n"
      "vmulps ymm4, ymm15, [r12 + 0]\t\n"
      "vmulps ymm5, ymm15, [r12 + 32]\t\n"
      "test r14,r14\t\n"
      "jz skip_preload%=\t\n"
      "vcvtph2ps ymm15,XMMWORD PTR [r10 + 32]\t\n"
      "skip_preload%=:\t\n"
      "vbroadcastss ymm6,DWORD PTR [r9+0]\t\n"
      "vfmadd231ps ymm0,ymm7,ymm6\t\n"
      "vfmadd231ps ymm1,ymm8,ymm6\t\n"
      "vbroadcastss ymm6,DWORD PTR [r9+4]\t\n"
      "vfmadd231ps ymm2,ymm7,ymm6\t\n"
      "vfmadd231ps ymm3,ymm8,ymm6\t\n"
      "vbroadcastss ymm6,DWORD PTR [r9+8]\t\n"
      "vfmadd231ps ymm4,ymm7,ymm6\t\n"
      "vfmadd231ps ymm5,ymm8,ymm6\t\n"
      "mov r12, rcx\t\n"
      "test r14,r14\t\n"
      "jnz next_inner%=\t\n"
      "add r10,32\t\n"
      "jmp dump_C%=\t\n"

      "zero_regs%=:\t\n"

      "test r14,r14\t\n"
      "jz skip_preload_b_zero%=\t\n"
      "vcvtph2ps ymm15,XMMWORD PTR [r10 + 32]\t\n"
      "skip_preload_b_zero%=:\t\n"
      "vbroadcastss ymm6,DWORD PTR [r9+0]\t\n"
      "vmulps ymm0,ymm7,ymm6\t\n"
      "vmulps ymm1,ymm8,ymm6\t\n"
      "add r12, r13\t\n"
      "vbroadcastss ymm6,DWORD PTR [r9+4]\t\n"
      "vmulps ymm2,ymm7,ymm6\t\n"
      "vmulps ymm3,ymm8,ymm6\t\n"
      "add r12, r13\t\n"
      "vbroadcastss ymm6,DWORD PTR [r9+8]\t\n"
      "vmulps ymm4,ymm7,ymm6\t\n"
      "vmulps ymm5,ymm8,ymm6\t\n"
      "mov r12, rcx\t\n"
      "test r14,r14\t\n"
      "jnz next_inner%=\t\n"
      "add r10,32\t\n"
      "jmp dump_C%=\t\n"

      "loop_inner%=:\t\n"

      "vmovaps ymm7,ymm15\t\n"
      "vcvtph2ps ymm8,XMMWORD PTR [r10 + 16]\t\n"
      "vcvtph2ps ymm15,XMMWORD PTR [r10 + 32]\t\n"
      "vbroadcastss ymm6,DWORD PTR [r9+0]\t\n"
      "vfmadd231ps ymm0,ymm7,ymm6\t\n"
      "vfmadd231ps ymm1,ymm8,ymm6\t\n"
      "vbroadcastss ymm6,DWORD PTR [r9+4]\t\n"
      "vfmadd231ps ymm2,ymm7,ymm6\t\n"
      "vfmadd231ps ymm3,ymm8,ymm6\t\n"
      "vbroadcastss ymm6,DWORD PTR [r9+8]\t\n"
      "vfmadd231ps ymm4,ymm7,ymm6\t\n"
      "vfmadd231ps ymm5,ymm8,ymm6\t\n"

      "next_inner%=:\t\n"
      "add r9,12\t\n"
      "add r10,32\t\n"
      "dec r14\t\n"
      "jnz loop_inner%=\t\n"

      "vmovaps ymm7,ymm15\t\n"
      "vcvtph2ps ymm8,XMMWORD PTR [r10 + 16]\t\n"
      "vbroadcastss ymm6,DWORD PTR [r9+0]\t\n"
      "vfmadd231ps ymm0,ymm7,ymm6\t\n"
      "vfmadd231ps ymm1,ymm8,ymm6\t\n"
      "vbroadcastss ymm6,DWORD PTR [r9+4]\t\n"
      "vfmadd231ps ymm2,ymm7,ymm6\t\n"
      "vfmadd231ps ymm3,ymm8,ymm6\t\n"
      "vbroadcastss ymm6,DWORD PTR [r9+8]\t\n"
      "vfmadd231ps ymm4,ymm7,ymm6\t\n"
      "vfmadd231ps ymm5,ymm8,ymm6\t\n"
      "add r9,12\t\n"
      "add r10,32\t\n"
      // Dump C
      "dump_C%=:\t\n"
      "vmovups ymmword PTR [r12 + 0], ymm0\t\n"
      "vmovups ymmword PTR [r12 + 32], ymm1\t\n"
      "add r12, r13\t\n"
      "vmovups ymmword PTR [r12 + 0], ymm2\t\n"
      "vmovups ymmword PTR [r12 + 32], ymm3\t\n"
      "add r12, r13\t\n"
      "vmovups ymmword PTR [r12 + 0], ymm4\t\n"
      "vmovups ymmword PTR [r12 + 32], ymm5\t\n"

      // next outer iteration
      "add rcx, 64\t\n"
      "mov r12, rcx\t\n"
      "mov r9, rax\t\n"
      "inc rbx\t\n"
      "cmp rbx, rdi\t\n"
      "jl loop_outter%=\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r13",
        "r14",
        "rax",
        "rcx",
        "rsi",
        "rdi",
        "rbx",
        "r12",
        "r15",
        "memory");
}
void NOINLINE
gemmkernel_4x2_Avx2_fp16_fA0fB0fC0(GemmParamsFP16* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // k
      "mov r8, [r14 + 0]\t\n"
      "dec r8\t\n"
      // A
      "mov r9, [r14 + 8]\t\n"
      // B
      "mov r10, [r14 + 16]\t\n"
      // beta
      "lea r15, [r14 + 24]\t\n"
      // C
      "mov r12, [r14 + 32]\t\n"
      // ldc
      "mov r13, [r14 + 40]\t\n"
      // b_block_cols
      "mov rdi, [r14 + 48]\t\n"
      // b_block_size
      "mov rsi, [r14 + 56]\t\n"

      // Make copies of A and C
      "mov rax, r9\t\n"
      "mov rcx, r12\t\n"

      "mov rbx, 0\t\n"
      "loop_outter%=:\t\n"
      "mov r14, r8\t\n"
      "vbroadcastss ymm15,DWORD PTR [r15]\t\n"
      "vcvtph2ps ymm9,XMMWORD PTR [r10 + 0]\t\n"
      "vcvtph2ps ymm10,XMMWORD PTR [r10 + 16]\t\n"
      "vxorps xmm0, xmm0, xmm0\t\n"
      "vcomiss xmm15, xmm0\t\n"
      "jz zero_regs%=\t\n"

      // Setup values with beta multiplication
      "vmulps ymm0, ymm15, [r12 + 0]\t\n"
      "vmulps ymm1, ymm15, [r12 + 32]\t\n"
      "add r12, r13\t\n"
      "vmulps ymm2, ymm15, [r12 + 0]\t\n"
      "vmulps ymm3, ymm15, [r12 + 32]\t\n"
      "add r12, r13\t\n"
      "vmulps ymm4, ymm15, [r12 + 0]\t\n"
      "vmulps ymm5, ymm15, [r12 + 32]\t\n"
      "add r12, r13\t\n"
      "vmulps ymm6, ymm15, [r12 + 0]\t\n"
      "vmulps ymm7, ymm15, [r12 + 32]\t\n"
      "test r14,r14\t\n"
      "jz skip_preload%=\t\n"
      "vcvtph2ps ymm15,XMMWORD PTR [r10 + 32]\t\n"
      "skip_preload%=:\t\n"
      "vbroadcastss ymm8,DWORD PTR [r9+0]\t\n"
      "vfmadd231ps ymm0,ymm9,ymm8\t\n"
      "vfmadd231ps ymm1,ymm10,ymm8\t\n"
      "vbroadcastss ymm8,DWORD PTR [r9+4]\t\n"
      "vfmadd231ps ymm2,ymm9,ymm8\t\n"
      "vfmadd231ps ymm3,ymm10,ymm8\t\n"
      "vbroadcastss ymm8,DWORD PTR [r9+8]\t\n"
      "vfmadd231ps ymm4,ymm9,ymm8\t\n"
      "vfmadd231ps ymm5,ymm10,ymm8\t\n"
      "vbroadcastss ymm8,DWORD PTR [r9+12]\t\n"
      "vfmadd231ps ymm6,ymm9,ymm8\t\n"
      "vfmadd231ps ymm7,ymm10,ymm8\t\n"
      "mov r12, rcx\t\n"
      "test r14,r14\t\n"
      "jnz next_inner%=\t\n"
      "add r10,32\t\n"
      "jmp dump_C%=\t\n"

      "zero_regs%=:\t\n"

      "test r14,r14\t\n"
      "jz skip_preload_b_zero%=\t\n"
      "vcvtph2ps ymm15,XMMWORD PTR [r10 + 32]\t\n"
      "skip_preload_b_zero%=:\t\n"
      "vbroadcastss ymm8,DWORD PTR [r9+0]\t\n"
      "vmulps ymm0,ymm9,ymm8\t\n"
      "vmulps ymm1,ymm10,ymm8\t\n"
      "add r12, r13\t\n"
      "vbroadcastss ymm8,DWORD PTR [r9+4]\t\n"
      "vmulps ymm2,ymm9,ymm8\t\n"
      "vmulps ymm3,ymm10,ymm8\t\n"
      "add r12, r13\t\n"
      "vbroadcastss ymm8,DWORD PTR [r9+8]\t\n"
      "vmulps ymm4,ymm9,ymm8\t\n"
      "vmulps ymm5,ymm10,ymm8\t\n"
      "add r12, r13\t\n"
      "vbroadcastss ymm8,DWORD PTR [r9+12]\t\n"
      "vmulps ymm6,ymm9,ymm8\t\n"
      "vmulps ymm7,ymm10,ymm8\t\n"
      "mov r12, rcx\t\n"
      "test r14,r14\t\n"
      "jnz next_inner%=\t\n"
      "add r10,32\t\n"
      "jmp dump_C%=\t\n"

      "loop_inner%=:\t\n"

      "vmovaps ymm9,ymm15\t\n"
      "vcvtph2ps ymm10,XMMWORD PTR [r10 + 16]\t\n"
      "vcvtph2ps ymm15,XMMWORD PTR [r10 + 32]\t\n"
      "vbroadcastss ymm8,DWORD PTR [r9+0]\t\n"
      "vfmadd231ps ymm0,ymm9,ymm8\t\n"
      "vfmadd231ps ymm1,ymm10,ymm8\t\n"
      "vbroadcastss ymm8,DWORD PTR [r9+4]\t\n"
      "vfmadd231ps ymm2,ymm9,ymm8\t\n"
      "vfmadd231ps ymm3,ymm10,ymm8\t\n"
      "vbroadcastss ymm8,DWORD PTR [r9+8]\t\n"
      "vfmadd231ps ymm4,ymm9,ymm8\t\n"
      "vfmadd231ps ymm5,ymm10,ymm8\t\n"
      "vbroadcastss ymm8,DWORD PTR [r9+12]\t\n"
      "vfmadd231ps ymm6,ymm9,ymm8\t\n"
      "vfmadd231ps ymm7,ymm10,ymm8\t\n"

      "next_inner%=:\t\n"
      "add r9,16\t\n"
      "add r10,32\t\n"
      "dec r14\t\n"
      "jnz loop_inner%=\t\n"

      "vmovaps ymm9,ymm15\t\n"
      "vcvtph2ps ymm10,XMMWORD PTR [r10 + 16]\t\n"
      "vbroadcastss ymm8,DWORD PTR [r9+0]\t\n"
      "vfmadd231ps ymm0,ymm9,ymm8\t\n"
      "vfmadd231ps ymm1,ymm10,ymm8\t\n"
      "vbroadcastss ymm8,DWORD PTR [r9+4]\t\n"
      "vfmadd231ps ymm2,ymm9,ymm8\t\n"
      "vfmadd231ps ymm3,ymm10,ymm8\t\n"
      "vbroadcastss ymm8,DWORD PTR [r9+8]\t\n"
      "vfmadd231ps ymm4,ymm9,ymm8\t\n"
      "vfmadd231ps ymm5,ymm10,ymm8\t\n"
      "vbroadcastss ymm8,DWORD PTR [r9+12]\t\n"
      "vfmadd231ps ymm6,ymm9,ymm8\t\n"
      "vfmadd231ps ymm7,ymm10,ymm8\t\n"
      "add r9,16\t\n"
      "add r10,32\t\n"
      // Dump C
      "dump_C%=:\t\n"
      "vmovups ymmword PTR [r12 + 0], ymm0\t\n"
      "vmovups ymmword PTR [r12 + 32], ymm1\t\n"
      "add r12, r13\t\n"
      "vmovups ymmword PTR [r12 + 0], ymm2\t\n"
      "vmovups ymmword PTR [r12 + 32], ymm3\t\n"
      "add r12, r13\t\n"
      "vmovups ymmword PTR [r12 + 0], ymm4\t\n"
      "vmovups ymmword PTR [r12 + 32], ymm5\t\n"
      "add r12, r13\t\n"
      "vmovups ymmword PTR [r12 + 0], ymm6\t\n"
      "vmovups ymmword PTR [r12 + 32], ymm7\t\n"

      // next outer iteration
      "add rcx, 64\t\n"
      "mov r12, rcx\t\n"
      "mov r9, rax\t\n"
      "inc rbx\t\n"
      "cmp rbx, rdi\t\n"
      "jl loop_outter%=\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r13",
        "r14",
        "rax",
        "rcx",
        "rsi",
        "rdi",
        "rbx",
        "r12",
        "r15",
        "memory");
}
void NOINLINE
gemmkernel_5x2_Avx2_fp16_fA0fB0fC0(GemmParamsFP16* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // k
      "mov r8, [r14 + 0]\t\n"
      "dec r8\t\n"
      // A
      "mov r9, [r14 + 8]\t\n"
      // B
      "mov r10, [r14 + 16]\t\n"
      // beta
      "lea r15, [r14 + 24]\t\n"
      // C
      "mov r12, [r14 + 32]\t\n"
      // ldc
      "mov r13, [r14 + 40]\t\n"
      // b_block_cols
      "mov rdi, [r14 + 48]\t\n"
      // b_block_size
      "mov rsi, [r14 + 56]\t\n"

      // Make copies of A and C
      "mov rax, r9\t\n"
      "mov rcx, r12\t\n"

      "mov rbx, 0\t\n"
      "loop_outter%=:\t\n"
      "mov r14, r8\t\n"
      "vbroadcastss ymm15,DWORD PTR [r15]\t\n"
      "vcvtph2ps ymm11,XMMWORD PTR [r10 + 0]\t\n"
      "vcvtph2ps ymm12,XMMWORD PTR [r10 + 16]\t\n"
      "vxorps xmm0, xmm0, xmm0\t\n"
      "vcomiss xmm15, xmm0\t\n"
      "jz zero_regs%=\t\n"

      // Setup values with beta multiplication
      "vmulps ymm0, ymm15, [r12 + 0]\t\n"
      "vmulps ymm1, ymm15, [r12 + 32]\t\n"
      "add r12, r13\t\n"
      "vmulps ymm2, ymm15, [r12 + 0]\t\n"
      "vmulps ymm3, ymm15, [r12 + 32]\t\n"
      "add r12, r13\t\n"
      "vmulps ymm4, ymm15, [r12 + 0]\t\n"
      "vmulps ymm5, ymm15, [r12 + 32]\t\n"
      "add r12, r13\t\n"
      "vmulps ymm6, ymm15, [r12 + 0]\t\n"
      "vmulps ymm7, ymm15, [r12 + 32]\t\n"
      "add r12, r13\t\n"
      "vmulps ymm8, ymm15, [r12 + 0]\t\n"
      "vmulps ymm9, ymm15, [r12 + 32]\t\n"
      "test r14,r14\t\n"
      "jz skip_preload%=\t\n"
      "vcvtph2ps ymm15,XMMWORD PTR [r10 + 32]\t\n"
      "skip_preload%=:\t\n"
      "vbroadcastss ymm10,DWORD PTR [r9+0]\t\n"
      "vfmadd231ps ymm0,ymm11,ymm10\t\n"
      "vfmadd231ps ymm1,ymm12,ymm10\t\n"
      "vbroadcastss ymm10,DWORD PTR [r9+4]\t\n"
      "vfmadd231ps ymm2,ymm11,ymm10\t\n"
      "vfmadd231ps ymm3,ymm12,ymm10\t\n"
      "vbroadcastss ymm10,DWORD PTR [r9+8]\t\n"
      "vfmadd231ps ymm4,ymm11,ymm10\t\n"
      "vfmadd231ps ymm5,ymm12,ymm10\t\n"
      "vbroadcastss ymm10,DWORD PTR [r9+12]\t\n"
      "vfmadd231ps ymm6,ymm11,ymm10\t\n"
      "vfmadd231ps ymm7,ymm12,ymm10\t\n"
      "vbroadcastss ymm10,DWORD PTR [r9+16]\t\n"
      "vfmadd231ps ymm8,ymm11,ymm10\t\n"
      "vfmadd231ps ymm9,ymm12,ymm10\t\n"
      "mov r12, rcx\t\n"
      "test r14,r14\t\n"
      "jnz next_inner%=\t\n"
      "add r10,32\t\n"
      "jmp dump_C%=\t\n"

      "zero_regs%=:\t\n"

      "test r14,r14\t\n"
      "jz skip_preload_b_zero%=\t\n"
      "vcvtph2ps ymm15,XMMWORD PTR [r10 + 32]\t\n"
      "skip_preload_b_zero%=:\t\n"
      "vbroadcastss ymm10,DWORD PTR [r9+0]\t\n"
      "vmulps ymm0,ymm11,ymm10\t\n"
      "vmulps ymm1,ymm12,ymm10\t\n"
      "add r12, r13\t\n"
      "vbroadcastss ymm10,DWORD PTR [r9+4]\t\n"
      "vmulps ymm2,ymm11,ymm10\t\n"
      "vmulps ymm3,ymm12,ymm10\t\n"
      "add r12, r13\t\n"
      "vbroadcastss ymm10,DWORD PTR [r9+8]\t\n"
      "vmulps ymm4,ymm11,ymm10\t\n"
      "vmulps ymm5,ymm12,ymm10\t\n"
      "add r12, r13\t\n"
      "vbroadcastss ymm10,DWORD PTR [r9+12]\t\n"
      "vmulps ymm6,ymm11,ymm10\t\n"
      "vmulps ymm7,ymm12,ymm10\t\n"
      "add r12, r13\t\n"
      "vbroadcastss ymm10,DWORD PTR [r9+16]\t\n"
      "vmulps ymm8,ymm11,ymm10\t\n"
      "vmulps ymm9,ymm12,ymm10\t\n"
      "mov r12, rcx\t\n"
      "test r14,r14\t\n"
      "jnz next_inner%=\t\n"
      "add r10,32\t\n"
      "jmp dump_C%=\t\n"

      "loop_inner%=:\t\n"

      "vmovaps ymm11,ymm15\t\n"
      "vcvtph2ps ymm12,XMMWORD PTR [r10 + 16]\t\n"
      "vcvtph2ps ymm15,XMMWORD PTR [r10 + 32]\t\n"
      "vbroadcastss ymm10,DWORD PTR [r9+0]\t\n"
      "vfmadd231ps ymm0,ymm11,ymm10\t\n"
      "vfmadd231ps ymm1,ymm12,ymm10\t\n"
      "vbroadcastss ymm10,DWORD PTR [r9+4]\t\n"
      "vfmadd231ps ymm2,ymm11,ymm10\t\n"
      "vfmadd231ps ymm3,ymm12,ymm10\t\n"
      "vbroadcastss ymm10,DWORD PTR [r9+8]\t\n"
      "vfmadd231ps ymm4,ymm11,ymm10\t\n"
      "vfmadd231ps ymm5,ymm12,ymm10\t\n"
      "vbroadcastss ymm10,DWORD PTR [r9+12]\t\n"
      "vfmadd231ps ymm6,ymm11,ymm10\t\n"
      "vfmadd231ps ymm7,ymm12,ymm10\t\n"
      "vbroadcastss ymm10,DWORD PTR [r9+16]\t\n"
      "vfmadd231ps ymm8,ymm11,ymm10\t\n"
      "vfmadd231ps ymm9,ymm12,ymm10\t\n"

      "next_inner%=:\t\n"
      "add r9,20\t\n"
      "add r10,32\t\n"
      "dec r14\t\n"
      "jnz loop_inner%=\t\n"

      "vmovaps ymm11,ymm15\t\n"
      "vcvtph2ps ymm12,XMMWORD PTR [r10 + 16]\t\n"
      "vbroadcastss ymm10,DWORD PTR [r9+0]\t\n"
      "vfmadd231ps ymm0,ymm11,ymm10\t\n"
      "vfmadd231ps ymm1,ymm12,ymm10\t\n"
      "vbroadcastss ymm10,DWORD PTR [r9+4]\t\n"
      "vfmadd231ps ymm2,ymm11,ymm10\t\n"
      "vfmadd231ps ymm3,ymm12,ymm10\t\n"
      "vbroadcastss ymm10,DWORD PTR [r9+8]\t\n"
      "vfmadd231ps ymm4,ymm11,ymm10\t\n"
      "vfmadd231ps ymm5,ymm12,ymm10\t\n"
      "vbroadcastss ymm10,DWORD PTR [r9+12]\t\n"
      "vfmadd231ps ymm6,ymm11,ymm10\t\n"
      "vfmadd231ps ymm7,ymm12,ymm10\t\n"
      "vbroadcastss ymm10,DWORD PTR [r9+16]\t\n"
      "vfmadd231ps ymm8,ymm11,ymm10\t\n"
      "vfmadd231ps ymm9,ymm12,ymm10\t\n"
      "add r9,20\t\n"
      "add r10,32\t\n"
      // Dump C
      "dump_C%=:\t\n"
      "vmovups ymmword PTR [r12 + 0], ymm0\t\n"
      "vmovups ymmword PTR [r12 + 32], ymm1\t\n"
      "add r12, r13\t\n"
      "vmovups ymmword PTR [r12 + 0], ymm2\t\n"
      "vmovups ymmword PTR [r12 + 32], ymm3\t\n"
      "add r12, r13\t\n"
      "vmovups ymmword PTR [r12 + 0], ymm4\t\n"
      "vmovups ymmword PTR [r12 + 32], ymm5\t\n"
      "add r12, r13\t\n"
      "vmovups ymmword PTR [r12 + 0], ymm6\t\n"
      "vmovups ymmword PTR [r12 + 32], ymm7\t\n"
      "add r12, r13\t\n"
      "vmovups ymmword PTR [r12 + 0], ymm8\t\n"
      "vmovups ymmword PTR [r12 + 32], ymm9\t\n"

      // next outer iteration
      "add rcx, 64\t\n"
      "mov r12, rcx\t\n"
      "mov r9, rax\t\n"
      "inc rbx\t\n"
      "cmp rbx, rdi\t\n"
      "jl loop_outter%=\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r13",
        "r14",
        "rax",
        "rcx",
        "rsi",
        "rdi",
        "rbx",
        "r12",
        "r15",
        "memory");
}
void NOINLINE
gemmkernel_6x2_Avx2_fp16_fA0fB0fC0(GemmParamsFP16* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // k
      "mov r8, [r14 + 0]\t\n"
      "dec r8\t\n"
      // A
      "mov r9, [r14 + 8]\t\n"
      // B
      "mov r10, [r14 + 16]\t\n"
      // beta
      "lea r15, [r14 + 24]\t\n"
      // C
      "mov r12, [r14 + 32]\t\n"
      // ldc
      "mov r13, [r14 + 40]\t\n"
      // b_block_cols
      "mov rdi, [r14 + 48]\t\n"
      // b_block_size
      "mov rsi, [r14 + 56]\t\n"

      // Make copies of A and C
      "mov rax, r9\t\n"
      "mov rcx, r12\t\n"

      "mov rbx, 0\t\n"
      "loop_outter%=:\t\n"
      "mov r14, r8\t\n"
      "vbroadcastss ymm15,DWORD PTR [r15]\t\n"
      "vcvtph2ps ymm13,XMMWORD PTR [r10 + 0]\t\n"
      "vcvtph2ps ymm14,XMMWORD PTR [r10 + 16]\t\n"
      "vxorps xmm0, xmm0, xmm0\t\n"
      "vcomiss xmm15, xmm0\t\n"
      "jz zero_regs%=\t\n"

      // Setup values with beta multiplication
      "vmulps ymm0, ymm15, [r12 + 0]\t\n"
      "vmulps ymm1, ymm15, [r12 + 32]\t\n"
      "add r12, r13\t\n"
      "vmulps ymm2, ymm15, [r12 + 0]\t\n"
      "vmulps ymm3, ymm15, [r12 + 32]\t\n"
      "add r12, r13\t\n"
      "vmulps ymm4, ymm15, [r12 + 0]\t\n"
      "vmulps ymm5, ymm15, [r12 + 32]\t\n"
      "add r12, r13\t\n"
      "vmulps ymm6, ymm15, [r12 + 0]\t\n"
      "vmulps ymm7, ymm15, [r12 + 32]\t\n"
      "add r12, r13\t\n"
      "vmulps ymm8, ymm15, [r12 + 0]\t\n"
      "vmulps ymm9, ymm15, [r12 + 32]\t\n"
      "add r12, r13\t\n"
      "vmulps ymm10, ymm15, [r12 + 0]\t\n"
      "vmulps ymm11, ymm15, [r12 + 32]\t\n"
      "test r14,r14\t\n"
      "jz skip_preload%=\t\n"
      "vcvtph2ps ymm15,XMMWORD PTR [r10 + 32]\t\n"
      "skip_preload%=:\t\n"
      "vbroadcastss ymm12,DWORD PTR [r9+0]\t\n"
      "vfmadd231ps ymm0,ymm13,ymm12\t\n"
      "vfmadd231ps ymm1,ymm14,ymm12\t\n"
      "vbroadcastss ymm12,DWORD PTR [r9+4]\t\n"
      "vfmadd231ps ymm2,ymm13,ymm12\t\n"
      "vfmadd231ps ymm3,ymm14,ymm12\t\n"
      "vbroadcastss ymm12,DWORD PTR [r9+8]\t\n"
      "vfmadd231ps ymm4,ymm13,ymm12\t\n"
      "vfmadd231ps ymm5,ymm14,ymm12\t\n"
      "vbroadcastss ymm12,DWORD PTR [r9+12]\t\n"
      "vfmadd231ps ymm6,ymm13,ymm12\t\n"
      "vfmadd231ps ymm7,ymm14,ymm12\t\n"
      "vbroadcastss ymm12,DWORD PTR [r9+16]\t\n"
      "vfmadd231ps ymm8,ymm13,ymm12\t\n"
      "vfmadd231ps ymm9,ymm14,ymm12\t\n"
      "vbroadcastss ymm12,DWORD PTR [r9+20]\t\n"
      "vfmadd231ps ymm10,ymm13,ymm12\t\n"
      "vfmadd231ps ymm11,ymm14,ymm12\t\n"
      "mov r12, rcx\t\n"
      "test r14,r14\t\n"
      "jnz next_inner%=\t\n"
      "add r10,32\t\n"
      "jmp dump_C%=\t\n"

      "zero_regs%=:\t\n"

      "test r14,r14\t\n"
      "jz skip_preload_b_zero%=\t\n"
      "vcvtph2ps ymm15,XMMWORD PTR [r10 + 32]\t\n"
      "skip_preload_b_zero%=:\t\n"
      "vbroadcastss ymm12,DWORD PTR [r9+0]\t\n"
      "vmulps ymm0,ymm13,ymm12\t\n"
      "vmulps ymm1,ymm14,ymm12\t\n"
      "add r12, r13\t\n"
      "vbroadcastss ymm12,DWORD PTR [r9+4]\t\n"
      "vmulps ymm2,ymm13,ymm12\t\n"
      "vmulps ymm3,ymm14,ymm12\t\n"
      "add r12, r13\t\n"
      "vbroadcastss ymm12,DWORD PTR [r9+8]\t\n"
      "vmulps ymm4,ymm13,ymm12\t\n"
      "vmulps ymm5,ymm14,ymm12\t\n"
      "add r12, r13\t\n"
      "vbroadcastss ymm12,DWORD PTR [r9+12]\t\n"
      "vmulps ymm6,ymm13,ymm12\t\n"
      "vmulps ymm7,ymm14,ymm12\t\n"
      "add r12, r13\t\n"
      "vbroadcastss ymm12,DWORD PTR [r9+16]\t\n"
      "vmulps ymm8,ymm13,ymm12\t\n"
      "vmulps ymm9,ymm14,ymm12\t\n"
      "add r12, r13\t\n"
      "vbroadcastss ymm12,DWORD PTR [r9+20]\t\n"
      "vmulps ymm10,ymm13,ymm12\t\n"
      "vmulps ymm11,ymm14,ymm12\t\n"
      "mov r12, rcx\t\n"
      "test r14,r14\t\n"
      "jnz next_inner%=\t\n"
      "add r10,32\t\n"
      "jmp dump_C%=\t\n"

      "loop_inner%=:\t\n"

      "vmovaps ymm13,ymm15\t\n"
      "vcvtph2ps ymm14,XMMWORD PTR [r10 + 16]\t\n"
      "vcvtph2ps ymm15,XMMWORD PTR [r10 + 32]\t\n"
      "vbroadcastss ymm12,DWORD PTR [r9+0]\t\n"
      "vfmadd231ps ymm0,ymm13,ymm12\t\n"
      "vfmadd231ps ymm1,ymm14,ymm12\t\n"
      "vbroadcastss ymm12,DWORD PTR [r9+4]\t\n"
      "vfmadd231ps ymm2,ymm13,ymm12\t\n"
      "vfmadd231ps ymm3,ymm14,ymm12\t\n"
      "vbroadcastss ymm12,DWORD PTR [r9+8]\t\n"
      "vfmadd231ps ymm4,ymm13,ymm12\t\n"
      "vfmadd231ps ymm5,ymm14,ymm12\t\n"
      "vbroadcastss ymm12,DWORD PTR [r9+12]\t\n"
      "vfmadd231ps ymm6,ymm13,ymm12\t\n"
      "vfmadd231ps ymm7,ymm14,ymm12\t\n"
      "vbroadcastss ymm12,DWORD PTR [r9+16]\t\n"
      "vfmadd231ps ymm8,ymm13,ymm12\t\n"
      "vfmadd231ps ymm9,ymm14,ymm12\t\n"
      "vbroadcastss ymm12,DWORD PTR [r9+20]\t\n"
      "vfmadd231ps ymm10,ymm13,ymm12\t\n"
      "vfmadd231ps ymm11,ymm14,ymm12\t\n"

      "next_inner%=:\t\n"
      "add r9,24\t\n"
      "add r10,32\t\n"
      "dec r14\t\n"
      "jnz loop_inner%=\t\n"

      "vmovaps ymm13,ymm15\t\n"
      "vcvtph2ps ymm14,XMMWORD PTR [r10 + 16]\t\n"
      "vbroadcastss ymm12,DWORD PTR [r9+0]\t\n"
      "vfmadd231ps ymm0,ymm13,ymm12\t\n"
      "vfmadd231ps ymm1,ymm14,ymm12\t\n"
      "vbroadcastss ymm12,DWORD PTR [r9+4]\t\n"
      "vfmadd231ps ymm2,ymm13,ymm12\t\n"
      "vfmadd231ps ymm3,ymm14,ymm12\t\n"
      "vbroadcastss ymm12,DWORD PTR [r9+8]\t\n"
      "vfmadd231ps ymm4,ymm13,ymm12\t\n"
      "vfmadd231ps ymm5,ymm14,ymm12\t\n"
      "vbroadcastss ymm12,DWORD PTR [r9+12]\t\n"
      "vfmadd231ps ymm6,ymm13,ymm12\t\n"
      "vfmadd231ps ymm7,ymm14,ymm12\t\n"
      "vbroadcastss ymm12,DWORD PTR [r9+16]\t\n"
      "vfmadd231ps ymm8,ymm13,ymm12\t\n"
      "vfmadd231ps ymm9,ymm14,ymm12\t\n"
      "vbroadcastss ymm12,DWORD PTR [r9+20]\t\n"
      "vfmadd231ps ymm10,ymm13,ymm12\t\n"
      "vfmadd231ps ymm11,ymm14,ymm12\t\n"
      "add r9,24\t\n"
      "add r10,32\t\n"
      // Dump C
      "dump_C%=:\t\n"
      "vmovups ymmword PTR [r12 + 0], ymm0\t\n"
      "vmovups ymmword PTR [r12 + 32], ymm1\t\n"
      "add r12, r13\t\n"
      "vmovups ymmword PTR [r12 + 0], ymm2\t\n"
      "vmovups ymmword PTR [r12 + 32], ymm3\t\n"
      "add r12, r13\t\n"
      "vmovups ymmword PTR [r12 + 0], ymm4\t\n"
      "vmovups ymmword PTR [r12 + 32], ymm5\t\n"
      "add r12, r13\t\n"
      "vmovups ymmword PTR [r12 + 0], ymm6\t\n"
      "vmovups ymmword PTR [r12 + 32], ymm7\t\n"
      "add r12, r13\t\n"
      "vmovups ymmword PTR [r12 + 0], ymm8\t\n"
      "vmovups ymmword PTR [r12 + 32], ymm9\t\n"
      "add r12, r13\t\n"
      "vmovups ymmword PTR [r12 + 0], ymm10\t\n"
      "vmovups ymmword PTR [r12 + 32], ymm11\t\n"

      // next outer iteration
      "add rcx, 64\t\n"
      "mov r12, rcx\t\n"
      "mov r9, rax\t\n"
      "inc rbx\t\n"
      "cmp rbx, rdi\t\n"
      "jl loop_outter%=\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r13",
        "r14",
        "rax",
        "rcx",
        "rsi",
        "rdi",
        "rbx",
        "r12",
        "r15",
        "memory");
}
#else // _MSC_VER
// Intrinsic kernel for MSVC
void gemmkernel_Avx2_fp16_fA0fB0fC0(GemmParamsFP16* gp, const size_t kernel_nrows) {
  // register buffer
  __m256 ymmSum[12];
  size_t idxA = 0, idxB = 0, idxC = 0;
  // ldc in float size
  size_t ldc_floatsize = gp->ldc / sizeof(float);
  // load beta
  __m256 ymmBeta;
  if (gp->beta != 0)
    ymmBeta = _mm256_broadcast_ss(&gp->beta);

  // outer loop - block columns
  for(uint64_t ii = 0; ii < gp->b_block_cols; ii++) {
    // reset index
    idxA = 0;
    // inner loop - k
    for(uint64_t kk = 0; kk < gp->k; kk++) {
      // load B
      __m256 ymmB0 = _mm256_cvtph_ps(_mm_load_si128((__m128i*)(gp->B + idxB)));
      __m256 ymmB1 = _mm256_cvtph_ps(_mm_load_si128((__m128i*)(gp->B + idxB + 8)));
      idxB += 16;

      // first element
      if (kk == 0) {
        if(gp->beta != 0) {  // accumulate
          for(size_t jj = 0; jj < kernel_nrows; jj++) {
            // load A
            __m256 ymmA = _mm256_broadcastss_ps(_mm_broadcast_ss((float const*)(gp->A + idxA + jj)));
            // C = A * B + beta * C
            ymmSum[2 * jj] = _mm256_fmadd_ps(ymmA, ymmB0, _mm256_mul_ps(ymmBeta, _mm256_loadu_ps(gp->C + idxC + jj * ldc_floatsize)));
            ymmSum[2 * jj + 1] = _mm256_fmadd_ps(ymmA, ymmB1, _mm256_mul_ps(ymmBeta, _mm256_loadu_ps(gp->C + idxC + 8 + jj * ldc_floatsize)));
          }
          idxA += kernel_nrows;
        } else {  // set zero
          for(size_t jj = 0; jj < kernel_nrows; jj++) {
            // load A
            __m256 ymmA = _mm256_broadcastss_ps(_mm_broadcast_ss((float const*)(gp->A + idxA + jj)));
            // C = A * B
            ymmSum[2 * jj] = _mm256_mul_ps(ymmA, ymmB0);
            ymmSum[2 * jj + 1] = _mm256_mul_ps(ymmA, ymmB1);
          }
          idxA += kernel_nrows;
        }
      } else {
        for(size_t jj = 0; jj < kernel_nrows; jj++) {
          // load A
          __m256 ymmA = _mm256_broadcastss_ps(_mm_broadcast_ss((float const*)(gp->A + idxA + jj)));
          // C = A * B + C
          ymmSum[2 * jj] = _mm256_fmadd_ps(ymmA, ymmB0, ymmSum[2 * jj]);
          ymmSum[2 * jj + 1] = _mm256_fmadd_ps(ymmA, ymmB1, ymmSum[2 * jj + 1]);
        }
        idxA += kernel_nrows;
      }
    }
    // store C
    for(size_t jj = 0; jj < kernel_nrows; jj++) {
      _mm256_storeu_ps(gp->C + idxC + jj * ldc_floatsize, ymmSum[2 * jj]);
      _mm256_storeu_ps(gp->C + idxC + 8 + jj * ldc_floatsize, ymmSum[2 * jj + 1]);
    }
    idxC += 16;
  }
}

void NOINLINE
gemmkernel_1x2_Avx2_fp16_fA0fB0fC0(GemmParamsFP16* gp) {
  gemmkernel_Avx2_fp16_fA0fB0fC0(gp, 1);
}
void NOINLINE
gemmkernel_2x2_Avx2_fp16_fA0fB0fC0(GemmParamsFP16* gp) {
  gemmkernel_Avx2_fp16_fA0fB0fC0(gp, 2);
}
void NOINLINE
gemmkernel_3x2_Avx2_fp16_fA0fB0fC0(GemmParamsFP16* gp) {
  gemmkernel_Avx2_fp16_fA0fB0fC0(gp, 3);
}
void NOINLINE
gemmkernel_4x2_Avx2_fp16_fA0fB0fC0(GemmParamsFP16* gp) {
  gemmkernel_Avx2_fp16_fA0fB0fC0(gp, 4);
}
void NOINLINE
gemmkernel_5x2_Avx2_fp16_fA0fB0fC0(GemmParamsFP16* gp) {
  gemmkernel_Avx2_fp16_fA0fB0fC0(gp, 5);
}
void NOINLINE
gemmkernel_6x2_Avx2_fp16_fA0fB0fC0(GemmParamsFP16* gp) {
  gemmkernel_Avx2_fp16_fA0fB0fC0(gp, 6);
}
#endif // _MSC_VER
} // namespace fbgemm
