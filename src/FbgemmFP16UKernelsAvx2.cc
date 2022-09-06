/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "./FbgemmFP16UKernelsAvx2.h"
#include "./InlineAsmDefines.h"

namespace fbgemm {

void NOINLINE gemmkernel_1x2_Avx2_fp16_fA0fB0fC0(GemmParamsFP16* gp) {
  asm volatile(
#if FBGEMM_USE_CLANG_INTEL_SYNTAX_ASM_HACK
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#else
      "mov r14, %[gp]\t\n"
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

      "xor ebx, ebx\t\n"
      "loop_outter%=:\t\n"
      "mov r14, r8\t\n"

      "vxorps ymm5, ymm5, ymm5\t\n"
      "vxorps ymm6, ymm6, ymm6\t\n"
      "vxorps ymm7, ymm7, ymm7\t\n"
      "vxorps ymm8, ymm8, ymm8\t\n"
      "vxorps ymm9, ymm9, ymm9\t\n"
      "vxorps ymm10, ymm10, ymm10\t\n"

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
      "jnz loop_inner_start%=\t\n"
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
      "jnz loop_inner_start%=\t\n"
      "add r10,32\t\n"
      "jmp dump_C%=\t\n"

      "loop_inner_start%=:\t\n"
      "add r9,4\t\n"
      "add r10,32\t\n"
      "cmp r14,4\t\n"

      "jle loop_inner_end%=\t\n"

      "loop_inner%=:\t\n"

      "vcvtph2ps ymm4,XMMWORD PTR [r10 + 16]\t\n"
      "vcvtph2ps ymm3,XMMWORD PTR [r10 + 32]\t\n"
      "vbroadcastss ymm2,DWORD PTR [r9+0]\t\n"
      "vfmadd231ps ymm0,ymm15,ymm2\t\n"
      "vfmadd231ps ymm1,ymm4,ymm2\t\n"

      "vcvtph2ps ymm4,XMMWORD PTR [r10 + 48]\t\n"
      "vcvtph2ps ymm15,XMMWORD PTR [r10 + 64]\t\n"
      "vbroadcastss ymm2,DWORD PTR [r9+4]\t\n"
      "vfmadd231ps ymm5,ymm3,ymm2\t\n"
      "vfmadd231ps ymm6,ymm4,ymm2\t\n"

      "vcvtph2ps ymm4,XMMWORD PTR [r10 + 80]\t\n"
      "vcvtph2ps ymm3,XMMWORD PTR [r10 + 96]\t\n"
      "vbroadcastss ymm2,DWORD PTR [r9+8]\t\n"
      "vfmadd231ps ymm7,ymm15,ymm2\t\n"
      "vfmadd231ps ymm8,ymm4,ymm2\t\n"

      "vcvtph2ps ymm4,XMMWORD PTR [r10 + 112]\t\n"
      "vcvtph2ps ymm15,XMMWORD PTR [r10 + 128]\t\n"
      "vbroadcastss ymm2,DWORD PTR [r9+12]\t\n"
      "vfmadd231ps ymm9,ymm3,ymm2\t\n"
      "vfmadd231ps ymm10,ymm4,ymm2\t\n"

      "next_inner%=:\t\n"
      "add r9,16\t\n"
      "add r10,128\t\n"
      "sub r14,4\t\n"

      "cmp r14, 4\t\n"
      "jg loop_inner%=\t\n"
      "loop_inner_end%=:\t\n"

      "cmp r14, 0\t\n"
      "jz loop_tail%=\t\n"

      "vcvtph2ps ymm3,XMMWORD PTR [r10]\t\n"
      "vcvtph2ps ymm4,XMMWORD PTR [r10 + 16]\t\n"
      "vbroadcastss ymm2,DWORD PTR [r9+0]\t\n"
      "vfmadd231ps ymm0,ymm3,ymm2\t\n"
      "vfmadd231ps ymm1,ymm4,ymm2\t\n"
      "add r9,4\t\n"
      "add r10,32\t\n"
      "dec r14\t\n"

      "jmp loop_inner_end%=\t\n"

      "loop_tail%=:\t\n"
      "vaddps ymm0, ymm0, ymm5\t\n"
      "vaddps ymm0, ymm0, ymm7\t\n"
      "vaddps ymm0, ymm0, ymm9\t\n"
      "vaddps ymm1, ymm1, ymm6\t\n"
      "vaddps ymm1, ymm1, ymm8\t\n"
      "vaddps ymm1, ymm1, ymm10\t\n"

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
void NOINLINE gemmkernel_2x2_Avx2_fp16_fA0fB0fC0(GemmParamsFP16* gp) {
  asm volatile(
#if FBGEMM_USE_CLANG_INTEL_SYNTAX_ASM_HACK
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#else
      "mov r14, %[gp]\t\n"
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

      "xor ebx, ebx\t\n"
      "loop_outter%=:\t\n"
      "mov r14, r8\t\n"

      "vxorps ymm7, ymm7, ymm7\t\n"
      "vxorps ymm8, ymm8, ymm8\t\n"
      "vxorps ymm9, ymm9, ymm9\t\n"
      "vxorps ymm10, ymm10, ymm10\t\n"

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
      "jnz loop_inner_start%=\t\n"
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
      "jnz loop_inner_start%=\t\n"
      "add r10,32\t\n"
      "jmp dump_C%=\t\n"

      "loop_inner_start%=:\t\n"
      "add r9,8\t\n"
      "add r10,32\t\n"
      "cmp r14,4\t\n"

      "jle loop_inner_end%=\t\n"

      "loop_inner%=:\t\n"

      "vcvtph2ps ymm6,XMMWORD PTR [r10 + 16]\t\n"
      "vcvtph2ps ymm5,XMMWORD PTR [r10 + 32]\t\n"
      "vbroadcastss ymm4,DWORD PTR [r9+0]\t\n"
      "vfmadd231ps ymm0,ymm15,ymm4\t\n"
      "vfmadd231ps ymm1,ymm6,ymm4\t\n"
      "vbroadcastss ymm4,DWORD PTR [r9+4]\t\n"
      "vfmadd231ps ymm2,ymm15,ymm4\t\n"
      "vfmadd231ps ymm3,ymm6,ymm4\t\n"

      "vcvtph2ps ymm6,XMMWORD PTR [r10 + 48]\t\n"
      "vcvtph2ps ymm15,XMMWORD PTR [r10 + 64]\t\n"
      "vbroadcastss ymm4,DWORD PTR [r9+8]\t\n"
      "vfmadd231ps ymm7,ymm5,ymm4\t\n"
      "vfmadd231ps ymm8,ymm6,ymm4\t\n"
      "vbroadcastss ymm4,DWORD PTR [r9+12]\t\n"
      "vfmadd231ps ymm9,ymm5,ymm4\t\n"
      "vfmadd231ps ymm10,ymm6,ymm4\t\n"

      "vcvtph2ps ymm6,XMMWORD PTR [r10 + 80]\t\n"
      "vcvtph2ps ymm5,XMMWORD PTR [r10 + 96]\t\n"
      "vbroadcastss ymm4,DWORD PTR [r9+16]\t\n"
      "vfmadd231ps ymm0,ymm15,ymm4\t\n"
      "vfmadd231ps ymm1,ymm6,ymm4\t\n"
      "vbroadcastss ymm4,DWORD PTR [r9+20]\t\n"
      "vfmadd231ps ymm2,ymm15,ymm4\t\n"
      "vfmadd231ps ymm3,ymm6,ymm4\t\n"

      "vcvtph2ps ymm6,XMMWORD PTR [r10 + 112]\t\n"
      "vcvtph2ps ymm15,XMMWORD PTR [r10 + 128]\t\n"
      "vbroadcastss ymm4,DWORD PTR [r9+24]\t\n"
      "vfmadd231ps ymm7,ymm5,ymm4\t\n"
      "vfmadd231ps ymm8,ymm6,ymm4\t\n"
      "vbroadcastss ymm4,DWORD PTR [r9+28]\t\n"
      "vfmadd231ps ymm9,ymm5,ymm4\t\n"
      "vfmadd231ps ymm10,ymm6,ymm4\t\n"

      "next_inner%=:\t\n"
      "add r9,32\t\n"
      "add r10,128\t\n"
      "sub r14,4\t\n"

      "cmp r14, 4\t\n"
      "jg loop_inner%=\t\n"
      "loop_inner_end%=:\t\n"

      "cmp r14, 0\t\n"
      "jz loop_tail%=\t\n"

      "vcvtph2ps ymm5,XMMWORD PTR [r10]\t\n"
      "vcvtph2ps ymm6,XMMWORD PTR [r10 + 16]\t\n"
      "vbroadcastss ymm4,DWORD PTR [r9+0]\t\n"
      "vfmadd231ps ymm0,ymm5,ymm4\t\n"
      "vfmadd231ps ymm1,ymm6,ymm4\t\n"
      "vbroadcastss ymm4,DWORD PTR [r9+4]\t\n"
      "vfmadd231ps ymm2,ymm5,ymm4\t\n"
      "vfmadd231ps ymm3,ymm6,ymm4\t\n"
      "add r9,8\t\n"
      "add r10,32\t\n"
      "dec r14\t\n"

      "jmp loop_inner_end%=\t\n"

      "loop_tail%=:\t\n"
      "vaddps ymm0, ymm0, ymm7\t\n"
      "vaddps ymm1, ymm1, ymm8\t\n"
      "vaddps ymm2, ymm2, ymm9\t\n"
      "vaddps ymm3, ymm3, ymm10\t\n"

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
void NOINLINE gemmkernel_3x2_Avx2_fp16_fA0fB0fC0(GemmParamsFP16* gp) {
  asm volatile(
#if FBGEMM_USE_CLANG_INTEL_SYNTAX_ASM_HACK
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#else
      "mov r14, %[gp]\t\n"
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

      "xor ebx, ebx\t\n"
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
void NOINLINE gemmkernel_4x2_Avx2_fp16_fA0fB0fC0(GemmParamsFP16* gp) {
  asm volatile(
#if FBGEMM_USE_CLANG_INTEL_SYNTAX_ASM_HACK
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#else
      "mov r14, %[gp]\t\n"
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

      "xor ebx, ebx\t\n"
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
void NOINLINE gemmkernel_5x2_Avx2_fp16_fA0fB0fC0(GemmParamsFP16* gp) {
  asm volatile(
#if FBGEMM_USE_CLANG_INTEL_SYNTAX_ASM_HACK
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#else
      "mov r14, %[gp]\t\n"
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

      "xor ebx, ebx\t\n"
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
void NOINLINE gemmkernel_6x2_Avx2_fp16_fA0fB0fC0(GemmParamsFP16* gp) {
  asm volatile(
#if FBGEMM_USE_CLANG_INTEL_SYNTAX_ASM_HACK
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#else
      "mov r14, %[gp]\t\n"
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

      "xor ebx, ebx\t\n"
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

} // namespace fbgemm
