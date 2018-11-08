/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "FbgemmFP16UKernels.h"

namespace fbgemm {

void __attribute__ ((noinline)) gemmkernel_1x1_AVX2_fA0fB0fC0(GemmParams *gp)
{
asm volatile
(
#if !defined(__clang__)
"mov r14, %[gp]\t\n"
#else
"mov %[gp], %%r14\t\n"
".intel_syntax noprefix\t\n"
#endif

// Copy parameters
// k
"mov r8, [r14 + 0]\t\n"
// A
"mov r9, [r14 + 8]\t\n"
// B
"mov r10, [r14 + 16]\t\n"
// beta
"mov r15, [r14 + 24]\t\n"
// accum
"mov rdx, [r14 + 32]\t\n"
// C
"mov r12, [r14 + 40]\t\n"
// ldc
"mov r13, [r14 + 48]\t\n"
// b_block_cols
"mov rdi, [r14 + 56]\t\n"
// b_block_size
"mov rsi, [r14 + 64]\t\n"
// Make copies of A and C
"mov rax, r9\t\n"
"mov rcx, r12\t\n"


"mov rbx, 0\t\n"
"loop_outter%=:\t\n"
"mov r14, 0\t\n"
"vxorps ymm0,ymm0,ymm0\t\n"

"vcvtph2ps ymm15, XMMWORD PTR [r10 + 0]\t\n"
"mov r11, 16\t\n"

"loop_inner%=:\t\n"

"vcvtph2ps ymm14,XMMWORD PTR [r10 + r11 + 0]\t\n"
"inc r14\t\n"
"vbroadcastss ymm1,DWORD PTR [r9+0]\t\n"
"vfmadd231ps ymm0,ymm15,ymm1\t\n"
"cmp r14, r8\t\n"
"jge L_exit%=\t\n"
"vcvtph2ps ymm15,XMMWORD PTR [r10 + r11 + 16]\t\n"
"inc r14\t\n"
"vbroadcastss ymm1,DWORD PTR [r9+4]\t\n"
"vfmadd231ps ymm0,ymm14,ymm1\t\n"
"add r11, 32\t\n"
"add r9,8\t\n"
"cmp r14, r8\t\n"
"jl loop_inner%=\t\n"

"L_exit%=:\t\n"
"add r10, rsi\t\n"

"cmp rdx, 1\t\n"
"je L_accum%=\t\n"
// Dump C
"vmovups YMMWORD PTR [r12 + 0], ymm0\t\n"
"add r12, r13\t\n"
"jmp L_done%=\t\n"


"L_accum%=:\t\n"
// Dump C with accumulate
"vbroadcastss ymm15,DWORD PTR [r15]\t\n"
"vfmadd231ps ymm0,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm0\t\n"
"add r12, r13\t\n"

"L_done%=:\t\n"

// next outer iteration
"add rcx, 32\t\n"
"mov r12, rcx\t\n"
"mov r9, rax\t\n"
"inc rbx\t\n"
"cmp rbx, rdi\t\n"
"jl loop_outter%=\t\n"
:
:
[gp] "rm" (gp)
: "r8", "r9", "r10", "r11", "r15",  "r13", "r14",
"rax", "rcx", "rdx", "rsi", "rdi", "rbx", "r12", "memory"
);
}
void __attribute__ ((noinline)) gemmkernel_2x1_AVX2_fA0fB0fC0(GemmParams *gp)
{
asm volatile
(
#if !defined(__clang__)
"mov r14, %[gp]\t\n"
#else
"mov %[gp], %%r14\t\n"
".intel_syntax noprefix\t\n"
#endif

// Copy parameters
// k
"mov r8, [r14 + 0]\t\n"
// A
"mov r9, [r14 + 8]\t\n"
// B
"mov r10, [r14 + 16]\t\n"
// beta
"mov r15, [r14 + 24]\t\n"
// accum
"mov rdx, [r14 + 32]\t\n"
// C
"mov r12, [r14 + 40]\t\n"
// ldc
"mov r13, [r14 + 48]\t\n"
// b_block_cols
"mov rdi, [r14 + 56]\t\n"
// b_block_size
"mov rsi, [r14 + 64]\t\n"
// Make copies of A and C
"mov rax, r9\t\n"
"mov rcx, r12\t\n"


"mov rbx, 0\t\n"
"loop_outter%=:\t\n"
"mov r14, 0\t\n"
"vxorps ymm0,ymm0,ymm0\t\n"
"vxorps ymm1,ymm1,ymm1\t\n"

"vcvtph2ps ymm15, XMMWORD PTR [r10 + 0]\t\n"
"mov r11, 16\t\n"

"loop_inner%=:\t\n"

"vcvtph2ps ymm14,XMMWORD PTR [r10 + r11 + 0]\t\n"
"inc r14\t\n"
"vbroadcastss ymm2,DWORD PTR [r9+0]\t\n"
"vfmadd231ps ymm0,ymm15,ymm2\t\n"
"vbroadcastss ymm2,DWORD PTR [r9+4]\t\n"
"vfmadd231ps ymm1,ymm15,ymm2\t\n"
"cmp r14, r8\t\n"
"jge L_exit%=\t\n"
"vcvtph2ps ymm15,XMMWORD PTR [r10 + r11 + 16]\t\n"
"inc r14\t\n"
"vbroadcastss ymm2,DWORD PTR [r9+8]\t\n"
"vfmadd231ps ymm0,ymm14,ymm2\t\n"
"vbroadcastss ymm2,DWORD PTR [r9+12]\t\n"
"vfmadd231ps ymm1,ymm14,ymm2\t\n"
"add r11, 32\t\n"
"add r9,16\t\n"
"cmp r14, r8\t\n"
"jl loop_inner%=\t\n"

"L_exit%=:\t\n"
"add r10, rsi\t\n"

"cmp rdx, 1\t\n"
"je L_accum%=\t\n"
// Dump C
"vmovups YMMWORD PTR [r12 + 0], ymm0\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm1\t\n"
"add r12, r13\t\n"
"jmp L_done%=\t\n"


"L_accum%=:\t\n"
// Dump C with accumulate
"vbroadcastss ymm15,DWORD PTR [r15]\t\n"
"vfmadd231ps ymm0,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm0\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm1,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm1\t\n"
"add r12, r13\t\n"

"L_done%=:\t\n"

// next outer iteration
"add rcx, 32\t\n"
"mov r12, rcx\t\n"
"mov r9, rax\t\n"
"inc rbx\t\n"
"cmp rbx, rdi\t\n"
"jl loop_outter%=\t\n"
:
:
[gp] "rm" (gp)
: "r8", "r9", "r10", "r11", "r15",  "r13", "r14",
"rax", "rcx", "rdx", "rsi", "rdi", "rbx", "r12", "memory"
);
}
void __attribute__ ((noinline)) gemmkernel_3x1_AVX2_fA0fB0fC0(GemmParams *gp)
{
asm volatile
(
#if !defined(__clang__)
"mov r14, %[gp]\t\n"
#else
"mov %[gp], %%r14\t\n"
".intel_syntax noprefix\t\n"
#endif

// Copy parameters
// k
"mov r8, [r14 + 0]\t\n"
// A
"mov r9, [r14 + 8]\t\n"
// B
"mov r10, [r14 + 16]\t\n"
// beta
"mov r15, [r14 + 24]\t\n"
// accum
"mov rdx, [r14 + 32]\t\n"
// C
"mov r12, [r14 + 40]\t\n"
// ldc
"mov r13, [r14 + 48]\t\n"
// b_block_cols
"mov rdi, [r14 + 56]\t\n"
// b_block_size
"mov rsi, [r14 + 64]\t\n"
// Make copies of A and C
"mov rax, r9\t\n"
"mov rcx, r12\t\n"


"mov rbx, 0\t\n"
"loop_outter%=:\t\n"
"mov r14, 0\t\n"
"vxorps ymm0,ymm0,ymm0\t\n"
"vxorps ymm1,ymm1,ymm1\t\n"
"vxorps ymm2,ymm2,ymm2\t\n"

"vcvtph2ps ymm15, XMMWORD PTR [r10 + 0]\t\n"
"mov r11, 16\t\n"

"loop_inner%=:\t\n"

"vcvtph2ps ymm14,XMMWORD PTR [r10 + r11 + 0]\t\n"
"inc r14\t\n"
"vbroadcastss ymm3,DWORD PTR [r9+0]\t\n"
"vfmadd231ps ymm0,ymm15,ymm3\t\n"
"vbroadcastss ymm3,DWORD PTR [r9+4]\t\n"
"vfmadd231ps ymm1,ymm15,ymm3\t\n"
"vbroadcastss ymm3,DWORD PTR [r9+8]\t\n"
"vfmadd231ps ymm2,ymm15,ymm3\t\n"
"cmp r14, r8\t\n"
"jge L_exit%=\t\n"
"vcvtph2ps ymm15,XMMWORD PTR [r10 + r11 + 16]\t\n"
"inc r14\t\n"
"vbroadcastss ymm3,DWORD PTR [r9+12]\t\n"
"vfmadd231ps ymm0,ymm14,ymm3\t\n"
"vbroadcastss ymm3,DWORD PTR [r9+16]\t\n"
"vfmadd231ps ymm1,ymm14,ymm3\t\n"
"add r11, 32\t\n"
"vbroadcastss ymm3,DWORD PTR [r9+20]\t\n"
"vfmadd231ps ymm2,ymm14,ymm3\t\n"
"add r9,24\t\n"
"cmp r14, r8\t\n"
"jl loop_inner%=\t\n"

"L_exit%=:\t\n"
"add r10, rsi\t\n"

"cmp rdx, 1\t\n"
"je L_accum%=\t\n"
// Dump C
"vmovups YMMWORD PTR [r12 + 0], ymm0\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm1\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm2\t\n"
"add r12, r13\t\n"
"jmp L_done%=\t\n"


"L_accum%=:\t\n"
// Dump C with accumulate
"vbroadcastss ymm15,DWORD PTR [r15]\t\n"
"vfmadd231ps ymm0,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm0\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm1,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm1\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm2,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm2\t\n"
"add r12, r13\t\n"

"L_done%=:\t\n"

// next outer iteration
"add rcx, 32\t\n"
"mov r12, rcx\t\n"
"mov r9, rax\t\n"
"inc rbx\t\n"
"cmp rbx, rdi\t\n"
"jl loop_outter%=\t\n"
:
:
[gp] "rm" (gp)
: "r8", "r9", "r10", "r11", "r15",  "r13", "r14",
"rax", "rcx", "rdx", "rsi", "rdi", "rbx", "r12", "memory"
);
}
void __attribute__ ((noinline)) gemmkernel_4x1_AVX2_fA0fB0fC0(GemmParams *gp)
{
asm volatile
(
#if !defined(__clang__)
"mov r14, %[gp]\t\n"
#else
"mov %[gp], %%r14\t\n"
".intel_syntax noprefix\t\n"
#endif

// Copy parameters
// k
"mov r8, [r14 + 0]\t\n"
// A
"mov r9, [r14 + 8]\t\n"
// B
"mov r10, [r14 + 16]\t\n"
// beta
"mov r15, [r14 + 24]\t\n"
// accum
"mov rdx, [r14 + 32]\t\n"
// C
"mov r12, [r14 + 40]\t\n"
// ldc
"mov r13, [r14 + 48]\t\n"
// b_block_cols
"mov rdi, [r14 + 56]\t\n"
// b_block_size
"mov rsi, [r14 + 64]\t\n"
// Make copies of A and C
"mov rax, r9\t\n"
"mov rcx, r12\t\n"


"mov rbx, 0\t\n"
"loop_outter%=:\t\n"
"mov r14, 0\t\n"
"vxorps ymm0,ymm0,ymm0\t\n"
"vxorps ymm1,ymm1,ymm1\t\n"
"vxorps ymm2,ymm2,ymm2\t\n"
"vxorps ymm3,ymm3,ymm3\t\n"

"vcvtph2ps ymm15, XMMWORD PTR [r10 + 0]\t\n"
"mov r11, 16\t\n"

"loop_inner%=:\t\n"

"vcvtph2ps ymm14,XMMWORD PTR [r10 + r11 + 0]\t\n"
"inc r14\t\n"
"vbroadcastss ymm4,DWORD PTR [r9+0]\t\n"
"vfmadd231ps ymm0,ymm15,ymm4\t\n"
"vbroadcastss ymm4,DWORD PTR [r9+4]\t\n"
"vfmadd231ps ymm1,ymm15,ymm4\t\n"
"vbroadcastss ymm4,DWORD PTR [r9+8]\t\n"
"vfmadd231ps ymm2,ymm15,ymm4\t\n"
"vbroadcastss ymm4,DWORD PTR [r9+12]\t\n"
"vfmadd231ps ymm3,ymm15,ymm4\t\n"
"cmp r14, r8\t\n"
"jge L_exit%=\t\n"
"vcvtph2ps ymm15,XMMWORD PTR [r10 + r11 + 16]\t\n"
"inc r14\t\n"
"vbroadcastss ymm4,DWORD PTR [r9+16]\t\n"
"vfmadd231ps ymm0,ymm14,ymm4\t\n"
"vbroadcastss ymm4,DWORD PTR [r9+20]\t\n"
"vfmadd231ps ymm1,ymm14,ymm4\t\n"
"vbroadcastss ymm4,DWORD PTR [r9+24]\t\n"
"vfmadd231ps ymm2,ymm14,ymm4\t\n"
"add r11, 32\t\n"
"vbroadcastss ymm4,DWORD PTR [r9+28]\t\n"
"vfmadd231ps ymm3,ymm14,ymm4\t\n"
"add r9,32\t\n"
"cmp r14, r8\t\n"
"jl loop_inner%=\t\n"

"L_exit%=:\t\n"
"add r10, rsi\t\n"

"cmp rdx, 1\t\n"
"je L_accum%=\t\n"
// Dump C
"vmovups YMMWORD PTR [r12 + 0], ymm0\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm1\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm2\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm3\t\n"
"add r12, r13\t\n"
"jmp L_done%=\t\n"


"L_accum%=:\t\n"
// Dump C with accumulate
"vbroadcastss ymm15,DWORD PTR [r15]\t\n"
"vfmadd231ps ymm0,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm0\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm1,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm1\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm2,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm2\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm3,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm3\t\n"
"add r12, r13\t\n"

"L_done%=:\t\n"

// next outer iteration
"add rcx, 32\t\n"
"mov r12, rcx\t\n"
"mov r9, rax\t\n"
"inc rbx\t\n"
"cmp rbx, rdi\t\n"
"jl loop_outter%=\t\n"
:
:
[gp] "rm" (gp)
: "r8", "r9", "r10", "r11", "r15",  "r13", "r14",
"rax", "rcx", "rdx", "rsi", "rdi", "rbx", "r12", "memory"
);
}
void __attribute__ ((noinline)) gemmkernel_5x1_AVX2_fA0fB0fC0(GemmParams *gp)
{
asm volatile
(
#if !defined(__clang__)
"mov r14, %[gp]\t\n"
#else
"mov %[gp], %%r14\t\n"
".intel_syntax noprefix\t\n"
#endif

// Copy parameters
// k
"mov r8, [r14 + 0]\t\n"
// A
"mov r9, [r14 + 8]\t\n"
// B
"mov r10, [r14 + 16]\t\n"
// beta
"mov r15, [r14 + 24]\t\n"
// accum
"mov rdx, [r14 + 32]\t\n"
// C
"mov r12, [r14 + 40]\t\n"
// ldc
"mov r13, [r14 + 48]\t\n"
// b_block_cols
"mov rdi, [r14 + 56]\t\n"
// b_block_size
"mov rsi, [r14 + 64]\t\n"
// Make copies of A and C
"mov rax, r9\t\n"
"mov rcx, r12\t\n"


"mov rbx, 0\t\n"
"loop_outter%=:\t\n"
"mov r14, 0\t\n"
"vxorps ymm0,ymm0,ymm0\t\n"
"vxorps ymm1,ymm1,ymm1\t\n"
"vxorps ymm2,ymm2,ymm2\t\n"
"vxorps ymm3,ymm3,ymm3\t\n"
"vxorps ymm4,ymm4,ymm4\t\n"

"vcvtph2ps ymm15, XMMWORD PTR [r10 + 0]\t\n"
"mov r11, 16\t\n"

"loop_inner%=:\t\n"

"vcvtph2ps ymm14,XMMWORD PTR [r10 + r11 + 0]\t\n"
"inc r14\t\n"
"vbroadcastss ymm5,DWORD PTR [r9+0]\t\n"
"vfmadd231ps ymm0,ymm15,ymm5\t\n"
"vbroadcastss ymm5,DWORD PTR [r9+4]\t\n"
"vfmadd231ps ymm1,ymm15,ymm5\t\n"
"vbroadcastss ymm5,DWORD PTR [r9+8]\t\n"
"vfmadd231ps ymm2,ymm15,ymm5\t\n"
"vbroadcastss ymm5,DWORD PTR [r9+12]\t\n"
"vfmadd231ps ymm3,ymm15,ymm5\t\n"
"vbroadcastss ymm5,DWORD PTR [r9+16]\t\n"
"vfmadd231ps ymm4,ymm15,ymm5\t\n"
"cmp r14, r8\t\n"
"jge L_exit%=\t\n"
"vcvtph2ps ymm15,XMMWORD PTR [r10 + r11 + 16]\t\n"
"inc r14\t\n"
"vbroadcastss ymm5,DWORD PTR [r9+20]\t\n"
"vfmadd231ps ymm0,ymm14,ymm5\t\n"
"vbroadcastss ymm5,DWORD PTR [r9+24]\t\n"
"vfmadd231ps ymm1,ymm14,ymm5\t\n"
"vbroadcastss ymm5,DWORD PTR [r9+28]\t\n"
"vfmadd231ps ymm2,ymm14,ymm5\t\n"
"add r11, 32\t\n"
"vbroadcastss ymm5,DWORD PTR [r9+32]\t\n"
"vfmadd231ps ymm3,ymm14,ymm5\t\n"
"vbroadcastss ymm5,DWORD PTR [r9+36]\t\n"
"vfmadd231ps ymm4,ymm14,ymm5\t\n"
"add r9,40\t\n"
"cmp r14, r8\t\n"
"jl loop_inner%=\t\n"

"L_exit%=:\t\n"
"add r10, rsi\t\n"

"cmp rdx, 1\t\n"
"je L_accum%=\t\n"
// Dump C
"vmovups YMMWORD PTR [r12 + 0], ymm0\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm1\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm2\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm3\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm4\t\n"
"add r12, r13\t\n"
"jmp L_done%=\t\n"


"L_accum%=:\t\n"
// Dump C with accumulate
"vbroadcastss ymm15,DWORD PTR [r15]\t\n"
"vfmadd231ps ymm0,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm0\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm1,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm1\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm2,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm2\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm3,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm3\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm4,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm4\t\n"
"add r12, r13\t\n"

"L_done%=:\t\n"

// next outer iteration
"add rcx, 32\t\n"
"mov r12, rcx\t\n"
"mov r9, rax\t\n"
"inc rbx\t\n"
"cmp rbx, rdi\t\n"
"jl loop_outter%=\t\n"
:
:
[gp] "rm" (gp)
: "r8", "r9", "r10", "r11", "r15",  "r13", "r14",
"rax", "rcx", "rdx", "rsi", "rdi", "rbx", "r12", "memory"
);
}
void __attribute__ ((noinline)) gemmkernel_6x1_AVX2_fA0fB0fC0(GemmParams *gp)
{
asm volatile
(
#if !defined(__clang__)
"mov r14, %[gp]\t\n"
#else
"mov %[gp], %%r14\t\n"
".intel_syntax noprefix\t\n"
#endif

// Copy parameters
// k
"mov r8, [r14 + 0]\t\n"
// A
"mov r9, [r14 + 8]\t\n"
// B
"mov r10, [r14 + 16]\t\n"
// beta
"mov r15, [r14 + 24]\t\n"
// accum
"mov rdx, [r14 + 32]\t\n"
// C
"mov r12, [r14 + 40]\t\n"
// ldc
"mov r13, [r14 + 48]\t\n"
// b_block_cols
"mov rdi, [r14 + 56]\t\n"
// b_block_size
"mov rsi, [r14 + 64]\t\n"
// Make copies of A and C
"mov rax, r9\t\n"
"mov rcx, r12\t\n"


"mov rbx, 0\t\n"
"loop_outter%=:\t\n"
"mov r14, 0\t\n"
"vxorps ymm0,ymm0,ymm0\t\n"
"vxorps ymm1,ymm1,ymm1\t\n"
"vxorps ymm2,ymm2,ymm2\t\n"
"vxorps ymm3,ymm3,ymm3\t\n"
"vxorps ymm4,ymm4,ymm4\t\n"
"vxorps ymm5,ymm5,ymm5\t\n"

"vcvtph2ps ymm15, XMMWORD PTR [r10 + 0]\t\n"
"mov r11, 16\t\n"

"loop_inner%=:\t\n"

"vcvtph2ps ymm14,XMMWORD PTR [r10 + r11 + 0]\t\n"
"inc r14\t\n"
"vbroadcastss ymm6,DWORD PTR [r9+0]\t\n"
"vfmadd231ps ymm0,ymm15,ymm6\t\n"
"vbroadcastss ymm6,DWORD PTR [r9+4]\t\n"
"vfmadd231ps ymm1,ymm15,ymm6\t\n"
"vbroadcastss ymm6,DWORD PTR [r9+8]\t\n"
"vfmadd231ps ymm2,ymm15,ymm6\t\n"
"vbroadcastss ymm6,DWORD PTR [r9+12]\t\n"
"vfmadd231ps ymm3,ymm15,ymm6\t\n"
"vbroadcastss ymm6,DWORD PTR [r9+16]\t\n"
"vfmadd231ps ymm4,ymm15,ymm6\t\n"
"vbroadcastss ymm6,DWORD PTR [r9+20]\t\n"
"vfmadd231ps ymm5,ymm15,ymm6\t\n"
"cmp r14, r8\t\n"
"jge L_exit%=\t\n"
"vcvtph2ps ymm15,XMMWORD PTR [r10 + r11 + 16]\t\n"
"inc r14\t\n"
"vbroadcastss ymm6,DWORD PTR [r9+24]\t\n"
"vfmadd231ps ymm0,ymm14,ymm6\t\n"
"vbroadcastss ymm6,DWORD PTR [r9+28]\t\n"
"vfmadd231ps ymm1,ymm14,ymm6\t\n"
"vbroadcastss ymm6,DWORD PTR [r9+32]\t\n"
"vfmadd231ps ymm2,ymm14,ymm6\t\n"
"vbroadcastss ymm6,DWORD PTR [r9+36]\t\n"
"vfmadd231ps ymm3,ymm14,ymm6\t\n"
"add r11, 32\t\n"
"vbroadcastss ymm6,DWORD PTR [r9+40]\t\n"
"vfmadd231ps ymm4,ymm14,ymm6\t\n"
"vbroadcastss ymm6,DWORD PTR [r9+44]\t\n"
"vfmadd231ps ymm5,ymm14,ymm6\t\n"
"add r9,48\t\n"
"cmp r14, r8\t\n"
"jl loop_inner%=\t\n"

"L_exit%=:\t\n"
"add r10, rsi\t\n"

"cmp rdx, 1\t\n"
"je L_accum%=\t\n"
// Dump C
"vmovups YMMWORD PTR [r12 + 0], ymm0\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm1\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm2\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm3\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm4\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm5\t\n"
"add r12, r13\t\n"
"jmp L_done%=\t\n"


"L_accum%=:\t\n"
// Dump C with accumulate
"vbroadcastss ymm15,DWORD PTR [r15]\t\n"
"vfmadd231ps ymm0,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm0\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm1,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm1\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm2,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm2\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm3,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm3\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm4,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm4\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm5,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm5\t\n"
"add r12, r13\t\n"

"L_done%=:\t\n"

// next outer iteration
"add rcx, 32\t\n"
"mov r12, rcx\t\n"
"mov r9, rax\t\n"
"inc rbx\t\n"
"cmp rbx, rdi\t\n"
"jl loop_outter%=\t\n"
:
:
[gp] "rm" (gp)
: "r8", "r9", "r10", "r11", "r15",  "r13", "r14",
"rax", "rcx", "rdx", "rsi", "rdi", "rbx", "r12", "memory"
);
}
void __attribute__ ((noinline)) gemmkernel_7x1_AVX2_fA0fB0fC0(GemmParams *gp)
{
asm volatile
(
#if !defined(__clang__)
"mov r14, %[gp]\t\n"
#else
"mov %[gp], %%r14\t\n"
".intel_syntax noprefix\t\n"
#endif

// Copy parameters
// k
"mov r8, [r14 + 0]\t\n"
// A
"mov r9, [r14 + 8]\t\n"
// B
"mov r10, [r14 + 16]\t\n"
// beta
"mov r15, [r14 + 24]\t\n"
// accum
"mov rdx, [r14 + 32]\t\n"
// C
"mov r12, [r14 + 40]\t\n"
// ldc
"mov r13, [r14 + 48]\t\n"
// b_block_cols
"mov rdi, [r14 + 56]\t\n"
// b_block_size
"mov rsi, [r14 + 64]\t\n"
// Make copies of A and C
"mov rax, r9\t\n"
"mov rcx, r12\t\n"


"mov rbx, 0\t\n"
"loop_outter%=:\t\n"
"mov r14, 0\t\n"
"vxorps ymm0,ymm0,ymm0\t\n"
"vxorps ymm1,ymm1,ymm1\t\n"
"vxorps ymm2,ymm2,ymm2\t\n"
"vxorps ymm3,ymm3,ymm3\t\n"
"vxorps ymm4,ymm4,ymm4\t\n"
"vxorps ymm5,ymm5,ymm5\t\n"
"vxorps ymm6,ymm6,ymm6\t\n"

"vcvtph2ps ymm15, XMMWORD PTR [r10 + 0]\t\n"
"mov r11, 16\t\n"

"loop_inner%=:\t\n"

"vcvtph2ps ymm14,XMMWORD PTR [r10 + r11 + 0]\t\n"
"inc r14\t\n"
"vbroadcastss ymm7,DWORD PTR [r9+0]\t\n"
"vfmadd231ps ymm0,ymm15,ymm7\t\n"
"vbroadcastss ymm7,DWORD PTR [r9+4]\t\n"
"vfmadd231ps ymm1,ymm15,ymm7\t\n"
"vbroadcastss ymm7,DWORD PTR [r9+8]\t\n"
"vfmadd231ps ymm2,ymm15,ymm7\t\n"
"vbroadcastss ymm7,DWORD PTR [r9+12]\t\n"
"vfmadd231ps ymm3,ymm15,ymm7\t\n"
"vbroadcastss ymm7,DWORD PTR [r9+16]\t\n"
"vfmadd231ps ymm4,ymm15,ymm7\t\n"
"vbroadcastss ymm7,DWORD PTR [r9+20]\t\n"
"vfmadd231ps ymm5,ymm15,ymm7\t\n"
"vbroadcastss ymm7,DWORD PTR [r9+24]\t\n"
"vfmadd231ps ymm6,ymm15,ymm7\t\n"
"cmp r14, r8\t\n"
"jge L_exit%=\t\n"
"vcvtph2ps ymm15,XMMWORD PTR [r10 + r11 + 16]\t\n"
"inc r14\t\n"
"vbroadcastss ymm7,DWORD PTR [r9+28]\t\n"
"vfmadd231ps ymm0,ymm14,ymm7\t\n"
"vbroadcastss ymm7,DWORD PTR [r9+32]\t\n"
"vfmadd231ps ymm1,ymm14,ymm7\t\n"
"vbroadcastss ymm7,DWORD PTR [r9+36]\t\n"
"vfmadd231ps ymm2,ymm14,ymm7\t\n"
"vbroadcastss ymm7,DWORD PTR [r9+40]\t\n"
"vfmadd231ps ymm3,ymm14,ymm7\t\n"
"add r11, 32\t\n"
"vbroadcastss ymm7,DWORD PTR [r9+44]\t\n"
"vfmadd231ps ymm4,ymm14,ymm7\t\n"
"vbroadcastss ymm7,DWORD PTR [r9+48]\t\n"
"vfmadd231ps ymm5,ymm14,ymm7\t\n"
"vbroadcastss ymm7,DWORD PTR [r9+52]\t\n"
"vfmadd231ps ymm6,ymm14,ymm7\t\n"
"add r9,56\t\n"
"cmp r14, r8\t\n"
"jl loop_inner%=\t\n"

"L_exit%=:\t\n"
"add r10, rsi\t\n"

"cmp rdx, 1\t\n"
"je L_accum%=\t\n"
// Dump C
"vmovups YMMWORD PTR [r12 + 0], ymm0\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm1\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm2\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm3\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm4\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm5\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm6\t\n"
"add r12, r13\t\n"
"jmp L_done%=\t\n"


"L_accum%=:\t\n"
// Dump C with accumulate
"vbroadcastss ymm15,DWORD PTR [r15]\t\n"
"vfmadd231ps ymm0,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm0\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm1,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm1\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm2,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm2\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm3,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm3\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm4,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm4\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm5,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm5\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm6,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm6\t\n"
"add r12, r13\t\n"

"L_done%=:\t\n"

// next outer iteration
"add rcx, 32\t\n"
"mov r12, rcx\t\n"
"mov r9, rax\t\n"
"inc rbx\t\n"
"cmp rbx, rdi\t\n"
"jl loop_outter%=\t\n"
:
:
[gp] "rm" (gp)
: "r8", "r9", "r10", "r11", "r15",  "r13", "r14",
"rax", "rcx", "rdx", "rsi", "rdi", "rbx", "r12", "memory"
);
}
void __attribute__ ((noinline)) gemmkernel_8x1_AVX2_fA0fB0fC0(GemmParams *gp)
{
asm volatile
(
#if !defined(__clang__)
"mov r14, %[gp]\t\n"
#else
"mov %[gp], %%r14\t\n"
".intel_syntax noprefix\t\n"
#endif

// Copy parameters
// k
"mov r8, [r14 + 0]\t\n"
// A
"mov r9, [r14 + 8]\t\n"
// B
"mov r10, [r14 + 16]\t\n"
// beta
"mov r15, [r14 + 24]\t\n"
// accum
"mov rdx, [r14 + 32]\t\n"
// C
"mov r12, [r14 + 40]\t\n"
// ldc
"mov r13, [r14 + 48]\t\n"
// b_block_cols
"mov rdi, [r14 + 56]\t\n"
// b_block_size
"mov rsi, [r14 + 64]\t\n"
// Make copies of A and C
"mov rax, r9\t\n"
"mov rcx, r12\t\n"


"mov rbx, 0\t\n"
"loop_outter%=:\t\n"
"mov r14, 0\t\n"
"vxorps ymm0,ymm0,ymm0\t\n"
"vxorps ymm1,ymm1,ymm1\t\n"
"vxorps ymm2,ymm2,ymm2\t\n"
"vxorps ymm3,ymm3,ymm3\t\n"
"vxorps ymm4,ymm4,ymm4\t\n"
"vxorps ymm5,ymm5,ymm5\t\n"
"vxorps ymm6,ymm6,ymm6\t\n"
"vxorps ymm7,ymm7,ymm7\t\n"

"vcvtph2ps ymm15, XMMWORD PTR [r10 + 0]\t\n"
"mov r11, 16\t\n"

"loop_inner%=:\t\n"

"vcvtph2ps ymm14,XMMWORD PTR [r10 + r11 + 0]\t\n"
"inc r14\t\n"
"vbroadcastss ymm8,DWORD PTR [r9+0]\t\n"
"vfmadd231ps ymm0,ymm15,ymm8\t\n"
"vbroadcastss ymm8,DWORD PTR [r9+4]\t\n"
"vfmadd231ps ymm1,ymm15,ymm8\t\n"
"vbroadcastss ymm8,DWORD PTR [r9+8]\t\n"
"vfmadd231ps ymm2,ymm15,ymm8\t\n"
"vbroadcastss ymm8,DWORD PTR [r9+12]\t\n"
"vfmadd231ps ymm3,ymm15,ymm8\t\n"
"vbroadcastss ymm8,DWORD PTR [r9+16]\t\n"
"vfmadd231ps ymm4,ymm15,ymm8\t\n"
"vbroadcastss ymm8,DWORD PTR [r9+20]\t\n"
"vfmadd231ps ymm5,ymm15,ymm8\t\n"
"vbroadcastss ymm8,DWORD PTR [r9+24]\t\n"
"vfmadd231ps ymm6,ymm15,ymm8\t\n"
"vbroadcastss ymm8,DWORD PTR [r9+28]\t\n"
"vfmadd231ps ymm7,ymm15,ymm8\t\n"
"cmp r14, r8\t\n"
"jge L_exit%=\t\n"
"vcvtph2ps ymm15,XMMWORD PTR [r10 + r11 + 16]\t\n"
"inc r14\t\n"
"vbroadcastss ymm8,DWORD PTR [r9+32]\t\n"
"vfmadd231ps ymm0,ymm14,ymm8\t\n"
"vbroadcastss ymm8,DWORD PTR [r9+36]\t\n"
"vfmadd231ps ymm1,ymm14,ymm8\t\n"
"vbroadcastss ymm8,DWORD PTR [r9+40]\t\n"
"vfmadd231ps ymm2,ymm14,ymm8\t\n"
"vbroadcastss ymm8,DWORD PTR [r9+44]\t\n"
"vfmadd231ps ymm3,ymm14,ymm8\t\n"
"vbroadcastss ymm8,DWORD PTR [r9+48]\t\n"
"vfmadd231ps ymm4,ymm14,ymm8\t\n"
"add r11, 32\t\n"
"vbroadcastss ymm8,DWORD PTR [r9+52]\t\n"
"vfmadd231ps ymm5,ymm14,ymm8\t\n"
"vbroadcastss ymm8,DWORD PTR [r9+56]\t\n"
"vfmadd231ps ymm6,ymm14,ymm8\t\n"
"vbroadcastss ymm8,DWORD PTR [r9+60]\t\n"
"vfmadd231ps ymm7,ymm14,ymm8\t\n"
"add r9,64\t\n"
"cmp r14, r8\t\n"
"jl loop_inner%=\t\n"

"L_exit%=:\t\n"
"add r10, rsi\t\n"

"cmp rdx, 1\t\n"
"je L_accum%=\t\n"
// Dump C
"vmovups YMMWORD PTR [r12 + 0], ymm0\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm1\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm2\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm3\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm4\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm5\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm6\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm7\t\n"
"add r12, r13\t\n"
"jmp L_done%=\t\n"


"L_accum%=:\t\n"
// Dump C with accumulate
"vbroadcastss ymm15,DWORD PTR [r15]\t\n"
"vfmadd231ps ymm0,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm0\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm1,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm1\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm2,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm2\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm3,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm3\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm4,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm4\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm5,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm5\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm6,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm6\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm7,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm7\t\n"
"add r12, r13\t\n"

"L_done%=:\t\n"

// next outer iteration
"add rcx, 32\t\n"
"mov r12, rcx\t\n"
"mov r9, rax\t\n"
"inc rbx\t\n"
"cmp rbx, rdi\t\n"
"jl loop_outter%=\t\n"
:
:
[gp] "rm" (gp)
: "r8", "r9", "r10", "r11", "r15",  "r13", "r14",
"rax", "rcx", "rdx", "rsi", "rdi", "rbx", "r12", "memory"
);
}
void __attribute__ ((noinline)) gemmkernel_9x1_AVX2_fA0fB0fC0(GemmParams *gp)
{
asm volatile
(
#if !defined(__clang__)
"mov r14, %[gp]\t\n"
#else
"mov %[gp], %%r14\t\n"
".intel_syntax noprefix\t\n"
#endif

// Copy parameters
// k
"mov r8, [r14 + 0]\t\n"
// A
"mov r9, [r14 + 8]\t\n"
// B
"mov r10, [r14 + 16]\t\n"
// beta
"mov r15, [r14 + 24]\t\n"
// accum
"mov rdx, [r14 + 32]\t\n"
// C
"mov r12, [r14 + 40]\t\n"
// ldc
"mov r13, [r14 + 48]\t\n"
// b_block_cols
"mov rdi, [r14 + 56]\t\n"
// b_block_size
"mov rsi, [r14 + 64]\t\n"
// Make copies of A and C
"mov rax, r9\t\n"
"mov rcx, r12\t\n"


"mov rbx, 0\t\n"
"loop_outter%=:\t\n"
"mov r14, 0\t\n"
"vxorps ymm0,ymm0,ymm0\t\n"
"vxorps ymm1,ymm1,ymm1\t\n"
"vxorps ymm2,ymm2,ymm2\t\n"
"vxorps ymm3,ymm3,ymm3\t\n"
"vxorps ymm4,ymm4,ymm4\t\n"
"vxorps ymm5,ymm5,ymm5\t\n"
"vxorps ymm6,ymm6,ymm6\t\n"
"vxorps ymm7,ymm7,ymm7\t\n"
"vxorps ymm8,ymm8,ymm8\t\n"

"vcvtph2ps ymm15, XMMWORD PTR [r10 + 0]\t\n"
"mov r11, 16\t\n"

"loop_inner%=:\t\n"

"vcvtph2ps ymm14,XMMWORD PTR [r10 + r11 + 0]\t\n"
"inc r14\t\n"
"vbroadcastss ymm9,DWORD PTR [r9+0]\t\n"
"vfmadd231ps ymm0,ymm15,ymm9\t\n"
"vbroadcastss ymm9,DWORD PTR [r9+4]\t\n"
"vfmadd231ps ymm1,ymm15,ymm9\t\n"
"vbroadcastss ymm9,DWORD PTR [r9+8]\t\n"
"vfmadd231ps ymm2,ymm15,ymm9\t\n"
"vbroadcastss ymm9,DWORD PTR [r9+12]\t\n"
"vfmadd231ps ymm3,ymm15,ymm9\t\n"
"vbroadcastss ymm9,DWORD PTR [r9+16]\t\n"
"vfmadd231ps ymm4,ymm15,ymm9\t\n"
"vbroadcastss ymm9,DWORD PTR [r9+20]\t\n"
"vfmadd231ps ymm5,ymm15,ymm9\t\n"
"vbroadcastss ymm9,DWORD PTR [r9+24]\t\n"
"vfmadd231ps ymm6,ymm15,ymm9\t\n"
"vbroadcastss ymm9,DWORD PTR [r9+28]\t\n"
"vfmadd231ps ymm7,ymm15,ymm9\t\n"
"vbroadcastss ymm9,DWORD PTR [r9+32]\t\n"
"vfmadd231ps ymm8,ymm15,ymm9\t\n"
"cmp r14, r8\t\n"
"jge L_exit%=\t\n"
"vcvtph2ps ymm15,XMMWORD PTR [r10 + r11 + 16]\t\n"
"inc r14\t\n"
"vbroadcastss ymm9,DWORD PTR [r9+36]\t\n"
"vfmadd231ps ymm0,ymm14,ymm9\t\n"
"vbroadcastss ymm9,DWORD PTR [r9+40]\t\n"
"vfmadd231ps ymm1,ymm14,ymm9\t\n"
"vbroadcastss ymm9,DWORD PTR [r9+44]\t\n"
"vfmadd231ps ymm2,ymm14,ymm9\t\n"
"vbroadcastss ymm9,DWORD PTR [r9+48]\t\n"
"vfmadd231ps ymm3,ymm14,ymm9\t\n"
"vbroadcastss ymm9,DWORD PTR [r9+52]\t\n"
"vfmadd231ps ymm4,ymm14,ymm9\t\n"
"add r11, 32\t\n"
"vbroadcastss ymm9,DWORD PTR [r9+56]\t\n"
"vfmadd231ps ymm5,ymm14,ymm9\t\n"
"vbroadcastss ymm9,DWORD PTR [r9+60]\t\n"
"vfmadd231ps ymm6,ymm14,ymm9\t\n"
"vbroadcastss ymm9,DWORD PTR [r9+64]\t\n"
"vfmadd231ps ymm7,ymm14,ymm9\t\n"
"vbroadcastss ymm9,DWORD PTR [r9+68]\t\n"
"vfmadd231ps ymm8,ymm14,ymm9\t\n"
"add r9,72\t\n"
"cmp r14, r8\t\n"
"jl loop_inner%=\t\n"

"L_exit%=:\t\n"
"add r10, rsi\t\n"

"cmp rdx, 1\t\n"
"je L_accum%=\t\n"
// Dump C
"vmovups YMMWORD PTR [r12 + 0], ymm0\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm1\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm2\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm3\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm4\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm5\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm6\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm7\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm8\t\n"
"add r12, r13\t\n"
"jmp L_done%=\t\n"


"L_accum%=:\t\n"
// Dump C with accumulate
"vbroadcastss ymm15,DWORD PTR [r15]\t\n"
"vfmadd231ps ymm0,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm0\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm1,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm1\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm2,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm2\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm3,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm3\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm4,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm4\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm5,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm5\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm6,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm6\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm7,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm7\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm8,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm8\t\n"
"add r12, r13\t\n"

"L_done%=:\t\n"

// next outer iteration
"add rcx, 32\t\n"
"mov r12, rcx\t\n"
"mov r9, rax\t\n"
"inc rbx\t\n"
"cmp rbx, rdi\t\n"
"jl loop_outter%=\t\n"
:
:
[gp] "rm" (gp)
: "r8", "r9", "r10", "r11", "r15",  "r13", "r14",
"rax", "rcx", "rdx", "rsi", "rdi", "rbx", "r12", "memory"
);
}
void __attribute__ ((noinline)) gemmkernel_10x1_AVX2_fA0fB0fC0(GemmParams *gp)
{
asm volatile
(
#if !defined(__clang__)
"mov r14, %[gp]\t\n"
#else
"mov %[gp], %%r14\t\n"
".intel_syntax noprefix\t\n"
#endif

// Copy parameters
// k
"mov r8, [r14 + 0]\t\n"
// A
"mov r9, [r14 + 8]\t\n"
// B
"mov r10, [r14 + 16]\t\n"
// beta
"mov r15, [r14 + 24]\t\n"
// accum
"mov rdx, [r14 + 32]\t\n"
// C
"mov r12, [r14 + 40]\t\n"
// ldc
"mov r13, [r14 + 48]\t\n"
// b_block_cols
"mov rdi, [r14 + 56]\t\n"
// b_block_size
"mov rsi, [r14 + 64]\t\n"
// Make copies of A and C
"mov rax, r9\t\n"
"mov rcx, r12\t\n"


"mov rbx, 0\t\n"
"loop_outter%=:\t\n"
"mov r14, 0\t\n"
"vxorps ymm0,ymm0,ymm0\t\n"
"vxorps ymm1,ymm1,ymm1\t\n"
"vxorps ymm2,ymm2,ymm2\t\n"
"vxorps ymm3,ymm3,ymm3\t\n"
"vxorps ymm4,ymm4,ymm4\t\n"
"vxorps ymm5,ymm5,ymm5\t\n"
"vxorps ymm6,ymm6,ymm6\t\n"
"vxorps ymm7,ymm7,ymm7\t\n"
"vxorps ymm8,ymm8,ymm8\t\n"
"vxorps ymm9,ymm9,ymm9\t\n"

"vcvtph2ps ymm15, XMMWORD PTR [r10 + 0]\t\n"
"mov r11, 16\t\n"

"loop_inner%=:\t\n"

"vcvtph2ps ymm14,XMMWORD PTR [r10 + r11 + 0]\t\n"
"inc r14\t\n"
"vbroadcastss ymm10,DWORD PTR [r9+0]\t\n"
"vfmadd231ps ymm0,ymm15,ymm10\t\n"
"vbroadcastss ymm10,DWORD PTR [r9+4]\t\n"
"vfmadd231ps ymm1,ymm15,ymm10\t\n"
"vbroadcastss ymm10,DWORD PTR [r9+8]\t\n"
"vfmadd231ps ymm2,ymm15,ymm10\t\n"
"vbroadcastss ymm10,DWORD PTR [r9+12]\t\n"
"vfmadd231ps ymm3,ymm15,ymm10\t\n"
"vbroadcastss ymm10,DWORD PTR [r9+16]\t\n"
"vfmadd231ps ymm4,ymm15,ymm10\t\n"
"vbroadcastss ymm10,DWORD PTR [r9+20]\t\n"
"vfmadd231ps ymm5,ymm15,ymm10\t\n"
"vbroadcastss ymm10,DWORD PTR [r9+24]\t\n"
"vfmadd231ps ymm6,ymm15,ymm10\t\n"
"vbroadcastss ymm10,DWORD PTR [r9+28]\t\n"
"vfmadd231ps ymm7,ymm15,ymm10\t\n"
"vbroadcastss ymm10,DWORD PTR [r9+32]\t\n"
"vfmadd231ps ymm8,ymm15,ymm10\t\n"
"vbroadcastss ymm10,DWORD PTR [r9+36]\t\n"
"vfmadd231ps ymm9,ymm15,ymm10\t\n"
"cmp r14, r8\t\n"
"jge L_exit%=\t\n"
"vcvtph2ps ymm15,XMMWORD PTR [r10 + r11 + 16]\t\n"
"inc r14\t\n"
"vbroadcastss ymm10,DWORD PTR [r9+40]\t\n"
"vfmadd231ps ymm0,ymm14,ymm10\t\n"
"vbroadcastss ymm10,DWORD PTR [r9+44]\t\n"
"vfmadd231ps ymm1,ymm14,ymm10\t\n"
"vbroadcastss ymm10,DWORD PTR [r9+48]\t\n"
"vfmadd231ps ymm2,ymm14,ymm10\t\n"
"vbroadcastss ymm10,DWORD PTR [r9+52]\t\n"
"vfmadd231ps ymm3,ymm14,ymm10\t\n"
"vbroadcastss ymm10,DWORD PTR [r9+56]\t\n"
"vfmadd231ps ymm4,ymm14,ymm10\t\n"
"vbroadcastss ymm10,DWORD PTR [r9+60]\t\n"
"vfmadd231ps ymm5,ymm14,ymm10\t\n"
"add r11, 32\t\n"
"vbroadcastss ymm10,DWORD PTR [r9+64]\t\n"
"vfmadd231ps ymm6,ymm14,ymm10\t\n"
"vbroadcastss ymm10,DWORD PTR [r9+68]\t\n"
"vfmadd231ps ymm7,ymm14,ymm10\t\n"
"vbroadcastss ymm10,DWORD PTR [r9+72]\t\n"
"vfmadd231ps ymm8,ymm14,ymm10\t\n"
"vbroadcastss ymm10,DWORD PTR [r9+76]\t\n"
"vfmadd231ps ymm9,ymm14,ymm10\t\n"
"add r9,80\t\n"
"cmp r14, r8\t\n"
"jl loop_inner%=\t\n"

"L_exit%=:\t\n"
"add r10, rsi\t\n"

"cmp rdx, 1\t\n"
"je L_accum%=\t\n"
// Dump C
"vmovups YMMWORD PTR [r12 + 0], ymm0\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm1\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm2\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm3\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm4\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm5\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm6\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm7\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm8\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm9\t\n"
"add r12, r13\t\n"
"jmp L_done%=\t\n"


"L_accum%=:\t\n"
// Dump C with accumulate
"vbroadcastss ymm15,DWORD PTR [r15]\t\n"
"vfmadd231ps ymm0,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm0\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm1,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm1\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm2,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm2\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm3,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm3\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm4,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm4\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm5,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm5\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm6,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm6\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm7,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm7\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm8,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm8\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm9,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm9\t\n"
"add r12, r13\t\n"

"L_done%=:\t\n"

// next outer iteration
"add rcx, 32\t\n"
"mov r12, rcx\t\n"
"mov r9, rax\t\n"
"inc rbx\t\n"
"cmp rbx, rdi\t\n"
"jl loop_outter%=\t\n"
:
:
[gp] "rm" (gp)
: "r8", "r9", "r10", "r11", "r15",  "r13", "r14",
"rax", "rcx", "rdx", "rsi", "rdi", "rbx", "r12", "memory"
);
}
void __attribute__ ((noinline)) gemmkernel_11x1_AVX2_fA0fB0fC0(GemmParams *gp)
{
asm volatile
(
#if !defined(__clang__)
"mov r14, %[gp]\t\n"
#else
"mov %[gp], %%r14\t\n"
".intel_syntax noprefix\t\n"
#endif

// Copy parameters
// k
"mov r8, [r14 + 0]\t\n"
// A
"mov r9, [r14 + 8]\t\n"
// B
"mov r10, [r14 + 16]\t\n"
// beta
"mov r15, [r14 + 24]\t\n"
// accum
"mov rdx, [r14 + 32]\t\n"
// C
"mov r12, [r14 + 40]\t\n"
// ldc
"mov r13, [r14 + 48]\t\n"
// b_block_cols
"mov rdi, [r14 + 56]\t\n"
// b_block_size
"mov rsi, [r14 + 64]\t\n"
// Make copies of A and C
"mov rax, r9\t\n"
"mov rcx, r12\t\n"


"mov rbx, 0\t\n"
"loop_outter%=:\t\n"
"mov r14, 0\t\n"
"vxorps ymm0,ymm0,ymm0\t\n"
"vxorps ymm1,ymm1,ymm1\t\n"
"vxorps ymm2,ymm2,ymm2\t\n"
"vxorps ymm3,ymm3,ymm3\t\n"
"vxorps ymm4,ymm4,ymm4\t\n"
"vxorps ymm5,ymm5,ymm5\t\n"
"vxorps ymm6,ymm6,ymm6\t\n"
"vxorps ymm7,ymm7,ymm7\t\n"
"vxorps ymm8,ymm8,ymm8\t\n"
"vxorps ymm9,ymm9,ymm9\t\n"
"vxorps ymm10,ymm10,ymm10\t\n"

"vcvtph2ps ymm15, XMMWORD PTR [r10 + 0]\t\n"
"mov r11, 16\t\n"

"loop_inner%=:\t\n"

"vcvtph2ps ymm14,XMMWORD PTR [r10 + r11 + 0]\t\n"
"inc r14\t\n"
"vbroadcastss ymm11,DWORD PTR [r9+0]\t\n"
"vfmadd231ps ymm0,ymm15,ymm11\t\n"
"vbroadcastss ymm11,DWORD PTR [r9+4]\t\n"
"vfmadd231ps ymm1,ymm15,ymm11\t\n"
"vbroadcastss ymm11,DWORD PTR [r9+8]\t\n"
"vfmadd231ps ymm2,ymm15,ymm11\t\n"
"vbroadcastss ymm11,DWORD PTR [r9+12]\t\n"
"vfmadd231ps ymm3,ymm15,ymm11\t\n"
"vbroadcastss ymm11,DWORD PTR [r9+16]\t\n"
"vfmadd231ps ymm4,ymm15,ymm11\t\n"
"vbroadcastss ymm11,DWORD PTR [r9+20]\t\n"
"vfmadd231ps ymm5,ymm15,ymm11\t\n"
"vbroadcastss ymm11,DWORD PTR [r9+24]\t\n"
"vfmadd231ps ymm6,ymm15,ymm11\t\n"
"vbroadcastss ymm11,DWORD PTR [r9+28]\t\n"
"vfmadd231ps ymm7,ymm15,ymm11\t\n"
"vbroadcastss ymm11,DWORD PTR [r9+32]\t\n"
"vfmadd231ps ymm8,ymm15,ymm11\t\n"
"vbroadcastss ymm11,DWORD PTR [r9+36]\t\n"
"vfmadd231ps ymm9,ymm15,ymm11\t\n"
"vbroadcastss ymm11,DWORD PTR [r9+40]\t\n"
"vfmadd231ps ymm10,ymm15,ymm11\t\n"
"cmp r14, r8\t\n"
"jge L_exit%=\t\n"
"vcvtph2ps ymm15,XMMWORD PTR [r10 + r11 + 16]\t\n"
"inc r14\t\n"
"vbroadcastss ymm11,DWORD PTR [r9+44]\t\n"
"vfmadd231ps ymm0,ymm14,ymm11\t\n"
"vbroadcastss ymm11,DWORD PTR [r9+48]\t\n"
"vfmadd231ps ymm1,ymm14,ymm11\t\n"
"vbroadcastss ymm11,DWORD PTR [r9+52]\t\n"
"vfmadd231ps ymm2,ymm14,ymm11\t\n"
"vbroadcastss ymm11,DWORD PTR [r9+56]\t\n"
"vfmadd231ps ymm3,ymm14,ymm11\t\n"
"vbroadcastss ymm11,DWORD PTR [r9+60]\t\n"
"vfmadd231ps ymm4,ymm14,ymm11\t\n"
"vbroadcastss ymm11,DWORD PTR [r9+64]\t\n"
"vfmadd231ps ymm5,ymm14,ymm11\t\n"
"add r11, 32\t\n"
"vbroadcastss ymm11,DWORD PTR [r9+68]\t\n"
"vfmadd231ps ymm6,ymm14,ymm11\t\n"
"vbroadcastss ymm11,DWORD PTR [r9+72]\t\n"
"vfmadd231ps ymm7,ymm14,ymm11\t\n"
"vbroadcastss ymm11,DWORD PTR [r9+76]\t\n"
"vfmadd231ps ymm8,ymm14,ymm11\t\n"
"vbroadcastss ymm11,DWORD PTR [r9+80]\t\n"
"vfmadd231ps ymm9,ymm14,ymm11\t\n"
"vbroadcastss ymm11,DWORD PTR [r9+84]\t\n"
"vfmadd231ps ymm10,ymm14,ymm11\t\n"
"add r9,88\t\n"
"cmp r14, r8\t\n"
"jl loop_inner%=\t\n"

"L_exit%=:\t\n"
"add r10, rsi\t\n"

"cmp rdx, 1\t\n"
"je L_accum%=\t\n"
// Dump C
"vmovups YMMWORD PTR [r12 + 0], ymm0\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm1\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm2\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm3\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm4\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm5\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm6\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm7\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm8\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm9\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm10\t\n"
"add r12, r13\t\n"
"jmp L_done%=\t\n"


"L_accum%=:\t\n"
// Dump C with accumulate
"vbroadcastss ymm15,DWORD PTR [r15]\t\n"
"vfmadd231ps ymm0,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm0\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm1,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm1\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm2,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm2\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm3,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm3\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm4,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm4\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm5,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm5\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm6,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm6\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm7,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm7\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm8,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm8\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm9,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm9\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm10,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm10\t\n"
"add r12, r13\t\n"

"L_done%=:\t\n"

// next outer iteration
"add rcx, 32\t\n"
"mov r12, rcx\t\n"
"mov r9, rax\t\n"
"inc rbx\t\n"
"cmp rbx, rdi\t\n"
"jl loop_outter%=\t\n"
:
:
[gp] "rm" (gp)
: "r8", "r9", "r10", "r11", "r15",  "r13", "r14",
"rax", "rcx", "rdx", "rsi", "rdi", "rbx", "r12", "memory"
);
}
void __attribute__ ((noinline)) gemmkernel_12x1_AVX2_fA0fB0fC0(GemmParams *gp)
{
asm volatile
(
#if !defined(__clang__)
"mov r14, %[gp]\t\n"
#else
"mov %[gp], %%r14\t\n"
".intel_syntax noprefix\t\n"
#endif

// Copy parameters
// k
"mov r8, [r14 + 0]\t\n"
// A
"mov r9, [r14 + 8]\t\n"
// B
"mov r10, [r14 + 16]\t\n"
// beta
"mov r15, [r14 + 24]\t\n"
// accum
"mov rdx, [r14 + 32]\t\n"
// C
"mov r12, [r14 + 40]\t\n"
// ldc
"mov r13, [r14 + 48]\t\n"
// b_block_cols
"mov rdi, [r14 + 56]\t\n"
// b_block_size
"mov rsi, [r14 + 64]\t\n"
// Make copies of A and C
"mov rax, r9\t\n"
"mov rcx, r12\t\n"


"mov rbx, 0\t\n"
"loop_outter%=:\t\n"
"mov r14, 0\t\n"
"vxorps ymm0,ymm0,ymm0\t\n"
"vxorps ymm1,ymm1,ymm1\t\n"
"vxorps ymm2,ymm2,ymm2\t\n"
"vxorps ymm3,ymm3,ymm3\t\n"
"vxorps ymm4,ymm4,ymm4\t\n"
"vxorps ymm5,ymm5,ymm5\t\n"
"vxorps ymm6,ymm6,ymm6\t\n"
"vxorps ymm7,ymm7,ymm7\t\n"
"vxorps ymm8,ymm8,ymm8\t\n"
"vxorps ymm9,ymm9,ymm9\t\n"
"vxorps ymm10,ymm10,ymm10\t\n"
"vxorps ymm11,ymm11,ymm11\t\n"

"vcvtph2ps ymm15, XMMWORD PTR [r10 + 0]\t\n"
"mov r11, 16\t\n"

"loop_inner%=:\t\n"

"vcvtph2ps ymm14,XMMWORD PTR [r10 + r11 + 0]\t\n"
"inc r14\t\n"
"vbroadcastss ymm12,DWORD PTR [r9+0]\t\n"
"vfmadd231ps ymm0,ymm15,ymm12\t\n"
"vbroadcastss ymm12,DWORD PTR [r9+4]\t\n"
"vfmadd231ps ymm1,ymm15,ymm12\t\n"
"vbroadcastss ymm12,DWORD PTR [r9+8]\t\n"
"vfmadd231ps ymm2,ymm15,ymm12\t\n"
"vbroadcastss ymm12,DWORD PTR [r9+12]\t\n"
"vfmadd231ps ymm3,ymm15,ymm12\t\n"
"vbroadcastss ymm12,DWORD PTR [r9+16]\t\n"
"vfmadd231ps ymm4,ymm15,ymm12\t\n"
"vbroadcastss ymm12,DWORD PTR [r9+20]\t\n"
"vfmadd231ps ymm5,ymm15,ymm12\t\n"
"vbroadcastss ymm12,DWORD PTR [r9+24]\t\n"
"vfmadd231ps ymm6,ymm15,ymm12\t\n"
"vbroadcastss ymm12,DWORD PTR [r9+28]\t\n"
"vfmadd231ps ymm7,ymm15,ymm12\t\n"
"vbroadcastss ymm12,DWORD PTR [r9+32]\t\n"
"vfmadd231ps ymm8,ymm15,ymm12\t\n"
"vbroadcastss ymm12,DWORD PTR [r9+36]\t\n"
"vfmadd231ps ymm9,ymm15,ymm12\t\n"
"vbroadcastss ymm12,DWORD PTR [r9+40]\t\n"
"vfmadd231ps ymm10,ymm15,ymm12\t\n"
"vbroadcastss ymm12,DWORD PTR [r9+44]\t\n"
"vfmadd231ps ymm11,ymm15,ymm12\t\n"
"cmp r14, r8\t\n"
"jge L_exit%=\t\n"
"vcvtph2ps ymm15,XMMWORD PTR [r10 + r11 + 16]\t\n"
"inc r14\t\n"
"vbroadcastss ymm12,DWORD PTR [r9+48]\t\n"
"vfmadd231ps ymm0,ymm14,ymm12\t\n"
"vbroadcastss ymm12,DWORD PTR [r9+52]\t\n"
"vfmadd231ps ymm1,ymm14,ymm12\t\n"
"vbroadcastss ymm12,DWORD PTR [r9+56]\t\n"
"vfmadd231ps ymm2,ymm14,ymm12\t\n"
"vbroadcastss ymm12,DWORD PTR [r9+60]\t\n"
"vfmadd231ps ymm3,ymm14,ymm12\t\n"
"vbroadcastss ymm12,DWORD PTR [r9+64]\t\n"
"vfmadd231ps ymm4,ymm14,ymm12\t\n"
"vbroadcastss ymm12,DWORD PTR [r9+68]\t\n"
"vfmadd231ps ymm5,ymm14,ymm12\t\n"
"vbroadcastss ymm12,DWORD PTR [r9+72]\t\n"
"vfmadd231ps ymm6,ymm14,ymm12\t\n"
"add r11, 32\t\n"
"vbroadcastss ymm12,DWORD PTR [r9+76]\t\n"
"vfmadd231ps ymm7,ymm14,ymm12\t\n"
"vbroadcastss ymm12,DWORD PTR [r9+80]\t\n"
"vfmadd231ps ymm8,ymm14,ymm12\t\n"
"vbroadcastss ymm12,DWORD PTR [r9+84]\t\n"
"vfmadd231ps ymm9,ymm14,ymm12\t\n"
"vbroadcastss ymm12,DWORD PTR [r9+88]\t\n"
"vfmadd231ps ymm10,ymm14,ymm12\t\n"
"vbroadcastss ymm12,DWORD PTR [r9+92]\t\n"
"vfmadd231ps ymm11,ymm14,ymm12\t\n"
"add r9,96\t\n"
"cmp r14, r8\t\n"
"jl loop_inner%=\t\n"

"L_exit%=:\t\n"
"add r10, rsi\t\n"

"cmp rdx, 1\t\n"
"je L_accum%=\t\n"
// Dump C
"vmovups YMMWORD PTR [r12 + 0], ymm0\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm1\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm2\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm3\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm4\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm5\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm6\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm7\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm8\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm9\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm10\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm11\t\n"
"add r12, r13\t\n"
"jmp L_done%=\t\n"


"L_accum%=:\t\n"
// Dump C with accumulate
"vbroadcastss ymm15,DWORD PTR [r15]\t\n"
"vfmadd231ps ymm0,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm0\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm1,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm1\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm2,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm2\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm3,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm3\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm4,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm4\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm5,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm5\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm6,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm6\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm7,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm7\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm8,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm8\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm9,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm9\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm10,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm10\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm11,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm11\t\n"
"add r12, r13\t\n"

"L_done%=:\t\n"

// next outer iteration
"add rcx, 32\t\n"
"mov r12, rcx\t\n"
"mov r9, rax\t\n"
"inc rbx\t\n"
"cmp rbx, rdi\t\n"
"jl loop_outter%=\t\n"
:
:
[gp] "rm" (gp)
: "r8", "r9", "r10", "r11", "r15",  "r13", "r14",
"rax", "rcx", "rdx", "rsi", "rdi", "rbx", "r12", "memory"
);
}
void __attribute__ ((noinline)) gemmkernel_13x1_AVX2_fA0fB0fC0(GemmParams *gp)
{
asm volatile
(
#if !defined(__clang__)
"mov r14, %[gp]\t\n"
#else
"mov %[gp], %%r14\t\n"
".intel_syntax noprefix\t\n"
#endif

// Copy parameters
// k
"mov r8, [r14 + 0]\t\n"
// A
"mov r9, [r14 + 8]\t\n"
// B
"mov r10, [r14 + 16]\t\n"
// beta
"mov r15, [r14 + 24]\t\n"
// accum
"mov rdx, [r14 + 32]\t\n"
// C
"mov r12, [r14 + 40]\t\n"
// ldc
"mov r13, [r14 + 48]\t\n"
// b_block_cols
"mov rdi, [r14 + 56]\t\n"
// b_block_size
"mov rsi, [r14 + 64]\t\n"
// Make copies of A and C
"mov rax, r9\t\n"
"mov rcx, r12\t\n"


"mov rbx, 0\t\n"
"loop_outter%=:\t\n"
"mov r14, 0\t\n"
"vxorps ymm0,ymm0,ymm0\t\n"
"vxorps ymm1,ymm1,ymm1\t\n"
"vxorps ymm2,ymm2,ymm2\t\n"
"vxorps ymm3,ymm3,ymm3\t\n"
"vxorps ymm4,ymm4,ymm4\t\n"
"vxorps ymm5,ymm5,ymm5\t\n"
"vxorps ymm6,ymm6,ymm6\t\n"
"vxorps ymm7,ymm7,ymm7\t\n"
"vxorps ymm8,ymm8,ymm8\t\n"
"vxorps ymm9,ymm9,ymm9\t\n"
"vxorps ymm10,ymm10,ymm10\t\n"
"vxorps ymm11,ymm11,ymm11\t\n"
"vxorps ymm12,ymm12,ymm12\t\n"

"vcvtph2ps ymm15, XMMWORD PTR [r10 + 0]\t\n"
"mov r11, 16\t\n"

"loop_inner%=:\t\n"

"vcvtph2ps ymm14,XMMWORD PTR [r10 + r11 + 0]\t\n"
"inc r14\t\n"
"vbroadcastss ymm13,DWORD PTR [r9+0]\t\n"
"vfmadd231ps ymm0,ymm15,ymm13\t\n"
"vbroadcastss ymm13,DWORD PTR [r9+4]\t\n"
"vfmadd231ps ymm1,ymm15,ymm13\t\n"
"vbroadcastss ymm13,DWORD PTR [r9+8]\t\n"
"vfmadd231ps ymm2,ymm15,ymm13\t\n"
"vbroadcastss ymm13,DWORD PTR [r9+12]\t\n"
"vfmadd231ps ymm3,ymm15,ymm13\t\n"
"vbroadcastss ymm13,DWORD PTR [r9+16]\t\n"
"vfmadd231ps ymm4,ymm15,ymm13\t\n"
"vbroadcastss ymm13,DWORD PTR [r9+20]\t\n"
"vfmadd231ps ymm5,ymm15,ymm13\t\n"
"vbroadcastss ymm13,DWORD PTR [r9+24]\t\n"
"vfmadd231ps ymm6,ymm15,ymm13\t\n"
"vbroadcastss ymm13,DWORD PTR [r9+28]\t\n"
"vfmadd231ps ymm7,ymm15,ymm13\t\n"
"vbroadcastss ymm13,DWORD PTR [r9+32]\t\n"
"vfmadd231ps ymm8,ymm15,ymm13\t\n"
"vbroadcastss ymm13,DWORD PTR [r9+36]\t\n"
"vfmadd231ps ymm9,ymm15,ymm13\t\n"
"vbroadcastss ymm13,DWORD PTR [r9+40]\t\n"
"vfmadd231ps ymm10,ymm15,ymm13\t\n"
"vbroadcastss ymm13,DWORD PTR [r9+44]\t\n"
"vfmadd231ps ymm11,ymm15,ymm13\t\n"
"vbroadcastss ymm13,DWORD PTR [r9+48]\t\n"
"vfmadd231ps ymm12,ymm15,ymm13\t\n"
"cmp r14, r8\t\n"
"jge L_exit%=\t\n"
"vcvtph2ps ymm15,XMMWORD PTR [r10 + r11 + 16]\t\n"
"inc r14\t\n"
"vbroadcastss ymm13,DWORD PTR [r9+52]\t\n"
"vfmadd231ps ymm0,ymm14,ymm13\t\n"
"vbroadcastss ymm13,DWORD PTR [r9+56]\t\n"
"vfmadd231ps ymm1,ymm14,ymm13\t\n"
"vbroadcastss ymm13,DWORD PTR [r9+60]\t\n"
"vfmadd231ps ymm2,ymm14,ymm13\t\n"
"vbroadcastss ymm13,DWORD PTR [r9+64]\t\n"
"vfmadd231ps ymm3,ymm14,ymm13\t\n"
"vbroadcastss ymm13,DWORD PTR [r9+68]\t\n"
"vfmadd231ps ymm4,ymm14,ymm13\t\n"
"vbroadcastss ymm13,DWORD PTR [r9+72]\t\n"
"vfmadd231ps ymm5,ymm14,ymm13\t\n"
"vbroadcastss ymm13,DWORD PTR [r9+76]\t\n"
"vfmadd231ps ymm6,ymm14,ymm13\t\n"
"add r11, 32\t\n"
"vbroadcastss ymm13,DWORD PTR [r9+80]\t\n"
"vfmadd231ps ymm7,ymm14,ymm13\t\n"
"vbroadcastss ymm13,DWORD PTR [r9+84]\t\n"
"vfmadd231ps ymm8,ymm14,ymm13\t\n"
"vbroadcastss ymm13,DWORD PTR [r9+88]\t\n"
"vfmadd231ps ymm9,ymm14,ymm13\t\n"
"vbroadcastss ymm13,DWORD PTR [r9+92]\t\n"
"vfmadd231ps ymm10,ymm14,ymm13\t\n"
"vbroadcastss ymm13,DWORD PTR [r9+96]\t\n"
"vfmadd231ps ymm11,ymm14,ymm13\t\n"
"vbroadcastss ymm13,DWORD PTR [r9+100]\t\n"
"vfmadd231ps ymm12,ymm14,ymm13\t\n"
"add r9,104\t\n"
"cmp r14, r8\t\n"
"jl loop_inner%=\t\n"

"L_exit%=:\t\n"
"add r10, rsi\t\n"

"cmp rdx, 1\t\n"
"je L_accum%=\t\n"
// Dump C
"vmovups YMMWORD PTR [r12 + 0], ymm0\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm1\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm2\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm3\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm4\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm5\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm6\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm7\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm8\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm9\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm10\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm11\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm12\t\n"
"add r12, r13\t\n"
"jmp L_done%=\t\n"


"L_accum%=:\t\n"
// Dump C with accumulate
"vbroadcastss ymm15,DWORD PTR [r15]\t\n"
"vfmadd231ps ymm0,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm0\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm1,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm1\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm2,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm2\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm3,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm3\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm4,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm4\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm5,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm5\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm6,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm6\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm7,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm7\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm8,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm8\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm9,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm9\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm10,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm10\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm11,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm11\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm12,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm12\t\n"
"add r12, r13\t\n"

"L_done%=:\t\n"

// next outer iteration
"add rcx, 32\t\n"
"mov r12, rcx\t\n"
"mov r9, rax\t\n"
"inc rbx\t\n"
"cmp rbx, rdi\t\n"
"jl loop_outter%=\t\n"
:
:
[gp] "rm" (gp)
: "r8", "r9", "r10", "r11", "r15",  "r13", "r14",
"rax", "rcx", "rdx", "rsi", "rdi", "rbx", "r12", "memory"
);
}
void __attribute__ ((noinline)) gemmkernel_14x1_AVX2_fA0fB0fC0(GemmParams *gp)
{
asm volatile
(
#if !defined(__clang__)
"mov r14, %[gp]\t\n"
#else
"mov %[gp], %%r14\t\n"
".intel_syntax noprefix\t\n"
#endif

// Copy parameters
// k
"mov r8, [r14 + 0]\t\n"
// A
"mov r9, [r14 + 8]\t\n"
// B
"mov r10, [r14 + 16]\t\n"
// beta
"mov r15, [r14 + 24]\t\n"
// accum
"mov rdx, [r14 + 32]\t\n"
// C
"mov r12, [r14 + 40]\t\n"
// ldc
"mov r13, [r14 + 48]\t\n"
// b_block_cols
"mov rdi, [r14 + 56]\t\n"
// b_block_size
"mov rsi, [r14 + 64]\t\n"
// Make copies of A and C
"mov rax, r9\t\n"
"mov rcx, r12\t\n"


"mov rbx, 0\t\n"
"loop_outter%=:\t\n"
"mov r14, 0\t\n"
"vxorps ymm0,ymm0,ymm0\t\n"
"vxorps ymm1,ymm1,ymm1\t\n"
"vxorps ymm2,ymm2,ymm2\t\n"
"vxorps ymm3,ymm3,ymm3\t\n"
"vxorps ymm4,ymm4,ymm4\t\n"
"vxorps ymm5,ymm5,ymm5\t\n"
"vxorps ymm6,ymm6,ymm6\t\n"
"vxorps ymm7,ymm7,ymm7\t\n"
"vxorps ymm8,ymm8,ymm8\t\n"
"vxorps ymm9,ymm9,ymm9\t\n"
"vxorps ymm10,ymm10,ymm10\t\n"
"vxorps ymm11,ymm11,ymm11\t\n"
"vxorps ymm12,ymm12,ymm12\t\n"
"vxorps ymm13,ymm13,ymm13\t\n"

"mov r11, 0\t\n"

"loop_inner%=:\t\n"

"vcvtph2ps ymm15,XMMWORD PTR [r10 + r11]\t\n"
"vbroadcastss ymm14,DWORD PTR [r9+0]\t\n"
"vfmadd231ps ymm0,ymm15,ymm14\t\n"
"vbroadcastss ymm14,DWORD PTR [r9+4]\t\n"
"vfmadd231ps ymm1,ymm15,ymm14\t\n"
"vbroadcastss ymm14,DWORD PTR [r9+8]\t\n"
"vfmadd231ps ymm2,ymm15,ymm14\t\n"
"vbroadcastss ymm14,DWORD PTR [r9+12]\t\n"
"vfmadd231ps ymm3,ymm15,ymm14\t\n"
"vbroadcastss ymm14,DWORD PTR [r9+16]\t\n"
"vfmadd231ps ymm4,ymm15,ymm14\t\n"
"vbroadcastss ymm14,DWORD PTR [r9+20]\t\n"
"vfmadd231ps ymm5,ymm15,ymm14\t\n"
"vbroadcastss ymm14,DWORD PTR [r9+24]\t\n"
"vfmadd231ps ymm6,ymm15,ymm14\t\n"
"vbroadcastss ymm14,DWORD PTR [r9+28]\t\n"
"vfmadd231ps ymm7,ymm15,ymm14\t\n"
"vbroadcastss ymm14,DWORD PTR [r9+32]\t\n"
"vfmadd231ps ymm8,ymm15,ymm14\t\n"
"vbroadcastss ymm14,DWORD PTR [r9+36]\t\n"
"vfmadd231ps ymm9,ymm15,ymm14\t\n"
"vbroadcastss ymm14,DWORD PTR [r9+40]\t\n"
"vfmadd231ps ymm10,ymm15,ymm14\t\n"
"vbroadcastss ymm14,DWORD PTR [r9+44]\t\n"
"vfmadd231ps ymm11,ymm15,ymm14\t\n"
"vbroadcastss ymm14,DWORD PTR [r9+48]\t\n"
"vfmadd231ps ymm12,ymm15,ymm14\t\n"
"vbroadcastss ymm14,DWORD PTR [r9+52]\t\n"
"vfmadd231ps ymm13,ymm15,ymm14\t\n"
"add r9,56\t\n"
"add r11, 16\t\n"
"inc r14\t\n"
"cmp r14, r8\t\n"
"jl loop_inner%=\t\n"
"add r10, rsi\t\n"

"cmp rdx, 1\t\n"
"je L_accum%=\t\n"
// Dump C
"vmovups YMMWORD PTR [r12 + 0], ymm0\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm1\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm2\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm3\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm4\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm5\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm6\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm7\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm8\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm9\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm10\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm11\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm12\t\n"
"add r12, r13\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm13\t\n"
"add r12, r13\t\n"
"jmp L_done%=\t\n"


"L_accum%=:\t\n"
// Dump C with accumulate
"vbroadcastss ymm15,DWORD PTR [r15]\t\n"
"vfmadd231ps ymm0,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm0\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm1,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm1\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm2,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm2\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm3,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm3\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm4,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm4\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm5,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm5\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm6,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm6\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm7,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm7\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm8,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm8\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm9,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm9\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm10,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm10\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm11,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm11\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm12,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm12\t\n"
"add r12, r13\t\n"
"vfmadd231ps ymm13,ymm15,YMMWORD PTR [r12 + 0]\t\n"
"vmovups YMMWORD PTR [r12 + 0], ymm13\t\n"
"add r12, r13\t\n"

"L_done%=:\t\n"

// next outer iteration
"add rcx, 32\t\n"
"mov r12, rcx\t\n"
"mov r9, rax\t\n"
"inc rbx\t\n"
"cmp rbx, rdi\t\n"
"jl loop_outter%=\t\n"
:
:
[gp] "rm" (gp)
: "r8", "r9", "r10", "r11", "r15",  "r13", "r14",
"rax", "rcx", "rdx", "rsi", "rdi", "rbx", "r12", "memory"
);
}

} // namespace fbgemm
