/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "FbgemmFP16UKernelsAvx2.h"
#include <immintrin.h>

namespace fbgemm {

void NOINLINE_ATTR gemmkernel_1x2_AVX2_fA0fB0fC0(GemmParams* gp) {
      char* r14 = (char*)gp;      //"mov r14, %[gp]\t\n"

      // Copy parameters
      // k
      uint64_t    r8  = *(uint64_t   *)((char*)r14 + 0 );      //"mov r8, [r14 + 0]\t\n"
      // A
      float*      r9  = *(float*     *)((char*)r14 + 8 );      //"mov r9, [r14 + 8]\t\n"
      // B
      const fp16* r10 = *(const fp16**)((char*)r14 + 16);      //"mov r10, [r14 + 16]\t\n"
      // beta
      float*      r15 = *(float*     *)((char*)r14 + 24);      //"mov r15, [r14 + 24]\t\n"
      // accum
      uint64_t    rdx = *(uint64_t   *)((char*)r14 + 32);      //"mov rdx, [r14 + 32]\t\n"
      // C
      float*      r12 = *(float*     *)((char*)r14 + 40);      //"mov r12, [r14 + 40]\t\n"
      // ldc
      uint64_t    r13 = *(uint64_t   *)((char*)r14 + 48);      //"mov r13, [r14 + 48]\t\n"
      // b_block_cols
      uint64_t    rdi = *(uint64_t   *)((char*)r14 + 56);      //"mov rdi, [r14 + 56]\t\n"
      // b_block_size
      uint64_t    rsi = *(uint64_t   *)((char*)r14 + 64);      //"mov rsi, [r14 + 64]\t\n"
      // Make copies of A and C
      float* rax = r9;      //"mov rax, r9\t\n"
      float* rcx = r12;      //"mov rcx, r12\t\n"

      uint64_t rbx = 0;      //"mov rbx, 0\t\n"
      for (; rbx < rdi; ++rbx) {      //"inc rbx; cmp rbx, rdi; jl loop_outter%=\t\n"
      //       //"loop_outter%=:\t\n"
        uint64_t r14_i = 0;      //"mov r14, 0\t\n"
        __m256 ymm0 = _mm256_setzero_ps();      //"vxorps ymm0,ymm0,ymm0\t\n"
        __m256 ymm1 = _mm256_setzero_ps();      //"vxorps ymm1,ymm1,ymm1\t\n"

        for (; r14_i < r8; ++r14_i) {      //"inc r14; cmp r14, r8; jl loop_inner%=\t\n"
        // loop_inner%=:      //"\t\n"
          auto fp16mem0 = _mm_load_si128((__m128i*)((char*)r10 + 0));      //"vcvtph2ps ymm3,XMMWORD PTR [r10 + 0]\t\n"
          auto ymm3 = _mm256_cvtph_ps(fp16mem0);      //"vcvtph2ps ymm3,XMMWORD PTR [r10 + 0]\t\n"
          auto fp16mem16 = _mm_load_si128((__m128i*)((char*)r10 + 16));      //"vcvtph2ps ymm4,XMMWORD PTR [r10 + 16]\t\n"
          auto ymm4 = _mm256_cvtph_ps(fp16mem16);      //"vcvtph2ps ymm4,XMMWORD PTR [r10 + 16]\t\n"
          auto ymm2 = _mm256_broadcast_ss((float*)((char*)r9 + 0));      //"vbroadcastss ymm2,DWORD PTR [r9+0]\t\n"
          ymm0 = _mm256_fmadd_ps(ymm2, ymm3, ymm0);      //"vfmadd231ps ymm0,ymm3,ymm2\t\n"
          ymm1 = _mm256_fmadd_ps(ymm2, ymm4, ymm1);      //"vfmadd231ps ymm1,ymm4,ymm2\t\n"
          r9 = (float*)((char*)r9 + 4);      //"add r9,4\t\n"
          r10 = (fp16*)((char*)r10 + 32);      //"add r10,32\t\n"
        }      //"inc r14; cmp r14, r8; jl loop_outter%=\t\n"

        if(rdx != 1) {      //"cmp rdx, 1; je L_accum%=\t\n"
          // Dump C
          _mm256_storeu_ps((float*)((char*)r12 + 0), ymm0);      //"vmovups YMMWORD PTR [r12 + 0], ymm0\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 32), ymm1);      //"vmovups YMMWORD PTR [r12 + 32], ymm1\t\n"
        } else {      //"jmp L_done%=\t\n"
          // Dump C with accumulate
          auto ymm15 = _mm256_broadcast_ss((float*)r15);      //"vbroadcastss ymm15,DWORD PTR [r15]\t\n"
          auto r12_0 = _mm256_load_ps((float*)((char*)r12 + 0));      //"vfmadd231ps ymm0,ymm15,YMMWORD PTR [r12 + 0]\t\n"
          ymm0 = _mm256_fmadd_ps(r12_0, ymm15, ymm0);      //"vfmadd231ps ymm0,ymm15,YMMWORD PTR [r12 + 0]\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 0), ymm0);      //"vmovups YMMWORD PTR [r12 + 0], ymm0\t\n"
          auto r12_32 = _mm256_load_ps((float*)((char*)r12 + 32));      //"vfmadd231ps ymm1,ymm15,YMMWORD PTR [r12 + 32]\t\n"
          ymm1 = _mm256_fmadd_ps(r12_32, ymm15, ymm1);      //"vfmadd231ps ymm1,ymm15,YMMWORD PTR [r12 + 32]\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 32), ymm1);      //"vmovups YMMWORD PTR [r12 + 32], ymm1\t\n"
        }      //"L_done%=:\t\n"

        // next outer iteration
        rcx = (float*)((char*)rcx + 64);      //"add rcx, 64\t\n"
        r12 = rcx;      //"mov r12, rcx\t\n"
        r9 = rax;      //"mov r9, rax\t\n"
      }      //"inc rbx; cmp rbx, rdi; jl loop_outter%=\t\n"
}

void NOINLINE_ATTR gemmkernel_2x2_AVX2_fA0fB0fC0(GemmParams* gp) {
      char* r14 = (char*)gp;      //"mov r14, %[gp]\t\n"

      // Copy parameters
      // k
      uint64_t    r8  = *(uint64_t   *)((char*)r14 + 0 );      //"mov r8, [r14 + 0]\t\n"
      // A
      float*      r9  = *(float*     *)((char*)r14 + 8 );      //"mov r9, [r14 + 8]\t\n"
      // B
      const fp16* r10 = *(const fp16**)((char*)r14 + 16);      //"mov r10, [r14 + 16]\t\n"
      // beta
      float*      r15 = *(float*     *)((char*)r14 + 24);      //"mov r15, [r14 + 24]\t\n"
      // accum
      uint64_t    rdx = *(uint64_t   *)((char*)r14 + 32);      //"mov rdx, [r14 + 32]\t\n"
      // C
      float*      r12 = *(float*     *)((char*)r14 + 40);      //"mov r12, [r14 + 40]\t\n"
      // ldc
      uint64_t    r13 = *(uint64_t   *)((char*)r14 + 48);      //"mov r13, [r14 + 48]\t\n"
      // b_block_cols
      uint64_t    rdi = *(uint64_t   *)((char*)r14 + 56);      //"mov rdi, [r14 + 56]\t\n"
      // b_block_size
      uint64_t    rsi = *(uint64_t   *)((char*)r14 + 64);      //"mov rsi, [r14 + 64]\t\n"
      // Make copies of A and C
      float* rax = r9;      //"mov rax, r9\t\n"
      float* rcx = r12;      //"mov rcx, r12\t\n"

      uint64_t rbx = 0;      //"mov rbx, 0\t\n"
      for (; rbx < rdi; ++rbx) {      //"inc rbx; cmp rbx, rdi; jl loop_outter%=\t\n"
      //       //"loop_outter%=:\t\n"
        uint64_t r14_i = 0;      //"mov r14, 0\t\n"
        __m256 ymm0 = _mm256_setzero_ps();      //"vxorps ymm0,ymm0,ymm0\t\n"
        __m256 ymm1 = _mm256_setzero_ps();      //"vxorps ymm1,ymm1,ymm1\t\n"
        __m256 ymm2 = _mm256_setzero_ps();      //"vxorps ymm2,ymm2,ymm2\t\n"
        __m256 ymm3 = _mm256_setzero_ps();      //"vxorps ymm3,ymm3,ymm3\t\n"

        for (; r14_i < r8; ++r14_i) {      //"inc r14; cmp r14, r8; jl loop_inner%=\t\n"
        // loop_inner%=:      //"\t\n"
          auto fp16mem0 = _mm_load_si128((__m128i*)((char*)r10 + 0));      //"vcvtph2ps ymm5,XMMWORD PTR [r10 + 0]\t\n"
          auto ymm5 = _mm256_cvtph_ps(fp16mem0);      //"vcvtph2ps ymm5,XMMWORD PTR [r10 + 0]\t\n"
          auto fp16mem16 = _mm_load_si128((__m128i*)((char*)r10 + 16));      //"vcvtph2ps ymm6,XMMWORD PTR [r10 + 16]\t\n"
          auto ymm6 = _mm256_cvtph_ps(fp16mem16);      //"vcvtph2ps ymm6,XMMWORD PTR [r10 + 16]\t\n"
          auto ymm4 = _mm256_broadcast_ss((float*)((char*)r9 + 0));      //"vbroadcastss ymm4,DWORD PTR [r9+0]\t\n"
          ymm0 = _mm256_fmadd_ps(ymm4, ymm5, ymm0);      //"vfmadd231ps ymm0,ymm5,ymm4\t\n"
          ymm1 = _mm256_fmadd_ps(ymm4, ymm6, ymm1);      //"vfmadd231ps ymm1,ymm6,ymm4\t\n"
          ymm4 = _mm256_broadcast_ss((float*)((char*)r9 + 4));      //"vbroadcastss ymm4,DWORD PTR [r9+4]\t\n"
          ymm2 = _mm256_fmadd_ps(ymm4, ymm5, ymm2);      //"vfmadd231ps ymm2,ymm5,ymm4\t\n"
          ymm3 = _mm256_fmadd_ps(ymm4, ymm6, ymm3);      //"vfmadd231ps ymm3,ymm6,ymm4\t\n"
          r9 = (float*)((char*)r9 + 8);      //"add r9,8\t\n"
          r10 = (fp16*)((char*)r10 + 32);      //"add r10,32\t\n"
        }      //"inc r14; cmp r14, r8; jl loop_outter%=\t\n"

        if(rdx != 1) {      //"cmp rdx, 1; je L_accum%=\t\n"
          // Dump C
          _mm256_storeu_ps((float*)((char*)r12 + 0), ymm0);      //"vmovups YMMWORD PTR [r12 + 0], ymm0\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 32), ymm1);      //"vmovups YMMWORD PTR [r12 + 32], ymm1\t\n"
          r12 = (float*)((char*)r12 + r13);      //"add r12, r13\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 0), ymm2);      //"vmovups YMMWORD PTR [r12 + 0], ymm2\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 32), ymm3);      //"vmovups YMMWORD PTR [r12 + 32], ymm3\t\n"
        } else {      //"jmp L_done%=\t\n"
          // Dump C with accumulate
          auto ymm15 = _mm256_broadcast_ss((float*)r15);      //"vbroadcastss ymm15,DWORD PTR [r15]\t\n"
          auto r12_0 = _mm256_load_ps((float*)((char*)r12 + 0));      //"vfmadd231ps ymm0,ymm15,YMMWORD PTR [r12 + 0]\t\n"
          ymm0 = _mm256_fmadd_ps(r12_0, ymm15, ymm0);      //"vfmadd231ps ymm0,ymm15,YMMWORD PTR [r12 + 0]\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 0), ymm0);      //"vmovups YMMWORD PTR [r12 + 0], ymm0\t\n"
          auto r12_32 = _mm256_load_ps((float*)((char*)r12 + 32));      //"vfmadd231ps ymm1,ymm15,YMMWORD PTR [r12 + 32]\t\n"
          ymm1 = _mm256_fmadd_ps(r12_32, ymm15, ymm1);      //"vfmadd231ps ymm1,ymm15,YMMWORD PTR [r12 + 32]\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 32), ymm1);      //"vmovups YMMWORD PTR [r12 + 32], ymm1\t\n"
          r12 = (float*)((char*)r12 + r13);      //"add r12, r13\t\n"
          r12_0 = _mm256_load_ps((float*)((char*)r12 + 0));      //"vfmadd231ps ymm2,ymm15,YMMWORD PTR [r12 + 0]\t\n"
          ymm2 = _mm256_fmadd_ps(r12_0, ymm15, ymm2);      //"vfmadd231ps ymm2,ymm15,YMMWORD PTR [r12 + 0]\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 0), ymm2);      //"vmovups YMMWORD PTR [r12 + 0], ymm2\t\n"
          r12_32 = _mm256_load_ps((float*)((char*)r12 + 32));      //"vfmadd231ps ymm3,ymm15,YMMWORD PTR [r12 + 32]\t\n"
          ymm3 = _mm256_fmadd_ps(r12_32, ymm15, ymm3);      //"vfmadd231ps ymm3,ymm15,YMMWORD PTR [r12 + 32]\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 32), ymm3);      //"vmovups YMMWORD PTR [r12 + 32], ymm3\t\n"
        }      //"L_done%=:\t\n"

        // next outer iteration
        rcx = (float*)((char*)rcx + 64);      //"add rcx, 64\t\n"
        r12 = rcx;      //"mov r12, rcx\t\n"
        r9 = rax;      //"mov r9, rax\t\n"
      }      //"inc rbx; cmp rbx, rdi; jl loop_outter%=\t\n"
}

void NOINLINE_ATTR gemmkernel_3x2_AVX2_fA0fB0fC0(GemmParams* gp) {
      char* r14 = (char*)gp;      //"mov r14, %[gp]\t\n"

      // Copy parameters
      // k
      uint64_t    r8  = *(uint64_t   *)((char*)r14 + 0 );      //"mov r8, [r14 + 0]\t\n"
      // A
      float*      r9  = *(float*     *)((char*)r14 + 8 );      //"mov r9, [r14 + 8]\t\n"
      // B
      const fp16* r10 = *(const fp16**)((char*)r14 + 16);      //"mov r10, [r14 + 16]\t\n"
      // beta
      float*      r15 = *(float*     *)((char*)r14 + 24);      //"mov r15, [r14 + 24]\t\n"
      // accum
      uint64_t    rdx = *(uint64_t   *)((char*)r14 + 32);      //"mov rdx, [r14 + 32]\t\n"
      // C
      float*      r12 = *(float*     *)((char*)r14 + 40);      //"mov r12, [r14 + 40]\t\n"
      // ldc
      uint64_t    r13 = *(uint64_t   *)((char*)r14 + 48);      //"mov r13, [r14 + 48]\t\n"
      // b_block_cols
      uint64_t    rdi = *(uint64_t   *)((char*)r14 + 56);      //"mov rdi, [r14 + 56]\t\n"
      // b_block_size
      uint64_t    rsi = *(uint64_t   *)((char*)r14 + 64);      //"mov rsi, [r14 + 64]\t\n"
      // Make copies of A and C
      float* rax = r9;      //"mov rax, r9\t\n"
      float* rcx = r12;      //"mov rcx, r12\t\n"

      uint64_t rbx = 0;      //"mov rbx, 0\t\n"
      for (; rbx < rdi; ++rbx) {      //"inc rbx; cmp rbx, rdi; jl loop_outter%=\t\n"
      //       //"loop_outter%=:\t\n"
        uint64_t r14_i = 0;      //"mov r14, 0\t\n"
        __m256 ymm0 = _mm256_setzero_ps();      //"vxorps ymm0,ymm0,ymm0\t\n"
        __m256 ymm1 = _mm256_setzero_ps();      //"vxorps ymm1,ymm1,ymm1\t\n"
        __m256 ymm2 = _mm256_setzero_ps();      //"vxorps ymm2,ymm2,ymm2\t\n"
        __m256 ymm3 = _mm256_setzero_ps();      //"vxorps ymm3,ymm3,ymm3\t\n"
        __m256 ymm4 = _mm256_setzero_ps();      //"vxorps ymm4,ymm4,ymm4\t\n"
        __m256 ymm5 = _mm256_setzero_ps();      //"vxorps ymm5,ymm5,ymm5\t\n"

        for (; r14_i < r8; ++r14_i) {      //"inc r14; cmp r14, r8; jl loop_inner%=\t\n"
        // loop_inner%=:      //"\t\n"
          auto fp16mem0 = _mm_load_si128((__m128i*)((char*)r10 + 0));      //"vcvtph2ps ymm7,XMMWORD PTR [r10 + 0]\t\n"
          auto ymm7 = _mm256_cvtph_ps(fp16mem0);      //"vcvtph2ps ymm7,XMMWORD PTR [r10 + 0]\t\n"
          auto fp16mem16 = _mm_load_si128((__m128i*)((char*)r10 + 16));      //"vcvtph2ps ymm8,XMMWORD PTR [r10 + 16]\t\n"
          auto ymm8 = _mm256_cvtph_ps(fp16mem16);      //"vcvtph2ps ymm8,XMMWORD PTR [r10 + 16]\t\n"
          auto ymm6 = _mm256_broadcast_ss((float*)((char*)r9 + 0));      //"vbroadcastss ymm6,DWORD PTR [r9+0]\t\n"
          ymm0 = _mm256_fmadd_ps(ymm6, ymm7, ymm0);      //"vfmadd231ps ymm0,ymm7,ymm6\t\n"
          ymm1 = _mm256_fmadd_ps(ymm6, ymm8, ymm1);      //"vfmadd231ps ymm1,ymm8,ymm6\t\n"
          ymm6 = _mm256_broadcast_ss((float*)((char*)r9 + 4));      //"vbroadcastss ymm6,DWORD PTR [r9+4]\t\n"
          ymm2 = _mm256_fmadd_ps(ymm6, ymm7, ymm2);      //"vfmadd231ps ymm2,ymm7,ymm6\t\n"
          ymm3 = _mm256_fmadd_ps(ymm6, ymm8, ymm3);      //"vfmadd231ps ymm3,ymm8,ymm6\t\n"
          ymm6 = _mm256_broadcast_ss((float*)((char*)r9 + 8));      //"vbroadcastss ymm6,DWORD PTR [r9+8]\t\n"
          ymm4 = _mm256_fmadd_ps(ymm6, ymm7, ymm4);      //"vfmadd231ps ymm4,ymm7,ymm6\t\n"
          ymm5 = _mm256_fmadd_ps(ymm6, ymm8, ymm5);      //"vfmadd231ps ymm5,ymm8,ymm6\t\n"
          r9 = (float*)((char*)r9 + 12);      //"add r9,12\t\n"
          r10 = (fp16*)((char*)r10 + 32);      //"add r10,32\t\n"
        }      //"inc r14; cmp r14, r8; jl loop_outter%=\t\n"

        if(rdx != 1) {      //"cmp rdx, 1; je L_accum%=\t\n"
          // Dump C
          _mm256_storeu_ps((float*)((char*)r12 + 0), ymm0);      //"vmovups YMMWORD PTR [r12 + 0], ymm0\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 32), ymm1);      //"vmovups YMMWORD PTR [r12 + 32], ymm1\t\n"
          r12 = (float*)((char*)r12 + r13);      //"add r12, r13\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 0), ymm2);      //"vmovups YMMWORD PTR [r12 + 0], ymm2\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 32), ymm3);      //"vmovups YMMWORD PTR [r12 + 32], ymm3\t\n"
          r12 = (float*)((char*)r12 + r13);      //"add r12, r13\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 0), ymm4);      //"vmovups YMMWORD PTR [r12 + 0], ymm4\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 32), ymm5);      //"vmovups YMMWORD PTR [r12 + 32], ymm5\t\n"
        } else {      //"jmp L_done%=\t\n"
          // Dump C with accumulate
          auto ymm15 = _mm256_broadcast_ss((float*)r15);      //"vbroadcastss ymm15,DWORD PTR [r15]\t\n"
          auto r12_0 = _mm256_load_ps((float*)((char*)r12 + 0));      //"vfmadd231ps ymm0,ymm15,YMMWORD PTR [r12 + 0]\t\n"
          ymm0 = _mm256_fmadd_ps(r12_0, ymm15, ymm0);      //"vfmadd231ps ymm0,ymm15,YMMWORD PTR [r12 + 0]\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 0), ymm0);      //"vmovups YMMWORD PTR [r12 + 0], ymm0\t\n"
          auto r12_32 = _mm256_load_ps((float*)((char*)r12 + 32));      //"vfmadd231ps ymm1,ymm15,YMMWORD PTR [r12 + 32]\t\n"
          ymm1 = _mm256_fmadd_ps(r12_32, ymm15, ymm1);      //"vfmadd231ps ymm1,ymm15,YMMWORD PTR [r12 + 32]\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 32), ymm1);      //"vmovups YMMWORD PTR [r12 + 32], ymm1\t\n"
          r12 = (float*)((char*)r12 + r13);      //"add r12, r13\t\n"
          r12_0 = _mm256_load_ps((float*)((char*)r12 + 0));      //"vfmadd231ps ymm2,ymm15,YMMWORD PTR [r12 + 0]\t\n"
          ymm2 = _mm256_fmadd_ps(r12_0, ymm15, ymm2);      //"vfmadd231ps ymm2,ymm15,YMMWORD PTR [r12 + 0]\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 0), ymm2);      //"vmovups YMMWORD PTR [r12 + 0], ymm2\t\n"
          r12_32 = _mm256_load_ps((float*)((char*)r12 + 32));      //"vfmadd231ps ymm3,ymm15,YMMWORD PTR [r12 + 32]\t\n"
          ymm3 = _mm256_fmadd_ps(r12_32, ymm15, ymm3);      //"vfmadd231ps ymm3,ymm15,YMMWORD PTR [r12 + 32]\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 32), ymm3);      //"vmovups YMMWORD PTR [r12 + 32], ymm3\t\n"
          r12 = (float*)((char*)r12 + r13);      //"add r12, r13\t\n"
          r12_0 = _mm256_load_ps((float*)((char*)r12 + 0));      //"vfmadd231ps ymm4,ymm15,YMMWORD PTR [r12 + 0]\t\n"
          ymm4 = _mm256_fmadd_ps(r12_0, ymm15, ymm4);      //"vfmadd231ps ymm4,ymm15,YMMWORD PTR [r12 + 0]\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 0), ymm4);      //"vmovups YMMWORD PTR [r12 + 0], ymm4\t\n"
          r12_32 = _mm256_load_ps((float*)((char*)r12 + 32));      //"vfmadd231ps ymm5,ymm15,YMMWORD PTR [r12 + 32]\t\n"
          ymm5 = _mm256_fmadd_ps(r12_32, ymm15, ymm5);      //"vfmadd231ps ymm5,ymm15,YMMWORD PTR [r12 + 32]\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 32), ymm5);      //"vmovups YMMWORD PTR [r12 + 32], ymm5\t\n"
        }      //"L_done%=:\t\n"

        // next outer iteration
        rcx = (float*)((char*)rcx + 64);      //"add rcx, 64\t\n"
        r12 = rcx;      //"mov r12, rcx\t\n"
        r9 = rax;      //"mov r9, rax\t\n"
      }      //"inc rbx; cmp rbx, rdi; jl loop_outter%=\t\n"
}

void NOINLINE_ATTR gemmkernel_4x2_AVX2_fA0fB0fC0(GemmParams* gp) {
      char* r14 = (char*)gp;      //"mov r14, %[gp]\t\n"

      // Copy parameters
      // k
      uint64_t    r8  = *(uint64_t   *)((char*)r14 + 0 );      //"mov r8, [r14 + 0]\t\n"
      // A
      float*      r9  = *(float*     *)((char*)r14 + 8 );      //"mov r9, [r14 + 8]\t\n"
      // B
      const fp16* r10 = *(const fp16**)((char*)r14 + 16);      //"mov r10, [r14 + 16]\t\n"
      // beta
      float*      r15 = *(float*     *)((char*)r14 + 24);      //"mov r15, [r14 + 24]\t\n"
      // accum
      uint64_t    rdx = *(uint64_t   *)((char*)r14 + 32);      //"mov rdx, [r14 + 32]\t\n"
      // C
      float*      r12 = *(float*     *)((char*)r14 + 40);      //"mov r12, [r14 + 40]\t\n"
      // ldc
      uint64_t    r13 = *(uint64_t   *)((char*)r14 + 48);      //"mov r13, [r14 + 48]\t\n"
      // b_block_cols
      uint64_t    rdi = *(uint64_t   *)((char*)r14 + 56);      //"mov rdi, [r14 + 56]\t\n"
      // b_block_size
      uint64_t    rsi = *(uint64_t   *)((char*)r14 + 64);      //"mov rsi, [r14 + 64]\t\n"
      // Make copies of A and C
      float* rax = r9;      //"mov rax, r9\t\n"
      float* rcx = r12;      //"mov rcx, r12\t\n"

      uint64_t rbx = 0;      //"mov rbx, 0\t\n"
      for (; rbx < rdi; ++rbx) {      //"inc rbx; cmp rbx, rdi; jl loop_outter%=\t\n"
      //       //"loop_outter%=:\t\n"
        uint64_t r14_i = 0;      //"mov r14, 0\t\n"
        __m256 ymm0 = _mm256_setzero_ps();      //"vxorps ymm0,ymm0,ymm0\t\n"
        __m256 ymm1 = _mm256_setzero_ps();      //"vxorps ymm1,ymm1,ymm1\t\n"
        __m256 ymm2 = _mm256_setzero_ps();      //"vxorps ymm2,ymm2,ymm2\t\n"
        __m256 ymm3 = _mm256_setzero_ps();      //"vxorps ymm3,ymm3,ymm3\t\n"
        __m256 ymm4 = _mm256_setzero_ps();      //"vxorps ymm4,ymm4,ymm4\t\n"
        __m256 ymm5 = _mm256_setzero_ps();      //"vxorps ymm5,ymm5,ymm5\t\n"
        __m256 ymm6 = _mm256_setzero_ps();      //"vxorps ymm6,ymm6,ymm6\t\n"
        __m256 ymm7 = _mm256_setzero_ps();      //"vxorps ymm7,ymm7,ymm7\t\n"

        for (; r14_i < r8; ++r14_i) {      //"inc r14; cmp r14, r8; jl loop_inner%=\t\n"
        // loop_inner%=:      //"\t\n"
          auto fp16mem0 = _mm_load_si128((__m128i*)((char*)r10 + 0));      //"vcvtph2ps ymm9,XMMWORD PTR [r10 + 0]\t\n"
          auto ymm9 = _mm256_cvtph_ps(fp16mem0);      //"vcvtph2ps ymm9,XMMWORD PTR [r10 + 0]\t\n"
          auto fp16mem16 = _mm_load_si128((__m128i*)((char*)r10 + 16));      //"vcvtph2ps ymm10,XMMWORD PTR [r10 + 16]\t\n"
          auto ymm10 = _mm256_cvtph_ps(fp16mem16);      //"vcvtph2ps ymm10,XMMWORD PTR [r10 + 16]\t\n"
          auto ymm8 = _mm256_broadcast_ss((float*)((char*)r9 + 0));      //"vbroadcastss ymm8,DWORD PTR [r9+0]\t\n"
          ymm0 = _mm256_fmadd_ps(ymm8, ymm9, ymm0);      //"vfmadd231ps ymm0,ymm9,ymm8\t\n"
          ymm1 = _mm256_fmadd_ps(ymm8, ymm10, ymm1);      //"vfmadd231ps ymm1,ymm10,ymm8\t\n"
          ymm8 = _mm256_broadcast_ss((float*)((char*)r9 + 4));      //"vbroadcastss ymm8,DWORD PTR [r9+4]\t\n"
          ymm2 = _mm256_fmadd_ps(ymm8, ymm9, ymm2);      //"vfmadd231ps ymm2,ymm9,ymm8\t\n"
          ymm3 = _mm256_fmadd_ps(ymm8, ymm10, ymm3);      //"vfmadd231ps ymm3,ymm10,ymm8\t\n"
          ymm8 = _mm256_broadcast_ss((float*)((char*)r9 + 8));      //"vbroadcastss ymm8,DWORD PTR [r9+8]\t\n"
          ymm4 = _mm256_fmadd_ps(ymm8, ymm9, ymm4);      //"vfmadd231ps ymm4,ymm9,ymm8\t\n"
          ymm5 = _mm256_fmadd_ps(ymm8, ymm10, ymm5);      //"vfmadd231ps ymm5,ymm10,ymm8\t\n"
          ymm8 = _mm256_broadcast_ss((float*)((char*)r9 + 12));      //"vbroadcastss ymm8,DWORD PTR [r9+12]\t\n"
          ymm6 = _mm256_fmadd_ps(ymm8, ymm9, ymm6);      //"vfmadd231ps ymm6,ymm9,ymm8\t\n"
          ymm7 = _mm256_fmadd_ps(ymm8, ymm10, ymm7);      //"vfmadd231ps ymm7,ymm10,ymm8\t\n"
          r9 = (float*)((char*)r9 + 16);      //"add r9,16\t\n"
          r10 = (fp16*)((char*)r10 + 32);      //"add r10,32\t\n"
        }      //"inc r14; cmp r14, r8; jl loop_outter%=\t\n"

        if(rdx != 1) {      //"cmp rdx, 1; je L_accum%=\t\n"
          // Dump C
          _mm256_storeu_ps((float*)((char*)r12 + 0), ymm0);      //"vmovups YMMWORD PTR [r12 + 0], ymm0\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 32), ymm1);      //"vmovups YMMWORD PTR [r12 + 32], ymm1\t\n"
          r12 = (float*)((char*)r12 + r13);      //"add r12, r13\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 0), ymm2);      //"vmovups YMMWORD PTR [r12 + 0], ymm2\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 32), ymm3);      //"vmovups YMMWORD PTR [r12 + 32], ymm3\t\n"
          r12 = (float*)((char*)r12 + r13);      //"add r12, r13\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 0), ymm4);      //"vmovups YMMWORD PTR [r12 + 0], ymm4\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 32), ymm5);      //"vmovups YMMWORD PTR [r12 + 32], ymm5\t\n"
          r12 = (float*)((char*)r12 + r13);      //"add r12, r13\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 0), ymm6);      //"vmovups YMMWORD PTR [r12 + 0], ymm6\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 32), ymm7);      //"vmovups YMMWORD PTR [r12 + 32], ymm7\t\n"
        } else {      //"jmp L_done%=\t\n"
          // Dump C with accumulate
          auto ymm15 = _mm256_broadcast_ss((float*)r15);      //"vbroadcastss ymm15,DWORD PTR [r15]\t\n"
          auto r12_0 = _mm256_load_ps((float*)((char*)r12 + 0));      //"vfmadd231ps ymm0,ymm15,YMMWORD PTR [r12 + 0]\t\n"
          ymm0 = _mm256_fmadd_ps(r12_0, ymm15, ymm0);      //"vfmadd231ps ymm0,ymm15,YMMWORD PTR [r12 + 0]\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 0), ymm0);      //"vmovups YMMWORD PTR [r12 + 0], ymm0\t\n"
          auto r12_32 = _mm256_load_ps((float*)((char*)r12 + 32));      //"vfmadd231ps ymm1,ymm15,YMMWORD PTR [r12 + 32]\t\n"
          ymm1 = _mm256_fmadd_ps(r12_32, ymm15, ymm1);      //"vfmadd231ps ymm1,ymm15,YMMWORD PTR [r12 + 32]\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 32), ymm1);      //"vmovups YMMWORD PTR [r12 + 32], ymm1\t\n"
          r12 = (float*)((char*)r12 + r13);      //"add r12, r13\t\n"
          r12_0 = _mm256_load_ps((float*)((char*)r12 + 0));      //"vfmadd231ps ymm2,ymm15,YMMWORD PTR [r12 + 0]\t\n"
          ymm2 = _mm256_fmadd_ps(r12_0, ymm15, ymm2);      //"vfmadd231ps ymm2,ymm15,YMMWORD PTR [r12 + 0]\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 0), ymm2);      //"vmovups YMMWORD PTR [r12 + 0], ymm2\t\n"
          r12_32 = _mm256_load_ps((float*)((char*)r12 + 32));      //"vfmadd231ps ymm3,ymm15,YMMWORD PTR [r12 + 32]\t\n"
          ymm3 = _mm256_fmadd_ps(r12_32, ymm15, ymm3);      //"vfmadd231ps ymm3,ymm15,YMMWORD PTR [r12 + 32]\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 32), ymm3);      //"vmovups YMMWORD PTR [r12 + 32], ymm3\t\n"
          r12 = (float*)((char*)r12 + r13);      //"add r12, r13\t\n"
          r12_0 = _mm256_load_ps((float*)((char*)r12 + 0));      //"vfmadd231ps ymm4,ymm15,YMMWORD PTR [r12 + 0]\t\n"
          ymm4 = _mm256_fmadd_ps(r12_0, ymm15, ymm4);      //"vfmadd231ps ymm4,ymm15,YMMWORD PTR [r12 + 0]\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 0), ymm4);      //"vmovups YMMWORD PTR [r12 + 0], ymm4\t\n"
          r12_32 = _mm256_load_ps((float*)((char*)r12 + 32));      //"vfmadd231ps ymm5,ymm15,YMMWORD PTR [r12 + 32]\t\n"
          ymm5 = _mm256_fmadd_ps(r12_32, ymm15, ymm5);      //"vfmadd231ps ymm5,ymm15,YMMWORD PTR [r12 + 32]\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 32), ymm5);      //"vmovups YMMWORD PTR [r12 + 32], ymm5\t\n"
          r12 = (float*)((char*)r12 + r13);      //"add r12, r13\t\n"
          r12_0 = _mm256_load_ps((float*)((char*)r12 + 0));      //"vfmadd231ps ymm6,ymm15,YMMWORD PTR [r12 + 0]\t\n"
          ymm6 = _mm256_fmadd_ps(r12_0, ymm15, ymm6);      //"vfmadd231ps ymm6,ymm15,YMMWORD PTR [r12 + 0]\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 0), ymm6);      //"vmovups YMMWORD PTR [r12 + 0], ymm6\t\n"
          r12_32 = _mm256_load_ps((float*)((char*)r12 + 32));      //"vfmadd231ps ymm7,ymm15,YMMWORD PTR [r12 + 32]\t\n"
          ymm7 = _mm256_fmadd_ps(r12_32, ymm15, ymm7);      //"vfmadd231ps ymm7,ymm15,YMMWORD PTR [r12 + 32]\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 32), ymm7);      //"vmovups YMMWORD PTR [r12 + 32], ymm7\t\n"
        }      //"L_done%=:\t\n"

        // next outer iteration
        rcx = (float*)((char*)rcx + 64);      //"add rcx, 64\t\n"
        r12 = rcx;      //"mov r12, rcx\t\n"
        r9 = rax;      //"mov r9, rax\t\n"
      }      //"inc rbx; cmp rbx, rdi; jl loop_outter%=\t\n"
}

void NOINLINE_ATTR gemmkernel_5x2_AVX2_fA0fB0fC0(GemmParams* gp) {
      char* r14 = (char*)gp;      //"mov r14, %[gp]\t\n"

      // Copy parameters
      // k
      uint64_t    r8  = *(uint64_t   *)((char*)r14 + 0 );      //"mov r8, [r14 + 0]\t\n"
      // A
      float*      r9  = *(float*     *)((char*)r14 + 8 );      //"mov r9, [r14 + 8]\t\n"
      // B
      const fp16* r10 = *(const fp16**)((char*)r14 + 16);      //"mov r10, [r14 + 16]\t\n"
      // beta
      float*      r15 = *(float*     *)((char*)r14 + 24);      //"mov r15, [r14 + 24]\t\n"
      // accum
      uint64_t    rdx = *(uint64_t   *)((char*)r14 + 32);      //"mov rdx, [r14 + 32]\t\n"
      // C
      float*      r12 = *(float*     *)((char*)r14 + 40);      //"mov r12, [r14 + 40]\t\n"
      // ldc
      uint64_t    r13 = *(uint64_t   *)((char*)r14 + 48);      //"mov r13, [r14 + 48]\t\n"
      // b_block_cols
      uint64_t    rdi = *(uint64_t   *)((char*)r14 + 56);      //"mov rdi, [r14 + 56]\t\n"
      // b_block_size
      uint64_t    rsi = *(uint64_t   *)((char*)r14 + 64);      //"mov rsi, [r14 + 64]\t\n"
      // Make copies of A and C
      float* rax = r9;      //"mov rax, r9\t\n"
      float* rcx = r12;      //"mov rcx, r12\t\n"

      uint64_t rbx = 0;      //"mov rbx, 0\t\n"
      for (; rbx < rdi; ++rbx) {      //"inc rbx; cmp rbx, rdi; jl loop_outter%=\t\n"
      //       //"loop_outter%=:\t\n"
        uint64_t r14_i = 0;      //"mov r14, 0\t\n"
        __m256 ymm0 = _mm256_setzero_ps();      //"vxorps ymm0,ymm0,ymm0\t\n"
        __m256 ymm1 = _mm256_setzero_ps();      //"vxorps ymm1,ymm1,ymm1\t\n"
        __m256 ymm2 = _mm256_setzero_ps();      //"vxorps ymm2,ymm2,ymm2\t\n"
        __m256 ymm3 = _mm256_setzero_ps();      //"vxorps ymm3,ymm3,ymm3\t\n"
        __m256 ymm4 = _mm256_setzero_ps();      //"vxorps ymm4,ymm4,ymm4\t\n"
        __m256 ymm5 = _mm256_setzero_ps();      //"vxorps ymm5,ymm5,ymm5\t\n"
        __m256 ymm6 = _mm256_setzero_ps();      //"vxorps ymm6,ymm6,ymm6\t\n"
        __m256 ymm7 = _mm256_setzero_ps();      //"vxorps ymm7,ymm7,ymm7\t\n"
        __m256 ymm8 = _mm256_setzero_ps();      //"vxorps ymm8,ymm8,ymm8\t\n"
        __m256 ymm9 = _mm256_setzero_ps();      //"vxorps ymm9,ymm9,ymm9\t\n"

        for (; r14_i < r8; ++r14_i) {      //"inc r14; cmp r14, r8; jl loop_inner%=\t\n"
        // loop_inner%=:      //"\t\n"
          auto fp16mem0 = _mm_load_si128((__m128i*)((char*)r10 + 0));      //"vcvtph2ps ymm11,XMMWORD PTR [r10 + 0]\t\n"
          auto ymm11 = _mm256_cvtph_ps(fp16mem0);      //"vcvtph2ps ymm11,XMMWORD PTR [r10 + 0]\t\n"
          auto fp16mem16 = _mm_load_si128((__m128i*)((char*)r10 + 16));      //"vcvtph2ps ymm12,XMMWORD PTR [r10 + 16]\t\n"
          auto ymm12 = _mm256_cvtph_ps(fp16mem16);      //"vcvtph2ps ymm12,XMMWORD PTR [r10 + 16]\t\n"
          auto ymm10 = _mm256_broadcast_ss((float*)((char*)r9 + 0));      //"vbroadcastss ymm10,DWORD PTR [r9+0]\t\n"
          ymm0 = _mm256_fmadd_ps(ymm10, ymm11, ymm0);      //"vfmadd231ps ymm0,ymm11,ymm10\t\n"
          ymm1 = _mm256_fmadd_ps(ymm10, ymm12, ymm1);      //"vfmadd231ps ymm1,ymm12,ymm10\t\n"
          ymm10 = _mm256_broadcast_ss((float*)((char*)r9 + 4));      //"vbroadcastss ymm10,DWORD PTR [r9+4]\t\n"
          ymm2 = _mm256_fmadd_ps(ymm10, ymm11, ymm2);      //"vfmadd231ps ymm2,ymm11,ymm10\t\n"
          ymm3 = _mm256_fmadd_ps(ymm10, ymm12, ymm3);      //"vfmadd231ps ymm3,ymm12,ymm10\t\n"
          ymm10 = _mm256_broadcast_ss((float*)((char*)r9 + 8));      //"vbroadcastss ymm10,DWORD PTR [r9+8]\t\n"
          ymm4 = _mm256_fmadd_ps(ymm10, ymm11, ymm4);      //"vfmadd231ps ymm4,ymm11,ymm10\t\n"
          ymm5 = _mm256_fmadd_ps(ymm10, ymm12, ymm5);      //"vfmadd231ps ymm5,ymm12,ymm10\t\n"
          ymm10 = _mm256_broadcast_ss((float*)((char*)r9 + 12));      //"vbroadcastss ymm10,DWORD PTR [r9+12]\t\n"
          ymm6 = _mm256_fmadd_ps(ymm10, ymm11, ymm6);      //"vfmadd231ps ymm6,ymm11,ymm10\t\n"
          ymm7 = _mm256_fmadd_ps(ymm10, ymm12, ymm7);      //"vfmadd231ps ymm7,ymm12,ymm10\t\n"
          ymm10 = _mm256_broadcast_ss((float*)((char*)r9 + 16));      //"vbroadcastss ymm10,DWORD PTR [r9+16]\t\n"
          ymm8 = _mm256_fmadd_ps(ymm10, ymm11, ymm8);      //"vfmadd231ps ymm8,ymm11,ymm10\t\n"
          ymm9 = _mm256_fmadd_ps(ymm10, ymm12, ymm9);      //"vfmadd231ps ymm9,ymm12,ymm10\t\n"
          r9 = (float*)((char*)r9 + 20);      //"add r9,20\t\n"
          r10 = (fp16*)((char*)r10 + 32);      //"add r10,32\t\n"
        }      //"inc r14; cmp r14, r8; jl loop_outter%=\t\n"

        if(rdx != 1) {      //"cmp rdx, 1; je L_accum%=\t\n"
          // Dump C
          _mm256_storeu_ps((float*)((char*)r12 + 0), ymm0);      //"vmovups YMMWORD PTR [r12 + 0], ymm0\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 32), ymm1);      //"vmovups YMMWORD PTR [r12 + 32], ymm1\t\n"
          r12 = (float*)((char*)r12 + r13);      //"add r12, r13\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 0), ymm2);      //"vmovups YMMWORD PTR [r12 + 0], ymm2\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 32), ymm3);      //"vmovups YMMWORD PTR [r12 + 32], ymm3\t\n"
          r12 = (float*)((char*)r12 + r13);      //"add r12, r13\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 0), ymm4);      //"vmovups YMMWORD PTR [r12 + 0], ymm4\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 32), ymm5);      //"vmovups YMMWORD PTR [r12 + 32], ymm5\t\n"
          r12 = (float*)((char*)r12 + r13);      //"add r12, r13\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 0), ymm6);      //"vmovups YMMWORD PTR [r12 + 0], ymm6\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 32), ymm7);      //"vmovups YMMWORD PTR [r12 + 32], ymm7\t\n"
          r12 = (float*)((char*)r12 + r13);      //"add r12, r13\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 0), ymm8);      //"vmovups YMMWORD PTR [r12 + 0], ymm8\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 32), ymm9);      //"vmovups YMMWORD PTR [r12 + 32], ymm9\t\n"
        } else {      //"jmp L_done%=\t\n"
          // Dump C with accumulate
          auto ymm15 = _mm256_broadcast_ss((float*)r15);      //"vbroadcastss ymm15,DWORD PTR [r15]\t\n"
          auto r12_0 = _mm256_load_ps((float*)((char*)r12 + 0));      //"vfmadd231ps ymm0,ymm15,YMMWORD PTR [r12 + 0]\t\n"
          ymm0 = _mm256_fmadd_ps(r12_0, ymm15, ymm0);      //"vfmadd231ps ymm0,ymm15,YMMWORD PTR [r12 + 0]\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 0), ymm0);      //"vmovups YMMWORD PTR [r12 + 0], ymm0\t\n"
          auto r12_32 = _mm256_load_ps((float*)((char*)r12 + 32));      //"vfmadd231ps ymm1,ymm15,YMMWORD PTR [r12 + 32]\t\n"
          ymm1 = _mm256_fmadd_ps(r12_32, ymm15, ymm1);      //"vfmadd231ps ymm1,ymm15,YMMWORD PTR [r12 + 32]\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 32), ymm1);      //"vmovups YMMWORD PTR [r12 + 32], ymm1\t\n"
          r12 = (float*)((char*)r12 + r13);      //"add r12, r13\t\n"
          r12_0 = _mm256_load_ps((float*)((char*)r12 + 0));      //"vfmadd231ps ymm2,ymm15,YMMWORD PTR [r12 + 0]\t\n"
          ymm2 = _mm256_fmadd_ps(r12_0, ymm15, ymm2);      //"vfmadd231ps ymm2,ymm15,YMMWORD PTR [r12 + 0]\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 0), ymm2);      //"vmovups YMMWORD PTR [r12 + 0], ymm2\t\n"
          r12_32 = _mm256_load_ps((float*)((char*)r12 + 32));      //"vfmadd231ps ymm3,ymm15,YMMWORD PTR [r12 + 32]\t\n"
          ymm3 = _mm256_fmadd_ps(r12_32, ymm15, ymm3);      //"vfmadd231ps ymm3,ymm15,YMMWORD PTR [r12 + 32]\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 32), ymm3);      //"vmovups YMMWORD PTR [r12 + 32], ymm3\t\n"
          r12 = (float*)((char*)r12 + r13);      //"add r12, r13\t\n"
          r12_0 = _mm256_load_ps((float*)((char*)r12 + 0));      //"vfmadd231ps ymm4,ymm15,YMMWORD PTR [r12 + 0]\t\n"
          ymm4 = _mm256_fmadd_ps(r12_0, ymm15, ymm4);      //"vfmadd231ps ymm4,ymm15,YMMWORD PTR [r12 + 0]\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 0), ymm4);      //"vmovups YMMWORD PTR [r12 + 0], ymm4\t\n"
          r12_32 = _mm256_load_ps((float*)((char*)r12 + 32));      //"vfmadd231ps ymm5,ymm15,YMMWORD PTR [r12 + 32]\t\n"
          ymm5 = _mm256_fmadd_ps(r12_32, ymm15, ymm5);      //"vfmadd231ps ymm5,ymm15,YMMWORD PTR [r12 + 32]\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 32), ymm5);      //"vmovups YMMWORD PTR [r12 + 32], ymm5\t\n"
          r12 = (float*)((char*)r12 + r13);      //"add r12, r13\t\n"
          r12_0 = _mm256_load_ps((float*)((char*)r12 + 0));      //"vfmadd231ps ymm6,ymm15,YMMWORD PTR [r12 + 0]\t\n"
          ymm6 = _mm256_fmadd_ps(r12_0, ymm15, ymm6);      //"vfmadd231ps ymm6,ymm15,YMMWORD PTR [r12 + 0]\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 0), ymm6);      //"vmovups YMMWORD PTR [r12 + 0], ymm6\t\n"
          r12_32 = _mm256_load_ps((float*)((char*)r12 + 32));      //"vfmadd231ps ymm7,ymm15,YMMWORD PTR [r12 + 32]\t\n"
          ymm7 = _mm256_fmadd_ps(r12_32, ymm15, ymm7);      //"vfmadd231ps ymm7,ymm15,YMMWORD PTR [r12 + 32]\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 32), ymm7);      //"vmovups YMMWORD PTR [r12 + 32], ymm7\t\n"
          r12 = (float*)((char*)r12 + r13);      //"add r12, r13\t\n"
          r12_0 = _mm256_load_ps((float*)((char*)r12 + 0));      //"vfmadd231ps ymm8,ymm15,YMMWORD PTR [r12 + 0]\t\n"
          ymm8 = _mm256_fmadd_ps(r12_0, ymm15, ymm8);      //"vfmadd231ps ymm8,ymm15,YMMWORD PTR [r12 + 0]\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 0), ymm8);      //"vmovups YMMWORD PTR [r12 + 0], ymm8\t\n"
          r12_32 = _mm256_load_ps((float*)((char*)r12 + 32));      //"vfmadd231ps ymm9,ymm15,YMMWORD PTR [r12 + 32]\t\n"
          ymm9 = _mm256_fmadd_ps(r12_32, ymm15, ymm9);      //"vfmadd231ps ymm9,ymm15,YMMWORD PTR [r12 + 32]\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 32), ymm9);      //"vmovups YMMWORD PTR [r12 + 32], ymm9\t\n"
        }      //"L_done%=:\t\n"

        // next outer iteration
        rcx = (float*)((char*)rcx + 64);      //"add rcx, 64\t\n"
        r12 = rcx;      //"mov r12, rcx\t\n"
        r9 = rax;      //"mov r9, rax\t\n"
      }      //"inc rbx; cmp rbx, rdi; jl loop_outter%=\t\n"
}

void NOINLINE_ATTR gemmkernel_6x2_AVX2_fA0fB0fC0(GemmParams* gp) {
      char* r14 = (char*)gp;      //"mov r14, %[gp]\t\n"

      // Copy parameters
      // k
      uint64_t    r8  = *(uint64_t   *)((char*)r14 + 0 );      //"mov r8, [r14 + 0]\t\n"
      // A
      float*      r9  = *(float*     *)((char*)r14 + 8 );      //"mov r9, [r14 + 8]\t\n"
      // B
      const fp16* r10 = *(const fp16**)((char*)r14 + 16);      //"mov r10, [r14 + 16]\t\n"
      // beta
      float*      r15 = *(float*     *)((char*)r14 + 24);      //"mov r15, [r14 + 24]\t\n"
      // accum
      uint64_t    rdx = *(uint64_t   *)((char*)r14 + 32);      //"mov rdx, [r14 + 32]\t\n"
      // C
      float*      r12 = *(float*     *)((char*)r14 + 40);      //"mov r12, [r14 + 40]\t\n"
      // ldc
      uint64_t    r13 = *(uint64_t   *)((char*)r14 + 48);      //"mov r13, [r14 + 48]\t\n"
      // b_block_cols
      uint64_t    rdi = *(uint64_t   *)((char*)r14 + 56);      //"mov rdi, [r14 + 56]\t\n"
      // b_block_size
      uint64_t    rsi = *(uint64_t   *)((char*)r14 + 64);      //"mov rsi, [r14 + 64]\t\n"
      // Make copies of A and C
      float* rax = r9;      //"mov rax, r9\t\n"
      float* rcx = r12;      //"mov rcx, r12\t\n"

      uint64_t rbx = 0;      //"mov rbx, 0\t\n"
      for (; rbx < rdi; ++rbx) {      //"inc rbx; cmp rbx, rdi; jl loop_outter%=\t\n"
      //       //"loop_outter%=:\t\n"
        uint64_t r14_i = 0;      //"mov r14, 0\t\n"
        __m256 ymm0 = _mm256_setzero_ps();      //"vxorps ymm0,ymm0,ymm0\t\n"
        __m256 ymm1 = _mm256_setzero_ps();      //"vxorps ymm1,ymm1,ymm1\t\n"
        __m256 ymm2 = _mm256_setzero_ps();      //"vxorps ymm2,ymm2,ymm2\t\n"
        __m256 ymm3 = _mm256_setzero_ps();      //"vxorps ymm3,ymm3,ymm3\t\n"
        __m256 ymm4 = _mm256_setzero_ps();      //"vxorps ymm4,ymm4,ymm4\t\n"
        __m256 ymm5 = _mm256_setzero_ps();      //"vxorps ymm5,ymm5,ymm5\t\n"
        __m256 ymm6 = _mm256_setzero_ps();      //"vxorps ymm6,ymm6,ymm6\t\n"
        __m256 ymm7 = _mm256_setzero_ps();      //"vxorps ymm7,ymm7,ymm7\t\n"
        __m256 ymm8 = _mm256_setzero_ps();      //"vxorps ymm8,ymm8,ymm8\t\n"
        __m256 ymm9 = _mm256_setzero_ps();      //"vxorps ymm9,ymm9,ymm9\t\n"
        __m256 ymm10 = _mm256_setzero_ps();      //"vxorps ymm10,ymm10,ymm10\t\n"
        __m256 ymm11 = _mm256_setzero_ps();      //"vxorps ymm11,ymm11,ymm11\t\n"

        for (; r14_i < r8; ++r14_i) {      //"inc r14; cmp r14, r8; jl loop_inner%=\t\n"
        // loop_inner%=:      //"\t\n"
          auto fp16mem0 = _mm_load_si128((__m128i*)((char*)r10 + 0));      //"vcvtph2ps ymm13,XMMWORD PTR [r10 + 0]\t\n"
          auto ymm13 = _mm256_cvtph_ps(fp16mem0);      //"vcvtph2ps ymm13,XMMWORD PTR [r10 + 0]\t\n"
          auto fp16mem16 = _mm_load_si128((__m128i*)((char*)r10 + 16));      //"vcvtph2ps ymm14,XMMWORD PTR [r10 + 16]\t\n"
          auto ymm14 = _mm256_cvtph_ps(fp16mem16);      //"vcvtph2ps ymm14,XMMWORD PTR [r10 + 16]\t\n"
          auto ymm12 = _mm256_broadcast_ss((float*)((char*)r9 + 0));      //"vbroadcastss ymm12,DWORD PTR [r9+0]\t\n"
          ymm0 = _mm256_fmadd_ps(ymm12, ymm13, ymm0);      //"vfmadd231ps ymm0,ymm13,ymm12\t\n"
          ymm1 = _mm256_fmadd_ps(ymm12, ymm14, ymm1);      //"vfmadd231ps ymm1,ymm14,ymm12\t\n"
          ymm12 = _mm256_broadcast_ss((float*)((char*)r9 + 4));      //"vbroadcastss ymm12,DWORD PTR [r9+4]\t\n"
          ymm2 = _mm256_fmadd_ps(ymm12, ymm13, ymm2);      //"vfmadd231ps ymm2,ymm13,ymm12\t\n"
          ymm3 = _mm256_fmadd_ps(ymm12, ymm14, ymm3);      //"vfmadd231ps ymm3,ymm14,ymm12\t\n"
          ymm12 = _mm256_broadcast_ss((float*)((char*)r9 + 8));      //"vbroadcastss ymm12,DWORD PTR [r9+8]\t\n"
          ymm4 = _mm256_fmadd_ps(ymm12, ymm13, ymm4);      //"vfmadd231ps ymm4,ymm13,ymm12\t\n"
          ymm5 = _mm256_fmadd_ps(ymm12, ymm14, ymm5);      //"vfmadd231ps ymm5,ymm14,ymm12\t\n"
          ymm12 = _mm256_broadcast_ss((float*)((char*)r9 + 12));      //"vbroadcastss ymm12,DWORD PTR [r9+12]\t\n"
          ymm6 = _mm256_fmadd_ps(ymm12, ymm13, ymm6);      //"vfmadd231ps ymm6,ymm13,ymm12\t\n"
          ymm7 = _mm256_fmadd_ps(ymm12, ymm14, ymm7);      //"vfmadd231ps ymm7,ymm14,ymm12\t\n"
          ymm12 = _mm256_broadcast_ss((float*)((char*)r9 + 16));      //"vbroadcastss ymm12,DWORD PTR [r9+16]\t\n"
          ymm8 = _mm256_fmadd_ps(ymm12, ymm13, ymm8);      //"vfmadd231ps ymm8,ymm13,ymm12\t\n"
          ymm9 = _mm256_fmadd_ps(ymm12, ymm14, ymm9);      //"vfmadd231ps ymm9,ymm14,ymm12\t\n"
          ymm12 = _mm256_broadcast_ss((float*)((char*)r9 + 20));      //"vbroadcastss ymm12,DWORD PTR [r9+20]\t\n"
          ymm10 = _mm256_fmadd_ps(ymm12, ymm13, ymm10);      //"vfmadd231ps ymm10,ymm13,ymm12\t\n"
          ymm11 = _mm256_fmadd_ps(ymm12, ymm14, ymm11);      //"vfmadd231ps ymm11,ymm14,ymm12\t\n"
          r9 = (float*)((char*)r9 + 24);      //"add r9,24\t\n"
          r10 = (fp16*)((char*)r10 + 32);      //"add r10,32\t\n"
        }      //"inc r14; cmp r14, r8; jl loop_outter%=\t\n"

        if(rdx != 1) {      //"cmp rdx, 1; je L_accum%=\t\n"
          // Dump C
          _mm256_storeu_ps((float*)((char*)r12 + 0), ymm0);      //"vmovups YMMWORD PTR [r12 + 0], ymm0\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 32), ymm1);      //"vmovups YMMWORD PTR [r12 + 32], ymm1\t\n"
          r12 = (float*)((char*)r12 + r13);      //"add r12, r13\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 0), ymm2);      //"vmovups YMMWORD PTR [r12 + 0], ymm2\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 32), ymm3);      //"vmovups YMMWORD PTR [r12 + 32], ymm3\t\n"
          r12 = (float*)((char*)r12 + r13);      //"add r12, r13\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 0), ymm4);      //"vmovups YMMWORD PTR [r12 + 0], ymm4\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 32), ymm5);      //"vmovups YMMWORD PTR [r12 + 32], ymm5\t\n"
          r12 = (float*)((char*)r12 + r13);      //"add r12, r13\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 0), ymm6);      //"vmovups YMMWORD PTR [r12 + 0], ymm6\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 32), ymm7);      //"vmovups YMMWORD PTR [r12 + 32], ymm7\t\n"
          r12 = (float*)((char*)r12 + r13);      //"add r12, r13\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 0), ymm8);      //"vmovups YMMWORD PTR [r12 + 0], ymm8\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 32), ymm9);      //"vmovups YMMWORD PTR [r12 + 32], ymm9\t\n"
          r12 = (float*)((char*)r12 + r13);      //"add r12, r13\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 0), ymm10);      //"vmovups YMMWORD PTR [r12 + 0], ymm10\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 32), ymm11);      //"vmovups YMMWORD PTR [r12 + 32], ymm11\t\n"
        } else {      //"jmp L_done%=\t\n"
          // Dump C with accumulate
          auto ymm15 = _mm256_broadcast_ss((float*)r15);      //"vbroadcastss ymm15,DWORD PTR [r15]\t\n"
          auto r12_0 = _mm256_load_ps((float*)((char*)r12 + 0));      //"vfmadd231ps ymm0,ymm15,YMMWORD PTR [r12 + 0]\t\n"
          ymm0 = _mm256_fmadd_ps(r12_0, ymm15, ymm0);      //"vfmadd231ps ymm0,ymm15,YMMWORD PTR [r12 + 0]\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 0), ymm0);      //"vmovups YMMWORD PTR [r12 + 0], ymm0\t\n"
          auto r12_32 = _mm256_load_ps((float*)((char*)r12 + 32));      //"vfmadd231ps ymm1,ymm15,YMMWORD PTR [r12 + 32]\t\n"
          ymm1 = _mm256_fmadd_ps(r12_32, ymm15, ymm1);      //"vfmadd231ps ymm1,ymm15,YMMWORD PTR [r12 + 32]\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 32), ymm1);      //"vmovups YMMWORD PTR [r12 + 32], ymm1\t\n"
          r12 = (float*)((char*)r12 + r13);      //"add r12, r13\t\n"
          r12_0 = _mm256_load_ps((float*)((char*)r12 + 0));      //"vfmadd231ps ymm2,ymm15,YMMWORD PTR [r12 + 0]\t\n"
          ymm2 = _mm256_fmadd_ps(r12_0, ymm15, ymm2);      //"vfmadd231ps ymm2,ymm15,YMMWORD PTR [r12 + 0]\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 0), ymm2);      //"vmovups YMMWORD PTR [r12 + 0], ymm2\t\n"
          r12_32 = _mm256_load_ps((float*)((char*)r12 + 32));      //"vfmadd231ps ymm3,ymm15,YMMWORD PTR [r12 + 32]\t\n"
          ymm3 = _mm256_fmadd_ps(r12_32, ymm15, ymm3);      //"vfmadd231ps ymm3,ymm15,YMMWORD PTR [r12 + 32]\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 32), ymm3);      //"vmovups YMMWORD PTR [r12 + 32], ymm3\t\n"
          r12 = (float*)((char*)r12 + r13);      //"add r12, r13\t\n"
          r12_0 = _mm256_load_ps((float*)((char*)r12 + 0));      //"vfmadd231ps ymm4,ymm15,YMMWORD PTR [r12 + 0]\t\n"
          ymm4 = _mm256_fmadd_ps(r12_0, ymm15, ymm4);      //"vfmadd231ps ymm4,ymm15,YMMWORD PTR [r12 + 0]\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 0), ymm4);      //"vmovups YMMWORD PTR [r12 + 0], ymm4\t\n"
          r12_32 = _mm256_load_ps((float*)((char*)r12 + 32));      //"vfmadd231ps ymm5,ymm15,YMMWORD PTR [r12 + 32]\t\n"
          ymm5 = _mm256_fmadd_ps(r12_32, ymm15, ymm5);      //"vfmadd231ps ymm5,ymm15,YMMWORD PTR [r12 + 32]\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 32), ymm5);      //"vmovups YMMWORD PTR [r12 + 32], ymm5\t\n"
          r12 = (float*)((char*)r12 + r13);      //"add r12, r13\t\n"
          r12_0 = _mm256_load_ps((float*)((char*)r12 + 0));      //"vfmadd231ps ymm6,ymm15,YMMWORD PTR [r12 + 0]\t\n"
          ymm6 = _mm256_fmadd_ps(r12_0, ymm15, ymm6);      //"vfmadd231ps ymm6,ymm15,YMMWORD PTR [r12 + 0]\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 0), ymm6);      //"vmovups YMMWORD PTR [r12 + 0], ymm6\t\n"
          r12_32 = _mm256_load_ps((float*)((char*)r12 + 32));      //"vfmadd231ps ymm7,ymm15,YMMWORD PTR [r12 + 32]\t\n"
          ymm7 = _mm256_fmadd_ps(r12_32, ymm15, ymm7);      //"vfmadd231ps ymm7,ymm15,YMMWORD PTR [r12 + 32]\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 32), ymm7);      //"vmovups YMMWORD PTR [r12 + 32], ymm7\t\n"
          r12 = (float*)((char*)r12 + r13);      //"add r12, r13\t\n"
          r12_0 = _mm256_load_ps((float*)((char*)r12 + 0));      //"vfmadd231ps ymm8,ymm15,YMMWORD PTR [r12 + 0]\t\n"
          ymm8 = _mm256_fmadd_ps(r12_0, ymm15, ymm8);      //"vfmadd231ps ymm8,ymm15,YMMWORD PTR [r12 + 0]\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 0), ymm8);      //"vmovups YMMWORD PTR [r12 + 0], ymm8\t\n"
          r12_32 = _mm256_load_ps((float*)((char*)r12 + 32));      //"vfmadd231ps ymm9,ymm15,YMMWORD PTR [r12 + 32]\t\n"
          ymm9 = _mm256_fmadd_ps(r12_32, ymm15, ymm9);      //"vfmadd231ps ymm9,ymm15,YMMWORD PTR [r12 + 32]\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 32), ymm9);      //"vmovups YMMWORD PTR [r12 + 32], ymm9\t\n"
          r12 = (float*)((char*)r12 + r13);      //"add r12, r13\t\n"
          r12_0 = _mm256_load_ps((float*)((char*)r12 + 0));      //"vfmadd231ps ymm10,ymm15,YMMWORD PTR [r12 + 0]\t\n"
          ymm10 = _mm256_fmadd_ps(r12_0, ymm15, ymm10);      //"vfmadd231ps ymm10,ymm15,YMMWORD PTR [r12 + 0]\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 0), ymm10);      //"vmovups YMMWORD PTR [r12 + 0], ymm10\t\n"
          r12_32 = _mm256_load_ps((float*)((char*)r12 + 32));      //"vfmadd231ps ymm11,ymm15,YMMWORD PTR [r12 + 32]\t\n"
          ymm11 = _mm256_fmadd_ps(r12_32, ymm15, ymm11);      //"vfmadd231ps ymm11,ymm15,YMMWORD PTR [r12 + 32]\t\n"
          _mm256_storeu_ps((float*)((char*)r12 + 32), ymm11);      //"vmovups YMMWORD PTR [r12 + 32], ymm11\t\n"
        }      //"L_done%=:\t\n"

        // next outer iteration
        rcx = (float*)((char*)rcx + 64);      //"add rcx, 64\t\n"
        r12 = rcx;      //"mov r12, rcx\t\n"
        r9 = rax;      //"mov r9, rax\t\n"
      }      //"inc rbx; cmp rbx, rdi; jl loop_outter%=\t\n"
}


} // namespace fbgemm
