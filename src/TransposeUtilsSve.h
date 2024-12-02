/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cassert>
#include <cstdint>
#include "fbgemm/Utils.h"

#ifdef __aarch64__
#include <arm_sve.h>
#endif

namespace fbgemm {

namespace internal {

#if HAVE_SVE
// NOTE: Make sure every function defined in here has static linkage because
// this header file is included by UtilsAvx512.cc compiled with -mavx512f option

// 4 * 4 = 16 instructions
static inline void transpose_kernel_4x4_sve(
    const float* src,
    int64_t ld_src,
    float* dst,
    int64_t ld_dst) {
  asm volatile(
      "mov x0, %[src]\t\n"
      "mov x1, %[ld_src]\t\n"
      "mov x2, %[dst]\t\n"
      "mov x3, %[ld_dst]\t\n"

      "ldr q0, [x0]\t\n"
      "lsl x1, x1, #2\t\n"
      "add x0, x0, x1\t\n"
      "ldr q1, [x0]\t\n"
      "add x0, x0, x1\t\n"
      "ldr q2, [x0]\t\n"
      "add x0, x0, x1\t\n"
      "ldr q3, [x0]\t\n"

      "zip1 z4.s, z0.s, z2.s\t\n"
      "zip1 z5.s, z1.s, z3.s\t\n"
      "zip2 z6.s, z0.s, z2.s\t\n"

      "zip1 z0.s, z4.s, z5.s\t\n"
      "lsl x3, x3, #2\t\n"

      "str q0, [x2]\t\n"

      "zip2 z7.s, z1.s, z3.s\t\n"
      "zip2 z1.s, z4.s, z5.s\t\n"
      "add x2, x2, x3\t\n"

      "str q1, [x2]\t\n"

      "zip1 z2.s, z6.s, z7.s\t\n"
      "add x2, x2, x3\t\n"

      "str q2, [x2]\t\n"

      "zip2 z3.s, z6.s, z7.s\t\n"
      "add x2, x2, x3\t\n"

      "str q3, [x2]\t\n"

      :
      :
      [src] "r"(src), [ld_src] "r"(ld_src), [dst] "r"(dst), [ld_dst] "r"(ld_dst)
      : "memory",
        "cc",
        "x0",
        "x1",
        "x2",
        "x3",
        "z0",
        "z1",
        "z2",
        "z3",
        "z4",
        "z5",
        "z6",
        "z7");
}

// kernel for transpose mxn where m, n <= 4
// M + (M + 1) / 2 * 2 + 2 * N instructions
template <unsigned M>
static void transpose_kernel_mxn_small_sve(
    unsigned N,
    const float* src,
    int64_t ld_src,
    float* dst,
    int64_t ld_dst) {
  asm volatile(

      "mov x0, %[N]\t\n"
      "mov x1, %[src]\t\n"
      "mov x2, %[ld_src]\t\n"
      "mov x3, %[dst]\t\n"
      "mov x4, %[ld_dst]\t\n"
      "mov x5, #%[M]\t\n"
      "mov x6, #0\t\n"
      "mov x7, #4\t\n"
      "lsl x5, x5, #2\t\n"

      // Lowest 128-bits
      "whilelo p0.s, w6, w7\t\n"

      "whilelo p1.s, w6, w0\t\n"
      "whilelo p2.s, w6, w5\t\n"
      "and p3.b, p0/z, p1.b, p2.b\t\n"

      "ld1rqw { z0.s }, p3/z, [x1]\t\n"

      "whilelo p2.s, w7, w5\t\n"
      "and p3.b, p0/z, p1.b, p2.b\t\n"

      "ld1rqw { z1.s }, p3/z, [x1, x2, lsl #2]\t\n"

      "mov x7, #8\t\n"
      "lsl x8, x2, 1\t\n"
      "whilelo p2.s, w7, w5\t\n"
      "and p3.b, p0/z, p1.b, p2.b\t\n"

      "ld1rqw { z2.s }, p3/z, [x1, x8, lsl #2]\t\n"

      "mov x7, #12\t\n"
      "add x8, x8, x2\t\n"
      "whilelo p2.s, w7, w5\t\n"
      "and p3.b, p0/z, p1.b, p2.b\t\n"

      "ld1rqw { z3.s }, p3/z, [x1, x8, lsl #2]\t\n"

      "zip1 z4.s, z0.s, z2.s\t\n"
      "zip1 z5.s, z1.s, z3.s\t\n"
      "zip2 z6.s, z0.s, z2.s\t\n"
      "zip2 z7.s, z1.s, z3.s\t\n"

      "zip1 z0.s, z4.s, z5.s\t\n"

      "lsr x5, x5, #2\t\n"
      "lsl x0, x0, #2\t\n"

      "whilelo p1.s, w6, w5\t\n"
      "whilelo p2.s, w6, w0\t\n"
      "and p3.b, p0/z, p1.b, p2.b\t\n"

      "st1w { z0.s }, p3, [x3]\t\n"

      "zip2 z1.s, z4.s, z5.s\t\n"

      "mov x7, #4\t\n"
      "whilelo p2.s, w7, w0\t\n"
      "and p3.b, p0/z, p1.b, p2.b\t\n"

      "st1w { z1.s }, p3, [x3, x4, lsl #2]\t\n"

      "zip1 z2.s, z6.s, z7.s\t\n"

      "mov x7, #8\t\n"
      "lsl x8, x4, 1\t\n"
      "whilelo p2.s, w7, w0\t\n"
      "and p3.b, p0/z, p1.b, p2.b\t\n"

      "st1w { z2.s }, p3, [x3, x8, lsl #2]\t\n"

      "zip2 z3.s, z6.s, z7.s\t\n"

      "mov x7, #12\t\n"
      "add x8, x8, x4\t\n"
      "whilelo p2.s, w7, w0\t\n"
      "and p3.b, p0/z, p1.b, p2.b\t\n"

      "st1w { z3.s }, p3, [x3, x8, lsl #2]\t\n"

      :
      : [M] "i"(M),
        [N] "r"(N),
        [src] "r"(src),
        [ld_src] "r"(ld_src),
        [dst] "r"(dst),
        [ld_dst] "r"(ld_dst)
      : "memory",
        "cc",
        "x0",
        "x1",
        "x2",
        "x3",
        "x4",
        "x5",
        "x6",
        "x7",
        "x8",
        "z0",
        "z1",
        "z2",
        "z3",
        "z4",
        "z5",
        "z6",
        "z7",
        "p0",
        "p1",
        "p2",
        "p3");
}

// 8 * 5 = 40 instructions
static inline void transpose_kernel_8x8_sve(
    const float* src,
    int64_t ld_src,
    float* dst,
    int64_t ld_dst) {
  asm volatile(
      "mov x0, %[src]\t\n"
      "mov x1, %[ld_src]\t\n"
      "mov x2, %[dst]\t\n"
      "mov x3, %[ld_dst]\t\n"

      "ldp q0, q1, [x0]\t\n"
      "lsl x1, x1, #2\t\n"
      "add x0, x0, x1\t\n"
      "ldp q2, q3, [x0]\t\n"
      "add x0, x0, x1\t\n"
      "ldp q4, q5, [x0]\t\n"
      "add x0, x0, x1\t\n"
      "ldp q6, q7, [x0]\t\n"
      "add x0, x0, x1\t\n"
      "ldp q8, q9, [x0]\t\n"
      "add x0, x0, x1\t\n"
      "ldp q10, q11, [x0]\t\n"
      "add x0, x0, x1\t\n"
      "ldp q12, q13, [x0]\t\n"
      "add x0, x0, x1\t\n"
      "ldp q14, q15, [x0]\t\n"

      "zip1 z16.s, z0.s, z4.s\t\n"
      "zip1 z17.s, z2.s, z6.s\t\n"
      "zip1 z18.s, z8.s, z12.s\t\n"
      "zip1 z19.s, z10.s, z14.s\t\n"

      "zip1 z20.s, z16.s, z17.s\t\n"
      "zip1 z21.s, z18.s, z19.s\t\n"
      "lsl x3, x3, #2\t\n"

      "stp q20, q21, [x2]\t\n"

      "zip2 z20.s, z16.s, z17.s\t\n"
      "zip2 z21.s, z18.s, z19.s\t\n"
      "add x2, x2, x3\t\n"

      "stp q20, q21, [x2]\t\n"

      "zip2 z16.s, z0.s, z4.s\t\n"
      "zip2 z17.s, z2.s, z6.s\t\n"
      "zip2 z18.s, z8.s, z12.s\t\n"
      "zip2 z19.s, z10.s, z14.s\t\n"

      "zip1 z20.s, z16.s, z17.s\t\n"
      "zip1 z21.s, z18.s, z19.s\t\n"
      "add x2, x2, x3\t\n"

      "stp q20, q21, [x2]\t\n"

      "zip2 z20.s, z16.s, z17.s\t\n"
      "zip2 z21.s, z18.s, z19.s\t\n"
      "add x2, x2, x3\t\n"

      "stp q20, q21, [x2]\t\n"

      "zip1 z16.s, z1.s, z5.s\t\n"
      "zip1 z17.s, z3.s, z7.s\t\n"
      "zip1 z18.s, z9.s, z13.s\t\n"
      "zip1 z19.s, z11.s, z15.s\t\n"

      "zip1 z20.s, z16.s, z17.s\t\n"
      "zip1 z21.s, z18.s, z19.s\t\n"
      "add x2, x2, x3\t\n"

      "stp q20, q21, [x2]\t\n"

      "zip2 z20.s, z16.s, z17.s\t\n"
      "zip2 z21.s, z18.s, z19.s\t\n"
      "add x2, x2, x3\t\n"

      "stp q20, q21, [x2]\t\n"

      "zip2 z16.s, z1.s, z5.s\t\n"
      "zip2 z17.s, z3.s, z7.s\t\n"
      "zip2 z18.s, z9.s, z13.s\t\n"
      "zip2 z19.s, z11.s, z15.s\t\n"

      "zip1 z20.s, z16.s, z17.s\t\n"
      "zip1 z21.s, z18.s, z19.s\t\n"
      "add x2, x2, x3\t\n"

      "stp q20, q21, [x2]\t\n"

      "zip2 z20.s, z16.s, z17.s\t\n"
      "zip2 z21.s, z18.s, z19.s\t\n"
      "add x2, x2, x3\t\n"

      "stp q20, q21, [x2]\t\n"

      :
      :
      [src] "r"(src), [ld_src] "r"(ld_src), [dst] "r"(dst), [ld_dst] "r"(ld_dst)
      : "memory",
        "cc",
        "x0",
        "x1",
        "x2",
        "x3",
        "z0",
        "z1",
        "z2",
        "z3",
        "z4",
        "z5",
        "z6",
        "z7",
        "z8",
        "z9",
        "z10",
        "z11",
        "z12",
        "z13",
        "z14",
        "z15",
        "z16",
        "z17",
        "z18",
        "z19",
        "z20",
        "z21");
}

// kernel for transposing mxn where m, n <= 8
// M + (M + 1) / 2 * 2 + (M + 3) / 4 * 4 + 2 * N instructions
template <unsigned M>
static void transpose_kernel_mxn_large_sve(
    unsigned N,
    const float* src,
    int64_t ld_src,
    float* dst,
    int64_t ld_dst) {
  asm volatile(

      "mov x0, %[N]\t\n"
      "mov x1, %[src]\t\n"
      "mov x2, %[ld_src]\t\n"
      "mov x3, %[dst]\t\n"
      "mov x4, %[ld_dst]\t\n"
      "mov x5, #%[M]\t\n"
      "mov x6, #0\t\n"
      "mov x7, #4\t\n"
      "lsl x5, x5, #2\t\n"
      "add x12, x2, #4\t\n"
      "add x14, x4, #4\t\n"

      // Lowest 128-bits
      "whilelo p0.s, w6, w7\t\n"

      "whilelo p1.s, w6, w0\t\n"
      "whilelo p4.s, w7, w0\t\n"
      "whilelo p2.s, w6, w5\t\n"
      "and p3.b, p0/z, p1.b, p2.b\t\n"

      "ld1rqw { z0.s }, p3/z, [x1]\t\n"

      "and p3.b, p0/z, p4.b, p2.b\t\n"

      "ld1rqw { z8.s }, p3/z, [x1, #16]\t\n"

      "whilelo p2.s, w7, w5\t\n"
      "and p3.b, p0/z, p1.b, p2.b\t\n"

      "ld1rqw { z1.s }, p3/z, [x1, x2, lsl #2]\t\n"

      "and p3.b, p0/z, p4.b, p2.b\t\n"

      "ld1rqw { z9.s }, p3/z, [x1, x12, lsl #2]\t\n"

      "mov x9, #8\t\n"
      "lsl x8, x2, 1\t\n"
      "whilelo p2.s, w9, w5\t\n"
      "add x12, x12, x2\t\n"

      "and p3.b, p0/z, p1.b, p2.b\t\n"

      "ld1rqw { z2.s }, p3/z, [x1, x8, lsl #2]\t\n"

      "and p3.b, p0/z, p4.b, p2.b\t\n"

      "ld1rqw { z10.s }, p3/z, [x1, x12, lsl #2]\t\n"

      "mov x9, #12\t\n"
      "add x8, x8, x2\t\n"
      "whilelo p2.s, w9, w5\t\n"
      "add x12, x12, x2\t\n"

      "and p3.b, p0/z, p1.b, p2.b\t\n"

      "ld1rqw { z3.s }, p3/z, [x1, x8, lsl #2]\t\n"

      "and p3.b, p0/z, p4.b, p2.b\t\n"

      "ld1rqw { z11.s }, p3/z, [x1, x12, lsl #2]\t\n"

      "mov x9, #16\t\n"
      "add x8, x8, x2\t\n"
      "whilelo p2.s, w9, w5\t\n"
      "add x12, x12, x2\t\n"

      "and p3.b, p0/z, p1.b, p2.b\t\n"

      "ld1rqw { z4.s }, p3/z, [x1, x8, lsl #2]\t\n"

      "and p3.b, p0/z, p4.b, p2.b\t\n"

      "ld1rqw { z12.s }, p3/z, [x1, x12, lsl #2]\t\n"

      "mov x9, #20\t\n"
      "add x8, x8, x2\t\n"
      "whilelo p2.s, w9, w5\t\n"
      "add x12, x12, x2\t\n"

      "and p3.b, p0/z, p1.b, p2.b\t\n"

      "ld1rqw { z5.s }, p3/z, [x1, x8, lsl #2]\t\n"

      "and p3.b, p0/z, p4.b, p2.b\t\n"

      "ld1rqw { z13.s }, p3/z, [x1, x12, lsl #2]\t\n"

      "mov x9, #24\t\n"
      "add x8, x8, x2\t\n"
      "whilelo p2.s, w9, w5\t\n"
      "add x12, x12, x2\t\n"

      "and p3.b, p0/z, p1.b, p2.b\t\n"

      "ld1rqw { z6.s }, p3/z, [x1, x8, lsl #2]\t\n"

      "and p3.b, p0/z, p4.b, p2.b\t\n"

      "ld1rqw { z14.s }, p3/z, [x1, x12, lsl #2]\t\n"

      "mov x9, #28\t\n"
      "add x8, x8, x2\t\n"
      "whilelo p2.s, w9, w5\t\n"
      "add x12, x12, x2\t\n"

      "and p3.b, p0/z, p1.b, p2.b\t\n"

      "ld1rqw { z7.s }, p3/z, [x1, x8, lsl #2]\t\n"

      "and p3.b, p0/z, p4.b, p2.b\t\n"

      "ld1rqw { z15.s }, p3/z, [x1, x12, lsl #2]\t\n"

      "zip1 z16.s, z0.s, z2.s\t\n"
      "zip1 z17.s, z1.s, z3.s\t\n"
      "zip1 z18.s, z4.s, z6.s\t\n"
      "zip1 z19.s, z5.s, z7.s\t\n"

      "lsr x5, x5, #2\t\n"
      "lsl x0, x0, #2\t\n"

      "zip1 z20.s, z16.s, z17.s\t\n"

      "whilelo p1.s, w6, w5\t\n"
      "whilelo p2.s, w6, w0\t\n"
      "whilelo p4.s, w7, w5\t\n"
      "and p3.b, p0/z, p1.b, p2.b\t\n"

      "st1w { z20.s }, p3, [x3]\t\n"

      "zip1 z21.s, z18.s, z19.s\t\n"
      "and p3.b, p0/z, p4.b, p2.b\t\n"

      "st1w { z21.s }, p3, [x3, x7, lsl #2]\t\n"

      "zip2 z20.s, z16.s, z17.s\t\n"

      "whilelo p2.s, w7, w0\t\n"
      "and p3.b, p0/z, p1.b, p2.b\t\n"

      "st1w { z20.s }, p3, [x3, x4, lsl #2]\t\n"

      "zip2 z21.s, z18.s, z19.s\t\n"
      "and p3.b, p0/z, p4.b, p2.b\t\n"

      "st1w { z21.s }, p3, [x3, x14, lsl #2]\t\n"

      "zip2 z16.s, z0.s, z2.s\t\n"
      "zip2 z17.s, z1.s, z3.s\t\n"
      "zip2 z18.s, z4.s, z6.s\t\n"
      "zip2 z19.s, z5.s, z7.s\t\n"

      "zip1 z20.s, z16.s, z17.s\t\n"

      "mov x7, #8\t\n"
      "lsl x8, x4, 1\t\n"
      "whilelo p2.s, w7, w0\t\n"
      "and p3.b, p0/z, p1.b, p2.b\t\n"
      "st1w { z20.s }, p3, [x3, x8, lsl #2]\t\n"

      "zip1 z21.s, z18.s, z19.s\t\n"
      "add x14, x14, x4\t\n"
      "and p3.b, p0/z, p4.b, p2.b\t\n"

      "st1w { z21.s }, p3, [x3, x14, lsl #2]\t\n"

      "zip2 z20.s, z16.s, z17.s\t\n"

      "mov x7, #12\t\n"
      "add x8, x8, x4\t\n"
      "whilelo p2.s, w7, w0\t\n"
      "and p3.b, p0/z, p1.b, p2.b\t\n"
      "st1w { z20.s }, p3, [x3, x8, lsl #2]\t\n"

      "zip2 z21.s, z18.s, z19.s\t\n"
      "add x14, x14, x4\t\n"
      "and p3.b, p0/z, p4.b, p2.b\t\n"

      "st1w { z21.s }, p3, [x3, x14, lsl #2]\t\n"

      "zip1 z16.s, z8.s, z10.s\t\n"
      "zip1 z17.s, z9.s, z11.s\t\n"
      "zip1 z18.s, z12.s, z14.s\t\n"
      "zip1 z19.s, z13.s, z15.s\t\n"

      "zip1 z20.s, z16.s, z17.s\t\n"

      "mov x7, #16\t\n"
      "add x8, x8, x4\t\n"
      "whilelo p2.s, w7, w0\t\n"
      "and p3.b, p0/z, p1.b, p2.b\t\n"
      "st1w { z20.s }, p3, [x3, x8, lsl #2]\t\n"

      "zip1 z21.s, z18.s, z19.s\t\n"
      "add x14, x14, x4\t\n"
      "and p3.b, p0/z, p4.b, p2.b\t\n"

      "st1w { z21.s }, p3, [x3, x14, lsl #2]\t\n"

      "zip2 z20.s, z16.s, z17.s\t\n"

      "mov x7, #20\t\n"
      "add x8, x8, x4\t\n"
      "whilelo p2.s, w7, w0\t\n"
      "and p3.b, p0/z, p1.b, p2.b\t\n"
      "st1w { z20.s }, p3, [x3, x8, lsl #2]\t\n"

      "zip2 z21.s, z18.s, z19.s\t\n"
      "add x14, x14, x4\t\n"
      "and p3.b, p0/z, p4.b, p2.b\t\n"

      "st1w { z21.s }, p3, [x3, x14, lsl #2]\t\n"

      "zip2 z16.s, z8.s, z10.s\t\n"
      "zip2 z17.s, z9.s, z11.s\t\n"
      "zip2 z18.s, z12.s, z14.s\t\n"
      "zip2 z19.s, z13.s, z15.s\t\n"

      "zip1 z20.s, z16.s, z17.s\t\n"

      "mov x7, #24\t\n"
      "add x8, x8, x4\t\n"
      "whilelo p2.s, w7, w0\t\n"
      "and p3.b, p0/z, p1.b, p2.b\t\n"
      "st1w { z20.s }, p3, [x3, x8, lsl #2]\t\n"

      "zip1 z21.s, z18.s, z19.s\t\n"
      "add x14, x14, x4\t\n"
      "and p3.b, p0/z, p4.b, p2.b\t\n"

      "st1w { z21.s }, p3, [x3, x14, lsl #2]\t\n"

      "zip2 z20.s, z16.s, z17.s\t\n"

      "mov x7, #28\t\n"
      "add x8, x8, x4\t\n"
      "whilelo p2.s, w7, w0\t\n"
      "and p3.b, p0/z, p1.b, p2.b\t\n"
      "st1w { z20.s }, p3, [x3, x8, lsl #2]\t\n"

      "zip2 z21.s, z18.s, z19.s\t\n"
      "add x14, x14, x4\t\n"
      "and p3.b, p0/z, p4.b, p2.b\t\n"

      "st1w { z21.s }, p3, [x3, x14, lsl #2]\t\n"

      :
      : [M] "i"(M),
        [N] "r"(N),
        [src] "r"(src),
        [ld_src] "r"(ld_src),
        [dst] "r"(dst),
        [ld_dst] "r"(ld_dst)
      : "memory",
        "cc",
        "x0",
        "x1",
        "x2",
        "x3",
        "x4",
        "x5",
        "x6",
        "x7",
        "x8",
        "x9",
        "x12",
        "x14",
        "z0",
        "z1",
        "z2",
        "z3",
        "z4",
        "z5",
        "z6",
        "z7",
        "z8",
        "z9",
        "z10",
        "z11",
        "z12",
        "z13",
        "z14",
        "z15",
        "z16",
        "z17",
        "z18",
        "z19",
        "z20",
        "z21",
        "p0",
        "p1",
        "p2",
        "p3",
        "p4");
}

#endif // __SVE__

} // namespace internal

} // namespace fbgemm
