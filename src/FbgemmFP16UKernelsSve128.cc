/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "./FbgemmFP16UKernelsSve128.h"

namespace fbgemm {

void NOINLINE gemmkernel_1x2_Sve128_fp16_fA0fB0fC0(GemmParamsFP16* gp) {
#if defined(__aarch64__) && __ARM_FEATURE_SVE
  asm volatile(

      "mov x0, %[gp]\t\n"

      "mov x1, #4\t\n"
      "mov x2, #8\t\n"

      // Lowest 128-bits
      "whilelo p1.s, w1, w2\t\n"

      // Copy parameters
      // k and A
      "ldp x11, x6, [x0]\t\n"
      "sub x11, x11, #1\t\n"

      // B
      "ldr x10, [x0, #16]\t\n"
      // beta
      "ld1rw { z12.s }, p1/z, [x0, #24]\t\n"
      // C
      "ldr x12, [x0, #32]\t\n"
      // b_block_cols
      "ldr x7, [x0, #48]\t\n"

      "fcmeq p3.s, p1/z, z12.s, #0.0\t\n"
      "fcmuo p4.s, p1/z, z12.s, z12.s\t\n"
      "nor p4.b, p1/z, p4.b, p3.b\t\n"

      "mov x3, #12\t\n"
      "mov x4, #16\t\n"
      "mov x5, #20\t\n"
      "mov x13, #24\t\n"
      "mov x15, #28\t\n"

      ".loop_outer:\t\n"
      "mov x9, x6\t\n"
      "lsl x14, x11, #2\t\n"

      "eor z5.d, z5.d, z5.d\t\n"
      "eor z25.d, z25.d, z25.d\t\n"
      "eor z6.d, z6.d, z6.d\t\n"
      "eor z26.d, z26.d, z26.d\t\n"
      "eor z7.d, z7.d, z7.d\t\n"
      "eor z27.d, z27.d, z27.d\t\n"
      "eor z8.d, z8.d, z8.d\t\n"
      "eor z28.d, z28.d, z28.d\t\n"
      "eor z9.d, z9.d, z9.d\t\n"
      "eor z29.d, z29.d, z29.d\t\n"
      "eor z10.d, z10.d, z10.d\t\n"
      "eor z30.d, z30.d, z30.d\t\n"

      // vcvtph2ps
      "ldr q11, [x10]\t\n"
      "fcvt z3.s, p1/m, z11.h\t\n"
      "fcvtlt z13.s, p1/m, z11.h\t\n"

      // vcvtph2ps
      "ldr q14, [x10, #16]\t\n"
      "add x10, x10, #32\t\n"
      "fcvt z4.s, p1/m, z14.h\t\n"
      "fcvtlt z14.s, p1/m, z14.h\t\n"

      // Setup values with beta multiplication
      "ld2w { z0.s, z1.s }, p4/z, [x12]\t\n"
      "fmul z0.s, p4/m, z0.s, z12.s\t\n"
      "fmul z1.s, p4/m, z1.s, z12.s\t\n"

      "ld2w { z20.s, z21.s }, p4/z, [x12, x2, lsl #2]\t\n"
      "fmul z20.s, p4/m, z20.s, z12.s\t\n"
      "fmul z21.s, p4/m, z21.s, z12.s\t\n"

      "ld1rw { z2.s }, p1/z, [x9]\t\n"

      "fmla z0.s, p1/m, z3.s, z2.s\t\n"
      "fmla z1.s, p1/m, z13.s, z2.s\t\n"
      "fmla z20.s, p1/m, z4.s, z2.s\t\n"
      "fmla z21.s, p1/m, z14.s, z2.s\t\n"

      "cbz x14, .dump_C\t\n"

      "add x9, x9, x1\t\n"
      "add x14, x14, x4\t\n"

      ".loop_inner:\t\n"

      "sub x14, x14, x4\t\n"

      // vcvtph2ps

      "ldr q11, [x10]\t\n"
      "fcvt z15.s, p1/m, z11.h\t\n"
      "fcvtlt z16.s, p1/m, z11.h\t\n"

      "ldr q14, [x10, #16]\t\n"
      "fcvt z4.s, p1/m, z14.h\t\n"
      "fcvtlt z14.s, p1/m, z14.h\t\n"

      // broadcast
      "ld1rw { z2.s }, p1/z, [x9]\t\n"

      // fma
      "fmla z0.s, p1/m, z15.s, z2.s\t\n"
      "fmla z1.s, p1/m, z16.s, z2.s\t\n"

      "fmla z20.s, p1/m, z4.s, z2.s\t\n"
      "fmla z21.s, p1/m, z14.s, z2.s\t\n"

      // next 32 bytes
      // update predicate
      "whilelo p2.s, x1, x14\t\n"

      // vcvtph2ps
      "ld1rqw { z11.s }, p2/z, [x10, x2, lsl #2]\t\n"
      "fcvt z3.s, p2/m, z11.h\t\n"
      "fcvtlt z13.s, p2/m, z11.h\t\n"

      "ld1rqw { z14.s }, p2/z, [x10, x3, lsl #2]\t\n"
      "fcvt z4.s, p2/m, z14.h\t\n"
      "fcvtlt z14.s, p2/m, z14.h\t\n"

      "ld1rw { z2.s }, p2/z, [x9, #4]\t\n"

      // fma
      "fmla z5.s, p2/m, z3.s, z2.s\t\n"
      "fmla z25.s, p2/m, z13.s, z2.s\t\n"

      "fmla z6.s, p2/m, z4.s, z2.s\t\n"
      "fmla z26.s, p2/m, z14.s, z2.s\t\n"

      // next 32 bytes

      // update predicates
      "whilelo p2.s, x2, x14\t\n"

      // vcvtph2ps
      "ld1rqw { z11.s }, p2/z, [x10, x4, lsl #2]\t\n"
      "fcvt z15.s, p2/m, z11.h\t\n"
      "fcvtlt z16.s, p2/m, z11.h\t\n"

      "ld1rqw { z14.s }, p2/z, [x10, x5, lsl #2]\t\n"
      "fcvt z4.s, p2/m, z14.h\t\n"
      "fcvtlt z14.s, p2/m, z14.h\t\n"

      "ld1rw { z2.s }, p2/z, [x9, #8]\t\n"

      // fma
      "fmla z7.s, p2/m, z15.s, z2.s\t\n"
      "fmla z27.s, p2/m, z16.s, z2.s\t\n"

      "fmla z8.s, p2/m, z4.s, z2.s\t\n"
      "fmla z28.s, p2/m, z14.s, z2.s\t\n"

      // next 32 bytes

      // update predicates
      "whilelo p2.s, x3, x14\t\n"

      // vcvtph2ps
      "ld1rqw { z11.s }, p2/z, [x10, x13, lsl #2]\t\n"
      "fcvt z3.s, p2/m, z11.h\t\n"
      "fcvtlt z13.s, p2/m, z11.h\t\n"

      "ld1rqw { z14.s }, p2/z, [x10, x15, lsl #2]\t\n"
      "add x10, x10, #128\t\n"
      "fcvt z4.s, p2/m, z14.h\t\n"
      "fcvtlt z14.s, p2/m, z14.h\t\n"

      "ld1rw { z2.s }, p2/z, [x9, #12]\t\n"

      "add x9, x9, x4\t\n"

      "cmp x14, x4\t\n"

      // fma
      "fmla z9.s, p2/m, z3.s, z2.s\t\n"
      "fmla z29.s, p2/m, z13.s, z2.s\t\n"

      "fmla z10.s, p2/m, z4.s, z2.s\t\n"
      "fmla z30.s, p2/m, z14.s, z2.s\t\n"

      "bgt .loop_inner\t\n"

      "sub x10, x10, #128\t\n"

      // hack works only whn processing 128-bit per register
      "add x10, x10, x14, lsl #3\t\n"

      "fadd z0.s, z0.s, z5.s\t\n"
      "fadd z1.s, z1.s, z25.s\t\n"
      "fadd z0.s, z0.s, z7.s\t\n"
      "fadd z1.s, z1.s, z27.s\t\n"
      "fadd z0.s, z0.s, z9.s\t\n"
      "fadd z1.s, z1.s, z29.s\t\n"

      "fadd z20.s, z20.s, z6.s\t\n"
      "fadd z21.s, z21.s, z26.s\t\n"
      "fadd z20.s, z20.s, z8.s\t\n"
      "fadd z21.s, z21.s, z28.s\t\n"
      "fadd z20.s, z20.s, z10.s\t\n"
      "fadd z21.s, z21.s, z30.s\t\n"

      // Dump C
      ".dump_C:\t\n"

      "st2w { z0.s, z1.s }, p1, [x12]\t\n"
      "st2w { z20.s, z21.s }, p1, [x12, x2, lsl #2]\t\n"
      "add x12, x12, #64\t\n"

      // next outer iteratio
      "subs x7, x7, #1\t\n"
      "bne .loop_outer\t\n"
      :
      : [gp] "r"(gp)
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
        "x9",
        "x10",
        "x11",
        "x12",
        "x13",
        "x14",
        "x15",
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
        "z20",
        "z21",
        "z25",
        "z26",
        "z27",
        "z28",
        "z29",
        "z30",
        "p1",
        "p2",
        "p3",
        "p4");
#endif
}

void NOINLINE gemmkernel_2x2_Sve128_fp16_fA0fB0fC0(GemmParamsFP16* gp) {
#if defined(__aarch64__) && __ARM_FEATURE_SVE
  asm volatile(

      "mov x0, %[gp]\t\n"

      "mov x1, #4\t\n"
      "mov x2, #8\t\n"

      // Lowest 128-bits
      "whilelo p1.s, w1, w2\t\n"

      // Copy parameters
      // k and A
      "ldp x11, x6, [x0]\t\n"
      "sub x11, x11, #1\t\n"
      // B
      "ldr x10, [x0, #16]\t\n"

      // beta
      "ld1rw { z12.s }, p1/z, [x0, #24]\t\n"

      // C and ldc
      "ldp x12, x8, [x0, #32]\t\n"
      // b_block_cols
      "ldr x7, [x0, #48]\t\n"

      "fcmeq p3.s, p1/z, z12.s, #0.0\t\n"
      "fcmuo p4.s, p1/z, z12.s, z12.s\t\n"
      "nor p4.b, p1/z, p4.b, p3.b\t\n"

      "add x8, x12, x8\t\n"

      "mov x3, #12\t\n"
      "mov x4, #16\t\n"
      "mov x5, #20\t\n"
      "mov x13, #24\t\n"
      "mov x15, #28\t\n"

      ".loop_outer2:\t\n"
      "mov x9, x6\t\n"
      "lsl x14, x11, #2\t\n"

      "eor z5.d, z5.d, z5.d\t\n"
      "eor z25.d, z25.d, z25.d\t\n"
      "eor z6.d, z6.d, z6.d\t\n"
      "eor z26.d, z26.d, z26.d\t\n"
      "eor z7.d, z7.d, z7.d\t\n"
      "eor z27.d, z27.d, z27.d\t\n"
      "eor z8.d, z8.d, z8.d\t\n"
      "eor z28.d, z28.d, z28.d\t\n"
      "eor z9.d, z9.d, z9.d\t\n"
      "eor z29.d, z29.d, z29.d\t\n"
      "eor z10.d, z10.d, z10.d\t\n"
      "eor z17.d, z17.d, z17.d\t\n"
      "eor z30.d, z30.d, z30.d\t\n"
      "eor z31.d, z31.d, z31.d\t\n"

      // vcvtph2ps
      "ldr q11, [x10]\t\n"
      "fcvt z3.s, p1/m, z11.h\t\n"
      "fcvtlt z13.s, p1/m, z11.h\t\n"

      // vcvtph2ps
      "ldr q14, [x10, #16]\t\n"
      "add x10, x10, #32\t\n"
      "fcvt z4.s, p1/m, z14.h\t\n"
      "fcvtlt z14.s, p1/m, z14.h\t\n"

      // Setup values with beta multiplication
      "ld2w { z0.s, z1.s }, p4/z, [x12]\t\n"
      "fmul z0.s, p4/m, z0.s, z12.s\t\n"
      "fmul z1.s, p4/m, z1.s, z12.s\t\n"

      "ld2w { z20.s, z21.s }, p4/z, [x12, x2, lsl #2]\t\n"
      "fmul z20.s, p4/m, z20.s, z12.s\t\n"
      "fmul z21.s, p4/m, z21.s, z12.s\t\n"

      "ld2w { z18.s, z19.s }, p4/z, [x8]\t\n"
      "fmul z18.s, p4/m, z18.s, z12.s\t\n"
      "fmul z19.s, p4/m, z19.s, z12.s\t\n"

      "ld2w { z22.s, z23.s }, p4/z, [x8, x2, lsl #2]\t\n"
      "fmul z22.s, p4/m, z22.s, z12.s\t\n"
      "fmul z23.s, p4/m, z23.s, z12.s\t\n"

      "ld1rw { z2.s }, p1/z, [x9]\t\n"

      "fmla z0.s, p1/m, z3.s, z2.s\t\n"
      "fmla z1.s, p1/m, z13.s, z2.s\t\n"
      "fmla z20.s, p1/m, z4.s, z2.s\t\n"
      "fmla z21.s, p1/m, z14.s, z2.s\t\n"

      "ld1rw { z24.s }, p1/z, [x9, #4]\t\n"

      "fmla z18.s, p1/m, z3.s, z24.s\t\n"
      "fmla z19.s, p1/m, z13.s, z24.s\t\n"
      "fmla z22.s, p1/m, z4.s, z24.s\t\n"
      "fmla z23.s, p1/m, z14.s, z24.s\t\n"

      "cbz x14, .dump_C2\t\n"

      "add x9, x9, x2\t\n"
      "add x14, x14, x4\t\n"

      ".loop_inner2:\t\n"

      "sub x14, x14, x4\t\n"

      // vcvtph2ps

      "ldr q11, [x10]\t\n"
      "fcvt z15.s, p1/m, z11.h\t\n"
      "fcvtlt z16.s, p1/m, z11.h\t\n"

      "ldr q14, [x10, #16]\t\n"
      "fcvt z4.s, p1/m, z14.h\t\n"
      "fcvtlt z14.s, p1/m, z14.h\t\n"

      // broadcast
      "ld1rw { z2.s }, p1/z, [x9]\t\n"

      // fma
      "fmla z0.s, p1/m, z15.s, z2.s\t\n"
      "fmla z1.s, p1/m, z16.s, z2.s\t\n"

      "fmla z20.s, p1/m, z4.s, z2.s\t\n"
      "fmla z21.s, p1/m, z14.s, z2.s\t\n"

      "ld1rw { z24.s }, p1/z, [x9, #4]\t\n"

      // fma
      "fmla z18.s, p1/m, z15.s, z24.s\t\n"
      "fmla z19.s, p1/m, z16.s, z24.s\t\n"

      "fmla z22.s, p1/m, z4.s, z24.s\t\n"
      "fmla z23.s, p1/m, z14.s, z24.s\t\n"

      // next 32 bytes
      // update predicate
      "whilelo p2.s, x1, x14\t\n"

      // vcvtph2ps
      "ld1rqw { z11.s }, p2/z, [x10, x2, lsl #2]\t\n"
      "fcvt z3.s, p2/m, z11.h\t\n"
      "fcvtlt z13.s, p2/m, z11.h\t\n"

      "ld1rqw { z14.s }, p2/z, [x10, x3, lsl #2]\t\n"
      "fcvt z4.s, p2/m, z14.h\t\n"
      "fcvtlt z14.s, p2/m, z14.h\t\n"

      "ld1rw { z2.s }, p2/z, [x9, #8]\t\n"

      // fma
      "fmla z5.s, p2/m, z3.s, z2.s\t\n"
      "fmla z25.s, p2/m, z13.s, z2.s\t\n"

      "fmla z6.s, p2/m, z4.s, z2.s\t\n"
      "fmla z26.s, p2/m, z14.s, z2.s\t\n"

      "ld1rw { z24.s }, p2/z, [x9, #12]\t\n"

      // fma
      "fmla z17.s, p2/m, z3.s, z24.s\t\n"
      "fmla z31.s, p2/m, z13.s, z24.s\t\n"

      "fmla z9.s, p2/m, z4.s, z24.s\t\n"
      "fmla z29.s, p2/m, z14.s, z24.s\t\n"

      // next 32 bytes

      // update predicates
      "whilelo p2.s, x2, x14\t\n"

      // vcvtph2ps
      "ld1rqw { z11.s }, p2/z, [x10, x4, lsl #2]\t\n"
      "fcvt z15.s, p2/m, z11.h\t\n"
      "fcvtlt z16.s, p2/m, z11.h\t\n"

      "ld1rqw { z14.s }, p2/z, [x10, x5, lsl #2]\t\n"
      "fcvt z4.s, p2/m, z14.h\t\n"
      "fcvtlt z14.s, p2/m, z14.h\t\n"

      "ld1rw { z2.s }, p2/z, [x9, #16]\t\n"

      // fma
      "fmla z7.s, p2/m, z15.s, z2.s\t\n"
      "fmla z27.s, p2/m, z16.s, z2.s\t\n"

      "fmla z8.s, p2/m, z4.s, z2.s\t\n"
      "fmla z28.s, p2/m, z14.s, z2.s\t\n"

      "ld1rw { z24.s }, p2/z, [x9, #20]\t\n"

      // fma
      "fmla z18.s, p2/m, z15.s, z24.s\t\n"
      "fmla z19.s, p2/m, z16.s, z24.s\t\n"

      "fmla z22.s, p2/m, z4.s, z24.s\t\n"
      "fmla z23.s, p2/m, z14.s, z24.s\t\n"

      // next 32 bytes

      // update predicates
      "whilelo p2.s, x3, x14\t\n"

      // vcvtph2ps
      "ld1rqw { z11.s }, p2/z, [x10, x13, lsl #2]\t\n"
      "fcvt z3.s, p2/m, z11.h\t\n"
      "fcvtlt z13.s, p2/m, z11.h\t\n"

      "ld1rqw { z14.s }, p2/z, [x10, x15, lsl #2]\t\n"
      "add x10, x10, #128\t\n"
      "fcvt z4.s, p2/m, z14.h\t\n"
      "fcvtlt z14.s, p2/m, z14.h\t\n"

      "ld1rw { z2.s }, p2/z, [x9, #24]\t\n"

      // fma
      "fmla z0.s, p2/m, z3.s, z2.s\t\n"
      "fmla z1.s, p2/m, z13.s, z2.s\t\n"

      "fmla z10.s, p2/m, z4.s, z2.s\t\n"
      "fmla z30.s, p2/m, z14.s, z2.s\t\n"

      "ld1rw { z24.s }, p2/z, [x9, #28]\t\n"

      "add x9, x9, #32\t\n"

      "cmp x14, x4\t\n"

      // fma
      "fmla z17.s, p2/m, z3.s, z24.s\t\n"
      "fmla z31.s, p2/m, z13.s, z24.s\t\n"

      "fmla z9.s, p2/m, z4.s, z24.s\t\n"
      "fmla z29.s, p2/m, z14.s, z24.s\t\n"

      "bgt .loop_inner2\t\n"

      "sub x10, x10, #128\t\n"

      // hack works only whn processing 128-bit per register
      "add x10, x10, x14, lsl #3\t\n"

      "fadd z0.s, z0.s, z5.s\t\n"
      "fadd z1.s, z1.s, z25.s\t\n"
      "fadd z0.s, z0.s, z7.s\t\n"
      "fadd z1.s, z1.s, z27.s\t\n"

      "fadd z20.s, z20.s, z6.s\t\n"
      "fadd z21.s, z21.s, z26.s\t\n"
      "fadd z20.s, z20.s, z8.s\t\n"
      "fadd z21.s, z21.s, z28.s\t\n"
      "fadd z20.s, z20.s, z10.s\t\n"
      "fadd z21.s, z21.s, z30.s\t\n"

      "fadd z18.s, z18.s, z17.s\t\n"
      "fadd z19.s, z19.s, z31.s\t\n"
      "fadd z22.s, z22.s, z9.s\t\n"
      "fadd z23.s, z23.s, z29.s\t\n"

      // Dump C
      ".dump_C2:\t\n"

      "st2w { z0.s, z1.s }, p1, [x12]\t\n"
      "st2w { z20.s, z21.s }, p1, [x12, x2, lsl #2]\t\n"

      "add x12, x12, #64\t\n"

      "st2w { z18.s, z19.s }, p1, [x8]\t\n"
      "st2w { z22.s, z23.s }, p1, [x8, x2, lsl #2]\t\n"

      "add x8, x8, #64\t\n"

      // next outer iteratio
      "subs x7, x7, #1\t\n"
      "bne .loop_outer2\t\n"
      :
      : [gp] "r"(gp)
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
        "x10",
        "x11",
        "x12",
        "x13",
        "x14",
        "x15",
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
        "z22",
        "z23",
        "z24",
        "z25",
        "z26",
        "z27",
        "z28",
        "z29",
        "z30",
        "z31",
        "p1",
        "p2",
        "p3",
        "p4");
#endif
}

void NOINLINE gemmkernel_3x2_Sve128_fp16_fA0fB0fC0(GemmParamsFP16* gp) {
#if defined(__aarch64__) && __ARM_FEATURE_SVE
  asm volatile(

      "mov x0, %[gp]\t\n"

      "mov x1, #4\t\n"
      "mov x2, #8\t\n"

      // Lowest 128-bits
      "whilelo p1.s, w1, w2\t\n"

      // Copy parameters
      // k and A
      "ldp x11, x6, [x0]\t\n"
      "sub x11, x11, #1\t\n"
      // B
      "ldr x10, [x0, #16]\t\n"

      // beta
      "ld1rw { z12.s }, p1/z, [x0, #24]\t\n"

      // C and ldc
      "ldp x12, x8, [x0, #32]\t\n"
      // b_block_cols
      "ldr x7, [x0, #48]\t\n"

      "fcmeq p3.s, p1/z, z12.s, #0.0\t\n"
      "fcmuo p4.s, p1/z, z12.s, z12.s\t\n"
      "nor p4.b, p1/z, p4.b, p3.b\t\n"

      "add x0, x12, x8, lsl #1\t\n"
      "add x8, x12, x8\t\n"

      "mov x3, #12\t\n"
      "mov x4, #16\t\n"
      "mov x5, #20\t\n"
      "mov x13, #24\t\n"
      "mov x15, #28\t\n"

      ".loop_outer3:\t\n"
      "mov x9, x6\t\n"
      "lsl x14, x11, #2\t\n"

      "eor z5.d, z5.d, z5.d\t\n"
      "eor z25.d, z25.d, z25.d\t\n"
      "eor z6.d, z6.d, z6.d\t\n"
      "eor z26.d, z26.d, z26.d\t\n"
      "eor z7.d, z7.d, z7.d\t\n"
      "eor z27.d, z27.d, z27.d\t\n"
      "eor z8.d, z8.d, z8.d\t\n"
      "eor z9.d, z9.d, z9.d\t\n"
      "eor z10.d, z10.d, z10.d\t\n"
      "eor z17.d, z17.d, z17.d\t\n"

      // vcvtph2ps
      "ldr q11, [x10]\t\n"
      "fcvt z3.s, p1/m, z11.h\t\n"
      "fcvtlt z13.s, p1/m, z11.h\t\n"

      // vcvtph2ps
      "ldr q14, [x10, #16]\t\n"
      "add x10, x10, #32\t\n"
      "fcvt z4.s, p1/m, z14.h\t\n"
      "fcvtlt z14.s, p1/m, z14.h\t\n"

      // Setup values with beta multiplication
      "ld2w { z0.s, z1.s }, p4/z, [x12]\t\n"
      "fmul z0.s, p4/m, z0.s, z12.s\t\n"
      "fmul z1.s, p4/m, z1.s, z12.s\t\n"

      "ld2w { z20.s, z21.s }, p4/z, [x12, x2, lsl #2]\t\n"
      "fmul z20.s, p4/m, z20.s, z12.s\t\n"
      "fmul z21.s, p4/m, z21.s, z12.s\t\n"

      "ld2w { z18.s, z19.s }, p4/z, [x8]\t\n"
      "fmul z18.s, p4/m, z18.s, z12.s\t\n"
      "fmul z19.s, p4/m, z19.s, z12.s\t\n"

      "ld2w { z22.s, z23.s }, p4/z, [x8, x2, lsl #2]\t\n"
      "fmul z22.s, p4/m, z22.s, z12.s\t\n"
      "fmul z23.s, p4/m, z23.s, z12.s\t\n"

      "ld2w { z28.s, z29.s }, p4/z, [x0]\t\n"
      "fmul z28.s, p4/m, z28.s, z12.s\t\n"
      "fmul z29.s, p4/m, z29.s, z12.s\t\n"

      "ld2w { z30.s, z31.s }, p4/z, [x0, x2, lsl #2]\t\n"
      "fmul z30.s, p4/m, z30.s, z12.s\t\n"
      "fmul z31.s, p4/m, z31.s, z12.s\t\n"

      "ld1rw { z24.s }, p1/z, [x9]\t\n"

      "fmla z0.s, p1/m, z3.s, z24.s\t\n"
      "fmla z1.s, p1/m, z13.s, z24.s\t\n"
      "fmla z20.s, p1/m, z4.s, z24.s\t\n"
      "fmla z21.s, p1/m, z14.s, z24.s\t\n"

      "ld1rw { z2.s }, p1/z, [x9, #4]\t\n"

      "fmla z18.s, p1/m, z3.s, z2.s\t\n"
      "fmla z19.s, p1/m, z13.s, z2.s\t\n"
      "fmla z22.s, p1/m, z4.s, z2.s\t\n"
      "fmla z23.s, p1/m, z14.s, z2.s\t\n"

      "ld1rw { z24.s }, p1/z, [x9, #8]\t\n"

      "fmla z28.s, p1/m, z3.s, z24.s\t\n"
      "fmla z29.s, p1/m, z13.s, z24.s\t\n"
      "fmla z30.s, p1/m, z4.s, z24.s\t\n"
      "fmla z31.s, p1/m, z14.s, z24.s\t\n"

      "cbz x14, .dump_C3\t\n"

      "add x9, x9, x3\t\n"
      "add x14, x14, x4\t\n"

      ".loop_inner3:\t\n"

      "sub x14, x14, x4\t\n"

      // vcvtph2ps

      "ldr q11, [x10]\t\n"
      "fcvt z15.s, p1/m, z11.h\t\n"
      "fcvtlt z16.s, p1/m, z11.h\t\n"

      "ldr q14, [x10, #16]\t\n"
      "fcvt z4.s, p1/m, z14.h\t\n"
      "fcvtlt z14.s, p1/m, z14.h\t\n"

      // broadcast
      "ld1rw { z2.s }, p1/z, [x9]\t\n"

      // fma
      "fmla z0.s, p1/m, z15.s, z2.s\t\n"
      "fmla z1.s, p1/m, z16.s, z2.s\t\n"

      "fmla z20.s, p1/m, z4.s, z2.s\t\n"
      "fmla z21.s, p1/m, z14.s, z2.s\t\n"

      "ld1rw { z24.s }, p1/z, [x9, #4]\t\n"

      // fma
      "fmla z18.s, p1/m, z15.s, z24.s\t\n"
      "fmla z19.s, p1/m, z16.s, z24.s\t\n"

      "fmla z22.s, p1/m, z4.s, z24.s\t\n"
      "fmla z23.s, p1/m, z14.s, z24.s\t\n"

      "ld1rw { z2.s }, p1/z, [x9, #8]\t\n"

      "fmla z28.s, p1/m, z15.s, z2.s\t\n"
      "fmla z29.s, p1/m, z16.s, z2.s\t\n"

      "fmla z30.s, p1/m, z4.s, z2.s\t\n"
      "fmla z31.s, p1/m, z14.s, z2.s\t\n"

      // next 32 bytes
      // update predicate
      "whilelo p2.s, x1, x14\t\n"

      // vcvtph2ps
      "ld1rqw { z11.s }, p2/z, [x10, x2, lsl #2]\t\n"
      "fcvt z3.s, p2/m, z11.h\t\n"
      "fcvtlt z13.s, p2/m, z11.h\t\n"

      "ld1rqw { z14.s }, p2/z, [x10, x3, lsl #2]\t\n"
      "fcvt z4.s, p2/m, z14.h\t\n"
      "fcvtlt z14.s, p2/m, z14.h\t\n"

      "ld1rw { z24.s }, p2/z, [x9, #12]\t\n"

      // fma
      "fmla z5.s, p2/m, z3.s, z24.s\t\n"
      "fmla z25.s, p2/m, z13.s, z24.s\t\n"

      "fmla z6.s, p2/m, z4.s, z24.s\t\n"
      "fmla z26.s, p2/m, z14.s, z24.s\t\n"

      "ld1rw { z2.s }, p2/z, [x9, #16]\t\n"

      // fma
      "fmla z17.s, p2/m, z3.s, z2.s\t\n"
      "fmla z10.s, p2/m, z13.s, z2.s\t\n"

      "fmla z8.s, p2/m, z4.s, z2.s\t\n"
      "fmla z9.s, p2/m, z14.s, z2.s\t\n"

      "ld1rw { z24.s }, p2/z, [x9, #20]\t\n"

      "fmla z7.s, p2/m, z3.s, z24.s\t\n"
      "fmla z27.s, p2/m, z13.s, z24.s\t\n"

      "fmla z30.s, p2/m, z4.s, z24.s\t\n"
      "fmla z31.s, p2/m, z14.s, z24.s\t\n"

      // next 32 bytes

      // update predicates
      "whilelo p2.s, x2, x14\t\n"

      // vcvtph2ps
      "ld1rqw { z11.s }, p2/z, [x10, x4, lsl #2]\t\n"
      "fcvt z15.s, p2/m, z11.h\t\n"
      "fcvtlt z16.s, p2/m, z11.h\t\n"

      "ld1rqw { z14.s }, p2/z, [x10, x5, lsl #2]\t\n"
      "fcvt z4.s, p2/m, z14.h\t\n"
      "fcvtlt z14.s, p2/m, z14.h\t\n"

      "ld1rw { z2.s }, p2/z, [x9, #24]\t\n"

      // fma
      "fmla z0.s, p2/m, z15.s, z2.s\t\n"
      "fmla z1.s, p2/m, z16.s, z2.s\t\n"

      "fmla z20.s, p2/m, z4.s, z2.s\t\n"
      "fmla z21.s, p2/m, z14.s, z2.s\t\n"

      "ld1rw { z24.s }, p2/z, [x9, #28]\t\n"

      // fma
      "fmla z18.s, p2/m, z15.s, z24.s\t\n"
      "fmla z19.s, p2/m, z16.s, z24.s\t\n"

      "fmla z22.s, p2/m, z4.s, z24.s\t\n"
      "fmla z23.s, p2/m, z14.s, z24.s\t\n"

      "ld1rw { z2.s }, p2/z, [x9, #32]\t\n"

      "fmla z28.s, p2/m, z15.s, z2.s\t\n"
      "fmla z29.s, p2/m, z16.s, z2.s\t\n"

      "fmla z30.s, p2/m, z4.s, z2.s\t\n"
      "fmla z31.s, p2/m, z14.s, z2.s\t\n"

      // next 32 bytes

      // update predicates
      "whilelo p2.s, x3, x14\t\n"

      // vcvtph2ps
      "ld1rqw { z11.s }, p2/z, [x10, x13, lsl #2]\t\n"
      "fcvt z3.s, p2/m, z11.h\t\n"
      "fcvtlt z13.s, p2/m, z11.h\t\n"

      "ld1rqw { z14.s }, p2/z, [x10, x15, lsl #2]\t\n"
      "add x10, x10, #128\t\n"
      "fcvt z4.s, p2/m, z14.h\t\n"
      "fcvtlt z14.s, p2/m, z14.h\t\n"

      "ld1rw { z24.s }, p2/z, [x9, #36]\t\n"

      // fma
      "fmla z5.s, p2/m, z3.s, z24.s\t\n"
      "fmla z25.s, p2/m, z13.s, z24.s\t\n"

      "fmla z6.s, p2/m, z4.s, z24.s\t\n"
      "fmla z26.s, p2/m, z14.s, z24.s\t\n"

      "ld1rw { z2.s }, p2/z, [x9, #40]\t\n"

      "fmla z17.s, p2/m, z3.s, z2.s\t\n"
      "fmla z10.s, p2/m, z13.s, z2.s\t\n"

      "fmla z8.s, p2/m, z4.s, z2.s\t\n"
      "fmla z9.s, p2/m, z14.s, z2.s\t\n"

      "ld1rw { z24.s }, p2/z, [x9, #44]\t\n"

      "add x9, x9, #48\t\n"

      "cmp x14, x4\t\n"

      "fmla z7.s, p2/m, z3.s, z24.s\t\n"
      "fmla z27.s, p2/m, z13.s, z24.s\t\n"

      "fmla z30.s, p2/m, z4.s, z24.s\t\n"
      "fmla z31.s, p2/m, z14.s, z24.s\t\n"

      // fma

      "bgt .loop_inner3\t\n"

      "sub x10, x10, #128\t\n"

      // hack works only whn processing 128-bit per register
      "add x10, x10, x14, lsl #3\t\n"

      "fadd z0.s, z0.s, z5.s\t\n"
      "fadd z1.s, z1.s, z25.s\t\n"

      "fadd z20.s, z20.s, z6.s\t\n"
      "fadd z21.s, z21.s, z26.s\t\n"

      "fadd z18.s, z18.s, z17.s\t\n"
      "fadd z19.s, z19.s, z10.s\t\n"

      "fadd z22.s, z22.s, z8.s\t\n"
      "fadd z23.s, z23.s, z9.s\t\n"

      "fadd z28.s, z28.s, z7.s\t\n"
      "fadd z29.s, z29.s, z27.s\t\n"

      // Dump C
      ".dump_C3:\t\n"

      "st2w { z0.s, z1.s }, p1, [x12]\t\n"
      "st2w { z20.s, z21.s }, p1, [x12, x2, lsl #2]\t\n"

      "add x12, x12, #64\t\n"

      "st2w { z18.s, z19.s }, p1, [x8]\t\n"
      "st2w { z22.s, z23.s }, p1, [x8, x2, lsl #2]\t\n"

      "add x8, x8, #64\t\n"

      "st2w { z28.s, z29.s }, p1, [x0]\t\n"
      "st2w { z30.s, z31.s }, p1, [x0, x2, lsl #2]\t\n"

      "add x0, x0, #64\t\n"

      // next outer iteratio
      "subs x7, x7, #1\t\n"
      "bne .loop_outer3\t\n"
      :
      : [gp] "r"(gp)
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
        "x10",
        "x11",
        "x12",
        "x13",
        "x14",
        "x15",
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
        "z22",
        "z23",
        "z24",
        "z25",
        "z26",
        "z27",
        "z28",
        "z29",
        "z30",
        "z31",
        "p1",
        "p2",
        "p3",
        "p4");
#endif
}

void NOINLINE gemmkernel_4x2_Sve128_fp16_fA0fB0fC0(GemmParamsFP16* gp) {
#if defined(__aarch64__) && __ARM_FEATURE_SVE
  asm volatile(

      "mov x0, %[gp]\t\n"

      "mov x1, #4\t\n"
      "mov x2, #8\t\n"

      // Lowest 128-bits
      "whilelo p1.s, w1, w2\t\n"

      // Copy parameters
      // k and A
      "ldp x11, x6, [x0]\t\n"
      "sub x11, x11, #1\t\n"
      // B
      "ldr x10, [x0, #16]\t\n"

      // beta
      "ld1rw { z12.s }, p1/z, [x0, #24]\t\n"

      // C and ldc
      "ldp x12, x8, [x0, #32]\t\n"
      // b_block_cols
      "ldr x7, [x0, #48]\t\n"

      "fcmeq p3.s, p1/z, z12.s, #0.0\t\n"
      "fcmuo p4.s, p1/z, z12.s, z12.s\t\n"
      "nor p4.b, p1/z, p4.b, p3.b\t\n"

      "add x0, x12, x8, lsl #1\t\n"
      "add x16, x0, x8\t\n"
      "add x8, x12, x8\t\n"

      "mov x3, #12\t\n"
      "mov x4, #16\t\n"
      "mov x5, #20\t\n"
      "mov x13, #24\t\n"
      "mov x15, #28\t\n"

      ".loop_outer4:\t\n"
      "mov x9, x6\t\n"
      "lsl x14, x11, #2\t\n"

      "eor z5.d, z5.d, z5.d\t\n"
      "eor z25.d, z25.d, z25.d\t\n"
      "eor z26.d, z26.d, z26.d\t\n"
      "eor z27.d, z27.d, z27.d\t\n"
      "eor z10.d, z10.d, z10.d\t\n"
      "eor z17.d, z17.d, z17.d\t\n"

      // vcvtph2ps
      "ldr q11, [x10]\t\n"
      "fcvt z3.s, p1/m, z11.h\t\n"
      "fcvtlt z13.s, p1/m, z11.h\t\n"

      // vcvtph2ps
      "ldr q14, [x10, #16]\t\n"
      "add x10, x10, #32\t\n"
      "fcvt z4.s, p1/m, z14.h\t\n"
      "fcvtlt z14.s, p1/m, z14.h\t\n"

      // Setup values with beta multiplication
      "ld2w { z0.s, z1.s }, p4/z, [x12]\t\n"
      "fmul z0.s, p4/m, z0.s, z12.s\t\n"
      "fmul z1.s, p4/m, z1.s, z12.s\t\n"

      "ld2w { z20.s, z21.s }, p4/z, [x12, x2, lsl #2]\t\n"
      "fmul z20.s, p4/m, z20.s, z12.s\t\n"
      "fmul z21.s, p4/m, z21.s, z12.s\t\n"

      "ld2w { z18.s, z19.s }, p4/z, [x8]\t\n"
      "fmul z18.s, p4/m, z18.s, z12.s\t\n"
      "fmul z19.s, p4/m, z19.s, z12.s\t\n"

      "ld2w { z22.s, z23.s }, p4/z, [x8, x2, lsl #2]\t\n"
      "fmul z22.s, p4/m, z22.s, z12.s\t\n"
      "fmul z23.s, p4/m, z23.s, z12.s\t\n"

      "ld2w { z28.s, z29.s }, p4/z, [x0]\t\n"
      "fmul z28.s, p4/m, z28.s, z12.s\t\n"
      "fmul z29.s, p4/m, z29.s, z12.s\t\n"

      "ld2w { z30.s, z31.s }, p4/z, [x0, x2, lsl #2]\t\n"
      "fmul z30.s, p4/m, z30.s, z12.s\t\n"
      "fmul z31.s, p4/m, z31.s, z12.s\t\n"

      "ld2w { z6.s, z7.s }, p4/z, [x16]\t\n"
      "fmul z6.s, p4/m, z6.s, z12.s\t\n"
      "fmul z7.s, p4/m, z7.s, z12.s\t\n"

      "ld2w { z8.s, z9.s }, p4/z, [x16, x2, lsl #2]\t\n"
      "fmul z8.s, p4/m, z8.s, z12.s\t\n"
      "fmul z9.s, p4/m, z9.s, z12.s\t\n"

      "ld1rw { z2.s }, p1/z, [x9]\t\n"

      "fmla z0.s, p1/m, z3.s, z2.s\t\n"
      "fmla z1.s, p1/m, z13.s, z2.s\t\n"
      "fmla z20.s, p1/m, z4.s, z2.s\t\n"
      "fmla z21.s, p1/m, z14.s, z2.s\t\n"

      "ld1rw { z24.s }, p1/z, [x9, #4]\t\n"

      "fmla z18.s, p1/m, z3.s, z24.s\t\n"
      "fmla z19.s, p1/m, z13.s, z24.s\t\n"
      "fmla z22.s, p1/m, z4.s, z24.s\t\n"
      "fmla z23.s, p1/m, z14.s, z24.s\t\n"

      "ld1rw { z2.s }, p1/z, [x9, #8]\t\n"

      "fmla z28.s, p1/m, z3.s, z2.s\t\n"
      "fmla z29.s, p1/m, z13.s, z2.s\t\n"
      "fmla z30.s, p1/m, z4.s, z2.s\t\n"
      "fmla z31.s, p1/m, z14.s, z2.s\t\n"

      "ld1rw { z24.s }, p1/z, [x9, #12]\t\n"

      "fmla z6.s, p1/m, z3.s, z24.s\t\n"
      "fmla z7.s, p1/m, z13.s, z24.s\t\n"
      "fmla z8.s, p1/m, z4.s, z24.s\t\n"
      "fmla z9.s, p1/m, z14.s, z24.s\t\n"

      "cbz x14, .dump_C4\t\n"

      "add x9, x9, x4\t\n"
      "add x14, x14, x4\t\n"

      ".loop_inner4:\t\n"

      "sub x14, x14, x4\t\n"

      // vcvtph2ps

      "ldr q11, [x10]\t\n"
      "fcvt z15.s, p1/m, z11.h\t\n"
      "fcvtlt z16.s, p1/m, z11.h\t\n"

      "ldr q14, [x10, #16]\t\n"
      "fcvt z4.s, p1/m, z14.h\t\n"
      "fcvtlt z14.s, p1/m, z14.h\t\n"

      // broadcast
      "ld1rw { z2.s }, p1/z, [x9]\t\n"

      // fma
      "fmla z0.s, p1/m, z15.s, z2.s\t\n"
      "fmla z1.s, p1/m, z16.s, z2.s\t\n"

      "fmla z20.s, p1/m, z4.s, z2.s\t\n"
      "fmla z21.s, p1/m, z14.s, z2.s\t\n"

      "ld1rw { z24.s }, p1/z, [x9, #4]\t\n"

      // fma
      "fmla z18.s, p1/m, z15.s, z24.s\t\n"
      "fmla z19.s, p1/m, z16.s, z24.s\t\n"

      "fmla z22.s, p1/m, z4.s, z24.s\t\n"
      "fmla z23.s, p1/m, z14.s, z24.s\t\n"

      "ld1rw { z2.s }, p1/z, [x9, #8]\t\n"

      "fmla z28.s, p1/m, z15.s, z2.s\t\n"
      "fmla z29.s, p1/m, z16.s, z2.s\t\n"

      "fmla z30.s, p1/m, z4.s, z2.s\t\n"
      "fmla z31.s, p1/m, z14.s, z2.s\t\n"

      "ld1rw { z24.s }, p1/z, [x9, #12]\t\n"

      "fmla z6.s, p1/m, z15.s, z24.s\t\n"
      "fmla z7.s, p1/m, z16.s, z24.s\t\n"

      "fmla z8.s, p1/m, z4.s, z24.s\t\n"
      "fmla z9.s, p1/m, z14.s, z24.s\t\n"

      // next 32 bytes
      // update predicate
      "whilelo p2.s, x1, x14\t\n"

      // vcvtph2ps
      "ld1rqw { z11.s }, p2/z, [x10, x2, lsl #2]\t\n"
      "fcvt z3.s, p2/m, z11.h\t\n"
      "fcvtlt z13.s, p2/m, z11.h\t\n"

      "ld1rqw { z14.s }, p2/z, [x10, x3, lsl #2]\t\n"
      "fcvt z4.s, p2/m, z14.h\t\n"
      "fcvtlt z14.s, p2/m, z14.h\t\n"

      "ld1rw { z2.s }, p2/z, [x9, #16]\t\n"

      // fma
      "fmla z5.s, p2/m, z3.s, z2.s\t\n"
      "fmla z25.s, p2/m, z13.s, z2.s\t\n"

      "fmla z26.s, p2/m, z4.s, z2.s\t\n"
      "fmla z27.s, p2/m, z14.s, z2.s\t\n"

      "ld1rw { z24.s }, p2/z, [x9, #20]\t\n"

      // fma
      "fmla z17.s, p2/m, z3.s, z24.s\t\n"
      "fmla z10.s, p2/m, z13.s, z24.s\t\n"

      "fmla z22.s, p2/m, z4.s, z24.s\t\n"
      "fmla z23.s, p2/m, z14.s, z24.s\t\n"

      "ld1rw { z2.s }, p2/z, [x9, #24]\t\n"

      "fmla z28.s, p2/m, z3.s, z2.s\t\n"
      "fmla z29.s, p2/m, z13.s, z2.s\t\n"

      "fmla z30.s, p2/m, z4.s, z2.s\t\n"
      "fmla z31.s, p2/m, z14.s, z2.s\t\n"

      "ld1rw { z24.s }, p2/z, [x9, #28]\t\n"

      // fma
      "fmla z6.s, p2/m, z3.s, z24.s\t\n"
      "fmla z7.s, p2/m, z13.s, z24.s\t\n"

      "fmla z8.s, p2/m, z4.s, z24.s\t\n"
      "fmla z9.s, p2/m, z14.s, z24.s\t\n"

      // next 32 bytes

      // update predicates
      "whilelo p2.s, x2, x14\t\n"

      // vcvtph2ps
      "ld1rqw { z11.s }, p2/z, [x10, x4, lsl #2]\t\n"
      "fcvt z15.s, p2/m, z11.h\t\n"
      "fcvtlt z16.s, p2/m, z11.h\t\n"

      "ld1rqw { z14.s }, p2/z, [x10, x5, lsl #2]\t\n"
      "fcvt z4.s, p2/m, z14.h\t\n"
      "fcvtlt z14.s, p2/m, z14.h\t\n"

      "ld1rw { z2.s }, p2/z, [x9, #32]\t\n"

      // fma
      "fmla z0.s, p2/m, z15.s, z2.s\t\n"
      "fmla z1.s, p2/m, z16.s, z2.s\t\n"

      "fmla z20.s, p2/m, z4.s, z2.s\t\n"
      "fmla z21.s, p2/m, z14.s, z2.s\t\n"

      "ld1rw { z24.s }, p2/z, [x9, #36]\t\n"

      // fma
      "fmla z18.s, p2/m, z15.s, z24.s\t\n"
      "fmla z19.s, p2/m, z16.s, z24.s\t\n"

      "fmla z22.s, p2/m, z4.s, z24.s\t\n"
      "fmla z23.s, p2/m, z14.s, z24.s\t\n"

      "ld1rw { z2.s }, p2/z, [x9, #40]\t\n"

      "fmla z28.s, p2/m, z15.s, z2.s\t\n"
      "fmla z29.s, p2/m, z16.s, z2.s\t\n"

      "fmla z30.s, p2/m, z4.s, z2.s\t\n"
      "fmla z31.s, p2/m, z14.s, z2.s\t\n"

      "ld1rw { z24.s }, p2/z, [x9, #44]\t\n"

      // fma
      "fmla z6.s, p2/m, z15.s, z24.s\t\n"
      "fmla z7.s, p2/m, z16.s, z24.s\t\n"

      "fmla z8.s, p2/m, z4.s, z24.s\t\n"
      "fmla z9.s, p2/m, z14.s, z24.s\t\n"

      // next 32 bytes

      // update predicates
      "whilelo p2.s, x3, x14\t\n"

      // vcvtph2ps
      "ld1rqw { z11.s }, p2/z, [x10, x13, lsl #2]\t\n"
      "fcvt z3.s, p2/m, z11.h\t\n"
      "fcvtlt z13.s, p2/m, z11.h\t\n"

      "ld1rqw { z11.s }, p2/z, [x10, x15, lsl #2]\t\n"
      "add x10, x10, #128\t\n"
      "fcvt z4.s, p2/m, z11.h\t\n"
      "fcvtlt z14.s, p2/m, z11.h\t\n"

      "ld1rw { z2.s }, p2/z, [x9, #48]\t\n"

      // fma
      "fmla z5.s, p2/m, z3.s, z2.s\t\n"
      "fmla z25.s, p2/m, z13.s, z2.s\t\n"

      "fmla z26.s, p2/m, z4.s, z2.s\t\n"
      "fmla z27.s, p2/m, z14.s, z2.s\t\n"

      "ld1rw { z24.s }, p2/z, [x9, #52]\t\n"

      "fmla z17.s, p2/m, z3.s, z24.s\t\n"
      "fmla z10.s, p2/m, z13.s, z24.s\t\n"

      "fmla z22.s, p2/m, z4.s, z24.s\t\n"
      "fmla z23.s, p2/m, z14.s, z24.s\t\n"

      "ld1rw { z2.s }, p2/z, [x9, #56]\t\n"

      "fmla z28.s, p2/m, z3.s, z2.s\t\n"
      "fmla z29.s, p2/m, z13.s, z2.s\t\n"

      "fmla z30.s, p2/m, z4.s, z2.s\t\n"
      "fmla z31.s, p2/m, z14.s, z2.s\t\n"

      "ld1rw { z24.s }, p2/z, [x9, #60]\t\n"

      "add x9, x9, #64\t\n"

      "cmp x14, x4\t\n"

      "fmla z6.s, p2/m, z3.s, z24.s\t\n"
      "fmla z7.s, p2/m, z13.s, z24.s\t\n"

      "fmla z8.s, p2/m, z4.s, z24.s\t\n"
      "fmla z9.s, p2/m, z14.s, z24.s\t\n"

      // fma
      "bgt .loop_inner4\t\n"

      "sub x10, x10, #128\t\n"

      // hack works only whn processing 128-bit per register
      "add x10, x10, x14, lsl #3\t\n"

      "fadd z0.s, z0.s, z5.s\t\n"
      "fadd z1.s, z1.s, z25.s\t\n"

      "fadd z20.s, z20.s, z26.s\t\n"
      "fadd z21.s, z21.s, z27.s\t\n"

      "fadd z18.s, z18.s, z17.s\t\n"
      "fadd z19.s, z19.s, z10.s\t\n"

      // Dump C
      ".dump_C4:\t\n"

      "st2w { z0.s, z1.s }, p1, [x12]\t\n"
      "st2w { z20.s, z21.s }, p1, [x12, x2, lsl #2]\t\n"

      "add x12, x12, #64\t\n"

      "st2w { z18.s, z19.s }, p1, [x8]\t\n"
      "st2w { z22.s, z23.s }, p1, [x8, x2, lsl #2]\t\n"

      "add x8, x8, #64\t\n"

      "st2w { z28.s, z29.s }, p1, [x0]\t\n"
      "st2w { z30.s, z31.s }, p1, [x0, x2, lsl #2]\t\n"

      "add x0, x0, #64\t\n"

      "st2w { z6.s, z7.s }, p1, [x16]\t\n"
      "st2w { z8.s, z9.s }, p1, [x16, x2, lsl #2]\t\n"

      "add x16, x16, #64\t\n"

      // next outer iteratio
      "subs x7, x7, #1\t\n"
      "bne .loop_outer4\t\n"
      :
      : [gp] "r"(gp)
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
        "x10",
        "x11",
        "x12",
        "x13",
        "x14",
        "x15",
        "x16",
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
        "z22",
        "z23",
        "z24",
        "z25",
        "z26",
        "z27",
        "z28",
        "z29",
        "z30",
        "z31",
        "p1",
        "p2",
        "p3",
        "p4");
#endif
}

void NOINLINE gemmkernel_5x2_Sve128_fp16_fA0fB0fC0(GemmParamsFP16* gp) {
#if defined(__aarch64__) && __ARM_FEATURE_SVE
  asm volatile(

      "mov x0, %[gp]\t\n"

      "mov x1, #4\t\n"
      "mov x2, #8\t\n"

      // Lowest 128-bits
      "whilelo p1.s, w1, w2\t\n"

      // Copy parameters
      // k and A
      "ldp x11, x6, [x0]\t\n"
      "sub x11, x11, #1\t\n"
      // B
      "ldr x10, [x0, #16]\t\n"

      // beta
      "ld1rw { z12.s }, p1/z, [x0, #24]\t\n"

      // C and ldc
      "ldp x12, x8, [x0, #32]\t\n"
      // b_block_cols
      "ldr x7, [x0, #48]\t\n"

      "fcmeq p3.s, p1/z, z12.s, #0.0\t\n"
      "fcmuo p4.s, p1/z, z12.s, z12.s\t\n"
      "nor p4.b, p1/z, p4.b, p3.b\t\n"

      "add x0, x12, x8, lsl #1\t\n"
      "add x17, x12, x8, lsl #2\t\n"
      "add x16, x0, x8\t\n"
      "add x8, x12, x8\t\n"

      "mov x3, #12\t\n"
      "mov x4, #16\t\n"
      "mov x5, #20\t\n"
      "mov x13, #24\t\n"
      "mov x15, #28\t\n"

      ".loop_outer5:\t\n"
      "mov x9, x6\t\n"
      "lsl x14, x11, #2\t\n"

      "eor z5.d, z5.d, z5.d\t\n"
      "eor z17.d, z17.d, z17.d\t\n"

      // vcvtph2ps
      "ldr q11, [x10]\t\n"
      "fcvt z3.s, p1/m, z11.h\t\n"
      "fcvtlt z13.s, p1/m, z11.h\t\n"

      // vcvtph2ps
      "ldr q14, [x10, #16]\t\n"
      "add x10, x10, #32\t\n"
      "fcvt z4.s, p1/m, z14.h\t\n"
      "fcvtlt z14.s, p1/m, z14.h\t\n"

      // Setup values with beta multiplication
      "ld2w { z0.s, z1.s }, p4/z, [x12]\t\n"
      "fmul z0.s, p4/m, z0.s, z12.s\t\n"
      "fmul z1.s, p4/m, z1.s, z12.s\t\n"

      "ld2w { z20.s, z21.s }, p4/z, [x12, x2, lsl #2]\t\n"
      "fmul z20.s, p4/m, z20.s, z12.s\t\n"
      "fmul z21.s, p4/m, z21.s, z12.s\t\n"

      "ld2w { z18.s, z19.s }, p4/z, [x8]\t\n"
      "fmul z18.s, p4/m, z18.s, z12.s\t\n"
      "fmul z19.s, p4/m, z19.s, z12.s\t\n"

      "ld2w { z22.s, z23.s }, p4/z, [x8, x2, lsl #2]\t\n"
      "fmul z22.s, p4/m, z22.s, z12.s\t\n"
      "fmul z23.s, p4/m, z23.s, z12.s\t\n"

      "ld2w { z28.s, z29.s }, p4/z, [x0]\t\n"
      "fmul z28.s, p4/m, z28.s, z12.s\t\n"
      "fmul z29.s, p4/m, z29.s, z12.s\t\n"

      "ld2w { z30.s, z31.s }, p4/z, [x0, x2, lsl #2]\t\n"
      "fmul z30.s, p4/m, z30.s, z12.s\t\n"
      "fmul z31.s, p4/m, z31.s, z12.s\t\n"

      "ld2w { z6.s, z7.s }, p4/z, [x16]\t\n"
      "fmul z6.s, p4/m, z6.s, z12.s\t\n"
      "fmul z7.s, p4/m, z7.s, z12.s\t\n"

      "ld2w { z8.s, z9.s }, p4/z, [x16, x2, lsl #2]\t\n"
      "fmul z8.s, p4/m, z8.s, z12.s\t\n"
      "fmul z9.s, p4/m, z9.s, z12.s\t\n"

      "ld2w { z24.s, z25.s }, p4/z, [x17]\t\n"
      "fmul z24.s, p4/m, z24.s, z12.s\t\n"
      "fmul z25.s, p4/m, z25.s, z12.s\t\n"

      "ld2w { z26.s, z27.s }, p4/z, [x17, x2, lsl #2]\t\n"
      "fmul z26.s, p4/m, z26.s, z12.s\t\n"
      "fmul z27.s, p4/m, z27.s, z12.s\t\n"

      "ld1rw { z2.s }, p1/z, [x9]\t\n"

      "fmla z0.s, p1/m, z3.s, z2.s\t\n"
      "fmla z1.s, p1/m, z13.s, z2.s\t\n"
      "fmla z20.s, p1/m, z4.s, z2.s\t\n"
      "fmla z21.s, p1/m, z14.s, z2.s\t\n"

      "ld1rw { z10.s }, p1/z, [x9, #4]\t\n"

      "fmla z18.s, p1/m, z3.s, z10.s\t\n"
      "fmla z19.s, p1/m, z13.s, z10.s\t\n"
      "fmla z22.s, p1/m, z4.s, z10.s\t\n"
      "fmla z23.s, p1/m, z14.s, z10.s\t\n"

      "ld1rw { z2.s }, p1/z, [x9, #8]\t\n"

      "fmla z28.s, p1/m, z3.s, z2.s\t\n"
      "fmla z29.s, p1/m, z13.s, z2.s\t\n"
      "fmla z30.s, p1/m, z4.s, z2.s\t\n"
      "fmla z31.s, p1/m, z14.s, z2.s\t\n"

      "ld1rw { z10.s }, p1/z, [x9, #12]\t\n"

      "fmla z6.s, p1/m, z3.s, z10.s\t\n"
      "fmla z7.s, p1/m, z13.s, z10.s\t\n"
      "fmla z8.s, p1/m, z4.s, z10.s\t\n"
      "fmla z9.s, p1/m, z14.s, z10.s\t\n"

      "ld1rw { z2.s }, p1/z, [x9, #16]\t\n"

      "fmla z24.s, p1/m, z3.s, z2.s\t\n"
      "fmla z25.s, p1/m, z13.s, z2.s\t\n"
      "fmla z26.s, p1/m, z4.s, z2.s\t\n"
      "fmla z27.s, p1/m, z14.s, z2.s\t\n"

      "cbz x14, .dump_C5\t\n"

      "add x9, x9, x5\t\n"
      "add x14, x14, x4\t\n"

      ".loop_inner5:\t\n"

      "sub x14, x14, x4\t\n"

      // vcvtph2ps

      "ldr q11, [x10]\t\n"
      "fcvt z15.s, p1/m, z11.h\t\n"
      "fcvtlt z16.s, p1/m, z11.h\t\n"

      "ldr q14, [x10, #16]\t\n"
      "fcvt z4.s, p1/m, z14.h\t\n"
      "fcvtlt z14.s, p1/m, z14.h\t\n"

      // broadcast
      "ld1rw { z10.s }, p1/z, [x9]\t\n"

      // fma
      "fmla z0.s, p1/m, z15.s, z10.s\t\n"
      "fmla z1.s, p1/m, z16.s, z10.s\t\n"

      "fmla z20.s, p1/m, z4.s, z10.s\t\n"
      "fmla z21.s, p1/m, z14.s, z10.s\t\n"

      "ld1rw { z2.s }, p1/z, [x9, #4]\t\n"

      // fma
      "fmla z18.s, p1/m, z15.s, z2.s\t\n"
      "fmla z19.s, p1/m, z16.s, z2.s\t\n"

      "fmla z22.s, p1/m, z4.s, z2.s\t\n"
      "fmla z23.s, p1/m, z14.s, z2.s\t\n"

      "ld1rw { z10.s }, p1/z, [x9, #8]\t\n"

      "fmla z28.s, p1/m, z15.s, z10.s\t\n"
      "fmla z29.s, p1/m, z16.s, z10.s\t\n"

      "fmla z30.s, p1/m, z4.s, z10.s\t\n"
      "fmla z31.s, p1/m, z14.s, z10.s\t\n"

      "ld1rw { z2.s }, p1/z, [x9, #12]\t\n"

      "fmla z6.s, p1/m, z15.s, z2.s\t\n"
      "fmla z7.s, p1/m, z16.s, z2.s\t\n"

      "fmla z8.s, p1/m, z4.s, z2.s\t\n"
      "fmla z9.s, p1/m, z14.s, z2.s\t\n"

      "ld1rw { z10.s }, p1/z, [x9, #16]\t\n"

      "fmla z24.s, p1/m, z15.s, z10.s\t\n"
      "fmla z25.s, p1/m, z16.s, z10.s\t\n"

      "fmla z26.s, p1/m, z4.s, z10.s\t\n"
      "fmla z27.s, p1/m, z14.s, z10.s\t\n"

      // next 32 bytes
      // update predicate
      "whilelo p2.s, x1, x14\t\n"

      // vcvtph2ps
      "ld1rqw { z11.s }, p2/z, [x10, x2, lsl #2]\t\n"
      "fcvt z3.s, p2/m, z11.h\t\n"
      "fcvtlt z13.s, p2/m, z11.h\t\n"

      "ld1rqw { z14.s }, p2/z, [x10, x3, lsl #2]\t\n"
      "fcvt z4.s, p2/m, z14.h\t\n"
      "fcvtlt z14.s, p2/m, z14.h\t\n"

      "ld1rw { z2.s }, p2/z, [x9, #20]\t\n"

      // fma
      "fmla z5.s, p2/m, z3.s, z2.s\t\n"
      "fmla z17.s, p2/m, z13.s, z2.s\t\n"

      "fmla z20.s, p2/m, z4.s, z2.s\t\n"
      "fmla z21.s, p2/m, z14.s, z2.s\t\n"

      "ld1rw { z10.s }, p2/z, [x9, #24]\t\n"

      // fma
      "fmla z18.s, p2/m, z3.s, z10.s\t\n"
      "fmla z19.s, p2/m, z13.s, z10.s\t\n"

      "fmla z22.s, p2/m, z4.s, z10.s\t\n"
      "fmla z23.s, p2/m, z14.s, z10.s\t\n"

      "ld1rw { z2.s }, p2/z, [x9, #28]\t\n"

      "fmla z28.s, p2/m, z3.s, z2.s\t\n"
      "fmla z29.s, p2/m, z13.s, z2.s\t\n"

      "fmla z30.s, p2/m, z4.s, z2.s\t\n"
      "fmla z31.s, p2/m, z14.s, z2.s\t\n"

      "ld1rw { z10.s }, p2/z, [x9, #32]\t\n"

      // fma
      "fmla z6.s, p2/m, z3.s, z10.s\t\n"
      "fmla z7.s, p2/m, z13.s, z10.s\t\n"

      "fmla z8.s, p2/m, z4.s, z10.s\t\n"
      "fmla z9.s, p2/m, z14.s, z10.s\t\n"

      "ld1rw { z2.s }, p2/z, [x9, #36]\t\n"

      "fmla z24.s, p2/m, z3.s, z2.s\t\n"
      "fmla z25.s, p2/m, z13.s, z2.s\t\n"

      "fmla z26.s, p2/m, z4.s, z2.s\t\n"
      "fmla z27.s, p2/m, z14.s, z2.s\t\n"

      // next 32 bytes

      // update predicates
      "whilelo p2.s, x2, x14\t\n"

      // vcvtph2ps
      "ld1rqw { z11.s }, p2/z, [x10, x4, lsl #2]\t\n"
      "fcvt z15.s, p2/m, z11.h\t\n"
      "fcvtlt z16.s, p2/m, z11.h\t\n"

      "ld1rqw { z14.s }, p2/z, [x10, x5, lsl #2]\t\n"
      "fcvt z4.s, p2/m, z14.h\t\n"
      "fcvtlt z14.s, p2/m, z14.h\t\n"

      "ld1rw { z10.s }, p2/z, [x9, #40]\t\n"

      // fma
      "fmla z0.s, p2/m, z15.s, z10.s\t\n"
      "fmla z1.s, p2/m, z16.s, z10.s\t\n"

      "fmla z20.s, p2/m, z4.s, z10.s\t\n"
      "fmla z21.s, p2/m, z14.s, z10.s\t\n"

      "ld1rw { z2.s }, p2/z, [x9, #44]\t\n"

      // fma
      "fmla z18.s, p2/m, z15.s, z2.s\t\n"
      "fmla z19.s, p2/m, z16.s, z2.s\t\n"

      "fmla z22.s, p2/m, z4.s, z2.s\t\n"
      "fmla z23.s, p2/m, z14.s, z2.s\t\n"

      "ld1rw { z10.s }, p2/z, [x9, #48]\t\n"

      "fmla z28.s, p2/m, z15.s, z10.s\t\n"
      "fmla z29.s, p2/m, z16.s, z10.s\t\n"

      "fmla z30.s, p2/m, z4.s, z10.s\t\n"
      "fmla z31.s, p2/m, z14.s, z10.s\t\n"

      "ld1rw { z2.s }, p2/z, [x9, #52]\t\n"

      // fma
      "fmla z6.s, p2/m, z15.s, z2.s\t\n"
      "fmla z7.s, p2/m, z16.s, z2.s\t\n"

      "fmla z8.s, p2/m, z4.s, z2.s\t\n"
      "fmla z9.s, p2/m, z14.s, z2.s\t\n"

      "ld1rw { z10.s }, p2/z, [x9, #56]\t\n"

      "fmla z24.s, p2/m, z15.s, z10.s\t\n"
      "fmla z25.s, p2/m, z16.s, z10.s\t\n"

      "fmla z26.s, p2/m, z4.s, z10.s\t\n"
      "fmla z27.s, p2/m, z14.s, z10.s\t\n"

      // next 32 bytes

      // update predicates
      "whilelo p2.s, x3, x14\t\n"

      // vcvtph2ps
      "ld1rqw { z11.s }, p2/z, [x10, x13, lsl #2]\t\n"
      "fcvt z3.s, p2/m, z11.h\t\n"
      "fcvtlt z13.s, p2/m, z11.h\t\n"

      "ld1rqw { z14.s }, p2/z, [x10, x15, lsl #2]\t\n"
      "add x10, x10, #128\t\n"
      "fcvt z4.s, p2/m, z14.h\t\n"
      "fcvtlt z14.s, p2/m, z14.h\t\n"

      "ld1rw { z2.s }, p2/z, [x9, #60]\t\n"

      // fma
      "fmla z5.s, p2/m, z3.s, z2.s\t\n"
      "fmla z17.s, p2/m, z13.s, z2.s\t\n"

      "fmla z20.s, p2/m, z4.s, z2.s\t\n"
      "fmla z21.s, p2/m, z14.s, z2.s\t\n"

      "ld1rw { z10.s }, p2/z, [x9, #64]\t\n"

      "fmla z18.s, p2/m, z3.s, z10.s\t\n"
      "fmla z19.s, p2/m, z13.s, z10.s\t\n"

      "fmla z22.s, p2/m, z4.s, z10.s\t\n"
      "fmla z23.s, p2/m, z14.s, z10.s\t\n"

      "ld1rw { z2.s }, p2/z, [x9, #68]\t\n"

      "fmla z28.s, p2/m, z3.s, z2.s\t\n"
      "fmla z29.s, p2/m, z13.s, z2.s\t\n"

      "fmla z30.s, p2/m, z4.s, z2.s\t\n"
      "fmla z31.s, p2/m, z14.s, z2.s\t\n"

      "ld1rw { z10.s }, p2/z, [x9, #72]\t\n"

      "fmla z6.s, p2/m, z3.s, z10.s\t\n"
      "fmla z7.s, p2/m, z13.s, z10.s\t\n"

      "fmla z8.s, p2/m, z4.s, z10.s\t\n"
      "fmla z9.s, p2/m, z14.s, z10.s\t\n"

      "ld1rw { z2.s }, p2/z, [x9, #76]\t\n"

      "add x9, x9, #80\t\n"

      "cmp x14, x4\t\n"

      "fmla z24.s, p2/m, z3.s, z2.s\t\n"
      "fmla z25.s, p2/m, z13.s, z2.s\t\n"

      "fmla z26.s, p2/m, z4.s, z2.s\t\n"
      "fmla z27.s, p2/m, z14.s, z2.s\t\n"

      // fma
      "bgt .loop_inner5\t\n"

      "sub x10, x10, #128\t\n"

      // hack works only whn processing 128-bit per register
      "add x10, x10, x14, lsl #3\t\n"

      "fadd z0.s, z0.s, z5.s\t\n"
      "fadd z1.s, z1.s, z17.s\t\n"

      // Dump C
      ".dump_C5:\t\n"

      "st2w { z0.s, z1.s }, p1, [x12]\t\n"
      "st2w { z20.s, z21.s }, p1, [x12, x2, lsl #2]\t\n"

      "add x12, x12, #64\t\n"

      "st2w { z18.s, z19.s }, p1, [x8]\t\n"
      "st2w { z22.s, z23.s }, p1, [x8, x2, lsl #2]\t\n"

      "add x8, x8, #64\t\n"

      "st2w { z28.s, z29.s }, p1, [x0]\t\n"
      "st2w { z30.s, z31.s }, p1, [x0, x2, lsl #2]\t\n"

      "add x0, x0, #64\t\n"

      "st2w { z6.s, z7.s }, p1, [x16]\t\n"
      "st2w { z8.s, z9.s }, p1, [x16, x2, lsl #2]\t\n"

      "add x16, x16, #64\t\n"

      "st2w { z24.s, z25.s }, p1, [x17]\t\n"
      "st2w { z26.s, z27.s }, p1, [x17, x2, lsl #2]\t\n"

      "add x17, x17, #64\t\n"

      // next outer iteratio
      "subs x7, x7, #1\t\n"
      "bne .loop_outer5\t\n"
      :
      : [gp] "r"(gp)
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
        "x10",
        "x11",
        "x12",
        "x13",
        "x14",
        "x15",
        "x16",
        "x17",
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
        "z22",
        "z23",
        "z24",
        "z25",
        "z26",
        "z27",
        "z28",
        "z29",
        "z30",
        "z31",
        "p1",
        "p2",
        "p3",
        "p4");
#endif
}

void NOINLINE gemmkernel_6x2_Sve128_fp16_fA0fB0fC0(GemmParamsFP16* gp) {
#if defined(__aarch64__) && __ARM_FEATURE_SVE
  asm volatile(

      "mov x0, %[gp]\t\n"

      "mov x1, #4\t\n"
      "mov x2, #8\t\n"

      // Lowest 128-bits
      "whilelo p1.s, w1, w2\t\n"

      // Copy parameters
      // k and A
      "ldp x11, x6, [x0]\t\n"
      "sub x11, x11, #1\t\n"
      // B
      "ldr x10, [x0, #16]\t\n"

      // beta
      "ld1rw { z12.s }, p1/z, [x0, #24]\t\n"

      // C and ldc
      "ldp x12, x8, [x0, #32]\t\n"
      // b_block_cols
      "ldr x7, [x0, #48]\t\n"

      "fcmeq p3.s, p1/z, z12.s, #0.0\t\n"
      "fcmuo p4.s, p1/z, z12.s, z12.s\t\n"
      "nor p4.b, p1/z, p4.b, p3.b\t\n"

      "add x0, x12, x8, lsl #1\t\n"
      "add x17, x12, x8, lsl #2\t\n"
      "add x16, x0, x8\t\n"
      "add x18, x17, x8\t\n"
      "add x8, x12, x8\t\n"

      "mov x3, #12\t\n"
      "mov x4, #16\t\n"
      "mov x5, #20\t\n"
      "mov x13, #24\t\n"
      "mov x15, #28\t\n"

      ".loop_outer6:\t\n"
      "mov x9, x6\t\n"
      "lsl x14, x11, #2\t\n"

      // vcvtph2ps
      "ldr q11, [x10]\t\n"
      "fcvt z3.s, p1/m, z11.h\t\n"
      "fcvtlt z13.s, p1/m, z11.h\t\n"

      // vcvtph2ps
      "ldr q15, [x10, #16]\t\n"
      "add x10, x10, #32\t\n"
      "fcvt z14.s, p1/m, z15.h\t\n"
      "fcvtlt z15.s, p1/m, z15.h\t\n"

      // Setup values with beta multiplication
      "ld2w { z0.s, z1.s }, p4/z, [x12]\t\n"
      "fmul z0.s, p4/m, z0.s, z12.s\t\n"
      "fmul z1.s, p4/m, z1.s, z12.s\t\n"

      "ld2w { z20.s, z21.s }, p4/z, [x12, x2, lsl #2]\t\n"
      "fmul z20.s, p4/m, z20.s, z12.s\t\n"
      "fmul z21.s, p4/m, z21.s, z12.s\t\n"

      "ld2w { z18.s, z19.s }, p4/z, [x8]\t\n"
      "fmul z18.s, p4/m, z18.s, z12.s\t\n"
      "fmul z19.s, p4/m, z19.s, z12.s\t\n"

      "ld2w { z22.s, z23.s }, p4/z, [x8, x2, lsl #2]\t\n"
      "fmul z22.s, p4/m, z22.s, z12.s\t\n"
      "fmul z23.s, p4/m, z23.s, z12.s\t\n"

      "ld2w { z28.s, z29.s }, p4/z, [x0]\t\n"
      "fmul z28.s, p4/m, z28.s, z12.s\t\n"
      "fmul z29.s, p4/m, z29.s, z12.s\t\n"

      "ld2w { z30.s, z31.s }, p4/z, [x0, x2, lsl #2]\t\n"
      "fmul z30.s, p4/m, z30.s, z12.s\t\n"
      "fmul z31.s, p4/m, z31.s, z12.s\t\n"

      "ld2w { z6.s, z7.s }, p4/z, [x16]\t\n"
      "fmul z6.s, p4/m, z6.s, z12.s\t\n"
      "fmul z7.s, p4/m, z7.s, z12.s\t\n"

      "ld2w { z8.s, z9.s }, p4/z, [x16, x2, lsl #2]\t\n"
      "fmul z8.s, p4/m, z8.s, z12.s\t\n"
      "fmul z9.s, p4/m, z9.s, z12.s\t\n"

      "ld2w { z24.s, z25.s }, p4/z, [x17]\t\n"
      "fmul z24.s, p4/m, z24.s, z12.s\t\n"
      "fmul z25.s, p4/m, z25.s, z12.s\t\n"

      "ld2w { z26.s, z27.s }, p4/z, [x17, x2, lsl #2]\t\n"
      "fmul z26.s, p4/m, z26.s, z12.s\t\n"
      "fmul z27.s, p4/m, z27.s, z12.s\t\n"

      "ld2w { z4.s, z5.s }, p4/z, [x18]\t\n"
      "fmul z4.s, p4/m, z4.s, z12.s\t\n"
      "fmul z5.s, p4/m, z5.s, z12.s\t\n"

      "ld2w { z16.s, z17.s }, p4/z, [x18, x2, lsl #2]\t\n"
      "fmul z16.s, p4/m, z16.s, z12.s\t\n"
      "fmul z17.s, p4/m, z17.s, z12.s\t\n"

      "ld1rw { z2.s }, p1/z, [x9]\t\n"

      "fmla z0.s, p1/m, z3.s, z2.s\t\n"
      "fmla z1.s, p1/m, z13.s, z2.s\t\n"
      "fmla z20.s, p1/m, z14.s, z2.s\t\n"
      "fmla z21.s, p1/m, z15.s, z2.s\t\n"

      "ld1rw { z10.s }, p1/z, [x9, #4]\t\n"

      "fmla z18.s, p1/m, z3.s, z10.s\t\n"
      "fmla z19.s, p1/m, z13.s, z10.s\t\n"
      "fmla z22.s, p1/m, z14.s, z10.s\t\n"
      "fmla z23.s, p1/m, z15.s, z10.s\t\n"

      "ld1rw { z2.s }, p1/z, [x9, #8]\t\n"

      "fmla z28.s, p1/m, z3.s, z2.s\t\n"
      "fmla z29.s, p1/m, z13.s, z2.s\t\n"
      "fmla z30.s, p1/m, z14.s, z2.s\t\n"
      "fmla z31.s, p1/m, z15.s, z2.s\t\n"

      "ld1rw { z10.s }, p1/z, [x9, #12]\t\n"

      "fmla z6.s, p1/m, z3.s, z10.s\t\n"
      "fmla z7.s, p1/m, z13.s, z10.s\t\n"
      "fmla z8.s, p1/m, z14.s, z10.s\t\n"
      "fmla z9.s, p1/m, z15.s, z10.s\t\n"

      "ld1rw { z2.s }, p1/z, [x9, #16]\t\n"

      "fmla z24.s, p1/m, z3.s, z2.s\t\n"
      "fmla z25.s, p1/m, z13.s, z2.s\t\n"
      "fmla z26.s, p1/m, z14.s, z2.s\t\n"
      "fmla z27.s, p1/m, z15.s, z2.s\t\n"

      "ld1rw { z10.s }, p1/z, [x9, #20]\t\n"

      "fmla z4.s, p1/m, z3.s, z10.s\t\n"
      "fmla z5.s, p1/m, z13.s, z10.s\t\n"
      "fmla z16.s, p1/m, z14.s, z10.s\t\n"
      "fmla z17.s, p1/m, z15.s, z10.s\t\n"

      "cbz x14, .dump_C6\t\n"

      "add x9, x9, x13\t\n"
      "add x14, x14, x4\t\n"

      ".loop_inner6:\t\n"

      "sub x14, x14, x4\t\n"

      // vcvtph2ps

      "ldr q11, [x10]\t\n"
      "fcvt z3.s, p1/m, z11.h\t\n"
      "fcvtlt z13.s, p1/m, z11.h\t\n"

      "ldr q15, [x10, #16]\t\n"
      "fcvt z14.s, p1/m, z15.h\t\n"
      "fcvtlt z15.s, p1/m, z15.h\t\n"

      // broadcast
      "ld1rw { z2.s }, p1/z, [x9]\t\n"

      // fma
      "fmla z0.s, p1/m, z3.s, z2.s\t\n"
      "fmla z1.s, p1/m, z13.s, z2.s\t\n"

      "fmla z20.s, p1/m, z14.s, z2.s\t\n"
      "fmla z21.s, p1/m, z15.s, z2.s\t\n"

      "ld1rw { z10.s }, p1/z, [x9, #4]\t\n"

      // fma
      "fmla z18.s, p1/m, z3.s, z10.s\t\n"
      "fmla z19.s, p1/m, z13.s, z10.s\t\n"

      "fmla z22.s, p1/m, z14.s, z10.s\t\n"
      "fmla z23.s, p1/m, z15.s, z10.s\t\n"

      "ld1rw { z2.s }, p1/z, [x9, #8]\t\n"

      "fmla z28.s, p1/m, z3.s, z2.s\t\n"
      "fmla z29.s, p1/m, z13.s, z2.s\t\n"

      "fmla z30.s, p1/m, z14.s, z2.s\t\n"
      "fmla z31.s, p1/m, z15.s, z2.s\t\n"

      "ld1rw { z10.s }, p1/z, [x9, #12]\t\n"

      "fmla z6.s, p1/m, z3.s, z10.s\t\n"
      "fmla z7.s, p1/m, z13.s, z10.s\t\n"

      "fmla z8.s, p1/m, z14.s, z10.s\t\n"
      "fmla z9.s, p1/m, z15.s, z10.s\t\n"

      "ld1rw { z2.s }, p1/z, [x9, #16]\t\n"

      "fmla z24.s, p1/m, z3.s, z2.s\t\n"
      "fmla z25.s, p1/m, z13.s, z2.s\t\n"

      "fmla z26.s, p1/m, z14.s, z2.s\t\n"
      "fmla z27.s, p1/m, z15.s, z2.s\t\n"

      "ld1rw { z10.s }, p1/z, [x9, #20]\t\n"

      "fmla z4.s, p1/m, z3.s, z10.s\t\n"
      "fmla z5.s, p1/m, z13.s, z10.s\t\n"

      "fmla z16.s, p1/m, z14.s, z10.s\t\n"
      "fmla z17.s, p1/m, z15.s, z10.s\t\n"

      // next 32 bytes
      // update predicate
      "whilelo p2.s, x1, x14\t\n"

      // vcvtph2ps
      "ld1rqw { z11.s }, p2/z, [x10, x2, lsl #2]\t\n"
      "fcvt z3.s, p2/m, z11.h\t\n"
      "fcvtlt z13.s, p2/m, z11.h\t\n"

      "ld1rqw { z15.s }, p2/z, [x10, x3, lsl #2]\t\n"
      "fcvt z14.s, p2/m, z15.h\t\n"
      "fcvtlt z15.s, p2/m, z15.h\t\n"

      "ld1rw { z2.s }, p2/z, [x9, #24]\t\n"

      // fma
      "fmla z0.s, p2/m, z3.s, z2.s\t\n"
      "fmla z1.s, p2/m, z13.s, z2.s\t\n"

      "fmla z20.s, p2/m, z14.s, z2.s\t\n"
      "fmla z21.s, p2/m, z15.s, z2.s\t\n"

      "ld1rw { z10.s }, p2/z, [x9, #28]\t\n"

      // fma
      "fmla z18.s, p2/m, z3.s, z10.s\t\n"
      "fmla z19.s, p2/m, z13.s, z10.s\t\n"

      "fmla z22.s, p2/m, z14.s, z10.s\t\n"
      "fmla z23.s, p2/m, z15.s, z10.s\t\n"

      "ld1rw { z2.s }, p2/z, [x9, #32]\t\n"

      "fmla z28.s, p2/m, z3.s, z2.s\t\n"
      "fmla z29.s, p2/m, z13.s, z2.s\t\n"

      "fmla z30.s, p2/m, z14.s, z2.s\t\n"
      "fmla z31.s, p2/m, z15.s, z2.s\t\n"

      "ld1rw { z10.s }, p2/z, [x9, #36]\t\n"

      // fma
      "fmla z6.s, p2/m, z3.s, z10.s\t\n"
      "fmla z7.s, p2/m, z13.s, z10.s\t\n"

      "fmla z8.s, p2/m, z14.s, z10.s\t\n"
      "fmla z9.s, p2/m, z15.s, z10.s\t\n"

      "ld1rw { z2.s }, p2/z, [x9, #40]\t\n"

      "fmla z24.s, p2/m, z3.s, z2.s\t\n"
      "fmla z25.s, p2/m, z13.s, z2.s\t\n"

      "fmla z26.s, p2/m, z14.s, z2.s\t\n"
      "fmla z27.s, p2/m, z15.s, z2.s\t\n"

      "ld1rw { z10.s }, p2/z, [x9, #44]\t\n"

      // fma
      "fmla z4.s, p2/m, z3.s, z10.s\t\n"
      "fmla z5.s, p2/m, z13.s, z10.s\t\n"

      "fmla z16.s, p2/m, z14.s, z10.s\t\n"
      "fmla z17.s, p2/m, z15.s, z10.s\t\n"

      // next 32 bytes

      // update predicates
      "whilelo p2.s, x2, x14\t\n"

      // vcvtph2ps
      "ld1rqw { z11.s }, p2/z, [x10, x4, lsl #2]\t\n"
      "fcvt z3.s, p2/m, z11.h\t\n"
      "fcvtlt z13.s, p2/m, z11.h\t\n"

      "ld1rqw { z15.s }, p2/z, [x10, x5, lsl #2]\t\n"
      "fcvt z14.s, p2/m, z15.h\t\n"
      "fcvtlt z15.s, p2/m, z15.h\t\n"

      "ld1rw { z2.s }, p2/z, [x9, #48]\t\n"

      // fma
      "fmla z0.s, p2/m, z3.s, z2.s\t\n"
      "fmla z1.s, p2/m, z13.s, z2.s\t\n"

      "fmla z20.s, p2/m, z14.s, z2.s\t\n"
      "fmla z21.s, p2/m, z15.s, z2.s\t\n"

      "ld1rw { z10.s }, p2/z, [x9, #52]\t\n"

      // fma
      "fmla z18.s, p2/m, z3.s, z10.s\t\n"
      "fmla z19.s, p2/m, z13.s, z10.s\t\n"

      "fmla z22.s, p2/m, z14.s, z10.s\t\n"
      "fmla z23.s, p2/m, z15.s, z10.s\t\n"

      "ld1rw { z2.s }, p2/z, [x9, #56]\t\n"

      "fmla z28.s, p2/m, z3.s, z2.s\t\n"
      "fmla z29.s, p2/m, z13.s, z2.s\t\n"

      "fmla z30.s, p2/m, z14.s, z2.s\t\n"
      "fmla z31.s, p2/m, z15.s, z2.s\t\n"

      "ld1rw { z10.s }, p2/z, [x9, #60]\t\n"

      // fma
      "fmla z6.s, p2/m, z3.s, z10.s\t\n"
      "fmla z7.s, p2/m, z13.s, z10.s\t\n"

      "fmla z8.s, p2/m, z14.s, z10.s\t\n"
      "fmla z9.s, p2/m, z15.s, z10.s\t\n"

      "ld1rw { z2.s }, p2/z, [x9, #64]\t\n"

      "fmla z24.s, p2/m, z3.s, z2.s\t\n"
      "fmla z25.s, p2/m, z13.s, z2.s\t\n"

      "fmla z26.s, p2/m, z14.s, z2.s\t\n"
      "fmla z27.s, p2/m, z15.s, z2.s\t\n"

      "ld1rw { z10.s }, p2/z, [x9, #68]\t\n"

      "fmla z4.s, p2/m, z3.s, z10.s\t\n"
      "fmla z5.s, p2/m, z13.s, z10.s\t\n"

      "fmla z16.s, p2/m, z14.s, z10.s\t\n"
      "fmla z17.s, p2/m, z15.s, z10.s\t\n"

      // next 32 bytes

      // update predicates
      "whilelo p2.s, x3, x14\t\n"

      // vcvtph2ps
      "ld1rqw { z11.s }, p2/z, [x10, x13, lsl #2]\t\n"
      "fcvt z3.s, p2/m, z11.h\t\n"
      "fcvtlt z13.s, p2/m, z11.h\t\n"

      "ld1rqw { z15.s }, p2/z, [x10, x15, lsl #2]\t\n"
      "add x10, x10, #128\t\n"
      "fcvt z14.s, p2/m, z15.h\t\n"
      "fcvtlt z15.s, p2/m, z15.h\t\n"

      "ld1rw { z2.s }, p2/z, [x9, #72]\t\n"

      // fma
      "fmla z0.s, p2/m, z3.s, z2.s\t\n"
      "fmla z1.s, p2/m, z13.s, z2.s\t\n"

      "fmla z20.s, p2/m, z14.s, z2.s\t\n"
      "fmla z21.s, p2/m, z15.s, z2.s\t\n"

      "ld1rw { z10.s }, p2/z, [x9, #76]\t\n"

      "fmla z18.s, p2/m, z3.s, z10.s\t\n"
      "fmla z19.s, p2/m, z13.s, z10.s\t\n"

      "fmla z22.s, p2/m, z14.s, z10.s\t\n"
      "fmla z23.s, p2/m, z15.s, z10.s\t\n"

      "ld1rw { z2.s }, p2/z, [x9, #80]\t\n"

      "fmla z28.s, p2/m, z3.s, z2.s\t\n"
      "fmla z29.s, p2/m, z13.s, z2.s\t\n"

      "fmla z30.s, p2/m, z14.s, z2.s\t\n"
      "fmla z31.s, p2/m, z15.s, z2.s\t\n"

      "ld1rw { z10.s }, p2/z, [x9, #84]\t\n"

      "fmla z6.s, p2/m, z3.s, z10.s\t\n"
      "fmla z7.s, p2/m, z13.s, z10.s\t\n"

      "fmla z8.s, p2/m, z14.s, z10.s\t\n"
      "fmla z9.s, p2/m, z15.s, z10.s\t\n"

      "ld1rw { z2.s }, p2/z, [x9, #88]\t\n"

      "fmla z24.s, p2/m, z3.s, z2.s\t\n"
      "fmla z25.s, p2/m, z13.s, z2.s\t\n"

      "fmla z26.s, p2/m, z14.s, z2.s\t\n"
      "fmla z27.s, p2/m, z15.s, z2.s\t\n"

      "ld1rw { z10.s }, p2/z, [x9, #92]\t\n"

      "add x9, x9, #96\t\n"

      "cmp x14, x4\t\n"

      "fmla z4.s, p2/m, z3.s, z10.s\t\n"
      "fmla z5.s, p2/m, z13.s, z10.s\t\n"

      "fmla z16.s, p2/m, z14.s, z10.s\t\n"
      "fmla z17.s, p2/m, z15.s, z10.s\t\n"

      // fma

      "bgt .loop_inner6\t\n"

      "sub x10, x10, #128\t\n"

      // hack works only whn processing 128-bit per register
      "add x10, x10, x14, lsl #3\t\n"

      // Dump C
      ".dump_C6:\t\n"

      "st2w { z0.s, z1.s }, p1, [x12]\t\n"
      "st2w { z20.s, z21.s }, p1, [x12, x2, lsl #2]\t\n"

      "add x12, x12, #64\t\n"

      "st2w { z18.s, z19.s }, p1, [x8]\t\n"
      "st2w { z22.s, z23.s }, p1, [x8, x2, lsl #2]\t\n"

      "add x8, x8, #64\t\n"

      "st2w { z28.s, z29.s }, p1, [x0]\t\n"
      "st2w { z30.s, z31.s }, p1, [x0, x2, lsl #2]\t\n"

      "add x0, x0, #64\t\n"

      "st2w { z6.s, z7.s }, p1, [x16]\t\n"
      "st2w { z8.s, z9.s }, p1, [x16, x2, lsl #2]\t\n"

      "add x16, x16, #64\t\n"

      "st2w { z24.s, z25.s }, p1, [x17]\t\n"
      "st2w { z26.s, z27.s }, p1, [x17, x2, lsl #2]\t\n"

      "add x17, x17, #64\t\n"

      "st2w { z4.s, z5.s }, p1, [x18]\t\n"
      "st2w { z16.s, z17.s }, p1, [x18, x2, lsl #2]\t\n"

      "add x18, x18, #64\t\n"

      // next outer iteratio
      "subs x7, x7, #1\t\n"
      "bne .loop_outer6\t\n"
      :
      : [gp] "r"(gp)
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
        "x10",
        "x11",
        "x12",
        "x13",
        "x14",
        "x15",
        "x16",
        "x17",
        "x18",
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
        "z22",
        "z23",
        "z24",
        "z25",
        "z26",
        "z27",
        "z28",
        "z29",
        "z30",
        "z31",
        "p1",
        "p2",
        "p3",
        "p4");
#endif
}

} // namespace fbgemm
