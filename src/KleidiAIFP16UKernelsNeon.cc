// @lint-ignore-every LICENSELINT
//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates
// <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#ifdef FBGEMM_ENABLE_KLEIDIAI

#include "KleidiAIFP16UKernelsNeon.h" // @manual

namespace kleidiai {

void NOINLINE gemmkernel_1x1_Neon_fp16_fA0fB0fC0(GemmParamsFP16* gp) {
#if defined(__aarch64__) && __ARM_FEATURE_FP16_FML
  __asm__ __volatile__(
      "ldr s16, [%x[gp], %[offsetof_beta]]\n"
      "mov x9, #0x1\n"
      "fmov v29.8h, #1.0\n"
      "ldr x10, [%x[gp], %[offsetof_k]]\n"
      "ldr x11, [%x[gp], %[offsetof_A]]\n"
      "ldr x15, [%x[gp], %[offsetof_B]]\n"
      "ldr x14, [%x[gp], %[offsetof_C]]\n"
      "ldr x8, [%x[gp], %[offsetof_b_block_cols]]\n"
      "fcmp s16, #0.0\n"
      "csel x9, XZR, x9, EQ\n"
      "csel x9, XZR, x9, VS\n"
      "1:" // Height 1: Column loop
      "tbz x9, #0, 2f\n"
      "ldr q30, [x14, #0x0]\n"
      "ldr q31, [x14, #0x10]\n"
      "add x12, %x[gp], %[offsetof_beta]\n"
      "ld1r { v16.4s }, [x12]\n"
      "fmul v30.4s, v30.4s, v16.4s\n"
      "fmul v31.4s, v31.4s, v16.4s\n"
      "b 3f\n"
      "2:" // Height 1: no accumulate
      "movi v30.16b, #0x0\n"
      "movi v31.16b, #0x0\n"
      "3:" // Height 1: setup done
      "cmp x10, #0x4\n"
      "mov x12, x11\n"
      "mov x13, x10\n"
      "blt 7f\n"
      "ldr q0, [x11, #0x0]\n"
      "ldr q1, [x15, #0x0]\n"
      "cmp x10, #0x8\n"
      "ldr q4, [x15, #0x10]\n"
      "ldr q7, [x15, #0x20]\n"
      "ldr q20, [x15, #0x30]\n"
      "blt 6f\n"
      "5:" // Height 1: Multiply loop: Main loop head
      "movi v2.16b, #0x0\n"
      "movi v3.16b, #0x0\n"
      "sub x13, x13, #0x4\n"
      "add x12, x12, #0x10\n"
      "movi v5.16b, #0x0\n"
      "movi v6.16b, #0x0\n"
      "cmp x13, #0x8\n"
      "fmlal v2.4s, v1.4h, v29.4h\n"
      "fmlal2 v3.4s, v1.4h, v29.4h\n"
      "ldr q1, [x15, #0x40]\n"
      "movi v26.16b, #0x0\n"
      "fmlal v5.4s, v4.4h, v29.4h\n"
      "fmlal2 v6.4s, v4.4h, v29.4h\n"
      "ldr q4, [x15, #0x50]\n"
      "movi v27.16b, #0x0\n"
      "fmlal v26.4s, v7.4h, v29.4h\n"
      "movi v21.16b, #0x0\n"
      "prfm pldl1keep, [x12, #0x80]\n"
      "fmlal2 v27.4s, v7.4h, v29.4h\n"
      "ldr q7, [x15, #0x60]\n"
      "movi v22.16b, #0x0\n"
      "fmla v30.4s, v2.4s, v0.s[0]\n"
      "fmla v31.4s, v3.4s, v0.s[0]\n"
      "fmlal v21.4s, v20.4h, v29.4h\n"
      "fmlal2 v22.4s, v20.4h, v29.4h\n"
      "ldr q20, [x15, #0x70]\n"
      "fmla v30.4s, v5.4s, v0.s[1]\n"
      "fmla v31.4s, v6.4s, v0.s[1]\n"
      "fmla v30.4s, v26.4s, v0.s[2]\n"
      "fmla v31.4s, v27.4s, v0.s[2]\n"
      "fmla v30.4s, v21.4s, v0.s[3]\n"
      "fmla v31.4s, v22.4s, v0.s[3]\n"
      "ldr q0, [x12, #0x0]\n"
      "add x15, x15, #0x40\n"
      "bge 5b\n"
      "6:" // Height 1: Multiply loop: Single iteration only
      "movi v2.16b, #0x0\n"
      "movi v3.16b, #0x0\n"
      "add x12, x12, #0x10\n"
      "sub x13, x13, #0x4\n"
      "movi v5.16b, #0x0\n"
      "movi v6.16b, #0x0\n"
      "prfm pldl1keep, [x12, #0x80]\n"
      "add x15, x15, #0x40\n"
      "fmlal v2.4s, v1.4h, v29.4h\n"
      "fmlal2 v3.4s, v1.4h, v29.4h\n"
      "movi v26.16b, #0x0\n"
      "fmlal v5.4s, v4.4h, v29.4h\n"
      "fmlal2 v6.4s, v4.4h, v29.4h\n"
      "movi v27.16b, #0x0\n"
      "fmlal v26.4s, v7.4h, v29.4h\n"
      "movi v21.16b, #0x0\n"
      "fmlal2 v27.4s, v7.4h, v29.4h\n"
      "movi v22.16b, #0x0\n"
      "fmla v30.4s, v2.4s, v0.s[0]\n"
      "fmla v31.4s, v3.4s, v0.s[0]\n"
      "fmlal v21.4s, v20.4h, v29.4h\n"
      "fmlal2 v22.4s, v20.4h, v29.4h\n"
      "fmla v30.4s, v5.4s, v0.s[1]\n"
      "fmla v31.4s, v6.4s, v0.s[1]\n"
      "fmla v30.4s, v26.4s, v0.s[2]\n"
      "fmla v31.4s, v27.4s, v0.s[2]\n"
      "fmla v30.4s, v21.4s, v0.s[3]\n"
      "fmla v31.4s, v22.4s, v0.s[3]\n"
      "7:" // Height 1: Multiply loop: Main loop skip
      "cbz x13, 9f\n"
      "8:" // Height 1: Multiply loop: Odd block loop
      "ldr q23, [x15, #0x0]\n"
      "ldr s0, [x12], #0x4\n"
      "movi v24.16b, #0x0\n"
      "movi v25.16b, #0x0\n"
      "sub x13, x13, #0x1\n"
      "add x15, x15, #0x10\n"
      "fmlal v24.4s, v23.4h, v29.4h\n"
      "fmlal2 v25.4s, v23.4h, v29.4h\n"
      "fmla v30.4s, v24.4s, v0.s[0]\n"
      "fmla v31.4s, v25.4s, v0.s[0]\n"
      "cbnz x13, 8b\n"
      "9:" // Height 1: Multiply loop: No odd multiplies
      "prfm pstl1keep, [x14, #0x0]\n"
      "str q30, [x14, #0x0]\n"
      "str q31, [x14, #0x10]\n"
      "add x14, x14, #0x20\n"
      "subs x8, x8, #0x1\n"
      "bgt 1b\n"
      :
      : [gp] "r"(gp),
        [offsetof_A] "I"(offsetof(GemmParamsFP16, A)),
        [offsetof_B] "I"(offsetof(GemmParamsFP16, B)),
        [offsetof_C] "I"(offsetof(GemmParamsFP16, C)),
        [offsetof_b_block_cols] "I"(offsetof(GemmParamsFP16, b_block_cols)),
        [offsetof_beta] "I"(offsetof(GemmParamsFP16, beta)),
        [offsetof_k] "I"(offsetof(GemmParamsFP16, k))
      : "cc",
        "memory",
        "v0",
        "v1",
        "v16",
        "v2",
        "v20",
        "v21",
        "v22",
        "v23",
        "v24",
        "v25",
        "v26",
        "v27",
        "v29",
        "v3",
        "v30",
        "v31",
        "v4",
        "v5",
        "v6",
        "v7",
        "x10",
        "x11",
        "x12",
        "x13",
        "x14",
        "x15",
        "x8",
        "x9");
#endif // __aarch64__
}

void NOINLINE gemmkernel_2x1_Neon_fp16_fA0fB0fC0(GemmParamsFP16* gp) {
#if defined(__aarch64__) && __ARM_FEATURE_FP16_FML
  __asm__ __volatile__(
      "ldr s16, [%x[gp], %[offsetof_beta]]\n"
      "mov x8, #0x1\n"
      "fmov v27.8h, #1.0\n"
      "ldr x12, [%x[gp], %[offsetof_k]]\n"
      "ldr x10, [%x[gp], %[offsetof_A]]\n"
      "ldr x6, [%x[gp], %[offsetof_B]]\n"
      "ldr x5, [%x[gp], %[offsetof_C]]\n"
      "ldr x11, [%x[gp], %[offsetof_lda]]\n"
      "fcmp s16, #0.0\n"
      "csel x8, XZR, x8, EQ\n"
      "csel x8, XZR, x8, VS\n"
      "ldr x9, [%x[gp], %[offsetof_ldc]]\n"
      "ldr x7, [%x[gp], %[offsetof_b_block_cols]]\n"
      "1:" // Height 2: Column loop
      "tbz x8, #0, 2f\n"
      "ldr q28, [x5, #0x0]\n"
      "ldr q29, [x5, #0x10]\n"
      "add x13, %x[gp], %[offsetof_beta]\n"
      "ld1r { v16.4s }, [x13]\n"
      "add x13, x5, x9\n"
      "ldr q30, [x13, #0x0]\n"
      "ldr q31, [x13, #0x10]\n"
      "fmul v28.4s, v28.4s, v16.4s\n"
      "fmul v29.4s, v29.4s, v16.4s\n"
      "fmul v30.4s, v30.4s, v16.4s\n"
      "fmul v31.4s, v31.4s, v16.4s\n"
      "b 3f\n"
      "2:" // Height 2: no accumulate
      "movi v28.16b, #0x0\n"
      "movi v29.16b, #0x0\n"
      "movi v30.16b, #0x0\n"
      "movi v31.16b, #0x0\n"
      "3:" // Height 2: setup done
      "add x13, x10, x11\n"
      "cmp x12, #0x4\n"
      "mov x15, x12\n"
      "mov x14, x10\n"
      "blt 7f\n"
      "ldr q0, [x10, #0x0]\n"
      "ldr q2, [x6, #0x0]\n"
      "cmp x12, #0x8\n"
      "ldr q1, [x13, #0x0]\n"
      "ldr q5, [x6, #0x10]\n"
      "ldr q18, [x6, #0x20]\n"
      "ldr q21, [x6, #0x30]\n"
      "blt 6f\n"
      "5:" // Height 2: Multiply loop: Main loop head
      "movi v3.16b, #0x0\n"
      "movi v4.16b, #0x0\n"
      "sub x15, x15, #0x4\n"
      "add x14, x14, #0x10\n"
      "movi v6.16b, #0x0\n"
      "movi v7.16b, #0x0\n"
      "add x13, x13, #0x10\n"
      "cmp x15, #0x8\n"
      "fmlal v3.4s, v2.4h, v27.4h\n"
      "fmlal2 v4.4s, v2.4h, v27.4h\n"
      "movi v19.16b, #0x0\n"
      "ldr q2, [x6, #0x40]\n"
      "fmlal v6.4s, v5.4h, v27.4h\n"
      "fmlal2 v7.4s, v5.4h, v27.4h\n"
      "ldr q5, [x6, #0x50]\n"
      "movi v20.16b, #0x0\n"
      "fmlal v19.4s, v18.4h, v27.4h\n"
      "prfm pldl1keep, [x14, #0x80]\n"
      "prfm pldl1keep, [x13, #0x80]\n"
      "movi v22.16b, #0x0\n"
      "movi v23.16b, #0x0\n"
      "fmla v28.4s, v3.4s, v0.s[0]\n"
      "fmla v30.4s, v3.4s, v1.s[0]\n"
      "fmla v29.4s, v4.4s, v0.s[0]\n"
      "fmla v31.4s, v4.4s, v1.s[0]\n"
      "fmlal2 v20.4s, v18.4h, v27.4h\n"
      "ldr q18, [x6, #0x60]\n"
      "fmlal v22.4s, v21.4h, v27.4h\n"
      "fmlal2 v23.4s, v21.4h, v27.4h\n"
      "ldr q21, [x6, #0x70]\n"
      "fmla v28.4s, v6.4s, v0.s[1]\n"
      "fmla v30.4s, v6.4s, v1.s[1]\n"
      "fmla v29.4s, v7.4s, v0.s[1]\n"
      "fmla v31.4s, v7.4s, v1.s[1]\n"
      "fmla v28.4s, v19.4s, v0.s[2]\n"
      "fmla v30.4s, v19.4s, v1.s[2]\n"
      "fmla v29.4s, v20.4s, v0.s[2]\n"
      "fmla v31.4s, v20.4s, v1.s[2]\n"
      "fmla v28.4s, v22.4s, v0.s[3]\n"
      "fmla v30.4s, v22.4s, v1.s[3]\n"
      "fmla v29.4s, v23.4s, v0.s[3]\n"
      "ldr q0, [x14, #0x0]\n"
      "fmla v31.4s, v23.4s, v1.s[3]\n"
      "ldr q1, [x13, #0x0]\n"
      "add x6, x6, #0x40\n"
      "bge 5b\n"
      "6:" // Height 2: Multiply loop: Single iteration only
      "movi v3.16b, #0x0\n"
      "movi v4.16b, #0x0\n"
      "add x14, x14, #0x10\n"
      "add x13, x13, #0x10\n"
      "movi v6.16b, #0x0\n"
      "movi v7.16b, #0x0\n"
      "sub x15, x15, #0x4\n"
      "prfm pldl1keep, [x14, #0x80]\n"
      "fmlal v3.4s, v2.4h, v27.4h\n"
      "fmlal2 v4.4s, v2.4h, v27.4h\n"
      "movi v19.16b, #0x0\n"
      "prfm pldl1keep, [x13, #0x80]\n"
      "fmlal v6.4s, v5.4h, v27.4h\n"
      "fmlal2 v7.4s, v5.4h, v27.4h\n"
      "movi v20.16b, #0x0\n"
      "add x6, x6, #0x40\n"
      "fmlal v19.4s, v18.4h, v27.4h\n"
      "movi v22.16b, #0x0\n"
      "fmlal2 v20.4s, v18.4h, v27.4h\n"
      "movi v23.16b, #0x0\n"
      "fmla v28.4s, v3.4s, v0.s[0]\n"
      "fmla v30.4s, v3.4s, v1.s[0]\n"
      "fmla v29.4s, v4.4s, v0.s[0]\n"
      "fmla v31.4s, v4.4s, v1.s[0]\n"
      "fmlal v22.4s, v21.4h, v27.4h\n"
      "fmlal2 v23.4s, v21.4h, v27.4h\n"
      "fmla v28.4s, v6.4s, v0.s[1]\n"
      "fmla v30.4s, v6.4s, v1.s[1]\n"
      "fmla v29.4s, v7.4s, v0.s[1]\n"
      "fmla v31.4s, v7.4s, v1.s[1]\n"
      "fmla v28.4s, v19.4s, v0.s[2]\n"
      "fmla v30.4s, v19.4s, v1.s[2]\n"
      "fmla v29.4s, v20.4s, v0.s[2]\n"
      "fmla v31.4s, v20.4s, v1.s[2]\n"
      "fmla v28.4s, v22.4s, v0.s[3]\n"
      "fmla v30.4s, v22.4s, v1.s[3]\n"
      "fmla v29.4s, v23.4s, v0.s[3]\n"
      "fmla v31.4s, v23.4s, v1.s[3]\n"
      "7:" // Height 2: Multiply loop: Main loop skip
      "cbz x15, 9f\n"
      "8:" // Height 2: Multiply loop: Odd block loop
      "ldr q24, [x6, #0x0]\n"
      "ldr s0, [x14], #0x4\n"
      "movi v25.16b, #0x0\n"
      "movi v16.16b, #0x0\n"
      "ldr s1, [x13], #0x4\n"
      "sub x15, x15, #0x1\n"
      "add x6, x6, #0x10\n"
      "fmlal v25.4s, v24.4h, v27.4h\n"
      "fmlal2 v16.4s, v24.4h, v27.4h\n"
      "fmla v28.4s, v25.4s, v0.s[0]\n"
      "fmla v30.4s, v25.4s, v1.s[0]\n"
      "fmla v29.4s, v16.4s, v0.s[0]\n"
      "fmla v31.4s, v16.4s, v1.s[0]\n"
      "cbnz x15, 8b\n"
      "9:" // Height 2: Multiply loop: No odd multiplies
      "prfm pstl1keep, [x5, #0x0]\n"
      "str q28, [x5, #0x0]\n"
      "str q29, [x5, #0x10]\n"
      "add x13, x5, x9\n"
      "add x5, x5, #0x20\n"
      "prfm pstl1keep, [x13, #0x0]\n"
      "str q30, [x13, #0x0]\n"
      "str q31, [x13, #0x10]\n"
      "subs x7, x7, #0x1\n"
      "bgt 1b\n"
      :
      : [gp] "r"(gp),
        [offsetof_A] "I"(offsetof(GemmParamsFP16, A)),
        [offsetof_B] "I"(offsetof(GemmParamsFP16, B)),
        [offsetof_C] "I"(offsetof(GemmParamsFP16, C)),
        [offsetof_b_block_cols] "I"(offsetof(GemmParamsFP16, b_block_cols)),
        [offsetof_beta] "I"(offsetof(GemmParamsFP16, beta)),
        [offsetof_k] "I"(offsetof(GemmParamsFP16, k)),
        [offsetof_lda] "I"(offsetof(GemmParamsFP16, lda)),
        [offsetof_ldc] "I"(offsetof(GemmParamsFP16, ldc))
      : "cc",
        "memory",
        "v0",
        "v1",
        "v16",
        "v18",
        "v19",
        "v2",
        "v20",
        "v21",
        "v22",
        "v23",
        "v24",
        "v25",
        "v27",
        "v28",
        "v29",
        "v3",
        "v30",
        "v31",
        "v4",
        "v5",
        "v6",
        "v7",
        "x10",
        "x11",
        "x12",
        "x13",
        "x14",
        "x15",
        "x5",
        "x6",
        "x7",
        "x8",
        "x9");
#endif // __aarch64__
}

void NOINLINE gemmkernel_3x1_Neon_fp16_fA0fB0fC0(GemmParamsFP16* gp) {
#if defined(__aarch64__) && __ARM_FEATURE_FP16_FML
  __asm__ __volatile__(
      "ldr s16, [%x[gp], %[offsetof_beta]]\n"
      "mov x7, #0x1\n"
      "fmov v25.8h, #1.0\n"
      "ldr x10, [%x[gp], %[offsetof_k]]\n"
      "ldr x11, [%x[gp], %[offsetof_A]]\n"
      "ldr x5, [%x[gp], %[offsetof_B]]\n"
      "ldr x4, [%x[gp], %[offsetof_C]]\n"
      "ldr x12, [%x[gp], %[offsetof_lda]]\n"
      "fcmp s16, #0.0\n"
      "csel x7, XZR, x7, EQ\n"
      "csel x7, XZR, x7, VS\n"
      "ldr x9, [%x[gp], %[offsetof_ldc]]\n"
      "ldr x6, [%x[gp], %[offsetof_b_block_cols]]\n"
      "1:" // Height 3: Column loop
      "tbz x7, #0, 2f\n"
      "ldr q26, [x4, #0x0]\n"
      "ldr q27, [x4, #0x10]\n"
      "add x13, %x[gp], %[offsetof_beta]\n"
      "ld1r { v16.4s }, [x13]\n"
      "add x13, x4, x9\n"
      "ldr q28, [x13, #0x0]\n"
      "ldr q29, [x13, #0x10]\n"
      "add x13, x13, x9\n"
      "ldr q30, [x13, #0x0]\n"
      "ldr q31, [x13, #0x10]\n"
      "fmul v26.4s, v26.4s, v16.4s\n"
      "fmul v27.4s, v27.4s, v16.4s\n"
      "fmul v28.4s, v28.4s, v16.4s\n"
      "fmul v29.4s, v29.4s, v16.4s\n"
      "fmul v30.4s, v30.4s, v16.4s\n"
      "fmul v31.4s, v31.4s, v16.4s\n"
      "b 3f\n"
      "2:" // Height 3: no accumulate
      "movi v26.16b, #0x0\n"
      "movi v27.16b, #0x0\n"
      "movi v28.16b, #0x0\n"
      "movi v29.16b, #0x0\n"
      "movi v30.16b, #0x0\n"
      "movi v31.16b, #0x0\n"
      "3:" // Height 3: setup done
      "add x14, x11, x12\n"
      "cmp x10, #0x4\n"
      "mov x8, x10\n"
      "mov x15, x11\n"
      "add x13, x14, x12\n"
      "blt 7f\n"
      "ldr q0, [x11, #0x0]\n"
      "ldr q3, [x5, #0x0]\n"
      "cmp x10, #0x8\n"
      "ldr q1, [x14, #0x0]\n"
      "ldr q2, [x13, #0x0]\n"
      "ldr q6, [x5, #0x10]\n"
      "ldr q19, [x5, #0x20]\n"
      "ldr q22, [x5, #0x30]\n"
      "blt 6f\n"
      "5:" // Height 3: Multiply loop: Main loop head
      "movi v4.16b, #0x0\n"
      "movi v5.16b, #0x0\n"
      "sub x8, x8, #0x4\n"
      "add x15, x15, #0x10\n"
      "movi v7.16b, #0x0\n"
      "movi v18.16b, #0x0\n"
      "add x14, x14, #0x10\n"
      "add x13, x13, #0x10\n"
      "fmlal v4.4s, v3.4h, v25.4h\n"
      "fmlal2 v5.4s, v3.4h, v25.4h\n"
      "movi v20.16b, #0x0\n"
      "cmp x8, #0x8\n"
      "fmlal v7.4s, v6.4h, v25.4h\n"
      "fmlal2 v18.4s, v6.4h, v25.4h\n"
      "movi v21.16b, #0x0\n"
      "ldr q3, [x5, #0x40]\n"
      "ldr q6, [x5, #0x50]\n"
      "fmlal v20.4s, v19.4h, v25.4h\n"
      "movi v23.16b, #0x0\n"
      "fmlal2 v21.4s, v19.4h, v25.4h\n"
      "ldr q19, [x5, #0x60]\n"
      "movi v24.16b, #0x0\n"
      "prfm pldl1keep, [x15, #0x80]\n"
      "fmla v26.4s, v4.4s, v0.s[0]\n"
      "fmla v28.4s, v4.4s, v1.s[0]\n"
      "prfm pldl1keep, [x14, #0x80]\n"
      "prfm pldl1keep, [x13, #0x80]\n"
      "fmla v30.4s, v4.4s, v2.s[0]\n"
      "fmla v27.4s, v5.4s, v0.s[0]\n"
      "fmla v29.4s, v5.4s, v1.s[0]\n"
      "fmla v31.4s, v5.4s, v2.s[0]\n"
      "fmlal v23.4s, v22.4h, v25.4h\n"
      "fmlal2 v24.4s, v22.4h, v25.4h\n"
      "ldr q22, [x5, #0x70]\n"
      "fmla v26.4s, v7.4s, v0.s[1]\n"
      "fmla v28.4s, v7.4s, v1.s[1]\n"
      "fmla v30.4s, v7.4s, v2.s[1]\n"
      "fmla v27.4s, v18.4s, v0.s[1]\n"
      "fmla v29.4s, v18.4s, v1.s[1]\n"
      "fmla v31.4s, v18.4s, v2.s[1]\n"
      "fmla v26.4s, v20.4s, v0.s[2]\n"
      "fmla v28.4s, v20.4s, v1.s[2]\n"
      "fmla v30.4s, v20.4s, v2.s[2]\n"
      "fmla v27.4s, v21.4s, v0.s[2]\n"
      "fmla v29.4s, v21.4s, v1.s[2]\n"
      "fmla v31.4s, v21.4s, v2.s[2]\n"
      "fmla v26.4s, v23.4s, v0.s[3]\n"
      "fmla v28.4s, v23.4s, v1.s[3]\n"
      "fmla v30.4s, v23.4s, v2.s[3]\n"
      "fmla v27.4s, v24.4s, v0.s[3]\n"
      "ldr q0, [x15, #0x0]\n"
      "fmla v29.4s, v24.4s, v1.s[3]\n"
      "ldr q1, [x14, #0x0]\n"
      "fmla v31.4s, v24.4s, v2.s[3]\n"
      "ldr q2, [x13, #0x0]\n"
      "add x5, x5, #0x40\n"
      "bge 5b\n"
      "6:" // Height 3: Multiply loop: Single iteration only
      "movi v4.16b, #0x0\n"
      "movi v5.16b, #0x0\n"
      "add x15, x15, #0x10\n"
      "add x14, x14, #0x10\n"
      "movi v7.16b, #0x0\n"
      "movi v18.16b, #0x0\n"
      "add x13, x13, #0x10\n"
      "sub x8, x8, #0x4\n"
      "fmlal v4.4s, v3.4h, v25.4h\n"
      "fmlal2 v5.4s, v3.4h, v25.4h\n"
      "movi v20.16b, #0x0\n"
      "prfm pldl1keep, [x15, #0x80]\n"
      "fmlal v7.4s, v6.4h, v25.4h\n"
      "fmlal2 v18.4s, v6.4h, v25.4h\n"
      "movi v21.16b, #0x0\n"
      "prfm pldl1keep, [x14, #0x80]\n"
      "fmlal v20.4s, v19.4h, v25.4h\n"
      "movi v23.16b, #0x0\n"
      "prfm pldl1keep, [x13, #0x80]\n"
      "add x5, x5, #0x40\n"
      "fmlal2 v21.4s, v19.4h, v25.4h\n"
      "movi v24.16b, #0x0\n"
      "fmla v26.4s, v4.4s, v0.s[0]\n"
      "fmla v28.4s, v4.4s, v1.s[0]\n"
      "fmla v30.4s, v4.4s, v2.s[0]\n"
      "fmla v27.4s, v5.4s, v0.s[0]\n"
      "fmla v29.4s, v5.4s, v1.s[0]\n"
      "fmla v31.4s, v5.4s, v2.s[0]\n"
      "fmlal v23.4s, v22.4h, v25.4h\n"
      "fmla v26.4s, v7.4s, v0.s[1]\n"
      "fmlal2 v24.4s, v22.4h, v25.4h\n"
      "fmla v28.4s, v7.4s, v1.s[1]\n"
      "fmla v30.4s, v7.4s, v2.s[1]\n"
      "fmla v27.4s, v18.4s, v0.s[1]\n"
      "fmla v29.4s, v18.4s, v1.s[1]\n"
      "fmla v31.4s, v18.4s, v2.s[1]\n"
      "fmla v26.4s, v20.4s, v0.s[2]\n"
      "fmla v28.4s, v20.4s, v1.s[2]\n"
      "fmla v30.4s, v20.4s, v2.s[2]\n"
      "fmla v27.4s, v21.4s, v0.s[2]\n"
      "fmla v29.4s, v21.4s, v1.s[2]\n"
      "fmla v31.4s, v21.4s, v2.s[2]\n"
      "fmla v26.4s, v23.4s, v0.s[3]\n"
      "fmla v28.4s, v23.4s, v1.s[3]\n"
      "fmla v30.4s, v23.4s, v2.s[3]\n"
      "fmla v27.4s, v24.4s, v0.s[3]\n"
      "fmla v29.4s, v24.4s, v1.s[3]\n"
      "fmla v31.4s, v24.4s, v2.s[3]\n"
      "7:" // Height 3: Multiply loop: Main loop skip
      "cbz x8, 9f\n"
      "8:" // Height 3: Multiply loop: Odd block loop
      "ldr q15, [x5, #0x0]\n"
      "ldr s0, [x15], #0x4\n"
      "movi v16.16b, #0x0\n"
      "movi v17.16b, #0x0\n"
      "ldr s1, [x14], #0x4\n"
      "ldr s2, [x13], #0x4\n"
      "sub x8, x8, #0x1\n"
      "add x5, x5, #0x10\n"
      "fmlal v16.4s, v15.4h, v25.4h\n"
      "fmlal2 v17.4s, v15.4h, v25.4h\n"
      "fmla v26.4s, v16.4s, v0.s[0]\n"
      "fmla v28.4s, v16.4s, v1.s[0]\n"
      "fmla v30.4s, v16.4s, v2.s[0]\n"
      "fmla v27.4s, v17.4s, v0.s[0]\n"
      "fmla v29.4s, v17.4s, v1.s[0]\n"
      "fmla v31.4s, v17.4s, v2.s[0]\n"
      "cbnz x8, 8b\n"
      "9:" // Height 3: Multiply loop: No odd multiplies
      "prfm pstl1keep, [x4, #0x0]\n"
      "str q26, [x4, #0x0]\n"
      "str q27, [x4, #0x10]\n"
      "add x14, x4, x9\n"
      "add x4, x4, #0x20\n"
      "prfm pstl1keep, [x14, #0x0]\n"
      "str q28, [x14, #0x0]\n"
      "add x13, x14, x9\n"
      "prfm pstl1keep, [x13, #0x0]\n"
      "str q29, [x14, #0x10]\n"
      "str q30, [x13, #0x0]\n"
      "str q31, [x13, #0x10]\n"
      "subs x6, x6, #0x1\n"
      "bgt 1b\n"
      :
      : [gp] "r"(gp),
        [offsetof_A] "I"(offsetof(GemmParamsFP16, A)),
        [offsetof_B] "I"(offsetof(GemmParamsFP16, B)),
        [offsetof_C] "I"(offsetof(GemmParamsFP16, C)),
        [offsetof_b_block_cols] "I"(offsetof(GemmParamsFP16, b_block_cols)),
        [offsetof_beta] "I"(offsetof(GemmParamsFP16, beta)),
        [offsetof_k] "I"(offsetof(GemmParamsFP16, k)),
        [offsetof_lda] "I"(offsetof(GemmParamsFP16, lda)),
        [offsetof_ldc] "I"(offsetof(GemmParamsFP16, ldc))
      : "cc",
        "memory",
        "v0",
        "v1",
        "v15",
        "v16",
        "v17",
        "v18",
        "v19",
        "v2",
        "v20",
        "v21",
        "v22",
        "v23",
        "v24",
        "v25",
        "v26",
        "v27",
        "v28",
        "v29",
        "v3",
        "v30",
        "v31",
        "v4",
        "v5",
        "v6",
        "v7",
        "x10",
        "x11",
        "x12",
        "x13",
        "x14",
        "x15",
        "x4",
        "x5",
        "x6",
        "x7",
        "x8",
        "x9");
#endif // __aarch64__
}

void NOINLINE gemmkernel_4x1_Neon_fp16_fA0fB0fC0(GemmParamsFP16* gp) {
#if defined(__aarch64__) && __ARM_FEATURE_FP16_FML
  __asm__ __volatile__(
      "ldr s16, [%x[gp], %[offsetof_beta]]\n"
      "mov x8, #0x1\n"
      "fmov v23.8h, #1.0\n"
      "ldr x10, [%x[gp], %[offsetof_k]]\n"
      "ldr x11, [%x[gp], %[offsetof_A]]\n"
      "ldr x6, [%x[gp], %[offsetof_B]]\n"
      "ldr x5, [%x[gp], %[offsetof_C]]\n"
      "ldr x12, [%x[gp], %[offsetof_lda]]\n"
      "fcmp s16, #0.0\n"
      "csel x8, XZR, x8, EQ\n"
      "csel x8, XZR, x8, VS\n"
      "ldr x9, [%x[gp], %[offsetof_ldc]]\n"
      "ldr x7, [%x[gp], %[offsetof_b_block_cols]]\n"
      "1:" // Height 4: Column loop
      "tbz x8, #0, 2f\n"
      "ldr q24, [x5, #0x0]\n"
      "ldr q25, [x5, #0x10]\n"
      "add x13, %x[gp], %[offsetof_beta]\n"
      "ld1r { v16.4s }, [x13]\n"
      "add x13, x5, x9\n"
      "ldr q26, [x13, #0x0]\n"
      "ldr q27, [x13, #0x10]\n"
      "add x13, x13, x9\n"
      "ldr q28, [x13, #0x0]\n"
      "ldr q29, [x13, #0x10]\n"
      "add x13, x13, x9\n"
      "ldr q30, [x13, #0x0]\n"
      "ldr q31, [x13, #0x10]\n"
      "fmul v24.4s, v24.4s, v16.4s\n"
      "fmul v25.4s, v25.4s, v16.4s\n"
      "fmul v26.4s, v26.4s, v16.4s\n"
      "fmul v27.4s, v27.4s, v16.4s\n"
      "fmul v28.4s, v28.4s, v16.4s\n"
      "fmul v29.4s, v29.4s, v16.4s\n"
      "fmul v30.4s, v30.4s, v16.4s\n"
      "fmul v31.4s, v31.4s, v16.4s\n"
      "b 3f\n"
      "2:" // Height 4: no accumulate
      "movi v24.16b, #0x0\n"
      "movi v25.16b, #0x0\n"
      "movi v26.16b, #0x0\n"
      "movi v27.16b, #0x0\n"
      "movi v28.16b, #0x0\n"
      "movi v29.16b, #0x0\n"
      "movi v30.16b, #0x0\n"
      "movi v31.16b, #0x0\n"
      "3:" // Height 4: setup done
      "add x15, x11, x12\n"
      "cmp x10, #0x4\n"
      "mov x4, x10\n"
      "mov x3, x11\n"
      "add x14, x15, x12\n"
      "add x13, x14, x12\n"
      "blt 7f\n"
      "ldr q0, [x11, #0x0]\n"
      "ldr q4, [x6, #0x0]\n"
      "cmp x10, #0x8\n"
      "ldr q1, [x15, #0x0]\n"
      "ldr q2, [x14, #0x0]\n"
      "ldr q3, [x13, #0x0]\n"
      "ldr q7, [x6, #0x10]\n"
      "ldr q21, [x6, #0x20]\n"
      "ldr q13, [x6, #0x30]\n"
      "blt 6f\n"
      "5:" // Height 4: Multiply loop: Main loop head
      "movi v5.16b, #0x0\n"
      "movi v6.16b, #0x0\n"
      "sub x4, x4, #0x4\n"
      "add x3, x3, #0x10\n"
      "movi v20.16b, #0x0\n"
      "movi v19.16b, #0x0\n"
      "add x15, x15, #0x10\n"
      "add x14, x14, #0x10\n"
      "fmlal v5.4s, v4.4h, v23.4h\n"
      "fmlal2 v6.4s, v4.4h, v23.4h\n"
      "movi v22.16b, #0x0\n"
      "add x13, x13, #0x10\n"
      "fmlal v20.4s, v7.4h, v23.4h\n"
      "fmlal2 v19.4s, v7.4h, v23.4h\n"
      "movi v12.16b, #0x0\n"
      "cmp x4, #0x8\n"
      "fmlal v22.4s, v21.4h, v23.4h\n"
      "movi v14.16b, #0x0\n"
      "prfm pldl1keep, [x3, #0x80]\n"
      "ldr q4, [x6, #0x40]\n"
      "ldr q7, [x6, #0x50]\n"
      "fmlal2 v12.4s, v21.4h, v23.4h\n"
      "movi v15.16b, #0x0\n"
      "ldr q21, [x6, #0x60]\n"
      "fmla v24.4s, v5.4s, v0.s[0]\n"
      "fmla v26.4s, v5.4s, v1.s[0]\n"
      "prfm pldl1keep, [x15, #0x80]\n"
      "fmla v28.4s, v5.4s, v2.s[0]\n"
      "fmla v30.4s, v5.4s, v3.s[0]\n"
      "prfm pldl1keep, [x14, #0x80]\n"
      "prfm pldl1keep, [x13, #0x80]\n"
      "fmla v25.4s, v6.4s, v0.s[0]\n"
      "fmla v27.4s, v6.4s, v1.s[0]\n"
      "fmla v29.4s, v6.4s, v2.s[0]\n"
      "fmla v31.4s, v6.4s, v3.s[0]\n"
      "fmla v24.4s, v20.4s, v0.s[1]\n"
      "fmla v26.4s, v20.4s, v1.s[1]\n"
      "fmla v28.4s, v20.4s, v2.s[1]\n"
      "fmla v30.4s, v20.4s, v3.s[1]\n"
      "fmla v25.4s, v19.4s, v0.s[1]\n"
      "fmla v27.4s, v19.4s, v1.s[1]\n"
      "fmla v29.4s, v19.4s, v2.s[1]\n"
      "fmla v31.4s, v19.4s, v3.s[1]\n"
      "fmlal v14.4s, v13.4h, v23.4h\n"
      "fmla v24.4s, v22.4s, v0.s[2]\n"
      "fmla v26.4s, v22.4s, v1.s[2]\n"
      "fmla v28.4s, v22.4s, v2.s[2]\n"
      "fmla v30.4s, v22.4s, v3.s[2]\n"
      "fmla v25.4s, v12.4s, v0.s[2]\n"
      "fmla v27.4s, v12.4s, v1.s[2]\n"
      "fmla v29.4s, v12.4s, v2.s[2]\n"
      "fmla v31.4s, v12.4s, v3.s[2]\n"
      "fmlal2 v15.4s, v13.4h, v23.4h\n"
      "ldr q13, [x6, #0x70]\n"
      "fmla v24.4s, v14.4s, v0.s[3]\n"
      "fmla v26.4s, v14.4s, v1.s[3]\n"
      "fmla v28.4s, v14.4s, v2.s[3]\n"
      "fmla v30.4s, v14.4s, v3.s[3]\n"
      "fmla v25.4s, v15.4s, v0.s[3]\n"
      "ldr q0, [x3, #0x0]\n"
      "fmla v27.4s, v15.4s, v1.s[3]\n"
      "ldr q1, [x15, #0x0]\n"
      "fmla v29.4s, v15.4s, v2.s[3]\n"
      "ldr q2, [x14, #0x0]\n"
      "fmla v31.4s, v15.4s, v3.s[3]\n"
      "ldr q3, [x13, #0x0]\n"
      "add x6, x6, #0x40\n"
      "bge 5b\n"
      "6:" // Height 4: Multiply loop: Single iteration only
      "movi v5.16b, #0x0\n"
      "movi v6.16b, #0x0\n"
      "add x3, x3, #0x10\n"
      "add x15, x15, #0x10\n"
      "movi v20.16b, #0x0\n"
      "movi v19.16b, #0x0\n"
      "add x14, x14, #0x10\n"
      "add x13, x13, #0x10\n"
      "fmlal v5.4s, v4.4h, v23.4h\n"
      "fmlal2 v6.4s, v4.4h, v23.4h\n"
      "movi v22.16b, #0x0\n"
      "sub x4, x4, #0x4\n"
      "fmlal v20.4s, v7.4h, v23.4h\n"
      "fmlal2 v19.4s, v7.4h, v23.4h\n"
      "movi v12.16b, #0x0\n"
      "prfm pldl1keep, [x3, #0x80]\n"
      "fmlal v22.4s, v21.4h, v23.4h\n"
      "movi v14.16b, #0x0\n"
      "prfm pldl1keep, [x15, #0x80]\n"
      "prfm pldl1keep, [x14, #0x80]\n"
      "fmlal2 v12.4s, v21.4h, v23.4h\n"
      "movi v15.16b, #0x0\n"
      "prfm pldl1keep, [x13, #0x80]\n"
      "add x6, x6, #0x40\n"
      "fmla v24.4s, v5.4s, v0.s[0]\n"
      "fmla v26.4s, v5.4s, v1.s[0]\n"
      "fmla v28.4s, v5.4s, v2.s[0]\n"
      "fmla v30.4s, v5.4s, v3.s[0]\n"
      "fmla v25.4s, v6.4s, v0.s[0]\n"
      "fmla v27.4s, v6.4s, v1.s[0]\n"
      "fmla v29.4s, v6.4s, v2.s[0]\n"
      "fmla v31.4s, v6.4s, v3.s[0]\n"
      "fmla v24.4s, v20.4s, v0.s[1]\n"
      "fmla v26.4s, v20.4s, v1.s[1]\n"
      "fmla v28.4s, v20.4s, v2.s[1]\n"
      "fmla v30.4s, v20.4s, v3.s[1]\n"
      "fmla v25.4s, v19.4s, v0.s[1]\n"
      "fmla v27.4s, v19.4s, v1.s[1]\n"
      "fmla v29.4s, v19.4s, v2.s[1]\n"
      "fmla v31.4s, v19.4s, v3.s[1]\n"
      "fmlal v14.4s, v13.4h, v23.4h\n"
      "fmla v24.4s, v22.4s, v0.s[2]\n"
      "fmla v26.4s, v22.4s, v1.s[2]\n"
      "fmla v28.4s, v22.4s, v2.s[2]\n"
      "fmla v30.4s, v22.4s, v3.s[2]\n"
      "fmla v25.4s, v12.4s, v0.s[2]\n"
      "fmla v27.4s, v12.4s, v1.s[2]\n"
      "fmla v29.4s, v12.4s, v2.s[2]\n"
      "fmla v31.4s, v12.4s, v3.s[2]\n"
      "fmlal2 v15.4s, v13.4h, v23.4h\n"
      "fmla v24.4s, v14.4s, v0.s[3]\n"
      "fmla v26.4s, v14.4s, v1.s[3]\n"
      "fmla v28.4s, v14.4s, v2.s[3]\n"
      "fmla v30.4s, v14.4s, v3.s[3]\n"
      "fmla v25.4s, v15.4s, v0.s[3]\n"
      "fmla v27.4s, v15.4s, v1.s[3]\n"
      "fmla v29.4s, v15.4s, v2.s[3]\n"
      "fmla v31.4s, v15.4s, v3.s[3]\n"
      "7:" // Height 4: Multiply loop: Main loop skip
      "cbz x4, 9f\n"
      "8:" // Height 4: Multiply loop: Odd block loop
      "ldr q16, [x6, #0x0]\n"
      "ldr s0, [x3], #0x4\n"
      "movi v17.16b, #0x0\n"
      "movi v18.16b, #0x0\n"
      "ldr s1, [x15], #0x4\n"
      "ldr s2, [x14], #0x4\n"
      "sub x4, x4, #0x1\n"
      "add x6, x6, #0x10\n"
      "ldr s3, [x13], #0x4\n"
      "fmlal v17.4s, v16.4h, v23.4h\n"
      "fmlal2 v18.4s, v16.4h, v23.4h\n"
      "fmla v24.4s, v17.4s, v0.s[0]\n"
      "fmla v26.4s, v17.4s, v1.s[0]\n"
      "fmla v28.4s, v17.4s, v2.s[0]\n"
      "fmla v30.4s, v17.4s, v3.s[0]\n"
      "fmla v25.4s, v18.4s, v0.s[0]\n"
      "fmla v27.4s, v18.4s, v1.s[0]\n"
      "fmla v29.4s, v18.4s, v2.s[0]\n"
      "fmla v31.4s, v18.4s, v3.s[0]\n"
      "cbnz x4, 8b\n"
      "9:" // Height 4: Multiply loop: No odd multiplies
      "prfm pstl1keep, [x5, #0x0]\n"
      "str q24, [x5, #0x0]\n"
      "str q25, [x5, #0x10]\n"
      "add x15, x5, x9\n"
      "add x5, x5, #0x20\n"
      "prfm pstl1keep, [x15, #0x0]\n"
      "str q26, [x15, #0x0]\n"
      "add x14, x15, x9\n"
      "add x13, x14, x9\n"
      "prfm pstl1keep, [x14, #0x0]\n"
      "prfm pstl1keep, [x13, #0x0]\n"
      "str q27, [x15, #0x10]\n"
      "str q28, [x14, #0x0]\n"
      "str q29, [x14, #0x10]\n"
      "str q30, [x13, #0x0]\n"
      "str q31, [x13, #0x10]\n"
      "subs x7, x7, #0x1\n"
      "bgt 1b\n"
      :
      : [gp] "r"(gp),
        [offsetof_A] "I"(offsetof(GemmParamsFP16, A)),
        [offsetof_B] "I"(offsetof(GemmParamsFP16, B)),
        [offsetof_C] "I"(offsetof(GemmParamsFP16, C)),
        [offsetof_b_block_cols] "I"(offsetof(GemmParamsFP16, b_block_cols)),
        [offsetof_beta] "I"(offsetof(GemmParamsFP16, beta)),
        [offsetof_k] "I"(offsetof(GemmParamsFP16, k)),
        [offsetof_lda] "I"(offsetof(GemmParamsFP16, lda)),
        [offsetof_ldc] "I"(offsetof(GemmParamsFP16, ldc))
      : "cc",
        "memory",
        "v0",
        "v1",
        "v12",
        "v13",
        "v14",
        "v15",
        "v16",
        "v17",
        "v18",
        "v19",
        "v2",
        "v20",
        "v21",
        "v22",
        "v23",
        "v24",
        "v25",
        "v26",
        "v27",
        "v28",
        "v29",
        "v3",
        "v30",
        "v31",
        "v4",
        "v5",
        "v6",
        "v7",
        "x10",
        "x11",
        "x12",
        "x13",
        "x14",
        "x15",
        "x3",
        "x4",
        "x5",
        "x6",
        "x7",
        "x8",
        "x9");
#endif // __aarch64__
}

void NOINLINE gemmkernel_5x1_Neon_fp16_fA0fB0fC0(GemmParamsFP16* gp) {
#if defined(__aarch64__) && __ARM_FEATURE_FP16_FML
  __asm__ __volatile__(
      "ldr s16, [%x[gp], %[offsetof_beta]]\n"
      "mov x9, #0x1\n"
      "fmov v21.8h, #1.0\n"
      "ldr x12, [%x[gp], %[offsetof_k]]\n"
      "ldr x10, [%x[gp], %[offsetof_A]]\n"
      "ldr x7, [%x[gp], %[offsetof_B]]\n"
      "ldr x6, [%x[gp], %[offsetof_C]]\n"
      "ldr x11, [%x[gp], %[offsetof_lda]]\n"
      "fcmp s16, #0.0\n"
      "csel x9, XZR, x9, EQ\n"
      "csel x9, XZR, x9, VS\n"
      "ldr x13, [%x[gp], %[offsetof_ldc]]\n"
      "ldr x8, [%x[gp], %[offsetof_b_block_cols]]\n"
      "1:" // Height 5: Column loop
      "tbz x9, #0, 2f\n"
      "ldr q22, [x6, #0x0]\n"
      "ldr q23, [x6, #0x10]\n"
      "add x14, %x[gp], %[offsetof_beta]\n"
      "ld1r { v16.4s }, [x14]\n"
      "add x14, x6, x13\n"
      "ldr q24, [x14, #0x0]\n"
      "ldr q25, [x14, #0x10]\n"
      "add x14, x14, x13\n"
      "ldr q26, [x14, #0x0]\n"
      "ldr q27, [x14, #0x10]\n"
      "add x14, x14, x13\n"
      "ldr q28, [x14, #0x0]\n"
      "ldr q29, [x14, #0x10]\n"
      "add x14, x14, x13\n"
      "fmul v22.4s, v22.4s, v16.4s\n"
      "ldr q30, [x14, #0x0]\n"
      "ldr q31, [x14, #0x10]\n"
      "fmul v23.4s, v23.4s, v16.4s\n"
      "fmul v24.4s, v24.4s, v16.4s\n"
      "fmul v25.4s, v25.4s, v16.4s\n"
      "fmul v26.4s, v26.4s, v16.4s\n"
      "fmul v27.4s, v27.4s, v16.4s\n"
      "fmul v28.4s, v28.4s, v16.4s\n"
      "fmul v29.4s, v29.4s, v16.4s\n"
      "fmul v30.4s, v30.4s, v16.4s\n"
      "fmul v31.4s, v31.4s, v16.4s\n"
      "b 3f\n"
      "2:" // Height 5: no accumulate
      "movi v22.16b, #0x0\n"
      "movi v23.16b, #0x0\n"
      "movi v24.16b, #0x0\n"
      "movi v25.16b, #0x0\n"
      "movi v26.16b, #0x0\n"
      "movi v27.16b, #0x0\n"
      "movi v28.16b, #0x0\n"
      "movi v29.16b, #0x0\n"
      "movi v30.16b, #0x0\n"
      "movi v31.16b, #0x0\n"
      "3:" // Height 5: setup done
      "add x3, x10, x11\n"
      "cmp x12, #0x4\n"
      "mov x5, x12\n"
      "mov x4, x10\n"
      "add x2, x3, x11\n"
      "add x15, x2, x11\n"
      "add x14, x15, x11\n"
      "blt 7f\n"
      "ldr q0, [x10, #0x0]\n"
      "ldr q5, [x7, #0x0]\n"
      "cmp x12, #0x8\n"
      "ldr q1, [x3, #0x0]\n"
      "ldr q2, [x2, #0x0]\n"
      "ldr q3, [x15, #0x0]\n"
      "ldr q4, [x14, #0x0]\n"
      "ldr q8, [x7, #0x10]\n"
      "ldr q11, [x7, #0x20]\n"
      "ldr q14, [x7, #0x30]\n"
      "blt 6f\n"
      "5:" // Height 5: Multiply loop: Main loop head
      "movi v6.16b, #0x0\n"
      "movi v7.16b, #0x0\n"
      "sub x5, x5, #0x4\n"
      "add x4, x4, #0x10\n"
      "movi v9.16b, #0x0\n"
      "movi v10.16b, #0x0\n"
      "add x3, x3, #0x10\n"
      "add x2, x2, #0x10\n"
      "fmlal v6.4s, v5.4h, v21.4h\n"
      "fmlal2 v7.4s, v5.4h, v21.4h\n"
      "movi v12.16b, #0x0\n"
      "add x15, x15, #0x10\n"
      "fmlal v9.4s, v8.4h, v21.4h\n"
      "fmlal2 v10.4s, v8.4h, v21.4h\n"
      "movi v13.16b, #0x0\n"
      "add x14, x14, #0x10\n"
      "fmlal v12.4s, v11.4h, v21.4h\n"
      "movi v20.16b, #0x0\n"
      "cmp x5, #0x8\n"
      "add x7, x7, #0x40\n"
      "ldr q5, [x7, #0x0]\n"
      "ldr q8, [x7, #0x10]\n"
      "fmlal2 v13.4s, v11.4h, v21.4h\n"
      "movi v16.16b, #0x0\n"
      "ldr q11, [x7, #0x20]\n"
      "fmla v22.4s, v6.4s, v0.s[0]\n"
      "fmla v24.4s, v6.4s, v1.s[0]\n"
      "prfm pldl1keep, [x4, #0x80]\n"
      "fmla v26.4s, v6.4s, v2.s[0]\n"
      "fmla v28.4s, v6.4s, v3.s[0]\n"
      "prfm pldl1keep, [x3, #0x80]\n"
      "prfm pldl1keep, [x2, #0x80]\n"
      "fmla v30.4s, v6.4s, v4.s[0]\n"
      "fmla v23.4s, v7.4s, v0.s[0]\n"
      "prfm pldl1keep, [x15, #0x80]\n"
      "prfm pldl1keep, [x14, #0x80]\n"
      "fmla v25.4s, v7.4s, v1.s[0]\n"
      "fmla v27.4s, v7.4s, v2.s[0]\n"
      "fmla v29.4s, v7.4s, v3.s[0]\n"
      "fmla v31.4s, v7.4s, v4.s[0]\n"
      "fmla v22.4s, v9.4s, v0.s[1]\n"
      "fmla v24.4s, v9.4s, v1.s[1]\n"
      "fmla v26.4s, v9.4s, v2.s[1]\n"
      "fmla v28.4s, v9.4s, v3.s[1]\n"
      "fmla v30.4s, v9.4s, v4.s[1]\n"
      "fmla v23.4s, v10.4s, v0.s[1]\n"
      "fmla v25.4s, v10.4s, v1.s[1]\n"
      "fmla v27.4s, v10.4s, v2.s[1]\n"
      "fmla v29.4s, v10.4s, v3.s[1]\n"
      "fmla v31.4s, v10.4s, v4.s[1]\n"
      "fmla v22.4s, v12.4s, v0.s[2]\n"
      "fmla v24.4s, v12.4s, v1.s[2]\n"
      "fmla v26.4s, v12.4s, v2.s[2]\n"
      "fmla v28.4s, v12.4s, v3.s[2]\n"
      "fmla v30.4s, v12.4s, v4.s[2]\n"
      "fmla v23.4s, v13.4s, v0.s[2]\n"
      "fmla v25.4s, v13.4s, v1.s[2]\n"
      "fmla v27.4s, v13.4s, v2.s[2]\n"
      "fmla v29.4s, v13.4s, v3.s[2]\n"
      "fmla v31.4s, v13.4s, v4.s[2]\n"
      "fmlal v20.4s, v14.4h, v21.4h\n"
      "fmlal2 v16.4s, v14.4h, v21.4h\n"
      "ldr q14, [x7, #0x30]\n"
      "fmla v22.4s, v20.4s, v0.s[3]\n"
      "fmla v24.4s, v20.4s, v1.s[3]\n"
      "fmla v26.4s, v20.4s, v2.s[3]\n"
      "fmla v28.4s, v20.4s, v3.s[3]\n"
      "fmla v30.4s, v20.4s, v4.s[3]\n"
      "fmla v23.4s, v16.4s, v0.s[3]\n"
      "ldr q0, [x4, #0x0]\n"
      "fmla v25.4s, v16.4s, v1.s[3]\n"
      "ldr q1, [x3, #0x0]\n"
      "fmla v27.4s, v16.4s, v2.s[3]\n"
      "ldr q2, [x2, #0x0]\n"
      "fmla v29.4s, v16.4s, v3.s[3]\n"
      "ldr q3, [x15, #0x0]\n"
      "fmla v31.4s, v16.4s, v4.s[3]\n"
      "ldr q4, [x14, #0x0]\n"
      "bge 5b\n"
      "6:" // Height 5: Multiply loop: Single iteration only
      "movi v6.16b, #0x0\n"
      "movi v7.16b, #0x0\n"
      "add x4, x4, #0x10\n"
      "add x3, x3, #0x10\n"
      "movi v9.16b, #0x0\n"
      "movi v10.16b, #0x0\n"
      "add x2, x2, #0x10\n"
      "add x15, x15, #0x10\n"
      "fmlal v6.4s, v5.4h, v21.4h\n"
      "fmlal2 v7.4s, v5.4h, v21.4h\n"
      "movi v12.16b, #0x0\n"
      "add x14, x14, #0x10\n"
      "fmlal v9.4s, v8.4h, v21.4h\n"
      "fmlal2 v10.4s, v8.4h, v21.4h\n"
      "movi v13.16b, #0x0\n"
      "sub x5, x5, #0x4\n"
      "fmlal v12.4s, v11.4h, v21.4h\n"
      "movi v20.16b, #0x0\n"
      "prfm pldl1keep, [x4, #0x80]\n"
      "prfm pldl1keep, [x3, #0x80]\n"
      "fmlal2 v13.4s, v11.4h, v21.4h\n"
      "movi v16.16b, #0x0\n"
      "prfm pldl1keep, [x2, #0x80]\n"
      "prfm pldl1keep, [x15, #0x80]\n"
      "fmla v22.4s, v6.4s, v0.s[0]\n"
      "fmla v24.4s, v6.4s, v1.s[0]\n"
      "prfm pldl1keep, [x14, #0x80]\n"
      "add x7, x7, #0x40\n"
      "fmla v26.4s, v6.4s, v2.s[0]\n"
      "fmla v28.4s, v6.4s, v3.s[0]\n"
      "fmla v30.4s, v6.4s, v4.s[0]\n"
      "fmla v23.4s, v7.4s, v0.s[0]\n"
      "fmla v25.4s, v7.4s, v1.s[0]\n"
      "fmla v27.4s, v7.4s, v2.s[0]\n"
      "fmla v29.4s, v7.4s, v3.s[0]\n"
      "fmla v31.4s, v7.4s, v4.s[0]\n"
      "fmla v22.4s, v9.4s, v0.s[1]\n"
      "fmla v24.4s, v9.4s, v1.s[1]\n"
      "fmla v26.4s, v9.4s, v2.s[1]\n"
      "fmla v28.4s, v9.4s, v3.s[1]\n"
      "fmla v30.4s, v9.4s, v4.s[1]\n"
      "fmla v23.4s, v10.4s, v0.s[1]\n"
      "fmla v25.4s, v10.4s, v1.s[1]\n"
      "fmla v27.4s, v10.4s, v2.s[1]\n"
      "fmla v29.4s, v10.4s, v3.s[1]\n"
      "fmla v31.4s, v10.4s, v4.s[1]\n"
      "fmla v22.4s, v12.4s, v0.s[2]\n"
      "fmla v24.4s, v12.4s, v1.s[2]\n"
      "fmla v26.4s, v12.4s, v2.s[2]\n"
      "fmla v28.4s, v12.4s, v3.s[2]\n"
      "fmla v30.4s, v12.4s, v4.s[2]\n"
      "fmla v23.4s, v13.4s, v0.s[2]\n"
      "fmla v25.4s, v13.4s, v1.s[2]\n"
      "fmla v27.4s, v13.4s, v2.s[2]\n"
      "fmla v29.4s, v13.4s, v3.s[2]\n"
      "fmla v31.4s, v13.4s, v4.s[2]\n"
      "fmlal v20.4s, v14.4h, v21.4h\n"
      "fmlal2 v16.4s, v14.4h, v21.4h\n"
      "fmla v22.4s, v20.4s, v0.s[3]\n"
      "fmla v24.4s, v20.4s, v1.s[3]\n"
      "fmla v26.4s, v20.4s, v2.s[3]\n"
      "fmla v28.4s, v20.4s, v3.s[3]\n"
      "fmla v30.4s, v20.4s, v4.s[3]\n"
      "fmla v23.4s, v16.4s, v0.s[3]\n"
      "fmla v25.4s, v16.4s, v1.s[3]\n"
      "fmla v27.4s, v16.4s, v2.s[3]\n"
      "fmla v29.4s, v16.4s, v3.s[3]\n"
      "fmla v31.4s, v16.4s, v4.s[3]\n"
      "7:" // Height 5: Multiply loop: Main loop skip
      "cbz x5, 9f\n"
      "8:" // Height 5: Multiply loop: Odd block loop
      "ldr q17, [x7, #0x0]\n"
      "ldr s0, [x4], #0x4\n"
      "movi v18.16b, #0x0\n"
      "movi v19.16b, #0x0\n"
      "ldr s1, [x3], #0x4\n"
      "ldr s2, [x2], #0x4\n"
      "sub x5, x5, #0x1\n"
      "add x7, x7, #0x10\n"
      "ldr s3, [x15], #0x4\n"
      "ldr s4, [x14], #0x4\n"
      "fmlal v18.4s, v17.4h, v21.4h\n"
      "fmlal2 v19.4s, v17.4h, v21.4h\n"
      "fmla v22.4s, v18.4s, v0.s[0]\n"
      "fmla v24.4s, v18.4s, v1.s[0]\n"
      "fmla v26.4s, v18.4s, v2.s[0]\n"
      "fmla v28.4s, v18.4s, v3.s[0]\n"
      "fmla v30.4s, v18.4s, v4.s[0]\n"
      "fmla v23.4s, v19.4s, v0.s[0]\n"
      "fmla v25.4s, v19.4s, v1.s[0]\n"
      "fmla v27.4s, v19.4s, v2.s[0]\n"
      "fmla v29.4s, v19.4s, v3.s[0]\n"
      "fmla v31.4s, v19.4s, v4.s[0]\n"
      "cbnz x5, 8b\n"
      "9:" // Height 5: Multiply loop: No odd multiplies
      "prfm pstl1keep, [x6, #0x0]\n"
      "str q22, [x6, #0x0]\n"
      "str q23, [x6, #0x10]\n"
      "add x3, x6, x13\n"
      "add x6, x6, #0x20\n"
      "prfm pstl1keep, [x3, #0x0]\n"
      "str q24, [x3, #0x0]\n"
      "add x2, x3, x13\n"
      "add x15, x2, x13\n"
      "add x14, x15, x13\n"
      "prfm pstl1keep, [x2, #0x0]\n"
      "prfm pstl1keep, [x15, #0x0]\n"
      "str q25, [x3, #0x10]\n"
      "prfm pstl1keep, [x14, #0x0]\n"
      "str q26, [x2, #0x0]\n"
      "str q27, [x2, #0x10]\n"
      "str q28, [x15, #0x0]\n"
      "str q29, [x15, #0x10]\n"
      "str q30, [x14, #0x0]\n"
      "str q31, [x14, #0x10]\n"
      "subs x8, x8, #0x1\n"
      "bgt 1b\n"
      :
      : [gp] "r"(gp),
        [offsetof_A] "I"(offsetof(GemmParamsFP16, A)),
        [offsetof_B] "I"(offsetof(GemmParamsFP16, B)),
        [offsetof_C] "I"(offsetof(GemmParamsFP16, C)),
        [offsetof_b_block_cols] "I"(offsetof(GemmParamsFP16, b_block_cols)),
        [offsetof_beta] "I"(offsetof(GemmParamsFP16, beta)),
        [offsetof_k] "I"(offsetof(GemmParamsFP16, k)),
        [offsetof_lda] "I"(offsetof(GemmParamsFP16, lda)),
        [offsetof_ldc] "I"(offsetof(GemmParamsFP16, ldc))
      : "cc",
        "memory",
        "v0",
        "v1",
        "v10",
        "v11",
        "v12",
        "v13",
        "v14",
        "v16",
        "v17",
        "v18",
        "v19",
        "v2",
        "v20",
        "v21",
        "v22",
        "v23",
        "v24",
        "v25",
        "v26",
        "v27",
        "v28",
        "v29",
        "v3",
        "v30",
        "v31",
        "v4",
        "v5",
        "v6",
        "v7",
        "v8",
        "v9",
        "x10",
        "x11",
        "x12",
        "x13",
        "x14",
        "x15",
        "x2",
        "x3",
        "x4",
        "x5",
        "x6",
        "x7",
        "x8",
        "x9");
#endif // __aarch64__
}

void NOINLINE gemmkernel_6x1_Neon_fp16_fA0fB0fC0(GemmParamsFP16* gp) {
#if defined(__aarch64__) && __ARM_FEATURE_FP16_FML
  __asm__ __volatile__(
      "ldr s16, [%x[gp], %[offsetof_beta]]\n"
      "mov x10, #0x1\n"
      "fmov v19.8h, #1.0\n"
      "ldr x13, [%x[gp], %[offsetof_k]]\n"
      "ldr x11, [%x[gp], %[offsetof_A]]\n"
      "ldr x8, [%x[gp], %[offsetof_B]]\n"
      "ldr x7, [%x[gp], %[offsetof_C]]\n"
      "ldr x12, [%x[gp], %[offsetof_lda]]\n"
      "fcmp s16, #0.0\n"
      "csel x10, XZR, x10, EQ\n"
      "csel x10, XZR, x10, VS\n"
      "ldr x14, [%x[gp], %[offsetof_ldc]]\n"
      "ldr x9, [%x[gp], %[offsetof_b_block_cols]]\n"
      "1:" // Height 6: Column loop
      "tbz x10, #0, 2f\n"
      "ldr q20, [x7, #0x0]\n"
      "ldr q21, [x7, #0x10]\n"
      "add x15, %x[gp], %[offsetof_beta]\n"
      "ld1r { v16.4s }, [x15]\n"
      "add x15, x7, x14\n"
      "ldr q22, [x15, #0x0]\n"
      "ldr q23, [x15, #0x10]\n"
      "add x15, x15, x14\n"
      "ldr q24, [x15, #0x0]\n"
      "ldr q25, [x15, #0x10]\n"
      "add x15, x15, x14\n"
      "ldr q26, [x15, #0x0]\n"
      "ldr q27, [x15, #0x10]\n"
      "add x15, x15, x14\n"
      "fmul v20.4s, v20.4s, v16.4s\n"
      "ldr q28, [x15, #0x0]\n"
      "ldr q29, [x15, #0x10]\n"
      "add x15, x15, x14\n"
      "fmul v21.4s, v21.4s, v16.4s\n"
      "ldr q30, [x15, #0x0]\n"
      "ldr q31, [x15, #0x10]\n"
      "fmul v22.4s, v22.4s, v16.4s\n"
      "fmul v23.4s, v23.4s, v16.4s\n"
      "fmul v24.4s, v24.4s, v16.4s\n"
      "fmul v25.4s, v25.4s, v16.4s\n"
      "fmul v26.4s, v26.4s, v16.4s\n"
      "fmul v27.4s, v27.4s, v16.4s\n"
      "fmul v28.4s, v28.4s, v16.4s\n"
      "fmul v29.4s, v29.4s, v16.4s\n"
      "fmul v30.4s, v30.4s, v16.4s\n"
      "fmul v31.4s, v31.4s, v16.4s\n"
      "b 3f\n"
      "2:" // Height 6: no accumulate
      "movi v20.16b, #0x0\n"
      "movi v21.16b, #0x0\n"
      "movi v22.16b, #0x0\n"
      "movi v23.16b, #0x0\n"
      "movi v24.16b, #0x0\n"
      "movi v25.16b, #0x0\n"
      "movi v26.16b, #0x0\n"
      "movi v27.16b, #0x0\n"
      "movi v28.16b, #0x0\n"
      "movi v29.16b, #0x0\n"
      "movi v30.16b, #0x0\n"
      "movi v31.16b, #0x0\n"
      "3:" // Height 6: setup done
      "add x4, x11, x12\n"
      "cmp x13, #0x4\n"
      "mov x6, x13\n"
      "mov x5, x11\n"
      "add x3, x4, x12\n"
      "add x2, x3, x12\n"
      "add x1, x2, x12\n"
      "add x15, x1, x12\n"
      "blt 7f\n"
      "ldr q0, [x11, #0x0]\n"
      "ldr q6, [x8, #0x0]\n"
      "cmp x13, #0x8\n"
      "ldr q1, [x4, #0x0]\n"
      "ldr q2, [x3, #0x0]\n"
      "ldr q3, [x2, #0x0]\n"
      "ldr q4, [x1, #0x0]\n"
      "ldr q5, [x15, #0x0]\n"
      "ldr q9, [x8, #0x10]\n"
      "ldr q12, [x8, #0x20]\n"
      "ldr q15, [x8, #0x30]\n"
      "blt 6f\n"
      "5:" // Height 6: Multiply loop: Main loop head
      "movi v7.16b, #0x0\n"
      "movi v8.16b, #0x0\n"
      "sub x6, x6, #0x4\n"
      "add x5, x5, #0x10\n"
      "movi v10.16b, #0x0\n"
      "movi v11.16b, #0x0\n"
      "add x4, x4, #0x10\n"
      "add x3, x3, #0x10\n"
      "fmlal v7.4s, v6.4h, v19.4h\n"
      "fmlal2 v8.4s, v6.4h, v19.4h\n"
      "movi v13.16b, #0x0\n"
      "add x2, x2, #0x10\n"
      "fmlal v10.4s, v9.4h, v19.4h\n"
      "fmlal2 v11.4s, v9.4h, v19.4h\n"
      "movi v14.16b, #0x0\n"
      "add x1, x1, #0x10\n"
      "fmlal v13.4s, v12.4h, v19.4h\n"
      "movi v16.16b, #0x0\n"
      "add x15, x15, #0x10\n"
      "cmp x6, #0x8\n"
      "fmlal2 v14.4s, v12.4h, v19.4h\n"
      "movi v17.16b, #0x0\n"
      "prfm pldl1keep, [x5, #0x80]\n"
      "ldr q6, [x8, #0x40]\n"
      "ldr q9, [x8, #0x50]\n"
      "fmla v20.4s, v7.4s, v0.s[0]\n"
      "fmla v22.4s, v7.4s, v1.s[0]\n"
      "ldr q12, [x8, #0x60]\n"
      "fmla v24.4s, v7.4s, v2.s[0]\n"
      "fmla v26.4s, v7.4s, v3.s[0]\n"
      "prfm pldl1keep, [x4, #0x80]\n"
      "fmla v28.4s, v7.4s, v4.s[0]\n"
      "fmla v30.4s, v7.4s, v5.s[0]\n"
      "prfm pldl1keep, [x3, #0x80]\n"
      "prfm pldl1keep, [x2, #0x80]\n"
      "fmla v21.4s, v8.4s, v0.s[0]\n"
      "fmla v23.4s, v8.4s, v1.s[0]\n"
      "prfm pldl1keep, [x1, #0x80]\n"
      "prfm pldl1keep, [x15, #0x80]\n"
      "fmla v25.4s, v8.4s, v2.s[0]\n"
      "fmla v27.4s, v8.4s, v3.s[0]\n"
      "fmla v29.4s, v8.4s, v4.s[0]\n"
      "fmla v31.4s, v8.4s, v5.s[0]\n"
      "fmla v20.4s, v10.4s, v0.s[1]\n"
      "fmla v22.4s, v10.4s, v1.s[1]\n"
      "fmla v24.4s, v10.4s, v2.s[1]\n"
      "fmla v26.4s, v10.4s, v3.s[1]\n"
      "fmla v28.4s, v10.4s, v4.s[1]\n"
      "fmla v30.4s, v10.4s, v5.s[1]\n"
      "fmla v21.4s, v11.4s, v0.s[1]\n"
      "fmla v23.4s, v11.4s, v1.s[1]\n"
      "fmla v25.4s, v11.4s, v2.s[1]\n"
      "fmla v27.4s, v11.4s, v3.s[1]\n"
      "fmla v29.4s, v11.4s, v4.s[1]\n"
      "fmla v31.4s, v11.4s, v5.s[1]\n"
      "fmla v20.4s, v13.4s, v0.s[2]\n"
      "fmla v22.4s, v13.4s, v1.s[2]\n"
      "fmla v24.4s, v13.4s, v2.s[2]\n"
      "fmla v26.4s, v13.4s, v3.s[2]\n"
      "fmla v28.4s, v13.4s, v4.s[2]\n"
      "fmla v30.4s, v13.4s, v5.s[2]\n"
      "fmla v21.4s, v14.4s, v0.s[2]\n"
      "fmla v23.4s, v14.4s, v1.s[2]\n"
      "fmla v25.4s, v14.4s, v2.s[2]\n"
      "fmla v27.4s, v14.4s, v3.s[2]\n"
      "fmla v29.4s, v14.4s, v4.s[2]\n"
      "fmla v31.4s, v14.4s, v5.s[2]\n"
      "fmlal v16.4s, v15.4h, v19.4h\n"
      "fmlal2 v17.4s, v15.4h, v19.4h\n"
      "ldr q15, [x8, #0x70]\n"
      "fmla v20.4s, v16.4s, v0.s[3]\n"
      "fmla v22.4s, v16.4s, v1.s[3]\n"
      "fmla v24.4s, v16.4s, v2.s[3]\n"
      "fmla v26.4s, v16.4s, v3.s[3]\n"
      "fmla v28.4s, v16.4s, v4.s[3]\n"
      "fmla v30.4s, v16.4s, v5.s[3]\n"
      "fmla v21.4s, v17.4s, v0.s[3]\n"
      "ldr q0, [x5, #0x0]\n"
      "fmla v23.4s, v17.4s, v1.s[3]\n"
      "ldr q1, [x4, #0x0]\n"
      "fmla v25.4s, v17.4s, v2.s[3]\n"
      "ldr q2, [x3, #0x0]\n"
      "fmla v27.4s, v17.4s, v3.s[3]\n"
      "ldr q3, [x2, #0x0]\n"
      "fmla v29.4s, v17.4s, v4.s[3]\n"
      "ldr q4, [x1, #0x0]\n"
      "fmla v31.4s, v17.4s, v5.s[3]\n"
      "ldr q5, [x15, #0x0]\n"
      "add x8, x8, #0x40\n"
      "bge 5b\n"
      "6:" // Height 6: Multiply loop: Single iteration only
      "movi v7.16b, #0x0\n"
      "movi v8.16b, #0x0\n"
      "add x5, x5, #0x10\n"
      "add x4, x4, #0x10\n"
      "movi v10.16b, #0x0\n"
      "movi v11.16b, #0x0\n"
      "add x3, x3, #0x10\n"
      "add x2, x2, #0x10\n"
      "fmlal v7.4s, v6.4h, v19.4h\n"
      "fmlal2 v8.4s, v6.4h, v19.4h\n"
      "movi v13.16b, #0x0\n"
      "add x1, x1, #0x10\n"
      "fmlal v10.4s, v9.4h, v19.4h\n"
      "fmlal2 v11.4s, v9.4h, v19.4h\n"
      "movi v14.16b, #0x0\n"
      "add x15, x15, #0x10\n"
      "fmlal v13.4s, v12.4h, v19.4h\n"
      "movi v16.16b, #0x0\n"
      "prfm pldl1keep, [x5, #0x80]\n"
      "sub x6, x6, #0x4\n"
      "fmlal2 v14.4s, v12.4h, v19.4h\n"
      "movi v17.16b, #0x0\n"
      "prfm pldl1keep, [x4, #0x80]\n"
      "prfm pldl1keep, [x3, #0x80]\n"
      "fmla v20.4s, v7.4s, v0.s[0]\n"
      "fmla v22.4s, v7.4s, v1.s[0]\n"
      "prfm pldl1keep, [x2, #0x80]\n"
      "prfm pldl1keep, [x1, #0x80]\n"
      "fmla v24.4s, v7.4s, v2.s[0]\n"
      "fmla v26.4s, v7.4s, v3.s[0]\n"
      "prfm pldl1keep, [x15, #0x80]\n"
      "add x8, x8, #0x40\n"
      "fmla v28.4s, v7.4s, v4.s[0]\n"
      "fmla v30.4s, v7.4s, v5.s[0]\n"
      "fmla v21.4s, v8.4s, v0.s[0]\n"
      "fmla v23.4s, v8.4s, v1.s[0]\n"
      "fmla v25.4s, v8.4s, v2.s[0]\n"
      "fmla v27.4s, v8.4s, v3.s[0]\n"
      "fmla v29.4s, v8.4s, v4.s[0]\n"
      "fmla v31.4s, v8.4s, v5.s[0]\n"
      "fmla v20.4s, v10.4s, v0.s[1]\n"
      "fmla v22.4s, v10.4s, v1.s[1]\n"
      "fmla v24.4s, v10.4s, v2.s[1]\n"
      "fmla v26.4s, v10.4s, v3.s[1]\n"
      "fmla v28.4s, v10.4s, v4.s[1]\n"
      "fmla v30.4s, v10.4s, v5.s[1]\n"
      "fmla v21.4s, v11.4s, v0.s[1]\n"
      "fmla v23.4s, v11.4s, v1.s[1]\n"
      "fmla v25.4s, v11.4s, v2.s[1]\n"
      "fmla v27.4s, v11.4s, v3.s[1]\n"
      "fmla v29.4s, v11.4s, v4.s[1]\n"
      "fmla v31.4s, v11.4s, v5.s[1]\n"
      "fmla v20.4s, v13.4s, v0.s[2]\n"
      "fmla v22.4s, v13.4s, v1.s[2]\n"
      "fmla v24.4s, v13.4s, v2.s[2]\n"
      "fmla v26.4s, v13.4s, v3.s[2]\n"
      "fmla v28.4s, v13.4s, v4.s[2]\n"
      "fmla v30.4s, v13.4s, v5.s[2]\n"
      "fmlal v16.4s, v15.4h, v19.4h\n"
      "fmla v21.4s, v14.4s, v0.s[2]\n"
      "fmla v23.4s, v14.4s, v1.s[2]\n"
      "fmla v25.4s, v14.4s, v2.s[2]\n"
      "fmla v27.4s, v14.4s, v3.s[2]\n"
      "fmla v29.4s, v14.4s, v4.s[2]\n"
      "fmla v31.4s, v14.4s, v5.s[2]\n"
      "fmlal2 v17.4s, v15.4h, v19.4h\n"
      "fmla v20.4s, v16.4s, v0.s[3]\n"
      "fmla v22.4s, v16.4s, v1.s[3]\n"
      "fmla v24.4s, v16.4s, v2.s[3]\n"
      "fmla v26.4s, v16.4s, v3.s[3]\n"
      "fmla v28.4s, v16.4s, v4.s[3]\n"
      "fmla v30.4s, v16.4s, v5.s[3]\n"
      "fmla v21.4s, v17.4s, v0.s[3]\n"
      "fmla v23.4s, v17.4s, v1.s[3]\n"
      "fmla v25.4s, v17.4s, v2.s[3]\n"
      "fmla v27.4s, v17.4s, v3.s[3]\n"
      "fmla v29.4s, v17.4s, v4.s[3]\n"
      "fmla v31.4s, v17.4s, v5.s[3]\n"
      "7:" // Height 6: Multiply loop: Main loop skip
      "cbz x6, 9f\n"
      "8:" // Height 6: Multiply loop: Odd block loop
      "ldr q18, [x8, #0x0]\n"
      "ldr s0, [x5], #0x4\n"
      "movi v6.16b, #0x0\n"
      "movi v7.16b, #0x0\n"
      "ldr s1, [x4], #0x4\n"
      "ldr s2, [x3], #0x4\n"
      "sub x6, x6, #0x1\n"
      "add x8, x8, #0x10\n"
      "ldr s3, [x2], #0x4\n"
      "ldr s4, [x1], #0x4\n"
      "ldr s5, [x15], #0x4\n"
      "fmlal v6.4s, v18.4h, v19.4h\n"
      "fmlal2 v7.4s, v18.4h, v19.4h\n"
      "fmla v20.4s, v6.4s, v0.s[0]\n"
      "fmla v22.4s, v6.4s, v1.s[0]\n"
      "fmla v24.4s, v6.4s, v2.s[0]\n"
      "fmla v26.4s, v6.4s, v3.s[0]\n"
      "fmla v28.4s, v6.4s, v4.s[0]\n"
      "fmla v30.4s, v6.4s, v5.s[0]\n"
      "fmla v21.4s, v7.4s, v0.s[0]\n"
      "fmla v23.4s, v7.4s, v1.s[0]\n"
      "fmla v25.4s, v7.4s, v2.s[0]\n"
      "fmla v27.4s, v7.4s, v3.s[0]\n"
      "fmla v29.4s, v7.4s, v4.s[0]\n"
      "fmla v31.4s, v7.4s, v5.s[0]\n"
      "cbnz x6, 8b\n"
      "9:" // Height 6: Multiply loop: No odd multiplies
      "prfm pstl1keep, [x7, #0x0]\n"
      "str q20, [x7, #0x0]\n"
      "str q21, [x7, #0x10]\n"
      "add x15, x7, x14\n"
      "add x7, x7, #0x20\n"
      "prfm pstl1keep, [x15, #0x0]\n"
      "str q22, [x15, #0x0]\n"
      "add x3, x15, x14\n"
      "add x2, x3, x14\n"
      "add x1, x2, x14\n"
      "prfm pstl1keep, [x3, #0x0]\n"
      "prfm pstl1keep, [x2, #0x0]\n"
      "str q23, [x15, #0x10]\n"
      "add x15, x1, x14\n"
      "prfm pstl1keep, [x1, #0x0]\n"
      "str q24, [x3, #0x0]\n"
      "prfm pstl1keep, [x15, #0x0]\n"
      "str q25, [x3, #0x10]\n"
      "str q26, [x2, #0x0]\n"
      "str q27, [x2, #0x10]\n"
      "str q28, [x1, #0x0]\n"
      "str q29, [x1, #0x10]\n"
      "str q30, [x15, #0x0]\n"
      "str q31, [x15, #0x10]\n"
      "subs x9, x9, #0x1\n"
      "bgt 1b\n"
      :
      : [gp] "r"(gp),
        [offsetof_A] "I"(offsetof(GemmParamsFP16, A)),
        [offsetof_B] "I"(offsetof(GemmParamsFP16, B)),
        [offsetof_C] "I"(offsetof(GemmParamsFP16, C)),
        [offsetof_b_block_cols] "I"(offsetof(GemmParamsFP16, b_block_cols)),
        [offsetof_beta] "I"(offsetof(GemmParamsFP16, beta)),
        [offsetof_k] "I"(offsetof(GemmParamsFP16, k)),
        [offsetof_lda] "I"(offsetof(GemmParamsFP16, lda)),
        [offsetof_ldc] "I"(offsetof(GemmParamsFP16, ldc))
      : "cc",
        "memory",
        "v0",
        "v1",
        "v10",
        "v11",
        "v12",
        "v13",
        "v14",
        "v15",
        "v16",
        "v17",
        "v18",
        "v19",
        "v2",
        "v20",
        "v21",
        "v22",
        "v23",
        "v24",
        "v25",
        "v26",
        "v27",
        "v28",
        "v29",
        "v3",
        "v30",
        "v31",
        "v4",
        "v5",
        "v6",
        "v7",
        "v8",
        "v9",
        "x10",
        "x11",
        "x12",
        "x13",
        "x14",
        "x15",
        "x1",
        "x2",
        "x3",
        "x4",
        "x5",
        "x6",
        "x7",
        "x8",
        "x9");
#endif // __aarch64__
}

void NOINLINE gemmkernel_7x1_Neon_fp16_fA0fB0fC0(GemmParamsFP16* gp) {
#if defined(__aarch64__) && __ARM_FEATURE_FP16_FML
  __asm__ __volatile__(
      "ldr s16, [%x[gp], %[offsetof_beta]]\n"
      "mov x11, #0x1\n"
      "fmov v17.8h, #1.0\n"
      "ldr x14, [%x[gp], %[offsetof_k]]\n"
      "ldr x12, [%x[gp], %[offsetof_A]]\n"
      "ldr x9, [%x[gp], %[offsetof_B]]\n"
      "ldr x8, [%x[gp], %[offsetof_C]]\n"
      "ldr x13, [%x[gp], %[offsetof_lda]]\n"
      "fcmp s16, #0.0\n"
      "csel x11, XZR, x11, EQ\n"
      "csel x11, XZR, x11, VS\n"
      "ldr x15, [%x[gp], %[offsetof_ldc]]\n"
      "ldr x10, [%x[gp], %[offsetof_b_block_cols]]\n"
      "1:" // Height 7: Column loop
      "tbz x11, #0, 2f\n"
      "ldr q18, [x8, #0x0]\n"
      "ldr q19, [x8, #0x10]\n"
      "add x17, %x[gp], %[offsetof_beta]\n"
      "ld1r { v16.4s }, [x17]\n"
      "add x17, x8, x15\n"
      "ldr q20, [x17, #0x0]\n"
      "ldr q21, [x17, #0x10]\n"
      "add x17, x17, x15\n"
      "ldr q22, [x17, #0x0]\n"
      "ldr q23, [x17, #0x10]\n"
      "add x17, x17, x15\n"
      "ldr q24, [x17, #0x0]\n"
      "ldr q25, [x17, #0x10]\n"
      "add x17, x17, x15\n"
      "fmul v18.4s, v18.4s, v16.4s\n"
      "ldr q26, [x17, #0x0]\n"
      "ldr q27, [x17, #0x10]\n"
      "add x17, x17, x15\n"
      "fmul v19.4s, v19.4s, v16.4s\n"
      "ldr q28, [x17, #0x0]\n"
      "ldr q29, [x17, #0x10]\n"
      "add x17, x17, x15\n"
      "fmul v20.4s, v20.4s, v16.4s\n"
      "ldr q30, [x17, #0x0]\n"
      "ldr q31, [x17, #0x10]\n"
      "fmul v21.4s, v21.4s, v16.4s\n"
      "fmul v22.4s, v22.4s, v16.4s\n"
      "fmul v23.4s, v23.4s, v16.4s\n"
      "fmul v24.4s, v24.4s, v16.4s\n"
      "fmul v25.4s, v25.4s, v16.4s\n"
      "fmul v26.4s, v26.4s, v16.4s\n"
      "fmul v27.4s, v27.4s, v16.4s\n"
      "fmul v28.4s, v28.4s, v16.4s\n"
      "fmul v29.4s, v29.4s, v16.4s\n"
      "fmul v30.4s, v30.4s, v16.4s\n"
      "fmul v31.4s, v31.4s, v16.4s\n"
      "b 3f\n"
      "2:" // Height 7: no accumulate
      "movi v18.16b, #0x0\n"
      "movi v19.16b, #0x0\n"
      "movi v20.16b, #0x0\n"
      "movi v21.16b, #0x0\n"
      "movi v22.16b, #0x0\n"
      "movi v23.16b, #0x0\n"
      "movi v24.16b, #0x0\n"
      "movi v25.16b, #0x0\n"
      "movi v26.16b, #0x0\n"
      "movi v27.16b, #0x0\n"
      "movi v28.16b, #0x0\n"
      "movi v29.16b, #0x0\n"
      "movi v30.16b, #0x0\n"
      "movi v31.16b, #0x0\n"
      "3:" // Height 7: setup done
      "add x5, x12, x13\n"
      "cmp x14, #0x4\n"
      "mov x6, x12\n"
      "mov x7, x14\n"
      "add x4, x5, x13\n"
      "add x3, x4, x13\n"
      "add x2, x3, x13\n"
      "add x1, x2, x13\n"
      "add x17, x1, x13\n"
      "blt 7f\n"
      "ldr q0, [x12, #0x0]\n"
      "ldr q7, [x9, #0x0]\n"
      "cmp x14, #0x8\n"
      "ldr q1, [x5, #0x0]\n"
      "ldr q2, [x4, #0x0]\n"
      "ldr q3, [x3, #0x0]\n"
      "ldr q4, [x2, #0x0]\n"
      "ldr q5, [x1, #0x0]\n"
      "ldr q6, [x17, #0x0]\n"
      "ldr q10, [x9, #0x10]\n"
      "ldr q13, [x9, #0x20]\n"
      "ldr q16, [x9, #0x30]\n"
      "blt 6f\n"
      "5:" // Height 7: Multiply loop: Main loop head
      "movi v8.16b, #0x0\n"
      "movi v9.16b, #0x0\n"
      "sub x7, x7, #0x4\n"
      "add x6, x6, #0x10\n"
      "movi v11.16b, #0x0\n"
      "movi v12.16b, #0x0\n"
      "add x5, x5, #0x10\n"
      "add x4, x4, #0x10\n"
      "fmlal v8.4s, v7.4h, v17.4h\n"
      "fmlal2 v9.4s, v7.4h, v17.4h\n"
      "movi v14.16b, #0x0\n"
      "add x3, x3, #0x10\n"
      "fmlal v11.4s, v10.4h, v17.4h\n"
      "fmlal2 v12.4s, v10.4h, v17.4h\n"
      "movi v15.16b, #0x0\n"
      "add x2, x2, #0x10\n"
      "fmlal v14.4s, v13.4h, v17.4h\n"
      "movi v7.16b, #0x0\n"
      "add x1, x1, #0x10\n"
      "add x17, x17, #0x10\n"
      "fmlal2 v15.4s, v13.4h, v17.4h\n"
      "cmp x7, #0x8\n"
      "prfm pldl1keep, [x6, #0x80]\n"
      "ldr q10, [x9, #0x50]\n"
      "ldr q13, [x9, #0x60]\n"
      "fmla v18.4s, v8.4s, v0.s[0]\n"
      "fmla v20.4s, v8.4s, v1.s[0]\n"
      "fmla v22.4s, v8.4s, v2.s[0]\n"
      "fmla v24.4s, v8.4s, v3.s[0]\n"
      "prfm pldl1keep, [x5, #0x80]\n"
      "prfm pldl1keep, [x4, #0x80]\n"
      "fmla v26.4s, v8.4s, v4.s[0]\n"
      "fmla v28.4s, v8.4s, v5.s[0]\n"
      "prfm pldl1keep, [x3, #0x80]\n"
      "prfm pldl1keep, [x2, #0x80]\n"
      "fmla v30.4s, v8.4s, v6.s[0]\n"
      "fmla v19.4s, v9.4s, v0.s[0]\n"
      "movi v8.16b, #0x0\n"
      "prfm pldl1keep, [x1, #0x80]\n"
      "fmla v21.4s, v9.4s, v1.s[0]\n"
      "fmla v23.4s, v9.4s, v2.s[0]\n"
      "prfm pldl1keep, [x17, #0x80]\n"
      "fmla v25.4s, v9.4s, v3.s[0]\n"
      "fmla v27.4s, v9.4s, v4.s[0]\n"
      "fmla v29.4s, v9.4s, v5.s[0]\n"
      "fmla v31.4s, v9.4s, v6.s[0]\n"
      "fmla v18.4s, v11.4s, v0.s[1]\n"
      "fmla v20.4s, v11.4s, v1.s[1]\n"
      "fmla v22.4s, v11.4s, v2.s[1]\n"
      "fmla v24.4s, v11.4s, v3.s[1]\n"
      "fmla v26.4s, v11.4s, v4.s[1]\n"
      "fmla v28.4s, v11.4s, v5.s[1]\n"
      "fmla v30.4s, v11.4s, v6.s[1]\n"
      "fmla v19.4s, v12.4s, v0.s[1]\n"
      "fmla v21.4s, v12.4s, v1.s[1]\n"
      "fmla v23.4s, v12.4s, v2.s[1]\n"
      "fmla v25.4s, v12.4s, v3.s[1]\n"
      "fmla v27.4s, v12.4s, v4.s[1]\n"
      "fmla v29.4s, v12.4s, v5.s[1]\n"
      "fmla v31.4s, v12.4s, v6.s[1]\n"
      "fmlal v7.4s, v16.4h, v17.4h\n"
      "fmlal2 v8.4s, v16.4h, v17.4h\n"
      "ldr q16, [x9, #0x70]\n"
      "fmla v18.4s, v14.4s, v0.s[2]\n"
      "fmla v20.4s, v14.4s, v1.s[2]\n"
      "fmla v22.4s, v14.4s, v2.s[2]\n"
      "fmla v24.4s, v14.4s, v3.s[2]\n"
      "fmla v26.4s, v14.4s, v4.s[2]\n"
      "fmla v28.4s, v14.4s, v5.s[2]\n"
      "fmla v30.4s, v14.4s, v6.s[2]\n"
      "fmla v19.4s, v15.4s, v0.s[2]\n"
      "fmla v21.4s, v15.4s, v1.s[2]\n"
      "fmla v23.4s, v15.4s, v2.s[2]\n"
      "fmla v25.4s, v15.4s, v3.s[2]\n"
      "fmla v27.4s, v15.4s, v4.s[2]\n"
      "fmla v29.4s, v15.4s, v5.s[2]\n"
      "fmla v31.4s, v15.4s, v6.s[2]\n"
      "fmla v18.4s, v7.4s, v0.s[3]\n"
      "fmla v20.4s, v7.4s, v1.s[3]\n"
      "fmla v22.4s, v7.4s, v2.s[3]\n"
      "fmla v24.4s, v7.4s, v3.s[3]\n"
      "fmla v26.4s, v7.4s, v4.s[3]\n"
      "fmla v28.4s, v7.4s, v5.s[3]\n"
      "fmla v30.4s, v7.4s, v6.s[3]\n"
      "ldr q7, [x9, #0x40]\n"
      "fmla v19.4s, v8.4s, v0.s[3]\n"
      "ldr q0, [x6, #0x0]\n"
      "fmla v21.4s, v8.4s, v1.s[3]\n"
      "ldr q1, [x5, #0x0]\n"
      "fmla v23.4s, v8.4s, v2.s[3]\n"
      "ldr q2, [x4, #0x0]\n"
      "fmla v25.4s, v8.4s, v3.s[3]\n"
      "ldr q3, [x3, #0x0]\n"
      "fmla v27.4s, v8.4s, v4.s[3]\n"
      "ldr q4, [x2, #0x0]\n"
      "fmla v29.4s, v8.4s, v5.s[3]\n"
      "ldr q5, [x1, #0x0]\n"
      "fmla v31.4s, v8.4s, v6.s[3]\n"
      "ldr q6, [x17, #0x0]\n"
      "add x9, x9, #0x40\n"
      "bge 5b\n"
      "6:" // Height 7: Multiply loop: Single iteration only
      "movi v8.16b, #0x0\n"
      "movi v9.16b, #0x0\n"
      "add x6, x6, #0x10\n"
      "add x5, x5, #0x10\n"
      "movi v11.16b, #0x0\n"
      "movi v12.16b, #0x0\n"
      "add x4, x4, #0x10\n"
      "add x3, x3, #0x10\n"
      "fmlal v8.4s, v7.4h, v17.4h\n"
      "fmlal2 v9.4s, v7.4h, v17.4h\n"
      "movi v14.16b, #0x0\n"
      "add x2, x2, #0x10\n"
      "fmlal v11.4s, v10.4h, v17.4h\n"
      "fmlal2 v12.4s, v10.4h, v17.4h\n"
      "movi v15.16b, #0x0\n"
      "add x1, x1, #0x10\n"
      "fmlal v14.4s, v13.4h, v17.4h\n"
      "movi v7.16b, #0x0\n"
      "add x17, x17, #0x10\n"
      "prfm pldl1keep, [x6, #0x80]\n"
      "fmlal2 v15.4s, v13.4h, v17.4h\n"
      "prfm pldl1keep, [x5, #0x80]\n"
      "sub x7, x7, #0x4\n"
      "prfm pldl1keep, [x4, #0x80]\n"
      "fmla v18.4s, v8.4s, v0.s[0]\n"
      "fmla v20.4s, v8.4s, v1.s[0]\n"
      "prfm pldl1keep, [x3, #0x80]\n"
      "prfm pldl1keep, [x2, #0x80]\n"
      "fmla v22.4s, v8.4s, v2.s[0]\n"
      "fmla v24.4s, v8.4s, v3.s[0]\n"
      "prfm pldl1keep, [x1, #0x80]\n"
      "prfm pldl1keep, [x17, #0x80]\n"
      "fmla v26.4s, v8.4s, v4.s[0]\n"
      "fmla v28.4s, v8.4s, v5.s[0]\n"
      "add x9, x9, #0x40\n"
      "fmla v30.4s, v8.4s, v6.s[0]\n"
      "fmla v19.4s, v9.4s, v0.s[0]\n"
      "movi v8.16b, #0x0\n"
      "fmla v21.4s, v9.4s, v1.s[0]\n"
      "fmla v23.4s, v9.4s, v2.s[0]\n"
      "fmla v25.4s, v9.4s, v3.s[0]\n"
      "fmla v27.4s, v9.4s, v4.s[0]\n"
      "fmla v29.4s, v9.4s, v5.s[0]\n"
      "fmla v31.4s, v9.4s, v6.s[0]\n"
      "fmla v18.4s, v11.4s, v0.s[1]\n"
      "fmla v20.4s, v11.4s, v1.s[1]\n"
      "fmla v22.4s, v11.4s, v2.s[1]\n"
      "fmla v24.4s, v11.4s, v3.s[1]\n"
      "fmla v26.4s, v11.4s, v4.s[1]\n"
      "fmla v28.4s, v11.4s, v5.s[1]\n"
      "fmla v30.4s, v11.4s, v6.s[1]\n"
      "fmla v19.4s, v12.4s, v0.s[1]\n"
      "fmla v21.4s, v12.4s, v1.s[1]\n"
      "fmla v23.4s, v12.4s, v2.s[1]\n"
      "fmla v25.4s, v12.4s, v3.s[1]\n"
      "fmla v27.4s, v12.4s, v4.s[1]\n"
      "fmla v29.4s, v12.4s, v5.s[1]\n"
      "fmla v31.4s, v12.4s, v6.s[1]\n"
      "fmlal v7.4s, v16.4h, v17.4h\n"
      "fmlal2 v8.4s, v16.4h, v17.4h\n"
      "fmla v18.4s, v14.4s, v0.s[2]\n"
      "fmla v20.4s, v14.4s, v1.s[2]\n"
      "fmla v22.4s, v14.4s, v2.s[2]\n"
      "fmla v24.4s, v14.4s, v3.s[2]\n"
      "fmla v26.4s, v14.4s, v4.s[2]\n"
      "fmla v28.4s, v14.4s, v5.s[2]\n"
      "fmla v30.4s, v14.4s, v6.s[2]\n"
      "fmla v19.4s, v15.4s, v0.s[2]\n"
      "fmla v21.4s, v15.4s, v1.s[2]\n"
      "fmla v23.4s, v15.4s, v2.s[2]\n"
      "fmla v25.4s, v15.4s, v3.s[2]\n"
      "fmla v27.4s, v15.4s, v4.s[2]\n"
      "fmla v29.4s, v15.4s, v5.s[2]\n"
      "fmla v31.4s, v15.4s, v6.s[2]\n"
      "fmla v18.4s, v7.4s, v0.s[3]\n"
      "fmla v20.4s, v7.4s, v1.s[3]\n"
      "fmla v22.4s, v7.4s, v2.s[3]\n"
      "fmla v24.4s, v7.4s, v3.s[3]\n"
      "fmla v26.4s, v7.4s, v4.s[3]\n"
      "fmla v28.4s, v7.4s, v5.s[3]\n"
      "fmla v30.4s, v7.4s, v6.s[3]\n"
      "fmla v19.4s, v8.4s, v0.s[3]\n"
      "fmla v21.4s, v8.4s, v1.s[3]\n"
      "fmla v23.4s, v8.4s, v2.s[3]\n"
      "fmla v25.4s, v8.4s, v3.s[3]\n"
      "fmla v27.4s, v8.4s, v4.s[3]\n"
      "fmla v29.4s, v8.4s, v5.s[3]\n"
      "fmla v31.4s, v8.4s, v6.s[3]\n"
      "7:" // Height 7: Multiply loop: Main loop skip
      "cbz x7, 9f\n"
      "8:" // Height 7: Multiply loop: Odd block loop
      "ldr q9, [x9, #0x0]\n"
      "ldr s0, [x6], #0x4\n"
      "movi v10.16b, #0x0\n"
      "movi v11.16b, #0x0\n"
      "ldr s1, [x5], #0x4\n"
      "ldr s2, [x4], #0x4\n"
      "sub x7, x7, #0x1\n"
      "add x9, x9, #0x10\n"
      "ldr s3, [x3], #0x4\n"
      "ldr s4, [x2], #0x4\n"
      "ldr s5, [x1], #0x4\n"
      "ldr s6, [x17], #0x4\n"
      "fmlal v10.4s, v9.4h, v17.4h\n"
      "fmlal2 v11.4s, v9.4h, v17.4h\n"
      "fmla v18.4s, v10.4s, v0.s[0]\n"
      "fmla v20.4s, v10.4s, v1.s[0]\n"
      "fmla v22.4s, v10.4s, v2.s[0]\n"
      "fmla v24.4s, v10.4s, v3.s[0]\n"
      "fmla v26.4s, v10.4s, v4.s[0]\n"
      "fmla v28.4s, v10.4s, v5.s[0]\n"
      "fmla v30.4s, v10.4s, v6.s[0]\n"
      "fmla v19.4s, v11.4s, v0.s[0]\n"
      "fmla v21.4s, v11.4s, v1.s[0]\n"
      "fmla v23.4s, v11.4s, v2.s[0]\n"
      "fmla v25.4s, v11.4s, v3.s[0]\n"
      "fmla v27.4s, v11.4s, v4.s[0]\n"
      "fmla v29.4s, v11.4s, v5.s[0]\n"
      "fmla v31.4s, v11.4s, v6.s[0]\n"
      "cbnz x7, 8b\n"
      "9:" // Height 7: Multiply loop: No odd multiplies
      "prfm pstl1keep, [x8, #0x0]\n"
      "str q18, [x8, #0x0]\n"
      "str q19, [x8, #0x10]\n"
      "add x17, x8, x15\n"
      "add x8, x8, #0x20\n"
      "prfm pstl1keep, [x17, #0x0]\n"
      "str q20, [x17, #0x0]\n"
      "add x4, x17, x15\n"
      "add x3, x4, x15\n"
      "add x2, x3, x15\n"
      "prfm pstl1keep, [x4, #0x0]\n"
      "prfm pstl1keep, [x3, #0x0]\n"
      "str q21, [x17, #0x10]\n"
      "add x1, x2, x15\n"
      "prfm pstl1keep, [x2, #0x0]\n"
      "str q22, [x4, #0x0]\n"
      "add x17, x1, x15\n"
      "prfm pstl1keep, [x1, #0x0]\n"
      "prfm pstl1keep, [x17, #0x0]\n"
      "str q23, [x4, #0x10]\n"
      "str q24, [x3, #0x0]\n"
      "str q25, [x3, #0x10]\n"
      "str q26, [x2, #0x0]\n"
      "str q27, [x2, #0x10]\n"
      "str q28, [x1, #0x0]\n"
      "str q29, [x1, #0x10]\n"
      "str q30, [x17, #0x0]\n"
      "str q31, [x17, #0x10]\n"
      "subs x10, x10, #0x1\n"
      "bgt 1b\n"
      :
      : [gp] "r"(gp),
        [offsetof_A] "I"(offsetof(GemmParamsFP16, A)),
        [offsetof_B] "I"(offsetof(GemmParamsFP16, B)),
        [offsetof_C] "I"(offsetof(GemmParamsFP16, C)),
        [offsetof_b_block_cols] "I"(offsetof(GemmParamsFP16, b_block_cols)),
        [offsetof_beta] "I"(offsetof(GemmParamsFP16, beta)),
        [offsetof_k] "I"(offsetof(GemmParamsFP16, k)),
        [offsetof_lda] "I"(offsetof(GemmParamsFP16, lda)),
        [offsetof_ldc] "I"(offsetof(GemmParamsFP16, ldc))
      : "cc",
        "memory",
        "v0",
        "v1",
        "v10",
        "v11",
        "v12",
        "v13",
        "v14",
        "v15",
        "v16",
        "v17",
        "v18",
        "v19",
        "v2",
        "v20",
        "v21",
        "v22",
        "v23",
        "v24",
        "v25",
        "v26",
        "v27",
        "v28",
        "v29",
        "v3",
        "v30",
        "v31",
        "v4",
        "v5",
        "v6",
        "v7",
        "v8",
        "v9",
        "x10",
        "x11",
        "x12",
        "x13",
        "x14",
        "x15",
        "x17",
        "x1",
        "x2",
        "x3",
        "x4",
        "x5",
        "x6",
        "x7",
        "x8",
        "x9");
#endif // __aarch64__
}

void NOINLINE gemmkernel_8x1_Neon_fp16_fA0fB0fC0(GemmParamsFP16* gp) {
#if defined(__aarch64__) && __ARM_FEATURE_FP16_FML
  __asm__ __volatile__(
      "ldr s16, [%x[gp], %[offsetof_beta]]\n"
      "mov x12, #0x1\n"
      "fmov v15.8h, #1.0\n"
      "ldr x15, [%x[gp], %[offsetof_k]]\n"
      "ldr x13, [%x[gp], %[offsetof_A]]\n"
      "ldr x10, [%x[gp], %[offsetof_B]]\n"
      "ldr x9, [%x[gp], %[offsetof_C]]\n"
      "ldr x14, [%x[gp], %[offsetof_lda]]\n"
      "fcmp s16, #0.0\n"
      "csel x12, XZR, x12, EQ\n"
      "csel x12, XZR, x12, VS\n"
      "ldr x16, [%x[gp], %[offsetof_ldc]]\n"
      "ldr x11, [%x[gp], %[offsetof_b_block_cols]]\n"
      "1:" // Height 8: Column loop
      "tbz x12, #0, 2f\n"
      "ldr q16, [x9, #0x0]\n"
      "ldr q17, [x9, #0x10]\n"
      "add x17, %x[gp], %[offsetof_beta]\n"
      "ld1r { v0.4s }, [x17]\n"
      "add x17, x9, x16\n"
      "ldr q18, [x17, #0x0]\n"
      "ldr q19, [x17, #0x10]\n"
      "add x17, x17, x16\n"
      "ldr q20, [x17, #0x0]\n"
      "ldr q21, [x17, #0x10]\n"
      "add x17, x17, x16\n"
      "ldr q22, [x17, #0x0]\n"
      "ldr q23, [x17, #0x10]\n"
      "add x17, x17, x16\n"
      "fmul v16.4s, v16.4s, v0.4s\n"
      "ldr q24, [x17, #0x0]\n"
      "ldr q25, [x17, #0x10]\n"
      "add x17, x17, x16\n"
      "fmul v17.4s, v17.4s, v0.4s\n"
      "ldr q26, [x17, #0x0]\n"
      "ldr q27, [x17, #0x10]\n"
      "add x17, x17, x16\n"
      "fmul v18.4s, v18.4s, v0.4s\n"
      "ldr q28, [x17, #0x0]\n"
      "ldr q29, [x17, #0x10]\n"
      "add x17, x17, x16\n"
      "fmul v19.4s, v19.4s, v0.4s\n"
      "ldr q30, [x17, #0x0]\n"
      "ldr q31, [x17, #0x10]\n"
      "fmul v20.4s, v20.4s, v0.4s\n"
      "fmul v21.4s, v21.4s, v0.4s\n"
      "fmul v22.4s, v22.4s, v0.4s\n"
      "fmul v23.4s, v23.4s, v0.4s\n"
      "fmul v24.4s, v24.4s, v0.4s\n"
      "fmul v25.4s, v25.4s, v0.4s\n"
      "fmul v26.4s, v26.4s, v0.4s\n"
      "fmul v27.4s, v27.4s, v0.4s\n"
      "fmul v28.4s, v28.4s, v0.4s\n"
      "fmul v29.4s, v29.4s, v0.4s\n"
      "fmul v30.4s, v30.4s, v0.4s\n"
      "fmul v31.4s, v31.4s, v0.4s\n"
      "b 3f\n"
      "2:" // Height 8: no accumulate
      "movi v16.16b, #0x0\n"
      "movi v17.16b, #0x0\n"
      "movi v18.16b, #0x0\n"
      "movi v19.16b, #0x0\n"
      "movi v20.16b, #0x0\n"
      "movi v21.16b, #0x0\n"
      "movi v22.16b, #0x0\n"
      "movi v23.16b, #0x0\n"
      "movi v24.16b, #0x0\n"
      "movi v25.16b, #0x0\n"
      "movi v26.16b, #0x0\n"
      "movi v27.16b, #0x0\n"
      "movi v28.16b, #0x0\n"
      "movi v29.16b, #0x0\n"
      "movi v30.16b, #0x0\n"
      "movi v31.16b, #0x0\n"
      "3:" // Height 8: setup done
      "add x6, x13, x14\n"
      "cmp x15, #0x4\n"
      "mov x7, x13\n"
      "mov x8, x15\n"
      "add x5, x6, x14\n"
      "add x4, x5, x14\n"
      "add x3, x4, x14\n"
      "add x2, x3, x14\n"
      "add x1, x2, x14\n"
      "add x17, x1, x14\n"
      "blt 7f\n"
      "ldr q0, [x13, #0x0]\n"
      "ldr q8, [x10, #0x0]\n"
      "cmp x15, #0x8\n"
      "ldr q1, [x6, #0x0]\n"
      "ldr q2, [x5, #0x0]\n"
      "ldr q3, [x4, #0x0]\n"
      "ldr q4, [x3, #0x0]\n"
      "ldr q5, [x2, #0x0]\n"
      "ldr q6, [x1, #0x0]\n"
      "ldr q7, [x17, #0x0]\n"
      "ldr q11, [x10, #0x10]\n"
      "ldr q14, [x10, #0x20]\n"
      "blt 6f\n"
      "5:" // Height 8: Multiply loop: Main loop head
      "movi v9.16b, #0x0\n"
      "movi v10.16b, #0x0\n"
      "sub x8, x8, #0x4\n"
      "add x7, x7, #0x10\n"
      "movi v12.16b, #0x0\n"
      "movi v13.16b, #0x0\n"
      "add x6, x6, #0x10\n"
      "add x5, x5, #0x10\n"
      "fmlal v9.4s, v8.4h, v15.4h\n"
      "fmlal2 v10.4s, v8.4h, v15.4h\n"
      "movi v8.16b, #0x0\n"
      "add x4, x4, #0x10\n"
      "fmlal v12.4s, v11.4h, v15.4h\n"
      "fmlal2 v13.4s, v11.4h, v15.4h\n"
      "movi v11.16b, #0x0\n"
      "add x3, x3, #0x10\n"
      "fmlal v8.4s, v14.4h, v15.4h\n"
      "add x2, x2, #0x10\n"
      "add x1, x1, #0x10\n"
      "prfm pldl1keep, [x7, #0x80]\n"
      "add x17, x17, #0x10\n"
      "cmp x8, #0x8\n"
      "prfm pldl1keep, [x6, #0x80]\n"
      "prfm pldl1keep, [x5, #0x80]\n"
      "fmla v16.4s, v9.4s, v0.s[0]\n"
      "fmla v18.4s, v9.4s, v1.s[0]\n"
      "prfm pldl1keep, [x4, #0x80]\n"
      "prfm pldl1keep, [x3, #0x80]\n"
      "fmla v20.4s, v9.4s, v2.s[0]\n"
      "fmla v22.4s, v9.4s, v3.s[0]\n"
      "prfm pldl1keep, [x2, #0x80]\n"
      "prfm pldl1keep, [x1, #0x80]\n"
      "fmla v24.4s, v9.4s, v4.s[0]\n"
      "fmla v26.4s, v9.4s, v5.s[0]\n"
      "prfm pldl1keep, [x17, #0x80]\n"
      "fmla v28.4s, v9.4s, v6.s[0]\n"
      "fmla v30.4s, v9.4s, v7.s[0]\n"
      "movi v9.16b, #0x0\n"
      "fmla v17.4s, v10.4s, v0.s[0]\n"
      "fmla v19.4s, v10.4s, v1.s[0]\n"
      "fmla v21.4s, v10.4s, v2.s[0]\n"
      "fmla v23.4s, v10.4s, v3.s[0]\n"
      "fmla v25.4s, v10.4s, v4.s[0]\n"
      "fmla v27.4s, v10.4s, v5.s[0]\n"
      "fmla v29.4s, v10.4s, v6.s[0]\n"
      "fmla v31.4s, v10.4s, v7.s[0]\n"
      "ldr q10, [x10, #0x30]\n"
      "fmlal2 v9.4s, v14.4h, v15.4h\n"
      "ldr q14, [x10, #0x60]\n"
      "fmla v16.4s, v12.4s, v0.s[1]\n"
      "fmla v18.4s, v12.4s, v1.s[1]\n"
      "fmla v20.4s, v12.4s, v2.s[1]\n"
      "fmla v22.4s, v12.4s, v3.s[1]\n"
      "fmla v24.4s, v12.4s, v4.s[1]\n"
      "fmla v26.4s, v12.4s, v5.s[1]\n"
      "fmla v28.4s, v12.4s, v6.s[1]\n"
      "fmla v30.4s, v12.4s, v7.s[1]\n"
      "fmla v17.4s, v13.4s, v0.s[1]\n"
      "movi v12.16b, #0x0\n"
      "fmla v19.4s, v13.4s, v1.s[1]\n"
      "fmla v21.4s, v13.4s, v2.s[1]\n"
      "fmla v23.4s, v13.4s, v3.s[1]\n"
      "fmla v25.4s, v13.4s, v4.s[1]\n"
      "fmla v27.4s, v13.4s, v5.s[1]\n"
      "fmla v29.4s, v13.4s, v6.s[1]\n"
      "fmla v31.4s, v13.4s, v7.s[1]\n"
      "fmlal v11.4s, v10.4h, v15.4h\n"
      "fmla v16.4s, v8.4s, v0.s[2]\n"
      "fmla v18.4s, v8.4s, v1.s[2]\n"
      "fmla v20.4s, v8.4s, v2.s[2]\n"
      "fmla v22.4s, v8.4s, v3.s[2]\n"
      "fmla v24.4s, v8.4s, v4.s[2]\n"
      "fmla v26.4s, v8.4s, v5.s[2]\n"
      "fmla v28.4s, v8.4s, v6.s[2]\n"
      "fmla v30.4s, v8.4s, v7.s[2]\n"
      "ldr q8, [x10, #0x40]\n"
      "fmla v17.4s, v9.4s, v0.s[2]\n"
      "fmla v19.4s, v9.4s, v1.s[2]\n"
      "fmla v21.4s, v9.4s, v2.s[2]\n"
      "fmla v23.4s, v9.4s, v3.s[2]\n"
      "fmla v25.4s, v9.4s, v4.s[2]\n"
      "fmla v27.4s, v9.4s, v5.s[2]\n"
      "fmla v29.4s, v9.4s, v6.s[2]\n"
      "fmla v31.4s, v9.4s, v7.s[2]\n"
      "fmlal2 v12.4s, v10.4h, v15.4h\n"
      "fmla v16.4s, v11.4s, v0.s[3]\n"
      "fmla v18.4s, v11.4s, v1.s[3]\n"
      "fmla v20.4s, v11.4s, v2.s[3]\n"
      "fmla v22.4s, v11.4s, v3.s[3]\n"
      "fmla v24.4s, v11.4s, v4.s[3]\n"
      "fmla v26.4s, v11.4s, v5.s[3]\n"
      "fmla v28.4s, v11.4s, v6.s[3]\n"
      "fmla v30.4s, v11.4s, v7.s[3]\n"
      "ldr q11, [x10, #0x50]\n"
      "fmla v17.4s, v12.4s, v0.s[3]\n"
      "ldr q0, [x7, #0x0]\n"
      "fmla v19.4s, v12.4s, v1.s[3]\n"
      "ldr q1, [x6, #0x0]\n"
      "fmla v21.4s, v12.4s, v2.s[3]\n"
      "ldr q2, [x5, #0x0]\n"
      "fmla v23.4s, v12.4s, v3.s[3]\n"
      "ldr q3, [x4, #0x0]\n"
      "fmla v25.4s, v12.4s, v4.s[3]\n"
      "ldr q4, [x3, #0x0]\n"
      "fmla v27.4s, v12.4s, v5.s[3]\n"
      "ldr q5, [x2, #0x0]\n"
      "fmla v29.4s, v12.4s, v6.s[3]\n"
      "ldr q6, [x1, #0x0]\n"
      "fmla v31.4s, v12.4s, v7.s[3]\n"
      "ldr q7, [x17, #0x0]\n"
      "add x10, x10, #0x40\n"
      "bge 5b\n"
      "6:" // Height 8: Multiply loop: Single iteration only
      "movi v9.16b, #0x0\n"
      "movi v10.16b, #0x0\n"
      "add x7, x7, #0x10\n"
      "add x6, x6, #0x10\n"
      "movi v12.16b, #0x0\n"
      "movi v13.16b, #0x0\n"
      "add x5, x5, #0x10\n"
      "add x4, x4, #0x10\n"
      "fmlal v9.4s, v8.4h, v15.4h\n"
      "fmlal2 v10.4s, v8.4h, v15.4h\n"
      "movi v8.16b, #0x0\n"
      "add x3, x3, #0x10\n"
      "fmlal v12.4s, v11.4h, v15.4h\n"
      "fmlal2 v13.4s, v11.4h, v15.4h\n"
      "movi v11.16b, #0x0\n"
      "add x2, x2, #0x10\n"
      "fmlal v8.4s, v14.4h, v15.4h\n"
      "add x1, x1, #0x10\n"
      "add x17, x17, #0x10\n"
      "prfm pldl1keep, [x7, #0x80]\n"
      "prfm pldl1keep, [x6, #0x80]\n"
      "prfm pldl1keep, [x5, #0x80]\n"
      "sub x8, x8, #0x4\n"
      "fmla v16.4s, v9.4s, v0.s[0]\n"
      "fmla v18.4s, v9.4s, v1.s[0]\n"
      "fmla v20.4s, v9.4s, v2.s[0]\n"
      "prfm pldl1keep, [x4, #0x80]\n"
      "prfm pldl1keep, [x3, #0x80]\n"
      "fmla v22.4s, v9.4s, v3.s[0]\n"
      "fmla v24.4s, v9.4s, v4.s[0]\n"
      "prfm pldl1keep, [x2, #0x80]\n"
      "prfm pldl1keep, [x1, #0x80]\n"
      "fmla v26.4s, v9.4s, v5.s[0]\n"
      "fmla v28.4s, v9.4s, v6.s[0]\n"
      "prfm pldl1keep, [x17, #0x80]\n"
      "fmla v30.4s, v9.4s, v7.s[0]\n"
      "fmla v17.4s, v10.4s, v0.s[0]\n"
      "movi v9.16b, #0x0\n"
      "fmla v19.4s, v10.4s, v1.s[0]\n"
      "fmla v21.4s, v10.4s, v2.s[0]\n"
      "fmla v23.4s, v10.4s, v3.s[0]\n"
      "fmla v25.4s, v10.4s, v4.s[0]\n"
      "fmla v27.4s, v10.4s, v5.s[0]\n"
      "fmla v29.4s, v10.4s, v6.s[0]\n"
      "fmla v31.4s, v10.4s, v7.s[0]\n"
      "ldr q10, [x10, #0x30]\n"
      "fmlal2 v9.4s, v14.4h, v15.4h\n"
      "add x10, x10, #0x40\n"
      "fmla v16.4s, v12.4s, v0.s[1]\n"
      "fmla v18.4s, v12.4s, v1.s[1]\n"
      "fmla v20.4s, v12.4s, v2.s[1]\n"
      "fmla v22.4s, v12.4s, v3.s[1]\n"
      "fmla v24.4s, v12.4s, v4.s[1]\n"
      "fmla v26.4s, v12.4s, v5.s[1]\n"
      "fmla v28.4s, v12.4s, v6.s[1]\n"
      "fmla v30.4s, v12.4s, v7.s[1]\n"
      "movi v12.16b, #0x0\n"
      "fmla v17.4s, v13.4s, v0.s[1]\n"
      "fmla v19.4s, v13.4s, v1.s[1]\n"
      "fmla v21.4s, v13.4s, v2.s[1]\n"
      "fmla v23.4s, v13.4s, v3.s[1]\n"
      "fmla v25.4s, v13.4s, v4.s[1]\n"
      "fmla v27.4s, v13.4s, v5.s[1]\n"
      "fmla v29.4s, v13.4s, v6.s[1]\n"
      "fmla v31.4s, v13.4s, v7.s[1]\n"
      "fmlal v11.4s, v10.4h, v15.4h\n"
      "fmla v16.4s, v8.4s, v0.s[2]\n"
      "fmla v18.4s, v8.4s, v1.s[2]\n"
      "fmla v20.4s, v8.4s, v2.s[2]\n"
      "fmla v22.4s, v8.4s, v3.s[2]\n"
      "fmla v24.4s, v8.4s, v4.s[2]\n"
      "fmla v26.4s, v8.4s, v5.s[2]\n"
      "fmla v28.4s, v8.4s, v6.s[2]\n"
      "fmla v30.4s, v8.4s, v7.s[2]\n"
      "fmla v17.4s, v9.4s, v0.s[2]\n"
      "fmla v19.4s, v9.4s, v1.s[2]\n"
      "fmla v21.4s, v9.4s, v2.s[2]\n"
      "fmla v23.4s, v9.4s, v3.s[2]\n"
      "fmla v25.4s, v9.4s, v4.s[2]\n"
      "fmla v27.4s, v9.4s, v5.s[2]\n"
      "fmla v29.4s, v9.4s, v6.s[2]\n"
      "fmla v31.4s, v9.4s, v7.s[2]\n"
      "fmlal2 v12.4s, v10.4h, v15.4h\n"
      "fmla v16.4s, v11.4s, v0.s[3]\n"
      "fmla v18.4s, v11.4s, v1.s[3]\n"
      "fmla v20.4s, v11.4s, v2.s[3]\n"
      "fmla v22.4s, v11.4s, v3.s[3]\n"
      "fmla v24.4s, v11.4s, v4.s[3]\n"
      "fmla v26.4s, v11.4s, v5.s[3]\n"
      "fmla v28.4s, v11.4s, v6.s[3]\n"
      "fmla v30.4s, v11.4s, v7.s[3]\n"
      "fmla v17.4s, v12.4s, v0.s[3]\n"
      "fmla v19.4s, v12.4s, v1.s[3]\n"
      "fmla v21.4s, v12.4s, v2.s[3]\n"
      "fmla v23.4s, v12.4s, v3.s[3]\n"
      "fmla v25.4s, v12.4s, v4.s[3]\n"
      "fmla v27.4s, v12.4s, v5.s[3]\n"
      "fmla v29.4s, v12.4s, v6.s[3]\n"
      "fmla v31.4s, v12.4s, v7.s[3]\n"
      "7:" // Height 8: Multiply loop: Main loop skip
      "cbz x8, 9f\n"
      "8:" // Height 8: Multiply loop: Odd block loop
      "ldr q13, [x10, #0x0]\n"
      "ldr s0, [x7], #0x4\n"
      "movi v14.16b, #0x0\n"
      "movi v8.16b, #0x0\n"
      "ldr s1, [x6], #0x4\n"
      "ldr s2, [x5], #0x4\n"
      "sub x8, x8, #0x1\n"
      "add x10, x10, #0x10\n"
      "ldr s3, [x4], #0x4\n"
      "ldr s4, [x3], #0x4\n"
      "ldr s5, [x2], #0x4\n"
      "ldr s6, [x1], #0x4\n"
      "fmlal v14.4s, v13.4h, v15.4h\n"
      "fmlal2 v8.4s, v13.4h, v15.4h\n"
      "ldr s7, [x17], #0x4\n"
      "fmla v16.4s, v14.4s, v0.s[0]\n"
      "fmla v18.4s, v14.4s, v1.s[0]\n"
      "fmla v20.4s, v14.4s, v2.s[0]\n"
      "fmla v22.4s, v14.4s, v3.s[0]\n"
      "fmla v24.4s, v14.4s, v4.s[0]\n"
      "fmla v26.4s, v14.4s, v5.s[0]\n"
      "fmla v28.4s, v14.4s, v6.s[0]\n"
      "fmla v30.4s, v14.4s, v7.s[0]\n"
      "fmla v17.4s, v8.4s, v0.s[0]\n"
      "fmla v19.4s, v8.4s, v1.s[0]\n"
      "fmla v21.4s, v8.4s, v2.s[0]\n"
      "fmla v23.4s, v8.4s, v3.s[0]\n"
      "fmla v25.4s, v8.4s, v4.s[0]\n"
      "fmla v27.4s, v8.4s, v5.s[0]\n"
      "fmla v29.4s, v8.4s, v6.s[0]\n"
      "fmla v31.4s, v8.4s, v7.s[0]\n"
      "cbnz x8, 8b\n"
      "9:" // Height 8: Multiply loop: No odd multiplies
      "prfm pstl1keep, [x9, #0x0]\n"
      "str q16, [x9, #0x0]\n"
      "str q17, [x9, #0x10]\n"
      "add x17, x9, x16\n"
      "add x9, x9, #0x20\n"
      "prfm pstl1keep, [x17, #0x0]\n"
      "str q18, [x17, #0x0]\n"
      "add x5, x17, x16\n"
      "add x4, x5, x16\n"
      "add x3, x4, x16\n"
      "prfm pstl1keep, [x5, #0x0]\n"
      "prfm pstl1keep, [x4, #0x0]\n"
      "str q19, [x17, #0x10]\n"
      "add x2, x3, x16\n"
      "prfm pstl1keep, [x3, #0x0]\n"
      "str q20, [x5, #0x0]\n"
      "add x1, x2, x16\n"
      "add x17, x1, x16\n"
      "prfm pstl1keep, [x2, #0x0]\n"
      "prfm pstl1keep, [x1, #0x0]\n"
      "str q21, [x5, #0x10]\n"
      "prfm pstl1keep, [x17, #0x0]\n"
      "str q22, [x4, #0x0]\n"
      "str q23, [x4, #0x10]\n"
      "str q24, [x3, #0x0]\n"
      "str q25, [x3, #0x10]\n"
      "str q26, [x2, #0x0]\n"
      "str q27, [x2, #0x10]\n"
      "str q28, [x1, #0x0]\n"
      "str q29, [x1, #0x10]\n"
      "str q30, [x17, #0x0]\n"
      "str q31, [x17, #0x10]\n"
      "subs x11, x11, #0x1\n"
      "bgt 1b\n"
      :
      : [gp] "r"(gp),
        [offsetof_A] "I"(offsetof(GemmParamsFP16, A)),
        [offsetof_B] "I"(offsetof(GemmParamsFP16, B)),
        [offsetof_C] "I"(offsetof(GemmParamsFP16, C)),
        [offsetof_b_block_cols] "I"(offsetof(GemmParamsFP16, b_block_cols)),
        [offsetof_beta] "I"(offsetof(GemmParamsFP16, beta)),
        [offsetof_k] "I"(offsetof(GemmParamsFP16, k)),
        [offsetof_lda] "I"(offsetof(GemmParamsFP16, lda)),
        [offsetof_ldc] "I"(offsetof(GemmParamsFP16, ldc))
      : "cc",
        "memory",
        "v0",
        "v1",
        "v10",
        "v11",
        "v12",
        "v13",
        "v14",
        "v15",
        "v16",
        "v17",
        "v18",
        "v19",
        "v2",
        "v20",
        "v21",
        "v22",
        "v23",
        "v24",
        "v25",
        "v26",
        "v27",
        "v28",
        "v29",
        "v3",
        "v30",
        "v31",
        "v4",
        "v5",
        "v6",
        "v7",
        "v8",
        "v9",
        "x10",
        "x11",
        "x12",
        "x13",
        "x14",
        "x15",
        "x16",
        "x17",
        "x1",
        "x2",
        "x3",
        "x4",
        "x5",
        "x6",
        "x7",
        "x8",
        "x9");
#endif // __aarch64__
}

} // namespace kleidiai

#endif
