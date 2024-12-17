// @lint-ignore-every LICENSELINT
//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates
// <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#ifdef FBGEMM_ENABLE_KLEIDIAI

#include "KleidiAIFP16UKernelsNeon.h"

namespace kleidiai {

void NOINLINE gemmkernel_1x1_Neon_fp16_fA0fB0fC0(GemmParamsFP16* gp) {
#ifdef __aarch64__
  __asm__ __volatile__(
      "ldr w20, [%x[gp], %[offsetof_beta]]\n"
      "mov x25, #0x1\n"
      "fmov v29.8h, #1.0\n"
      "ldr x24, [%x[gp], %[offsetof_b_block_cols]]\n"
      "ldr x23, [%x[gp], %[offsetof_B]]\n"
      "ldr x22, [%x[gp], %[offsetof_C]]\n"
      "bic x20, x20, #0x80000000\n"
      "cmp x20, #0x0\n"
      "csel x25, XZR, x25, EQ\n"
      "1:" // Height 1: Column loop
      "tbz x25, #0, 2f\n"
      "ldr q30, [x22, #0x0]\n"
      "ldr q31, [x22, #0x10]\n"
      "add x20, %x[gp], %[offsetof_beta]\n"
      "ld1r { v16.4s }, [x20]\n"
      "fmul v30.4s, v30.4s, v16.4s\n"
      "fmul v31.4s, v31.4s, v16.4s\n"
      "b 3f\n"
      "2:" // Height 1: no accumulate
      "movi v30.16b, #0x0\n"
      "movi v31.16b, #0x0\n"
      "3:" // Height 1: setup done
      "ldr x20, [%x[gp], %[offsetof_A]]\n"
      "ldr x21, [%x[gp], %[offsetof_k]]\n"
      "mov x20, x20\n"
      "cmp x21, #0x4\n"
      "blt 7f\n"
      "ldr q0, [x20, #0x0]\n"
      "ldr q1, [x23, #0x0]\n"
      "cmp x21, #0x8\n"
      "ldr q4, [x23, #0x10]\n"
      "ldr q7, [x23, #0x20]\n"
      "ldr q10, [x23, #0x30]\n"
      "blt 6f\n"
      "5:" // Height 1: Multiply loop: Main loop head
      "movi v2.16b, #0x0\n"
      "movi v3.16b, #0x0\n"
      "sub x21, x21, #0x4\n"
      "add x20, x20, #0x10\n"
      "movi v5.16b, #0x0\n"
      "movi v6.16b, #0x0\n"
      "cmp x21, #0x8\n"
      "add x23, x23, #0x40\n"
      "fmlal v2.4s, v1.4h, v29.4h\n"
      "fmlal2 v3.4s, v1.4h, v29.4h\n"
      "ldr q1, [x23, #0x0]\n"
      "movi v8.16b, #0x0\n"
      "fmlal v5.4s, v4.4h, v29.4h\n"
      "fmlal2 v6.4s, v4.4h, v29.4h\n"
      "ldr q4, [x23, #0x10]\n"
      "movi v9.16b, #0x0\n"
      "fmlal v8.4s, v7.4h, v29.4h\n"
      "movi v11.16b, #0x0\n"
      "prfm pldl1keep, [x20, #0x80]\n"
      "fmlal2 v9.4s, v7.4h, v29.4h\n"
      "ldr q7, [x23, #0x20]\n"
      "movi v12.16b, #0x0\n"
      "fmla v30.4s, v2.4s, v0.s[0]\n"
      "fmla v31.4s, v3.4s, v0.s[0]\n"
      "fmlal v11.4s, v10.4h, v29.4h\n"
      "fmlal2 v12.4s, v10.4h, v29.4h\n"
      "ldr q10, [x23, #0x30]\n"
      "fmla v30.4s, v5.4s, v0.s[1]\n"
      "fmla v31.4s, v6.4s, v0.s[1]\n"
      "fmla v30.4s, v8.4s, v0.s[2]\n"
      "fmla v31.4s, v9.4s, v0.s[2]\n"
      "fmla v30.4s, v11.4s, v0.s[3]\n"
      "fmla v31.4s, v12.4s, v0.s[3]\n"
      "ldr q0, [x20, #0x0]\n"
      "bge 5b\n"
      "6:" // Height 1: Multiply loop: Single iteration only
      "movi v2.16b, #0x0\n"
      "movi v3.16b, #0x0\n"
      "add x20, x20, #0x10\n"
      "sub x21, x21, #0x4\n"
      "movi v5.16b, #0x0\n"
      "movi v6.16b, #0x0\n"
      "prfm pldl1keep, [x20, #0x80]\n"
      "add x23, x23, #0x40\n"
      "fmlal v2.4s, v1.4h, v29.4h\n"
      "fmlal2 v3.4s, v1.4h, v29.4h\n"
      "movi v8.16b, #0x0\n"
      "fmlal v5.4s, v4.4h, v29.4h\n"
      "fmlal2 v6.4s, v4.4h, v29.4h\n"
      "movi v9.16b, #0x0\n"
      "fmlal v8.4s, v7.4h, v29.4h\n"
      "movi v11.16b, #0x0\n"
      "fmlal2 v9.4s, v7.4h, v29.4h\n"
      "movi v12.16b, #0x0\n"
      "fmla v30.4s, v2.4s, v0.s[0]\n"
      "fmla v31.4s, v3.4s, v0.s[0]\n"
      "fmlal v11.4s, v10.4h, v29.4h\n"
      "fmlal2 v12.4s, v10.4h, v29.4h\n"
      "fmla v30.4s, v5.4s, v0.s[1]\n"
      "fmla v31.4s, v6.4s, v0.s[1]\n"
      "fmla v30.4s, v8.4s, v0.s[2]\n"
      "fmla v31.4s, v9.4s, v0.s[2]\n"
      "fmla v30.4s, v11.4s, v0.s[3]\n"
      "fmla v31.4s, v12.4s, v0.s[3]\n"
      "7:" // Height 1: Multiply loop: Main loop skip
      "cbz x21, 9f\n"
      "8:" // Height 1: Multiply loop: Odd block loop
      "ldr q13, [x23, #0x0]\n"
      "ldr s0, [x20], #0x4\n"
      "movi v14.16b, #0x0\n"
      "movi v15.16b, #0x0\n"
      "sub x21, x21, #0x1\n"
      "add x23, x23, #0x10\n"
      "fmlal v14.4s, v13.4h, v29.4h\n"
      "fmlal2 v15.4s, v13.4h, v29.4h\n"
      "fmla v30.4s, v14.4s, v0.s[0]\n"
      "fmla v31.4s, v15.4s, v0.s[0]\n"
      "cbnz x21, 8b\n"
      "9:" // Height 1: Multiply loop: No odd multiplies
      "prfm pstl1keep, [x22, #0x0]\n"
      "str q30, [x22, #0x0]\n"
      "str q31, [x22, #0x10]\n"
      "add x22, x22, #0x20\n"
      "subs x24, x24, #0x1\n"
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
        "v10",
        "v11",
        "v12",
        "v13",
        "v14",
        "v15",
        "v16",
        "v2",
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
        "x20",
        "x21",
        "x22",
        "x23",
        "x24",
        "x25");
#endif // __aarch64__
}

void NOINLINE gemmkernel_2x1_Neon_fp16_fA0fB0fC0(GemmParamsFP16* gp) {
#ifdef __aarch64__
  __asm__ __volatile__(
      "ldr w20, [%x[gp], %[offsetof_beta]]\n"
      "mov x26, #0x1\n"
      "fmov v27.8h, #1.0\n"
      "ldr x25, [%x[gp], %[offsetof_b_block_cols]]\n"
      "ldr x24, [%x[gp], %[offsetof_B]]\n"
      "ldr x23, [%x[gp], %[offsetof_C]]\n"
      "bic x20, x20, #0x80000000\n"
      "cmp x20, #0x0\n"
      "csel x26, XZR, x26, EQ\n"
      "1:" // Height 2: Column loop
      "tbz x26, #0, 2f\n"
      "ldr q28, [x23, #0x0]\n"
      "ldr q29, [x23, #0x10]\n"
      "add x20, %x[gp], %[offsetof_beta]\n"
      "ld1r { v16.4s }, [x20]\n"
      "ldr x20, [%x[gp], %[offsetof_ldc]]\n"
      "add x20, x23, x20\n"
      "ldr q30, [x20, #0x0]\n"
      "ldr q31, [x20, #0x10]\n"
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
      "ldr x21, [%x[gp], %[offsetof_A]]\n"
      "ldr x20, [%x[gp], %[offsetof_lda]]\n"
      "ldr x22, [%x[gp], %[offsetof_k]]\n"
      "mov x21, x21\n"
      "add x20, x21, x20\n"
      "cmp x22, #0x4\n"
      "blt 7f\n"
      "ldr q0, [x21, #0x0]\n"
      "ldr q2, [x24, #0x0]\n"
      "cmp x22, #0x8\n"
      "ldr q1, [x20, #0x0]\n"
      "ldr q5, [x24, #0x10]\n"
      "ldr q8, [x24, #0x20]\n"
      "ldr q11, [x24, #0x30]\n"
      "blt 6f\n"
      "5:" // Height 2: Multiply loop: Main loop head
      "movi v3.16b, #0x0\n"
      "movi v4.16b, #0x0\n"
      "sub x22, x22, #0x4\n"
      "add x21, x21, #0x10\n"
      "movi v6.16b, #0x0\n"
      "movi v7.16b, #0x0\n"
      "add x20, x20, #0x10\n"
      "cmp x22, #0x8\n"
      "fmlal v3.4s, v2.4h, v27.4h\n"
      "fmlal2 v4.4s, v2.4h, v27.4h\n"
      "movi v9.16b, #0x0\n"
      "add x24, x24, #0x40\n"
      "ldr q2, [x24, #0x0]\n"
      "fmlal v6.4s, v5.4h, v27.4h\n"
      "fmlal2 v7.4s, v5.4h, v27.4h\n"
      "ldr q5, [x24, #0x10]\n"
      "movi v10.16b, #0x0\n"
      "fmlal v9.4s, v8.4h, v27.4h\n"
      "prfm pldl1keep, [x21, #0x80]\n"
      "prfm pldl1keep, [x20, #0x80]\n"
      "movi v12.16b, #0x0\n"
      "movi v13.16b, #0x0\n"
      "fmla v28.4s, v3.4s, v0.s[0]\n"
      "fmla v30.4s, v3.4s, v1.s[0]\n"
      "fmla v29.4s, v4.4s, v0.s[0]\n"
      "fmla v31.4s, v4.4s, v1.s[0]\n"
      "fmlal2 v10.4s, v8.4h, v27.4h\n"
      "ldr q8, [x24, #0x20]\n"
      "fmlal v12.4s, v11.4h, v27.4h\n"
      "fmlal2 v13.4s, v11.4h, v27.4h\n"
      "ldr q11, [x24, #0x30]\n"
      "fmla v28.4s, v6.4s, v0.s[1]\n"
      "fmla v30.4s, v6.4s, v1.s[1]\n"
      "fmla v29.4s, v7.4s, v0.s[1]\n"
      "fmla v31.4s, v7.4s, v1.s[1]\n"
      "fmla v28.4s, v9.4s, v0.s[2]\n"
      "fmla v30.4s, v9.4s, v1.s[2]\n"
      "fmla v29.4s, v10.4s, v0.s[2]\n"
      "fmla v31.4s, v10.4s, v1.s[2]\n"
      "fmla v28.4s, v12.4s, v0.s[3]\n"
      "fmla v30.4s, v12.4s, v1.s[3]\n"
      "fmla v29.4s, v13.4s, v0.s[3]\n"
      "ldr q0, [x21, #0x0]\n"
      "fmla v31.4s, v13.4s, v1.s[3]\n"
      "ldr q1, [x20, #0x0]\n"
      "bge 5b\n"
      "6:" // Height 2: Multiply loop: Single iteration only
      "movi v3.16b, #0x0\n"
      "movi v4.16b, #0x0\n"
      "add x21, x21, #0x10\n"
      "add x20, x20, #0x10\n"
      "movi v6.16b, #0x0\n"
      "movi v7.16b, #0x0\n"
      "sub x22, x22, #0x4\n"
      "prfm pldl1keep, [x21, #0x80]\n"
      "fmlal v3.4s, v2.4h, v27.4h\n"
      "fmlal2 v4.4s, v2.4h, v27.4h\n"
      "movi v9.16b, #0x0\n"
      "prfm pldl1keep, [x20, #0x80]\n"
      "fmlal v6.4s, v5.4h, v27.4h\n"
      "fmlal2 v7.4s, v5.4h, v27.4h\n"
      "movi v10.16b, #0x0\n"
      "add x24, x24, #0x40\n"
      "fmlal v9.4s, v8.4h, v27.4h\n"
      "movi v12.16b, #0x0\n"
      "fmlal2 v10.4s, v8.4h, v27.4h\n"
      "movi v13.16b, #0x0\n"
      "fmla v28.4s, v3.4s, v0.s[0]\n"
      "fmla v30.4s, v3.4s, v1.s[0]\n"
      "fmla v29.4s, v4.4s, v0.s[0]\n"
      "fmla v31.4s, v4.4s, v1.s[0]\n"
      "fmlal v12.4s, v11.4h, v27.4h\n"
      "fmlal2 v13.4s, v11.4h, v27.4h\n"
      "fmla v28.4s, v6.4s, v0.s[1]\n"
      "fmla v30.4s, v6.4s, v1.s[1]\n"
      "fmla v29.4s, v7.4s, v0.s[1]\n"
      "fmla v31.4s, v7.4s, v1.s[1]\n"
      "fmla v28.4s, v9.4s, v0.s[2]\n"
      "fmla v30.4s, v9.4s, v1.s[2]\n"
      "fmla v29.4s, v10.4s, v0.s[2]\n"
      "fmla v31.4s, v10.4s, v1.s[2]\n"
      "fmla v28.4s, v12.4s, v0.s[3]\n"
      "fmla v30.4s, v12.4s, v1.s[3]\n"
      "fmla v29.4s, v13.4s, v0.s[3]\n"
      "fmla v31.4s, v13.4s, v1.s[3]\n"
      "7:" // Height 2: Multiply loop: Main loop skip
      "cbz x22, 9f\n"
      "8:" // Height 2: Multiply loop: Odd block loop
      "ldr q14, [x24, #0x0]\n"
      "ldr s0, [x21], #0x4\n"
      "movi v15.16b, #0x0\n"
      "movi v16.16b, #0x0\n"
      "ldr s1, [x20], #0x4\n"
      "sub x22, x22, #0x1\n"
      "add x24, x24, #0x10\n"
      "fmlal v15.4s, v14.4h, v27.4h\n"
      "fmlal2 v16.4s, v14.4h, v27.4h\n"
      "fmla v28.4s, v15.4s, v0.s[0]\n"
      "fmla v30.4s, v15.4s, v1.s[0]\n"
      "fmla v29.4s, v16.4s, v0.s[0]\n"
      "fmla v31.4s, v16.4s, v1.s[0]\n"
      "cbnz x22, 8b\n"
      "9:" // Height 2: Multiply loop: No odd multiplies
      "ldr x20, [%x[gp], %[offsetof_ldc]]\n"
      "prfm pstl1keep, [x23, #0x0]\n"
      "str q28, [x23, #0x0]\n"
      "str q29, [x23, #0x10]\n"
      "add x20, x23, x20\n"
      "add x23, x23, #0x20\n"
      "prfm pstl1keep, [x20, #0x0]\n"
      "str q30, [x20, #0x0]\n"
      "str q31, [x20, #0x10]\n"
      "subs x25, x25, #0x1\n"
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
        "v2",
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
        "x20",
        "x21",
        "x22",
        "x23",
        "x24",
        "x25",
        "x26");
#endif // __aarch64__
}

void NOINLINE gemmkernel_3x1_Neon_fp16_fA0fB0fC0(GemmParamsFP16* gp) {
#ifdef __aarch64__
  __asm__ __volatile__(
      "ldr w20, [%x[gp], %[offsetof_beta]]\n"
      "mov x27, #0x1\n"
      "fmov v25.8h, #1.0\n"
      "ldr x26, [%x[gp], %[offsetof_b_block_cols]]\n"
      "ldr x25, [%x[gp], %[offsetof_B]]\n"
      "ldr x24, [%x[gp], %[offsetof_C]]\n"
      "bic x20, x20, #0x80000000\n"
      "cmp x20, #0x0\n"
      "csel x27, XZR, x27, EQ\n"
      "1:" // Height 3: Column loop
      "tbz x27, #0, 2f\n"
      "ldr q26, [x24, #0x0]\n"
      "ldr q27, [x24, #0x10]\n"
      "add x20, %x[gp], %[offsetof_beta]\n"
      "ld1r { v16.4s }, [x20]\n"
      "ldr x21, [%x[gp], %[offsetof_ldc]]\n"
      "add x20, x24, x21\n"
      "ldr q28, [x20, #0x0]\n"
      "ldr q29, [x20, #0x10]\n"
      "add x20, x20, x21\n"
      "ldr q30, [x20, #0x0]\n"
      "ldr q31, [x20, #0x10]\n"
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
      "ldr x21, [%x[gp], %[offsetof_A]]\n"
      "ldr x20, [%x[gp], %[offsetof_lda]]\n"
      "ldr x23, [%x[gp], %[offsetof_k]]\n"
      "mov x22, x21\n"
      "add x21, x22, x20\n"
      "add x20, x21, x20\n"
      "cmp x23, #0x4\n"
      "blt 7f\n"
      "ldr q0, [x22, #0x0]\n"
      "ldr q3, [x25, #0x0]\n"
      "cmp x23, #0x8\n"
      "ldr q1, [x21, #0x0]\n"
      "ldr q2, [x20, #0x0]\n"
      "ldr q6, [x25, #0x10]\n"
      "ldr q9, [x25, #0x20]\n"
      "ldr q12, [x25, #0x30]\n"
      "blt 6f\n"
      "5:" // Height 3: Multiply loop: Main loop head
      "movi v4.16b, #0x0\n"
      "movi v5.16b, #0x0\n"
      "sub x23, x23, #0x4\n"
      "add x22, x22, #0x10\n"
      "movi v7.16b, #0x0\n"
      "movi v8.16b, #0x0\n"
      "add x21, x21, #0x10\n"
      "add x20, x20, #0x10\n"
      "fmlal v4.4s, v3.4h, v25.4h\n"
      "fmlal2 v5.4s, v3.4h, v25.4h\n"
      "movi v10.16b, #0x0\n"
      "cmp x23, #0x8\n"
      "fmlal v7.4s, v6.4h, v25.4h\n"
      "fmlal2 v8.4s, v6.4h, v25.4h\n"
      "movi v11.16b, #0x0\n"
      "add x25, x25, #0x40\n"
      "ldr q3, [x25, #0x0]\n"
      "ldr q6, [x25, #0x10]\n"
      "fmlal v10.4s, v9.4h, v25.4h\n"
      "movi v13.16b, #0x0\n"
      "fmlal2 v11.4s, v9.4h, v25.4h\n"
      "ldr q9, [x25, #0x20]\n"
      "movi v14.16b, #0x0\n"
      "prfm pldl1keep, [x22, #0x80]\n"
      "fmla v26.4s, v4.4s, v0.s[0]\n"
      "fmla v28.4s, v4.4s, v1.s[0]\n"
      "prfm pldl1keep, [x21, #0x80]\n"
      "prfm pldl1keep, [x20, #0x80]\n"
      "fmla v30.4s, v4.4s, v2.s[0]\n"
      "fmla v27.4s, v5.4s, v0.s[0]\n"
      "fmla v29.4s, v5.4s, v1.s[0]\n"
      "fmla v31.4s, v5.4s, v2.s[0]\n"
      "fmlal v13.4s, v12.4h, v25.4h\n"
      "fmlal2 v14.4s, v12.4h, v25.4h\n"
      "ldr q12, [x25, #0x30]\n"
      "fmla v26.4s, v7.4s, v0.s[1]\n"
      "fmla v28.4s, v7.4s, v1.s[1]\n"
      "fmla v30.4s, v7.4s, v2.s[1]\n"
      "fmla v27.4s, v8.4s, v0.s[1]\n"
      "fmla v29.4s, v8.4s, v1.s[1]\n"
      "fmla v31.4s, v8.4s, v2.s[1]\n"
      "fmla v26.4s, v10.4s, v0.s[2]\n"
      "fmla v28.4s, v10.4s, v1.s[2]\n"
      "fmla v30.4s, v10.4s, v2.s[2]\n"
      "fmla v27.4s, v11.4s, v0.s[2]\n"
      "fmla v29.4s, v11.4s, v1.s[2]\n"
      "fmla v31.4s, v11.4s, v2.s[2]\n"
      "fmla v26.4s, v13.4s, v0.s[3]\n"
      "fmla v28.4s, v13.4s, v1.s[3]\n"
      "fmla v30.4s, v13.4s, v2.s[3]\n"
      "fmla v27.4s, v14.4s, v0.s[3]\n"
      "ldr q0, [x22, #0x0]\n"
      "fmla v29.4s, v14.4s, v1.s[3]\n"
      "ldr q1, [x21, #0x0]\n"
      "fmla v31.4s, v14.4s, v2.s[3]\n"
      "ldr q2, [x20, #0x0]\n"
      "bge 5b\n"
      "6:" // Height 3: Multiply loop: Single iteration only
      "movi v4.16b, #0x0\n"
      "movi v5.16b, #0x0\n"
      "add x22, x22, #0x10\n"
      "add x21, x21, #0x10\n"
      "movi v7.16b, #0x0\n"
      "movi v8.16b, #0x0\n"
      "add x20, x20, #0x10\n"
      "sub x23, x23, #0x4\n"
      "fmlal v4.4s, v3.4h, v25.4h\n"
      "fmlal2 v5.4s, v3.4h, v25.4h\n"
      "movi v10.16b, #0x0\n"
      "prfm pldl1keep, [x22, #0x80]\n"
      "fmlal v7.4s, v6.4h, v25.4h\n"
      "fmlal2 v8.4s, v6.4h, v25.4h\n"
      "movi v11.16b, #0x0\n"
      "prfm pldl1keep, [x21, #0x80]\n"
      "fmlal v10.4s, v9.4h, v25.4h\n"
      "movi v13.16b, #0x0\n"
      "prfm pldl1keep, [x20, #0x80]\n"
      "add x25, x25, #0x40\n"
      "fmlal2 v11.4s, v9.4h, v25.4h\n"
      "movi v14.16b, #0x0\n"
      "fmla v26.4s, v4.4s, v0.s[0]\n"
      "fmla v28.4s, v4.4s, v1.s[0]\n"
      "fmla v30.4s, v4.4s, v2.s[0]\n"
      "fmla v27.4s, v5.4s, v0.s[0]\n"
      "fmla v29.4s, v5.4s, v1.s[0]\n"
      "fmla v31.4s, v5.4s, v2.s[0]\n"
      "fmlal v13.4s, v12.4h, v25.4h\n"
      "fmla v26.4s, v7.4s, v0.s[1]\n"
      "fmlal2 v14.4s, v12.4h, v25.4h\n"
      "fmla v28.4s, v7.4s, v1.s[1]\n"
      "fmla v30.4s, v7.4s, v2.s[1]\n"
      "fmla v27.4s, v8.4s, v0.s[1]\n"
      "fmla v29.4s, v8.4s, v1.s[1]\n"
      "fmla v31.4s, v8.4s, v2.s[1]\n"
      "fmla v26.4s, v10.4s, v0.s[2]\n"
      "fmla v28.4s, v10.4s, v1.s[2]\n"
      "fmla v30.4s, v10.4s, v2.s[2]\n"
      "fmla v27.4s, v11.4s, v0.s[2]\n"
      "fmla v29.4s, v11.4s, v1.s[2]\n"
      "fmla v31.4s, v11.4s, v2.s[2]\n"
      "fmla v26.4s, v13.4s, v0.s[3]\n"
      "fmla v28.4s, v13.4s, v1.s[3]\n"
      "fmla v30.4s, v13.4s, v2.s[3]\n"
      "fmla v27.4s, v14.4s, v0.s[3]\n"
      "fmla v29.4s, v14.4s, v1.s[3]\n"
      "fmla v31.4s, v14.4s, v2.s[3]\n"
      "7:" // Height 3: Multiply loop: Main loop skip
      "cbz x23, 9f\n"
      "8:" // Height 3: Multiply loop: Odd block loop
      "ldr q15, [x25, #0x0]\n"
      "ldr s0, [x22], #0x4\n"
      "movi v16.16b, #0x0\n"
      "movi v17.16b, #0x0\n"
      "ldr s1, [x21], #0x4\n"
      "ldr s2, [x20], #0x4\n"
      "sub x23, x23, #0x1\n"
      "add x25, x25, #0x10\n"
      "fmlal v16.4s, v15.4h, v25.4h\n"
      "fmlal2 v17.4s, v15.4h, v25.4h\n"
      "fmla v26.4s, v16.4s, v0.s[0]\n"
      "fmla v28.4s, v16.4s, v1.s[0]\n"
      "fmla v30.4s, v16.4s, v2.s[0]\n"
      "fmla v27.4s, v17.4s, v0.s[0]\n"
      "fmla v29.4s, v17.4s, v1.s[0]\n"
      "fmla v31.4s, v17.4s, v2.s[0]\n"
      "cbnz x23, 8b\n"
      "9:" // Height 3: Multiply loop: No odd multiplies
      "ldr x20, [%x[gp], %[offsetof_ldc]]\n"
      "prfm pstl1keep, [x24, #0x0]\n"
      "str q26, [x24, #0x0]\n"
      "str q27, [x24, #0x10]\n"
      "add x21, x24, x20\n"
      "add x24, x24, #0x20\n"
      "prfm pstl1keep, [x21, #0x0]\n"
      "str q28, [x21, #0x0]\n"
      "add x20, x21, x20\n"
      "prfm pstl1keep, [x20, #0x0]\n"
      "str q29, [x21, #0x10]\n"
      "str q30, [x20, #0x0]\n"
      "str q31, [x20, #0x10]\n"
      "subs x26, x26, #0x1\n"
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
        "v2",
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
        "x20",
        "x21",
        "x22",
        "x23",
        "x24",
        "x25",
        "x26",
        "x27");
#endif // __aarch64__
}

void NOINLINE gemmkernel_4x1_Neon_fp16_fA0fB0fC0(GemmParamsFP16* gp) {
#ifdef __aarch64__
  __asm__ __volatile__(
      "ldr w20, [%x[gp], %[offsetof_beta]]\n"
      "mov x28, #0x1\n"
      "fmov v23.8h, #1.0\n"
      "ldr x27, [%x[gp], %[offsetof_b_block_cols]]\n"
      "ldr x26, [%x[gp], %[offsetof_B]]\n"
      "ldr x25, [%x[gp], %[offsetof_C]]\n"
      "bic x20, x20, #0x80000000\n"
      "cmp x20, #0x0\n"
      "csel x28, XZR, x28, EQ\n"
      "1:" // Height 4: Column loop
      "tbz x28, #0, 2f\n"
      "ldr q24, [x25, #0x0]\n"
      "ldr q25, [x25, #0x10]\n"
      "add x20, %x[gp], %[offsetof_beta]\n"
      "ld1r { v16.4s }, [x20]\n"
      "ldr x21, [%x[gp], %[offsetof_ldc]]\n"
      "add x20, x25, x21\n"
      "ldr q26, [x20, #0x0]\n"
      "ldr q27, [x20, #0x10]\n"
      "add x20, x20, x21\n"
      "ldr q28, [x20, #0x0]\n"
      "ldr q29, [x20, #0x10]\n"
      "add x20, x20, x21\n"
      "ldr q30, [x20, #0x0]\n"
      "ldr q31, [x20, #0x10]\n"
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
      "ldr x21, [%x[gp], %[offsetof_A]]\n"
      "ldr x20, [%x[gp], %[offsetof_lda]]\n"
      "ldr x24, [%x[gp], %[offsetof_k]]\n"
      "mov x23, x21\n"
      "add x22, x23, x20\n"
      "add x21, x22, x20\n"
      "add x20, x21, x20\n"
      "cmp x24, #0x4\n"
      "blt 7f\n"
      "ldr q0, [x23, #0x0]\n"
      "ldr q4, [x26, #0x0]\n"
      "cmp x24, #0x8\n"
      "ldr q1, [x22, #0x0]\n"
      "ldr q2, [x21, #0x0]\n"
      "ldr q3, [x20, #0x0]\n"
      "ldr q7, [x26, #0x10]\n"
      "ldr q10, [x26, #0x20]\n"
      "ldr q13, [x26, #0x30]\n"
      "blt 6f\n"
      "5:" // Height 4: Multiply loop: Main loop head
      "movi v5.16b, #0x0\n"
      "movi v6.16b, #0x0\n"
      "sub x24, x24, #0x4\n"
      "add x23, x23, #0x10\n"
      "movi v8.16b, #0x0\n"
      "movi v9.16b, #0x0\n"
      "add x22, x22, #0x10\n"
      "add x21, x21, #0x10\n"
      "fmlal v5.4s, v4.4h, v23.4h\n"
      "fmlal2 v6.4s, v4.4h, v23.4h\n"
      "movi v11.16b, #0x0\n"
      "add x20, x20, #0x10\n"
      "fmlal v8.4s, v7.4h, v23.4h\n"
      "fmlal2 v9.4s, v7.4h, v23.4h\n"
      "movi v12.16b, #0x0\n"
      "cmp x24, #0x8\n"
      "fmlal v11.4s, v10.4h, v23.4h\n"
      "movi v14.16b, #0x0\n"
      "add x26, x26, #0x40\n"
      "prfm pldl1keep, [x23, #0x80]\n"
      "ldr q4, [x26, #0x0]\n"
      "ldr q7, [x26, #0x10]\n"
      "fmlal2 v12.4s, v10.4h, v23.4h\n"
      "movi v15.16b, #0x0\n"
      "ldr q10, [x26, #0x20]\n"
      "fmla v24.4s, v5.4s, v0.s[0]\n"
      "fmla v26.4s, v5.4s, v1.s[0]\n"
      "prfm pldl1keep, [x22, #0x80]\n"
      "fmla v28.4s, v5.4s, v2.s[0]\n"
      "fmla v30.4s, v5.4s, v3.s[0]\n"
      "prfm pldl1keep, [x21, #0x80]\n"
      "prfm pldl1keep, [x20, #0x80]\n"
      "fmla v25.4s, v6.4s, v0.s[0]\n"
      "fmla v27.4s, v6.4s, v1.s[0]\n"
      "fmla v29.4s, v6.4s, v2.s[0]\n"
      "fmla v31.4s, v6.4s, v3.s[0]\n"
      "fmla v24.4s, v8.4s, v0.s[1]\n"
      "fmla v26.4s, v8.4s, v1.s[1]\n"
      "fmla v28.4s, v8.4s, v2.s[1]\n"
      "fmla v30.4s, v8.4s, v3.s[1]\n"
      "fmla v25.4s, v9.4s, v0.s[1]\n"
      "fmla v27.4s, v9.4s, v1.s[1]\n"
      "fmla v29.4s, v9.4s, v2.s[1]\n"
      "fmla v31.4s, v9.4s, v3.s[1]\n"
      "fmlal v14.4s, v13.4h, v23.4h\n"
      "fmla v24.4s, v11.4s, v0.s[2]\n"
      "fmla v26.4s, v11.4s, v1.s[2]\n"
      "fmla v28.4s, v11.4s, v2.s[2]\n"
      "fmla v30.4s, v11.4s, v3.s[2]\n"
      "fmla v25.4s, v12.4s, v0.s[2]\n"
      "fmla v27.4s, v12.4s, v1.s[2]\n"
      "fmla v29.4s, v12.4s, v2.s[2]\n"
      "fmla v31.4s, v12.4s, v3.s[2]\n"
      "fmlal2 v15.4s, v13.4h, v23.4h\n"
      "ldr q13, [x26, #0x30]\n"
      "fmla v24.4s, v14.4s, v0.s[3]\n"
      "fmla v26.4s, v14.4s, v1.s[3]\n"
      "fmla v28.4s, v14.4s, v2.s[3]\n"
      "fmla v30.4s, v14.4s, v3.s[3]\n"
      "fmla v25.4s, v15.4s, v0.s[3]\n"
      "ldr q0, [x23, #0x0]\n"
      "fmla v27.4s, v15.4s, v1.s[3]\n"
      "ldr q1, [x22, #0x0]\n"
      "fmla v29.4s, v15.4s, v2.s[3]\n"
      "ldr q2, [x21, #0x0]\n"
      "fmla v31.4s, v15.4s, v3.s[3]\n"
      "ldr q3, [x20, #0x0]\n"
      "bge 5b\n"
      "6:" // Height 4: Multiply loop: Single iteration only
      "movi v5.16b, #0x0\n"
      "movi v6.16b, #0x0\n"
      "add x23, x23, #0x10\n"
      "add x22, x22, #0x10\n"
      "movi v8.16b, #0x0\n"
      "movi v9.16b, #0x0\n"
      "add x21, x21, #0x10\n"
      "add x20, x20, #0x10\n"
      "fmlal v5.4s, v4.4h, v23.4h\n"
      "fmlal2 v6.4s, v4.4h, v23.4h\n"
      "movi v11.16b, #0x0\n"
      "sub x24, x24, #0x4\n"
      "fmlal v8.4s, v7.4h, v23.4h\n"
      "fmlal2 v9.4s, v7.4h, v23.4h\n"
      "movi v12.16b, #0x0\n"
      "prfm pldl1keep, [x23, #0x80]\n"
      "fmlal v11.4s, v10.4h, v23.4h\n"
      "movi v14.16b, #0x0\n"
      "prfm pldl1keep, [x22, #0x80]\n"
      "prfm pldl1keep, [x21, #0x80]\n"
      "fmlal2 v12.4s, v10.4h, v23.4h\n"
      "movi v15.16b, #0x0\n"
      "prfm pldl1keep, [x20, #0x80]\n"
      "add x26, x26, #0x40\n"
      "fmla v24.4s, v5.4s, v0.s[0]\n"
      "fmla v26.4s, v5.4s, v1.s[0]\n"
      "fmla v28.4s, v5.4s, v2.s[0]\n"
      "fmla v30.4s, v5.4s, v3.s[0]\n"
      "fmla v25.4s, v6.4s, v0.s[0]\n"
      "fmla v27.4s, v6.4s, v1.s[0]\n"
      "fmla v29.4s, v6.4s, v2.s[0]\n"
      "fmla v31.4s, v6.4s, v3.s[0]\n"
      "fmla v24.4s, v8.4s, v0.s[1]\n"
      "fmla v26.4s, v8.4s, v1.s[1]\n"
      "fmla v28.4s, v8.4s, v2.s[1]\n"
      "fmla v30.4s, v8.4s, v3.s[1]\n"
      "fmla v25.4s, v9.4s, v0.s[1]\n"
      "fmla v27.4s, v9.4s, v1.s[1]\n"
      "fmla v29.4s, v9.4s, v2.s[1]\n"
      "fmla v31.4s, v9.4s, v3.s[1]\n"
      "fmlal v14.4s, v13.4h, v23.4h\n"
      "fmla v24.4s, v11.4s, v0.s[2]\n"
      "fmla v26.4s, v11.4s, v1.s[2]\n"
      "fmla v28.4s, v11.4s, v2.s[2]\n"
      "fmla v30.4s, v11.4s, v3.s[2]\n"
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
      "cbz x24, 9f\n"
      "8:" // Height 4: Multiply loop: Odd block loop
      "ldr q16, [x26, #0x0]\n"
      "ldr s0, [x23], #0x4\n"
      "movi v17.16b, #0x0\n"
      "movi v18.16b, #0x0\n"
      "ldr s1, [x22], #0x4\n"
      "ldr s2, [x21], #0x4\n"
      "sub x24, x24, #0x1\n"
      "add x26, x26, #0x10\n"
      "ldr s3, [x20], #0x4\n"
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
      "cbnz x24, 8b\n"
      "9:" // Height 4: Multiply loop: No odd multiplies
      "ldr x20, [%x[gp], %[offsetof_ldc]]\n"
      "prfm pstl1keep, [x25, #0x0]\n"
      "str q24, [x25, #0x0]\n"
      "str q25, [x25, #0x10]\n"
      "add x22, x25, x20\n"
      "add x25, x25, #0x20\n"
      "prfm pstl1keep, [x22, #0x0]\n"
      "str q26, [x22, #0x0]\n"
      "add x21, x22, x20\n"
      "add x20, x21, x20\n"
      "prfm pstl1keep, [x21, #0x0]\n"
      "prfm pstl1keep, [x20, #0x0]\n"
      "str q27, [x22, #0x10]\n"
      "str q28, [x21, #0x0]\n"
      "str q29, [x21, #0x10]\n"
      "str q30, [x20, #0x0]\n"
      "str q31, [x20, #0x10]\n"
      "subs x27, x27, #0x1\n"
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
        "v2",
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
        "x20",
        "x21",
        "x22",
        "x23",
        "x24",
        "x25",
        "x26",
        "x27",
        "x28");
#endif // __aarch64__
}

void NOINLINE gemmkernel_5x1_Neon_fp16_fA0fB0fC0(GemmParamsFP16* gp) {
#ifdef __aarch64__
  __asm__ __volatile__(
      "ldr w20, [%x[gp], %[offsetof_beta]]\n"
      "mov x9, #0x1\n"
      "fmov v21.8h, #1.0\n"
      "ldr x28, [%x[gp], %[offsetof_b_block_cols]]\n"
      "ldr x27, [%x[gp], %[offsetof_B]]\n"
      "ldr x26, [%x[gp], %[offsetof_C]]\n"
      "bic x20, x20, #0x80000000\n"
      "cmp x20, #0x0\n"
      "csel x9, XZR, x9, EQ\n"
      "1:" // Height 5: Column loop
      "tbz x9, #0, 2f\n"
      "ldr q22, [x26, #0x0]\n"
      "ldr q23, [x26, #0x10]\n"
      "add x20, %x[gp], %[offsetof_beta]\n"
      "ld1r { v16.4s }, [x20]\n"
      "ldr x21, [%x[gp], %[offsetof_ldc]]\n"
      "add x20, x26, x21\n"
      "ldr q24, [x20, #0x0]\n"
      "ldr q25, [x20, #0x10]\n"
      "add x20, x20, x21\n"
      "ldr q26, [x20, #0x0]\n"
      "ldr q27, [x20, #0x10]\n"
      "add x20, x20, x21\n"
      "ldr q28, [x20, #0x0]\n"
      "ldr q29, [x20, #0x10]\n"
      "add x20, x20, x21\n"
      "fmul v22.4s, v22.4s, v16.4s\n"
      "ldr q30, [x20, #0x0]\n"
      "ldr q31, [x20, #0x10]\n"
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
      "ldr x21, [%x[gp], %[offsetof_A]]\n"
      "ldr x20, [%x[gp], %[offsetof_lda]]\n"
      "ldr x25, [%x[gp], %[offsetof_k]]\n"
      "mov x24, x21\n"
      "add x23, x24, x20\n"
      "add x22, x23, x20\n"
      "add x21, x22, x20\n"
      "add x20, x21, x20\n"
      "cmp x25, #0x4\n"
      "blt 7f\n"
      "ldr q0, [x24, #0x0]\n"
      "ldr q5, [x27, #0x0]\n"
      "cmp x25, #0x8\n"
      "ldr q1, [x23, #0x0]\n"
      "ldr q2, [x22, #0x0]\n"
      "ldr q3, [x21, #0x0]\n"
      "ldr q4, [x20, #0x0]\n"
      "ldr q8, [x27, #0x10]\n"
      "ldr q11, [x27, #0x20]\n"
      "ldr q14, [x27, #0x30]\n"
      "blt 6f\n"
      "5:" // Height 5: Multiply loop: Main loop head
      "movi v6.16b, #0x0\n"
      "movi v7.16b, #0x0\n"
      "sub x25, x25, #0x4\n"
      "add x24, x24, #0x10\n"
      "movi v9.16b, #0x0\n"
      "movi v10.16b, #0x0\n"
      "add x23, x23, #0x10\n"
      "add x22, x22, #0x10\n"
      "fmlal v6.4s, v5.4h, v21.4h\n"
      "fmlal2 v7.4s, v5.4h, v21.4h\n"
      "movi v12.16b, #0x0\n"
      "add x21, x21, #0x10\n"
      "fmlal v9.4s, v8.4h, v21.4h\n"
      "fmlal2 v10.4s, v8.4h, v21.4h\n"
      "movi v13.16b, #0x0\n"
      "add x20, x20, #0x10\n"
      "fmlal v12.4s, v11.4h, v21.4h\n"
      "movi v15.16b, #0x0\n"
      "cmp x25, #0x8\n"
      "add x27, x27, #0x40\n"
      "ldr q5, [x27, #0x0]\n"
      "ldr q8, [x27, #0x10]\n"
      "fmlal2 v13.4s, v11.4h, v21.4h\n"
      "movi v16.16b, #0x0\n"
      "ldr q11, [x27, #0x20]\n"
      "fmla v22.4s, v6.4s, v0.s[0]\n"
      "fmla v24.4s, v6.4s, v1.s[0]\n"
      "prfm pldl1keep, [x24, #0x80]\n"
      "fmla v26.4s, v6.4s, v2.s[0]\n"
      "fmla v28.4s, v6.4s, v3.s[0]\n"
      "prfm pldl1keep, [x23, #0x80]\n"
      "prfm pldl1keep, [x22, #0x80]\n"
      "fmla v30.4s, v6.4s, v4.s[0]\n"
      "fmla v23.4s, v7.4s, v0.s[0]\n"
      "prfm pldl1keep, [x21, #0x80]\n"
      "prfm pldl1keep, [x20, #0x80]\n"
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
      "fmlal v15.4s, v14.4h, v21.4h\n"
      "fmlal2 v16.4s, v14.4h, v21.4h\n"
      "ldr q14, [x27, #0x30]\n"
      "fmla v22.4s, v15.4s, v0.s[3]\n"
      "fmla v24.4s, v15.4s, v1.s[3]\n"
      "fmla v26.4s, v15.4s, v2.s[3]\n"
      "fmla v28.4s, v15.4s, v3.s[3]\n"
      "fmla v30.4s, v15.4s, v4.s[3]\n"
      "fmla v23.4s, v16.4s, v0.s[3]\n"
      "ldr q0, [x24, #0x0]\n"
      "fmla v25.4s, v16.4s, v1.s[3]\n"
      "ldr q1, [x23, #0x0]\n"
      "fmla v27.4s, v16.4s, v2.s[3]\n"
      "ldr q2, [x22, #0x0]\n"
      "fmla v29.4s, v16.4s, v3.s[3]\n"
      "ldr q3, [x21, #0x0]\n"
      "fmla v31.4s, v16.4s, v4.s[3]\n"
      "ldr q4, [x20, #0x0]\n"
      "bge 5b\n"
      "6:" // Height 5: Multiply loop: Single iteration only
      "movi v6.16b, #0x0\n"
      "movi v7.16b, #0x0\n"
      "add x24, x24, #0x10\n"
      "add x23, x23, #0x10\n"
      "movi v9.16b, #0x0\n"
      "movi v10.16b, #0x0\n"
      "add x22, x22, #0x10\n"
      "add x21, x21, #0x10\n"
      "fmlal v6.4s, v5.4h, v21.4h\n"
      "fmlal2 v7.4s, v5.4h, v21.4h\n"
      "movi v12.16b, #0x0\n"
      "add x20, x20, #0x10\n"
      "fmlal v9.4s, v8.4h, v21.4h\n"
      "fmlal2 v10.4s, v8.4h, v21.4h\n"
      "movi v13.16b, #0x0\n"
      "sub x25, x25, #0x4\n"
      "fmlal v12.4s, v11.4h, v21.4h\n"
      "movi v15.16b, #0x0\n"
      "prfm pldl1keep, [x24, #0x80]\n"
      "prfm pldl1keep, [x23, #0x80]\n"
      "fmlal2 v13.4s, v11.4h, v21.4h\n"
      "movi v16.16b, #0x0\n"
      "prfm pldl1keep, [x22, #0x80]\n"
      "prfm pldl1keep, [x21, #0x80]\n"
      "fmla v22.4s, v6.4s, v0.s[0]\n"
      "fmla v24.4s, v6.4s, v1.s[0]\n"
      "prfm pldl1keep, [x20, #0x80]\n"
      "add x27, x27, #0x40\n"
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
      "fmlal v15.4s, v14.4h, v21.4h\n"
      "fmlal2 v16.4s, v14.4h, v21.4h\n"
      "fmla v22.4s, v15.4s, v0.s[3]\n"
      "fmla v24.4s, v15.4s, v1.s[3]\n"
      "fmla v26.4s, v15.4s, v2.s[3]\n"
      "fmla v28.4s, v15.4s, v3.s[3]\n"
      "fmla v30.4s, v15.4s, v4.s[3]\n"
      "fmla v23.4s, v16.4s, v0.s[3]\n"
      "fmla v25.4s, v16.4s, v1.s[3]\n"
      "fmla v27.4s, v16.4s, v2.s[3]\n"
      "fmla v29.4s, v16.4s, v3.s[3]\n"
      "fmla v31.4s, v16.4s, v4.s[3]\n"
      "7:" // Height 5: Multiply loop: Main loop skip
      "cbz x25, 9f\n"
      "8:" // Height 5: Multiply loop: Odd block loop
      "ldr q17, [x27, #0x0]\n"
      "ldr s0, [x24], #0x4\n"
      "movi v18.16b, #0x0\n"
      "movi v19.16b, #0x0\n"
      "ldr s1, [x23], #0x4\n"
      "ldr s2, [x22], #0x4\n"
      "sub x25, x25, #0x1\n"
      "add x27, x27, #0x10\n"
      "ldr s3, [x21], #0x4\n"
      "ldr s4, [x20], #0x4\n"
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
      "cbnz x25, 8b\n"
      "9:" // Height 5: Multiply loop: No odd multiplies
      "ldr x20, [%x[gp], %[offsetof_ldc]]\n"
      "prfm pstl1keep, [x26, #0x0]\n"
      "str q22, [x26, #0x0]\n"
      "str q23, [x26, #0x10]\n"
      "add x23, x26, x20\n"
      "add x26, x26, #0x20\n"
      "prfm pstl1keep, [x23, #0x0]\n"
      "str q24, [x23, #0x0]\n"
      "add x22, x23, x20\n"
      "add x21, x22, x20\n"
      "add x20, x21, x20\n"
      "prfm pstl1keep, [x22, #0x0]\n"
      "prfm pstl1keep, [x21, #0x0]\n"
      "str q25, [x23, #0x10]\n"
      "prfm pstl1keep, [x20, #0x0]\n"
      "str q26, [x22, #0x0]\n"
      "str q27, [x22, #0x10]\n"
      "str q28, [x21, #0x0]\n"
      "str q29, [x21, #0x10]\n"
      "str q30, [x20, #0x0]\n"
      "str q31, [x20, #0x10]\n"
      "subs x28, x28, #0x1\n"
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
        "x20",
        "x21",
        "x22",
        "x23",
        "x24",
        "x25",
        "x26",
        "x27",
        "x28",
        "x9");
#endif // __aarch64__
}

void NOINLINE gemmkernel_6x1_Neon_fp16_fA0fB0fC0(GemmParamsFP16* gp) {
#ifdef __aarch64__
  __asm__ __volatile__(
      "ldr w20, [%x[gp], %[offsetof_beta]]\n"
      "mov x10, #0x1\n"
      "fmov v19.8h, #1.0\n"
      "ldr x9, [%x[gp], %[offsetof_b_block_cols]]\n"
      "ldr x28, [%x[gp], %[offsetof_B]]\n"
      "ldr x27, [%x[gp], %[offsetof_C]]\n"
      "bic x20, x20, #0x80000000\n"
      "cmp x20, #0x0\n"
      "csel x10, XZR, x10, EQ\n"
      "1:" // Height 6: Column loop
      "tbz x10, #0, 2f\n"
      "ldr q20, [x27, #0x0]\n"
      "ldr q21, [x27, #0x10]\n"
      "add x20, %x[gp], %[offsetof_beta]\n"
      "ld1r { v16.4s }, [x20]\n"
      "ldr x21, [%x[gp], %[offsetof_ldc]]\n"
      "add x20, x27, x21\n"
      "ldr q22, [x20, #0x0]\n"
      "ldr q23, [x20, #0x10]\n"
      "add x20, x20, x21\n"
      "ldr q24, [x20, #0x0]\n"
      "ldr q25, [x20, #0x10]\n"
      "add x20, x20, x21\n"
      "ldr q26, [x20, #0x0]\n"
      "ldr q27, [x20, #0x10]\n"
      "add x20, x20, x21\n"
      "fmul v20.4s, v20.4s, v16.4s\n"
      "ldr q28, [x20, #0x0]\n"
      "ldr q29, [x20, #0x10]\n"
      "add x20, x20, x21\n"
      "fmul v21.4s, v21.4s, v16.4s\n"
      "ldr q30, [x20, #0x0]\n"
      "ldr q31, [x20, #0x10]\n"
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
      "ldr x21, [%x[gp], %[offsetof_A]]\n"
      "ldr x20, [%x[gp], %[offsetof_lda]]\n"
      "ldr x26, [%x[gp], %[offsetof_k]]\n"
      "mov x25, x21\n"
      "add x24, x25, x20\n"
      "add x23, x24, x20\n"
      "add x22, x23, x20\n"
      "add x21, x22, x20\n"
      "add x20, x21, x20\n"
      "cmp x26, #0x4\n"
      "blt 7f\n"
      "ldr q0, [x25, #0x0]\n"
      "ldr q6, [x28, #0x0]\n"
      "cmp x26, #0x8\n"
      "ldr q1, [x24, #0x0]\n"
      "ldr q2, [x23, #0x0]\n"
      "ldr q3, [x22, #0x0]\n"
      "ldr q4, [x21, #0x0]\n"
      "ldr q5, [x20, #0x0]\n"
      "ldr q9, [x28, #0x10]\n"
      "ldr q12, [x28, #0x20]\n"
      "ldr q15, [x28, #0x30]\n"
      "blt 6f\n"
      "5:" // Height 6: Multiply loop: Main loop head
      "movi v7.16b, #0x0\n"
      "movi v8.16b, #0x0\n"
      "sub x26, x26, #0x4\n"
      "add x25, x25, #0x10\n"
      "movi v10.16b, #0x0\n"
      "movi v11.16b, #0x0\n"
      "add x24, x24, #0x10\n"
      "add x23, x23, #0x10\n"
      "fmlal v7.4s, v6.4h, v19.4h\n"
      "fmlal2 v8.4s, v6.4h, v19.4h\n"
      "movi v13.16b, #0x0\n"
      "add x22, x22, #0x10\n"
      "fmlal v10.4s, v9.4h, v19.4h\n"
      "fmlal2 v11.4s, v9.4h, v19.4h\n"
      "movi v14.16b, #0x0\n"
      "add x21, x21, #0x10\n"
      "fmlal v13.4s, v12.4h, v19.4h\n"
      "movi v16.16b, #0x0\n"
      "add x20, x20, #0x10\n"
      "cmp x26, #0x8\n"
      "fmlal2 v14.4s, v12.4h, v19.4h\n"
      "movi v17.16b, #0x0\n"
      "add x28, x28, #0x40\n"
      "prfm pldl1keep, [x25, #0x80]\n"
      "ldr q6, [x28, #0x0]\n"
      "ldr q9, [x28, #0x10]\n"
      "fmla v20.4s, v7.4s, v0.s[0]\n"
      "fmla v22.4s, v7.4s, v1.s[0]\n"
      "ldr q12, [x28, #0x20]\n"
      "fmla v24.4s, v7.4s, v2.s[0]\n"
      "fmla v26.4s, v7.4s, v3.s[0]\n"
      "prfm pldl1keep, [x24, #0x80]\n"
      "fmla v28.4s, v7.4s, v4.s[0]\n"
      "fmla v30.4s, v7.4s, v5.s[0]\n"
      "prfm pldl1keep, [x23, #0x80]\n"
      "prfm pldl1keep, [x22, #0x80]\n"
      "fmla v21.4s, v8.4s, v0.s[0]\n"
      "fmla v23.4s, v8.4s, v1.s[0]\n"
      "prfm pldl1keep, [x21, #0x80]\n"
      "prfm pldl1keep, [x20, #0x80]\n"
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
      "ldr q15, [x28, #0x30]\n"
      "fmla v20.4s, v16.4s, v0.s[3]\n"
      "fmla v22.4s, v16.4s, v1.s[3]\n"
      "fmla v24.4s, v16.4s, v2.s[3]\n"
      "fmla v26.4s, v16.4s, v3.s[3]\n"
      "fmla v28.4s, v16.4s, v4.s[3]\n"
      "fmla v30.4s, v16.4s, v5.s[3]\n"
      "fmla v21.4s, v17.4s, v0.s[3]\n"
      "ldr q0, [x25, #0x0]\n"
      "fmla v23.4s, v17.4s, v1.s[3]\n"
      "ldr q1, [x24, #0x0]\n"
      "fmla v25.4s, v17.4s, v2.s[3]\n"
      "ldr q2, [x23, #0x0]\n"
      "fmla v27.4s, v17.4s, v3.s[3]\n"
      "ldr q3, [x22, #0x0]\n"
      "fmla v29.4s, v17.4s, v4.s[3]\n"
      "ldr q4, [x21, #0x0]\n"
      "fmla v31.4s, v17.4s, v5.s[3]\n"
      "ldr q5, [x20, #0x0]\n"
      "bge 5b\n"
      "6:" // Height 6: Multiply loop: Single iteration only
      "movi v7.16b, #0x0\n"
      "movi v8.16b, #0x0\n"
      "add x25, x25, #0x10\n"
      "add x24, x24, #0x10\n"
      "movi v10.16b, #0x0\n"
      "movi v11.16b, #0x0\n"
      "add x23, x23, #0x10\n"
      "add x22, x22, #0x10\n"
      "fmlal v7.4s, v6.4h, v19.4h\n"
      "fmlal2 v8.4s, v6.4h, v19.4h\n"
      "movi v13.16b, #0x0\n"
      "add x21, x21, #0x10\n"
      "fmlal v10.4s, v9.4h, v19.4h\n"
      "fmlal2 v11.4s, v9.4h, v19.4h\n"
      "movi v14.16b, #0x0\n"
      "add x20, x20, #0x10\n"
      "fmlal v13.4s, v12.4h, v19.4h\n"
      "movi v16.16b, #0x0\n"
      "prfm pldl1keep, [x25, #0x80]\n"
      "sub x26, x26, #0x4\n"
      "fmlal2 v14.4s, v12.4h, v19.4h\n"
      "movi v17.16b, #0x0\n"
      "prfm pldl1keep, [x24, #0x80]\n"
      "prfm pldl1keep, [x23, #0x80]\n"
      "fmla v20.4s, v7.4s, v0.s[0]\n"
      "fmla v22.4s, v7.4s, v1.s[0]\n"
      "prfm pldl1keep, [x22, #0x80]\n"
      "prfm pldl1keep, [x21, #0x80]\n"
      "fmla v24.4s, v7.4s, v2.s[0]\n"
      "fmla v26.4s, v7.4s, v3.s[0]\n"
      "prfm pldl1keep, [x20, #0x80]\n"
      "add x28, x28, #0x40\n"
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
      "cbz x26, 9f\n"
      "8:" // Height 6: Multiply loop: Odd block loop
      "ldr q18, [x28, #0x0]\n"
      "ldr s0, [x25], #0x4\n"
      "movi v6.16b, #0x0\n"
      "movi v7.16b, #0x0\n"
      "ldr s1, [x24], #0x4\n"
      "ldr s2, [x23], #0x4\n"
      "sub x26, x26, #0x1\n"
      "add x28, x28, #0x10\n"
      "ldr s3, [x22], #0x4\n"
      "ldr s4, [x21], #0x4\n"
      "ldr s5, [x20], #0x4\n"
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
      "cbnz x26, 8b\n"
      "9:" // Height 6: Multiply loop: No odd multiplies
      "ldr x24, [%x[gp], %[offsetof_ldc]]\n"
      "prfm pstl1keep, [x27, #0x0]\n"
      "str q20, [x27, #0x0]\n"
      "str q21, [x27, #0x10]\n"
      "add x20, x27, x24\n"
      "add x27, x27, #0x20\n"
      "prfm pstl1keep, [x20, #0x0]\n"
      "str q22, [x20, #0x0]\n"
      "add x23, x20, x24\n"
      "add x22, x23, x24\n"
      "add x21, x22, x24\n"
      "prfm pstl1keep, [x23, #0x0]\n"
      "prfm pstl1keep, [x22, #0x0]\n"
      "str q23, [x20, #0x10]\n"
      "add x20, x21, x24\n"
      "prfm pstl1keep, [x21, #0x0]\n"
      "str q24, [x23, #0x0]\n"
      "prfm pstl1keep, [x20, #0x0]\n"
      "str q25, [x23, #0x10]\n"
      "str q26, [x22, #0x0]\n"
      "str q27, [x22, #0x10]\n"
      "str q28, [x21, #0x0]\n"
      "str q29, [x21, #0x10]\n"
      "str q30, [x20, #0x0]\n"
      "str q31, [x20, #0x10]\n"
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
        "x20",
        "x21",
        "x22",
        "x23",
        "x24",
        "x25",
        "x26",
        "x27",
        "x28",
        "x9");
#endif // __aarch64__
}

void NOINLINE gemmkernel_7x1_Neon_fp16_fA0fB0fC0(GemmParamsFP16* gp) {
#ifdef __aarch64__
  __asm__ __volatile__(
      "ldr w20, [%x[gp], %[offsetof_beta]]\n"
      "mov x11, #0x1\n"
      "fmov v17.8h, #1.0\n"
      "ldr x10, [%x[gp], %[offsetof_b_block_cols]]\n"
      "ldr x9, [%x[gp], %[offsetof_B]]\n"
      "ldr x28, [%x[gp], %[offsetof_C]]\n"
      "bic x20, x20, #0x80000000\n"
      "cmp x20, #0x0\n"
      "csel x11, XZR, x11, EQ\n"
      "1:" // Height 7: Column loop
      "tbz x11, #0, 2f\n"
      "ldr q18, [x28, #0x0]\n"
      "ldr q19, [x28, #0x10]\n"
      "add x20, %x[gp], %[offsetof_beta]\n"
      "ld1r { v16.4s }, [x20]\n"
      "ldr x21, [%x[gp], %[offsetof_ldc]]\n"
      "add x20, x28, x21\n"
      "ldr q20, [x20, #0x0]\n"
      "ldr q21, [x20, #0x10]\n"
      "add x20, x20, x21\n"
      "ldr q22, [x20, #0x0]\n"
      "ldr q23, [x20, #0x10]\n"
      "add x20, x20, x21\n"
      "ldr q24, [x20, #0x0]\n"
      "ldr q25, [x20, #0x10]\n"
      "add x20, x20, x21\n"
      "fmul v18.4s, v18.4s, v16.4s\n"
      "ldr q26, [x20, #0x0]\n"
      "ldr q27, [x20, #0x10]\n"
      "add x20, x20, x21\n"
      "fmul v19.4s, v19.4s, v16.4s\n"
      "ldr q28, [x20, #0x0]\n"
      "ldr q29, [x20, #0x10]\n"
      "add x20, x20, x21\n"
      "fmul v20.4s, v20.4s, v16.4s\n"
      "ldr q30, [x20, #0x0]\n"
      "ldr q31, [x20, #0x10]\n"
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
      "ldr x21, [%x[gp], %[offsetof_A]]\n"
      "ldr x20, [%x[gp], %[offsetof_lda]]\n"
      "ldr x27, [%x[gp], %[offsetof_k]]\n"
      "mov x26, x21\n"
      "add x25, x26, x20\n"
      "add x24, x25, x20\n"
      "add x23, x24, x20\n"
      "add x22, x23, x20\n"
      "add x21, x22, x20\n"
      "add x20, x21, x20\n"
      "cmp x27, #0x4\n"
      "blt 7f\n"
      "ldr q0, [x26, #0x0]\n"
      "ldr q7, [x9, #0x0]\n"
      "cmp x27, #0x8\n"
      "ldr q1, [x25, #0x0]\n"
      "ldr q2, [x24, #0x0]\n"
      "ldr q3, [x23, #0x0]\n"
      "ldr q4, [x22, #0x0]\n"
      "ldr q5, [x21, #0x0]\n"
      "ldr q6, [x20, #0x0]\n"
      "ldr q10, [x9, #0x10]\n"
      "ldr q13, [x9, #0x20]\n"
      "ldr q16, [x9, #0x30]\n"
      "blt 6f\n"
      "5:" // Height 7: Multiply loop: Main loop head
      "movi v8.16b, #0x0\n"
      "movi v9.16b, #0x0\n"
      "sub x27, x27, #0x4\n"
      "add x26, x26, #0x10\n"
      "movi v11.16b, #0x0\n"
      "movi v12.16b, #0x0\n"
      "add x25, x25, #0x10\n"
      "add x24, x24, #0x10\n"
      "fmlal v8.4s, v7.4h, v17.4h\n"
      "fmlal2 v9.4s, v7.4h, v17.4h\n"
      "movi v14.16b, #0x0\n"
      "add x23, x23, #0x10\n"
      "fmlal v11.4s, v10.4h, v17.4h\n"
      "fmlal2 v12.4s, v10.4h, v17.4h\n"
      "movi v15.16b, #0x0\n"
      "add x22, x22, #0x10\n"
      "fmlal v14.4s, v13.4h, v17.4h\n"
      "movi v7.16b, #0x0\n"
      "add x21, x21, #0x10\n"
      "add x20, x20, #0x10\n"
      "fmlal2 v15.4s, v13.4h, v17.4h\n"
      "cmp x27, #0x8\n"
      "add x9, x9, #0x40\n"
      "prfm pldl1keep, [x26, #0x80]\n"
      "ldr q10, [x9, #0x10]\n"
      "ldr q13, [x9, #0x20]\n"
      "fmla v18.4s, v8.4s, v0.s[0]\n"
      "fmla v20.4s, v8.4s, v1.s[0]\n"
      "fmla v22.4s, v8.4s, v2.s[0]\n"
      "fmla v24.4s, v8.4s, v3.s[0]\n"
      "prfm pldl1keep, [x25, #0x80]\n"
      "prfm pldl1keep, [x24, #0x80]\n"
      "fmla v26.4s, v8.4s, v4.s[0]\n"
      "fmla v28.4s, v8.4s, v5.s[0]\n"
      "prfm pldl1keep, [x23, #0x80]\n"
      "prfm pldl1keep, [x22, #0x80]\n"
      "fmla v30.4s, v8.4s, v6.s[0]\n"
      "fmla v19.4s, v9.4s, v0.s[0]\n"
      "movi v8.16b, #0x0\n"
      "prfm pldl1keep, [x21, #0x80]\n"
      "fmla v21.4s, v9.4s, v1.s[0]\n"
      "fmla v23.4s, v9.4s, v2.s[0]\n"
      "prfm pldl1keep, [x20, #0x80]\n"
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
      "ldr q16, [x9, #0x30]\n"
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
      "ldr q7, [x9, #0x0]\n"
      "fmla v19.4s, v8.4s, v0.s[3]\n"
      "ldr q0, [x26, #0x0]\n"
      "fmla v21.4s, v8.4s, v1.s[3]\n"
      "ldr q1, [x25, #0x0]\n"
      "fmla v23.4s, v8.4s, v2.s[3]\n"
      "ldr q2, [x24, #0x0]\n"
      "fmla v25.4s, v8.4s, v3.s[3]\n"
      "ldr q3, [x23, #0x0]\n"
      "fmla v27.4s, v8.4s, v4.s[3]\n"
      "ldr q4, [x22, #0x0]\n"
      "fmla v29.4s, v8.4s, v5.s[3]\n"
      "ldr q5, [x21, #0x0]\n"
      "fmla v31.4s, v8.4s, v6.s[3]\n"
      "ldr q6, [x20, #0x0]\n"
      "bge 5b\n"
      "6:" // Height 7: Multiply loop: Single iteration only
      "movi v8.16b, #0x0\n"
      "movi v9.16b, #0x0\n"
      "add x26, x26, #0x10\n"
      "add x25, x25, #0x10\n"
      "movi v11.16b, #0x0\n"
      "movi v12.16b, #0x0\n"
      "add x24, x24, #0x10\n"
      "add x23, x23, #0x10\n"
      "fmlal v8.4s, v7.4h, v17.4h\n"
      "fmlal2 v9.4s, v7.4h, v17.4h\n"
      "movi v14.16b, #0x0\n"
      "add x22, x22, #0x10\n"
      "fmlal v11.4s, v10.4h, v17.4h\n"
      "fmlal2 v12.4s, v10.4h, v17.4h\n"
      "movi v15.16b, #0x0\n"
      "add x21, x21, #0x10\n"
      "fmlal v14.4s, v13.4h, v17.4h\n"
      "movi v7.16b, #0x0\n"
      "add x20, x20, #0x10\n"
      "prfm pldl1keep, [x26, #0x80]\n"
      "fmlal2 v15.4s, v13.4h, v17.4h\n"
      "prfm pldl1keep, [x25, #0x80]\n"
      "sub x27, x27, #0x4\n"
      "prfm pldl1keep, [x24, #0x80]\n"
      "fmla v18.4s, v8.4s, v0.s[0]\n"
      "fmla v20.4s, v8.4s, v1.s[0]\n"
      "prfm pldl1keep, [x23, #0x80]\n"
      "prfm pldl1keep, [x22, #0x80]\n"
      "fmla v22.4s, v8.4s, v2.s[0]\n"
      "fmla v24.4s, v8.4s, v3.s[0]\n"
      "prfm pldl1keep, [x21, #0x80]\n"
      "prfm pldl1keep, [x20, #0x80]\n"
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
      "cbz x27, 9f\n"
      "8:" // Height 7: Multiply loop: Odd block loop
      "ldr q9, [x9, #0x0]\n"
      "ldr s0, [x26], #0x4\n"
      "movi v10.16b, #0x0\n"
      "movi v11.16b, #0x0\n"
      "ldr s1, [x25], #0x4\n"
      "ldr s2, [x24], #0x4\n"
      "sub x27, x27, #0x1\n"
      "add x9, x9, #0x10\n"
      "ldr s3, [x23], #0x4\n"
      "ldr s4, [x22], #0x4\n"
      "ldr s5, [x21], #0x4\n"
      "ldr s6, [x20], #0x4\n"
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
      "cbnz x27, 8b\n"
      "9:" // Height 7: Multiply loop: No odd multiplies
      "ldr x25, [%x[gp], %[offsetof_ldc]]\n"
      "prfm pstl1keep, [x28, #0x0]\n"
      "str q18, [x28, #0x0]\n"
      "str q19, [x28, #0x10]\n"
      "add x20, x28, x25\n"
      "add x28, x28, #0x20\n"
      "prfm pstl1keep, [x20, #0x0]\n"
      "str q20, [x20, #0x0]\n"
      "add x24, x20, x25\n"
      "add x23, x24, x25\n"
      "add x22, x23, x25\n"
      "prfm pstl1keep, [x24, #0x0]\n"
      "prfm pstl1keep, [x23, #0x0]\n"
      "str q21, [x20, #0x10]\n"
      "add x21, x22, x25\n"
      "prfm pstl1keep, [x22, #0x0]\n"
      "str q22, [x24, #0x0]\n"
      "add x20, x21, x25\n"
      "prfm pstl1keep, [x21, #0x0]\n"
      "prfm pstl1keep, [x20, #0x0]\n"
      "str q23, [x24, #0x10]\n"
      "str q24, [x23, #0x0]\n"
      "str q25, [x23, #0x10]\n"
      "str q26, [x22, #0x0]\n"
      "str q27, [x22, #0x10]\n"
      "str q28, [x21, #0x0]\n"
      "str q29, [x21, #0x10]\n"
      "str q30, [x20, #0x0]\n"
      "str q31, [x20, #0x10]\n"
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
        "x20",
        "x21",
        "x22",
        "x23",
        "x24",
        "x25",
        "x26",
        "x27",
        "x28",
        "x9");
#endif // __aarch64__
}

void NOINLINE gemmkernel_8x1_Neon_fp16_fA0fB0fC0(GemmParamsFP16* gp) {
#ifdef __aarch64__
  __asm__ __volatile__(
      "ldr w20, [%x[gp], %[offsetof_beta]]\n"
      "mov x12, #0x1\n"
      "fmov v15.8h, #1.0\n"
      "ldr x11, [%x[gp], %[offsetof_b_block_cols]]\n"
      "ldr x10, [%x[gp], %[offsetof_B]]\n"
      "ldr x9, [%x[gp], %[offsetof_C]]\n"
      "bic x20, x20, #0x80000000\n"
      "cmp x20, #0x0\n"
      "csel x12, XZR, x12, EQ\n"
      "1:" // Height 8: Column loop
      "tbz x12, #0, 2f\n"
      "ldr q16, [x9, #0x0]\n"
      "ldr q17, [x9, #0x10]\n"
      "add x20, %x[gp], %[offsetof_beta]\n"
      "ld1r { v0.4s }, [x20]\n"
      "ldr x21, [%x[gp], %[offsetof_ldc]]\n"
      "add x20, x9, x21\n"
      "ldr q18, [x20, #0x0]\n"
      "ldr q19, [x20, #0x10]\n"
      "add x20, x20, x21\n"
      "ldr q20, [x20, #0x0]\n"
      "ldr q21, [x20, #0x10]\n"
      "add x20, x20, x21\n"
      "ldr q22, [x20, #0x0]\n"
      "ldr q23, [x20, #0x10]\n"
      "add x20, x20, x21\n"
      "fmul v16.4s, v16.4s, v0.4s\n"
      "ldr q24, [x20, #0x0]\n"
      "ldr q25, [x20, #0x10]\n"
      "add x20, x20, x21\n"
      "fmul v17.4s, v17.4s, v0.4s\n"
      "ldr q26, [x20, #0x0]\n"
      "ldr q27, [x20, #0x10]\n"
      "add x20, x20, x21\n"
      "fmul v18.4s, v18.4s, v0.4s\n"
      "ldr q28, [x20, #0x0]\n"
      "ldr q29, [x20, #0x10]\n"
      "add x20, x20, x21\n"
      "fmul v19.4s, v19.4s, v0.4s\n"
      "ldr q30, [x20, #0x0]\n"
      "ldr q31, [x20, #0x10]\n"
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
      "ldr x21, [%x[gp], %[offsetof_A]]\n"
      "ldr x20, [%x[gp], %[offsetof_lda]]\n"
      "ldr x28, [%x[gp], %[offsetof_k]]\n"
      "mov x27, x21\n"
      "add x26, x27, x20\n"
      "add x25, x26, x20\n"
      "add x24, x25, x20\n"
      "add x23, x24, x20\n"
      "add x22, x23, x20\n"
      "add x21, x22, x20\n"
      "add x20, x21, x20\n"
      "cmp x28, #0x4\n"
      "blt 7f\n"
      "ldr q0, [x27, #0x0]\n"
      "ldr q8, [x10, #0x0]\n"
      "cmp x28, #0x8\n"
      "ldr q1, [x26, #0x0]\n"
      "ldr q2, [x25, #0x0]\n"
      "ldr q3, [x24, #0x0]\n"
      "ldr q4, [x23, #0x0]\n"
      "ldr q5, [x22, #0x0]\n"
      "ldr q6, [x21, #0x0]\n"
      "ldr q7, [x20, #0x0]\n"
      "ldr q11, [x10, #0x10]\n"
      "ldr q14, [x10, #0x20]\n"
      "blt 6f\n"
      "5:" // Height 8: Multiply loop: Main loop head
      "movi v9.16b, #0x0\n"
      "movi v10.16b, #0x0\n"
      "sub x28, x28, #0x4\n"
      "add x27, x27, #0x10\n"
      "movi v12.16b, #0x0\n"
      "movi v13.16b, #0x0\n"
      "add x26, x26, #0x10\n"
      "add x25, x25, #0x10\n"
      "fmlal v9.4s, v8.4h, v15.4h\n"
      "fmlal2 v10.4s, v8.4h, v15.4h\n"
      "movi v8.16b, #0x0\n"
      "add x24, x24, #0x10\n"
      "fmlal v12.4s, v11.4h, v15.4h\n"
      "fmlal2 v13.4s, v11.4h, v15.4h\n"
      "movi v11.16b, #0x0\n"
      "add x23, x23, #0x10\n"
      "fmlal v8.4s, v14.4h, v15.4h\n"
      "add x22, x22, #0x10\n"
      "add x21, x21, #0x10\n"
      "prfm pldl1keep, [x27, #0x80]\n"
      "add x20, x20, #0x10\n"
      "cmp x28, #0x8\n"
      "prfm pldl1keep, [x26, #0x80]\n"
      "prfm pldl1keep, [x25, #0x80]\n"
      "fmla v16.4s, v9.4s, v0.s[0]\n"
      "fmla v18.4s, v9.4s, v1.s[0]\n"
      "prfm pldl1keep, [x24, #0x80]\n"
      "prfm pldl1keep, [x23, #0x80]\n"
      "fmla v20.4s, v9.4s, v2.s[0]\n"
      "fmla v22.4s, v9.4s, v3.s[0]\n"
      "prfm pldl1keep, [x22, #0x80]\n"
      "prfm pldl1keep, [x21, #0x80]\n"
      "fmla v24.4s, v9.4s, v4.s[0]\n"
      "fmla v26.4s, v9.4s, v5.s[0]\n"
      "prfm pldl1keep, [x20, #0x80]\n"
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
      "add x10, x10, #0x40\n"
      "fmlal2 v9.4s, v14.4h, v15.4h\n"
      "ldr q14, [x10, #0x20]\n"
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
      "ldr q8, [x10, #0x0]\n"
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
      "ldr q11, [x10, #0x10]\n"
      "fmla v17.4s, v12.4s, v0.s[3]\n"
      "ldr q0, [x27, #0x0]\n"
      "fmla v19.4s, v12.4s, v1.s[3]\n"
      "ldr q1, [x26, #0x0]\n"
      "fmla v21.4s, v12.4s, v2.s[3]\n"
      "ldr q2, [x25, #0x0]\n"
      "fmla v23.4s, v12.4s, v3.s[3]\n"
      "ldr q3, [x24, #0x0]\n"
      "fmla v25.4s, v12.4s, v4.s[3]\n"
      "ldr q4, [x23, #0x0]\n"
      "fmla v27.4s, v12.4s, v5.s[3]\n"
      "ldr q5, [x22, #0x0]\n"
      "fmla v29.4s, v12.4s, v6.s[3]\n"
      "ldr q6, [x21, #0x0]\n"
      "fmla v31.4s, v12.4s, v7.s[3]\n"
      "ldr q7, [x20, #0x0]\n"
      "bge 5b\n"
      "6:" // Height 8: Multiply loop: Single iteration only
      "movi v9.16b, #0x0\n"
      "movi v10.16b, #0x0\n"
      "add x27, x27, #0x10\n"
      "add x26, x26, #0x10\n"
      "movi v12.16b, #0x0\n"
      "movi v13.16b, #0x0\n"
      "add x25, x25, #0x10\n"
      "add x24, x24, #0x10\n"
      "fmlal v9.4s, v8.4h, v15.4h\n"
      "fmlal2 v10.4s, v8.4h, v15.4h\n"
      "movi v8.16b, #0x0\n"
      "add x23, x23, #0x10\n"
      "fmlal v12.4s, v11.4h, v15.4h\n"
      "fmlal2 v13.4s, v11.4h, v15.4h\n"
      "movi v11.16b, #0x0\n"
      "add x22, x22, #0x10\n"
      "fmlal v8.4s, v14.4h, v15.4h\n"
      "add x21, x21, #0x10\n"
      "add x20, x20, #0x10\n"
      "prfm pldl1keep, [x27, #0x80]\n"
      "prfm pldl1keep, [x26, #0x80]\n"
      "prfm pldl1keep, [x25, #0x80]\n"
      "sub x28, x28, #0x4\n"
      "fmla v16.4s, v9.4s, v0.s[0]\n"
      "fmla v18.4s, v9.4s, v1.s[0]\n"
      "fmla v20.4s, v9.4s, v2.s[0]\n"
      "prfm pldl1keep, [x24, #0x80]\n"
      "prfm pldl1keep, [x23, #0x80]\n"
      "fmla v22.4s, v9.4s, v3.s[0]\n"
      "fmla v24.4s, v9.4s, v4.s[0]\n"
      "prfm pldl1keep, [x22, #0x80]\n"
      "prfm pldl1keep, [x21, #0x80]\n"
      "fmla v26.4s, v9.4s, v5.s[0]\n"
      "fmla v28.4s, v9.4s, v6.s[0]\n"
      "prfm pldl1keep, [x20, #0x80]\n"
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
      "cbz x28, 9f\n"
      "8:" // Height 8: Multiply loop: Odd block loop
      "ldr q13, [x10, #0x0]\n"
      "ldr s0, [x27], #0x4\n"
      "movi v14.16b, #0x0\n"
      "movi v8.16b, #0x0\n"
      "ldr s1, [x26], #0x4\n"
      "ldr s2, [x25], #0x4\n"
      "sub x28, x28, #0x1\n"
      "add x10, x10, #0x10\n"
      "ldr s3, [x24], #0x4\n"
      "ldr s4, [x23], #0x4\n"
      "ldr s5, [x22], #0x4\n"
      "ldr s6, [x21], #0x4\n"
      "fmlal v14.4s, v13.4h, v15.4h\n"
      "fmlal2 v8.4s, v13.4h, v15.4h\n"
      "ldr s7, [x20], #0x4\n"
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
      "cbnz x28, 8b\n"
      "9:" // Height 8: Multiply loop: No odd multiplies
      "ldr x26, [%x[gp], %[offsetof_ldc]]\n"
      "prfm pstl1keep, [x9, #0x0]\n"
      "str q16, [x9, #0x0]\n"
      "str q17, [x9, #0x10]\n"
      "add x20, x9, x26\n"
      "add x9, x9, #0x20\n"
      "prfm pstl1keep, [x20, #0x0]\n"
      "str q18, [x20, #0x0]\n"
      "add x25, x20, x26\n"
      "add x24, x25, x26\n"
      "add x23, x24, x26\n"
      "prfm pstl1keep, [x25, #0x0]\n"
      "prfm pstl1keep, [x24, #0x0]\n"
      "str q19, [x20, #0x10]\n"
      "add x22, x23, x26\n"
      "prfm pstl1keep, [x23, #0x0]\n"
      "str q20, [x25, #0x0]\n"
      "add x21, x22, x26\n"
      "add x20, x21, x26\n"
      "prfm pstl1keep, [x22, #0x0]\n"
      "prfm pstl1keep, [x21, #0x0]\n"
      "str q21, [x25, #0x10]\n"
      "prfm pstl1keep, [x20, #0x0]\n"
      "str q22, [x24, #0x0]\n"
      "str q23, [x24, #0x10]\n"
      "str q24, [x23, #0x0]\n"
      "str q25, [x23, #0x10]\n"
      "str q26, [x22, #0x0]\n"
      "str q27, [x22, #0x10]\n"
      "str q28, [x21, #0x0]\n"
      "str q29, [x21, #0x10]\n"
      "str q30, [x20, #0x0]\n"
      "str q31, [x20, #0x10]\n"
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
        "x20",
        "x21",
        "x22",
        "x23",
        "x24",
        "x25",
        "x26",
        "x27",
        "x28",
        "x9");
#endif // __aarch64__
}

} // namespace kleidiai

#endif
