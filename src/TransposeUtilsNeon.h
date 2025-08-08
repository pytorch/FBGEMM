/*
 * SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliate
 * <open-source-office@arm.com> SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#ifdef __aarch64__

#include <arm_neon.h>

namespace fbgemm {

namespace internal {

static inline void transpose_kernel_8x8_neon(
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
      "ldp q16, q17, [x0]\t\n"
      "add x0, x0, x1\t\n"
      "ldp q18, q19, [x0]\t\n"
      "add x0, x0, x1\t\n"
      "ldp q20, q21, [x0]\t\n"
      "add x0, x0, x1\t\n"
      "ldp q22, q23, [x0]\t\n"

      "zip1 v24.4s, v0.4s, v2.4s\t\n"
      "zip1 v25.4s, v4.4s, v6.4s\t\n"
      "zip1 v26.4s, v16.4s, v18.4s\t\n"
      "zip1 v27.4s, v20.4s, v22.4s\t\n"

      "zip1 v28.2d, v24.2d, v25.2d\t\n"
      "zip1 v29.2d, v26.2d, v27.2d\t\n"
      "lsl x3, x3, #2\t\n"
      "stp q28, q29, [x2]\t\n"

      "zip2 v30.2d, v24.2d, v25.2d\t\n"
      "zip2 v31.2d, v26.2d, v27.2d\t\n"
      "add x2, x2, x3\t\n"
      "stp q30, q31, [x2]\t\n"

      "zip2 v24.4s, v0.4s, v2.4s\t\n"
      "zip2 v25.4s, v4.4s, v6.4s\t\n"
      "zip2 v26.4s, v16.4s, v18.4s\t\n"
      "zip2 v27.4s, v20.4s, v22.4s\t\n"

      "zip1 v28.2d, v24.2d, v25.2d\t\n"
      "zip1 v29.2d, v26.2d, v27.2d\t\n"
      "add x2, x2, x3\t\n"
      "stp q28, q29, [x2]\t\n"

      "zip2 v30.2d, v24.2d, v25.2d\t\n"
      "zip2 v31.2d, v26.2d, v27.2d\t\n"
      "add x2, x2, x3\t\n"
      "stp q30, q31, [x2]\t\n"

      "zip1 v24.4s, v1.4s, v3.4s\t\n"
      "zip1 v25.4s, v5.4s, v7.4s\t\n"
      "zip1 v26.4s, v17.4s, v19.4s\t\n"
      "zip1 v27.4s, v21.4s, v23.4s\t\n"

      "zip1 v28.2d, v24.2d, v25.2d\t\n"
      "zip1 v29.2d, v26.2d, v27.2d\t\n"
      "add x2, x2, x3\t\n"
      "stp q28, q29, [x2]\t\n"

      "zip2 v30.2d, v24.2d, v25.2d\t\n"
      "zip2 v31.2d, v26.2d, v27.2d\t\n"
      "add x2, x2, x3\t\n"
      "stp q30, q31, [x2]\t\n"

      "zip2 v24.4s, v1.4s, v3.4s\t\n"
      "zip2 v25.4s, v5.4s, v7.4s\t\n"
      "zip2 v26.4s, v17.4s, v19.4s\t\n"
      "zip2 v27.4s, v21.4s, v23.4s\t\n"

      "zip1 v28.2d, v24.2d, v25.2d\t\n"
      "zip1 v29.2d, v26.2d, v27.2d\t\n"
      "add x2, x2, x3\t\n"
      "stp q28, q29, [x2]\t\n"

      "zip2 v30.2d, v24.2d, v25.2d\t\n"
      "zip2 v31.2d, v26.2d, v27.2d\t\n"
      "add x2, x2, x3\t\n"
      "stp q30, q31, [x2]\t\n"

      :
      :
      [src] "r"(src), [ld_src] "r"(ld_src), [dst] "r"(dst), [ld_dst] "r"(ld_dst)
      : "memory",
        "cc",
        "x0",
        "x1",
        "x2",
        "x3",
        "v0",
        "v1",
        "v2",
        "v3",
        "v4",
        "v5",
        "v6",
        "v7",
        "v16",
        "v17",
        "v18",
        "v19",
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
        "v30",
        "v31");
}

static inline void transpose_kernel_8x4_neon(
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
      "add x0, x0, x1\t\n"
      "ldr q4, [x0]\t\n"
      "add x0, x0, x1\t\n"
      "ldr q5, [x0]\t\n"
      "add x0, x0, x1\t\n"
      "ldr q6, [x0]\t\n"
      "add x0, x0, x1\t\n"
      "ldr q7, [x0]\t\n"

      "zip1 v16.4s, v0.4s, v1.4s\t\n"
      "zip1 v17.4s, v2.4s, v3.4s\t\n"
      "zip1 v18.4s, v4.4s, v5.4s\t\n"
      "zip1 v19.4s, v6.4s, v7.4s\t\n"

      "zip1 v20.2d, v16.2d, v17.2d\t\n"
      "zip1 v21.2d, v18.2d, v19.2d\t\n"
      "lsl x3, x3, #2\t\n"
      "stp q20, q21, [x2]\t\n"

      "zip2 v22.2d, v16.2d, v17.2d\t\n"
      "zip2 v23.2d, v18.2d, v19.2d\t\n"
      "add x2, x2, x3\t\n"
      "stp q22, q23, [x2]\t\n"

      "zip2 v24.4s, v0.4s, v1.4s\t\n"
      "zip2 v25.4s, v2.4s, v3.4s\t\n"
      "zip2 v26.4s, v4.4s, v5.4s\t\n"
      "zip2 v27.4s, v6.4s, v7.4s\t\n"

      "zip1 v28.2d, v24.2d, v25.2d\t\n"
      "zip1 v29.2d, v26.2d, v27.2d\t\n"
      "add x2, x2, x3\t\n"
      "stp q28, q29, [x2]\t\n"

      "zip2 v30.2d, v24.2d, v25.2d\t\n"
      "zip2 v31.2d, v26.2d, v27.2d\t\n"
      "add x2, x2, x3\t\n"
      "stp q30, q31, [x2]\t\n"

      :
      :
      [src] "r"(src), [ld_src] "r"(ld_src), [dst] "r"(dst), [ld_dst] "r"(ld_dst)
      : "memory",
        "cc",
        "x0",
        "x1",
        "x2",
        "x3",
        "v0",
        "v1",
        "v2",
        "v3",
        "v4",
        "v5",
        "v6",
        "v7",
        "v16",
        "v17",
        "v18",
        "v19",
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
        "v30",
        "v31");
}

static inline void transpose_kernel_8x2_neon(
    const float* src,
    int64_t ld_src,
    float* dst,
    int64_t ld_dst) {
  asm volatile(
      "mov x0, %[src]\t\n"
      "mov x1, %[ld_src]\t\n"
      "mov x2, %[dst]\t\n"
      "mov x3, %[ld_dst]\t\n"

      "ldr d0, [x0]\t\n"
      "lsl x1, x1, #2\t\n"
      "add x0, x0, x1\t\n"
      "ldr d1, [x0]\t\n"
      "add x0, x0, x1\t\n"
      "ldr d2, [x0]\t\n"
      "add x0, x0, x1\t\n"
      "ldr d3, [x0]\t\n"
      "add x0, x0, x1\t\n"
      "ldr d4, [x0]\t\n"
      "add x0, x0, x1\t\n"
      "ldr d5, [x0]\t\n"
      "add x0, x0, x1\t\n"
      "ldr d6, [x0]\t\n"
      "add x0, x0, x1\t\n"
      "ldr d7, [x0]\t\n"

      "zip1 v16.2s, v0.2s, v1.2s\t\n"
      "zip1 v17.2s, v2.2s, v3.2s\t\n"
      "zip1 v18.2s, v4.2s, v5.2s\t\n"
      "zip1 v19.2s, v6.2s, v7.2s\t\n"

      "zip1 v20.2d, v16.2d, v17.2d\t\n"
      "zip1 v21.2d, v18.2d, v19.2d\t\n"
      "lsl x3, x3, #2\t\n"
      "stp q20, q21, [x2]\t\n"

      "zip2 v22.2s, v0.2s, v1.2s\t\n"
      "zip2 v23.2s, v2.2s, v3.2s\t\n"
      "zip2 v24.2s, v4.2s, v5.2s\t\n"
      "zip2 v25.2s, v6.2s, v7.2s\t\n"

      "zip1 v26.2d, v22.2d, v23.2d\t\n"
      "zip1 v27.2d, v24.2d, v25.2d\t\n"
      "add x2, x2, x3\t\n"
      "stp q26, q27, [x2]\t\n"

      :
      :
      [src] "r"(src), [ld_src] "r"(ld_src), [dst] "r"(dst), [ld_dst] "r"(ld_dst)
      : "memory",
        "cc",
        "x0",
        "x1",
        "x2",
        "x3",
        "v0",
        "v1",
        "v2",
        "v3",
        "v4",
        "v5",
        "v6",
        "v7",
        "v16",
        "v17",
        "v18",
        "v19",
        "v20",
        "v21",
        "v22",
        "v23",
        "v24",
        "v25",
        "v26",
        "v27");
}

static inline void transpose_kernel_4x8_neon(
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

      "zip1 v16.4s, v0.4s, v2.4s\t\n"
      "zip1 v17.4s, v4.4s, v6.4s\t\n"

      "zip1 v18.2d, v16.2d, v17.2d\t\n"
      "lsl x3, x3, #2\t\n"
      "str q18, [x2]\t\n"

      "zip2 v19.2d, v16.2d, v17.2d\t\n"
      "add x2, x2, x3\t\n"
      "str q19, [x2]\t\n"

      "zip2 v20.4s, v0.4s, v2.4s\t\n"
      "zip2 v21.4s, v4.4s, v6.4s\t\n"

      "zip1 v22.2d, v20.2d, v21.2d\t\n"
      "add x2, x2, x3\t\n"
      "str q22, [x2]\t\n"

      "zip2 v23.2d, v20.2d, v21.2d\t\n"
      "add x2, x2, x3\t\n"
      "str q23, [x2]\t\n"

      "zip1 v24.4s, v1.4s, v3.4s\t\n"
      "zip1 v25.4s, v5.4s, v7.4s\t\n"

      "zip1 v26.2d, v24.2d, v25.2d\t\n"
      "add x2, x2, x3\t\n"
      "str q26, [x2]\t\n"

      "zip2 v27.2d, v24.2d, v25.2d\t\n"
      "add x2, x2, x3\t\n"
      "str q27, [x2]\t\n"

      "zip2 v28.4s, v1.4s, v3.4s\t\n"
      "zip2 v29.4s, v5.4s, v7.4s\t\n"

      "zip1 v30.2d, v28.2d, v29.2d\t\n"
      "add x2, x2, x3\t\n"
      "str q30, [x2]\t\n"

      "zip2 v31.2d, v28.2d, v29.2d\t\n"
      "add x2, x2, x3\t\n"
      "str q31, [x2]\t\n"

      :
      :
      [src] "r"(src), [ld_src] "r"(ld_src), [dst] "r"(dst), [ld_dst] "r"(ld_dst)
      : "memory",
        "cc",
        "x0",
        "x1",
        "x2",
        "x3",
        "v0",
        "v1",
        "v2",
        "v3",
        "v4",
        "v5",
        "v6",
        "v7",
        "v16",
        "v17",
        "v18",
        "v19",
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
        "v30",
        "v31");
}

static inline void transpose_kernel_4x4_neon(
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

      "zip1 v4.4s, v0.4s, v1.4s\t\n"
      "zip1 v5.4s, v2.4s, v3.4s\t\n"

      "zip1 v6.2d, v4.2d, v5.2d\t\n"
      "lsl x3, x3, #2\t\n"
      "str q6, [x2]\t\n"

      "zip2 v7.2d, v4.2d, v5.2d\t\n"
      "add x2, x2, x3\t\n"
      "str q7, [x2]\t\n"

      "zip2 v16.4s, v0.4s, v1.4s\t\n"
      "zip2 v17.4s, v2.4s, v3.4s\t\n"

      "zip1 v18.2d, v16.2d, v17.2d\t\n"
      "add x2, x2, x3\t\n"
      "str q18, [x2]\t\n"

      "zip2 v19.2d, v16.2d, v17.2d\t\n"
      "add x2, x2, x3\t\n"
      "str q19, [x2]\t\n"

      :
      :
      [src] "r"(src), [ld_src] "r"(ld_src), [dst] "r"(dst), [ld_dst] "r"(ld_dst)
      : "memory",
        "cc",
        "x0",
        "x1",
        "x2",
        "x3",
        "v0",
        "v1",
        "v2",
        "v3",
        "v4",
        "v5",
        "v6",
        "v7",
        "v16",
        "v17",
        "v18",
        "v19");
}

static inline void transpose_kernel_4x2_neon(
    const float* src,
    int64_t ld_src,
    float* dst,
    int64_t ld_dst) {
  asm volatile(
      "mov x0, %[src]\t\n"
      "mov x1, %[ld_src]\t\n"
      "mov x2, %[dst]\t\n"
      "mov x3, %[ld_dst]\t\n"

      "ldr d0, [x0]\t\n"
      "lsl x1, x1, #2\t\n"
      "add x0, x0, x1\t\n"
      "ldr d1, [x0]\t\n"
      "add x0, x0, x1\t\n"
      "ldr d2, [x0]\t\n"
      "add x0, x0, x1\t\n"
      "ldr d3, [x0]\t\n"

      "zip1 v16.2s, v0.2s, v1.2s\t\n"
      "zip1 v17.2s, v2.2s, v3.2s\t\n"

      "zip1 v18.2d, v16.2d, v17.2d\t\n"
      "lsl x3, x3, #2\t\n"
      "str q18, [x2]\t\n"

      "zip2 v19.2s, v0.2s, v1.2s\t\n"
      "zip2 v20.2s, v2.2s, v3.2s\t\n"

      "zip1 v21.2d, v19.2d, v20.2d\t\n"
      "add x2, x2, x3\t\n"
      "str q21, [x2]\t\n"

      :
      :
      [src] "r"(src), [ld_src] "r"(ld_src), [dst] "r"(dst), [ld_dst] "r"(ld_dst)
      : "memory",
        "cc",
        "x0",
        "x1",
        "x2",
        "x3",
        "v0",
        "v1",
        "v2",
        "v3",
        "v4",
        "v5",
        "v6",
        "v7",
        "v16",
        "v17",
        "v18",
        "v19",
        "v20",
        "v21");
}

static inline void transpose_kernel_2x8_neon(
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

      "zip1 v4.4s, v0.4s, v2.4s\t\n"
      "lsl x3, x3, #2\t\n"
      "str d4, [x2]\t\n"

      "dup v5.2d, v4.d[1]\t\n"
      "add x2, x2, x3\t\n"
      "str d5, [x2]\t\n"

      "zip2 v6.4s, v0.4s, v2.4s\t\n"
      "add x2, x2, x3\t\n"
      "str d6, [x2]\t\n"

      "dup v7.2d, v6.d[1]\t\n"
      "add x2, x2, x3\t\n"
      "str d7, [x2]\t\n"

      "zip1 v16.4s, v1.4s, v3.4s\t\n"
      "add x2, x2, x3\t\n"
      "str d16, [x2]\t\n"

      "dup v17.2d, v16.d[1]\t\n"
      "add x2, x2, x3\t\n"
      "str d17, [x2]\t\n"

      "zip2 v18.4s, v1.4s, v3.4s\t\n"
      "add x2, x2, x3\t\n"
      "str d18, [x2]\t\n"

      "dup v19.2d, v18.d[1]\t\n"
      "add x2, x2, x3\t\n"
      "str d19, [x2]\t\n"

      :
      :
      [src] "r"(src), [ld_src] "r"(ld_src), [dst] "r"(dst), [ld_dst] "r"(ld_dst)
      : "memory",
        "cc",
        "x0",
        "x1",
        "x2",
        "x3",
        "v0",
        "v1",
        "v2",
        "v3",
        "v4",
        "v5",
        "v6",
        "v7",
        "v16",
        "v17",
        "v18",
        "v19");
}

static inline void transpose_kernel_2x4_neon(
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

      "zip1 v2.4s, v0.4s, v1.4s\t\n"

      "st1 {v2.d}[0], [x2]\t\n"
      "lsl x3, x3, #2\t\n"
      "add x2, x2, x3\t\n"
      "st1 {v2.d}[1], [x2]\t\n"

      "zip2 v3.4s, v0.4s, v1.4s\t\n"

      "add x2, x2, x3\t\n"
      "st1 {v3.d}[0], [x2]\t\n"
      "add x2, x2, x3\t\n"
      "st1 {v3.d}[1], [x2]\t\n"

      :
      :
      [src] "r"(src), [ld_src] "r"(ld_src), [dst] "r"(dst), [ld_dst] "r"(ld_dst)
      : "memory", "cc", "x0", "x1", "x2", "x3", "v0", "v1", "v2", "v3");
}

static inline void transpose_kernel_2x2_neon(
    const float* src,
    int64_t ld_src,
    float* dst,
    int64_t ld_dst) {
  asm volatile(
      "mov x0, %[src]\t\n"
      "mov x1, %[ld_src]\t\n"
      "mov x2, %[dst]\t\n"
      "mov x3, %[ld_dst]\t\n"

      "ldr d0, [x0]\t\n"
      "lsl x1, x1, #2\t\n"
      "add x0, x0, x1\t\n"
      "ldr d1, [x0]\t\n"

      "zip1 v2.4s, v0.4s, v1.4s\t\n"

      "st1 {v2.d}[0], [x2]\t\n"
      "lsl x3, x3, #2\t\n"
      "add x2, x2, x3\t\n"
      "st1 {v2.d}[1], [x2]\t\n"

      :
      :
      [src] "r"(src), [ld_src] "r"(ld_src), [dst] "r"(dst), [ld_dst] "r"(ld_dst)
      : "memory", "cc", "x0", "x1", "x2", "x3", "v0", "v1", "v2");
}

#define TRANSPOSE_FP16_4x4(row0, row1, row2, row3)                           \
  do {                                                                       \
    float16x4x2_t row01 = vtrn_f16(row0, row1);                              \
    float16x4x2_t row23 = vtrn_f16(row2, row3);                              \
    row0 =                                                                   \
      vcvt_f16_f32(vcombine_f32(vget_low_f32(vcvt_f32_f16(row01.val[0])),    \
                                vget_low_f32(vcvt_f32_f16(row23.val[0]))));  \
    row1 =                                                                   \
      vcvt_f16_f32(vcombine_f32(vget_low_f32(vcvt_f32_f16(row01.val[1])),    \
                                vget_low_f32(vcvt_f32_f16(row23.val[1]))));  \
    row2 =                                                                   \
      vcvt_f16_f32(vcombine_f32(vget_high_f32(vcvt_f32_f16(row01.val[0])),   \
                                vget_high_f32(vcvt_f32_f16(row23.val[0])))); \
    row3 =                                                                   \
      vcvt_f16_f32(vcombine_f32(vget_high_f32(vcvt_f32_f16(row01.val[1])),   \
                                vget_high_f32(vcvt_f32_f16(row23.val[1])))); \
  } while (0)

static inline void transpose_kernel_4x4_neon(const __fp16 *src,
                                             int64_t ld_src, __fp16 *dst,
                                             int64_t ld_dst) {
  float16x4_t a = vld1_f16(&src[0 * ld_src]);
  float16x4_t b = vld1_f16(&src[1 * ld_src]);
  float16x4_t c = vld1_f16(&src[2 * ld_src]);
  float16x4_t d = vld1_f16(&src[3 * ld_src]);

  TRANSPOSE_FP16_4x4(a, b, c, d);

  vst1_f16(&dst[0 * ld_dst], a);
  vst1_f16(&dst[1 * ld_dst], b);
  vst1_f16(&dst[2 * ld_dst], c);
  vst1_f16(&dst[3 * ld_dst], d);
}

template <int64_t M>
static void transpose_kernel_mxn_neon_64(int64_t N, const __fp16 *src,
                                          int64_t ld_src, __fp16 *dst,
                                          int64_t ld_dst) {

  float16x4_t input[4];
  float16x4_t ZEROS = vmov_n_f16(0.F);

  unsigned i;
  for (i = 0; i < M; ++i) {
    if (N == 4) {
      input[i] = vld1_f16(&src[i * ld_src]);
    } else {
      float16x4_t tmp = ZEROS;
      for (int64_t n = 0; n < N; ++n) {
        tmp[n] = src[i * ld_src + n];
      }
      input[i] = tmp;
    }
  }
  for (; i < 4; ++i) {
    input[i] = vmov_n_f16(0.F);
  }

  float16x4_t temp[4];
  for (i = 0; i < (M + 1) / 2; ++i) {
    temp[2 * i] = vzip1_f16(input[2 * i], input[2 * i + 1]);
    temp[2 * i + 1] = vzip2_f16(input[2 * i], input[2 * i + 1]);
  }
  for (i = i * 2; i < 4; ++i) {
    temp[i] = vmov_n_f16(0.F);
  }

  for (i = 0; i < N; ++i) {
    if (i % 2 == 0) {
      input[i] =
        vcvt_f16_f32(vcombine_f32(vget_low_f32(vcvt_f32_f16(temp[i / 2])),
                                  vget_low_f32(vcvt_f32_f16(temp[2 + i / 2]))));
    } else {
      input[i] = vcvt_f16_f32(
        vcombine_f32(vget_high_f32(vcvt_f32_f16(temp[i / 2])),
                     vget_high_f32(vcvt_f32_f16(temp[2 + i / 2]))));
    }
    if (M == 4) {
      vst1_f16(&dst[i * ld_dst], input[i]);
    } else {
      for (int64_t m = 0; m < M; ++m) {
        dst[i * ld_dst + m] = input[i][m];
      }
    }
  }
}

static inline void transpose_kernel_8x8_neon(const __fp16 *src,
                                             int64_t ld_src, __fp16 *dst,
                                             int64_t ld_dst) {
  float16x8_t a = vld1q_f16(&src[0 * ld_src]);
  float16x8_t b = vld1q_f16(&src[1 * ld_src]);
  float16x8_t c = vld1q_f16(&src[2 * ld_src]);
  float16x8_t d = vld1q_f16(&src[3 * ld_src]);
  float16x8_t e = vld1q_f16(&src[4 * ld_src]);
  float16x8_t f = vld1q_f16(&src[5 * ld_src]);
  float16x8_t g = vld1q_f16(&src[6 * ld_src]);
  float16x8_t h = vld1q_f16(&src[7 * ld_src]);

  float16x8_t ab0145, ab2367, cd0145, cd2367, ef0145, ef2367, gh0145, gh2367;
  float16x8_t abcd04, abcd15, efgh04, efgh15, abcd26, abcd37, efgh26, efgh37;

  ab0145 = vcombine_f16(vzip1_f16(vget_low_f16(a), vget_low_f16(b)),
                        vzip1_f16(vget_high_f16(a), vget_high_f16(b)));
  ab2367 = vcombine_f16(vzip2_f16(vget_low_f16(a), vget_low_f16(b)),
                        vzip2_f16(vget_high_f16(a), vget_high_f16(b)));
  cd0145 = vcombine_f16(vzip1_f16(vget_low_f16(c), vget_low_f16(d)),
                        vzip1_f16(vget_high_f16(c), vget_high_f16(d)));
  cd2367 = vcombine_f16(vzip2_f16(vget_low_f16(c), vget_low_f16(d)),
                        vzip2_f16(vget_high_f16(c), vget_high_f16(d)));
  ef0145 = vcombine_f16(vzip1_f16(vget_low_f16(e), vget_low_f16(f)),
                        vzip1_f16(vget_high_f16(e), vget_high_f16(f)));
  ef2367 = vcombine_f16(vzip2_f16(vget_low_f16(e), vget_low_f16(f)),
                        vzip2_f16(vget_high_f16(e), vget_high_f16(f)));
  gh0145 = vcombine_f16(vzip1_f16(vget_low_f16(g), vget_low_f16(h)),
                        vzip1_f16(vget_high_f16(g), vget_high_f16(h)));
  gh2367 = vcombine_f16(vzip2_f16(vget_low_f16(g), vget_low_f16(h)),
                        vzip2_f16(vget_high_f16(g), vget_high_f16(h)));

  uint16x8_t shuffle_mask =
    vld1q_u16(reinterpret_cast<const uint16_t *>(shuffle_masks));
  abcd04 = vbslq_f16(shuffle_mask, ab0145, vextq_f16(cd0145, cd0145, 6));
  abcd15 = vbslq_f16(shuffle_mask, vextq_f16(ab0145, ab0145, 2), cd0145);

  efgh04 = vbslq_f16(shuffle_mask, ef0145, vextq_f16(gh0145, gh0145, 6));
  efgh15 = vbslq_f16(shuffle_mask, vextq_f16(ef0145, ef0145, 2), gh0145);

  abcd26 = vbslq_f16(shuffle_mask, ab2367, vextq_f16(cd2367, cd2367, 6));
  abcd37 = vbslq_f16(shuffle_mask, vextq_f16(ab2367, ab2367, 2), cd2367);

  efgh26 = vbslq_f16(shuffle_mask, ef2367, vextq_f16(gh2367, gh2367, 6));
  efgh37 = vbslq_f16(shuffle_mask, vextq_f16(ef2367, ef2367, 2), gh2367);

  a = vcombine_f16(vget_low_f16(abcd04), vget_low_f16(efgh04));
  b = vcombine_f16(vget_low_f16(abcd15), vget_low_f16(efgh15));
  c = vcombine_f16(vget_low_f16(abcd26), vget_low_f16(efgh26));
  d = vcombine_f16(vget_low_f16(abcd37), vget_low_f16(efgh37));
  e = vcombine_f16(vget_high_f16(abcd04), vget_high_f16(efgh04));
  f = vcombine_f16(vget_high_f16(abcd15), vget_high_f16(efgh15));
  g = vcombine_f16(vget_high_f16(abcd26), vget_high_f16(efgh26));
  h = vcombine_f16(vget_high_f16(abcd37), vget_high_f16(efgh37));

  vst1q_f16(&dst[0 * ld_dst], a);
  vst1q_f16(&dst[1 * ld_dst], b);
  vst1q_f16(&dst[2 * ld_dst], c);
  vst1q_f16(&dst[3 * ld_dst], d);
  vst1q_f16(&dst[4 * ld_dst], e);
  vst1q_f16(&dst[5 * ld_dst], f);
  vst1q_f16(&dst[6 * ld_dst], g);
  vst1q_f16(&dst[7 * ld_dst], h);
}

template <int64_t M>
static void transpose_kernel_mxn_neon_128(int64_t N, const __fp16 *src,
                                          int64_t ld_src, __fp16 *dst,
                                          int64_t ld_dst) {
  float16x8_t ZEROS = vmovq_n_f16(0.F);
  float16x8_t input[8];
  unsigned i;
  for (i = 0; i < M; ++i) {
    if (N == 8) {
      input[i] = vld1q_f16(&src[i * ld_src]);
    } else {
      float16x8_t tmp = ZEROS;
      for (int64_t n = 0; n < N; ++n) {
        tmp[n] = src[i * ld_src + n];
      }
      input[i] = tmp;
    }
  }
  for (; i < 8; ++i) {
    input[i] = ZEROS;
  }
  float16x8_t temp[8];
  for (i = 0; i < (M + 1) / 2; ++i) {
    temp[2 * i] = vcombine_f16(
      vzip1_f16(vget_low_f16(input[2 * i]), vget_low_f16(input[2 * i + 1])),
      vzip1_f16(vget_high_f16(input[2 * i]), vget_high_f16(input[2 * i + 1])));
    temp[2 * i + 1] = vcombine_f16(
      vzip2_f16(vget_low_f16(input[2 * i]), vget_low_f16(input[2 * i + 1])),
      vzip2_f16(vget_high_f16(input[2 * i]), vget_high_f16(input[2 * i + 1])));
  }
  for (i = i * 2; i < 8; ++i) {
    temp[i] = ZEROS;
  }

  uint16x8_t shuffle_mask =
    vld1q_u16(reinterpret_cast<const uint16_t *>(shuffle_masks));
  for (i = 0; i < (M + 3) / 4; ++i) {
    input[4 * i] = vbslq_f16(shuffle_mask, temp[4 * i],
                             vextq_f16(temp[4 * i + 2], temp[4 * i + 2], 6));
    input[4 * i + 1] = vbslq_f16(
      shuffle_mask, vextq_f16(temp[4 * i], temp[4 * i], 2), temp[4 * i + 2]);
    input[4 * i + 2] =
      vbslq_f16(shuffle_mask, temp[4 * i + 1],
                vextq_f16(temp[4 * i + 3], temp[4 * i + 3], 6));
    input[4 * i + 3] =
      vbslq_f16(shuffle_mask, vextq_f16(temp[4 * i + 1], temp[4 * i + 1], 2),
                temp[4 * i + 3]);
  }
  for (i = 0; i < N; ++i) {
    if (i < 4) {
      temp[i] =
        vcombine_f16(vget_low_f16(input[i]), vget_low_f16(input[4 + i]));
    } else {
      temp[i] =
        vcombine_f16(vget_high_f16(input[i - 4]), vget_high_f16(input[i]));
    }
    if (M == 8) {
      vst1q_f16(&dst[i * ld_dst], temp[i]);
    } else {
      for (int64_t m = 0; m < M; ++m) {
        dst[i * ld_dst + m] = temp[i][m];
      }
    }
  }
}

} // namespace internal

} // namespace fbgemm

#endif // __aarch64__
