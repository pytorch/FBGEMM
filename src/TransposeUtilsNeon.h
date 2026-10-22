/*
 * Portions Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

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

      "ldp q0, q1, [%[src]]\t\n"
      "lsl %[ld_src], %[ld_src], #2\t\n"
      "add %[src], %[src], %[ld_src]\t\n"
      "ldp q2, q3, [%[src]]\t\n"
      "add %[src], %[src], %[ld_src]\t\n"
      "ldp q4, q5, [%[src]]\t\n"
      "add %[src], %[src], %[ld_src]\t\n"
      "ldp q6, q7, [%[src]]\t\n"
      "add %[src], %[src], %[ld_src]\t\n"
      "ldp q16, q17, [%[src]]\t\n"
      "add %[src], %[src], %[ld_src]\t\n"
      "ldp q18, q19, [%[src]]\t\n"
      "add %[src], %[src], %[ld_src]\t\n"
      "ldp q20, q21, [%[src]]\t\n"
      "add %[src], %[src], %[ld_src]\t\n"
      "ldp q22, q23, [%[src]]\t\n"

      "zip1 v24.4s, v0.4s, v2.4s\t\n"
      "zip1 v25.4s, v4.4s, v6.4s\t\n"
      "zip1 v26.4s, v16.4s, v18.4s\t\n"
      "zip1 v27.4s, v20.4s, v22.4s\t\n"

      "zip1 v28.2d, v24.2d, v25.2d\t\n"
      "zip1 v29.2d, v26.2d, v27.2d\t\n"
      "lsl %[ld_dst], %[ld_dst], #2\t\n"
      "stp q28, q29, [%[dst]]\t\n"

      "zip2 v30.2d, v24.2d, v25.2d\t\n"
      "zip2 v31.2d, v26.2d, v27.2d\t\n"
      "add %[dst], %[dst], %[ld_dst]\t\n"
      "stp q30, q31, [%[dst]]\t\n"

      "zip2 v24.4s, v0.4s, v2.4s\t\n"
      "zip2 v25.4s, v4.4s, v6.4s\t\n"
      "zip2 v26.4s, v16.4s, v18.4s\t\n"
      "zip2 v27.4s, v20.4s, v22.4s\t\n"

      "zip1 v28.2d, v24.2d, v25.2d\t\n"
      "zip1 v29.2d, v26.2d, v27.2d\t\n"
      "add %[dst], %[dst], %[ld_dst]\t\n"
      "stp q28, q29, [%[dst]]\t\n"

      "zip2 v30.2d, v24.2d, v25.2d\t\n"
      "zip2 v31.2d, v26.2d, v27.2d\t\n"
      "add %[dst], %[dst], %[ld_dst]\t\n"
      "stp q30, q31, [%[dst]]\t\n"

      "zip1 v24.4s, v1.4s, v3.4s\t\n"
      "zip1 v25.4s, v5.4s, v7.4s\t\n"
      "zip1 v26.4s, v17.4s, v19.4s\t\n"
      "zip1 v27.4s, v21.4s, v23.4s\t\n"

      "zip1 v28.2d, v24.2d, v25.2d\t\n"
      "zip1 v29.2d, v26.2d, v27.2d\t\n"
      "add %[dst], %[dst], %[ld_dst]\t\n"
      "stp q28, q29, [%[dst]]\t\n"

      "zip2 v30.2d, v24.2d, v25.2d\t\n"
      "zip2 v31.2d, v26.2d, v27.2d\t\n"
      "add %[dst], %[dst], %[ld_dst]\t\n"
      "stp q30, q31, [%[dst]]\t\n"

      "zip2 v24.4s, v1.4s, v3.4s\t\n"
      "zip2 v25.4s, v5.4s, v7.4s\t\n"
      "zip2 v26.4s, v17.4s, v19.4s\t\n"
      "zip2 v27.4s, v21.4s, v23.4s\t\n"

      "zip1 v28.2d, v24.2d, v25.2d\t\n"
      "zip1 v29.2d, v26.2d, v27.2d\t\n"
      "add %[dst], %[dst], %[ld_dst]\t\n"
      "stp q28, q29, [%[dst]]\t\n"

      "zip2 v30.2d, v24.2d, v25.2d\t\n"
      "zip2 v31.2d, v26.2d, v27.2d\t\n"
      "add %[dst], %[dst], %[ld_dst]\t\n"
      "stp q30, q31, [%[dst]]\t\n"

      : [src] "=r"(src),
        [ld_src] "=r"(ld_src),
        [dst] "=r"(dst),
        [ld_dst] "=r"(ld_dst)
      : "[src]"(src), "[ld_src]"(ld_src), "[dst]"(dst), "[ld_dst]"(ld_dst)
      : "memory",
        "cc",
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

      "ldr q0, [%[src]]\t\n"
      "lsl %[ld_src], %[ld_src], #2\t\n"
      "add %[src], %[src], %[ld_src]\t\n"
      "ldr q1, [%[src]]\t\n"
      "add %[src], %[src], %[ld_src]\t\n"
      "ldr q2, [%[src]]\t\n"
      "add %[src], %[src], %[ld_src]\t\n"
      "ldr q3, [%[src]]\t\n"
      "add %[src], %[src], %[ld_src]\t\n"
      "ldr q4, [%[src]]\t\n"
      "add %[src], %[src], %[ld_src]\t\n"
      "ldr q5, [%[src]]\t\n"
      "add %[src], %[src], %[ld_src]\t\n"
      "ldr q6, [%[src]]\t\n"
      "add %[src], %[src], %[ld_src]\t\n"
      "ldr q7, [%[src]]\t\n"

      "zip1 v16.4s, v0.4s, v1.4s\t\n"
      "zip1 v17.4s, v2.4s, v3.4s\t\n"
      "zip1 v18.4s, v4.4s, v5.4s\t\n"
      "zip1 v19.4s, v6.4s, v7.4s\t\n"

      "zip1 v20.2d, v16.2d, v17.2d\t\n"
      "zip1 v21.2d, v18.2d, v19.2d\t\n"
      "lsl %[ld_dst], %[ld_dst], #2\t\n"
      "stp q20, q21, [%[dst]]\t\n"

      "zip2 v22.2d, v16.2d, v17.2d\t\n"
      "zip2 v23.2d, v18.2d, v19.2d\t\n"
      "add %[dst], %[dst], %[ld_dst]\t\n"
      "stp q22, q23, [%[dst]]\t\n"

      "zip2 v24.4s, v0.4s, v1.4s\t\n"
      "zip2 v25.4s, v2.4s, v3.4s\t\n"
      "zip2 v26.4s, v4.4s, v5.4s\t\n"
      "zip2 v27.4s, v6.4s, v7.4s\t\n"

      "zip1 v28.2d, v24.2d, v25.2d\t\n"
      "zip1 v29.2d, v26.2d, v27.2d\t\n"
      "add %[dst], %[dst], %[ld_dst]\t\n"
      "stp q28, q29, [%[dst]]\t\n"

      "zip2 v30.2d, v24.2d, v25.2d\t\n"
      "zip2 v31.2d, v26.2d, v27.2d\t\n"
      "add %[dst], %[dst], %[ld_dst]\t\n"
      "stp q30, q31, [%[dst]]\t\n"

      : [src] "=r"(src),
        [ld_src] "=r"(ld_src),
        [dst] "=r"(dst),
        [ld_dst] "=r"(ld_dst)
      : "[src]"(src), "[ld_src]"(ld_src), "[dst]"(dst), "[ld_dst]"(ld_dst)
      : "memory",
        "cc",
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

      "ldr d0, [%[src]]\t\n"
      "lsl %[ld_src], %[ld_src], #2\t\n"
      "add %[src], %[src], %[ld_src]\t\n"
      "ldr d1, [%[src]]\t\n"
      "add %[src], %[src], %[ld_src]\t\n"
      "ldr d2, [%[src]]\t\n"
      "add %[src], %[src], %[ld_src]\t\n"
      "ldr d3, [%[src]]\t\n"
      "add %[src], %[src], %[ld_src]\t\n"
      "ldr d4, [%[src]]\t\n"
      "add %[src], %[src], %[ld_src]\t\n"
      "ldr d5, [%[src]]\t\n"
      "add %[src], %[src], %[ld_src]\t\n"
      "ldr d6, [%[src]]\t\n"
      "add %[src], %[src], %[ld_src]\t\n"
      "ldr d7, [%[src]]\t\n"

      "zip1 v16.2s, v0.2s, v1.2s\t\n"
      "zip1 v17.2s, v2.2s, v3.2s\t\n"
      "zip1 v18.2s, v4.2s, v5.2s\t\n"
      "zip1 v19.2s, v6.2s, v7.2s\t\n"

      "zip1 v20.2d, v16.2d, v17.2d\t\n"
      "zip1 v21.2d, v18.2d, v19.2d\t\n"
      "lsl %[ld_dst], %[ld_dst], #2\t\n"
      "stp q20, q21, [%[dst]]\t\n"

      "zip2 v22.2s, v0.2s, v1.2s\t\n"
      "zip2 v23.2s, v2.2s, v3.2s\t\n"
      "zip2 v24.2s, v4.2s, v5.2s\t\n"
      "zip2 v25.2s, v6.2s, v7.2s\t\n"

      "zip1 v26.2d, v22.2d, v23.2d\t\n"
      "zip1 v27.2d, v24.2d, v25.2d\t\n"
      "add %[dst], %[dst], %[ld_dst]\t\n"
      "stp q26, q27, [%[dst]]\t\n"

      : [src] "=r"(src),
        [ld_src] "=r"(ld_src),
        [dst] "=r"(dst),
        [ld_dst] "=r"(ld_dst)
      : "[src]"(src), "[ld_src]"(ld_src), "[dst]"(dst), "[ld_dst]"(ld_dst)
      : "memory",
        "cc",
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

      "ldp q0, q1, [%[src]]\t\n"
      "lsl %[ld_src], %[ld_src], #2\t\n"
      "add %[src], %[src], %[ld_src]\t\n"
      "ldp q2, q3, [%[src]]\t\n"
      "add %[src], %[src], %[ld_src]\t\n"
      "ldp q4, q5, [%[src]]\t\n"
      "add %[src], %[src], %[ld_src]\t\n"
      "ldp q6, q7, [%[src]]\t\n"

      "zip1 v16.4s, v0.4s, v2.4s\t\n"
      "zip1 v17.4s, v4.4s, v6.4s\t\n"

      "zip1 v18.2d, v16.2d, v17.2d\t\n"
      "lsl %[ld_dst], %[ld_dst], #2\t\n"
      "str q18, [%[dst]]\t\n"

      "zip2 v19.2d, v16.2d, v17.2d\t\n"
      "add %[dst], %[dst], %[ld_dst]\t\n"
      "str q19, [%[dst]]\t\n"

      "zip2 v20.4s, v0.4s, v2.4s\t\n"
      "zip2 v21.4s, v4.4s, v6.4s\t\n"

      "zip1 v22.2d, v20.2d, v21.2d\t\n"
      "add %[dst], %[dst], %[ld_dst]\t\n"
      "str q22, [%[dst]]\t\n"

      "zip2 v23.2d, v20.2d, v21.2d\t\n"
      "add %[dst], %[dst], %[ld_dst]\t\n"
      "str q23, [%[dst]]\t\n"

      "zip1 v24.4s, v1.4s, v3.4s\t\n"
      "zip1 v25.4s, v5.4s, v7.4s\t\n"

      "zip1 v26.2d, v24.2d, v25.2d\t\n"
      "add %[dst], %[dst], %[ld_dst]\t\n"
      "str q26, [%[dst]]\t\n"

      "zip2 v27.2d, v24.2d, v25.2d\t\n"
      "add %[dst], %[dst], %[ld_dst]\t\n"
      "str q27, [%[dst]]\t\n"

      "zip2 v28.4s, v1.4s, v3.4s\t\n"
      "zip2 v29.4s, v5.4s, v7.4s\t\n"

      "zip1 v30.2d, v28.2d, v29.2d\t\n"
      "add %[dst], %[dst], %[ld_dst]\t\n"
      "str q30, [%[dst]]\t\n"

      "zip2 v31.2d, v28.2d, v29.2d\t\n"
      "add %[dst], %[dst], %[ld_dst]\t\n"
      "str q31, [%[dst]]\t\n"

      : [src] "=r"(src),
        [ld_src] "=r"(ld_src),
        [dst] "=r"(dst),
        [ld_dst] "=r"(ld_dst)
      : "[src]"(src), "[ld_src]"(ld_src), "[dst]"(dst), "[ld_dst]"(ld_dst)
      : "memory",
        "cc",
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

      "ldr q0, [%[src]]\t\n"
      "lsl %[ld_src], %[ld_src], #2\t\n"
      "add %[src], %[src], %[ld_src]\t\n"
      "ldr q1, [%[src]]\t\n"
      "add %[src], %[src], %[ld_src]\t\n"
      "ldr q2, [%[src]]\t\n"
      "add %[src], %[src], %[ld_src]\t\n"
      "ldr q3, [%[src]]\t\n"

      "zip1 v4.4s, v0.4s, v1.4s\t\n"
      "zip1 v5.4s, v2.4s, v3.4s\t\n"

      "zip1 v6.2d, v4.2d, v5.2d\t\n"
      "lsl %[ld_dst], %[ld_dst], #2\t\n"
      "str q6, [%[dst]]\t\n"

      "zip2 v7.2d, v4.2d, v5.2d\t\n"
      "add %[dst], %[dst], %[ld_dst]\t\n"
      "str q7, [%[dst]]\t\n"

      "zip2 v16.4s, v0.4s, v1.4s\t\n"
      "zip2 v17.4s, v2.4s, v3.4s\t\n"

      "zip1 v18.2d, v16.2d, v17.2d\t\n"
      "add %[dst], %[dst], %[ld_dst]\t\n"
      "str q18, [%[dst]]\t\n"

      "zip2 v19.2d, v16.2d, v17.2d\t\n"
      "add %[dst], %[dst], %[ld_dst]\t\n"
      "str q19, [%[dst]]\t\n"

      : [src] "=r"(src),
        [ld_src] "=r"(ld_src),
        [dst] "=r"(dst),
        [ld_dst] "=r"(ld_dst)
      : "[src]"(src), "[ld_src]"(ld_src), "[dst]"(dst), "[ld_dst]"(ld_dst)
      : "memory",
        "cc",
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

      "ldr d0, [%[src]]\t\n"
      "lsl %[ld_src], %[ld_src], #2\t\n"
      "add %[src], %[src], %[ld_src]\t\n"
      "ldr d1, [%[src]]\t\n"
      "add %[src], %[src], %[ld_src]\t\n"
      "ldr d2, [%[src]]\t\n"
      "add %[src], %[src], %[ld_src]\t\n"
      "ldr d3, [%[src]]\t\n"

      "zip1 v16.2s, v0.2s, v1.2s\t\n"
      "zip1 v17.2s, v2.2s, v3.2s\t\n"

      "zip1 v18.2d, v16.2d, v17.2d\t\n"
      "lsl %[ld_dst], %[ld_dst], #2\t\n"
      "str q18, [%[dst]]\t\n"

      "zip2 v19.2s, v0.2s, v1.2s\t\n"
      "zip2 v20.2s, v2.2s, v3.2s\t\n"

      "zip1 v21.2d, v19.2d, v20.2d\t\n"
      "add %[dst], %[dst], %[ld_dst]\t\n"
      "str q21, [%[dst]]\t\n"

      : [src] "=r"(src),
        [ld_src] "=r"(ld_src),
        [dst] "=r"(dst),
        [ld_dst] "=r"(ld_dst)
      : "[src]"(src), "[ld_src]"(ld_src), "[dst]"(dst), "[ld_dst]"(ld_dst)
      : "memory",
        "cc",
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

      "ldp q0, q1, [%[src]]\t\n"
      "lsl %[ld_src], %[ld_src], #2\t\n"
      "add %[src], %[src], %[ld_src]\t\n"
      "ldp q2, q3, [%[src]]\t\n"

      "zip1 v4.4s, v0.4s, v2.4s\t\n"
      "lsl %[ld_dst], %[ld_dst], #2\t\n"
      "str d4, [%[dst]]\t\n"

      "dup v5.2d, v4.d[1]\t\n"
      "add %[dst], %[dst], %[ld_dst]\t\n"
      "str d5, [%[dst]]\t\n"

      "zip2 v6.4s, v0.4s, v2.4s\t\n"
      "add %[dst], %[dst], %[ld_dst]\t\n"
      "str d6, [%[dst]]\t\n"

      "dup v7.2d, v6.d[1]\t\n"
      "add %[dst], %[dst], %[ld_dst]\t\n"
      "str d7, [%[dst]]\t\n"

      "zip1 v16.4s, v1.4s, v3.4s\t\n"
      "add %[dst], %[dst], %[ld_dst]\t\n"
      "str d16, [%[dst]]\t\n"

      "dup v17.2d, v16.d[1]\t\n"
      "add %[dst], %[dst], %[ld_dst]\t\n"
      "str d17, [%[dst]]\t\n"

      "zip2 v18.4s, v1.4s, v3.4s\t\n"
      "add %[dst], %[dst], %[ld_dst]\t\n"
      "str d18, [%[dst]]\t\n"

      "dup v19.2d, v18.d[1]\t\n"
      "add %[dst], %[dst], %[ld_dst]\t\n"
      "str d19, [%[dst]]\t\n"

      : [src] "=r"(src),
        [ld_src] "=r"(ld_src),
        [dst] "=r"(dst),
        [ld_dst] "=r"(ld_dst)
      : "[src]"(src), "[ld_src]"(ld_src), "[dst]"(dst), "[ld_dst]"(ld_dst)
      : "memory",
        "cc",
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

      "ldr q0, [%[src]]\t\n"
      "lsl %[ld_src], %[ld_src], #2\t\n"
      "add %[src], %[src], %[ld_src]\t\n"
      "ldr q1, [%[src]]\t\n"

      "zip1 v2.4s, v0.4s, v1.4s\t\n"

      "st1 {v2.d}[0], [%[dst]]\t\n"
      "lsl %[ld_dst], %[ld_dst], #2\t\n"
      "add %[dst], %[dst], %[ld_dst]\t\n"
      "st1 {v2.d}[1], [%[dst]]\t\n"

      "zip2 v3.4s, v0.4s, v1.4s\t\n"

      "add %[dst], %[dst], %[ld_dst]\t\n"
      "st1 {v3.d}[0], [%[dst]]\t\n"
      "add %[dst], %[dst], %[ld_dst]\t\n"
      "st1 {v3.d}[1], [%[dst]]\t\n"

      : [src] "=r"(src),
        [ld_src] "=r"(ld_src),
        [dst] "=r"(dst),
        [ld_dst] "=r"(ld_dst)
      : "[src]"(src), "[ld_src]"(ld_src), "[dst]"(dst), "[ld_dst]"(ld_dst)
      : "memory", "cc", "v0", "v1", "v2", "v3");
}

static inline void transpose_kernel_2x2_neon(
    const float* src,
    int64_t ld_src,
    float* dst,
    int64_t ld_dst) {
  asm volatile(

      "ldr d0, [%[src]]\t\n"
      "lsl %[ld_src], %[ld_src], #2\t\n"
      "add %[src], %[src], %[ld_src]\t\n"
      "ldr d1, [%[src]]\t\n"

      "zip1 v2.4s, v0.4s, v1.4s\t\n"

      "st1 {v2.d}[0], [%[dst]]\t\n"
      "lsl %[ld_dst], %[ld_dst], #2\t\n"
      "add %[dst], %[dst], %[ld_dst]\t\n"
      "st1 {v2.d}[1], [%[dst]]\t\n"

      : [src] "=r"(src),
        [ld_src] "=r"(ld_src),
        [dst] "=r"(dst),
        [ld_dst] "=r"(ld_dst)
      : "[src]"(src), "[ld_src]"(ld_src), "[dst]"(dst), "[ld_dst]"(ld_dst)
      : "memory", "cc", "v0", "v1", "v2");
}

static inline uint16x4_t load_u16x2_neon(const uint16_t* src) {
  uint16x4_t result = vdup_n_u16(0);
  result = vld1_lane_u16(src, result, 0);
  return vld1_lane_u16(src + 1, result, 1);
}

static inline void store_u16x2_neon(uint16_t* dst, uint16x4_t value) {
  vst1_lane_u16(dst, value, 0);
  vst1_lane_u16(dst + 1, value, 1);
}

static inline void transpose_kernel_8x8_neon(
    const uint16_t* src,
    int64_t ld_src,
    uint16_t* dst,
    int64_t ld_dst) {
  const uint16x8_t r0 = vld1q_u16(src);
  const uint16x8_t r1 = vld1q_u16(src + ld_src);
  const uint16x8_t r2 = vld1q_u16(src + 2 * ld_src);
  const uint16x8_t r3 = vld1q_u16(src + 3 * ld_src);
  const uint16x8_t r4 = vld1q_u16(src + 4 * ld_src);
  const uint16x8_t r5 = vld1q_u16(src + 5 * ld_src);
  const uint16x8_t r6 = vld1q_u16(src + 6 * ld_src);
  const uint16x8_t r7 = vld1q_u16(src + 7 * ld_src);

  const uint32x4_t a0l = vreinterpretq_u32_u16(vzip1q_u16(r0, r1));
  const uint32x4_t a0h = vreinterpretq_u32_u16(vzip2q_u16(r0, r1));
  const uint32x4_t a1l = vreinterpretq_u32_u16(vzip1q_u16(r2, r3));
  const uint32x4_t a1h = vreinterpretq_u32_u16(vzip2q_u16(r2, r3));
  const uint32x4_t a2l = vreinterpretq_u32_u16(vzip1q_u16(r4, r5));
  const uint32x4_t a2h = vreinterpretq_u32_u16(vzip2q_u16(r4, r5));
  const uint32x4_t a3l = vreinterpretq_u32_u16(vzip1q_u16(r6, r7));
  const uint32x4_t a3h = vreinterpretq_u32_u16(vzip2q_u16(r6, r7));

  const uint64x2_t b00 = vreinterpretq_u64_u32(vzip1q_u32(a0l, a1l));
  const uint64x2_t b01 = vreinterpretq_u64_u32(vzip2q_u32(a0l, a1l));
  const uint64x2_t b02 = vreinterpretq_u64_u32(vzip1q_u32(a0h, a1h));
  const uint64x2_t b03 = vreinterpretq_u64_u32(vzip2q_u32(a0h, a1h));
  const uint64x2_t b10 = vreinterpretq_u64_u32(vzip1q_u32(a2l, a3l));
  const uint64x2_t b11 = vreinterpretq_u64_u32(vzip2q_u32(a2l, a3l));
  const uint64x2_t b12 = vreinterpretq_u64_u32(vzip1q_u32(a2h, a3h));
  const uint64x2_t b13 = vreinterpretq_u64_u32(vzip2q_u32(a2h, a3h));

  vst1q_u16(dst, vreinterpretq_u16_u64(vzip1q_u64(b00, b10)));
  vst1q_u16(dst + ld_dst, vreinterpretq_u16_u64(vzip2q_u64(b00, b10)));
  vst1q_u16(dst + 2 * ld_dst, vreinterpretq_u16_u64(vzip1q_u64(b01, b11)));
  vst1q_u16(dst + 3 * ld_dst, vreinterpretq_u16_u64(vzip2q_u64(b01, b11)));
  vst1q_u16(dst + 4 * ld_dst, vreinterpretq_u16_u64(vzip1q_u64(b02, b12)));
  vst1q_u16(dst + 5 * ld_dst, vreinterpretq_u16_u64(vzip2q_u64(b02, b12)));
  vst1q_u16(dst + 6 * ld_dst, vreinterpretq_u16_u64(vzip1q_u64(b03, b13)));
  vst1q_u16(dst + 7 * ld_dst, vreinterpretq_u16_u64(vzip2q_u64(b03, b13)));
}

static inline void transpose_kernel_8x4_neon(
    const uint16_t* src,
    int64_t ld_src,
    uint16_t* dst,
    int64_t ld_dst) {
  const uint16x4_t zero = vdup_n_u16(0);
  const uint16x8_t r0 = vcombine_u16(vld1_u16(src), zero);
  const uint16x8_t r1 = vcombine_u16(vld1_u16(src + ld_src), zero);
  const uint16x8_t r2 = vcombine_u16(vld1_u16(src + 2 * ld_src), zero);
  const uint16x8_t r3 = vcombine_u16(vld1_u16(src + 3 * ld_src), zero);
  const uint16x8_t r4 = vcombine_u16(vld1_u16(src + 4 * ld_src), zero);
  const uint16x8_t r5 = vcombine_u16(vld1_u16(src + 5 * ld_src), zero);
  const uint16x8_t r6 = vcombine_u16(vld1_u16(src + 6 * ld_src), zero);
  const uint16x8_t r7 = vcombine_u16(vld1_u16(src + 7 * ld_src), zero);

  const uint32x4_t a0 = vreinterpretq_u32_u16(vzip1q_u16(r0, r1));
  const uint32x4_t a1 = vreinterpretq_u32_u16(vzip1q_u16(r2, r3));
  const uint32x4_t a2 = vreinterpretq_u32_u16(vzip1q_u16(r4, r5));
  const uint32x4_t a3 = vreinterpretq_u32_u16(vzip1q_u16(r6, r7));

  const uint64x2_t b0 = vreinterpretq_u64_u32(vzip1q_u32(a0, a1));
  const uint64x2_t b1 = vreinterpretq_u64_u32(vzip2q_u32(a0, a1));
  const uint64x2_t b2 = vreinterpretq_u64_u32(vzip1q_u32(a2, a3));
  const uint64x2_t b3 = vreinterpretq_u64_u32(vzip2q_u32(a2, a3));

  vst1q_u16(dst, vreinterpretq_u16_u64(vzip1q_u64(b0, b2)));
  vst1q_u16(dst + ld_dst, vreinterpretq_u16_u64(vzip2q_u64(b0, b2)));
  vst1q_u16(dst + 2 * ld_dst, vreinterpretq_u16_u64(vzip1q_u64(b1, b3)));
  vst1q_u16(
      dst + 3 * ld_dst, vreinterpretq_u16_u64(vzip2q_u64(b1, b3)));
}

static inline void transpose_kernel_8x2_neon(
    const uint16_t* src,
    int64_t ld_src,
    uint16_t* dst,
    int64_t ld_dst) {
  const uint16x4_t zero = vdup_n_u16(0);
  const uint16x8_t r0 = vcombine_u16(load_u16x2_neon(src), zero);
  const uint16x8_t r1 =
      vcombine_u16(load_u16x2_neon(src + ld_src), zero);
  const uint16x8_t r2 =
      vcombine_u16(load_u16x2_neon(src + 2 * ld_src), zero);
  const uint16x8_t r3 =
      vcombine_u16(load_u16x2_neon(src + 3 * ld_src), zero);
  const uint16x8_t r4 =
      vcombine_u16(load_u16x2_neon(src + 4 * ld_src), zero);
  const uint16x8_t r5 =
      vcombine_u16(load_u16x2_neon(src + 5 * ld_src), zero);
  const uint16x8_t r6 =
      vcombine_u16(load_u16x2_neon(src + 6 * ld_src), zero);
  const uint16x8_t r7 =
      vcombine_u16(load_u16x2_neon(src + 7 * ld_src), zero);

  const uint32x4_t a0 = vreinterpretq_u32_u16(vzip1q_u16(r0, r1));
  const uint32x4_t a1 = vreinterpretq_u32_u16(vzip1q_u16(r2, r3));
  const uint32x4_t a2 = vreinterpretq_u32_u16(vzip1q_u16(r4, r5));
  const uint32x4_t a3 = vreinterpretq_u32_u16(vzip1q_u16(r6, r7));

  const uint64x2_t b0 = vreinterpretq_u64_u32(vzip1q_u32(a0, a1));
  const uint64x2_t b1 = vreinterpretq_u64_u32(vzip1q_u32(a2, a3));

  vst1q_u16(dst, vreinterpretq_u16_u64(vzip1q_u64(b0, b1)));
  vst1q_u16(dst + ld_dst, vreinterpretq_u16_u64(vzip2q_u64(b0, b1)));
}

static inline void transpose_kernel_4x8_neon(
    const uint16_t* src,
    int64_t ld_src,
    uint16_t* dst,
    int64_t ld_dst) {
  const uint16x8_t r0 = vld1q_u16(src);
  const uint16x8_t r1 = vld1q_u16(src + ld_src);
  const uint16x8_t r2 = vld1q_u16(src + 2 * ld_src);
  const uint16x8_t r3 = vld1q_u16(src + 3 * ld_src);

  const uint32x4_t a0l = vreinterpretq_u32_u16(vzip1q_u16(r0, r1));
  const uint32x4_t a0h = vreinterpretq_u32_u16(vzip2q_u16(r0, r1));
  const uint32x4_t a1l = vreinterpretq_u32_u16(vzip1q_u16(r2, r3));
  const uint32x4_t a1h = vreinterpretq_u32_u16(vzip2q_u16(r2, r3));

  const uint16x8_t b0 = vreinterpretq_u16_u32(vzip1q_u32(a0l, a1l));
  const uint16x8_t b1 = vreinterpretq_u16_u32(vzip2q_u32(a0l, a1l));
  const uint16x8_t b2 = vreinterpretq_u16_u32(vzip1q_u32(a0h, a1h));
  const uint16x8_t b3 = vreinterpretq_u16_u32(vzip2q_u32(a0h, a1h));

  vst1_u16(dst, vget_low_u16(b0));
  vst1_u16(dst + ld_dst, vget_high_u16(b0));
  vst1_u16(dst + 2 * ld_dst, vget_low_u16(b1));
  vst1_u16(dst + 3 * ld_dst, vget_high_u16(b1));
  vst1_u16(dst + 4 * ld_dst, vget_low_u16(b2));
  vst1_u16(dst + 5 * ld_dst, vget_high_u16(b2));
  vst1_u16(dst + 6 * ld_dst, vget_low_u16(b3));
  vst1_u16(dst + 7 * ld_dst, vget_high_u16(b3));
}

static inline void transpose_kernel_4x4_neon(
    const uint16_t* src,
    int64_t ld_src,
    uint16_t* dst,
    int64_t ld_dst) {
  const uint16x4_t r0 = vld1_u16(src);
  const uint16x4_t r1 = vld1_u16(src + ld_src);
  const uint16x4_t r2 = vld1_u16(src + 2 * ld_src);
  const uint16x4_t r3 = vld1_u16(src + 3 * ld_src);
  const uint32x2_t a0 = vreinterpret_u32_u16(vzip1_u16(r0, r1));
  const uint32x2_t a1 = vreinterpret_u32_u16(vzip2_u16(r0, r1));
  const uint32x2_t a2 = vreinterpret_u32_u16(vzip1_u16(r2, r3));
  const uint32x2_t a3 = vreinterpret_u32_u16(vzip2_u16(r2, r3));
  vst1_u16(dst, vreinterpret_u16_u32(vzip1_u32(a0, a2)));
  vst1_u16(dst + ld_dst, vreinterpret_u16_u32(vzip2_u32(a0, a2)));
  vst1_u16(dst + 2 * ld_dst, vreinterpret_u16_u32(vzip1_u32(a1, a3)));
  vst1_u16(dst + 3 * ld_dst, vreinterpret_u16_u32(vzip2_u32(a1, a3)));
}

static inline void transpose_kernel_4x2_neon(
    const uint16_t* src,
    int64_t ld_src,
    uint16_t* dst,
    int64_t ld_dst) {
  const uint16x4_t r0 = load_u16x2_neon(src);
  const uint16x4_t r1 = load_u16x2_neon(src + ld_src);
  const uint16x4_t r2 = load_u16x2_neon(src + 2 * ld_src);
  const uint16x4_t r3 = load_u16x2_neon(src + 3 * ld_src);
  const uint32x2_t a0 = vreinterpret_u32_u16(vzip1_u16(r0, r1));
  const uint32x2_t a1 = vreinterpret_u32_u16(vzip1_u16(r2, r3));
  vst1_u16(dst, vreinterpret_u16_u32(vzip1_u32(a0, a1)));
  vst1_u16(dst + ld_dst, vreinterpret_u16_u32(vzip2_u32(a0, a1)));
}

static inline void transpose_kernel_2x8_neon(
    const uint16_t* src,
    int64_t ld_src,
    uint16_t* dst,
    int64_t ld_dst) {
  const uint16x8_t r0 = vld1q_u16(src);
  const uint16x8_t r1 = vld1q_u16(src + ld_src);
  const uint16x8_t a0 = vzip1q_u16(r0, r1);
  const uint16x8_t a1 = vzip2q_u16(r0, r1);

  store_u16x2_neon(dst, vget_low_u16(a0));
  store_u16x2_neon(
      dst + ld_dst, vext_u16(vget_low_u16(a0), vget_low_u16(a0), 2));
  store_u16x2_neon(dst + 2 * ld_dst, vget_high_u16(a0));
  store_u16x2_neon(
      dst + 3 * ld_dst,
      vext_u16(vget_high_u16(a0), vget_high_u16(a0), 2));
  store_u16x2_neon(dst + 4 * ld_dst, vget_low_u16(a1));
  store_u16x2_neon(
      dst + 5 * ld_dst, vext_u16(vget_low_u16(a1), vget_low_u16(a1), 2));
  store_u16x2_neon(dst + 6 * ld_dst, vget_high_u16(a1));
  store_u16x2_neon(
      dst + 7 * ld_dst,
      vext_u16(vget_high_u16(a1), vget_high_u16(a1), 2));
}

static inline void transpose_kernel_2x4_neon(
    const uint16_t* src,
    int64_t ld_src,
    uint16_t* dst,
    int64_t ld_dst) {
  const uint16x4_t r0 = vld1_u16(src);
  const uint16x4_t r1 = vld1_u16(src + ld_src);
  const uint16x4_t a0 = vzip1_u16(r0, r1);
  const uint16x4_t a1 = vzip2_u16(r0, r1);

  store_u16x2_neon(dst, a0);
  store_u16x2_neon(dst + ld_dst, vext_u16(a0, a0, 2));
  store_u16x2_neon(dst + 2 * ld_dst, a1);
  store_u16x2_neon(dst + 3 * ld_dst, vext_u16(a1, a1, 2));
}

static inline void transpose_kernel_2x2_neon(
    const uint16_t* src,
    int64_t ld_src,
    uint16_t* dst,
    int64_t ld_dst) {
  const uint16x4_t r0 = load_u16x2_neon(src);
  const uint16x4_t r1 = load_u16x2_neon(src + ld_src);
  const uint16x4_t a0 = vzip1_u16(r0, r1);
  store_u16x2_neon(dst, a0);
  store_u16x2_neon(dst + ld_dst, vext_u16(a0, a0, 2));
}

} // namespace internal

} // namespace fbgemm

#endif // __aarch64__
