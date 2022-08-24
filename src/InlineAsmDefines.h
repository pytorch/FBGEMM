/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// We need to do a hack in inline assembly in some clang versions where we have
// to do `.intel_syntax noprefix`. This was fixed in clang in
// https://reviews.llvm.org/D113707, which made it into clang-14, but not in
// Apple's clang-14 that ships with Xcode 14.
#if defined(__clang__)

#if (                                                                      \
    defined(__apple_build_version__) ||                                    \
    (defined(__has_builtin) && __has_builtin(__builtin_pika_xxhash64))) && \
    (__clang_major__ < 15)
#define FBGEMM_USE_CLANG_INTEL_SYNTAX_ASM_HACK 1
#elif (__clang_major__ < 14)
#define FBGEMM_USE_CLANG_INTEL_SYNTAX_ASM_HACK 1
#endif

#endif // defined(__clang__)

#ifndef FBGEMM_USE_CLANG_INTEL_SYNTAX_ASM_HACK
#define FBGEMM_USE_CLANG_INTEL_SYNTAX_ASM_HACK 0
#endif
