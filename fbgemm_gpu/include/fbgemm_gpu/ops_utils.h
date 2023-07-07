/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

/*
 * We annotate the public FBGEMM functions and hide the rest. Those
 * public symbols can be called via fbgemm_gpu::func() or pytorch
 * operator dispatcher. We'll hide other symbols, especially CUB APIs,
 * because different .so may include the same CUB CUDA kernels, which
 * results in confusion and libA may end up calling libB's CUB kernel,
 * causing failures when we static link libcudart_static.a
 */
#define DLL_PUBLIC __attribute__((visibility("default")))

#define FBGEMM_OP_DISPATCH(DISPATCH_KEY, EXPORT_NAME, FUNC_NAME)               \
  TORCH_LIBRARY_IMPL(fbgemm, DISPATCH_KEY, m) {                                \
    m.impl(                                                                    \
        EXPORT_NAME,                                                           \
        torch::dispatch(c10::DispatchKey::DISPATCH_KEY, TORCH_FN(FUNC_NAME))); \
  }
