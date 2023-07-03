/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>

#define FBGEMM_OP_DISPATCH(DISPATCH_KEY, EXPORT_NAME, FUNC_NAME)               \
  TORCH_LIBRARY_IMPL(fbgemm, DISPATCH_KEY, m) {                                \
    m.impl(                                                                    \
        EXPORT_NAME,                                                           \
        torch::dispatch(c10::DispatchKey::DISPATCH_KEY, TORCH_FN(FUNC_NAME))); \
  }
