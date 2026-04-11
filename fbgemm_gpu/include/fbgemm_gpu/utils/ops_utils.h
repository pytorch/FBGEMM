/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/library.h>

// NOLINTNEXTLINE(misc-unused-using-decls)
#include "fbgemm_gpu/utils/function_types.h"

////////////////////////////////////////////////////////////////////////////////
/// Op Dispatch Macros
////////////////////////////////////////////////////////////////////////////////

#define FBGEMM_OP_DISPATCH(DISPATCH_KEY, EXPORT_NAME, FUNC_NAME)               \
  TORCH_LIBRARY_IMPL(fbgemm, DISPATCH_KEY, m) {                                \
    m.impl(                                                                    \
        EXPORT_NAME,                                                           \
        torch::dispatch(c10::DispatchKey::DISPATCH_KEY, TORCH_FN(FUNC_NAME))); \
  }

#define DISPATCH_TO_CUDA(name, function) \
  m.impl(name, torch::dispatch(c10::DispatchKey::CUDA, TORCH_FN(function)))

#define DISPATCH_TO_CPU(name, function) \
  m.impl(name, torch::dispatch(c10::DispatchKey::CPU, TORCH_FN(function)))

#define DISPATCH_TO_QUANTIZED_CPU(name, function) \
  m.impl(                                         \
      name,                                       \
      torch::dispatch(c10::DispatchKey::QuantizedCPU, TORCH_FN(function)))

#define DISPATCH_TO_META(name, function) \
  m.impl(name, torch::dispatch(c10::DispatchKey::Meta, TORCH_FN(function)))

#define DISPATCH_TO_ALL(name, function) \
  m.impl(name, torch::dispatch(c10::DispatchKey::CatchAll, TORCH_FN(function)))

#define DISPATCH_TO_AUTOGRAD(name, function) \
  m.impl(name, torch::dispatch(c10::DispatchKey::Autograd, TORCH_FN(function)))

#define DISPATCH_TO_AUTOGRAD_CPU(name, function) \
  m.impl(                                        \
      name,                                      \
      torch::dispatch(c10::DispatchKey::AutogradCPU, TORCH_FN(function)))

#define DISPATCH_TO_AUTOGRAD_CUDA(name, function) \
  m.impl(                                         \
      name,                                       \
      torch::dispatch(c10::DispatchKey::AutogradCUDA, TORCH_FN(function)))

#define DISPATCH_TO_AUTOGRAD_META(name, function) \
  m.impl(                                         \
      name,                                       \
      torch::dispatch(c10::DispatchKey::AutogradMETA, TORCH_FN(function)))
