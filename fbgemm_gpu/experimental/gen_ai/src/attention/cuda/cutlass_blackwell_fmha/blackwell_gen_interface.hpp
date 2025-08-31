/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <cutlass/bfloat16.h>
#include <cutlass/cutlass.h>
#include <cutlass/float8.h>
#include <cutlass/half.h>
#include <torch/library.h>
#include <torch/torch.h>

using namespace cutlass;

enum class KernelType { UMMA_I = 0, UMMA_P = 1 };

// Template function definition for type conversion
template <typename T>
at::ScalarType to_torch_type() {
  if constexpr (std::is_same_v<T, cutlass::half_t>) {
    return at::kHalf;
  } else if constexpr (std::is_same_v<T, cutlass::bfloat16_t>) {
    return at::kBFloat16;
  } else if constexpr (std::is_same_v<T, cutlass::float_e4m3_t>) {
    return at::kFloat8_e4m3fn;
  } else if constexpr (std::is_same_v<T, float>) {
    return at::kFloat;
  } else {
    static_assert(sizeof(T) == 0, "Unsupported type for to_torch_type");
    return at::kFloat;
  }
}

// Main dispatch function for the generation FMHA
at::Tensor dispatch_fmha_gen_fwd(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& seqlen_kv,
    const at::Tensor& batch_idx,
    int64_t kernel_type);
