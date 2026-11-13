/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "./PlatformSpecificKernelsIsa.h"

#define FBGEMM_SIMD_STRINGIFY_IMPL(x) #x
#define FBGEMM_SIMD_STRINGIFY(x) FBGEMM_SIMD_STRINGIFY_IMPL(x)

namespace fbgemm::FBGEMM_SIMD_NAMESPACE {

namespace {
constinit Kernels instance;
} // namespace

std::string_view Kernels::name() const noexcept {
  return FBGEMM_SIMD_STRINGIFY(FBGEMM_SIMD_NAMESPACE);
}

const PlatformSpecificKernels& kernels() noexcept {
  return instance;
}

} // namespace fbgemm::FBGEMM_SIMD_NAMESPACE
