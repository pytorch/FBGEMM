/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <vector>

#include "fbgemm/FbgemmBuild.h"
#include "fbgemm/detail/PlatformSpecificKernelsInterface.h"

namespace fbgemm {

namespace ref {
const PlatformSpecificKernels& kernels() noexcept;
} // namespace ref
namespace avx2 {
const PlatformSpecificKernels& kernels() noexcept;
} // namespace avx2
namespace avx512_256 {
const PlatformSpecificKernels& kernels() noexcept;
} // namespace avx512_256
namespace avx512_256_bf16 {
const PlatformSpecificKernels& kernels() noexcept;
} // namespace avx512_256_bf16
namespace avx512 {
const PlatformSpecificKernels& kernels() noexcept;
} // namespace avx512
namespace avx512_bf16 {
const PlatformSpecificKernels& kernels() noexcept;
} // namespace avx512_bf16
namespace neon {
const PlatformSpecificKernels& kernels() noexcept;
} // namespace neon
namespace sve128 {
const PlatformSpecificKernels& kernels() noexcept;
} // namespace sve128

FBGEMM_API std::vector<const PlatformSpecificKernels*>
supportedPlatformKernels();

FBGEMM_API void initializePlatformDispatch(bool force = false) noexcept;

} // namespace fbgemm
