/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <span>

namespace fbgemm_gpu::utils {

int64_t count_true(std::span<const bool> mask);

} // namespace fbgemm_gpu::utils
