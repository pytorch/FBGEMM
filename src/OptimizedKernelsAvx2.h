/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <cstdint> // for std::int32_t

namespace fbgemm {

/**
 * @brief Sum a given vector
 */
std::int32_t reduceAvx2(const std::uint8_t* A, int len);

} // namespace fbgemm
