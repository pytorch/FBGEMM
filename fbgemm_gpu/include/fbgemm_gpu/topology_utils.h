/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <functional>

using Node = int64_t;
using Links = int64_t;
template <typename T>
using AdjacencyMatrix = std::function<T(Node, Node)>;

namespace fbgemm_gpu {
AdjacencyMatrix<Links> get_nvlink_matrix();
} // namespace fbgemm_gpu
