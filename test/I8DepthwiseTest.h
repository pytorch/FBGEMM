/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <vector>

namespace fbgemm {

// From ResNeXt-3D-101
// clang-format off
static std::vector<std::vector<int>> shapes_3d = {
  // NOTE: clang-format wants to use a different formatting but the current
  // formatting should be easier to read.
  // N, K, T_in, H_in, W_in, stride
  {   1,  64,   32,  56, 56, 1, },
  {   1, 128,   16,  28, 28, 1, },
  {   1, 256,    8,  14, 14, 1, },
  {   1, 512,    4,   7,  7, 1, },

  {   1, 128,   32,  56, 56, 2, },
  {   1, 256,   16,  28, 28, 2, },
  {   1, 512,    8,  14, 14, 2, },

  {   5,  64,   32,  56, 56, 1, },
  {   5, 128,   16,  28, 28, 1, },
  {   5, 256,    8,  14, 14, 1, },
  {   5, 512,    4,   7,  7, 1, },

  {   5, 128,   32,  56, 56, 2, },
  {   5, 256,   16,  28, 28, 2, },
  {   5, 512,    8,  14, 14, 2, },

  {   1,   8,    4,   4,  4, 1, },
};
// clang-format on

} // namespace fbgemm
