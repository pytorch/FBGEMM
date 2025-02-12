/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <c10/util/Exception.h>

namespace fbgemm_gpu {

namespace {

void check_if_valid_block_dimensions(int m, int n, int k, dim3 block_dim) {
  TORCH_CHECK(
      n % block_dim.y == 0,
      "Invalid block dimensions: n (",
      n,
      ") must be divisible by block_dim.y (",
      block_dim.y,
      "). Received n: ",
      n,
      ", block_dim.y: ",
      block_dim.y,
      " Please either use a `n` which is divisible by `block_dim.y`, or update "
      "`get_best_block_dim()` heuristics to choose another `block_dim.y`. "
      " All current params - m: ",
      m,
      ", n: ",
      n,
      ", k: ",
      k,
      ", block_dim.x: ",
      block_dim.x,
      ", block_dim.y: ",
      block_dim.y,
      ".");
  TORCH_CHECK(
      k % block_dim.x == 0,
      "Invalid block dimensions: k (",
      k,
      ") must be divisible by block_dim.x (",
      block_dim.x,
      "). Received k: ",
      k,
      ", block_dim.x: ",
      block_dim.x,
      " Please either use a `k` which is divisible by `block_dim.x`, or update "
      "`get_best_block_dim()` heuristics to choose another `block_dim.x`."
      " All current params - m: ",
      m,
      ", n: ",
      n,
      ", k: ",
      k,
      ", block_dim.x: ",
      block_dim.x,
      ", block_dim.y: ",
      block_dim.y,
      ".");
  TORCH_CHECK(
      (k / block_dim.x) % 8 == 0,
      "Invalid num_per_thread: (",
      k / block_dim.x,
      ") must be divisible by 8.",
      " Received k: ",
      k,
      ", block_dim.x: ",
      block_dim.x,
      " Please either use a `k` that `k / block_dim.x` that is divisble by 8, or update "
      "`get_best_block_dim()` heuristics to choose another `block_dim.x`."
      " All current params - m: ",
      m,
      ", n: ",
      n,
      ", k: ",
      k,
      ", block_dim.x: ",
      block_dim.x,
      ", block_dim.y: ",
      block_dim.y,
      ".");
}
} // namespace
} // namespace fbgemm_gpu
