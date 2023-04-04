/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * Copyright (c) Intel Corporation.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <ATen/ATen.h>
#include <cstdint>
#include <utility>

namespace fbgemm_gpu {

/**
 * report error from fbgemm cpu embedding lookup kernels
 * @params allow_minus_one true for embedding kernels generated with
 *         scale_bias_last == false that can take -1 indices (output from
 *         pruned embedding id mapping)
 */
template <typename IndexType>
void report_embedding_error(
    int t,
    int B,
    int b_begin,
    int b_end,
    const IndexType* offsets_data,
    const IndexType* indices_data,
    int64_t hash_size,
    bool allow_minus_one = false) {
  for (int b = b_begin; b < b_end; ++b) {
    const auto pool_begin = offsets_data[t * B + b];
    const auto pool_end = offsets_data[t * B + b + 1];
    for (auto p = pool_begin; p < pool_end; ++p) {
      auto idx = indices_data[p];
      TORCH_CHECK(
          (allow_minus_one ? -1 : 0) <= idx && idx < hash_size,
          "Index ",
          p,
          " is out of bounds: ",
          idx,
          ", range ",
          (allow_minus_one ? "-1" : "0"),
          " to ",
          hash_size);
    }
  }
}

} // namespace fbgemm_gpu
