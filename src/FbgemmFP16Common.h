/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <fbgemm/Types.h>
#include <fbgemm/Utils.h>
#include <array>

namespace fbgemm {
using partition_array_t = std::array<std::array<std::array<int, 2>, 2>, 121>;

template <typename T>
struct GemmParams {
  uint64_t k;
  float* A;
  const T* B;
  float beta;
  float* C;
  uint64_t ldc;
  uint64_t b_block_cols;
  uint64_t b_block_size;
};

template <typename T>
using funcptr_t = void (*)(GemmParams<T>*);

using fp16 = float16;
using fp32 = float;
using GemmParamsFP16 = GemmParams<fp16>;

} // namespace fbgemm
