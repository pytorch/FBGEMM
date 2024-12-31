/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>

namespace fbgemm_gpu {

enum class KernelMode { Small, Medium, Large, Default };

inline KernelMode get_kernel_mode(at::Tensor XQ, at::Tensor WQ) {
  auto M = XQ.size(0);
  auto K = XQ.size(1);
  auto N = WQ.size(0);
  // Use a large kernel if at least two shapes are large....
  bool use_large_kernel =
      ((M >= 2048 && K >= 2048) || (M >= 2048 && N >= 2048) ||
       (K >= 2048 && N >= 2048));
  if (M <= 128 || N <= 128) {
    return KernelMode::Small;
  } else if (use_large_kernel) {
    return KernelMode::Large;
  } else {
    return KernelMode::Default;
  }
}

inline std::tuple<int64_t, int64_t, int64_t> selectLargestProductDimensions(
    at::TensorList XQ,
    at::TensorList WQ) {
  size_t maxProduct = 0;
  std::tuple<int64_t, int64_t, int64_t> dimensions;
  for (size_t i = 0; i < XQ.size(); ++i) {
    auto& tensor1 = XQ[i];
    auto& tensor2 = WQ[i];
    auto M = tensor1.size(0);
    auto K1 = tensor1.size(1);
    auto N = tensor2.size(0);
    auto K2 = tensor2.size(1);
    if (K1 == K2) { // Ensure the inner dimensions match
      size_t product = M * K1 * N;
      if (product > maxProduct) {
        maxProduct = product;
        dimensions = std::make_tuple(M, N, K1);
      }
    }
  }
  return dimensions;
}

inline KernelMode get_grouped_kernel_mode(
    at::TensorList XQ,
    at::TensorList WQ) {
  // Select the dimensions M, N, K from the pair of tensors with the largest
  // product
  auto [M, N, K] = selectLargestProductDimensions(XQ, WQ);
  // Use a large kernel if at least two shapes are large....
  bool use_large_kernel =
      ((M >= 2048 && K >= 2048) || (M >= 2048 && N >= 2048) ||
       (K >= 2048 && N >= 2048));
  if (M <= 128 || N <= 128) {
    return KernelMode::Small;
  } else if (use_large_kernel) {
    return KernelMode::Large;
  } else {
    return KernelMode::Default;
  }
}

inline KernelMode get_batched_kernel_mode(at::Tensor XQ, at::Tensor WQ) {
  auto B = XQ.size(0);
  auto M = XQ.size(1);
  auto K = XQ.size(2);
  auto N = WQ.size(1);
  auto BM = B * M;
  // Heuristic to determine kernel mode
  bool use_medium_kernel =
      ((BM <= 512 && ((N <= 8192 && K < 8192) || (N < 8192 && K <= 8192))));
  bool use_large_kernel = ((BM > 512 && (N >= 1024 || K >= 1024)));
  if (BM <= 128 || N <= 128) {
    return KernelMode::Small;
  } else if (use_medium_kernel) {
    return KernelMode::Medium;
  } else if (use_large_kernel) {
    return KernelMode::Large;
  } else {
    return KernelMode::Default;
  }
}

} // namespace fbgemm_gpu
