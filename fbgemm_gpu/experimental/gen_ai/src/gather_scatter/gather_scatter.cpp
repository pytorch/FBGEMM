/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <torch/library.h>

namespace fbgemm_gpu {

#ifdef USE_ROCM
at::Tensor gather_along_first_dim(at::Tensor data, at::Tensor index) {
  return at::index_select(data, 0, index);
}

void scatter_add_along_first_dim(
    at::Tensor dst,
    at::Tensor src,
    at::Tensor index) {
  const int M = src.size(0);
  const int K = src.size(1);
  const int N = dst.size(0);
  if (N == 0 || M == 0) {
    TORCH_CHECK(M == 0, "Expected M == 0 when N == 0 || M == 0, but got M=", M);
    return;
  }
  dst.scatter_add_(0, index.to(at::kLong).unsqueeze(1).expand({-1, K}), src);
}
#else
at::Tensor gather_along_first_dim(at::Tensor data, at::Tensor index);

void scatter_add_along_first_dim(
    at::Tensor dst,
    at::Tensor src,
    at::Tensor index);
#endif

at::Tensor gather_along_first_dim_meta(at::Tensor data, at::Tensor index) {
  int K = data.size(1);
  int N = index.size(0);
  at::Tensor output = at::empty({N, K}, data.options());
  return output;
}

void scatter_add_along_first_dim_meta(
    at::Tensor /*dst*/,
    at::Tensor /*src*/,
    at::Tensor /*index*/) {
  return;
}

TORCH_LIBRARY_IMPL(fbgemm, CUDA, m) {
  m.impl("gather_along_first_dim", gather_along_first_dim);
  m.impl("scatter_add_along_first_dim", scatter_add_along_first_dim);
}

TORCH_LIBRARY_IMPL(fbgemm, Meta, m) {
  m.impl("gather_along_first_dim", gather_along_first_dim_meta);
  m.impl("scatter_add_along_first_dim", scatter_add_along_first_dim_meta);
}

} // namespace fbgemm_gpu
