/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdlib>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <numeric>
#include <string>
#include <tuple>
#include <unordered_map>

#include <ATen/ATen.h>
#include <c10/hip/HIPStream.h>
#include <torch/torch.h>

#include "kernels/fp8_rowwise_grouped_kernel_manifest.h"

namespace fbgemm_gpu {

std::vector<at::Tensor> f8f8bf16_rowwise_grouped(
    at::TensorList XQ,
    at::TensorList WQ,
    at::TensorList x_scale,
    at::TensorList w_scale,
    std::optional<at::TensorList> output = std::nullopt,
    std::optional<std::string> kernel_name = std::nullopt) {
  // Check that input datatypes are valid.
  // First confirm that there are the same number of groups in all inputs.
  TORCH_CHECK(
      XQ.size() == WQ.size() && XQ.size() == x_scale.size() &&
          XQ.size() == w_scale.size(),
      "All inputs must have the same number of groups.");
  // Iterate over inputs and check they are valid.
  for (at::Tensor x : XQ) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous());
    TORCH_CHECK(x.dim() == 2, "Inputs must be 2D.");
    TORCH_CHECK(
        x.dtype() == at::kFloat8_e4m3fnuz,
        "Inputs must be type float8_e4m3fnuz.");
  }
  for (at::Tensor w : WQ) {
    TORCH_CHECK(w.is_cuda() && w.is_contiguous());
    TORCH_CHECK(w.dim() == 2, "Inputs must be 2D.");
    TORCH_CHECK(
        w.dtype() == at::kFloat8_e4m3fnuz,
        "Inputs must be type float8_e4m3fnuz.");
  }
  for (at::Tensor xs : x_scale) {
    TORCH_CHECK(xs.dtype() == at::kFloat, "Scales must be float32.");
  }
  for (at::Tensor ws : x_scale) {
    TORCH_CHECK(ws.dtype() == at::kFloat, "Scales must be float32.");
  }

  // Allocate output if needed.
  std::vector<at::Tensor> Y;
  Y.reserve(XQ.size());
  if (output.has_value()) {
    TORCH_CHECK(output.value().size() == XQ.size(), "Output and input must have same number of groups.");
    // Check that output shapes are correct.
    for (int i = 0; i < output.value().size(); i++) {
      int M = XQ[i].size(0);
      int N = WQ[i].size(0);
      int out_M = output.value()[i].size(0);
      int out_N = output.value()[i].size(1);
      TORCH_CHECK(M == out_M && N == out_N, "Output tensors do not have the expected shape.");
      TORCH_CHECK(output.value()[i].dtype() == at::kBFloat16, "Output dtype must be bfloat16.");
      Y.push_back(output.value()[i]);
    }
  } else {
    for (int i = 0; i < XQ.size(); i++) {
      int M = XQ[i].size(0);
      int N = WQ[i].size(0);
      Y.push_back(at::empty({M, N}, XQ[i].options().dtype(at::kBFloat16)));
    }
  }

  // If provided a specific kernel implementation, dispatch to it.
  if (kernel_name.has_value()) {
    auto it = kernel_name_map.find(kernel_name.value());
    // If not found, raise an error.
    TORCH_CHECK(it != kernel_name_map.end(), "Could not find kernel " + kernel_name.value());
    // If found, always use requested kernel.
    return it->second(XQ, WQ, x_scale, w_scale, Y);
  }
  return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3(
      XQ, WQ, x_scale, w_scale, Y);
}

} // namespace fbgemm_gpu
