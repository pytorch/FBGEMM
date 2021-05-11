/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/library.h>
#include "fbgemm/QuantUtils.h"
#include "fbgemm_gpu/sparse_ops_utils.h"

namespace at {

at::Tensor& _float_to_fused8bitrowwise_cpu_out(
    at::Tensor& output,
    const at::Tensor& input) {
  TENSOR_ON_CPU(input);
  TORCH_CHECK(
      input.dim() >= 2,
      "Tensor 'input' must have >= 2 dimension(s). Found ",
      input.ndimension());

  const auto input_sizes = input.sizes();
  const auto last_dim = input_sizes.size() - 1;
  const int32_t nrows = c10::size_to_dim_(last_dim, input_sizes);
  const int32_t ncols = input_sizes[last_dim];
  const int32_t output_columns = ncols + 2 * sizeof(float);

  auto output_dims = input_sizes.vec();
  output_dims[last_dim] = output_columns;
  at::native::resize_(output, output_dims, c10::nullopt);

  fbgemm::FloatToFused8BitRowwiseQuantizedSBFloat(
      input.data_ptr<float>(), nrows, ncols, output.data_ptr<uint8_t>());

  return output;
}

at::Tensor& _fused8bitrowwise_to_float_cpu_out(
    at::Tensor& output,
    const at::Tensor& input) {
  TENSOR_ON_CPU(input);
  TORCH_CHECK(
      input.dim() >= 2,
      "Tensor 'input' must have >= 2 dimension(s). Found ",
      input.ndimension());

  const auto input_sizes = input.sizes();
  const auto last_dim = input_sizes.size() - 1;
  const int32_t nrows = c10::size_to_dim_(last_dim, input_sizes);
  const int32_t ncols = input_sizes[last_dim];
  const int32_t output_columns = ncols - 2 * sizeof(float);

  auto output_dims = input_sizes.vec();
  output_dims[last_dim] = output_columns;
  at::native::resize_(output, output_dims, c10::nullopt);

  fbgemm::Fused8BitRowwiseQuantizedSBFloatToFloat(
      input.data_ptr<uint8_t>(), nrows, ncols, output.data_ptr<float>());

  return output;
}

namespace {

at::Tensor _float_to_fused8bitrowwise_cpu(const at::Tensor& input) {
  auto output = at::empty(
      {0},
      input.options().dtype(at::kByte)); // at::kBytes for uint8_t
  return _float_to_fused8bitrowwise_cpu_out(output, input);
}

at::Tensor _fused8bitrowwise_to_float_cpu(const at::Tensor& input) {
  auto output = at::empty(
      {0},
      input.options().dtype(at::kFloat)); // at::kBytes for uint8_t
  return _fused8bitrowwise_to_float_cpu_out(output, input);
}

} // namespace
} // namespace at

using namespace at;
TORCH_LIBRARY_FRAGMENT(fb, m) {
  m.def("FloatToFused8BitRowwiseQuantized(Tensor t) -> Tensor");
  m.def(
      "FloatToFused8BitRowwiseQuantizedOut(Tensor output, Tensor input) -> Tensor");
  m.def("Fused8BitRowwiseQuantizedToFloat(Tensor input) -> Tensor");
  m.def(
      "Fused8BitRowwiseQuantizedToFloatOut(Tensor output, Tensor input) -> Tensor");
}

TORCH_LIBRARY_IMPL(fb, CPU, m) {
  m.impl("FloatToFused8BitRowwiseQuantized", _float_to_fused8bitrowwise_cpu);
  m.impl(
      "FloatToFused8BitRowwiseQuantizedOut",
      _float_to_fused8bitrowwise_cpu_out);
  m.impl(
      "Fused8BitRowwiseQuantizedToFloat", at::_fused8bitrowwise_to_float_cpu);
  m.impl(
      "Fused8BitRowwiseQuantizedToFloatOut",
      at::_fused8bitrowwise_to_float_cpu_out);
}
