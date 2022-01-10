/*
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <fbgemm_gpu/sparse_ops.h>
#include <torch/library.h>
#include "fbgemm/QuantUtils.h"
#include "fbgemm_gpu/sparse_ops_utils.h"

using Tensor = at::Tensor;

namespace fbgemm_gpu {

template <typename input_t>
Tensor& _float_to_fused8bitrowwise_cpu_out_t(
    Tensor& output,
    const Tensor& input) {
  TENSOR_ON_CPU(input);
  TORCH_CHECK(
      input.dim() >= 2,
      "Tensor 'input' must have >= 2 dimension(s). Found ",
      input.ndimension());

  const auto input_sizes = input.sizes();
  const auto last_dim = input_sizes.size() - 1;
  const int64_t nrows = c10::size_to_dim_(last_dim, input_sizes);
  const int32_t ncols = input_sizes[last_dim];
  // Output scale and bias are always of type float.
  const int32_t output_columns = ncols + 2 * sizeof(float);

  auto output_dims = input_sizes.vec();
  output_dims[last_dim] = output_columns;
  at::native::resize_(output, output_dims, c10::nullopt);

  const auto input_data =
      (input_t*)input.data_ptr(); // input.data_ptr<input_t>(); -> Yields
                                  // unresolved data_ptr symbol.
  fbgemm::FloatOrHalfToFused8BitRowwiseQuantizedSBFloat<input_t>(
      input_data, nrows, ncols, output.data_ptr<uint8_t>());

  return output;
}

template <typename output_t>
Tensor& _fused8bitrowwise_to_float_cpu_out_t(
    Tensor& output,
    const Tensor& input) {
  TENSOR_ON_CPU(input);
  TORCH_CHECK(
      input.dim() >= 2,
      "Tensor 'input' must have >= 2 dimension(s). Found ",
      input.ndimension());

  const auto input_sizes = input.sizes();
  const auto last_dim = input_sizes.size() - 1;
  const int64_t nrows = c10::size_to_dim_(last_dim, input_sizes);
  const int32_t ncols = input_sizes[last_dim];
  const int32_t output_columns = ncols - 2 * sizeof(float);

  auto output_dims = input_sizes.vec();
  output_dims[last_dim] = output_columns;
  at::native::resize_(output, output_dims, c10::nullopt);

  auto output_data =
      (output_t*)output.data_ptr(); // output.data_ptr<output_t>(); -> Yields
                                    // unresolved data_ptr symbol.
  fbgemm::Fused8BitRowwiseQuantizedSBFloatToFloatOrHalf<output_t>(
      input.data_ptr<uint8_t>(), nrows, ncols, output_data);

  return output;
}

template <typename input_t>
Tensor _float_to_fusednbitrowwise_cpu(
    const Tensor& input,
    const int64_t bit_rate) {
  TENSOR_ON_CPU(input);
  TENSOR_NDIM_EQUALS(input, 2);

  const auto input_sizes = input.sizes();
  const int64_t nrows = input_sizes[0];
  const int32_t ncols = input_sizes[1];
  const int32_t num_elem_per_byte = 8 / bit_rate;
  TORCH_CHECK(
      ncols % (2 * num_elem_per_byte) == 0,
      "ncols needs to be multiple of 2 Bytes (half type size) to make the address aligned");
  const int64_t output_columns =
      (ncols + num_elem_per_byte - 1) / num_elem_per_byte +
      2 * sizeof(at::Half);
  auto output = at::empty(
      {nrows, output_columns},
      input.options().dtype(at::kByte)); // at::kBytes for uint8_t

  const auto input_data =
      (input_t*)input.data_ptr(); // input.data_ptr<input_t>(); -> Yields
                                  // unresolved data_ptr symbol.
  fbgemm::FloatOrHalfToFusedNBitRowwiseQuantizedSBHalf<input_t>(
      bit_rate, input_data, nrows, ncols, output.data_ptr<uint8_t>());

  return output;
}

template <typename output_t>
Tensor _fusednbitrowwise_to_float_cpu(
    const Tensor& input,
    const int64_t bit_rate) {
  TENSOR_ON_CPU(input);
  TENSOR_NDIM_EQUALS(input, 2);

  const auto input_sizes = input.sizes();
  const int64_t nrows = input_sizes[0];
  const int32_t ncols = input_sizes[1];
  const int32_t num_elem_per_byte = 8 / bit_rate;
  const int32_t output_columns =
      (ncols - 2 * sizeof(at::Half)) * num_elem_per_byte;

  Tensor output;
  if (std::is_same<output_t, float>::value) {
    output = at::empty(
        {nrows, output_columns}, // 4 = sizeof(float)
        input.options().dtype(at::kFloat));
  } else { // T = at::Half
    output = at::empty(
        {nrows, output_columns}, // 4 = sizeof(float)
        input.options().dtype(at::kHalf));
  }

  auto output_data =
      (output_t*)output.data_ptr(); // output.data_ptr<output_t>(); -> Yields
                                    // unresolved data_ptr symbol.

  fbgemm::FusedNBitRowwiseQuantizedSBHalfToFloatOrHalf<output_t>(
      bit_rate, input.data_ptr<uint8_t>(), nrows, ncols, output_data);

  return output;
}

Tensor& _fused8bitrowwise_to_float_cpu_out(
    Tensor& output,
    const Tensor& input) {
  return _fused8bitrowwise_to_float_cpu_out_t<float>(output, input);
}

Tensor& _float_to_fused8bitrowwise_cpu_out(
    Tensor& output,
    const Tensor& input) {
  return _float_to_fused8bitrowwise_cpu_out_t<float>(output, input);
}

Tensor& fused8bitrowwise_to_half_cpu_out(Tensor& output, const Tensor& input) {
  return _fused8bitrowwise_to_float_cpu_out_t<fbgemm::float16>(output, input);
}

Tensor& half_to_fused8bitrowwise_cpu_out(Tensor& output, const Tensor& input) {
  return _float_to_fused8bitrowwise_cpu_out_t<fbgemm::float16>(output, input);
}

Tensor float_to_fused8bitrowwise_cpu(const Tensor& input) {
  auto output = at::empty(
      {0},
      input.options().dtype(at::kByte)); // at::kBytes for uint8_t
  return _float_to_fused8bitrowwise_cpu_out(output, input);
}

Tensor half_to_fused8bitrowwise_cpu(const Tensor& input) {
  auto output = at::empty(
      {0},
      input.options().dtype(at::kByte)); // at::kBytes for uint8_t
  return half_to_fused8bitrowwise_cpu_out(output, input);
}

Tensor fused8bitrowwise_to_float_cpu(const Tensor& input) {
  auto output = at::empty(
      {0},
      input.options().dtype(at::kFloat)); // at::kBytes for uint8_t
  return _fused8bitrowwise_to_float_cpu_out(output, input);
}

Tensor fused8bitrowwise_to_half_cpu(const Tensor& input) {
  auto output = at::empty(
      {0},
      input.options().dtype(at::kHalf)); // at::kBytes for uint8_t
  return fused8bitrowwise_to_half_cpu_out(output, input);
}

Tensor fusednbitrowwise_to_float_cpu(
    const Tensor& input,
    const int64_t bit_rate) {
  return _fusednbitrowwise_to_float_cpu<float>(input, bit_rate);
}

Tensor fusednbitrowwise_to_half_cpu(
    const Tensor& input,
    const int64_t bit_rate) {
  return _fusednbitrowwise_to_float_cpu<fbgemm::float16>(input, bit_rate);
}

Tensor float_to_fusednbitrowwise_cpu(
    const Tensor& input,
    const int64_t bit_rate) {
  return _float_to_fusednbitrowwise_cpu<float>(input, bit_rate);
}
Tensor half_to_fusednbitrowwise_cpu(
    const Tensor& input,
    const int64_t bit_rate) {
  return _float_to_fusednbitrowwise_cpu<fbgemm::float16>(input, bit_rate);
}

} // namespace fbgemm_gpu

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.def("FloatToFused8BitRowwiseQuantized(Tensor t) -> Tensor");
  m.def(
      "FloatToFused8BitRowwiseQuantizedOut(Tensor output, Tensor input) -> Tensor");
  m.def("HalfToFused8BitRowwiseQuantized(Tensor t) -> Tensor");
  m.def("Fused8BitRowwiseQuantizedToFloat(Tensor input) -> Tensor");
  m.def("Fused8BitRowwiseQuantizedToHalf(Tensor input) -> Tensor");
  m.def(
      "Fused8BitRowwiseQuantizedToFloatOut(Tensor output, Tensor input) -> Tensor");
  m.def(
      "Fused8BitRowwiseQuantizedToFloatMixedDim(Tensor input, Tensor D_offsets, int output_dtype) -> Tensor");
  m.def(
      "FloatToFusedNBitRowwiseQuantizedSBHalf(Tensor input, int bit_rate) -> Tensor");
  m.def(
      "HalfToFusedNBitRowwiseQuantizedSBHalf(Tensor input, int bit_rate) -> Tensor");
  m.def(
      "FusedNBitRowwiseQuantizedSBHalfToFloat(Tensor input, int bit_rate) -> Tensor");
  m.def(
      "FusedNBitRowwiseQuantizedSBHalfToHalf(Tensor input, int bit_rate) -> Tensor");
}

TORCH_LIBRARY_IMPL(fbgemm, CPU, m) {
  m.impl(
      "FloatToFused8BitRowwiseQuantized",
      fbgemm_gpu::float_to_fused8bitrowwise_cpu);
  m.impl(
      "HalfToFused8BitRowwiseQuantized",
      fbgemm_gpu::half_to_fused8bitrowwise_cpu);
  m.impl(
      "FloatToFused8BitRowwiseQuantizedOut",
      fbgemm_gpu::_float_to_fused8bitrowwise_cpu_out);
  m.impl(
      "Fused8BitRowwiseQuantizedToFloat",
      fbgemm_gpu::fused8bitrowwise_to_float_cpu);
  m.impl(
      "Fused8BitRowwiseQuantizedToFloatOut",
      fbgemm_gpu::_fused8bitrowwise_to_float_cpu_out);
  m.impl(
      "Fused8BitRowwiseQuantizedToHalf",
      fbgemm_gpu::fused8bitrowwise_to_half_cpu);
  m.impl(
      "FloatToFusedNBitRowwiseQuantizedSBHalf",
      fbgemm_gpu::float_to_fusednbitrowwise_cpu);
  m.impl(
      "FusedNBitRowwiseQuantizedSBHalfToFloat",
      fbgemm_gpu::fusednbitrowwise_to_float_cpu);
  m.impl(
      "FusedNBitRowwiseQuantizedSBHalfToHalf",
      fbgemm_gpu::fusednbitrowwise_to_half_cpu);
  m.impl(
      "HalfToFusedNBitRowwiseQuantizedSBHalf",
      fbgemm_gpu::half_to_fusednbitrowwise_cpu);
}
