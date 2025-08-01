/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <fbgemm_gpu/sparse_ops.h>
#include <torch/library.h>
#include "fbgemm/QuantUtils.h"
#include "fbgemm_gpu/embedding_common.h"
#include "fbgemm_gpu/quantize_ops_utils.h"
#include "fbgemm_gpu/utils/dispatch_macros.h"
#include "fbgemm_gpu/utils/ops_utils.h"
#include "fbgemm_gpu/utils/tensor_utils.h"

using Tensor = at::Tensor;

/// @defgroup quantize-data-cpu Quantize Data CPU Operators
/// The following are CPU Operators
///

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
  at::native::resize_(output, output_dims, std::nullopt);

  const auto input_data = static_cast<input_t*>(
      input.data_ptr()); // input.data_ptr<input_t>(); -> Yields
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
  at::native::resize_(output, output_dims, std::nullopt);

  auto output_data = static_cast<output_t*>(
      output.data_ptr()); // output.data_ptr<output_t>(); -> Yields
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

  const auto input_data = static_cast<input_t*>(
      input.data_ptr()); // input.data_ptr<input_t>(); -> Yields
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
  // Here we want the number of bytes in a row
  const int32_t ncols = nbit_elems_to_bytes(input);
  const int32_t num_elem_per_byte = 8 / bit_rate;
  const int32_t output_columns =
      (ncols - 2 * sizeof(at::Half)) * num_elem_per_byte;

  Tensor output;
  if constexpr (std::is_same_v<output_t, float>) {
    output = at::empty(
        {nrows, output_columns}, // 4 = sizeof(float)
        input.options().dtype(at::kFloat));
  } else { // T = at::Half
    output = at::empty(
        {nrows, output_columns}, // 4 = sizeof(float)
        input.options().dtype(at::kHalf));
  }

  auto output_data = static_cast<output_t*>(
      output.data_ptr()); // output.data_ptr<output_t>(); -> Yields
                          // unresolved data_ptr symbol.

  fbgemm::FusedNBitRowwiseQuantizedSBHalfToFloatOrHalf<output_t>(
      bit_rate, input.data_ptr<uint8_t>(), nrows, ncols, output_data);

  return output;
}

// Both float16 and bfloat16 are of same type uint16_t
template <typename output_t>
Tensor _fusednbitrowwise_sbfront_to_float_or_half_cpu(
    const Tensor& input,
    const int64_t bit_rate) {
  TENSOR_ON_CPU(input);
  TENSOR_NDIM_EQUALS(input, 2);

  const auto input_sizes = input.sizes();
  const int64_t nrows = input_sizes[0];
  // Here we want the number of bytes in a row
  const int32_t ncols = nbit_elems_to_bytes(input);
  const int32_t num_elem_per_byte = 8 / bit_rate;
  const int32_t output_columns =
      (ncols - 2 * sizeof(at::Half)) * num_elem_per_byte;

  Tensor output;
  if constexpr (std::is_same_v<output_t, float>) {
    output = at::empty(
        {nrows, output_columns}, // 4 = sizeof(float)
        input.options().dtype(at::kFloat));
  } else if constexpr (std::is_same_v<output_t, at::Half>) {
    output = at::empty(
        {nrows, output_columns}, // 2 = sizeof(half)
        input.options().dtype(at::kHalf));
  } else if constexpr (std::is_same_v<output_t, at::BFloat16>) {
    output = at::empty(
        {nrows, output_columns}, // 2 = sizeof(half)
        input.options().dtype(at::kBFloat16));
  } else {
    TORCH_CHECK(
        false,
        "Unsupported output dtype for _fusednbitrowwise_sbfront_to_float_or_half_cpu");
  }

  using output_ty = std::
      conditional_t<std::is_same_v<output_t, float>, float, fbgemm::float16>;
  output_ty* output_data = static_cast<output_ty*>(
      output.data_ptr()); // output.data_ptr<output_t>(); -> Yields
                          // unresolved data_ptr symbol.

  constexpr bool is_uint16_t_of_type_bf16 =
      std::is_same_v<output_t, at::BFloat16>;
  fbgemm::FusedNBitRowwiseQuantizedSBHalfToFloatOrHalfRef<
      output_ty,
      is_uint16_t_of_type_bf16>(
      bit_rate,
      input.data_ptr<uint8_t>(),
      nrows,
      ncols,
      output_data,
      /*scale_bias_last=*/false);

  return output;
}

/// @ingroup quantize-data-cpu
///
Tensor& _fused8bitrowwise_to_float_cpu_out(
    Tensor& output,
    const Tensor& input) {
  return _fused8bitrowwise_to_float_cpu_out_t<float>(output, input);
}

Tensor& fused8bitrowwise_to_half_cpu_out(Tensor& output, const Tensor& input) {
  return _fused8bitrowwise_to_float_cpu_out_t<fbgemm::float16>(output, input);
}

/// @ingroup quantize-data-cpu
///
Tensor& _float_to_fused8bitrowwise_cpu_out(
    Tensor& output,
    const Tensor& input) {
  return _float_to_fused8bitrowwise_cpu_out_t<float>(output, input);
}

Tensor& _half_to_fused8bitrowwise_cpu_out(Tensor& output, const Tensor& input) {
  return _float_to_fused8bitrowwise_cpu_out_t<fbgemm::float16>(output, input);
}

/// @ingroup quantize-data-cpu
///
Tensor float_to_fused8bitrowwise_cpu(const Tensor& input) {
  auto output = at::empty(
      {0},
      input.options().dtype(at::kByte)); // at::kBytes for uint8_t
  return _float_to_fused8bitrowwise_cpu_out(output, input);
}

/// @ingroup quantize-data-cpu
///
Tensor half_to_fused8bitrowwise_cpu(const Tensor& input) {
  auto output = at::empty(
      {0},
      input.options().dtype(at::kByte)); // at::kBytes for uint8_t
  return _half_to_fused8bitrowwise_cpu_out(output, input);
}

/// @ingroup quantize-data-cpu
///
Tensor float_or_half_to_fused8bitrowwise_cpu(const Tensor& input) {
  auto output = at::empty(
      {0},
      input.options().dtype(at::kByte)); // at::kBytes for uint8_t
  FBGEMM_DISPATCH_FLOAT_AND_HALF(
      input.scalar_type(), "float_or_half_to_fused8bitrowwise_cpu", [&] {
        if constexpr (std::is_same_v<scalar_t, float>) {
          _float_to_fused8bitrowwise_cpu_out(output, input);
        } else { // scalar_t = at::Half
          _half_to_fused8bitrowwise_cpu_out(output, input);
        }
      });
  return output;
}

/// @ingroup quantize-data-cpu
///
Tensor fused8bitrowwise_to_float_cpu(const Tensor& input) {
  auto output = at::empty({0}, input.options().dtype(at::kFloat));
  return _fused8bitrowwise_to_float_cpu_out(output, input);
}

/// @ingroup quantize-data-cpu
///
Tensor fused8bitrowwise_to_half_cpu(const Tensor& input) {
  auto output = at::empty({0}, input.options().dtype(at::kHalf));
  return fused8bitrowwise_to_half_cpu_out(output, input);
}

/// @ingroup quantize-data-cpu
///
Tensor fused8bitrowwise_to_float_or_half_cpu(
    const Tensor& input,
    const int64_t output_dtype,
    [[maybe_unused]] const bool scale_bias_last,
    [[maybe_unused]] const bool quant_padding_float_type) {
  Tensor output;
  SparseType output_sparse_dtype = static_cast<SparseType>(output_dtype);
  switch (output_sparse_dtype) {
    case SparseType::FP32:
      output = at::empty({0}, input.options().dtype(at::kFloat));

      output = _fused8bitrowwise_to_float_cpu_out(output, input);

      break;
    case SparseType::FP16:
      output = at::empty({0}, input.options().dtype(at::kHalf));
      output = fused8bitrowwise_to_half_cpu_out(output, input);
      break;
    default:
      TORCH_CHECK(false);
  }

  return output;
}
// dummy cpu code for gpu fp8_rowwise conversions
/// @ingroup quantize-data-cpu
///
Tensor float_to_FP8rowwise_cpu(const Tensor& input, bool /*forward*/) {
  TORCH_CHECK(false, "fp8 is not supported by CPU");
  return input;
}

/// @ingroup quantize-data-cpu
///
Tensor FP8rowwise_to_float_cpu(
    const Tensor& input,
    bool /*forward*/,
    const int64_t /*output_dtype*/) {
  TORCH_CHECK(false, "fp8 is not supported by CPU");
  return input;
}

/// @ingroup quantize-data-cpu
///
Tensor fusednbitrowwise_to_float_cpu(
    const Tensor& input,
    const int64_t bit_rate) {
  return _fusednbitrowwise_to_float_cpu<float>(input, bit_rate);
}

/// @ingroup quantize-data-cpu
/// @brief Dequantize int4/int2 rows with scale and bias stored in the front
/// into float32/float16/Bfloat16.
/// @param input Tensor of int4/int2 rows with scale and bias stored in the
/// front.
/// @param bit_rate Bit rate of each element. Should be 4 or 2.
/// @return Tensor of float32, holding dequantized numbers.
///
/// Dequantize int4/int2 rows with scale and bias stored in the front into
/// float32. The input tensor should have torch.quint4x2 or torch.quint2x4 dtype
/// and QuantizedCPU backend. This operator is only recommended for testing
/// purpose because its kernel is reference implementation and not optimized.
Tensor fusednbitrowwise_sbfront_to_float_cpu(
    const Tensor& input,
    const int64_t bit_rate,
    const int64_t output_dtype) {
  SparseType output_sparse_dtype = static_cast<SparseType>(output_dtype);
  switch (output_sparse_dtype) {
    case SparseType::FP32:
      return _fusednbitrowwise_sbfront_to_float_or_half_cpu<float>(
          input, bit_rate);
      break;
    case SparseType::FP16:
      return _fusednbitrowwise_sbfront_to_float_or_half_cpu<at::Half>(
          input, bit_rate);
      break;
    case SparseType::BF16:
      return _fusednbitrowwise_sbfront_to_float_or_half_cpu<at::BFloat16>(
          input, bit_rate);
      break;
    default:
      TORCH_CHECK(false);
  }
}

/// @ingroup quantize-data-cpu
///
Tensor fusednbitrowwise_to_half_cpu(
    const Tensor& input,
    const int64_t bit_rate) {
  return _fusednbitrowwise_to_float_cpu<fbgemm::float16>(input, bit_rate);
}

/// @ingroup quantize-data-cpu
///
Tensor fusednbitrowwise_to_float_or_half_cpu(
    const Tensor& input,
    const int64_t bit_rate,
    const int64_t output_dtype,
    [[maybe_unused]] const bool scale_bias_last) {
  Tensor output;

  SparseType output_sparse_dtype = static_cast<SparseType>(output_dtype);
  switch (output_sparse_dtype) {
    case SparseType::FP32:
      output = _fusednbitrowwise_to_float_cpu<float>(input, bit_rate);

      break;
    case SparseType::FP16:
      output = _fusednbitrowwise_to_float_cpu<fbgemm::float16>(input, bit_rate);
      break;
    default:
      TORCH_CHECK(false);
  }

  return output;
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

Tensor float_or_half_to_fusednbitrowwise_cpu(
    const Tensor& input,
    const int64_t bit_rate) {
  Tensor output;
  FBGEMM_DISPATCH_FLOAT_AND_HALF(
      input.scalar_type(), "float_or_half_to_fusednbitrowwise_cpu", [&] {
        if constexpr (std::is_same_v<scalar_t, float>) {
          output = _float_to_fusednbitrowwise_cpu<float>(input, bit_rate);
        } else { // scalar_t = at::Half
          output =
              _float_to_fusednbitrowwise_cpu<fbgemm::float16>(input, bit_rate);
        }
      });
  return output;
}

/// @ingroup quantize-data-cpu
///
void FloatToFP8Quantized_ref(
    const float* const input,
    const size_t nrows,
    const size_t ncols,
    uint8_t* const output,
    const int ebits,
    const int exponent_bias,
    const double max_pos) {
  for (const auto row : c10::irange(nrows)) {
    const float* input_row = input + row * ncols;
    uint8_t* output_row = output + row * ncols;

    for (const auto col : c10::irange(ncols)) {
      output_row[col] =
          float_to_hfp8(input_row[col], ebits, exponent_bias, max_pos);
    }
  }
}

/// @ingroup quantize-data-cpu
///
void FP8QuantizedToFloat_ref(
    const uint8_t* const input,
    const size_t nrows,
    const size_t ncols,
    float* const output,
    const int ebits,
    const int exponent_bias) {
  const int32_t output_columns = ncols;

  for (const auto row : c10::irange(nrows)) {
    const uint8_t* input_row = input + row * ncols;
    float* output_row = output + row * output_columns;

    for (const auto col : c10::irange(ncols)) {
      output_row[col] = hfp8_to_float(input_row[col], ebits, exponent_bias);
    }
  }
}

at::Tensor _float_to_hfp8_cpu(
    const at::Tensor& input,
    const int64_t ebits,
    const int64_t exponent_bias,
    const double max_pos) {
  TENSOR_ON_CPU(input);
  TENSOR_NDIM_EQUALS(input, 2);

  const auto input_sizes = input.sizes();
  const int32_t nrows = input_sizes[0];
  const int32_t ncols = input_sizes[1];
  auto output = at::empty({nrows, ncols}, input.options().dtype(at::kByte));

  FloatToFP8Quantized_ref(
      input.data_ptr<float>(),
      nrows,
      ncols,
      output.data_ptr<uint8_t>(),
      ebits,
      exponent_bias,
      max_pos);

  return output;
}

at::Tensor _hfp8_to_float_cpu(
    const at::Tensor& input,
    const int64_t ebits,
    const int64_t exponent_bias) {
  TENSOR_ON_CPU(input);
  TENSOR_NDIM_EQUALS(input, 2);

  const auto input_sizes = input.sizes();
  const int32_t nrows = input_sizes[0];
  const int32_t ncols = input_sizes[1];
  const int32_t output_columns = ncols;
  auto output = at::empty(
      {nrows, output_columns}, // 4 = sizeof(float)
      input.options().dtype(at::kFloat)); //

  FP8QuantizedToFloat_ref(
      input.data_ptr<uint8_t>(),
      nrows,
      ncols,
      output.data_ptr<float>(),
      ebits,
      exponent_bias);

  return output;
}
} // namespace fbgemm_gpu

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
#ifdef HAS_IMPL_ABSTRACT_PYSTUB
  m.impl_abstract_pystub(
      "fbgemm_gpu.sparse_ops",
      "//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_py");
#endif

  m.set_python_module("fbgemm_gpu.sparse_ops");

  m.def("FloatToFused8BitRowwiseQuantized(Tensor t) -> Tensor");
  m.def(
      "FloatToFP8RowwiseQuantized(Tensor t, bool forward) -> Tensor",
      {PT2_COMPLIANT_TAG});
  m.def(
      "FloatToPaddedFP8RowwiseQuantized(Tensor t, bool forward, int row_dim) -> Tensor");
  m.def(
      "FloatToFused8BitRowwiseQuantizedOut(Tensor output, Tensor input) -> Tensor");
  m.def("HalfToFused8BitRowwiseQuantized(Tensor t) -> Tensor");
  m.def("FloatOrHalfToFused8BitRowwiseQuantized(Tensor t) -> Tensor");
  m.def("Fused8BitRowwiseQuantizedToFloat(Tensor input) -> Tensor");
  m.def(
      "FP8RowwiseQuantizedToFloat(Tensor input, bool forward, int output_dtype=0) -> Tensor",
      {PT2_COMPLIANT_TAG});
  m.def("Fused8BitRowwiseQuantizedToHalf(Tensor input) -> Tensor");
  m.def(
      "Fused8BitRowwiseQuantizedToFloatOrHalf(Tensor input, int output_dtype=0, bool scale_bias_last=True, bool quant_padding_float_type=True) -> Tensor");
  m.def(
      "Fused8BitRowwiseQuantizedToFloatOut(Tensor output, Tensor input) -> Tensor");
  m.def(
      "Fused8BitRowwiseQuantizedToFloatMixedDim(Tensor input, Tensor D_offsets, int output_dtype) -> Tensor");
  m.def(
      "FloatToFusedNBitRowwiseQuantizedSBHalf(Tensor input, int bit_rate) -> Tensor");
  m.def(
      "HalfToFusedNBitRowwiseQuantizedSBHalf(Tensor input, int bit_rate) -> Tensor");
  m.def(
      "FloatOrHalfToFusedNBitRowwiseQuantizedSBHalf(Tensor input, int bit_rate) -> Tensor");
  m.def(
      "FusedNBitRowwiseQuantizedSBHalfToFloat(Tensor input, int bit_rate) -> Tensor");
  m.def(
      "FusedNBitRowwiseQuantizedSBHalfFrontToFloatOrHalf(Tensor input, int bit_rate, int output_dtype) -> Tensor");
  m.def(
      "FusedNBitRowwiseQuantizedSBHalfToHalf(Tensor input, int bit_rate) -> Tensor");
  m.def(
      "FusedNBitRowwiseQuantizedSBHalfToFloatOrHalf(Tensor input, int bit_rate, int output_dtype=0, bool scale_bias_last=True) -> Tensor");
  m.def(
      "FloatToHFP8Quantized(Tensor input, int ebits, int exponent_bias, float max_pos) -> Tensor");
  m.def(
      "HFP8QuantizedToFloat(Tensor input, int ebits, int exponent_bias) -> Tensor");
  m.def(
      "FloatToMSFPQuantized(Tensor input, int bounding_box_size, int ebits, int mbits, int bias, float min_pos, float max_pos) -> Tensor");
  m.def(
      "MSFPQuantizedToFloat(Tensor input, int ebits, int mbits, int bias) -> Tensor");
  m.def(
      "PaddedFP8RowwiseQuantizedToFloat(Tensor input, bool forward, int row_dim, int output_last_dim=-1, int output_dtype=0) -> Tensor");
  m.def(
      "quantize_mx_cuda(Tensor input, int scale_bits, int elem_ebits, int elem_mbits, float elem_max_norm, int mx_group_size, bool flush_fp32_subnorms=False, int rounding_mode=0) -> Tensor");
  m.def("dequantize_mx_cuda(Tensor input, int mx_group_size) -> Tensor");
}

TORCH_LIBRARY_IMPL(fbgemm, QuantizedCPU, m) {
  DISPATCH_TO_QUANTIZED_CPU(
      "FusedNBitRowwiseQuantizedSBHalfFrontToFloatOrHalf",
      fbgemm_gpu::fusednbitrowwise_sbfront_to_float_cpu);
}

TORCH_LIBRARY_IMPL(fbgemm, CPU, m) {
  DISPATCH_TO_CPU(
      "FloatToFused8BitRowwiseQuantized",
      fbgemm_gpu::float_to_fused8bitrowwise_cpu);
  DISPATCH_TO_CPU(
      "FloatToFP8RowwiseQuantized", fbgemm_gpu::float_to_FP8rowwise_cpu);
  DISPATCH_TO_CPU(
      "HalfToFused8BitRowwiseQuantized",
      fbgemm_gpu::half_to_fused8bitrowwise_cpu);
  DISPATCH_TO_CPU(
      "FloatOrHalfToFused8BitRowwiseQuantized",
      fbgemm_gpu::float_or_half_to_fused8bitrowwise_cpu);
  DISPATCH_TO_CPU(
      "FloatToFused8BitRowwiseQuantizedOut",
      fbgemm_gpu::_float_to_fused8bitrowwise_cpu_out);
  DISPATCH_TO_CPU(
      "Fused8BitRowwiseQuantizedToFloat",
      fbgemm_gpu::fused8bitrowwise_to_float_cpu);
  DISPATCH_TO_CPU(
      "FP8RowwiseQuantizedToFloat", fbgemm_gpu::FP8rowwise_to_float_cpu);
  DISPATCH_TO_CPU(
      "Fused8BitRowwiseQuantizedToHalf",
      fbgemm_gpu::fused8bitrowwise_to_half_cpu);
  DISPATCH_TO_CPU(
      "Fused8BitRowwiseQuantizedToFloatOrHalf",
      fbgemm_gpu::fused8bitrowwise_to_float_or_half_cpu);
  DISPATCH_TO_CPU(
      "Fused8BitRowwiseQuantizedToFloatOut",
      fbgemm_gpu::_fused8bitrowwise_to_float_cpu_out);
  DISPATCH_TO_CPU(
      "FloatToFusedNBitRowwiseQuantizedSBHalf",
      fbgemm_gpu::float_to_fusednbitrowwise_cpu);
  DISPATCH_TO_CPU(
      "HalfToFusedNBitRowwiseQuantizedSBHalf",
      fbgemm_gpu::half_to_fusednbitrowwise_cpu);
  DISPATCH_TO_CPU(
      "FloatOrHalfToFusedNBitRowwiseQuantizedSBHalf",
      fbgemm_gpu::float_or_half_to_fusednbitrowwise_cpu);
  DISPATCH_TO_CPU(
      "FusedNBitRowwiseQuantizedSBHalfToFloat",
      fbgemm_gpu::fusednbitrowwise_to_float_cpu);
  DISPATCH_TO_CPU(
      "FusedNBitRowwiseQuantizedSBHalfToHalf",
      fbgemm_gpu::fusednbitrowwise_to_half_cpu);
  DISPATCH_TO_CPU(
      "FusedNBitRowwiseQuantizedSBHalfToFloatOrHalf",
      fbgemm_gpu::fusednbitrowwise_to_float_or_half_cpu);
  DISPATCH_TO_CPU("FloatToHFP8Quantized", fbgemm_gpu::_float_to_hfp8_cpu);
  DISPATCH_TO_CPU("HFP8QuantizedToFloat", fbgemm_gpu::_hfp8_to_float_cpu);
}
