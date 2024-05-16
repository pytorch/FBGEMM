/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <torch/library.h>

#include "c10/core/ScalarType.h"

#include <ATen/cuda/CUDAEvent.h>
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <vector>
#include "c10/util/Exception.h"

namespace fbgemm_gpu {

// SmoothQuant kernels
at::Tensor
i8i8bf16(at::Tensor XQ, at::Tensor WQ, double scale, int64_t split_k);
at::Tensor i8i8bf16_dynamic(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor scale,
    int64_t split_k);

at::Tensor silu_mul_quantize_i8(at::Tensor X1, at::Tensor X2, double scale);

// Cutlass kernel
at::Tensor f8f8bf16(
    at::Tensor XQ,
    at::Tensor WQ,
    double scale,
    bool use_fast_accum = true);
at::Tensor f8f8bf16_rowwise(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    c10::optional<at::Tensor> bias = c10::nullopt,
    bool use_fast_accum = true);
at::Tensor f8f8bf16_cublas(
    at::Tensor A,
    at::Tensor B,
    at::Tensor Ainvs,
    at::Tensor Binvs,
    bool use_fast_accum,
    c10::optional<at::Tensor> output);
at::Tensor f8i4bf16_rowwise(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor w_zp);

at::Tensor per_tensor_quantize_i8(at::Tensor X, double scale);
std::tuple<at::Tensor, at::Tensor> per_tensor_dynamic_quantize_i8(at::Tensor X);

std::tuple<at::Tensor, double> quantize_fp8_per_tensor(
    at::Tensor input,
    c10::optional<at::Tensor> bs, // batch size
    c10::optional<at::Tensor> scale_ub); // scale upperbound

std::tuple<at::Tensor, at::Tensor> quantize_fp8_per_tensor_tensor_scale(
    at::Tensor input,
    c10::optional<at::Tensor> bs, // batch size
    c10::optional<at::Tensor> scale_ub); // scale upperbound

std::vector<at::Tensor> quantize_fp8_per_row(
    at::Tensor input,
    c10::optional<at::Tensor> bs, // batch size
    c10::optional<at::Tensor> scale_ub, // scale upperbound
    c10::optional<c10::ScalarType> output_dtype); // output dtype

#if CUDART_VERSION >= 12000
std::vector<at::Tensor> quantize_fp8_per_col(
    at::Tensor input,
    c10::optional<at::Tensor> bs, // batch size
    c10::optional<at::Tensor> scale_ub); // scale upperbound
#endif

at::Tensor quantize_fp8_per_tensor_fixed_scale(
    at::Tensor input,
    at::Tensor scale,
    c10::optional<at::Tensor> bs);

at::Tensor get_fp8_per_tensor_scale(
    at::Tensor input,
    c10::optional<at::Tensor> bs,
    c10::optional<at::Tensor> scale_ub); // scale upperbound

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
#ifndef USE_ROCM
  // TODO: on AMD this throws "Undefined symbol" when loading
  // quantize_ops with
  // torch.ops.load_library, similar to below for quantize_fp8_per_tensor
  m.def("i8i8bf16(Tensor XQ, Tensor WQ, float scale, int split_k=1) -> Tensor");

  m.def(
      "f8f8bf16(Tensor XQ, Tensor WQ, float scale, bool use_fast_accum=True) -> Tensor");

  m.def(
      "f8f8bf16_rowwise(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias=None, bool use_fast_accum=True) -> Tensor");

  m.def(
      "f8f8bf16_cublas(Tensor A, Tensor B, Tensor Ainvs, Tensor Binvs, bool use_fast_accum=True, Tensor(a!)? output=None) -> Tensor");

  m.def(
      "f8i4bf16_rowwise(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor w_zp) -> Tensor");
  m.impl("f8i4bf16_rowwise", f8i4bf16_rowwise);

  m.def(
      "i8i8bf16_dynamic(Tensor XQ, Tensor WQ, Tensor scale, int split_k=1) -> Tensor");
  m.impl("i8i8bf16_dynamic", i8i8bf16_dynamic);
#endif

  m.def("per_tensor_quantize_i8(Tensor X, float scale) -> Tensor");
  m.impl("per_tensor_quantize_i8", per_tensor_quantize_i8);
  m.def("per_tensor_dynamic_quantize_i8(Tensor X) -> (Tensor, Tensor)");
  m.impl("per_tensor_dynamic_quantize_i8", per_tensor_dynamic_quantize_i8);

  m.def("silu_mul_quantize_i8(Tensor X1, Tensor X2, float scale) -> Tensor");
  m.impl("silu_mul_quantize_i8", silu_mul_quantize_i8);

#ifndef USE_ROCM
  // TODO: On AMD this throws "undefined symbol:
  // _ZN8facebook6gen_ai13llm_inference23quantize_fp8_per_tensorEN2at6TensorEN3c108optionalIS3_EE"
  // i.e. facebook::gen_ai::llm_inference::quantize_fp8_per_tensor(at::Tensor,
  // c10::optional<at::Tensor>) when loading
  // quantize_ops with
  // torch.ops.load_library
  m.def(
      "quantize_fp8_per_tensor(Tensor input, Tensor? bs=None, Tensor? scale_ub=None) -> (Tensor, float)");
  m.impl("quantize_fp8_per_tensor", quantize_fp8_per_tensor);
  m.def(
      "quantize_fp8_per_tensor_tensor_scale(Tensor input, Tensor? bs=None, Tensor? scale_ub=None) -> (Tensor, Tensor)");
  m.impl(
      "quantize_fp8_per_tensor_tensor_scale",
      quantize_fp8_per_tensor_tensor_scale);
  m.def(
      "quantize_fp8_per_row(Tensor input, Tensor? bs=None, Tensor? scale_ub=None, ScalarType? output_dtype=None) -> Tensor[]");
  m.impl("quantize_fp8_per_row", quantize_fp8_per_row);

#if CUDART_VERSION >= 12000
  m.def(
      "quantize_fp8_per_col(Tensor input, Tensor? bs=None, Tensor? scale_ub=None) -> Tensor[]");
  m.impl("quantize_fp8_per_col", quantize_fp8_per_col);
#endif

  m.def(
      "get_fp8_per_tensor_scale(Tensor input, Tensor? bs=None, Tensor? scale_ub=None) -> Tensor");
  m.impl("get_fp8_per_tensor_scale", get_fp8_per_tensor_scale);

  m.def(
      "quantize_fp8_per_tensor_fixed_scale(Tensor input, Tensor scale, Tensor? bs=None) -> Tensor");
  m.impl(
      "quantize_fp8_per_tensor_fixed_scale",
      quantize_fp8_per_tensor_fixed_scale);
#endif
}

#ifndef USE_ROCM
TORCH_LIBRARY_IMPL(fbgemm, CUDA, m) {
  m.impl("i8i8bf16", i8i8bf16);
  m.impl("f8f8bf16", f8f8bf16);
  m.impl("f8f8bf16_rowwise", f8f8bf16_rowwise);
  m.impl("quantize_fp8_per_tensor", quantize_fp8_per_tensor);
  m.impl(
      "quantize_fp8_per_tensor_tensor_scale",
      quantize_fp8_per_tensor_tensor_scale);
  m.impl("f8f8bf16_cublas", f8f8bf16_cublas);
}

at::Tensor i8i8bf16_meta(
    at::Tensor XQ, // INT8
    at::Tensor WQ, // INT8
    double scale,
    int64_t split_k) {
  int M = XQ.size(0);
  int N = WQ.size(0);
  auto Y = at::empty({M, N}, XQ.options().dtype(at::kBFloat16));
  return Y;
}

at::Tensor f8f8bf16_rowwise_meta(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    at::Tensor x_scale,
    at::Tensor w_scale,
    c10::optional<at::Tensor> bias = c10::nullopt,
    bool use_fast_accum = true) {
  int M = XQ.size(0);
  int N = WQ.size(0);
  auto Y = at::empty({M, N}, XQ.options().dtype(at::kBFloat16));
  return Y;
}

std::tuple<at::Tensor, double> quantize_fp8_per_tensor_meta(
    at::Tensor X,
    c10::optional<at::Tensor> bs,
    c10::optional<at::Tensor> scale_ub) {
  auto Y = at::empty_like(X, X.options().dtype(at::kFloat8_e4m3fn));
  auto scale = 0.0;
  return std::tuple<at::Tensor, double>{Y, scale};
}

std::tuple<at::Tensor, at::Tensor> quantize_fp8_per_tensor_tensor_scale_meta(
    at::Tensor X,
    c10::optional<at::Tensor> bs,
    c10::optional<at::Tensor> scale_ub) {
  auto Y = at::empty_like(X, X.options().dtype(at::kFloat8_e4m3fn));
  auto scale = at::empty({}, X.options().dtype(at::kBFloat16));
  return std::tuple<at::Tensor, at::Tensor>{Y, scale};
}

at::Tensor f8f8bf16_cublas_meta(
    at::Tensor X,
    at::Tensor W,
    at::Tensor x_scale,
    at::Tensor w_scale,
    bool use_fast_accum = true,
    c10::optional<at::Tensor> output = c10::nullopt) {
  const at::SymInt M = X.sym_size(0);
  const at::SymInt N = W.sym_size(0);
  auto Y = at::empty_symint({M, N}, X.options().dtype(at::kBFloat16));
  return Y;
}

at::Tensor f8f8bf16_meta(
    at::Tensor X,
    at::Tensor W,
    double scale,
    bool use_fast_accum = true) {
  const at::SymInt M = X.sym_size(0);
  const at::SymInt N = W.sym_size(0);
  auto Y = at::empty_symint({M, N}, X.options().dtype(at::kBFloat16));
  return Y;
}

at::Tensor f8i4bf16_rowwise_meta(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor w_zp) {
  int M = XQ.size(0);
  int N = WQ.size(0);
  auto Y = at::empty({M, N}, XQ.options().dtype(at::kBFloat16));
  return Y;
}

TORCH_LIBRARY_IMPL(fbgemm, Meta, m) {
  m.impl("i8i8bf16", i8i8bf16_meta);
  m.impl("f8f8bf16", f8f8bf16_meta);
  m.impl("f8f8bf16_rowwise", f8f8bf16_rowwise_meta);
  m.impl("quantize_fp8_per_tensor", quantize_fp8_per_tensor_meta);
  m.impl(
      "quantize_fp8_per_tensor_tensor_scale",
      quantize_fp8_per_tensor_tensor_scale_meta);
  m.impl("f8f8bf16_cublas", f8f8bf16_cublas_meta);
  m.impl("f8i4bf16_rowwise", f8i4bf16_rowwise_meta);
}

#endif

} // namespace fbgemm_gpu
