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
    at::Tensor scale,
    bool use_fast_accum = true);
at::Tensor f8f8bf16_tensorwise(
    at::Tensor XQ,
    at::Tensor WQ,
    double scale,
    bool use_fast_accum = true);
at::Tensor f8f8bf16_rowwise(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias = c10::nullopt,
    bool use_fast_accum = true,
    std::optional<at::Tensor> output = c10::nullopt);
at::Tensor f8f8bf16_blockwise(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    int64_t block_m = 256,
    int64_t block_n = 256,
    int64_t block_k = 256);
at::Tensor f8f8bf16_cublas(
    at::Tensor A,
    at::Tensor B,
    std::optional<at::Tensor> Ainvs = c10::nullopt,
    std::optional<at::Tensor> Binvs = c10::nullopt,
    bool use_fast_accum = true,
    std::optional<at::Tensor> output = c10::nullopt);
at::Tensor f8i4bf16_rowwise(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor w_zp);
at::Tensor bf16i4bf16_rowwise(
    at::Tensor X,
    at::Tensor WQ,
    at::Tensor w_scale,
    at::Tensor w_zp);

at::Tensor per_tensor_quantize_i8(at::Tensor X, double scale);
std::tuple<at::Tensor, at::Tensor> per_tensor_dynamic_quantize_i8(at::Tensor X);

std::vector<at::Tensor> quantize_fp8_per_tensor(
    at::Tensor input,
    std::optional<at::Tensor> bs, // batch size
    std::optional<at::Tensor> scale_ub, // scale upperbound
    const bool stochastic_rounding); // whether apply stochastic rounding

std::vector<at::Tensor> quantize_fp8_per_row(
    at::Tensor input,
    std::optional<at::Tensor> bs, // batch size
    std::optional<at::Tensor> scale_ub, // scale upperbound
    std::optional<c10::ScalarType> output_dtype, // output dtype
    bool stochastic_rounding); // whether apply stochastic rounding

#if CUDART_VERSION >= 12000
std::vector<at::Tensor> quantize_fp8_per_col(
    at::Tensor input,
    std::optional<at::Tensor> bs, // batch size
    std::optional<at::Tensor> scale_ub); // scale upperbound
#endif

at::Tensor quantize_fp8_per_tensor_fixed_scale(
    at::Tensor input,
    at::Tensor scale,
    std::optional<at::Tensor> bs,
    bool stochatic_rounding);

at::Tensor get_fp8_per_tensor_scale(
    at::Tensor input,
    std::optional<at::Tensor> bs,
    std::optional<at::Tensor> scale_ub); // scale upperbound

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
#ifndef USE_ROCM
  // TODO: on AMD this throws "Undefined symbol" when loading
  // quantize_ops with
  // torch.ops.load_library, similar to below for quantize_fp8_per_tensor
  m.def("i8i8bf16(Tensor XQ, Tensor WQ, float scale, int split_k=1) -> Tensor");

  m.def(
      "f8f8bf16(Tensor XQ, Tensor WQ, Tensor scale, bool use_fast_accum=True) -> Tensor");

  m.def(
      "f8f8bf16_cublas(Tensor A, Tensor B, Tensor? Ainvs=None, Tensor? Binvs=None, bool use_fast_accum=True, Tensor(a!)? output=None) -> Tensor");

  m.def(
      "f8i4bf16_rowwise(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor w_zp) -> Tensor");
  m.impl("f8i4bf16_rowwise", f8i4bf16_rowwise);

  m.def(
      "bf16i4bf16_rowwise(Tensor X, Tensor WQ, Tensor w_scale, Tensor w_zp) -> Tensor");
  m.impl("bf16i4bf16_rowwise", bf16i4bf16_rowwise);

  m.def(
      "i8i8bf16_dynamic(Tensor XQ, Tensor WQ, Tensor scale, int split_k=1) -> Tensor");
  m.impl("i8i8bf16_dynamic", i8i8bf16_dynamic);
#endif
  m.def(
      "f8f8bf16_blockwise(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, int block_m=256, int block_n=256, int block_k=256) -> Tensor");
  m.def(
      "f8f8bf16_rowwise(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias=None, bool use_fast_accum=True, Tensor(a!)? output=None) -> Tensor");
  m.def(
      "f8f8bf16_tensorwise(Tensor XQ, Tensor WQ, float scale, bool use_fast_accum=True) -> Tensor");
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
  // std::optional<at::Tensor>) when loading
  // quantize_ops with
  // torch.ops.load_library
  m.def(
      "quantize_fp8_per_tensor(Tensor input, Tensor? bs=None, Tensor? scale_ub=None, bool stochastic_rounding=False) -> Tensor[]");
  m.impl("quantize_fp8_per_tensor", quantize_fp8_per_tensor);
  m.def(
      "quantize_fp8_per_row(Tensor input, Tensor? bs=None, Tensor? scale_ub=None, ScalarType? output_dtype=None, bool stochastic_rounding=False) -> Tensor[]");
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
      "quantize_fp8_per_tensor_fixed_scale(Tensor input, Tensor scale, Tensor? bs=None, bool stochatic_rounding=False) -> Tensor");
  m.impl(
      "quantize_fp8_per_tensor_fixed_scale",
      quantize_fp8_per_tensor_fixed_scale);
#endif
}

TORCH_LIBRARY_IMPL(fbgemm, CUDA, m) {
  m.impl("f8f8bf16_blockwise", f8f8bf16_blockwise);
  m.impl("f8f8bf16_tensorwise", f8f8bf16_tensorwise);
  m.impl("f8f8bf16_rowwise", f8f8bf16_rowwise);
#ifndef USE_ROCM
  m.impl("i8i8bf16", i8i8bf16);
  m.impl("quantize_fp8_per_tensor", quantize_fp8_per_tensor);
  m.impl("f8f8bf16", f8f8bf16);
  m.impl("f8f8bf16_cublas", f8f8bf16_cublas);
  m.impl("quantize_fp8_per_row", quantize_fp8_per_row);
#endif
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
    at::Tensor /* x_scale */,
    at::Tensor /* w_scale */,
    std::optional<at::Tensor> /* bias = c10::nullopt */,
    bool /* use_fast_accum = true */,
    std::optional<at::Tensor> /* output = c10::nullopt */) {
  int M = XQ.size(0);
  int N = WQ.size(0);
  auto Y = at::empty({M, N}, XQ.options().dtype(at::kBFloat16));
  return Y;
}

at::Tensor f8f8bf16_blockwise_meta(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    at::Tensor /* x_scale */,
    at::Tensor /* w_scale */,
    int64_t /* block_m = 256*/,
    int64_t /* block_n = 256*/,
    int64_t /* block_k = 256*/) {
  int M = XQ.size(0);
  int N = WQ.size(0);
  auto Y = at::empty({M, N}, XQ.options().dtype(at::kBFloat16));
  return Y;
}

std::vector<at::Tensor> quantize_fp8_per_tensor_meta(
    at::Tensor X,
    std::optional<at::Tensor> bs,
    std::optional<at::Tensor> /*scale_ub*/,
    const bool /*stochastic_rounding*/) {
  auto Y = at::empty_like(X, X.options().dtype(at::kFloat8_e4m3fn));
  auto scale = at::empty({}, X.options().dtype(at::kBFloat16));
  return {Y, scale};
}

at::Tensor f8f8bf16_cublas_meta(
    at::Tensor X,
    at::Tensor W,
    std::optional<at::Tensor> /* x_scale = c10::nullopt */,
    std::optional<at::Tensor> /* w_scale = c10::nullopt */,
    bool /* use_fast_accum = true */,
    std::optional<at::Tensor> /* output = c10::nullopt */) {
  const at::SymInt M = X.sym_size(0);
  const at::SymInt N = W.sym_size(0);
  auto Y = at::empty_symint({M, N}, X.options().dtype(at::kBFloat16));
  return Y;
}

at::Tensor f8f8bf16_meta(
    at::Tensor X,
    at::Tensor W,
    at::Tensor scale,
    bool use_fast_accum = true) {
  const at::SymInt M = X.sym_size(0);
  const at::SymInt N = W.sym_size(0);
  auto Y = at::empty_symint({M, N}, X.options().dtype(at::kBFloat16));
  return Y;
}

at::Tensor f8f8bf16_tensorwise_meta(
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
    at::Tensor WQ, // INT4
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor w_zp) {
  int M = XQ.size(0);
  int N = WQ.size(0);
  auto Y = at::empty({M, N}, XQ.options().dtype(at::kBFloat16));
  return Y;
}

at::Tensor bf16i4bf16_rowwise_meta(
    at::Tensor X, // BF16
    at::Tensor WQ, // INT4
    at::Tensor w_scale,
    at::Tensor w_zp) {
  int M = X.size(0);
  int N = WQ.size(0);
  auto Y = at::empty({M, N}, X.options().dtype(at::kBFloat16));
  return Y;
}

std::vector<at::Tensor> quantize_fp8_per_row_meta(
    at::Tensor input,
    std::optional<at::Tensor> bs,
    std::optional<at::Tensor> scale_ub,
    std::optional<c10::ScalarType> /* output_dtype */,
    bool /* stochastic_rounding */) {
  const at::SymInt M = input.sym_size(0);
  auto Y = at::empty_like(input, input.options().dtype(at::kFloat8_e4m3fn));
  auto scale = at::empty_symint({M}, input.options().dtype(at::kFloat));
  return {Y, scale};
}

TORCH_LIBRARY_IMPL(fbgemm, Meta, m) {
  m.impl("f8f8bf16_blockwise", f8f8bf16_blockwise_meta);
  m.impl("f8f8bf16_tensorwise", f8f8bf16_tensorwise_meta);
  m.impl("f8f8bf16_rowwise", f8f8bf16_rowwise_meta);
#ifndef USE_ROCM
  m.impl("i8i8bf16", i8i8bf16_meta);
  m.impl("quantize_fp8_per_tensor", quantize_fp8_per_tensor_meta);
  m.impl("f8f8bf16", f8f8bf16_meta);
  m.impl("f8f8bf16_cublas", f8f8bf16_cublas_meta);
  m.impl("f8i4bf16_rowwise", f8i4bf16_rowwise_meta);
  m.impl("quantize_fp8_per_row", quantize_fp8_per_row_meta);
  m.impl("bf16i4bf16_rowwise", bf16i4bf16_rowwise_meta);
#endif
}

} // namespace fbgemm_gpu
