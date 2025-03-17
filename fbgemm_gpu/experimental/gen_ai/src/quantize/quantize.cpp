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
#include <string>
#include <vector>
#include "c10/util/Exception.h"

#if (defined(USE_ROCM) && ROCM_VERSION >= 60200)
#define torch_fp8_e4m3 at::kFloat8_e4m3fnuz
#else
#define torch_fp8_e4m3 at::kFloat8_e4m3fn
#endif

namespace fbgemm_gpu {

#ifdef USE_ROCM
// flush icache
void flush_icache_ck();
#endif

// SmoothQuant kernels
at::Tensor
i8i8bf16(at::Tensor XQ, at::Tensor WQ, double scale, int64_t split_k);
at::Tensor i8i8bf16_dynamic(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor scale,
    int64_t split_k = 1);

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
at::Tensor f8f8bf16_lite(at::Tensor XQ, at::Tensor WQ, at::Tensor scale);
std::vector<at::Tensor> bf16bf16bf16_grouped(
    at::TensorList X,
    at::TensorList W,
    std::optional<std::vector<at::Tensor>> output = std::nullopt);
at::Tensor bf16bf16bf16_grouped_dynamic(
    at::TensorList X,
    at::TensorList W,
    std::optional<at::Tensor> zero_start_index_M = std::nullopt);
at::Tensor f8f8bf16_rowwise(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias = std::nullopt,
    bool use_fast_accum = true);
void f8f8bf16_rowwise_out(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    std::optional<at::Tensor> bias = std::nullopt,
    bool use_fast_accum = true);
at::Tensor f8f8bf16_rowwise_batched(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias = std::nullopt,
    bool use_fast_accum = true,
    std::optional<at::Tensor> output = std::nullopt);
std::vector<at::Tensor> f8f8bf16_rowwise_grouped(
    at::TensorList XQ,
    at::TensorList WQ,
    at::TensorList x_scale,
    at::TensorList w_scale,
    std::optional<std::vector<at::Tensor>> output = std::nullopt);
at::Tensor f8f8bf16_rowwise_grouped_stacked(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor M_sizes,
    std::optional<at::Tensor> output = std::nullopt);
at::Tensor f8f8bf16_rowwise_grouped_dynamic(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor zero_start_index_M,
    bool zeroing_output_tensor = true);
at::Tensor f8f8bf16_blockwise(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    int64_t block_m = 128,
    int64_t block_n = 128,
    int64_t block_k = 128);
at::Tensor f8f8bf16_cublas(
    at::Tensor A,
    at::Tensor B,
    std::optional<at::Tensor> Ainvs = std::nullopt,
    std::optional<at::Tensor> Binvs = std::nullopt,
    bool use_fast_accum = true,
    std::optional<at::Tensor> output = std::nullopt);
at::Tensor bf16_fast_gemv(at::Tensor X, at::Tensor W);
at::Tensor
bf16fp8bf16_fast_gemv(at::Tensor X, at::Tensor W, at::Tensor w_scale);
at::Tensor fp8fp8bf16_fast_gemv(at::Tensor X, at::Tensor W, at::Tensor scale);

at::Tensor f8i4bf16_rowwise(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor w_zp);
at::Tensor f8i4bf16_shuffled(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor w_scale_group);
std::tuple<at::Tensor, at::Tensor> preshuffle_i4(
    at::Tensor WQ,
    at::Tensor w_scale);
at::Tensor bf16i4bf16_rowwise(
    at::Tensor X,
    at::Tensor WQ,
    at::Tensor w_scale,
    at::Tensor w_zp);
at::Tensor bf16i4bf16_rowwise_batched(
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

std::vector<at::Tensor> quantize_fp8_per_col(
    at::Tensor input,
    std::optional<at::Tensor> bs, // batch size
    std::optional<at::Tensor> scale_ub); // scale upperbound

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
  m.set_python_module("fbgemm_gpu.experimental.gen_ai.quantize_ops");

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
  m.def(
      "f8i4bf16_shuffled(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor w_scale_group) -> Tensor");
  m.def("preshuffle_i4(Tensor WQ, Tensor w_scale) -> (Tensor, Tensor)");
  m.def("bf16_fast_gemv(Tensor X, Tensor W) -> Tensor");
  m.def("bf16fp8bf16_fast_gemv(Tensor X, Tensor W, Tensor w_scale) -> Tensor");
  m.def("fp8fp8bf16_fast_gemv(Tensor X, Tensor W, Tensor scale) -> Tensor");
  m.def("f8f8bf16_lite(Tensor XQ, Tensor WQ, Tensor scale) -> Tensor");
  m.def(
      "bf16i4bf16_rowwise(Tensor X, Tensor WQ, Tensor w_scale, Tensor w_zp) -> Tensor");
  m.def(
      "bf16i4bf16_rowwise_batched(Tensor X, Tensor WQ, Tensor w_scale, Tensor w_zp) -> Tensor");
  m.def(
      "i8i8bf16_dynamic(Tensor XQ, Tensor WQ, Tensor scale, int split_k=1) -> Tensor");
  m.impl("i8i8bf16_dynamic", i8i8bf16_dynamic);
#endif
  m.def(
      "bf16bf16bf16_grouped(Tensor[] X, Tensor[] W, Tensor[](a!)? output=None) -> Tensor[]");
  m.def(
      "bf16bf16bf16_grouped_dynamic(Tensor[] X, Tensor[] W, Tensor? zero_start_index_M=None) -> Tensor");
  m.def(
      "f8f8bf16_blockwise(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, int block_m=128, int block_n=128, int block_k=128) -> Tensor");
  m.def(
      "f8f8bf16_rowwise(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias=None, bool use_fast_accum=True) -> Tensor");
  m.def(
      "f8f8bf16_rowwise_out(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor(a!) output, Tensor? bias=None, bool use_fast_accum=True) -> ()");
  m.def(
      "f8f8bf16_rowwise_batched(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias=None, bool use_fast_accum=True, Tensor(a!)? output=None) -> Tensor");
  m.def(
      "f8f8bf16_rowwise_grouped(Tensor[] XQ, Tensor[] WQ, Tensor[] x_scale, Tensor[] w_scale, Tensor[](a!)? output=None) -> Tensor[]");
  m.def(
      "f8f8bf16_rowwise_grouped_stacked(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor M_sizes, Tensor(a!)? output=None) -> Tensor");
  m.def(
      "f8f8bf16_rowwise_grouped_dynamic(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor zero_start_index_M, bool zeroing_output_tensor=True) -> Tensor");
  m.def(
      "f8f8bf16_tensorwise(Tensor XQ, Tensor WQ, float scale, bool use_fast_accum=True) -> Tensor");
  m.def("per_tensor_quantize_i8(Tensor X, float scale) -> Tensor");
  m.impl("per_tensor_quantize_i8", per_tensor_quantize_i8);
  m.def("per_tensor_dynamic_quantize_i8(Tensor X) -> (Tensor, Tensor)");
  m.impl("per_tensor_dynamic_quantize_i8", per_tensor_dynamic_quantize_i8);

  m.def("silu_mul_quantize_i8(Tensor X1, Tensor X2, float scale) -> Tensor");
  m.impl("silu_mul_quantize_i8", silu_mul_quantize_i8);

  m.def(
      "quantize_fp8_per_tensor(Tensor input, Tensor? bs=None, Tensor? scale_ub=None, bool stochastic_rounding=False) -> Tensor[]");
  m.def(
      "quantize_fp8_per_row(Tensor input, Tensor? bs=None, Tensor? scale_ub=None, ScalarType? output_dtype=None, bool stochastic_rounding = False) -> Tensor[] ");

  m.def(
      "quantize_fp8_per_col(Tensor input, Tensor? bs=None, Tensor? scale_ub=None) -> Tensor[]");

  m.def(
      "get_fp8_per_tensor_scale(Tensor input, Tensor? bs=None, Tensor? scale_ub=None) -> Tensor");
  m.impl("get_fp8_per_tensor_scale", get_fp8_per_tensor_scale);

  m.def(
      "quantize_fp8_per_tensor_fixed_scale(Tensor input, Tensor scale, Tensor? bs=None, bool stochatic_rounding=False) -> Tensor");
  m.impl(
      "quantize_fp8_per_tensor_fixed_scale",
      quantize_fp8_per_tensor_fixed_scale);

#ifdef USE_ROCM
  m.def("flush_icache_hip() -> ()");
  m.impl("flush_icache_hip", flush_icache_ck);
#endif
}

TORCH_LIBRARY_IMPL(fbgemm, CUDA, m) {
  m.impl("f8f8bf16_blockwise", f8f8bf16_blockwise);
  m.impl("f8f8bf16_tensorwise", f8f8bf16_tensorwise);
  m.impl("f8f8bf16_rowwise", f8f8bf16_rowwise);
  m.impl("f8f8bf16_rowwise_out", f8f8bf16_rowwise_out);
  m.impl("f8f8bf16_rowwise_batched", f8f8bf16_rowwise_batched);
  m.impl("f8f8bf16_rowwise_grouped", f8f8bf16_rowwise_grouped);
  m.impl("f8f8bf16_rowwise_grouped_stacked", f8f8bf16_rowwise_grouped_stacked);
  m.impl("f8f8bf16_rowwise_grouped_dynamic", f8f8bf16_rowwise_grouped_dynamic);
  m.impl("quantize_fp8_per_tensor", quantize_fp8_per_tensor);
  m.impl("quantize_fp8_per_row", quantize_fp8_per_row);
  m.impl("quantize_fp8_per_col", quantize_fp8_per_col);
  m.impl("bf16bf16bf16_grouped", bf16bf16bf16_grouped);
  m.impl("bf16bf16bf16_grouped_dynamic", bf16bf16bf16_grouped_dynamic);

#ifndef USE_ROCM
  m.impl("i8i8bf16", i8i8bf16);
  m.impl("f8f8bf16", f8f8bf16);
  m.impl("f8f8bf16_cublas", f8f8bf16_cublas);
  m.impl("bf16_fast_gemv", bf16_fast_gemv);
  m.impl("bf16fp8bf16_fast_gemv", bf16fp8bf16_fast_gemv);
  m.impl("fp8fp8bf16_fast_gemv", fp8fp8bf16_fast_gemv);
  m.impl("f8f8bf16_lite", f8f8bf16_lite);
  m.impl("f8i4bf16_rowwise", f8i4bf16_rowwise);
  m.impl("f8i4bf16_shuffled", f8i4bf16_shuffled);
  m.impl("preshuffle_i4", preshuffle_i4);
  m.impl("bf16i4bf16_rowwise_batched", bf16i4bf16_rowwise_batched);
  m.impl("bf16i4bf16_rowwise", bf16i4bf16_rowwise);
#endif
}

// Though it should never be used, it still seems helpful to define these
// functions for CPU to accommodate model creation.
TORCH_LIBRARY_IMPL(fbgemm, CPU, m) {
  m.impl("f8f8bf16_blockwise", f8f8bf16_blockwise);
  m.impl("f8f8bf16_tensorwise", f8f8bf16_tensorwise);
  m.impl("f8f8bf16_rowwise", f8f8bf16_rowwise);
  m.impl("f8f8bf16_rowwise_batched", f8f8bf16_rowwise_batched);
  m.impl("f8f8bf16_rowwise_grouped", f8f8bf16_rowwise_grouped);
  m.impl("f8f8bf16_rowwise_grouped_stacked", f8f8bf16_rowwise_grouped_stacked);
  m.impl("f8f8bf16_rowwise_grouped_dynamic", f8f8bf16_rowwise_grouped_dynamic);
  m.impl("quantize_fp8_per_tensor", quantize_fp8_per_tensor);
  m.impl("quantize_fp8_per_row", quantize_fp8_per_row);
  m.impl("quantize_fp8_per_col", quantize_fp8_per_col);
  m.impl("bf16bf16bf16_grouped", bf16bf16bf16_grouped);
  m.impl("bf16bf16bf16_grouped_dyanmic", bf16bf16bf16_grouped_dynamic);
#ifndef USE_ROCM
  m.impl("i8i8bf16", i8i8bf16);
  m.impl("f8f8bf16", f8f8bf16);
  m.impl("f8f8bf16_cublas", f8f8bf16_cublas);
  m.impl("bf16_fast_gemv", bf16_fast_gemv);
  m.impl("bf16fp8bf16_fast_gemv", bf16fp8bf16_fast_gemv);
  m.impl("fp8fp8bf16_fast_gemv", fp8fp8bf16_fast_gemv);
  m.impl("f8f8bf16_lite", f8f8bf16_lite);
  m.impl("f8i4bf16_rowwise", f8i4bf16_rowwise);
  m.impl("f8i4bf16_shuffled", f8i4bf16_shuffled);
  m.impl("preshuffle_i4", preshuffle_i4);
  m.impl("bf16i4bf16_rowwise_batched", bf16i4bf16_rowwise_batched);
  m.impl("bf16i4bf16_rowwise", bf16i4bf16_rowwise);
#endif
}

// Shape registration functions.
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
    std::optional<at::Tensor> /* bias = std::nullopt */,
    bool /* use_fast_accum = true */) {
  const at::SymInt M = XQ.sym_size(0);
  const at::SymInt N = WQ.sym_size(0);
  auto Y = at::empty_symint({M, N}, XQ.options().dtype(at::kBFloat16));
  return Y;
}

void f8f8bf16_rowwise_out_meta(
    at::Tensor /* XQ */,
    at::Tensor /* WQ */, // FP8
    at::Tensor /* x_scale */,
    at::Tensor /* w_scale */,
    at::Tensor /* output */,
    std::optional<at::Tensor> /* bias = std::nullopt */,
    bool /* use_fast_accum = true */) {
  return;
}

at::Tensor f8f8bf16_rowwise_batched_meta(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    at::Tensor /* x_scale */,
    at::Tensor /* w_scale */,
    std::optional<at::Tensor> /* bias = std::nullopt */,
    bool /* use_fast_accum = true */,
    std::optional<at::Tensor> /* output = std::nullopt */) {
  int B = XQ.size(0);
  int M = XQ.size(1);
  int N = WQ.size(1);
  auto Y = at::empty({B, M, N}, XQ.options().dtype(at::kBFloat16));
  return Y;
}

at::Tensor f8f8bf16_blockwise_meta(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    at::Tensor /* x_scale */,
    at::Tensor /* w_scale */,
    int64_t /* block_m = 128*/,
    int64_t /* block_n = 128*/,
    int64_t /* block_k = 128*/) {
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
  auto Y = at::empty_like(X, X.options().dtype(torch_fp8_e4m3));
  auto scale = at::empty({}, X.options().dtype(at::kBFloat16));
  return {Y, scale};
}

at::Tensor f8f8bf16_cublas_meta(
    at::Tensor X,
    at::Tensor W,
    std::optional<at::Tensor> /* x_scale = std::nullopt */,
    std::optional<at::Tensor> /* w_scale = std::nullopt */,
    bool /* use_fast_accum = true */,
    std::optional<at::Tensor> /* output = std::nullopt */) {
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

at::Tensor bf16_fast_gemv_meta(at::Tensor X, at::Tensor W) {
  const at::SymInt M = X.sym_size(0);
  const at::SymInt N = W.sym_size(0);
  auto Y = at::empty_symint({M, N}, X.options().dtype(at::kHalf));
  return Y;
}

at::Tensor bf16fp8bf16_fast_gemv_meta(
    at::Tensor X,
    at::Tensor W,
    at::Tensor /* w_scale */) {
  const at::SymInt M = X.sym_size(0);
  const at::SymInt N = W.sym_size(0);
  auto Y = at::empty_symint({M, N}, X.options().dtype(at::kBFloat16));
  return Y;
}

at::Tensor
fp8fp8bf16_fast_gemv_meta(at::Tensor X, at::Tensor W, at::Tensor /* scale */) {
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

at::Tensor f8f8bf16_lite_meta(at::Tensor X, at::Tensor W, at::Tensor scale) {
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
    at::Tensor /*  w_scale */,
    at::Tensor /* w_zp */
) {
  int M = X.size(0);
  int N = WQ.size(0);
  auto Y = at::empty({M, N}, X.options().dtype(at::kBFloat16));
  return Y;
}

at::Tensor bf16i4bf16_rowwise_batched_meta(
    at::Tensor X, // BF16
    at::Tensor WQ, // INT4
    at::Tensor /* w_scale */,
    at::Tensor /* w_zp */
) {
  int B = X.size(0);
  int M = X.size(1);
  int N = WQ.size(1);
  auto Y = at::empty({B, M, N}, X.options().dtype(at::kBFloat16));
  return Y;
}

std::vector<at::Tensor> quantize_fp8_per_row_meta(
    at::Tensor input,
    std::optional<at::Tensor> bs,
    std::optional<at::Tensor> scale_ub,
    std::optional<c10::ScalarType> /* output_dtype */,
    bool /* stochastic_rounding */) {
  const at::SymInt M = input.sym_size(0);
  auto Y = at::empty_like(input, input.options().dtype(torch_fp8_e4m3));
  auto scale = at::empty_symint({M}, input.options().dtype(at::kFloat));
  return {Y, scale};
}

std::vector<at::Tensor> quantize_fp8_per_col_meta(
    at::Tensor input,
    std::optional<at::Tensor> /* bs */,
    std::optional<at::Tensor> /* scale_ub */) {
  const at::SymInt M = input.sym_size(0);
  auto Y = at::empty_like(input, input.options().dtype(torch_fp8_e4m3));
  auto scale = at::empty_symint({M}, input.options().dtype(at::kFloat));
  return {Y, scale};
}

std::vector<at::Tensor> bf16bf16bf16_grouped_meta(
    at::TensorList X,
    at::TensorList W,
    std::optional<std::vector<at::Tensor>> /* output = std::nullopt */
) {
  std::vector<at::Tensor> Y;
  for (int i = 0; i < X.size(); i++) {
    const at::SymInt M = X[i].sym_size(0);
    const at::SymInt N = W[i].sym_size(0);
    Y.push_back(at::empty_symint({M, N}, X[i].options().dtype(at::kBFloat16)));
  }
  return Y;
}

at::Tensor bf16bf16bf16_grouped_dynamic_meta(
    at::TensorList X,
    at::TensorList W,
    std::optional<at::Tensor> /* zero_start_index_M = std::nullopt */) {
  int G = X.size();
  int M = X[0].size(0);
  int N = W[0].size(0);
  at::Tensor Y = at::empty({G, M, N}, X[0].options().dtype(at::kBFloat16));
  return Y;
}

TORCH_LIBRARY_IMPL(fbgemm, Meta, m) {
  m.impl("f8f8bf16_blockwise", f8f8bf16_blockwise_meta);
  m.impl("f8f8bf16_tensorwise", f8f8bf16_tensorwise_meta);
  m.impl("f8f8bf16_rowwise", f8f8bf16_rowwise_meta);
  m.impl("f8f8bf16_rowwise_out", f8f8bf16_rowwise_out_meta);
  m.impl("quantize_fp8_per_tensor", quantize_fp8_per_tensor_meta);
  m.impl("quantize_fp8_per_row", quantize_fp8_per_row_meta);
  m.impl("quantize_fp8_per_col", quantize_fp8_per_col_meta);
  m.impl("bf16bf16bf16_grouped", bf16bf16bf16_grouped_meta);
  m.impl("bf16bf16bf16_grouped_dynamic", bf16bf16bf16_grouped_dynamic_meta);
#ifndef USE_ROCM
  m.impl("i8i8bf16", i8i8bf16_meta);
  m.impl("f8f8bf16", f8f8bf16_meta);
  m.impl("f8f8bf16_cublas", f8f8bf16_cublas_meta);
  m.impl("bf16_fast_gemv", bf16_fast_gemv_meta);
  m.impl("bf16fp8bf16_fast_gemv", bf16fp8bf16_fast_gemv_meta);
  m.impl("fp8fp8bf16_fast_gemv", fp8fp8bf16_fast_gemv_meta);
  m.impl("f8f8bf16_rowwise_batched", f8f8bf16_rowwise_batched_meta);
  m.impl("f8i4bf16_rowwise", f8i4bf16_rowwise_meta);
  m.impl("bf16i4bf16_rowwise", bf16i4bf16_rowwise_meta);
  m.impl("bf16i4bf16_rowwise_batched", bf16i4bf16_rowwise_batched_meta);
  m.impl("f8f8bf16_lite", f8f8bf16_lite_meta);
#endif
}

} // namespace fbgemm_gpu
