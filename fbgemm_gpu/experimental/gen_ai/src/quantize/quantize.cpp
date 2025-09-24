/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <vector>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAEvent.h>
#include <fbgemm_gpu/torch_ops.h>
#include <torch/library.h>
#include "c10/core/ScalarType.h"
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
at::Tensor f4f4bf16(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> global_scale = std::nullopt,
    bool use_mx = true);
at::Tensor f4f4bf16_grouped_stacked(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor M_sizes,
    std::optional<at::Tensor> global_scale = std::nullopt,
    std::optional<at::Tensor> starting_row_after_padding = std::nullopt,
    bool use_mx = true);
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
    at::TensorList W);
at::Tensor bf16bf16bf16_grouped_cat(at::TensorList X, at::TensorList W);
at::Tensor bf16bf16bf16_grouped_dynamic(
    at::Tensor X,
    at::Tensor W,
    at::Tensor zero_start_index_M);
at::Tensor bf16bf16bf16_grouped_stacked(
    at::Tensor X,
    at::Tensor W,
    at::Tensor M_sizes,
    std::optional<at::Tensor> out = std::nullopt);
at::Tensor f8f8bf16_rowwise(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias = std::nullopt,
    bool use_fast_accum = true);
at::Tensor f8f8f16_rowwise(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias = std::nullopt,
    bool use_fast_accum = true);
at::Tensor f8f8bf16_groupwise(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale);
at::Tensor f8f8bf16_rowwise_preshuffle(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias = std::nullopt,
    bool use_fast_accum = true);
at::Tensor f8f8f16_rowwise_preshuffle(
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
    at::TensorList w_scale);
at::Tensor f8f8bf16_rowwise_grouped_cat(
    at::TensorList XQ,
    at::TensorList WQ,
    at::TensorList x_scale,
    at::TensorList w_scale);
at::Tensor f8f8bf16_rowwise_grouped_stacked(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor M_sizes);
at::Tensor f8f8bf16_rowwise_grouped_dynamic(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor zero_start_index_M,
    bool zeroing_output_tensor = true);
at::Tensor f8f8bf16_groupwise_grouped(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor M_sizes);
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
at::Tensor fp8fp8bf16_fast_gemv(
    at::Tensor X,
    at::Tensor W,
    at::Tensor x_scale,
    at::Tensor w_scale,
    bool is_batched = false);

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
at::Tensor bf16i4bf16_shuffled(
    at::Tensor X,
    at::Tensor W,
    at::Tensor w_scale_group,
    at::Tensor w_zero_group);
at::Tensor f8i4bf16_shuffled_grouped(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor w_scale_group,
    at::Tensor M_sizes);
at::Tensor bf16i4bf16_shuffled_grouped(
    at::Tensor X,
    at::Tensor WQ,
    at::Tensor w_scale_group,
    at::Tensor w_zero_group,
    at::Tensor M_sizes);
std::tuple<at::Tensor, at::Tensor> preshuffle_i4(
    at::Tensor WQ,
    at::Tensor w_scale);
at::Tensor bf16i4bf16_rowwise(
    at::Tensor X,
    at::Tensor W,
    at::Tensor w_scale_group,
    at::Tensor w_zero_group);
at::Tensor bf16i4bf16_shuffled_batched(
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

void scaled_fp4_quant(
    at::Tensor const& output,
    at::Tensor const& input,
    at::Tensor const& output_sf,
    at::Tensor const& input_sf);

std::vector<at::Tensor> fake_quantize_nvfp4_per_tensor(
    at::Tensor input,
    std::optional<at::Tensor> static_scales,
    std::optional<at::Tensor> bs, // batch size
    std::optional<at::Tensor> scale_ub); // scale upperbound

TORCH_LIBRARY_IMPL(fbgemm, CUDA, m) {
  m.impl("f8f8bf16_blockwise", f8f8bf16_blockwise);
  m.impl("f8f8bf16_tensorwise", f8f8bf16_tensorwise);
  m.impl("f8f8bf16_rowwise", f8f8bf16_rowwise);
  m.impl("f8f8bf16_rowwise_out", f8f8bf16_rowwise_out);
  m.impl("f8f8bf16_rowwise_batched", f8f8bf16_rowwise_batched);
  m.impl("f8f8bf16_rowwise_grouped", f8f8bf16_rowwise_grouped);
  m.impl("f8f8bf16_rowwise_grouped_cat", f8f8bf16_rowwise_grouped_cat);
  m.impl("f8f8bf16_rowwise_grouped_stacked", f8f8bf16_rowwise_grouped_stacked);
  m.impl("f8f8bf16_rowwise_grouped_dynamic", f8f8bf16_rowwise_grouped_dynamic);
  m.impl("quantize_fp8_per_tensor", quantize_fp8_per_tensor);
  m.impl("quantize_fp8_per_row", quantize_fp8_per_row);
  m.impl("quantize_fp8_per_col", quantize_fp8_per_col);
  m.impl("bf16bf16bf16_grouped", bf16bf16bf16_grouped);
  m.impl("bf16bf16bf16_grouped_cat", bf16bf16bf16_grouped_cat);
  m.impl("bf16bf16bf16_grouped_dynamic", bf16bf16bf16_grouped_dynamic);
  m.impl("bf16bf16bf16_grouped_stacked", bf16bf16bf16_grouped_stacked);
  m.impl("per_tensor_quantize_i8", per_tensor_quantize_i8);
  m.impl("per_tensor_dynamic_quantize_i8", per_tensor_dynamic_quantize_i8);
  m.impl("silu_mul_quantize_i8", silu_mul_quantize_i8);
  m.impl("get_fp8_per_tensor_scale", get_fp8_per_tensor_scale);
  m.impl(
      "quantize_fp8_per_tensor_fixed_scale",
      quantize_fp8_per_tensor_fixed_scale);

#ifndef USE_ROCM
  m.impl("f8f8bf16_groupwise", f8f8bf16_groupwise);
  m.impl("f8f8bf16_groupwise_grouped", f8f8bf16_groupwise_grouped);
  m.impl("i8i8bf16", i8i8bf16);
  m.impl("f4f4bf16", f4f4bf16);
  m.impl("f4f4bf16_grouped_stacked", f4f4bf16_grouped_stacked);
  m.impl("mx8mx8bf16_grouped_mm", mx8mx8bf16_grouped_mm);
  m.impl("f8f8bf16", f8f8bf16);
  m.impl("f8f8bf16_cublas", f8f8bf16_cublas);
  m.impl("bf16_fast_gemv", bf16_fast_gemv);
  m.impl("bf16fp8bf16_fast_gemv", bf16fp8bf16_fast_gemv);
  m.impl("fp8fp8bf16_fast_gemv", fp8fp8bf16_fast_gemv);
  m.impl("f8f8bf16_lite", f8f8bf16_lite);
  m.impl("f8i4bf16_rowwise", f8i4bf16_rowwise);
  m.impl("f8i4bf16_shuffled", f8i4bf16_shuffled);
  m.impl("bf16i4bf16_shuffled", bf16i4bf16_shuffled);
  m.impl("f8i4bf16_shuffled_grouped", f8i4bf16_shuffled_grouped);
  m.impl("bf16i4bf16_shuffled_grouped", bf16i4bf16_shuffled_grouped);
  m.impl("preshuffle_i4", preshuffle_i4);
  m.impl("bf16i4bf16_shuffled_batched", bf16i4bf16_shuffled_batched);
  m.impl("bf16i4bf16_rowwise_batched", bf16i4bf16_rowwise_batched);
  m.impl("bf16i4bf16_rowwise", bf16i4bf16_rowwise);
  m.impl("scaled_fp4_quant", scaled_fp4_quant);
  m.impl("i8i8bf16_dynamic", i8i8bf16_dynamic);
  m.impl("fake_quantize_nvfp4_per_tensor", fake_quantize_nvfp4_per_tensor);
#endif

#ifdef USE_ROCM
  m.impl("flush_icache_hip", flush_icache_ck);
  m.impl("f8f8bf16_rowwise_grouped_mm", f8f8bf16_rowwise_grouped_mm);
#endif
#ifdef USE_ROCM
  m.impl("f8f8f16_rowwise", f8f8f16_rowwise);
  m.impl("f8f8bf16_rowwise_preshuffle", f8f8bf16_rowwise_preshuffle);
  m.impl("f8f8f16_rowwise_preshuffle", f8f8bf16_rowwise_preshuffle);
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
  m.impl("bf16bf16bf16_grouped_cat", bf16bf16bf16_grouped_cat);
  m.impl("bf16bf16bf16_grouped_dyanmic", bf16bf16bf16_grouped_dynamic);
  m.impl("bf16bf16bf16_grouped_stacked", bf16bf16bf16_grouped_stacked);
#ifndef USE_ROCM
  m.impl("f8f8bf16_groupwise", f8f8bf16_groupwise);
  m.impl("f8f8bf16_groupwise_grouped", f8f8bf16_groupwise_grouped);
  m.impl("i8i8bf16", i8i8bf16);
  m.impl("f4f4bf16", f4f4bf16);
  m.impl("f4f4bf16_grouped_stacked", f4f4bf16_grouped_stacked);
  m.impl("mx8mx8bf16_grouped_mm", mx8mx8bf16_grouped_mm);
  m.impl("f8f8bf16", f8f8bf16);
  m.impl("f8f8bf16_cublas", f8f8bf16_cublas);
  m.impl("bf16_fast_gemv", bf16_fast_gemv);
  m.impl("bf16fp8bf16_fast_gemv", bf16fp8bf16_fast_gemv);
  m.impl("fp8fp8bf16_fast_gemv", fp8fp8bf16_fast_gemv);
  m.impl("f8f8bf16_lite", f8f8bf16_lite);
  m.impl("f8i4bf16_rowwise", f8i4bf16_rowwise);
  m.impl("f8i4bf16_shuffled", f8i4bf16_shuffled);
  m.impl("bf16i4bf16_shuffled", bf16i4bf16_shuffled);
  m.impl("f8i4bf16_shuffled_grouped", f8i4bf16_shuffled_grouped);
  m.impl("bf16i4bf16_shuffled_grouped", bf16i4bf16_shuffled_grouped);
  m.impl("preshuffle_i4", preshuffle_i4);
  m.impl("bf16i4bf16_shuffled_batched", bf16i4bf16_shuffled_batched);
  m.impl("bf16i4bf16_rowwise_batched", bf16i4bf16_rowwise_batched);
  m.impl("bf16i4bf16_rowwise", bf16i4bf16_rowwise);
  m.impl("scaled_fp4_quant", scaled_fp4_quant);
  m.impl("fake_quantize_nvfp4_per_tensor", fake_quantize_nvfp4_per_tensor);
#endif
}

// Shape registration functions.
at::Tensor i8i8bf16_meta(
    at::Tensor XQ, // INT8
    at::Tensor WQ, // INT8
    double scale,
    int64_t split_k) {
  const at::SymInt M = XQ.sym_size(0);
  const at::SymInt N = WQ.sym_size(0);
  auto Y = at::empty_symint({M, N}, XQ.options().dtype(at::kBFloat16));
  return Y;
}

at::Tensor f4f4bf16_meta(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor /* x_scale */,
    at::Tensor /* w_scale */,
    std::optional<at::Tensor> /* global_scale = std::nullopt */,
    bool /* use_mx */) {
  const at::SymInt M = XQ.sym_size(0);
  const at::SymInt N = WQ.sym_size(0);
  auto Y = at::empty_symint({M, N}, XQ.options().dtype(at::kBFloat16));
  return Y;
}

at::Tensor f8f8bf16_rowwise_meta(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    at::Tensor /* x_scale */,
    at::Tensor /* w_scale */,
    std::optional<at::Tensor> /* bias = std::nullopt */,
    bool /* use_fast_accum = true */) {
  int64_t x_dims = XQ.dim();
  int64_t w_dims = WQ.dim();
  TORCH_CHECK(
      (x_dims == 2 || x_dims == 3) && (w_dims == 2),
      "The dim of XQ must be 2 or 3, and dim of WQ must be 2");
  at::Tensor Y;
  if (x_dims == 2) {
    const at::SymInt M = XQ.sym_size(0);
    const at::SymInt N = WQ.sym_size(0);
    Y = at::empty_symint({M, N}, XQ.options().dtype(at::kBFloat16));
  } else {
    const at::SymInt B = XQ.sym_size(0);
    const at::SymInt M = XQ.sym_size(1);
    const at::SymInt N = WQ.sym_size(0);
    Y = at::empty_symint({B, M, N}, XQ.options().dtype(at::kBFloat16));
  }
  return Y;
}

at::Tensor f8f8f16_rowwise_meta(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    at::Tensor /* x_scale */,
    at::Tensor /* w_scale */,
    std::optional<at::Tensor> /* bias = std::nullopt */,
    bool /* use_fast_accum = true */) {
  const at::SymInt M = XQ.sym_size(0);
  const at::SymInt N = WQ.sym_size(0);
  auto Y = at::empty_symint({M, N}, XQ.options().dtype(at::kHalf));
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
  const at::SymInt B = XQ.sym_size(0);
  const at::SymInt M = XQ.sym_size(1);
  const at::SymInt N = WQ.sym_size(1);
  auto Y = at::empty_symint({B, M, N}, XQ.options().dtype(at::kBFloat16));
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
  int64_t x_dims = XQ.dim();
  int64_t w_dims = WQ.dim();
  TORCH_CHECK(
      (x_dims == 2 || x_dims == 3) && (w_dims == 2),
      "The dim of XQ must be 2 or 3, and dim of WQ must be 2");
  at::Tensor Y;
  if (x_dims == 2) {
    const at::SymInt M = XQ.sym_size(0);
    const at::SymInt N = WQ.sym_size(0);
    Y = at::empty_symint({M, N}, XQ.options().dtype(at::kBFloat16));
  } else {
    const at::SymInt B = XQ.sym_size(0);
    const at::SymInt M = XQ.sym_size(1);
    const at::SymInt N = WQ.sym_size(0);
    Y = at::empty_symint({B, M, N}, XQ.options().dtype(at::kBFloat16));
  }
  return Y;
}

std::vector<at::Tensor> quantize_fp8_per_tensor_meta(
    at::Tensor input,
    std::optional<at::Tensor> /* bs */,
    std::optional<at::Tensor> /*scale_ub*/,
    const bool /*stochastic_rounding*/) {
  int dims = input.dim();
  TORCH_CHECK(dims == 2 || dims == 3, "The dim of input should be 2 or 3");
  at::Tensor Y = at::empty_like(input, input.options().dtype(torch_fp8_e4m3));
  at::Tensor scale;
  if (dims <= 2) {
    scale = at::empty_symint({}, input.options().dtype(at::kFloat));
  } else {
    const at::SymInt B = input.sym_size(0);
    scale = at::empty_symint({B}, input.options().dtype(at::kFloat));
  }
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

at::Tensor fp8fp8bf16_fast_gemv_meta(
    at::Tensor X,
    at::Tensor W,
    at::Tensor /* x_scale */,
    at::Tensor /* w_scale */,
    bool is_batched) {
  at::Tensor Y;
  if (is_batched) {
    const at::SymInt B = X.sym_size(0);
    const at::SymInt M = X.sym_size(1);
    const at::SymInt N = W.sym_size(1);
    Y = at::empty_symint({B, M, N}, X.options().dtype(at::kBFloat16));
  } else {
    const at::SymInt M = X.sym_size(0);
    const at::SymInt N = W.sym_size(0);
    auto Y = at::empty_symint({M, N}, X.options().dtype(at::kBFloat16));
  }
  return Y;
}

at::Tensor f8f8bf16_tensorwise_meta(
    at::Tensor XQ,
    at::Tensor WQ,
    double /* scale */,
    bool /* use_fast_accum = true */) {
  int64_t x_dims = XQ.dim();
  int64_t w_dims = WQ.dim();
  TORCH_CHECK(
      (x_dims == 2 || x_dims == 3) && (w_dims == 2),
      "The dim of XQ must be 2 or 3, and dim of WQ must be 2");
  at::Tensor Y;
  if (x_dims == 2) {
    const at::SymInt M = XQ.sym_size(0);
    const at::SymInt N = WQ.sym_size(0);
    Y = at::empty_symint({M, N}, XQ.options().dtype(at::kBFloat16));
  } else {
    const at::SymInt B = XQ.sym_size(0);
    const at::SymInt M = XQ.sym_size(1);
    const at::SymInt N = WQ.sym_size(0);
    Y = at::empty_symint({B, M, N}, XQ.options().dtype(at::kBFloat16));
  }
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
    at::Tensor /* x_scale */,
    at::Tensor /* w_scale */,
    at::Tensor /* w_zp */) {
  int64_t x_dims = XQ.dim();
  int64_t w_dims = WQ.dim();
  TORCH_CHECK(
      (x_dims == 2 || x_dims == 3) && (w_dims == 2),
      "The dim of X must be 2 or 3, and dim of W must be 2");
  at::Tensor Y;
  if (x_dims == 2) {
    const at::SymInt M = XQ.sym_size(0);
    const at::SymInt N = WQ.sym_size(0);
    Y = at::empty_symint({M, N}, XQ.options().dtype(at::kBFloat16));
  } else {
    const at::SymInt B = XQ.sym_size(0);
    const at::SymInt M = XQ.sym_size(1);
    const at::SymInt N = WQ.sym_size(0);
    Y = at::empty_symint({B, M, N}, XQ.options().dtype(at::kBFloat16));
  }
  return Y;
}

std::tuple<at::Tensor, at::Tensor> preshuffle_i4_meta(
    at::Tensor WQ,
    at::Tensor w_scale) {
  auto WS = at::empty_like(w_scale);
  if (w_scale.dtype() != at::kBFloat16) {
    WS = at::empty({w_scale.size(0), 8, w_scale.size(1)}, w_scale.options());
  }
  return {at::empty_like(WQ), WS};
}

at::Tensor f8i4bf16_shuffled_meta(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // INT4
    at::Tensor /* x_scale */,
    at::Tensor /* w_scale */,
    at::Tensor /* w_scale_group */) {
  const at::SymInt M = XQ.sym_size(0);
  const at::SymInt N = WQ.sym_size(0);
  auto Y = at::empty_symint({M, N}, XQ.options().dtype(at::kBFloat16));
  return Y;
}

at::Tensor bf16i4bf16_rowwise_meta(
    at::Tensor X, // BF16
    at::Tensor W, // INT4
    at::Tensor /*  w_scale_group */,
    at::Tensor /* w_zero_group */
) {
  int64_t x_dims = X.dim();
  int64_t w_dims = W.dim();
  TORCH_CHECK(
      (x_dims == 2 || x_dims == 3) && (w_dims == 2),
      "The dim of XQ must be 2 or 3, and dim of WQ must be 2");
  at::Tensor Y;
  if (x_dims == 2) {
    const at::SymInt M = X.sym_size(0);
    const at::SymInt N = W.sym_size(0);
    Y = at::empty_symint({M, N}, X.options().dtype(at::kBFloat16));
  } else {
    const at::SymInt B = X.sym_size(0);
    const at::SymInt M = X.sym_size(1);
    const at::SymInt N = W.sym_size(0);
    Y = at::empty_symint({B, M, N}, X.options().dtype(at::kBFloat16));
  }
  return Y;
}

at::Tensor bf16i4bf16_shuffled_batched_meta(
    at::Tensor X, // BF16
    at::Tensor W, // INT4
    at::Tensor /* w_scale_group */,
    at::Tensor /* w_zero_group */
) {
  const at::SymInt B = X.sym_size(0);
  const at::SymInt M = X.sym_size(1);
  const at::SymInt N = W.sym_size(1);
  auto Y = at::empty_symint({B, M, N}, X.options().dtype(at::kBFloat16));
  return Y;
}

at::Tensor bf16i4bf16_rowwise_batched_meta(
    at::Tensor X, // BF16
    at::Tensor W, // INT4
    at::Tensor /* w_scale_group */,
    at::Tensor /* w_zero_group */
) {
  const at::SymInt B = X.sym_size(0);
  const at::SymInt M = X.sym_size(1);
  const at::SymInt N = W.sym_size(1);
  auto Y = at::empty_symint({B, M, N}, X.options().dtype(at::kBFloat16));
  return Y;
}

std::vector<at::Tensor> quantize_fp8_per_row_meta(
    at::Tensor input,
    std::optional<at::Tensor> /* bs */,
    std::optional<at::Tensor> /* scale_ub */,
    std::optional<c10::ScalarType> /* output_dtype */,
    bool /* stochastic_rounding */) {
  int dims = input.dim();
  TORCH_CHECK(dims == 2 || dims == 3, "The dim of input should be 2 or 3");
  at::Tensor Y = at::empty_like(input, input.options().dtype(torch_fp8_e4m3));
  at::Tensor scale;
  if (dims == 2) {
    const at::SymInt M = input.sym_size(0);
    scale = at::empty_symint({M}, input.options().dtype(at::kFloat));
  } else {
    const at::SymInt B = input.sym_size(0);
    const at::SymInt M = input.sym_size(1);
    scale = at::empty_symint({B, M}, input.options().dtype(at::kFloat));
  }
  return {Y, scale};
}

void scaled_fp4_quant_meta(
    at::Tensor const& output,
    at::Tensor const& input,
    at::Tensor const& output_sf,
    at::Tensor const& input_sf) {
  return;
}

std::vector<at::Tensor> quantize_fp8_per_col_meta(
    at::Tensor input,
    std::optional<at::Tensor> /* bs */,
    std::optional<at::Tensor> /* scale_ub */) {
  int dims = input.dim();
  TORCH_CHECK(dims == 2 || dims == 3, "The dim of input should be 2 or 3");
  at::Tensor Y = at::empty_like(input, input.options().dtype(torch_fp8_e4m3));
  at::Tensor scale;
  if (dims == 2) {
    const at::SymInt M = input.sym_size(0);
    scale = at::empty_symint({M}, input.options().dtype(at::kFloat));
  } else {
    const at::SymInt B = input.sym_size(0);
    const at::SymInt M = input.sym_size(1);
    scale = at::empty_symint({B, M}, input.options().dtype(at::kFloat));
  }
  return {Y, scale};
}

std::vector<at::Tensor> bf16bf16bf16_grouped_meta(
    at::TensorList X,
    at::TensorList W) {
  std::vector<at::Tensor> Y;
  for (int i = 0; i < X.size(); i++) {
    const at::SymInt M = X[i].sym_size(0);
    const at::SymInt N = W[i].sym_size(0);
    Y.push_back(at::empty_symint({M, N}, X[i].options().dtype(at::kBFloat16)));
  }
  return Y;
}

at::Tensor bf16bf16bf16_grouped_dynamic_meta(
    at::Tensor X,
    at::Tensor W,
    at::Tensor /* zero_start_index_M */) {
  const at::SymInt G = X.sym_size(0);
  const at::SymInt M = X.sym_size(1);
  const at::SymInt N = W.sym_size(1);
  at::Tensor Y =
      at::empty_symint({G, M, N}, X[0].options().dtype(at::kBFloat16));
  return Y;
}

at::Tensor bf16bf16bf16_grouped_stacked_meta(
    at::Tensor X,
    at::Tensor W,
    at::Tensor /* M_sizes */,
    std::optional<at::Tensor> out) {
  const at::SymInt total_M = X.sym_size(0);
  const at::SymInt N = W.sym_size(1);

  if (out.has_value()) {
    return out.value();
  } else {
    at::Tensor output =
        at::empty_symint({total_M, N}, X.options().dtype(at::kBFloat16));
    return output;
  }
}

at::Tensor f8f8bf16_rowwise_grouped_stacked_meta(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor /* x_scale */,
    at::Tensor /* w_scale */,
    at::Tensor /* M_sizes */) {
  const at::SymInt total_M = XQ.sym_size(0);
  const at::SymInt N = WQ.sym_size(1);
  at::Tensor Y =
      at::empty_symint({total_M, N}, XQ.options().dtype(at::kBFloat16));
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
  m.impl("bf16bf16bf16_grouped_stacked", bf16bf16bf16_grouped_stacked_meta);
  m.impl(
      "f8f8bf16_rowwise_grouped_stacked",
      f8f8bf16_rowwise_grouped_stacked_meta);
#ifndef USE_ROCM
  m.impl("i8i8bf16", i8i8bf16_meta);
  m.impl("f4f4bf16", f4f4bf16_meta);
  m.impl("f8f8bf16", f8f8bf16_meta);
  m.impl("f8f8bf16_cublas", f8f8bf16_cublas_meta);
  m.impl("bf16_fast_gemv", bf16_fast_gemv_meta);
  m.impl("bf16fp8bf16_fast_gemv", bf16fp8bf16_fast_gemv_meta);
  m.impl("fp8fp8bf16_fast_gemv", fp8fp8bf16_fast_gemv_meta);
  m.impl("f8f8bf16_rowwise_batched", f8f8bf16_rowwise_batched_meta);
  m.impl("f8i4bf16_rowwise", f8i4bf16_rowwise_meta);
  m.impl("bf16i4bf16_rowwise", bf16i4bf16_rowwise_meta);
  m.impl("bf16i4bf16_shuffled_batched", bf16i4bf16_shuffled_batched_meta);
  m.impl("bf16i4bf16_rowwise_batched", bf16i4bf16_rowwise_batched_meta);
  m.impl("f8f8bf16_lite", f8f8bf16_lite_meta);
  m.impl("scaled_fp4_quant", scaled_fp4_quant_meta);
  m.impl("preshuffle_i4", preshuffle_i4_meta);
  m.impl("f8i4bf16_shuffled", f8i4bf16_shuffled_meta);
#endif
#ifdef USE_ROCM
  m.impl("f8f8f16_rowwise", f8f8f16_rowwise_meta);
  m.impl("f8f8bf16_rowwise_preshuffle", f8f8bf16_rowwise_meta);
  m.impl("f8f8f16_rowwise_preshuffle", f8f8f16_rowwise_meta);
#endif
}

} // namespace fbgemm_gpu
