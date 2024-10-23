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
at::Tensor f8f8bf16_rowwise(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias = std::nullopt,
    bool use_fast_accum = true,
    std::optional<at::Tensor> output = std::nullopt);
at::Tensor f8f8bf16_rowwise_batched(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias = std::nullopt,
    bool use_fast_accum = true,
    std::optional<at::Tensor> output = std::nullopt);
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
  m.set_python_module("fbgemm_gpu.experimental.gen_ai.custom_ops.quantize_ops");

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
      "f8f8bf16_rowwise_batched(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias=None, bool use_fast_accum=True, Tensor(a!)? output=None) -> Tensor");
  m.def(
      "bf16i4bf16_rowwise(Tensor X, Tensor WQ, Tensor w_scale, Tensor w_zp) -> Tensor");
  m.def(
      "bf16i4bf16_rowwise_batched(Tensor X, Tensor WQ, Tensor w_scale, Tensor w_zp) -> Tensor");
  m.def(
      "i8i8bf16_dynamic(Tensor XQ, Tensor WQ, Tensor scale, int split_k=1) -> Tensor");
  m.impl("i8i8bf16_dynamic", i8i8bf16_dynamic);
#endif
  m.def(
      "f8f8bf16_blockwise(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, int block_m=128, int block_n=128, int block_k=128) -> Tensor");
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
  m.impl("quantize_fp8_per_tensor", quantize_fp8_per_tensor);
  m.impl("quantize_fp8_per_row", quantize_fp8_per_row);
  m.impl("quantize_fp8_per_col", quantize_fp8_per_col);
#ifndef USE_ROCM
  m.impl("i8i8bf16", i8i8bf16);
  m.impl("f8f8bf16", f8f8bf16);
  m.impl("f8f8bf16_cublas", f8f8bf16_cublas);
  m.impl("f8f8bf16_rowwise_batched", f8f8bf16_rowwise_batched);
  m.impl("f8i4bf16_rowwise", f8i4bf16_rowwise);
  m.impl("bf16i4bf16_rowwise_batched", bf16i4bf16_rowwise_batched);
  m.impl("bf16i4bf16_rowwise", bf16i4bf16_rowwise);
#endif
}

// Though it should never be used, it still seems helpful to define these
// functions for CPU to accomodate model creation.
TORCH_LIBRARY_IMPL(fbgemm, CPU, m) {
  m.impl("f8f8bf16_blockwise", f8f8bf16_blockwise);
  m.impl("f8f8bf16_tensorwise", f8f8bf16_tensorwise);
  m.impl("f8f8bf16_rowwise", f8f8bf16_rowwise);
  m.impl("quantize_fp8_per_tensor", quantize_fp8_per_tensor);
  m.impl("quantize_fp8_per_row", quantize_fp8_per_row);
  m.impl("quantize_fp8_per_col", quantize_fp8_per_col);
#ifndef USE_ROCM
  m.impl("i8i8bf16", i8i8bf16);
  m.impl("f8f8bf16", f8f8bf16);
  m.impl("f8f8bf16_cublas", f8f8bf16_cublas);
  m.impl("f8f8bf16_rowwise_batched", f8f8bf16_rowwise_batched);
  m.impl("f8i4bf16_rowwise", f8i4bf16_rowwise);
  m.impl("bf16i4bf16_rowwise_batched", bf16i4bf16_rowwise_batched);
  m.impl("bf16i4bf16_rowwise", bf16i4bf16_rowwise);
#endif
}

} // namespace fbgemm_gpu
