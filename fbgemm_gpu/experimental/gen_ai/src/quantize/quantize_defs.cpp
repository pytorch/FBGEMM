/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <torch/library.h>

namespace fbgemm_gpu {

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.set_python_module("fbgemm_gpu.experimental.gen_ai.quantize_ops");

#ifndef USE_ROCM
  // TODO: on AMD this throws "Undefined symbol" when loading
  // quantize_ops with
  // torch.ops.load_library, similar to below for quantize_fp8_per_tensor
  m.def("i8i8bf16(Tensor XQ, Tensor WQ, float scale, int split_k=1) -> Tensor");
  m.def(
      "f4f4bf16(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? global_scale=None, bool use_mx=True) -> Tensor");
  m.def(
      "f4f4bf16_grouped(Tensor[] XQ, Tensor[] WQ, Tensor[] x_scale, Tensor[] w_scale, Tensor[]? global_scale=None, bool use_mx=True) -> Tensor[]");
  m.def(
      "f4f4bf16_grouped_stacked(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor M_sizes, Tensor? global_scale=None, Tensor? starting_row_after_padding=None, bool use_mx=True) -> Tensor");
  m.def(
      "mx8mx8bf16_grouped_mm(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor offsets, Tensor(a!)? output=None) -> Tensor");
  m.def(
      "f8f8bf16(Tensor XQ, Tensor WQ, Tensor scale, bool use_fast_accum=True) -> Tensor");
  m.def(
      "f8f8bf16_groupwise(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale) -> Tensor");
  m.def(
      "f8f8bf16_groupwise_grouped(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor M_sizes) -> Tensor");
  m.def(
      "f8f8bf16_cublas(Tensor A, Tensor B, Tensor? Ainvs=None, Tensor? Binvs=None, bool use_fast_accum=True, Tensor(a!)? output=None) -> Tensor");
  m.def(
      "f8i4bf16_rowwise(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor w_zp) -> Tensor");
  m.def(
      "f8i4bf16_shuffled(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor w_scale_group) -> Tensor");
  m.def(
      "bf16i4bf16_shuffled(Tensor X, Tensor W, Tensor w_scale_group, Tensor w_zero_group) -> Tensor");
  m.def(
      "f8i4bf16_shuffled_grouped(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor w_scale_group, Tensor M_sizes) -> Tensor");
  m.def(
      "bf16i4bf16_shuffled_grouped(Tensor X, Tensor WQ, Tensor w_scale_group, Tensor w_zero_group, Tensor M_sizes) -> Tensor");
  m.def("preshuffle_i4(Tensor WQ, Tensor w_scale) -> (Tensor, Tensor)");
  m.def("bf16_fast_gemv(Tensor X, Tensor W) -> Tensor");
  m.def("bf16fp8bf16_fast_gemv(Tensor X, Tensor W, Tensor w_scale) -> Tensor");
  m.def(
      "fp8fp8bf16_fast_gemv(Tensor X, Tensor W, Tensor x_scale, Tensor w_scale, bool is_batched=False) -> Tensor");
  m.def("f8f8bf16_lite(Tensor XQ, Tensor WQ, Tensor scale) -> Tensor");
  m.def(
      "bf16i4bf16_rowwise(Tensor X, Tensor W, Tensor w_scale_group, Tensor w_zero_group) -> Tensor");
  m.def(
      "bf16i4bf16_shuffled_batched(Tensor X, Tensor WQ, Tensor w_scale, Tensor w_zp) -> Tensor");
  m.def(
      "bf16i4bf16_rowwise_batched(Tensor X, Tensor WQ, Tensor w_scale, Tensor w_zp) -> Tensor");
  m.def(
      "i8i8bf16_dynamic(Tensor XQ, Tensor WQ, Tensor scale, int split_k=1) -> Tensor");

#endif
  m.def("bf16bf16bf16_grouped(Tensor[] X, Tensor[] W) -> Tensor[]");
  m.def("bf16bf16bf16_grouped_cat(Tensor[] X, Tensor[] W) -> Tensor");
  m.def(
      "bf16bf16bf16_grouped_dynamic(Tensor X, Tensor W, Tensor zero_start_index_M) -> Tensor");
  m.def(
      "bf16bf16bf16_grouped_stacked(Tensor X, Tensor W, Tensor M_sizes) -> Tensor");
  m.def(
      "f8f8bf16_blockwise(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, int block_m=128, int block_n=128, int block_k=128) -> Tensor");
  m.def(
      "f8f8bf16_rowwise(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias=None, bool use_fast_accum=True) -> Tensor");
  m.def(
      "f8f8bf16_rowwise_out(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor(a!) output, Tensor? bias=None, bool use_fast_accum=True) -> ()");
  m.def(
      "f8f8bf16_rowwise_batched(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias=None, bool use_fast_accum=True, Tensor(a!)? output=None) -> Tensor");
  m.def(
      "f8f8bf16_rowwise_grouped(Tensor[] XQ, Tensor[] WQ, Tensor[] x_scale, Tensor[] w_scale) -> Tensor[]");
  m.def(
      "f8f8bf16_rowwise_grouped_cat(Tensor[] XQ, Tensor[] WQ, Tensor[] x_scale, Tensor[] w_scale) -> Tensor");
  m.def(
      "f8f8bf16_rowwise_grouped_stacked(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor M_sizes) -> Tensor");
  m.def(
      "f8f8bf16_rowwise_grouped_dynamic(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor zero_start_index_M, bool zeroing_output_tensor=True) -> Tensor");
  m.def(
      "f8f8bf16_tensorwise(Tensor XQ, Tensor WQ, float scale, bool use_fast_accum=True) -> Tensor");
  m.def("per_tensor_quantize_i8(Tensor X, float scale) -> Tensor");
  m.def("per_tensor_dynamic_quantize_i8(Tensor X) -> (Tensor, Tensor)");

  m.def("silu_mul_quantize_i8(Tensor X1, Tensor X2, float scale) -> Tensor");

  m.def(
      "quantize_fp8_per_tensor(Tensor input, Tensor? bs=None, Tensor? scale_ub=None, bool stochastic_rounding=False) -> Tensor[]");
  m.def(
      "quantize_fp8_per_row(Tensor input, Tensor? bs=None, Tensor? scale_ub=None, ScalarType? output_dtype=None, bool stochastic_rounding = False) -> Tensor[] ");

  m.def(
      "quantize_fp8_per_col(Tensor input, Tensor? bs=None, Tensor? scale_ub=None) -> Tensor[]");

  m.def(
      "get_fp8_per_tensor_scale(Tensor input, Tensor? bs=None, Tensor? scale_ub=None) -> Tensor");

  m.def(
      "quantize_fp8_per_tensor_fixed_scale(Tensor input, Tensor scale, Tensor? bs=None, bool stochatic_rounding=False) -> Tensor");

  m.def(
      "scaled_fp4_quant(Tensor! output, Tensor input, Tensor! output_scale, Tensor input_scale) -> ()");

  m.def(
      "fake_quantize_nvfp4_per_tensor(Tensor input, Tensor? static_scales=None, Tensor? bs=None, Tensor? scale_ub=None) -> Tensor[]");

#ifdef USE_ROCM
  m.def("flush_icache_hip() -> ()");
  m.def(
      "f8f8f16_rowwise(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias=None, bool use_fast_accum=True) -> Tensor");
  m.def(
      "f8f8bf16_rowwise_preshuffle(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias=None, bool use_fast_accum=True) -> Tensor");
  m.def(
      "f8f8f16_rowwise_preshuffle(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias=None, bool use_fast_accum=True) -> Tensor");
  // Generic PyTorch grouped GEMM API is only available on AMD for now.
  m.def(
      "f8f8bf16_rowwise_grouped_mm(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? offsets, Tensor(a!) output) -> Tensor");
#endif
}

} // namespace fbgemm_gpu
