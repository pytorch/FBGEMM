/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "kv_cache.h"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAEvent.h>
#include <torch/library.h>

namespace fbgemm_gpu {

TORCH_LIBRARY_IMPL(fbgemm, CPU, m) {
  m.impl("rope_qkv_varseq_prefill", rope_qkv_varseq_prefill);
  m.impl("rope_qkv_decoding", rope_qkv_decoding);
  m.impl("nope_qkv_varseq_prefill", nope_qkv_varseq_prefill);
  m.impl("nope_qkv_decoding", nope_qkv_decoding);
  m.impl("xpos_qkv_varseq_prefill", xpos_qkv_varseq_prefill);
  m.impl("xpos_qkv_decoding", xpos_qkv_decoding);
  m.impl("dequantize_int4_cache", dequantize_int4_cache);
  m.impl("dequantize_fp8_cache", dequantize_fp8_cache);
  m.impl("quantize_qkv_per_head", quantize_qkv_per_head);
}

TORCH_LIBRARY_IMPL(fbgemm, CUDA, m) {
  m.impl("rope_qkv_varseq_prefill", rope_qkv_varseq_prefill);
  m.impl("rope_qkv_decoding", rope_qkv_decoding);
  m.impl("nope_qkv_varseq_prefill", nope_qkv_varseq_prefill);
  m.impl("nope_qkv_decoding", nope_qkv_decoding);
  m.impl("xpos_qkv_varseq_prefill", xpos_qkv_varseq_prefill);
  m.impl("xpos_qkv_decoding", xpos_qkv_decoding);
  m.impl("dequantize_int4_cache", dequantize_int4_cache);
  m.impl("dequantize_fp8_cache", dequantize_fp8_cache);
  m.impl("quantize_qkv_per_head", quantize_qkv_per_head);
  m.impl(
      "convert_e4m3fn_kv_cache_to_e4m3fnuz_inplace",
      fbgemm_gpu::convert_e4m3fn_kv_cache_to_e4m3fnuz_inplace);
}

at::Tensor rope_qkv_varseq_prefill_meta(
    at::Tensor XQ,
    std::optional<at::Tensor> /* XK */,
    std::optional<at::Tensor> /* XV */,
    at::Tensor /* cache_K */,
    at::Tensor /* cache_V */,
    at::Tensor /* varseq_batch */,
    at::Tensor /* varseq_seqpos */,
    double /* theta */,
    std::optional<int64_t> /* num_groups */,
    std::optional<at::Tensor> /* block_tables */,
    int64_t /* page_size */,
    std::optional<at::Tensor> /* actual_batch_size */,
    std::optional<at::Tensor> /* varseq_cache_seqpos */,
    int64_t /* cache_logical_dtype_int */,
    bool /* rope_scaling */,
    int64_t /* old_context_len */,
    double /* scaling_factor */,
    double /* lo_freq_factor */,
    double /* hi_freq_factor */,
    std::optional<at::Tensor> /* qparam_k */,
    std::optional<at::Tensor> /* qparam_v */,
    bool /* write_k_back */,
    bool /* k_norm */,
    bool /* update_kv */,
    std::optional<at::Tensor> /* amax_qkv */,
    std::optional<at::Tensor> /* kv_quant_scale_precomputed */
) {
  return at::empty_like(XQ);
}

at::Tensor rope_qkv_decoding_meta(
    at::Tensor XQ,
    std::optional<at::Tensor> /* XK */,
    std::optional<at::Tensor> /* XV */,
    at::Tensor /* cache_K */,
    at::Tensor /* cache_V */,
    at::Tensor /* seqpos */,
    double /* theta */,
    std::optional<int64_t> /* num_groups */,
    std::optional<at::Tensor> /* block_tables */,
    int64_t /* page_size */,
    std::optional<at::Tensor> /* actual_batch_size */,
    std::optional<at::Tensor> /* batch */,
    std::optional<at::Tensor> /* cache_seqpos */,
    int64_t /* cache_logical_dtype_int */,
    bool /* rope_scaling */,
    int64_t /* old_context_len */,
    double /* scaling_factor */,
    double /* lo_freq_factor */,
    double /* hi_freq_factor */,
    std::optional<at::Tensor> /* qparam_k */,
    std::optional<at::Tensor> /* qparam_v */,
    bool /* k_norm */,
    bool /* update_kv */,
    std::optional<at::Tensor> /* amax_qkv */
) {
  return at::empty_like(XQ);
}

at::Tensor nope_qkv_varseq_prefill_meta(
    at::Tensor XQ,
    std::optional<at::Tensor> /* XK */,
    std::optional<at::Tensor> /* XV */,
    at::Tensor /* cache_K */,
    at::Tensor /* cache_V */,
    at::Tensor /* varseq_batch */,
    at::Tensor /* varseq_seqpos */,
    std::optional<at::Tensor> /* block_tables */,
    int64_t /* page_size */,
    std::optional<at::Tensor> /* actual_batch_size */,
    std::optional<at::Tensor> /* varseq_cache_seqpos */,
    int64_t /* cache_logical_dtype_int */,
    std::optional<int64_t> /* num_groups */,
    std::optional<at::Tensor> /* qparam_k */,
    std::optional<at::Tensor> /* qparam_v */,
    bool /* k_norm */,
    bool /* update_kv */,
    std::optional<at::Tensor> /* amax_qkv */,
    std::optional<at::Tensor> /* kv_quant_scale_precomputed */
) {
  return at::empty_like(XQ);
}

at::Tensor nope_qkv_decoding_meta(
    at::Tensor XQ,
    std::optional<at::Tensor> /* XK */,
    std::optional<at::Tensor> /* XV */,
    at::Tensor /* cache_K */,
    at::Tensor /* cache_V */,
    at::Tensor /* seqpos */,
    std::optional<at::Tensor> /* block_tables */,
    int64_t /* page_size */,
    std::optional<at::Tensor> /* actual_batch_size */,
    std::optional<at::Tensor> /* batch */,
    std::optional<at::Tensor> /* cache_seqpos */,
    int64_t /* cache_logical_dtype_int */,
    std::optional<int64_t> /* num_groups */,
    std::optional<at::Tensor> /* qparam_k */,
    std::optional<at::Tensor> /* qparam_v */,
    bool /* k_norm */,
    bool /* update_kv */,
    std::optional<at::Tensor> /* amax_qkv */
) {
  return at::empty_like(XQ);
}

at::Tensor xpos_qkv_varseq_prefill_meta(
    at::Tensor XQ,
    at::Tensor /* XK */,
    at::Tensor /* XV */,
    at::Tensor /* cache_K */,
    at::Tensor /* cache_V */,
    at::Tensor /* varseq_batch */,
    at::Tensor /* varseq_seqpos */,
    double /* theta */,
    double /* gamma */,
    double /* scale_base */,
    double /* exponent_offset */,
    std::optional<int64_t> /* num_groups */,
    std::optional<at::Tensor> /* block_tables */,
    int64_t /* page_size */,
    std::optional<at::Tensor> /* actual_batch_size */,
    std::optional<at::Tensor> /* varseq_cache_seqpos */,
    int64_t /* cache_logical_dtype_int */,
    bool /* rope_scaling */,
    int64_t /* old_context_len */,
    double /* scaling_factor */,
    double /* lo_freq_factor */,
    double /* hi_freq_factor */,
    std::optional<at::Tensor> /* qparam_k */,
    std::optional<at::Tensor> /* qparam_v */
) {
  return at::empty_like(XQ);
}

at::Tensor xpos_qkv_decoding_meta(
    at::Tensor XQ,
    at::Tensor /* XK */,
    at::Tensor /* XV */,
    at::Tensor /* cache_K */,
    at::Tensor /* cache_V */,
    at::Tensor /* seqpos */,
    double /* theta */,
    double /* gamma */,
    double /* scale_base */,
    double /* exponent_offset */,
    std::optional<int64_t> /* num_groups */,
    std::optional<at::Tensor> /* block_tables */,
    int64_t /* page_size */,
    std::optional<at::Tensor> /* actual_batch_size */,
    std::optional<at::Tensor> /* batch */,
    std::optional<at::Tensor> /* cache_seqpos */,
    int64_t /* cache_logical_dtype_int */,
    bool /* rope_scaling */,
    int64_t /* old_context_len */,
    double /* scaling_factor */,
    double /* lo_freq_factor */,
    double /* hi_freq_factor */,
    std::optional<at::Tensor> /* qparam_k */,
    std::optional<at::Tensor> /* qparam_v */
) {
  return at::empty_like(XQ);
}

std::tuple<at::Tensor, at::Tensor> dequantize_int4_cache_meta(
    at::Tensor cache_K,
    at::Tensor /* cache_V */,
    at::Tensor /* kv_seqlen */,
    std::optional<int64_t> num_groups,
    std::optional<at::Tensor> /* qparam_k */,
    std::optional<at::Tensor> /* qparam_v */) {
  const at::SymInt B = cache_K.sym_size(0);
  const at::SymInt MAX_T = cache_K.sym_size(1);
  const at::SymInt N_KVH = cache_K.sym_size(2);
  const at::SymInt D_HQ = cache_K.sym_size(3);
  auto num_groups_ = num_groups ? num_groups.value() : 1;
  auto int4_qparam_offset = 4 * num_groups_;
  const at::SymInt D_H = (D_HQ - int4_qparam_offset) * 2;
  auto cache_K_dq = at::empty_symint(
      {B, MAX_T, N_KVH, D_H}, cache_K.options().dtype(at::kBFloat16));
  auto cache_V_dq = at::empty_symint(
      {B, MAX_T, N_KVH, D_H}, cache_K.options().dtype(at::kBFloat16));
  return {cache_K_dq, cache_V_dq};
}

std::tuple<at::Tensor, at::Tensor> dequantize_fp8_cache_meta(
    at::Tensor cache_K,
    at::Tensor /* cache_V */,
    at::Tensor /* kv_seqlen */,
    std::optional<at::Tensor> qparam_k,
    std::optional<at::Tensor> /* qparam_v */,
    std::optional<at::Tensor> /* block_tables */,
    int64_t /* page_size */) {
  const at::SymInt B_KV = cache_K.sym_size(0);
  const at::SymInt MAX_T = cache_K.sym_size(1);
  const at::SymInt N_KVH = cache_K.sym_size(2);
  const at::SymInt D_HQ = cache_K.sym_size(3);
  auto fp8_qparam_offset = qparam_k ? 0 : 4;
  const at::SymInt D_H = (D_HQ - fp8_qparam_offset);
  auto cache_K_dq = at::empty_symint(
      {B_KV, MAX_T, N_KVH, D_H}, cache_K.options().dtype(at::kBFloat16));
  auto cache_V_dq = at::empty_symint(
      {B_KV, MAX_T, N_KVH, D_H}, cache_K.options().dtype(at::kBFloat16));
  return {cache_K_dq, cache_V_dq};
}

at::Tensor quantize_qkv_per_head_meta(
    at::Tensor /* amax */,
    at::Tensor XQKV,
    at::Tensor /* varseq_seqpos */,
    std::optional<at::Tensor> /* varseq_batch */,
    std::optional<at::Tensor> /* is_precalculated_qparam */,
    at::Tensor cache_K /* cache_K */,
    at::Tensor /* cache_V */,
    at::Tensor /* XQ_O */,
    int64_t /* B */,
    std::optional<at::Tensor> /* qparam_k */,
    std::optional<at::Tensor> /* qparam_v */) {
  const at::SymInt B_KV = cache_K.sym_size(0);
  const at::SymInt N_KVH = cache_K.sym_size(2);
  auto xq_scale =
      at::empty_symint({B_KV, N_KVH}, cache_K.options().dtype(at::kFloat));
  return at::empty_like(XQKV);
}

void convert_e4m3fn_kv_cache_to_e4m3fnuz_inplace_meta(
    at::Tensor cache_K,
    at::Tensor cache_V,
    at::Tensor qparam_K,
    at::Tensor qparam_v) {};

TORCH_LIBRARY_IMPL(fbgemm, Meta, m) {
  m.impl("rope_qkv_varseq_prefill", rope_qkv_varseq_prefill_meta);
  m.impl("rope_qkv_decoding", rope_qkv_decoding_meta);
  m.impl("nope_qkv_varseq_prefill", nope_qkv_varseq_prefill_meta);
  m.impl("nope_qkv_decoding", nope_qkv_decoding_meta);
  m.impl("xpos_qkv_varseq_prefill", xpos_qkv_varseq_prefill_meta);
  m.impl("xpos_qkv_decoding", xpos_qkv_decoding_meta);
  m.impl("dequantize_int4_cache", dequantize_int4_cache_meta);
  m.impl("dequantize_fp8_cache", dequantize_fp8_cache_meta);
  m.impl(
      "convert_e4m3fn_kv_cache_to_e4m3fnuz_inplace",
      convert_e4m3fn_kv_cache_to_e4m3fnuz_inplace_meta);
}

} // namespace fbgemm_gpu
