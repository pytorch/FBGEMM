/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>

#include <torch/library.h>

#include <ATen/cuda/CUDAEvent.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <vector>

#include "c10/util/Exception.h"

namespace fbgemm_gpu {

#define DEFAULT_PAGE_SIZE 64
#define STRING_(s) #s
#define STRING(x) STRING_(x)

at::Tensor nope_qkv_varseq_prefill(
    at::Tensor XQ,
    at::Tensor XK,
    at::Tensor XV,
    at::Tensor cache_K,
    at::Tensor cache_V,
    at::Tensor varseq_batch,
    at::Tensor varseq_seqpos,
    std::optional<at::Tensor> block_tables,
    int64_t page_size,
    std::optional<at::Tensor> varseq_cache_seqpos,
    std::optional<at::Tensor> qparam_k,
    std::optional<at::Tensor> qparam_v,
    bool k_norm);

at::Tensor nope_qkv_decoding(
    at::Tensor XQ,
    at::Tensor XK,
    at::Tensor XV,
    at::Tensor cache_K,
    at::Tensor cache_V,
    at::Tensor seqpos,
    std::optional<at::Tensor> block_tables,
    int64_t page_size,
    std::optional<at::Tensor> actual_batch_size,
    std::optional<at::Tensor> batch,
    std::optional<at::Tensor> cache_seqpos,
    std::optional<at::Tensor> qparam_k,
    std::optional<at::Tensor> qparam_v,
    bool k_norm);

at::Tensor rope_qkv_varseq_prefill(
    at::Tensor XQ,
    at::Tensor XK,
    at::Tensor XV,
    at::Tensor cache_K,
    at::Tensor cache_V,
    at::Tensor varseq_batch,
    at::Tensor varseq_seqpos,
    double theta,
    std::optional<int64_t> num_groups,
    std::optional<at::Tensor> block_tables,
    int64_t page_size,
    std::optional<at::Tensor> varseq_cache_seqpos,
    int64_t cache_logical_dtype_int,
    bool rope_scaling,
    int64_t old_context_len,
    double scaling_factor,
    double lo_freq_factor,
    double hi_freq_factor,
    std::optional<at::Tensor> qparam_k,
    std::optional<at::Tensor> qparam_v,
    bool write_k_back,
    bool k_norm);

at::Tensor rope_qkv_decoding(
    at::Tensor XQ,
    at::Tensor XK,
    at::Tensor XV,
    at::Tensor cache_K,
    at::Tensor cache_V,
    at::Tensor seqpos,
    double theta,
    std::optional<int64_t> num_groups,
    std::optional<at::Tensor> block_tables,
    int64_t page_size,
    std::optional<at::Tensor> actual_batch_size,
    std::optional<at::Tensor> batch,
    std::optional<at::Tensor> cache_seqpos,
    int64_t cache_logical_dtype_int,
    bool rope_scaling,
    int64_t old_context_len,
    double scaling_factor,
    double lo_freq_factor,
    double hi_freq_factor,
    std::optional<at::Tensor> qparam_k,
    std::optional<at::Tensor> qparam_v,
    bool k_norm);

at::Tensor xpos_qkv_varseq_prefill(
    at::Tensor XQ,
    at::Tensor XK,
    at::Tensor XV,
    at::Tensor cache_K,
    at::Tensor cache_V,
    at::Tensor varseq_batch,
    at::Tensor varseq_seqpos,
    double theta,
    double gamma,
    double scale_base,
    double exponent_offset,
    std::optional<int64_t> num_groups,
    std::optional<at::Tensor> block_tables,
    int64_t page_size,
    std::optional<at::Tensor> varseq_cache_seqpos,
    int64_t cache_logical_dtype_int,
    bool rope_scaling,
    int64_t old_context_len,
    double scaling_factor,
    double lo_freq_factor,
    double hi_freq_factor,
    std::optional<at::Tensor> qparam_k,
    std::optional<at::Tensor> qparam_v);

at::Tensor xpos_qkv_decoding(
    at::Tensor XQ,
    at::Tensor XK,
    at::Tensor XV,
    at::Tensor cache_K,
    at::Tensor cache_V,
    at::Tensor seqpos,
    double theta,
    double gamma,
    double scale_base,
    double exponent_offset,
    std::optional<int64_t> num_groups,
    std::optional<at::Tensor> block_tables,
    int64_t page_size,
    std::optional<at::Tensor> actual_batch_size,
    std::optional<at::Tensor> batch,
    std::optional<at::Tensor> cache_seqpos,
    int64_t cache_logical_dtype_int,
    bool rope_scaling,
    int64_t old_context_len,
    double scaling_factor,
    double lo_freq_factor,
    double hi_freq_factor,
    std::optional<at::Tensor> qparam_k,
    std::optional<at::Tensor> qparam_v);

std::tuple<at::Tensor, at::Tensor> dequantize_int4_cache(
    at::Tensor cache_K,
    at::Tensor cache_V,
    at::Tensor kv_seqlen,
    std::optional<int64_t> num_groups);

std::tuple<at::Tensor, at::Tensor> dequantize_fp8_cache(
    at::Tensor cache_K,
    at::Tensor cache_V,
    at::Tensor kv_seqlen,
    std::optional<at::Tensor> qparam_k,
    std::optional<at::Tensor> qparam_v,
    std::optional<at::Tensor> block_tables,
    int64_t page_size);

at::Tensor mqa_attn(
    at::Tensor XQ,
    at::Tensor cache_K,
    at::Tensor cache_V,
    at::Tensor seq_positions,
    double qk_scale,
    std::optional<int64_t> num_groups,
    int64_t cache_logical_dtype_int);

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.def("rope_qkv_varseq_prefill(Tensor XQ, Tensor(a!) XK, Tensor XV, Tensor(b!) cache_K, Tensor(c!) cache_V,  Tensor varseq_batch, Tensor varseq_seqpos, float theta, int? num_groups=1, Tensor? block_tables=None, int page_size=" STRING(
      DEFAULT_PAGE_SIZE) ", Tensor? varseq_cache_seqpos=None, int cache_logical_dtype_int=0, bool rope_scaling=False, int old_context_len=8192"
      ", float scaling_factor=16, float lo_freq_factor=1, float hi_freq_factor=32,  Tensor? qparam_k=None, Tensor? qparam_v=None, bool write_k_back=False, bool k_norm=False) -> Tensor");
  m.def("rope_qkv_decoding(Tensor XQ, Tensor XK, Tensor XV, Tensor(a!) cache_K, Tensor(b!) cache_V,  Tensor seqpos, float theta, int? num_groups=1, Tensor? block_tables=None, int page_size=" STRING(
      DEFAULT_PAGE_SIZE) ", Tensor? actual_batch_size=None, Tensor? batch=None, Tensor? cache_seqpos=None,  int cache_logical_dtype_int=0, bool rope_scaling=False, int old_context_len=8192, float scaling_factor=16, float lo_freq_factor=1, float hi_freq_factor=32, Tensor? qparam_k=None, Tensor? qparam_v=None, bool k_norm=False) -> Tensor");
  m.def("nope_qkv_varseq_prefill(Tensor XQ, Tensor XK, Tensor XV, Tensor(a!) cache_K, Tensor(b!) cache_V,  Tensor varseq_batch, Tensor varseq_seqpos, Tensor? block_tables=None, int page_size=" STRING(
      DEFAULT_PAGE_SIZE) ", Tensor? varseq_cache_seqpos=None, Tensor? qparam_k=None, Tensor? qparam_v=None, bool k_norm=False) -> Tensor");
  m.def("nope_qkv_decoding(Tensor XQ, Tensor XK, Tensor XV, Tensor(a!) cache_K, Tensor(b!) cache_V,  Tensor seqpos, Tensor? block_tables=None, int page_size=" STRING(
      DEFAULT_PAGE_SIZE) ", Tensor? actual_batch_size=None, Tensor? batch=None, Tensor? cache_seqpos=None, Tensor? qparam_k=None, Tensor? qparam_v=None, bool k_norm=False) -> Tensor");
  m.def("xpos_qkv_varseq_prefill(Tensor XQ, Tensor XK, Tensor XV, Tensor(a!) cache_K, Tensor(b!) cache_V, Tensor varseq_batch, Tensor varseq_seqpos, float theta, float gamma, float scale_base, float exponent_offset, int? num_groups=1, Tensor? block_tables=None, int page_size=" STRING(
      DEFAULT_PAGE_SIZE) ", Tensor? varseq_cache_seqpos=None, int cache_logical_dtype_int=0, bool rope_scaling=False, int old_context_len=8192, float scaling_factor=16, float lo_freq_factor=1, float hi_freq_factor=32,  Tensor? qparam_k=None, Tensor? qparam_v=None) -> Tensor");
  m.def("xpos_qkv_decoding(Tensor XQ, Tensor XK, Tensor XV, Tensor(a!) cache_K, Tensor(b!) cache_V,  Tensor seqpos, float theta, float gamma, float scale_base, float exponent_offset, int? num_groups=1, Tensor? block_tables=None, int page_size=" STRING(
      DEFAULT_PAGE_SIZE) ", Tensor? actual_batch_size=None, Tensor? batch=None, Tensor? cache_seqpos=None, int cache_logical_dtype_int=0, bool rope_scaling=False, int old_context_len=8192, float scaling_factor=16, float lo_freq_factor=1, float hi_freq_factor=32,  Tensor? qparam_k=None, Tensor? qparam_v=None) -> Tensor");
  m.def(
      "dequantize_int4_cache(Tensor cache_K, Tensor cache_V, Tensor kv_seqlen, int? num_groups=1) -> (Tensor, Tensor)");
  m.def(
      "dequantize_fp8_cache(Tensor cache_K, Tensor cache_V, Tensor kv_seqlen, Tensor? qparam_k=None, Tensor? qparam_v=None, Tensor? block_tables=None, int page_size=" STRING(
          DEFAULT_PAGE_SIZE) ") -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(fbgemm, CPU, m) {
  m.impl("rope_qkv_varseq_prefill", rope_qkv_varseq_prefill);
  m.impl("rope_qkv_decoding", rope_qkv_decoding);
  m.impl("nope_qkv_varseq_prefill", nope_qkv_varseq_prefill);
  m.impl("nope_qkv_decoding", nope_qkv_decoding);
  m.impl("xpos_qkv_varseq_prefill", xpos_qkv_varseq_prefill);
  m.impl("xpos_qkv_decoding", xpos_qkv_decoding);
  m.impl("dequantize_int4_cache", dequantize_int4_cache);
  m.impl("dequantize_fp8_cache", dequantize_fp8_cache);
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
}

at::Tensor rope_qkv_varseq_prefill_meta(
    at::Tensor XQ,
    at::Tensor /* XK */,
    at::Tensor /* XV */,
    at::Tensor /* cache_K */,
    at::Tensor /* cache_V */,
    at::Tensor /* varseq_batch */,
    at::Tensor /* varseq_seqpos */,
    double /* theta */,
    std::optional<int64_t> /* num_groups */,
    std::optional<at::Tensor> /* block_tables */,
    int64_t /* page_size */,
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
    bool /* k_norm */
) {
  return at::empty_like(XQ);
}

at::Tensor rope_qkv_decoding_meta(
    at::Tensor XQ,
    at::Tensor /* XK */,
    at::Tensor /* XV */,
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
    bool /* k_norm */
) {
  return at::empty_like(XQ);
}

at::Tensor nope_qkv_varseq_prefill_meta(
    at::Tensor XQ,
    at::Tensor /* XK */,
    at::Tensor /* XV */,
    at::Tensor /* cache_K */,
    at::Tensor /* cache_V */,
    at::Tensor /* varseq_batch */,
    at::Tensor /* varseq_seqpos */,
    std::optional<at::Tensor> /* block_tables */,
    int64_t /* page_size */,
    std::optional<at::Tensor> /* varseq_cache_seqpos */,
    std::optional<at::Tensor> /* qparam_k */,
    std::optional<at::Tensor> /* qparam_v */,
    bool /* k_norm */
) {
  return at::empty_like(XQ);
}

at::Tensor nope_qkv_decoding_meta(
    at::Tensor XQ,
    at::Tensor /* XK */,
    at::Tensor /* XV */,
    at::Tensor /* cache_K */,
    at::Tensor /* cache_V */,
    at::Tensor /* seqpos */,
    std::optional<at::Tensor> /* block_tables */,
    int64_t /* page_size */,
    std::optional<at::Tensor> /* actual_batch_size */,
    std::optional<at::Tensor> /* batch */,
    std::optional<at::Tensor> /* cache_seqpos */,
    std::optional<at::Tensor> /* qparam_k */,
    std::optional<at::Tensor> /* qparam_v */,
    bool /* k_norm */
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
    std::optional<int64_t> num_groups) {
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

TORCH_LIBRARY_IMPL(fbgemm, Meta, m) {
  m.impl("rope_qkv_varseq_prefill", rope_qkv_varseq_prefill_meta);
  m.impl("rope_qkv_decoding", rope_qkv_decoding_meta);
  m.impl("nope_qkv_varseq_prefill", nope_qkv_varseq_prefill_meta);
  m.impl("nope_qkv_decoding", nope_qkv_decoding_meta);
  m.impl("xpos_qkv_varseq_prefill", xpos_qkv_varseq_prefill_meta);
  m.impl("xpos_qkv_decoding", xpos_qkv_decoding_meta);
  m.impl("dequantize_int4_cache", dequantize_int4_cache_meta);
  m.impl("dequantize_fp8_cache", dequantize_fp8_cache_meta);
}

} // namespace fbgemm_gpu
