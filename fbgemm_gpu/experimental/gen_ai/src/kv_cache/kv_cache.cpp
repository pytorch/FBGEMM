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
    std::optional<at::Tensor> qparam_v);

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
    std::optional<at::Tensor> qparam_v);

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
  m.def("rope_qkv_varseq_prefill(Tensor XQ, Tensor XK, Tensor XV, Tensor(a!) cache_K, Tensor(b!) cache_V,  Tensor varseq_batch, Tensor varseq_seqpos, float theta, int? num_groups=1, Tensor? block_tables=None, int page_size=" STRING(
      DEFAULT_PAGE_SIZE) ", Tensor? varseq_cache_seqpos=None, int cache_logical_dtype_int=0, bool rope_scaling=False, int old_context_len=8192, float scaling_factor=16, float lo_freq_factor=1, float hi_freq_factor=32,  Tensor? qparam_k=None, Tensor? qparam_v=None) -> Tensor");
  m.impl("rope_qkv_varseq_prefill", rope_qkv_varseq_prefill);
  m.def("rope_qkv_decoding(Tensor XQ, Tensor XK, Tensor XV, Tensor(a!) cache_K, Tensor(b!) cache_V,  Tensor seqpos, float theta, int? num_groups=1, Tensor? block_tables=None, int page_size=" STRING(
      DEFAULT_PAGE_SIZE) ", Tensor? actual_batch_size=None, Tensor? batch=None, Tensor? cache_seqpos=None,  int cache_logical_dtype_int=0, bool rope_scaling=False, int old_context_len=8192, float scaling_factor=16, float lo_freq_factor=1, float hi_freq_factor=32, Tensor? qparam_k=None, Tensor? qparam_v=None) -> Tensor");
  m.impl("rope_qkv_decoding", rope_qkv_decoding);
  m.def("xpos_qkv_varseq_prefill(Tensor XQ, Tensor XK, Tensor XV, Tensor(a!) cache_K, Tensor(b!) cache_V, Tensor varseq_batch, Tensor varseq_seqpos, float theta, float gamma, float scale_base, float exponent_offset, int? num_groups=1, Tensor? block_tables=None, int page_size=" STRING(
      DEFAULT_PAGE_SIZE) ", Tensor? varseq_cache_seqpos=None, int cache_logical_dtype_int=0, bool rope_scaling=False, int old_context_len=8192, float scaling_factor=16, float lo_freq_factor=1, float hi_freq_factor=32,  Tensor? qparam_k=None, Tensor? qparam_v=None) -> Tensor");
  m.impl("xpos_qkv_varseq_prefill", xpos_qkv_varseq_prefill);
  m.def("xpos_qkv_decoding(Tensor XQ, Tensor XK, Tensor XV, Tensor(a!) cache_K, Tensor(b!) cache_V,  Tensor seqpos, float theta, float gamma, float scale_base, float exponent_offset, int? num_groups=1, Tensor? block_tables=None, int page_size=" STRING(
      DEFAULT_PAGE_SIZE) ", Tensor? actual_batch_size=None, Tensor? batch=None, Tensor? cache_seqpos=None, int cache_logical_dtype_int=0, bool rope_scaling=False, int old_context_len=8192, float scaling_factor=16, float lo_freq_factor=1, float hi_freq_factor=32,  Tensor? qparam_k=None, Tensor? qparam_v=None) -> Tensor");
  m.impl("xpos_qkv_decoding", xpos_qkv_decoding);

  m.def(
      "dequantize_int4_cache(Tensor cache_K, Tensor cache_V, Tensor kv_seqlen, int? num_groups=1) -> (Tensor, Tensor)");
  m.impl("dequantize_int4_cache", dequantize_int4_cache);
  m.def(
      "dequantize_fp8_cache(Tensor cache_K, Tensor cache_V, Tensor kv_seqlen, Tensor? qparam_k=None, Tensor? qparam_v=None, Tensor? block_tables=None, int page_size=" STRING(
          DEFAULT_PAGE_SIZE) ") -> (Tensor, Tensor)");
  m.impl("dequantize_fp8_cache", dequantize_fp8_cache);
}

} // namespace fbgemm_gpu
