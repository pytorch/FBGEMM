/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <torch/library.h>

namespace fbgemm_gpu {

#define DEFAULT_PAGE_SIZE 64
#define STRING_(s) #s
#define STRING(x) STRING_(x)

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.def("rope_qkv_varseq_prefill(Tensor XQ, Tensor(a!)? XK, Tensor? XV, Tensor(b!) cache_K, Tensor(c!) cache_V,  Tensor varseq_batch, Tensor varseq_seqpos, float theta, int? num_groups=1, Tensor? block_tables=None, int page_size=" STRING(
        DEFAULT_PAGE_SIZE) ", Tensor? actual_batch_size=None, Tensor? varseq_cache_seqpos=None, int cache_logical_dtype_int=0, bool rope_scaling=False, int old_context_len=8192"
        ", float scaling_factor=16, float lo_freq_factor=1, float hi_freq_factor=32,  Tensor? qparam_k=None, Tensor? qparam_v=None, bool write_k_back=False, bool k_norm=False,bool update_kv=True, Tensor?amax_qkv=None, Tensor?kv_quant_scale_precomputed=None) -> Tensor");
  m.def("rope_qkv_decoding(Tensor XQ, Tensor? XK, Tensor? XV, Tensor(a!) cache_K, Tensor(b!) cache_V,  Tensor seqpos, float theta, int? num_groups=1, Tensor? block_tables=None, int page_size=" STRING(
      DEFAULT_PAGE_SIZE) ", Tensor? actual_batch_size=None, Tensor? batch=None, Tensor? cache_seqpos=None,  int cache_logical_dtype_int=0, bool rope_scaling=False, int old_context_len=8192, float scaling_factor=16, float lo_freq_factor=1, float hi_freq_factor=32, Tensor? qparam_k=None, Tensor? qparam_v=None, bool k_norm=False, bool update_kv=True, Tensor?amax_qkv=None) -> Tensor");
  m.def("nope_qkv_varseq_prefill(Tensor XQ, Tensor? XK, Tensor? XV, Tensor(a!) cache_K, Tensor(b!) cache_V,  Tensor varseq_batch, Tensor varseq_seqpos, Tensor? block_tables=None, int page_size=" STRING(
      DEFAULT_PAGE_SIZE) ", Tensor? actual_batch_size=None, Tensor? varseq_cache_seqpos=None, int cache_logical_dtype_int=0, int? num_groups=1, Tensor? qparam_k=None, Tensor? qparam_v=None, bool k_norm=False, bool update_kv=True, Tensor?amax_qkv=None, Tensor?kv_quant_scale_precomputed=None) -> Tensor");
  m.def("nope_qkv_decoding(Tensor XQ, Tensor? XK, Tensor? XV, Tensor(a!) cache_K, Tensor(b!) cache_V,  Tensor seqpos, Tensor? block_tables=None, int page_size=" STRING(
      DEFAULT_PAGE_SIZE) ", Tensor? actual_batch_size=None, Tensor? batch=None, Tensor? cache_seqpos=None, int cache_logical_dtype_int=0, int? num_groups=1, Tensor? qparam_k=None, Tensor? qparam_v=None, bool k_norm=False, bool update_kv=True, Tensor?amax_qkv=None) -> Tensor");
  m.def("xpos_qkv_varseq_prefill(Tensor XQ, Tensor XK, Tensor XV, Tensor(a!) cache_K, Tensor(b!) cache_V, Tensor varseq_batch, Tensor varseq_seqpos, float theta, float gamma, float scale_base, float exponent_offset, int? num_groups=1, Tensor? block_tables=None, int page_size=" STRING(
      DEFAULT_PAGE_SIZE) ", Tensor? actual_batch_size=None, Tensor? varseq_cache_seqpos=None, int cache_logical_dtype_int=0, bool rope_scaling=False, int old_context_len=8192, float scaling_factor=16, float lo_freq_factor=1, float hi_freq_factor=32,  Tensor? qparam_k=None, Tensor? qparam_v=None) -> Tensor");
  m.def("xpos_qkv_decoding(Tensor XQ, Tensor XK, Tensor XV, Tensor(a!) cache_K, Tensor(b!) cache_V,  Tensor seqpos, float theta, float gamma, float scale_base, float exponent_offset, int? num_groups=1, Tensor? block_tables=None, int page_size=" STRING(
      DEFAULT_PAGE_SIZE) ", Tensor? actual_batch_size=None, Tensor? batch=None, Tensor? cache_seqpos=None, int cache_logical_dtype_int=0, bool rope_scaling=False, int old_context_len=8192, float scaling_factor=16, float lo_freq_factor=1, float hi_freq_factor=32,  Tensor? qparam_k=None, Tensor? qparam_v=None) -> Tensor");
  m.def(
      "dequantize_int4_cache(Tensor cache_K, Tensor cache_V, Tensor kv_seqlen, int? num_groups=1, Tensor? qparam_k=None, Tensor? qparam_v=None) -> (Tensor, Tensor)");
  m.def(
      "dequantize_fp8_cache(Tensor cache_K, Tensor cache_V, Tensor kv_seqlen, Tensor? qparam_k=None, Tensor? qparam_v=None, Tensor? block_tables=None, int page_size=" STRING(
          DEFAULT_PAGE_SIZE) ") -> (Tensor, Tensor)");
  m.def(
      "quantize_qkv_per_head(Tensor amax, Tensor XQKV, Tensor varseq_seqpos, Tensor? varseq_batch, Tensor? is_precalculated_qparam, Tensor cache_K, Tensor cache_V, Tensor XQ_O, int B, Tensor? qparam_k=None, Tensor? qparam_v=None) -> Tensor");
  m.def(
      "convert_e4m3fn_kv_cache_to_e4m3fnuz_inplace(Tensor cache_K, Tensor cache_V, Tensor qparam_K, Tensor qparam_V) -> ()");
}

} // namespace fbgemm_gpu
