/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>

namespace fbgemm_gpu {

at::Tensor nope_qkv_varseq_prefill(
    at::Tensor XQ,
    std::optional<at::Tensor> XK,
    std::optional<at::Tensor> XV,
    at::Tensor cache_K,
    at::Tensor cache_V,
    at::Tensor varseq_batch,
    at::Tensor varseq_seqpos,
    std::optional<at::Tensor> block_tables,
    int64_t page_size,
    std::optional<at::Tensor> varseq_cache_seqpos,
    int64_t cache_logical_dtype_int,
    std::optional<int64_t> num_groups,
    std::optional<at::Tensor> qparam_k,
    std::optional<at::Tensor> qparam_v,
    bool k_norm,
    bool update_kv,
    std::optional<at::Tensor> amax_qkv);

at::Tensor nope_qkv_decoding(
    at::Tensor XQ,
    std::optional<at::Tensor> XK,
    std::optional<at::Tensor> XV,
    at::Tensor cache_K,
    at::Tensor cache_V,
    at::Tensor seqpos,
    std::optional<at::Tensor> block_tables,
    int64_t page_size,
    std::optional<at::Tensor> actual_batch_size,
    std::optional<at::Tensor> batch,
    std::optional<at::Tensor> cache_seqpos,
    int64_t cache_logical_dtype_int,
    std::optional<int64_t> num_groups,
    std::optional<at::Tensor> qparam_k,
    std::optional<at::Tensor> qparam_v,
    bool k_norm,
    bool update_kv,
    std::optional<at::Tensor> amax_qkv);

at::Tensor rope_qkv_varseq_prefill(
    at::Tensor XQ,
    std::optional<at::Tensor> XK,
    std::optional<at::Tensor> XV,
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
    bool k_norm,
    bool update_kv,
    std::optional<at::Tensor> amax_qkv);

at::Tensor rope_qkv_decoding(
    at::Tensor XQ,
    std::optional<at::Tensor> XK,
    std::optional<at::Tensor> XV,
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
    bool k_norm,
    bool update_kv,
    std::optional<at::Tensor> amax_qkv);

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
    std::optional<int64_t> num_groups,
    std::optional<at::Tensor> qparam_k,
    std::optional<at::Tensor> qparam_v);

std::tuple<at::Tensor, at::Tensor> dequantize_fp8_cache(
    at::Tensor cache_K,
    at::Tensor cache_V,
    at::Tensor kv_seqlen,
    std::optional<at::Tensor> qparam_k,
    std::optional<at::Tensor> qparam_v,
    std::optional<at::Tensor> block_tables,
    int64_t page_size);

at::Tensor quantize_qkv_per_head(
    at::Tensor amax,
    at::Tensor XQKV,
    at::Tensor varseq_seqpos,
    std::optional<at::Tensor> varseq_batch,
    at::Tensor q_seqstarts,
    at::Tensor cache_K,
    at::Tensor cache_V,
    at::Tensor XQ_O,
    int64_t max_seq_len,
    std::optional<at::Tensor> qparam_k,
    std::optional<at::Tensor> qparam_v);

at::Tensor mqa_attn(
    at::Tensor XQ,
    at::Tensor cache_K,
    at::Tensor cache_V,
    at::Tensor seq_positions,
    double qk_scale,
    std::optional<int64_t> num_groups,
    int64_t cache_logical_dtype_int,
    std::optional<at::Tensor> qparam_k,
    std::optional<at::Tensor> qparam_v);

void convert_e4m3fn_kv_cache_to_e4m3fnuz_inplace(
    at::Tensor cache_K,
    at::Tensor cache_V,
    at::Tensor qparam_K,
    at::Tensor qparam_v);

} // namespace fbgemm_gpu
