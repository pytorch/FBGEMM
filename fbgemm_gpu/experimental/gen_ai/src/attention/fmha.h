#pragma once

#include <ATen/core/Tensor.h>
#include <stdint.h>

using at::Tensor;

namespace fbgemm_gpu::gen_ai::attention {

void run_cudnn_sdpa_fprop(
    int64_t b,
    int64_t h,
    int64_t max_seq_len_q,
    int64_t max_seq_len_kv,
    int64_t d,
    float attention_scale,
    double dropout_p,
    bool is_causal,
    bool return_softmax_stats,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& seq_len_q,
    const Tensor& seq_len_kv,
    const Tensor& seq_offset_q,
    const Tensor& seq_offset_kv,
    Tensor& o,
    Tensor& softmax_stats,
    Tensor& dropout_seed,
    Tensor& dropout_offset);

void run_cudnn_sdpa_bprop(
    int64_t b,
    int64_t h,
    int64_t max_seq_len_q,
    int64_t max_seq_len_kv,
    int64_t d,
    float attention_scale,
    double dropout_p,
    bool is_causal,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& seq_len_q,
    const Tensor& seq_len_kv,
    const Tensor& seq_offset_q,
    const Tensor& seq_offset_kv,
    const Tensor& o,
    const Tensor& softmax_stats,
    const Tensor& dO,
    Tensor& dQ,
    Tensor& dK,
    Tensor& dV,
    const Tensor& dropout_seed,
    const Tensor& dropout_offset);

} // namespace fbgemm_gpu::gen_ai::attention
