/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <tuple>
#include <type_traits>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/NestedTensorImpl.h>
#include <ATen/TensorAccessor.h>
#include <ATen/TensorOperators.h>
#include <ATen/core/Reduction.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/NonSymbolicBC.h>
#include <ATen/native/transformers/sdp_utils_cpp.h>
#include <c10/core/QScheme.h>
#include <c10/core/Scalar.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Deprecated.h>
#include <c10/util/Logging.h>
#include <c10/util/Optional.h>
#include <c10/util/bit_cast.h>
#include <torch/library.h>

#include "fmha.h"

using at::Tensor;

namespace fbgemm_gpu::gen_ai::attention {

std::tuple<at::Tensor, at::Tensor, at::Tensor> gqa_attn_splitk(
    const at::Tensor &XQ, const at::Tensor &cache_K, const at::Tensor &cache_V,
    const at::Tensor &seq_positions, const double qk_scale,
    const int64_t num_split_ks, const int64_t kv_cache_quant_num_groups,
    const bool use_tensor_cores, const int64_t cache_logical_dtype_int);

std::tuple<Tensor, Tensor, Tensor, Tensor> fmha_cudnn_forward(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& seq_len_q,
    const Tensor& seq_len_kv,
    const Tensor& seq_offset_q,
    const Tensor& seq_offset_kv,
    const int64_t max_seq_len_q,
    const int64_t max_seq_len_kv,
    double attention_scale,
    double dropout_p,
    bool is_causal,
    bool return_softmax_stats) {
  // Used for tracking usage statistics
  C10_LOG_API_USAGE_ONCE("fbgemm.fmha_fwd");

  // QKV: THD
  TORCH_CHECK(query.dim() == 3 && key.dim() == 3 && value.dim() == 3);

  const int64_t total_seq_len_q = query.size(0);
  const int64_t total_seq_len_kv = key.size(0);
  TORCH_CHECK(total_seq_len_kv == value.size(0));

  const int64_t num_heads = query.size(1);
  TORCH_CHECK(num_heads == value.size(1) && num_heads == key.size(1));

  const int64_t head_dim = query.size(2);
  TORCH_CHECK(head_dim == value.size(2) && head_dim == key.size(2));

  TORCH_CHECK(seq_len_q.dim() == 1 && seq_len_kv.dim() == 1);
  const int64_t batch_size = seq_len_q.size(0);
  TORCH_CHECK(batch_size == seq_len_kv.size(0));

  TORCH_CHECK(seq_offset_q.dim() == 1 && seq_offset_kv.dim() == 1);
  TORCH_CHECK(
      batch_size + 1 == seq_offset_q.size(0) &&
      seq_offset_q.size(0) == seq_offset_kv.size(0));

  Tensor attention, softmax_stats;
  attention = at::empty_like(query);

  if (return_softmax_stats) {
    // TODO(shikaili): verify that this is correct
    softmax_stats = at::empty(
        {batch_size, head_dim, max_seq_len_q},
        query.options().dtype(at::kFloat));
  }
  auto cudnn_seed = at::zeros({1}, query.options().dtype(at::kLong));
  auto cudnn_offset = at::zeros({1}, query.options().dtype(at::kLong));

  run_cudnn_sdpa_fprop(
      batch_size /*int64_t b*/,
      num_heads /*int64_t h*/,
      max_seq_len_q /*int64_t max_seq_len_q*/,
      max_seq_len_kv /*int64_t max_seq_len_kv*/,
      head_dim /*int64_t d*/,
      attention_scale /*float attention_scale*/,
      dropout_p /*double dropout_p*/,
      is_causal /* bool is_causal*/,
      return_softmax_stats /* bool return_softmax_stats*/,
      query /* Tensor q*/,
      key /* Tensor k*/,
      value /* Tensor v*/,
      seq_len_q /* Tensor seq_len_q*/,
      seq_len_kv /* Tensor seq_len_kv*/,
      seq_offset_q /* Tensor seq_offset_q*/,
      seq_offset_kv /* Tensor seq_offset_k*/,
      attention /*Tensor o*/,
      softmax_stats /*Tensor softmax_stats*/,
      cudnn_seed /*Tensor dropout_seed*/,
      cudnn_offset /*Tensor dropout_offset*/);

  return std::make_tuple(attention, softmax_stats, cudnn_seed, cudnn_offset);
}

std::tuple<Tensor, Tensor, Tensor> fmha_cudnn_backward(
    const Tensor& grad_out,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& seq_len_q,
    const Tensor& seq_len_kv,
    const Tensor& seq_offset_q,
    const Tensor& seq_offset_kv,
    const Tensor& out,
    const Tensor& softmax_stats,
    const int64_t max_seq_len_q,
    const int64_t max_seq_len_kv,
    double attention_scale,
    double dropout_p,
    bool is_causal,
    const Tensor& philox_seed,
    const Tensor& philox_offset) {
  // Used for tracking usage statistics
  C10_LOG_API_USAGE_ONCE("fbgemm.fmha_bwd");

  TORCH_CHECK(query.dim() == 3 && key.dim() == 3 && value.dim() == 3);

  const int64_t total_seq_len_q = query.size(0);
  const int64_t total_seq_len_kv = key.size(0);
  TORCH_CHECK(total_seq_len_kv == value.size(0));

  const int64_t num_heads = query.size(1);
  TORCH_CHECK(num_heads == value.size(1) && num_heads == key.size(1));

  const int64_t head_dim = query.size(2);
  TORCH_CHECK(head_dim == value.size(2) && head_dim == key.size(2));

  TORCH_CHECK(seq_len_q.dim() == 1 && seq_len_kv.dim() == 1);
  const int64_t batch_size = seq_len_q.size(0);
  TORCH_CHECK(batch_size == seq_len_kv.size(0));

  TORCH_CHECK(seq_offset_q.dim() == 1 && seq_offset_kv.dim() == 1);
  TORCH_CHECK(
      batch_size + 1 == seq_offset_q.size(0) &&
      seq_offset_q.size(0) == seq_offset_kv.size(0));

  auto dq = at::empty_like(query);
  auto dk = at::empty_like(key);
  auto dv = at::empty_like(value);
  run_cudnn_sdpa_bprop(
      batch_size /*int64_t b*/,
      num_heads /*int64_t h*/,
      max_seq_len_q /*int64_t max_seq_len_q*/,
      max_seq_len_kv /*int64_t max_seq_len_kv*/,
      head_dim /*int64_t d*/,
      attention_scale /*float attention_scale*/,
      dropout_p /*double attention_dropout*/,
      is_causal /* bool is_causal*/,
      query /*const Tensor& q*/,
      key /*const Tensor& k*/,
      value /*const Tensor& v*/,
      seq_len_q /* Tensor seq_len_q*/,
      seq_len_kv /* Tensor seq_len_kv*/,
      seq_offset_q /* Tensor seq_offset_q*/,
      seq_offset_kv /* Tensor seq_offset_k*/,
      out /*const Tensor& o*/,
      softmax_stats.unsqueeze(-1) /*const Tensor& softmax_stats*/,
      grad_out /*const Tensor& dO*/,
      dq /*Tensor& dQ*/,
      dk /*Tensor& dK*/,
      dv /*Tensor& dV*/,
      philox_seed /*Tensor& dropout_seed*/,
      philox_offset /*Tensor& dropout_offset*/);
  return std::make_tuple(dq, dk, dv);
}

} // namespace fbgemm_gpu::gen_ai::attention


TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.def("gqa_attn_splitk("
        "    Tensor XQ, "
        "    Tensor cache_K, "
        "    Tensor cache_V, "
        "    Tensor seq_positions, "
        "    float qk_scale, "
        "    int num_split_ks, "
        "    int kv_cache_quant_num_groups=1, "
        "    bool use_tensor_cores=True,"
        "    int cache_logical_dtype_int=0"
        ") -> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(fbgemm, CUDA, m) {
  m.impl("gqa_attn_splitk",
         torch::dispatch(
             c10::DispatchKey::CUDA,
             TORCH_FN(fbgemm_gpu::gen_ai::attention::gqa_attn_splitk)));
}

#ifndef USE_ROCM
TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.def("fmha_fwd("
        "    Tensor query, "
        "    Tensor key, "
        "    Tensor value, "
        "    Tensor seq_len_q, "
        "    Tensor seq_len_kv, "
        "    Tensor seq_offset_q, "
        "    Tensor seq_offset_kv, "
        "    int max_seq_len_q, "
        "    int max_seq_len_kv, "
        "    float attention_scale, "
        "    float dropout_p, "
        "    bool is_casual, "
        "    bool return_softmax_stats"
        ") -> (Tensor, Tensor, Tensor, Tensor)");
  m.def("fmha_bwd("
        "    Tensor grad_out, "
        "    Tensor query, "
        "    Tensor key, "
        "    Tensor value, "
        "    Tensor seq_len_q, "
        "    Tensor seq_len_kv, "
        "    Tensor seq_offset_q, "
        "    Tensor seq_offset_kv, "
        "    Tensor out, "
        "    Tensor softmax_stats, "
        "    int max_seq_len_q, "
        "    int max_seq_len_kv, "
        "    float attention_scale, "
        "    float dropout_p, "
        "    bool is_casual, "
        "    Tensor dropout_seed, "
        "    Tensor dropout_offset"
        ") -> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(fbgemm, CUDA, m) {
  m.impl("fmha_fwd", fbgemm_gpu::gen_ai::attention::fmha_cudnn_forward);
  m.impl("fmha_bwd", fbgemm_gpu::gen_ai::attention::fmha_cudnn_backward);
}
#endif
