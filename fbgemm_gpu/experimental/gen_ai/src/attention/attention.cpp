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

std::tuple<Tensor, Tensor, Tensor, Tensor, c10::SymInt, c10::SymInt, Tensor,
           Tensor, Tensor>
fmha_cudnn_forward(const Tensor &query, const Tensor &key, const Tensor &value,
                   const Tensor &seq_q, const Tensor &seq_kv, double dropout_p,
                   bool is_causal, bool training, std::optional<double> scale) {
  // Used for tracking usage statistics
  C10_LOG_API_USAGE_ONCE("cudnn_fmha.fwd");

  // Query (Batch x Num_heads x Q_seq_len  x Dim_per_head)
  // Key   (Batch x Num_heads x KV_seq_len x Dim_per_head)
  // Value (Batch x Num_heads x KV_seq_len x Dim_per_head)
  const int64_t batch_size = query.size(0);
  const int64_t num_heads = query.size(1);
  const int64_t max_seqlen_batch_q = query.size(2);
  const int64_t head_dim = query.size(3);

  const int64_t max_seqlen_batch_k = key.size(2);
  const int64_t max_seqlen_batch_v = value.size(2);
  TORCH_CHECK(max_seqlen_batch_k == max_seqlen_batch_v,
              "Key and Value must have the same sequence length");

  Tensor attention, log_sumexp;

  auto cudnn_seed = at::zeros({1}, query.options().dtype(at::kLong));
  auto cudnn_offset = at::zeros({1}, query.options().dtype(at::kLong));
  const auto softmax_scale =
      sdp::calculate_scale(query, scale).as_float_unchecked();

  run_cudnn_sdpa_fprop(
      batch_size /*int64_t b*/, num_heads /*int64_t h*/,
      max_seqlen_batch_q /*int64_t s_q*/, max_seqlen_batch_k /*int64_t s_kv*/,
      head_dim /*int64_t d*/, softmax_scale /*float scaling_factor*/,
      training /* bool */, is_causal /* bool */,
      dropout_p /*double dropout_probability*/, query /* Tensor q*/,
      key /* Tensor k*/, value /* Tensor v*/, seq_q /* Tensor seq_q*/,
      seq_kv /* Tensor seq_k*/, log_sumexp /*Tensor softmaxstats*/,
      attention /*Tensor o*/, cudnn_seed /*Tensor dropoutseed*/,
      cudnn_offset /*Tensor dropoutoffset*/);

  return std::make_tuple(attention, log_sumexp, Tensor(), Tensor(),
                         max_seqlen_batch_q, max_seqlen_batch_k, cudnn_seed,
                         cudnn_offset, Tensor());
}

std::tuple<Tensor, Tensor, Tensor> fmha_cudnn_backward(
    const Tensor &grad_out, const Tensor &query, const Tensor &key,
    const Tensor &value, const Tensor &seq_q, const Tensor &seq_kv,
    const Tensor &out, const Tensor &logsumexp,
    const Tensor &cumulative_sequence_length_q,
    const Tensor &cumulative_sequence_length_k,
    const int64_t max_seqlen_batch_q, const int64_t max_seqlen_batch_k,
    double dropout_p, bool is_causal, const Tensor &philox_seed,
    const Tensor &philox_offset, std::optional<double> scale) {
  // Used for tracking usage statistics
  C10_LOG_API_USAGE_ONCE("cudnn_fmha.bwd");

  const int64_t batch_size = query.size(0);
  const int64_t num_heads = query.size(1);
  const int64_t head_dim = query.size(3);

  const auto softmax_scale =
      sdp::calculate_scale(query, scale).as_float_unchecked();

  auto dq = at::empty_like(query);
  auto dk = at::empty_like(key);
  auto dv = at::empty_like(value);
  run_cudnn_sdpa_bprop(
      batch_size /*int64_t b*/, num_heads /*int64_t h*/,
      max_seqlen_batch_q /*int64_t s_q*/, max_seqlen_batch_k /*int64_t s_kv*/,
      head_dim /*int64_t d*/, softmax_scale /*float scaling_factor*/,
      is_causal /*bool is_causal*/, dropout_p /*float dropout_probability*/,
      query /*const Tensor& q*/, key /*const Tensor& k*/,
      value /*const Tensor& v*/, seq_q /* Tensor seq_q*/,
      seq_kv /* Tensor seq_k*/, out /*const Tensor& o*/,
      grad_out /*const Tensor& dO*/,
      logsumexp.unsqueeze(-1) /*const Tensor& softmaxstats*/, dq /*Tensor& dQ*/,
      dk /*Tensor& dK*/, dv /*Tensor& dV*/, philox_seed /*Tensor& dropoutseed*/,
      philox_offset /*Tensor& dropoutoffset*/);
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
        "    float dropout, "
        "    bool is_casual, "
        "    bool training, "
        "    float? scale, "
        ") -> (Tensor, Tensor, Tensor, Tensor, int, int, Tensor, Tensor, "
        "Tensor)");
  m.def("fmha_bwd("
        "    Tensor grad_out, "
        "    Tensor query, "
        "    Tensor key, "
        "    Tensor value, "
        "    Tensor seq_len_q, "
        "    Tensor seq_len_kv, "
        "    Tensor out, "
        "    Tensor logsumexp, "
        "    Tensor seq_len_q, "
        "    Tensor seq_len_kv, "
        "    int max_seq_len_q, "
        "    int max_seq_len_kv, "
        "    float dropout, "
        "    bool is_casual, "
        "    Tensor seed, "
        "    Tensor seed_offset, "
        "    float? scale, "
        ") -> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(fbgemm, CUDA, m) {
  m.impl("fmha_fwd", fbgemm_gpu::gen_ai::attention::fmha_cudnn_forward);
  m.impl("fmha_bwd", fbgemm_gpu::gen_ai::attention::fmha_cudnn_backward);
}
#endif
