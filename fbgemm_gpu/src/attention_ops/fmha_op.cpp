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

#include "fbgemm_gpu/fmha.h"

using at::Tensor;

namespace fbgemm {

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

} // namespace fbgemm

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  DISPATCH_TO_CUDA("fmha_fwd", fbgemm_gpu::fmha_cudnn_forward);
  DISPATCH_TO_CUDA("fmha_bwd", fbgemm_gpu::fmha_cudnn_backward);
}
