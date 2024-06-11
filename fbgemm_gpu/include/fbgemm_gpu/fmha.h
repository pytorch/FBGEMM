#pragma once

#include <stdint.h>
#include <ATen/core/Tensor.h>

using at::Tensor;

namespace fbgemm {

void run_cudnn_sdpa_fprop(int64_t b, int64_t h, int64_t s_q, int64_t s_kv,
                          int64_t d, float scaling_factor, bool isTraining,
                          bool is_causal, double dropout_probability,
                          const Tensor &q, const Tensor &k, const Tensor &v,
                          const Tensor &seq_q, const Tensor &seq_kv,
                          Tensor &softmaxstats, Tensor &o, Tensor &dropoutseed,
                          Tensor &dropoutoffset);

void run_cudnn_sdpa_bprop(int64_t b, int64_t h, int64_t s_q, int64_t s_kv,
                          int64_t d, float scaling_factor, bool is_causal,
                          float dropout_probability, const Tensor &q,
                          const Tensor &k, const Tensor &v, const Tensor &seq_q,
                          const Tensor &seq_kv, const Tensor &o,
                          const Tensor &dO, const Tensor &softmaxstats,
                          Tensor &dQ, Tensor &dK, Tensor &dV,
                          const Tensor &dropoutseed,
                          const Tensor &dropoutoffset);

} // namespace fbgemm
