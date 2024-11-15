/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "common.cuh"

namespace fbgemm_gpu {

template <
    int TBS_M,
    int TBS_N,
    int TBS_K,
    bool FAST_ACCUM,
    bool USE_BIAS,
    bool Transposed,
    typename INPUT_DTYPE,
    typename BIAS_DTYPE>
at::Tensor dispatch_fp8_rowwise_batched_kernel_on_tile_size(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    std::optional<at::Tensor> out);

template <
    int TBS_M,
    int TBS_N,
    int TBS_K,
    bool FAST_ACCUM,
    bool USE_BIAS,
    bool Transposed,
    typename INPUT_DTYPE,
    typename BIAS_DTYPE>
at::Tensor handle_transposition(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    std::optional<at::Tensor> out) {
  if constexpr (!Transposed) {
    return dispatch_fp8_rowwise_batched_kernel_on_tile_size<
        TBS_M,
        TBS_N,
        TBS_K,
        FAST_ACCUM,
        USE_BIAS,
        Transposed,
        INPUT_DTYPE,
        BIAS_DTYPE>(XQ, WQ, x_scale, w_scale, bias, out);
  } else {
    at::Tensor out_;
    if (out.has_value()) {
      out_ = dispatch_fp8_rowwise_batched_kernel_on_tile_size<
          TBS_M,
          TBS_N,
          TBS_K,
          FAST_ACCUM,
          USE_BIAS,
          Transposed,
          INPUT_DTYPE,
          BIAS_DTYPE>(
          WQ.transpose(1, 2),
          XQ.transpose(1, 2),
          w_scale,
          x_scale,
          bias,
          out.value().transpose(1, 2));
    } else {
      out_ = dispatch_fp8_rowwise_batched_kernel_on_tile_size<
          TBS_M,
          TBS_N,
          TBS_K,
          FAST_ACCUM,
          USE_BIAS,
          Transposed,
          INPUT_DTYPE,
          BIAS_DTYPE>(
          WQ.transpose(1, 2), XQ.transpose(1, 2), w_scale, x_scale, bias, out);
    }
    return out_.transpose(1, 2).contiguous();
  }
}

#define INSTANTIATE_FUNC_0(                 \
    TBS_M,                                  \
    TBS_N,                                  \
    TBS_K,                                  \
    FAST_ACCUM,                             \
    USE_BIAS,                               \
    Transposed,                             \
    INPUT_DTYPE,                            \
    BIAS_DTYPE)                             \
  template at::Tensor handle_transposition< \
      TBS_M,                                \
      TBS_N,                                \
      TBS_K,                                \
      FAST_ACCUM,                           \
      USE_BIAS,                             \
      Transposed,                           \
      INPUT_DTYPE,                          \
      BIAS_DTYPE>(                          \
      at::Tensor XQ,                        \
      at::Tensor WQ,                        \
      at::Tensor x_scale,                   \
      at::Tensor w_scale,                   \
      std::optional<at::Tensor> bias,       \
      std::optional<at::Tensor> output);

#define INSTANTIATE_FUNC_1(InputDType, FastAccum, UseBias, BiasDType) \
  INSTANTIATE_FUNC_0(                                                 \
      1, 2, 1, FastAccum, UseBias, false, InputDType, BiasDType);     \
  INSTANTIATE_FUNC_0(2, 1, 1, FastAccum, UseBias, false, InputDType, BiasDType);

#if CUDART_VERSION >= 12000

// Create instantiations for the cartesian product of input dtypes, bias dtypes,
// fast-accum options, and use-bias options
FOR_FLOAT_TYPES(INSTANTIATE_FUNC_1);

#endif

#undef INSTANTIATE_FUNC_1
#undef INSTANTIATE_FUNC_0

} // namespace fbgemm_gpu
