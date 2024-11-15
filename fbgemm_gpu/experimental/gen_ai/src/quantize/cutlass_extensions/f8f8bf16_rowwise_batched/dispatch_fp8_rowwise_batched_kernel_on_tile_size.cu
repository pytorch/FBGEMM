/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "common.cuh"

namespace fbgemm_gpu {

// Cutlass rowwise batched kernel
template <
    int TB_M,
    int TB_N,
    int TB_K,
    int TBS_M,
    int TBS_N,
    int TBS_K,
    bool PONG,
    bool FAST_ACCUM,
    bool USE_BIAS,
    bool Transposed,
    typename INPUT_DTYPE,
    typename BIAS_DTYPE>
at::Tensor f8f8bf16_rowwise_batched_impl(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    std::optional<at::Tensor> output);

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
    std::optional<at::Tensor> out) {
  int M, N;
  if constexpr (Transposed) {
    M = XQ.size(2);
    N = WQ.size(2);
  } else {
    M = XQ.size(1);
    N = WQ.size(1);
  }

  if ((ceildiv(M, 64 * TBS_M) * ceildiv(N, 128 * TBS_N)) <= kNumSMsForH100 /
          cute::size(cute::Shape<
                     cute::Int<TBS_M>,
                     cute::Int<TBS_N>,
                     cute::Int<TBS_K>>{})) {
    return f8f8bf16_rowwise_batched_impl<
        64,
        128,
        128,
        TBS_M,
        TBS_N,
        TBS_K,
        false,
        FAST_ACCUM,
        USE_BIAS,
        Transposed,
        INPUT_DTYPE,
        BIAS_DTYPE>(XQ, WQ, x_scale, w_scale, bias, out);
  } else {
    return f8f8bf16_rowwise_batched_impl<
        128,
        128,
        128,
        TBS_M,
        TBS_N,
        TBS_K,
        true,
        FAST_ACCUM,
        USE_BIAS,
        Transposed,
        INPUT_DTYPE,
        BIAS_DTYPE>(XQ, WQ, x_scale, w_scale, bias, out);
  }
}

#define INSTANTIATE_FUNC_0(                                             \
    TBS_M,                                                              \
    TBS_N,                                                              \
    TBS_K,                                                              \
    FAST_ACCUM,                                                         \
    USE_BIAS,                                                           \
    Transposed,                                                         \
    INPUT_DTYPE,                                                        \
    BIAS_DTYPE)                                                         \
  template at::Tensor dispatch_fp8_rowwise_batched_kernel_on_tile_size< \
      TBS_M,                                                            \
      TBS_N,                                                            \
      TBS_K,                                                            \
      FAST_ACCUM,                                                       \
      USE_BIAS,                                                         \
      Transposed,                                                       \
      INPUT_DTYPE,                                                      \
      BIAS_DTYPE>(                                                      \
      at::Tensor XQ,                                                    \
      at::Tensor WQ,                                                    \
      at::Tensor x_scale,                                               \
      at::Tensor w_scale,                                               \
      std::optional<at::Tensor> bias,                                   \
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
