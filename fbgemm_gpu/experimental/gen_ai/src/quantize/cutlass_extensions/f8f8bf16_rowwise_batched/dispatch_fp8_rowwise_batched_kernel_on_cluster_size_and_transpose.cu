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
at::Tensor handle_transposition(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    std::optional<at::Tensor> out);

// FP8 Rowwise batched Cutlass kernel dispatch.
template <typename InputDType, bool FastAccum, bool UseBias, typename BiasDType>
at::Tensor dispatch_fp8_rowwise_batched_kernel_on_cluster_size_and_transpose(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    std::optional<at::Tensor> output) {
  TORCH_CHECK(
      (XQ.dim() == 3 && WQ.dim() == 3),
      "FP8 rowwise batched GEMM only supports 3D inputs");
  int M, N;
  M = XQ.size(1);
  N = WQ.size(1);
  // All the tiles we use have sizes which are multiples of 64, hence any
  // non-multiple of 64 will get padded anyways. Let's round up to simplify.
  M = round_up_to_nearest_multiple(M, 64);
  N = round_up_to_nearest_multiple(N, 64);

  // Small/skinny shapes with odd multiples of 64.
  if (M == 64 && N >= 3072) {
    return handle_transposition<
        1,
        2,
        1,
        FastAccum,
        UseBias,
        false,
        InputDType,
        BiasDType>(XQ, WQ, x_scale, w_scale, bias, output);
  }

  if (N == 64 && M >= 3072) {
    return handle_transposition<
        1,
        2,
        1,
        FastAccum,
        UseBias,
        false,
        InputDType,
        BiasDType>(XQ, WQ, x_scale, w_scale, bias, output);
  }

  if (M == 192 && N >= 4096) {
    return handle_transposition<
        1,
        2,
        1,
        FastAccum,
        UseBias,
        false,
        InputDType,
        BiasDType>(XQ, WQ, x_scale, w_scale, bias, output);
  }

  if (N == 192 && M >= 4096) {
    return handle_transposition<
        1,
        2,
        1,
        FastAccum,
        UseBias,
        false,
        InputDType,
        BiasDType>(XQ, WQ, x_scale, w_scale, bias, output);
  }

  // Now to odd multiples of 128 (but only if not too large).
  if (M * N <= 4096 * 4096) {
    if (M % 256 > 0 && N % 256 == 0) {
      return handle_transposition<
          2,
          1,
          1,
          FastAccum,
          UseBias,
          false,
          InputDType,
          BiasDType>(XQ, WQ, x_scale, w_scale, bias, output);
    }
    if (N % 256 > 0 && M % 256 == 0) {
      return handle_transposition<
          2,
          1,
          1,
          FastAccum,
          UseBias,
          false,
          InputDType,
          BiasDType>(XQ, WQ, x_scale, w_scale, bias, output);
    }
  }
  if (M % 256 > 0 && N % 256 > 0) {
    if ((M <= N) ^ (M * N <= 1024 * 1024)) {
      return handle_transposition<
          2,
          1,
          1,
          FastAccum,
          UseBias,
          false,
          InputDType,
          BiasDType>(XQ, WQ, x_scale, w_scale, bias, output);
    } else {
      return handle_transposition<
          2,
          1,
          1,
          FastAccum,
          UseBias,
          false,
          InputDType,
          BiasDType>(XQ, WQ, x_scale, w_scale, bias, output);
    }
  }

  // General case for large tensors.
  if (M >= 1024 && N >= 1024) {
    return handle_transposition<
        2,
        1,
        1,
        FastAccum,
        UseBias,
        false,
        InputDType,
        BiasDType>(XQ, WQ, x_scale, w_scale, bias, output);
  } else {
    return handle_transposition<
        1,
        2,
        1,
        FastAccum,
        UseBias,
        false,
        InputDType,
        BiasDType>(XQ, WQ, x_scale, w_scale, bias, output);
  }
}

#define INSTANTIATE_FUNC_0(FAST_ACCUM, USE_BIAS, INPUT_DTYPE, BIAS_DTYPE) \
  template at::Tensor                                                     \
  dispatch_fp8_rowwise_batched_kernel_on_cluster_size_and_transpose<      \
      FAST_ACCUM,                                                         \
      USE_BIAS,                                                           \
      INPUT_DTYPE,                                                        \
      BIAS_DTYPE>(                                                        \
      at::Tensor XQ,                                                      \
      at::Tensor WQ,                                                      \
      at::Tensor x_scale,                                                 \
      at::Tensor w_scale,                                                 \
      std::optional<at::Tensor> bias,                                     \
      std::optional<at::Tensor> output);

#if CUDART_VERSION >= 12000

// Create instantiations for the cartesian product of input dtypes, bias dtypes,
// fast-accum options, and use-bias options
FOR_FLOAT_TYPES(INSTANTIATE_FUNC_0);

#endif

#undef INSTANTIATE_FUNC_0

} // namespace fbgemm_gpu
