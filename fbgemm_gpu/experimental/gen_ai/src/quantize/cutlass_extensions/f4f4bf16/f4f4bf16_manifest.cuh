/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>

namespace fbgemm_gpu {

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12080)

at::Tensor f4f4bf16_128_128_4_1_1_f(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> global_scale);

at::Tensor f4f4bf16_128_128_4_1_1_t(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> global_scale);

at::Tensor f4f4bf16_128_192_2_2_1_f(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> global_scale);

at::Tensor f4f4bf16_128_192_2_2_1_t(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> global_scale);

at::Tensor f4f4bf16_128_256_2_1_1_f(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> global_scale);

at::Tensor f4f4bf16_128_256_2_1_1_t(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> global_scale);

at::Tensor f4f4bf16_256_128_2_2_1_f(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> global_scale);

at::Tensor f4f4bf16_256_128_2_2_1_t(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> global_scale);

at::Tensor f4f4bf16_256_128_2_4_1_f(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> global_scale);

at::Tensor f4f4bf16_256_128_2_4_1_t(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> global_scale);

at::Tensor f4f4bf16_256_192_2_2_1_f(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> global_scale);

at::Tensor f4f4bf16_256_192_2_2_1_t(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> global_scale);

at::Tensor f4f4bf16_256_192_2_4_1_f(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> global_scale);

at::Tensor f4f4bf16_256_192_2_4_1_t(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> global_scale);

at::Tensor f4f4bf16_256_192_4_1_1_f(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> global_scale);

at::Tensor f4f4bf16_256_192_4_1_1_t(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> global_scale);

at::Tensor f4f4bf16_256_256_2_1_1_f(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> global_scale);

at::Tensor f4f4bf16_256_256_2_1_1_t(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> global_scale);

at::Tensor f4f4bf16_256_256_2_2_1_f(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> global_scale);

at::Tensor f4f4bf16_256_256_2_2_1_t(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> global_scale);

at::Tensor f4f4bf16_256_256_2_4_1_f(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> global_scale);

at::Tensor f4f4bf16_256_256_2_4_1_t(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> global_scale);

at::Tensor f4f4bf16_256_256_4_1_1_f(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> global_scale);

at::Tensor f4f4bf16_256_256_4_1_1_t(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> global_scale);

#endif
} // namespace fbgemm_gpu
