/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

namespace fbgemm_gpu_test {
void batched_unary_embeddings_forward(
    const int32_t N,
    const int32_t B,
    const int32_t T,
    const float* __restrict__ weight,
    const long* __restrict__ table_offsets,
    const long* __restrict__ offsets,
    const long* __restrict__ indices,
    float* __restrict__ output);
void batched_unary_embeddings_backward(
    const int32_t N,
    const int32_t B,
    const int32_t T,
    const float* __restrict__ grad_output,
    const long* __restrict__ table_offsets,
    const long* __restrict__ offsets,
    const long* __restrict__ indices,
    float* __restrict__ grad_weight);
} // namespace fbgemm_gpu_test
