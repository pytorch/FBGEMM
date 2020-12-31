/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

namespace fbgemm_gpu_test {
void permute_sparse_features(
    int weights_size,
    int T,
    int B,
    const int* __restrict__ permute,
    const long* __restrict__ lengths,
    const long* __restrict__ indices,
    const float* __restrict__ weights,
    long* __restrict__ permuted_lengths,
    long* __restrict__ permuted_indices,
    float* __restrict__ permuted_weights);
void bucketize_sparse_features(
    int lengths_size,
    int my_size,
    const long* __restrict__ lengths,
    const long* __restrict__ indices,
    const float* __restrict__ weights,
    long* __restrict__ bucketized_lengths,
    long* __restrict__ bucketized_indices,
    float* __restrict__ bucketized_weights,
    long* __restrict__ bucketized_pos);
} // namespace fbgemm_gpu_test
