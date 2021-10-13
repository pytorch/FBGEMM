/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>

namespace at {
namespace fbgemm {

// Return array of size T_in.numel(), representing incomplete exclusive cumsum
at::Tensor asynchronous_exclusive_cumsum_gpu(const at::Tensor& t_in);

at::Tensor asynchronous_complete_cumsum_gpu(const at::Tensor& t_in);

at::Tensor asynchronous_inclusive_cumsum_gpu(const at::Tensor& t_in);

at::Tensor asynchronous_exclusive_cumsum_cpu(const at::Tensor& t_in);

at::Tensor asynchronous_complete_cumsum_cpu(const at::Tensor& t_in);

at::Tensor asynchronous_inclusive_cumsum_cpu(const at::Tensor& t_in);

at::Tensor offsets_range_cuda(const at::Tensor& offsets, int64_t range_size);

at::Tensor offsets_range_cpu(const at::Tensor& offsets, int64_t range_size);

std::tuple<Tensor, Tensor, c10::optional<Tensor>> permute_sparse_data_cuda(
    const Tensor& permute,
    const Tensor& lengths,
    const Tensor& indices,
    const c10::optional<Tensor>& weights,
    const c10::optional<int64_t>& permuted_lengths_sum);

std::tuple<
    Tensor,
    Tensor,
    c10::optional<Tensor>,
    c10::optional<Tensor>,
    c10::optional<Tensor>>
block_bucketize_sparse_features_cuda(
    Tensor lengths,
    Tensor indices,
    bool bucketize_pos,
    bool sequence,
    Tensor block_sizes,
    int64_t my_size,
    c10::optional<Tensor> weights);

std::tuple<
    at::Tensor,
    at::Tensor,
    c10::optional<at::Tensor>,
    c10::optional<at::Tensor>,
    c10::optional<at::Tensor>>
block_bucketize_sparse_features_cpu(
    at::Tensor lengths,
    at::Tensor indices,
    bool bucketize_pos,
    bool sequence,
    at::Tensor block_sizes,
    int64_t my_size,
    c10::optional<at::Tensor> weights);

std::tuple<Tensor, Tensor, c10::optional<Tensor>> permute_sparse_data_cpu(
    const Tensor& permute,
    const Tensor& lengths,
    const Tensor& indices,
    const c10::optional<Tensor>& weights,
    const c10::optional<int64_t>& permuted_lengths_sum);

at::Tensor _float_to_fused8bitrowwise_gpu(const at::Tensor& input);
at::Tensor _fused8bitrowwise_to_float_gpu(const at::Tensor& input);
at::Tensor _float_to_fusednbitrowwise_gpu(
    const at::Tensor& input,
    const int64_t bit_rate);
at::Tensor _fusednbitrowwise_to_float_gpu(
    const at::Tensor& input,
    const int64_t bit_rate);

Tensor reorder_batched_ad_lengths_gpu(
    const Tensor& cat_ad_lengths,
    const Tensor& batch_offsets,
    const int64_t num_ads_in_batch);

Tensor reorder_batched_ad_indices_gpu(
    const Tensor& cat_ad_offsets,
    const Tensor& cat_ad_indices,
    const Tensor& reordered_cat_ad_offsets,
    const Tensor& batch_offsets,
    const int64_t num_ads_in_batch);

Tensor reorder_batched_ad_lengths_cpu(
    const Tensor& cat_ad_lengths,
    const Tensor& batch_offsets,
    const int64_t num_ads_in_batch);

Tensor reorder_batched_ad_indices_cpu(
    const Tensor& cat_ad_offsets,
    const Tensor& cat_ad_indices,
    const Tensor& reordered_cat_ad_offsets,
    const Tensor& batch_offsets,
    const int64_t num_ads_in_batch);

at::Tensor recat_embedding_grad_output_cuda(
    at::Tensor grad_output, // [B_local][T_global][D]
    std::vector<int64_t> num_features_per_rank);

at::Tensor recat_embedding_grad_output_mixed_D_cuda(
    const Tensor& grad_output, // [B_local][Sum_T_global(D)]
    const std::vector<int64_t>& dim_sum_per_rank);

at::Tensor recat_embedding_grad_output_mixed_D_batch_cuda(
    const Tensor& grad_output, // [B_local][Sum_T_global(D)]
    const Tensor& dim_sum_per_rank,
    const Tensor& cumsum_dim_sum_per_rank);

at::Tensor recat_embedding_grad_output_mixed_D_cpu(
    const Tensor& grad_output, // [B_local][Sum_T_global(D)]
    const std::vector<int64_t>& dim_sum_per_rank);
} // namespace fbgemm
} // namespace at
