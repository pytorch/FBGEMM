/*
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>

namespace fbgemm_gpu {

// Return array of size T_in.numel(), representing incomplete exclusive cumsum
at::Tensor asynchronous_exclusive_cumsum_gpu(const at::Tensor& t_in);

at::Tensor asynchronous_complete_cumsum_gpu(const at::Tensor& t_in);

at::Tensor asynchronous_inclusive_cumsum_gpu(const at::Tensor& t_in);

at::Tensor asynchronous_exclusive_cumsum_cpu(const at::Tensor& t_in);

at::Tensor asynchronous_complete_cumsum_cpu(const at::Tensor& t_in);

at::Tensor asynchronous_inclusive_cumsum_cpu(const at::Tensor& t_in);

at::Tensor offsets_range_cuda(const at::Tensor& offsets, int64_t range_size);

at::Tensor offsets_range_cpu(const at::Tensor& offsets, int64_t range_size);

at::Tensor segment_sum_csr_cuda(
    const int64_t batch_size,
    const at::Tensor& csr_seg,
    const at::Tensor& values);

at::Tensor segment_sum_csr_cpu(
    const int64_t batch_size,
    const at::Tensor& csr_seg,
    const at::Tensor& values);

std::tuple<at::Tensor, at::Tensor, c10::optional<at::Tensor>>
permute_sparse_data_cuda(
    const at::Tensor& permute,
    const at::Tensor& lengths,
    const at::Tensor& indices,
    const c10::optional<at::Tensor>& weights,
    const c10::optional<int64_t>& permuted_lengths_sum);

std::tuple<
    at::Tensor,
    at::Tensor,
    c10::optional<at::Tensor>,
    c10::optional<at::Tensor>,
    c10::optional<at::Tensor>>
block_bucketize_sparse_features_cuda(
    at::Tensor lengths,
    at::Tensor indices,
    bool bucketize_pos,
    bool sequence,
    at::Tensor block_sizes,
    int64_t my_size,
    c10::optional<at::Tensor> weights);

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

std::tuple<at::Tensor, at::Tensor, c10::optional<at::Tensor>>
permute_sparse_data_cpu(
    const at::Tensor& permute,
    const at::Tensor& lengths,
    const at::Tensor& indices,
    const c10::optional<at::Tensor>& weights,
    const c10::optional<int64_t>& permuted_lengths_sum);

at::Tensor _float_to_fused8bitrowwise_gpu(const at::Tensor& input);
at::Tensor _half_to_fused8bitrowwise_gpu(const at::Tensor& input);
at::Tensor _fused8bitrowwise_to_float_gpu(const at::Tensor& input);
at::Tensor _fused8bitrowwise_to_half_gpu(const at::Tensor& input);
at::Tensor float_to_fused8bitrowwise_cpu(const at::Tensor& input);
at::Tensor half_to_fused8bitrowwise_cpu(const at::Tensor& input);
at::Tensor fused8bitrowwise_to_float_cpu(const at::Tensor& input);
at::Tensor fused8bitrowwise_to_half_cpu(const at::Tensor& input);

at::Tensor _fused8bitrowwise_to_float_mixed_dim_gpu(
    const at::Tensor& input,
    const at::Tensor& D_offsets,
    const int64_t output_dtype);
at::Tensor _float_to_fusednbitrowwise_gpu(
    const at::Tensor& input,
    const int64_t bit_rate);
at::Tensor _half_to_fusednbitrowwise_gpu(
    const at::Tensor& input,
    const int64_t bit_rate);
at::Tensor _fusednbitrowwise_to_float_gpu(
    const at::Tensor& input,
    const int64_t bit_rate);
at::Tensor _fusednbitrowwise_to_half_gpu(
    const at::Tensor& input,
    const int64_t bit_rate);
at::Tensor& _fused8bitrowwise_to_float_cpu_out(
    at::Tensor& output,
    const at::Tensor& input);
at::Tensor& _float_to_fused8bitrowwise_cpu_out(
    at::Tensor& output,
    const at::Tensor& input);
at::Tensor float_to_fusednbitrowwise_cpu(
    const at::Tensor& input,
    const int64_t bit_rate);
at::Tensor half_to_fusednbitrowwise_cpu(
    const at::Tensor& input,
    const int64_t bit_rate);
at::Tensor fusednbitrowwise_to_float_cpu(
    const at::Tensor& input,
    const int64_t bit_rate);
at::Tensor fusednbitrowwise_to_half_cpu(
    const at::Tensor& input,
    const int64_t bit_rate);

at::Tensor reorder_batched_ad_lengths_gpu(
    const at::Tensor& cat_ad_lengths,
    const at::Tensor& batch_offsets,
    const int64_t num_ads_in_batch);

at::Tensor reorder_batched_ad_indices_gpu(
    const at::Tensor& cat_ad_offsets,
    const at::Tensor& cat_ad_indices,
    const at::Tensor& reordered_cat_ad_offsets,
    const at::Tensor& batch_offsets,
    const int64_t num_ads_in_batch);

at::Tensor reorder_batched_ad_lengths_cpu(
    const at::Tensor& cat_ad_lengths,
    const at::Tensor& batch_offsets,
    const int64_t num_ads_in_batch);

at::Tensor reorder_batched_ad_indices_cpu(
    const at::Tensor& cat_ad_offsets,
    const at::Tensor& cat_ad_indices,
    const at::Tensor& reordered_cat_ad_offsets,
    const at::Tensor& batch_offsets,
    const int64_t num_ads_in_batch);

at::Tensor recat_embedding_grad_output_cuda(
    at::Tensor grad_output, // [B_local][T_global][D]
    std::vector<int64_t> num_features_per_rank);

at::Tensor recat_embedding_grad_output_mixed_D_cuda(
    const at::Tensor& grad_output, // [B_local][Sum_T_global(D)]
    const std::vector<int64_t>& dim_sum_per_rank);

at::Tensor recat_embedding_grad_output_mixed_D_batch_cuda(
    const at::Tensor& grad_output, // [B_local][Sum_T_global(D)]
    const at::Tensor& dim_sum_per_rank,
    const at::Tensor& cumsum_dim_sum_per_rank);

at::Tensor recat_embedding_grad_output_mixed_D_cpu(
    const at::Tensor& grad_output, // [B_local][Sum_T_global(D)]
    const std::vector<int64_t>& dim_sum_per_rank);

at::Tensor batched_unary_embeddings_forward_cuda(
    const at::Tensor& weight,
    const at::Tensor& table_offsets,
    const at::Tensor& offsets,
    const at::Tensor& indices);

at::Tensor batched_unary_embeddings_backward_cuda(
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    const at::Tensor& table_offsets,
    const at::Tensor& offsets,
    const at::Tensor& indices);

at::Tensor jagged_2d_to_dense_forward_cuda(
    at::Tensor values,
    at::Tensor offsets,
    int32_t max_L);

at::Tensor jagged_2d_to_dense_backward_cuda(
    at::Tensor grad_padded_values,
    at::Tensor offsets,
    int32_t total_L);

std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>>
stacked_jagged_2d_to_dense_forward_cuda(
    at::Tensor values,
    at::Tensor lengths,
    const std::vector<int64_t>& offset_per_key,
    const std::vector<int64_t>& max_lengths_per_key);

at::Tensor stacked_jagged_2d_to_dense_backward_cuda(
    int64_t B,
    int64_t D,
    int64_t total_L,
    const std::vector<at::Tensor>& grad_padded_values_per_key,
    const std::vector<at::Tensor>& offsets_tensor_per_key,
    const std::vector<int64_t>& offset_per_key);

at::Tensor jagged_1d_to_dense_gpu(
    at::Tensor values,
    at::Tensor offsets,
    int64_t max_L,
    int64_t padding_value);

std::vector<at::Tensor> stacked_jagged_1d_to_dense_gpu(
    at::Tensor values,
    at::Tensor lengths,
    const std::vector<int64_t>& offset_per_key,
    const std::vector<int64_t>& max_lengths_per_key,
    int64_t padding_value);

// Divide the prediction range (e.g., [0, 1]) into B bins. In each bin, use
// two parameters to store the number of positive examples and the number of
// examples that fall into this bucket. So we basically have a histogram for
// the model prediction. As a result, for each bin, we have a statistical
// value for the real CTR (num_pos / num_example). We use this statistical
// value as the final calibrated prediction if the pre-cali prediction falls
// into the corresponding bin. In this way, the predictions within each bin
// should be well-calibrated if we have sufficient examples. That is, we have
// a fine-grained calibrated model by this calibration module. Theoretically,
// this calibration layer can fix any uncalibrated model or prediction if we
// have sufficient bins and examples.
//
// Returns [calibrated_prediction, bin_ids].
//
// "logit" is input tensor before applying Sigmoid.
//
// Assumes positive weight calibration is used for calibartion target, and
// "positive_weight" is passed as input argument.
//
// # of bins is automatically derived from "bin_num_examples", and
// "bin_num_positives", all of which should be the same size.
//
// "lower/upper_bound":
// Bounds of the bins.
//
// "bin_ctr_in_use_after":
// We will use the calibration_target for the final calibrated prediction if we
// don't have sufficient examples. Only use the statistical value of bin CTR
// after we observe `bin_ctr_in_use_after` examples that fall in this bin.
// Default: 0.
//
// "bin_ctr_weight_value":
// Weight for statistical value of bin CTR. When this is specified, we perform
// a weighted sum for the statisctical bin CTR and the calibration_target:
// final_calibrated_prediction = bin_ctr_weight * bin_ctr + (1 -
// bin_ctr_weight) * calibration_target.
// Default: 1.0
std::tuple<at::Tensor, at::Tensor> histogram_binning_calibration_cpu(
    const at::Tensor& logit,
    const at::Tensor& bin_num_examples,
    const at::Tensor& bin_num_positives,
    double positive_weight,
    double lower_bound = 0.0,
    double upper_bound = 1.0,
    int64_t bin_ctr_in_use_after = 0,
    double bin_ctr_weight_value = 1.0);

std::tuple<at::Tensor, at::Tensor> histogram_binning_calibration_cuda(
    const at::Tensor& logit,
    const at::Tensor& bin_num_examples,
    const at::Tensor& bin_num_positives,
    double positive_weight,
    double lower_bound = 0.0,
    double upper_bound = 1.0,
    int64_t bin_ctr_in_use_after = 0,
    double bin_ctr_weight_value = 1.0);

// An extension of histogram binning calibration model which divides data into
// bins based on one specific feature and prediction/ECTR range. In each bin,
// use two parameters to store the number of positive examples and the number of
// examples that fall into this bucket. So we basically have a histogram for
// the model prediction. As a result, for each bin, we have a statistical
// value for the real CTR (num_pos / num_example). We use this statistical
// value as the final calibrated prediction if the pre-cali prediction falls
// into the corresponding bin. In this way, the predictions within each bin
// should be well-calibrated if we have sufficient examples. That is, we have
// a fine-grained calibrated model by this calibration module. Theoretically,
// this calibration layer can fix any uncalibrated model or prediction if we
// have sufficient bins and examples.
//
// Returns [calibrated_prediction, bin_ids].
//
// "logit" is input tensor before applying Sigmoid.
//
// Assumes positive weight calibration is used for calibartion target, and
// "positive_weight" is passed as input argument.
//
// "segment_value/lengths":
// Values and lengths in KeyJaggedTensor. Assumes value of length is either 0
// or 1.
//
// "num_bins":
// # of bins is no longer the same as "bin_num_examples", and
// "bin_num_positives", all of which should be still the same size.
//
// "lower/upper_bound":
// Bounds of the bins.
//
// "bin_ctr_in_use_after":
// We will use the calibration_target for the final calibrated prediction if we
// don't have sufficient examples. Only use the statistical value of bin CTR
// after we observe `bin_ctr_in_use_after` examples that fall in this bin.
// Default: 0.
//
// "bin_ctr_weight_value":
// Weight for statistical value of bin CTR. When this is specified, we perform
// a weighted sum for the statisctical bin CTR and the calibration_target:
// final_calibrated_prediction = bin_ctr_weight * bin_ctr + (1 -
// bin_ctr_weight) * calibration_target.
// Default: 1.0
std::tuple<at::Tensor, at::Tensor> histogram_binning_calibration_by_feature_cpu(
    const at::Tensor& logit,
    const at::Tensor& segment_value,
    const at::Tensor& segment_lengths,
    int64_t num_segments,
    const at::Tensor& bin_num_examples,
    const at::Tensor& bin_num_positives,
    int64_t num_bins,
    double positive_weight,
    double lower_bound = 0.0,
    double upper_bound = 1.0,
    int64_t bin_ctr_in_use_after = 0,
    double bin_ctr_weight_value = 1.0);

std::tuple<at::Tensor, at::Tensor>
histogram_binning_calibration_by_feature_cuda(
    const at::Tensor& logit,
    const at::Tensor& segment_value,
    const at::Tensor& segment_lengths,
    int64_t num_segments,
    const at::Tensor& bin_num_examples,
    const at::Tensor& bin_num_positives,
    int64_t num_bins,
    double positive_weight,
    double lower_bound = 0.0,
    double upper_bound = 1.0,
    int64_t bin_ctr_in_use_after = 0,
    double bin_ctr_weight_value = 1.0);

// Same as above, but accepts generic "bin_boundaries", which is assumed to be
// sorted.
//
// Returns calibrated_prediction.
std::tuple<at::Tensor, at::Tensor>
generic_histogram_binning_calibration_by_feature_cpu(
    const at::Tensor& logit,
    const at::Tensor& segment_value,
    const at::Tensor& segment_lengths,
    int64_t num_segments,
    const at::Tensor& bin_num_examples,
    const at::Tensor& bin_num_positives,
    const at::Tensor& bin_boundaries,
    double positive_weight,
    int64_t bin_ctr_in_use_after = 0,
    double bin_ctr_weight_value = 1.0);

std::tuple<at::Tensor, at::Tensor>
generic_histogram_binning_calibration_by_feature_cuda(
    const at::Tensor& logit,
    const at::Tensor& segment_value,
    const at::Tensor& segment_lengths,
    int64_t num_segments,
    const at::Tensor& bin_num_examples,
    const at::Tensor& bin_num_positives,
    const at::Tensor& bin_boundaries,
    double positive_weight,
    int64_t bin_ctr_in_use_after = 0,
    double bin_ctr_weight_value = 1.0);

std::tuple<at::Tensor, at::Tensor> embedding_bag_rowwise_prune(
    const at::Tensor& weights,
    const at::Tensor& indicator,
    const double threshold,
    at::ScalarType compressed_indices_dtype,
    const bool abs,
    const int64_t min_non_pruned_rows,
    const c10::optional<double>& min_save_ratio);

} // namespace fbgemm_gpu
