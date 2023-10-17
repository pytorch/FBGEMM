/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>
#include <cstdint>

namespace fbgemm_gpu {

/// @defgroup sparse-data-cuda Sparse Data CUDA Operators
/// The following are CUDA operators
///

/// @defgroup sparse-data-cpu Sparse Data CPU Operators
/// The following are CPU Operators
///

// Return array of size T_in.numel(), representing incomplete exclusive cumsum

#ifndef DOXYGEN_THIS_WILL_BE_SKIPPED
///@ingroup sparse-data-cuda
at::Tensor asynchronous_exclusive_cumsum_gpu(const at::Tensor& t_in);

///@ingroup sparse-data-cuda
at::Tensor asynchronous_complete_cumsum_gpu(const at::Tensor& t_in);

///@ingroup sparse-data-cuda
at::Tensor asynchronous_inclusive_cumsum_gpu(const at::Tensor& t_in);

///@ingroup sparse-data-cpu
at::Tensor asynchronous_exclusive_cumsum_cpu(const at::Tensor& t_in);

///@ingroup sparse-data-cpu
at::Tensor asynchronous_complete_cumsum_cpu(const at::Tensor& t_in);

///@ingroup sparse-data-cpu
at::Tensor asynchronous_inclusive_cumsum_cpu(const at::Tensor& t_in);

///@ingroup sparse-data-cuda
at::Tensor asynchronous_complete_cumsum_meta(const at::Tensor& t_in);

///@ingroup sparse-data-cuda
at::Tensor asynchronous_exclusive_cumsum_meta(const at::Tensor& t_in);

///@ingroup sparse-data-cuda
at::Tensor offsets_range_cuda(const at::Tensor& offsets, int64_t range_size);

///@ingroup sparse-data-cpu
at::Tensor offsets_range_cpu(const at::Tensor& offsets, int64_t range_size);

///@ingroup sparse-data-cuda
at::Tensor segment_sum_csr_cuda(
    const int64_t batch_size,
    const at::Tensor& csr_seg,
    const at::Tensor& values);

///@ingroup sparse-data-cpu
at::Tensor segment_sum_csr_cpu(
    const int64_t batch_size,
    const at::Tensor& csr_seg,
    const at::Tensor& values);
#endif

#ifndef DOXYGEN_THIS_WILL_BE_SKIPPED
/// Description of my method <br>
///**Example:**
///```
/// Here comes
/// my code block
///```
///@param param1 this is my test param #1
///@param param2 this is my test param #2
///@return This function returns abc
///@note This is my test note
///@warning I'm warning you! =)
///@throw fbgemm_gpu::my_error if something something
///@see You can find more info <a
/// href="https://www.doxygen.nl/manual/commands.html#cmdlink">here</a>

std::tuple<at::Tensor, at::Tensor, c10::optional<at::Tensor>>
permute_2D_sparse_data_cuda(
    const at::Tensor& permute,
    const at::Tensor& lengths,
    const at::Tensor& indices,
    const c10::optional<at::Tensor>& weights,
    const c10::optional<int64_t>& permuted_lengths_sum);

std::tuple<at::Tensor, at::Tensor, c10::optional<at::Tensor>>
permute_1D_sparse_data_cuda(
    const at::Tensor& permute,
    const at::Tensor& lengths,
    const at::Tensor& indices,
    const c10::optional<at::Tensor>& weights,
    const c10::optional<int64_t>& permuted_lengths_sum);

at::Tensor invert_permute_cuda(const at::Tensor& permute);
#endif

/// @ingroup sparse-data-cuda
/// expand_into_jagged_permute expand the sparse data permute index from
/// table dimension to batch dimension, for cases where the sparse features
/// has different batch sizes across ranks.
///
/// @param permute the table level permute index.
/// @param input_offsets the exclusive offsets of table-level length.
/// @param output_offsets the exclusive offsets of table-level permuted length.
/// The op expands the permute from table level to batch level by
/// contiguously mapping each bag of its corresponding tables to the position
/// the batch sits on after feature permute. We will derive offset array of
/// table and batch to compute the output permute.
/// @return The output follows the following formula:
/// ```
/// output_permute[table_offset[permute[table]] + batch] <- bag_offset[batch]
/// ```
at::Tensor expand_into_jagged_permute_cuda(
    const at::Tensor& permute,
    const at::Tensor& input_offsets,
    const at::Tensor& output_offsets,
    int64_t output_size);

#ifndef DOXYGEN_THIS_WILL_BE_SKIPPED
///@ingroup sparse-data-cpu
at::Tensor expand_into_jagged_permute_cpu(
    const at::Tensor& permute,
    const at::Tensor& input_offsets,
    const at::Tensor& output_offsets,
    int64_t output_size);

std::tuple<
    at::Tensor,
    at::Tensor,
    c10::optional<at::Tensor>,
    c10::optional<at::Tensor>,
    c10::optional<at::Tensor>>

///@ingroup sparse-data-cuda
block_bucketize_sparse_features_cuda(
    const at::Tensor& lengths,
    const at::Tensor& indices,
    const bool bucketize_pos,
    const bool sequence,
    const at::Tensor& block_sizes,
    const int64_t my_size,
    const c10::optional<at::Tensor>& weights,
    const c10::optional<at::Tensor>& batch_size_per_feature,
    const int64_t max_batch_size);

std::tuple<
    at::Tensor,
    at::Tensor,
    c10::optional<at::Tensor>,
    c10::optional<at::Tensor>,
    c10::optional<at::Tensor>>

///@ingroup sparse-data-cpu
block_bucketize_sparse_features_cpu(
    const at::Tensor& lengths,
    const at::Tensor& indices,
    const bool bucketize_pos,
    const bool sequence,
    const at::Tensor& block_sizes,
    const int64_t my_size,
    const c10::optional<at::Tensor>& weights,
    const c10::optional<at::Tensor>& batch_size_per_feature,
    const int64_t max_batch_size);

std::tuple<
    at::Tensor,
    at::Tensor,
    c10::optional<at::Tensor>,
    c10::optional<at::Tensor>>

///@ingroup sparse-data-cuda
bucketize_sparse_features_cuda(
    const at::Tensor& lengths,
    const at::Tensor& indices,
    const bool bucketize_pos,
    const int64_t my_size,
    const c10::optional<at::Tensor>& weights);

std::tuple<
    at::Tensor,
    at::Tensor,
    c10::optional<at::Tensor>,
    c10::optional<at::Tensor>>
///@ingroup sparse-data-cpu
bucketize_sparse_features_cpu(
    const at::Tensor& lengths,
    const at::Tensor& indices,
    const bool bucketize_pos,
    const int64_t my_size,
    const c10::optional<at::Tensor>& weights);

///@ingroup sparse-data-cpu
std::tuple<at::Tensor, at::Tensor, c10::optional<at::Tensor>>
permute_2D_sparse_data_cpu(
    const at::Tensor& permute,
    const at::Tensor& lengths,
    const at::Tensor& indices,
    const c10::optional<at::Tensor>& weights,
    const c10::optional<int64_t>& permuted_lengths_sum);

///@ingroup sparse-data-cpu
std::tuple<at::Tensor, at::Tensor, c10::optional<at::Tensor>>
permute_1D_sparse_data_cpu(
    const at::Tensor& permute,
    const at::Tensor& lengths,
    const at::Tensor& indices,
    const c10::optional<at::Tensor>& weights,
    const c10::optional<int64_t>& permuted_lengths_sum);

at::Tensor _float_to_fused8bitrowwise_gpu(const at::Tensor& input);
at::Tensor _float_to_paddedFP8rowwise_gpu(
    const at::Tensor& input,
    const bool forward = true,
    const int64_t row_dim = 256);
at::Tensor _float_to_FP8rowwise_gpu(
    const at::Tensor& input,
    const bool forward = true);
at::Tensor _half_to_fused8bitrowwise_gpu(const at::Tensor& input);
at::Tensor _float_or_half_to_fused8bitrowwise_gpu(const at::Tensor& input);
at::Tensor _fused8bitrowwise_to_float_gpu(const at::Tensor& input);
at::Tensor _FP8rowwise_to_float_gpu(
    const at::Tensor& input,
    const bool forward = true,
    const int64_t output_dtype = 0);
at::Tensor _paddedFP8rowwise_to_float_gpu(
    const at::Tensor& input,
    const bool forward = true,
    const int64_t row_dim = 256,
    const int64_t output_last_dim = -1,
    const int64_t output_dtype = 0);
at::Tensor _fused8bitrowwise_to_half_gpu(const at::Tensor& input);
at::Tensor _fused8bitrowwise_to_float_or_half_gpu(
    const at::Tensor& input,
    const int64_t output_dtype);
at::Tensor float_to_fused8bitrowwise_cpu(const at::Tensor& input);
at::Tensor float_to_FP8rowwise_cpu(
    const at::Tensor& input,
    const bool forward = true);
at::Tensor half_to_fused8bitrowwise_cpu(const at::Tensor& input);
at::Tensor float_or_half_to_fused8bitrowwise_cpu(const at::Tensor& input);
at::Tensor fused8bitrowwise_to_float_cpu(const at::Tensor& input);
at::Tensor FP8rowwise_to_float_cpu(
    const at::Tensor& input,
    const bool forward = true,
    const int64_t output_dtype = 0);
at::Tensor fused8bitrowwise_to_half_cpu(const at::Tensor& input);
at::Tensor fused8bitrowwise_to_float_or_half_cpu(
    const at::Tensor& input,
    const int64_t output_dtype);
at::Tensor _float_to_bfloat16_gpu(const at::Tensor&);
at::Tensor _bfloat16_to_float_gpu(const at::Tensor&);
at::Tensor _float_to_bfloat16_cpu(const at::Tensor&);
at::Tensor _bfloat16_to_float_cpu(const at::Tensor&);

at::Tensor _float_to_hfp8_gpu(
    const at::Tensor& input,
    const int64_t ebits,
    const int64_t exponent_bias,
    const double max_pos);
at::Tensor _hfp8_to_float_gpu(
    const at::Tensor& input,
    const int64_t ebits,
    const int64_t exponent_bias);
at::Tensor _float_to_msfp_gpu(
    const at::Tensor& input,
    const int64_t bounding_box_size,
    const int64_t ebits,
    const int64_t mbits,
    const int64_t bias,
    const double min_pos,
    const double max_pos);
at::Tensor _msfp_to_float_gpu(
    const at::Tensor& input,
    const int64_t ebits,
    const int64_t mbits,
    const int64_t bias);
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
at::Tensor _float_or_half_to_fusednbitrowwise_gpu(
    const at::Tensor& input,
    const int64_t bit_rate);
at::Tensor _fusednbitrowwise_to_float_gpu(
    const at::Tensor& input,
    const int64_t bit_rate);
at::Tensor _fusednbitrowwise_to_half_gpu(
    const at::Tensor& input,
    const int64_t bit_rate);
at::Tensor _fusednbitrowwise_to_float_or_half_gpu(
    const at::Tensor& input,
    const int64_t bit_rate,
    const int64_t output_dtype);
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
at::Tensor float_or_half_to_fusednbitrowwise_cpu(
    const at::Tensor& input,
    const int64_t bit_rate);
at::Tensor fusednbitrowwise_to_float_cpu(
    const at::Tensor& input,
    const int64_t bit_rate);
at::Tensor fusednbitrowwise_to_half_cpu(
    const at::Tensor& input,
    const int64_t bit_rate);
at::Tensor fusednbitrowwise_to_float_or_half_cpu(
    const at::Tensor& input,
    const int64_t bit_rate,
    const int64_t output_dtype);

///@ingroup sparse-data-cuda
at::Tensor reorder_batched_ad_lengths_gpu(
    const at::Tensor& cat_ad_lengths,
    const at::Tensor& batch_offsets,
    const int64_t num_ads_in_batch,
    const bool broadcast_lengths = false);

///@ingroup sparse-data-cuda
at::Tensor reorder_batched_ad_indices_gpu(
    const at::Tensor& cat_ad_offsets,
    const at::Tensor& cat_ad_indices,
    const at::Tensor& reordered_cat_ad_offsets,
    const at::Tensor& batch_offsets,
    const int64_t num_ads_in_batch,
    const bool broadcast_indices = false,
    const int64_t num_indices_after_broadcast = -1);

///@ingroup sparse-data-cpu
at::Tensor reorder_batched_ad_lengths_cpu(
    const at::Tensor& cat_ad_lengths,
    const at::Tensor& batch_offsets,
    const int64_t num_ads_in_batch,
    const bool broadcast_lengths = false);
///@ingroup sparse-data-cpu
at::Tensor reorder_batched_ad_indices_cpu(
    const at::Tensor& cat_ad_offsets,
    const at::Tensor& cat_ad_indices,
    const at::Tensor& reordered_cat_ad_offsets,
    const at::Tensor& batch_offsets,
    const int64_t num_ads_in_batch,
    const bool broadcast_indices = false,
    const int64_t num_indices_after_broadcast = -1);
///@ingroup sparse-data-cpu
at::Tensor cat_reorder_batched_ad_indices_cpu(
    const at::Tensor& cat_ad_offsets,
    const std::vector<at::Tensor>& cat_ad_indices,
    const at::Tensor& reordered_cat_ad_offsets,
    const at::Tensor& batch_offsets,
    const int64_t num_ads_in_batch,
    const bool broadcast_indices,
    const int64_t num_indices_after_broadcast,
    const bool pinned_memory = false);
at::Tensor recat_embedding_grad_output_cuda(
    at::Tensor grad_output, // [B_local][T_global][D]
    const std::vector<int64_t>& num_features_per_rank);

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

///@ingroup sparse-data-cuda
at::Tensor batched_unary_embeddings_forward_cuda(
    const at::Tensor& weight,
    const at::Tensor& table_offsets,
    const at::Tensor& offsets,
    const at::Tensor& indices);

///@ingroup sparse-data-cuda
at::Tensor batched_unary_embeddings_backward_cuda(
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    const at::Tensor& table_offsets,
    const at::Tensor& offsets,
    const at::Tensor& indices);

///@ingroup sparse-data-cpu
std::vector<at::Tensor> stacked_jagged_2d_to_dense_cpu(
    at::Tensor values,
    at::Tensor lengths,
    const std::vector<int64_t>& offset_per_key,
    const std::vector<int64_t>& max_lengths_per_key,
    int64_t padding_value);

at::Tensor jagged_to_padded_dense(
    const at::Tensor& values,
    const std::vector<at::Tensor>& offsets,
    const c10::SymIntArrayRef max_lengths,
    const double padding_value);

at::Tensor jagged_dense_elementwise_add(
    const at::Tensor& x_values,
    const std::vector<at::Tensor>& x_offsets,
    const at::Tensor& y);

at::Tensor jagged_1d_to_dense(
    at::Tensor values,
    at::Tensor offsets,
    c10::SymInt max_L,
    int64_t padding_value);

at::Tensor jagged_2d_to_dense(
    at::Tensor values,
    at::Tensor offsets,
    c10::SymInt max_sequence_length);

at::Tensor jagged_1d_to_dense_meta(
    at::Tensor values,
    at::Tensor offsets,
    c10::SymInt max_L,
    int64_t padding_value);

at::Tensor jagged_2d_to_dense_meta(
    at::Tensor values,
    at::Tensor offsets,
    c10::SymInt max_sequence_length);

std::tuple<at::Tensor, std::vector<at::Tensor>>
jagged_dense_dense_elementwise_add_jagged_output(
    const at::Tensor& x_values,
    const std::vector<at::Tensor>& x_offsets,
    const at::Tensor& y_0,
    const at::Tensor& y_1);

at::Tensor batched_dense_vec_jagged_2d_mul(
    const at::Tensor& v,
    const at::Tensor& a_values,
    const at::Tensor& a_offsets);

std::tuple<at::Tensor, std::vector<at::Tensor>> jagged_dense_elementwise_mul(
    const at::Tensor& x_values,
    const std::vector<at::Tensor>& x_offsets,
    const at::Tensor& y);

std::tuple<at::Tensor, std::vector<at::Tensor>> dense_to_jagged(
    const at::Tensor& dense,
    const std::vector<at::Tensor>& offsets,
    c10::optional<at::SymInt> total_L);

std::tuple<at::Tensor, std::vector<at::Tensor>>
jagged_dense_elementwise_add_jagged_output(
    const at::Tensor& x_values,
    const std::vector<at::Tensor>& x_offsets,
    const at::Tensor& y);

///@ingroup sparse-data-cpu
at::Tensor jagged_2d_to_dense_forward_cpu(
    at::Tensor values,
    at::Tensor offsets,
    int64_t max_L);

///@ingroup sparse-data-cuda
at::Tensor jagged_2d_to_dense_gpu_forward(
    at::Tensor values,
    at::Tensor offsets,
    int64_t max_sequence_length);

///@ingroup sparse-data-cuda
at::Tensor jagged_2d_to_dense_gpu_backward(
    at::Tensor grad_output,
    at::Tensor offsets,
    int64_t max_lengths);

std::tuple<at::Tensor, at::Tensor> jagged_softmax(
    const at::Tensor& values,
    const at::Tensor& offsets,
    const int64_t max_L);

at::Tensor jagged_jagged_bmm(
    const at::Tensor& x_values,
    const at::Tensor& y_values,
    const at::Tensor& offsets,
    const int64_t max_L);

std::tuple<at::Tensor, at::Tensor> jagged_dense_bmm(
    const at::Tensor& x_values,
    const at::Tensor& x_offsets,
    const at::Tensor& y,
    const int64_t max_L);

std::tuple<at::Tensor, at::Tensor> masked_select_jagged_1d(
    const at::Tensor& values,
    const at::Tensor& lengths,
    const at::Tensor& mask);

#endif

///@ingroup sparse-data-cpu
/// Divide the prediction range (e.g., [0, 1]) into B bins. In each bin, use
/// two parameters to store the number of positive examples and the number
/// of examples that fall into this bucket. So we basically have a histogram
/// for the model prediction. As a result, for each bin, we have a
/// statistical value for the real CTR (`num_pos / num_example`). We use
/// this statistical value as the final calibrated prediction if the
/// pre-cali prediction falls into the corresponding bin. In this way, the
/// predictions within each bin should be well-calibrated if we have
/// sufficient examples. That is, we have a fine-grained calibrated model by
/// this calibration module. Theoretically, this calibration layer can fix
/// any uncalibrated model or prediction if we have sufficient bins and
/// examples.
///@return `[calibrated_prediction, bin_ids]`
///@param logit is input tensor before applying Sigmoid.
/// Assumes positive weight calibration is used for calibartion target, and
///@param positive_weight is passed as input argument.
/// The number of bins is automatically derived from `bin_num_examples`, and
///`bin_num_positives`, all of which should be the same size.
///@param lower/upper_bound Bounds of the bins.
///@param bin_ctr_in_use_after We will use the calibration_target for the
/// final calibrated prediction if we don't have sufficient examples. Only
/// use the statistical value of bin CTR after we observe
/// `bin_ctr_in_use_after` examples that fall in this bin. Default value: 0.
///@param bin_ctr_weight_value Weight for statistical value of bin CTR.
/// When this is specified, we perform a weighted sum for the statisctical
/// bin CTR and the calibration_target:
///```
/// final_calibrated_prediction = bin_ctr_weight * bin_ctr + (1 -
/// bin_ctr_weight) * calibration_target
///```
/// Default value: 1.0
std::tuple<at::Tensor, at::Tensor> histogram_binning_calibration_cpu(
    const at::Tensor& logit,
    const at::Tensor& bin_num_examples,
    const at::Tensor& bin_num_positives,
    double positive_weight,
    double lower_bound = 0.0,
    double upper_bound = 1.0,
    int64_t bin_ctr_in_use_after = 0,
    double bin_ctr_weight_value = 1.0);

#ifndef DOXYGEN_THIS_WILL_BE_SKIPPED
///@ingroup sparse-data-cuda
std::tuple<at::Tensor, at::Tensor> histogram_binning_calibration_cuda(
    const at::Tensor& logit,
    const at::Tensor& bin_num_examples,
    const at::Tensor& bin_num_positives,
    double positive_weight,
    double lower_bound = 0.0,
    double upper_bound = 1.0,
    int64_t bin_ctr_in_use_after = 0,
    double bin_ctr_weight_value = 1.0);
#endif

///@ingroup sparse-data-cpu
/// An extension of histogram binning calibration model which divides data into
/// bins based on one specific feature and prediction/ECTR range. In each bin,
/// use two parameters to store the number of positive examples and the number
/// of examples that fall into this bucket. So we basically have a histogram for
/// the model prediction. As a result, for each bin, we have a statistical
/// value for the real CTR (num_pos / num_example). We use this statistical
/// value as the final calibrated prediction if the pre-cali prediction falls
/// into the corresponding bin. In this way, the predictions within each bin
/// should be well-calibrated if we have sufficient examples. That is, we have
/// a fine-grained calibrated model by this calibration module. Theoretically,
/// this calibration layer can fix any uncalibrated model or prediction if we
/// have sufficient bins and examples.
///
///@return `[calibrated_prediction, bin_ids]`
///@param logit is input tensor before applying Sigmoid.
///
/// Assumes positive weight calibration is used for calibartion target, and
///`positive_weight` is passed as input argument.
///@param segment_value/lengths Values and lengths in KeyJaggedTensor.
/// Assumes value of length is either 0 or 1.
///@param num_bins # of bins is no longer the same as `bin_num_examples`,
/// and `bin_num_positives`, all of which should be still the same size.
///@param lower/upper_bound Bounds of the bins.
///@param bin_ctr_in_use_after We will use the calibration_target for
/// the final calibrated prediction if we don't have sufficient examples.
/// Only use the statistical value of bin CTR after we observe
/// `bin_ctr_in_use_after` examples that fall in this bin. Default value is `0`.
///@parambin_ctr_weight_value Weight for statistical value of bin CTR. When
/// this is specified, we perform a weighted sum for the statisctical
/// bin CTR and the calibration_target:
///```
/// final_calibrated_prediction = bin_ctr_weight * bin_ctr + (1 -
/// bin_ctr_weight) * calibration_target.
///```
/// Default value: `1.0`

#ifndef DOXYGEN_THIS_WILL_BE_SKIPPED
///@ingroup sparse-data-cpu
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

///@ingroup sparse-data-cuda
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
#endif

///@ingroup sparse-data-cpu
/// Same as above, but accepts generic "bin_boundaries", which is assumed to be
/// sorted.
///@return calibrated_prediction.
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

#ifndef DOXYGEN_THIS_WILL_BE_SKIPPED
///@ingroup sparse-data-cuda
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

///@ingroup sparse-data-cpu
at::Tensor lengths_range(
    const at::Tensor& t_in,
    const c10::optional<std::vector<int64_t>>& shape);

///@ingroup sparse-data-cpu
at::Tensor& lengths_range_out(
    at::Tensor& output,
    const at::Tensor& t_in,
    const c10::optional<std::vector<int64_t>>& shape);

///@ingroup sparse-data-cuda
at::Tensor lengths_range_cuda(
    const at::Tensor& t_in,
    const c10::optional<std::vector<int64_t>>& shape);
std::tuple<at::Tensor, at::Tensor, c10::optional<at::Tensor>>

///@ingroup sparse-data-cpu
permute_sparse_features_cpu(
    const at::Tensor& permute,
    const at::Tensor& lengths,
    const at::Tensor& indices,
    const c10::optional<at::Tensor>& weights);

///@ingroup sparse-data-cuda
std::tuple<at::Tensor, at::Tensor, c10::optional<at::Tensor>>
permute_sparse_features_cuda(
    const at::Tensor& permute,
    const at::Tensor& lengths,
    const at::Tensor& indices,
    const c10::optional<at::Tensor>& weights);

///@ingroup sparse-data-cuda
at::Tensor permute102_baddbmm_permute102_cuda(
    const at::Tensor& bias,
    const at::Tensor& A,
    const at::Tensor& B);

///@ingroup sparse-data-cpu
std::tuple<at::Tensor, at::Tensor> permute_sequence_embeddings_cpu(
    const at::Tensor& permute,
    const at::Tensor& lengths,
    const at::Tensor& embeddings);

///@ingroup sparse-data-cuda
std::tuple<at::Tensor, at::Tensor> permute_sequence_embeddings_cuda(
    const at::Tensor& permute,
    const at::Tensor& lengths,
    const at::Tensor& embeddings);

///@ingroup sparse-data-cpu
at::Tensor pack_segments_cpu(
    const at::Tensor& t_in,
    const at::Tensor& lengths,
    int64_t max_length);

///@ingroup sparse-data-cuda
at::Tensor pack_segments_cuda(
    const at::Tensor& t_in,
    const at::Tensor& lengths,
    int64_t max_length);

at::Tensor pack_segments_forward_cuda(
    const at::Tensor& t_in,
    const at::Tensor& lengths,
    int64_t max_length);

at::Tensor pack_segments_backward_cuda(
    const at::Tensor& data,
    const at::Tensor& lengths,
    int64_t total_length,
    int64_t max_length);

///@ingroup sparse-data-cuda
void compute_frequency_sequence(
    const at::Tensor& input,
    at::Tensor& output,
    const int start_input,
    const int output_size);

///@ingroup sparse-data-cuda
at::Tensor index_select_cuda(
    const at::Tensor& input,
    const at::Tensor& sorted_indices,
    const at::Tensor& orig_indices,
    const bool indices_sorted);

at::Tensor index_add_with_unique_indices_cuda(
    const at::Tensor& grad_output,
    const at::Tensor& sorted_indices,
    const at::Tensor& orig_indices,
    std::vector<int64_t>& input_shape,
    const int consecutive_range_start,
    const int consecutive_range_length);

///@ingroup sparse-data-cuda
void group_index_select_or_add_cuda(
    const int64_t* input_ptrs,
    const int64_t* output_ptrs,
    const int64_t* indices_ptrs,
    const int64_t* warp_offsets_group,
    const int32_t* num_cols_group,
    const c10::ScalarType& input_scalar_type,
    const c10::ScalarType& indices_scalar_type,
    const c10::DeviceIndex& device,
    const int num_work_rows,
    const int64_t total_num_warps,
    const int group_size,
    const bool use_index_select,
    const bool use_var_cols);

int get_group_index_select_cols_per_warp();

std::vector<at::Tensor> jagged_index_select_2d(
    const at::Tensor& values,
    const at::Tensor& lengths,
    const at::Tensor& indices);

at::Tensor jagged_index_select_2d_forward_cpu(
    const at::Tensor& values,
    const at::Tensor& indices,
    const at::Tensor& input_offsets,
    const at::Tensor& output_offsets,
    const int64_t num_dense_output_rows);

at::Tensor jagged_index_add_2d_forward_cpu(
    const at::Tensor& grad,
    const at::Tensor& indices,
    const at::Tensor& grad_offsets,
    const at::Tensor& output_offsets,
    const int64_t num_dense_grad_rows,
    const int64_t num_output_rows);

std::tuple<at::Tensor, at::Tensor> jagged_slice(
    const at::Tensor& x_values,
    const at::Tensor& x_lengths,
    const at::Tensor& start,
    const int64_t max_L);

at::Tensor jagged_slice_forward_cpu(
    const at::Tensor& x_values,
    const at::Tensor& x_lengths,
    const at::Tensor& src_start,
    const at::Tensor& output_lengths,
    const at::Tensor& tgt_start,
    const int64_t num_output_rows,
    const int64_t max_L,
    const bool fill_zeros);
#endif
} // namespace fbgemm_gpu
