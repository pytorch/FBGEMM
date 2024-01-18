/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAGuard.h>
#include "fbgemm_gpu/fbgemm_cuda_utils.cuh"
#include "fbgemm_gpu/sparse_ops.h"
#include "fbgemm_gpu/sparse_ops_utils.h"

using Tensor = at::Tensor;

namespace fbgemm_gpu {

template <typename T>
__global__
__launch_bounds__(kMaxThreads) void histogram_binning_calibration_kernel(
    const int64_t num_logits,
    const int64_t num_bins,
    const double recalibrate_value,
    const double step,
    const int64_t bin_ctr_in_use_after,
    const double bin_ctr_weight_value,
    const T* const logit_data,
    const double* const bin_num_examples_data,
    const double* const bin_num_positives_data,
    T* const calibrated_prediction_data,
    int64_t* const bin_ids_data) {
  const int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= num_logits) {
    return;
  }

  const T pre_sigmoid = logit_data[index] + recalibrate_value;
  const double uncalibrated = 1.0 / (1.0 + exp(-pre_sigmoid));

  bin_ids_data[index] = ceil(uncalibrated / step) - 1;

  const auto curr_bin_num_examples = bin_num_examples_data[bin_ids_data[index]];
  if (curr_bin_num_examples > bin_ctr_in_use_after) {
    const auto curr_bin_ctr =
        bin_num_positives_data[bin_ids_data[index]] / curr_bin_num_examples;
    calibrated_prediction_data[index] = curr_bin_ctr * bin_ctr_weight_value +
        uncalibrated * (1.0 - bin_ctr_weight_value);
  } else {
    calibrated_prediction_data[index] = uncalibrated;
  }
}

std::tuple<Tensor, Tensor> histogram_binning_calibration_cuda(
    const Tensor& logit,
    const Tensor& bin_num_examples,
    const Tensor& bin_num_positives,
    double positive_weight,
    double lower_bound,
    double upper_bound,
    int64_t bin_ctr_in_use_after,
    double bin_ctr_weight_value) {
  TENSOR_ON_CUDA_GPU(logit);
  TENSOR_ON_CUDA_GPU(bin_num_examples);
  TENSOR_ON_CUDA_GPU(bin_num_positives);
  TORCH_CHECK_EQ(bin_num_examples.numel(), bin_num_positives.numel());
  CUDA_DEVICE_GUARD(logit);

  Tensor calibrated_prediction = at::empty_like(logit);
  Tensor bin_ids = at::empty({logit.numel()}, logit.options().dtype(at::kLong));
  const double recalibrate_value = std::log(positive_weight);
  const double step = (upper_bound - lower_bound) /
      static_cast<double>(bin_num_examples.numel());

  const auto logit_packed = logit.contiguous();
  const auto bin_num_examples_packed = bin_num_examples.contiguous();
  const auto bin_num_positives_packed = bin_num_positives.contiguous();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      logit.scalar_type(),
      "histogram_binning_calibration_cuda",
      [&] {
        histogram_binning_calibration_kernel<scalar_t>
            <<<fbgemm_gpu::div_round_up(logit.numel(), kMaxThreads),
               kMaxThreads,
               0,
               at::cuda::getCurrentCUDAStream()>>>(
                logit.numel(),
                bin_num_examples.numel(),
                recalibrate_value,
                step,
                bin_ctr_in_use_after,
                bin_ctr_weight_value,
                logit_packed.data_ptr<scalar_t>(),
                bin_num_examples_packed.data_ptr<double>(),
                bin_num_positives_packed.data_ptr<double>(),
                calibrated_prediction.data_ptr<scalar_t>(),
                bin_ids.data_ptr<int64_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });

  return std::make_tuple(calibrated_prediction, bin_ids);
}

template <typename OffsetType, typename ValueType>
__global__ __launch_bounds__(kMaxThreads) void to_dense_segment_value_kernel(
    const int64_t num_lengths,
    const ValueType* const segment_value_data,
    const OffsetType* const segment_offsets_data,
    ValueType* const dense_segment_value_data) {
  const int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= num_lengths - 1) {
    return;
  }

  const auto curr_offset = segment_offsets_data[index];
  const auto next_offset = segment_offsets_data[index + 1];
  if (next_offset == curr_offset + 1) {
    // Add 1 to distinguish between 0 inserted by densification vs. original
    // value.
    dense_segment_value_data[index] = segment_value_data[curr_offset] + 1;
  } else {
    dense_segment_value_data[index] = 0;
  }
}

template <typename LogitType, typename SegmentValueType>
__global__
__launch_bounds__(kMaxThreads) void histogram_binning_calibration_by_feature_kernel(
    const int64_t num_logits,
    const int64_t num_bins,
    const int64_t num_segments,
    const double recalibrate_value,
    const double step,
    const int64_t bin_ctr_in_use_after,
    const double bin_ctr_weight_value,
    const LogitType* const logit_data,
    const SegmentValueType* const dense_segment_value_data,
    const double* const bin_num_examples_data,
    const double* const bin_num_positives_data,
    LogitType* const calibrated_prediction_data,
    int64_t* const bin_ids_data) {
  const int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= num_logits) {
    return;
  }

  const LogitType pre_sigmoid = logit_data[index] + recalibrate_value;
  const double uncalibrated = 1.0 / (1.0 + exp(-pre_sigmoid));

  const int64_t curr_segment_value =
      dense_segment_value_data[index] > num_segments
      ? 0
      : std::max(0L, dense_segment_value_data[index] * num_bins);

  bin_ids_data[index] = ceil(uncalibrated / step) - 1 + curr_segment_value;

  const auto curr_bin_num_examples = bin_num_examples_data[bin_ids_data[index]];
  if (curr_bin_num_examples > bin_ctr_in_use_after) {
    const auto curr_bin_ctr =
        bin_num_positives_data[bin_ids_data[index]] / curr_bin_num_examples;
    calibrated_prediction_data[index] = curr_bin_ctr * bin_ctr_weight_value +
        uncalibrated * (1.0 - bin_ctr_weight_value);
  } else {
    calibrated_prediction_data[index] = uncalibrated;
  }
}

std::tuple<Tensor, Tensor> histogram_binning_calibration_by_feature_cuda(
    const Tensor& logit,
    const Tensor& segment_value,
    const Tensor& segment_lengths,
    int64_t num_segments,
    const Tensor& bin_num_examples,
    const Tensor& bin_num_positives,
    int64_t num_bins,
    double positive_weight,
    double lower_bound,
    double upper_bound,
    int64_t bin_ctr_in_use_after,
    double bin_ctr_weight_value) {
  TENSOR_ON_CUDA_GPU(logit);
  TENSOR_ON_CUDA_GPU(segment_value);
  TENSOR_ON_CUDA_GPU(segment_lengths);
  TENSOR_ON_CUDA_GPU(bin_num_examples);
  TENSOR_ON_CUDA_GPU(bin_num_positives);
  TORCH_CHECK_EQ(bin_num_examples.numel(), bin_num_positives.numel());
  CUDA_DEVICE_GUARD(logit);

  // Convert lengths to offsets for better handling on GPUs.
  const auto segment_lengths_packed = segment_lengths.contiguous();
  auto segment_offsets =
      asynchronous_complete_cumsum_gpu(segment_lengths_packed.view(-1));

  // dense_segment_value is used as a temporary storage.
  Tensor dense_segment_value =
      at::empty({logit.numel()}, segment_value.options());

  const auto segment_value_packed = segment_value.contiguous();
  const auto segment_offsets_packed = segment_offsets.contiguous();
  auto dense_segment_value_packed = dense_segment_value.contiguous();
  AT_DISPATCH_INDEX_TYPES(
      segment_offsets.scalar_type(),
      "to_dense_segment_value_cuda_wrapper",
      [&] {
        using offset_t = index_t;
        AT_DISPATCH_INDEX_TYPES(
            segment_value.scalar_type(), "to_dense_segment_value_cuda", [&] {
              using value_t = index_t;
              to_dense_segment_value_kernel<offset_t, value_t>
                  <<<fbgemm_gpu::div_round_up(
                         segment_offsets.numel(), kMaxThreads),
                     kMaxThreads,
                     0,
                     at::cuda::getCurrentCUDAStream()>>>(
                      segment_offsets.numel(),
                      segment_value_packed.data_ptr<value_t>(),
                      segment_offsets_packed.data_ptr<offset_t>(),
                      dense_segment_value_packed.data_ptr<value_t>());
              C10_CUDA_KERNEL_LAUNCH_CHECK();
            });
      });

  Tensor calibrated_prediction = at::empty_like(logit);
  Tensor bin_ids = at::empty({logit.numel()}, logit.options().dtype(at::kLong));
  const double recalibrate_value = std::log(positive_weight);
  const double step =
      (upper_bound - lower_bound) / static_cast<double>(num_bins);

  const auto logit_packed = logit.contiguous();
  const auto bin_num_examples_packed = bin_num_examples.contiguous();
  const auto bin_num_positives_packed = bin_num_positives.contiguous();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      logit.scalar_type(),
      "histogram_binning_calibration_by_feature_cuda_wrapper",
      [&] {
        using logit_t = scalar_t;
        AT_DISPATCH_INDEX_TYPES(
            dense_segment_value_packed.scalar_type(),
            "histogram_binning_calibration_by_feature_cuda",
            [&] {
              using segment_value_t = index_t;
              histogram_binning_calibration_by_feature_kernel<
                  logit_t,
                  segment_value_t>
                  <<<fbgemm_gpu::div_round_up(logit.numel(), kMaxThreads),
                     kMaxThreads,
                     0,
                     at::cuda::getCurrentCUDAStream()>>>(
                      logit.numel(),
                      num_bins,
                      num_segments,
                      recalibrate_value,
                      step,
                      bin_ctr_in_use_after,
                      bin_ctr_weight_value,
                      logit_packed.data_ptr<logit_t>(),
                      dense_segment_value_packed.data_ptr<segment_value_t>(),
                      bin_num_examples_packed.data_ptr<double>(),
                      bin_num_positives_packed.data_ptr<double>(),
                      calibrated_prediction.data_ptr<logit_t>(),
                      bin_ids.data_ptr<int64_t>());
              C10_CUDA_KERNEL_LAUNCH_CHECK();
            });
      });

  return std::make_tuple(calibrated_prediction, bin_ids);
}

template <typename LogitType, typename SegmentValueType>
__global__
__launch_bounds__(kMaxThreads) void generic_histogram_binning_calibration_by_feature_kernel(
    const int64_t num_logits,
    const int64_t num_bins,
    const int64_t num_segments,
    const double recalibrate_value,
    const int64_t bin_ctr_in_use_after,
    const double bin_ctr_weight_value,
    const LogitType* const logit_data,
    const SegmentValueType* const dense_segment_value_data,
    const double* const bin_num_examples_data,
    const double* const bin_num_positives_data,
    const double* const bin_boundaries,
    LogitType* const calibrated_prediction_data,
    int64_t* const bin_ids_data) {
  const int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= num_logits) {
    return;
  }

  const LogitType pre_sigmoid = logit_data[index] + recalibrate_value;
  const double uncalibrated = 1.0 / (1.0 + exp(-pre_sigmoid));

  // Perform binary search.
  int left = 0;
  int right = num_bins - 1;
  while (left != right) {
    const int middle = (left + right) >> 1;
    if (bin_boundaries[middle] < uncalibrated) {
      left = middle + 1;
    } else {
      right = middle;
    }
  }
  const int curr_bin_id = left;

  const int64_t curr_segment_value =
      dense_segment_value_data[index] > num_segments
      ? 0
      : std::max(0L, dense_segment_value_data[index] * num_bins);

  bin_ids_data[index] = curr_bin_id + curr_segment_value;

  const auto curr_bin_num_examples = bin_num_examples_data[bin_ids_data[index]];
  if (curr_bin_num_examples > bin_ctr_in_use_after) {
    const auto curr_bin_ctr =
        bin_num_positives_data[bin_ids_data[index]] / curr_bin_num_examples;
    calibrated_prediction_data[index] = curr_bin_ctr * bin_ctr_weight_value +
        uncalibrated * (1.0 - bin_ctr_weight_value);
  } else {
    calibrated_prediction_data[index] = uncalibrated;
  }
}

std::tuple<Tensor, Tensor>
generic_histogram_binning_calibration_by_feature_cuda(
    const Tensor& logit,
    const Tensor& segment_value,
    const Tensor& segment_lengths,
    int64_t num_segments,
    const Tensor& bin_num_examples,
    const Tensor& bin_num_positives,
    const Tensor& bin_boundaries,
    double positive_weight,
    int64_t bin_ctr_in_use_after,
    double bin_ctr_weight_value) {
  TENSOR_ON_CUDA_GPU(logit);
  TENSOR_ON_CUDA_GPU(segment_value);
  TENSOR_ON_CUDA_GPU(segment_lengths);
  TENSOR_ON_CUDA_GPU(bin_num_examples);
  TENSOR_ON_CUDA_GPU(bin_num_positives);
  TENSOR_ON_CUDA_GPU(bin_boundaries);
  TORCH_CHECK_EQ(bin_num_examples.numel(), bin_num_positives.numel());
  TORCH_CHECK(
      bin_num_examples.numel() ==
      (num_segments + 1) * (bin_boundaries.numel() + 1));
  CUDA_DEVICE_GUARD(logit);

  // Convert lengths to offsets for better handling on GPUs.
  const auto segment_lengths_packed = segment_lengths.contiguous();
  auto segment_offsets =
      asynchronous_complete_cumsum_gpu(segment_lengths_packed.view(-1));

  // dense_segment_value is used as a temporary storage.
  Tensor dense_segment_value =
      at::empty({logit.numel()}, segment_value.options());

  const auto segment_value_packed = segment_value.contiguous();
  const auto segment_offsets_packed = segment_offsets.contiguous();
  auto dense_segment_value_packed = dense_segment_value.contiguous();
  AT_DISPATCH_INDEX_TYPES(
      segment_offsets.scalar_type(),
      "to_dense_segment_value_cuda_wrapper",
      [&] {
        using offset_t = index_t;
        AT_DISPATCH_INDEX_TYPES(
            segment_value.scalar_type(), "to_dense_segment_value_cuda", [&] {
              using value_t = index_t;
              to_dense_segment_value_kernel<offset_t, value_t>
                  <<<fbgemm_gpu::div_round_up(
                         segment_offsets.numel(), kMaxThreads),
                     kMaxThreads,
                     0,
                     at::cuda::getCurrentCUDAStream()>>>(
                      segment_offsets.numel(),
                      segment_value_packed.data_ptr<value_t>(),
                      segment_offsets_packed.data_ptr<offset_t>(),
                      dense_segment_value_packed.data_ptr<value_t>());
              C10_CUDA_KERNEL_LAUNCH_CHECK();
            });
      });

  Tensor calibrated_prediction = at::empty_like(logit);
  Tensor bin_ids = at::empty({logit.numel()}, logit.options().dtype(at::kLong));
  const double recalibrate_value = std::log(positive_weight);

  const auto logit_packed = logit.contiguous();
  const auto bin_num_examples_packed = bin_num_examples.contiguous();
  const auto bin_num_positives_packed = bin_num_positives.contiguous();
  const auto bin_boundaries_packed = bin_boundaries.contiguous();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      logit.scalar_type(),
      "generic_histogram_binning_calibration_by_feature_cuda_wrapper",
      [&] {
        using logit_t = scalar_t;
        AT_DISPATCH_INDEX_TYPES(
            dense_segment_value_packed.scalar_type(),
            "generic_histogram_binning_calibration_by_feature_cuda",
            [&] {
              using segment_value_t = index_t;
              generic_histogram_binning_calibration_by_feature_kernel<
                  logit_t,
                  segment_value_t>
                  <<<fbgemm_gpu::div_round_up(logit.numel(), kMaxThreads),
                     kMaxThreads,
                     0,
                     at::cuda::getCurrentCUDAStream()>>>(
                      logit.numel(),
                      bin_boundaries.numel() + 1,
                      num_segments,
                      recalibrate_value,
                      bin_ctr_in_use_after,
                      bin_ctr_weight_value,
                      logit_packed.data_ptr<logit_t>(),
                      dense_segment_value_packed.data_ptr<segment_value_t>(),
                      bin_num_examples_packed.data_ptr<double>(),
                      bin_num_positives_packed.data_ptr<double>(),
                      bin_boundaries_packed.data_ptr<double>(),
                      calibrated_prediction.data_ptr<logit_t>(),
                      bin_ids.data_ptr<int64_t>());
              C10_CUDA_KERNEL_LAUNCH_CHECK();
            });
      });

  return std::make_tuple(calibrated_prediction, bin_ids);
}

} // namespace fbgemm_gpu
