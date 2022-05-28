/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>

namespace fbgemm_gpu {

std::tuple<at::Tensor, at::Tensor, at::Tensor> tbe_input_combine_cpu(
    const std::vector<at::Tensor>& indices_list,
    const std::vector<at::Tensor>& offsets_list,
    const std::vector<at::Tensor>& per_sample_weights,
    const at::Tensor& include_last_offsets);

std::tuple<at::Tensor, at::Tensor, at::Tensor>
padding_fused_tbe_input_combine_cpu(
    const std::vector<at::Tensor>& indices_list,
    const std::vector<at::Tensor>& offsets_list,
    const std::vector<at::Tensor>& per_sample_weights,
    const at::Tensor& include_last_offsets,
    int64_t batch_size);

template <typename TensorList>
void _cat_int_tensors_out(
    at::Tensor& out,
    const int64_t n_out,
    const TensorList& tensor_list) {
  at::native::resize_(out, n_out);
  int32_t* out_ptr = out.data_ptr<int32_t>();

  for (const at::Tensor& t : tensor_list) {
    AT_DISPATCH_INDEX_TYPES(t.scalar_type(), "tbe_cat_inputs_", [&out_ptr, &t] {
      const int64_t n_elements = t.numel();
      const index_t* in_ptr = t.template data_ptr<index_t>();
      for (int64_t j = 0; j < n_elements; j++) {
        *out_ptr++ = static_cast<int32_t>(*in_ptr++);
      }
    });
  }
}

template <typename TensorList>
void _cat_per_sample_weights_list_out(
    at::Tensor& out,
    const TensorList& per_sample_weights,
    const bool use_weights,
    const TensorList& indices_list,
    const int64_t total_indices) {
  if (use_weights) {
    at::native::resize_(out, total_indices);
    out.fill_(1.);
    float* out_ptr = out.data_ptr<float>();
    // it is expected to have same dim for per_sample_weights and indices_list
    for (size_t i = 0; i < per_sample_weights.size(); i++) {
      const int64_t n_weights = per_sample_weights[i].numel();
      const int64_t n_out_advance = indices_list[i].numel();
      if (n_weights) {
        memcpy(
            out_ptr,
            per_sample_weights[i].template data_ptr<float>(),
            n_out_advance * sizeof(float));
      }
      out_ptr += n_out_advance;
    }
  } else {
    at::native::resize_(out, 0);
  }
}

template <typename TensorList>
void tbe_input_combine_with_length_cpu_out(
    at::Tensor& combined_indices,
    at::Tensor& combined_lengths,
    at::Tensor& combined_per_sample_weights,
    const TensorList& indices_list,
    const TensorList& lengths_list,
    const TensorList& per_sample_weights) {
  TORCH_CHECK(indices_list.size() > 0);
  TORCH_CHECK(lengths_list.size() == indices_list.size());
  TORCH_CHECK(per_sample_weights.size() == indices_list.size());

  for (size_t i = 0; i < indices_list.size(); i++) {
    TORCH_CHECK(
        indices_list[i].dtype() == c10::kInt ||
        indices_list[i].dtype() == c10::kLong);
    TORCH_CHECK(
        lengths_list[i].dtype() == c10::kInt ||
        lengths_list[i].dtype() == c10::kLong);
    TORCH_CHECK(indices_list[i].ndimension() == 1);
    TORCH_CHECK(lengths_list[i].ndimension() == 1);
    TORCH_CHECK(indices_list[i].is_contiguous());
    TORCH_CHECK(lengths_list[i].is_contiguous());

    if (per_sample_weights[i].numel() > 0) {
      TORCH_CHECK(per_sample_weights[i].ndimension() == 1);
      TORCH_CHECK(per_sample_weights[i].numel() == indices_list[i].numel());
      TORCH_CHECK(per_sample_weights[i].is_contiguous());
    }
  }

  const auto length_accumulator = [](int64_t acc, const at::Tensor& t) {
    return t.numel() + acc;
  };
  const int64_t total_indices = std::accumulate(
      indices_list.begin(), indices_list.end(), 0LL, length_accumulator);
  const int64_t total_lengths = std::accumulate(
      lengths_list.begin(), lengths_list.end(), 0LL, length_accumulator);
  const int64_t total_weights = std::accumulate(
      per_sample_weights.begin(),
      per_sample_weights.end(),
      0LL,
      length_accumulator);

  _cat_int_tensors_out(combined_indices, total_indices, indices_list);

  _cat_int_tensors_out(combined_lengths, total_lengths, lengths_list);

  _cat_per_sample_weights_list_out(
      combined_per_sample_weights,
      per_sample_weights,
      /* use_weights = (bool) */ total_weights,
      indices_list,
      total_indices);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
tbe_input_combine_with_length_cpu(
    const std::vector<at::Tensor>& indices_list,
    const std::vector<at::Tensor>& lengths_list,
    const std::vector<at::Tensor>& per_sample_weights);

} // namespace fbgemm_gpu
