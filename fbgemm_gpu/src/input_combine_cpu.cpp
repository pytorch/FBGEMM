/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "fbgemm_gpu/input_combine.h"
#include "fbgemm_gpu/sparse_ops_utils.h"

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/Functions.h>
#include <ATen/TypeDefault.h>
#include <ATen/core/op_registration/op_registration.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Exception.h>
#include <torch/script.h>

using Tensor = at::Tensor;

namespace fbgemm_gpu {

void _cat_int_tensors_out(
    Tensor& out,
    const int64_t n_out,
    const std::vector<Tensor>& tensor_list) {
  at::native::resize_(out, n_out);
  int32_t* out_ptr = out.data_ptr<int32_t>();

  for (const Tensor& t : tensor_list) {
    AT_DISPATCH_INDEX_TYPES(t.scalar_type(), "tbe_cat_inputs_", [&out_ptr, &t] {
      const int64_t n_elements = t.numel();
      const index_t* in_ptr = t.data_ptr<index_t>();
      for (int64_t j = 0; j < n_elements; j++) {
        *out_ptr++ = static_cast<int32_t>(*in_ptr++);
      }
    });
  }
}

Tensor _cat_int_tensors(
    const std::vector<Tensor>& tensor_list,
    int64_t total_num,
    bool use_pin_memory) {
  Tensor combined_tensors = at::empty(
      {total_num},
      at::TensorOptions()
          .dtype(c10::kInt)
          .device(tensor_list[0].device())
          .pinned_memory(use_pin_memory));

  _cat_int_tensors_out(combined_tensors, total_num, tensor_list);
  return combined_tensors;
}

void _cat_per_sample_weights_list_out(
    Tensor& out,
    const std::vector<Tensor>& per_sample_weights,
    const bool use_weights,
    const std::vector<Tensor>& indices_list,
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
            per_sample_weights[i].data_ptr<float>(),
            n_out_advance * sizeof(float));
      }
      out_ptr += n_out_advance;
    }
  } else {
    at::native::resize_(out, 0);
  }
}

Tensor _cat_per_sample_weights_list(
    const std::vector<Tensor>& per_sample_weights,
    const bool use_weights,
    const std::vector<Tensor>& indices_list,
    const int64_t total_indices,
    bool use_pin_memory) {
  auto combined_weights = at::ones(
      {total_indices},
      at::TensorOptions()
          .dtype(c10::kFloat)
          .device(per_sample_weights[0].device())
          .pinned_memory(use_pin_memory));
  _cat_per_sample_weights_list_out(
      combined_weights,
      per_sample_weights,
      use_weights,
      indices_list,
      total_indices);
  return combined_weights;
}

std::tuple<Tensor, Tensor, Tensor> tbe_input_combine_cpu(
    const std::vector<Tensor>& indices_list,
    const std::vector<Tensor>& offsets_list,
    const std::vector<Tensor>& per_sample_weights,
    const Tensor& include_last_offsets) {
  TORCH_CHECK(indices_list.size() > 0);
  TORCH_CHECK(offsets_list.size() == indices_list.size());
  TORCH_CHECK(per_sample_weights.size() == indices_list.size());
  TORCH_CHECK(
      static_cast<uint64_t>(include_last_offsets.numel()) ==
      indices_list.size());
  auto include_last_offsets_acc = include_last_offsets.accessor<bool, 1>();
  int64_t total_indices = 0;
  int64_t total_offsets = 1;
  bool need_weights = false;
  bool pin_memory = false;
  if (at::Context::hasCUDA() && at::getNumGPUs() > 0) {
    pin_memory = true;
  }

  for (size_t i = 0; i < indices_list.size(); i++) {
    TORCH_CHECK(
        indices_list[i].dtype() == c10::kInt ||
        indices_list[i].dtype() == c10::kLong);
    TORCH_CHECK(
        offsets_list[i].dtype() == c10::kInt ||
        offsets_list[i].dtype() == c10::kLong);
    TORCH_CHECK(indices_list[i].ndimension() == 1);
    TORCH_CHECK(offsets_list[i].ndimension() == 1);
    TORCH_CHECK(indices_list[i].is_contiguous());
    TORCH_CHECK(offsets_list[i].is_contiguous());
    total_indices += indices_list[i].numel();
    auto num_offset =
        offsets_list[i].numel() - (include_last_offsets_acc[i] ? 1 : 0);
    total_offsets += num_offset == 0 ? 1 : num_offset;

    if (per_sample_weights[i].numel() > 0) {
      TORCH_CHECK(per_sample_weights[i].ndimension() == 1);
      TORCH_CHECK(per_sample_weights[i].numel() == indices_list[i].numel());
      TORCH_CHECK(per_sample_weights[i].is_contiguous());
      need_weights = true;
    }
  }

  auto combined_indices =
      _cat_int_tensors(indices_list, total_indices, pin_memory);

  auto combined_offsets = at::empty(
      {total_offsets},
      at::TensorOptions()
          .dtype(c10::kInt)
          .device(offsets_list[0].device())
          .pinned_memory(pin_memory));

  auto combined_offsets_acc = combined_offsets.accessor<int32_t, 1>();
  int32_t offset = 0;
  size_t offsets_acc_idx = 0;
  combined_offsets_acc[offsets_acc_idx++] = 0;

  for (size_t i = 0; i < offsets_list.size(); i++) {
    AT_DISPATCH_INDEX_TYPES(
        offsets_list[i].scalar_type(), "tbe_input_offsets_", [&] {
          auto offsets_acc = offsets_list[i].accessor<index_t, 1>();
          for (int64_t j = 1,
                       size = offsets_list[i].numel() -
                   (include_last_offsets_acc[i] ? 1 : 0);
               j < size;
               j++) {
            combined_offsets_acc[offsets_acc_idx++] =
                offset + static_cast<int32_t>(offsets_acc[j]);
          }

          offset += static_cast<int32_t>(indices_list[i].numel());
          combined_offsets_acc[offsets_acc_idx++] = offset;
        });
  }

  auto combined_per_sample_weights = _cat_per_sample_weights_list(
      per_sample_weights,
      need_weights,
      indices_list,
      total_indices,
      pin_memory);

  return {combined_indices, combined_offsets, combined_per_sample_weights};
}

void tbe_input_combine_with_length_cpu_out(
    Tensor& combined_indices,
    Tensor& combined_lengths,
    Tensor& combined_per_sample_weights,
    const std::vector<Tensor>& indices_list,
    const std::vector<Tensor>& lengths_list,
    const std::vector<Tensor>& per_sample_weights) {
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

  const auto length_accumulator = [](int64_t acc, const Tensor& t) {
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

std::tuple<Tensor, Tensor, Tensor> tbe_input_combine_with_length_cpu(
    const std::vector<Tensor>& indices_list,
    const std::vector<Tensor>& lengths_list,
    const std::vector<Tensor>& per_sample_weights) {
  const bool use_pin_memory = at::Context::hasCUDA() && at::getNumGPUs() > 0;
  TORCH_CHECK(indices_list.size() > 0);
  TORCH_CHECK(lengths_list.size() == indices_list.size());
  TORCH_CHECK(per_sample_weights.size() == indices_list.size());

  Tensor combined_indices = at::empty(
      {0},
      at::TensorOptions()
          .dtype(c10::kInt)
          .device(indices_list[0].device())
          .pinned_memory(use_pin_memory));
  Tensor combined_lengths = at::empty(
      {0},
      at::TensorOptions()
          .dtype(c10::kInt)
          .device(lengths_list[0].device())
          .pinned_memory(use_pin_memory));
  Tensor combined_per_sample_weights = at::empty(
      {0},
      at::TensorOptions()
          .dtype(c10::kFloat)
          .device(per_sample_weights[0].device())
          .pinned_memory(use_pin_memory));
  tbe_input_combine_with_length_cpu_out(
      combined_indices,
      combined_lengths,
      combined_per_sample_weights,
      indices_list,
      lengths_list,
      per_sample_weights);
  return {combined_indices, combined_lengths, combined_per_sample_weights};
}

// Similar to tbe_input_combine_cpu, but padding all the offsets
// to the size specified by batch_size.

std::tuple<Tensor, Tensor, Tensor> padding_fused_tbe_input_combine_cpu(
    const std::vector<Tensor>& indices_list,
    const std::vector<Tensor>& offsets_list,
    const std::vector<Tensor>& per_sample_weights,
    const Tensor& include_last_offsets,
    int64_t batch_size) {
  TORCH_CHECK(indices_list.size() > 0);
  TORCH_CHECK(offsets_list.size() == indices_list.size());
  TORCH_CHECK(per_sample_weights.size() == indices_list.size());
  TORCH_CHECK(
      static_cast<uint64_t>(include_last_offsets.numel()) ==
      indices_list.size());
  auto include_last_offsets_acc = include_last_offsets.accessor<bool, 1>();
  int64_t total_indices = 0;
  int64_t total_offsets = 1 + batch_size * indices_list.size();
  bool need_weights = false;
  bool pin_memory = false;
  if (at::Context::hasCUDA() && at::getNumGPUs() > 0) {
    pin_memory = true;
  }

  for (size_t i = 0; i < indices_list.size(); i++) {
    TORCH_CHECK(
        indices_list[i].dtype() == c10::kInt ||
        indices_list[i].dtype() == c10::kLong);
    TORCH_CHECK(
        offsets_list[i].dtype() == c10::kInt ||
        offsets_list[i].dtype() == c10::kLong);
    TORCH_CHECK(indices_list[i].ndimension() == 1);
    TORCH_CHECK(offsets_list[i].ndimension() == 1);
    TORCH_CHECK(indices_list[i].is_contiguous());
    TORCH_CHECK(offsets_list[i].is_contiguous());
    total_indices += indices_list[i].numel();

    if (per_sample_weights[i].numel() > 0) {
      TORCH_CHECK(per_sample_weights[i].ndimension() == 1);
      TORCH_CHECK(per_sample_weights[i].numel() == indices_list[i].numel());
      TORCH_CHECK(per_sample_weights[i].is_contiguous());
      need_weights = true;
    }
  }

  auto combined_indices =
      _cat_int_tensors(indices_list, total_indices, pin_memory);

  auto combined_offsets = at::empty(
      {total_offsets},
      at::TensorOptions()
          .dtype(c10::kInt)
          .device(offsets_list[0].device())
          .pinned_memory(pin_memory));

  auto combined_offsets_acc = combined_offsets.accessor<int32_t, 1>();
  int32_t offset = 0;
  size_t offsets_acc_idx = 0;
  combined_offsets_acc[offsets_acc_idx++] = 0;

  for (size_t i = 0; i < offsets_list.size(); i++) {
    AT_DISPATCH_INDEX_TYPES(
        offsets_list[i].scalar_type(), "tbe_input_offsets_", [&] {
          auto offsets_acc = offsets_list[i].accessor<index_t, 1>();
          int64_t offsets_size =
              offsets_list[i].numel() - (include_last_offsets_acc[i] ? 1 : 0);
          for (int64_t j = 1; j < offsets_size; j++) {
            combined_offsets_acc[offsets_acc_idx++] =
                offset + static_cast<int32_t>(offsets_acc[j]);
          }
          offset += static_cast<int32_t>(indices_list[i].numel());
          for (int64_t j = offsets_size; j <= batch_size; j++) {
            combined_offsets_acc[offsets_acc_idx++] = offset;
          }
        });
  }

  auto combined_per_sample_weights = _cat_per_sample_weights_list(
      per_sample_weights,
      need_weights,
      indices_list,
      total_indices,
      pin_memory);

  return {combined_indices, combined_offsets, combined_per_sample_weights};
}

} // namespace fbgemm_gpu

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.def(
      "tbe_input_combine(Tensor[] indices_list, Tensor[] offsets_list, Tensor[] per_sample_weights, Tensor include_last_offsets) -> (Tensor, Tensor, Tensor)");
  m.def(
      "tbe_input_combine_with_length(Tensor[] indices_list, Tensor[] lengths_list, Tensor[] per_sample_weights) -> (Tensor, Tensor, Tensor)");
  m.def(
      "padding_fused_tbe_input_combine(Tensor[] indices_list, Tensor[] offsets_list, Tensor[] per_sample_weights, Tensor include_last_offsets, int batch_size) -> (Tensor, Tensor, Tensor)");
  DISPATCH_TO_CPU("tbe_input_combine", fbgemm_gpu::tbe_input_combine_cpu);
  DISPATCH_TO_CPU(
      "tbe_input_combine_with_length",
      fbgemm_gpu::tbe_input_combine_with_length_cpu);
  DISPATCH_TO_CPU(
      "padding_fused_tbe_input_combine",
      fbgemm_gpu::padding_fused_tbe_input_combine_cpu);
}
