/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "fbgemm_gpu/dispatch_macros.h"
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

Tensor _cat_int_tensors(
    const std::vector<Tensor>& tensor_list,
    int64_t total_num,
    bool use_pin_memory) {
  auto combined_tensors = at::empty(
      {total_num},
      at::TensorOptions()
          .dtype(c10::kInt)
          .device(tensor_list[0].device())
          .pinned_memory(use_pin_memory));

  auto combined_tensors_acc = combined_tensors.accessor<int32_t, 1>();
  size_t idx = 0;

  for (size_t i = 0; i < tensor_list.size(); i++) {
    AT_DISPATCH_INDEX_TYPES(
        tensor_list[i].scalar_type(), "tbe_cat_inputs_", [&] {
          auto indices_acc = tensor_list[i].accessor<index_t, 1>();
          for (auto j = 0; j < tensor_list[i].numel(); j++) {
            combined_tensors_acc[idx++] = static_cast<int32_t>(indices_acc[j]);
          }
        });
  }
  return combined_tensors;
}

Tensor _cat_int_tensors_with_padding(
    const std::vector<Tensor>& tensor_list,
    int64_t total_num,
    bool use_pin_memory,
    int64_t batch_size) {
  auto combined_tensors = at::zeros(
      {total_num},
      at::TensorOptions()
          .dtype(c10::kInt)
          .device(tensor_list[0].device())
          .pinned_memory(use_pin_memory));

  auto combined_tensors_acc = combined_tensors.accessor<int32_t, 1>();

  for (size_t i = 0; i < tensor_list.size(); i++) {
    size_t idx = i * batch_size;
    AT_DISPATCH_INDEX_TYPES(
        tensor_list[i].scalar_type(), "tbe_cat_inputs_", [&] {
          auto indices_acc = tensor_list[i].accessor<index_t, 1>();
          for (auto j = 0; j < tensor_list[i].numel(); j++) {
            combined_tensors_acc[idx++] = static_cast<int32_t>(indices_acc[j]);
          }
        });
  }
  return combined_tensors;
}

Tensor _cat_per_sample_weights_list(
    const std::vector<Tensor>& per_sample_weights,
    const std::vector<Tensor>& indices_list,
    int64_t total_num,
    bool use_pin_memory) {
  auto combined_weights = at::ones(
      {total_num},
      at::TensorOptions()
          .dtype(c10::kFloat)
          .device(per_sample_weights[0].device())
          .pinned_memory(use_pin_memory));
  auto* combined_weights_ptr = combined_weights.data_ptr<float>();

  for (size_t i = 0; i < per_sample_weights.size(); i++) {
    auto element_size = per_sample_weights[i].numel();
    if (element_size != 0) {
      memcpy(
          combined_weights_ptr,
          per_sample_weights[i].data_ptr<float>(),
          element_size * sizeof(float));
    }
    combined_weights_ptr += indices_list[i].numel();
  }
  return combined_weights;
}

std::tuple<Tensor, Tensor, Tensor> tbe_input_combine_cpu(
    const std::vector<Tensor>& indices_list,
    const std::vector<Tensor>& offsets_list,
    const std::vector<Tensor>& per_sample_weights,
    const Tensor& include_last_offsets) {
  TORCH_CHECK_GT(indices_list.size(), 0);
  TORCH_CHECK_EQ(offsets_list.size(), indices_list.size());
  TORCH_CHECK_EQ(per_sample_weights.size(), indices_list.size());
  TORCH_CHECK(
      static_cast<uint64_t>(include_last_offsets.numel()) ==
      indices_list.size());
  auto include_last_offsets_acc = include_last_offsets.accessor<bool, 1>();
  int64_t total_indices = 0;
  int64_t total_offsets = 1;
  bool need_weights = false;
  bool pin_memory = false;

  for (size_t i = 0; i < indices_list.size(); i++) {
    TORCH_CHECK(
        indices_list[i].dtype() == c10::kInt ||
        indices_list[i].dtype() == c10::kLong);
    TORCH_CHECK(
        offsets_list[i].dtype() == c10::kInt ||
        offsets_list[i].dtype() == c10::kLong);
    TORCH_CHECK_EQ(indices_list[i].ndimension(), 1);
    TORCH_CHECK_EQ(offsets_list[i].ndimension(), 1);
    TORCH_CHECK(indices_list[i].is_contiguous());
    TORCH_CHECK(offsets_list[i].is_contiguous());
    total_indices += indices_list[i].numel();
    auto num_offset =
        offsets_list[i].numel() - (include_last_offsets_acc[i] ? 1 : 0);
    total_offsets += num_offset == 0 ? 1 : num_offset;

    if (per_sample_weights[i].numel() > 0) {
      TORCH_CHECK_EQ(per_sample_weights[i].ndimension(), 1);
      TORCH_CHECK_EQ(per_sample_weights[i].numel(), indices_list[i].numel());
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

  if (need_weights) {
    return {
        std::move(combined_indices),
        std::move(combined_offsets),
        _cat_per_sample_weights_list(
            per_sample_weights, indices_list, total_indices, pin_memory)};
  }
  return {combined_indices, combined_offsets, at::empty({0})};
}

std::tuple<Tensor, Tensor, Tensor> tbe_input_combine_with_length_cpu(
    const std::vector<Tensor>& indices_list,
    const std::vector<Tensor>& lengths_list,
    const std::vector<Tensor>& per_sample_weights) {
  TORCH_CHECK_GT(indices_list.size(), 0);
  TORCH_CHECK_EQ(lengths_list.size(), indices_list.size());
  TORCH_CHECK_EQ(per_sample_weights.size(), indices_list.size());
  int64_t total_indices = 0;
  int64_t total_lengths = 0;
  bool need_weights = false;
  bool pin_memory = false;

  for (size_t i = 0; i < indices_list.size(); i++) {
    TORCH_CHECK(
        indices_list[i].dtype() == c10::kInt ||
        indices_list[i].dtype() == c10::kLong);
    TORCH_CHECK(
        lengths_list[i].dtype() == c10::kInt ||
        lengths_list[i].dtype() == c10::kLong);
    TORCH_CHECK_EQ(indices_list[i].ndimension(), 1);
    TORCH_CHECK_EQ(lengths_list[i].ndimension(), 1);
    TORCH_CHECK(indices_list[i].is_contiguous());
    TORCH_CHECK(lengths_list[i].is_contiguous());
    total_indices += indices_list[i].numel();
    total_lengths += lengths_list[i].numel();

    if (per_sample_weights[i].numel() > 0) {
      TORCH_CHECK_EQ(per_sample_weights[i].ndimension(), 1);
      TORCH_CHECK_EQ(per_sample_weights[i].numel(), indices_list[i].numel());
      TORCH_CHECK(per_sample_weights[i].is_contiguous());
      need_weights = true;
    }
  }

  auto combined_indices =
      _cat_int_tensors(indices_list, total_indices, pin_memory);

  auto combined_lengths =
      _cat_int_tensors(lengths_list, total_lengths, pin_memory);

  if (need_weights) {
    return {
        std::move(combined_indices),
        std::move(combined_lengths),
        _cat_per_sample_weights_list(
            per_sample_weights, indices_list, total_indices, pin_memory)};
  }
  return {combined_indices, combined_lengths, at::empty({0})};
}

// Similar to tbe_input_combine_cpu, but padding all the offsets
// to the size specified by batch_size.

std::tuple<Tensor, Tensor, Tensor> padding_fused_tbe_input_combine_cpu(
    const std::vector<Tensor>& indices_list,
    const std::vector<Tensor>& offsets_list,
    const std::vector<Tensor>& per_sample_weights,
    const Tensor& include_last_offsets,
    int64_t batch_size) {
  TORCH_CHECK_GT(indices_list.size(), 0);
  TORCH_CHECK_EQ(offsets_list.size(), indices_list.size());
  TORCH_CHECK_EQ(per_sample_weights.size(), indices_list.size());
  TORCH_CHECK(
      static_cast<uint64_t>(include_last_offsets.numel()) ==
      indices_list.size());
  auto include_last_offsets_acc = include_last_offsets.accessor<bool, 1>();
  int64_t total_indices = 0;
  int64_t total_offsets = 1 + batch_size * indices_list.size();
  bool need_weights = false;
  bool pin_memory = false;

  for (size_t i = 0; i < indices_list.size(); i++) {
    TORCH_CHECK(
        indices_list[i].dtype() == c10::kInt ||
        indices_list[i].dtype() == c10::kLong);
    TORCH_CHECK(
        offsets_list[i].dtype() == c10::kInt ||
        offsets_list[i].dtype() == c10::kLong);
    TORCH_CHECK_EQ(indices_list[i].ndimension(), 1);
    TORCH_CHECK_EQ(offsets_list[i].ndimension(), 1);
    TORCH_CHECK(indices_list[i].is_contiguous());
    TORCH_CHECK(offsets_list[i].is_contiguous());
    total_indices += indices_list[i].numel();

    if (per_sample_weights[i].numel() > 0) {
      TORCH_CHECK_EQ(per_sample_weights[i].ndimension(), 1);
      TORCH_CHECK_EQ(per_sample_weights[i].numel(), indices_list[i].numel());
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
          for (const auto j : c10::irange(1, offsets_size)) {
            combined_offsets_acc[offsets_acc_idx++] =
                offset + static_cast<int32_t>(offsets_acc[j]);
          }
          offset += static_cast<int32_t>(indices_list[i].numel());
          for (int64_t j = offsets_size; j <= batch_size; j++) {
            combined_offsets_acc[offsets_acc_idx++] = offset;
          }
        });
  }

  if (need_weights) {
    return {
        std::move(combined_indices),
        std::move(combined_offsets),
        _cat_per_sample_weights_list(
            per_sample_weights, indices_list, total_indices, pin_memory)};
  }
  return {combined_indices, combined_offsets, at::empty({0})};
}

/// padding_fused_tbe_input_combine_with_length_cpu is similar to
/// tbe_input_combine_with_length_cpu, but padding all the lengths to the size
/// specified by batch_size.
///
/// @param indices_list list of indices.
/// @param lengths_list list of lengths.
/// @param per_sample_weights list of per_sample_weights
/// @return tuple of combined indices, lengths, and per_sample_weights
std::tuple<Tensor, Tensor, Tensor>
padding_fused_tbe_input_combine_with_length_cpu(
    const std::vector<Tensor>& indices_list,
    const std::vector<Tensor>& lengths_list,
    const std::vector<Tensor>& per_sample_weights,
    int64_t batch_size) {
  TORCH_CHECK_GT(indices_list.size(), 0);
  TORCH_CHECK_EQ(lengths_list.size(), indices_list.size());
  TORCH_CHECK_EQ(per_sample_weights.size(), indices_list.size());
  int64_t total_indices = 0;
  int64_t total_lengths = batch_size * indices_list.size();
  bool need_weights = false;
  bool pin_memory = false;

  for (size_t i = 0; i < indices_list.size(); i++) {
    TORCH_CHECK(
        indices_list[i].dtype() == c10::kInt ||
        indices_list[i].dtype() == c10::kLong);
    TORCH_CHECK(
        lengths_list[i].dtype() == c10::kInt ||
        lengths_list[i].dtype() == c10::kLong);
    TORCH_CHECK_EQ(indices_list[i].ndimension(), 1);
    TORCH_CHECK_EQ(lengths_list[i].ndimension(), 1);
    TORCH_CHECK(indices_list[i].is_contiguous());
    TORCH_CHECK(lengths_list[i].is_contiguous());
    total_indices += indices_list[i].numel();

    if (per_sample_weights[i].numel() > 0) {
      TORCH_CHECK_EQ(per_sample_weights[i].ndimension(), 1);
      TORCH_CHECK_EQ(per_sample_weights[i].numel(), indices_list[i].numel());
      TORCH_CHECK(per_sample_weights[i].is_contiguous());
      need_weights = true;
    }
  }

  auto combined_indices =
      _cat_int_tensors(indices_list, total_indices, pin_memory);

  auto combined_lengths = _cat_int_tensors_with_padding(
      lengths_list, total_lengths, pin_memory, batch_size);

  if (need_weights) {
    return {
        std::move(combined_indices),
        std::move(combined_lengths),
        _cat_per_sample_weights_list(
            per_sample_weights, indices_list, total_indices, pin_memory)};
  }
  return {combined_indices, combined_lengths, at::empty({0})};
}

} // namespace fbgemm_gpu

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
#ifdef HAS_IMPL_ABSTRACT_PYSTUB
  m.impl_abstract_pystub(
      "fbgemm_gpu.sparse_ops",
      "//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_py");
#endif
  m.def(
      "tbe_input_combine(Tensor[] indices_list, Tensor[] offsets_list, Tensor[] per_sample_weights, Tensor include_last_offsets) -> (Tensor, Tensor, Tensor)",
      {PT2_COMPLIANT_TAG});
  m.def(
      "tbe_input_combine_with_length(Tensor[] indices_list, Tensor[] lengths_list, Tensor[] per_sample_weights) -> (Tensor, Tensor, Tensor)",
      {PT2_COMPLIANT_TAG});
  m.def(
      "padding_fused_tbe_input_combine(Tensor[] indices_list, Tensor[] offsets_list, Tensor[] per_sample_weights, Tensor include_last_offsets, int batch_size) -> (Tensor, Tensor, Tensor)");
  m.def(
      "padding_fused_tbe_input_combine_with_length(Tensor[] indices_list, Tensor[] lengths_list, Tensor[] per_sample_weights, int batch_size) -> (Tensor, Tensor, Tensor)");
  DISPATCH_TO_CPU("tbe_input_combine", fbgemm_gpu::tbe_input_combine_cpu);
  DISPATCH_TO_CPU(
      "tbe_input_combine_with_length",
      fbgemm_gpu::tbe_input_combine_with_length_cpu);
  DISPATCH_TO_CPU(
      "padding_fused_tbe_input_combine",
      fbgemm_gpu::padding_fused_tbe_input_combine_cpu);
  DISPATCH_TO_CPU(
      "padding_fused_tbe_input_combine_with_length",
      fbgemm_gpu::padding_fused_tbe_input_combine_with_length_cpu);
}
