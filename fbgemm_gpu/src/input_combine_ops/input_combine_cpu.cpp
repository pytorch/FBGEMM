/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/Functions.h>
#include <ATen/TypeDefault.h>
#include <ATen/core/op_registration/op_registration.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>
#include <torch/script.h>

#include "fbgemm_gpu/input_combine.h"
#include "fbgemm_gpu/utils/dispatch_macros.h"
#include "fbgemm_gpu/utils/ops_utils.h"
#include "fbgemm_gpu/utils/tensor_utils.h"

using Tensor = at::Tensor;

namespace fbgemm_gpu {

void _cat_int_tensors_out(
    Tensor& combined_tensors,
    const std::vector<Tensor>& tensor_list,
    int64_t total_num,
    bool to_trim_padding = false,
    const std::vector<int64_t>& indices_terminating_idx =
        std::vector<int64_t>()) {
  if (to_trim_padding) {
    // We need to define the teminating idx for each indices tensor
    TORCH_CHECK(indices_terminating_idx.size() == tensor_list.size());
  }
  at::native::resize_(combined_tensors, {total_num});
  auto* combined_tensors_data_ptr =
      combined_tensors.mutable_data_ptr<int32_t>();
  size_t idx = 0;

  // Let's keep the original paddings and later pad them in the end
  std::vector<int64_t> paddings;
  paddings.reserve(total_num);

  for (size_t i = 0; i < tensor_list.size(); ++i) {
    const auto& tensor = tensor_list[i];
    AT_DISPATCH_INDEX_TYPES(tensor.scalar_type(), "tbe_cat_inputs_", [&] {
      // Necessary to use data_ptr. Checked in caller, but let's
      // be safe in case somebody changes that.
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(tensor.is_contiguous());
      auto* indices_data_ptr = tensor.const_data_ptr<index_t>();
      auto numel = tensor.numel();
      if (to_trim_padding) {
        const auto terminating_idx = indices_terminating_idx.at(i);
        numel = terminating_idx > 0 && terminating_idx < numel ? terminating_idx
                                                               : numel;
      }
      size_t j = 0;
      for (; j < numel; j++) {
        combined_tensors_data_ptr[idx++] =
            static_cast<int32_t>(indices_data_ptr[j]);
      }
      for (; j < tensor.numel(); j++) {
        paddings.push_back(indices_data_ptr[j]);
      }
    });
  }

  // Pad the original paddings in the end
  int i = 0;
  while (idx < total_num) {
    if (i < paddings.size()) [[likely]] {
      combined_tensors_data_ptr[idx++] = paddings[i++];
    } else {
      combined_tensors_data_ptr[idx++] = 0;
    }
  }
}

Tensor _cat_int_tensors(
    const std::vector<Tensor>& tensor_list,
    int64_t total_num,
    bool use_pin_memory,
    bool to_trim_padding = false,
    const std::vector<int64_t>& indices_terminating_idx =
        std::vector<int64_t>()) {
  // Using int type to maintain original behavior
  // in https://fburl.com/code/h2lwews2
  auto combined_tensors = at::empty(
      {total_num},
      at::TensorOptions()
          .dtype(c10::kInt)
          .device(tensor_list[0].device())
          .pinned_memory(use_pin_memory));

  _cat_int_tensors_out(
      combined_tensors,
      tensor_list,
      total_num,
      to_trim_padding,
      indices_terminating_idx);
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

  auto* combined_tensors_data_ptr =
      combined_tensors.mutable_data_ptr<int32_t>();

  for (const auto i : c10::irange(tensor_list.size())) {
    size_t idx = i * batch_size;
    const auto& tensor = tensor_list[i];
    AT_DISPATCH_INDEX_TYPES(tensor.scalar_type(), "tbe_cat_inputs_", [&] {
      // Necessary to use data_ptr. Checked in caller, but let's
      // be safe in case somebody changes that.
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(tensor.is_contiguous());
      auto indices_data_ptr = tensor.const_data_ptr<index_t>();
      const auto numel = tensor.numel();
      for (const auto j : c10::irange(numel)) {
        combined_tensors_data_ptr[idx++] =
            static_cast<int32_t>(indices_data_ptr[j]);
      }
    });
  }
  return combined_tensors;
}

void _cat_per_sample_weights_list_out(
    Tensor& out,
    const std::vector<Tensor>& per_sample_weights,
    const std::vector<Tensor>& indices_list,
    int64_t total_num,
    bool to_trim_padding = false,
    const std::vector<int64_t>& indices_terminating_idx =
        std::vector<int64_t>()) {
  if (to_trim_padding) {
    // We need to define the teminating idx for each indices tensor
    TORCH_CHECK(indices_terminating_idx.size() == indices_list.size());
  }
  at::native::resize_(out, {total_num});
  out.fill_(1.);

  auto* out_weights_ptr = out.mutable_data_ptr<float>();

  for (const auto i : c10::irange(per_sample_weights.size())) {
    auto element_size = per_sample_weights[i].numel();
    auto actual_indices_size = indices_list[i].numel();
    if (to_trim_padding) {
      element_size = element_size > indices_terminating_idx.at(i)
          ? indices_terminating_idx.at(i)
          : element_size;
      actual_indices_size = actual_indices_size > indices_terminating_idx.at(i)
          ? indices_terminating_idx.at(i)
          : actual_indices_size;
    }
    if (element_size != 0) {
      memcpy(
          out_weights_ptr,
          per_sample_weights[i].data_ptr<float>(),
          element_size * sizeof(float));
    }
    out_weights_ptr += actual_indices_size;
  }
}

Tensor _cat_per_sample_weights_list(
    const std::vector<Tensor>& per_sample_weights,
    const std::vector<Tensor>& indices_list,
    int64_t total_num,
    bool use_pin_memory,
    bool to_trim_padding = false,
    const std::vector<int64_t>& indices_terminating_idx =
        std::vector<int64_t>()) {
  auto combined_weights = at::empty(
      {0},
      at::TensorOptions()
          .dtype(c10::kFloat)
          .device(per_sample_weights[0].device())
          .pinned_memory(use_pin_memory));
  _cat_per_sample_weights_list_out(
      combined_weights,
      per_sample_weights,
      indices_list,
      total_num,
      to_trim_padding,
      indices_terminating_idx);
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

  // We only enable this feature when all the elements of `include_last_offsets`
  // are True and there is at least one indices tensor has paddings
  bool indices_tensor_has_padding = false;
  for (size_t i = 0; i < indices_list.size(); i++) {
    if (indices_list[i].numel() > offsets_list[i][-1].item().toLong()) {
      indices_tensor_has_padding = true;
      break;
    }
  }
  auto to_trim_padding =
      indices_tensor_has_padding && include_last_offsets.all().item<bool>();
  // In case of index tensors have padding, we need to determine the boundary
  // i.e. the terminating idx, to properly combine the TBE inputs
  // `indices_terminating_idx` is a list of the terminating idx for each index
  // tensor
  std::vector<int64_t> indices_terminating_idx;
  indices_terminating_idx.reserve(indices_list.size());

  for (const auto i : c10::irange(indices_list.size())) {
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
    if (to_trim_padding) {
      // When the offsets tensor has last offset, we respect this value
      // And the last offset value should be less than (in case there are
      // paddings) or equal to the number of elements in the indices tensor
      TORCH_CHECK_LE(
          offsets_list[i][-1].item().toLong(), indices_list[i].numel());
      indices_terminating_idx.push_back(offsets_list[i][-1].item().toLong());
    }
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

  auto combined_indices = _cat_int_tensors(
      indices_list,
      total_indices,
      pin_memory,
      to_trim_padding,
      indices_terminating_idx);

  auto combined_offsets = at::empty(
      {total_offsets},
      at::TensorOptions()
          .dtype(c10::kInt)
          .device(offsets_list[0].device())
          .pinned_memory(pin_memory));

  auto combined_offsets_data_ptr = combined_offsets.mutable_data_ptr<int32_t>();
  int32_t offset = 0;
  size_t offsets_acc_idx = 0;
  combined_offsets_data_ptr[offsets_acc_idx++] = 0;

  for (const auto i : c10::irange(offsets_list.size())) {
    AT_DISPATCH_INDEX_TYPES(
        offsets_list[i].scalar_type(), "tbe_input_offsets_", [&] {
          // TORCH_CHECKed to be contiguous above, so data_ptr is safe.
          auto offsets_data_ptr = offsets_list[i].const_data_ptr<index_t>();
          for (int64_t j = 1,
                       size = offsets_list[i].numel() -
                   (include_last_offsets_acc[i] ? 1 : 0);
               j < size;
               j++) {
            combined_offsets_data_ptr[offsets_acc_idx++] =
                offset + static_cast<int32_t>(offsets_data_ptr[j]);
          }

          if (to_trim_padding) {
            offset += static_cast<int32_t>(offsets_list[i][-1].item().toInt());
          } else {
            offset += static_cast<int32_t>(indices_list[i].numel());
          }
          combined_offsets_data_ptr[offsets_acc_idx++] = offset;
        });
  }

  if (need_weights) {
    return {
        combined_indices,
        combined_offsets,
        _cat_per_sample_weights_list(
            per_sample_weights,
            indices_list,
            total_indices,
            pin_memory,
            to_trim_padding,
            indices_terminating_idx)};
  }
  return {combined_indices, combined_offsets, at::empty({0})};
}

void tbe_input_combine_with_length_cpu_out(
    Tensor& combined_indices,
    Tensor& combined_lengths,
    Tensor& combined_per_sample_weights,
    const std::vector<Tensor>& indices_list,
    const std::vector<Tensor>& lengths_list,
    const std::vector<Tensor>& per_sample_weights) {
  TORCH_CHECK_GT(indices_list.size(), 0);
  TORCH_CHECK_EQ(lengths_list.size(), indices_list.size());
  TORCH_CHECK_EQ(per_sample_weights.size(), indices_list.size());
  int64_t total_indices = 0;
  int64_t total_lengths = 0;
  bool need_weights = false;

  for (const auto i : c10::irange(indices_list.size())) {
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

  _cat_int_tensors_out(combined_indices, indices_list, total_indices);
  _cat_int_tensors_out(combined_lengths, lengths_list, total_lengths);
  if (need_weights) {
    _cat_per_sample_weights_list_out(
        combined_per_sample_weights,
        per_sample_weights,
        indices_list,
        total_indices);
    return;
  }
  combined_per_sample_weights.resize_({0});
}

std::tuple<Tensor, Tensor, Tensor> tbe_input_combine_with_length_cpu(
    const std::vector<Tensor>& indices_list,
    const std::vector<Tensor>& lengths_list,
    const std::vector<Tensor>& per_sample_weights) {
  constexpr bool pin_memory = false;
  const auto num_lists = indices_list.size();
  TORCH_CHECK_GT(indices_list.size(), 0);
  TORCH_CHECK_EQ(lengths_list.size(), indices_list.size());
  TORCH_CHECK_EQ(per_sample_weights.size(), indices_list.size());
  for (const auto i : c10::irange(num_lists)) {
    TENSOR_CONTIGUOUS_AND_ON_CPU(indices_list[i]);
    TENSOR_CONTIGUOUS_AND_ON_CPU(lengths_list[i]);
    if (per_sample_weights[i].numel() > 0) {
      TENSOR_CONTIGUOUS_AND_ON_CPU(per_sample_weights[i]);
    } else {
      TENSOR_EMPTY_OR_ON_CPU(per_sample_weights[i]);
    }
  }
  // Using int type to maintain original behavior
  // in https://fburl.com/code/h2lwews2
  auto combined_indices = at::empty(
      {0},
      at::TensorOptions()
          .dtype(c10::kInt)
          .device(indices_list[0].device())
          .pinned_memory(pin_memory));
  auto combined_lengths = at::empty(
      {0},
      at::TensorOptions()
          .dtype(c10::kInt)
          .device(lengths_list[0].device())
          .pinned_memory(pin_memory));
  // Using float type to maintain original behavior
  // in https://fburl.com/code/lp6u8j81
  auto combined_per_sample_weights = at::empty(
      {0},
      at::TensorOptions()
          .dtype(c10::kFloat)
          .device(per_sample_weights[0].device())
          .pinned_memory(pin_memory));
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

  for (const auto i : c10::irange(indices_list.size())) {
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

  auto combined_offsets_data_ptr = combined_offsets.mutable_data_ptr<int32_t>();
  int32_t offset = 0;
  size_t offsets_acc_idx = 0;
  combined_offsets_data_ptr[offsets_acc_idx++] = 0;

  for (const auto i : c10::irange(offsets_list.size())) {
    AT_DISPATCH_INDEX_TYPES(
        offsets_list[i].scalar_type(), "tbe_input_offsets_", [&] {
          // TORCH_CHECKed to be contiguous above, so data_ptr is safe.
          auto* offsets_data_ptr = offsets_list[i].const_data_ptr<index_t>();
          int64_t offsets_size =
              offsets_list[i].numel() - (include_last_offsets_acc[i] ? 1 : 0);
          for (const auto j : c10::irange(1, offsets_size)) {
            combined_offsets_data_ptr[offsets_acc_idx++] =
                offset + static_cast<int32_t>(offsets_data_ptr[j]);
          }
          offset += static_cast<int32_t>(indices_list[i].numel());
          for (int64_t j = offsets_size; j <= batch_size; j++) {
            combined_offsets_data_ptr[offsets_acc_idx++] = offset;
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

  for (const auto i : c10::irange(indices_list.size())) {
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

  auto combined_per_sample_weights = need_weights
      ? _cat_per_sample_weights_list(
            per_sample_weights, indices_list, total_indices, pin_memory)
      : at::empty({0});

  return {combined_indices, combined_lengths, combined_per_sample_weights};
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
