/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <torch/torch.h>

#include "fbgemm_gpu/intraining_embedding_pruning.h"

using Tensor = at::Tensor;

namespace fbgemm_gpu {

/**
 * CPU implementation for remap_indices_update_utils
 *
 * This function remaps indices using an address_lookup table and optionally
 * updates row utilization statistics.
 *
 * @param iter Iteration counter for determining when to update row utils
 * @param buffer_idx CPU tensor indicating which table each feature belongs to
 * @param feature_lengths CPU tensor with length of each feature
 * @param feature_offsets CPU tensor with starting offset of each feature in
 * values
 * @param values Indices to remap (modified in-place)
 * @param address_lookup Lookup table for remapping indices
 * @param row_util Row utilization statistics (modified in-place if update_util)
 * @param buffer_offsets Starting offset for each table in address_lookup
 * @param full_values_list Optional per-feature value lists for GDT mode
 * @param update_util Optional flag to force enable/disable util updates
 *
 * @return Modified values tensor
 */
Tensor remap_indices_update_utils_cpu(
    const int64_t iter,
    const Tensor& buffer_idx,
    const Tensor& feature_lengths,
    const Tensor& feature_offsets,
    const Tensor& values,
    const Tensor& address_lookup,
    Tensor& row_util,
    const Tensor& buffer_offsets,
    const std::optional<std::vector<Tensor>>& full_values_list,
    const std::optional<bool>& update_util) {
  // Validate inputs
  TORCH_CHECK(buffer_idx.device().is_cpu(), "buffer_idx must be on CPU");
  TORCH_CHECK(
      feature_lengths.device().is_cpu(), "feature_lengths must be on CPU");
  TORCH_CHECK(
      feature_offsets.device().is_cpu(), "feature_offsets must be on CPU");

  const int32_t num_tables = buffer_offsets.size(0) - 1;
  if (num_tables <= 0) {
    return values;
  }

  const int32_t num_indices = values.size(0);
  if (num_indices <= 0) {
    return values;
  }

  const auto buffer_idx_a = buffer_idx.accessor<int32_t, 1>();
  const auto feature_lengths_a = feature_lengths.accessor<int64_t, 1>();
  const auto feature_offsets_a = feature_offsets.accessor<int64_t, 1>();
  const auto buffer_offsets_a = buffer_offsets.accessor<int64_t, 1>();

  auto row_util_a = row_util.accessor<float, 1>();

  const int32_t num_features = feature_lengths.numel();
  const bool use_gdt = full_values_list.has_value();
  const bool update_util_value = update_util.has_value()
      ? update_util.value()
      : ((iter < 10) || (iter < 100 && (iter + 1) % 19 == 0) ||
         ((iter + 1) % 39 == 0));

  AT_DISPATCH_INDEX_TYPES(
      values.scalar_type(), "remap_indices_update_utils_cpu", [&] {
        auto values_a = values.accessor<index_t, 1>();
        // address_lookup is always int64 to match GPU/CUDA implementation
        const auto address_lookup_a = address_lookup.accessor<int64_t, 1>();

        for (int32_t i = 0; i < num_features; i++) {
          const auto start = feature_offsets_a[i];
          const auto length = feature_lengths_a[i];

          if (length == 0) {
            continue;
          }

          const int32_t buf_idx = buffer_idx_a[i];
          const int64_t buffer_offset = buffer_offsets_a[buf_idx];

          if (update_util_value) {
            // Get the full_values tensor for this feature (or use values if not
            // GDT)
            Tensor full_values_tensor;
            int64_t full_start, full_length;

            if (use_gdt) {
              full_values_tensor = full_values_list.value()[i];
              full_start = 0;
              full_length = full_values_tensor.numel();
            } else {
              full_values_tensor = values;
              full_start = start;
              full_length = length;
            }

            // Extract the slice for this feature
            auto feature_slice = full_values_tensor.slice(
                0, full_start, full_start + full_length);

            // Sort the indices to count frequencies
            auto values_sorted = feature_slice.clone();
            std::sort(
                values_sorted.data_ptr<index_t>(),
                values_sorted.data_ptr<index_t>() + full_length);

            // Count frequencies using run-length encoding logic
            if (full_length > 0) {
              const auto sorted_data = values_sorted.accessor<index_t, 1>();

              index_t current_val = sorted_data[0];
              int32_t current_count = 1;

              auto process_unique_value = [&](index_t val, int32_t count) {
                const int64_t row_util_idx = buffer_offset + val;
                row_util_a[row_util_idx] += count;
              };

              for (int64_t j = 1; j < full_length; j++) {
                if (sorted_data[j] == current_val) {
                  current_count++;
                } else {
                  process_unique_value(current_val, current_count);
                  current_val = sorted_data[j];
                  current_count = 1;
                }
              }
              // Process the last run
              process_unique_value(current_val, current_count);
            }
          }

          // Remap indices using address_lookup
          for (int64_t j = 0; j < length; j++) {
            const int64_t idx = start + j;
            const index_t original_val = values_a[idx];
            const int64_t address_lookup_idx = buffer_offset + original_val;
            values_a[idx] = address_lookup_a[address_lookup_idx];
          }
        }
      });

  return values;
}

} // namespace fbgemm_gpu
