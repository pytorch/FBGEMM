/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/TypeDefault.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/script.h>
#include "fbgemm_gpu/dispatch_macros.h"
#include "fbgemm_gpu/embedding_common.h"
#include "fbgemm_gpu/sparse_ops_utils.h"

using Tensor = at::Tensor;
using namespace fbgemm_gpu;

/// @defgroup embedding-cpu Embedding CPU Operators
///

namespace {

template <typename index_t>
void adjust_offset_cpu(
    index_t& indices_start,
    index_t& indices_end,
    int64_t num_indices,
    index_t* offsets_acc_start,
    index_t* offsets_acc_end) {
  indices_start =
      std::max(0L, std::min(static_cast<int64_t>(indices_start), num_indices));
  indices_end = std::max(
      static_cast<int64_t>(indices_start),
      std::min(static_cast<int64_t>(indices_end), num_indices));
  *offsets_acc_start = indices_start;
  *offsets_acc_end = indices_end;
}

///@addtogroup embedding-cpu
void bounds_check_indices_cpu(
    Tensor& rows_per_table,
    Tensor& indices,
    Tensor& offsets,
    int64_t bounds_check_mode_,
    Tensor& warning,
    const c10::optional<Tensor>& weights,
    const c10::optional<Tensor>& B_offsets,
    const int64_t /*max_B*/) {
  TORCH_CHECK(
      !B_offsets.has_value(),
      "bounds_check_indices on CPU does not support variable length (batch size)");
  auto bounds_check_mode = static_cast<BoundsCheckMode>(bounds_check_mode_);
  if (bounds_check_mode == BoundsCheckMode::WARNING) {
    warning.zero_();
  }

  int32_t T = rows_per_table.size(0);
  int32_t B = (offsets.size(0) - 1) / T;
  const auto rows_per_table_acc = rows_per_table.accessor<int64_t, 1>();
  auto warning_acc = warning.data_ptr<int64_t>();

  AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "bounds_check_indices", [&] {
    auto offsets_acc = offsets.accessor<index_t, 1>();
    auto indices_acc = indices.accessor<index_t, 1>();
    auto num_indices = indices.numel();

    TORCH_CHECK(
        offsets.size(0) == B * T + 1,
        "offsets size " + std::to_string(offsets.size(0)) +
            " is not equal to B (" + std::to_string(B) + ") * T (" +
            std::to_string(T) + ") + 1");
    if (weights.has_value()) {
      TORCH_CHECK(
          weights.value().size(0) == num_indices,
          "weights size " + std::to_string(weights.value().size(0)) +
              " is not equal to indices size " + std::to_string(num_indices));
    }

    if (bounds_check_mode == BoundsCheckMode::FATAL) {
      TORCH_CHECK(num_indices == offsets_acc[B * T]);
    } else if (bounds_check_mode == BoundsCheckMode::WARNING) {
      if (num_indices != offsets_acc[B * T]) {
        if (__sync_fetch_and_add(&warning_acc[0], 1) == 0) {
          LOG(ERROR)
              << "The last element in offsets is incorrect for "
              << "total batch size B: " << B << ", total table num T: " << T
              << ", last element in offsets: " << offsets_acc[B * T]
              << ", indices size: " << num_indices
              << ". Setting the last element in offsets to be indices size.";
        }
        offsets_acc[B * T] = num_indices;
      }
    } else if (bounds_check_mode == BoundsCheckMode::IGNORE) {
      if (num_indices != offsets_acc[B * T]) {
        offsets_acc[B * T] = num_indices;
      }
    }
    for (const auto t : c10::irange(T)) {
      auto num_rows = rows_per_table_acc[t];
      for (const auto b : c10::irange(B)) {
        auto indices_start = offsets_acc[t * B + b];
        auto indices_end = offsets_acc[t * B + b + 1];
        if (bounds_check_mode == BoundsCheckMode::FATAL) {
          TORCH_CHECK(indices_start >= 0);
          TORCH_CHECK(indices_start <= indices_end);
          TORCH_CHECK(indices_end <= num_indices);
        } else if (bounds_check_mode == BoundsCheckMode::WARNING) {
          if (indices_start < 0 || indices_start > indices_end ||
              indices_end > num_indices) {
            if (__sync_fetch_and_add(&warning_acc[0], 1) == 0) {
              LOG(ERROR)
                  << "(at least one) Out of bounds access for batch: " << b
                  << ", table: " << t << ", indices_start: " << indices_start
                  << ", indices_end: " << indices_end
                  << ", num_indices: " << num_indices
                  << ". Setting indices_start and indices_end within the range";
            }
            adjust_offset_cpu(
                indices_start,
                indices_end,
                num_indices,
                &offsets_acc[t * B + b],
                &offsets_acc[t * B + b + 1]);
          }
        } else if (bounds_check_mode == BoundsCheckMode::IGNORE) {
          adjust_offset_cpu(
              indices_start,
              indices_end,
              num_indices,
              &offsets_acc[t * B + b],
              &offsets_acc[t * B + b + 1]);
        }

        auto L = indices_end - indices_start;
        for (const auto l : c10::irange(L)) {
          auto idx = indices_acc[indices_start + l];
          if (idx == -1) {
            // -1 indicates pruned rows.
            continue;
          }
          if (bounds_check_mode == BoundsCheckMode::FATAL) {
            TORCH_CHECK(idx >= 0);
            TORCH_CHECK(idx < num_rows);
          } else if (bounds_check_mode == BoundsCheckMode::WARNING) {
            if (idx < 0 || idx >= num_rows) {
              if (__sync_fetch_and_add(&warning_acc[0], 1) == 0) {
                LOG(ERROR) << "(at least one) Out of bounds access for batch: "
                           << b << ", table: " << t << ", bag element: " << l
                           << ", idx: " << idx << ", num_rows: " << num_rows
                           << ". Setting idx to zero.";
              }
              indices_acc[indices_start + l] = 0;
            }
          } else if (bounds_check_mode == BoundsCheckMode::IGNORE) {
            if (idx < 0 || idx >= num_rows) {
              indices_acc[indices_start + l] = 0;
            }
          }
        }
      }
    }
  });
}
} // namespace

// Deprecated for fb namespace! Please use fbgemm namespace instead!
TORCH_LIBRARY_FRAGMENT(fb, m) {
  // The (a!) tells PyTorch this is an impure operation and so cannot be CSE'd
  // or DCE'd, etc.
  m.def(
      "bounds_check_indices(Tensor rows_per_table, Tensor(a!) indices, Tensor(b!) offsets, int bounds_check_mode, Tensor(c!) warning, Tensor(d!)? weights=None, Tensor? B_offsets=None, int max_B=-1) -> ()",
      {PT2_COMPLIANT_TAG});
  DISPATCH_TO_CPU("bounds_check_indices", bounds_check_indices_cpu);
}

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  // The (a!) tells PyTorch this is an impure operation and so cannot be CSE'd
  // or DCE'd, etc.
  m.def(
      "bounds_check_indices(Tensor rows_per_table, Tensor(a!) indices, Tensor(b!) offsets, int bounds_check_mode, Tensor(c!) warning, Tensor(d!)? weights=None, Tensor? B_offsets=None, int max_B=-1) -> ()",
      {PT2_COMPLIANT_TAG});
  DISPATCH_TO_CPU("bounds_check_indices", bounds_check_indices_cpu);
}
