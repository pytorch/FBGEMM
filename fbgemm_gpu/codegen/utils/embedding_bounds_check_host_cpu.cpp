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

#include <atomic>
#include <chrono>
#include <optional>
#include <string>

#ifdef FBGEMM_FBCODE
#include <folly/synchronization/AtomicRef.h>
#endif

#include "fbgemm_gpu/embedding_common.h"
#include "fbgemm_gpu/utils/dispatch_macros.h"
#include "fbgemm_gpu/utils/ops_utils.h"
#include "fbgemm_gpu/utils/tensor_accessor_builder.h"
#include "fbgemm_gpu/utils/tensor_utils.h"

#if FBGEMM_GPU_MEMCHECK
#define FBGEMM_MEM_CHECK_ONLY
#else
#define FBGEMM_MEM_CHECK_ONLY maybe_unused
#endif

using Tensor = at::Tensor;
using namespace fbgemm_gpu;

/// @defgroup embedding-cpu Embedding CPU Operators
///

namespace {

#ifdef FBGEMM_FBCODE
template <typename T>
using atomic_ref = folly::atomic_ref<T>;
#else
template <typename T>
using atomic_ref = std::atomic_ref<T>;
#endif

// The warning tensor is reset on every call, so it cannot limit logging across
// calls; this process-wide limiter does.
class BoundsCheckLogLimiter {
 public:
  // Returns the number of warnings suppressed since the previous log if the
  // caller should log now, or std::nullopt if this warning is suppressed.
  std::optional<int64_t> try_acquire() {
    const int64_t now_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count();
    int64_t next_log_time_ns =
        next_log_time_ns_.load(std::memory_order_relaxed);
    if (now_ns < next_log_time_ns ||
        !next_log_time_ns_.compare_exchange_strong(
            next_log_time_ns,
            now_ns + kLogIntervalNs,
            std::memory_order_relaxed)) {
      num_suppressed_.fetch_add(1, std::memory_order_relaxed);
      return std::nullopt;
    }
    return num_suppressed_.exchange(0, std::memory_order_relaxed);
  }

 private:
  static constexpr int64_t kLogIntervalNs = 10'000'000'000;
  std::atomic<int64_t> next_log_time_ns_{0};
  std::atomic<int64_t> num_suppressed_{0};
};

// Namespace scope rather than function-local statics: the log sites are inside
// AT_DISPATCH_INDEX_TYPES, which would create one limiter per index type.
BoundsCheckLogLimiter final_offset_log_limiter;
BoundsCheckLogLimiter offsets_log_limiter;
BoundsCheckLogLimiter indices_log_limiter;

std::string suppressed_warnings_suffix(int64_t num_suppressed) {
  if (num_suppressed == 0) {
    return "";
  }
  return " (" + std::to_string(num_suppressed) +
      " similar warnings suppressed since the last one)";
}

// Records the warning and returns the number of suppressed warnings if the
// caller should log it. Only the first warning of a call is a log candidate.
std::optional<int64_t> record_warning(
    int64_t& warning,
    BoundsCheckLogLimiter& log_limiter) {
  const auto previous_warnings =
      atomic_ref<int64_t>(warning).fetch_add(1, std::memory_order_relaxed);
  if (previous_warnings != 0) {
    return std::nullopt;
  }
  return log_limiter.try_acquire();
}

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

void check_weights_dim_matches_indices(
    const std::optional<Tensor>& weights,
    int64_t num_indices) {
  if (weights.has_value() && weights->numel() != 0) {
    TORCH_CHECK(
        weights.value().size(0) == num_indices,
        "weights size " + std::to_string(weights.value().size(0)) +
            " is not equal to indices size " + std::to_string(num_indices));
  }
}

///@addtogroup embedding-cpu
void bounds_check_indices_cpu(
    Tensor& rows_per_table,
    Tensor& indices,
    Tensor& offsets,
    int64_t bounds_check_mode_,
    Tensor& warning,
    const std::optional<Tensor>& weights,
    const std::optional<Tensor>& B_offsets,
    const int64_t max_B,
    const std::optional<Tensor>& /*b_t_map*/,
    const int64_t /*info_B_num_bits*/,
    const int64_t /*info_B_mask*/,
    const int8_t /*bounds_check_version*/,
    const bool /*prefetch_pipeline*/) {
  if (offsets.scalar_type() != indices.scalar_type()) {
    offsets = offsets.to(indices.scalar_type());
  }
  const auto vbe = B_offsets.has_value();
  if (vbe) {
    TENSOR_NDIM_EQUALS(B_offsets.value(), 1);
  }
  const int32_t* const B_offsets_ptr =
      vbe ? B_offsets.value().const_data_ptr<int32_t>() : nullptr;

  auto bounds_check_mode = static_cast<BoundsCheckMode>(bounds_check_mode_);
  const bool is_warning_mode = bounds_check_mode == BoundsCheckMode::WARNING ||
      bounds_check_mode == BoundsCheckMode::WARNING_ALLOW_TRAILING_INDICES;
  if (is_warning_mode) {
    warning.zero_();
  }

  const int32_t T = rows_per_table.size(0);
  const int32_t total_B = offsets.size(0) - 1;
  const int32_t B = total_B / T;
  const auto rows_per_table_acc = rows_per_table.accessor<int64_t, 1>();
  auto warning_acc = warning.mutable_data_ptr<int64_t>();

  AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "bounds_check_indices_cpu", [&] {
    [[FBGEMM_MEM_CHECK_ONLY]] const auto func_name = "bounds_check_indices_cpu";
    auto indices_acc = MAKE_TA_WITH_NAME(func_name, indices, index_t, 1);
    auto offsets_acc = MAKE_TA_WITH_NAME(func_name, offsets, index_t, 1);
    const auto num_indices = indices.numel();

    if (vbe) {
      TORCH_CHECK(max_B >= 0);
    } else {
      TORCH_CHECK(
          offsets.size(0) == B * T + 1,
          "offsets size " + std::to_string(offsets.size(0)) +
              " is not equal to B (" + std::to_string(B) + ") * T (" +
              std::to_string(T) + ") + 1");
    }
    check_weights_dim_matches_indices(weights, num_indices);

    const auto final_offset = offsets_acc[total_B];
    const bool invalid_final_offset = final_offset > num_indices ||
        (bounds_check_mode != BoundsCheckMode::WARNING_ALLOW_TRAILING_INDICES &&
         final_offset != num_indices);
    if (bounds_check_mode == BoundsCheckMode::FATAL) {
      TORCH_CHECK(!invalid_final_offset);
    } else if (is_warning_mode) {
      if (invalid_final_offset) {
        if (const auto num_suppressed =
                record_warning(warning_acc[0], final_offset_log_limiter)) {
          LOG(ERROR)
              << "EmbeddingBoundsCheck (VBE " << (vbe ? "true" : "false")
              << "): The last element in offsets is incorrect for "
              << "total batch size " << (vbe ? "total_B" : "B") << ": "
              << (vbe ? total_B : B) << ", total table num T: " << T
              << ", last element in offsets: " << offsets_acc[total_B]
              << ", indices size: " << num_indices
              << ". Setting the last element in offsets to be indices size."
              << suppressed_warnings_suffix(*num_suppressed);
        }
        offsets_acc[total_B] = num_indices;
      }
    } else if (bounds_check_mode == BoundsCheckMode::IGNORE) {
      if (invalid_final_offset) {
        offsets_acc[total_B] = num_indices;
      }
    }
    for (const auto t : c10::irange(T)) {
      auto num_rows = rows_per_table_acc[t];
      auto B_begin = vbe ? B_offsets_ptr[t] : t * B;
      auto B_end = vbe ? B_offsets_ptr[t + 1] : (t + 1) * B;
      for (const auto b : c10::irange(B_end - B_begin)) {
        auto indices_start = offsets_acc[B_begin + b];
        auto indices_end = offsets_acc[B_begin + b + 1];
        if (bounds_check_mode == BoundsCheckMode::FATAL) {
          TORCH_CHECK(indices_start >= 0);
          TORCH_CHECK(indices_start <= indices_end);
          TORCH_CHECK(indices_end <= num_indices);
        } else if (is_warning_mode) {
          if (indices_start < 0 || indices_start > indices_end ||
              indices_end > num_indices) {
            if (const auto num_suppressed =
                    record_warning(warning_acc[0], offsets_log_limiter)) {
              LOG(ERROR)
                  << "EmbeddingBoundsCheck (VBE " << (vbe ? "true" : "false")
                  << "): (at least one) Out of bounds access for batch: " << b
                  << ", table: " << t << ", indices_start: " << indices_start
                  << ", indices_end: " << indices_end
                  << ", num_indices: " << num_indices
                  << ". Setting indices_start and indices_end within the range"
                  << suppressed_warnings_suffix(*num_suppressed);
            }
            adjust_offset_cpu(
                indices_start,
                indices_end,
                num_indices,
                &offsets_acc[B_begin + b],
                &offsets_acc[B_begin + b + 1]);
          }
        } else if (bounds_check_mode == BoundsCheckMode::IGNORE) {
          adjust_offset_cpu(
              indices_start,
              indices_end,
              num_indices,
              &offsets_acc[B_begin + b],
              &offsets_acc[B_begin + b + 1]);
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
          } else if (is_warning_mode) {
            if (idx < 0 || idx >= num_rows) {
              if (const auto num_suppressed =
                      record_warning(warning_acc[0], indices_log_limiter)) {
                LOG(ERROR)
                    << "EmbeddingBoundsCheck (VBE " << (vbe ? "true" : "false")
                    << "): (at least one) Out of bounds access for batch: " << b
                    << ", table: " << t << ", bag element: " << l
                    << ", idx: " << idx << ", num_rows: " << num_rows
                    << ". Setting idx to zero."
                    << suppressed_warnings_suffix(*num_suppressed);
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
      "bounds_check_indices("
      "    Tensor rows_per_table, "
      "    Tensor(a!) indices, "
      "    Tensor(b!) offsets, "
      "    int bounds_check_mode, "
      "    Tensor(c!) warning, "
      "    Tensor(d!)? weights=None, "
      "    Tensor? B_offsets=None, "
      "    SymInt max_B=-1, "
      "    Tensor? b_t_map=None, "
      "    int info_B_num_bits=-1, "
      "    int info_B_mask=-1, "
      "    int bounds_check_version=1, "
      "    bool prefetch_pipeline=False"
      ") -> ()",
      {PT2_COMPLIANT_TAG});
  DISPATCH_TO_CPU("bounds_check_indices", bounds_check_indices_cpu);
}

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  // The (a!) tells PyTorch this is an impure operation and so cannot be CSE'd
  // or DCE'd, etc.
  m.impl_abstract_pystub(
      "fbgemm_gpu.sparse_ops",
      "//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_py");
  m.def(
      "bounds_check_indices("
      "    Tensor rows_per_table, "
      "    Tensor(a!) indices, "
      "    Tensor(b!) offsets, "
      "    int bounds_check_mode, "
      "    Tensor(c!) warning, "
      "    Tensor(d!)? weights=None, "
      "    Tensor? B_offsets=None, "
      "    SymInt max_B=-1, "
      "    Tensor? b_t_map=None, "
      "    int info_B_num_bits=-1, "
      "    int info_B_mask=-1, "
      "    int bounds_check_version=1, "
      "    bool prefetch_pipeline=False"
      ") -> ()",
      {PT2_COMPLIANT_TAG});
  DISPATCH_TO_CPU("bounds_check_indices", bounds_check_indices_cpu);
}
