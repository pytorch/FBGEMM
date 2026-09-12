/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>
#include <c10/macros/Macros.h>
#include <cstdint>

namespace fbgemm_gpu {

/// A runtime-typed view over global weight decay's `prev_iter` buffer.
///
/// `prev_iter` records the iteration at which each embedding row was last
/// touched. It is float32 by default and int64 when the frontend opts in (see
/// `GlobalWeightDecayDefinition.use_int64_prev_iter`). The Python frontend and
/// the CUDA backend ship in separately-versioned packages that roll out at
/// different times, so the kernels have to accept either dtype. Selecting the
/// dtype here instead of templating the kernels on it keeps the number of GWD
/// kernel instantiations flat.
///
/// Exactly one of `data_f32` / `data_i64` is valid, named by `is_int64`.
struct PrevIterRef {
  const float* __restrict__ data_f32;
  int64_t* __restrict__ data_i64;
  int64_t size;
  bool is_int64;

  /// A view starting `n` rows in, for indexing within a single table.
  __device__ inline PrevIterRef offset(const int64_t n) const {
    return PrevIterRef{
        data_f32 != nullptr ? data_f32 + n : nullptr,
        data_i64 != nullptr ? data_i64 + n : nullptr,
        size - n,
        is_int64};
  }

  /// The global weight decay multiplier for row `idx` at iteration `iter`.
  /// `prev_iter == 0` is the first-touch sentinel and yields 1 (no decay).
  ///
  /// NOTE: the float32 branch must stay bit-identical to the behavior that
  /// predates int64 support. It computes the gap in float, which rounds `iter`
  /// once it exceeds 2^24 -- that rounding is precisely the bug int64 fixes,
  /// so reproducing it here is what makes opting into int64 the only thing
  /// that can change a model's results.
  __device__ inline float gwd(
      const int64_t idx,
      const int64_t iter,
      const float weight_decay_base,
      const float gwd_lower_bound) const {
    bounds_check(idx);

    float exponent;
    if (is_int64) {
      const auto prev_iter = data_i64[idx];
      if (prev_iter == 0) {
        return 1.0f;
      }
      const auto gap = iter - prev_iter - 1;
      exponent = static_cast<float>(gap > 0 ? gap : 0);
    } else {
      const auto prev_iter = data_f32[idx];
      if (prev_iter == 0) {
        return 1.0f;
      }
      exponent = max(iter - prev_iter - 1, 0.0f);
    }
    return max(gwd_lower_bound, powf(weight_decay_base, exponent));
  }

  /// Record that row `idx` was last touched at iteration `iter`.
  __device__ inline void store(const int64_t idx, const int64_t iter) const {
    bounds_check(idx);

    if (is_int64) {
      data_i64[idx] = iter;
    } else {
      const_cast<float*>(data_f32)[idx] = static_cast<float>(iter);
    }
  }

 private:
  __device__ inline void bounds_check(
      [[maybe_unused]] const int64_t idx) const {
#ifdef FBGEMM_GPU_MEMCHECK
    // Replaces the checking that the PackedTensorAccessor used to provide.
    CUDA_KERNEL_ASSERT(
        idx >= 0 && idx < size && "prev_iter row index is out of bounds");
#endif
  }
};

/// Build a `PrevIterRef` over `prev_iter_dev`, which must be float32 or int64.
inline PrevIterRef make_prev_iter_ref(const at::Tensor& prev_iter_dev) {
  TORCH_CHECK(
      prev_iter_dev.defined(), "prev_iter is required for global weight decay");

  const auto dtype = prev_iter_dev.scalar_type();
  TORCH_CHECK(
      dtype == at::kFloat || dtype == at::kLong,
      "prev_iter must be float32 or int64, got ",
      dtype);

  const auto is_int64 = (dtype == at::kLong);
  return PrevIterRef{
      is_int64 ? nullptr : prev_iter_dev.data_ptr<float>(),
      is_int64 ? prev_iter_dev.data_ptr<int64_t>() : nullptr,
      prev_iter_dev.numel(),
      is_int64};
}

} // namespace fbgemm_gpu
