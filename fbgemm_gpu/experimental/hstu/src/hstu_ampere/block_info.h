/*
 * Copyright (c) 2023, Tri Dao.
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

namespace flash {

////////////////////////////////////////////////////////////////////////////////////////////////////

struct HstuBlockInfo {
  template <typename Params>
  __device__ HstuBlockInfo(const Params& params, const int bidb)
      : sum_s_q(params.cu_seqlens_q[bidb]),
        sum_s_k(params.cu_seqlens_k[bidb]),
        actual_seqlen_c(
            params.num_contexts == nullptr ? 0 : params.num_contexts[bidb]),
        actual_seqlen_q(params.cu_seqlens_q[bidb + 1] - sum_s_q),
        actual_seqlen_k(params.cu_seqlens_k[bidb + 1] - sum_s_k),
        actual_seqlen_t(
            params.num_targets == nullptr ? 0 : params.num_targets[bidb]) {}

  template <typename index_t>
  __forceinline__ __device__ index_t q_offset(const index_t row_stride) const {
    return uint32_t(sum_s_q) * row_stride;
  }

  template <typename index_t>
  __forceinline__ __device__ index_t k_offset(const index_t row_stride) const {
    return uint32_t(sum_s_k) * row_stride;
  }

  const int sum_s_q;
  const int sum_s_k;

  const int actual_seqlen_c;
  const int actual_seqlen_q;
  const int actual_seqlen_k;
  const int actual_seqlen_t;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace flash
