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

template <typename Kernel_traits, typename Params>
struct HstuBlockInfo {
  __device__ HstuBlockInfo(const Params& params, const int bidb)
      : sum_s_q(params.cu_seqlens_q[bidb]),
        sum_s_k(params.cu_seqlens_k[bidb]),
        sum_s_page(Kernel_traits::Paged_KV ? params.page_offsets[bidb] : 0),
        actual_seqlen_c(
            Kernel_traits::Is_context ? params.num_contexts[bidb] : 0),
        actual_seqlen_q(
          params.seqused_q ? params.seqused_q[bidb] : params.cu_seqlens_q[bidb + 1] - sum_s_q),
        actual_seqlen_k(
          params.seqused_k ? params.seqused_k[bidb] : params.cu_seqlens_k[bidb + 1] - sum_s_k),
        actual_seqlen_q_padded(
          params.cu_seqlens_q[bidb + 1] - sum_s_q),
        actual_seqlen_k_padded(
          params.cu_seqlens_k[bidb + 1] - sum_s_k),
        actual_seqlen_t(Kernel_traits::Is_target ? params.num_targets[bidb] : 0),
        actual_page_num(
            Kernel_traits::Paged_KV ? params.page_offsets[bidb + 1] - sum_s_page : 0),
        last_page_seqlen(Kernel_traits::Paged_KV ? params.last_page_lens[bidb] : 0) {}

  template <typename index_t>
  __forceinline__ __device__ index_t q_offset(const index_t row_stride) const {
    return uint32_t(sum_s_q) * row_stride;
  }

  template <typename index_t>
  __forceinline__ __device__ index_t k_offset(const index_t row_stride) const {
    return uint32_t(sum_s_k) * row_stride;
  }

  template <typename index_t>
  __forceinline__ __device__ index_t kv_cache_offset(const index_t row_stride) const {
    return uint32_t(sum_s_q) * row_stride + (actual_seqlen_q - actual_seqlen_t) * row_stride;
  }

  const int sum_s_q;
  const int sum_s_k;
  const int sum_s_page;

  const int actual_seqlen_c;
  const int actual_seqlen_q;
  const int actual_seqlen_k;
  const int actual_seqlen_q_padded;
  const int actual_seqlen_k_padded;
  const int actual_seqlen_t;
  const int actual_page_num;
  const int last_page_seqlen;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace flash
