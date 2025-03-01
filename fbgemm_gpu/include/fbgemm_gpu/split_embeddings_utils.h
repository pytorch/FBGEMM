/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>

// These values are adjusted in backward based on B and T
constexpr int DEFAULT_INFO_NUM_BITS = 32;
constexpr int DEFAULT_INFO_B_NUM_BITS = 26;
constexpr uint32_t DEFAULT_INFO_B_MASK = (1u << DEFAULT_INFO_B_NUM_BITS) - 1;
constexpr uint32_t MAX_T =
    (1u << (DEFAULT_INFO_NUM_BITS - DEFAULT_INFO_B_NUM_BITS)) - 1;
constexpr uint32_t MAX_B = (1u << DEFAULT_INFO_B_NUM_BITS) - 1;

std::tuple<int64_t, int64_t>
get_infos_metadata(at::Tensor unused, int64_t B, int64_t T);

std::tuple<int32_t, uint32_t> adjust_info_B_num_bits(int32_t B, int32_t T);
std::tuple<int32_t, uint32_t> get_info_B_num_bits_from_T(int32_t T, int32_t B);

std::tuple<at::Tensor /*row_output_offsets*/, at::Tensor /*b_t_map*/>
generate_vbe_metadata(
    const at::Tensor& B_offsets,
    const at::Tensor& B_offsets_rank_per_feature,
    const at::Tensor& output_offsets_feature_rank,
    const at::Tensor& D_offsets,
    const int64_t D,
    const bool nobag,
    const int64_t max_B_feature_rank,
    const int64_t info_B_num_bits,
    const int64_t total_B);
