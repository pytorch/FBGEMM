/*
 * Portions Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include <ck_tile/core.hpp>
#include "fused_moegemm.hpp"
#include "fused_moegemm_api_traits.hpp"
#include "fused_moegemm_api_internal.hpp"

// clang-format off
template float fused_moegemm_<
    fmoe_<ck_tile::fp16_t, ck_tile::fp16_t, ck_tile::fp16_t, float, float, float, float, S<32, 512, 128, 128>, S<1, 4, 1>, S<16, 16, 32>, 0, 0, 0>
>(const ck_tile::stream_config& s, fused_moegemm_args a);

template float fused_moegemm_<
    fmoe_<ck_tile::fp16_t, ck_tile::fp16_t, ck_tile::fp16_t, float, float, float, float, S<32, 512, 128, 128>, S<1, 4, 1>, S<16, 16, 32>, 0, 1, 0>
>(const ck_tile::stream_config& s, fused_moegemm_args a);

template float fused_moegemm_<
    fmoe_<ck_tile::fp16_t, ck_tile::fp16_t, ck_tile::fp16_t, float, float, float, float, S<32, 512, 128, 128>, S<1, 4, 1>, S<16, 16, 32>, 1, 0, 0>
>(const ck_tile::stream_config& s, fused_moegemm_args a);

template float fused_moegemm_<
    fmoe_<ck_tile::fp16_t, ck_tile::fp16_t, ck_tile::fp16_t, float, float, float, float, S<32, 512, 128, 128>, S<1, 4, 1>, S<16, 16, 32>, 1, 1, 0>
>(const ck_tile::stream_config& s, fused_moegemm_args a);

// clang-format on
