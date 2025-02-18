// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core.hpp>

// this is used to pattern-match internl kernel implementation, not to instantiate kernel
template <typename I,
          typename W,
          typename O,
          typename ST,
          typename SW,
          typename SQ,
          typename KW,
          typename BlockTIle_, // seq<b_token, b_interm, b_hidden, b_down>
          typename WarpPerBlock_,
          typename WarpTile_,               // seq<*,*,*>, used to select mfma
          ck_tile::index_t Activation_ = 0, // 0: Gelu 1: Silu
          ck_tile::index_t GateOnly_   = 0,
          ck_tile::index_t FusedQuant_ = 0>
struct fmoe_ // traits, ugly name, only used for internal
{
    using TypeConfig = FusedMoeGemmTypeConfig<I, W, O, ST, SW, SQ, KW>;

    using ADataType            = ck_tile::remove_cvref_t<typename TypeConfig::ADataType>;
    using GDataType            = ck_tile::remove_cvref_t<typename TypeConfig::GDataType>;
    using DDataType            = ck_tile::remove_cvref_t<typename TypeConfig::DDataType>;
    using AccDataType          = ck_tile::remove_cvref_t<typename TypeConfig::AccDataType>;
    using ODataType            = ck_tile::remove_cvref_t<typename TypeConfig::ODataType>;
    using AScaleDataType       = ck_tile::remove_cvref_t<typename TypeConfig::AScaleDataType>;
    using GScaleDataType       = ck_tile::remove_cvref_t<typename TypeConfig::GScaleDataType>;
    using DScaleDataType       = ck_tile::remove_cvref_t<typename TypeConfig::DScaleDataType>;
    using YSmoothScaleDataType = ck_tile::remove_cvref_t<typename TypeConfig::YSmoothScaleDataType>;
    using TopkWeightDataType   = ck_tile::remove_cvref_t<typename TypeConfig::TopkWeightDataType>;
    using IndexDataType        = ck_tile::remove_cvref_t<typename TypeConfig::IndexDataType>;

    static constexpr ck_tile::index_t BT_ = BlockTIle_::at(ck_tile::number<0>{}); // block token
    static constexpr ck_tile::index_t BI_ =
        BlockTIle_::at(ck_tile::number<1>{}); // block intermediate
    static constexpr ck_tile::index_t BH_ = BlockTIle_::at(ck_tile::number<2>{}); // block hidden
    static constexpr ck_tile::index_t BD_ = BlockTIle_::at(ck_tile::number<3>{}); // block down

    using BlockTile_0    = ck_tile::sequence<BT_, BI_, BH_>;
    using WarpPerBlock_0 = ck_tile::remove_cvref_t<WarpPerBlock_>;
    using WarpTile_0     = ck_tile::remove_cvref_t<WarpTile_>;

    using BlockTile_1    = ck_tile::sequence<BT_, BD_, BI_>;
    using WarpPerBlock_1 = ck_tile::remove_cvref_t<WarpPerBlock_>;
    using WarpTile_1     = ck_tile::remove_cvref_t<WarpTile_>;

    static constexpr ck_tile::index_t Activation = Activation_; // 0: Gelu 1: Silu
    static constexpr ck_tile::index_t GateOnly   = GateOnly_;
    static constexpr ck_tile::index_t FusedQuant = FusedQuant_;
};
