// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include <ck_tile/core.hpp>
#include "fused_moegemm.hpp"
#include "fused_moegemm_api_traits.hpp"

// Note: this internal API only declare, not define here, otherwise will block `make -j`
template <typename Traits_>
float fused_moegemm_(const ck_tile::stream_config& s, fused_moegemm_args a);

template <ck_tile::index_t... Is>
using S = ck_tile::sequence<Is...>;

float fused_moegemm(fused_moegemm_traits t, fused_moegemm_args a, const ck_tile::stream_config& s)
{
    // clang-format off
    float r = -1;
    if(t.prec_i == "bf16" && t.prec_w == "bf16" && t.prec_o == "bf16" && t.prec_st == "fp32" &&
       t.prec_sw == "fp32" && t.prec_sq == "fp32" && t.prec_kw == "fp32" && t.block_m == 32 && t.gate_only == 1 && t.activation == 0)
    {
        constexpr ck_tile::index_t act_ = 0;
        constexpr ck_tile::index_t go_  = 1;
        using t_ = fmoe_<ck_tile::bf16_t, ck_tile::bf16_t, ck_tile::bf16_t, float, float, float, float, S<32, 512, 128, 128>, S<1, 4, 1>, S<16, 16, 32>, act_, go_, 0>;
        r = fused_moegemm_<t_>(s, a);
    }
    else if(t.prec_i == "bf16" && t.prec_w == "bf16" && t.prec_o == "bf16" && t.prec_st == "fp32" &&
       t.prec_sw == "fp32" && t.prec_sq == "fp32" && t.prec_kw == "fp32" && t.block_m == 32 && t.gate_only == 0 && t.activation == 0)
    {
        constexpr ck_tile::index_t act_ = 0;
        constexpr ck_tile::index_t go_  = 0;
        using t_ = fmoe_<ck_tile::bf16_t, ck_tile::bf16_t, ck_tile::bf16_t, float, float, float, float, S<32, 512, 128, 128>, S<1, 4, 1>, S<16, 16, 32>, act_, go_, 0>;
        r = fused_moegemm_<t_>(s, a);
    }
    else if(t.prec_i == "fp16" && t.prec_w == "fp16" && t.prec_o == "fp16" && t.prec_st == "fp32" &&
       t.prec_sw == "fp32" && t.prec_sq == "fp32" && t.prec_kw == "fp32" && t.block_m == 32 && t.gate_only == 1 && t.activation == 0)
    {
        constexpr ck_tile::index_t act_ = 0;
        constexpr ck_tile::index_t go_  = 1;
        using t_ = fmoe_<ck_tile::fp16_t, ck_tile::fp16_t, ck_tile::fp16_t, float, float, float, float, S<32, 512, 128, 128>, S<1, 4, 1>, S<16, 16, 32>, act_, go_, 0>;
        r = fused_moegemm_<t_>(s, a);
    }
    else if(t.prec_i == "fp16" && t.prec_w == "fp16" && t.prec_o == "fp16" && t.prec_st == "fp32" &&
       t.prec_sw == "fp32" && t.prec_sq == "fp32" && t.prec_kw == "fp32" && t.block_m == 32 && t.gate_only == 0 && t.activation == 0)
    {
        constexpr ck_tile::index_t act_ = 0;
        constexpr ck_tile::index_t go_  = 0;
        using t_ = fmoe_<ck_tile::fp16_t, ck_tile::fp16_t, ck_tile::fp16_t, float, float, float, float, S<32, 512, 128, 128>, S<1, 4, 1>, S<16, 16, 32>, act_, go_, 0>;
        r = fused_moegemm_<t_>(s, a);
    }
    else if(t.prec_i == "bf16" && t.prec_w == "bf16" && t.prec_o == "bf16" && t.prec_st == "fp32" &&
       t.prec_sw == "fp32" && t.prec_sq == "fp32" && t.prec_kw == "fp32" && t.block_m == 32 && t.gate_only == 1 && t.activation == 1)
    {
        constexpr ck_tile::index_t act_ = 1;
        constexpr ck_tile::index_t go_  = 1;
        using t_ = fmoe_<ck_tile::bf16_t, ck_tile::bf16_t, ck_tile::bf16_t, float, float, float, float, S<32, 512, 128, 128>, S<1, 4, 1>, S<16, 16, 32>, act_, go_, 0>;
        r = fused_moegemm_<t_>(s, a);
    }
    else if(t.prec_i == "bf16" && t.prec_w == "bf16" && t.prec_o == "bf16" && t.prec_st == "fp32" &&
       t.prec_sw == "fp32" && t.prec_sq == "fp32" && t.prec_kw == "fp32" && t.block_m == 32 && t.gate_only == 0 && t.activation == 1)
    {
        constexpr ck_tile::index_t act_ = 1;
        constexpr ck_tile::index_t go_  = 0;
        using t_ = fmoe_<ck_tile::bf16_t, ck_tile::bf16_t, ck_tile::bf16_t, float, float, float, float, S<32, 512, 128, 128>, S<1, 4, 1>, S<16, 16, 32>, act_, go_, 0>;
        r = fused_moegemm_<t_>(s, a);
    }
    else if(t.prec_i == "fp16" && t.prec_w == "fp16" && t.prec_o == "fp16" && t.prec_st == "fp32" &&
       t.prec_sw == "fp32" && t.prec_sq == "fp32" && t.prec_kw == "fp32" && t.block_m == 32 && t.gate_only == 1 && t.activation == 1)
    {
        constexpr ck_tile::index_t act_ = 1;
        constexpr ck_tile::index_t go_  = 1;
        using t_ = fmoe_<ck_tile::fp16_t, ck_tile::fp16_t, ck_tile::fp16_t, float, float, float, float, S<32, 512, 128, 128>, S<1, 4, 1>, S<16, 16, 32>, act_, go_, 0>;
        r = fused_moegemm_<t_>(s, a);
    }
    else if(t.prec_i == "fp16" && t.prec_w == "fp16" && t.prec_o == "fp16" && t.prec_st == "fp32" &&
       t.prec_sw == "fp32" && t.prec_sq == "fp32" && t.prec_kw == "fp32" && t.block_m == 32 && t.gate_only == 0 && t.activation == 1)
    {
        constexpr ck_tile::index_t act_ = 1;
        constexpr ck_tile::index_t go_  = 0;
        using t_ = fmoe_<ck_tile::fp16_t, ck_tile::fp16_t, ck_tile::fp16_t, float, float, float, float, S<32, 512, 128, 128>, S<1, 4, 1>, S<16, 16, 32>, act_, go_, 0>;
        r = fused_moegemm_<t_>(s, a);
    }
    // clang-format on
    return r;
}
