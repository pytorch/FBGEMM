// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "fused_moe.hpp"

float fused_moe(fused_moe_traits t, fused_moe_args a, const ck_tile::stream_config& s)
{
    auto s_sub = ck_tile::stream_config{s.stream_id_, false, s.log_level_, 0, 1};

    auto o_data_bytes = [&]() {
        if(t.prec_o == "fp32")
            return 4;
        else if(t.prec_o == "fp16" || t.prec_o == "bf16")
            return 2;
        else if(t.prec_o == "int8" || t.prec_o == "fp8")
            return 1;
        return 1;
    }();

    auto t0 = fused_moesorting_trait{"int32", "fp32", t.local_expert_masking};
    auto a0 = fused_moesorting_args{
        a.topk_ids_ptr,                              // const void* p_topk_ids;
        a.topk_weight_ptr,                           // const void* p_weights;
        a.local_expert_mask_ptr,                     // const void* p_local_expert_mask;
        a.sorted_token_ids_ptr,                      // void* p_sorted_token_ids;
        a.sorted_weight_ptr,                         // void* p_sorted_weights;
        a.sorted_expert_ids_ptr,                     // void* p_sorted_expert_ids;
        a.num_sorted_tiles_ptr,                      // void* p_total_tokens_post_pad;
        a.o_ptr,                                     // void* p_moe_buf;
        a.ws_ptr,                                    // moe_sorting_ws
        a.num_tokens,                                // index_t tokens;
        a.block_m,                                   // index_t unit_size;
        a.num_experts,                               // index_t num_experts;
        a.topk,                                      // index_t topk;
        a.num_tokens * a.stride_token * o_data_bytes // index_t moe_buf_bytes;
    };

    auto t1 = fused_moegemm_traits{t.prec_i,
                                   t.prec_w,
                                   t.prec_o,
                                   t.prec_st,
                                   t.prec_sw,
                                   t.prec_sq,
                                   t.prec_kw,
                                   t.block_m,
                                   t.activation,
                                   t.gate_only,
                                   t.fused_quant};
    auto a1 = fused_moegemm_args{
        a.a_ptr,                 // const void* a_ptr;
        a.a_scale_ptr,           // const void* a_scale_ptr;
        a.g_ptr,                 // const void* g_ptr;
        a.d_ptr,                 // const void* d_ptr;
        a.g_scale_ptr,           // const void* g_scale_ptr;
        a.d_scale_ptr,           // const void* d_scale_ptr;
        a.y_smooth_scale_ptr,    // const void* y_smooth_scale_ptr;
        a.o_ptr,                 // void* o_ptr;
        a.sorted_token_ids_ptr,  // const void* sorted_token_ids_ptr;
        a.sorted_weight_ptr,     // const void* sorted_weight_ptr;
        a.sorted_expert_ids_ptr, // const void* sorted_expert_ids_ptr;
        a.num_sorted_tiles_ptr,  // const void* num_sorted_tiles_ptr;
        a.hidden_size,           // index_t hidden_size;
        a.intermediate_size,     // index_t intermediate_size;
        a.num_tokens,            // index_t num_tokens;
        a.num_experts,           // index_t num_experts;
        a.topk,                  // index_t topk;
        a.stride_token           // index_t stride_token;
    };

    float r0 = -1;
    float r1 = -1;

    float r = ck_tile::launch_kernel(
        s,
        [=, &r0](const ck_tile::stream_config&) { r0 = fused_moesorting(t0, a0, s_sub); },
        [=, &r1](const ck_tile::stream_config&) { r1 = fused_moegemm(t1, a1, s_sub); });

    // keep unsupported case return negative
    if(r0 < 0 || r1 < 0)
        return -1;

    return r;
}
