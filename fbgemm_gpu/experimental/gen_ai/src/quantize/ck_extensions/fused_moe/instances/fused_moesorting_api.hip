// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "fused_moesorting.hpp"

#define MOE_SORTING_DISPATCH(unroll_num_)                                                   \
    constexpr ck_tile::index_t unroll_num = unroll_num_;                                    \
    using ms_problem     = ck_tile::MoeSortingProblem<index_t, ms_weight_type, unroll_num>; \
    using kernel         = ck_tile::MoeSortingKernel<ms_problem>;                           \
    auto kargs           = kernel::MakeKargs(a);                                            \
    const dim3 grids     = kernel::GridSize(a);                                             \
    const dim3 blocks    = kernel::BlockSize(a);                                            \
    const auto lds_bytes = kernel::GetSmemSize(a);                                          \
    float ave_time       = ck_tile::launch_kernel(                                          \
        s, ck_tile::make_kernel(kernel{}, grids, blocks, lds_bytes, kargs));          \
    return ave_time;

float fused_moesorting(fused_moesorting_trait t, fused_moesorting_args a, ck_tile::stream_config s)
{
    if(t.weight_type == "fp32" && t.index_type == "int32")
    {
        if(a.num_experts > 127)
        {
            printf("lds size exceed, only support experts <127 \n");
            return -1;
        }
        if(a.moe_buf_bytes % 16)
        {
            printf("buf set size %d unaligned, must be multiple of 16\n", a.moe_buf_bytes);
            return -1;
        }
        using index_t              = ck_tile::index_t;
        using ms_weight_type       = float;
        index_t smem_io_unroll_num = ck_tile::integer_divide_ceil(a.tokens * a.topk, 64);
        switch(smem_io_unroll_num)
        {
        case(1): {
            MOE_SORTING_DISPATCH(1);
        }
        case(2): {
            MOE_SORTING_DISPATCH(2);
        }
        case(3): {
            MOE_SORTING_DISPATCH(3);
        }
        case(5): {
            MOE_SORTING_DISPATCH(5);
        }
        case(6): {
            MOE_SORTING_DISPATCH(6);
        }
        case(7): {
            MOE_SORTING_DISPATCH(7);
        }
        case(8): {
            MOE_SORTING_DISPATCH(8);
        }
        case(9): {
            MOE_SORTING_DISPATCH(9);
        }
        case(10): {
            MOE_SORTING_DISPATCH(10);
        }
        case(11): {
            MOE_SORTING_DISPATCH(11);
        }
        default: {
            MOE_SORTING_DISPATCH(4);
        }
        }
    }
    return -1;
}
