/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// clang-format off

#include "fbgemm_gpu/embedding_backward_template_helpers.cuh"
#include "fbgemm_gpu/fbgemm_tensor_accessor.h"
#include "fbgemm_gpu/split_embeddings_utils.cuh"

using namespace fbgemm_gpu;

{%- if gen_once %}
{#- /*
    The kernels in this section will be generated only once for all TBE configs
    as they are common across the different configs. The generated file name is
    `gen_embedding_backward_common_split_device_kernel.cuh`
     */
#}

template<
    typename emb_t,
    typename cache_t,
    size_t kMaxVecsPerThread,
    int32_t kThreadGroupSize,
    int32_t VEC_WIDTH
>
DEVICE_INLINE void store_grad_sum(
    pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits>& grad_dev_weights,
    const Vec4T<at::acc_type<cache_t, true>>* grad_sum,
    const int32_t D,
    const int64_t weights_offset,
    const int64_t idx
) {
    #pragma unroll kMaxVecsPerThread
    for (int32_t i = 0;
        i < kMaxVecsPerThread && (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
        ++i) {
        int32_t d = (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH;
        auto& grad = grad_sum[i];
        grad.store(&grad_dev_weights[weights_offset + idx * D + d]);
    }
}

{%- else %}

{#- /*
    The kernels in this section will be generated multiple times based on the
    TBE configs (weighted/unweighted, bag/no bag, vbe/no vbe). The generated
    file name's pattern is
    `gen_embedding_backward_[weighted|unweighted][_nobag|][_vbe|]_split_device_kernel.cuh`
     */
#}

template <
    typename grad_t,
    typename cache_t,
    size_t kMaxVecsPerThread,
    int32_t kThreadGroupSize = kWarpSize,
    int32_t VEC_WIDTH
>
DEVICE_INLINE void compute_grad_sum_{{ kdesc }}(
    Vec4T<at::acc_type<cache_t, true>>* grad_sum,
    const pta::PackedTensorAccessor64<grad_t, {{ "1" if is_index_select else "2" }}, at::RestrictPtrTraits>& grad_output,
    {%- if not nobag or is_index_select %}
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>& D_offsets,
    {%- endif %}
    const int32_t D,
    const int32_t T,
    const pta::PackedTensorAccessor32<{{ "int64_t" if nobag else "int32_t" }}, 1, at::RestrictPtrTraits>& sorted_infos,
    {%- if weighted %}
    const pta::PackedTensorAccessor32<at::acc_type<cache_t, true>, 1, at::RestrictPtrTraits>& sorted_indice_weights,
    {%- endif %}
    {%- if not nobag and vbe %}
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>& B_offsets,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>& row_output_offsets,
    {%- endif %}
    {%- if is_index_select %}
    const int64_t grad_offset,
    const int32_t grad_stride,
    {%- endif %}
    {%- if not nobag %}
    const int32_t info_B_num_bits,
    const uint32_t info_B_mask,
    {%- endif %}
    const int32_t segment_start,
    const int32_t sl_start,
    const int32_t sl_end,
    const unsigned int shfl_sync_mask
) {
    for (int32_t sl = sl_start; sl < sl_end; sl += kThreadGroupSize) {
        int32_t sl_j = sl + threadIdx.x;
        {%- if not nobag %}
        const auto b_t = sl_j < sl_end ? reinterpret_cast<const uint32_t*>(&sorted_infos[0])[segment_start + sl_j] : 0;
        const auto b = b_t & info_B_mask;
        const auto t = b_t >> info_B_num_bits;
        {%- if vbe %}
        const auto grad_offset = row_output_offsets[B_offsets[t] + b];
        {%- else %} // if vbe
        int32_t D_start = sl_j < sl_end ? D_offsets[t] : 0;
        {%- endif %} // if vbe
        {%- else %} // if not nobag
        int64_t l_t = sl_j < sl_end ? sorted_infos[segment_start + sl_j] : 0;
        int32_t l = l_t / T;
        {%- endif %} // if not nobag
        {%- if weighted %}
        at::acc_type<cache_t, true> idx_weight = sl_j < sl_end ? sorted_indice_weights[segment_start + sl_j] : 0.0;
        {%- endif %}
        for (int32_t j = 0; j < kThreadGroupSize && sl + j < sl_end; ++j) {
            {%- if nobag %}
            int32_t l_j = SHFL_SYNC(l, j);
            {%- elif vbe %}
            const auto grad_offset_j = SHFL_SYNC(grad_offset, j);
            {%- else %}
            int32_t b_j = SHFL_SYNC(b, j);
            int32_t D_start_j = SHFL_SYNC(D_start, j);
            {%- endif %}

            {%- if weighted %}
            at::acc_type<cache_t, true> idx_weight_j = SHFL_SYNC(idx_weight, j);
            {%- endif %}

            #pragma unroll kMaxVecsPerThread
            for (int32_t i = 0;
                i < kMaxVecsPerThread && (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
                ++i) {
                int32_t d = (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH;
                Vec4T<at::acc_type<grad_t, true>> grad_out_vec(
                    {%- if nobag and is_index_select %}
                    // grad_output is 1d
                    &grad_output[grad_offset + l_j * grad_stride + d]
                    {%- elif nobag %}
                    &grad_output[l_j][d]
                    {%- elif vbe %}
                    &grad_output[0][grad_offset_j + d]
                    {%- else %}
                    &grad_output[b_j][0] + D_start_j + d
                    {%- endif %} // if nobag
                );

                {%- if weighted %}
                grad_sum[i].fma_(grad_out_vec, idx_weight_j);
                {%- else %}
                grad_sum[i].add_(grad_out_vec);
                {%- endif %}
            }
        }
    }
}

{%- endif %}

    // clang-format on
