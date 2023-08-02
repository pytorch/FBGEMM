/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

{#
// @lint-ignore LINTIGNORE
// @lint-ignore-every CLANGFORMAT
// clang-format off
// Note: clang-format off doesn't work with this templaterized code,
// so we need to keep lint-ignore-every.
// See https://fburl.com/dw9ljh4h
#}

#include "codegen/embedding_forward_template_helpers.cuh"

using Tensor = at::Tensor;
using namespace fbgemm_gpu;

template <
    typename emb_t,
    typename cache_t,
    typename output_t,
    typename index_t,
    size_t kThreadGroupSize
    >
__launch_bounds__(kForwardMaxThreads) __global__ void
{%- if is_index_select %}
batch_index_select_dim0_codegen_forward_small_kernel(
{%- else %}
{{ "dense" if dense else "split" }}_embedding_nobag_codegen_forward_unweighted_small_kernel(
{%- endif %}
    const pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> dev_weights,
    {%- if not dense %}
    const pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> uvm_weights,
    const pta::PackedTensorAccessor64<cache_t, 2, at::RestrictPtrTraits> lxu_cache_weights,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> weights_placements,
    {%- endif %}
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> weights_offsets,
    {%- if is_index_select %}
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> D_offsets,
    {%- else %}
    int64_t D,
    {%- endif %}
    FixedDivisor fd_B,
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> indices,
    {%- if not is_index_select %}
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> offsets,
    {%- endif %}
    {%- if not dense %}
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> lxu_cache_locations,
    {%- endif %}
    {%- if is_index_select %}
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> output_offsets,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> total_L_offsets,
    const int32_t fixed_L_per_warp,
    const bool permute_output_dim_0_1,
    {%- endif %} // if dense
    // If 2D, shape is [B][total_D]
    pta::PackedTensorAccessor64<output_t, {{ "1" if is_index_select else "2" }}, at::RestrictPtrTraits> output
    ) {
    int32_t T = weights_offsets.size(0);
    int32_t b_t = blockIdx.x * blockDim.y + threadIdx.y;
    {%- if not is_index_select %}
    if (b_t >= offsets.size(0) - 1) {
        return;
    }
    {%- endif %}
    int32_t t;
    int32_t b;

    fd_B.DivMod(b_t, &t, &b);

    {%- if is_index_select %}
    index_t indices_start;
    int32_t L;
    int32_t L_start;
    if (t >= T) {
        return;
    }
    const auto total_L_start = total_L_offsets[t];
    const auto total_L = total_L_offsets[t + 1] - total_L_start;
    L_start = b * fixed_L_per_warp;
    if (L_start >= total_L) {
        return;
    }
    indices_start = total_L_start + L_start;
    L = (total_L - L_start >= fixed_L_per_warp) ? fixed_L_per_warp : (total_L - L_start);
    {%- else %}
    index_t indices_start = offsets[b_t];
    int32_t L = offsets[b_t + 1] - indices_start;
    {%- endif %}

    {%- if is_index_select %}
    const int32_t D_start = D_offsets[t];
    const int32_t D_end = D_offsets[t + 1];
    const int32_t D = D_end - D_start;

    // Check D in the kernel to avoid iterating through the list on host
    CUDA_KERNEL_ASSERT(D % 4 == 0 && "The column size must be multiple of 4");
    const auto output_offset = permute_output_dim_0_1 ? D_start : output_offsets[t];
    const auto output_stride = permute_output_dim_0_1 ? D_offsets[T] : D;
    {%- endif %} // dense

    int64_t weights_offset = weights_offsets[t];
    const emb_t* __restrict__ weights;
    {%- if not dense %}
    const auto placement = static_cast<PlacementType>(weights_placements[t]);
    if (placement == PlacementType::DEVICE) {
        weights = &dev_weights[weights_offset];
    } else {
        weights = &uvm_weights[weights_offset];
    }
    {%- else %}
    weights = &dev_weights[weights_offset];
    {%- endif %}

    int32_t D_emb = D;
    if (std::is_same<emb_t, uint8_t>::value) {
        D_emb += kINT8QparamsBytes;
    }

    const int32_t group_start = threadIdx.x / kThreadGroupSize * kThreadGroupSize;
    const int32_t group_end = group_start + kThreadGroupSize;
    const int32_t d = threadIdx.x % kThreadGroupSize * 4;

    for (int32_t l_start = 0; l_start < L; l_start += kWarpSize) {
        int32_t l = l_start + threadIdx.x;
        int64_t idx = l < L ? indices[indices_start + l] : 0;
        {%- if not dense %}
        int32_t cache_idx = (placement == PlacementType::MANAGED_CACHING && l < L) ? lxu_cache_locations[indices_start + l] : 0;
        {%- endif %}
        for (auto j = group_start; j < group_end && l_start + j < L; ++j) {
            int64_t idx_j = shfl_sync(idx, j);
            {%- if is_index_select %}
            int64_t output_j = L_start + l_start + j;
            {%- else %}
            int64_t output_j = indices_start + l_start + j;
            {%- endif %}
            {%- if not dense %}
            int32_t cache_idx_j = shfl_sync(cache_idx, j);
            {%- endif %}

            {%- if not dense %}

            // assume cache is fp16/fp32 which doesn't require qparams
            float2 qparams_cache = make_float2(0.0f, 0.0f);

            {%- endif %}
            auto weight_row_emb = WeightRow<emb_t, cache_t, cache_t>(
                const_cast<emb_t*>(&weights[idx_j * D_emb]),
                nullptr,
                D,
                nullptr);
            [[maybe_unused]] float2 qparams_emb;
            if (std::is_same<emb_t, uint8_t>::value) {
                qparams_emb = weight_row_emb.load_qparams();
            }

            if (d < D) {
                {%- if not dense %}
                if (placement == PlacementType::MANAGED_CACHING && cache_idx_j != kCacheLocationMissing) {
                    auto weight_row_cache = WeightRow<emb_t, cache_t, cache_t>(
                        const_cast<emb_t*>(&weights[idx_j * D_emb]),
                        const_cast<cache_t*>(&lxu_cache_weights[cache_idx_j][0]),
                        D,
                        nullptr);
                    Vec4T<cache_t> weight = weight_row_cache.load(d, qparams_cache);
                    weight.store(&output[output_j][d]);
                } else {
                    Vec4T<cache_t> weight = weight_row_emb.load(d, qparams_emb);
                    weight.store(&output[output_j][d]);
                }
                {%- else %}
                    Vec4T<cache_t> weight = weight_row_emb.load(d, qparams_emb);
                    {%- if is_index_select %}
                    // output is 1D (because the stride can be irregular)
                    weight.store(&output[output_offset + output_j * output_stride + d]);
                    {%- else %}
                    // output is 2D
                    weight.store(&output[output_j][d]);
                    {%- endif %}
                {%- endif %}
            }
        }
    }
}

/*
    Explicitly instantiate the kernel function template.  The instantiations are
    based on the types enumerated by DISPATCH_EMB_GRAD_CACHE_TYPES macro used in
    embedding_forward_split_template.cu
*/

{%- for output_type in ['uint8_t', 'at::Half', 'float'] %}
{%- for emb_type in ['uint8_t', 'float', 'at::Half'] %}
{%- for cache_type in ['float', 'at::Half'] %}
{%- for kEmbeddingSize in [4, 8, 16, 32] %}
{%- set index_type = 'int64_t' %}

template __launch_bounds__(kForwardMaxThreads) __global__ void
{%- if is_index_select %}
batch_index_select_dim0_codegen_forward_small_kernel
{%- else %}
{{ "dense" if dense else "split" }}_embedding_nobag_codegen_forward_unweighted_small_kernel
{%- endif %}
<
  {{ emb_type }},
  {{ cache_type }},
  {{ output_type }},
  {{ index_type }},
  {{ kEmbeddingSize // 4 }}
> (
    const pta::PackedTensorAccessor64<{{ emb_type }}, 1, at::RestrictPtrTraits> dev_weights,
    {%- if not dense %}
    const pta::PackedTensorAccessor64<{{ emb_type }}, 1, at::RestrictPtrTraits> uvm_weights,
    const pta::PackedTensorAccessor64<{{ cache_type }}, 2, at::RestrictPtrTraits> lxu_cache_weights,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> weights_placements,
    {%- endif %}
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> weights_offsets,
    {%- if is_index_select %}
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> D_offsets,
    {%- else %}
    int64_t D,
    {%- endif %}
    FixedDivisor fd_B,
    const pta::PackedTensorAccessor32<{{ index_type }}, 1, at::RestrictPtrTraits> indices,
    {%- if not is_index_select %}
    const pta::PackedTensorAccessor32<{{ index_type }}, 1, at::RestrictPtrTraits> offsets,
    {%- endif %}
    {%- if not dense %}
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> lxu_cache_locations,
    {%- endif %}
    {%- if is_index_select %}
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> output_offsets,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> total_L_offsets,
    const int32_t fixed_L_per_warp,
    const bool permute_output_dim_0_1,
    {%- endif %}
    pta::PackedTensorAccessor64<{{ output_type }}, {{ "1" if is_index_select else "2" }}, at::RestrictPtrTraits> output);

{%- endfor %}
{%- endfor %}
{%- endfor %}
{%- endfor %}
