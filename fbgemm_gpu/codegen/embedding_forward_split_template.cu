/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
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

{%- set wdesc =  "weighted" if weighted else "unweighted" %}
#include "codegen/embedding_forward_template_helpers.cuh"

using Tensor = at::Tensor;
using namespace fbgemm_gpu;

{%- if not weighted %}
template <
    typename emb_t,
    typename cache_t,
    typename output_t,
    typename index_t,
    size_t kThreadGroupSize
    >
__launch_bounds__(kForwardMaxThreads)
__global__ void {{ "dense" if dense else "split" }}_embedding_nobag_codegen_forward_unweighted_small_kernel(
    const at::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> dev_weights,
    {%- if not dense %}
    const at::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> uvm_weights,
    const at::PackedTensorAccessor64<cache_t, 2, at::RestrictPtrTraits> lxu_cache_weights,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> weights_placements,
    {%- endif %}
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> weights_offsets,
    int64_t D,
    FixedDivisor fd_B,
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> indices,
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> offsets,
    {%- if not dense %}
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> lxu_cache_locations,
    {%- endif %}
    at::PackedTensorAccessor64<output_t, 2, at::RestrictPtrTraits> output // [B][total_D],
    );
{%- endif %}

{%- for nobag in [True, False] %}
{%- if not nobag or not weighted %}
template <
    typename emb_t,
    typename cache_t,
    typename output_t,
    {%- if not dense %}
    bool use_lxu_cache,
    {%- endif %}
    typename index_t,
    {%- if not nobag %}
    size_t kMaxVecsPerThread,
    {%- endif %}
    size_t kThreadGroupSize = kWarpSize
    >
__launch_bounds__(kForwardMaxThreads)
__global__ void {{ "dense" if dense else "split" }}_embedding{{ "_nobag" if nobag else "" }}_codegen_forward_{{ wdesc }}_kernel(
    const at::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> dev_weights,
    {%- if not dense %}
    const at::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> uvm_weights,
    const at::PackedTensorAccessor64<cache_t, 2, at::RestrictPtrTraits> lxu_cache_weights,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> weights_placements,
    {%- endif %}
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> weights_offsets,
    {%- if not nobag %}
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> D_offsets,
    {%- else %}
    int64_t D,
    {%- endif %}
    FixedDivisor fd_B,
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> indices,
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> offsets,
    {%- if not nobag %}
    int64_t pooling_mode,
    {%- endif %}
    {%- if weighted %}
    at::PackedTensorAccessor32<at::acc_type<cache_t, true>, 1, at::RestrictPtrTraits> indice_weights,
    {%- endif %}
    {%- if not dense %}
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> lxu_cache_locations,
    {%- endif %}
    at::PackedTensorAccessor64<output_t, 2, at::RestrictPtrTraits> output // [B][total_D],
    );

Tensor {{ "dense" if dense else "split" }}_embedding{{ "_nobag" if nobag else "" }}_codegen_forward_{{ wdesc }}_cuda(
    Tensor dev_weights,
    {%- if not dense %}
    Tensor uvm_weights,
    Tensor lxu_cache_weights,
    Tensor weights_placements,
    {%- endif %}
    Tensor weights_offsets,
    {%- if not nobag %}
    Tensor D_offsets,
    int64_t total_D,
    int64_t max_D,
    {%- else %}
    int64_t D,
    {%- endif %}
    Tensor indices,
    Tensor offsets,
    {%- if not nobag %}
    int64_t pooling_mode,
    {%- endif %}
    {%- if weighted %}
    Tensor indice_weights,
    {%- endif %}
    {%- if not dense %}
    Tensor lxu_cache_locations,
    {%- endif %}
    int64_t output_dtype,
    int64_t unused
) {
    TENSOR_ON_CUDA_GPU(dev_weights);
    {%- if not dense %}
    TENSOR_ON_CUDA_GPU(uvm_weights);
    TENSOR_ON_CUDA_GPU(lxu_cache_weights);
    TENSOR_ON_CUDA_GPU(weights_placements);
    {%- endif %}
    TENSOR_ON_CUDA_GPU(weights_offsets);
    {%- if not nobag %}
    TENSOR_ON_CUDA_GPU(D_offsets);
    {%- endif %}
    TENSOR_ON_CUDA_GPU(indices);
    TENSOR_ON_CUDA_GPU(offsets);
    {%- if weighted %}
    TENSOR_ON_CUDA_GPU(indice_weights);
    {%- endif %}
    {%- if not dense %}
    TENSOR_ON_CUDA_GPU(lxu_cache_locations);
    {%- endif %}

    at::cuda::OptionalCUDAGuard device_guard;
    device_guard.set_index(dev_weights.get_device());

    {%- if not nobag %}
    int32_t T = D_offsets.numel() - 1;
    {%- else %}
    int32_t total_L = indices.numel();
    int32_t T = weights_offsets.numel();
    {%- endif %}
    TORCH_CHECK_GT(T, 0);
    // offsets = [B x T  + 1]
    int32_t B = (offsets.size(0) - 1) / T;
    TORCH_CHECK_GE(B, 0);
    {%- if not nobag %}
    TORCH_CHECK_GT(total_D, 0);
    TORCH_CHECK_EQ(total_D % 4, 0);
    TORCH_CHECK_LE(max_D, {{ max_embedding_dim }});
    {%- else %}
    TORCH_CHECK_GT(D, 0);
    TORCH_CHECK_EQ(D % 4, 0);
    {%- endif %}

    Tensor output;
    {%- if nobag %}
    SparseType o_dtype = static_cast<SparseType>(output_dtype);
    TORCH_CHECK(o_dtype == SparseType::FP32 || o_dtype == SparseType::FP16 ||
                o_dtype == SparseType::BF16 || o_dtype == SparseType::INT8);
    int64_t adjusted_D = D;
    if (o_dtype == SparseType::INT8) {
        adjusted_D += T * kINT8QparamsBytes;
    }
    output = at::empty({total_L, adjusted_D}, dev_weights.options().dtype(getScalarType(o_dtype)));
    {%- else %}
    SparseType o_dtype = static_cast<SparseType>(output_dtype);
    TORCH_CHECK(o_dtype == SparseType::FP32 || o_dtype == SparseType::FP16 ||
                o_dtype == SparseType::BF16 || o_dtype == SparseType::INT8);
    int64_t total_adjusted_D = total_D;
    if (o_dtype == SparseType::INT8) {
        total_adjusted_D += T * kINT8QparamsBytes;
    }
    output = at::empty({B, total_adjusted_D}, dev_weights.options().dtype(getScalarType(o_dtype)));

    {%- endif %}

    if (B == 0) {
        return output;
    }

    DISPATCH_EMB_CACHE_OUTPUT_TYPES(
        dev_weights.scalar_type(),
        {%- if not dense %}
        lxu_cache_weights.scalar_type(),
        {%- else %}
        dev_weights.scalar_type(),
        {%- endif %}
        output.scalar_type(),
        "batched_embedding{{ "_nobag" if nobag else "" }}_forward_kernel_2", [&] {
        {%- if not dense %}
        // Check if LXU cache is used
        bool use_lxu_cache = lxu_cache_weights.numel() > 0;
        {%- endif %}
        {%- if not nobag %}
        {%- for use_cache in ["false", "true"] %}
        // The dense case does not have cache so we have to generate code for
        // only one case (value of use_cache does not matter)
        {%- if (not dense) or (use_cache == "true") %}
        {%- if not dense %}
        if (use_lxu_cache == {{ use_cache }}) {
        {%- endif %}
            // kMaxElemPerThread is # of elements handled by thread if we use a full warp for a row
            // We consider kMaxElemPerThread 1 and 2, and then a multiple of 4.
            {%- for kMaxElemPerThread in range(1, max_embedding_dim // (items_per_warp // 4) + 1) %}
            {%- if kMaxElemPerThread in [1, 2] or kMaxElemPerThread % 4 == 0 %}
            if (max_D <= {{ items_per_warp // 4 * kMaxElemPerThread }}) {
                // hipcc can't use max in constexpr
                constexpr int kMaxVecsPerThread = {{ kMaxElemPerThread }} / 4 >= 1 ? {{ kMaxElemPerThread }} / 4 : 1;
                // If max_D is small, use fewer number of threads than kWarpSize.
#ifdef FBGEMM_USE_SUBWARP_SHUFFLE
                constexpr int kThreadGroupSize = kWarpSize / std::max(4 / {{ kMaxElemPerThread }}, 1);
#else
                constexpr int kThreadGroupSize = kWarpSize;
#endif
                {%- if not dense %}
                split_embedding_codegen_forward_{{ wdesc }}_kernel<emb_t, cache_t, output_t, {{ use_cache }}, int64_t, kMaxVecsPerThread, kThreadGroupSize><<<
                {%- else %}
                dense_embedding_codegen_forward_{{ wdesc }}_kernel<emb_t, cache_t, output_t, int64_t, kMaxVecsPerThread, kThreadGroupSize><<<
                {%- endif %}
                    div_round_up((B * T), kForwardMaxThreads / kThreadGroupSize),
                    dim3(kThreadGroupSize, kForwardMaxThreads / kThreadGroupSize),
                    0,
                    at::cuda::getCurrentCUDAStream()>>>(
                    dev_weights.packed_accessor64<emb_t, 1, at::RestrictPtrTraits>(),
                    {%- if not dense %}
                    uvm_weights.packed_accessor64<emb_t, 1, at::RestrictPtrTraits>(),
                    lxu_cache_weights.packed_accessor64<cache_t, 2, at::RestrictPtrTraits>(),
                    weights_placements.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                    {%- endif %}
                    weights_offsets.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                    D_offsets.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                    FixedDivisor(B),
                    indices.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                    offsets.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                    pooling_mode,
                    {%- if weighted %}
                    indice_weights.packed_accessor32<at::acc_type<cache_t, true>, 1, at::RestrictPtrTraits>(),
                    {%- endif %}
                    {%- if not dense %}
                    lxu_cache_locations.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                    {%- endif %}
                    output.packed_accessor64<
                        output_t,
                        2,
                        at::RestrictPtrTraits>()
                    );

                return;
            }
            {%- endif %}
            {%- endfor %}
        {%- if not dense %}
        } // if (use_lxu_cache == {{ use_cache }})
        {%- endif %}
        {%- endif %} // if (not dense) or (use_cache == "true")
        {%- endfor %} // for use_cache in ["false", "true"]
        {%- else %}
        {%- for kEmbeddingSize in [4, 8, 16, 32] %}
        if (D <= {{ kEmbeddingSize }}) {
        {%- if not dense %}
        split_embedding_nobag_codegen_forward_unweighted_small_kernel<emb_t, cache_t, output_t, int64_t, {{ kEmbeddingSize // 4 }}><<<
        {%- else %}
        dense_embedding_nobag_codegen_forward_unweighted_small_kernel<emb_t, cache_t, output_t, int64_t, {{ kEmbeddingSize // 4 }}><<<
        {%- endif %}
            div_round_up((B * T), kForwardMaxThreads / kWarpSize),
            dim3(kWarpSize, kForwardMaxThreads / kWarpSize),
            0,
            at::cuda::getCurrentCUDAStream()>>>(
            dev_weights.packed_accessor64<emb_t, 1, at::RestrictPtrTraits>(),
            {%- if not dense %}
            uvm_weights.packed_accessor64<emb_t, 1, at::RestrictPtrTraits>(),
            lxu_cache_weights.packed_accessor64<cache_t, 2, at::RestrictPtrTraits>(),
            weights_placements.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
            {%- endif %}
            weights_offsets.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
            D,
            FixedDivisor(B),
            indices.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
            offsets.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
            {%- if not dense %}
            lxu_cache_locations.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
            {%- endif %}
            output.packed_accessor64<
                output_t,
                2,
                at::RestrictPtrTraits>()
            );
            return;
        }
        {%- endfor %}
        {%- for use_cache in ["false", "true"] %}
        // The dense case does not have cache so we have to generate code for
        // only one case (value of use_cache does not matter)
        {%- if (not dense) or (use_cache == "true") %}
        {%- if not dense %}
        if (use_lxu_cache == {{ use_cache }}) {
            split_embedding_nobag_codegen_forward_unweighted_kernel<emb_t, cache_t, output_t, {{ use_cache }}, int64_t><<<
        {%- else %}
            dense_embedding_nobag_codegen_forward_unweighted_kernel<emb_t, cache_t, output_t, int64_t><<<
        {%- endif %}
                div_round_up((B * T), kForwardMaxThreads / kWarpSize),
                dim3(kWarpSize, kForwardMaxThreads / kWarpSize),
                0,
                at::cuda::getCurrentCUDAStream()>>>(
                dev_weights.packed_accessor64<emb_t, 1, at::RestrictPtrTraits>(),
                {%- if not dense %}
                uvm_weights.packed_accessor64<emb_t, 1, at::RestrictPtrTraits>(),
                lxu_cache_weights.packed_accessor64<cache_t, 2, at::RestrictPtrTraits>(),
                weights_placements.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                {%- endif %}
                weights_offsets.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                D,
                FixedDivisor(B),
                indices.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                offsets.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                {%- if not dense %}
                lxu_cache_locations.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                {%- endif %}
                output.packed_accessor64<
                    output_t,
                    2,
                    at::RestrictPtrTraits>()
                );
                return;
        {%- if not dense %}
        } // if (use_lxu_cache == {{ use_cache }})
        {%- endif %}
        {%- endif %} // if (not dense) or (use_cache == "true")
        {%- endfor %} // for use_cache in ["false", "true"]
        {%- endif %}
        });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
{%- endif %}
{%- endfor %}
    // clang-format on
