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

{%- set wdesc =  "weighted" if weighted else "unweighted" %}
{%- set vbe_desc = "_vbe" if vbe else "" %}
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
    const pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> dev_weights,
    {%- if not dense %}
    const pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> uvm_weights,
    const pta::PackedTensorAccessor64<cache_t, 2, at::RestrictPtrTraits> lxu_cache_weights,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> weights_placements,
    {%- endif %}
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> weights_offsets,
    int64_t D,
    FixedDivisor fd_B,
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> indices,
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> offsets,
    {%- if not dense %}
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> lxu_cache_locations,
    {%- endif %}
    pta::PackedTensorAccessor64<output_t, 2, at::RestrictPtrTraits> output // [B][total_D],
    );
{%- endif %}

{% if not dense %}
#ifndef __HIP_PLATFORM_HCC__
// Support only the split-pooled TBE case
template <
    typename emb_t,
    typename cache_t,
    typename output_t,
    typename index_t,
    bool USE_LXU_CACHE
    >
__launch_bounds__(kForwardMaxThreads, 2048 / kForwardMaxThreads)
__global__ void split_embedding_codegen_forward_{{ wdesc }}_v2_kernel(
    const emb_t* __restrict__ const dev_weights,
    const emb_t* __restrict__ const uvm_weights,
    const cache_t* __restrict__ const lxu_cache_weights,
    const int32_t* __restrict__ const weights_placements,
    const uint32_t B,
    const uint32_t T,
    const bool mean_pooling,
    const uint32_t max_D_cache,
    const FixedDivisor fd_num_warps_per_table,
    const index_t* __restrict__ const indices,
    {%- if weighted %}
    const float* __restrict__ const index_weights,
    {%- endif %}
    const index_t* __restrict__ const  offsets,
    const uint32_t* __restrict__ const D_offsets,
    const int64_t* __restrict__ const weights_offsets,
    const int32_t* __restrict__ const lxu_cache_locations,
    output_t* __restrict__ const output);
#endif
{% endif %} // if not dense

{%- for nobag in [True, False] %}
{%- if not nobag or (not weighted and not vbe) %}
{%- set has_experimental = (not dense and not nobag and not vbe) %}
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
__global__ void {{ "dense" if dense else "split" }}_embedding{{ "_nobag" if nobag else "" }}_codegen_forward_{{ wdesc }}{{ vbe_desc }}_kernel(
    const pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> dev_weights,
    {%- if not dense %}
    const pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> uvm_weights,
    const pta::PackedTensorAccessor64<cache_t, 2, at::RestrictPtrTraits> lxu_cache_weights,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> weights_placements,
    {%- endif %}
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> weights_offsets,
    {%- if not nobag %}
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> D_offsets,
    {%- else %}
    int64_t D,
    {%- endif %}
    {%- if vbe %}
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> output_offsets,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> b_t_map,
    const int32_t info_B_num_bits,
    const uint32_t info_B_mask,
    {%- else %}
    FixedDivisor fd_B,
    {%- endif %}
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> indices,
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> offsets,
    {%- if not nobag %}
    int64_t pooling_mode,
    {%- endif %}
    {%- if weighted %}
    pta::PackedTensorAccessor32<at::acc_type<cache_t, true>, 1, at::RestrictPtrTraits> indice_weights,
    {%- endif %}
    {%- if not dense %}
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> lxu_cache_locations,
    {%- endif %}
    pta::PackedTensorAccessor64<output_t, 2, at::RestrictPtrTraits> output // [B][total_D]
    );

Tensor {{ "dense" if dense else "split" }}_embedding{{ "_nobag" if nobag else "" }}_codegen_forward_{{ wdesc }}{{ vbe_desc }}_cuda(
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
    {%- if vbe %}
    const VBEMetadata& vbe_metadata,
    const int32_t info_B_num_bits,
    const uint32_t info_B_mask,
    {%- endif %}
    bool is_experimental
) {
    TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
        {%- if not dense %}
        uvm_weights,
        lxu_cache_weights,
        weights_placements,
        {%- endif %}
        weights_offsets,
        {%- if not nobag %}
        D_offsets,
        {%- endif %}
        indices,
        offsets,
        {%- if weighted %}
        indice_weights,
        {%- endif %}
        {%- if not dense %}
        lxu_cache_locations,
        {%- endif %}
        {%- if vbe %}
        vbe_metadata.output_offsets,
        vbe_metadata.b_t_map,
        {%- endif %}
        dev_weights
    );

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
    const auto total_B = offsets.size(0) - 1;
    const int32_t B = (total_B) / T;
    TORCH_CHECK_GE(B, 0);
    {%- if not nobag %}
    TORCH_CHECK_GT(total_D, 0);
    TORCH_CHECK_EQ(total_D % 4, 0);
    TORCH_CHECK_LE(max_D, {{ max_embedding_dim }});
    {%- else %}
    TORCH_CHECK_GT(D, 0);
    TORCH_CHECK_EQ(D % 4, 0);
    {%- endif %}
    {%- if vbe %}
    TORCH_CHECK(vbe_metadata.output_offsets.numel() == total_B);
    TORCH_CHECK(vbe_metadata.b_t_map.numel() == total_B);
    TORCH_CHECK(vbe_metadata.output_size >= 0);
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

    {%- if vbe %}
    // Use a 2D tensor to make it compatible with 2D PackedTensorsAccessor of other output
    output = at::empty(
        {1, vbe_metadata.output_size},
        dev_weights.options().dtype(getScalarType(o_dtype))
    );
    {%- else %}
    output = at::empty(
        {B, total_adjusted_D},
        dev_weights.options().dtype(getScalarType(o_dtype))
    );
    {%- endif %}
    {%- endif %} // if nobag

    if (B == 0) {
        {%- if vbe %}
        output = output.reshape({-1});
        {%- endif %}
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

        {%- if has_experimental %}
        if (is_experimental) {
          if (std::is_same<emb_t, uint8_t>() || std::is_same<output_t, uint8_t>()) {
            is_experimental = false;
          }
        }

        if (!is_experimental) {
        {%- endif %} // if has_experimental

        {%- if not nobag %}
        {%- for use_cache in ["false", "true"] %}
        // The dense case does not have cache so we have to generate code for
        // only one case (value of use_cache/vbe does not matter)
        {%- if (not dense) or (use_cache == "true" and not vbe) %}
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

#ifdef FBGEMM_GPU_MEMCHECK
                const auto func_name = "{{ "dense" if dense else "split" }}_embedding_codegen_forward_{{ wdesc }}_kernel";
#endif

                {%- if not dense %}
                split_embedding_codegen_forward_{{ wdesc }}{{ vbe_desc }}_kernel<emb_t, cache_t, output_t, {{ use_cache }}, int64_t, kMaxVecsPerThread, kThreadGroupSize><<<
                {%- else %}
                dense_embedding_codegen_forward_{{ wdesc }}_kernel<emb_t, cache_t, output_t, int64_t, kMaxVecsPerThread, kThreadGroupSize><<<
                {%- endif %}
                    div_round_up(total_B, kForwardMaxThreads / kThreadGroupSize),
                    dim3(kThreadGroupSize, kForwardMaxThreads / kThreadGroupSize),
                    0,
                    at::cuda::getCurrentCUDAStream()>>>(
                    MAKE_PTA(dev_weights, emb_t, 1, 64),
                    {%- if not dense %}
                    MAKE_PTA(uvm_weights, emb_t, 1, 64),
                    MAKE_PTA(lxu_cache_weights, cache_t, 2, 64),
                    MAKE_PTA(weights_placements, int32_t, 1, 32),
                    {%- endif %}
                    MAKE_PTA(weights_offsets, int64_t, 1, 32),
                    MAKE_PTA(D_offsets, int32_t, 1, 32),
                    {%- if vbe %}
                    MAKE_PTA(vbe_metadata.output_offsets, int64_t, 1, 32),
                    MAKE_PTA(vbe_metadata.b_t_map, int32_t, 1, 32),
                    info_B_num_bits,
                    info_B_mask,
                    {%- else %}
                    FixedDivisor(B),
                    {%- endif %}
                    MAKE_PTA(indices, int64_t, 1, 32),
                    MAKE_PTA(offsets, int64_t, 1, 32),
                    pooling_mode,
                    {%- if weighted %}
                    MAKE_PTA_ACC(indice_weights, cache_t, 1, 32),
                    {%- endif %}
                    {%- if not dense %}
                    MAKE_PTA(lxu_cache_locations, int32_t, 1, 32),
                    {%- endif %} // if not dense
                    MAKE_PTA(output, output_t, 2, 64)
                    );
                {%- if vbe %}
                output = output.reshape({-1});
                {%- endif %}
                return;
            }
            {%- endif %}
            {%- endfor %}
        {%- if not dense %}
        } // if (use_lxu_cache == {{ use_cache }})
        {%- endif %}
        {%- endif %} // if (not dense) or (use_cache == "true" and not vbe)
        {%- endfor %} // for use_cache in ["false", "true"]
        {%- else %}
        {%- for kEmbeddingSize in [4, 8, 16, 32] %}
        if (D <= {{ kEmbeddingSize }}) {
        {%- if not dense %}
#ifdef FBGEMM_GPU_MEMCHECK
        const auto func_name = "split_embedding_nobag_codegen_forward_unweighted_small_kernel";
#endif
        split_embedding_nobag_codegen_forward_unweighted_small_kernel<emb_t, cache_t, output_t, int64_t, {{ kEmbeddingSize // 4 }}><<<
        {%- else %}
#ifdef FBGEMM_GPU_MEMCHECK
        const auto func_name = "dense_embedding_nobag_codegen_forward_unweighted_small_kernel";
#endif
        dense_embedding_nobag_codegen_forward_unweighted_small_kernel<emb_t, cache_t, output_t, int64_t, {{ kEmbeddingSize // 4 }}><<<
        {%- endif %}
            div_round_up(total_B, kForwardMaxThreads / kWarpSize),
            dim3(kWarpSize, kForwardMaxThreads / kWarpSize),
            0,
            at::cuda::getCurrentCUDAStream()>>>(
            MAKE_PTA(dev_weights, emb_t, 1, 64),
            {%- if not dense %}
            MAKE_PTA(uvm_weights, emb_t, 1, 64),
            MAKE_PTA(lxu_cache_weights, cache_t, 2, 64),
            MAKE_PTA(weights_placements, int32_t, 1, 32),
            {%- endif %}
            MAKE_PTA(weights_offsets, int64_t, 1, 32),
            D,
            FixedDivisor(B),
            MAKE_PTA(indices, int64_t, 1, 32),
            MAKE_PTA(offsets, int64_t, 1, 32),
            {%- if not dense %}
            MAKE_PTA(lxu_cache_locations, int32_t, 1, 32),
            {%- endif %}
            MAKE_PTA(output, output_t, 2, 64)
            );
            return;
        }
        {%- endfor %}
        {%- for use_cache in ["false", "true"] %}
        // The dense case does not have cache so we have to generate code for
        // only one case (value of use_cache/vbe does not matter)
        {%- if (not dense) or (use_cache == "true" and not vbe) %}
        {%- if not dense %}
        if (use_lxu_cache == {{ use_cache }}) {
#ifdef FBGEMM_GPU_MEMCHECK
            const auto func_name = "split_embedding_nobag_codegen_forward_unweighted_kernel";
#endif
            split_embedding_nobag_codegen_forward_unweighted_kernel<emb_t, cache_t, output_t, {{ use_cache }}, int64_t><<<
        {%- else %}
#ifdef FBGEMM_GPU_MEMCHECK
            const auto func_name = "dense_embedding_nobag_codegen_forward_unweighted_kernel";
#endif
            dense_embedding_nobag_codegen_forward_unweighted_kernel<emb_t, cache_t, output_t, int64_t><<<
        {%- endif %}
                div_round_up(total_B, kForwardMaxThreads / kWarpSize),
                dim3(kWarpSize, kForwardMaxThreads / kWarpSize),
                0,
                at::cuda::getCurrentCUDAStream()>>>(
                MAKE_PTA(dev_weights, emb_t, 1, 64),
                {%- if not dense %}
                MAKE_PTA(uvm_weights, emb_t, 1, 64),
                MAKE_PTA(lxu_cache_weights, cache_t, 2, 64),
                MAKE_PTA(weights_placements, int32_t, 1, 32),
                {%- endif %}
                MAKE_PTA(weights_offsets, int64_t, 1, 32),
                D,
                FixedDivisor(B),
                MAKE_PTA(indices, int64_t, 1, 32),
                MAKE_PTA(offsets, int64_t, 1, 32),
                {%- if not dense %}
                MAKE_PTA(lxu_cache_locations, int32_t, 1, 32),
                {%- endif %}
                MAKE_PTA(output, output_t, 2, 64)
                );
                return;
        {%- if not dense %}
        } // if (use_lxu_cache == {{ use_cache }})
        {%- endif %}
        {%- endif %} // if (not dense) or (use_cache == "true" and not vbe)
        {%- endfor %} // for use_cache in ["false", "true"]
        {%- endif %}
        {%- if has_experimental %}
        } // if (!is_experimental)
        else {
#ifdef __HIP_PLATFORM_HCC__
            TORCH_CHECK(false, "is_experimental=True is not supported in ROCm");
#else
            // Allocate num warps per table based on max_D
            const int num_warps_per_table = B * div_round_up(max_D, kWarpSize * 4);
            const uint32_t num_warps_per_threadblock = kForwardMaxThreads / kWarpSize;

            const auto split_embedding_codegen_forward_{{ wdesc }}_v2_kernel_ =
                (use_lxu_cache ? split_embedding_codegen_forward_{{ wdesc }}_v2_kernel<emb_t, cache_t, output_t, int64_t, true>
                               : split_embedding_codegen_forward_{{ wdesc }}_v2_kernel<emb_t, cache_t, output_t, int64_t, false>);

            split_embedding_codegen_forward_{{ wdesc }}_v2_kernel_
              <<<div_round_up(T * num_warps_per_table, num_warps_per_threadblock),
              dim3(kWarpSize, num_warps_per_threadblock),
              0,
              at::cuda::getCurrentCUDAStream()>>>(
                  dev_weights.data_ptr<emb_t>(),
                  uvm_weights.data_ptr<emb_t>(),
                  lxu_cache_weights.data_ptr<cache_t>(),
                  weights_placements.data_ptr<int32_t>(),
                  B,
                  T,
                  static_cast<PoolingMode>(pooling_mode) == PoolingMode::MEAN,
                  use_lxu_cache ? lxu_cache_weights.size(1) : 0,
                  FixedDivisor(num_warps_per_table),
                  indices.data_ptr<int64_t>(),
                  {%- if weighted %}
                  // TODO: update indice_weights type
                  indice_weights.data_ptr<float>(),
                  {%- endif %}
                  offsets.data_ptr<int64_t>(),
                  reinterpret_cast<uint32_t*>(D_offsets.data_ptr<int32_t>()),
                  weights_offsets.data_ptr<int64_t>(),
                  lxu_cache_locations.data_ptr<int32_t>(),
                  output.data_ptr<output_t>()
                  );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
#endif
        }
        {%- endif %} // if has_experimental
        });

  return output;
}
{%- endif %}
{%- endfor %}
    // clang-format on
