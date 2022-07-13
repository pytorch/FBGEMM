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

{% set wdesc =  "weighted" if weighted else "unweighted" %}
#include "codegen/embedding_forward_template_helpers.cuh"

{% if not dense %}
constexpr int32_t kCacheLocationMissing = -1;
{% endif %}

constexpr size_t kForwardMaxThreads = 512;

using Tensor = at::Tensor;
using namespace fbgemm_gpu;

{% if not weighted %}
template <
    typename emb_t,
    typename cache_t,
    {% if not dense %}
    typename output_t,
    {% endif %}
    typename index_t,
    size_t kThreadGroupSize
    >
__launch_bounds__(kForwardMaxThreads)
__global__ void {{ "dense" if dense else "split" }}_embedding_nobag_codegen_forward_unweighted_small_kernel(
    const at::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> dev_weights,
    {% if not dense %}
    const at::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> uvm_weights,
    const at::PackedTensorAccessor64<cache_t, 2, at::RestrictPtrTraits>
        lxu_cache_weights,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        weights_placements,
    {% endif %}
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> weights_offsets,
    int64_t D,
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> indices,
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> offsets,
    {% if not dense %}
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        lxu_cache_locations,
    at::PackedTensorAccessor32<output_t, 2, at::RestrictPtrTraits>
        output // [B][total_D],
    {% else %}
    at::PackedTensorAccessor32<at::acc_type<cache_t,true>, 2, at::RestrictPtrTraits>
        output // [B][total_D],
    {% endif %}
    ) {
    int32_t T = weights_offsets.size(0);
    int32_t B = (offsets.size(0) - 1) / T;
    int32_t b_t = blockIdx.x * blockDim.y + threadIdx.y;
    int32_t t = b_t / B;
    int32_t b = b_t % B;

    if (b_t >= B * T) {
        return;
    }
    int64_t weights_offset = weights_offsets[t];
    index_t indices_start = offsets[t * B + b];
    index_t indices_end = offsets[t * B + b + 1];
    int32_t L = indices_end - indices_start;
    const emb_t* __restrict__ weights;
    {% if not dense %}
    const auto placement = static_cast<PlacementType>(weights_placements[t]);
    if (placement == PlacementType::DEVICE) {
        weights = &dev_weights[weights_offset];
    } else {
        weights = &uvm_weights[weights_offset];
    }
    {% else %}
    weights = &dev_weights[weights_offset];
    {% endif %}

    int32_t D_emb = D;
    if (std::is_same<emb_t, uint8_t>::value) {
        D_emb += kINT8QparamsBytes;
    }

    constexpr int32_t kNumThreadGroup = kWarpSize / kThreadGroupSize;
    const int32_t group_start = threadIdx.x / kThreadGroupSize * kThreadGroupSize;
    const int32_t group_end = group_start + kThreadGroupSize;
    const int32_t d = threadIdx.x % kThreadGroupSize * 4;

    for (int32_t l_start = 0; l_start < L; l_start += kWarpSize) {
        int32_t l = l_start + threadIdx.x;
        int64_t idx = l < L ? indices[indices_start + l] : 0;
        {% if not dense %}
        int32_t cache_idx = (placement == PlacementType::MANAGED_CACHING && l < L) ? lxu_cache_locations[indices_start + l] : 0;
        {% endif %}
        for (auto j = group_start; j < group_end && l_start + j < L; ++j) {
            int64_t idx_j = shfl_sync(idx, j);
            int64_t output_j = indices_start + l_start + j;
            {% if not dense %}
            int32_t cache_idx_j = shfl_sync(cache_idx, j);
            {% endif %}

            {% if not dense %}
            auto weight_row_cache = WeightRow<emb_t, cache_t, cache_t>(
                const_cast<emb_t*>(&weights[idx_j * D_emb]),
                const_cast<cache_t*>(&lxu_cache_weights[cache_idx_j][0]),
                D,
                nullptr);
            float2 qparams_cache; // assume cache is fp16/fp32 which doesn't require qparams

            {% endif %}
            auto weight_row_emb = WeightRow<emb_t, cache_t, cache_t>(
                const_cast<emb_t*>(&weights[idx_j * D_emb]),
                nullptr,
                D,
                nullptr);
            float2 qparams_emb;
            if (std::is_same<emb_t, uint8_t>::value) {
                qparams_emb = weight_row_emb.load_qparams();
            }

            if (d < D) {
                {% if not dense %}
                if (placement == PlacementType::MANAGED_CACHING && cache_idx_j != kCacheLocationMissing) {
                    Vec4T<cache_t> weight = weight_row_cache.load(d, qparams_cache);
                    weight.store(&output[output_j][d]);
                } else {
                    Vec4T<cache_t> weight = weight_row_emb.load(d, qparams_emb);
                    weight.store(&output[output_j][d]);
                }
                {% else %}
                    Vec4T<cache_t> weight = weight_row_emb.load(d, qparams_emb);
                    weight.store(&output[output_j][d]);
                {% endif %}
            }
        }
    }
}
{% endif %}

{% for nobag in [True, False] %}
{% if not nobag or not weighted %}
template <
    typename emb_t,
    typename cache_t,
    {% if not dense %}
    typename output_t,
    {% endif %}
    typename index_t
    {% if not nobag %}
    ,size_t kMaxVecsPerThread
    {% endif %}
    >
__launch_bounds__(kForwardMaxThreads)
__global__ void {{ "dense" if dense else "split" }}_embedding{{ "_nobag" if nobag else "" }}_codegen_forward_{{ wdesc }}_kernel(
    const at::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> dev_weights,
    {% if not dense %}
    const at::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> uvm_weights,
    const at::PackedTensorAccessor64<cache_t, 2, at::RestrictPtrTraits>
        lxu_cache_weights,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        weights_placements,
    {% endif %}
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> weights_offsets,
    {% if not nobag %}
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> D_offsets,
    {% else %}
    int64_t D,
    {% endif %}
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> indices,
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> offsets,
    {% if not nobag %}
    int64_t pooling_mode,
    {% endif %}
    {% if weighted %}
    at::PackedTensorAccessor32<at::acc_type<cache_t, true>, 1, at::RestrictPtrTraits>
        indice_weights,
    {% endif %}
    {% if not dense %}
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        lxu_cache_locations,
    at::PackedTensorAccessor32<output_t, 2, at::RestrictPtrTraits>
        output // [B][total_D],
    {% else %}
    at::PackedTensorAccessor32<at::acc_type<cache_t,true>, 2, at::RestrictPtrTraits>
        output // [B][total_D],
    {% endif %}
    ) {
    {% if not nobag %}
    int32_t B = output.size(0);
    int32_t T = D_offsets.size(0) - 1;
    {% else %}
    int32_t T = weights_offsets.size(0);
    int32_t B = (offsets.size(0) - 1) / T;
    {% endif %}
    int32_t b_t = blockIdx.x * blockDim.y + threadIdx.y;
    int32_t t = b_t / B;
    int32_t b = b_t % B;

    if (b_t >= B * T) {
        return;
    }
    int64_t weights_offset = weights_offsets[t];
    {% if not nobag %}
    int32_t D_start = D_offsets[t];
    int32_t D_end = D_offsets[t + 1];
    int32_t D = D_end - D_start;
    {% endif %}
    index_t indices_start = offsets[t * B + b];
    index_t indices_end = offsets[t * B + b + 1];
    int32_t L = indices_end - indices_start;
    const emb_t* __restrict__ weights;
    {% if not dense %}
    const auto placement = static_cast<PlacementType>(weights_placements[t]);
    if (placement == PlacementType::DEVICE) {
        weights = &dev_weights[weights_offset];
    } else {
        weights = &uvm_weights[weights_offset];
    }
    {% else %}
    weights = &dev_weights[weights_offset];
    {% endif %}

    int32_t D_emb = D;
    if (std::is_same<emb_t, uint8_t>::value) {
        D_emb += kINT8QparamsBytes;
    }

    {% if not nobag %}
    Vec4T<cache_t> accumulators[kMaxVecsPerThread];
    {% endif %}
    for (int32_t l_start = 0; l_start < L; l_start += kWarpSize) {
        int32_t l = l_start + threadIdx.x;
        int64_t idx = l < L ? indices[indices_start + l] : 0;
        {% if not dense %}
        int32_t cache_idx = (placement == PlacementType::MANAGED_CACHING && l < L) ? lxu_cache_locations[indices_start + l] : 0;
        {% endif %}
        {% if weighted %}
        at::acc_type<cache_t, true> idx_weight = l < L ? indice_weights[indices_start + l] : 0;
        {% endif %}
        for (auto j = 0; j < kWarpSize && l_start + j < L; ++j) {
            int64_t idx_j = shfl_sync(idx, j);
            {% if nobag %}
            int64_t output_j = indices_start + l_start + j;
            {% endif %}
            {% if not dense %}
            int32_t cache_idx_j = shfl_sync(cache_idx, j);
            {% endif %}

            {% if weighted %}
            at::acc_type<cache_t, true> idx_weight_j = shfl_sync(idx_weight, j);
            {% endif %}

            {% if not dense %}
            auto weight_row_cache = WeightRow<emb_t, cache_t, cache_t>(
                const_cast<emb_t*>(&weights[idx_j * D_emb]),
                const_cast<cache_t*>(&lxu_cache_weights[cache_idx_j][0]),
                D,
                nullptr);
            float2 qparams_cache; // assume cache is fp16/fp32 which doesn't require qparams

            {% endif %}
            auto weight_row_emb = WeightRow<emb_t, cache_t, cache_t>(
                const_cast<emb_t*>(&weights[idx_j * D_emb]),
                nullptr,
                D,
                nullptr);
            float2 qparams_emb;
            if (std::is_same<emb_t, uint8_t>::value) {
                qparams_emb = weight_row_emb.load_qparams();
            }

            {% if not nobag %}
            #pragma unroll kMaxVecsPerThread
            for (int32_t i = 0;
                i < kMaxVecsPerThread && 4 * kWarpSize * i + threadIdx.x * 4 < D;
                ++i) {
                int32_t d = 4 * kWarpSize * i + threadIdx.x * 4;
                {% if not dense %}
                if (placement == PlacementType::MANAGED_CACHING && cache_idx_j != kCacheLocationMissing) {
                    Vec4T<cache_t> weight = weight_row_cache.load(d, qparams_cache);
                    {% if weighted %}
                    accumulators[i].fma_(weight, idx_weight_j);
                    {% else %}
                    accumulators[i].add_(weight);
                    {% endif %}
                } else {
                    Vec4T<cache_t> weight = weight_row_emb.load(d, qparams_emb);
                    {% if weighted %}
                    accumulators[i].fma_(weight, idx_weight_j);
                    {% else %}
                    accumulators[i].add_(weight);
                    {% endif %}
                }
                {% else %}
                    Vec4T<cache_t> weight = weight_row_emb.load(d, qparams_emb);
                    {% if weighted %}
                    accumulators[i].fma_(weight, idx_weight_j);
                    {% else %}
                    accumulators[i].add_(weight);
                    {% endif %}
                {% endif %}
            }
            {% else %}
            for (int32_t i = 0; i < D; i+=4 * kWarpSize) {
                int32_t d = i + threadIdx.x * 4;
                if (d < D) {
                    {% if not dense %}
                    if (placement == PlacementType::MANAGED_CACHING && cache_idx_j != kCacheLocationMissing) {
                        Vec4T<cache_t> weight = weight_row_cache.load(d, qparams_cache);
                        weight.store(&output[output_j][d]);
                    } else {
                        Vec4T<cache_t> weight = weight_row_emb.load(d, qparams_emb);
                        weight.store(&output[output_j][d]);
                    }
                    {% else %}
                        Vec4T<cache_t> weight = weight_row_emb.load(d, qparams_emb);
                        weight.store(&output[output_j][d]);
                    {% endif %}
                }
            }
            {% endif %}
        }
    }

    {% if not nobag %}
    {% if not dense %}
    if (!std::is_same<output_t, uint8_t>::value) {
        #pragma unroll kMaxVecsPerThread
        for (int32_t i = 0;
        i < kMaxVecsPerThread && 4 * kWarpSize * i + threadIdx.x * 4 < D;
        ++i) {
            if (static_cast<PoolingMode>(pooling_mode) == PoolingMode::MEAN && L != 0) {
                accumulators[i].mul_(1.0 / L);
            }
            int32_t d = 4 * kWarpSize * i + threadIdx.x * 4;
            accumulators[i].store(&output[b][D_start + d]);
        }
    } else {
        // apply per feature row-wise int8
        float thread_local_min = std::numeric_limits<float>::max();
        float thread_local_max = std::numeric_limits<float>::lowest();
        float2 qparams;

        #pragma unroll kMaxVecsPerThread
        for (int32_t i = 0;
            i < kMaxVecsPerThread && 4 * kWarpSize * i + threadIdx.x * 4 < D;
            ++i) {
            if (static_cast<PoolingMode>(pooling_mode) == PoolingMode::MEAN && L != 0) {
                accumulators[i].mul_(1.0 / L);
            }
            thread_local_max = max(thread_local_max, vec4_max(accumulators[i]));
            thread_local_min = min(thread_local_max, vec4_min(accumulators[i]));
        }

        qparams = warp_find_qparams(thread_local_min, thread_local_max);
        int output_D_start = D_start + t * 8;
        int output_D_end = output_D_start + D;

        #pragma unroll kMaxVecsPerThread
        for (int32_t i = 0;
            i < kMaxVecsPerThread && 4 * kWarpSize * i + threadIdx.x * 4 < D;
            ++i) {
            int32_t d = 4 * kWarpSize * i + threadIdx.x * 4;
            nearest_rounding_vector<output_t, cache_t>(&output[b][output_D_start + d], accumulators[i], qparams);
        }
        if (threadIdx.x == 0) {
            store_qparams_to_row(&output[b][output_D_end], qparams);
        }

    }
    {% else %}
    // no pooled embedding quantization fusion for dense embeddings
    #pragma unroll kMaxVecsPerThread
    for (int32_t i = 0;
        i < kMaxVecsPerThread && 4 * kWarpSize * i + threadIdx.x * 4 < D;
        ++i) {
        int32_t d = 4 * kWarpSize * i + threadIdx.x * 4;
        if (static_cast<PoolingMode>(pooling_mode) == PoolingMode::MEAN && L != 0) {
            accumulators[i].mul_(1.0 / L);
        }
        accumulators[i].store(&output[b][D_start + d]);
    }
    {% endif %}
    {% endif %}
}

Tensor {{ "dense" if dense else "split" }}_embedding{{ "_nobag" if nobag else "" }}_codegen_forward_{{ wdesc }}_cuda(
    Tensor dev_weights,
    {% if not dense %}
    Tensor uvm_weights,
    Tensor lxu_cache_weights,
    Tensor weights_placements,
    {% endif %}
    Tensor weights_offsets,
    {% if not nobag %}
    Tensor D_offsets,
    int64_t total_D,
    int64_t max_D,
    {% else %}
    int64_t D,
    {% endif %}
    Tensor indices,
    Tensor offsets,
    {% if not nobag %}
    int64_t pooling_mode,
    {% endif %}
    {% if weighted %}
    Tensor indice_weights,
    {% endif %}
    {% if not dense %}
    Tensor lxu_cache_locations,
    {% endif %}
    {% if not dense %}
    int64_t output_dtype,
    {% endif %}
    int64_t unused
) {
    TENSOR_ON_CUDA_GPU(dev_weights);
    {% if not dense %}
    TENSOR_ON_CUDA_GPU(uvm_weights);
    TENSOR_ON_CUDA_GPU(lxu_cache_weights);
    TENSOR_ON_CUDA_GPU(weights_placements);
    {% endif %}
    TENSOR_ON_CUDA_GPU(weights_offsets);
    {% if not nobag %}
    TENSOR_ON_CUDA_GPU(D_offsets);
    {% endif %}
    TENSOR_ON_CUDA_GPU(indices);
    TENSOR_ON_CUDA_GPU(offsets);
    {% if weighted %}
    TENSOR_ON_CUDA_GPU(indice_weights);
    {% endif %}
    {% if not dense %}
    TENSOR_ON_CUDA_GPU(lxu_cache_locations);
    {% endif %}

    at::cuda::OptionalCUDAGuard device_guard;
    device_guard.set_index(dev_weights.get_device());

    {% if not nobag %}
    int32_t T = D_offsets.numel() - 1;
    {% else %}
    int32_t total_L = indices.numel();
    int32_t T = weights_offsets.numel();
    {% endif %}
    TORCH_CHECK(T > 0);
    // offsets = [B x T  + 1]
    int32_t B = (offsets.size(0) - 1) / T;
    TORCH_CHECK(B >= 0);
    {% if not nobag %}
    TORCH_CHECK(total_D > 0);
    TORCH_CHECK(total_D % 4 == 0);
    TORCH_CHECK(max_D <= {{ max_embedding_dim }});
    {% else %}
    TORCH_CHECK(D > 0);
    TORCH_CHECK(D % 4 == 0);
    {% endif %}

    Tensor output;
    {% if nobag %}
    {% if dense %}
        output = at::empty({total_L, D}, dev_weights.options().dtype(at::kFloat));
    {% else %}
    SparseType o_dtype = static_cast<SparseType>(output_dtype);
    TORCH_CHECK(o_dtype == SparseType::FP32 || o_dtype == SparseType::FP16 ||
                o_dtype == SparseType::BF16 || o_dtype == SparseType::INT8);
    int64_t adjusted_D = D;
    if (o_dtype == SparseType::INT8) {
        adjusted_D += T * kINT8QparamsBytes;
    }
    output = at::empty({total_L, adjusted_D}, dev_weights.options().dtype(getScalarType(o_dtype)));
    {% endif %}
    {% else %}
    {% if dense %}
    if (dev_weights.scalar_type() == at::kHalf || dev_weights.scalar_type() == at::kByte) {
        output = at::empty({B, total_D}, dev_weights.options().dtype(at::kFloat));
    } else {
        output = at::empty({B, total_D}, dev_weights.options());
    }
    {% else %}
    SparseType o_dtype = static_cast<SparseType>(output_dtype);
    TORCH_CHECK(o_dtype == SparseType::FP32 || o_dtype == SparseType::FP16 ||
                o_dtype == SparseType::BF16 || o_dtype == SparseType::INT8);
    int64_t total_adjusted_D = total_D;
    if (o_dtype == SparseType::INT8) {
        total_adjusted_D += T * kINT8QparamsBytes;
    }
    output = at::empty({B, total_adjusted_D}, dev_weights.options().dtype(getScalarType(o_dtype)));

    {% endif %}
    {% endif %}

    if (B == 0) {
        return output;
    }

    {% if not dense %}
    DISPATCH_EMB_CACHE_OUTPUT_TYPES(
    {% else %}
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    {% endif %}
        dev_weights.scalar_type(),
        {% if not dense %}
        lxu_cache_weights.scalar_type(),
        output.scalar_type(),
        {% endif %}
        "batched_embedding{{ "_nobag" if nobag else "" }}_forward_kernel_2", [&] {
        {% if not nobag %}
        {% for kMaxVecsPerThread in range(1, max_embedding_dim // 128 + 1) %}
        if (max_D <= {{ 128 * kMaxVecsPerThread }}) {
            {% if not dense %}
            split_embedding_codegen_forward_{{ wdesc }}_kernel<emb_t, cache_t, output_t, int64_t, {{ kMaxVecsPerThread }}><<<
            {% else %}
            dense_embedding_codegen_forward_{{ wdesc }}_kernel<scalar_t, scalar_t, int64_t, {{ kMaxVecsPerThread }}><<<
            {% endif %}
                div_round_up((B * T), kForwardMaxThreads / kWarpSize),
                dim3(kWarpSize, kForwardMaxThreads / kWarpSize),
                0,
                at::cuda::getCurrentCUDAStream()>>>(
                dev_weights.packed_accessor64<{{ "scalar_t" if dense else "emb_t" }}, 1, at::RestrictPtrTraits>(),
                {% if not dense %}
                uvm_weights.packed_accessor64<emb_t, 1, at::RestrictPtrTraits>(),
                lxu_cache_weights.packed_accessor64<cache_t, 2, at::RestrictPtrTraits>(),
                weights_placements.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                {% endif %}
                weights_offsets.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                D_offsets.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                indices.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                offsets.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                pooling_mode,
                {% if weighted %}
                indice_weights.packed_accessor32<at::acc_type<{{ "scalar_t" if dense else "cache_t" }}, true>, 1, at::RestrictPtrTraits>(),
                {% endif %}
                {% if not dense %}
                lxu_cache_locations.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                output.packed_accessor32<
                    output_t,
                    2,
                    at::RestrictPtrTraits>()
                );
                {% else %}
                output.packed_accessor32<
                    at::acc_type<scalar_t, true>,
                    2,
                    at::RestrictPtrTraits>()
                );
                {% endif %}

            return;
        }
        {% endfor %}
        {% else %}
        {% for kEmbeddingSize in [4, 8, 16, 32] %}
        if (D <= {{ kEmbeddingSize }}) {
        {% if not dense %}
        split_embedding_nobag_codegen_forward_unweighted_small_kernel<emb_t, cache_t, output_t, int64_t, {{ kEmbeddingSize // 4 }}><<<
        {% else %}
        dense_embedding_nobag_codegen_forward_unweighted_small_kernel<scalar_t, scalar_t, int64_t, {{ kEmbeddingSize // 4 }}><<<
        {% endif %}
            div_round_up((B * T), kForwardMaxThreads / kWarpSize),
            dim3(kWarpSize, kForwardMaxThreads / kWarpSize),
            0,
            at::cuda::getCurrentCUDAStream()>>>(
            dev_weights.packed_accessor64<{{ "scalar_t" if dense else "emb_t" }}, 1, at::RestrictPtrTraits>(),
            {% if not dense %}
            uvm_weights.packed_accessor64<emb_t, 1, at::RestrictPtrTraits>(),
            lxu_cache_weights.packed_accessor64<cache_t, 2, at::RestrictPtrTraits>(),
            weights_placements.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
            {% endif %}
            weights_offsets.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
            D,
            indices.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
            offsets.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
            {% if not dense %}
            lxu_cache_locations.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
            output.packed_accessor32<
                output_t,
                2,
                at::RestrictPtrTraits>()
            );
            {% else %}
            output.packed_accessor32<
                at::acc_type<scalar_t, true>,
                2,
                at::RestrictPtrTraits>()
            );
            {% endif %}

            return;
        }
        {% endfor %}
        {% if not dense %}
        split_embedding_nobag_codegen_forward_unweighted_kernel<emb_t, cache_t, output_t, int64_t><<<
        {% else %}
        dense_embedding_nobag_codegen_forward_unweighted_kernel<scalar_t, scalar_t, int64_t><<<
        {% endif %}
            div_round_up((B * T), kForwardMaxThreads / kWarpSize),
            dim3(kWarpSize, kForwardMaxThreads / kWarpSize),
            0,
            at::cuda::getCurrentCUDAStream()>>>(
            dev_weights.packed_accessor64<{{ "scalar_t" if dense else "emb_t" }}, 1, at::RestrictPtrTraits>(),
            {% if not dense %}
            uvm_weights.packed_accessor64<emb_t, 1, at::RestrictPtrTraits>(),
            lxu_cache_weights.packed_accessor64<cache_t, 2, at::RestrictPtrTraits>(),
            weights_placements.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
            {% endif %}
            weights_offsets.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
            D,
            indices.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
            offsets.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
            {% if not dense %}
            lxu_cache_locations.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
            output.packed_accessor32<
                output_t,
                2,
                at::RestrictPtrTraits>()
            );
            {% else %}
            output.packed_accessor32<
                at::acc_type<scalar_t, true>,
                2,
                at::RestrictPtrTraits>()
            );
            {% endif %}

            return;
        {% endif %}
        });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
{% endif %}
{% endfor %}
    // clang-format on
