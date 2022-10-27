/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// clang-format off
#include "codegen/embedding_forward_template_helpers.cuh"

using Tensor = at::Tensor;
using namespace fbgemm_gpu;

{% if not dense %}
constexpr int32_t kCacheLocationMissing = -1;
{% endif %}

constexpr size_t kForwardMaxThreads = 512;

// TODO: optimization to use multiple warps per row.
template <typename emb_t, typename grad_t, typename cache_t, size_t kMaxVecsPerThread>
__global__
__launch_bounds__(kForwardMaxThreads) void {{ "dense" if dense else "split" }}_embedding_codegen_grad_indice_weights_kernel(
    // [\sum_t E_t x D_t]
    const at::PackedTensorAccessor64<grad_t, 2, at::RestrictPtrTraits>
        grad_output,
    at::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> dev_weights,
    {% if not dense %}
    at::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> uvm_weights,
    at::PackedTensorAccessor64<cache_t, 2, at::RestrictPtrTraits> lxu_cache_weights,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        weights_placements,
    {% endif %}
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> weights_offsets,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> D_offsets,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        indices, // [N = \sum_{b,t} L_{b,t} total indices, i.e. flattened
                 // [B][T][L]
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        offsets, // [B x T + 1]
    {% if not dense %}
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        lxu_cache_locations,
    {% endif %}
    at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        feature_requires_grad, // [T],
    at::PackedTensorAccessor32<at::acc_type<cache_t, true>, 1, at::RestrictPtrTraits>
        grad_indice_weights
    ) {
    int32_t B = grad_output.size(0);
    int32_t T = D_offsets.size(0) - 1;
    int32_t b_t = blockIdx.x * blockDim.y + threadIdx.y;
    int32_t t = b_t / B;
    int32_t b = b_t % B;

    if (b_t >= B * T) {
        return;
    }
    int64_t weights_offset = weights_offsets[t];
    int32_t D_start = D_offsets[t];
    int32_t D_end = D_offsets[t + 1];
    int32_t D = D_end - D_start;
    int64_t indices_start = offsets[t * B + b];
    int64_t indices_end = offsets[t * B + b + 1];
    int32_t L = indices_end - indices_start;
    if (feature_requires_grad.size(0) > 0 && !feature_requires_grad[t]) {
        // If the table does not require gradient computation, we set the gradient to zero.
        for (int32_t l_start = 0; l_start < L; l_start += kWarpSize) {
            int32_t l = l_start + threadIdx.x;
            if (l < L) {
                grad_indice_weights[indices_start + l] = 0.0;
            }
        }
        return;
    }

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


    Vec4T<at::acc_type<cache_t, true>> grad_out[kMaxVecsPerThread];
    #pragma unroll kMaxVecsPerThread
    for (int32_t i = 0;
        i < kMaxVecsPerThread && 4 * kWarpSize * i + threadIdx.x * 4 < D;
        ++i) {
        int32_t d = 4 * kWarpSize * i + threadIdx.x * 4;
        Vec4T<at::acc_type<grad_t, true>> go((&grad_output[b][0]) + D_start + d);
        grad_out[i] = go;
    }

    for (int32_t l_start = 0; l_start < L; l_start += kWarpSize) {
        int32_t l = l_start + threadIdx.x;
        int64_t idx = l < L ? indices[indices_start + l] : 0;
        {% if not dense %}
        int32_t cache_idx = (placement == PlacementType::MANAGED_CACHING && l < L) ? lxu_cache_locations[indices_start + l] : 0;
        {% endif %}
        for (auto j = 0; j < kWarpSize && l_start + j < L; ++j) {
            int64_t idx_j = shfl_sync(idx, j);
            {% if not dense %}
            int32_t cache_idx_j = shfl_sync(cache_idx, j);
            {% endif %}
            at::acc_type<cache_t, true> grad_indice_weight = 0.0;

        #pragma unroll kMaxVecsPerThread
            for (int32_t i = 0;
                i < kMaxVecsPerThread && 4 * kWarpSize * i + threadIdx.x * 4 < D;
                ++i) {
                int32_t d = 4 * kWarpSize * i + threadIdx.x * 4;
                {% if not dense %}
                if (placement == PlacementType::MANAGED_CACHING && cache_idx_j != kCacheLocationMissing) {
                    Vec4T<cache_t> weight(&lxu_cache_weights[cache_idx_j][d]);
                    grad_indice_weight += weight.acc.x * grad_out[i].acc.x +
                        weight.acc.y * grad_out[i].acc.y +
                        weight.acc.z * grad_out[i].acc.z + weight.acc.w * grad_out[i].acc.w;
                } else {
                    int32_t D_emb = D;
                    if (std::is_same<emb_t, uint8_t>::value) {
                        D_emb += kINT8QparamsBytes;
                    }
                    auto weight_row = WeightRow<emb_t, cache_t, at::acc_type<cache_t, true>>(
                        const_cast<emb_t*>(&weights[idx_j * D_emb]),
                        nullptr,
                        D,
                        nullptr);
                    float2 qparams;
                    if (std::is_same<emb_t, uint8_t>::value) {
                        qparams = weight_row.load_qparams();
                    }
                    Vec4T<at::acc_type<cache_t, true>> weight =
                    weight_row.load(d, qparams);
                    grad_indice_weight += weight.acc.x * grad_out[i].acc.x +
                        weight.acc.y * grad_out[i].acc.y +
                        weight.acc.z * grad_out[i].acc.z + weight.acc.w * grad_out[i].acc.w;
                }
                {% else %}
                int32_t D_emb = D;
                if (std::is_same<emb_t, uint8_t>::value) {
                    D_emb += kINT8QparamsBytes;
                }
                auto weight_row = WeightRow<emb_t, cache_t, at::acc_type<cache_t, true>>(
                    const_cast<emb_t*>(&weights[idx_j * D_emb]),
                    nullptr,
                    D,
                    nullptr);
                float2 qparams;
                if (std::is_same<emb_t, uint8_t>::value) {
                    qparams = weight_row.load_qparams();
                }
                Vec4T<at::acc_type<cache_t, true>> weight =
                weight_row.load(d, qparams);
                grad_indice_weight += weight.acc.x * grad_out[i].acc.x +
                    weight.acc.y * grad_out[i].acc.y +
                    weight.acc.z * grad_out[i].acc.z + weight.acc.w * grad_out[i].acc.w;
                {% endif %}
            }
            grad_indice_weight =
                warpReduceAllSum<at::acc_type<cache_t, true>>(grad_indice_weight);
            if (threadIdx.x == 0) {
                grad_indice_weights[indices_start + l_start + j] = grad_indice_weight;
            }
        }
    }
}

Tensor {{ "dense" if dense else "split" }}_embedding_codegen_grad_indice_weights_cuda(
    Tensor grad_output,
    Tensor dev_weights,
    {% if not dense %}
    Tensor uvm_weights,
    Tensor lxu_cache_weights,
    Tensor weights_placements,
    {% endif %}
    Tensor weights_offsets,
    Tensor D_offsets,
    int64_t max_D,
    Tensor indices,
    Tensor offsets,
    {% if not dense %}
    Tensor lxu_cache_locations,
    {% endif %}
    Tensor feature_requires_grad) {
    TENSOR_ON_CUDA_GPU(grad_output);
    TENSOR_ON_CUDA_GPU(dev_weights);
    {% if not dense %}
    TENSOR_ON_CUDA_GPU(uvm_weights);
    TENSOR_ON_CUDA_GPU(lxu_cache_weights);
    TENSOR_ON_CUDA_GPU(weights_placements);
    {% endif %}
    TENSOR_ON_CUDA_GPU(weights_offsets);
    TENSOR_ON_CUDA_GPU(D_offsets);
    TENSOR_ON_CUDA_GPU(indices);
    TENSOR_ON_CUDA_GPU(offsets);
    {% if not dense %}
    TENSOR_ON_CUDA_GPU(lxu_cache_locations);
    {% endif %}
    if (feature_requires_grad.defined()) {
        TENSOR_ON_CUDA_GPU(feature_requires_grad);
    }

    at::cuda::OptionalCUDAGuard device_guard;
    device_guard.set_index(dev_weights.get_device());
    const auto T = D_offsets.size(0) - 1;
    TORCH_CHECK(T > 0);
    // offsets = [B x T  + 1]
    const auto B = (offsets.size(0) - 1) / T;
    TORCH_CHECK(B >= 0);
    TORCH_CHECK(max_D <= {{ max_embedding_dim }});
    auto grad_indice_weights = empty_like(indices, indices.options().dtype(at::toAccumulateType(grad_output.scalar_type(), true)));
    if (B == 0) {
      return grad_indice_weights;
    }
    feature_requires_grad = feature_requires_grad.defined() ? feature_requires_grad : at::empty({0}, indices.options().dtype(at::kInt));
    DISPATCH_EMB_GRAD_CACHE_TYPES(
        dev_weights.scalar_type(),
        grad_output.scalar_type(),
        {% if not dense %}
        lxu_cache_weights.scalar_type(),
        {% else %}
        dev_weights.scalar_type(),
        {% endif %}
        "split_embedding_codegen_grad_indice_weights_kernel",
        [&] {
            {% for kMaxVecsPerThread in range(1, max_embedding_dim // items_per_warp + 1) %}
            if (max_D <= {{ items_per_warp * kMaxVecsPerThread }}) {
            {{ "dense" if dense else "split" }}_embedding_codegen_grad_indice_weights_kernel<
                emb_t,
                grad_t,
                cache_t,
                {{ kMaxVecsPerThread }}><<<
                div_round_up((B * T), kForwardMaxThreads / kWarpSize),
                dim3(kWarpSize, kForwardMaxThreads / kWarpSize),
                0,
                at::cuda::getCurrentCUDAStream()>>>(
                grad_output.packed_accessor64<grad_t, 2, at::RestrictPtrTraits>(),
                dev_weights.packed_accessor64<emb_t, 1, at::RestrictPtrTraits>(),
                {% if not dense %}
                uvm_weights.packed_accessor64<emb_t, 1, at::RestrictPtrTraits>(),
                lxu_cache_weights.packed_accessor64<cache_t, 2, at::RestrictPtrTraits>(),
                weights_placements.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                {% endif %}
                weights_offsets.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                D_offsets.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                indices.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                offsets.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                {% if not dense %}
                lxu_cache_locations.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                {% endif %}
                feature_requires_grad.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                grad_indice_weights.packed_accessor32<at::acc_type<grad_t, true>, 1, at::RestrictPtrTraits>()
            );
            return;
            }
            {% endfor %}
        });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grad_indice_weights;
}
    // clang-format on
