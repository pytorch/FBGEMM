/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// clang-format off

{%- set ddesc =  "dense" if dense else "split" %}

////////////////////////////////////////////////////////////////////////////////
// Required for op registrations
#include "fbgemm_gpu/embedding_op_registration.h"
////////////////////////////////////////////////////////////////////////////////
#include "codegen/embedding_forward_template_helpers.cuh"

using Tensor = at::Tensor;
using namespace fbgemm_gpu;

{%- for vbe in [True, False] %}
{%- set vdesc = "_vbe" if vbe else "" %}
{%- if not dense or not vbe %}

// TODO: optimization to use multiple warps per row.
template <
  typename emb_t,
  typename grad_t,
  typename cache_t,
  size_t kMaxVecsPerThread
>
__global__ __launch_bounds__(kForwardMaxThreads) void
{{ ddesc }}_embedding_codegen_grad_indice_weights{{ vdesc }}_kernel(
    // [\sum_t E_t x D_t]
    const pta::PackedTensorAccessor64<grad_t, 2, at::RestrictPtrTraits> grad_output,
    pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> dev_weights,
    {%- if not dense %}
    pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> uvm_weights,
    pta::PackedTensorAccessor64<cache_t, 2, at::RestrictPtrTraits> lxu_cache_weights,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> weights_placements,
    {%- endif %}
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> weights_offsets,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> D_offsets,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> indices, // [N = \sum_{b,t} L_{b,t} total indices, i.e. flattened [B][T][L]
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> offsets, // [B x T + 1]
    {%- if not dense %}
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> lxu_cache_locations,
    {%- endif %}
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> feature_requires_grad, // [T],
    pta::PackedTensorAccessor32<at::acc_type<cache_t, true>, 1, at::RestrictPtrTraits> grad_indice_weights,
    {%- if vbe %}
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> row_grad_offsets,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> b_t_map,
    const int32_t info_B_num_bits,
    const uint32_t info_B_mask
    {%- else %}
    FixedDivisor fd_B
    {%- endif %}
    ) {

    int32_t T = D_offsets.size(0) - 1;
    int32_t b_t = blockIdx.x * blockDim.y + threadIdx.y;
    if (b_t >= offsets.size(0) - 1) {
        return;
    }

    int32_t t;
    [[maybe_unused]] int32_t b;

    {%- if vbe %}
    const auto info = reinterpret_cast<const uint32_t*>(&b_t_map[b_t])[0];
    reinterpret_cast<uint32_t*>(&t)[0] = info >> info_B_num_bits;
    reinterpret_cast<uint32_t*>(&b)[0] = info & info_B_mask;
    {%- else %}
    fd_B.DivMod(b_t, &t, &b);
    {%- endif %}

    int64_t weights_offset = weights_offsets[t];
    int32_t D_start = D_offsets[t];
    int32_t D_end = D_offsets[t + 1];
    int32_t D = D_end - D_start;
    int64_t indices_start = offsets[b_t];
    int64_t indices_end = offsets[b_t + 1];
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

    {%- if vbe %}
    const grad_t* grad_output_ = &grad_output[0][row_grad_offsets[b_t]];
    {%- else %}
    const grad_t* grad_output_ = &grad_output[b][D_start];
    {%- endif %}

    Vec4T<at::acc_type<cache_t, true>> grad_out[kMaxVecsPerThread];
    #pragma unroll kMaxVecsPerThread
    for (int32_t i = 0;
        i < kMaxVecsPerThread && 4 * kWarpSize * i + threadIdx.x * 4 < D;
        ++i) {
        int32_t d = 4 * kWarpSize * i + threadIdx.x * 4;
        Vec4T<at::acc_type<grad_t, true>> go(grad_output_ + d);
        grad_out[i] = go;
    }

    for (int32_t l_start = 0; l_start < L; l_start += kWarpSize) {
        int32_t l = l_start + threadIdx.x;
        int64_t idx = l < L ? indices[indices_start + l] : 0;
        {%- if not dense %}
        int32_t cache_idx = (placement == PlacementType::MANAGED_CACHING && l < L) ? lxu_cache_locations[indices_start + l] : 0;
        {%- endif %}
        for (auto j = 0; j < kWarpSize && l_start + j < L; ++j) {
            int64_t idx_j = shfl_sync(idx, j);
            {%- if not dense %}
            int32_t cache_idx_j = shfl_sync(cache_idx, j);
            {%- endif %}
            at::acc_type<cache_t, true> grad_indice_weight = 0.0;

        #pragma unroll kMaxVecsPerThread
            for (int32_t i = 0;
                i < kMaxVecsPerThread && 4 * kWarpSize * i + threadIdx.x * 4 < D;
                ++i) {
                int32_t d = 4 * kWarpSize * i + threadIdx.x * 4;
                {%- if not dense %}
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
                {%- else %}
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
                {%- endif %}
            }
            grad_indice_weight =
                warpReduceAllSum<at::acc_type<cache_t, true>>(grad_indice_weight);
            if (threadIdx.x == 0) {
                grad_indice_weights[indices_start + l_start + j] = grad_indice_weight;
            }
        }
    }
}

Tensor {{ ddesc }}_embedding_codegen_grad_indice_weights{{ vdesc }}_cuda(
    const Tensor& grad_output,
    const Tensor& dev_weights,
    {%- if not dense %}
    const Tensor& uvm_weights,
    const Tensor& lxu_cache_weights,
    const Tensor& weights_placements,
    {%- endif %}
    const Tensor& weights_offsets,
    const Tensor& D_offsets,
    const int64_t max_D,
    const Tensor& indices,
    const Tensor& offsets,
    {%- if not dense %}
    const Tensor& lxu_cache_locations,
    {%- endif %}
    {%- if vbe %}
    const Tensor& feature_requires_grad,
    const Tensor& vbe_row_output_offsets,
    const Tensor& vbe_b_t_map,
    const int64_t info_B_num_bits, // int32_t
    const int64_t info_B_mask_int64 // uint32_t
    {%- else %}
    const Tensor& feature_requires_grad
    {%- endif %}
) {
   TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
        dev_weights,
        {%- if not dense %}
        uvm_weights,
        lxu_cache_weights,
        weights_placements,
        {%- endif %}
        weights_offsets,
        D_offsets,
        indices,
        offsets,
        {%- if not dense %}
        lxu_cache_locations,
        {%- endif %}
        {%- if vbe %}
        vbe_row_output_offsets,
        vbe_b_t_map,
        {%- endif %}
        grad_output
    );

    if (feature_requires_grad.defined()) {
        TENSOR_ON_CUDA_GPU(feature_requires_grad);
    }

    CUDA_DEVICE_GUARD(dev_weights);

    const auto T = D_offsets.size(0) - 1;
    TORCH_CHECK_GT(T, 0);
    // offsets = [B x T  + 1]
    const auto total_B = offsets.size(0) - 1;
    TORCH_CHECK_GE(total_B, 0);
    TORCH_CHECK_LE(max_D, {{ max_embedding_dim }});
    auto grad_indice_weights = empty_like(indices, indices.options().dtype(at::toAccumulateType(grad_output.scalar_type(), true)));
    if (total_B == 0) {
      return grad_indice_weights;
    }
    const auto feature_requires_grad_ = feature_requires_grad.defined() ? feature_requires_grad : at::empty({0}, indices.options().dtype(at::kInt));

    {%- if vbe %}
    // Cast info_B_mask from int64_t to uint32_t
    const uint32_t info_B_mask = info_B_mask_int64;
    {%- endif %}

    DISPATCH_EMB_GRAD_CACHE_TYPES(
        dev_weights.scalar_type(),
        grad_output.scalar_type(),
        {%- if not dense %}
        lxu_cache_weights.scalar_type(),
        {%- else %}
        dev_weights.scalar_type(),
        {%- endif %}
        "split_embedding_codegen_grad_indice_weights_kernel",
        [&] {
            {%- if vbe %}
            const auto& grad_output_reshaped = grad_output.reshape({1, -1});
            {%- else %}
            const auto& grad_output_reshaped = grad_output;
            {%- endif %}

            {%- for kMaxVecsPerThread in range(1, max_embedding_dim // items_per_warp + 1) %}
            if (max_D <= {{ items_per_warp * kMaxVecsPerThread }}) {

#ifdef FBGEMM_GPU_MEMCHECK
            const auto func_name = "{{ ddesc }}_embedding_codegen_grad_indice_weights{{ vdesc }}_kernel";
#endif

            {{ ddesc }}_embedding_codegen_grad_indice_weights{{ vdesc }}_kernel<
                emb_t,
                grad_t,
                cache_t,
                {{ kMaxVecsPerThread }}><<<
                div_round_up(total_B, kForwardMaxThreads / kWarpSize),
                dim3(kWarpSize, kForwardMaxThreads / kWarpSize),
                0,
                at::cuda::getCurrentCUDAStream()>>>(
                MAKE_PTA_WITH_NAME(func_name, grad_output_reshaped, grad_t, 2, 64),
                MAKE_PTA_WITH_NAME(func_name, dev_weights, emb_t, 1, 64),
                {%- if not dense %}
                MAKE_PTA_WITH_NAME(func_name, uvm_weights, emb_t, 1, 64),
                MAKE_PTA_WITH_NAME(func_name, lxu_cache_weights, cache_t, 2, 64),
                MAKE_PTA_WITH_NAME(func_name, weights_placements, int32_t, 1, 32),
                {%- endif %}
                MAKE_PTA_WITH_NAME(func_name, weights_offsets, int64_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name, D_offsets, int32_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name, indices, int64_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name, offsets, int64_t, 1, 32),
                {%- if not dense %}
                MAKE_PTA_WITH_NAME(func_name, lxu_cache_locations, int32_t, 1, 32),
                {%- endif %}
                MAKE_PTA_WITH_NAME(func_name, feature_requires_grad_, int32_t, 1, 32),
                MAKE_PTA_ACC_WITH_NAME(func_name, grad_indice_weights, grad_t, 1, 32),
                {%- if vbe %}
                MAKE_PTA_WITH_NAME(func_name, vbe_row_output_offsets, int64_t, 1, 32),
                MAKE_PTA_WITH_NAME(func_name, vbe_b_t_map, int32_t, 1, 32),
                info_B_num_bits,
                info_B_mask
                {%- else %}
                FixedDivisor(total_B / T)
                {%- endif %}
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            return;
            }
            {%- endfor %}
        });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grad_indice_weights;
}

////////////////////////////////////////////////////////////////////////////////
// Op registrations
////////////////////////////////////////////////////////////////////////////////
TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
    {%- set embedding_codegen_grad_indice_weights_op =
        "{}_embedding_codegen_grad_indice_weights{}_cuda".format(
            ddesc, vdesc
        )
    %}
    m.def("{{ embedding_codegen_grad_indice_weights_op }}("
          "    Tensor grad_output, "
          "    Tensor dev_weights, "
          {%- if not dense %}
          "    Tensor uvm_weights, "
          "    Tensor lxu_cache_weights, "
          "    Tensor weights_placements, "
          {%- endif %}
          "    Tensor weights_offsets, "
          "    Tensor D_offsets, "
          "    int max_D, "
          "    Tensor indices, "
          "    Tensor offsets, "
          {%- if not dense %}
          "    Tensor lxu_cache_locations, "
          {%- endif %}
          {%- if vbe %}
          "    Tensor feature_requires_grad, "
          "    Tensor vbe_row_output_offsets, "
          "    Tensor vbe_b_t_map, "
          "    int info_B_num_bits, "
          "    int info_B_mask_int64"
          {%- else %}
          "    Tensor feature_requires_grad"
          {%- endif %}
          ") -> Tensor");
    DISPATCH_TO_CUDA(
        "{{ embedding_codegen_grad_indice_weights_op }}",
        {{ embedding_codegen_grad_indice_weights_op }}
    );
}
{%- endif %} {#-/* if not dense or not vbe */#}
{%- endfor %} {#-/* for vbe */#}
    // clang-format on
