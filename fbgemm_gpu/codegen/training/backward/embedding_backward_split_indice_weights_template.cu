/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// clang-format off

{%- set mdesc =  "dense" if dense else ("ssd" if ssd else "split") %}

{%- set locs_or_addrs_tensor = "ssd_row_addrs" if ssd else "lxu_cache_locations" %}
{%- set locs_or_addrs_type = "int64_t" if ssd else "int32_t" %}
{%- set locs_or_addrs_idx = "row_idx" if ssd else "cache_idx" %}

////////////////////////////////////////////////////////////////////////////////
// Required for op registrations
////////////////////////////////////////////////////////////////////////////////
#include "fbgemm_gpu/embedding_forward_template_helpers.cuh"
#include "fbgemm_gpu/utils/ops_utils.h"
#include "fbgemm_gpu/utils/tensor_utils.h"

using Tensor = at::Tensor;
using namespace fbgemm_gpu;

#define DISPATCH_NON_VEC_BLOCKING_KERNEL(MAX_D, ...) \
  [&] {                                              \
    {{
       dispatch_non_vec_blocking_kernel(
           items_per_warp,
           fixed_max_vecs_per_thread["backward_indice_weights"],
           use_subwarp_shuffle=False,
       )
    -}}
  }()

#define DISPATCH_VEC_BLOCKING_KERNEL(MAX_D, ...)     \
  [&] {                                              \
    {{
       dispatch_vec_blocking_kernel(
           items_per_warp,
           fixed_max_vecs_per_thread["backward_indice_weights"],
       )
    -}}
  }()

{%- for vbe in ([True, False]) %}
{%- set vdesc = "_vbe" if vbe else "" %}

{#-
  /* Generate different kernels for different kUseVecBlocking using Jinja
     instead of using C++ template because the kernel is sensitive to the
     number of registers. Introducing new variables into the kernel can
     increase the number of registers and reduce the kernel occupancy which can
     result in kernel slowdown
   */
#}
{%- for use_vec_blocking in [True, False] %}
{%- set vbdesc = "vec_blocking_" if use_vec_blocking else "" %}

// TODO: optimization to use multiple warps per row.
template <
  typename emb_t,
  typename grad_t,
  typename cache_t,
  typename index_t,
  int32_t kFixedMaxVecsPerThread
>
__global__ __launch_bounds__(kForwardMaxThreads) void
{{ mdesc }}_embedding_codegen_grad_indice_weights{{ vdesc }}_{{ vbdesc }}kernel(
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
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> indices, // [N = \sum_{b,t} L_{b,t} total indices, i.e. flattened [B][T][L]
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> offsets, // [B x T + 1]
    {%- if not dense %}
    const pta::PackedTensorAccessor32<{{ locs_or_addrs_type }}, 1, at::RestrictPtrTraits> {{ locs_or_addrs_tensor }},
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
    constexpr int32_t kVecWidth = 4;

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

    const auto weights_offset = weights_offsets[t];
    const auto D_start = D_offsets[t];
    const auto D_end = D_offsets[t + 1];
    const auto D = D_end - D_start;
    const auto indices_start = offsets[b_t];
    const auto indices_end = offsets[b_t + 1];
    const auto L = indices_end - indices_start;
    if (feature_requires_grad.size(0) > 0 && !feature_requires_grad[t]) {
        // If the table does not require gradient computation, we set the gradient to zero.
        for (auto l_start = 0; l_start < L; l_start += kWarpSize) {
            auto l = l_start + threadIdx.x;
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

    Vec4TAcc<cache_t> grad_out[kFixedMaxVecsPerThread];

    {%- if use_vec_blocking %}
    const int32_t num_vecs = div_round_up(D, kWarpSize * kVecWidth);
    for (int32_t vec_start = 0;
         vec_start < num_vecs;
         vec_start += kFixedMaxVecsPerThread) {
        {%- set d = "(kWarpSize * (vec + vec_start) + threadIdx.x) * kVecWidth" %}
    {%- else %}
        {%- set d = "(kWarpSize * vec + threadIdx.x) * kVecWidth" %}
    {%- endif %} {# /* if use_vec_blocking */ #}

        // Load gradients
        // TODO: Maybe using a combination of shared memory and registers is
        // better for performance
        #pragma unroll kFixedMaxVecsPerThread
        for (int32_t vec = 0; vec < kFixedMaxVecsPerThread && {{ d }} < D; ++vec) {
            const int32_t d = {{ d }};
            Vec4TAcc<grad_t> go(grad_output_ + d);
            grad_out[vec] = go;
        }

        for (int32_t l_start = 0; l_start < L; l_start += kWarpSize) {
            int32_t l = l_start + threadIdx.x;
            index_t idx = l < L ? indices[indices_start + l] : 0;
            {%- if not dense %}
            const auto {{ locs_or_addrs_idx }} =
                (placement == PlacementType::MANAGED_CACHING && l < L)
                    ? {{ locs_or_addrs_tensor }}[indices_start + l] : 0;
            {%- endif %}
            for (auto j = 0; j < kWarpSize && l_start + j < L; ++j) {
                auto idx_j = shfl_sync(idx, j);
                {%- if not dense %}
                const auto {{ locs_or_addrs_idx }}_j = shfl_sync({{ locs_or_addrs_idx }}, j);
                {%- endif %}
                at::acc_type<cache_t, true> grad_indice_weight = 0.0;

                #pragma unroll kFixedMaxVecsPerThread
                for (int32_t vec = 0;
                    vec < kFixedMaxVecsPerThread && {{ d }} < D;
                    ++vec) {
                    const int32_t d = {{ d }};
                    {%- if not dense %}
                    if ({{ "true || " if ssd else "" }}
                      (
                          placement == PlacementType::MANAGED_CACHING
                          && ({{ locs_or_addrs_idx }}_j != kCacheLocationMissing)
                      )
                    ) {
                        const cache_t* cache_weights =
                          {%- if ssd  %}
                          reinterpret_cast<cache_t*>(
                              *reinterpret_cast<const uint64_t*>(&{{ locs_or_addrs_idx }}_j));
                          {%- else %}
                          &lxu_cache_weights[{{ locs_or_addrs_idx }}_j][d];
                          {%- endif %}
                        Vec4T<cache_t> weight(cache_weights);
                        grad_indice_weight += weight.acc.x * grad_out[vec].acc.x +
                            weight.acc.y * grad_out[vec].acc.y +
                            weight.acc.z * grad_out[vec].acc.z +
                            weight.acc.w * grad_out[vec].acc.w;
                    } else {
                        int32_t D_emb = D;
                        if (std::is_same<emb_t, uint8_t>::value) {
                            D_emb += kINT8QparamsBytes;
                        }
                        auto weight_row = WeightRow<emb_t, cache_t, at::acc_type<cache_t, true>>(
                            const_cast<emb_t*>(&weights[idx_j * D_emb]),
                            nullptr,
                            D);
                        float2 qparams;
                        if (std::is_same<emb_t, uint8_t>::value) {
                            qparams = weight_row.load_qparams();
                        }
                        Vec4TAcc<cache_t> weight =
                        weight_row.load(d, qparams);
                        grad_indice_weight += weight.acc.x * grad_out[vec].acc.x +
                            weight.acc.y * grad_out[vec].acc.y +
                            weight.acc.z * grad_out[vec].acc.z +
                            weight.acc.w * grad_out[vec].acc.w;
                    }
                    {%- else %}
                    int32_t D_emb = D;
                    if (std::is_same<emb_t, uint8_t>::value) {
                        D_emb += kINT8QparamsBytes;
                    }
                    auto weight_row = WeightRow<emb_t, cache_t, at::acc_type<cache_t, true>>(
                        const_cast<emb_t*>(&weights[idx_j * D_emb]),
                        nullptr,
                        D);
                    float2 qparams;
                    if (std::is_same<emb_t, uint8_t>::value) {
                        qparams = weight_row.load_qparams();
                    }
                    Vec4TAcc<cache_t> weight =
                    weight_row.load(d, qparams);
                    grad_indice_weight += weight.acc.x * grad_out[vec].acc.x +
                        weight.acc.y * grad_out[vec].acc.y +
                        weight.acc.z * grad_out[vec].acc.z +
                        weight.acc.w * grad_out[vec].acc.w;
                    {%- endif %}
                }
                grad_indice_weight =
                    warpReduceAllSum<at::acc_type<cache_t, true>>(grad_indice_weight);
                if (threadIdx.x == 0) {
                    {%- if use_vec_blocking %}
                    if (vec_start == 0) {
                        grad_indice_weights[indices_start + l_start + j] =
                            grad_indice_weight;
                    }
                    else {
                        grad_indice_weights[indices_start + l_start + j] +=
                            grad_indice_weight;
                    }
                    {%- else %}
                    grad_indice_weights[indices_start + l_start + j] =
                        grad_indice_weight;
                    {%- endif %}
                }
            }
        }
    {%- if use_vec_blocking %}
    } // for vec_start
    {%- endif %}
}
{%- endfor %} {# /* for use_vec_blocking */ #}

Tensor {{ mdesc }}_embedding_codegen_grad_indice_weights{{ vdesc }}_cuda(
    const Tensor& grad_output,
    const Tensor& dev_weights,
    {%- if not dense %}
    const Tensor& uvm_weights,
    const Tensor& lxu_cache_weights,
    const Tensor& weights_placements,
    {%- endif %}
    const Tensor& weights_offsets,
    const Tensor& D_offsets,
    const c10::SymInt max_D_,
    const Tensor& indices,
    const Tensor& offsets,
    {%- if not dense %}
    const Tensor& {{ locs_or_addrs_tensor }},
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
   const int64_t max_D = max_D_.guard_int(__FILE__, __LINE__);
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
        {{ locs_or_addrs_tensor }},
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

    auto aligned_grad_output = aligned_grad_output_tensor_for_cuda_backwards(grad_output);

    CUDA_DEVICE_GUARD(dev_weights);

    const auto T = D_offsets.size(0) - 1;
    TORCH_CHECK_GT(T, 0);
    // offsets = [B x T  + 1]
    const auto total_B = offsets.size(0) - 1;
    TORCH_CHECK_GE(total_B, 0);
    TORCH_CHECK_LE(max_D, {{ max_embedding_dim }});
    auto grad_indice_weights = empty_like(indices, indices.options().dtype(
          at::toAccumulateType(aligned_grad_output.scalar_type(), true)));

    if (total_B == 0) {
      return grad_indice_weights;
    }

    const auto feature_requires_grad_ = feature_requires_grad.defined()
        ? feature_requires_grad
        : at::empty({0}, indices.options().dtype(at::kInt));

    {%- if vbe %}
    // Cast info_B_mask from int64_t to uint32_t
    const uint32_t info_B_mask = info_B_mask_int64;
    {%- endif %}

    AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "split_embedding_codegen_grad_indice_weights{{ vdesc }}_kernel_1", [&] {
    DISPATCH_EMB_GRAD_CACHE_TYPES(
        dev_weights.scalar_type(),
        aligned_grad_output.scalar_type(),
        {%- if not dense %}
        lxu_cache_weights.scalar_type(),
        {%- else %}
        dev_weights.scalar_type(),
        {%- endif %}
        "split_embedding_codegen_grad_indice_weights{{ vdesc }}_kernel_2",
        [&] {
            {%- if vbe %}
            const auto& grad_output_reshaped = aligned_grad_output.reshape({1, -1});
            {%- else %}
            const auto& grad_output_reshaped = aligned_grad_output;
            {%- endif %}

            {%- for use_vec_blocking in [False, True] %}
            {%- set vbdesc = "vec_blocking_" if use_vec_blocking else "" %}
            {%- set dpdesc = "NON_" if not use_vec_blocking else "" %}
            DISPATCH_{{ dpdesc }}VEC_BLOCKING_KERNEL(max_D, [&] {
                {%- set kernel_name =
                    "{}_embedding_codegen_grad_indice_weights{}_{}kernel".format(
                        mdesc, vdesc, vbdesc)
                 %}
#ifdef FBGEMM_GPU_MEMCHECK
                const auto func_name = "{{ kernel_name }}";
#endif
                {{ kernel_name }}<
                    emb_t,
                    grad_t,
                    cache_t,
                    index_t,
                    kFixedMaxVecsPerThread><<<
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
                    MAKE_PTA_WITH_NAME(func_name, indices, index_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name, offsets, index_t, 1, 32),
                    {%- if not dense %}
                    MAKE_PTA_WITH_NAME(func_name, {{ locs_or_addrs_tensor }}, {{ locs_or_addrs_type }}, 1, 32),
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
            });
            {%- endfor %} {# /* for use_vec_blocking */ #}
        });
    });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grad_indice_weights;
}


Tensor {{ mdesc }}_embedding_codegen_grad_indice_weights{{ vdesc }}_meta(
    const Tensor& grad_output,
    const Tensor& dev_weights,
    {%- if not dense %}
    const Tensor& uvm_weights,
    const Tensor& lxu_cache_weights,
    const Tensor& weights_placements,
    {%- endif %}
    const Tensor& weights_offsets,
    const Tensor& D_offsets,
    const c10::SymInt max_D,
    const Tensor& indices,
    const Tensor& offsets,
    {%- if not dense %}
    const Tensor& {{ locs_or_addrs_tensor }},
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

    const auto T = D_offsets.sym_size(0) - 1;
    TORCH_CHECK_GT(T, 0);
    // offsets = [B x T  + 1]
    const auto total_B = offsets.sym_size(0) - 1;
    TORCH_CHECK_GE(total_B, 0);
    TORCH_CHECK_LE(max_D, {{ max_embedding_dim }});

    auto grad_indice_weights = empty_like(indices, indices.options().dtype(
          at::toAccumulateType(grad_output.scalar_type(), true)));

    return grad_indice_weights;
}

////////////////////////////////////////////////////////////////////////////////
// Op registrations
////////////////////////////////////////////////////////////////////////////////
TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
    {%- set embedding_codegen_grad_indice_weights_op =
        "{}_embedding_codegen_grad_indice_weights{}".format(
            mdesc, vdesc
        )
    %}
    {%- set embedding_codegen_grad_indice_weights_op_cuda = embedding_codegen_grad_indice_weights_op + "_cuda" %}
    m.def("{{ embedding_codegen_grad_indice_weights_op_cuda }}("
          "    Tensor grad_output, "
          "    Tensor dev_weights, "
          {%- if not dense %}
          "    Tensor uvm_weights, "
          "    Tensor lxu_cache_weights, "
          "    Tensor weights_placements, "
          {%- endif %}
          "    Tensor weights_offsets, "
          "    Tensor D_offsets, "
          "    SymInt max_D, "
          "    Tensor indices, "
          "    Tensor offsets, "
          {%- if not dense %}
          "    Tensor {{ locs_or_addrs_tensor }}, "
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
        "{{ embedding_codegen_grad_indice_weights_op_cuda }}",
        {{ embedding_codegen_grad_indice_weights_op_cuda }}
    );
    m.impl("{{ embedding_codegen_grad_indice_weights_op_cuda }}",
        torch::dispatch(c10::DispatchKey::Meta,
          TORCH_FN({{ embedding_codegen_grad_indice_weights_op }}_meta)));
}
{%- endfor %} {#-/* for vbe */#}
  // clang-format on
