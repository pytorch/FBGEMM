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

{%- set mdesc = "dense" if dense else ("ssd" if ssd else "split") %}
{%- set wdesc = "weighted" if weighted else "unweighted" %}

{%- macro get_desc_suffix(gwd) %}
{%- set vdesc = "_vbe" if vbe else "" %}
{%- set gwddesc = "_gwd" if gwd else "" %}
{{- wdesc + vdesc + gwddesc }}
{%- endmacro %}

{%- set locs_or_addrs_tensor = "ssd_row_addrs" if ssd else "lxu_cache_locations" %}
{%- set locs_or_addrs_type = "int64_t" if ssd else "int32_t" %}

{%- if not dense and not nobag and not vbe %}
#include "fbgemm_gpu/utils/dispatch_macros.h"
{%- endif %}

{%- if not is_index_select %}
////////////////////////////////////////////////////////////////////////////////
// Required for op registrations
////////////////////////////////////////////////////////////////////////////////
#include "fbgemm_gpu/utils/ops_utils.h"
{%- endif %}
#include "fbgemm_gpu/utils/cuda_utilities.cuh"
#include "fbgemm_gpu/utils/kernel_launcher.cuh"
#include "fbgemm_gpu/embedding_forward_template_helpers.cuh"
#include "fbgemm_gpu/split_embeddings_cache_cuda.cuh"

using Tensor = at::Tensor;
using namespace fbgemm_gpu;

////////////////////////////////////////////////////////////////////////////////
// External Function Declarations
////////////////////////////////////////////////////////////////////////////////

{%- if not weighted %}
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
{{ mdesc }}_embedding_nobag_codegen_forward_unweighted_small_kernel(
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
    const pta::PackedTensorAccessor32<{{ locs_or_addrs_type }}, 1, at::RestrictPtrTraits> {{ locs_or_addrs_tensor }},
    {%- endif %}
    {%- if is_index_select %}
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> output_offsets,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> total_L_offsets,
    const int32_t fixed_L_per_warp,
    const bool permute_output_dim_0_1,
    {%- endif %}
    pta::PackedTensorAccessor64<output_t, {{ "1" if is_index_select else "2" }}, at::RestrictPtrTraits> output
    );
{%- endif %} {#-/* if not weighted */#}

{% if not dense %}
// Support only the split-pooled TBE case
template <
    typename emb_t,
    typename cache_t,
    typename output_t,
    typename index_t,
    bool use_lxu_cache
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
{% endif %} {#-/* if not dense */#}


{%- for nobag in ([True, False] if (not is_gwd) else [False]) %}
{%- set ndesc = "_nobag" if nobag else "" %}
{%- if is_valid_forward_config(nobag, weighted, vbe, is_index_select) %}
{%- set has_experimental = has_experimental_support(dense, nobag, vbe, is_index_select, ssd) %}

{%- set is_gwd_kernel = is_gwd and is_valid_gwd_config(
    dense,
    nobag,
    vbe,
    is_index_select,
    has_global_weight_decay_support=True,
    ssd=ssd) %}
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
__launch_bounds__(kForwardMaxThreads) __global__ void
{%- if is_index_select %}
batch_index_select_dim0_codegen_forward_kernel(
{%- else %}
{{ mdesc }}_embedding{{ ndesc }}_codegen_forward_{{ get_desc_suffix(is_gwd_kernel) }}_kernel(
{%- endif %}
    const pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> dev_weights,
    {%- if not dense %}
    const pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> uvm_weights,
    const pta::PackedTensorAccessor64<cache_t, 2, at::RestrictPtrTraits> lxu_cache_weights,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> weights_placements,
    {%- endif %}
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> weights_offsets,
    {%- if not nobag or is_index_select %}
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> D_offsets,
    {%- else %}
    int64_t D,
    {%- endif %}
    {%- if vbe %}
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> row_output_offsets,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> b_t_map,
    const int32_t info_B_num_bits,
    const uint32_t info_B_mask,
    {%- else %}
    FixedDivisor fd_B,
    {%- endif %}
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> indices,
    {%- if not is_index_select %}
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> offsets,
    {%- endif %}
    {%- if not nobag %}
    int64_t pooling_mode,
    {%- endif %}
    {%- if weighted %}
    pta::PackedTensorAccessor32<at::acc_type<cache_t, true>, 1, at::RestrictPtrTraits> indice_weights,
    {%- endif %}
    {%- if not dense %}
    const pta::PackedTensorAccessor32<{{ locs_or_addrs_type }}, 1, at::RestrictPtrTraits> {{ locs_or_addrs_tensor }},
    const int32_t* lxu_cache_conflict_misses,
    {%- endif %}
    {%- if is_index_select %}
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> output_offsets,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> total_L_offsets,
    const int32_t fixed_L_per_warp,
    const bool permute_output_dim_0_1,
    {%- endif %} // if dense
    {%- if is_gwd_kernel %}
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> hash_size_cumsum,
    const pta::PackedTensorAccessor64<float, 1, at::RestrictPtrTraits> prev_iter_dev,
    const float learning_rate,
    const float weight_decay,
    const int64_t iter,
    const float gwd_lower_bound,
    {%- endif %}
    pta::PackedTensorAccessor64<output_t, {{ "1" if is_index_select else "2" }}, at::RestrictPtrTraits> output
    );
{%- endif %} {#-/* if is_valid_forward_config(...) */#}
{%- endfor %} {#-/* for nobag in [True, False] */#}


////////////////////////////////////////////////////////////////////////////////
// Utility Macros
////////////////////////////////////////////////////////////////////////////////

/*
  The macro definition for both cases are almost the same except for the
  definition of kThreadGroupSize.  In the FBGEMM_USE_SUBWARP_SHUFFLE case, if
  MAX_D is small, then we use fewer number of threads than kWarpSize.

  NOTE: kMaxVecsPerThread is computed using the ternary operator because HIPCC
  is unable to use std::max in constexpr context.
*/
{%- macro dispatch_optimal_forward_kernel(
    dispatch_macro_name,
    max_forward_embedding_dim
  )
%}
  {%- set fixed_max_vecs_per_thread = max_forward_embedding_dim // items_per_warp%}
#ifdef FBGEMM_USE_SUBWARP_SHUFFLE
#define {{ dispatch_macro_name }}(MAX_D, ...) \
  [&] {                                        \
    {{
       dispatch_non_vec_blocking_kernel(
           items_per_warp,
           fixed_max_vecs_per_thread,
           use_subwarp_shuffle=True)
    -}}
    return;                                    \
  }()

#else
#define {{ dispatch_macro_name }}(MAX_D, ...) \
  [&] {                                        \
    {{
       dispatch_non_vec_blocking_kernel(
           items_per_warp,
           fixed_max_vecs_per_thread,
           use_subwarp_shuffle=False)
    -}}
    return;                                    \
  }()

#endif
{% endmacro %}

{#-
  /* Generate a dispatch macro for forward kernels that
     has_experimental=False. We generate kernel templates up to
     max_embedding_dim.
   */
#}
{{
  dispatch_optimal_forward_kernel(
      "DISPATCH_OPTIMAL_FORWARD_KERNEL",
      max_embedding_dim
  )
}}

{#-
  /* Generate a dispatch macro for forward kernels that
     has_experimental=True. We generate kernel templates up to
     legacy_max_embedding_dim which <= max_embedding_dim. If max_D is larger
     than legacy_max_embedding_dim, TBE v2 (experimental TBE) will be used
     instead of the legacy kernel.
   */
#}
{{
  dispatch_optimal_forward_kernel(
      "DISPATCH_OPTIMAL_LEGACY_FORWARD_KERNEL",
      legacy_max_embedding_dim
  )
}}

#define DISPATCH_OPTIMAL_NOBAG_FORWARD_KERNEL(DD_, ...)                        \
  [&] {                                                                        \
    {%- for kEmbeddingSize in [4, 8, 16, 32] %}
    if (DD_ <= {{ kEmbeddingSize }}) {                                         \
      constexpr int kEmbeddingSize = {{ kEmbeddingSize }};                     \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    {%- endfor %}
    return;                                                                    \
  }()


#define DISPATCH_KERNEL_FOR_CACHE_CASE(CACHE_CASE_, ...)                       \
  [&] {                                                                        \
    {%- if dense %}
      return __VA_ARGS__();                                                    \
    {%- else %}
    {%- for use_cache in ["false", "true"] %}
    if (CACHE_CASE_ == {{ use_cache }}) {                                      \
      constexpr auto use_cache_t = {{ use_cache }};                            \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    {%- endfor %}
    return;                                                                    \
    {%- endif %}
  }()


////////////////////////////////////////////////////////////////////////////////
// Kernel Definitions
////////////////////////////////////////////////////////////////////////////////

{%- for nobag in ([True, False] if (not is_gwd) else [False]) %}
{%- set ndesc = "_nobag" if nobag else "" %}
{%- if is_valid_forward_config(nobag, weighted, vbe, is_index_select) %}
{%- set has_experimental = has_experimental_support(dense, nobag, vbe, is_index_select, ssd) %}

{#- /* Generate a separate cuda host to enable global weight decay using Jinja */ #}
{%- set is_gwd_kernel = is_gwd and is_valid_gwd_config(
    dense,
    nobag,
    vbe,
    is_index_select,
    has_global_weight_decay_support=True,
    ssd=ssd) %}
{%- set desc_suffix = get_desc_suffix(is_gwd_kernel) %}
Tensor
{%- if is_index_select %}
batch_index_select_dim0_codegen_forward_cuda(
{%- else %}
{{ mdesc }}_embedding{{ ndesc }}_codegen_forward_{{ desc_suffix }}_cuda(
{%- endif %}
    const Tensor& dev_weights,
    {%- if not dense %}
    const Tensor& uvm_weights,
    const Tensor& lxu_cache_weights,
    const Tensor& weights_placements,
    {%- endif %}
    const Tensor& weights_offsets,
    {%- if not nobag or is_index_select %}
    const Tensor& D_offsets,
    {%- else %}
    const c10::SymInt D_,
    {%- endif %}
    {%- if not nobag %}
    const c10::SymInt total_D_,
    {%- endif %}
    {%- if not nobag or is_index_select %}
    const c10::SymInt max_D_,
    {% endif %}
    const Tensor& indices,
    {%- if not is_index_select %}
    const Tensor& offsets,
    {%- endif %}
    {%- if not nobag %}
    const int64_t pooling_mode,
    {%- endif %}
    {%- if weighted %}
    const Tensor& indice_weights,
    {%- endif %}
    {%- if not dense %}
    const Tensor& {{ locs_or_addrs_tensor }},
    const Tensor& uvm_cache_stats,
    {%- endif %}
    const int64_t output_dtype,
    {%- if is_index_select %}
    const Tensor& output_offsets,
    const Tensor& total_L_offsets,
    const int64_t output_size,
    const int32_t fixed_L_per_warp,
    const int32_t num_warps_per_feature,
    const bool permute_output_dim_0_1
    {%- else %}
    {%- if vbe %}
    const Tensor& vbe_row_output_offsets,
    const Tensor& vbe_b_t_map,
    const c10::SymInt vbe_output_size_,
    const int64_t info_B_num_bits, // int32_t
    const int64_t info_B_mask_int64, // uint32_t
    {%- endif %}
    {%- if is_gwd_kernel %}
    const Tensor& hash_size_cumsum,
    const Tensor& prev_iter_dev,
    const Tensor& learning_rate_tensor,
    const double weight_decay,
    const int64_t iter,
    const double gwd_lower_bound,
    {%- endif %}
    const bool is_experimental
    {%- endif %} {#- /*if is_index_select*/ #}
) {
    {%- if not nobag or is_index_select %}
    {%- else %}
    const int64_t D = D_.guard_int(__FILE__, __LINE__);
    {%- endif %}

    {%- if not nobag %}
    const int64_t total_D = total_D_.guard_int(__FILE__, __LINE__);
    {%- endif %}

    {%- if not nobag or is_index_select %}
    const int64_t max_D = max_D_.guard_int(__FILE__, __LINE__);
    {%- endif %}
    {%- if vbe %}
    const int64_t vbe_output_size = vbe_output_size_.guard_int(__FILE__, __LINE__);
    {%- endif %}

    TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
        {%- if not dense %}
        uvm_weights,
        lxu_cache_weights,
        weights_placements,
        {%- endif %}
        weights_offsets,
        {%- if not nobag or is_index_select %}
        D_offsets,
        {%- endif %}
        indices,
        {%- if not is_index_select %}
        offsets,
        {%- endif %}
        {%- if weighted %}
        indice_weights,
        {%- endif %}
        {%- if not dense %}
        {{ locs_or_addrs_tensor }},
        {%- endif %}
        {%- if vbe %}
        vbe_row_output_offsets,
        vbe_b_t_map,
        {%- endif %}
        {%- if is_index_select %}
        total_L_offsets,
        {%- endif %}
        {%- if is_gwd_kernel %}
        prev_iter_dev,
        {%- endif %}
        dev_weights
    );

    {%- if is_index_select %}
    if (!permute_output_dim_0_1) {
        TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
            output_offsets,
            dev_weights
        );
    }
    {%- endif %}

    CUDA_DEVICE_GUARD(dev_weights);

    {%- if not nobag %}
    int32_t T = D_offsets.numel() - 1;
    {%- else %}
    int32_t total_L = indices.numel();
    int32_t T = weights_offsets.numel();
    {%- endif %}
    TORCH_CHECK_GT(T, 0);
    // offsets = [B x T  + 1]
    {%- if is_index_select %}
    const auto total_B = num_warps_per_feature * T;
    const int32_t B = num_warps_per_feature;
    {%- else %}
    const auto total_B = offsets.size(0) - 1;
    const int32_t B = total_B / T;
    {%- endif %}
    TORCH_CHECK_GE(B, 0);
    {%- if not nobag or is_index_select %}
    {%- if not nobag %}
    TORCH_CHECK_GT(total_D, 0);
    TORCH_CHECK_EQ(total_D % 4, 0);
    {%- endif %}
    TORCH_CHECK_LE(max_D, {{ max_embedding_dim }});
    {%- elif not is_index_select %}
    TORCH_CHECK_GT(D, 0);
    TORCH_CHECK_EQ(D % 4, 0);
    {%- endif %}
    {%- if vbe %}
    TORCH_CHECK_EQ(vbe_row_output_offsets.numel(), total_B);
    TENSORS_HAVE_SAME_NUMEL(vbe_row_output_offsets, vbe_b_t_map);
    TORCH_CHECK_GE(vbe_output_size, 0);

    // Cast info_B_mask from int64_t to uint32_t
    const uint32_t info_B_mask = info_B_mask_int64;
    {%- endif %}

    {%- if is_gwd_kernel %}
    // convert `learning rate` to float since `learning rate` is float in kernels
    const float learning_rate = learning_rate_tensor.item<float>();
    TORCH_CHECK(learning_rate >= 0, "Expect to apply weight decay but learning rate is < 0");
    {%- endif %}

    Tensor output;
    {%- if nobag %}
    SparseType o_dtype = static_cast<SparseType>(output_dtype);
    {%- if is_index_select %}
    TORCH_CHECK(o_dtype == SparseType::FP32 || o_dtype == SparseType::FP16 ||
                o_dtype == SparseType::BF16);

    TORCH_CHECK_GT(fixed_L_per_warp, 0);
    TORCH_CHECK_GT(num_warps_per_feature, 0);
    if (!permute_output_dim_0_1) {
        TORCH_CHECK_GE(output_size, 0);
        TORCH_CHECK_GT(output_offsets.numel(), 0);
    }

    // If permute_output_dim_0_1 is true, output shape is (batch_size * total_D)
    // Else, output shape is (output_size)
    output = at::empty({output_size}, dev_weights.options().dtype(getScalarType(o_dtype)));
    {%- else %}
    TORCH_CHECK(o_dtype == SparseType::FP32 || o_dtype == SparseType::FP16 ||
                o_dtype == SparseType::BF16 || o_dtype == SparseType::INT8);

    int64_t adjusted_D = D;
    if (o_dtype == SparseType::INT8) {
        adjusted_D += T * kINT8QparamsBytes;
    }

    output = at::empty({total_L, adjusted_D}, dev_weights.options().dtype(getScalarType(o_dtype)));
    {%- endif %}
    {%- else %}
    SparseType o_dtype = static_cast<SparseType>(output_dtype);
    TORCH_CHECK(o_dtype == SparseType::FP32 || o_dtype == SparseType::FP16 ||
                o_dtype == SparseType::BF16 || o_dtype == SparseType::INT8);

    {%- if vbe %}
    // Use a 2D tensor to make it compatible with 2D PackedTensorsAccessor of other output
    output = at::empty(
        {1, vbe_output_size},
        dev_weights.options().dtype(getScalarType(o_dtype))
    );
    {%- else %}
    int64_t total_adjusted_D = total_D;
    if (o_dtype == SparseType::INT8) {
        total_adjusted_D += T * kINT8QparamsBytes;
    }

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


    AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "batched_embedding{{ ndesc }}_forward_kernel_1", [&] {
    DISPATCH_EMB_CACHE_OUTPUT_TYPES(
        dev_weights.scalar_type(),
        {%- if not dense %}
        lxu_cache_weights.scalar_type(),
        {%- else %}
        dev_weights.scalar_type(),
        {%- endif %}
        output.scalar_type(),
        "batched_embedding{{ ndesc }}_forward_kernel_2", [&] {

        {%- if dense %}
        [[maybe_unused]] bool use_lxu_cache = false;
        {%- else %}
        // Check if LXU cache is used
        bool use_lxu_cache = lxu_cache_weights.numel() > 0;
        {%- endif %}

        {%- if has_experimental %}
        const bool is_experimental_ = (
            is_experimental && !(std::is_same<emb_t, uint8_t>() || std::is_same<output_t, uint8_t>())
        );
        // if max_D > {{ legacy_max_embedding_dim }}, use TBE v2
        if (!is_experimental_ && max_D <= {{ legacy_max_embedding_dim }}) {
        {%- endif %} {#-/* if has_experimental */#}

        {#-/* Sequence TBE Case (nobag=True) ****************************************************/#}
        {%- if nobag %}

        DISPATCH_OPTIMAL_NOBAG_FORWARD_KERNEL({{ "D" if not is_index_select else "max_D" }}, [&] {
          {%- set nobag_small_kernel =
              "batch_index_select_dim0_codegen_forward_small_kernel"
              if is_index_select else
              "{}_embedding_nobag_codegen_forward_unweighted_small_kernel".format(mdesc)
          %}

          FBGEMM_LAUNCH_KERNEL(
            ({{ nobag_small_kernel }}<
              emb_t,
              cache_t,
              output_t,
              index_t,
              kEmbeddingSize / 4>),
            div_round_up(total_B, kForwardMaxThreads / kWarpSize),
            dim3(kWarpSize, kForwardMaxThreads / kWarpSize),
            0,
            at::cuda::getCurrentCUDAStream(),
            PTA_B(dev_weights, emb_t, 1, 64),
            {%- if not dense %}
            PTA_B(uvm_weights, emb_t, 1, 64),
            PTA_B(lxu_cache_weights, cache_t, 2, 64),
            PTA_B(weights_placements, int32_t, 1, 32),
            {%- endif %}
            PTA_B(weights_offsets, int64_t, 1, 32),
            {%- if is_index_select %}
            PTA_B(D_offsets, int32_t, 1, 32),
            {%- else %}
            D,
            {%- endif %}
            FixedDivisor(B),
            PTA_B(indices, index_t, 1, 32),
            {%- if not is_index_select %}
            PTA_B(offsets, index_t, 1, 32),
            {%- endif %}
            {%- if not dense %}
            PTA_B({{ locs_or_addrs_tensor }}, {{ locs_or_addrs_type }}, 1, 32),
            {%- endif %}
            {%- if is_index_select %}
            PTA_B(output_offsets, int64_t, 1, 32),
            PTA_B(total_L_offsets, int64_t, 1, 32),
            fixed_L_per_warp,
            permute_output_dim_0_1,
            {%- endif %}
            PTA_B(output, output_t, {{ "1" if is_index_select else "2" }}, 64)
          );

          return;
        });

        DISPATCH_KERNEL_FOR_CACHE_CASE(use_lxu_cache, [&] {
          {%- set nobag_kernel =
              "batch_index_select_dim0_codegen_forward_kernel"
              if is_index_select else
              "{}_embedding_nobag_codegen_forward_unweighted_kernel".format(mdesc)
          %}
          FBGEMM_LAUNCH_KERNEL(
            ({{ nobag_kernel }}
              {%- if dense or is_index_select %}
              <emb_t, cache_t, output_t, index_t>
              {%- else %}
              <emb_t, cache_t, output_t, use_cache_t, index_t>
              {%- endif %}
            ),
            div_round_up(total_B, kForwardMaxThreads / kWarpSize),
            dim3(kWarpSize, kForwardMaxThreads / kWarpSize),
            0,
            at::cuda::getCurrentCUDAStream(),
            PTA_B(dev_weights, emb_t, 1, 64),
            {%- if not dense %}
            PTA_B(uvm_weights, emb_t, 1, 64),
            PTA_B(lxu_cache_weights, cache_t, 2, 64),
            PTA_B(weights_placements, int32_t, 1, 32),
            {%- endif %}
            PTA_B(weights_offsets, int64_t, 1, 32),
            {%- if is_index_select %}
            PTA_B(D_offsets, int32_t, 1, 32),
            {%- else %}
            D,
            {%- endif %}
            FixedDivisor(B),
            PTA_B(indices, index_t, 1, 32),
            {%- if not is_index_select %}
            PTA_B(offsets, index_t, 1, 32),
            {%- endif %}
            {%- if not dense %}
            PTA_B({{ locs_or_addrs_tensor }}, {{ locs_or_addrs_type }}, 1, 32),
            uvm_cache_stats.size(0) == 0
                ? nullptr
                : (uvm_cache_stats.data_ptr<int32_t>() + uvm_cache_stats_index::num_conflict_unique_misses),
            {%- endif %}
            {%- if is_index_select %}
            PTA_B(output_offsets, int64_t, 1, 32),
            PTA_B(total_L_offsets, int64_t, 1, 32),
            fixed_L_per_warp,
            permute_output_dim_0_1,
            {%- endif %}
            PTA_B(output, output_t, {{ "1" if is_index_select else "2" }}, 64)
          );
          return;
        });


        {#-/* Pooled TBE Case (nobag=False) *********************************/#}
        {%- else %}

        DISPATCH_KERNEL_FOR_CACHE_CASE(use_lxu_cache, [&] {
          {%- set dispatcher =
                "DISPATCH_OPTIMAL_LEGACY_FORWARD_KERNEL"
                if has_experimental
                else "DISPATCH_OPTIMAL_FORWARD_KERNEL"
          %}
          {{ dispatcher }}(max_D, [&] {
            // Other components in TBE (backward, backward_indice_weights) use
            // kFixedMaxVecsPerThread. Thus, the codegen generates
            // kFixedMaxVecsPerThread instead of kMaxVecsPerThread. But
            // kMaxVecsPerThread and kFixedMaxVecsPerThread are the same
            // forward
            {%- if is_rocm %}
            // Account for Vec2 load for ROCm
            constexpr auto kMaxVecsPerThread = 2 * kFixedMaxVecsPerThread;
            {%- else %}
            constexpr auto kMaxVecsPerThread = kFixedMaxVecsPerThread;
            {%- endif %}

            const auto grid = min(
              div_round_up(total_B, kForwardMaxThreads / kThreadGroupSize),
              utils::cuda::get_max_thread_blocks(at::cuda::getCurrentCUDAStream()));

            FBGEMM_LAUNCH_KERNEL(
              ({{ mdesc }}_embedding_codegen_forward_{{ desc_suffix }}_kernel
                <emb_t,
                cache_t,
                output_t,
                {%- if not dense%}
                use_cache_t,
                {%- endif %}
                index_t,
                kMaxVecsPerThread,
                kThreadGroupSize>),
              grid,
              dim3(kThreadGroupSize, kForwardMaxThreads / kThreadGroupSize),
              0,
              at::cuda::getCurrentCUDAStream(),
              PTA_B(dev_weights, emb_t, 1, 64),
              {%- if not dense %}
              PTA_B(uvm_weights, emb_t, 1, 64),
              PTA_B(lxu_cache_weights, cache_t, 2, 64),
              PTA_B(weights_placements, int32_t, 1, 32),
              {%- endif %}
              PTA_B(weights_offsets, int64_t, 1, 32),
              PTA_B(D_offsets, int32_t, 1, 32),
              {%- if vbe %}
              PTA_B(vbe_row_output_offsets, int64_t, 1, 32),
              PTA_B(vbe_b_t_map, int32_t, 1, 32),
              info_B_num_bits,
              info_B_mask,
              {%- else %}
              FixedDivisor(B),
              {%- endif %}
              PTA_B(indices, index_t, 1, 32),
              PTA_B(offsets, index_t, 1, 32),
              pooling_mode,
              {%- if weighted %}
              PTA_ACC_B(indice_weights, cache_t, 1, 32),
              {%- endif %}
              {%- if not dense %}
              PTA_B({{ locs_or_addrs_tensor }}, {{ locs_or_addrs_type }}, 1, 32),
              uvm_cache_stats.size(0) == 0
                  ? nullptr
                  : (uvm_cache_stats.data_ptr<int32_t>() + uvm_cache_stats_index::num_conflict_unique_misses),
              {%- endif %} // if not dense
              {%- if is_gwd_kernel %}
              PTA_B(hash_size_cumsum, int64_t, 1, 32),
              PTA_B(prev_iter_dev, float, 1, 64),
              learning_rate,
              weight_decay,
              iter,
              gwd_lower_bound,
              {%- endif %} // if not dense
              PTA_B(output, output_t, 2, 64)
            );

            {%- if vbe %}
            output = output.reshape({-1});
            {%- endif %}
            return;
          });
        });

        {#-/* End NoBag Check ***********************************************/#}
        {%- endif %}

        {%- if has_experimental %}
        // if (!is_experimental)
        } else {
            // Allocate num warps per table based on max_D
            const int num_warps_per_table = B * div_round_up(max_D, kWarpSize * 4);
            const uint32_t num_warps_per_threadblock = kForwardMaxThreads / kWarpSize;

            const auto kernel_func =
              (use_lxu_cache ? split_embedding_codegen_forward_{{ wdesc }}_v2_kernel<
                                  emb_t, cache_t, output_t, index_t, true>
                              : split_embedding_codegen_forward_{{ wdesc }}_v2_kernel<
                                  emb_t, cache_t, output_t, index_t, false>);

            FBGEMM_LAUNCH_KERNEL(
              kernel_func,
              div_round_up(T * num_warps_per_table, num_warps_per_threadblock),
              dim3(kWarpSize, num_warps_per_threadblock),
              0,
              at::cuda::getCurrentCUDAStream(),
              dev_weights.data_ptr<emb_t>(),
              uvm_weights.data_ptr<emb_t>(),
              lxu_cache_weights.data_ptr<cache_t>(),
              weights_placements.data_ptr<int32_t>(),
              B,
              T,
              static_cast<PoolingMode>(pooling_mode) == PoolingMode::MEAN,
              use_lxu_cache ? lxu_cache_weights.size(1) : 0,
              FixedDivisor(num_warps_per_table),
              indices.data_ptr<index_t>(),
              {%- if weighted %}
              // TODO: update indice_weights type
              indice_weights.data_ptr<float>(),
              {%- endif %}
              offsets.data_ptr<index_t>(),
              reinterpret_cast<uint32_t*>(D_offsets.data_ptr<int32_t>()),
              weights_offsets.data_ptr<int64_t>(),
              lxu_cache_locations.data_ptr<int32_t>(),
              output.data_ptr<output_t>()
            );
        }
        {%- endif %} // if has_experimental
        });
      });
  return output;
}


////////////////////////////////////////////////////////////////////////////////
// Op registrations
////////////////////////////////////////////////////////////////////////////////
{%- if not is_index_select %}
TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
    {%- set embedding_codegen_forward_op =
        "{}_embedding{}_codegen_forward_{}_cuda".format(
            mdesc, ndesc, desc_suffix
        )
    %}
    m.def("{{ embedding_codegen_forward_op }}("
          "    Tensor dev_weights, "
          {%- if not dense %}
          "    Tensor uvm_weights, "
          "    Tensor lxu_cache_weights, "
          "    Tensor weights_placements, "
          {%- endif %}
          "    Tensor weights_offsets, "
          {%- if nobag %}
          "    SymInt D, "
          {%- else %}
          "    Tensor D_offsets, "
          "    SymInt total_D, "
          "    SymInt max_D, "
          {%- endif %}
          "    Tensor indices, "
          "    Tensor offsets, "
          {%- if not nobag %}
          "    int pooling_mode, "
          {%- endif %}
          {%- if weighted %}
          "    Tensor indice_weights, "
          {%- endif %}
          {%- if not dense %}
          "    Tensor {{ locs_or_addrs_tensor }}, "
          "    Tensor uvm_cache_stats, "
          {%- endif %}
          "    int output_dtype, "
          {%- if vbe %}
          "    Tensor vbe_row_output_offsets, "
          "    Tensor vbe_b_t_map, "
          "    SymInt vbe_output_size, "
          "    int info_B_num_bits, "
          "    int info_B_mask_int64, "
          {%- endif %}
          {%- if is_gwd_kernel %}
          "    Tensor hash_size_cumsum, "
          "    Tensor prev_iter_dev, "
          "    Tensor learning_rate_tensor, "
          "    float weight_decay, "
          "    int iter, "
          "    float gwd_lower_bound, "
          {%- endif %}
          "    bool is_experimental"
          ") -> Tensor"
          {%- if not dense and not nobag and not vbe %}
          // only split_embedding_codegen_forward_[un]weighted_cuda
          // are tested to be PT2 compliant
          , {PT2_COMPLIANT_TAG}
          {%- endif %}
    );
    DISPATCH_TO_CUDA(
        "{{ embedding_codegen_forward_op }}",
        {{ embedding_codegen_forward_op }}
    );
}
{%- endif %} {#-/* if not is_index_select */#}
{%- endif %} {#-/* if is_valid_forward_config(...) */#}
{%- endfor %} {#-/* for nobag */#}
    // clang-format on
