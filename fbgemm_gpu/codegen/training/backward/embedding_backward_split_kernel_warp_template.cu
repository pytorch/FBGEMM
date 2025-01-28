/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// clang-format off
{%- set mdesc = "ssd" if ssd else "split" %}
{%- set wdesc = "weighted" if weighted else "unweighted" %}
{%- set ndesc = "_nobag" if nobag else "" %}
{%- set vdesc = "_vbe" if vbe else "" %}

{# /*
    `has_global_weight_decay_support` tells whether the optimizer has support for
    global weight decay (gwd)
    `is_gwd` is whether to generate gwd source code, determined in `generate_backward_split.py`
    For example, rowwise_adagrad has gwd support, so this template will be used twice:
    with `is_gwd` being True (for gwd kernel) and False (for regular kernel)
 */ #}
{%- set is_gwd_kernel = is_gwd and is_valid_gwd_config(
    dense,
    nobag,
    vbe,
    is_index_select,
    has_global_weight_decay_support,
    ssd) %}
{%- set gwddesc = "_gwd" if is_gwd_kernel else "" %}

{%- set desc_suffix = wdesc + vdesc + gwddesc %}

{%- set locs_or_addrs_tensor = "ssd_row_addrs" if ssd else "lxu_cache_locations" %}
{%- set locs_or_addrs_type = "int64_t" if ssd else "int32_t" %}

#include "fbgemm_gpu/embedding_backward_template_helpers.cuh"
#include "fbgemm_gpu/utils/tensor_accessor.h"
#include "fbgemm_gpu/split_embeddings_utils.cuh"
{%- if optimizer != "none" and not dense %}
#include "gen_embedding_optimizer_{{ optimizer }}_{{ mdesc }}_device_kernel.cuh"
{%- endif %}
#include "gen_embedding_backward_split_{{ kdesc }}_device_kernel.cuh"
#include "gen_embedding_backward_split_common_device_kernel.cuh"

using Tensor = at::Tensor;
using namespace fbgemm_gpu;

////////////////////////////////////////////////////////////////////////////////
// Kernel Template Definition
////////////////////////////////////////////////////////////////////////////////

{#- /*
    Generate a separate kernel to enable global weight decay using Jinja
    as the kernel is sensitive to the number of registers. Additional variables
    required to computer global weight decay increase number of registers and
    thus reduce the kernel occupancy, which can degrade the kernel performance.
    This increases the binary size, but the increase is minimal.
*/ #}

{%- set gwddesc = "_gwd" if is_gwd_kernel else "" %}
template <
    typename emb_t,
    typename grad_t,
    typename cache_t,
    typename index_t,
    {%- for ph_name in args.placeholder_tensor_names %}
    typename {{ ph_name + "_ph_t"}},
    {%- endfor %}
    int32_t kFixedMaxVecsPerThread,
    int32_t kThreadGroupSize,
    bool kUseVecBlocking>
__global__ __launch_bounds__(kBackwardMaxThreads) void
{%- if is_index_select %}
batch_index_select_dim0_codegen_backward_kernel_warp_per_row(
{%- else %}
{{ mdesc }}_embedding{{ ndesc }}_backward_codegen_{{ optimizer }}_{{ desc_suffix }}_kernel_warp_per_row_1(
{%- endif %}
    const pta::PackedTensorAccessor64<grad_t, {{ "1" if is_index_select else "2" }}, at::RestrictPtrTraits> grad_output,
    {%- if optimizer != "none" %}
    pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> dev_weights,
    {%- if not dense %}
    pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> uvm_weights,
    pta::PackedTensorAccessor64<cache_t, 2, at::RestrictPtrTraits> lxu_cache_weights,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> weights_placements,
    {%- endif %}
    {%- endif %}
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> weights_offsets,
    {%- if not nobag or is_index_select %}
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> D_offsets,
    {%- else %}
    int64_t D,
    {%- endif %}
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> hash_size_cumsum,
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> sorted_linear_indices_run,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> sorted_linear_indices_cumulative_run_lengths,
    {%- if not nobag %}
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> sorted_infos,
    {%- else %}
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> sorted_infos,
    {%- endif %}
    {%- if not dense %}
    const pta::PackedTensorAccessor32<{{ locs_or_addrs_type }}, 1, at::RestrictPtrTraits> sorted_{{ locs_or_addrs_tensor }},
    const bool use_uniq_cache_locations,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> table_unique_indices_offsets,
    {%- endif %}
    {%- if weighted %}
    const pta::PackedTensorAccessor32<at::acc_type<cache_t, true>, 1, at::RestrictPtrTraits> sorted_indice_weights,
    {%- endif %}
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> sorted_linear_indices_num_runs,
    int32_t max_segment_length_per_warp,
    {%- if not dense and optimizer != "none" %}
    bool stochastic_rounding,
    at::PhiloxCudaState stochastic_rounding_philox_args,
    {%- else %}
    pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> grad_dev_weights,
    {%- endif %} // if not dense and optimizer != "none"
    {%- if not nobag and vbe %}
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> B_offsets,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> row_output_offsets,
    {%- endif %}
    {%- if not nobag %}
    const int32_t info_B_num_bits,
    const uint32_t info_B_mask,
    {%- endif %}
    const int32_t max_D,
    const int32_t max_vecs_per_thread,
    {%- if is_gwd_kernel %}
    {%- if "prev_iter_dev" not in args.split_function_arg_names %}
    pta::PackedTensorAccessor64<float, 1, at::RestrictPtrTraits> prev_iter_dev,
    {%- endif %}
    {%- if "iter" not in args.split_function_arg_names %}
    const int64_t iter,
    {%- endif %}
    const float gwd_lower_bound,
    {%- endif %}
    {%- if is_index_select %}
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> grad_offsets,
    const bool permute_output_dim_0_1
    {%- else %}
    {{ args.split_kernel_args | replace_pta_namespace() | join(",\n    ") }}
    {%- endif %}
) {
    {%- if not nobag %}
    int32_t T = D_offsets.size(0) - 1;
    {%- else %}
    int32_t T = weights_offsets.size(0);
    {%- endif %}
    const int32_t start_run_id = blockIdx.x * blockDim.y + threadIdx.y;
    {%- if is_gwd_kernel %}
    const float weight_decay_base = 1 - learning_rate * weight_decay;
    {%- endif %}

#ifdef FBGEMM_USE_SUBWARP_SHUFFLE
    const unsigned int shfl_sync_mask =
        ((1L << kThreadGroupSize) - 1) <<
        (threadIdx.y % (kWarpSize / kThreadGroupSize) * kThreadGroupSize);
#else
    const unsigned int shfl_sync_mask = 0xffffffffu;
#endif
    constexpr int VEC_WIDTH = 4;
    constexpr auto kIsInt8 = std::is_same<emb_t, uint8_t>::value;

    struct SharedMemory<Vec4TAcc<cache_t>> smem;
    const int32_t grad_sum_stride = max_D / VEC_WIDTH;
    auto* smem_grad_sum = (kUseVecBlocking || kIsInt8)
      ? smem.getPointer() + threadIdx.y * grad_sum_stride
      : nullptr;

    for (uint32_t run_id = start_run_id;
         run_id < sorted_linear_indices_run.size(0) && run_id < sorted_linear_indices_num_runs[0];
             run_id += gridDim.x * blockDim.y) {

        const int64_t linear_index = sorted_linear_indices_run[run_id];
        const int32_t segment_start =
            sorted_linear_indices_cumulative_run_lengths[run_id];
        const int32_t segment_end =
            sorted_linear_indices_cumulative_run_lengths[run_id + 1];
        const int32_t SL = segment_end - segment_start;


        if (SL >= max_segment_length_per_warp) {
            continue;
        }

        // now, each segment corresponds to exactly one table `t` and row in
        // that table (`idx`). Thus, we can hoist out some of the book-keeping.
        {%- if not nobag %}
        const auto info_0 = reinterpret_cast<const uint32_t*>(&sorted_infos[0])[segment_start];
        const auto t_0 = info_0 >> info_B_num_bits;
        {%- else %}
        const auto info_0 = sorted_infos[segment_start];
        int32_t t_0 = info_0 % T;
        {%- endif %}

        int64_t hash_size = hash_size_cumsum[t_0];
        {%- if not nobag or is_index_select %}
        const auto D_start_t0 = D_offsets[t_0];
        // D can be hoisted here because D is the same if features share the
        // same table, but D_start is different
        const int32_t D = D_offsets[t_0 + 1] - D_start_t0;
        {%- if is_index_select %}
        // grad_offset can be hoisted here for batch_index_select because it
        // does not allow multiple features to share a single embedding table
        const auto grad_offset = permute_output_dim_0_1 ? D_start_t0 : grad_offsets[t_0];
        const auto grad_stride = permute_output_dim_0_1 ? D_offsets[T] : D;
        {%- endif %}
        {%- endif %}
        int64_t idx = linear_index - hash_size;

        {{ compute_global_weight_decay(is_gwd_kernel) }}

        const int32_t SL_per_warp = div_round_up(SL, blockDim.y);
        const int32_t sl_start = 0;
        const int32_t sl_end = SL;
        Vec4TAcc<cache_t> grad_sum[kFixedMaxVecsPerThread];
        constexpr int32_t kGroupVecWidth = kThreadGroupSize * VEC_WIDTH;
        const int32_t num_vecs = (D + kGroupVecWidth - 1) / kGroupVecWidth;

        compute_grad_sum_{{ kdesc }}<
          grad_t,
          cache_t,
          kFixedMaxVecsPerThread,
          kThreadGroupSize,
          VEC_WIDTH,
          kUseVecBlocking>(
            grad_sum,
            smem_grad_sum,
            grad_output,
            {%- if not nobag or is_index_select %}
            D_offsets,
            {%- endif %}
            D,
            T,
            sorted_infos,
            {%- if weighted %}
            sorted_indice_weights,
            {%- endif %}
            {%- if not nobag and vbe %}
            B_offsets,
            row_output_offsets,
            {%- endif %}
            {%- if is_index_select %}
            grad_offset,
            grad_stride,
            {%- endif %}
            {%- if not nobag %}
            info_B_num_bits,
            info_B_mask,
            {%- endif %}
            segment_start,
            sl_start,
            sl_end,
            shfl_sync_mask,
            num_vecs
        );

        // Copy value to max_vecs to make max_vecs_per_thread known at compile time
        // when kUseVecBlocking == false
        const int32_t max_vecs =
            kUseVecBlocking ? max_vecs_per_thread : kFixedMaxVecsPerThread;

        {%- if not dense and optimizer != "none" %}
        {{ mdesc }}_{{ optimizer }}_table_update_kernel<
          emb_t,
          cache_t,
          {%- for ph_name in args.placeholder_tensor_names %}
          {{ ph_name + "_ph_t" }},
          {%- endfor %}
          kFixedMaxVecsPerThread,
          kThreadGroupSize,
          VEC_WIDTH,
          kUseVecBlocking>(
              dev_weights,
              uvm_weights,
              lxu_cache_weights,
              weights_placements,
              weights_offsets,
              sorted_{{ locs_or_addrs_tensor }},
              grad_sum,
              smem_grad_sum,
              smem_grad_sum, // shared_weight_update_row (reuse smem_grad_sum)
              stochastic_rounding,
              stochastic_rounding_philox_args,
              run_id,
              use_uniq_cache_locations
                  ? (run_id - table_unique_indices_offsets[t_0])
                  : segment_start,
              D,
              t_0,
              idx,
              {%- if is_gwd_kernel %}
              global_weight_decay,
              {%- elif has_global_weight_decay_support %}
              {# /* cases where gwd is not enabled/supported */ #}
              1, // global_weight_decay
              {%- endif %}
              shfl_sync_mask,
              max_vecs,
              {{ args.split_kernel_arg_names | join(", ") }}
        );
        {%- else %}
        // Write deduplicated gradient to grad_dev_weights gradient is sparse
        // for split_embedding and dense for dense_embedding
        {%- if dense %}
        const int64_t weights_offset = weights_offsets[t_0];
        {%- else %}
        // Compute offset of sparse gradient
        const int64_t weights_offset = run_id * max_D;
        idx = 0;
        {%- endif %}
        store_grad_sum<
            emb_t,
            cache_t,
            kFixedMaxVecsPerThread,
            kThreadGroupSize,
            VEC_WIDTH,
            kUseVecBlocking>(
              grad_dev_weights,
              grad_sum,
              kUseVecBlocking ? smem_grad_sum : nullptr,
              D,
              weights_offset,
              idx,
              max_vecs
        );
        {%- endif %} // if not dense and optimizer != "none"
    }
}


////////////////////////////////////////////////////////////////////////////////
// Explicit Template Instantiations
////////////////////////////////////////////////////////////////////////////////

/*
    Explicitly instantiate the kernel function template.  The instantiations are
    based on the types enumerated by DISPATCH_EMB_GRAD_CACHE_TYPES macro used in
    embedding_backward_split_template.cu
*/

{%- macro template_instantiation(
      emb_type,
      grad_type,
      cache_type,
      index_type,
      ph_type_combo,
      kFixedMaxVecsPerThread,
      kThreadGroupSize,
      kUseVecBlocking
    )
%}

{%- set gwddesc = "_gwd" if is_gwd_kernel else "" %}
template __global__ __launch_bounds__(kBackwardMaxThreads) void
{%- if is_index_select %}
batch_index_select_dim0_codegen_backward_kernel_warp_per_row
{%- else %}
{{ mdesc }}_embedding{{ ndesc }}_backward_codegen_{{ optimizer }}_{{ desc_suffix }}_kernel_warp_per_row_1
{%- endif %}
< {{ emb_type }},
  {{ grad_type }},
  {{ cache_type }},
  {{ index_type }},
  {%- for ph_name in args.placeholder_tensor_names %}
  {{ ph_type_combo[ph_name].primitive_type }},
  {%- endfor %}
  {{ kFixedMaxVecsPerThread }},
  {{ kThreadGroupSize }},
  {{ kUseVecBlocking }}
> (
    const pta::PackedTensorAccessor64<{{ grad_type }}, {{ "1" if is_index_select else "2" }}, at::RestrictPtrTraits> grad_output,
    {%- if optimizer != "none" %}
    pta::PackedTensorAccessor64<{{ emb_type }}, 1, at::RestrictPtrTraits> dev_weights,
    {%- if not dense %}
    pta::PackedTensorAccessor64<{{ emb_type }}, 1, at::RestrictPtrTraits> uvm_weights,
    pta::PackedTensorAccessor64<{{ cache_type }}, 2, at::RestrictPtrTraits> lxu_cache_weights,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> weights_placements,
    {%- endif %}
    {%- endif %}
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> weights_offsets,
    {%- if not nobag or is_index_select %}
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> D_offsets,
    {%- else %}
    int64_t D,
    {%- endif %}
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> hash_size_cumsum,
    const pta::PackedTensorAccessor32<{{ index_type }}, 1, at::RestrictPtrTraits> sorted_linear_indices_run,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> sorted_linear_indices_cumulative_run_lengths,
    {%- if not nobag %}
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> sorted_infos,
    {%- else %}
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> sorted_infos,
    {%- endif %}
    {%- if not dense %}
    const pta::PackedTensorAccessor32<{{ locs_or_addrs_type }}, 1, at::RestrictPtrTraits> sorted_{{ locs_or_addrs_tensor }},
    const bool use_uniq_cache_locations,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> table_unique_indices_offsets,
    {%- endif %}
    {%- if weighted %}
    const pta::PackedTensorAccessor32<at::acc_type<{{ cache_type }}, true>, 1, at::RestrictPtrTraits> sorted_indice_weights,
    {%- endif %}
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> sorted_linear_indices_num_runs,
    int32_t max_segment_length_per_warp,
    {%- if not dense and optimizer != "none" %}
    bool stochastic_rounding,
    at::PhiloxCudaState stochastic_rounding_philox_args,
    {%- else %}
    pta::PackedTensorAccessor64<{{ emb_type }}, 1, at::RestrictPtrTraits> grad_dev_weights,
    {%- endif %} // if not dense and optimizer != "none"
    {%- if not nobag and vbe %}
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> B_offsets,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> row_output_offsets,
    {%- endif %}
    {%- if not nobag %}
    const int32_t info_B_num_bits,
    const uint32_t info_B_mask,
    {%- endif %}
    const int32_t max_D,
    const int32_t max_vecs_per_thread,
    {%- if is_gwd_kernel %}
    {%- if "prev_iter_dev" not in args.split_function_arg_names %}
    const pta::PackedTensorAccessor64<float, 1, at::RestrictPtrTraits> prev_iter_dev,
    {%- endif %}
    {%- if "iter" not in args.split_function_arg_names %}
    const int64_t iter,
    {%- endif %}
    const float gwd_lower_bound,
    {%- endif %}
    {%- if is_index_select %}
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> grad_offsets,
    const bool permute_output_dim_0_1
    {%- else %}
    {{ args.split_kernel_args_no_defaults |
         replace_pta_namespace() |
         replace_placeholder_types(ph_type_combo) |
         join(",\n    ") |
         replace("cache_t", cache_type)
    }}
    {%- endif %}
);
{%- endmacro %}

{%- macro bulk_template_instantiations(kFixedMaxVecsPerThread, kThreadGroupSize, kUseVecBlocking) %}
    {%- for grad_type in ['float', 'at::Half', 'at::BFloat16'] %}
    {%- for emb_type in ['float', 'at::Half'] %}
    {%- for cache_type in ['float', 'at::Half'] %}
    {%- for index_type in ['int32_t', 'int64_t'] %}
    {%- for ph_type_combo in args.placeholder_type_combos %}
        {{ template_instantiation(
            emb_type,
            grad_type,
            cache_type,
            index_type,
            ph_type_combo,
            kFixedMaxVecsPerThread,
            kThreadGroupSize,
            kUseVecBlocking
          )
        }}
    {%- endfor %}
    {%- endfor %}
    {%- endfor %}
    {%- endfor %}
    {%- endfor %}
{%- endmacro %}


{%- if is_experimental_optimizer %}

{{
  bulk_template_instantiations(
    fixed_max_vecs_per_thread["backward"],
    'kWarpSize',
    'true'
  )
}}

{%- else %}

{%- macro instantiate_templates(use_subwarp_shuffle) %}
{%- for (kFixedMaxVecsPerThread, kThreadGroupSize, kUseVecBlocking)
    in get_max_vecs_template_configs(
        items_per_warp,
        fixed_max_vecs_per_thread["backward"],
        use_subwarp_shuffle,
        use_vec_blocking=True,
    )
%}
    {{
      bulk_template_instantiations(
        kFixedMaxVecsPerThread,
        kThreadGroupSize,
        kUseVecBlocking,
      )
    }}
{%- endfor %}
{%- endmacro %}


////////////////////////////////////////////////////////////////////////////////
#ifdef FBGEMM_USE_SUBWARP_SHUFFLE
////////////////////////////////////////////////////////////////////////////////

{#- /*
    Explicitly instantiate kernels for the FBGEMM_USE_SUBWARP_SHUFFLE case

    Please see get_max_vecs_template_configs in
    codegen/embedding_common_code_generator.py for more details
*/ #}

{{ instantiate_templates(use_subwarp_shuffle=True) }}

////////////////////////////////////////////////////////////////////////////////
#else
////////////////////////////////////////////////////////////////////////////////

{#- /*
    Explicitly instantiate kernels for the non-FBGEMM_USE_SUBWARP_SHUFFLE case

    Please see get_max_vecs_template_configs in
    codegen/embedding_common_code_generator.py for more details
*/ #}

{{ instantiate_templates(use_subwarp_shuffle=False) }}

////////////////////////////////////////////////////////////////////////////////
#endif
////////////////////////////////////////////////////////////////////////////////

{%- endif %}

{%- if is_rocm and not is_index_select and optimizer == "rowwise_adagrad" and not dense and not is_gwd_kernel and not vbe and not ssd %}
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "fbgemm_gpu/rocm/split_embeddings_common.h"
#include "gen_embedding_backward_split_{{ desc_suffix }}{{ ndesc }}_device_kernel_hip.hip"

template <
    typename emb_t,
    typename grad_t,
    typename cache_t,
    typename index_t,
    int32_t kFixedMaxVecsPerThread,
    int32_t kThreadGroupSize,
    bool kUseVecBlocking,
    int32_t embedding_dim,
    int32_t weight_decay_mode_v>
__global__ void
hip_split_embedding{{ ndesc }}_backward_codegen_{{ optimizer }}_{{ wdesc }}{{ vdesc }}_kernel_warp_per_row_1(
    const pta::PackedTensorAccessor64<grad_t, {{ "1" if is_index_select else "2" }}, at::RestrictPtrTraits> grad_output,
    {%- if optimizer != "none" %}
    pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> dev_weights,
    {%- if not dense %}
    pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> uvm_weights,
    pta::PackedTensorAccessor64<cache_t, 2, at::RestrictPtrTraits> lxu_cache_weights,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> weights_placements,
    {%- endif %}
    {%- endif %}
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> weights_offsets,
    {%- if not nobag or is_index_select %}
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> D_offsets,
    {%- else %}
    int64_t D,
    {%- endif %}
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> hash_size_cumsum,
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> sorted_linear_indices_run,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> sorted_linear_indices_cumulative_run_lengths,
    {%- if not nobag %}
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> sorted_infos,
    {%- else %}
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> sorted_infos,
    {%- endif %}
    {%- if not dense %}
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> sorted_lxu_cache_locations,
    const bool use_uniq_cache_locations,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> table_unique_indices_offsets,
    {%- endif %}
    {%- if weighted %}
    const pta::PackedTensorAccessor32<at::acc_type<cache_t, true>, 1, at::RestrictPtrTraits> sorted_indice_weights,
    {%- endif %}
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> sorted_linear_indices_num_runs,
    int32_t max_segment_length_per_warp,
    {%- if not dense and optimizer != "none" %}
    bool stochastic_rounding,
    at::PhiloxCudaState stochastic_rounding_philox_args,
    {%- else %}
    pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> grad_dev_weights,
    {%- endif %} // if not dense and optimizer != "none"
    {%- if not nobag and vbe %}
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> B_offsets,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> row_output_offsets,
    {%- endif %}
    {%- if not nobag %}
    const int32_t info_B_num_bits,
    const uint32_t info_B_mask,
    {%- endif %}
    const int32_t max_D,
    const int32_t max_vecs_per_thread,
    {%- if is_index_select %}
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> grad_offsets,
    const bool permute_output_dim_0_1
    {%- else %}
    {{ args.split_kernel_args | replace_pta_namespace() | join(",\n    ") }}
    {%- endif %}
) {
    {%- if not nobag %}
    int32_t T = D_offsets.size(0) - 1;
    {%- else %}
    int32_t T = weights_offsets.size(0);
    {%- endif %}

    auto p_output_grad = grad_output.data();
    auto p_emb_table = dev_weights.data();
    auto p_hash_size_cumsum = hash_size_cumsum.data();
    auto p_sorted_linear_indices_run = sorted_linear_indices_run.data();
    auto p_sorted_linear_indices_cumulative_run_lengths = sorted_linear_indices_cumulative_run_lengths.data();
    auto p_sorted_linear_indices_num_runs = sorted_linear_indices_num_runs.data();
    auto p_sorted_infos = sorted_infos.data();
    {%- if weighted %}
    auto p_indice_weights_sorted = sorted_indice_weights.data();
    {%- endif %}
    auto emb_dim = embedding_dim;
    constexpr int32_t segment_prefetch = 2;
    constexpr int32_t segment_unroll = 8;
    constexpr int32_t segment_split = 0;
    auto batch = grad_output.size(0);
    auto num_rows = dev_weights.size(0) / T / max_D;
    {%- if weighted %}
    constexpr bool is_weighted = true;
    {%- else %}
    constexpr bool is_weighted = false;
    {%- endif %}
    rocm::{{optimizer}}_kernel_arg_t opt_karg;
    opt_karg.p_momentum = momentum1_dev.data();
    opt_karg.eps = eps;
    opt_karg.learning_rate = learning_rate;
    // weight_decay(_mode) is supplied as args.split_function_args_no_defaults
    opt_karg.weight_decay_mode = weight_decay_mode_v;
    opt_karg.weight_decay = weight_decay;
    auto batch_mdiv = [](uint32_t d) -> rocm::magic_div_u32_t {
        assert(d >= 1 && d <= INT32_MAX);
        uint8_t shift;
        for(shift = 0; shift < 32; shift++)
            if((1U << shift) >= d)
                break;

        uint64_t one   = 1;
        uint64_t magic = ((one << 32) * ((one << shift) - d)) / d + 1;
        assert(magic <= 0xffffffffUL);

        rocm::magic_div_u32_t result;
        result.magic = magic;
        result.shift = shift;
        return result;
    }(batch);
    rocm::split_tbe_backward_hip_kernel_{{kdesc}}<
        rocm::{{optimizer}}_optimizer_t<cache_t, emb_t, embedding_dim, weight_decay_mode_v>,
        rocm::{{optimizer}}_kernel_arg_t,
        emb_t,
        cache_t,
        grad_t,
        index_t,
        BLOCK_SIZE,
        embedding_dim,
        segment_prefetch,
        segment_unroll,
        segment_split,
        is_weighted>(p_output_grad,
               p_emb_table,
               p_hash_size_cumsum,
               p_sorted_linear_indices_run,
               p_sorted_linear_indices_cumulative_run_lengths,
               p_sorted_linear_indices_num_runs,
               {%- if not nobag %}
               info_B_num_bits,
               info_B_mask,
               {%- endif %}
               p_sorted_infos,
               batch_mdiv,
               max_segment_length_per_warp,
               emb_dim,
               batch,
               num_rows,
               T,
               opt_karg
               {%- if weighted %}
               , p_indice_weights_sorted
               {%- endif %});
}

{%- macro hip_template_instantiation(
      emb_type,
      grad_type,
      cache_type,
      index_type,
      kFixedMaxVecsPerThread,
      kThreadGroupSize,
      kUseVecBlocking,
      kEmbeddingDim,
      kWeighDecayMode
    )
%}
template __global__ __launch_bounds__(kBackwardMaxThreads) void
hip_split_embedding{{ ndesc }}_backward_codegen_{{ optimizer }}_{{ wdesc }}{{ vdesc }}_kernel_warp_per_row_1
< {{ emb_type }},
  {{ grad_type }},
  {{ cache_type }},
  {{ index_type }},
  {{ kFixedMaxVecsPerThread }},
  {{ kThreadGroupSize }},
  {{ kUseVecBlocking }},
  {{ kEmbeddingDim }},
  {{ kWeighDecayMode }}
> (
    const pta::PackedTensorAccessor64<{{ grad_type }}, {{ "1" if is_index_select else "2" }}, at::RestrictPtrTraits> grad_output,
    {%- if optimizer != "none" %}
    pta::PackedTensorAccessor64<{{ emb_type }}, 1, at::RestrictPtrTraits> dev_weights,
    {%- if not dense %}
    pta::PackedTensorAccessor64<{{ emb_type }}, 1, at::RestrictPtrTraits> uvm_weights,
    pta::PackedTensorAccessor64<{{ cache_type }}, 2, at::RestrictPtrTraits> lxu_cache_weights,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> weights_placements,
    {%- endif %}
    {%- endif %}
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> weights_offsets,
    {%- if not nobag or is_index_select %}
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> D_offsets,
    {%- else %}
    int64_t D,
    {%- endif %}
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> hash_size_cumsum,
    const pta::PackedTensorAccessor32<{{ index_type }}, 1, at::RestrictPtrTraits> sorted_linear_indices_run,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> sorted_linear_indices_cumulative_run_lengths,
    {%- if not nobag %}
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> sorted_infos,
    {%- else %}
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> sorted_infos,
    {%- endif %}
    {%- if not dense %}
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> sorted_lxu_cache_locations,
    const bool use_uniq_cache_locations,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> table_unique_indices_offsets,
    {%- endif %}
    {%- if weighted %}
    const pta::PackedTensorAccessor32<at::acc_type<{{ cache_type }}, true>, 1, at::RestrictPtrTraits> sorted_indice_weights,
    {%- endif %}
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> sorted_linear_indices_num_runs,
    int32_t max_segment_length_per_warp,
    {%- if not dense and optimizer != "none" %}
    bool stochastic_rounding,
    at::PhiloxCudaState stochastic_rounding_philox_args,
    {%- else %}
    pta::PackedTensorAccessor64< {{ emb_type }}, 1, at::RestrictPtrTraits> grad_dev_weights,
    {%- endif %} // if not dense and optimizer != "none"
    {%- if not nobag and vbe %}
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> B_offsets,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> row_output_offsets,
    {%- endif %}
    {%- if not nobag %}
    const int32_t info_B_num_bits,
    const uint32_t info_B_mask,
    {%- endif %}
    const int32_t max_D,
    const int32_t max_vecs_per_thread,
    {%- if is_index_select %}
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> grad_offsets,
    const bool permute_output_dim_0_1
    {%- else %}
    {{ args.split_kernel_args_no_defaults | replace_pta_namespace() | join(",\n    ") | replace("cache_t", cache_type) }}
    {%- endif %}
);
{%- endmacro %}

{%- macro hip_bulk_template_instantiations(kFixedMaxVecsPerThread, kThreadGroupSize, kUseVecBlocking) %}
    {%- for grad_type in ['float', 'at::Half', 'at::BFloat16'] %}
    {%- for emb_type in ['float', 'at::Half'] %}
    {%- for cache_type in ['float', 'at::Half'] %}
    {%- for index_type in ['int32_t', 'int64_t'] %}
    {%- for kEmbeddingDim in [64, 128, 160, 192, 256] %}
    {%- for kWeighDecayMode in [0, 1, 2] %}
        {{ hip_template_instantiation(
            emb_type,
            grad_type,
            cache_type,
            index_type,
            kFixedMaxVecsPerThread,
            kThreadGroupSize,
            kUseVecBlocking,
            kEmbeddingDim,
            kWeighDecayMode
          )
        }}
    {%- endfor %}
    {%- endfor %}
    {%- endfor %}
    {%- endfor %}
    {%- endfor %}
    {%- endfor %}
{%- endmacro %}

{%- macro hip_instantiate_templates(use_subwarp_shuffle) %}
{%- for (kFixedMaxVecsPerThread, kThreadGroupSize, kUseVecBlocking)
    in get_max_vecs_template_configs(
        items_per_warp,
        fixed_max_vecs_per_thread["backward"],
        use_subwarp_shuffle,
        use_vec_blocking=True,
    )
%}
    {{
      hip_bulk_template_instantiations(
        kFixedMaxVecsPerThread,
        kThreadGroupSize,
        kUseVecBlocking,
      )
    }}
{%- endfor %}
{%- endmacro %}

////////////////////////////////////////////////////////////////////////////////
#ifdef FBGEMM_USE_SUBWARP_SHUFFLE
////////////////////////////////////////////////////////////////////////////////

{#- /*
    Explicitly instantiate kernels for the FBGEMM_USE_SUBWARP_SHUFFLE case
    Please see get_max_vecs_template_configs in
    codegen/embedding_common_code_generator.py for more details
*/ #}

{{ hip_instantiate_templates(use_subwarp_shuffle=True) }}

////////////////////////////////////////////////////////////////////////////////
#else
////////////////////////////////////////////////////////////////////////////////

{#- /*
    Explicitly instantiate kernels for the non-FBGEMM_USE_SUBWARP_SHUFFLE case
    Please see get_max_vecs_template_configs in
    codegen/embedding_common_code_generator.py for more details
*/ #}

{{ hip_instantiate_templates(use_subwarp_shuffle=False) }}

////////////////////////////////////////////////////////////////////////////////
#endif
////////////////////////////////////////////////////////////////////////////////
{%- endif %}
        // clang-format on
