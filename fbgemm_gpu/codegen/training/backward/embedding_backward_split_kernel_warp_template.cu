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
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> sorted_linear_indices_run,
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
              {{ args.split_function_arg_names | join(", ") }}
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
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> sorted_linear_indices_run,
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
    {%- for ph_type_combo in args.placeholder_type_combos %}
        {{ template_instantiation(
            emb_type,
            grad_type,
            cache_type,
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

{%- if is_rocm and not is_index_select and optimizer == "rowwise_adagrad" and not dense %}
// PR23: ROCM
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "fbgemm_gpu/hip_kernel_inc/split_tbe_common.h"

template <typename cache_t, typename emb_t, int32_t embedding_dim, int32_t weight_decay_mode>
struct rowwise_adagrad_optimizer_t
{
    __device__ rowwise_adagrad_optimizer_t(const rowwise_adagrad_kernel_arg_t& karg_)
        : karg(karg_)
    {
    }

    // template<int32_t acc_length>
    // __device__ static void precompute(float * acc){
    //     // compute per row square sum
    // }
    template <int32_t thread_length, int32_t segment_split>
    __device__ void update(cache_t* acc, emb_t* weight, uint32_t row_index)
    {
        if constexpr(segment_split == 0)
        {
            cache_t * p_momentum = reinterpret_cast<cache_t*>(karg.p_momentum);
            cache_t momentum = p_momentum[row_index]; // should be s_load
            // compute per row square sum
            cache_t local_sum_squre = .0f;
            if constexpr(weight_decay_mode == 1)
            {
#pragma unroll
                for(auto i = 0; i < thread_length; i++)
                {
                    cache_t w = static_cast<cache_t>(weight[i]);
                    cache_t a = acc[i] + w * karg.weight_decay;
                    local_sum_squre += a * a;
                }
            }
            else
            {
#pragma unroll
                for(auto i = 0; i < thread_length; i++)
                {
                    cache_t a = acc[i];
                    local_sum_squre += a * a;
                }
            }

            cache_t avg_square =
                wave_reduce<reduce_op_sum_t<cache_t>, cache_t, AMDGCN_WAVE_SIZE>(local_sum_squre) /
                embedding_dim;

            cache_t momentum_new = momentum + avg_square;

            cache_t multiplier = karg.learning_rate / (sqrtf(momentum_new) + karg.eps);
            cache_t correction;

            if constexpr(weight_decay_mode == 1)
            {
                correction = 1.0 - multiplier * karg.weight_decay;
            }
            else if constexpr(weight_decay_mode == 2)
            {
                correction = 1.0 - karg.learning_rate * karg.weight_decay;
            }
            else
            {
                correction = 1.0;
            }

// update new weight value
#pragma unroll
            for(auto i = 0; i < thread_length; i++)
            {
                cache_t w = static_cast<cache_t>(weight[i]);
                cache_t a = acc[i];
                w         = correction * w - multiplier * a;
                weight[i] = static_cast<emb_t>(w);
            }

            // printf("momentum_new:%f, avg_square:%f, row_index:%d, momentum:%f\n",  momentum_new,  avg_square, row_index, momentum);
            // printf("momentum_new:%f",  momentum_new);

            p_momentum[row_index] = momentum_new;
        }
    }

    rowwise_adagrad_kernel_arg_t karg;
};

template <typename optimizer_t,
          typename optimizer_karg_t,
          typename emb_t,
          typename cache_t,
          typename grad_t,
          int32_t block_size,
          int32_t embedding_dim,
          int32_t segment_prefetch, // 2
          int32_t segment_unroll, // 8
          int32_t segment_split,  // 0-warp per row, 1-cta per row, 2-atomic(needed?)
          bool    weighted>
__device__ void split_tbe_backward_hip_kernel{{ ndesc }}(
    const grad_t* p_output_grad,
    emb_t* p_emb_table,
    const int64_t* p_hash_size_cumsum,
    const int64_t* p_sorted_linear_indices_run,
    const int32_t* p_sorted_linear_indices_cumulative_run_lengths,
    const int32_t* p_sorted_linear_indices_num_runs,
    // const int32_t* p_long_run_ids,  // unused
    // const int32_t* p_num_long_run_ids, // unused
    {%- if not nobag %}
    const int32_t info_B_num_bits,
    const uint32_t info_B_mask,
    {%- endif %}
    {%- if not nobag %}
    const int32_t* p_sorted_infos,  // FIXME: this is for not nobag, TODO support nobag
    {%- else %}
    const int64_t* p_sorted_infos,
    {%- endif %}
    magic_div_u32_t batch_mdiv,
    uint32_t max_segment_length_per_warp,
    uint32_t emb_dim,
    uint32_t batch,
    uint32_t num_rows,
    uint32_t num_tables,
    optimizer_karg_t opt_karg,
    const float * p_sorted_indice_weights = nullptr)
{
    constexpr uint32_t dword_per_row   = (embedding_dim + THREADS_PER_ROW - 1) / THREADS_PER_ROW;
    constexpr uint32_t waves_per_block = block_size / AMDGCN_WAVE_SIZE;
    constexpr uint32_t length_mask     = ~(segment_unroll - 1);
    const uint32_t wave_id = __builtin_amdgcn_readfirstlane(threadIdx.x / AMDGCN_WAVE_SIZE);
    const uint32_t lane_id = threadIdx.x % AMDGCN_WAVE_SIZE;
    const uint32_t run_id  = wave_id + blockIdx.x * waves_per_block;

    // printf("wave_id:%d, run_id:%d(%d), batch:%d(%d, %d)\n",
    //     wave_id, run_id, p_sorted_linear_indices_num_runs[0], batch, batch_mdiv.magic, batch_mdiv.shift);

    if(run_id >= p_sorted_linear_indices_num_runs[0])
        return;

    const int64_t linear_index  = p_sorted_linear_indices_run[run_id];

    const int32_t segment_start = p_sorted_linear_indices_cumulative_run_lengths[run_id];
    const int32_t segment_end   = p_sorted_linear_indices_cumulative_run_lengths[run_id + 1];

    // PR23 FIXME: support nobag
    // avbokovoy: WIP
    {%- if nobag %}
    const auto info_0 = p_sorted_infos[segment_start];
    int32_t t_0 = info_0 % num_tables;
    //magic_div_u32_run(batch_mdiv, info_0);
    {%- else %}
    const auto info_0 = reinterpret_cast<const uint32_t*>(&p_sorted_infos[0])[segment_start];
    const auto t_0 = info_0 >> info_B_num_bits;
    {%- endif %}
    int64_t hash_size = p_hash_size_cumsum[t_0];

    const int64_t emb_idx       = linear_index - hash_size;

    // printf("[%d] segment_start:%d, info_0:%d, t_0:%d, num_rows:%d, emb_dim:%d, linear_index:%ld\n", wave_id, segment_start, info_0, t_0, num_rows, emb_dim, linear_index);

    // p_output_grad += t_0 * emb_dim;

    p_emb_table += hash_size * emb_dim;
    opt_karg.p_momentum = reinterpret_cast<void*>(reinterpret_cast<cache_t*>(opt_karg.p_momentum) + hash_size);

    const int32_t segment_length = segment_end - segment_start;

    if(segment_length >= max_segment_length_per_warp)
        return;

    // printf("[%d] segment_length:%d\n", wave_id, segment_length);

    const int32_t segment_length_mod = segment_length & length_mask;

    cache_t grad_acc[dword_per_row];
    int32_t infos[segment_unroll];
    grad_t grad_data[dword_per_row * segment_prefetch];
    emb_t emb_data[dword_per_row];
    float indice_weights[segment_unroll];

    #pragma unroll
    for(int i=0; i < dword_per_row; i++)
    {
        grad_acc[i] = .0f;
    }

    int itr = 0;
    if(segment_length_mod == 0)
        goto L_tail_grad_acc;

if constexpr (!weighted) {
    #pragma unroll
    for(int i = 0; i < segment_unroll; i++)
    {
        infos[i] = p_sorted_infos[segment_start + i];
    }
} else {
    for(int i = 0; i < segment_unroll; i++)
    {
        infos[i] = p_sorted_infos[segment_start + i];
        indice_weights[i] = p_sorted_indice_weights[segment_start + i];
    }
}

    itr += segment_unroll;
    p_sorted_infos += segment_unroll;

if constexpr (weighted) {
    p_sorted_indice_weights += segment_unroll;
}

    uint32_t bag_index;
    uint32_t table_index;

    // LOOP
    for(; itr < segment_length_mod; itr += segment_unroll)
    {
        magic_div_u32_run_with_mod(batch_mdiv, infos[0], batch, table_index, bag_index);
        load_row_per_warp<grad_t, embedding_dim, int32_t>::run(
            &grad_data[0], bag_index * num_tables, p_output_grad + table_index * embedding_dim, lane_id);

        magic_div_u32_run_with_mod(batch_mdiv, infos[1], batch, table_index, bag_index);
        load_row_per_warp<grad_t, embedding_dim, int32_t>::run(
            &grad_data[dword_per_row], bag_index * num_tables, p_output_grad + table_index * embedding_dim, lane_id);
        if constexpr (!weighted){
            #pragma unroll
            for(int j = 2; j < segment_unroll; j += 2)
            {
                accumulate_row_per_warp<grad_t, embedding_dim, cache_t, weighted>::run(
                    &grad_acc[0], &grad_data[0], lane_id);
                magic_div_u32_run_with_mod(batch_mdiv, infos[j], batch, table_index, bag_index);
                load_row_per_warp<grad_t, embedding_dim, int32_t>::run(
                    &grad_data[0], bag_index * num_tables, p_output_grad + table_index * embedding_dim, lane_id);

                accumulate_row_per_warp<grad_t, embedding_dim, cache_t, weighted>::run(
                    &grad_acc[0], &grad_data[dword_per_row], lane_id);
                magic_div_u32_run_with_mod(
                    batch_mdiv, infos[j + 1], batch, table_index, bag_index);
                load_row_per_warp<grad_t, embedding_dim, int32_t>::run(
                    &grad_data[dword_per_row], bag_index * num_tables, p_output_grad + table_index * embedding_dim, lane_id);
            }

            accumulate_row_per_warp<grad_t, embedding_dim, cache_t, weighted>::run(
                &grad_acc[0], &grad_data[0], lane_id);
            accumulate_row_per_warp<grad_t, embedding_dim, cache_t, weighted>::run(
                &grad_acc[0], &grad_data[dword_per_row], lane_id);

            #pragma unroll
            for(int i = 0; i < segment_unroll; i++)
            {
                infos[i] = p_sorted_infos[segment_start + i];
            }
            p_sorted_infos += segment_unroll;


        } else {
            #pragma unroll
            for(int j = 2; j < segment_unroll; j += 2)
            {
                accumulate_row_per_warp<grad_t, embedding_dim, cache_t, weighted>::run(
                    &grad_acc[0], &grad_data[0], lane_id, indice_weights[j-2]);
                magic_div_u32_run_with_mod(batch_mdiv, infos[j], batch, table_index, bag_index);
                load_row_per_warp<grad_t, embedding_dim, int32_t>::run(
                    &grad_data[0], bag_index * num_tables, p_output_grad + table_index * embedding_dim, lane_id);

                accumulate_row_per_warp<grad_t, embedding_dim, cache_t, weighted>::run(
                    &grad_acc[0], &grad_data[dword_per_row], lane_id, indice_weights[j-1]);
                magic_div_u32_run_with_mod(
                    batch_mdiv, infos[j + 1], batch, table_index, bag_index);
                load_row_per_warp<grad_t, embedding_dim, int32_t>::run(
                    &grad_data[dword_per_row], bag_index * num_tables, p_output_grad + table_index * embedding_dim, lane_id);
            }

            accumulate_row_per_warp<grad_t, embedding_dim, cache_t, weighted>::run(
                &grad_acc[0], &grad_data[0], lane_id, indice_weights[segment_unroll-2]);
            accumulate_row_per_warp<grad_t, embedding_dim, cache_t, weighted>::run(
                &grad_acc[0], &grad_data[dword_per_row], lane_id, indice_weights[segment_unroll-1]);

            #pragma unroll
            for(int i = 0; i < segment_unroll; i++)
            {
                infos[i] = p_sorted_infos[segment_start + i];
                indice_weights[i] = p_sorted_indice_weights[segment_start + i];
            }
            p_sorted_infos += segment_unroll;
            p_sorted_indice_weights += segment_unroll;
        }
    }

    // LAST
    magic_div_u32_run_with_mod(batch_mdiv, infos[0], batch, table_index, bag_index);
    load_row_per_warp<grad_t, embedding_dim, int32_t>::run(
        &grad_data[0], bag_index * num_tables, p_output_grad + table_index * embedding_dim, lane_id);

    magic_div_u32_run_with_mod(batch_mdiv, infos[1], batch, table_index, bag_index);
    load_row_per_warp<grad_t, embedding_dim, int32_t>::run(
        &grad_data[dword_per_row], bag_index * num_tables, p_output_grad + table_index * embedding_dim, lane_id);

    if constexpr (!weighted) {
        #pragma unroll
        for(int j = 2; j < segment_unroll; j += 2)
        {
            accumulate_row_per_warp<grad_t, embedding_dim, cache_t, weighted>::run(
                &grad_acc[0], &grad_data[0], lane_id);
            magic_div_u32_run_with_mod(batch_mdiv, infos[j], batch, table_index, bag_index);
            load_row_per_warp<grad_t, embedding_dim, int32_t>::run(
                &grad_data[0], bag_index * num_tables, p_output_grad + table_index * embedding_dim, lane_id);

            accumulate_row_per_warp<grad_t, embedding_dim, cache_t, weighted>::run(
                &grad_acc[0], &grad_data[dword_per_row], lane_id);
            magic_div_u32_run_with_mod(batch_mdiv, infos[j + 1], batch, table_index, bag_index);
            load_row_per_warp<grad_t, embedding_dim, int32_t>::run(
                &grad_data[dword_per_row], bag_index * num_tables, p_output_grad + table_index * embedding_dim, lane_id);
        }

        accumulate_row_per_warp<grad_t, embedding_dim, cache_t, weighted>::run(
            &grad_acc[0], &grad_data[0], lane_id);
        accumulate_row_per_warp<grad_t, embedding_dim, cache_t, weighted>::run(
            &grad_acc[0], &grad_data[dword_per_row], lane_id);
    } else {
        #pragma unroll
        for(int j = 2; j < segment_unroll; j += 2)
        {
            accumulate_row_per_warp<grad_t, embedding_dim, cache_t, weighted>::run(
                &grad_acc[0], &grad_data[0], lane_id, indice_weights[j-2]);
            magic_div_u32_run_with_mod(batch_mdiv, infos[j], batch, table_index, bag_index);
            load_row_per_warp<grad_t, embedding_dim, int32_t>::run(
                &grad_data[0], bag_index * num_tables, p_output_grad + table_index * embedding_dim, lane_id);

            accumulate_row_per_warp<grad_t, embedding_dim, cache_t, weighted>::run(
                &grad_acc[0], &grad_data[dword_per_row], lane_id, indice_weights[j-1]);
            magic_div_u32_run_with_mod(batch_mdiv, infos[j + 1], batch, table_index, bag_index);
            load_row_per_warp<grad_t, embedding_dim, int32_t>::run(
                &grad_data[dword_per_row], bag_index * num_tables, p_output_grad + table_index * embedding_dim, lane_id);
        }

        accumulate_row_per_warp<grad_t, embedding_dim, cache_t, weighted>::run(
            &grad_acc[0], &grad_data[0], lane_id, indice_weights[segment_unroll-2]);
        accumulate_row_per_warp<grad_t, embedding_dim, cache_t, weighted>::run(
            &grad_acc[0], &grad_data[dword_per_row], lane_id, indice_weights[segment_unroll-1]);
    }

L_tail_grad_acc:
    if(segment_length & (segment_unroll - 1))
    {
        if constexpr (!weighted){
            // last, load one by one
            do
            {
                infos[0] = p_sorted_infos[segment_start];
                p_sorted_infos++;

                magic_div_u32_run_with_mod(batch_mdiv, infos[0], batch, table_index, bag_index);
                load_row_per_warp<grad_t, embedding_dim, int32_t>::run(
                    &grad_data[0], bag_index * num_tables, p_output_grad + table_index * embedding_dim, lane_id);
                accumulate_row_per_warp<grad_t, embedding_dim, cache_t, weighted>::run(
                    &grad_acc[0], &grad_data[0], lane_id);

                itr++;
            } while(itr < segment_length);
        } else {
            do
            {
                infos[0] = p_sorted_infos[segment_start];
                indice_weights[0] = p_sorted_indice_weights[segment_start];
                p_sorted_infos++;
                p_sorted_indice_weights++;

                magic_div_u32_run_with_mod(batch_mdiv, infos[0], batch, table_index, bag_index);
                load_row_per_warp<grad_t, embedding_dim, int32_t>::run(
                    &grad_data[0], bag_index * num_tables, p_output_grad + table_index * embedding_dim, lane_id);
                accumulate_row_per_warp<grad_t, embedding_dim, cache_t, weighted>::run(
                    &grad_acc[0], &grad_data[0], lane_id, indice_weights[0]);

                itr++;
            } while(itr < segment_length);
        }
    }

    // printf("[%d] segment_length:%d ==<< %f, emb_idx:%ld\n", wave_id, segment_length, grad_acc[0], emb_idx);

    // load the old emb weight data
    load_row_per_warp<emb_t, embedding_dim, int64_t>::run(
        &emb_data[0], emb_idx, p_emb_table, lane_id);
    optimizer_t optimizer(opt_karg);
    optimizer.template update<dword_per_row, segment_split>(grad_acc, emb_data, emb_idx);

    // store updated weight to grad ??
    store_row_per_warp<emb_t, embedding_dim, emb_t>::run(&emb_data[0], p_emb_table + emb_idx * embedding_dim, lane_id);
}


// PR23: for the reference, the template is
// hip_split_embedding{{ ndesc }}_backward_codegen_{{ optimizer }}_{{ wdesc }}{{ vdesc }}_kernel_warp_per_row_1(

template <
    typename emb_t,
    typename grad_t,
    typename cache_t,
    int32_t kFixedMaxVecsPerThread,
    int32_t kThreadGroupSize,
    bool kUseVecBlocking,
    int32_t embedding_dim>
//__global__ __launch_bounds__(kBackwardMaxThreads) void
__global__ void
// {%- if is_index_select %}
// batch_index_select_dim0_codegen_backward_kernel_warp_per_row(
// {%- else %}
hip_split_embedding{{ ndesc }}_backward_codegen_{{ optimizer }}_{{ wdesc }}{{ vdesc }}_kernel_warp_per_row_1(
// {%- endif %}
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
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> sorted_linear_indices_run,
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
    magic_div_u32_t batch_mdiv, // PR23 extra
    const int32_t batch, // PR23 extra
    const int32_t num_rows, // PR23 extra
    const int32_t num_tables, // PR23 extra
    // {%- if is_index_select %}
    // const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> grad_offsets,
    // const bool permute_output_dim_0_1
    // {%- else %}
    {{optimizer}}_kernel_arg_t opt_karg
    // {%- endif %}
) {
    // WIP: test the build system
#if 1
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
    // max_segment_length_per_warp = max_segment_length_per_warp;
    // indice_weights_sorted = indice_weights_sorted.packed_accessor32<float, 1, at::RestrictPtrTraits>().data();
    auto emb_dim = embedding_dim;
    constexpr int32_t segment_prefetch = 2; // always 2 in split_bwd.hip
    constexpr int32_t segment_unroll = 8;   // always 8 in split_bwd.hip
    constexpr int32_t segment_split = 0;    // always 0 in split_bwd.hip
    // avbokovoy: num_rows and num_tables should come from outside
    // num_rows = dev_weights.numel() / T / max_D;
    num_tables = T;
    split_tbe_backward_hip_kernel{{ndesc}}<
        {{optimizer}}_optimizer_t<cache_t, emb_t, embedding_dim, /* weight_decay_mode */ 0>,
        {{optimizer}}_kernel_arg_t,
        emb_t,
        cache_t, // cache_t
        grad_t,
        BLOCK_SIZE,
        embedding_dim,
        segment_prefetch,
        segment_unroll,
        segment_split,
        false>(p_output_grad,
               p_emb_table,
               p_hash_size_cumsum,
               p_sorted_linear_indices_run,
               p_sorted_linear_indices_cumulative_run_lengths,
               p_sorted_linear_indices_num_runs,
               // p_long_run_ids,  // unused
               // p_num_long_run_ids,  // unused
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
               num_tables,
               opt_karg);
#endif
}

{%- macro hip_template_instantiation(
      emb_type,
      grad_type,
      cache_type,
      kFixedMaxVecsPerThread,
      kThreadGroupSize,
      kUseVecBlocking,
      kEmbeddingDim
    )
%}
template __global__ __launch_bounds__(kBackwardMaxThreads) void
hip_split_embedding{{ ndesc }}_backward_codegen_{{ optimizer }}_{{ wdesc }}{{ vdesc }}_kernel_warp_per_row_1
< {{ emb_type }},
  {{ grad_type }},
  {{ cache_type }},
  {{ kFixedMaxVecsPerThread }},
  {{ kThreadGroupSize }},
  {{ kUseVecBlocking }},
  {{ kEmbeddingDim }}
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
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> sorted_linear_indices_run,
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
    magic_div_u32_t batch_mdiv, // PR23 extra
    const int32_t batch, // PR23 extra
    const int32_t num_rows, // PR23 extra
    const int32_t num_tables, // PR23 extra
    // {%- if is_index_select %}
    // const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> grad_offsets,
    // const bool permute_output_dim_0_1
    // {%- else %}
    {{optimizer}}_kernel_arg_t opt_karg
    // {%- endif %}
);
{%- endmacro %}

{%- macro hip_bulk_template_instantiations(kFixedMaxVecsPerThread, kThreadGroupSize, kUseVecBlocking) %}
    {%- for grad_type in ['float', 'at::Half', 'at::BFloat16'] %}
    {%- for emb_type in ['float', 'at::Half'] %}
    {%- for cache_type in ['float'] %}
    {%- for kEmbeddingDim in [64, 128, 192, 256] %}
        {{ hip_template_instantiation(
            emb_type,
            grad_type,
            cache_type,
            kFixedMaxVecsPerThread,
            kThreadGroupSize,
            kUseVecBlocking,
            kEmbeddingDim
          )
        }}
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
