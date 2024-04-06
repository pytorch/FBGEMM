/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// clang-format off
{%- set wdesc = "weighted" if weighted else "unweighted" %}
{%- set ndesc = "_nobag" if nobag else "" %}
{%- set vdesc = "_vbe" if vbe else "" %}

#include "fbgemm_gpu/embedding_backward_template_helpers.cuh"
#include "fbgemm_gpu/fbgemm_tensor_accessor.h"
#include "fbgemm_gpu/split_embeddings_utils.cuh"
{%- if optimizer != "none" and not dense %}
#include "gen_embedding_optimizer_{{ optimizer }}_split_device_kernel.cuh"
{%- endif %}
#include "gen_embedding_backward_{{ kdesc }}_split_device_kernel.cuh"
#include "gen_embedding_backward_common_split_device_kernel.cuh"

using Tensor = at::Tensor;
using namespace fbgemm_gpu;

////////////////////////////////////////////////////////////////////////////////
// Kernel Template Definition
////////////////////////////////////////////////////////////////////////////////

{%- macro sync_grad_sums(kBlockDim) %}
    {%- set kWarpId = kBlockDim // 2 %}
    {%- set d_vec = "(vec * kThreadGroupSize + lane_id)" %}
    if (blockDim.y >= {{ kBlockDim }}) {
      if (warp_id < {{ kWarpId }}) {
        for (int32_t vec = 0; vec < max_vecs && {{ d_vec }} * VEC_WIDTH < D; ++vec) {
          const int32_t d_vec = {{ d_vec }};
          smem_grad_sum[d_vec] = vec4_acc(
              smem_grad_sum[d_vec],
              smem_grad_sum[d_vec +
                  {{ kWarpId }} * max_vecs * kThreadGroupSize]);
        }
      }
      __syncthreads();
  }
{%- endmacro %}

template <
    typename emb_t,
    typename grad_t,
    typename cache_t,
    int32_t kFixedMaxVecsPerThread,
    int32_t kThreadGroupSize,
    bool kUseVecBlocking>
__global__ __launch_bounds__(kMaxThreads) void
{%- if is_index_select %}
batch_index_select_dim0_codegen_backward_kernel_cta_per_row(
{%- else %}
split_embedding{{ ndesc }}_backward_codegen_{{ optimizer }}_{{ wdesc }}{{ vdesc }}_kernel_cta_per_row_1(
{%- endif %}
    const pta::PackedTensorAccessor64<grad_t, {{ "1" if is_index_select else "2" }}, at::RestrictPtrTraits> grad_output,
    {%- if optimizer != "none" %}
    pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> dev_weights,
    {%- if not dense %}
    pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> uvm_weights,
    pta::PackedTensorAccessor64<cache_t, 2, at::RestrictPtrTraits> lxu_cache_weights,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> weights_placements,
    {%- endif %}
    {%- endif %} // if optimizer != "none"
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> weights_offsets,
    {%- if not nobag or is_index_select %}
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> D_offsets,
    {%- else %}
    int64_t D,
    {%- endif %}
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> hash_size_cumsum,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> sorted_linear_indices_run,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> sorted_linear_indices_cumulative_run_lengths,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> long_run_ids,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> num_long_run_ids,
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
    {%- if not dense and optimizer != "none" %}
    bool stochastic_rounding,
    at::PhiloxCudaState stochastic_rounding_philox_args,
    {%- else %}
    pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> grad_dev_weights,
    {%- if optimizer == "none" %}
    const int32_t max_D,
    {%- endif %}
    {%- endif %} // if not dense and optimizer != "none"
    {%- if not nobag and vbe %}
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> B_offsets,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> row_output_offsets,
    {%- endif %}
    {%- if not nobag %}
    const int32_t info_B_num_bits,
    const uint32_t info_B_mask,
    {%- endif %}
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> long_run_id_to_really_long_run_ids,
    pta::PackedTensorAccessor32<at::acc_type<cache_t, true>, 2, at::RestrictPtrTraits> temp_grad_accum,
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> grad_accum_counter,
    const int32_t max_segment_length_per_cta,
    const bool use_deterministic_algorithms,
    const int32_t max_vecs_per_thread,
    {%- if is_index_select %}
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> grad_offsets,
    const bool permute_output_dim_0_1
    {%- else %}
    {{ args.split_kernel_args | replace_pta_namespace() | join(",\n    ") }}
    {%- endif %}
) {
#ifdef FBGEMM_USE_SUBWARP_SHUFFLE
  const unsigned int shfl_sync_mask =
        ((1L << kThreadGroupSize) - 1) <<
        (threadIdx.y % (kWarpSize / kThreadGroupSize) * kThreadGroupSize);
#else
  const unsigned int shfl_sync_mask = 0xffffffffu;
#endif
  constexpr int VEC_WIDTH = 4;
  constexpr auto kIsInt8 = std::is_same<emb_t, uint8_t>::value;
  int32_t T = weights_offsets.size(0);
  const int32_t num_long_runs = num_long_run_ids[0];
  const int32_t warp_id = threadIdx.y;
  const int32_t lane_id = threadIdx.x;

  // Copy value to max_vecs to make max_vecs_per_thread known at compile time
  // when kUseVecBlocking == false
  const int32_t max_vecs =
      kUseVecBlocking ? max_vecs_per_thread : kFixedMaxVecsPerThread;
  struct SharedMemory<Vec4TAcc<cache_t>> smem;
  auto* smem_grad_sum =
      smem.getPointer() + warp_id * max_vecs * kThreadGroupSize;

  for (int32_t long_run_id = blockIdx.x; long_run_id < num_long_runs; long_run_id += gridDim.x) {
        // The first thread block in the really long run has run_id in long_run_ids
        // and the rest have the negative of its offset (see find_long_segments kernel).
        int32_t cta_rank_on_current_run = 0;
        int32_t current_run_id = long_run_ids[long_run_id];
        if (current_run_id < 0) {
            cta_rank_on_current_run = -long_run_ids[long_run_id];
            current_run_id = long_run_ids[long_run_id - cta_rank_on_current_run];
        }
        const int32_t run_length =
            sorted_linear_indices_cumulative_run_lengths[current_run_id + 1] -
            sorted_linear_indices_cumulative_run_lengths[current_run_id];
        // This computation must agree with how we compute num_ctas_for_run in
        // find_long_segments kernel!
        const int32_t num_ctas_on_current_run =
            use_deterministic_algorithms ? 1 : div_round_up(run_length, max_segment_length_per_cta);


        const int64_t linear_index = sorted_linear_indices_run[current_run_id];
        const int32_t segment_start =
            sorted_linear_indices_cumulative_run_lengths[current_run_id] +
            cta_rank_on_current_run * max_segment_length_per_cta;
        const int32_t segment_end = std::min(
            use_deterministic_algorithms ? INT_MAX : segment_start + max_segment_length_per_cta,
            sorted_linear_indices_cumulative_run_lengths[current_run_id + 1]);
        const int32_t SL = segment_end - segment_start;

        // Note that with shared embedding tables we can have multiple tables
        // (i.e. different values of `t` sharing the same segment).
        {%- if not nobag %}
        const auto info_0 = reinterpret_cast<const uint32_t*>(&sorted_infos[0])[segment_start];
        const auto t_0 = info_0 >> info_B_num_bits;
        {%- else %}
        const auto info_0 = sorted_infos[segment_start];
        int32_t t_0 = info_0 % T;
        {%- endif %}

        int64_t hash_size = hash_size_cumsum[t_0];
        {%- if not nobag or is_index_select %}
        const int32_t D_start_t0 = D_offsets[t_0];
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

        const int32_t SL_per_warp = div_round_up(SL, blockDim.y);
        const int32_t sl_start = SL_per_warp * warp_id;
        const int32_t sl_end = min(SL_per_warp * (warp_id + 1), SL);

        // Accumulate gradients (compute grad_sum)
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
        // Do shared memory reduction only if we used multiple warps.
        if (SL > SL_per_warp) {
            __syncthreads();

            {{ sync_grad_sums(32) }}
            {{ sync_grad_sums(16) }}
            {{ sync_grad_sums(8) }}
            {{ sync_grad_sums(4) }}

            if (warp_id == 0) {
                {{
                   generate_optimized_grad_sum_loop_access(
                       """
                        {grad_vec} = vec4_acc(
                            smem_grad_sum[d_vec],
                            smem_grad_sum[d_vec + max_vecs * kThreadGroupSize]
                        );
                       """
                   )
                }}
            }
        }

        if (warp_id != 0) {
            continue;
        }

        if (num_ctas_on_current_run > 1) {
            int really_long_run_id = long_run_id_to_really_long_run_ids[long_run_id];
            Vec4TAcc<cache_t> *temp_grad_accum_ptr =
                reinterpret_cast<Vec4TAcc<cache_t>*>(&temp_grad_accum[really_long_run_id][0]);
            {{
                generate_optimized_grad_sum_loop_access(
                    """
                    gpuAtomicAdd(&temp_grad_accum_ptr[d_vec].acc.x, {grad_vec}.acc.x);
                    gpuAtomicAdd(&temp_grad_accum_ptr[d_vec].acc.y, {grad_vec}.acc.y);
                    gpuAtomicAdd(&temp_grad_accum_ptr[d_vec].acc.z, {grad_vec}.acc.z);
                    gpuAtomicAdd(&temp_grad_accum_ptr[d_vec].acc.w, {grad_vec}.acc.w);
                    """
                )
            }}

            int counter;
            if (threadIdx.x == 0) {
                __threadfence();
                counter = gpuAtomicAdd(&grad_accum_counter[really_long_run_id], -1);
            }
            counter = SHFL_SYNC(counter, 0);
            // Only the thread block accumulated the gradient last does the weight update.
            if (counter > 1) {
                continue;
            }
            CUDA_KERNEL_ASSERT(counter == 1 && "Invalid grad_accum_counter. Race condition?");
            {{
                generate_optimized_grad_sum_loop_access(
                    """
                    {grad_vec} = temp_grad_accum_ptr[d_vec];
                    """
                )
            }}
        }

        {%- if not dense and optimizer != "none" %}
        split_{{ optimizer }}_table_update_kernel<
          emb_t,
          cache_t,
          kFixedMaxVecsPerThread,
          kThreadGroupSize,
          VEC_WIDTH,
          kUseVecBlocking>(
              dev_weights,
              uvm_weights,
              lxu_cache_weights,
              weights_placements,
              weights_offsets,
              sorted_lxu_cache_locations,
              grad_sum,
              kUseVecBlocking ? smem_grad_sum : nullptr,
              kIsInt8 ? smem_grad_sum : nullptr,
              stochastic_rounding,
              stochastic_rounding_philox_args,
              current_run_id,
              use_uniq_cache_locations
                  ? (current_run_id - table_unique_indices_offsets[t_0])
                  : segment_start,
              D,
              t_0,
              idx,
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
        const int64_t weights_offset = current_run_id * max_D;
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
        {%- endif %}
    } // for each run
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
      kFixedMaxVecsPerThread,
      kThreadGroupSize,
      kUseVecBlocking
    )
%}
template __global__ __launch_bounds__(kMaxThreads) void
{%- if is_index_select %}
batch_index_select_dim0_codegen_backward_kernel_cta_per_row
{%- else %}
split_embedding{{ ndesc }}_backward_codegen_{{ optimizer }}_{{ wdesc }}{{ vdesc }}_kernel_cta_per_row_1
{%- endif %}
< {{ emb_type }},
  {{ grad_type }},
  {{ cache_type }},
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
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        weights_placements,
    {%- endif %}
    {%- endif %} // if optimizer != "none"
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> weights_offsets,
    {%- if not nobag or is_index_select %}
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> D_offsets,
    {%- else %}
    int64_t D,
    {%- endif %}
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> hash_size_cumsum,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> sorted_linear_indices_run,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> sorted_linear_indices_cumulative_run_lengths,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> long_run_ids,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> num_long_run_ids,
    {%- if not nobag %}
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> sorted_infos,
    {%- else %}
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> sorted_infos,
    {%- endif %}
    {%- if not dense %}
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        sorted_lxu_cache_locations,
    const bool use_uniq_cache_locations,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> table_unique_indices_offsets,
    {%- endif %}
    {%- if weighted %}
    const pta::PackedTensorAccessor32<at::acc_type<{{ cache_type }}, true>, 1, at::RestrictPtrTraits> sorted_indice_weights,
    {%- endif %}
    {%- if not dense and optimizer != "none" %}
    bool stochastic_rounding,
    at::PhiloxCudaState stochastic_rounding_philox_args,
    {%- else %}
    pta::PackedTensorAccessor64<{{ emb_type }}, 1, at::RestrictPtrTraits> grad_dev_weights,
    {%- if optimizer == "none" %}
    const int32_t max_D,
    {%- endif %}
    {%- endif %} // if not dense and optimizer != "none"
    {%- if not nobag and vbe %}
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> B_offsets,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> row_output_offsets,
    {%- endif %}
    {%- if not nobag %}
    const int32_t info_B_num_bits,
    const uint32_t info_B_mask,
    {%- endif %}
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> long_run_id_to_really_long_run_ids,
    pta::PackedTensorAccessor32<at::acc_type<{{ cache_type }}, true>, 2, at::RestrictPtrTraits> temp_grad_accum,
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> grad_accum_counter,
    const int32_t max_segment_length_per_cta,
    const bool use_deterministic_algorithms,
    const int32_t max_vecs_per_thread,
    {%- if is_index_select %}
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> grad_offsets,
    const bool permute_output_dim_0_1
    {%- else %}
    {{ args.split_kernel_args_no_defaults | replace_pta_namespace() | join(",\n    ") | replace("cache_t", cache_type) }}
    {%- endif %}
);
{%- endmacro %}

{%- macro bulk_template_instantiations(kFixedMaxVecsPerThread, kThreadGroupSize, kUseVecBlocking) %}
    {%- for grad_type in ['float', 'at::Half', 'at::BFloat16'] %}
    {%- for emb_type in ['float', 'at::Half'] %}
    {%- for cache_type in ['float', 'at::Half'] %}
        {{ template_instantiation(
            emb_type,
            grad_type,
            cache_type,
            kFixedMaxVecsPerThread,
            kThreadGroupSize,
            kUseVecBlocking)
         }}
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
  // clang-format on
