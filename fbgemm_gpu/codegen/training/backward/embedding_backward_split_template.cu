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
{%- set mdesc = "ssd" if ssd else "split" %}

{%- macro get_desc_suffix(gwd) %}
{%- set gwddesc = "_gwd" if gwd else "" %}
{{- wdesc + vdesc + gwddesc }}
{%- endmacro %}

{%- set locs_or_addrs_tensor = "ssd_row_addrs" if ssd else "lxu_cache_locations" %}
{%- set locs_or_addrs_type = "int64_t" if ssd else "int32_t" %}

#include "fbgemm_gpu/embedding_backward_template_helpers.cuh"
#include "fbgemm_gpu/utils/tensor_accessor.h"
#include "fbgemm_gpu/sparse_ops.h"
#include "fbgemm_gpu/split_embeddings_utils.cuh"
#include "fbgemm_gpu/utils/ops_utils.h"

using Tensor = at::Tensor;
using namespace fbgemm_gpu;

////////////////////////////////////////////////////////////////////////////////
// External Function Declarations
////////////////////////////////////////////////////////////////////////////////

{%- set is_gwd_kernel = is_gwd and is_valid_gwd_config(
    dense,
    nobag,
    vbe,
    is_index_select,
    has_global_weight_decay_support,
    ssd) %}
{%- set desc_suffix = get_desc_suffix(is_gwd_kernel) %}
template <
    typename emb_t,
    typename grad_t,
    typename cache_t,
    {%- for ph_name in args.placeholder_tensor_names %}
    typename {{ ph_name + "_ph_t" }},
    {%- endfor %}
    int32_t kFixedMaxVecsPerThread,
    int32_t kThreadGroupSize,
    bool kUseVecBlocking>
__global__ __launch_bounds__(kMaxThreads) void
{%- if is_index_select %}
batch_index_select_dim0_codegen_backward_kernel_cta_per_row(
{%- else %}
{{ mdesc }}_embedding{{ ndesc }}_backward_codegen_{{ optimizer }}_{{ desc_suffix }}_kernel_cta_per_row_1(
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
    const pta::PackedTensorAccessor32<{{ locs_or_addrs_type }}, 1, at::RestrictPtrTraits> sorted_{{ locs_or_addrs_tensor }},
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
    {%- if vbe %}
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
);

template <
    typename emb_t,
    typename grad_t,
    typename cache_t,
    {%- for ph_name in args.placeholder_tensor_names %}
    typename {{ ph_name + "_ph_t" }},
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
    {%- if vbe %}
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
);

{%- if is_rocm and optimizer == "rowwise_adagrad" and not dense and not is_index_select
               and not is_gwd_kernel and not vbe and not ssd %}
#include "fbgemm_gpu/rocm/split_embeddings_common.h"
template <
    typename emb_t,
    typename grad_t,
    typename cache_t,
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
    {%- if is_index_select %}
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> grad_offsets,
    const bool permute_output_dim_0_1
    {%- else %}
    {{ args.split_kernel_args | replace_pta_namespace() | join(",\n    ") }}
    {%- endif %}
);
{%- endif %}
{% if is_index_select %}
namespace index_select {
{% else %}
namespace embedding_ops {
{% endif %}


__global__ __launch_bounds__(kMaxThreads) void
split_embedding_backward_codegen_find_long_segments(
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> sorted_linear_indices_num_runs,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> sorted_linear_indices_run_lengths,
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> long_run_ids,
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> num_long_run_ids,
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> long_run_id_to_really_long_run_ids,
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> num_really_long_run_ids,
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> grad_accum_counter,
    const int32_t max_segment_length_per_warp,
    const int32_t max_segment_length_per_cta,
    const bool use_deterministic_algorithms);


template <typename grad_t>
__global__ __launch_bounds__(kMaxThreads) void
grad_mean{{ vdesc }}_kernel(
    pta::PackedTensorAccessor64<grad_t, 2, at::RestrictPtrTraits> grad_output_mean,
    const pta::PackedTensorAccessor64<grad_t, 2, at::RestrictPtrTraits> grad_output,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> D_offsets,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> offsets,
    {%- if vbe %}
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> grad_offsets,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> b_t_map,
    const int32_t info_B_num_bits,
    const uint32_t info_B_mask
    {%- else %}
    FixedDivisor fd_B
    {%- endif %}
);

template <typename info_pta_t, typename info_t, bool nobag>
__global__ __launch_bounds__(kMaxThreads) void
split_embedding_backward_count_unique_indices_kernel(
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        sorted_linear_indices_num_runs,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        sorted_linear_indices_cumulative_run_lengths,
    const pta::PackedTensorAccessor32<info_pta_t, 1, at::RestrictPtrTraits>
        sorted_infos,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        weights_placements,
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        dev_or_uvm_unique_indices,
    const int info_B_num_bits
);

}

{% if is_index_select %}
using namespace index_select;
{% else %}
using namespace embedding_ops;
{% endif %}

////////////////////////////////////////////////////////////////////////////////
// Utility Macros
////////////////////////////////////////////////////////////////////////////////

{%- if is_experimental_optimizer %}

/*
  For the experimental optimizers, kThreadGroupSize, kFixedMaxVecsPerThread,
  and kUseVecBlocking are fixed to kWarpSize, {{ fixed_max_vecs_per_thread["backward"] }},
  and true.
*/
#define DISPATCH_OPTIMAL_KERNEL(MAX_D, ...)                                     \
  [&] {                                                                         \
    const int max_vecs_per_thread =                                             \
      (max_D + {{ items_per_warp }} - 1) / {{ items_per_warp }};                \
    constexpr int kThreadGroupSize = kWarpSize;                                 \
    constexpr int kFixedMaxVecsPerThread =                                      \
      {{ fixed_max_vecs_per_thread["backward"] }};                              \
    constexpr bool kUseVecBlocking = true;                                      \
    return __VA_ARGS__();                                                       \
  }()

{%- else %}

/*
  For the non-experimental optimizers, we determine the kernel template
  instantiation that is best optimized for MAX_D and invoke it.

  Please see dispatch_optimal_kernel in
  codegen/embedding_common_code_generator.py for more details
*/
#ifdef FBGEMM_USE_SUBWARP_SHUFFLE
#define DISPATCH_OPTIMAL_KERNEL(MAX_D, ...)                                   \
  [&] {                                                                       \
    {{
       dispatch_optimal_kernel(
           items_per_warp,
           fixed_max_vecs_per_thread["backward"],
           use_subwarp_shuffle=True)
    -}}
  }()

#else
#define DISPATCH_OPTIMAL_KERNEL(MAX_D, ...)                                   \
  [&] {                                                                       \
    {{
       dispatch_optimal_kernel(
           items_per_warp,
           fixed_max_vecs_per_thread["backward"],
           use_subwarp_shuffle=False)
    -}}
  }()

#endif

{%- endif %}

////////////////////////////////////////////////////////////////////////////////
// Auto generated placeholder tensor dispatch macros
////////////////////////////////////////////////////////////////////////////////

{%- if args.placeholder_tensor_names | length == 0 %}

#define DISPATCH_PLACEHOLDER_TYPES(NAME, ...) \
  return __VA_ARGS__();

{%- else %}

// TODO: create a macro for this
#define DISPATCH_PLACEHOLDER_TYPES({{ args.placeholder_tensor_names | to_upper_placeholder_types() | join(", ") }}, NAME, ...) \
  {%- for ph_name in args.placeholder_tensor_names %}
  at::ScalarType _{{ph_name}}_t =                                       \
    ::detail::scalar_type({{ ph_name.upper() + "_T" }});                \
  {%- endfor %}
  {%- for ph_combo in args.placeholder_type_combos %}
  if (                                                                  \
    {%- for ph_name, ph_ty in ph_combo.items() %}
    {{ ph_name.upper() + "_T" }} == {{ ph_ty.scalar_type }} &&          \
    {%- endfor %}
    true                                                                \
  ) {                                                                   \
    {%- for ph_name, ph_ty in ph_combo.items() %}
    using {{ ph_name + "_ph_t" }} = {{ ph_ty.primitive_type }};         \
    {%- endfor  %}
    return __VA_ARGS__();                                               \
  }                                                                     \
  {%- endfor %}
  else {                                                                \
    AT_ERROR(                                                           \
        #NAME, \
        " not implemented for ",                                        \
        {%- for ph_name in args.placeholder_tensor_names %}
        "{{ ph_name + "_ph_t" }} = '", toString(_{{ph_name}}_t) , "' ", \
        {%- endfor %}
        ""                                                              \
    );                                                                  \
  }

{%- endif %}


////////////////////////////////////////////////////////////////////////////////
// Helper functions
////////////////////////////////////////////////////////////////////////////////

template<typename func_t, typename bwd_func_t>
int32_t compute_num_groups_and_dynamic_smem_bytes(
    int32_t* num_groups,
    const func_t compute_smem_bytes_fn,
    const bwd_func_t bwd_kernel_fn,
    const int32_t used_shared_bytes) {
  int32_t smem_bytes = 0;
  // Stay under used_shared_kb of shared memory (V100: 64 KB;
  // A100: 96 KB; H100: 144 KB), num_groups must be a power
  // of two.
  while (
      (smem_bytes = compute_smem_bytes_fn(*num_groups))
      >= used_shared_bytes
  ) {
    *num_groups /= 2;
  }
  TORCH_CHECK_GE(*num_groups, 1);

  // Check https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-7-x
  // "Compute capability 7.x devices allow a single thread block to
  // address the full capacity of shared memory: 96 KB on Volta,
  // 64 KB on Turing. Kernels relying on shared memory allocations
  // over 48 KB per block are architecture-specific, as such they
  // must use dynamic shared memory (rather than statically sized
  // arrays) and require an explicit opt-in using cudaFuncSetAttribute()".
#ifndef USE_ROCM
  cudaFuncSetAttribute(
      bwd_kernel_fn,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      used_shared_bytes); // V100: 64 KB; A100: 96 KB; H100: 144 KB
  C10_CUDA_KERNEL_LAUNCH_CHECK();
#endif
  return smem_bytes;
}

////////////////////////////////////////////////////////////////////////////////
// Kernel Definition
////////////////////////////////////////////////////////////////////////////////

{#- /* Generate a separate cuda host to enable global weight decay using Jinja */ #}
{%- set is_gwd_kernel = is_gwd and is_valid_gwd_config(
    dense,
    nobag,
    vbe,
    is_index_select,
    has_global_weight_decay_support,
    ssd) %}
{%- set desc_suffix = get_desc_suffix(is_gwd_kernel) %}

{%- set embedding_cuda_op =
    "batch_index_select_dim0_codegen_backward_cuda"
    if is_index_select
    else "{}_embedding{}_backward_codegen_{}_{}_exact_cuda".format(
    mdesc,
    ndesc,
    optimizer,
    desc_suffix)
%}

Tensor {{ embedding_cuda_op }}(
    const Tensor& grad_output,
    const Tensor& dev_weights,
    {%- if not dense %}
    const Tensor& uvm_weights,
    const Tensor& lxu_cache_weights,
    const Tensor& weights_placements,
    {%- endif %}
    const Tensor& weights_offsets,
    {%- if not nobag or is_index_select %}
    const Tensor& D_offsets,
    const c10::SymInt max_D_,
    {%- else %}
    const c10::SymInt D_,
    {%- endif %}
    const Tensor& hash_size_cumsum,
    const int64_t total_hash_size_bits,
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
    {%- endif %}
    {%- if not is_index_select %}
    const int64_t unused_,
    {%- endif %}
    const int64_t max_segment_length_per_warp,
    {%- if not dense %}
    {%- if optimizer != "none" %}
    const bool stochastic_rounding,
    {%- endif %}
    const int64_t info_B_num_bits, // int32_t
    const int64_t info_B_mask_int64, // uint32_t
    {%- endif %}
    {%- if vbe %}
    const Tensor& B_offsets,
    const Tensor& vbe_row_output_offsets,
    const Tensor& vbe_b_t_map,
    {%- endif %}
    {%- if not is_index_select and not dense %}
    const bool use_uniq_cache_locations,
    const bool use_homogeneous_placements,
    {%- endif %}
    {%- if is_index_select %}
    const Tensor& grad_offsets,
    const Tensor& total_L_offsets,
    const int32_t fixed_L_per_warp,
    const int32_t num_warps_per_feature,
    const bool permute_output_dim_0_1
    {%- elif optimizer != "none" %}
    {%- if is_gwd_kernel %}
    {%- if "prev_iter_dev" not in args.split_function_arg_names %}
    const Tensor& prev_iter_dev,
    {%- endif %}
    {%- if "iter" not in args.split_function_arg_names %}
    const int64_t iter,
    {%- endif %}
    const double gwd_lower_bound,
    {%- endif %}
    {{ args.split_function_args_no_defaults | join(", ") }}
    {%- else %}
    // This is actually passed via args.split_function_args_no_defaults but explicitly list
    // it here for code readability
    int64_t total_hash_size,
    c10::SymInt total_unique_indices_
    {%- endif %}
) {
    {%- if not nobag or is_index_select %}
    const int64_t max_D = max_D_.guard_int(__FILE__, __LINE__);
    {%- else %}
    const int64_t D = D_.guard_int(__FILE__, __LINE__);
    {%- endif %}

    {%- if "learning_rate" in args.split_kernel_arg_names %}
    // convert `learning rate` to float since `learning rate` is float in kernels
    const float learning_rate = learning_rate_tensor.item<float>();
    {%- endif %}

    TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
        {%- if optimizer != "none" %}
        dev_weights,
        {%- endif %}
        {%- if not dense %}
        uvm_weights,
        lxu_cache_weights,
        weights_placements,
        {%- endif %}
        {%- if vbe %}
        B_offsets,
        vbe_row_output_offsets,
        vbe_b_t_map,
        {%- endif %}
        weights_offsets,
        {%- if not nobag or is_index_select %}
        D_offsets,
        {%- endif %}
        hash_size_cumsum,
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
        {%- if is_gwd_kernel %}
        prev_iter_dev,
        {%- endif %}
        grad_output);

    {%- if is_index_select %}
    if (!permute_output_dim_0_1) {
        TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
            grad_offsets,
            dev_weights
        );
    }
    {%- endif %}

    auto aligned_grad_output = aligned_grad_output_tensor_for_cuda_backwards(grad_output);

    CUDA_DEVICE_GUARD(dev_weights);

    {%- if nobag and not is_index_select %}
    auto max_D = D;
    {%- endif %}
    {%- if not is_index_select %}
    TORCH_CHECK_LE(max_D, {{ max_embedding_dim }});
    {%- endif %}

    {%- if optimizer == "none" %}
    // grad_dev_weights has emb_t type
    const auto total_unique_indices = total_unique_indices_.guard_int(__FILE__, __LINE__);
    auto grad_dev_weights = at::empty({total_unique_indices * max_D}, dev_weights.options());
    {%- else %}
    // Set total_unique_indices to total num indices by default
    const auto total_unique_indices = indices.numel();
    {%- if dense %}
    auto grad_dev_weights = zeros_like(dev_weights);
    {%- endif %}
    {%- endif %}

    // short-circuit if there are zero indices.
    if (indices.numel() == 0) {
        {%- if dense %}
        return grad_dev_weights;
        {%- elif optimizer == "none" %}
        return at::sparse_coo_tensor(
            at::empty({1, 0}, indices.options()),
            grad_dev_weights.reshape({0, max_D}),
            {total_hash_size, max_D},
            dev_weights.options().layout(at::kSparse)
        );
        {%- else %}
        return Tensor();
        {%- endif %}
    }

    {%- if not nobag %}
    int32_t T = D_offsets.numel() - 1;
    {%- else %}
    int32_t T = weights_offsets.numel();
    {%- endif %}

    TORCH_CHECK_GT(T, 0);
    // offsets = [B x T  + 1]
    {%- if is_index_select %}
    const auto total_B = num_warps_per_feature * T;
    {%- else %}
    const auto total_B = offsets.size(0) - 1;
    {%- endif %}
    TORCH_CHECK_GT(total_B, 0);

    {%- if vbe %}
    TORCH_CHECK_EQ(B_offsets.numel(), T + 1);
    TORCH_CHECK_EQ(vbe_row_output_offsets.numel(), total_B);
    TENSORS_HAVE_SAME_NUMEL(vbe_row_output_offsets, vbe_b_t_map);
    {%- endif %}

    {%- if dense %}
    int32_t info_B_num_bits;
    uint32_t info_B_mask;
    std::tie(info_B_num_bits, info_B_mask) = adjust_info_B_num_bits(total_B / T, T);
    {%- else %}
    // Cast info_B_mask from int64_t to uint32_t
    const uint32_t info_B_mask = info_B_mask_int64;
    {%- endif %}

    // V100: 96 KB; A100: 160 KB; H100: 228 KB.
    int max_shared_bytes = 0;
#ifndef USE_ROCM
    cudaDeviceGetAttribute(&max_shared_bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev_weights.get_device());
#else
    // MI100 has 64 KB local memory (shared memory) per workgroup
    max_shared_bytes = 64 << 10;
#endif
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    int shared_kb = max_shared_bytes >> 10;
    // V100: 64 KB; A100: 96 KB; H100: 144 KB
#ifndef USE_ROCM
    // Use 2/3 of the available GPU shared mem; leave rooms for L1$.
    int used_shared_kb = round_down(shared_kb * 2 / 3, 16);
    TORCH_CHECK_GT(used_shared_kb, 0);
#else
    // MI100 has independent shared mem and L1
    int used_shared_kb = shared_kb;
#endif
    const int used_shared_bytes = used_shared_kb << 10;

    Tensor linear_indices, linear_indices_sorted, infos_sorted,
        sorted_linear_indices_run, sorted_linear_indices_run_lengths,
        sorted_linear_indices_num_runs,
        sorted_linear_indices_cumulative_run_lengths;
    std::tie(
        linear_indices,
        linear_indices_sorted,
        infos_sorted,
        sorted_linear_indices_run,
        sorted_linear_indices_run_lengths,
        sorted_linear_indices_num_runs,
        sorted_linear_indices_cumulative_run_lengths) =
        transpose_embedding_input(
            hash_size_cumsum,
            total_hash_size_bits,
            indices,
            {{ "offsets" if not is_index_select else "Tensor()" }},
            {{ "true" if nobag else "false" }},
            {{ "std::optional<Tensor>(vbe_b_t_map)" if vbe else "std::optional<Tensor>()" }},
            info_B_num_bits,
            info_B_mask,
            total_unique_indices,
            {%- if is_index_select %}
            true, // is_index_select
            std::optional<Tensor>(total_L_offsets),
            fixed_L_per_warp,
            num_warps_per_feature
            {%- else %}
            false // is_index_select
            {%- endif %}
        );

    {%- if not dense %}
    Tensor {{ locs_or_addrs_tensor }}_sorted = {{ locs_or_addrs_tensor }};
    Tensor table_unique_indices_offsets;
    if ({{ locs_or_addrs_tensor }}.size(0) > 0) {
      if (use_uniq_cache_locations) {
        if (!use_homogeneous_placements) {
          // When use_uniq_cache_locations=true, {{ locs_or_addrs_tensor }} are unique
          // and sorted in an ascending order based on the linear cache indices.
          // Linear cache indices of tables that are not placed in cache are set
          // to a sentinel value (i.e., the sum of hash sizes of all embedding
          // tables).  Since the sentinel value is larger than the max linear
          // cache index value, the {{ locs_or_addrs_tensor }} can be sorted differently
          // than the sorted_linear_indices.
          //
          // For this reason, the run ids of sorted and unique
          // {{ locs_or_addrs_tensor }} can be different from those of the
          // sorted_linear_indices.  We need the following code to compute
          // table_unique_indices_offsets which contains the differences between
          // {{ locs_or_addrs_tensor }} run ids and sorted_linear_indices run ids.
          auto dev_or_uvm_unique_indices = at::zeros_like(weights_placements);

#ifdef FBGEMM_GPU_MEMCHECK
          const auto func_name = "split_embedding_backward_count_unique_indices_kernel";
#endif
          split_embedding_backward_count_unique_indices_kernel<
          {{ "int64_t" if nobag else "int32_t" }},
          {{ "int64_t" if nobag else "uint32_t" }},
          {{ "true" if nobag else "false" }}
          ><<<
            div_round_up(total_unique_indices, kMaxThreads),
            kMaxThreads,
            0,
            at::cuda::getCurrentCUDAStream()
              >>>(
                  MAKE_PTA_WITH_NAME(
                    func_name, sorted_linear_indices_num_runs, int32_t, 1, 32),
                  MAKE_PTA_WITH_NAME(
                    func_name, sorted_linear_indices_cumulative_run_lengths, int32_t, 1, 32),
                  MAKE_PTA_WITH_NAME(
                    func_name, infos_sorted, {{ "int64_t" if nobag else "int32_t" }}, 1, 32),
                  MAKE_PTA_WITH_NAME(
                    func_name, weights_placements, int32_t, 1, 32),
                  MAKE_PTA_WITH_NAME(
                    func_name, dev_or_uvm_unique_indices, int32_t, 1, 32),
                  info_B_num_bits
                 );
          C10_CUDA_KERNEL_LAUNCH_CHECK();

          table_unique_indices_offsets =
            fbgemm_gpu::asynchronous_complete_cumsum_gpu(dev_or_uvm_unique_indices).to(at::kInt);
        }
      }
      else {
        {{ locs_or_addrs_tensor }}_sorted = at::empty_like({{ locs_or_addrs_tensor }});
        size_t temp_storage_bytes = 0;
        AT_CUDA_CHECK(radix_sort_pairs(
              nullptr,
              temp_storage_bytes,
              linear_indices.data_ptr<int64_t>(),
              linear_indices_sorted.data_ptr<int64_t>(),
              {{ locs_or_addrs_tensor }}.data_ptr<{{ locs_or_addrs_type }}>(),
              {{ locs_or_addrs_tensor }}_sorted.data_ptr<{{ locs_or_addrs_type }}>(),
              linear_indices.numel(),
              0,
              total_hash_size_bits,
              at::cuda::getCurrentCUDAStream()));
        auto temp_storage = at::empty(
            {static_cast<int64_t>(temp_storage_bytes)},
            indices.options().dtype(at::kByte));
        AT_CUDA_CHECK(radix_sort_pairs(
              temp_storage.data_ptr(),
              temp_storage_bytes,
              linear_indices.data_ptr<int64_t>(),
              linear_indices_sorted.data_ptr<int64_t>(),
              {{ locs_or_addrs_tensor }}.data_ptr<{{ locs_or_addrs_type }}>(),
              {{ locs_or_addrs_tensor }}_sorted.data_ptr<{{ locs_or_addrs_type }}>(),
              linear_indices.numel(),
              0,
              total_hash_size_bits,
              at::cuda::getCurrentCUDAStream()));
      }
    }

    if ({{ locs_or_addrs_tensor }}.size(0) == 0 || !use_uniq_cache_locations || use_homogeneous_placements) {
        table_unique_indices_offsets = at::zeros_like(weights_placements);
    }
    {%- endif %}

    {%- if is_rocm and optimizer == "rowwise_adagrad" and not dense and not is_index_select 
                   and not is_gwd_kernel and not vbe and not ssd %}
    {%- set hip_kernel = "hip_split_embedding{}_backward_codegen_{}_{}{}_kernel_warp_per_row_1".format(
            ndesc,
            optimizer,
            wdesc,
            vdesc,
            )
     %}
    {%- endif %}

    DISPATCH_EMB_GRAD_CACHE_TYPES(
        dev_weights.scalar_type(),
        aligned_grad_output.scalar_type(),
        {%- if not dense %}
        lxu_cache_weights.scalar_type(),
        {%- else %}
        dev_weights.scalar_type(),
        {%- endif %}
            "{{ embedding_cuda_op }}",
        [&] {
            {%- if weighted %}
            auto indice_weights_sorted = at::empty_like(indice_weights);
            {
            size_t temp_storage_bytes = 0;
            AT_CUDA_CHECK(radix_sort_pairs(
                nullptr,
                temp_storage_bytes,
                linear_indices.data_ptr<int64_t>(),
                linear_indices_sorted.data_ptr<int64_t>(),
                indice_weights.data_ptr<at::acc_type<cache_t, true>>(),
                indice_weights_sorted.data_ptr<at::acc_type<cache_t, true>>(),
                linear_indices.numel(),
                0,
                total_hash_size_bits,
                at::cuda::getCurrentCUDAStream()));
            auto temp_storage = at::empty(
                {static_cast<int64_t>(temp_storage_bytes)},
                indices.options().dtype(at::kByte));
            AT_CUDA_CHECK(radix_sort_pairs(
                temp_storage.data_ptr(),
                temp_storage_bytes,
                linear_indices.data_ptr<int64_t>(),
                linear_indices_sorted.data_ptr<int64_t>(),
                indice_weights.data_ptr<at::acc_type<cache_t, true>>(),
                indice_weights_sorted.data_ptr<at::acc_type<cache_t, true>>(),
                linear_indices.numel(),
                0,
                total_hash_size_bits,
                at::cuda::getCurrentCUDAStream()));
            }
            {%- endif %}

            // early memory release
            linear_indices.reset();
            linear_indices_sorted.reset();

            {%- if vbe %}
            const auto grad_output_reshaped = aligned_grad_output.reshape({1, -1});
            {%- else %}
            const auto grad_output_reshaped = aligned_grad_output;
            {%- endif %}

            auto grad_output_accessor = MAKE_PTA_WITH_NAME(
                "{{ embedding_cuda_op }}.1",
                grad_output_reshaped,
                grad_t, {{ "1" if is_index_select else "2" }},
                64
            );

            {%- if not nobag %}
            Tensor grad_output_mean;
            if (static_cast<PoolingMode>(pooling_mode) == PoolingMode::MEAN) {
                grad_output_mean = at::empty_like(grad_output_reshaped);
                {%- if not dense or not vbe %}

#ifdef FBGEMM_GPU_MEMCHECK
                const auto func_name1 = "grad_mean{{ vdesc }}_kernel";
#endif

                grad_mean{{ vdesc }}_kernel<<<
                    div_round_up(total_B, kMaxThreads / kWarpSize),
                    dim3(kWarpSize, kMaxThreads / kWarpSize),
                    0,
                    at::cuda::getCurrentCUDAStream()>>>
                    (
                        MAKE_PTA_WITH_NAME(func_name1, grad_output_mean, grad_t, 2, 64),
                        MAKE_PTA_WITH_NAME(func_name1, grad_output_reshaped, grad_t, 2, 64),
                        MAKE_PTA_WITH_NAME(func_name1, D_offsets, int32_t, 1, 32),
                        MAKE_PTA_WITH_NAME(func_name1, offsets, int64_t, 1, 32),
                        {%- if vbe %}
                        MAKE_PTA_WITH_NAME(func_name1, vbe_row_output_offsets, int64_t, 1, 32),
                        MAKE_PTA_WITH_NAME(func_name1, vbe_b_t_map, int32_t, 1, 32),
                        info_B_num_bits,
                        info_B_mask
                        {%- else %}
                        FixedDivisor(total_B / T)
                        {%- endif %}
                    );

                C10_CUDA_KERNEL_LAUNCH_CHECK();
                {%- endif %} // if not dense or not vbe

                grad_output_accessor = MAKE_PTA_WITH_NAME("{{ embedding_cuda_op }}.2", grad_output_mean, grad_t, 2, 64);
            }
            {%- endif %}

            {%- if not dense and optimizer != "none" %}
            at::PhiloxCudaState rng_engine_inputs;
            if (stochastic_rounding && !std::is_same<emb_t, float>::value) {
                auto gen = at::cuda::detail::getDefaultCUDAGenerator();
                std::lock_guard<std::mutex> lock(gen.mutex());
                rng_engine_inputs =
                    at::check_generator<at::CUDAGeneratorImpl>(gen)
                        ->philox_cuda_state(4);
            }
            {%- endif %}

            DISPATCH_OPTIMAL_KERNEL(max_D, [&] {

                auto long_run_ids = at::empty({indices.numel()}, sorted_linear_indices_run_lengths.options());
                auto num_long_run_ids = at::zeros({1}, indices.options().dtype(at::kInt));

                const bool use_deterministic_algorithms = at::globalContext().deterministicAlgorithms();
                const int max_segment_length_per_cta = use_deterministic_algorithms ? INT_MAX : 1024;

                Tensor long_run_id_to_really_long_run_ids;
                if (use_deterministic_algorithms) {
                    long_run_id_to_really_long_run_ids =
                        at::empty(0, sorted_linear_indices_run_lengths.options());
                } else {
                    long_run_id_to_really_long_run_ids =
                        at::empty({indices.numel()}, sorted_linear_indices_run_lengths.options());
                }


                auto num_really_long_run_ids = at::zeros({1}, indices.options().dtype(at::kInt));
                auto grad_accum_counter = at::empty(
                    use_deterministic_algorithms ? 0 : (indices.numel() / max_segment_length_per_cta),
                    indices.options().dtype(at::kInt));

#ifdef FBGEMM_GPU_MEMCHECK
                const auto func_name2 = "split_embedding_backward_codegen_find_long_segments";
#endif

                split_embedding_backward_codegen_find_long_segments<<<
                    div_round_up(total_unique_indices, kMaxThreads),
                    kMaxThreads,
                    0,
                    at::cuda::getCurrentCUDAStream()
                >>>(
                    MAKE_PTA_WITH_NAME(func_name2, sorted_linear_indices_num_runs, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name2, sorted_linear_indices_run_lengths, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name2, long_run_ids, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name2, num_long_run_ids, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name2, long_run_id_to_really_long_run_ids, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name2, num_really_long_run_ids, int32_t, 1, 32),
                    MAKE_PTA_WITH_NAME(func_name2, grad_accum_counter, int32_t, 1, 32),
                    max_segment_length_per_warp,
                    max_segment_length_per_cta,
                    use_deterministic_algorithms);
                C10_CUDA_KERNEL_LAUNCH_CHECK();

                // A temp buffer to accumulate gradients with atomics.
                auto temp_grad_accum = at::zeros(
                    {use_deterministic_algorithms ? 0 : grad_accum_counter.numel(), max_D},
                    aligned_grad_output.options().dtype(std::is_same<cache_t, double>::value ? at::kDouble : at::kFloat));

                DISPATCH_PLACEHOLDER_TYPES(
                  {%- for ph_name in args.placeholder_tensor_names %}
                  {{ ph_name + "_dev" }}.scalar_type(),
                  {%- endfor %}
                  "{{ mdesc }}_embedding_backward_{{ optimizer }}_exact_placeholder_type_kernel",
                  [&] {

                    {%- set cta_kernel =
                        "batch_index_select_dim0_codegen_backward_kernel_cta_per_row"
                        if is_index_select else
                        "{}_embedding{}_backward_codegen_{}_{}_kernel_cta_per_row_1".format(
                            mdesc,
                            ndesc,
                            optimizer,
                            desc_suffix,
                        )
                    %}

                    const auto backward_cta_per_row_kernel =
                        {{ cta_kernel }}
                            <emb_t,
                             grad_t,
                             cache_t,
                             {%- for ph_name in args.placeholder_tensor_names %}
                             {{ ph_name + "_ph_t" }},
                             {%- endfor %}
                             kFixedMaxVecsPerThread,
                             kThreadGroupSize,
                             kUseVecBlocking>;

                    // Compute shared memory size for cta_per_row
                    constexpr auto kCacheAccBytes = sizeof(at::acc_type<cache_t, true>);
                    int32_t num_cta_per_row_groups = kMaxThreads / kWarpSize;
                    const int32_t cta_per_row_smem_bytes = compute_num_groups_and_dynamic_smem_bytes(
                        &num_cta_per_row_groups,
                        [&] (int num_groups) {
                          return num_groups * kCacheAccBytes * 4 * kWarpSize * max_vecs_per_thread;
                        },
                        backward_cta_per_row_kernel,
                        used_shared_bytes
                    );

                    const int32_t cta_per_row_grid_size = std::min(
                        div_round_up(total_unique_indices, kMaxThreads),
                        get_max_thread_blocks_());

#ifdef FBGEMM_GPU_MEMCHECK
                    const auto func_name3 = "{{ cta_kernel }}";
#endif
                    backward_cta_per_row_kernel
                        <<<cta_per_row_grid_size,
                            dim3(kThreadGroupSize, num_cta_per_row_groups),
                            cta_per_row_smem_bytes,
                            at::cuda::getCurrentCUDAStream()>>>(
                            grad_output_accessor,
                            {%- if optimizer != "none" %}
                            {%- if not dense %}
                            MAKE_PTA_WITH_NAME(func_name3, dev_weights, emb_t, 1, 64),
                            MAKE_PTA_WITH_NAME(func_name3, uvm_weights, emb_t, 1, 64),
                            MAKE_PTA_WITH_NAME(func_name3, lxu_cache_weights, cache_t, 2, 64),
                            MAKE_PTA_WITH_NAME(func_name3, weights_placements, int32_t, 1, 32),
                            {%- else %}
                            MAKE_PTA_WITH_NAME(func_name3, dev_weights, emb_t, 1, 64),
                            {%- endif %}
                            {%- endif %} // if optimizer != "none"
                            MAKE_PTA_WITH_NAME(func_name3, weights_offsets, int64_t, 1, 32),
                            {%- if not nobag or is_index_select %}
                            MAKE_PTA_WITH_NAME(func_name3, D_offsets, int32_t, 1, 32),
                            {%- else %}
                            D,
                            {%- endif %}
                            MAKE_PTA_WITH_NAME(func_name3, hash_size_cumsum, int64_t, 1, 32),
                            MAKE_PTA_WITH_NAME(func_name3, sorted_linear_indices_run, int64_t, 1, 32),
                            MAKE_PTA_WITH_NAME(func_name3, sorted_linear_indices_cumulative_run_lengths, int32_t, 1, 32),
                            MAKE_PTA_WITH_NAME(func_name3, long_run_ids, int32_t, 1, 32),
                            MAKE_PTA_WITH_NAME(func_name3, num_long_run_ids, int32_t, 1, 32),
                            {%- if not nobag %}
                            MAKE_PTA_WITH_NAME(func_name3, infos_sorted, int32_t, 1, 32),
                            {%- else %}
                            MAKE_PTA_WITH_NAME(func_name3, infos_sorted, int64_t, 1, 32),
                            {%- endif %}
                            {%- if not dense %}
                            MAKE_PTA_WITH_NAME(func_name3, {{ locs_or_addrs_tensor }}_sorted, {{ locs_or_addrs_type }}, 1, 32),
                            use_uniq_cache_locations,
                            MAKE_PTA_WITH_NAME(func_name3, table_unique_indices_offsets, int32_t, 1, 32),
                            {%- endif %}
                            {%- if weighted %}
                            MAKE_PTA_ACC_WITH_NAME(func_name3, indice_weights_sorted, cache_t, 1, 32),
                            {%- endif %}
                            {%- if not dense and optimizer != "none" %}
                            stochastic_rounding,
                            rng_engine_inputs,
                            {%- else %}
                            MAKE_PTA_WITH_NAME(func_name3, grad_dev_weights, emb_t, 1, 64),
                            {%- if optimizer == "none" %}
                            max_D,
                            {%- endif %}
                            {%- endif %} // if not dense and optimizer != "none"
                            {%- if vbe %}
                            MAKE_PTA_WITH_NAME(func_name3, B_offsets, int32_t, 1, 32),
                            MAKE_PTA_WITH_NAME(func_name3, vbe_row_output_offsets, int64_t, 1, 32),
                            {%- endif %}
                            {%- if not nobag %}
                            info_B_num_bits,
                            info_B_mask,
                            {%- endif %}
                            MAKE_PTA_WITH_NAME(func_name3, long_run_id_to_really_long_run_ids, int32_t, 1, 32),
                            MAKE_PTA_ACC_WITH_NAME(func_name3, temp_grad_accum, cache_t, 2, 32),
                            MAKE_PTA_WITH_NAME(func_name3, grad_accum_counter, int32_t, 1, 32),
                            max_segment_length_per_cta,
                            use_deterministic_algorithms,
                            max_vecs_per_thread,
                            {%- if is_gwd_kernel %}
                            MAKE_PTA_WITH_NAME(func_name3, prev_iter_dev, float, 1, 64),
                            {%- if "iter" not in args.split_function_arg_names %}
                            iter,
                            {%- endif %}
                            gwd_lower_bound,
                            {%- endif %}
                            {%- if is_index_select %}
                            grad_offsets.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                            permute_output_dim_0_1
                            {%- else %}
                            {{ args.split_kernel_arg_constructors | make_pta_acc_format("func_name3") | join(",\n                        ") }}
                            {%- endif %}
                    );

                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                    {%- set warp_kernel =
                        "batch_index_select_dim0_codegen_backward_kernel_warp_per_row"
                        if is_index_select else
                        "{}_embedding{}_backward_codegen_{}_{}_kernel_warp_per_row_1".format(
                            mdesc,
                            ndesc,
                            optimizer,
                            desc_suffix,
                        )
                    %}
                    auto backward_warp_per_row_kernel =
                        {{ warp_kernel }}
                            <emb_t,
                             grad_t,
                             cache_t,
                             {%- for ph_name in args.placeholder_tensor_names %}
                             {{ ph_name + "_ph_t" }},
                             {%- endfor %}
                             kFixedMaxVecsPerThread,
                             kThreadGroupSize,
                             kUseVecBlocking>;

                    // Compute shared memory size for warp_per_row
                    int32_t num_warp_per_row_groups = kBackwardMaxThreads / kThreadGroupSize;
                    int32_t warp_per_row_smem_bytes = 0;
                    auto blockSize = dim3(kThreadGroupSize, num_warp_per_row_groups);
                    if (kUseVecBlocking) {
                      warp_per_row_smem_bytes = compute_num_groups_and_dynamic_smem_bytes(
                          &num_warp_per_row_groups,
                          // Use max_D to compute shmem_bytes (for smem_grad_sum)
                          // instead of using kMaxVecsPerThread to minimize the
                          // shared memory allocation
                          [&] (int32_t num_groups) {
                              return max_D * num_groups * kCacheAccBytes;
                          },
                          backward_warp_per_row_kernel,
                          used_shared_bytes);
                    }

                    int32_t warp_per_row_grid_size = std::min(
                        div_round_up(total_unique_indices, num_warp_per_row_groups),
                        get_max_thread_blocks_());

#ifdef USE_ROCM
                    {%- if is_rocm and not is_index_select and optimizer == "rowwise_adagrad" and 
                        not dense and not is_gwd_kernel and not vbe and not ssd %}
                    if(dev_weights.scalar_type() == at::ScalarType::Half || dev_weights.scalar_type() == at::ScalarType::Float)
                    {
                        constexpr int segments_per_workgroup = 4;
                        {%- for kDimSize in [64, 128, 160, 192, 256] %}
                        {%- for kWeightDecayMode in [0, 1, 2] %}
                        if(max_D == {{ kDimSize }} && weight_decay_mode == {{ kWeightDecayMode }})
                        {
                            warp_per_row_grid_size = div_round_up(sorted_linear_indices_num_runs[0].item<int32_t>(), segments_per_workgroup);
                            blockSize = dim3(256);
                            warp_per_row_smem_bytes = 0;
                            
                            backward_warp_per_row_kernel =
                                {{ hip_kernel }}
                                    <emb_t,
                                    grad_t,
                                    cache_t,
                                    kFixedMaxVecsPerThread,
                                    kThreadGroupSize,
                                    kUseVecBlocking,
                                    {{ kDimSize }},
                                    {{ kWeightDecayMode }}>;
                        }
                        {%- endfor %}
                        {%- endfor %}
                    }
                    {%- endif %}
#endif


#ifdef FBGEMM_GPU_MEMCHECK
                    const auto func_name4 = "{{ warp_kernel }}";
#endif
                    backward_warp_per_row_kernel
                        <<<warp_per_row_grid_size,
                            blockSize,
                            warp_per_row_smem_bytes,
                            at::cuda::getCurrentCUDAStream()>>>(
                            grad_output_accessor,
                            {%- if optimizer != "none" %}
                            {%- if not dense %}
                            MAKE_PTA_WITH_NAME(func_name4, dev_weights, emb_t, 1, 64),
                            MAKE_PTA_WITH_NAME(func_name4, uvm_weights, emb_t, 1, 64),
                            MAKE_PTA_WITH_NAME(func_name4, lxu_cache_weights, cache_t, 2, 64),
                            MAKE_PTA_WITH_NAME(func_name4, weights_placements, int32_t, 1, 32),
                            {%- else %}
                            MAKE_PTA_WITH_NAME(func_name4, dev_weights, emb_t, 1, 64),
                            {%- endif %}
                            {%- endif %}
                            MAKE_PTA_WITH_NAME(func_name4, weights_offsets, int64_t, 1, 32),
                            {%- if not nobag or is_index_select %}
                            MAKE_PTA_WITH_NAME(func_name4, D_offsets, int32_t, 1, 32),
                            {%- else %}
                            D,
                            {%- endif %}
                            MAKE_PTA_WITH_NAME(func_name4, hash_size_cumsum, int64_t, 1, 32),
                            MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_run, int64_t, 1, 32),
                            MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_cumulative_run_lengths, int32_t, 1, 32),
                            {%- if not nobag %}
                            MAKE_PTA_WITH_NAME(func_name4, infos_sorted, int32_t, 1, 32),
                            {%- else %}
                            MAKE_PTA_WITH_NAME(func_name4, infos_sorted, int64_t, 1, 32),
                            {%- endif %}
                            {%- if not dense %}
                            MAKE_PTA_WITH_NAME(func_name4, {{ locs_or_addrs_tensor }}_sorted, {{ locs_or_addrs_type }}, 1, 32),
                            use_uniq_cache_locations,
                            MAKE_PTA_WITH_NAME(func_name4, table_unique_indices_offsets, int32_t, 1, 32),
                            {%- endif %}
                            {%- if weighted %}
                            MAKE_PTA_ACC_WITH_NAME(func_name4, indice_weights_sorted, cache_t, 1, 32),
                            {%- endif %}
                            MAKE_PTA_WITH_NAME(func_name4, sorted_linear_indices_num_runs, int32_t, 1, 32),
                            max_segment_length_per_warp,
                            {%- if not dense and optimizer != "none" %}
                            stochastic_rounding,
                            rng_engine_inputs,
                            {%- else %}
                            MAKE_PTA_WITH_NAME(func_name4, grad_dev_weights, emb_t, 1, 64),
                            {%- endif %} // if not dense and optimizer != "none"
                            {%- if vbe %}
                            MAKE_PTA_WITH_NAME(func_name4, B_offsets, int32_t, 1, 32),
                            MAKE_PTA_WITH_NAME(func_name4, vbe_row_output_offsets, int64_t, 1, 32),
                            {%- endif %}
                            {%- if not nobag %}
                            info_B_num_bits,
                            info_B_mask,
                            {%- endif %}
                            max_D,
                            max_vecs_per_thread,
                            {%- if is_gwd_kernel %}
                            MAKE_PTA_WITH_NAME(func_name4, prev_iter_dev, float, 1, 64),
                            {%- if "iter" not in args.split_function_arg_names %}
                            iter,
                            {%- endif %}
                            gwd_lower_bound,
                            {%- endif %}
                            {%- if is_index_select %}
                            grad_offsets.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                            permute_output_dim_0_1
                            {%- else %}
                            {{ args.split_kernel_arg_constructors | make_pta_acc_format("func_name4") | join(",\n                        ") }}
                            {%- endif %}
                    );
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                }); // DISPATCH_PLACEHOLDER_TYPES
                return;

            }); // DISPATCH_OPTIMAL_KERNEL
        }); // DISPATCH_EMB_GRAD_CACHE_TYPES

    {%- if dense %}
    return grad_dev_weights;

    {%- elif optimizer == "none" %}
    return at::sparse_coo_tensor(
        sorted_linear_indices_run.unsqueeze(0),
        grad_dev_weights.reshape({total_unique_indices, max_D}),
        {total_hash_size, max_D},
        dev_weights.options().layout(at::kSparse));

    {%- else %}
    return Tensor();
    {%- endif %}
}

////////////////////////////////////////////////////////////////////////////////
// Op registrations
////////////////////////////////////////////////////////////////////////////////
{%- if not is_index_select %}
TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
    {%- set embedding_codegen_backward_op =
        "{}_embedding{}_backward_codegen_{}_{}_exact_cuda".format(
            mdesc, ndesc, optimizer, desc_suffix
        )
    %}
    m.def("{{ embedding_codegen_backward_op }}("
          "    Tensor grad_output, "
          "    Tensor(a!) dev_weights, "
          {%- if not dense %}
          "    Tensor(b!) uvm_weights, "
          "    Tensor lxu_cache_weights, "
          "    Tensor weights_placements, "
          {%- endif %}
          "    Tensor weights_offsets, "
          {%- if not nobag or is_index_select %}
          "    Tensor D_offsets, "
          "    SymInt max_D, "
          {%- else %}
          "    SymInt D, "
          {%- endif %}
          "    Tensor hash_size_cumsum, "
          "    int total_hash_size_bits, "
          "    Tensor indices, "
          {%- if not is_index_select %}
          "    Tensor offsets, "
          {%- endif %}
          {%- if not nobag %}
          "    int pooling_mode, "
          {%- endif %}
          {%- if weighted %}
          "    Tensor indice_weights, "
          {%- endif %}
          {%- if not dense %}
          "    Tensor {{ locs_or_addrs_tensor }}, "
          {%- endif %}
          {%- if not is_index_select %}
          "    int unused_, "
          {%- endif %}
          "    int max_segment_length_per_warp, "
          {%- if not dense %}
          {%- if optimizer != "none" %}
          "    bool stochastic_rounding, "
          {%- endif %}
          "    int info_B_num_bits, "
          "    int info_B_mask_int64, "
          {%- endif %}
          {%- if vbe %}
          "    Tensor B_offsets, "
          "    Tensor vbe_row_output_offsets, "
          "    Tensor vbe_b_t_map, "
          {%- endif %}
          {%- if not is_index_select and not dense %}
          "    bool use_uniq_cache_locations, "
          "    bool use_homogeneous_placements, "
          {%- endif %}
          {%- if is_gwd_kernel %}
          {%- if "prev_iter_dev" not in args.split_function_arg_names %}
          "    Tensor prev_iter_dev, "
          {%- endif %}
          {%- if "iter" not in args.split_function_arg_names %}
          "    int iter, "
          {%- endif %}
          "    float gwd_lower_bound, "
          {%- endif %}
          "    {{ args.split_function_schemas | join(", ") }}"
          ") -> Tensor");
    DISPATCH_TO_CUDA(
        "{{ embedding_codegen_backward_op }}",
        {{ embedding_codegen_backward_op }}
    );
}
{%- endif %} {#-/* if not is_index_select */#}
// clang-format on
