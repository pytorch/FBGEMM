/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// clang-format off
#include "gen_embedding_optimizer_{{ optimizer }}_split_device_kernel.cuh"

// Template parameter kSubwarpDivisor: kThreadGroupSize is derived per-arch
// inside the kernel body as (kWarpSize / kSubwarpDivisor). This keeps the
// kernel's mangled name free of warpSize, so the host pass and every
// per-arch device pass agree on the symbol. The actual kThreadGroupSize
// used at runtime is set by the host launcher to (kWarpSizeHost() /
// kSubwarpDivisor), which matches the device-side value because kWarpSize
// (device) and kWarpSizeHost() (host) report the same warp size for the
// active arch. kSubwarpDivisor = 1 for the full-warp case.
template <
    typename emb_t,
    typename cache_t,
    {%- for ph_name in args.placeholder_tensor_names %}
    typename {{ ph_name + "_ph_t"}},
    {%- endfor %}
    size_t kMaxVecsPerThread,
    int32_t kSubwarpDivisor,
    int32_t VEC_WIDTH
>
__global__ __launch_bounds__(kMaxThreads)
void split_{{ optimizer }}_update_kernel(
    pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> dev_weights,
    pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> uvm_weights,
    pta::PackedTensorAccessor64<cache_t, 2, at::RestrictPtrTraits> lxu_cache_weights,
    const pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> grad_dev_weights,
    // grad_dev_indices is equivalent to sorted_linear_indices_run
    const pta::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> grad_dev_indices,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        weights_placements,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> weights_offsets,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        sorted_lxu_cache_locations,
    const int32_t max_D,
    bool stochastic_rounding,
    at::PhiloxCudaState stochastic_rounding_philox_args,
    {{ args.split_kernel_args | replace_pta_namespace() | join(",\n    ") }}
) {
    constexpr int32_t kThreadGroupSize = kWarpSize / kSubwarpDivisor;
    const auto run_id = blockIdx.x * blockDim.y + threadIdx.y;
    if (run_id >= grad_dev_indices.size(0)) {
      return;
    }

#ifdef FBGEMM_USE_SUBWARP_SHUFFLE
    const unsigned int shfl_sync_mask =
        ((1L << kThreadGroupSize) - 1) <<
        (threadIdx.y % (kWarpSize / kThreadGroupSize) * kThreadGroupSize);
#else
    const unsigned int shfl_sync_mask = 0xffffffffu;
#endif

    Vec4TAcc<cache_t> grad_sum[kMaxVecsPerThread];
    const auto D = max_D;

    // Load grad_dev_weights into grad_sum
#pragma unroll kMaxVecsPerThread
    for (int32_t i = 0;
        i < kMaxVecsPerThread && (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
        ++i) {
        int32_t d = (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH;
        grad_sum[i].load(&grad_dev_weights[run_id * D + d]);
    }

    // TODO: Enable smem grad sum
    constexpr bool kUseVecBlocking = false;

    split_{{ optimizer }}_table_update_kernel<
      emb_t,
      cache_t,
      {%- for ph_name in args.placeholder_tensor_names %}
      {{ ph_name + "_ph_t"}},
      {%- endfor %}
      kMaxVecsPerThread,
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
          nullptr, // smem_grad_sum (not yet supported)
          nullptr, // shared_weight_update_row (not yet supported INT8)
          stochastic_rounding,
          stochastic_rounding_philox_args,
          run_id,
          0, // segment_start (not used right now because lxu_cache is not
             // supported)
          D,
          0, // t
          grad_dev_indices[run_id], // idx
          // global weight decay is not supported in split optimizer
          {%- if has_global_weight_decay_support %}
          1.0, // global_weight_decay
          {%- endif %}
          shfl_sync_mask,
          kMaxVecsPerThread,
          {{ args.split_kernel_arg_names | join(", ") }});
}

{%- for use_subwarp in [True, False] %}

{{ "#ifdef FBGEMM_USE_SUBWARP_SHUFFLE" if use_subwarp else "#else" }}

{%- for emb_type in (['uint8_t', 'float', 'at::Half'] + (['at::Float8_e4m3fnuz'] if is_rocm else ['at::Float8_e4m3fn'])) %}
{%- for cache_type in ['float', 'at::Half'] %}
{%- for ph_type_combo in args.placeholder_type_combos %}

{#- Emit instantiations for the union of (kMaxVecsPerThread, kSubwarpDivisor)
    needed by every wave size in scope. Wave32 needs more brackets than wave64
    to cover the same max_D range (smaller kThreadGroupSize per arch), so the
    wave32 set is a superset of the wave64 set; we iterate it whenever wave32
    is enabled, and just iterate the wave64 set otherwise. -#}
{%- set _items_per_warp_eff = items_per_warp32 if has_wave32 else items_per_wave64 %}
{%- set tuples = [] %}
{%- for kMaxElemPerThread in range(1, legacy_max_embedding_dim // (_items_per_warp_eff // 4) + 1) %}
{%- if kMaxElemPerThread in [1, 2] or kMaxElemPerThread % 4 == 0 %}
    {%- set t0 = [ (kMaxElemPerThread // 4), 1 ] | max if not nobag else "NULL" %}
    {%- set t1 = [ 4 // kMaxElemPerThread, 1] | max %}
    {%- set temp = tuples.append((t0, t1 if use_subwarp else 1)) %}
{%- endif %}
{%- endfor %}

{%- for (kMaxVecsPerThread, kSubwarpDivisor) in tuples | unique %}
template __global__ __launch_bounds__(kMaxThreads)
void split_{{ optimizer }}_update_kernel
< {{ emb_type }},
  {{ cache_type }},
  {%- for ph_name in args.placeholder_tensor_names %}
  {{ ph_type_combo[ph_name] }},
  {%- endfor %}
  {{ kMaxVecsPerThread }},
  {{ kSubwarpDivisor }},
  4 // VEC_WIDTH
>(
    pta::PackedTensorAccessor64<{{ emb_type }}, 1, at::RestrictPtrTraits> dev_weights,
    pta::PackedTensorAccessor64<{{ emb_type }}, 1, at::RestrictPtrTraits> uvm_weights,
    pta::PackedTensorAccessor64<{{ cache_type }}, 2, at::RestrictPtrTraits> lxu_cache_weights,
    const pta::PackedTensorAccessor64<{{ emb_type }}, 1, at::RestrictPtrTraits> grad_dev_weights,
    // grad_dev_indices is equivalent to sorted_linear_indices_run
    const pta::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> grad_dev_indices,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        weights_placements,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> weights_offsets,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        sorted_lxu_cache_locations,
    const int32_t max_D,
    bool stochastic_rounding,
    at::PhiloxCudaState stochastic_rounding_philox_args,
    {{ args.split_kernel_args_no_defaults |
         replace_pta_namespace() |
         replace_placeholder_types(ph_type_combo) |
         join(",\n    ") |
         replace("cache_t", cache_type)
    }});

{%- endfor %} // for (kMaxVecsPerThread, kSubwarpDivisor)
{%- endfor %} // for ph_type_combo
{%- endfor %} // for cache_type
{%- endfor %} // for emb_type
{%- endfor %} // for use_subwarp

#endif // FBGEMM_USE_SUBWARP_SHUFFLE

// clang-format on
