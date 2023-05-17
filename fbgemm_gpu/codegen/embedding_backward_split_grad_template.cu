/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// clang-format off
#include "fbgemm_gpu/embedding_backward_template_helpers.cuh"
#include "fbgemm_gpu/split_embeddings_utils.cuh"

using Tensor = at::Tensor;
using namespace fbgemm_gpu;

__global__ __launch_bounds__(kMaxThreads) void
split_embedding_backward_codegen_find_long_segments(
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        sorted_linear_indices_num_runs,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        sorted_linear_indices_run_lengths,
    at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        long_run_ids,
    at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        num_long_run_ids,
    at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        long_run_id_to_really_long_run_ids,
    at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        num_really_long_run_ids,
    at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        grad_accum_counter,
    const int32_t max_segment_length_per_warp,
    const int32_t max_segment_length_per_cta,
    const bool use_deterministic_algorithms) {
  const int32_t num_runs = sorted_linear_indices_num_runs[0];
  for (auto run_id = blockIdx.x * blockDim.x + threadIdx.x; run_id < num_runs; run_id += blockDim.x * gridDim.x) {
    if (sorted_linear_indices_run_lengths[run_id] >= max_segment_length_per_warp) {
        // A segment with length > max_segment_length_per_cta is handled by more than 1 thread block.
        const int num_ctas_for_run =
            use_deterministic_algorithms ? 1 : div_round_up(sorted_linear_indices_run_lengths[run_id], max_segment_length_per_cta);
        const auto long_run_idx = gpuAtomicAdd(&num_long_run_ids[0], num_ctas_for_run);
        // The first thread block in the really long run gets run_id in long_run_ids
        // and the rest get the negative of its offset.
        long_run_ids[long_run_idx] = run_id;
        for (int i = 1; i < num_ctas_for_run; ++i) {
            long_run_ids[long_run_idx + i] = -i;
        }
        if (num_ctas_for_run > 1) {
            const auto really_long_run_idx = gpuAtomicAdd(&num_really_long_run_ids[0], 1);
            grad_accum_counter[really_long_run_idx] = num_ctas_for_run;
            for (int i = 0; i < num_ctas_for_run; ++i) {
                long_run_id_to_really_long_run_ids[long_run_idx + i] = really_long_run_idx;
            }
        }
    }
  }
}

{% for vbe in [True, False] %}
{% set vbe_desc = "_vbe" if vbe else "" %}
template <typename grad_t>
__global__ __launch_bounds__(kMaxThreads) void grad_mean{{ vbe_desc }}_kernel(
    at::PackedTensorAccessor64<grad_t, 2, at::RestrictPtrTraits>
        grad_output_mean,
    const at::PackedTensorAccessor64<grad_t, 2, at::RestrictPtrTraits>
        grad_output,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> D_offsets,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> offsets,
    {% if vbe %}
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> grad_offsets,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> b_t_map,
    const int32_t info_B_num_bits,
    const uint32_t info_B_mask
    {% else %}
    FixedDivisor fd_B
    {% endif %}
) {
  int32_t T = D_offsets.size(0) - 1;
  int32_t b_t = blockIdx.x * blockDim.y + threadIdx.y;
  int32_t b;
  int32_t t;
  const auto total_B = offsets.size(0) - 1;

  if (b_t >= total_B) {
    return;
  }

  {% if vbe %}
  const auto info = reinterpret_cast<const uint32_t*>(&b_t_map[b_t])[0];
  reinterpret_cast<uint32_t*>(&t)[0] = info >> info_B_num_bits;
  reinterpret_cast<uint32_t*>(&b)[0] = info | info_B_mask;
  {% else %}
  fd_B.DivMod(b_t, &t, &b);
  {% endif %}

  int32_t D_start = D_offsets[t];
  int32_t D_end = D_offsets[t + 1];
  int32_t D = D_end - D_start;
  int64_t indices_start = offsets[b_t];
  int64_t indices_end = offsets[b_t + 1];
  int32_t L = indices_end - indices_start;

  {% if vbe %}
  const auto grad_offset = grad_offsets[b_t];
  const auto grad_outer_offset = 0;
  {% else %}
  const auto grad_offset = D_start;
  const auto grad_outer_offset = b;
  {% endif %}

  const grad_t* shifted_grad_output = &grad_output[grad_outer_offset][grad_offset];
  grad_t* shifted_grad_output_mean = &grad_output_mean[grad_outer_offset][grad_offset];

  if (L != 0) {
    for (int32_t d = threadIdx.x; d * 4 < D; d += blockDim.x) {
      Vec4T<grad_t> grad_out_vec(&shifted_grad_output[d * 4]);
      grad_out_vec.mul_(1.0 / L);
      grad_out_vec.store(&shifted_grad_output_mean[d * 4]);
    }
  } else {
    for (int32_t d = threadIdx.x; d * 4 < D; d += blockDim.x) {
      Vec4T<grad_t> grad_out_vec(&shifted_grad_output[d * 4]);
      grad_out_vec.store(&shifted_grad_output_mean[d * 4]);
    }
  }
}

// Explicitly instantiate the template based on DISPATCH_EMB_GRAD_CACHE_TYPES
{% for grad_type in ['at::Half', 'float'] %}
template __global__ __launch_bounds__(kMaxThreads)
void grad_mean{{ vbe_desc }}_kernel
<{{ grad_type }}> (
    at::PackedTensorAccessor64<{{ grad_type }}, 2, at::RestrictPtrTraits>
        grad_output_mean,
    const at::PackedTensorAccessor64<{{ grad_type }}, 2, at::RestrictPtrTraits>
        grad_output,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> D_offsets,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> offsets,
    {% if vbe %}
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> grad_offsets,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> b_t_map,
    const int32_t info_B_num_bits,
    const uint32_t info_B_mask
    {% else %}
    FixedDivisor fd_B
    {% endif %}
);
{% endfor %} // for grad_type in ['at::Half', 'float']
{% endfor %} // for vbe in [True, False]

// clang-format on
