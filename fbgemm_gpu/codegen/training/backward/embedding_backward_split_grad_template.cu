/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// clang-format off
#include "fbgemm_gpu/embedding_backward_template_helpers.cuh"
#include "fbgemm_gpu/utils/tensor_accessor_builder.h"
#include "fbgemm_gpu/split_embeddings_utils.cuh"
#include "fbgemm_gpu/config/feature_gates.h"
#include "fbgemm_gpu/utils/kernel_launcher.cuh"
#include "fbgemm_gpu/utils/ops_utils.h"
#include <ATen/cuda/CUDAContext.h>
#include <torch/library.h>

using Tensor = at::Tensor;

using namespace fbgemm_gpu;

{% if is_index_select %}
namespace index_select {
{% else %}
namespace embedding_ops {
{% endif %}


__global__ __launch_bounds__(kMaxThreads) void
split_embedding_backward_codegen_find_long_segments(
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        sorted_linear_indices_num_runs,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        sorted_linear_indices_run_lengths,
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        long_run_ids,
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        num_long_run_ids,
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        long_run_id_to_really_long_run_ids,
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        num_really_long_run_ids,
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
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

template <typename info_pta_t, typename info_t, bool nobag>
__global__ __launch_bounds__(kMaxThreads)
void split_embedding_backward_count_unique_indices_kernel(
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
) {
  const int32_t num_runs = sorted_linear_indices_num_runs[0];
  const auto T = weights_placements.size(0);
  for (auto run_id = blockIdx.x * blockDim.x + threadIdx.x;
       run_id < num_runs;
       run_id += blockDim.x * gridDim.x) {
    // Obtain the associated table id of the run id
    const auto segment_start = sorted_linear_indices_cumulative_run_lengths[run_id];
    const auto info = reinterpret_cast<const info_t*>(&sorted_infos[0])[segment_start];
    const auto t = nobag ? (info % T) : (info >> info_B_num_bits);

    int32_t t_next = -1;
    const auto unique_count_offset = run_id + 1;
    if (unique_count_offset < num_runs) {
      const auto segment_start_next = sorted_linear_indices_cumulative_run_lengths[unique_count_offset];
      const auto info_next = reinterpret_cast<const info_t*>(&sorted_infos[0])[segment_start_next];
      t_next = nobag ? (info_next % T) : (info_next >> info_B_num_bits);
    }

    if (t != t_next) {
      const auto placement = static_cast<PlacementType>(weights_placements[t]);
      if (placement != PlacementType::MANAGED_CACHING) {
        // Record num unique indices for PlacementType::DEVICE from unique_count_offset
        gpuAtomicAdd(&dev_or_uvm_unique_indices[t], unique_count_offset);
      }
      if (t_next != -1) {
        const auto placement_next = static_cast<PlacementType>(weights_placements[t_next]);
        if (placement_next != PlacementType::MANAGED_CACHING) {
          // Record num unique indices for PlacementType::DEVICE from unique_count_offset
          gpuAtomicAdd(&dev_or_uvm_unique_indices[t_next], -unique_count_offset);
        }
      }
    }
  }
}

{% for nobag in [True, False] %}
{% set info_pta_t = "int64_t" if nobag else "int32_t" %}
template __global__ __launch_bounds__(kMaxThreads)
void split_embedding_backward_count_unique_indices_kernel
<
  {{ info_pta_t }},
  {{ "int64_t" if nobag else "uint32_t" }},
  {{ "true" if nobag else "false" }}
> (
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        sorted_linear_indices_num_runs,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        sorted_linear_indices_cumulative_run_lengths,
    const pta::PackedTensorAccessor32<{{ info_pta_t }}, 1, at::RestrictPtrTraits>
        sorted_infos,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        weights_placements,
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        dev_or_uvm_unique_indices,
    const int info_B_num_bits
);
{% endfor %}

{% for vbe in [True, False] %}
{% set vdesc = "_vbe" if vbe else "" %}
template <typename grad_t, typename offset_t>
__global__ __launch_bounds__(kMaxThreads) void grad_mean{{ vdesc }}_kernel(
    pta::PackedTensorAccessor64<grad_t, 2, at::RestrictPtrTraits>
        grad_output_mean,
    const pta::PackedTensorAccessor64<grad_t, 2, at::RestrictPtrTraits>
        grad_output,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> D_offsets,
    const pta::PackedTensorAccessor32<offset_t, 1, at::RestrictPtrTraits> offsets,
    {% if vbe %}
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> row_grad_offsets,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> b_t_map,
    const int32_t info_B_num_bits,
    const uint32_t info_B_mask
    {% else %}
    FixedDivisor fd_B
    {% endif %}
) {
  int32_t T = D_offsets.size(0) - 1;
  [[maybe_unused]] int32_t b;
  int32_t t;
  const auto total_B = offsets.size(0) - 1;

  // On ROCm the launch caps the grid to stay within the HIP 2^32
  // threads-per-launch limit, so we grid-stride to cover the full workload.
  // On CUDA the grid is not capped and the loop body runs once per warp.
#ifdef USE_ROCM
  for (auto b_t = blockIdx.x * blockDim.y + threadIdx.y;
       b_t < total_B;
       b_t += blockDim.y * gridDim.x) {
#else
  auto b_t = blockIdx.x * blockDim.y + threadIdx.y;
  if (b_t >= total_B) {
    return;
  }
#endif

  {% if vbe %}
  const auto info = reinterpret_cast<const uint32_t*>(&b_t_map[b_t])[0];
  reinterpret_cast<uint32_t*>(&t)[0] = info >> info_B_num_bits;
  reinterpret_cast<uint32_t*>(&b)[0] = info & info_B_mask;
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
  const auto grad_offset = row_grad_offsets[b_t];
  const auto grad_outer_offset = 0;
  {% else %}
  const auto grad_offset = D_start;
  const auto grad_outer_offset = b;
  {% endif %}

  const grad_t* shifted_grad_output = &grad_output[grad_outer_offset][grad_offset];
  grad_t* shifted_grad_output_mean = &grad_output_mean[grad_outer_offset][grad_offset];

  if (L != 0) {
    for (auto d = threadIdx.x; d * 4 < D; d += blockDim.x) {
      Vec4T<grad_t> grad_out_vec(&shifted_grad_output[d * 4]);
      grad_out_vec.mul_(1.0 / L);
      grad_out_vec.store(&shifted_grad_output_mean[d * 4]);
    }
  } else {
    for (auto d = threadIdx.x; d * 4 < D; d += blockDim.x) {
      Vec4T<grad_t> grad_out_vec(&shifted_grad_output[d * 4]);
      grad_out_vec.store(&shifted_grad_output_mean[d * 4]);
    }
  }
#ifdef USE_ROCM
  } // for b_t (grid-stride loop, ROCm only)
#endif
}

////////////////////////////////////////////////////////////////////////////////
// Explicitly instantiate the template based on dispatch_emb_grad_cache_types
////////////////////////////////////////////////////////////////////////////////

{% for grad_type in ['at::Half', 'float', 'at::BFloat16'] %}
{% for offset_type in ['int32_t', 'int64_t'] %}
template __global__ __launch_bounds__(kMaxThreads)
void grad_mean{{ vdesc }}_kernel
<{{ grad_type }}, {{ offset_type }}> (
    pta::PackedTensorAccessor64<{{ grad_type }}, 2, at::RestrictPtrTraits>
        grad_output_mean,
    const pta::PackedTensorAccessor64<{{ grad_type }}, 2, at::RestrictPtrTraits>
        grad_output,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> D_offsets,
    const pta::PackedTensorAccessor32<{{ offset_type }}, 1, at::RestrictPtrTraits> offsets,
    {% if vbe %}
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> row_grad_offsets,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> b_t_map,
    const int32_t info_B_num_bits,
    const uint32_t info_B_mask
    {% else %}
    FixedDivisor fd_B
    {% endif %}
);
{% endfor %} // for offset_type in ['int32_t', 'int64_t']
{% endfor %} // for grad_type in ['at::Half', 'float']
{% endfor %} // for vbe in [True, False]

}

{% if not is_index_select %}
// ===========================================================================
// tbe_bwd_indices_preproc: combined index-preprocessing op for the TBE
// backward. Folded into this single, non-optimizer-templated TU so it shares
// the split_embedding_backward_codegen_find_long_segments __global__ defined
// above (embedding_ops namespace) -- no forward-decl, compiled once, NO
// separate build target/file. Wraps the two grad-independent steps
//   1) transpose_embedding_input  (linearize -> radix-sort -> RLE + cumsum)
//   2) find_long_segments         (segment partition)
// so they can be hoisted OFF the backward critical path. CUDA, common path
// (bagged, non-index-select). max_segment_length_per_cta +
// use_deterministic_algorithms are derived internally, mirroring the driver.
// The 12-tensor output order matches the driver's preproc_tensors[0..11]
// unpack contract in embedding_backward_split_template.cu.
// Design doc:
// docs.google.com/document/d/1Z8_1zI_4WSF-gsaHKVLY3wUNPyAZSYRJLDSbDZfRE2o
// ===========================================================================
namespace fbgemm_gpu {

std::tuple<
    Tensor, // linear_indices
    Tensor, // linear_indices_sorted
    Tensor, // sorted_linear_indices_run
    Tensor, // sorted_linear_indices_run_lengths
    Tensor, // sorted_linear_indices_num_runs
    Tensor, // sorted_linear_indices_cumulative_run_lengths
    Tensor, // infos_sorted
    Tensor, // long_run_ids
    Tensor, // num_long_run_ids
    Tensor, // long_run_id_to_really_long_run_ids
    Tensor, // num_really_long_run_ids
    Tensor> // grad_accum_counter
tbe_bwd_indices_preproc_cuda(
    const Tensor& hash_size_cumsum,
    const int64_t total_hash_size_bits,
    const Tensor& indices,
    const Tensor& offsets,
    const int64_t info_B_num_bits,
    const int64_t info_B_mask,
    const int64_t total_unique_indices,
    const std::optional<Tensor>& vbe_b_t_map,
    const bool nobag,
    const bool is_index_select) {
  CUDA_DEVICE_GUARD(indices);

  // ---- Part A: transpose_embedding_input ----------------------------------
  auto
      [linear_indices,
       linear_indices_sorted,
       infos_sorted,
       sorted_linear_indices_run,
       sorted_linear_indices_run_lengths,
       sorted_linear_indices_num_runs,
       sorted_linear_indices_cumulative_run_lengths] =
          transpose_embedding_input(
              hash_size_cumsum,
              total_hash_size_bits,
              indices,
              offsets,
              nobag,
              vbe_b_t_map,
              info_B_num_bits,
              info_B_mask,
              total_unique_indices,
              is_index_select);

  // ---- Part B: find_long_segments -----------------------------------------
  // Grid bound: when total_unique_indices is unknown at call time (-1, e.g.
  // hoisted into the forward before the run count is available), fall back to
  // indices.numel() -- a safe upper bound on the number of runs. The kernel
  // bounds its real work by the device-side run count, so extra blocks are
  // no-ops; this only over-launches, it does not affect correctness.
  const auto num_unique =
      total_unique_indices >= 0 ? total_unique_indices : indices.numel();

  auto long_run_ids =
      at::empty({indices.numel()}, sorted_linear_indices_run_lengths.options());
  auto num_long_run_ids = at::zeros({1}, indices.options().dtype(at::kInt));

  const bool use_deterministic_algorithms =
      at::globalContext().deterministicAlgorithms();

  // max_segment_length_per_warp is a fixed policy constant (warp/CTA routing
  // threshold), not a runtime input -- derived internally to mirror the driver.
#ifdef USE_ROCM
  constexpr int32_t max_segment_length_per_warp = 16384;
  const int max_segment_length_per_cta =
      use_deterministic_algorithms ? INT_MAX : 4096;
#else
  constexpr int32_t max_segment_length_per_warp = 32;
  const auto device_properties = at::cuda::getCurrentDeviceProperties();
  int default_segment_length = 1024;
  const bool b200_feature_enabled =
      (device_properties->major >= 10) &&
      fbgemm_gpu::config::is_feature_enabled(
          fbgemm_gpu::config::FeatureGateName::
              TBE_USE_TUNED_SEGMENT_LENGTHS_CTA_B200);
  if (b200_feature_enabled) {
    default_segment_length = 4096;
  }
  const int max_segment_length_per_cta =
      use_deterministic_algorithms ? INT_MAX : default_segment_length;
#endif

  Tensor long_run_id_to_really_long_run_ids;
  if (use_deterministic_algorithms) {
    long_run_id_to_really_long_run_ids =
        at::empty(0, sorted_linear_indices_run_lengths.options());
  } else {
    long_run_id_to_really_long_run_ids = at::empty(
        {indices.numel()}, sorted_linear_indices_run_lengths.options());
  }

  auto num_really_long_run_ids =
      at::zeros({1}, indices.options().dtype(at::kInt));
  auto grad_accum_counter = at::empty(
      use_deterministic_algorithms
          ? 0
          : (indices.numel() / max_segment_length_per_cta),
      indices.options().dtype(at::kInt));

  constexpr auto fls_ctx = "find_long_segments";
  FBGEMM_LAUNCH_KERNEL(
      embedding_ops::split_embedding_backward_codegen_find_long_segments,
      div_round_up(num_unique, kMaxThreads),
      kMaxThreads,
      0,
      at::cuda::getCurrentCUDAStream(),
      PTA_B(sorted_linear_indices_num_runs, int32_t, 1, 32).build(fls_ctx),
      PTA_B(sorted_linear_indices_run_lengths, int32_t, 1, 32).build(fls_ctx),
      PTA_B(long_run_ids, int32_t, 1, 32).build(fls_ctx),
      PTA_B(num_long_run_ids, int32_t, 1, 32).build(fls_ctx),
      PTA_B(long_run_id_to_really_long_run_ids, int32_t, 1, 32).build(fls_ctx),
      PTA_B(num_really_long_run_ids, int32_t, 1, 32).build(fls_ctx),
      PTA_B(grad_accum_counter, int32_t, 1, 32).build(fls_ctx),
      max_segment_length_per_warp,
      max_segment_length_per_cta,
      use_deterministic_algorithms);

  return {
      linear_indices,
      linear_indices_sorted,
      sorted_linear_indices_run,
      sorted_linear_indices_run_lengths,
      sorted_linear_indices_num_runs,
      sorted_linear_indices_cumulative_run_lengths,
      infos_sorted,
      long_run_ids,
      num_long_run_ids,
      long_run_id_to_really_long_run_ids,
      num_really_long_run_ids,
      grad_accum_counter};
}

} // namespace fbgemm_gpu

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.def(
      "tbe_bwd_indices_preproc("
      "    Tensor hash_size_cumsum, "
      "    int total_hash_size_bits, "
      "    Tensor indices, "
      "    Tensor offsets, "
      "    int info_B_num_bits=26, "
      "    int info_B_mask=0x2FFFFFF, "
      "    int total_unique_indices=-1, "
      "    Tensor? vbe_b_t_map=None, "
      "    bool nobag=False, "
      "    bool is_index_select=False"
      ") -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, "
      "Tensor, Tensor, Tensor, Tensor)");
  DISPATCH_TO_CUDA("tbe_bwd_indices_preproc", tbe_bwd_indices_preproc_cuda);
}
{% endif %}

// clang-format on
