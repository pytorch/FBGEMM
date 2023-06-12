/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// clang-format off
{% set wdesc = "weighted" if weighted else "unweighted" %}
{% set vbe_desc = "_vbe" if vbe else "" %}
#include "fbgemm_gpu/embedding_backward_template_helpers.cuh"
#include "fbgemm_gpu/split_embeddings_utils.cuh"

using Tensor = at::Tensor;
using namespace fbgemm_gpu;

template <
    typename emb_t,
    typename grad_t,
    typename cache_t,
    size_t kMaxVecsPerThread,
    int32_t kThreadGroupSize>
__global__ __launch_bounds__(kMaxThreads) void
split_embedding{{ "_nobag" if nobag else "" }}_backward_codegen_{{ optimizer }}_{{ wdesc }}{{ vbe_desc }}_kernel_cta_per_row_1(
    const at::PackedTensorAccessor64<grad_t, 2, at::RestrictPtrTraits> grad_output,
    {% if optimizer != "none" %}
    at::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> dev_weights,
    {% if not dense %}
    at::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> uvm_weights,
    at::PackedTensorAccessor64<cache_t, 2, at::RestrictPtrTraits> lxu_cache_weights,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        weights_placements,
    {% endif %}
    {% endif %} // if optimizer != "none"
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> weights_offsets,
    {% if not nobag %}
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> D_offsets,
    {% else %}
    int64_t D,
    {% endif %}
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        hash_size_cumsum,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        sorted_linear_indices_run,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        sorted_linear_indices_cumulative_run_lengths,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        long_run_ids,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        num_long_run_ids,
    {% if not nobag %}
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> sorted_infos,
    {% else %}
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> sorted_infos,
    {% endif %}
    {% if not dense %}
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        sorted_lxu_cache_locations,
    {% endif %}
    {% if weighted %}
    const at::PackedTensorAccessor32<at::acc_type<cache_t, true>, 1, at::RestrictPtrTraits> sorted_indice_weights,
    {% endif %}
    {% if not dense and optimizer != "none" %}
    bool stochastic_rounding,
    at::PhiloxCudaState stochastic_rounding_philox_args,
    {% else %}
    at::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> grad_dev_weights,
    {% if optimizer == "none" %}
    const int32_t max_D,
    {% endif %}
    {% endif %} // if not dense and optimizer != "none"
    {% if vbe %}
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> B_offsets,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> output_offsets,
    {% endif %}
    {% if not nobag %}
    const int32_t info_B_num_bits,
    const uint32_t info_B_mask,
    {% endif %}
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> long_run_id_to_really_long_run_ids,
    at::PackedTensorAccessor32<at::acc_type<cache_t, true>, 2, at::RestrictPtrTraits> temp_grad_accum,
    at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> grad_accum_counter,
    const int32_t max_segment_length_per_cta,
    const bool use_deterministic_algorithms,
    {{ args.split_kernel_args | join(",\n    ") }});

template <
    typename emb_t,
    typename grad_t,
    typename cache_t,
    size_t kMaxVecsPerThread,
    int32_t kThreadGroupSize = kWarpSize>
__global__
__launch_bounds__(kBackwardMaxThreads)
void
split_embedding{{ "_nobag" if nobag else "" }}_backward_codegen_{{ optimizer }}_{{ wdesc }}{{ vbe_desc }}_kernel_warp_per_row_1(
    const at::PackedTensorAccessor64<grad_t, 2, at::RestrictPtrTraits>
        grad_output,
    {% if optimizer != "none" %}
    at::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> dev_weights,
    {% if not dense %}
    at::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> uvm_weights,
    at::PackedTensorAccessor64<cache_t, 2, at::RestrictPtrTraits> lxu_cache_weights,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        weights_placements,
    {% endif %}
    {% endif %}
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> weights_offsets,
    {% if not nobag %}
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> D_offsets,
    {% else %}
    int64_t D,
    {% endif %}
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        hash_size_cumsum,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        sorted_linear_indices_run,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        sorted_linear_indices_cumulative_run_lengths,
    {% if not nobag %}
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> sorted_infos,
    {% else %}
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> sorted_infos,
    {% endif %}
    {% if not dense %}
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        sorted_lxu_cache_locations,
    {% endif %}
    {% if weighted %}
    const at::PackedTensorAccessor32<at::acc_type<cache_t, true>, 1, at::RestrictPtrTraits> sorted_indice_weights,
    {% endif %}
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        sorted_linear_indices_num_runs,
    int32_t max_segment_length_per_warp,
    {% if not dense and optimizer != "none" %}
    bool stochastic_rounding,
    at::PhiloxCudaState stochastic_rounding_philox_args,
    {% else %}
    at::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> grad_dev_weights,
    {% if optimizer == "none" %}
    const int32_t max_D,
    {% endif %}
    {% endif %} // if not dense and optimizer != "none"
    {% if vbe %}
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> B_offsets,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> output_offsets,
    {% endif %}
    {% if not nobag %}
    const int32_t info_B_num_bits,
    const uint32_t info_B_mask,
    {% endif %}
    {{ args.split_kernel_args | join(", ") }});

Tensor split_embedding{{ "_nobag" if nobag else "" }}_backward_codegen_{{ optimizer }}_{{ wdesc }}_exact{{ vbe_desc }}_cuda(
    Tensor grad_output,
    Tensor dev_weights,
    {% if not dense %}
    Tensor uvm_weights,
    Tensor lxu_cache_weights,
    Tensor weights_placements,
    {% endif %}
    Tensor weights_offsets,
    {% if not nobag %}
    Tensor D_offsets,
    int64_t max_D,
    {% else %}
    int64_t D,
    {% endif %}
    Tensor hash_size_cumsum,
    int64_t total_hash_size_bits,
    Tensor indices,
    Tensor offsets,
    {% if not nobag %}
    int64_t pooling_mode,
    {% endif %}
    {% if weighted %}
    Tensor indice_weights,
    {% endif %}
    {% if not dense %}
    Tensor lxu_cache_locations,
    {% endif %}
    int64_t unused_,
    int64_t max_segment_length_per_warp,
    {% if not dense %}
    {% if optimizer != "none" %}
    bool stochastic_rounding,
    {% endif %}
    const int32_t info_B_num_bits,
    const uint32_t info_B_mask,
    {% endif %}
    {% if vbe %}
    const VBEMetadata& vbe_metadata,
    {% endif %}
    {% if optimizer != "none" %}
    {{ args.split_function_args | join(", ") }}
    {% else %}
    // This is acutally passed via args.split_function_args but explicitly list
    // it here for code readability
    int64_t total_hash_size,
    int64_t total_unique_indices
    {% endif %}
    ) {

   TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
        {% if optimizer != "none" %}
        dev_weights,
        {% endif %}
        {% if not dense %}
        uvm_weights,
        lxu_cache_weights,
        weights_placements,
        {% endif %}
        {% if vbe %}
        vbe_metadata.B_offsets,
        vbe_metadata.output_offsets,
        vbe_metadata.b_t_map,
        {% endif %}
        weights_offsets,
        {% if not nobag %}
        D_offsets,
        {% endif %}
        hash_size_cumsum,
        indices,
        offsets,
        {% if weighted %}
        indice_weights,
        {% endif %}
        {% if not dense %}
        lxu_cache_locations,
        {% endif %}
        grad_output);

    at::cuda::OptionalCUDAGuard device_guard;
    device_guard.set_index(dev_weights.get_device());

    {% if nobag %}
    auto max_D = D;
    {% endif %}
    TORCH_CHECK(max_D <= {{ max_embedding_dim }});

    {% if optimizer == "none" %}
    // grad_dev_weights has emb_t type
    auto grad_dev_weights = at::empty({total_unique_indices * max_D}, dev_weights.options());
    {% else %}
    // Set total_unique_indices to total num indices by default
    const auto total_unique_indices = indices.numel();
    {% if dense %}
    auto grad_dev_weights = zeros_like(dev_weights);
    {% endif %}
    {% endif %}

    // short-circuit if there are zero indices.
    if (indices.numel() == 0) {
        {% if dense %}
        return grad_dev_weights;
        {% elif optimizer == "none" %}
        return at::sparse_coo_tensor(
            at::empty({1, 0}, indices.options()),
            grad_dev_weights.reshape({0, max_D}),
            {total_hash_size, max_D},
            dev_weights.options().layout(at::kSparse)
        );
        {% else %}
        return Tensor();
        {% endif %}
    }

    {% if not nobag %}
    int32_t T = D_offsets.numel() - 1;
    {% else %}
    int32_t T = weights_offsets.numel();
    {% endif %}

    TORCH_CHECK(T > 0);
    // offsets = [B x T  + 1]
    const auto total_B = offsets.size(0) - 1;
    TORCH_CHECK(total_B > 0);
    auto BT_block_size = kMaxThreads / kWarpSize;
    TORCH_CHECK(BT_block_size * kWarpSize <= kMaxThreads);

    {% if vbe %}
    TORCH_CHECK(vbe_metadata.B_offsets.numel() == T + 1);
    TORCH_CHECK(vbe_metadata.output_offsets.numel() == total_B);
    TORCH_CHECK(vbe_metadata.b_t_map.numel() == total_B);
    {% endif %}

    {% if dense %}
    int32_t info_B_num_bits;
    uint32_t info_B_mask;
    std::tie(info_B_num_bits, info_B_mask) = adjust_info_B_num_bits(total_B / T, T);
    {% endif %}

    // V100: 96 KB; A100: 160 KB; H100: 228 KB.
    int max_shared_bytes = 0;
#ifndef __HIP_PLATFORM_HCC__
    cudaDeviceGetAttribute(&max_shared_bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev_weights.get_device());
#else
    // MI100 has 64 KB local memory (shared memory) per workgroup
    max_shared_bytes = 64 << 10;
#endif
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    int shared_kb = max_shared_bytes >> 10;
    // V100: 64 KB; A100: 96 KB; H100: 144 KB
#ifndef __HIP_PLATFORM_HCC__
    // Use 2/3 of the available GPU shared mem; leave rooms for L1$.
    int used_shared_kb = round_down(shared_kb * 2 / 3, 16);
    TORCH_CHECK(used_shared_kb > 0);
#else
    // MI100 has independent shared mem and L1
    int used_shared_kb = shared_kb;
#endif
    int used_shared_bytes = used_shared_kb << 10;

    Tensor linear_indices, linear_indices_sorted;
    Tensor infos_sorted;
    Tensor sorted_linear_indices_run, sorted_linear_indices_run_lengths,
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
            offsets,
            {{ "true" if nobag else "false" }},
            {{ "c10::optional<Tensor>(vbe_metadata.b_t_map)" if vbe else "c10::optional<Tensor>()" }},
            info_B_num_bits,
            info_B_mask,
            total_unique_indices);

    {% if not dense %}
    auto lxu_cache_locations_sorted = at::empty_like(lxu_cache_locations);
    if (lxu_cache_locations.size(0) > 0) {
        size_t temp_storage_bytes = 0;
        AT_CUDA_CHECK(radix_sort_pairs(
            nullptr,
            temp_storage_bytes,
            linear_indices.data_ptr<int64_t>(),
            linear_indices_sorted.data_ptr<int64_t>(),
            lxu_cache_locations.data_ptr<int32_t>(),
            lxu_cache_locations_sorted.data_ptr<int32_t>(),
            linear_indices.numel(),
            0,
            total_hash_size_bits,
            at::cuda::getCurrentCUDAStream(),
            false));
        auto temp_storage = at::empty(
            {static_cast<int64_t>(temp_storage_bytes)},
            indices.options().dtype(at::kByte));
        AT_CUDA_CHECK(radix_sort_pairs(
            temp_storage.data_ptr(),
            temp_storage_bytes,
            linear_indices.data_ptr<int64_t>(),
            linear_indices_sorted.data_ptr<int64_t>(),
            lxu_cache_locations.data_ptr<int32_t>(),
            lxu_cache_locations_sorted.data_ptr<int32_t>(),
            linear_indices.numel(),
            0,
            total_hash_size_bits,
            at::cuda::getCurrentCUDAStream(),
            false));
    }
    {% endif %}

    DISPATCH_EMB_GRAD_CACHE_TYPES(
        dev_weights.scalar_type(),
        grad_output.scalar_type(),
        {% if not dense %}
        lxu_cache_weights.scalar_type(),
        {% else %}
        dev_weights.scalar_type(),
        {% endif %}
            "split_embedding_backward_{{ optimizer }}_exact_kernel",
        [&] {
            {% if weighted %}
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
                at::cuda::getCurrentCUDAStream(),
                false));
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
                at::cuda::getCurrentCUDAStream(),
                false));
            }
            {% endif %}

            // early memory release
            linear_indices.reset();
            linear_indices_sorted.reset();

            {% if vbe %}
            grad_output = grad_output.reshape({1, -1});
            {% endif %}
            auto grad_output_accessor = grad_output.packed_accessor64<grad_t, 2, at::RestrictPtrTraits>();
            {% if not nobag %}
            Tensor grad_output_mean;
            if (static_cast<PoolingMode>(pooling_mode) == PoolingMode::MEAN) {
              grad_output_mean = at::empty_like(grad_output);
              {% if not dense or not vbe %}
              {{ "grad_mean_vbe_kernel" if vbe else "grad_mean_kernel" }}
                  <<<div_round_up(total_B, kMaxThreads / kWarpSize),
                     dim3(kWarpSize, kMaxThreads / kWarpSize),
                     0,
                     at::cuda::getCurrentCUDAStream()>>>
                     (
                         grad_output_mean.packed_accessor64<
                             grad_t, 2, at::RestrictPtrTraits>(),
                         grad_output_accessor,
                         D_offsets
                             .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                         offsets
                             .packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                         {% if vbe %}
                         vbe_metadata.output_offsets.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                         vbe_metadata.b_t_map.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                         info_B_num_bits,
                         info_B_mask
                         {% else %}
                         FixedDivisor(total_B / T)
                         {% endif %}
                    );
              C10_CUDA_KERNEL_LAUNCH_CHECK();
              {% endif %} // if not dense or not vbe
              grad_output_accessor = grad_output_mean.packed_accessor64<
                  grad_t, 2, at::RestrictPtrTraits>();
            }
            {% endif %}

            {% if not dense and optimizer != "none" %}
            at::PhiloxCudaState rng_engine_inputs;
            if (stochastic_rounding && !std::is_same<emb_t, float>::value) {
                auto gen = at::cuda::detail::getDefaultCUDAGenerator();
                std::lock_guard<std::mutex> lock(gen.mutex());
                rng_engine_inputs =
                    at::check_generator<at::CUDAGeneratorImpl>(gen)
                        ->philox_cuda_state(4);
            }
            {% endif %}
            // kMaxElemPerThread is # of elements handled by thread if we use a full warp for a row
            // We consider kMaxElemPerThread 1 and 2, and then a multiple of 4.
            {% for kMaxElemPerThread in range(1, max_embedding_dim // (items_per_warp // 4) + 1) %}
            {% if kMaxElemPerThread in [1, 2] or kMaxElemPerThread % 4 == 0 %}
            if (max_D <= {{ items_per_warp // 4 * kMaxElemPerThread }}) {
            // hipcc can't use max in constexpr
            constexpr int kMaxVecsPerThread = {{ kMaxElemPerThread }} / 4 >= 1 ? {{ kMaxElemPerThread }} / 4 : 1;
            // If max_D is small, use fewer number of threads than kWarpSize.
#ifdef FBGEMM_USE_SUBWARP_SHUFFLE
            constexpr int kThreadGroupSize = kWarpSize / std::max(4 / {{ kMaxElemPerThread }}, 1);
#else
            constexpr int kThreadGroupSize = kWarpSize;
#endif
            // Stay under used_shared_kb of shared memory (V100: 64 KB; A100: 96 KB; H100: 144 KB), BT_block_size must be a power of two.
            while (BT_block_size * sizeof(at::acc_type<cache_t, true>) * 4 * kWarpSize * kMaxVecsPerThread >= used_shared_bytes) {
                BT_block_size /= 2;
            }
            TORCH_CHECK(BT_block_size >= 1);
            if (std::is_same<emb_t, double>::value) {
                // Otherwise we see CUDA kernel launch failures despite the above checks.
                BT_block_size = 1;
            }

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

            split_embedding_backward_codegen_find_long_segments<<<
                div_round_up(total_unique_indices, kMaxThreads),
                kMaxThreads,
                0,
                at::cuda::getCurrentCUDAStream()
            >>>(
                sorted_linear_indices_num_runs.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                sorted_linear_indices_run_lengths.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                long_run_ids.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                num_long_run_ids.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                long_run_id_to_really_long_run_ids.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                num_really_long_run_ids.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                grad_accum_counter.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                max_segment_length_per_warp,
                max_segment_length_per_cta,
                use_deterministic_algorithms);
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            // A temp buffer to accumulate gradients with atomics.
            auto temp_grad_accum = at::zeros(
                {use_deterministic_algorithms ? 0 : grad_accum_counter.numel(), max_D},
                grad_output.options().dtype(std::is_same<cache_t, double>::value ? at::kDouble : at::kFloat));

            int32_t grid_size = std::min(
                div_round_up(total_unique_indices, kMaxThreads),
                get_max_thread_blocks_());

            // Check https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-7-x
            // "Compute capability 7.x devices allow a single thread block to
            // address the full capacity of shared memory: 96 KB on Volta,
            // 64 KB on Turing. Kernels relying on shared memory allocations
            // over 48 KB per block are architecture-specific, as such they
            // must use dynamic shared memory (rather than statically sized
            // arrays) and require an explicit opt-in using cudaFuncSetAttribute()".

#ifndef __HIP_PLATFORM_HCC__
            cudaFuncSetAttribute(
                split_embedding{{ "_nobag" if nobag else "" }}_backward_codegen_{{ optimizer }}_{{ wdesc }}{{ vbe_desc }}_kernel_cta_per_row_1<
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                used_shared_bytes); // V100: 64 KB; A100: 96 KB; H100: 144 KB
#endif
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            // dividing by kMaxThreads is a heuristic to avoid num of blocks far exceeding num_long_run_ids[0]
            split_embedding{{ "_nobag" if nobag else "" }}_backward_codegen_{{ optimizer }}_{{ wdesc }}{{ vbe_desc }}_kernel_cta_per_row_1<
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize>
                <<<grid_size,
                    dim3(kThreadGroupSize, BT_block_size),
                    BT_block_size * sizeof(at::acc_type<cache_t, true>) * 4 * kWarpSize *
                        kMaxVecsPerThread,
                    at::cuda::getCurrentCUDAStream()>>>(
                    grad_output_accessor,
                    {% if optimizer != "none" %}
                    {% if not dense %}
                    dev_weights.packed_accessor64<emb_t, 1, at::RestrictPtrTraits>(),
                    uvm_weights.packed_accessor64<emb_t, 1, at::RestrictPtrTraits>(),
                    lxu_cache_weights.packed_accessor64<cache_t, 2, at::RestrictPtrTraits>(),
                    weights_placements.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                    {% else %}
                    dev_weights.packed_accessor64<emb_t, 1, at::RestrictPtrTraits>(),
                    {% endif %}
                    {% endif %} // if optimizer != "none"
                    weights_offsets.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                    {% if not nobag %}
                    D_offsets.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                    {% else %}
                    D,
                    {% endif %}
                    hash_size_cumsum.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                    sorted_linear_indices_run
                        .packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                    sorted_linear_indices_cumulative_run_lengths
                        .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                    long_run_ids.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                    num_long_run_ids.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                    {% if not nobag %}
                    infos_sorted.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                    {% else %}
                    infos_sorted.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                    {% endif %}
                    {% if not dense %}
                    lxu_cache_locations_sorted.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                    {% endif %}
                    {% if weighted %}
                    indice_weights_sorted.packed_accessor32<at::acc_type<cache_t, true>, 1, at::RestrictPtrTraits>(),
                    {% endif %}
                    {% if not dense and optimizer != "none" %}
                    stochastic_rounding,
                    rng_engine_inputs,
                    {% else %}
                    grad_dev_weights.packed_accessor64<emb_t, 1, at::RestrictPtrTraits>(),
                    {% if optimizer == "none" %}
                    max_D,
                    {% endif %}
                    {% endif %} // if not dense and optimizer != "none"
                    {% if vbe %}
                    vbe_metadata.B_offsets.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                    vbe_metadata.output_offsets.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                    {% endif %}
                    {% if not nobag %}
                    info_B_num_bits,
                    info_B_mask,
                    {% endif %}
                    long_run_id_to_really_long_run_ids.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                    temp_grad_accum.packed_accessor32<at::acc_type<cache_t, true>, 2, at::RestrictPtrTraits>(),
                    grad_accum_counter.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                    max_segment_length_per_cta,
                    use_deterministic_algorithms,
                    {{ args.split_kernel_arg_constructors | join(", ") }});

            C10_CUDA_KERNEL_LAUNCH_CHECK();
            grid_size = std::min(
                div_round_up(total_unique_indices, kBackwardMaxThreads / kThreadGroupSize),
                get_max_thread_blocks_());

            // Shared memory is not needed for non uint8_t weights
            size_t shmem_bytes = 0;
            if (std::is_same<emb_t, uint8_t>::value) {
                shmem_bytes = BT_block_size * sizeof(
                    at::acc_type<cache_t, true>) * 4 * kWarpSize * kMaxVecsPerThread;
#ifndef __HIP_PLATFORM_HCC__
                cudaFuncSetAttribute(
                    split_embedding{{ "_nobag" if nobag else "" }}_backward_codegen_{{ optimizer }}_{{ wdesc }}{{ vbe_desc }}_kernel_warp_per_row_1<
                    emb_t,
                    grad_t,
                    cache_t,
                    kMaxVecsPerThread,
                    kThreadGroupSize>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    used_shared_bytes); // V100: 64 KB; A100: 96 KB; H100: 144 KB
#endif
            }

            C10_CUDA_KERNEL_LAUNCH_CHECK();
            split_embedding{{ "_nobag" if nobag else "" }}_backward_codegen_{{ optimizer }}_{{ wdesc }}{{ vbe_desc }}_kernel_warp_per_row_1<
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize>
                <<<grid_size,
                    dim3(kThreadGroupSize, kBackwardMaxThreads / kThreadGroupSize),
                    shmem_bytes,
                    at::cuda::getCurrentCUDAStream()>>>(
                    grad_output_accessor,
                    {% if optimizer != "none" %}
                    {% if not dense %}
                    dev_weights.packed_accessor64<emb_t, 1, at::RestrictPtrTraits>(),
                    uvm_weights.packed_accessor64<emb_t, 1, at::RestrictPtrTraits>(),
                    lxu_cache_weights.packed_accessor64<cache_t, 2, at::RestrictPtrTraits>(),
                    weights_placements.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                    {% else %}
                    dev_weights.packed_accessor64<emb_t, 1, at::RestrictPtrTraits>(),
                    {% endif %}
                    {% endif %}
                    weights_offsets.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                    {% if not nobag %}
                    D_offsets.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                    {% else %}
                    D,
                    {% endif %}
                    hash_size_cumsum.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                    sorted_linear_indices_run
                        .packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                    sorted_linear_indices_cumulative_run_lengths
                        .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                    {% if not nobag %}
                    infos_sorted.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                    {% else %}
                    infos_sorted.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                    {% endif %}
                    {% if not dense %}
                    lxu_cache_locations_sorted.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                    {% endif %}
                    {% if weighted %}
                    indice_weights_sorted.packed_accessor32<at::acc_type<cache_t, true>, 1, at::RestrictPtrTraits>(),
                    {% endif %}
                    sorted_linear_indices_num_runs
                        .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                    max_segment_length_per_warp,
                    {% if not dense and optimizer != "none" %}
                    stochastic_rounding,
                    rng_engine_inputs,
                    {% else %}
                    grad_dev_weights.packed_accessor64<emb_t, 1, at::RestrictPtrTraits>(),
                    {% if optimizer == "none" %}
                    max_D,
                    {% endif %}
                    {% endif %} // if not dense and optimizer != "none"
                    {% if vbe %}
                    vbe_metadata.B_offsets.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                    vbe_metadata.output_offsets.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                    {% endif %}
                    {% if not nobag %}
                    info_B_num_bits,
                    info_B_mask,
                    {% endif %}
                    {{ args.split_kernel_arg_constructors | join(", ") }});
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            return;
        }
        {% endif %}
        {% endfor %}
        });

    {% if dense %}
    return grad_dev_weights;
    {% elif optimizer == "none" %}
    return at::sparse_coo_tensor(
        sorted_linear_indices_run.unsqueeze(0),
        grad_dev_weights.reshape({total_unique_indices, max_D}),
        {total_hash_size, max_D},
        dev_weights.options().layout(at::kSparse));
    {% else %}
    return Tensor();
    {% endif %}
}

// clang-format on
