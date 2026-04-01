/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef USE_ROCM
#include "fbgemm_gpu/utils/rocm/sparse_group_utils.h"

namespace fbgemm_gpu::rocm {
DLL_PUBLIC size_t get_sort_temp_storage_bytes(
    const size_t num_items,
    const c10::ScalarType scalar_type,
    const at::cuda::CUDAStream& stream) {
    size_t temp_storage_bytes = 0;
    AT_DISPATCH_INTEGRAL_TYPES(scalar_type, "get_sort_temp_storage_bytes", [&] {
        using index_t = scalar_t;
        AT_CUDA_CHECK(rocprim::radix_sort_pairs<sort_config>(
            nullptr,
            temp_storage_bytes,
            static_cast<index_t*>(nullptr),
            static_cast<index_t*>(nullptr),
            static_cast<int64_t*>(nullptr),
            static_cast<int64_t*>(nullptr),
            num_items,
            0,
            sizeof(index_t) * 8,
            stream,
            false));
    });
    return temp_storage_bytes;
}

DLL_PUBLIC size_t get_segmented_sort_temp_storage_bytes(
    const size_t num_items_per_segment,
    const int64_t num_groups,
    const c10::ScalarType scalar_type,
    const at::cuda::CUDAStream& stream) {
    size_t temp_storage_bytes = 0;
    const size_t total_items = num_items_per_segment * static_cast<size_t>(num_groups);
    AT_DISPATCH_INTEGRAL_TYPES(scalar_type, "get_segmented_sort_temp_storage_bytes", [&] {
        using index_t = scalar_t;
        // segmented_radix_sort_pairs requires segmented_radix_sort_config, not
        // radix_sort_config — use default config (radix sort, no merge fallback).
        AT_CUDA_CHECK(rocprim::segmented_radix_sort_pairs(
            nullptr,
            temp_storage_bytes,
            static_cast<const index_t*>(nullptr),
            static_cast<index_t*>(nullptr),
            static_cast<const int64_t*>(nullptr),
            static_cast<int64_t*>(nullptr),
            total_items,
            static_cast<unsigned int>(num_groups),
            static_cast<const int64_t*>(nullptr),
            static_cast<const int64_t*>(nullptr),
            0,
            sizeof(index_t) * 8,
            stream,
            false));
    });
    return temp_storage_bytes;
}

DLL_PUBLIC void sort_indices_segmented_rocprim(
    const at::Tensor& all_keys_in,
    at::Tensor& all_keys_out,
    const at::Tensor& all_values_in,
    at::Tensor& all_values_out,
    const at::Tensor& segment_offsets,
    const size_t num_items_per_segment,
    const int64_t num_groups,
    at::Tensor& temp_storage,
    const at::cuda::CUDAStream& stream) {
    if (num_items_per_segment == 0 || num_groups == 0) {
        return;
    }

    size_t temp_storage_bytes = static_cast<size_t>(temp_storage.numel());
    const size_t total_items = num_items_per_segment * static_cast<size_t>(num_groups);
    // segment_offsets is [0, N, 2N, ..., K*N]: begin[i] = ptr[i], end[i] = ptr[i+1]
    const auto* begin_offsets = segment_offsets.const_data_ptr<int64_t>();
    const auto* end_offsets = begin_offsets + 1;

    AT_DISPATCH_INTEGRAL_TYPES(all_keys_in.scalar_type(), "sort_indices_segmented_rocprim", [&] {
        using index_t = scalar_t;
        // segmented_radix_sort_pairs requires segmented_radix_sort_config —
        // radix_sort_config is not accepted here, so default config is used.
        // Only call this path when num_items_per_segment >= k_sort_merge_threshold
        // so there is no regression vs the per-group merge sort path.
        AT_CUDA_CHECK(rocprim::segmented_radix_sort_pairs(
            temp_storage.data_ptr(),
            temp_storage_bytes,
            all_keys_in.const_data_ptr<index_t>(),
            all_keys_out.data_ptr<index_t>(),
            all_values_in.const_data_ptr<int64_t>(),
            all_values_out.data_ptr<int64_t>(),
            total_items,
            static_cast<unsigned int>(num_groups),
            begin_offsets,
            end_offsets,
            0,
            sizeof(index_t) * 8,
            stream,
            false));
    });
}

DLL_PUBLIC void sort_indices_batch_rocprim(
    const int64_t* keys_in_ptrs,
    void* keys_out_base,
    int64_t* values_out_base,
    const int64_t* values_in,
    const size_t num_items,
    const int64_t num_groups,
    at::Tensor& temp_storage,
    const c10::ScalarType scalar_type,
    const at::cuda::CUDAStream& stream) {
    if (num_items == 0 || num_groups == 0) {
        return;
    }
    size_t temp_storage_bytes = static_cast<size_t>(temp_storage.numel());
    void* temp_ptr = temp_storage.data_ptr();
    AT_DISPATCH_INTEGRAL_TYPES(scalar_type, "sort_indices_batch_rocprim", [&] {
        using index_t = scalar_t;
        auto* keys_out = static_cast<index_t*>(keys_out_base);
        for (int64_t i = 0; i < num_groups; ++i) {
            AT_CUDA_CHECK(rocprim::radix_sort_pairs<sort_config>(
                temp_ptr,
                temp_storage_bytes,
                reinterpret_cast<const index_t*>(keys_in_ptrs[i]),
                keys_out + i * num_items,
                values_in,
                values_out_base + i * num_items,
                num_items,
                0,
                sizeof(index_t) * 8,
                stream,
                false));
        }
    });
}
} // namespace fbgemm::rocm

#endif
