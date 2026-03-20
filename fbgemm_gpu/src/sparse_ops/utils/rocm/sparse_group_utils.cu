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
DLL_PUBLIC std::tuple<at::Tensor, at::Tensor> sort_indices_with_rocprim(const at::Tensor& indices) {
    TORCH_CHECK(
        indices.dim() == 1,
        "sort_indices_with_rocprim expects a 1D tensor, got ",
        indices.dim());
    TORCH_CHECK(
        indices.is_cuda(),
        "sort_indices_with_rocprim expects a CUDA tensor for indices");

    CUDA_DEVICE_GUARD(indices);
    auto contiguous_indices = indices.contiguous();
    auto sorted_indices = at::empty_like(contiguous_indices);
    auto reverse_indices = at::empty(
        contiguous_indices.sizes(),
        contiguous_indices.options().dtype(at::kLong));
    auto original_positions = at::arange(
        contiguous_indices.numel(),
        contiguous_indices.options().dtype(at::kLong));

    const auto numel = contiguous_indices.numel();
    if (numel == 0) {
        return {sorted_indices, reverse_indices};
    }

    const auto num_items = static_cast<size_t>(numel);
    auto stream = at::cuda::getCurrentCUDAStream();

    const auto scalar_type = contiguous_indices.scalar_type();
    AT_DISPATCH_INTEGRAL_TYPES(
        scalar_type, "sort_indices_with_rocprim", [&] {
            using index_t = scalar_t;
            auto keys_in = contiguous_indices.data_ptr<index_t>();
            auto keys_out = sorted_indices.data_ptr<index_t>();
            auto values_in = original_positions.data_ptr<int64_t>();
            auto values_out = reverse_indices.data_ptr<int64_t>();

            size_t temp_storage_bytes = 0;
            // Selected empirically
            constexpr int k_merge_sort_threshold = 400'000;

            using sort_config = rocprim::radix_sort_config<
                rocprim::default_config,
                rocprim::default_config,
                rocprim::default_config,
                k_merge_sort_threshold>;
            AT_CUDA_CHECK(rocprim::radix_sort_pairs<sort_config>(
                nullptr,
                temp_storage_bytes,
                keys_in,
                keys_out,
                values_in,
                values_out,
                num_items,
                0,
                sizeof(index_t) * 8,
                stream,
                false));
            auto temp_storage = at::empty(
                {static_cast<int64_t>(temp_storage_bytes)},
                contiguous_indices.options().dtype(at::kByte));
            AT_CUDA_CHECK(rocprim::radix_sort_pairs<sort_config>(
                temp_storage.data_ptr(),
                temp_storage_bytes,
                keys_in,
                keys_out,
                values_in,
                values_out,
                num_items,
                0,
                sizeof(index_t) * 8,
                stream,
                false));
    });

    return {sorted_indices, reverse_indices};
}
} // namespace fbgemm::rocm

#endif
