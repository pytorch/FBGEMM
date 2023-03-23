/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>

namespace fbgemm_gpu {

///@defgroup input-combine Combine Input Operators
///

///@ingroup input-combine
std::tuple<at::Tensor, at::Tensor, at::Tensor> tbe_input_combine_cpu(
    const std::vector<at::Tensor>& indices_list,
    const std::vector<at::Tensor>& offsets_list,
    const std::vector<at::Tensor>& per_sample_weights,
    const at::Tensor& include_last_offsets);

///@ingroup input-combine
std::tuple<at::Tensor, at::Tensor, at::Tensor>
padding_fused_tbe_input_combine_cpu(
    const std::vector<at::Tensor>& indices_list,
    const std::vector<at::Tensor>& offsets_list,
    const std::vector<at::Tensor>& per_sample_weights,
    const at::Tensor& include_last_offsets,
    int64_t batch_size);

std::tuple<at::Tensor, at::Tensor, at::Tensor>
tbe_input_combine_with_length_cuda(
    const uint64_t* const indices_addrs,
    const uint64_t* const lengths_addrs,
    const uint64_t* const per_sample_weights_addrs,
    const uint32_t* const indices_is_long,
    const uint32_t* const lengths_is_long,
    const uint64_t* const indices_offsets,
    const uint64_t* const lengths_offsets,
    const uint64_t num_lists,
    const uint64_t total_indices,
    const uint64_t total_lengths,
    const uint64_t max_list_size,
    const c10::DeviceIndex& device);

} // namespace fbgemm_gpu
