/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>

namespace fbgemm_gpu {

std::tuple<at::Tensor, at::Tensor, at::Tensor> tbe_input_combine_cpu(
    const std::vector<at::Tensor>& indices_list,
    const std::vector<at::Tensor>& offsets_list,
    const std::vector<at::Tensor>& per_sample_weights,
    const at::Tensor& include_last_offsets);

std::tuple<at::Tensor, at::Tensor, at::Tensor>
padding_fused_tbe_input_combine_cpu(
    const std::vector<at::Tensor>& indices_list,
    const std::vector<at::Tensor>& offsets_list,
    const std::vector<at::Tensor>& per_sample_weights,
    const at::Tensor& include_last_offsets,
    int64_t batch_size);

void tbe_input_combine_with_length_cpu_out(
    at::Tensor& combined_indices,
    at::Tensor& combined_lengths,
    at::Tensor& combined_per_sample_weights,
    const std::vector<at::Tensor>& indices_list,
    const std::vector<at::Tensor>& lengths_list,
    const std::vector<at::Tensor>& per_sample_weights);

std::tuple<at::Tensor, at::Tensor, at::Tensor>
tbe_input_combine_with_length_cpu(
    const std::vector<at::Tensor>& indices_list,
    const std::vector<at::Tensor>& lengths_list,
    const std::vector<at::Tensor>& per_sample_weights);

} // namespace fbgemm_gpu
