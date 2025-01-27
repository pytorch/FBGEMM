/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>
#include <ATen/core/TensorAccessor.h>
#include <torch/torch.h>
#include <cstdint>

#include "fbgemm_gpu/embedding_common.h"

using Tensor = at::Tensor;

namespace fbgemm_gpu {

/*
 * In-training embedding pruning util functions
 */

void init_address_lookup_cuda(
    Tensor address_lookups,
    Tensor buffer_offsets,
    Tensor emb_sizes);

std::tuple<Tensor, Tensor, int64_t> prune_embedding_tables_cuda(
    int64_t iter,
    int64_t pruning_interval,
    Tensor address_lookups,
    Tensor row_utils,
    Tensor buffer_offsets,
    Tensor emb_sizes);

Tensor remap_indices_update_utils_cuda(
    const int64_t iter,
    const Tensor& buffer_idx,
    const Tensor& feature_lengths,
    const Tensor& feature_offsets,
    const Tensor& values,
    const Tensor& address_lookup,
    Tensor& row_util,
    const Tensor& buffer_offsets,
    const std::optional<std::vector<Tensor>>& full_values_list,
    const std::optional<bool>& update_util);

} // namespace fbgemm_gpu
