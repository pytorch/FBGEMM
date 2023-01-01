/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <cmath>
#include <functional>

#include <ATen/ATen.h>
#include <torch/library.h>

#include "fbgemm_gpu/embedding_inplace_update.h"

using Tensor = at::Tensor;

namespace fbgemm_gpu {

template <typename index_t>
void embedding_inplace_update_cpu_kernel(
    at::TensorAccessor<uint8_t, 1> dev_weights,
    at::TensorAccessor<uint8_t, 1> uvm_weights,
    const at::TensorAccessor<int32_t, 1>& weights_placements,
    const at::TensorAccessor<int64_t, 1>& weights_offsets,
    const at::TensorAccessor<uint8_t, 1>& weights_tys,
    const at::TensorAccessor<int32_t, 1>& D_offsets,
    const at::TensorAccessor<uint8_t, 1>& update_weights,
    const at::TensorAccessor<int32_t, 1>& update_table_idx,
    const at::TensorAccessor<index_t, 1>& update_row_idx,
    const at::TensorAccessor<int64_t, 1>& update_offsets,
    int64_t row_alignment) {
  for (int64_t n = 0; n < update_row_idx.size(0); n++) {
    int32_t t = update_table_idx[n];
    auto row_idx = update_row_idx[n];

    SparseType weight_ty = static_cast<SparseType>(weights_tys[t]);
    const int32_t D_start = D_offsets[t];
    const int32_t D_end = D_offsets[t + 1];
    const int32_t D = D_end - D_start;
    const int32_t D_bytes =
        nbit::padded_row_size_in_bytes(D, weight_ty, row_alignment);
    int64_t weight_offset = weights_offsets[t];

    uint8_t* __restrict__ weight_row;
    const auto placement = static_cast<PlacementType>(weights_placements[t]);
    if (placement == PlacementType::HOST) {
      weight_row =
          &dev_weights
              [weight_offset +
               static_cast<int64_t>(D_bytes) * static_cast<int64_t>(row_idx)];
    } else {
      weight_row =
          &uvm_weights
              [weight_offset +
               static_cast<int64_t>(D_bytes) * static_cast<int64_t>(row_idx)];
    }

    int64_t update_weight_offset = update_offsets[n];

    const uint8_t* __restrict__ update_weight_row =
        &update_weights[update_weight_offset];

    for (int32_t d = 0; d < D_bytes; d++) {
      weight_row[d] = update_weight_row[d];
    }
  }
}

void embedding_inplace_update_cpu(
    Tensor dev_weights,
    Tensor uvm_weights,
    Tensor weights_placements,
    Tensor weights_offsets,
    Tensor weights_tys,
    Tensor D_offsets,
    Tensor update_weights,
    Tensor update_table_idx,
    Tensor update_row_idx,
    Tensor update_offsets,
    const int64_t row_alignment,
    c10::optional<Tensor> lxu_cache_weights,
    c10::optional<Tensor> lxu_cache_locations) {
  TENSOR_ON_CPU(dev_weights);
  TENSOR_ON_CPU(uvm_weights);
  TENSOR_ON_CPU(weights_placements);
  TENSOR_ON_CPU(weights_offsets);
  TENSOR_ON_CPU(weights_tys);
  TENSOR_ON_CPU(D_offsets);

  TENSOR_ON_CPU(update_table_idx);
  TENSOR_ON_CPU(update_row_idx);
  TENSOR_ON_CPU(update_offsets);
  TENSOR_ON_CPU(update_weights);

  int64_t N = update_row_idx.numel();
  if (N == 0) {
    return;
  }

  AT_DISPATCH_INDEX_TYPES(
      update_row_idx.scalar_type(), "embedding_inplace_update_kernel", [&] {
        embedding_inplace_update_cpu_kernel(
            dev_weights.accessor<uint8_t, 1>(),
            uvm_weights.accessor<uint8_t, 1>(),
            weights_placements.accessor<int32_t, 1>(),
            weights_offsets.accessor<int64_t, 1>(),
            weights_tys.accessor<uint8_t, 1>(),
            D_offsets.accessor<int32_t, 1>(),
            update_weights.accessor<uint8_t, 1>(),
            update_table_idx.accessor<int32_t, 1>(),
            update_row_idx.accessor<index_t, 1>(),
            update_offsets.accessor<int64_t, 1>(),
            row_alignment);
      });
}

} // namespace fbgemm_gpu

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.def(
      "emb_inplace_update(Tensor(a!) dev_weights, Tensor(b!) uvm_weights, Tensor weights_placements, Tensor weights_offsets, Tensor weights_tys, Tensor D_offsets, Tensor update_weights, Tensor update_table_indices, Tensor update_row_indices, Tensor update_offsets, int row_alignment=1, Tensor(c!)? lxu_cache_weights=None, Tensor? lxu_cache_locations=None) -> ()");
}

TORCH_LIBRARY_IMPL(fbgemm, CPU, m) {
  DISPATCH_TO_CPU(
      "emb_inplace_update", fbgemm_gpu::embedding_inplace_update_cpu);
}
