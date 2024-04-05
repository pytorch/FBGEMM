/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// clang-format off
#include <ATen/ATen.h>
#include <ATen/TypeDefault.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/script.h>

#include "fbgemm_gpu/sparse_ops_utils.h"
#include "fbgemm_gpu/split_embeddings_utils.cuh"
#include "fbgemm_gpu/embedding_common.h"

using Tensor = at::Tensor;

using namespace fbgemm_gpu;

void split_embedding_{{ optimizer }}_update(
    Tensor& dev_weights,
    Tensor& uvm_weights,
    Tensor& lxu_cache_weights,
    const Tensor& grad_dev_weights,
    const Tensor& grad_dev_indices,
    const Tensor& weights_placements,
    const Tensor& weights_offsets,
    const int64_t max_D,
    const bool stochastic_rounding,
    {{ args.split_function_args | join(", ") }});

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
    m.def("split_embedding_{{ optimizer }}_update("
          "Tensor dev_weights, Tensor uvm_weights, "
          "Tensor lxu_cache_weights, "
          "Tensor grad_dev_weights, "
          "Tensor grad_dev_indices, "
          "Tensor weights_placement, "
          "Tensor weights_offsets, "
          "int max_D, "
          "bool stochastic_rounding, "
          "{{ args.split_function_schemas | join(", ") }}) -> ()");
    DISPATCH_TO_CUDA(
        "split_embedding_{{ optimizer }}_update", split_embedding_{{ optimizer }}_update);
}
// clang-format on
