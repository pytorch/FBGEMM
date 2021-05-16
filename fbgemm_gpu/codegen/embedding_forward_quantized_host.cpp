/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <ATen/ATen.h>
#include <ATen/TypeDefault.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/script.h>

using namespace at;

Tensor int4_split_embedding_codegen_forward_unweighted_cuda(
    Tensor dev_weights,
    Tensor weights_offsets,
    Tensor D_offsets,
    int64_t total_D,
    int64_t max_D,
    Tensor indices,
    Tensor offsets,
    int64_t pooling_mode,
    int64_t unused);

Tensor int4_split_embedding_codegen_forward_weighted_cuda(
    Tensor dev_weights,
    Tensor weights_offsets,
    Tensor D_offsets,
    int64_t total_D,
    int64_t max_D,
    Tensor indices,
    Tensor offsets,
    int64_t pooling_mode,
    Tensor indice_weights,
    int64_t unused);

Tensor int4_split_embedding_codegen_lookup_function(
    Tensor dev_weights,
    Tensor weights_offsets,
    Tensor D_offsets,
    int64_t total_D,
    int64_t max_D,
    Tensor indices,
    Tensor offsets,
    int64_t pooling_mode,
    c10::optional<Tensor> indice_weights) {
  if (!indice_weights) {
    return int4_split_embedding_codegen_forward_unweighted_cuda(
        dev_weights,
        weights_offsets,
        D_offsets,
        total_D,
        max_D,
        indices,
        offsets,
        pooling_mode,
        0);
  }
  return int4_split_embedding_codegen_forward_weighted_cuda(
      dev_weights,
      weights_offsets,
      D_offsets,
      total_D,
      max_D,
      indices,
      offsets,
      pooling_mode,
      *indice_weights,
      0);
}

Tensor pruned_hashmap_lookup_unweighted_cuda(
    Tensor indices,
    Tensor offsets,
    Tensor hash_table,
    int64_t T);

TORCH_LIBRARY_FRAGMENT(fb, m) {
  m.def(
      "int4_split_embedding_codegen_lookup_function(Tensor dev_weights, Tensor weights_offsets, Tensor D_offsets, int total_D, int max_D, Tensor indices, Tensor offsets, int pooling_mode, Tensor? indice_weights) -> Tensor");
  m.impl(
      "int4_split_embedding_codegen_lookup_function",
      torch::dispatch(
          c10::DispatchKey::CUDA,
          TORCH_FN(int4_split_embedding_codegen_lookup_function)));

  m.def(
      "pruned_hashmap_lookup(Tensor indices, Tensor offsets, Tensor hash_table, int T) -> Tensor");
  m.impl(
      "pruned_hashmap_lookup",
      torch::dispatch(
          c10::DispatchKey::CUDA,
          TORCH_FN(pruned_hashmap_lookup_unweighted_cuda)));
}
