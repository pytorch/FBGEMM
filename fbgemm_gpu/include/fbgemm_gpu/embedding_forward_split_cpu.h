/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>

at::Tensor split_embedding_codegen_forward_cpu(
    at::Tensor weights,
    at::Tensor weights_offsets,
    at::Tensor D_offsets,
    c10::SymInt total_D,
    at::Tensor hash_size_cumsum,
    at::Tensor indices,
    at::Tensor offsets,
    int64_t pooling_mode,
    at::Tensor indice_weights,
    int64_t output_dtype = 0 /* SparseType.FP32 */);

at::Tensor split_embedding_codegen_grad_indice_weights_cpu(
    at::Tensor grad_output,
    at::Tensor weights,
    at::Tensor weights_offsets,
    at::Tensor D_offsets,
    at::Tensor indices,
    at::Tensor offsets,
    at::Tensor feature_requires_grad);
