/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <ATen/ATen.h>

enum PoolingMode { SUM = 0, MEAN = 1 };

at::Tensor split_embedding_codegen_forward_cpu(
    at::Tensor weights,
    at::Tensor weights_offsets,
    at::Tensor D_offsets,
    int64_t total_D,
    at::Tensor indices,
    at::Tensor offsets,
    int64_t pooling_mode,
    at::Tensor indice_weights);

at::Tensor split_embedding_codegen_grad_indice_weights_cpu(
    at::Tensor grad_output,
    at::Tensor weights,
    at::Tensor weights_offsets,
    at::Tensor D_offsets,
    at::Tensor indices,
    at::Tensor offsets,
    at::Tensor feature_requires_grad);
