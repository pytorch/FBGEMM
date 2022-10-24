/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <ATen/ATen.h>
#include <ATen/TypeDefault.h>
#include <torch/library.h>
#include "c10/core/ScalarType.h"

using Tensor = at::Tensor;

Tensor int_nbit_split_embedding_codegen_lookup_function_cpu(
    Tensor dev_weights,
    Tensor uvm_weights,
    Tensor weights_placements,
    Tensor weights_offsets,
    Tensor weights_tys,
    Tensor D_offsets,
    int64_t total_D,
    int64_t max_int2_D,
    int64_t max_int4_D,
    int64_t max_int8_D,
    int64_t max_float16_D,
    int64_t max_float32_D,
    Tensor indices,
    Tensor offsets,
    int64_t pooling_mode,
    c10::optional<Tensor> indice_weights,
    int64_t output_dtype,
    c10::optional<Tensor> lxu_cache_weights,
    c10::optional<Tensor> lxu_cache_locations,
    c10::optional<int64_t> row_alignment,
    c10::optional<int64_t> max_float8_D,
    c10::optional<int64_t> fp8_exponent_bits,
    c10::optional<int64_t> fp8_exponent_bias);
