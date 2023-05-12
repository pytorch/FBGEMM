/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>

namespace fbgemm_gpu {
///@defgroup merge-pooled-emb Merge Operators

///@ingroup merge-pooled-emb
std::vector<at::Tensor> all_to_one_device(
    std::vector<at::Tensor> inputTensors,
    at::Device target_device);

} // namespace fbgemm_gpu
