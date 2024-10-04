/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/TypeDefault.h>
// #include <ATen/core/op_registration/op_registration.h>
// #include <torch/script.h>
// #include "fbgemm_gpu/embedding_common.h"
// #include "fbgemm_gpu/utils/dispatch_macros.h"
// #include "fbgemm_gpu/utils/ops_utils.h"
// #include "fbgemm_gpu/utils/tensor_utils.h"

using Tensor = at::Tensor;

namespace fbgemm_gpu {

////////////////////////////////////////////////////////////////////////////////
// Helper Functions
////////////////////////////////////////////////////////////////////////////////

Tensor reshape_vbe_output(
    const Tensor& grad_output,
    const Tensor& B_offsets,
    const Tensor& B_offsets_rank_per_feature,
    const Tensor& D_offsets);
} // namespace fbgemm_gpu
