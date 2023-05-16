/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>

namespace fbgemm_gpu {

at::Tensor batch_auc(
    const int64_t num_tasks,
    const at::Tensor& indices,
    const at::Tensor& labels,
    const at::Tensor& weights);

} // namespace fbgemm_gpu
