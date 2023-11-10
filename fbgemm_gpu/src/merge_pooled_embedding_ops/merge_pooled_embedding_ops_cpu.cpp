/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <c10/core/TensorOptions.h>
#include <torch/library.h>
#include "fbgemm_gpu/dispatch_macros.h"
#include "fbgemm_gpu/ops_utils.h"

using Tensor = at::Tensor;

namespace fbgemm_gpu {

Tensor merge_pooled_embeddings_cpu(
    std::vector<Tensor> pooled_embeddings,
    int64_t /*uncat_dim_size*/,
    at::Device target_device,
    int64_t cat_dim = 1) {
  auto cat_host_0 = [&](const std::vector<Tensor>& ts) {
    int64_t n = 0;
    for (auto& t : ts) {
      n += t.numel();
    }
    Tensor r;
    if (n == 0) {
      r = at::empty({n});
    } else {
      r = at::empty({n}, ts[0].options());
    }
    r.resize_(0);
    return at::cat_out(r, ts, cat_dim); // concat the tensor list in dim = 1
  };
  auto result = cat_host_0(pooled_embeddings);

  // There is some corner case, the target_device is not CPU. So we move the
  // target results to the target device. This would allow sample inputs
  // staying on CPU.
  if (!target_device.is_cpu()) {
    result = result.to(target_device, true);
  }

  return result;
}

} // namespace fbgemm_gpu

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
#ifdef HAS_IMPL_ABSTRACT_PYSTUB
  m.impl_abstract_pystub(
      "fbgemm_gpu.sparse_ops",
      "//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_py");
#endif
  m.def(
      "merge_pooled_embeddings(Tensor[] pooled_embeddings, SymInt uncat_dim_size, Device target_device, SymInt cat_dim=1) -> Tensor",
      {PT2_COMPLIANT_TAG});
  m.def(
      "all_to_one_device(Tensor[] input_tensors, Device target_device) -> Tensor[]");
  m.def(
      "sum_reduce_to_one(Tensor[] input_tensors, Device target_device) -> Tensor");
}

FBGEMM_OP_DISPATCH(
    CPU,
    "merge_pooled_embeddings",
    fbgemm_gpu::merge_pooled_embeddings_cpu);
