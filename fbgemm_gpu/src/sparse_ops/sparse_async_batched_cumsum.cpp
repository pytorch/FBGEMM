/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include "fbgemm_gpu/sparse_ops.h"
#include "fbgemm_gpu/utils/dispatch_macros.h"

namespace fbgemm_gpu {

at::Tensor asynchronous_batched_complete_cumsum_cpu(const at::Tensor& values) {
  auto B = values.size(0);
  auto len = values.size(1);
  auto output = at::empty({B, len + 1}, values.options());
  const at::Tensor index = at::range(0, len, at::kLong).cpu();
  for (auto i : c10::irange(B)) {
    at::Tensor t = output[i];
    at::index_put_(
        t, {index}, fbgemm_gpu::asynchronous_complete_cumsum_cpu(values[i]));
  }
  return output;
}

at::Tensor asynchronous_batched_complete_cumsum_meta(const at::Tensor& values) {
  auto B = values.sym_size(0);
  auto len = values.sym_size(1);
  auto output = at::native::empty_meta_symint(
      {B, len + 1},
      /*dtype=*/::std::make_optional(values.scalar_type()),
      /*layout=*/::std::make_optional(values.layout()),
      /*device=*/::std::make_optional(c10::Device(c10::kMeta)),
      /*pin_memory=*/::std::nullopt);
  return output;
}

} // namespace fbgemm_gpu

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.def("asynchronous_batched_complete_cumsum(Tensor values) -> Tensor");
}

TORCH_LIBRARY_IMPL(fbgemm, CPU, m) {
  m.impl(
      "asynchronous_batched_complete_cumsum",
      fbgemm_gpu::asynchronous_batched_complete_cumsum_cpu);
}

TORCH_LIBRARY_IMPL(fbgemm, Meta, m) {
  m.impl(
      "asynchronous_batched_complete_cumsum",
      fbgemm_gpu::asynchronous_batched_complete_cumsum_meta);
}
