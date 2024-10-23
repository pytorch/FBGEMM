/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>

using Tensor = at::Tensor;

namespace fbgemm_gpu {

namespace {
inline Tensor native_empty_like(const Tensor& self) {
  return at::native::empty_like(
      self,
      c10::optTypeMetaToScalarType(self.options().dtype_opt()),
      self.options().layout_opt(),
      self.options().device_opt(),
      self.options().pinned_memory_opt(),
      std::nullopt);
}

} // namespace

}; // namespace fbgemm_gpu
