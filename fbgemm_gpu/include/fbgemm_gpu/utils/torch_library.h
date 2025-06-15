/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/core/dispatch/Dispatcher.h>

namespace fbgemm_gpu::utils::torch {

inline bool schemaExists(const std::string& qualified_name) {
  return c10::Dispatcher::singleton()
      .findSchema({qualified_name, ""})
      .has_value();
}

} // namespace fbgemm_gpu::utils::torch
