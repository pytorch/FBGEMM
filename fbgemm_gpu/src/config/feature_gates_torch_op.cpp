/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <torch/library.h>
#include "fbgemm_gpu/config/feature_gates.h"

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.def(
      "check_feature_gate_key(str key) -> bool",
      fbgemm_gpu::config::check_feature_gate_key);
}
