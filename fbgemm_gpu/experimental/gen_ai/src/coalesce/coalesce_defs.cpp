/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <torch/library.h>

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.def(
      "coalesce_batches(Tensor(a!)[] input_tensors, Tensor(a!)[] output_tensors, Tensor old_bids, Tensor new_bids) -> Tensor[]");
}
