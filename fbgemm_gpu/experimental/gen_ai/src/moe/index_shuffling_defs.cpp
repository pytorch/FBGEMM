/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <torch/library.h>

namespace fbgemm_gpu {

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.set_python_module("fbgemm_gpu.experimental.gen_ai.moe");
  m.def(
      "index_shuffling(Tensor routing_scores,             "
      "                int? expert_index_start=None,      "
      "                int? expert_index_end=None,        "
      "                Tensor? valid_token_count=None,    "
      "                int top_k=1) ->                    "
      "(Tensor, Tensor, Tensor)");
}

} // namespace fbgemm_gpu
