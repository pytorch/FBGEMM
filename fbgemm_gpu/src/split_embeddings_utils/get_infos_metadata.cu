/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "fbgemm_gpu/embedding_backward_template_helpers.cuh" // @manual
#include "fbgemm_gpu/split_embeddings_utils.cuh" // @manual
#include "fbgemm_gpu/utils/ops_utils.h" // @manual

using Tensor = at::Tensor;
using namespace fbgemm_gpu;

DLL_PUBLIC std::tuple<int64_t, int64_t>
get_infos_metadata(Tensor unused, int64_t B, int64_t T) {
  return get_info_B_num_bits_from_T(T, B);
}
