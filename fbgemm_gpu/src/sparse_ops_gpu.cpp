/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "fbgemm_gpu/sparse_ops.h"
#include "fbgemm_gpu/sparse_ops_utils.h"

#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/library.h>

TORCH_LIBRARY_IMPL(fbgemm, CUDA, m) {
  DISPATCH_TO_CUDA("permute_sparse_data", at::fbgemm::permute_sparse_data_cuda);
  DISPATCH_TO_CUDA(
      "block_bucketize_sparse_features",
      at::fbgemm::block_bucketize_sparse_features_cuda);
  DISPATCH_TO_CUDA(
      "asynchronous_exclusive_cumsum", at::fbgemm::asynchronous_exclusive_cumsum_gpu);
  DISPATCH_TO_CUDA(
      "asynchronous_complete_cumsum", at::fbgemm::asynchronous_complete_cumsum_gpu);
  DISPATCH_TO_CUDA(
      "asynchronous_inclusive_cumsum", at::fbgemm::asynchronous_inclusive_cumsum_gpu);
  DISPATCH_TO_CUDA("reorder_batched_ad_lengths", at::fbgemm::reorder_batched_ad_lengths_gpu);
  DISPATCH_TO_CUDA("reorder_batched_ad_indices", at::fbgemm::reorder_batched_ad_indices_gpu);
}
