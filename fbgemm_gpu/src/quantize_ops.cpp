/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// Copyright 2004-present Facebook. All Rights Reserved.
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>

#include "fbgemm_gpu/quantize_wrappers.cuh"

at::Tensor _float_to_fused8bitrowwise_gpu(const at::Tensor& input) {
  TORCH_CHECK(input.is_cuda());
  TORCH_CHECK(input.is_contiguous());

  c10::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(input.get_device());

  const auto input_sizes = input.sizes();
  const auto last_dim = input_sizes.size() - 1;
  const int nrows = c10::size_to_dim_(last_dim, input_sizes);
  const int ncols = input_sizes[last_dim];
  const int ncols_aligned = (ncols + 4 - 1) / 4 * 4;
  const int output_columns = ncols_aligned + 2 * sizeof(float);

  auto output_dims = input_sizes.vec();
  output_dims[last_dim] = output_columns;
  auto output = at::empty(
      output_dims, // 4 = sizeof(float)
      input.options().dtype(at::kByte));

  if (nrows == 0 || ncols == 0) {
    return output;
  }
  fbgemm_gpu_test::FloatToFused8BitRowwiseQuantized(
      nrows, ncols, input.data_ptr<float>(), output.data_ptr<std::uint8_t>());
  return output;
}
