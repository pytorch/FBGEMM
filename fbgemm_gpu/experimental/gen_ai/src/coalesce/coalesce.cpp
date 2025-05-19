/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "fbgemm_gpu/utils/ops_utils.h"

#include <ATen/core/op_registration/op_registration.h>
#include <torch/library.h>

#include <execution>
#include "coalesce.h"

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.def(
      "coalesce_batches(Tensor(a!)[] input_tensors, Tensor(a!)[] output_tensors, Tensor old_bids, Tensor new_bids) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(fbgemm, CUDA, m) {
  DISPATCH_TO_CUDA("coalesce_batches", fbgemm_gpu::coalesce_batches_gpu);
}

TORCH_LIBRARY_IMPL(fbgemm, CPU, m) {
  DISPATCH_TO_CPU("coalesce_batches", fbgemm_gpu::coalesce_batches_cpu);
}

namespace fbgemm_gpu {

template <typename BATCH_T>
void coalesc_batches_copy(
    const at::Tensor& src,
    const at::Tensor& dst,
    BATCH_T* old_bids_ptr,
    BATCH_T* new_bids_ptr,
    const int64_t num_bids) {
  uint8_t* src_data_ptr = reinterpret_cast<uint8_t*>(src.data_ptr());
  uint8_t* dst_data_ptr = reinterpret_cast<uint8_t*>(dst.data_ptr());
  int64_t element_size = src.numel() * src.element_size() / src.size(0);
  TORCH_CHECK_EQ(src.stride(0), dst.stride(0));
  int64_t stride = src.stride(0) * src.element_size();
  for (int i = 0; i < num_bids; ++i) {
    std::copy(
        src_data_ptr + old_bids_ptr[i] * stride,
        src_data_ptr + old_bids_ptr[i] * stride + element_size,
        dst_data_ptr + new_bids_ptr[i] * stride);
  }
}

std::vector<at::Tensor> coalesce_batches_cpu(
    const std::vector<at::Tensor>& input,
    const std::vector<at::Tensor>& output,
    const at::Tensor& old_bids,
    const at::Tensor& new_bids) {
  TORCH_CHECK_EQ(input.size(), output.size());

  const auto r = c10::irange(input.size());

  std::for_each(std::execution::par_unseq, r.begin(), r.end(), [&](auto i) {
    auto& src = input[i];
    auto& dst = output[i];
    if (old_bids.dtype() == at::kInt) {
      coalesc_batches_copy<int32_t>(
          src,
          dst,
          old_bids.data_ptr<int32_t>(),
          new_bids.data_ptr<int32_t>(),
          old_bids.numel());
    } else {
      coalesc_batches_copy<int64_t>(
          src,
          dst,
          old_bids.data_ptr<int64_t>(),
          new_bids.data_ptr<int64_t>(),
          old_bids.numel());
    }
  });

  return output;
}
} // namespace fbgemm_gpu
