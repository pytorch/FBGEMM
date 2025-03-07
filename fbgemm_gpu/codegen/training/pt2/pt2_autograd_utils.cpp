/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/TypeDefault.h>

using Tensor = at::Tensor;

namespace fbgemm_gpu {

////////////////////////////////////////////////////////////////////////////////
// Utils Functions for PT2 Autograd
////////////////////////////////////////////////////////////////////////////////

/// Reshape VBE output (or grad_output) to be in a fixed-batch size format to
/// work with non-VBE CPU kernel.
///
/// reshaped_grad_output              A 2D tensor that contains the grad output
///                                    of shape [max_B, total_D] where max_B is
///                                    maximum batch size and total_D is the
///                                    accumulated embedding dimensions
/// @param grad_output                A 1D tensor that contains the grad_output
///                                    in a VBE output format
/// @param max_B                      Maximum batch size
/// @param B_offsets_rank_per_feature A 2D tensor that contains batch size
///                                    offsets for all ranks for each feature
///                                    size(0) is number of features
///                                    size(1) is number of ranks
/// @param D_offsets                  Embedding dimension offsets. Dimension of
///                                    feature t = D_offsets[t-1] - D_offsets[t]
Tensor reshape_vbe_output(
    const Tensor& grad_output,
    const int64_t max_B,
    const Tensor& B_offsets_rank_per_feature,
    const Tensor& D_offsets) {
  /* FOR CPU VBE to use the same backend */
  const auto T = D_offsets.numel() - 1;
  auto D_offsets_acc = D_offsets.accessor<int32_t, 1>();
  const int32_t total_D = D_offsets_acc[T] - D_offsets_acc[0];
  auto grad_output_ = at::empty({max_B, total_D}, grad_output.options());
  // for each feature
  auto offset = 0;

  const int32_t R = B_offsets_rank_per_feature.size(1) - 1;
  auto B_offsets_rank_per_feature_acc =
      B_offsets_rank_per_feature.accessor<int32_t, 2>();
  for (int32_t r = 0; r < R; r++) {
    auto D_offset = 0;
    for (int32_t t = 0; t < T; t++) {
      const int32_t b_begin = B_offsets_rank_per_feature_acc[t][r];
      const int32_t b_end = B_offsets_rank_per_feature_acc[t][r + 1];
      const int32_t D = D_offsets_acc[t + 1] - D_offsets_acc[t];
      const int32_t b = b_end - b_begin;
      const int32_t num_elm = b * D;
      auto values = grad_output.slice(0, offset, offset + num_elm);
      values = values.reshape({b, D});
      grad_output_.index_put_(
          {at::indexing::Slice(b_begin, b_end),
           at::indexing::Slice(D_offset, D_offset + D)},
          values);
      D_offset += D;
      offset += num_elm;
    }
  }
  return grad_output_;
}

/// Bounds checking to ensure there's no buffer overflow for memcpy.
///
/// @param dst          Pointer to the destination.
/// @param dst_index    Starting index where the content is to be copied to
/// @param dst_size     Max number of bytes to modify in the destination.
/// @param src          Pointer to the source.
/// @param src_index    Starting index where the content is to be copied from
/// @param src_size     Max number of bytes to copy from the source
/// @param copy_size    Number of bytes to copy.
void checked_memcpy(
    void* dst,
    size_t dst_buffer,
    const void* src,
    size_t src_buffer,
    size_t copy_size) {
  TORCH_CHECK(
      dst_buffer >= copy_size,
      "Possible buffer overflow for memcpy. Expected to copy ",
      copy_size,
      " bytes, but destination buffer has only ",
      dst_buffer,
      " bytes.");
  TORCH_CHECK(
      src_buffer >= copy_size,
      "Possible buffer overflow for memcpy. Expected to copy ",
      copy_size,
      " bytes, but source buffer has only ",
      src_buffer,
      " bytes.");
  std::memcpy(dst, src, copy_size);
}

/// Pad VBE offsets to work with non-VBE CPU kernel.
///
/// reshaped_offsets                  A 1D tensor that contains the offsets
///                                    in a fixed-batch-size format
/// @param offsets                    A 1D tensor of VBE offsets for indices
/// @param B_offsets_rank_per_feature A 2D tensor that contains batch size
///                                    offsets for all ranks for each feature.
///                                    size(0) is number of features
///                                    size(1) is number of ranks
/// @param max_B                      Maximum batch size
/// @param T                          Number of embedding tables (features)
template <typename index_t>
Tensor reshape_vbe_offsets(
    const Tensor& offsets,
    const Tensor& B_offsets_rank_per_feature,
    const int64_t max_B,
    const int32_t T) {
  if (offsets.numel() == 0) {
    return offsets;
  }
  auto B_offsets_rank_per_feature_acc =
      B_offsets_rank_per_feature.accessor<int32_t, 2>();
  auto reshaped_offsets = at::empty({T * max_B + 1}, offsets.options());
  // TODO: support other types
  auto reshaped_offsets_acc = reshaped_offsets.accessor<index_t, 1>();
  auto offsets_acc = offsets.accessor<index_t, 1>();
  auto begin = 0;
  for (int32_t t = 0; t < T; t++) {
    const auto batch_size =
        B_offsets_rank_per_feature_acc[t]
                                      [B_offsets_rank_per_feature[t].numel() -
                                       1];
    const auto end = batch_size + begin;
    TORCH_CHECK(
        batch_size <= max_B && max_B >= 0 && batch_size >= 0,
        "batch size cannot exceed max_B of ",
        max_B,
        " but got ",
        batch_size);

    // copy the offsets
    auto dst_idx = t * max_B;
    checked_memcpy(
        &reshaped_offsets_acc[dst_idx],
        sizeof(offsets) * (reshaped_offsets.numel() - dst_idx),
        &offsets_acc[begin],
        sizeof(offsets) * (offsets.numel() - begin),
        sizeof(offsets) * batch_size);

    // fill the rest of the offsets with the last offset
    if (max_B - batch_size > 0) {
      dst_idx += batch_size;
      for (auto i = 0; i < max_B - batch_size; i++) {
        reshaped_offsets_acc[dst_idx + i] = offsets_acc[end];
      }
    }
    begin = end;
  }
  reshaped_offsets[reshaped_offsets.numel() - 1] = offsets[offsets.numel() - 1];
  return reshaped_offsets;
}

template Tensor reshape_vbe_offsets<int32_t>(
    const Tensor& offsets,
    const Tensor& B_offsets_rank_per_feature,
    const int64_t max_B,
    const int32_t T);

template Tensor reshape_vbe_offsets<int64_t>(
    const Tensor& offsets,
    const Tensor& B_offsets_rank_per_feature,
    const int64_t max_B,
    const int32_t T);

} // namespace fbgemm_gpu
