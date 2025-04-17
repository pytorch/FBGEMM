/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include "cute/layout.hpp"
#include "cutlass/detail/layout.hpp"
#include "cutlass/layout/matrix.h"
#include "cutlass/util/mixed_dtype_utils.hpp"
#include "cutlass/util/packed_stride.hpp"

namespace fbgemm_gpu {

std::tuple<at::Tensor, at::Tensor> fp8_preshuffle_i4(
    at::Tensor WQ,
    at::Tensor w_scale) {
  // Start by allocating space for shuffled tensors.
  at::Tensor WQ_shuffled = at::empty_like(WQ);
  // Packed scale contains 8 lookup values for each original scale element.
  at::Tensor w_scale_packed =
      at::empty({w_scale.size(0), 8, w_scale.size(1)}, w_scale.options());
  // WQ has two int4 values packed into each int8 dtype, so the size
  // is larger than it seems.
  size_t WQ_size = 2 * WQ.numel();
  // Encode weights to enable efficient lookup.
  cutlass::unified_encode_int4b(
      reinterpret_cast<cutlass::int4b_t*>(WQ.data_ptr()),
      reinterpret_cast<cutlass::int4b_t*>(WQ_shuffled.data_ptr()),
      WQ_size);

  size_t w_scale_size = w_scale.numel();
  cutlass::pack_scale_fp8(
      reinterpret_cast<cutlass::float_e4m3_t*>(w_scale.data_ptr()),
      reinterpret_cast<cutlass::Array<cutlass::float_e4m3_t, 8>*>(
          w_scale_packed.data_ptr()),
      w_scale_size);

  // Next we need to shuffle B. To do this, we define a few helper objects.
  const int N = WQ.size(0);
  const int K = 2 * WQ.size(1);
  auto shape_B = cute::make_shape(N, K, 1);
  using LayoutB = cutlass::layout::ColumnMajor;
  using StrideB = cutlass::detail::TagToStrideB_t<LayoutB>;
  using LayoutAtomQuant = decltype(cutlass::compute_memory_reordering_atom<
                                   cutlass::float_e4m3_t>());
  using LayoutB_Reordered = decltype(cute::tile_to_shape(
      LayoutAtomQuant{}, cute::Layout<cute::Shape<int, int, int>, StrideB>{}));
  StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, shape_B);
  auto layout_B = make_layout(shape_B, stride_B);
  LayoutB_Reordered layout_B_reordered =
      cute::tile_to_shape(LayoutAtomQuant{}, shape_B);
  ;

  // Now we're ready to reorder the tensor into proper layout.
  cutlass::reorder_tensor(
      reinterpret_cast<cutlass::int4b_t*>(WQ_shuffled.data_ptr()),
      layout_B,
      layout_B_reordered);

  // Tensors should now be preshuffled and ready for use.
  return {WQ_shuffled, w_scale_packed};
}

std::tuple<at::Tensor, at::Tensor> bf16_preshuffle_i4(
    at::Tensor WQ,
    at::Tensor w_scale) {
  // For bf16 we only preshuffle the weight tensor, scales arent modified.
  // Next we need to shuffle B. To do this, we define a few helper objects.
  const int N = WQ.size(0);
  const int K = 2 * WQ.size(1);
  auto shape_B = cute::make_shape(N, K, 1);
  using LayoutB = cutlass::layout::ColumnMajor;
  using StrideB = cutlass::detail::TagToStrideB_t<LayoutB>;
  using ValueShuffle = cute::Layout<
      cute::Shape<cute::_2, cute::_4>,
      cute::Stride<cute::_4, cute::_1>>; // order [0,2,4,6,1,3,5,7]
  int constexpr NumShuffleAtoms = 1;
  using MmaAtomShape =
      cute::Layout<cute::Shape<cute::_1, cute::Int<NumShuffleAtoms>>>;
  using LayoutAtomQuant = decltype(cutlass::compute_memory_reordering_atom<
                                   cutlass::bfloat16_t,
                                   MmaAtomShape,
                                   ValueShuffle>());
  using LayoutB_Reordered = decltype(cute::tile_to_shape(
      LayoutAtomQuant{}, cute::Layout<cute::Shape<int, int, int>, StrideB>{}));
  StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, shape_B);
  auto layout_B = make_layout(shape_B, stride_B);
  LayoutB_Reordered layout_B_reordered =
      cute::tile_to_shape(LayoutAtomQuant{}, shape_B);
  ;

  // Now we're ready to reorder the tensor into proper layout.
  cutlass::reorder_tensor(
      reinterpret_cast<cutlass::int4b_t*>(WQ.data_ptr()),
      layout_B,
      layout_B_reordered);

  // Tensors should now be preshuffled and ready for use.
  return {WQ, w_scale};
}

std::tuple<at::Tensor, at::Tensor> preshuffle_i4(
    at::Tensor WQ,
    at::Tensor w_scale) {
  TORCH_CHECK(
      w_scale.dtype() == at::kFloat8_e4m3fn || w_scale.dtype() == at::kBFloat16,
      "Activation type must be FP8 or BF16.");
  if (w_scale.dtype() == at::kFloat8_e4m3fn) {
    return fp8_preshuffle_i4(WQ, w_scale);
  } else {
    return bf16_preshuffle_i4(WQ, w_scale);
  }
}

} // namespace fbgemm_gpu
