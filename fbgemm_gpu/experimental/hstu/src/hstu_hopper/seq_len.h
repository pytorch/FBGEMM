/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
 ******************************************************************************/

#pragma once

#include <cutlass/cutlass.h>
#include <cute/layout.hpp>

namespace flash {

class VarSeqLenTraits {
public:
  // Total number of queries / keys. Unpadded.
  const int sum_s = 0;
  // seq len offsets.
  const int *cu_seq_len = nullptr;
  // targets nums
  const int *num_targets = nullptr;
  // context nums
  const int *num_contexts = nullptr;
  // seq len of the current batch.
  int max_seq_len = -1;
  int actual_seq_len = -1;
  int actual_seq_len_h = -1;
  int actual_seq_len_c = 0;

  using ShapeT = cute::Shape<int32_t, int32_t, int32_t>;
  using StrideT = cute::Shape<int64_t, _1, int64_t>;
  using LayoutT = cute::Layout<ShapeT, StrideT>;

  using ShapeRabT = cute::Shape<int32_t, int32_t, int32_t, int32_t>;
  using StrideRabT = cute::Shape<int64_t, _1, int64_t, int64_t>;
  using LayoutRabT = cute::Layout<ShapeRabT, StrideRabT>;

  CUTLASS_HOST_DEVICE VarSeqLenTraits() {}

  CUTLASS_HOST_DEVICE VarSeqLenTraits(
      const int sum_s, const int max_seq_len, const int *cu_seq_len, const int *num_targets = nullptr, const int *num_contexts = nullptr):
      sum_s(sum_s), max_seq_len(max_seq_len), cu_seq_len(cu_seq_len), num_targets(num_targets), num_contexts(num_contexts) {}

  // Returns the layout of a tensor in MKHB format in global memory.
  CUTLASS_HOST_DEVICE auto get_gmem_layout(
      int m, int k, int h, int b,
      int64_t m_stride, int64_t h_stride) const {
    return make_layout(make_shape(sum_s, k, h),
                       make_stride(m_stride, cute::_1{}, h_stride));
  }

  CUTLASS_DEVICE void init(int bidb) {
    actual_seq_len = cu_seq_len[bidb + 1] - cu_seq_len[bidb];
  }

  CUTLASS_DEVICE void init_h(int bidb) {
    actual_seq_len_h = actual_seq_len - num_targets[bidb];
  }

  CUTLASS_DEVICE void init_c(int bidb) {
    actual_seq_len_c = num_contexts[bidb];
  }

  template <typename MTensor, typename Shape>
  CUTLASS_DEVICE auto get_local_tile_tensor(
      const MTensor &m_tensor, const Shape &tile_shape,
      int bidh, int bidb) const {
    auto g_offset = local_tile(
      m_tensor(_, _, bidh),
      cute::make_shape(1, get<1>(tile_shape)),
      make_coord(cu_seq_len[bidb], _0{}));
    auto g_sequence = make_tensor(
        g_offset.data(),
        make_layout(
          cute::make_shape(actual_seq_len, get<1>(tile_shape)),
          g_offset.stride()
        ));
    auto g_tensor = local_tile(g_sequence, tile_shape, make_coord(_, _0{}));
    return g_tensor;
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace flash
