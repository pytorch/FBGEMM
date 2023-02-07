/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "fbgemm_gpu/group_gemm_ops.cuh"

namespace fbgemm_gpu {

using namespace cutlass;
using Tensor = at::Tensor;

namespace {
// Specialization to use tfloat32 tensor core operations in A100
template <>
struct cutlass_traits<float, arch::Sm80> {
  using Element = float;
  using ElementAcc = float;
  using OpClass = arch::OpClassTensorOp;

  // Using settings from
  // https://github.com/NVIDIA/cutlass/blob/v2.9.0/test/unit/gemm/device/gemm_tf32t_tf32t_f32t_tensor_op_f32_sm80.cu#L71-L73
  using ThreadblockShape = gemm::GemmShape<128, 256, 32>;
  using WarpShape = gemm::GemmShape<64, 64, 32>;
  using InstructionShape = gemm::GemmShape<16, 8, 8>;
};

// Specialization to use tensor core operations in A100
template <>
struct cutlass_traits<double, arch::Sm80> {
  using Element = double;
  using ElementAcc = double;
  using OpClass = arch::OpClassTensorOp;

  // TODO: should we use the settings at
  // https://github.com/NVIDIA/cutlass/blob/v2.9.0/test/unit/gemm/device/gemm_f64t_f64n_f64t_tensor_op_f64_sm80.cu#L200-L202

  // uses WarpShape 128, 256, 64 for double and Sm80, but this requires too big
  // shared memory size of 589,824 Bytes
  using ThreadblockShape = gemm::GemmShape<64, 128, 32>;
  // Also need to adjust WarpShape. Otherwise, get too many predicates error in
  // predicated_tile_access_iterator.h
  using WarpShape = gemm::GemmShape<64, 64, 32>;
  // default_gemm_configuration.h uses InstructionShape 16, 8, 16 for double
  // and Sm80, but arch/mma_sm80.h only defines Mma for 8, 8, 4
  using InstructionShape = gemm::GemmShape<8, 8, 4>;
};

template <>
struct cutlass_traits<at::Half, arch::Sm80> {
  using Element = half_t;
  using ElementAcc = float;
  using OpClass = arch::OpClassTensorOp;

  using ThreadblockShape = gemm::GemmShape<128, 128, 32>;
  using WarpShape = gemm::GemmShape<64, 64, 32>;
  using InstructionShape = gemm::GemmShape<16, 8, 16>;
};

} // namespace

#define INSTANTIATE_GEMM_GROUPED(scalar_t, LayoutB) \
  template std::vector<Tensor>                      \
  gemm_grouped_cuda<scalar_t, LayoutB, arch::Sm80>( \
      const std::vector<Tensor>& a_group,           \
      const std::vector<Tensor>& b_group,           \
      const c10::optional<std::vector<Tensor>>& c_group);

#define INSTANTIATE_LAYOUT(scalar_t)                   \
  INSTANTIATE_GEMM_GROUPED(scalar_t, layout::RowMajor) \
  INSTANTIATE_GEMM_GROUPED(scalar_t, layout::ColumnMajor)

INSTANTIATE_LAYOUT(double)
INSTANTIATE_LAYOUT(float)
INSTANTIATE_LAYOUT(at::Half)

} // namespace fbgemm_gpu
