/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "fbgemm_gpu/utils/dispatch_macros.h"

TORCH_LIBRARY_FRAGMENT(fb, m) {
  m.def(
      "batch_index_select_dim0("
      "    Tensor inputs,"
      "    Tensor indices,"
      "    SymInt[] input_num_indices,"
      "    SymInt[] input_rows,"
      "    SymInt[] input_columns,"
      "    bool permute_output_dim_0_1=False) -> Tensor");
}

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.set_python_module("fbgemm_gpu.sparse_ops");

  m.impl_abstract_pystub(
      "fbgemm_gpu.sparse_ops",
      "//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_py");

  m.def(
      "batch_index_select_dim0("
      "    Tensor inputs,"
      "    Tensor indices,"
      "    SymInt[] input_num_indices,"
      "    SymInt[] input_rows,"
      "    SymInt[] input_columns,"
      "    bool permute_output_dim_0_1=False) -> Tensor",
      {PT2_COMPLIANT_TAG});

  m.def(
      "batch_index_select_dim0_tensor("
      "    Tensor inputs,"
      "    Tensor indices,"
      "    Tensor input_num_indices,"
      "    Tensor input_rows,"
      "    Tensor input_columns,"
      "    bool permute_output_dim_0_1=False) -> Tensor");

  m.def(
      "batch_index_select_dim0_forward_cpu_impl("
      "   Tensor inputs,"
      "   Tensor indices,"
      "   SymInt[] input_num_indices,"
      "   SymInt[] input_rows,"
      "   SymInt[] input_columns,"
      "   bool permute_output_dim_0_1) -> Tensor[]");

  m.def(
      "batch_index_select_dim0_backward_cpu_impl("
      "   Tensor grad_output,"
      "   Tensor indices,"
      "   Tensor indices_numels,"
      "   Tensor input_num_indices,"
      "   Tensor input_rows,"
      "   Tensor input_columns,"
      "   bool permute_output_dim_0_1,"
      "   Tensor saved_tensor) -> Tensor");

  m.def(
      "batch_index_select_dim0_tensor_forward_cpu_impl("
      "   Tensor inputs,"
      "   Tensor indices,"
      "   Tensor input_num_indices,"
      "   Tensor input_rows,"
      "   Tensor input_columns,"
      "   bool permute_output_dim_0_1) -> Tensor[]");

  // CUDA ops

  m.def(
      "batch_index_select_dim0_forward_cuda_impl("
      "   Tensor inputs,"
      "   Tensor indices,"
      "   SymInt[] input_num_indices,"
      "   SymInt[] input_rows,"
      "   SymInt[] input_columns,"
      "   bool permute_output_dim_0_1) -> Tensor[]");

  m.def(
      "batch_index_select_dim0_backward_cuda_impl("
      "   Tensor grad_output,"
      "   Tensor dev_weights,"
      "   Tensor weights_offsets,"
      "   Tensor D_offsets,"
      "   Tensor hash_size_cumsum,"
      "   Tensor indices,"
      "   int max_segment_length_per_warp,"
      "   Tensor grad_offsets,"
      "   Tensor total_L_offsets,"
      "   bool permute_output_dim_0_1,"
      "   Tensor saved_tensor) -> Tensor");

  m.def(
      "batch_index_select_dim0_tensor_forward_cuda_impl("
      "   Tensor inputs,"
      "   Tensor indices,"
      "   Tensor input_num_indices,"
      "   Tensor input_rows,"
      "   Tensor input_columns,"
      "   bool permute_output_dim_0_1) -> Tensor[]");

  m.def(
      "batch_index_select_dim0_tensor_backward_cuda_impl("
      "   Tensor grad_output,"
      "   Tensor dev_weights,"
      "   Tensor weights_offsets,"
      "   Tensor D_offsets,"
      "   Tensor hash_size_cumsum,"
      "   Tensor indices,"
      "   int max_segment_length_per_warp,"
      "   Tensor grad_offsets,"
      "   Tensor total_L_offsets,"
      "   bool permute_output_dim_0_1,"
      "   Tensor saved_tensor) -> Tensor");
}
