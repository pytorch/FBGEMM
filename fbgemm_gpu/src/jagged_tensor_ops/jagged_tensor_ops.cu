/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "common.cuh"

FBGEMM_OP_DISPATCH(CUDA, "dense_to_jagged", fbgemm_gpu::dense_to_jagged);
FBGEMM_OP_DISPATCH(
    CUDA,
    "jagged_to_padded_dense",
    fbgemm_gpu::jagged_to_padded_dense);
FBGEMM_OP_DISPATCH(
    CUDA,
    "jagged_dense_elementwise_add",
    fbgemm_gpu::jagged_dense_elementwise_add);
FBGEMM_OP_DISPATCH(
    CUDA,
    "jagged_dense_dense_elementwise_add_jagged_output",
    fbgemm_gpu::jagged_dense_dense_elementwise_add_jagged_output);
FBGEMM_OP_DISPATCH(
    CUDA,
    "jagged_dense_elementwise_mul",
    fbgemm_gpu::jagged_dense_elementwise_mul);
FBGEMM_OP_DISPATCH(
    CUDA,
    "batched_dense_vec_jagged_2d_mul",
    fbgemm_gpu::batched_dense_vec_jagged_2d_mul);
FBGEMM_OP_DISPATCH(CUDA, "jagged_1d_to_dense", fbgemm_gpu::jagged_1d_to_dense);
FBGEMM_OP_DISPATCH(CUDA, "jagged_2d_to_dense", fbgemm_gpu::jagged_2d_to_dense);

// TODO: combine the API with permute_2D_sparse_data and implement a CPU op

FBGEMM_OP_DISPATCH(CUDA, "jagged_softmax", fbgemm_gpu::jagged_softmax);
FBGEMM_OP_DISPATCH(CUDA, "jagged_jagged_bmm", fbgemm_gpu::jagged_jagged_bmm);
FBGEMM_OP_DISPATCH(CUDA, "jagged_dense_bmm", fbgemm_gpu::jagged_dense_bmm);

FBGEMM_OP_DISPATCH(
    CUDA,
    "jagged_index_select",
    fbgemm_gpu::jagged_index_select_2d);
