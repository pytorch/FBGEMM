/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "common.cuh"

JAGGED_TENSOR_OPS_CUDA_DISPATCH("dense_to_jagged", fbgemm_gpu::dense_to_jagged);
JAGGED_TENSOR_OPS_CUDA_DISPATCH(
    "jagged_to_padded_dense",
    fbgemm_gpu::jagged_to_padded_dense);
JAGGED_TENSOR_OPS_CUDA_DISPATCH(
    "jagged_dense_elementwise_add",
    fbgemm_gpu::jagged_dense_elementwise_add);
JAGGED_TENSOR_OPS_CUDA_DISPATCH(
    "jagged_dense_dense_elementwise_add_jagged_output",
    fbgemm_gpu::jagged_dense_dense_elementwise_add_jagged_output);
JAGGED_TENSOR_OPS_CUDA_DISPATCH(
    "jagged_dense_elementwise_mul",
    fbgemm_gpu::jagged_dense_elementwise_mul);
JAGGED_TENSOR_OPS_CUDA_DISPATCH(
    "batched_dense_vec_jagged_2d_mul",
    fbgemm_gpu::batched_dense_vec_jagged_2d_mul);
JAGGED_TENSOR_OPS_CUDA_DISPATCH(
    "jagged_1d_to_dense",
    fbgemm_gpu::jagged_1d_to_dense);
JAGGED_TENSOR_OPS_CUDA_DISPATCH(
    "jagged_2d_to_dense",
    fbgemm_gpu::jagged_2d_to_dense);

// TODO: combine the API with permute_2D_sparse_data and implement a CPU op

JAGGED_TENSOR_OPS_CUDA_DISPATCH("jagged_softmax", fbgemm_gpu::jagged_softmax);
JAGGED_TENSOR_OPS_CUDA_DISPATCH(
    "jagged_jagged_bmm",
    fbgemm_gpu::jagged_jagged_bmm);
JAGGED_TENSOR_OPS_CUDA_DISPATCH(
    "jagged_dense_bmm",
    fbgemm_gpu::jagged_dense_bmm);
