# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

def get_fbgemm_gpu_wrapper_srcs():
    return [
        "src/batched_unary_embedding_wrappers.cu",
        "src/quantize_wrappers.cu",
        "src/sparse_wrappers.cu",
    ]

def get_fbgemm_gpu_public_headers():
    return [
        "include/fbgemm_gpu/batched_unary_embedding_ops.cuh",
        "include/fbgemm_gpu/batched_unary_embedding_wrappers.cuh",
        "include/fbgemm_gpu/bench_utils.cuh",
        "include/fbgemm_gpu/cuda_utils.cuh",
        "include/fbgemm_gpu/quantize_ops.cuh",
        "include/fbgemm_gpu/quantize_wrappers.cuh",
        "include/fbgemm_gpu/sparse_ops.cuh",
        "include/fbgemm_gpu/sparse_wrappers.cuh",
    ]
