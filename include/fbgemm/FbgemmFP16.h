/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

// WARNING: this is a legacy fp16 fbgemm implementation and will soon be
// upgraded to match with new fbgemm interface.

#include <cpuinfo.h>
#include <cassert>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <vector>

#include "./Types.h"
#include "./Utils.h"
#include "./FbgemmPackMatrixB.h"

namespace fbgemm {

using PackedGemmMatrixFP16 = PackedGemmMatrixB<float16>;
/**
 * restrictions: transa == CblasNoTrans
 */
FBGEMM_API void cblas_gemm_compute(
    const matrix_op_t transa,
    const int m,
    const float* A,
    const PackedGemmMatrixFP16& Bp,
    const float beta,
    float* C,
    int thread_id = 0,
    int num_threads = 1);

}; // namespace fbgemm
