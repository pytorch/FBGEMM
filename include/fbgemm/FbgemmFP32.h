// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

// WARNING: this is a legacy fp16 fbgemm implementation and will soon be
// upgraded to match with new fbgemm interface.

#include <cpuinfo.h>
#include <cassert>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <vector>

#include "fbgemm/FbgemmFPCommon.h"
#include "fbgemm/FbgemmPackMatrixB.h"
#include "fbgemm/Types.h"
#include "fbgemm/Utils.h"

namespace fbgemm {
template <>
struct TypeConverter<float> {
  float operator()(float src) const {
    return src;
  }
};

using GemmParamsFP32 = GemmParams<float>;
using PackedGemmMatrixFP32 = PackedGemmMatrixB<float>;

template <typename T, int _kernel_ncol_blocks, int _brow>
void cblas_gemm_compute(
    const matrix_op_t transa,
    const int m,
    const float* A,
    const PackedGemmMatrixB<T>& Bp,
    const float beta,
    float* C,
    int thread_id = 0,
    int num_threads = 1);

extern template void cblas_gemm_compute(
    const matrix_op_t transa,
    const int m,
    const float* A,
    const PackedGemmMatrixFP32& Bp,
    const float beta,
    float* C,
    int thread_id,
    int num_threads);

template <>
const isa_descriptor<float>& getIsaHandlers(inst_set_t isa, float);

} // namespace fbgemm
