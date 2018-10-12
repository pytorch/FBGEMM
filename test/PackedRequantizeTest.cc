/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <gtest/gtest.h>

#include "fbgemm/Fbgemm.h"
#include "bench/BenchUtils.h"
#include "src/RefImplementations.h"
#include "QuantizationHelpers.h"
#include "TestUtils.h"

using namespace std;
using namespace fbgemm2;

std::vector<matrix_op_t> transposeVals{matrix_op_t::NoTranspose,
                                       matrix_op_t::Transpose};

namespace {
class fbgemmu8s8acc32test : public testing::TestWithParam<
                                std::tuple<matrix_op_t, matrix_op_t, bool>> {};
}; // namespace

INSTANTIATE_TEST_CASE_P(
    InstantiationName,
    fbgemmu8s8acc32test,
    ::testing::Combine(
        ::testing::Values(matrix_op_t::NoTranspose),
        ::testing::ValuesIn(transposeVals),
        ::testing::Bool()));

/**
 * @brief Shapes for unit test.
 */
static vector<vector<int>> GetShapes_() {
  // NMT
  vector<vector<int>> shapes = {
      // {M,    N,    K}
      {1, 128, 512},
      {1, 1024, 256},
      {1, 2048, 512},
      {1, 2048, 513},
      {1, 2048, 514},

      {6, 512, 512},
      {6, 2048, 512},
      {6, 256, 1024},
      {6, 1024, 256},
      {6, 2048, 256},
      {6, 2048, 257},
      {6, 2048, 258},

      {102, 1024, 512},
      {102, 2323, 256},
      {102, 512, 256},
      {102, 512, 257},
      {102, 512, 258},
  };
  return shapes;
}

/**
 * @brief Unit test for uint8 matrix A, int8 matrix B, and 32-bit
 * accumulation. Output processing: requantization -> nothing
 */
TEST_P(fbgemmu8s8acc32test, Test) {
  vector<vector<int>> shapes(GetShapes_());
  matrix_op_t atrans, btrans;
  bool test_ld;
  tie(atrans, btrans, test_ld) = GetParam();

  for (auto shape : shapes) {
    int m = shape[0];
    int n = shape[1];
    int k = shape[2];

    aligned_vector<uint8_t> Aint8(m * k, 0);

    // nxk matrix
    aligned_vector<int8_t> Bint8(k * n, 0);
    // kxn matrix
    aligned_vector<int8_t> Bint8_ref(k * n, 0);

    aligned_vector<int32_t> Cint32_ref(m * n, 0.0f);
    aligned_vector<int32_t> Cint32_fb(m * n, 0);
    aligned_vector<uint8_t> Cint8_fb(m * n, 0);
    aligned_vector<int32_t> Cint32_local(m * n, 0);
    aligned_vector<int32_t> Cint32_buffer(m * n, 0);
    aligned_vector<uint8_t> Cint8_local(m * n, 0);

    randFill(Aint8, 0, 255);
    int32_t Aint8_zero_point = 43;

    randFill(Bint8_ref, -128, 127);
    avoidOverflow(m, n, k, Aint8.data(), Bint8_ref.data());

    for (auto i = 0; i < Bint8.size(); ++i) {
      Bint8[i] = Bint8_ref[i];
    }

    if (btrans == matrix_op_t::Transpose) {
      transpose_matrix(Bint8.data(), k, n);
    }

    int32_t Bint8_zero_point = -30;
    // To test lda != k , we just reduce k by half and use the original k
    // as lda.
    int k_adjusted = k;
    int n_adjusted = n;
    if (test_ld) {
      assert(
          atrans == matrix_op_t::NoTranspose && "This case is not handled yet");
      k_adjusted = std::max(k / 2, 1);
      if (btrans == matrix_op_t::NoTranspose) {
        n_adjusted = std::max(n / 2, 1);
      }
    }

    // computing column offset
    vector<int32_t> col_offsets;
    col_offsets.resize(n_adjusted);
    col_offsets_with_zero_pt_s8acc32_ref(
        k_adjusted,
        n_adjusted,
        n,
        Bint8_ref.data(),
        Bint8_zero_point,
        col_offsets.data());

    vector<int32_t> row_offsets;
    row_offsets.resize(m);

    float C_multiplier = 0.1234;
    int32_t C_zero_pt = 5;

    matmul_u8i8acc32_ref(
        m,
        n_adjusted,
        k_adjusted,
        k,
        n,
        n,
        Aint8.data(),
        Bint8_ref.data(),
        Cint32_local.data());

    row_offsets_u8acc32_ref(m, k_adjusted, k, Aint8.data(), row_offsets.data());

    requantize_u8acc32_ref(
        m,
        n_adjusted,
        n,
        Cint32_local.data(),
        Cint8_local.data(),
        C_multiplier,
        C_zero_pt,
        Aint8_zero_point,
        Bint8_zero_point,
        row_offsets.data(),
        col_offsets.data(),
        nullptr);

    vector<int32_t> row_offset_buf;
    row_offset_buf.resize(PackAWithRowOffset<uint8_t>::rowOffsetBufferSize());

    PackAWithRowOffset<uint8_t> packAN(
        matrix_op_t::NoTranspose,
        m,
        k_adjusted,
        Aint8.data(),
        k,
        nullptr,
        1,
        Aint8_zero_point,
        row_offset_buf.data());

    PackBMatrix<int8_t> packedBN(
        btrans,
        k_adjusted,
        n_adjusted,
        Bint8.data(),
        (btrans == matrix_op_t::Transpose) ? k : n,
        nullptr,
        1,
        Bint8_zero_point);

    DoNothing<> doNothingObj{};
    ReQuantizeOutput<false> outputProcObj(
        doNothingObj,
        C_multiplier,
        C_zero_pt,
        Aint8_zero_point,
        Bint8_zero_point,
        packAN.getRowOffsetBuffer(),
        col_offsets.data(),
        nullptr);

    fbgemmPacked(
        packAN,
        packedBN,
        Cint8_fb.data(),
        Cint32_buffer.data(),
        n,
        outputProcObj,
        0,
        1);
    // printMatrix(matrix_op_t::NoTranspose, Cint32_local.data(),
    // m, n_adjusted, n, "C local");
    compare_validate_buffers(
        Cint8_local.data(), Cint8_fb.data(), m, n, n, static_cast<uint8_t>(0));
  }
}

/**
 * @brief Unit test for uint8 matrix A, int8 matrix B, and 32-bit
 * accumulation. Directly output fp32 matrix C. Output processing:
 * requantization -> nothing
 */
TEST_P(fbgemmu8s8acc32test, TestFloatInputOutput) {
  vector<vector<int>> shapes(GetShapes_());
  matrix_op_t atrans, btrans;
  bool test_ld;
  tie(atrans, btrans, test_ld) = GetParam();

  for (auto shape : shapes) {
    int m = shape[0];
    int n = shape[1];
    int k = shape[2];

    aligned_vector<float> Afp32(m * k, 0.0f);
    aligned_vector<uint8_t> Aint8(m * k, 0);

    aligned_vector<float> Bfp32(k * n, 0.0f);
    aligned_vector<int8_t> Bint8(k * n, 0);

    aligned_vector<float> Cfp32_ref(m * n, 0.0f);
    aligned_vector<float> Cfp32_fb(m * n, 0.0f);

    aligned_vector<uint8_t> Cint8_fb(m * n, 0);
    aligned_vector<int32_t> Cint32_buffer(m * n, 0);

    randFill(Aint8, 0, 255);
    int32_t Aint8_zero_point = 43;
    float Aint8_scale = 0.11;
    for (auto i = 0; i < Afp32.size(); ++i) {
      Afp32[i] = Aint8_scale * (Aint8[i] - Aint8_zero_point);
    }

    randFill(Bint8, -128, 127);
    avoidOverflow(m, n, k, Aint8.data(), Bint8.data());
    int32_t Bint8_zero_point = -30;
    float Bint8_scale = 0.49;
    for (auto i = 0; i < Bfp32.size(); ++i) {
      Bfp32[i] = Bint8_scale * (Bint8[i] - Bint8_zero_point);
    }

    // To test lda != k , we just reduce k by half and use the original k
    // as lda.
    int k_adjusted = k;
    int n_adjusted = n;
    if (test_ld) {
      assert(
          atrans == matrix_op_t::NoTranspose && "This case is not handled yet");
      k_adjusted = std::max(k / 2, 1);
      if (btrans == matrix_op_t::NoTranspose) {
        n_adjusted = std::max(n / 2, 1);
      }
    }

    // computing column offset
    vector<int32_t> col_offsets;
    col_offsets.resize(n_adjusted);
    col_offsets_with_zero_pt_s8acc32_ref(
        k_adjusted,
        n_adjusted,
        n,
        Bint8.data(),
        Bint8_zero_point,
        col_offsets.data());

    if (btrans == matrix_op_t::Transpose) {
      transpose_matrix(Bint8.data(), k, n);
    }

    matmul_fp_ref(
        m,
        n_adjusted,
        k_adjusted,
        k,
        n,
        n,
        Afp32.data(),
        Bfp32.data(),
        Cfp32_ref.data());

    vector<int32_t> row_offset_buf;
    row_offset_buf.resize(
        PackAWithQuantRowOffset<uint8_t>::rowOffsetBufferSize());

    PackAWithQuantRowOffset<uint8_t> packAN(
        matrix_op_t::NoTranspose,
        m,
        k_adjusted,
        Afp32.data(),
        k,
        nullptr, /*buffer for packed matrix*/
        Aint8_scale,
        Aint8_zero_point,
        1, /*groups*/
        row_offset_buf.data());

    PackBMatrix<int8_t> packedBN(
        btrans,
        k_adjusted,
        n_adjusted,
        Bint8.data(),
        (btrans == matrix_op_t::Transpose) ? k : n,
        nullptr,
        1,
        Bint8_zero_point);

    DoNothing<float, float> doNothingObj{};
    ReQuantizeForFloat<false> outputProcObj(
        doNothingObj,
        Aint8_scale,
        Bint8_scale,
        Aint8_zero_point,
        Bint8_zero_point,
        packAN.getRowOffsetBuffer(),
        col_offsets.data(),
        nullptr);

    fbgemmPacked(
        packAN,
        packedBN,
        Cfp32_fb.data(),
        (int32_t*)Cfp32_fb.data(),
        n,
        outputProcObj,
        0,
        1);

    float maximum = *max_element(Cfp32_ref.begin(), Cfp32_ref.end());
    float minimum = *min_element(Cfp32_ref.begin(), Cfp32_ref.end());
    float atol = (maximum - minimum) / 255 / 1.9;

    compare_validate_buffers(Cfp32_ref.data(), Cfp32_fb.data(), m, n, n, atol);
  }
}

/**
 * @brief Unit test for uint8 matrix A, int8 matrix B, and 32-bit
 * accumulation. Output processing: requantization -> nothing. Symmetric: the
 * zero point is 0.
 */
TEST_P(fbgemmu8s8acc32test, TestSymmetricQuantizedInputOutput) {
  vector<vector<int>> shapes(GetShapes_());
  matrix_op_t atrans, btrans;
  bool test_ld;
  tie(atrans, btrans, test_ld) = GetParam();

  for (auto shape : shapes) {
    int m = shape[0];
    int n = shape[1];
    int k = shape[2];

    aligned_vector<float> Afp32(m * k, 0.0f);
    aligned_vector<uint8_t> Aint8(m * k, 0);

    aligned_vector<float> Bfp32(k * n, 0.0f);
    aligned_vector<int8_t> Bint8(k * n, 0);

    aligned_vector<float> Cfp32_ref(m * n, 0.0f);
    aligned_vector<int32_t> Cint32_fb(m * n, 0);

    randFill(Afp32, 0, 255);
    for (auto i = 0; i < Afp32.size(); i++) {
      Aint8[i] = (uint8_t)Afp32[i];
    }

    // initialize B matrix
    randFill(Bfp32, -128, 127);
    avoidOverflow(m, n, k, Aint8.data(), Bfp32.data());

    for (auto i = 0; i < Bfp32.size(); ++i) {
      Bint8[i] = (int8_t)Bfp32[i];
    }

    // To test lda != k , we just reduce k by half and use the original k
    // as lda.
    int m_adjusted = m;
    int n_adjusted = n;
    int k_adjusted = k;
    if (test_ld) {
      assert(
          atrans == matrix_op_t::NoTranspose && "This case is not handled yet");
      k_adjusted = std::max(k / 2, 1);
      if (btrans == matrix_op_t::NoTranspose) {
        n_adjusted = std::max(n / 2, 1);
      }
    }

    if (btrans == matrix_op_t::Transpose) {
      transpose_matrix(Bint8.data(), k, n);
    }

    matmul_fp_ref(
        m,
        n_adjusted,
        k_adjusted,
        k,
        n,
        n,
        Afp32.data(),
        Bfp32.data(),
        Cfp32_ref.data());

    DoNothing<int32_t, int32_t> doNothingObj{};
    memCopy<> outputProcObj(doNothingObj);
    // A zero point and row offset not required
    PackAMatrix<uint8_t> packAN(
        matrix_op_t::NoTranspose, m, k_adjusted, Aint8.data(), k);

    // B zero point defaults to 0
    PackBMatrix<int8_t> packedBN(
        btrans,
        k_adjusted,
        n_adjusted,
        Bint8.data(),
        (btrans == matrix_op_t::Transpose) ? k : n);

    fbgemmPacked(
        packAN,
        packedBN,
        Cint32_fb.data(),
        Cint32_fb.data(),
        n,
        outputProcObj,
        0,
        1);

    // correctness check
    for (int i = 0; i < m_adjusted; ++i) {
      for (int j = 0; j < n_adjusted; ++j) {
        float expected = Cfp32_ref[i * n + j];
        int32_t actual = Cint32_fb[i * n + j];
        EXPECT_EQ(expected, actual)
            << "GEMM results differ at (" << i << ", " << j << "). ref "
            << expected << " FBGemm " << actual;
      }
    }
  }
}

/**
 * @brief Unit test for unt8 matrix A, int8 matrix B, and 32-bit
 * accumulation. Output processing: requantization with bias -> nothing.
 * Asymmetric: the zero point is not 0.
 */
TEST_P(fbgemmu8s8acc32test, TestAsymmetricQuantizedWithBias) {
  vector<vector<int>> shapes(GetShapes_());
  matrix_op_t atrans, btrans;
  bool test_ld;
  tie(atrans, btrans, test_ld) = GetParam();

  for (auto shape : shapes) {
    int m = shape[0];
    int n = shape[1];
    int k = shape[2];

    aligned_vector<uint8_t> Aint8(m * k, 0);
    aligned_vector<uint8_t> Aint8_ref(m * k, 0);

    aligned_vector<int8_t> Bint8(k * n, 0);
    aligned_vector<int8_t> Bint8_ref(k * n, 0);

    aligned_vector<int32_t> Cint32_fb(m * n, 0);
    aligned_vector<int32_t> Cint32_ref(m * n, 0);

    aligned_vector<uint8_t> Cint8_fb(m * n, 0);
    aligned_vector<uint8_t> Cint8_ref(m * n, 0);

    int n_adjusted = n;
    int k_adjusted = k;

    if (test_ld) {
      assert(
          atrans == matrix_op_t::NoTranspose && "This case is not handled yet");
      k_adjusted = std::max(k / 2, 1);
      if (btrans == matrix_op_t::NoTranspose) {
        n_adjusted = std::max(n / 2, 1);
      }
    }

    // A and B have scale 1, so exactly represented after quantization
    randFill(Aint8, 0, 255);
    randFill(Bint8, -128, 127);
    avoidOverflow(m, n, k, Aint8.data(), Bint8.data());

    for (auto i = 0; i < Bint8.size(); ++i) {
      Bint8_ref[i] = Bint8[i];
    }

    for (auto i = 0; i < Aint8.size(); ++i) {
      Aint8_ref[i] = Aint8[i];
    }

    int32_t Aint8_zero_point = 55;
    int32_t Bint8_zero_point = -17;

    // initialize bias
    aligned_vector<int32_t> bias_int32(n);
    randFill(bias_int32, -128, 127);

    if (btrans == matrix_op_t::Transpose) {
      transpose_matrix(Bint8.data(), k, n);
    }

    // computing column offset
    vector<int32_t> col_offsets;
    col_offsets.resize(n_adjusted);
    col_offsets_with_zero_pt_s8acc32_ref(
        k_adjusted,
        n_adjusted,
        n,
        Bint8_ref.data(),
        Bint8_zero_point,
        col_offsets.data());

    matmul_u8i8acc32_ref(
        m,
        n_adjusted,
        k_adjusted,
        k,
        n,
        n,
        Aint8.data(),
        Bint8_ref.data(),
        Cint32_ref.data());

    vector<int32_t> row_offsets;
    row_offsets.resize(m);

    row_offsets_u8acc32_ref(
        m, k_adjusted, k, Aint8_ref.data(), row_offsets.data());

    float C_multiplier = 0.1234;
    int32_t C_zero_pt = 5;

    requantize_u8acc32_ref(
        m,
        n_adjusted,
        n,
        Cint32_ref.data(),
        Cint8_ref.data(),
        C_multiplier,
        C_zero_pt,
        Aint8_zero_point,
        Bint8_zero_point,
        row_offsets.data(),
        col_offsets.data(),
        bias_int32.data());

    vector<int32_t> row_offset_buf;
    row_offset_buf.resize(PackAWithRowOffset<uint8_t>::rowOffsetBufferSize());

    PackAWithRowOffset<uint8_t> packAN(
        matrix_op_t::NoTranspose,
        m,
        k_adjusted,
        Aint8.data(),
        k,
        nullptr,
        1,
        Aint8_zero_point,
        row_offset_buf.data());

    PackBMatrix<int8_t> packedBN(
        btrans,
        k_adjusted,
        n_adjusted,
        Bint8.data(),
        (btrans == matrix_op_t::Transpose) ? k : n,
        nullptr,
        1,
        Bint8_zero_point);

    DoNothing<> doNothingObj{};
    ReQuantizeOutput<false> outputProcObj(
        doNothingObj,
        C_multiplier,
        C_zero_pt,
        Aint8_zero_point,
        Bint8_zero_point,
        packAN.getRowOffsetBuffer(),
        col_offsets.data(),
        bias_int32.data());

    fbgemmPacked(
        packAN,
        packedBN,
        Cint8_fb.data(),
        Cint32_fb.data(),
        n,
        outputProcObj,
        0,
        1);

    compare_validate_buffers(
        Cint8_fb.data(), Cint8_ref.data(), m, n, n, static_cast<uint8_t>(0));
  }
}
