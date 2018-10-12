/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <algorithm>
#include <chrono>
#include <cmath>
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
class fbgemmu8s8acc16test : public testing::TestWithParam<
                                std::tuple<matrix_op_t, matrix_op_t, bool>> {};
}; // namespace

INSTANTIATE_TEST_CASE_P(
    InstantiationName,
    fbgemmu8s8acc16test,
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
 * @brief Unit test for uint8 matrix A, int8 matrix B, and 16-bit
 * accumulation. Output processing: requantization -> nothing
 */
TEST_P(fbgemmu8s8acc16test, Test) {
  vector<vector<int>> shapes(GetShapes_());
  matrix_op_t atrans, btrans;
  bool test_ld;
  tie(atrans, btrans, test_ld) = GetParam();

  for (auto shape : shapes) {
    int m = shape[0];
    int n = shape[1];
    int k = shape[2];

    aligned_vector<uint8_t> Aint8(m * k, 0);
    aligned_vector<int8_t> Bint8(k * n, 0);
    aligned_vector<int8_t> Bint8_ref(k * n, 0);
    aligned_vector<int32_t> Cint32_local(m * n, 0);
    aligned_vector<int32_t> Cint32_buffer(m * n, 0);
    aligned_vector<int32_t> Cint32_fb(m * n, 0);
    aligned_vector<uint8_t> Cint8_fb(m * n, 0);
    aligned_vector<uint8_t> Cint8_local(m * n, 0);

    randFill(Aint8, 0, 255);
    int32_t Aint8_zero_point = 43;

    randFill(Bint8_ref, -128, 127);

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

    int brow = 256;
    matmul_u8i8acc16_ref(
        m,
        n_adjusted,
        k_adjusted,
        k,
        n,
        n,
        brow,
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
    row_offset_buf.resize(
        PackAWithRowOffset<uint8_t, int16_t>::rowOffsetBufferSize());

    PackAWithRowOffset<uint8_t, int16_t> packAN(
        matrix_op_t::NoTranspose,
        m,
        k_adjusted,
        Aint8.data(),
        k,
        nullptr,
        1,
        Aint8_zero_point,
        row_offset_buf.data());

    PackBMatrix<int8_t, int16_t> packedBN(
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

    compare_validate_buffers(
        Cint8_local.data(), Cint8_fb.data(), m, n, n, static_cast<uint8_t>(0));
  }
}

/**
 * @brief Unit test for uint8 matrix A, int8 matrix B, and 16-bit
 * accumulation. Output processing: spmdm -> requantization -> nothing
 */
TEST_P(fbgemmu8s8acc16test, SpMDMTest) {
  vector<vector<int>> shapes(GetShapes_());
  matrix_op_t atrans, btrans;
  bool test_ld;
  tie(atrans, btrans, test_ld) = GetParam();

  for (auto shape : shapes) {
    int m = shape[0];
    int n = shape[1];
    int k = shape[2];

    aligned_vector<uint8_t> Aint8(m * k, 0);
    aligned_vector<int8_t> Bint8(k * n, 0);
    aligned_vector<int8_t> Bint8_ref(k * n, 0);
    aligned_vector<int32_t> Cint32_local(m * n, 0);
    aligned_vector<int32_t> Cint32_buffer(m * n, 0);
    aligned_vector<int32_t> Cint32_fb(m * n, 0);
    aligned_vector<uint8_t> Cint8_fb(m * n, 0);
    aligned_vector<uint8_t> Cint8_local(m * n, 0);

    randFill(Aint8, 0, 255);
    int32_t Aint8_zero_point = 43;

    randFill(Bint8, -128, 127);

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

    int32_t Bint8_zero_point = -30;
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

    CompressedSparseColumn B_csc(k_adjusted, n_adjusted);
    float density = 0.001f;
    // deterministic random number
    default_random_engine eng;
    binomial_distribution<> per_col_nnz_dist(k_adjusted, density);
    uniform_int_distribution<> value_dist(
        numeric_limits<int8_t>::min() / 2, numeric_limits<int8_t>::max() / 2);

    vector<int> row_indices(k_adjusted);
    int total_nnz = 0;
    for (int j = 0; j < n_adjusted; ++j) {
      B_csc.ColPtr()[j] = total_nnz;

      int nnz_of_j = per_col_nnz_dist(eng);
      total_nnz += nnz_of_j;

      iota(row_indices.begin(), row_indices.end(), 0);
      shuffle(row_indices.begin(), row_indices.end(), eng);
      sort(row_indices.begin(), row_indices.begin() + nnz_of_j);

      for (int kidx = 0; kidx < nnz_of_j; ++kidx) {
        B_csc.RowIdx().push_back(row_indices[kidx]);
        // put the current B value
        B_csc.Values().push_back(Bint8[row_indices[kidx] * n + j]);
        // make current B value zero
        Bint8[row_indices[kidx] * n + j] = 0;
      }
    }
    B_csc.ColPtr()[n_adjusted] = total_nnz;

    for (auto i = 0; i < Bint8.size(); ++i) {
      Bint8_ref[i] = Bint8[i];
    }

    if (btrans == matrix_op_t::Transpose) {
      transpose_matrix(Bint8.data(), k, n);
    }

    vector<int32_t> row_offsets;
    row_offsets.resize(m);

    float C_multiplier = 0.1234;
    int32_t C_zero_pt = 5;

    int brow = 256;
    matmul_u8i8acc16_ref(
        m,
        n_adjusted,
        k_adjusted,
        k,
        n,
        n,
        brow,
        Aint8.data(),
        Bint8_ref.data(),
        Cint32_local.data());

    bool accumulation = true;
    spmdm_ref(m, Aint8.data(), k, B_csc, accumulation, Cint32_local.data(), n);

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
    row_offset_buf.resize(
        PackAWithRowOffset<uint8_t, int16_t>::rowOffsetBufferSize());

    PackAWithRowOffset<uint8_t, int16_t> packAN(
        matrix_op_t::NoTranspose,
        m,
        k_adjusted,
        Aint8.data(),
        k,
        nullptr,
        1,
        Aint8_zero_point,
        row_offset_buf.data());

    // spmdm -> requantization -> nothing
    // construct an output processing pipeline in reverse order
    // i.e. last output operation first
    // Last operation should always be DoNothing with
    // correct input and output type.
    DoNothing<> doNothingObj{};
    // The second last operation is requantization back
    // to int8
    ReQuantizeOutput<false> reqObj(
        doNothingObj,
        C_multiplier,
        C_zero_pt,
        Aint8_zero_point,
        Bint8_zero_point,
        packAN.getRowOffsetBuffer(),
        col_offsets.data(),
        nullptr);
    // the top most (first) operation in the output processing
    // pipeline is spmdm
    // outType = final output type after fullly processing through pipeline
    // inType = initial input type at the first call to the whole pipeline
    DoSpmdmOnInpBuffer<
        ReQuantizeOutput<false>::outType,
        int32_t,
        ReQuantizeOutput<false>>
        spmdmObj(reqObj, Aint8.data(), k, B_csc);

    PackBMatrix<int8_t, int16_t> packedB(
        btrans,
        k_adjusted,
        n_adjusted,
        Bint8.data(),
        (btrans == matrix_op_t::Transpose) ? k : n);

    fbgemmPacked(
        packAN, packedB, Cint8_fb.data(), Cint32_fb.data(), n, spmdmObj, 0, 1);

    compare_validate_buffers(
        Cint8_local.data(), Cint8_fb.data(), m, n, n, static_cast<uint8_t>(0));
  }
}

/**
 * @brief Unit test for uint8 matrix A, int8 matrix B, and 16-bit
 * accumulation. Output processing: nothing
 */
TEST_P(fbgemmu8s8acc16test, NoRequantizeTest) {
  vector<vector<int>> shapes(GetShapes_());
  matrix_op_t atrans, btrans;
  bool test_ld;
  tie(atrans, btrans, test_ld) = GetParam();

  for (auto shape : shapes) {
    int m = shape[0];
    int n = shape[1];
    int k = shape[2];

    aligned_vector<uint8_t> Aint8(m * k, 0);
    aligned_vector<int8_t> Bint8(k * n, 0);
    aligned_vector<int8_t> Bint8_ref(k * n, 0);
    aligned_vector<int32_t> Cint32_local(m * n, 0);
    aligned_vector<int32_t> Cint32_buffer(m * n, 0);
    aligned_vector<int32_t> Cint32_fb(m * n, 0);
    aligned_vector<uint8_t> Cint8_fb(m * n, 0);
    aligned_vector<uint8_t> Cint8_local(m * n, 0);

    randFill(Aint8, 0, 255);
    int32_t Aint8_zero_point = 43;

    randFill(Bint8_ref, -128, 127);

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

    int brow = 256;
    matmul_u8i8acc16_ref(
        m,
        n_adjusted,
        k_adjusted,
        k,
        n,
        n,
        brow,
        Aint8.data(),
        Bint8_ref.data(),
        Cint32_local.data());

    row_offsets_u8acc32_ref(m, k_adjusted, k, Aint8.data(), row_offsets.data());

    vector<int32_t> row_offset_buf;
    row_offset_buf.resize(
        PackAWithRowOffset<uint8_t, int16_t>::rowOffsetBufferSize());

    PackAWithRowOffset<uint8_t, int16_t> packAN(
        matrix_op_t::NoTranspose,
        m,
        k_adjusted,
        Aint8.data(),
        k,
        nullptr,
        1,
        Aint8_zero_point,
        row_offset_buf.data());

    PackBMatrix<int8_t, int16_t> packedBN(
        btrans,
        k_adjusted,
        n_adjusted,
        Bint8.data(),
        (btrans == matrix_op_t::Transpose) ? k : n,
        nullptr,
        1,
        Bint8_zero_point);

    // DoNothing<> doNothingObj{};
    DoNothing<int32_t, int32_t> doNothingObj{};
    memCopy<> outputProcObj(doNothingObj);
    fbgemmPacked(
        packAN,
        packedBN,
        Cint32_fb.data(),
        Cint32_buffer.data(),
        n,
        outputProcObj,
        0,
        1);

    compare_validate_buffers(
        Cint32_local.data(),
        Cint32_fb.data(),
        m,
        n,
        n,
        static_cast<int32_t>(0));
  }
}
