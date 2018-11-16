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
using namespace fbgemm;

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
    for (int groups : {1, 3, 4}) {
      for (bool test_bias: {false, true}) {
        int m = shape[0];
        int n = shape[1];
        int k = shape[2];
        if (k % groups != 0) {
          continue;
        }
        int k_per_group = k / groups;

        // mxk matrix
        aligned_vector<uint8_t> Aint8(m * k, 0);

        // kxn matrix
        aligned_vector<int8_t> Bint8(k * n, 0);
        aligned_vector<int8_t> Bint8_ref(Bint8.size(), 0);

        aligned_vector<int32_t> Cint32_ref(m * n * groups, 0);
        aligned_vector<uint8_t> Cint8_ref(Cint32_ref.size(), 0);
        aligned_vector<int32_t> Cint32_fb(Cint32_ref.size(), 0);
        aligned_vector<uint8_t> Cint8_fb(Cint32_ref.size(), 0);
        aligned_vector<int32_t> Cint32_buffer(Cint32_ref.size(), 0);

        randFill(Aint8, 0, 255);
        int32_t Aint8_zero_point = 43;

        randFill(Bint8_ref, -128, 127);
        for (int g = 0; g < groups; ++g) {
          avoidOverflow(
              m,
              n,
              k_per_group,
              Aint8.data() + g * k_per_group,
              k,
              Bint8_ref.data() + g * k_per_group * n,
              n);
        }

        Bint8 = Bint8_ref;

        // initialize bias
        aligned_vector<int32_t> bias_int32(groups * n);
        int32_t* bias = nullptr;
        if (test_bias) {
          randFill(bias_int32, -128, 127);
          bias = bias_int32.data();
        }

        if (btrans == matrix_op_t::Transpose) {
          aligned_vector<int8_t> Bint8_temp(Bint8.size());
          for (int g = 0; g < groups; ++g) {
            transpose_matrix(
                k_per_group,
                n,
                Bint8.data() + g * k_per_group * n,
                n,
                Bint8_temp.data() + g * k_per_group,
                groups * k_per_group);
          }
          Bint8 = Bint8_temp;
        }

        int32_t Bint8_zero_point = -30;
        // To test lda != k , we just reduce k by half and use the original k
        // as lda.
        int n_adjusted = n;
        if (test_ld) {
          assert(
              atrans == matrix_op_t::NoTranspose &&
              "This case is not handled yet");
          if (btrans == matrix_op_t::NoTranspose) {
            n_adjusted = std::max(n / 2, 1);
          }
        }

        // computing column offset
        vector<int32_t> col_offsets;
        col_offsets.resize(groups * n_adjusted);
        for (int g = 0; g < groups; ++g) {
          col_offsets_with_zero_pt_s8acc32_ref(
              k_per_group,
              n_adjusted,
              n,
              Bint8_ref.data() + g * k_per_group * n,
              Bint8_zero_point,
              col_offsets.data() + g * n_adjusted);
        }

        vector<int32_t> row_offsets;
        row_offsets.resize(m);

        float C_multiplier = 0.001234;
        int32_t C_zero_pt = 5;

        for (int g = 0; g < groups; ++g) {
          matmul_u8i8acc32_ref(
              m,
              n_adjusted,
              k_per_group,
              k,
              n,
              groups * n,
              Aint8.data() + g * k_per_group,
              Bint8_ref.data() + g * k_per_group * n,
              Cint32_ref.data() + g * n_adjusted);

          row_offsets_u8acc32_ref(
              m,
              k_per_group,
              k,
              Aint8.data() + g * k_per_group,
              row_offsets.data());

          requantize_u8acc32_ref(
              m,
              n_adjusted,
              groups * n,
              Cint32_ref.data() + g * n_adjusted,
              Cint8_ref.data() + g * n_adjusted,
              C_multiplier,
              C_zero_pt,
              Aint8_zero_point,
              Bint8_zero_point,
              row_offsets.data(),
              col_offsets.data() + g * n_adjusted,
              bias ? (bias + g * n_adjusted) : nullptr);
        }

        vector<int32_t> row_offset_buf;
        row_offset_buf.resize(
            PackAWithRowOffset<uint8_t>::rowOffsetBufferSize());

        PackAWithRowOffset<uint8_t> packAN(
            matrix_op_t::NoTranspose,
            m,
            k,
            Aint8.data(),
            k,
            nullptr,
            groups,
            Aint8_zero_point,
            row_offset_buf.data());

        PackBMatrix<int8_t> packedBN(
            btrans,
            k,
            n_adjusted,
            Bint8.data(),
            (btrans == matrix_op_t::Transpose) ? k : n,
            nullptr,
            groups,
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
            bias);

        fbgemmPacked(
            packAN,
            packedBN,
            Cint8_fb.data(),
            Cint32_buffer.data(),
            groups * n,
            outputProcObj,
            0,
            1);
        // printMatrix(matrix_op_t::NoTranspose, Cint32_local.data(),
        // m, n_adjusted, n, "C local");
        compare_validate_buffers(
            Cint8_ref.data(),
            Cint8_fb.data(),
            m,
            groups * n_adjusted,
            groups * n,
            static_cast<uint8_t>(0));
      } // test_bias
    } // for each groups
  } // for each shape
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
    for (int groups : {1, 3, 4}) {
      int m = shape[0];
      int n = shape[1];
      int k = shape[2];
      if (k % groups != 0) {
        continue;
      }
      int k_per_group = k / groups;

      aligned_vector<float> Afp32(m * k, 0.0f);
      aligned_vector<uint8_t> Aint8(Afp32.size(), 0);

      aligned_vector<float> Bfp32(k * n, 0.0f);
      aligned_vector<int8_t> Bint8(Bfp32.size(), 0);

      aligned_vector<float> Cfp32_ref(m * n * groups, 0.0f);
      aligned_vector<float> Cfp32_fb(Cfp32_ref.size(), 0.0f);

      aligned_vector<uint8_t> Cint8_fb(Cfp32_ref.size(), 0);
      aligned_vector<int32_t> Cint32_buffer(Cfp32_ref.size(), 0);

      randFill(Aint8, 0, 255);
      int32_t Aint8_zero_point = 43;
      float Aint8_scale = 0.11;
      for (auto i = 0; i < Afp32.size(); ++i) {
        Afp32[i] = Aint8_scale * (Aint8[i] - Aint8_zero_point);
      }

      randFill(Bint8, -128, 127);
      for (int g = 0; g < groups; ++g) {
        avoidOverflow(
            m,
            n,
            k_per_group,
            Aint8.data() + g * k_per_group,
            k,
            Bint8.data() + g * k_per_group * n,
            n);
      }
      int32_t Bint8_zero_point = -30;
      float Bint8_scale = 0.49;
      for (auto i = 0; i < Bfp32.size(); ++i) {
        Bfp32[i] = Bint8_scale * (Bint8[i] - Bint8_zero_point);
      }

      // To test lda != k , we just reduce k by half and use the original k
      // as lda.
      int n_adjusted = n;
      if (test_ld) {
        assert(
            atrans == matrix_op_t::NoTranspose &&
            "This case is not handled yet");
        if (btrans == matrix_op_t::NoTranspose) {
          n_adjusted = std::max(n / 2, 1);
        }
      }

      // computing column offset
      vector<int32_t> col_offsets;
      col_offsets.resize(groups * n_adjusted);
      for (int g = 0; g < groups; ++g) {
        col_offsets_with_zero_pt_s8acc32_ref(
            k_per_group,
            n_adjusted,
            n,
            Bint8.data() + g * k_per_group * n,
            Bint8_zero_point,
            col_offsets.data() + g * n_adjusted);
      }

      if (btrans == matrix_op_t::Transpose) {
        aligned_vector<int8_t> Bint8_temp(Bint8.size());
        for (int g = 0; g < groups; ++g) {
          transpose_matrix(
              k_per_group,
              n,
              Bint8.data() + g * k_per_group * n,
              n,
              Bint8_temp.data() + g * k_per_group,
              groups * k_per_group);
        }
        Bint8 = Bint8_temp;
      }

      for (int g = 0; g < groups; ++g) {
        matmul_fp_ref(
            m,
            n_adjusted,
            k_per_group,
            k,
            n,
            groups * n,
            Afp32.data() + g * k_per_group,
            Bfp32.data() + g * k_per_group * n,
            Cfp32_ref.data() + g * n_adjusted);
      }

      vector<int32_t> row_offset_buf;
      row_offset_buf.resize(
          PackAWithQuantRowOffset<uint8_t>::rowOffsetBufferSize());

      PackAWithQuantRowOffset<uint8_t> packAN(
          matrix_op_t::NoTranspose,
          m,
          k,
          Afp32.data(),
          k,
          nullptr, /*buffer for packed matrix*/
          Aint8_scale,
          Aint8_zero_point,
          groups,
          row_offset_buf.data());

      PackBMatrix<int8_t> packedBN(
          btrans,
          k,
          n_adjusted,
          Bint8.data(),
          (btrans == matrix_op_t::Transpose) ? k : n,
          nullptr,
          groups,
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
          reinterpret_cast<int32_t*>(Cfp32_fb.data()),
          groups * n,
          outputProcObj,
          0,
          1);

      float maximum = *max_element(Cfp32_ref.begin(), Cfp32_ref.end());
      float minimum = *min_element(Cfp32_ref.begin(), Cfp32_ref.end());
      float atol = (maximum - minimum) / 255 / 1.9;

      compare_validate_buffers(
          Cfp32_ref.data(),
          Cfp32_fb.data(),
          m,
          groups * n_adjusted,
          groups * n,
          atol);
    } // for each groups
  } // for each shape
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
    for (int groups : {1, 3, 4}) {
      int m = shape[0];
      int n = shape[1];
      int k = shape[2];
      if (k % groups != 0) {
        continue;
      }
      int k_per_group = k / groups;

      aligned_vector<float> Afp32(m * k, 0.0f);
      aligned_vector<uint8_t> Aint8(Afp32.size(), 0);

      aligned_vector<float> Bfp32(k * n, 0.0f);
      aligned_vector<int8_t> Bint8(Bfp32.size(), 0);

      aligned_vector<float> Cfp32_ref(m * n * groups, 0.0f);
      aligned_vector<int32_t> Cint32_fb(Cfp32_ref.size(), 0);

      randFill(Afp32, 0, 255);
      for (auto i = 0; i < Afp32.size(); i++) {
        Aint8[i] = (uint8_t)Afp32[i];
      }

      // initialize B matrix
      randFill(Bfp32, -128, 127);
      for (int g = 0; g < groups; ++g) {
        avoidOverflow(
            m,
            n,
            k_per_group,
            Aint8.data() + g * k_per_group,
            k,
            Bfp32.data() + g * k_per_group * n,
            n);
      }

      for (auto i = 0; i < Bfp32.size(); ++i) {
        Bint8[i] = (int8_t)Bfp32[i];
      }

      // To test lda != k , we just reduce k by half and use the original k
      // as lda.
      int n_adjusted = n;
      if (test_ld) {
        assert(
            atrans == matrix_op_t::NoTranspose &&
            "This case is not handled yet");
        if (btrans == matrix_op_t::NoTranspose) {
          n_adjusted = std::max(n / 2, 1);
        }
      }

      if (btrans == matrix_op_t::Transpose) {
        aligned_vector<int8_t> Bint8_temp(Bint8.size());
        for (int g = 0; g < groups; ++g) {
          transpose_matrix(
              k_per_group,
              n,
              Bint8.data() + g * k_per_group * n,
              n,
              Bint8_temp.data() + g * k_per_group,
              groups * k_per_group);
        }
        Bint8 = Bint8_temp;
      }

      for (int g = 0; g < groups; ++g) {
        matmul_fp_ref(
            m,
            n_adjusted,
            k_per_group,
            k,
            n,
            groups * n,
            Afp32.data() + g * k_per_group,
            Bfp32.data() + g * k_per_group * n,
            Cfp32_ref.data() + g * n_adjusted);
      }

      DoNothing<int32_t, int32_t> doNothingObj{};
      memCopy<> outputProcObj(doNothingObj);
      // A zero point and row offset not required
      PackAMatrix<uint8_t> packAN(
          matrix_op_t::NoTranspose, m, k, Aint8.data(), k, nullptr, groups);

      // B zero point defaults to 0
      PackBMatrix<int8_t> packedBN(
          btrans,
          k,
          n_adjusted,
          Bint8.data(),
          (btrans == matrix_op_t::Transpose) ? k : n,
          nullptr,
          groups);

      fbgemmPacked(
          packAN,
          packedBN,
          Cint32_fb.data(),
          Cint32_fb.data(),
          groups * n,
          outputProcObj,
          0,
          1);

      // correctness check
      for (int i = 0; i < m; ++i) {
        for (int j = 0; j < groups * n_adjusted; ++j) {
          float expected = Cfp32_ref[i * groups * n + j];
          int32_t actual = Cint32_fb[i * groups * n + j];
          EXPECT_EQ(expected, actual)
              << "GEMM results differ at (" << i << ", " << j << "). ref "
              << expected << " FBGemm " << actual;
        }
      }
    } // for each groups
  } // for each shape
}
