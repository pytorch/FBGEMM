/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <cmath>
#include <cstdio>

#include <gtest/gtest.h>

#include "bench/AlignedVec.h"
#include "bench/BenchUtils.h"
#include "fbgemm/Fbgemm.h"
#include "src/RefImplementations.h"
#include "TestUtils.h"

using namespace std;

namespace fbgemm2 {

// From Faster-RCNN with ShuffleNet
static vector<conv_param_t<>> shapes = {
    // MB, IC, OC, IH, IW, G, KH, KW, stride_h, stride_w, pad_h, pad_w
    conv_param_t<>(1, 32, 32, {14, 14}, 1, {3, 3}, {1, 1}, {0, 0}),
    conv_param_t<>(1, 32, 32, {14, 14}, 1, {3, 3}, {1, 1}, {1, 1}),
    conv_param_t<>(2, 32, 32, {14, 14}, 1, {3, 3}, {1, 1}, {0, 0}),
    conv_param_t<>(2, 32, 32, {14, 14}, 1, {3, 3}, {1, 1}, {1, 1}),
    // conv_param_t<>(1, 272, 272, {47, 125}, 1, {3, 3}, {1, 1}, {1, 1}),
    // conv_param_t<>(1, 272, 272, {64, 125}, 1, {3, 3}, {1, 1}, {1, 1}),
    // conv_param_t<>(1, 272, 272, {66, 125}, 1, {3, 3}, {1, 1}, {1, 1}),
    // conv_param_t<>(1, 272, 272, {67, 100}, 1, {3, 3}, {1, 1}, {1, 1}),
    // conv_param_t<>(1, 272, 272, {75, 75}, 1, {3, 3}, {1, 1}, {1, 1}),
    // conv_param_t<>(1, 272, 272, {75, 76}, 1, {3, 3}, {1, 1}, {1, 1}),
    // conv_param_t<>(1, 272, 272, {75, 100}, 1, {3, 3}, {1, 1}, {1, 1}),
    // conv_param_t<>(1, 272, 272, {94, 75}, 1, {3, 3}, {1, 1}, {1, 1}),
    // conv_param_t<>(1, 272, 272, {109, 75}, 1, {3, 3}, {1, 1}, {1, 1}),
    // conv_param_t<>(1, 544, 544, {24, 63}, 1, {3, 3}, {1, 1}, {1, 1}),
    // conv_param_t<>(1, 544, 544, {33, 63}, 1, {3, 3}, {1, 1}, {1, 1}),
    // conv_param_t<>(1, 544, 544, {34, 50}, 1, {3, 3}, {1, 1}, {1, 1}),
    // conv_param_t<>(1, 544, 544, {36, 63}, 1, {3, 3}, {1, 1}, {1, 1}),
    // conv_param_t<>(1, 544, 544, {38, 38}, 1, {3, 3}, {1, 1}, {1, 1}),
    // conv_param_t<>(1, 544, 544, {38, 40}, 1, {3, 3}, {1, 1}, {1, 1}),
    // conv_param_t<>(1, 544, 544, {47, 38}, 1, {3, 3}, {1, 1}, {1, 1}),
    // conv_param_t<>(1, 1088, 1088, {7, 7}, 1, {3, 3}, {1, 1}, {1, 1}),
    // conv_param_t<>(51, 1088, 1088, {7, 7}, 1, {3, 3}, {1, 1}, {1, 1}),
    // conv_param_t<>(100, 1088, 1088, {7, 7}, 1, {3, 3}, {1, 1}, {1, 1}),
    // conv_param_t<>(1, 248, 248, {93, 250}, 1, {3, 3}, {2, 2}, {1, 1}),
    // conv_param_t<>(1, 248, 248, {128, 250}, 1, {3, 3}, {2, 2}, {1, 1}),
    // conv_param_t<>(1, 248, 248, {133, 200}, 1, {3, 3}, {2, 2}, {1, 1}),
    // conv_param_t<>(1, 248, 248, {150, 150}, 1, {3, 3}, {2, 2}, {1, 1}),
    // conv_param_t<>(1, 248, 248, {150, 151}, 1, {3, 3}, {2, 2}, {1, 1}),
    // conv_param_t<>(1, 248, 248, {150, 158}, 1, {3, 3}, {2, 2}, {1, 1}),
    // conv_param_t<>(1, 248, 248, {188, 150}, 1, {3, 3}, {2, 2}, {1, 1}),
    // conv_param_t<>(1, 248, 248, {225, 150}, 1, {3, 3}, {2, 2}, {1, 1}),
    // conv_param_t<>(1, 272, 272, {47, 125}, 1, {3, 3}, {2, 2}, {1, 1}),
    // conv_param_t<>(1, 272, 272, {64, 125}, 1, {3, 3}, {2, 2}, {1, 1}),
    // conv_param_t<>(1, 272, 272, {66, 125}, 1, {3, 3}, {2, 2}, {1, 1}),
    // conv_param_t<>(1, 272, 272, {67, 100}, 1, {3, 3}, {2, 2}, {1, 1}),
    // conv_param_t<>(1, 272, 272, {75, 75}, 1, {3, 3}, {2, 2}, {1, 1}),
    // conv_param_t<>(1, 272, 272, {75, 76}, 1, {3, 3}, {2, 2}, {1, 1}),
    // conv_param_t<>(1, 272, 272, {94, 75}, 1, {3, 3}, {2, 2}, {1, 1}),
    conv_param_t<>(1, 544, 544, {14, 14}, 1, {3, 3}, {2, 2}, {1, 1}),
    // conv_param_t<>(51, 544, 544, {14, 14}, 1, {3, 3}, {2, 2}, {1, 1}),
    // conv_param_t<>(3, 544, 544, {14, 14}, 1, {3, 3}, {2, 2}, {1, 1}),
    // conv_param_t<>(100, 544, 544, {14, 14}, 1, {3, 3}, {2, 2}, {1, 1}),
    conv_param_t<>(1, 8, 8, {4, 4}, 1, {3, 3}, {1, 1}, {1, 1}),
};

TEST(FBGemmIm2colTest, Acc32Test) {
  for (auto conv_p : shapes) {
    aligned_vector<uint8_t> Aint8(
        conv_p.MB * conv_p.IN_DIM[0] * conv_p.IN_DIM[1] * conv_p.IC, 0);
    aligned_vector<int8_t> Bint8(
        conv_p.K[0] * conv_p.K[1] * conv_p.IC * conv_p.OC, 0);
    aligned_vector<int32_t> Cint32_ref(
        conv_p.MB * conv_p.OUT_DIM[0] * conv_p.OUT_DIM[1] * conv_p.OC, 0.0f);
    aligned_vector<int32_t> Cint32_fb(
        conv_p.MB * conv_p.OUT_DIM[0] * conv_p.OUT_DIM[1] * conv_p.OC, 0);

    randFill(Aint8, 0, 80);
    int32_t Aint8_zero_point = 43;
    randFill(Bint8, -16, 16);

    conv_ref(
        conv_p,
        Aint8.data(),
        Aint8_zero_point,
        Bint8.data(),
        Cint32_ref.data());

    int NDim = conv_p.OC;
    int KDim = conv_p.K[0] * conv_p.K[1] * conv_p.IC;

    vector<int32_t> row_offset_buf;
    row_offset_buf.resize(
        PackAWithIm2Col<uint8_t, int32_t>::rowOffsetBufferSize());

    PackAWithIm2Col<uint8_t, int32_t> packA(
        conv_p, Aint8.data(), nullptr, Aint8_zero_point, row_offset_buf.data());

    PackBMatrix<int8_t, int32_t> packedB(
        matrix_op_t::NoTranspose, KDim, NDim, Bint8.data(), NDim);

    // no-op output process objects
    DoNothing<int32_t, int32_t> doNothing32BitObj;
    memCopy<> memcopyObj(doNothing32BitObj);

    fbgemmPacked(
        packA,
        packedB,
        Cint32_fb.data(),
        Cint32_fb.data(),
        NDim,
        memcopyObj,
        0,
        1);

    // correctness check
    for (int n = 0; n < conv_p.MB; ++n) {
      for (int h = 0; h < conv_p.OUT_DIM[0]; ++h) {
        for (int w = 0; w < conv_p.OUT_DIM[1]; ++w) {
          for (int k = 0; k < conv_p.OC; ++k) {
            int32_t expected = Cint32_ref
                [((n * conv_p.OUT_DIM[0] + h) * conv_p.OUT_DIM[1] + w) *
                     conv_p.OC +
                 k];
            int32_t actual = Cint32_fb
                [((n * conv_p.OUT_DIM[0] + h) * conv_p.OUT_DIM[1] + w) *
                     conv_p.OC +
                 k];
            EXPECT_EQ(expected, actual)
                << "Im2Col fused results differ at (" << n << ", " << h << ", "
                << w << ", " << k << ").";
          }
        }
      }
    }

  } // for each shape
} // Acc32Test


TEST(FBGemmIm2colTest, Acc16Test) {
  for (auto conv_p : shapes) {
    aligned_vector<uint8_t> Aint8(
        conv_p.MB * conv_p.IN_DIM[0] * conv_p.IN_DIM[1] * conv_p.IC, 0);
    aligned_vector<int8_t> Bint8(
        conv_p.K[0] * conv_p.K[1] * conv_p.IC * conv_p.OC, 0);
    aligned_vector<int32_t> Cint32_ref(
        conv_p.MB * conv_p.OUT_DIM[0] * conv_p.OUT_DIM[1] * conv_p.OC, 0.0f);
    aligned_vector<int32_t> Cint32_fb(
        conv_p.MB * conv_p.OUT_DIM[0] * conv_p.OUT_DIM[1] * conv_p.OC, 0);

    randFill(Aint8, 0, 5);
    int32_t Aint8_zero_point = 4;
    randFill(Bint8, -4, 4);

    conv_ref(
        conv_p,
        Aint8.data(),
        Aint8_zero_point,
        Bint8.data(),
        Cint32_ref.data());

    int NDim = conv_p.OC;
    int KDim = conv_p.K[0] * conv_p.K[1] * conv_p.IC;

    vector<int32_t> row_offset_buf;
    row_offset_buf.resize(
        PackAWithIm2Col<uint8_t, int16_t>::rowOffsetBufferSize());

    PackAWithIm2Col<uint8_t, int16_t> packA(
        conv_p, Aint8.data(), nullptr, Aint8_zero_point, row_offset_buf.data());

    PackBMatrix<int8_t, int16_t> packedB(
        matrix_op_t::NoTranspose, KDim, NDim, Bint8.data(), NDim);

    // no-op output process objects
    DoNothing<int32_t, int32_t> doNothing32BitObj;
    memCopy<> memcopyObj(doNothing32BitObj);

    fbgemmPacked(
        packA,
        packedB,
        Cint32_fb.data(),
        Cint32_fb.data(),
        NDim,
        memcopyObj,
        0,
        1);

    // correctness check
    for (int n = 0; n < conv_p.MB; ++n) {
      for (int h = 0; h < conv_p.OUT_DIM[0]; ++h) {
        for (int w = 0; w < conv_p.OUT_DIM[1]; ++w) {
          for (int k = 0; k < conv_p.OC; ++k) {
            int32_t expected = Cint32_ref
                [((n * conv_p.OUT_DIM[0] + h) * conv_p.OUT_DIM[1] + w) *
                     conv_p.OC +
                 k];
            int32_t actual = Cint32_fb
                [((n * conv_p.OUT_DIM[0] + h) * conv_p.OUT_DIM[1] + w) *
                     conv_p.OC +
                 k];
            EXPECT_EQ(expected, actual)
                << "Im2Col fused results differ at (" << n << ", " << h << ", "
                << w << ", " << k << ").";
          }
        }
      }
    }

  } // for each shape
} // Acc16Test

static vector<conv_param_t<3>> shapes_3d = {
    // MB, IC, OC, IT, IH, IW, G, KT, KH, KW, stride_t, stride_h, stride_w,
    // pad_t, pad_h, pad_w
    // conv_param_t<
    //     3>(1, 3, 64, {32, 112, 112}, 1, {3, 7, 7}, {1, 2, 2}, {1, 3, 3}),
    // conv_param_t<
    //     3>(1, 64, 64, {32, 56, 56}, 1, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}),
    // conv_param_t<
    //     3>(1, 64, 256, {32, 56, 56}, 1, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}),
    // conv_param_t<
    //     3>(1, 256, 64, {32, 56, 56}, 1, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}),
    // conv_param_t<
    //     3>(1, 256, 128, {32, 56, 56}, 1, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}),
    // conv_param_t<
    //     3>(1, 256, 512, {32, 56, 56}, 1, {1, 1, 1}, {2, 2, 2}, {0, 0, 0}),
    // conv_param_t<
    //     3>(1, 128, 512, {16, 28, 28}, 1, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}),
    // conv_param_t<
    //     3>(1, 512, 128, {16, 28, 28}, 1, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}),
    // conv_param_t<
    //     3>(1, 512, 256, {16, 28, 28}, 1, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}),
    // conv_param_t<
    //     3>(1, 512, 1024, {16, 28, 28}, 1, {1, 1, 1}, {2, 2, 2}, {0, 0, 0}),
    // conv_param_t<
    //     3>(1, 256, 1024, {8, 14, 14}, 1, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}),
    // conv_param_t<
    //     3>(1, 1024, 256, {8, 14, 14}, 1, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}),
    // conv_param_t<
    //     3>(1, 1024, 512, {8, 14, 14}, 1, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}),
    // conv_param_t<
    //     3>(1, 1024, 2048, {8, 14, 14}, 1, {1, 1, 1}, {2, 2, 2}, {0, 0, 0}),
    // conv_param_t<
    //     3>(1, 2048, 512, {8, 14, 14}, 1, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}),
    // conv_param_t<
    //     3>(1, 512, 2048, {4, 7, 7}, 1, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}),

    conv_param_t<
        3>(1, 3, 4, {32, 112, 112}, 1, {3, 7, 7}, {1, 2, 2}, {1, 3, 3}),
    conv_param_t<
        3>(1, 8, 16, {4, 7, 7}, 1, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}),
    conv_param_t<
        3>(1, 8, 16, {8, 14, 14}, 1, {1, 1, 1}, {2, 2, 2}, {0, 0, 0}),
};

TEST(FBGemmIm2colTest, 3DAcc32Test) {
  for (auto conv_p : shapes_3d) {
    aligned_vector<uint8_t> Aint8(
        conv_p.MB * conv_p.IN_DIM[0] * conv_p.IN_DIM[1] * conv_p.IN_DIM[2] *
            conv_p.IC,
        0);
    aligned_vector<int8_t> Bint8(
        conv_p.K[0] * conv_p.K[1] * conv_p.K[2] * conv_p.IC * conv_p.OC, 0);
    aligned_vector<int32_t> Cint32_ref(
        conv_p.MB * conv_p.OUT_DIM[0] * conv_p.OUT_DIM[1] * conv_p.OUT_DIM[2] *
            conv_p.OC,
        0.0f);
    aligned_vector<int32_t> Cint32_fb(
        conv_p.MB * conv_p.OUT_DIM[0] * conv_p.OUT_DIM[1] * conv_p.OUT_DIM[2] *
            conv_p.OC,
        0);

    randFill(Aint8, 0, 80);
    int32_t Aint8_zero_point = 43;
    randFill(Bint8, -16, 16);

    conv3d_ref(
        conv_p,
        Aint8.data(),
        Aint8_zero_point,
        Bint8.data(),
        Cint32_ref.data());

    int NDim = conv_p.OC;
    int KDim = conv_p.K[0] * conv_p.K[1] * conv_p.K[2] * conv_p.IC;

    vector<int32_t> row_offset_buf;
    row_offset_buf.resize(
        PackAWithIm2Col<uint8_t, int32_t, 3>::rowOffsetBufferSize());

    PackAWithIm2Col<uint8_t, int32_t, 3> packA(
        conv_p, Aint8.data(), nullptr, Aint8_zero_point, row_offset_buf.data());

    PackBMatrix<int8_t, int32_t> packedB(
        matrix_op_t::NoTranspose, KDim, NDim, Bint8.data(), NDim);

    // no-op output process objects
    DoNothing<int32_t, int32_t> doNothing32BitObj;
    memCopy<> memcopyObj(doNothing32BitObj);

    fbgemmPacked(
        packA,
        packedB,
        Cint32_fb.data(),
        Cint32_fb.data(),
        NDim,
        memcopyObj,
        0,
        1);

    // correctness check
    for (int n = 0; n < conv_p.MB; ++n) {
      for (int t = 0; t < conv_p.OUT_DIM[0]; ++t) {
        for (int h = 0; h < conv_p.OUT_DIM[1]; ++h) {
          for (int w = 0; w < conv_p.OUT_DIM[2]; ++w) {
            for (int k = 0; k < conv_p.OC; ++k) {
              int32_t expected = Cint32_ref
                  [(((n * conv_p.OUT_DIM[0] + t) * conv_p.OUT_DIM[1] + h) *
                        conv_p.OUT_DIM[2] +
                    w) *
                       conv_p.OC +
                   k];
              int32_t actual = Cint32_fb
                  [(((n * conv_p.OUT_DIM[0] + t) * conv_p.OUT_DIM[1] + h) *
                        conv_p.OUT_DIM[2] +
                    w) *
                       conv_p.OC +
                   k];
              EXPECT_EQ(expected, actual)
                  << "Im2Col fused results differ at (" << n << ", " << t
                  << ", " << h << ", " << w << ", " << k << ").";
            }
          }
        }
      }
    }
  } // for each shape
} // Acc32Test


TEST(FBGemmIm2colTest, 3DAcc16Test) {
  for (auto conv_p : shapes_3d) {
    aligned_vector<uint8_t> Aint8(
        conv_p.MB * conv_p.IN_DIM[0] * conv_p.IN_DIM[1] * conv_p.IN_DIM[2] *
            conv_p.IC,
        0);
    aligned_vector<int8_t> Bint8(
        conv_p.K[0] * conv_p.K[1] * conv_p.K[2] * conv_p.IC * conv_p.OC, 0);
    aligned_vector<int32_t> Cint32_ref(
        conv_p.MB * conv_p.OUT_DIM[0] * conv_p.OUT_DIM[1] * conv_p.OUT_DIM[2] *
            conv_p.OC,
        0.0f);
    aligned_vector<int32_t> Cint32_fb(
        conv_p.MB * conv_p.OUT_DIM[0] * conv_p.OUT_DIM[1] * conv_p.OUT_DIM[2] *
            conv_p.OC,
        0);

    randFill(Aint8, 0, 5);
    int32_t Aint8_zero_point = 4;
    randFill(Bint8, -4, 4);

    conv3d_ref(
        conv_p,
        Aint8.data(),
        Aint8_zero_point,
        Bint8.data(),
        Cint32_ref.data());

    int NDim = conv_p.OC;
    int KDim = conv_p.K[0] * conv_p.K[1] * conv_p.K[2] * conv_p.IC;

    vector<int32_t> row_offset_buf;
    row_offset_buf.resize(
        PackAWithIm2Col<uint8_t, int16_t, 3>::rowOffsetBufferSize());

    PackAWithIm2Col<uint8_t, int16_t, 3> packA(
        conv_p, Aint8.data(), nullptr, Aint8_zero_point, row_offset_buf.data());

    PackBMatrix<int8_t, int16_t> packedB(
        matrix_op_t::NoTranspose, KDim, NDim, Bint8.data(), NDim);

    // no-op output process objects
    DoNothing<int32_t, int32_t> doNothing32BitObj;
    memCopy<> memcopyObj(doNothing32BitObj);

    fbgemmPacked(
        packA,
        packedB,
        Cint32_fb.data(),
        Cint32_fb.data(),
        NDim,
        memcopyObj,
        0,
        1);

    // correctness check
    for (int n = 0; n < conv_p.MB; ++n) {
      for (int t = 0; t < conv_p.OUT_DIM[0]; ++t) {
        for (int h = 0; h < conv_p.OUT_DIM[1]; ++h) {
          for (int w = 0; w < conv_p.OUT_DIM[2]; ++w) {
            for (int k = 0; k < conv_p.OC; ++k) {
              int32_t expected = Cint32_ref
                  [(((n * conv_p.OUT_DIM[0] + t) * conv_p.OUT_DIM[1] + h) *
                        conv_p.OUT_DIM[2] +
                    w) *
                       conv_p.OC +
                   k];
              int32_t actual = Cint32_fb
                  [(((n * conv_p.OUT_DIM[0] + t) * conv_p.OUT_DIM[1] + h) *
                        conv_p.OUT_DIM[2] +
                    w) *
                       conv_p.OC +
                   k];
              EXPECT_EQ(expected, actual)
                  << "Im2Col fused results differ at (" << n << ", " << t
                  << ", " << h << ", " << w << ", " << k << ").";
            }
          }
        }
      }
    }
  } // for each shape
} // Acc16Test

} // namespace fbgemm2
