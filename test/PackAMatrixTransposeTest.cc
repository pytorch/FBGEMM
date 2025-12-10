/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <vector>
#include <iostream>
#include "fbgemm/Fbgemm.h"

using namespace fbgemm;

TEST(PackAMatrixTest, TransposeWarning) {
  int nRow = 10;
  int nCol = 10;
  std::vector<uint8_t> smat(nRow * nCol, 0);
  int ld = nCol;
  int groups = 1;

  // Capture stderr to verify the warning
  testing::internal::CaptureStderr();

  PackAMatrix<uint8_t, int32_t> packA(
      matrix_op_t::Transpose,
      nRow,
      nCol,
      smat.data(),
      ld,
      nullptr, // pmat
      groups);

  block_type_t block = {0, nRow, 0, nCol};
  packA.pack(block);

  std::string output = testing::internal::GetCapturedStderr();
  EXPECT_NE(output.find("Warning: PackAMatrix Transpose path is not optimized yet!"), std::string::npos);
  std::cout << "Captured stderr: " << output << std::endl;
}
