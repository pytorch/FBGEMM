/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "fbgemm/FbgemmFPCommon.h"
#include "fbgemm/Fbgemm.h"

#include <cpuinfo.h>
#include <array>
#include <cmath>
#include <utility>

namespace fbgemm {

/// class that performs packing of matrix in
/// row-major or col-major format into
/// internal packed blocked-row major format

/// Todo: make it fast with AVX2 transpose
void PackA(int nrow, int ncol, const float* from, int ldim, float* to) {
  // for (int r = 0; r < nrow; ++r) {
  //   for (int c = 0; c < ncol; ++c) {
  //     to[r + c * nrow] = from[r * ldim + c];
  //   }
  // }
  transpose_simd(nrow, ncol, from, ldim, to, nrow);
}

// Each kernel does the following computation that multiplies
// mb x k A sub-matrix with k x b_block_cols*64 B sub-matrix
// for (int j = 0; j < b_block_cols * 64; j += 64) {
//   for (int kk = 0; kk < k; ++k) {
//     for (int i = 0; i < mb; ++i) {
//       c[i][j:j+64] += a[i][kk] * b[kk][j:j+64]
//     }
//   }
// }

// autotuned kernel splits for various cases m = 1:mb_max
// may need re-autotuning for new uarch
// clang-format off
partition_array_t partition_avx2 = {
  // NOTE: clang-format wants to use a different formatting but the current
  // formatting should be easier to read.
  {
    {{ { 0, 0 }, { 0, 0 } } }, // 0
    {{ { 1, 1 }, { 0, 0 } } }, // 1
    {{ { 2, 1 }, { 0, 0 } } }, // 2
    {{ { 3, 1 }, { 0, 0 } } }, // 3
    {{ { 4, 1 }, { 0, 0 } } }, // 4
    {{ { 5, 1 }, { 0, 0 } } }, // 5
    {{ { 6, 1 }, { 0, 0 } } }, // 6
    {{ { 5, 1 }, { 2, 1 } } }, // 7
    {{ { 4, 2 }, { 0, 0 } } }, // 8
    {{ { 5, 1 }, { 4, 1 } } }, // 9
    {{ { 5, 2 }, { 0, 0 } } }, // 10
    {{ { 6, 1 }, { 5, 1 } } }, // 11
    {{ { 6, 2 }, { 0, 0 } } }, // 12
    {{ { 5, 2 }, { 3, 1 } } }, // 13
    {{ { 6, 2 }, { 2, 1 } } }, // 14
    {{ { 5, 3 }, { 0, 0 } } }, // 15
    {{ { 6, 2 }, { 4, 1 } } }, // 16
    {{ { 6, 2 }, { 5, 1 } } }, // 17
    {{ { 6, 3 }, { 0, 0 } } }, // 18
    {{ { 5, 3 }, { 4, 1 } } }, // 19
    {{ { 5, 4 }, { 0, 0 } } }, // 20
    {{ { 5, 3 }, { 6, 1 } } }, // 21
    {{ { 6, 3 }, { 4, 1 } } }, // 22
    {{ { 6, 3 }, { 5, 1 } } }, // 23
    {{ { 6, 4 }, { 0, 0 } } }, // 24
    {{ { 5, 5 }, { 0, 0 } } }, // 25
    {{ { 5, 4 }, { 6, 1 } } }, // 26
    {{ { 6, 4 }, { 3, 1 } } }, // 27
    {{ { 6, 4 }, { 4, 1 } } }, // 28
    {{ { 6, 4 }, { 5, 1 } } }, // 29
    {{ { 6, 5 }, { 0, 0 } } }, // 30
    {{ { 6, 5 }, { 1, 1 } } }, // 31
    {{ { 6, 5 }, { 2, 1 } } }, // 32
    {{ { 6, 5 }, { 3, 1 } } }, // 33
    {{ { 6, 5 }, { 4, 1 } } }, // 34
    {{ { 6, 5 }, { 5, 1 } } }, // 35
    {{ { 6, 6 }, { 0, 0 } } }, // 36
    {{ { 6, 6 }, { 1, 1 } } }, // 37
    {{ { 6, 6 }, { 2, 1 } } }, // 38
    {{ { 6, 6 }, { 3, 1 } } }, // 39
    {{ { 6, 6 }, { 4, 1 } } }, // 40
    {{ { 6, 6 }, { 5, 1 } } }, // 41
    {{ { 6, 7 }, { 0, 0 } } }, // 42
    {{ { 6, 7 }, { 1, 1 } } }, // 43
    {{ { 6, 7 }, { 2, 1 } } }, // 44
    {{ { 6, 7 }, { 3, 1 } } }, // 45
    {{ { 6, 7 }, { 4, 1 } } }, // 46
    {{ { 6, 7 }, { 5, 1 } } }, // 47
    {{ { 6, 8 }, { 0, 0 } } }, // 48
    {{ { 6, 8 }, { 1, 1 } } }, // 49
    {{ { 6, 8 }, { 2, 1 } } }, // 50
    {{ { 6, 8 }, { 3, 1 } } }, // 51
    {{ { 6, 8 }, { 4, 1 } } }, // 52
    {{ { 6, 8 }, { 5, 1 } } }, // 53
    {{ { 6, 9 }, { 0, 0 } } }, // 54
    {{ { 6, 9 }, { 1, 1 } } }, // 55
    {{ { 6, 9 }, { 2, 1 } } }, // 56
    {{ { 6, 9 }, { 3, 1 } } }, // 57
    {{ { 6, 9 }, { 4, 1 } } }, // 58
    {{ { 6, 9 }, { 5, 1 } } }, // 59
    {{ { 6, 10 }, { 0, 0 } } }, // 60
    {{ { 6, 10 }, { 1, 1 } } }, // 61
    {{ { 6, 10 }, { 2, 1 } } }, // 62
    {{ { 6, 10 }, { 3, 1 } } }, // 63
    {{ { 6, 10 }, { 4, 1 } } }, // 64
    {{ { 6, 10 }, { 5, 1 } } }, // 65
    {{ { 6, 11 }, { 0, 0 } } }, // 66
    {{ { 6, 11 }, { 1, 1 } } }, // 67
    {{ { 6, 11 }, { 2, 1 } } }, // 68
    {{ { 6, 11 }, { 3, 1 } } }, // 69
    {{ { 6, 11 }, { 4, 1 } } }, // 70
    {{ { 6, 11 }, { 5, 1 } } }, // 71
    {{ { 6, 12 }, { 0, 0 } } }, // 72
    {{ { 6, 12 }, { 1, 1 } } }, // 73
    {{ { 6, 12 }, { 2, 1 } } }, // 74
    {{ { 6, 12 }, { 3, 1 } } }, // 75
    {{ { 6, 12 }, { 4, 1 } } }, // 76
    {{ { 6, 12 }, { 5, 1 } } }, // 77
    {{ { 6, 13 }, { 0, 0 } } }, // 78
    {{ { 6, 13 }, { 1, 1 } } }, // 79
    {{ { 6, 13 }, { 2, 1 } } }, // 80
    {{ { 6, 13 }, { 3, 1 } } }, // 81
    {{ { 6, 13 }, { 4, 1 } } }, // 82
    {{ { 6, 13 }, { 5, 1 } } }, // 83
    {{ { 6, 14 }, { 0, 0 } } }, // 84
    {{ { 6, 14 }, { 1, 1 } } }, // 85
    {{ { 6, 14 }, { 2, 1 } } }, // 86
    {{ { 6, 14 }, { 3, 1 } } }, // 87
    {{ { 6, 14 }, { 4, 1 } } }, // 88
    {{ { 6, 14 }, { 5, 1 } } }, // 89
    {{ { 6, 15 }, { 0, 0 } } }, // 90
    {{ { 6, 15 }, { 1, 1 } } }, // 91
    {{ { 6, 15 }, { 2, 1 } } }, // 92
    {{ { 6, 15 }, { 3, 1 } } }, // 93
    {{ { 6, 15 }, { 4, 1 } } }, // 94
    {{ { 6, 15 }, { 5, 1 } } }, // 95
    {{ { 6, 16 }, { 0, 0 } } }, // 96
    {{ { 6, 16 }, { 1, 1 } } }, // 97
    {{ { 6, 16 }, { 2, 1 } } }, // 98
    {{ { 6, 16 }, { 3, 1 } } }, // 99
    {{ { 6, 16 }, { 4, 1 } } }, // 100
    {{ { 6, 16 }, { 5, 1 } } }, // 101
    {{ { 6, 17 }, { 0, 0 } } }, // 102
    {{ { 6, 17 }, { 1, 1 } } }, // 103
    {{ { 6, 17 }, { 2, 1 } } }, // 104
    {{ { 6, 17 }, { 3, 1 } } }, // 105
    {{ { 6, 17 }, { 4, 1 } } }, // 106
    {{ { 6, 17 }, { 5, 1 } } }, // 107
    {{ { 6, 18 }, { 0, 0 } } }, // 108
    {{ { 6, 18 }, { 1, 1 } } }, // 109
    {{ { 6, 18 }, { 2, 1 } } }, // 110
    {{ { 6, 18 }, { 3, 1 } } }, // 111
    {{ { 6, 18 }, { 4, 1 } } }, // 112
    {{ { 6, 18 }, { 5, 1 } } }, // 113
    {{ { 6, 19 }, { 0, 0 } } }, // 114
    {{ { 6, 19 }, { 1, 1 } } }, // 115
    {{ { 6, 19 }, { 2, 1 } } }, // 116
    {{ { 6, 19 }, { 3, 1 } } }, // 117
    {{ { 6, 19 }, { 4, 1 } } }, // 118
    {{ { 6, 19 }, { 5, 1 } } }, // 119
    {{ { 6, 20 }, { 0, 0 } } }, // 120
  }
};

partition_array_t partition_avx512 = {
  // NOTE: clang-format wants to use a different formatting but the current
  // formatting should be easier to read.
  {
    {{ { 0, 0 }, { 0, 0 } } }, // 0
    {{ { 1, 1 }, { 0, 0 } } }, // 1
    {{ { 2, 1 }, { 0, 0 } } }, // 2
    {{ { 3, 1 }, { 0, 0 } } }, // 3
    {{ { 4, 1 }, { 0, 0 } } }, // 4
    {{ { 5, 1 }, { 0, 0 } } }, // 5
    {{ { 6, 1 }, { 0, 0 } } }, // 6
    {{ { 7, 1 }, { 0, 0 } } }, // 7
    {{ { 8, 1 }, { 0, 0 } } }, // 8
    {{ { 9, 1 }, { 0, 0 } } }, // 9
    {{ { 10, 1 }, { 0, 0 } } }, // 10
    {{ { 11, 1 }, { 0, 0 } } }, // 11
    {{ { 12, 1 }, { 0, 0 } } }, // 12
    {{ { 13, 1 }, { 0, 0 } } }, // 13
    {{ { 14, 1 }, { 0, 0 } } }, // 14
    {{ { 8, 1 }, { 7, 1 } } }, // 15
    {{ { 8, 2 }, { 0, 0 } } }, // 16
    {{ { 9, 1 }, { 8, 1 } } }, // 17
    {{ { 9, 2 }, { 0, 0 } } }, // 18
    {{ { 10, 1 }, { 9, 1 } } }, // 19
    {{ { 10, 2 }, { 0, 0 } } }, // 20
    {{ { 11, 1 }, { 10, 1 } } }, // 21
    {{ { 11, 2 }, { 0, 0 } } }, // 22
    {{ { 12, 1 }, { 11, 1 } } }, // 23
    {{ { 12, 2 }, { 0, 0 } } }, // 24
    {{ { 13, 1 }, { 12, 1 } } }, // 25
    {{ { 13, 2 }, { 0, 0 } } }, // 26
    {{ { 14, 1 }, { 13, 1 } } }, // 27
    {{ { 14, 2 }, { 0, 0 } } }, // 28
    {{ { 10, 2 }, { 9, 1 } } }, // 29
    {{ { 10, 3 }, { 0, 0 } } }, // 30
    {{ { 11, 2 }, { 9, 1 } } }, // 31
    {{ { 11, 2 }, { 10, 1 } } }, // 32
    {{ { 11, 3 }, { 0, 0 } } }, // 33
    {{ { 12, 2 }, { 10, 1 } } }, // 34
    {{ { 12, 2 }, { 11, 1 } } }, // 35
    {{ { 12, 3 }, { 0, 0 } } }, // 36
    {{ { 13, 2 }, { 11, 1 } } }, // 37
    {{ { 13, 2 }, { 12, 1 } } }, // 38
    {{ { 13, 3 }, { 0, 0 } } }, // 39
    {{ { 14, 2 }, { 12, 1 } } }, // 40
    {{ { 14, 2 }, { 13, 1 } } }, // 41
    {{ { 14, 3 }, { 0, 0 } } }, // 42
    {{ { 11, 3 }, { 10, 1 } } }, // 43
    {{ { 11, 4 }, { 0, 0 } } }, // 44
    {{ { 12, 3 }, { 9, 1 } } }, // 45
    {{ { 12, 3 }, { 10, 1 } } }, // 46
    {{ { 12, 3 }, { 11, 1 } } }, // 47
    {{ { 12, 4 }, { 0, 0 } } }, // 48
    {{ { 13, 3 }, { 10, 1 } } }, // 49
    {{ { 13, 3 }, { 11, 1 } } }, // 50
    {{ { 13, 3 }, { 12, 1 } } }, // 51
    {{ { 13, 4 }, { 0, 0 } } }, // 52
    {{ { 14, 3 }, { 11, 1 } } }, // 53
    {{ { 14, 3 }, { 12, 1 } } }, // 54
    {{ { 14, 3 }, { 13, 1 } } }, // 55
    {{ { 14, 4 }, { 0, 0 } } }, // 56
    {{ { 12, 4 }, { 9, 1 } } }, // 57
    {{ { 12, 4 }, { 10, 1 } } }, // 58
    {{ { 12, 4 }, { 11, 1 } } }, // 59
    {{ { 12, 5 }, { 0, 0 } } }, // 60
    {{ { 13, 4 }, { 9, 1 } } }, // 61
    {{ { 13, 4 }, { 10, 1 } } }, // 62
    {{ { 13, 4 }, { 11, 1 } } }, // 63
    {{ { 13, 4 }, { 12, 1 } } }, // 64
    {{ { 13, 5 }, { 0, 0 } } }, // 65
    {{ { 14, 4 }, { 10, 1 } } }, // 66
    {{ { 14, 4 }, { 11, 1 } } }, // 67
    {{ { 14, 4 }, { 12, 1 } } }, // 68
    {{ { 14, 4 }, { 13, 1 } } }, // 69
    {{ { 14, 5 }, { 0, 0 } } }, // 70
    {{ { 12, 5 }, { 11, 1 } } }, // 71
    {{ { 12, 6 }, { 0, 0 } } }, // 72
    {{ { 13, 5 }, { 8, 1 } } }, // 73
    {{ { 13, 5 }, { 9, 1 } } }, // 74
    {{ { 13, 5 }, { 10, 1 } } }, // 75
    {{ { 13, 5 }, { 11, 1 } } }, // 76
    {{ { 13, 5 }, { 12, 1 } } }, // 77
    {{ { 13, 6 }, { 0, 0 } } }, // 78
    {{ { 14, 5 }, { 9, 1 } } }, // 79
    {{ { 14, 5 }, { 10, 1 } } }, // 80
    {{ { 14, 5 }, { 11, 1 } } }, // 81
    {{ { 14, 5 }, { 12, 1 } } }, // 82
    {{ { 14, 5 }, { 13, 1 } } }, // 83
    {{ { 14, 6 }, { 0, 0 } } }, // 84
    {{ { 13, 6 }, { 7, 1 } } }, // 85
    {{ { 13, 6 }, { 8, 1 } } }, // 86
    {{ { 13, 6 }, { 9, 1 } } }, // 87
    {{ { 13, 6 }, { 10, 1 } } }, // 88
    {{ { 13, 6 }, { 11, 1 } } }, // 89
    {{ { 13, 6 }, { 12, 1 } } }, // 90
    {{ { 13, 7 }, { 0, 0 } } }, // 91
    {{ { 14, 6 }, { 8, 1 } } }, // 92
    {{ { 14, 6 }, { 9, 1 } } }, // 93
    {{ { 14, 6 }, { 10, 1 } } }, // 94
    {{ { 14, 6 }, { 11, 1 } } }, // 95
    {{ { 14, 6 }, { 12, 1 } } }, // 96
    {{ { 14, 6 }, { 13, 1 } } }, // 97
    {{ { 14, 7 }, { 0, 0 } } }, // 98
    {{ { 13, 7 }, { 8, 1 } } }, // 99
    {{ { 13, 7 }, { 9, 1 } } }, // 100
    {{ { 13, 7 }, { 10, 1 } } }, // 101
    {{ { 13, 7 }, { 11, 1 } } }, // 102
    {{ { 13, 7 }, { 12, 1 } } }, // 103
    {{ { 13, 8 }, { 0, 0 } } }, // 104
    {{ { 14, 7 }, { 7, 1 } } }, // 105
    {{ { 14, 7 }, { 8, 1 } } }, // 106
    {{ { 14, 7 }, { 9, 1 } } }, // 107
    {{ { 14, 7 }, { 10, 1 } } }, // 108
    {{ { 14, 7 }, { 11, 1 } } }, // 109
    {{ { 14, 7 }, { 12, 1 } } }, // 110
    {{ { 14, 7 }, { 13, 1 } } }, // 111
    {{ { 14, 8 }, { 0, 0 } } }, // 112
    {{ { 13, 8 }, { 9, 1 } } }, // 113
    {{ { 13, 8 }, { 10, 1 } } }, // 114
    {{ { 13, 8 }, { 11, 1 } } }, // 115
    {{ { 13, 8 }, { 12, 1 } } }, // 116
    {{ { 13, 9 }, { 0, 0 } } }, // 117
    {{ { 14, 8 }, { 6, 1 } } }, // 118
    {{ { 14, 8 }, { 7, 1 } } }, // 119
    {{ { 14, 8 }, { 8, 1 } } }, // 120
  }
};

}
