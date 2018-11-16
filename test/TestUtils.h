/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once
#include <cmath>
#include <vector>

namespace fbgemm {

/*
 * @brief Check and validate the buffers for reference and FBGEMM result.
 */
template <typename T>
int compare_validate_buffers(
    const T* ref,
    const T* test,
    int m,
    int n,
    int ld,
    T atol);

/*
 * @brief Check if all entries are zero or not.
 * If any entry is non-zero, return True;
 * otherwise, return False.
 */
template <typename T>
bool check_all_zero_entries(const T* test, int m, int n);

/*
 * @brief In-place transposition for nxk matrix ref.
 * @params n number of rows in input (number of columns in output)
 * @params k number of columns in input (number of rows in output)
 */
template <typename T>
void transpose_matrix(T* ref, int n, int k);

/*
 * @brief Out-of-place transposition for M*N matrix ref.
 * @params M number of rows in input
 * @params K number of columns in input
 */
template <typename T>
void transpose_matrix(
    int M,
    int N,
    const T* src,
    int ld_src,
    T* dst,
    int ld_dst);

} // namespace fbgemm
