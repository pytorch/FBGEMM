/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once
#include <vector>
#include "bench/AlignedVec.h"

namespace fbgemm {

template <typename T>
void randFill(aligned_vector<T>& vec, T low, T high);

void llc_flush(std::vector<char>& llc);

int fbgemm_get_num_threads();
int fbgemm_get_thread_num();

} // namespace fbgemm
