/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "BenchUtils.h"
#include <random>

namespace fbgemm2 {

std::default_random_engine eng;

template <typename T>
void randFill(aligned_vector<T> &vec, const int low, const int high) {
  std::random_device r;
  std::uniform_int_distribution<int> dis(low, high);
  for (auto &v : vec) {
    v = static_cast<T>(dis(eng));
  }
}

template
void randFill<float>(aligned_vector<float> &vec,
                     const int low, const int high);
template
void randFill<uint8_t>(aligned_vector<uint8_t> &vec,
                       const int low, const int high);
template
void randFill<int8_t>(aligned_vector<int8_t> &vec,
                      const int low, const int high);

template
void randFill<int>(aligned_vector<int> &vec,
                   const int low, const int high);

void llc_flush(std::vector<char>& llc) {
  volatile char* data = llc.data();
  for (int i = 0; i < llc.size(); i++) {
    data[i]++;
  }
}

} // namespace fbgemm2
