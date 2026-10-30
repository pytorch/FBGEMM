/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string_view>
#include <vector>

#include "fbgemm/Fbgemm.h"
#include "fbgemm/FbgemmConvert.h"

namespace fbgemm {
namespace {

template <typename InType>
void runBenchmark(std::string_view type_name) {
  constexpr int64_t kRows = 1024;
  constexpr int64_t kBatchSize = 64;
  constexpr int64_t kIndicesPerBag = 4;
  constexpr int64_t kIterations = 10000;
  std::vector<InType> input(kRows);
  for (int64_t i = 0; i < kRows; ++i) {
    input[i] = from_float<InType>(static_cast<float>(i % 17) - 8.0f);
  }
  std::vector<int32_t> indices(kBatchSize * kIndicesPerBag);
  for (size_t i = 0; i < indices.size(); ++i) {
    indices[i] = static_cast<int32_t>((i * 13) % kRows);
  }
  std::vector<int32_t> offsets(kBatchSize + 1);
  for (int64_t i = 0; i <= kBatchSize; ++i) {
    offsets[i] = static_cast<int32_t>(i * kIndicesPerBag);
  }
  std::vector<float> output(kBatchSize);
  auto kernel = GenerateEmbeddingSpMDM<InType, int32_t, int32_t, float>(
      1, false, false, 16, false, true);

  for (int i = 0; i < 10; ++i) {
    if (!kernel(
            kBatchSize,
            indices.size(),
            kRows,
            input.data(),
            indices.data(),
            offsets.data(),
            nullptr,
            output.data())) {
      std::abort();
    }
  }
  const auto start = std::chrono::steady_clock::now();
  for (int64_t i = 0; i < kIterations; ++i) {
    if (!kernel(
            kBatchSize,
            indices.size(),
            kRows,
            input.data(),
            indices.data(),
            offsets.data(),
            nullptr,
            output.data())) {
      std::abort();
    }
  }
  const auto elapsed = std::chrono::duration<double, std::nano>(
      std::chrono::steady_clock::now() - start);
  std::cout << type_name << ": " << elapsed.count() / kIterations
            << " ns/call; checksum=" << output[0] << '\n';
}

} // namespace
} // namespace fbgemm

int main() {
  fbgemm::runBenchmark<fbgemm::float16>("FP16");
  fbgemm::runBenchmark<fbgemm::bfloat16>("BF16");
  return 0;
}
