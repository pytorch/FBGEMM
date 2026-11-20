/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <ios>
#include <random>
#include <span>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "fbgemm/Assert.h"
#include "fbgemm/Types.h"
#include "src/PlatformSpecificKernels.h" // @manual

namespace fbgemm {

inline uint32_t bits(float x) {
  return std::bit_cast<uint32_t>(x);
}

inline uint32_t bits(bfloat16 x) {
  return x.val;
}

inline uint32_t bits(float16 x) {
  return x.val;
}

std::string hexBits(auto x) {
  std::ostringstream ss;
  ss << "0x" << std::hex << bits(x);
  return ss.str();
}

void checkSame(
    const auto& in,
    const auto& got,
    const auto& expected,
    const auto&... what) {
  const auto gotBytes = std::as_bytes(std::span(got));
  const auto mismatch =
      std::ranges::mismatch(gotBytes, std::as_bytes(std::span(expected)));
  if (mismatch.in1 == gotBytes.end()) {
    return;
  }
  const size_t i = (mismatch.in1 - gotBytes.begin()) / sizeof(got[0]);
  FBGEMM_CHECK(
      false,
      what...,
      ": mismatch at index ",
      i,
      " of ",
      got.size(),
      ": input ",
      hexBits(in[i]),
      ", got ",
      hexBits(got[i]),
      ", expected ",
      hexBits(expected[i]));
}

template <typename In, typename Out>
void runAllKernels(
    std::span<const PlatformSpecificKernels* const> kernels,
    const auto& convert,
    std::span<const In> in,
    std::span<Out> out) {
  FBGEMM_CHECK(
      in.size() == out.size(),
      "size mismatch: ",
      in.size(),
      " vs ",
      out.size());
  const auto run = [&](size_t k) {
    std::ranges::fill(
        std::as_writable_bytes(out), static_cast<std::byte>(k + 1));
    convert(*kernels[k], in.data(), out.data(), in.size());
  };
  run(0);
  const std::vector<Out> expected(out.begin(), out.end());
  for (size_t k = 1; k != kernels.size(); ++k) {
    run(k);
    checkSame(
        in, out, expected, kernels[k]->name(), " vs ", kernels[0]->name());
  }
}

class RandomSubspanGenerator {
 public:
  size_t size(size_t len) {
    constexpr size_t kMaxSize = 4096;
    return std::uniform_int_distribution<size_t>{
        0, std::min(len, kMaxSize)}(gen_);
  }

  auto subspan(auto whole, size_t size) {
    return whole.subspan(
        std::uniform_int_distribution<size_t>{0, whole.size() - size}(gen_),
        size);
  }

 private:
  std::mt19937 gen_{0};
};

void forEachTestSubspan(const auto& check, auto in, auto... outs) {
  check(in, outs...);

  for (size_t offset = 0; offset != 10; ++offset) {
    for (size_t size : {0, 1, 15, 30, 64}) {
      check(in.subspan(offset, size), outs.subspan(offset, size)...);
    }
  }

  RandomSubspanGenerator gen;
  for (size_t i = 0; i != 10000; ++i) {
    const size_t size = gen.size(in.size());
    std::apply(
        check, std::tuple{gen.subspan(in, size), gen.subspan(outs, size)...});
  }
}

} // namespace fbgemm
