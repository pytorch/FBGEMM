/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "./TestUtils.h"
#include <gtest/gtest.h>
#include "fbgemm/Fbgemm.h"

namespace fbgemm {

template <typename T>
int compare_validate_buffers(
    const T* ref,
    const T* test,
    int m,
    int n,
    int ld,
    T atol) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      if (std::is_integral<T>::value) {
        EXPECT_EQ(test[i * ld + j], ref[i * ld + j])
            << "GEMM results differ at (" << i << ", " << j
            << ") reference: " << (int64_t)ref[i * ld + j]
            << ", FBGEMM: " << (int64_t)test[i * ld + j];
      } else {
        EXPECT_LE(std::abs(ref[i * ld + j] - test[i * ld + j]), atol)
            << "GEMM results differ at (" << i << ", " << j
            << ") reference: " << ref[i * ld + j]
            << ", FBGEMM: " << test[i * ld + j];
      }
    }
  }
  return 0;
}

template int compare_validate_buffers<float>(
    const float* ref,
    const float* test,
    int m,
    int n,
    int ld,
    float atol);

template int compare_validate_buffers<int32_t>(
    const int32_t* ref,
    const int32_t* test,
    int m,
    int n,
    int ld,
    int32_t atol);

template int compare_validate_buffers<uint8_t>(
    const uint8_t* ref,
    const uint8_t* test,
    int m,
    int n,
    int ld,
    uint8_t atol);

template int compare_validate_buffers<int64_t>(
    const int64_t* ref,
    const int64_t* test,
    int m,
    int n,
    int ld,
    int64_t atol);

template <typename T>
bool check_all_zero_entries(const T* test, int m, int n) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      if (test[i * n + j] != 0)
        return true;
    }
  }
  return false;
}

template bool check_all_zero_entries<float>(const float* test, int m, int n);
template bool
check_all_zero_entries<int32_t>(const int32_t* test, int m, int n);
template bool
check_all_zero_entries<uint8_t>(const uint8_t* test, int m, int n);

// atol: absolute tolerance. <=0 means do not consider atol.
// rtol: relative tolerance. <=0 means do not consider rtol.
template <>
::testing::AssertionResult floatCloseAll<float, float>(
    const std::vector<float>& a,
    const std::vector<float>& b,
    const float atol,
    const float rtol) {
  std::stringstream ss;
  bool match = true;
  if (a.size() != b.size()) {
    ss << " size mismatch ";
    match = false;
  }
  if (!match) {
    return ::testing::AssertionFailure()
        << " results do not match. " << ss.str();
  }
  for (size_t i = 0; i < a.size(); i++) {
    const bool consider_absDiff = atol > 0;
    const bool consider_relDiff = rtol > 0 &&
        std::fabs(a[i]) > std::numeric_limits<float>::epsilon() &&
        std::fabs(b[i]) > std::numeric_limits<float>::epsilon();

    const float absDiff = std::fabs(a[i] - b[i]);
    const float relDiff = absDiff / std::fabs(a[i]);

    if (consider_absDiff && consider_relDiff) {
      match = absDiff <= atol || relDiff <= rtol;
    } else if (consider_absDiff) {
      match = absDiff <= atol;
    } else if (consider_relDiff) {
      match = relDiff <= rtol;
    }
    if (!match) {
      ss << " mismatch at (" << i << ") " << std::endl;
      ss << "\t  ref: " << a[i] << " test: " << b[i] << std::endl;
      if (consider_absDiff) {
        ss << "\t absolute diff: " << absDiff << " > " << atol << std::endl;
      }
      if (consider_relDiff) {
        ss << "\t relative diff: " << relDiff << " > " << rtol << std::endl;
      }
      return ::testing::AssertionFailure()
          << " results do not match. " << ss.str();
    }
  }
  return ::testing::AssertionSuccess();
}

template <>
::testing::AssertionResult floatCloseAll<float, float16>(
    const std::vector<float>& a,
    const std::vector<float16>& b,
    const float atol,
    const float rtol) {
  std::vector<float> b_float(b.size());
  const auto transform = [](float16 input) { return cpu_half2float(input); };
  std::transform(b.begin(), b.end(), b_float.begin(), transform);
  return floatCloseAll(a, b_float, atol, rtol);
}

template <>
::testing::AssertionResult floatCloseAll<float16, float16>(
    const std::vector<float16>& a,
    const std::vector<float16>& b,
    const float atol,
    const float rtol) {
  std::vector<float> a_float(a.size());
  std::vector<float> b_float(b.size());
  const auto transform = [](float16 input) { return cpu_half2float(input); };
  std::transform(a.begin(), a.end(), a_float.begin(), transform);
  std::transform(b.begin(), b.end(), b_float.begin(), transform);
  return floatCloseAll(a_float, b_float, atol, rtol);
}
} // namespace fbgemm
