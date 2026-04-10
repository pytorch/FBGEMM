/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <type_traits>
#include "tbe/eeg/eeg_utils.h" // @manual

using namespace fbgemm_gpu::tbe;

TEST(SpecFuncTest, TestZipfianNormalization) {
  static constexpr double atol = 1e-9;
  EXPECT_NEAR(
      getZipfianConstant(2.0, 1.0),
      std::numbers::pi * std::numbers::pi / 6.0,
      atol);
  EXPECT_NEAR(
      getZipfianConstant(0.5, 1.0, 100), 18.589603824784153422358163109, atol);
  EXPECT_NEAR(
      getZipfianConstant(0.75, 1.0, 1000),
      19.055178975831392013112531682843,
      atol);
  EXPECT_NEAR(
      getZipfianConstant(1.15, 5.5, 10000),
      3.560807540800668684934073384599772,
      atol);
  EXPECT_NEAR(
      getZipfianConstant(1.0, 1.0, 10000),
      9.787606036044382264178477904851605,
      atol);
  EXPECT_NEAR(
      getZipfianConstant(1.0, 2.2, 10000000),
      15.573802384217160717007977478400428,
      atol);
  EXPECT_NEAR(
      getZipfianConstant(1.0 + 1.0e-9, 1.0, 1000),
      7.4854708367611659306042150150989584,
      atol);
}

// Helper function to compute KL divergence D(p||q) between two (not necessarily
// normalized) frequency distribution. This may be used to test if RNG is
// actually matching a desired distribution or not.
template <typename T, typename U>
static double klDivergence(const std::vector<T>& p, const std::vector<U>& q) {
  static_assert(std::is_arithmetic_v<T>, "Not an arithmetic type");
  static_assert(std::is_arithmetic_v<T>, "Not an arithmetic type");
  assert((p.size() == q.size()) && "Input vectors have unequal length!");

  std::vector<double> pDist(std::begin(p), std::end(p));
  std::vector<double> qDist(std::begin(q), std::end(q));

  double pNorm = std::reduce(std::begin(p), std::end(p));
  double qNorm = std::reduce(std::begin(q), std::end(q));
  std::for_each(
      std::begin(pDist), std::end(pDist), [=](double& freq) { freq /= pNorm; });
  std::for_each(
      std::begin(qDist), std::end(qDist), [=](double& freq) { freq /= qNorm; });

  double kl = 0.0;
  for (int i = 0; i < p.size(); ++i) {
    if (pDist[i] == 0.0) {
      continue;
    }
    kl += pDist[i] * std::log2(pDist[i] / qDist[i]);
  }
  return kl;
}

TEST(ZipfianDistTest, TestSamples) {
  static constexpr int kNumTrials = 100000;
  static const double sVals[] = {0.0, 0.5, 1.0, 1.0000000000000006661, 1.5};
  static const double qVals[] = {1.0, 1.5, 55.0};
  static const double nVals[] = {10, 100, 100000};
  auto rng = absl::BitGen();

  // The KL divergance declines as c * n * log(kNumTrials)/kNumTrials
  // asymptotically, where c should be a small constant. Ref: Sanov's theorem
  // (https://en.wikipedia.org/wiki/Sanov%27s_theorem)
  static constexpr double rtol = 2;

  for (auto n : nVals) {
    std::vector<double> freqs(n);
    std::vector<double> targetFreqs(n);

    for (auto q : qVals) {
      for (auto s : sVals) {
        // Setup target distribution
        std::fill(std::begin(freqs), std::end(freqs), 0.0);
        for (int i = 0; i < n; ++i) {
          targetFreqs[i] = std::pow(i + q, -s);
        }

        // Generate samples and sanity check
        auto dist = ZipfianDistribution(s, q, n);
        for (int i = 0; i < kNumTrials; ++i) {
          auto sample = dist(rng);
          ASSERT_GE(sample, 0);
          ASSERT_LE(sample, n - 1);
          freqs[sample] += 1;
        }

        // Perform KL divergence test
        auto kl = klDivergence(freqs, targetFreqs);
        ASSERT_LE(kl, rtol * n * std::log(kNumTrials) / kNumTrials);
      }
    }
  }
}
