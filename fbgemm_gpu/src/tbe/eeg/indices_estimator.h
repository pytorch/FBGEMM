/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <torch/torch.h>
#include <filesystem>
#include <vector>

#ifdef FBGEMM_USE_FOLLY
#include <folly/container/F14Map.h>
#else
#include <map>
#endif

#include "eeg_models.h"

namespace fbgemm_gpu::tbe {

class IndicesEstimator {
 public:
  explicit IndicesEstimator(const std::filesystem::path& tensorPath);

  explicit IndicesEstimator(const torch::Tensor& indices);

  // maximum likelihood estimate of the heavy hitters + Zipf parameters
  // Returns std::nullopt if there is no index data
  std::optional<IndicesDistributionParameters> estimate();

  // Quality of the fit, measured by KL divergence to a set of parameters. Lower
  // score is better, with best possible score of 0.0.
  double getEstimateQuality(const IndicesDistributionParameters& params);

 private:
  // Hardcoded parameters
  // Heavy hitter threshold: we make sure they account for >=90% of the mass or
  // the top 20 frequencies, whichever comes first. We also ensure that we cover
  // at least the top 5 entries.
  static constexpr double kHeavyHitterThreshold_ = 0.90;
  static constexpr int kHeavyHitterMaxEntries_ = 20;

  // Regularizer term on the "q" parameter of zipf to prevent overfit
  static constexpr double kQRegularizer_ = 0.0001;
  // Minimum, maximum, and step size of s to consider
  static constexpr double kMinS_ = 0.01;
  static constexpr double kMaxS_ = 2.0;
  static constexpr double kSStep_ = 0.01;

  // log table for caching and quickly computing maximum likelihood
  // Log table stores log(1), log(1 + 1/level), log(1 + 2/level), ...,
  // log(maxIndex+1)
  static constexpr auto kLevels_ = 8;
  static constexpr int kQSweepGranularity_ =
      4; // controls how many q we sweep over
  static_assert(
      kLevels_ % kQSweepGranularity_ == 0,
      "kLevels_ must be a multiple of kQSweepGranularity_");
  static constexpr auto kMaxQ = 100000; // Maximum q over which we sweep over
  std::vector<double> logTable_;

  // containers for the raw index data
  std::vector<int64_t> indices_;
  torch::Tensor tensor_;

  // index to locations/frequencies data
#ifdef FBGEMM_USE_FOLLY
  folly::F14FastMap<int64_t, std::vector<int64_t>> indexToLocations_;
#else
  std::map<int64_t, std::vector<int64_t>> indexToLocations_;
#endif

  // After constructor is called, freqs_ is kept in descending order and is
  // normalized to sum up to one
  std::vector<double> freqs_;

  // Estimate the heavy hitters. Returns a vector containing their
  // probabilities.
  std::vector<double> estimateHeavyHitters_();

  // Fill the log table.
  void computeLogTable_();
};

} // namespace fbgemm_gpu::tbe
