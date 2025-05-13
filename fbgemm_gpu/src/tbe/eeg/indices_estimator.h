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
  std::optional<IndicesDistributionParameters> estimate() const;

  // Quality of the fit, measured by KL divergence to a set of parameters. Lower
  // score is better, with best possible score of 0.0.
  double estimateQuality(const IndicesDistributionParameters& params) const;

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

  // index to locations/frequencies data
#ifdef FBGEMM_USE_FOLLY
  folly::F14FastMap<int64_t, int64_t> indexCounts_;
#else
  std::map<int64_t, int64_t> indexCounts_;
#endif

  // After constructor is called, freqs_ is kept in descending order and is
  // normalized to sum up to one
  std::vector<double> freqs_;

  // Stores the log table for fast computation of maximum likelihood estimation
  std::vector<double> logTable_;

  // Cache of the computed max index value
  mutable std::optional<int64_t> cacheMaxIndex_;

  // Populate the index frequencies
  void populateIndexFreqs_(const torch::Tensor&);

  // Populate the log table from the index frequencies
  void populateLogTable_();

  // Returns a vector containing the probabilities of the heavy hitters
  std::vector<double> heavyHitters() const;

  // Returns the number of indices used to estimate the distribution
  int64_t numIndices() const;

  // Returns the max index value in the indices data
  int64_t maxIndex() const;

  // Returns zipf parameters used to fit the distribution
  ZipfParameters zipfParams(const std::vector<double>&) const;
};

} // namespace fbgemm_gpu::tbe
