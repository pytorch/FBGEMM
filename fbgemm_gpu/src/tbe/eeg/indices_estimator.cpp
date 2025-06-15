/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "indices_estimator.h"
#include <cassert>
#include <chrono>
#include <fstream>
#include <vector>
#include "eeg_utils.h"

namespace fbgemm_gpu::tbe {

void IndicesEstimator::populateIndexFreqs_(const torch::Tensor& indices) {
  // Count the frequency of indices
  const auto* data = indices.data_ptr<int64_t>();
  for (auto i = 0; i < indices.numel(); i++) {
    const auto idx = data[i];
    indexCounts_[idx] += 1;
  }

  // Collect the frequencies
  for (const auto& [_, count] : indexCounts_) {
    freqs_.emplace_back(static_cast<double>(count));
  }

  // Sort and normalize the frequencies
  std::sort(std::begin(freqs_), std::end(freqs_), std::greater{});
  auto normalize_const = std::reduce(std::begin(freqs_), std::end(freqs_));
  for (auto& freq : freqs_) {
    freq /= normalize_const;
  }
}

void IndicesEstimator::populateLogTable_() {
  logTable_.resize((maxIndex() + kMaxQ + 1) * kLevels_);
  double cur = 1.0;
  for (int64_t i = 0; i < logTable_.size(); ++i) {
    logTable_[i] = log(cur);
    cur += 1.0 / kLevels_;
  }
}

IndicesEstimator::IndicesEstimator(const torch::Tensor& indices) {
  TORCH_CHECK(
      indices.numel() > 0, "indices numel is ", indices.numel(), "(< 1)");

  TORCH_CHECK(
      indices.dtype() == at::kLong,
      "indices dtype is ",
      indices.dtype(),
      "(!= I64)");

  // Populate the index frequencies
  populateIndexFreqs_(indices);

  // Populate the log table
  populateLogTable_();
}

IndicesEstimator::IndicesEstimator(const std::filesystem::path& tensors_path) {
  // NOTE: PyTorch API requires us to use a torch::pickle_load on a
  // vector<char> (torch::load doesn't work here)
  // https://fb.workplace.com/groups/1405155842844877/posts/4947064988653927/?comment_id=4947149218645504

  // Open the file
  std::ifstream input(tensors_path, std::ios::binary);
  std::vector<char> bytes(
      (std::istreambuf_iterator<char>(input)),
      (std::istreambuf_iterator<char>()));
  input.close();

  // Load the tensor
  auto ival = torch::pickle_load(bytes);
  assert((ival.isTensor()) && "Loaded file is not a tensor!");

  // Pass it to the tensor-based constructor
  IndicesEstimator(ival.toTensor());
}

std::vector<double> IndicesEstimator::heavyHitters() const {
  std::vector<double> hitters;

  double cdf = 0.0;
  for (int i = 0; i < kHeavyHitterMaxEntries_; ++i) {
    hitters.emplace_back(freqs_[i]);
    cdf += freqs_[i];
    if (cdf > kHeavyHitterThreshold_) {
      break;
    }
  }

  return hitters;
}

int64_t IndicesEstimator::numIndices() const {
  int64_t sum = 0;
  for (const auto& [_, value] : indexCounts_) {
    sum += value;
  }
  return sum;
}

int64_t IndicesEstimator::maxIndex() const {
  if (!cacheMaxIndex_.has_value()) {
    cacheMaxIndex_ =
        std::max_element(
            indexCounts_.begin(),
            indexCounts_.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; })
            ->first;
  }

  return *cacheMaxIndex_;
}

ZipfParameters IndicesEstimator::zipfParams(
    const std::vector<double>& heavyHitters) const {
  ZipfParameters zipfParams;

  auto zipfStart = heavyHitters.size();

  // Now do the MLE estimation conditioned on the rest
  auto zipfTotalFreq =
      std::reduce(std::begin(freqs_) + zipfStart, std::end(freqs_));
  auto maxLogLikelihood = -std::numeric_limits<double>::infinity();

  // Sweep over q in outer loop, and s in the inner
  // loop since q requires recomputing inner products

  // For q, incr by 1/kSweepGranularity_ for q in [1, 10], then by
  // 10/kSweepGranularity_ in [10, 100], etc. For s, incr by 0.01. We also
  // enforce a "consistency" with the heavy hitters, namely min heavyHitter
  // probability >= kHeavyHitterLowerBound * max zipf probability.
  // Similarly, we also enforce min heavyHitter probability <=
  // kHeavyHitterUpperBound * max zipf probability (for some k > 1)
  static constexpr double kHeavyHitterUpperBound_ = 5.0;
  static constexpr double kHeavyHitterLowerBound_ = 1.0;

  double minHeavyHitterProb = heavyHitters.back();
  double qIncr = 1.0;

  for (double q = 1.0; q < kMaxQ; q += qIncr) {
    double freqTerm = 0.0;
    long logTableIdx = lrint((zipfStart + q) * kLevels_ - 1);

    for (int64_t k = zipfStart; k < freqs_.size();
         ++k, logTableIdx += kLevels_) {
      freqTerm -= freqs_[k] * logTable_[logTableIdx];
    }

    for (double s = kMinS_; s < kMaxS_; s += kSStep_) {
      // Consistency test before proceeding with estimate.
      double normalizeConst =
          getZipfianConstant(s, q + zipfStart, freqs_.size() - zipfStart);
      double zipfMaxProb =
          (zipfTotalFreq / normalizeConst) * std::pow(q + zipfStart, -s);
      double ratio = minHeavyHitterProb / zipfMaxProb;

      if ((ratio < kHeavyHitterLowerBound_) ||
          (ratio > kHeavyHitterUpperBound_)) {
        std::cout << "Skipping (s,q) (" << s << ", " << q
                  << "): " << " inconsistent with heavy hitters!" << "\n";
        continue;
      }

      double logLikelihood = -zipfTotalFreq * log(normalizeConst) +
          s * freqTerm - kQRegularizer_ * q;
      if (logLikelihood > maxLogLikelihood) {
        std::cout << "Found best Log likelihood so far on (s,q) (" << s << ", "
                  << q << "): " << logLikelihood << "\n";
        maxLogLikelihood = logLikelihood;
        zipfParams.q = q;
        zipfParams.s = s;
      }
    }

    qIncr = exp10(std::floor(std::log10(q))) / kQSweepGranularity_;
  }

  return zipfParams;
}

std::optional<IndicesDistributionParameters> IndicesEstimator::estimate()
    const {
  if (indexCounts_.empty()) {
    return std::nullopt;
  }

  using timer = std::chrono::high_resolution_clock;
  using us = std::chrono::microseconds;
  auto t0 = timer::now();

  const auto hitters = heavyHitters();
  auto params = IndicesDistributionParameters{
      hitters,
      zipfParams(hitters),
      maxIndex(),
      numIndices(),
  };

  auto t1 = timer::now();
  std::cout << "Time taken to estimate parameters (us): "
            << std::chrono::duration_cast<us>(t1 - t0).count() << "\n";
  return params;
}

double IndicesEstimator::estimateQuality(
    const IndicesDistributionParameters& params) const {
  // KL divergence between the true indices distribution and the estimated one
  // (in bits)
  const auto zipfStart = params.heavyHitters.size();
  const auto zipfTotalFreq =
      std::reduce(std::begin(freqs_) + zipfStart, std::end(freqs_));
  const double normalizeConst = getZipfianConstant(
      params.zipfParams.s,
      params.zipfParams.q + zipfStart,
      freqs_.size() - zipfStart);

  double kl = 0.0;
  for (size_t i = 0; i < freqs_.size(); ++i) {
    double trueProb = freqs_[i];
    if (trueProb == 0.0) {
      continue;
    }

    double estProb = 1; // Initializing to 1 to silence lint errors
    if (i < zipfStart) {
      estProb = params.heavyHitters[i];
    } else {
      estProb = (zipfTotalFreq / normalizeConst) *
          std::pow(params.zipfParams.q + i, -params.zipfParams.s);
    }

    kl += trueProb * std::log2(trueProb / estProb);
  }

  return kl;
}

} // namespace fbgemm_gpu::tbe
