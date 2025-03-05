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
#include <iomanip>
#include <iostream>
#include <limits>

#ifdef FBGEMM_USE_ABSL
#include <absl/random/random.h>
#else
#include <random>
#endif

namespace fbgemm_gpu::tbe {

torch::Tensor loadTensorFromFile(const std::filesystem::path& path);

void saveTensorToFile(
    const torch::Tensor& t,
    const std::filesystem::path& path);

// Normalization constant Z for Zipfian distribution p(k) = Z * (k+q)^{-s} for k
// in [0, ..., n-1]
// n < 0 is treated as the infinite Zipfian distribution on [0, 1, 2, ...]
// NOTE: this function is not double precision ULP accurate, but is good enough
// for our purposes (8+ digits of accuracy)
double getZipfianConstant(double s, double q, int64_t n = -1);

// rng for Zipfian distribution Zipfian distribution p(k) = Z * (k+q)^{-s} for k
// in [0, ..., n-1].
// Use a simple rejection sampler, modified from
// https://jasoncrease.medium.com/rejection-sampling-the-zipf-distribution-6b359792cffa
// NOTE: only supports finite Zipfian distributions (n >= 1) for now
// NOTE: ideally should just wrap absl, but note
// https://github.com/abseil/abseil-cpp/issues/1818 (s < 1 is unsupported)
class ZipfianDistribution {
 public:
  ZipfianDistribution(double s, double q, int64_t n) : s_{s}, q_{q}, n_{n} {
    assert((n >= 1) && "Only finite Zipfian supported for now");
    assert((s >= 0) && "s must be >= 0 in Zipf!");
    assert((q > 0) && "q must be > 0 in Zipfian distribution!");
    // We treat |s - 1| <= kEps as s = 1.0 to avoid singularities in the
    // rejection sampler that blow up error. NOTE: this strictly speaking
    // introduces a bias, but in practice it is small enough to ignore.
    if ((s != 1.0) && (std::abs(s - 1.0) <= kEps_)) {
      auto printPrecision = std::cout.precision();
      std::cout << "WARNING: changing Zipfian s from "
                << std::setprecision(std::numeric_limits<double>::max_digits10)
                << s << " to 1.0 to improve generator accuracy"
                << std::setprecision(printPrecision) << std::endl;
      s_ = 1.0;
    }

    if (s_ == 1.0) {
      t_ = 1.0 + std::log(q + n - 1) - std::log(q);
    } else {
      t_ = std::pow(q + n - 1, 1.0 - s_) - std::pow(q, 1.0 - s_);
      t_ /= (1.0 - s_);
      t_ += 1.0;
    }
    invCdfTerm_ = std::pow(q_, 1.0 - s_); // unused when s == 1.0
  }

  template <class Generator>
  int64_t operator()(Generator& rng) {
    while (true) {
      double invBound = boundingInvCdf(uniformDist(rng));
      // Although it is extremely likely 0 <= x <= n, floating point error near
      // s==1.0, large q, large n, etc can push things over slightly.
      // We clip in such cases to fulfil our [0, ..., n-1] sampling contract.
      long long x = std::min(
          static_cast<long long>(n_), std::llround(std::floor(invBound + 1)));
      double ratio = std::pow(x + q_ - 1, -s_);
      if (x > 1) {
        ratio *= std::pow(invBound + q_ - 1, s_);
      }
      double y = uniformDist(rng);
      if (y < ratio) {
        return x - 1;
      }
    }
    return 0;
  }

 private:
  static constexpr double kEps_ = 1e-12;
  double s_;
  double q_;
  int64_t n_;

  double t_;
  double invCdfTerm_;

#ifdef FBGEMM_USE_ABSL
  absl::uniform_real_distribution<double> uniformDist;
#else
  std::uniform_real_distribution<double> uniformDist;
#endif

  // Rejection sampling (continuous) bound distribution
  inline double boundingInvCdf(double p) {
    double pt = p * t_;
    if (pt <= 1.0) {
      return pt;
    }

    if (s_ != 1.0) {
      return (1.0 - q_) +
          std::pow((1.0 - s_) * (pt - 1.0) + invCdfTerm_, 1.0 / (1.0 - s_));
    } else {
      return (1.0 - q_) + q_ * std::exp(pt - 1.0);
    }
  }
};

} // namespace fbgemm_gpu::tbe
