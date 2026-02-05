/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "indices_generator.h"

#include <chrono>
#include <cmath>
#include <execution>

namespace fbgemm_gpu::tbe {

IndicesGenerator::IndicesGenerator(
    const IndicesDistributionParameters& params,
    uint64_t seed)
    : params_(params),
      rng_(seed_seq{seed}),
      heavyHittersDist_(
          std::begin(params.heavyHitters),
          std::end(params.heavyHitters)),
      zipfianDist_(
          params.zipfParams.s,
          params.zipfParams.q + params.heavyHitters.size(),
          params.maxIndex - params.heavyHitters.size()) {
  double heavyHitterTotalProb = std::reduce(
      std::begin(params.heavyHitters), std::end(params.heavyHitters));
  heavyHitterSelectorDist_ =
      random::bernoulli_distribution(heavyHitterTotalProb);
}

// Helper function to convert a tagged indices vector to an ATen tensor.
static torch::Tensor convertVectorToTensor(
    const std::vector<std::pair<int64_t, double>>& indicesWithTags) {
  std::vector<int64_t> indices(indicesWithTags.size());
  std::transform(
      std::begin(indicesWithTags),
      std::end(indicesWithTags),
      std::begin(indices),
      [](const auto& [index, tag]) {
        return index;
      });
  return torch::from_blob(
             indices.data(),
             {static_cast<long>(indices.size())},
             at::TensorOptions().dtype(torch::kInt64))
      .clone();
}

// Metadata structure for an index
struct IndexMetadata {
  std::vector<double> tags;
  int64_t freq{};
};

torch::Tensor IndicesGenerator::generate() {
  using timer = std::chrono::high_resolution_clock;
  using us = std::chrono::microseconds;

  const auto t0 = timer::now();
  std::vector<std::pair<int64_t, double>> indicesWithTags(params_.numIndices);
  std::vector<IndexMetadata> indicesMetadata(params_.maxIndex + 1);

  // Tag generation for the algorithm

  // Tag hyperparams
  // TODO: this may have to vary shard to shard and we need a way to estimate it
  // from a real shard
  static constexpr double kTagClusterProbability =
      0.6; // Keep between 0 and 1. Exact 0 corresponds to
           // standard uniformly random shuffle.
  static constexpr double kTagClusterCoeff =
      5; // Keep above 1.0, 1.0 is (roughly) a uniformly random shuffle.
         // Distribution isn't too sensitive to it's precise value.

  random::uniform_real_distribution<double> tagUniformDist;
  random::bernoulli_distribution tagUniformSelector(1 - kTagClusterProbability);

  // First handle the index
  for (auto& [index, tag] : indicesWithTags) {
    if (heavyHitterSelectorDist_(rng_)) {
      index = heavyHittersDist_(rng_);
    } else {
      index = zipfianDist_(rng_) + params_.heavyHitters.size();
    }
    auto curIdx = index;
    indicesMetadata[curIdx].freq++;
  }

  // Now reserve enough space in each of the metadata tag vectors
  for (int64_t i = 0; i <= params_.maxIndex; ++i) {
    indicesMetadata[i].tags.reserve(indicesMetadata[i].freq);
  }

  // Now handle the tags
  random::exponential_distribution exponentialDist;
  for (auto& [index, tag] : indicesWithTags) {
    double newTag = NAN;

    // In the case where the current metadata for the index is empty, simply
    // push in a U[0,1]
    auto curIdx = index;
    if (indicesMetadata[curIdx].tags.empty()) {
      newTag = tagUniformDist(rng_);
    }

    // Otherwise follow the algorithm sketch
    else {
      if (tagUniformSelector(rng_)) {
        newTag = tagUniformDist(rng_);
      } else {
        // Pick a nearby tag at random from the existing tags for the idx
        random::uniform_int_distribution<int64_t> nearbyTagSelector(
            0, indicesMetadata[curIdx].tags.size() - 1);
        auto nearbyTag = indicesMetadata[curIdx].tags[nearbyTagSelector(rng_)];

        // Shift by an exponential rv
        double exponentialRate =
            kTagClusterCoeff * indicesMetadata[curIdx].freq;
        newTag = nearbyTag + exponentialDist(rng_) / exponentialRate;
      }
    }

    tag = newTag;
    indicesMetadata[curIdx].tags.push_back(newTag);
  }

  // Now sort the indices by their tags. Use parallel sort for some extra speed
  // (vector is very large).
  std::sort(
      std::execution::par,
      std::begin(indicesWithTags),
      std::end(indicesWithTags),
      [](const auto& [lhs_idx, lhs_tag],
         const auto& [rhs_idx, rhs_tag]) {
        return lhs_tag < rhs_tag;
      });

  auto t = convertVectorToTensor(indicesWithTags);

  auto t1 = timer::now();
  std::cout << "Time taken to generate indices (us): "
            << std::chrono::duration_cast<us>(t1 - t0).count() << "\n";
  return t;
}

} // namespace fbgemm_gpu::tbe
