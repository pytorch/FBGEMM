/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <torch/torch.h>

#ifdef FBGEMM_USE_ABSL
#include <absl/random/random.h>
// Technically should be part of the absl/random/random.h header, but isn't
#include <absl/random/discrete_distribution.h>
#else
#include <random>
#endif

#include "eeg_models.h"
#include "eeg_utils.h"

namespace fbgemm_gpu::tbe {

#ifdef FBGEMM_USE_ABSL

namespace random = absl;
using bitgen = absl::BitGen;
using seed_seq = absl::SeedSeq;

#else

namespace random = std;
using bitgen = std::mt19937_64;
using seed_seq = uint64_t;

#endif

// Indices generator class takes in distribution parameters and generates
// indices Algorithm sketch:
// 1. Generate raw indices according to the distribution parameters.
// 2. Attach a random tag to each index.
// 3. Sort the indices based on the tags, and return the sorted indices.

// For step 2 above, we elaborate on the "random tag".
// If the tags were all iid U[0, 1], this would generate a uniformly random
// permutation/shuffle of the indices with frequencies governed by Step 1. We
// modify the iid U[0, 1] as follows, in order to encourage some short range
// clustering/reuse of indices. Loop over all the slots of the indices vector.
// With probability 1 - \epsilon, pick independent U[0, 1].
// With probability \epsilon, pick a tag at random from the preceding tags for
// the given index i. Call this t. Then, add a small exponential rv to t, and
// use it as the tag for the current slot. Here, by "small", we mean an
// exponential rv with mean = 1/(k * frequency of i) for some hyperparameter k
// > 1.

// This algorithm takes time O(n log n), where n is the number of indices to
// generate. NOTE: it maybe possible to have an O(n) algorithm based on
// modifying Fisher-Yates shuffle, but it appears more difficult to naturally
// modify it to favor extra clustering.
class IndicesGenerator {
 public:
  // Constructor takes in distribution parameters along with a uint64_t seed.
  // NOTE: absl does not guarantee determinism across runs even with a fixed
  // seed: https://abseil.io/docs/cpp/guides/random
  // TODO: if we want fully deterministic interface, go via std::random at cost
  // of performance.
  explicit IndicesGenerator(
      const IndicesDistributionParameters& params,
      uint64_t seed = 0);

  // Generate indices according to the distribution
  torch::Tensor generate();

 private:
  IndicesDistributionParameters params_;

  // Use absl for performance reasons
  bitgen rng_;
  random::discrete_distribution<int64_t> heavyHittersDist_;
  ZipfianDistribution zipfianDist_;
  // Selector between heavy hitters and Zipfian
  random::bernoulli_distribution heavyHitterSelectorDist_;
};

} // namespace fbgemm_gpu::tbe
