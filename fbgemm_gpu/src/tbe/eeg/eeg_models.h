/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cassert>
#include <iostream>
#include <optional>
#include <vector>

namespace fbgemm_gpu::tbe {

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::optional<T>& ot) {
  if (ot.has_value()) {
    return (os << ot.value());
  } else {
    return (os << "null");
  }
}

#define DECL_OSTREAM_OUT(T) \
  std::ostream& operator<<(std::ostream& os, const T& t);

// Zipf distribution written as p(k) \propto (k+q)^{-s}
// Note: our convention is to start at k=0, so we MUST have q > 0 (and
// realistically q >=1 for modeling)
struct ZipfParameters {
  ZipfParameters() : q{1}, s{2} {
    assert((q > 0) && "q must be > 0 in Zipf!");
  }

  ZipfParameters(double q, double s) : q{q}, s{s} {
    assert((q > 0) && "q must be > 0 in Zipf!");
  }

  double q;
  double s;

  std::string json() const;
};

DECL_OSTREAM_OUT(ZipfParameters);

// Modeled indices distribution
struct IndicesDistributionParameters {
  // Heavy hitters for the Zipf distribution, i.e. a probability density map
  // for the most hot indices.  There should not ever be more than 100
  // elements, and currently it is limited to 20 entries (kHeavyHittersMaxSize)
  std::vector<double> heavyHitters;

  // Parameters for the Zipf distribution (x+q)^{-s}
  ZipfParameters zipfParams;

  // Max index value in the distribution - should be in the range [0, E), where
  // E is the number of rows in the embedding table
  int64_t maxIndex;

  // Number of indices to generate
  int64_t numIndices;

  // NOTE: Compiler-generated aggregate initialization constructors (P0960R3,
  // P1975R0) did not exist prior to C++20, but FBGEMM_GPU OSS still uses C++17,
  // namely when building against CUDA 11.8.  Remove this constructor once CUDA
  // 11.8 is deprecated from FBGEMM_GPU support.
  IndicesDistributionParameters(
      const std::vector<double>& _1,
      const ZipfParameters& _2,
      int64_t _3,
      int64_t _4)
      : heavyHitters{_1}, zipfParams{_2}, maxIndex{_3}, numIndices{_4} {}

  IndicesDistributionParameters() = default;

  // JSON string representation of
  std::string json() const;
};

DECL_OSTREAM_OUT(IndicesDistributionParameters);

struct TBEBatchStats {
  // batch size, i.e., number of lookups
  int64_t B;
  // Standard deviation of B (for variable batch size configuration)
  std::optional<int64_t> sigmaB;

  std::string json() const;
};

DECL_OSTREAM_OUT(TBEBatchStats);

struct TBEIndicesStats {
  // zipf*: parameters for the Zipf distribution (x+q)^{-s}
  double zipfQ;
  double zipfS;
  // Heavy hitters for the Zipf distribution, i.e. a probability density map
  // for the most hot indices.  There should not ever be more than 100
  // elements, and currently it is limited to 20 entries
  std::vector<double> heavyHitters;

  std::string json() const;
};

DECL_OSTREAM_OUT(TBEIndicesStats);

struct TBEPoolingStats {
  // Bag size, i.e., pooling factor
  int64_t L;
  // Standard deviation of L(for variable bag size configuration)
  std::optional<int64_t> sigmaL;

  std::string json() const;
};

DECL_OSTREAM_OUT(TBEPoolingStats);

struct TBEAnalysisStats {
  // Number of tables
  int64_t T;
  // Number of rows in the embedding table
  int64_t E;
  // Embedding dimension (number of columns)
  int64_t D;
  // Batch stats
  TBEBatchStats batch;
  // Indices stats
  TBEIndicesStats indices;
  // Pooling stats
  TBEPoolingStats pooling;

  std::string json() const;
};

DECL_OSTREAM_OUT(TBEAnalysisStats);

#undef DECL_OSTREAM_OUT

} // namespace fbgemm_gpu::tbe
