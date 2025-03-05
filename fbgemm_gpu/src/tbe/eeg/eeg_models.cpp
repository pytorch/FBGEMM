/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "eeg_models.h"
#include <sstream>

namespace fbgemm_gpu::tbe {

#define DEFINE_OSTREAM_OUT(T)                              \
  std::ostream& operator<<(std::ostream& os, const T& t) { \
    return (os << t.json());                               \
  }

std::string ZipfParameters::json() const {
  std::stringstream ss;
  ss << "{ \"q\": " << q << ", \"s\": " << s << " }";
  return ss.str();
}

DEFINE_OSTREAM_OUT(ZipfParameters);

std::string IndicesDistributionParameters::json() const {
  std::stringstream ss;
  ss << "{ \"zipf\": " << zipfParams << ", \"maxIndex\": " << maxIndex
     << ", \"numIndices\": " << numIndices << ", \"heavyHitters\": [";
  for (auto freq : heavyHitters) {
    ss << freq << ", ";
  }
  ss << "] }";
  return ss.str();
}

DEFINE_OSTREAM_OUT(IndicesDistributionParameters);

std::string TBEBatchStats::json() const {
  std::stringstream ss;
  ss << "{ \"B\": " << B << ", \"sigmaB\": " << sigmaB << " }";
  return ss.str();
}

DEFINE_OSTREAM_OUT(TBEBatchStats);

std::string TBEIndicesStats::json() const {
  std::stringstream ss;
  ss << "{ \"zipfQ\": " << zipfQ << ", \"zipfS\": " << zipfS
     << ", \"heavyHitters\": [";
  for (auto freq : heavyHitters) {
    ss << freq << ", ";
  }
  ss << "] }";
  return ss.str();
}

DEFINE_OSTREAM_OUT(TBEIndicesStats);

std::string TBEPoolingStats::json() const {
  std::stringstream ss;
  ss << "{ \"B\": " << L << ", \"sigmaB\": " << sigmaL << " }";
  return ss.str();
}

DEFINE_OSTREAM_OUT(TBEPoolingStats);

std::string TBEAnalysisStats::json() const {
  std::stringstream ss;
  ss << "{ \"batch\": " << batch << ", \"indices\": " << indices
     << ", \"pooling\": " << pooling << ", \"T\": " << T << ", \"E\": " << E
     << ", \"D\": " << D << " }";
  return ss.str();
}

DEFINE_OSTREAM_OUT(TBEAnalysisStats);

#undef DEFINE_OSTREAM_OUT

} // namespace fbgemm_gpu::tbe
