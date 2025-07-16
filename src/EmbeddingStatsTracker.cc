/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#define FBGEMM_EXPORTS
#include "./EmbeddingStatsTracker.h"
#include <iostream>
#include "fbgemm/Utils.h"

namespace fbgemm {

EmbeddingStatsTracker& EmbeddingStatsTracker::getInstance() {
  static EmbeddingStatsTracker instance;
  return instance;
}

void EmbeddingStatsTracker::recordPattern(
    int64_t rows,
    int64_t dims,
    DataType input_data_type,
    DataType output_data_type,
    int64_t batch_size,
    int64_t bag_size) {
  if (!is_stats_enabled() || bag_size == 0) {
    return;
  }
  std::lock_guard<std::mutex> lock(mutex_);

  // Create the entry and ensure the pattern exists
  AccessPatternEntry key(
      rows, dims, batch_size, bag_size, input_data_type, output_data_type);
  auto result = tables_.find(key);
  if (result == tables_.end()) {
    tables_[key] = 1;
  } else {
    result->second += 1;
  }

  sampleCount_ += 1;

  if (sampleCount_ % config_.getLogFreq() == 0) {
    // Log the table statistics - only try to open the file if it's not
    logFile_.open(config_.getLogFilePath(), std::ios::out | std::ios::trunc);

    if (!logFile_) {
      std::cerr << "Failed to open log file: " << config_.getLogFilePath()
                << '\n';
      return;
    }
    for (const auto& pair : tables_) {
      const auto& pattern = pair.first;
      logFile_ << pattern.toString() << "freq=" << pair.second << ";" << '\n';
    }
    logFile_.flush();
    logFile_.close();
  }
}

} // namespace fbgemm
