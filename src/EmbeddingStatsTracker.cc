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
    DataType data_type,
    int64_t batch_size,
    int64_t bag_size) {
  if (!is_stats_enabled()) {
    return;
  }
  std::lock_guard<std::mutex> lock(mutex_);

  // Create the entry and ensure the pattern exists
  AccessPatternEntry key(rows, dims, batch_size, bag_size, data_type);
  if (tables_.find(key) == tables_.end()) {
    tables_[key] = 0;
  }

  tables_[key] += 1;
  sampleCount_ += 1;

  if (sampleCount_ % logFreq_ == 0) {
    // Log the table statistics - only try to open the file if it's not
    logFile_.open(logFilePath_, std::ios::out | std::ios::trunc);

    if (!logFile_) {
      std::cerr << "Failed to open log file: " << logFilePath_ << '\n';
      return;
    }
    logFile_ << "=== Sample " << sampleCount_ << " ===" << std::endl;
    for (const auto& pair : tables_) {
      const auto& pattern = pair.first;
      logFile_ << "rows=" << pattern.rows << "; " << "dims=" << pattern.dims
               << "; " << "data_type=" << dataTypeToString(pattern.data_type)
               << "; " << "batch_size=" << pattern.batch_size << "; "
               << "bag_size=" << pattern.bag_size << "; "
               << "freq=" << pair.second << ";" << std::endl;
    }
    logFile_ << "==============================\n" << std::endl;
    logFile_.flush();
    logFile_.close();
  }
}

} // namespace fbgemm
