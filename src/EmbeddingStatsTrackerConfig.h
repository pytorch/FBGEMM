/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <cstdlib>
#include <string>

namespace fbgemm {

/**
 * @brief Configuration for EmbeddingStatsTracker
 *
 * This class provides configuration parameters for the EmbeddingStatsTracker.
 * It controls aspects such as logging frequency and log file path through the
 * following environment variables:
 *
 * FBGEMM_STATS_FREQ: Specifies the number of samples after which the tracker
 * logs statistics to the log file. The default is 1,000,000.
 *
 * FBGEMM_STATS_FILENAME: Specifies the
 * path of the log file provided by the user. The default is
 * "/tmp/fbgemm_embedding_stats.txt".
 *
 */
class EmbeddingStatsTrackerConfig {
 public:
  /**
   * @brief Get the frequency of logging (every N samples)
   *
   * @return uint64_t The logging frequency
   */
  uint64_t getLogFreq() const {
    return logFreq_;
  }

  /**
   * @brief Get the path to the log file
   *
   * @return std::string The log file path
   */
  std::string getLogFilePath() const {
    return logFilePath_;
  }

 private:
  // Frequency of logging (every N samples)
  // Controls how often statistics are written to the log file
  // Log frequency can be configured via FBGEMM_STATS_FREQ environment variable
  // Default is every 100,000 samples
  uint64_t logFreq_{
      std::getenv("FBGEMM_STATS_FREQ") == nullptr
          ? 1000000
          : std::stoul(std::getenv("FBGEMM_STATS_FREQ"))};

  // Path to the log file can be configured via FBGEMM_STATS_LOGPATH
  // environment variable.
  // Default is /tmp/fbgemm_embedding_stats.txt
  std::string logFilePath_{
      std::getenv("FBGEMM_STATS_LOGPATH") == nullptr
          ? "/tmp/fbgemm_embedding_stats.txt"
          : std::getenv("FBGEMM_STATS_LOGPATH")};
};

} // namespace fbgemm
