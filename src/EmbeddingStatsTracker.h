/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <fstream>
#include <mutex>
#include <string>
#include <unordered_map>
#include "./EmbeddingStatsTrackerConfig.h"

namespace fbgemm {

/**
 * @brief Statistics tracker for FBGEMM kernel embedding tables
 *
 * This class provides functionality to track and analyze statistics about
 * embedding tables access pattern in the FBGEMM kernels. It tracks every
 * access pattern inside the fbgemm kernel, collecting statistics such as batch
 * size, embedding dimensions, bag size, and data type to provide insights into
 * embedding table usage patterns.
 *
 * To enable statistics tracking for FBGEMM, set the following environment
 * variables:
 * - FBGEMM_STATS_ENABLE: Enables tracking. Set to 1 to enable and unset to
 * disable.
 */
class EmbeddingStatsTracker {
 public:
  /**
   * @brief Supported data types for embedding tables
   *
   * This enum defines the various data types that can be used for storing
   * embedding table values.
   */
  enum class DataType {
    FP32,
    BF16,
    FP16,
    FP8,
    INT8,
    INT4,
    INT2,
    SPARSE_INT8,
    SPARSE_FP32
  };

  /**
   * @brief Convert DataType enum to string representation
   *
   * @param type The DataType enum value to convert
   * @return std::string The string representation of the data type
   *
   * This function is used for logging to provide
   * human-readable names for the different data types.
   */
  static std::string dataTypeToString(DataType type) {
    switch (type) {
      case DataType::FP32:
        return "fp32";
      case DataType::FP16:
        return "fp16";
      case DataType::BF16:
        return "bf16";
      case DataType::FP8:
        return "fp8";
      case DataType::INT8:
        return "int8";
      case DataType::INT4:
        return "int4";
      case DataType::INT2:
        return "int2";
      case DataType::SPARSE_INT8:
        return "sparse-int8";
      case DataType::SPARSE_FP32:
        return "sparse-fp32";
      default:
        throw std::invalid_argument("Unknown data type");
    }
  }

  /**
   * @brief Composite entry for identifying embedding access patterns
   *
   * Embedding accesses with the same dimensions, row count, data type, batch
   * size and bag size will be considered the same pattern for statistics
   * purposes.
   */
  struct AccessPatternEntry {
    int64_t rows; // Number of rows in the table
    int64_t dims; // Embedding dimension
    int64_t batch_size; // Batch size
    int64_t bag_size; // Bag size
    DataType
        input_data_type; // Data type (e.g., "fp32", "fp16", "int8", "int4")
    DataType
        output_data_type; // Data type (e.g., "fp32", "fp16", "int8", "int4")

    /**
     * @brief Construct a new AccessPatternEntry object
     *
     * @param r Number of rows in the embedding table
     * @param d Embedding dimension
     * @param batch_size Number of embeddings being looked up in a batch
     * @param bag_size Number of embeddings pooled together (pooling factor)
     * @param dt Data type used for storing the embedding values
     *
     * This constructor creates a unique entry that identifies an embedding
     * access pattern based on dimensions and access characteristics.
     */
    AccessPatternEntry(
        int64_t r,
        int64_t d,
        int64_t batch_size,
        int64_t bag_size,
        DataType input_dt,
        DataType output_dt)
        : rows(r),
          dims(d),
          batch_size{batch_size},
          bag_size(bag_size),
          input_data_type(input_dt),
          output_data_type(output_dt) {}

    // Equality operator for hash map
    // Used by the unordered_map to determine if two AccessPatternEntry objects
    // represent the same entry
    bool operator==(const AccessPatternEntry& other) const {
      return rows == other.rows && dims == other.dims &&
          batch_size == other.batch_size && bag_size == other.bag_size &&
          input_data_type == other.input_data_type &&
          output_data_type == other.output_data_type;
    }

    // Generate a string representation for debugging and logging purposes
    std::string toString() const {
      return "rows=" + std::to_string(rows) + ";" +
          "dims=" + std::to_string(dims) + ";" +
          "input_data_type=" + dataTypeToString(input_data_type) + ";" +
          "output_data_type=" + dataTypeToString(output_data_type) + ";" +
          "batch_size=" + std::to_string(batch_size) + ";" +
          "bag_size=" + std::to_string(bag_size) + ";";
    }
  };

  /**
   * @brief Hash function for AccessPatternEntry to use in unordered_map
   *
   * This struct provides a hash function for AccessPatternEntry objects so they
   * can be used as keys in an unordered_map. It combines the hashes of each
   * component of the AccessPatternEntry to create a unique hash value.
   */
  struct AccessPatternEntryHash {
    /**
     * @brief Generate a hash value for an AccessPatternEntry
     *
     * @param key The AccessPatternEntry to hash
     * @return std::size_t The hash value
     *
     * This function combines the hash values of each component of the
     * AccessPatternEntry using bit shifts and XOR operations to create a
     * well-distributed hash.
     */
    std::size_t operator()(const AccessPatternEntry& key) const {
      // Combine the hash of each component using bit shifts and XOR
      std::size_t h1 = std::hash<int64_t>{}(key.rows);
      std::size_t h2 = std::hash<int64_t>{}(key.dims);
      std::size_t h3 = std::hash<int>{}(static_cast<int>(key.input_data_type));
      std::size_t h4 = std::hash<int>{}(static_cast<int>(key.output_data_type));
      std::size_t h5 = std::hash<int64_t>{}(key.batch_size);
      std::size_t h6 = std::hash<int64_t>{}(key.bag_size);
      return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3) ^ (h5 << 4) ^ (h6 << 5);
    }
  };

  /**
   * @brief Get the singleton instance of EmbeddingStatsTracker
   *
   * @return Reference to the singleton instance
   *
   * This class follows the singleton pattern to ensure that only one instance
   * of the statistics tracker exists throughout the application. All code
   * should access the tracker through this method.
   */

  static EmbeddingStatsTracker& getInstance();

  /**
   * @brief Record inference statistics for an embedding table
   *
   * Tracks every inference operation inside the fbgemm kernel.
   * This method is called during embedding lookups to collect usage statistics
   * which can be used for performance analysis and optimization.
   *
   * @param rows Number of rows in the table (data_size)
   * @param dims Embedding dimension
   * @param input_data_type Data type used for input
   * @param output_data_type Data type used for output
   * @param batch_size Number of output rows (output_size)
   * @param bag_size Bag size (pooling factor)
   *
   */
  void recordPattern(
      int64_t rows,
      int64_t dims,
      DataType input_data_type,
      DataType output_data_type,
      int64_t batch_size,
      int64_t bag_size);

  /**
   * @brief Reset all statistics
   *
   * Clears all recorded statistics and resets the internal state.
   * This can be useful when starting a new profiling session or when
   * you want to discard previously collected data.
   */
  void reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    tables_.clear();
  }

  // Destructor
  ~EmbeddingStatsTracker() {
    if (logFile_.is_open()) {
      logFile_.close();
    }
  }

 private:
  // Private constructor for singleton pattern
  EmbeddingStatsTracker() = default;

  // Private copy constructor and assignment operator to enforce singleton
  // pattern
  EmbeddingStatsTracker(const EmbeddingStatsTracker&) = delete;
  EmbeddingStatsTracker& operator=(const EmbeddingStatsTracker&) = delete;

  // Private move constructor and move assignment operator to enforce singleton
  // pattern
  EmbeddingStatsTracker(EmbeddingStatsTracker&&) = delete;
  EmbeddingStatsTracker& operator=(EmbeddingStatsTracker&&) = delete;

  // Map of table keys to their frequency of access
  // Each entry represents a unique embedding stats and how many
  // times it was accessed
  std::unordered_map<AccessPatternEntry, uint64_t, AccessPatternEntryHash>
      tables_;

  // Counter for total number of inference samples recorded
  uint64_t sampleCount_ = 0;

  // Mutex for thread safety when recording statistics from multiple threads
  std::mutex mutex_;

  // Log file stream for writing statistics to disk
  std::ofstream logFile_;

  EmbeddingStatsTrackerConfig config_;
};

} // namespace fbgemm
