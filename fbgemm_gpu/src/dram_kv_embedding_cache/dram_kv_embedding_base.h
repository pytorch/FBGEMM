/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "../ssd_split_embeddings_cache/kv_db_table_batched_embeddings.h"
#include "feature_evict.h"

namespace kv_mem {

/**
 * @brief Interface for DRAM KV embedding.
 *
 * This class extends the base EmbeddingKVDB with additional methods
 * specific to memory-based embedding caches, particularly for feature
 * eviction functionality.
 */
class DramKVEmbeddingBase : public kv_db::EmbeddingKVDB {
 public:
  // Inherit constructors from the base class
  using kv_db::EmbeddingKVDB::EmbeddingKVDB;

  /**
   * @brief Potentially evict features based on configured strategy.
   *
   * This method is called in between get and set to check if feature eviction
   * should be triggered based on the configured eviction strategy.
   */
  virtual void maybe_evict() = 0;

  /**
   * @brief Get the memory usage of the map.
   *
   * @return Size of memory used by the map in bytes.
   */
  virtual size_t get_map_used_memsize() const = 0;

  /**
   * @brief Get feature eviction metrics.
   *
   * @return Metrics about feature eviction including counts and durations.
   */
  virtual FeatureEvictMetricTensors get_feature_evict_metric() const = 0;
};

} // namespace kv_mem
