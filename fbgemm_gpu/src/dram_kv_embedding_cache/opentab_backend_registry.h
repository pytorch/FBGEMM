/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <folly/container/F14Map.h>
#include <folly/coro/Task.h>
#include <folly/logging/xlog.h>
#include <functional>
#include <memory>
#include <string>
#include <vector>

// Forward declaration — avoids pulling in torch/script.h
namespace kv_mem {
struct EnrichmentConfig;
} // namespace kv_mem

namespace oneflow_enrichment {

/// Type-erased reader pointer (actual type is ObjectReader, resolved at
/// runtime by the registered backend).
using ReaderPtr = std::shared_ptr<void>;

/// Registry for OpenTab backend implementation.
/// The actual maple/opentab implementation is registered at static-init time
/// by the oneflow_enrichment_backend library, which carries the heavy deps.
/// This decouples the main ssd_split_table_batched_embeddings target from
/// instagram/ranking transitive dependencies.
struct OpenTabBackend {
  using InitFn = std::function<ReaderPtr(const kv_mem::EnrichmentConfig&)>;
  using FetchFn =
      std::function<folly::coro::Task<folly::F14FastMap<int64_t, int64_t>>(
          const ReaderPtr&,
          const kv_mem::EnrichmentConfig&,
          const std::vector<int64_t>&)>;

  InitFn initReader;
  FetchFn fetchPayloads;

  static OpenTabBackend& instance() {
    static OpenTabBackend backend;
    return backend;
  }

  bool isRegistered() const {
    return initReader != nullptr && fetchPayloads != nullptr;
  }
};

/// Initialize OpenTab/Maple reader (calls through registered backend).
inline ReaderPtr initializeOpenTabReader(
    const kv_mem::EnrichmentConfig& config) {
  auto& backend = OpenTabBackend::instance();
  if (!backend.isRegistered()) {
    XLOG(ERR) << "[EmbeddingCacheEnrich] OpenTab backend not registered. "
              << "Ensure oneflow_enrichment_backend library is linked.";
    return nullptr;
  }
  return backend.initReader(config);
}

/// Fetch payloads from OpenTab/Maple (calls through registered backend).
inline folly::coro::Task<folly::F14FastMap<int64_t, int64_t>> fetchFromOpenTab(
    const ReaderPtr& reader,
    const kv_mem::EnrichmentConfig& config,
    const std::vector<int64_t>& objectIds) {
  auto& backend = OpenTabBackend::instance();
  if (!backend.isRegistered() || !reader) {
    co_return folly::F14FastMap<int64_t, int64_t>{};
  }
  co_return co_await backend.fetchPayloads(reader, config, objectIds);
}

} // namespace oneflow_enrichment
