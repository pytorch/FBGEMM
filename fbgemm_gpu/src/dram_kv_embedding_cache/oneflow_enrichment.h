/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <folly/String.h>
#include <folly/container/F14Map.h>
#include <folly/container/F14Set.h>
#include <folly/coro/Collect.h>
#include <folly/coro/Task.h>
#include <folly/coro/Timeout.h>
#include <folly/logging/xlog.h>
#include <torch/torch.h>
#include <optional>
#include <string>
#include <vector>

#include "enrichment_config.h"
#include "igr_enrichment.h"
#include "multifeed/leaf5/maple/client/coro/MapleClient.h"
#include "multifeed/opentab/db/reader/MapleReader.h"
#include "multifeed/opentab/db/reader/ObjectReaderImpl.h"
#include "multifeed/opentab/schema/ConfiguratorSchemaManagerProvider.h"

namespace oneflow_enrichment {

/// Parse comma-separated string into vector<int32_t>
inline std::vector<int32_t> parseCommaSeparatedInts(const std::string& s) {
  std::vector<int32_t> result;
  if (s.empty()) {
    return result;
  }
  std::vector<folly::StringPiece> parts;
  folly::split(',', s, parts);
  result.reserve(parts.size());
  for (const auto& part : parts) {
    result.push_back(folly::to<int32_t>(folly::trimWhitespace(part)));
  }
  return result;
}

/// Initialize OpenTab/Maple reader using EnrichmentConfig parameters.
/// Based on SideInfoInjectFromOpenTab.cpp UDF pattern.
inline std::shared_ptr<facebook::multifeed::opentab::ObjectReader>
initializeOpenTabReader(const kv_mem::EnrichmentConfig& config) {
  auto reader = std::make_shared<
      facebook::multifeed::opentab::SimpleObjectReader>(
      folly::make_not_null_unique<facebook::multifeed::opentab::MapleReader>(
          folly::make_not_null_unique<facebook::maple::MapleClient>(
              config.opentab_tier_name_)),
      facebook::multifeed::opentab::schema::ConfiguratorSchemaManagerProvider::
          getInstance());

  XLOG(INFO) << "[EmbeddingCacheEnrich] OpenTab reader initialized: "
             << "tier=" << config.opentab_tier_name_
             << ", client_id=" << config.client_id_
             << ", timeout_ms=" << config.opentab_timeout_ms_
             << ", batch_size=" << config.opentab_batch_size_
             << ", payload_ids=" << config.opentab_payload_ids_
             << ", column_group_ids=" << config.opentab_column_group_ids_;

  return reader;
}

/// Fetch payloads from OpenTab/Maple for given object IDs.
/// Returns map from objectId to single int64_t payload value.
/// Returns -1 for missing objects (which will be skipped during write).
/// Based on SideInfoInjectFromOpenTab.cpp UDF pattern.
inline folly::coro::Task<folly::F14FastMap<int64_t, int64_t>> fetchFromOpenTab(
    const std::shared_ptr<facebook::multifeed::opentab::ObjectReader>& reader,
    const kv_mem::EnrichmentConfig& config,
    const std::vector<int64_t>& objectIds) {
  folly::F14FastMap<int64_t, int64_t> objectIdToPayloadMap;

  auto column_group_ids =
      parseCommaSeparatedInts(config.opentab_column_group_ids_);
  auto payload_ids = parseCommaSeparatedInts(config.opentab_payload_ids_);
  auto payload_types = parseCommaSeparatedInts(config.opentab_payload_types_);
  auto vec_payload_indexes =
      parseCommaSeparatedInts(config.opentab_vec_payload_indexes_);

  if (!reader || objectIds.empty() || column_group_ids.empty()) {
    co_return objectIdToPayloadMap;
  }

  // Dedupe object IDs
  folly::F14FastSet<int64_t> dedupedIds(objectIds.begin(), objectIds.end());

  // Create batches of OpenTabKeys
  std::vector<std::vector<facebook::multifeed::opentab::OpenTabKey>> keyBatches;
  const size_t batchSize = static_cast<size_t>(config.opentab_batch_size_);
  const size_t numBatches = (dedupedIds.size() + batchSize - 1) / batchSize;
  keyBatches.reserve(numBatches);

  auto it = dedupedIds.begin();
  const auto end = dedupedIds.end();

  while (it != end) {
    std::vector<facebook::multifeed::opentab::OpenTabKey> batch;
    const size_t remainingKeys = std::distance(it, end);
    const size_t keysInThisBatch = std::min(batchSize, remainingKeys);
    batch.reserve(keysInThisBatch * column_group_ids.size());

    for (size_t count = 0; count < batchSize && it != end; ++count, ++it) {
      const int64_t objectId = *it;
      for (int32_t columnGroupId : column_group_ids) {
        batch.emplace_back(objectId, columnGroupId);
      }
    }
    keyBatches.push_back(std::move(batch));
  }

  XLOG(INFO) << "[EmbeddingCacheEnrich] OpenTab fetching " << dedupedIds.size()
             << " unique IDs in " << keyBatches.size() << " batches";

  // Create futures for all batch requests
  std::vector<folly::coro::Task<
      facebook::multifeed::opentab::ObjectReadObjectSummariesResult>>
      batchTasks;
  batchTasks.reserve(keyBatches.size());

  for (auto&& batch : keyBatches) {
    batchTasks.push_back(reader->coReadIntoObjectSummaries(
        std::move(batch),
        0 /* retention */,
        0 /* queryTimepoint */,
        config.client_id_,
        column_group_ids.size(),
        std::nullopt /* tierName */,
        {} /* batchConfig */));
  }

  // Collect all results with timeout
  std::vector<
      folly::Try<facebook::multifeed::opentab::ObjectReadObjectSummariesResult>>
      batchResults;
  try {
    batchResults = co_await folly::coro::co_withCancellation(
        folly::CancellationToken{},
        folly::coro::timeout(
            folly::coro::collectAllTryRange(std::move(batchTasks)),
            std::chrono::milliseconds(config.opentab_timeout_ms_)));
  } catch (const folly::FutureTimeout&) {
    XLOG(WARNING) << "[EmbeddingCacheEnrich] OpenTab fetch timeout after "
                  << config.opentab_timeout_ms_ << "ms";
    co_return objectIdToPayloadMap;
  }

  // Merge results from all batches
  folly::F14FastMap<
      int64_t,
      std::shared_ptr<const facebook::multifeed::ObjectSummary>>
      objectSummaryMap;
  int64_t missingObjectCount = 0;

  for (auto& batchResult : batchResults) {
    if (batchResult.hasException()) {
      XLOG(WARNING) << "[EmbeddingCacheEnrich] OpenTab batch failed: "
                    << batchResult.exception().what();
      continue;
    }
    for (const auto& osPtr : batchResult.value().values) {
      if (osPtr != nullptr && osPtr->object_id().value() > 0) {
        objectSummaryMap.emplace(osPtr->object_id().value(), osPtr);
      } else {
        missingObjectCount++;
      }
    }
  }

  XLOG(INFO) << "[EmbeddingCacheEnrich] OpenTab fetched "
             << objectSummaryMap.size() << " objects, " << missingObjectCount
             << " missing";

  // Extract single payload value for each object
  for (const auto& [objectId, os] : objectSummaryMap) {
    int64_t payloadValue = -1; // default to -1 (will be skipped during write)

    if (!payload_ids.empty()) {
      int32_t payloadId = payload_ids[0];
      int32_t payloadType = payload_types.empty() ? 0 : payload_types[0];
      int32_t vecPayloadIndex =
          vec_payload_indexes.empty() ? 0 : vec_payload_indexes[0];

      if (payloadType == 0) {
        // INT payload type
        auto pit = os->intPayload()->find(payloadId);
        if (pit != os->intPayload()->end()) {
          payloadValue = pit->second;
        }
      } else if (payloadType == 1 || payloadType == 2) {
        // VEC payload type (1 or 2)
        auto pit = os->vecPayload()->find(payloadId);
        if (pit != os->vecPayload()->end() &&
            static_cast<size_t>(vecPayloadIndex) < pit->second.size()) {
          payloadValue = pit->second.at(vecPayloadIndex);
        }
      } else {
        XLOG_EVERY_MS(WARNING, 5000)
            << "[EmbeddingCacheEnrich] Unsupported payload type: "
            << payloadType;
      }
    }

    objectIdToPayloadMap.emplace(objectId, payloadValue);
  }

  co_return objectIdToPayloadMap;
}

/// Prepare tensors for writing int64 payloads into the cache.
/// Each int64 payload is encoded as 16 nibbles -> power-of-2 float32 (FP8-safe
/// encoding). Each 4-bit nibble n maps to: pow(2, n>>1) * (-1 if n&1 else 1)
/// All encoded values are exact in FP8 E4M3 after per-row scaling.
template <typename weight_type>
inline std::optional<igr_enrichment::EnrichmentResult>
prepareInt64PayloadTensors(
    const std::vector<int64_t>& hashed_ids,
    const std::vector<int64_t>& unhashed_ids,
    const folly::F14FastMap<int64_t, int64_t>& payloads,
    int64_t /*max_D*/) {
  if (hashed_ids.size() != unhashed_ids.size()) {
    XLOG(ERR) << "[EmbeddingCacheEnrich] hashed_ids and unhashed_ids size "
                 "mismatch: "
              << hashed_ids.size() << " vs " << unhashed_ids.size();
    return std::nullopt;
  }

  std::vector<int64_t> indices_vec;
  std::vector<float> weights_vec;

  for (size_t idx = 0; idx < unhashed_ids.size(); ++idx) {
    int64_t unhashed_id = unhashed_ids[idx];
    int64_t hashed_id = hashed_ids[idx];

    auto it = payloads.find(unhashed_id);
    if (it == payloads.end()) {
      continue;
    }

    int64_t payload_value = it->second;

    // Skip if payload is -1 or 0 (missing data)
    if (payload_value == -1 || payload_value == 0) {
      continue;
    }

    indices_vec.push_back(hashed_id);

    // Encode int64_t as 16 nibbles -> power-of-2 float32 (FP8-safe encoding)
    // Each 4-bit nibble n maps to: pow(2, n>>1) * (-1 if n&1 else 1)
    // All encoded values are exact in FP8 E4M3 after per-row scaling
    // (they map to ±3.5×2^k which are exact E4M3 representable values)
    static constexpr float kNibbleToFloat[16] = {
        1.0f,
        -1.0f,
        2.0f,
        -2.0f,
        4.0f,
        -4.0f,
        8.0f,
        -8.0f,
        16.0f,
        -16.0f,
        32.0f,
        -32.0f,
        64.0f,
        -64.0f,
        128.0f,
        -128.0f};
    uint64_t uval = static_cast<uint64_t>(payload_value);
    for (int i = 0; i < 16; ++i) {
      uint8_t nibble = (uval >> (i * 4)) & 0xF;
      weights_vec.push_back(kNibbleToFloat[nibble]);
    }
  }

  if (indices_vec.empty()) {
    XLOG(INFO) << "[EmbeddingCacheEnrich] prepareInt64PayloadTensors: "
               << "no valid payloads to write (all -1 or missing)";
    return std::nullopt;
  }

  int64_t num_embeddings = indices_vec.size();
  constexpr int64_t kEmbeddingDim = 16;

  auto indices_tensor =
      torch::from_blob(indices_vec.data(), {num_embeddings}, torch::kInt64)
          .clone();
  auto weights_tensor =
      torch::from_blob(
          weights_vec.data(), {num_embeddings, kEmbeddingDim}, torch::kFloat32)
          .clone();

  // Convert to weight_type if needed
  if constexpr (!std::is_same_v<weight_type, float>) {
    weights_tensor =
        weights_tensor.to(torch::CppTypeToScalarType<weight_type>());
  }

  auto count_tensor = torch::tensor({num_embeddings}, torch::kLong);

  XLOG(INFO) << "[EmbeddingCacheEnrich] prepareInt64PayloadTensors: "
             << "num_embeddings=" << num_embeddings
             << ", embedding_dim=" << kEmbeddingDim
             << ", total_payloads=" << payloads.size()
             << ", skipped=" << (payloads.size() - num_embeddings);

  return igr_enrichment::EnrichmentResult{
      indices_tensor,
      weights_tensor,
      count_tensor,
  };
}

} // namespace oneflow_enrichment
