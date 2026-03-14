/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <folly/container/F14Map.h>
#include <folly/container/F14Set.h>
#include <folly/coro/Task.h>
#include <folly/logging/xlog.h>
#include <servicerouter/client/cpp2/ServiceRouter.h>
#include <vector>

#include "enrichment_config.h"
#include "fblearner/feature_store/thrift/gen-cpp2/FeatureStoreComputeServiceAsyncClient.h"
#include "fblearner/feature_store/thrift/gen-cpp2/computation_input_types.h"
#include "fblearner/feature_store/thrift/gen-cpp2/compute_service_types.h"
#include "fblearner/feature_store/thrift/gen-cpp2/example_types.h"

namespace feature_store_enrichment {

// Knowledge Feature API context field IDs
// (from KnowledgeFeatureAPIFeatureStoreGeneratorSchema)
static constexpr int64_t kObjectIdField = 19766576;
static constexpr int64_t kFeatureGroupNameField = 19767797;
static constexpr int64_t kFeatureNamesField = 1138923261;
static constexpr int64_t kApiField = 1138923262;
static constexpr int64_t kSidFieldId = 1138923263;

/// Fetch SID from Feature Store (Knowledge Feature API) for given object IDs.
/// Returns map from objectId to int64_t SID payload value.
inline folly::coro::Task<folly::F14FastMap<int64_t, int64_t>>
fetchSIDFromFeatureStore(
    const kv_mem::EnrichmentConfig& config,
    const std::vector<int64_t>& objectIds) {
  folly::F14FastMap<int64_t, int64_t> objectIdToSidMap;
  if (objectIds.empty() || config.fs_tier_.empty()) {
    co_return objectIdToSidMap;
  }

  // Dedupe object IDs
  folly::F14FastSet<int64_t> dedupedIds(objectIds.begin(), objectIds.end());

  // Split into batches
  std::vector<std::vector<int64_t>> batches;
  {
    std::vector<int64_t> currentBatch;
    currentBatch.reserve(config.fs_batch_size_);
    for (auto id : dedupedIds) {
      currentBatch.push_back(id);
      if (static_cast<int64_t>(currentBatch.size()) >= config.fs_batch_size_) {
        batches.push_back(std::move(currentBatch));
        currentBatch = {};
        currentBatch.reserve(config.fs_batch_size_);
      }
    }
    if (!currentBatch.empty()) {
      batches.push_back(std::move(currentBatch));
    }
  }

  XLOG(INFO) << "[EmbeddingCacheEnrich] FeatureStore fetching "
             << dedupedIds.size() << " unique IDs in " << batches.size()
             << " batches";

  // Create client params
  auto clientParams = facebook::servicerouter::ClientParams();
  clientParams
      .setProcessingTimeoutMs(std::chrono::milliseconds(config.fs_timeout_ms_))
      .setOverallTimeoutMs(std::chrono::milliseconds(config.fs_timeout_ms_))
      .setClientId(config.fs_caller_id_);

  int64_t totalHits = 0;
  int64_t totalErrors = 0;

  for (const auto& batch : batches) {
    using namespace facebook::fblearner::feature_store::thrift;

    // Build one Example per object ID
    std::vector<example::Example> examples;
    examples.reserve(batch.size());

    for (auto objectId : batch) {
      example::Example ex;
      (*ex.int_single_categorical_features())[kObjectIdField] = objectId;
      (*ex.string_single_categorical_features())[kFeatureGroupNameField] =
          config.fs_feature_group_name_;
      (*ex.string_single_categorical_features())[kFeatureNamesField] =
          config.fs_feature_name_;
      (*ex.string_single_categorical_features())[kApiField] = "read";
      examples.push_back(std::move(ex));
    }

    // Build request
    computation_input::ComputationInput input;
    input.typed_contexts() = std::move(examples);

    compute_service::Request request;
    (*request.group_id_to_input())[config.fs_feature_group_id_] =
        std::move(input);
    request.request_context()->caller_id() = config.fs_caller_id_;

    // Send request
    try {
      auto client = facebook::servicerouter::cpp2::getClientFactory()
                        .getSRClientUnique<apache::thrift::Client<
                            compute_service::FeatureStoreComputeService>>(
                            config.fs_tier_, clientParams);

      auto response = co_await client->co_compute(request);

      // Parse results
      if (response.group_id_to_examples().has_value()) {
        auto groupIt =
            response.group_id_to_examples()->find(config.fs_feature_group_id_);
        if (groupIt != response.group_id_to_examples()->end()) {
          const auto& results = groupIt->second;
          for (size_t i = 0; i < results.size() && i < batch.size(); ++i) {
            if (results[i].getType() ==
                compute_service::ValueOrError::Type::value) {
              const auto& outputExample = results[i].get_value();
              int64_t sidValue = -1;

              // Try int_tensor_features first
              if (outputExample.int_tensor_features().has_value()) {
                auto it =
                    outputExample.int_tensor_features()->find(kSidFieldId);
                if (it != outputExample.int_tensor_features()->end() &&
                    !it->second.empty()) {
                  sidValue = it->second[0];
                }
              }
              // Fallback: try int_features
              if (sidValue == -1 && outputExample.int_features().has_value()) {
                for (const auto& [fid, val] : *outputExample.int_features()) {
                  sidValue = val;
                  break;
                }
              }
              // Fallback: try int_single_categorical_features
              if (sidValue == -1 &&
                  outputExample.int_single_categorical_features().has_value()) {
                for (const auto& [fid, val] :
                     *outputExample.int_single_categorical_features()) {
                  if (fid != kObjectIdField) {
                    sidValue = val;
                    break;
                  }
                }
              }

              if (sidValue != -1 && sidValue != 0) {
                objectIdToSidMap[batch[i]] = sidValue;
                totalHits++;
              }
            } else {
              totalErrors++;
            }
          }
        }
      }
    } catch (const std::exception& e) {
      XLOG(WARNING)
          << "[EmbeddingCacheEnrich] FeatureStore batch query failed: "
          << e.what();
      totalErrors += batch.size();
    }
  }

  XLOG(INFO) << "[EmbeddingCacheEnrich] FeatureStore result: hits=" << totalHits
             << ", errors=" << totalErrors
             << ", total_deduped=" << dedupedIds.size();

  co_return objectIdToSidMap;
}

} // namespace feature_store_enrichment
