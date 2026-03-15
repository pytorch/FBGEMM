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
#include <folly/io/IOBuf.h>
#include <folly/logging/xlog.h>
#include <laser/client/cpp2/LaserClient.h>
#include <thrift/lib/cpp2/protocol/CompactProtocol.h>
#include <torch/torch.h>
#include <cstring>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "enrichment_config.h"

namespace igr_enrichment {

/// Tensors ready to be written into the embedding cache.
struct EnrichmentResult {
  at::Tensor indices; // [N] int64 hashed IDs
  at::Tensor weights; // [N, max_D]
  at::Tensor count; // [1]
};

/// Generic thrift parser: extract list<float> from Compact Protocol bytes.
/// Works for any thrift struct of the form { 1: optional list<float> field }.
/// No generated thrift header is needed.
inline std::optional<std::vector<float>> parseThriftListFloat(
    const std::string& raw) {
  if (raw.empty()) {
    return std::nullopt;
  }
  try {
    auto buf = folly::IOBuf::wrapBuffer(raw.data(), raw.size());
    apache::thrift::CompactProtocolReader reader;
    reader.setInput(buf.get());

    std::string name;
    reader.readStructBegin(name);

    std::string fieldName;
    int16_t fieldId;
    apache::thrift::protocol::TType fieldType;
    reader.readFieldBegin(fieldName, fieldType, fieldId);

    if (fieldId == 1 && fieldType == apache::thrift::protocol::TType::T_LIST) {
      apache::thrift::protocol::TType elemType;
      uint32_t listSize;
      reader.readListBegin(elemType, listSize);
      std::vector<float> result(listSize);
      for (uint32_t i = 0; i < listSize; ++i) {
        if (elemType == apache::thrift::protocol::TType::T_FLOAT) {
          reader.readFloat(result[i]);
        } else if (elemType == apache::thrift::protocol::TType::T_DOUBLE) {
          double d;
          reader.readDouble(d);
          result[i] = static_cast<float>(d);
        } else {
          return std::nullopt;
        }
      }
      reader.readListEnd();
      return result;
    }
  } catch (const std::exception& e) {
    XLOGF_EVERY_MS(
        ERR,
        6000,
        "Exception in parseThriftListFloat: {}",
        folly::exceptionStr(e));
  }
  return std::nullopt;
}

/// Parse embedding from text format "[v1,v2,v3]" with comma or \x02 delimiters.
inline void parseEmbeddingJson(
    std::string_view json,
    std::vector<float>& embedding,
    int64_t dim) {
  static constexpr char kScribeDelim = '\x02';
  static constexpr char kHiveDelim = ',';

  embedding.reserve(dim);

  if (!json.empty() && json.front() == '[') {
    json.remove_prefix(1);
  }
  if (!json.empty() && json.back() == ']') {
    json.remove_suffix(1);
  }

  char delim = kHiveDelim;
  if (json.find(kScribeDelim) != std::string_view::npos) {
    delim = kScribeDelim;
  }

  folly::split(delim, json, embedding);
}

/// Generic thrift parser: extract list<i64> from Compact Protocol bytes.
/// Works for any thrift struct of the form { 1: optional list<i64> field }.
inline std::optional<std::vector<int64_t>> parseThriftListInt64(
    const std::string& raw) {
  if (raw.empty()) {
    return std::nullopt;
  }
  try {
    auto buf = folly::IOBuf::wrapBuffer(raw.data(), raw.size());
    apache::thrift::CompactProtocolReader reader;
    reader.setInput(buf.get());

    std::string name;
    reader.readStructBegin(name);

    std::string fieldName;
    int16_t fieldId;
    apache::thrift::protocol::TType fieldType;
    reader.readFieldBegin(fieldName, fieldType, fieldId);

    if (fieldId == 1 && fieldType == apache::thrift::protocol::TType::T_LIST) {
      apache::thrift::protocol::TType elemType;
      uint32_t listSize;
      reader.readListBegin(elemType, listSize);
      std::vector<int64_t> result(listSize);
      for (uint32_t i = 0; i < listSize; ++i) {
        reader.readI64(result[i]);
      }
      reader.readListEnd();
      return result;
    }
  } catch (const std::exception& e) {
    XLOGF_EVERY_MS(
        ERR,
        6000,
        "Exception in parseThriftListInt64: {}",
        folly::exceptionStr(e));
  }
  return std::nullopt;
}

/// Parse SID from text format "[v1,v2,v3]" with comma or \x02 delimiters.
/// Each value is an int64_t.
inline void parseSIDJson(
    std::string_view json,
    std::vector<int64_t>& sids,
    int64_t expected_count) {
  static constexpr char kScribeDelim = '\x02';
  static constexpr char kHiveDelim = ',';

  sids.reserve(expected_count);

  if (!json.empty() && json.front() == '[') {
    json.remove_prefix(1);
  }
  if (!json.empty() && json.back() == ']') {
    json.remove_suffix(1);
  }

  char delim = kHiveDelim;
  if (json.find(kScribeDelim) != std::string_view::npos) {
    delim = kScribeDelim;
  }

  folly::split(delim, json, sids);
}

/// Fetch SIDs (int64 values) from Laser for given object IDs.
/// Uses config.response_format_ to choose deserialization:
///   THRIFT_INT64 — generic CompactProtocol reader for list<i64>
///   JSON — text "[v1,v2,v3,v4]" format
inline folly::coro::Task<folly::F14FastMap<int64_t, std::vector<int64_t>>>
fetchSIDsFromLaser(
    const std::shared_ptr<facebook::laser::LaserClient>& laserClient,
    const kv_mem::EnrichmentConfig& config,
    const std::vector<int64_t>& objectIds) {
  folly::F14FastMap<int64_t, std::vector<int64_t>> objectIdToSIDMap;
  if (!laserClient || objectIds.empty()) {
    co_return objectIdToSIDMap;
  }

  // enrichment_dim is total FP16 count; each int64 = 4 FP16, so
  // expected SID count = enrichment_dim / 4
  const int64_t expectedSIDCount = config.enrichment_dim_ / 4;

  std::vector<std::string> laserKeys;
  laserKeys.reserve(objectIds.size());
  for (const auto& id : objectIds) {
    laserKeys.push_back(std::to_string(id));
  }

  try {
    auto laserMultiRet = co_await laserClient->coMultiget(std::move(laserKeys));

    if (laserMultiRet.empty()) {
      co_return objectIdToSIDMap;
    }

    for (size_t i = 0; i < laserMultiRet.size(); ++i) {
      const auto& laserRet = laserMultiRet[i];
      const int64_t objectId = objectIds[i];

      if (!laserRet.valueExists()) {
        continue;
      }

      if (config.response_format_ ==
          kv_mem::EnrichmentResponseFormat::THRIFT_INT64) {
        auto parsed = parseThriftListInt64(laserRet.getValue());
        if (parsed.has_value() &&
            static_cast<int64_t>(parsed->size()) == expectedSIDCount) {
          objectIdToSIDMap[objectId] = std::move(*parsed);
        } else {
          XLOGF_EVERY_MS(
              WARN,
              6000,
              "SID count mismatch for objectId {}: expected {}, got {} "
              "(thrift_int64 format)",
              objectId,
              expectedSIDCount,
              parsed.has_value() ? parsed->size() : 0);
        }
      } else {
        // Default: JSON/text format "[v1,v2,v3,v4]"
        std::vector<int64_t> sids;
        parseSIDJson(laserRet.getValue(), sids, expectedSIDCount);
        if (static_cast<int64_t>(sids.size()) == expectedSIDCount) {
          objectIdToSIDMap.emplace(objectId, std::move(sids));
        } else {
          XLOGF_EVERY_MS(
              WARN,
              6000,
              "SID count mismatch for objectId {}: expected {}, got {} "
              "(JSON format)",
              objectId,
              expectedSIDCount,
              sids.size());
        }
      }
    }
  } catch (const std::exception& e) {
    XLOGF_EVERY_MS(
        ERR,
        6000,
        "Exception when fetching SIDs from laser for provider {} "
        "client {}: {}",
        config.provider_name_,
        config.client_id_,
        folly::exceptionStr(e));
  }

  co_return objectIdToSIDMap;
}

/// Prepare tensors for writing SIDs into the cache.
/// Each int64_t SID is bit-cast to 4 x at::Half (FP16), preserving exact bits
/// for lossless round-trip. The output tensor is always kHalf regardless of
/// weight_type, since we need exact bit representation.
inline std::optional<EnrichmentResult> prepareSIDCacheWriteTensors(
    const std::vector<int64_t>& hashed_ids,
    const std::vector<int64_t>& unhashed_ids,
    const folly::F14FastMap<int64_t, std::vector<int64_t>>& sid_map,
    int64_t max_D) {
  if (hashed_ids.size() != unhashed_ids.size()) {
    XLOG(ERR) << "hashed_ids and unhashed_ids size mismatch: "
              << hashed_ids.size() << " vs " << unhashed_ids.size();
    return std::nullopt;
  }

  std::vector<int64_t> indices_vec;
  std::vector<at::Half> weights_vec;
  indices_vec.reserve(sid_map.size());

  for (size_t i = 0; i < unhashed_ids.size(); ++i) {
    int64_t unhashed_id = unhashed_ids[i];
    int64_t hashed_id = hashed_ids[i];

    auto it = sid_map.find(unhashed_id);
    if (it == sid_map.end()) {
      continue;
    }

    const auto& sids = it->second;
    if (sids.empty()) {
      continue;
    }

    indices_vec.push_back(hashed_id);

    // Each int64 SID → 4 x FP16 via bit-cast (memcpy)
    // Total FP16 values = sids.size() * 4
    int64_t fp16_written = 0;
    for (const auto& sid : sids) {
      at::Half fp16_vals[4];
      std::memcpy(fp16_vals, &sid, sizeof(int64_t));
      for (int k = 0; k < 4; ++k) {
        if (fp16_written < max_D) {
          weights_vec.push_back(fp16_vals[k]);
          ++fp16_written;
        }
      }
    }
    // Pad remaining slots to max_D
    for (; fp16_written < max_D; ++fp16_written) {
      weights_vec.push_back(at::Half(0));
    }
  }

  if (indices_vec.empty()) {
    return std::nullopt;
  }

  int64_t num_embeddings = indices_vec.size();

  auto indices_tensor = std::make_shared<at::Tensor>(
      torch::from_blob(indices_vec.data(), {num_embeddings}, torch::kInt64)
          .clone());
  auto weights_tensor = std::make_shared<at::Tensor>(
      torch::from_blob(
          weights_vec.data(), {num_embeddings, max_D}, torch::kHalf)
          .clone());

  auto count_tensor = std::make_shared<at::Tensor>(
      torch::tensor({num_embeddings}, torch::kLong));

  XLOG(INFO) << "[EmbeddingCacheEnrich] prepareSIDCacheWriteTensors: "
             << "num_embeddings=" << num_embeddings;

  return EnrichmentResult{
      *indices_tensor,
      *weights_tensor,
      *count_tensor,
  };
}

/// Create a reusable LaserClient from enrichment config.
/// Called once in the constructor so the client is shared across all fetches.
inline std::shared_ptr<facebook::laser::LaserClient> initializeLaserClient(
    const kv_mem::EnrichmentConfig& config) {
  return std::make_shared<facebook::laser::LaserClient>(
      config.provider_name_,
      nullptr /* executor */,
      facebook::laser::LaserClient::Options().setClientId(config.client_id_));
}

/// Fetch embeddings from Laser for given object IDs.
/// Uses config.response_format_ to choose deserialization:
///   "thrift_float" — generic CompactProtocol reader (no generated header)
///   "json" — text "[v1,v2,v3]" format
inline folly::coro::Task<folly::F14FastMap<int64_t, std::vector<float>>>
fetchEmbeddingsFromLaser(
    const std::shared_ptr<facebook::laser::LaserClient>& laserClient,
    const kv_mem::EnrichmentConfig& config,
    const std::vector<int64_t>& objectIds) {
  folly::F14FastMap<int64_t, std::vector<float>> objectIdToEmbeddingMap;
  if (!laserClient || objectIds.empty()) {
    co_return objectIdToEmbeddingMap;
  }

  std::vector<std::string> laserKeys;
  laserKeys.reserve(objectIds.size());
  for (const auto& id : objectIds) {
    laserKeys.push_back(std::to_string(id));
  }

  try {
    auto laserMultiRet = co_await laserClient->coMultiget(std::move(laserKeys));

    if (laserMultiRet.empty()) {
      co_return objectIdToEmbeddingMap;
    }

    for (size_t i = 0; i < laserMultiRet.size(); ++i) {
      const auto& laserRet = laserMultiRet[i];
      const int64_t objectId = objectIds[i];

      if (!laserRet.valueExists()) {
        continue;
      }

      std::vector<float> embedding;

      if (config.response_format_ ==
          kv_mem::EnrichmentResponseFormat::THRIFT_FLOAT) {
        auto parsed = parseThriftListFloat(laserRet.getValue());
        if (parsed.has_value() &&
            parsed->size() == static_cast<size_t>(config.enrichment_dim_)) {
          objectIdToEmbeddingMap[objectId] = std::move(*parsed);
        }
      } else {
        // Default: JSON/text format
        parseEmbeddingJson(
            laserRet.getValue(), embedding, config.enrichment_dim_);
        if (embedding.size() == static_cast<size_t>(config.enrichment_dim_)) {
          objectIdToEmbeddingMap.emplace(objectId, std::move(embedding));
        }
      }
    }
  } catch (const std::exception& e) {
    XLOGF_EVERY_MS(
        ERR,
        6000,
        "Exception when fetching embeddings from laser for provider {} "
        "client {}: {}",
        config.provider_name_,
        config.client_id_,
        folly::exceptionStr(e));
  }

  co_return objectIdToEmbeddingMap;
}

/// Prepare tensors for writing embeddings into the cache.
/// Pads or truncates each embedding to max_D and converts to weight_type.
template <typename weight_type>
inline std::optional<EnrichmentResult> prepareCacheWriteTensors(
    const std::vector<int64_t>& hashed_ids,
    const std::vector<int64_t>& unhashed_ids,
    const folly::F14FastMap<int64_t, std::vector<float>>& embeddings,
    int64_t max_D) {
  if (hashed_ids.size() != unhashed_ids.size()) {
    XLOG(ERR) << "hashed_ids and unhashed_ids size mismatch: "
              << hashed_ids.size() << " vs " << unhashed_ids.size();
    return std::nullopt;
  }

  std::vector<int64_t> indices_vec;
  std::vector<float> weights_vec;
  indices_vec.reserve(embeddings.size());

  for (size_t i = 0; i < unhashed_ids.size(); ++i) {
    int64_t unhashed_id = unhashed_ids[i];
    int64_t hashed_id = hashed_ids[i];

    auto it = embeddings.find(unhashed_id);
    if (it == embeddings.end()) {
      continue;
    }

    const auto& embedding = it->second;
    if (embedding.empty()) {
      continue;
    }

    indices_vec.push_back(hashed_id);

    // Pad or truncate to max_D
    for (int64_t j = 0; j < max_D; ++j) {
      if (j < static_cast<int64_t>(embedding.size())) {
        weights_vec.push_back(embedding[j]);
      } else {
        weights_vec.push_back(0.0f);
      }
    }
  }

  if (indices_vec.empty()) {
    return std::nullopt;
  }

  int64_t num_embeddings = indices_vec.size();

  auto indices_tensor = std::make_shared<at::Tensor>(
      torch::from_blob(indices_vec.data(), {num_embeddings}, torch::kInt64)
          .clone());
  auto weights_tensor = std::make_shared<at::Tensor>(
      torch::from_blob(
          weights_vec.data(), {num_embeddings, max_D}, torch::kFloat32)
          .clone());

  // Convert to weight_type if needed
  if constexpr (!std::is_same_v<weight_type, float>) {
    *weights_tensor =
        weights_tensor->to(torch::CppTypeToScalarType<weight_type>());
  }

  auto count_tensor = std::make_shared<at::Tensor>(
      torch::tensor({num_embeddings}, torch::kLong));

  XLOG(INFO) << "[EmbeddingCacheEnrich] prepareCacheWriteTensors: "
             << "num_embeddings=" << num_embeddings;

  // Return by value; the shared_ptrs ensure tensor data survives
  // until the caller (set_kv_db_async_on_laser_executor) clones or
  // finishes consuming them.  We move into the result struct so
  // the caller can capture the shared_ptrs in the continuation.
  return EnrichmentResult{
      *indices_tensor,
      *weights_tensor,
      *count_tensor,
  };
}

} // namespace igr_enrichment
