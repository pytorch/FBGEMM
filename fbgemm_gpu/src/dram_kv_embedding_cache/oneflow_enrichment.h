/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <torch/torch.h>
#include <optional>

#include "enrichment_config.h"
#include "igr_enrichment.h"
#include "opentab_backend_registry.h"

namespace oneflow_enrichment {

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
