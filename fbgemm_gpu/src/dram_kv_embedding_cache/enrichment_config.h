/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <torch/script.h>
#include <string>

namespace kv_mem {

// TODO(xiujinl): Define these enums in Thrift to avoid maintaining duplicate
// definitions between Python and C++.

/// Must match Python EnrichmentType(IntEnum) in
/// split_table_batched_embeddings_ops_common.py
enum class EnrichmentType : int64_t {
  IGR_LASER_EMBEDDING = 0,
  IGR_LASER_SID = 1,
};

/// Must match Python EnrichmentResponseFormat(IntEnum) in
/// split_table_batched_embeddings_ops_common.py
enum class EnrichmentResponseFormat : int64_t {
  JSON = 0,
  THRIFT_FLOAT = 1,
  THRIFT_INT64 = 2,
};

/// Configuration for embedding enrichment from external sources.
/// Passed from Python as a TorchScript custom class so that switching
/// providers or enrichment methods requires no C++ rebuild.
struct EnrichmentConfig : public torch::jit::CustomClassHolder {
  /// @param enrichment_type EnrichmentType int value
  /// @param provider_name External provider name (e.g. Laser provider)
  /// @param client_id Client identifier for the external service
  /// @param enrichment_dim Dimension of data returned by the source
  /// @param response_format EnrichmentResponseFormat int value
  explicit EnrichmentConfig(
      int64_t enrichment_type,
      std::string provider_name,
      std::string client_id,
      int64_t enrichment_dim,
      int64_t response_format)
      : enrichment_type_(static_cast<EnrichmentType>(enrichment_type)),
        provider_name_(std::move(provider_name)),
        client_id_(std::move(client_id)),
        enrichment_dim_(enrichment_dim),
        response_format_(
            static_cast<EnrichmentResponseFormat>(response_format)) {}

  EnrichmentType enrichment_type_;
  std::string provider_name_;
  std::string client_id_;
  int64_t enrichment_dim_;
  EnrichmentResponseFormat response_format_;
};

} // namespace kv_mem
