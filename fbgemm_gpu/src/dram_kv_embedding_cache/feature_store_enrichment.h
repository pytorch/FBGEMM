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
#include <vector>

#include "enrichment_config.h"

namespace feature_store_enrichment {

/// Fetch SID from Feature Store (Knowledge Feature API) for given object IDs.
/// Returns map from objectId to int64_t SID payload value.
folly::coro::Task<folly::F14FastMap<int64_t, int64_t>> fetchSIDFromFeatureStore(
    const kv_mem::EnrichmentConfig& config,
    const std::vector<int64_t>& objectIds);

} // namespace feature_store_enrichment
