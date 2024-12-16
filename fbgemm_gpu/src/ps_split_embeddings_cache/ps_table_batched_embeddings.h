/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include "../ssd_split_embeddings_cache/kv_db_table_batched_embeddings.h"

#include <folly/coro/BlockingWait.h>
#include <folly/coro/Task.h>
#include "mvai_infra/experimental/ps_training/tps_client/TrainingParameterServiceClient.h"

namespace ps {

/// @ingroup embedding-ssd
///
/// @brief An implementation of EmbeddingKVDB for Training Parameter Service
/// (TPS) client
///
class EmbeddingParameterServer : public kv_db::EmbeddingKVDB {
 public:
  explicit EmbeddingParameterServer(
      std::vector<std::pair<std::string, int>>&& tps_hosts,
      int64_t tbe_id,
      int64_t maxLocalIndexLength = 54,
      int64_t num_threads = 32,
      int64_t maxKeysPerRequest = 500,
      int64_t l2_cache_size_gb = 0,
      int64_t max_D = 0)
      : kv_db::EmbeddingKVDB(
            num_threads,
            max_D,
            l2_cache_size_gb,
            tbe_id), // update this interface
        tps_client_(
            std::make_shared<mvai_infra::experimental::ps_training::tps_client::
                                 TrainingParameterServiceClient>(
                std::move(tps_hosts),
                tbe_id,
                maxLocalIndexLength,
                num_threads,
                maxKeysPerRequest)) {}

  folly::SemiFuture<std::vector<folly::Unit>> set_kv_db_async(
      const at::Tensor& indices,
      const at::Tensor& weights,
      const at::Tensor& count,
      const kv_db::RocksdbWriteMode w_mode =
          kv_db::RocksdbWriteMode::FWD_ROCKSDB_READ) override {
    // RECORD_USER_SCOPE("EmbeddingParameterServer::set");
    // co_await tps_client_->set(indices, weights, count.item().toLong());
    return std::vector<folly::Unit>(1);
  }
  virtual folly::SemiFuture<std::vector<folly::Unit>> get_kv_db_async(
      const at::Tensor& indices,
      const at::Tensor& weights,
      const at::Tensor& count) override {
    return std::vector<folly::Unit>(1);
  }
  void flush() {}
  void compact() override {}
  // cleanup cached results in server side
  // This is a test helper, please do not use it in production
  void cleanup() {
    folly::coro::blockingWait(tps_client_->cleanup());
  }

 private:
  void flush_or_compact(const int64_t /*timestep*/) override {}

  std::shared_ptr<mvai_infra::experimental::ps_training::tps_client::
                      TrainingParameterServiceClient>
      tps_client_;
}; // class EmbeddingKVDB

} // namespace ps
