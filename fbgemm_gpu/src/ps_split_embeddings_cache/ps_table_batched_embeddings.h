/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include "../ssd_split_embeddings_cache/kv_db_table_batched_embeddings.h"

#include <folly/experimental/coro/BlockingWait.h>
#include "mvai_infra/experimental/ps_training/tps_client/TrainingParameterServiceClient.h"

namespace ps {

class EmbeddingParameterServer : public kv_db::EmbeddingKVDB {
 public:
  explicit EmbeddingParameterServer(
      std::vector<std::pair<std::string, int>>&& tps_hosts,
      int64_t tbe_id,
      int64_t maxLocalIndexLength = 54,
      int64_t num_threads = 32)
      : tps_client_(
            std::make_shared<mvai_infra::experimental::ps_training::tps_client::
                                 TrainingParameterServiceClient>(
                std::move(tps_hosts),
                tbe_id,
                maxLocalIndexLength,
                num_threads)) {}

  void set(
      const at::Tensor& indices,
      const at::Tensor& weights,
      const at::Tensor& count) override {
    RECORD_USER_SCOPE("EmbeddingParameterServer::set");
    folly::coro::blockingWait(
        tps_client_->set(indices, weights, count.item().toLong()));
  }
  void get(
      const at::Tensor& indices,
      const at::Tensor& weights,
      const at::Tensor& count) override {
    RECORD_USER_SCOPE("EmbeddingParameterServer::get");
    folly::coro::blockingWait(
        tps_client_->get(indices, weights, count.item().toLong()));
  }
  void flush() override {}
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
