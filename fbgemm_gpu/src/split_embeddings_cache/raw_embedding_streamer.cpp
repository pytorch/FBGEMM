/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef FBGEMM_FBCODE
#include <folly/coro/BlockingWait.h>
#include <folly/stop_watch.h>
#include <utility>
#include "aiplatform/gmpp/experimental/training_ps/gen-cpp2/TrainingParameterServerService.h"
#include "caffe2/torch/fb/distributed/wireSerializer/WireSerializer.h"
#include "servicerouter/client/cpp2/ClientParams.h"
#include "servicerouter/client/cpp2/ServiceRouter.h"
#include "torch/csrc/autograd/record_function_ops.h"
#include "torch/types.h"
#endif
#include "fbgemm_gpu/split_embeddings_cache/raw_embedding_streamer.h"
#include "fbgemm_gpu/utils/dispatch_macros.h"

namespace fbgemm_gpu {
namespace {

#ifdef FBGEMM_FBCODE
/*
 * Get the thrift client to the training parameter server service
 * There is a destruction double free issue when wrapping the member
 * variable under ifdef, and creating client is relatively cheap, so create this
 * helper function to get the client just before sending requests.
 */
std::unique_ptr<
    apache::thrift::Client<aiplatform::gmpp::experimental::training_ps::
                               TrainingParameterServerService>>
get_res_client(int64_t res_server_port) {
  auto& factory = facebook::servicerouter::cpp2::getClientFactory();
  auto params =
      folly::copy(facebook::servicerouter::ClientParams().setSingleHost(
          "::", res_server_port));
  return factory.getSRClientUnique<
      apache::thrift::Client<aiplatform::gmpp::experimental::training_ps::
                                 TrainingParameterServerService>>(
      "realtime.delta.publish.esr", params);
}
#endif

/// Read a scalar value from a tensor that is maybe a UVM tensor
/// Note that `tensor.item<type>()` is not allowed on a UVM tensor in
/// PyTorch
inline int64_t get_maybe_uvm_scalar(const at::Tensor& tensor) {
  return tensor.scalar_type() == at::ScalarType::Long
      ? *(tensor.data_ptr<int64_t>())
      : *(tensor.data_ptr<int32_t>());
}

fbgemm_gpu::StreamQueueItem tensor_copy(
    const at::Tensor& indices,
    const at::Tensor& weights,
    std::optional<at::Tensor> identities,
    const at::Tensor& count) {
  auto num_sets = get_maybe_uvm_scalar(count);
  auto new_indices = at::empty(
      num_sets, at::TensorOptions().device(at::kCPU).dtype(indices.dtype()));
  auto new_weights = at::empty(
      {num_sets, weights.size(1)},
      at::TensorOptions().device(at::kCPU).dtype(weights.dtype()));
  std::optional<at::Tensor> new_identities = std::nullopt;
  if (identities.has_value()) {
    new_identities = at::empty(
        num_sets,
        at::TensorOptions().device(at::kCPU).dtype(identities->dtype()));
  }
  auto new_count =
      at::empty({1}, at::TensorOptions().device(at::kCPU).dtype(at::kLong));
  FBGEMM_DISPATCH_FLOAT_HALF_AND_BYTE(
      weights.scalar_type(), "tensor_copy", [&] {
        using value_t = scalar_t;
        FBGEMM_DISPATCH_INTEGRAL_TYPES(
            indices.scalar_type(), "tensor_copy", [&] {
              using index_t = scalar_t;
              auto indices_addr = indices.data_ptr<index_t>();
              auto new_indices_addr = new_indices.data_ptr<index_t>();
              std::copy(
                  indices_addr,
                  indices_addr + num_sets,
                  new_indices_addr); // dst_start

              auto weights_addr = weights.data_ptr<value_t>();
              auto new_weights_addr = new_weights.data_ptr<value_t>();
              std::copy(
                  weights_addr,
                  weights_addr + num_sets * weights.size(1),
                  new_weights_addr); // dst_start
              if (identities.has_value()) {
                FBGEMM_DISPATCH_INTEGRAL_TYPES(
                    identities->scalar_type(), "tensor_copy", [&] {
                      using identities_t = scalar_t;
                      auto identities_addr =
                          identities->data_ptr<identities_t>();
                      auto new_identities_addr =
                          new_identities->data_ptr<identities_t>();
                      std::copy(
                          identities_addr,
                          identities_addr + num_sets,
                          new_identities_addr); // dst_start
                    });
              }
            });
      });
  *new_count.data_ptr<int64_t>() = num_sets;
  return fbgemm_gpu::StreamQueueItem{
      new_indices, new_weights, new_identities, new_count};
}

} // namespace

RawEmbeddingStreamer::RawEmbeddingStreamer(
    std::string unique_id,
    bool enable_raw_embedding_streaming,
    int64_t res_store_shards,
    int64_t res_server_port,
    std::vector<std::string> table_names,
    std::vector<int64_t> table_offsets,
    const std::vector<int64_t>& table_sizes)
    : unique_id_(std::move(unique_id)),
      enable_raw_embedding_streaming_(enable_raw_embedding_streaming),
      res_store_shards_(res_store_shards),
      res_server_port_(res_server_port),
      table_names_(std::move(table_names)),
      table_offsets_(std::move(table_offsets)),
      table_sizes_(at::tensor(table_sizes)) {
#ifdef FBGEMM_FBCODE
  if (enable_raw_embedding_streaming_) {
    XLOG(INFO) << "[TBE_ID" << unique_id_
               << "] Raw embedding streaming enabled with res_server_port at"
               << res_server_port;
    // The first call to get the client is expensive, so eagerly get it here
    auto _eager_client = get_res_client(res_server_port_);

    weights_stream_thread_ = std::make_unique<std::thread>([=, this] {
      while (!stop_) {
        auto stream_item_ptr = weights_to_stream_queue_.try_peek();
        if (!stream_item_ptr) {
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
          continue;
        }
        if (stop_) {
          return;
        }
        auto& indices = stream_item_ptr->indices;
        auto& weights = stream_item_ptr->weights;
        auto& identities = stream_item_ptr->identities;
        folly::stop_watch<std::chrono::milliseconds> stop_watch;
        folly::coro::blockingWait(tensor_stream(indices, weights, identities));

        weights_to_stream_queue_.dequeue();
        XLOG_EVERY_MS(INFO, 60000)
            << "[TBE_ID" << unique_id_
            << "] end stream queue size: " << weights_to_stream_queue_.size()
            << " stream takes " << stop_watch.elapsed().count() << "ms";
      }
    });
  }
#endif
}

RawEmbeddingStreamer::~RawEmbeddingStreamer() {
  stop_ = true;
#ifdef FBGEMM_FBCODE
  if (enable_raw_embedding_streaming_) {
    join_stream_tensor_copy_thread();
    join_weights_stream_thread();
  }
#endif
}

void RawEmbeddingStreamer::stream(
    const at::Tensor& indices,
    const at::Tensor& weights,
    std::optional<at::Tensor> identities,
    const at::Tensor& count,
    bool require_tensor_copy,
    bool blocking_tensor_copy) {
  if (!enable_raw_embedding_streaming_) {
    return;
  }
#ifdef FBGEMM_FBCODE
  auto rec = torch::autograd::profiler::record_function_enter_new(
      "## RawEmbeddingStreamer::stream_callback ##");
  if (!require_tensor_copy) {
    StreamQueueItem stream_item(indices, weights, std::move(identities), count);
    weights_to_stream_queue_.enqueue(stream_item);
    return;
  }
  if (blocking_tensor_copy) {
    copy_and_enqueue_stream_tensors(
        indices, weights, std::move(identities), count);
    return;
  }
  // Make sure the previous thread is done before starting a new one
  join_stream_tensor_copy_thread();
  // Cuda dispatches the host callbacks all in the same CPU thread. But the
  // callbacks don't need to be serialized.
  // So, We need to spin up a new thread to unblock the CUDA stream, so the CUDA
  // can continue executing other host callbacks, eg. get/evict.
  stream_tensor_copy_thread_ = std::make_unique<std::thread>([=, this]() {
    copy_and_enqueue_stream_tensors(indices, weights, identities, count);
  });
  rec->record.end();
#endif
}

#ifdef FBGEMM_FBCODE
folly::coro::Task<void> RawEmbeddingStreamer::tensor_stream(
    const at::Tensor& indices,
    const at::Tensor& weights,
    std::optional<at::Tensor> identities) {
  using namespace ::aiplatform::gmpp::experimental::training_ps;
  if (indices.size(0) != weights.size(0)) {
    XLOG(ERR) << "[TBE_ID" << unique_id_
              << "] Indices and weights size mismatched " << indices.size(0)
              << " " << weights.size(0);
    co_return;
  }
  folly::stop_watch<std::chrono::milliseconds> stop_watch;
  XLOG_EVERY_MS(INFO, 60000)
      << "[TBE_ID" << unique_id_
      << "] send streaming request: indices = " << indices.size(0)
      << ", weights = " << weights.size(0) << ", identities =  "
      << (identities.has_value() ? std::to_string(identities->size(0))
                                 : "none");

  auto biggest_idx = table_sizes_.index({table_sizes_.size(0) - 1});
  auto mask =
      at::logical_and(indices >= 0, indices < biggest_idx).nonzero().squeeze();
  auto filtered_indices = indices.index_select(0, mask);
  auto filtered_weights = weights.index_select(0, mask);
  std::optional<at::Tensor> filtered_identities = std::nullopt;
  if (identities.has_value()) {
    filtered_identities = identities->index_select(0, mask);
  }
  auto num_invalid_indices = indices.size(0) - filtered_indices.size(0);
  if (num_invalid_indices > 0) {
    XLOG(INFO) << "[TBE_ID" << unique_id_
               << "] number of invalid indices: " << num_invalid_indices;
  }
  // 1. Transform local row indices to embedding table global row indices
  at::Tensor table_indices =
      (at::searchsorted(table_sizes_, filtered_indices, false, true) - 1)
          .to(torch::kInt8);
  auto tb_ac = table_indices.accessor<int8_t, 1>();
  auto indices_ac = filtered_indices.accessor<int64_t, 1>();
  auto tb_sizes_ac = table_sizes_.accessor<int64_t, 1>();
  std::vector<int64_t> global_indices(tb_ac.size(0), 0);
  std::vector<int16_t> shard_indices(tb_ac.size(0), 0);

  for (int i = 0; i < tb_ac.size(0); ++i) {
    auto tb_idx = tb_ac[i];
    global_indices[i] =
        indices_ac[i] - tb_sizes_ac[tb_idx] + table_offsets_[tb_idx];
    // hash to shard
    // if we do row range sharding, also shard here.
    auto fqn = table_names_[tb_idx];
    auto hash_key = folly::to<std::string>(fqn, global_indices[i]);
    auto shard_id =
        furcHash(hash_key.data(), hash_key.size(), res_store_shards_);
    shard_indices[i] = shard_id;
  }
  auto global_indices_tensor = at::tensor(global_indices);
  auto shard_indices_tensor = at::tensor(shard_indices);
  auto total_rows = global_indices_tensor.size(0);
  XLOG_EVERY_MS(INFO, 60000)
      << "[TBE_ID" << unique_id_ << "] hash and gloablize rows " << total_rows
      << " in: " << stop_watch.elapsed().count() << "ms";
  stop_watch.reset();

  auto res_client = get_res_client(res_server_port_);
  // 2. Split by shards
  for (int i = 0; i < res_store_shards_; ++i) {
    auto shrad_mask = shard_indices_tensor.eq(i).nonzero().squeeze();
    auto table_indices_masked = table_indices.index_select(0, shrad_mask);
    auto rows_in_shard = table_indices_masked.numel();
    if (rows_in_shard == 0) {
      continue;
    }
    auto global_indices_masked =
        global_indices_tensor.index_select(0, shrad_mask);
    auto weights_masked = filtered_weights.index_select(0, shrad_mask);

    if (weights_masked.size(0) != rows_in_shard ||
        global_indices_masked.numel() != rows_in_shard) {
      XLOG(ERR)
          << "[TBE_ID" << unique_id_
          << "] don't send the request for size mismatched tensors table: "
          << rows_in_shard << " weights: " << weights_masked.size(0)
          << " global_indices: " << global_indices_masked.numel();
      continue;
    }
    SetEmbeddingsRequest req;
    req.shardId() = i;
    req.fqns() = table_names_;

    req.tableIndices() =
        torch::distributed::wireDumpTensor(table_indices_masked);
    req.rowIndices() =
        torch::distributed::wireDumpTensor(global_indices_masked);
    req.weights() = torch::distributed::wireDumpTensor(weights_masked);
    if (filtered_identities.has_value()) {
      auto identities_masked = filtered_identities->index_select(0, shrad_mask);
      req.identities() = torch::distributed::wireDumpTensor(identities_masked);
    }
    co_await res_client->co_setEmbeddings(req);
  }
  co_return;
}

void RawEmbeddingStreamer::copy_and_enqueue_stream_tensors(
    const at::Tensor& indices,
    const at::Tensor& weights,
    std::optional<at::Tensor> identities,
    const at::Tensor& count) {
  auto rec = torch::autograd::profiler::record_function_enter_new(
      "## RawEmbeddingStreamer::copy_and_enqueue_stream_tensors ##");
  auto stream_item =
      tensor_copy(indices, weights, std::move(identities), count);
  weights_to_stream_queue_.enqueue(stream_item);
  rec->record.end();
}

void RawEmbeddingStreamer::join_stream_tensor_copy_thread() {
  auto rec = torch::autograd::profiler::record_function_enter_new(
      "## RawEmbeddingStreamer::join_stream_tensor_copy_thread ##");
  if (stream_tensor_copy_thread_ != nullptr &&
      stream_tensor_copy_thread_->joinable()) {
    stream_tensor_copy_thread_->join();
  }
  rec->record.end();
}

void RawEmbeddingStreamer::join_weights_stream_thread() {
  if (weights_stream_thread_ != nullptr && weights_stream_thread_->joinable()) {
    stop_ = true;
    weights_stream_thread_->join();
  }
}

uint64_t RawEmbeddingStreamer::get_weights_to_stream_queue_size() {
  return weights_to_stream_queue_.size();
}
#endif

} // namespace fbgemm_gpu
