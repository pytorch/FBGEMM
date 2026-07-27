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
#include "aiplatform/gmpp/experimental/training_ps/TrainingPsOdsLogger.h"
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

// Timeout for copy_done_flag polling loop (microseconds).
constexpr int64_t kCopyDonePollTimeoutUs = 10'000'000;

// Max rows copied into one enqueued chunk.
constexpr size_t kChunkSize = 500000;
// Parallel chunk-copy threads spawned per non-blocking stream() call.
constexpr size_t kNumCopyThreads = 4;

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
  auto params = folly::copy(
      facebook::servicerouter::ClientParams().setSingleHost(
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
      ? *(tensor.const_data_ptr<int64_t>())
      : *(tensor.const_data_ptr<int32_t>());
}

} // namespace

fbgemm_gpu::StreamQueueItem tensor_copy_chunk(
    const at::Tensor& indices,
    const at::Tensor& weights,
    std::optional<at::Tensor> identities,
    std::optional<at::Tensor> runtime_meta,
    int64_t start_row,
    int64_t end_row) {
  int64_t n = end_row - start_row;
  auto new_indices =
      at::empty(n, at::TensorOptions().device(at::kCPU).dtype(indices.dtype()));
  auto new_weights = at::empty(
      {n, weights.size(1)},
      at::TensorOptions().device(at::kCPU).dtype(weights.dtype()));
  std::optional<at::Tensor> new_identities = std::nullopt;
  if (identities.has_value()) {
    new_identities = at::empty(
        {n, identities->size(1)},
        at::TensorOptions().device(at::kCPU).dtype(identities->dtype()));
  }
  std::optional<at::Tensor> new_runtime_meta = std::nullopt;
  if (runtime_meta.has_value()) {
    new_runtime_meta = at::empty(
        {n, runtime_meta->size(1)},
        at::TensorOptions().device(at::kCPU).dtype(runtime_meta->dtype()));
  }
  auto new_count =
      at::empty({1}, at::TensorOptions().device(at::kCPU).dtype(at::kLong));
  FBGEMM_DISPATCH_FLOAT_HALF_AND_BYTE(
      weights.scalar_type(), "tensor_copy_chunk", [&] {
        using value_t = scalar_t;
        FBGEMM_DISPATCH_INTEGRAL_TYPES(
            indices.scalar_type(), "tensor_copy_chunk", [&] {
              using index_t = scalar_t;
              std::copy(
                  indices.const_data_ptr<index_t>() + start_row,
                  indices.const_data_ptr<index_t>() + end_row,
                  new_indices.mutable_data_ptr<index_t>());
              std::copy(
                  weights.const_data_ptr<value_t>() +
                      start_row * weights.size(1),
                  weights.const_data_ptr<value_t>() + end_row * weights.size(1),
                  new_weights.mutable_data_ptr<value_t>());
              if (identities.has_value()) {
                FBGEMM_DISPATCH_INTEGRAL_TYPES(
                    identities->scalar_type(), "tensor_copy_chunk", [&] {
                      using id_t = scalar_t;
                      std::copy(
                          identities->const_data_ptr<id_t>() +
                              start_row * identities->size(1),
                          identities->const_data_ptr<id_t>() +
                              end_row * identities->size(1),
                          new_identities->mutable_data_ptr<id_t>());
                    });
              }
              if (runtime_meta.has_value()) {
                FBGEMM_DISPATCH_ALL_TYPES(
                    runtime_meta->scalar_type(), "tensor_copy_chunk", [&] {
                      using rm_t = scalar_t;
                      std::copy(
                          runtime_meta->const_data_ptr<rm_t>() +
                              start_row * runtime_meta->size(1),
                          runtime_meta->const_data_ptr<rm_t>() +
                              end_row * runtime_meta->size(1),
                          new_runtime_meta->mutable_data_ptr<rm_t>());
                    });
              }
            });
      });
  *new_count.mutable_data_ptr<int64_t>() = n;
  return fbgemm_gpu::StreamQueueItem{
      new_indices, new_weights, new_identities, new_runtime_meta, new_count};
}

std::vector<std::vector<std::pair<int64_t, int64_t>>>
computeChunkRanges(int64_t num_rows, size_t chunk_size, size_t num_threads) {
  // Split [0, num_rows) across up to num_threads contiguous per-thread bands,
  // then split each band into <= chunk_size chunks. Returns one inner vector of
  // [start, end) chunk ranges per thread (outer index = thread); ranges are
  // contiguous, non-overlapping, and their union is the whole range. Empty
  // bands produce no group.
  std::vector<std::vector<std::pair<int64_t, int64_t>>> thread_chunks;
  if (num_rows <= 0 || chunk_size == 0 || num_threads == 0) {
    return thread_chunks;
  }
  // ceil-div (a + b - 1) / b: rounds up so a partial final chunk/band counts.
  const size_t n_chunks =
      (static_cast<size_t>(num_rows) + chunk_size - 1) / chunk_size;
  const size_t n_threads = std::min(n_chunks, num_threads);
  const size_t rows_per_thread =
      (static_cast<size_t>(num_rows) + n_threads - 1) / n_threads;
  for (size_t ti = 0; ti < n_threads; ++ti) {
    const int64_t thread_start = static_cast<int64_t>(ti * rows_per_thread);
    const int64_t thread_end =
        std::min(static_cast<int64_t>((ti + 1) * rows_per_thread), num_rows);
    std::vector<std::pair<int64_t, int64_t>> chunks;
    for (int64_t s = thread_start; s < thread_end;
         s += static_cast<int64_t>(chunk_size)) {
      const int64_t e =
          std::min(s + static_cast<int64_t>(chunk_size), thread_end);
      chunks.emplace_back(s, e);
    }
    if (!chunks.empty()) {
      thread_chunks.push_back(std::move(chunks));
    }
  }
  return thread_chunks;
}

RawEmbeddingStreamer::RawEmbeddingStreamer(
    std::string unique_id,
    bool enable_raw_embedding_streaming,
    int64_t res_store_shards [[maybe_unused]],
    int64_t res_server_port [[maybe_unused]],
    std::vector<std::string> table_names,
    std::vector<int64_t> table_offsets,
    const std::vector<int64_t>& table_sizes)
    : unique_id_(std::move(unique_id)),
      enable_raw_embedding_streaming_(enable_raw_embedding_streaming),
#ifdef FBGEMM_FBCODE
      res_store_shards_(res_store_shards),
      res_server_port_(res_server_port),
#endif
      table_names_(std::move(table_names)),
      table_offsets_(std::move(table_offsets)),
      table_sizes_(at::tensor(table_sizes)) {
#ifdef FBGEMM_FBCODE
  if (enable_raw_embedding_streaming_) {
    XLOG(INFO) << "[TBE_ID" << unique_id_
               << "] Raw embedding streaming enabled with res_server_port at"
               << res_server_port_;
    // The first call to get the client is expensive, so eagerly get it here
    auto _eager_client = get_res_client(res_server_port_);

    ods_logger_ = std::make_unique<facebook::aiplatform::gmpp::experimental::
                                       training_ps::TrainingPsOdsLogger>();

    weights_stream_thread_ = std::make_unique<std::thread>([this] {
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
        auto& runtime_meta = stream_item_ptr->runtime_meta;
        folly::stop_watch<std::chrono::milliseconds> stop_watch;
        folly::coro::blockingWait(
            tensor_stream(indices, weights, identities, runtime_meta));

        weights_to_stream_queue_.dequeue();
        auto post_dequeue_depth = weights_to_stream_queue_.size();
        if (ods_logger_) {
          ods_logger_->bumpKeyGauge(
              "stream_mpsc_depth", static_cast<double>(post_dequeue_depth));
        }
        XLOG_EVERY_MS(INFO, 60000)
            << "[TBE_ID" << unique_id_
            << "] end stream queue size: " << post_dequeue_depth
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
    join_dispatch_thread();
    join_chunk_copy_threads();
    join_weights_stream_thread();
  }
#endif
}

void RawEmbeddingStreamer::stream(
    const at::Tensor& indices [[maybe_unused]],
    const at::Tensor& weights [[maybe_unused]],
    std::optional<at::Tensor> identities [[maybe_unused]],
    std::optional<at::Tensor> runtime_meta [[maybe_unused]],
    const at::Tensor& count [[maybe_unused]],
    bool require_tensor_copy [[maybe_unused]],
    bool blocking_tensor_copy [[maybe_unused]],
    std::optional<at::Tensor> copy_done_flag [[maybe_unused]]) {
  if (!enable_raw_embedding_streaming_) {
    return;
  }
#ifdef FBGEMM_FBCODE
  auto rec = torch::autograd::profiler::record_function_enter_new(
      "## RawEmbeddingStreamer::stream_callback ##");
  if (!require_tensor_copy) {
    StreamQueueItem stream_item(
        indices,
        weights,
        std::move(identities),
        std::move(runtime_meta),
        count);
    weights_to_stream_queue_.enqueue(stream_item);
    return;
  }
  auto poll_flag = [this, copy_done_flag]() {
    if (copy_done_flag.has_value()) {
      auto* ptr = static_cast<volatile int32_t*>(copy_done_flag->data_ptr());
      folly::stop_watch<std::chrono::microseconds> poll_watch;
      while (*ptr == 0) {
        std::this_thread::yield();
        if (poll_watch.elapsed().count() > kCopyDonePollTimeoutUs) {
          LOG(ERROR) << "[TBE_ID" << unique_id_
                     << "] copy_done_flag poll timed out after "
                     << kCopyDonePollTimeoutUs / 1'000'000 << "s";
          return false;
        }
      }
      *ptr = 0;
    }
    return true;
  };

  if (blocking_tensor_copy) {
    if (!poll_flag()) {
      return;
    }
    chunked_copy_and_enqueue(
        indices,
        weights,
        std::move(identities),
        std::move(runtime_meta),
        count,
        chunk_copy_threads_);
    join_chunk_copy_threads();
    return;
  }
  // Non-blocking: join the previous dispatch + copy threads, then spawn new
  // ones. The join is the serializer: it guarantees iter i's copy finished
  // reading the source cache rows before iter i+1 overwrites them.
  join_dispatch_thread();
  dispatch_thread_ = std::make_unique<std::thread>(
      [this, poll_flag, indices, weights, identities, runtime_meta, count]() {
        // Guard the dispatcher body so a copy/enqueue failure logs instead of
        // escaping the std::thread and calling std::terminate.
        try {
          if (!poll_flag()) {
            return;
          }
          chunked_copy_and_enqueue(
              indices,
              weights,
              identities,
              runtime_meta,
              count,
              chunk_copy_threads_);
        } catch (const std::exception& e) {
          XLOG(ERR) << "[TBE_ID" << unique_id_
                    << "] stream dispatcher thread caught exception: "
                    << e.what();
        } catch (...) {
          XLOG(ERR) << "[TBE_ID" << unique_id_
                    << "] stream dispatcher thread caught unknown exception";
        }
      });
  rec->record.end();
#endif
}

void RawEmbeddingStreamer::join_dispatch_thread() {
#ifdef FBGEMM_FBCODE
  auto rec = torch::autograd::profiler::record_function_enter_new(
      "## RawEmbeddingStreamer::join_dispatch_thread ##");
  if (dispatch_thread_ != nullptr && dispatch_thread_->joinable()) {
    dispatch_thread_->join();
  }
  join_chunk_copy_threads();
  rec->record.end();
#endif
}

#ifdef FBGEMM_FBCODE
void RawEmbeddingStreamer::join_chunk_copy_threads() {
  for (auto& t : chunk_copy_threads_) {
    if (t && t->joinable()) {
      t->join();
    }
  }
  chunk_copy_threads_.clear();
}

void RawEmbeddingStreamer::chunked_copy_and_enqueue(
    const at::Tensor& indices,
    const at::Tensor& weights,
    std::optional<at::Tensor> identities,
    std::optional<at::Tensor> runtime_meta,
    const at::Tensor& count,
    std::vector<std::unique_ptr<std::thread>>& target_copy_threads) {
  const auto num_rows = get_maybe_uvm_scalar(count);
  const auto thread_chunks =
      computeChunkRanges(num_rows, kChunkSize, kNumCopyThreads);

  for (auto& t : target_copy_threads) { // join+clear the previous batch
    if (t && t->joinable()) {
      t->join();
    }
  }
  target_copy_threads.clear();
  if (thread_chunks.empty()) {
    return;
  }

  // One copy thread per pre-computed group. Chunk boundaries and per-thread
  // grouping live entirely in computeChunkRanges, so the enqueued row set is
  // identical regardless of how threads are laid out here.
  target_copy_threads.reserve(thread_chunks.size());
  for (size_t ti = 0; ti < thread_chunks.size(); ++ti) {
    target_copy_threads.push_back(
        std::make_unique<std::thread>([this,
                                       indices,
                                       weights,
                                       identities,
                                       runtime_meta,
                                       chunks = thread_chunks[ti],
                                       ti]() {
          // Guard the copy body so a per-chunk failure logs instead of escaping
          // the std::thread and calling std::terminate.
          try {
            folly::stop_watch<std::chrono::milliseconds> thread_watch;
            int64_t rows_done = 0;
            for (const auto& [s, e] : chunks) {
              auto chunk_item = tensor_copy_chunk(
                  indices, weights, identities, runtime_meta, s, e);
              weights_to_stream_queue_.enqueue(std::move(chunk_item));
              rows_done += (e - s);
            }
            XLOG_EVERY_MS(INFO, 15000)
                << "[TBE_ID" << unique_id_ << "] copy_thread tid=" << ti
                << " rows=" << rows_done << " chunks=" << chunks.size()
                << " copy_ms=" << thread_watch.elapsed().count();
          } catch (const std::exception& e) {
            XLOG(ERR) << "[TBE_ID" << unique_id_ << "] copy_thread tid=" << ti
                      << " caught exception: " << e.what();
          } catch (...) {
            XLOG(ERR) << "[TBE_ID" << unique_id_ << "] copy_thread tid=" << ti
                      << " caught unknown exception";
          }
        }));
  }
  XLOG_EVERY_MS(INFO, 15000)
      << "[RES] chunked_copy tbe=" << unique_id_ << " rows=" << num_rows
      << " threads=" << thread_chunks.size();
}

folly::coro::Task<void> RawEmbeddingStreamer::tensor_stream(
    const at::Tensor& indices,
    const at::Tensor& weights,
    std::optional<at::Tensor> identities,
    std::optional<at::Tensor> runtime_meta) {
  using namespace ::aiplatform::gmpp::experimental::training_ps;
  if (indices.size(0) != weights.size(0)) {
    XLOG(ERR) << "[TBE_ID" << unique_id_
              << "] Indices and weights size mismatched " << indices.size(0)
              << " " << weights.size(0);
    if (ods_logger_) {
      ods_logger_->bumpKey("shard_size_mismatch", 1);
    }
    co_return;
  }
  folly::stop_watch<std::chrono::milliseconds> stop_watch;
  XLOG_EVERY_MS(INFO, 60000)
      << "[TBE_ID" << unique_id_
      << "] send streaming request: indices = " << indices.size(0)
      << ", weights = " << weights.size(0) << ", identities =  "
      << (identities.has_value() ? std::to_string(identities->size(0)) : "none")
      << ", runtime_meta =  "
      << (runtime_meta.has_value() ? std::to_string(runtime_meta->size(0))
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
  std::optional<at::Tensor> filtered_runtime_meta = std::nullopt;
  if (runtime_meta.has_value()) {
    filtered_runtime_meta = runtime_meta->index_select(0, mask);
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
    auto shard_mask = shard_indices_tensor.eq(i).nonzero().squeeze();
    auto table_indices_masked = table_indices.index_select(0, shard_mask);
    auto rows_in_shard = table_indices_masked.numel();
    if (rows_in_shard == 0) {
      continue;
    }
    auto global_indices_masked =
        global_indices_tensor.index_select(0, shard_mask);
    auto weights_masked = filtered_weights.index_select(0, shard_mask);

    if (weights_masked.size(0) != rows_in_shard ||
        global_indices_masked.numel() != rows_in_shard) {
      XLOG(ERR)
          << "[TBE_ID" << unique_id_
          << "] don't send the request for size mismatched tensors table: "
          << rows_in_shard << " weights: " << weights_masked.size(0)
          << " global_indices: " << global_indices_masked.numel();
      if (ods_logger_) {
        ods_logger_->bumpKey("shard_size_mismatch", 1);
      }
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
      auto identities_masked = filtered_identities->index_select(0, shard_mask);
      req.identities() = torch::distributed::wireDumpTensor(identities_masked);
    }
    if (filtered_runtime_meta.has_value()) {
      auto runtime_meta_masked =
          filtered_runtime_meta->index_select(0, shard_mask);
      req.runtimeMeta() =
          torch::distributed::wireDumpTensor(runtime_meta_masked);
    }
    try {
      co_await res_client->co_setEmbeddings(req);
    } catch (const std::exception& e) {
      if (ods_logger_) {
        ods_logger_->bumpKey("set_embeddings_rpc", 1);
      }
      XLOG(ERR) << "[TBE_ID" << unique_id_
                << "] co_setEmbeddings threw on shard " << i << ": "
                << e.what();
      throw;
    }
  }
  co_return;
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
