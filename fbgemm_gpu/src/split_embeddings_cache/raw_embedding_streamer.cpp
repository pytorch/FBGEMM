/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef FBGEMM_FBCODE
#include <fmt/format.h>
#include <folly/coro/BlockingWait.h>
#include <folly/coro/Collect.h>
#include <folly/coro/Task.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/executors/thread_factory/NamedThreadFactory.h>
#include <folly/futures/Future.h>
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

// One shard's prepared request, built up front so the ship section is pure I/O.
struct ShardReq {
  int shard_id{0};
  ::aiplatform::gmpp::experimental::training_ps::SetEmbeddingsRequest req;
};

// Per-shard RPC timing recorded by ship_one_shard for the tensor_stream
// breakdown log.
struct ShardTiming {
  int shard_id{0};
  int64_t rpc_ms{0};
};

// Ship one shard's SetEmbeddingsRequest and record its timing. A named
// coroutine (not a capturing lambda) so every input is owned by the coroutine
// frame and nothing dangles across the co_await -- the capture-free form the
// cppcoreguidelines-avoid-capturing-lambda-coroutines rule requires. Isolates
// its own failure: an RPC throw is logged/counted and swallowed, so it neither
// cancels its still-in-flight siblings under collectAllRange nor escapes the
// ship task.
folly::coro::Task<void> ship_one_shard(
    apache::thrift::Client<::aiplatform::gmpp::experimental::training_ps::
                               TrainingParameterServerService>* client,
    ::aiplatform::gmpp::experimental::training_ps::SetEmbeddingsRequest req,
    int shard_id,
    std::string unique_id,
    facebook::aiplatform::gmpp::experimental::training_ps::TrainingPsOdsLogger*
        ods_logger,
    std::shared_ptr<std::vector<ShardTiming>> timings,
    size_t idx) {
  folly::stop_watch<std::chrono::milliseconds> sw;
  try {
    co_await client->co_setEmbeddings(req);
  } catch (const std::exception& e) {
    if (ods_logger != nullptr) {
      ods_logger->bumpKey("set_embeddings_rpc_failure", 1);
    }
    XLOG(ERR) << "[TBE_ID" << unique_id << "] co_setEmbeddings threw on shard "
              << shard_id << ": " << e.what();
  }
  (*timings)[idx] = ShardTiming{shard_id, sw.elapsed().count()};
}

// Find the slowest shard and emit a rate-limited tensor_stream timing
// breakdown.
void log_shard_ship_breakdown(
    const std::string& unique_id,
    const std::vector<ShardTiming>& shard_timings,
    int64_t total_rpc_ms,
    int64_t total_rows,
    int64_t num_shards) {
  int64_t max_shard_ms = 0;
  std::optional<int> max_shard_id;
  for (const auto& st : shard_timings) {
    if (!max_shard_id || st.rpc_ms > max_shard_ms) {
      max_shard_ms = st.rpc_ms;
      max_shard_id = st.shard_id;
    }
  }
  XLOG_EVERY_MS(INFO, 15000)
      << "[TBE_ID" << unique_id
      << "] tensor_stream breakdown: total_rpc_ms=" << total_rpc_ms
      << " max_shard_ms=" << max_shard_ms
      << " max_shard_id=" << max_shard_id.value_or(-1) << " rows=" << total_rows
      << " shards=" << num_shards << " parallel_rpcs=" << shard_timings.size();
}

/// Read a scalar value from a tensor that is maybe a UVM tensor
/// Note that `tensor.item<type>()` is not allowed on a UVM tensor in
/// PyTorch
inline int64_t get_maybe_uvm_scalar(const at::Tensor& tensor) {
  return tensor.scalar_type() == at::ScalarType::Long
      ? *(tensor.const_data_ptr<int64_t>())
      : *(tensor.const_data_ptr<int32_t>());
}

#endif

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
  // Each tensor is copied under its own dispatch. They are independent copies,
  // so there is no need to nest the dispatches (nesting would only reintroduce
  // scalar_t shadowing and multiply template instantiations).
  FBGEMM_DISPATCH_INTEGRAL_TYPES(
      indices.scalar_type(), "tensor_copy_chunk", [&] {
        std::copy(
            indices.const_data_ptr<scalar_t>() + start_row,
            indices.const_data_ptr<scalar_t>() + end_row,
            new_indices.mutable_data_ptr<scalar_t>());
      });
  FBGEMM_DISPATCH_FLOAT_HALF_AND_BYTE(
      weights.scalar_type(), "tensor_copy_chunk", [&] {
        std::copy(
            weights.const_data_ptr<scalar_t>() + start_row * weights.size(1),
            weights.const_data_ptr<scalar_t>() + end_row * weights.size(1),
            new_weights.mutable_data_ptr<scalar_t>());
      });
  if (identities.has_value()) {
    FBGEMM_DISPATCH_INTEGRAL_TYPES(
        identities->scalar_type(), "tensor_copy_chunk", [&] {
          std::copy(
              identities->const_data_ptr<scalar_t>() +
                  start_row * identities->size(1),
              identities->const_data_ptr<scalar_t>() +
                  end_row * identities->size(1),
              new_identities->mutable_data_ptr<scalar_t>());
        });
  }
  if (runtime_meta.has_value()) {
    FBGEMM_DISPATCH_ALL_TYPES(
        runtime_meta->scalar_type(), "tensor_copy_chunk", [&] {
          std::copy(
              runtime_meta->const_data_ptr<scalar_t>() +
                  start_row * runtime_meta->size(1),
              runtime_meta->const_data_ptr<scalar_t>() +
                  end_row * runtime_meta->size(1),
              new_runtime_meta->mutable_data_ptr<scalar_t>());
        });
  }
  *new_count.mutable_data_ptr<int64_t>() = n;
  return fbgemm_gpu::StreamQueueItem{
      new_indices, new_weights, new_identities, new_runtime_meta, new_count};
}

std::vector<std::pair<int64_t, int64_t>> computeChunks(
    int64_t num_rows,
    size_t chunk_size) {
  // Split [0, num_rows) into flat [start, end) chunks of <= chunk_size rows;
  // chunks are contiguous, non-overlapping, and their union is the whole range.
  std::vector<std::pair<int64_t, int64_t>> chunks;
  if (num_rows <= 0 || chunk_size == 0) {
    return chunks;
  }
  for (int64_t s = 0; s < num_rows; s += static_cast<int64_t>(chunk_size)) {
    const int64_t e = std::min(s + static_cast<int64_t>(chunk_size), num_rows);
    chunks.emplace_back(s, e);
  }
  return chunks;
}

RawEmbeddingStreamer::RawEmbeddingStreamer(
    std::string unique_id,
    bool enable_raw_embedding_streaming,
    int64_t res_store_shards [[maybe_unused]],
    int64_t res_server_port [[maybe_unused]],
    std::vector<std::string> table_names,
    std::vector<int64_t> table_offsets,
    const std::vector<int64_t>& table_sizes,
    int64_t res_chunk_size [[maybe_unused]],
    int64_t res_num_consumers [[maybe_unused]],
    int64_t res_num_copy_threads [[maybe_unused]],
    int64_t res_num_hbm_copy_threads [[maybe_unused]])
    : unique_id_(std::move(unique_id)),
      enable_raw_embedding_streaming_(enable_raw_embedding_streaming),
#ifdef FBGEMM_FBCODE
      res_store_shards_(res_store_shards),
      res_server_port_(res_server_port),
#endif
      table_names_(std::move(table_names)),
      table_offsets_(std::move(table_offsets)),
      table_sizes_(at::tensor(table_sizes))
#ifdef FBGEMM_FBCODE
      ,
      res_chunk_size_(res_chunk_size),
      res_num_consumers_(res_num_consumers),
      res_num_copy_threads_(res_num_copy_threads),
      res_num_hbm_copy_threads_(res_num_hbm_copy_threads)
#endif
{
#ifdef FBGEMM_FBCODE
  if (enable_raw_embedding_streaming_) {
    // Fail loud on a misconfigured knob. These are now caller-supplied (were
    // compile-time constants), and 0 -- or a negative that wrapped to a huge
    // size_t -- would silently break streaming: res_num_consumers=0 spawns no
    // drain threads (queue grows unbounded), res_chunk_size=0 makes
    // computeChunks return empty (enqueues nothing), res_num_copy_threads=0
    // sizes copy_executor_ to zero workers (copy tasks never run). Reject
    // rather than silently no-op.
    TORCH_CHECK(
        res_chunk_size > 0 && res_num_consumers > 0 &&
            res_num_copy_threads > 0 && res_num_hbm_copy_threads > 0,
        "RES config knobs must be > 0: res_chunk_size=",
        res_chunk_size,
        ", res_num_consumers=",
        res_num_consumers,
        ", res_num_copy_threads=",
        res_num_copy_threads,
        ", res_num_hbm_copy_threads=",
        res_num_hbm_copy_threads);
    XLOG(INFO) << "[TBE_ID" << unique_id_
               << "] Raw embedding streaming enabled with res_server_port at"
               << res_server_port_;
    // The first call to get the client is expensive, so eagerly get it here
    auto _eager_client = get_res_client(res_server_port_);

    ods_logger_ = std::make_unique<facebook::aiplatform::gmpp::experimental::
                                       training_ps::TrainingPsOdsLogger>();

    // Persistent size-1 executor that runs the non-blocking per-iteration
    // dispatch (poll + chunked_copy_and_enqueue) as a coroutine, off the
    // trainer thread. Named so its thread is identifiable in traces; the
    // prefix is kept short because Linux caps thread names at 15 chars
    // (pthread_setname_np) and NamedThreadFactory still appends a suffix, so a
    // longer prefix would truncate unique_id_ away.
    dispatch_executor_ = std::make_unique<folly::CPUThreadPoolExecutor>(
        1,
        std::make_unique<folly::NamedThreadFactory>(
            fmt::format("RESDisp{}", unique_id_)));

    XLOG(INFO) << "[TBE_ID" << unique_id_
               << "] Starting RES ship executor with " << res_num_consumers_
               << " threads"
               << ", chunk_size=" << res_chunk_size_
               << ", copy_threads=" << res_num_copy_threads_;
    // Push model: ship tasks are submitted onto this executor (one per enqueued
    // StreamQueueItem) and its workers wake on submit -- no polled queue, no
    // raw std::thread that could std::terminate on an escaped exception.
    //
    // Ordering caveat: with res_num_consumers_ > 1, ship tasks run
    // concurrently, so arrival order at the PS is NOT enqueue (iteration)
    // order. The store (TrainingPsHandler) applies same-(fqn,row_id) writes
    // arrival-wins with no version compare, so a stale iter-i write can
    // transiently clobber a fresh iter-(i+1) one for a hot row (self-heals on
    // the row's next in-order update). res_num_consumers_ == 1 is
    // ordered/safe. TODO(T281413204): proper fix is to carry a per-row
    // iteration/version and keep-newest in the store.
    consumer_executor_ = std::make_unique<folly::CPUThreadPoolExecutor>(
        res_num_consumers_,
        std::make_unique<folly::NamedThreadFactory>(
            fmt::format("RESShip.{}", unique_id_)));

    // Persistent pool that runs the per-chunk tensor copies (CPU-bound
    // std::copy) off the trainer thread. Sized res_num_copy_threads_ so N
    // chunks run concurrently -- the same N-way parallelism the old
    // one-std::thread-per-group model gave.
    copy_executor_ = std::make_unique<folly::CPUThreadPoolExecutor>(
        res_num_copy_threads_,
        std::make_unique<folly::NamedThreadFactory>(
            fmt::format("RESCopy.{}", unique_id_)));

    // Dedicated HBM-path executors so the HBM path does not contend with the
    // hit path at the dispatch (size-1) or copy level. The ship path
    // (consumer_executor_) stays shared across both lanes. Inert until a caller
    // opts a stream onto the HBM path. Prefixes are kept short for the same
    // 15-char thread-name cap as above, so unique_id_ is not truncated away.
    hbm_dispatch_executor_ = std::make_unique<folly::CPUThreadPoolExecutor>(
        1,
        std::make_unique<folly::NamedThreadFactory>(
            fmt::format("RESHbmDisp{}", unique_id_)));
    hbm_copy_executor_ = std::make_unique<folly::CPUThreadPoolExecutor>(
        res_num_hbm_copy_threads_,
        std::make_unique<folly::NamedThreadFactory>(
            fmt::format("RESHbmCopy{}", unique_id_)));
  }
#endif
}

RawEmbeddingStreamer::~RawEmbeddingStreamer() {
#ifdef FBGEMM_FBCODE
  if (enable_raw_embedding_streaming_) {
    // Drain in producer order: dispatch schedules onto copy_executor_, and copy
    // groups submit ship tasks onto consumer_executor_, so join dispatch ->
    // copy -> consumer. A wrong order risks joining an executor while a
    // producer still schedules onto it.
    join_dispatch_and_workers();
    // The HBM path has its own dispatch + copy executors; drain its dispatch
    // future too before joining those executors.
    join_hbm_dispatch_and_workers();
    if (dispatch_executor_ != nullptr) {
      dispatch_executor_->join();
    }
    if (hbm_dispatch_executor_ != nullptr) {
      hbm_dispatch_executor_->join();
    }
    if (copy_executor_ != nullptr) {
      copy_executor_->join();
    }
    if (hbm_copy_executor_ != nullptr) {
      hbm_copy_executor_->join();
    }
    // Producers (dispatch + copy) are all joined above, so no further ship
    // tasks will be submitted. join() drains the in-flight ones before the
    // executor's threads stop, then destroys members in a safe order.
    if (consumer_executor_ != nullptr) {
      consumer_executor_->join();
    }
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
    std::optional<at::Tensor> copy_done_flag [[maybe_unused]],
    bool use_hbm [[maybe_unused]]) {
  if (!enable_raw_embedding_streaming_) {
    return;
  }
#ifdef FBGEMM_FBCODE
  auto rec = torch::autograd::profiler::record_function_enter_new(
      "## RawEmbeddingStreamer::stream_callback ##");
  if (!require_tensor_copy) {
    submit_stream_item(StreamQueueItem(
        indices,
        weights,
        std::move(identities),
        std::move(runtime_meta),
        count));
    return;
  }
  auto poll_flag = [this, copy_done_flag]() {
    return poll_copy_done_flag(copy_done_flag);
  };

  // Select the lane's executors once: the HBM path runs on its own dispatch +
  // copy pools so it does not contend with the hit path. Only the ship path
  // (consumer_executor_) is shared.
  //
  // CAVEAT for a future use_hbm caller: the two lanes have independent
  // futures/executors, so there is NO cross-lane read-before-overwrite barrier.
  // A row that is cache-miss in iter i (HBM path) then cache-hit in iter i+1
  // (hit path) has no happens-before across the lanes -- the caller must pin a
  // given row/table to a single lane (or join both futures on a lane
  // transition) and rely on the arrival-wins / versioned store (T281413204) for
  // cross-lane freshness.
  auto* dispatch_exec =
      use_hbm ? hbm_dispatch_executor_.get() : dispatch_executor_.get();

  if (blocking_tensor_copy) {
    if (!poll_flag()) {
      return;
    }
    // blockingWait is itself the barrier: it blocks the trainer thread until
    // every copy group (on the selected copy pool) has finished and enqueued --
    // the read-before-overwrite barrier. The trainer thread is not a copy-pool
    // worker, so this cannot self-deadlock. The per-group try/catch keeps
    // collectAllRange from rethrowing here. The ship path (consumer_executor_)
    // is shared, so both lanes' enqueued items drain the same way.
    folly::coro::blockingWait(chunked_copy_and_enqueue(
        indices,
        weights,
        std::move(identities),
        std::move(runtime_meta),
        count,
        use_hbm));
    return;
  }
  // Non-blocking: join the previous dispatch on the SELECTED lane (which
  // awaited its copies on that lane's copy pool) before starting a new one. The
  // join is the serializer: it guarantees iter i's copies finished reading the
  // source cache rows before iter i+1 overwrites them.
  if (use_hbm) {
    join_hbm_dispatch_and_workers();
  } else {
    join_dispatch_and_workers();
  }
  // Dispatch runs as a coroutine on the selected size-1 dispatch executor;
  // folly captures any exception into the selected future, which is logged when
  // the future is waited in join_(hbm_)dispatch() (log-and-continue, never
  // terminate).
  TORCH_CHECK(
      dispatch_exec != nullptr,
      "dispatch executors are only constructed when raw embedding streaming is "
      "enabled; stream() must have early-returned otherwise");
  auto& dispatch_future = use_hbm ? hbm_dispatch_future_ : dispatch_future_;
  dispatch_future = folly::coro::co_withExecutor(
                        dispatch_exec,
                        dispatch_copy_task(
                            indices,
                            weights,
                            std::move(identities),
                            std::move(runtime_meta),
                            count,
                            std::move(copy_done_flag),
                            use_hbm))
                        .start();
  rec->record.end();
#endif
}

void RawEmbeddingStreamer::join_dispatch_and_workers() {
#ifdef FBGEMM_FBCODE
  auto rec = torch::autograd::profiler::record_function_enter_new(
      "## RawEmbeddingStreamer::join_dispatch_and_workers ##");
  // Wait the previous dispatch. The dispatch future resolves only after
  // chunked_copy_and_enqueue's collectAllRange completes, so this is also the
  // read-before-overwrite barrier: iter i's copies finish reading the source
  // cache rows before iter i+1 overwrites them. Log-and-continue: an exception
  // the dispatch deferred into the future must not escape (would std::terminate
  // the trainer).
  if (dispatch_future_.valid()) {
    try {
      std::move(dispatch_future_).get();
    } catch (const std::exception& e) {
      XLOG(ERR) << "[TBE_ID" << unique_id_
                << "] stream dispatcher caught exception: " << e.what();
    } catch (...) {
      XLOG(ERR) << "[TBE_ID" << unique_id_
                << "] stream dispatcher caught unknown exception";
    }
    dispatch_future_ = folly::makeSemiFuture();
  }
  rec->record.end();
#endif
}

void RawEmbeddingStreamer::join_hbm_dispatch_and_workers() {
#ifdef FBGEMM_FBCODE
  auto rec = torch::autograd::profiler::record_function_enter_new(
      "## RawEmbeddingStreamer::join_hbm_dispatch_and_workers ##");
  // HBM-path mirror of join_dispatch(). The HBM dispatch future resolves only
  // after its chunked_copy_and_enqueue collectAllRange completes (copies run on
  // the dedicated hbm_copy_executor_), so this is also the HBM path's
  // read-before-overwrite barrier. Log-and-continue: a deferred exception must
  // not escape (would std::terminate the trainer).
  if (hbm_dispatch_future_.valid()) {
    try {
      std::move(hbm_dispatch_future_).get();
    } catch (const std::exception& e) {
      XLOG(ERR) << "[TBE_ID" << unique_id_
                << "] HBM stream dispatcher caught exception: " << e.what();
    } catch (...) {
      XLOG(ERR) << "[TBE_ID" << unique_id_
                << "] HBM stream dispatcher caught unknown exception";
    }
    hbm_dispatch_future_ = folly::makeSemiFuture();
  }
  rec->record.end();
#endif
}

#ifdef FBGEMM_FBCODE
void RawEmbeddingStreamer::submit_stream_item(StreamQueueItem item) {
  // Push model: hand the item to a ship worker that wakes on submit. folly
  // captures any task exception (so an escaped throw can't std::terminate the
  // trainer); we still wrap the body to log a transient tensor_stream failure
  // and to keep the depth-gauge / periodic-log behavior of the old consumer.
  consumer_executor_->add([this, item = std::move(item)]() mutable {
    try {
      folly::stop_watch<std::chrono::milliseconds> stop_watch;
      folly::coro::blockingWait(tensor_stream(
          item.indices, item.weights, item.identities, item.runtime_meta));
      if (ods_logger_) {
        ods_logger_->bumpKeyGauge(
            "stream_mpmc_depth",
            static_cast<double>(consumer_executor_->getTaskQueueSize()));
      }
      XLOG_EVERY_MS(INFO, 60000)
          << "[TBE_ID" << unique_id_ << "] end stream queue size: "
          << consumer_executor_->getTaskQueueSize() << " stream takes "
          << stop_watch.elapsed().count() << "ms"
          << " rows=" << item.indices.size(0);
    } catch (const std::exception& e) {
      XLOG(ERR) << "[TBE_ID" << unique_id_
                << "] ship task caught exception: " << e.what();
    } catch (...) {
      XLOG(ERR) << "[TBE_ID" << unique_id_
                << "] ship task caught unknown exception";
    }
  });
}

folly::coro::Task<void> RawEmbeddingStreamer::copy_chunk_task(
    at::Tensor indices,
    at::Tensor weights,
    std::optional<at::Tensor> identities,
    std::optional<at::Tensor> runtime_meta,
    int64_t start,
    int64_t end) {
  // Guard the copy body so a per-chunk failure logs instead of escaping into
  // collectAllRange (which would cancel siblings and rethrow onto the blocking
  // caller / std::terminate).
  try {
    auto chunk_item = tensor_copy_chunk(
        indices, weights, identities, runtime_meta, start, end);
    submit_stream_item(std::move(chunk_item));
  } catch (const std::exception& e) {
    XLOG(ERR) << "[TBE_ID" << unique_id_ << "] copy_chunk [" << start << ", "
              << end << ") caught exception: " << e.what();
  } catch (...) {
    XLOG(ERR) << "[TBE_ID" << unique_id_ << "] copy_chunk [" << start << ", "
              << end << ") caught unknown exception";
  }
  co_return;
}

folly::coro::Task<void> RawEmbeddingStreamer::chunked_copy_and_enqueue(
    at::Tensor indices,
    at::Tensor weights,
    std::optional<at::Tensor> identities,
    std::optional<at::Tensor> runtime_meta,
    at::Tensor count,
    bool use_hbm) {
  auto rec = torch::autograd::profiler::record_function_enter_new(
      "## RawEmbeddingStreamer::chunked_copy_and_enqueue ##");
  const auto num_rows = get_maybe_uvm_scalar(count);
  const auto chunks = computeChunks(num_rows, res_chunk_size_);
  XLOG_EVERY_MS(INFO, 15000)
      << "[RES] chunked_copy tbe=" << unique_id_ << " rows=" << num_rows
      << " chunks=" << chunks.size();
  if (chunks.empty()) {
    rec->record.end();
    co_return;
  }

  // Pick the copy pool for this stream's path: the dedicated hbm_copy_executor_
  // for HBM drain, else the main copy_executor_. Only the ship path
  // (consumer_executor_) is shared between the two.
  auto* copy_exec = use_hbm ? hbm_copy_executor_.get() : copy_executor_.get();

  // One copy task per chunk. Chunk boundaries live entirely in computeChunks,
  // so the enqueued row set is identical regardless of how the tasks run.
  // co_withExecutor binds each task to copy_exec (the selected copy pool) -- so
  // a bare Task does not inherit the size-1 dispatch executor and serialize the
  // copies; collectAllRange then runs the tasks across the pool -- the workers
  // pull them, so the pool load-balances -- and completes only after every one
  // is done (the barrier).
  std::vector<folly::coro::TaskWithExecutor<void>> tasks;
  tasks.reserve(chunks.size());
  for (const auto& [start, end] : chunks) {
    tasks.push_back(
        folly::coro::co_withExecutor(
            copy_exec,
            copy_chunk_task(
                indices, weights, identities, runtime_meta, start, end)));
  }
  co_await folly::coro::collectAllRange(std::move(tasks));
  rec->record.end();
  co_return;
}

bool RawEmbeddingStreamer::poll_copy_done_flag(
    const std::optional<at::Tensor>& copy_done_flag) {
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
}

folly::coro::Task<void> RawEmbeddingStreamer::dispatch_copy_task(
    at::Tensor indices,
    at::Tensor weights,
    std::optional<at::Tensor> identities,
    std::optional<at::Tensor> runtime_meta,
    at::Tensor count,
    std::optional<at::Tensor> copy_done_flag,
    bool use_hbm) {
  if (!poll_copy_done_flag(copy_done_flag)) {
    co_return;
  }
  // Log at failure time instead of letting the exception ride dispatch_future_
  // to the next join_dispatch_and_workers(): if streaming stalls right after a
  // failure, that log would be delayed until the next iteration or the
  // destructor. join_dispatch_and_workers()'s catch remains as a safety net.
  try {
    co_await chunked_copy_and_enqueue(
        indices,
        weights,
        std::move(identities),
        std::move(runtime_meta),
        count,
        use_hbm);
  } catch (const std::exception& e) {
    XLOG(ERR) << "[TBE_ID" << unique_id_
              << "] stream dispatch caught exception: " << e.what();
  } catch (...) {
    XLOG(ERR) << "[TBE_ID" << unique_id_
              << "] stream dispatch caught unknown exception";
  }
  co_return;
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
  // 2. Split by shards -- prepare every shard's request up front so the
  // subsequent ship section is pure I/O with nothing serialized between RPCs.
  std::vector<ShardReq> shard_requests;
  shard_requests.reserve(res_store_shards_);
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
    shard_requests.push_back(ShardReq{i, std::move(req)});
  }

  // 3. Ship every shard RPC in parallel (all suspended on co_setEmbeddings at
  // once, so their round-trips overlap), tracking per-shard timing. Each task
  // isolates its own failure: a shard whose RPC throws is logged and counted,
  // and the task completes normally WITHOUT propagating -- so one dead shard
  // neither cancels its still-in-flight siblings (as a bare collectAllRange
  // over throwing tasks would) nor escapes the ship task.
  auto shard_timings =
      std::make_shared<std::vector<ShardTiming>>(shard_requests.size());
  folly::stop_watch<std::chrono::milliseconds> rpc_sw;
  std::vector<folly::coro::Task<void>> rpc_tasks;
  rpc_tasks.reserve(shard_requests.size());
  for (size_t idx = 0; idx < shard_requests.size(); ++idx) {
    auto& shard_request = shard_requests[idx];
    rpc_tasks.push_back(ship_one_shard(
        res_client.get(),
        std::move(shard_request.req),
        shard_request.shard_id,
        unique_id_,
        ods_logger_.get(),
        shard_timings,
        idx));
  }
  co_await folly::coro::collectAllRange(std::move(rpc_tasks));
  const auto total_rpc_ms = rpc_sw.elapsed().count();

  log_shard_ship_breakdown(
      unique_id_, *shard_timings, total_rpc_ms, total_rows, res_store_shards_);
  co_return;
}

void RawEmbeddingStreamer::join_weights_stream_thread() {
  // TESTING only: drop the ship executor to 0 worker threads so subsequently
  // submitted tasks accumulate in its queue (observable via
  // get_weights_to_stream_queue_size()) instead of being shipped. Mirrors the
  // old "stop the consumer threads" behavior; unlike join() the executor still
  // accepts newly submitted tasks.
  if (consumer_executor_ != nullptr) {
    consumer_executor_->setNumThreads(0);
  }
}

uint64_t RawEmbeddingStreamer::get_weights_to_stream_queue_size() {
  return consumer_executor_ != nullptr ? consumer_executor_->getTaskQueueSize()
                                       : 0;
}
#endif

} // namespace fbgemm_gpu
