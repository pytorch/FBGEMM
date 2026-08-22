/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <ATen/ATen.h>
#ifdef FBGEMM_FBCODE
#include <folly/coro/Task.h>
#include <folly/futures/Future.h>
#endif

#include <utility>
#include <vector>

#ifdef FBGEMM_FBCODE
namespace folly {
class CPUThreadPoolExecutor;
} // namespace folly

namespace facebook::aiplatform::gmpp::experimental::training_ps {
class TrainingPsOdsLogger;
} // namespace facebook::aiplatform::gmpp::experimental::training_ps
#endif

namespace fbgemm_gpu {

struct StreamQueueItem {
  at::Tensor indices;
  at::Tensor weights;
  std::optional<at::Tensor> identities;
  std::optional<at::Tensor> runtime_meta;
  at::Tensor count;
  StreamQueueItem(
      at::Tensor src_indices,
      at::Tensor src_weights,
      std::optional<at::Tensor> src_identities,
      std::optional<at::Tensor> src_runtime_meta,
      at::Tensor src_count)
      : indices(std::move(src_indices)),
        weights(std::move(src_weights)),
        identities(std::move(src_identities)),
        runtime_meta(std::move(src_runtime_meta)),
        count(std::move(src_count)) {}
};

class RawEmbeddingStreamer : public torch::jit::CustomClassHolder {
 public:
  explicit RawEmbeddingStreamer(
      std::string unique_id,
      bool enable_raw_embedding_streaming,
      int64_t res_store_shards,
      int64_t res_server_port,
      std::vector<std::string> table_names,
      std::vector<int64_t> table_offsets,
      const std::vector<int64_t>& table_sizes,
      int64_t res_chunk_size = 500000,
      // TODO(T282801601): 8 was an arbitrary high value picked during
      // experimentation; too many consumer threads per TBE may be wasteful --
      // tune via experiments.
      int64_t res_num_consumers = 8,
      int64_t res_num_uvm_hit_copy_threads = 4,
      int64_t res_num_hbm_copy_threads = 4);

  ~RawEmbeddingStreamer() override;

  /// Stream out non-negative elements in <indices> and its paired embeddings
  /// from <weights> for the first <count> elements in the tensor.
  /// It spins up a dispatcher thread that copies the 4 tensors (indices,
  /// weights, and the optional identities / runtime_meta) to CPU and injects
  /// them into the background queue, which is drained by a pool of consumer
  /// threads that stream out to the thrift server (co-located on same host
  /// now). The copy is split into <= res_chunk_size-row chunks that run
  /// concurrently across the res_num_uvm_hit_copy_threads-worker copy pool.
  ///
  /// This is used in cuda stream callback, which doesn't require to be
  /// serialized with other callbacks, thus a separate thread is used to
  /// maximize the overlapping with other callbacks.
  ///
  /// @param indices The 1D embedding index tensor, should skip on negative
  /// value
  /// @param weights The 2D tensor that each row(embeddings) is paired up with
  /// relative element in <indices>
  /// @param count A single element tensor that contains the number of indices
  /// to be processed
  /// @param blocking_tensor_copy whether to copy the tensors to be streamed in
  /// a blocking manner
  /// @param use_hbm route a stream onto the dedicated HBM path: the
  /// non-blocking path joins/assigns the separate HBM dispatch future
  /// (hbm_dispatch_future_) on hbm_dispatch_executor_ instead of the main
  /// cache-hit one, and the copies fan out on the dedicated hbm_copy_executor_
  /// instead of the shared uvm_hit_copy_executor_, so HBM/UVM-miss drain does
  /// not interfere with cache-hit streaming at the dispatch or copy level. Only
  /// the ship path (consumer_executor_) stays shared. Defaults false (main
  /// path) -- inert until an out-of-scope caller opts in.
  /// @param expected_flag_value when set, wait for copy_done_flag to equal
  /// exactly this value and do not reset it, instead of waiting for any
  /// nonzero and resetting. Must be in [1, 2^31-1]. See poll_copy_done_flag.
  ///
  /// @return None
  void stream(
      const at::Tensor& indices,
      const at::Tensor& weights,
      std::optional<at::Tensor> identities,
      std::optional<at::Tensor> runtime_meta,
      const at::Tensor& count,
      bool require_tensor_copy,
      bool blocking_tensor_copy = true,
      std::optional<at::Tensor> copy_done_flag = std::nullopt,
      bool use_hbm = false,
      std::optional<int64_t> expected_flag_value = std::nullopt);

  /*
   * Join the pending dispatch future (which resolves only after all copies for
   * the previous iteration have been enqueued), making sure it is properly
   * finished before creating new.
   */
  void join_dispatch_and_workers();

  /*
   * HBM-path counterpart of join_dispatch_and_workers(): wait the pending HBM
   * dispatch future, which resolves only after its chunked_copy_and_enqueue
   * collectAllRange completes -- i.e. after the copy workers on
   * hbm_copy_executor_ finish. The executor itself is drained in the destructor
   * (join()); this call is the per-iteration wait for that copy work.
   */
  void join_hbm_dispatch_and_workers();

#ifdef FBGEMM_FBCODE
  folly::coro::Task<void> tensor_stream(
      const at::Tensor& indices,
      const at::Tensor& weights,
      std::optional<at::Tensor> identities,
      std::optional<at::Tensor> runtime_meta);

  /*
   * FOR TESTING ONLY: drops the ship executor to 0 worker threads so a test can
   * read a stable queue size (via get_weights_to_stream_queue_size()). Not
   * reversible -- the executor stops shipping after this call.
   */
  void join_weights_stream_thread();
  // FOR TESTING: get queue size.
  uint64_t get_weights_to_stream_queue_size();
#endif
 private:
  std::string unique_id_;
  bool enable_raw_embedding_streaming_;
#ifdef FBGEMM_FBCODE
  int64_t res_store_shards_;
  int64_t res_server_port_;
#endif
  std::vector<std::string> table_names_;
  std::vector<int64_t> table_offsets_;
  at::Tensor table_sizes_;
#ifdef FBGEMM_FBCODE
  size_t res_chunk_size_;
  size_t res_num_consumers_;
  size_t res_num_uvm_hit_copy_threads_;
  size_t res_num_hbm_copy_threads_;
#endif
#ifdef FBGEMM_FBCODE
  // Named executor that ships enqueued StreamQueueItems to the PS. Push model:
  // producers submit one ship task per item and workers wake on submit (no
  // polling). Sized to res_num_consumers_.
  std::unique_ptr<folly::CPUThreadPoolExecutor> consumer_executor_;
  // Persistent pool that runs the per-chunk tensor copies (CPU-bound
  // std::copy), sized res_num_uvm_hit_copy_threads_ so chunks run concurrently.
  // Used by the blocking and non-blocking main (cache-hit) stream() paths; this
  // assumes a given table streams in a single mode at a time (blocking OR
  // non-blocking), never concurrently.
  std::unique_ptr<folly::CPUThreadPoolExecutor> uvm_hit_copy_executor_;
  // Dedicated HBM copy pool sized res_num_hbm_copy_threads_, so HBM copies
  // don't contend with hit copies for the shared pool and delay the other
  // lane's read-before-overwrite barrier.
  std::unique_ptr<folly::CPUThreadPoolExecutor> hbm_copy_executor_;
  // Persistent size-1 executor that runs the per-iteration dispatch (poll_flag
  // + chunked_copy_and_enqueue). Named so its thread is identifiable in traces.
  std::unique_ptr<folly::CPUThreadPoolExecutor> uvm_hit_dispatch_executor_;
  // Size-1 executor dedicated to the HBM path's dispatch coroutine, so a
  // hit-path poll_copy_done_flag spin can't hold the only dispatch worker and
  // stall HBM drain.
  std::unique_ptr<folly::CPUThreadPoolExecutor> hbm_dispatch_executor_;
  folly::SemiFuture<folly::Unit> dispatch_future_{folly::makeSemiFuture()};
  // Second dispatch future for the HBM path. The HBM path runs on its own
  // hbm_dispatch_executor_ + hbm_copy_executor_, so it does not contend with
  // the main path at the dispatch or copy level; the only shared resource is
  // the ship path (consumer_executor_). This future is the per-lane state that
  // stream() joins/assigns.
  folly::SemiFuture<folly::Unit> hbm_dispatch_future_{folly::makeSemiFuture()};
  // OBC logger for RES silent-failure counters (res.fail.*). Emits to the
  // host-level OBC agent, so it reaches ODS from the trainer process without
  // per-process fb303 scrape config. Only constructed when streaming is on.
  std::unique_ptr<facebook::aiplatform::gmpp::experimental::training_ps::
                      TrainingPsOdsLogger>
      ods_logger_;

  // Submit one ship task (blockingWait(tensor_stream(...))) for `item` onto
  // consumer_executor_.
  void submit_stream_item(StreamQueueItem item);
  // Copies `count` rows of the source tensors into CPU chunks and enqueues each
  // via submit_stream_item, fanning the per-chunk copies out across the
  // selected copy pool (hbm_copy_executor_ when use_hbm, else
  // uvm_hit_copy_executor_) and awaiting them all (the read-before-overwrite
  // barrier). A coroutine so the non-blocking dispatch can co_await it; tensor
  // args are by value so nothing dangles once a chunk is scheduled.
  folly::coro::Task<void> chunked_copy_and_enqueue(
      at::Tensor indices,
      at::Tensor weights,
      std::optional<at::Tensor> identities,
      std::optional<at::Tensor> runtime_meta,
      at::Tensor count,
      bool use_hbm);
  // Copies one chunk [start, end) and enqueues it via submit_stream_item. Runs
  // to completion on a single uvm_hit_copy_executor_ worker; the per-chunk
  // try/catch keeps an exception from escaping into collectAllRange (which
  // awaits every sibling to completion and then rethrows one exception onto the
  // blocking caller / dispatch future). By value for the same reason as
  // chunked_copy_and_enqueue.
  folly::coro::Task<void> copy_chunk_task(
      at::Tensor indices,
      at::Tensor weights,
      std::optional<at::Tensor> identities,
      std::optional<at::Tensor> runtime_meta,
      int64_t start,
      int64_t end);

  // Waits (spinning) for copy_done_flag to signal the source tensors are safe
  // to read. Returns false on timeout. Shared by the blocking stream() path and
  // dispatch_copy_task.
  //
  // By default the flag is a boolean: the producer writes any nonzero and this
  // resets it to 0. Passing expected_flag_value instead waits for the flag to
  // equal exactly that value and leaves it in place. Prefer that where
  // possible -- a boolean poll that times out leaves the flag set, so the next
  // drain reads it as already signalled and copies the source tensors while
  // they are still being written. The boolean form is kept for backward
  // compatibility.
  bool poll_copy_done_flag(
      const std::optional<at::Tensor>& copy_done_flag,
      std::optional<int64_t> expected_flag_value);

  // Coroutine form of the non-blocking dispatch (poll_copy_done_flag +
  // chunked_copy_and_enqueue). Args are taken by value so nothing dangles once
  // it is scheduled on uvm_hit_dispatch_executor_ -- a capturing lambda
  // coroutine would risk a use-after-free
  // (clang-tidy cppcoreguidelines-avoid-capturing-lambda-coroutines).
  folly::coro::Task<void> dispatch_copy_task(
      at::Tensor indices,
      at::Tensor weights,
      std::optional<at::Tensor> identities,
      std::optional<at::Tensor> runtime_meta,
      at::Tensor count,
      std::optional<at::Tensor> copy_done_flag,
      bool use_hbm,
      std::optional<int64_t> expected_flag_value);
#endif
};

fbgemm_gpu::StreamQueueItem tensor_copy_chunk(
    const at::Tensor& indices,
    const at::Tensor& weights,
    std::optional<at::Tensor> identities,
    std::optional<at::Tensor> runtime_meta,
    int64_t start_row,
    int64_t end_row);

// Tiles [0, num_rows) into flat [start, end) chunks of size <= chunk_size,
// contiguous and non-overlapping (their union is the whole range). One task per
// chunk is submitted to uvm_hit_copy_executor_, which load-balances them across
// its workers. Pure/build-agnostic so it is unit-testable without a GPU or
// FBGEMM_FBCODE.
std::vector<std::pair<int64_t, int64_t>> computeChunks(
    int64_t num_rows,
    size_t chunk_size);
} // namespace fbgemm_gpu
