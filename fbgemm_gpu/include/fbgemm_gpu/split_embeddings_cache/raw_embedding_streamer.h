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
      int64_t res_chunk_size,
      int64_t res_num_consumers,
      int64_t res_num_copy_threads,
      std::vector<std::string> table_names,
      std::vector<int64_t> table_offsets,
      const std::vector<int64_t>& table_sizes);

  ~RawEmbeddingStreamer() override;

  /// Stream out non-negative elements in <indices> and its paired embeddings
  /// from <weights> for the first <count> elements in the tensor.
  /// It spins up a dispatcher thread that copies the 4 tensors (indices,
  /// weights, and the optional identities / runtime_meta) to CPU and injects
  /// them into the background queue, which is drained by a pool of consumer
  /// threads that stream out to the thrift server (co-located on same host
  /// now). The copy is split into <= res_chunk_size-row chunks across up to
  /// res_num_copy_threads copy threads.
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
      std::optional<at::Tensor> copy_done_flag = std::nullopt);

  /*
   * Join the pending dispatch (and the copy threads it spawned), making sure it
   * is properly finished before creating new.
   */
  void join_dispatch();

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
  size_t res_chunk_size_;
  size_t res_num_consumers_;
  size_t res_num_copy_threads_;
#endif
  std::vector<std::string> table_names_;
  std::vector<int64_t> table_offsets_;
  at::Tensor table_sizes_;
#ifdef FBGEMM_FBCODE
  // Named executor that ships enqueued StreamQueueItems to the PS. Push model:
  // producers submit one ship task per item and workers wake on submit (no
  // polling). Sized to res_num_consumers_.
  std::unique_ptr<folly::CPUThreadPoolExecutor> consumer_executor_;
  // Copy threads for UVM cache (joined every iteration). Shared by the blocking
  // and non-blocking stream() paths; this assumes a given table streams in a
  // single mode at a time (blocking OR non-blocking), never concurrently.
  std::vector<std::unique_ptr<std::thread>> chunk_copy_threads_;
  // Persistent size-1 executor that runs the per-iteration dispatch (poll_flag
  // + chunked_copy_and_enqueue). Named so its thread is identifiable in traces.
  std::unique_ptr<folly::CPUThreadPoolExecutor> dispatch_executor_;
  folly::SemiFuture<folly::Unit> dispatch_future_{folly::makeSemiFuture()};
  // OBC logger for RES silent-failure counters (res.fail.*). Emits to the
  // host-level OBC agent, so it reaches ODS from the trainer process without
  // per-process fb303 scrape config. Only constructed when streaming is on.
  std::unique_ptr<facebook::aiplatform::gmpp::experimental::training_ps::
                      TrainingPsOdsLogger>
      ods_logger_;

  void join_chunk_copy_threads();
  // Submit one ship task (blockingWait(tensor_stream(...))) for `item` onto
  // consumer_executor_.
  void submit_stream_item(StreamQueueItem item);
  void chunked_copy_and_enqueue(
      const at::Tensor& indices,
      const at::Tensor& weights,
      std::optional<at::Tensor> identities,
      std::optional<at::Tensor> runtime_meta,
      const at::Tensor& count,
      std::vector<std::unique_ptr<std::thread>>& target_copy_threads);

  // Waits (spinning) for copy_done_flag to signal the source tensors are safe
  // to read, resetting it. Returns false on timeout. Shared by the blocking
  // stream() path and dispatch_copy_task.
  bool poll_copy_done_flag(const std::optional<at::Tensor>& copy_done_flag);

  // Coroutine form of the non-blocking dispatch (poll_copy_done_flag +
  // chunked_copy_and_enqueue). Args are taken by value so nothing dangles once
  // it is scheduled on dispatch_executor_ -- a capturing lambda coroutine would
  // risk a use-after-free (clang-tidy cppcoreguidelines-avoid-capturing-lambda-
  // coroutines).
  folly::coro::Task<void> dispatch_copy_task(
      at::Tensor indices,
      at::Tensor weights,
      std::optional<at::Tensor> identities,
      std::optional<at::Tensor> runtime_meta,
      at::Tensor count,
      std::optional<at::Tensor> copy_done_flag);
#endif
};

fbgemm_gpu::StreamQueueItem tensor_copy_chunk(
    const at::Tensor& indices,
    const at::Tensor& weights,
    std::optional<at::Tensor> identities,
    std::optional<at::Tensor> runtime_meta,
    int64_t start_row,
    int64_t end_row);

// Tiles [0, num_rows) into per-thread groups of [start, end) chunk ranges, each
// chunk of size <= chunk_size, contiguous and non-overlapping (union is the
// whole range). The outer index is the thread; each inner vector is that
// thread's contiguous band split into chunks. Empty bands produce no group.
// Pure/build-agnostic so it is unit-testable without a GPU or FBGEMM_FBCODE.
// num_threads bounds how the rows are pre-split before chunking, matching
// chunked_copy_and_enqueue's tiling.
std::vector<std::vector<std::pair<int64_t, int64_t>>>
computeChunkRanges(int64_t num_rows, size_t chunk_size, size_t num_threads);
} // namespace fbgemm_gpu
