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
#endif

#include <utility>

namespace fbgemm_gpu {

struct StreamQueueItem {
  at::Tensor indices;
  at::Tensor weights;
  std::optional<at::Tensor> identities;
  at::Tensor count;
  StreamQueueItem(
      at::Tensor src_indices,
      at::Tensor src_weights,
      std::optional<at::Tensor> src_identities,
      at::Tensor src_count) {
    indices = std::move(src_indices);
    weights = std::move(src_weights);
    identities = std::move(src_identities);
    count = std::move(src_count);
  }
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
      const std::vector<int64_t>& table_sizes);

  virtual ~RawEmbeddingStreamer();

  /// Stream out non-negative elements in <indices> and its paired embeddings
  /// from <weights> for the first <count> elements in the tensor.
  /// It spins up a thread that will copy all 3 tensors to CPU and inject them
  /// into the background queue which will be picked up by another set of thread
  /// pools for streaming out to the thrift server (co-located on same host
  /// now).
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
      const at::Tensor& count,
      bool require_tensor_copy,
      bool blocking_tensor_copy = true);

#ifdef FBGEMM_FBCODE
  folly::coro::Task<void> tensor_stream(
      const at::Tensor& indices,
      const at::Tensor& weights,
      std::optional<at::Tensor> identities);
  /*
   * Copy the indices, weights and count tensors and enqueue them for
   * asynchronous stream.
   */
  void copy_and_enqueue_stream_tensors(
      const at::Tensor& indices,
      const at::Tensor& weights,
      std::optional<at::Tensor> identities,
      const at::Tensor& count);

  /*
   * Join the stream tensor copy thread, make sure the thread is properly
   * finished before creating new.
   */
  void join_stream_tensor_copy_thread();

  /*
   * FOR TESTING: Join the weight stream thread, make sure the thread is
   * properly finished for destruction and testing.
   */
  void join_weights_stream_thread();
  // FOR TESTING: get queue size.
  uint64_t get_weights_to_stream_queue_size();
#endif
 private:
  std::atomic<bool> stop_{false};
  std::string unique_id_;
  bool enable_raw_embedding_streaming_;
  int64_t res_store_shards_;
  int64_t res_server_port_;
  std::vector<std::string> table_names_;
  std::vector<int64_t> table_offsets_;
  at::Tensor table_sizes_;
#ifdef FBGEMM_FBCODE
  std::unique_ptr<std::thread> weights_stream_thread_;
  folly::UMPSCQueue<StreamQueueItem, true> weights_to_stream_queue_;
  std::unique_ptr<std::thread> stream_tensor_copy_thread_;
#endif
};

} // namespace fbgemm_gpu
