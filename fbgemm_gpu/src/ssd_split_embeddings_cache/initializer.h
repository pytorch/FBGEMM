/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <ATen/ATen.h>
#include <folly/concurrency/UnboundedQueue.h>
#include <torch/nn/init.h>

namespace ssd {

constexpr size_t kRowInitBufferSize = 32 * 1024;

class Initializer {
 public:
  Initializer(
      uint64_t random_seed,
      int64_t max_D,
      float uniform_init_lower,
      float uniform_init_upper,
      int64_t row_storage_bitwidth = 32)
      : producer_queue_(), consumer_queue_() {
    CHECK(
        row_storage_bitwidth == 32 || row_storage_bitwidth == 16 ||
        row_storage_bitwidth == 8);
    if (row_storage_bitwidth == 32) {
      row_storage_ = at::empty(
          {kRowInitBufferSize, max_D}, at::TensorOptions().dtype(at::kFloat));
    } else if (row_storage_bitwidth == 16) {
      row_storage_ = at::empty(
          {kRowInitBufferSize, max_D}, at::TensorOptions().dtype(at::kHalf));
    } else {
      row_storage_ = at::empty(
          {kRowInitBufferSize, max_D}, at::TensorOptions().dtype(at::kByte));
    }
    // Sanity check
    CHECK_EQ(row_storage_.element_size(), row_storage_bitwidth / 8);
    producer_ = std::make_unique<std::thread>([=] {
      const auto init = row_storage_.scalar_type() == at::ScalarType::Float ||
          row_storage_.scalar_type() == at::ScalarType::Half;
      if (init) {
        torch::nn::init::uniform_(
            row_storage_, uniform_init_lower, uniform_init_upper);
      }
      for (auto i = 0; i < kRowInitBufferSize; ++i) {
        producer_queue_.enqueue(i);
      }

      while (!stop_) {
        int64_t i;
        while (!stop_ &&
               !consumer_queue_.try_dequeue_until(
                   i,
                   std::chrono::steady_clock::now() +
                       std::chrono::milliseconds(100))) {
          // loop.
        }
        if (stop_) {
          return;
        }

        if (init) {
          // dequeued a row. Reinitialize and enqueue it.
          torch::nn::init::uniform_(
              row_storage_[i], uniform_init_lower, uniform_init_upper);
        }
        producer_queue_.enqueue(i);
      }
    });
  }

  ~Initializer() {
    stop_ = true;
    producer_->join();
  }

  folly::USPSCQueue<int64_t, true> producer_queue_;
  folly::USPSCQueue<int64_t, true> consumer_queue_;

  at::Tensor row_storage_;
  std::atomic<bool> stop_{false};
  std::unique_ptr<std::thread> producer_;
};

} // namespace ssd
