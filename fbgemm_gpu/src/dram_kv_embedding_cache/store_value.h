/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <ATen/core/ivalue.h>
#include "common/time/Time.h"

namespace kv_mem {

/// @ingroup embedding-dram-kvstore
///
/// @brief data structure to store tensor value and it's timestamp
//
template <typename scalar_t>
class StoreValue {
 public:
  explicit StoreValue(std::vector<scalar_t>&& value, int64_t timestamp) {
    value_ = std::move(value);
    timestamp_ = timestamp;
  }

  explicit StoreValue(StoreValue&& pv) noexcept {
    timestamp_ = pv.timestamp_;
    value_ = std::move(pv.value_);
  }

  StoreValue& operator=(StoreValue&& other) noexcept {
    if (this != &other) {
      value_ = std::move(other.value_);
      timestamp_ = other.timestamp_;
    }
    return *this;
  }

  int64_t getTimestamp() const {
    return timestamp_;
  }

  const std::vector<scalar_t>& getValue() const {
    return value_;
  }

  const std::vector<scalar_t>& getValueAndPromote() {
    timestamp_ = facebook::WallClockUtil::NowInUsecFast();
    return value_;
  }

 private:
  StoreValue& operator=(const StoreValue&) = delete;
  StoreValue& operator=(const StoreValue&&) = delete;
  StoreValue(const StoreValue& other) = delete;

  // cached tensor value
  std::vector<scalar_t> value_;

  // last visit timestamp
  int64_t timestamp_;
};
} // namespace kv_mem
