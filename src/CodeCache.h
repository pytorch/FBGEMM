/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once
#include <condition_variable>
#include <future>
#include <map>
#include <mutex>

namespace fbgemm {

/**
 * @brief Thread safe cache for microkernels, ensures single creation per key.
 * @tparam Key Type of unique key (typically a tuple)
 * @tparam Value Type of the microkernel function (Typically a function pointer)
 */
template <typename KEY, typename VALUE>
class CodeCache {
 private:
  std::map<KEY, std::shared_future<VALUE>> values_;
  std::mutex mutex_;

 public:
  CodeCache(const CodeCache&) = delete;
  CodeCache& operator=(const CodeCache&) = delete;

  CodeCache(){};

  VALUE getOrCreate(const KEY& key, std::function<VALUE()> generatorFunction) {
    std::shared_future<VALUE> returnFuture;
    std::promise<VALUE> returnPromise;
    bool needsToGenerate = false;

    // Check for existance of the key
    {
      std::unique_lock<std::mutex> lock(mutex_);

      auto it = values_.find(key);
      if (it != values_.end()) {
        returnFuture = it->second;
      } else {
        values_[key] = returnFuture = returnPromise.get_future().share();
        needsToGenerate = true;
      }
    }

    // The value (code) generation is not happening under a lock
    if (needsToGenerate) {
      returnPromise.set_value(generatorFunction());
    }

    // Wait for the future and return the value
    return returnFuture.get();
  }
};

} // namespace fbgemm
