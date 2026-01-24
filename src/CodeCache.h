/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <future>
#include <map>
#include <shared_mutex>

#ifdef FBCODE_CAFFE2
#include <folly/container/F14Map.h>
#endif

namespace fbgemm {

/**
 * @brief Thread safe cache for microkernels, ensures single creation per key.
 * @tparam KEY Type of unique key (typically a tuple)
 * @tparam VALUE Type of the microkernel function (Typically a function pointer)
 * @tparam THREAD_LOCAL use thread local and avoid locking (default false)
 */
template <typename KEY, typename VALUE, bool THREAD_LOCAL = false>
class CodeCache {
 private:
#ifdef FBCODE_CAFFE2
  folly::F14FastMap<KEY, std::shared_future<VALUE>> values_;
#else
  std::map<KEY, std::shared_future<VALUE>> values_;
#endif

  std::shared_timed_mutex mutex_;

 public:
  CodeCache(const CodeCache&) = delete;
  CodeCache& operator=(const CodeCache&) = delete;

  CodeCache(CodeCache&&) = default;
  CodeCache& operator=(CodeCache&&) = default;

  CodeCache() = default;
  ~CodeCache() = default;

  template <typename GENFUNC>
  VALUE getOrCreate(const KEY& key, GENFUNC generatorFunction) {
    std::shared_lock<std::shared_timed_mutex> sharedLock(mutex_);

    // Check for existence of the key
    auto it = values_.find(key);
    if (it != values_.end()) {
      return it->second.get();
    } else {
      sharedLock.unlock();
      std::unique_lock<std::shared_timed_mutex> uniqueLock(mutex_);

      // Need to look up again because there could be race condition from
      // the time gap between sharedLock.unlock() and creating uniqueLock.
      it = values_.find(key);
      if (it == values_.end()) {
        std::promise<VALUE> returnPromise;
        values_[key] = returnPromise.get_future().share();

        uniqueLock.unlock();
        // The value (code) generation is not happening under a lock
        VALUE val = generatorFunction();
        returnPromise.set_value(val);
        return val;
      } else {
        return it->second.get();
      }
    }
  }
};

// This class must be used as a static variable.
template <typename KEY, typename VALUE>
class CodeCache<KEY, VALUE, /*THREAD_LOCAL=*/true> {
 private:
#ifdef FBCODE_CAFFE2
  static folly::F14FastMap<KEY, VALUE>& getValues_() {
    static thread_local folly::F14FastMap<KEY, VALUE>
        values_; /* library-local */
    return values_;
  }
#else
  static std::map<KEY, VALUE>& getValues_() {
    static thread_local std::map<KEY, VALUE> values_;
    return values_;
  }
#endif

 public:
  CodeCache(const CodeCache&) = delete;
  CodeCache& operator=(const CodeCache&) = delete;

  CodeCache(CodeCache&&) = default;
  CodeCache& operator=(CodeCache&&) = default;

  CodeCache() = default;
  ~CodeCache() = default;

  template <typename GENFUNC>
  VALUE getOrCreate(const KEY& key, GENFUNC generatorFunction) {
    // Check for existence of the key
    auto it = getValues_().find(key);
    if (it != getValues_().end()) {
      return it->second;
    } else {
      VALUE val = generatorFunction();
      getValues_()[key] = val;
      return val;
    }
  }
};

} // namespace fbgemm
