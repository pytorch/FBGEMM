/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <future>
#include <stdexcept>

#include <gtest/gtest.h>

#include "src/CodeCache.h" // @manual

using namespace fbgemm;

namespace {

struct GenFailure : std::runtime_error {
  using std::runtime_error::runtime_error;
};

} // namespace

// Kernel generators report failure by throwing -- GenerateI8Depthwise.cc does
// so when asmjit cannot register the generated code. The caller has to receive
// that exception, not a std::future_error from the cache's own bookkeeping.
TEST(CodeCacheTest, GeneratorExceptionPropagates) {
  CodeCache<int, int> cache;

  EXPECT_THROW(
      cache.getOrCreate(0, []() -> int { throw GenFailure("kernel gen"); }),
      GenFailure);
}

// A failed generation must not be cached. The cache publishes the future
// before running the generator, so an escaping exception used to leave a
// broken promise behind under that key.
TEST(CodeCacheTest, FailureIsNotCachedAndRetries) {
  CodeCache<int, int> cache;
  int calls = 0;

  EXPECT_THROW(
      cache.getOrCreate(
          0,
          [&]() -> int {
            ++calls;
            throw GenFailure("kernel gen");
          }),
      GenFailure);
  EXPECT_EQ(calls, 1);

  // Same key: the generator must run again rather than replay the failure.
  EXPECT_EQ(
      cache.getOrCreate(
          0,
          [&]() -> int {
            ++calls;
            return 42;
          }),
      42);
  EXPECT_EQ(calls, 2);
}

// Every repeat of a persistent failure keeps reporting the generator's own
// diagnostic, instead of degrading to broken_promise after the first attempt.
TEST(CodeCacheTest, RepeatedFailuresKeepTheDiagnostic) {
  CodeCache<int, int> cache;

  for (int attempt = 0; attempt < 3; ++attempt) {
    try {
      cache.getOrCreate(
          0, []() -> int { throw GenFailure("asmjit registration failed"); });
      FAIL() << "expected GenFailure on attempt " << attempt;
    } catch (const GenFailure& e) {
      EXPECT_STREQ(e.what(), "asmjit registration failed");
    }
  }
}

// The single-threaded cases above all pass on values_.erase() alone -- delete
// the set_exception() call and they stay green. What set_exception() is for is
// a *waiter*: the cache publishes the future before running the generator, so
// another thread can already be parked in get() when the generator throws.
// Without it that thread wakes on future_error(broken_promise) instead of the
// generator's own diagnostic, which is the bug this all exists to fix.
TEST(CodeCacheTest, WaiterNeverSeesBrokenPromise) {
  CodeCache<int, int> cache;
  std::promise<void> generatorEntered;
  std::promise<void> releaseGenerator;
  auto entered = generatorEntered.get_future();
  auto release = releaseGenerator.get_future();

  auto first = std::async(std::launch::async, [&]() -> int {
    return cache.getOrCreate(0, [&]() -> int {
      generatorEntered.set_value();
      release.wait();
      throw GenFailure("asmjit registration failed");
    });
  });

  // Only start the second caller once the generator is actually running, so it
  // finds the already-published future and parks on it.
  entered.wait();
  auto second = std::async(std::launch::async, [&]() -> int {
    return cache.getOrCreate(0, []() -> int { return 0; });
  });
  releaseGenerator.set_value();

  EXPECT_THROW(first.get(), GenFailure);

  // std::future_error must never surface. Which of the two legal outcomes we
  // get is a real race -- the waiter either parked before the failure and sees
  // the generator's exception, or arrived after the erase and re-ran
  // generation -- so accept both. A broken promise is neither, and escapes.
  try {
    EXPECT_EQ(second.get(), 0);
  } catch (const GenFailure&) {
    SUCCEED() << "waiter observed the generator's own exception";
  }
}

// The success path still memoizes: one generator call per key.
TEST(CodeCacheTest, SuccessIsCached) {
  CodeCache<int, int> cache;
  int calls = 0;
  auto gen = [&]() -> int {
    ++calls;
    return 7;
  };

  EXPECT_EQ(cache.getOrCreate(1, gen), 7);
  EXPECT_EQ(cache.getOrCreate(1, gen), 7);
  EXPECT_EQ(calls, 1);
}
